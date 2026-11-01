"""Corrective RAG: route, retrieve, grade, rewrite at most twice, answer."""

import logging
import re
from contextlib import nullcontext
from functools import lru_cache
from typing import TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph
from langgraph.config import get_stream_writer

from app.answering import NOT_FOUND, render_answer
from app.errors import ServiceError
from app.llm import GeminiClient
from app.prompts import ANSWER_PROMPT, GRADE_PROMPT, REWRITE_PROMPT, CONTEXT_PROMPT
from app.schemas import AnswerDraft, AskResponse, GradeDecision, RewriteDecision, StandaloneQuestion, ThreadHistory
from app.memory import get_checkpointer, thread_guard, thread_is_busy
from app.telemetry import current_trace_id, observation, update_observation
from app.streaming import complete_statements

MAX_REWRITES = 2
MAX_SAVED_TURNS = 20
CONTEXT_TURNS = 6


class AgentState(TypedDict, total=False):
    question: str
    resolved_question: str
    thread_id: str
    history: list[dict]
    streaming: bool
    query: str
    queries: list[str]
    direct: bool
    passages: list[dict]
    selected: list[dict]
    sufficient: bool
    reason: str
    attempts: int
    rewrites: int
    steps: list[str]
    result: dict


def evidence(passages):
    return [{key: passage[key] for key in ("id", "title", "heading", "text")} for passage in passages]


def build_graph(llm, retriever, *, trace_nodes: bool = False, checkpointer=None):
    def route(state):
        greeting = re.fullmatch(r"(?:hi|hello|hey|thanks|thank you|good morning|good evening)[\s!.?]*", state["question"].strip(), re.I)
        # Reset per-question fields; only completed conversation turns carry over.
        return {"direct": bool(greeting), "query": state["question"], "queries": [],
                "resolved_question": state["question"], "passages": [], "selected": [],
                "sufficient": False, "reason": "", "result": {},
                "attempts": 0, "rewrites": 0, "steps": ["route"]}

    def context(state):
        recent = [{"question": turn["question"], "resolved_question": turn["resolved_question"],
                   "answer": turn["response"]["answer"][:2000]}
                  for turn in state.get("history", [])[-CONTEXT_TURNS:]]
        decision = llm.generate(CONTEXT_PROMPT,
            {"question": state["question"], "recent_turns": recent}, StandaloneQuestion, task="context")
        resolved = decision.question.strip()
        if not resolved:
            raise ServiceError("Gemini could not resolve this follow-up question.")
        return {"resolved_question": resolved, "query": resolved, "steps": state["steps"] + ["context"]}

    def direct(state):
        return {"result": AskResponse(answer="Hi! Ask me about the indexed Docker or FastAPI documentation.", status="direct").model_dump(),
                "steps": state["steps"] + ["direct"]}

    def search(state):
        try:
            passages = retriever(state["query"])
        except ServiceError:
            raise
        except Exception:
            raise ServiceError("Document search is unavailable. Check Qdrant and run ingestion if needed.", 503) from None
        return {"passages": passages, "attempts": state["attempts"] + 1,
                "queries": state["queries"] + [state["query"]], "steps": state["steps"] + ["retrieve"]}

    def grade(state):
        if not state["passages"]:
            return {"sufficient": False, "selected": [], "reason": "Search returned no passages.",
                    "steps": state["steps"] + ["grade"]}
        decision = llm.generate(GRADE_PROMPT,
            {"original_question": state["resolved_question"], "passages": evidence(state["passages"])},
            GradeDecision, task="grade")
        known = {passage["id"]: passage for passage in state["passages"]}
        if any(chunk_id not in known for chunk_id in decision.relevant_ids) or (decision.sufficient and not decision.relevant_ids):
            raise ServiceError("Gemini's evidence assessment referenced invalid passages.")
        return {"sufficient": decision.sufficient, "selected": [known[key] for key in dict.fromkeys(decision.relevant_ids)],
                "reason": decision.reason, "steps": state["steps"] + ["grade"]}

    def rewrite(state):
        decision = llm.generate(REWRITE_PROMPT,
            {"original_question": state["resolved_question"], "previous_queries": state["queries"], "grading_reason": state["reason"]},
            RewriteDecision, task="rewrite")
        query = decision.query.strip()
        if not query:
            raise ServiceError("Gemini returned an empty rewritten query.")
        return {"query": query, "rewrites": state["rewrites"] + 1, "steps": state["steps"] + ["rewrite"]}

    def answer(state):
        options = {}
        if state.get("streaming"):
            writer = get_stream_writer()
            previous_count = 0

            def preview(text):
                nonlocal previous_count
                statements = complete_statements(text)
                if len(statements) <= previous_count:
                    return
                # Reuse the same citation and formatting checks as the final answer.
                result = render_answer(AnswerDraft(supported=True, statements=statements), state["selected"])
                writer({"event": "answer", "data": {
                    **result.model_dump(), "retrieval_attempts": state["attempts"],
                    "rewrite_count": state["rewrites"],
                }})
                previous_count = len(statements)

            options["on_partial"] = preview
        draft = llm.generate(ANSWER_PROMPT,
            {"original_question": state["resolved_question"], "passages": evidence(state["selected"])},
            AnswerDraft, task="answer", **options)
        result = render_answer(draft, state["selected"])
        return {"result": result.model_dump(), "reason": "Answer generation found insufficient supporting evidence.",
                "steps": state["steps"] + ["answer"]}

    def fallback(state):
        return {"result": AskResponse(answer=NOT_FOUND, status="not_found").model_dump(),
                "steps": state["steps"] + ["fallback"]}

    def finish(state):
        response = {**state["result"], "retrieval_attempts": state["attempts"],
                    "rewrite_count": state["rewrites"], "trace_id": current_trace_id(),
                    "thread_id": state.get("thread_id"), "resolved_question": state["resolved_question"]}
        turn = {"question": state["question"], "resolved_question": state["resolved_question"], "response": response}
        return {"result": response, "history": (state.get("history", []) + [turn])[-MAX_SAVED_TURNS:]}

    def after_grade(state):
        if state["sufficient"]:
            return "answer"
        return "rewrite" if state["rewrites"] < MAX_REWRITES else "fallback"

    def after_answer(state):
        if state["result"]["status"] == "answered":
            return "finish"
        return "rewrite" if state["rewrites"] < MAX_REWRITES else "fallback"

    def traced(name, function):
        def call(state):
            if state.get("streaming") and name != "finish":
                writer = get_stream_writer()
                if name in {"rewrite", "answer", "fallback"}:
                    writer({"event": "reset", "data": {}})
                messages = {
                    "route": "Understanding your question…",
                    "direct": "Preparing a reply…",
                    "context": "Using the conversation to understand your follow-up…",
                    "retrieve": "Searching the documentation…",
                    "grade": "Checking whether the passages answer your question…",
                    "rewrite": "Refining the search and trying again…",
                    "answer": "Writing an answer with citations…",
                    "fallback": "No supporting answer found in the indexed docs.",
                }
                writer({"event": "status", "data": {
                    "step": name, "message": messages[name],
                    "retrieval_attempts": state.get("attempts", 0) + (name == "retrieve"),
                    "rewrite_count": state.get("rewrites", 0) + (name == "rewrite"),
                }})
            context = observation(name, input={
                "question": state["question"], "query": state.get("query"),
                "retrieval_attempts": state.get("attempts", 0),
                "rewrite_count": state.get("rewrites", 0),
            }) if trace_nodes else nullcontext(None)
            with context as span:
                update = function(state)
                update_observation(span, output=update)
                return update
        return call

    graph = StateGraph(AgentState)
    for name, function in {"route": route, "direct": direct, "retrieve": search,
                           "grade": grade, "rewrite": rewrite, "answer": answer,
                           "fallback": fallback, "context": context, "finish": finish}.items():
        graph.add_node(name, traced(name, function))
    graph.add_edge(START, "route")
    graph.add_conditional_edges("route", lambda state: "direct" if state["direct"] else ("context" if state.get("history") else "retrieve"), ["direct", "context", "retrieve"])
    graph.add_edge("context", "retrieve")
    graph.add_edge("direct", "finish")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", after_grade, ["answer", "rewrite", "fallback"])
    graph.add_edge("rewrite", "retrieve")
    graph.add_conditional_edges("answer", after_answer, ["finish", "rewrite", "fallback"])
    graph.add_edge("fallback", "finish")
    graph.add_edge("finish", END)
    return graph.compile(checkpointer=checkpointer)


@lru_cache
def get_graph():
    from app.retrieval import retrieve

    return build_graph(GeminiClient(), retrieve, trace_nodes=True, checkpointer=get_checkpointer())


def thread_config(thread_id):
    return {"recursion_limit": 30, "configurable": {"thread_id": thread_id}}


def read_thread(thread_id: str, graph) -> ThreadHistory:
    state = graph.get_state(thread_config(thread_id)).values
    return ThreadHistory(thread_id=thread_id, turns=state.get("history", []))


def run_question(question: str, graph, *, thread_id: str | None = None) -> AskResponse:
    thread_id = thread_id or str(uuid4())
    with thread_guard(thread_id), observation("docspilot.ask", input={"question": question, "thread_id": thread_id}) as span:
        state = graph.invoke({"question": question, "streaming": False, "thread_id": thread_id}, config=thread_config(thread_id))
        result = AskResponse.model_validate(state["result"])
        update_observation(span, output=result.model_dump(), metadata={"steps": state["steps"], "thread_id": thread_id})
        return result


def stream_question(question: str, graph, *, thread_id: str | None = None):
    """Expose only public progress/previews, then a fully validated final result."""
    thread_id = thread_id or str(uuid4())
    try:
        yield {"event": "thread", "data": {"thread_id": thread_id}}
        if thread_is_busy(thread_id):
            yield {"event": "status", "data": {
                "step": "waiting", "message": "Waiting for this chat’s previous request to finish…",
            }}
        with thread_guard(thread_id, wait=True), observation("docspilot.ask", input={"question": question, "thread_id": thread_id, "streaming": True}) as span:
            state = None
            for mode, chunk in graph.stream(
                {"question": question, "streaming": True, "thread_id": thread_id},
                config=thread_config(thread_id), stream_mode=["custom", "values"],
            ):
                if mode == "custom":
                    yield chunk
                else:
                    state = chunk
            result = AskResponse.model_validate(state["result"])
            update_observation(span, output=result.model_dump(), metadata={"steps": state["steps"], "thread_id": thread_id, "streaming": True})
        # The browser may stop reading as soon as it sees the terminal event.
        # Release the thread guard and tracing context before exposing completion.
        yield {"event": "result", "data": result.model_dump()}
        yield {"event": "done", "data": {}}
    except ServiceError as error:
        yield {"event": "error", "data": {"detail": str(error), "status_code": error.status_code}}
    except Exception:
        logging.getLogger(__name__).error("Streaming request failed unexpectedly.")
        yield {"event": "error", "data": {"detail": "The answer stream failed. Please try again.", "status_code": 500}}
