"""Corrective RAG: route, retrieve, grade, rewrite at most twice, answer."""

import re
from contextlib import nullcontext
from functools import lru_cache
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from app.answering import NOT_FOUND, render_answer
from app.errors import ServiceError
from app.llm import GeminiClient
from app.prompts import ANSWER_PROMPT, GRADE_PROMPT, REWRITE_PROMPT
from app.schemas import AnswerDraft, AskResponse, GradeDecision, RewriteDecision
from app.telemetry import current_trace_id, observation, update_observation

MAX_REWRITES = 2


class AgentState(TypedDict, total=False):
    question: str
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


def build_graph(llm, retriever, *, trace_nodes: bool = False):
    def route(state):
        greeting = re.fullmatch(r"(?:hi|hello|hey|thanks|thank you|good morning|good evening)[\s!.?]*", state["question"].strip(), re.I)
        return {"direct": bool(greeting), "query": state["question"], "queries": [],
                "attempts": 0, "rewrites": 0, "steps": ["route"]}

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
            {"original_question": state["question"], "passages": evidence(state["passages"])},
            GradeDecision, task="grade")
        known = {passage["id"]: passage for passage in state["passages"]}
        if any(chunk_id not in known for chunk_id in decision.relevant_ids) or (decision.sufficient and not decision.relevant_ids):
            raise ServiceError("Gemini's evidence assessment referenced invalid passages.")
        return {"sufficient": decision.sufficient, "selected": [known[key] for key in dict.fromkeys(decision.relevant_ids)],
                "reason": decision.reason, "steps": state["steps"] + ["grade"]}

    def rewrite(state):
        decision = llm.generate(REWRITE_PROMPT,
            {"original_question": state["question"], "previous_queries": state["queries"], "grading_reason": state["reason"]},
            RewriteDecision, task="rewrite")
        query = decision.query.strip()
        if not query:
            raise ServiceError("Gemini returned an empty rewritten query.")
        return {"query": query, "rewrites": state["rewrites"] + 1, "steps": state["steps"] + ["rewrite"]}

    def answer(state):
        draft = llm.generate(ANSWER_PROMPT,
            {"original_question": state["question"], "passages": evidence(state["selected"])},
            AnswerDraft, task="answer")
        result = render_answer(draft, state["selected"])
        return {"result": result.model_dump(), "reason": "Answer generation found insufficient supporting evidence.",
                "steps": state["steps"] + ["answer"]}

    def fallback(state):
        return {"result": AskResponse(answer=NOT_FOUND, status="not_found").model_dump(),
                "steps": state["steps"] + ["fallback"]}

    def after_grade(state):
        if state["sufficient"]:
            return "answer"
        return "rewrite" if state["rewrites"] < MAX_REWRITES else "fallback"

    def after_answer(state):
        if state["result"]["status"] == "answered":
            return END
        return "rewrite" if state["rewrites"] < MAX_REWRITES else "fallback"

    def traced(name, function):
        def call(state):
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
                           "fallback": fallback}.items():
        graph.add_node(name, traced(name, function))
    graph.add_edge(START, "route")
    graph.add_conditional_edges("route", lambda state: "direct" if state["direct"] else "retrieve", ["direct", "retrieve"])
    graph.add_edge("direct", END)
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", after_grade, ["answer", "rewrite", "fallback"])
    graph.add_edge("rewrite", "retrieve")
    graph.add_conditional_edges("answer", after_answer, [END, "rewrite", "fallback"])
    graph.add_edge("fallback", END)
    return graph.compile()


@lru_cache
def get_graph():
    from app.retrieval import retrieve

    return build_graph(GeminiClient(), retrieve, trace_nodes=True)


def run_question(question: str, graph) -> AskResponse:
    with observation("docspilot.ask", input={"question": question}) as span:
        state = graph.invoke({"question": question}, config={"recursion_limit": 30})
        result = AskResponse.model_validate({
            **state["result"], "retrieval_attempts": state["attempts"],
            "rewrite_count": state["rewrites"], "trace_id": current_trace_id(),
        })
        update_observation(span, output=result.model_dump(), metadata={"steps": state["steps"]})
        return result
