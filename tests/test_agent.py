import unittest
from collections import defaultdict, deque
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.answering import NOT_FOUND, render_answer
from app.errors import ServiceError
from app.graph import build_graph, get_graph
from app.main import app
from app.schemas import AnswerDraft, AskResponse, GradeDecision, RewriteDecision


def passage(chunk_id="chunk-a"):
    return {"id": chunk_id, "title": "Docker Compose", "heading": "Volumes",
            "url": "https://docs.docker.com/reference/compose-file/services/#volumes",
            "text": "Use services.web.volumes to mount data in the container."}


def answer_draft(chunk_id="chunk-a"):
    return {"supported": True, "statements": [{"kind": "text", "text": "Define the mount under the service's volumes field.", "citation_ids": [chunk_id]}]}


class FakeLLM:
    def __init__(self, **responses):
        self.responses = defaultdict(deque, {key: deque(value) for key, value in responses.items()})
        self.calls = []

    def generate(self, instruction, data, schema, *, task):
        self.calls.append((task, data))
        if not self.responses[task]:
            raise AssertionError(f"Unexpected LLM call: {task}")
        response = self.responses[task].popleft()
        if isinstance(response, Exception):
            raise response
        return schema.model_validate(response)


class TestGraph(unittest.TestCase):
    def run_graph(self, llm, retriever, question="How do I mount a volume?"):
        return build_graph(llm, retriever).invoke({"question": question}, config={"recursion_limit": 30})

    def test_greeting_skips_retrieval_and_llm(self):
        llm = FakeLLM()
        def search(query):
            self.fail("Greeting must not retrieve")
        state = self.run_graph(llm, search, "Hi!")
        self.assertEqual(state["result"]["status"], "direct")
        self.assertEqual(state["attempts"], 0)
        self.assertEqual(llm.calls, [])

    def test_supported_answer_uses_stored_source_and_passage(self):
        llm = FakeLLM(grade=[{"sufficient": True, "relevant_ids": ["chunk-a"], "reason": "Has instructions."}], answer=[answer_draft()])
        state = self.run_graph(llm, lambda query: [passage()])
        self.assertEqual(state["attempts"], 1)
        self.assertEqual(state["result"]["status"], "answered")
        self.assertIn("[1]", state["result"]["answer"])
        self.assertEqual(state["result"]["sources"][0]["url"], passage()["url"])
        self.assertEqual(state["result"]["sources"][0]["passage"], passage()["text"])

    def test_rewrite_keeps_original_question_and_uses_new_evidence(self):
        llm = FakeLLM(
            grade=[{"sufficient": False, "relevant_ids": [], "reason": "Missing mount syntax."},
                   {"sufficient": True, "relevant_ids": ["chunk-b"], "reason": "Mount syntax found."}],
            rewrite=[{"query": "Docker Compose service volumes mount syntax"}], answer=[answer_draft("chunk-b")])
        queries = []
        def search(query):
            queries.append(query)
            return [passage("chunk-a" if len(queries) == 1 else "chunk-b")]
        state = self.run_graph(llm, search)
        self.assertEqual(queries, ["How do I mount a volume?", "Docker Compose service volumes mount syntax"])
        self.assertEqual(state["attempts"], 2)
        self.assertEqual(state["rewrites"], 1)
        self.assertEqual(state["result"]["sources"][0]["chunk_id"], "chunk-b")
        self.assertTrue(all(data["original_question"] == "How do I mount a volume?" for _, data in llm.calls))

    def test_retry_exhaustion_stops_after_three_searches(self):
        llm = FakeLLM(grade=[{"sufficient": False, "relevant_ids": [], "reason": "Not covered."}] * 3,
                      rewrite=[{"query": "query two"}, {"query": "query three"}])
        state = self.run_graph(llm, lambda query: [passage()], "How do I configure Kubernetes RBAC?")
        self.assertEqual(state["attempts"], 3)
        self.assertEqual(state["rewrites"], 2)
        self.assertEqual(state["result"]["answer"], NOT_FOUND)
        self.assertEqual(state["result"]["sources"], [])
        self.assertNotIn("answer", [task for task, _ in llm.calls])

    def test_empty_search_skips_grading_but_still_has_bounded_rewrites(self):
        llm = FakeLLM(rewrite=[{"query": "query two"}, {"query": "query three"}])
        state = self.run_graph(llm, lambda query: [])
        self.assertEqual(state["attempts"], 3)
        self.assertEqual(state["result"]["status"], "not_found")
        self.assertEqual([task for task, _ in llm.calls], ["rewrite", "rewrite"])

    def test_quota_error_is_not_reported_as_missing_docs_or_retried(self):
        llm = FakeLLM(grade=[ServiceError("Gemini quota reached.", 429)])
        with self.assertRaises(ServiceError) as caught:
            self.run_graph(llm, lambda query: [passage()])
        self.assertEqual(caught.exception.status_code, 429)
        self.assertEqual(len(llm.calls), 1)

    def test_retrieval_failure_is_a_service_error(self):
        def failing_search(query):
            raise RuntimeError("Private implementation details")
        with self.assertRaises(ServiceError) as caught:
            self.run_graph(FakeLLM(), failing_search)
        self.assertEqual(caught.exception.status_code, 503)
        self.assertNotIn("Private", str(caught.exception))

    def test_grader_cannot_select_an_unknown_passage(self):
        llm = FakeLLM(grade=[{"sufficient": True, "relevant_ids": ["invented"], "reason": "Claims to answer."}])
        with self.assertRaises(ServiceError):
            self.run_graph(llm, lambda query: [passage()])

    def test_answerer_can_trigger_retry_when_grade_was_overconfident(self):
        llm = FakeLLM(grade=[{"sufficient": True, "relevant_ids": ["chunk-a"], "reason": "Looks useful."}] * 2,
                      answer=[{"supported": False, "statements": []}, answer_draft()],
                      rewrite=[{"query": "more precise query"}])
        state = self.run_graph(llm, lambda query: [passage()])
        self.assertEqual(state["attempts"], 2)
        self.assertEqual(state["result"]["status"], "answered")


class TestCitations(unittest.TestCase):
    def test_unknown_citations_are_rejected(self):
        with self.assertRaises(ServiceError):
            render_answer(AnswerDraft.model_validate(answer_draft("invented")), [passage()])

    def test_duplicate_citations_share_one_number_and_source(self):
        draft = answer_draft()
        draft["statements"] *= 2
        result = render_answer(AnswerDraft.model_validate(draft), [passage()])
        self.assertEqual(len(result.sources), 1)
        self.assertEqual(result.answer.count("[1]"), 2)

    def test_model_written_links_and_numeric_citations_are_rejected(self):
        for text in ("See [docs](https://invented.example)", "Use a mount. [99]"):
            with self.subTest(text=text), self.assertRaises(ServiceError):
                draft = answer_draft()
                draft["statements"][0]["text"] = text
                render_answer(AnswerDraft.model_validate(draft), [passage()])

    def test_code_fence_is_not_broken_by_appended_citation(self):
        draft = answer_draft()
        draft["statements"][0]["text"] = "```yaml\nvolumes:\n  - data:/data\n```"
        result = render_answer(AnswerDraft.model_validate(draft), [passage()])
        self.assertTrue(result.answer.endswith("```\n\n[1]"))

    def test_supported_answer_requires_statements(self):
        with self.assertRaises(ServiceError):
            render_answer(AnswerDraft(supported=True, statements=[]), [passage()])

    def test_raw_code_is_fenced_with_citations_outside_copyable_code(self):
        draft = answer_draft()
        draft["statements"][0].update(kind="code", language="yaml", text="volumes:\n  data:")
        result = render_answer(AnswerDraft.model_validate(draft), [passage()])
        self.assertEqual(result.answer, "```yaml\nvolumes:\n  data:\n```\n\n[1]")


class TestAPI(unittest.TestCase):
    def setUp(self):
        app.dependency_overrides[get_graph] = lambda: object()
        self.client = TestClient(app)

    def tearDown(self):
        app.dependency_overrides.clear()
        self.client.close()

    def test_health_and_question_response(self):
        self.assertEqual(self.client.get("/health").status_code, 200)
        result = AskResponse(answer="Hi!", status="direct")
        with patch("app.main.run_question", return_value=result) as run:
            response = self.client.post("/ask", json={"question": "  hi  "})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "direct")
        self.assertEqual(run.call_args.args[0], "hi")

    def test_blank_oversized_wrong_type_and_extra_fields_are_rejected(self):
        for body in ({"question": "  "}, {"question": "a" * 2001}, {"question": True},
                     {"question": "hi", "unexpected": "value"}, {}):
            with self.subTest(body=str(body)[:80]):
                self.assertEqual(self.client.post("/ask", json=body).status_code, 422)

    def test_provider_errors_have_distinct_http_status(self):
        for status in (429, 502, 503, 504):
            with self.subTest(status=status), patch("app.main.run_question", side_effect=ServiceError("Safe service message.", status)):
                response = self.client.post("/ask", json={"question": "volume syntax?"})
                self.assertEqual(response.status_code, status)
                self.assertEqual(response.json(), {"detail": "Safe service message."})
