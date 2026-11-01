import io
import json
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.answering import render_answer
from app.errors import ServiceError
from app.graph import build_graph, get_graph, stream_question
from app.llm import GeminiClient
from app.main import app
from app.schemas import AnswerDraft
from app.streaming import complete_statements, sse, threaded_sse
from app.verify_access import request_json_stream

PASSAGE = {"id": "chunk-a", "title": "Volumes", "heading": "Compose",
           "url": "https://docs.docker.com/storage/volumes/", "text": "Use volumes."}
STATEMENT = {"kind": "text", "text": "Use a named volume.", "citation_ids": ["chunk-a"]}
DRAFT = {"supported": True, "statements": [STATEMENT, {**STATEMENT, "kind": "code", "language": "yaml", "text": "volumes:\n  data:"}]}


class FakeLLM:
    def __init__(self, unsupported=False, invalid=False):
        self.unsupported = unsupported
        self.invalid = invalid
        self.answer_calls = 0

    def generate(self, instruction, data, schema, *, task, on_partial=None):
        if task == "grade":
            return schema.model_validate({"sufficient": not self.unsupported,
                "relevant_ids": [] if self.unsupported else ["chunk-a"], "reason": "Evidence checked."})
        if task == "rewrite":
            return schema.model_validate({"query": "refined query"})
        self.answer_calls += 1
        draft = json.loads(json.dumps(DRAFT))
        if self.invalid:
            draft["statements"][0]["citation_ids"] = ["invented"]
        text = json.dumps(draft)
        if on_partial:
            for index in range(1, len(text) + 1):
                on_partial(text[:index])
        return schema.model_validate(draft)


class TestStreaming(unittest.TestCase):
    def setUp(self):
        self.trace = patch('app.graph.observation', side_effect=lambda *args, **kwargs: nullcontext(None))
        self.trace.start()
        self.addCleanup(self.trace.stop)

    def test_only_closed_statements_are_extracted_even_with_escaped_braces(self):
        statement = {**STATEMENT, "text": 'Use {data} and "quoted" text.'}
        prefix = '{"supported":true,"statements":[' + json.dumps(statement)
        self.assertEqual(complete_statements(prefix[:-1]), [])
        self.assertEqual(complete_statements(prefix)[0].text, statement['text'])
        self.assertEqual(len(complete_statements(prefix + ',{"kind":"code","text":"partial')), 1)
        # Property order does not matter; fragments with no array return nothing.
        self.assertEqual(len(complete_statements('{"statements":[' + json.dumps(statement))), 1)
        self.assertEqual(complete_statements('{"supported":tr'), [])

    def test_progress_previews_and_final_answer_preserve_citations(self):
        events = list(stream_question('How do I mount data?', build_graph(FakeLLM(), lambda q: [PASSAGE])))
        steps = [event['data']['step'] for event in events if event['event'] == 'status']
        self.assertEqual(steps, ['route', 'retrieve', 'grade', 'answer'])
        previews = [event['data'] for event in events if event['event'] == 'answer']
        self.assertEqual(len(previews), 2)
        self.assertNotIn('```yaml', previews[0]['answer'])
        self.assertIn('```yaml', previews[1]['answer'])
        final = next(event['data'] for event in events if event['event'] == 'result')
        self.assertEqual(final['answer'], render_answer(AnswerDraft.model_validate(DRAFT), [PASSAGE]).answer)
        self.assertEqual(final['retrieval_attempts'], 1)
        self.assertEqual(final['sources'][0]['url'], PASSAGE['url'])
        self.assertEqual(events[-1]['event'], 'done')

    def test_greeting_and_retry_exhaustion_have_terminal_results(self):
        greeting = list(stream_question('hi', build_graph(FakeLLM(), lambda q: self.fail('must not retrieve'))))
        self.assertEqual(greeting[-2]['data']['status'], 'direct')
        llm = FakeLLM(unsupported=True)
        events = list(stream_question('Uncovered topic?', build_graph(llm, lambda q: [PASSAGE])))
        result = next(event['data'] for event in events if event['event'] == 'result')
        self.assertEqual((result['status'], result['retrieval_attempts'], result['rewrite_count']), ('not_found',3,2))
        self.assertEqual(llm.answer_calls, 0)
        self.assertEqual(sum(e['event']=='status' and e['data']['step']=='rewrite' for e in events), 2)

    def test_unknown_citation_is_never_sent_in_a_preview(self):
        events = list(stream_question('How?', build_graph(FakeLLM(invalid=True), lambda q:[PASSAGE])))
        self.assertNotIn('answer', [e['event'] for e in events])
        self.assertEqual(events[-1]['event'], 'error')
        self.assertNotIn('result', [e['event'] for e in events])

    def test_stream_errors_are_distinct_from_not_found(self):
        def fail(q):
            raise ServiceError('Document search unavailable.', 503)
        events=list(stream_question('How?',build_graph(FakeLLM(),fail)))
        self.assertEqual(events[-1]['data']['status_code'],503)
        self.assertNotIn('result',[e['event'] for e in events])

    def test_api_stream_framing_and_validation(self):
        app.dependency_overrides[get_graph]=lambda:build_graph(FakeLLM(),lambda q:[PASSAGE])
        try:
            with TestClient(app) as client:
                response=client.post('/ask/stream',json={'question':'How?'})
                self.assertEqual(response.status_code,200)
                self.assertIn('text/event-stream',response.headers['content-type'])
                frames=response.text.strip().split('\n\n')
                self.assertTrue(any(frame.startswith('event: answer\n') for frame in frames))
                self.assertTrue(frames[-1].startswith('event: done\n'))
                self.assertEqual(client.post('/ask/stream',json={'question':' '}).status_code,422)
        finally:
            app.dependency_overrides.clear()
        self.assertEqual(len(sse('status',{'message':'a\nb'}).strip().splitlines()),2)

    def test_preview_is_not_a_success_when_generation_later_fails(self):
        class FailingLLM(FakeLLM):
            def generate(self, instruction, data, schema, *, task, on_partial=None):
                if task == "answer":
                    on_partial(json.dumps(DRAFT))
                    raise ServiceError("Incomplete model output.", 502)
                return super().generate(instruction, data, schema, task=task)
        events = list(stream_question('How?', build_graph(FailingLLM(), lambda q:[PASSAGE])))
        self.assertIn('answer', [e['event'] for e in events])
        self.assertEqual(events[-1]['event'], 'error')
        self.assertNotIn('result', [e['event'] for e in events])

    def test_worker_keeps_trace_context_and_closes_on_disconnect(self):
        from contextvars import ContextVar
        from threading import Event
        marker = ContextVar('stream-test-marker', default=None)
        closed = Event()
        def events():
            token = marker.set('worker-context')
            try:
                while True:
                    self.assertEqual(marker.get(), 'worker-context')
                    yield {'event':'status','data':{'message':'working'}}
            finally:
                marker.reset(token)
                closed.set()
        stream = threaded_sse(events)
        self.assertIn('event: status', next(stream))
        stream.close()
        self.assertTrue(closed.wait(2), 'worker should stop and close its event generator')
        self.assertIsNone(marker.get())

    def test_provider_sse_packets_are_decoded_without_waiting_for_eof(self):
        response=io.BytesIO(b': comment\r\ndata: {"candidates": []}\r\n\r\ndata: {"usageMetadata": {}}\n\n')
        with patch('app.verify_access.build_opener') as opener:
            opener.return_value.open.return_value=response
            packets=request_json_stream('https://provider.example',{}, {})
            self.assertEqual(next(packets),{'candidates':[]})
            self.assertFalse(response.closed)
            self.assertEqual(list(packets),[{'usageMetadata':{}}])
        self.assertTrue(response.closed)


class TestGeminiStreaming(unittest.TestCase):
    def setUp(self):
        for patcher in [patch('app.llm.get_settings',return_value=SimpleNamespace(
                gemini_api_key=SecretStr('private-key'),gemini_model='gemini-test')),
                patch('app.llm.observation',side_effect=lambda *a,**k:nullcontext(None))]:
            patcher.start();self.addCleanup(patcher.stop)

    def generate(self,callback):
        return GeminiClient().generate('instruction',{},AnswerDraft,task='answer',on_partial=callback)

    def packet(self,text,finish=None):
        candidate={'content':{'parts':[{'text':text}]}}
        if finish: candidate['finishReason']=finish
        return {'candidates':[candidate]}

    def test_real_provider_deltas_and_final_schema_are_used(self):
        text=json.dumps(DRAFT)
        packets=[self.packet(text[:100]),self.packet(text[100:],'STOP'),{'usageMetadata':{'promptTokenCount':10,'candidatesTokenCount':20}}]
        previews=[]
        with patch('app.llm.request_json_stream',return_value=iter(packets)) as request:
            result=self.generate(previews.append)
        self.assertEqual(result,AnswerDraft.model_validate(DRAFT))
        self.assertEqual(previews,[text[:100],text])
        self.assertIn(':streamGenerateContent?alt=sse',request.call_args.args[0])
        self.assertNotIn('private-key',request.call_args.args[0])

    def test_truncated_output_fails_even_after_a_valid_preview(self):
        text=json.dumps(DRAFT)
        with patch('app.llm.request_json_stream',return_value=iter([self.packet(text,'MAX_TOKENS')])):
            with self.assertRaises(ServiceError):self.generate(lambda text:None)

    def test_midstream_disconnect_is_not_retried(self):
        def packets(*args):
            yield self.packet('{"supported":')
            raise URLError('private provider failure')
        with patch('app.llm.request_json_stream',side_effect=packets) as request:
            with self.assertRaises(ServiceError) as error:self.generate(lambda text:None)
        self.assertEqual(request.call_count,1)
        self.assertEqual(error.exception.status_code,504)
        self.assertNotIn('private',str(error.exception))

    def test_transient_failure_before_first_packet_is_retried_once(self):
        error=HTTPError('https://provider.example',503,'private',{},None)
        with patch('app.llm.request_json_stream',side_effect=[error,iter([self.packet(json.dumps(DRAFT),'STOP')])]) as request, patch('app.llm.time.sleep'):
            self.generate(lambda text:None)
        self.assertEqual(request.call_count,2)

    def test_quota_error_is_sent_safely(self):
        error=HTTPError('https://provider.example',429,'private-key',{},None)
        with patch('app.llm.request_json_stream',side_effect=error) as request:
            with self.assertRaises(ServiceError) as caught:self.generate(lambda text:None)
        self.assertEqual(caught.exception.status_code,429)
        self.assertNotIn('private-key',str(caught.exception))
        self.assertEqual(request.call_count,1)
