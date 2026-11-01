import json
import sqlite3
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from threading import Event, Thread
from unittest.mock import patch

from fastapi.testclient import TestClient
from langgraph.checkpoint.sqlite import SqliteSaver

from app.errors import ServiceError
from app.graph import build_graph, get_graph, read_thread, run_question, stream_question, thread_config
from app.main import app
from app.memory import thread_guard
from app.schemas import AskRequest


def passage(text):
    return {'id':'chunk','title':'Volumes','heading':'Compose','url':'https://docs.docker.com/storage/volumes/','text':text}


class MemoryLLM:
    def __init__(self, resolved='How do I delete a Docker named volume?'):
        self.calls=[]
        self.resolved=resolved
        self.fail=False

    def generate(self, instruction, data, schema, *, task, on_partial=None):
        self.calls.append((task,data))
        if self.fail:
            raise ServiceError('Model temporarily unavailable.',502)
        if task=='context':return schema.model_validate({'question':self.resolved})
        if task=='grade':return schema.model_validate({'sufficient':True,'relevant_ids':['chunk'],'reason':'Instructions found.'})
        if task=='answer':
            draft={'supported':True,'statements':[{'kind':'text','text':data['passages'][0]['text'],'citation_ids':['chunk']}]}
            if on_partial:on_partial(json.dumps(draft))
            return schema.model_validate(draft)
        raise AssertionError(task)


class TestConversationMemory(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.db=Path(self.directory.name)/'checkpoints.sqlite3'
        self.connections=[]
        self.addCleanup(lambda:[connection.close() for connection in self.connections])
        for patcher in [patch('app.graph.observation',side_effect=lambda *a,**k:nullcontext(None)),
                        patch('app.graph.current_trace_id',return_value=None)]:
            patcher.start();self.addCleanup(patcher.stop)

    def graph(self,llm=None,retriever=None):
        connection=sqlite3.connect(self.db,check_same_thread=False)
        self.connections.append(connection)
        return build_graph(llm or MemoryLLM(),retriever or (lambda q:[passage('Use Docker named volumes.')]),checkpointer=SqliteSaver(connection))

    def test_followup_resolves_using_saved_context_but_answers_from_new_docs(self):
        llm=MemoryLLM();queries=[]
        def retrieve(query):
            queries.append(query)
            return [passage('Delete a named volume with docker volume rm.' if 'delete' in query else 'Create a named Docker volume.')]
        graph=self.graph(llm,retrieve)
        first=run_question('How do I create a named Docker volume?',graph,thread_id='docker-chat')
        second=run_question('How do I delete it?',graph,thread_id='docker-chat')
        self.assertEqual(queries,['How do I create a named Docker volume?','How do I delete a Docker named volume?'])
        self.assertEqual(second.resolved_question,queries[-1])
        self.assertEqual(second.thread_id,'docker-chat')
        self.assertEqual(second.retrieval_attempts,1)
        self.assertEqual(second.rewrite_count,0)
        self.assertIn('docker volume rm',second.answer)
        context=next(data for task,data in llm.calls if task=='context')
        self.assertEqual(context['recent_turns'][0]['answer'],first.answer)
        for task,data in llm.calls:
            if task in {'grade','answer'}:self.assertNotIn('recent_turns',data)
        turns=read_thread('docker-chat',graph).turns
        self.assertEqual(len(turns),2)
        self.assertEqual(turns[1].question,'How do I delete it?')
        self.assertEqual(turns[1].response.sources[0].passage,'Delete a named volume with docker volume rm.')

    def test_sqlite_survives_closing_and_reopening_the_graph(self):
        graph=self.graph()
        run_question('How do I create a Docker volume?',graph,thread_id='persistent-chat')
        self.connections.pop().close()
        llm=MemoryLLM();restored=self.graph(llm)
        self.assertEqual(len(read_thread('persistent-chat',restored).turns),1)
        run_question('How do I delete it?',restored,thread_id='persistent-chat')
        self.assertEqual(llm.calls[0][0],'context')
        self.assertEqual(len(read_thread('persistent-chat',restored).turns),2)

    def test_new_threads_are_isolated_and_missing_id_creates_a_new_one(self):
        llm=MemoryLLM();graph=self.graph(llm)
        first=run_question('How do I create a volume?',graph,thread_id='one')
        second=run_question('How do I delete it?',graph,thread_id='two')
        third=run_question('hi',graph)
        fourth=run_question('hi',graph)
        self.assertNotIn('context',[task for task,_ in llm.calls])
        self.assertNotEqual(third.thread_id,fourth.thread_id)
        self.assertEqual(len(read_thread('one',graph).turns),1)
        self.assertEqual(read_thread('unknown',graph).turns,[])

    def test_failed_turn_does_not_pollute_history_or_reuse_old_result(self):
        llm=MemoryLLM();graph=self.graph(llm)
        run_question('How do I create a volume?',graph,thread_id='chat')
        llm.fail=True
        with self.assertRaises(ServiceError):run_question('How do I delete it?',graph,thread_id='chat')
        self.assertEqual(len(read_thread('chat',graph).turns),1)
        state=graph.get_state(thread_config('chat')).values
        self.assertEqual(state['result'],{})
        self.assertEqual(state['selected'],[])
        llm.fail=False
        result=run_question('How do I delete it?',graph,thread_id='chat')
        self.assertEqual(result.retrieval_attempts,1)
        self.assertEqual(len(read_thread('chat',graph).turns),2)

    def test_json_and_streaming_requests_share_the_same_thread(self):
        llm=MemoryLLM();graph=self.graph(llm)
        events=list(stream_question('How do I create a volume?',graph,thread_id='shared'))
        self.assertEqual(events[0],{'event':'thread','data':{'thread_id':'shared'}})
        self.assertEqual(events[-2]['data']['thread_id'],'shared')
        result=run_question('How do I delete it?',graph,thread_id='shared')
        self.assertEqual(result.status,'answered')
        self.assertFalse(graph.get_state(thread_config('shared')).values['streaming'])
        self.assertEqual(len(read_thread('shared',graph).turns),2)

    def test_history_and_prompt_context_are_bounded(self):
        llm=MemoryLLM();graph=self.graph(llm)
        for _ in range(22):run_question('How?',graph,thread_id='long-chat')
        self.assertEqual(len(read_thread('long-chat',graph).turns),20)
        contexts=[data for task,data in llm.calls if task=='context']
        self.assertEqual(len(contexts[-1]['recent_turns']),6)

    def test_api_thread_validation_and_history(self):
        graph=self.graph()
        app.dependency_overrides[get_graph]=lambda:graph
        try:
            with TestClient(app) as client:
                for value in ['', '../other', 'x'*129, 42, True]:
                    self.assertEqual(client.post('/ask',json={'question':'hi','thread_id':value}).status_code,422)
                result=client.post('/ask',json={'question':'hi','thread_id':'api-chat'})
                self.assertEqual(result.status_code,200)
                self.assertEqual(result.json()['thread_id'],'api-chat')
                history=client.get('/threads/api-chat').json()
                self.assertEqual(history['thread_id'],'api-chat')
                self.assertEqual(history['turns'][0]['question'],'hi')
                self.assertEqual(client.get('/threads/fresh-chat').json()['turns'],[])
        finally:app.dependency_overrides.clear()

    def test_overlapping_requests_are_rejected_only_for_the_same_thread(self):
        with thread_guard('one'):
            with self.assertRaises(ServiceError) as error:
                run_question('hi',self.graph(),thread_id='one')
            self.assertEqual(error.exception.status_code,409)
            self.assertEqual(run_question('hi',self.graph(),thread_id='two').status,'direct')
        with thread_guard('one'):pass


    def test_next_question_can_start_without_exhausting_the_previous_stream(self):
        graph=self.graph()
        for terminal in ['result','done']:
            thread_id=f'terminal-{terminal}'
            events=stream_question('hi',graph,thread_id=thread_id)
            try:
                while next(events)['event']!=terminal:
                    pass
                # Real browsers stop reading at done instead of exhausting a
                # Python generator, so completion must already release the lock.
                result=run_question('hi',graph,thread_id=thread_id)
                self.assertEqual(result.status,'direct')
                self.assertEqual(len(read_thread(thread_id,graph).turns),2)
            finally:
                events.close()


    def test_streaming_followup_waits_and_uses_the_completed_previous_turn(self):
        from queue import Queue
        from threading import Event, Thread
        llm=MemoryLLM()
        graph=self.graph(llm)
        first=stream_question('How do I create a Docker volume?',graph,thread_id='queued')
        output=Queue()
        finished=Event()
        worker=None
        try:
            # Hold the first stream after it enters the graph, before completion.
            while True:
                event=next(first)
                if event['event']=='status':break
            def followup():
                try:
                    for event in stream_question('How do I delete it?',graph,thread_id='queued'):
                        output.put(event)
                finally:finished.set()
            worker=Thread(target=followup,daemon=True)
            worker.start()
            self.assertEqual(output.get(timeout=2)['event'],'thread')
            waiting=output.get(timeout=2)
            self.assertEqual(waiting['data']['step'],'waiting')
            self.assertFalse(finished.is_set())
            list(first)
            self.assertTrue(finished.wait(5))
            events=[]
            while not output.empty():events.append(output.get_nowait())
            self.assertNotIn('error',[event['event'] for event in events])
            result=next(event['data'] for event in events if event['event']=='result')
            self.assertIn('delete a Docker named volume',result['resolved_question'])
            self.assertEqual(len(read_thread('queued',graph).turns),2)
        finally:
            first.close()
            if worker:worker.join(timeout=5)

    def test_timed_out_waiter_does_not_remove_the_owners_lock(self):
        with thread_guard('busy-owner'):
            with self.assertRaises(ServiceError):
                with thread_guard('busy-owner',wait=True,timeout=0.01):pass
            with self.assertRaises(ServiceError):
                with thread_guard('busy-owner'):pass
        with thread_guard('busy-owner'):pass
