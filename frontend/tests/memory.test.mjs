import test from 'node:test';
import assert from 'node:assert/strict';
import { forwardQuestion, forwardHistory } from '../lib/proxy.mjs';
import { validThreadId } from '../lib/chat.mjs';
const answer={answer:'Hi',status:'direct',sources:[],retrieval_attempts:0,rewrite_count:0,trace_id:null,thread_id:'chat-1'};

test('forwards a thread ID with questions while dropping unrelated fields',async()=>{
 const request=new Request('http://localhost/api/chat',{method:'POST',body:JSON.stringify({question:'How?',thread_id:'chat-1',secret:'do-not-send'})});
 const response=await forwardQuestion(request,{fetcher:async(url,options)=>{
  assert.deepEqual(JSON.parse(options.body),{question:'How?',thread_id:'chat-1'});
  return Response.json(answer);
 }});
 assert.equal(response.status,200);
 assert.equal((await response.json()).thread_id,'chat-1');
});

test('rejects invalid thread IDs without calling the backend',async()=>{
 for(const thread_id of ['', '../other','x'.repeat(129),42,true]){
  const response=await forwardQuestion(new Request('http://localhost',{method:'POST',body:JSON.stringify({question:'How?',thread_id})}),{fetcher:()=>assert.fail('must not fetch')});
  assert.equal(response.status,422);
 }
 assert.equal(validThreadId('d93de710-2508-4b3b-9471-03aa639bce8f'),true);
});

test('loads only the requested thread history and disables caching',async()=>{
 const history={thread_id:'chat-1',turns:[{question:'hi',resolved_question:'hi',response:answer}]};
 const response=await forwardHistory(new Request('http://localhost/api/chat/history?thread_id=chat-1'),{baseUrl:'http://api:8000',fetcher:async(url,options)=>{
  assert.equal(url.href,'http://api:8000/threads/chat-1');
  assert.equal(options.cache,'no-store');
  return Response.json(history);
 }});
 assert.deepEqual(await response.json(),history);
 assert.equal(response.headers.get('Cache-Control'),'no-store');
});

test('rejects invalid/mismatched history and handles storage outages',async()=>{
 assert.equal((await forwardHistory(new Request('http://localhost/api/chat/history?thread_id=../other'))).status,422);
 for(const fetcher of [async()=>Response.json({thread_id:'other',turns:[]}),async()=>{throw new Error('private-url');},async()=>new Response('',{status:500})]){
  const response=await forwardHistory(new Request('http://localhost/api/chat/history?thread_id=chat-1'),{fetcher});
  assert.equal(response.status,502);
  assert.ok(!(await response.text()).includes('private-url'));
 }
});

test('preserves a useful conflict message for overlapping requests',async()=>{
 const response=await forwardQuestion(new Request('http://localhost',{method:'POST',body:JSON.stringify({question:'How?',thread_id:'chat-1'})}),{fetcher:async()=>new Response('',{status:409})});
 assert.equal(response.status,409);
 assert.match((await response.json()).detail,/still processing/);
});
