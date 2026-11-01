import test from 'node:test';
import assert from 'node:assert/strict';
import { readEvents } from '../lib/stream.mjs';
import { forwardQuestion } from '../lib/proxy.mjs';

const encoder = new TextEncoder();
function response(parts) {
  return new Response(new ReadableStream({ start(controller) {
    for (const part of parts) controller.enqueue(typeof part === 'string' ? encoder.encode(part) : part);
    controller.close();
  } }), { headers: { 'Content-Type': 'text/event-stream' } });
}
const collect = async response => { const events = []; for await (const event of readEvents(response)) events.push(event); return events; };

test('parses multiple events across arbitrary packet and UTF-8 boundaries', async () => {
  const text = ': heartbeat\r\nevent: status\r\ndata: {"message":"Checking…"}\r\n\r\nevent: done\r\ndata: {}\r\n\r\n';
  const bytes = encoder.encode(text);
  const events = await collect(response(Array.from(bytes, byte => new Uint8Array([byte]))));
  assert.deepEqual(events, [{ event:'status',data:{message:'Checking…'} },{event:'done',data:{}}]);
});

test('rejects invalid JSON and truncated event frames', async () => {
  await assert.rejects(collect(response(['event: status\ndata: broken\n\n'])), /invalid update/);
  await assert.rejects(collect(response(['event: result\ndata: {"answer":"partial"}'])), /ended unexpectedly/);
  await assert.rejects(collect(new Response('{}')), /did not return/);
});

test('forwards streaming packets before upstream generation completes', async () => {
  let upstream;
  let cancelled = false;
  const stream = new ReadableStream({ start(controller) { upstream = controller; }, cancel() { cancelled = true; } });
  const request = new Request('http://localhost/api/chat/stream',{method:'POST',body:JSON.stringify({question:'How?'})});
  const result = await forwardQuestion(request,{stream:true,baseUrl:'http://api:8000',fetcher:async (url, options) => {
    assert.equal(url.href,'http://api:8000/ask/stream');
    assert.equal(JSON.parse(options.body).question,'How?');
    return new Response(stream,{headers:{'Content-Type':'text/event-stream'}});
  }});
  assert.equal(result.headers.get('Content-Type'),'text/event-stream');
  upstream.enqueue(encoder.encode('event: status\ndata: {}\n\n'));
  const reader=result.body.getReader();
  const first=await reader.read();
  assert.equal(new TextDecoder().decode(first.value),'event: status\ndata: {}\n\n');
  await reader.cancel();
  assert.equal(cancelled,true);
});

test('rejects a buffered JSON response where SSE was expected', async () => {
  const result=await forwardQuestion(new Request('http://localhost',{method:'POST',body:JSON.stringify({question:'Hi'})}),{
    stream:true,fetcher:async()=>Response.json({answer:'Hi'})
  });
  assert.equal(result.status,502);
});
