import test from 'node:test';
import assert from 'node:assert/strict';
import { forwardQuestion } from '../lib/proxy.mjs';
import { safeSourceUrl, remarkCitations } from '../lib/chat.mjs';

const request = body => new Request('http://localhost/api/chat', { method: 'POST', body: JSON.stringify(body) });
const answer = { answer: 'Use volumes. [1]', status: 'answered', sources: [{ id: 1, title: 'Volumes', heading: 'Compose', url: 'https://docs.docker.com/', passage: 'Use volumes.' }], retrieval_attempts: 1, rewrite_count: 0, trace_id: null };

test('forwards only the trimmed question and preserves evidence and metadata', async () => {
  const response = await forwardQuestion(request({ question: '  How?  ', api_key: 'do-not-forward' }), {
    baseUrl: 'http://api:8000', fetcher: async (url, options) => {
      assert.equal(url.href, 'http://api:8000/ask');
      assert.deepEqual(JSON.parse(options.body), { question: 'How?' });
      assert.equal(options.cache, 'no-store');
      return Response.json(answer);
    },
  });
  assert.equal(response.status, 200);
  assert.deepEqual(await response.json(), answer);
});

test('rejects invalid inputs before contacting the API', async () => {
  for (const value of [{}, null, { question: '  ' }, { question: 3 }, { question: 'x'.repeat(2001) }]) {
    const response = await forwardQuestion(request(value), { fetcher: () => assert.fail('should not fetch') });
    assert.equal(response.status, 422);
  }
  assert.equal((await forwardQuestion(new Request('http://localhost', { method: 'POST', body: '{' }))).status, 400);
  assert.equal((await forwardQuestion(request({ question: 'x'.repeat(11000) }))).status, 413);
});

test('preserves not_found as a successful response', async () => {
  const result = { ...answer, answer: "I couldn't find this in the docs.", status: 'not_found', sources: [], retrieval_attempts: 3, rewrite_count: 2 };
  const response = await forwardQuestion(request({ question: 'Unsupported?' }), { fetcher: async () => Response.json(result) });
  assert.equal(response.status, 200);
  assert.deepEqual(await response.json(), result);
});

test('maps outages and quota errors without exposing upstream details', async () => {
  for (const [upstream, expected] of [[429,429], [503,503], [504,504], [500,502], [422,422]]) {
    const response = await forwardQuestion(request({ question: 'Hi' }), { fetcher: async () => Response.json({ detail: 'private-provider-details' }, { status: upstream }) });
    assert.equal(response.status, expected);
    assert.ok(!(await response.text()).includes('private-provider-details'));
  }
  const offline = await forwardQuestion(request({ question: 'Hi' }), { fetcher: async () => { throw new Error('private-url'); } });
  assert.equal(offline.status, 502);
  assert.ok(!(await offline.text()).includes('private-url'));
});

test('rejects malformed API success responses and handles timeout', async () => {
  const response = await forwardQuestion(request({ question: 'Hi' }), { fetcher: async () => Response.json({ answer: 5 }) });
  assert.equal(response.status, 502);
  const timeout = await forwardQuestion(request({ question: 'Hi' }), { fetcher: async () => { throw new DOMException('timeout', 'TimeoutError'); } });
  assert.equal(timeout.status, 504);
});

test('blocks unsafe source URL schemes', () => {
  assert.equal(safeSourceUrl('javascript:alert(1)'), undefined);
  assert.equal(safeSourceUrl('data:text/html,bad'), undefined);
  assert.equal(safeSourceUrl('https://docs.docker.com/'), 'https://docs.docker.com/');
});

test('links known citations without transforming code or existing links', () => {
  const code = { type: 'code', value: 'array[1]' };
  const link = { type: 'link', url: 'https://example.com', children: [{ type: 'text', value: '[1]' }] };
  const tree = { type: 'root', children: [{ type: 'paragraph', children: [{ type: 'text', value: 'See [1], [2], and [99].' }, { type: 'inlineCode', value: 'x[1]' }, link] }, code] };
  remarkCitations({ prefix: 'source-msg', ids: [1,2] })()(tree);
  const children = tree.children[0].children;
  assert.equal(children.filter(node => node.type === 'link').length, 3);
  assert.equal(children[1].url, '#source-msg-1');
  assert.ok(children.some(node => node.type === 'text' && node.value.includes('[99]')));
  assert.equal(code.value, 'array[1]');
  assert.equal(link.children[0].value, '[1]');
});
