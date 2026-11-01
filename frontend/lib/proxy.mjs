import { isAnswer, validThreadId, isThreadHistory } from './chat.mjs';

const reply = (detail, status) => Response.json({ detail }, { status });

export async function forwardQuestion(request, { baseUrl, fetcher = fetch, timeoutMs = 240_000, stream = false } = {}) {
  let body;
  try {
    const raw = await request.text();
    if (raw.length > 10_000) return reply('Request is too large.', 413);
    body = JSON.parse(raw);
  } catch {
    return reply('Send a valid JSON question.', 400);
  }
  if (!body || typeof body.question !== 'string' || !body.question.trim()
      || Array.from(body.question.trim()).length > 2000) {
    return reply('Enter a question between 1 and 2,000 characters.', 422);
  }
  if (body.thread_id != null && !validThreadId(body.thread_id)) {
    return reply('The chat ID is invalid. Start a new chat.', 422);
  }
  const payload = { question: body.question.trim() };
  if (body.thread_id != null) payload.thread_id = body.thread_id;
  try {
    const url = new URL(stream ? '/ask/stream' : '/ask', baseUrl || 'http://localhost:8000');
    const response = await fetcher(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      cache: 'no-store',
      signal: AbortSignal.any([request.signal, AbortSignal.timeout(timeoutMs)]),
    });
    if (!response.ok) {
      // Keep raw upstream responses, URLs, and configuration out of the browser.
      if (response.status === 409) return reply('This chat is still processing a question. Please wait and retry.', 409);
      if (response.status === 429) return reply('The model request limit was reached. Please try again later.', 429);
      if (response.status === 422) return reply('The API could not accept this question.', 422);
      if (response.status === 503) return reply('Document search is unavailable. Check the API and Qdrant services.', 503);
      if (response.status === 504) return reply('The answer took too long. Please try again.', 504);
      return reply('The answer service could not complete this request. Please try again.', 502);
    }
    if (stream) {
      if (!response.body || !response.headers.get('content-type')?.includes('text/event-stream')) {
        return reply('The API did not return an answer stream.', 502);
      }
      // Forward bytes immediately instead of buffering with response.json().
      return new Response(response.body, { headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache, no-transform',
        'X-Accel-Buffering': 'no',
      } });
    }
    const result = await response.json();
    if (!isAnswer(result)) return reply('The API returned an unexpected response.', 502);
    return Response.json(result, { headers: { 'Cache-Control': 'no-store' } });
  } catch (error) {
    if (error.name === 'TimeoutError' || error.name === 'AbortError') {
      return reply('The request stopped or timed out. Please try again.', 504);
    }
    return reply('Cannot reach the API. Check that the API container is running.', 502);
  }
}


export async function forwardHistory(request, { baseUrl, fetcher = fetch } = {}) {
  const threadId = new URL(request.url).searchParams.get('thread_id');
  if (!validThreadId(threadId)) return reply('The chat ID is invalid.', 422);
  try {
    const response = await fetcher(new URL(`/threads/${encodeURIComponent(threadId)}`, baseUrl || 'http://localhost:8000'), {
      cache: 'no-store', signal: AbortSignal.any([request.signal, AbortSignal.timeout(30_000)]),
    });
    if (!response.ok) return reply('Could not load this chat. Please try again.', 502);
    const result = await response.json();
    if (!isThreadHistory(result) || result.thread_id !== threadId) return reply('The API returned an invalid chat history.', 502);
    return Response.json(result, { headers: { 'Cache-Control': 'no-store' } });
  } catch {
    return reply('Could not reach conversation storage. Check the API container.', 502);
  }
}
