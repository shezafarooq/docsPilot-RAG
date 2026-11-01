// Network packets need not end on event boundaries, or even UTF-8 boundaries.
export async function* readEvents(response) {
  if (!response.body || !response.headers.get('content-type')?.includes('text/event-stream')) {
    throw new Error('The API did not return an answer stream.');
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    while (true) {
      const { done, value } = await reader.read();
      buffer += decoder.decode(value, { stream: !done });
      buffer = buffer.replace(/\r\n/g, '\n');
      if (buffer.length > 1_000_000) throw new Error('The answer stream exceeded its size limit.');
      let end;
      while ((end = buffer.indexOf('\n\n')) !== -1) {
        const frame = buffer.slice(0, end);
        buffer = buffer.slice(end + 2);
        let event = 'message';
        const data = [];
        for (const line of frame.split('\n')) {
          if (line.startsWith('event:')) event = line.slice(6).trim();
          if (line.startsWith('data:')) data.push(line.slice(5).replace(/^ /, ''));
        }
        if (data.length) {
          let parsed;
          try { parsed = JSON.parse(data.join('\n')); }
          catch { throw new Error('The answer stream contained an invalid update.'); }
          yield { event, data: parsed };
        }
      }
      if (done) {
        if (buffer.trim()) throw new Error('The answer stream ended unexpectedly. Please retry.');
        return;
      }
    }
  } finally {
    try { await reader.cancel(); } catch { /* Already closed or aborted. */ }
    reader.releaseLock();
  }
}
