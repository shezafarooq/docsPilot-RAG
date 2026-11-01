'use client';

import { useEffect, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { readEvents } from '../lib/stream.mjs';
import { isAnswer, remarkCitations, safeSourceUrl, validThreadId, isThreadHistory } from '../lib/chat.mjs';

const examples = [
  { topic: 'DOCKER', question: 'How do I mount a volume in Docker Compose?' },
  { topic: 'FASTAPI', question: 'How do I declare a request body in FastAPI?' },
  { topic: 'DOCKER', question: 'What is a Docker container?' },
];

function Mark({ small = false }) {
  return <span className={`brand-mark ${small ? 'small' : ''}`} aria-hidden="true">d<span>p</span><i /></span>;
}

function Answer({ message }) {
  const result = message.result;
  const prefix = `source-${message.id}`;
  return <>
    <div className="message-label"><Mark small /> DocsPilot <span className={`result-status ${result.status}`}>{message.pending ? 'Writing · preview' : result.status === 'answered' ? 'Sources checked' : result.status === 'not_found' ? 'No supporting docs' : 'Assistant'}</span></div>
    <div className="markdown">
      <ReactMarkdown remarkPlugins={[remarkGfm, remarkCitations({ prefix, ids: result.sources.map(source => source.id) })]}
        components={{ a: ({ href, children }) => href?.startsWith(`#${prefix}-`)
          ? <a className="citation" href={href}>{children}</a>
          : <a href={safeSourceUrl(href)} target="_blank" rel="noopener noreferrer">{children}</a> }}>
        {result.answer}
      </ReactMarkdown>
    </div>
    {result.sources.length > 0 && <div className="sources"><h3>Sources <span>{result.sources.length}</span></h3>
      {result.sources.map(source => <div className="source-card" id={`${prefix}-${source.id}`} key={source.id}>
        <div className="source-title"><span className="source-number">{source.id}</span><a href={safeSourceUrl(source.url)} target="_blank" rel="noopener noreferrer">{source.title}<span aria-hidden="true"> ↗</span></a></div>
        <p className="source-heading">{source.heading}</p>
        <details><summary>View supporting passage</summary><div className="markdown passage"><ReactMarkdown remarkPlugins={[remarkGfm]}>{source.passage}</ReactMarkdown></div></details>
      </div>)}
    </div>}
    {!message.pending && <details className="request-details"><summary>Request details</summary><dl>
      <div><dt>Retrieval attempts</dt><dd>{result.retrieval_attempts}</dd></div>
      <div><dt>Query rewrites</dt><dd>{result.rewrite_count}</dd></div>
      {result.thread_id && <div><dt>Thread ID</dt><dd className="trace-id">{result.thread_id}</dd></div>}
      {result.resolved_question && <div><dt>Understood question</dt><dd>{result.resolved_question}</dd></div>}
      {result.trace_id && <div><dt>Trace ID</dt><dd className="trace-id">{result.trace_id}</dd></div>}
    </dl></details>}
  </>;
}

export default function ChatPage() {
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState('');
  const [busy, setBusy] = useState(false);
  const [restoring, setRestoring] = useState(true);
  const threadId = useRef(null);
  const [error, setError] = useState(null);
  const [progress, setProgress] = useState('');
  const [failedQuestion, setFailedQuestion] = useState('');
  const controller = useRef(null);
  const inFlight = useRef(false);
  const bottom = useRef(null);
  const input = useRef(null);

  useEffect(() => { bottom.current?.scrollIntoView({ behavior: 'smooth', block: 'end' }); }, [messages, busy, error, progress]);
  useEffect(() => () => controller.current?.abort(), []);
  useEffect(() => {
    const request = new AbortController();
    let active = true;
    let saved = null;
    try { saved = sessionStorage.getItem('docspilot.thread_id'); } catch { /* Use in-page state if storage is unavailable. */ }
    const existing = validThreadId(saved);
    threadId.current = existing ? saved : crypto.randomUUID();
    try { sessionStorage.setItem('docspilot.thread_id', threadId.current); } catch { /* Server checkpoints still work. */ }
    if (!existing) {
      setRestoring(false);
      return () => { active = false; request.abort(); };
    }
    fetch(`/api/chat/history?thread_id=${encodeURIComponent(threadId.current)}`, { signal: request.signal, cache: 'no-store' })
      .then(async response => {
        if (!response.ok) throw new Error('History unavailable');
        const data = await response.json();
        if (!isThreadHistory(data) || data.thread_id !== threadId.current) throw new Error('Invalid history');
        if (active) setMessages(data.turns.flatMap((turn, index) => [
          { id: `saved-user-${index}`, role: 'user', text: turn.question },
          { id: `saved-answer-${index}`, role: 'assistant', result: turn.response },
        ]));
      })
      .catch(() => { if (active && !request.signal.aborted) setError('Could not load this saved chat. Reload to retry, or start a new chat.'); })
      .finally(() => { if (active) setRestoring(false); });
    return () => { active = false; request.abort(); };
  }, []);

  async function send(text, retry = false) {
    const trimmed = text.trim();
    if (!trimmed || restoring || !threadId.current || inFlight.current || Array.from(trimmed).length > 2000) return;
    inFlight.current = true;
    setBusy(true);
    setProgress('Connecting to DocsPilot…');
    setError(null);
    setFailedQuestion('');
    setQuestion('');
    if (!retry) setMessages(previous => [...previous, { id: crypto.randomUUID(), role: 'user', text: trimmed }]);
    const request = new AbortController();
    controller.current = request;
    const answerId = crypto.randomUUID();
    function showAnswer(result, pending) {
      setMessages(previous => {
        const next = { id: answerId, role: 'assistant', result, pending };
        return previous.some(message => message.id === answerId)
          ? previous.map(message => message.id === answerId ? next : message)
          : [...previous, next];
      });
    }
    try {
      const response = await fetch('/api/chat/stream', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: trimmed, thread_id: threadId.current }), signal: request.signal,
      });
      if (!response.ok) {
        const data = await response.json();
        throw new Error(typeof data.detail === 'string' ? data.detail : 'The request failed. Please try again.');
      }
      let finalResult = null;
      let finished = false;
      for await (const { event, data } of readEvents(response)) {
        if (event === 'thread' && data.thread_id !== threadId.current) throw new Error('The API returned a different chat ID.');
        if (event === 'error') throw new Error(typeof data.detail === 'string' ? data.detail : 'The answer stream failed.');
        if (event === 'status' && typeof data.message === 'string') setProgress(data.message);
        if (event === 'reset') setMessages(previous => previous.filter(message => message.id !== answerId));
        if (event === 'answer' || event === 'result') {
          if (!isAnswer(data)) throw new Error('The API returned an invalid answer update.');
          showAnswer(data, event === 'answer');
          if (event === 'result') finalResult = data;
        }
        if (event === 'done') {
          if (!finalResult) throw new Error('The stream ended without a final answer.');
          finished = true;
          break;
        }
      }
      if (!finished) throw new Error('The connection ended before the answer was complete. Please retry.');
    } catch (failure) {
      // A stopped/invalid stream must not leave a preview looking like a final answer.
      setMessages(previous => previous.filter(message => message.id !== answerId));
      setError(request.signal.aborted ? 'Request stopped. You can retry your question.' : failure.message || 'Connection lost. Please try again.');
      setFailedQuestion(trimmed);
    } finally {
      inFlight.current = false;
      controller.current = null;
      setBusy(false);
    }
  }

  function newChat() {
    if (inFlight.current || restoring) return;
    threadId.current = crypto.randomUUID();
    try { sessionStorage.setItem('docspilot.thread_id', threadId.current); } catch { /* Keep the new ID in memory. */ }
    setMessages([]); setQuestion(''); setError(null); setFailedQuestion(''); input.current?.focus();
  }

  return <div className="app-shell">
    <aside className="sidebar">
      <a className="brand" href="/"><Mark /><span>DocsPilot<span className="brand-subtitle">YOUR DOCUMENTATION COPILOT</span></span></a>
      <button className="new-chat" onClick={newChat} disabled={busy || restoring}><span aria-hidden="true">＋</span> New chat <span className="button-arrow" aria-hidden="true">↗</span></button>
      <div className="sidebar-section"><span className="eyebrow">THE KNOWLEDGE BASE</span>
        <div className="doc-item"><span className="doc-icon docker" aria-hidden="true">▦</span><div>Docker<span>Containers & Compose</span></div><span className="available-dot" /></div>
        <div className="doc-item"><span className="doc-icon fastapi" aria-hidden="true">ϟ</span><div>FastAPI<span>Build Python APIs</span></div><span className="available-dot" /></div>
      </div>
      <div className="sidebar-note"><span className="note-icon" aria-hidden="true">↗</span><h3>Follow the source.</h3><p>Every documentation answer comes with the passages behind it.</p></div>
      <div className="sidebar-footer"><span className="local-dot" /> Local workspace<span className="version">v0.3</span></div>
    </aside>
    <main>
      <header className="topbar"><div><span className="breadcrumb">Workspace</span><span className="slash">/</span><span>Ask the docs</span></div><span className="header-badge">Docker + FastAPI</span></header>
      <div className="chat-layout">
        {messages.length === 0 ? <section className="welcome">
          <span className="welcome-label"><span className="spark" aria-hidden="true">✳</span> LESS SEARCHING. MORE BUILDING.</span>
          <h1>Your docs.<br /><span>A clear answer.</span></h1>
          <p className="intro">Ask a question about Docker or FastAPI.<br />Get an answer grounded in documentation, with sources to back it up.</p>
          <div className="example-grid">{examples.map((example, index) => <button key={example.question} className="example-card" onClick={() => send(example.question)} disabled={busy || restoring}>
            <span className="example-top"><span className="example-topic">{example.topic}</span><span className="example-index">0{index + 1}</span></span>
            <span className="example-question">{example.question}</span><span className="example-arrow" aria-hidden="true">↗</span>
          </button>)}</div>
          <div className="how-it-works"><span>Retrieve</span><span aria-hidden="true">→</span><span>Check evidence</span><span aria-hidden="true">→</span><span>Answer with sources</span></div>
        </section> : <section className="conversation" aria-label="Conversation" aria-busy={busy}>
          {messages.map(message => <article key={message.id} className={`message ${message.role}`}>
            {message.role === 'user' ? <><div className="message-label"><span className="user-avatar" aria-hidden="true">Y</span>You</div><p className="user-question">{message.text}</p></> : <Answer message={message} />}
          </article>)}
        </section>}
        <div role="status" className="status-region">{(busy || restoring) && <div className="thinking"><span className="loading-dot" /> {restoring ? 'Loading your saved chat…' : progress}</div>}</div>
        {error && <div className="error-box" role="alert"><p>{error}</p>{failedQuestion ? <button onClick={() => send(failedQuestion, true)} disabled={busy || restoring}>Retry question <span aria-hidden="true">↗</span></button> : <button onClick={() => window.location.reload()}>Reload chat ↗</button>}</div>}
        <div ref={bottom} />
      </div>
      <div className="composer-area"><form className="composer" onSubmit={event => { event.preventDefault(); send(question); }}>
        <label className="sr-only" htmlFor="question">Ask about Docker or FastAPI</label>
        <textarea ref={input} id="question" rows={2} placeholder="How do I…" value={question} maxLength={2000} disabled={busy || restoring}
          onChange={event => setQuestion(event.target.value)} onKeyDown={event => { if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) { event.preventDefault(); send(question); } }} />
        <div className="composer-toolbar"><span className="composer-hint">Docker & FastAPI docs<span className="input-count">{Array.from(question).length}/2000</span></span>
          {busy ? <button type="button" className="send-button stop-button" onClick={() => controller.current?.abort()}>Stop <span aria-hidden="true">■</span></button>
            : <button type="submit" className="send-button" disabled={restoring || !question.trim()}>Ask DocsPilot <span aria-hidden="true">↑</span></button>}
        </div>
      </form><p className="composer-note">Answers use selected docs. Follow-up questions use this chat’s context. <span>Enter to send · Shift + Enter for a new line</span></p></div>
    </main>
  </div>;
}
