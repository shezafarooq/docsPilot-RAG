export function isAnswer(value) {
  return value && typeof value.answer === 'string'
    && ['answered', 'direct', 'not_found'].includes(value.status)
    && Array.isArray(value.sources)
    && value.sources.every(source => Number.isInteger(source.id)
      && ['title', 'heading', 'url', 'passage'].every(key => typeof source[key] === 'string'))
    && Number.isInteger(value.retrieval_attempts)
    && Number.isInteger(value.rewrite_count)
    && (value.trace_id === null || typeof value.trace_id === 'string');
}

export function safeSourceUrl(value) {
  try {
    const url = new URL(value);
    return url.protocol === 'https:' || url.protocol === 'http:' ? url.href : undefined;
  } catch {
    return undefined;
  }
}

// Transform citation markers in prose, leaving code and existing links untouched.
export function remarkCitations({ prefix, ids }) {
  return () => tree => {
    function walk(node) {
      if (!node.children || ['link', 'code', 'inlineCode'].includes(node.type)) return;
      node.children = node.children.flatMap(child => {
        if (child.type !== 'text') {
          walk(child);
          return [child];
        }
        const output = [];
        let start = 0;
        for (const match of child.value.matchAll(/\[(\d+)\]/g)) {
          if (!ids.includes(Number(match[1]))) continue;
          if (match.index > start) output.push({ type: 'text', value: child.value.slice(start, match.index) });
          output.push({ type: 'link', url: `#${prefix}-${match[1]}`, children: [{ type: 'text', value: match[0] }] });
          start = match.index + match[0].length;
        }
        if (start < child.value.length) output.push({ type: 'text', value: child.value.slice(start) });
        return output.length ? output : [child];
      });
    }
    walk(tree);
  };
}


export function validThreadId(value) {
  return typeof value === 'string' && /^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$/.test(value);
}

export function isThreadHistory(value) {
  return value && validThreadId(value.thread_id) && Array.isArray(value.turns)
    && value.turns.every(turn => typeof turn.question === 'string'
      && typeof turn.resolved_question === 'string' && isAnswer(turn.response));
}
