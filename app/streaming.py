"""SSE framing and complete-statement previews for structured model output."""

import json

from app.schemas import Statement


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def complete_statements(text: str) -> list[Statement]:
    """Read only closed statement objects; never render partial strings or IDs."""
    decoder = json.JSONDecoder()
    cursor = 0
    statements = []

    def space(index):
        while index < len(text) and text[index].isspace():
            index += 1
        return index

    cursor = space(cursor)
    if cursor >= len(text) or text[cursor] != "{":
        return []
    cursor += 1
    try:
        while True:
            cursor = space(cursor)
            key, cursor = decoder.raw_decode(text, cursor)
            cursor = space(cursor)
            if cursor >= len(text) or text[cursor] != ":":
                return []
            cursor = space(cursor + 1)
            if key == "statements":
                if cursor >= len(text) or text[cursor] != "[":
                    return []
                cursor += 1
                while True:
                    cursor = space(cursor)
                    if cursor >= len(text) or text[cursor] == "]":
                        return statements
                    value, cursor = decoder.raw_decode(text, cursor)
                    statements.append(Statement.model_validate(value))
                    if len(statements) > 12:
                        raise ValueError("Too many answer sections")
                    cursor = space(cursor)
                    if cursor >= len(text) or text[cursor] != ",":
                        return statements
                    cursor += 1
            _, cursor = decoder.raw_decode(text, cursor)
            cursor = space(cursor)
            if cursor >= len(text) or text[cursor] != ",":
                return []
            cursor += 1
    except json.JSONDecodeError:
        # The next network chunk may finish the current JSON value.
        return statements


def threaded_sse(events_factory):
    """Keep graph execution and tracing in one worker across HTTP iterator yields."""
    from contextlib import closing
    from queue import Empty, Full, Queue
    from threading import Event, Thread

    queue = Queue(maxsize=8)
    stopped = Event()

    def put(item):
        while not stopped.is_set():
            try:
                queue.put(item, timeout=0.2)
                return True
            except Full:
                pass
        return False

    def produce():
        try:
            with closing(events_factory()) as events:
                for item in events:
                    if not put(item):
                        break
        finally:
            put(None)

    Thread(target=produce, daemon=True, name="docspilot-stream").start()
    try:
        while True:
            try:
                item = queue.get(timeout=10)
            except Empty:
                yield ": keep-alive\n\n"
                continue
            if item is None:
                return
            yield sse(item["event"], item["data"])
    finally:
        stopped.set()
