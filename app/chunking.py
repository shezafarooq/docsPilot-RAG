"""Split section text using the embedding model's actual tokenizer."""

from uuid import NAMESPACE_URL, uuid5


def semantic_blocks(text: str) -> list[tuple[str, str]]:
    """Keep paragraphs separate and preserve blank lines inside code fences."""
    blocks = []
    lines = []
    in_code = False

    def flush(kind):
        content = "".join(lines).strip()
        if content:
            blocks.append((kind, content))
        lines.clear()

    for line in text.splitlines(keepends=True):
        if line.strip().startswith("```"):
            if not in_code:
                flush("text")
                lines.append(line)
                in_code = True
            else:
                lines.append(line)
                flush("code")
                in_code = False
        elif not in_code and not line.strip():
            flush("text")
        else:
            lines.append(line)
    flush("code" if in_code else "text")
    return blocks


def chunk_document(document: dict, model, overlap: int = 30) -> list[dict]:
    chunks = []
    tokenizer = model.tokenizer
    budget = model.max_seq_length - tokenizer.num_special_tokens_to_add() - 8
    if not 0 <= overlap < budget:
        raise ValueError("Overlap must be smaller than the available token budget.")
    for section_index, section in enumerate(document["sections"]):
        groups = []
        for kind, text in semantic_blocks(section["text"]):
            if groups and groups[-1][0] == "text":
                previous = groups[-1][1]
                combined = previous + "\n\n" + text
                # Attach short introductions to their paragraph or code example,
                # without burying standalone definitions in unrelated prose.
                if (kind == "code" or len(tokenizer.encode(previous, add_special_tokens=False)) < 12) and len(tokenizer.encode(combined, add_special_tokens=False, verbose=False)) <= budget:
                    groups[-1] = (kind, combined)
                    continue
            groups.append((kind, text))
        for group_index, (_, text) in enumerate(groups):
            offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, verbose=False)["offset_mapping"]
            start = 0
            while start < len(offsets):
                end = min(start + budget, len(offsets))
                passage = text[offsets[start][0]:offsets[end - 1][1]]
                token_count = len(tokenizer.encode(passage, add_special_tokens=True))
                if token_count > model.max_seq_length:
                    raise ValueError("Chunk exceeds the embedding limit; refusing silent truncation.")
                identity = f"{document['source_url']}|{section_index}|{section['url']}|{group_index}|{start}|{passage}"
                chunks.append({
                    "id": str(uuid5(NAMESPACE_URL, identity)),
                    "source_url": document["source_url"],
                    "url": section["url"],
                    "title": document["title"],
                    "heading": section["heading"],
                    "project": document["project"],
                    "text": passage,
                    "embedding_text": passage,
                    "token_count": token_count,
                })
                if end == len(offsets):
                    break
                start = end - overlap
    return chunks
