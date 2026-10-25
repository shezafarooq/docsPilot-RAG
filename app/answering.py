"""Construct source citations from retrieved records, never from model-written URLs."""

import re

from app.errors import ServiceError
from app.schemas import AnswerDraft, AskResponse, Source

NOT_FOUND = "I couldn't find this in the docs."


def render_answer(draft: AnswerDraft, passages: list[dict]) -> AskResponse:
    if not draft.supported:
        if draft.statements:
            raise ServiceError("Gemini returned an inconsistent answer.")
        return AskResponse(answer=NOT_FOUND, status="not_found")
    if not draft.statements:
        raise ServiceError("Gemini returned an answer without supporting statements.")
    available = {passage["id"]: passage for passage in passages}
    sources = []
    numbers = {}
    paragraphs = []
    for statement in draft.statements:
        prose = "" if statement.kind == "code" else re.sub(r"```.*?```", "", statement.text, flags=re.S)
        if (not statement.text.strip() or re.search(r"\[[^\]]+\]\(", statement.text)
                or re.search(r"\[\d+\]", prose)):
            raise ServiceError("Gemini returned invalid statement text or a model-written source link.")
        labels = []
        for chunk_id in dict.fromkeys(statement.citation_ids):
            if chunk_id not in available:
                raise ServiceError("Gemini cited a passage that was not supplied as evidence.")
            if chunk_id not in numbers:
                passage = available[chunk_id]
                numbers[chunk_id] = len(sources) + 1
                sources.append(Source(
                    id=numbers[chunk_id], chunk_id=chunk_id, title=passage["title"],
                    heading=passage["heading"], url=passage["url"], passage=passage["text"],
                ))
            labels.append(f"[{numbers[chunk_id]}]")
        text = statement.text.strip()
        if statement.kind == "code":
            if "```" in text:
                raise ServiceError("Gemini returned malformed code formatting.")
            text = f"```{statement.language}\n{text}\n```"
        separator = "\n\n" if text.endswith("```") else " "
        paragraphs.append(text + separator + " ".join(labels))
    return AskResponse(answer="\n\n".join(paragraphs), status="answered", sources=sources)
