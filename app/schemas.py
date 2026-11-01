from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


THREAD_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$"


class AskRequest(StrictModel):
    thread_id: str | None = Field(default=None, min_length=1, max_length=128, pattern=THREAD_ID_PATTERN)
    question: str = Field(min_length=1, max_length=2000,
                          examples=["How do I mount a volume in Docker Compose?"])

    @field_validator("question")
    @classmethod
    def strip_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Question cannot be blank.")
        return value


class GradeDecision(StrictModel):
    sufficient: bool
    relevant_ids: list[str] = Field(max_length=5)
    reason: str = Field(min_length=1, max_length=500)


class RewriteDecision(StrictModel):
    query: str = Field(min_length=1, max_length=500)


class StandaloneQuestion(StrictModel):
    question: str = Field(min_length=1, max_length=2000)


class Statement(StrictModel):
    kind: Literal["text", "code"]
    language: Literal["yaml", "python", "bash", "json", "text"] = "text"
    text: str = Field(min_length=1, max_length=4000)
    citation_ids: list[str] = Field(min_length=1, max_length=5)


class AnswerDraft(StrictModel):
    supported: bool
    statements: list[Statement] = Field(max_length=12)


class Source(StrictModel):
    id: int
    chunk_id: str
    title: str
    heading: str
    url: str
    passage: str


class AskResponse(StrictModel):
    answer: str
    status: Literal["answered", "direct", "not_found"]
    sources: list[Source] = Field(default_factory=list)
    retrieval_attempts: int = 0
    rewrite_count: int = 0
    trace_id: str | None = None
    thread_id: str | None = None
    resolved_question: str | None = None


class ChatTurn(StrictModel):
    question: str
    resolved_question: str
    response: AskResponse


class ThreadHistory(StrictModel):
    thread_id: str
    turns: list[ChatTurn] = Field(default_factory=list)
