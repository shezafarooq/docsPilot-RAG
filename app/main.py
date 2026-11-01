from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Path
from fastapi.responses import JSONResponse, StreamingResponse

from app.errors import ServiceError
from app.graph import get_graph, run_question, stream_question, read_thread
from app.schemas import AskRequest, AskResponse, ThreadHistory, THREAD_ID_PATTERN
from app.memory import shutdown_memory
from app.streaming import threaded_sse
from app.telemetry import shutdown_tracing


@asynccontextmanager
async def lifespan(app):
    yield
    shutdown_tracing()
    get_graph.cache_clear()
    shutdown_memory()

app = FastAPI(
    title="DocsPilot",
    description="Technical documentation Q&A with citations and bounded corrective retrieval.",
    version="0.2.0",
    lifespan=lifespan,
)


@app.exception_handler(ServiceError)
async def service_error_handler(request, error: ServiceError):
    return JSONResponse(status_code=error.status_code, content={"detail": str(error)})


@app.get("/health")
def health() -> dict[str, str]:
    """Process liveness only; this does not check Qdrant or Gemini."""
    return {"status": "ok", "service": "docspilot"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, graph=Depends(get_graph)) -> AskResponse:
    return run_question(request.question, graph, thread_id=request.thread_id)


@app.post("/ask/stream", response_class=StreamingResponse,
          responses={200: {"content": {"text/event-stream": {}}}})
def ask_stream(request: AskRequest, graph=Depends(get_graph)):
    return StreamingResponse(threaded_sse(lambda: stream_question(request.question, graph, thread_id=request.thread_id)), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no",
    })


@app.get("/threads/{thread_id}", response_model=ThreadHistory)
def thread_history(thread_id: Annotated[str, Path(min_length=1, max_length=128, pattern=THREAD_ID_PATTERN)], graph=Depends(get_graph)):
    return read_thread(thread_id, graph)
