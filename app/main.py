from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse

from app.errors import ServiceError
from app.graph import get_graph, run_question
from app.schemas import AskRequest, AskResponse
from app.telemetry import shutdown_tracing


@asynccontextmanager
async def lifespan(app):
    yield
    shutdown_tracing()

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
    return run_question(request.question, graph)
