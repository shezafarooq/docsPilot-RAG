"""One cached CPU embedding model shared by ingestion and retrieval."""

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.config import get_settings


@lru_cache
def get_embedding_model(allow_download: bool = False) -> SentenceTransformer:
    name = get_settings().embedding_model
    try:
        # Answering should use the already-downloaded model, with no Hub checks.
        return SentenceTransformer(name, device="cpu", local_files_only=True)
    except OSError:
        if not allow_download:
            raise ValueError("Embedding model is not cached. Run document ingestion first.") from None
        return SentenceTransformer(name, device="cpu")
