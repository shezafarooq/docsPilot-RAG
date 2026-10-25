"""Search indexed passages. This step retrieves evidence; it does not call Gemini."""

import argparse
import json

from qdrant_client import QdrantClient

from app.config import get_settings
from app.embeddings import get_embedding_model


def retrieve(question: str, limit: int = 5) -> list[dict]:
    if not question.strip():
        raise ValueError("Question cannot be empty.")
    if not 1 <= limit <= 20:
        raise ValueError("Limit must be between 1 and 20.")
    settings = get_settings()
    model = get_embedding_model()
    vector = model.encode(question, normalize_embeddings=True).tolist()
    client = QdrantClient(url=settings.qdrant_url, timeout=30)
    try:
        if not client.collection_exists(settings.qdrant_collection):
            raise ValueError("No indexed collection found. Run python -m app.ingest first.")
        hits = client.query_points(
            collection_name=settings.qdrant_collection,
            query=vector, limit=limit, with_payload=True,
        ).points
    finally:
        client.close()
    if any(hit.payload.get("embedding_model") != settings.embedding_model for hit in hits):
        raise ValueError("Indexed passages use a different embedding model; reindex into a new collection.")
    return [{"id": str(hit.id), "score": hit.score, **hit.payload} for hit in hits]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    for rank, result in enumerate(retrieve(args.question, args.limit), start=1):
        print(json.dumps({"rank": rank, **result}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
