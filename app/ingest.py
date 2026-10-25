"""Download, chunk, embed, and index the curated documentation set."""

import argparse
import json
from datetime import datetime, timezone

from qdrant_client import QdrantClient

from app.chunking import chunk_document
from app.config import get_settings
from app.documents import SOURCES, load_document
from app.embeddings import get_embedding_model
from app.vector_store import ensure_collection, sync_source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Download new copies instead of using saved documents")
    args = parser.parse_args()
    settings = get_settings()
    documents = [load_document(url, settings.data_dir, args.refresh) for url in SOURCES]
    print("Loading the local embedding model (first run downloads its weights)...", flush=True)
    model = get_embedding_model(allow_download=True)
    grouped = [(document, chunk_document(document, model)) for document in documents]
    all_chunks = [chunk for _, chunks in grouped for chunk in chunks]
    snapshot = settings.data_dir / "chunks.jsonl"
    temporary = snapshot.with_suffix(".tmp")
    temporary.write_text("".join(json.dumps(chunk, ensure_ascii=False) + "\n" for chunk in all_chunks))
    temporary.replace(snapshot)
    client = QdrantClient(url=settings.qdrant_url, timeout=30)
    try:
        ensure_collection(client, settings.qdrant_collection, model.get_embedding_dimension(), settings.embedding_model)
        for document, chunks in grouped:
            print(f"Embedding and storing {len(chunks)} chunks: {document['source_url']}", flush=True)
            vectors = model.encode([chunk["embedding_text"] for chunk in chunks], batch_size=32, normalize_embeddings=True)
            sync_source(client, settings.qdrant_collection, chunks, vectors.tolist(), settings.embedding_model)
        count = client.count(settings.qdrant_collection, exact=True).count
    finally:
        client.close()
    summary = {
        "indexed_at": datetime.now(timezone.utc).isoformat(),
        "documents": len(documents), "chunks_in_this_run": len(all_chunks),
        "collection_points": count, "collection": settings.qdrant_collection,
        "embedding_model": settings.embedding_model,
        "max_chunk_tokens": max(chunk["token_count"] for chunk in all_chunks),
        "sources": list(SOURCES),
    }
    (settings.data_dir / "index_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
