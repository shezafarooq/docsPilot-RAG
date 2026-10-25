"""Create a compatible collection and update documents without duplicate points."""

from qdrant_client import QdrantClient, models


def ensure_collection(client: QdrantClient, name: str, dimension: int, embedding_model: str) -> None:
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
        )
        return
    config = client.get_collection(name).config.params.vectors
    if not isinstance(config, models.VectorParams) or config.size != dimension or config.distance != models.Distance.COSINE:
        raise ValueError("Existing collection uses incompatible vectors; use a new collection name.")
    points, _ = client.scroll(name, limit=1, with_vectors=False)
    if points and points[0].payload.get("embedding_model") != embedding_model:
        raise ValueError("Existing collection uses a different embedding model; use a new collection name.")


def source_filter(source_url: str) -> models.Filter:
    return models.Filter(must=[models.FieldCondition(key="source_url", match=models.MatchValue(value=source_url))])


def sync_source(client: QdrantClient, name: str, chunks: list[dict], vectors, embedding_model: str) -> None:
    if not chunks:
        raise ValueError("Refusing to index an empty document.")
    if len(chunks) != len(vectors):
        raise ValueError("Every chunk must have an embedding.")
    old_ids = set()
    offset = None
    while True:
        points, offset = client.scroll(
            name, scroll_filter=source_filter(chunks[0]["source_url"]),
            limit=256, offset=offset, with_payload=False, with_vectors=False,
        )
        old_ids.update(str(point.id) for point in points)
        if offset is None:
            break
    for start in range(0, len(chunks), 64):
        batch = []
        for chunk, vector in zip(chunks[start:start + 64], vectors[start:start + 64]):
            payload = {key: value for key, value in chunk.items() if key not in {"id", "embedding_text"}}
            payload["embedding_model"] = embedding_model
            batch.append(models.PointStruct(id=chunk["id"], vector=list(vector), payload=payload))
        client.upsert(name, points=batch, wait=True)
    # Remove superseded passages only after replacements have been stored.
    stale_ids = old_ids - {chunk["id"] for chunk in chunks}
    if stale_ids:
        client.delete(name, points_selector=models.PointIdsList(points=sorted(stale_ids)), wait=True)
