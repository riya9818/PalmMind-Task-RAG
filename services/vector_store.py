import logging
import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from config import settings
from services.embeddings import get_embedding_dimension

logger = logging.getLogger(__name__)


def _get_client() -> QdrantClient:
    return QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)


def ensure_collection() -> None:
    """
    Create the Qdrant collection if it doesn't exist yet.
    Called once at startup.
    """
    client = _get_client()
    existing = [c.name for c in client.get_collections().collections]

    if settings.QDRANT_COLLECTION not in existing:
        dim = get_embedding_dimension()
        client.create_collection(
            collection_name=settings.QDRANT_COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        logger.info("Created Qdrant collection '%s' with dim=%d", settings.QDRANT_COLLECTION, dim)
    else:
        logger.info("Qdrant collection '%s' already exists", settings.QDRANT_COLLECTION)


def store_chunks(
    chunks: list[str],
    vectors: list[list[float]],
    document_id: int,
    filename: str,
) -> list[str]:
    """
    Upsert chunk vectors into Qdrant.
    Returns the list of vector IDs assigned to each chunk.
    """
    client = _get_client()
    vector_ids: list[str] = []
    points: list[PointStruct] = []

    for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
        vid = str(uuid.uuid4())
        vector_ids.append(vid)
        points.append(
            PointStruct(
                id=vid,
                vector=vector,
                payload={
                    "document_id": document_id,
                    "filename": filename,
                    "chunk_index": idx,
                    "content": chunk,
                },
            )
        )

    client.upsert(collection_name=settings.QDRANT_COLLECTION, points=points)
    logger.info("Stored %d vectors for document_id=%d", len(points), document_id)
    return vector_ids


def search_similar(query_vector: list[float], top_k: int = 5) -> list[dict[str, Any]]:
    """
    Search Qdrant for the top_k most similar chunks to the query vector.
    Returns a list of payloads (content + metadata).
    """
    client = _get_client()
    results = client.search(
        collection_name=settings.QDRANT_COLLECTION,
        query_vector=query_vector,
        limit=top_k,
        with_payload=True,
    )
    return [hit.payload for hit in results if hit.payload]
