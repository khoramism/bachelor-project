import os
import numpy as np
from qdrant_client import QdrantClient

def search(embedding: list, limit: int = 3):
    """Query the Qdrant collection for closest verses."""
    client = QdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", 6333)),
    )

    if isinstance(embedding, np.ndarray):
        embedding = embedding.tolist()

    hits = client.query_points(
        collection_name="ghazals",
        query=embedding,
        limit=limit,
    )

    results = [
        {
            "verse_text": hit.payload.get("verse_text"),
            "ghazal_id": hit.payload.get("ghazal_id"),
            "score": hit.score,
        }
        for hit in hits
    ]

    return results

