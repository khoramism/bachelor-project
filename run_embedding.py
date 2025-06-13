import re
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
import os

with open("./hafez.txt", 'r', encoding='utf8') as file:
    text = file.read()

# Parse ghazals and verses
pattern = re.compile(r'(غزل\s*\d+)(.*?)(?=غزل\s*\d+|$)', re.DOTALL)
matches = pattern.findall(text)

# New data structure
verse_data = []
ghazal_data = []

for header, body in matches:
    header = header.strip()
    verses = [v.strip() for v in body.strip().split('\n') if v.strip()]
    full_ghazal = "\n".join(verses)
    
    # Store ghazal metadata
    ghazal_data.append({
        "ghazal_id": header,
        "full_ghazal": f"{header}\n{full_ghazal}"
    })
    
    # Store complete بيتs (couplets)
    for i in range(0, len(verses), 2):
        if i+1 < len(verses):  # Ensure we have pairs
            beyt = f"{verses[i]}\n{verses[i+1]}"
            verse_data.append({
                "verse_id": f"{header}_beyt{(i//2)+1}",
                "verse_text": beyt,
                "ghazal_id": header,
                "full_ghazal": f"{header}\n{full_ghazal}"
            })

# Create embeddings for complete beits
model = SentenceTransformer('heydariAI/persian-embeddings')
beyt_texts = [v["verse_text"] for v in verse_data]
embeddings = model.encode(beyt_texts)

# Connect to Qdrant
client = QdrantClient(host=os.getenv("QDRANT_HOST", "localhost"), port=int(os.getenv("QDRANT_PORT", 6333)))

# Recreate collection
client.recreate_collection(
    collection_name="ghazals",
    vectors_config=VectorParams(size=embeddings.shape[1], distance=Distance.COSINE),
)

# Prepare points
points = []
for verse, emb in zip(verse_data, embeddings):
    points.append(
        PointStruct(
            id=verse["verse_id"],
            vector=emb.tolist(),
            payload={
                "verse_text": verse["verse_text"],
                "ghazal_id": verse["ghazal_id"],
                "full_ghazal": verse["full_ghazal"],
            },
        )
    )

# Upload points
client.upsert(collection_name="ghazals", points=points)
