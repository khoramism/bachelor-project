import re
import os
import logging
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

# Configure root logger
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# 1. Load text file
file_path = "./hafez.txt"
try:
    logger.debug(f"Attempting to open file at: {file_path}")
    with open(file_path, 'r', encoding='utf8') as file:
        text = file.read()
    logger.info(f"Loaded text file, length={len(text)} characters")
except Exception as e:
    logger.exception(f"Failed to read file: {file_path}")
    raise

# 2. Parse ghazals and verses
pattern = re.compile(r'(غزل\s*\d+)(.*?)(?=غزل\s*\d+|$)', re.DOTALL)
matches = pattern.findall(text)
logger.info(f"Found {len(matches)} ghazal headers in text")
if not matches:
    logger.warning("No ghazals matched the regex. Check the pattern or file content.")

verse_data = []
ghazal_data = []
for header, body in matches:
    header = header.strip()
    verses = [v.strip() for v in body.strip().split('\n') if v.strip()]
    logger.debug(f"Processing {header}: {len(verses)} verses found")
    full_ghazal = "\n".join(verses)

    ghazal_data.append({
        "ghazal_id": header,
        "full_ghazal": f"{header}\n{full_ghazal}"
    })

    for i in range(0, len(verses), 2):
        if i + 1 < len(verses):
            beyt = f"{verses[i]}\n{verses[i+1]}"
            verse_id = f"{header}_beyt{(i//2)+1}"
            verse_data.append({
                "verse_id": verse_id,
                "verse_text": beyt,
                "ghazal_id": header,
                "full_ghazal": f"{header}\n{full_ghazal}"
            })
            logger.debug(f"Appended verse {verse_id}")

logger.info(f"Total ghazals parsed: {len(ghazal_data)}")
logger.info(f"Total couplets (beyts) parsed: {len(verse_data)}")

if not verse_data:
    logger.error("No verses parsed—nothing to embed or upload.")

# 3. Create embeddings
try:
    logger.debug("Loading embedding model: heydariAI/persian-embeddings")
    model = SentenceTransformer('heydariAI/persian-embeddings')
    beyt_texts = [v["verse_text"] for v in verse_data]
    logger.info(f"Encoding {len(beyt_texts)} couplets")
    embeddings = model.encode(beyt_texts, show_progress_bar=True, convert_to_numpy=True)
    logger.info(f"Embeddings shape: {embeddings.shape}")
except Exception as e:
    logger.exception("Failed to generate embeddings")
    raise

# 4. Connect to Qdrant
host = os.getenv("QDRANT_HOST", "localhost")
port = int(os.getenv("QDRANT_PORT", 6333))
logger.debug(f"Connecting to Qdrant at {host}:{port}")
try:
    client = QdrantClient(host=host, port=port, timeout=120)
    logger.info("Successfully connected to Qdrant")
except Exception as e:
    logger.exception("Failed to connect to Qdrant")
    raise

# 5. Recreate collection
collection_name = "ghazals"
try:
    logger.debug(f"Recreating collection: {collection_name}")
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=embeddings.shape[1], distance=Distance.COSINE),
    )
    logger.info(f"Collection '{collection_name}' recreated")
except Exception as e:
    logger.exception(f"Failed to recreate collection '{collection_name}'")
    raise

# 6. Prepare and upload points
points = []
for verse, emb in zip(verse_data, embeddings):
    points.append(
        PointStruct(
            id=len(points) + 1,  # Use a simple incremental ID
            vector=emb.tolist(),
            payload={
                "verse_text": verse["verse_text"],
                "ghazal_id": verse["ghazal_id"],
                "full_ghazal": verse["full_ghazal"],
            },
        )
    )
logger.info(f"Prepared {len(points)} points for upload to Qdrant")
# Upload points in small batches to avoid failures
batch_size = 64
total_points = len(points)
logger.info(f"Uploading points in batches of {batch_size}")

for start in range(0, total_points, batch_size):
    end = min(start + batch_size, total_points)
    batch = points[start:end]
    try:
        result = client.upsert(collection_name=collection_name, points=batch, wait=True)
        logger.info(f"Uploaded batch {start+1}-{end} ({len(batch)} points): "
                    f"{result.count if hasattr(result, 'count') else 'unknown count'} points upserted")
    except Exception as e:
        logger.exception(f"Failed to upload batch {start+1}-{end} to Qdrant")
        raise
