import re
from sentence_transformers import SentenceTransformer
import lancedb
import pyarrow as pa
import pandas as pd

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

# Connect to LanceDB
db = lancedb.connect("lancedb_dir")
db.drop_all_tables()

# Create schema 
schema = pa.schema([
    pa.field("verse_id", pa.string()),
    pa.field("verse_text", pa.string()),
    pa.field("ghazal_id", pa.string()),
    pa.field("full_ghazal", pa.string()),
    pa.field("embedding", pa.list_(pa.float32(), list_size=embeddings.shape[1]))
])

# Create table
table = db.create_table("ghazals", schema=schema)

# Prepare data for insertion
data = []
for verse, emb in zip(verse_data, embeddings):
    data.append({
        "verse_id": verse["verse_id"],
        "verse_text": verse["verse_text"],
        "ghazal_id": verse["ghazal_id"],
        "full_ghazal": verse["full_ghazal"],
        "embedding": emb.tolist()
    })

# Insert data
table.add(data)