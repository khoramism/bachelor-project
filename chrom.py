import re
from sentence_transformers import SentenceTransformer
import lancedb
import pyarrow as pa
import pandas as pd

with open("./hafez.txt", 'r', encoding='utf8') as file:
    text = file.read()
# print(text)
# This regex captures a header (like "غزل   ۱") and then all following text until the next header or end-of-text.
pattern = re.compile(r'(غزل\s*\d+)(.*?)(?=غزل\s*\d+|$)', re.DOTALL)
matches = pattern.findall(text)
# matches = matches[:3]

documents = []
doc_ids = []

for header, body in matches:
    header = header.strip()
    content = body.strip()
    full_doc = f"{header}\n{content}"
    documents.append(full_doc)
    doc_ids.append(header)

# 3. Create embeddings and store in LanceDB
model = SentenceTransformer('heydariAI/persian-embeddings')
embeddings = model.encode(documents)

# Connect to LanceDB
db = lancedb.connect("lancedb_dir")
db.drop_all_tables()
# Create a schema with proper vector type
schema = pa.schema([
    pa.field("id", pa.string()),
    pa.field("document", pa.string()),
    pa.field("embedding", pa.list_(pa.float32(), list_size=embeddings.shape[1]))
])

# Create the table
table = db.create_table("ghazals", schema=schema)

# Prepare data for insertion
data = []
for doc_id, doc, emb in zip(doc_ids, documents, embeddings):
    data.append({
        "id": doc_id,
        "document": doc,
        "embedding": emb.tolist()  # Convert numpy array to list
    })

# Insert data
table.add(data)