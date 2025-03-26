from sentence_transformers import SentenceTransformer
from query import search

# Define a query sentence (in Persian, relevant to the topic of interest)
sentences = [
    'تضاد میان تقوا و صداقت واقعی را نشان می‌دهد. شاعر از ریاکاری و ظواهر دینی ناراضی است و در آرزوی رهایی از محدودیت‌های اجتماعی و رسیدن به عشق واقعی است'
]

# Load Persian embedding model
model = SentenceTransformer('heydariAI/persian-embeddings')

# Generate embeddings for the query sentence
embeddings = model.encode(sentences)

# Debug: Ensure the embeddings are generated
print("Embedding Shape:", embeddings.shape)

# Perform the search using the `search` function
results = search(embeddings[0])  # Pass the first (and only) embedding

# Display the search results
print("Search Results:")
print(results)
