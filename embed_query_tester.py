from sentence_transformers import SentenceTransformer
from query import search

# Define a query sentence (in Persian, relevant to the topic of interest)
sentences = [
    'تضاد میان تقوا و صداقت واقعی را نشان می‌دهد. شاعر از ریاکاری و ظواهر دینی ناراضی است و در آرزوی رهایی از محدودیت‌های اجتماعی و رسیدن به عشق واقعی است'
]

model = SentenceTransformer('heydariAI/persian-embeddings')

embeddings = model.encode(sentences)

print("Embedding Shape:", embeddings.shape)

results = search(embeddings[0])  

# Display the search results
print("Search Results:")
print(results)
