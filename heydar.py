from sentence_transformers import SentenceTransformer
sentences = ['What are Large Language Models?','مدل های زبانی بزرگ چه هستند؟']

model = SentenceTransformer('heydariAI/persian-embeddings')
embeddings = model.encode(sentences)
print(embeddings)