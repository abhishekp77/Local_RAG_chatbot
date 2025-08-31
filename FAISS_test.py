from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "The sky is blue.",
    "The sun is bright today.",
    "Dogs are loyal animals.",
    "I love programming in Python.",
    "Pizza tastes great with extra cheese."
]

embeddings = model.encode(sentences)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

print(f"Stored {index.ntotal} sentences in FAISS index")

query = "What is your favourite food?" 
query_embedding = model.encode([query]) 

k=2
distances , indices = index.search(np.array(query_embedding),k)

print("\nQuery:", query)
print("Top matches:")
for i, idx in enumerate(indices[0]):
    print(f"{i+1}. {sentences[idx]} (distance: {distances[0][i]:.4f})")