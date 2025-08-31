import faiss
from sentence_transformers import SentenceTransformer
from ollama import chat

embedder = SentenceTransformer("all-MiniLM-L6-v2")

docs = [
    "The capital of France is Paris.",
    "Pizza tastes great with extra cheese.",
    "Python is a popular programming language.",
    "Ollama lets you run LLMs locally."
]

doc_embeddings = embedder.encode(docs)

dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings)

def rag_query(query, top_k=2):
    query_embed = embedder.encode([query])
    distances, indices = index.search(query_embed, top_k)
    retrieved = [docs[i] for i in indices[0]]

    context = "\n".join(retrieved)
    prompt = f"Answer the question based on the context.\n\nContext:\n{context}\n\nQuestion: {query}"

    response = chat(model="llama3", messages=[
        {"role": "user", "content": prompt}
    ])
    
    return response["message"]["content"]

# 🔍 Test it
print(rag_query("What is the capital of France?"))
print(rag_query("What does Ollama do?"))
