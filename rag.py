import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import ollama

DOCS_PATH = "data/docs"
INDEX_FILE = "faiss_index.bin"
TEXTS_FILE = "texts.pkl"
METAS_FILE = "metas.pkl"
EMBED_MODEL = "all-MiniLM-L6-v2"   

def load_docs(path=DOCS_PATH):
    docs, metas = [], []
    for root, _, files in os.walk(path):
        for f in files:
            file_path = os.path.join(root, f)
            if f.endswith(".pdf"):
                reader = PdfReader(file_path)
                text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
            elif f.endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as infile:
                    text = infile.read()
            else:
                continue
            
            chunks = [text[i:i+500] for i in range(0, len(text), 500)]
            docs.extend(chunks)
            metas.extend([{"source": f}] * len(chunks))
    return docs, metas

def build_faiss(docs, metas):
    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(docs, convert_to_numpy=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    with open(TEXTS_FILE, "wb") as f: pickle.dump(docs, f)
    with open(METAS_FILE, "wb") as f: pickle.dump(metas, f)
    faiss.write_index(index, INDEX_FILE)
    return index, model

def load_faiss():
    with open(TEXTS_FILE, "rb") as f: docs = pickle.load(f)
    with open(METAS_FILE, "rb") as f: metas = pickle.load(f)
    index = faiss.read_index(INDEX_FILE)
    model = SentenceTransformer(EMBED_MODEL)
    return index, model, docs, metas

def retrieve(query, index, model, docs, metas, k=2):
    q_emb = model.encode([query], convert_to_numpy=True)
    D, I = index.search(q_emb, k)
    results = [(docs[i], metas[i], D[0][j]) for j, i in enumerate(I[0])]
    return results

def ollama_chat(prompt, history):
    MAX_HISTORY = 4  
    messages = history[-MAX_HISTORY:] + [{"role": "user", "content": prompt}]
    response = ollama.chat(model="llama3", messages=messages)
    return response["message"]["content"]

def chat():
    if os.path.exists(INDEX_FILE):
        print("🔹 Loading existing FAISS index...")
        index, model, docs, metas = load_faiss()
    else:
        print("🔹 Building FAISS index...")
        docs, metas = load_docs()
        index, model = build_faiss(docs, metas)

    history = []
    while True:
        query = input("\nYou: ")
        if query.lower() in ["exit", "quit"]:
            break

        results = retrieve(query, index, model, docs, metas, k=3)
        context = "\n\n".join([f"[Source: {r[1]['source']}] {r[0]}" for r in results])

        prompt = f"Answer the question using the context below.\n\nContext:\n{context}\n\nQuestion: {query}"
        answer = ollama_chat(prompt, history)

        print(f"\n Bot: {answer}")
        print("\n📚 Sources:")
        for r in results:
            print(f" - {r[1]['source']} (score {r[2]:.2f})")

        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    chat()
