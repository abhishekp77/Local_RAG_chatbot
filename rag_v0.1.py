import os
import pickle
import faiss
import re
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import ollama

# -----------------------------
# CONFIG
# -----------------------------
DOCS_PATH = "data/docs"
INDEX_FILE = "faiss_index.bin"
TEXTS_FILE = "texts.pkl"
METAS_FILE = "metas.pkl"
EMBED_MODEL = "all-MiniLM-L6-v2"   # fast + lightweight

# -----------------------------
# STEP 1: Parse Bhagavad Gita PDFs (chapter + verse)
# -----------------------------
def parse_gita_pdf(pdf_path, filename):
    reader = PdfReader(pdf_path)
    docs, metas = [], []
    current_chapter = None
    current_verse = None
    buffer = []

    for page in reader.pages:
        text = page.extract_text()
        if not text:
            continue

        for line in text.splitlines():
            line = line.strip()

            # Detect chapter
            chap_match = re.match(r"CHAPTER\s+(\d+)", line, re.IGNORECASE)
            if chap_match:
                current_chapter = int(chap_match.group(1))
                continue

            # Detect verse
            verse_match = re.match(r"VERSE\s+(\d+)", line, re.IGNORECASE)
            if verse_match:
                # save previous verse if buffer has content
                if buffer and current_verse is not None:
                    docs.append(" ".join(buffer).strip())
                    metas.append({
                        "source": filename,
                        "chapter": current_chapter,
                        "verse": current_verse
                    })
                    buffer = []
                current_verse = int(verse_match.group(1))
                continue

            # Otherwise, collect verse content
            if current_chapter and current_verse:
                buffer.append(line)

    # Save last verse
    if buffer and current_verse is not None:
        docs.append(" ".join(buffer).strip())
        metas.append({
            "source": filename,
            "chapter": current_chapter,
            "verse": current_verse
        })

    return docs, metas

# -----------------------------
# STEP 2: Load & Chunk Documents (general fallback for txt/non-Gita)
# -----------------------------
def load_docs(path=DOCS_PATH):
    docs, metas = [], []
    for root, _, files in os.walk(path):
        for f in files:
            file_path = os.path.join(root, f)
            if f.endswith(".pdf"):
                # assume Gita-like format
                gita_docs, gita_metas = parse_gita_pdf(file_path, f)
                docs.extend(gita_docs)
                metas.extend(gita_metas)

            elif f.endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as infile:
                    text = infile.read()
                chunks = [text[i:i+500] for i in range(0, len(text), 500)]
                docs.extend(chunks)
                metas.extend([{"source": f}] * len(chunks))
            else:
                continue
    return docs, metas

# -----------------------------
# STEP 3: Build or Load FAISS
# -----------------------------
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

# -----------------------------
# STEP 4: RAG Query
# -----------------------------
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

# -----------------------------
# STEP 5: CLI Chat Loop
# -----------------------------
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

        # retrieve relevant docs
        results = retrieve(query, index, model, docs, metas, k=3)

        context_lines = []
        for r in results:
            meta = r[1]
            if "chapter" in meta and "verse" in meta:
                context_lines.append(f"[Ch {meta['chapter']}, Verse {meta['verse']}] {r[0]}")
            else:
                context_lines.append(f"[Source: {meta['source']}] {r[0]}")

        context = "\n\n".join(context_lines)

        prompt = f"Answer the question using the context below.\n\nContext:\n{context}\n\nQuestion: {query}"
        answer = ollama_chat(prompt, history)

        print(f"\n🤖 Bot: {answer}")
        print("\n📚 Sources:")
        for r in results:
            meta = r[1]
            if "chapter" in meta and "verse" in meta:
                print(f" - {meta['source']} (Ch {meta['chapter']}, Verse {meta['verse']}, score {r[2]:.2f})")
            else:
                print(f" - {meta['source']} (score {r[2]:.2f})")

        # keep short-term memory
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    chat()
