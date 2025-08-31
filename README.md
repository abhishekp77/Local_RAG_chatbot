
# 📖 Local Gita Chatbot  

An AI-powered chatbot that allows you to interact with the **Bhagavad Gita** in natural language.  
It works completely **offline** using local embeddings and a local LLM (LLaMA 2).  

You can run it in two modes:  
- 🐍 **Python Script (CLI/Terminal)**  
- 🌐 **Streamlit Web App (GUI)**  

---

## 🚀 Features
- 🔹 **Offline AI chatbot** — no API key required.  
- 🔹 **Fast response** using FAISS vector search.  
- 🔹 **Two modes**: script-based & web-based.  
- 🔹 **Transparent answers** — shows relevant Gita verses.  
- 🔹 **Prebuilt FAISS index support** for faster startup.  

---

## 🛠️ Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/local-gita-chatbot.git
cd local-gita-chatbot
```

### 2️⃣ Create a Conda Environment
```bash
conda create -n localbot python=3.10 -y
conda activate localbot
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 📂 Project Structure
```
LocalBot/
│── model/                      # LLaMA 2 model files (not included in repo)
│── data/                       # Bhagavad Gita PDF/Text
│── faiss_index.bin             # Prebuilt FAISS index
│── metas.pkl                   # Metadata for verses
│── text.pkl                    # Verse texts
│── gita_chatbot.py             # Core Python script (CLI chatbot)
│── Gita_chatbot_gui.py         # Streamlit web app
│── requirements.txt            # Python dependencies
│── README.md                   # Project documentation
```

---

## 💻 Usage

### 🐍 Run CLI Version
```bash
python gita_chatbot.py
```
- Type your question and get instant answers.  
- Example:  
  ```
  Q: What does the Gita say about karma?
  A: "You have a right to perform your prescribed duties, but you are not entitled to the fruits of actions." (2.47)
  ```

---

### 🌐 Run Streamlit Web App
```bash
streamlit run Gita_chatbot_gui.py
```
- Opens a **web-based chatbot UI**.  
- Type your query and see answers with relevant verses.  
- Example UI:  

*(screenshot placeholder)*  

---

## 📦 Requirements
- Python 3.9+  
- Conda (recommended)  
- Libraries: `streamlit`, `faiss-cpu`, `sentence-transformers`, `pickle`, `transformers`, `torch`  

Install them via:
```bash
pip install -r requirements.txt
```

---

## 🔮 Future Scope
- ✅ Add multi-language support (Hindi, Sanskrit, etc.).  
- ✅ Deploy online with Streamlit Cloud.  
- ✅ Extend chatbot to other scriptures or texts.  
- ✅ Improve response generation with fine-tuned models.  

---

## ⚠️ Note
- The **LLaMA 2 model file** is **not included** in the repo (due to size).  
- You need to place your downloaded `.gguf` model inside the `model/` folder.  
