# 🎓 RSET Academic Assistant
### Intelligent Document Q&A System using LangChain + RAG

A conversational AI assistant that answers questions about **Rajagiri School of Engineering & Technology (RSET)** college regulations and academic handbook — built using Retrieval-Augmented Generation (RAG).

---

## 🧠 How It Works

```
Your Question
      ↓
Search vectorstore (FAISS) for top 5 relevant chunks
      ↓
Send chunks + question to Groq LLaMA 3
      ↓
Get accurate, cited answer ✅
```

**RAG = Retrieval + Augmented + Generation**
- The AI only answers from your official college documents
- Every answer comes with a page citation
- Supports follow-up questions in the same session

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Framework | LangChain |
| LLM | Groq LLaMA 3.1 (Free) |
| Embeddings | all-MiniLM-L6-v2 (Local, Free) |
| Vector Database | FAISS |
| PDF Parsing | PyPDF |
| Web Interface | Streamlit |

---

## 📁 Project Structure

```
rset-academic-assistant/
│
├── docs/                    ← Put your PDF documents here
│   ├── AHB_25-26.pdf
│   └── B_Tech_2023_regulations.pdf
│
├── vectorstore/             ← Auto-created by ingest.py (not on GitHub)
│
├── ingest.py                ← Step 1: Process PDFs → build vector database
├── rag_chain.py             ← Step 2: RAG pipeline (retrieval + generation)
├── app.py                   ← Step 3: Streamlit web interface
│
├── requirements.txt         ← All Python dependencies
├── .env.example             ← API key template
├── .gitignore               ← Keeps secrets and large files off GitHub
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/rset-academic-assistant.git
cd rset-academic-assistant
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Get a free Groq API key
- Go to [console.groq.com](https://console.groq.com)
- Sign up and create an API key (free, no credit card)

### 4. Set up your API key
```bash
# Create a .env file
echo GROQ_API_KEY=your-key-here > .env
```

### 5. Add your PDF documents
Place your college PDF files inside the `docs/` folder.

### 6. Build the vector database (run once)
```bash
python ingest.py
```

### 7. Launch the app
```bash
python -m streamlit run app.py
```

Open your browser at `http://localhost:8501` 🎉

---

## 💬 Example Questions

- *"What is the minimum attendance required to write exams?"*
- *"How is CGPA calculated?"*
- *"What happens if I get an FE grade?"*
- *"Can I apply for B.Tech with Honours? What's the eligibility?"*
- *"What are the grace marks rules for sports?"*
- *"How many credits do I need to graduate?"*

---

## 📊 Documents Supported
- **Academic Handbook 2025-26** — college rules, facilities, student services
- **B.Tech Regulations 2023** — grading, attendance, exams, promotions

> You can add any PDF to the `docs/` folder and re-run `ingest.py` to include it.

---

## 👩‍💻 Built By
Third Year B.Tech Student — Rajagiri School of Engineering & Technology

*This project was built as part of an academic mini-project on Intelligent Document-Based Question Answering using LangChain and RAG.*
