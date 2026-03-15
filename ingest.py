"""
ingest.py - Step 1: Load PDFs and build the vector database

Uses a FREE LOCAL embedding model that runs on your computer.
No API calls needed. No rate limits. Works offline.

Run it with:
    python ingest.py
"""

import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

DOCS_FOLDER     = "docs"
VECTORSTORE     = "vectorstore"
CHUNK_SIZE      = 800
CHUNK_OVERLAP   = 150
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def load_all_pdfs(folder):
    pdf_paths = list(Path(folder).glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDF files found in '{folder}/' folder.\n"
            "Please put your PDFs there and run this script again."
        )
    all_docs = []
    for pdf_path in pdf_paths:
        print(f"  Loading: {pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        pages  = loader.load()
        for page in pages:
            page.metadata["source_file"] = pdf_path.name
        all_docs.extend(pages)
        print(f"     -> {len(pages)} pages loaded")
    return all_docs


def split_into_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size    = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP,
        separators    = ["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def build_vectorstore(chunks):
    print(f"\n Loading local embedding model: {EMBEDDING_MODEL}")
    print("  First run downloads ~90MB. After that it is instant.\n")

    embeddings = HuggingFaceEmbeddings(
        model_name   = EMBEDDING_MODEL,
        model_kwargs = {"device": "cpu"},
        encode_kwargs= {"normalize_embeddings": True}
    )

    print("Model loaded! Embedding all chunks now (30-60 seconds)...")
    vectordb = FAISS.from_documents(chunks, embeddings)
    vectordb.save_local(VECTORSTORE)
    print(f"\nDone! Vector database saved to '{VECTORSTORE}/' folder.")


def main():
    print("=" * 55)
    print("  RSET Document Ingestion - Building Knowledge Base")
    print("=" * 55)

    print(f"\nLoading PDFs from '{DOCS_FOLDER}/' ...")
    documents = load_all_pdfs(DOCS_FOLDER)
    print(f"\n   Total pages loaded: {len(documents)}")

    print(f"\nSplitting into chunks...")
    chunks = split_into_chunks(documents)
    print(f"   Total chunks created: {len(chunks)}")

    build_vectorstore(chunks)

    print("\nDone! Now launch the app with:")
    print("   streamlit run app.py\n")


if __name__ == "__main__":
    main()
