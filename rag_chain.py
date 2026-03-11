"""
rag_chain.py - Q&A chain using Groq (FREE, fast, no daily limits)
Uses modern LangChain LCEL (LangChain Expression Language) instead of
the deprecated ConversationalRetrievalChain.
"""

import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

VECTORSTORE     = "vectorstore"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K_CHUNKS    = 5
GROQ_MODEL      = "llama-3.1-8b-instant"
TEMPERATURE     = 0.1


def load_vectorstore():
    if not os.path.exists(VECTORSTORE):
        raise FileNotFoundError(
            f"Vector database not found at '{VECTORSTORE}/'.\n"
            "Please run:  python ingest.py"
        )
    embeddings = HuggingFaceEmbeddings(
        model_name    = EMBEDDING_MODEL,
        model_kwargs  = {"device": "cpu"},
        encode_kwargs = {"normalize_embeddings": True}
    )
    return FAISS.load_local(
        VECTORSTORE,
        embeddings,
        allow_dangerous_deserialization=True
    )


def format_docs(docs):
    """Combine retrieved chunks into one context string."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def format_sources(source_docs):
    seen, sources = set(), []
    for doc in source_docs:
        file_name = doc.metadata.get("source_file", "Unknown")
        page_num  = doc.metadata.get("page", "?")
        key = (file_name, page_num)
        if key not in seen:
            seen.add(key)
            display = file_name.replace("_", " ").replace(".pdf", "")
            sources.append(f"📄 {display} — Page {int(page_num) + 1}")
    return "\n".join(sources) if sources else "Source not identified"


def build_rag_chain():
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError(
            "GROQ_API_KEY not found!\n"
            "1. Go to https://console.groq.com\n"
            "2. Sign up free and create an API key\n"
            "3. Add to your .env file: GROQ_API_KEY=gsk_your-key-here"
        )

    vectordb  = load_vectorstore()
    retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K_CHUNKS}
    )

    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=TEMPERATURE,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful academic assistant for Rajagiri School of \
Engineering & Technology (RSET). Answer student questions based ONLY on the \
provided context from official college documents.

Rules:
- Answer ONLY from the context below. Do not make things up.
- If the answer is not in the context, say: "I couldn't find this in the official \
documents. Please check with your Class Teacher or the Academic Office."
- Be friendly and clear like a senior student explaining things.
- Be precise with rules about attendance, marks and grades.
- Use bullet points when listing multiple rules.

Context from official documents:
{context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    # Modern LCEL chain — no deprecated classes
    chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["question"]))
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return retriever, chain


def ask(retriever, chain, question: str, chat_history: list) -> dict:
    """
    Ask a question and get an answer with sources.
    chat_history: list of (human_msg, ai_msg) tuples
    """
    # Convert history tuples to LangChain message objects
    messages = []
    for human, ai in chat_history:
        messages.append(HumanMessage(content=human))
        messages.append(AIMessage(content=ai))

    # Get relevant source docs for citation
    source_docs = retriever.invoke(question)

    # Get answer
    answer = chain.invoke({
        "question"    : question,
        "chat_history": messages
    })

    return {
        "answer"          : answer,
        "source_documents": source_docs
    }


if __name__ == "__main__":
    print("Loading RAG chain (Groq + local embeddings)...")
    retriever, chain = build_rag_chain()
    print("Ready! Type your question (or quit to exit)\n")
    history = []
    while True:
        question = input("You: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue
        result = ask(retriever, chain, question, history)
        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nSources:\n{format_sources(result['source_documents'])}")
        print("\n" + "-"*50 + "\n")
        history.append((question, result["answer"]))
