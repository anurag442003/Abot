from pathlib import Path
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages
from langchain_core.documents import Document
from ollama import Client

from dotenv import load_dotenv
from openai import OpenAI




load_dotenv(override=True)

ollama_client= Client()
MODEL="qwen3:4b"
DB_NAME = str(Path(__file__).parent.parent / "vector_db")

embeddings = HuggingFaceEmbeddings(model_name="Qwen/Qwen3-Embedding-0.6B", model_kwargs={"trust_remote_code": True})
RETRIEVAL_K = 10

SYSTEM_PROMPT = """
You are a friendly and knowledgeable personal assistant on Anurag's portfolio website.
Visitors come here to learn about Anurag — his background, skills, projects, and experience.
Your job is to help them quickly find what they're looking for in a conversational way.
Answer questions about Anurag's skills, projects, work experience, education, and anything else on the portfolio.
Keep answers concise and engaging. If you don't know something, say so honestly.
Context:
{context}
"""

vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
retriever = vectorstore.as_retriever()
# llm = ChatGoogleGenerativeAI(
#     model=GEMINI_MODEL,
#     api_key=os.getenv("GEMINI_API_KEY"),
# )

def fetch_context(question: str) -> list[Document]:
    return retriever.invoke(question, k=RETRIEVAL_K)


def combined_question(question: str, history: list[dict] = []) -> str:
    prior = "\n".join(m["content"] for m in history if m["role"] == "user")
    return prior + "\n" + question


def answer_question(question: str, history: list[dict] = []) -> tuple[str, list[Document]]:
    combined = combined_question(question, history)
    docs = fetch_context(combined)
    context = "\n\n".join(doc.page_content for doc in docs)
    system_prompt = SYSTEM_PROMPT.format(context=context)
    messages = [{"role": "system", "content": system_prompt}]
    for m in history:
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": question})
    response = ollama_client.chat(model="qwen3:4b", messages=messages)
    content = response.message.content
    if "</think>" in content:
        content = content.split("</think>")[-1].strip()
    return content, docs
