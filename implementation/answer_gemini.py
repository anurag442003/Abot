from pathlib import Path
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv(override=True)

MODEL = "gemini-2.5-flash"
DB_NAME = str(Path(__file__).parent.parent / "vector_db")

embeddings = HuggingFaceEmbeddings(model_name="Qwen/Qwen3-Embedding-0.6B", model_kwargs={"trust_remote_code": True})
RETRIEVAL_K = 10

llm = ChatGoogleGenerativeAI(model=MODEL)

SYSTEM_PROMPT = """
You are an AI assistant for Anurag's portfolio.

Your job is to give clean, structured, and easy-to-read answers.

STRICT RULES:
- Never output raw markdown like ** or ##
- Never dump raw context
- Always summarize information
- Keep answers concise and structured

FORMATTING RULES:

If the user asks to list projects:
Return in this format:

Projects:

1. Project Name
   - Description: short 1-line summary
   - Tech: comma-separated tools
   - Purpose: what problem it solves

2. Project Name
   - Description: ...
   - Tech: ...
   - Purpose: ...

If the user asks about ONE project:
Return:

Project: <name>
Description: ...
Tech: ...
Key Features:
- ...
- ...

GENERAL RULES:
- Use bullet points instead of paragraphs
- Keep spacing clean
- No long text blocks

Context:
{context}
"""

vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
retriever = vectorstore.as_retriever()


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

    messages = [SystemMessage(content=system_prompt)]
    for m in history:
        if m["role"] == "user":
            messages.append(HumanMessage(content=m["content"]))
        else:
            messages.append(AIMessage(content=m["content"]))
    messages.append(HumanMessage(content=question))

    response = llm.invoke(messages)
    return response.content, docs