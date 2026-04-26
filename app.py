from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from implementation.answer_gemini import answer_question

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-portfolio.com", "http://localhost:5173"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str
    history: list[dict] = []

@app.post("/chat")
def chat(req: ChatRequest):
    answer, docs = answer_question(req.question, req.history)
    context = [
        {"content": d.page_content, "source": d.metadata.get("source", "")}
        for d in docs
    ]
    return {"answer": answer, "context": context}