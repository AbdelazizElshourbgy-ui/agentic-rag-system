"""Optional standalone FastAPI service. It is independent from Streamlit."""
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from rag_core import RAGEngine

ROOT = Path(__file__).resolve().parent
CORPUS_PATH = ROOT / "data" / "corpus.csv"

app = FastAPI(
    title="Agentic RAG API",
    summary="Question answering over a document knowledge base",
    description=(
        "A standalone REST API for retrieving relevant document chunks, "
        "preparing grounded answers, and returning source chunk IDs. "
        "The Streamlit interface is a separate application and is not required."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)
engine = None


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        examples=["What are the main concepts in the uploaded documents?"]
    )
    top_k: int = Field(default=5, ge=1, le=20)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home():
    return """
    <!doctype html>
    <html><head><title>Agentic RAG API</title>
    <style>
    body{font-family:Arial,sans-serif;background:#f3f6fb;color:#172b4d;
    display:grid;place-items:center;min-height:90vh}
    main{background:white;padding:42px;border-radius:18px;max-width:650px;
    box-shadow:0 12px 35px #10243e18}
    h1{color:#153e63}a{display:inline-block;margin:10px 12px 0 0;
    padding:12px 18px;background:#173f65;color:white;text-decoration:none;
    border-radius:8px}
    </style></head><body><main>
    <h1>Agentic RAG API</h1>
    <p>Independent API service for document retrieval and grounded question answering.</p>
    <a href="/docs">Swagger UI</a><a href="/redoc">ReDoc</a>
    </main></body></html>
    """


@app.on_event("startup")
def load_knowledge_base():
    global engine
    if CORPUS_PATH.exists():
        engine = RAGEngine.from_csv(CORPUS_PATH)


@app.get("/health", tags=["System"])
def health():
    return {
        "status": "ok",
        "knowledge_base_loaded": engine is not None
    }


@app.post("/ask", tags=["RAG"])
def ask(request: AskRequest):
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="Knowledge base is missing. Create data/corpus.csv first."
        )
    try:
        result = engine.ask(request.question, request.top_k)
        return {
            "question": result["question"],
            "answer": result["answer"],
            "plan": result["plan"],
            "cited_chunk_ids": result["cited_chunk_ids"]
        }
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
