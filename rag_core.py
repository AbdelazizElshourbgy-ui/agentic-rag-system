"""Shared retrieval and answer pipeline used independently by Streamlit and FastAPI."""
from pathlib import Path
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

REQUIRED_COLUMNS = {"chunk_id", "chunk_text"}


def load_corpus(csv_path):
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Knowledge base not found: {path}. Upload documents in Streamlit first "
            "or place corpus.csv in the data folder."
        )
    frame = pd.read_csv(path).fillna("")
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Corpus is missing required columns: {sorted(missing)}")
    for column in ("title", "doc_id", "doc_type"):
        if column not in frame.columns:
            frame[column] = ""
    frame["chunk_id"] = frame["chunk_id"].astype(str)
    frame["chunk_text"] = frame["chunk_text"].astype(str)
    frame["search_text"] = frame["title"].astype(str) + " " + frame["chunk_text"]
    return frame


class RAGEngine:
    def __init__(self, corpus):
        if corpus.empty:
            raise ValueError("The knowledge base is empty.")
        self.corpus = corpus.copy().reset_index(drop=True)
        if "search_text" not in self.corpus.columns:
            self.corpus["search_text"] = (
                self.corpus.get("title", "").astype(str) + " "
                + self.corpus["chunk_text"].astype(str)
            )
        self.vectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), max_features=100000
        )
        self.matrix = self.vectorizer.fit_transform(self.corpus["search_text"])

    @classmethod
    def from_csv(cls, csv_path):
        return cls(load_corpus(csv_path))

    def retrieve(self, question, top_k=5):
        query = self.vectorizer.transform([question])
        scores = cosine_similarity(query, self.matrix).ravel()
        indexes = scores.argsort()[::-1][:max(1, min(top_k, len(scores)))]
        results = self.corpus.iloc[indexes].copy()
        results["score"] = scores[indexes]
        return results.reset_index(drop=True)

    @staticmethod
    def make_plan(question):
        q = question.lower()
        if any(word in q for word in ("summarize", "summary", "overview")):
            action = "Retrieve relevant passages and summarize their content."
        elif any(word in q for word in ("compare", "difference", "versus")):
            action = "Retrieve evidence about the compared topics."
        elif any(word in q for word in ("calculate", "compute", "average", "percentage")):
            action = "Retrieve relevant evidence and inspect numerical information."
        else:
            action = "Retrieve the most relevant passages for the question."
        return [
            "Understand the user question.",
            action,
            "Check retrieval results and prepare a grounded response.",
            "Return the supporting source chunk IDs."
        ]

    @staticmethod
    def summarize(text, max_sentences=5):
        sentences = re.split(r"(?<=[.!?])\s+", str(text).strip())
        useful = [item.strip() for item in sentences if len(item.strip()) > 20]
        return " ".join(useful[:max_sentences])

    def ask(self, question, top_k=5):
        question = str(question).strip()
        if not question:
            raise ValueError("Question cannot be empty.")

        results = self.retrieve(question, top_k)
        context = " ".join(results["chunk_text"].astype(str).tolist())
        answer = self.summarize(context)

        if not answer:
            answer = "I could not find enough information in the knowledge base."

        return {
            "question": question,
            "answer": answer,
            "plan": self.make_plan(question),
            "cited_chunk_ids": results["chunk_id"].astype(str).tolist(),
            "retrieved_chunks": results
        }


def chunk_text(text, chunk_words=180, overlap=30):
    words = text.split()
    if not words:
        return []
    step = max(1, chunk_words - overlap)
    return [" ".join(words[i:i + chunk_words]) for i in range(0, len(words), step)]


def build_corpus_from_files(file_paths, output_csv):
    """Extract supported files and save a simple chunk-based corpus."""
    rows = []
    for file_path in file_paths:
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix in (".txt", ".md"):
            content = path.read_text(encoding="utf-8", errors="ignore")
        elif suffix == ".pdf":
            from pypdf import PdfReader
            content = "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
        elif suffix == ".docx":
            from docx import Document
            doc = Document(str(path))
            content = "\n".join(p.text for p in doc.paragraphs)
        else:
            continue

        content = re.sub(r"\s+", " ", content).strip()
        for index, chunk in enumerate(chunk_text(content)):
            rows.append({
                "chunk_id": f"{path.stem}_chunk_{index}",
                "doc_id": path.stem,
                "title": path.stem,
                "doc_type": suffix.lstrip("."),
                "chunk_text": chunk
            })

    frame = pd.DataFrame(rows, columns=[
        "chunk_id", "doc_id", "title", "doc_type", "chunk_text"
    ])
    if frame.empty:
        raise ValueError("No readable PDF, DOCX, TXT, or Markdown content was found.")
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    return frame
