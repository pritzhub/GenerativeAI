# pip install fastapi uvicorn

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List

import sys
from pathlib import Path

CORE_ROOT = Path(__file__).resolve().parents[2] / "project3-llm-rag-assistant"
sys.path.append(str(CORE_ROOT))

# Import from the installed core package
from src.rag_query import load_index, retrieve_top_k, build_context, answer_query
from src.settings import get_active_profile, get_setting

app = FastAPI()


class RAGQueryRequest(BaseModel):
    question: str
    profile: Optional[str] = None  # not used yet, uses core config
    top_k: Optional[int] = None


class RAGChunk(BaseModel):
    source: str
    chunk_id: int
    similarity: float


class RAGQueryResponse(BaseModel):
    answer: str
    profile: str
    used_top_k: int
    chunks: List[RAGChunk]


@app.post("/rag/query", response_model=RAGQueryResponse)
async def rag_query(req: RAGQueryRequest):
    profile = get_active_profile()

    df_index, emb_array = load_index()
    used_top_k = req.top_k if req.top_k is not None else int(get_setting("rag.top_k", 5))

    chunks = retrieve_top_k(df_index, emb_array, req.question, k=used_top_k)
    context = build_context(chunks)
    answer = answer_query(req.question, context)

    chunk_models = [
        RAGChunk(
            source=row["source"],
            chunk_id=int(row["chunk_id"]),
            similarity=float(row["similarity"]),
        )
        for _, row in chunks.iterrows()
    ]

    return RAGQueryResponse(
        answer=answer,
        profile=profile,
        used_top_k=used_top_k,
        chunks=chunk_models,
    )
