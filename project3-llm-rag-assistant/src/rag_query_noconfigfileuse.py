# src/rag_query.py
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from src.config import get_openai_client  # same helper as ingest.py


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
INDEX_PATH = DATA_DIR / "rag_index.parquet"
EMB_PATH = DATA_DIR / "rag_embeddings.npy"

EMBEDDING_MODEL = "text-embedding-3-small"
ANSWER_MODEL = "gpt-4.1-mini"  # for summarization/QA


def load_index() -> Tuple[pd.DataFrame, np.ndarray]:
    if not INDEX_PATH.exists() or not EMB_PATH.exists():
        raise FileNotFoundError(
            f"Index files not found.\n"
            f"Expected:\n  {INDEX_PATH}\n  {EMB_PATH}\n"
            f"Run `python -m src.ingest` first."
        )

    df = pd.read_parquet(INDEX_PATH)
    emb_array = np.load(EMB_PATH)
    print(f"Loaded index with {df.shape[0]} chunks and embeddings shape {emb_array.shape}")
    return df, emb_array


def embed_query(query: str) -> np.ndarray:
    client = get_openai_client()
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[query],
    )
    vec = np.array(resp.data[0].embedding, dtype="float32")
    return vec


def cosine_similarity_matrix(query_vec: np.ndarray, doc_embs: np.ndarray) -> np.ndarray:
    # Normalize to unit length then dot product ≈ cosine similarity.[web:170][web:171][web:177]
    q = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    d = doc_embs / (np.linalg.norm(doc_embs, axis=1, keepdims=True) + 1e-8)
    sims = d @ q  # shape (num_docs,)
    return sims


def retrieve_top_k(df: pd.DataFrame, emb_array: np.ndarray, query: str, k: int = 5):
    q_vec = embed_query(query)
    sims = cosine_similarity_matrix(q_vec, emb_array)
    top_idx = np.argsort(-sims)[:k]

    top_chunks = df.iloc[top_idx].copy()
    top_chunks["similarity"] = sims[top_idx]
    return top_chunks


def build_context(chunks: pd.DataFrame) -> str:
    """Format retrieved chunks into a context block for the model."""
    parts: List[str] = []
    for _, row in chunks.iterrows():
        parts.append(
            f"Source: {row['source']} (chunk {row['chunk_id']}, sim={row['similarity']:.3f})\n"
            f"{row['text']}\n"
            "-----"
        )
    return "\n".join(parts)


def answer_query(query: str, context: str) -> str:
    client = get_openai_client()

    system_msg = (
        "You are an assistant that answers questions about energy-sector RFP documents. "
        "Use ONLY the provided context to answer. If something is not in the context, say you don't know."
    )

    user_msg = (
        f"You are analyzing one or more energy-sector RFP documents.\n\n"
        f"Question:\n{query}\n\n"
        f"Context from RFP documents:\n{context}\n\n"
        "Using ONLY the context above, do the following when relevant:\n"
        "- List technical, functional, and non-functional requirements.\n"
        "- Highlight potential implementation/delivery challenges and risks.\n"
        "- Describe business impact and expected outcomes.\n"
        "- Note any financial constraints (budget, penalties, payment terms).\n"
        "- Infer expected business KPIs or ROI if mentioned.\n"
        "- If asked, outline a structured RFP response (sections and key points).\n"
        "If the context does not contain enough information for any point, explicitly say so.\n"
    )

    resp = client.chat.completions.create(
        model=ANSWER_MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,  # Keep temperature low (0.1–0.3) to make answers more precise and less speculative.
    )

    return resp.choices[0].message.content


def interactive_loop():
    df, emb_array = load_index()

    print("\nRAG assistant ready. Ask questions about your energy RFPs.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        query = input("Your question> ").strip()
        if query.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break
        if not query:
            continue

        print("\nRetrieving relevant chunks...")
        chunks = retrieve_top_k(df, emb_array, query, k=5)
        context = build_context(chunks)

        print("Generating answer...\n")
        answer = answer_query(query, context)

        print("=== Answer ===")
        print(answer)
        print("\n--- Retrieved sources (top 5) ---")
        print(chunks[["source", "chunk_id", "similarity"]])
        print("\n")


def main():
    interactive_loop()


if __name__ == "__main__":
    main()
