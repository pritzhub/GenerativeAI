# src/ingest.py
# pip install openai python-dotenv pypdf python-docx pandas numpy pyarrow PyYAML

import argparse
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from pypdf import PdfReader
import docx  # from python-docx

from src.settings import get_docs_dir, get_data_dir, get_setting
from src.models.factory import get_embeddings_client


DOCS_DIR = get_docs_dir()
DATA_DIR = get_data_dir()


def load_txt(path: Path) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    texts = []
    for page in reader.pages:
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""
        texts.append(page_text)
    return "\n".join(texts)


def load_docx(path: Path) -> str:
    document = docx.Document(str(path))
    return "\n".join(p.text for p in document.paragraphs)


def load_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return load_txt(path)
    elif suffix == ".pdf":
        return load_pdf(path)
    elif suffix == ".docx":
        return load_docx(path)
    else:
        print(f"Skipping unsupported file type: {path.name}")
        return ""


def chunk_text(text: str) -> List[str]:
    max_chars = int(get_setting("rag.chunk_size", 1000))
    overlap = int(get_setting("rag.chunk_overlap", 200))

    chunks = []
    start = 0
    text = text.replace("\r", " ").replace("\n", " ")
    n = len(text)

    while start < n:
        end = start + max_chars
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start = end - overlap

    return chunks


def embed_texts(texts: List[str]) -> List[List[float]]:
    emb_client = get_embeddings_client()
    embeddings: List[List[float]] = []

    batch_size = 64
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        embeddings.extend(emb_client.embed(batch))
    return embeddings


def build_corpus() -> pd.DataFrame:
    records: List[Dict] = []

    if not DOCS_DIR.exists():
        raise FileNotFoundError(f"Docs folder not found at: {DOCS_DIR}")

    files = list(DOCS_DIR.glob("**/*"))
    if not files:
        raise RuntimeError(f"No files found under docs/ at {DOCS_DIR}")

    print(f"Found {len(files)} files under {DOCS_DIR}")

    for path in files:
        if not path.is_file():
            continue

        print(f"Loading {path.name}...")
        text = load_document(path)
        if not text.strip():
            print(f"  Skipping empty or unreadable file: {path.name}")
            continue

        chunks = chunk_text(text)
        print(f"  Created {len(chunks)} chunks.")

        for idx, ch in enumerate(chunks):
            records.append(
                {
                    "source": str(path.relative_to(DOCS_DIR)),
                    "chunk_id": idx,
                    "text": ch,
                }
            )

    df = pd.DataFrame(records)
    print(f"\nTotal chunks: {df.shape[0]}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Ingest docs and build RAG index.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild index even if existing files are present.",
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)

    index_path = DATA_DIR / "rag_index.parquet"
    emb_path = DATA_DIR / "rag_embeddings.npy"

    if index_path.exists() and emb_path.exists() and not args.force:
        print(f"Index already exists at:\n  {index_path}\n  {emb_path}")
        print("Use --force to rebuild the index.")
        return

    print(f"Docs dir: {DOCS_DIR}")
    print(f"Data dir: {DATA_DIR}")

    df = build_corpus()

    print("\nGenerating embeddings...")
    embeddings = embed_texts(df["text"].tolist())
    print("Done generating embeddings.")

    emb_array = np.array(embeddings, dtype="float32")
    print(f"Embeddings shape: {emb_array.shape}")

    df.to_parquet(index_path, index=False)
    np.save(emb_path, emb_array)

    print(f"\nSaved chunk metadata to: {index_path}")
    print(f"Saved embeddings to:     {emb_path}")
    print("\nIngest complete.")


if __name__ == "__main__":
    main()
