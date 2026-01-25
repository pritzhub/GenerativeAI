# rag-streamlit-ui/app.py
# pip install streamlit

import sys
from pathlib import Path

import streamlit as st

# Adjust this path so it points to your core project root
CORE_ROOT = Path(__file__).resolve().parents[1] / "project3-llm-rag-assistant"
sys.path.append(str(CORE_ROOT))

from src.settings import (
    get_active_profile,
    get_setting,
    get_docs_dir,
    get_data_dir,
    get_prompt_template,
)
from src.rag_query import load_index, retrieve_top_k, build_context, answer_query
from src.ingest import main as ingest_main  # reuse existing ingest CLI entry point


st.set_page_config(page_title="RAG Assistant", layout="wide")

st.title("Profile-aware RAG Assistant (Streamlit UI)")


# --- Sidebar: profile & config view (read-only for now) ---
st.sidebar.header("Profile & Config")

current_profile = get_active_profile()
st.sidebar.markdown(f"**Active profile (from config.yml):** `{current_profile}`")

rag_cfg = get_setting("rag", {})
llm_cfg = get_setting("llm", {})
emb_cfg = get_setting("embeddings", {})

st.sidebar.subheader("RAG settings")
st.sidebar.json(rag_cfg)

st.sidebar.subheader("LLM settings")
st.sidebar.json(llm_cfg)

st.sidebar.subheader("Embeddings settings")
st.sidebar.json(emb_cfg)


# --- Tabs ---
tab_upload, tab_qa = st.tabs(["📁 Upload & Ingest", "❓ Q&A"])


# --- Tab 1: Upload docs & ingest ---
with tab_upload:
    st.header("Upload documents and build index for active profile")

    docs_dir = get_docs_dir()
    data_dir = get_data_dir()
    st.write(f"**Docs directory (active profile):** `{docs_dir}`")
    st.write(f"**Data directory (active profile):** `{data_dir}`")

    uploaded_files = st.file_uploader(
        "Upload PDF/DOCX/TXT files for this profile",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.write("Uploaded files:")
        for f in uploaded_files:
            st.write(f"- {f.name}")

        if st.button("Save files to docs directory"):
            docs_dir.mkdir(parents=True, exist_ok=True)
            for f in uploaded_files:
                dest = docs_dir / f.name
                with open(dest, "wb") as out:
                    out.write(f.getbuffer())
            st.success("Files saved. You can now (re)run ingestion for this profile.")

    if st.button("Run ingestion for active profile (force rebuild)"):
        with st.spinner("Running ingestion..."):
            # Equivalent to `python -m src.ingest --force`:
            # quick hack: simulate args; or you can expose a function without argparse.
            import argparse

            # Call ingest_main via monkey-patched args
            # Easiest: temporarily patch sys.argv for this call
            old_argv = sys.argv
            sys.argv = ["ingest.py", "--force"]
            try:
                ingest_main()
            finally:
                sys.argv = old_argv
        st.success("Ingestion complete.")


# --- Tab 2: Q&A ---
with tab_qa:
    st.header("Ask questions against indexed documents")

    st.write(f"Using profile: `{current_profile}`")

    # Try to load index; show friendly error if missing
    index_loaded = False
    df_index = None
    emb_array = None
    try:
        df_index, emb_array = load_index()
        index_loaded = True
        st.success(f"Loaded index with {df_index.shape[0]} chunks.")
    except FileNotFoundError as e:
        st.error(
            "Index not found for this profile. "
            "Go to 'Upload & Ingest' tab, upload docs, and run ingestion."
        )
        st.stop()

    question = st.text_area(
        "Enter your question",
        placeholder="e.g., What are the main technical requirements in these RFPs?",
        height=100,
    )

    top_k_override = st.number_input(
        "Top K chunks (optional override)",
        min_value=1,
        max_value=20,
        value=int(get_setting("rag.top_k", 5)),
    )

    if st.button("Get answer") and question.strip():
        with st.spinner("Retrieving and generating answer..."):
            chunks = retrieve_top_k(df_index, emb_array, question, k=top_k_override)
            context = build_context(chunks)
            answer = answer_query(question, context)

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Retrieved chunks")
        st.dataframe(chunks[["source", "chunk_id", "similarity"]])
