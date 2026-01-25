# GenerativeAI
Project 3 - LLM RAG FineTuning AIAgent LangChain LangGraph VectorDB

**project3-llm-rag-assistant**
**==================****==================****==================**
A small, profile‑aware Retrieval Augmented Generation (RAG) assistant that lets you index different document sets (RFPs, resumes, tech docs, etc.) and ask questions against them using local parquet/npy indices and an LLM backend.

The core project is a library, designed to be reused by both an API layer and a Streamlit UI.

**Project structure**
**=================**
text
project3-llm-rag-assistant/
├─ src/
│  ├─ __init__.py
│  ├─ settings.py
│  ├─ ingest.py
│  ├─ rag_query.py
│  ├─ eval.py                # optional: evaluation utilities
│  └─ ...
├─ data/
│  ├─ rfp/
│  │  ├─ docs/               # raw docs (PDF/DOCX/TXT) for RFP profile
│  │  └─ index/              # parquet + npy index for RFP
│  ├─ resume/
│  │  ├─ docs/               # raw docs for resume profile
│  │  └─ index/              # parquet + npy index for resume
│  └─ tech_docs/
│     ├─ docs/
│     └─ index/
├─ config.yml
├─ pyproject.toml / setup.cfg (or setup.py)
└─ README.md

**src/settings.py:** Central place for reading config.yml, resolving active profile, and returning paths such as docs and index directories.
**src/ingest.py:** CLI-style script that reads raw documents, chunks them, generates embeddings, and writes a parquet + npy index for the active profile.
**src/rag_query.py:** Core RAG functions: load index, retrieve top‑k chunks, assemble context, and call the LLM to answer a query.
**data/<profile>/docs:** Where you place the documents you want to index for that profile.
**data/<profile>/index:** Where ingestion writes chunks.parquet and embeddings.npy (or similar).

You can add more profiles simply by adding new entries in config.yml and creating matching folders under data/.

**YAML configuration**
**==================**
config.yml controls profiles, paths, model settings, and RAG behavior. A typical structure looks like this:

text
profile: rfp

paths:
  base_dir: ./data
  rfp:
    docs_dir: ./data/rfp/docs
    index_dir: ./data/rfp/index
  resume:
    docs_dir: ./data/resume/docs
    index_dir: ./data/resume/index

llm:
  provider: openai
  model: gpt-4.1-mini
  temperature: 0.1
  max_tokens: 512

embeddings:
  provider: openai
  model: text-embedding-3-small
  dim: 1536

rag:
  top_k: 5
  chunk_size: 800
  chunk_overlap: 200

prompts:
  rfp:
    system: >
      You are an assistant specialized in analyzing RFP documents.
  resume:
    system: >
      You are an assistant helping to analyze resumes and match them to jobs.

**Explanation of key elements:**
**============================**
**profile**:
The active profile (e.g., rfp, resume, tech_docs). This controls which docs and index directories are used.
**paths.base_dir:**
Root data directory that holds all profile‑specific folders.
**paths.<profile>.docs_dir:**
Location where you store raw documents for that profile.
**paths.<profile>.index_dir:**
Location where ingestion writes the parquet and npy index files for that profile.
**llm.provider / llm.model:**
Which LLM backend to call and which model name to use.
**llm.temperature:**
Controls answer randomness; lower values make responses more deterministic.
**llm.max_tokens:**
Max tokens for generated answers.
**embeddings.provider / embeddings.model:**
Embedding backend and model used to encode chunks.
**embeddings.dim:**
Expected dimensionality of embedding vectors; used when loading embeddings.
**rag.top_k:**
Number of most similar chunks to retrieve for each query.
**rag.chunk_size / rag.chunk_overlap:**
Text splitting parameters used during ingestion.
**prompts.<profile>.system:**
System prompt to specialize behavior per profile.

src/settings.py is responsible for reading this file and exposing helpers like get_active_profile(), get_docs_dir(), get_data_dir(), and get_prompt_template().

**How the main scripts work**
**=========================**
**settings.py**
Loads config.yml at startup (using PyYAML) and caches it.
**Provides helper functions:**
get_active_profile(): Returns the current value of profile from the YAML.
get_docs_dir(): Returns the docs directory for the active profile.
get_data_dir() / get_index_dir(): Return index/data directories for the active profile.
get_setting(path, default=None): Get nested config values like rag.top_k.

**ingest.py**
Reads documents from get_docs_dir() for the active profile.
Splits text into chunks using rag.chunk_size and rag.chunk_overlap.
Calls the embedding model to encode chunks.
Saves:
A parquet file with metadata and text chunks.
An npy file with the corresponding embedding vectors.
Typically exposes a main() that supports flags like --force to rebuild the index from scratch.

**rag_query.py**
load_index():
Loads the parquet file and npy embeddings from the active profile’s index directory.
retrieve_top_k(df, emb, query, k):
Encodes the query, computes similarity to all chunk embeddings, and returns the top‑k rows.
build_context(chunks):
Concatenates the retrieved chunks into a single context string.
answer_query(question, context):
Calls the LLM using llm.* settings and the profile‑specific system prompt to generate an answer.

**Installation and setup**
**======================**

**1. Clone the repo**
bash
  git clone https://github.com/<your-username>/project3-llm-rag-assistant.git
  cd project3-llm-rag-assistant

**2. Create and activate a virtual environment**
bash
  python -m venv .venv
  .\.venv\Scripts\activate   # Windows
  # source .venv/bin/activate  # macOS/Linux

**3. Install dependencies**
If using pyproject.toml or requirements.txt, run:

bash
  pip install -e .

Make sure pyyaml is included, as it is required by settings.py to read config.yml.
​
If you don’t have a requirements file, install the core deps manually (example):

bash
  pip install pyyaml streamlit openai python-dotenv pypdf python-docx pandas numpy pyarrow fastparquet

**4. Set environment variables**
Provide your LLM/embedding API keys (for example in .env or your shell):

bash
  set OPENAI_API_KEY=sk-...
# or export OPENAI_API_KEY=...

**Running ingestion**
**=================**
**1. Place documents for a profile**
For example, for the resume profile:
Set profile: resume in config.yml.
Put your resume PDFs/DOCX/TXT into:

text
data/resume/docs/

**2. Run ingestion**
From the project root:

bash
  .\.venv\Scripts\activate
  python -m src.ingest --force

**This will:**

Read docs from data/<profile>/docs.
Build or rebuild the index under data/<profile>/index as parquet + npy.

Repeat with other profiles by changing profile in config.yml and running ingestion again.

**Running queries from code (optional)**
**====================================**
You can test RAG directly in Python:

**python**
  from src.settings import get_active_profile
  from src.rag_query import load_index, retrieve_top_k, build_context, answer_query
  
  print("Active profile:", get_active_profile())
  
  df, emb = load_index()
  question = "What are the main skills highlighted in this resume?"
  chunks = retrieve_top_k(df, emb, question, k=5)
  context = build_context(chunks)
  answer = answer_query(question, context)
  
  print(answer)

**Streamlit app (separate UI project)**
**===================================**
There is a separate Streamlit UI project (for example in rag-streamlit-ui/) that uses this core library to provide:

  A file‑upload UI to add docs to the active profile’s docs folder.
  A button to run ingestion from the UI.
  A chat‑like interface to ask questions using the parquet/npy index.

**Setup for the Streamlit app**

**1. Ensure you are in the same virtual environment**

From the core project root:

bash
  cd project3-llm-rag-assistant
  .\.venv\Scripts\activate
  pip install pyyaml streamlit
You must install pyyaml in this activated venv, otherwise the Streamlit app will fail on import yaml when it imports src.settings.

**2. Install the core library in editable mode (once)**

Still in the same venv:

bash
  pip install -e .

**3. Run the Streamlit app**

Assuming the UI project is rag-streamlit-ui at the same level:

bash
  cd ..\rag-streamlit-ui
  streamlit run app.py
The app will open at http://localhost:8501.

**Typical flow in the UI:**

  Select or rely on the active profile from config.yml.
  Upload files; the app saves them into the active profile’s docs folder.
  Click “Run ingestion” to build the index.
  Ask questions in the Q&A tab; the app uses the same parquet/npy index and RAG logic as the core project.
