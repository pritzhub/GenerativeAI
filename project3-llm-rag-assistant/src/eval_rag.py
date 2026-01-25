#Suggestions after running the evals : if required
#    What the scores mean
#    Correctness 5.00: Answers match the RFP content for your golden questions; the judge found no factual issues.
#
#    Groundedness 5.00: The assistant is staying within retrieved context and not hallucinating beyond what the RFPs say.
#
#    Completeness 4.50: Most answers fully address the questions, but some may miss minor aspects (e.g., not listing every requirement or risk).
#
#    How to improve completeness (if you want)
#    If you open eval_results.csv and look at questions with lower completeness, you can:
#
#    Adjust prompts to explicitly say “enumerate all distinct requirements/risks you can find in the context; group by category if helpful.”
#
#    Increase k in retrieve_top_k from 5 to 8–10 if a question spans many parts of the RFP.

# src/eval_rag.py
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

from src.settings import get_data_dir
from src.config import get_openai_client
from src.rag_query import load_index, retrieve_top_k, build_context, answer_query


DATA_DIR = get_data_dir()
EVAL_CSV = DATA_DIR / "eval_questions.csv"

# We will reuse ANSWER_MODEL as the judge for now; later you can switch to a bigger judge model.
JUDGE_MODEL = "gpt-4.1-mini"

def load_eval_questions() -> pd.DataFrame:
    if not EVAL_CSV.exists():
        raise FileNotFoundError(f"Eval CSV not found at: {EVAL_CSV}")
    df = pd.read_csv(EVAL_CSV)
    if "question" not in df.columns:
        raise ValueError("Eval CSV must have a 'question' column.")
    return df


def judge_answer(question: str, context: str, answer: str) -> Dict[str, Any]:
    """
    LLM-as-a-judge: ask the model to score the answer's correctness, groundedness,
    and completeness on a 1–5 scale, and optionally add comments.[web:123][web:185]
    """
    client = get_openai_client()

    system_msg = (
        "You are an impartial evaluator for a Retrieval-Augmented Generation (RAG) system "
        "answering questions about energy-sector RFPs.\n"
        "You will be given a question, the context (retrieved chunks), and the system's answer.\n"
        "Evaluate the answer on a 1–5 scale for:\n"
        "- correctness: Is the answer factually correct given the context?\n"
        "- groundedness: Does the answer stay within the information in the context (no hallucinations)?\n"
        "- completeness: Does the answer fully address the question, given the context?\n"
        "Return a short JSON object with numeric scores and a brief comment."
    )

    user_msg = f"""
Question:
{question}

Context:
{context}

Answer:
{answer}

Provide your evaluation as a JSON object with this exact structure:
{{
  "correctness": <number 1-5>,
  "groundedness": <number 1-5>,
  "completeness": <number 1-5>,
  "comment": "<short explanation>"
}}
"""

    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
    )

    content = resp.choices[0].message.content.strip()

    # Very simple JSON extraction; for robustness you could use json.loads with try/except.
    import json

    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        # Fallback: wrap into a default structure
        result = {
            "correctness": None,
            "groundedness": None,
            "completeness": None,
            "comment": f"Could not parse JSON. Raw response: {content}",
        }

    return result


def evaluate_single(df_index: pd.DataFrame, emb_array: np.ndarray, question: str) -> Dict[str, Any]:
    # 1) Retrieve context using the same RAG pipeline as rag_query.py
    chunks = retrieve_top_k(df_index, emb_array, question, k=5)
    context = build_context(chunks)

    # 2) Generate an answer with the RAG assistant
    answer = answer_query(question, context)

    # 3) Ask the judge model to score it
    eval_scores = judge_answer(question, context, answer)

    return {
        "question": question,
        "answer": answer,
        "context_sources": "; ".join(
            f"{row['source']}#{row['chunk_id']}" for _, row in chunks.iterrows()
        ),
        "correctness": eval_scores.get("correctness"),
        "groundedness": eval_scores.get("groundedness"),
        "completeness": eval_scores.get("completeness"),
        "comment": eval_scores.get("comment"),
    }


def main():
    df_eval = load_eval_questions()
    df_index, emb_array = load_index()

    results = []

    print(f"Running RAG eval on {len(df_eval)} questions...\n")

    for i, row in df_eval.iterrows():
        q = row["question"]
        print(f"[{i+1}/{len(df_eval)}] Evaluating question: {q}")
        res = evaluate_single(df_index, emb_array, q)
        results.append(res)

    df_results = pd.DataFrame(results)

    out_path = DATA_DIR / "eval_results.csv"
    df_results.to_csv(out_path, index=False)
    print(f"\nSaved evaluation results to: {out_path}")

    # Print simple aggregate scores
    for metric in ["correctness", "groundedness", "completeness"]:
        vals = pd.to_numeric(df_results[metric], errors="coerce").dropna()
        if len(vals) > 0:
            print(f"{metric} mean: {vals.mean():.2f}")
        else:
            print(f"{metric} mean: N/A (no numeric values)")


if __name__ == "__main__":
    main()
