import csv
import time
from pathlib import Path

from src.settings import get_active_profile, get_data_dir, get_setting
from src.rag_query import load_index, retrieve_top_k, build_context, answer_query


def normalize(text: str) -> str:
    return text.lower().strip()


def get_eval_file() -> Path:
    # data dir for active profile, e.g. data/general
    data_dir = get_data_dir()
    return data_dir / "eval_samples.csv"


def evaluate_sample(row, df_index, emb_array):
    question = row["question"].strip()
    expected_keywords = [
        k.strip().lower()
        for k in row["expected_keywords"].split(";")
        if k.strip()
    ]

    t0 = time.time()
    chunks = retrieve_top_k(
        df_index,
        emb_array,
        question,
        k=int(get_setting("rag.top_k", 5)),
    )
    context = build_context(chunks)
    answer = answer_query(question, context)
    latency = time.time() - t0

    answer_norm = normalize(answer)
    hits = [kw for kw in expected_keywords if kw in answer_norm]
    passed = len(hits) == len(expected_keywords)

    return {
        "question": question,
        "expected_keywords": expected_keywords,
        "answer": answer,
        "hits": hits,
        "passed": passed,
        "latency_sec": latency,
    }


def main():
    profile = get_active_profile()
    eval_file = get_eval_file()
    print(f"Active profile: {profile}")
    print(f"Eval file: {eval_file}")

    if not eval_file.exists():
        print("Eval file not found. Create eval_samples.csv in this profile's data folder.")
        return

    try:
        df_index, emb_array = load_index()
    except FileNotFoundError:
        print("Index not found. Run ingestion first for this profile.")
        return

    results = []
    with eval_file.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            res = evaluate_sample(row, df_index, emb_array)
            results.append(res)
            status = "PASS" if res["passed"] else "FAIL"
            print(
                f"[{status}] Q: {res['question']} | "
                f"hits={res['hits']} | "
                f"latency={res['latency_sec']:.2f}s"
            )

    if not results:
        print("No eval rows found.")
        return

    passed_count = sum(1 for r in results if r["passed"])
    total = len(results)
    avg_latency = sum(r["latency_sec"] for r in results) / total

    print("-" * 60)
    print(f"Overall: {passed_count}/{total} passed ({passed_count/total:.0%})")
    print(f"Average latency: {avg_latency:.2f}s")


if __name__ == "__main__":
    main()
