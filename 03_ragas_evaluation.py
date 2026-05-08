import json
import warnings
from pathlib import Path

import numpy as np
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

from config import get_embeddings, get_int_env, get_llm
from qa_pairs import QA_PAIRS
from rag_common import PROMPTS, answer_with_context, build_vectorstore

warnings.filterwarnings("ignore")


def collect_rag_outputs(vectorstore, prompt_version: str) -> list:
    retriever = vectorstore.as_retriever(search_kwargs={"k": get_int_env("RETRIEVER_K", 3)})
    prompt = PROMPTS[prompt_version]
    results = []

    print(f"\nRunning {len(QA_PAIRS)} questions with prompt {prompt_version} ...")
    for i, qa in enumerate(QA_PAIRS, 1):
        out = answer_with_context(retriever, prompt, qa["question"])
        results.append(
            {
                "question": qa["question"],
                "reference": qa["reference"],
                "answer": out["answer"],
                "contexts": out["contexts"],
            }
        )
        print(f"  [{i:02d}/{len(QA_PAIRS)}] {qa['question'][:70]}")

    return results


def build_ragas_dataset(rag_results: list):
    samples = [
        SingleTurnSample(
            user_input=row["question"],
            response=row["answer"],
            retrieved_contexts=row["contexts"],
            reference=row["reference"],
        )
        for row in rag_results
    ]
    return EvaluationDataset(samples=samples)


def _mean_metric(result, key: str) -> float:
    values = result[key]
    clean_values = [float(value) for value in values if value is not None and not np.isnan(value)]
    return float(np.mean(clean_values)) if clean_values else 0.0


def run_ragas_eval(rag_results: list, version: str) -> dict:
    print(f"\nRunning RAGAS evaluation for prompt {version} ...")
    dataset = build_ragas_dataset(rag_results)
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=get_llm(temperature=0),
        embeddings=get_embeddings(),
    )
    scores = {
        key: _mean_metric(result, key)
        for key in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
    }
    for key, value in scores.items():
        marker = " target met" if key == "faithfulness" and value >= 0.8 else ""
        print(f"  {key:30s}: {value:.4f}{marker}")
    return scores


def main() -> None:
    print("=" * 60)
    print("  Step 3: RAGAS Evaluation")
    print("=" * 60)

    vectorstore = build_vectorstore()
    v1_results = collect_rag_outputs(vectorstore, "v1")
    v2_results = collect_rag_outputs(vectorstore, "v2")

    v1_scores = run_ragas_eval(v1_results, "v1")
    v2_scores = run_ragas_eval(v2_results, "v2")

    print("\nMetric comparison")
    print("-" * 72)
    for metric in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        s1 = v1_scores[metric]
        s2 = v2_scores[metric]
        winner = "V1" if s1 > s2 else "V2"
        print(f"{metric:30s} V1={s1:.4f}  V2={s2:.4f}  winner={winner}")

    best_faithfulness = max(v1_scores["faithfulness"], v2_scores["faithfulness"])
    target_met = best_faithfulness >= 0.8
    print(f"\nTarget {'met' if target_met else 'not met'}: best faithfulness = {best_faithfulness:.4f}")

    report = {
        "prompt_v1_scores": v1_scores,
        "prompt_v2_scores": v2_scores,
        "target_met": target_met,
        "best_faithfulness": best_faithfulness,
    }
    Path("data").mkdir(exist_ok=True)
    Path("data/ragas_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Saved data/ragas_report.json")


if __name__ == "__main__":
    main()
