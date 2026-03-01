"""
Script: Run evaluation suite against the medical QA pipeline.

Usage:
    python -m scripts.evaluate [--dataset path/to/dataset.json]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import dspy

from config.logging_config import setup_logging, get_logger
from config.settings import get_settings
from core.modules import MedicalQAPipeline
from optimization.datasets import MedicalDatasetManager
from optimization.evaluators import MedicalEvaluator, CompositeMetric
from retrieval.retriever import MedicalRetriever
from retrieval.vectorstore import PineconeManager


def main():
    parser = argparse.ArgumentParser(description="Evaluate medical QA pipeline")
    parser.add_argument("--dataset", type=str, default=None, help="Path to evaluation dataset")
    parser.add_argument("--output", type=str, default="evaluation_results", help="Output directory")
    parser.add_argument("--sample-size", type=int, default=None, help="Number of examples to evaluate")
    args = parser.parse_args()

    setup_logging(level="INFO", json_output=False)
    logger = get_logger("evaluate_script")

    settings = get_settings()

    # Configure DSPy
    lm = dspy.LM(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )
    dspy.configure(lm=lm)

    # Load dataset
    dataset_mgr = MedicalDatasetManager()
    if args.dataset:
        dataset_mgr.load_from_json(args.dataset)
    else:
        dataset_mgr.load_seed_examples()

    examples = dataset_mgr.get_dspy_examples()
    if args.sample_size and args.sample_size < len(examples):
        examples = examples[: args.sample_size]

    logger.info(f"Evaluating {len(examples)} examples")

    # Initialize pipeline
    pinecone_mgr = PineconeManager()
    retriever = MedicalRetriever(pinecone_manager=pinecone_mgr)
    pipeline = MedicalQAPipeline(retriever_fn=retriever)

    # Run evaluation
    evaluator = MedicalEvaluator()
    results = []

    for i, ex in enumerate(examples):
        logger.info(f"Evaluating example {i + 1}/{len(examples)}: {ex.question[:60]}...")
        start = time.perf_counter()

        try:
            response = pipeline(query=ex.question)
            latency = (time.perf_counter() - start) * 1000

            eval_result = evaluator.evaluate_single(
                question=ex.question,
                predicted_answer=response.answer,
                ground_truth=ex.get("answer", ""),
            )

            results.append({
                "question": ex.question,
                "predicted_answer": response.answer,
                "ground_truth": ex.get("answer", ""),
                "composite_score": eval_result["composite_score"],
                "confidence": response.reasoning_trace.overall_confidence,
                "safety_level": response.safety.level.value,
                "latency_ms": round(latency, 1),
            })
        except Exception as e:
            logger.error(f"Error evaluating example {i + 1}: {e}")
            results.append({
                "question": ex.question,
                "error": str(e),
                "composite_score": 0.0,
            })

    # Aggregate stats
    scores = [r["composite_score"] for r in results]
    latencies = [r.get("latency_ms", 0) for r in results if "latency_ms" in r]

    summary = {
        "num_examples": len(results),
        "mean_score": sum(scores) / len(scores) if scores else 0,
        "min_score": min(scores) if scores else 0,
        "max_score": max(scores) if scores else 0,
        "mean_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
        "results": results,
    }

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"eval_{int(time.time())}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(
        f"Evaluation complete: "
        f"mean_score={summary['mean_score']:.3f}, "
        f"mean_latency={summary['mean_latency_ms']:.0f}ms. "
        f"Results saved to {output_file}"
    )


if __name__ == "__main__":
    main()
