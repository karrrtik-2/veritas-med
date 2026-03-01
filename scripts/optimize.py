"""
Script: Run DSPy prompt optimization pipeline.

Usage:
    python -m scripts.optimize [--strategy bootstrap_fewshot|mipro_v2|copro]
"""

from __future__ import annotations

import argparse
import sys
import time

import dspy

from config.logging_config import setup_logging, get_logger
from config.settings import OptimizerStrategy, get_settings
from core.modules import MedicalQAPipeline
from optimization.datasets import MedicalDatasetManager
from optimization.evaluators import CompositeMetric
from optimization.optimizers import PipelineOptimizer
from retrieval.retriever import MedicalRetriever
from retrieval.vectorstore import PineconeManager


def main():
    parser = argparse.ArgumentParser(description="Run DSPy prompt optimization")
    parser.add_argument(
        "--strategy",
        choices=["bootstrap_fewshot", "mipro_v2", "copro"],
        default="bootstrap_fewshot",
        help="Optimization strategy",
    )
    parser.add_argument("--dataset", type=str, default=None, help="Path to dataset JSON")
    parser.add_argument("--output", type=str, default="optimized_pipelines", help="Output directory")
    args = parser.parse_args()

    setup_logging(level="INFO", json_output=False)
    logger = get_logger("optimize_script")

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

    trainset, devset, testset = dataset_mgr.split()
    logger.info(f"Dataset: train={len(trainset)}, dev={len(devset)}, test={len(testset)}")

    # Initialize pipeline
    pinecone_mgr = PineconeManager()
    retriever = MedicalRetriever(pinecone_manager=pinecone_mgr)
    pipeline = MedicalQAPipeline(retriever_fn=retriever)

    # Metric
    metric = CompositeMetric(weights=settings.eval_metric_weights)

    # Optimize
    strategy = OptimizerStrategy(args.strategy)
    optimizer = PipelineOptimizer(strategy=strategy, metric=metric)

    logger.info(f"Starting optimization with strategy={strategy.value}")
    start = time.perf_counter()

    optimized_pipeline = optimizer.optimize(
        program=pipeline,
        trainset=trainset,
        valset=devset,
        metric=metric,
    )

    elapsed = time.perf_counter() - start
    logger.info(f"Optimization completed in {elapsed:.1f}s")

    # Save
    save_path = optimizer.save_optimized(args.output)
    logger.info(f"Optimized pipeline saved to: {save_path}")

    # Compare
    if testset:
        logger.info("Running A/B comparison...")
        comparison = optimizer.compare(
            original=pipeline,
            optimized=optimized_pipeline,
            testset=testset,
            metric=metric,
        )
        logger.info(f"Comparison results: {comparison}")


if __name__ == "__main__":
    main()
