"""
DSPy Optimizers for automatic prompt optimization and self-improvement.

Supports:
- BootstrapFewShot — bootstrap demonstrations from a teacher model
- MIPROv2 — multi-instruction prompt optimization
- COPRO — collaborative prompt optimization

Each optimizer wraps the DSPy compiler API and integrates with
the evaluation framework for metric-driven optimization.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

import dspy

from config.logging_config import get_logger
from config.settings import OptimizerStrategy, get_settings

logger = get_logger("optimizers")


class PipelineOptimizer:
    """
    Facade for DSPy optimization strategies.

    Wraps DSPy's optimizer ecosystem and provides:
    - Strategy selection via configuration
    - Metric-driven compilation
    - Optimized pipeline serialization
    - A/B comparison between unoptimized and optimized pipelines
    """

    def __init__(
        self,
        strategy: Optional[OptimizerStrategy] = None,
        metric: Optional[Any] = None,
    ) -> None:
        self._settings = get_settings()
        self.strategy = strategy or self._settings.optimizer_strategy
        self.metric = metric
        self._optimized_program = None

    def optimize(
        self,
        program: dspy.Module,
        trainset: list[dspy.Example],
        valset: Optional[list[dspy.Example]] = None,
        metric: Optional[Any] = None,
    ) -> dspy.Module:
        """
        Optimize a DSPy program using the configured strategy.

        Args:
            program: The DSPy Module to optimize.
            trainset: Training examples.
            valset: Optional validation examples.
            metric: Optional override for the evaluation metric.

        Returns:
            Optimized DSPy Module.
        """
        active_metric = metric or self.metric
        if active_metric is None:
            raise ValueError("A metric function is required for optimization")

        logger.info(
            f"Starting optimization with strategy={self.strategy.value}, "
            f"trainset={len(trainset)} examples"
        )
        start = time.perf_counter()

        if self.strategy == OptimizerStrategy.BOOTSTRAP_FEWSHOT:
            optimized = self._bootstrap_fewshot(program, trainset, active_metric)
        elif self.strategy == OptimizerStrategy.MIPRO_V2:
            optimized = self._mipro_v2(program, trainset, valset or trainset, active_metric)
        elif self.strategy == OptimizerStrategy.COPRO:
            optimized = self._copro(program, trainset, active_metric)
        elif self.strategy == OptimizerStrategy.NONE:
            logger.info("No optimization strategy selected — returning unoptimized program")
            return program
        else:
            raise ValueError(f"Unknown optimizer strategy: {self.strategy}")

        elapsed = time.perf_counter() - start
        logger.info(f"Optimization complete in {elapsed:.1f}s")

        self._optimized_program = optimized
        return optimized

    def _bootstrap_fewshot(
        self,
        program: dspy.Module,
        trainset: list[dspy.Example],
        metric: Any,
    ) -> dspy.Module:
        """BootstrapFewShot: Generate demonstrations from a teacher model."""
        optimizer = dspy.BootstrapFewShot(
            metric=metric,
            max_bootstrapped_demos=self._settings.optimization_max_bootstrapped_demos,
            max_labeled_demos=self._settings.optimization_max_labeled_demos,
        )
        return optimizer.compile(program, trainset=trainset)

    def _mipro_v2(
        self,
        program: dspy.Module,
        trainset: list[dspy.Example],
        valset: list[dspy.Example],
        metric: Any,
    ) -> dspy.Module:
        """MIPROv2: Multi-instruction prompt optimization."""
        optimizer = dspy.MIPROv2(
            metric=metric,
            num_candidates=self._settings.optimization_num_candidates,
            num_threads=self._settings.optimization_num_threads,
        )
        return optimizer.compile(
            program,
            trainset=trainset,
            valset=valset,
        )

    def _copro(
        self,
        program: dspy.Module,
        trainset: list[dspy.Example],
        metric: Any,
    ) -> dspy.Module:
        """COPRO: Collaborative prompt optimization."""
        optimizer = dspy.COPRO(
            metric=metric,
            breadth=self._settings.optimization_num_candidates,
        )
        return optimizer.compile(program, trainset=trainset)

    def save_optimized(self, path: Optional[str] = None) -> str:
        """Serialize optimized program to disk."""
        if self._optimized_program is None:
            raise RuntimeError("No optimized program to save. Run optimize() first.")

        save_dir = Path(path) if path else Path("optimized_pipelines")
        save_dir.mkdir(parents=True, exist_ok=True)

        filepath = save_dir / f"optimized_{self.strategy.value}_{int(time.time())}.json"
        self._optimized_program.save(str(filepath))

        logger.info(f"Saved optimized pipeline to {filepath}")
        return str(filepath)

    def load_optimized(self, program: dspy.Module, path: str) -> dspy.Module:
        """Load a previously optimized program from disk."""
        program.load(path)
        self._optimized_program = program
        logger.info(f"Loaded optimized pipeline from {path}")
        return program

    def compare(
        self,
        original: dspy.Module,
        optimized: dspy.Module,
        testset: list[dspy.Example],
        metric: Optional[Any] = None,
    ) -> dict[str, Any]:
        """
        A/B comparison between original and optimized pipelines.

        Returns comparison statistics.
        """
        active_metric = metric or self.metric
        if active_metric is None:
            raise ValueError("A metric function is required for comparison")

        def _evaluate(program: dspy.Module, examples: list[dspy.Example]) -> list[float]:
            scores = []
            for ex in examples:
                try:
                    pred = program(query=ex.question)
                    prediction = dspy.Prediction(answer=pred.answer if hasattr(pred, "answer") else str(pred))
                    score = active_metric(ex, prediction)
                    scores.append(score)
                except Exception as e:
                    logger.error(f"Evaluation error: {e}")
                    scores.append(0.0)
            return scores

        logger.info(f"Running A/B comparison on {len(testset)} examples")

        original_scores = _evaluate(original, testset)
        optimized_scores = _evaluate(optimized, testset)

        result = {
            "original": {
                "mean": sum(original_scores) / len(original_scores) if original_scores else 0,
                "scores": original_scores,
            },
            "optimized": {
                "mean": sum(optimized_scores) / len(optimized_scores) if optimized_scores else 0,
                "scores": optimized_scores,
            },
            "improvement": (
                (sum(optimized_scores) - sum(original_scores)) / len(original_scores)
                if original_scores else 0
            ),
            "num_examples": len(testset),
        }

        logger.info(
            f"A/B comparison: original={result['original']['mean']:.3f}, "
            f"optimized={result['optimized']['mean']:.3f}, "
            f"improvement={result['improvement']:.3f}"
        )

        return result
