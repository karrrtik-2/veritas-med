"""Optimization and evaluation framework for self-improving pipelines."""

from optimization.evaluators import (
    MedicalEvaluator,
    FactualAccuracyMetric,
    RelevanceMetric,
    CompletenessMetric,
    SafetyMetric,
    CompositeMetric,
)
from optimization.optimizers import PipelineOptimizer
from optimization.datasets import MedicalDatasetManager, MedicalExample
from optimization.feedback import FeedbackLoop, FeedbackEntry

__all__ = [
    "MedicalEvaluator",
    "FactualAccuracyMetric",
    "RelevanceMetric",
    "CompletenessMetric",
    "SafetyMetric",
    "CompositeMetric",
    "PipelineOptimizer",
    "MedicalDatasetManager",
    "MedicalExample",
    "FeedbackLoop",
    "FeedbackEntry",
]
