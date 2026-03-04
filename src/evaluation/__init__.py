"""Evaluation module for model assessment and benchmarking."""

from src.evaluation.benchmark import ComparisonReport, ModelBenchmark
from src.evaluation.evaluator import EvaluationResult, ModelEvaluator
from src.evaluation.human_eval import GeneratedSample, HumanEvalGenerator

__all__ = [
    "ComparisonReport",
    "EvaluationResult",
    "GeneratedSample",
    "HumanEvalGenerator",
    "ModelBenchmark",
    "ModelEvaluator",
]
