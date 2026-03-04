"""Model benchmarking for comparing base vs fine-tuned models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from datasets import Dataset

from src.evaluation.evaluator import EvaluationResult, ModelEvaluator
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

log = get_logger(__name__)


@dataclass
class ComparisonReport:
    """Report comparing two or more model evaluations."""

    results: list[EvaluationResult] = field(default_factory=list)
    dataset_name: str = ""

    @property
    def model_names(self) -> list[str]:
        """Get names of all compared models."""
        return [r.model_name for r in self.results]

    def get_best_model(self, metric: str = "perplexity") -> str | None:
        """Get the model with the best score for a metric.

        Args:
            metric: Metric name to compare by.

        Returns:
            Name of the best model, or None if no results.
        """
        if not self.results:
            return None

        valid = [(r.model_name, r.metrics.get(metric)) for r in self.results]
        valid = [(name, val) for name, val in valid if val is not None]
        if not valid:
            return None

        # For perplexity, lower is better; for others, higher is better
        lower_is_better = metric in ("perplexity", "loss", "eval_loss")
        best = min(valid, key=lambda x: x[1]) if lower_is_better else max(valid, key=lambda x: x[1])
        return best[0]

    def get_improvement(self, metric: str = "perplexity") -> dict[str, Any] | None:
        """Calculate improvement between first and last model.

        Args:
            metric: Metric to compare.

        Returns:
            Dict with base_value, finetuned_value, absolute_change, relative_change.
        """
        if len(self.results) < 2:
            return None

        base_val = self.results[0].metrics.get(metric)
        ft_val = self.results[-1].metrics.get(metric)
        if base_val is None or ft_val is None:
            return None

        absolute = ft_val - base_val
        relative = absolute / base_val if base_val != 0 else 0.0

        return {
            "metric": metric,
            "base_value": base_val,
            "finetuned_value": ft_val,
            "absolute_change": absolute,
            "relative_change": relative,
        }

    def summary(self) -> str:
        """Human-readable comparison summary."""
        lines = [
            "Model Comparison Report:",
            f"  Dataset: {self.dataset_name}",
            f"  Models compared: {len(self.results)}",
            "",
        ]

        for result in self.results:
            lines.append(f"  {result.model_name}:")
            for key, value in result.metrics.items():
                if isinstance(value, float):
                    lines.append(f"    {key}: {value:.4f}")
                else:
                    lines.append(f"    {key}: {value}")
            lines.append("")

        best_ppl = self.get_best_model("perplexity")
        if best_ppl:
            lines.append(f"  Best perplexity: {best_ppl}")

        improvement = self.get_improvement("perplexity")
        if improvement:
            lines.append(
                f"  Perplexity change: {improvement['absolute_change']:+.2f} "
                f"({improvement['relative_change']:+.1%})"
            )

        return "\n".join(lines)


class ModelBenchmark:
    """Compare multiple models on the same dataset.

    Args:
        dataset: Evaluation dataset.
        dataset_name: Name for reporting.
        text_column: Text column in dataset.
        max_length: Maximum sequence length.
    """

    def __init__(
        self,
        dataset: Dataset,
        dataset_name: str = "",
        text_column: str = "text",
        max_length: int = 2048,
    ) -> None:
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.text_column = text_column
        self.max_length = max_length
        self._results: list[EvaluationResult] = []

    def add_model(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        model_name: str,
    ) -> EvaluationResult:
        """Evaluate a model and add it to the comparison.

        Args:
            model: Model to evaluate.
            tokenizer: Model's tokenizer.
            model_name: Name for reporting.

        Returns:
            EvaluationResult for this model.
        """
        evaluator = ModelEvaluator(model, tokenizer)
        result = evaluator.evaluate_all(
            self.dataset,
            model_name=model_name,
            dataset_name=self.dataset_name,
            text_column=self.text_column,
            max_length=self.max_length,
        )
        self._results.append(result)
        log.info(
            "benchmark_model_added",
            model=model_name,
            perplexity=result.perplexity,
        )
        return result

    def compare(self) -> ComparisonReport:
        """Generate a comparison report for all evaluated models.

        Returns:
            ComparisonReport with all results.
        """
        report = ComparisonReport(
            results=list(self._results),
            dataset_name=self.dataset_name,
        )
        log.info(
            "benchmark_comparison_complete",
            num_models=len(self._results),
            best_model=report.get_best_model(),
        )
        return report

    def reset(self) -> None:
        """Clear all stored results."""
        self._results.clear()
