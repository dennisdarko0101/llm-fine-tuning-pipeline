"""Model evaluation with perplexity, generation metrics, and task accuracy."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from datasets import Dataset

from src.utils.logger import get_logger

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

log = get_logger(__name__)


@dataclass
class EvaluationResult:
    """Results from a model evaluation run."""

    metrics: dict[str, Any] = field(default_factory=dict)
    model_name: str = ""
    dataset_name: str = ""
    num_samples: int = 0

    @property
    def perplexity(self) -> float | None:
        """Extract perplexity from metrics."""
        return self.metrics.get("perplexity")

    @property
    def bleu(self) -> float | None:
        """Extract BLEU score from metrics."""
        return self.metrics.get("bleu")

    @property
    def rouge_l(self) -> float | None:
        """Extract ROUGE-L F1 score from metrics."""
        return self.metrics.get("rouge_l")

    @property
    def accuracy(self) -> float | None:
        """Extract task accuracy from metrics."""
        return self.metrics.get("accuracy")

    def summary(self) -> str:
        """Human-readable summary of evaluation results."""
        lines = [
            "Evaluation Results:",
            f"  Model: {self.model_name}",
            f"  Dataset: {self.dataset_name}",
            f"  Samples: {self.num_samples}",
        ]
        if self.perplexity is not None:
            lines.append(f"  Perplexity: {self.perplexity:.2f}")
        if self.bleu is not None:
            lines.append(f"  BLEU: {self.bleu:.4f}")
        if self.rouge_l is not None:
            lines.append(f"  ROUGE-L: {self.rouge_l:.4f}")
        if self.accuracy is not None:
            lines.append(f"  Accuracy: {self.accuracy:.2%}")
        return "\n".join(lines)


class ModelEvaluator:
    """Evaluate a language model on various metrics.

    Args:
        model: Pretrained or fine-tuned model.
        tokenizer: Tokenizer matching the model.
        device: Device to use for evaluation.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        device: str | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def evaluate_perplexity(
        self,
        dataset: Dataset,
        text_column: str = "text",
        max_length: int = 2048,
        batch_size: int = 4,
    ) -> float:
        """Compute perplexity on a dataset.

        Args:
            dataset: Dataset with text column.
            text_column: Name of the text column.
            max_length: Maximum sequence length for tokenization.
            batch_size: Batch size for evaluation.

        Returns:
            Perplexity score (lower is better).
        """
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0

        for i in range(0, len(dataset), batch_size):
            batch_texts = dataset[i : i + batch_size][text_column]
            encodings = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True,
            )
            input_ids = encodings["input_ids"].to(self.device)
            attention_mask = encodings["attention_mask"].to(self.device)

            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids,
                )
            loss = outputs.loss
            num_tokens = attention_mask.sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

        avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
        perplexity = math.exp(avg_loss) if avg_loss < 100 else float("inf")

        log.info(
            "perplexity_computed",
            perplexity=f"{perplexity:.2f}",
            avg_loss=f"{avg_loss:.4f}",
            total_tokens=total_tokens,
        )
        return perplexity

    def evaluate_generation(
        self,
        dataset: Dataset,
        reference_column: str = "output",
        input_column: str = "text",
        max_new_tokens: int = 256,
    ) -> dict[str, float]:
        """Evaluate generation quality with BLEU and ROUGE.

        Args:
            dataset: Dataset with input and reference columns.
            reference_column: Column containing reference outputs.
            input_column: Column containing input prompts.
            max_new_tokens: Maximum tokens to generate.

        Returns:
            Dict with bleu, rouge_1, rouge_2, rouge_l scores.
        """
        predictions = []
        references = []

        self.model.eval()
        for sample in dataset:
            prompt = sample[input_column]
            reference = sample[reference_column]

            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
            input_ids = inputs["input_ids"].to(self.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            generated = self.tokenizer.decode(
                output_ids[0][input_ids.shape[1] :], skip_special_tokens=True
            )
            predictions.append(generated)
            references.append(reference)

        metrics = self._compute_text_metrics(predictions, references)
        log.info(
            "generation_metrics_computed",
            bleu=f"{metrics.get('bleu', 0):.4f}",
            rouge_l=f"{metrics.get('rouge_l', 0):.4f}",
            num_samples=len(predictions),
        )
        return metrics

    def evaluate_task_accuracy(
        self,
        dataset: Dataset,
        input_column: str = "text",
        label_column: str = "output",
        max_new_tokens: int = 64,
    ) -> float:
        """Evaluate exact-match task accuracy.

        Args:
            dataset: Dataset with input and label columns.
            input_column: Column containing input prompts.
            label_column: Column containing expected labels.
            max_new_tokens: Maximum tokens to generate.

        Returns:
            Accuracy as a float between 0 and 1.
        """
        correct = 0
        total = len(dataset)

        self.model.eval()
        for sample in dataset:
            prompt = sample[input_column]
            expected = sample[label_column].strip().lower()

            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
            input_ids = inputs["input_ids"].to(self.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            generated = self.tokenizer.decode(
                output_ids[0][input_ids.shape[1] :], skip_special_tokens=True
            ).strip().lower()

            if expected in generated or generated in expected:
                correct += 1

        accuracy = correct / total if total > 0 else 0.0
        log.info("task_accuracy_computed", accuracy=f"{accuracy:.2%}", total=total, correct=correct)
        return accuracy

    def evaluate_all(
        self,
        dataset: Dataset,
        model_name: str = "",
        dataset_name: str = "",
        text_column: str = "text",
        max_length: int = 2048,
    ) -> EvaluationResult:
        """Run perplexity evaluation and return full results.

        Args:
            dataset: Dataset to evaluate on.
            model_name: Name for reporting.
            dataset_name: Dataset name for reporting.
            text_column: Text column name.
            max_length: Maximum sequence length.

        Returns:
            EvaluationResult with all computed metrics.
        """
        metrics: dict[str, Any] = {}

        perplexity = self.evaluate_perplexity(
            dataset, text_column=text_column, max_length=max_length
        )
        metrics["perplexity"] = perplexity

        return EvaluationResult(
            metrics=metrics,
            model_name=model_name,
            dataset_name=dataset_name,
            num_samples=len(dataset),
        )

    @staticmethod
    def _compute_text_metrics(
        predictions: list[str], references: list[str]
    ) -> dict[str, float]:
        """Compute BLEU and ROUGE metrics between predictions and references.

        Uses simple n-gram overlap for BLEU and longest common subsequence for ROUGE-L.
        Falls back to basic implementations if evaluate library is not available.
        """
        if not predictions or not references:
            return {"bleu": 0.0, "rouge_1": 0.0, "rouge_2": 0.0, "rouge_l": 0.0}

        bleu = _compute_bleu(predictions, references)
        rouge_scores = _compute_rouge(predictions, references)

        return {
            "bleu": bleu,
            "rouge_1": rouge_scores["rouge_1"],
            "rouge_2": rouge_scores["rouge_2"],
            "rouge_l": rouge_scores["rouge_l"],
        }


def _compute_bleu(predictions: list[str], references: list[str]) -> float:
    """Compute corpus-level BLEU score using simple n-gram overlap."""
    from collections import Counter

    total_score = 0.0
    for pred, ref in zip(predictions, references, strict=True):
        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()
        if not pred_tokens or not ref_tokens:
            continue

        # Unigram precision
        pred_counts = Counter(pred_tokens)
        ref_counts = Counter(ref_tokens)
        clipped = sum(min(pred_counts[w], ref_counts[w]) for w in pred_counts)
        precision = clipped / len(pred_tokens) if pred_tokens else 0.0

        # Brevity penalty
        bp = min(1.0, len(pred_tokens) / len(ref_tokens)) if ref_tokens else 0.0
        total_score += bp * precision

    return total_score / len(predictions) if predictions else 0.0


def _compute_rouge(predictions: list[str], references: list[str]) -> dict[str, float]:
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L F1 scores."""

    rouge_1_scores = []
    rouge_2_scores = []
    rouge_l_scores = []

    for pred, ref in zip(predictions, references, strict=True):
        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()

        # ROUGE-1 (unigram)
        rouge_1_scores.append(_ngram_f1(pred_tokens, ref_tokens, 1))
        # ROUGE-2 (bigram)
        rouge_2_scores.append(_ngram_f1(pred_tokens, ref_tokens, 2))
        # ROUGE-L (longest common subsequence)
        rouge_l_scores.append(_lcs_f1(pred_tokens, ref_tokens))

    return {
        "rouge_1": sum(rouge_1_scores) / len(rouge_1_scores) if rouge_1_scores else 0.0,
        "rouge_2": sum(rouge_2_scores) / len(rouge_2_scores) if rouge_2_scores else 0.0,
        "rouge_l": sum(rouge_l_scores) / len(rouge_l_scores) if rouge_l_scores else 0.0,
    }


def _ngram_f1(pred_tokens: list[str], ref_tokens: list[str], n: int) -> float:
    """Compute n-gram F1 score."""
    from collections import Counter

    def get_ngrams(tokens: list[str], n: int) -> Counter:
        return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))

    pred_ngrams = get_ngrams(pred_tokens, n)
    ref_ngrams = get_ngrams(ref_tokens, n)

    if not pred_ngrams or not ref_ngrams:
        return 0.0

    overlap = sum(min(pred_ngrams[ng], ref_ngrams[ng]) for ng in pred_ngrams)
    precision = overlap / sum(pred_ngrams.values())
    recall = overlap / sum(ref_ngrams.values())

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _lcs_f1(pred_tokens: list[str], ref_tokens: list[str]) -> float:
    """Compute ROUGE-L F1 using longest common subsequence."""
    if not pred_tokens or not ref_tokens:
        return 0.0

    m, n = len(pred_tokens), len(ref_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == ref_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_len = dp[m][n]
    precision = lcs_len / m
    recall = lcs_len / n

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
