"""Tests for model evaluation (no GPU required)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
from datasets import Dataset

from src.evaluation.evaluator import (
    EvaluationResult,
    ModelEvaluator,
    _compute_bleu,
    _compute_rouge,
    _lcs_f1,
    _ngram_f1,
)

# ---------------------------------------------------------------
# EvaluationResult
# ---------------------------------------------------------------


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""

    def test_default_values(self) -> None:
        result = EvaluationResult()
        assert result.metrics == {}
        assert result.model_name == ""
        assert result.dataset_name == ""
        assert result.num_samples == 0

    def test_perplexity_property(self) -> None:
        result = EvaluationResult(metrics={"perplexity": 15.5})
        assert result.perplexity == 15.5

    def test_bleu_property(self) -> None:
        result = EvaluationResult(metrics={"bleu": 0.42})
        assert result.bleu == 0.42

    def test_rouge_l_property(self) -> None:
        result = EvaluationResult(metrics={"rouge_l": 0.65})
        assert result.rouge_l == 0.65

    def test_accuracy_property(self) -> None:
        result = EvaluationResult(metrics={"accuracy": 0.85})
        assert result.accuracy == 0.85

    def test_missing_metrics_return_none(self) -> None:
        result = EvaluationResult()
        assert result.perplexity is None
        assert result.bleu is None
        assert result.rouge_l is None
        assert result.accuracy is None

    def test_summary_format(self) -> None:
        result = EvaluationResult(
            metrics={"perplexity": 15.5, "bleu": 0.42, "rouge_l": 0.65, "accuracy": 0.85},
            model_name="test-model",
            dataset_name="test-data",
            num_samples=100,
        )
        summary = result.summary()
        assert "test-model" in summary
        assert "test-data" in summary
        assert "100" in summary
        assert "15.50" in summary
        assert "0.4200" in summary
        assert "0.6500" in summary
        assert "85.00%" in summary

    def test_summary_minimal(self) -> None:
        result = EvaluationResult(model_name="m", num_samples=10)
        summary = result.summary()
        assert "Evaluation Results" in summary
        assert "10" in summary


# ---------------------------------------------------------------
# Text metrics helpers
# ---------------------------------------------------------------


class TestBleuComputation:
    """Tests for BLEU score computation."""

    def test_identical_texts(self) -> None:
        preds = ["the cat sat on the mat"]
        refs = ["the cat sat on the mat"]
        score = _compute_bleu(preds, refs)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_completely_different(self) -> None:
        preds = ["hello world"]
        refs = ["foo bar baz"]
        score = _compute_bleu(preds, refs)
        assert score == 0.0

    def test_partial_overlap(self) -> None:
        preds = ["the cat sat"]
        refs = ["the cat ran away"]
        score = _compute_bleu(preds, refs)
        assert 0.0 < score < 1.0

    def test_empty_inputs(self) -> None:
        assert _compute_bleu([], []) == 0.0
        assert _compute_bleu(["hello"], [""]) == 0.0


class TestRougeComputation:
    """Tests for ROUGE score computation."""

    def test_identical_texts(self) -> None:
        preds = ["the cat sat on the mat"]
        refs = ["the cat sat on the mat"]
        scores = _compute_rouge(preds, refs)
        assert scores["rouge_1"] == pytest.approx(1.0, abs=0.01)
        assert scores["rouge_l"] == pytest.approx(1.0, abs=0.01)

    def test_partial_overlap(self) -> None:
        preds = ["the cat sat on a mat"]
        refs = ["the cat sat on the mat"]
        scores = _compute_rouge(preds, refs)
        assert 0.5 < scores["rouge_1"] < 1.0
        assert 0.5 < scores["rouge_l"] < 1.0

    def test_empty_inputs(self) -> None:
        scores = _compute_rouge([], [])
        assert scores["rouge_1"] == 0.0


class TestNgramF1:
    """Tests for n-gram F1 helper."""

    def test_unigram_identical(self) -> None:
        tokens = ["a", "b", "c"]
        assert _ngram_f1(tokens, tokens, 1) == pytest.approx(1.0)

    def test_bigram_identical(self) -> None:
        tokens = ["a", "b", "c", "d"]
        assert _ngram_f1(tokens, tokens, 2) == pytest.approx(1.0)

    def test_no_overlap(self) -> None:
        assert _ngram_f1(["a", "b"], ["c", "d"], 1) == 0.0

    def test_empty_tokens(self) -> None:
        assert _ngram_f1([], ["a"], 1) == 0.0
        assert _ngram_f1(["a"], [], 1) == 0.0


class TestLcsF1:
    """Tests for LCS-based ROUGE-L."""

    def test_identical(self) -> None:
        tokens = ["a", "b", "c", "d"]
        assert _lcs_f1(tokens, tokens) == pytest.approx(1.0)

    def test_no_overlap(self) -> None:
        assert _lcs_f1(["a", "b"], ["c", "d"]) == 0.0

    def test_partial_overlap(self) -> None:
        pred = ["a", "b", "c"]
        ref = ["a", "c", "d"]
        score = _lcs_f1(pred, ref)
        assert 0.0 < score < 1.0

    def test_empty(self) -> None:
        assert _lcs_f1([], ["a"]) == 0.0
        assert _lcs_f1(["a"], []) == 0.0


# ---------------------------------------------------------------
# ModelEvaluator
# ---------------------------------------------------------------


@pytest.fixture()
def mock_model() -> MagicMock:
    """Create a mock model for evaluation."""
    model = MagicMock()
    model.eval = MagicMock()
    # Mock forward pass returning loss
    output = MagicMock()
    output.loss = MagicMock()
    output.loss.item.return_value = 2.0
    model.return_value = output
    # Mock generate
    model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
    return model


@pytest.fixture()
def mock_tokenizer() -> MagicMock:
    """Create a mock tokenizer for evaluation."""
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    # Mock tokenizer call
    encoded = MagicMock()
    encoded.__getitem__ = MagicMock(side_effect=lambda key: {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }[key])
    encoded.keys = MagicMock(return_value=["input_ids", "attention_mask"])
    tokenizer.return_value = encoded
    tokenizer.decode.return_value = "generated response"
    return tokenizer


class TestModelEvaluator:
    """Tests for ModelEvaluator with mocked model."""

    def test_evaluate_perplexity(
        self, mock_model: MagicMock, mock_tokenizer: MagicMock
    ) -> None:
        dataset = Dataset.from_list([{"text": f"sample {i}"} for i in range(4)])
        evaluator = ModelEvaluator(mock_model, mock_tokenizer, device="cpu")
        ppl = evaluator.evaluate_perplexity(dataset, batch_size=2)
        assert ppl > 0
        assert isinstance(ppl, float)

    def test_evaluate_task_accuracy(
        self, mock_model: MagicMock, mock_tokenizer: MagicMock
    ) -> None:
        dataset = Dataset.from_list([
            {"text": "Q: Is 2+2=4?", "output": "generated response"},
            {"text": "Q: Color of sky?", "output": "blue"},
        ])
        evaluator = ModelEvaluator(mock_model, mock_tokenizer, device="cpu")
        accuracy = evaluator.evaluate_task_accuracy(dataset)
        assert 0.0 <= accuracy <= 1.0

    def test_evaluate_all(
        self, mock_model: MagicMock, mock_tokenizer: MagicMock
    ) -> None:
        dataset = Dataset.from_list([{"text": f"sample {i}"} for i in range(4)])
        evaluator = ModelEvaluator(mock_model, mock_tokenizer, device="cpu")
        result = evaluator.evaluate_all(
            dataset, model_name="test", dataset_name="test-data"
        )
        assert isinstance(result, EvaluationResult)
        assert result.model_name == "test"
        assert result.num_samples == 4
        assert result.perplexity is not None

    def test_compute_text_metrics_empty(self) -> None:
        metrics = ModelEvaluator._compute_text_metrics([], [])
        assert metrics["bleu"] == 0.0
        assert metrics["rouge_l"] == 0.0

    def test_compute_text_metrics_with_data(self) -> None:
        preds = ["the cat sat on the mat"]
        refs = ["the cat sat on the mat"]
        metrics = ModelEvaluator._compute_text_metrics(preds, refs)
        assert metrics["bleu"] > 0
        assert metrics["rouge_1"] > 0
        assert metrics["rouge_l"] > 0
