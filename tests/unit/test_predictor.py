"""Tests for inference predictor (no GPU required)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from src.inference.predictor import GenerationConfig, Predictor, PredictionResult


# ---------------------------------------------------------------
# PredictionResult
# ---------------------------------------------------------------


class TestPredictionResult:
    """Tests for PredictionResult dataclass."""

    def test_default_values(self) -> None:
        result = PredictionResult()
        assert result.prompt == ""
        assert result.generated_text == ""
        assert result.num_tokens == 0
        assert result.generation_time == 0.0

    def test_tokens_per_second(self) -> None:
        result = PredictionResult(num_tokens=100, generation_time=2.0)
        assert result.tokens_per_second == 50.0

    def test_tokens_per_second_zero_time(self) -> None:
        result = PredictionResult(num_tokens=100, generation_time=0.0)
        assert result.tokens_per_second == 0.0

    def test_tokens_per_second_negative_time(self) -> None:
        result = PredictionResult(num_tokens=100, generation_time=-1.0)
        assert result.tokens_per_second == 0.0

    def test_metadata(self) -> None:
        result = PredictionResult(metadata={"model": "test"})
        assert result.metadata["model"] == "test"


# ---------------------------------------------------------------
# GenerationConfig
# ---------------------------------------------------------------


class TestGenerationConfig:
    """Tests for GenerationConfig defaults."""

    def test_defaults(self) -> None:
        config = GenerationConfig()
        assert config.max_new_tokens == 256
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.top_k == 50
        assert config.do_sample is True
        assert config.repetition_penalty == 1.1
        assert config.num_beams == 1

    def test_custom_values(self) -> None:
        config = GenerationConfig(
            max_new_tokens=512, temperature=0.0, do_sample=False, num_beams=4
        )
        assert config.max_new_tokens == 512
        assert config.temperature == 0.0
        assert config.do_sample is False
        assert config.num_beams == 4


# ---------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------


@pytest.fixture()
def mock_model() -> MagicMock:
    """Create a mock model for prediction."""
    model = MagicMock()
    model.eval = MagicMock()
    model.generate.return_value = torch.tensor([[1, 2, 3, 10, 20, 30]])
    return model


@pytest.fixture()
def mock_tokenizer() -> MagicMock:
    """Create a mock tokenizer for prediction."""
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.pad_token = "<pad>"
    tokenizer.eos_token = "<eos>"

    # Mock tokenizer call
    encoded = MagicMock()
    encoded.__getitem__ = MagicMock(side_effect=lambda key: {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }[key])
    encoded.keys = MagicMock(return_value=["input_ids", "attention_mask"])
    tokenizer.return_value = encoded
    tokenizer.decode.return_value = "Hello, this is generated text."
    return tokenizer


class TestPredictor:
    """Tests for Predictor with mocked model."""

    def test_predict_returns_result(
        self, mock_model: MagicMock, mock_tokenizer: MagicMock
    ) -> None:
        predictor = Predictor(mock_model, mock_tokenizer, device="cpu")
        result = predictor.predict("Hello")
        assert isinstance(result, PredictionResult)
        assert result.generated_text == "Hello, this is generated text."
        assert result.num_tokens == 3
        assert result.generation_time > 0
        assert result.prompt == "Hello"

    def test_predict_with_custom_config(
        self, mock_model: MagicMock, mock_tokenizer: MagicMock
    ) -> None:
        predictor = Predictor(mock_model, mock_tokenizer, device="cpu")
        config = GenerationConfig(max_new_tokens=128, temperature=0.0, do_sample=False)
        result = predictor.predict("Test", config=config)
        assert isinstance(result, PredictionResult)

        # Verify generate was called with correct kwargs
        call_kwargs = mock_model.generate.call_args[1]
        assert call_kwargs["max_new_tokens"] == 128
        assert call_kwargs["do_sample"] is False

    def test_predict_batch(
        self, mock_model: MagicMock, mock_tokenizer: MagicMock
    ) -> None:
        # Mock batch output
        mock_model.generate.return_value = torch.tensor([
            [1, 2, 3, 10, 20, 30],
            [1, 2, 3, 40, 50, 60],
        ])
        # Mock attention_mask for batch
        encoded = MagicMock()
        encoded.__getitem__ = MagicMock(side_effect=lambda key: {
            "input_ids": torch.tensor([[1, 2, 3], [1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
        }[key])
        encoded.keys = MagicMock(return_value=["input_ids", "attention_mask"])
        mock_tokenizer.return_value = encoded

        predictor = Predictor(mock_model, mock_tokenizer, device="cpu")
        results = predictor.predict_batch(["Hello", "World"], batch_size=2)
        assert len(results) == 2
        assert all(isinstance(r, PredictionResult) for r in results)

    def test_build_gen_kwargs_sampling(
        self, mock_model: MagicMock, mock_tokenizer: MagicMock
    ) -> None:
        predictor = Predictor(mock_model, mock_tokenizer, device="cpu")
        config = GenerationConfig(
            max_new_tokens=100, temperature=0.8, top_p=0.95, top_k=40, do_sample=True
        )
        kwargs = predictor._build_gen_kwargs(config)
        assert kwargs["max_new_tokens"] == 100
        assert kwargs["temperature"] == 0.8
        assert kwargs["top_p"] == 0.95
        assert kwargs["top_k"] == 40
        assert kwargs["do_sample"] is True

    def test_build_gen_kwargs_greedy(
        self, mock_model: MagicMock, mock_tokenizer: MagicMock
    ) -> None:
        predictor = Predictor(mock_model, mock_tokenizer, device="cpu")
        config = GenerationConfig(do_sample=False)
        kwargs = predictor._build_gen_kwargs(config)
        assert kwargs["do_sample"] is False
        assert "temperature" not in kwargs
        assert "top_p" not in kwargs

    def test_build_gen_kwargs_repetition_penalty(
        self, mock_model: MagicMock, mock_tokenizer: MagicMock
    ) -> None:
        predictor = Predictor(mock_model, mock_tokenizer, device="cpu")
        config = GenerationConfig(repetition_penalty=1.5)
        kwargs = predictor._build_gen_kwargs(config)
        assert kwargs["repetition_penalty"] == 1.5

    def test_build_gen_kwargs_no_repetition_penalty(
        self, mock_model: MagicMock, mock_tokenizer: MagicMock
    ) -> None:
        predictor = Predictor(mock_model, mock_tokenizer, device="cpu")
        config = GenerationConfig(repetition_penalty=1.0)
        kwargs = predictor._build_gen_kwargs(config)
        assert "repetition_penalty" not in kwargs

    def test_build_gen_kwargs_beam_search(
        self, mock_model: MagicMock, mock_tokenizer: MagicMock
    ) -> None:
        predictor = Predictor(mock_model, mock_tokenizer, device="cpu")
        config = GenerationConfig(num_beams=4)
        kwargs = predictor._build_gen_kwargs(config)
        assert kwargs["num_beams"] == 4

    def test_default_config(
        self, mock_model: MagicMock, mock_tokenizer: MagicMock
    ) -> None:
        custom = GenerationConfig(max_new_tokens=512)
        predictor = Predictor(mock_model, mock_tokenizer, device="cpu", default_config=custom)
        assert predictor.default_config.max_new_tokens == 512

    def test_model_set_to_eval(
        self, mock_model: MagicMock, mock_tokenizer: MagicMock
    ) -> None:
        Predictor(mock_model, mock_tokenizer, device="cpu")
        mock_model.eval.assert_called_once()
