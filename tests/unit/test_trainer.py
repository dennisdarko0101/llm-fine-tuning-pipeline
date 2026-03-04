"""Tests for the FineTuneTrainer and TrainResult (no GPU required)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset
from transformers import TrainingArguments

from src.config.training_config import TrainingConfig
from src.training.trainer import FineTuneTrainer, TrainResult


class TestTrainResult:
    """Tests for TrainResult dataclass."""

    def test_default_values(self) -> None:
        result = TrainResult()
        assert result.metrics == {}
        assert result.checkpoint_path == ""
        assert result.training_time == 0.0

    def test_train_loss_property(self) -> None:
        result = TrainResult(metrics={"train_loss": 0.5})
        assert result.train_loss == 0.5

    def test_eval_loss_property(self) -> None:
        result = TrainResult(metrics={"eval_loss": 0.3})
        assert result.eval_loss == 0.3

    def test_samples_per_second(self) -> None:
        result = TrainResult(metrics={"train_samples_per_second": 12.5})
        assert result.samples_per_second == 12.5

    def test_missing_metrics_return_none(self) -> None:
        result = TrainResult()
        assert result.train_loss is None
        assert result.eval_loss is None
        assert result.samples_per_second is None

    def test_summary_format(self) -> None:
        result = TrainResult(
            metrics={"train_loss": 0.5, "eval_loss": 0.3, "train_samples_per_second": 10.0},
            checkpoint_path="/tmp/checkpoint",
            training_time=120.5,
        )
        summary = result.summary()
        assert "Training Results" in summary
        assert "120.5s" in summary
        assert "0.5000" in summary
        assert "0.3000" in summary
        assert "10.0" in summary
        assert "/tmp/checkpoint" in summary

    def test_summary_minimal(self) -> None:
        result = TrainResult(training_time=60.0)
        summary = result.summary()
        assert "60.0s" in summary


@pytest.fixture()
def mock_tokenizer() -> MagicMock:
    """Create a mock tokenizer."""
    tokenizer = MagicMock()
    tokenizer.pad_token = None
    tokenizer.eos_token = "<eos>"
    tokenizer.model_max_length = 2048
    return tokenizer


@pytest.fixture()
def mock_model() -> MagicMock:
    """Create a mock model."""
    model = MagicMock()
    model.config = MagicMock()
    model.save_pretrained = MagicMock()
    return model


@pytest.fixture()
def train_dataset() -> Dataset:
    """Create a small training dataset."""
    return Dataset.from_list([
        {"text": f"Training sample {i}: This is a test instruction with a response."}
        for i in range(20)
    ])


@pytest.fixture()
def val_dataset() -> Dataset:
    """Create a small validation dataset."""
    return Dataset.from_list([
        {"text": f"Validation sample {i}: This is a test with output."}
        for i in range(5)
    ])


class TestSetupTrainingArgs:
    """Tests for TrainingArguments creation from config."""

    def test_default_args_creation(
        self, mock_model: MagicMock, mock_tokenizer: MagicMock, train_dataset: Dataset
    ) -> None:
        config = TrainingConfig()
        with patch("src.training.trainer.SFTTrainer"):
            trainer = FineTuneTrainer(
                config, mock_model, mock_tokenizer, train_dataset
            )
        args = trainer.training_args
        assert isinstance(args, TrainingArguments)
        assert args.num_train_epochs == 3
        assert args.per_device_train_batch_size == 4
        assert args.learning_rate == 2e-4
        assert args.lr_scheduler_type == "cosine"
        assert args.warmup_ratio == 0.03

    def test_eval_strategy_no_when_no_val(
        self, mock_model: MagicMock, mock_tokenizer: MagicMock, train_dataset: Dataset
    ) -> None:
        config = TrainingConfig()
        with patch("src.training.trainer.SFTTrainer"):
            trainer = FineTuneTrainer(
                config, mock_model, mock_tokenizer, train_dataset, val_dataset=None
            )
        assert trainer.training_args.eval_strategy.value == "no"

    def test_eval_strategy_steps_when_val(
        self,
        mock_model: MagicMock,
        mock_tokenizer: MagicMock,
        train_dataset: Dataset,
        val_dataset: Dataset,
    ) -> None:
        config = TrainingConfig(eval_strategy="steps")
        with patch("src.training.trainer.SFTTrainer"):
            trainer = FineTuneTrainer(
                config, mock_model, mock_tokenizer, train_dataset, val_dataset=val_dataset
            )
        assert trainer.training_args.eval_strategy.value == "steps"

    def test_output_dir_from_config(
        self, mock_model: MagicMock, mock_tokenizer: MagicMock, train_dataset: Dataset
    ) -> None:
        config = TrainingConfig(output_dir="custom/output")
        with patch("src.training.trainer.SFTTrainer"):
            trainer = FineTuneTrainer(
                config, mock_model, mock_tokenizer, train_dataset
            )
        # Normalize separators for cross-platform compatibility
        assert Path(trainer.training_args.output_dir) == Path("custom/output")

    def test_custom_lr_and_scheduler(
        self, mock_model: MagicMock, mock_tokenizer: MagicMock, train_dataset: Dataset
    ) -> None:
        config = TrainingConfig(learning_rate=1e-5, lr_scheduler_type="linear")
        with patch("src.training.trainer.SFTTrainer"):
            trainer = FineTuneTrainer(
                config, mock_model, mock_tokenizer, train_dataset
            )
        assert trainer.training_args.learning_rate == 1e-5
        assert trainer.training_args.lr_scheduler_type == "linear"

    def test_gradient_accumulation(
        self, mock_model: MagicMock, mock_tokenizer: MagicMock, train_dataset: Dataset
    ) -> None:
        config = TrainingConfig(gradient_accumulation_steps=8)
        with patch("src.training.trainer.SFTTrainer"):
            trainer = FineTuneTrainer(
                config, mock_model, mock_tokenizer, train_dataset
            )
        assert trainer.training_args.gradient_accumulation_steps == 8

    def test_report_to_none(
        self, mock_model: MagicMock, mock_tokenizer: MagicMock, train_dataset: Dataset
    ) -> None:
        config = TrainingConfig(report_to="none")
        with patch("src.training.trainer.SFTTrainer"):
            trainer = FineTuneTrainer(
                config, mock_model, mock_tokenizer, train_dataset
            )
        # "none" is normalized to empty list in newer transformers
        assert trainer.training_args.report_to == [] or trainer.training_args.report_to == ["none"]


class TestTokenizerSetup:
    """Tests for tokenizer configuration."""

    def test_pad_token_set_to_eos(
        self, mock_model: MagicMock, train_dataset: Dataset
    ) -> None:
        tokenizer = MagicMock()
        tokenizer.pad_token = None
        tokenizer.eos_token = "<eos>"
        with patch("src.training.trainer.SFTTrainer"):
            FineTuneTrainer(TrainingConfig(), mock_model, tokenizer, train_dataset)
        assert tokenizer.pad_token == "<eos>"

    def test_existing_pad_token_preserved(
        self, mock_model: MagicMock, train_dataset: Dataset
    ) -> None:
        tokenizer = MagicMock()
        tokenizer.pad_token = "<pad>"
        tokenizer.eos_token = "<eos>"
        with patch("src.training.trainer.SFTTrainer"):
            FineTuneTrainer(TrainingConfig(), mock_model, tokenizer, train_dataset)
        assert tokenizer.pad_token == "<pad>"


class TestSaveModel:
    """Tests for model saving."""

    def test_save_creates_directory(
        self,
        mock_model: MagicMock,
        mock_tokenizer: MagicMock,
        train_dataset: Dataset,
        tmp_path: Path,
    ) -> None:
        config = TrainingConfig()
        with patch("src.training.trainer.SFTTrainer"):
            trainer = FineTuneTrainer(
                config, mock_model, mock_tokenizer, train_dataset
            )
        save_path = tmp_path / "saved_model"
        trainer.save_model(save_path)
        assert save_path.exists()
        mock_model.save_pretrained.assert_called_once_with(str(save_path))
        mock_tokenizer.save_pretrained.assert_called_once_with(str(save_path))


class TestTrainMethod:
    """Tests for the train method using mocks."""

    def test_train_returns_result(
        self,
        mock_model: MagicMock,
        mock_tokenizer: MagicMock,
        train_dataset: Dataset,
        tmp_path: Path,
    ) -> None:
        config = TrainingConfig(output_dir=str(tmp_path / "output"))

        mock_train_output = MagicMock()
        mock_train_output.metrics = {"train_loss": 0.42, "train_samples_per_second": 5.0}

        mock_sft = MagicMock()
        mock_sft.train.return_value = mock_train_output

        with patch("src.training.trainer.SFTTrainer", return_value=mock_sft):
            trainer = FineTuneTrainer(
                config, mock_model, mock_tokenizer, train_dataset
            )
            result = trainer.train()

        assert isinstance(result, TrainResult)
        assert result.train_loss == 0.42
        assert result.training_time > 0
        assert "final" in result.checkpoint_path

    def test_train_with_validation(
        self,
        mock_model: MagicMock,
        mock_tokenizer: MagicMock,
        train_dataset: Dataset,
        val_dataset: Dataset,
        tmp_path: Path,
    ) -> None:
        config = TrainingConfig(output_dir=str(tmp_path / "output"))

        mock_train_output = MagicMock()
        mock_train_output.metrics = {"train_loss": 0.5}

        mock_sft = MagicMock()
        mock_sft.train.return_value = mock_train_output
        mock_sft.evaluate.return_value = {"eval_loss": 0.35}

        with patch("src.training.trainer.SFTTrainer", return_value=mock_sft):
            trainer = FineTuneTrainer(
                config, mock_model, mock_tokenizer, train_dataset, val_dataset=val_dataset
            )
            result = trainer.train()

        assert result.eval_loss == 0.35
        mock_sft.evaluate.assert_called_once()
