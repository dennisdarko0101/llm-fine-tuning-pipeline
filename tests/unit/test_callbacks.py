"""Tests for custom training callbacks (no GPU required)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from transformers import TrainerControl, TrainerState, TrainingArguments

from src.training.callbacks import (
    CheckpointCallback,
    EarlyStoppingCallback,
    LoggingCallback,
)


@pytest.fixture()
def training_args(tmp_path: Path) -> TrainingArguments:
    """Minimal TrainingArguments for testing."""
    return TrainingArguments(
        output_dir=str(tmp_path / "output"),
        logging_steps=10,
        num_train_epochs=1,
        use_cpu=True,
    )


@pytest.fixture()
def state() -> TrainerState:
    """Create a TrainerState with sensible defaults."""
    s = TrainerState()
    s.global_step = 0
    s.max_steps = 100
    s.epoch = 0.0
    return s


@pytest.fixture()
def control() -> TrainerControl:
    """Create a fresh TrainerControl."""
    return TrainerControl()


# ---------------------------------------------------------------
# EarlyStoppingCallback
# ---------------------------------------------------------------


class TestEarlyStoppingCallback:
    """Tests for early stopping behavior."""

    def test_no_stop_when_improving(
        self,
        training_args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
    ) -> None:
        cb = EarlyStoppingCallback(patience=3)
        # Simulating improving losses
        for loss in [1.0, 0.8, 0.6, 0.4]:
            cb.on_evaluate(training_args, state, control, metrics={"eval_loss": loss})
        assert control.should_training_stop is False

    def test_stop_after_patience_exceeded(
        self,
        training_args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
    ) -> None:
        cb = EarlyStoppingCallback(patience=2)
        cb.on_evaluate(training_args, state, control, metrics={"eval_loss": 0.5})
        cb.on_evaluate(training_args, state, control, metrics={"eval_loss": 0.6})
        cb.on_evaluate(training_args, state, control, metrics={"eval_loss": 0.7})
        assert control.should_training_stop is True

    def test_patience_resets_on_improvement(
        self,
        training_args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
    ) -> None:
        cb = EarlyStoppingCallback(patience=2)
        cb.on_evaluate(training_args, state, control, metrics={"eval_loss": 0.5})
        cb.on_evaluate(training_args, state, control, metrics={"eval_loss": 0.6})  # wait 1
        cb.on_evaluate(training_args, state, control, metrics={"eval_loss": 0.3})  # improvement!
        cb.on_evaluate(training_args, state, control, metrics={"eval_loss": 0.4})  # wait 1
        assert control.should_training_stop is False
        assert cb.wait_count == 1

    def test_min_delta_respected(
        self,
        training_args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
    ) -> None:
        cb = EarlyStoppingCallback(patience=2, min_delta=0.01)
        cb.on_evaluate(training_args, state, control, metrics={"eval_loss": 0.50})
        # Tiny improvement below min_delta — should NOT count
        cb.on_evaluate(training_args, state, control, metrics={"eval_loss": 0.499})
        assert cb.wait_count == 1  # Not enough improvement

    def test_no_eval_loss_is_ignored(
        self,
        training_args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
    ) -> None:
        cb = EarlyStoppingCallback(patience=1)
        cb.on_evaluate(training_args, state, control, metrics={})
        assert control.should_training_stop is False
        assert cb.best_loss is None

    def test_none_metrics_ignored(
        self,
        training_args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
    ) -> None:
        cb = EarlyStoppingCallback(patience=1)
        cb.on_evaluate(training_args, state, control, metrics=None)
        assert control.should_training_stop is False


# ---------------------------------------------------------------
# CheckpointCallback
# ---------------------------------------------------------------


class TestCheckpointCallback:
    """Tests for checkpoint management."""

    def test_saves_checkpoint(
        self,
        training_args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        tmp_path: Path,
    ) -> None:
        cb = CheckpointCallback(save_dir=str(tmp_path / "best"), top_k=2)
        model = MagicMock()
        state.global_step = 10

        cb.on_evaluate(
            training_args, state, control, metrics={"eval_loss": 0.5}, model=model
        )
        assert len(cb.checkpoints) == 1
        model.save_pretrained.assert_called_once()

    def test_keeps_only_top_k(
        self,
        training_args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        tmp_path: Path,
    ) -> None:
        save_dir = tmp_path / "best"
        cb = CheckpointCallback(save_dir=str(save_dir), top_k=2)
        model = MagicMock()

        # Create actual directories so cleanup works
        for step, loss in [(10, 0.5), (20, 0.4), (30, 0.3)]:
            state.global_step = step
            ckpt_dir = save_dir / f"checkpoint-step-{step}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            cb.on_evaluate(
                training_args, state, control, metrics={"eval_loss": loss}, model=model
            )

        assert len(cb.checkpoints) == 2
        # Best two should be kept (0.3 and 0.4)
        losses = [c.eval_loss for c in cb.checkpoints]
        assert 0.3 in losses
        assert 0.4 in losses

    def test_does_not_save_worse_than_top_k(
        self,
        training_args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        tmp_path: Path,
    ) -> None:
        save_dir = tmp_path / "best"
        cb = CheckpointCallback(save_dir=str(save_dir), top_k=1)
        model = MagicMock()

        # Save a good checkpoint
        state.global_step = 10
        (save_dir / "checkpoint-step-10").mkdir(parents=True, exist_ok=True)
        cb.on_evaluate(
            training_args, state, control, metrics={"eval_loss": 0.3}, model=model
        )

        # Worse checkpoint should not be saved
        state.global_step = 20
        call_count_before = model.save_pretrained.call_count
        cb.on_evaluate(
            training_args, state, control, metrics={"eval_loss": 0.9}, model=model
        )
        assert model.save_pretrained.call_count == call_count_before
        assert len(cb.checkpoints) == 1

    def test_get_best_checkpoint(
        self,
        training_args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        tmp_path: Path,
    ) -> None:
        save_dir = tmp_path / "best"
        cb = CheckpointCallback(save_dir=str(save_dir), top_k=3)
        model = MagicMock()

        for step, loss in [(10, 0.5), (20, 0.3), (30, 0.4)]:
            state.global_step = step
            (save_dir / f"checkpoint-step-{step}").mkdir(parents=True, exist_ok=True)
            cb.on_evaluate(
                training_args, state, control, metrics={"eval_loss": loss}, model=model
            )

        best = cb.get_best_checkpoint()
        assert best is not None
        assert "step-20" in best

    def test_get_best_checkpoint_empty(self) -> None:
        cb = CheckpointCallback()
        assert cb.get_best_checkpoint() is None

    def test_no_metrics_ignored(
        self,
        training_args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
    ) -> None:
        cb = CheckpointCallback(top_k=2)
        cb.on_evaluate(training_args, state, control, metrics=None)
        assert len(cb.checkpoints) == 0


# ---------------------------------------------------------------
# LoggingCallback
# ---------------------------------------------------------------


class TestLoggingCallback:
    """Tests for structured logging callback."""

    def test_on_train_begin_sets_state(
        self,
        training_args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
    ) -> None:
        cb = LoggingCallback()
        state.max_steps = 200
        cb.on_train_begin(training_args, state, control)
        assert cb._start_time is not None
        assert cb._total_steps == 200

    def test_on_log_with_loss(
        self,
        training_args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
    ) -> None:
        cb = LoggingCallback()
        cb.on_train_begin(training_args, state, control)
        state.global_step = 10
        state.epoch = 0.5
        # Should not raise
        cb.on_log(training_args, state, control, logs={"loss": 0.5, "learning_rate": 1e-4})

    def test_on_evaluate_logs_metrics(
        self,
        training_args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
    ) -> None:
        cb = LoggingCallback()
        state.global_step = 50
        # Should not raise
        cb.on_evaluate(
            training_args, state, control,
            metrics={"eval_loss": 0.35, "eval_runtime": 1.5}
        )

    def test_custom_log_frequency(
        self,
        training_args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
    ) -> None:
        cb = LoggingCallback(log_every_n_steps=5)
        assert cb.log_every_n_steps == 5

    def test_none_logs_ignored(
        self,
        training_args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
    ) -> None:
        cb = LoggingCallback()
        cb.on_train_begin(training_args, state, control)
        # Should not raise
        cb.on_log(training_args, state, control, logs=None)
