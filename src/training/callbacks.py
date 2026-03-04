"""Custom training callbacks for logging, early stopping, and checkpoint management."""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from src.utils.logger import get_logger

log = get_logger(__name__)


class WandbMetricsCallback(TrainerCallback):
    """Log detailed metrics to Weights & Biases at each training step.

    Logs learning rate, loss, gradient norm, and GPU memory usage.
    """

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if logs is None:
            return

        try:
            import torch
            import wandb

            if wandb.run is None:
                return

            metrics: dict[str, Any] = {}

            # Core training metrics
            for key in ("loss", "learning_rate", "grad_norm", "eval_loss"):
                if key in logs:
                    metrics[f"train/{key}"] = logs[key]

            # GPU memory usage
            if torch.cuda.is_available():
                metrics["system/gpu_memory_allocated_mb"] = (
                    torch.cuda.memory_allocated() / (1024**2)
                )
                metrics["system/gpu_memory_reserved_mb"] = (
                    torch.cuda.memory_reserved() / (1024**2)
                )

            if metrics:
                wandb.log(metrics, step=state.global_step)

        except ImportError:
            pass


class EarlyStoppingCallback(TrainerCallback):
    """Stop training when validation loss stops improving.

    Args:
        patience: Number of evaluations to wait for improvement.
        min_delta: Minimum decrease in eval_loss to count as improvement.
    """

    def __init__(self, patience: int = 3, min_delta: float = 0.001) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss: float | None = None
        self.wait_count: int = 0

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if metrics is None:
            return

        eval_loss = metrics.get("eval_loss")
        if eval_loss is None:
            return

        if self.best_loss is None or eval_loss < self.best_loss - self.min_delta:
            self.best_loss = eval_loss
            self.wait_count = 0
            log.info(
                "early_stopping_improved",
                eval_loss=eval_loss,
                best_loss=self.best_loss,
            )
        else:
            self.wait_count += 1
            log.info(
                "early_stopping_no_improvement",
                eval_loss=eval_loss,
                best_loss=self.best_loss,
                wait=self.wait_count,
                patience=self.patience,
            )

            if self.wait_count >= self.patience:
                log.warning(
                    "early_stopping_triggered",
                    patience=self.patience,
                    best_loss=self.best_loss,
                    current_loss=eval_loss,
                )
                control.should_training_stop = True


@dataclass
class _CheckpointEntry:
    """Internal: tracks a saved checkpoint."""

    path: str
    eval_loss: float
    step: int


class CheckpointCallback(TrainerCallback):
    """Save best model checkpoints based on eval_loss, keeping only top-K.

    Args:
        save_dir: Directory for best checkpoints.
        top_k: Number of best checkpoints to keep.
    """

    def __init__(self, save_dir: str = "checkpoints/best", top_k: int = 3) -> None:
        self.save_dir = Path(save_dir)
        self.top_k = top_k
        self.checkpoints: list[_CheckpointEntry] = []

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict[str, Any] | None = None,
        model: Any = None,
        **kwargs: Any,
    ) -> None:
        if metrics is None:
            return

        eval_loss = metrics.get("eval_loss")
        if eval_loss is None:
            return

        # Check if this is a top-K checkpoint
        should_save = len(self.checkpoints) < self.top_k or eval_loss < max(
            c.eval_loss for c in self.checkpoints
        )

        if not should_save:
            return

        # Save checkpoint
        ckpt_path = self.save_dir / f"checkpoint-step-{state.global_step}"
        self.save_dir.mkdir(parents=True, exist_ok=True)

        if model is not None and hasattr(model, "save_pretrained"):
            model.save_pretrained(str(ckpt_path))
            log.info(
                "best_checkpoint_saved",
                path=str(ckpt_path),
                eval_loss=eval_loss,
                step=state.global_step,
            )

        entry = _CheckpointEntry(
            path=str(ckpt_path),
            eval_loss=eval_loss,
            step=state.global_step,
        )
        self.checkpoints.append(entry)

        # Prune to top-K
        if len(self.checkpoints) > self.top_k:
            self.checkpoints.sort(key=lambda c: c.eval_loss)
            to_remove = self.checkpoints[self.top_k :]
            self.checkpoints = self.checkpoints[: self.top_k]

            for old in to_remove:
                old_path = Path(old.path)
                if old_path.exists():
                    shutil.rmtree(old_path)
                    log.info("old_checkpoint_removed", path=old.path)

    def get_best_checkpoint(self) -> str | None:
        """Return path to the best checkpoint by eval_loss."""
        if not self.checkpoints:
            return None
        best = min(self.checkpoints, key=lambda c: c.eval_loss)
        return best.path


class LoggingCallback(TrainerCallback):
    """Structured logging of training progress.

    Logs every N steps with step, loss, learning rate, epoch, and ETA.
    Logs evaluation metrics at each evaluation.

    Args:
        log_every_n_steps: Frequency of progress logging (0 to use trainer's logging_steps).
    """

    def __init__(self, log_every_n_steps: int = 0) -> None:
        self.log_every_n_steps = log_every_n_steps
        self._start_time: float | None = None
        self._total_steps: int = 0

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        self._start_time = time.time()
        self._total_steps = state.max_steps
        log.info(
            "training_begin",
            total_steps=state.max_steps,
            epochs=args.num_train_epochs,
            batch_size=args.per_device_train_batch_size,
            gradient_accumulation=args.gradient_accumulation_steps,
        )

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if logs is None or self._start_time is None:
            return

        step = state.global_step
        freq = self.log_every_n_steps if self.log_every_n_steps > 0 else args.logging_steps
        if step % freq != 0 and step != self._total_steps:
            return

        elapsed = time.time() - self._start_time
        eta = (
            (elapsed / step) * (self._total_steps - step)
            if step > 0 and self._total_steps > 0
            else 0
        )

        log_data: dict[str, Any] = {
            "step": step,
            "total_steps": self._total_steps,
            "epoch": round(state.epoch, 2) if state.epoch is not None else None,
            "elapsed_s": round(elapsed, 1),
            "eta_s": round(eta, 1),
        }

        if "loss" in logs:
            log_data["loss"] = round(logs["loss"], 4)
        if "learning_rate" in logs:
            log_data["lr"] = logs["learning_rate"]
        if "grad_norm" in logs:
            log_data["grad_norm"] = round(logs["grad_norm"], 4)

        log.info("training_progress", **log_data)

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if metrics is None:
            return

        log_data: dict[str, Any] = {"step": state.global_step}
        for key in ("eval_loss", "eval_runtime", "eval_samples_per_second"):
            if key in metrics:
                log_data[key] = metrics[key]

        log.info("evaluation_complete", **log_data)

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        elapsed = time.time() - self._start_time if self._start_time else 0
        log.info(
            "training_end",
            total_steps=state.global_step,
            total_time_s=round(elapsed, 1),
        )
