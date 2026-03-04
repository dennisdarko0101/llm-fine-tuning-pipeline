"""Fine-tuning trainer setup using HuggingFace TRL SFTTrainer."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from datasets import Dataset
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainerCallback,
    TrainingArguments,
)
from trl import SFTTrainer

from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.config.training_config import TrainingConfig

log = get_logger(__name__)


@dataclass
class TrainResult:
    """Results from a training run."""

    metrics: dict[str, Any] = field(default_factory=dict)
    checkpoint_path: str = ""
    training_time: float = 0.0

    @property
    def train_loss(self) -> float | None:
        """Extract train loss from metrics."""
        return self.metrics.get("train_loss")

    @property
    def eval_loss(self) -> float | None:
        """Extract eval loss from metrics."""
        return self.metrics.get("eval_loss")

    @property
    def samples_per_second(self) -> float | None:
        """Extract samples per second from metrics."""
        return self.metrics.get("train_samples_per_second")

    def summary(self) -> str:
        """Human-readable summary of training results."""
        lines = [
            "Training Results:",
            f"  Training time: {self.training_time:.1f}s",
            f"  Checkpoint: {self.checkpoint_path}",
        ]
        if self.train_loss is not None:
            lines.append(f"  Train loss: {self.train_loss:.4f}")
        if self.eval_loss is not None:
            lines.append(f"  Eval loss: {self.eval_loss:.4f}")
        if self.samples_per_second is not None:
            lines.append(f"  Samples/sec: {self.samples_per_second:.1f}")
        return "\n".join(lines)


class FineTuneTrainer:
    """Orchestrates QLoRA fine-tuning with SFTTrainer.

    Args:
        training_config: Training hyperparameters.
        model: Pretrained model (possibly quantized with LoRA adapters).
        tokenizer: Tokenizer matching the model.
        train_dataset: Training dataset with 'text' column.
        val_dataset: Optional validation dataset with 'text' column.
        callbacks: Optional list of trainer callbacks.
    """

    def __init__(
        self,
        training_config: TrainingConfig,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        train_dataset: Dataset,
        val_dataset: Dataset | None = None,
        callbacks: list[TrainerCallback] | None = None,
    ) -> None:
        self.config = training_config
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.callbacks = callbacks or []

        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            log.info("pad_token_set_to_eos")

        self.training_args = self.setup_training_args()
        self.trainer = self.setup_trainer()

    def setup_training_args(self) -> TrainingArguments:
        """Create HuggingFace TrainingArguments from training config.

        Returns:
            Configured TrainingArguments.
        """
        cfg = self.config
        output_dir = Path(cfg.output_dir)

        # Determine mixed precision settings based on GPU availability
        gpu_available = torch.cuda.is_available()
        use_bf16 = cfg.bf16 and gpu_available
        use_fp16 = cfg.fp16 and gpu_available
        use_cpu = not gpu_available

        if not gpu_available and (cfg.bf16 or cfg.fp16):
            log.warning("no_gpu_mixed_precision_disabled", bf16=cfg.bf16, fp16=cfg.fp16)

        args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=cfg.num_epochs,
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            per_device_eval_batch_size=cfg.per_device_eval_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            warmup_ratio=cfg.warmup_ratio,
            lr_scheduler_type=cfg.lr_scheduler_type,
            max_grad_norm=cfg.max_grad_norm,
            optim=cfg.optim,
            bf16=use_bf16,
            fp16=use_fp16,
            use_cpu=use_cpu,
            gradient_checkpointing=cfg.gradient_checkpointing if gpu_available else False,
            save_steps=cfg.save_steps,
            eval_steps=cfg.eval_steps,
            eval_strategy=cfg.eval_strategy if self.val_dataset is not None else "no",
            save_total_limit=cfg.save_total_limit,
            logging_steps=cfg.logging_steps,
            logging_dir=str(output_dir / "logs"),
            report_to=cfg.report_to,
            seed=cfg.seed,
            remove_unused_columns=False,
            load_best_model_at_end=self.val_dataset is not None,
        )

        log.info(
            "training_args_created",
            output_dir=str(output_dir),
            epochs=cfg.num_epochs,
            effective_batch_size=cfg.effective_batch_size,
            lr=cfg.learning_rate,
            scheduler=cfg.lr_scheduler_type,
        )
        return args

    def setup_trainer(self) -> SFTTrainer:
        """Create and configure SFTTrainer.

        Returns:
            Configured SFTTrainer ready to train.
        """
        trainer = SFTTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            processing_class=self.tokenizer,
            max_seq_length=self.config.max_seq_length,
            packing=False,
            callbacks=self.callbacks if self.callbacks else None,
        )

        log.info(
            "sft_trainer_created",
            train_samples=len(self.train_dataset),
            val_samples=len(self.val_dataset) if self.val_dataset else 0,
            max_seq_length=self.config.max_seq_length,
            packing=False,
        )
        return trainer

    def train(self) -> TrainResult:
        """Run the training loop.

        Returns:
            TrainResult with metrics and checkpoint path.
        """
        log.info("training_started")
        start_time = time.time()

        train_output = self.trainer.train()
        training_time = time.time() - start_time

        metrics = {**train_output.metrics}

        # Run final evaluation if validation set exists
        if self.val_dataset is not None:
            eval_metrics = self.trainer.evaluate()
            metrics.update(eval_metrics)

        checkpoint_path = str(Path(self.config.output_dir) / "final")
        self.save_model(checkpoint_path)

        result = TrainResult(
            metrics=metrics,
            checkpoint_path=checkpoint_path,
            training_time=training_time,
        )

        log.info(
            "training_completed",
            training_time=f"{training_time:.1f}s",
            train_loss=result.train_loss,
            eval_loss=result.eval_loss,
            checkpoint=checkpoint_path,
        )
        return result

    def save_model(self, output_path: str | Path) -> None:
        """Save model adapter weights and tokenizer.

        Args:
            output_path: Directory to save to.
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(str(output_path))
        self.tokenizer.save_pretrained(str(output_path))

        log.info("model_saved", path=str(output_path))
