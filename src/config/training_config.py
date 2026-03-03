"""Training configuration with QLoRA defaults and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TrainingConfig:
    """Full training configuration for QLoRA fine-tuning.

    Provides sensible defaults for Mistral-7B / Llama-3-8B QLoRA training.
    Load from YAML with `from_yaml()` or override individual fields.
    """

    # --- Model ---
    model_name: str = "mistralai/Mistral-7B-v0.3"
    dataset_name: str = ""
    dataset_split: str = "train"
    output_dir: str = "outputs"

    # --- LoRA ---
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    # --- Quantization (QLoRA) ---
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True

    # --- Training ---
    num_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    max_seq_length: int = 2048
    max_grad_norm: float = 0.3
    optim: str = "paged_adamw_32bit"

    # --- Save / Eval ---
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 3
    logging_steps: int = 10
    eval_strategy: str = "steps"

    # --- Misc ---
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    group_by_length: bool = True
    seed: int = 42
    report_to: str = "wandb"

    # --- Prompt ---
    prompt_template: str = ""

    # ------------------------------------------------------------------
    # Factory: load from YAML
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainingConfig:
        """Load configuration from a YAML file.

        Keys in the YAML are mapped directly to dataclass fields.
        Unknown keys are silently ignored so YAML files can contain
        comments / metadata without breaking the loader.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}

        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in raw.items() if k in valid_fields}
        return cls(**filtered)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> list[str]:
        """Validate configuration values.

        Returns a list of warning/error messages. An empty list means
        the configuration is valid.
        """
        issues: list[str] = []

        # LoRA checks
        if self.lora_r < 1 or self.lora_r > 256:
            issues.append(f"lora_r={self.lora_r} outside reasonable range [1, 256]")
        if self.lora_alpha < 1:
            issues.append(f"lora_alpha={self.lora_alpha} must be >= 1")
        if not (0.0 <= self.lora_dropout < 1.0):
            issues.append(f"lora_dropout={self.lora_dropout} must be in [0.0, 1.0)")
        if not self.target_modules:
            issues.append("target_modules is empty — no layers will be adapted")

        # Quantization checks
        if self.bnb_4bit_quant_type not in ("nf4", "fp4"):
            issues.append(
                f"bnb_4bit_quant_type='{self.bnb_4bit_quant_type}' must be 'nf4' or 'fp4'"
            )
        if self.bnb_4bit_compute_dtype not in ("bfloat16", "float16", "float32"):
            issues.append(
                f"bnb_4bit_compute_dtype='{self.bnb_4bit_compute_dtype}' "
                "must be 'bfloat16', 'float16', or 'float32'"
            )

        # Training checks
        if self.num_epochs < 1 or self.num_epochs > 100:
            issues.append(f"num_epochs={self.num_epochs} outside reasonable range [1, 100]")
        if self.per_device_train_batch_size < 1:
            issues.append("per_device_train_batch_size must be >= 1")
        if self.gradient_accumulation_steps < 1:
            issues.append("gradient_accumulation_steps must be >= 1")
        if not (1e-7 <= self.learning_rate <= 1.0):
            issues.append(f"learning_rate={self.learning_rate} outside range [1e-7, 1.0]")
        if not (0.0 <= self.warmup_ratio <= 1.0):
            issues.append(f"warmup_ratio={self.warmup_ratio} must be in [0.0, 1.0]")
        if self.max_seq_length < 32:
            issues.append(f"max_seq_length={self.max_seq_length} is very small (< 32)")
        if self.max_seq_length > 32768:
            issues.append(
                f"max_seq_length={self.max_seq_length} is very large — check GPU memory"
            )

        # Scheduler
        valid_schedulers = {
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        }
        if self.lr_scheduler_type not in valid_schedulers:
            issues.append(
                f"lr_scheduler_type='{self.lr_scheduler_type}' not in {valid_schedulers}"
            )

        # Mixed precision
        if self.bf16 and self.fp16:
            issues.append("bf16 and fp16 cannot both be True")

        # Save / eval
        if self.save_total_limit < 1:
            issues.append("save_total_limit must be >= 1")

        # Model name
        if not self.model_name:
            issues.append("model_name is required")

        return issues

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def effective_batch_size(self) -> int:
        """Effective batch size = per-device batch * gradient accumulation."""
        return self.per_device_train_batch_size * self.gradient_accumulation_steps

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict (useful for W&B logging)."""
        from dataclasses import asdict

        return asdict(self)

    def to_yaml(self, path: str | Path) -> None:
        """Write configuration to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
