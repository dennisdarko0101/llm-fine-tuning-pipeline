"""Quantization and LoRA configuration for QLoRA fine-tuning."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft.peft_model import PeftModel
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, PreTrainedModel

from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.config.training_config import TrainingConfig

log = get_logger(__name__)

_DTYPE_MAP: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def is_gpu_available() -> bool:
    """Check if CUDA GPU is available."""
    return torch.cuda.is_available()


def get_compute_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype to torch dtype.

    Args:
        dtype_str: One of 'bfloat16', 'float16', 'float32'.

    Returns:
        Corresponding torch.dtype.

    Raises:
        ValueError: If dtype string is not recognized.
    """
    dtype = _DTYPE_MAP.get(dtype_str)
    if dtype is None:
        raise ValueError(
            f"Unknown compute dtype: {dtype_str!r}. "
            f"Must be one of {list(_DTYPE_MAP.keys())}"
        )
    return dtype


def create_bnb_config(training_config: TrainingConfig) -> BitsAndBytesConfig:
    """Create BitsAndBytes quantization config from training config.

    Args:
        training_config: Training configuration with quantization params.

    Returns:
        BitsAndBytesConfig for 4-bit NF4 quantization.
    """
    compute_dtype = get_compute_dtype(training_config.bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=training_config.load_in_4bit,
        bnb_4bit_quant_type=training_config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=training_config.bnb_4bit_use_double_quant,
    )

    log.info(
        "bnb_config_created",
        load_in_4bit=training_config.load_in_4bit,
        quant_type=training_config.bnb_4bit_quant_type,
        compute_dtype=training_config.bnb_4bit_compute_dtype,
        double_quant=training_config.bnb_4bit_use_double_quant,
    )
    return bnb_config


def load_quantized_model(
    model_name: str,
    bnb_config: BitsAndBytesConfig | None = None,
    device_map: str = "auto",
    trust_remote_code: bool = False,
    token: str | None = None,
) -> PreTrainedModel:
    """Load a pretrained model with optional quantization.

    Falls back to CPU loading without quantization when no GPU is available.

    Args:
        model_name: HuggingFace model identifier.
        bnb_config: BitsAndBytes config for quantization. Ignored on CPU.
        device_map: Device placement strategy.
        trust_remote_code: Allow custom model code from Hub.
        token: HuggingFace auth token.

    Returns:
        Loaded (possibly quantized) PreTrainedModel.
    """
    gpu_available = is_gpu_available()

    kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "use_cache": False,  # Required for gradient checkpointing
    }

    if token:
        kwargs["token"] = token

    if gpu_available and bnb_config is not None:
        kwargs["quantization_config"] = bnb_config
        kwargs["device_map"] = device_map
        log.info("loading_quantized_model", model=model_name, device_map=device_map)
    else:
        if not gpu_available:
            log.warning(
                "no_gpu_available",
                msg="Loading model on CPU without quantization",
                model=model_name,
            )
            kwargs["device_map"] = "cpu"
            kwargs["torch_dtype"] = torch.float32
        else:
            kwargs["device_map"] = device_map
            log.info("loading_model_no_quantization", model=model_name)

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    # Enable gradient checkpointing for memory efficiency
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        log.info("gradient_checkpointing_enabled")

    # Prepare for k-bit training if quantized
    if gpu_available and bnb_config is not None:
        model = prepare_model_for_kbit_training(model)
        log.info("model_prepared_for_kbit_training")

    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
    log.info(
        "model_loaded",
        model=model_name,
        total_params=f"{total_params:,}",
        model_size_mb=f"{model_size_mb:.1f}",
        gpu=gpu_available,
    )

    return model


def create_lora_config(training_config: TrainingConfig) -> LoraConfig:
    """Create LoRA adapter configuration from training config.

    Args:
        training_config: Training configuration with LoRA params.

    Returns:
        LoraConfig for PEFT.
    """
    lora_config = LoraConfig(
        r=training_config.lora_r,
        lora_alpha=training_config.lora_alpha,
        lora_dropout=training_config.lora_dropout,
        target_modules=training_config.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    log.info(
        "lora_config_created",
        r=training_config.lora_r,
        alpha=training_config.lora_alpha,
        dropout=training_config.lora_dropout,
        target_modules=training_config.target_modules,
    )
    return lora_config


def apply_lora(model: PreTrainedModel, lora_config: LoraConfig) -> PeftModel:
    """Apply LoRA adapters to a model.

    Args:
        model: Base model (possibly quantized).
        lora_config: LoRA configuration.

    Returns:
        PeftModel with LoRA adapters applied.
    """
    peft_model = get_peft_model(model, lora_config)

    # Log trainable parameter stats
    trainable_params, total_params = peft_model.get_nb_trainable_parameters()
    trainable_pct = 100.0 * trainable_params / total_params if total_params > 0 else 0.0

    log.info(
        "lora_applied",
        trainable_params=f"{trainable_params:,}",
        total_params=f"{total_params:,}",
        trainable_pct=f"{trainable_pct:.2f}%",
    )

    return peft_model


def get_trainable_param_stats(model: PreTrainedModel | PeftModel) -> dict[str, Any]:
    """Get trainable parameter statistics for a model.

    Args:
        model: Model to inspect.

    Returns:
        Dict with total_params, trainable_params, trainable_pct, size_mb.
    """
    if hasattr(model, "get_nb_trainable_parameters"):
        trainable, total = model.get_nb_trainable_parameters()
    else:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    pct = 100.0 * trainable / total if total > 0 else 0.0
    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)

    return {
        "total_params": total,
        "trainable_params": trainable,
        "trainable_pct": round(pct, 4),
        "model_size_mb": round(size_mb, 2),
    }
