"""Model loading utilities for inference (base, fine-tuned, merged)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from src.utils.logger import get_logger

log = get_logger(__name__)


class ModelLoader:
    """Load models for inference — base, fine-tuned (LoRA), or merged.

    Supports loading from HuggingFace Hub or local checkpoint directories.
    """

    @staticmethod
    def load_base_model(
        model_name: str,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        trust_remote_code: bool = False,
        token: str | None = None,
    ) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        """Load a base pretrained model and tokenizer.

        Args:
            model_name: HuggingFace model identifier or local path.
            device_map: Device placement strategy.
            torch_dtype: Torch dtype for model weights.
            trust_remote_code: Allow custom model code.
            token: HuggingFace auth token.

        Returns:
            Tuple of (model, tokenizer).
        """
        dtype = _resolve_dtype(torch_dtype)
        kwargs: dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
        }
        if dtype is not None:
            kwargs["torch_dtype"] = dtype

        if not torch.cuda.is_available():
            kwargs["device_map"] = "cpu"
        else:
            kwargs["device_map"] = device_map

        if token:
            kwargs["token"] = token

        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code, token=token
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        log.info("base_model_loaded", model=model_name)
        return model, tokenizer

    @staticmethod
    def load_finetuned(
        base_model_name: str,
        adapter_path: str | Path,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        token: str | None = None,
    ) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        """Load a base model with LoRA adapters applied.

        Args:
            base_model_name: Base model identifier.
            adapter_path: Path to saved LoRA adapter weights.
            device_map: Device placement strategy.
            torch_dtype: Torch dtype.
            token: HuggingFace auth token.

        Returns:
            Tuple of (peft_model, tokenizer).
        """
        from peft import PeftModel

        dtype = _resolve_dtype(torch_dtype)
        kwargs: dict[str, Any] = {}
        if dtype is not None:
            kwargs["torch_dtype"] = dtype

        if not torch.cuda.is_available():
            kwargs["device_map"] = "cpu"
        else:
            kwargs["device_map"] = device_map

        if token:
            kwargs["token"] = token

        base_model = AutoModelForCausalLM.from_pretrained(base_model_name, **kwargs)
        model = PeftModel.from_pretrained(base_model, str(adapter_path))
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        log.info(
            "finetuned_model_loaded",
            base_model=base_model_name,
            adapter_path=str(adapter_path),
        )
        return model, tokenizer

    @staticmethod
    def load_merged(
        base_model_name: str,
        adapter_path: str | Path,
        output_path: str | Path | None = None,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        token: str | None = None,
    ) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        """Load and merge LoRA adapters into the base model.

        Args:
            base_model_name: Base model identifier.
            adapter_path: Path to LoRA adapter weights.
            output_path: If provided, save the merged model here.
            device_map: Device placement strategy.
            torch_dtype: Torch dtype.
            token: HuggingFace auth token.

        Returns:
            Tuple of (merged_model, tokenizer).
        """
        from peft import PeftModel

        dtype = _resolve_dtype(torch_dtype)
        kwargs: dict[str, Any] = {}
        if dtype is not None:
            kwargs["torch_dtype"] = dtype

        if not torch.cuda.is_available():
            kwargs["device_map"] = "cpu"
        else:
            kwargs["device_map"] = device_map

        if token:
            kwargs["token"] = token

        base_model = AutoModelForCausalLM.from_pretrained(base_model_name, **kwargs)
        peft_model = PeftModel.from_pretrained(base_model, str(adapter_path))
        merged_model = peft_model.merge_and_unload()
        merged_model.eval()

        tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if output_path is not None:
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            merged_model.save_pretrained(str(output_path))
            tokenizer.save_pretrained(str(output_path))
            log.info("merged_model_saved", path=str(output_path))

        log.info(
            "merged_model_loaded",
            base_model=base_model_name,
            adapter_path=str(adapter_path),
        )
        return merged_model, tokenizer

    @staticmethod
    def load_from_checkpoint(
        checkpoint_path: str | Path,
        device_map: str = "auto",
        torch_dtype: str = "auto",
    ) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        """Load model and tokenizer from a local checkpoint directory.

        Args:
            checkpoint_path: Path to saved model directory.
            device_map: Device placement strategy.
            torch_dtype: Torch dtype.

        Returns:
            Tuple of (model, tokenizer).
        """
        checkpoint_path = Path(checkpoint_path)
        dtype = _resolve_dtype(torch_dtype)
        kwargs: dict[str, Any] = {}
        if dtype is not None:
            kwargs["torch_dtype"] = dtype

        if not torch.cuda.is_available():
            kwargs["device_map"] = "cpu"
        else:
            kwargs["device_map"] = device_map

        model = AutoModelForCausalLM.from_pretrained(str(checkpoint_path), **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path))

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        log.info("checkpoint_loaded", path=str(checkpoint_path))
        return model, tokenizer


def _resolve_dtype(dtype_str: str) -> torch.dtype | None:
    """Resolve dtype string to torch.dtype, or None for 'auto'."""
    if dtype_str == "auto":
        return None
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(dtype_str)
    if dtype is None:
        raise ValueError(f"Unknown torch_dtype: {dtype_str!r}. Use 'auto', 'float16', 'bfloat16', or 'float32'.")
    return dtype
