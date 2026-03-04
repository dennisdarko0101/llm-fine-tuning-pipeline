"""Tests for quantization and LoRA configuration (no GPU required)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
from peft import LoraConfig
from transformers import BitsAndBytesConfig

from src.config.training_config import TrainingConfig
from src.training.quantization import (
    apply_lora,
    create_bnb_config,
    create_lora_config,
    get_compute_dtype,
    get_trainable_param_stats,
    is_gpu_available,
)


class TestComputeDtype:
    """Tests for dtype string to torch dtype conversion."""

    def test_bfloat16(self) -> None:
        assert get_compute_dtype("bfloat16") == torch.bfloat16

    def test_float16(self) -> None:
        assert get_compute_dtype("float16") == torch.float16

    def test_float32(self) -> None:
        assert get_compute_dtype("float32") == torch.float32

    def test_invalid_dtype_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown compute dtype"):
            get_compute_dtype("int8")


class TestCreateBnbConfig:
    """Tests for BitsAndBytes config creation."""

    def test_default_config(self) -> None:
        config = TrainingConfig()
        bnb = create_bnb_config(config)
        assert isinstance(bnb, BitsAndBytesConfig)
        assert bnb.load_in_4bit is True
        assert bnb.bnb_4bit_quant_type == "nf4"
        assert bnb.bnb_4bit_use_double_quant is True

    def test_fp4_quant_type(self) -> None:
        config = TrainingConfig(bnb_4bit_quant_type="fp4")
        bnb = create_bnb_config(config)
        assert bnb.bnb_4bit_quant_type == "fp4"

    def test_float16_compute_dtype(self) -> None:
        config = TrainingConfig(bnb_4bit_compute_dtype="float16")
        bnb = create_bnb_config(config)
        assert bnb.bnb_4bit_compute_dtype == torch.float16

    def test_no_double_quant(self) -> None:
        config = TrainingConfig(bnb_4bit_use_double_quant=False)
        bnb = create_bnb_config(config)
        assert bnb.bnb_4bit_use_double_quant is False

    def test_invalid_compute_dtype_raises(self) -> None:
        config = TrainingConfig(bnb_4bit_compute_dtype="invalid")
        with pytest.raises(ValueError, match="Unknown compute dtype"):
            create_bnb_config(config)


class TestCreateLoraConfig:
    """Tests for LoRA config creation."""

    def test_default_lora_config(self) -> None:
        config = TrainingConfig()
        lora = create_lora_config(config)
        assert isinstance(lora, LoraConfig)
        assert lora.r == 16
        assert lora.lora_alpha == 32
        assert lora.lora_dropout == 0.05
        assert lora.task_type == "CAUSAL_LM"
        assert lora.bias == "none"

    def test_custom_lora_params(self) -> None:
        config = TrainingConfig(lora_r=8, lora_alpha=16, lora_dropout=0.1)
        lora = create_lora_config(config)
        assert lora.r == 8
        assert lora.lora_alpha == 16
        assert lora.lora_dropout == 0.1

    def test_target_modules(self) -> None:
        config = TrainingConfig(target_modules=["q_proj", "v_proj"])
        lora = create_lora_config(config)
        assert set(lora.target_modules) == {"q_proj", "v_proj"}

    def test_high_rank(self) -> None:
        config = TrainingConfig(lora_r=64, lora_alpha=128)
        lora = create_lora_config(config)
        assert lora.r == 64
        assert lora.lora_alpha == 128


class TestGpuAvailability:
    """Tests for GPU detection."""

    @patch("src.training.quantization.torch.cuda.is_available", return_value=True)
    def test_gpu_available(self, mock_cuda: MagicMock) -> None:
        assert is_gpu_available() is True

    @patch("src.training.quantization.torch.cuda.is_available", return_value=False)
    def test_no_gpu(self, mock_cuda: MagicMock) -> None:
        assert is_gpu_available() is False


class TestApplyLora:
    """Tests for LoRA application using mock model."""

    def test_apply_lora_returns_peft_model(self) -> None:
        """Test that apply_lora wraps model with PEFT."""
        # Create a minimal mock model that get_peft_model can work with
        mock_model = MagicMock()
        mock_model.parameters.return_value = [torch.randn(10, 10)]

        mock_peft = MagicMock()
        mock_peft.get_nb_trainable_parameters.return_value = (1000, 100000)

        with patch("src.training.quantization.get_peft_model", return_value=mock_peft):
            result = apply_lora(mock_model, LoraConfig(r=8, task_type="CAUSAL_LM"))
            assert result is mock_peft


class TestTrainableParamStats:
    """Tests for trainable parameter statistics."""

    def test_stats_with_peft_model(self) -> None:
        mock_model = MagicMock()
        mock_model.get_nb_trainable_parameters.return_value = (50000, 7000000)
        param = torch.randn(100, 100)
        mock_model.parameters.return_value = [param]

        stats = get_trainable_param_stats(mock_model)
        assert stats["trainable_params"] == 50000
        assert stats["total_params"] == 7000000
        assert stats["trainable_pct"] == pytest.approx(0.7143, abs=0.01)
        assert stats["model_size_mb"] > 0

    def test_stats_with_regular_model(self) -> None:
        mock_model = MagicMock(spec=[])  # No get_nb_trainable_parameters

        # Create real tensors — one trainable, one frozen
        trainable = torch.nn.Parameter(torch.randn(50, 50))
        frozen = torch.nn.Parameter(torch.randn(100, 100), requires_grad=False)
        mock_model.parameters = MagicMock(return_value=[trainable, frozen])

        stats = get_trainable_param_stats(mock_model)
        assert stats["trainable_params"] == 2500  # 50*50
        assert stats["total_params"] == 12500  # 50*50 + 100*100
        assert stats["trainable_pct"] == 20.0

    def test_stats_zero_params(self) -> None:
        mock_model = MagicMock(spec=[])
        mock_model.parameters = MagicMock(return_value=[])

        stats = get_trainable_param_stats(mock_model)
        assert stats["total_params"] == 0
        assert stats["trainable_params"] == 0
        assert stats["trainable_pct"] == 0.0
