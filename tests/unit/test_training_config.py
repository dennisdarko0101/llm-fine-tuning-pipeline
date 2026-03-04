"""Tests for TrainingConfig."""

import tempfile
from pathlib import Path

import pytest
import yaml

from src.config.training_config import TrainingConfig


class TestTrainingConfigDefaults:
    def test_defaults_are_valid(self):
        config = TrainingConfig()
        issues = config.validate()
        assert issues == [], f"Default config has issues: {issues}"

    def test_effective_batch_size(self):
        config = TrainingConfig(per_device_train_batch_size=4, gradient_accumulation_steps=4)
        assert config.effective_batch_size == 16

    def test_default_lora_params(self):
        config = TrainingConfig()
        assert config.lora_r == 16
        assert config.lora_alpha == 32
        assert config.lora_dropout == 0.05
        assert len(config.target_modules) == 7

    def test_default_quantization(self):
        config = TrainingConfig()
        assert config.load_in_4bit is True
        assert config.bnb_4bit_quant_type == "nf4"
        assert config.bnb_4bit_compute_dtype == "bfloat16"


class TestTrainingConfigValidation:
    def test_invalid_lora_r(self):
        config = TrainingConfig(lora_r=0)
        issues = config.validate()
        assert any("lora_r" in i for i in issues)

    def test_invalid_dropout(self):
        config = TrainingConfig(lora_dropout=1.5)
        issues = config.validate()
        assert any("lora_dropout" in i for i in issues)

    def test_invalid_quant_type(self):
        config = TrainingConfig(bnb_4bit_quant_type="invalid")
        issues = config.validate()
        assert any("bnb_4bit_quant_type" in i for i in issues)

    def test_invalid_compute_dtype(self):
        config = TrainingConfig(bnb_4bit_compute_dtype="int8")
        issues = config.validate()
        assert any("bnb_4bit_compute_dtype" in i for i in issues)

    def test_invalid_learning_rate(self):
        config = TrainingConfig(learning_rate=5.0)
        issues = config.validate()
        assert any("learning_rate" in i for i in issues)

    def test_both_bf16_fp16(self):
        config = TrainingConfig(bf16=True, fp16=True)
        issues = config.validate()
        assert any("bf16" in i and "fp16" in i for i in issues)

    def test_invalid_scheduler(self):
        config = TrainingConfig(lr_scheduler_type="invalid")
        issues = config.validate()
        assert any("lr_scheduler_type" in i for i in issues)

    def test_empty_model_name(self):
        config = TrainingConfig(model_name="")
        issues = config.validate()
        assert any("model_name" in i for i in issues)

    def test_empty_target_modules(self):
        config = TrainingConfig(target_modules=[])
        issues = config.validate()
        assert any("target_modules" in i for i in issues)

    def test_very_large_seq_length(self):
        config = TrainingConfig(max_seq_length=65536)
        issues = config.validate()
        assert any("max_seq_length" in i for i in issues)


class TestTrainingConfigYaml:
    def test_roundtrip(self):
        original = TrainingConfig(model_name="test/model", lora_r=32, num_epochs=5)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            original.to_yaml(path)
            loaded = TrainingConfig.from_yaml(path)

        assert loaded.model_name == "test/model"
        assert loaded.lora_r == 32
        assert loaded.num_epochs == 5

    def test_from_yaml_ignores_unknown_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            with open(path, "w") as f:
                yaml.dump({"model_name": "test/model", "unknown_key": 42}, f)

            config = TrainingConfig.from_yaml(path)
            assert config.model_name == "test/model"

    def test_from_yaml_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            TrainingConfig.from_yaml("nonexistent.yaml")

    def test_to_dict(self):
        config = TrainingConfig()
        d = config.to_dict()
        assert isinstance(d, dict)
        assert d["lora_r"] == 16
        assert d["model_name"] == "mistralai/Mistral-7B-v0.3"
