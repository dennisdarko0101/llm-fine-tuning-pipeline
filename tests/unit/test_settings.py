"""Tests for application settings."""

from src.config.settings import Settings


def test_default_settings():
    s = Settings()
    assert s.aws_region == "us-east-1"
    assert s.model_name == "mistralai/Mistral-7B-v0.3"
    assert s.log_level == "INFO"
    assert s.wandb_project == "llm-fine-tuning"
