"""Application settings loaded from environment variables."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global application settings.

    Loaded from environment variables and .env file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # AWS
    aws_region: str = "us-east-1"
    aws_profile: str | None = None
    s3_bucket: str = ""

    # Weights & Biases
    wandb_project: str = "llm-fine-tuning"
    wandb_entity: str | None = None

    # Hugging Face
    hf_token: str | None = None

    # Model defaults
    model_name: str = "mistralai/Mistral-7B-v0.3"

    # Paths
    output_dir: Path = Path("outputs")
    cache_dir: Path = Path(".cache")

    # Logging
    log_level: str = "INFO"


settings = Settings()
