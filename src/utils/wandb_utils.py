"""Weights & Biases initialization and logging helpers."""

from __future__ import annotations

from typing import Any

import wandb

from src.config.settings import settings
from src.utils.logger import get_logger

log = get_logger(__name__)


def init_wandb(
    run_name: str,
    config: dict[str, Any] | None = None,
    project: str | None = None,
    tags: list[str] | None = None,
) -> wandb.sdk.wandb_run.Run | None:
    """Initialize a Weights & Biases run.

    Args:
        run_name: Name for the run.
        config: Hyperparameters / config dict to log.
        project: W&B project name. Defaults to settings.wandb_project.
        tags: Optional tags for the run.

    Returns:
        The W&B Run object, or None if initialization fails.
    """
    try:
        run = wandb.init(
            project=project or settings.wandb_project,
            entity=settings.wandb_entity,
            name=run_name,
            config=config or {},
            tags=tags or [],
            reinit=True,
        )
        log.info("wandb_initialized", run_name=run_name, run_id=run.id)
        return run
    except Exception:
        log.warning("wandb_init_failed", run_name=run_name, exc_info=True)
        return None


def log_metrics(metrics: dict[str, Any], step: int | None = None) -> None:
    """Log metrics to W&B if a run is active.

    Args:
        metrics: Dictionary of metric names to values.
        step: Optional global step number.
    """
    if wandb.run is not None:
        wandb.log(metrics, step=step)


def log_artifact(
    name: str,
    artifact_type: str,
    path: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Log a file or directory as a W&B artifact.

    Args:
        name: Artifact name.
        artifact_type: Type (e.g., "model", "dataset").
        path: Local path to the artifact file/directory.
        metadata: Optional metadata dict.
    """
    if wandb.run is None:
        log.warning("wandb_not_active", action="log_artifact", name=name)
        return

    artifact = wandb.Artifact(name=name, type=artifact_type, metadata=metadata or {})
    artifact.add_dir(path) if __import__("os").path.isdir(path) else artifact.add_file(path)
    wandb.run.log_artifact(artifact)
    log.info("wandb_artifact_logged", name=name, artifact_type=artifact_type)


def finish_wandb() -> None:
    """Finish the active W&B run."""
    if wandb.run is not None:
        wandb.finish()
        log.info("wandb_finished")
