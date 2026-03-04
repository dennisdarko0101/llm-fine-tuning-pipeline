"""JSON-based local model registry for tracking model versions."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.utils.logger import get_logger

log = get_logger(__name__)

_DEFAULT_REGISTRY_PATH = Path("outputs/.model_registry.json")


@dataclass
class ModelRecord:
    """A single registered model version."""

    name: str = ""
    version: str = ""
    path: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    status: str = "registered"  # registered | deployed | archived

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelRecord:
        """Deserialize from dict."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class ModelRegistry:
    """File-based model registry for tracking model versions and metrics.

    Stores records in a JSON file for simplicity — no database needed.

    Args:
        registry_path: Path to the registry JSON file.
    """

    def __init__(self, registry_path: str | Path | None = None) -> None:
        self.registry_path = Path(registry_path) if registry_path else _DEFAULT_REGISTRY_PATH
        self._records: list[ModelRecord] = []
        self._load()

    def _load(self) -> None:
        """Load registry from disk."""
        if self.registry_path.exists():
            try:
                data = json.loads(self.registry_path.read_text(encoding="utf-8"))
                self._records = [ModelRecord.from_dict(r) for r in data]
                log.info("registry_loaded", path=str(self.registry_path), count=len(self._records))
            except (json.JSONDecodeError, KeyError):
                log.warning("registry_corrupt_resetting", path=str(self.registry_path))
                self._records = []
        else:
            self._records = []

    def _save(self) -> None:
        """Persist registry to disk."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        data = [r.to_dict() for r in self._records]
        self.registry_path.write_text(
            json.dumps(data, indent=2, default=str), encoding="utf-8"
        )

    def register(
        self,
        model_path: str,
        model_name: str,
        version: str,
        metrics: dict[str, Any] | None = None,
    ) -> ModelRecord:
        """Register a new model version.

        Args:
            model_path: Path to the model files.
            model_name: Model name identifier.
            version: Version string (e.g., "v1.0", "20240101").
            metrics: Training/evaluation metrics.

        Returns:
            The created ModelRecord.
        """
        record = ModelRecord(
            name=model_name,
            version=version,
            path=model_path,
            metrics=metrics or {},
            timestamp=datetime.now(UTC).isoformat(),
            status="registered",
        )
        self._records.append(record)
        self._save()
        log.info("model_registered", name=model_name, version=version, path=model_path)
        return record

    def get_latest(self, model_name: str) -> ModelRecord | None:
        """Get the most recently registered version of a model.

        Args:
            model_name: Model name to look up.

        Returns:
            Latest ModelRecord, or None if not found.
        """
        matches = [r for r in self._records if r.name == model_name]
        if not matches:
            return None
        return max(matches, key=lambda r: r.timestamp)

    def get_best(
        self, model_name: str, metric: str = "eval_loss"
    ) -> ModelRecord | None:
        """Get the best model version by a metric.

        Args:
            model_name: Model name to look up.
            metric: Metric name to compare by.

        Returns:
            Best ModelRecord, or None if not found.
        """
        matches = [
            r for r in self._records
            if r.name == model_name and metric in r.metrics
        ]
        if not matches:
            return None

        lower_is_better = metric in ("eval_loss", "train_loss", "perplexity", "loss")
        if lower_is_better:
            return min(matches, key=lambda r: r.metrics[metric])
        return max(matches, key=lambda r: r.metrics[metric])

    def list_models(self, model_name: str | None = None) -> list[ModelRecord]:
        """List all registered models, optionally filtered by name.

        Args:
            model_name: Optional filter by model name.

        Returns:
            List of ModelRecord objects.
        """
        if model_name:
            return [r for r in self._records if r.name == model_name]
        return list(self._records)

    def update_status(self, model_name: str, version: str, status: str) -> ModelRecord | None:
        """Update the status of a model version.

        Args:
            model_name: Model name.
            version: Version string.
            status: New status (registered, deployed, archived).

        Returns:
            Updated ModelRecord, or None if not found.
        """
        for record in self._records:
            if record.name == model_name and record.version == version:
                record.status = status
                self._save()
                log.info("model_status_updated", name=model_name, version=version, status=status)
                return record
        return None

    def delete(self, model_name: str, version: str) -> bool:
        """Delete a model record.

        Args:
            model_name: Model name.
            version: Version string.

        Returns:
            True if deleted, False if not found.
        """
        before = len(self._records)
        self._records = [
            r for r in self._records
            if not (r.name == model_name and r.version == version)
        ]
        if len(self._records) < before:
            self._save()
            log.info("model_deleted", name=model_name, version=version)
            return True
        return False
