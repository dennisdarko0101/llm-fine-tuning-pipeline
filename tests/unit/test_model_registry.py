"""Tests for the local model registry (no AWS required)."""

from __future__ import annotations

import json
from pathlib import Path

from src.deployment.model_registry import ModelRecord, ModelRegistry


class TestModelRecord:
    """Tests for ModelRecord dataclass."""

    def test_default_values(self) -> None:
        record = ModelRecord()
        assert record.name == ""
        assert record.version == ""
        assert record.path == ""
        assert record.metrics == {}
        assert record.status == "registered"

    def test_to_dict(self) -> None:
        record = ModelRecord(name="test", version="v1", path="/tmp/model")
        d = record.to_dict()
        assert d["name"] == "test"
        assert d["version"] == "v1"
        assert d["path"] == "/tmp/model"

    def test_from_dict(self) -> None:
        data = {"name": "test", "version": "v2", "path": "/tmp", "metrics": {"loss": 0.5}}
        record = ModelRecord.from_dict(data)
        assert record.name == "test"
        assert record.version == "v2"
        assert record.metrics["loss"] == 0.5

    def test_from_dict_ignores_extra_keys(self) -> None:
        data = {"name": "test", "version": "v1", "unknown_field": "ignored"}
        record = ModelRecord.from_dict(data)
        assert record.name == "test"


class TestModelRegistry:
    """Tests for ModelRegistry CRUD operations."""

    def test_register_model(self, tmp_path: Path) -> None:
        registry = ModelRegistry(tmp_path / "registry.json")
        record = registry.register(
            model_path="/models/v1",
            model_name="my-llm",
            version="v1.0",
            metrics={"eval_loss": 0.5},
        )
        assert record.name == "my-llm"
        assert record.version == "v1.0"
        assert record.metrics["eval_loss"] == 0.5
        assert record.status == "registered"
        assert record.timestamp != ""

    def test_register_persists_to_disk(self, tmp_path: Path) -> None:
        path = tmp_path / "registry.json"
        registry = ModelRegistry(path)
        registry.register("/models/v1", "my-llm", "v1.0")
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert len(data) == 1
        assert data[0]["name"] == "my-llm"

    def test_list_models_empty(self, tmp_path: Path) -> None:
        registry = ModelRegistry(tmp_path / "registry.json")
        assert registry.list_models() == []

    def test_list_models(self, tmp_path: Path) -> None:
        registry = ModelRegistry(tmp_path / "registry.json")
        registry.register("/m1", "model-a", "v1")
        registry.register("/m2", "model-b", "v1")
        registry.register("/m3", "model-a", "v2")
        assert len(registry.list_models()) == 3
        assert len(registry.list_models("model-a")) == 2
        assert len(registry.list_models("model-b")) == 1

    def test_get_latest(self, tmp_path: Path) -> None:
        registry = ModelRegistry(tmp_path / "registry.json")
        registry.register("/m1", "my-llm", "v1.0")
        registry.register("/m2", "my-llm", "v2.0")
        latest = registry.get_latest("my-llm")
        assert latest is not None
        assert latest.version == "v2.0"

    def test_get_latest_not_found(self, tmp_path: Path) -> None:
        registry = ModelRegistry(tmp_path / "registry.json")
        assert registry.get_latest("nonexistent") is None

    def test_get_best_lower_is_better(self, tmp_path: Path) -> None:
        registry = ModelRegistry(tmp_path / "registry.json")
        registry.register("/m1", "my-llm", "v1", metrics={"eval_loss": 0.5})
        registry.register("/m2", "my-llm", "v2", metrics={"eval_loss": 0.3})
        registry.register("/m3", "my-llm", "v3", metrics={"eval_loss": 0.7})
        best = registry.get_best("my-llm", metric="eval_loss")
        assert best is not None
        assert best.version == "v2"
        assert best.metrics["eval_loss"] == 0.3

    def test_get_best_higher_is_better(self, tmp_path: Path) -> None:
        registry = ModelRegistry(tmp_path / "registry.json")
        registry.register("/m1", "my-llm", "v1", metrics={"accuracy": 0.7})
        registry.register("/m2", "my-llm", "v2", metrics={"accuracy": 0.9})
        best = registry.get_best("my-llm", metric="accuracy")
        assert best is not None
        assert best.version == "v2"

    def test_get_best_no_metric(self, tmp_path: Path) -> None:
        registry = ModelRegistry(tmp_path / "registry.json")
        registry.register("/m1", "my-llm", "v1", metrics={"other": 0.5})
        assert registry.get_best("my-llm", metric="eval_loss") is None

    def test_get_best_not_found(self, tmp_path: Path) -> None:
        registry = ModelRegistry(tmp_path / "registry.json")
        assert registry.get_best("nonexistent") is None

    def test_update_status(self, tmp_path: Path) -> None:
        registry = ModelRegistry(tmp_path / "registry.json")
        registry.register("/m1", "my-llm", "v1")
        result = registry.update_status("my-llm", "v1", "deployed")
        assert result is not None
        assert result.status == "deployed"
        # Verify persisted
        registry2 = ModelRegistry(tmp_path / "registry.json")
        record = registry2.get_latest("my-llm")
        assert record is not None
        assert record.status == "deployed"

    def test_update_status_not_found(self, tmp_path: Path) -> None:
        registry = ModelRegistry(tmp_path / "registry.json")
        assert registry.update_status("nonexistent", "v1", "deployed") is None

    def test_delete_model(self, tmp_path: Path) -> None:
        registry = ModelRegistry(tmp_path / "registry.json")
        registry.register("/m1", "my-llm", "v1")
        registry.register("/m2", "my-llm", "v2")
        assert registry.delete("my-llm", "v1") is True
        assert len(registry.list_models()) == 1
        assert registry.list_models()[0].version == "v2"

    def test_delete_not_found(self, tmp_path: Path) -> None:
        registry = ModelRegistry(tmp_path / "registry.json")
        assert registry.delete("nonexistent", "v1") is False

    def test_reload_from_disk(self, tmp_path: Path) -> None:
        path = tmp_path / "registry.json"
        reg1 = ModelRegistry(path)
        reg1.register("/m1", "my-llm", "v1", metrics={"loss": 0.5})
        # Create new instance — should load from disk
        reg2 = ModelRegistry(path)
        assert len(reg2.list_models()) == 1
        assert reg2.list_models()[0].metrics["loss"] == 0.5

    def test_corrupt_registry_resets(self, tmp_path: Path) -> None:
        path = tmp_path / "registry.json"
        path.write_text("not valid json", encoding="utf-8")
        registry = ModelRegistry(path)
        assert registry.list_models() == []
