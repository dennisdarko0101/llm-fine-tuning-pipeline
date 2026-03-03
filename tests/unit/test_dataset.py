"""Tests for dataset loading, formatting, and statistics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from datasets import Dataset

from src.data.dataset import DatasetFormatter, DatasetLoader, DatasetStats

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "data" / "sample"


class TestDatasetLoaderJSON:
    """Tests for loading datasets from JSON files."""

    def test_load_from_json_alpaca(self) -> None:
        ds = DatasetLoader.load_from_json(SAMPLE_DIR / "sample_alpaca.json")
        assert len(ds) >= 50
        assert "instruction" in ds.column_names
        assert "output" in ds.column_names

    def test_load_from_json_chat(self) -> None:
        ds = DatasetLoader.load_from_json(SAMPLE_DIR / "sample_chat.json")
        assert len(ds) >= 15
        assert "conversations" in ds.column_names

    def test_load_from_json_missing_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            DatasetLoader.load_from_json("/nonexistent/path.json")

    def test_load_from_json_invalid_format(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "bad.json"
        bad_file.write_text('{"not": "a list"}')
        with pytest.raises(ValueError, match="Expected JSON array"):
            DatasetLoader.load_from_json(bad_file)

    def test_load_from_json_custom_data(self, tmp_path: Path) -> None:
        data = [
            {"instruction": "Say hello", "input": "", "output": "Hello!"},
            {"instruction": "Say goodbye", "input": "", "output": "Goodbye!"},
        ]
        path = tmp_path / "test.json"
        path.write_text(json.dumps(data))
        ds = DatasetLoader.load_from_json(path)
        assert len(ds) == 2
        assert ds[0]["instruction"] == "Say hello"


class TestDatasetLoaderCSV:
    """Tests for loading datasets from CSV files."""

    def test_load_from_csv(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "test.csv"
        csv_path.write_text(
            "instruction,input,output\nSay hello,,Hello!\nSay bye,,Bye!\n"
        )
        ds = DatasetLoader.load_from_csv(csv_path)
        assert len(ds) == 2
        assert ds[0]["instruction"] == "Say hello"

    def test_load_from_csv_missing_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            DatasetLoader.load_from_csv("/nonexistent/path.csv")


class TestDatasetFormatterDetection:
    """Tests for format auto-detection."""

    def test_detect_alpaca_format(self) -> None:
        ds = Dataset.from_list([
            {"instruction": "Do X", "input": "Y", "output": "Z"},
        ])
        assert DatasetFormatter.detect_format(ds) == "alpaca"

    def test_detect_dolly_format(self) -> None:
        ds = Dataset.from_list([
            {"instruction": "Do X", "context": "Y", "response": "Z"},
        ])
        assert DatasetFormatter.detect_format(ds) == "dolly"

    def test_detect_oasst_format(self) -> None:
        ds = Dataset.from_list([
            {"instruction": "Do X", "response": "Z"},
        ])
        assert DatasetFormatter.detect_format(ds) == "oasst"

    def test_detect_sharegpt_format(self) -> None:
        ds = Dataset.from_list([
            {"conversations": [{"from": "human", "value": "Hi"}, {"from": "gpt", "value": "Hello"}]},
        ])
        assert DatasetFormatter.detect_format(ds) == "sharegpt"

    def test_detect_unknown_format(self) -> None:
        ds = Dataset.from_list([{"foo": "bar", "baz": "qux"}])
        with pytest.raises(ValueError, match="Cannot auto-detect"):
            DatasetFormatter.detect_format(ds)


class TestDatasetFormatterConversion:
    """Tests for dataset format conversion."""

    def test_convert_alpaca_format(self) -> None:
        ds = Dataset.from_list([
            {"instruction": "Do X", "input": "Y", "output": "Z"},
        ])
        unified = DatasetFormatter.to_unified(ds)
        assert set(unified.column_names) == {"instruction", "input", "output"}
        assert unified[0]["instruction"] == "Do X"
        assert unified[0]["input"] == "Y"
        assert unified[0]["output"] == "Z"

    def test_convert_dolly_format(self) -> None:
        ds = Dataset.from_list([
            {"instruction": "Do X", "context": "Y", "response": "Z"},
        ])
        unified = DatasetFormatter.to_unified(ds)
        assert unified[0]["instruction"] == "Do X"
        assert unified[0]["input"] == "Y"
        assert unified[0]["output"] == "Z"

    def test_convert_oasst_format(self) -> None:
        ds = Dataset.from_list([
            {"instruction": "Do X", "response": "Z"},
        ])
        unified = DatasetFormatter.to_unified(ds)
        assert unified[0]["instruction"] == "Do X"
        assert unified[0]["input"] == ""
        assert unified[0]["output"] == "Z"

    def test_convert_sharegpt_format(self) -> None:
        ds = Dataset.from_list([
            {
                "conversations": [
                    {"from": "human", "value": "Hello"},
                    {"from": "gpt", "value": "Hi there!"},
                ]
            },
        ])
        unified = DatasetFormatter.to_unified(ds)
        assert len(unified) == 1
        assert unified[0]["instruction"] == "Hello"
        assert unified[0]["output"] == "Hi there!"

    def test_convert_sharegpt_skips_incomplete(self) -> None:
        ds = Dataset.from_list([
            {"conversations": [{"from": "human", "value": "Hello"}]},
        ])
        unified = DatasetFormatter.to_unified(ds)
        assert len(unified) == 0

    def test_convert_explicit_format(self) -> None:
        ds = Dataset.from_list([
            {"instruction": "Do X", "context": "Y", "response": "Z"},
        ])
        unified = DatasetFormatter.to_unified(ds, format_name="dolly")
        assert unified[0]["output"] == "Z"

    def test_convert_unknown_format_raises(self) -> None:
        ds = Dataset.from_list([{"instruction": "Do X", "input": "Y", "output": "Z"}])
        with pytest.raises(ValueError, match="Unknown format"):
            DatasetFormatter.to_unified(ds, format_name="nonexistent")

    def test_convert_sample_alpaca_file(self) -> None:
        ds = DatasetLoader.load_from_json(SAMPLE_DIR / "sample_alpaca.json")
        unified = DatasetFormatter.to_unified(ds)
        assert len(unified) >= 50
        assert all(unified[i]["instruction"] for i in range(len(unified)))

    def test_convert_sample_chat_file(self) -> None:
        ds = DatasetLoader.load_from_json(SAMPLE_DIR / "sample_chat.json")
        unified = DatasetFormatter.to_unified(ds)
        assert len(unified) > 0
        assert all(unified[i]["instruction"] for i in range(len(unified)))


class TestDatasetStats:
    """Tests for dataset statistics computation."""

    def test_compute_stats_basic(self) -> None:
        ds = Dataset.from_list([
            {"instruction": "Hello", "input": "", "output": "Hi there!"},
            {"instruction": "Goodbye world", "input": "friend", "output": "Bye!"},
        ])
        stats = DatasetStats.compute(ds)
        assert stats.num_samples == 2
        assert stats.avg_instruction_len > 0
        assert stats.avg_output_len > 0
        assert stats.max_instruction_len == len("Goodbye world")
        assert stats.min_instruction_len == len("Hello")

    def test_compute_stats_empty_dataset(self) -> None:
        ds = Dataset.from_list([])
        stats = DatasetStats.compute(ds)
        assert stats.num_samples == 0

    def test_compute_stats_empty_input_count(self) -> None:
        ds = Dataset.from_list([
            {"instruction": "Say hi", "input": "", "output": "Hi!"},
            {"instruction": "Add", "input": "1+1", "output": "2"},
            {"instruction": "Say bye", "input": "  ", "output": "Bye!"},
        ])
        stats = DatasetStats.compute(ds)
        assert stats.empty_input_count == 2  # empty and whitespace-only

    def test_compute_stats_token_distribution(self) -> None:
        ds = Dataset.from_list([
            {"instruction": "x" * 100, "input": "", "output": "y" * 100},
        ])
        stats = DatasetStats.compute(ds)
        assert isinstance(stats.token_length_distribution, dict)
        assert sum(stats.token_length_distribution.values()) == 1

    def test_compute_stats_sample_file(self) -> None:
        ds = DatasetLoader.load_from_json(SAMPLE_DIR / "sample_alpaca.json")
        stats = DatasetStats.compute(ds)
        assert stats.num_samples >= 50
        assert stats.avg_output_len > stats.avg_instruction_len
