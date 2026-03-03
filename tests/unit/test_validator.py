"""Tests for dataset validation."""

from __future__ import annotations

import pytest
from datasets import Dataset

from src.data.validator import DatasetValidator, ValidationReport


class TestValidationReport:
    """Tests for ValidationReport dataclass."""

    def test_removed_count(self) -> None:
        report = ValidationReport(original_count=100, filtered_count=80, reasons={"dup": 20})
        assert report.removed_count == 20

    def test_removal_rate(self) -> None:
        report = ValidationReport(original_count=100, filtered_count=75, reasons={})
        assert report.removal_rate == 0.25

    def test_removal_rate_empty(self) -> None:
        report = ValidationReport(original_count=0, filtered_count=0, reasons={})
        assert report.removal_rate == 0.0

    def test_summary_format(self) -> None:
        report = ValidationReport(
            original_count=100,
            filtered_count=90,
            reasons={"duplicate": 5, "empty_output": 5},
        )
        summary = report.summary()
        assert "Original samples: 100" in summary
        assert "Filtered samples: 90" in summary
        assert "Removed: 10" in summary
        assert "duplicate: 5" in summary


class TestSchemaValidation:
    """Tests for schema validation."""

    def test_valid_schema(self) -> None:
        ds = Dataset.from_list([
            {"instruction": "Do X", "input": "", "output": "Done"},
        ])
        is_valid, missing = DatasetValidator.validate_schema(ds)
        assert is_valid is True
        assert missing == []

    def test_missing_instruction(self) -> None:
        ds = Dataset.from_list([{"input": "", "output": "Done"}])
        is_valid, missing = DatasetValidator.validate_schema(ds)
        assert is_valid is False
        assert "instruction" in missing

    def test_missing_output(self) -> None:
        ds = Dataset.from_list([{"instruction": "Do X", "input": ""}])
        is_valid, missing = DatasetValidator.validate_schema(ds)
        assert is_valid is False
        assert "output" in missing

    def test_missing_both(self) -> None:
        ds = Dataset.from_list([{"input": "", "text": "something"}])
        is_valid, missing = DatasetValidator.validate_schema(ds)
        assert is_valid is False
        assert len(missing) == 2


class TestLengthValidation:
    """Tests for length-based filtering."""

    def test_filter_too_short(self) -> None:
        ds = Dataset.from_list([
            {"instruction": "X", "input": "", "output": "Y"},  # len=2, too short
            {"instruction": "Do something useful", "input": "", "output": "Here is the result"},
        ])
        filtered, removed = DatasetValidator.validate_lengths(ds, min_len=10)
        assert len(filtered) == 1
        assert removed == 1

    def test_filter_too_long(self) -> None:
        ds = Dataset.from_list([
            {"instruction": "X" * 5000, "input": "", "output": "Y" * 5000},  # too long
            {"instruction": "Short", "input": "", "output": "Also short"},
        ])
        filtered, removed = DatasetValidator.validate_lengths(ds, max_len=100)
        assert len(filtered) == 1
        assert removed == 1

    def test_no_filtering_needed(self) -> None:
        ds = Dataset.from_list([
            {"instruction": "Do something", "input": "", "output": "Done it"},
        ])
        filtered, removed = DatasetValidator.validate_lengths(ds, min_len=5, max_len=1000)
        assert len(filtered) == 1
        assert removed == 0


class TestDuplicateValidation:
    """Tests for duplicate detection."""

    def test_remove_exact_duplicates(self) -> None:
        ds = Dataset.from_list([
            {"instruction": "Say hi", "input": "", "output": "Hello!"},
            {"instruction": "Say hi", "input": "", "output": "Hello!"},
            {"instruction": "Say bye", "input": "", "output": "Goodbye!"},
        ])
        filtered, removed = DatasetValidator.validate_duplicates(ds)
        assert len(filtered) == 2
        assert removed == 1

    def test_no_duplicates(self) -> None:
        ds = Dataset.from_list([
            {"instruction": "Say hi", "input": "", "output": "Hello!"},
            {"instruction": "Say bye", "input": "", "output": "Goodbye!"},
        ])
        filtered, removed = DatasetValidator.validate_duplicates(ds)
        assert len(filtered) == 2
        assert removed == 0

    def test_different_output_not_duplicate(self) -> None:
        ds = Dataset.from_list([
            {"instruction": "Say hi", "input": "", "output": "Hello!"},
            {"instruction": "Say hi", "input": "", "output": "Hi there!"},
        ])
        filtered, removed = DatasetValidator.validate_duplicates(ds)
        assert len(filtered) == 2
        assert removed == 0


class TestQualityValidation:
    """Tests for quality checks."""

    def test_empty_instruction_filtered(self) -> None:
        ds = Dataset.from_list([
            {"instruction": "", "input": "", "output": "Hello!"},
            {"instruction": "Say hello", "input": "", "output": "Hello!"},
        ])
        filtered, reasons = DatasetValidator.validate_quality(ds)
        assert len(filtered) == 1
        assert reasons.get("empty_instruction", 0) == 1

    def test_empty_output_filtered(self) -> None:
        ds = Dataset.from_list([
            {"instruction": "Say hello", "input": "", "output": ""},
            {"instruction": "Say bye", "input": "", "output": "Bye!"},
        ])
        filtered, reasons = DatasetValidator.validate_quality(ds)
        assert len(filtered) == 1
        assert reasons.get("empty_output", 0) == 1

    def test_short_instruction_filtered(self) -> None:
        ds = Dataset.from_list([
            {"instruction": "Hi", "input": "", "output": "Hello there!"},
            {"instruction": "Say hello to the world", "input": "", "output": "Hello world!"},
        ])
        filtered, reasons = DatasetValidator.validate_quality(ds)
        assert len(filtered) == 1
        assert reasons.get("instruction_too_short", 0) == 1

    def test_corrupted_text_filtered(self) -> None:
        ds = Dataset.from_list([
            {"instruction": "######!!!!@@@@$$$$%%%%", "input": "", "output": "&&&&****^^^^"},
            {"instruction": "Normal instruction here", "input": "", "output": "Normal output here"},
        ])
        filtered, reasons = DatasetValidator.validate_quality(ds)
        assert len(filtered) == 1
        assert reasons.get("corrupted_text", 0) == 1

    def test_all_good_quality(self) -> None:
        ds = Dataset.from_list([
            {"instruction": "Explain Python lists", "input": "", "output": "A list is a mutable sequence."},
            {"instruction": "What is a function?", "input": "", "output": "A reusable block of code."},
        ])
        filtered, reasons = DatasetValidator.validate_quality(ds)
        assert len(filtered) == 2
        assert len(reasons) == 0


class TestValidateAll:
    """Tests for full validation pipeline."""

    def test_validate_all_good_data(self) -> None:
        ds = Dataset.from_list([
            {"instruction": "Explain Python lists", "input": "", "output": "A list is a mutable ordered sequence in Python."},
            {"instruction": "What is a function?", "input": "", "output": "A function is a reusable block of code."},
        ])
        filtered, report = DatasetValidator.validate_all(ds)
        assert report.original_count == 2
        assert report.filtered_count == 2
        assert report.removed_count == 0

    def test_validate_all_removes_bad_data(self) -> None:
        ds = Dataset.from_list([
            {"instruction": "Good instruction here", "input": "", "output": "Good output here with content"},
            {"instruction": "", "input": "", "output": "No instruction"},
            {"instruction": "Good instruction here", "input": "", "output": "Good output here with content"},  # dup
            {"instruction": "X", "input": "", "output": "Y"},  # too short
        ])
        filtered, report = DatasetValidator.validate_all(ds, min_len=10)
        assert report.original_count == 4
        assert report.filtered_count < 4
        assert len(report.reasons) > 0

    def test_validate_all_missing_schema_raises(self) -> None:
        ds = Dataset.from_list([{"text": "no instruction or output columns"}])
        with pytest.raises(ValueError, match="missing required columns"):
            DatasetValidator.validate_all(ds)

    def test_validate_all_report_summary(self) -> None:
        ds = Dataset.from_list([
            {"instruction": "Good one here", "input": "", "output": "Result here yes"},
        ])
        _, report = DatasetValidator.validate_all(ds)
        summary = report.summary()
        assert "Validation Report" in summary
