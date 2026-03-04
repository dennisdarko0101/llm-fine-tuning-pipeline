"""Dataset validation for quality control and filtering."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from datasets import Dataset

from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class ValidationReport:
    """Report of dataset validation results."""

    original_count: int = 0
    filtered_count: int = 0
    reasons: dict[str, int] = field(default_factory=dict)

    @property
    def removed_count(self) -> int:
        """Number of samples removed during validation."""
        return self.original_count - self.filtered_count

    @property
    def removal_rate(self) -> float:
        """Fraction of samples removed (0.0 to 1.0)."""
        if self.original_count == 0:
            return 0.0
        return self.removed_count / self.original_count

    def summary(self) -> str:
        """Human-readable summary of the validation report."""
        lines = [
            "Validation Report:",
            f"  Original samples: {self.original_count}",
            f"  Filtered samples: {self.filtered_count}",
            f"  Removed: {self.removed_count} ({self.removal_rate:.1%})",
        ]
        if self.reasons:
            lines.append("  Removal reasons:")
            for reason, count in sorted(self.reasons.items(), key=lambda x: -x[1]):
                lines.append(f"    - {reason}: {count}")
        return "\n".join(lines)


class DatasetValidator:
    """Validate and filter datasets for training quality."""

    REQUIRED_COLUMNS = ("instruction", "output")

    @staticmethod
    def validate_schema(dataset: Dataset) -> tuple[bool, list[str]]:
        """Check that required columns exist in the dataset.

        Args:
            dataset: Dataset to validate.

        Returns:
            Tuple of (is_valid, list of missing columns).
        """
        columns = set(dataset.column_names)
        missing = [col for col in DatasetValidator.REQUIRED_COLUMNS if col not in columns]
        is_valid = len(missing) == 0
        if not is_valid:
            log.warning("schema_validation_failed", missing_columns=missing)
        else:
            log.info("schema_validation_passed")
        return is_valid, missing

    @staticmethod
    def validate_lengths(
        dataset: Dataset,
        min_len: int = 10,
        max_len: int = 8192,
    ) -> tuple[Dataset, int]:
        """Filter samples by total character length.

        Args:
            dataset: Dataset to filter.
            min_len: Minimum total character length (instruction + input + output).
            max_len: Maximum total character length.

        Returns:
            Tuple of (filtered dataset, number of removed samples).
        """
        original_count = len(dataset)
        log.info("validating_lengths", min_len=min_len, max_len=max_len)

        def length_ok(example: dict[str, str]) -> bool:
            total = (
                len(str(example.get("instruction", "")))
                + len(str(example.get("input", "")))
                + len(str(example.get("output", "")))
            )
            return min_len <= total <= max_len

        filtered = dataset.filter(length_ok)
        removed = original_count - len(filtered)
        log.info("length_validation_done", removed=removed, remaining=len(filtered))
        return filtered, removed

    @staticmethod
    def validate_duplicates(dataset: Dataset) -> tuple[Dataset, int]:
        """Remove exact duplicate samples based on instruction + output.

        Args:
            dataset: Dataset to deduplicate.

        Returns:
            Tuple of (deduplicated dataset, number of duplicates removed).
        """
        original_count = len(dataset)
        log.info("validating_duplicates", num_samples=original_count)

        seen: set[str] = set()
        keep_indices: list[int] = []

        for i in range(len(dataset)):
            key = (
                str(dataset[i].get("instruction", "")).strip()
                + "|||"
                + str(dataset[i].get("output", "")).strip()
            )
            if key not in seen:
                seen.add(key)
                keep_indices.append(i)

        filtered = dataset.select(keep_indices)
        removed = original_count - len(filtered)
        log.info("duplicate_validation_done", removed=removed, remaining=len(filtered))
        return filtered, removed

    @staticmethod
    def validate_quality(dataset: Dataset) -> tuple[Dataset, dict[str, int]]:
        """Basic quality checks: not empty, not too short, no corrupted text.

        Args:
            dataset: Dataset to validate.

        Returns:
            Tuple of (filtered dataset, dict of reason -> count).
        """
        log.info("validating_quality", num_samples=len(dataset))
        reasons: dict[str, int] = defaultdict(int)
        keep_indices: list[int] = []

        for i in range(len(dataset)):
            sample = dataset[i]
            instruction = str(sample.get("instruction", "")).strip()
            output = str(sample.get("output", "")).strip()

            # Check for empty instruction
            if not instruction:
                reasons["empty_instruction"] += 1
                continue

            # Check for empty output
            if not output:
                reasons["empty_output"] += 1
                continue

            # Check for very short content (likely garbage)
            if len(instruction) < 5:
                reasons["instruction_too_short"] += 1
                continue

            if len(output) < 3:
                reasons["output_too_short"] += 1
                continue

            # Check for corrupted text (excessive special chars)
            combined = instruction + output
            special_ratio = sum(1 for c in combined if not c.isalnum() and not c.isspace()) / max(
                len(combined), 1
            )
            if special_ratio > 0.5:
                reasons["corrupted_text"] += 1
                continue

            keep_indices.append(i)

        filtered = dataset.select(keep_indices)
        reasons_dict = dict(reasons)
        log.info(
            "quality_validation_done",
            removed=len(dataset) - len(filtered),
            reasons=reasons_dict,
        )
        return filtered, reasons_dict

    @classmethod
    def validate_all(
        cls,
        dataset: Dataset,
        min_len: int = 10,
        max_len: int = 8192,
    ) -> tuple[Dataset, ValidationReport]:
        """Run all validation steps and return filtered dataset with report.

        Args:
            dataset: Dataset to validate (must be in unified format).
            min_len: Minimum total character length.
            max_len: Maximum total character length.

        Returns:
            Tuple of (validated dataset, ValidationReport).
        """
        log.info("running_full_validation", num_samples=len(dataset))
        report = ValidationReport(original_count=len(dataset))
        all_reasons: dict[str, int] = {}

        # 1. Schema validation
        is_valid, missing = cls.validate_schema(dataset)
        if not is_valid:
            raise ValueError(f"Dataset missing required columns: {missing}")

        # 2. Quality checks
        dataset, quality_reasons = cls.validate_quality(dataset)
        all_reasons.update(quality_reasons)

        # 3. Length filtering
        dataset, length_removed = cls.validate_lengths(dataset, min_len, max_len)
        if length_removed > 0:
            all_reasons["length_out_of_range"] = length_removed

        # 4. Duplicate removal
        dataset, dup_removed = cls.validate_duplicates(dataset)
        if dup_removed > 0:
            all_reasons["duplicate"] = dup_removed

        report.filtered_count = len(dataset)
        report.reasons = all_reasons

        log.info(
            "full_validation_complete",
            original=report.original_count,
            filtered=report.filtered_count,
            removed=report.removed_count,
        )
        return dataset, report
