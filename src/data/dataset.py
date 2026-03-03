"""Dataset loading, formatting, and statistics for fine-tuning pipelines."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from datasets import Dataset

from src.utils.logger import get_logger

log = get_logger(__name__)

# Known format column mappings: format_name -> {unified_col: source_col}
FORMAT_MAPPINGS: dict[str, dict[str, str]] = {
    "alpaca": {"instruction": "instruction", "input": "input", "output": "output"},
    "alpaca_no_input": {"instruction": "instruction", "input": "", "output": "output"},
    "dolly": {"instruction": "instruction", "input": "context", "output": "response"},
    "oasst": {"instruction": "instruction", "input": "", "output": "response"},
    "sharegpt": {"instruction": "conversations", "input": "", "output": ""},
}

# Auto-detection: column sets -> format name
_DETECTION_RULES: list[tuple[set[str], str]] = [
    ({"instruction", "input", "output"}, "alpaca"),
    ({"instruction", "context", "response"}, "dolly"),
    ({"instruction", "response"}, "oasst"),
    ({"conversations"}, "sharegpt"),
    ({"instruction", "output"}, "alpaca_no_input"),
]


class DatasetLoader:
    """Load datasets from various sources into HuggingFace Dataset objects."""

    @staticmethod
    def load_from_huggingface(
        dataset_name: str,
        split: str = "train",
        token: str | None = None,
    ) -> Dataset:
        """Load a dataset from the Hugging Face Hub.

        Args:
            dataset_name: HF dataset identifier (e.g. "tatsu-lab/alpaca").
            split: Dataset split to load.
            token: Optional HF auth token for gated datasets.

        Returns:
            HuggingFace Dataset object.
        """
        from datasets import load_dataset

        log.info("loading_hf_dataset", dataset=dataset_name, split=split)
        ds = load_dataset(dataset_name, split=split, token=token)
        log.info("hf_dataset_loaded", dataset=dataset_name, num_samples=len(ds))
        return ds

    @staticmethod
    def load_from_json(path: str | Path) -> Dataset:
        """Load a dataset from a JSON file.

        Supports JSON files containing a list of objects.

        Args:
            path: Path to the JSON file.

        Returns:
            HuggingFace Dataset object.
        """
        path = Path(path)
        log.info("loading_json_dataset", path=str(path))

        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"Expected JSON array, got {type(data).__name__}")

        ds = Dataset.from_list(data)
        log.info("json_dataset_loaded", path=str(path), num_samples=len(ds))
        return ds

    @staticmethod
    def load_from_csv(path: str | Path) -> Dataset:
        """Load a dataset from a CSV file.

        Args:
            path: Path to the CSV file.

        Returns:
            HuggingFace Dataset object.
        """
        path = Path(path)
        log.info("loading_csv_dataset", path=str(path))

        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        ds = Dataset.from_csv(str(path))
        log.info("csv_dataset_loaded", path=str(path), num_samples=len(ds))
        return ds


class DatasetFormatter:
    """Convert datasets to a unified instruction/input/output format."""

    UNIFIED_COLUMNS = ("instruction", "input", "output")

    @staticmethod
    def detect_format(dataset: Dataset) -> str:
        """Auto-detect dataset format from column names.

        Args:
            dataset: HuggingFace Dataset to inspect.

        Returns:
            Detected format name string.

        Raises:
            ValueError: If format cannot be determined.
        """
        columns = set(dataset.column_names)
        for required_cols, fmt_name in _DETECTION_RULES:
            if required_cols.issubset(columns):
                log.info("format_detected", format=fmt_name, columns=list(columns))
                return fmt_name

        raise ValueError(
            f"Cannot auto-detect format from columns: {sorted(columns)}. "
            f"Supported formats: {list(FORMAT_MAPPINGS.keys())}"
        )

    @classmethod
    def to_unified(cls, dataset: Dataset, format_name: str | None = None) -> Dataset:
        """Convert a dataset to unified format.

        Args:
            dataset: Input dataset in any supported format.
            format_name: Explicit format name; auto-detected if None.

        Returns:
            Dataset with columns: instruction, input, output.
        """
        if format_name is None:
            format_name = cls.detect_format(dataset)

        log.info("converting_to_unified", format=format_name, num_samples=len(dataset))

        if format_name == "sharegpt":
            return cls._convert_sharegpt(dataset)

        mapping = FORMAT_MAPPINGS.get(format_name)
        if mapping is None:
            raise ValueError(f"Unknown format: {format_name}")

        return cls._convert_mapped(dataset, mapping)

    @staticmethod
    def _convert_mapped(dataset: Dataset, mapping: dict[str, str]) -> Dataset:
        """Convert dataset using a column mapping."""

        def transform(example: dict[str, Any]) -> dict[str, str]:
            result: dict[str, str] = {}
            for unified_col, source_col in mapping.items():
                if source_col and source_col in example:
                    result[unified_col] = str(example[source_col] or "")
                else:
                    result[unified_col] = ""
            return result

        converted = dataset.map(transform, remove_columns=dataset.column_names)
        log.info("dataset_converted", num_samples=len(converted))
        return converted

    @staticmethod
    def _convert_sharegpt(dataset: Dataset) -> Dataset:
        """Convert ShareGPT conversation format to unified format."""
        records: list[dict[str, str]] = []
        for example in dataset:
            convos = example.get("conversations", [])
            if not convos or len(convos) < 2:
                continue
            # Take first user/assistant pair
            instruction = ""
            output = ""
            for msg in convos:
                role = msg.get("from", msg.get("role", ""))
                value = msg.get("value", msg.get("content", ""))
                if role in ("human", "user") and not instruction:
                    instruction = str(value)
                elif role in ("gpt", "assistant") and not output:
                    output = str(value)
                if instruction and output:
                    break
            if instruction and output:
                records.append({"instruction": instruction, "input": "", "output": output})

        converted = Dataset.from_list(records)
        log.info("sharegpt_converted", num_samples=len(converted))
        return converted


@dataclass
class DatasetStats:
    """Statistics about a dataset."""

    num_samples: int = 0
    avg_instruction_len: float = 0.0
    avg_input_len: float = 0.0
    avg_output_len: float = 0.0
    max_instruction_len: int = 0
    max_input_len: int = 0
    max_output_len: int = 0
    min_instruction_len: int = 0
    min_input_len: int = 0
    min_output_len: int = 0
    empty_input_count: int = 0
    token_length_distribution: dict[str, int] = field(default_factory=dict)

    @classmethod
    def compute(cls, dataset: Dataset) -> DatasetStats:
        """Compute statistics for a unified-format dataset.

        Args:
            dataset: Dataset with instruction, input, output columns.

        Returns:
            DatasetStats instance with computed metrics.
        """
        log.info("computing_dataset_stats", num_samples=len(dataset))

        if len(dataset) == 0:
            return cls()

        inst_lens = [len(str(x)) for x in dataset["instruction"]]
        inp_lens = [len(str(x)) for x in dataset["input"]]
        out_lens = [len(str(x)) for x in dataset["output"]]

        # Approximate token counts (chars / 4)
        total_lens = [
            (il + inl + ol) // 4 for il, inl, ol in zip(inst_lens, inp_lens, out_lens)
        ]

        # Token length distribution buckets
        buckets = {"0-128": 0, "128-256": 0, "256-512": 0, "512-1024": 0, "1024+": 0}
        for tl in total_lens:
            if tl < 128:
                buckets["0-128"] += 1
            elif tl < 256:
                buckets["128-256"] += 1
            elif tl < 512:
                buckets["256-512"] += 1
            elif tl < 1024:
                buckets["512-1024"] += 1
            else:
                buckets["1024+"] += 1

        empty_inputs = sum(1 for x in dataset["input"] if not str(x).strip())

        stats = cls(
            num_samples=len(dataset),
            avg_instruction_len=sum(inst_lens) / len(inst_lens),
            avg_input_len=sum(inp_lens) / len(inp_lens),
            avg_output_len=sum(out_lens) / len(out_lens),
            max_instruction_len=max(inst_lens),
            max_input_len=max(inp_lens),
            max_output_len=max(out_lens),
            min_instruction_len=min(inst_lens),
            min_input_len=min(inp_lens),
            min_output_len=min(out_lens),
            empty_input_count=empty_inputs,
            token_length_distribution=buckets,
        )
        log.info(
            "dataset_stats_computed",
            num_samples=stats.num_samples,
            avg_output_len=round(stats.avg_output_len, 1),
        )
        return stats
