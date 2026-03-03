#!/usr/bin/env python3
"""Dataset preparation CLI: load → validate → format → split → save.

Usage:
    python scripts/prepare_data.py --source data/sample/sample_alpaca.json --template alpaca
    python scripts/prepare_data.py --source data/sample/sample_chat.json --template chatml
    python scripts/prepare_data.py --source tatsu-lab/alpaca --hf --template alpaca
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to path for script execution
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import DatasetFormatter, DatasetLoader, DatasetStats
from src.data.splitter import DatasetSplitter
from src.data.templates import TemplateFactory
from src.data.validator import DatasetValidator
from src.utils.logger import get_logger, setup_logging

log = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare datasets for LLM fine-tuning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Dataset source: file path (JSON/CSV) or HuggingFace dataset name with --hf",
    )
    parser.add_argument(
        "--hf",
        action="store_true",
        help="Load from HuggingFace Hub instead of local file",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="HuggingFace dataset split to load",
    )
    parser.add_argument(
        "--format",
        default=None,
        help="Dataset format (alpaca, dolly, oasst, sharegpt). Auto-detected if not specified.",
    )
    parser.add_argument(
        "--template",
        default="alpaca",
        choices=TemplateFactory.available_templates(),
        help="Prompt template for formatting",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="Training set ratio",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.05,
        help="Validation set ratio",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.05,
        help="Test set ratio",
    )
    parser.add_argument(
        "--min-len",
        type=int,
        default=10,
        help="Minimum total character length",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=8192,
        help="Maximum total character length",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    # Stage 1: Load dataset
    print(f"\n{'='*60}")
    print("Stage 1: Loading dataset")
    print(f"{'='*60}")

    if args.hf:
        dataset = DatasetLoader.load_from_huggingface(args.source, split=args.split)
    elif args.source.endswith(".csv"):
        dataset = DatasetLoader.load_from_csv(args.source)
    else:
        dataset = DatasetLoader.load_from_json(args.source)

    print(f"  Loaded {len(dataset)} samples")
    print(f"  Columns: {dataset.column_names}")

    # Stage 2: Convert to unified format
    print(f"\n{'='*60}")
    print("Stage 2: Converting to unified format")
    print(f"{'='*60}")

    dataset = DatasetFormatter.to_unified(dataset, format_name=args.format)
    print(f"  Unified format: {dataset.column_names}")
    print(f"  Samples: {len(dataset)}")

    # Stage 3: Compute pre-validation stats
    print(f"\n{'='*60}")
    print("Stage 3: Pre-validation statistics")
    print(f"{'='*60}")

    stats = DatasetStats.compute(dataset)
    print(f"  Num samples: {stats.num_samples}")
    print(f"  Avg instruction length: {stats.avg_instruction_len:.0f} chars")
    print(f"  Avg input length: {stats.avg_input_len:.0f} chars")
    print(f"  Avg output length: {stats.avg_output_len:.0f} chars")
    print(f"  Empty input count: {stats.empty_input_count}")
    print(f"  Token distribution: {stats.token_length_distribution}")

    # Stage 4: Validate and filter
    print(f"\n{'='*60}")
    print("Stage 4: Validation and filtering")
    print(f"{'='*60}")

    dataset, report = DatasetValidator.validate_all(
        dataset, min_len=args.min_len, max_len=args.max_len
    )
    print(report.summary())

    # Stage 5: Apply template
    print(f"\n{'='*60}")
    print("Stage 5: Applying prompt template")
    print(f"{'='*60}")

    template = TemplateFactory.get_template(args.template)
    print(f"  Template: {template.name}")

    formatted = dataset.map(
        lambda x: {"text": template.format_train(x)},
        remove_columns=dataset.column_names,
    )
    print(f"  Formatted {len(formatted)} samples")

    # Show an example
    if len(formatted) > 0:
        print(f"\n  Example (first 200 chars):")
        print(f"  {formatted[0]['text'][:200]}...")

    # Stage 6: Split dataset
    print(f"\n{'='*60}")
    print("Stage 6: Splitting dataset")
    print(f"{'='*60}")

    splits = DatasetSplitter.split(
        formatted,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    print(f"  Train: {len(splits['train'])} samples")
    print(f"  Validation: {len(splits['validation'])} samples")
    print(f"  Test: {len(splits['test'])} samples")

    # Stage 7: Save to disk
    print(f"\n{'='*60}")
    print("Stage 7: Saving processed data")
    print(f"{'='*60}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_ds in splits.items():
        if len(split_ds) == 0:
            continue
        output_path = output_dir / f"{split_name}.jsonl"
        with open(output_path, "w", encoding="utf-8") as f:
            for sample in split_ds:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        print(f"  Saved {split_name}: {output_path} ({len(split_ds)} samples)")

    print(f"\n{'='*60}")
    print("Pipeline complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
