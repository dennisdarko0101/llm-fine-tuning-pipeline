#!/usr/bin/env python3
"""Training entry point: load config → prepare data → train → save.

Usage:
    python scripts/train.py --config configs/mistral_7b_qlora.yaml --dataset data/sample/sample_alpaca.json
    python scripts/train.py --config configs/llama3_8b_qlora.yaml --dataset tatsu-lab/alpaca --hf
    python scripts/train.py --config configs/mistral_7b_qlora.yaml --dataset data/sample/sample_alpaca.json --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path for script execution
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.training_config import TrainingConfig
from src.data.dataset import DatasetFormatter, DatasetLoader, DatasetStats
from src.data.splitter import DatasetSplitter
from src.data.templates import TemplateFactory
from src.data.validator import DatasetValidator
from src.training.callbacks import (
    CheckpointCallback,
    EarlyStoppingCallback,
    LoggingCallback,
    WandbMetricsCallback,
)
from src.training.quantization import (
    apply_lora,
    create_bnb_config,
    create_lora_config,
    get_trainable_param_stats,
    load_quantized_model,
)
from src.training.trainer import FineTuneTrainer
from src.utils.logger import get_logger, setup_logging

log = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tuning pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to training config YAML file",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset source: local file path or HuggingFace name with --hf",
    )
    parser.add_argument(
        "--hf",
        action="store_true",
        help="Load dataset from HuggingFace Hub",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory from config",
    )
    parser.add_argument(
        "--template",
        default=None,
        help="Override prompt template from config",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Set up everything but don't actually train",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
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

    # ---------------------------------------------------------------
    # Stage 1: Load and validate training config
    # ---------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Stage 1: Loading training config")
    print(f"{'='*60}")

    config = TrainingConfig.from_yaml(args.config)

    if args.output_dir:
        config.output_dir = args.output_dir
    if args.no_wandb:
        config.report_to = "none"

    issues = config.validate()
    if issues:
        print("Config validation issues:")
        for issue in issues:
            print(f"  - {issue}")
        sys.exit(1)

    print(f"  Model: {config.model_name}")
    print(f"  Output: {config.output_dir}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Effective batch size: {config.effective_batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  LoRA r={config.lora_r}, alpha={config.lora_alpha}")

    # ---------------------------------------------------------------
    # Stage 2: Load and prepare dataset
    # ---------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Stage 2: Preparing dataset")
    print(f"{'='*60}")

    if args.hf:
        dataset = DatasetLoader.load_from_huggingface(args.dataset)
    elif args.dataset.endswith(".csv"):
        dataset = DatasetLoader.load_from_csv(args.dataset)
    else:
        dataset = DatasetLoader.load_from_json(args.dataset)

    print(f"  Loaded {len(dataset)} samples")

    # Convert to unified format
    dataset = DatasetFormatter.to_unified(dataset)
    print(f"  Unified format: {len(dataset)} samples")

    # Validate
    dataset, report = DatasetValidator.validate_all(dataset)
    print(f"  After validation: {report.filtered_count} samples (removed {report.removed_count})")

    # Apply template
    template_name = args.template or config.prompt_template or "alpaca"
    template = TemplateFactory.get_template(template_name)
    dataset = dataset.map(
        lambda x: {"text": template.format_train(x)},
        remove_columns=dataset.column_names,
    )
    print(f"  Template: {template.name}")

    # Split
    splits = DatasetSplitter.split(dataset, train_ratio=0.9, val_ratio=0.1, test_ratio=0.0)
    train_dataset = splits["train"]
    val_dataset = splits["validation"] if len(splits["validation"]) > 0 else None
    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset) if val_dataset else 0}")

    # Stats
    stats = DatasetStats.compute(
        DatasetLoader.load_from_json(args.dataset)
        if not args.hf
        else train_dataset
    )
    print(f"  Avg output length: {stats.avg_output_len:.0f} chars")

    # ---------------------------------------------------------------
    # Stage 3: Load model + apply LoRA
    # ---------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Stage 3: Loading model and applying LoRA")
    print(f"{'='*60}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=False,
    )

    bnb_config = create_bnb_config(config)
    model = load_quantized_model(config.model_name, bnb_config)

    lora_config = create_lora_config(config)
    model = apply_lora(model, lora_config)

    param_stats = get_trainable_param_stats(model)
    print(f"  Total params: {param_stats['total_params']:,}")
    print(f"  Trainable params: {param_stats['trainable_params']:,}")
    print(f"  Trainable %: {param_stats['trainable_pct']:.2f}%")
    print(f"  Model size: {param_stats['model_size_mb']:.1f} MB")

    # ---------------------------------------------------------------
    # Stage 4: Setup callbacks and trainer
    # ---------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Stage 4: Setting up trainer")
    print(f"{'='*60}")

    callbacks = [
        LoggingCallback(),
        EarlyStoppingCallback(patience=3),
        CheckpointCallback(save_dir=str(Path(config.output_dir) / "best"), top_k=2),
    ]

    if config.report_to == "wandb":
        callbacks.append(WandbMetricsCallback())

    trainer = FineTuneTrainer(
        training_config=config,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        callbacks=callbacks,
    )

    print("  Trainer ready")
    print(f"  Callbacks: {[type(c).__name__ for c in callbacks]}")

    if args.dry_run:
        print(f"\n{'='*60}")
        print("DRY RUN — skipping training")
        print(f"{'='*60}")
        print("  Setup complete. Remove --dry-run to start training.")
        return

    # ---------------------------------------------------------------
    # Stage 5: Train
    # ---------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Stage 5: Training")
    print(f"{'='*60}")

    result = trainer.train()
    print(result.summary())

    # ---------------------------------------------------------------
    # Stage 6: Summary
    # ---------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}")
    print(f"  Final checkpoint: {result.checkpoint_path}")
    print(f"  Training time: {result.training_time:.1f}s")


if __name__ == "__main__":
    main()
