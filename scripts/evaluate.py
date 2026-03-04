"""CLI script for model evaluation.

Usage:
    python scripts/evaluate.py --model outputs/final --dataset data/eval.json
    python scripts/evaluate.py --model outputs/final --hf tatsu-lab/alpaca_eval --split test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.utils.logger import get_logger, setup_logging

log = get_logger(__name__)


def main(argv: list[str] | None = None) -> None:
    """Run evaluation pipeline."""
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned model")
    parser.add_argument("--model", required=True, help="Model path or HF identifier")
    parser.add_argument("--adapter", default=None, help="LoRA adapter path (for fine-tuned)")
    parser.add_argument("--dataset", default=None, help="Path to evaluation dataset (JSON/CSV)")
    parser.add_argument("--hf", default=None, help="HuggingFace dataset name")
    parser.add_argument("--split", default="test", help="Dataset split to use")
    parser.add_argument("--output-dir", default="outputs/eval", help="Output directory for results")
    parser.add_argument("--text-column", default="text", help="Text column name")
    parser.add_argument("--max-length", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--batch-size", type=int, default=4, help="Evaluation batch size")
    parser.add_argument("--human-eval", type=int, default=0, help="Number of human eval samples")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    args = parser.parse_args(argv)

    setup_logging(args.log_level)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    log.info("loading_eval_dataset")
    from src.data.dataset import DatasetLoader

    if args.hf:
        dataset = DatasetLoader.load_from_huggingface(args.hf, split=args.split)
    elif args.dataset:
        p = Path(args.dataset)
        if p.suffix == ".csv":
            dataset = DatasetLoader.load_from_csv(p)
        else:
            dataset = DatasetLoader.load_from_json(p)
    else:
        log.error("no_dataset", msg="Provide --dataset or --hf")
        sys.exit(1)

    log.info("eval_dataset_loaded", num_samples=len(dataset))

    # Load model
    log.info("loading_model", model=args.model)
    from src.inference.model_loader import ModelLoader

    if args.adapter:
        model, tokenizer = ModelLoader.load_finetuned(args.model, args.adapter)
    else:
        model, tokenizer = ModelLoader.load_from_checkpoint(args.model)

    # Run evaluation
    log.info("running_evaluation")
    from src.evaluation.evaluator import ModelEvaluator

    evaluator = ModelEvaluator(model, tokenizer)
    result = evaluator.evaluate_all(
        dataset,
        model_name=args.model,
        dataset_name=args.hf or args.dataset or "unknown",
        text_column=args.text_column,
        max_length=args.max_length,
    )

    # Save results
    results_path = output_dir / "eval_results.json"
    results_path.write_text(
        json.dumps(result.metrics, indent=2, default=str), encoding="utf-8"
    )
    log.info("results_saved", path=str(results_path))

    # Human evaluation samples
    if args.human_eval > 0:
        log.info("generating_human_eval_samples", num_samples=args.human_eval)
        from src.evaluation.human_eval import HumanEvalGenerator

        gen = HumanEvalGenerator(model, tokenizer)
        samples = gen.generate_samples(dataset, num_samples=args.human_eval)
        HumanEvalGenerator.export_markdown(samples, output_dir / "human_eval.md")
        HumanEvalGenerator.export_csv(samples, output_dir / "human_eval.csv")

    # W&B logging
    if not args.no_wandb:
        try:
            from src.utils.wandb_utils import finish_wandb, init_wandb, log_metrics

            run = init_wandb(
                run_name=f"eval-{Path(args.model).name}",
                config={"model": args.model, "dataset": args.hf or args.dataset},
                tags=["evaluation"],
            )
            if run:
                log_metrics(result.metrics)
                finish_wandb()
        except Exception:
            log.warning("wandb_logging_failed", exc_info=True)

    # Print summary
    print("\n" + result.summary())


if __name__ == "__main__":
    main()
