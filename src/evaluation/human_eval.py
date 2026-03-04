"""Human evaluation sample generation for manual review."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from src.utils.logger import get_logger

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

log = get_logger(__name__)


@dataclass
class GeneratedSample:
    """A single generated sample for human review."""

    prompt: str = ""
    reference: str = ""
    generated: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class HumanEvalGenerator:
    """Generate samples for human evaluation.

    Args:
        model: Model to generate with.
        tokenizer: Tokenizer for the model.
        device: Device to use for generation.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        device: str | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def generate_samples(
        self,
        dataset: Dataset,
        num_samples: int = 20,
        input_column: str = "text",
        reference_column: str | None = "output",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        seed: int = 42,
    ) -> list[GeneratedSample]:
        """Generate samples from the dataset for human review.

        Args:
            dataset: Dataset to sample from.
            num_samples: Number of samples to generate.
            input_column: Column with input prompts.
            reference_column: Column with reference outputs (optional).
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            seed: Random seed for reproducible sampling.

        Returns:
            List of GeneratedSample objects.
        """
        import random

        random.seed(seed)
        num_samples = min(num_samples, len(dataset))
        indices = random.sample(range(len(dataset)), num_samples)

        samples = []
        self.model.eval()

        for idx in indices:
            row = dataset[idx]
            prompt = row[input_column]
            reference = row.get(reference_column, "") if reference_column else ""

            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
            input_ids = inputs["input_ids"].to(self.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else None,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            generated = self.tokenizer.decode(
                output_ids[0][input_ids.shape[1] :], skip_special_tokens=True
            )

            samples.append(
                GeneratedSample(
                    prompt=prompt,
                    reference=reference if isinstance(reference, str) else "",
                    generated=generated,
                    metadata={"index": idx, "temperature": temperature},
                )
            )

        log.info("human_eval_samples_generated", num_samples=len(samples))
        return samples

    @staticmethod
    def export_markdown(samples: list[GeneratedSample], output_path: str | Path) -> Path:
        """Export samples as a Markdown file for review.

        Args:
            samples: Generated samples.
            output_path: Output file path.

        Returns:
            Path to the created file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        lines = ["# Human Evaluation Samples\n"]
        for i, sample in enumerate(samples, 1):
            lines.append(f"## Sample {i}\n")
            lines.append(f"**Prompt:**\n```\n{sample.prompt}\n```\n")
            if sample.reference:
                lines.append(f"**Reference:**\n```\n{sample.reference}\n```\n")
            lines.append(f"**Generated:**\n```\n{sample.generated}\n```\n")
            lines.append("**Rating:** [ ] 1  [ ] 2  [ ] 3  [ ] 4  [ ] 5\n")
            lines.append("**Notes:**\n\n---\n")

        output_path.write_text("\n".join(lines), encoding="utf-8")
        log.info("human_eval_markdown_exported", path=str(output_path), num_samples=len(samples))
        return output_path

    @staticmethod
    def export_csv(samples: list[GeneratedSample], output_path: str | Path) -> Path:
        """Export samples as a CSV file for review.

        Args:
            samples: Generated samples.
            output_path: Output file path.

        Returns:
            Path to the created file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["sample_id", "prompt", "reference", "generated", "rating", "notes"])
            for i, sample in enumerate(samples, 1):
                writer.writerow([i, sample.prompt, sample.reference, sample.generated, "", ""])

        log.info("human_eval_csv_exported", path=str(output_path), num_samples=len(samples))
        return output_path

    @staticmethod
    def export_json(samples: list[GeneratedSample], output_path: str | Path) -> Path:
        """Export samples as a JSON file.

        Args:
            samples: Generated samples.
            output_path: Output file path.

        Returns:
            Path to the created file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = [
            {
                "sample_id": i,
                "prompt": s.prompt,
                "reference": s.reference,
                "generated": s.generated,
                "metadata": s.metadata,
                "rating": None,
                "notes": "",
            }
            for i, s in enumerate(samples, 1)
        ]

        output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        log.info("human_eval_json_exported", path=str(output_path), num_samples=len(samples))
        return output_path
