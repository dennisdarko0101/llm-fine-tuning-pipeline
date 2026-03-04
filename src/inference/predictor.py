"""Inference predictor for single, streaming, and batch predictions."""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from src.utils.logger import get_logger

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

log = get_logger(__name__)


@dataclass
class PredictionResult:
    """Result from a single prediction."""

    prompt: str = ""
    generated_text: str = ""
    num_tokens: int = 0
    generation_time: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def tokens_per_second(self) -> float:
        """Calculate generation speed."""
        if self.generation_time <= 0:
            return 0.0
        return self.num_tokens / self.generation_time


@dataclass
class GenerationConfig:
    """Configuration for text generation."""

    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    repetition_penalty: float = 1.1
    num_beams: int = 1


class Predictor:
    """Inference predictor supporting single, streaming, and batch prediction.

    Args:
        model: Model for generation.
        tokenizer: Tokenizer matching the model.
        device: Device to use.
        default_config: Default generation config.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        device: str | None = None,
        default_config: GenerationConfig | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.default_config = default_config or GenerationConfig()
        self.model.eval()

    def predict(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> PredictionResult:
        """Generate text for a single prompt.

        Args:
            prompt: Input prompt text.
            config: Generation config (uses default if None).

        Returns:
            PredictionResult with generated text.
        """
        import time

        config = config or self.default_config
        start = time.time()

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        input_ids = inputs["input_ids"].to(self.device)

        gen_kwargs = self._build_gen_kwargs(config)

        with torch.no_grad():
            output_ids = self.model.generate(input_ids, **gen_kwargs)

        new_tokens = output_ids[0][input_ids.shape[1] :]
        generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        elapsed = time.time() - start

        result = PredictionResult(
            prompt=prompt,
            generated_text=generated_text,
            num_tokens=len(new_tokens),
            generation_time=elapsed,
        )
        log.info(
            "prediction_complete",
            tokens=result.num_tokens,
            time=f"{elapsed:.2f}s",
            tps=f"{result.tokens_per_second:.1f}",
        )
        return result

    def predict_stream(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> Generator[str, None, None]:
        """Generate text token-by-token in a streaming fashion.

        Args:
            prompt: Input prompt text.
            config: Generation config.

        Yields:
            Generated text tokens one at a time.
        """
        from transformers import TextIteratorStreamer
        from threading import Thread

        config = config or self.default_config

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        input_ids = inputs["input_ids"].to(self.device)

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        gen_kwargs = self._build_gen_kwargs(config)
        gen_kwargs["input_ids"] = input_ids
        gen_kwargs["streamer"] = streamer

        thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        for text in streamer:
            yield text

        thread.join()

    def predict_batch(
        self,
        prompts: list[str],
        config: GenerationConfig | None = None,
        batch_size: int = 4,
    ) -> list[PredictionResult]:
        """Generate text for multiple prompts.

        Args:
            prompts: List of input prompts.
            config: Generation config.
            batch_size: Number of prompts to process at once.

        Returns:
            List of PredictionResult for each prompt.
        """
        import time

        config = config or self.default_config
        results = []

        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            start = time.time()

            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
            )
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)

            gen_kwargs = self._build_gen_kwargs(config)
            gen_kwargs["attention_mask"] = attention_mask

            with torch.no_grad():
                output_ids = self.model.generate(input_ids, **gen_kwargs)

            elapsed = time.time() - start

            for j, (prompt, out_ids) in enumerate(zip(batch, output_ids)):
                input_len = inputs["input_ids"][j].ne(self.tokenizer.pad_token_id).sum().item()
                new_tokens = out_ids[input_len:]
                generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                results.append(
                    PredictionResult(
                        prompt=prompt,
                        generated_text=generated_text,
                        num_tokens=len(new_tokens),
                        generation_time=elapsed / len(batch),
                    )
                )

        log.info("batch_prediction_complete", num_prompts=len(prompts), num_results=len(results))
        return results

    def _build_gen_kwargs(self, config: GenerationConfig) -> dict[str, Any]:
        """Build generation keyword arguments from config."""
        kwargs: dict[str, Any] = {
            "max_new_tokens": config.max_new_tokens,
            "do_sample": config.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if config.do_sample:
            kwargs["temperature"] = config.temperature
            kwargs["top_p"] = config.top_p
            kwargs["top_k"] = config.top_k
        if config.repetition_penalty != 1.0:
            kwargs["repetition_penalty"] = config.repetition_penalty
        if config.num_beams > 1:
            kwargs["num_beams"] = config.num_beams
        return kwargs
