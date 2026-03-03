"""Prompt templates for formatting training and inference data."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.utils.logger import get_logger

log = get_logger(__name__)


class PromptTemplate(ABC):
    """Base class for prompt templates."""

    name: str = "base"

    @abstractmethod
    def format(self, instruction: str, input: str = "", output: str = "") -> str:
        """Format a complete training example with instruction, input, and output.

        Args:
            instruction: The task instruction.
            input: Optional additional input/context.
            output: The expected response.

        Returns:
            Formatted prompt string.
        """

    @abstractmethod
    def format_inference(self, instruction: str, input: str = "") -> str:
        """Format a prompt for inference (no output).

        Args:
            instruction: The task instruction.
            input: Optional additional input/context.

        Returns:
            Formatted prompt string ready for model generation.
        """

    def format_train(self, sample: dict[str, Any]) -> str:
        """Format a training sample dict.

        Args:
            sample: Dict with instruction, input, output keys.

        Returns:
            Formatted training string.
        """
        return self.format(
            instruction=str(sample.get("instruction", "")),
            input=str(sample.get("input", "")),
            output=str(sample.get("output", "")),
        )


class AlpacaTemplate(PromptTemplate):
    """Alpaca-style instruction template.

    Format:
        Below is an instruction that describes a task...
        ### Instruction:
        {instruction}
        ### Input:
        {input}
        ### Response:
        {output}
    """

    name = "alpaca"

    _PROMPT_WITH_INPUT = (
        "Below is an instruction that describes a task, paired with further input that "
        "provides further context. Write a response that appropriately completes the request."
        "\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"
    )

    _PROMPT_NO_INPUT = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
        "\n\n### Instruction:\n{instruction}\n\n### Response:\n{output}"
    )

    def format(self, instruction: str, input: str = "", output: str = "") -> str:
        if input.strip():
            return self._PROMPT_WITH_INPUT.format(
                instruction=instruction, input=input, output=output
            )
        return self._PROMPT_NO_INPUT.format(instruction=instruction, output=output)

    def format_inference(self, instruction: str, input: str = "") -> str:
        return self.format(instruction=instruction, input=input, output="")


class ChatMLTemplate(PromptTemplate):
    """ChatML format used by OpenAI and many open models.

    Format:
        <|im_start|>user
        {instruction}
        {input}<|im_end|>
        <|im_start|>assistant
        {output}<|im_end|>
    """

    name = "chatml"

    def format(self, instruction: str, input: str = "", output: str = "") -> str:
        user_content = instruction
        if input.strip():
            user_content = f"{instruction}\n{input}"
        return (
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            f"<|im_start|>assistant\n{output}<|im_end|>"
        )

    def format_inference(self, instruction: str, input: str = "") -> str:
        user_content = instruction
        if input.strip():
            user_content = f"{instruction}\n{input}"
        return f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"


class Llama3Template(PromptTemplate):
    """Llama 3 chat template.

    Format:
        <|begin_of_text|><|start_header_id|>user<|end_header_id|>
        {instruction}
        {input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        {output}<|eot_id|>
    """

    name = "llama3"

    def format(self, instruction: str, input: str = "", output: str = "") -> str:
        user_content = instruction
        if input.strip():
            user_content = f"{instruction}\n{input}"
        return (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_content}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{output}<|eot_id|>"
        )

    def format_inference(self, instruction: str, input: str = "") -> str:
        user_content = instruction
        if input.strip():
            user_content = f"{instruction}\n{input}"
        return (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_content}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )


class MistralTemplate(PromptTemplate):
    """Mistral instruction template.

    Format:
        [INST] {instruction}
        {input} [/INST] {output}
    """

    name = "mistral"

    def format(self, instruction: str, input: str = "", output: str = "") -> str:
        content = instruction
        if input.strip():
            content = f"{instruction}\n{input}"
        return f"[INST] {content} [/INST] {output}"

    def format_inference(self, instruction: str, input: str = "") -> str:
        content = instruction
        if input.strip():
            content = f"{instruction}\n{input}"
        return f"[INST] {content} [/INST] "


class TemplateFactory:
    """Factory for creating prompt templates by name."""

    _templates: dict[str, type[PromptTemplate]] = {
        "alpaca": AlpacaTemplate,
        "chatml": ChatMLTemplate,
        "llama3": Llama3Template,
        "mistral": MistralTemplate,
    }

    @classmethod
    def get_template(cls, name: str) -> PromptTemplate:
        """Get a prompt template by name.

        Args:
            name: Template name (alpaca, chatml, llama3, mistral).

        Returns:
            Instantiated PromptTemplate.

        Raises:
            ValueError: If template name is unknown.
        """
        template_cls = cls._templates.get(name.lower())
        if template_cls is None:
            available = sorted(cls._templates.keys())
            raise ValueError(f"Unknown template: {name!r}. Available: {available}")
        log.info("template_created", template=name)
        return template_cls()

    @classmethod
    def available_templates(cls) -> list[str]:
        """Return list of available template names."""
        return sorted(cls._templates.keys())

    @classmethod
    def register(cls, name: str, template_cls: type[PromptTemplate]) -> None:
        """Register a custom template.

        Args:
            name: Template name.
            template_cls: Template class to register.
        """
        cls._templates[name.lower()] = template_cls
        log.info("template_registered", template=name)
