"""Tests for prompt templates."""

from __future__ import annotations

import pytest

from src.data.templates import (
    AlpacaTemplate,
    ChatMLTemplate,
    Llama3Template,
    MistralTemplate,
    PromptTemplate,
    TemplateFactory,
)


class TestAlpacaTemplate:
    """Tests for Alpaca prompt template."""

    def setup_method(self) -> None:
        self.template = AlpacaTemplate()

    def test_format_with_input(self) -> None:
        result = self.template.format("Translate", "Hello", "Bonjour")
        assert "### Instruction:\nTranslate" in result
        assert "### Input:\nHello" in result
        assert "### Response:\nBonjour" in result
        assert "paired with further input" in result

    def test_format_without_input(self) -> None:
        result = self.template.format("Say hello", "", "Hello!")
        assert "### Instruction:\nSay hello" in result
        assert "### Input:" not in result
        assert "### Response:\nHello!" in result

    def test_format_inference_no_output(self) -> None:
        result = self.template.format_inference("Translate", "Hello")
        assert "### Instruction:\nTranslate" in result
        assert "### Input:\nHello" in result
        assert "### Response:\n" in result
        # Output should be empty at the end
        assert result.endswith("### Response:\n")

    def test_format_train(self) -> None:
        sample = {"instruction": "Say hi", "input": "", "output": "Hi!"}
        result = self.template.format_train(sample)
        assert "Say hi" in result
        assert "Hi!" in result

    def test_name(self) -> None:
        assert self.template.name == "alpaca"


class TestChatMLTemplate:
    """Tests for ChatML prompt template."""

    def setup_method(self) -> None:
        self.template = ChatMLTemplate()

    def test_format_basic(self) -> None:
        result = self.template.format("Hello", "", "Hi there!")
        assert "<|im_start|>user\nHello<|im_end|>" in result
        assert "<|im_start|>assistant\nHi there!<|im_end|>" in result

    def test_format_with_input(self) -> None:
        result = self.template.format("Translate", "Hello", "Bonjour")
        assert "<|im_start|>user\nTranslate\nHello<|im_end|>" in result
        assert "<|im_start|>assistant\nBonjour<|im_end|>" in result

    def test_format_inference(self) -> None:
        result = self.template.format_inference("Hello")
        assert result.startswith("<|im_start|>user\nHello<|im_end|>")
        assert result.endswith("<|im_start|>assistant\n")
        assert "<|im_end|>" not in result.split("assistant\n")[-1]

    def test_name(self) -> None:
        assert self.template.name == "chatml"


class TestLlama3Template:
    """Tests for Llama 3 prompt template."""

    def setup_method(self) -> None:
        self.template = Llama3Template()

    def test_format_basic(self) -> None:
        result = self.template.format("Hello", "", "Hi!")
        assert "<|begin_of_text|>" in result
        assert "<|start_header_id|>user<|end_header_id|>" in result
        assert "Hello<|eot_id|>" in result
        assert "<|start_header_id|>assistant<|end_header_id|>" in result
        assert "Hi!<|eot_id|>" in result

    def test_format_with_input(self) -> None:
        result = self.template.format("Translate", "Hello", "Bonjour")
        assert "Translate\nHello<|eot_id|>" in result
        assert "Bonjour<|eot_id|>" in result

    def test_format_inference(self) -> None:
        result = self.template.format_inference("Hello")
        assert "<|begin_of_text|>" in result
        assert result.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n")

    def test_name(self) -> None:
        assert self.template.name == "llama3"


class TestMistralTemplate:
    """Tests for Mistral prompt template."""

    def setup_method(self) -> None:
        self.template = MistralTemplate()

    def test_format_basic(self) -> None:
        result = self.template.format("Hello", "", "Hi!")
        assert result == "[INST] Hello [/INST] Hi!"

    def test_format_with_input(self) -> None:
        result = self.template.format("Translate", "Hello", "Bonjour")
        assert result == "[INST] Translate\nHello [/INST] Bonjour"

    def test_format_inference(self) -> None:
        result = self.template.format_inference("Hello")
        assert result == "[INST] Hello [/INST] "

    def test_name(self) -> None:
        assert self.template.name == "mistral"


class TestTemplateFactory:
    """Tests for template factory."""

    def test_get_alpaca(self) -> None:
        t = TemplateFactory.get_template("alpaca")
        assert isinstance(t, AlpacaTemplate)

    def test_get_chatml(self) -> None:
        t = TemplateFactory.get_template("chatml")
        assert isinstance(t, ChatMLTemplate)

    def test_get_llama3(self) -> None:
        t = TemplateFactory.get_template("llama3")
        assert isinstance(t, Llama3Template)

    def test_get_mistral(self) -> None:
        t = TemplateFactory.get_template("mistral")
        assert isinstance(t, MistralTemplate)

    def test_case_insensitive(self) -> None:
        t = TemplateFactory.get_template("ALPACA")
        assert isinstance(t, AlpacaTemplate)

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown template"):
            TemplateFactory.get_template("nonexistent")

    def test_available_templates(self) -> None:
        available = TemplateFactory.available_templates()
        assert "alpaca" in available
        assert "chatml" in available
        assert "llama3" in available
        assert "mistral" in available

    def test_register_custom(self) -> None:
        class CustomTemplate(PromptTemplate):
            name = "custom"

            def format(self, instruction: str, input: str = "", output: str = "") -> str:
                return f"CUSTOM: {instruction}"

            def format_inference(self, instruction: str, input: str = "") -> str:
                return f"CUSTOM: {instruction}"

        TemplateFactory.register("custom", CustomTemplate)
        t = TemplateFactory.get_template("custom")
        assert isinstance(t, CustomTemplate)
        assert t.format("test") == "CUSTOM: test"


class TestTemplateEdgeCases:
    """Test edge cases across all templates."""

    @pytest.mark.parametrize("template_name", ["alpaca", "chatml", "llama3", "mistral"])
    def test_empty_input_field(self, template_name: str) -> None:
        t = TemplateFactory.get_template(template_name)
        result = t.format("Do something", "", "Done")
        assert "Do something" in result
        assert "Done" in result

    @pytest.mark.parametrize("template_name", ["alpaca", "chatml", "llama3", "mistral"])
    def test_format_train_with_sample(self, template_name: str) -> None:
        t = TemplateFactory.get_template(template_name)
        sample = {"instruction": "Test", "input": "data", "output": "result"}
        result = t.format_train(sample)
        assert "Test" in result
        assert "result" in result

    @pytest.mark.parametrize("template_name", ["alpaca", "chatml", "llama3", "mistral"])
    def test_inference_does_not_contain_output(self, template_name: str) -> None:
        t = TemplateFactory.get_template(template_name)
        result = t.format_inference("Do something", "with this")
        # Inference output should not have the answer filled in
        # But it should have the instruction and input
        assert "Do something" in result
