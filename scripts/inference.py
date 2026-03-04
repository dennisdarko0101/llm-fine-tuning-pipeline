"""Interactive inference REPL and server launcher.

Usage:
    # Interactive REPL
    python scripts/inference.py --model outputs/final --mode repl

    # Start API server
    python scripts/inference.py --model outputs/final --mode server --port 8000

    # Single prediction
    python scripts/inference.py --model outputs/final --mode single --prompt "Explain Python decorators"
"""

from __future__ import annotations

import argparse
import sys

from src.utils.logger import get_logger, setup_logging

log = get_logger(__name__)


def run_repl(predictor) -> None:
    """Run an interactive read-eval-print loop."""
    from src.inference.predictor import GenerationConfig

    print("\nLLM Inference REPL")
    print("Type 'quit' or 'exit' to stop. Type 'config' to adjust generation settings.\n")

    config = GenerationConfig()

    while True:
        try:
            prompt = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not prompt:
            continue
        if prompt.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        if prompt.lower() == "config":
            print(f"  max_new_tokens: {config.max_new_tokens}")
            print(f"  temperature: {config.temperature}")
            print(f"  top_p: {config.top_p}")
            print(f"  top_k: {config.top_k}")
            print(f"  do_sample: {config.do_sample}")
            print(f"  repetition_penalty: {config.repetition_penalty}")
            continue
        if prompt.lower().startswith("set "):
            _handle_set_command(prompt, config)
            continue
        if prompt.lower() == "stream":
            prompt = input("stream>>> ").strip()
            if prompt:
                for token in predictor.predict_stream(prompt, config=config):
                    print(token, end="", flush=True)
                print()
            continue

        result = predictor.predict(prompt, config=config)
        print(f"\n{result.generated_text}")
        print(f"\n[{result.num_tokens} tokens, {result.generation_time:.2f}s, "
              f"{result.tokens_per_second:.1f} tok/s]\n")


def _handle_set_command(prompt: str, config) -> None:
    """Handle 'set key=value' commands in the REPL."""
    try:
        _, rest = prompt.split(" ", 1)
        key, value = rest.split("=", 1)
        key = key.strip()
        value = value.strip()

        if key == "max_new_tokens":
            config.max_new_tokens = int(value)
        elif key == "temperature":
            config.temperature = float(value)
        elif key == "top_p":
            config.top_p = float(value)
        elif key == "top_k":
            config.top_k = int(value)
        elif key == "do_sample":
            config.do_sample = value.lower() in ("true", "1", "yes")
        elif key == "repetition_penalty":
            config.repetition_penalty = float(value)
        else:
            print(f"Unknown setting: {key}")
            return
        print(f"  {key} = {getattr(config, key)}")
    except (ValueError, AttributeError) as e:
        print(f"  Error: {e}")


def run_server(predictor, model_name: str, host: str, port: int) -> None:
    """Start the FastAPI inference server."""
    import uvicorn

    from src.inference.api import app, configure_app

    configure_app(predictor, model_name=model_name)
    log.info("starting_server", host=host, port=port)
    uvicorn.run(app, host=host, port=port)


def main(argv: list[str] | None = None) -> None:
    """Run inference in the specified mode."""
    parser = argparse.ArgumentParser(description="LLM inference tool")
    parser.add_argument("--model", required=True, help="Model path or HF identifier")
    parser.add_argument("--adapter", default=None, help="LoRA adapter path")
    parser.add_argument(
        "--mode", choices=["repl", "server", "single"], default="repl",
        help="Inference mode"
    )
    parser.add_argument("--prompt", default=None, help="Prompt for single mode")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    args = parser.parse_args(argv)

    setup_logging(args.log_level)

    # Load model
    log.info("loading_model", model=args.model)
    from src.inference.model_loader import ModelLoader

    if args.adapter:
        model, tokenizer = ModelLoader.load_finetuned(args.model, args.adapter)
    else:
        model, tokenizer = ModelLoader.load_from_checkpoint(args.model)

    # Create predictor
    from src.inference.predictor import GenerationConfig, Predictor

    default_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    predictor = Predictor(model, tokenizer, default_config=default_config)

    # Run in specified mode
    if args.mode == "repl":
        run_repl(predictor)
    elif args.mode == "server":
        run_server(predictor, model_name=args.model, host=args.host, port=args.port)
    elif args.mode == "single":
        if not args.prompt:
            log.error("no_prompt", msg="Provide --prompt for single mode")
            sys.exit(1)
        result = predictor.predict(args.prompt)
        print(result.generated_text)
    else:
        log.error("unknown_mode", mode=args.mode)
        sys.exit(1)


if __name__ == "__main__":
    main()
