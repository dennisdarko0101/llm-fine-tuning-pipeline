# LLM Fine-Tuning Pipeline

[![CI](https://github.com/yourusername/llm-fine-tuning-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/llm-fine-tuning-pipeline/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Production pipeline for fine-tuning open-source LLMs using **QLoRA** (4-bit quantized LoRA) and deploying to **AWS SageMaker**. Supports Mistral-7B, Llama-3-8B, and other HuggingFace causal language models.

## Architecture

```
  Data Preparation        Training            Evaluation          Deployment
 ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
 │  Load        │    │  QLoRA       │    │  Perplexity  │    │  Model       │
 │  (HF/JSON/   │───>│  Fine-tuning │───>│  BLEU/ROUGE  │───>│  Registry    │
 │   CSV)       │    │  (SFTTrainer)│    │  Accuracy    │    │              │
 ├──────────────┤    ├──────────────┤    ├──────────────┤    ├──────────────┤
 │  Validate    │    │  Callbacks   │    │  Benchmark   │    │  Package     │
 │  Format      │    │  (EarlyStopping│  │  (Base vs FT)│    │  (tar.gz→S3) │
 │  Template    │    │   Checkpoint) │    │              │    │              │
 │  Split       │    │  W&B Logging │    │  Human Eval  │    │  SageMaker   │
 └──────────────┘    └──────────────┘    └──────────────┘    │  Endpoint    │
                                                              └──────────────┘
                                          Inference
                                         ┌──────────────┐
                                         │  FastAPI      │
                                         │  /predict     │
                                         │  /stream      │
                                         │  /batch       │
                                         └──────────────┘
```

## Features

- **QLoRA Training** — 4-bit NF4 quantization + LoRA adapters for memory-efficient fine-tuning
- **Multi-Format Data** — Auto-detect Alpaca, Dolly, OASST, ShareGPT formats; validate and split
- **Prompt Templates** — Alpaca, ChatML, Llama-3, Mistral templates with factory pattern
- **Evaluation Suite** — Perplexity, BLEU, ROUGE-1/2/L, task accuracy, human eval export
- **Model Benchmarking** — Compare base vs fine-tuned with improvement tracking
- **FastAPI Inference** — Single, streaming, and batch prediction endpoints
- **SageMaker Deployment** — Package, deploy, invoke, and manage endpoints
- **Model Registry** — JSON-based versioning with best-model selection
- **Experiment Tracking** — W&B integration for metrics, artifacts, and callbacks
- **CI/CD** — GitHub Actions for testing, training, and deployment workflows

## Supported Models

| Model | Config | Context | Memory (QLoRA) |
|-------|--------|---------|----------------|
| Mistral-7B-v0.3 | `configs/mistral_7b_qlora.yaml` | 2048 | ~6 GB |
| Llama-3-8B | `configs/llama3_8b_qlora.yaml` | 4096 | ~8 GB |

## Quick Start

```bash
# 1. Clone and install
git clone <repo-url>
cd llm-fine-tuning-pipeline
pip install -e ".[dev]"

# 2. Configure environment
cp .env.example .env
# Edit .env with your AWS, W&B, and HF credentials

# 3. Prepare data
python -m scripts.prepare_data \
  --source data/sample/sample_alpaca.json \
  --template alpaca \
  --output-dir outputs/data

# 4. Train (dry run — no GPU needed)
python -m scripts.train \
  --config configs/mistral_7b_qlora.yaml \
  --dataset outputs/data/train.jsonl \
  --dry-run --no-wandb

# 5. Evaluate
python -m scripts.evaluate \
  --model outputs/final \
  --dataset outputs/data/test.jsonl \
  --no-wandb

# 6. Run inference
python -m scripts.inference --model outputs/final --mode repl

# 7. Start API server
python -m scripts.inference --model outputs/final --mode server --port 8000

# 8. Deploy to SageMaker
python -m scripts.deploy --action deploy \
  --model-path outputs/final \
  --instance-type ml.g5.xlarge
```

## Inference API

Start the server: `make inference-server MODEL=outputs/final`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check and model status |
| `/model/info` | GET | Model name, device, parameters |
| `/predict` | POST | Single text generation |
| `/predict/stream` | POST | Streaming generation (SSE) |
| `/predict/batch` | POST | Batch generation (up to 32) |

**Example:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain quantum computing", "max_new_tokens": 256, "temperature": 0.7}'
```

## Training Configuration

Key hyperparameters in YAML configs:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lora_r` | 16 | LoRA rank |
| `lora_alpha` | 32 | LoRA scaling factor |
| `lora_dropout` | 0.05 | LoRA dropout |
| `learning_rate` | 2e-4 | Peak learning rate |
| `num_epochs` | 3 | Training epochs |
| `per_device_train_batch_size` | 4 | Batch size per GPU |
| `gradient_accumulation_steps` | 4 | Effective batch = 16 |
| `lr_scheduler_type` | cosine | LR schedule |
| `max_seq_length` | 2048 | Max token sequence |
| `bnb_4bit_quant_type` | nf4 | QLoRA quantization |

See `configs/` for complete configurations and `docs/TRAINING_GUIDE.md` for detailed tuning guidance.

## Project Structure

```
llm-fine-tuning-pipeline/
├── src/
│   ├── config/           # Settings + training hyperparameters
│   │   ├── settings.py       # Pydantic env settings (AWS, W&B, HF)
│   │   └── training_config.py # Training config dataclass + YAML
│   ├── data/             # Dataset loading and preprocessing
│   │   ├── dataset.py        # DatasetLoader, DatasetFormatter, DatasetStats
│   │   ├── templates.py      # PromptTemplate ABC + 4 implementations
│   │   ├── validator.py      # DatasetValidator + ValidationReport
│   │   └── splitter.py       # Train/val/test splitting
│   ├── training/         # QLoRA training pipeline
│   │   ├── quantization.py   # BnB config, LoRA, model loading
│   │   ├── trainer.py        # FineTuneTrainer + TrainResult
│   │   └── callbacks.py      # EarlyStopping, Checkpoint, Logging, W&B
│   ├── evaluation/       # Model evaluation and benchmarking
│   │   ├── evaluator.py      # Perplexity, BLEU, ROUGE, accuracy
│   │   ├── benchmark.py      # Multi-model comparison
│   │   └── human_eval.py     # Human eval sample generation
│   ├── inference/        # Local inference and API
│   │   ├── model_loader.py   # Base, fine-tuned, merged loading
│   │   ├── predictor.py      # Single, stream, batch prediction
│   │   └── api.py            # FastAPI server
│   ├── deployment/       # AWS SageMaker deployment
│   │   ├── model_registry.py # JSON-based model versioning
│   │   ├── sagemaker.py      # Package, deploy, invoke, delete
│   │   └── infrastructure.py # S3, IAM, cost estimation
│   └── utils/            # Shared utilities
│       ├── logger.py         # Structured logging (structlog)
│       └── wandb_utils.py    # W&B init, metrics, artifacts
├── configs/              # YAML training configurations
├── scripts/              # CLI entry points
│   ├── prepare_data.py       # Data pipeline CLI
│   ├── train.py              # Training CLI
│   ├── evaluate.py           # Evaluation CLI
│   ├── inference.py          # REPL / server CLI
│   └── deploy.py             # SageMaker deployment CLI
├── docker/               # Container images
│   ├── Dockerfile.train      # GPU training image (CUDA 12.1)
│   └── Dockerfile.inference  # Lightweight inference image
├── tests/                # 264+ tests
│   ├── unit/                 # Unit tests (no GPU, mocked AWS)
│   └── integration/          # API integration tests
├── .github/workflows/    # CI/CD pipelines
│   ├── ci.yml                # Lint + test + coverage
│   ├── train.yml             # Manual training workflow
│   └── deploy.yml            # SageMaker deployment workflow
├── docs/                 # Documentation
├── data/sample/          # Sample training data
└── Makefile              # Build targets
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Fine-tuning | PyTorch, Transformers, PEFT, TRL, bitsandbytes |
| Data | HuggingFace Datasets |
| API | FastAPI, Uvicorn |
| Deployment | AWS SageMaker, boto3 |
| Tracking | Weights & Biases |
| CI/CD | GitHub Actions |
| Containers | Docker (CUDA 12.1 / Python 3.11) |
| Logging | structlog |
| Config | Pydantic Settings, PyYAML |
| Testing | pytest, pytest-cov |
| Linting | Ruff, mypy |

## Development

```bash
make install          # Install with dev deps
make test             # Run all tests (264+) with coverage
make test-unit        # Unit tests only
make lint             # Ruff + mypy
make format           # Auto-format
make clean            # Remove caches and artifacts
```

## Documentation

- [Training Guide](docs/TRAINING_GUIDE.md) — QLoRA concepts, hyperparameter tuning, troubleshooting
- [Architecture](docs/ARCHITECTURE.md) — System design, data flow, deployment architecture
- [Deployment](docs/DEPLOYMENT.md) — Local, Docker, and SageMaker deployment
- [Results Template](docs/RESULTS.md) — Training results reporting template
- [Contributing](CONTRIBUTING.md) — Development workflow and guidelines

## License

MIT
