# LLM Fine-Tuning Pipeline

Production pipeline for fine-tuning open-source LLMs (Mistral-7B, Llama-3) using QLoRA and deploying to AWS SageMaker.

## Architecture

```
src/
├── config/          # Settings + training hyperparameters
├── data/            # Dataset loading, preprocessing, formatting
├── training/        # QLoRA training loop and SFT trainer
├── evaluation/      # Model evaluation and benchmarking
├── inference/       # Local inference and batch prediction
├── deployment/      # SageMaker packaging and deployment
└── utils/           # Logging, W&B helpers, common utilities

configs/             # YAML training configurations per model
scripts/             # CLI entry points (train, evaluate, deploy)
docker/              # Dockerfiles for training and serving
tests/               # Unit and integration tests
```

## Supported Models

| Model | Config | Context |
|-------|--------|---------|
| Mistral-7B-v0.3 | `configs/mistral_7b_qlora.yaml` | 2048 |
| Llama-3-8B | `configs/llama3_8b_qlora.yaml` | 4096 |

## Setup

```bash
# Clone
git clone <repo-url>
cd llm-fine-tuning-pipeline

# Install
pip install -e ".[dev]"

# Configure
cp .env.example .env
# Edit .env with your AWS, W&B, and HF credentials
```

## Usage

```bash
# Train with a config
make train CONFIG=configs/mistral_7b_qlora.yaml

# Evaluate
make evaluate CONFIG=configs/mistral_7b_qlora.yaml

# Deploy to SageMaker
make deploy CONFIG=configs/mistral_7b_qlora.yaml
```

## Development

```bash
make lint       # Ruff + mypy
make format     # Auto-format
make test       # Run all tests
make test-unit  # Unit tests only
```

## Key Technologies

- **QLoRA**: 4-bit quantized LoRA for memory-efficient fine-tuning
- **PEFT**: Parameter-Efficient Fine-Tuning library
- **TRL**: Transformer Reinforcement Learning (SFTTrainer)
- **bitsandbytes**: 4-bit quantization
- **Weights & Biases**: Experiment tracking
- **AWS SageMaker**: Model deployment and serving
