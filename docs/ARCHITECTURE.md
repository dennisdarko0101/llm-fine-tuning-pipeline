# Architecture

## System Overview

The LLM Fine-Tuning Pipeline is a modular system for fine-tuning open-source language models using QLoRA and deploying them to production.

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLI Layer (scripts/)                     │
│  prepare_data.py │ train.py │ evaluate.py │ inference.py │ deploy.py  │
└───────┬──────────┴────┬─────┴──────┬──────┴──────┬───────┴────┬──────┘
        │               │            │             │            │
┌───────▼──────┐ ┌──────▼─────┐ ┌────▼───────┐ ┌──▼─────────┐ ┌▼───────────┐
│  src/data/   │ │src/training│ │ src/eval/  │ │src/inference│ │src/deploy/ │
│              │ │            │ │            │ │             │ │            │
│ DatasetLoader│ │QuantConfig │ │Evaluator   │ │ModelLoader  │ │Registry    │
│ Formatter    │ │LoRA Config │ │Benchmark   │ │Predictor    │ │SageMaker   │
│ Validator    │ │Trainer     │ │HumanEval   │ │FastAPI      │ │Infra       │
│ Splitter     │ │Callbacks   │ │            │ │             │ │            │
│ Templates    │ │            │ │            │ │             │ │            │
└──────────────┘ └────────────┘ └────────────┘ └─────────────┘ └────────────┘
        │               │            │             │            │
┌───────▼───────────────▼────────────▼─────────────▼────────────▼──────┐
│                        Shared Layer (src/utils/, src/config/)        │
│  Settings │ TrainingConfig │ Logger (structlog) │ W&B Utils          │
└─────────────────────────────────────────────────────────────────────┘
```

## Data Flow

### Training Pipeline

```
Raw Data (HF Hub / JSON / CSV)
  │
  ▼
DatasetLoader.load_from_*()
  │
  ▼
DatasetFormatter.to_unified()      ← Auto-detect format (alpaca/dolly/oasst/sharegpt)
  │                                   Normalize to {instruction, input, output}
  ▼
DatasetValidator.validate_all()    ← Schema check → Length filter → Dedup → Quality
  │                                   Returns (filtered_dataset, ValidationReport)
  ▼
PromptTemplate.format_train()      ← Apply template (Alpaca/ChatML/Llama3/Mistral)
  │                                   Output: single "text" column
  ▼
DatasetSplitter.split()            ← Train/Val/Test splits (configurable ratios)
  │
  ▼
FineTuneTrainer                    ← TrainingConfig → TrainingArguments
  ├── setup_training_args()           CPU fallback (no bf16, use_cpu=True)
  ├── setup_trainer() → SFTTrainer    max_seq_length, packing=False
  ├── Callbacks                       EarlyStopping, Checkpoint, W&B, Logging
  └── train() → TrainResult           metrics, checkpoint_path, training_time
  │
  ▼
save_model() → outputs/final/     ← Adapter weights + tokenizer
```

### Evaluation Pipeline

```
Trained Model + Test Dataset
  │
  ▼
ModelEvaluator
  ├── evaluate_perplexity()        ← Batched cross-entropy loss → exp(avg_loss)
  ├── evaluate_generation()        ← Generate → compute BLEU, ROUGE-1/2/L
  ├── evaluate_task_accuracy()     ← Generate → exact-match comparison
  └── evaluate_all()               ← Returns EvaluationResult
  │
  ▼
ModelBenchmark
  ├── add_model(base)              ← Evaluate base model
  ├── add_model(finetuned)         ← Evaluate fine-tuned model
  └── compare() → ComparisonReport ← Best model, improvement stats
  │
  ▼
HumanEvalGenerator
  ├── generate_samples()           ← Random sample generation
  └── export_markdown/csv/json()   ← Human review artifacts
```

### Inference Pipeline

```
Model Checkpoint
  │
  ▼
ModelLoader
  ├── load_base_model()            ← HF model + tokenizer
  ├── load_finetuned()             ← Base + LoRA adapters
  ├── load_merged()                ← Merge LoRA into base
  └── load_from_checkpoint()       ← Local checkpoint directory
  │
  ▼
Predictor
  ├── predict(prompt)              ← Single generation
  ├── predict_stream(prompt)       ← Token-by-token streaming
  └── predict_batch(prompts)       ← Batched generation
  │
  ▼
FastAPI Server
  ├── POST /predict                ← Single prediction
  ├── POST /predict/stream         ← Streaming response
  ├── POST /predict/batch          ← Batch prediction
  ├── GET  /health                 ← Health check
  └── GET  /model/info             ← Model metadata
```

### Deployment Pipeline

```
Model Checkpoint
  │
  ▼
ModelRegistry
  ├── register(path, name, version, metrics)
  ├── get_best(name, metric)       ← Find best version
  └── update_status(deployed)
  │
  ▼
SageMakerDeployer
  ├── package_model()              ← tar.gz → S3
  ├── create_model()               ← SageMaker model resource
  ├── deploy_endpoint()            ← Create/update endpoint
  ├── invoke_endpoint()            ← Send inference request
  └── delete_endpoint()            ← Cleanup all resources
```

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| **dataclass for TrainingConfig** | HF Trainer serialization compatibility (Pydantic breaks some HF internals) |
| **Pydantic Settings for env** | Automatic .env loading, validation, and type coercion |
| **SFTTrainer (TRL)** | Handles tokenization, data collation, and text formatting automatically |
| **packing=False** | One sample per sequence for predictable training behavior |
| **Built-in BLEU/ROUGE** | No external `evaluate` library dependency; simple n-gram implementations |
| **JSON model registry** | No database needed; sufficient for single-user workflows |
| **CPU fallback everywhere** | Allows dry-run testing and CI without GPU |
| **structlog for logging** | Structured key-value logging for production observability |
| **Factory pattern for templates** | Easy to add new templates; runtime registration |
| **Separate Dockerfiles** | Training image (CUDA + full deps) vs inference (minimal + CPU) |

## Module Dependencies

```
config/settings.py          ← No internal deps (reads env)
config/training_config.py   ← No internal deps (pure dataclass)
utils/logger.py             ← No internal deps (structlog setup)
utils/wandb_utils.py        ← config/settings

data/*                      ← utils/logger
training/*                  ← config/training_config, utils/logger
evaluation/*                ← utils/logger
inference/*                 ← utils/logger
deployment/*                ← config/settings, utils/logger
```
