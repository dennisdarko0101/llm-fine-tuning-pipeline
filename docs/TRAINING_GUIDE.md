# QLoRA Fine-Tuning Guide

## What is QLoRA?

QLoRA (Quantized Low-Rank Adaptation) combines two techniques to make LLM fine-tuning possible on consumer hardware:

1. **4-bit Quantization** — Compresses the base model weights from 16-bit to 4-bit, reducing memory by ~4x
2. **LoRA Adapters** — Trains only small adapter matrices (~0.1-1% of total parameters) instead of all weights

### Memory comparison for a 7B parameter model

| Method | Model Memory | Trainable Params | Min GPU |
|--------|-------------|------------------|---------|
| Full fine-tuning (FP32) | ~28 GB | 7B (100%) | 4x A100 80GB |
| Full fine-tuning (FP16) | ~14 GB | 7B (100%) | 2x A100 80GB |
| LoRA (FP16) | ~14 GB | ~7M (0.1%) | 1x A100 40GB |
| **QLoRA (4-bit)** | **~3.5 GB** | **~7M (0.1%)** | **1x RTX 4090 24GB** |

## How LoRA Works

LoRA decomposes weight updates into low-rank matrices. Instead of updating the full weight matrix W (d × d), it learns two smaller matrices:

```
W_new = W_frozen + B × A

Where:
  W_frozen: Original weights (d × d) — NOT updated
  A: Down-projection (d × r) — trained
  B: Up-projection (r × d) — trained
  r: Rank (typically 8-64, much smaller than d)
```

For a layer with d=4096 and r=16:
- Full update: 4096 × 4096 = 16.7M parameters
- LoRA update: (4096 × 16) + (16 × 4096) = 131K parameters (**127x fewer**)

### Why it works

Most fine-tuning tasks can be captured in a low-dimensional subspace. The rank `r` controls expressiveness — higher rank = more capacity but more memory/compute.

## Quantization Details

### NF4 (Normal Float 4-bit)

NF4 is optimized for normally distributed weights (which neural network weights tend to be). It provides better accuracy than standard 4-bit integers.

### Double Quantization

Quantization uses scaling constants per block of weights. Double quantization quantizes these constants too, saving ~0.37 bits per parameter (~0.5 GB for a 7B model).

### Configuration

```python
BitsAndBytesConfig(
    load_in_4bit=True,              # Enable 4-bit quantization
    bnb_4bit_quant_type="nf4",      # NF4 > FP4 for most cases
    bnb_4bit_compute_dtype=bfloat16, # Compute in bf16 for speed
    bnb_4bit_use_double_quant=True,  # Extra memory savings
)
```

## Hyperparameter Guide

### LoRA Parameters

| Parameter | Default | Guidance |
|-----------|---------|----------|
| `lora_r` | 16 | Start with 8-16. Increase to 32-64 for complex tasks. Higher = more capacity but slower. |
| `lora_alpha` | 32 | Usually 2× `lora_r`. Controls the scaling of LoRA updates. |
| `lora_dropout` | 0.05 | Regularization. 0.05-0.1 for most tasks. Increase if overfitting. |

### Target Modules

Which layers to apply LoRA to. More modules = more capacity + more memory.

| Strategy | Modules | When to use |
|----------|---------|-------------|
| Minimal | `q_proj, v_proj` | Quick experiments, small datasets |
| Standard | `q_proj, k_proj, v_proj, o_proj` | Most fine-tuning tasks |
| **Full (default)** | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` | Best quality, our default |

### Training Parameters

| Parameter | Default | Why |
|-----------|---------|-----|
| `learning_rate` | 2e-4 | QLoRA paper recommendation. Higher than full fine-tuning because we're training fewer params. |
| `lr_scheduler` | cosine | Smooth decay prevents loss spikes. Linear also works. |
| `warmup_ratio` | 0.03 | Gradual LR increase stabilizes early training. |
| `num_epochs` | 3 | 1-3 for large datasets, 3-5 for small. Watch for overfitting. |
| `batch_size` | 4 | Per-device. Effective batch = batch × gradient_accumulation. |
| `gradient_accumulation` | 4 | Simulates larger batch (4×4=16 effective). |
| `max_grad_norm` | 0.3 | Aggressive clipping stabilizes QLoRA training. |
| `weight_decay` | 0.01 | Light regularization. |
| `optim` | paged_adamw_32bit | Paged optimizer offloads states to CPU when GPU memory is full. |

### Mixed Precision

- **bf16=True** (default): Use bfloat16 on Ampere+ GPUs (A100, RTX 3090+). Better numeric range than fp16.
- **fp16=True**: Fallback for older GPUs (V100, T4). May need loss scaling.
- Never enable both.

## Common Issues and Solutions

### Out of Memory (OOM)

1. **Reduce batch size** — most impactful
2. **Increase gradient accumulation** — maintains effective batch size
3. **Reduce max_seq_length** — memory scales linearly with sequence length
4. **Reduce lora_r** — fewer trainable parameters
5. **Use fewer target_modules** — less LoRA overhead

### Loss Not Decreasing

1. **Check learning rate** — try 1e-4 to 5e-4 range
2. **Check data quality** — run validation pipeline
3. **Increase epochs** — small datasets may need more passes
4. **Check template format** — wrong template = model sees garbage

### Overfitting (train loss drops, eval loss increases)

1. **Increase dropout** — 0.1 or 0.15
2. **Reduce epochs** — stop earlier
3. **Reduce lora_r** — less model capacity
4. **Add more training data**
5. **Enable early stopping** — `EarlyStoppingCallback(patience=3)`

### Catastrophic Forgetting

1. **Lower learning rate** — 1e-5 range
2. **Fewer epochs** — 1-2 may suffice
3. **Reduce lora_r** — constrain adaptation
4. **Use evaluation on diverse tasks** — catch degradation early

## Running Training

### Basic usage

```bash
python scripts/train.py \
    --config configs/mistral_7b_qlora.yaml \
    --dataset data/sample/sample_alpaca.json
```

### Dry run (verify setup without training)

```bash
python scripts/train.py \
    --config configs/mistral_7b_qlora.yaml \
    --dataset data/sample/sample_alpaca.json \
    --dry-run
```

### With custom template and output

```bash
python scripts/train.py \
    --config configs/llama3_8b_qlora.yaml \
    --dataset data/sample/sample_alpaca.json \
    --template llama3 \
    --output-dir outputs/my-experiment
```

### From HuggingFace dataset

```bash
python scripts/train.py \
    --config configs/mistral_7b_qlora.yaml \
    --dataset tatsu-lab/alpaca \
    --hf
```
