# Deployment Guide

## Local Development

### Setup

```bash
# Install all dependencies
pip install -e ".[dev]"

# Configure environment
cp .env.example .env
# Edit .env with your credentials
```

### Local Inference

```bash
# Interactive REPL
python -m scripts.inference --model outputs/final --mode repl

# API server
python -m scripts.inference --model outputs/final --mode server --port 8000

# Single prediction
python -m scripts.inference --model outputs/final --mode single \
  --prompt "Explain machine learning"
```

## Docker

### Training Image

```bash
# Build
docker build -f docker/Dockerfile.train -t llm-ft-train .

# Run training (with GPU)
docker run --gpus all -v $(pwd)/outputs:/app/outputs llm-ft-train \
  --config configs/mistral_7b_qlora.yaml

# Dry run (no GPU)
docker run -v $(pwd)/outputs:/app/outputs llm-ft-train \
  --config configs/mistral_7b_qlora.yaml --dry-run
```

### Inference Image

```bash
# Build
docker build -f docker/Dockerfile.inference -t llm-ft-inference .

# Run server (mount model directory)
docker run -p 8000:8000 -v /path/to/model:/models/model llm-ft-inference

# Test
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello world"}'
```

## AWS SageMaker Deployment

### Prerequisites

1. **AWS Account** with SageMaker access
2. **IAM Role** with SageMaker and S3 permissions
3. **S3 Bucket** for model artifacts

### Required Environment Variables

```bash
AWS_REGION=us-east-1
AWS_SAGEMAKER_ROLE=arn:aws:iam::<account-id>:role/SageMakerExecutionRole
S3_BUCKET=your-model-bucket
```

### Required IAM Permissions

The SageMaker execution role needs:

```json
{
  "Effect": "Allow",
  "Action": [
    "sagemaker:CreateModel",
    "sagemaker:CreateEndpointConfig",
    "sagemaker:CreateEndpoint",
    "sagemaker:UpdateEndpoint",
    "sagemaker:DeleteEndpoint",
    "sagemaker:DeleteEndpointConfig",
    "sagemaker:DeleteModel",
    "sagemaker:DescribeEndpoint",
    "sagemaker:InvokeEndpoint",
    "s3:GetObject",
    "s3:PutObject",
    "s3:ListBucket",
    "ecr:GetAuthorizationToken",
    "ecr:BatchGetImage",
    "logs:CreateLogGroup",
    "logs:CreateLogStream",
    "logs:PutLogEvents"
  ],
  "Resource": "*"
}
```

### Step-by-Step Deployment

```bash
# 1. Verify AWS permissions
python -c "
from src.deployment.infrastructure import AWSInfrastructure
infra = AWSInfrastructure()
print(infra.verify_permissions())
"

# 2. Estimate costs
python -m scripts.deploy --action estimate \
  --instance-type ml.g5.xlarge --hours 24

# 3. Package model to S3
python -m scripts.deploy --action package \
  --model-path outputs/final

# 4. Deploy endpoint
python -m scripts.deploy --action deploy \
  --model-path outputs/final \
  --endpoint-name my-llm-endpoint \
  --instance-type ml.g5.xlarge

# 5. Test endpoint
python -m scripts.deploy --action invoke \
  --endpoint-name my-llm-endpoint \
  --prompt "Explain transformers in one sentence"

# 6. Check status
python -m scripts.deploy --action status \
  --endpoint-name my-llm-endpoint

# 7. Clean up (when done)
python -m scripts.deploy --action delete \
  --endpoint-name my-llm-endpoint
```

### Cost Estimation

| Instance Type | GPU | vCPU | RAM | Cost/hr |
|--------------|-----|------|-----|---------|
| ml.g4dn.xlarge | T4 (16GB) | 4 | 16 GB | $0.74 |
| ml.g5.xlarge | A10G (24GB) | 4 | 16 GB | $1.41 |
| ml.g5.2xlarge | A10G (24GB) | 8 | 32 GB | $1.52 |
| ml.p3.2xlarge | V100 (16GB) | 8 | 61 GB | $3.83 |

**Recommendations:**
- **Development/Testing:** ml.g4dn.xlarge (~$18/day)
- **Production (7B models):** ml.g5.xlarge (~$34/day)
- **Production (13B+ models):** ml.g5.2xlarge (~$36/day)

### Monitoring

```bash
# Check endpoint health via CLI
python -m scripts.deploy --action status --endpoint-name my-llm-endpoint

# CloudWatch metrics are automatically available:
# - Invocations
# - ModelLatency
# - OverheadLatency
# - InvocationModelErrors
```

## GitHub Actions Deployment

### Automated Training

Trigger via GitHub Actions UI:
1. Go to Actions tab
2. Select "Train" workflow
3. Click "Run workflow"
4. Configure: config file, dataset, dry run option

### Automated Deployment

Trigger via GitHub Actions UI:
1. Go to Actions tab
2. Select "Deploy" workflow
3. Click "Run workflow"
4. Configure: model artifact, instance type, endpoint name

The workflow will:
- Deploy to SageMaker
- Run a smoke test
- Rollback on failure
- Post deployment summary

### Required GitHub Secrets

| Secret | Description |
|--------|-------------|
| `AWS_ACCESS_KEY_ID` | AWS access key |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key |
| `AWS_REGION` | AWS region (default: us-east-1) |
