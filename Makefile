.PHONY: install test lint format train evaluate deploy docker-build clean help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install all dependencies (including dev)
	pip install -e ".[dev]"

test: ## Run tests with coverage
	pytest tests/ -v --cov=src --cov-report=term-missing

test-unit: ## Run unit tests only
	pytest tests/unit/ -v

test-integration: ## Run integration tests only
	pytest tests/integration/ -v -m integration

lint: ## Run linter
	ruff check src/ tests/
	mypy src/

format: ## Format code
	ruff check --fix src/ tests/
	ruff format src/ tests/

train: ## Run training (pass CONFIG=path/to/config.yaml)
	python -m scripts.train --config $(CONFIG)

evaluate: ## Run evaluation (pass CONFIG=path/to/config.yaml)
	python -m scripts.evaluate --config $(CONFIG)

deploy: ## Deploy model to SageMaker (pass CONFIG=path/to/config.yaml)
	python -m scripts.deploy --config $(CONFIG)

docker-build: ## Build Docker image
	docker build -f docker/Dockerfile -t llm-fine-tuning-pipeline .

clean: ## Clean generated files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ build/ .coverage htmlcov/
