.PHONY: install test lint format train evaluate deploy docker-build docker-train docker-inference clean help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-25s\033[0m %s\n", $$1, $$2}'

install: ## Install all dependencies (including dev)
	pip install -e ".[dev]"

test: ## Run all tests with coverage
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-fail-under=80

test-unit: ## Run unit tests only
	pytest tests/unit/ -v

test-integration: ## Run integration tests only
	pytest tests/integration/ -v

lint: ## Run linter and type checker
	ruff check src/ tests/
	mypy src/

format: ## Auto-format code
	ruff check --fix src/ tests/
	ruff format src/ tests/

train: ## Run training (CONFIG=path/to/config.yaml)
	python -m scripts.train --config $(CONFIG)

evaluate: ## Run evaluation (MODEL=path/to/model)
	python -m scripts.evaluate --model $(MODEL)

deploy: ## Deploy to SageMaker (MODEL=path/to/model)
	python -m scripts.deploy --action deploy --model-path $(MODEL)

deploy-status: ## Check endpoint status (ENDPOINT=name)
	python -m scripts.deploy --action status --endpoint-name $(ENDPOINT)

deploy-delete: ## Delete endpoint (ENDPOINT=name)
	python -m scripts.deploy --action delete --endpoint-name $(ENDPOINT)

inference-repl: ## Start inference REPL (MODEL=path/to/model)
	python -m scripts.inference --model $(MODEL) --mode repl

inference-server: ## Start inference API server (MODEL=path/to/model)
	python -m scripts.inference --model $(MODEL) --mode server

docker-train: ## Build training Docker image
	docker build -f docker/Dockerfile.train -t llm-ft-train .

docker-inference: ## Build inference Docker image
	docker build -f docker/Dockerfile.inference -t llm-ft-inference .

docker-build: ## Build both Docker images
	$(MAKE) docker-train
	$(MAKE) docker-inference

clean: ## Clean generated files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ build/ .coverage htmlcov/
