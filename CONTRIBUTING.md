# Contributing

## Development Setup

```bash
# Clone the repository
git clone <repo-url>
cd llm-fine-tuning-pipeline

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install with dev dependencies
pip install -e ".[dev]"
```

## Development Workflow

1. Create a feature branch: `git checkout -b feature/my-feature`
2. Make your changes
3. Run linting: `make lint`
4. Run tests: `make test`
5. Commit with a descriptive message
6. Push and create a Pull Request

## Code Standards

### Style

- **Python 3.11+** — Use modern type hints (`str | None`, `list[str]`)
- **Ruff** — Linter and formatter (line length: 100)
- **mypy** — Strict mode enabled (`disallow_untyped_defs = true`)
- Run `make format` before committing

### Testing

- All tests must pass without GPU access
- Mock external services (AWS, HuggingFace Hub, W&B)
- Aim for 80%+ test coverage
- Place unit tests in `tests/unit/`, integration tests in `tests/integration/`

### Logging

- Use `structlog` via `from src.utils.logger import get_logger`
- Log key events with structured key-value pairs
- Example: `log.info("model_loaded", model=name, params=count)`

### Error Handling

- Raise specific exceptions with helpful messages
- Use `log.warning()` for recoverable issues
- Never silently swallow exceptions

## Project Structure

```
src/<module>/
├── __init__.py     # Public API re-exports
├── module.py       # Implementation
tests/unit/
└── test_module.py  # Tests
```

### Adding a New Module

1. Create the module file in the appropriate `src/` subdirectory
2. Add re-exports to `__init__.py`
3. Create matching test file in `tests/unit/`
4. Update `docs/HANDOFF.md` with the new module

### Adding a Prompt Template

1. Subclass `PromptTemplate` in `src/data/templates.py`
2. Implement `format()` and `format_inference()`
3. Register in `TemplateFactory._templates`
4. Add tests in `tests/unit/test_templates.py`

## Running Tests

```bash
make test             # All tests with coverage
make test-unit        # Unit tests only
make test-integration # Integration tests only

# Run specific test file
pytest tests/unit/test_evaluator.py -v

# Run specific test
pytest tests/unit/test_evaluator.py::TestEvaluationResult::test_perplexity_property -v
```

## CI/CD

Pull requests trigger the CI workflow which:
1. Lints code with Ruff
2. Type-checks with mypy
3. Runs all tests with 80% coverage threshold
4. Scans for accidentally committed secrets

All checks must pass before merging.
