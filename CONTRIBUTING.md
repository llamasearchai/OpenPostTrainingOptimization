# Contributing to OpenPostTrainingOptimizations

Thank you for your interest in contributing to OpenPostTrainingOptimizations! This document provides guidelines and instructions for contributing.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. Please read CODE_OF_CONDUCT.md before contributing.

## Development Setup

### Prerequisites

- Python 3.10 or higher
- Git
- uv, tox, or hatch for package management

### Setup with uv

```bash
# Clone the repository
git clone https://github.com/nikjois/openposttraining.git
cd openposttraining

# Install with uv
uv sync
uv pip install -e ".[dev,serve,agents,data]"

# Install pre-commit hooks
pre-commit install
```

### Setup with virtualenv

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev,serve,agents,data]"

# Install pre-commit hooks
pre-commit install
```

### Setup with hatch

```bash
# Install dependencies
hatch env create

# Run development environment
hatch shell
```

## Development Workflow

### Running Tests

```bash
# Run all tests
pytest -q

# Run with coverage
pytest --cov=openposttraining --cov-report=term-missing

# Run with tox (all environments)
tox -q -p auto

# Run with hatch
hatch run test
```

### Code Quality

```bash
# Format code
ruff format src tests
black src tests
isort src tests

# Run linters
ruff check src tests

# Type checking
mypy src

# Security scanning
bandit -r src -ll

# All quality checks
make all-checks
```

### Running the Application

```bash
# CLI
opt status
opt quantize -m gpt2 --method int8 -o outputs/test

# TUI
opt-tui

# With Docker
docker build -t openposttraining:test -f docker/Dockerfile .
docker run openposttraining:test opt status
```

## Contribution Guidelines

### Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Make your changes
4. Add tests for your changes
5. Ensure all tests pass: `pytest -q`
6. Ensure code quality: `make all-checks`
7. Commit your changes: `git commit -m "Description of changes"`
8. Push to your fork: `git push origin feature/your-feature-name`
9. Open a Pull Request

### Commit Message Guidelines

- Use clear, descriptive commit messages
- Start with a verb in present tense (e.g., "Add", "Fix", "Update")
- Reference issue numbers when applicable: "Fix #123: Description"
- Keep first line under 72 characters
- Provide detailed description in commit body if needed

### Code Style

- Follow PEP 8 style guidelines
- Use type hints for function signatures
- Write docstrings for all public functions and classes
- Keep functions focused and modular
- No emojis in code or documentation
- No placeholders or stubs in production code

### Testing Requirements

- All new features must include tests
- Aim for 100% code coverage
- Include unit tests, integration tests, and e2e tests where applicable
- Tests must pass on Python 3.10, 3.11, and 3.12
- Tests must pass on Linux and macOS

### Documentation

- Update README.md if adding user-facing features
- Add docstrings to all public APIs
- Update CHANGELOG.md with your changes
- Create or update documentation in docs/ for significant features

## Project Structure

```
openposttraining/
├── src/openposttraining/    # Main package
│   ├── agents/              # OpenAI Agents SDK integration
│   ├── cli.py               # CLI interface
│   ├── core/                # Core optimization modules
│   ├── deployment/          # Model serving
│   ├── integrations/        # Ollama, llm, Datasette
│   ├── kernels/             # CUDA/Triton kernels
│   ├── models/              # LLM optimizer
│   ├── profiling/           # Performance profiling
│   ├── tui/                 # Terminal UI
│   └── utils/               # Utilities
├── tests/                   # Test suite
├── docs/                    # Documentation
├── docker/                  # Docker configuration
└── .github/                 # CI/CD workflows
```

## Getting Help

- Open an issue for bugs or feature requests
- Join our Discord community for discussions
- Check existing issues and PRs before creating new ones

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.

## Author

Nik Jois

Thank you for contributing to OpenPostTrainingOptimizations!

