.PHONY: help install install-dev install-uv test test-cov test-all lint format type security clean docker docker-compose docs serve-docs

help:
	@echo "OpenPostTrainingOptimizations - Makefile commands"
	@echo ""
	@echo "Installation:"
	@echo "  install          Install package with pip"
	@echo "  install-dev      Install with development dependencies"
	@echo "  install-uv       Install using uv package manager"
	@echo ""
	@echo "Testing:"
	@echo "  test             Run tests with pytest"
	@echo "  test-cov         Run tests with coverage report"
	@echo "  test-all         Run tox across all Python versions"
	@echo ""
	@echo "Quality:"
	@echo "  lint             Run linters (ruff)"
	@echo "  format           Format code (ruff, black, isort)"
	@echo "  type             Run type checking (mypy)"
	@echo "  security         Run security scans (bandit)"
	@echo "  all-checks       Run all quality checks"
	@echo ""
	@echo "Development:"
	@echo "  clean            Clean temporary files"
	@echo "  pre-commit       Install pre-commit hooks"
	@echo ""
	@echo "Docker:"
	@echo "  docker           Build Docker image"
	@echo "  docker-compose   Start services with docker-compose"
	@echo ""
	@echo "Documentation:"
	@echo "  docs             Build documentation"
	@echo "  serve-docs       Serve documentation locally"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev,agents,data,serve]"

install-uv:
	uv pip install -e ".[dev,agents,data,serve]"

test:
	pytest -q

test-cov:
	pytest --cov=openposttraining --cov-report=term-missing --cov-report=html --cov-report=xml

test-all:
	tox -q -p auto

lint:
	ruff check src tests

format:
	ruff format src tests
	black src tests
	isort src tests

type:
	mypy src

security:
	bandit -r src -ll

all-checks: format lint type security test-cov

clean:
	rm -rf build dist .eggs *.egg-info
	rm -rf .pytest_cache .mypy_cache .ruff_cache .tox .hypothesis
	rm -rf htmlcov .coverage .coverage.*
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.py[co]' -delete

pre-commit:
	pre-commit install

docker:
	docker build --target runtime -t openposttraining:latest -f docker/Dockerfile .

docker-compose:
	docker compose -f docker/docker-compose.yml up -d

docs:
	cd docs && mkdocs build

serve-docs:
	cd docs && mkdocs serve -a localhost:8080

uv-sync:
	uv sync

uv-lock:
	uv lock

uv-run-test:
	uv run pytest -q

hatch-test:
	hatch run test

hatch-cov:
	hatch run dev:cov

hatch-all:
	hatch run dev:all

virtualenv-setup:
	python -m venv .venv
	. .venv/bin/activate && pip install -e ".[dev,agents,data,serve]"

datasette-serve:
	datasette serve data/openposttraining.db -p 9000 --cors

sqlite-utils-tables:
	sqlite-utils tables data/openposttraining.db

llm-models:
	llm models

ollama-list:
	ollama list

ollama-pull-llama3:
	ollama pull llama3

test-openai-agents:
	python -m openposttraining.agents.test_integration

test-datasette:
	python -m openposttraining.integrations.datasette_test

run-tui:
	opt-tui

run-cli-status:
	opt status --device auto

sbom-generate:
	docker sbom openposttraining:latest --format spdx-json -o sbom.json

trivy-scan:
	trivy image openposttraining:latest

