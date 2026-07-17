# AFS Development Makefile

.PHONY: help setup install install-dev test test-unit test-integration test-cov test-parallel lint lint-all lint-fix format format-check type-check clean build package-check docs docs-serve release-check ci ci-full quick check pre-commit docker-build docker-run

BLUE := \033[0;34m
GREEN := \033[0;32m
NC := \033[0m
SYSTEM_PYTHON ?= python3
VENV ?= .venv
PYTHON ?= $(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/python,$(SYSTEM_PYTHON))
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python
AFS := ./scripts/afs
PYTEST := $(PYTHON) -m pytest
RUFF := $(PYTHON) -m ruff
MYPY := $(PYTHON) -m mypy
MKDOCS := $(PYTHON) -m mkdocs

help: ## Show this help message
	@echo "$(BLUE)AFS - Agentic File System$(NC)"
	@echo ""
	@echo "$(GREEN)Available targets:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(BLUE)%-20s$(NC) %s\n", $$1, $$2}'

setup: ## Create .venv and install dev/test/docs dependencies
	$(SYSTEM_PYTHON) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PIP) install -e ".[dev,test,docs]"
	@echo "$(GREEN)✓ AFS dev environment ready. Try: $(AFS) --help$(NC)"

install: ## Install package and runtime dependencies into the active environment
	$(PYTHON) -m pip install -e .

install-dev: ## Install package with development dependencies into the active environment
	$(PYTHON) -m pip install -e ".[dev,test,docs]"

test: ## Run all tests quietly
	$(PYTEST) -q

test-unit: ## Run unit tests only
	$(PYTEST) tests/ -v -m "not slow and not integration and not requires_gpu"

test-integration: ## Run integration tests
	$(PYTEST) tests/ -v -m "integration and not requires_gpu"

test-cov: ## Run tests with coverage report
	$(PYTEST) tests/ -v --cov=src/afs --cov-report=term --cov-report=html --cov-report=xml

test-parallel: ## Run tests in parallel
	$(PYTEST) tests/ -v -n auto -m "not slow and not integration"

lint: ## Run CI lint surface
	$(RUFF) check src/ tests/

lint-all: ## Run broader lint across public Python surfaces
	$(RUFF) check src/ tests/ examples/ scripts/*.py tools/*.py

lint-fix: ## Run ruff linter with auto-fix on CI surface
	$(RUFF) check --fix src/ tests/

format: ## Format code with ruff format
	$(RUFF) format src/ tests/ examples/

format-check: ## Check code formatting without changes
	$(RUFF) format --check src/ tests/ examples/

type-check: ## Run mypy on the whole afs package (ratcheted baseline; new modules fully checked)
	$(MYPY) -p afs
	$(MYPY) scripts/check_release.py

package-check: ## Build wheel/sdist and smoke-test package metadata
	$(PYTHON) -m pip install --upgrade build >/dev/null
	$(PYTHON) -m build
	$(PYTHON) scripts/check_release.py --dist

docs: ## Build MkDocs documentation
	$(MKDOCS) build --strict

docs-serve: ## Serve documentation locally
	$(MKDOCS) serve

release-check: lint test package-check docs ## Run release readiness checks
	$(PYTHON) scripts/check_release.py

quality: lint format-check type-check ## Run quality checks used during development

clean: ## Clean build artifacts and caches
	rm -rf build/ dist/ *.egg-info htmlcov/ .coverage .pytest_cache/ .mypy_cache/ .ruff_cache/ site/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean package-check ## Build distribution packages from a clean tree

docker-build: ## Build Docker image
	docker build -t afs:latest .

docker-run: ## Run Docker container
	docker run -it -v $(PWD):/workspace afs:latest

ci: install-dev lint type-check test-unit package-check ## Run CI-equivalent local checks

ci-full: install-dev quality test test-integration package-check docs ## Run full CI and docs checks

quick: lint test package-check ## Quick checks before commit

check: quick ## Run the standard local share-readiness check

pre-commit: format lint type-check test-unit ## Run pre-commit-style checks
	@echo "$(GREEN)✓ All checks passed!$(NC)"
