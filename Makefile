# AFS Development Makefile

.PHONY: help install install-dev test test-unit test-integration test-cov lint format type-check clean build docs docker-build docker-run

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
RED := \033[0;31m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)AFS - Advanced Fine-tuning System$(NC)"
	@echo ""
	@echo "$(GREEN)Available targets:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(BLUE)%-20s$(NC) %s\n", $$1, $$2}'

install: ## Install package and dependencies
	pip install -e .

install-dev: ## Install package with development dependencies
	pip install -e ".[dev,test,docs]"

test: ## Run all tests
	pytest tests/ -v

test-unit: ## Run unit tests only (fast)
	pytest tests/ -v -m "not slow and not integration and not requires_gpu"

test-integration: ## Run integration tests
	pytest tests/ -v -m "integration and not requires_gpu"

test-slow: ## Run slow tests
	pytest tests/ -v -m "slow"

test-cov: ## Run tests with coverage report
	pytest tests/ -v \
		--cov=src/afs \
		--cov-report=term \
		--cov-report=html \
		--cov-report=xml

test-parallel: ## Run tests in parallel
	pytest tests/ -v -n auto -m "not slow and not integration"

lint: ## Run ruff linter
	ruff check src/ tests/

lint-fix: ## Run ruff linter with auto-fix
	ruff check --fix src/ tests/

format: ## Format code with black and isort
	black src/ tests/
	isort src/ tests/

format-check: ## Check code formatting without changes
	black --check src/ tests/
	isort --check src/ tests/

type-check: ## Run mypy type checker
	mypy src/afs

quality: lint format-check type-check ## Run all quality checks

clean: ## Clean build artifacts and caches
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean ## Build distribution packages
	python -m build

docs: ## Build documentation
	cd docs && make html

docs-serve: docs ## Build and serve documentation
	cd docs/_build/html && python -m http.server 8000

docker-build: ## Build Docker image
	docker build -t afs:latest .

docker-run: ## Run Docker container
	docker run -it --gpus all -v $(PWD):/workspace afs:latest

train-example: ## Run example training (Majora v1)
	python scripts/train_majora.py \
		--data models/majora_v1_training.jsonl \
		--output /tmp/majora-v1-test \
		--epochs 1 \
		--max-samples 100

eval-example: ## Run example evaluation
	python scripts/run_eval.py \
		--model http://localhost:8080 \
		--eval evaluations/majora_v1_oracle_eval.jsonl \
		--output /tmp/eval_results.json

pipeline-example: ## Run example training pipeline
	python -m afs.cli.pipeline run \
		--input models/example_data.jsonl \
		--output /tmp/pipeline_output \
		--score-quality

# Continuous Integration targets
ci: install-dev lint format-check type-check test-unit ## Run CI checks

ci-full: install-dev quality test test-integration ## Run full CI suite

# Development workflow shortcuts
dev-setup: install-dev ## Setup development environment
	@echo "$(GREEN)Development environment ready!$(NC)"
	@echo "Run 'make test' to run tests"
	@echo "Run 'make help' to see all available commands"

watch-test: ## Watch files and run tests on changes
	@echo "$(BLUE)Watching for changes...$(NC)"
	@while true; do \
		find src tests -name "*.py" | entr -d make test-unit; \
		sleep 1; \
	done

# Quick development checks
quick: lint type-check test-unit ## Quick checks before commit

# Pre-commit hook
pre-commit: format lint type-check test-unit ## Run pre-commit checks
	@echo "$(GREEN)✓ All checks passed!$(NC)"
