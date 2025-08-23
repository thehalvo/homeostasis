.PHONY: setup install test test-all lint format security run demo clean help

# Default virtual environment directory
VENV = venv

help:
	@echo "Available commands:"
	@echo "  make setup       - Create virtual environment and install dev dependencies"
	@echo "  make install     - Install package in development mode"
	@echo "  make test        - Run basic pytest tests"
	@echo "  make test-all    - Run ALL tests (same as GitHub Actions)"
	@echo "  make lint        - Run linting checks"
	@echo "  make format      - Auto-format code with black and isort"
	@echo "  make security    - Run security checks"
	@echo "  make run         - Run the example service"
	@echo "  make demo        - Run the orchestrator demo"
	@echo "  make clean       - Clean up cache and temp files"

setup:
	python -m venv $(VENV)
	$(VENV)/bin/pip install -e ".[dev]"

install:
	pip install -e ".[dev]"

test:
	pytest tests/

test-all:
	@./run_all_tests_local.sh

lint:
	flake8 modules/ services/ orchestrator/ tests/
	black --check modules/ services/ orchestrator/ tests/
	isort --check modules/ services/ orchestrator/ tests/

format:
	black modules/ services/ orchestrator/ tests/
	isort modules/ services/ orchestrator/ tests/

security:
	@echo "Running security checks..."
	@safety check -r requirements.txt
	@bandit -r modules/ tests/ -ll
	@detect-secrets scan --all-files

run:
	uvicorn services.example_service.app:app --reload

demo:
	python orchestrator/orchestrator.py

clean:
	rm -rf __pycache__
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".coverage" -delete