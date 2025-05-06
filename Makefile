.PHONY: setup install test lint format run demo clean

# Default virtual environment directory
VENV = venv

setup:
	python -m venv $(VENV)
	$(VENV)/bin/pip install -e ".[dev]"

install:
	pip install -e ".[dev]"

test:
	pytest tests/

lint:
	flake8 modules/ services/ orchestrator/ tests/
	black --check modules/ services/ orchestrator/ tests/
	isort --check modules/ services/ orchestrator/ tests/

format:
	black modules/ services/ orchestrator/ tests/
	isort modules/ services/ orchestrator/ tests/

run:
	uvicorn services.example_service.app:app --reload

demo:
	python orchestrator/orchestrator.py

clean:
	rm -rf __pycache__
	rm -rf *.egg-info
	rm -rf .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +