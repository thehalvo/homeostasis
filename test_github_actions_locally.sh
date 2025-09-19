#!/bin/bash

# This script simulates the GitHub Actions environment locally
# to help you debug test failures before pushing

set -e

# Activate virtual environment if it exists
if [ -f venv/bin/activate ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

echo "==================================="
echo "Testing GitHub Actions E2E workflow"
echo "==================================="

# Create test directories like GitHub Actions does
mkdir -p logs
mkdir -p test_results

# Set the environment variables that GitHub Actions would set
export PYTHONPATH=$PWD
export USE_MOCK_TESTS=true

echo "Running E2E tests with the same command as GitHub Actions..."
python -m pytest tests/e2e/healing_scenarios/ \
    -v \
    --tb=short \
    --json-report \
    --json-report-file=test_results/report.json \
    --html=test_results/report.html \
    --self-contained-html \
    --timeout=1800

echo "Test completed! Check test_results/ for output."