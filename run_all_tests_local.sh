#!/bin/bash

# Script to run all GitHub Actions tests locally before pushing
# This ensures code passes all checks before uploading to GitHub

set -e  # Exit on error

echo "============================================="
echo "Running ALL GitHub Actions Tests Locally"
echo "============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo -e "${YELLOW}Warning: Virtual environment not activated. Activating venv...${NC}"
    source venv/bin/activate
fi

# Function to run a test and report status
run_test() {
    local test_name=$1
    local test_command=$2
    
    echo -e "\n${YELLOW}Running: $test_name${NC}"
    echo "Command: $test_command"
    echo "-----------------------------------"
    
    if eval "$test_command"; then
        echo -e "${GREEN}✓ $test_name passed${NC}"
        return 0
    else
        echo -e "${RED}✗ $test_name failed${NC}"
        return 1
    fi
}

# Track failures
FAILED_TESTS=()

# 1. Basic linting (from CI workflow)
echo -e "\n${YELLOW}=== LINTING CHECKS ===${NC}"
run_test "Flake8 Syntax Check" \
    "flake8 modules/ services/ orchestrator/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics" || FAILED_TESTS+=("Flake8 Syntax")

# 2. Extended Flake8 checks
run_test "Flake8 Full Check" \
    "flake8 modules/ services/ orchestrator/ tests/ --max-line-length=120 --exclude=venv,__pycache__,.git --ignore=W293,W291,E128,E501,W292" || FAILED_TESTS+=("Flake8 Full")

# 3. Black formatting check
echo -e "\n${YELLOW}=== FORMATTING CHECKS ===${NC}"
run_test "Black Formatting" \
    "black --check modules/ services/ orchestrator/ tests/" || FAILED_TESTS+=("Black")

# 4. isort import ordering
run_test "Import Sorting" \
    "isort --check-only modules/ services/ orchestrator/ tests/" || FAILED_TESTS+=("isort")

# 5. Type checking with mypy
echo -e "\n${YELLOW}=== TYPE CHECKING ===${NC}"
run_test "MyPy Type Check" \
    "mypy modules/ services/ orchestrator/ --ignore-missing-imports" || FAILED_TESTS+=("MyPy")

# 6. Security checks (from security-scanning workflow)
echo -e "\n${YELLOW}=== SECURITY CHECKS ===${NC}"

# Install dependencies if needed
if ! command -v safety &> /dev/null; then
    echo "Installing safety..."
    pip install safety
fi

run_test "Safety Dependency Check" \
    "safety check -r requirements.txt" || FAILED_TESTS+=("Safety")

if ! command -v bandit &> /dev/null; then
    echo "Installing bandit..."
    pip install bandit
fi

run_test "Bandit Security Check" \
    "bandit -r modules/ tests/ -ll --exclude=/venv/,/modules/healing/venv/" || FAILED_TESTS+=("Bandit")

# 7. Unit tests
echo -e "\n${YELLOW}=== UNIT TESTS ===${NC}"
run_test "Pytest Unit Tests" \
    "python -m pytest tests/unit/ -v" || FAILED_TESTS+=("Unit Tests")

# 8. Integration tests
echo -e "\n${YELLOW}=== INTEGRATION TESTS ===${NC}"
run_test "Pytest Integration Tests" \
    "python -m pytest tests/integration/ -v" || FAILED_TESTS+=("Integration Tests")

# 9. E2E tests (basic subset for local testing)
echo -e "\n${YELLOW}=== E2E TESTS (Subset) ===${NC}"
run_test "E2E Basic Healing Scenarios" \
    "PYTHONPATH=$PWD python -m pytest tests/e2e/healing_scenarios/test_basic_healing_scenarios.py -v" || FAILED_TESTS+=("E2E Tests")

# 10. Check for secrets
echo -e "\n${YELLOW}=== SECRET DETECTION ===${NC}"
if ! command -v detect-secrets &> /dev/null; then
    echo "Installing detect-secrets..."
    pip install detect-secrets
fi

run_test "Secret Detection" \
    "detect-secrets scan --all-files | grep -v 'venv/' | grep -v 'modules/healing/venv/'" || FAILED_TESTS+=("Secret Detection")

# 11. Check requirements files
echo -e "\n${YELLOW}=== REQUIREMENTS VALIDATION ===${NC}"
run_test "Requirements Installation Test" \
    "pip install -r requirements.txt -r requirements-dev.txt --dry-run --quiet" || FAILED_TESTS+=("Requirements")

# 12. Documentation build test
echo -e "\n${YELLOW}=== DOCUMENTATION ===${NC}"
if [ -d "docs" ]; then
    run_test "Sphinx Documentation Build" \
        "cd docs && make html" || FAILED_TESTS+=("Documentation")
fi

# Summary
echo -e "\n============================================="
echo -e "${YELLOW}TEST SUMMARY${NC}"
echo "============================================="

if [ ${#FAILED_TESTS[@]} -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed! Safe to push to GitHub.${NC}"
    exit 0
else
    echo -e "${RED}✗ Failed tests:${NC}"
    for test in "${FAILED_TESTS[@]}"; do
        echo -e "  ${RED}- $test${NC}"
    done
    echo -e "\n${RED}Fix these issues before pushing to GitHub!${NC}"
    exit 1
fi