#!/bin/bash

# test_github_actions_locally.sh - Simulate GitHub Actions environment locally
# This script creates isolated environments for each Python version and runs the same tests as CI

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Track failures
FAILED_TESTS=()
PYTHON_VERSIONS=("3.9" "3.10" "3.11")

echo -e "${BLUE}=== GitHub Actions Local Test Runner ===${NC}"
echo -e "${BLUE}Testing with Python versions: ${PYTHON_VERSIONS[*]}${NC}"
echo ""

# Function to print test results
print_result() {
    local test_name=$1
    local exit_code=$2
    local python_version=$3

    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}✓ $test_name (Python $python_version) passed${NC}"
    else
        echo -e "${RED}✗ $test_name (Python $python_version) failed${NC}"
        FAILED_TESTS+=("$test_name-py$python_version")
    fi
}

# Function to test with specific Python version
test_python_version() {
    local py_version=$1
    local venv_name="test_env_${py_version//./}"

    echo -e "\n${YELLOW}=== Testing with Python $py_version ===${NC}"

    # Check if Python version is available
    if ! command -v "python$py_version" &> /dev/null; then
        echo -e "${YELLOW}WARNING: Python $py_version not found, skipping...${NC}"
        return
    fi

    # Create fresh virtual environment
    echo "Creating virtual environment..."
    rm -rf "$venv_name"
    "python$py_version" -m venv "$venv_name"

    # Activate and upgrade pip
    source "$venv_name/bin/activate"
    pip install --upgrade pip setuptools wheel >/dev/null 2>&1

    # Install dependencies (simulating CI)
    echo "Installing dependencies..."
    pip install -r requirements.txt >/dev/null 2>&1
    local req_exit=$?
    print_result "Requirements installation" $req_exit $py_version

    if [ $req_exit -ne 0 ]; then
        echo -e "${RED}Failed to install requirements, skipping other tests${NC}"
        deactivate
        return
    fi

    # Install dev dependencies
    pip install -r requirements-dev.txt >/dev/null 2>&1

    # Run linting (only needs to run once, but doing per environment to catch version-specific issues)
    echo -e "\n${BLUE}Running linting checks...${NC}"

    flake8 modules/ services/ orchestrator/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics >/dev/null 2>&1
    print_result "Flake8 syntax check" $? $py_version

    # Run tests exactly as CI does
    echo -e "\n${BLUE}Running tests...${NC}"

    # Set environment variables as CI does
    export USE_MOCK_TESTS=true
    export DISABLE_PERFORMANCE_TRACKING=true

    # Run pytest with same options as CI
    python -m pytest tests/ -k "not test_concurrent_error_processing_performance" >/dev/null 2>&1
    print_result "Pytest suite" $? $py_version

    # Clean up
    deactivate
    rm -rf "$venv_name"
}

# Function to check Docker build
check_docker_build() {
    echo -e "\n${YELLOW}=== Checking Docker build ===${NC}"

    if ! command -v docker &> /dev/null; then
        echo -e "${YELLOW}WARNING: Docker not available, skipping Docker tests${NC}"
        return
    fi

    echo "Building Docker image..."
    docker build -t homeostasis-test:local-test . >/dev/null 2>&1
    print_result "Docker build" $? "N/A"

    # Clean up
    docker rmi homeostasis-test:local-test >/dev/null 2>&1
}

# Function to check workflows
check_workflows() {
    echo -e "\n${YELLOW}=== Checking GitHub workflows ===${NC}"

    # Check for Python 3.8 references
    if grep -r "'3.8'\|\"3.8\"" .github/workflows/*.yml >/dev/null 2>&1; then
        echo -e "${RED}✗ Found Python 3.8 references in workflows${NC}"
        FAILED_TESTS+=("workflow-python38-check")
    else
        echo -e "${GREEN}✓ No Python 3.8 references${NC}"
    fi

    # Check for docker-compose (should be docker compose)
    if grep -r "docker-compose " .github/workflows/*.yml | grep -v "docker-compose.yml" >/dev/null 2>&1; then
        echo -e "${RED}✗ Found 'docker-compose' commands (should be 'docker compose')${NC}"
        FAILED_TESTS+=("workflow-docker-compose-check")
    else
        echo -e "${GREEN}✓ Docker compose commands are correct${NC}"
    fi
}

# Main execution
main() {
    # Check workflows first
    check_workflows

    # Test with each Python version
    for py_version in "${PYTHON_VERSIONS[@]}"; do
        test_python_version "$py_version"
    done

    # Check Docker build
    check_docker_build

    # Summary
    echo -e "\n${BLUE}=== Test Summary ===${NC}"
    if [ ${#FAILED_TESTS[@]} -eq 0 ]; then
        echo -e "${GREEN}✓ All tests passed! Safe to push to GitHub.${NC}"
        exit 0
    else
        echo -e "${RED}✗ Failed tests: ${FAILED_TESTS[*]}${NC}"
        echo -e "${RED}Fix these issues before pushing to GitHub.${NC}"
        exit 1
    fi
}

# Run main function
main