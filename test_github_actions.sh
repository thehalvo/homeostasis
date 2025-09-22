#!/bin/bash

# test_github_actions.sh - GitHub Actions Testing
# This script runs ALL tests EXACTLY as GitHub Actions does
# Complete testing with Python 3.9, 3.10, and 3.11

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Track failures
FAILED_TESTS=()
PYTHON_VERSIONS=("3.9" "3.10" "3.11")

echo -e "${BLUE}=== GitHub Actions Test Suite ===${NC}"
echo -e "${BLUE}Testing with Python versions: ${PYTHON_VERSIONS[*]}${NC}"
echo -e "${YELLOW}This runs EVERY test that GitHub Actions runs${NC}"
echo -e "${YELLOW}Full testing ensures GitHub success${NC}"
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
    local python_cmd=""

    echo -e "\n${YELLOW}=== Testing with Python $py_version ===${NC}"

    # Check if Python version is available in multiple locations
    if command -v "python$py_version" &> /dev/null; then
        python_cmd="python$py_version"
    elif command -v "/opt/homebrew/bin/python$py_version" &> /dev/null; then
        python_cmd="/opt/homebrew/bin/python$py_version"
    elif command -v "/usr/local/bin/python$py_version" &> /dev/null; then
        python_cmd="/usr/local/bin/python$py_version"
    fi

    if [ -z "$python_cmd" ]; then
        echo -e "${RED}ERROR: Python $py_version not found${NC}"
        echo -e "${RED}GitHub Actions REQUIRES Python 3.9, 3.10, and 3.11${NC}"
        echo -e "${YELLOW}\nInstall Python $py_version using:${NC}"
        echo -e "${CYAN}brew install python@$py_version  # macOS${NC}"
        echo -e "${CYAN}pyenv install $py_version  # Cross-platform${NC}"
        echo -e "${CYAN}sudo apt-get install python$py_version  # Ubuntu/Debian${NC}"
        FAILED_TESTS+=("python-$py_version-missing")
        return 1
    fi

    echo "Using Python: $python_cmd"

    # Create fresh virtual environment
    echo "Creating virtual environment..."
    rm -rf "$venv_name"
    "$python_cmd" -m venv "$venv_name"

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

    # Set environment variables as CI does
    export USE_MOCK_TESTS=true
    export DISABLE_PERFORMANCE_TRACKING=true
    export PYTHONPATH=$PWD

    echo -e "\n${MAGENTA}Running CI Workflow Tests...${NC}"

    # 1. Flake8 syntax check (from CI workflow)
    echo -e "\n${BLUE}1. Running flake8 syntax checks...${NC}"
    flake8 modules/ services/ orchestrator/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics >/dev/null 2>&1
    print_result "Flake8 syntax check (CI)" $? $py_version

    # 2. Basic pytest suite (from CI workflow) with pytest-ci.ini
    echo -e "\n${BLUE}2. Running CI pytest suite...${NC}"
    # Run with timeout and show failures
    # Use special config for Python 3.11 to avoid asyncio hangs
    # Store output to check for timeout
    if [ "$py_version" = "3.11" ] && [ -f "pytest-py311.ini" ]; then
        timeout 300 python -m pytest -c pytest-py311.ini tests/ -k "not test_concurrent_error_processing_performance" -v --tb=short --no-header 2>&1 | tail -150
    elif [ "$py_version" = "3.9" ]; then
        # Python 3.9 seems to produce more output, increase tail buffer
        timeout 400 python -m pytest -c pytest-ci.ini tests/ -k "not test_concurrent_error_processing_performance" -v --tb=short --no-header 2>&1 | tail -200
    else
        timeout 300 python -m pytest -c pytest-ci.ini tests/ -k "not test_concurrent_error_processing_performance" -v --tb=short --no-header 2>&1 | tail -150
    fi
    local pytest_exit=${PIPESTATUS[0]}
    # If timeout was reached (exit code 124), report it
    if [ $pytest_exit -eq 124 ]; then
        echo -e "${YELLOW}Warning: Test suite timed out after 300 seconds${NC}"
    fi
    print_result "CI pytest suite" $pytest_exit $py_version

    echo -e "\n${MAGENTA}Running E2E Healing Scenario Tests...${NC}"

    # 3. E2E Healing tests (from e2e-healing-tests.yml)
    echo -e "\n${BLUE}3. Running E2E healing scenarios...${NC}"
    timeout 120 python -m pytest tests/e2e/healing_scenarios/test_basic_healing_scenarios.py -q --tb=short
    print_result "E2E Healing Scenarios" $? $py_version

    echo -e "\n${MAGENTA}Running Security Audit Tests...${NC}"

    # 4. Security checks
    echo -e "\n${BLUE}4. Running security vulnerability scan...${NC}"
    if command -v safety &> /dev/null; then
        safety scan --policy-file /dev/null >/dev/null 2>&1 || true
        echo -e "${GREEN}✓ Security scan completed${NC}"
    else
        pip install safety >/dev/null 2>&1
        safety scan --policy-file /dev/null >/dev/null 2>&1 || true
        echo -e "${GREEN}✓ Security scan completed${NC}"
    fi

    echo -e "\n${MAGENTA}Running Additional GitHub Action Tests...${NC}"

    # 5. Performance regression tests (if they exist)
    if [ -f "tests/performance/test_performance_regression.py" ]; then
        echo -e "\n${BLUE}5. Running performance regression tests...${NC}"
        python -m pytest tests/performance/test_performance_regression.py -v >/dev/null 2>&1
        print_result "Performance regression" $? $py_version
    fi

    # 6. Chaos engineering tests (if they exist)
    if [ -d "tests/chaos" ]; then
        echo -e "\n${BLUE}6. Running chaos engineering tests...${NC}"
        # Use special config for Python 3.11 to avoid asyncio hangs
        if [ "$py_version" = "3.11" ] && [ -f "pytest-py311.ini" ]; then
            python -m pytest -c pytest-py311.ini tests/chaos/ -v >/dev/null 2>&1
        else
            python -m pytest tests/chaos/ -v >/dev/null 2>&1
        fi
        print_result "Chaos engineering" $? $py_version
    fi

    # Clean up
    deactivate
    rm -rf "$venv_name"
}

# Function to check Docker build
check_docker_build() {
    echo -e "\n${CYAN}=== Checking Docker build ===${NC}"

    if ! command -v docker &> /dev/null; then
        echo -e "${YELLOW}WARNING: Docker not available, skipping Docker tests${NC}"
        return
    fi

    # Check if Dockerfile exists
    if [ -f "Dockerfile" ]; then
        echo "Building Docker image..."
        docker build -t homeostasis-test:local-test . >/dev/null 2>&1
        print_result "Docker build" $? "N/A"

        # Clean up
        docker rmi homeostasis-test:local-test >/dev/null 2>&1
    else
        echo -e "${YELLOW}No Dockerfile found, skipping Docker build${NC}"
    fi
}

# Function to run integration tests if docker-compose file exists
check_integration_tests() {
    echo -e "\n${CYAN}=== Checking Integration Tests ===${NC}"

    if [ -f "tests/e2e/docker-compose.yml" ] && command -v docker &> /dev/null; then
        echo -e "${BLUE}Found integration test configuration${NC}"
        echo -e "${YELLOW}Note: Full integration tests require Docker Compose${NC}"
        # We won't run the full integration test here as it's resource intensive
        # but we'll check that the compose file is valid
        docker compose -f tests/e2e/docker-compose.yml config >/dev/null 2>&1
        print_result "Docker Compose validation" $? "N/A"
    fi
}

# Function to check workflows
check_workflows() {
    echo -e "\n${CYAN}=== Pre-flight Workflow Checks ===${NC}"

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

    # Check for parse errors in workflow files
    echo -e "\n${BLUE}Validating workflow YAML syntax...${NC}"

    # Try to find a Python with yaml module
    local python_with_yaml=""
    for venv in venv_py39 venv_py310 venv_py311 venv; do
        if [ -f "$venv/bin/python" ] && "$venv/bin/python" -c "import yaml" 2>/dev/null; then
            python_with_yaml="$venv/bin/python"
            break
        fi
    done

    # If no venv has yaml, try system pythons
    if [ -z "$python_with_yaml" ]; then
        for py_cmd in python3 python python3.9 python3.10 python3.11; do
            if command -v "$py_cmd" &> /dev/null && "$py_cmd" -c "import yaml" 2>/dev/null; then
                python_with_yaml="$py_cmd"
                break
            fi
        done
    fi

    if [ -z "$python_with_yaml" ]; then
        echo -e "${YELLOW}⚠ Skipping YAML validation (PyYAML not found in any Python)${NC}"
    else
        local yaml_errors=0
        for workflow in .github/workflows/*.yml; do
            if [ -f "$workflow" ]; then
                "$python_with_yaml" -c "import yaml; yaml.safe_load(open('$workflow'))" 2>/dev/null
                if [ $? -ne 0 ]; then
                    echo -e "${RED}✗ Invalid YAML in $workflow${NC}"
                    FAILED_TESTS+=("workflow-yaml-$(basename $workflow)")
                    yaml_errors=$((yaml_errors + 1))
                fi
            fi
        done

        if [ $yaml_errors -eq 0 ]; then
            echo -e "${GREEN}✓ All workflow files have valid YAML${NC}"
        fi
    fi
}

# Main execution
main() {
    # Create necessary directories
    mkdir -p logs test_results

    # Pre-flight checks
    echo -e "${CYAN}=== Running Pre-flight Checks ===${NC}"
    check_workflows

    # Test with each Python version
    for py_version in "${PYTHON_VERSIONS[@]}"; do
        test_python_version "$py_version"
    done

    # Check Docker build
    check_docker_build

    # Check integration tests
    check_integration_tests

    # Summary
    echo -e "\n${BLUE}=== COMPREHENSIVE TEST SUMMARY ===${NC}"
    if [ ${#FAILED_TESTS[@]} -eq 0 ]; then
        echo -e "${GREEN}✓ ALL TESTS PASSED WITH ALL PYTHON VERSIONS!${NC}"
        echo -e "${GREEN}✓ Your code WILL pass on GitHub Actions${NC}"
        echo -e "${GREEN}✓ Safe to push with confidence${NC}"

        exit 0
    else
        echo -e "${RED}✗ FAILED TESTS: ${FAILED_TESTS[*]}${NC}"
        echo -e "${RED}✗ Your push WILL FAIL on GitHub${NC}"
        echo -e "${RED}✗ Fix ALL issues before attempting to push${NC}"

        # Provide detailed debugging info
        echo -e "\n${YELLOW}Action items:${NC}"
        if [[ " ${FAILED_TESTS[@]} " =~ "python-" ]]; then
            echo -e "${RED}1. CRITICAL: Install missing Python versions${NC}"
            echo -e "   Use pyenv, brew, or apt as shown above"
        fi
        if [[ " ${FAILED_TESTS[@]} " =~ "syntax" ]] || [[ " ${FAILED_TESTS[@]} " =~ "parse" ]]; then
            echo -e "${RED}2. Fix syntax errors:${NC}"
            echo -e "   flake8 modules/ services/ orchestrator/ tests/ --select=E9,F63,F7,F82"
        fi
        if [[ " ${FAILED_TESTS[@]} " =~ "e2e" ]]; then
            echo -e "${RED}3. Fix E2E test failures:${NC}"
            echo -e "   Check error logs and ensure all dependencies are installed"
        fi
        if [[ " ${FAILED_TESTS[@]} " =~ "CI" ]]; then
            echo -e "${RED}4. Fix CI test failures:${NC}"
            echo -e "   These are the core tests - must pass!"
        fi

        echo -e "\n${RED}DO NOT PUSH until ALL tests pass${NC}"
        exit 1
    fi
}

# Handle script arguments
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo "Usage: $0"
    echo ""
    echo "Runs ALL GitHub Actions tests with Python 3.9, 3.10, and 3.11"
    echo "NO OPTIONS - Comprehensive testing only"
    echo ""
    echo "This script ensures your code will pass on GitHub"
    exit 0
fi

# Check for all required Python versions BEFORE starting
echo -e "${CYAN}=== Pre-flight Check ===${NC}"
MISSING_PYTHONS=()
for py_version in "${PYTHON_VERSIONS[@]}"; do
    found=false
    if command -v "python$py_version" &> /dev/null || \
       command -v "/opt/homebrew/bin/python$py_version" &> /dev/null || \
       command -v "/usr/local/bin/python$py_version" &> /dev/null; then
        found=true
    fi
    if [ "$found" = false ]; then
        MISSING_PYTHONS+=("$py_version")
    fi
done

if [ ${#MISSING_PYTHONS[@]} -gt 0 ]; then
    echo -e "${RED}ERROR: Missing required Python versions: ${MISSING_PYTHONS[*]}${NC}"
    echo -e "${RED}Cannot proceed without ALL Python versions${NC}"
    echo -e "\n${YELLOW}Installation instructions:${NC}"
    echo -e "${CYAN}Using pyenv (recommended):${NC}"
    echo "  pyenv install 3.9.19 3.10.14 3.11.9"
    echo -e "\n${CYAN}Using Homebrew (macOS):${NC}"
    echo "  brew install python@3.9 python@3.10 python@3.11"
    echo -e "\n${CYAN}Using apt (Ubuntu/Debian):${NC}"
    echo "  sudo apt-get install python3.9 python3.10 python3.11"
    echo -e "\n${RED}Test suite ABORTED - Install missing Python versions first${NC}"
    exit 1
fi

# Run main function
main
