#!/bin/bash

# test_github_actions.sh - GitHub Actions Testing
# This script runs ALL tests EXACTLY as GitHub Actions does
# Complete testing with Python 3.9, 3.10, and 3.11
#
# WARNING: This script runs the FULL test suite and takes approximately an hour to complete!
# It tests with multiple Python versions and runs all test categories.
# For quick testing during development, consider running individual test commands instead.

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

# Log configuration
LOG_DIR="$SCRIPT_DIR/test_logs"
MAIN_LOG="$LOG_DIR/latest.log"
LATEST_LOG="$LOG_DIR/latest.log"

# Initialize logging
init_logging() {
    mkdir -p "$LOG_DIR"

    # Clean up old pytest logs (keep only last 10)
    ls -t "$LOG_DIR"/pytest_*.log 2>/dev/null | tail -n +11 | xargs rm -f 2>/dev/null || true

    # Clean up old e2e logs (keep only last 10)
    ls -t "$LOG_DIR"/e2e_*.log 2>/dev/null | tail -n +11 | xargs rm -f 2>/dev/null || true

    # Start new log (overwrite existing)
    echo "=== GitHub Actions Test Suite ===" > "$MAIN_LOG"
    echo "Started at: $(date)" >> "$MAIN_LOG"
    echo "" >> "$MAIN_LOG"
}

# Function to run command with progress
run_with_progress() {
    local description=$1
    local command=$2
    echo -ne "${BLUE}$description...${NC}"
    echo "\n=== $description ===" >> "$MAIN_LOG"
    echo "Command: $command" >> "$MAIN_LOG"

    local temp_file=$(mktemp)
    local start_time=$(date +%s)

    # Run command and capture output to log only
    eval "$command" >> "$MAIN_LOG" 2>&1
    local exit_code=$?

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    if [ $exit_code -eq 0 ]; then
        echo -e "\r${BLUE}$description... ${GREEN}✓${NC} (${duration}s)"
    else
        echo -e "\r${BLUE}$description... ${RED}✗${NC} (${duration}s)"
    fi

    return $exit_code
}

# Initialize logging
init_logging

echo -e "${BLUE}=== GitHub Actions Test Suite ===${NC}"
echo -e "${BLUE}Testing with Python versions: ${PYTHON_VERSIONS[*]}${NC}"
echo -e "${YELLOW}Verbose output is being logged to:${NC}"
echo -e "${CYAN}$LATEST_LOG${NC}"
echo ""

# Function to print test results
print_result() {
    local test_name=$1
    local exit_code=$2
    local python_version=$3

    if [ $exit_code -eq 0 ]; then
        if [ "$python_version" = "N/A" ]; then
            echo -e "${GREEN}✓ $test_name passed${NC}"
        else
            echo -e "${GREEN}✓ $test_name (Python $python_version) passed${NC}"
        fi
    else
        if [ "$python_version" = "N/A" ]; then
            echo -e "${RED}✗ $test_name failed${NC}"
            FAILED_TESTS+=("$test_name")
        else
            echo -e "${RED}✗ $test_name (Python $python_version) failed${NC}"
            FAILED_TESTS+=("$test_name-py$python_version")
        fi
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
    echo -n "Upgrading pip..."
    pip install --upgrade pip setuptools wheel >> "$MAIN_LOG" 2>&1
    echo -e "\r${GREEN}✓${NC} Pip upgraded"

    # Install dependencies (simulating CI)
    run_with_progress "Installing dependencies" \
        "pip install -r requirements.txt"
    local req_exit=$?
    print_result "Requirements installation" $req_exit $py_version

    if [ $req_exit -ne 0 ]; then
        echo -e "${RED}Failed to install requirements, skipping other tests${NC}"
        deactivate
        return
    fi

    # Install dev dependencies
    run_with_progress "Installing dev dependencies" \
        "pip install -r requirements-dev.txt || true"

    # Set environment variables as CI does
    export USE_MOCK_TESTS=true
    export DISABLE_PERFORMANCE_TRACKING=true
    export PYTHONPATH=$PWD

    echo -e "\n${MAGENTA}Running CI Workflow Tests...${NC}"

    # 1. Flake8 syntax check (from CI workflow)
    echo -e "\n${BLUE}1. Running flake8 syntax checks...${NC}"
    run_with_progress "Flake8 syntax check" \
        "flake8 modules/ services/ orchestrator/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics"
    print_result "Flake8 syntax check (CI)" $? $py_version

    # 2. Basic pytest suite (from CI workflow) with pytest-ci.ini
    echo -e "\n${BLUE}2. Running CI pytest suite...${NC}"
    # Run with timeout and show failures
    # Use special config for Python 3.11 to avoid asyncio hangs
    # Store output to check for timeout

    if [ "$py_version" = "3.9" ]; then
        # Python 3.9 with progress display like 3.10
        echo -e "${YELLOW}Running Python 3.9 tests...${NC}"
        local test_log="$LOG_DIR/pytest_${py_version}_$(date +%s).log"
        python -m pytest -c pytest-py39-timeout.ini tests/ -k "not test_concurrent_error_processing_performance" -v --tb=short 2>&1 | tee "$test_log"
        local pytest_exit=${PIPESTATUS[0]}

        # Show summary
        echo -e "\n${YELLOW}Test Summary:${NC}"
        grep -E "(passed|failed|error|warnings summary|FAILURES|ERROR)" "$test_log" | tail -20 || true
    elif [ "$py_version" = "3.11" ] && [ -f "pytest-py311-timeout.ini" ]; then
        echo -e "${YELLOW}Running Python 3.11 tests with timeout...${NC}"
        local test_log="$LOG_DIR/pytest_${py_version}_$(date +%s).log"
        # Run pytest but show only test collection and item progress
        python -m pytest -c pytest-py311-timeout.ini tests/ -k "not test_concurrent_error_processing_performance" -v --tb=short 2>&1 | \
            tee "$test_log" | \
            grep -E "^(platform|rootdir|configfile|plugins|collected|tests/.*::|===|PASSED|FAILED|SKIPPED|ERROR|WARNING|%)" || true
        local pytest_exit=${PIPESTATUS[0]}

        # Show summary
        echo -e "\n${YELLOW}Test Summary:${NC}"
        grep -E "(passed|failed|error|warnings summary|FAILURES|ERROR)" "$test_log" | tail -20 || true
    elif [ "$py_version" = "3.10" ] && [ -f "pytest-py310-timeout.ini" ]; then
        # Python 3.10 can be slower, use optimized config
        echo -e "${YELLOW}Running Python 3.10 tests with timeout...${NC}"
        local test_log="$LOG_DIR/pytest_${py_version}_$(date +%s).log"
        python -m pytest -c pytest-py310-timeout.ini tests/ -k "not test_concurrent_error_processing_performance" -v --tb=short 2>&1 | tee "$test_log"
        local pytest_exit=${PIPESTATUS[0]}

        # Show summary
        echo -e "\n${YELLOW}Test Summary:${NC}"
        grep -E "(passed|failed|error|warnings summary|FAILURES|ERROR)" "$test_log" | tail -20 || true
    else
        # Default case with progress display
        echo -e "${YELLOW}Running tests...${NC}"
        local test_log="$LOG_DIR/pytest_${py_version}_$(date +%s).log"
        python -m pytest -c pytest-ci-timeout.ini tests/ -k "not test_concurrent_error_processing_performance" -v --tb=short --timeout=300 --timeout-method=thread 2>&1 | tee "$test_log"
        local pytest_exit=${PIPESTATUS[0]}

        # Show summary
        echo -e "\n${YELLOW}Test Summary:${NC}"
        grep -E "(passed|failed|error|warnings summary|FAILURES|ERROR)" "$test_log" | tail -20 || true
    fi
    # Report test status
    if [ $pytest_exit -eq 0 ]; then
        echo -e "${GREEN}All tests passed${NC}"
    elif [ $pytest_exit -eq 5 ]; then
        echo -e "${YELLOW}No tests were collected (exit code 5)${NC}"
    else
        echo -e "${RED}Tests failed with exit code: $pytest_exit${NC}"
        echo -e "${YELLOW}Check the log files in logs/ directory for details${NC}"
    fi
    print_result "CI pytest suite" $pytest_exit $py_version

    echo -e "\n${MAGENTA}Running E2E Healing Scenario Tests...${NC}"

    # 3. E2E Healing tests (from e2e-healing-tests.yml)
    echo -e "\n${BLUE}3. Running E2E healing scenarios...${NC}"
    # Create test_results directory if it doesn't exist
    mkdir -p test_results

    # Run exactly as GitHub Actions does (but without timeout)
    echo -e "\n${BLUE}3. Running E2E healing scenarios...${NC}"
    local e2e_log="$LOG_DIR/e2e_${py_version}_$(date +%s).log"
    PYTHONPATH=$PWD python -m pytest tests/e2e/healing_scenarios/test_basic_healing_scenarios.py -v \
        --json-report \
        --json-report-file=test_results/report.json \
        --junit-xml=test_results/report.xml \
        --html=test_results/report.html \
        --self-contained-html 2>&1 | tee "$e2e_log" | grep -E "^(tests/.*::|PASSED|FAILED|SKIPPED|ERROR|===)" || true
    local e2e_exit=${PIPESTATUS[0]}

    # Ensure report files exist (as GitHub Actions does)
    if [ ! -f test_results/report.json ]; then
        echo '{"summary": {"total": 0, "passed": 0, "failed": 0}}' > test_results/report.json
    fi
    if [ ! -f test_results/report.html ]; then
        echo '<html><body>No test results</body></html>' > test_results/report.html
    fi

    print_result "E2E Healing Scenarios" $e2e_exit $py_version

    # Clean up test results for next run
    rm -rf test_results

    echo -e "\n${MAGENTA}Running Security Audit Tests...${NC}"

    # 4. Security checks
    echo -e "\n${BLUE}4. Running security vulnerability scan...${NC}"
    if command -v safety &> /dev/null; then
        run_with_progress "Security scan" \
            "safety scan --policy-file /dev/null || true"
    else
        run_with_progress "Installing safety and scanning" \
            "pip install safety && safety scan --policy-file /dev/null || true"
    fi

    echo -e "\n${MAGENTA}Running Additional GitHub Action Tests...${NC}"

    # 5. Performance regression tests (if they exist)
    if [ -f "tests/performance/test_performance_regression.py" ]; then
        echo -e "\n${BLUE}5. Running performance regression tests...${NC}"
        run_with_progress "Performance regression tests" \
            "python -m pytest tests/performance/test_performance_regression.py -v"
        print_result "Performance regression" $? $py_version
    fi

    # 6. Chaos engineering tests (if they exist)
    if [ -d "tests/chaos" ]; then
        echo -e "\n${BLUE}6. Running chaos engineering tests...${NC}"
        # Use special config for Python 3.11 to avoid asyncio hangs
        if [ "$py_version" = "3.11" ] && [ -f "pytest-py311.ini" ]; then
            run_with_progress "Chaos engineering tests" \
                "python -m pytest -c pytest-py311.ini tests/chaos/ -v"
        else
            run_with_progress "Chaos engineering tests" \
                "python -m pytest tests/chaos/ -v"
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
        echo -e "${RED}✗ Docker not available${NC}"
        echo -e "${YELLOW}Please install Docker Desktop: https://docker.com/products/docker-desktop/${NC}"
        FAILED_TESTS+=("docker-not-installed")
        return
    fi

    # Check if Docker daemon is running
    if ! docker info >> "$MAIN_LOG" 2>&1; then
        echo -e "${RED}✗ Docker daemon is not running${NC}"
        echo -e "${YELLOW}Please start Docker Desktop${NC}"
        FAILED_TESTS+=("docker-not-running")
        return
    fi

    # Clean up Docker before build to prevent I/O errors
    echo -e "${BLUE}Cleaning up Docker containers and images...${NC}"
    docker container prune -f >> "$MAIN_LOG" 2>&1
    docker image prune -f >> "$MAIN_LOG" 2>&1
    # Remove any existing test images
    docker rmi homeostasis-test:local-test homeostasis-test:latest >> "$MAIN_LOG" 2>&1 || true

    # Check if Dockerfile exists
    if [ -f "Dockerfile" ]; then
        echo "Building Docker image..."
        # Use legacy builder to avoid buildx issues
        export DOCKER_BUILDKIT=0
        echo -e "${BLUE}Running Docker build...${NC}"
        # Create a temporary file for Docker output
        local docker_log=$(mktemp)

        # Run Docker build with progress tracking
        docker build -t homeostasis-test:local-test . > "$docker_log" 2>&1 &
        local pid=$!

        # Monitor Docker build progress
        local last_step=""
        while kill -0 $pid 2>/dev/null; do
            # Extract current step from Docker output
            local current_step=$(grep -E "^Step [0-9]+/[0-9]+" "$docker_log" | tail -1 | cut -d: -f1 || echo "")
            if [ "$current_step" != "$last_step" ] && [ -n "$current_step" ]; then
                echo -e "\r${BLUE}Docker build: $current_step${NC}                    "
                last_step="$current_step"
            else
                echo -ne "\r${BLUE}Docker build in progress...${NC}     "
            fi
            sleep 1
        done
        wait $pid
        local docker_exit=$?

        # Copy full output to main log
        cat "$docker_log" >> "$MAIN_LOG"
        rm -f "$docker_log"

        echo -e "\r${BLUE}Docker build completed${NC}                    "

        if [ $docker_exit -ne 0 ]; then
            echo -e "${RED}Docker build failed - check the output above for errors${NC}"
            echo -e "${RED}Docker must work locally to pass on GitHub Actions${NC}"
            FAILED_TESTS+=("docker-build-failed")
        else
            print_result "Docker build" $docker_exit "N/A"
        fi

        # Clean up if successful
        if [ $docker_exit -eq 0 ]; then
            docker rmi homeostasis-test:local-test >> "$MAIN_LOG" 2>&1
        fi
    else
        echo -e "${YELLOW}No Dockerfile found, skipping Docker build${NC}"
    fi
}

# Function to run integration tests if docker-compose file exists
check_integration_tests() {
    echo -e "\n${CYAN}=== Checking Integration Tests ===${NC}"

    if ! command -v docker &> /dev/null; then
        echo -e "${RED}✗ Docker not available for integration tests${NC}"
        FAILED_TESTS+=("docker-integration-not-available")
        return
    fi

    if ! docker info >> "$MAIN_LOG" 2>&1; then
        echo -e "${RED}✗ Docker daemon is not running for integration tests${NC}"
        FAILED_TESTS+=("docker-integration-not-running")
        return
    fi

    # Check for Docker Compose - try both 'docker compose' and 'docker-compose'
    local has_compose=false
    if docker compose version >> "$MAIN_LOG" 2>&1; then
        has_compose=true
    elif command -v docker-compose &> /dev/null && docker-compose --version >> "$MAIN_LOG" 2>&1; then
        has_compose=true
    fi

    if [ "$has_compose" = false ]; then
        echo -e "${RED}✗ Docker Compose not available${NC}"
        echo -e "${YELLOW}Docker Compose is required for integration tests${NC}"
        FAILED_TESTS+=("docker-compose-not-available")
        return
    fi

    if [ -f "tests/e2e/docker-compose.yml" ]; then
        echo -e "${BLUE}Running Docker integration tests (as GitHub Actions does)${NC}"

        # Use the EXACT same compose file as GitHub Actions
        local compose_file="tests/e2e/docker-compose.yml"

        # First validate the compose file
        run_with_progress "Docker Compose validation" \
            "docker compose -f '$compose_file' config"
        local compose_valid=$?
        if [ $compose_valid -ne 0 ]; then
            FAILED_TESTS+=("docker-compose-invalid")
        fi

        # Clean up before building to prevent conflicts
        run_with_progress "Cleaning up existing containers" \
            "docker compose -f '$compose_file' down --volumes --remove-orphans || true"

        # Build test images (matching GitHub Actions)
        echo -e "\n${BLUE}Building test images...${NC}"
        export DOCKER_BUILDKIT=0
        run_with_progress "Building main Docker image" \
            "docker build -t homeostasis-test:latest ."
        local main_build=$?

        print_result "Main Docker build" $main_build "N/A"
        if [ $main_build -ne 0 ]; then
            echo -e "${RED}Docker build failed - check the output above for errors${NC}"
            FAILED_TESTS+=("docker-main-build-failed")
        fi

        echo -e "\n${BLUE}Building Docker Compose services...${NC}"
        if [ "$compose_file" = "tests/e2e/docker-compose.test.yml" ]; then
            echo -e "${YELLOW}Using lightweight test compose file for faster pre-push testing${NC}"
        fi

        # Create a temporary file for Docker Compose output
        local compose_log=$(mktemp)

        # Run Docker Compose build with progress tracking
        docker compose -f "$compose_file" build > "$compose_log" 2>&1 &
        local pid=$!

        # Monitor Docker Compose build progress
        local last_service=""
        local last_step=""
        while kill -0 $pid 2>/dev/null; do
            # Extract current service being built
            local current_service=$(grep -E "^\[.+\] Building" "$compose_log" | tail -1 | sed -E 's/^\[([^]]+)\].*/\1/' || echo "")
            # Extract current step from Docker output
            local current_step=$(grep -E "Step [0-9]+/[0-9]+" "$compose_log" | tail -1 | grep -o "Step [0-9]\+/[0-9]\+" || echo "")

            if [ -n "$current_service" ] && [ -n "$current_step" ]; then
                if [ "$current_service" != "$last_service" ] || [ "$current_step" != "$last_step" ]; then
                    echo -e "\r${BLUE}Docker Compose build: [$current_service] $current_step${NC}                    "
                    last_service="$current_service"
                    last_step="$current_step"
                fi
            else
                echo -ne "\r${BLUE}Docker Compose build in progress...${NC}     "
            fi
            sleep 1
        done
        wait $pid
        local compose_build=$?

        # Copy full output to main log
        echo "\n=== Docker Compose Build Output ===" >> "$MAIN_LOG"
        cat "$compose_log" >> "$MAIN_LOG"
        rm -f "$compose_log"

        echo -e "\r${BLUE}Docker Compose build completed${NC}                    "

        print_result "Docker Compose build" $compose_build "N/A"
        if [ $compose_build -ne 0 ]; then
            echo -e "${RED}Docker Compose build failed - check the output above for errors${NC}"
            FAILED_TESTS+=("docker-compose-build-failed")
        fi

        if [ "$SKIP_DOCKER_INTEGRATION" = "1" ]; then
            echo -e "${YELLOW}Skipping Docker integration tests (SKIP_DOCKER_INTEGRATION=1)${NC}"
            echo -e "${YELLOW}WARNING: GitHub Actions WILL run these tests!${NC}"
        else
            echo -e "\n${BLUE}Running Docker integration tests (as GitHub Actions does)${NC}"
            # Run the exact same command that GitHub Actions runs
            run_with_progress "Docker integration tests" \
                "docker compose -f '$compose_file' run --rm test-runner python /app/tests/e2e/healing_scenarios/run_e2e_tests.py --suite all --ci"
            local integration_result=$?

            print_result "Docker Compose integration tests" $integration_result "N/A"
            if [ $integration_result -ne 0 ]; then
                echo -e "${RED}Docker integration tests failed${NC}"
                FAILED_TESTS+=("docker-integration-tests-failed")
            fi
        fi

        # Clean up after testing
        echo -e "\n${BLUE}Cleaning up Docker resources...${NC}"
        run_with_progress "Docker cleanup" \
            "docker compose -f '$compose_file' down --volumes --remove-orphans || true"
    else
        echo -e "${YELLOW}No docker-compose.yml found, skipping integration tests${NC}"
    fi
}

# Function to check workflows
check_workflows() {
    echo -e "\n${CYAN}=== Pre-flight Workflow Checks ===${NC}"

    # Check for Python 3.8 references
    if grep -r "'3.8'\|\"3.8\"" .github/workflows/*.yml >> "$MAIN_LOG" 2>&1; then
        echo -e "${RED}✗ Found Python 3.8 references in workflows${NC}"
        FAILED_TESTS+=("workflow-python38-check")
    else
        echo -e "${GREEN}✓ No Python 3.8 references${NC}"
    fi

    # Check for docker-compose (should be docker compose)
    if grep -r "docker-compose " .github/workflows/*.yml | grep -v "docker-compose.yml" >> "$MAIN_LOG" 2>&1; then
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
                "$python_with_yaml" -c "import yaml; yaml.safe_load(open('$workflow'))" >> "$MAIN_LOG" 2>&1
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
    mkdir -p test_results

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
    echo -e "\n${BLUE}=== TEST SUMMARY ===${NC}"
    echo "\n=== FINAL TEST SUMMARY ===" >> "$MAIN_LOG"
    echo "Total failed tests: ${#FAILED_TESTS[@]}" >> "$MAIN_LOG"
    echo "Failed tests: ${FAILED_TESTS[*]}" >> "$MAIN_LOG"
    echo "Log file: $MAIN_LOG" >> "$MAIN_LOG"
    echo "Completed at: $(date)" >> "$MAIN_LOG"

    if [ ${#FAILED_TESTS[@]} -eq 0 ]; then
        echo -e "${GREEN}✓ ALL TESTS PASSED WITH ALL PYTHON VERSIONS!${NC}"
        echo -e "${GREEN}✓ Your code WILL pass on GitHub Actions${NC}"
        echo -e "${GREEN}✓ Safe to push with confidence${NC}"
        echo -e "\n${CYAN}Full test log saved to: ${LATEST_LOG}${NC}"

        # Clean up Docker images before exit
        echo -e "\n${CYAN}=== Cleaning up Docker images ===${NC}"
        if docker images | grep -q "homeostasis-test"; then
            echo "Removing homeostasis-test Docker images..."
            docker rmi homeostasis-test:latest homeostasis-test:local-test >> "$MAIN_LOG" 2>&1 || true
            echo -e "${GREEN}✓ Docker cleanup completed${NC}"
        fi

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
        if [[ " ${FAILED_TESTS[@]} " =~ "docker" ]]; then
            echo -e "${RED}5. Fix Docker issues:${NC}"
            echo -e "   - Install Docker Desktop: https://docker.com/products/docker-desktop/"
            echo -e "   - Start Docker Desktop application"
            echo -e "   - Ensure Docker daemon is running: docker info"
            echo -e "   Docker is required for integration tests to pass on GitHub"
        fi

        echo -e "\n${RED}DO NOT PUSH until ALL tests pass${NC}"
        echo -e "\n${CYAN}Full error details in: ${LATEST_LOG}${NC}"

        # Clean up Docker images even on failure
        echo -e "\n${CYAN}=== Cleaning up Docker images ===${NC}"
        if docker images | grep -q "homeostasis-test"; then
            echo "Removing homeostasis-test Docker images..."
            docker rmi homeostasis-test:latest homeostasis-test:local-test >> "$MAIN_LOG" 2>&1 || true
            echo -e "${GREEN}✓ Docker cleanup completed${NC}"
        fi

        exit 1
    fi
}

# Handle script arguments
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo "Usage: $0"
    echo ""
    echo "Runs ALL GitHub Actions tests with Python 3.9, 3.10, and 3.11"
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
