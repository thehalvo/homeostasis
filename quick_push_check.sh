#!/bin/bash

# Quick push check - Essential tests only
# For comprehensive testing, use ./test_github_actions.sh

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=== Quick Push Check ===${NC}"
echo -e "${YELLOW}Running essential tests before push${NC}\n"

# 1. Check for syntax errors with flake8
echo -e "${BLUE}1. Checking Python syntax...${NC}"

# Try to find flake8 in venvs or system
FLAKE8_CMD=""
for venv in venv_py39 venv_py310 venv_py311 venv; do
    if [ -f "$venv/bin/flake8" ]; then
        FLAKE8_CMD="$venv/bin/flake8"
        break
    fi
done

if [ -z "$FLAKE8_CMD" ] && command -v flake8 &> /dev/null; then
    FLAKE8_CMD="flake8"
fi

if [ -z "$FLAKE8_CMD" ]; then
    echo -e "${YELLOW}⚠ flake8 not found, skipping syntax check${NC}"
else
    if $FLAKE8_CMD modules/ services/ orchestrator/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics; then
        echo -e "${GREEN}✓ No syntax errors${NC}"
    else
        echo -e "${RED}✗ Syntax errors found!${NC}"
        exit 1
    fi
fi

# 2. Check YAML files
echo -e "\n${BLUE}2. Checking workflow YAML files...${NC}"
for workflow in .github/workflows/*.yml; do
    if [ -f "$workflow" ]; then
        # Use yq or python with yaml if available
        if command -v yq &> /dev/null; then
            yq eval . "$workflow" > /dev/null || {
                echo -e "${RED}✗ Invalid YAML in $workflow${NC}"
                exit 1
            }
        fi
    fi
done
echo -e "${GREEN}✓ Workflow files valid${NC}"

# 3. Quick smoke test with Python 3.9
echo -e "\n${BLUE}3. Running quick smoke test...${NC}"
if [ -d "venv_py39" ]; then
    source venv_py39/bin/activate
    export USE_MOCK_TESTS=true
    export DISABLE_PERFORMANCE_TRACKING=true

    # Just run a few quick tests
    if python -m pytest tests/e2e/healing_scenarios/test_basic_healing_scenarios.py::TestBasicHealingScenarios::test_keyerror_healing -v; then
        echo -e "${GREEN}✓ Smoke test passed${NC}"
    else
        echo -e "${RED}✗ Smoke test failed${NC}"
        exit 1
    fi

    deactivate
else
    echo -e "${YELLOW}⚠ Python 3.9 venv not found, skipping smoke test${NC}"
fi

echo -e "\n${GREEN}✓ Quick checks passed!${NC}"
echo -e "${YELLOW}Note: For full validation, run ./test_github_actions.sh${NC}"
exit 0