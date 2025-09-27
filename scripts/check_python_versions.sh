#!/bin/bash

# check_python_versions.sh - Verify Python installation status for GitHub Actions testing

# Get the script directory and repo root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$REPO_ROOT"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${BLUE}=== Python Version Check for GitHub Actions ===${NC}"
echo -e "${YELLOW}GitHub Actions requires Python 3.9, 3.10, and 3.11${NC}\n"

REQUIRED_VERSIONS=("3.9" "3.10" "3.11")
MISSING_VERSIONS=()
INSTALLED_VERSIONS=()

# Check each required version
for version in "${REQUIRED_VERSIONS[@]}"; do
    python_cmd=""

    # Check multiple possible locations
    if command -v "python$version" &> /dev/null; then
        python_cmd="python$version"
    elif command -v "/opt/homebrew/bin/python$version" &> /dev/null; then
        python_cmd="/opt/homebrew/bin/python$version"
    elif command -v "/usr/local/bin/python$version" &> /dev/null; then
        python_cmd="/usr/local/bin/python$version"
    fi

    if [ -n "$python_cmd" ]; then
        FULL_VERSION=$("$python_cmd" --version 2>&1 | cut -d' ' -f2)
        echo -e "${GREEN}✓ Python $version installed: $FULL_VERSION (at $python_cmd)${NC}"
        INSTALLED_VERSIONS+=("$version")
    else
        echo -e "${RED}✗ Python $version NOT FOUND${NC}"
        MISSING_VERSIONS+=("$version")
    fi
done

echo ""

# Check if pyenv is available
if command -v pyenv &> /dev/null; then
    echo -e "${CYAN}pyenv detected - Good for managing Python versions${NC}"
    PYENV_AVAILABLE=true
else
    echo -e "${YELLOW}pyenv not detected - Consider installing for easier Python management${NC}"
    PYENV_AVAILABLE=false
fi

# Summary and recommendations
echo -e "\n${BLUE}=== Summary ===${NC}"

if [ ${#MISSING_VERSIONS[@]} -eq 0 ]; then
    echo -e "${GREEN}✓ All required Python versions are installed!${NC}"
    echo -e "${GREEN}✓ You can run: ./scripts/test_github_actions.sh${NC}"
    echo -e "${GREEN}✓ Git push will work with comprehensive testing${NC}"
else
    echo -e "${RED}Missing Python versions: ${MISSING_VERSIONS[*]}${NC}"
    echo -e "${RED}You CANNOT push until ALL versions are installed${NC}"

    echo -e "\n${YELLOW}Installation Instructions:${NC}"

    if [ "$PYENV_AVAILABLE" = true ]; then
        echo -e "\n${CYAN}Using pyenv (recommended):${NC}"
        echo "# Install missing versions"
        for v in "${MISSING_VERSIONS[@]}"; do
            case $v in
                "3.9") echo "pyenv install 3.9.19" ;;
                "3.10") echo "pyenv install 3.10.14" ;;
                "3.11") echo "pyenv install 3.11.9" ;;
            esac
        done
        echo ""
        echo "# Make them available globally"
        echo "pyenv global 3.9.19 3.10.14 3.11.9 $(pyenv version-name)"
    else
        echo -e "\n${CYAN}Install pyenv first (recommended):${NC}"
        echo "# macOS:"
        echo "brew install pyenv"
        echo ""
        echo "# Linux:"
        echo 'curl https://pyenv.run | bash'
        echo ""
        echo "# Then add to your shell profile:"
        echo 'export PATH="$HOME/.pyenv/bin:$PATH"'
        echo 'eval "$(pyenv init -)"'
    fi

    echo -e "\n${CYAN}Alternative: Using system package manager:${NC}"
    echo "# macOS with Homebrew:"
    for v in "${MISSING_VERSIONS[@]}"; do
        echo "brew install python@$v"
    done

    echo ""
    echo "# Ubuntu/Debian:"
    echo "sudo add-apt-repository ppa:deadsnakes/ppa"
    echo "sudo apt-get update"
    for v in "${MISSING_VERSIONS[@]}"; do
        echo "sudo apt-get install python$v python$v-venv python$v-dev"
    done

    echo ""
    echo "# Fedora/RHEL:"
    for v in "${MISSING_VERSIONS[@]}"; do
        echo "sudo dnf install python$v"
    done
fi

# Test command availability
echo -e "\n${BLUE}=== Test Infrastructure ===${NC}"
if [ -f "scripts/test_github_actions.sh" ]; then
    echo -e "${GREEN}✓ scripts/test_github_actions.sh found${NC}"
else
    echo -e "${RED}✗ scripts/test_github_actions.sh NOT FOUND${NC}"
fi

if [ -f ".git/hooks/pre-push" ]; then
    echo -e "${GREEN}✓ Git pre-push hook installed${NC}"
else
    echo -e "${RED}✗ Git pre-push hook NOT INSTALLED${NC}"
fi

# Final message
if [ ${#MISSING_VERSIONS[@]} -gt 0 ]; then
    echo -e "\n${RED}ACTION REQUIRED: Install missing Python versions before pushing${NC}"
    exit 1
else
    echo -e "\n${GREEN}Ready for comprehensive testing!${NC}"
    exit 0
fi
