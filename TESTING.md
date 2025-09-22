# Testing Guide

This project enforces testing that EXACTLY matches GitHub Actions. When you run `git push origin main`, ALL tests that GitHub would run MUST pass locally first. No shortcuts, no exceptions.

## Overview

**EVERY PUSH MUST PASS ALL TESTS WITH PYTHON 3.9, 3.10, AND 3.11**

## Pre-requisites

### Required Python Versions
GitHub Actions tests with Python 3.9, 3.10, and 3.11. You MUST have all three installed:

```bash
# Using pyenv (recommended)
pyenv install 3.9.19
pyenv install 3.10.14
pyenv install 3.11.9

# Or using your system package manager
# macOS with Homebrew:
brew install python@3.9 python@3.10 python@3.11

# Ubuntu/Debian:
sudo apt-get install python3.9 python3.10 python3.11
```

## Automatic Testing on Push

When you run `git push origin main`, the pre-push hook automatically runs:

1. **Workflow validation** - Checks all GitHub workflow files
2. **Syntax checks** - Flake8 with exact GitHub Actions configuration
3. **Full test suite** - With Python 3.9, 3.10, and 3.11 in isolated environments
4. **E2E Healing tests** - All end-to-end healing scenario tests
5. **Security audits** - Security vulnerability scanning
6. **Integration tests** - If Docker is available

**THE PUSH WILL BE BLOCKED IF ANY TEST FAILS**

## Manual Testing Commands

### Run ALL GitHub Actions Tests (REQUIRED before push)
```bash
./test_github_actions.sh
```

This runs:
- Creates fresh virtual environments for Python 3.9, 3.10, and 3.11
- Installs all dependencies in each environment
- Runs the EXACT same tests as GitHub Actions
- Validates all workflow files
- Checks Docker builds if available

### Individual Test Categories

If you need to debug specific test failures:

```bash
# Activate your virtual environment
source venv/bin/activate

# Set required environment variables
export USE_MOCK_TESTS=true
export DISABLE_PERFORMANCE_TRACKING=true
export PYTHONPATH=$PWD

# Run specific test suites
# CI tests (flake8 + basic pytest)
flake8 modules/ services/ orchestrator/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics
python -m pytest -c pytest-ci.ini tests/ -k "not test_concurrent_error_processing_performance"

# E2E Healing tests
python -m pytest tests/e2e/healing_scenarios/test_basic_healing_scenarios.py -v

# Performance tests (if applicable)
python -m pytest tests/performance/ -v

# Chaos tests (if applicable)
python -m pytest tests/chaos/ -v
```

## Test Configuration Files

- `pytest-ci.ini` - Pytest configuration matching GitHub Actions
- `.github/workflows/` - GitHub Actions workflow definitions
- `test_github_actions.sh` - Comprehensive local test runner

## Common Issues and Solutions

### Python Version Not Found
```
ERROR: Python 3.9 not found
```
**Solution**: Install the missing Python version using pyenv or your package manager.

### Parse Errors on GitHub but Not Locally
This means you're testing with a different Python version than GitHub uses.
**Solution**: Run `./test_github_actions.sh` to test with ALL required versions.

### Tests Pass Individually but Fail in CI
Environment variables or configuration differences.
**Solution**: Always test with the full script that sets up the exact CI environment.

### Docker Tests Failing
```
Docker not available, skipping Docker tests
```
**Solution**: Install Docker if you need to test containerized services.

## Workflow Test Matrix

GitHub Actions runs the following test matrix:

| Workflow | Python Versions | Tests |
|----------|-----------------|--------|
| CI | 3.10 | Flake8, Core pytest suite |
| E2E Healing | 3.9, 3.10, 3.11 | Healing scenario tests |
| Security Audit | 3.9, 3.10, 3.11 | Security scans |
| Performance | 3.9 | Performance regression tests |
| Chaos | 3.9, 3.10, 3.11 | Chaos engineering tests |

## Debugging Test Failures

When tests fail:

1. **Check Python version**: Ensure you have 3.9, 3.10, and 3.11
2. **Check dependencies**: Each version needs clean dependency installation
3. **Check syntax**: Run flake8 with the exact same flags as CI
4. **Check imports**: Ensure no Python 3.10+ only features if supporting 3.9

## Summary

**BEFORE EVERY PUSH:**
```bash
./test_github_actions.sh
```

If this passes, your push WILL succeed on GitHub. If it fails, fix the issues before pushing.
