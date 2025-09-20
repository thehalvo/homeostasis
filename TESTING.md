# Testing Guide

## Overview

This project uses a simplified testing approach with two main commands:

### Quick Test (Development)
```bash
USE_MOCK_TESTS=true make test
```
- Runs basic pytest suite
- Uses mock infrastructure for speed
- Good for development iterations

### Full Test Suite (Pre-Push)
```bash
USE_MOCK_TESTS=true make test-all
```
- Runs `test_github_actions_locally.sh`
- Tests with Python 3.9, 3.10, and 3.11 in isolated environments
- Runs the same checks as GitHub Actions
- Recommended before pushing to GitHub

## Automatic Pre-Push Validation

A git hook automatically validates code before pushing:
1. Python syntax checks
2. Workflow compatibility (no Python 3.8, correct docker compose syntax)
3. Flake8 linting
4. Black formatting
5. Quick unit tests
6. Optional full test suite

To bypass (not recommended):
```bash
git push --no-verify
```

## Key Points

- **Always use `USE_MOCK_TESTS=true`** to avoid slow infrastructure setup
- **Run `make test-all` before pushing** to ensure GitHub Actions will pass
- **The pre-push hook helps catch issues early** but can be skipped if needed
- **All test commands use the same environment variables** for consistency

## Test Script Details

### test_github_actions_locally.sh
This is the only test script you need to know about. It:
- Creates virtual environments for each Python version
- Installs dependencies fresh (like CI does)
- Runs linting and tests
- Checks workflow files for common issues
- Optionally builds Docker image if available

The script is called by `make test-all` and the pre-push hook.