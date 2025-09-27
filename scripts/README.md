# Scripts Directory

This directory contains utility scripts for the Homeostasis project. All scripts should be run from the project root directory.

## Scripts Overview

### check_python_versions.sh
**Purpose:** Verifies Python installation status for GitHub Actions testing
**Usage:** `./scripts/check_python_versions.sh`

Checks if Python 3.9, 3.10, and 3.11 are installed on the system. These versions are required for comprehensive GitHub Actions testing. The script:
- Checks multiple locations for Python installations
- Detects if pyenv is available for Python management
- Provides installation instructions for missing versions
- Validates test infrastructure availability

### demo.sh
**Purpose:** Runs the orchestrator in demo mode
**Usage:** `./scripts/demo.sh`

Sets up and runs the Homeostasis orchestrator in demonstration mode:
- Creates necessary directories (logs, sessions)
- Installs required dependencies (pyyaml, requests)
- Launches the orchestrator with demo configuration

### quick_push_check.sh
**Purpose:** Performs essential pre-push tests
**Usage:** `./scripts/quick_push_check.sh`

Runs a quick set of essential tests before pushing code:
- Python syntax checking with flake8
- YAML workflow validation
- Quick smoke test with Python 3.9
- Much faster than full test suite but less comprehensive

### setup_docker.sh
**Purpose:** Helps set up Docker for testing
**Usage:** `./scripts/setup_docker.sh`

Ensures Docker is properly installed and configured for running tests:
- Checks Docker installation and daemon status
- Supports both Docker Desktop and Colima
- Provides installation instructions if Docker is missing
- Configures Docker context for Colima users

### test_docker_only.sh
**Purpose:** Tests Docker build functionality
**Usage:** `./scripts/test_docker_only.sh`

Quickly tests if Docker builds are working:
- Cleans up existing Docker containers/images
- Attempts to build the project's Dockerfile
- 30-second timeout to catch hanging builds
- Useful for debugging Docker-specific issues

### test_github_actions.sh
**Purpose:** Comprehensive GitHub Actions test suite
**Usage:** `./scripts/test_github_actions.sh`

**WARNING:** This script takes approximately an hour to complete!

Runs the FULL test suite exactly as GitHub Actions does:
- Tests with Python 3.9, 3.10, and 3.11
- Runs all test categories (unit, E2E, security, performance, chaos)
- Validates Docker builds and integration tests
- Checks workflow YAML files
- Provides detailed logging and failure reporting

Use this before pushing to ensure your code will pass on GitHub Actions.

## Important Notes

1. **Working Directory:** All scripts should be executed from the project root directory, not from within the scripts folder.

2. **Python Requirements:** Most testing scripts require Python 3.9, 3.10, and 3.11 to be installed for full compatibility with GitHub Actions.

3. **Docker Requirements:** Docker must be installed and running for integration tests and Docker build validation.

4. **Time Considerations:**
   - `quick_push_check.sh` - Fast (~2 minutes)
   - `test_github_actions.sh` - Slow (~1 hour)

5. **Logging:** Test scripts create logs in the `test_logs/` directory for debugging failures.
