# Homeostasis: An Open-Source Framework for Self-Healing Systems

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

```
╔═════════════════════════════════════════════════════════════╗
║                                                             ║
║  ☆ ★    $ git push origin homeostasis-fix             ✓     ║
║  ↑                                                          ║
║  ✗     WE DON'T WANT YOUR STARS                             ║
║                                                 ┌────────┐  ║
║  →     WE WANT YOUR PRs                         │APPROVED│  ║
║                                                 └────────┘  ║
╚═════════════════════════════════════════════════════════════╝
```

## Vision

Homeostasis is an open-source framework that mimics biological self-repair mechanisms in software systems. Just as living organisms heal injuries autonomously, Homeostasis enables applications to detect failures, generate fixes, and deploy them without human intervention.

## Goals

- **Reduce Downtime**: Automatically fix errors before they impact users
- **Decrease Manual Repairs**: Free developers from repetitive bug-fixing
- **Universal Framework**: Support multiple languages and platforms
- **Safety First**: Ensure all generated fixes are thoroughly tested
- **Open and Extensible**: Foster community-driven healing strategies

## Architecture Overview

Homeostasis implements a self-healing cycle through six interconnected modules:

1. **Monitoring & Error Collection**: Captures logs, exceptions, and performance metrics
2. **Root Cause Analysis (RCA)**: Identifies underlying error causes using rules or ML
3. **Code Generation & Patching**: Creates fixes through templates or AI assistance
4. **Parallel Environment Deployment**: Tests new code in isolated environments
5. **Validation & Observation**: Verifies fixes through comprehensive test suites
6. **Hot Swap / Replacement**: Safely replaces broken components with fixed versions

![Homeostasis Architecture](docs/assets/architecture-placeholder.png)

## Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/thehalvo/homeostasis.git
   cd homeostasis
   ```

2. Install the package and dependencies:
   ```bash
   pip install -e .
   # For development dependencies
   pip install -e ".[dev]"
   ```

### Running the Demo

Homeostasis includes a demo service with intentional bugs that the framework can automatically fix:

```bash
# Make the demo script executable
chmod +x demo.sh

# Run the demo script to see self-healing in action
./demo.sh
```

If you encounter any issues, you can run the commands manually:

```bash
# Create necessary directories
mkdir -p logs logs/patches logs/backups sessions

# Install additional required dependencies that might be missing
pip install pyyaml requests

# Run the orchestrator in demo mode
python3 orchestrator/orchestrator.py --demo
```

The demo will:
1. Start the example FastAPI service with known bugs
2. Monitor for errors
3. Analyze error root causes
4. Generate and apply patches
5. Test the fixes
6. Restart the service with applied patches

#### Troubleshooting Demo Issues

- If you get permission errors, make sure the demo script is executable with `chmod +x demo.sh`
- If you get "command not found" errors, try running with `python3` explicitly
- Ensure all dependencies are installed with `pip install -e ".[dev]"`
- If you're using a Mac with Apple Silicon (M1/M2/M3), you might need to install additional dependencies: `pip install pytest-asyncio httpx`

### Using Homeostasis in Your Project

1. Configure your project in `orchestrator/config.yaml`:
   ```yaml
   service:
     path: "path/to/your/service"
     start_command: "your service start command"
     health_check_url: "http://localhost:your_port/health"
   ```

2. Run the orchestrator:
   ```bash
   python orchestrator/orchestrator.py
   ```

3. The orchestrator will:
   - Monitor your service logs for errors
   - Analyze detected errors using rule-based analysis
   - Generate and apply patches based on identified issues
   - Test and validate the fixes
   - Restart your service with the fixes applied

### Command Line Options

```bash
# Run with custom config file
python orchestrator/orchestrator.py --config custom_config.yaml

# Run in demo mode
python orchestrator/orchestrator.py --demo

# Roll back the latest applied patches
python orchestrator/orchestrator.py --rollback

# Set custom log level
python orchestrator/orchestrator.py --log-level DEBUG
```

### Known Issues

When running the demo for the first time, you might encounter these issues:

1. **Missing YAML and Requests dependencies**
   - Solution: The demo script will attempt to install them automatically
   - If that fails, install them manually: `pip install pyyaml requests`

2. **Missing directories**
   - Solution: The demo script creates logs/, logs/patches/, logs/backups/, and sessions/ directories
   - If that fails, create them manually: `mkdir -p logs logs/patches logs/backups sessions`

## Project Status

Homeostasis is in its early development phase. We're currently:

- Setting up project infrastructure
- Designing the architecture
- Building a minimal proof-of-concept

## Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) to get started.

## Troubleshooting

### Common Issues

1. **Demo script fails to run:**
   - Make sure the script is executable: `chmod +x demo.sh`
   - Try running with explicit Python path: `python3 orchestrator/orchestrator.py --demo`
   - Ensure all required directories exist: `mkdir -p logs logs/patches logs/backups sessions`

2. **Import errors:**
   - Ensure you're in the activated virtual environment (`source venv/bin/activate`)
   - Verify all dependencies are installed: `pip install -e ".[dev]"`
   - Install missing dependencies manually: `pip install pyyaml requests`
   - Try installing specific dependencies explicitly: `pip install fastapi uvicorn pydantic loguru pytest pyyaml requests`

3. **Service startup issues:**
   - Check if port 8000 is already in use: `lsof -i :8000`
   - Kill any process using the port: `kill -9 $(lsof -ti:8000)`
   - Verify the example service is configured correctly in `orchestrator/config.yaml`

4. **Failed to generate patches:**
   - Ensure all template directories exist: `ls modules/patch_generation/templates/`
   - Check if rule matching is working: `ls modules/analysis/rules/`

5. **Platform-specific issues:**
   - Mac with Apple Silicon (M1/M2/M3): You might need additional dependencies
   - Windows: Adjust path separators and commands in configuration files

For more detailed help, please check the documentation in the `docs/` directory or open an issue on GitHub.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
