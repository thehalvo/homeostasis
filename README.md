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

For detailed architecture diagrams, see [Homeostasis Architecture](docs/assets/architecture-diagram.txt)

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

## Supported Languages

Homeostasis currently provides support for the following programming languages:

- **Python**: Primary language with comprehensive support for most Python frameworks
- **JavaScript/TypeScript**: Support for Node.js environments and common JavaScript errors 
- **Java**: Support for Java exceptions, Spring Framework, Hibernate/JPA, and concurrency issues

For more information about Java integration, see [Java Integration](docs/java_integration.md).

## Project Status

Homeostasis is actively being developed, here are some recent updates:

- **Rules Engine**: Implemented 80+ detection rules for Python errors including framework-specific rules for Django, FastAPI, and SQLAlchemy
- **Template System**: Developed a hierarchical template system with inheritance and specialization for precise patching
- **Testing Environment**: Created Docker container management, parallel test execution, and regression test generation
- **Monitoring**: Implemented post-deployment monitoring hooks and feedback loops for fix quality improvement
- **Analysis**: Added AST-based code analysis for context-aware patching and function signature analysis
- **Advanced Analysis**: Completed ML-based error classification, causal chain analysis for cascading errors, environmental factor correlation, and APM tool integration
- **Machine Learning**: Added error classification models, training data collection system, and confidence-based hybrid (rules + ML) analysis
- **Framework Support**: Expanded framework support with Django middleware, Flask blueprint-specific error handling, FastAPI dependency analysis, and ASGI framework support
- **Python Ecosystem**: Added support for Python 3.11+ features, Celery tasks, asyncio-specific error detection, NumPy/Pandas error handling, and AI/ML library error detection
- **Multi-Language Support**: Implemented language-agnostic error schema with support for JavaScript/Node.js and Java through a pluggable adapter architecture
- **Java Integration**: Added support for Java exceptions, Spring Framework, Hibernate/JPA, and Java concurrency issues, and Maven/Gradle build analysis
- **Production Readiness**: Enhanced security model with RBAC, implemented approval workflows for critical changes, added rate limiting/throttling for healing actions, and developed comprehensive audit logging
- **Infrastructure Integration**: Created Kubernetes operator for container healing, implemented cloud-specific adapters (AWS, GCP, Azure), developed service mesh integration, added serverless function support, and built edge deployment capabilities
- **User Experience**: Released web dashboard for monitoring healing activities, implemented fix suggestion interface for human review, added configuration management UI, performance reporting, and custom rule/template editors
- **Deployment Options**: Added canary deployment support for gradual rollout of fixes with automatic promotion/rollback based on metrics

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
