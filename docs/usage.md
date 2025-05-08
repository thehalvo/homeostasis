# Homeostasis Usage Guide

*This document explains how to install, configure, and use the Homeostasis self-healing framework.*

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Operating system: Linux, macOS, or Windows (with WSL recommended for best experience)
- Git (to clone the repository)

## Installation

### Standard Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/thehalvo/homeostasis.git
   cd homeostasis
   ```

2. Install the package and dependencies:
   ```bash
   pip install -e .
   ```

3. For development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

### Common Installation Issues

1. **Missing dependencies**:
   ```bash
   pip install pyyaml requests
   ```

2. **Apple Silicon (M1/M2/M3) specific**:
   ```bash
   pip install pytest-asyncio httpx
   ```

## Quick Start

### Running the Demo

Homeostasis includes a demo service with intentional bugs that the framework can automatically fix:

```bash
# Make the demo script executable
chmod +x demo.sh

# Run the demo script
./demo.sh
```

The demo will:
1. Start the example FastAPI service with known bugs
2. Monitor for errors
3. Analyze error root causes
4. Generate and apply patches
5. Test the fixes
6. Restart the service with applied patches

### Manual Execution

If you prefer to run commands manually:

```bash
# Create necessary directories
mkdir -p logs logs/patches logs/backups sessions

# Run the orchestrator in demo mode
python3 orchestrator/orchestrator.py --demo
```

### Accessing the Example Service

After running the demo, you can access the example service at:
- API: http://localhost:8000/
- Swagger UI: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

## Configuration

Homeostasis is configured through `orchestrator/config.yaml`. The main configuration sections include:

### General Settings

```yaml
general:
  project_root: "."
  log_level: "INFO"
  environment: "development"
```

### Service Settings

```yaml
service:
  name: "your_service_name"
  path: "path/to/your/service"
  start_command: "command to start your service"
  stop_command: "command to stop your service"
  health_check_url: "http://localhost:port/health"
  health_check_timeout: 5
  log_file: "logs/yourservice.log"
```

### Monitoring Settings

```yaml
monitoring:
  enabled: true
  log_level: "INFO"
  watch_patterns:
    - "*.log"
    - "logs/*.log"
  check_interval: 5
```

### Analysis Settings

```yaml
analysis:
  rule_based:
    enabled: true
  ai_based:
    enabled: false
```

### Patch Generation Settings

```yaml
patch_generation:
  templates_dir: "modules/patch_generation/templates"
  generated_patches_dir: "logs/patches"
  backup_original_files: true
```

### Testing and Deployment

```yaml
testing:
  enabled: true
  test_command: "pytest tests/"
  test_timeout: 30

deployment:
  enabled: true
  restart_service: true
  backup_before_deployment: true
  backup_dir: "logs/backups"
```

### Rollback Settings

```yaml
rollback:
  enabled: true
  auto_rollback_on_failure: true
  max_sessions_to_keep: 10
```

## Integrating with Your Project

### 1. Configure Your Service

Edit `orchestrator/config.yaml` to point to your service:

```yaml
service:
  name: "your_service"
  path: "/path/to/your/service"
  start_command: "your service start command"
  health_check_url: "http://localhost:your_port/health"
```

### 2. Add Monitoring to Your Application

#### For FastAPI applications:

```python
from modules.monitoring.middleware import add_logging_middleware

app = FastAPI()

add_logging_middleware(
    app, 
    service_name="your_service",
    log_level="INFO", 
    exclude_paths=["/health", "/metrics"]
)
```

#### For other Python applications:

```python
from modules.monitoring.logger import MonitoringLogger

logger = MonitoringLogger(
    service_name="your_service",
    log_level="INFO",
    include_system_info=True
)

# Log different levels
logger.info("Information message")
logger.warning("Warning message", context="additional information")
logger.error("Error occurred", component="database")

# Log exceptions with rich context
try:
    # Some code that might raise an exception
    result = problematic_function()
except Exception as e:
    logger.exception(e, include_locals=True, operation="data_processing")
```

### 3. Run the Orchestrator

```bash
python orchestrator/orchestrator.py
```

## Command Line Options

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

## Example Workflows

### 1. Detect and Fix Missing Error Handling

**Error Scenario**: An endpoint doesn't check if a requested ID exists before accessing it, causing a KeyError exception.

**Detection**:
- Homeostasis monitors the service logs and detects the KeyError
- The analysis module identifies this as a missing error handling issue

**Fix Generation**:
- A patch is generated using the try_except_block.py.template
- The code is wrapped in a try/except block that handles the KeyError
- A proper 404 response is returned when the item is not found

**Validation and Deployment**:
- The patch is tested to ensure it fixes the issue
- If tests pass, the patch is deployed to the service
- The service is restarted to apply the fix

### 2. Detect and Fix Unsafe Environment Variable Conversion

**Error Scenario**: The code directly converts environment variables to integers without error handling.

**Detection**:
- Homeostasis detects a ValueError related to type conversion

**Fix Generation**:
- A patch is generated using the int_conversion_error.py.template
- A try/except block is added to handle conversion errors
- A default value is provided when conversion fails

**Validation and Deployment**:
- The patch is tested to ensure it handles invalid inputs correctly
- If tests pass, the patch is deployed and the service is restarted

## Monitoring Setup

### Basic Setup

```python
from modules.monitoring.logger import MonitoringLogger

logger = MonitoringLogger(
    service_name="my_service",
    log_level="INFO",
    include_system_info=True,
    enable_console_output=True
)
```

### Advanced Configuration

```python
logger = MonitoringLogger(
    service_name="my_service",
    log_level="INFO",
    include_system_info=True,
    enable_console_output=True,
    log_file_path="logs/custom_log.log",
    max_log_file_size_mb=10,
    backup_count=5,
    redact_sensitive_data=True,
    sensitive_keys=["password", "api_key", "secret"]
)
```

### Working with Error Reports

```python
from modules.monitoring.extractor import get_latest_errors, get_error_summary

# Get recent errors
errors = get_latest_errors(
    limit=10,
    levels=["ERROR", "CRITICAL"],
    service_name="my_service",
    exception_types=["KeyError", "ValueError"],
    hours_back=24
)

# Generate error summary
summary = get_error_summary(days_back=7)
```

## Troubleshooting

### Demo Script Issues

1. **Permission errors**:
   ```bash
   chmod +x demo.sh
   ```

2. **Missing directories**:
   ```bash
   mkdir -p logs logs/patches logs/backups sessions
   ```

3. **Missing dependencies**:
   ```bash
   pip install pyyaml requests fastapi uvicorn pydantic loguru pytest
   ```

### Service Startup Issues

1. **Port in use**:
   ```bash
   lsof -i :8000
   kill -9 $(lsof -ti:8000)
   ```

2. **Service not starting**:
   - Check logs in `logs/homeostasis.log`
   - Verify config in `orchestrator/config.yaml`

### Analysis and Patching Issues

1. **Rule matching not working**:
   - Check available rules: `ls modules/analysis/rules/`
   - Test rule matching with the CLI tool: `python modules/analysis/rule_cli.py test --error "KeyError: 'user_id'"`

2. **Failed to generate patches**:
   - Verify templates exist: `ls modules/patch_generation/templates/`
   - Check logs for detailed error information

### Rolling Back Changes

If a patch causes issues, you can roll back to the previous state:

```bash
python orchestrator/orchestrator.py --rollback
```

### When to Restart the Orchestrator

1. After changing configuration in `config.yaml`
2. After adding or modifying rules in the analysis module
3. After modifying patch templates
4. When changing the service being monitored