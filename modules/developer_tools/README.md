# Homeostasis Developer Tools

This module provides developer tools for testing, simulating, and integrating with the Homeostasis self-healing framework.

## Components

### 1. Healing Simulation/Sandbox Mode

The sandbox module provides an isolated environment for testing healing scenarios without affecting production systems.

**Features:**
- Isolated sandboxed environments
- Docker support for enhanced isolation
- Parallel simulation execution
- Test case management
- Performance metrics collection

**Usage:**
```bash
# Run a simulation
homeostasis dev simulate --name test-sim --language python --code "def broken(): return 1/0"

# Run with Docker isolation
homeostasis dev simulate --name test-sim --language python --file broken.py --docker

# Export results
homeostasis dev simulate --name test-sim --language python --file broken.py --output results.json
```

### 2. API Client Libraries

Client libraries for integrating Homeostasis into your applications.

#### Python Client

```python
from modules.developer_tools.api_clients.python_client import HomeostasisClient, ErrorReport

# Initialize client
client = HomeostasisClient("http://localhost:8080", api_key="your-api-key")

# Report an error
error = ErrorReport(
    error_message="Division by zero",
    stack_trace="Traceback...",
    language="python"
)
error_id = client.report_error(error)

# Check healing status
result = client.get_healing_status(healing_id)
print(f"Status: {result.status}, Success: {result.success}")

# Real-time monitoring
client.connect_websocket()
client.subscribe("healing_completed", lambda data: print(f"Healing completed: {data}"))
```

#### JavaScript Client

```javascript
import { HomeostasisClient, ErrorReport } from './api_clients/javascript_client.js';

// Initialize client
const client = new HomeostasisClient('http://localhost:8080', 'your-api-key');

// Report an error
const error = new ErrorReport({
  errorMessage: 'Cannot read property of undefined',
  stackTrace: 'Error: ...',
  language: 'javascript'
});

const errorId = await client.reportError(error);

// Check healing status
const result = await client.getHealingStatus(healingId);
console.log(`Status: ${result.status}, Success: ${result.success}`);

// Real-time monitoring
client.connectWebSocket();
client.subscribe('healing_completed', (data) => {
  console.log('Healing completed:', data);
});
```

#### Go Client

```go
import (
    "context"
    homeostasis "github.com/homeostasis/go-client"
)

// Initialize client
client := homeostasis.NewClient("http://localhost:8080", 
    homeostasis.WithAPIKey("your-api-key"))

// Report an error
error := &homeostasis.ErrorReport{
    ErrorMessage: "nil pointer dereference",
    StackTrace:   "goroutine 1 [running]...",
    Language:     "go",
}

errorID, err := client.ReportError(context.Background(), error)

// Check healing status
result, err := client.GetHealingStatus(context.Background(), healingID)
fmt.Printf("Status: %s, Success: %v\n", result.Status, result.Success)

// Real-time monitoring
client.ConnectWebSocket(context.Background(), 
    func(data interface{}) {
        fmt.Println("Message:", data)
    }, nil, nil)
```

### 3. Healing Effectiveness Calculator

Calculate and analyze the effectiveness of healing operations.

**Features:**
- Success rate calculation
- Performance metrics
- Time trend analysis
- Language and error type breakdown
- Recommendations generation
- Multi-format reporting (JSON, HTML, Markdown)

**Usage:**
```bash
# Calculate overall effectiveness
homeostasis dev effectiveness

# Filter by date range
homeostasis dev effectiveness --start-date 2024-01-01 --end-date 2024-12-31

# Filter by language
homeostasis dev effectiveness --language python --format html --output report.html

# Export as markdown
homeostasis dev effectiveness --format markdown --output effectiveness.md
```

**Metrics Tracked:**
- Total healings and success rate
- Average, median, fastest, and slowest healing times
- Patches generated and applied
- Rollback and human intervention rates
- Confidence and complexity distributions

### 4. Template Validation CLI Tool

Validate healing templates to ensure they generate correct code.

**Features:**
- Syntax validation for Jinja2 templates
- Language-specific code validation
- Metadata extraction and validation
- Variable usage analysis
- Batch validation for directories
- Multiple output formats

**Usage:**
```bash
# Validate a single template
homeostasis dev validate /path/to/template.py.template

# Validate all templates in a directory
homeostasis dev validate /path/to/templates -r

# Generate JSON report
homeostasis dev validate /path/to/templates -r --format json --output validation.json

# Fail on warnings
homeostasis dev validate /path/to/templates --fail-on-warning
```

**Template Metadata Format:**
```jinja2
{# metadata:
language: python
error_type: division_by_zero
description: Fix division by zero errors
variables:
  - error_message
  - file_path
  - line_number
  - denominator_var
required_imports:
  - math
supported_frameworks:
  - django
  - flask
tags:
  - arithmetic
  - exception-handling
#}

{% if denominator_var == "0" or denominator_var == 0 %}
if {{ denominator_var }} == 0:
    raise ValueError("Division by zero is not allowed")
result = numerator / {{ denominator_var }}
{% endif %}
```

## Installation

The developer tools are included with the main Homeostasis installation. Additional dependencies:

```bash
# For Python client
pip install requests websocket-client

# For effectiveness calculator
pip install numpy pandas

# For template validator
pip install jinja2 pyyaml
```

## API Endpoints

The client libraries interact with these main endpoints:

- `POST /api/v1/errors` - Report errors
- `GET /api/v1/healings/{id}` - Get healing status
- `POST /api/v1/healings` - Trigger healing
- `GET /api/v1/health` - System health
- `GET /api/v1/metrics` - Performance metrics
- `GET /api/v1/rules` - List healing rules
- `WS /ws` - WebSocket for real-time updates

## Examples

### Running a Complete Simulation

```python
from modules.developer_tools.sandbox import HealingSimulator, SimulationConfig

# Configure simulation
config = SimulationConfig(
    name="null-pointer-test",
    language="python",
    error_type="AttributeError",
    code_snippet="""
def process_user(user):
    return user.name.upper()

# This will fail if user is None
process_user(None)
""",
    test_cases=[
        {
            "name": "test_with_valid_user",
            "type": "unit",
            "input": {"name": "John"},
            "expected": "JOHN"
        }
    ]
)

# Run simulation
simulator = HealingSimulator()
with simulator.simulation_context() as sim:
    result = sim.simulate(config)
    
    if result.success:
        print("Healing successful!")
        print(f"Patches applied: {result.patches_applied}")
    else:
        print("Healing failed")
        print(f"Original error: {result.original_error}")
```

### Monitoring Healing Effectiveness

```python
from modules.developer_tools.effectiveness_calculator import EffectivenessCalculator, HealingMetrics
from datetime import datetime

calculator = EffectivenessCalculator()

# Add a metric
metric = HealingMetrics(
    healing_id="abc123",
    error_type="NullPointerException",
    language="java",
    success=True,
    time_to_detect=0.5,
    time_to_analyze=2.0,
    time_to_generate_patch=1.5,
    time_to_test=3.0,
    time_to_apply=0.5,
    patches_generated=3,
    patches_tested=3,
    patches_applied=1,
    confidence_score=0.85
)
calculator.add_metric(metric)

# Generate report
report = calculator.calculate_effectiveness()
print(f"Success Rate: {report.success_rate:.1%}")
print(f"Average Healing Time: {report.average_healing_time:.1f}s")

# Predict effectiveness for new error
success_prob, expected_time = calculator.predict_effectiveness(
    error_type="NullPointerException",
    language="java",
    complexity=0.7
)
print(f"Predicted success: {success_prob:.1%}, Expected time: {expected_time:.1f}s")
```

## Contributing

When adding new developer tools:

1. Create a new module in the `developer_tools` directory
2. Add appropriate CLI commands in `homeostasis_cli.py`
3. Include comprehensive tests
4. Update this README with usage examples
5. Ensure compatibility with existing tools

## Testing

Run the developer tools test suite:

```bash
pytest tests/test_developer_tools/
```

For integration testing with actual healing scenarios:

```bash
python -m pytest tests/test_developer_tools/test_integration.py -v
```