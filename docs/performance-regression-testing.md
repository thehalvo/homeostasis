# Performance Regression Testing

The Homeostasis framework includes a comprehensive performance regression testing infrastructure to ensure that system performance remains optimal as the codebase evolves.

## Overview

The performance regression testing system consists of several components:

1. **Performance Tracker**: Measures execution time, memory usage, and CPU utilization
2. **Regression Detector**: Compares current performance against established baselines
3. **Test Framework**: Provides decorators and utilities for writing performance tests
4. **CI/CD Integration**: Automated testing in GitHub Actions
5. **Dashboard**: Web-based visualization of performance trends

## Writing Performance Tests

### Basic Usage

```python
from testing.performance_regression import PerformanceRegressionTester

def test_my_component():
    tester = PerformanceRegressionTester()
    
    def operation_to_test():
        # Your code here
        result = expensive_operation()
        return result
    
    # Run benchmark
    results = tester.benchmark(
        operation_to_test,
        "my_component_operation",
        iterations=20,
        metadata={"input_size": 1000}
    )
    
    # Check for regressions
    assert len(results["regressions"]) == 0
```

### Using Decorators

```python
from testing.performance_regression import performance_test

@performance_test(name="matrix_multiplication", iterations=10)
def test_matrix_operations():
    matrix1 = generate_matrix(100, 100)
    matrix2 = generate_matrix(100, 100)
    result = multiply_matrices(matrix1, matrix2)
    return result
```

### Context Manager

```python
from testing.performance_regression import PerformanceRegressionTester

def test_with_context():
    tester = PerformanceRegressionTester()
    
    with tester.measure("database_query"):
        # Perform database operations
        results = db.query("SELECT * FROM large_table")
        process_results(results)
```

## Performance Metrics

The system tracks three key metrics:

1. **Duration**: Execution time in seconds
2. **Memory Delta**: Change in memory usage (MB)
3. **CPU Usage**: Average CPU percentage during execution

## Regression Detection

Regressions are detected using statistical analysis:

- **Warning Threshold**: 20% slower than baseline (duration)
- **Critical Threshold**: 50% slower than baseline (duration)
- **Confidence**: Based on standard deviation from baseline

## Baseline Management

### Establishing Baselines

Baselines are automatically created when running tests with the environment variable:

```bash
UPDATE_PERFORMANCE_BASELINE=true pytest tests/test_performance_regression.py
```

### Viewing Baselines

Use the performance dashboard or query the SQLite database:

```python
from testing.performance_regression import PerformanceRegressionDetector

detector = PerformanceRegressionDetector()
baseline = detector.get_baseline("test_name")
print(f"Mean duration: {baseline.mean_duration:.3f}s")
```

## CI/CD Integration

Performance tests run automatically on:

- Every push to main/develop branches
- Pull requests to main
- Nightly scheduled runs
- Manual workflow dispatch

### GitHub Actions Workflow

The workflow:
1. Runs performance tests across Python versions
2. Downloads previous baselines
3. Compares results and detects regressions
4. Comments on PRs with performance impact
5. Updates baselines on main branch

## Performance Dashboard

Run the web dashboard to visualize performance trends:

```bash
python -m modules.testing.performance_dashboard
```

Access at `http://localhost:5000`

Features:
- Real-time performance metrics
- Trend analysis over time
- Regression detection alerts
- Test coverage statistics

## Best Practices

1. **Isolate Tests**: Ensure tests are independent and repeatable
2. **Warm-up Runs**: Include warm-up iterations to stabilize measurements
3. **Multiple Iterations**: Run tests multiple times for statistical significance
4. **Environment Consistency**: Document hardware/software requirements
5. **Metadata**: Include relevant context (input size, configuration, etc.)

## Interpreting Results

### Performance Report Structure

```json
{
  "name": "test_name",
  "iterations": 20,
  "duration": {
    "mean": 0.052,
    "std": 0.003,
    "min": 0.048,
    "max": 0.061
  },
  "memory": {
    "mean": 12.5,
    "std": 0.8
  },
  "regressions": [
    {
      "metric_type": "duration",
      "baseline_value": 0.035,
      "current_value": 0.052,
      "regression_factor": 1.49,
      "severity": "warning"
    }
  ]
}
```

### Common Causes of Regressions

1. **Algorithm Changes**: O(n) to O(n²) complexity
2. **Resource Leaks**: Memory not properly released
3. **Dependencies**: Updated libraries with performance impact
4. **Concurrency Issues**: Lock contention or synchronization overhead
5. **I/O Operations**: Increased disk/network access

## Troubleshooting

### False Positives

If experiencing false regression alerts:

1. Check system load during test execution
2. Verify baseline was established under similar conditions
3. Increase iteration count for better statistics
4. Review recent environment changes

### Missing Baselines

If baselines are missing:

```bash
# Download from CI artifacts
gh run download --name performance-baselines-py3.10

# Or create new baselines
UPDATE_PERFORMANCE_BASELINE=true pytest tests/test_performance_regression.py
```

## Advanced Usage

### Custom Thresholds

```python
detector = PerformanceRegressionDetector()
detector.thresholds["duration"]["warning"] = 1.1  # 10% threshold
detector.thresholds["duration"]["critical"] = 1.3  # 30% threshold
```

### Performance Profiling Integration

```python
import cProfile
from testing.performance_regression import PerformanceRegressionTester

def test_with_profiling():
    tester = PerformanceRegressionTester()
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    with tester.measure("complex_operation"):
        result = complex_operation()
    
    profiler.disable()
    profiler.dump_stats("performance_profile.stats")
```

## Contributing

When adding new performance tests:

1. Follow naming convention: `test_<component>_performance`
2. Include appropriate metadata
3. Document expected performance characteristics
4. Add to CI test suite
5. Monitor dashboard after deployment

For more information, see the [Testing Guide](testing-guide.md).