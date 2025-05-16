# Backend Integration Testing Guide

This document provides comprehensive guidance for using Homeostasis' backend integration testing framework to validate error detection, analysis, and healing capabilities across multiple programming languages.

## Overview

The backend integration testing framework provides a unified approach to testing Homeostasis' capabilities across different programming languages. It allows you to:

- Test language-specific error detection and analysis
- Validate cross-language error handling and rule sharing
- Verify orchestration and healing workflows
- Measure and report healing efficiency metrics

## Architecture

The backend testing framework is built around these key components:

1. **LanguageTestFramework**: Core testing infrastructure in `modules/analysis/language_test_framework.py`
2. **SharedErrorSchema**: Error normalization system in `modules/analysis/shared_error_schema.py`
3. **BackendTestSuite**: Test case collections in `tests/test_cases/backend_test_suite.json`
4. **EnhancedCrossLanguageOrchestrator**: Language handling in `modules/analysis/enhanced_cross_language_orchestrator.py`
5. **SharedRuleSystem**: Language-agnostic rules in `modules/analysis/shared_rule_system.py`
6. **BackendTestingIntegration**: Orchestrator integration in `orchestrator/backend_testing_integration.py`
7. **HealingMetrics**: Performance tracking in `modules/analysis/healing_metrics.py`

## Writing Test Cases

### Basic Test Case Structure

Test cases are defined in JSON with this structure:

```json
{
  "test_id": "python_index_error_001",
  "language": "python",
  "error_type": "IndexError",
  "error_message": "list index out of range",
  "code_context": "data = [1, 2, 3]\nresult = data[10]",
  "expected_analysis": {
    "confidence": 0.95,
    "rule_id": "python.common_exceptions.index_error",
    "suggested_fix": "Ensure index is within valid range"
  },
  "cross_language_tests": [
    {
      "target_language": "javascript",
      "equivalent_error": "TypeError: Cannot read property of undefined",
      "shared_rule_id": "shared.collection_errors.index_out_of_bounds"
    }
  ]
}
```

### Test Categories

The framework supports several test categories:

1. **Language-Specific Tests**: Focus on one language's error patterns
2. **Cross-Language Tests**: Verify shared patterns across languages
3. **Rule Sharing Tests**: Validate language-agnostic rules
4. **Integration Tests**: Test the full error-to-healing pipeline
5. **Regression Tests**: Ensure previously fixed errors don't recur

## Running Tests

### Command Line Usage

```bash
# Run all backend integration tests
python -m tests.test_backend_integration

# Run tests for a specific language
python -m tests.test_backend_integration --language python

# Run a specific test suite
python -m tests.test_backend_integration --suite collection_errors

# Generate a detailed report
python -m tests.test_backend_integration --report detailed
```

### Programmatic Usage

```python
from modules.analysis.language_test_framework import LanguageTestRunner
from tests.test_backend_integration import load_test_suites

# Load and run all test suites
test_suites = load_test_suites()
runner = LanguageTestRunner()
results = runner.run_all(test_suites)

# Generate and save report
runner.generate_report(results, "backend_test_report.html")
```

## Interpreting Results

Test results include:

- Overall pass/fail status
- Language-specific success rates
- Cross-language pattern matching accuracy
- Rule application success rates
- Performance metrics

The HTML report provides detailed visualizations of these metrics.

## Extending the Framework

### Adding Support for New Languages

1. Create a language adapter in `modules/analysis/language_adapters.py`
2. Add language configuration to `modules/analysis/schemas/language_configs.json`
3. Implement a language plugin in `modules/analysis/plugins/`
4. Create language-specific test cases
5. Register the language with the `LanguageRegistry`

### Creating Custom Test Suites

1. Define a new test suite JSON file
2. Follow the schema defined in `modules/analysis/schemas/test_suite_schema.json`
3. Register the test suite in `tests/test_backend_integration.py`

## Integration with Orchestrator

The `BackendTestingManager` class in `orchestrator/backend_testing_integration.py` provides:

1. **Test Execution**: Run tests during deployment or on-demand
2. **Metrics Collection**: Track healing efficiency over time
3. **Regression Prevention**: Validate fixes against known issues
4. **Continuous Improvement**: Feed metrics back to the orchestrator

## Best Practices

1. **Test Coverage**: Ensure tests cover all supported languages and error patterns
2. **Cross-Language Validation**: Always include cross-language equivalence tests
3. **Realistic Test Data**: Use real-world error samples whenever possible
4. **Regression Testing**: Add test cases for every fixed issue
5. **Performance Tracking**: Monitor metrics to identify improvement areas

## Troubleshooting

### Common Issues

- **Test Failures**: Check language-specific error pattern definitions
- **Cross-Language Mismatches**: Verify shared error schema mappings
- **Performance Issues**: Examine language detection and rule matching times
- **Integration Errors**: Check orchestrator configuration and plugin registration

### Debugging Tools

The framework provides several debugging tools:

```python
# Enable debug mode
python -m tests.test_backend_integration --debug

# Generate verbose output
python -m tests.test_backend_integration --verbose

# Dump intermediate states
python -m tests.test_backend_integration --dump-states
```

## Reference

- [Shared Error Schema Documentation](./error_schema.md)
- [Language Plugin System](./plugin_architecture.md)
- [Cross-Language Orchestration](./cross_language_features.md)