# Homeostasis Integration Testing Framework

This directory contains a comprehensive integration testing framework that validates Homeostasis's self-healing capabilities across all 40+ supported programming languages.

## Overview

The integration testing framework provides:

- **Language-specific test runners** for executing code in each supported language
- **Automated test generation** for common error patterns
- **Cross-language testing** for polyglot applications
- **Framework-specific tests** for popular frameworks in each language
- **Comprehensive reporting** with detailed metrics and insights

## Directory Structure

```
tests/integration/
├── language_integration_framework.py   # Core framework classes
├── language_runners.py                 # Language-specific test runners
├── test_suite_generator.py            # Automatic test generation
├── run_all_language_tests.py         # Main test execution script
├── test_suites/                       # Test cases organized by language
│   ├── python/
│   │   ├── basic_errors.json
│   │   └── framework_errors.json
│   ├── javascript/
│   │   └── basic_errors.json
│   ├── go/
│   │   └── basic_errors.json
│   ├── rust/
│   │   └── basic_errors.json
│   └── cross_language/
│       └── polyglot_scenarios.json
└── README.md                          # This file
```

## Supported Languages

The framework currently supports test runners for:

### Implemented Runners
- Python
- JavaScript/Node.js
- TypeScript
- Go
- Rust
- Java
- Ruby
- PHP
- C#/.NET
- Swift
- Kotlin

### Planned Runners
- Scala, Elixir, Erlang, Clojure
- Haskell, F#, OCaml
- Lua, R, MATLAB, Julia
- Nim, Crystal, Zig
- PowerShell, Bash
- SQL, YAML/JSON
- Terraform, Dockerfile, Ansible
- C++, Objective-C
- Perl, Dart, Groovy
- And more...

## Running Tests

### Run All Language Tests

```bash
# Run tests for all languages with available runners
python tests/integration/run_all_language_tests.py

# Run tests for specific languages
python tests/integration/run_all_language_tests.py --languages python javascript go

# Run tests by category
python tests/integration/run_all_language_tests.py --categories web systems mobile

# Run tests sequentially (instead of parallel)
python tests/integration/run_all_language_tests.py --sequential

# Specify number of parallel workers
python tests/integration/run_all_language_tests.py --workers 8

# Save report to specific location
python tests/integration/run_all_language_tests.py --output report.json
```

### Generate Test Suites

```bash
# Generate test suites for all languages
python tests/integration/test_suite_generator.py

# This will create test cases in test_suites/ directory
```

## Test Case Format

Test cases are defined in JSON format with the following structure:

```json
{
  "name": "python_null_pointer_exception",
  "language": "python",
  "description": "Test handling of AttributeError on None object",
  "test_type": "single",
  "source_code": {
    "main.py": "# Python code that triggers the error"
  },
  "expected_errors": [
    {
      "error_type": "AttributeError",
      "message": "'NoneType' object has no attribute 'name'"
    }
  ],
  "expected_fixes": [
    {
      "fix_type": "null_check",
      "description": "Add null check before accessing attribute"
    }
  ],
  "environment": {},
  "dependencies": ["requests", "pytest"],
  "frameworks": ["django"],
  "tags": ["basic", "null_safety"],
  "timeout": 300
}
```

## Test Types

### 1. Basic Error Tests
Test common programming errors across languages:
- Null/nil/undefined reference errors
- Array/list index out of bounds
- Type mismatches and casting errors
- Division by zero
- Resource leaks

### 2. Framework-Specific Tests
Test framework-specific error patterns:
- Django model and migration errors
- React component lifecycle issues
- Spring dependency injection problems
- Express.js middleware errors

### 3. Cross-Language Tests
Test interactions between different languages:
- Python calling Rust via WebAssembly
- JavaScript calling Java via gRPC
- Microservices communication errors
- Shared memory race conditions

### 4. Deployment Tests
Test deployment-related issues:
- Container configuration errors
- Environment variable problems
- Missing dependencies
- Port conflicts

## Adding New Language Support

To add support for a new language:

1. **Create a Test Runner** in `language_runners.py`:

```python
class NewLanguageIntegrationTestRunner(LanguageIntegrationTestRunner):
    async def setup_environment(self, test_case: IntegrationTestCase) -> Path:
        # Set up language-specific environment
        pass
        
    async def execute_code(self, test_dir: Path, test_case: IntegrationTestCase) -> Tuple[int, str, str]:
        # Execute the test code
        pass
        
    async def validate_environment(self, test_dir: Path) -> bool:
        # Validate language runtime is available
        pass
```

2. **Register the Runner** in `LANGUAGE_RUNNERS`:

```python
LANGUAGE_RUNNERS = {
    # ... existing runners ...
    "newlanguage": NewLanguageIntegrationTestRunner,
}
```

3. **Create Test Cases** in `test_suites/newlanguage/`:
   - `basic_errors.json` - Common error patterns
   - `framework_errors.json` - Framework-specific errors

## Test Execution Flow

1. **Environment Setup**: Create isolated test environment for each language
2. **Code Execution**: Run test code to trigger expected errors
3. **Error Detection**: Use language plugins to detect and analyze errors
4. **Fix Generation**: Generate fixes using Homeostasis healing system
5. **Fix Application**: Apply generated fixes to the code
6. **Validation**: Re-run code to verify fixes work correctly
7. **Cleanup**: Remove test environment and collect metrics

## Reports and Metrics

The framework generates comprehensive reports including:

- **Summary Statistics**: Total tests, pass/fail rates, duration
- **Language Statistics**: Performance by language
- **Category Statistics**: Results grouped by language category
- **Error Analysis**: Distribution of error types
- **Fix Effectiveness**: Success rates of generated fixes
- **Detailed Results**: Individual test case outcomes

## Integration with CI/CD

The test framework can be integrated into CI/CD pipelines:

```yaml
# GitHub Actions example
- name: Run Integration Tests
  run: |
    python tests/integration/run_all_language_tests.py \
      --categories ${{ matrix.category }} \
      --output test-report-${{ matrix.category }}.json
      
- name: Upload Test Report
  uses: actions/upload-artifact@v3
  with:
    name: integration-test-report
    path: test-report-*.json
```

## Performance Considerations

- Tests run in parallel by default (configurable workers)
- Each test has a configurable timeout (default: 300 seconds)
- Container-based isolation prevents interference between tests
- Resource limits can be set per language/test type

## Troubleshooting

### Common Issues

1. **Language runtime not found**: Ensure the language runtime is installed
2. **Permission errors**: Check file permissions in test directories
3. **Timeout errors**: Increase timeout for resource-intensive tests
4. **Container errors**: Ensure Docker is running if using containerized tests

### Debug Mode

Run with verbose logging:

```bash
python tests/integration/run_all_language_tests.py --verbose
```

## Contributing

To contribute new test cases or language support:

1. Add test cases to appropriate `test_suites/` directory
2. Implement language runner if needed
3. Update documentation
4. Submit PR with test results

## Future Enhancements

- [ ] Container-based test isolation for all languages
- [ ] Performance benchmarking suite
- [ ] Security vulnerability testing
- [ ] Load testing for concurrent error scenarios
- [ ] Integration with cloud testing platforms
- [ ] Visual test result dashboard
- [ ] Automated test case generation from real-world errors