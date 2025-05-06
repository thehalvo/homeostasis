# Contributing Rules to Homeostasis

This guide provides detailed instructions for contributing new rules to the Homeostasis framework's analysis module.

## What are Rules?

In Homeostasis, rules are pattern-matching logic used to:

1. Identify specific error types
2. Determine root causes
3. Suggest appropriate fixes

Rules form the backbone of the analysis engine, enabling the system to recognize and respond to errors automatically.

## Rule Structure

Each rule consists of the following components:

1. **Rule ID**: A unique identifier for the rule
2. **Description**: A clear explanation of what the rule detects
3. **Target**: The application type, framework, or language the rule applies to
4. **Pattern**: The error pattern to match (regex, string, or structured data)
5. **Context Requirements**: Additional context needed to confirm the match
6. **Fix Template**: The suggested fix template to use
7. **Parameters**: Variables to extract from the error for generating the fix

## Creating a New Rule

### 1. Identify the Error Pattern

Start by identifying a common error pattern that would benefit from automated healing. Good candidates:

- Frequently occurring errors
- Errors with clear signatures
- Errors with well-understood fixes

### 2. Choose the Appropriate Rule Location

Rules are organized in directories based on their targets:

```
modules/analysis/rules/
├── authentication/       # Authentication-related rules
├── authorization/        # Authorization-related rules
├── custom/              # Custom application-specific rules
├── database/            # Database-related rules
├── django/              # Django-specific rules
├── fastapi/             # FastAPI-specific rules
├── flask/               # Flask-specific rules
├── network/             # Network-related rules
├── python/              # General Python rules
└── validation/          # Input validation rules
```

Choose the appropriate directory or suggest a new one if needed.

### 3. Create the Rule File

Rules can be defined in JSON format. Create a new file with the following structure:

```json
{
  "rule_id": "unique_rule_id",
  "description": "Clear description of what this rule detects",
  "target": "framework_or_language",
  "version_range": "applicable_versions",
  "pattern": {
    "type": "regex",
    "value": "error_pattern_regex"
  },
  "context_requirements": {
    "stack_trace": true,
    "code_snippet": true,
    "variables": ["var1", "var2"]
  },
  "severity": "high|medium|low",
  "fix": {
    "template": "template_name",
    "parameters": {
      "param1": "{{extraction.param1}}",
      "param2": "{{extraction.param2}}"
    }
  },
  "examples": [
    {
      "input": "Example error message",
      "expected_match": true,
      "expected_parameters": {
        "param1": "expected_value1",
        "param2": "expected_value2"
      }
    }
  ]
}
```

### 4. Define the Pattern

The pattern can be defined in several ways:

- **Regex**: For matching error messages with regular expressions
  ```json
  "pattern": {
    "type": "regex",
    "value": "KeyError: '([^']+)'"
  }
  ```

- **JSON Path**: For structured data like JSON logs
  ```json
  "pattern": {
    "type": "json_path",
    "value": "$.error.type"
  }
  ```

- **Complex Condition**: For more sophisticated matching
  ```json
  "pattern": {
    "type": "condition",
    "value": {
      "operator": "and",
      "conditions": [
        {"field": "error_type", "operator": "equals", "value": "KeyError"},
        {"field": "module", "operator": "contains", "value": "my_module"}
      ]
    }
  }
  ```

### 5. Link to Fix Templates

Each rule should reference an appropriate fix template from the patch generation module:

```json
"fix": {
  "template": "keyerror_fix",
  "parameters": {
    "missing_key": "{{extraction.key_name}}",
    "dict_variable": "{{extraction.dict_name}}"
  }
}
```

### 6. Add Test Examples

Include examples to validate your rule:

```json
"examples": [
  {
    "input": "KeyError: 'user_id' in function get_user at line 42",
    "expected_match": true,
    "expected_parameters": {
      "missing_key": "user_id",
      "dict_variable": "user_data"
    }
  },
  {
    "input": "ValueError: Invalid input",
    "expected_match": false
  }
]
```

## Testing Your Rule

Before submitting a rule contribution:

1. **Unit Test**: Create a unit test in the `tests/` directory
2. **Integration Test**: Test the rule with the entire analysis pipeline
3. **Performance Check**: Ensure the rule doesn't significantly impact analysis performance

Example test:

```python
def test_keyerror_rule():
    # Sample error input
    error_input = {
        "message": "KeyError: 'user_id'",
        "stack_trace": ["line 1", "line 2"],
        "code_context": {"line": "user_data['user_id']", "filename": "app.py"}
    }
    
    # Run analysis with rule
    result = analyzer.analyze(error_input)
    
    # Verify correct rule was matched
    assert result.rule_id == "python_keyerror_001"
    
    # Verify parameters were correctly extracted
    assert result.parameters["missing_key"] == "user_id"
    assert result.parameters["dict_variable"] == "user_data"
    
    # Verify correct fix template is suggested
    assert result.fix_template == "keyerror_fix"
```

## Rule Contribution Checklist

- [ ] Rule has a unique ID following the naming convention
- [ ] Rule is placed in the appropriate directory
- [ ] Pattern is clear and specific
- [ ] Fix template exists and is properly referenced
- [ ] Test examples are included
- [ ] Unit tests are provided
- [ ] Documentation is clear

## Best Practices

1. **Be Specific**: Make rule patterns as specific as possible to avoid false positives
2. **Document Well**: Include clear descriptions of what the rule detects
3. **Test Thoroughly**: Include multiple examples and edge cases
4. **Consider Performance**: Complex patterns may impact analysis speed
5. **Provide Context**: Include information about when this rule should be applied
6. **Balance Precision**: Too strict and it won't catch variants; too loose and it will cause false positives

## Frequently Asked Questions

### How do I decide between multiple rule types?
Consider which provides the clearest match with the fewest false positives.

### What if I need to add a new fix template?
See the [Contributing Patch Templates guide](contributing-templates.md) for instructions.

### How do I test my rule against real-world errors?
Use the `analyzer.py` tool with sample logs containing the target error patterns.

---

By contributing rules to Homeostasis, you help build a more robust self-healing system capable of addressing a wider variety of issues. Thank you for your contribution!