# Contributing Patch Templates to Homeostasis

This guide provides detailed instructions for contributing new patch templates to the Homeostasis framework's patch generation module.

## What are Patch Templates?

Patch templates are predefined code patterns used to generate fixes for common errors. They serve as the "healing" part of the self-healing system by providing:

1. Structured code patterns for common fixes
2. Parameterizable sections that adapt to specific error contexts
3. Safe, tested solutions to known issues

## Template Structure

Each template consists of the following components:

1. **Template File**: A file containing the code pattern with placeholders
2. **Metadata**: Information about the template's purpose and usage
3. **Parameters**: Defines the variables that can be substituted into the template
4. **Validation Rules**: Conditions that must be met before applying the template

## Creating a New Template

### 1. Identify Common Fix Patterns

Start by identifying a recurring fix pattern that would benefit from templatization. Good candidates:

- Frequently applied fixes
- Fixes with consistent structure
- Patterns that vary in only a few parameters

### 2. Create the Template File

Templates are stored in the `modules/patch_generation/templates/` directory. Create a new file with a descriptive name following the naming convention:

```
<issue_type>_<fix_type>.py.template
```

For example:
- `keyerror_fix.py.template`
- `try_except_block.py.template`
- `dict_missing_param.py.template`

### 3. Write the Template Content

The template file should contain the code pattern with placeholders for variable parts. Placeholders are indicated with `{{placeholder_name}}` syntax:

```python
# Template: try_except_block.py.template
try:
    {{original_code}}
except {{exception_type}} as e:
    {{exception_handling_code}}
    {{fallback_code}}
```

### 4. Add Template Metadata

Create a JSON file with the same name but a `.json` extension to provide metadata for your template:

```json
{
  "template_id": "try_except_block",
  "description": "Wraps code in a try-except block to handle specific exceptions",
  "language": "python",
  "applicability": {
    "error_types": ["KeyError", "IndexError", "AttributeError"],
    "frameworks": ["any"]
  },
  "parameters": {
    "original_code": {
      "description": "The original code to wrap in the try block",
      "required": true,
      "type": "code_block"
    },
    "exception_type": {
      "description": "Exception type to catch",
      "required": true,
      "type": "string",
      "default": "Exception"
    },
    "exception_handling_code": {
      "description": "Code to handle the exception",
      "required": false,
      "type": "code_block",
      "default": "logger.error(f\"Error: {e}\")"
    },
    "fallback_code": {
      "description": "Code to execute as a fallback",
      "required": false,
      "type": "code_block",
      "default": "return None"
    }
  },
  "examples": [
    {
      "parameters": {
        "original_code": "data['user_id']",
        "exception_type": "KeyError",
        "exception_handling_code": "logger.error(f\"Missing key: {e}\")",
        "fallback_code": "return {}"
      },
      "result": "try:\n    data['user_id']\nexcept KeyError as e:\n    logger.error(f\"Missing key: {e}\")\n    return {}"
    }
  ],
  "validation": {
    "indent_sensitive": true,
    "requires_context_lines": 2,
    "max_affected_lines": 10
  }
}
```

### 5. Consider Template Variations

For complex templates, consider creating variations or composable templates:

```
basic_version.py.template
with_logging.py.template
with_retry.py.template
```

This allows for more flexibility without making templates overly complex.

## Template Best Practices

### 1. Keep Templates Focused

Each template should address a specific issue pattern. Avoid creating overly generic templates that try to solve multiple problems.

### 2. Handle Indentation

Templates should preserve code indentation. Use the `{{indent}}` placeholder if needed to maintain proper indentation:

```python
def some_function():
    {{indent}}try:
    {{indent}}    {{original_code}}
    {{indent}}except {{exception_type}} as e:
    {{indent}}    {{exception_handling_code}}
```

### 3. Include Documentation

Add comments to explain the template's purpose and usage:

```python
# Template: keyerror_fix.py.template
# Purpose: Safely access dictionary keys with a fallback value
# Usage: Replaces direct dictionary access with a get() method call

# Before:
# value = data['key']

# After:
# value = data.get('key', default_value)

{{dict_name}}.get({{key_name}}, {{default_value}})
```

### 4. Add Test Cases

Provide test cases for your template to demonstrate correct application:

```json
"test_cases": [
  {
    "original_code": "user = users['admin']",
    "expected_fix": "user = users.get('admin', None)",
    "parameters": {
      "dict_name": "users",
      "key_name": "'admin'",
      "default_value": "None"
    }
  }
]
```

## Testing Your Template

Before submitting:

1. **Unit Testing**: Create tests for template rendering with various parameters
2. **Integration Testing**: Test the template with the full patch generation pipeline
3. **Edge Cases**: Test with unusual inputs, indentation, and code structures

Example test:

```python
def test_try_except_template():
    # Setup template parameters
    params = {
        "original_code": "result = data['user_id']",
        "exception_type": "KeyError",
        "exception_handling_code": "logger.error(f\"Missing key: {e}\")",
        "fallback_code": "result = None"
    }
    
    # Generate patch using template
    patch = template_engine.render("try_except_block", params)
    
    # Verify output
    expected_output = """try:
    result = data['user_id']
except KeyError as e:
    logger.error(f"Missing key: {e}")
    result = None"""
    
    assert patch == expected_output
```

## Template Contribution Checklist

- [ ] Template file has descriptive name and `.py.template` extension
- [ ] Template has accompanying metadata JSON file
- [ ] Placeholders use consistent naming and format
- [ ] Documentation explains purpose and usage
- [ ] Test cases demonstrate proper functioning
- [ ] Edge cases are considered and handled
- [ ] Integration with analysis rules is defined

## Common Template Categories

Consider contributing templates for these common fix patterns:

1. **Error Handling**
   - try-except blocks
   - fallback values
   - defensive coding patterns

2. **Validation**
   - input checking
   - parameter validation
   - type conversion

3. **Resource Management**
   - context managers
   - proper cleanup
   - resource release

4. **Concurrency**
   - locking mechanisms
   - race condition fixes
   - thread safety

5. **Performance Optimizations**
   - caching
   - lazy loading
   - batching

## Frequently Asked Questions

### How do I determine the right level of abstraction for a template?
Focus on common patterns that require minimal context to understand and apply.

### Can templates include imports or affect multiple files?
Yes, but multi-file templates require special handling. See the advanced templates section.

### How are templates actually applied to code?
Templates are combined with extracted parameters and applied using the diff generation system.

---

By contributing high-quality templates, you help Homeostasis provide more effective healing capabilities across a wider range of issues. Thank you for your contribution!