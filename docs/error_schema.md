# Shared Error Schema Documentation

This document details the standardized error schema used by Homeostasis for cross-language error handling and analysis.

## Overview

The Shared Error Schema provides a language-agnostic representation of runtime errors that enables:

- Consistent error processing across languages
- Cross-language pattern matching and analysis
- Unified error classification and healing strategies
- Interoperability between language-specific plugins

## Schema Definition

### Core Error Structure

```json
{
  "error_id": "unique_identifier",
  "timestamp": "2023-05-16T14:32:45.123Z",
  "language": "python",
  "error_type": "TypeError",
  "error_message": "cannot use a string pattern on a bytes-like object",
  "stack_trace": [
    {
      "file": "app.py",
      "line": 42,
      "function": "process_data",
      "code_context": "result = re.match(pattern, binary_data)"
    }
  ],
  "environment": {
    "language_version": "3.9.5",
    "os": "linux",
    "framework": "fastapi",
    "framework_version": "0.68.0"
  },
  "normalized_classification": {
    "category": "type_error",
    "subcategory": "incompatible_types",
    "severity": "error",
    "root_cause_hint": "type_mismatch"
  },
  "metadata": {
    "service": "user_service",
    "component": "data_processor",
    "request_id": "req-12345"
  }
}
```

### Field Definitions

| Field | Type | Description |
|-------|------|-------------|
| `error_id` | string | Unique identifier for this error instance |
| `timestamp` | string | ISO 8601 timestamp when the error occurred |
| `language` | string | Programming language (python, javascript, java, go, etc.) |
| `error_type` | string | Language-specific error type or class |
| `error_message` | string | Full error message text |
| `stack_trace` | array | List of stack frames from most recent to origin |
| `environment` | object | Runtime environment details |
| `normalized_classification` | object | Language-agnostic error classification |
| `metadata` | object | Additional contextual information |

### Normalized Classification Categories

The `normalized_classification` field maps language-specific errors to universal categories:

| Category | Description | Examples |
|----------|-------------|----------|
| `reference_error` | Errors related to null/nil/undefined references | Python: `AttributeError`, JavaScript: `TypeError`, Java: `NullPointerException`, Go: `nil pointer dereference` |
| `type_error` | Type compatibility issues | Python: `TypeError`, JavaScript: `TypeError`, Java: `ClassCastException`, Go: `cannot convert X to Y` |
| `index_error` | Array/collection index violations | Python: `IndexError`, JavaScript: `RangeError`, Java: `ArrayIndexOutOfBoundsException`, Go: `index out of range` |
| `key_error` | Missing keys in maps/dictionaries | Python: `KeyError`, JavaScript: `TypeError`, Java: `NoSuchElementException`, Go: `key not in map` |
| `value_error` | Invalid value issues | Python: `ValueError`, JavaScript: `RangeError`, Java: `IllegalArgumentException`, Go: `strconv.ParseInt: parsing "x": invalid syntax` |
| `syntax_error` | Code syntax problems | Python: `SyntaxError`, JavaScript: `SyntaxError`, Java: `CompilationException`, Go: compilation errors |
| `io_error` | Input/output operation failures | Python: `IOError`, JavaScript: `Error`, Java: `IOException`, Go: `os.PathError` |
| `concurrency_error` | Thread/goroutine synchronization issues | Python: `RuntimeError` (deadlock), JavaScript: `RaceConditionError`, Java: `ConcurrentModificationException`, Go: `fatal error: concurrent map writes` |
| `dependency_error` | Missing or incompatible dependencies | Python: `ImportError`, JavaScript: `Error`, Java: `ClassNotFoundException`, Go: `package not found` |
| `database_error` | Database-related issues | Python: `SQLAlchemyError`, JavaScript: `MongoError`, Java: `SQLException`, Go: `sql.ErrNoRows` |
| `network_error` | Network communication failures | Python: `ConnectionError`, JavaScript: `NetworkError`, Java: `SocketException`, Go: `net.Error` |
| `memory_error` | Memory allocation/usage problems | Python: `MemoryError`, JavaScript: `RangeError`, Java: `OutOfMemoryError`, Go: `runtime: out of memory` |
| `permission_error` | Authorization/access issues | Python: `PermissionError`, JavaScript: `SecurityError`, Java: `SecurityException`, Go: `permission denied` |
| `timeout_error` | Operation time limits exceeded | Python: `TimeoutError`, JavaScript: `TimeoutError`, Java: `TimeoutException`, Go: `context deadline exceeded` |

## Language Adapter Implementation

Each supported language has an adapter that normalizes native errors to the shared schema:

```python
# Example Python adapter
class PythonErrorAdapter(BaseLanguageAdapter):
    def normalize_error(self, error, stack_trace, context):
        """Convert Python exception to shared schema format"""
        error_type = error.__class__.__name__
        
        normalized = {
            "error_id": generate_error_id(),
            "timestamp": get_iso_timestamp(),
            "language": "python",
            "error_type": error_type,
            "error_message": str(error),
            "stack_trace": self._format_stack_trace(stack_trace),
            "environment": self._get_environment_info(),
            "normalized_classification": self._classify_error(error_type, str(error)),
            "metadata": context.get("metadata", {})
        }
        
        return normalized
        
    def _classify_error(self, error_type, message):
        """Map Python error to normalized classification"""
        classifications = {
            "AttributeError": {"category": "reference_error", "subcategory": "missing_attribute"},
            "TypeError": {"category": "type_error", "subcategory": "incompatible_types"},
            "IndexError": {"category": "index_error", "subcategory": "out_of_bounds"},
            # ... more mappings
        }
        
        if error_type in classifications:
            classification = classifications[error_type]
        else:
            classification = {"category": "unknown", "subcategory": "general"}
            
        classification["severity"] = "error"
        return classification
```

## Cross-Language Matching

The schema enables pattern matching across languages:

```json
{
  "pattern_id": "null_reference_error",
  "description": "Attempting to access properties on null/nil/undefined",
  "languages": {
    "python": {
      "error_type": "AttributeError",
      "message_pattern": "'NoneType' object has no attribute '.*'"
    },
    "javascript": {
      "error_type": "TypeError",
      "message_pattern": "Cannot read propert.* of (null|undefined)"
    },
    "java": {
      "error_type": "NullPointerException",
      "message_pattern": ".*"
    },
    "go": {
      "error_type": "panic",
      "message_pattern": "runtime error: invalid memory address or nil pointer dereference"
    }
  },
  "normalized_classification": {
    "category": "reference_error",
    "subcategory": "null_reference",
    "severity": "error",
    "root_cause_hint": "missing_null_check"
  }
}
```

## Schema Validation

The schema is defined in JSON Schema format and can be found at `modules/analysis/schemas/error_schema.json`. All language adapters validate their output against this schema to ensure consistency.

## Extensions

The schema supports extensions for language-specific details while maintaining cross-language compatibility:

```json
{
  "error_id": "unique_identifier",
  "language": "python",
  "error_type": "ImportError",
  "error_message": "No module named 'missing_lib'",
  "normalized_classification": {
    "category": "dependency_error",
    "subcategory": "missing_module"
  },
  "language_specific": {
    "python": {
      "package_name": "missing_lib",
      "import_type": "module",
      "is_builtin": false
    }
  }
}
```

## Usage Examples

### Converting Native Errors

```python
# Python example
try:
    result = data["missing_key"]
except KeyError as e:
    adapter = PythonErrorAdapter()
    normalized_error = adapter.normalize_error(e, traceback.extract_tb(sys.exc_info()[2]), {})
    orchestrator.process_error(normalized_error)
```

```java
// Java example
try {
    String value = map.get("missing_key");
    if (value == null) {
        throw new NoSuchElementException("Key 'missing_key' not found");
    }
} catch (NoSuchElementException e) {
    JavaErrorAdapter adapter = new JavaErrorAdapter();
    Map<String, Object> normalizedError = adapter.normalizeError(e, e.getStackTrace(), new HashMap<>());
    orchestrator.processError(normalizedError);
}
```

### Cross-Language Rule Matching

```python
def find_matching_rules(normalized_error):
    """Find rules that match this error across languages"""
    matches = []
    
    for rule in shared_rule_registry:
        # Check if the normalized classification matches
        if rule["normalized_classification"]["category"] == normalized_error["normalized_classification"]["category"]:
            # Check language-specific pattern
            if rule.has_language_pattern(normalized_error["language"]):
                if rule.matches_error(normalized_error):
                    matches.append(rule)
    
    return matches
```

## Integration with Test Framework

The Shared Error Schema integrates with the backend testing framework to validate error handling:

```python
def test_cross_language_error_normalization():
    """Test that errors normalize consistently across languages"""
    # Create test errors in different languages
    python_error = create_test_python_error()
    js_error = create_test_js_error()
    java_error = create_test_java_error()
    go_error = create_test_go_error()
    
    # Normalize them
    python_adapter = PythonErrorAdapter()
    js_adapter = JavaScriptErrorAdapter()
    java_adapter = JavaErrorAdapter()
    go_adapter = GoErrorAdapter()
    
    norm_python = python_adapter.normalize_error(python_error, python_traceback, {})
    norm_js = js_adapter.normalize_error(js_error, js_traceback, {})
    norm_java = java_adapter.normalize_error(java_error, java_traceback, {})
    norm_go = go_adapter.normalize_error(go_error, go_traceback, {})
    
    # Verify they all map to the same normalized classification
    assert norm_python["normalized_classification"]["category"] == "index_error"
    assert norm_js["normalized_classification"]["category"] == "index_error"
    assert norm_java["normalized_classification"]["category"] == "index_error"
    assert norm_go["normalized_classification"]["category"] == "index_error"
```