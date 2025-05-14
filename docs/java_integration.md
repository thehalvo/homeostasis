# Java Integration for Homeostasis

This document describes the Java language support in Homeostasis, providing detailed information about the implementation, capabilities, and usage.

## Overview

The Java integration module enables Homeostasis to detect, analyze, and fix errors in Java applications. It provides comprehensive exception handling for standard Java errors and framework-specific issues including Spring, Hibernate, and more.

## Features

- **Java Exception Handling**: Detect and analyze common Java exceptions including NullPointerException, ClassCastException, and many others
- **Spring Framework Support**: Handle Spring-specific issues like dependency injection failures, bean definition errors, and security exceptions
- **Hibernate/JPA Error Resolution**: Detect and fix common ORM issues like lazy loading initialization and transaction errors
- **Concurrency Error Detection**: Identify threading issues, deadlocks, race conditions, and concurrent modification exceptions
- **Build System Integration**: Analyze Maven and Gradle dependency issues

## Architecture

The Java integration is implemented as a plugin following Homeostasis's language plugin system. The main components are:

1. **JavaLanguagePlugin**: The entry point that implements the LanguagePlugin interface
2. **JavaExceptionHandler**: Processes Java exceptions and identifies their root causes
3. **JavaPatchGenerator**: Generates code fixes based on templates and analysis
4. **Java Error Rules**: JSON-based rule definitions for matching specific error patterns

## Integration with Orchestrator

The Java plugin is automatically registered with the Cross-Language Orchestrator, allowing Homeostasis to transparently handle Java errors alongside errors from other languages like Python and JavaScript.

## Java Error Detection

The system can detect a wide range of Java errors, including:

### Core Java Exceptions
- NullPointerException
- ClassCastException
- ArrayIndexOutOfBoundsException
- IllegalArgumentException
- UnsupportedOperationException
- NumberFormatException
- ClassNotFoundException

### Collection Framework Issues
- ConcurrentModificationException
- NoSuchElementException
- EmptyStackException

### I/O and Network Errors
- IOException
- FileNotFoundException
- SocketTimeoutException
- MalformedURLException

### Concurrency Problems
- InterruptedException
- IllegalMonitorStateException
- ExecutionException
- TimeoutException
- RejectedExecutionException
- Deadlocks and race conditions

### Database and JDBC Exceptions
- SQLException
- DataTruncation

### Framework-Specific Exceptions
- **Spring**: UnsatisfiedDependencyException, BeanCurrentlyInCreationException, AccessDeniedException
- **Hibernate**: LazyInitializationException, NonUniqueObjectException, PropertyValueException

## Patch Generation

For detected Java errors, the system can generate appropriate fixes using templates. These templates provide:

- Null checks to prevent NullPointerExceptions
- Bounds checking for array access
- Thread-safe collection usage
- Exception handling improvements
- Spring bean configuration corrections
- Hibernate session management fixes

## Configuration

### Enabling Java Support

Java support is enabled by default when the Java plugin is in the classpath. No additional configuration is required.

### Custom Java Rules

You can add custom rules for Java-specific errors in the following locations:

- `modules/analysis/rules/java/` - Core Java rules
- `modules/analysis/rules/spring/` - Spring framework rules
- `modules/analysis/rules/hibernate/` - Hibernate/JPA rules

Rules are defined in JSON format. Here's an example rule for detecting a common Spring error:

```json
{
  "id": "spring_autowired_failure",
  "pattern": "org\\.springframework\\.beans\\.factory\\.UnsatisfiedDependencyException.*No qualifying bean of type '([^']+)' available",
  "type": "UnsatisfiedDependencyException",
  "description": "Spring could not autowire a dependency because the required bean was not found",
  "root_cause": "spring_missing_bean",
  "suggestion": "Make sure the dependency is properly declared as a bean. Check that component scanning is properly configured to include the appropriate package. You may need to add @Component, @Service, @Repository, or @Bean annotation.",
  "confidence": "high",
  "severity": "high",
  "category": "spring",
  "framework": "spring"
}
```

### Custom Templates

Custom templates for Java patches can be added in:

```
modules/analysis/patch_generation/templates/java/
```

Each template corresponds to a specific error type and provides a parameterized code fix.

## Usage Examples

### Analyzing a Java Exception

```python
from modules.analysis.cross_language_orchestrator import get_orchestrator

# Initialize the orchestrator
orchestrator = get_orchestrator()

# Example Java exception
java_error = {
    "error_type": "java.lang.NullPointerException",
    "message": "Cannot invoke \"User.getName()\" because \"user\" is null",
    "stack_trace": [
        {
            "file": "UserService.java",
            "line": 42,
            "class": "UserServiceImpl",
            "function": "processUser",
            "package": "com.example.service"
        }
    ]
}

# Analyze the error
analysis = orchestrator.analyze_error(java_error)
print(f"Root cause: {analysis.get('root_cause')}")
print(f"Suggestion: {analysis.get('suggestion')}")
```

### Generating a Fix

```python
from modules.analysis.cross_language_orchestrator import get_orchestrator

# Initialize the orchestrator
orchestrator = get_orchestrator()

# Get the analysis for a Java error
analysis = orchestrator.analyze_error(java_error)

# Additional context about the code
context = {
    "code_snippet": "String name = user.getName();",
    "method_params": "User user"
}

# Generate a fix
fix = orchestrator.generate_fix(analysis, context)
print(f"Fix type: {fix.get('patch_type')}")
print(f"Patch code: {fix.get('patch_code')}")
```

## Advanced Integration

For advanced use cases, you can directly use the Java plugin:

```python
from modules.analysis.plugins.java_plugin import JavaLanguagePlugin

# Create an instance of the Java plugin
java_plugin = JavaLanguagePlugin()

# Use the plugin directly
analysis = java_plugin.analyze_error(java_error)
fix = java_plugin.generate_fix(analysis, context)
```

## Extending Java Support

To extend the Java integration:

1. **Add New Rules**: Create new rule JSON files in the appropriate directories
2. **Create New Templates**: Add templates for specific error types in the templates directory
3. **Enhance the Plugin**: Modify `java_plugin.py` to handle additional Java frameworks or error types

## Testing

The Java integration includes comprehensive tests in `tests/test_java_plugin.py`. Run the tests with:

```bash
pytest tests/test_java_plugin.py
```

## Limitations

Current limitations of the Java integration:

- Limited support for advanced Spring features like WebFlux
- No direct bytecode analysis (relies on exception information and stack traces)
- Requires structured error data for optimal analysis

## Future Improvements

Planned enhancements for the Java integration:

- Static code analysis for more precise fixes
- Integration with bytecode analysis tools
- Support for additional frameworks (Quarkus, Micronaut)
- Enhanced build tool integration
- Support for Java modules system

## Related Documentation

- [Homeostasis Architecture](architecture.md)
- [Plugin Architecture](plugin_architecture.md)
- [Language Adapters](language_adapters.md)