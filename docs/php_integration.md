# PHP Integration for Homeostasis

This document describes the PHP language support in Homeostasis, providing detailed information about the implementation, capabilities, and usage.

## Overview

The PHP integration module enables Homeostasis to detect, analyze, and fix errors in PHP applications. It provides comprehensive error handling for standard PHP errors and framework-specific issues including Laravel, Symfony, and other PHP frameworks.

## Features

- **PHP Error Handling**: Detect and analyze common PHP errors including undefined variables, null references, type errors, and syntax errors
- **Laravel Framework Support**: Handle Laravel-specific issues like model not found exceptions, validation errors, and route issues
- **Symfony Framework Support**: Detect and fix Symfony-specific errors such as dependency injection issues, container exceptions, and routing problems
- **WordPress Integration**: Support for WordPress-specific issues and common CMS patterns
- **Database Error Detection**: Identify and fix PDO exceptions, query errors, and ORM issues
- **Framework-Agnostic Solutions**: Provide generic PHP fixes that work across frameworks

## Architecture

The PHP integration is implemented as a plugin following Homeostasis's language plugin system. The main components are:

1. **PHPLanguagePlugin**: The entry point that implements the LanguagePlugin interface
2. **PHPExceptionHandler**: Processes PHP errors and exceptions and identifies their root causes
3. **PHPPatchGenerator**: Generates code fixes based on templates and analysis
4. **PHP Error Rules**: JSON-based rule definitions for matching specific error patterns

## Integration with Orchestrator

The PHP plugin is automatically registered with the Cross-Language Orchestrator, allowing Homeostasis to transparently handle PHP errors alongside errors from other languages like Python, JavaScript, and Java.

## PHP Error Detection

The system can detect a wide range of PHP errors, including:

### Core PHP Errors
- Undefined variable
- Undefined method/function
- Null reference (Call to a member function on null)
- Type errors
- Parse/Syntax errors
- Undefined index/offset
- Class not found
- Autoloader errors
- Division by zero
- Memory limit exceeded
- Maximum execution time exceeded

### Laravel Framework Issues
- ModelNotFoundException
- ValidationException
- NotFoundHttpException (route not found)
- QueryException
- MassAssignmentException
- AuthenticationException
- AuthorizationException
- ViewException
- TokenMismatchException (CSRF)

### Symfony Framework Issues
- ServiceNotFoundException
- TwigError
- FormException
- RouteNotFoundException
- ParameterNotFoundException
- MissingTemplateException
- CacheException
- MessengerException

### Database Errors
- PDOException
- Database connection failures
- Query syntax errors
- Constraint violations

## Patch Generation

For detected PHP errors, the system can generate appropriate fixes using templates. These templates provide:

- Variable initialization to prevent undefined variable errors
- Null checks to prevent null reference errors
- Array key existence checks
- Exception handling improvements
- Laravel model lookup improvements
- Symfony service container access patterns
- Type checking and conversion

## Configuration

### Enabling PHP Support

PHP support is enabled by default when the PHP plugin is in the classpath. No additional configuration is required.

### Custom PHP Rules

You can add custom rules for PHP-specific errors in the following locations:

- `modules/analysis/rules/php/` - Core PHP rules
- `modules/analysis/rules/php/laravel_errors.json` - Laravel framework rules
- `modules/analysis/rules/php/symfony_errors.json` - Symfony framework rules
- `modules/analysis/rules/php/wordpress_errors.json` - WordPress rules

Rules are defined in JSON format. Here's an example rule for detecting a common PHP error:

```json
{
  "id": "php_undefined_variable",
  "pattern": "Undefined variable\\s*:\\s*\\$(\\w+)",
  "type": "E_NOTICE",
  "description": "Undefined variable accessed",
  "root_cause": "php_undefined_variable",
  "suggestion": "Initialize the variable before using it or check with isset() if it exists. Consider using null coalescing operator (??) for fallback values.",
  "confidence": "high",
  "severity": "medium",
  "category": "core",
  "framework": "php"
}
```

### Custom Templates

Custom templates for PHP patches can be added in:

```
modules/analysis/patch_generation/templates/php/
```

Each template corresponds to a specific error type and provides a parameterized code fix.

## Usage Examples

### Analyzing a PHP Exception

```python
from modules.analysis.cross_language_orchestrator import get_orchestrator

# Initialize the orchestrator
orchestrator = get_orchestrator()

# Example PHP error
php_error = {
    "type": "ErrorException",
    "message": "Undefined variable: user",
    "file": "/var/www/html/app/Controllers/UserController.php",
    "line": 25,
    "trace": [
        {
            "file": "/var/www/html/app/Controllers/UserController.php",
            "line": 25,
            "function": "getUserProfile",
            "class": "App\\Controllers\\UserController"
        }
    ],
    "level": "E_NOTICE"
}

# Analyze the error
analysis = orchestrator.analyze_error(php_error)
print(f"Root cause: {analysis.get('root_cause')}")
print(f"Suggestion: {analysis.get('suggestion')}")
```

### Generating a Fix

```python
from modules.analysis.cross_language_orchestrator import get_orchestrator

# Initialize the orchestrator
orchestrator = get_orchestrator()

# Get the analysis for a PHP error
analysis = orchestrator.analyze_error(php_error)

# Additional context about the code
context = {
    "code_snippet": "public function getUserProfile()\n{\n    return $user->profile;\n}"
}

# Generate a fix
fix = orchestrator.generate_fix(analysis, context)
print(f"Fix type: {fix.get('patch_type')}")
print(f"Patch code: {fix.get('patch_code')}")
```

## Advanced Integration

For advanced use cases, you can directly use the PHP plugin:

```python
from modules.analysis.plugins.php_plugin import PHPLanguagePlugin

# Create an instance of the PHP plugin
php_plugin = PHPLanguagePlugin()

# Use the plugin directly
analysis = php_plugin.analyze_error(php_error)
fix = php_plugin.generate_fix(analysis, context)
```

## Extending PHP Support

To extend the PHP integration:

1. **Add New Rules**: Create new rule JSON files in the appropriate directories
2. **Create New Templates**: Add templates for specific error types in the templates directory
3. **Enhance the Plugin**: Modify `php_plugin.py` to handle additional PHP frameworks or error types

## Testing

The PHP integration includes comprehensive unit tests that verify the functionality of:

- Error normalization and format conversion
- Rule-based error detection for various PHP errors
- Patch generation for common error types
- Framework-specific error handling (Laravel, Symfony)

You can run the tests with:

```bash
python -m unittest tests/test_php_plugin.py
```

## Limitations and Future Work

Current limitations of the PHP integration include:

1. **Limited WordPress Support**: While basic WordPress support is included, specialized WordPress error patterns will be enhanced in future versions
2. **Framework Versions**: The current implementation is primarily tested with PHP 7.x and 8.x, Laravel 8/9, and Symfony 5/6
3. **Composer Integration**: Future updates will provide deeper integration with Composer dependency management

Planned enhancements:

1. **PHP 8.x Feature Support**: Enhanced support for PHP 8.x features like attributes, named arguments, and match expressions
2. **Additional Frameworks**: Support for more PHP frameworks like CodeIgniter, Yii, and CakePHP
3. **Static Analysis Integration**: Integration with static analysis tools like PHPStan and Psalm
4. **Composer Dependency Resolution**: Better handling of Composer dependency issues

## PHP Error Codes and Reporting Levels

The system handles various PHP error levels and reporting configurations:

- E_ERROR: Fatal runtime errors
- E_WARNING: Non-fatal runtime errors
- E_PARSE: Syntax errors
- E_NOTICE: Runtime notices
- E_CORE_ERROR: Fatal errors during PHP initialization
- E_CORE_WARNING: Non-fatal errors during PHP initialization
- E_COMPILE_ERROR: Fatal compilation errors
- E_COMPILE_WARNING: Non-fatal compilation warnings
- E_USER_ERROR: User-generated errors
- E_USER_WARNING: User-generated warnings
- E_USER_NOTICE: User-generated notices
- E_STRICT: Runtime notices for best practices
- E_RECOVERABLE_ERROR: Catchable fatal errors
- E_DEPRECATED: Runtime notices for deprecated features
- E_USER_DEPRECATED: User-generated deprecated notices

## Contributing

To contribute to the PHP integration:

1. Add or enhance rule definitions in the PHP rule files
2. Create additional templates for common PHP error patterns
3. Extend support for PHP frameworks
4. Improve stack trace parsing and error detection

Refer to the Homeostasis contribution guidelines for more details on the contribution process.