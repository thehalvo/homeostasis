# JavaScript Integration Guide

This guide covers the JavaScript language integration for Homeostasis, providing support for error detection and healing in JavaScript applications, including both browser and Node.js environments.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Supported Error Types](#supported-error-types)
5. [Usage Examples](#usage-examples)
6. [Advanced Features](#advanced-features)
7. [Framework Support](#framework-support)
8. [Troubleshooting](#troubleshooting)
9. [API Reference](#api-reference)

## Overview

The JavaScript plugin for Homeostasis provides intelligent error detection, analysis, and automated fixing capabilities for JavaScript applications. It supports:

- **Browser JavaScript**: DOM errors, async operations, modern ES6+ features
- **Node.js**: Server-side errors, module loading, file system operations
- **Transpilation**: Babel, TypeScript, webpack, and other build tool errors
- **Dependencies**: npm/yarn package management and dependency conflicts
- **Frameworks**: React, Vue, Angular, Express, and many more

### Key Features

- Automatic error detection and categorization
- Intelligent fix generation with multiple options
- Dependency analysis and conflict resolution
- Transpilation error handling
- Support for modern JavaScript patterns (async/await, modules, etc.)
- Integration with popular build tools and frameworks

## Installation

### Prerequisites

- Python 3.8+ with the Homeostasis framework installed
- Node.js 14+ (for JavaScript project analysis)
- npm or yarn package manager

### Setup

1. **Install the JavaScript plugin** (if not already included):

```bash
# The JavaScript plugin is included by default with Homeostasis
# No additional installation required
```

2. **Verify installation**:

```python
from modules.analysis.plugins.javascript_plugin import JavaScriptLanguagePlugin

plugin = JavaScriptLanguagePlugin()
print(f"JavaScript plugin version: {plugin.VERSION}")
```

## Configuration

### Basic Configuration

The JavaScript plugin works out-of-the-box with minimal configuration. However, you can customize its behavior:

```python
# In your Homeostasis configuration
JAVASCRIPT_CONFIG = {
    "enabled": True,
    "environments": ["browser", "nodejs", "electron"],
    "frameworks": ["react", "vue", "angular", "express"],
    "transpilers": ["babel", "typescript", "webpack"],
    "package_managers": ["npm", "yarn", "pnpm"]
}
```

### Project-Specific Configuration

For JavaScript projects, create a `.homeostasis.json` file:

```json
{
  "language": "javascript",
  "project_type": "nodejs",
  "framework": "express",
  "entry_points": ["src/index.js", "app.js"],
  "dependencies": {
    "analyze_on_error": true,
    "auto_install_missing": false
  },
  "transpilation": {
    "detect_babel_errors": true,
    "detect_typescript_errors": true,
    "detect_webpack_errors": true
  }
}
```

## Supported Error Types

### Core JavaScript Errors

#### TypeError
- **Property access on undefined/null**
  ```javascript
  // Error: Cannot read property 'id' of undefined
  const id = user.id;
  
  // Auto-fix with optional chaining
  const id = user?.id;
  ```

- **Function call on non-function**
  ```javascript
  // Error: someFunction is not a function
  someFunction();
  
  // Auto-fix with type check
  if (typeof someFunction === 'function') {
    someFunction();
  }
  ```

#### ReferenceError
- **Undefined variables**
  ```javascript
  // Error: someVar is not defined
  console.log(someVar);
  
  // Auto-fix with existence check
  if (typeof someVar !== 'undefined') {
    console.log(someVar);
  }
  ```

#### SyntaxError
- **Missing brackets, semicolons**
- **Invalid syntax patterns**

### Node.js Specific Errors

#### Module Loading
```javascript
// Error: Cannot find module 'express'
const express = require('express');

// Auto-fix suggestion: npm install express
```

#### File System Operations
```javascript
// Error: ENOENT: no such file or directory
fs.readFileSync('./config.json');

// Auto-fix with existence check
if (fs.existsSync('./config.json')) {
  fs.readFileSync('./config.json');
}
```

#### Network Operations
```javascript
// Error: EADDRINUSE: address already in use
app.listen(3000);

// Auto-fix with dynamic port allocation
```

### Async/Promise Errors

#### Unhandled Promise Rejections
```javascript
// Error: Uncaught (in promise)
fetch('/api/data').then(data => data.json());

// Auto-fix with error handling
fetch('/api/data')
  .then(data => data.json())
  .catch(error => console.error('Fetch failed:', error));
```

### Transpilation Errors

#### Babel Errors
- Unsupported syntax
- Missing presets/plugins
- Configuration issues

#### TypeScript Errors
- Type mismatches (TS2322)
- Missing imports (TS2307)
- Property access (TS2339)

#### Webpack Errors
- Module resolution
- Loading errors
- Configuration issues

## Usage Examples

### Basic Error Analysis

```python
from modules.analysis.plugins.javascript_plugin import JavaScriptLanguagePlugin

plugin = JavaScriptLanguagePlugin()

# JavaScript error from runtime
error_data = {
    "name": "TypeError",
    "message": "Cannot read property 'map' of undefined",
    "stack": "TypeError: Cannot read property 'map' of undefined\\n    at processData (app.js:25:8)"
}

# Analyze the error
analysis = plugin.analyze_error(error_data)
print(f"Category: {analysis['category']}")
print(f"Confidence: {analysis['confidence']}")
print(f"Suggested fix: {analysis['suggested_fix']}")
```

### Dependency Analysis

```python
# Analyze project dependencies
project_path = "/path/to/javascript/project"
dependency_analysis = plugin.analyze_dependencies(project_path)

print(f"Dependencies: {dependency_analysis['dependencies']['count']}")
print(f"Conflicts: {len(dependency_analysis['conflicts'])}")
print(f"Missing: {len(dependency_analysis['missing_dependencies'])}")

# Get suggestions
for suggestion in dependency_analysis['suggestions']:
    print(f"- {suggestion}")
```

### Fix Generation

```python
# Generate a fix for an error
error_data = {
    "error_type": "TypeError",
    "message": "Cannot read property 'id' of undefined"
}

analysis = {
    "root_cause": "js_property_access_on_undefined",
    "confidence": "high"
}

context = {
    "error_data": error_data,
    "source_code": "const userId = user.id;"
}

fix = plugin.generate_fix(analysis, context)
if fix:
    print(f"Fix type: {fix['type']}")
    print(f"Original: {fix['original']}")
    print(f"Fixed: {fix['replacement']}")
```

### Integration with Error Monitoring

```javascript
// Browser error monitoring
window.addEventListener('error', (event) => {
  const errorData = {
    name: event.error.name,
    message: event.error.message,
    stack: event.error.stack,
    filename: event.filename,
    lineno: event.lineno,
    colno: event.colno
  };
  
  // Send to Homeostasis for analysis
  sendToHomeostasis(errorData);
});

// Unhandled promise rejection monitoring
window.addEventListener('unhandledrejection', (event) => {
  const errorData = {
    name: 'UnhandledPromiseRejection',
    message: event.reason.message || event.reason,
    stack: event.reason.stack
  };
  
  sendToHomeostasis(errorData);
});
```

```javascript
// Node.js error monitoring
process.on('uncaughtException', (error) => {
  const errorData = {
    name: error.name,
    message: error.message,
    stack: error.stack
  };
  
  sendToHomeostasis(errorData);
});

process.on('unhandledRejection', (reason, promise) => {
  const errorData = {
    name: 'UnhandledPromiseRejection',
    message: reason.message || reason,
    stack: reason.stack
  };
  
  sendToHomeostasis(errorData);
});
```

## Advanced Features

### Custom Rule Configuration

Create custom rules for specific error patterns:

```python
custom_rule = {
    "id": "custom_api_error",
    "pattern": "API request failed with status (\\d+)",
    "type": "APIError",
    "description": "Custom API error handling",
    "root_cause": "api_request_failed",
    "suggestion": "Implement retry logic or handle specific status codes",
    "category": "javascript",
    "severity": "medium",
    "confidence": "high"
}

# Add to plugin
plugin.exception_handler.rules.append(custom_rule)
```

### Framework-Specific Analysis

#### React Error Boundaries

```javascript
// React error boundary integration
class ErrorBoundary extends React.Component {
  componentDidCatch(error, errorInfo) {
    const errorData = {
      name: error.name,
      message: error.message,
      stack: error.stack,
      framework: 'react',
      componentStack: errorInfo.componentStack
    };
    
    sendToHomeostasis(errorData);
  }
}
```

#### Express.js Error Middleware

```javascript
// Express error handling middleware
app.use((error, req, res, next) => {
  const errorData = {
    name: error.name,
    message: error.message,
    stack: error.stack,
    framework: 'express',
    url: req.url,
    method: req.method
  };
  
  sendToHomeostasis(errorData);
  res.status(500).send('Internal Server Error');
});
```

## Framework Support

### React

- Component lifecycle errors
- Hook usage errors
- State management issues
- Performance optimization suggestions

### Vue.js

- Template compilation errors
- Composition API issues
- Vuex state management
- Router navigation errors

### Angular

- Dependency injection errors
- Template binding issues
- Service errors
- RxJS observable problems

### Express.js

- Route handling errors
- Middleware issues
- Database connection problems
- Security vulnerabilities

### Next.js

- Server-side rendering errors
- API route issues
- Static generation problems
- Image optimization errors

## Troubleshooting

### Common Issues

#### Plugin Not Loading

```bash
# Check if plugin is registered
python -c "from modules.analysis.language_plugin_system import get_plugin; print(get_plugin('javascript'))"
```

#### Error Not Detected

1. **Check error format**: Ensure error data includes required fields
2. **Verify patterns**: Check if error matches detection patterns
3. **Enable debug logging**: Set logging level to DEBUG

```python
import logging
logging.getLogger('modules.analysis.plugins.javascript_plugin').setLevel(logging.DEBUG)
```

#### Fix Not Generated

1. **Check source code**: Ensure source code is provided
2. **Verify error type**: Some errors only provide suggestions
3. **Check templates**: Ensure fix templates are available

### Debug Mode

Enable detailed debugging:

```python
plugin = JavaScriptLanguagePlugin()
plugin.debug = True

# Analyze with detailed output
analysis = plugin.analyze_error(error_data)
```

## API Reference

### JavaScriptLanguagePlugin

#### Methods

##### `analyze_error(error_data: Dict[str, Any]) -> Dict[str, Any]`

Analyze a JavaScript error and return analysis results.

**Parameters:**
- `error_data`: Error data in JavaScript format

**Returns:**
- Analysis results with category, confidence, and suggested fixes

##### `analyze_dependencies(project_path: str) -> Dict[str, Any]`

Analyze project dependencies for conflicts and issues.

**Parameters:**
- `project_path`: Path to the JavaScript project root

**Returns:**
- Dependency analysis results

##### `generate_fix(analysis: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]`

Generate a fix based on error analysis.

**Parameters:**
- `analysis`: Error analysis results
- `context`: Additional context including source code

**Returns:**
- Fix information or suggestion

### Error Data Format

#### Standard Format

```python
{
    "error_id": "unique-id",
    "timestamp": "2024-01-01T12:00:00Z",
    "language": "javascript",
    "error_type": "TypeError",
    "message": "Error message",
    "stack_trace": ["frame1", "frame2"],
    "severity": "high",
    "context": {
        "file": "app.js",
        "line": 25,
        "column": 8
    }
}
```

#### JavaScript-Specific Format

```python
{
    "name": "TypeError",
    "message": "Error message",
    "stack": "Stack trace string",
    "filename": "app.js",
    "lineno": 25,
    "colno": 8
}
```

## Best Practices

### Error Handling

1. **Use proper error boundaries** in React applications
2. **Implement global error handlers** for unhandled exceptions
3. **Add context information** to error reports
4. **Use structured logging** for better analysis

### Dependency Management

1. **Pin dependency versions** in production
2. **Regularly audit dependencies** for security issues
3. **Use lock files** (package-lock.json, yarn.lock)
4. **Monitor for outdated packages**

### Code Quality

1. **Use TypeScript** for better error detection
2. **Enable strict mode** in JavaScript
3. **Use linting tools** (ESLint) with Homeostasis integration
4. **Write tests** with error scenarios

### Performance

1. **Enable caching** for dependency analysis
2. **Limit file scanning** in large projects
3. **Use sampling** for high-volume error monitoring
4. **Configure appropriate timeouts**

## Contributing

To contribute to the JavaScript plugin:

1. **Fork the repository**
2. **Create feature branch**
3. **Add tests** for new functionality
4. **Update documentation**
5. **Submit pull request**

### Adding New Error Patterns

1. Add to `js_common_errors.json` or `nodejs_errors.json`
2. Update `JavaScriptExceptionHandler`
3. Add corresponding tests
4. Update documentation

### Adding Framework Support

1. Create framework-specific rule files
2. Add detection patterns
3. Implement fix templates
4. Add integration examples

For more information, see the [Contributing Guide](../CONTRIBUTING.md).