# JavaScript/TypeScript Integration in Homeostasis

This document describes the JavaScript and TypeScript support in Homeostasis, outlining its features, capabilities, implementation details, and usage examples.

## Overview

Homeostasis provides support for JavaScript and TypeScript applications, enabling error detection, analysis, and healing capabilities. The integration focuses primarily on Node.js environments but includes basic support for browser-based applications as well.

## Features

- **JavaScript Error Detection**: Identifies common JavaScript runtime and syntax errors
- **TypeScript Support**: Handles TypeScript-specific issues and type errors
- **Node.js Integration**: Specialized detection for server-side JavaScript environments
- **Framework Support**: Basic support for Express, Nest.js, and other Node.js frameworks
- **Normalized Error Format**: Converts JavaScript-specific errors to Homeostasis standard format

## Supported Error Types

The JavaScript/TypeScript integration can detect and analyze various error types including:

### JavaScript Core Errors
- Reference errors (undefined variables)
- Type errors
- Syntax errors
- Range errors
- URI errors
- Evaluation errors

### Node.js Specific Errors
- File system errors
- Network errors
- Stream errors
- Buffer errors
- Module loading errors
- Process and child process errors

### Framework-Specific Errors
- Express middleware and routing errors
- Database connection issues
- API endpoint errors
- Authentication failures
- JSON parsing errors

## Architecture

The JavaScript/TypeScript integration consists of several key components:

1. **JavaScriptErrorAdapter**: Normalizes JavaScript errors to/from the standard format
2. **JavaScriptAnalyzer**: Analyzes JavaScript errors using pattern matching and rules
3. **Error Detection System**: Uses stack trace analysis to identify error patterns
4. **Error Classification**: Categorizes errors based on type and context

### Component Interaction

```
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│                 │      │                 │      │                 │
│  Node.js App    ├──────►   JavaScript    ├──────►  JavaScript     │
│                 │      │   ErrorAdapter  │      │   Analyzer      │
└─────────────────┘      └─────────────────┘      └─────────────────┘
                                 │                        │
                                 │                        │
                                 ▼                        ▼
                         ┌─────────────────┐      ┌─────────────────┐
                         │                 │      │                 │
                         │Cross-Language   │◄─────┤ Rule-Based      │
                         │  Orchestrator   │      │   Detection     │
                         └─────────────────┘      └─────────────────┘
```

## Implementation Details

### Error Format Normalization

JavaScript errors are normalized to the standard Homeostasis format:

```javascript
// Original JavaScript error
{
  "name": "TypeError",
  "message": "Cannot read property 'id' of undefined",
  "stack": "TypeError: Cannot read property 'id' of undefined\n    at getUser (/app/src/utils.js:45:20)"
}

// Normalized format
{
  "error_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2023-08-15T12:34:56Z",
  "language": "javascript",
  "error_type": "TypeError",
  "message": "Cannot read property 'id' of undefined",
  "stack_trace": [
    {
      "function": "getUser",
      "file": "/app/src/utils.js",
      "line": 45,
      "column": 20
    }
  ],
  "severity": "error"
}
```

### Stack Trace Analysis

The JavaScript integration includes sophisticated stack trace parsing to extract meaningful information:

- Function names and contexts
- File paths with line and column information
- Error propagation patterns
- Asynchronous code stack spanning

### Error Patterns

Common JavaScript error patterns are detected through pattern matching:

- "Cannot read property 'X' of undefined" → Null/undefined object access
- "X is not a function" → Invalid function call
- "Assignment to constant variable" → Const reassignment
- "Cannot set property 'X' of null" → Null object property assignment

## Example Usage

### Null Check Error

```javascript
// Original code with an error
function getUserName(user) {
  return user.profile.name;  // Error if user or user.profile is undefined
}

// Homeostasis generated fix
function getUserName(user) {
  if (!user || !user.profile) {
    return null;  // Or a default value, or throw a specific error
  }
  return user.profile.name;
}
```

### Async/Await Error

```javascript
// Original code with an error
async function fetchData() {
  const response = await fetch('/api/data');
  const data = response.json();  // Missing await
  return data;
}

// Homeostasis generated fix
async function fetchData() {
  const response = await fetch('/api/data');
  const data = await response.json();  // Added missing await
  return data;
}
```

## Framework-Specific Support

### Express.js

- Middleware error detection
- Route parameter validation
- Request and response handling errors
- Express-specific error patterns

### TypeScript Integration

- Type inference issues
- Interface compliance errors
- Generic type resolution
- Type assertion failures

## Configuration

JavaScript/TypeScript integration can be configured in the Homeostasis `config.yaml`:

```yaml
analysis:
  languages:
    javascript:
      enabled: true
      min_confidence: 0.7
      environments:
        - node
        - browser
      frameworks:
        - express
        - nestjs
    typescript:
      enabled: true
      transpilation_check: true
```

## Current Limitations

- Limited support for client-side framework errors (React, Angular, Vue)
- Partial TypeScript type checking capabilities
- Basic fix generation (compared to Python integration)
- Limited template library for JavaScript-specific fixes

## Future Work

- Enhanced support for modern JavaScript frameworks (React, Vue, Angular)
- Improved TypeScript type checking and error analysis
- Frontend-specific error detection and healing
- Browser runtime error support
- JavaScript bundler and transpiler integration

## Conclusion

The JavaScript/TypeScript support in Homeostasis provides organizations with the ability to detect and fix common JavaScript errors. While currently not as comprehensive as Python support, it offers valuable capabilities for Node.js applications and is continuously being improved.