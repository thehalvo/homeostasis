# TypeScript Integration for Homeostasis

This document provides comprehensive information about the TypeScript language plugin for Homeostasis, enabling automatic error detection, analysis, and fix generation for TypeScript applications.

## Overview

The TypeScript plugin extends Homeostasis to support:
- TypeScript compilation errors (TS2xxx codes)
- Type system errors and type checking
- Advanced TypeScript features (generics, decorators, mapped types)
- TSX/JSX support for React and other frameworks
- TypeScript configuration validation
- Runtime error handling in TypeScript applications
- Integration with TypeScript tooling and build systems

## Features

### Core Capabilities

1. **Type Checking Error Resolution**
   - Detects and fixes common TypeScript type errors
   - Handles type assignment mismatches
   - Resolves missing type definitions
   - Provides suggestions for type compatibility issues

2. **Compilation Error Detection**
   - Syntax error analysis and fixes
   - TypeScript configuration validation
   - Compiler option error resolution
   - Build process error handling

3. **Advanced TypeScript Features**
   - Generic type constraint errors
   - Conditional type issues
   - Mapped type problems
   - Template literal type errors
   - Utility type misuse detection

4. **Module Resolution**
   - Missing import detection
   - Module not found error resolution
   - Type declaration file issues
   - Path mapping configuration problems

5. **Framework Integration**
   - React/JSX error handling
   - Angular-specific TypeScript issues
   - Vue TypeScript support
   - Node.js TypeScript integration

## Supported Error Types

### Type System Errors

| Error Code | Description | Fix Strategy |
|------------|-------------|--------------|
| TS2304 | Cannot find name | Import resolution, type declaration |
| TS2322 | Type not assignable | Type assertion, interface extension |
| TS2339 | Property does not exist | Interface extension, optional chaining |
| TS2307 | Cannot find module | Package installation, import path fixes |
| TS2345 | Argument type mismatch | Type conversion, function overloads |
| TS2540 | Read-only property | Immutable update patterns |
| TS2571 | Object is of type 'unknown' | Type guards, type assertions |

### Compilation Errors

| Error Code | Description | Fix Strategy |
|------------|-------------|--------------|
| TS1005 | Token expected | Syntax correction |
| TS1127 | Invalid character | Character encoding fixes |
| TS5023 | Unknown compiler option | Configuration validation |
| TS5024 | Option requires value | Configuration completion |
| TS6133 | Unused variable | Code cleanup suggestions |

### Advanced Features

| Error Code | Description | Fix Strategy |
|------------|-------------|--------------|
| TS2344 | Type constraint violation | Generic constraint fixes |
| TS2589 | Type instantiation too deep | Type simplification |
| TS2590 | Union type too complex | Type structure optimization |
| TS1238 | Decorator signature error | Decorator configuration |

## Installation and Configuration

### Prerequisites

- Node.js 14.0 or higher
- TypeScript 3.0 or higher
- Python 3.8+ (for Homeostasis core)

### Setup

1. **Install TypeScript and Dependencies**
   ```bash
   npm install -g typescript
   npm install --save-dev @types/node
   ```

2. **Configure TypeScript Project**
   Create or update `tsconfig.json`:
   ```json
   {
     "compilerOptions": {
       "target": "es2020",
       "module": "commonjs",
       "strict": true,
       "esModuleInterop": true,
       "skipLibCheck": true,
       "forceConsistentCasingInFileNames": true
     },
     "include": ["src/**/*"],
     "exclude": ["node_modules", "dist"]
   }
   ```

3. **Enable Homeostasis TypeScript Plugin**
   The TypeScript plugin is automatically loaded when Homeostasis starts. No additional configuration is required.

## Usage Examples

### Basic Error Analysis

```python
from homeostasis import analyze_error

# TypeScript compilation error
error_data = {
    "code": "TS2304",
    "message": "Cannot find name 'React'.",
    "file": "src/App.tsx",
    "line": 1,
    "column": 8
}

analysis = analyze_error(error_data, language="typescript")
print(analysis["suggested_fix"])
# Output: "Add import statement: import React from 'react';"
```

### Error Detection from Build Output

```python
# Parse TypeScript compiler output
tsc_output = """
src/utils.ts(15,8): error TS2339: Property 'length' does not exist on type 'string | undefined'.
src/components/App.tsx(23,5): error TS2322: Type 'number' is not assignable to type 'string'.
"""

errors = parse_typescript_output(tsc_output)
for error in errors:
    analysis = analyze_error(error, language="typescript")
    fix = generate_fix(error, analysis)
    print(f"Error: {error['message']}")
    print(f"Fix: {fix['description']}")
```

### Framework-Specific Error Handling

```python
# React TypeScript error
react_error = {
    "code": "TS2786",
    "message": "'MyComponent' cannot be used as a JSX component.",
    "file": "src/components/MyComponent.tsx",
    "framework": "react"
}

analysis = analyze_error(react_error, language="typescript")
# Analysis includes React-specific fix suggestions
```

## Integration with Development Tools

### VS Code Integration

The TypeScript plugin can integrate with VS Code through the Language Server Protocol:

```json
{
  "typescript.preferences.includePackageJsonAutoImports": "auto",
  "typescript.suggest.autoImports": true,
  "typescript.updateImportsOnFileMove.enabled": "always"
}
```

### Build System Integration

#### Webpack Integration

```javascript
// webpack.config.js
module.exports = {
  module: {
    rules: [
      {
        test: /\.tsx?$/,
        use: [
          {
            loader: 'ts-loader',
            options: {
              // Enable Homeostasis error reporting
              errorFormatter: (error) => {
                return homeostasis.formatError(error);
              }
            }
          }
        ]
      }
    ]
  }
};
```

#### ESBuild Integration

```javascript
// esbuild.config.js
require('esbuild').build({
  entryPoints: ['src/index.ts'],
  bundle: true,
  outfile: 'dist/index.js',
  plugins: [
    {
      name: 'homeostasis-typescript',
      setup(build) {
        build.onEnd((result) => {
          if (result.errors.length > 0) {
            homeostasis.analyzeErrors(result.errors);
          }
        });
      }
    }
  ]
});
```

## Advanced Configuration

### Custom Type Definitions

Create custom type definitions for better error detection:

```typescript
// types/homeostasis.d.ts
declare namespace Homeostasis {
  interface ErrorData {
    code: string;
    message: string;
    file: string;
    line: number;
    column: number;
  }
  
  interface AnalysisResult {
    category: string;
    confidence: string;
    suggested_fix: string;
  }
}
```

### Plugin Configuration

Configure the TypeScript plugin behavior:

```python
from homeostasis.config import TypeScriptConfig

# Configure TypeScript-specific settings
config = TypeScriptConfig({
    "strict_mode": True,
    "enable_jsx": True,
    "target_version": "es2020",
    "module_resolution": "node",
    "enable_decorators": True
})

homeostasis.configure_plugin("typescript", config)
```

## Error Fix Templates

The TypeScript plugin includes templates for common fix patterns:

### Type Assertion Template

```typescript
// Before: Type 'unknown' error
const data = getValue(); // Type: unknown
const length = data.length; // Error: Property 'length' does not exist

// After: Type assertion
const data = getValue() as string;
const length = data.length; // Fixed
```

### Null Check Template

```typescript
// Before: Possible null reference
const result = obj.property.method(); // Error: Object is possibly null

// After: Optional chaining
const result = obj?.property?.method?.(); // Fixed
```

### Import Fix Template

```typescript
// Before: Cannot find name
const element = <div>Hello</div>; // Error: Cannot find name 'React'

// After: Import added
import React from 'react';
const element = <div>Hello</div>; // Fixed
```

## Troubleshooting

### Common Issues

1. **Plugin Not Loading**
   - Ensure TypeScript plugin file is in the correct directory
   - Check Python path includes Homeostasis modules
   - Verify no import errors in plugin dependencies

2. **Error Detection Not Working**
   - Check TypeScript version compatibility
   - Verify tsconfig.json is properly configured
   - Ensure error data format matches expected schema

3. **Fix Generation Failing**
   - Check source code accessibility
   - Verify file permissions for reading/writing
   - Ensure TypeScript AST parsing is working

### Debug Mode

Enable debug logging for detailed plugin information:

```python
import logging
logging.getLogger('homeostasis.typescript').setLevel(logging.DEBUG)
```

### Performance Tuning

For large TypeScript projects:

```python
# Configure caching and performance options
config = {
    "cache_analysis_results": True,
    "cache_ttl": 3600,  # 1 hour
    "parallel_analysis": True,
    "max_file_size": "10MB"
}
```

## API Reference

### TypeScriptLanguagePlugin

Main plugin class providing TypeScript support.

#### Methods

- `analyze_error(error_data)`: Analyze TypeScript error
- `generate_fix(error_data, analysis, source_code)`: Generate fix for error
- `can_handle(error_data)`: Check if plugin can handle error
- `get_language_info()`: Get plugin information

### TypeScriptErrorAdapter

Handles conversion between TypeScript and standard error formats.

#### Methods

- `to_standard_format(error_data)`: Convert to standard format
- `from_standard_format(standard_error)`: Convert from standard format

### TypeScriptExceptionHandler

Analyzes TypeScript exceptions and provides categorization.

#### Methods

- `analyze_exception(error_data)`: Analyze TypeScript exception
- `analyze_compilation_error(error_data)`: Analyze compilation error
- `analyze_type_error(error_data)`: Analyze type system error

## Best Practices

### Code Organization

1. **Type Safety**
   - Use strict TypeScript configuration
   - Enable all strict mode flags
   - Avoid `any` type usage

2. **Error Handling**
   - Implement proper error boundaries
   - Use type guards for runtime safety
   - Handle async operations properly

3. **Module Organization**
   - Use barrel exports for clean imports
   - Organize types in separate files
   - Follow consistent naming conventions

### Integration Workflow

1. **Development Phase**
   - Enable real-time error detection
   - Use IDE integration for immediate feedback
   - Configure pre-commit hooks for validation

2. **Build Phase**
   - Integrate with build system
   - Generate comprehensive error reports
   - Apply automatic fixes where safe

3. **Deployment Phase**
   - Monitor runtime TypeScript errors
   - Track fix effectiveness metrics
   - Continuous improvement of error patterns

## Contributing

### Adding New Error Types

1. **Update Rules Files**
   Add new error patterns to appropriate rule files in `modules/analysis/rules/typescript/`

2. **Extend Exception Handler**
   Add specific analysis logic for new error types

3. **Create Fix Templates**
   Add templates for common fix patterns

4. **Add Test Cases**
   Include comprehensive tests for new functionality

### Testing Guidelines

Run the TypeScript plugin tests:

```bash
python -m pytest tests/test_typescript_plugin.py -v
```

### Documentation Updates

Update this documentation when adding new features or changing behavior.

## Roadmap

### Planned Features

1. **Enhanced Framework Support**
   - Improved React hooks analysis
   - Angular dependency injection errors
   - Vue 3 composition API support

2. **Advanced Type Analysis**
   - Complex generic type inference
   - Template literal type optimization
   - Conditional type debugging

3. **IDE Integration**
   - VS Code extension
   - JetBrains plugin
   - Vim/Neovim integration

4. **Performance Improvements**
   - Incremental type checking
   - Cached analysis results
   - Parallel error processing

## Support

For issues, questions, or contributions:

- GitHub Issues: [homeostasis/issues](https://github.com/homeostasis/homeostasis/issues)
- Documentation: [homeostasis.dev/docs](https://homeostasis.dev/docs)
- Community: [discord.gg/homeostasis](https://discord.gg/homeostasis)

## License

This TypeScript integration is part of the Homeostasis project and is licensed under the same terms as the main project.