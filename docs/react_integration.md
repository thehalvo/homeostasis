# React Framework Integration

This document describes how to use Homeostasis with React applications for automatic error detection, analysis, and healing.

## Overview

The React plugin for Homeostasis provides error handling for React applications, including:

- **React Hooks** error detection and rule validation
- **Component lifecycle** error monitoring
- **State management** healing (React state, Redux, Context API)
- **JSX and React patterns** error detection
- **Performance optimization** suggestions
- **Server Components** support (React 18+)
- **Memory leak prevention** and cleanup suggestions

## Supported React Features

### React Hooks
- useState, useEffect, useContext, useReducer
- useCallback, useMemo, useRef, useImperativeHandle
- useLayoutEffect, useDebugValue, useDeferredValue
- useTransition, useId, useSyncExternalStore, useInsertionEffect
- Custom hooks validation

### Component Patterns
- Function components and class components
- Error boundaries
- Higher-order components (HOCs)
- Render props pattern
- Compound components

### State Management
- React built-in state (useState, useReducer)
- Redux and Redux Toolkit
- React Context API
- External state libraries

### Modern React Features
- Server Components (React 18+)
- Suspense and concurrent features
- Automatic batching
- Transitions and deferred values
- Streaming SSR

## Installation and Setup

### Prerequisites
- Python 3.8+
- React 16.8+ (for hooks support)
- Node.js and npm/yarn (for React projects)

### Basic Setup

1. **Install Homeostasis** in your React project directory:
```bash
pip install homeostasis
```

2. **Configure Homeostasis** for React by creating a configuration file:
```python
# homeostasis_config.py
HOMEOSTASIS_CONFIG = {
    "language": "react",
    "framework": "react",
    "rules_directories": ["./rules"],
    "auto_healing": True,
    "monitoring": {
        "enabled": True,
        "hooks_validation": True,
        "performance_monitoring": True
    }
}
```

3. **Initialize Homeostasis** in your React application:
```javascript
// For development monitoring
import { initHomeostasis } from 'homeostasis-js';

if (process.env.NODE_ENV === 'development') {
  initHomeostasis({
    apiEndpoint: 'http://localhost:8000/api/errors',
    framework: 'react',
    captureHooksErrors: true,
    capturePerformanceIssues: true
  });
}
```

## Error Detection Categories

### 1. React Hooks Errors

**Invalid Hook Calls**
```javascript
// ❌ Problematic code
function MyComponent() {
  if (condition) {
    const [state, setState] = useState(0); // Hook called conditionally
  }
  return <div>Content</div>;
}

// ✅ Homeostasis suggested fix
function MyComponent() {
  const [state, setState] = useState(0); // Hook at top level
  
  if (condition) {
    // Use state conditionally
  }
  return <div>Content</div>;
}
```

**Missing Dependencies**
```javascript
// ❌ Problematic code
function Counter() {
  const [count, setCount] = useState(0);
  
  useEffect(() => {
    document.title = `Count: ${count}`;
  }, []); // Missing 'count' dependency
  
  return <div>{count}</div>;
}

// ✅ Homeostasis suggested fix
function Counter() {
  const [count, setCount] = useState(0);
  
  useEffect(() => {
    document.title = `Count: ${count}`;
  }, [count]); // Added missing dependency
  
  return <div>{count}</div>;
}
```

### 2. JSX and Component Errors

**Missing Key Props**
```javascript
// ❌ Problematic code
function ItemList({ items }) {
  return (
    <ul>
      {items.map(item => (
        <li>{item.name}</li> // Missing key prop
      ))}
    </ul>
  );
}

// ✅ Homeostasis suggested fix
function ItemList({ items }) {
  return (
    <ul>
      {items.map(item => (
        <li key={item.id}>{item.name}</li> // Added unique key
      ))}
    </ul>
  );
}
```

**React Scope Issues**
```javascript
// ❌ Problematic code (React 17 and below)
function App() {
  return <div>Hello World</div>; // React not in scope
}

// ✅ Homeostasis suggested fix
import React from 'react';

function App() {
  return <div>Hello World</div>;
}
```

### 3. State Management Errors

**Redux Store Issues**
```javascript
// ❌ Problematic code
function App() {
  return <MyComponent />; // Redux store not provided
}

// ✅ Homeostasis suggested fix
import { Provider } from 'react-redux';
import store from './store';

function App() {
  return (
    <Provider store={store}>
      <MyComponent />
    </Provider>
  );
}
```

**Context Provider Missing**
```javascript
// ❌ Problematic code
function MyComponent() {
  const theme = useContext(ThemeContext); // Provider missing
  return <div>Theme: {theme}</div>;
}

// ✅ Homeostasis suggested fix
function App() {
  return (
    <ThemeContext.Provider value={themeValue}>
      <MyComponent />
    </ThemeContext.Provider>
  );
}
```

### 4. Performance Issues

**Unnecessary Re-renders**
```javascript
// ❌ Problematic code
function ExpensiveComponent({ data }) {
  const expensiveValue = heavyCalculation(data); // Runs on every render
  return <div>{expensiveValue}</div>;
}

// ✅ Homeostasis suggested fix
function ExpensiveComponent({ data }) {
  const expensiveValue = useMemo(() => heavyCalculation(data), [data]);
  return <div>{expensiveValue}</div>;
}
```

**Memory Leaks**
```javascript
// ❌ Problematic code
function Timer() {
  useEffect(() => {
    const timer = setInterval(() => {
      console.log('Timer tick');
    }, 1000);
    // Missing cleanup
  }, []);
  
  return <div>Timer Component</div>;
}

// ✅ Homeostasis suggested fix
function Timer() {
  useEffect(() => {
    const timer = setInterval(() => {
      console.log('Timer tick');
    }, 1000);
    
    return () => clearInterval(timer); // Cleanup added
  }, []);
  
  return <div>Timer Component</div>;
}
```

### 5. Server Components Errors (React 18+)

**Client Code in Server Components**
```javascript
// ❌ Problematic code
function ServerComponent() {
  const [state, setState] = useState(0); // useState not allowed in Server Components
  return <div>{state}</div>;
}

// ✅ Homeostasis suggested fix
'use client'; // Mark as Client Component

function ClientComponent() {
  const [state, setState] = useState(0);
  return <div>{state}</div>;
}
```

## Configuration Options

### React-Specific Settings

```python
# homeostasis_config.py
REACT_CONFIG = {
    "hooks_validation": {
        "enforce_rules_of_hooks": True,
        "check_exhaustive_deps": True,
        "detect_stale_closures": True
    },
    "performance_monitoring": {
        "detect_unnecessary_rerenders": True,
        "suggest_memoization": True,
        "analyze_bundle_size": True,
        "check_list_virtualization": True
    },
    "jsx_validation": {
        "require_key_props": True,
        "validate_jsx_syntax": True,
        "check_component_exports": True
    },
    "state_management": {
        "redux_integration": True,
        "context_validation": True,
        "immutability_checks": True
    },
    "server_components": {
        "validate_server_client_boundary": True,
        "check_serialization": True,
        "detect_hydration_mismatches": True
    }
}
```

### Error Severity Levels

```python
REACT_SEVERITY_CONFIG = {
    "hooks_violations": "error",      # Rules of Hooks violations
    "missing_keys": "warning",        # Missing key props
    "performance_issues": "warning",   # Performance suggestions
    "memory_leaks": "error",          # Memory leak detection
    "server_component_violations": "error",  # Server/Client boundary issues
    "prop_type_mismatches": "warning", # PropTypes validation
    "jsx_syntax_errors": "error"      # JSX syntax issues
}
```

## Integration with Build Tools

### Webpack Integration

```javascript
// webpack.config.js
const HomeostasisWebpackPlugin = require('homeostasis-webpack-plugin');

module.exports = {
  plugins: [
    new HomeostasisWebpackPlugin({
      framework: 'react',
      enableHooksValidation: true,
      enablePerformanceMonitoring: true,
      apiEndpoint: process.env.HOMEOSTASIS_API_ENDPOINT
    })
  ]
};
```

### Vite Integration

```javascript
// vite.config.js
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import homeostasis from 'homeostasis-vite-plugin';

export default defineConfig({
  plugins: [
    react(),
    homeostasis({
      framework: 'react',
      enableHooksValidation: true,
      enablePerformanceMonitoring: true
    })
  ]
});
```

### Next.js Integration

```javascript
// next.config.js
const withHomeostasis = require('homeostasis-next-plugin');

module.exports = withHomeostasis({
  homeostasis: {
    framework: 'react',
    serverComponents: true,
    enableSSRErrorTracking: true,
    enableHooksValidation: true
  }
});
```

## Error Categories and Fixes

### Hooks-Related Errors

| Error Type | Description | Auto-Fix Available |
|------------|-------------|--------------------|
| Invalid Hook Call | Hooks called outside function components | ✅ Suggestion |
| Conditional Hook Call | Hooks called inside conditions/loops | ✅ Code restructure |
| Missing Dependencies | useEffect/useCallback missing deps | ✅ Dependency addition |
| Stale Closures | State not updated in callbacks | ✅ Functional updates |

### Performance Errors

| Error Type | Description | Auto-Fix Available |
|------------|-------------|--------------------|
| Unnecessary Re-renders | Components re-rendering too often | ✅ Memoization |
| Expensive Calculations | Heavy computations on every render | ✅ useMemo |
| Large Lists | Rendering many items without virtualization | ✅ Virtualization |
| Memory Leaks | Missing cleanup in useEffect | ✅ Cleanup functions |

### State Management Errors

| Error Type | Description | Auto-Fix Available |
|------------|-------------|--------------------|
| Redux Store Missing | Components not wrapped with Provider | ✅ Provider setup |
| Context Provider Missing | useContext without Provider | ✅ Provider addition |
| State Mutations | Direct state mutations detected | ✅ Immutable updates |
| Action Not Dispatched | Actions called without dispatch | ✅ useDispatch usage |

## Best Practices

### 1. Hooks Usage
- Always call hooks at the top level of components
- Include all dependencies in useEffect, useCallback, useMemo
- Use cleanup functions for subscriptions and timers
- Prefer functional state updates when state depends on previous state

### 2. Performance Optimization
- Use React.memo for pure components
- Implement useMemo for expensive calculations
- Use useCallback for function props to prevent child re-renders
- Consider virtualization for large lists

### 3. State Management
- Keep state close to where it's used
- Use Context for truly global state
- Prefer useReducer for complex state logic
- Avoid deep nesting in state objects

### 4. Server Components (React 18+)
- Mark client-only components with 'use client'
- Only pass serializable props from Server to Client Components
- Use Server Actions for server-side functions
- Implement proper Suspense boundaries

## Advanced Features

### Custom Rules
Create custom React rules for your specific patterns:

```python
# custom_react_rules.py
CUSTOM_REACT_RULES = [
    {
        "id": "custom_hook_pattern",
        "pattern": r"use\w+Hook",
        "message": "Custom hook should follow naming convention",
        "severity": "warning",
        "fix_suggestion": "Rename to follow useXxx pattern"
    }
]
```

### Error Boundaries Integration
Homeostasis can integrate with React Error Boundaries:

```javascript
class HomeostasisErrorBoundary extends React.Component {
  componentDidCatch(error, errorInfo) {
    // Send to Homeostasis for analysis
    window.homeostasis?.captureError(error, {
      componentStack: errorInfo.componentStack,
      framework: 'react'
    });
  }
  
  render() {
    if (this.state.hasError) {
      return <h1>Something went wrong.</h1>;
    }
    return this.props.children;
  }
}
```

## Troubleshooting

### Common Issues

1. **Plugin not detecting React errors**
   - Ensure React framework is specified in configuration
   - Check that error reports include React-specific information

2. **False positives in hooks validation**
   - Update to latest React version
   - Check ESLint React Hooks plugin compatibility

3. **Performance monitoring overhead**
   - Disable in production builds
   - Configure sampling rates for large applications

### Debug Mode

Enable debug mode for detailed analysis:

```javascript
// Development only
if (process.env.NODE_ENV === 'development') {
  window.homeostasis?.enableDebugMode({
    logHooksAnalysis: true,
    logPerformanceMetrics: true,
    logStateChanges: true
  });
}
```

## Migration Guide

### From Class Components
When migrating from class components to hooks, Homeostasis can help identify patterns and suggest modern equivalents.

### React 18 Upgrade
Homeostasis includes specific guidance for React 18 features like concurrent rendering and Server Components.

## API Reference

### React Plugin Methods

```python
from homeostasis.plugins import ReactPlugin

plugin = ReactPlugin()

# Analyze React-specific error
analysis = plugin.analyze_error(error_data)

# Generate React-specific fix
fix = plugin.generate_fix(error_data, analysis, source_code)

# Validate hooks usage
hooks_validation = plugin.validate_hooks(component_code)
```

## Examples

See the `/examples/react/` directory for complete example projects demonstrating:

- Basic React app with Homeostasis integration
- Redux application with state management healing
- Next.js app with Server Components
- Performance optimization examples
- Custom rules and configurations

## Contributing

To contribute React-specific rules or improvements:

1. Follow the plugin development guide in `/docs/plugin_development.md`
2. Add test cases for new React patterns
3. Update documentation with new features
4. Submit pull requests with clear descriptions

## Support

For React-specific issues:
- Check the FAQ section
- Review existing GitHub issues
- Create new issues with React error examples
- Join the community discussions

The React integration in Homeostasis provides error detection and healing capabilities, helping you build more robust React applications with automatic error resolution and performance optimization.