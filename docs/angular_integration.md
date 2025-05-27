# Angular Integration Guide

## Overview

The Homeostasis Angular plugin provides comprehensive error detection and automatic healing for Angular applications. It supports Angular 2+ and covers dependency injection errors, NgRx state management issues, template binding problems, module configuration errors, and Angular Universal SSR issues.

## Features

### Core Angular Support
- **Dependency Injection**: Automatic detection and resolution of DI errors
- **Component Lifecycle**: Error handling for component lifecycle methods
- **Template Binding**: Safe navigation and property binding fixes
- **Module Management**: Lazy loading and module configuration issues
- **Router Integration**: Navigation and route configuration errors

### NgRx State Management
- **Store Configuration**: Automatic StoreModule setup
- **Action Validation**: Ensures actions have required properties
- **Reducer Safety**: Prevents undefined state returns
- **Effect Error Handling**: Proper error handling in effects
- **Selector Optimization**: Default value handling in selectors

### Angular Universal SSR
- **Platform Detection**: Browser vs server environment checks
- **DOM Manipulation**: Safe DOM access patterns
- **Transfer State**: Data synchronization between server and client
- **External Libraries**: Browser-only library loading
- **Meta Tag Management**: SEO-friendly meta tag updates

## Installation

The Angular plugin is automatically registered when the Homeostasis system starts. No additional installation is required.

## Configuration

### Basic Configuration

Add Angular configuration to your `homeostasis.yml`:

```yaml
analyzer:
  plugins:
    angular:
      enabled: true
      version: "2+"
      frameworks:
        - angular
        - ionic
        - nativescript
      file_extensions:
        - .ts
        - .js
        - .html
      config_files:
        - angular.json
        - package.json
        - tsconfig.json
```

### Advanced Configuration

```yaml
analyzer:
  plugins:
    angular:
      enabled: true
      dependency_injection:
        auto_provide: true
        use_provided_in_root: true
      ngrx:
        strict_mode: true
        enable_devtools: true
      templates:
        safe_navigation: true
        strict_property_binding: true
      ssr:
        platform_checks: true
        transfer_state: true
```

## Supported Error Types

### 1. Dependency Injection Errors

#### No Provider Error
```typescript
// Error: No provider for UserService!
// Auto-fix: Add provider configuration

@Injectable({
  providedIn: 'root'  // Added automatically
})
export class UserService {
  // Service implementation
}
```

#### Circular Dependency
```typescript
// Error: Circular dependency detected
// Auto-fix: Use forwardRef()

@Injectable()
export class ServiceA {
  constructor(
    @Inject(forwardRef(() => ServiceB)) private serviceB: ServiceB
  ) {}
}
```

### 2. NgRx State Management Errors

#### Missing Store Configuration
```typescript
// Error: Store has not been provided
// Auto-fix: Add StoreModule configuration

@NgModule({
  imports: [
    StoreModule.forRoot(reducers, {
      runtimeChecks: {
        strictStateImmutability: true,
        strictActionImmutability: true
      }
    })
  ]
})
export class AppModule {}
```

#### Action Type Missing
```typescript
// Error: Action must have a type
// Auto-fix: Use createAction helper

export const loadUsers = createAction(
  '[User] Load Users'
);

export const loadUsersSuccess = createAction(
  '[User] Load Users Success',
  props<{ users: User[] }>()
);
```

### 3. Template Binding Errors

#### Safe Navigation
```html
<!-- Error: Cannot read property 'name' of undefined -->
<!-- Auto-fix: Add safe navigation operator -->

<div>{{ user?.name }}</div>
<div>{{ user?.address?.street }}</div>

<!-- Alternative fix: Use *ngIf -->
<div *ngIf="user">
  <h3>{{ user.name }}</h3>
  <p>{{ user.email }}</p>
</div>
```

#### Property Binding
```typescript
// Error: Can't bind to 'customProperty'
// Auto-fix: Add @Input() decorator

@Component({
  selector: 'app-child',
  template: '...'
})
export class ChildComponent {
  @Input() customProperty: string;  // Added automatically
}
```

### 4. Angular Universal SSR Errors

#### Browser-Only Code
```typescript
// Error: window is not defined
// Auto-fix: Add platform check

import { isPlatformBrowser } from '@angular/common';
import { PLATFORM_ID, Inject } from '@angular/core';

export class Component {
  constructor(@Inject(PLATFORM_ID) private platformId: Object) {}
  
  someMethod() {
    if (isPlatformBrowser(this.platformId)) {
      // Browser-only code
      const width = window.innerWidth;
    }
  }
}
```

## Integration with Build Tools

### Angular CLI

Add Homeostasis to your Angular CLI workflow:

```json
{
  "scripts": {
    "build": "ng build && homeostasis analyze",
    "test": "ng test && homeostasis test",
    "heal": "homeostasis heal --framework angular"
  }
}
```

### Webpack Configuration

```javascript
// webpack.config.js
const HomeostasisPlugin = require('homeostasis-webpack-plugin');

module.exports = {
  plugins: [
    new HomeostasisPlugin({
      framework: 'angular',
      autoFix: true,
      patterns: ['src/**/*.ts', 'src/**/*.html']
    })
  ]
};
```

### IDE Integration

#### Visual Studio Code

Install the Homeostasis extension for real-time error detection:

```json
{
  "homeostasis.angular": {
    "enabled": true,
    "realTimeAnalysis": true,
    "autoFix": true,
    "showSuggestions": true
  }
}
```

#### WebStorm/IntelliJ

Enable Homeostasis plugin in IDE settings:

1. Go to Settings > Plugins
2. Install Homeostasis plugin
3. Configure Angular-specific settings

## Best Practices

### 1. Dependency Injection

- Use `providedIn: 'root'` for singleton services
- Avoid circular dependencies
- Use injection tokens for configuration
- Implement proper service hierarchies

### 2. NgRx State Management

- Always define action types using `createAction`
- Ensure reducers never return undefined
- Use proper error handling in effects
- Implement proper state serialization

### 3. Template Development

- Use safe navigation operator for nested properties
- Implement proper null checks
- Use `*ngIf` for conditional rendering
- Avoid direct DOM manipulation

### 4. SSR Considerations

- Check platform before using browser APIs
- Use TransferState for data synchronization
- Implement proper meta tag management
- Handle external library loading properly

## Troubleshooting

### Common Issues

#### Plugin Not Loading
```bash
# Check plugin registration
homeostasis plugins list

# Verify Angular detection
homeostasis analyze --debug --framework angular
```

#### False Positives
```yaml
# Configure exclusions
analyzer:
  plugins:
    angular:
      exclude_patterns:
        - "**/*.spec.ts"
        - "**/test/**"
        - "**/mock/**"
```

#### Performance Issues
```yaml
# Optimize analysis
analyzer:
  plugins:
    angular:
      max_file_size: "1MB"
      parallel_analysis: true
      cache_enabled: true
```

### Debug Mode

Enable debug mode for detailed logging:

```bash
homeostasis analyze --debug --framework angular --verbose
```

## Examples

### Basic Angular Application

```typescript
// app.component.ts
import { Component, OnInit } from '@angular/core';
import { UserService } from './user.service';

@Component({
  selector: 'app-root',
  template: `
    <div *ngIf="user">
      <h1>{{ user?.name }}</h1>
      <p>{{ user?.email }}</p>
    </div>
  `
})
export class AppComponent implements OnInit {
  user: User | null = null;
  
  constructor(private userService: UserService) {}
  
  ngOnInit() {
    this.userService.getCurrentUser()
      .subscribe(user => this.user = user);
  }
}
```

### NgRx Integration

```typescript
// user.actions.ts
import { createAction, props } from '@ngrx/store';

export const loadUser = createAction('[User] Load User');
export const loadUserSuccess = createAction(
  '[User] Load User Success',
  props<{ user: User }>()
);
export const loadUserFailure = createAction(
  '[User] Load User Failure',
  props<{ error: any }>()
);

// user.reducer.ts
import { createReducer, on } from '@ngrx/store';

const initialState: UserState = {
  user: null,
  loading: false,
  error: null
};

export const userReducer = createReducer(
  initialState,
  on(loadUser, state => ({ ...state, loading: true })),
  on(loadUserSuccess, (state, { user }) => ({ 
    ...state, 
    user, 
    loading: false, 
    error: null 
  })),
  on(loadUserFailure, (state, { error }) => ({ 
    ...state, 
    loading: false, 
    error 
  }))
);
```

### SSR-Compatible Service

```typescript
// platform.service.ts
import { Injectable, Inject, PLATFORM_ID } from '@angular/core';
import { isPlatformBrowser, DOCUMENT } from '@angular/common';

@Injectable({
  providedIn: 'root'
})
export class PlatformService {
  constructor(
    @Inject(PLATFORM_ID) private platformId: Object,
    @Inject(DOCUMENT) private document: Document
  ) {}
  
  getWindowWidth(): number {
    if (isPlatformBrowser(this.platformId)) {
      return window.innerWidth;
    }
    return 1024; // Default for SSR
  }
  
  setTitle(title: string): void {
    if (isPlatformBrowser(this.platformId)) {
      this.document.title = title;
    }
  }
}
```

## API Reference

### AngularLanguagePlugin

Main plugin class for Angular error detection and healing.

#### Methods

- `can_handle(error_data: Dict[str, Any]) -> bool`
- `analyze_error(error_data: Dict[str, Any]) -> Dict[str, Any]`
- `generate_fix(error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str) -> Optional[Dict[str, Any]]`
- `get_language_info() -> Dict[str, Any]`

### AngularExceptionHandler

Handles Angular-specific exception analysis.

#### Methods

- `analyze_dependency_injection_error(error_data: Dict[str, Any]) -> Dict[str, Any]`
- `analyze_ngrx_error(error_data: Dict[str, Any]) -> Dict[str, Any]`
- `analyze_template_binding_error(error_data: Dict[str, Any]) -> Dict[str, Any]`

### AngularPatchGenerator

Generates fixes for Angular errors.

#### Methods

- `generate_patch(error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str) -> Optional[Dict[str, Any]]`

## Contributing

To contribute to Angular plugin development:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Update documentation
5. Submit a pull request

### Adding New Error Types

1. Add rule to appropriate JSON file in `modules/analysis/rules/angular/`
2. Implement handler method in `AngularExceptionHandler`
3. Add patch generation logic in `AngularPatchGenerator`
4. Create template file if needed
5. Add test cases

### Testing

Run Angular plugin tests:

```bash
pytest tests/test_angular_plugin.py -v
```

## License

This Angular integration is part of the Homeostasis project and is licensed under the same terms.
