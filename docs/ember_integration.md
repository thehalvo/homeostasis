# Ember.js Integration with Homeostasis

This document outlines how to integrate Ember.js applications with the Homeostasis self-healing framework.

## Overview

The Ember.js plugin for Homeostasis provides error detection, analysis, and automatic fix generation for common Ember.js errors. It supports:

- Component and template-related errors
- Ember Data store and model issues
- Router and URL handling problems
- Ember Octane features (tracked properties, modifiers, etc.)
- Testing environment debugging

## Supported Ember.js Versions

The plugin is designed to work with Ember.js 3.x and 4.x, including Ember Octane features.

## Installation

To use Homeostasis with your Ember.js application:

1. Install the Homeostasis framework:

```bash
pip install homeostasis
```

2. Add the Ember.js monitoring to your application.

## Configuration

### Basic Setup

Configure Homeostasis to monitor your Ember.js application by adding the following to your `config/environment.js`:

```javascript
// config/environment.js
'use strict';

module.exports = function(environment) {
  let ENV = {
    // other configuration...
    
    homeostasis: {
      enabled: true,
      errorReporting: true,
      autoHealing: true
    }
  };
  
  // Environment-specific configurations
  if (environment === 'development') {
    ENV.homeostasis.developmentMode = true;
  }
  
  return ENV;
};
```

### Ember Data Integration

To enable monitoring for Ember Data issues, add the following adapter:

```javascript
// app/adapters/application.js
import JSONAPIAdapter from '@ember-data/adapter/json-api';
import { inject as service } from '@ember/service';

export default class ApplicationAdapter extends JSONAPIAdapter {
  @service store;
  
  handleResponse(status, headers, payload, requestData) {
    const response = super.handleResponse(...arguments);
    
    // Report errors to Homeostasis
    if (!response.isSuccess) {
      window.homeostasis?.reportError({
        type: 'EmberDataError',
        message: response.message || 'Ember Data request failed',
        status,
        payload,
        requestData
      });
    }
    
    return response;
  }
}
```

### Router Integration

To monitor router-related issues:

```javascript
// app/router.js
import EmberRouter from '@ember/routing/router';
import config from './config/environment';

export default class Router extends EmberRouter {
  location = config.locationType;
  rootURL = config.rootURL;
  
  constructor() {
    super(...arguments);
    
    this.on('routeDidChange', (transition) => {
      // Track successful transitions
    });
    
    this.on('error', (error, transition) => {
      // Report router errors to Homeostasis
      window.homeostasis?.reportError({
        type: 'RouterError',
        message: error.message,
        transition: {
          from: transition.from?.name,
          to: transition.to?.name,
          params: transition.to?.params
        },
        stack: error.stack
      });
    });
  }
}

// Define your routes...
Router.map(function() {
  // ...
});
```

### Component Error Handling

For component-level error tracking:

```javascript
// app/components/error-boundary.js
import Component from '@glimmer/component';
import { tracked } from '@glimmer/tracking';
import { action } from '@ember/object';

export default class ErrorBoundaryComponent extends Component {
  @tracked hasError = false;
  @tracked error = null;
  
  constructor() {
    super(...arguments);
    window.addEventListener('error', this.handleError);
  }
  
  willDestroy() {
    super.willDestroy();
    window.removeEventListener('error', this.handleError);
  }
  
  @action
  handleError(event) {
    this.hasError = true;
    this.error = event.error;
    
    // Report to Homeostasis
    window.homeostasis?.reportError({
      type: event.error?.name || 'Error',
      message: event.error?.message || 'Unknown error',
      stack: event.error?.stack,
      componentName: this.args.componentName
    });
    
    // Prevent default browser error handling
    event.preventDefault();
    return true;
  }
}
```

```handlebars
{{!-- app/components/error-boundary.hbs --}}
{{#if this.hasError}}
  <div class="error-boundary">
    <h3>Something went wrong</h3>
    <p>{{this.error.message}}</p>
    {{#if @onReset}}
      <button {{on "click" @onReset}}>Try Again</button>
    {{/if}}
  </div>
{{else}}
  {{yield}}
{{/if}}
```

Usage:

```handlebars
<ErrorBoundary @componentName="UserProfile">
  <UserProfile @userId={{this.userId}} />
</ErrorBoundary>
```

## Error Detection

The Homeostasis Ember.js plugin can detect and diagnose various types of errors:

### Template Errors

- Syntax errors in Handlebars templates
- Missing or improperly defined helpers
- Component not found errors
- Block parameter errors in helpers like `{{#each}}`

### Component Errors

- Lifecycle hook issues
- Property access errors
- Binding errors
- Action handler problems

### Ember Data Errors

- Record not found errors
- Relationship loading issues
- Store injection problems
- Adapter and serializer errors

### Router Errors

- Route not found errors
- Transition aborted issues
- Dynamic segment errors
- Model hook failures

### Ember Octane Errors

- Tracked property issues
- Component args access errors
- Modifier-related problems
- Class-based component issues

## Automatic Fixes

Homeostasis can automatically generate and apply fixes for common Ember.js errors:

### Template Fixes

- Creating missing helpers
- Fixing template syntax errors
- Correcting component invocations
- Adding proper block parameters

### Component Fixes

- Adding missing tracked properties
- Implementing proper action handlers
- Fixing lifecycle hook issues
- Correcting service injections

### Ember Data Fixes

- Adding store service injection
- Implementing proper model definitions
- Adding relationship error handling
- Fixing adapter configurations

### Router Fixes

- Defining missing routes
- Adding transition error handling
- Implementing proper model hooks
- Handling dynamic segments

### Ember Octane Fixes

- Converting to tracked properties
- Implementing proper args access
- Adding modifiers for DOM interactions
- Converting to class-based components

## Testing Integration

To ensure Homeostasis works well with your Ember.js tests:

```javascript
// tests/test-helper.js
import Application from 'your-app-name/app';
import config from 'your-app-name/config/environment';
import * as QUnit from 'qunit';
import { setApplication } from '@ember/test-helpers';
import { setup } from 'qunit-dom';
import { start } from 'ember-qunit';

// Configure Homeostasis for testing
window.homeostasis = {
  reportError(error) {
    console.log('Homeostasis test error:', error);
    // In test mode, you might want to collect errors rather than auto-fix
  }
};

setApplication(Application.create(config.APP));
setup(QUnit.assert);
start();
```

## Best Practices

1. **Structured Error Reporting**: Use consistent error reporting patterns across your application.

2. **Component Boundaries**: Use error boundary components to isolate errors and prevent cascade failures.

3. **Detailed Context**: Include additional context with errors, such as component names, route information, and user actions.

4. **Staged Rollout**: Enable monitoring first, then gradually enable auto-healing features.

5. **Test Coverage**: Ensure you have tests for error scenarios that Homeostasis will handle.

## Troubleshooting

### Common Issues

1. **Errors not being detected**:
   - Ensure Homeostasis is properly initialized
   - Check that error reporting is enabled
   - Verify error events are not being stopped by other handlers

2. **Fixes not being applied**:
   - Check that auto-healing is enabled
   - Ensure you're using a supported Ember.js version
   - Verify the error pattern is supported by the rules engine

3. **Unexpected behavior after fixes**:
   - Review the applied fixes in the Homeostasis dashboard
   - Consider adjusting fix templates for your specific code patterns
   - Add more tests for the affected components

## Examples

### Handling a Missing Component

When Homeostasis detects a "component not found" error:

```
Error: Component not found: user-profile
```

It can automatically generate the component files:

```javascript
// app/components/user-profile.js
import Component from '@glimmer/component';
import { tracked } from '@glimmer/tracking';
import { action } from '@ember/object';
import { inject as service } from '@ember/service';

export default class UserProfileComponent extends Component {
  @service store;
  @tracked user = null;
  
  constructor() {
    super(...arguments);
    this.loadUser();
  }
  
  async loadUser() {
    try {
      this.user = await this.store.findRecord('user', this.args.userId);
    } catch (error) {
      console.error('Failed to load user:', error);
    }
  }
  
  @action
  updateProfile() {
    // Implementation
  }
}
```

```handlebars
{{!-- app/components/user-profile.hbs --}}
<div class="user-profile">
  {{#if this.user}}
    <h2>{{this.user.name}}</h2>
    <p>{{this.user.email}}</p>
    <button {{on "click" this.updateProfile}}>Update Profile</button>
  {{else}}
    <p>Loading user profile...</p>
  {{/if}}
</div>
```

### Fixing Ember Data Store Issues

When Homeostasis detects a "store is not injected" error:

```
Error: Cannot read property 'findRecord' of undefined
```

It can add the missing service injection:

```javascript
// Before:
export default class UserProfileComponent extends Component {
  user = null;
  
  constructor() {
    super(...arguments);
    this.loadUser();
  }
  
  async loadUser() {
    try {
      this.user = await this.store.findRecord('user', this.args.userId);
    } catch (error) {
      console.error('Failed to load user:', error);
    }
  }
}

// After:
export default class UserProfileComponent extends Component {
  @service store;  // Added service injection
  @tracked user = null;
  
  constructor() {
    super(...arguments);
    this.loadUser();
  }
  
  async loadUser() {
    try {
      this.user = await this.store.findRecord('user', this.args.userId);
    } catch (error) {
      console.error('Failed to load user:', error);
    }
  }
}
```

## Further Reading

- [Ember.js Official Documentation](https://emberjs.com/docs/)
- [Ember Octane Guide](https://guides.emberjs.com/release/upgrading/current-edition/)
- [Homeostasis Architecture](./architecture.md)
- [Homeostasis Rule System](./contributing-rules.md)
- [Cross-Language Features](./cross_language_features.md)