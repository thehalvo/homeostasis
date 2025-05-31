# Web Components Integration Guide

This document provides a comprehensive guide for integrating the Homeostasis self-healing framework with Web Components. It covers error detection, healing strategies, and integration with popular Web Component frameworks.

## Overview

Web Components is a suite of different technologies allowing you to create reusable custom elements with their functionality encapsulated away from the rest of your code. The Homeostasis Web Components plugin provides error detection and healing for:

1. **Custom Elements lifecycle issues**
2. **Shadow DOM encapsulation problems**
3. **HTML Template element optimization**
4. **Framework interoperability healing**
5. **Lit and Stencil framework integration**

## Setup and Configuration

### Installation

To enable Web Components support in your Homeostasis installation:

1. The Web Components plugin is included by default in Homeostasis.
2. Ensure you're using the latest version of Homeostasis to access all Web Components features.

### Configuration

Configure the Web Components plugin in your `config.yaml`:

```yaml
plugins:
  enabled:
    - webcomponents
  
  webcomponents:
    error_detection:
      enabled: true
      frameworks:
        - lit
        - stencil
        - vanilla
    
    healing:
      enabled: true
      templates_directory: "custom/templates/web_components"  # Optional - defaults to built-in templates
      
    interoperability:
      frameworks:
        - react
        - angular
        - vue
```

## Error Detection

The Web Components plugin detects various error categories:

### Custom Elements Lifecycle Errors

- Missing `super()` calls in constructors
- Errors in lifecycle callbacks: 
  - `connectedCallback`
  - `disconnectedCallback`
  - `attributeChangedCallback`
  - `adoptedCallback`
- Issues with `observedAttributes` implementation
- Custom Element registration problems

### Shadow DOM Encapsulation Issues

- Closed shadow root access attempts
- Style leakage between components
- Event retargeting confusion
- Slot content distribution problems
- Part attribute styling issues

### HTML Template Element Problems

- Template content not properly cloned
- Inefficient template creation
- Template content modification errors
- Inefficient template cloning

### Framework Interoperability Issues

- React event binding problems
- Angular binding syntax issues
- Vue custom event naming
- Property/attribute reflection
- Event bubbling and composition
- Framework styling leakage

## Healing Strategies

The Web Components plugin provides automatic healing for detected issues:

### Custom Elements Healing

- Adds missing `super()` calls in constructors
- Implements proper `observedAttributes` getter
- Fixes attribute change handling
- Resolves registration conflicts

Example fix for missing `super()`:

```javascript
// Original problematic code
class MyElement extends HTMLElement {
  constructor() {
    this.foo = 'bar';  // Missing super() call
  }
}

// Fixed code
class MyElement extends HTMLElement {
  constructor() {
    super();  // Added super() call
    this.foo = 'bar';
  }
}
```

### Shadow DOM Healing

- Converts closed shadow roots to open mode when appropriate
- Adds proper CSS encapsulation
- Fixes event retargeting with `composedPath()`
- Resolves slot content distribution

Example fix for Shadow DOM access:

```javascript
// Original problematic code
const element = document.createElement('my-element');
element.shadowRoot.querySelector('.item');  // May be null if closed mode

// Fixed code
const element = document.createElement('my-element');
// Use if check or optional chaining
if (element.shadowRoot) {
  element.shadowRoot.querySelector('.item');
}
```

### HTML Template Optimization

- Implements class-level template caching
- Adds proper template cloning with `importNode`
- Optimizes rendering performance
- Implements lazy initialization

Example template optimization:

```javascript
// Original inefficient code
class MyElement extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
    
    // Creates template every instance
    const template = document.createElement('template');
    template.innerHTML = `<div>Content</div>`;
    this.shadowRoot.appendChild(template.content.cloneNode(true));
  }
}

// Optimized code
class MyElement extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
    
    // Use class-level cached template
    if (!MyElement.template) {
      MyElement.template = document.createElement('template');
      MyElement.template.innerHTML = `<div>Content</div>`;
    }
    
    // Proper importNode usage
    this.shadowRoot.appendChild(document.importNode(MyElement.template.content, true));
  }
}
```

### Framework Interoperability

- Implements cross-framework event handling
- Fixes property/attribute reflection
- Adds proper event bubbling and composed flags
- Resolves styling encapsulation issues

## Lit and Stencil Integration

The plugin provides specialized support for the Lit and Stencil frameworks:

### Lit Integration

- Detects and fixes property declaration issues
- Resolves reactivity problems
- Fixes template syntax errors
- Handles Lit-specific lifecycle issues

Example Lit error detection:

```javascript
// Problematic Lit code
import { LitElement, html } from 'lit';

class MyElement extends LitElement {
  static properties = {
    name: { type: String }
  }
  
  render() {
    return html`
      <div>${this.namee}</div>  <!-- Typo: should be this.name -->
    `;
  }
}
```

### Stencil Integration

- Fixes decorator usage issues
- Resolves render function errors
- Handles component tag mismatches
- Addresses lifecycle order problems

Example Stencil error detection:

```javascript
// Problematic Stencil code
import { Component, h } from '@stencil/core';

@Component({
  tag: 'my-component',
  styleUrl: 'my-component.css',
  shadow: true
})
export class MyComponent {
  // Missing @Prop() decorator
  name: string;
  
  render() {
    return <div>{this.name}</div>;
  }
}
```

## Testing Web Components

When implementing Web Components with Homeostasis, testing is crucial:

1. **Unit Testing**: Test component behavior in isolation
2. **Integration Testing**: Test component interaction with other components
3. **End-to-End Testing**: Test components in actual browser environments

Homeostasis provides specialized test helpers for Web Components:

```python
from homeostasis.testing import WebComponentTestHelper

# Create test helper
helper = WebComponentTestHelper()

# Test custom element
helper.register_element('my-element', source_code)
result = helper.test_element_creation('my-element')

if result.has_errors():
    print(f"Detected errors: {result.errors}")
    print(f"Suggested fixes: {result.fixes}")
```

## Best Practices

For optimal results with Homeostasis and Web Components:

1. **Always call `super()` first** in your custom element constructor
2. **Use open shadow mode** unless closed mode is absolutely necessary
3. **Cache templates at class level** for better performance
4. **Implement both property setters and attribute handlers** for framework compatibility
5. **Add event listeners in `connectedCallback`** and remove them in `disconnectedCallback`
6. **Use `bubbles: true, composed: true`** when dispatching events that need to cross shadow boundaries
7. **Use the `::part()` API** for styling hooks instead of CSS custom properties
8. **Reflect properties to attributes** for better HTML interoperability
9. **Use proper ARIA roles and attributes** for accessibility

## Troubleshooting

Common issues and their solutions:

| Problem | Solution |
|---------|----------|
| Constructor error with super() | Ensure super() is called before any property access |
| Shadow DOM is null | Check if attachShadow() was called or if mode: 'closed' is preventing access |
| Events not crossing shadow boundary | Set both bubbles: true and composed: true when dispatching events |
| Template performance issues | Cache templates at class level rather than creating in each instance |
| React events not working | Use uppercase event names (onClick) and ensure events bubble/compose |
| Angular binding not working | Implement both property setters and attributeChangedCallback |
| Vue custom events not detected | Use kebab-case for custom event names |

## Version Compatibility

The Web Components plugin is compatible with:

- Modern browsers supporting Web Components v1 spec
- Lit 2.0+
- Stencil 2.0+
- React 16.8+ (with support for refs and custom events)
- Angular 9+ (with ViewEncapsulation.ShadowDom support)
- Vue 3+ (with custom element support)

## Further Resources

- [Web Components Specification](https://www.webcomponents.org/)
- [MDN Web Components Guide](https://developer.mozilla.org/en-US/docs/Web/Web_Components)
- [Lit Documentation](https://lit.dev/)
- [Stencil Documentation](https://stenciljs.com/)
- [Homeostasis Core Documentation](./architecture.md)