# YAML/JSON Integration

The Homeostasis YAML/JSON Language Plugin provides error analysis and patch generation for configuration files across various tools and frameworks. It supports both YAML and JSON formats with intelligent framework-specific error detection.

## Overview

The YAML/JSON plugin enables Homeostasis to:
- Analyze syntax errors in YAML and JSON configuration files
- Detect framework-specific configuration issues
- Handle schema validation and structure problems
- Provide intelligent suggestions for configuration optimization
- Support multiple configuration frameworks and tools

## Supported Formats

- **YAML 1.2** - Full YAML specification support
- **JSON RFC 7159** - Standard JSON format
- **JSON5** - Extended JSON with comments and trailing commas
- **TOML** - Basic TOML configuration support

## Supported Frameworks

### Container & Orchestration
- **Kubernetes** - Pod, Service, Deployment configurations
- **Docker Compose** - Multi-container application definitions
- **Helm** - Kubernetes package manager charts

### CI/CD Platforms
- **GitHub Actions** - Workflow and action definitions
- **GitLab CI** - Pipeline and job configurations
- **Circle CI** - Build and deployment workflows
- **Azure DevOps** - Pipeline YAML configurations

### Infrastructure as Code
- **Terraform** - Variable and configuration files
- **Ansible** - Playbook and inventory files
- **CloudFormation** - AWS resource templates

### Development Tools
- **ESLint** - JavaScript/TypeScript linting configuration
- **Prettier** - Code formatting configuration
- **Babel** - JavaScript compilation configuration
- **Webpack** - Module bundling configuration
- **Package Managers** - npm, yarn, composer configuration

## Key Features

### Error Detection Categories

1. **Syntax Errors**
   - Invalid YAML/JSON syntax
   - Indentation problems
   - Quote and bracket mismatches
   - Encoding issues

2. **Structure Errors**
   - Missing required fields
   - Invalid field types
   - Duplicate keys
   - Schema violations

3. **Framework-Specific Errors**
   - Invalid configuration values
   - Deprecated options
   - Missing dependencies
   - Version compatibility issues

4. **Validation Errors**
   - Schema validation failures
   - Type mismatches
   - Range violations
   - Format errors

### Intelligent Framework Detection

The plugin automatically detects configuration frameworks based on:
- File naming patterns (`docker-compose.yml`, `.github/workflows/`)
- Content structure and key patterns
- Schema definitions and required fields

## Usage Examples

### Basic YAML Error Analysis

```python
from homeostasis import analyze_error

# Example YAML syntax error
error_data = {
    "error_type": "YAMLError",
    "message": "found character '\\t' that cannot start any token",
    "file_path": "docker-compose.yml",
    "line_number": 5,
    "content": "services:\n\tapp:\n\t\timage: nginx"
}

analysis = analyze_error(error_data, language="yaml_json")
print(analysis["suggested_fix"])
# Output: "Use spaces instead of tabs for indentation"
```

### JSON Validation Error

```python
# JSON syntax error
json_error = {
    "error_type": "JSONDecodeError",
    "message": "Expecting ',' delimiter: line 3 column 5",
    "file_path": "package.json",
    "content": '{"name": "myapp"\n"version": "1.0.0"}'
}

analysis = analyze_error(json_error, language="yaml_json")
```

### Framework-Specific Analysis

```python
# Kubernetes configuration error
k8s_error = {
    "error_type": "ValidationError",
    "message": "spec.containers is required",
    "file_path": "deployment.yaml",
    "framework": "kubernetes"
}

analysis = analyze_error(k8s_error, language="yaml_json")
```

## Configuration

### Plugin Configuration

Configure the YAML/JSON plugin in your `homeostasis.yaml`:

```yaml
plugins:
  yaml_json:
    enabled: true
    supported_formats: [yaml, json, json5]
    frameworks:
      kubernetes: true
      docker: true
      github_actions: true
      ansible: true
      terraform: true
    validation:
      schema_checking: true
      type_validation: true
      format_validation: true
    patch_generation:
      auto_fix_syntax: true
      suggest_best_practices: true
      framework_specific: true
```

### Framework-Specific Settings

```yaml
plugins:
  yaml_json:
    kubernetes:
      api_versions: ["v1", "apps/v1", "networking.k8s.io/v1"]
      validate_selectors: true
    docker:
      compose_version: "3.8"
      validate_networks: true
    github_actions:
      validate_actions: true
      check_marketplace: false
```

## Error Pattern Recognition

### YAML Syntax Errors

```yaml
# Indentation error (tabs instead of spaces)
services:
	app:  # Error: tabs not allowed
		image: nginx

# Fix: Use spaces for indentation
services:
  app:
    image: nginx

# Missing colon in key-value pair
services
  app:  # Error: missing colon after 'services'
    image: nginx

# Fix: Add colon
services:
  app:
    image: nginx

# Unmatched quotes
name: "My App  # Error: missing closing quote
# Fix: Close the quote
name: "My App"
```

### JSON Syntax Errors

```json
// Missing comma
{
  "name": "myapp"
  "version": "1.0.0"  // Error: missing comma
}

// Fix: Add comma
{
  "name": "myapp",
  "version": "1.0.0"
}

// Trailing comma (invalid in JSON)
{
  "name": "myapp",
  "version": "1.0.0",  // Error: trailing comma
}

// Fix: Remove trailing comma
{
  "name": "myapp",
  "version": "1.0.0"
}
```

### Framework-Specific Errors

#### Kubernetes

```yaml
# Missing required field
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app
spec:
  # Error: Missing 'selector' field
  template:
    spec:
      containers:
      - name: app
        image: nginx

# Fix: Add required selector
spec:
  selector:
    matchLabels:
      app: my-app
  template:
    metadata:
      labels:
        app: my-app
```

#### Docker Compose

```yaml
version: '3.8'
services:
  app:
    image: nginx
    ports:
      - "80"  # Error: Invalid port mapping format
    
# Fix: Use proper port mapping
ports:
  - "80:80"
  # or
  - "8080:80"
```

#### GitHub Actions

```yaml
name: CI
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    step:  # Error: Should be 'steps'
      - uses: actions/checkout@v2

# Fix: Correct field name
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
```

## Framework-Specific Features

### Kubernetes

- **Resource Validation**: Validates Kubernetes resource definitions
- **API Version Checking**: Ensures correct API versions
- **Label Consistency**: Checks selector and label matching
- **Security Policies**: Validates security contexts and policies

### Docker Compose

- **Version Compatibility**: Checks compose file version compatibility
- **Service Dependencies**: Validates service dependencies
- **Network Configuration**: Validates custom networks
- **Volume Mappings**: Checks volume mount syntax

### GitHub Actions

- **Workflow Syntax**: Validates workflow and job structure
- **Action References**: Checks action marketplace references
- **Secret Usage**: Validates secret and environment variable usage
- **Matrix Builds**: Validates build matrix configurations

### GitLab CI

- **Pipeline Structure**: Validates stages and job definitions
- **Script Syntax**: Checks script and before_script sections
- **Variable Usage**: Validates CI/CD variables
- **Artifacts and Caching**: Validates artifact and cache configurations

### Ansible

- **Playbook Structure**: Validates playbook and task structure
- **Module Parameters**: Checks module parameter usage
- **Variable Templating**: Validates Jinja2 template syntax
- **Inventory Format**: Validates inventory file structure

## Best Practices

### YAML Writing

1. **Consistent Indentation**: Use 2 spaces consistently
2. **Quote Strings**: Quote strings with special characters
3. **Use Explicit Types**: Be explicit about data types when needed
4. **Comment Documentation**: Add comments for complex configurations

```yaml
# Good YAML practices
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app
  labels:
    app: my-app
    version: "1.0.0"  # Quoted to ensure string type
spec:
  replicas: 3  # Explicit number
  selector:
    matchLabels:
      app: my-app
  template:
    metadata:
      labels:
        app: my-app
    spec:
      containers:
      - name: app
        image: "nginx:1.21"  # Specific version
        ports:
        - containerPort: 80
```

### JSON Writing

1. **Use Double Quotes**: Always use double quotes for strings
2. **No Trailing Commas**: Avoid trailing commas in objects/arrays
3. **Proper Nesting**: Maintain consistent nesting levels
4. **Schema Validation**: Use JSON Schema for validation

```json
{
  "name": "my-application",
  "version": "1.0.0",
  "description": "Application description",
  "scripts": {
    "start": "node index.js",
    "test": "jest"
  },
  "dependencies": {
    "express": "^4.17.1"
  }
}
```

### Configuration Management

1. **Environment Separation**: Use separate configs for different environments
2. **Secret Management**: Never store secrets in configuration files
3. **Version Control**: Track configuration changes in version control
4. **Validation**: Implement automated configuration validation

## Integration Examples

### CI/CD Validation

```yaml
# GitHub Actions workflow for config validation
name: Validate Configurations
on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Setup Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    
    - name: Install Homeostasis
      run: pip install homeostasis
    
    - name: Validate YAML/JSON files
      run: |
        python -c "
        import os
        import yaml
        import json
        from homeostasis import analyze_error
        
        def validate_file(filepath):
            try:
                with open(filepath, 'r') as f:
                    if filepath.endswith('.json'):
                        json.load(f)
                    else:
                        yaml.safe_load(f)
                print(f'✓ {filepath} is valid')
            except Exception as e:
                error_data = {
                    'error_type': type(e).__name__,
                    'message': str(e),
                    'file_path': filepath
                }
                analysis = analyze_error(error_data, language='yaml_json')
                print(f'✗ {filepath}: {analysis[\"suggested_fix\"]}')
                return False
            return True
        
        # Validate all YAML/JSON files
        all_valid = True
        for root, dirs, files in os.walk('.'):
            for file in files:
                if file.endswith(('.yml', '.yaml', '.json')):
                    filepath = os.path.join(root, file)
                    if not validate_file(filepath):
                        all_valid = False
        
        exit(0 if all_valid else 1)
        "
```

### Pre-commit Hook

```yaml
# .pre-commit-config.yaml
repos:
- repo: local
  hooks:
  - id: yaml-json-validation
    name: YAML/JSON Validation
    entry: python
    language: python
    files: \.(ya?ml|json)$
    args:
    - -c
    - |
      import sys
      import yaml
      import json
      from homeostasis import analyze_error
      
      def validate_file(filepath):
          try:
              with open(filepath, 'r') as f:
                  if filepath.endswith('.json'):
                      json.load(f)
                  else:
                      yaml.safe_load(f)
              return True
          except Exception as e:
              error_data = {
                  'error_type': type(e).__name__,
                  'message': str(e),
                  'file_path': filepath
              }
              analysis = analyze_error(error_data, language='yaml_json')
              print(f'Error in {filepath}: {analysis["suggested_fix"]}')
              return False
      
      for filepath in sys.argv[1:]:
          if not validate_file(filepath):
              sys.exit(1)
```

### Docker Build Integration

```dockerfile
# Dockerfile with configuration validation
FROM python:3.9-slim

COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy configuration files
COPY configs/ /app/configs/

# Validate configurations during build
RUN python -c "
import os
import yaml
import json
from homeostasis import analyze_error

for root, dirs, files in os.walk('/app/configs'):
    for file in files:
        if file.endswith(('.yml', '.yaml', '.json')):
            filepath = os.path.join(root, file)
            try:
                with open(filepath, 'r') as f:
                    if file.endswith('.json'):
                        json.load(f)
                    else:
                        yaml.safe_load(f)
            except Exception as e:
                error_data = {
                    'error_type': type(e).__name__,
                    'message': str(e),
                    'file_path': filepath
                }
                analysis = analyze_error(error_data, language='yaml_json')
                print(f'Configuration error: {analysis[\"suggested_fix\"]}')
                exit(1)
"

COPY . /app
WORKDIR /app
CMD ["python", "app.py"]
```

## Troubleshooting

### Common Issues

1. **Indentation Problems**: Use consistent spacing (2 or 4 spaces)
2. **Quote Mismatches**: Ensure proper quote pairing
3. **Type Confusion**: Be explicit about data types
4. **Framework Detection**: Ensure proper file naming and structure

### Debug Mode

Enable detailed validation output:

```python
import logging
logging.getLogger('homeostasis.yaml_json').setLevel(logging.DEBUG)

# Or use validation tools
import yaml
try:
    with open('config.yml', 'r') as f:
        data = yaml.safe_load(f)
except yaml.YAMLError as e:
    print(f"YAML Error: {e}")
```

### Validation Tools

Use external validation tools:

```bash
# YAML validation
yamllint config.yml

# JSON validation
jq . config.json

# Kubernetes validation
kubectl apply --dry-run=client -f deployment.yaml

# Docker Compose validation
docker-compose config
```

### Custom Schemas

Define custom validation schemas:

```python
from jsonschema import validate, ValidationError
from homeostasis import analyze_error

schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "version": {"type": "string"},
        "dependencies": {"type": "object"}
    },
    "required": ["name", "version"]
}

try:
    validate(instance=config_data, schema=schema)
except ValidationError as e:
    error_data = {
        "error_type": "ValidationError",
        "message": str(e),
        "file_path": "package.json"
    }
    analysis = analyze_error(error_data, language="yaml_json")
    print(analysis["suggested_fix"])
```

## Performance Considerations

- **Parser Efficiency**: Uses fast YAML/JSON parsers
- **Framework Detection**: Quick pattern matching for framework identification
- **Schema Caching**: Caches validation schemas for performance
- **Memory Usage**: Efficient memory usage for large configuration files

## Security Considerations

1. **Secret Detection**: Avoid hardcoded secrets in configuration
2. **Path Validation**: Validate file paths to prevent traversal attacks
3. **Input Sanitization**: Sanitize configuration inputs
4. **Access Control**: Implement proper access controls for configuration files

## Contributing

To extend the YAML/JSON plugin:

1. Add new framework patterns to framework detection
2. Implement framework-specific validation rules
3. Add schema definitions for new frameworks
4. Update documentation with examples

## Related Documentation

- [Error Schema](error_schema.md) - Standard error format
- [Plugin Architecture](plugin_architecture.md) - Plugin development guide
- [Kubernetes Integration](kubernetes_integration.md) - Kubernetes-specific features
- [Docker Integration](docker_integration.md) - Docker and containerization
- [CI/CD Integration](cicd/) - Continuous integration setup