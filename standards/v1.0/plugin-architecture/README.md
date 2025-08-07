# USHS Plugin Architecture Documentation

This directory contains the standards and specifications for the Universal Self-Healing Standard (USHS) plugin architecture, enabling a robust ecosystem of extensions and integrations.

## Overview

The USHS plugin architecture provides:

- **Extensibility**: Add new capabilities without modifying core systems
- **Security**: Sandboxed execution with capability-based permissions
- **Discovery**: Automatic plugin discovery and loading
- **Marketplace**: Centralized distribution and management
- **Quality**: Validation, testing, and security scanning

## Documentation Structure

### Standards
- [Plugin Standard](PLUGIN_STANDARD.md) - Complete plugin architecture specification
- [Plugin Manifest Schema](../schemas/plugin-manifest.json) - JSON schema for plugin manifests

### Implementation
- [Plugin Discovery](../../modules/plugin_marketplace/plugin_discovery.py) - Plugin discovery and registry
- [Plugin Security](../../modules/plugin_marketplace/plugin_security.py) - Security framework and sandboxing
- [Marketplace API](../../modules/plugin_marketplace/marketplace_api.py) - REST API for plugin marketplace

### Examples
- [Example Plugin](../../examples/plugin-example/) - Complete example plugin implementation

## Quick Start

### Creating a Plugin

1. **Create plugin directory structure**:
```
my-plugin/
├── manifest.json       # Plugin metadata
├── README.md          # Documentation
├── src/               # Source code
│   └── index.py       # Entry point
├── tests/             # Test suite
└── assets/            # Icons, etc.
```

2. **Define manifest.json**:
```json
{
  "name": "my-plugin",
  "version": "1.0.0",
  "type": "analysis",
  "displayName": "My Plugin",
  "description": "Plugin description",
  "author": {
    "name": "Your Name",
    "email": "you@example.com"
  },
  "license": "MIT",
  "engines": {
    "ushs": "^1.0.0"
  },
  "capabilities": {
    "required": ["error.analyze"]
  }
}
```

3. **Implement plugin class**:
```python
from ushs_core import USHSPlugin

class MyPlugin(USHSPlugin):
    async def initialize(self, context):
        # Initialize plugin
        pass
    
    async def execute(self, input_data):
        # Process input and return output
        return PluginOutput(success=True, data={})
```

### Publishing a Plugin

1. **Validate plugin**:
```bash
ushs plugin validate ./my-plugin
```

2. **Build package**:
```bash
ushs plugin build ./my-plugin
```

3. **Sign plugin**:
```bash
ushs plugin sign ./my-plugin.tar.gz
```

4. **Publish to marketplace**:
```bash
ushs plugin publish ./my-plugin.tar.gz
```

### Installing Plugins

**From marketplace**:
```bash
ushs plugin install enhanced-python-analyzer
```

**From file**:
```bash
ushs plugin install ./plugin.tar.gz
```

**From URL**:
```bash
ushs plugin install https://example.com/plugin.tar.gz
```

## Plugin Categories

### Language Plugins
Provide support for programming languages and frameworks:
- Error parsing and normalization
- Code analysis and AST manipulation
- Fix generation for language-specific patterns
- Framework detection and specialization

### Analysis Plugins
Enhance error analysis and root cause detection:
- Pattern matching algorithms
- Machine learning models
- Statistical analysis tools
- Correlation engines

### Integration Plugins
Connect with external tools and services:
- Monitoring systems (DataDog, Prometheus)
- Issue trackers (Jira, GitHub Issues)
- Communication tools (Slack, Teams)
- Cloud services (AWS, Azure, GCP)

### Deployment Plugins
Support various deployment targets:
- Container orchestration (Kubernetes, Docker Swarm)
- Serverless platforms (Lambda, Functions)
- Edge computing (Cloudflare Workers)
- Traditional servers

### Monitoring Plugins
Extend observability and metrics:
- Custom metric exporters
- Log aggregation adapters
- Trace correlation tools
- Alert generation systems

## Security Model

### Sandboxing
Plugins run in isolated environments based on security level:
- **Basic**: Permission checks and resource limits
- **Standard**: Filesystem isolation and environment restrictions
- **Strict**: Container-based isolation
- **Paranoid**: Maximum isolation with minimal capabilities

### Permissions
Capability-based permission model:
```json
{
  "permissions": {
    "filesystem": ["read"],
    "network": ["https://api.example.com"],
    "environment": ["NODE_ENV"],
    "process": ["spawn"]
  }
}
```

### Code Signing
All plugins must be signed:
- GPG signatures for authentication
- SHA-256 checksums for integrity
- Public key verification
- Certificate chain validation

### Vulnerability Scanning
Automated security checks:
- Dependency vulnerability scanning
- Static code analysis
- Dynamic behavior monitoring
- Regular security audits

## Marketplace Integration

### Publishing Requirements
- Complete documentation
- Test coverage >80%
- Security validation passed
- Code signing implemented
- License specified

### Review Process
1. Automated validation
2. Security scanning
3. Manual code review
4. Community testing
5. Final approval

### Pricing Models
- **Free**: No cost, full features
- **Paid**: One-time or subscription
- **Freemium**: Basic free, premium features paid

## Development Tools

### CLI Commands
```bash
# Plugin management
ushs plugin create <name>        # Create plugin from template
ushs plugin validate <path>      # Validate plugin structure
ushs plugin test <path>          # Run plugin tests
ushs plugin build <path>         # Build plugin package
ushs plugin sign <package>       # Sign plugin package

# Marketplace
ushs plugin search <query>       # Search marketplace
ushs plugin info <plugin-id>     # Get plugin details
ushs plugin install <plugin-id>  # Install plugin
ushs plugin update <plugin-id>   # Update plugin
ushs plugin remove <plugin-id>   # Remove plugin

# Development
ushs plugin dev <path>           # Run in development mode
ushs plugin debug <plugin-id>    # Debug installed plugin
```

### Testing Framework
```python
from ushs_testing import PluginTestCase

class TestMyPlugin(PluginTestCase):
    def test_error_analysis(self):
        # Test plugin functionality
        result = self.plugin.analyze_error(error_data)
        self.assertIsNotNone(result)
```

### Development Workflow
1. Use plugin template generator
2. Implement plugin functionality
3. Write comprehensive tests
4. Validate security requirements
5. Build and sign package
6. Submit for review

## Best Practices

### Code Quality
- Follow language-specific style guides
- Implement comprehensive error handling
- Use type hints/annotations
- Document all public APIs
- Keep dependencies minimal

### Performance
- Lazy load heavy dependencies
- Implement caching where appropriate
- Profile resource usage
- Optimize for cold starts
- Handle timeouts gracefully

### Security
- Never log sensitive data
- Validate all inputs
- Use HTTPS for external calls
- Follow principle of least privilege
- Regular dependency updates

### User Experience
- Clear error messages
- Helpful documentation
- Configuration examples
- Troubleshooting guides
- Responsive support

## Troubleshooting

### Common Issues

**Plugin not loading**:
- Check manifest.json validity
- Verify USHS version compatibility
- Ensure all dependencies available
- Check file permissions

**Permission denied**:
- Review requested permissions
- Check sandbox configuration
- Verify security policies
- Contact admin if needed

**Performance issues**:
- Profile plugin execution
- Check resource limits
- Optimize heavy operations
- Consider caching

## Contributing

We welcome contributions to the plugin ecosystem!

### How to Contribute
1. Read the [Contributing Guide](../CONTRIBUTING.md)
2. Check existing plugins for examples
3. Follow the plugin standards
4. Submit for review
5. Engage with community feedback

### Plugin Ideas
- Language support (Go, Rust, Ruby, etc.)
- Cloud service integrations
- Specialized error analyzers
- Custom deployment strategies
- Monitoring integrations

## Resources

### Documentation
- [USHS Specification](../SPECIFICATION.md)
- [API Reference](../protocols/rest-api.yaml)
- [Security Guidelines](PLUGIN_STANDARD.md#security-requirements)

### Examples
- [Enhanced Python Analyzer](../../examples/plugin-example/)
- [Language Plugin Template](../../modules/analysis/language_plugin_system.py)

### Community
- GitHub Discussions
- Discord Server
- Plugin Showcase
- Developer Forum

## License

The USHS Plugin Architecture is licensed under the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).

---

For questions or support, please contact the USHS team or visit our [community forums](https://community.ushs.org).