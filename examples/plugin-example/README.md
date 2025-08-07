# Enhanced Python Analyzer Plugin

An advanced Python error analysis plugin for the Universal Self-Healing Standard (USHS) that provides ML-powered insights and framework-specific optimizations.

## Features

- **Advanced Error Analysis**: Deep analysis of Python errors with context understanding
- **ML-Powered Insights**: Machine learning models provide intelligent fix suggestions
- **Framework Support**: Optimized for Django, Flask, FastAPI, and more
- **Pattern Detection**: Identifies common anti-patterns and suggests improvements
- **Performance Analysis**: Detects performance bottlenecks in Python code

## Installation

### Via USHS Marketplace

```bash
ushs plugin install enhanced-python-analyzer
```

### Manual Installation

```bash
git clone https://github.com/homeostasis/enhanced-python-analyzer.git
cd enhanced-python-analyzer
ushs plugin install .
```

## Configuration

Create a configuration file at `~/.homeostasis/plugins/enhanced-python-analyzer/config.json`:

```json
{
  "mlApiKey": "your-api-key-here",
  "analysisDepth": "deep",
  "frameworkOptimizations": true,
  "telemetryEnabled": false,
  "cacheTTL": 7200
}
```

### Configuration Options

- **mlApiKey**: (Optional) API key for enhanced ML features
- **analysisDepth**: Analysis depth - `shallow`, `standard`, or `deep`
- **frameworkOptimizations**: Enable framework-specific optimizations
- **telemetryEnabled**: Send anonymous usage statistics
- **cacheTTL**: Cache time-to-live in seconds

## Usage

The plugin automatically integrates with the USHS error analysis pipeline. When a Python error is detected:

1. The plugin performs deep code analysis
2. ML models analyze the error context
3. Framework-specific patterns are checked
4. Intelligent fix suggestions are generated

### Example

```python
# Error: KeyError: 'user_id'
# File: app.py, line 42

# The plugin will analyze the context and suggest:
# 1. Use .get() method with default value
# 2. Add validation before accessing the key
# 3. Consider using TypedDict for better type safety
```

## API Reference

### Plugin Interface

```python
class EnhancedPythonAnalyzer(USHSPlugin):
    def analyze_error(self, error_data: Dict[str, Any]) -> AnalysisResult:
        """Analyze Python error with ML-powered insights."""
        
    def detect_patterns(self, code_context: CodeContext) -> List[Pattern]:
        """Detect code patterns and anti-patterns."""
        
    def suggest_fixes(self, analysis: AnalysisResult) -> List[FixSuggestion]:
        """Generate intelligent fix suggestions."""
```

### Analysis Result Schema

```json
{
  "error_type": "KeyError",
  "confidence": 0.95,
  "root_cause": "Missing key validation",
  "patterns_detected": ["unsafe_dict_access", "missing_validation"],
  "fix_suggestions": [
    {
      "type": "code_change",
      "description": "Use safe dictionary access",
      "patch": "...",
      "confidence": 0.98
    }
  ],
  "framework_specific": {
    "framework": "django",
    "recommendations": ["Use Django's get_object_or_404"]
  }
}
```

## Development

### Prerequisites

- Python 3.8+
- USHS SDK
- Development dependencies: `pip install -r requirements-dev.txt`

### Building from Source

```bash
# Clone the repository
git clone https://github.com/homeostasis/enhanced-python-analyzer.git
cd enhanced-python-analyzer

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/

# Build plugin package
ushs plugin build
```

### Testing

```bash
# Run unit tests
pytest tests/unit/

# Run integration tests
pytest tests/integration/

# Run with coverage
pytest --cov=src tests/
```

## Troubleshooting

### Common Issues

1. **ML API Connection Failed**
   - Check your API key configuration
   - Ensure network permissions are granted
   - Verify firewall settings

2. **High Memory Usage**
   - Reduce `analysisDepth` to `standard` or `shallow`
   - Decrease `cacheTTL` to free memory more frequently

3. **Framework Detection Issues**
   - Ensure project dependencies are installed
   - Check that framework files are accessible

### Debug Mode

Enable debug logging:

```bash
export USHS_PLUGIN_DEBUG=enhanced-python-analyzer
ushs run --verbose
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Workflow

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Security

This plugin has been security audited and follows USHS security guidelines:

- ✅ Code signed with GPG
- ✅ No filesystem write access
- ✅ Limited network access (HTTPS only)
- ✅ Sandboxed execution
- ✅ Regular security updates

Report security issues to: security@homeostasis.dev

## License

MIT License - see [LICENSE](LICENSE) for details.

## Support

- **Documentation**: https://docs.homeostasis.dev/plugins/enhanced-python-analyzer
- **Issues**: https://github.com/homeostasis/enhanced-python-analyzer/issues
- **Email**: support@homeostasis.dev
- **Community**: https://discord.gg/homeostasis

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history.

---

Made with ❤️ by the Homeostasis Community