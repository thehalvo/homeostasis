# ADR-003: Language Plugin Architecture

Technical Story: #ARCH-003

## Context

Homeostasis needs to support multiple programming languages, each with unique syntax, error patterns, and fix strategies. Hardcoding support for each language would make the codebase unmaintainable and slow down adding new language support. We need a plugin architecture that allows language-specific logic to be developed, tested, and deployed independently while maintaining consistency across all supported languages.

## Decision Drivers

- Extensibility: Easy to add new programming languages
- Maintainability: Language-specific code isolated from core
- Consistency: Uniform interface across all languages
- Performance: Minimal overhead from plugin system
- Developer Experience: Simple plugin development process
- Testing: Language plugins testable in isolation
- Version Management: Support multiple versions of language specs

## Considered Options

1. **Monolithic Integration** - All language support in core codebase
2. **Dynamic Library Plugins** - Compiled plugins loaded at runtime
3. **Microservice Plugins** - Each language as a separate service
4. **Script-Based Plugins** - Interpreted plugins (Python/JS)
5. **Hybrid Approach** - Core interface with pluggable implementations

## Decision Outcome

Chosen option: "Hybrid Approach", combining a well-defined plugin interface in the core with language-specific implementations that can be either embedded or run as separate services, because it provides maximum flexibility while maintaining performance and consistency.

### Positive Consequences

- **Flexible Deployment**: Plugins can be embedded or remote
- **Language Choice**: Plugin developers can use optimal language
- **Clear Interface**: Well-defined contracts between core and plugins
- **Independent Development**: Teams can work autonomously
- **Version Support**: Multiple versions can coexist
- **Performance Options**: Critical plugins can be embedded
- **Testing Isolation**: Each plugin tested independently

### Negative Consequences

- **Interface Complexity**: Need to maintain stable plugin API
- **Versioning Challenges**: API changes affect all plugins
- **Documentation Overhead**: Each plugin needs documentation
- **Quality Variance**: Plugin quality depends on maintainers
- **Debugging Complexity**: Issues may span core and plugins
- **Resource Usage**: Each plugin adds overhead

## Implementation Details

### Plugin Interface

```python
class LanguagePlugin(ABC):
    """Base interface for all language plugins"""
    
    @abstractmethod
    def detect_language(self, file_path: str) -> bool:
        """Detect if file is this language"""
        pass
    
    @abstractmethod
    def parse_error(self, error_text: str) -> ErrorInfo:
        """Parse language-specific error format"""
        pass
    
    @abstractmethod
    def analyze_code(self, file_content: str) -> CodeAnalysis:
        """Perform language-specific analysis"""
        pass
    
    @abstractmethod
    def generate_fix(self, error: ErrorInfo, context: CodeContext) -> Fix:
        """Generate language-specific fix"""
        pass
    
    @abstractmethod
    def validate_syntax(self, code: str) -> ValidationResult:
        """Validate code syntax"""
        pass
    
    @abstractmethod
    def get_test_runner(self) -> TestRunner:
        """Return language-specific test runner"""
        pass
```

### Plugin Structure

```
plugins/
├── python_plugin/
│   ├── manifest.json
│   ├── plugin.py
│   ├── requirements.txt
│   ├── rules/
│   │   ├── syntax_errors.json
│   │   ├── runtime_errors.json
│   │   └── performance_patterns.json
│   ├── templates/
│   │   ├── fix_templates.py
│   │   └── test_templates.py
│   └── tests/
│       └── test_plugin.py
├── javascript_plugin/
├── java_plugin/
└── ...
```

### Plugin Manifest

```json
{
  "name": "python-plugin",
  "version": "1.0.0",
  "language": "python",
  "supported_versions": ["3.8", "3.9", "3.10", "3.11"],
  "author": "Homeostasis Team",
  "description": "Python language support for Homeostasis",
  "entry_point": "plugin.PythonPlugin",
  "deployment": "embedded|service",
  "dependencies": {
    "ast": "builtin",
    "black": "^23.0.0"
  },
  "capabilities": {
    "error_parsing": true,
    "syntax_validation": true,
    "fix_generation": true,
    "test_generation": true,
    "performance_analysis": true
  }
}
```

### Plugin Loading

```python
class PluginManager:
    def __init__(self):
        self.plugins = {}
        self.load_plugins()
    
    def load_plugins(self):
        """Discover and load all plugins"""
        for plugin_dir in glob("plugins/*_plugin"):
            manifest = self.load_manifest(plugin_dir)
            if manifest["deployment"] == "embedded":
                self.load_embedded_plugin(plugin_dir, manifest)
            else:
                self.register_service_plugin(manifest)
    
    def get_plugin_for_file(self, file_path: str) -> LanguagePlugin:
        """Return appropriate plugin for file"""
        for plugin in self.plugins.values():
            if plugin.detect_language(file_path):
                return plugin
        raise NoPluginFoundError(f"No plugin for {file_path}")
```

### Plugin Communication

#### Embedded Plugins
- Direct Python imports
- Shared memory space
- Minimal overhead
- Best for performance-critical operations

#### Service Plugins
- gRPC communication
- Protocol buffer definitions
- Service discovery via Consul
- Best for resource-intensive or isolated operations

### Plugin Development Workflow

1. **Create Plugin Structure**
   ```bash
   homeostasis-cli create-plugin --language=rust
   ```

2. **Implement Interface**
   - Extend LanguagePlugin base class
   - Implement all required methods

3. **Define Rules**
   - Create error pattern rules
   - Define fix templates
   - Add test cases

4. **Test Plugin**
   ```bash
   homeostasis-cli test-plugin rust_plugin/
   ```

5. **Register Plugin**
   ```bash
   homeostasis-cli register-plugin rust_plugin/manifest.json
   ```

### Quality Standards

- Minimum 80% test coverage
- Performance benchmarks required
- Security review for service plugins
- Documentation requirements
- Example fixes required

## Links

- [Plugin Development Guide](../plugin_architecture.md)
- [ADR-001: Microservices Architecture](001-use-microservices-architecture.md)
- [Language Integration Guides](../integration_guides.md)