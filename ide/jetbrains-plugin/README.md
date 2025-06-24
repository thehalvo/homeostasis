# Homeostasis JetBrains Plugin

Real-time code healing and error prevention plugin for JetBrains IDEs (IntelliJ IDEA, PyCharm, WebStorm, and more).

## Features

### 🔧 Real-time Code Healing
- Automatic error detection and fixing as you type
- Support for 15+ programming languages
- Confidence-based auto-fixes with configurable thresholds
- Language-specific healing rules and patterns

### 🔍 Advanced Inspections
- Integration with JetBrains inspection system
- Custom Homeostasis inspections for each supported language
- Quick fixes with detailed explanations
- Preventive healing suggestions

### ⚡ Intelligent Actions
- **Heal File**: Analyze and fix the current file
- **Heal Project**: Comprehensive project-wide healing
- **Toggle Real-time Healing**: Enable/disable automatic healing
- Keyboard shortcuts for quick access

### 🎛️ Configuration UI
- Embedded settings panel in IDE preferences
- Server configuration and API key management
- Language selection and confidence thresholds
- Telemetry and notification preferences

### 📊 Healing Dashboard
- Real-time statistics and success rates
- Healing history with detailed logs
- Language and rule breakdowns
- Performance metrics

### 🌐 Remote Development Support
- Full compatibility with JetBrains Gateway
- Optimized for remote development environments
- Configuration synchronization across environments
- Network-aware healing strategies

## Supported Languages

- Python
- Java
- JavaScript/TypeScript
- Go
- Rust
- C#
- PHP
- Ruby
- Scala
- Elixir
- Clojure
- Swift
- Kotlin
- Dart

## Installation

### From JetBrains Marketplace
1. Open your JetBrains IDE
2. Go to **File → Settings → Plugins**
3. Search for "Homeostasis Self-Healing"
4. Click **Install**

### Manual Installation
1. Download the plugin JAR from releases
2. Go to **File → Settings → Plugins**
3. Click the gear icon → **Install Plugin from Disk**
4. Select the downloaded JAR file

## Configuration

### Server Setup
1. Go to **File → Settings → Tools → Homeostasis**
2. Set your Homeostasis server URL (default: `http://localhost:8080`)
3. Enter your API key if authentication is required

### Language Configuration
1. Select which programming languages to enable healing for
2. Adjust confidence threshold (0.0-1.0) for auto-applying fixes
3. Configure healing delay for real-time analysis

### Telemetry
- Enable/disable telemetry for improving the healing experience
- All data is anonymized and used only for improving suggestions

## Usage

### Automatic Healing
- Real-time healing is enabled by default
- Errors are detected and fixed automatically as you type
- Configurable delay before healing triggers (default: 2 seconds)

### Manual Healing
- **Ctrl+Alt+H**: Heal current file
- **Ctrl+Alt+Shift+H**: Heal entire project
- **Ctrl+Alt+T**: Toggle real-time healing

### Inspections and Quick Fixes
- Homeostasis inspections appear alongside standard IDE inspections
- Apply individual fixes through intention actions
- Bulk apply preventive healing optimizations

### Tool Window
- Access healing statistics and history
- View applied healings by language and rule type
- Clear history and refresh statistics

## Remote Development

The plugin fully supports JetBrains Remote Development:

- Automatic detection of remote environments
- Optimized healing strategies for network latency
- Configuration synchronization between local and remote
- Increased healing delays and confidence thresholds for stability

## Building from Source

### Prerequisites
- JDK 17 or higher
- Gradle 7.0 or higher

### Build Commands
```bash
# Build the plugin
./gradlew build

# Run in development IDE
./gradlew runIde

# Package for distribution
./gradlew buildPlugin
```

### Development Setup
1. Clone the repository
2. Open in IntelliJ IDEA
3. Import Gradle project
4. Run the "runIde" Gradle task

## Configuration Options

### Server Settings
- **Server URL**: Homeostasis healing server endpoint
- **API Key**: Authentication key for server access
- **Connection Timeout**: Network timeout settings

### Healing Behavior
- **Real-time Healing**: Enable/disable automatic healing
- **Healing Delay**: Milliseconds to wait before triggering healing
- **Confidence Threshold**: Minimum confidence for auto-applying fixes
- **Enabled Languages**: Select which languages to heal

### UI Preferences
- **Inline Hints**: Show healing suggestions inline
- **Inspections**: Enable Homeostasis inspections
- **Notifications**: Show healing completion notifications
- **Telemetry**: Enable usage analytics

## Troubleshooting

### Common Issues

**Plugin not detecting errors:**
- Check that your language is enabled in settings
- Verify server connectivity in the tool window
- Ensure confidence threshold isn't set too high

**Slow healing response:**
- Increase healing delay for better performance
- Check network connectivity to healing server
- Consider disabling real-time healing for large files

**Remote development issues:**
- Verify server is accessible from remote environment
- Check that ports are properly forwarded
- Use higher confidence thresholds for stability

### Debug Mode
Enable debug logging by adding to IDE custom VM options:
```
-Didea.log.debug.categories=com.homeostasis.healing
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

### Code Style
- Follow Kotlin coding conventions
- Use KDoc for public APIs
- Include unit tests for new features
- Follow existing project structure

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

- **Documentation**: [https://homeostasis.dev/docs](https://homeostasis.dev/docs)
- **Issues**: [GitHub Issues](https://github.com/homeostasis/homeostasis/issues)
- **Discussions**: [GitHub Discussions](https://github.com/homeostasis/homeostasis/discussions)
- **Email**: support@homeostasis.dev