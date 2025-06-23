# Homeostasis Self-Healing VSCode Extension

The Homeostasis Self-Healing extension brings real-time code healing capabilities directly into Visual Studio Code. This extension automatically detects errors, suggests fixes, and can even apply high-confidence patches to keep your code healthy as you write it.

## Features

### 🔧 Real-time Healing
- Automatically analyzes your code as you type
- Provides instant feedback on potential issues
- Suggests context-aware fixes with confidence scores

### 💡 Inline Fix Suggestions
- Quick fixes available through VS Code's built-in code actions
- One-click application of healing suggestions
- Batch healing for multiple issues

### 🎯 Code Lens Integration
- Visual indicators for error-prone code patterns
- Preventive suggestions for common issues
- Technical debt tracking and resolution

### 📊 Multi-language Support
Currently supports:
- Python
- JavaScript/TypeScript
- Java
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

### 🔗 Central System Integration
- Connects to your Homeostasis healing server
- Synchronized settings across development environments
- Telemetry for continuous improvement

## Installation

1. Install the extension from the VS Code marketplace
2. Configure your Homeostasis server URL in settings
3. Start coding with real-time healing enabled!

## Configuration

### Basic Settings

```json
{
  "homeostasis.serverUrl": "http://localhost:8080",
  "homeostasis.realTimeHealing": true,
  "homeostasis.healingDelay": 2000,
  "homeostasis.confidenceThreshold": 0.7
}
```

### Language Configuration

```json
{
  "homeostasis.enabledLanguages": [
    "python",
    "javascript",
    "typescript",
    "java",
    "go"
  ]
}
```

### Privacy Settings

```json
{
  "homeostasis.enableTelemetry": true,
  "homeostasis.showInlineHints": true,
  "homeostasis.enableCodeLens": true
}
```

## Commands

- `Homeostasis: Heal Current File` - Apply healing fixes to the active file
- `Homeostasis: Heal Entire Workspace` - Scan and heal all supported files in workspace
- `Homeostasis: Enable Real-time Healing` - Turn on automatic healing
- `Homeostasis: Disable Real-time Healing` - Turn off automatic healing
- `Homeostasis: Show Healing Dashboard` - Open the web dashboard
- `Homeostasis: Configure Telemetry` - Manage privacy settings

## How It Works

1. **Analysis**: Code is continuously analyzed using Homeostasis rules and ML models
2. **Detection**: Potential issues are identified with confidence scores
3. **Suggestion**: Context-aware fixes are generated and presented
4. **Application**: High-confidence fixes can be applied automatically or manually
5. **Learning**: The system learns from successful fixes to improve future suggestions

## Requirements

- VS Code 1.74.0 or higher
- Homeostasis healing server (local or remote)
- Internet connection for telemetry (optional)

## Privacy

The extension respects your privacy:
- Telemetry can be disabled in settings
- No source code is transmitted without explicit consent
- All data transmission is encrypted
- Anonymous usage statistics only (if enabled)

## Support

For issues, feature requests, or contributions:
- GitHub: [homeostasis/issues](https://github.com/homeostasis/homeostasis/issues)
- Documentation: [docs.homeostasis.dev](https://docs.homeostasis.dev)

## License

This extension is part of the Homeostasis project and is licensed under the same terms as the main project.