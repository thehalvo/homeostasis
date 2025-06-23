# Change Log

All notable changes to the Homeostasis Self-Healing extension will be documented in this file.

## [0.1.0] - 2024-01-01

### Added
- Initial release of Homeostasis Self-Healing VSCode extension
- Real-time code healing and error detection
- Inline fix suggestions with confidence scores
- Code lens for error prevention and proactive suggestions
- Settings synchronization across devices and environments
- Telemetry for developer experience improvement
- Multi-language support (Python, JavaScript, TypeScript, Java, Go, Rust, C#, PHP, Ruby, Scala, Elixir, Clojure, Swift, Kotlin, Dart)
- Integration with Homeostasis healing server
- Command palette integration
- Context menu integration
- Status bar integration
- Configurable healing thresholds and delays
- Settings profiles for different development environments
- Import/export settings functionality
- Privacy-focused telemetry with user control

### Features
- **Real-time Analysis**: Code is analyzed as you type with configurable delays
- **Intelligent Suggestions**: AI-powered fix suggestions with confidence scores
- **Batch Healing**: Apply multiple fixes at once or heal entire workspace
- **Visual Feedback**: Code lens and inline hints for potential issues
- **Cross-device Sync**: Keep settings synchronized across development environments
- **Privacy First**: All telemetry is optional and anonymous

### Commands
- `Homeostasis: Heal Current File` - Apply healing fixes to active file
- `Homeostasis: Heal Entire Workspace` - Scan and heal all supported files
- `Homeostasis: Enable/Disable Real-time Healing` - Control automatic healing
- `Homeostasis: Show Healing Dashboard` - Open web dashboard
- `Homeostasis: Sync Settings` - Synchronize settings with server
- `Homeostasis: Create/Load Settings Profile` - Manage settings profiles
- `Homeostasis: Export/Import Settings` - Backup and restore settings
- `Homeostasis: Configure Telemetry` - Manage privacy settings

### Configuration
- Server URL and API key configuration
- Language-specific enablement
- Confidence thresholds for auto-fixing
- Real-time healing delays and behavior
- Visual feedback preferences
- Privacy and telemetry controls