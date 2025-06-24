# Building the Homeostasis JetBrains Plugin

This document describes how to build, test, and distribute the Homeostasis JetBrains plugin.

## Prerequisites

- **JDK 17 or higher**: Required for building the plugin
- **Gradle 7.0+**: Build system (wrapper included)
- **IntelliJ IDEA**: Recommended for development

## Build Commands

### Basic Build
```bash
# Build the plugin
./gradlew build

# Clean and build
./gradlew clean build
```

### Development
```bash
# Run plugin in development IDE
./gradlew runIde

# Run with specific IDE version
./gradlew runIde -PideVersion=2023.3

# Run tests
./gradlew test

# Run tests with coverage
./gradlew test jacocoTestReport
```

### Packaging
```bash
# Build distributable plugin JAR
./gradlew buildPlugin

# The plugin JAR will be in build/distributions/
```

### Code Quality
```bash
# Run lint checks
./gradlew detekt

# Format code
./gradlew ktlintFormat

# Check code formatting
./gradlew ktlintCheck
```

## IDE Setup

### IntelliJ IDEA
1. Open the `jetbrains-plugin` folder in IntelliJ IDEA
2. The IDE should automatically detect the Gradle project
3. Wait for initial import and indexing to complete
4. Run the "runIde" Gradle task to test the plugin

### VS Code
1. Install the Kotlin extension
2. Open the project folder
3. Use the integrated terminal to run Gradle commands

## Testing

### Unit Tests
```bash
# Run all tests
./gradlew test

# Run specific test class
./gradlew test --tests HealingServiceTest

# Run tests with debugging
./gradlew test --debug-jvm
```

### Integration Testing
```bash
# Run plugin in test IDE with sample projects
./gradlew runIde

# Test with specific language projects
./gradlew runIde -Pidea.plugins.path=/path/to/test/projects
```

### Manual Testing Checklist

1. **Plugin Installation**
   - [ ] Plugin loads without errors
   - [ ] Settings panel appears in preferences
   - [ ] Tool window is available

2. **Basic Functionality**
   - [ ] Real-time healing works
   - [ ] Manual healing actions work
   - [ ] Inspections appear and suggest fixes
   - [ ] Configuration changes take effect

3. **Language Support**
   - [ ] Test with Python files
   - [ ] Test with Java files
   - [ ] Test with JavaScript/TypeScript files
   - [ ] Test with other supported languages

4. **Remote Development**
   - [ ] Plugin works with JetBrains Gateway
   - [ ] Configuration syncs properly
   - [ ] Performance is acceptable over network

## Configuration

### Build Configuration
Edit `build.gradle.kts` to modify:
- Target IntelliJ version
- Plugin dependencies
- Build settings

### Plugin Metadata
Edit `plugin.xml` to update:
- Version information
- Compatibility range
- Feature descriptions

## Distribution

### Local Testing
```bash
# Install plugin locally for testing
./gradlew publishPlugin -Pplugin.verifier.home.dir=/path/to/verifier
```

### Release Build
```bash
# Create release build
./gradlew buildPlugin -Prelease=true

# Sign plugin (requires certificates)
./gradlew signPlugin
```

### Publishing to Marketplace
```bash
# Set publish token
export PUBLISH_TOKEN=your_token_here

# Publish to JetBrains Marketplace
./gradlew publishPlugin
```

## Troubleshooting

### Common Build Issues

**Kotlin compilation errors:**
- Ensure JDK 17+ is being used
- Check Kotlin version compatibility
- Clean and rebuild: `./gradlew clean build`

**Plugin verification failures:**
- Check plugin.xml syntax
- Verify compatibility declarations
- Use: `./gradlew verifyPlugin`

**IDE compatibility issues:**
- Update `sinceBuild` and `untilBuild` in plugin.xml
- Test with target IDE version
- Check deprecated API usage

### Performance Issues

**Slow build times:**
- Enable Gradle daemon: `./gradlew --daemon`
- Increase JVM heap: `-Xmx4g` in gradle.properties
- Use parallel builds: `--parallel`

**Large plugin size:**
- Review dependencies in build.gradle.kts
- Exclude unnecessary libraries
- Use ProGuard for optimization

## Development Guidelines

### Code Organization
```
src/main/kotlin/com/homeostasis/healing/
├── actions/          # IDE actions
├── annotators/       # External annotators
├── components/       # IDE components
├── inspections/      # Code inspections
├── intentions/       # Intention actions
├── remote/          # Remote development support
├── services/        # Application services
├── settings/        # Configuration UI
├── toolwindows/     # Tool windows
└── utils/           # Utility classes
```

### Coding Standards
- Use Kotlin for all new code
- Follow JetBrains plugin development guidelines
- Document public APIs with KDoc
- Include unit tests for business logic
- Use dependency injection through services

### Dependencies
- Minimize external dependencies
- Use IntelliJ Platform APIs when possible
- Prefer coroutines for async operations
- Use Gson for JSON serialization

## Continuous Integration

The project includes GitHub Actions for:
- Building on multiple platforms
- Running tests
- Plugin verification
- Automated releases

### Local CI Testing
```bash
# Run the same checks as CI
./gradlew clean build test verifyPlugin

# Check plugin compatibility
./gradlew runPluginVerifier
```

## Resources

- [IntelliJ Platform SDK](https://plugins.jetbrains.com/docs/intellij/)
- [Plugin Development Guidelines](https://plugins.jetbrains.com/docs/intellij/plugin-development-guidelines.html)
- [Kotlin Style Guide](https://kotlinlang.org/docs/coding-conventions.html)
- [JetBrains Marketplace](https://plugins.jetbrains.com/)