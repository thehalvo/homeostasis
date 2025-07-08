# Homeostasis: A Framework for Self-Healing Systems

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

<img src="header.png" alt="Homeostasis Header" width="600">

## Vision

Homeostasis is an open-source framework that mimics biological self-repair mechanisms in software systems. Just as living organisms heal injuries autonomously, Homeostasis enables applications to detect failures, generate fixes, and deploy them without human intervention.

## Goals

- **Reduce Downtime**: Automatically fix errors before they impact users
- **Decrease Manual Repairs**: Free developers from repetitive bug-fixing
- **Universal Framework**: Support multiple languages and platforms
- **Safety First**: Ensure all generated fixes are thoroughly tested
- **Open and Extensible**: Foster community-driven healing strategies

## Architecture Overview

Homeostasis implements a self-healing cycle through six interconnected modules:

1. **Monitoring & Error Collection**: Captures logs, exceptions, and performance metrics
2. **Root Cause Analysis (RCA)**: Identifies underlying error causes using rules or ML
3. **Code Generation & Patching**: Creates fixes through templates or AI assistance
4. **Parallel Environment Deployment**: Tests new code in isolated environments
5. **Validation & Observation**: Verifies fixes through comprehensive test suites
6. **Hot Swap / Replacement**: Safely replaces broken components with fixed versions

For detailed architecture diagrams, see [Homeostasis Architecture](docs/assets/architecture-diagram.txt)

## Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/thehalvo/homeostasis.git
   cd homeostasis
   ```

2. Install the package and dependencies:
   ```bash
   pip install -e .
   # For development dependencies
   pip install -e ".[dev]"
   ```

### Setting Up API Keys

Homeostasis integrates with multiple LLM providers for AI-assisted error analysis and code generation. The system provides secure multi-provider key management with fallback support and unified access patterns:

> **For detailed API key configuration, see [API Keys Guide](docs/api_keys.md)**

#### Basic Key Management

```bash
# Set API key for OpenAI (interactive with validation)
homeostasis set-key openai

# Set API key for Anthropic
homeostasis set-key anthropic

# Set API key for OpenRouter (unified endpoint for multiple providers)
homeostasis set-key openrouter

# List all configured providers with status
homeostasis list-keys --show-masked --verbose

# Validate a specific provider's configuration
homeostasis validate-key openai

# Test connectivity for all configured providers
homeostasis test-providers
```

#### Multi-Provider Configuration

```bash
# Set active provider (or 'auto' for intelligent selection)
homeostasis set-active-provider anthropic

# Configure fallback order for automatic provider switching
homeostasis set-fallback-order anthropic openai openrouter

# Enable/disable automatic fallback on provider failures
homeostasis set-fallback-enabled true

# Configure provider selection policies
homeostasis set-provider-policies --cost balanced --latency low --reliability high

# Enable OpenRouter as unified endpoint (proxy to other providers)
homeostasis set-openrouter-unified true --proxy-anthropic --proxy-openai

# View complete multi-provider status
homeostasis provider-status --verbose
```

#### Security Features

- **PBKDF2 + Fernet Encryption**: Local storage uses 100,000 iterations for key derivation
- **External Secrets Integration**: Automatic discovery and integration with AWS Secrets Manager, Azure Key Vault, and HashiCorp Vault
- **Hierarchical Key Lookup**: Environment variables → External secrets → Encrypted local storage
- **Format Validation**: Provider-specific key format checking with correction suggestions
- **API Validation**: Live endpoint testing during key setup
- **Secure Display**: Keys are masked in all CLI output and logs

#### Storage Options

**Environment Variables** (highest priority):
```bash
export HOMEOSTASIS_OPENAI_API_KEY="sk-..."
export HOMEOSTASIS_ANTHROPIC_API_KEY="sk-ant-..."
export HOMEOSTASIS_OPENROUTER_API_KEY="sk-or-..."
```

**External Secrets Managers** (auto-detected):
```bash
# AWS Secrets Manager
export AWS_DEFAULT_REGION="us-east-1"

# Azure Key Vault  
export AZURE_KEY_VAULT_URL="https://vault-name.vault.azure.net/"

# HashiCorp Vault
export VAULT_ADDR="https://vault-server:8200"
export VAULT_TOKEN="hvs.token..."
```

**Encrypted Local Storage**: `~/.homeostasis/llm_keys.enc` with password-based encryption

#### Provider Information

- **OpenAI**: Keys start with `sk-` (51+ characters), requires account with credits
- **Anthropic**: Keys start with `sk-ant-` (90+ characters), requires API access
- **OpenRouter**: Keys start with `sk-or-` (60+ characters), can proxy requests to other providers

#### Web Dashboard Key Management

Homeostasis provides a web-based interface for API key management accessible through the configuration panel:

```bash
# Start the dashboard server
homeostasis dashboard --port 5000

# Open the configuration panel
open http://localhost:5000/config
```

**Dashboard Features:**
- **Visual Key Status**: Real-time indicators showing which keys are configured and their validation status
- **Secure Input Forms**: Password-masked input fields with toggle visibility for all three providers
- **Live Validation**: Immediate feedback on key format and API connectivity during entry
- **Source Indicators**: Visual badges showing key sources (Environment, External Secrets, Encrypted Storage)
- **Bulk Testing**: Test all configured providers simultaneously with detailed results
- **Synchronized Management**: Changes made in dashboard are immediately reflected in CLI and vice versa
- **Provider Configuration**: Set default provider, configure failover order, and enable automatic switching
- **Security Features**: All keys are masked in display, never stored in plaintext, and validated before storage

**Key Management Workflow:**
1. Navigate to Configuration → LLM Keys tab
2. Enter API keys in provider-specific input fields
3. Test keys individually or all at once for validation
4. Configure provider preferences and failover settings
5. Save configuration - keys are encrypted and stored securely
6. Monitor key sources and status through visual indicators

#### Advanced Features

- **Intelligent Provider Selection**: Automatic provider choice based on cost, latency, and reliability policies  
- **Seamless Fallback**: Automatic switching between providers on errors or rate limits
- **OpenRouter Unified Mode**: Use OpenRouter as a single endpoint for multiple AI providers
- **Key Rotation Support**: Update keys without service interruption through CLI or dashboard
- **Usage Monitoring**: Track provider performance and costs
- **Cross-Interface Sync**: Keys set via CLI appear in dashboard and vice versa

### Running the Demo

Homeostasis includes a demo service with intentional bugs that the framework can automatically fix:

```bash
# Make the demo script executable
chmod +x demo.sh

# Run the demo script to see self-healing in action
./demo.sh
```

If you encounter any issues, you can run the commands manually:

```bash
# Create necessary directories
mkdir -p logs logs/patches logs/backups sessions

# Install additional required dependencies that might be missing
pip install pyyaml requests

# Run the orchestrator in demo mode
python3 orchestrator/orchestrator.py --demo
```

The demo will:
1. Start the example FastAPI service with known bugs
2. Monitor for errors
3. Analyze error root causes
4. Generate and apply patches
5. Test the fixes
6. Restart the service with applied patches

#### Troubleshooting Demo Issues

- If you get permission errors, make sure the demo script is executable with `chmod +x demo.sh`
- If you get "command not found" errors, try running with `python3` explicitly
- Ensure all dependencies are installed with `pip install -e ".[dev]"`
- If you're using a Mac with Apple Silicon (M1/M2/M3), you might need to install additional dependencies: `pip install pytest-asyncio httpx`

### Using Homeostasis in Your Project

1. Configure your project in `orchestrator/config.yaml`:
   ```yaml
   service:
     path: "path/to/your/service"
     start_command: "your service start command"
     health_check_url: "http://localhost:your_port/health"
   ```

2. Run the orchestrator:
   ```bash
   python orchestrator/orchestrator.py
   ```

3. The orchestrator will:
   - Monitor your service logs for errors
   - Analyze detected errors using rule-based analysis
   - Generate and apply patches based on identified issues
   - Test and validate the fixes
   - Restart your service with the fixes applied

### Command Line Options

```bash
# Run with custom config file
python orchestrator/orchestrator.py --config custom_config.yaml

# Run in demo mode
python orchestrator/orchestrator.py --demo

# Roll back the latest applied patches
python orchestrator/orchestrator.py --rollback

# Set custom log level
python orchestrator/orchestrator.py --log-level DEBUG
```

### Known Issues

When running the demo for the first time, you might encounter these issues:

1. **Missing YAML and Requests dependencies**
   - Solution: The demo script will attempt to install them automatically
   - If that fails, install them manually: `pip install pyyaml requests`

2. **Missing directories**
   - Solution: The demo script creates logs/, logs/patches/, logs/backups/, and sessions/ directories
   - If that fails, create them manually: `mkdir -p logs logs/patches logs/backups sessions`

## Supported Languages

Homeostasis currently provides support for the following programming languages:

- **Python**: Primary language with extensive support for most Python frameworks
- **JavaScript**: Support for Node.js environments, browser environments, and common JavaScript errors
- **TypeScript**: Full TypeScript support including compilation errors, type system validation, JSX/TSX handling, and framework integration 
- **React**: Support for React component lifecycle errors, hooks validation, state management (Redux, Context), JSX issues, and performance optimization
- **Vue.js**: Support for Vue component and directive errors, Vuex state management, Composition API, Vue Router navigation, and Vue 3 features
- **Angular**: Support for Angular dependency injection errors, NgRx state management issues, template binding problems, module configuration, and Angular Universal SSR
- **Svelte**: Support for Svelte component reactivity errors, SvelteKit routing and SSR issues, store management problems, transition and animation errors, and compiler optimization
- **Next.js**: Support for Next.js data fetching methods, API routes, App Router and Pages Router, image optimization, middleware configuration, and Vercel deployment issues
- **Ember.js**: Support for Ember components, template errors, Ember Data store, router transitions, Octane features, tracked properties, and modifiers
- **Web Components**: Support for Custom Elements lifecycle, Shadow DOM operations, HTML templates, slot distribution, framework interoperability, and libraries like Lit and Stencil
- **CSS Frameworks**: Support for Tailwind CSS utility class validation, CSS-in-JS libraries (Styled Components, Emotion), CSS Modules, SASS/LESS preprocessing, CSS Grid and Flexbox layouts, animations and transitions
- **Java**: Support for Java exceptions, Spring Framework, Hibernate/JPA, and concurrency issues
- **Go**: Support for Go runtime errors, goroutine management, and web frameworks like Gin and Echo
- **Ruby**: Support for Ruby exceptions, Rails ActiveRecord, Sinatra/Rack, and metaprogramming patterns
- **Rust**: Support for Rust runtime errors, memory safety issues, and frameworks like Actix, Rocket, and Tokio
- **C#**: Support for .NET exceptions, ASP.NET Core, Entity Framework, and async patterns
- **PHP**: Support for PHP errors, Laravel and Symfony frameworks, and database interaction issues
- **Scala**: Support for Scala errors, functional programming patterns, Akka actor system, and Play Framework
- **Elixir/Erlang**: Support for Elixir errors, Phoenix Framework, Ecto database errors, and OTP supervision tree issues
- **Clojure**: Support for Clojure JVM errors, functional programming patterns, Ring/Compojure frameworks, and core.async concurrency issues
- **Swift**: Support for iOS, macOS, watchOS, and tvOS applications with UIKit and SwiftUI frameworks, Core Data persistence, memory management, concurrency, and Swift Package Manager
- **Kotlin**: Support for Android, JVM, JavaScript, and Native platforms with null safety validation, coroutine management, Android lifecycle handling, Jetpack Compose UI framework, Room database persistence, and multiplatform development
- **React Native**: Support for React Native mobile applications with Metro bundler error resolution, native module integration, iOS/Android build issues, bridge communication, and platform-specific deployment
- **Flutter**: Support for Flutter cross-platform applications with Dart language error handling, widget lifecycle management, layout overflow resolution, state management, and mobile/web/desktop deployment
- **Xamarin**: Support for Xamarin.Forms, Xamarin.iOS, and Xamarin.Android applications with MVVM pattern validation, DependencyService integration, custom renderer handling, and platform binding resolution
- **Unity**: Support for Unity game development with C# scripting error detection, GameObject/Component lifecycle management, mobile build configuration, Unity UI handling, and performance optimization
- **Capacitor/Cordova**: Support for hybrid mobile applications with plugin integration, native bridge communication, WebView configuration, Content Security Policy handling, and cross-platform deployment
- **SQL**: Support for database query errors across PostgreSQL, MySQL, SQLite, SQL Server, and Oracle with syntax validation, constraint violations, and performance optimization
- **Bash/Shell**: Support for shell scripting errors across Bash, Zsh, Fish, and other shells with syntax validation, command resolution, and script execution issues
- **YAML/JSON**: Support for configuration file validation across Kubernetes, Docker Compose, Ansible, GitHub Actions, GitLab CI, and other tools with syntax checking and structure validation
- **Terraform**: Support for infrastructure-as-code errors with multi-provider support for AWS, Azure, Google Cloud, Kubernetes, and other platforms including resource configuration and state management
- **Dockerfile**: Support for container configuration errors with Docker build issues, multi-stage builds, security best practices, and optimization recommendations
- **Ansible**: Support for configuration management errors including playbook syntax, module usage, inventory management, variable templating, and role dependencies
- **Zig**: Support for systems programming errors including memory safety issues, compile-time evaluation, async programming patterns, optional types, and error unions

For more information about language integrations, see:
- [Python Integration](docs/python_integration.md)
- [JavaScript Integration](docs/javascript_integration.md)
- [TypeScript Integration](docs/typescript_integration.md)
- [React Integration](docs/react_integration.md)
- [Vue Integration](docs/vue_integration.md)
- [Angular Integration](docs/angular_integration.md)
- [Svelte Integration](docs/svelte_integration.md)
- [Next.js Integration](docs/nextjs_integration.md)
- [Ember.js Integration](docs/ember_integration.md)
- [Web Components Integration](docs/web_components_integration.md)
- [Java Integration](docs/java_integration.md)
- [Android Integration](docs/android_integration.md)
- [Go Integration](docs/go_integration.md)
- [Ruby Integration](docs/ruby_integration.md)
- [Rust Integration](docs/rust_integration.md)
- [C# Integration](docs/csharp_integration.md)
- [PHP Integration](docs/php_integration.md)
- [Scala Integration](docs/scala_integration.md)
- [Elixir/Erlang Integration](docs/elixir_integration.md)
- [Clojure Integration](docs/clojure_integration.md)
- [Swift Integration](docs/swift_integration.md)
- [Kotlin Integration](docs/kotlin_integration.md)
- [React Native Integration](docs/react_native_integration.md)
- [Flutter Integration](docs/flutter_integration.md)
- [Xamarin Integration](docs/xamarin_integration.md)
- [Unity Integration](docs/unity_integration.md)
- [Capacitor/Cordova Integration](docs/capacitor_cordova_integration.md)
- [SQL Integration](docs/sql_integration.md)
- [Bash/Shell Integration](docs/bash_integration.md)
- [YAML/JSON Integration](docs/yaml_json_integration.md)
- [Terraform Integration](docs/terraform_integration.md)
- [Dockerfile Integration](docs/dockerfile_integration.md)
- [Ansible Integration](docs/ansible_integration.md)
- [Zig Integration](docs/zig_integration.md)

## Recent Updates

- **Multi-Language Support**: Added support for 15+ programming languages including Python, JavaScript, TypeScript, Java, Go, Rust, Swift, Kotlin, and more
- **Framework Integration**: Built specialized support for React, Vue, Angular, Django, FastAPI, Spring, and 20+ other frameworks  
- **LLM Integration**: Completed secure multi-provider integration with OpenAI, Anthropic, and OpenRouter
- **Intelligent Retry & Failover**: Implemented advanced retry strategies with exponential backoff, automatic multi-provider failover, and test re-validation for enhanced LLM reliability
- **IDE Extensions**: Released plugins for VS Code and JetBrains IDEs with real-time error healing
- **CI/CD Integration**: Added support for GitHub Actions, GitLab CI, Jenkins, and CircleCI
- **Production Features**: Implemented RBAC security, approval workflows, audit logging, and Kubernetes operator

## Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) to get started.

## Troubleshooting

### Common Issues

1. **Demo script fails to run:**
   - Make sure the script is executable: `chmod +x demo.sh`
   - Try running with explicit Python path: `python3 orchestrator/orchestrator.py --demo`
   - Ensure all required directories exist: `mkdir -p logs logs/patches logs/backups sessions`

2. **Import errors:**
   - Ensure you're in the activated virtual environment (`source venv/bin/activate`)
   - Verify all dependencies are installed: `pip install -e ".[dev]"`
   - Install missing dependencies manually: `pip install pyyaml requests`
   - Try installing specific dependencies explicitly: `pip install fastapi uvicorn pydantic loguru pytest pyyaml requests`

3. **Service startup issues:**
   - Check if port 8000 is already in use: `lsof -i :8000`
   - Kill any process using the port: `kill -9 $(lsof -ti:8000)`
   - Verify the example service is configured correctly in `orchestrator/config.yaml`

4. **Failed to generate patches:**
   - Ensure all template directories exist: `ls modules/patch_generation/templates/`
   - Check if rule matching is working: `ls modules/analysis/rules/`

5. **Platform-specific issues:**
   - Mac with Apple Silicon (M1/M2/M3): You might need additional dependencies
   - Windows: Adjust path separators and commands in configuration files

For more detailed help, please check the documentation in the `docs/` directory or open an issue on GitHub.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
