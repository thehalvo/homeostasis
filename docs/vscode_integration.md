# Visual Studio Code Integration

The Homeostasis VSCode extension brings self-healing capabilities directly into your development environment, providing real-time error detection, intelligent fix suggestions, and seamless integration with the Homeostasis healing ecosystem.

## Overview

The VSCode extension serves as a bridge between your local development environment and the Homeostasis healing server, offering:

- **Real-time Analysis**: Code is continuously analyzed as you type
- **Intelligent Suggestions**: Context-aware fix suggestions with confidence scores
- **Proactive Prevention**: Code lens indicators for potential issues
- **Cross-device Sync**: Settings synchronization across development environments
- **Privacy-first Telemetry**: Optional, anonymous usage analytics

## Features

### 🔧 Real-time Healing

The extension monitors your code in real-time, automatically detecting potential issues and offering fixes as you write code. This includes:

- Syntax errors and typos
- Common programming mistakes
- Framework-specific issues
- Performance anti-patterns
- Security vulnerabilities

### 💡 Inline Fix Suggestions

When issues are detected, the extension provides quick fixes through VSCode's built-in code actions system:

```typescript
// Example: Missing null check
const user = getUser(id);
console.log(user.name); // ⚡ Quick fix available

// After applying fix:
const user = getUser(id);
if (user) {
    console.log(user.name);
}
```

### 🎯 Code Lens Integration

Visual indicators appear above potentially problematic code:

```python
# 💡 Improve error handling
try:
    process_data()
except Exception:
    pass

# 🛡️ Strengthen assertion  
assert data is not None

# 🎯 Track and resolve
# TODO: Optimize this algorithm
```

### 📊 Multi-language Support

Currently supports:
- **Backend**: Python, Java, Go, Rust, C#, PHP, Ruby, Scala, Elixir, Clojure
- **Frontend**: JavaScript, TypeScript, React, Vue, Angular, Svelte
- **Mobile**: Swift, Kotlin, Dart (Flutter)
- **Web**: HTML, CSS, Web Components

### 🔗 Settings Synchronization

Keep your Homeostasis preferences synchronized across all development environments:

- **Profiles**: Create named settings profiles for different projects
- **Cross-device Sync**: Automatic synchronization when connected to server
- **Import/Export**: Backup and share settings with team members
- **Conflict Resolution**: Smart merging when settings differ between devices

## Installation

### From VSCode Marketplace

1. Open VSCode
2. Navigate to Extensions (`Ctrl+Shift+X`)
3. Search for "Homeostasis Self-Healing"
4. Click "Install"

### From Source

1. Clone the Homeostasis repository
2. Navigate to `ide/vscode-extension/`
3. Run `npm install && npm run package-extension`
4. Install the generated `.vsix` file in VSCode

## Configuration

### Basic Setup

1. **Server Connection**:
   ```json
   {
     "homeostasis.serverUrl": "http://localhost:8080",
     "homeostasis.apiKey": "your-api-key-here"
   }
   ```

2. **Language Preferences**:
   ```json
   {
     "homeostasis.enabledLanguages": [
       "python", "javascript", "typescript", "java", "go"
     ]
   }
   ```

3. **Healing Behavior**:
   ```json
   {
     "homeostasis.realTimeHealing": true,
     "homeostasis.healingDelay": 2000,
     "homeostasis.confidenceThreshold": 0.7
   }
   ```

### Advanced Configuration

#### Workspace-specific Settings

Create `.vscode/settings.json` in your project:

```json
{
  "homeostasis.enabledLanguages": ["python", "javascript"],
  "homeostasis.confidenceThreshold": 0.8,
  "homeostasis.healingDelay": 1000
}
```

#### Team Settings

Share settings across your team by committing configuration:

```json
{
  "homeostasis.serverUrl": "https://healing.yourcompany.com",
  "homeostasis.enabledLanguages": ["python", "typescript"],
  "homeostasis.showInlineHints": true,
  "homeostasis.enableCodeLens": true
}
```

## Commands

Access these commands through the Command Palette (`Ctrl+Shift+P`):

| Command | Description |
|---------|-------------|
| `Homeostasis: Heal Current File` | Apply fixes to the active file |
| `Homeostasis: Heal Entire Workspace` | Scan and heal all supported files |
| `Homeostasis: Enable Real-time Healing` | Turn on automatic healing |
| `Homeostasis: Disable Real-time Healing` | Turn off automatic healing |
| `Homeostasis: Show Healing Dashboard` | Open web dashboard |
| `Homeostasis: Sync Settings` | Sync with server |
| `Homeostasis: Create Settings Profile` | Save current settings as profile |
| `Homeostasis: Load Settings Profile` | Load a saved settings profile |
| `Homeostasis: Export Settings` | Export settings to JSON |
| `Homeostasis: Import Settings` | Import settings from JSON |

## Usage Examples

### Python Development

```python
# Before healing
def process_user(user_id):
    user = users[user_id]  # ⚠️ KeyError risk
    return user.name.upper()  # ⚠️ AttributeError risk

# After healing (auto-applied)
def process_user(user_id):
    user = users.get(user_id)
    if user and user.name:
        return user.name.upper()
    return None
```

### JavaScript/TypeScript

```typescript
// Before healing
async function fetchUserData(id: string) {
    const response = await fetch(`/api/users/${id}`);
    return response.json(); // ⚠️ No error handling
}

// After healing
async function fetchUserData(id: string) {
    try {
        const response = await fetch(`/api/users/${id}`);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        return await response.json();
    } catch (error) {
        console.error('Failed to fetch user data:', error);
        throw error;
    }
}
```

### React Components

```jsx
// Before healing
function UserProfile({ userId }) {
    const [user, setUser] = useState();
    
    useEffect(() => {
        fetchUser(userId).then(setUser);
    }); // ⚠️ Missing dependency array
    
    return <div>{user.name}</div>; // ⚠️ Potential null reference
}

// After healing
function UserProfile({ userId }) {
    const [user, setUser] = useState(null);
    
    useEffect(() => {
        fetchUser(userId).then(setUser);
    }, [userId]); // ✅ Proper dependencies
    
    return <div>{user?.name || 'Loading...'}</div>; // ✅ Safe access
}
```

## Workflow Integration

### Git Hooks

The extension can integrate with your Git workflow:

```bash
# Pre-commit hook
#!/bin/sh
code --command homeostasis.healWorkspace
git add -A
```

### CI/CD Integration

Include healing in your build pipeline:

```yaml
# GitHub Actions example
- name: Heal codebase
  run: |
    code --install-extension homeostasis.homeostasis-healing
    code --command homeostasis.healWorkspace
    git diff --exit-code || echo "Healing applied changes"
```

## Privacy and Security

### Data Handling

- **Local Processing**: Analysis can run locally when possible
- **Encrypted Communication**: All server communication uses HTTPS
- **No Code Transmission**: Only metadata and error patterns are sent
- **Opt-in Telemetry**: All telemetry is optional and anonymous

### Configuration

```json
{
  "homeostasis.enableTelemetry": false,
  "homeostasis.localProcessingOnly": true,
  "homeostasis.dataRetentionDays": 30
}
```

## Troubleshooting

### Common Issues

1. **Extension not activating**
   - Check VSCode version (requires 1.74.0+)
   - Verify installation completed successfully
   - Check Developer Console for errors

2. **Server connection failed**
   - Verify Homeostasis server is running
   - Check firewall and network settings
   - Validate API key if authentication is required

3. **No suggestions appearing**
   - Check if language is enabled in settings
   - Verify confidence threshold isn't too high
   - Ensure real-time healing is enabled

4. **Performance issues**
   - Increase healing delay in settings
   - Disable languages you don't use
   - Consider local-only processing

### Debug Mode

Enable detailed logging:

```json
{
  "homeostasis.debug": true,
  "homeostasis.logLevel": "verbose"
}
```

## Contributing

The VSCode extension is part of the Homeostasis project. To contribute:

1. Fork the repository
2. Navigate to `ide/vscode-extension/`
3. Follow the build instructions in `BUILD.md`
4. Submit pull requests with your improvements

## Support

- **Documentation**: [docs.homeostasis.dev](https://docs.homeostasis.dev)
- **Issues**: [GitHub Issues](https://github.com/homeostasis/homeostasis/issues)
- **Discussions**: [GitHub Discussions](https://github.com/homeostasis/homeostasis/discussions)
- **Community**: [Discord Server](https://discord.gg/homeostasis)

## Roadmap

Upcoming features:

- **Collaborative Healing**: Share fixes with team members
- **Custom Rules**: Define project-specific healing rules
- **AI Training**: Local model training on your codebase
- **Refactoring Suggestions**: Large-scale code improvements
- **Performance Analysis**: Runtime performance optimization

## License

The VSCode extension is licensed under the same terms as the main Homeostasis project.