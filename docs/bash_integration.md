# Bash/Shell Integration

The Homeostasis Bash/Shell Language Plugin provides error analysis and patch generation for shell scripts and command-line operations. It supports multiple shell environments and provides intelligent error detection for common shell scripting issues.

## Overview

The Bash/Shell plugin enables Homeostasis to:
- Analyze shell script syntax errors across multiple shell types
- Detect and fix command execution failures
- Handle variable and parameter issues
- Provide intelligent suggestions for script optimization
- Support shell-specific error patterns and best practices

## Supported Shell Environments

- **Bash** - Bourne Again Shell (most common)
- **Zsh** - Z Shell with advanced features
- **Fish** - Friendly Interactive Shell
- **Dash** - Debian Almquist Shell (POSIX)
- **Ksh** - Korn Shell
- **Csh/Tcsh** - C Shell variants
- **Sh** - POSIX-compliant shell
- **Ash** - Almquist Shell
- **BusyBox** - Embedded shell

## Key Features

### Error Detection Categories

1. **Syntax Errors**
   - Invalid shell syntax
   - Unmatched quotes or brackets
   - Missing keywords or delimiters
   - Command substitution errors

2. **Command Errors**
   - Command not found
   - Permission denied
   - Execution failures
   - Path resolution issues

3. **Variable Errors**
   - Undefined variables
   - Parameter expansion issues
   - Bad substitution
   - Variable scope problems

4. **File System Errors**
   - File not found
   - Permission issues
   - Directory access problems
   - Path validation

5. **I/O Redirection Errors**
   - Invalid redirection syntax
   - File descriptor issues
   - Pipe failures
   - Ambiguous redirects

### Exit Code Analysis

The plugin provides exit code interpretation:

```bash
# Common exit codes and their meanings
0   - Success
1   - General errors
2   - Misuse of shell builtins
126 - Command invoked cannot execute
127 - Command not found
128+- Fatal error signals (SIGTERM, SIGKILL, etc.)
```

## Usage Examples

### Basic Shell Error Analysis

```python
from homeostasis import analyze_error

# Example shell error
error_data = {
    "error_type": "ShellError",
    "message": "command not found: git",
    "shell_type": "bash",
    "exit_code": 127,
    "command": "git status"
}

analysis = analyze_error(error_data, language="bash")
print(analysis["suggested_fix"])
# Output: "Check if command exists and is in PATH"
```

### Syntax Error Detection

```python
# Shell syntax error
syntax_error = {
    "error_type": "SyntaxError",
    "message": "syntax error near unexpected token ')'",
    "shell_type": "bash",
    "script_content": "if [ $var == 'test' ) then..."
}

analysis = analyze_error(syntax_error, language="bash")
```

### Variable Error Handling

```python
# Undefined variable error
variable_error = {
    "error_type": "VariableError", 
    "message": "MY_VAR: unbound variable",
    "shell_type": "bash",
    "line_number": 15
}

analysis = analyze_error(variable_error, language="bash")
```

## Configuration

### Plugin Configuration

Configure the Bash plugin in your `homeostasis.yaml`:

```yaml
plugins:
  bash:
    enabled: true
    supported_shells: [bash, zsh, fish, sh, dash]
    error_detection:
      syntax_checking: true
      command_validation: true
      variable_checking: true
      permission_checking: true
    patch_generation:
      auto_suggest_fixes: true
      best_practices: true
      security_checks: true
```

### Shell-Specific Settings

```yaml
plugins:
  bash:
    bash:
      version: "4.0+"
      strict_mode: false
    zsh:
      version: "5.0+"
      oh_my_zsh: true
    fish:
      version: "3.0+"
      universal_variables: true
```

## Error Pattern Recognition

### Syntax Error Patterns

```bash
# Unmatched quotes
echo "Hello World
# Fix: Add closing quote

# Missing semicolon in compound command
if [ $var = "test" ] then echo "found"; fi
# Fix: Add semicolon or use proper if syntax

# Invalid command substitution
result = `command with spaces`
# Fix: Use $() or proper quoting
```

### Command Error Patterns

```bash
# Command not found
gti status
# Suggestion: Check spelling (git status)

# Permission denied
./script.sh
# Suggestion: Add execute permission (chmod +x script.sh)

# Path not found
cd /non/existent/path
# Suggestion: Check path exists or create directory
```

### Variable Error Patterns

```bash
# Unbound variable (with set -u)
echo $UNDEFINED_VAR
# Fix: Initialize variable or use ${VAR:-default}

# Bad substitution
echo ${VAR[invalid]}
# Fix: Use proper array syntax ${VAR[0]}

# Parameter expansion error
echo ${VAR:}
# Fix: Complete parameter expansion ${VAR:-default}
```

## Shell-Specific Features

### Bash

- **Associative Arrays**: Advanced array handling
- **Process Substitution**: `<()` and `>()` syntax
- **Extended Glob**: Pattern matching extensions
- **Bash Completion**: Tab completion errors
- **Set Options**: `set -e`, `set -u`, `set -x` handling

### Zsh

- **Extended Globbing**: Zsh-specific glob patterns
- **Parameter Flags**: Advanced parameter expansion
- **Autoload Functions**: Function loading mechanisms
- **Zsh Modules**: Module-specific errors

### Fish

- **Universal Variables**: Fish-specific variable scoping
- **Function Syntax**: Fish function definition syntax
- **Event Handlers**: Fish event system
- **Abbreviations**: Fish abbreviation system

### POSIX Shells

- **POSIX Compliance**: Strict POSIX shell compatibility
- **Limited Features**: Feature limitation warnings
- **Portability**: Cross-shell compatibility checks

## Best Practices

### Script Writing

1. **Use Strict Mode**: `set -euo pipefail` for safer scripts
2. **Quote Variables**: Always quote variable expansions
3. **Check Dependencies**: Verify required commands exist
4. **Error Handling**: Implement proper error checking

```bash
#!/bin/bash
set -euo pipefail

# Good practices
if command -v git >/dev/null 2>&1; then
    git_version=$(git --version)
    echo "Git version: ${git_version}"
else
    echo "Git not found" >&2
    exit 1
fi
```

### Variable Management

```bash
# Safe variable usage
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly CONFIG_FILE="${CONFIG_FILE:-/etc/myapp/config.conf}"

# Check if variable is set
if [[ -n "${DEBUG:-}" ]]; then
    set -x
fi
```

### Function Definition

```bash
# Proper function syntax
cleanup() {
    local exit_code=$?
    # Cleanup logic here
    exit $exit_code
}

# Set trap for cleanup
trap cleanup EXIT
```

### Error Handling

```bash
# Command error handling
if ! command_that_might_fail; then
    echo "Command failed" >&2
    exit 1
fi

# Alternative with explicit check
command_that_might_fail || {
    echo "Command failed" >&2
    exit 1
}
```

## Integration Examples

### CI/CD Pipeline Integration

```yaml
# GitHub Actions example
- name: Run Shell Script with Error Analysis
  run: |
    set -e
    if ! ./scripts/deploy.sh; then
        # Analyze shell errors with Homeostasis
        python -c "
        from homeostasis import analyze_error
        import subprocess
        
        result = subprocess.run(['./scripts/deploy.sh'], 
                              capture_output=True, text=True)
        
        error_data = {
            'error_type': 'ShellError',
            'message': result.stderr,
            'exit_code': result.returncode,
            'command': './scripts/deploy.sh'
        }
        
        analysis = analyze_error(error_data, language='bash')
        print(f'Error Analysis: {analysis[\"suggested_fix\"]}')
        "
        exit 1
    fi
```

### Python Subprocess Integration

```python
import subprocess
from homeostasis import analyze_error

def run_shell_command(command, shell="bash"):
    """Run shell command with error analysis."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            executable=f"/bin/{shell}"
        )
        
        if result.returncode != 0:
            error_data = {
                "error_type": "ShellError",
                "message": result.stderr,
                "exit_code": result.returncode,
                "command": command,
                "shell_type": shell
            }
            
            analysis = analyze_error(error_data, language="bash")
            
            print(f"Command failed: {command}")
            print(f"Error: {result.stderr}")
            print(f"Suggested fix: {analysis['suggested_fix']}")
            
            return None
            
        return result.stdout
        
    except Exception as e:
        print(f"Failed to execute command: {e}")
        return None

# Usage
output = run_shell_command("ls -la /nonexistent")
```

### Shell Script Wrapper

```bash
#!/bin/bash
# wrapper.sh - Shell script with error analysis

# Enable strict mode
set -euo pipefail

# Error handler
error_handler() {
    local exit_code=$?
    local line_number=$1
    
    echo "Error on line $line_number: exit code $exit_code" >&2
    
    # Call Homeostasis for analysis (if Python available)
    if command -v python3 >/dev/null 2>&1; then
        python3 -c "
from homeostasis import analyze_error
import sys

error_data = {
    'error_type': 'ShellError',
    'exit_code': $exit_code,
    'line_number': $line_number,
    'shell_type': 'bash'
}

analysis = analyze_error(error_data, language='bash')
print(f'Suggested fix: {analysis[\"suggested_fix\"]}', file=sys.stderr)
"
    fi
    
    exit $exit_code
}

# Set error trap
trap 'error_handler ${LINENO}' ERR

# Your script logic here
main() {
    echo "Running main script logic..."
    # Script commands here
}

main "$@"
```

## Troubleshooting

### Common Issues

1. **Shell Detection**: Ensure correct shell type is specified
2. **Path Issues**: Check PATH environment variable
3. **Permission Problems**: Verify script execution permissions
4. **Quote Handling**: Proper quoting of variables and strings

### Debug Mode

Enable debug output for shell scripts:

```bash
# Enable debug mode
set -x

# Or for specific sections
{
    set -x
    command_that_needs_debugging
    set +x
}
```

### Syntax Checking

Use shell built-in syntax checking:

```bash
# Check bash syntax
bash -n script.sh

# Check with specific shell
zsh -n script.zsh
fish -n script.fish
```

### Custom Rules

Add custom shell error rules:

```json
{
  "rules": [
    {
      "id": "custom_bash_rule",
      "pattern": "custom error pattern",
      "category": "custom",
      "shell_type": "bash",
      "suggestion": "Custom fix suggestion",
      "confidence": "high"
    }
  ]
}
```

## Performance Considerations

- **Pattern Compilation**: Error patterns are pre-compiled for efficiency
- **Shell Detection**: Fast shell type detection from error context
- **Exit Code Mapping**: Quick lookup for common exit codes
- **Memory Usage**: Minimal memory footprint for rule storage

## Security Considerations

1. **Command Injection**: Avoid dynamic command construction
2. **Path Traversal**: Validate file paths and directories
3. **Privilege Escalation**: Careful handling of sudo/su commands
4. **Environment Variables**: Sanitize environment variable usage

```bash
# Secure practices
readonly SAFE_PATH="/safe/directory"
readonly USER_INPUT="${1//[^a-zA-Z0-9]/}"  # Sanitize input

# Avoid
eval "command $user_input"  # Dangerous

# Use instead
case "$user_input" in
    "valid_option") command --safe-flag ;;
    *) echo "Invalid input" >&2; exit 1 ;;
esac
```

## Contributing

To extend the Bash/Shell plugin:

1. Add new error patterns to `rules/bash/`
2. Implement shell-specific handlers
3. Add test cases for new error types
4. Update documentation with examples

## Related Documentation

- [Error Schema](error_schema.md) - Standard error format
- [Plugin Architecture](plugin_architecture.md) - Plugin development guide
- [CI/CD Integration](cicd/) - Continuous integration setup
- [Best Practices](best_practices.md) - General scripting best practices