# PowerShell Integration

Homeostasis provides full support for PowerShell, a task automation and configuration management framework from Microsoft. This integration handles PowerShell's unique features including cmdlets, pipeline operations, object-oriented data handling, and Windows system integration.

## Overview

The PowerShell integration includes:
- **Syntax Error Detection**: Parse errors, cmdlet syntax issues, and language-specific syntax validation
- **Cmdlet Management**: Parameter validation, pipeline operations, and cmdlet execution errors
- **Object Handling**: .NET object manipulation, property access, and method invocation
- **Module Management**: Import/export issues, dependency resolution, and version conflicts
- **System Integration**: Registry access, WMI operations, and Windows-specific functionality

## Supported Error Types

### Syntax Errors
- Parse errors and unexpected tokens
- Invalid cmdlet syntax and parameters
- String interpolation and variable issues
- Script block and function definition errors

### Cmdlet Errors
- Parameter validation failures
- Pipeline operation issues
- Cmdlet not found errors
- Execution policy violations

### Object Handling
- .NET object access errors
- Property and method invocation failures
- Type casting and conversion issues
- Null reference exceptions

### Module Management
- Module import/export failures
- Dependency resolution issues
- Version compatibility problems
- Path resolution errors

### System Integration
- Registry access violations
- WMI query failures
- File system permission issues
- Service management errors

## Configuration

### Basic Setup

```powershell
# example.ps1
# Function with error handling
function Get-SafeValue {
    param(
        [Parameter(Mandatory=$true)]
        [string]$InputValue,
        
        [string]$DefaultValue = "Unknown"
    )
    
    try {
        if ([string]::IsNullOrEmpty($InputValue)) {
            return $DefaultValue
        }
        return $InputValue.Trim()
    }
    catch {
        Write-Error "Error processing value: $_"
        return $DefaultValue
    }
}

# Pipeline operations
Get-Process | Where-Object { $_.CPU -gt 100 } | Select-Object Name, CPU

# Object handling
$obj = New-Object PSObject -Property @{
    Name = "Example"
    Value = 42
}
```

### Error Handling Patterns

**Try-Catch-Finally:**
```powershell
# Structured error handling
try {
    $result = Get-WmiObject -Class Win32_OperatingSystem
    Write-Output "OS: $($result.Caption)"
}
catch [System.Management.ManagementException] {
    Write-Error "WMI error: $($_.Exception.Message)"
}
catch {
    Write-Error "General error: $($_.Exception.Message)"
}
finally {
    Write-Output "Cleanup completed"
}
```

**Error Action Preference:**
```powershell
# Setting error behavior
$ErrorActionPreference = "Stop"

# Per-cmdlet error handling
Get-Process -Name "nonexistent" -ErrorAction SilentlyContinue
if ($?) {
    Write-Output "Process found"
} else {
    Write-Output "Process not found"
}
```

**Parameter Validation:**
```powershell
# Function with parameter validation
function Test-FileExists {
    param(
        [Parameter(Mandatory=$true)]
        [ValidateScript({Test-Path $_ -PathType Leaf})]
        [string]$FilePath
    )
    
    return $true
}
```

## Common Fix Patterns

### Null Checking
```powershell
# Before (unsafe)
$value.Length

# After (safe)
if ($value -ne $null) {
    $value.Length
} else {
    0
}
```

### Pipeline Error Handling
```powershell
# Before (no error handling)
Get-Process | ForEach-Object { $_.Kill() }

# After (with error handling)
Get-Process | ForEach-Object {
    try {
        $_.Kill()
        Write-Output "Killed process: $($_.Name)"
    }
    catch {
        Write-Warning "Failed to kill process $($_.Name): $($_.Exception.Message)"
    }
}
```

### Registry Access
```powershell
# Before (unsafe)
$value = Get-ItemProperty -Path "HKLM:\SOFTWARE\MyApp" -Name "Version"

# After (safe)
try {
    if (Test-Path "HKLM:\SOFTWARE\MyApp") {
        $value = Get-ItemProperty -Path "HKLM:\SOFTWARE\MyApp" -Name "Version" -ErrorAction Stop
    } else {
        Write-Warning "Registry key not found"
        $value = $null
    }
}
catch {
    Write-Error "Registry access failed: $($_.Exception.Message)"
    $value = $null
}
```

## Best Practices

1. **Use Proper Error Handling**: Implement try-catch blocks for risky operations
2. **Validate Parameters**: Use parameter validation attributes
3. **Check Execution Policy**: Ensure scripts can run in the target environment
4. **Handle Null Values**: Always check for null before accessing properties
5. **Use Approved Verbs**: Follow PowerShell verb naming conventions

## Framework Support

The PowerShell integration supports popular PowerShell modules and frameworks:
- **Active Directory**: AD cmdlet error handling
- **Exchange**: Exchange management shell support
- **Azure**: Azure PowerShell module integration
- **SharePoint**: SharePoint PowerShell cmdlets
- **System Center**: SCCM and SCOM PowerShell support

## Error Examples

### Syntax Error
```powershell
# Error: Missing closing brace
function Test-Function {
    Write-Output "Hello"

# Fix: Add closing brace
function Test-Function {
    Write-Output "Hello"
}
```

### Cmdlet Error
```powershell
# Error: Invalid parameter
Get-Process -InvalidParameter "value"

# Fix: Use correct parameter
Get-Process -Name "notepad"
```

### Object Access Error
```powershell
# Error: Property access on null object
$null.Length

# Fix: Check for null
if ($obj -ne $null) {
    $obj.Length
}
```

## Advanced Features

### Custom Error Classes
```powershell
# Custom error handling
class CustomError : System.Exception {
    [string]$ErrorCode
    
    CustomError([string]$message, [string]$code) : base($message) {
        $this.ErrorCode = $code
    }
}

function Invoke-RiskyOperation {
    try {
        # Risky operation
    }
    catch {
        throw [CustomError]::new("Operation failed", "ERR001")
    }
}
```

### Advanced Parameter Validation
```powershell
# Complex parameter validation
function Test-ComplexValidation {
    param(
        [Parameter(Mandatory=$true)]
        [ValidatePattern('^[A-Z]{2,3}-\d{3,4}$')]
        [string]$Code,
        
        [Parameter()]
        [ValidateRange(1, 100)]
        [int]$Percentage = 50,
        
        [Parameter()]
        [ValidateSet("Development", "Test", "Production")]
        [string]$Environment = "Development"
    )
    
    # Function implementation
}
```

### Module Error Handling
```powershell
# Safe module importing
function Import-ModuleSafely {
    param([string]$ModuleName)
    
    try {
        if (Get-Module -Name $ModuleName -ListAvailable) {
            Import-Module $ModuleName -ErrorAction Stop
            Write-Output "Module $ModuleName imported successfully"
        } else {
            Write-Warning "Module $ModuleName not available"
        }
    }
    catch {
        Write-Error "Failed to import module $ModuleName: $($_.Exception.Message)"
    }
}
```

## Integration Testing

The PowerShell integration includes extensive testing:

```bash
# Run PowerShell plugin tests
python -m pytest tests/test_powershell_plugin.py -v

# Test specific error types
python -m pytest tests/test_powershell_plugin.py::TestPowerShellExceptionHandler::test_analyze_cmdlet_error -v
```

## Performance Considerations

- **Pipeline Efficiency**: Use efficient pipeline operations
- **Memory Management**: Be aware of object retention in pipelines
- **Module Loading**: Load modules only when needed
- **Remote Operations**: Optimize remote PowerShell sessions

## Troubleshooting

### Common Issues

1. **Execution Policy**: Ensure scripts can execute in the target environment
2. **Module Dependencies**: Verify all required modules are installed
3. **Permission Issues**: Check user permissions for system operations
4. **Version Compatibility**: Ensure compatibility across PowerShell versions

### Debug Commands

```powershell
# Check PowerShell version
$PSVersionTable

# Get execution policy
Get-ExecutionPolicy

# Debug script execution
Set-PSDebug -Trace 1
```

## Related Documentation

- [Error Schema](error_schema.md)
- [Plugin Architecture](plugin_architecture.md)
- [Best Practices](best_practices.md)
- [Integration Guides](integration_guides.md)