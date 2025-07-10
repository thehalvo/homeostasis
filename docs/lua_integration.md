# Lua Integration

Homeostasis provides robust support for Lua programming language, focusing on embedded scripting use cases, game development, and configuration management systems.

## Overview

The Lua integration handles common patterns in embedded scripting including:
- Nil safety and table operations
- Coroutine management and threading
- Module loading and dependency resolution
- Game engine integration patterns
- Configuration script validation

## Supported Error Types

### Syntax Errors
- Unexpected tokens and symbols
- Missing 'end' keywords
- Unfinished string literals
- Invalid syntax constructs

### Runtime Errors
- Nil value access attempts
- Stack overflow from infinite recursion
- Memory allocation failures
- Variable scope issues

### Table Operations
- Table indexing errors
- Nil key access
- Invalid table operations
- Metatable configuration issues

### Function Errors
- Function call failures
- Argument validation issues
- Function definition problems
- Closure and upvalue errors

### Module System
- Module loading failures
- Circular dependency detection
- Package path resolution
- Require pattern issues

### Coroutine Management
- Coroutine state errors
- Yield/resume coordination
- Threading synchronization
- Concurrent access issues

## Common Fix Patterns

### Nil Safety
```lua
-- Before: Unsafe access
local value = object.field

-- After: Safe access with nil check
if object ~= nil then
    local value = object.field
else
    print("Error: object is nil")
end

-- Alternative: Using logical operators
local value = object and object.field or default_value
```

### Table Operations
```lua
-- Before: Unsafe table access
local value = table[key]

-- After: Safe table access
if type(table) == "table" and key ~= nil then
    local value = table[key]
end

-- Safe initialization
table = table or {}
```

### Function Validation
```lua
-- Before: Unsafe function call
func(arg1, arg2)

-- After: Safe function call
if type(func) == "function" then
    func(arg1, arg2)
end

-- Using pcall for error handling
local ok, result = pcall(func, arg1, arg2)
if not ok then
    print("Function call failed:", result)
end
```

### Module Loading
```lua
-- Before: Unsafe require
local module = require("module_name")

-- After: Safe module loading
local ok, module = pcall(require, "module_name")
if not ok then
    print("Failed to load module:", module)
    module = nil
end
```

## Supported Frameworks

### Game Engines
- **LÖVE (Love2D)**: 2D game engine error handling
- **World of Warcraft**: Addon development patterns
- **Wireshark**: Lua dissector scripting
- **Nginx**: OpenResty and lua-resty modules

### Embedded Systems
- **NodeMCU**: IoT device scripting
- **OpenWrt**: Router configuration
- **Redis**: Lua scripting in Redis
- **Nmap**: Network scanning scripts

## Configuration

### Plugin Settings
```yaml
# config.yaml
lua_plugin:
  enabled: true
  frameworks:
    - love2d
    - nginx_lua
    - wireshark
    - world_of_warcraft
  error_detection:
    nil_safety: true
    table_validation: true
    coroutine_monitoring: true
  performance:
    memory_tracking: true
    gc_monitoring: true
```

### Rule Configuration
```json
{
  "lua_rules": {
    "nil_access": {
      "enabled": true,
      "severity": "high",
      "auto_fix": true
    },
    "table_operations": {
      "enabled": true,
      "severity": "medium",
      "auto_fix": true
    },
    "module_loading": {
      "enabled": true,
      "severity": "high",
      "auto_fix": false
    }
  }
}
```

## Best Practices

### Error Handling
1. **Always check for nil** before accessing object properties
2. **Use pcall** for safe function execution
3. **Validate table keys** before access
4. **Initialize tables** before use
5. **Handle module loading failures** gracefully

### Performance Considerations
1. **Avoid global variables** in embedded environments
2. **Use local variables** for better performance
3. **Manage memory** in long-running scripts
4. **Monitor garbage collection** in game loops
5. **Optimize hot paths** in embedded systems

### Security
1. **Validate external input** in configuration scripts
2. **Sanitize user data** in game mods
3. **Use sandboxing** for untrusted scripts
4. **Limit resource usage** in embedded environments
5. **Avoid eval-like constructs** with user input

## Example Fixes

### Nil Access Error
```lua
-- Error: attempt to index nil value
local result = object.property

-- Fix: Add nil check
if object ~= nil then
    local result = object.property
    -- Process result
else
    print("Warning: object is nil")
end
```

### Table Operation Error
```lua
-- Error: table index is nil
table[nil] = value

-- Fix: Validate key
if key ~= nil then
    table[key] = value
else
    print("Error: cannot use nil as table key")
end
```

### Module Loading Error
```lua
-- Error: module 'nonexistent' not found
local module = require("nonexistent")

-- Fix: Safe module loading
local function safe_require(module_name)
    local ok, module = pcall(require, module_name)
    if ok then
        return module
    else
        print("Failed to load module:", module_name)
        return nil
    end
end

local module = safe_require("nonexistent")
if module then
    -- Use module
end
```

## Testing Integration

### Unit Testing
```lua
-- test_example.lua
local function test_nil_safety()
    local obj = nil
    
    -- Test nil check
    if obj ~= nil then
        assert(false, "Should not reach here")
    end
    
    -- Test safe access
    local value = obj and obj.field or "default"
    assert(value == "default", "Default value should be returned")
end

test_nil_safety()
```

### Integration Testing
```lua
-- Test module loading
local function test_module_loading()
    local ok, module = pcall(require, "test_module")
    assert(ok, "Module should load successfully")
    assert(type(module) == "table", "Module should be a table")
end

test_module_loading()
```

## Troubleshooting

### Common Issues

1. **Nil access errors**
   - Check variable initialization
   - Add nil checks before access
   - Use logical operators for defaults

2. **Table operation failures**
   - Validate table existence
   - Check key validity
   - Initialize tables properly

3. **Module loading problems**
   - Verify module path
   - Check package.path configuration
   - Use safe loading patterns

4. **Coroutine issues**
   - Check coroutine state
   - Handle yield/resume properly
   - Avoid yielding from main thread

### Performance Issues

1. **Memory leaks**
   - Monitor garbage collection
   - Avoid circular references
   - Clear unused variables

2. **Slow execution**
   - Use local variables
   - Avoid global lookups
   - Optimize hot paths

## Related Documentation

- [Plugin Architecture](plugin_architecture.md)
- [Error Schema](error_schema.md)
- [Contributing Rules](contributing-rules.md)
- [Best Practices](best_practices.md)