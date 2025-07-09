# Crystal Integration

Homeostasis provides full support for Crystal, a Ruby-like language with static typing and performance. This integration handles Crystal's unique features including compile-time type checking, union types, fiber-based concurrency, and memory safety.

## Overview

The Crystal integration includes:
- **Syntax Error Detection**: Parse errors, macro expansion issues, and language-specific syntax validation
- **Type System Support**: Static type checking, union types, generics, and type inference
- **Memory Safety**: Null pointer safety, bounds checking, and memory management
- **Concurrency**: Fiber management, channel communication, and async patterns
- **Metaprogramming**: Macro errors, annotation processing, and compile-time evaluation

## Supported Error Types

### Syntax Errors
- Parse errors and unexpected tokens
- Invalid macro syntax and expansion
- Unterminated string and character literals
- Missing operators and syntax elements

### Type System Errors
- Type mismatches and inference failures
- Union type handling issues
- Generic type instantiation problems
- Null safety violations

### Memory Management
- Null pointer access prevention
- Array and slice bounds checking
- Memory allocation and deallocation errors
- Reference counting issues

### Concurrency
- Fiber creation and management
- Channel communication errors
- Async/await pattern issues
- Deadlock and race condition detection

### Metaprogramming
- Macro definition and expansion errors
- Annotation processing failures
- Compile-time evaluation issues
- Code generation problems

## Configuration

### Basic Setup

```crystal
# example.cr
require "http/client"

# Union type example
alias StringOrInt = String | Int32

def process_value(value : StringOrInt) : String
  case value
  when String
    "String: #{value}"
  when Int32
    "Number: #{value}"
  else
    "Unknown type"
  end
end

# Fiber example
spawn do
  puts "Background task running"
  sleep 1
end
```

### Error Handling Patterns

**Null Safety:**
```crystal
# Safe null handling
value = get_nullable_value()
if value.nil?
  puts "No value found"
else
  puts "Value: #{value}"
end

# Using try method
result = risky_operation.try(&.upcase)
puts result || "Operation failed"
```

**Union Types:**
```crystal
# Handling union types safely
def handle_union(value : String | Int32 | Nil)
  case value
  when String
    puts "Got string: #{value}"
  when Int32
    puts "Got integer: #{value}"
  when Nil
    puts "Got nil"
  end
end
```

**Fiber Communication:**
```crystal
# Channel-based communication
channel = Channel(String).new

spawn do
  channel.send("Hello from fiber")
end

message = channel.receive
puts message
```

## Common Fix Patterns

### Null Safety
```crystal
# Before (unsafe)
obj.method

# After (safe)
obj.try(&.method) || default_value
```

### Union Type Handling
```crystal
# Before (unsafe)
value.as(String).upcase

# After (safe)
case value
when String
  value.upcase
else
  "default"
end
```

### Fiber Error Handling
```crystal
# Before (unsafe)
spawn do
  risky_operation
end

# After (safe)
spawn do
  begin
    risky_operation
  rescue ex
    puts "Fiber error: #{ex.message}"
  end
end
```

## Best Practices

1. **Use Union Types**: Leverage Crystal's union types for safer code
2. **Null Safety**: Always check for nil before accessing values
3. **Channel Communication**: Use channels for fiber communication
4. **Error Handling**: Wrap risky operations in begin/rescue blocks
5. **Type Annotations**: Use explicit type annotations for clarity

## Framework Support

The Crystal integration supports popular Crystal frameworks and libraries:
- **Kemal**: Web framework error handling
- **Lucky**: Full-stack web framework support
- **Amber**: Web application framework integration
- **Sidekiq**: Background job processing
- **DB**: Database connection and query errors

## Error Examples

### Syntax Error
```crystal
# Error: Missing end keyword
def example
  puts "Hello"

# Fix: Add end keyword
def example
  puts "Hello"
end
```

### Type Error
```crystal
# Error: Type mismatch
def add_numbers(a : Int32, b : Int32) : Int32
  a + b
end

result = add_numbers("5", "10")

# Fix: Use correct types
result = add_numbers(5, 10)
```

### Null Access
```crystal
# Error: Potential null access
value = get_nullable_value()
puts value.upcase

# Fix: Check for nil
value = get_nullable_value()
if value
  puts value.upcase
else
  puts "No value"
end
```

## Advanced Features

### Custom Error Types
```crystal
class CustomError < Exception
  def initialize(@code : Int32, message : String)
    super(message)
  end
  
  getter code
end

def risky_operation
  raise CustomError.new(404, "Not found")
rescue ex : CustomError
  puts "Error #{ex.code}: #{ex.message}"
end
```

### Macro Error Handling
```crystal
# Macro with error checking
macro define_getter(name, type)
  def {{name.id}} : {{type.id}}
    @{{name.id}} || raise "{{name.id}} not initialized"
  end
end

class Example
  define_getter(value, String)
  
  def initialize(@value : String)
  end
end
```

### Channel Error Handling
```crystal
# Safe channel operations
channel = Channel(String).new(capacity: 1)

spawn do
  begin
    channel.send("message")
  rescue Channel::ClosedError
    puts "Channel is closed"
  end
end

begin
  message = channel.receive
  puts "Received: #{message}"
rescue Channel::ClosedError
  puts "Channel was closed"
end
```

## Integration Testing

The Crystal integration includes extensive testing:

```bash
# Run Crystal plugin tests
python -m pytest tests/test_crystal_plugin.py -v

# Test specific error types
python -m pytest tests/test_crystal_plugin.py::TestCrystalExceptionHandler::test_analyze_nil_error -v
```

## Performance Considerations

- **Compile-time Optimization**: Leverage Crystal's compile-time features for better performance
- **Memory Efficiency**: Use appropriate data structures and avoid unnecessary allocations
- **Fiber Performance**: Optimize fiber usage for better concurrency
- **Type Safety**: Use static typing to avoid runtime type checks

## Troubleshooting

### Common Issues

1. **Compilation Failures**: Check syntax and ensure all dependencies are available
2. **Type Errors**: Verify type annotations and union type handling
3. **Fiber Deadlocks**: Ensure proper channel usage and avoid blocking operations
4. **Memory Issues**: Check for null safety violations and bounds checking

### Debug Commands

```bash
# Check Crystal version
crystal version

# Compile with debug info
crystal build --debug example.cr

# Run with verbose output
crystal run --verbose example.cr
```

## Related Documentation

- [Error Schema](error_schema.md)
- [Plugin Architecture](plugin_architecture.md)
- [Best Practices](best_practices.md)
- [Integration Guides](integration_guides.md)