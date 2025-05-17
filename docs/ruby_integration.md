# Ruby Integration in Homeostasis

This document describes the Ruby integration in the Homeostasis framework, including the supported features, error detection capabilities, and integration options.

## Overview

The Ruby integration in Homeostasis allows the framework to detect, analyze, and heal errors in Ruby applications. It supports various Ruby frameworks including Rails, Sinatra, and Rack-based applications, providing comprehensive error handling across the Ruby ecosystem.

## Supported Features

### Core Ruby Error Detection

The Ruby plugin can detect and provide fixes for a wide range of standard Ruby errors:

- `NoMethodError` including nil references
- `NameError` and undefined constants
- `ArgumentError` including wrong argument count
- `TypeError` for type conversion issues
- `LoadError` for missing files and gems
- `SyntaxError` for syntax issues
- `ThreadError` including deadlock detection
- File and IO errors via `Errno` classes
- Network connection issues
- Encoding errors
- Regular expression errors
- Concurrency issues

### Framework Support

#### Ruby on Rails Integration

The Rails integration provides specialized handling for:

- ActiveRecord errors (RecordNotFound, ValidationErrors, etc.)
- ActionController errors (RoutingError, ParameterMissing, etc.)
- ActionView template errors
- Database connection issues
- Session and cookie management
- Authentication and authorization errors

#### Sinatra and Rack Integration

Support for Sinatra and Rack middleware includes:

- Route matching errors
- Template rendering issues
- Parameter validation
- HTTP status and response handling
- Middleware configuration errors
- Session management issues

### Ruby Metaprogramming Support

The Ruby plugin includes specialized detection for metaprogramming errors:

- `method_missing` and `const_missing` failures
- Dynamic method definition errors
- Module inclusion and extension issues
- Class and instance eval errors
- Class variable management
- Dynamic constant assignment issues
- Method reflection errors (send, public_send)

## Integration Options

### Standalone Ruby Applications

For standalone Ruby applications, integrate Homeostasis by adding the following to your application:

```ruby
require 'homeostasis'

Homeostasis::Configuration.configure do |config|
  config.framework = :ruby
  config.environment = ENV['RUBY_ENV'] || 'development'
  config.logger = Logger.new(STDOUT)
end

# Wrap your application code in the Homeostasis executor
Homeostasis::Executor.run do
  # Your application code here
end
```

### Rails Applications

For Rails applications, add the Homeostasis Rails middleware:

```ruby
# In config/application.rb
config.middleware.use Homeostasis::Middleware::Rails
```

Or use the provided initializer:

```ruby
# In config/initializers/homeostasis.rb
Rails.application.config.homeostasis = {
  enabled: true,
  healing_mode: Rails.env.development? ? :immediate : :suggest,
  alert_on_healing: true,
  ignored_errors: ['ActionController::RoutingError']
}
```

### Sinatra Applications

For Sinatra applications, add the middleware to your application:

```ruby
# In your Sinatra app
require 'homeostasis/middleware/sinatra'

use Homeostasis::Middleware::Sinatra
```

## Configuration Options

The Ruby integration supports the following configuration options:

| Option | Description | Default |
|--------|-------------|---------|
| `enabled` | Enable or disable healing | `true` |
| `healing_mode` | Mode of operation: `:immediate`, `:suggest`, or `:log` | `:suggest` |
| `alert_on_healing` | Send alerts when healing is performed | `true` |
| `ignored_errors` | Array of error classes to ignore | `[]` |
| `custom_rules_path` | Path to custom error rules | `nil` |
| `max_healing_attempts` | Maximum healing attempts per error | `3` |
| `healing_cooldown` | Seconds to wait between healing attempts | `300` |

## Custom Rules

You can extend the Ruby error detection with custom rules:

```ruby
# In your initialization code
Homeostasis::Configuration.add_rules_file('/path/to/custom_rules.json')
```

Custom rule files should follow this format:

```json
{
  "name": "Custom Ruby Rules",
  "description": "Custom rules for application-specific errors",
  "version": "0.1.0",
  "rules": [
    {
      "id": "custom_error_rule",
      "pattern": "CustomError: (.*)",
      "type": "CustomError",
      "description": "Application-specific custom error",
      "root_cause": "custom_error_cause",
      "suggestion": "Handle the custom error with specific logic",
      "confidence": "high",
      "severity": "medium",
      "category": "application"
    }
  ]
}
```

## Gem Dependencies

The Ruby integration analyzes your Gemfile and Gemfile.lock to detect dependency issues. Common problems detected include:

- Version conflicts
- Missing dependencies
- Incompatible gem versions
- Unresolved dependencies
- Security vulnerabilities in dependencies

## Metrics and Monitoring

The Ruby integration collects the following metrics:

- Error frequency by type
- Healing success rate
- Common error patterns
- Performance impact of errors
- Response time improvements after healing

These metrics are available through the Homeostasis dashboard and can be integrated with common Ruby monitoring tools.

## Example Usage

### Basic Error Handling

```ruby
begin
  # Your code that might raise errors
rescue => e
  # Homeostasis will intercept and analyze the error
  raise e  # Re-raise to let Homeostasis handle it
end
```

### Rails Controller Example

```ruby
class UsersController < ApplicationController
  # Homeostasis will track exceptions and fix common issues
  def show
    @user = User.find(params[:id])
  rescue ActiveRecord::RecordNotFound
    # Homeostasis will suggest using find_by instead
    flash[:alert] = "User not found"
    redirect_to users_path
  end
end
```

## Best Practices

1. **Enable Detailed Logging**: In development, enable detailed Homeostasis logging to understand how errors are being detected and fixed.

2. **Review Healing Suggestions**: Always review and validate healing suggestions before applying them in production.

3. **Custom Rules for Business Logic**: Create custom rules for application-specific errors related to your business logic.

4. **Regular Updates**: Keep your Homeostasis integration updated to benefit from new rule sets and detection capabilities.

5. **Combine with Monitoring**: Integrate Homeostasis with your existing monitoring solutions for comprehensive observability.

## Limitations

- Complex metaprogramming patterns might not be fully analyzable
- Some Rails engine errors may require custom handling
- JRuby-specific errors have limited support
- Some C extension errors cannot be fully analyzed

## Further Reading

- [Ruby Error Handling Best Practices](https://link/to/docs)
- [Rails Integration Details](https://link/to/docs)
- [Custom Rule Development Guide](https://link/to/docs)
- [Homeostasis API Reference](https://link/to/docs)