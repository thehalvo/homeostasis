# Elixir/Erlang Integration

This document explains how to integrate Homeostasis with Elixir and Erlang applications for automated error detection and healing.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Advanced Features](#advanced-features)
- [Rule Customization](#rule-customization)
- [Templates](#templates)
- [Limitations](#limitations)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [FAQs](#faqs)

## Overview

Homeostasis provides support for automatically detecting and fixing common errors in Elixir and Erlang applications. It includes:

- Error detection for standard Elixir errors
- Framework-specific error handling for Phoenix, Ecto, and OTP applications
- BEAM VM error detection and remediation
- Template-based automated fixes

## Installation

### Prerequisites

- Python 3.7+
- Elixir 1.10+ / Erlang/OTP 22+
- Homeostasis core installed

### Setup

1. **Install the Homeostasis Python package:**

```bash
pip install homeostasis
```

2. **Add the Elixir integration in your project:**

```elixir
# mix.exs
def deps do
  [
    {:homeostasis_ex, "~> 0.1.0"}
  ]
end
```

3. **Configure Elixir Logger to send errors to Homeostasis:**

```elixir
# config/config.exs
config :logger,
  backends: [:console, Homeostasis.LoggerBackend]

config :homeostasis_ex,
  # API key for cloud-based Homeostasis (optional)
  api_key: System.get_env("HOMEOSTASIS_API_KEY"),
  # Homeostasis endpoint (optional, uses default if not specified)
  endpoint: System.get_env("HOMEOSTASIS_ENDPOINT"),
  # Logging level to capture (defaults to :error)
  level: :error
```

## Configuration

### Environment Variables

- `HOMEOSTASIS_API_KEY`: API key for cloud-based Homeostasis (optional)
- `HOMEOSTASIS_ENDPOINT`: HTTP endpoint for Homeostasis API (optional)
- `HOMEOSTASIS_LOG_LEVEL`: Minimum log level to process (default: `error`)
- `HOMEOSTASIS_DISABLE_AUTO_FIX`: Set to `true` to disable automatic fixes (default: `false`)

### Configuration File

You can configure Homeostasis through a YAML file at `config/homeostasis.yaml`:

```yaml
elixir:
  # Framework detection
  frameworks:
    - phoenix
    - ecto
    - otp
    
  # Rules configuration
  rules:
    enabled: true
    custom_path: "priv/homeostasis/rules"
    
  # Fix templates
  templates:
    enabled: true
    custom_path: "priv/homeostasis/templates"
    
  # Authentication
  auth:
    api_key: "${HOMEOSTASIS_API_KEY}"
    
  # Logging
  logging:
    level: "error"
    include_stacktrace: true
    anonymize_data: false
```

## Usage

### Basic Usage

Once configured, the Homeostasis integration will automatically capture errors from your Elixir application and analyze them.

For manual integration:

```elixir
defmodule MyApp.ErrorHandler do
  def handle_error(error, stacktrace) do
    # Send error to Homeostasis for analysis
    case Homeostasis.analyze_error(error, stacktrace) do
      {:ok, analysis} ->
        # The error was analyzed successfully
        if analysis.fix do
          # Apply fix if available
          Homeostasis.apply_fix(analysis.fix)
        else
          # Log the suggestion
          require Logger
          Logger.warning("Homeostasis suggestion: #{analysis.suggestion}")
        end
        
      {:error, reason} ->
        # Failed to analyze the error
        require Logger
        Logger.error("Homeostasis analysis failed: #{inspect(reason)}")
    end
    
    # Continue with regular error handling
    :error_logger.error_report(error, stacktrace)
  end
end
```

### Phoenix Integration

For Phoenix applications, include the Homeostasis plug in your endpoint:

```elixir
# lib/my_app_web/endpoint.ex
defmodule MyAppWeb.Endpoint do
  use Phoenix.Endpoint, otp_app: :my_app
  
  # Add Homeostasis plug for error monitoring
  plug Homeostasis.Phoenix.ErrorMonitor
  
  # Rest of your endpoint configuration
  # ...
end
```

### Ecto Integration

For Ecto, include the Homeostasis hooks in your Repo:

```elixir
# lib/my_app/repo.ex
defmodule MyApp.Repo do
  use Ecto.Repo,
    otp_app: :my_app,
    adapter: Ecto.Adapters.Postgres
    
  # Add Homeostasis hooks for Ecto error monitoring
  use Homeostasis.Ecto.ErrorMonitor
end
```

### OTP Integration

For OTP applications, use the Homeostasis supervisor wrapper:

```elixir
# lib/my_app/application.ex
defmodule MyApp.Application do
  use Application
  
  def start(_type, _args) do
    children = [
      # ... your regular child specs
    ]
    
    # Use Homeostasis-wrapped supervisor
    opts = [strategy: :one_for_one, name: MyApp.Supervisor]
    Homeostasis.OTP.Supervisor.start_link(children, opts)
  end
end
```

## Advanced Features

### Custom Error Handlers

You can create custom error handlers for specific types of errors:

```elixir
defmodule MyApp.CustomErrorHandler do
  use Homeostasis.ErrorHandler
  
  # Handle specific errors differently
  def handle_error(%Phoenix.Router.NoRouteError{} = error, stacktrace) do
    # Custom handling for route not found errors
    # ...
    
    # Call the default handler
    super(error, stacktrace)
  end
end
```

### Telemetry Integration

Homeostasis integrates with `:telemetry` to monitor key metrics:

```elixir
# Configure Homeostasis to listen for telemetry events
config :homeostasis_ex,
  telemetry: [
    # Phoenix telemetry events
    [:phoenix, :router_dispatch, :exception],
    # Ecto telemetry events
    [:my_app, :repo, :query, :exception],
    # Custom telemetry events
    [:my_app, :service, :exception]
  ]
```

## Rule Customization

### Creating Custom Rules

Create custom rules in JSON format:

```json
{
  "rules": [
    {
      "id": "my_app_custom_error",
      "pattern": "\\*\\* \\(MyApp\\.CustomError\\) (.*)",
      "type": "MyApp.CustomError",
      "description": "Custom application-specific error",
      "root_cause": "my_app_custom_error",
      "suggestion": "Check the configuration in config/custom.exs.",
      "confidence": "high",
      "severity": "medium",
      "category": "application",
      "framework": "elixir"
    }
  ]
}
```

Save this file to `priv/homeostasis/rules/custom_errors.json`.

### Extending Existing Rules

You can extend or override existing rules by creating files with the same name in your custom rules directory.

## Templates

### Creating Custom Fix Templates

Create custom fix templates in Elixir:

```elixir
# Template for fixing MyApp.CustomError
# Original code with error:
# data = MyApp.get_data()
# result = MyApp.process_data!(data)  # This may raise MyApp.CustomError

# Fix: Add error handling with rescue
try do
  data = MyApp.get_data()
  result = MyApp.process_data!(data)
  {:ok, result}
rescue
  e in MyApp.CustomError ->
    # Log the error
    require Logger
    Logger.error("Error processing data: #{inspect(e)}")
    
    # Return error tuple
    {:error, :processing_failed}
end
```

Save this file to `priv/homeostasis/templates/my_app_custom_error.ex.template`.

## Limitations

- **OTP Distribution**: Currently, Homeostasis has limited support for errors across distributed Erlang nodes.
- **Hot Code Reloading**: Careful consideration is needed when applying fixes in systems that use hot code reloading.
- **Umbrella Projects**: Requires special configuration for umbrella projects with multiple applications.

## Examples

### Error Detection Example

```elixir
# Example error handling in a Phoenix controller
def show(conn, %{"id" => id}) do
  case Repo.get(User, id) do
    nil ->
      # This will be detected by Homeostasis
      raise MyApp.UserNotFoundError, "User with ID #{id} not found"
      
    user ->
      render(conn, "show.html", user: user)
  end
end
```

### Applying Fixes Example

```elixir
# Manually request and apply a fix
def risky_operation(data) do
  try do
    # Potentially error-prone code
    process_data!(data)
  rescue
    e ->
      # Use Homeostasis to analyze and fix
      case Homeostasis.analyze_and_fix(e, __STACKTRACE__) do
        {:ok, fixed_result} -> fixed_result
        {:error, _} -> reraise e, __STACKTRACE__
      end
  end
end
```

## Troubleshooting

### Common Issues

1. **No errors are being captured**
   - Check that the logger backend is properly configured
   - Verify the log level is set appropriately

2. **Rules aren't triggering**
   - Check that the error pattern matches your custom rules
   - Enable debug logging for Homeostasis

3. **Fixes aren't working**
   - Verify that auto-fix is enabled
   - Check permissions for file writing if applying code changes

### Debugging

Enable debug mode for more detailed logging:

```elixir
config :homeostasis_ex,
  debug: true,
  log_level: :debug
```

## FAQs

**Q: Is Homeostasis compatible with LiveView?**  
A: Yes, Homeostasis can detect and fix errors in Phoenix LiveView applications.

**Q: Can I use Homeostasis in production?**  
A: Yes, but we recommend starting with monitoring-only mode (disable auto-fix) and gradually enabling fixes after testing.

**Q: What's the performance impact?**  
A: Homeostasis is designed to have minimal impact on normal application performance. Error analysis happens asynchronously.

**Q: Does it work with release builds?**  
A: Yes, Homeostasis is compatible with Elixir releases, but requires additional configuration.

---

For more information, see the [main Homeostasis documentation](../README.md) or join our community on [GitHub Discussions](https://github.com/homeostasis/homeostasis/discussions).