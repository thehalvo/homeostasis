# API Keys Configuration Guide

Homeostasis integrates with multiple Large Language Model (LLM) providers to enable AI-assisted error analysis and code generation. This guide covers all aspects of API key configuration, management, and usage.

## Overview

The system supports three primary LLM providers:
- **OpenAI**: GPT models for code generation and analysis
- **Anthropic**: Claude models for detailed error understanding
- **OpenRouter**: Unified endpoint supporting multiple providers

## Quick Setup

### Command Line Interface

```bash
# Set up API keys interactively
homeostasis set-key openai
homeostasis set-key anthropic
homeostasis set-key openrouter

# Verify configuration
homeostasis list-keys --show-status
homeostasis test-providers
```

### Web Dashboard Interface

1. Start the dashboard: `homeostasis dashboard`
2. Navigate to: `http://localhost:5000/config`
3. Click on the "LLM Keys" tab
4. Enter your API keys in the provider forms
5. Test and save your configuration

## Detailed Configuration

### Provider-Specific Setup

#### OpenAI Configuration

```bash
# Set OpenAI API key
homeostasis set-key openai

# Test the key
homeostasis validate-key openai

# Configure as primary provider
homeostasis set-active-provider openai
```

**Key Requirements:**
- Format: `sk-` followed by 48+ characters
- Account: OpenAI account with available credits
- Permissions: API access enabled
- Get Key: [OpenAI Platform](https://platform.openai.com/api-keys)

#### Anthropic Configuration

```bash
# Set Anthropic API key
homeostasis set-key anthropic

# Test the key
homeostasis validate-key anthropic

# Configure as primary provider
homeostasis set-active-provider anthropic
```

**Key Requirements:**
- Format: `sk-ant-` followed by 86+ characters
- Account: Anthropic account with API access
- Permissions: Claude API access enabled
- Get Key: [Anthropic Console](https://console.anthropic.com/)

#### OpenRouter Configuration

```bash
# Set OpenRouter API key
homeostasis set-key openrouter

# Enable unified mode (proxy to other providers)
homeostasis set-openrouter-unified true --proxy-anthropic --proxy-openai

# Configure as primary provider
homeostasis set-active-provider openrouter
```

**Key Requirements:**
- Format: `sk-or-` followed by 57+ characters
- Account: OpenRouter account with credits
- Benefits: Access to multiple models through single endpoint
- Get Key: [OpenRouter Keys](https://openrouter.ai/keys)

### Multi-Provider Configuration

#### Failover Setup

```bash
# Configure automatic failover order
homeostasis set-fallback-order anthropic openai openrouter

# Enable automatic switching on failures
homeostasis set-fallback-enabled true

# Configure retry settings
homeostasis set-retry-config --max-retries 3 --backoff exponential
```

#### Provider Selection Policies

```bash
# Configure selection based on cost optimization
homeostasis set-provider-policies --cost balanced

# Configure for low latency
homeostasis set-provider-policies --latency low

# Configure for high reliability
homeostasis set-provider-policies --reliability high

# Mixed policy configuration
homeostasis set-provider-policies --cost low --latency medium --reliability high
```

## Storage Options

### Environment Variables (Highest Priority)

```bash
# Set in shell profile
export HOMEOSTASIS_OPENAI_API_KEY="sk-..."
export HOMEOSTASIS_ANTHROPIC_API_KEY="sk-ant-..."
export HOMEOSTASIS_OPENROUTER_API_KEY="sk-or-..."

# Verify environment variables
homeostasis list-keys --show-sources
```

### External Secrets Managers (Medium Priority)

#### AWS Secrets Manager

```bash
# Configure AWS credentials
export AWS_DEFAULT_REGION="us-east-1"
aws configure  # Set up access keys

# Homeostasis will automatically detect and use AWS Secrets Manager
homeostasis list-keys --show-sources
```

Store secrets with these names:
- `homeostasis/openai-api-key`
- `homeostasis/anthropic-api-key`
- `homeostasis/openrouter-api-key`

#### Azure Key Vault

```bash
# Configure Azure credentials
export AZURE_KEY_VAULT_URL="https://your-vault.vault.azure.net/"
az login  # Authenticate with Azure

# Homeostasis will automatically detect and use Azure Key Vault
homeostasis list-keys --show-sources
```

Store secrets with these names:
- `homeostasis-openai-api-key`
- `homeostasis-anthropic-api-key`
- `homeostasis-openrouter-api-key`

#### HashiCorp Vault

```bash
# Configure Vault connection
export VAULT_ADDR="https://vault.example.com:8200"
export VAULT_TOKEN="hvs.token..."

# Homeostasis will automatically detect and use Vault
homeostasis list-keys --show-sources
```

Store secrets at these paths:
- `secret/homeostasis/openai-api-key`
- `secret/homeostasis/anthropic-api-key`
- `secret/homeostasis/openrouter-api-key`

### Encrypted Local Storage (Lowest Priority)

```bash
# Keys are automatically encrypted when set via CLI
homeostasis set-key openai

# Storage location: ~/.homeostasis/llm_keys.enc
# Encryption: PBKDF2 + Fernet (100,000 iterations)
```

## Web Dashboard Management

### Accessing the Dashboard

```bash
# Start dashboard with default settings
homeostasis dashboard

# Start with custom port
homeostasis dashboard --port 8080

# Start with debug mode
homeostasis dashboard --debug
```

### Dashboard Features

#### LLM Keys Tab

The dashboard provides a dedicated "LLM Keys" tab in the configuration section with:

**Provider Cards:**
- Individual cards for OpenAI, Anthropic, and OpenRouter
- Status badges showing key configuration state
- Source indicators (Environment, External, Encrypted)
- Input fields with password masking and visibility toggle

**Key Management Functions:**
- **Set Keys**: Enter API keys with real-time format validation
- **Test Keys**: Individual or bulk testing with detailed results
- **Remove Keys**: Secure key removal with confirmation
- **View Sources**: See which storage backend is providing each key

**Configuration Panel:**
- **Default Provider**: Select primary LLM provider
- **Failover Order**: Configure automatic provider switching sequence
- **Test All**: Bulk testing of all configured providers
- **Status Monitoring**: Real-time key validation and connectivity status

#### Security Features

- **Password Masking**: All key inputs are masked by default
- **Toggle Visibility**: Show/hide keys temporarily for verification
- **Live Validation**: Immediate feedback on key format and connectivity
- **Encrypted Display**: Keys are never shown in plaintext after storage
- **Session Security**: Keys are not persisted in browser sessions

### CLI-Dashboard Synchronization

Changes made in either interface are immediately reflected in the other:

```bash
# Set key via CLI
homeostasis set-key openai

# Key appears in dashboard immediately
# No restart required

# Set key via dashboard
# Key is immediately available to CLI
homeostasis list-keys  # Shows dashboard-configured key
```

## Key Management Operations

### Listing Keys

```bash
# Basic key listing
homeostasis list-keys

# Show masked keys
homeostasis list-keys --show-masked

# Show key sources
homeostasis list-keys --show-sources

# Verbose output with all details
homeostasis list-keys --verbose
```

### Testing Keys

```bash
# Test specific provider
homeostasis validate-key openai

# Test all providers
homeostasis test-providers

# Test with detailed output
homeostasis test-providers --verbose

# Test with specific timeout
homeostasis test-providers --timeout 30
```

### Removing Keys

```bash
# Remove specific provider key
homeostasis remove-key openai

# Remove all keys
homeostasis remove-key --all

# Remove with confirmation bypass
homeostasis remove-key anthropic --force
```

### Key Rotation

```bash
# Update existing key
homeostasis set-key openai  # Will replace existing key

# Rotate all keys
homeostasis rotate-keys --interactive

# Schedule automatic rotation
homeostasis set-key-rotation --interval 90d --notification email
```

## Advanced Usage

### Provider Status Monitoring

```bash
# Check provider status
homeostasis provider-status

# Monitor in real-time
homeostasis provider-status --watch

# Get JSON output for automation
homeostasis provider-status --format json
```

### Usage Analytics

```bash
# View usage statistics
homeostasis usage-stats

# Export usage data
homeostasis usage-stats --export csv --output usage.csv

# Monitor costs
homeostasis usage-stats --cost-breakdown
```

### Integration with CI/CD

```bash
# In CI/CD pipelines, use environment variables
export HOMEOSTASIS_OPENAI_API_KEY="$OPENAI_API_KEY"
export HOMEOSTASIS_ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY"

# Validate in pipeline
homeostasis test-providers --non-interactive
```

## Troubleshooting

### Common Issues

#### Key Format Errors
```bash
# Error: Invalid key format
homeostasis validate-key openai --verbose
# Check key prefix and length requirements

# Get format help
homeostasis key-formats
```

#### Connection Issues
```bash
# Test network connectivity
homeostasis test-providers --debug

# Check proxy settings
homeostasis test-providers --proxy http://proxy:8080

# Verify firewall settings
curl -v https://api.openai.com/v1/models
```

#### Permission Errors
```bash
# Check file permissions
ls -la ~/.homeostasis/llm_keys.enc

# Reset permissions
chmod 600 ~/.homeostasis/llm_keys.enc

# Verify directory permissions
chmod 700 ~/.homeostasis/
```

### Debug Commands

```bash
# Enable debug logging
export HOMEOSTASIS_LOG_LEVEL=DEBUG
homeostasis test-providers

# Show configuration path
homeostasis config-info

# Validate configuration files
homeostasis validate-config
```

### Getting Help

```bash
# Command-specific help
homeostasis set-key --help
homeostasis list-keys --help

# Provider-specific information
homeostasis provider-info openai
homeostasis provider-info --all

# Configuration examples
homeostasis examples api-keys
```

## Security Best Practices

1. **Use Environment Variables**: For production deployments
2. **External Secrets**: Leverage cloud-native secret management
3. **Key Rotation**: Implement regular key rotation schedules
4. **Access Control**: Limit access to key storage locations
5. **Monitoring**: Track key usage and set up alerts
6. **Backup**: Ensure external secrets managers are backed up
7. **Audit**: Review key access logs regularly

## API Reference

### Key Management Endpoints

When using the dashboard, these endpoints are available:

- `GET /api/llm-keys` - List all key statuses
- `POST /api/llm-keys/<provider>` - Set provider key
- `DELETE /api/llm-keys/<provider>` - Remove provider key
- `POST /api/llm-keys/<provider>/test` - Test provider key
- `POST /api/llm-keys/test-all` - Test all keys

### Configuration Schema

```json
{
  "llm_providers": {
    "default_provider": "anthropic",
    "fallback_enabled": true,
    "fallback_order": ["anthropic", "openai", "openrouter"],
    "retry_config": {
      "max_retries": 3,
      "backoff": "exponential"
    },
    "selection_policy": {
      "cost": "balanced",
      "latency": "low",
      "reliability": "high"
    }
  }
}
```

This guide covers all aspects of API key management in Homeostasis. For additional help, see the main documentation or open an issue on GitHub.