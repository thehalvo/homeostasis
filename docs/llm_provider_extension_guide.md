# LLM Provider Extension Guide

This guide explains how to add new LLM providers to the Homeostasis framework's provider abstraction layer.

## Overview

The Homeostasis LLM integration system uses a provider abstraction layer that allows seamless integration of different LLM providers. This guide walks you through adding support for new providers.

## Architecture

The provider system is built on these core components:

- `LLMProviderInterface`: Abstract base class defining the provider contract
- `ProviderFactory`: Factory for creating provider instances
- `LLMManager`: High-level manager for multi-provider operations
- Data classes: `LLMRequest`, `LLMResponse`, `LLMMessage` for unified data structures

## Adding a New Provider

### Step 1: Implement the Provider Interface

Create a new class that inherits from `LLMProviderInterface`:

```python
from modules.llm_integration.provider_abstraction import LLMProviderInterface, LLMRequest, LLMResponse, LLMError
import requests

class MyLLMProvider(LLMProviderInterface):
    """Your custom LLM provider implementation."""
    
    def __init__(self, api_key: str, base_url: str = "https://api.myllm.com/v1"):
        """
        Initialize your provider.
        
        Args:
            api_key: API key for authentication
            base_url: Base URL for API endpoints
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self._session = requests.Session()
        self._session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        })
    
    @property
    def name(self) -> str:
        """Return the provider name (lowercase, no spaces)."""
        return "myllm"
    
    @property
    def default_model(self) -> str:
        """Return the default model for this provider."""
        return "myllm-v1"
    
    def complete(self, request: LLMRequest) -> LLMResponse:
        """Implement chat completion."""
        # Convert unified request format to provider-specific format
        messages = self._convert_messages(request)
        
        data = {
            "model": request.model or self.default_model,
            "messages": messages,
            "max_tokens": request.max_tokens or 1000,
            "temperature": request.temperature or 0.7
        }
        
        try:
            response = self._session.post(
                f"{self.base_url}/chat/completions",
                json=data,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            return self._convert_response(result)
            
        except requests.RequestException as e:
            raise LLMError(f"MyLLM API request failed: {e}")
        except Exception as e:
            raise LLMError(f"MyLLM completion failed: {e}")
    
    def validate_key(self, api_key: str) -> bool:
        """Validate API key with a lightweight request."""
        headers = {'Authorization': f'Bearer {api_key}'}
        
        try:
            response = requests.get(
                f"{self.base_url}/models",  # Or equivalent endpoint
                headers=headers,
                timeout=10
            )
            return response.status_code == 200
        except Exception:
            return False
    
    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        try:
            response = self._session.get(f"{self.base_url}/models", timeout=10)
            response.raise_for_status()
            
            result = response.json()
            return [model['id'] for model in result.get('data', [])]
        except Exception:
            return [self.default_model]
    
    def _convert_messages(self, request: LLMRequest) -> List[dict]:
        """Convert unified message format to provider-specific format."""
        messages = []
        
        # Handle system prompt based on provider requirements
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        
        # Convert messages
        for msg in request.messages:
            messages.append({"role": msg.role, "content": msg.content})
        
        return messages
    
    def _convert_response(self, result: dict) -> LLMResponse:
        """Convert provider response to unified format."""
        # Extract content based on provider response format
        content = result['choices'][0]['message']['content']  # Adjust as needed
        
        return LLMResponse(
            content=content,
            provider=self.name,
            model=result.get('model'),
            usage=result.get('usage'),
            metadata={
                'finish_reason': result['choices'][0].get('finish_reason'),
                'response_id': result.get('id')
            }
        )
```

### Step 2: Register with ProviderFactory

Add your provider to the `ProviderFactory` in `provider_abstraction.py`:

```python
# In ProviderFactory.create_provider method
elif provider_name == "myllm":
    return MyLLMProvider(api_key, **kwargs)

# In ProviderFactory.get_supported_providers method
return ["openai", "anthropic", "openrouter", "myllm"]
```

### Step 3: Add API Key Management Support

Update the `APIKeyManager` to support your new provider:

```python
# In api_key_manager.py, add to SUPPORTED_PROVIDERS
SUPPORTED_PROVIDERS = {
    'openai': 'sk-',
    'anthropic': 'sk-ant-',
    'openrouter': 'sk-or-',
    'myllm': 'ml-'  # Add your provider's key prefix
}
```

### Step 4: Add CLI Support

The CLI automatically supports new providers once they're registered with the factory. Test with:

```bash
homeostasis llm set-key myllm --key your-api-key
homeostasis llm validate-key myllm
homeostasis llm test-providers --providers myllm
```

### Step 5: Add Configuration Support

Update the default configuration schema if needed:

```python
# In api_key_manager.py, update DEFAULT_CONFIG
DEFAULT_CONFIG = {
    "active_provider": "anthropic",
    "fallback_order": ["anthropic", "openai", "openrouter", "myllm"],
    "enable_fallback": True,
    # ... rest of config
}
```

## Provider Implementation Guidelines

### Message Format Handling

Different providers handle messages differently:

- **OpenAI/OpenRouter**: Support system messages in the messages array
- **Anthropic**: System messages go in a separate `system` field
- **Your Provider**: Implement according to your API specification

### Error Handling

Always wrap provider-specific errors in `LLMError`:

```python
try:
    # Provider API call
    response = self._session.post(...)
except requests.RequestException as e:
    raise LLMError(f"Provider API request failed: {e}")
except KeyError as e:
    raise LLMError(f"Unexpected response format: {e}")
```

### Authentication Patterns

Common authentication patterns:

```python
# Bearer token (OpenAI, OpenRouter)
headers = {'Authorization': f'Bearer {api_key}'}

# Custom header (Anthropic)
headers = {'x-api-key': api_key}

# API key in query params
params = {'api_key': api_key}
```

### Response Conversion

Ensure your `_convert_response` method handles:

- Content extraction (text, streaming, multi-part responses)
- Usage statistics (tokens, cost)
- Provider-specific metadata
- Error conditions

### Validation Strategy

For `validate_key`, use the lightest possible request:

```python
def validate_key(self, api_key: str) -> bool:
    """Use minimal request for key validation."""
    # Option 1: Models endpoint (if available)
    response = requests.get(f"{self.base_url}/models", headers=headers)
    
    # Option 2: Minimal completion request
    minimal_request = {
        "model": self.default_model,
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 1
    }
    response = requests.post(f"{self.base_url}/chat", json=minimal_request, headers=headers)
    
    return response.status_code == 200
```

## Testing Your Provider

### Unit Tests

Create tests for your provider in `tests/`:

```python
import unittest
from unittest.mock import Mock, patch
from modules.llm_integration.provider_abstraction import MyLLMProvider, LLMRequest, LLMMessage

class TestMyLLMProvider(unittest.TestCase):
    def setUp(self):
        self.provider = MyLLMProvider("test-key")
    
    @patch('requests.Session.post')
    def test_complete_success(self, mock_post):
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'choices': [{'message': {'content': 'Test response'}}],
            'model': 'myllm-v1'
        }
        mock_post.return_value = mock_response
        
        request = LLMRequest(messages=[LLMMessage(role="user", content="Hello")])
        response = self.provider.complete(request)
        
        self.assertEqual(response.content, "Test response")
        self.assertEqual(response.provider, "myllm")
    
    @patch('requests.get')
    def test_validate_key(self, mock_get):
        mock_get.return_value.status_code = 200
        self.assertTrue(self.provider.validate_key("valid-key"))
        
        mock_get.return_value.status_code = 401
        self.assertFalse(self.provider.validate_key("invalid-key"))
```

### Integration Tests

Test with the full LLM manager:

```python
from modules.llm_integration.api_key_manager import APIKeyManager
from modules.llm_integration.provider_abstraction import LLMManager

# Test integration
key_manager = APIKeyManager()
key_manager.set_key('myllm', 'your-test-key')

llm_manager = LLMManager(key_manager)
providers = llm_manager.get_available_providers()
assert 'myllm' in providers
```

## Advanced Features

### Custom Model Mapping

For providers with different model naming:

```python
def _map_model_name(self, unified_model: str) -> str:
    """Map unified model names to provider-specific names."""
    mapping = {
        'gpt-3.5-turbo': 'myllm-chat-v1',
        'gpt-4': 'myllm-advanced-v1'
    }
    return mapping.get(unified_model, unified_model)
```

### Streaming Support

If your provider supports streaming:

```python
def complete_stream(self, request: LLMRequest) -> Iterator[str]:
    """Stream completion responses."""
    data = self._prepare_request_data(request)
    data['stream'] = True
    
    response = self._session.post(
        f"{self.base_url}/chat/completions",
        json=data,
        stream=True
    )
    
    for line in response.iter_lines():
        if line.startswith(b'data: '):
            chunk = json.loads(line[6:])
            if 'choices' in chunk:
                yield chunk['choices'][0]['delta'].get('content', '')
```

### Rate Limiting

Implement provider-specific rate limiting:

```python
import time
from collections import defaultdict

class MyLLMProvider(LLMProviderInterface):
    def __init__(self, api_key: str, **kwargs):
        super().__init__(api_key, **kwargs)
        self._last_request_time = 0
        self._min_request_interval = 0.1  # 100ms between requests
    
    def _rate_limit(self):
        """Enforce rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()
    
    def complete(self, request: LLMRequest) -> LLMResponse:
        self._rate_limit()
        # ... rest of implementation
```

## Provider-Specific Considerations

### Context Length Handling

Handle different context limits:

```python
def _validate_context_length(self, request: LLMRequest) -> None:
    """Validate request fits within context limits."""
    model_limits = {
        'myllm-v1': 4096,
        'myllm-large': 32768
    }
    
    model = request.model or self.default_model
    limit = model_limits.get(model, 4096)
    
    # Estimate token count (simplified)
    estimated_tokens = sum(len(msg.content.split()) for msg in request.messages) * 1.3
    
    if estimated_tokens > limit:
        raise LLMError(f"Request exceeds context limit for {model}: {estimated_tokens} > {limit}")
```

### Custom Parameters

Support provider-specific parameters:

```python
@dataclass
class MyLLMRequest(LLMRequest):
    """Extended request with provider-specific options."""
    custom_param: Optional[str] = None
    special_mode: bool = False

def complete(self, request: Union[LLMRequest, MyLLMRequest]) -> LLMResponse:
    data = self._prepare_base_data(request)
    
    # Add provider-specific parameters
    if isinstance(request, MyLLMRequest):
        if request.custom_param:
            data['custom_param'] = request.custom_param
        data['special_mode'] = request.special_mode
    
    # ... rest of implementation
```

## Contributing Back

When you've implemented a new provider:

1. **Test thoroughly** with unit and integration tests
2. **Document the provider** in this guide
3. **Submit a pull request** with:
   - Provider implementation
   - Tests
   - Documentation updates
   - Example usage

## Troubleshooting

### Common Issues

**Provider not recognized:**
- Ensure it's added to `ProviderFactory.get_supported_providers()`
- Check the provider name matches exactly (case-sensitive)

**Authentication errors:**
- Verify the authentication pattern matches the provider's API
- Test with a minimal request first

**Response parsing errors:**
- Check the provider's actual response format
- Add logging to see raw responses during development

**Rate limiting issues:**
- Implement appropriate delays between requests
- Handle rate limit responses gracefully

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Your provider will now log detailed request/response information
```

## Summary

Adding a new LLM provider involves:

1. **Implement `LLMProviderInterface`** with all required methods
2. **Register with `ProviderFactory`** for automatic discovery
3. **Add API key management** support with appropriate prefixes
4. **Test thoroughly** with unit and integration tests
5. **Handle provider-specific quirks** (authentication, message format, etc.)

The abstraction layer handles multi-provider logic, failover, and configuration automatically once your provider is properly integrated.