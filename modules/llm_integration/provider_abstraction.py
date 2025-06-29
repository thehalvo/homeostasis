#!/usr/bin/env python3
"""
Provider abstraction layer for LLM integration.

Provides a unified interface for different LLM providers (OpenAI, Anthropic, OpenRouter).
"""

import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import requests


@dataclass
class LLMMessage:
    """Represents a message in a conversation."""
    role: str  # 'user', 'assistant', 'system'
    content: str


@dataclass
class LLMResponse:
    """Represents a response from an LLM provider."""
    content: str
    provider: str
    model: str
    usage: Optional[Dict[str, int]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class LLMRequest:
    """Represents a request to an LLM provider."""
    messages: List[LLMMessage]
    model: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    system_prompt: Optional[str] = None


class LLMError(Exception):
    """Base exception for LLM provider errors."""
    pass


class LLMProviderInterface(ABC):
    """Abstract interface for LLM providers."""
    
    @abstractmethod
    def complete(self, request: LLMRequest) -> LLMResponse:
        """
        Complete a chat request.
        
        Args:
            request: The LLM request
            
        Returns:
            LLM response
            
        Raises:
            LLMError: If the request fails
        """
        pass
    
    @abstractmethod
    def validate_key(self, api_key: str) -> bool:
        """
        Validate an API key.
        
        Args:
            api_key: API key to validate
            
        Returns:
            True if valid
            
        Raises:
            LLMError: If validation fails
        """
        pass
    
    @abstractmethod
    def get_available_models(self) -> List[str]:
        """
        Get list of available models.
        
        Returns:
            List of model names
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        pass
    
    @property
    @abstractmethod
    def default_model(self) -> str:
        """Default model for this provider."""
        pass


class OpenAIProvider(LLMProviderInterface):
    """OpenAI API provider."""
    
    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1"):
        """
        Initialize OpenAI provider.
        
        Args:
            api_key: OpenAI API key
            base_url: Base URL for API (default: OpenAI)
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
        return "openai"
    
    @property
    def default_model(self) -> str:
        return "gpt-3.5-turbo"
    
    def complete(self, request: LLMRequest) -> LLMResponse:
        """Complete a chat request using OpenAI API."""
        # Convert messages to OpenAI format
        messages = []
        
        # Add system prompt if provided
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        
        # Add conversation messages
        for msg in request.messages:
            messages.append({"role": msg.role, "content": msg.content})
        
        # Prepare request data
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
            
            if 'choices' not in result or not result['choices']:
                raise LLMError("No response choices returned")
            
            choice = result['choices'][0]
            content = choice['message']['content']
            
            return LLMResponse(
                content=content,
                provider=self.name,
                model=result.get('model', data['model']),
                usage=result.get('usage'),
                metadata={
                    'finish_reason': choice.get('finish_reason'),
                    'response_id': result.get('id')
                }
            )
            
        except requests.RequestException as e:
            raise LLMError(f"OpenAI API request failed: {e}")
        except Exception as e:
            raise LLMError(f"OpenAI completion failed: {e}")
    
    def validate_key(self, api_key: str) -> bool:
        """Validate OpenAI API key."""
        headers = {'Authorization': f'Bearer {api_key}'}
        
        try:
            response = requests.get(
                f"{self.base_url}/models",
                headers=headers,
                timeout=10
            )
            return response.status_code == 200
        except Exception:
            return False
    
    def get_available_models(self) -> List[str]:
        """Get available OpenAI models."""
        try:
            response = self._session.get(f"{self.base_url}/models", timeout=10)
            response.raise_for_status()
            
            result = response.json()
            return [model['id'] for model in result.get('data', [])]
        except Exception:
            return [self.default_model]


class AnthropicProvider(LLMProviderInterface):
    """Anthropic API provider."""
    
    def __init__(self, api_key: str, base_url: str = "https://api.anthropic.com"):
        """
        Initialize Anthropic provider.
        
        Args:
            api_key: Anthropic API key
            base_url: Base URL for API
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self._session = requests.Session()
        self._session.headers.update({
            'x-api-key': api_key,
            'Content-Type': 'application/json',
            'anthropic-version': '2023-06-01'
        })
    
    @property
    def name(self) -> str:
        return "anthropic"
    
    @property
    def default_model(self) -> str:
        return "claude-3-haiku-20240307"
    
    def complete(self, request: LLMRequest) -> LLMResponse:
        """Complete a chat request using Anthropic API."""
        # Convert messages to Anthropic format
        messages = []
        system_message = request.system_prompt
        
        for msg in request.messages:
            if msg.role == 'system':
                # Anthropic handles system messages separately
                system_message = msg.content
            else:
                messages.append({"role": msg.role, "content": msg.content})
        
        # Prepare request data
        data = {
            "model": request.model or self.default_model,
            "messages": messages,
            "max_tokens": request.max_tokens or 1000,
            "temperature": request.temperature or 0.7
        }
        
        if system_message:
            data["system"] = system_message
        
        try:
            response = self._session.post(
                f"{self.base_url}/v1/messages",
                json=data,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            
            if 'content' not in result or not result['content']:
                raise LLMError("No content in response")
            
            # Extract text content
            content = ""
            for content_block in result['content']:
                if content_block.get('type') == 'text':
                    content += content_block.get('text', '')
            
            return LLMResponse(
                content=content,
                provider=self.name,
                model=result.get('model', data['model']),
                usage=result.get('usage'),
                metadata={
                    'stop_reason': result.get('stop_reason'),
                    'response_id': result.get('id')
                }
            )
            
        except requests.RequestException as e:
            raise LLMError(f"Anthropic API request failed: {e}")
        except Exception as e:
            raise LLMError(f"Anthropic completion failed: {e}")
    
    def validate_key(self, api_key: str) -> bool:
        """Validate Anthropic API key."""
        headers = {
            'x-api-key': api_key,
            'Content-Type': 'application/json',
            'anthropic-version': '2023-06-01'
        }
        
        # Minimal request for validation
        data = {
            'model': self.default_model,
            'max_tokens': 1,
            'messages': [{'role': 'user', 'content': 'Hi'}]
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/v1/messages",
                headers=headers,
                json=data,
                timeout=10
            )
            return response.status_code == 200
        except Exception:
            return False
    
    def get_available_models(self) -> List[str]:
        """Get available Anthropic models."""
        # Anthropic doesn't have a models endpoint, return known models
        return [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-2.1",
            "claude-2.0"
        ]


class OpenRouterProvider(LLMProviderInterface):
    """OpenRouter API provider."""
    
    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai/api/v1"):
        """
        Initialize OpenRouter provider.
        
        Args:
            api_key: OpenRouter API key
            base_url: Base URL for API
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self._session = requests.Session()
        self._session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
            'HTTP-Referer': 'https://homeostasis.dev',
            'X-Title': 'Homeostasis'
        })
    
    @property
    def name(self) -> str:
        return "openrouter"
    
    @property
    def default_model(self) -> str:
        return "anthropic/claude-3-haiku"
    
    def complete(self, request: LLMRequest) -> LLMResponse:
        """Complete a chat request using OpenRouter API."""
        # Convert messages to OpenAI-compatible format
        messages = []
        
        # Add system prompt if provided
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        
        # Add conversation messages
        for msg in request.messages:
            messages.append({"role": msg.role, "content": msg.content})
        
        # Prepare request data
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
            
            if 'choices' not in result or not result['choices']:
                raise LLMError("No response choices returned")
            
            choice = result['choices'][0]
            content = choice['message']['content']
            
            return LLMResponse(
                content=content,
                provider=self.name,
                model=result.get('model', data['model']),
                usage=result.get('usage'),
                metadata={
                    'finish_reason': choice.get('finish_reason'),
                    'response_id': result.get('id')
                }
            )
            
        except requests.RequestException as e:
            raise LLMError(f"OpenRouter API request failed: {e}")
        except Exception as e:
            raise LLMError(f"OpenRouter completion failed: {e}")
    
    def validate_key(self, api_key: str) -> bool:
        """Validate OpenRouter API key."""
        headers = {'Authorization': f'Bearer {api_key}'}
        
        try:
            response = requests.get(
                f"{self.base_url}/models",
                headers=headers,
                timeout=10
            )
            return response.status_code == 200
        except Exception:
            return False
    
    def get_available_models(self) -> List[str]:
        """Get available OpenRouter models."""
        try:
            response = self._session.get(f"{self.base_url}/models", timeout=10)
            response.raise_for_status()
            
            result = response.json()
            return [model['id'] for model in result.get('data', [])]
        except Exception:
            return [self.default_model]


class ProviderFactory:
    """Factory for creating LLM providers."""
    
    @staticmethod
    def create_provider(provider_name: str, api_key: str, **kwargs) -> LLMProviderInterface:
        """
        Create a provider instance.
        
        Args:
            provider_name: Name of the provider
            api_key: API key for the provider
            **kwargs: Additional provider-specific arguments
            
        Returns:
            Provider instance
            
        Raises:
            ValueError: If provider is unknown
        """
        provider_name = provider_name.lower()
        
        if provider_name == "openai":
            return OpenAIProvider(api_key, **kwargs)
        elif provider_name == "anthropic":
            return AnthropicProvider(api_key, **kwargs)
        elif provider_name == "openrouter":
            return OpenRouterProvider(api_key, **kwargs)
        else:
            raise ValueError(f"Unknown provider: {provider_name}")
    
    @staticmethod
    def get_supported_providers() -> List[str]:
        """Get list of supported providers."""
        return ["openai", "anthropic", "openrouter"]


class LLMManager:
    """
    High-level manager for LLM operations with automatic provider selection.
    """
    
    def __init__(self, api_key_manager):
        """
        Initialize LLM manager.
        
        Args:
            api_key_manager: APIKeyManager instance
        """
        self.api_key_manager = api_key_manager
        self._providers = {}
        self._provider_order = ["anthropic", "openai", "openrouter"]  # Default preference order
    
    def _get_provider(self, provider_name: str, use_openrouter_proxy: bool = False) -> Optional[LLMProviderInterface]:
        """Get or create a provider instance."""
        cache_key = f"{provider_name}{'_via_openrouter' if use_openrouter_proxy else ''}"
        
        if cache_key in self._providers:
            return self._providers[cache_key]
        
        # Check for OpenRouter unified mode
        if use_openrouter_proxy:
            openrouter_key = self.api_key_manager.get_key('openrouter')
            if not openrouter_key:
                return None
            
            try:
                # Create OpenRouter provider with unified endpoint
                provider = ProviderFactory.create_provider('openrouter', openrouter_key)
                
                # Get model mapping for the target provider
                unified_config = self.api_key_manager.get_openrouter_unified_config()
                model_mapping = unified_config.get('fallback_model_mapping', {})
                
                # Override default model for the target provider
                if provider_name in model_mapping:
                    provider._default_model = model_mapping[provider_name]
                
                self._providers[cache_key] = provider
                return provider
            except Exception:
                return None
        else:
            # Regular provider creation
            api_key = self.api_key_manager.get_key(provider_name)
            if not api_key:
                return None
            
            try:
                provider = ProviderFactory.create_provider(provider_name, api_key)
                self._providers[cache_key] = provider
                return provider
            except Exception:
                return None
    
    def complete(self, request: LLMRequest, preferred_provider: Optional[str] = None) -> LLMResponse:
        """
        Complete a request using the best available provider.
        
        Args:
            request: LLM request
            preferred_provider: Preferred provider name
            
        Returns:
            LLM response
            
        Raises:
            LLMError: If no providers are available or all fail
        """
        # Get configuration
        active_provider = preferred_provider or self.api_key_manager.get_active_provider()
        fallback_enabled = self.api_key_manager.is_fallback_enabled()
        fallback_order = self.api_key_manager.get_fallback_order()
        unified_config = self.api_key_manager.get_openrouter_unified_config()
        
        # Determine provider order
        if active_provider:
            # Start with active provider, then fallback order
            provider_order = [active_provider]
            if fallback_enabled:
                for provider in fallback_order:
                    if provider != active_provider:
                        provider_order.append(provider)
        else:
            # Use fallback order as primary order
            provider_order = fallback_order if fallback_enabled else self._provider_order
        
        last_error = None
        
        for provider_name in provider_order:
            # Check if OpenRouter unified mode should be used
            use_openrouter_proxy = False
            if unified_config.get('enabled', False):
                if provider_name == 'anthropic' and unified_config.get('proxy_to_anthropic', False):
                    use_openrouter_proxy = True
                elif provider_name == 'openai' and unified_config.get('proxy_to_openai', False):
                    use_openrouter_proxy = True
            
            # Try direct provider first
            provider = self._get_provider(provider_name, use_openrouter_proxy=False)
            if provider:
                try:
                    return provider.complete(request)
                except LLMError as e:
                    last_error = e
                    # If direct provider fails and unified mode is available, try OpenRouter proxy
                    if use_openrouter_proxy:
                        try:
                            proxy_provider = self._get_provider(provider_name, use_openrouter_proxy=True)
                            if proxy_provider:
                                return proxy_provider.complete(request)
                        except LLMError:
                            pass  # Continue to next provider
            
            # If direct provider is not available but OpenRouter proxy is enabled, try proxy
            elif use_openrouter_proxy:
                try:
                    proxy_provider = self._get_provider(provider_name, use_openrouter_proxy=True)
                    if proxy_provider:
                        return proxy_provider.complete(request)
                except LLMError as e:
                    last_error = e
        
        # No providers worked
        if last_error:
            raise last_error
        else:
            raise LLMError("No LLM providers are configured or available")
    
    def get_available_providers(self) -> List[str]:
        """Get list of available providers (with valid API keys)."""
        available = []
        for provider_name in ProviderFactory.get_supported_providers():
            if self.api_key_manager.get_key(provider_name):
                available.append(provider_name)
        return available
    
    def get_recommended_provider_order(self) -> List[str]:
        """
        Get recommended provider order based on current policies.
        
        Returns:
            List of provider names in recommended order
        """
        policies = self.api_key_manager.get_provider_policies()
        available_providers = self.get_available_providers()
        
        # Provider characteristics (simplified scoring)
        provider_scores = {
            'anthropic': {
                'cost': 2,      # Medium cost
                'latency': 2,   # Medium latency  
                'reliability': 3  # High reliability
            },
            'openai': {
                'cost': 2,      # Medium cost
                'latency': 3,   # Low latency
                'reliability': 3  # High reliability
            },
            'openrouter': {
                'cost': 1,      # Lower cost (aggregator)
                'latency': 1,   # Higher latency
                'reliability': 2  # Medium reliability
            }
        }
        
        # Weight preferences (higher is better)
        preference_weights = {
            'low': 1, 'balanced': 2, 'high': 3
        }
        
        # Calculate scores for each available provider
        scored_providers = []
        for provider in available_providers:
            if provider in provider_scores:
                score = 0
                chars = provider_scores[provider]
                
                # Cost preference (inverted - lower cost is better when preference is low)
                cost_pref = policies.get('cost_preference', 'balanced')
                if cost_pref == 'low':
                    score += (4 - chars['cost']) * 2  # Prefer lower cost
                elif cost_pref == 'high':
                    score += chars['cost'] * 2  # Don't mind higher cost
                else:
                    score += 2  # Balanced
                
                # Latency preference (lower latency is better when preference is low)
                latency_pref = policies.get('latency_preference', 'balanced')
                if latency_pref == 'low':
                    score += chars['latency'] * 2  # Prefer lower latency
                else:
                    score += preference_weights[latency_pref]
                
                # Reliability preference
                reliability_pref = policies.get('reliability_preference', 'high')
                score += chars['reliability'] * preference_weights[reliability_pref]
                
                scored_providers.append((provider, score))
        
        # Sort by score (highest first)
        scored_providers.sort(key=lambda x: x[1], reverse=True)
        
        return [provider for provider, _ in scored_providers]
    
    def auto_configure_fallback_order(self) -> None:
        """
        Automatically configure fallback order based on current policies.
        """
        recommended_order = self.get_recommended_provider_order()
        if recommended_order:
            self.api_key_manager.set_fallback_order(recommended_order)
            print(f"✓ Auto-configured fallback order based on current policies")
        else:
            print("⚠️  No providers available for auto-configuration")