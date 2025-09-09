#!/usr/bin/env python3
"""
Built-in provider plugins for LLM integration.

Contains plugin implementations for OpenAI, Anthropic, and OpenRouter.
"""

from typing import Any, Dict

from .provider_abstraction import (AnthropicProvider, LLMProviderInterface,
                                   OpenAIProvider, OpenRouterProvider)
from .provider_registry import (ProviderCapability, ProviderMetadata,
                                ProviderPlugin)


class OpenAIPlugin(ProviderPlugin):
    """Plugin for OpenAI provider."""

    def get_metadata(self) -> ProviderMetadata:
        """Get OpenAI provider metadata."""
        return ProviderMetadata(
            name="openai",
            display_name="OpenAI",
            description="OpenAI GPT models including GPT-3.5 and GPT-4 series",
            version="1.0.0",
            author="Homeostasis Team",
            homepage="https://openai.com",
            documentation="https://docs.openai.com",
            capabilities=[
                ProviderCapability(
                    "chat_completion", "Chat-based text completion", required=True
                ),
                ProviderCapability("function_calling", "Function calling support"),
                ProviderCapability("streaming", "Response streaming"),
                ProviderCapability("vision", "Image understanding (GPT-4V)"),
                ProviderCapability("code_interpreter", "Code execution and analysis"),
            ],
            supported_models=[
                "gpt-3.5-turbo",
                "gpt-3.5-turbo-16k",
                "gpt-4",
                "gpt-4-turbo",
                "gpt-4-turbo-preview",
                "gpt-4-vision-preview",
                "gpt-4o",
                "gpt-4o-mini",
            ],
            model_families={
                "gpt-3.5": ["gpt-3.5-turbo", "gpt-3.5-turbo-16k"],
                "gpt-4": [
                    "gpt-4",
                    "gpt-4-turbo",
                    "gpt-4-turbo-preview",
                    "gpt-4-vision-preview",
                ],
                "gpt-4o": ["gpt-4o", "gpt-4o-mini"],
            },
            pricing_tier="medium",
            latency_class="low",
            reliability_score=0.95,
            default_model="gpt-3.5-turbo",
            authentication_methods=["api_key"],
            rate_limits={
                "requests_per_minute": 3500,
                "tokens_per_minute": 90000,
                "requests_per_day": 10000,
            },
            context_limits={
                "gpt-3.5-turbo": 4096,
                "gpt-3.5-turbo-16k": 16384,
                "gpt-4": 8192,
                "gpt-4-turbo": 128000,
                "gpt-4o": 128000,
            },
            features={
                "structured_output",
                "json_mode",
                "function_calling",
                "parallel_function_calling",
                "vision_understanding",
                "code_execution",
                "web_browsing",
            },
            tags={"commercial", "popular", "well_documented", "feature_rich"},
        )

    def create_provider(self, api_key: str, **kwargs) -> LLMProviderInterface:
        """Create OpenAI provider instance."""
        base_url = kwargs.get("base_url", "https://api.openai.com/v1")
        return OpenAIProvider(api_key, base_url)

    def validate_configuration(self, config: Dict[str, Any]) -> bool:
        """Validate OpenAI configuration."""
        required_fields = ["api_key"]

        for field in required_fields:
            if field not in config or not config[field]:
                return False

        # Validate base_url if provided
        if "base_url" in config:
            base_url = config["base_url"]
            if not isinstance(base_url, str) or not base_url.startswith(
                ("http://", "https://")
            ):
                return False

        return True

    def get_default_configuration(self) -> Dict[str, Any]:
        """Get default OpenAI configuration."""
        return {
            "base_url": "https://api.openai.com/v1",
            "timeout": 30,
            "max_retries": 3,
        }

    def get_configuration_schema(self) -> Dict[str, Any]:
        """Get OpenAI configuration schema."""
        return {
            "type": "object",
            "properties": {
                "api_key": {
                    "type": "string",
                    "description": "OpenAI API key",
                    "minLength": 1,
                },
                "base_url": {
                    "type": "string",
                    "description": "Base URL for API requests",
                    "default": "https://api.openai.com/v1",
                },
                "timeout": {
                    "type": "number",
                    "description": "Request timeout in seconds",
                    "default": 30,
                },
                "max_retries": {
                    "type": "integer",
                    "description": "Maximum number of retries",
                    "default": 3,
                },
            },
            "required": ["api_key"],
        }


class AnthropicPlugin(ProviderPlugin):
    """Plugin for Anthropic provider."""

    def get_metadata(self) -> ProviderMetadata:
        """Get Anthropic provider metadata."""
        return ProviderMetadata(
            name="anthropic",
            display_name="Anthropic",
            description="Anthropic Claude models with strong reasoning and safety features",
            version="1.0.0",
            author="Homeostasis Team",
            homepage="https://anthropic.com",
            documentation="https://docs.anthropic.com",
            capabilities=[
                ProviderCapability(
                    "chat_completion", "Chat-based text completion", required=True
                ),
                ProviderCapability("long_context", "Long context understanding"),
                ProviderCapability("reasoning", "Strong reasoning capabilities"),
                ProviderCapability("safety", "Built-in safety measures"),
                ProviderCapability("constitutional_ai", "Constitutional AI training"),
            ],
            supported_models=[
                "claude-3-haiku-20240307",
                "claude-3-sonnet-20240229",
                "claude-3-opus-20240229",
                "claude-3-5-sonnet-20241022",
                "claude-2.1",
                "claude-2.0",
            ],
            model_families={
                "claude-3": [
                    "claude-3-haiku-20240307",
                    "claude-3-sonnet-20240229",
                    "claude-3-opus-20240229",
                    "claude-3-5-sonnet-20241022",
                ],
                "claude-2": ["claude-2.1", "claude-2.0"],
            },
            pricing_tier="medium",
            latency_class="medium",
            reliability_score=0.92,
            default_model="claude-3-haiku-20240307",
            authentication_methods=["api_key"],
            rate_limits={
                "requests_per_minute": 1000,
                "tokens_per_minute": 100000,
                "requests_per_day": 50000,
            },
            context_limits={
                "claude-3-haiku-20240307": 200000,
                "claude-3-sonnet-20240229": 200000,
                "claude-3-opus-20240229": 200000,
                "claude-3-5-sonnet-20241022": 200000,
                "claude-2.1": 200000,
                "claude-2.0": 100000,
            },
            features={
                "long_context",
                "safety_filtering",
                "constitutional_ai",
                "reasoning",
                "analysis",
                "writing_assistance",
            },
            tags={"safety_focused", "long_context", "reasoning", "ethical"},
        )

    def create_provider(self, api_key: str, **kwargs) -> LLMProviderInterface:
        """Create Anthropic provider instance."""
        base_url = kwargs.get("base_url", "https://api.anthropic.com")
        return AnthropicProvider(api_key, base_url)

    def validate_configuration(self, config: Dict[str, Any]) -> bool:
        """Validate Anthropic configuration."""
        required_fields = ["api_key"]

        for field in required_fields:
            if field not in config or not config[field]:
                return False

        # Validate base_url if provided
        if "base_url" in config:
            base_url = config["base_url"]
            if not isinstance(base_url, str) or not base_url.startswith(
                ("http://", "https://")
            ):
                return False

        return True

    def get_default_configuration(self) -> Dict[str, Any]:
        """Get default Anthropic configuration."""
        return {
            "base_url": "https://api.anthropic.com",
            "timeout": 30,
            "max_retries": 3,
            "anthropic_version": "2023-06-01",
        }

    def get_configuration_schema(self) -> Dict[str, Any]:
        """Get Anthropic configuration schema."""
        return {
            "type": "object",
            "properties": {
                "api_key": {
                    "type": "string",
                    "description": "Anthropic API key",
                    "minLength": 1,
                },
                "base_url": {
                    "type": "string",
                    "description": "Base URL for API requests",
                    "default": "https://api.anthropic.com",
                },
                "timeout": {
                    "type": "number",
                    "description": "Request timeout in seconds",
                    "default": 30,
                },
                "max_retries": {
                    "type": "integer",
                    "description": "Maximum number of retries",
                    "default": 3,
                },
                "anthropic_version": {
                    "type": "string",
                    "description": "Anthropic API version",
                    "default": "2023-06-01",
                },
            },
            "required": ["api_key"],
        }


class OpenRouterPlugin(ProviderPlugin):
    """Plugin for OpenRouter provider."""

    def get_metadata(self) -> ProviderMetadata:
        """Get OpenRouter provider metadata."""
        return ProviderMetadata(
            name="openrouter",
            display_name="OpenRouter",
            description="OpenRouter API aggregator providing access to multiple LLM providers",
            version="1.0.0",
            author="Homeostasis Team",
            homepage="https://openrouter.ai",
            documentation="https://openrouter.ai/docs",
            capabilities=[
                ProviderCapability(
                    "chat_completion", "Chat-based text completion", required=True
                ),
                ProviderCapability(
                    "multi_provider", "Access to multiple LLM providers"
                ),
                ProviderCapability(
                    "cost_optimization", "Cost-optimized model selection"
                ),
                ProviderCapability("fallback", "Automatic fallback between models"),
                ProviderCapability("unified_api", "Unified API for multiple providers"),
            ],
            supported_models=[
                "anthropic/claude-3-haiku",
                "anthropic/claude-3-sonnet",
                "anthropic/claude-3-opus",
                "anthropic/claude-3-5-sonnet",
                "openai/gpt-3.5-turbo",
                "openai/gpt-4",
                "openai/gpt-4-turbo",
                "openai/gpt-4o",
                "meta-llama/llama-3-8b-instruct",
                "meta-llama/llama-3-70b-instruct",
                "google/gemini-pro",
                "mistralai/mistral-7b-instruct",
                "cohere/command-r-plus",
            ],
            model_families={
                "anthropic": [
                    "anthropic/claude-3-haiku",
                    "anthropic/claude-3-sonnet",
                    "anthropic/claude-3-opus",
                    "anthropic/claude-3-5-sonnet",
                ],
                "openai": [
                    "openai/gpt-3.5-turbo",
                    "openai/gpt-4",
                    "openai/gpt-4-turbo",
                    "openai/gpt-4o",
                ],
                "llama": [
                    "meta-llama/llama-3-8b-instruct",
                    "meta-llama/llama-3-70b-instruct",
                ],
                "google": ["google/gemini-pro"],
                "mistral": ["mistralai/mistral-7b-instruct"],
                "cohere": ["cohere/command-r-plus"],
            },
            pricing_tier="low",
            latency_class="medium",
            reliability_score=0.88,
            default_model="anthropic/claude-3-haiku",
            authentication_methods=["api_key"],
            rate_limits={
                "requests_per_minute": 200,
                "tokens_per_minute": 20000,
                "requests_per_day": 100000,
            },
            context_limits={
                "anthropic/claude-3-haiku": 200000,
                "anthropic/claude-3-sonnet": 200000,
                "anthropic/claude-3-opus": 200000,
                "openai/gpt-3.5-turbo": 4096,
                "openai/gpt-4": 8192,
                "openai/gpt-4-turbo": 128000,
            },
            features={
                "multi_provider_access",
                "cost_optimization",
                "unified_billing",
                "model_comparison",
                "automatic_fallback",
                "usage_analytics",
            },
            tags={"aggregator", "cost_effective", "multiple_providers", "unified"},
        )

    def create_provider(self, api_key: str, **kwargs) -> LLMProviderInterface:
        """Create OpenRouter provider instance."""
        base_url = kwargs.get("base_url", "https://openrouter.ai/api/v1")
        return OpenRouterProvider(api_key, base_url)

    def validate_configuration(self, config: Dict[str, Any]) -> bool:
        """Validate OpenRouter configuration."""
        required_fields = ["api_key"]

        for field in required_fields:
            if field not in config or not config[field]:
                return False

        # Validate base_url if provided
        if "base_url" in config:
            base_url = config["base_url"]
            if not isinstance(base_url, str) or not base_url.startswith(
                ("http://", "https://")
            ):
                return False

        return True

    def get_default_configuration(self) -> Dict[str, Any]:
        """Get default OpenRouter configuration."""
        return {
            "base_url": "https://openrouter.ai/api/v1",
            "timeout": 30,
            "max_retries": 3,
            "http_referer": "https://homeostasis.dev",
            "x_title": "Homeostasis",
        }

    def get_configuration_schema(self) -> Dict[str, Any]:
        """Get OpenRouter configuration schema."""
        return {
            "type": "object",
            "properties": {
                "api_key": {
                    "type": "string",
                    "description": "OpenRouter API key",
                    "minLength": 1,
                },
                "base_url": {
                    "type": "string",
                    "description": "Base URL for API requests",
                    "default": "https://openrouter.ai/api/v1",
                },
                "timeout": {
                    "type": "number",
                    "description": "Request timeout in seconds",
                    "default": 30,
                },
                "max_retries": {
                    "type": "integer",
                    "description": "Maximum number of retries",
                    "default": 3,
                },
                "http_referer": {
                    "type": "string",
                    "description": "HTTP Referer header",
                    "default": "https://homeostasis.dev",
                },
                "x_title": {
                    "type": "string",
                    "description": "X-Title header",
                    "default": "Homeostasis",
                },
            },
            "required": ["api_key"],
        }
