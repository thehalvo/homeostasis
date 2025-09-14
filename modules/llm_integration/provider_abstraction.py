#!/usr/bin/env python3
"""
Provider abstraction layer for LLM integration.

Provides a unified interface for different LLM providers (OpenAI, Anthropic, OpenRouter).
"""

import logging
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

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

    def __init__(
        self,
        message: str,
        error_type: str = "unknown",
        retryable: bool = True,
        provider: str = "",
    ):
        super().__init__(message)
        self.error_type = error_type
        self.retryable = retryable
        self.provider = provider


class ErrorType(Enum):
    """Types of errors that can occur during LLM operations."""

    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    NETWORK = "network"
    AUTHENTICATION = "authentication"
    QUOTA_EXCEEDED = "quota_exceeded"
    MODEL_OVERLOADED = "model_overloaded"
    INVALID_REQUEST = "invalid_request"
    INTERNAL_ERROR = "internal_error"
    UNKNOWN = "unknown"


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    backoff_factor: float = 1.0
    retry_on_errors: List[ErrorType] = field(
        default_factory=lambda: [
            ErrorType.RATE_LIMIT,
            ErrorType.TIMEOUT,
            ErrorType.NETWORK,
            ErrorType.MODEL_OVERLOADED,
            ErrorType.INTERNAL_ERROR,
        ]
    )
    # Retry budgets and cooldown periods
    retry_budget_window: float = 300.0  # 5 minutes
    max_retries_per_window: int = 10
    global_retry_cooldown: float = 60.0  # 1 minute cooldown after budget exhaustion
    per_error_cooldown: Dict[ErrorType, float] = field(
        default_factory=lambda: {
            ErrorType.RATE_LIMIT: 120.0,  # 2 minutes
            ErrorType.QUOTA_EXCEEDED: 300.0,  # 5 minutes
            ErrorType.AUTHENTICATION: 0.0,  # No cooldown for auth errors
        }
    )


@dataclass
class ProviderHealthMetrics:
    """Health metrics for a provider."""

    success_count: int = 0
    failure_count: int = 0
    total_requests: int = 0
    average_latency: float = 0.0
    last_success_time: Optional[float] = None
    last_failure_time: Optional[float] = None
    consecutive_failures: int = 0
    circuit_breaker_open: bool = False
    circuit_breaker_open_until: Optional[float] = None
    # Retry budget tracking
    retry_budget_window_start: Optional[float] = None
    retries_in_current_window: int = 0
    global_retry_cooldown_until: Optional[float] = None
    error_specific_cooldowns: Dict[ErrorType, float] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 1.0
        return self.success_count / self.total_requests

    @property
    def failure_rate(self) -> float:
        """Calculate failure rate."""
        return 1.0 - self.success_rate


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
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()
        self._session.headers.update(
            {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        )

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
            "temperature": request.temperature or 0.7,
        }

        try:
            response = self._session.post(
                f"{self.base_url}/chat/completions", json=data, timeout=30
            )
            response.raise_for_status()

            result = response.json()

            if "choices" not in result or not result["choices"]:
                raise LLMError("No response choices returned")

            choice = result["choices"][0]
            content = choice["message"]["content"]

            return LLMResponse(
                content=content,
                provider=self.name,
                model=result.get("model", data["model"]),
                usage=result.get("usage"),
                metadata={
                    "finish_reason": choice.get("finish_reason"),
                    "response_id": result.get("id"),
                },
            )

        except requests.RequestException as e:
            raise LLMError(f"OpenAI API request failed: {e}")
        except Exception as e:
            raise LLMError(f"OpenAI completion failed: {e}")

    def validate_key(self, api_key: str) -> bool:
        """Validate OpenAI API key."""
        headers = {"Authorization": f"Bearer {api_key}"}

        try:
            response = requests.get(
                f"{self.base_url}/models", headers=headers, timeout=10
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
            return [model["id"] for model in result.get("data", [])]
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
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()
        self._session.headers.update(
            {
                "x-api-key": api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01",
            }
        )

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
            if msg.role == "system":
                # Anthropic handles system messages separately
                system_message = msg.content
            else:
                messages.append({"role": msg.role, "content": msg.content})

        # Prepare request data
        data = {
            "model": request.model or self.default_model,
            "messages": messages,
            "max_tokens": request.max_tokens or 1000,
            "temperature": request.temperature or 0.7,
        }

        if system_message:
            data["system"] = system_message

        try:
            response = self._session.post(
                f"{self.base_url}/v1/messages", json=data, timeout=30
            )
            response.raise_for_status()

            result = response.json()

            if "content" not in result or not result["content"]:
                raise LLMError("No content in response")

            # Extract text content
            content = ""
            for content_block in result["content"]:
                if content_block.get("type") == "text":
                    content += content_block.get("text", "")

            return LLMResponse(
                content=content,
                provider=self.name,
                model=result.get("model", data["model"]),
                usage=result.get("usage"),
                metadata={
                    "stop_reason": result.get("stop_reason"),
                    "response_id": result.get("id"),
                },
            )

        except requests.RequestException as e:
            raise LLMError(f"Anthropic API request failed: {e}")
        except Exception as e:
            raise LLMError(f"Anthropic completion failed: {e}")

    def validate_key(self, api_key: str) -> bool:
        """Validate Anthropic API key."""
        headers = {
            "x-api-key": api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        # Minimal request for validation
        data = {
            "model": self.default_model,
            "max_tokens": 1,
            "messages": [{"role": "user", "content": "Hi"}],
        }

        try:
            response = requests.post(
                f"{self.base_url}/v1/messages", headers=headers, json=data, timeout=10
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
            "claude-2.0",
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
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://homeostasis.dev",
                "X-Title": "Homeostasis",
            }
        )

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
            "temperature": request.temperature or 0.7,
        }

        try:
            response = self._session.post(
                f"{self.base_url}/chat/completions", json=data, timeout=30
            )
            response.raise_for_status()

            result = response.json()

            if "choices" not in result or not result["choices"]:
                raise LLMError("No response choices returned")

            choice = result["choices"][0]
            content = choice["message"]["content"]

            return LLMResponse(
                content=content,
                provider=self.name,
                model=result.get("model", data["model"]),
                usage=result.get("usage"),
                metadata={
                    "finish_reason": choice.get("finish_reason"),
                    "response_id": result.get("id"),
                },
            )

        except requests.RequestException as e:
            raise LLMError(f"OpenRouter API request failed: {e}")
        except Exception as e:
            raise LLMError(f"OpenRouter completion failed: {e}")

    def validate_key(self, api_key: str) -> bool:
        """Validate OpenRouter API key."""
        headers = {"Authorization": f"Bearer {api_key}"}

        try:
            response = requests.get(
                f"{self.base_url}/models", headers=headers, timeout=10
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
            return [model["id"] for model in result.get("data", [])]
        except Exception:
            return [self.default_model]


class ProviderFactory:
    """Factory for creating LLM providers."""

    @staticmethod
    def create_provider(
        provider_name: str, api_key: str, **kwargs
    ) -> LLMProviderInterface:
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
        # Try to use the new registry system first
        try:
            from .provider_registry import get_provider_registry

            registry = get_provider_registry()
            instance = registry.create_provider_instance(
                provider_name, api_key, **kwargs
            )
            if instance:
                return instance
        except ImportError:
            pass  # Fall back to legacy system

        # Legacy fallback
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
        # Try to use the new registry system first
        try:
            from .provider_registry import get_provider_registry

            registry = get_provider_registry()
            return registry.list_providers()
        except ImportError:
            pass  # Fall back to legacy system

        # Legacy fallback
        return ["openai", "anthropic", "openrouter"]


class LLMManager:
    """
    High-level manager for LLM operations with automatic provider selection,
    intelligent retry, failover, and health monitoring.
    """

    def __init__(self, api_key_manager, retry_config: Optional[RetryConfig] = None):
        """
        Initialize LLM manager.

        Args:
            api_key_manager: APIKeyManager instance
            retry_config: Configuration for retry behavior
        """
        self.api_key_manager = api_key_manager
        self._providers: Dict[str, LLMProviderInterface] = {}
        self._provider_order = [
            "anthropic",
            "openai",
            "openrouter",
        ]  # Default preference order
        self.retry_config = retry_config or RetryConfig()
        self._provider_health: Dict[str, ProviderHealthMetrics] = (
            {}
        )  # Track health metrics for each provider
        self._circuit_breaker_threshold = (
            5  # Consecutive failures before opening circuit
        )
        self._circuit_breaker_timeout = 300  # 5 minutes
        self.logger = logging.getLogger(__name__)
        # Global retry tracking
        self._global_retry_history: List[float] = []  # List of retry timestamps
        self._last_global_cooldown_check = 0.0

    def _classify_error(
        self, error: Exception, provider_name: str
    ) -> Tuple[ErrorType, bool]:
        """
        Classify an error and determine if it's retryable.

        Args:
            error: The exception that occurred
            provider_name: Name of the provider that generated the error

        Returns:
            Tuple of (error_type, is_retryable)
        """
        error_message = str(error).lower()

        # Rate limiting errors
        if any(
            keyword in error_message
            for keyword in ["rate limit", "too many requests", "quota exceeded"]
        ):
            return ErrorType.RATE_LIMIT, True

        # Timeout errors
        if any(
            keyword in error_message
            for keyword in ["timeout", "timed out", "connection timeout"]
        ):
            return ErrorType.TIMEOUT, True

        # Network errors
        if any(
            keyword in error_message
            for keyword in ["connection", "network", "dns", "ssl"]
        ):
            return ErrorType.NETWORK, True

        # Authentication errors
        if any(
            keyword in error_message
            for keyword in ["unauthorized", "forbidden", "api key", "authentication"]
        ):
            return ErrorType.AUTHENTICATION, False

        # Model overloaded
        if any(
            keyword in error_message for keyword in ["overloaded", "busy", "capacity"]
        ):
            return ErrorType.MODEL_OVERLOADED, True

        # Invalid request format
        if any(
            keyword in error_message
            for keyword in ["invalid", "bad request", "malformed"]
        ):
            return ErrorType.INVALID_REQUEST, False

        # Internal server errors
        if any(
            keyword in error_message for keyword in ["internal", "server error", "500"]
        ):
            return ErrorType.INTERNAL_ERROR, True

        return ErrorType.UNKNOWN, True

    def _calculate_retry_delay(self, attempt: int, error_type: ErrorType) -> float:
        """
        Calculate delay before next retry using exponential backoff.

        Args:
            attempt: Current attempt number (0-based)
            error_type: Type of error that occurred

        Returns:
            Delay in seconds
        """
        base_delay = self.retry_config.base_delay

        # Special handling for rate limits
        if error_type == ErrorType.RATE_LIMIT:
            base_delay *= 2  # Longer delays for rate limits

        # Exponential backoff
        delay = base_delay * (self.retry_config.exponential_base**attempt)
        delay *= self.retry_config.backoff_factor

        # Cap at max delay
        delay = min(delay, self.retry_config.max_delay)

        # Add jitter to avoid thundering herd
        if self.retry_config.jitter:
            jitter_factor = random.uniform(0.5, 1.5)
            delay *= jitter_factor

        return delay

    def _get_provider_health(self, provider_name: str) -> ProviderHealthMetrics:
        """
        Get health metrics for a provider.

        Args:
            provider_name: Name of the provider

        Returns:
            Provider health metrics
        """
        if provider_name not in self._provider_health:
            self._provider_health[provider_name] = ProviderHealthMetrics()
        return self._provider_health[provider_name]

    def _update_provider_health(
        self,
        provider_name: str,
        success: bool,
        latency: float = 0.0,
        error_type: Optional[ErrorType] = None,
    ):
        """
        Update health metrics for a provider.

        Args:
            provider_name: Name of the provider
            success: Whether the request was successful
            latency: Request latency in seconds
            error_type: Type of error if request failed
        """
        health = self._get_provider_health(provider_name)
        current_time = time.time()

        health.total_requests += 1

        if success:
            health.success_count += 1
            health.last_success_time = current_time
            health.consecutive_failures = 0

            # Close circuit breaker if it was open
            if health.circuit_breaker_open:
                health.circuit_breaker_open = False
                health.circuit_breaker_open_until = None
                self.logger.info(f"Circuit breaker closed for provider {provider_name}")
        else:
            health.failure_count += 1
            health.last_failure_time = current_time
            health.consecutive_failures += 1

            # Open circuit breaker if too many consecutive failures
            if health.consecutive_failures >= self._circuit_breaker_threshold:
                health.circuit_breaker_open = True
                health.circuit_breaker_open_until = (
                    current_time + self._circuit_breaker_timeout
                )
                self.logger.warning(
                    f"Circuit breaker opened for provider {provider_name} after {health.consecutive_failures} consecutive failures"
                )

        # Update average latency
        if latency > 0:
            if health.average_latency == 0:
                health.average_latency = latency
            else:
                # Exponential moving average
                health.average_latency = 0.9 * health.average_latency + 0.1 * latency

    def _is_circuit_breaker_open(self, provider_name: str) -> bool:
        """
        Check if circuit breaker is open for a provider.

        Args:
            provider_name: Name of the provider

        Returns:
            True if circuit breaker is open
        """
        health = self._get_provider_health(provider_name)

        if not health.circuit_breaker_open:
            return False

        # Check if timeout has expired
        if health.circuit_breaker_open_until is not None:
            if time.time() >= health.circuit_breaker_open_until:
                # Reset circuit breaker to half-open state
                health.circuit_breaker_open = False
                health.circuit_breaker_open_until = None
                health.consecutive_failures = 0
                self.logger.info(f"Circuit breaker reset for provider {provider_name}")
                return False

        return True

    def _is_retry_budget_exhausted(self, provider_name: str) -> bool:
        """
        Check if retry budget is exhausted for a provider.

        Args:
            provider_name: Name of the provider

        Returns:
            True if retry budget is exhausted
        """
        health = self._get_provider_health(provider_name)
        current_time = time.time()

        # Check global retry cooldown
        if (
            health.global_retry_cooldown_until
            and current_time < health.global_retry_cooldown_until
        ):
            return True

        # Check error-specific cooldowns
        for error_type, cooldown_until in health.error_specific_cooldowns.items():
            if current_time < cooldown_until:
                self.logger.info(
                    f"Provider {provider_name} is in cooldown for {error_type.value} until {cooldown_until - current_time:.1f}s"
                )
                return True

        # Check retry budget window
        if health.retry_budget_window_start is None:
            # Initialize window
            health.retry_budget_window_start = current_time
            health.retries_in_current_window = 0
            return False

        # Check if window has expired
        if (
            current_time - health.retry_budget_window_start
            > self.retry_config.retry_budget_window
        ):
            # Reset window
            health.retry_budget_window_start = current_time
            health.retries_in_current_window = 0
            return False

        # Check if budget is exhausted in current window
        if health.retries_in_current_window >= self.retry_config.max_retries_per_window:
            # Set global cooldown
            health.global_retry_cooldown_until = (
                current_time + self.retry_config.global_retry_cooldown
            )
            self.logger.warning(
                f"Retry budget exhausted for provider {provider_name}. Cooldown until {health.global_retry_cooldown_until}"
            )
            return True

        return False

    def _consume_retry_budget(self, provider_name: str, error_type: ErrorType) -> None:
        """
        Consume retry budget for a provider.

        Args:
            provider_name: Name of the provider
            error_type: Type of error that triggered the retry
        """
        health = self._get_provider_health(provider_name)
        current_time = time.time()

        # Increment retry count in current window
        health.retries_in_current_window += 1

        # Set error-specific cooldown if configured
        if error_type in self.retry_config.per_error_cooldown:
            cooldown_duration = self.retry_config.per_error_cooldown[error_type]
            if cooldown_duration > 0:
                health.error_specific_cooldowns[error_type] = (
                    current_time + cooldown_duration
                )

        # Track global retry history
        self._global_retry_history.append(current_time)

        # Clean up old entries from global history
        cutoff_time = current_time - self.retry_config.retry_budget_window
        self._global_retry_history = [
            t for t in self._global_retry_history if t > cutoff_time
        ]

    def _should_retry_error(
        self, error_type: ErrorType, attempt: int, provider_name: str
    ) -> bool:
        """
        Determine if an error should be retried based on comprehensive heuristics.

        Args:
            error_type: Type of error
            attempt: Current attempt number
            provider_name: Name of the provider

        Returns:
            True if error should be retried
        """
        # Check if error type is retryable
        if error_type not in self.retry_config.retry_on_errors:
            return False

        # Check retry budget
        if self._is_retry_budget_exhausted(provider_name):
            return False

        # Check maximum retries
        if attempt >= self.retry_config.max_retries:
            return False

        # Check circuit breaker
        if self._is_circuit_breaker_open(provider_name):
            return False

        # Special handling for authentication errors - never retry
        if error_type == ErrorType.AUTHENTICATION:
            return False

        # Special handling for quota exceeded - longer delays
        if error_type == ErrorType.QUOTA_EXCEEDED and attempt > 1:
            return False

        # Check global retry rate
        current_time = time.time()
        recent_retries = len(
            [
                t
                for t in self._global_retry_history
                if current_time - t < self.retry_config.retry_budget_window
            ]
        )

        if (
            recent_retries > self.retry_config.max_retries_per_window * 2
        ):  # Global limit
            self.logger.warning(
                f"Global retry rate exceeded: {recent_retries} retries in last {self.retry_config.retry_budget_window}s"
            )
            return False

        return True

    def _retry_with_backoff(
        self,
        operation: Callable[[], LLMResponse],
        provider_name: str,
        max_retries: Optional[int] = None,
    ) -> LLMResponse:
        """
        Execute an operation with retry and exponential backoff.

        Args:
            operation: Function to execute
            provider_name: Name of the provider
            max_retries: Maximum number of retries (overrides config)

        Returns:
            LLM response

        Raises:
            LLMError: If all retries are exhausted
        """
        max_attempts = (max_retries or self.retry_config.max_retries) + 1

        for attempt in range(max_attempts):
            start_time = time.time()

            try:
                response = operation()
                latency = time.time() - start_time
                self._update_provider_health(
                    provider_name, success=True, latency=latency
                )
                return response

            except Exception as e:
                latency = time.time() - start_time
                error_type, is_retryable = self._classify_error(e, provider_name)

                # Update health metrics
                self._update_provider_health(
                    provider_name, success=False, latency=latency, error_type=error_type
                )

                # Check if we should retry using comprehensive heuristics
                should_retry = self._should_retry_error(
                    error_type, attempt, provider_name
                )

                # Don't retry on the last attempt or if error is not retryable
                if attempt == max_attempts - 1 or not should_retry:
                    if isinstance(e, LLMError):
                        raise e
                    else:
                        raise LLMError(
                            str(e),
                            error_type=error_type.value,
                            retryable=is_retryable,
                            provider=provider_name,
                        )

                # Consume retry budget before attempting retry
                self._consume_retry_budget(provider_name, error_type)

                # Calculate delay before next retry
                delay = self._calculate_retry_delay(attempt, error_type)

                self.logger.warning(
                    f"Attempt {attempt + 1}/{max_attempts} failed for provider {provider_name}: {e}. Retrying in {delay:.2f}s"
                )
                time.sleep(delay)

        # This should never be reached due to the loop logic, but just in case
        raise LLMError(
            f"All retry attempts exhausted for provider {provider_name}",
            retryable=False,
            provider=provider_name,
        )

    def _get_provider(
        self, provider_name: str, use_openrouter_proxy: bool = False
    ) -> Optional[LLMProviderInterface]:
        """Get or create a provider instance."""
        cache_key = (
            f"{provider_name}{'_via_openrouter' if use_openrouter_proxy else ''}"
        )

        if cache_key in self._providers:
            return self._providers[cache_key]

        # Check for OpenRouter unified mode
        if use_openrouter_proxy:
            openrouter_key = self.api_key_manager.get_key("openrouter")
            if not openrouter_key:
                return None

            try:
                # Create OpenRouter provider with unified endpoint
                provider = ProviderFactory.create_provider("openrouter", openrouter_key)

                # Get model mapping for the target provider
                unified_config = self.api_key_manager.get_openrouter_unified_config()
                model_mapping = unified_config.get("fallback_model_mapping", {})

                # Override default model for the target provider
                if provider_name in model_mapping:
                    # Note: This sets the default model through a custom attribute
                    setattr(provider, "_default_model", model_mapping[provider_name])

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

    def complete(
        self,
        request: LLMRequest,
        preferred_provider: Optional[str] = None,
        enable_retry: bool = True,
    ) -> LLMResponse:
        """
        Complete a request using the best available provider with intelligent retry and failover.

        Args:
            request: LLM request
            preferred_provider: Preferred provider name
            enable_retry: Whether to enable retry logic

        Returns:
            LLM response

        Raises:
            LLMError: If no providers are available or all fail
        """
        # Get configuration
        active_provider = (
            preferred_provider or self.api_key_manager.get_active_provider()
        )
        fallback_enabled = self.api_key_manager.is_fallback_enabled()
        fallback_order = self.api_key_manager.get_fallback_order()
        unified_config = self.api_key_manager.get_openrouter_unified_config()

        # Determine provider order based on health metrics
        provider_order = self._get_optimal_provider_order(
            active_provider, fallback_enabled, fallback_order
        )

        last_error = None
        attempted_providers = []

        for provider_name in provider_order:
            # Skip if circuit breaker is open
            if self._is_circuit_breaker_open(provider_name):
                self.logger.info(
                    f"Skipping provider {provider_name} due to open circuit breaker"
                )
                continue

            attempted_providers.append(provider_name)

            # Check if OpenRouter unified mode should be used
            use_openrouter_proxy = False
            if unified_config.get("enabled", False):
                if provider_name == "anthropic" and unified_config.get(
                    "proxy_to_anthropic", False
                ):
                    use_openrouter_proxy = True
                elif provider_name == "openai" and unified_config.get(
                    "proxy_to_openai", False
                ):
                    use_openrouter_proxy = True

            # Try direct provider first
            provider = self._get_provider(provider_name, use_openrouter_proxy=False)
            if provider:
                try:
                    if enable_retry:
                        # Use retry logic
                        def operation():
                            return provider.complete(request)

                        return self._retry_with_backoff(operation, provider_name)
                    else:
                        # Single attempt
                        start_time = time.time()
                        response = provider.complete(request)
                        latency = time.time() - start_time
                        self._update_provider_health(
                            provider_name, success=True, latency=latency
                        )
                        return response

                except LLMError as e:
                    last_error = e

                    # If direct provider fails and unified mode is available, try OpenRouter proxy
                    if use_openrouter_proxy:
                        try:
                            proxy_provider = self._get_provider(
                                provider_name, use_openrouter_proxy=True
                            )
                            if proxy_provider:
                                if enable_retry:

                                    def operation():
                                        assert proxy_provider is not None
                                        return proxy_provider.complete(request)

                                    return self._retry_with_backoff(
                                        operation, f"{provider_name}_proxy"
                                    )
                                else:
                                    start_time = time.time()
                                    assert proxy_provider is not None
                                    response = proxy_provider.complete(request)
                                    latency = time.time() - start_time
                                    self._update_provider_health(
                                        f"{provider_name}_proxy",
                                        success=True,
                                        latency=latency,
                                    )
                                    return response
                        except LLMError:
                            pass  # Continue to next provider

            # If direct provider is not available but OpenRouter proxy is enabled, try proxy
            elif use_openrouter_proxy:
                try:
                    proxy_provider = self._get_provider(
                        provider_name, use_openrouter_proxy=True
                    )
                    if proxy_provider:
                        if enable_retry:

                            def operation():
                                return proxy_provider.complete(request)

                            return self._retry_with_backoff(
                                operation, f"{provider_name}_proxy"
                            )
                        else:
                            start_time = time.time()
                            response = proxy_provider.complete(request)
                            latency = time.time() - start_time
                            self._update_provider_health(
                                f"{provider_name}_proxy", success=True, latency=latency
                            )
                            return response
                except LLMError as e:
                    last_error = e

        # No providers worked
        error_msg = f"All providers failed. Attempted: {', '.join(attempted_providers)}"
        if last_error:
            error_msg += f". Last error: {last_error}"

        raise LLMError(error_msg, error_type="all_providers_failed", retryable=False)

    def _get_optimal_provider_order(
        self,
        active_provider: Optional[str],
        fallback_enabled: bool,
        fallback_order: List[str],
    ) -> List[str]:
        """
        Get optimal provider order based on health metrics and configuration.

        Args:
            active_provider: Active provider name
            fallback_enabled: Whether fallback is enabled
            fallback_order: Configured fallback order

        Returns:
            Optimal provider order
        """
        # Start with configured order
        if active_provider:
            provider_order = [active_provider]
            if fallback_enabled:
                for provider in fallback_order:
                    if provider != active_provider:
                        provider_order.append(provider)
        else:
            provider_order = (
                fallback_order if fallback_enabled else self._provider_order
            )

        # Filter out providers with open circuit breakers and sort by health
        available_providers = []
        unavailable_providers = []

        for provider in provider_order:
            if self._is_circuit_breaker_open(provider):
                unavailable_providers.append(provider)
            else:
                available_providers.append(provider)

        # Sort available providers by success rate (descending)
        available_providers.sort(
            key=lambda p: self._get_provider_health(p).success_rate, reverse=True
        )

        # Add unavailable providers at the end (in case circuit breaker times out)
        return available_providers + unavailable_providers

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
            "anthropic": {
                "cost": 2,  # Medium cost
                "latency": 2,  # Medium latency
                "reliability": 3,  # High reliability
            },
            "openai": {
                "cost": 2,  # Medium cost
                "latency": 3,  # Low latency
                "reliability": 3,  # High reliability
            },
            "openrouter": {
                "cost": 1,  # Lower cost (aggregator)
                "latency": 1,  # Higher latency
                "reliability": 2,  # Medium reliability
            },
        }

        # Weight preferences (higher is better)
        preference_weights = {"low": 1, "balanced": 2, "high": 3}

        # Calculate scores for each available provider
        scored_providers = []
        for provider in available_providers:
            if provider in provider_scores:
                score = 0
                chars = provider_scores[provider]

                # Cost preference (inverted - lower cost is better when preference is low)
                cost_pref = policies.get("cost_preference", "balanced")
                if cost_pref == "low":
                    score += (4 - chars["cost"]) * 2  # Prefer lower cost
                elif cost_pref == "high":
                    score += chars["cost"] * 2  # Don't mind higher cost
                else:
                    score += 2  # Balanced

                # Latency preference (lower latency is better when preference is low)
                latency_pref = policies.get("latency_preference", "balanced")
                if latency_pref == "low":
                    score += chars["latency"] * 2  # Prefer lower latency
                else:
                    score += preference_weights[latency_pref]

                # Reliability preference
                reliability_pref = policies.get("reliability_preference", "high")
                score += chars["reliability"] * preference_weights[reliability_pref]

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
            print("✓ Auto-configured fallback order based on current policies")
        else:
            print("⚠️  No providers available for auto-configuration")

    def get_provider_health_metrics(
        self, provider_name: Optional[str] = None
    ) -> Dict[str, ProviderHealthMetrics]:
        """
        Get health metrics for providers.

        Args:
            provider_name: Specific provider name, or None for all providers

        Returns:
            Dictionary of provider health metrics
        """
        if provider_name:
            return {provider_name: self._get_provider_health(provider_name)}
        else:
            return dict(self._provider_health)

    def reset_provider_health(self, provider_name: Optional[str] = None) -> None:
        """
        Reset health metrics for providers.

        Args:
            provider_name: Specific provider name, or None for all providers
        """
        if provider_name:
            if provider_name in self._provider_health:
                del self._provider_health[provider_name]
                self.logger.info(f"Reset health metrics for provider {provider_name}")
        else:
            self._provider_health.clear()
            self.logger.info("Reset health metrics for all providers")

    def force_close_circuit_breaker(self, provider_name: str) -> bool:
        """
        Force close a circuit breaker for a provider.

        Args:
            provider_name: Provider name

        Returns:
            True if circuit breaker was closed
        """
        if provider_name in self._provider_health:
            health = self._provider_health[provider_name]
            if health.circuit_breaker_open:
                health.circuit_breaker_open = False
                health.circuit_breaker_open_until = None
                health.consecutive_failures = 0
                self.logger.info(
                    f"Manually closed circuit breaker for provider {provider_name}"
                )
                return True
        return False

    def get_provider_status_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all provider statuses.

        Returns:
            Provider status summary
        """
        summary: Dict[str, Any] = {
            "total_providers": len(self._provider_health),
            "healthy_providers": 0,
            "unhealthy_providers": 0,
            "circuit_breaker_open": 0,
            "providers": {},
        }

        for provider_name, health in self._provider_health.items():
            is_healthy = health.success_rate >= 0.8 and not health.circuit_breaker_open

            if is_healthy:
                summary["healthy_providers"] += 1
            else:
                summary["unhealthy_providers"] += 1

            if health.circuit_breaker_open:
                summary["circuit_breaker_open"] += 1

            summary["providers"][provider_name] = {
                "healthy": is_healthy,
                "success_rate": health.success_rate,
                "total_requests": health.total_requests,
                "average_latency": health.average_latency,
                "consecutive_failures": health.consecutive_failures,
                "circuit_breaker_open": health.circuit_breaker_open,
                "last_success": health.last_success_time,
                "last_failure": health.last_failure_time,
            }

        return summary
