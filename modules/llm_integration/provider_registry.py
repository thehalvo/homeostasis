#!/usr/bin/env python3
"""
Provider registry for LLM integration.

Provides a modular system for registering and discovering LLM providers.
"""

import importlib
import inspect
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Type

from .provider_abstraction import LLMProviderInterface


@dataclass
class ProviderCapability:
    """Represents a capability that a provider supports."""

    name: str
    description: str
    required: bool = False


@dataclass
class ProviderMetadata:
    """Metadata about a provider."""

    name: str
    display_name: str
    description: str
    version: str
    author: str
    homepage: Optional[str] = None
    documentation: Optional[str] = None
    capabilities: List[ProviderCapability] = field(default_factory=list)
    supported_models: List[str] = field(default_factory=list)
    model_families: Dict[str, List[str]] = field(default_factory=dict)
    pricing_tier: str = "unknown"  # free, low, medium, high
    latency_class: str = "unknown"  # low, medium, high
    reliability_score: float = 0.0  # 0.0 to 1.0
    default_model: Optional[str] = None
    authentication_methods: List[str] = field(default_factory=list)
    rate_limits: Dict[str, Any] = field(default_factory=dict)
    context_limits: Dict[str, int] = field(default_factory=dict)
    features: Set[str] = field(default_factory=set)
    tags: Set[str] = field(default_factory=set)


class ProviderPlugin(ABC):
    """Abstract base class for provider plugins."""

    @abstractmethod
    def get_metadata(self) -> ProviderMetadata:
        """Get provider metadata."""
        pass

    @abstractmethod
    def create_provider(self, api_key: str, **kwargs) -> LLMProviderInterface:
        """Create provider instance."""
        pass

    @abstractmethod
    def validate_configuration(self, config: Dict[str, Any]) -> bool:
        """Validate provider configuration."""
        pass

    def get_default_configuration(self) -> Dict[str, Any]:
        """Get default configuration for this provider."""
        return {}

    def get_configuration_schema(self) -> Dict[str, Any]:
        """Get configuration schema for validation."""
        return {}

    def supports_feature(self, feature: str) -> bool:
        """Check if provider supports a specific feature."""
        return feature in self.get_metadata().features


class ProviderRegistry:
    """Registry for LLM provider plugins."""

    def __init__(self):
        """Initialize provider registry."""
        self._providers: Dict[str, ProviderPlugin] = {}
        self._provider_classes: Dict[str, Type[ProviderPlugin]] = {}
        self._capabilities: Dict[str, List[str]] = {}
        self.logger = logging.getLogger(__name__)

        # Load built-in providers
        self._load_builtin_providers()

    def register_provider(self, provider_class: Type[ProviderPlugin]) -> None:
        """
        Register a provider plugin class.

        Args:
            provider_class: Provider plugin class
        """
        try:
            # Create temporary instance to get metadata
            temp_instance = provider_class()
            metadata = temp_instance.get_metadata()

            # Validate metadata
            if not metadata.name:
                raise ValueError("Provider name cannot be empty")

            if metadata.name in self._provider_classes:
                self.logger.warning(
                    f"Provider {metadata.name} is already registered, overwriting"
                )

            # Register the class
            self._provider_classes[metadata.name] = provider_class

            # Update capabilities index
            for capability in metadata.capabilities:
                if capability.name not in self._capabilities:
                    self._capabilities[capability.name] = []
                self._capabilities[capability.name].append(metadata.name)

            self.logger.info(
                f"Registered provider: {metadata.name} v{metadata.version}"
            )

        except Exception as e:
            self.logger.error(
                f"Failed to register provider {provider_class.__name__}: {e}"
            )
            raise

    def get_provider(self, name: str) -> Optional[ProviderPlugin]:
        """
        Get a provider instance by name.

        Args:
            name: Provider name

        Returns:
            Provider instance or None if not found
        """
        if name in self._providers:
            return self._providers[name]

        if name in self._provider_classes:
            try:
                self._providers[name] = self._provider_classes[name]()
                return self._providers[name]
            except Exception as e:
                self.logger.error(f"Failed to instantiate provider {name}: {e}")
                return None

        return None

    def list_providers(
        self,
        with_capability: Optional[str] = None,
        feature_filter: Optional[str] = None,
        tag_filter: Optional[str] = None,
    ) -> List[str]:
        """
        List registered provider names.

        Args:
            with_capability: Filter by capability
            feature_filter: Filter by feature
            tag_filter: Filter by tag

        Returns:
            List of provider names
        """
        providers = list(self._provider_classes.keys())

        if with_capability and with_capability in self._capabilities:
            providers = [
                p for p in providers if p in self._capabilities[with_capability]
            ]

        if feature_filter or tag_filter:
            filtered_providers = []
            for provider_name in providers:
                provider = self.get_provider(provider_name)
                if not provider:
                    continue

                metadata = provider.get_metadata()

                if feature_filter and feature_filter not in metadata.features:
                    continue

                if tag_filter and tag_filter not in metadata.tags:
                    continue

                filtered_providers.append(provider_name)

            providers = filtered_providers

        return sorted(providers)

    def get_metadata(self, name: str) -> Optional[ProviderMetadata]:
        """
        Get metadata for a provider.

        Args:
            name: Provider name

        Returns:
            Provider metadata or None if not found
        """
        provider = self.get_provider(name)
        return provider.get_metadata() if provider else None

    def list_capabilities(self) -> List[str]:
        """
        List all available capabilities.

        Returns:
            List of capability names
        """
        return sorted(self._capabilities.keys())

    def find_providers_by_capability(self, capability: str) -> List[str]:
        """
        Find providers that support a specific capability.

        Args:
            capability: Capability name

        Returns:
            List of provider names
        """
        return self._capabilities.get(capability, [])

    def create_provider_instance(
        self, name: str, api_key: str, **kwargs
    ) -> Optional[LLMProviderInterface]:
        """
        Create a provider instance.

        Args:
            name: Provider name
            api_key: API key
            **kwargs: Additional configuration

        Returns:
            Provider instance or None if creation fails
        """
        provider = self.get_provider(name)
        if not provider:
            return None

        try:
            return provider.create_provider(api_key, **kwargs)
        except Exception as e:
            self.logger.error(f"Failed to create provider instance for {name}: {e}")
            return None

    def validate_provider_config(self, name: str, config: Dict[str, Any]) -> bool:
        """
        Validate configuration for a provider.

        Args:
            name: Provider name
            config: Configuration to validate

        Returns:
            True if configuration is valid
        """
        provider = self.get_provider(name)
        if not provider:
            return False

        try:
            return provider.validate_configuration(config)
        except Exception as e:
            self.logger.error(f"Configuration validation failed for {name}: {e}")
            return False

    def get_provider_schema(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get configuration schema for a provider.

        Args:
            name: Provider name

        Returns:
            Configuration schema or None
        """
        provider = self.get_provider(name)
        if not provider:
            return None

        try:
            return provider.get_configuration_schema()
        except Exception:
            return None

    def load_plugin_from_path(self, path: Path) -> None:
        """
        Load a provider plugin from a file path.

        Args:
            path: Path to the plugin file
        """
        if not path.exists() or not path.is_file():
            raise ValueError(f"Plugin file not found: {path}")

        if path.suffix != ".py":
            raise ValueError(f"Plugin file must be a Python file: {path}")

        try:
            # Import the module
            spec = importlib.util.spec_from_file_location(path.stem, path)
            if not spec or not spec.loader:
                raise ImportError(f"Could not load spec for {path}")

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Find provider plugin classes
            for name, obj in inspect.getmembers(module):
                if (
                    inspect.isclass(obj)
                    and issubclass(obj, ProviderPlugin)
                    and obj != ProviderPlugin
                ):
                    self.register_provider(obj)
                    self.logger.info(f"Loaded provider plugin {name} from {path}")

        except Exception as e:
            self.logger.error(f"Failed to load plugin from {path}: {e}")
            raise

    def load_plugins_from_directory(self, directory: Path) -> None:
        """
        Load all provider plugins from a directory.

        Args:
            directory: Directory containing plugin files
        """
        if not directory.exists() or not directory.is_dir():
            self.logger.warning(f"Plugin directory not found: {directory}")
            return

        for plugin_file in directory.glob("*.py"):
            if plugin_file.name.startswith("__"):
                continue

            try:
                self.load_plugin_from_path(plugin_file)
            except Exception as e:
                self.logger.error(f"Failed to load plugin {plugin_file}: {e}")

    def _load_builtin_providers(self) -> None:
        """Load built-in provider plugins."""
        try:
            from .builtin_providers import (AnthropicPlugin, OpenAIPlugin,
                                            OpenRouterPlugin)

            self.register_provider(OpenAIPlugin)
            self.register_provider(AnthropicPlugin)
            self.register_provider(OpenRouterPlugin)

        except ImportError as e:
            self.logger.warning(f"Could not load built-in providers: {e}")

    def get_providers_by_tier(self, tier: str) -> List[str]:
        """
        Get providers by pricing tier.

        Args:
            tier: Pricing tier (free, low, medium, high)

        Returns:
            List of provider names
        """
        providers = []
        for name in self._provider_classes:
            metadata = self.get_metadata(name)
            if metadata and metadata.pricing_tier == tier:
                providers.append(name)
        return sorted(providers)

    def get_providers_by_latency(self, latency_class: str) -> List[str]:
        """
        Get providers by latency class.

        Args:
            latency_class: Latency class (low, medium, high)

        Returns:
            List of provider names
        """
        providers = []
        for name in self._provider_classes:
            metadata = self.get_metadata(name)
            if metadata and metadata.latency_class == latency_class:
                providers.append(name)
        return sorted(providers)

    def get_most_reliable_providers(self, min_score: float = 0.8) -> List[str]:
        """
        Get providers with reliability score above threshold.

        Args:
            min_score: Minimum reliability score

        Returns:
            List of provider names sorted by reliability (desc)
        """
        providers_with_scores = []
        for name in self._provider_classes:
            metadata = self.get_metadata(name)
            if metadata and metadata.reliability_score >= min_score:
                providers_with_scores.append((name, metadata.reliability_score))

        # Sort by reliability score (descending)
        providers_with_scores.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in providers_with_scores]

    def export_registry_info(self) -> Dict[str, Any]:
        """
        Export registry information for debugging/inspection.

        Returns:
            Registry information
        """
        info = {
            "total_providers": len(self._provider_classes),
            "capabilities": dict(self._capabilities),
            "providers": {},
        }

        for name in self._provider_classes:
            metadata = self.get_metadata(name)
            if metadata:
                info["providers"][name] = {
                    "display_name": metadata.display_name,
                    "version": metadata.version,
                    "capabilities": [c.name for c in metadata.capabilities],
                    "features": list(metadata.features),
                    "tags": list(metadata.tags),
                    "pricing_tier": metadata.pricing_tier,
                    "latency_class": metadata.latency_class,
                    "reliability_score": metadata.reliability_score,
                }

        return info


# Global registry instance
_registry = None


def get_provider_registry() -> ProviderRegistry:
    """Get the global provider registry instance."""
    global _registry
    if _registry is None:
        _registry = ProviderRegistry()
    return _registry
