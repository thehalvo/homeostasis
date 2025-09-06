"""
Security configuration utilities for Homeostasis.

Provides functionality for loading, validating, and managing security configurations.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from cryptography.fernet import Fernet

logger = logging.getLogger(__name__)


class SecurityConfigError(Exception):
    """Exception raised for security configuration errors."""

    pass


class SecurityConfig:
    """Manages security configuration for Homeostasis."""

    def __init__(
        self, config_path: Optional[str] = None, env_prefix: str = "HOMEOSTASIS"
    ):
        """Initialize the security configuration.

        Args:
            config_path: Path to security configuration file
            env_prefix: Prefix for environment variables
        """
        self.env_prefix = env_prefix
        self.config = {}

        # Load configuration
        if config_path and os.path.exists(config_path):
            self._load_from_file(config_path)

        # Override with environment variables
        self._load_from_env()

        # Generate missing keys if needed
        self._ensure_keys()

        # Validate configuration
        self._validate_config()

    def _load_from_file(self, config_path: str) -> None:
        """Load configuration from a file.

        Args:
            config_path: Path to configuration file

        Raises:
            SecurityConfigError: If configuration file is invalid
        """
        try:
            with open(config_path, "r") as f:
                loaded_config = json.load(f)

            if not isinstance(loaded_config, dict):
                raise SecurityConfigError("Invalid configuration format")

            self.config = loaded_config
            logger.info(f"Loaded security configuration from {config_path}")
        except json.JSONDecodeError as e:
            raise SecurityConfigError(f"Error parsing configuration file: {str(e)}")
        except Exception as e:
            raise SecurityConfigError(f"Error loading configuration file: {str(e)}")

    def _load_from_env(self) -> None:
        """Load configuration from environment variables."""
        env_vars = {
            f"{self.env_prefix}_JWT_SECRET": "jwt_secret",
            f"{self.env_prefix}_ENCRYPTION_KEY": "encryption_key",
            f"{self.env_prefix}_SESSION_SECRET": "session_secret",
            f"{self.env_prefix}_TOKEN_EXPIRY": "token_expiry",
            f"{self.env_prefix}_REFRESH_TOKEN_EXPIRY": "refresh_token_expiry",
            f"{self.env_prefix}_RATE_LIMIT_GLOBAL": "rate_limits.global",
            f"{self.env_prefix}_RATE_LIMIT_USER": "rate_limits.user",
            f"{self.env_prefix}_RATE_LIMIT_IP": "rate_limits.ip",
        }

        # Load environment variables
        for env_var, config_key in env_vars.items():
            value = os.environ.get(env_var)
            if value is not None:
                # Convert number values
                if config_key in ["token_expiry", "refresh_token_expiry"]:
                    try:
                        value = int(value)
                    except ValueError:
                        logger.warning(f"Invalid value for {env_var}: {value}")
                        continue

                # Handle nested keys
                if "." in config_key:
                    parts = config_key.split(".")
                    current = self.config
                    for part in parts[:-1]:
                        if part not in current:
                            current[part] = {}
                        current = current[part]
                    current[parts[-1]] = value
                else:
                    self.config[config_key] = value

    def _ensure_keys(self) -> None:
        """Ensure required keys exist, generate them if necessary."""
        # Generate JWT secret if not present
        if "jwt_secret" not in self.config:
            self.config["jwt_secret"] = Fernet.generate_key().decode("utf-8")
            logger.warning(
                "Generated random JWT secret. "
                "This is insecure for production environments."
            )

        # Generate encryption key if not present
        if "encryption_key" not in self.config:
            self.config["encryption_key"] = Fernet.generate_key().decode("utf-8")
            logger.warning(
                "Generated random encryption key. "
                "This is insecure for production environments."
            )

        # Set default token expiry if not present
        if "token_expiry" not in self.config:
            self.config["token_expiry"] = 3600  # 1 hour

        # Set default refresh token expiry if not present
        if "refresh_token_expiry" not in self.config:
            self.config["refresh_token_expiry"] = 604800  # 7 days

        # Set default rate limits if not present
        if "rate_limits" not in self.config:
            self.config["rate_limits"] = {
                "global": (100, 60),  # 100 requests per 60 seconds
                "user": (20, 60),  # 20 requests per 60 seconds per user
                "ip": (50, 60),  # 50 requests per 60 seconds per IP
            }

    def _validate_config(self) -> None:
        """Validate the configuration.

        Raises:
            SecurityConfigError: If configuration is invalid
        """
        # Check required keys
        required_keys = ["jwt_secret", "encryption_key"]
        for key in required_keys:
            if key not in self.config:
                raise SecurityConfigError(f"Missing required configuration key: {key}")

        # Validate token expiry
        if (
            not isinstance(self.config.get("token_expiry", 0), int)
            or self.config.get("token_expiry", 0) <= 0
        ):
            raise SecurityConfigError(
                "Invalid token expiry, must be a positive integer"
            )

        # Validate refresh token expiry
        if (
            not isinstance(self.config.get("refresh_token_expiry", 0), int)
            or self.config.get("refresh_token_expiry", 0) <= 0
        ):
            raise SecurityConfigError(
                "Invalid refresh token expiry, must be a positive integer"
            )

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value.

        Args:
            key: Configuration key
            default: Default value if key is not found

        Returns:
            Any: Configuration value
        """
        # Handle nested keys
        if "." in key:
            parts = key.split(".")
            current = self.config
            for part in parts:
                if not isinstance(current, dict) or part not in current:
                    return default
                current = current[part]
            return current

        return self.config.get(key, default)

    def get_all(self) -> Dict:
        """Get the entire configuration.

        Returns:
            Dict: Configuration dictionary
        """
        return dict(self.config)

    def set(self, key: str, value: Any) -> None:
        """Set a configuration value.

        Args:
            key: Configuration key
            value: Configuration value
        """
        # Handle nested keys
        if "." in key:
            parts = key.split(".")
            current = self.config
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
        else:
            self.config[key] = value

    def save(self, config_path: str) -> None:
        """Save the configuration to a file.

        Args:
            config_path: Path to save the configuration to

        Raises:
            SecurityConfigError: If saving fails
        """
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, "w") as f:
                json.dump(self.config, f, indent=2)

            logger.info(f"Saved security configuration to {config_path}")
        except Exception as e:
            raise SecurityConfigError(
                f"Error saving configuration to {config_path}: {str(e)}"
            )


# Singleton instance for app-wide use
_security_config = None


def get_security_config(config_path: Optional[str] = None) -> SecurityConfig:
    """Get or create the singleton SecurityConfig instance.

    Args:
        config_path: Optional path to configuration file

    Returns:
        SecurityConfig: The security configuration instance
    """
    global _security_config
    if _security_config is None:
        _security_config = SecurityConfig(config_path)
    return _security_config


def get_config(key: str, default: Any = None) -> Any:
    """Get a configuration value.

    Args:
        key: Configuration key
        default: Default value if key is not found

    Returns:
        Any: Configuration value
    """
    return get_security_config().get(key, default)


def get_all_config() -> Dict:
    """Get the entire configuration.

    Returns:
        Dict: Configuration dictionary
    """
    return get_security_config().get_all()


def set_config(key: str, value: Any) -> None:
    """Set a configuration value.

    Args:
        key: Configuration key
        value: Configuration value
    """
    get_security_config().set(key, value)


def save_config(config_path: str) -> None:
    """Save the configuration to a file.

    Args:
        config_path: Path to save the configuration to

    Raises:
        SecurityConfigError: If saving fails
    """
    get_security_config().save(config_path)
