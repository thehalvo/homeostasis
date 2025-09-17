#!/usr/bin/env python3
"""
External Secrets Manager Integration for LLM API Keys

Provides integrations with various external secrets management systems
to securely store and retrieve API keys.
"""

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class SecretsManagerError(Exception):
    """Raised when secrets manager operations fail."""

    pass


class SecretsManagerBase(ABC):
    """Base class for secrets manager integrations."""

    @abstractmethod
    def get_secret(self, secret_name: str) -> Optional[str]:
        """Get a secret value by name."""
        pass

    @abstractmethod
    def set_secret(self, secret_name: str, secret_value: str) -> bool:
        """Set a secret value."""
        pass

    @abstractmethod
    def delete_secret(self, secret_name: str) -> bool:
        """Delete a secret."""
        pass

    @abstractmethod
    def list_secrets(self) -> List[str]:
        """List available secrets."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the secrets manager is available and configured."""
        pass


class AWSSecretsManager(SecretsManagerBase):
    """AWS Secrets Manager integration."""

    def __init__(self, region_name: Optional[str] = None):
        """Initialize AWS Secrets Manager client."""
        self._client = None
        self._region_name = region_name or os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        self._secret_prefix = "homeostasis/llm/"

    def _get_client(self):
        """Get or create AWS Secrets Manager client."""
        if self._client is None:
            try:
                import boto3
                from botocore.exceptions import BotoCoreError, ClientError

                self._client = boto3.client(
                    "secretsmanager", region_name=self._region_name
                )
                self._boto_exceptions = (BotoCoreError, ClientError)
            except ImportError:
                raise SecretsManagerError(
                    "AWS SDK (boto3) not installed. Install with: pip install boto3"
                )
        return self._client

    def get_secret(self, secret_name: str) -> Optional[str]:
        """Get a secret from AWS Secrets Manager."""
        try:
            client = self._get_client()
            full_name = f"{self._secret_prefix}{secret_name}"

            response = client.get_secret_value(SecretId=full_name)
            return str(response["SecretString"])

        except self._boto_exceptions as e:
            error_code = (
                getattr(e, "response", {}).get("Error", {}).get("Code", "Unknown")
            )
            if error_code == "ResourceNotFoundException":
                return None
            raise SecretsManagerError(f"AWS Secrets Manager error: {e}")
        except Exception as e:
            raise SecretsManagerError(f"Failed to get secret from AWS: {e}")

    def set_secret(self, secret_name: str, secret_value: str) -> bool:
        """Set a secret in AWS Secrets Manager."""
        try:
            client = self._get_client()
            full_name = f"{self._secret_prefix}{secret_name}"

            try:
                # Try to update existing secret
                client.update_secret(SecretId=full_name, SecretString=secret_value)
            except self._boto_exceptions as e:
                error_code = (
                    getattr(e, "response", {}).get("Error", {}).get("Code", "Unknown")
                )
                if error_code == "ResourceNotFoundException":
                    # Create new secret
                    client.create_secret(Name=full_name, SecretString=secret_value)
                else:
                    raise

            return True

        except self._boto_exceptions as e:
            raise SecretsManagerError(f"AWS Secrets Manager error: {e}")
        except Exception as e:
            raise SecretsManagerError(f"Failed to set secret in AWS: {e}")

    def delete_secret(self, secret_name: str) -> bool:
        """Delete a secret from AWS Secrets Manager."""
        try:
            client = self._get_client()
            full_name = f"{self._secret_prefix}{secret_name}"

            client.delete_secret(SecretId=full_name, ForceDeleteWithoutRecovery=True)
            return True

        except self._boto_exceptions as e:
            error_code = (
                getattr(e, "response", {}).get("Error", {}).get("Code", "Unknown")
            )
            if error_code == "ResourceNotFoundException":
                return False
            raise SecretsManagerError(f"AWS Secrets Manager error: {e}")
        except Exception as e:
            raise SecretsManagerError(f"Failed to delete secret from AWS: {e}")

    def list_secrets(self) -> List[str]:
        """List secrets in AWS Secrets Manager."""
        try:
            client = self._get_client()

            response = client.list_secrets()
            secrets = []

            for secret in response.get("SecretList", []):
                name = secret.get("Name", "")
                if name.startswith(self._secret_prefix):
                    secrets.append(name[len(self._secret_prefix) :])

            return secrets

        except self._boto_exceptions as e:
            raise SecretsManagerError(f"AWS Secrets Manager error: {e}")
        except Exception as e:
            raise SecretsManagerError(f"Failed to list secrets from AWS: {e}")

    def is_available(self) -> bool:
        """Check if AWS Secrets Manager is available."""
        try:
            client = self._get_client()
            # Test with a simple list operation
            client.list_secrets(MaxResults=1)
            return True
        except Exception:
            return False


class AzureKeyVault(SecretsManagerBase):
    """Azure Key Vault integration."""

    def __init__(self, vault_url: Optional[str] = None):
        """Initialize Azure Key Vault client."""
        self._client = None
        self._vault_url = vault_url or os.getenv("AZURE_KEY_VAULT_URL")
        self._secret_prefix = "homeostasis-llm-"

        if not self._vault_url:
            raise SecretsManagerError(
                "Azure Key Vault URL not provided. Set AZURE_KEY_VAULT_URL environment variable."
            )

    def _get_client(self):
        """Get or create Azure Key Vault client."""
        if self._client is None:
            try:
                from azure.core.exceptions import AzureError
                from azure.identity import DefaultAzureCredential
                from azure.keyvault.secrets import SecretClient

                credential = DefaultAzureCredential()
                self._client = SecretClient(
                    vault_url=self._vault_url, credential=credential
                )
                self._azure_exceptions = (AzureError,)

            except ImportError:
                raise SecretsManagerError(
                    "Azure SDK not installed. Install with: pip install azure-keyvault-secrets azure-identity"
                )
        return self._client

    def get_secret(self, secret_name: str) -> Optional[str]:
        """Get a secret from Azure Key Vault."""
        try:
            client = self._get_client()
            full_name = f"{self._secret_prefix}{secret_name.replace('_', '-')}"

            secret = client.get_secret(full_name)
            return str(secret.value) if secret.value else None

        except self._azure_exceptions as e:
            if "SecretNotFound" in str(e):
                return None
            raise SecretsManagerError(f"Azure Key Vault error: {e}")
        except Exception as e:
            raise SecretsManagerError(f"Failed to get secret from Azure: {e}")

    def set_secret(self, secret_name: str, secret_value: str) -> bool:
        """Set a secret in Azure Key Vault."""
        try:
            client = self._get_client()
            full_name = f"{self._secret_prefix}{secret_name.replace('_', '-')}"

            client.set_secret(full_name, secret_value)
            return True

        except self._azure_exceptions as e:
            raise SecretsManagerError(f"Azure Key Vault error: {e}")
        except Exception as e:
            raise SecretsManagerError(f"Failed to set secret in Azure: {e}")

    def delete_secret(self, secret_name: str) -> bool:
        """Delete a secret from Azure Key Vault."""
        try:
            client = self._get_client()
            full_name = f"{self._secret_prefix}{secret_name.replace('_', '-')}"

            client.begin_delete_secret(full_name).wait()
            return True

        except self._azure_exceptions as e:
            if "SecretNotFound" in str(e):
                return False
            raise SecretsManagerError(f"Azure Key Vault error: {e}")
        except Exception as e:
            raise SecretsManagerError(f"Failed to delete secret from Azure: {e}")

    def list_secrets(self) -> List[str]:
        """List secrets in Azure Key Vault."""
        try:
            client = self._get_client()

            secrets = []
            for secret_properties in client.list_properties_of_secrets():
                name = secret_properties.name
                if name.startswith(self._secret_prefix):
                    original_name = name[len(self._secret_prefix) :].replace("-", "_")
                    secrets.append(original_name)

            return secrets

        except self._azure_exceptions as e:
            raise SecretsManagerError(f"Azure Key Vault error: {e}")
        except Exception as e:
            raise SecretsManagerError(f"Failed to list secrets from Azure: {e}")

    def is_available(self) -> bool:
        """Check if Azure Key Vault is available."""
        try:
            client = self._get_client()
            # Test with a simple list operation
            list(client.list_properties_of_secrets(max_page_size=1))
            return True
        except Exception:
            return False


class HashiCorpVault(SecretsManagerBase):
    """HashiCorp Vault integration."""

    def __init__(
        self, vault_url: Optional[str] = None, vault_token: Optional[str] = None
    ):
        """Initialize HashiCorp Vault client."""
        self._client: Optional[Any] = None
        self._vault_url = vault_url or os.getenv("VAULT_ADDR", "http://localhost:8200")
        self._vault_token = vault_token or os.getenv("VAULT_TOKEN")
        self._secret_path = "secret/homeostasis/llm"

        if not self._vault_token:
            raise SecretsManagerError(
                "HashiCorp Vault token not provided. Set VAULT_TOKEN environment variable."
            )

    def _get_client(self) -> Any:
        """Get or create HashiCorp Vault client."""
        if self._client is None:
            try:
                import hvac

                self._client = hvac.Client(url=self._vault_url, token=self._vault_token)

                if self._client is None:
                    raise SecretsManagerError("Failed to create HashiCorp Vault client")

                if not self._client.is_authenticated():
                    raise SecretsManagerError("HashiCorp Vault authentication failed")

            except ImportError:
                raise SecretsManagerError(
                    "HashiCorp Vault client not installed. Install with: pip install hvac"
                )
        return self._client

    def get_secret(self, secret_name: str) -> Optional[str]:
        """Get a secret from HashiCorp Vault."""
        try:
            client = self._get_client()
            path = f"{self._secret_path}/{secret_name}"

            response = client.secrets.kv.v2.read_secret_version(path=path)
            value = response["data"]["data"].get("value")
            return str(value) if value is not None else None

        except Exception as e:
            if "404" in str(e) or "Not Found" in str(e):
                return None
            raise SecretsManagerError(f"HashiCorp Vault error: {e}")

    def set_secret(self, secret_name: str, secret_value: str) -> bool:
        """Set a secret in HashiCorp Vault."""
        try:
            client = self._get_client()
            path = f"{self._secret_path}/{secret_name}"

            client.secrets.kv.v2.create_or_update_secret(
                path=path, secret={"value": secret_value}
            )
            return True

        except Exception as e:
            raise SecretsManagerError(f"Failed to set secret in HashiCorp Vault: {e}")

    def delete_secret(self, secret_name: str) -> bool:
        """Delete a secret from HashiCorp Vault."""
        try:
            client = self._get_client()
            path = f"{self._secret_path}/{secret_name}"

            client.secrets.kv.v2.delete_metadata_and_all_versions(path=path)
            return True

        except Exception as e:
            if "404" in str(e) or "Not Found" in str(e):
                return False
            raise SecretsManagerError(
                f"Failed to delete secret from HashiCorp Vault: {e}"
            )

    def list_secrets(self) -> List[str]:
        """List secrets in HashiCorp Vault."""
        try:
            client = self._get_client()

            response = client.secrets.kv.v2.list_secrets(path=self._secret_path)
            return list(response["data"]["keys"])

        except Exception as e:
            if "404" in str(e) or "Not Found" in str(e):
                return []
            raise SecretsManagerError(
                f"Failed to list secrets from HashiCorp Vault: {e}"
            )

    def is_available(self) -> bool:
        """Check if HashiCorp Vault is available."""
        try:
            client = self._get_client()
            return bool(client.is_authenticated())
        except Exception:
            return False


class SecretsManagerRegistry:
    """Registry for managing multiple secrets managers."""

    def __init__(self):
        self._managers: Dict[str, SecretsManagerBase] = {}
        self._preferred_order = ["aws", "azure", "vault"]

    def register_manager(self, name: str, manager: SecretsManagerBase) -> None:
        """Register a secrets manager."""
        self._managers[name] = manager

    def get_manager(self, name: str) -> Optional[SecretsManagerBase]:
        """Get a specific secrets manager."""
        return self._managers.get(name)

    def get_available_managers(self) -> Dict[str, SecretsManagerBase]:
        """Get all available and configured secrets managers."""
        available = {}
        for name, manager in self._managers.items():
            if manager.is_available():
                available[name] = manager
        return available

    def get_preferred_manager(self) -> Optional[SecretsManagerBase]:
        """Get the first available manager from the preferred order."""
        available = self.get_available_managers()

        for preferred in self._preferred_order:
            if preferred in available:
                return available[preferred]

        # If no preferred manager is available, return any available one
        if available:
            return list(available.values())[0]

        return None

    def setup_default_managers(self) -> None:
        """Set up default secrets managers with environment-based configuration."""
        # AWS Secrets Manager
        try:
            aws_manager = AWSSecretsManager()
            self.register_manager("aws", aws_manager)
        except SecretsManagerError:
            pass  # AWS not available

        # Azure Key Vault
        try:
            if os.getenv("AZURE_KEY_VAULT_URL"):
                azure_manager = AzureKeyVault()
                self.register_manager("azure", azure_manager)
        except SecretsManagerError:
            pass  # Azure not available

        # HashiCorp Vault
        try:
            if os.getenv("VAULT_TOKEN"):
                vault_manager = HashiCorpVault()
                self.register_manager("vault", vault_manager)
        except SecretsManagerError:
            pass  # Vault not available


# Global registry instance
secrets_registry = SecretsManagerRegistry()
secrets_registry.setup_default_managers()
