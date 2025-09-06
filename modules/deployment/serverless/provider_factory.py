"""
Serverless Provider Factory for Homeostasis.

This module provides factory functions to create and manage serverless providers.
"""

import logging
from typing import Any, Dict, Optional

from modules.deployment.serverless.aws_lambda import get_lambda_provider
from modules.deployment.serverless.azure_functions import get_azure_functions_provider
from modules.deployment.serverless.base_provider import ServerlessProvider
from modules.deployment.serverless.gcp_functions import get_functions_provider

logger = logging.getLogger(__name__)


def get_serverless_provider(
    provider_type: str, config: Dict[str, Any] = None
) -> Optional[ServerlessProvider]:
    """Get a serverless provider by type.

    Args:
        provider_type: Type of provider ("aws", "gcp", "azure")
        config: Optional configuration for the provider

    Returns:
        Optional[ServerlessProvider]: The serverless provider, or None if not available
    """
    if provider_type.lower() == "aws":
        return get_lambda_provider(config)
    elif provider_type.lower() == "gcp":
        return get_functions_provider(config)
    elif provider_type.lower() == "azure":
        return get_azure_functions_provider(config)
    else:
        logger.error(f"Unknown serverless provider type: {provider_type}")
        return None
