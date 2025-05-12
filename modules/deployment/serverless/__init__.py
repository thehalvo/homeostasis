"""
Serverless function deployment module for Homeostasis.

This module provides integration with serverless platforms
to deploy and manage self-healing functions.
"""

from modules.deployment.serverless.base_provider import ServerlessProvider
from modules.deployment.serverless.aws_lambda import (
    AWSLambdaProvider,
    get_lambda_provider
)
from modules.deployment.serverless.gcp_functions import (
    GCPFunctionsProvider,
    get_functions_provider
)
from modules.deployment.serverless.azure_functions import (
    AzureFunctionsProvider,
    get_azure_functions_provider
)
from modules.deployment.serverless.provider_factory import (
    get_serverless_provider
)

__all__ = [
    'ServerlessProvider',
    'AWSLambdaProvider',
    'GCPFunctionsProvider',
    'AzureFunctionsProvider',
    'get_lambda_provider',
    'get_functions_provider',
    'get_azure_functions_provider',
    'get_serverless_provider'
]