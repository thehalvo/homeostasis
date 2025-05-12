"""
AWS integration for Homeostasis.

This package provides AWS integration for deploying fixes to:
- Lambda functions
- ECS (Elastic Container Service)
- EKS (Elastic Kubernetes Service)
"""

from modules.deployment.cloud.aws.provider import AWSProvider

__all__ = [
    "AWSProvider"
]