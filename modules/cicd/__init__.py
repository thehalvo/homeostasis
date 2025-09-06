"""
CI/CD Pipeline Integration Module

This module provides integrations with various CI/CD platforms to enable
automated healing during build and deployment processes.

Supported platforms:
- GitHub Actions
- GitLab CI
- Jenkins
- CircleCI
- Deployment platforms (Vercel, Netlify, etc.)
"""

from .circleci import CircleCIIntegration
from .deployment_platforms import DeploymentPlatformIntegration
from .github_actions import GitHubActionsIntegration
from .gitlab_ci import GitLabCIIntegration
from .jenkins import JenkinsIntegration

__all__ = [
    "GitHubActionsIntegration",
    "GitLabCIIntegration",
    "JenkinsIntegration",
    "CircleCIIntegration",
    "DeploymentPlatformIntegration",
]
