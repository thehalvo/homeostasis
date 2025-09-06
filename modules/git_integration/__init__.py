"""
Git Workflow Integration Module

This module provides comprehensive Git workflow integration for the Homeostasis
self-healing framework, including pre-commit hooks, PR analysis, branch-aware
healing, commit analysis, and commit security features.
"""

from .branch_strategy import BranchStrategy
from .commit_analyzer import CommitAnalyzer
from .commit_security import CommitSecurity
from .git_integration import GitIntegration
from .pr_analyzer import PRAnalyzer
from .pre_commit_hooks import PreCommitHooks

__all__ = [
    "GitIntegration",
    "PreCommitHooks",
    "PRAnalyzer",
    "BranchStrategy",
    "CommitAnalyzer",
    "CommitSecurity",
]

__version__ = "0.1.0"
__author__ = "Homeostasis Project"
__description__ = "Git workflow integration for self-healing systems"
