"""
Main Git Integration class that orchestrates all Git workflow components.

This module provides a unified interface for all Git workflow integration
features including pre-commit hooks, PR analysis, branch strategies,
commit analysis, and commit security.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from modules.monitoring.logger import MonitoringLogger

from .branch_strategy import BranchStrategy
from .commit_analyzer import CommitAnalyzer
from .commit_security import CommitSecurity
from .pr_analyzer import PRAnalyzer
from .pre_commit_hooks import PreCommitHooks


class GitIntegration:
    """Main class for Git workflow integration."""

    def __init__(self, repo_path: str, config_path: Optional[str] = None):
        """
        Initialize Git integration.

        Args:
            repo_path: Path to the Git repository
            config_path: Optional path to configuration file
        """
        self.repo_path = Path(repo_path)
        self.logger = MonitoringLogger(__name__)

        # Load configuration
        self.config = self._load_config(config_path)

        # Initialize components
        self.pre_commit_hooks = PreCommitHooks(
            str(self.repo_path), self.config.get("pre_commit", {})
        )

        self.pr_analyzer = PRAnalyzer(
            str(self.repo_path), self.config.get("pr_analysis", {})
        )

        self.branch_strategy = BranchStrategy(
            str(self.repo_path), self.config.get("branch_strategy", {})
        )

        self.commit_analyzer = CommitAnalyzer(
            str(self.repo_path), self.config.get("commit_analysis", {})
        )

        self.commit_security = CommitSecurity(
            str(self.repo_path), self.config.get("commit_signing", {})
        )

        self.logger.info("Git integration initialized")

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, "r") as f:
                    if config_path.endswith(".yaml") or config_path.endswith(".yml"):
                        return yaml.safe_load(f)
                    else:
                        return json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading config from {config_path}: {e}")

        # Try to load from orchestrator config
        orchestrator_config_path = self.repo_path / "orchestrator" / "config.yaml"
        if orchestrator_config_path.exists():
            try:
                with open(orchestrator_config_path, "r") as f:
                    config = yaml.safe_load(f)
                    return config.get("git_integration", {})
            except Exception as e:
                self.logger.warning(f"Could not load orchestrator config: {e}")

        # Return default configuration
        return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for Git integration."""
        return {
            "pre_commit": {
                "enabled": True,
                "block_on_critical": True,
                "analyze_changed_files_only": True,
                "auto_fix": False,
            },
            "pr_analysis": {
                "enabled": True,
                "auto_comment": False,
                "risk_threshold": 0.7,
            },
            "branch_strategy": {
                "enabled": True,
                "protected_branches": ["main", "master", "production"],
            },
            "commit_analysis": {"enabled": True, "conventional_commits": True},
            "commit_signing": {"enabled": False, "require_verification": False},
        }

    # Pre-commit hooks methods
    def install_pre_commit_hooks(self) -> bool:
        """Install pre-commit hooks for error prevention."""
        return self.pre_commit_hooks.install_hooks()

    def uninstall_pre_commit_hooks(self) -> bool:
        """Uninstall pre-commit hooks."""
        return self.pre_commit_hooks.uninstall_hooks()

    def run_pre_commit_analysis(self) -> tuple:
        """Run pre-commit analysis on staged files."""
        return self.pre_commit_hooks.run_pre_commit_analysis()

    # PR analysis methods
    def analyze_pull_request(self, pr_number: int) -> Any:
        """Analyze a pull request for issues and risks."""
        return self.pr_analyzer.analyze_pull_request(pr_number)

    def analyze_multiple_prs(self, pr_numbers: List[int]) -> List[Any]:
        """Analyze multiple pull requests in batch."""
        return self.pr_analyzer.analyze_pr_batch(pr_numbers)

    # Branch strategy methods
    def get_current_branch_info(self) -> Any:
        """Get information about the current branch."""
        return self.branch_strategy.get_current_branch_info()

    def get_healing_strategy(self, branch_name: Optional[str] = None) -> Any:
        """Get healing strategy for a branch."""
        return self.branch_strategy.get_healing_strategy(branch_name)

    def filter_issues_by_branch_strategy(
        self, issues: List[Dict[str, Any]], branch_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Filter issues based on branch healing strategy."""
        return self.branch_strategy.filter_issues_by_strategy(issues, branch_name)

    def get_branch_risk_assessment(
        self, branch_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get risk assessment for performing healing on a branch."""
        return self.branch_strategy.get_branch_risk_assessment(branch_name)

    # Commit analysis methods
    def analyze_commit(self, commit_hash: str) -> Any:
        """Analyze a specific commit for contextual information."""
        return self.commit_analyzer.analyze_commit(commit_hash)

    def analyze_commit_history(self, limit: Optional[int] = None) -> List[Any]:
        """Analyze commit history for patterns."""
        return self.commit_analyzer.analyze_commit_history(limit)

    def get_fix_patterns(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """Analyze patterns in fix commits."""
        return self.commit_analyzer.get_fix_patterns(limit)

    def get_healing_context_for_error(
        self, error_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get healing context by analyzing related commits."""
        return self.commit_analyzer.get_healing_context_for_error(error_info)

    # Commit security methods
    def setup_commit_signing(self, key_id: str) -> bool:
        """Setup GPG key for signing healing commits."""
        return self.commit_security.setup_gpg_key(key_id)

    def create_healing_commit(
        self, file_changes: Dict[str, str], healing_metadata: Dict[str, Any]
    ) -> bool:
        """Create a signed healing commit."""
        return self.commit_security.create_healing_commit(
            file_changes, healing_metadata
        )

    def verify_healing_commit(self, commit_hash: str) -> Dict[str, Any]:
        """Verify authenticity of a healing commit."""
        return self.commit_security.verify_healing_commit(commit_hash)

    def get_audit_trail(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get audit trail of healing commits."""
        return self.commit_security.get_audit_trail(limit)

    # Integration workflow methods
    def apply_branch_healing(
        self, issues: List[Dict[str, Any]], branch_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Apply healing with branch-aware strategies.

        Args:
            issues: List of issues to heal
            branch_name: Target branch (current branch if None)

        Returns:
            Healing results with branch context
        """
        try:
            # Get branch information and strategy
            branch_info = (
                self.branch_strategy.get_branch_info(branch_name)
                if branch_name
                else self.branch_strategy.get_current_branch_info()
            )

            strategy = self.branch_strategy.get_healing_strategy(branch_name)

            # Filter issues based on branch strategy
            filtered_issues = self.branch_strategy.filter_issues_by_strategy(
                issues, branch_name
            )

            # Check if healing is allowed
            if not strategy.auto_healing_enabled:
                return {
                    "success": False,
                    "reason": "Auto-healing disabled for this branch type",
                    "branch_info": branch_info,
                    "strategy": strategy,
                    "requires_approval": strategy.require_approval,
                }

            # Get risk assessment
            risk_assessment = self.branch_strategy.get_branch_risk_assessment(
                branch_name
            )

            if risk_assessment["risk_level"] == "high" and strategy.require_approval:
                return {
                    "success": False,
                    "reason": "High-risk branch requires manual approval",
                    "risk_assessment": risk_assessment,
                    "requires_approval": True,
                }

            # Apply healing based on strategy
            healing_results = []

            for issue in filtered_issues:
                # Get healing context from commit history
                healing_context = self.commit_analyzer.get_healing_context_for_error(
                    issue
                )

                # Apply confidence boost from similar fixes
                issue["confidence"] = min(
                    issue.get("confidence", 0.5)
                    + healing_context.get("confidence_boost", 0.0),
                    1.0,
                )

                # Add healing metadata
                healing_metadata = {
                    "rule_id": issue.get("rule_id"),
                    "confidence": issue["confidence"],
                    "file_path": issue.get("file_path"),
                    "branch": branch_info.name,
                    "branch_type": branch_info.branch_type.value,
                    "original_error": issue.get("message"),
                    "healing_type": (
                        "auto" if strategy.auto_healing_enabled else "manual"
                    ),
                }

                healing_results.append(
                    {
                        "issue": issue,
                        "metadata": healing_metadata,
                        "context": healing_context,
                        "applied": False,  # Would be set by actual healing implementation
                    }
                )

            return {
                "success": True,
                "branch_info": branch_info,
                "strategy": strategy,
                "risk_assessment": risk_assessment,
                "filtered_issues": len(filtered_issues),
                "total_issues": len(issues),
                "healing_results": healing_results,
                "requires_commit": strategy.auto_commit_enabled,
                "requires_testing": strategy.testing_required,
            }

        except Exception as e:
            self.logger.error(f"Error applying branch healing: {e}")
            return {"success": False, "error": str(e)}

    def create_comprehensive_report(
        self, branch_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a comprehensive Git integration report.

        Args:
            branch_name: Branch to analyze (current branch if None)

        Returns:
            Comprehensive report with all Git integration data
        """
        try:
            # Branch analysis
            branch_report = self.branch_strategy.create_branch_healing_report(
                branch_name
            )

            # Recent commit analysis
            recent_commits = self.commit_analyzer.analyze_commit_history(limit=20)

            # Fix patterns
            fix_patterns = self.commit_analyzer.get_fix_patterns(limit=100)

            # Audit trail
            audit_trail = self.commit_security.get_audit_trail(limit=10)

            # Pre-commit hook status
            hooks_installed = self.pre_commit_hooks.hooks_installed

            # Repository info
            repo_info = self.pr_analyzer.repo_info

            return {
                "repository": repo_info,
                "branch_analysis": branch_report,
                "recent_commits": {
                    "total_analyzed": len(recent_commits),
                    "fix_commits": len([c for c in recent_commits if c.is_fix_commit]),
                    "feature_commits": len(
                        [c for c in recent_commits if c.is_feature_commit]
                    ),
                    "avg_confidence": (
                        sum(c.confidence for c in recent_commits) / len(recent_commits)
                        if recent_commits
                        else 0
                    ),
                },
                "fix_patterns": fix_patterns,
                "security": {
                    "signing_enabled": self.commit_security.signing_enabled,
                    "audit_trail_entries": len(audit_trail),
                    "recent_healing_commits": audit_trail,
                },
                "pre_commit_hooks": {
                    "installed": hooks_installed,
                    "enabled": self.config.get("pre_commit", {}).get("enabled", True),
                },
                "report_timestamp": datetime.now().isoformat(),
                "configuration": self.config,
            }

        except Exception as e:
            self.logger.error(f"Error creating comprehensive report: {e}")
            return {"error": str(e), "report_timestamp": datetime.now().isoformat()}

    def save_configuration(self, config_path: Optional[str] = None) -> bool:
        """
        Save current configuration to file.

        Args:
            config_path: Path to save configuration

        Returns:
            True if configuration was saved successfully
        """
        try:
            if not config_path:
                config_path = self.repo_path / ".homeostasis-git-config.yaml"

            with open(config_path, "w") as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)

            self.logger.info(f"Configuration saved to {config_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
            return False

    def validate_git_integration(self) -> Dict[str, Any]:
        """
        Validate Git integration setup and configuration.

        Returns:
            Validation results with any issues found
        """
        validation_results = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "recommendations": [],
        }

        try:
            # Check if we're in a Git repository
            import subprocess

            result = subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                validation_results["valid"] = False
                validation_results["errors"].append("Not a Git repository")
                return validation_results

            # Check GPG availability for signing
            if self.config.get("commit_signing", {}).get("enabled", False):
                if not self.commit_security.gpg_available:
                    validation_results["warnings"].append(
                        "Commit signing enabled but GPG not available"
                    )
                    validation_results["recommendations"].append(
                        "Install GPG or disable commit signing"
                    )

            # Check pre-commit hooks
            if self.config.get("pre_commit", {}).get("enabled", True):
                if not self.pre_commit_hooks.hooks_installed:
                    validation_results["warnings"].append(
                        "Pre-commit hooks not installed"
                    )
                    validation_results["recommendations"].append(
                        "Run install_pre_commit_hooks() to enable"
                    )

            # Check API tokens for PR analysis
            if self.config.get("pr_analysis", {}).get("enabled", True):
                repo_type = self.pr_analyzer.repo_info.get("type")
                if repo_type == "github" and not self.pr_analyzer.github_token:
                    validation_results["warnings"].append(
                        "GitHub token not configured for PR analysis"
                    )
                    validation_results["recommendations"].append(
                        "Set GITHUB_TOKEN environment variable"
                    )
                elif repo_type == "gitlab" and not self.pr_analyzer.gitlab_token:
                    validation_results["warnings"].append(
                        "GitLab token not configured for PR analysis"
                    )
                    validation_results["recommendations"].append(
                        "Set GITLAB_TOKEN environment variable"
                    )

            # Check branch protection
            current_branch = self.branch_strategy.get_current_branch_info()
            if current_branch.is_protected and current_branch.branch_type.value in [
                "main",
                "production",
            ]:
                healing_strategy = self.branch_strategy.get_healing_strategy()
                if healing_strategy.auto_healing_enabled:
                    validation_results["warnings"].append(
                        f"Auto-healing enabled on protected branch: {current_branch.name}"
                    )
                    validation_results["recommendations"].append(
                        "Consider disabling auto-healing on protected branches"
                    )

            self.logger.info("Git integration validation completed")

        except Exception as e:
            validation_results["valid"] = False
            validation_results["errors"].append(f"Validation error: {e}")

        return validation_results
