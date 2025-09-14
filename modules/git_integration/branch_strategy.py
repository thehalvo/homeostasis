"""
Branch-aware healing strategies for Git workflows.

This module implements intelligent healing strategies that adapt based on
the Git branch context, applying different healing approaches for different
branch types and managing healing scope appropriately.
"""

import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from modules.monitoring.logger import MonitoringLogger


class BranchType(Enum):
    """Enumeration of branch types for healing strategies."""

    MAIN = "main"
    PRODUCTION = "production"
    DEVELOPMENT = "development"
    FEATURE = "feature"
    HOTFIX = "hotfix"
    RELEASE = "release"
    EXPERIMENTAL = "experimental"
    UNKNOWN = "unknown"


@dataclass
class BranchInfo:
    """Information about a Git branch."""

    name: str
    branch_type: BranchType
    is_protected: bool
    last_commit: str
    last_commit_date: datetime
    commits_behind_main: int
    commits_ahead_main: int
    active_prs: List[int]


@dataclass
class HealingStrategy:
    """Healing strategy configuration for a branch type."""

    branch_type: BranchType
    auto_healing_enabled: bool
    auto_commit_enabled: bool
    require_approval: bool
    max_changes_per_healing: int
    confidence_threshold: float
    allowed_severity_levels: List[str]
    notification_required: bool
    rollback_enabled: bool
    testing_required: bool


class BranchStrategy:
    """Implements branch-aware healing strategies."""

    def __init__(self, repo_path: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize branch strategy manager.

        Args:
            repo_path: Path to the Git repository
            config: Configuration dictionary for branch strategies
        """
        self.repo_path = Path(repo_path)
        self.config = config or self._load_default_config()
        self.logger = MonitoringLogger(__name__)

        # Initialize branch strategies
        self.strategies = self._initialize_strategies()

        # Cache for branch information
        self._branch_cache: Dict[str, BranchInfo] = {}
        self._cache_expiry: Dict[str, datetime] = {}

    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration for branch strategies."""
        return {
            "branch_patterns": {
                "main": ["main", "master"],
                "production": ["prod", "production", "live"],
                "development": ["dev", "develop", "development"],
                "feature": ["feature/", "feat/"],
                "hotfix": ["hotfix/", "fix/"],
                "release": ["release/", "rel/"],
                "experimental": ["exp/", "experiment/", "poc/"],
            },
            "protected_branches": ["main", "master", "production", "develop"],
            "healing_strategies": {
                "main": {
                    "auto_healing_enabled": False,
                    "auto_commit_enabled": False,
                    "require_approval": True,
                    "max_changes_per_healing": 1,
                    "confidence_threshold": 0.9,
                    "allowed_severity_levels": ["critical"],
                    "notification_required": True,
                    "rollback_enabled": True,
                    "testing_required": True,
                },
                "production": {
                    "auto_healing_enabled": False,
                    "auto_commit_enabled": False,
                    "require_approval": True,
                    "max_changes_per_healing": 1,
                    "confidence_threshold": 0.95,
                    "allowed_severity_levels": ["critical"],
                    "notification_required": True,
                    "rollback_enabled": True,
                    "testing_required": True,
                },
                "development": {
                    "auto_healing_enabled": True,
                    "auto_commit_enabled": True,
                    "require_approval": False,
                    "max_changes_per_healing": 5,
                    "confidence_threshold": 0.7,
                    "allowed_severity_levels": ["critical", "warning"],
                    "notification_required": False,
                    "rollback_enabled": True,
                    "testing_required": False,
                },
                "feature": {
                    "auto_healing_enabled": True,
                    "auto_commit_enabled": True,
                    "require_approval": False,
                    "max_changes_per_healing": 10,
                    "confidence_threshold": 0.6,
                    "allowed_severity_levels": ["critical", "warning", "info"],
                    "notification_required": False,
                    "rollback_enabled": True,
                    "testing_required": False,
                },
                "hotfix": {
                    "auto_healing_enabled": True,
                    "auto_commit_enabled": False,
                    "require_approval": True,
                    "max_changes_per_healing": 3,
                    "confidence_threshold": 0.8,
                    "allowed_severity_levels": ["critical", "warning"],
                    "notification_required": True,
                    "rollback_enabled": True,
                    "testing_required": True,
                },
                "release": {
                    "auto_healing_enabled": False,
                    "auto_commit_enabled": False,
                    "require_approval": True,
                    "max_changes_per_healing": 1,
                    "confidence_threshold": 0.9,
                    "allowed_severity_levels": ["critical"],
                    "notification_required": True,
                    "rollback_enabled": True,
                    "testing_required": True,
                },
                "experimental": {
                    "auto_healing_enabled": True,
                    "auto_commit_enabled": True,
                    "require_approval": False,
                    "max_changes_per_healing": 20,
                    "confidence_threshold": 0.4,
                    "allowed_severity_levels": ["critical", "warning", "info"],
                    "notification_required": False,
                    "rollback_enabled": False,
                    "testing_required": False,
                },
            },
            "cache_ttl_minutes": 10,
        }

    def _initialize_strategies(self) -> Dict[BranchType, HealingStrategy]:
        """Initialize healing strategies for each branch type."""
        strategies = {}

        for branch_type_name, strategy_config in self.config[
            "healing_strategies"
        ].items():
            try:
                branch_type = BranchType(branch_type_name)
                strategy = HealingStrategy(branch_type=branch_type, **strategy_config)
                strategies[branch_type] = strategy
            except ValueError:
                self.logger.warning(
                    f"Unknown branch type in config: {branch_type_name}"
                )

        return strategies

    def get_current_branch_info(self) -> BranchInfo:
        """Get information about the current Git branch."""
        try:
            # Get current branch name
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                raise Exception(f"Failed to get current branch: {result.stderr}")

            branch_name = result.stdout.strip()
            return self.get_branch_info(branch_name)

        except Exception as e:
            self.logger.error(f"Error getting current branch info: {e}")
            return self._create_unknown_branch_info("unknown")

    def get_branch_info(self, branch_name: str) -> BranchInfo:
        """
        Get detailed information about a specific branch.

        Args:
            branch_name: Name of the branch

        Returns:
            Branch information object
        """
        # Check cache first
        if self._is_cached_and_valid(branch_name):
            return self._branch_cache[branch_name]

        try:
            # Determine branch type
            branch_type = self._classify_branch(branch_name)

            # Check if branch is protected
            is_protected = self._is_protected_branch(branch_name)

            # Get last commit information
            last_commit, last_commit_date = self._get_last_commit_info(branch_name)

            # Get commits ahead/behind main
            commits_behind, commits_ahead = self._get_commit_comparison(branch_name)

            # Get active PRs (this would need API integration)
            active_prs = self._get_active_prs(branch_name)

            branch_info = BranchInfo(
                name=branch_name,
                branch_type=branch_type,
                is_protected=is_protected,
                last_commit=last_commit,
                last_commit_date=last_commit_date,
                commits_behind_main=commits_behind,
                commits_ahead_main=commits_ahead,
                active_prs=active_prs,
            )

            # Cache the result
            self._cache_branch_info(branch_name, branch_info)

            return branch_info

        except Exception as e:
            self.logger.error(f"Error getting branch info for {branch_name}: {e}")
            return self._create_unknown_branch_info(branch_name)

    def _classify_branch(self, branch_name: str) -> BranchType:
        """Classify a branch based on its name."""
        patterns = self.config["branch_patterns"]

        # Check exact matches first
        for branch_type, branch_patterns in patterns.items():
            for pattern in branch_patterns:
                if not pattern.endswith("/"):
                    # Exact match
                    if branch_name == pattern:
                        return BranchType(branch_type)
                else:
                    # Prefix match
                    if branch_name.startswith(pattern):
                        return BranchType(branch_type)

        return BranchType.UNKNOWN

    def _is_protected_branch(self, branch_name: str) -> bool:
        """Check if a branch is protected."""
        protected_branches = self.config.get("protected_branches", [])
        return branch_name in protected_branches

    def _get_last_commit_info(self, branch_name: str) -> Tuple[str, datetime]:
        """Get last commit hash and date for a branch."""
        try:
            # Get commit hash
            hash_result = subprocess.run(
                ["git", "rev-parse", f"{branch_name}^{{commit}}"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
            )

            if hash_result.returncode != 0:
                return "unknown", datetime.now()

            commit_hash = hash_result.stdout.strip()

            # Get commit date
            date_result = subprocess.run(
                ["git", "log", "-1", "--format=%ci", branch_name],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
            )

            if date_result.returncode != 0:
                return commit_hash, datetime.now()

            date_str = date_result.stdout.strip()
            commit_date = datetime.fromisoformat(date_str.replace(" ", "T", 1))

            return commit_hash, commit_date

        except Exception as e:
            self.logger.error(f"Error getting last commit info: {e}")
            return "unknown", datetime.now()

    def _get_commit_comparison(self, branch_name: str) -> Tuple[int, int]:
        """Get commits behind and ahead of main branch."""
        try:
            # Find main branch
            main_branch = self._find_main_branch()
            if not main_branch or main_branch == branch_name:
                return 0, 0

            # Get commits behind main
            behind_result = subprocess.run(
                ["git", "rev-list", "--count", f"{branch_name}..{main_branch}"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
            )

            commits_behind = 0
            if behind_result.returncode == 0:
                commits_behind = int(behind_result.stdout.strip())

            # Get commits ahead of main
            ahead_result = subprocess.run(
                ["git", "rev-list", "--count", f"{main_branch}..{branch_name}"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
            )

            commits_ahead = 0
            if ahead_result.returncode == 0:
                commits_ahead = int(ahead_result.stdout.strip())

            return commits_behind, commits_ahead

        except Exception as e:
            self.logger.error(f"Error getting commit comparison: {e}")
            return 0, 0

    def _find_main_branch(self) -> Optional[str]:
        """Find the main branch of the repository."""
        main_patterns = self.config["branch_patterns"].get("main", ["main", "master"])

        for pattern in main_patterns:
            try:
                result = subprocess.run(
                    ["git", "rev-parse", "--verify", pattern],
                    cwd=self.repo_path,
                    capture_output=True,
                    text=True,
                )

                if result.returncode == 0:
                    return str(pattern)
            except Exception:
                continue

        return None

    def _get_active_prs(self, branch_name: str) -> List[int]:
        """Get active PRs for a branch (placeholder implementation)."""
        # This would require integration with GitHub/GitLab APIs
        # For now, return empty list
        return []

    def _is_cached_and_valid(self, branch_name: str) -> bool:
        """Check if branch info is cached and still valid."""
        if branch_name not in self._branch_cache:
            return False

        expiry_time = self._cache_expiry.get(branch_name)
        if not expiry_time:
            return False

        return datetime.now() < expiry_time

    def _cache_branch_info(self, branch_name: str, branch_info: BranchInfo) -> None:
        """Cache branch information."""
        self._branch_cache[branch_name] = branch_info

        ttl_minutes = self.config.get("cache_ttl_minutes", 10)
        self._cache_expiry[branch_name] = datetime.now() + timedelta(
            minutes=ttl_minutes
        )

    def _create_unknown_branch_info(self, branch_name: str) -> BranchInfo:
        """Create a default branch info object for unknown branches."""
        return BranchInfo(
            name=branch_name,
            branch_type=BranchType.UNKNOWN,
            is_protected=False,
            last_commit="unknown",
            last_commit_date=datetime.now(),
            commits_behind_main=0,
            commits_ahead_main=0,
            active_prs=[],
        )

    def get_healing_strategy(
        self, branch_name: Optional[str] = None
    ) -> HealingStrategy:
        """
        Get the healing strategy for a branch.

        Args:
            branch_name: Name of the branch (current branch if None)

        Returns:
            Healing strategy for the branch
        """
        if branch_name is None:
            branch_info = self.get_current_branch_info()
        else:
            branch_info = self.get_branch_info(branch_name)

        return self.strategies.get(
            branch_info.branch_type, self._get_default_strategy()
        )

    def _get_default_strategy(self) -> HealingStrategy:
        """Get default healing strategy for unknown branch types."""
        return HealingStrategy(
            branch_type=BranchType.UNKNOWN,
            auto_healing_enabled=False,
            auto_commit_enabled=False,
            require_approval=True,
            max_changes_per_healing=1,
            confidence_threshold=0.8,
            allowed_severity_levels=["critical"],
            notification_required=True,
            rollback_enabled=True,
            testing_required=True,
        )

    def can_auto_heal(self, branch_name: Optional[str] = None) -> bool:
        """Check if auto-healing is allowed for a branch."""
        strategy = self.get_healing_strategy(branch_name)
        return strategy.auto_healing_enabled

    def can_auto_commit(self, branch_name: Optional[str] = None) -> bool:
        """Check if auto-commit is allowed for a branch."""
        strategy = self.get_healing_strategy(branch_name)
        return strategy.auto_commit_enabled

    def requires_approval(self, branch_name: Optional[str] = None) -> bool:
        """Check if healing requires approval for a branch."""
        strategy = self.get_healing_strategy(branch_name)
        return strategy.require_approval

    def get_max_changes_per_healing(self, branch_name: Optional[str] = None) -> int:
        """Get maximum changes allowed per healing session."""
        strategy = self.get_healing_strategy(branch_name)
        return strategy.max_changes_per_healing

    def get_confidence_threshold(self, branch_name: Optional[str] = None) -> float:
        """Get confidence threshold for healing decisions."""
        strategy = self.get_healing_strategy(branch_name)
        return strategy.confidence_threshold

    def is_severity_allowed(
        self, severity: str, branch_name: Optional[str] = None
    ) -> bool:
        """Check if healing is allowed for a specific severity level."""
        strategy = self.get_healing_strategy(branch_name)
        return severity in strategy.allowed_severity_levels

    def requires_notification(self, branch_name: Optional[str] = None) -> bool:
        """Check if healing requires notification."""
        strategy = self.get_healing_strategy(branch_name)
        return strategy.notification_required

    def is_rollback_enabled(self, branch_name: Optional[str] = None) -> bool:
        """Check if rollback is enabled for a branch."""
        strategy = self.get_healing_strategy(branch_name)
        return strategy.rollback_enabled

    def requires_testing(self, branch_name: Optional[str] = None) -> bool:
        """Check if healing requires testing."""
        strategy = self.get_healing_strategy(branch_name)
        return strategy.testing_required

    def filter_issues_by_strategy(
        self, issues: List[Dict[str, Any]], branch_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Filter issues based on branch healing strategy.

        Args:
            issues: List of issues to filter
            branch_name: Branch name (current branch if None)

        Returns:
            Filtered list of issues that can be healed on this branch
        """
        strategy = self.get_healing_strategy(branch_name)
        filtered_issues = []

        for issue in issues:
            # Check severity level
            severity = issue.get("severity", "unknown")
            if not self.is_severity_allowed(severity, branch_name):
                continue

            # Check confidence threshold
            confidence = issue.get("confidence", 0.0)
            if confidence < strategy.confidence_threshold:
                continue

            filtered_issues.append(issue)

        # Limit number of issues based on strategy
        return filtered_issues[: strategy.max_changes_per_healing]

    def get_branch_risk_assessment(
        self, branch_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Assess the risk of performing healing on a branch.

        Args:
            branch_name: Branch name (current branch if None)

        Returns:
            Risk assessment information
        """
        branch_info = (
            self.get_branch_info(branch_name)
            if branch_name
            else self.get_current_branch_info()
        )
        strategy = self.get_healing_strategy(branch_name)

        risk_factors = []
        risk_score = 0.0

        # Protected branch risk
        if branch_info.is_protected:
            risk_factors.append("Protected branch")
            risk_score += 0.3

        # Branch type risk
        high_risk_types = [BranchType.MAIN, BranchType.PRODUCTION, BranchType.RELEASE]
        if branch_info.branch_type in high_risk_types:
            risk_factors.append(
                f"High-risk branch type ({branch_info.branch_type.value})"
            )
            risk_score += 0.4

        # Outdated branch risk
        if branch_info.commits_behind_main > 10:
            risk_factors.append(
                f"Branch is {branch_info.commits_behind_main} commits behind main"
            )
            risk_score += 0.2

        # Active PRs risk
        if branch_info.active_prs:
            risk_factors.append(f"Branch has {len(branch_info.active_prs)} active PRs")
            risk_score += 0.1

        # Strategy restrictions
        if strategy.require_approval:
            risk_factors.append("Requires approval")

        if not strategy.auto_healing_enabled:
            risk_factors.append("Auto-healing disabled")

        risk_level = "low"
        if risk_score > 0.7:
            risk_level = "high"
        elif risk_score > 0.4:
            risk_level = "medium"

        return {
            "risk_score": min(risk_score, 1.0),
            "risk_level": risk_level,
            "risk_factors": risk_factors,
            "branch_info": branch_info,
            "healing_strategy": strategy,
        }

    def get_healing_recommendations(
        self, branch_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get healing recommendations based on branch strategy.

        Args:
            branch_name: Branch name (current branch if None)

        Returns:
            List of healing recommendations
        """
        risk_assessment = self.get_branch_risk_assessment(branch_name)
        strategy = risk_assessment["healing_strategy"]

        recommendations = []

        # Auto-healing recommendation
        if strategy.auto_healing_enabled:
            recommendations.append(
                {
                    "type": "auto_healing",
                    "message": "Auto-healing is enabled for this branch",
                    "action": "Issues will be automatically fixed",
                    "priority": "info",
                }
            )
        else:
            recommendations.append(
                {
                    "type": "manual_healing",
                    "message": "Auto-healing is disabled - manual review required",
                    "action": "Review and approve healing changes manually",
                    "priority": "medium",
                }
            )

        # Testing recommendation
        if strategy.testing_required:
            recommendations.append(
                {
                    "type": "testing",
                    "message": "Testing is required before applying healing",
                    "action": "Run full test suite before healing",
                    "priority": "high",
                }
            )

        # Approval recommendation
        if strategy.require_approval:
            recommendations.append(
                {
                    "type": "approval",
                    "message": "Healing changes require approval",
                    "action": "Get approval from team lead or senior developer",
                    "priority": "high",
                }
            )

        # Risk-based recommendations
        if risk_assessment["risk_level"] == "high":
            recommendations.append(
                {
                    "type": "risk_mitigation",
                    "message": "High-risk branch detected",
                    "action": "Consider additional safeguards or manual healing",
                    "priority": "high",
                }
            )

        return recommendations

    def create_branch_healing_report(
        self, branch_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a comprehensive healing report for a branch.

        Args:
            branch_name: Branch name (current branch if None)

        Returns:
            Comprehensive branch healing report
        """
        branch_info = (
            self.get_branch_info(branch_name)
            if branch_name
            else self.get_current_branch_info()
        )
        strategy = self.get_healing_strategy(branch_name)
        risk_assessment = self.get_branch_risk_assessment(branch_name)
        recommendations = self.get_healing_recommendations(branch_name)

        return {
            "branch_info": {
                "name": branch_info.name,
                "type": branch_info.branch_type.value,
                "is_protected": branch_info.is_protected,
                "last_commit": branch_info.last_commit[:8],
                "last_commit_date": branch_info.last_commit_date.isoformat(),
                "commits_behind_main": branch_info.commits_behind_main,
                "commits_ahead_main": branch_info.commits_ahead_main,
                "active_prs": branch_info.active_prs,
            },
            "healing_strategy": {
                "auto_healing_enabled": strategy.auto_healing_enabled,
                "auto_commit_enabled": strategy.auto_commit_enabled,
                "require_approval": strategy.require_approval,
                "max_changes_per_healing": strategy.max_changes_per_healing,
                "confidence_threshold": strategy.confidence_threshold,
                "allowed_severity_levels": strategy.allowed_severity_levels,
                "notification_required": strategy.notification_required,
                "rollback_enabled": strategy.rollback_enabled,
                "testing_required": strategy.testing_required,
            },
            "risk_assessment": risk_assessment,
            "recommendations": recommendations,
            "report_timestamp": datetime.now().isoformat(),
        }
