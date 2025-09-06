"""
Pull Request Analysis and Suggestion System.

This module analyzes pull requests to identify potential issues, generate
healing suggestions, and provide risk assessments for code changes.
"""

import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from modules.analysis.language_adapters import LanguageAdapterManager
from modules.analysis.rule_based import RuleBasedAnalyzer
from modules.monitoring.logger import HomeostasisLogger
from modules.patch_generation.patcher import Patcher


@dataclass
class PRChange:
    """Represents a change in a pull request."""

    file_path: str
    language: str
    additions: int
    deletions: int
    changes_type: str  # 'added', 'modified', 'deleted'
    diff_content: str


@dataclass
class PRAnalysisResult:
    """Result of pull request analysis."""

    pr_number: int
    risk_score: float
    issues_found: List[Dict[str, Any]]
    suggestions: List[Dict[str, Any]]
    complexity_metrics: Dict[str, Any]
    affected_components: List[str]
    healing_recommendations: List[Dict[str, Any]]
    analysis_timestamp: str


class PRAnalyzer:
    """Analyzes pull requests for potential issues and healing opportunities."""

    def __init__(self, repo_path: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize PR analyzer.

        Args:
            repo_path: Path to the Git repository
            config: Configuration dictionary for PR analysis
        """
        self.repo_path = Path(repo_path)
        self.config = config or self._load_default_config()
        self.logger = HomeostasisLogger(__name__)

        # Initialize analysis components
        self.analyzer = RuleBasedAnalyzer()
        self.language_manager = LanguageAdapterManager()
        self.patcher = Patcher()

        # GitHub/GitLab API configuration
        self.github_token = self.config.get("github_token", os.getenv("GITHUB_TOKEN"))
        self.gitlab_token = self.config.get("gitlab_token", os.getenv("GITLAB_TOKEN"))

        # Repository information
        self.repo_info = self._detect_repository_info()

    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration for PR analysis."""
        return {
            "enabled": True,
            "auto_comment": False,
            "risk_threshold": 0.7,
            "max_files_per_pr": 100,
            "analysis_timeout_minutes": 15,
            "complexity_metrics": {
                "cyclomatic_complexity": True,
                "code_churn": True,
                "test_coverage_impact": True,
            },
            "notification_channels": [],
            "ignore_patterns": ["*.md", "*.txt", "docs/", "examples/", "__pycache__/"],
        }

    def _detect_repository_info(self) -> Dict[str, Any]:
        """Detect repository information (GitHub, GitLab, etc.)."""
        try:
            # Get remote URL
            result = subprocess.run(
                ["git", "remote", "get-url", "origin"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                return {"type": "local", "url": None}

            remote_url = result.stdout.strip()

            # Parse repository info
            if "github.com" in remote_url:
                return self._parse_github_info(remote_url)
            elif "gitlab.com" in remote_url or "gitlab" in remote_url:
                return self._parse_gitlab_info(remote_url)
            else:
                return {"type": "unknown", "url": remote_url}

        except Exception as e:
            self.logger.error(f"Error detecting repository info: {e}")
            return {"type": "local", "url": None}

    def _parse_github_info(self, remote_url: str) -> Dict[str, Any]:
        """Parse GitHub repository information."""
        # Handle both HTTPS and SSH URLs
        patterns = [
            r"https://github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$",
            r"git@github\.com:([^/]+)/([^/]+?)(?:\.git)?/?$",
        ]

        for pattern in patterns:
            match = re.search(pattern, remote_url)
            if match:
                owner, repo = match.groups()
                return {
                    "type": "github",
                    "owner": owner,
                    "repo": repo,
                    "url": remote_url,
                    "api_url": f"https://api.github.com/repos/{owner}/{repo}",
                }

        return {"type": "github", "url": remote_url}

    def _parse_gitlab_info(self, remote_url: str) -> Dict[str, Any]:
        """Parse GitLab repository information."""
        # Handle both HTTPS and SSH URLs
        patterns = [
            r"https://gitlab\.com/([^/]+)/([^/]+?)(?:\.git)?/?$",
            r"git@gitlab\.com:([^/]+)/([^/]+?)(?:\.git)?/?$",
        ]

        for pattern in patterns:
            match = re.search(pattern, remote_url)
            if match:
                owner, repo = match.groups()
                return {
                    "type": "gitlab",
                    "owner": owner,
                    "repo": repo,
                    "url": remote_url,
                    "api_url": f"https://gitlab.com/api/v4/projects/{owner}%2F{repo}",
                }

        return {"type": "gitlab", "url": remote_url}

    def analyze_pull_request(self, pr_number: int) -> PRAnalysisResult:
        """
        Analyze a pull request for potential issues and healing opportunities.

        Args:
            pr_number: Pull request number

        Returns:
            Analysis result with issues, suggestions, and recommendations
        """
        try:
            self.logger.info(f"Analyzing PR #{pr_number}")

            # Get PR changes
            pr_changes = self._get_pr_changes(pr_number)
            if not pr_changes:
                self.logger.warning(f"No changes found for PR #{pr_number}")
                return self._create_empty_result(pr_number)

            # Analyze changes
            issues_found = []
            suggestions = []
            affected_components = set()

            for change in pr_changes:
                # Skip ignored files
                if self._should_ignore_file(change.file_path):
                    continue

                # Analyze file changes
                file_issues = self._analyze_file_changes(change)
                issues_found.extend(file_issues)

                # Generate suggestions
                file_suggestions = self._generate_suggestions(change, file_issues)
                suggestions.extend(file_suggestions)

                # Track affected components
                component = self._identify_component(change.file_path)
                if component:
                    affected_components.add(component)

            # Calculate complexity metrics
            complexity_metrics = self._calculate_complexity_metrics(pr_changes)

            # Calculate risk score
            risk_score = self._calculate_risk_score(
                issues_found, complexity_metrics, pr_changes
            )

            # Generate healing recommendations
            healing_recommendations = self._generate_healing_recommendations(
                issues_found, pr_changes, risk_score
            )

            # Create analysis result
            result = PRAnalysisResult(
                pr_number=pr_number,
                risk_score=risk_score,
                issues_found=issues_found,
                suggestions=suggestions,
                complexity_metrics=complexity_metrics,
                affected_components=list(affected_components),
                healing_recommendations=healing_recommendations,
                analysis_timestamp=datetime.now().isoformat(),
            )

            # Post comment if enabled
            if self.config.get("auto_comment", False):
                self._post_pr_comment(pr_number, result)

            self.logger.info(
                f"PR #{pr_number} analysis completed: {len(issues_found)} issues, risk={risk_score:.2f}"
            )
            return result

        except Exception as e:
            self.logger.error(f"Error analyzing PR #{pr_number}: {e}")
            return self._create_empty_result(pr_number)

    def _get_pr_changes(self, pr_number: int) -> List[PRChange]:
        """Get changes from a pull request."""
        if self.repo_info["type"] == "github":
            return self._get_github_pr_changes(pr_number)
        elif self.repo_info["type"] == "gitlab":
            return self._get_gitlab_pr_changes(pr_number)
        else:
            # Fallback to local git analysis
            return self._get_local_pr_changes(pr_number)

    def _get_github_pr_changes(self, pr_number: int) -> List[PRChange]:
        """Get PR changes from GitHub API."""
        if not self.github_token:
            self.logger.warning(
                "GitHub token not configured, falling back to local analysis"
            )
            return self._get_local_pr_changes(pr_number)

        try:
            api_url = self.repo_info["api_url"]
            headers = {
                "Authorization": f"token {self.github_token}",
                "Accept": "application/vnd.github.v3+json",
            }

            # Get PR files
            files_url = f"{api_url}/pulls/{pr_number}/files"
            response = requests.get(files_url, headers=headers, timeout=30)
            response.raise_for_status()

            files_data = response.json()
            changes = []

            for file_data in files_data:
                change = PRChange(
                    file_path=file_data["filename"],
                    language=self.language_manager.detect_language(
                        file_data["filename"]
                    ),
                    additions=file_data["additions"],
                    deletions=file_data["deletions"],
                    changes_type=file_data["status"],
                    diff_content=file_data.get("patch", ""),
                )
                changes.append(change)

            return changes

        except Exception as e:
            self.logger.error(f"Error getting GitHub PR changes: {e}")
            return self._get_local_pr_changes(pr_number)

    def _get_gitlab_pr_changes(self, pr_number: int) -> List[PRChange]:
        """Get PR changes from GitLab API."""
        if not self.gitlab_token:
            self.logger.warning(
                "GitLab token not configured, falling back to local analysis"
            )
            return self._get_local_pr_changes(pr_number)

        try:
            api_url = self.repo_info["api_url"]
            headers = {
                "Authorization": f"Bearer {self.gitlab_token}",
                "Content-Type": "application/json",
            }

            # Get MR changes
            changes_url = f"{api_url}/merge_requests/{pr_number}/changes"
            response = requests.get(changes_url, headers=headers, timeout=30)
            response.raise_for_status()

            changes_data = response.json()
            changes = []

            for file_data in changes_data.get("changes", []):
                change = PRChange(
                    file_path=file_data["new_path"] or file_data["old_path"],
                    language=self.language_manager.detect_language(
                        file_data["new_path"] or file_data["old_path"]
                    ),
                    additions=file_data.get("additions", 0),
                    deletions=file_data.get("deletions", 0),
                    changes_type="modified",  # GitLab doesn't provide clear status
                    diff_content=file_data.get("diff", ""),
                )
                changes.append(change)

            return changes

        except Exception as e:
            self.logger.error(f"Error getting GitLab MR changes: {e}")
            return self._get_local_pr_changes(pr_number)

    def _get_local_pr_changes(self, pr_number: int) -> List[PRChange]:
        """Get PR changes using local git commands."""
        try:
            # This is a simplified approach - in practice, you'd need the branch names
            # For now, we'll analyze recent changes
            result = subprocess.run(
                ["git", "diff", "--name-status", "HEAD~1", "HEAD"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                return []

            changes = []
            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue

                parts = line.split("\t")
                if len(parts) < 2:
                    continue

                status = parts[0]
                file_path = parts[1]

                # Get diff for this file
                diff_result = subprocess.run(
                    ["git", "diff", "HEAD~1", "HEAD", "--", file_path],
                    cwd=self.repo_path,
                    capture_output=True,
                    text=True,
                )

                change = PRChange(
                    file_path=file_path,
                    language=self.language_manager.detect_language(file_path),
                    additions=0,  # Would need to parse diff to get accurate counts
                    deletions=0,
                    changes_type=self._map_git_status(status),
                    diff_content=(
                        diff_result.stdout if diff_result.returncode == 0 else ""
                    ),
                )
                changes.append(change)

            return changes

        except Exception as e:
            self.logger.error(f"Error getting local PR changes: {e}")
            return []

    def _map_git_status(self, status: str) -> str:
        """Map git status codes to change types."""
        status_map = {
            "A": "added",
            "M": "modified",
            "D": "deleted",
            "R": "renamed",
            "C": "copied",
        }
        return status_map.get(status[0], "modified")

    def _should_ignore_file(self, file_path: str) -> bool:
        """Check if file should be ignored based on patterns."""
        ignore_patterns = self.config.get("ignore_patterns", [])

        for pattern in ignore_patterns:
            if pattern in file_path or file_path.endswith(pattern.lstrip("*")):
                return True

        return False

    def _analyze_file_changes(self, change: PRChange) -> List[Dict[str, Any]]:
        """Analyze changes in a single file."""
        try:
            full_path = self.repo_path / change.file_path

            # Read current file content if it exists
            if full_path.exists():
                with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                # Run analysis
                issues = self.analyzer.analyze_code(
                    content=content, language=change.language, file_path=str(full_path)
                )

                # Add change context to issues
                for issue in issues:
                    issue["file_path"] = change.file_path
                    issue["language"] = change.language
                    issue["change_type"] = change.changes_type
                    issue["pr_context"] = True

                return issues

            return []

        except Exception as e:
            self.logger.error(
                f"Error analyzing file changes for {change.file_path}: {e}"
            )
            return []

    def _generate_suggestions(
        self, change: PRChange, issues: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate suggestions for addressing issues in the changed file."""
        suggestions = []

        for issue in issues:
            if issue.get("fixable", False):
                suggestion = {
                    "file_path": change.file_path,
                    "issue_id": issue.get("rule_id"),
                    "type": "auto_fix",
                    "message": f"Auto-fix available for: {issue.get('message')}",
                    "confidence": issue.get("confidence", 0.5),
                    "suggested_fix": issue.get("suggested_fix", ""),
                }
                suggestions.append(suggestion)
            else:
                # Manual review suggestions
                suggestion = {
                    "file_path": change.file_path,
                    "issue_id": issue.get("rule_id"),
                    "type": "manual_review",
                    "message": f"Manual review needed: {issue.get('message')}",
                    "confidence": issue.get("confidence", 0.3),
                    "review_guidance": issue.get("suggestion", ""),
                }
                suggestions.append(suggestion)

        return suggestions

    def _identify_component(self, file_path: str) -> Optional[str]:
        """Identify which component/module a file belongs to."""
        # Simple heuristic based on path structure
        path_parts = Path(file_path).parts

        if len(path_parts) >= 2:
            # Return top-level directory as component
            return path_parts[0]

        return None

    def _calculate_complexity_metrics(self, changes: List[PRChange]) -> Dict[str, Any]:
        """Calculate complexity metrics for the PR."""
        total_additions = sum(change.additions for change in changes)
        total_deletions = sum(change.deletions for change in changes)
        files_changed = len(changes)

        # Language distribution
        languages = {}
        for change in changes:
            lang = change.language
            if lang:
                languages[lang] = languages.get(lang, 0) + 1

        # Change type distribution
        change_types = {}
        for change in changes:
            change_type = change.changes_type
            change_types[change_type] = change_types.get(change_type, 0) + 1

        return {
            "total_additions": total_additions,
            "total_deletions": total_deletions,
            "total_changes": total_additions + total_deletions,
            "files_changed": files_changed,
            "languages": languages,
            "change_types": change_types,
            "churn_ratio": total_deletions / max(total_additions, 1),
        }

    def _calculate_risk_score(
        self,
        issues: List[Dict[str, Any]],
        complexity_metrics: Dict[str, Any],
        changes: List[PRChange],
    ) -> float:
        """Calculate overall risk score for the PR."""
        risk_factors = []

        # Issue-based risk
        critical_issues = len([i for i in issues if i.get("severity") == "critical"])
        warning_issues = len([i for i in issues if i.get("severity") == "warning"])

        issue_risk = (critical_issues * 0.8) + (warning_issues * 0.3)
        risk_factors.append(min(issue_risk / 10, 1.0))  # Normalize to 0-1

        # Complexity-based risk
        files_changed = complexity_metrics["files_changed"]
        total_changes = complexity_metrics["total_changes"]

        complexity_risk = min((files_changed * 0.1) + (total_changes / 1000), 1.0)
        risk_factors.append(complexity_risk)

        # Language diversity risk (more languages = higher risk)
        language_count = len(complexity_metrics["languages"])
        language_risk = min(language_count * 0.15, 1.0)
        risk_factors.append(language_risk)

        # Calculate weighted average
        weights = [0.5, 0.3, 0.2]  # Issue risk gets highest weight
        risk_score = sum(risk * weight for risk, weight in zip(risk_factors, weights))

        return min(risk_score, 1.0)

    def _generate_healing_recommendations(
        self, issues: List[Dict[str, Any]], changes: List[PRChange], risk_score: float
    ) -> List[Dict[str, Any]]:
        """Generate healing recommendations for the PR."""
        recommendations = []

        # Auto-fix recommendations
        fixable_issues = [i for i in issues if i.get("fixable", False)]
        if fixable_issues:
            recommendations.append(
                {
                    "type": "auto_fix",
                    "priority": "high",
                    "message": f"{len(fixable_issues)} issues can be automatically fixed",
                    "action": "Run automated healing on affected files",
                    "files": list(set(i["file_path"] for i in fixable_issues)),
                }
            )

        # High-risk PR recommendations
        if risk_score > self.config.get("risk_threshold", 0.7):
            recommendations.append(
                {
                    "type": "risk_mitigation",
                    "priority": "high",
                    "message": "High-risk PR detected - consider additional review",
                    "action": "Add extra reviewers or staging deployment",
                    "risk_score": risk_score,
                }
            )

        # Testing recommendations
        test_files = [c for c in changes if "test" in c.file_path.lower()]
        code_files = [c for c in changes if "test" not in c.file_path.lower()]

        if code_files and not test_files:
            recommendations.append(
                {
                    "type": "testing",
                    "priority": "medium",
                    "message": "No test files found - consider adding tests",
                    "action": "Add unit tests for changed functionality",
                    "files": [c.file_path for c in code_files],
                }
            )

        return recommendations

    def _create_empty_result(self, pr_number: int) -> PRAnalysisResult:
        """Create an empty analysis result."""
        return PRAnalysisResult(
            pr_number=pr_number,
            risk_score=0.0,
            issues_found=[],
            suggestions=[],
            complexity_metrics={},
            affected_components=[],
            healing_recommendations=[],
            analysis_timestamp=datetime.now().isoformat(),
        )

    def _post_pr_comment(self, pr_number: int, result: PRAnalysisResult) -> None:
        """Post analysis results as a PR comment."""
        try:
            comment_body = self._format_pr_comment(result)

            if self.repo_info["type"] == "github" and self.github_token:
                self._post_github_comment(pr_number, comment_body)
            elif self.repo_info["type"] == "gitlab" and self.gitlab_token:
                self._post_gitlab_comment(pr_number, comment_body)

        except Exception as e:
            self.logger.error(f"Error posting PR comment: {e}")

    def _format_pr_comment(self, result: PRAnalysisResult) -> str:
        """Format analysis results as a markdown comment."""
        comment = "## 🏥 Homeostasis PR Analysis\n\n"

        # Risk score
        risk_emoji = (
            "🟢"
            if result.risk_score < 0.3
            else "🟡" if result.risk_score < 0.7 else "🔴"
        )
        comment += f"**Risk Score:** {risk_emoji} {result.risk_score:.2f}\n\n"

        # Issues summary
        if result.issues_found:
            critical = len(
                [i for i in result.issues_found if i.get("severity") == "critical"]
            )
            warnings = len(
                [i for i in result.issues_found if i.get("severity") == "warning"]
            )

            comment += "### Issues Found\n"
            if critical > 0:
                comment += f"- 🚨 {critical} critical issues\n"
            if warnings > 0:
                comment += f"- ⚠️ {warnings} warnings\n"
            comment += "\n"

        # Healing recommendations
        if result.healing_recommendations:
            comment += "### Healing Recommendations\n"
            for rec in result.healing_recommendations:
                priority_emoji = "🔥" if rec["priority"] == "high" else "📋"
                comment += f"- {priority_emoji} {rec['message']}\n"
            comment += "\n"

        # Complexity metrics
        if result.complexity_metrics:
            metrics = result.complexity_metrics
            comment += "### Complexity Metrics\n"
            comment += f"- Files changed: {metrics.get('files_changed', 0)}\n"
            comment += f"- Total changes: {metrics.get('total_changes', 0)}\n"
            comment += (
                f"- Languages: {', '.join(metrics.get('languages', {}).keys())}\n\n"
            )

        comment += f"*Analysis completed at {result.analysis_timestamp}*"

        return comment

    def _post_github_comment(self, pr_number: int, comment_body: str) -> None:
        """Post comment to GitHub PR."""
        api_url = self.repo_info["api_url"]
        headers = {
            "Authorization": f"token {self.github_token}",
            "Accept": "application/vnd.github.v3+json",
        }

        comment_url = f"{api_url}/issues/{pr_number}/comments"
        data = {"body": comment_body}

        response = requests.post(comment_url, headers=headers, json=data, timeout=30)
        response.raise_for_status()

    def _post_gitlab_comment(self, pr_number: int, comment_body: str) -> None:
        """Post comment to GitLab MR."""
        api_url = self.repo_info["api_url"]
        headers = {
            "Authorization": f"Bearer {self.gitlab_token}",
            "Content-Type": "application/json",
        }

        comment_url = f"{api_url}/merge_requests/{pr_number}/notes"
        data = {"body": comment_body}

        response = requests.post(comment_url, headers=headers, json=data, timeout=30)
        response.raise_for_status()

    def analyze_pr_batch(self, pr_numbers: List[int]) -> List[PRAnalysisResult]:
        """Analyze multiple PRs in batch."""
        results = []

        for pr_number in pr_numbers:
            try:
                result = self.analyze_pull_request(pr_number)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error analyzing PR #{pr_number}: {e}")
                results.append(self._create_empty_result(pr_number))

        return results

    def get_analysis_summary(self, results: List[PRAnalysisResult]) -> Dict[str, Any]:
        """Generate summary statistics for multiple PR analyses."""
        total_prs = len(results)
        total_issues = sum(len(r.issues_found) for r in results)
        high_risk_prs = len([r for r in results if r.risk_score > 0.7])

        avg_risk = sum(r.risk_score for r in results) / max(total_prs, 1)

        return {
            "total_prs_analyzed": total_prs,
            "total_issues_found": total_issues,
            "high_risk_prs": high_risk_prs,
            "average_risk_score": avg_risk,
            "analysis_timestamp": datetime.now().isoformat(),
        }
