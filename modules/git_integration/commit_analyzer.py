"""
Commit message analysis for contextual healing.

This module analyzes Git commit messages and metadata to extract context
that can be used to improve healing decisions and track healing patterns.
"""

import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from modules.monitoring.logger import HomeostasisLogger


@dataclass
class CommitInfo:
    """Information about a Git commit."""

    hash: str
    short_hash: str
    author: str
    author_email: str
    date: datetime
    message: str
    subject: str
    body: str
    files_changed: List[str]
    insertions: int
    deletions: int
    branch: Optional[str] = None


@dataclass
class CommitAnalysis:
    """Analysis results for a commit."""

    commit_info: CommitInfo
    message_type: str
    is_fix_commit: bool
    is_feature_commit: bool
    is_refactor_commit: bool
    is_test_commit: bool
    referenced_issues: List[str]
    breaking_changes: bool
    scope: Optional[str]
    confidence: float
    extracted_context: Dict[str, Any]


class CommitAnalyzer:
    """Analyzes Git commits for contextual information."""

    def __init__(self, repo_path: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize commit analyzer.

        Args:
            repo_path: Path to the Git repository
            config: Configuration dictionary for commit analysis
        """
        self.repo_path = Path(repo_path)
        self.config = config or self._load_default_config()
        self.logger = HomeostasisLogger(__name__)

        # Compile regex patterns for performance
        self._compile_patterns()

        # Cache for commit analyses
        self._analysis_cache = {}

    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration for commit analysis."""
        return {
            "conventional_commits": True,
            "issue_patterns": [
                r"#(\d+)",  # GitHub issues
                r"fixes?\s+#(\d+)",  # Fixes #123
                r"closes?\s+#(\d+)",  # Closes #123
                r"resolves?\s+#(\d+)",  # Resolves #123
                r"JIRA-(\d+)",  # JIRA tickets
                r"[A-Z]+-(\d+)",  # Generic ticket format
            ],
            "breaking_change_patterns": [
                r"BREAKING\s+CHANGE",
                r"BREAKING:",
                r"breaking\s+change",
                r"!:",
            ],
            "commit_types": {
                "fix": ["fix", "hotfix", "patch", "bug"],
                "feat": ["feat", "feature", "add"],
                "refactor": ["refactor", "refact", "restructure"],
                "test": ["test", "tests", "testing"],
                "docs": ["docs", "doc", "documentation"],
                "style": ["style", "format", "formatting"],
                "chore": ["chore", "maintenance", "maint"],
                "perf": ["perf", "performance", "optimize"],
                "ci": ["ci", "build", "deploy"],
                "revert": ["revert", "rollback", "undo"],
            },
            "scope_patterns": [
                r"^\w+\(([^)]+)\):",  # feat(scope): message
                r"^\w+\s+\(([^)]+)\):",  # feat (scope): message
                r"^\[([^\]]+)\]",  # [scope] message
            ],
            "max_commit_history": 1000,
            "cache_enabled": True,
        }

    def _compile_patterns(self) -> None:
        """Compile regex patterns for better performance."""
        self._issue_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.config["issue_patterns"]
        ]

        self._breaking_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.config["breaking_change_patterns"]
        ]

        self._scope_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.config["scope_patterns"]
        ]

    def analyze_commit(self, commit_hash: str) -> CommitAnalysis:
        """
        Analyze a specific commit.

        Args:
            commit_hash: Git commit hash

        Returns:
            Commit analysis results
        """
        # Check cache first
        if (
            self.config.get("cache_enabled", True)
            and commit_hash in self._analysis_cache
        ):
            return self._analysis_cache[commit_hash]

        try:
            # Get commit information
            commit_info = self._get_commit_info(commit_hash)
            if not commit_info:
                raise Exception(f"Could not retrieve commit info for {commit_hash}")

            # Analyze commit message
            analysis = self._analyze_commit_message(commit_info)

            # Cache the result
            if self.config.get("cache_enabled", True):
                self._analysis_cache[commit_hash] = analysis

            return analysis

        except Exception as e:
            self.logger.error(f"Error analyzing commit {commit_hash}: {e}")
            return self._create_empty_analysis(commit_hash)

    def _get_commit_info(self, commit_hash: str) -> Optional[CommitInfo]:
        """Get detailed information about a commit."""
        try:
            # Get commit details
            format_string = "%H%n%h%n%an%n%ae%n%ci%n%s%n%b"
            result = subprocess.run(
                ["git", "show", "--format=" + format_string, "--no-patch", commit_hash],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                return None

            lines = result.stdout.strip().split("\n")
            if len(lines) < 6:
                return None

            full_hash = lines[0]
            short_hash = lines[1]
            author = lines[2]
            author_email = lines[3]
            date_str = lines[4]
            subject = lines[5]
            body = "\n".join(lines[6:]) if len(lines) > 6 else ""

            # Parse date
            commit_date = datetime.fromisoformat(date_str.replace(" ", "T", 1))

            # Get file changes
            files_changed = self._get_commit_files(commit_hash)

            # Get insertions/deletions
            insertions, deletions = self._get_commit_stats(commit_hash)

            return CommitInfo(
                hash=full_hash,
                short_hash=short_hash,
                author=author,
                author_email=author_email,
                date=commit_date,
                message=f"{subject}\n{body}".strip(),
                subject=subject,
                body=body,
                files_changed=files_changed,
                insertions=insertions,
                deletions=deletions,
            )

        except Exception as e:
            self.logger.error(f"Error getting commit info: {e}")
            return None

    def _get_commit_files(self, commit_hash: str) -> List[str]:
        """Get list of files changed in a commit."""
        try:
            result = subprocess.run(
                ["git", "show", "--name-only", "--format=", commit_hash],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                files = [
                    f.strip() for f in result.stdout.strip().split("\n") if f.strip()
                ]
                return files

            return []

        except Exception:
            return []

    def _get_commit_stats(self, commit_hash: str) -> Tuple[int, int]:
        """Get insertion/deletion statistics for a commit."""
        try:
            result = subprocess.run(
                ["git", "show", "--shortstat", "--format=", commit_hash],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                stats_line = result.stdout.strip()

                # Parse stats (e.g., "2 files changed, 10 insertions(+), 5 deletions(-)")
                insertions = 0
                deletions = 0

                insertion_match = re.search(r"(\d+) insertions?\(\+\)", stats_line)
                if insertion_match:
                    insertions = int(insertion_match.group(1))

                deletion_match = re.search(r"(\d+) deletions?\(-\)", stats_line)
                if deletion_match:
                    deletions = int(deletion_match.group(1))

                return insertions, deletions

            return 0, 0

        except Exception:
            return 0, 0

    def _analyze_commit_message(self, commit_info: CommitInfo) -> CommitAnalysis:
        """Analyze commit message to extract contextual information."""
        message = commit_info.message
        subject = commit_info.subject

        # Determine commit type
        message_type = self._classify_commit_type(subject)

        # Check for specific commit types
        is_fix_commit = self._is_fix_commit(subject, message)
        is_feature_commit = self._is_feature_commit(subject, message)
        is_refactor_commit = self._is_refactor_commit(subject, message)
        is_test_commit = self._is_test_commit(
            subject, message, commit_info.files_changed
        )

        # Extract referenced issues
        referenced_issues = self._extract_referenced_issues(message)

        # Check for breaking changes
        breaking_changes = self._has_breaking_changes(message)

        # Extract scope
        scope = self._extract_scope(subject)

        # Calculate confidence based on message quality
        confidence = self._calculate_message_confidence(commit_info)

        # Extract additional context
        extracted_context = self._extract_additional_context(commit_info)

        return CommitAnalysis(
            commit_info=commit_info,
            message_type=message_type,
            is_fix_commit=is_fix_commit,
            is_feature_commit=is_feature_commit,
            is_refactor_commit=is_refactor_commit,
            is_test_commit=is_test_commit,
            referenced_issues=referenced_issues,
            breaking_changes=breaking_changes,
            scope=scope,
            confidence=confidence,
            extracted_context=extracted_context,
        )

    def _classify_commit_type(self, subject: str) -> str:
        """Classify commit type based on subject line."""
        subject_lower = subject.lower()

        for commit_type, keywords in self.config["commit_types"].items():
            for keyword in keywords:
                if subject_lower.startswith(keyword) or f"({keyword})" in subject_lower:
                    return commit_type

        # Check conventional commit format
        if self.config.get("conventional_commits", True):
            conventional_match = re.match(r"^(\w+)(?:\([^)]+\))?:", subject)
            if conventional_match:
                type_name = conventional_match.group(1).lower()
                if type_name in self.config["commit_types"]:
                    return type_name

        return "unknown"

    def _is_fix_commit(self, subject: str, message: str) -> bool:
        """Check if commit is a fix commit."""
        fix_keywords = ["fix", "bug", "issue", "problem", "error", "crash", "broken"]
        text = (subject + " " + message).lower()

        return any(keyword in text for keyword in fix_keywords)

    def _is_feature_commit(self, subject: str, message: str) -> bool:
        """Check if commit introduces a new feature."""
        feature_keywords = ["feat", "feature", "add", "new", "implement", "introduce"]
        text = (subject + " " + message).lower()

        return any(keyword in text for keyword in feature_keywords)

    def _is_refactor_commit(self, subject: str, message: str) -> bool:
        """Check if commit is a refactoring commit."""
        refactor_keywords = [
            "refactor",
            "restructure",
            "reorganize",
            "cleanup",
            "improve",
        ]
        text = (subject + " " + message).lower()

        return any(keyword in text for keyword in refactor_keywords)

    def _is_test_commit(
        self, subject: str, message: str, files_changed: List[str]
    ) -> bool:
        """Check if commit is test-related."""
        test_keywords = ["test", "testing", "spec", "unit", "integration"]
        text = (subject + " " + message).lower()

        # Check message content
        if any(keyword in text for keyword in test_keywords):
            return True

        # Check if only test files were changed
        test_files = [
            f
            for f in files_changed
            if "test" in f.lower()
            or f.endswith((".test.js", ".spec.js", "_test.py", "test_*.py"))
        ]
        return len(test_files) == len(files_changed) and len(test_files) > 0

    def _extract_referenced_issues(self, message: str) -> List[str]:
        """Extract referenced issues from commit message."""
        issues = []

        for pattern in self._issue_patterns:
            matches = pattern.findall(message)
            issues.extend(matches)

        return list(set(issues))  # Remove duplicates

    def _has_breaking_changes(self, message: str) -> bool:
        """Check if commit introduces breaking changes."""
        for pattern in self._breaking_patterns:
            if pattern.search(message):
                return True

        return False

    def _extract_scope(self, subject: str) -> Optional[str]:
        """Extract scope from commit subject."""
        for pattern in self._scope_patterns:
            match = pattern.search(subject)
            if match:
                return match.group(1)

        return None

    def _calculate_message_confidence(self, commit_info: CommitInfo) -> float:
        """Calculate confidence score based on commit message quality."""
        confidence = 0.0

        # Subject line quality
        subject = commit_info.subject
        if len(subject) >= 10:
            confidence += 0.2
        if len(subject) <= 72:  # Good practice
            confidence += 0.1
        if subject[0].isupper():
            confidence += 0.1

        # Has body
        if commit_info.body.strip():
            confidence += 0.3

        # References issues
        if self._extract_referenced_issues(commit_info.message):
            confidence += 0.2

        # Conventional commit format
        if re.match(r"^\w+(?:\([^)]+\))?:", subject):
            confidence += 0.2

        return min(confidence, 1.0)

    def _extract_additional_context(self, commit_info: CommitInfo) -> Dict[str, Any]:
        """Extract additional contextual information."""
        context = {}

        # File types changed
        file_extensions = set()
        for file_path in commit_info.files_changed:
            ext = Path(file_path).suffix.lower()
            if ext:
                file_extensions.add(ext)

        context["file_extensions"] = list(file_extensions)

        # Directories affected
        directories = set()
        for file_path in commit_info.files_changed:
            directory = str(Path(file_path).parent)
            if directory != ".":
                directories.add(directory)

        context["directories"] = list(directories)

        # Commit size
        total_changes = commit_info.insertions + commit_info.deletions
        if total_changes < 10:
            context["size"] = "small"
        elif total_changes < 100:
            context["size"] = "medium"
        else:
            context["size"] = "large"

        # Time of day
        hour = commit_info.date.hour
        if 6 <= hour < 12:
            context["time_of_day"] = "morning"
        elif 12 <= hour < 18:
            context["time_of_day"] = "afternoon"
        elif 18 <= hour < 22:
            context["time_of_day"] = "evening"
        else:
            context["time_of_day"] = "night"

        return context

    def _create_empty_analysis(self, commit_hash: str) -> CommitAnalysis:
        """Create empty analysis for error cases."""
        empty_commit = CommitInfo(
            hash=commit_hash,
            short_hash=commit_hash[:8],
            author="unknown",
            author_email="",
            date=datetime.now(),
            message="",
            subject="",
            body="",
            files_changed=[],
            insertions=0,
            deletions=0,
        )

        return CommitAnalysis(
            commit_info=empty_commit,
            message_type="unknown",
            is_fix_commit=False,
            is_feature_commit=False,
            is_refactor_commit=False,
            is_test_commit=False,
            referenced_issues=[],
            breaking_changes=False,
            scope=None,
            confidence=0.0,
            extracted_context={},
        )

    def analyze_commit_history(
        self, limit: Optional[int] = None, since: Optional[datetime] = None
    ) -> List[CommitAnalysis]:
        """
        Analyze commit history.

        Args:
            limit: Maximum number of commits to analyze
            since: Only analyze commits since this date

        Returns:
            List of commit analyses
        """
        try:
            # Build git log command
            cmd = ["git", "log", "--format=%H"]

            if limit:
                cmd.extend(["-n", str(limit)])

            if since:
                cmd.extend(["--since", since.isoformat()])

            # Get commit hashes
            result = subprocess.run(
                cmd, cwd=self.repo_path, capture_output=True, text=True
            )

            if result.returncode != 0:
                self.logger.error(f"Failed to get commit history: {result.stderr}")
                return []

            commit_hashes = result.stdout.strip().split("\n")
            if not commit_hashes or commit_hashes == [""]:
                return []

            # Analyze each commit
            analyses = []
            for commit_hash in commit_hashes[
                : self.config.get("max_commit_history", 1000)
            ]:
                analysis = self.analyze_commit(commit_hash)
                analyses.append(analysis)

            return analyses

        except Exception as e:
            self.logger.error(f"Error analyzing commit history: {e}")
            return []

    def get_fix_patterns(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Analyze patterns in fix commits to understand common issues.

        Args:
            limit: Maximum number of commits to analyze

        Returns:
            Dictionary with fix patterns and statistics
        """
        # Get recent commit history
        analyses = self.analyze_commit_history(limit)

        # Filter fix commits
        fix_commits = [a for a in analyses if a.is_fix_commit]

        if not fix_commits:
            return {"total_fixes": 0, "patterns": {}}

        # Analyze patterns
        issue_types = defaultdict(int)
        affected_files = defaultdict(int)
        fix_scopes = defaultdict(int)
        referenced_issues = defaultdict(int)

        for fix_commit in fix_commits:
            # Issue types from message
            message_lower = fix_commit.commit_info.message.lower()

            # Common issue keywords
            if "null" in message_lower or "none" in message_lower:
                issue_types["null_pointer"] += 1
            if "index" in message_lower and "error" in message_lower:
                issue_types["index_error"] += 1
            if "type" in message_lower and "error" in message_lower:
                issue_types["type_error"] += 1
            if "import" in message_lower or "module" in message_lower:
                issue_types["import_error"] += 1
            if "syntax" in message_lower:
                issue_types["syntax_error"] += 1

            # Affected file types
            for file_path in fix_commit.commit_info.files_changed:
                ext = Path(file_path).suffix.lower()
                if ext:
                    affected_files[ext] += 1

            # Fix scopes
            if fix_commit.scope:
                fix_scopes[fix_commit.scope] += 1

            # Referenced issues
            for issue in fix_commit.referenced_issues:
                referenced_issues[issue] += 1

        return {
            "total_fixes": len(fix_commits),
            "total_commits": len(analyses),
            "fix_ratio": len(fix_commits) / len(analyses) if analyses else 0,
            "patterns": {
                "issue_types": dict(issue_types),
                "affected_files": dict(affected_files),
                "fix_scopes": dict(fix_scopes),
                "referenced_issues": dict(referenced_issues),
            },
            "analysis_date": datetime.now().isoformat(),
        }

    def get_healing_context_for_error(
        self, error_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get healing context by analyzing related commits.

        Args:
            error_info: Information about the current error

        Returns:
            Contextual information for healing decisions
        """
        try:
            # Get recent commits
            analyses = self.analyze_commit_history(limit=100)

            context = {
                "similar_fixes": [],
                "related_files": [],
                "common_patterns": [],
                "suggested_scope": None,
                "confidence_boost": 0.0,
            }

            error_file = error_info.get("file_path", "")
            error_message = error_info.get("message", "").lower()

            # Find similar fixes
            for analysis in analyses:
                if not analysis.is_fix_commit:
                    continue

                # Check if fix affected same file
                if error_file in analysis.commit_info.files_changed:
                    context["similar_fixes"].append(
                        {
                            "commit_hash": analysis.commit_info.short_hash,
                            "message": analysis.commit_info.subject,
                            "confidence": analysis.confidence,
                            "scope": analysis.scope,
                        }
                    )

                # Check for similar error messages
                commit_message = analysis.commit_info.message.lower()
                if any(word in commit_message for word in error_message.split()[:3]):
                    context["common_patterns"].append(
                        {
                            "pattern": analysis.commit_info.subject,
                            "files": analysis.commit_info.files_changed,
                        }
                    )

            # Suggest scope based on file path
            if error_file:
                directory = str(Path(error_file).parent)
                context["suggested_scope"] = (
                    directory.split("/")[-1] if directory != "." else None
                )

            # Calculate confidence boost based on similar fixes
            if context["similar_fixes"]:
                avg_confidence = sum(
                    fix["confidence"] for fix in context["similar_fixes"]
                ) / len(context["similar_fixes"])
                context["confidence_boost"] = min(avg_confidence * 0.3, 0.5)

            return context

        except Exception as e:
            self.logger.error(f"Error getting healing context: {e}")
            return {
                "similar_fixes": [],
                "related_files": [],
                "common_patterns": [],
                "suggested_scope": None,
                "confidence_boost": 0.0,
            }

    def generate_healing_commit_message(self, healing_info: Dict[str, Any]) -> str:
        """
        Generate a commit message for healing changes.

        Args:
            healing_info: Information about the healing operation

        Returns:
            Generated commit message
        """
        issue_type = healing_info.get("issue_type", "error")
        file_path = healing_info.get("file_path", "")
        rule_id = healing_info.get("rule_id", "")

        # Determine scope
        scope = None
        if file_path:
            directory = str(Path(file_path).parent)
            if directory != ".":
                scope = directory.split("/")[-1]

        # Generate subject line
        subject = "fix"
        if scope:
            subject += f"({scope})"

        subject += f": {issue_type.replace('_', ' ')}"

        if file_path:
            filename = Path(file_path).name
            subject += f" in {filename}"

        # Generate body
        body_lines = []
        body_lines.append("Automated healing by Homeostasis")
        body_lines.append("")

        if rule_id:
            body_lines.append(f"Rule ID: {rule_id}")

        if healing_info.get("confidence"):
            body_lines.append(f"Confidence: {healing_info['confidence']:.2f}")

        if healing_info.get("description"):
            body_lines.append("")
            body_lines.append(healing_info["description"])

        body_lines.append("")
        body_lines.append("🏥 Generated with Homeostasis")

        return subject + "\n\n" + "\n".join(body_lines)
