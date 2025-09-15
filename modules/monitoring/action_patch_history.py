"""
Advanced Action and Patch History Logging for LLM Integration.

This module provides comprehensive logging capabilities for all healing actions
and patch operations, including detailed file changes, commit diffs, LLM
prompt/response pairs, and complete version history tracking.
"""

import difflib
import hashlib
import json
import logging
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import git
from git import Repo

from modules.monitoring.logger import MonitoringLogger
from modules.security.audit import get_audit_logger, log_event

logger = logging.getLogger(__name__)


class ActionPatchHistoryLogger:
    """
    Comprehensive logger for all healing actions and patch operations.

    Features:
    - Complete file change tracking with diffs
    - LLM prompt/response logging with metadata
    - Git integration for commit history
    - Version history with rollback capabilities
    - Performance and metrics tracking
    - Security and compliance logging
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the action patch history logger.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.monitoring_logger = MonitoringLogger("action_patch_history")
        self.audit_logger = get_audit_logger()

        # Extract configuration
        self.enabled = self.config.get("enabled", True)
        self.max_history_days = self.config.get("max_history_days", 30)
        self.max_file_size_mb = self.config.get("max_file_size_mb", 10)
        self.include_llm_responses = self.config.get("include_llm_responses", True)
        self.include_file_diffs = self.config.get("include_file_diffs", True)
        self.include_git_history = self.config.get("include_git_history", True)
        self.compress_old_entries = self.config.get("compress_old_entries", True)

        # Setup storage directories
        self.base_dir = Path(self.config.get("storage_dir", "logs/action_history"))
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.patches_dir = self.base_dir / "patches"
        self.patches_dir.mkdir(exist_ok=True)

        self.diffs_dir = self.base_dir / "diffs"
        self.diffs_dir.mkdir(exist_ok=True)

        self.llm_logs_dir = self.base_dir / "llm_interactions"
        self.llm_logs_dir.mkdir(exist_ok=True)

        self.versions_dir = self.base_dir / "versions"
        self.versions_dir.mkdir(exist_ok=True)

        # Initialize git repository if needed
        self.git_repo: Optional[Repo] = None
        if self.include_git_history:
            self._init_git_repo()

    def _init_git_repo(self) -> None:
        """Initialize or connect to the git repository."""
        try:
            # Try to find existing git repo
            current_dir = Path.cwd()
            while current_dir != current_dir.parent:
                if (current_dir / ".git").exists():
                    self.git_repo = git.Repo(current_dir)
                    self.monitoring_logger.info(
                        f"Connected to git repository at {current_dir}"
                    )
                    return
                current_dir = current_dir.parent

            # If no git repo found, initialize one in the base directory
            git_dir = self.base_dir / ".git"
            if not git_dir.exists():
                self.git_repo = git.Repo.init(self.base_dir)
                self.monitoring_logger.info(
                    f"Initialized git repository at {self.base_dir}"
                )
            else:
                self.git_repo = git.Repo(self.base_dir)

        except Exception as e:
            self.monitoring_logger.warning(f"Failed to initialize git repository: {e}")
            self.git_repo = None

    def log_action_start(
        self,
        session_id: str,
        action_type: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Log the start of a healing action.

        Args:
            session_id: Healing session ID
            action_type: Type of action (detection, analysis, generation, etc.)
            context: Additional context about the action

        Returns:
            str: Action ID
        """
        if not self.enabled:
            return str(uuid.uuid4())

        action_id = str(uuid.uuid4())
        timestamp = datetime.utcnow()

        action_record = {
            "action_id": action_id,
            "session_id": session_id,
            "action_type": action_type,
            "status": "started",
            "start_time": timestamp.isoformat(),
            "context": context or {},
            "performance_metrics": {},
            "resources_used": [],
            "security_flags": [],
        }

        # Store action record
        action_file = self.base_dir / f"action_{action_id}.json"
        with open(action_file, "w") as f:
            json.dump(action_record, f, indent=2)

        # Log to audit system
        log_event(
            event_type="healing_action_started",
            details={
                "action_id": action_id,
                "session_id": session_id,
                "action_type": action_type,
                **(context or {}),
            },
        )

        self.monitoring_logger.info(
            f"Started healing action {action_type}",
            action_id=action_id,
            session_id=session_id,
        )

        return action_id

    def log_llm_interaction(
        self,
        action_id: str,
        provider: str,
        model: str,
        prompt: str,
        response: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Log an LLM interaction with full prompt/response details.

        Args:
            action_id: Action ID this interaction belongs to
            provider: LLM provider (openai, anthropic, etc.)
            model: Model name
            prompt: The prompt sent to the LLM
            response: The response from the LLM
            metadata: Additional metadata (tokens, cost, etc.)

        Returns:
            str: Interaction ID
        """
        if not self.enabled or not self.include_llm_responses:
            return str(uuid.uuid4())

        interaction_id = str(uuid.uuid4())
        timestamp = datetime.utcnow()

        # Create interaction record
        interaction_record = {
            "interaction_id": interaction_id,
            "action_id": action_id,
            "timestamp": timestamp.isoformat(),
            "provider": provider,
            "model": model,
            "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest(),
            "response_hash": hashlib.sha256(response.encode()).hexdigest(),
            "metadata": metadata or {},
        }

        # Store prompt and response separately for security
        prompt_file = self.llm_logs_dir / f"prompt_{interaction_id}.txt"
        response_file = self.llm_logs_dir / f"response_{interaction_id}.txt"

        with open(prompt_file, "w") as f:
            f.write(prompt)

        with open(response_file, "w") as f:
            f.write(response)

        # Store interaction metadata
        interaction_file = self.llm_logs_dir / f"interaction_{interaction_id}.json"
        with open(interaction_file, "w") as f:
            json.dump(interaction_record, f, indent=2)

        # Update action record
        self._update_action_record(
            action_id, {"llm_interactions": [interaction_id]}, append_lists=True
        )

        # Log to monitoring
        self.monitoring_logger.info(
            "LLM interaction logged",
            action_id=action_id,
            interaction_id=interaction_id,
            provider=provider,
            model=model,
            prompt_length=len(prompt),
            response_length=len(response),
        )

        # Security logging for LLM usage
        log_event(
            event_type="llm_interaction_logged",
            details={
                "action_id": action_id,
                "interaction_id": interaction_id,
                "provider": provider,
                "model": model,
                "prompt_length": len(prompt),
                "response_length": len(response),
                **(metadata or {}),
            },
        )

        return interaction_id

    def log_file_change(
        self,
        action_id: str,
        file_path: str,
        change_type: str,
        old_content: Optional[str] = None,
        new_content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Log a file change with complete diff tracking.

        Args:
            action_id: Action ID this change belongs to
            file_path: Path to the file being changed
            change_type: Type of change (create, modify, delete)
            old_content: Original file content
            new_content: New file content
            metadata: Additional metadata about the change

        Returns:
            str: Change ID
        """
        if not self.enabled:
            return str(uuid.uuid4())

        change_id = str(uuid.uuid4())
        timestamp = datetime.utcnow()

        # Create change record
        change_record: Dict[str, Any] = {
            "change_id": change_id,
            "action_id": action_id,
            "timestamp": timestamp.isoformat(),
            "file_path": file_path,
            "change_type": change_type,
            "metadata": metadata or {},
        }

        # Generate and store diff if both contents are provided
        if (
            self.include_file_diffs
            and old_content is not None
            and new_content is not None
        ):
            diff = list(
                difflib.unified_diff(
                    old_content.splitlines(keepends=True),
                    new_content.splitlines(keepends=True),
                    fromfile=f"{file_path} (before)",
                    tofile=f"{file_path} (after)",
                    lineterm="",
                )
            )

            diff_content = "".join(diff)
            diff_file = self.diffs_dir / f"diff_{change_id}.patch"
            with open(diff_file, "w") as f:
                f.write(diff_content)

            change_record["diff_file"] = str(diff_file)
            change_record["lines_added"] = len(
                [line for line in diff if line.startswith("+")]
            )
            change_record["lines_removed"] = len(
                [line for line in diff if line.startswith("-")]
            )

        # Store file versions if content is provided
        if old_content is not None:
            old_version_file = self.versions_dir / f"{change_id}_before.txt"
            with open(old_version_file, "w") as f:
                f.write(old_content)
            change_record["old_version_file"] = str(old_version_file)

        if new_content is not None:
            new_version_file = self.versions_dir / f"{change_id}_after.txt"
            with open(new_version_file, "w") as f:
                f.write(new_content)
            change_record["new_version_file"] = str(new_version_file)

        # Calculate file statistics
        if new_content is not None:
            change_record["file_size_bytes"] = len(new_content.encode("utf-8"))
            change_record["line_count"] = len(new_content.splitlines())

        # Store change record
        change_file = self.base_dir / f"change_{change_id}.json"
        with open(change_file, "w") as f:
            json.dump(change_record, f, indent=2)

        # Update action record
        self._update_action_record(
            action_id, {"file_changes": [change_id]}, append_lists=True
        )

        # Git commit if enabled
        if self.git_repo and new_content is not None:
            try:
                # Write file to git repo
                git_file_path = self.base_dir / "tracked_files" / file_path
                git_file_path.parent.mkdir(parents=True, exist_ok=True)

                with open(git_file_path, "w") as f:
                    f.write(new_content)

                # Add and commit
                self.git_repo.index.add([str(git_file_path)])
                commit_message = f"Action {action_id}: {change_type} {file_path}"
                commit = self.git_repo.index.commit(commit_message)

                change_record["git_commit"] = commit.hexsha

                # Update change record with git info
                with open(change_file, "w") as f:
                    json.dump(change_record, f, indent=2)

            except Exception as e:
                self.monitoring_logger.warning(f"Failed to commit to git: {e}")

        # Log to monitoring
        self.monitoring_logger.info(
            f"File change logged: {change_type} {file_path}",
            action_id=action_id,
            change_id=change_id,
            file_path=file_path,
            change_type=change_type,
        )

        return change_id

    def log_patch_application(
        self,
        action_id: str,
        patch_id: str,
        target_file: str,
        patch_content: str,
        application_result: Dict[str, Any],
    ) -> str:
        """
        Log a patch application with full details.

        Args:
            action_id: Action ID this patch belongs to
            patch_id: Unique patch identifier
            target_file: File being patched
            patch_content: The actual patch content
            application_result: Result of applying the patch

        Returns:
            str: Patch application ID
        """
        if not self.enabled:
            return str(uuid.uuid4())

        patch_app_id = str(uuid.uuid4())
        timestamp = datetime.utcnow()

        # Create patch application record
        patch_record = {
            "patch_application_id": patch_app_id,
            "action_id": action_id,
            "patch_id": patch_id,
            "timestamp": timestamp.isoformat(),
            "target_file": target_file,
            "application_result": application_result,
            "patch_hash": hashlib.sha256(patch_content.encode()).hexdigest(),
        }

        # Store patch content
        patch_file = self.patches_dir / f"patch_{patch_app_id}.patch"
        with open(patch_file, "w") as f:
            f.write(patch_content)

        patch_record["patch_file"] = str(patch_file)

        # Store patch application record
        patch_app_file = self.base_dir / f"patch_app_{patch_app_id}.json"
        with open(patch_app_file, "w") as f:
            json.dump(patch_record, f, indent=2)

        # Update action record
        self._update_action_record(
            action_id, {"patch_applications": [patch_app_id]}, append_lists=True
        )

        # Log to monitoring
        self.monitoring_logger.info(
            "Patch application logged",
            action_id=action_id,
            patch_id=patch_id,
            patch_app_id=patch_app_id,
            target_file=target_file,
            success=application_result.get("success", False),
        )

        return patch_app_id

    def log_action_completion(
        self,
        action_id: str,
        status: str,
        performance_metrics: Optional[Dict[str, Any]] = None,
        error_details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log the completion of a healing action.

        Args:
            action_id: Action ID
            status: Final status (success, failure, partial)
            performance_metrics: Performance data
            error_details: Error information if failed
        """
        if not self.enabled:
            return

        timestamp = datetime.utcnow()

        # Update action record
        updates = {
            "status": status,
            "end_time": timestamp.isoformat(),
            "performance_metrics": performance_metrics or {},
        }

        if error_details:
            updates["error_details"] = error_details

        self._update_action_record(action_id, updates)

        # Calculate duration
        action_record = self._get_action_record(action_id)
        if action_record and "start_time" in action_record:
            start_time = datetime.fromisoformat(action_record["start_time"])
            duration = (timestamp - start_time).total_seconds()
            self._update_action_record(action_id, {"duration_seconds": duration})

        # Log to audit system
        log_event(
            event_type="healing_action_completed",
            details={
                "action_id": action_id,
                "status": status,
                "duration_seconds": duration if "duration" in locals() else None,
                **(performance_metrics or {}),
                **(error_details or {}),
            },
        )

        self.monitoring_logger.info(
            f"Healing action completed with status: {status}",
            action_id=action_id,
            status=status,
        )

    def get_action_history(
        self,
        session_id: Optional[str] = None,
        action_type: Optional[str] = None,
        days_back: int = 7,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve action history with optional filtering.

        Args:
            session_id: Filter by session ID
            action_type: Filter by action type
            days_back: Number of days to look back

        Returns:
            List[Dict[str, Any]]: List of action records
        """
        if not self.enabled:
            return []

        cutoff_date = datetime.utcnow() - timedelta(days=days_back)
        actions = []

        # Find all action files
        for action_file in self.base_dir.glob("action_*.json"):
            try:
                with open(action_file, "r") as f:
                    action_record = json.load(f)

                # Check date filter
                start_time = datetime.fromisoformat(action_record.get("start_time", ""))
                if start_time < cutoff_date:
                    continue

                # Check session filter
                if session_id and action_record.get("session_id") != session_id:
                    continue

                # Check action type filter
                if action_type and action_record.get("action_type") != action_type:
                    continue

                actions.append(action_record)

            except Exception as e:
                self.monitoring_logger.warning(
                    f"Failed to read action file {action_file}: {e}"
                )

        # Sort by start time
        actions.sort(key=lambda x: x.get("start_time", ""), reverse=True)

        return actions

    def get_file_change_history(
        self, file_path: Optional[str] = None, days_back: int = 7
    ) -> List[Dict[str, Any]]:
        """
        Retrieve file change history.

        Args:
            file_path: Filter by specific file path
            days_back: Number of days to look back

        Returns:
            List[Dict[str, Any]]: List of change records
        """
        if not self.enabled:
            return []

        cutoff_date = datetime.utcnow() - timedelta(days=days_back)
        changes = []

        # Find all change files
        for change_file in self.base_dir.glob("change_*.json"):
            try:
                with open(change_file, "r") as f:
                    change_record = json.load(f)

                # Check date filter
                timestamp = datetime.fromisoformat(change_record.get("timestamp", ""))
                if timestamp < cutoff_date:
                    continue

                # Check file path filter
                if file_path and change_record.get("file_path") != file_path:
                    continue

                changes.append(change_record)

            except Exception as e:
                self.monitoring_logger.warning(
                    f"Failed to read change file {change_file}: {e}"
                )

        # Sort by timestamp
        changes.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

        return changes

    def get_llm_interaction_history(
        self,
        action_id: Optional[str] = None,
        provider: Optional[str] = None,
        days_back: int = 7,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve LLM interaction history.

        Args:
            action_id: Filter by action ID
            provider: Filter by LLM provider
            days_back: Number of days to look back

        Returns:
            List[Dict[str, Any]]: List of interaction records
        """
        if not self.enabled or not self.include_llm_responses:
            return []

        cutoff_date = datetime.utcnow() - timedelta(days=days_back)
        interactions = []

        # Find all interaction files
        for interaction_file in self.llm_logs_dir.glob("interaction_*.json"):
            try:
                with open(interaction_file, "r") as f:
                    interaction_record = json.load(f)

                # Check date filter
                timestamp = datetime.fromisoformat(
                    interaction_record.get("timestamp", "")
                )
                if timestamp < cutoff_date:
                    continue

                # Check action ID filter
                if action_id and interaction_record.get("action_id") != action_id:
                    continue

                # Check provider filter
                if provider and interaction_record.get("provider") != provider:
                    continue

                interactions.append(interaction_record)

            except Exception as e:
                self.monitoring_logger.warning(
                    f"Failed to read interaction file {interaction_file}: {e}"
                )

        # Sort by timestamp
        interactions.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

        return interactions

    def get_patch_application_history(self, days_back: int = 7) -> List[Dict[str, Any]]:
        """
        Retrieve patch application history.

        Args:
            days_back: Number of days to look back

        Returns:
            List[Dict[str, Any]]: List of patch application records
        """
        if not self.enabled:
            return []

        cutoff_date = datetime.utcnow() - timedelta(days=days_back)
        patch_apps = []

        # Find all patch application files
        for patch_app_file in self.base_dir.glob("patch_app_*.json"):
            try:
                with open(patch_app_file, "r") as f:
                    patch_record = json.load(f)

                # Check date filter
                timestamp = datetime.fromisoformat(patch_record.get("timestamp", ""))
                if timestamp < cutoff_date:
                    continue

                patch_apps.append(patch_record)

            except Exception as e:
                self.monitoring_logger.warning(
                    f"Failed to read patch application file {patch_app_file}: {e}"
                )

        # Sort by timestamp
        patch_apps.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

        return patch_apps

    def cleanup_old_entries(self) -> int:
        """
        Clean up old history entries based on configured retention.

        Returns:
            int: Number of entries cleaned up
        """
        if not self.enabled:
            return 0

        cutoff_date = datetime.utcnow() - timedelta(days=self.max_history_days)
        cleaned_count = 0

        # Clean up action files
        for action_file in self.base_dir.glob("action_*.json"):
            try:
                with open(action_file, "r") as f:
                    action_record = json.load(f)

                start_time = datetime.fromisoformat(action_record.get("start_time", ""))
                if start_time < cutoff_date:
                    action_file.unlink()
                    cleaned_count += 1

            except Exception as e:
                self.monitoring_logger.warning(
                    f"Failed to process action file {action_file}: {e}"
                )

        # Clean up change files
        for change_file in self.base_dir.glob("change_*.json"):
            try:
                with open(change_file, "r") as f:
                    change_record = json.load(f)

                timestamp = datetime.fromisoformat(change_record.get("timestamp", ""))
                if timestamp < cutoff_date:
                    # Also clean up associated files
                    change_id = change_record.get("change_id")
                    if change_id:
                        # Clean up diff file
                        diff_file = self.diffs_dir / f"diff_{change_id}.patch"
                        if diff_file.exists():
                            diff_file.unlink()

                        # Clean up version files
                        old_version_file = self.versions_dir / f"{change_id}_before.txt"
                        new_version_file = self.versions_dir / f"{change_id}_after.txt"
                        if old_version_file.exists():
                            old_version_file.unlink()
                        if new_version_file.exists():
                            new_version_file.unlink()

                    change_file.unlink()
                    cleaned_count += 1

            except Exception as e:
                self.monitoring_logger.warning(
                    f"Failed to process change file {change_file}: {e}"
                )

        # Clean up LLM interaction files
        for interaction_file in self.llm_logs_dir.glob("interaction_*.json"):
            try:
                with open(interaction_file, "r") as f:
                    interaction_record = json.load(f)

                timestamp = datetime.fromisoformat(
                    interaction_record.get("timestamp", "")
                )
                if timestamp < cutoff_date:
                    interaction_id = interaction_record.get("interaction_id")
                    if interaction_id:
                        # Clean up prompt and response files
                        prompt_file = self.llm_logs_dir / f"prompt_{interaction_id}.txt"
                        response_file = (
                            self.llm_logs_dir / f"response_{interaction_id}.txt"
                        )
                        if prompt_file.exists():
                            prompt_file.unlink()
                        if response_file.exists():
                            response_file.unlink()

                    interaction_file.unlink()
                    cleaned_count += 1

            except Exception as e:
                self.monitoring_logger.warning(
                    f"Failed to process interaction file {interaction_file}: {e}"
                )

        # Clean up patch application files
        for patch_app_file in self.base_dir.glob("patch_app_*.json"):
            try:
                with open(patch_app_file, "r") as f:
                    patch_record = json.load(f)

                timestamp = datetime.fromisoformat(patch_record.get("timestamp", ""))
                if timestamp < cutoff_date:
                    patch_app_id = patch_record.get("patch_application_id")
                    if patch_app_id:
                        # Clean up patch file
                        patch_file = self.patches_dir / f"patch_{patch_app_id}.patch"
                        if patch_file.exists():
                            patch_file.unlink()

                    patch_app_file.unlink()
                    cleaned_count += 1

            except Exception as e:
                self.monitoring_logger.warning(
                    f"Failed to process patch app file {patch_app_file}: {e}"
                )

        if cleaned_count > 0:
            self.monitoring_logger.info(
                f"Cleaned up {cleaned_count} old history entries"
            )

        return cleaned_count

    def _update_action_record(
        self, action_id: str, updates: Dict[str, Any], append_lists: bool = False
    ) -> None:
        """Update an action record with new data."""
        action_file = self.base_dir / f"action_{action_id}.json"
        if not action_file.exists():
            return

        try:
            with open(action_file, "r") as f:
                action_record = json.load(f)

            for key, value in updates.items():
                if (
                    append_lists
                    and key in action_record
                    and isinstance(action_record[key], list)
                ):
                    if isinstance(value, list):
                        action_record[key].extend(value)
                    else:
                        action_record[key].append(value)
                else:
                    action_record[key] = value

            with open(action_file, "w") as f:
                json.dump(action_record, f, indent=2)

        except Exception as e:
            self.monitoring_logger.warning(
                f"Failed to update action record {action_id}: {e}"
            )

    def _get_action_record(self, action_id: str) -> Optional[Dict[str, Any]]:
        """Get an action record by ID."""
        action_file = self.base_dir / f"action_{action_id}.json"
        if not action_file.exists():
            return None

        try:
            with open(action_file, "r") as f:
                data: Dict[str, Any] = json.load(f)
                return data
        except Exception as e:
            self.monitoring_logger.warning(
                f"Failed to read action record {action_id}: {e}"
            )
            return None


# Singleton instance
_action_patch_history_logger = None


def get_action_patch_history_logger(
    config: Optional[Dict[str, Any]] = None,
) -> ActionPatchHistoryLogger:
    """
    Get or create the singleton ActionPatchHistoryLogger instance.

    Args:
        config: Optional configuration for the logger

    Returns:
        ActionPatchHistoryLogger: The action patch history logger instance
    """
    global _action_patch_history_logger
    if _action_patch_history_logger is None:
        _action_patch_history_logger = ActionPatchHistoryLogger(config)
    return _action_patch_history_logger


# Convenience functions
def log_action_start(
    session_id: str, action_type: str, context: Optional[Dict[str, Any]] = None
) -> str:
    """Log the start of a healing action."""
    return get_action_patch_history_logger().log_action_start(
        session_id, action_type, context
    )


def log_llm_interaction(
    action_id: str,
    provider: str,
    model: str,
    prompt: str,
    response: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Log an LLM interaction."""
    return get_action_patch_history_logger().log_llm_interaction(
        action_id, provider, model, prompt, response, metadata
    )


def log_file_change(
    action_id: str,
    file_path: str,
    change_type: str,
    old_content: Optional[str] = None,
    new_content: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Log a file change."""
    return get_action_patch_history_logger().log_file_change(
        action_id, file_path, change_type, old_content, new_content, metadata
    )


def log_patch_application(
    action_id: str,
    patch_id: str,
    target_file: str,
    patch_content: str,
    application_result: Dict[str, Any],
) -> str:
    """Log a patch application."""
    return get_action_patch_history_logger().log_patch_application(
        action_id, patch_id, target_file, patch_content, application_result
    )


def log_action_completion(
    action_id: str,
    status: str,
    performance_metrics: Optional[Dict[str, Any]] = None,
    error_details: Optional[Dict[str, Any]] = None,
) -> None:
    """Log the completion of a healing action."""
    get_action_patch_history_logger().log_action_completion(
        action_id, status, performance_metrics, error_details
    )
