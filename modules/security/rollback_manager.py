"""
Rollback Manager for Safe Patch Deployment and Recovery.

This module provides automatic rollback mechanisms to revert patches that introduce
vulnerabilities or break system functionality, with support for quick manual rollback.
"""

import json
import logging
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..security.security_config import SecurityConfig, get_security_config

logger = logging.getLogger(__name__)


class RollbackTrigger(Enum):
    """Triggers for rollback."""

    MANUAL = "manual"
    SECURITY_VIOLATION = "security_violation"
    TEST_FAILURE = "test_failure"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    ERROR_RATE_SPIKE = "error_rate_spike"
    AUTOMATIC_DETECTION = "automatic_detection"
    COMPLIANCE_VIOLATION = "compliance_violation"


class RollbackStatus(Enum):
    """Status of rollback operations."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BackupSnapshot:
    """Snapshot of code state before patch application."""

    snapshot_id: str
    context_id: str
    timestamp: str
    file_paths: List[str]
    backup_location: str
    git_commit_hash: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RollbackOperation:
    """Record of a rollback operation."""

    rollback_id: str
    snapshot_id: str
    context_id: str
    trigger: RollbackTrigger
    triggered_by: str
    triggered_at: str
    status: RollbackStatus = RollbackStatus.PENDING
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    files_restored: List[str] = field(default_factory=list)
    validation_results: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityMonitoringConfig:
    """Configuration for security monitoring that triggers rollbacks."""

    enable_vulnerability_scanning: bool = True
    enable_dependency_checking: bool = True
    enable_static_analysis: bool = True
    enable_runtime_monitoring: bool = True
    max_error_rate_threshold: float = 0.1  # 10% error rate
    max_response_time_degradation: float = 2.0  # 2x slower
    monitoring_window_minutes: int = 5
    alert_cooldown_minutes: int = 15


class RollbackManager:
    """
    Manages safe rollback mechanisms for LLM-generated patches.

    Features:
    - Pre-patch snapshot creation
    - Automatic rollback on security violations
    - Manual rollback commands
    - Validation after rollback
    - Integration with monitoring systems
    """

    def __init__(
        self,
        config: Optional[SecurityConfig] = None,
        storage_dir: Optional[Path] = None,
        monitoring_config: Optional[SecurityMonitoringConfig] = None,
    ):
        """Initialize the rollback manager."""
        self.config = config or get_security_config()
        self.storage_dir = storage_dir or Path.cwd() / "rollback"
        self.storage_dir.mkdir(exist_ok=True)

        # Create subdirectories
        (self.storage_dir / "snapshots").mkdir(exist_ok=True)
        (self.storage_dir / "backups").mkdir(exist_ok=True)
        (self.storage_dir / "logs").mkdir(exist_ok=True)

        self.monitoring_config = monitoring_config or SecurityMonitoringConfig()

        # In-memory stores
        self.snapshots: Dict[str, BackupSnapshot] = {}
        self.rollback_operations: Dict[str, RollbackOperation] = {}

        # Load existing data
        self._load_rollback_data()

        logger.info(f"Rollback Manager initialized with storage at {self.storage_dir}")

    def create_snapshot(
        self,
        context_id: str,
        file_paths: List[str],
        project_root: Optional[Path] = None,
    ) -> str:
        """
        Create a snapshot before applying a patch.

        Args:
            context_id: Context ID for the patch
            file_paths: List of files that will be modified
            project_root: Root directory of the project

        Returns:
            Snapshot ID
        """
        snapshot_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        # Create backup directory for this snapshot
        backup_dir = self.storage_dir / "backups" / snapshot_id
        backup_dir.mkdir(exist_ok=True)

        # Backup files
        backed_up_files = []
        for file_path in file_paths:
            try:
                file_path_obj = Path(file_path)
                if file_path_obj.exists():
                    # Create subdirectory structure in backup
                    relative_path = (
                        file_path_obj.relative_to(project_root)
                        if project_root
                        else file_path_obj
                    )
                    backup_file = backup_dir / relative_path
                    backup_file.parent.mkdir(parents=True, exist_ok=True)

                    # Copy the file
                    shutil.copy2(file_path_obj, backup_file)
                    backed_up_files.append(str(file_path_obj))

            except Exception as e:
                logger.warning(f"Failed to backup file {file_path}: {e}")

        # Get current git commit hash if in a git repository
        git_commit_hash = self._get_current_git_commit(project_root)

        # Create snapshot record
        snapshot = BackupSnapshot(
            snapshot_id=snapshot_id,
            context_id=context_id,
            timestamp=timestamp,
            file_paths=backed_up_files,
            backup_location=str(backup_dir),
            git_commit_hash=git_commit_hash,
            metadata={
                "project_root": str(project_root) if project_root else None,
                "total_files": len(backed_up_files),
                "backup_size_bytes": self._calculate_backup_size(backup_dir),
            },
        )

        self.snapshots[snapshot_id] = snapshot
        self._save_snapshots()

        logger.info(f"Created snapshot {snapshot_id} with {len(backed_up_files)} files")
        return snapshot_id

    def rollback_patch(
        self,
        snapshot_id: str,
        trigger: RollbackTrigger,
        triggered_by: str,
        reason: Optional[str] = None,
    ) -> str:
        """
        Rollback a patch using a snapshot.

        Args:
            snapshot_id: Snapshot ID to rollback to
            trigger: What triggered the rollback
            triggered_by: User or system that triggered the rollback
            reason: Optional reason for rollback

        Returns:
            Rollback operation ID
        """
        snapshot = self.snapshots.get(snapshot_id)
        if not snapshot:
            raise ValueError(f"Snapshot {snapshot_id} not found")

        rollback_id = str(uuid.uuid4())

        # Create rollback operation record
        rollback_op = RollbackOperation(
            rollback_id=rollback_id,
            snapshot_id=snapshot_id,
            context_id=snapshot.context_id,
            trigger=trigger,
            triggered_by=triggered_by,
            triggered_at=datetime.now().isoformat(),
            metadata={"reason": reason} if reason else {},
        )

        self.rollback_operations[rollback_id] = rollback_op

        try:
            # Start rollback process
            rollback_op.status = RollbackStatus.IN_PROGRESS
            self._save_rollback_operations()

            # Restore files from backup
            backup_dir = Path(snapshot.backup_location)
            restored_files = []

            for original_file_path in snapshot.file_paths:
                try:
                    original_path = Path(original_file_path)

                    # Find corresponding backup file
                    if snapshot.metadata.get("project_root"):
                        project_root = Path(snapshot.metadata["project_root"])
                        relative_path = original_path.relative_to(project_root)
                        backup_file = backup_dir / relative_path
                    else:
                        backup_file = backup_dir / original_path.name

                    if backup_file.exists():
                        # Restore the file
                        shutil.copy2(backup_file, original_path)
                        restored_files.append(str(original_path))
                        logger.debug(f"Restored file: {original_path}")
                    else:
                        logger.warning(f"Backup file not found: {backup_file}")

                except Exception as e:
                    logger.error(f"Failed to restore file {original_file_path}: {e}")

            rollback_op.files_restored = restored_files

            # Validate rollback
            validation_results = self._validate_rollback(rollback_op, snapshot)
            rollback_op.validation_results = validation_results

            # Mark as completed
            rollback_op.status = RollbackStatus.COMPLETED
            rollback_op.completed_at = datetime.now().isoformat()

            self._save_rollback_operations()

            logger.info(
                f"Successfully rolled back {len(restored_files)} files using snapshot {snapshot_id}"
            )
            return rollback_id

        except Exception as e:
            rollback_op.status = RollbackStatus.FAILED
            rollback_op.error_message = str(e)
            self._save_rollback_operations()

            logger.error(f"Rollback failed for snapshot {snapshot_id}: {e}")
            raise

    def monitor_and_auto_rollback(
        self, context_id: str, snapshot_id: str, monitoring_duration_minutes: int = 5
    ) -> Optional[str]:
        """
        Monitor system for issues and automatically rollback if problems detected.

        Args:
            context_id: Context ID of the applied patch
            snapshot_id: Snapshot ID to rollback to if needed
            monitoring_duration_minutes: How long to monitor

        Returns:
            Rollback ID if rollback was triggered, None otherwise
        """
        logger.info(
            f"Starting monitoring for context {context_id} for {monitoring_duration_minutes} minutes"
        )

        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=monitoring_duration_minutes)

        baseline_metrics = self._collect_baseline_metrics()

        while datetime.now() < end_time:
            try:
                # Collect current metrics
                current_metrics = self._collect_current_metrics()

                # Check for security violations
                security_issues = self._check_security_violations(current_metrics)
                if security_issues:
                    logger.warning(f"Security violations detected: {security_issues}")
                    return self.rollback_patch(
                        snapshot_id,
                        RollbackTrigger.SECURITY_VIOLATION,
                        "auto_monitor",
                        f"Security violations: {security_issues}",
                    )

                # Check for performance degradation
                if self._check_performance_degradation(
                    baseline_metrics, current_metrics
                ):
                    logger.warning("Performance degradation detected")
                    return self.rollback_patch(
                        snapshot_id,
                        RollbackTrigger.PERFORMANCE_DEGRADATION,
                        "auto_monitor",
                        "Performance degradation detected",
                    )

                # Check for error rate spikes
                if self._check_error_rate_spike(baseline_metrics, current_metrics):
                    logger.warning("Error rate spike detected")
                    return self.rollback_patch(
                        snapshot_id,
                        RollbackTrigger.ERROR_RATE_SPIKE,
                        "auto_monitor",
                        "Error rate spike detected",
                    )

                # Wait before next check
                time.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"Error during monitoring: {e}")
                time.sleep(30)

        logger.info(
            f"Monitoring completed for context {context_id} - no issues detected"
        )
        return None

    def get_rollback_history(
        self, context_id: Optional[str] = None
    ) -> List[RollbackOperation]:
        """
        Get rollback history, optionally filtered by context ID.

        Args:
            context_id: Optional context ID to filter by

        Returns:
            List of rollback operations
        """
        operations = list(self.rollback_operations.values())

        if context_id:
            operations = [op for op in operations if op.context_id == context_id]

        # Sort by triggered_at descending
        operations.sort(key=lambda op: op.triggered_at, reverse=True)

        return operations

    def cleanup_old_snapshots(self, retention_days: int = 30) -> int:
        """
        Clean up old snapshots to save storage space.

        Args:
            retention_days: Number of days to retain snapshots

        Returns:
            Number of snapshots cleaned up
        """
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        cleaned_count = 0

        snapshots_to_remove = []

        for snapshot_id, snapshot in self.snapshots.items():
            snapshot_date = datetime.fromisoformat(snapshot.timestamp)
            if snapshot_date < cutoff_date:
                snapshots_to_remove.append(snapshot_id)

        for snapshot_id in snapshots_to_remove:
            try:
                snapshot = self.snapshots[snapshot_id]

                # Remove backup directory
                backup_dir = Path(snapshot.backup_location)
                if backup_dir.exists():
                    shutil.rmtree(backup_dir)

                # Remove from memory
                del self.snapshots[snapshot_id]
                cleaned_count += 1

            except Exception as e:
                logger.error(f"Error cleaning up snapshot {snapshot_id}: {e}")

        if cleaned_count > 0:
            self._save_snapshots()
            logger.info(f"Cleaned up {cleaned_count} old snapshots")

        return cleaned_count

    def get_rollback_summary(self) -> Dict[str, Any]:
        """
        Get a summary of rollback activities.

        Returns:
            Rollback summary
        """
        total_snapshots = len(self.snapshots)
        total_rollbacks = len(self.rollback_operations)

        successful_rollbacks = len(
            [
                op
                for op in self.rollback_operations.values()
                if op.status == RollbackStatus.COMPLETED
            ]
        )

        failed_rollbacks = len(
            [
                op
                for op in self.rollback_operations.values()
                if op.status == RollbackStatus.FAILED
            ]
        )

        rollbacks_by_trigger = {}
        for op in self.rollback_operations.values():
            trigger = op.trigger.value
            rollbacks_by_trigger[trigger] = rollbacks_by_trigger.get(trigger, 0) + 1

        return {
            "total_snapshots": total_snapshots,
            "total_rollbacks": total_rollbacks,
            "successful_rollbacks": successful_rollbacks,
            "failed_rollbacks": failed_rollbacks,
            "success_rate": (
                successful_rollbacks / total_rollbacks if total_rollbacks > 0 else 0
            ),
            "rollbacks_by_trigger": rollbacks_by_trigger,
            "storage_usage_mb": self._calculate_total_storage_usage() / (1024 * 1024),
            "recent_rollbacks": [
                {
                    "rollback_id": op.rollback_id,
                    "trigger": op.trigger.value,
                    "status": op.status.value,
                    "triggered_at": op.triggered_at,
                }
                for op in sorted(
                    self.rollback_operations.values(),
                    key=lambda x: x.triggered_at,
                    reverse=True,
                )[:10]
            ],
        }

    def _get_current_git_commit(self, project_root: Optional[Path]) -> Optional[str]:
        """Get the current git commit hash."""
        try:
            if project_root:
                result = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=project_root,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode == 0:
                    return result.stdout.strip()
        except Exception as e:
            logger.debug(f"Could not get git commit hash: {e}")
        return None

    def _calculate_backup_size(self, backup_dir: Path) -> int:
        """Calculate the total size of backup directory."""
        try:
            total_size = 0
            for file_path in backup_dir.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            return total_size
        except Exception:
            return 0

    def _calculate_total_storage_usage(self) -> int:
        """Calculate total storage usage for all backups."""
        try:
            total_size = 0
            backup_root = self.storage_dir / "backups"
            for file_path in backup_root.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            return total_size
        except Exception:
            return 0

    def _validate_rollback(
        self, rollback_op: RollbackOperation, snapshot: BackupSnapshot
    ) -> Dict[str, Any]:
        """Validate that rollback was successful."""
        validation_results = {
            "files_restored": len(rollback_op.files_restored),
            "files_expected": len(snapshot.file_paths),
            "all_files_restored": len(rollback_op.files_restored)
            == len(snapshot.file_paths),
            "validation_timestamp": datetime.now().isoformat(),
        }

        # Additional validation could include:
        # - Syntax checking
        # - Running basic tests
        # - Verifying file checksums

        return validation_results

    def _collect_baseline_metrics(self) -> Dict[str, Any]:
        """Collect baseline metrics before monitoring."""
        # In a real implementation, this would collect actual metrics
        return {
            "timestamp": datetime.now().isoformat(),
            "error_rate": 0.01,  # 1% baseline error rate
            "avg_response_time": 100,  # 100ms baseline
            "cpu_usage": 30,  # 30% CPU
            "memory_usage": 50,  # 50% memory
        }

    def _collect_current_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics."""
        # In a real implementation, this would collect actual metrics from monitoring systems
        return {
            "timestamp": datetime.now().isoformat(),
            "error_rate": 0.02,  # Current error rate
            "avg_response_time": 120,  # Current response time
            "cpu_usage": 35,  # Current CPU
            "memory_usage": 55,  # Current memory
        }

    def _check_security_violations(self, metrics: Dict[str, Any]) -> List[str]:
        """Check for security violations in current metrics."""
        violations = []

        # In a real implementation, this would check:
        # - Failed authentication attempts
        # - Suspicious network activity
        # - Unauthorized access attempts
        # - Security scanner alerts

        return violations

    def _check_performance_degradation(
        self, baseline: Dict[str, Any], current: Dict[str, Any]
    ) -> bool:
        """Check if performance has degraded significantly."""
        baseline_response_time = baseline.get("avg_response_time", 0)
        current_response_time = current.get("avg_response_time", 0)

        if baseline_response_time > 0:
            degradation_ratio = current_response_time / baseline_response_time
            return (
                degradation_ratio > self.monitoring_config.max_response_time_degradation
            )

        return False

    def _check_error_rate_spike(
        self, baseline: Dict[str, Any], current: Dict[str, Any]
    ) -> bool:
        """Check if error rate has spiked."""
        current_error_rate = current.get("error_rate", 0)
        return current_error_rate > self.monitoring_config.max_error_rate_threshold

    def _load_rollback_data(self):
        """Load rollback data from storage."""
        try:
            # Load snapshots
            snapshots_file = self.storage_dir / "snapshots.json"
            if snapshots_file.exists():
                with open(snapshots_file, "r") as f:
                    snapshots_data = json.load(f)
                    for snapshot_data in snapshots_data:
                        snapshot = BackupSnapshot(**snapshot_data)
                        self.snapshots[snapshot.snapshot_id] = snapshot

            # Load rollback operations
            rollbacks_file = self.storage_dir / "rollback_operations.json"
            if rollbacks_file.exists():
                with open(rollbacks_file, "r") as f:
                    rollbacks_data = json.load(f)
                    for rollback_data in rollbacks_data:
                        # Convert enum values
                        rollback_data["trigger"] = RollbackTrigger(
                            rollback_data["trigger"]
                        )
                        rollback_data["status"] = RollbackStatus(
                            rollback_data["status"]
                        )

                        rollback_op = RollbackOperation(**rollback_data)
                        self.rollback_operations[rollback_op.rollback_id] = rollback_op

        except Exception as e:
            logger.warning(f"Error loading rollback data: {e}")

    def _save_snapshots(self):
        """Save snapshots to storage."""
        try:
            snapshots_data = [
                {
                    "snapshot_id": snapshot.snapshot_id,
                    "context_id": snapshot.context_id,
                    "timestamp": snapshot.timestamp,
                    "file_paths": snapshot.file_paths,
                    "backup_location": snapshot.backup_location,
                    "git_commit_hash": snapshot.git_commit_hash,
                    "metadata": snapshot.metadata,
                }
                for snapshot in self.snapshots.values()
            ]

            with open(self.storage_dir / "snapshots.json", "w") as f:
                json.dump(snapshots_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving snapshots: {e}")

    def _save_rollback_operations(self):
        """Save rollback operations to storage."""
        try:
            rollbacks_data = [
                {
                    "rollback_id": op.rollback_id,
                    "snapshot_id": op.snapshot_id,
                    "context_id": op.context_id,
                    "trigger": op.trigger.value,
                    "triggered_by": op.triggered_by,
                    "triggered_at": op.triggered_at,
                    "status": op.status.value,
                    "completed_at": op.completed_at,
                    "error_message": op.error_message,
                    "files_restored": op.files_restored,
                    "validation_results": op.validation_results,
                    "metadata": op.metadata,
                }
                for op in self.rollback_operations.values()
            ]

            with open(self.storage_dir / "rollback_operations.json", "w") as f:
                json.dump(rollbacks_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving rollback operations: {e}")


def create_rollback_manager(
    config: Optional[SecurityConfig] = None,
    storage_dir: Optional[Path] = None,
    monitoring_config: Optional[SecurityMonitoringConfig] = None,
) -> RollbackManager:
    """Create and return a configured rollback manager."""
    return RollbackManager(config, storage_dir, monitoring_config)
