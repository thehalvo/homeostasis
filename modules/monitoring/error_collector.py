"""
Error Collector Module

This module provides error collection and aggregation capabilities for Homeostasis.
"""

import json
import logging
import threading
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class ErrorCollector:
    """
    Collects and aggregates errors from various sources.

    This class provides a centralized way to collect, store, and analyze errors
    from different components of the system.
    """

    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize the error collector.

        Args:
            storage_path: Path to store collected errors (optional)
        """
        self.errors: List[Dict[str, Any]] = []
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.storage_path = storage_path
        self._lock = threading.Lock()

        if self.storage_path:
            self.storage_path.mkdir(parents=True, exist_ok=True)

        logger.info("Error collector initialized")

    def collect_error(self, error: Dict[str, Any]) -> str:
        """
        Collect a single error.

        Args:
            error: Error data containing type, message, stack trace, etc.

        Returns:
            Error ID
        """
        with self._lock:
            # Add metadata
            error_id = f"error_{len(self.errors)}_{datetime.now().timestamp()}"
            error["id"] = error_id
            error["collected_at"] = datetime.now().isoformat()

            # Store error
            self.errors.append(error)

            # Update counts
            error_type = error.get("error_type", "unknown")
            self.error_counts[error_type] += 1

            # Persist if storage path is set
            if self.storage_path:
                self._persist_error(error)

            logger.debug(f"Collected error: {error_id}")
            return error_id

    def collect_batch(self, errors: List[Dict[str, Any]]) -> List[str]:
        """
        Collect multiple errors at once.

        Args:
            errors: List of error data

        Returns:
            List of error IDs
        """
        error_ids = []
        for error in errors:
            error_id = self.collect_error(error)
            error_ids.append(error_id)
        return error_ids

    def get_errors(
        self,
        error_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get collected errors with optional filtering.

        Args:
            error_type: Filter by error type
            start_time: Filter by start time
            end_time: Filter by end time
            limit: Maximum number of errors to return

        Returns:
            List of errors matching the filters
        """
        with self._lock:
            results = self.errors.copy()

        # Apply filters
        if error_type:
            results = [e for e in results if e.get("error_type") == error_type]

        if start_time:
            results = [
                e
                for e in results
                if datetime.fromisoformat(e["collected_at"]) >= start_time
            ]

        if end_time:
            results = [
                e
                for e in results
                if datetime.fromisoformat(e["collected_at"]) <= end_time
            ]

        # Apply limit
        if limit:
            results = results[-limit:]

        return results

    def get_error_by_id(self, error_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific error by ID.

        Args:
            error_id: Error ID

        Returns:
            Error data or None if not found
        """
        with self._lock:
            for error in self.errors:
                if error.get("id") == error_id:
                    return error.copy()
        return None

    def get_error_summary(self) -> Dict[str, Any]:
        """
        Get a summary of collected errors.

        Returns:
            Summary statistics
        """
        with self._lock:
            total_errors = len(self.errors)
            error_types = dict(self.error_counts)

            if self.errors:
                latest_error = self.errors[-1]
                oldest_error = self.errors[0]
            else:
                latest_error = None
                oldest_error = None

        summary = {
            "total_errors": total_errors,
            "error_types": error_types,
            "latest_error": latest_error,
            "oldest_error": oldest_error,
            "collection_start": oldest_error["collected_at"] if oldest_error else None,
            "collection_end": latest_error["collected_at"] if latest_error else None,
        }

        return summary

    def clear_errors(self, before: Optional[datetime] = None):
        """
        Clear collected errors.

        Args:
            before: Clear only errors before this time (optional)
        """
        with self._lock:
            if before:
                self.errors = [
                    e
                    for e in self.errors
                    if datetime.fromisoformat(e["collected_at"]) >= before
                ]
                # Recalculate counts
                self.error_counts.clear()
                for error in self.errors:
                    error_type = error.get("error_type", "unknown")
                    self.error_counts[error_type] += 1
            else:
                self.errors.clear()
                self.error_counts.clear()

        logger.info(f"Cleared errors (before: {before})")

    def analyze_patterns(self) -> Dict[str, Any]:
        """
        Analyze error patterns.

        Returns:
            Pattern analysis results
        """
        with self._lock:
            errors_copy = self.errors.copy()

        if not errors_copy:
            return {"patterns": [], "insights": []}

        patterns: List[Dict[str, Any]] = []

        # Time-based patterns
        error_times = [datetime.fromisoformat(e["collected_at"]) for e in errors_copy]
        if len(error_times) > 1:
            time_diff = (error_times[-1] - error_times[0]).total_seconds()
            error_rate = len(errors_copy) / time_diff if time_diff > 0 else 0

            patterns.append(
                {"type": "error_rate", "value": error_rate, "unit": "errors_per_second"}
            )

        # Error type patterns
        type_percentages = {}
        total = len(errors_copy)
        for error_type, count in self.error_counts.items():
            type_percentages[error_type] = (count / total) * 100

        patterns.append({"type": "error_distribution", "value": type_percentages})

        # Burst detection
        bursts = self._detect_error_bursts(errors_copy)
        if bursts:
            patterns.append({"type": "error_bursts", "value": bursts})

        # Generate insights
        insights = []

        # Most common error
        if self.error_counts:
            most_common = max(self.error_counts.items(), key=lambda x: x[1])
            insights.append(
                {
                    "type": "most_common_error",
                    "error_type": most_common[0],
                    "count": most_common[1],
                    "percentage": (most_common[1] / total) * 100,
                }
            )

        return {"patterns": patterns, "insights": insights}

    def _detect_error_bursts(
        self,
        errors: List[Dict[str, Any]],
        window_seconds: int = 60,
        threshold: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Detect error bursts in the collected errors.

        Args:
            errors: List of errors to analyze
            window_seconds: Time window for burst detection
            threshold: Minimum errors to consider a burst

        Returns:
            List of detected bursts
        """
        if len(errors) < threshold:
            return []

        bursts = []
        window_start = 0

        for i in range(len(errors)):
            window_end_time = datetime.fromisoformat(
                errors[window_start]["collected_at"]
            ) + timedelta(seconds=window_seconds)

            # Move window start if current error is outside window
            current_time = datetime.fromisoformat(errors[i]["collected_at"])
            while window_start < i and current_time > window_end_time:
                window_start += 1
                if window_start < len(errors):
                    window_end_time = datetime.fromisoformat(
                        errors[window_start]["collected_at"]
                    ) + timedelta(seconds=window_seconds)

            # Check if we have a burst
            window_size = i - window_start + 1
            if window_size >= threshold:
                burst_errors = errors[window_start : i + 1]
                bursts.append(
                    {
                        "start_time": burst_errors[0]["collected_at"],
                        "end_time": burst_errors[-1]["collected_at"],
                        "error_count": len(burst_errors),
                        "error_types": list(
                            set(e.get("error_type", "unknown") for e in burst_errors)
                        ),
                    }
                )

        return bursts

    def _persist_error(self, error: Dict[str, Any]):
        """
        Persist an error to storage.

        Args:
            error: Error data to persist
        """
        if not self.storage_path:
            return

        try:
            # Create filename based on date
            date_str = datetime.now().strftime("%Y-%m-%d")
            file_path = self.storage_path / f"errors_{date_str}.jsonl"

            # Append error to file
            with open(file_path, "a") as f:
                json.dump(error, f)
                f.write("\n")

        except Exception as e:
            logger.error(f"Failed to persist error: {e}")

    def load_from_storage(self, date: Optional[datetime] = None):
        """
        Load errors from storage.

        Args:
            date: Specific date to load (loads all if None)
        """
        if not self.storage_path:
            logger.warning("No storage path configured")
            return

        files_to_load = []

        if date:
            # Load specific date
            date_str = date.strftime("%Y-%m-%d")
            file_path = self.storage_path / f"errors_{date_str}.jsonl"
            if file_path.exists():
                files_to_load.append(file_path)
        else:
            # Load all files
            files_to_load = list(self.storage_path.glob("errors_*.jsonl"))

        loaded_count = 0
        for file_path in files_to_load:
            try:
                with open(file_path, "r") as f:
                    for line in f:
                        if line.strip():
                            error = json.loads(line)
                            # Don't use collect_error to avoid re-persisting
                            with self._lock:
                                self.errors.append(error)
                                error_type = error.get("error_type", "unknown")
                                self.error_counts[error_type] += 1
                            loaded_count += 1

            except Exception as e:
                logger.error(f"Failed to load errors from {file_path}: {e}")

        logger.info(f"Loaded {loaded_count} errors from storage")

    def export_errors(self, format: str = "json") -> Union[str, Dict[str, Any]]:
        """
        Export collected errors in specified format.

        Args:
            format: Export format ('json', 'csv', etc.)

        Returns:
            Exported data
        """
        with self._lock:
            errors_copy = self.errors.copy()

        if format == "json":
            return json.dumps(errors_copy, indent=2)

        elif format == "csv":
            # Simple CSV export
            if not errors_copy:
                return "id,error_type,message,collected_at\n"

            lines = ["id,error_type,message,collected_at"]
            for error in errors_copy:
                lines.append(
                    f"{error.get('id', '')},{error.get('error_type', '')},"
                    f'"{error.get("message", "")}",{error.get("collected_at", "")}'
                )
            return "\n".join(lines)

        else:
            raise ValueError(f"Unsupported format: {format}")


# Additional utility functions
def create_error_from_exception(
    e: Exception, context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create error data from an exception.

    Args:
        e: Exception instance
        context: Additional context

    Returns:
        Error data dictionary
    """
    import traceback

    error_data = {
        "error_type": type(e).__name__,
        "message": str(e),
        "traceback": traceback.format_exc(),
        "context": context or {},
    }

    return error_data
