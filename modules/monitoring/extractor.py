"""
Utility for extracting and analyzing errors from enhanced logs.
Provides advanced filtering and grouping capabilities.
"""

import json
import re
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# Path to log file
LOGS_DIR = Path(__file__).parent.parent.parent / "logs"
LOG_FILE = LOGS_DIR / "homeostasis.log"

# Define error severity levels for classification
ERROR_SEVERITY = {
    "critical": ["CRITICAL"],
    "high": ["ERROR"],
    "medium": ["WARNING"],
    "low": ["INFO", "DEBUG"],
}


def extract_errors(
    log_file: Path = LOG_FILE,
    levels: Optional[List[str]] = None,
    limit: int = 10,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    service_name: Optional[str] = None,
    exception_types: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Extract error logs from the log file with enhanced filtering options.

    Args:
        log_file: Path to the log file
        levels: List of log levels to extract (e.g., ["ERROR", "CRITICAL"]). If None, extract ERROR and CRITICAL.
        limit: Maximum number of errors to extract
        start_time: Only include logs after this time
        end_time: Only include logs before this time
        service_name: Filter by service name
        exception_types: Filter by exception types
        tags: Filter by tags

    Returns:
        List of error log entries
    """
    if not log_file.exists():
        return []

    # Default to ERROR and CRITICAL if not specified
    levels = levels or ["ERROR", "CRITICAL"]
    levels = [level.upper() for level in levels]  # Normalize to uppercase

    # Build the regex pattern for log level matching
    level_pattern = "|".join([f" - {level} - " for level in levels])

    errors = []

    with open(log_file, "r") as f:
        for line in f:
            try:
                # Try to parse the line as pure JSON first (for testing)
                if line.strip().startswith("{"):
                    error_data = json.loads(line.strip())
                    # Check if level matches
                    if error_data.get("level") not in levels:
                        continue
                else:
                    # Check if line matches any of the log levels in formatted log
                    if not re.search(f"({level_pattern})", line):
                        continue

                    # Extract the JSON part from the log line
                    json_match = re.search(r"({.*})$", line)
                    if not json_match:
                        continue

                    json_str = json_match.group(1)
                    error_data = json.loads(json_str)

                # Apply filters

                # Check timestamp if start_time or end_time provided
                if start_time or end_time:
                    try:
                        timestamp = datetime.fromisoformat(
                            error_data.get("timestamp", "")
                        )

                        if start_time and timestamp < start_time:
                            continue

                        if end_time and timestamp > end_time:
                            continue
                    except (ValueError, TypeError):
                        # Skip entries with invalid timestamps
                        continue

                # Check service name if provided
                if service_name and error_data.get("service") != service_name:
                    continue

                # Check exception type if provided
                if exception_types:
                    # Check in both the top-level exception_type and in error_details.exception_type
                    error_type = error_data.get("exception_type")
                    error_details = error_data.get("error_details", {})
                    error_details_type = (
                        error_details.get("exception_type") if error_details else None
                    )

                    if not (
                        error_type in exception_types
                        or error_details_type in exception_types
                    ):
                        continue

                # Check tags if provided
                if tags:
                    error_tags = error_data.get("tags", [])
                    # Skip if not all required tags are present
                    if not all(tag in error_tags for tag in tags):
                        continue

                # If all filters passed, add the error to the list
                errors.append(error_data)

                if len(errors) >= limit:
                    break
            except json.JSONDecodeError:
                # Skip lines with invalid JSON
                continue

    return errors


def get_latest_errors(
    limit: int = 10,
    levels: Optional[List[str]] = None,
    service_name: Optional[str] = None,
    exception_types: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    hours_back: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Get the latest errors from the log file with enhanced filtering options.

    Args:
        limit: Maximum number of errors to return
        levels: List of log levels to extract
        service_name: Filter by service name
        exception_types: Filter by exception types
        tags: Filter by tags
        hours_back: Only include logs from the last N hours

    Returns:
        List of error log entries, sorted by timestamp (newest first)
    """
    # Calculate start time if hours_back is provided
    start_time = None
    if hours_back is not None:
        start_time = datetime.now() - timedelta(hours=hours_back)

    # Extract errors with filters
    errors = extract_errors(
        limit=limit * 2,  # Get more than needed to account for filtering
        levels=levels,
        start_time=start_time,
        service_name=service_name,
        exception_types=exception_types,
        tags=tags,
    )

    # Sort by timestamp, newest first
    sorted_errors = sorted(
        errors,
        key=lambda x: datetime.fromisoformat(x.get("timestamp", "1970-01-01T00:00:00")),
        reverse=True,
    )

    # Return up to the requested limit
    return sorted_errors[:limit]


def get_error_summary(
    days_back: int = 7, service_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate a comprehensive summary of errors from the log file.

    Args:
        days_back: Number of days back to include in the summary
        service_name: Filter by service name

    Returns:
        Summary statistics about errors
    """
    # Calculate start time
    start_time = datetime.now() - timedelta(days=days_back)

    # Extract errors for analysis
    errors = extract_errors(
        limit=10000,  # Get up to 10000 errors for analysis
        start_time=start_time,
        service_name=service_name,
    )

    # Count errors by type
    error_types: Dict[str, int] = defaultdict(int)
    for error in errors:
        # Check both top-level and in error_details
        error_type = error.get("exception_type", "Unknown")
        if error_type == "Unknown" and "error_details" in error:
            error_type = error["error_details"].get("exception_type", "Unknown")

        error_types[error_type] += 1

    # Count errors by service
    services: Dict[str, int] = defaultdict(int)
    for error in errors:
        service = error.get("service", "Unknown")
        services[service] += 1

    # Count errors by endpoint (for HTTP requests)
    endpoints: Dict[str, int] = defaultdict(int)
    for error in errors:
        request_info = error.get("request_info", {})
        if request_info:
            path = request_info.get("path", "Unknown")
            method = request_info.get("method", "")
            if path != "Unknown":
                endpoint = f"{method} {path}"
                endpoints[endpoint] += 1

    # Count errors by tag
    tags: Dict[str, int] = defaultdict(int)
    for error in errors:
        for tag in error.get("tags", []):
            tags[tag] += 1

    # Get error counts by time periods
    current_time = datetime.now()
    time_periods: Dict[str, Any] = {
        "last_hour": 0,
        "last_day": 0,
        "last_week": 0,
        "by_day": defaultdict(int),
        "by_hour": defaultdict(int),
    }

    for error in errors:
        try:
            timestamp = datetime.fromisoformat(error.get("timestamp", ""))
            delta = current_time - timestamp

            # Count by time period
            if delta.total_seconds() <= 3600:  # Last hour
                time_periods["last_hour"] += 1

            if delta.total_seconds() <= 86400:  # Last day
                time_periods["last_day"] += 1

            if delta.total_seconds() <= 604800:  # Last week
                time_periods["last_week"] += 1

            # Count by day
            day_key = timestamp.strftime("%Y-%m-%d")
            time_periods["by_day"][day_key] += 1

            # Count by hour
            hour_key = timestamp.strftime("%Y-%m-%d %H:00")
            time_periods["by_hour"][hour_key] += 1

        except ValueError:
            # Skip entries with invalid timestamps
            continue

    # Create severity classification
    severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}

    for error in errors:
        level = error.get("level", "")

        for severity, levels in ERROR_SEVERITY.items():
            if level in levels:
                severity_counts[severity] += 1
                break

    # Get most frequent error stacks
    error_patterns = defaultdict(list)
    for error in errors:
        error_details = error.get("error_details", {})
        if error_details:
            exception_type = error_details.get("exception_type", "Unknown")
            message = error_details.get("message", "")
            key = f"{exception_type}: {message}"
            error_patterns[key].append(error)

    # Sort error patterns by frequency
    frequent_patterns = [
        {
            "pattern": pattern,
            "count": len(errors_list),
            "latest": max(
                [e.get("timestamp", "1970-01-01T00:00:00") for e in errors_list]
            ),
        }
        for pattern, errors_list in error_patterns.items()
    ]
    frequent_patterns.sort(key=lambda x: x["count"], reverse=True)

    return {
        "summary_period": f"Last {days_back} days",
        "timestamp": datetime.now().isoformat(),
        "total_errors": len(errors),
        "error_types": dict(error_types),
        "services": dict(services),
        "endpoints": dict(endpoints),
        "tags": dict(tags),
        "time_periods": time_periods,
        "severity": severity_counts,
        "frequent_patterns": frequent_patterns[
            :10
        ],  # Top 10 most frequent error patterns
    }


if __name__ == "__main__":
    # Enhanced example usage
    print("Latest errors (last 24 hours):")
    for error in get_latest_errors(5, hours_back=24):
        # Extract error details if available
        error_details = error.get("error_details", {})
        error_type = error_details.get(
            "exception_type", error.get("exception_type", "Unknown")
        )
        message = error_details.get("message", error.get("message", "Unknown error"))

        # Print with timestamp
        timestamp = error.get("timestamp", "Unknown time")
        print(f"- {timestamp}: {message} ({error_type})")

        # Print tags if available
        if "tags" in error:
            print(f"  Tags: {', '.join(error.get('tags', []))}")

        # Print request info if available
        if "request_info" in error:
            req_info = error["request_info"]
            method = req_info.get("method", "")
            path = req_info.get("path", "")
            status = error.get("response_info", {}).get("status_code", "")
            print(f"  Request: {method} {path} - {status}")

    print("\nComprehensive error summary:")
    summary = get_error_summary(days_back=7)
    print(
        f"Total errors: {summary['total_errors']} (in the last {summary['summary_period']})"
    )
    print(
        f"Errors by severity: Critical: {summary['severity']['critical']}, High: {summary['severity']['high']}, Medium: {summary['severity']['medium']}"
    )
    print(f"Errors in the last hour: {summary['time_periods']['last_hour']}")
    print(f"Errors in the last day: {summary['time_periods']['last_day']}")

    print("\nError types:")
    for error_type, count in sorted(
        summary["error_types"].items(), key=lambda x: x[1], reverse=True
    )[:5]:
        print(f"- {error_type}: {count}")

    print("\nMost affected endpoints:")
    for endpoint, count in sorted(
        summary["endpoints"].items(), key=lambda x: x[1], reverse=True
    )[:5]:
        print(f"- {endpoint}: {count}")

    print("\nMost frequent error patterns:")
    for i, pattern in enumerate(summary["frequent_patterns"][:3], 1):
        print(
            f"{i}. {pattern['pattern']} - {pattern['count']} occurrences, latest at {pattern['latest']}"
        )
