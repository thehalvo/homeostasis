"""
Utility for extracting and analyzing errors from logs.
"""
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# Path to log file
LOGS_DIR = Path(__file__).parent.parent.parent / "logs"
LOG_FILE = LOGS_DIR / "homeostasis.log"


def extract_errors(log_file: Path = LOG_FILE, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Extract error logs from the log file.

    Args:
        log_file: Path to the log file
        limit: Maximum number of errors to extract

    Returns:
        List of error log entries
    """
    if not log_file.exists():
        return []
    
    errors = []
    
    with open(log_file, "r") as f:
        for line in f:
            # Look for ERROR or CRITICAL level logs
            if " - ERROR - " in line or " - CRITICAL - " in line:
                try:
                    # Extract the JSON part from the log line
                    json_match = re.search(r'({.*})$', line)
                    if json_match:
                        json_str = json_match.group(1)
                        error_data = json.loads(json_str)
                        errors.append(error_data)
                        
                        if len(errors) >= limit:
                            break
                except json.JSONDecodeError:
                    # Skip lines with invalid JSON
                    continue
    
    return errors


def get_latest_errors(limit: int = 10) -> List[Dict[str, Any]]:
    """
    Get the latest errors from the log file.

    Args:
        limit: Maximum number of errors to return

    Returns:
        List of error log entries, sorted by timestamp (newest first)
    """
    errors = extract_errors(limit=limit)
    
    # Sort by timestamp, newest first
    return sorted(
        errors,
        key=lambda x: datetime.fromisoformat(x.get("timestamp", "1970-01-01T00:00:00")),
        reverse=True
    )


def get_error_summary() -> Dict[str, Any]:
    """
    Generate a summary of errors from the log file.

    Returns:
        Summary statistics about errors
    """
    errors = extract_errors(limit=1000)  # Get up to 1000 errors for analysis
    
    # Count errors by type
    error_types = {}
    for error in errors:
        error_type = error.get("exception_type", "Unknown")
        error_types[error_type] = error_types.get(error_type, 0) + 1
    
    # Count errors by service
    services = {}
    for error in errors:
        service = error.get("service", "Unknown")
        services[service] = services.get(service, 0) + 1
    
    # Get error counts by time
    current_time = datetime.now()
    last_hour = 0
    last_day = 0
    
    for error in errors:
        try:
            timestamp = datetime.fromisoformat(error.get("timestamp", ""))
            delta = current_time - timestamp
            
            if delta.total_seconds() <= 3600:  # Last hour
                last_hour += 1
            
            if delta.total_seconds() <= 86400:  # Last day
                last_day += 1
        except ValueError:
            # Skip entries with invalid timestamps
            continue
    
    return {
        "total_errors": len(errors),
        "error_types": error_types,
        "services": services,
        "last_hour": last_hour,
        "last_day": last_day
    }


if __name__ == "__main__":
    # Example usage
    print("Latest errors:")
    for error in get_latest_errors(5):
        print(f"- {error.get('timestamp')}: {error.get('message')} ({error.get('exception_type', 'Unknown')})")
    
    print("\nError summary:")
    summary = get_error_summary()
    print(f"Total errors: {summary['total_errors']}")
    print(f"Errors in the last hour: {summary['last_hour']}")
    print(f"Errors in the last day: {summary['last_day']}")
    
    print("\nError types:")
    for error_type, count in summary['error_types'].items():
        print(f"- {error_type}: {count}")