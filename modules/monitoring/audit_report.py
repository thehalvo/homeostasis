#!/usr/bin/env python3
"""
Audit Report Generator for Homeostasis.

This tool generates reports from audit logs for:
1. Healing activities (by time period and service)
2. Security events
3. Deployment statistics
4. Human interventions
5. Anomalous activities

It provides command-line options for various report types, formats, and date ranges.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add project root to sys.path
project_root = Path(__file__).parents[2]
sys.path.insert(0, str(project_root))

from modules.monitoring.audit_monitor import (export_audit_data,  # noqa: E402
                                              generate_activity_report,
                                              get_audit_monitor)


def format_table(data: List[List[Any]], headers: List[str]) -> str:
    """Format data as a text table.

    Args:
        data: Table rows
        headers: Column headers

    Returns:
        str: Formatted table
    """
    if not data:
        return "No data to display."

    # Determine column widths
    col_widths = []
    for i in range(len(headers)):
        col_width = len(str(headers[i]))
        for row in data:
            if i < len(row):
                col_width = max(col_width, len(str(row[i])))
        col_widths.append(col_width + 2)  # Add padding

    # Build separator line
    separator = "+"
    for width in col_widths:
        separator += "-" * width + "+"

    # Build header row
    header_row = "|"
    for i, header in enumerate(headers):
        header_row += f" {header:{col_widths[i] - 2}} |"

    # Build result
    result = [separator, header_row, separator]

    # Add data rows
    for row in data:
        data_row = "|"
        for i, cell in enumerate(row):
            if i < len(col_widths):
                data_row += f" {str(cell):{col_widths[i] - 2}} |"
        result.append(data_row)

    # Add bottom separator
    result.append(separator)

    return "\n".join(result)


def get_summary_report(report: Dict[str, Any]) -> str:
    """Generate a summary report in text format.

    Args:
        report: Activity report from generate_activity_report

    Returns:
        str: Formatted summary report
    """
    # Format header
    header = f"Healing Activity Report ({report['time_period']})\n"
    header += f"From: {report['start_time']} to {report['end_time']}\n"
    header += f"Total Events: {report['total_events']}\n\n"

    # Format event counts
    event_counts = report["event_counts"]
    event_count_table = []
    for event_type, count in sorted(
        event_counts.items(), key=lambda x: x[1], reverse=True
    ):
        event_count_table.append([event_type, count])

    event_section = "Event Counts:\n"
    event_section += format_table(event_count_table, ["Event Type", "Count"])

    # Format healing activities
    healing = report["healing_activities"]
    healing_table = [
        ["Errors Detected", healing.get("errors_detected", 0)],
        ["Fixes Generated", healing.get("fixes_generated", 0)],
        ["Fixes Deployed", healing.get("fixes_deployed", 0)],
        ["Fixes Approved", healing.get("fixes_approved", 0)],
        ["Fixes Rejected", healing.get("fixes_rejected", 0)],
        ["Success Rate", f"{healing.get('success_rate', 0) * 100:.1f}%"],
    ]

    healing_section = "\nHealing Activities:\n"
    healing_section += format_table(healing_table, ["Activity", "Count"])

    # Format error events (limited to 10)
    error_events = report.get("error_events", [])[:10]
    error_table = []
    for event in error_events:
        timestamp = event.get("timestamp", "")
        event_type = event.get("event_type", "")
        user = event.get("user", "")
        details_str = str(event.get("details", {}))
        if len(details_str) > 50:
            details_str = details_str[:47] + "..."
        error_table.append([timestamp, event_type, user, details_str])

    error_section = "\nRecent Error Events:\n"
    if error_table:
        error_section += format_table(
            error_table, ["Timestamp", "Event Type", "User", "Details"]
        )
    else:
        error_section += "No error events recorded."

    return header + event_section + healing_section + error_section


def get_user_activity_report(report: Dict[str, Any]) -> str:
    """Generate a user activity report in text format.

    Args:
        report: Activity report from generate_activity_report

    Returns:
        str: Formatted user activity report
    """
    # Format header
    header = f"User Activity Report ({report['time_period']})\n"
    header += f"From: {report['start_time']} to {report['end_time']}\n\n"

    # Format user activity
    user_activity = report.get("user_activity", {})

    user_section = ""
    for user, activities in sorted(
        user_activity.items(), key=lambda x: sum(x[1].values()), reverse=True
    ):
        user_section += f"User: {user}\n"

        activity_table = []
        for event_type, count in sorted(
            activities.items(), key=lambda x: x[1], reverse=True
        ):
            activity_table.append([event_type, count])

        user_section += format_table(activity_table, ["Event Type", "Count"])
        user_section += "\n"

    if not user_activity:
        user_section = "No user activity recorded.\n"

    return header + user_section


def main():
    """Main entry point for audit report generation."""
    parser = argparse.ArgumentParser(description="Homeostasis Audit Report Generator")

    # Define arguments
    parser.add_argument("--log-file", "-l", type=str, help="Path to audit log file")

    parser.add_argument(
        "--period",
        "-p",
        type=str,
        choices=["hour", "day", "week", "month"],
        default="day",
        help="Time period for the report",
    )

    parser.add_argument(
        "--format",
        "-f",
        type=str,
        choices=["text", "json", "csv"],
        default="text",
        help="Output format",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output file path (if not provided, prints to stdout)",
    )

    parser.add_argument(
        "--report-type",
        "-r",
        type=str,
        choices=["summary", "user", "healing", "security", "full", "export"],
        default="summary",
        help="Type of report to generate",
    )

    parser.add_argument(
        "--export-format",
        type=str,
        choices=["json", "csv"],
        default="json",
        help="Format for data export (only for --report-type=export)",
    )

    args = parser.parse_args()

    # Configure the audit monitor
    config = {}
    if args.log_file:
        config["log_file"] = args.log_file

    # Initialize the audit monitor
    monitor = get_audit_monitor(config)

    # Ensure we have the latest data
    monitor.read_new_events()

    # Generate the appropriate report
    if args.report_type == "export":
        # Export the raw data
        if args.output:
            success = export_audit_data(args.output, args.export_format)
            if success:
                print(
                    f"Exported audit data to {args.output} in {args.export_format} format"
                )
            else:
                print(f"Failed to export audit data to {args.output}")
        else:
            print("Error: --output is required for export report type")
            sys.exit(1)

        sys.exit(0)

    # Generate activity report for the specified period
    report = generate_activity_report(args.period)

    # Format the report based on report type and format
    if args.format == "json":
        if args.report_type == "summary":
            output = json.dumps(
                {
                    "summary": {
                        "time_period": report["time_period"],
                        "start_time": report["start_time"],
                        "end_time": report["end_time"],
                        "total_events": report["total_events"],
                        "event_counts": report["event_counts"],
                        "healing_activities": report["healing_activities"],
                    }
                },
                indent=2,
            )
        elif args.report_type == "user":
            output = json.dumps({"user_activity": report["user_activity"]}, indent=2)
        elif args.report_type == "healing":
            output = json.dumps(
                {"healing_activities": report["healing_activities"]}, indent=2
            )
        elif args.report_type == "security":
            output = json.dumps(
                {
                    "security_events": [
                        e
                        for e in report.get("error_events", [])
                        if "security" in e.get("event_type", "")
                    ]
                },
                indent=2,
            )
        else:  # full
            output = json.dumps(report, indent=2)
    elif args.format == "csv":
        if args.output:
            if args.report_type == "summary":
                with open(args.output, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Event Type", "Count"])
                    for event_type, count in report["event_counts"].items():
                        writer.writerow([event_type, count])
            elif args.report_type == "user":
                with open(args.output, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["User", "Event Type", "Count"])
                    for user, activities in report["user_activity"].items():
                        for event_type, count in activities.items():
                            writer.writerow([user, event_type, count])
            elif args.report_type == "healing":
                with open(args.output, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Activity", "Count"])
                    for activity, count in report["healing_activities"].items():
                        writer.writerow([activity, count])
            elif args.report_type == "security":
                with open(args.output, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Timestamp", "Event Type", "User", "Details"])
                    for event in report.get("error_events", []):
                        if "security" in event.get("event_type", ""):
                            writer.writerow(
                                [
                                    event.get("timestamp", ""),
                                    event.get("event_type", ""),
                                    event.get("user", ""),
                                    str(event.get("details", {})),
                                ]
                            )
            else:  # full
                with open(args.output, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Event Type", "Count"])
                    for event_type, count in report["event_counts"].items():
                        writer.writerow([event_type, count])

                    writer.writerow([])
                    writer.writerow(["User", "Event Type", "Count"])
                    for user, activities in report["user_activity"].items():
                        for event_type, count in activities.items():
                            writer.writerow([user, event_type, count])

                    writer.writerow([])
                    writer.writerow(["Activity", "Count"])
                    for activity, count in report["healing_activities"].items():
                        writer.writerow([activity, count])

                    writer.writerow([])
                    writer.writerow(["Timestamp", "Event Type", "User", "Details"])
                    for event in report.get("error_events", []):
                        writer.writerow(
                            [
                                event.get("timestamp", ""),
                                event.get("event_type", ""),
                                event.get("user", ""),
                                str(event.get("details", {})),
                            ]
                        )

            print(f"Report saved to {args.output}")
            sys.exit(0)
        else:
            print("Error: --output is required for CSV format")
            sys.exit(1)
    else:  # text
        if args.report_type == "summary":
            output = get_summary_report(report)
        elif args.report_type == "user":
            output = get_user_activity_report(report)
        elif args.report_type == "healing":
            healing = report["healing_activities"]
            healing_table = [
                ["Errors Detected", healing.get("errors_detected", 0)],
                ["Fixes Generated", healing.get("fixes_generated", 0)],
                ["Fixes Deployed", healing.get("fixes_deployed", 0)],
                ["Fixes Approved", healing.get("fixes_approved", 0)],
                ["Fixes Rejected", healing.get("fixes_rejected", 0)],
                ["Success Rate", f"{healing.get('success_rate', 0) * 100:.1f}%"],
            ]

            output = f"Healing Activities Report ({report['time_period']})\n"
            output += f"From: {report['start_time']} to {report['end_time']}\n\n"
            output += format_table(healing_table, ["Activity", "Count"])
        elif args.report_type == "security":
            security_events = [
                e
                for e in report.get("error_events", [])
                if "security" in e.get("event_type", "")
            ]

            output = f"Security Events Report ({report['time_period']})\n"
            output += f"From: {report['start_time']} to {report['end_time']}\n\n"

            if security_events:
                security_table = []
                for event in security_events:
                    timestamp = event.get("timestamp", "")
                    event_type = event.get("event_type", "")
                    user = event.get("user", "")
                    details_str = str(event.get("details", {}))
                    if len(details_str) > 50:
                        details_str = details_str[:47] + "..."
                    security_table.append([timestamp, event_type, user, details_str])

                output += format_table(
                    security_table, ["Timestamp", "Event Type", "User", "Details"]
                )
            else:
                output += "No security events recorded."
        else:  # full
            summary = get_summary_report(report)
            user_activity = get_user_activity_report(report)

            output = summary + "\n\n" + user_activity

    # Output the report
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Report saved to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
