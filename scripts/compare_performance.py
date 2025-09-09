#!/usr/bin/env python3
"""
Compare performance between baseline and current runs.

This script generates a comparison report showing performance changes
between two performance test runs.
"""
import argparse
import json
import sqlite3
import statistics
from datetime import datetime
from typing import Dict, List, Tuple


def load_metrics(db_path: str, test_name: str = None) -> List[Dict]:
    """Load metrics from database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    query = "SELECT * FROM metrics"
    params = []
    if test_name:
        query += " WHERE name = ?"
        params.append(test_name)
    query += " ORDER BY timestamp DESC LIMIT 100"

    cursor.execute(query, params)
    columns = [desc[0] for desc in cursor.description]

    metrics = []
    for row in cursor.fetchall():
        metric = dict(zip(columns, row))
        if metric.get("environment"):
            metric["environment"] = json.loads(metric["environment"])
        if metric.get("metadata"):
            metric["metadata"] = json.loads(metric["metadata"])
        metrics.append(metric)

    conn.close()
    return metrics


def load_baselines(db_path: str) -> Dict[str, Dict]:
    """Load all baselines from database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM baselines")
    columns = [desc[0] for desc in cursor.description]

    baselines = {}
    for row in cursor.fetchall():
        baseline = dict(zip(columns, row))
        baselines[baseline["name"]] = baseline

    conn.close()
    return baselines


def calculate_change(baseline: float, current: float) -> Tuple[float, str]:
    """Calculate percentage change and direction."""
    if baseline == 0:
        return 0, "→"

    change = ((current - baseline) / baseline) * 100

    if abs(change) < 1:
        direction = "→"  # No significant change
    elif change > 0:
        direction = "↑"  # Slower/worse
    else:
        direction = "↓"  # Faster/better

    return change, direction


def generate_comparison_report(baseline_db: str, current_db: str) -> str:
    """Generate markdown comparison report."""
    baseline_data = load_baselines(baseline_db)
    current_metrics = {}

    # Group current metrics by test name
    all_current = load_metrics(current_db)
    for metric in all_current:
        name = metric["name"]
        if name not in current_metrics:
            current_metrics[name] = []
        current_metrics[name].append(metric)

    report = "# Performance Comparison Report\n\n"
    report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    # Summary table
    report += "## Summary\n\n"
    report += "| Test | Duration | Memory | CPU | Status |\n"
    report += "|------|----------|--------|-----|--------|\n"

    regressions = []
    improvements = []

    for test_name in sorted(
        set(list(baseline_data.keys()) + list(current_metrics.keys()))
    ):
        baseline = baseline_data.get(test_name)
        current = current_metrics.get(test_name, [])

        if not baseline and not current:
            continue

        if not current:
            report += f"| {test_name} | ⚠️ No data | - | - | Missing |\n"
            continue

        # Calculate current averages
        current_durations = [m["duration"] for m in current]
        current_memories = [m["memory_delta"] for m in current]
        current_cpus = [m["cpu_percent"] for m in current]

        avg_duration = statistics.mean(current_durations)
        avg_memory = statistics.mean(current_memories)
        avg_cpu = statistics.mean(current_cpus)

        if not baseline:
            report += f"| {test_name} | {avg_duration*1000:.1f}ms | {avg_memory:.1f}MB | {avg_cpu:.1f}% | New |\n"
            continue

        # Calculate changes
        duration_change, duration_dir = calculate_change(
            baseline["mean_duration"], avg_duration
        )
        memory_change, memory_dir = calculate_change(
            baseline["mean_memory"], avg_memory
        )
        cpu_change, cpu_dir = calculate_change(baseline["mean_cpu"], avg_cpu)

        # Determine status
        if duration_change > 20 or memory_change > 30:
            status = "⚠️ Regression"
            regressions.append(
                {
                    "test": test_name,
                    "duration_change": duration_change,
                    "memory_change": memory_change,
                }
            )
        elif duration_change < -10:
            status = "✅ Improved"
            improvements.append({"test": test_name, "duration_change": duration_change})
        else:
            status = "→ Stable"

        # Format row
        duration_str = (
            f"{avg_duration*1000:.1f}ms {duration_dir} {duration_change:+.1f}%"
        )
        memory_str = f"{avg_memory:.1f}MB {memory_dir} {memory_change:+.1f}%"
        cpu_str = f"{avg_cpu:.1f}% {cpu_dir} {cpu_change:+.1f}%"

        report += (
            f"| {test_name} | {duration_str} | {memory_str} | {cpu_str} | {status} |\n"
        )

    # Detailed sections
    if regressions:
        report += "\n## ⚠️ Performance Regressions\n\n"
        for reg in regressions:
            report += f"- **{reg['test']}**: Duration {reg['duration_change']:+.1f}%, Memory {reg['memory_change']:+.1f}%\n"

    if improvements:
        report += "\n## ✅ Performance Improvements\n\n"
        for imp in improvements:
            report += (
                f"- **{imp['test']}**: Duration {imp['duration_change']:.1f}% faster\n"
            )

    # Recommendations
    report += "\n## Recommendations\n\n"
    if regressions:
        report += "- Investigate the performance regressions listed above\n"
        report += "- Consider profiling the affected code paths\n"
        report += "- Review recent changes that might have impacted performance\n"
    else:
        report += "- No significant performance regressions detected\n"
        report += "- Continue monitoring performance metrics\n"

    return report


def generate_json_report(baseline_db: str, current_db: str, output_path: str):
    """Generate JSON report for programmatic consumption."""
    baseline_data = load_baselines(baseline_db)
    current_metrics = {}

    # Group current metrics by test name
    all_current = load_metrics(current_db)
    for metric in all_current:
        name = metric["name"]
        if name not in current_metrics:
            current_metrics[name] = []
        current_metrics[name].append(metric)

    report = {
        "timestamp": datetime.now().isoformat(),
        "tests": {},
        "regressions": [],
        "improvements": [],
    }

    for test_name in sorted(
        set(list(baseline_data.keys()) + list(current_metrics.keys()))
    ):
        baseline = baseline_data.get(test_name)
        current = current_metrics.get(test_name, [])

        if not current:
            continue

        # Calculate current averages
        current_durations = [m["duration"] for m in current]
        current_memories = [m["memory_delta"] for m in current]
        current_cpus = [m["cpu_percent"] for m in current]

        test_data = {
            "current": {
                "duration": statistics.mean(current_durations),
                "memory": statistics.mean(current_memories),
                "cpu": statistics.mean(current_cpus),
                "samples": len(current),
            }
        }

        if baseline:
            test_data["baseline"] = {
                "duration": baseline["mean_duration"],
                "memory": baseline["mean_memory"],
                "cpu": baseline["mean_cpu"],
                "samples": baseline["sample_count"],
            }

            # Calculate changes
            duration_change = (
                (test_data["current"]["duration"] - baseline["mean_duration"])
                / baseline["mean_duration"]
                * 100
            )

            test_data["changes"] = {
                "duration": duration_change,
                "memory": (
                    (test_data["current"]["memory"] - baseline["mean_memory"])
                    / (baseline["mean_memory"] + 0.001)
                    * 100
                ),
                "cpu": (
                    (test_data["current"]["cpu"] - baseline["mean_cpu"])
                    / (baseline["mean_cpu"] + 0.001)
                    * 100
                ),
            }

            # Categorize
            if duration_change > 20:
                report["regressions"].append(
                    {
                        "test_name": test_name,
                        "metric_type": "duration",
                        "baseline_value": baseline["mean_duration"],
                        "current_value": test_data["current"]["duration"],
                        "regression_factor": test_data["current"]["duration"]
                        / baseline["mean_duration"],
                    }
                )
            elif duration_change < -10:
                report["improvements"].append(
                    {"test_name": test_name, "improvement_percent": -duration_change}
                )

        report["tests"][test_name] = test_data

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Compare performance test results")
    parser.add_argument("--baseline", required=True, help="Path to baseline database")
    parser.add_argument("--current", required=True, help="Path to current database")
    parser.add_argument(
        "--output", default="performance_comparison.md", help="Output file path"
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "json", "both"],
        default="markdown",
        help="Output format",
    )

    args = parser.parse_args()

    if args.format in ["markdown", "both"]:
        report = generate_comparison_report(args.baseline, args.current)

        output_path = args.output
        if args.format == "both":
            output_path = (
                args.output.replace(".md", ".md")
                if args.output.endswith(".md")
                else args.output + ".md"
            )

        with open(output_path, "w") as f:
            f.write(report)

        print(f"Markdown report written to {output_path}")

    if args.format in ["json", "both"]:
        json_path = (
            args.output.replace(".md", ".json")
            if args.output.endswith(".md")
            else args.output + ".json"
        )
        generate_json_report(args.baseline, args.current, json_path)
        print(f"JSON report written to {json_path}")


if __name__ == "__main__":
    main()
