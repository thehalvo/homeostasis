#!/usr/bin/env python3
"""
Update security baselines with latest scan results.

This script updates the security baseline file with results from the latest
security scan, helping track improvements and regressions over time.
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any


def load_json_file(file_path: Path) -> Dict[str, Any]:
    """Load JSON data from file."""
    if not file_path.exists():
        return {}
    
    with open(file_path, 'r') as f:
        return json.load(f)


def update_baselines(results_file: Path, baseline_file: Path) -> None:
    """Update security baselines with new scan results."""
    # Load current results
    results = load_json_file(results_file)
    if not results:
        print(f"Error: No results found in {results_file}")
        sys.exit(1)
    
    # Load existing baselines
    baselines = load_json_file(baseline_file)
    
    # Initialize baseline structure if empty
    if not baselines:
        baselines = {
            "version": "1.0",
            "last_updated": None,
            "history": [],
            "current_baseline": {},
            "known_vulnerabilities": {},
            "exceptions": []
        }
    
    # Update baseline with current results
    current_time = datetime.now().isoformat()
    
    # Store previous baseline in history
    if baselines.get("current_baseline"):
        baselines["history"].append({
            "timestamp": baselines.get("last_updated", current_time),
            "baseline": baselines["current_baseline"]
        })
    
    # Update current baseline
    baselines["last_updated"] = current_time
    baselines["current_baseline"] = {
        "timestamp": current_time,
        "summary": results.get("summary", {}),
        "by_severity": results.get("summary", {}).get("by_severity", {}),
        "by_category": results.get("summary", {}).get("by_category", {}),
        "total_vulnerabilities": results.get("summary", {}).get("total_vulnerabilities", 0)
    }
    
    # Update known vulnerabilities
    if "vulnerabilities" in results:
        for vuln in results["vulnerabilities"]:
            vuln_id = vuln.get("id")
            if vuln_id and vuln_id not in baselines["known_vulnerabilities"]:
                baselines["known_vulnerabilities"][vuln_id] = {
                    "first_seen": current_time,
                    "last_seen": current_time,
                    "severity": vuln.get("severity"),
                    "category": vuln.get("category"),
                    "title": vuln.get("title"),
                    "status": "active"
                }
            elif vuln_id:
                baselines["known_vulnerabilities"][vuln_id]["last_seen"] = current_time
    
    # Mark vulnerabilities as resolved if not in current results
    current_vuln_ids = {v.get("id") for v in results.get("vulnerabilities", []) if v.get("id")}
    for vuln_id, vuln_data in baselines["known_vulnerabilities"].items():
        if vuln_id not in current_vuln_ids and vuln_data.get("status") == "active":
            vuln_data["status"] = "resolved"
            vuln_data["resolved_at"] = current_time
    
    # Keep only recent history (last 30 entries)
    if len(baselines["history"]) > 30:
        baselines["history"] = baselines["history"][-30:]
    
    # Save updated baselines
    with open(baseline_file, 'w') as f:
        json.dump(baselines, f, indent=2)
    
    print(f"Successfully updated security baselines at {baseline_file}")
    print(f"Current vulnerabilities: {baselines['current_baseline']['total_vulnerabilities']}")
    
    # Check for improvements or regressions
    if baselines["history"]:
        prev_total = baselines["history"][-1]["baseline"].get("total_vulnerabilities", 0)
        curr_total = baselines["current_baseline"]["total_vulnerabilities"]
        
        if curr_total < prev_total:
            print(f"✅ Improvement: {prev_total - curr_total} vulnerabilities resolved")
        elif curr_total > prev_total:
            print(f"⚠️  Regression: {curr_total - prev_total} new vulnerabilities detected")
        else:
            print("No change in vulnerability count")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Update security baselines with latest scan results"
    )
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Path to security scan results JSON file"
    )
    parser.add_argument(
        "--baseline-file",
        type=Path,
        required=True,
        help="Path to security baseline JSON file"
    )
    
    args = parser.parse_args()
    
    try:
        update_baselines(args.results, args.baseline_file)
    except Exception as e:
        print(f"Error updating baselines: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()