#!/usr/bin/env python3
"""
Consolidate security reports from multiple scanners into a unified report.
"""

import argparse
import glob
import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


class SecurityReportConsolidator:
    """Consolidates security reports from various scanners."""

    def __init__(self):
        self.reports = {}
        self.vulnerabilities = []

    def load_reports(self, report_pattern: str = "*-report.json"):
        """Load all security reports matching the pattern."""
        for report_file in glob.glob(report_pattern):
            try:
                with open(report_file, "r") as f:
                    report_name = Path(report_file).stem.replace("-report", "")
                    self.reports[report_name] = json.load(f)
                    print(f"Loaded report: {report_name}")
            except Exception as e:
                print(f"Error loading {report_file}: {e}")

    def parse_bandit_report(self, report: Dict[str, Any]):
        """Parse Bandit Python security report."""
        for result in report.get("results", []):
            self.vulnerabilities.append(
                {
                    "tool": "bandit",
                    "severity": result.get("issue_severity", "medium").lower(),
                    "category": "static_code",
                    "title": result.get("issue_text", ""),
                    "file": result.get("filename", ""),
                    "line": result.get("line_number", 0),
                    "confidence": result.get("issue_confidence", "medium").lower(),
                    "cwe": result.get("issue_cwe", {}).get("id", ""),
                }
            )

    def parse_safety_report(self, report: List[Dict[str, Any]]):
        """Parse Safety Python dependency report."""
        for vuln in report:
            self.vulnerabilities.append(
                {
                    "tool": "safety",
                    "severity": "high",  # Safety doesn't provide severity
                    "category": "dependency",
                    "title": vuln.get("advisory", ""),
                    "package": f"{vuln.get('package', '')}=={vuln.get('installed_version', '')}",
                    "cve": vuln.get("cve", ""),
                    "recommendation": f"Update to {vuln.get('safe_version', 'latest')}",
                }
            )

    def parse_npm_audit_report(self, report: Dict[str, Any]):
        """Parse NPM audit report."""
        for advisory_id, advisory in report.get("advisories", {}).items():
            self.vulnerabilities.append(
                {
                    "tool": "npm-audit",
                    "severity": advisory.get("severity", "medium"),
                    "category": "dependency",
                    "title": advisory.get("title", ""),
                    "package": advisory.get("module_name", ""),
                    "cve": advisory.get("cves", []),
                    "recommendation": advisory.get("recommendation", ""),
                }
            )

    def parse_gosec_report(self, report: Dict[str, Any]):
        """Parse GoSec Go security report."""
        for issue in report.get("Issues", []):
            self.vulnerabilities.append(
                {
                    "tool": "gosec",
                    "severity": issue.get("severity", "medium").lower(),
                    "category": "static_code",
                    "title": issue.get("details", ""),
                    "file": issue.get("file", ""),
                    "line": issue.get("line", ""),
                    "confidence": issue.get("confidence", "medium").lower(),
                    "cwe": issue.get("cwe", {}).get("id", ""),
                }
            )

    def parse_hadolint_report(self, report: List[Dict[str, Any]]):
        """Parse Hadolint Dockerfile report."""
        for issue in report:
            severity_map = {"error": "high", "warning": "medium", "info": "low"}
            self.vulnerabilities.append(
                {
                    "tool": "hadolint",
                    "severity": severity_map.get(issue.get("level", "info"), "low"),
                    "category": "container",
                    "title": f"{issue.get('code', '')}: {issue.get('message', '')}",
                    "file": issue.get("file", ""),
                    "line": issue.get("line", 0),
                }
            )

    def parse_checkov_report(self, report: Dict[str, Any]):
        """Parse Checkov IaC report."""
        for check in report.get("results", {}).get("failed_checks", []):
            self.vulnerabilities.append(
                {
                    "tool": "checkov",
                    "severity": "medium",  # Checkov doesn't provide severity
                    "category": "iac",
                    "title": check.get("check_name", ""),
                    "file": check.get("file_path", ""),
                    "line": check.get("file_line_range", [0])[0],
                    "resource": check.get("resource", ""),
                    "guideline": check.get("guideline", ""),
                }
            )

    def consolidate(self):
        """Consolidate all loaded reports."""
        # Parse different report formats
        for report_name, report_data in self.reports.items():
            if "bandit" in report_name:
                self.parse_bandit_report(report_data)
            elif "safety" in report_name or "python-vuln" in report_name:
                if isinstance(report_data, list):
                    self.parse_safety_report(report_data)
            elif "npm" in report_name:
                self.parse_npm_audit_report(report_data)
            elif "gosec" in report_name:
                self.parse_gosec_report(report_data)
            elif "hadolint" in report_name:
                if isinstance(report_data, list):
                    self.parse_hadolint_report(report_data)
            elif "checkov" in report_name:
                self.parse_checkov_report(report_data)

    def generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        summary = {
            "total_vulnerabilities": len(self.vulnerabilities),
            "by_severity": {"critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0},
            "by_category": {},
            "by_tool": {},
            "scan_date": datetime.now().isoformat(),
        }

        for vuln in self.vulnerabilities:
            # Count by severity
            severity = vuln.get("severity", "medium").lower()
            if severity in summary["by_severity"]:
                summary["by_severity"][severity] += 1

            # Count by category
            category = vuln.get("category", "unknown")
            summary["by_category"][category] = (
                summary["by_category"].get(category, 0) + 1
            )

            # Count by tool
            tool = vuln.get("tool", "unknown")
            summary["by_tool"][tool] = summary["by_tool"].get(tool, 0) + 1

        return summary

    def filter_by_severity(self, min_severity: str) -> List[Dict[str, Any]]:
        """Filter vulnerabilities by minimum severity."""
        severity_order = {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
        min_level = severity_order.get(min_severity.lower(), 0)

        filtered = []
        for vuln in self.vulnerabilities:
            vuln_level = severity_order.get(vuln.get("severity", "medium").lower(), 2)
            if vuln_level >= min_level:
                filtered.append(vuln)

        return filtered

    def generate_html_report(self, severity_threshold: str = "low") -> str:
        """Generate HTML report."""
        summary = self.generate_summary()
        filtered_vulns = self.filter_by_severity(severity_threshold)

        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Security Scan Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1, h2, h3 {{
            color: #333;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            border: 1px solid #e9ecef;
        }}
        .stat-card h3 {{
            margin: 0 0 10px 0;
            color: #666;
            font-size: 14px;
            text-transform: uppercase;
        }}
        .stat-value {{
            font-size: 32px;
            font-weight: bold;
            color: #333;
        }}
        .severity-critical {{ color: #d73a49; }}
        .severity-high {{ color: #e36209; }}
        .severity-medium {{ color: #f9c513; }}
        .severity-low {{ color: #28a745; }}
        .severity-info {{ color: #0366d6; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e9ecef;
        }}
        th {{
            background-color: #f8f9fa;
            font-weight: 600;
        }}
        tr:hover {{
            background-color: #f8f9fa;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
        }}
        .badge-critical {{ background: #fee; color: #d73a49; }}
        .badge-high {{ background: #fff5eb; color: #e36209; }}
        .badge-medium {{ background: #fffbdd; color: #b08800; }}
        .badge-low {{ background: #e6ffed; color: #22863a; }}
        .badge-info {{ background: #e1f5ff; color: #0366d6; }}
        .file-path {{
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 13px;
            color: #666;
        }}
        .chart {{
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔒 Security Scan Report</h1>
        <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>Summary</h2>
        <div class="summary">
            <div class="stat-card">
                <h3>Total Vulnerabilities</h3>
                <div class="stat-value">{summary['total_vulnerabilities']}</div>
            </div>
            <div class="stat-card">
                <h3>Critical</h3>
                <div class="stat-value severity-critical">{summary['by_severity']['critical']}</div>
            </div>
            <div class="stat-card">
                <h3>High</h3>
                <div class="stat-value severity-high">{summary['by_severity']['high']}</div>
            </div>
            <div class="stat-card">
                <h3>Medium</h3>
                <div class="stat-value severity-medium">{summary['by_severity']['medium']}</div>
            </div>
            <div class="stat-card">
                <h3>Low</h3>
                <div class="stat-value severity-low">{summary['by_severity']['low']}</div>
            </div>
        </div>
        
        <h2>Vulnerabilities by Category</h2>
        <table>
            <thead>
                <tr>
                    <th>Category</th>
                    <th>Count</th>
                </tr>
            </thead>
            <tbody>
                {''.join(f'<tr><td>{cat}</td><td>{count}</td></tr>' for cat, count in summary['by_category'].items())}
            </tbody>
        </table>
        
        <h2>Detailed Findings (Severity ≥ {severity_threshold})</h2>
        <table>
            <thead>
                <tr>
                    <th>Severity</th>
                    <th>Tool</th>
                    <th>Title</th>
                    <th>Location</th>
                    <th>Details</th>
                </tr>
            </thead>
            <tbody>
"""

        for vuln in sorted(
            filtered_vulns,
            key=lambda x: ["critical", "high", "medium", "low", "info"].index(
                x.get("severity", "medium").lower()
            ),
        ):
            severity = vuln.get("severity", "medium").lower()
            location = ""
            if vuln.get("file"):
                location = f"<span class='file-path'>{html.escape(vuln['file'])}"
                if vuln.get("line"):
                    location += f":{vuln['line']}"
                location += "</span>"
            elif vuln.get("package"):
                location = (
                    f"<span class='file-path'>{html.escape(vuln['package'])}</span>"
                )

            details = []
            if vuln.get("cve"):
                details.append(f"CVE: {vuln['cve']}")
            if vuln.get("cwe"):
                details.append(f"CWE: {vuln['cwe']}")
            if vuln.get("recommendation"):
                details.append(f"Fix: {vuln['recommendation']}")

            html_content += f"""
                <tr>
                    <td><span class="badge badge-{severity}">{severity}</span></td>
                    <td>{vuln.get('tool', 'unknown')}</td>
                    <td>{html.escape(vuln.get('title', 'Unknown'))}</td>
                    <td>{location}</td>
                    <td>{html.escape(' | '.join(details))}</td>
                </tr>
"""

        html_content += """
            </tbody>
        </table>
    </div>
</body>
</html>
"""

        return html_content

    def generate_json_report(self) -> Dict[str, Any]:
        """Generate JSON report."""
        return {
            "summary": self.generate_summary(),
            "vulnerabilities": self.vulnerabilities,
        }


def main():
    parser = argparse.ArgumentParser(description="Consolidate security scan reports")
    parser.add_argument("--output", "-o", required=True, help="Output file path")
    parser.add_argument(
        "--format", "-f", choices=["html", "json"], default="json", help="Output format"
    )
    parser.add_argument(
        "--severity-threshold",
        "-s",
        default="low",
        choices=["info", "low", "medium", "high", "critical"],
        help="Minimum severity to include in report",
    )
    parser.add_argument(
        "--report-pattern",
        "-p",
        default="*-report.json",
        help="Pattern to match report files",
    )

    args = parser.parse_args()

    consolidator = SecurityReportConsolidator()
    consolidator.load_reports(args.report_pattern)
    consolidator.consolidate()

    if args.format == "html":
        report = consolidator.generate_html_report(args.severity_threshold)
        with open(args.output, "w") as f:
            f.write(report)
    else:
        report = consolidator.generate_json_report()
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)

    print(f"Report generated: {args.output}")

    # Exit with error if critical vulnerabilities found
    summary = consolidator.generate_summary()
    if summary["by_severity"]["critical"] > 0:
        print(
            f"❌ Critical vulnerabilities found: {summary['by_severity']['critical']}"
        )
        exit(1)


if __name__ == "__main__":
    main()
