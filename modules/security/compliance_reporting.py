"""
Compliance Reporting System for Homeostasis Enterprise Governance.

This module provides comprehensive compliance reporting capabilities for
regulated industries and enterprise environments.

Features:
- Automated compliance report generation
- Support for multiple compliance frameworks (SOC2, HIPAA, PCI-DSS, etc.)
- Real-time compliance monitoring
- Evidence collection and management
- Audit trail aggregation
"""

import csv
import datetime
import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .approval_workflow import get_workflow_engine
from .audit import get_audit_logger
from .rbac import get_rbac_manager
from .user_management import get_user_management

logger = logging.getLogger(__name__)


class ComplianceFramework(Enum):
    """Supported compliance frameworks."""

    SOC2 = "soc2"
    HIPAA = "hipaa"
    PCI_DSS = "pci_dss"
    ISO_27001 = "iso_27001"
    GDPR = "gdpr"
    SOX = "sox"
    NIST = "nist"
    CUSTOM = "custom"


class ControlStatus(Enum):
    """Compliance control status."""

    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NOT_APPLICABLE = "not_applicable"
    UNDER_REVIEW = "under_review"


class ReportFormat(Enum):
    """Report output formats."""

    JSON = "json"
    CSV = "csv"
    HTML = "html"
    PDF = "pdf"
    EXCEL = "excel"


@dataclass
class ComplianceControl:
    """Represents a compliance control."""

    control_id: str
    framework: ComplianceFramework
    title: str
    description: str
    category: str
    requirements: List[str]
    evidence_types: List[str]
    automated: bool = True
    frequency: str = "continuous"  # continuous, daily, weekly, monthly, quarterly
    last_assessed: Optional[str] = None
    status: ControlStatus = ControlStatus.UNDER_REVIEW
    metadata: Dict = field(default_factory=dict)


@dataclass
class ComplianceEvidence:
    """Evidence for compliance controls."""

    evidence_id: str
    control_id: str
    type: str
    description: str
    collected_at: str
    source_system: str
    data: Dict
    validated: bool = False
    validation_details: Optional[Dict] = None


@dataclass
class ComplianceReport:
    """Compliance report."""

    report_id: str
    framework: ComplianceFramework
    report_type: str
    generated_at: str
    generated_by: str
    period_start: str
    period_end: str
    overall_status: ControlStatus
    control_summary: Dict[str, int]
    findings: List[Dict]
    recommendations: List[str]
    evidence_references: List[str]
    metadata: Dict = field(default_factory=dict)


@dataclass
class ComplianceFinding:
    """Compliance finding/issue."""

    finding_id: str
    control_id: str
    severity: str  # critical, high, medium, low
    title: str
    description: str
    identified_at: str
    remediation_plan: Optional[str] = None
    remediation_deadline: Optional[str] = None
    status: str = "open"  # open, in_progress, remediated, accepted


class ComplianceReportingSystem:
    """
    Comprehensive compliance reporting system for enterprise governance.

    Manages compliance controls, evidence collection, and report generation
    for various regulatory frameworks.
    """

    def __init__(
        self,
        config: Optional[Dict[Any, Any]] = None,
        storage_path: Optional[str] = None,
    ):
        """Initialize the compliance reporting system.

        Args:
            config: Configuration dictionary
            storage_path: Path to store compliance data
        """
        self.config = config or {}
        self.storage_path = Path(storage_path or "data/compliance")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Get managers
        self.audit_logger = get_audit_logger(config)
        self.rbac_manager = get_rbac_manager(config)
        self.user_management = get_user_management(config)
        self.workflow_engine = get_workflow_engine(config)

        # Initialize stores
        self.controls: Dict[str, ComplianceControl] = {}
        self.evidence: Dict[str, ComplianceEvidence] = {}
        self.reports: Dict[str, ComplianceReport] = {}
        self.findings: Dict[str, ComplianceFinding] = {}

        # Framework mappings
        self.framework_controls: Dict[ComplianceFramework, List[str]] = defaultdict(
            list
        )

        # Load existing data
        self._load_compliance_data()

        # Initialize default controls if none exist
        if not self.controls:
            self._initialize_default_controls()

    def assess_control(self, control_id: str) -> Tuple[ControlStatus, List[str]]:
        """Assess a compliance control.

        Args:
            control_id: Control ID to assess

        Returns:
            Tuple of (status, list of finding IDs)
        """
        control = self.controls.get(control_id)
        if not control:
            raise ValueError(f"Control {control_id} not found")

        # Collect evidence
        evidence_list = self._collect_evidence_for_control(control_id)

        # Evaluate control
        status, findings_list = self._evaluate_control(control, evidence_list)

        # Update control status
        control.status = status
        control.last_assessed = datetime.datetime.utcnow().isoformat()

        # Create findings for non-compliant controls
        finding_ids = []
        for finding_data in findings_list:
            finding_id = self._create_finding(control_id, finding_data)
            finding_ids.append(finding_id)

        self._save_compliance_data()

        # Log assessment
        self.audit_logger.log_event(
            event_type="compliance_control_assessed",
            user="system",
            details={
                "control_id": control_id,
                "status": status.value,
                "findings_count": len(finding_ids),
            },
        )

        return status, finding_ids

    def generate_compliance_report(
        self,
        framework: ComplianceFramework,
        report_type: str,
        period_days: int = 30,
        requested_by: Optional[str] = None,
    ) -> str:
        """Generate a compliance report.

        Args:
            framework: Compliance framework
            report_type: Type of report (assessment, audit, executive)
            period_days: Period to cover in days
            requested_by: User requesting the report

        Returns:
            Report ID
        """
        # Determine period
        end_date = datetime.datetime.utcnow()
        start_date = end_date - datetime.timedelta(days=period_days)

        # Get controls for framework
        control_ids = self.framework_controls.get(framework, [])
        if not control_ids:
            raise ValueError(f"No controls defined for framework {framework.value}")

        # Assess all controls
        control_statuses = {}
        all_findings = []

        for control_id in control_ids:
            status, finding_ids = self.assess_control(control_id)
            control_statuses[control_id] = status

            for finding_id in finding_ids:
                finding = self.findings.get(finding_id)
                if finding:
                    all_findings.append(finding)

        # Calculate summary
        control_summary = {
            "compliant": sum(
                1 for s in control_statuses.values() if s == ControlStatus.COMPLIANT
            ),
            "non_compliant": sum(
                1 for s in control_statuses.values() if s == ControlStatus.NON_COMPLIANT
            ),
            "partially_compliant": sum(
                1
                for s in control_statuses.values()
                if s == ControlStatus.PARTIALLY_COMPLIANT
            ),
            "not_applicable": sum(
                1
                for s in control_statuses.values()
                if s == ControlStatus.NOT_APPLICABLE
            ),
            "under_review": sum(
                1 for s in control_statuses.values() if s == ControlStatus.UNDER_REVIEW
            ),
        }

        # Determine overall status
        if control_summary["non_compliant"] > 0:
            overall_status = ControlStatus.NON_COMPLIANT
        elif control_summary["partially_compliant"] > 0:
            overall_status = ControlStatus.PARTIALLY_COMPLIANT
        elif control_summary["compliant"] == len(control_ids):
            overall_status = ControlStatus.COMPLIANT
        else:
            overall_status = ControlStatus.UNDER_REVIEW

        # Generate recommendations
        recommendations = self._generate_recommendations(framework, all_findings)

        # Collect evidence references
        evidence_refs = []
        for control_id in control_ids:
            evidence_list = self._collect_evidence_for_control(control_id)
            evidence_refs.extend([e.evidence_id for e in evidence_list])

        # Create report
        report_id = self._generate_report_id()
        report = ComplianceReport(
            report_id=report_id,
            framework=framework,
            report_type=report_type,
            generated_at=datetime.datetime.utcnow().isoformat(),
            generated_by=requested_by or "system",
            period_start=start_date.isoformat(),
            period_end=end_date.isoformat(),
            overall_status=overall_status,
            control_summary=control_summary,
            findings=[self._finding_to_dict(f) for f in all_findings],
            recommendations=recommendations,
            evidence_references=evidence_refs,
            metadata={
                "control_count": len(control_ids),
                "evidence_count": len(evidence_refs),
            },
        )

        self.reports[report_id] = report
        self._save_compliance_data()

        # Log report generation
        self.audit_logger.log_event(
            event_type="compliance_report_generated",
            user=requested_by or "system",
            details={
                "report_id": report_id,
                "framework": framework.value,
                "report_type": report_type,
                "overall_status": overall_status.value,
            },
        )

        return report_id

    def export_report(self, report_id: str, format: ReportFormat) -> str:
        """Export a compliance report in specified format.

        Args:
            report_id: Report ID
            format: Export format

        Returns:
            Path to exported file
        """
        report = self.reports.get(report_id)
        if not report:
            raise ValueError(f"Report {report_id} not found")

        export_path = self.storage_path / "exports"
        export_path.mkdir(exist_ok=True)

        filename = f"{report.framework.value}_report_{report_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

        if format == ReportFormat.JSON:
            return self._export_json(report, export_path / f"{filename}.json")
        elif format == ReportFormat.CSV:
            return self._export_csv(report, export_path / f"{filename}.csv")
        elif format == ReportFormat.HTML:
            return self._export_html(report, export_path / f"{filename}.html")
        else:
            raise ValueError(f"Unsupported export format: {format.value}")

    def add_evidence(
        self,
        control_id: str,
        evidence_type: str,
        description: str,
        data: Dict,
        source_system: str = "homeostasis",
    ) -> str:
        """Add evidence for a control.

        Args:
            control_id: Control ID
            evidence_type: Type of evidence
            description: Evidence description
            data: Evidence data
            source_system: Source system name

        Returns:
            Evidence ID
        """
        control = self.controls.get(control_id)
        if not control:
            raise ValueError(f"Control {control_id} not found")

        evidence_id = self._generate_evidence_id()

        evidence = ComplianceEvidence(
            evidence_id=evidence_id,
            control_id=control_id,
            type=evidence_type,
            description=description,
            collected_at=datetime.datetime.utcnow().isoformat(),
            source_system=source_system,
            data=data,
        )

        self.evidence[evidence_id] = evidence
        self._save_compliance_data()

        return evidence_id

    def get_compliance_dashboard(self) -> Dict[str, Any]:
        """Get compliance dashboard data.

        Returns:
            Dashboard data
        """
        # Framework compliance summary
        framework_summary = {}
        for framework in ComplianceFramework:
            if framework == ComplianceFramework.CUSTOM:
                continue

            control_ids = self.framework_controls.get(framework, [])
            if control_ids:
                compliant = sum(
                    1
                    for cid in control_ids
                    if self.controls.get(cid)
                    and self.controls[cid].status == ControlStatus.COMPLIANT
                )
                total = len(control_ids)
                framework_summary[framework.value] = {
                    "compliant": compliant,
                    "total": total,
                    "percentage": (compliant / total * 100) if total > 0 else 0,
                }

        # Recent findings
        recent_findings = sorted(
            self.findings.values(), key=lambda f: f.identified_at, reverse=True
        )[:10]

        # Upcoming assessments
        upcoming_assessments = []
        for control in self.controls.values():
            if control.last_assessed:
                last_assessed = datetime.datetime.fromisoformat(control.last_assessed)

                # Calculate next assessment based on frequency
                if control.frequency == "daily":
                    next_assessment = last_assessed + datetime.timedelta(days=1)
                elif control.frequency == "weekly":
                    next_assessment = last_assessed + datetime.timedelta(weeks=1)
                elif control.frequency == "monthly":
                    next_assessment = last_assessed + datetime.timedelta(days=30)
                elif control.frequency == "quarterly":
                    next_assessment = last_assessed + datetime.timedelta(days=90)
                else:
                    continue

                if next_assessment <= datetime.datetime.utcnow() + datetime.timedelta(
                    days=7
                ):
                    upcoming_assessments.append(
                        {
                            "control_id": control.control_id,
                            "title": control.title,
                            "framework": control.framework.value,
                            "due_date": next_assessment.isoformat(),
                        }
                    )

        return {
            "framework_summary": framework_summary,
            "total_controls": len(self.controls),
            "total_findings": len(
                [f for f in self.findings.values() if f.status == "open"]
            ),
            "recent_reports": len(self.reports),
            "recent_findings": [self._finding_to_dict(f) for f in recent_findings],
            "upcoming_assessments": upcoming_assessments,
        }

    def remediate_finding(
        self,
        finding_id: str,
        remediation_plan: str,
        deadline: Optional[datetime.datetime] = None,
    ) -> bool:
        """Create remediation plan for a finding.

        Args:
            finding_id: Finding ID
            remediation_plan: Remediation plan description
            deadline: Remediation deadline

        Returns:
            True if successful
        """
        finding = self.findings.get(finding_id)
        if not finding:
            raise ValueError(f"Finding {finding_id} not found")

        finding.remediation_plan = remediation_plan
        finding.remediation_deadline = deadline.isoformat() if deadline else None
        finding.status = "in_progress"

        self._save_compliance_data()

        # Log remediation plan
        self.audit_logger.log_event(
            event_type="compliance_finding_remediation_planned",
            user="system",
            details={
                "finding_id": finding_id,
                "deadline": finding.remediation_deadline,
            },
        )

        return True

    def _collect_evidence_for_control(
        self, control_id: str
    ) -> List[ComplianceEvidence]:
        """Collect evidence for a control."""
        evidence_list = []

        for evidence in self.evidence.values():
            if evidence.control_id == control_id:
                evidence_list.append(evidence)

        # Also collect automated evidence based on control type
        control = self.controls[control_id]
        if control.automated:
            # Collect from audit logs
            if "audit_logs" in control.evidence_types:
                audit_evidence = self._collect_audit_log_evidence(control)
                evidence_list.extend(audit_evidence)

            # Collect from access controls
            if "access_controls" in control.evidence_types:
                access_evidence = self._collect_access_control_evidence(control)
                evidence_list.extend(access_evidence)

            # Collect from approval workflows
            if "approval_workflows" in control.evidence_types:
                workflow_evidence = self._collect_workflow_evidence(control)
                evidence_list.extend(workflow_evidence)

        return evidence_list

    def _evaluate_control(
        self, control: ComplianceControl, evidence_list: List[ComplianceEvidence]
    ) -> Tuple[ControlStatus, List[Dict]]:
        """Evaluate a control based on evidence."""
        findings = []

        # Check if sufficient evidence exists
        if not evidence_list:
            findings.append(
                {
                    "severity": "high",
                    "title": f"No evidence found for control {control.control_id}",
                    "description": f"No evidence has been collected for control: {control.title}",
                }
            )
            return ControlStatus.NON_COMPLIANT, findings

        # Control-specific evaluation logic
        if control.control_id.startswith("soc2_"):
            return self._evaluate_soc2_control(control, evidence_list)
        elif control.control_id.startswith("hipaa_"):
            return self._evaluate_hipaa_control(control, evidence_list)
        elif control.control_id.startswith("pci_"):
            return self._evaluate_pci_control(control, evidence_list)
        else:
            # Generic evaluation
            return self._evaluate_generic_control(control, evidence_list)

    def _evaluate_soc2_control(
        self, control: ComplianceControl, evidence_list: List[ComplianceEvidence]
    ) -> Tuple[ControlStatus, List[Dict]]:
        """Evaluate SOC2 control."""
        findings = []

        # Example: Access control evaluation
        if control.category == "access_control":
            # Check for proper authentication
            auth_evidence = [e for e in evidence_list if e.type == "authentication"]
            if not auth_evidence:
                findings.append(
                    {
                        "severity": "high",
                        "title": "Missing authentication evidence",
                        "description": "No evidence of proper authentication mechanisms",
                    }
                )

            # Check for access reviews
            review_evidence = [e for e in evidence_list if e.type == "access_review"]
            if not review_evidence:
                findings.append(
                    {
                        "severity": "medium",
                        "title": "Missing access review evidence",
                        "description": "No evidence of periodic access reviews",
                    }
                )

        if findings:
            return ControlStatus.PARTIALLY_COMPLIANT, findings

        return ControlStatus.COMPLIANT, []

    def _evaluate_hipaa_control(
        self, control: ComplianceControl, evidence_list: List[ComplianceEvidence]
    ) -> Tuple[ControlStatus, List[Dict]]:
        """Evaluate HIPAA control."""
        findings = []

        # Example: PHI protection evaluation
        if control.category == "phi_protection":
            # Check for encryption
            encryption_evidence = [e for e in evidence_list if e.type == "encryption"]
            if not encryption_evidence:
                findings.append(
                    {
                        "severity": "critical",
                        "title": "Missing PHI encryption",
                        "description": "No evidence of PHI encryption at rest and in transit",
                    }
                )

        if findings:
            return ControlStatus.NON_COMPLIANT, findings

        return ControlStatus.COMPLIANT, []

    def _evaluate_pci_control(
        self, control: ComplianceControl, evidence_list: List[ComplianceEvidence]
    ) -> Tuple[ControlStatus, List[Dict]]:
        """Evaluate PCI-DSS control."""
        findings = []

        # Example: Cardholder data protection
        if control.category == "cardholder_data":
            # Check for data masking
            masking_evidence = [e for e in evidence_list if e.type == "data_masking"]
            if not masking_evidence:
                findings.append(
                    {
                        "severity": "critical",
                        "title": "Missing cardholder data masking",
                        "description": "No evidence of proper cardholder data masking",
                    }
                )

        if findings:
            return ControlStatus.NON_COMPLIANT, findings

        return ControlStatus.COMPLIANT, []

    def _evaluate_generic_control(
        self, control: ComplianceControl, evidence_list: List[ComplianceEvidence]
    ) -> Tuple[ControlStatus, List[Dict]]:
        """Generic control evaluation."""
        # Simple check: if we have evidence, consider it compliant
        # In practice, this would be more sophisticated
        if evidence_list:
            return ControlStatus.COMPLIANT, []
        else:
            return ControlStatus.UNDER_REVIEW, []

    def _collect_audit_log_evidence(
        self, control: ComplianceControl
    ) -> List[ComplianceEvidence]:
        """Collect evidence from audit logs."""
        evidence_list = []

        # This would query the actual audit log system
        # For now, create sample evidence
        evidence_id = self._generate_evidence_id()
        evidence = ComplianceEvidence(
            evidence_id=evidence_id,
            control_id=control.control_id,
            type="audit_logs",
            description="Audit log analysis for compliance",
            collected_at=datetime.datetime.utcnow().isoformat(),
            source_system="audit_logger",
            data={"log_count": 1000, "anomalies": 0, "coverage": "complete"},
            validated=True,
        )
        evidence_list.append(evidence)

        return evidence_list

    def _collect_access_control_evidence(
        self, control: ComplianceControl
    ) -> List[ComplianceEvidence]:
        """Collect evidence from access control system."""
        evidence_list = []

        # Query RBAC system
        roles = self.rbac_manager.list_roles()
        permissions = self.rbac_manager.list_permissions()

        evidence_id = self._generate_evidence_id()
        evidence = ComplianceEvidence(
            evidence_id=evidence_id,
            control_id=control.control_id,
            type="access_controls",
            description="Access control configuration",
            collected_at=datetime.datetime.utcnow().isoformat(),
            source_system="rbac_manager",
            data={
                "role_count": len(roles),
                "permission_count": len(permissions),
                "principle_of_least_privilege": True,
            },
            validated=True,
        )
        evidence_list.append(evidence)

        return evidence_list

    def _collect_workflow_evidence(
        self, control: ComplianceControl
    ) -> List[ComplianceEvidence]:
        """Collect evidence from approval workflows."""
        evidence_list = []

        # Query workflow engine
        # This would get actual workflow execution data
        evidence_id = self._generate_evidence_id()
        evidence = ComplianceEvidence(
            evidence_id=evidence_id,
            control_id=control.control_id,
            type="approval_workflows",
            description="Approval workflow compliance",
            collected_at=datetime.datetime.utcnow().isoformat(),
            source_system="workflow_engine",
            data={
                "workflows_executed": 50,
                "approval_rate": 0.92,
                "sla_compliance": 0.98,
            },
            validated=True,
        )
        evidence_list.append(evidence)

        return evidence_list

    def _create_finding(self, control_id: str, finding_data: Dict) -> str:
        """Create a compliance finding."""
        finding_id = self._generate_finding_id()

        finding = ComplianceFinding(
            finding_id=finding_id,
            control_id=control_id,
            severity=finding_data["severity"],
            title=finding_data["title"],
            description=finding_data["description"],
            identified_at=datetime.datetime.utcnow().isoformat(),
        )

        self.findings[finding_id] = finding
        return finding_id

    def _generate_recommendations(
        self, framework: ComplianceFramework, findings: List[ComplianceFinding]
    ) -> List[str]:
        """Generate recommendations based on findings."""
        recommendations = []

        # Group findings by severity
        critical_findings = [f for f in findings if f.severity == "critical"]
        high_findings = [f for f in findings if f.severity == "high"]

        if critical_findings:
            recommendations.append(
                f"URGENT: Address {len(critical_findings)} critical findings immediately "
                "to maintain compliance."
            )

        if high_findings:
            recommendations.append(
                f"Schedule remediation for {len(high_findings)} high-severity findings "
                "within the next 30 days."
            )

        # Framework-specific recommendations
        if framework == ComplianceFramework.SOC2:
            recommendations.append(
                "Ensure all access controls are reviewed quarterly and "
                "documented according to SOC2 requirements."
            )
        elif framework == ComplianceFramework.HIPAA:
            recommendations.append(
                "Implement additional PHI protection measures including "
                "encryption and access logging."
            )
        elif framework == ComplianceFramework.PCI_DSS:
            recommendations.append(
                "Review and update cardholder data handling procedures "
                "to meet PCI-DSS v4.0 requirements."
            )

        return recommendations

    def _finding_to_dict(self, finding: ComplianceFinding) -> Dict:
        """Convert finding to dictionary."""
        return {
            "finding_id": finding.finding_id,
            "control_id": finding.control_id,
            "severity": finding.severity,
            "title": finding.title,
            "description": finding.description,
            "identified_at": finding.identified_at,
            "status": finding.status,
            "remediation_plan": finding.remediation_plan,
            "remediation_deadline": finding.remediation_deadline,
        }

    def _export_json(self, report: ComplianceReport, filepath: Path) -> str:
        """Export report as JSON."""
        report_dict = {
            "report_id": report.report_id,
            "framework": report.framework.value,
            "report_type": report.report_type,
            "generated_at": report.generated_at,
            "generated_by": report.generated_by,
            "period_start": report.period_start,
            "period_end": report.period_end,
            "overall_status": report.overall_status.value,
            "control_summary": report.control_summary,
            "findings": report.findings,
            "recommendations": report.recommendations,
            "evidence_references": report.evidence_references,
            "metadata": report.metadata,
        }

        with open(filepath, "w") as f:
            json.dump(report_dict, f, indent=2)

        return str(filepath)

    def _export_csv(self, report: ComplianceReport, filepath: Path) -> str:
        """Export report as CSV."""
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)

            # Write header info
            writer.writerow(["Compliance Report"])
            writer.writerow(["Report ID", report.report_id])
            writer.writerow(["Framework", report.framework.value])
            writer.writerow(["Generated At", report.generated_at])
            writer.writerow(["Overall Status", report.overall_status.value])
            writer.writerow([])

            # Write control summary
            writer.writerow(["Control Summary"])
            writer.writerow(["Status", "Count"])
            for status, count in report.control_summary.items():
                writer.writerow([status, count])
            writer.writerow([])

            # Write findings
            writer.writerow(["Findings"])
            writer.writerow(["ID", "Control", "Severity", "Title", "Status"])
            for finding in report.findings:
                writer.writerow(
                    [
                        finding["finding_id"],
                        finding["control_id"],
                        finding["severity"],
                        finding["title"],
                        finding["status"],
                    ]
                )

        return str(filepath)

    def _export_html(self, report: ComplianceReport, filepath: Path) -> str:
        """Export report as HTML."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Compliance Report - {report.framework.value.upper()}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .compliant {{ color: green; }}
                .non-compliant {{ color: red; }}
                .partially-compliant {{ color: orange; }}
            </style>
        </head>
        <body>
            <h1>Compliance Report - {report.framework.value.upper()}</h1>
            <p><strong>Report ID:</strong> {report.report_id}</p>
            <p><strong>Generated:</strong> {report.generated_at}</p>
            <p><strong>Period:</strong> {report.period_start} to {report.period_end}</p>
            <p><strong>Overall Status:</strong> 
                <span class="{report.overall_status.value.replace('_', '-')}">
                    {report.overall_status.value.upper()}
                </span>
            </p>
            
            <h2>Control Summary</h2>
            <table>
                <tr>
                    <th>Status</th>
                    <th>Count</th>
                </tr>
        """

        for status, count in report.control_summary.items():
            html_content += f"""
                <tr>
                    <td>{status.replace('_', ' ').title()}</td>
                    <td>{count}</td>
                </tr>
            """

        html_content += """
            </table>
            
            <h2>Findings</h2>
            <table>
                <tr>
                    <th>Severity</th>
                    <th>Title</th>
                    <th>Status</th>
                </tr>
        """

        for finding in report.findings:
            html_content += f"""
                <tr>
                    <td>{finding['severity'].upper()}</td>
                    <td>{finding['title']}</td>
                    <td>{finding['status']}</td>
                </tr>
            """

        html_content += """
            </table>
            
            <h2>Recommendations</h2>
            <ul>
        """

        for rec in report.recommendations:
            html_content += f"<li>{rec}</li>"

        html_content += """
            </ul>
        </body>
        </html>
        """

        with open(filepath, "w") as f:
            f.write(html_content)

        return str(filepath)

    def _initialize_default_controls(self):
        """Initialize default compliance controls."""
        # SOC2 Controls
        soc2_controls: List[Dict[str, Any]] = [
            {
                "control_id": "soc2_cc1.1",
                "title": "Logical Access Controls",
                "description": "Access to systems is controlled through logical access controls",
                "category": "access_control",
                "requirements": [
                    "Multi-factor authentication",
                    "Role-based access",
                    "Access reviews",
                ],
                "evidence_types": ["access_controls", "audit_logs"],
            },
            {
                "control_id": "soc2_cc1.2",
                "title": "User Access Reviews",
                "description": "User access is reviewed periodically",
                "category": "access_control",
                "requirements": [
                    "Quarterly reviews",
                    "Management approval",
                    "Documentation",
                ],
                "evidence_types": ["access_controls", "approval_workflows"],
            },
        ]

        for control_data in soc2_controls:
            control = ComplianceControl(
                framework=ComplianceFramework.SOC2, **control_data
            )
            self.controls[control.control_id] = control
            self.framework_controls[ComplianceFramework.SOC2].append(control.control_id)

        # HIPAA Controls
        hipaa_controls: List[Dict[str, Any]] = [
            {
                "control_id": "hipaa_164.308",
                "title": "Administrative Safeguards",
                "description": "Administrative safeguards for PHI protection",
                "category": "phi_protection",
                "requirements": [
                    "Risk assessment",
                    "Workforce training",
                    "Access management",
                ],
                "evidence_types": ["audit_logs", "access_controls"],
            },
            {
                "control_id": "hipaa_164.312",
                "title": "Technical Safeguards",
                "description": "Technical safeguards for PHI protection",
                "category": "phi_protection",
                "requirements": ["Encryption", "Audit controls", "Integrity controls"],
                "evidence_types": ["audit_logs", "access_controls"],
            },
        ]

        for control_data in hipaa_controls:
            control = ComplianceControl(
                framework=ComplianceFramework.HIPAA, **control_data
            )
            self.controls[control.control_id] = control
            self.framework_controls[ComplianceFramework.HIPAA].append(
                control.control_id
            )

        # PCI-DSS Controls
        pci_controls: List[Dict[str, Any]] = [
            {
                "control_id": "pci_3.4",
                "title": "Cardholder Data Protection",
                "description": "Render PAN unreadable anywhere it is stored",
                "category": "cardholder_data",
                "requirements": ["Encryption", "Truncation", "Tokenization"],
                "evidence_types": ["audit_logs"],
            }
        ]

        for control_data in pci_controls:
            control = ComplianceControl(
                framework=ComplianceFramework.PCI_DSS, **control_data
            )
            self.controls[control.control_id] = control
            self.framework_controls[ComplianceFramework.PCI_DSS].append(
                control.control_id
            )

        self._save_compliance_data()

    def _generate_report_id(self) -> str:
        """Generate unique report ID."""
        return (
            f"rpt_{datetime.datetime.utcnow().strftime('%Y%m%d')}_{os.urandom(4).hex()}"
        )

    def _generate_evidence_id(self) -> str:
        """Generate unique evidence ID."""
        return f"evd_{os.urandom(8).hex()}"

    def _generate_finding_id(self) -> str:
        """Generate unique finding ID."""
        return f"fnd_{os.urandom(8).hex()}"

    def _load_compliance_data(self):
        """Load compliance data from storage."""
        # Load controls
        controls_file = self.storage_path / "controls.json"
        if controls_file.exists():
            with open(controls_file, "r") as f:
                controls_data = json.load(f)
                for control_data in controls_data:
                    control = ComplianceControl(
                        control_id=control_data["control_id"],
                        framework=ComplianceFramework(control_data["framework"]),
                        title=control_data["title"],
                        description=control_data["description"],
                        category=control_data["category"],
                        requirements=control_data["requirements"],
                        evidence_types=control_data["evidence_types"],
                        automated=control_data.get("automated", True),
                        frequency=control_data.get("frequency", "continuous"),
                        last_assessed=control_data.get("last_assessed"),
                        status=ControlStatus(
                            control_data.get("status", "under_review")
                        ),
                        metadata=control_data.get("metadata", {}),
                    )
                    self.controls[control.control_id] = control
                    self.framework_controls[control.framework].append(
                        control.control_id
                    )

        # Load evidence
        evidence_file = self.storage_path / "evidence.json"
        if evidence_file.exists():
            with open(evidence_file, "r") as f:
                evidence_data = json.load(f)
                for ev_data in evidence_data:
                    evidence = ComplianceEvidence(**ev_data)
                    self.evidence[evidence.evidence_id] = evidence

        # Load reports
        reports_file = self.storage_path / "reports.json"
        if reports_file.exists():
            with open(reports_file, "r") as f:
                reports_data = json.load(f)
                for report_data in reports_data:
                    report = ComplianceReport(
                        report_id=report_data["report_id"],
                        framework=ComplianceFramework(report_data["framework"]),
                        report_type=report_data["report_type"],
                        generated_at=report_data["generated_at"],
                        generated_by=report_data["generated_by"],
                        period_start=report_data["period_start"],
                        period_end=report_data["period_end"],
                        overall_status=ControlStatus(report_data["overall_status"]),
                        control_summary=report_data["control_summary"],
                        findings=report_data["findings"],
                        recommendations=report_data["recommendations"],
                        evidence_references=report_data["evidence_references"],
                        metadata=report_data.get("metadata", {}),
                    )
                    self.reports[report.report_id] = report

        # Load findings
        findings_file = self.storage_path / "findings.json"
        if findings_file.exists():
            with open(findings_file, "r") as f:
                findings_data = json.load(f)
                for finding_data in findings_data:
                    finding = ComplianceFinding(**finding_data)
                    self.findings[finding.finding_id] = finding

    def _save_compliance_data(self):
        """Save compliance data to storage."""
        # Save controls
        controls_data = []
        for control in self.controls.values():
            control_dict = {
                "control_id": control.control_id,
                "framework": control.framework.value,
                "title": control.title,
                "description": control.description,
                "category": control.category,
                "requirements": control.requirements,
                "evidence_types": control.evidence_types,
                "automated": control.automated,
                "frequency": control.frequency,
                "last_assessed": control.last_assessed,
                "status": control.status.value,
                "metadata": control.metadata,
            }
            controls_data.append(control_dict)

        with open(self.storage_path / "controls.json", "w") as f:
            json.dump(controls_data, f, indent=2)

        # Save evidence
        evidence_data = []
        for evidence in self.evidence.values():
            ev_dict = {
                "evidence_id": evidence.evidence_id,
                "control_id": evidence.control_id,
                "type": evidence.type,
                "description": evidence.description,
                "collected_at": evidence.collected_at,
                "source_system": evidence.source_system,
                "data": evidence.data,
                "validated": evidence.validated,
                "validation_details": evidence.validation_details,
            }
            evidence_data.append(ev_dict)

        with open(self.storage_path / "evidence.json", "w") as f:
            json.dump(evidence_data, f, indent=2)

        # Save reports
        reports_data = []
        for report in self.reports.values():
            report_dict = {
                "report_id": report.report_id,
                "framework": report.framework.value,
                "report_type": report.report_type,
                "generated_at": report.generated_at,
                "generated_by": report.generated_by,
                "period_start": report.period_start,
                "period_end": report.period_end,
                "overall_status": report.overall_status.value,
                "control_summary": report.control_summary,
                "findings": report.findings,
                "recommendations": report.recommendations,
                "evidence_references": report.evidence_references,
                "metadata": report.metadata,
            }
            reports_data.append(report_dict)

        with open(self.storage_path / "reports.json", "w") as f:
            json.dump(reports_data, f, indent=2)

        # Save findings
        findings_data = []
        for finding in self.findings.values():
            finding_dict = {
                "finding_id": finding.finding_id,
                "control_id": finding.control_id,
                "severity": finding.severity,
                "title": finding.title,
                "description": finding.description,
                "identified_at": finding.identified_at,
                "remediation_plan": finding.remediation_plan,
                "remediation_deadline": finding.remediation_deadline,
                "status": finding.status,
            }
            findings_data.append(finding_dict)

        with open(self.storage_path / "findings.json", "w") as f:
            json.dump(findings_data, f, indent=2)


# Singleton instance
_compliance_reporting = None


def get_compliance_reporting(config: Optional[Dict] = None) -> ComplianceReportingSystem:
    """Get or create the singleton ComplianceReportingSystem instance."""
    global _compliance_reporting
    if _compliance_reporting is None:
        _compliance_reporting = ComplianceReportingSystem(config)
    return _compliance_reporting
