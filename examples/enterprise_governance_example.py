"""
Enterprise Governance Framework Example with Regulated Industries.

This example demonstrates the complete enterprise governance setup
including regulated industries support for healthcare and financial services.
"""

import json
from datetime import datetime

from modules.security.governance_framework import (
    ComplianceFramework,
    EnterpriseGovernanceFramework,
    GovernanceCapability,
    GovernanceConfig,
    HealingActionRequest,
)
from modules.security.regulated_industries import (
    ComplianceRequirement,
    RegulatedIndustry,
    ValidationLevel,
)


def setup_enterprise_governance():
    """Set up complete enterprise governance with regulated industries."""
    print("=== Setting up Enterprise Governance Framework ===\n")

    # Configure with all capabilities including regulated industries
    config = GovernanceConfig(
        enabled_capabilities=[
            GovernanceCapability.RBAC,
            GovernanceCapability.USER_MANAGEMENT,
            GovernanceCapability.APPROVAL_WORKFLOWS,
            GovernanceCapability.COMPLIANCE_REPORTING,
            GovernanceCapability.POLICY_ENFORCEMENT,
            GovernanceCapability.IDENTITY_FEDERATION,
            GovernanceCapability.AUDIT_LOGGING,
            GovernanceCapability.REGULATED_INDUSTRIES,  # New capability
        ],
        compliance_frameworks=[
            ComplianceFramework.HIPAA,
            ComplianceFramework.SOC2,
            ComplianceFramework.PCI_DSS,
        ],
        require_approval_for_production=True,
        enforce_policies=True,
        enable_sso=True,
        audit_retention_days=2555,  # 7 years for compliance
        session_timeout_minutes=30,
        mfa_required_roles=["admin", "operator", "compliance_officer"],
    )

    governance = EnterpriseGovernanceFramework(config)

    print("Governance framework initialized with capabilities:")
    for cap in config.enabled_capabilities:
        print(f"  - {cap.value}")

    return governance


def configure_healthcare_environment(governance):
    """Configure healthcare-specific compliance."""
    print("\n=== Configuring Healthcare Environment ===\n")

    # Configure HIPAA compliance for healthcare
    healthcare_config = {
        "validation_level": "enhanced",
        "critical_systems": [
            "patient_records_db",
            "ehr_application",
            "lab_results_system",
            "appointment_scheduler",
            "medical_imaging",
        ],
        "data_classifications": {
            "phi": "highly_sensitive",
            "medical_records": "protected",
            "lab_results": "protected",
            "imaging_data": "protected",
            "appointment_data": "sensitive",
        },
        "approval_thresholds": {
            "standard": 1,
            "enhanced": 2,
            "critical": 3,
            "timeout": 120,
        },
        "retention_policies": {
            "audit_logs": 2555,  # 7 years
            "phi_access_logs": 2190,  # 6 years
            "security_events": 365,
        },
        "encryption_requirements": {
            "algorithm": "AES-256",
            "key_rotation": "quarterly",
            "at_rest": "required",
            "in_transit": "required",
            "key_management": "hsm_required",
        },
        "audit_frequency": "continuous",
        "metadata": {
            "covered_entity": "Regional Medical Center",
            "compliance_officer": "hipaa.officer@hospital.org",
            "business_associates": ["lab_partner", "imaging_center"],
        },
    }

    success = governance.configure_regulated_industry(
        RegulatedIndustry.HEALTHCARE,
        [
            ComplianceRequirement.HIPAA_PRIVACY,
            ComplianceRequirement.HIPAA_SECURITY,
            ComplianceRequirement.HIPAA_BREACH,
        ],
        healthcare_config,
    )

    print(f"Healthcare configuration: {'Success' if success else 'Failed'}")

    # Create healthcare-specific policies
    governance.configure_policy(
        name="PHI Protection Policy",
        rules=[
            {
                "name": "no_phi_in_logs",
                "condition": {
                    "type": "regex",
                    "pattern": r"\b\d{3}-\d{2}-\d{4}\b|patient|diagnosis|medical",
                },
                "action": "block",
                "message": "PHI detected in patch - must be removed",
            },
            {
                "name": "phi_encryption_required",
                "condition": {
                    "type": "service_check",
                    "services": healthcare_config["critical_systems"],
                },
                "action": "require_approval",
                "message": "Changes to PHI systems require security review",
            },
        ],
        policy_type="compliance",
        priority=100,
    )

    print("Created PHI Protection Policy")


def configure_financial_environment(governance):
    """Configure financial services compliance."""
    print("\n=== Configuring Financial Services Environment ===\n")

    # Configure SOX and PCI-DSS compliance
    financial_config = {
        "validation_level": "critical",
        "critical_systems": [
            "general_ledger",
            "payment_gateway",
            "transaction_processor",
            "financial_reporting",
            "audit_system",
        ],
        "data_classifications": {
            "cardholder_data": "highly_sensitive",
            "financial_records": "protected",
            "transaction_logs": "sensitive",
            "audit_trails": "protected",
        },
        "approval_thresholds": {
            "standard": 2,
            "enhanced": 3,
            "critical": 5,
            "timeout": 60,
        },
        "retention_policies": {
            "financial_records": 2555,  # 7 years for SOX
            "transaction_logs": 1095,  # 3 years
            "audit_trails": 2190,  # 6 years
        },
        "encryption_requirements": {
            "algorithm": "AES-256",
            "tokenization": "required_for_pan",
            "key_management": "hsm_required",
            "tls_version": "min_1.2",
        },
        "metadata": {
            "organization": "Global Finance Corp",
            "sox_officer": "sox.compliance@finance.com",
            "pci_level": "1",
        },
    }

    success = governance.configure_regulated_industry(
        RegulatedIndustry.FINANCIAL_SERVICES,
        [
            ComplianceRequirement.SOX_CONTROLS,
            ComplianceRequirement.PCI_DSS_LEVEL1,
            ComplianceRequirement.FINRA_COMPLIANCE,
        ],
        financial_config,
    )

    print(f"Financial services configuration: {'Success' if success else 'Failed'}")

    # Create financial-specific policies
    governance.configure_policy(
        name="Financial Data Protection",
        rules=[
            {
                "name": "mask_card_numbers",
                "condition": {
                    "type": "regex",
                    "pattern": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
                },
                "action": "block",
                "message": "Card numbers must be tokenized",
            },
            {
                "name": "sox_change_control",
                "condition": {
                    "type": "service_check",
                    "services": ["general_ledger", "financial_reporting"],
                },
                "action": "require_approval",
                "message": "SOX systems require change advisory board approval",
            },
        ],
        policy_type="compliance",
        priority=100,
    )

    print("Created Financial Data Protection Policy")


def demonstrate_healthcare_healing():
    """Demonstrate healing action in healthcare environment."""
    print("\n=== Healthcare Healing Action Demo ===\n")

    governance = setup_enterprise_governance()
    configure_healthcare_environment(governance)

    # Create a user (developer)
    developer_id = governance.manage_user(
        action="create",
        username="john.developer",
        email="john@hospital.org",
        password="SecurePass123!",
        full_name="John Developer",
        roles=["developer"],
        groups=["ehr_team"],
    )["user_id"]

    print(f"Created developer user: {developer_id}")

    # Simulate a healing action request for patient data issue
    healing_request = HealingActionRequest(
        request_id="heal_med_001",
        action_type="code_fix",
        error_context={
            "error_type": "NullPointerException",
            "error_message": "Cannot read property of null",
            "stack_trace": "at PatientService.getPatientInfo()",
            "service": "ehr_application",
            "file": "services/patient_service.py",
            "line": 145,
        },
        patch_content='''
def get_patient_info(patient_id):
    """Get patient information with proper null checking."""
    try:
        # Add null check for patient record
        patient = db.patients.find_one({'id': patient_id})
        if not patient:
            logger.warning(f"Patient not found: {patient_id}")
            return None
        
        # Safely access nested properties
        return {
            'id': patient.get('id'),
            'name': patient.get('name', 'Unknown'),
            'dob': patient.get('dob'),
            'last_visit': patient.get('visits', {}).get('last')
        }
    except Exception as e:
        logger.error(f"Error retrieving patient info: {str(e)}")
        return None
        ''',
        environment="production",
        service_name="ehr_application",
        language="python",
        requested_by=developer_id,
    )

    print("\nEvaluating healing action...")
    decision = governance.evaluate_healing_action(healing_request)

    print(f"\nGovernance Decision:")
    print(f"  Allowed: {decision.allowed}")
    print(f"  Reason: {decision.reason}")
    print(f"  Approval Required: {decision.approval_required}")

    if decision.policy_violations:
        print(f"  Policy Violations: {decision.policy_violations}")

    if decision.compliance_issues:
        print(f"  Compliance Issues:")
        for issue in decision.compliance_issues:
            print(f"    - {issue}")

    if decision.metadata.get("industry_validation"):
        industry_val = decision.metadata["industry_validation"]
        print(f"\n  Industry Validation:")
        print(f"    Passed: {industry_val['passed']}")
        if industry_val.get("issues"):
            print(f"    Issues:")
            for issue in industry_val["issues"]:
                print(f"      - {issue}")

    # Show industry dashboard
    print("\n=== Healthcare Compliance Dashboard ===")
    dashboard = governance.get_industry_dashboard(RegulatedIndustry.HEALTHCARE)
    print(f"Validation Level: {dashboard['configuration']['validation_level']}")
    print(f"Critical Systems: {dashboard['configuration']['critical_systems_count']}")
    print(f"Audit Frequency: {dashboard['configuration']['audit_frequency']}")


def demonstrate_financial_healing():
    """Demonstrate healing action in financial environment."""
    print("\n=== Financial Services Healing Action Demo ===\n")

    governance = setup_enterprise_governance()
    configure_financial_environment(governance)

    # Create a DBA user
    dba_id = governance.manage_user(
        action="create",
        username="sarah.dba",
        email="sarah@finance.com",
        password="SecurePass123!",
        full_name="Sarah DBA",
        roles=["database_admin"],
        groups=["payment_team"],
    )["user_id"]

    print(f"Created DBA user: {dba_id}")

    # Simulate a healing action for payment processing
    healing_request = HealingActionRequest(
        request_id="heal_fin_001",
        action_type="database_change",
        error_context={
            "error_type": "DeadlockException",
            "error_message": "Transaction deadlock detected",
            "stack_trace": "at PaymentProcessor.processTransaction()",
            "service": "payment_gateway",
            "database": "payments_db",
            "table": "transactions",
        },
        patch_content="""
-- Add index to prevent deadlocks in payment processing
CREATE INDEX CONCURRENTLY idx_transactions_merchant_timestamp 
ON transactions(merchant_id, created_at DESC) 
WHERE status IN ('pending', 'processing');

-- Update isolation level for payment procedures
ALTER PROCEDURE process_payment SET TRANSACTION ISOLATION LEVEL READ COMMITTED;
        """,
        environment="production",
        service_name="payment_gateway",
        language="sql",
        requested_by=dba_id,
        metadata={
            "change_type": "performance_optimization",
            "tested_in": "staging",
            "rollback_script": "DROP INDEX idx_transactions_merchant_timestamp;",
        },
    )

    print("\nEvaluating healing action...")
    decision = governance.evaluate_healing_action(healing_request)

    print(f"\nGovernance Decision:")
    print(f"  Allowed: {decision.allowed}")
    print(f"  Reason: {decision.reason}")
    print(f"  Approval Required: {decision.approval_required}")

    if decision.approval_request_id:
        print(f"  Approval Request ID: {decision.approval_request_id}")
        print(f"  Workflow Instance ID: {decision.workflow_instance_id}")

    # Show compliance report
    print("\n=== Generating Compliance Report ===")
    report_id = governance.generate_compliance_report(
        ComplianceFramework.PCI_DSS, report_type="assessment"
    )
    print(f"Generated PCI-DSS compliance report: {report_id}")


def demonstrate_multi_industry_scenario():
    """Demonstrate a scenario affecting both healthcare and financial systems."""
    print("\n=== Multi-Industry Integration Scenario ===\n")

    governance = setup_enterprise_governance()
    configure_healthcare_environment(governance)
    configure_financial_environment(governance)

    # Create an integration developer
    dev_id = governance.manage_user(
        action="create",
        username="alex.integration",
        email="alex@company.com",
        password="SecurePass123!",
        full_name="Alex Integration Developer",
        roles=["developer", "integration_engineer"],
        groups=["integration_team"],
    )["user_id"]

    # Healing action for healthcare billing integration
    healing_request = HealingActionRequest(
        request_id="heal_int_001",
        action_type="code_fix",
        error_context={
            "error_type": "IntegrationError",
            "error_message": "Failed to process medical billing",
            "service": "billing_integration",
            "affected_systems": ["ehr_application", "payment_gateway"],
        },
        patch_content='''
def process_medical_billing(patient_id, procedure_codes, insurance_info):
    """Process medical billing with HIPAA and PCI compliance."""
    # Validate patient access (HIPAA)
    if not validate_phi_access(current_user, patient_id):
        raise AuthorizationError("Unauthorized PHI access")
    
    # Get procedure costs
    total_amount = calculate_procedure_costs(procedure_codes)
    
    # Tokenize payment information (PCI-DSS)
    payment_token = tokenize_payment_method(
        insurance_info.get('payment_method'),
        encryption='AES-256'
    )
    
    # Create compliant transaction
    transaction = {
        'patient_id': hash_patient_id(patient_id),  # De-identify
        'amount': total_amount,
        'payment_token': payment_token,
        'procedure_codes': procedure_codes,
        'timestamp': datetime.utcnow().isoformat(),
        'compliance': {
            'hipaa_compliant': True,
            'pci_compliant': True,
            'audit_trail': generate_audit_id()
        }
    }
    
    # Process with both systems
    result = process_compliant_transaction(transaction)
    
    # Log for compliance
    audit_medical_billing(patient_id, transaction['audit_trail'])
    
    return result
        ''',
        environment="production",
        service_name="billing_integration",
        language="python",
        requested_by=dev_id,
        metadata={"affects_phi": True, "affects_payment": True, "cross_system": True},
    )

    print("Evaluating multi-industry healing action...")
    decision = governance.evaluate_healing_action(healing_request)

    print(f"\nGovernance Decision:")
    print(f"  Allowed: {decision.allowed}")
    print(f"  Reason: {decision.reason}")
    print(f"  Approval Required: {decision.approval_required}")

    if decision.compliance_issues:
        print(f"\n  Compliance Issues ({len(decision.compliance_issues)} total):")
        # Group by industry
        hipaa_issues = [i for i in decision.compliance_issues if "HIPAA" in i]
        pci_issues = [i for i in decision.compliance_issues if "PCI" in i]

        if hipaa_issues:
            print("    Healthcare (HIPAA):")
            for issue in hipaa_issues[:2]:  # Show first 2
                print(f"      - {issue}")

        if pci_issues:
            print("    Financial (PCI-DSS):")
            for issue in pci_issues[:2]:  # Show first 2
                print(f"      - {issue}")

    # Show both industry dashboards
    print("\n=== Industry Compliance Status ===")

    healthcare_dash = governance.get_industry_dashboard(RegulatedIndustry.HEALTHCARE)
    print(f"\nHealthcare:")
    print(
        f"  Validation Pass Rate: {healthcare_dash['validation_metrics']['pass_rate']:.1f}%"
    )
    print(
        f"  Total Validations: {healthcare_dash['validation_metrics']['total_validations']}"
    )

    financial_dash = governance.get_industry_dashboard(
        RegulatedIndustry.FINANCIAL_SERVICES
    )
    print(f"\nFinancial Services:")
    print(
        f"  Validation Pass Rate: {financial_dash['validation_metrics']['pass_rate']:.1f}%"
    )
    print(
        f"  Total Validations: {financial_dash['validation_metrics']['total_validations']}"
    )


def main():
    """Main demonstration function."""
    print("=" * 70)
    print("Homeostasis Enterprise Governance with Regulated Industries")
    print("=" * 70)

    # Basic setup
    governance = setup_enterprise_governance()

    # Healthcare scenario
    demonstrate_healthcare_healing()

    print("\n" + "=" * 70 + "\n")

    # Financial services scenario
    demonstrate_financial_healing()

    print("\n" + "=" * 70 + "\n")

    # Multi-industry scenario
    demonstrate_multi_industry_scenario()

    print("\n" + "=" * 70)
    print("Demo Complete")
    print("=" * 70)
    print("\nThe Enterprise Governance Framework with Regulated Industries Support")
    print("ensures compliance across healthcare, financial services, government,")
    print("pharmaceutical, and telecommunications sectors while enabling safe,")
    print("automated healing of production systems.")


if __name__ == "__main__":
    main()
