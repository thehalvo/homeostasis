"""
Example usage of Regulated Industries Support.

This example demonstrates how to configure and use industry-specific
compliance controls for healthcare, financial services, government,
pharmaceutical, and telecommunications sectors.
"""

import json
from datetime import datetime
from modules.security.regulated_industries import (
    get_regulated_industries,
    RegulatedIndustry,
    ComplianceRequirement,
    ValidationLevel
)


def setup_healthcare_compliance():
    """Configure healthcare industry compliance."""
    print("\n=== Setting up Healthcare Compliance ===")
    
    regulated = get_regulated_industries()
    
    # Configure HIPAA compliance
    healthcare_config = {
        'validation_level': 'enhanced',
        'critical_systems': [
            'patient_records_db',
            'ehr_application',
            'lab_results_system',
            'billing_system',
            'appointment_scheduler'
        ],
        'data_classifications': {
            'phi': 'highly_sensitive',
            'medical_records': 'protected',
            'billing_info': 'sensitive',
            'appointment_data': 'internal'
        },
        'approval_thresholds': {
            'standard': 1,
            'enhanced': 2,
            'critical': 3,
            'timeout': 120  # minutes
        },
        'retention_policies': {
            'audit_logs': 2555,  # 7 years
            'phi_access_logs': 2190,  # 6 years
            'security_events': 365,  # 1 year
            'system_logs': 90  # 90 days
        },
        'encryption_requirements': {
            'algorithm': 'AES-256',
            'key_rotation': 'quarterly',
            'at_rest': 'required',
            'in_transit': 'required',
            'key_management': 'hsm_required'
        },
        'audit_frequency': 'continuous',
        'metadata': {
            'covered_entity': 'Regional Medical Center',
            'compliance_officer': 'jane.doe@hospital.org'
        }
    }
    
    success = regulated.configure_industry(
        RegulatedIndustry.HEALTHCARE,
        [
            ComplianceRequirement.HIPAA_PRIVACY,
            ComplianceRequirement.HIPAA_SECURITY,
            ComplianceRequirement.HIPAA_BREACH
        ],
        healthcare_config
    )
    
    print(f"Healthcare configuration: {'Success' if success else 'Failed'}")
    
    # Create healthcare-specific validation workflow
    workflow_id = regulated.create_validation_workflow(
        RegulatedIndustry.HEALTHCARE,
        'phi_data_modification',
        [
            {'step': 1, 'action': 'privacy_impact_assessment', 'required': True},
            {'step': 2, 'action': 'security_review', 'required': True},
            {'step': 3, 'action': 'compliance_approval', 'required': True},
            {'step': 4, 'action': 'deployment_verification', 'required': False}
        ]
    )
    
    print(f"Created workflow: {workflow_id}")


def setup_financial_compliance():
    """Configure financial services compliance."""
    print("\n=== Setting up Financial Services Compliance ===")
    
    regulated = get_regulated_industries()
    
    # Configure SOX and PCI-DSS compliance
    financial_config = {
        'validation_level': 'critical',
        'critical_systems': [
            'general_ledger',
            'payment_gateway',
            'trading_platform',
            'risk_management_system',
            'reporting_engine'
        ],
        'data_classifications': {
            'cardholder_data': 'highly_sensitive',
            'financial_records': 'protected',
            'transaction_logs': 'sensitive',
            'market_data': 'confidential'
        },
        'approval_thresholds': {
            'standard': 2,
            'enhanced': 3,
            'critical': 5,
            'timeout': 60
        },
        'retention_policies': {
            'financial_records': 2555,  # 7 years for SOX
            'transaction_logs': 1095,  # 3 years
            'audit_trails': 2190,  # 6 years
            'trading_records': 2190  # 6 years for FINRA
        },
        'encryption_requirements': {
            'algorithm': 'AES-256',
            'tokenization': 'required_for_pan',
            'key_management': 'hsm_required',
            'tls_version': 'min_1.2'
        },
        'audit_frequency': 'continuous',
        'metadata': {
            'organization': 'Global Finance Corp',
            'sox_officer': 'john.smith@finance.com',
            'pci_level': '1'
        }
    }
    
    success = regulated.configure_industry(
        RegulatedIndustry.FINANCIAL_SERVICES,
        [
            ComplianceRequirement.SOX_CONTROLS,
            ComplianceRequirement.PCI_DSS_LEVEL1,
            ComplianceRequirement.FINRA_COMPLIANCE,
            ComplianceRequirement.BASEL_III
        ],
        financial_config
    )
    
    print(f"Financial services configuration: {'Success' if success else 'Failed'}")


def setup_government_compliance():
    """Configure government and defense compliance."""
    print("\n=== Setting up Government/Defense Compliance ===")
    
    regulated = get_regulated_industries()
    
    # Configure FedRAMP and CMMC compliance
    government_config = {
        'validation_level': 'critical',
        'critical_systems': [
            'classified_data_store',
            'secure_communications',
            'identity_management',
            'audit_aggregation'
        ],
        'data_classifications': {
            'cui': 'controlled',
            'classified': 'secret',
            'public': 'unclassified',
            'fouo': 'restricted'
        },
        'approval_thresholds': {
            'standard': 2,
            'enhanced': 4,
            'critical': 6,
            'timeout': 30
        },
        'retention_policies': {
            'audit_logs': 3650,  # 10 years
            'security_events': 2555,  # 7 years
            'access_logs': 1825,  # 5 years
            'incident_reports': 3650  # 10 years
        },
        'encryption_requirements': {
            'algorithm': 'AES-256',
            'key_length': 'min_256_bit',
            'fips_140_2': 'level_3',
            'quantum_resistant': 'planning_required'
        },
        'audit_frequency': 'continuous',
        'metadata': {
            'agency': 'Department of Example',
            'ato_date': '2024-01-15',
            'authorization_official': 'ao@agency.gov'
        }
    }
    
    success = regulated.configure_industry(
        RegulatedIndustry.GOVERNMENT_DEFENSE,
        [
            ComplianceRequirement.FEDRAMP_MODERATE,
            ComplianceRequirement.FISMA_COMPLIANCE,
            ComplianceRequirement.CMMC_LEVEL3
        ],
        government_config
    )
    
    print(f"Government configuration: {'Success' if success else 'Failed'}")


def setup_pharmaceutical_compliance():
    """Configure pharmaceutical compliance."""
    print("\n=== Setting up Pharmaceutical Compliance ===")
    
    regulated = get_regulated_industries()
    
    # Configure FDA 21 CFR Part 11 and GxP compliance
    pharma_config = {
        'validation_level': 'critical',
        'critical_systems': [
            'clinical_trial_management',
            'batch_processing',
            'quality_management',
            'laboratory_information',
            'electronic_batch_records'
        ],
        'data_classifications': {
            'clinical_data': 'gcp_protected',
            'manufacturing_data': 'gmp_protected',
            'quality_data': 'proprietary',
            'regulatory_submissions': 'confidential'
        },
        'approval_thresholds': {
            'standard': 2,
            'enhanced': 3,
            'critical': 4,
            'timeout': 90
        },
        'retention_policies': {
            'batch_records': 3650,  # 10 years minimum
            'clinical_trial_data': 5475,  # 15 years
            'quality_records': 3650,  # 10 years
            'audit_trails': 3650  # 10 years
        },
        'encryption_requirements': {
            'algorithm': 'AES-256',
            'digital_signatures': 'required',
            'time_stamping': 'trusted_authority',
            'data_integrity': 'cryptographic_hash'
        },
        'audit_frequency': 'continuous',
        'metadata': {
            'company': 'BioPharma Inc',
            'quality_head': 'quality@biopharma.com',
            'sites': ['US-Manufacturing', 'EU-Research']
        }
    }
    
    success = regulated.configure_industry(
        RegulatedIndustry.PHARMACEUTICAL,
        [
            ComplianceRequirement.FDA_21_CFR_11,
            ComplianceRequirement.GMP_COMPLIANCE,
            ComplianceRequirement.GLP_COMPLIANCE,
            ComplianceRequirement.GCP_COMPLIANCE
        ],
        pharma_config
    )
    
    print(f"Pharmaceutical configuration: {'Success' if success else 'Failed'}")


def setup_telecom_compliance():
    """Configure telecommunications compliance."""
    print("\n=== Setting up Telecom/Utilities Compliance ===")
    
    regulated = get_regulated_industries()
    
    # Configure NERC CIP and telecom compliance
    telecom_config = {
        'validation_level': 'critical',
        'critical_systems': [
            'network_operations_center',
            'emergency_call_routing',
            'billing_system',
            'network_monitoring',
            'customer_data_warehouse'
        ],
        'data_classifications': {
            'cpni': 'protected',
            'network_diagrams': 'confidential',
            'customer_data': 'sensitive',
            'operational_data': 'internal'
        },
        'approval_thresholds': {
            'standard': 1,
            'enhanced': 2,
            'critical': 3,
            'timeout': 45
        },
        'retention_policies': {
            'call_records': 1095,  # 3 years
            'network_logs': 365,  # 1 year
            'security_events': 730,  # 2 years
            'customer_data': 2555  # 7 years
        },
        'encryption_requirements': {
            'algorithm': 'AES-256',
            'voice_encryption': 'optional',
            'data_encryption': 'required',
            'key_exchange': 'ecdhe'
        },
        'audit_frequency': 'continuous',
        'metadata': {
            'carrier': 'National Telecom',
            'nerc_registered': True,
            'e911_provider': True
        }
    }
    
    success = regulated.configure_industry(
        RegulatedIndustry.TELECOM_UTILITIES,
        [
            ComplianceRequirement.NERC_CIP,
            ComplianceRequirement.SOC_TELECOM,
            ComplianceRequirement.E911_COMPLIANCE
        ],
        telecom_config
    )
    
    print(f"Telecom configuration: {'Success' if success else 'Failed'}")


def validate_healthcare_action():
    """Example of validating a healthcare-related healing action."""
    print("\n=== Healthcare Action Validation Example ===")
    
    regulated = get_regulated_industries()
    
    # Simulate a healing action that might affect PHI
    action = {
        'id': 'heal_001',
        'type': 'patch_application',
        'description': 'Fix patient record display issue',
        'patch': '''
def get_patient_info(patient_id):
    # Fixed query to prevent SQL injection
    query = "SELECT name, dob FROM patients WHERE id = ?"
    return db.execute(query, (patient_id,))
        ''',
        'target_system': 'ehr_application',
        'required_permissions': ['read_phi'],
        'electronic_signature': {
            'user_id': 'dev_john_doe',
            'timestamp': datetime.utcnow().isoformat(),
            'meaning': 'Authorized this change',
            'hash': 'sha256:abcd1234',
            'record_link': 'change_001'
        }
    }
    
    context = {
        'user': 'dev_john_doe',
        'user_role': 'developer',
        'environment': 'staging',
        'data_types': ['phi', 'medical'],
        'system': 'ehr_application',
        'audit_enabled': True,
        'audit_config': {
            'fields': ['user_id', 'timestamp', 'action_type', 
                      'affected_data', 'source_ip', 'result'],
            'secure_timestamp': True,
            'editable': False,
            'review_process': True
        },
        'security_config': {
            'encryption': {
                'at_rest': {'enabled': True},
                'in_transit': {'enabled': True},
                'key_management': {'hsm_enabled': True}
            }
        }
    }
    
    # Validate the action
    validation = regulated.validate_healing_action(action, context)
    
    print(f"Validation ID: {validation.validation_id}")
    print(f"Passed: {validation.passed}")
    print(f"Industry: {validation.industry.value}")
    print(f"Requirements Checked: {[r.value for r in validation.requirements_checked]}")
    
    if validation.failures:
        print("\nFailures:")
        for failure in validation.failures:
            print(f"  - {failure['requirement']}: {failure['message']}")
            print(f"    Remediation: {failure['remediation']}")
    
    if validation.warnings:
        print("\nWarnings:")
        for warning in validation.warnings:
            print(f"  - {warning['requirement']}: {warning['message']}")
    
    if validation.recommendations:
        print("\nRecommendations:")
        for rec in validation.recommendations:
            print(f"  - {rec}")
    
    return validation


def validate_financial_action():
    """Example of validating a financial services action."""
    print("\n=== Financial Services Action Validation Example ===")
    
    regulated = get_regulated_industries()
    
    # Simulate a change to payment processing
    action = {
        'id': 'heal_002',
        'type': 'database_change',
        'description': 'Update payment processing stored procedure',
        'patch': '''
CREATE OR REPLACE PROCEDURE process_payment(
    @amount DECIMAL(10,2),
    @merchant_id VARCHAR(50)
) AS
BEGIN
    -- Updated logic with better error handling
    BEGIN TRANSACTION;
    -- Processing logic here
    COMMIT;
END;
        ''',
        'target_system': 'payment_gateway',
        'approvals': {
            'change_advisory_board': True,
            'security_review': True
        },
        'testing': {
            'unit_testing': True,
            'integration_testing': True,
            'user_acceptance_testing': True,
            'security_testing': True
        },
        'rollback_plan': 'Restore previous stored procedure version from backup'
    }
    
    context = {
        'user': 'dba_admin',
        'user_roles': ['database_admin'],
        'environment': 'production',
        'data_types': ['financial', 'payment', 'cardholder'],
        'affected_systems': ['payment_gateway', 'general_ledger'],
        'network_config': {
            'segmentation_enabled': True,
            'firewall_rules_reviewed': True
        }
    }
    
    validation = regulated.validate_healing_action(action, context)
    
    print(f"Validation ID: {validation.validation_id}")
    print(f"Passed: {validation.passed}")
    print(f"Failures: {len(validation.failures)}")
    print(f"Warnings: {len(validation.warnings)}")
    
    return validation


def get_compliance_dashboards():
    """Get compliance dashboards for all configured industries."""
    print("\n=== Compliance Dashboards ===")
    
    regulated = get_regulated_industries()
    
    industries = [
        RegulatedIndustry.HEALTHCARE,
        RegulatedIndustry.FINANCIAL_SERVICES,
        RegulatedIndustry.GOVERNMENT_DEFENSE,
        RegulatedIndustry.PHARMACEUTICAL,
        RegulatedIndustry.TELECOM_UTILITIES
    ]
    
    for industry in industries:
        dashboard = regulated.get_industry_dashboard(industry)
        
        if 'error' not in dashboard:
            print(f"\n{industry.value.upper()} Dashboard:")
            print(f"  Validation Level: {dashboard['configuration']['validation_level']}")
            print(f"  Enabled Requirements: {len(dashboard['configuration']['enabled_requirements'])}")
            print(f"  Compliance Status: {json.dumps(dashboard.get('compliance_status', {}), indent=4)}")
            print(f"  Validation Pass Rate: {dashboard['validation_metrics']['pass_rate']:.2f}%")
            print(f"  Active Workflows: {dashboard['active_workflows']}")


def demonstrate_multi_industry_validation():
    """Demonstrate validation across multiple industries."""
    print("\n=== Multi-Industry Validation Example ===")
    
    regulated = get_regulated_industries()
    
    # Healthcare payment integration (affects both healthcare and financial)
    action = {
        'id': 'heal_003',
        'type': 'integration_update',
        'description': 'Update healthcare payment processing integration',
        'patch': '''
def process_medical_payment(patient_id, amount, insurance_info):
    # Validate insurance coverage
    coverage = validate_insurance(patient_id, insurance_info)
    
    # Process payment with tokenization
    payment_token = tokenize_payment_info(insurance_info.payment_method)
    
    # Create transaction
    transaction = create_medical_transaction(
        patient_id=patient_id,
        amount=amount,
        payment_token=payment_token,
        coverage=coverage
    )
    
    return transaction.id
        ''',
        'target_system': 'billing_integration'
    }
    
    context = {
        'user': 'integration_developer',
        'environment': 'production',
        'data_types': ['phi', 'medical', 'financial', 'payment'],
        'affected_systems': ['ehr_application', 'billing_system', 'payment_gateway'],
        'audit_enabled': True,
        'security_config': {
            'encryption': {
                'at_rest': {'enabled': True},
                'in_transit': {'enabled': True},
                'key_management': {'hsm_enabled': True}
            }
        },
        'network_config': {
            'segmentation_enabled': True,
            'firewall_rules_reviewed': True
        }
    }
    
    validation = regulated.validate_healing_action(action, context)
    
    print(f"Validation ID: {validation.validation_id}")
    print(f"Passed: {validation.passed}")
    print(f"Industries Checked: Multiple (Healthcare + Financial)")
    print(f"Total Requirements Checked: {len(validation.requirements_checked)}")
    print(f"Requirements: {[r.value for r in validation.requirements_checked]}")
    
    if validation.failures:
        print(f"\nTotal Failures: {len(validation.failures)}")
        # Group failures by industry
        healthcare_failures = [f for f in validation.failures if 'HIPAA' in f.get('requirement', '')]
        financial_failures = [f for f in validation.failures if any(x in f.get('requirement', '') for x in ['SOX', 'PCI'])]
        
        if healthcare_failures:
            print(f"  Healthcare Failures: {len(healthcare_failures)}")
        if financial_failures:
            print(f"  Financial Failures: {len(financial_failures)}")


def main():
    """Main demonstration function."""
    print("=== Homeostasis Regulated Industries Support Demo ===")
    print("This demo shows how to configure and use compliance controls")
    print("for various regulated industries.")
    
    # Setup all industries
    setup_healthcare_compliance()
    setup_financial_compliance()
    setup_government_compliance()
    setup_pharmaceutical_compliance()
    setup_telecom_compliance()
    
    # Demonstrate validations
    validate_healthcare_action()
    validate_financial_action()
    
    # Show dashboards
    get_compliance_dashboards()
    
    # Demonstrate multi-industry validation
    demonstrate_multi_industry_validation()
    
    print("\n=== Demo Complete ===")
    print("The regulated industries support system is now configured and ready")
    print("to validate healing actions according to industry-specific compliance requirements.")


if __name__ == "__main__":
    main()