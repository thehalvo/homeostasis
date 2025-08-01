"""
Test suite for Regulated Industries Support.

Tests industry-specific compliance controls, validation workflows,
and resilience features for healthcare, financial services, government,
pharmaceutical, and telecommunications sectors.
"""

import datetime
import json
import pytest
from unittest.mock import Mock, patch

from modules.security.regulated_industries import (
    RegulatedIndustriesSupport,
    RegulatedIndustry,
    ComplianceRequirement,
    ValidationLevel,
    get_regulated_industries
)


class TestRegulatedIndustriesSupport:
    """Test cases for regulated industries support."""
    
    @pytest.fixture
    def config(self, tmp_path):
        """Test configuration."""
        return {
            'storage_path': str(tmp_path / 'regulated'),
            'audit_enabled': True,
            'compliance_frameworks': ['HIPAA', 'SOX', 'PCI-DSS', 'FedRAMP']
        }
    
    @pytest.fixture
    def regulated_support(self, config):
        """Create regulated industries support instance."""
        with patch('modules.security.regulated_industries.get_compliance_reporting'):
            with patch('modules.security.regulated_industries.get_governance_framework'):
                with patch('modules.security.regulated_industries.get_policy_engine'):
                    with patch('modules.security.regulated_industries.get_rbac_manager'):
                        with patch('modules.security.regulated_industries.get_audit_logger'):
                            return RegulatedIndustriesSupport(config)
    
    def test_healthcare_configuration(self, regulated_support):
        """Test healthcare industry configuration."""
        # Configure healthcare with HIPAA requirements
        requirements = [
            ComplianceRequirement.HIPAA_PRIVACY,
            ComplianceRequirement.HIPAA_SECURITY,
            ComplianceRequirement.HIPAA_BREACH
        ]
        
        config = {
            'validation_level': 'enhanced',
            'critical_systems': ['patient_records', 'ehr_system', 'lab_results'],
            'data_classifications': {
                'phi': 'highly_sensitive',
                'medical_records': 'protected',
                'billing': 'sensitive'
            },
            'approval_thresholds': {
                'standard': 1,
                'enhanced': 2,
                'critical': 3,
                'timeout': 120
            },
            'retention_policies': {
                'audit_logs': 2555,  # 7 years
                'phi_access_logs': 2190,  # 6 years
                'security_events': 365
            },
            'encryption_requirements': {
                'algorithm': 'AES-256',
                'key_rotation': 'quarterly',
                'at_rest': 'required',
                'in_transit': 'required'
            },
            'audit_frequency': 'continuous'
        }
        
        result = regulated_support.configure_industry(
            RegulatedIndustry.HEALTHCARE,
            requirements,
            config
        )
        
        assert result is True
        assert RegulatedIndustry.HEALTHCARE in regulated_support.industry_configs
        
        # Verify configuration
        industry_config = regulated_support.industry_configs[RegulatedIndustry.HEALTHCARE]
        assert len(industry_config.enabled_requirements) == 3
        assert industry_config.validation_level == ValidationLevel.ENHANCED
        assert 'patient_records' in industry_config.critical_systems
    
    def test_financial_services_configuration(self, regulated_support):
        """Test financial services configuration."""
        requirements = [
            ComplianceRequirement.SOX_CONTROLS,
            ComplianceRequirement.PCI_DSS_LEVEL1,
            ComplianceRequirement.FINRA_COMPLIANCE
        ]
        
        config = {
            'validation_level': 'critical',
            'critical_systems': ['general_ledger', 'payment_processing', 'trading_platform'],
            'data_classifications': {
                'cardholder_data': 'highly_sensitive',
                'financial_records': 'protected',
                'transaction_logs': 'sensitive'
            },
            'approval_thresholds': {
                'timeout': 60
            },
            'retention_policies': {
                'financial_records': 2555,  # 7 years for SOX
                'transaction_logs': 1095,  # 3 years
                'audit_trails': 2190  # 6 years
            },
            'encryption_requirements': {
                'algorithm': 'AES-256',
                'tokenization': 'enabled',
                'key_management': 'hsm_required'
            }
        }
        
        result = regulated_support.configure_industry(
            RegulatedIndustry.FINANCIAL_SERVICES,
            requirements,
            config
        )
        
        assert result is True
        assert RegulatedIndustry.FINANCIAL_SERVICES in regulated_support.industry_configs
    
    def test_healthcare_validation(self, regulated_support):
        """Test healthcare compliance validation."""
        # Configure healthcare
        regulated_support.configure_industry(
            RegulatedIndustry.HEALTHCARE,
            [ComplianceRequirement.HIPAA_PRIVACY, ComplianceRequirement.HIPAA_SECURITY],
            {'validation_level': 'enhanced'}
        )
        
        # Test action with PHI
        action = {
            'id': 'action_001',
            'type': 'patch_application',
            'patch': 'Fix patient record display issue for patient ID 12345',
            'target_system': 'ehr_system',
            'required_permissions': ['read_phi', 'write_phi']
        }
        
        context = {
            'user': 'developer1',
            'user_role': 'developer',
            'environment': 'production',
            'data_types': ['phi', 'medical'],
            'system': 'ehr_system',
            'audit_enabled': True,
            'audit_config': {
                'fields': ['user_id', 'timestamp', 'action_type', 
                          'affected_data', 'source_ip', 'result']
            },
            'security_config': {
                'encryption': {
                    'at_rest': {'enabled': True},
                    'in_transit': {'enabled': True},
                    'key_management': {'hsm_enabled': True}
                }
            }
        }
        
        validation = regulated_support.validate_healing_action(action, context)
        
        assert validation.passed is False  # Should fail due to PHI in patch
        assert len(validation.failures) > 0
        assert any('PHI' in f['message'] for f in validation.failures)
        assert RegulatedIndustry.HEALTHCARE in [validation.industry]
    
    def test_financial_validation(self, regulated_support):
        """Test financial services compliance validation."""
        # Configure financial services
        regulated_support.configure_industry(
            RegulatedIndustry.FINANCIAL_SERVICES,
            [ComplianceRequirement.PCI_DSS_LEVEL1, ComplianceRequirement.SOX_CONTROLS],
            {'validation_level': 'critical'}
        )
        
        # Test action affecting financial systems
        action = {
            'id': 'action_002',
            'type': 'database_change',
            'patch': 'Update transaction processing logic',
            'target_system': 'payment_gateway',
            'approvals': {
                'change_advisory_board': True
            },
            'testing': {
                'user_acceptance_testing': True
            },
            'rollback_plan': 'Documented rollback procedure'
        }
        
        context = {
            'user': 'admin1',
            'environment': 'production',
            'data_types': ['financial', 'payment'],
            'affected_systems': ['payment_processing', 'general_ledger'],
            'network_config': {
                'segmentation_enabled': True,
                'firewall_rules_reviewed': True
            },
            'user_roles': ['admin']
        }
        
        validation = regulated_support.validate_healing_action(action, context)
        
        # Should pass basic SOX controls
        sox_failures = [f for f in validation.failures 
                       if 'SOX' in f.get('requirement', '')]
        assert len(sox_failures) == 0
    
    def test_government_validation(self, regulated_support):
        """Test government and defense compliance validation."""
        # Configure government
        regulated_support.configure_industry(
            RegulatedIndustry.GOVERNMENT_DEFENSE,
            [ComplianceRequirement.FEDRAMP_MODERATE, ComplianceRequirement.CMMC_LEVEL3],
            {'validation_level': 'critical'}
        )
        
        # Test action with proper controls
        action = {
            'id': 'action_003',
            'type': 'security_patch',
            'patch': 'Apply security update to authentication system',
            'target_boundary': 'authorized_boundary',
            'vendor': 'approved_vendor_001',
            'components': [
                {'name': 'auth_module', 'integrity_verified': True},
                {'name': 'crypto_lib', 'integrity_verified': True}
            ]
        }
        
        context = {
            'user': 'security_admin',
            'environment': 'government_cloud',
            'data_types': ['government'],
            'source_boundary': 'authorized_boundary',
            'approved_vendors': ['approved_vendor_001', 'approved_vendor_002']
        }
        
        validation = regulated_support.validate_healing_action(action, context)
        
        # Should pass with proper controls
        fedramp_failures = [f for f in validation.failures 
                           if 'FEDRAMP' in f.get('requirement', '')]
        assert len(fedramp_failures) == 0
    
    def test_pharmaceutical_validation(self, regulated_support):
        """Test pharmaceutical compliance validation."""
        # Configure pharmaceutical
        regulated_support.configure_industry(
            RegulatedIndustry.PHARMACEUTICAL,
            [ComplianceRequirement.FDA_21_CFR_11, ComplianceRequirement.GMP_COMPLIANCE],
            {'validation_level': 'critical'}
        )
        
        # Test action with proper validation
        action = {
            'id': 'action_004',
            'type': 'system_update',
            'patch': 'Update batch processing algorithm',
            'electronic_signature': {
                'user_id': 'quality_manager',
                'timestamp': datetime.datetime.utcnow().isoformat(),
                'meaning': 'Approved for GMP compliance',
                'hash': 'abc123def456',
                'record_link': 'change_record_004'
            },
            'change_control': {
                'quality_approval': True,
                'impact_assessment': True,
                'requires_validation': True,
                'validation_completed': True
            },
            'data_handling': {
                'source_attribution': True,
                'legibility_ensured': True,
                'real_time_recording': True,
                'original_preserved': True,
                'accuracy_verified': True
            }
        }
        
        context = {
            'user': 'validation_engineer',
            'environment': 'gmp_production',
            'data_types': ['pharmaceutical', 'clinical'],
            'audit_config': {
                'secure_timestamp': True,
                'editable': False,
                'review_process': True
            }
        }
        
        validation = regulated_support.validate_healing_action(action, context)
        
        # Should pass with proper controls
        assert validation.passed is True
        assert len(validation.failures) == 0
    
    def test_telecom_validation(self, regulated_support):
        """Test telecommunications compliance validation."""
        # Configure telecom
        regulated_support.configure_industry(
            RegulatedIndustry.TELECOM_UTILITIES,
            [ComplianceRequirement.NERC_CIP, ComplianceRequirement.E911_COMPLIANCE],
            {'validation_level': 'critical'}
        )
        
        # Test action affecting critical infrastructure
        action = {
            'id': 'action_005',
            'type': 'network_change',
            'patch': 'Update routing configuration',
            'affects_critical_services': True,
            'redundancy_maintained': True,
            'maintenance_window_approved': True,
            'change_documentation': True,
            'security_impact_assessment': True,
            'cip_testing_completed': True
        }
        
        context = {
            'user': 'network_engineer',
            'environment': 'production',
            'data_types': ['telecom'],
            'affected_systems': ['routing_system'],
            'bes_cyber_systems': ['routing_system', 'control_system'],
            'affected_services': ['voice_routing']
        }
        
        validation = regulated_support.validate_healing_action(action, context)
        
        # Should pass with proper controls
        nerc_failures = [f for f in validation.failures 
                        if 'CIP' in f.get('requirement', '')]
        assert len(nerc_failures) == 0
    
    def test_validation_workflow_creation(self, regulated_support):
        """Test creation of validation workflows."""
        # Configure healthcare
        regulated_support.configure_industry(
            RegulatedIndustry.HEALTHCARE,
            [ComplianceRequirement.HIPAA_PRIVACY],
            {'validation_level': 'enhanced'}
        )
        
        # Create validation workflow
        steps = [
            {'step': 1, 'action': 'privacy_assessment', 'required': True},
            {'step': 2, 'action': 'security_review', 'required': True},
            {'step': 3, 'action': 'compliance_approval', 'required': True}
        ]
        
        workflow_id = regulated_support.create_validation_workflow(
            RegulatedIndustry.HEALTHCARE,
            'phi_data_change',
            steps
        )
        
        assert workflow_id is not None
        assert workflow_id in regulated_support.validation_workflows
        
        workflow = regulated_support.validation_workflows[workflow_id]
        assert workflow.industry == RegulatedIndustry.HEALTHCARE
        assert len(workflow.steps) == 3
        assert 'privacy_officer' in workflow.required_approvers
    
    def test_industry_dashboard(self, regulated_support):
        """Test industry-specific dashboard."""
        # Configure and create some data
        regulated_support.configure_industry(
            RegulatedIndustry.HEALTHCARE,
            [ComplianceRequirement.HIPAA_PRIVACY, ComplianceRequirement.HIPAA_SECURITY],
            {'validation_level': 'enhanced'}
        )
        
        # Create some validations
        for i in range(5):
            action = {'id': f'action_{i}', 'patch': 'Test patch'}
            context = {'user': 'test_user', 'data_types': ['phi']}
            regulated_support.validate_healing_action(action, context)
        
        # Get dashboard
        dashboard = regulated_support.get_industry_dashboard(RegulatedIndustry.HEALTHCARE)
        
        assert dashboard['industry'] == 'healthcare'
        assert 'configuration' in dashboard
        assert 'validation_metrics' in dashboard
        assert dashboard['validation_metrics']['total_validations'] >= 5
    
    def test_multi_industry_validation(self, regulated_support):
        """Test validation across multiple industries."""
        # Configure multiple industries
        regulated_support.configure_industry(
            RegulatedIndustry.HEALTHCARE,
            [ComplianceRequirement.HIPAA_PRIVACY],
            {'validation_level': 'enhanced', 'critical_systems': ['ehr_system']}
        )
        
        regulated_support.configure_industry(
            RegulatedIndustry.FINANCIAL_SERVICES,
            [ComplianceRequirement.PCI_DSS_LEVEL1],
            {'validation_level': 'critical', 'critical_systems': ['payment_system']}
        )
        
        # Test action affecting both industries
        action = {
            'id': 'action_multi',
            'type': 'integration_update',
            'patch': 'Update healthcare payment integration'
        }
        
        context = {
            'user': 'integration_admin',
            'data_types': ['phi', 'payment'],  # Both healthcare and financial
            'system': 'payment_integration'
        }
        
        validation = regulated_support.validate_healing_action(action, context)
        
        # Should check requirements for both industries
        assert len(validation.requirements_checked) > 0
        hipaa_checked = any(r.value.startswith('hipaa') for r in validation.requirements_checked)
        pci_checked = any(r.value.startswith('pci') for r in validation.requirements_checked)
        
        # At least one industry's requirements should be checked
        assert hipaa_checked or pci_checked
    
    def test_emergency_validation_level(self, regulated_support):
        """Test emergency validation level handling."""
        # Configure with emergency level
        regulated_support.configure_industry(
            RegulatedIndustry.HEALTHCARE,
            [ComplianceRequirement.HIPAA_SECURITY],
            {'validation_level': 'emergency'}
        )
        
        # Create emergency workflow
        workflow_id = regulated_support.create_validation_workflow(
            RegulatedIndustry.HEALTHCARE,
            'emergency_patch',
            [{'step': 1, 'action': 'emergency_approval', 'required': True}]
        )
        
        workflow = regulated_support.validation_workflows[workflow_id]
        assert 'emergency_approver' in workflow.required_approvers
        
    def test_compliance_evidence_collection(self, regulated_support):
        """Test evidence collection for compliance validation."""
        # Configure industry
        regulated_support.configure_industry(
            RegulatedIndustry.PHARMACEUTICAL,
            [ComplianceRequirement.FDA_21_CFR_11],
            {'validation_level': 'critical'}
        )
        
        # Mock audit logger to return events
        regulated_support.audit_logger.search_events = Mock(return_value=[
            {'event_id': 'evt_001', 'event_type': 'action_executed'},
            {'event_id': 'evt_002', 'event_type': 'validation_completed'}
        ])
        
        # Test action with evidence
        action = {
            'id': 'action_evidence',
            'approval_id': 'appr_001',
            'test_results': {'id': 'test_001'},
            'validation_docs': [
                {'id': 'val_doc_001'},
                {'id': 'val_doc_002'}
            ]
        }
        
        context = {
            'user': 'validator',
            'data_types': ['pharmaceutical']
        }
        
        validation = regulated_support.validate_healing_action(action, context)
        
        # Should collect various evidence types
        assert len(validation.evidence_collected) > 0
        assert any('approval' in e for e in validation.evidence_collected)
        assert any('test' in e for e in validation.evidence_collected)
    
    def test_data_detection_patterns(self, regulated_support):
        """Test sensitive data detection patterns."""
        # Test PHI detection
        phi_samples = [
            "Patient SSN: 123-45-6789",
            "Medical Record: M12345678",
            "Diagnosis: Type 2 Diabetes",
            "Prescription for medication",
            "Lab result showing elevated levels"
        ]
        
        for sample in phi_samples:
            assert regulated_support._contains_phi(sample) is True
        
        # Test cardholder data detection
        card_samples = [
            "Card number: 4111 1111 1111 1111",
            "CVV: 123",
            "Credit card payment processed",
            "Cardholder name: John Doe"
        ]
        
        for sample in card_samples:
            assert regulated_support._contains_cardholder_data(sample) is True
        
        # Test CUI detection
        cui_samples = [
            "Document marked CONFIDENTIAL",
            "CUI Category: Export Controlled",
            "For Official Use Only (FOUO)",
            "ITAR restricted information"
        ]
        
        for sample in cui_samples:
            assert regulated_support._contains_cui(sample) is True
    
    def test_industry_specific_recommendations(self, regulated_support):
        """Test generation of industry-specific recommendations."""
        # Configure healthcare
        regulated_support.configure_industry(
            RegulatedIndustry.HEALTHCARE,
            [ComplianceRequirement.HIPAA_PRIVACY],
            {'validation_level': 'enhanced'}
        )
        
        # Create validation with failures
        action = {
            'id': 'action_rec',
            'patch': 'Update patient data display for SSN 123-45-6789'
        }
        
        context = {
            'user': 'developer',
            'data_types': ['phi']
        }
        
        validation = regulated_support.validate_healing_action(action, context)
        
        # Should have healthcare-specific recommendations
        assert len(validation.recommendations) > 0
        assert any('HIPAA' in rec or 'PHI' in rec for rec in validation.recommendations)
    
    def test_configuration_persistence(self, regulated_support, tmp_path):
        """Test saving and loading configurations."""
        # Configure multiple industries
        regulated_support.configure_industry(
            RegulatedIndustry.HEALTHCARE,
            [ComplianceRequirement.HIPAA_PRIVACY],
            {'validation_level': 'enhanced', 'critical_systems': ['ehr']}
        )
        
        regulated_support.configure_industry(
            RegulatedIndustry.FINANCIAL_SERVICES,
            [ComplianceRequirement.SOX_CONTROLS],
            {'validation_level': 'critical', 'critical_systems': ['ledger']}
        )
        
        # Create a workflow
        workflow_id = regulated_support.create_validation_workflow(
            RegulatedIndustry.HEALTHCARE,
            'test_workflow',
            [{'step': 1, 'action': 'test'}]
        )
        
        # Save configurations
        regulated_support._save_configurations()
        
        # Create new instance and verify it loads configurations
        new_support = RegulatedIndustriesSupport({'storage_path': str(tmp_path / 'regulated')})
        
        # Verify configurations were loaded
        assert RegulatedIndustry.HEALTHCARE in new_support.industry_configs
        assert RegulatedIndustry.FINANCIAL_SERVICES in new_support.industry_configs
        assert workflow_id in new_support.validation_workflows
        
        # Verify details match
        healthcare_config = new_support.industry_configs[RegulatedIndustry.HEALTHCARE]
        assert healthcare_config.validation_level == ValidationLevel.ENHANCED
        assert 'ehr' in healthcare_config.critical_systems


if __name__ == '__main__':
    pytest.main([__file__, '-v'])