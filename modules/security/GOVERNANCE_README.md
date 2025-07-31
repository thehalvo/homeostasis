# Enterprise Governance Framework for Homeostasis

## Overview

The Enterprise Governance Framework provides comprehensive control and oversight capabilities for Homeostasis in mission-critical and regulated environments. It ensures that all automated healing actions comply with organizational policies, regulatory requirements, and security best practices.

## Key Features

### 1. Role-Based Access Control (RBAC)
- Fine-grained permission management
- Predefined roles: Admin, Operator, Developer, Reviewer, Analyst
- Custom role creation and management
- Dynamic permission assignment

### 2. User Management System
- Comprehensive user lifecycle management
- Password policies and enforcement
- Multi-factor authentication support
- Session management
- Group-based permissions

### 3. Approval Workflows
- Multi-stage approval processes
- Conditional routing based on risk and context
- SLA management and escalation
- Customizable workflow templates
- Integration with external approval systems

### 4. Compliance Reporting
- Support for major frameworks: SOC2, HIPAA, PCI-DSS, ISO 27001, GDPR, SOX
- Automated evidence collection
- Real-time compliance monitoring
- Comprehensive audit trails
- Export capabilities (JSON, CSV, HTML)

### 5. Identity Provider Integration
- OAuth 2.0 support
- SAML 2.0 integration
- LDAP/Active Directory authentication
- OpenID Connect compatibility
- SSO capabilities

### 6. Policy Enforcement
- Dynamic policy evaluation
- Context-aware rules
- Risk-based decision making
- Policy versioning and history
- Real-time violation detection

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 Enterprise Governance Framework              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │    RBAC     │  │     User     │  │    Identity      │  │
│  │  Manager    │  │  Management  │  │   Providers      │  │
│  └─────────────┘  └──────────────┘  └──────────────────┘  │
│                                                              │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  Approval   │  │  Compliance  │  │     Policy       │  │
│  │  Workflows  │  │  Reporting   │  │   Enforcement    │  │
│  └─────────────┘  └──────────────┘  └──────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                   Audit Logger                        │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Basic Configuration

```python
from modules.security.governance_framework import (
    get_governance_framework, 
    GovernanceConfig, 
    GovernanceCapability,
    ComplianceFramework
)

# Configure governance
config = GovernanceConfig(
    enabled_capabilities=[
        GovernanceCapability.RBAC,
        GovernanceCapability.USER_MANAGEMENT,
        GovernanceCapability.APPROVAL_WORKFLOWS,
        GovernanceCapability.COMPLIANCE_REPORTING,
        GovernanceCapability.POLICY_ENFORCEMENT
    ],
    compliance_frameworks=[
        ComplianceFramework.SOC2,
        ComplianceFramework.HIPAA
    ],
    require_approval_for_production=True,
    enforce_policies=True
)

# Initialize framework
governance = get_governance_framework(config)
```

### 2. User Management

```python
# Create a user
user_id = governance.manage_user(
    action='create',
    username='john.doe',
    email='john.doe@company.com',
    password='SecurePassword123!',
    full_name='John Doe',
    roles=['developer'],
    department='Engineering'
)

# Authenticate user
auth_result = governance.authenticate_user(
    username='john.doe',
    password='SecurePassword123!'
)
```

### 3. Evaluate Healing Actions

```python
from modules.security.governance_framework import HealingActionRequest

# Create healing action request
request = HealingActionRequest(
    request_id='heal_12345',
    action_type='code_fix',
    error_context={
        'error_type': 'AttributeError',
        'message': 'NoneType has no attribute foo'
    },
    patch_content='if obj is not None:\n    obj.foo()',
    environment='production',
    service_name='user_service',
    language='python',
    requested_by=user_id
)

# Evaluate against governance rules
decision = governance.evaluate_healing_action(request)

if decision.allowed:
    print("Healing action approved")
elif decision.approval_required:
    print(f"Approval required: {decision.approval_request_id}")
else:
    print(f"Action denied: {decision.reason}")
```

### 4. Configure Policies

```python
# Create a security policy
policy_id = governance.configure_policy(
    name='No Hardcoded Secrets',
    description='Prevent patches with hardcoded credentials',
    type='security',
    scope='global',
    rules=[
        {
            'name': 'Check for passwords',
            'conditions': [
                {
                    'field': 'patch_content',
                    'operator': 'regex',
                    'value': r'password\s*=\s*["\'][^"\']+["\']'
                }
            ],
            'action': 'deny'
        }
    ]
)
```

### 5. Generate Compliance Reports

```python
# Generate SOC2 compliance report
report_id = governance.generate_compliance_report(
    framework=ComplianceFramework.SOC2,
    report_type='assessment'
)

# Export report
from modules.security.compliance_reporting import ReportFormat
export_path = governance.compliance_reporting.export_report(
    report_id=report_id,
    format=ReportFormat.HTML
)
```

## Use Cases

### 1. Production Environment Protection

```python
# Policy for production database changes
governance.configure_policy(
    name='Production Database Protection',
    type='operational',
    scope='environment',
    rules=[
        {
            'name': 'Require approval for DB changes',
            'conditions': [
                {'field': 'environment', 'operator': 'equals', 'value': 'production'},
                {'field': 'patch_content', 'operator': 'regex', 'value': r'(ALTER|DROP|DELETE)'}
            ],
            'action': 'require_approval'
        }
    ]
)
```

### 2. HIPAA Compliance

```python
# Configure HIPAA-specific policies
governance.configure_policy(
    name='PHI Protection',
    type='compliance',
    compliance_frameworks=[ComplianceFramework.HIPAA],
    rules=[
        {
            'name': 'Prevent PHI exposure',
            'conditions': [
                {'field': 'service_name', 'operator': 'in', 
                 'value': ['patient_service', 'medical_records']},
                {'field': 'patch_content', 'operator': 'regex', 
                 'value': r'(ssn|patient|medical|diagnosis)'}
            ],
            'action': 'require_approval'
        }
    ]
)
```

### 3. Multi-Stage Approval Workflow

```python
# Create custom workflow template
workflow_id = governance.workflow_engine.create_workflow_template(
    name='Critical Security Fix',
    description='Multi-stage approval for security patches',
    category='security',
    stages=[
        {
            'name': 'Security Team Review',
            'approver_roles': ['security_reviewer'],
            'min_approvals': 2,
            'timeout_hours': 4
        },
        {
            'name': 'Management Approval',
            'approver_roles': ['manager', 'director'],
            'min_approvals': 1,
            'timeout_hours': 8,
            'condition': 'risk_level',
            'condition_params': {'risk_levels': ['high', 'critical']}
        }
    ]
)
```

### 4. SSO Integration

```python
# Configure OAuth2 provider
provider_id = governance.identity_integration.configure_provider(
    name='Company OAuth',
    type='oauth2',
    config={
        'client_id': 'your-client-id',
        'client_secret': 'your-client-secret',
        'authorization_url': 'https://auth.company.com/oauth/authorize',
        'token_url': 'https://auth.company.com/oauth/token',
        'userinfo_url': 'https://auth.company.com/userinfo'
    }
)

# Initiate SSO
sso_response = governance.initiate_sso(
    provider_name='Company OAuth',
    redirect_uri='https://homeostasis.company.com/auth/callback'
)
```

## API Reference

### Core Classes

#### GovernanceConfig
Configuration for the governance framework.

**Parameters:**
- `enabled_capabilities`: List of enabled governance capabilities
- `compliance_frameworks`: List of compliance frameworks to enforce
- `require_approval_for_production`: Whether to require approval for production changes
- `enforce_policies`: Whether to enforce policies
- `enable_sso`: Whether to enable SSO
- `audit_retention_days`: Number of days to retain audit logs
- `session_timeout_minutes`: Session timeout in minutes
- `mfa_required_roles`: Roles that require MFA

#### HealingActionRequest
Request for a healing action that requires governance evaluation.

**Parameters:**
- `request_id`: Unique request identifier
- `action_type`: Type of healing action
- `error_context`: Error context information
- `patch_content`: Proposed patch content
- `environment`: Target environment
- `service_name`: Service name
- `language`: Programming language
- `requested_by`: User ID requesting the action

#### GovernanceDecision
Decision from governance evaluation.

**Parameters:**
- `allowed`: Whether the action is allowed
- `reason`: Reason for the decision
- `approval_required`: Whether approval is required
- `approval_request_id`: Approval request ID if created
- `workflow_instance_id`: Workflow instance ID if created
- `policy_violations`: List of policy violations
- `compliance_issues`: List of compliance issues

### Decorators

#### @require_approval
Decorator to require governance approval for a function.

```python
from modules.security.governance_framework import require_approval

@require_approval
def apply_patch(context):
    """Apply a patch with governance approval."""
    # Function implementation
    pass
```

## Best Practices

### 1. Policy Design
- Start with restrictive policies and relax as needed
- Use risk-based decision making
- Implement defense in depth
- Regular policy reviews and updates

### 2. User Management
- Enforce strong password policies
- Implement MFA for privileged accounts
- Regular access reviews
- Principle of least privilege

### 3. Compliance
- Map controls to specific requirements
- Automate evidence collection
- Regular compliance assessments
- Maintain audit trails

### 4. Approval Workflows
- Define clear escalation paths
- Set appropriate SLAs
- Implement emergency override procedures
- Regular workflow optimization

### 5. Integration
- Use SSO where possible
- Centralize identity management
- Implement proper session management
- Monitor for anomalous authentication

## Security Considerations

1. **Data Protection**: All sensitive data is encrypted at rest and in transit
2. **Audit Logging**: All governance decisions are logged with full context
3. **Session Security**: Secure session management with configurable timeouts
4. **Access Control**: Fine-grained permissions with RBAC
5. **Compliance**: Built-in support for major compliance frameworks

## Troubleshooting

### Common Issues

1. **Permission Denied Errors**
   - Check user roles and permissions
   - Verify policy configurations
   - Review audit logs for details

2. **SSO Authentication Failures**
   - Verify provider configuration
   - Check redirect URIs
   - Validate certificates

3. **Workflow Timeouts**
   - Review SLA configurations
   - Check escalation settings
   - Monitor approver availability

4. **Policy Conflicts**
   - Review policy priorities
   - Check condition logic
   - Use policy simulation tools

## Performance Optimization

1. **Caching**: User permissions and policy evaluations are cached
2. **Async Processing**: Approval workflows run asynchronously
3. **Batch Operations**: Compliance assessments can be batched
4. **Connection Pooling**: LDAP and database connections are pooled

## Monitoring and Metrics

Key metrics to monitor:
- Authentication success/failure rates
- Policy evaluation performance
- Approval workflow SLA compliance
- Compliance control status
- User activity patterns

## Future Enhancements

1. **Machine Learning**: Anomaly detection for governance violations
2. **Blockchain**: Immutable audit trails
3. **Zero Trust**: Enhanced verification at every step
4. **API Gateway**: Centralized governance enforcement
5. **Mobile Support**: Mobile approval applications

## Support

For issues or questions:
1. Check the documentation
2. Review audit logs
3. Contact your system administrator
4. Submit issues to the Homeostasis project