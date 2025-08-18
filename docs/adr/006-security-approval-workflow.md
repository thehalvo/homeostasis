# ADR-006: Security Approval Workflow

Technical Story: #SEC-001

## Context

Automated code generation and deployment pose significant security risks. Malicious or flawed patches could introduce vulnerabilities, backdoors, or data leaks. While automation is key to self-healing, we need safeguards to prevent security incidents. We must balance security with the need for rapid automated fixes.

## Decision Drivers

- Security: Prevent introduction of vulnerabilities
- Automation: Maintain self-healing capabilities
- Speed: Minimize delay for critical fixes
- Compliance: Meet regulatory requirements
- Auditability: Track all changes and approvals
- Flexibility: Different policies for different environments
- Learning: System improves over time

## Considered Options

1. **Full Manual Approval** - All patches require human review
2. **Fully Automated** - No security checks, trust the system
3. **Risk-Based Approval** - Automatic for low-risk, manual for high-risk
4. **Sandbox Testing Only** - Rely on testing to catch issues
5. **ML-Based Security Analysis** - AI determines security risk

## Decision Outcome

Chosen option: "Risk-Based Approval", implementing automated approval for low-risk changes and requiring human review for high-risk modifications, with ML-assisted risk assessment, because it balances security needs with operational efficiency.

### Positive Consequences

- **Rapid Low-Risk Fixes**: Simple fixes deploy quickly
- **Security for Critical Changes**: High-risk changes get review
- **Learning System**: ML improves risk assessment over time
- **Audit Trail**: All decisions are logged
- **Flexible Policies**: Rules can be customized per environment
- **Gradual Trust Building**: Automation increases as confidence grows
- **Compliance Ready**: Satisfies regulatory requirements

### Negative Consequences

- **Complex Decision Logic**: Risk assessment is non-trivial
- **Potential Delays**: High-risk fixes wait for approval
- **False Positives**: May flag safe changes as risky
- **Reviewer Burden**: Humans must review complex changes
- **Training Required**: ML model needs historical data
- **Policy Management**: Rules need regular updates

## Implementation Details

### Risk Assessment Framework

```python
class RiskAssessment:
    def calculate_risk_score(self, patch: Patch) -> RiskScore:
        scores = {
            'code_complexity': self._assess_complexity(patch),
            'security_patterns': self._check_security_patterns(patch),
            'affected_scope': self._analyze_scope(patch),
            'data_exposure': self._check_data_handling(patch),
            'privilege_changes': self._check_privileges(patch),
            'external_calls': self._analyze_external_calls(patch),
            'historical_success': self._check_history(patch)
        }
        
        weighted_score = self._calculate_weighted_score(scores)
        ml_adjustment = self._ml_risk_prediction(patch, scores)
        
        return RiskScore(
            value=weighted_score + ml_adjustment,
            components=scores,
            category=self._categorize_risk(weighted_score + ml_adjustment)
        )
```

### Risk Categories

```python
class RiskCategory(Enum):
    MINIMAL = "minimal"     # Score 0-20: Auto-approve
    LOW = "low"            # Score 21-40: Auto-approve with notification
    MEDIUM = "medium"      # Score 41-60: Requires one reviewer
    HIGH = "high"          # Score 61-80: Requires two reviewers
    CRITICAL = "critical"  # Score 81-100: Requires security team review
```

### Security Pattern Detection

```python
SECURITY_PATTERNS = {
    'auth_bypass': r'(auth|authenticate|authorize).*=.*false',
    'sql_injection': r'(query|execute).*\+.*input',
    'hardcoded_secrets': r'(password|api_key|secret).*=.*["\']',
    'unsafe_eval': r'eval\s*\(|exec\s*\(',
    'file_access': r'(open|read|write).*file',
    'network_calls': r'(http|request|fetch|curl)',
    'privilege_escalation': r'(sudo|admin|root|privilege)',
    'data_exposure': r'(console\.log|print|debug).*sensitive',
    'crypto_weakness': r'(md5|sha1|des|rc4)',
    'command_injection': r'(system|shell|exec).*\+.*input'
}

def check_security_patterns(code: str) -> List[SecurityIssue]:
    issues = []
    for pattern_name, pattern in SECURITY_PATTERNS.items():
        if re.search(pattern, code, re.IGNORECASE):
            issues.append(SecurityIssue(
                type=pattern_name,
                severity=PATTERN_SEVERITIES[pattern_name],
                line=find_pattern_line(code, pattern)
            ))
    return issues
```

### Approval Workflow

```python
class ApprovalWorkflow:
    async def process_patch(self, patch: Patch) -> ApprovalResult:
        # Step 1: Risk Assessment
        risk_score = self.risk_assessor.calculate_risk_score(patch)
        
        # Step 2: Apply Approval Rules
        if risk_score.category == RiskCategory.MINIMAL:
            return await self._auto_approve(patch, risk_score)
        
        elif risk_score.category == RiskCategory.LOW:
            await self._notify_team(patch, risk_score)
            return await self._auto_approve(patch, risk_score)
        
        elif risk_score.category in [RiskCategory.MEDIUM, RiskCategory.HIGH]:
            return await self._request_review(patch, risk_score)
        
        else:  # CRITICAL
            return await self._security_team_review(patch, risk_score)
    
    async def _request_review(self, patch: Patch, risk_score: RiskScore):
        review_request = ReviewRequest(
            patch=patch,
            risk_score=risk_score,
            required_approvers=self._get_required_approvers(risk_score),
            timeout=self._get_review_timeout(risk_score),
            escalation_path=self._get_escalation_path(risk_score)
        )
        
        return await self.review_system.request_review(review_request)
```

### Notification System

```python
class SecurityNotification:
    def __init__(self):
        self.channels = {
            'slack': SlackNotifier(),
            'email': EmailNotifier(),
            'pagerduty': PagerDutyNotifier()
        }
    
    async def notify(self, event: SecurityEvent):
        severity_channels = {
            'minimal': [],
            'low': ['slack'],
            'medium': ['slack', 'email'],
            'high': ['slack', 'email', 'pagerduty'],
            'critical': ['slack', 'email', 'pagerduty']
        }
        
        for channel in severity_channels[event.severity]:
            await self.channels[channel].send(event)
```

### ML Risk Prediction

```python
class MLRiskPredictor:
    def __init__(self):
        self.model = self._load_model()
        self.feature_extractor = FeatureExtractor()
    
    def predict_risk(self, patch: Patch, initial_scores: dict) -> float:
        features = self.feature_extractor.extract(
            patch=patch,
            initial_scores=initial_scores,
            historical_data=self._get_historical_context(patch)
        )
        
        risk_adjustment = self.model.predict(features)[0]
        
        # Log prediction for continuous learning
        self._log_prediction(patch, features, risk_adjustment)
        
        return risk_adjustment
```

### Audit Trail

```python
class SecurityAudit:
    def log_decision(self, decision: ApprovalDecision):
        audit_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'patch_id': decision.patch.id,
            'risk_score': decision.risk_score.value,
            'risk_category': decision.risk_score.category,
            'decision': decision.result,
            'approvers': decision.approvers,
            'automated': decision.automated,
            'security_issues': decision.security_issues,
            'override_reason': decision.override_reason,
            'environment': decision.environment
        }
        
        # Immutable audit log
        self.audit_store.append(audit_entry)
        
        # Search index for queries
        self.search_index.index(audit_entry)
```

### Policy Configuration

```yaml
security_policies:
  production:
    auto_approve_threshold: 20
    require_security_review: 80
    notification_threshold: 40
    patterns:
      - name: sql_injection
        severity: critical
        action: block
    excluded_files:
      - "*.test.js"
      - "*.spec.py"
  
  staging:
    auto_approve_threshold: 40
    require_security_review: 90
    notification_threshold: 60
```

### Emergency Override

```python
class EmergencyOverride:
    def __init__(self):
        self.required_approvers = ['security-lead', 'engineering-lead']
    
    async def request_override(self, patch: Patch, reason: str) -> bool:
        override_request = {
            'patch': patch,
            'reason': reason,
            'requester': get_current_user(),
            'timestamp': datetime.utcnow()
        }
        
        approvals = await self._get_emergency_approvals(override_request)
        
        if len(approvals) >= len(self.required_approvers):
            self._log_emergency_override(override_request, approvals)
            return True
        
        return False
```

## Links

- [Security Documentation](../modules/security/README.md)
- [ADR-004: LLM Integration Approach](004-llm-integration-approach.md)
- [Governance Documentation](../modules/security/GOVERNANCE_README.md)