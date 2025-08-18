# Security Hardening Guide

This comprehensive guide provides security hardening recommendations for deploying and operating the Homeostasis self-healing framework in production environments. Follow these guidelines to protect against common vulnerabilities and ensure secure operation.

## Table of Contents

1. [Security Principles](#security-principles)
2. [Infrastructure Security](#infrastructure-security)
3. [Application Security](#application-security)
4. [Authentication and Authorization](#authentication-and-authorization)
5. [Data Protection](#data-protection)
6. [Network Security](#network-security)
7. [Container Security](#container-security)
8. [Supply Chain Security](#supply-chain-security)
9. [Monitoring and Incident Response](#monitoring-and-incident-response)
10. [Compliance and Auditing](#compliance-and-auditing)

## Security Principles

### Defense in Depth

Implement multiple layers of security controls:

```yaml
security_layers:
  - network_segmentation
  - firewall_rules
  - authentication
  - authorization
  - encryption
  - monitoring
  - incident_response
```

### Least Privilege

Grant minimum required permissions:

```python
# Role-based access control
class RBACPolicy:
    roles = {
        'viewer': ['read:errors', 'read:patches'],
        'operator': ['read:*', 'execute:patches'],
        'admin': ['read:*', 'write:*', 'execute:*'],
        'security': ['read:*', 'audit:*', 'configure:security']
    }
    
    def check_permission(self, user, action, resource):
        user_permissions = self.get_user_permissions(user)
        required_permission = f"{action}:{resource}"
        
        return any(
            self._matches_permission(perm, required_permission)
            for perm in user_permissions
        )
```

### Zero Trust

Never trust, always verify:

```python
class ZeroTrustGateway:
    def __init__(self):
        self.verifiers = [
            IdentityVerifier(),
            DeviceVerifier(),
            NetworkVerifier(),
            BehaviorVerifier()
        ]
    
    async def authorize_request(self, request):
        for verifier in self.verifiers:
            result = await verifier.verify(request)
            if not result.trusted:
                raise SecurityException(f"Failed {verifier.name}: {result.reason}")
        
        return self.generate_limited_token(request)
```

## Infrastructure Security

### Secure Installation

```bash
#!/bin/bash
# Secure installation script

# Create dedicated user
useradd -r -s /bin/false homeostasis

# Set secure permissions
chmod 750 /opt/homeostasis
chown -R homeostasis:homeostasis /opt/homeostasis

# Disable unnecessary services
systemctl disable telnet
systemctl disable ftp

# Configure firewall
ufw default deny incoming
ufw default allow outgoing
ufw allow 443/tcp  # HTTPS only
ufw enable

# Enable security updates
apt-get install unattended-upgrades
dpkg-reconfigure -plow unattended-upgrades
```

### Hardened Configuration

```yaml
# config/security.yaml
security:
  # Disable debug mode in production
  debug_mode: false
  
  # Enable security headers
  headers:
    X-Frame-Options: DENY
    X-Content-Type-Options: nosniff
    X-XSS-Protection: 1; mode=block
    Strict-Transport-Security: max-age=31536000; includeSubDomains
    Content-Security-Policy: default-src 'self'
  
  # Session security
  session:
    secure_cookie: true
    http_only: true
    same_site: strict
    timeout: 900  # 15 minutes
  
  # Rate limiting
  rate_limits:
    api_calls: 100/minute
    auth_attempts: 5/minute
    patch_generation: 10/minute
```

### Secrets Management

```python
class SecureSecretsManager:
    def __init__(self):
        # Use external secret store
        self.vault_client = hvac.Client(
            url=os.environ['VAULT_ADDR'],
            token=self._get_vault_token()
        )
        
        # Rotate encryption keys
        self.key_rotation_interval = timedelta(days=90)
        self.last_rotation = datetime.now()
    
    def get_secret(self, path):
        # Never log secrets
        with self._no_logging():
            response = self.vault_client.secrets.kv.v2.read_secret_version(
                path=path
            )
            return response['data']['data']
    
    def store_secret(self, path, secret):
        # Encrypt at rest
        encrypted = self._encrypt(secret)
        
        self.vault_client.secrets.kv.v2.create_or_update_secret(
            path=path,
            secret=encrypted
        )
        
        # Audit log (without secret value)
        self.audit_log(f"Secret stored at {path}")
```

## Application Security

### Input Validation

```python
class InputValidator:
    def __init__(self):
        self.validators = {
            'error_message': self._validate_error_message,
            'file_path': self._validate_file_path,
            'code_snippet': self._validate_code_snippet,
            'patch_content': self._validate_patch_content
        }
    
    def validate(self, input_type, value):
        # Sanitize all inputs
        if isinstance(value, str):
            value = self._sanitize_string(value)
        
        # Type-specific validation
        validator = self.validators.get(input_type)
        if not validator:
            raise ValueError(f"Unknown input type: {input_type}")
        
        return validator(value)
    
    def _validate_file_path(self, path):
        # Prevent directory traversal
        if '..' in path or path.startswith('/'):
            raise SecurityError("Invalid file path")
        
        # Whitelist allowed directories
        allowed_dirs = ['/app', '/data', '/logs']
        if not any(path.startswith(d) for d in allowed_dirs):
            raise SecurityError("Path outside allowed directories")
        
        return os.path.normpath(path)
```

### Secure Code Generation

```python
class SecureCodeGenerator:
    def __init__(self):
        self.forbidden_patterns = [
            r'eval\s*\(',
            r'exec\s*\(',
            r'__import__',
            r'subprocess',
            r'os\.system',
            r'shell=True'
        ]
        
        self.sanitizers = {
            'python': PythonSanitizer(),
            'javascript': JavaScriptSanitizer(),
            'sql': SQLSanitizer()
        }
    
    def generate_patch(self, error, context):
        # Generate initial patch
        patch = self._generate_base_patch(error, context)
        
        # Security scanning
        security_issues = self._scan_for_vulnerabilities(patch)
        if security_issues:
            patch = self._fix_security_issues(patch, security_issues)
        
        # Sandbox testing
        sandbox_result = self._test_in_sandbox(patch)
        if not sandbox_result.safe:
            raise SecurityError(f"Patch failed sandbox: {sandbox_result.reason}")
        
        return patch
```

### API Security

```python
class SecureAPIEndpoint:
    def __init__(self):
        self.rate_limiter = RateLimiter()
        self.auth_handler = AuthHandler()
        self.validator = InputValidator()
    
    @require_auth
    @rate_limit(calls=100, period=60)
    @validate_input
    async def handle_request(self, request):
        # Verify API token
        token = request.headers.get('Authorization')
        if not self.auth_handler.verify_token(token):
            raise AuthenticationError("Invalid token")
        
        # Check permissions
        if not self.auth_handler.has_permission(token, request.endpoint):
            raise AuthorizationError("Insufficient permissions")
        
        # Process request with timeout
        try:
            async with timeout(30):
                result = await self.process(request)
        except asyncio.TimeoutError:
            raise RequestTimeout("Request took too long")
        
        # Sanitize response
        return self._sanitize_response(result)
```

## Authentication and Authorization

### Multi-Factor Authentication

```python
class MFAHandler:
    def __init__(self):
        self.totp_secret_key = os.environ['MFA_SECRET']
        self.backup_codes = BackupCodeManager()
    
    def setup_mfa(self, user):
        # Generate TOTP secret
        secret = pyotp.random_base32()
        
        # Store encrypted secret
        encrypted_secret = self.encrypt(secret)
        user.mfa_secret = encrypted_secret
        
        # Generate backup codes
        backup_codes = self.backup_codes.generate(user, count=10)
        
        # Return QR code for authenticator app
        totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(
            name=user.email,
            issuer_name='Homeostasis'
        )
        
        return {
            'qr_code': self.generate_qr(totp_uri),
            'backup_codes': backup_codes
        }
    
    def verify_mfa(self, user, code):
        # Check if it's a backup code
        if self.backup_codes.verify(user, code):
            return True
        
        # Verify TOTP
        secret = self.decrypt(user.mfa_secret)
        totp = pyotp.TOTP(secret)
        
        # Allow for time drift
        return totp.verify(code, valid_window=1)
```

### JWT Token Security

```python
class SecureJWTHandler:
    def __init__(self):
        # Use RS256 for signing
        self.private_key = self._load_private_key()
        self.public_key = self._load_public_key()
        
        # Token configuration
        self.token_lifetime = timedelta(minutes=15)
        self.refresh_lifetime = timedelta(days=7)
    
    def create_token(self, user, scopes):
        now = datetime.utcnow()
        
        payload = {
            'sub': user.id,
            'email': user.email,
            'scopes': scopes,
            'iat': now,
            'exp': now + self.token_lifetime,
            'jti': str(uuid.uuid4())  # Unique token ID
        }
        
        # Sign with private key
        token = jwt.encode(
            payload,
            self.private_key,
            algorithm='RS256'
        )
        
        # Store token ID for revocation
        self.token_store.add(payload['jti'], payload['exp'])
        
        return token
    
    def verify_token(self, token):
        try:
            # Verify with public key
            payload = jwt.decode(
                token,
                self.public_key,
                algorithms=['RS256']
            )
            
            # Check if revoked
            if self.token_store.is_revoked(payload['jti']):
                raise SecurityError("Token has been revoked")
            
            return payload
            
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token expired")
        except jwt.InvalidTokenError:
            raise AuthenticationError("Invalid token")
```

## Data Protection

### Encryption at Rest

```python
class DataEncryption:
    def __init__(self):
        # Use AES-256-GCM
        self.cipher_suite = Fernet(self._get_encryption_key())
        
        # Field-level encryption for sensitive data
        self.encrypted_fields = [
            'api_keys',
            'passwords',
            'personal_data',
            'source_code'
        ]
    
    def encrypt_record(self, record):
        encrypted_record = record.copy()
        
        for field in self.encrypted_fields:
            if field in record:
                # Encrypt with authenticated encryption
                encrypted_value = self.cipher_suite.encrypt(
                    record[field].encode()
                )
                encrypted_record[field] = encrypted_value
        
        return encrypted_record
    
    def encrypt_database(self):
        # Transparent database encryption
        return {
            'encryption': 'AES256',
            'key_management': 'AWS_KMS',
            'rotation_period': '90_days'
        }
```

### Data Masking

```python
class DataMasker:
    def __init__(self):
        self.masking_rules = {
            'email': self._mask_email,
            'ip_address': self._mask_ip,
            'api_key': self._mask_api_key,
            'credit_card': self._mask_credit_card,
            'ssn': self._mask_ssn
        }
    
    def mask_sensitive_data(self, data):
        masked_data = data.copy()
        
        for field, value in data.items():
            # Detect sensitive data type
            data_type = self._detect_data_type(field, value)
            
            if data_type in self.masking_rules:
                masked_data[field] = self.masking_rules[data_type](value)
        
        return masked_data
    
    def _mask_email(self, email):
        parts = email.split('@')
        if len(parts) == 2:
            name = parts[0]
            masked_name = name[0] + '*' * (len(name) - 2) + name[-1]
            return f"{masked_name}@{parts[1]}"
        return '***@***'
```

## Network Security

### TLS Configuration

```nginx
# nginx.conf
server {
    listen 443 ssl http2;
    
    # Modern TLS configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256;
    ssl_prefer_server_ciphers off;
    
    # Certificate configuration
    ssl_certificate /etc/ssl/certs/homeostasis.crt;
    ssl_certificate_key /etc/ssl/private/homeostasis.key;
    
    # OCSP stapling
    ssl_stapling on;
    ssl_stapling_verify on;
    
    # Session configuration
    ssl_session_timeout 1d;
    ssl_session_cache shared:SSL:10m;
    ssl_session_tickets off;
    
    # Security headers
    add_header Strict-Transport-Security "max-age=63072000" always;
}
```

### Network Segmentation

```yaml
# Network security zones
network_zones:
  dmz:
    - load_balancer
    - api_gateway
  
  application:
    - app_servers
    - cache_servers
  
  data:
    - database_servers
    - message_queues
  
  management:
    - monitoring
    - logging

# Firewall rules
firewall_rules:
  - from: dmz
    to: application
    allow: [443]
  
  - from: application
    to: data
    allow: [5432, 6379]  # PostgreSQL, Redis
  
  - from: management
    to: all
    allow: [9090]  # Prometheus
```

### DDoS Protection

```python
class DDoSProtection:
    def __init__(self):
        self.rate_limiter = TokenBucket(
            capacity=1000,
            refill_rate=100
        )
        
        self.ip_reputation = IPReputationService()
        self.geo_blocker = GeoBlocker()
    
    async def check_request(self, request):
        ip = request.remote_addr
        
        # Check IP reputation
        reputation = await self.ip_reputation.check(ip)
        if reputation.score < 0.3:
            raise SecurityError(f"Blocked IP: {ip}")
        
        # Geo-blocking
        country = await self.geo_blocker.get_country(ip)
        if country in self.blocked_countries:
            raise SecurityError(f"Blocked country: {country}")
        
        # Rate limiting
        if not self.rate_limiter.consume(ip):
            raise RateLimitError(f"Rate limit exceeded: {ip}")
        
        # Pattern detection
        if self._detect_attack_pattern(request):
            await self._trigger_mitigation(ip)
            raise SecurityError("Attack pattern detected")
```

## Container Security

### Dockerfile Hardening

```dockerfile
# Use specific version, not latest
FROM python:3.11.5-slim-bookworm

# Run as non-root user
RUN useradd -r -u 1001 -g root homeostasis

# Install security updates
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Copy only necessary files
COPY --chown=homeostasis:root requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=homeostasis:root src/ /app/

# Set secure permissions
RUN chmod -R 550 /app

# Switch to non-root user
USER homeostasis

# Health check
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health')"

# Run with security options
ENTRYPOINT ["python", "-u", "app.py"]
```

### Kubernetes Security

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: homeostasis
spec:
  # Security context
  securityContext:
    runAsNonRoot: true
    runAsUser: 1001
    fsGroup: 2000
    seccompProfile:
      type: RuntimeDefault
  
  containers:
  - name: app
    image: homeostasis:latest
    
    # Container security context
    securityContext:
      allowPrivilegeEscalation: false
      readOnlyRootFilesystem: true
      capabilities:
        drop:
          - ALL
    
    # Resource limits
    resources:
      requests:
        memory: "256Mi"
        cpu: "250m"
      limits:
        memory: "512Mi"
        cpu: "500m"
    
    # Liveness and readiness probes
    livenessProbe:
      httpGet:
        path: /health
        port: 8080
      initialDelaySeconds: 30
      periodSeconds: 10
    
    readinessProbe:
      httpGet:
        path: /ready
        port: 8080
      initialDelaySeconds: 5
      periodSeconds: 5

---
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: homeostasis-netpol
spec:
  podSelector:
    matchLabels:
      app: homeostasis
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: api-gateway
    ports:
    - protocol: TCP
      port: 8080
```

## Supply Chain Security

### Dependency Scanning

```python
class DependencyScanner:
    def __init__(self):
        self.vulnerability_db = VulnerabilityDatabase()
        self.allowed_licenses = ['MIT', 'Apache-2.0', 'BSD-3-Clause']
    
    def scan_dependencies(self, requirements_file):
        vulnerabilities = []
        license_issues = []
        
        # Parse dependencies
        dependencies = self._parse_requirements(requirements_file)
        
        for dep in dependencies:
            # Check for known vulnerabilities
            vulns = self.vulnerability_db.check(dep.name, dep.version)
            if vulns:
                vulnerabilities.extend(vulns)
            
            # Check license compatibility
            license = self._get_license(dep)
            if license not in self.allowed_licenses:
                license_issues.append({
                    'dependency': dep.name,
                    'license': license
                })
        
        # Generate SBOM (Software Bill of Materials)
        sbom = self._generate_sbom(dependencies)
        
        return {
            'vulnerabilities': vulnerabilities,
            'license_issues': license_issues,
            'sbom': sbom
        }
```

### Code Signing

```python
class CodeSigner:
    def __init__(self):
        self.signing_key = self._load_signing_key()
        self.certificate = self._load_certificate()
    
    def sign_release(self, artifact_path):
        # Calculate hash
        sha256_hash = hashlib.sha256()
        with open(artifact_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha256_hash.update(chunk)
        
        # Create signature
        signature = self.signing_key.sign(
            sha256_hash.digest(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        # Create signed manifest
        manifest = {
            'artifact': os.path.basename(artifact_path),
            'hash': sha256_hash.hexdigest(),
            'signature': base64.b64encode(signature).decode(),
            'certificate': self.certificate,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return manifest
```

## Monitoring and Incident Response

### Security Event Monitoring

```python
class SecurityMonitor:
    def __init__(self):
        self.event_patterns = {
            'brute_force': self._detect_brute_force,
            'privilege_escalation': self._detect_privilege_escalation,
            'data_exfiltration': self._detect_data_exfiltration,
            'code_injection': self._detect_code_injection
        }
        
        self.alert_channels = {
            'low': ['email'],
            'medium': ['email', 'slack'],
            'high': ['email', 'slack', 'pagerduty'],
            'critical': ['email', 'slack', 'pagerduty', 'phone']
        }
    
    async def monitor_events(self, event_stream):
        async for event in event_stream:
            # Check all patterns
            for pattern_name, detector in self.event_patterns.items():
                if detector(event):
                    await self.handle_security_event(pattern_name, event)
    
    async def handle_security_event(self, event_type, event):
        # Log event
        self.security_logger.log(event)
        
        # Determine severity
        severity = self._calculate_severity(event_type, event)
        
        # Send alerts
        for channel in self.alert_channels[severity]:
            await self.send_alert(channel, event)
        
        # Trigger automated response
        if severity in ['high', 'critical']:
            await self.incident_response.trigger(event)
```

### Incident Response Plan

```python
class IncidentResponse:
    def __init__(self):
        self.playbooks = {
            'data_breach': DataBreachPlaybook(),
            'malware': MalwarePlaybook(),
            'ddos': DDoSPlaybook(),
            'insider_threat': InsiderThreatPlaybook()
        }
    
    async def trigger(self, incident):
        # 1. Contain
        await self.contain_incident(incident)
        
        # 2. Assess
        impact = await self.assess_impact(incident)
        
        # 3. Notify
        await self.notify_stakeholders(incident, impact)
        
        # 4. Eradicate
        await self.eradicate_threat(incident)
        
        # 5. Recover
        await self.recover_systems(incident)
        
        # 6. Document
        await self.document_incident(incident)
```

## Compliance and Auditing

### Audit Logging

```python
class AuditLogger:
    def __init__(self):
        # Immutable audit log
        self.audit_store = ImmutableLogStore()
        
        # Required fields for compliance
        self.required_fields = [
            'timestamp',
            'user_id',
            'action',
            'resource',
            'result',
            'ip_address',
            'session_id'
        ]
    
    def log_event(self, event):
        # Ensure all required fields
        for field in self.required_fields:
            if field not in event:
                raise ValueError(f"Missing required field: {field}")
        
        # Add integrity check
        event['hash'] = self._calculate_hash(event)
        event['previous_hash'] = self.audit_store.get_last_hash()
        
        # Sign event
        event['signature'] = self._sign_event(event)
        
        # Store in immutable log
        self.audit_store.append(event)
        
        # Real-time compliance checking
        self.compliance_checker.check(event)
```

### Compliance Automation

```python
class ComplianceAutomation:
    def __init__(self):
        self.frameworks = {
            'gdpr': GDPRCompliance(),
            'hipaa': HIPAACompliance(),
            'pci_dss': PCIDSSCompliance(),
            'sox': SOXCompliance()
        }
    
    def run_compliance_checks(self):
        results = {}
        
        for framework_name, framework in self.frameworks.items():
            # Run automated checks
            checks = framework.run_checks()
            
            # Generate evidence
            evidence = framework.collect_evidence()
            
            # Create report
            report = framework.generate_report(checks, evidence)
            
            results[framework_name] = {
                'passed': all(check.passed for check in checks),
                'report': report,
                'evidence': evidence
            }
        
        return results
```

## Security Checklist

### Pre-Deployment

- [ ] All dependencies scanned for vulnerabilities
- [ ] Security headers configured
- [ ] TLS/SSL properly configured
- [ ] Authentication and authorization implemented
- [ ] Input validation in place
- [ ] Secrets management configured
- [ ] Logging and monitoring enabled
- [ ] Network segmentation implemented
- [ ] Container security policies applied
- [ ] Incident response plan documented

### Post-Deployment

- [ ] Regular security scans scheduled
- [ ] Penetration testing performed
- [ ] Security patches applied
- [ ] Access reviews conducted
- [ ] Audit logs reviewed
- [ ] Compliance checks passed
- [ ] Incident response drills conducted
- [ ] Security training completed
- [ ] Third-party assessments done
- [ ] Security metrics tracked

## Security Tools Integration

```yaml
# Security tool configuration
security_tools:
  vulnerability_scanning:
    - tool: trivy
      schedule: "0 */6 * * *"  # Every 6 hours
    - tool: snyk
      schedule: "0 2 * * *"    # Daily at 2 AM
  
  static_analysis:
    - tool: bandit          # Python
    - tool: semgrep         # Multi-language
    - tool: sonarqube       # Comprehensive
  
  runtime_protection:
    - tool: falco           # Container runtime
    - tool: osquery         # Host monitoring
  
  compliance:
    - tool: open-scap       # Security compliance
    - tool: inspec          # Infrastructure testing
```

## Best Practices

1. **Security by Design**
   - Consider security from the start
   - Threat model your architecture
   - Regular security reviews

2. **Continuous Security**
   - Automate security checks
   - Integrate into CI/CD pipeline
   - Regular vulnerability scanning

3. **Incident Preparedness**
   - Have response plan ready
   - Regular drills and training
   - Clear communication channels

4. **Regular Updates**
   - Patch management process
   - Dependency updates
   - Security configuration reviews

5. **Security Culture**
   - Security training for all team members
   - Security champions program
   - Bug bounty program

Remember: Security is not a one-time effort but a continuous process. Stay informed about emerging threats and adapt your security posture accordingly.