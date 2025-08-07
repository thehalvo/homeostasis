"""
Plugin Security Framework

This module provides comprehensive security features for the USHS plugin system,
including sandboxing, permission enforcement, code signing, and vulnerability scanning.
"""

import os
import json
import logging
import hashlib
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta
from enum import Enum
import gnupg
import requests
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.x509.oid import NameOID
import docker

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security levels for plugin execution."""
    UNRESTRICTED = "unrestricted"  # Development only
    BASIC = "basic"  # Basic sandboxing
    STANDARD = "standard"  # Default level
    STRICT = "strict"  # High security
    PARANOID = "paranoid"  # Maximum isolation


class PermissionType(Enum):
    """Types of permissions plugins can request."""
    FILESYSTEM_READ = "filesystem.read"
    FILESYSTEM_WRITE = "filesystem.write"
    FILESYSTEM_EXECUTE = "filesystem.execute"
    NETWORK_HTTP = "network.http"
    NETWORK_HTTPS = "network.https"
    NETWORK_ANY = "network.any"
    PROCESS_SPAWN = "process.spawn"
    PROCESS_EXEC = "process.exec"
    PROCESS_FORK = "process.fork"
    ENVIRONMENT_READ = "environment.read"
    ENVIRONMENT_WRITE = "environment.write"
    SYSTEM_INFO = "system.info"
    MEMORY_ALLOCATE = "memory.allocate"


class PluginSandbox:
    """Provides sandboxed execution environment for plugins."""
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.STANDARD):
        """
        Initialize plugin sandbox.
        
        Args:
            security_level: Security level for sandboxing
        """
        self.security_level = security_level
        self.docker_client = None
        
        if security_level in [SecurityLevel.STRICT, SecurityLevel.PARANOID]:
            try:
                self.docker_client = docker.from_env()
            except Exception as e:
                logger.warning(f"Docker not available for strict sandboxing: {e}")
    
    def create_sandbox(self, plugin_id: str, permissions: Dict[str, List[str]]) -> 'SandboxContext':
        """
        Create a sandboxed execution context.
        
        Args:
            plugin_id: Plugin identifier
            permissions: Granted permissions
            
        Returns:
            Sandbox context
        """
        if self.security_level == SecurityLevel.UNRESTRICTED:
            return UnrestrictedSandbox(plugin_id, permissions)
        elif self.security_level == SecurityLevel.BASIC:
            return BasicSandbox(plugin_id, permissions)
        elif self.security_level == SecurityLevel.STANDARD:
            return StandardSandbox(plugin_id, permissions)
        elif self.security_level in [SecurityLevel.STRICT, SecurityLevel.PARANOID]:
            if self.docker_client:
                return ContainerSandbox(plugin_id, permissions, self.docker_client, 
                                      paranoid=self.security_level == SecurityLevel.PARANOID)
            else:
                logger.warning("Falling back to standard sandbox")
                return StandardSandbox(plugin_id, permissions)


class SandboxContext:
    """Base class for sandbox contexts."""
    
    def __init__(self, plugin_id: str, permissions: Dict[str, List[str]]):
        """
        Initialize sandbox context.
        
        Args:
            plugin_id: Plugin identifier
            permissions: Granted permissions
        """
        self.plugin_id = plugin_id
        self.permissions = permissions
        self.active = False
    
    def __enter__(self):
        """Enter sandbox context."""
        self.setup()
        self.active = True
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit sandbox context."""
        self.teardown()
        self.active = False
    
    def setup(self):
        """Set up sandbox environment."""
        pass
    
    def teardown(self):
        """Tear down sandbox environment."""
        pass
    
    def check_permission(self, permission: PermissionType) -> bool:
        """
        Check if permission is granted.
        
        Args:
            permission: Permission to check
            
        Returns:
            True if permission is granted
        """
        perm_category, perm_action = permission.value.split('.')
        return perm_action in self.permissions.get(perm_category, [])
    
    def enforce_permission(self, permission: PermissionType):
        """
        Enforce permission requirement.
        
        Args:
            permission: Permission to enforce
            
        Raises:
            PermissionError: If permission is not granted
        """
        if not self.check_permission(permission):
            raise PermissionError(f"Plugin {self.plugin_id} does not have {permission.value} permission")


class UnrestrictedSandbox(SandboxContext):
    """Unrestricted sandbox for development."""
    pass


class BasicSandbox(SandboxContext):
    """Basic sandbox with permission checks."""
    
    def setup(self):
        """Set up basic restrictions."""
        # Store original functions
        self._original_open = open
        self._original_subprocess = subprocess.run
        
        # Wrap with permission checks
        import builtins
        builtins.open = self._wrapped_open
        subprocess.run = self._wrapped_subprocess
    
    def teardown(self):
        """Restore original functions."""
        import builtins
        builtins.open = self._original_open
        subprocess.run = self._original_subprocess
    
    def _wrapped_open(self, file, mode='r', *args, **kwargs):
        """Wrapped open function with permission checks."""
        if 'w' in mode or 'a' in mode:
            self.enforce_permission(PermissionType.FILESYSTEM_WRITE)
        else:
            self.enforce_permission(PermissionType.FILESYSTEM_READ)
        
        return self._original_open(file, mode, *args, **kwargs)
    
    def _wrapped_subprocess(self, *args, **kwargs):
        """Wrapped subprocess.run with permission checks."""
        self.enforce_permission(PermissionType.PROCESS_SPAWN)
        return self._original_subprocess(*args, **kwargs)


class StandardSandbox(BasicSandbox):
    """Standard sandbox with filesystem isolation."""
    
    def setup(self):
        """Set up standard sandbox."""
        super().setup()
        
        # Create isolated temp directory
        self.sandbox_dir = tempfile.mkdtemp(prefix=f"plugin_{self.plugin_id}_")
        self.original_cwd = os.getcwd()
        os.chdir(self.sandbox_dir)
        
        # Restrict environment variables
        self.original_env = os.environ.copy()
        if not self.check_permission(PermissionType.ENVIRONMENT_READ):
            # Clear sensitive environment variables
            for key in list(os.environ.keys()):
                if key.startswith(('AWS_', 'GOOGLE_', 'AZURE_', 'DATABASE_', 'API_')):
                    del os.environ[key]
    
    def teardown(self):
        """Tear down standard sandbox."""
        super().teardown()
        
        # Restore working directory
        os.chdir(self.original_cwd)
        
        # Clean up sandbox directory
        shutil.rmtree(self.sandbox_dir, ignore_errors=True)
        
        # Restore environment
        os.environ.clear()
        os.environ.update(self.original_env)


class ContainerSandbox(SandboxContext):
    """Container-based sandbox for strict isolation."""
    
    def __init__(self, plugin_id: str, permissions: Dict[str, List[str]], 
                 docker_client: docker.DockerClient, paranoid: bool = False):
        """
        Initialize container sandbox.
        
        Args:
            plugin_id: Plugin identifier
            permissions: Granted permissions
            docker_client: Docker client instance
            paranoid: Enable paranoid mode
        """
        super().__init__(plugin_id, permissions)
        self.docker_client = docker_client
        self.paranoid = paranoid
        self.container = None
    
    def setup(self):
        """Set up container sandbox."""
        # Build container configuration
        config = {
            'image': 'homeostasis/plugin-sandbox:latest',
            'name': f'plugin-{self.plugin_id}-{datetime.now().timestamp()}',
            'detach': True,
            'remove': True,
            'network_mode': 'none' if self.paranoid else 'bridge',
            'mem_limit': '512m',
            'cpu_quota': 50000,  # 50% of one CPU
            'read_only': not self.check_permission(PermissionType.FILESYSTEM_WRITE),
            'security_opt': ['no-new-privileges'],
            'cap_drop': ['ALL'],
            'cap_add': self._get_capabilities(),
            'environment': self._get_environment(),
            'volumes': self._get_volumes()
        }
        
        # Create container
        self.container = self.docker_client.containers.create(**config)
        self.container.start()
    
    def teardown(self):
        """Tear down container sandbox."""
        if self.container:
            try:
                self.container.stop(timeout=5)
            except Exception as e:
                logger.error(f"Failed to stop container: {e}")
                self.container.kill()
    
    def _get_capabilities(self) -> List[str]:
        """Get Linux capabilities based on permissions."""
        capabilities = []
        
        if self.check_permission(PermissionType.NETWORK_ANY):
            capabilities.append('NET_BIND_SERVICE')
        
        return capabilities
    
    def _get_environment(self) -> Dict[str, str]:
        """Get environment variables for container."""
        env = {
            'PLUGIN_ID': self.plugin_id,
            'SANDBOX_MODE': 'container'
        }
        
        if self.check_permission(PermissionType.ENVIRONMENT_READ):
            # Add allowed environment variables
            for key in ['PATH', 'HOME', 'USER']:
                if key in os.environ:
                    env[key] = os.environ[key]
        
        return env
    
    def _get_volumes(self) -> Dict[str, Dict[str, str]]:
        """Get volume mounts for container."""
        volumes = {}
        
        if self.check_permission(PermissionType.FILESYSTEM_READ):
            # Mount plugin directory as read-only
            volumes['/plugin'] = {'bind': '/plugin', 'mode': 'ro'}
        
        if self.check_permission(PermissionType.FILESYSTEM_WRITE):
            # Mount workspace directory
            volumes['/workspace'] = {'bind': '/workspace', 'mode': 'rw'}
        
        return volumes


class PluginSigner:
    """Handles plugin code signing and verification."""
    
    def __init__(self, key_dir: Optional[Path] = None):
        """
        Initialize plugin signer.
        
        Args:
            key_dir: Directory for storing keys
        """
        self.key_dir = key_dir or Path.home() / ".homeostasis/keys"
        self.key_dir.mkdir(parents=True, exist_ok=True)
        self.gpg = gnupg.GPG(gnupghome=str(self.key_dir))
    
    def generate_key_pair(self, plugin_author: str, email: str) -> str:
        """
        Generate a new key pair for plugin signing.
        
        Args:
            plugin_author: Author name
            email: Author email
            
        Returns:
            Key fingerprint
        """
        input_data = self.gpg.gen_key_input(
            name_real=plugin_author,
            name_email=email,
            key_type="RSA",
            key_length=4096,
            key_usage='sign',
            expire_date='2y'
        )
        
        key = self.gpg.gen_key(input_data)
        return str(key.fingerprint)
    
    def sign_plugin(self, plugin_path: Path, key_fingerprint: str, 
                   passphrase: Optional[str] = None) -> bool:
        """
        Sign a plugin package.
        
        Args:
            plugin_path: Path to plugin directory
            key_fingerprint: Signing key fingerprint
            passphrase: Key passphrase
            
        Returns:
            True if signed successfully
        """
        # Create plugin archive
        archive_path = plugin_path.parent / f"{plugin_path.name}.tar.gz"
        shutil.make_archive(
            str(archive_path.with_suffix('')), 
            'gztar', 
            plugin_path
        )
        
        # Sign the archive
        with open(archive_path, 'rb') as f:
            signed = self.gpg.sign_file(
                f,
                keyid=key_fingerprint,
                passphrase=passphrase,
                detach=True,
                output=str(archive_path) + '.sig'
            )
        
        # Clean up archive
        archive_path.unlink()
        
        return signed.status == 'signature created'
    
    def verify_plugin(self, plugin_path: Path) -> Tuple[bool, Optional[str]]:
        """
        Verify plugin signature.
        
        Args:
            plugin_path: Path to plugin directory
            
        Returns:
            Tuple of (is_valid, signer_fingerprint)
        """
        sig_file = plugin_path / 'manifest.json.sig'
        manifest_file = plugin_path / 'manifest.json'
        
        if not sig_file.exists():
            return False, None
        
        # Verify signature
        with open(manifest_file, 'rb') as f:
            verified = self.gpg.verify_file(f, str(sig_file))
        
        return verified.valid, verified.fingerprint if verified.valid else None
    
    def import_public_key(self, key_data: str) -> str:
        """
        Import a public key.
        
        Args:
            key_data: Public key data
            
        Returns:
            Key fingerprint
        """
        import_result = self.gpg.import_keys(key_data)
        if import_result.count > 0:
            return import_result.fingerprints[0]
        raise ValueError("Failed to import public key")
    
    def export_public_key(self, key_fingerprint: str) -> str:
        """
        Export a public key.
        
        Args:
            key_fingerprint: Key fingerprint
            
        Returns:
            Public key data
        """
        return self.gpg.export_keys(key_fingerprint)


class VulnerabilityScanner:
    """Scans plugins for known vulnerabilities."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize vulnerability scanner.
        
        Args:
            api_key: API key for vulnerability database
        """
        self.api_key = api_key
        self.cache_dir = Path.home() / ".homeostasis/vuln_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl = timedelta(hours=24)
    
    def scan_dependencies(self, plugin_path: Path) -> List[Dict[str, Any]]:
        """
        Scan plugin dependencies for vulnerabilities.
        
        Args:
            plugin_path: Path to plugin directory
            
        Returns:
            List of vulnerability findings
        """
        vulnerabilities = []
        
        # Check different dependency files
        dependency_scanners = {
            'package.json': self._scan_npm_dependencies,
            'requirements.txt': self._scan_python_dependencies,
            'go.mod': self._scan_go_dependencies,
            'Cargo.toml': self._scan_rust_dependencies
        }
        
        for filename, scanner in dependency_scanners.items():
            dep_file = plugin_path / filename
            if dep_file.exists():
                vulns = scanner(dep_file)
                vulnerabilities.extend(vulns)
        
        return vulnerabilities
    
    def _scan_npm_dependencies(self, package_file: Path) -> List[Dict[str, Any]]:
        """Scan NPM dependencies."""
        vulnerabilities = []
        
        try:
            # Run npm audit
            result = subprocess.run(
                ['npm', 'audit', '--json'],
                cwd=package_file.parent,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                audit_data = json.loads(result.stdout)
                for advisory in audit_data.get('advisories', {}).values():
                    vulnerabilities.append({
                        'type': 'npm',
                        'package': advisory['module_name'],
                        'severity': advisory['severity'],
                        'title': advisory['title'],
                        'cve': advisory.get('cves', []),
                        'recommendation': advisory['recommendation']
                    })
        
        except Exception as e:
            logger.error(f"Failed to scan NPM dependencies: {e}")
        
        return vulnerabilities
    
    def _scan_python_dependencies(self, requirements_file: Path) -> List[Dict[str, Any]]:
        """Scan Python dependencies."""
        vulnerabilities = []
        
        try:
            # Use safety or similar tool
            result = subprocess.run(
                ['safety', 'check', '--json', '-r', str(requirements_file)],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                safety_data = json.loads(result.stdout)
                for vuln in safety_data:
                    vulnerabilities.append({
                        'type': 'python',
                        'package': vuln['package'],
                        'severity': 'high',  # Safety doesn't provide severity
                        'title': vuln['advisory'],
                        'cve': vuln.get('cve', ''),
                        'recommendation': f"Upgrade to {vuln['safe_version']}"
                    })
        
        except Exception as e:
            logger.error(f"Failed to scan Python dependencies: {e}")
        
        return vulnerabilities
    
    def _scan_go_dependencies(self, go_mod_file: Path) -> List[Dict[str, Any]]:
        """Scan Go dependencies."""
        vulnerabilities = []
        
        try:
            # Use govulncheck
            result = subprocess.run(
                ['govulncheck', '-json', './...'],
                cwd=go_mod_file.parent,
                capture_output=True,
                text=True
            )
            
            for line in result.stdout.splitlines():
                if line.strip():
                    vuln_data = json.loads(line)
                    if vuln_data.get('vulns'):
                        for vuln in vuln_data['vulns']:
                            vulnerabilities.append({
                                'type': 'go',
                                'package': vuln['package'],
                                'severity': 'high',
                                'title': vuln['summary'],
                                'cve': vuln.get('aliases', []),
                                'recommendation': 'Update to latest version'
                            })
        
        except Exception as e:
            logger.error(f"Failed to scan Go dependencies: {e}")
        
        return vulnerabilities
    
    def _scan_rust_dependencies(self, cargo_file: Path) -> List[Dict[str, Any]]:
        """Scan Rust dependencies."""
        vulnerabilities = []
        
        try:
            # Use cargo-audit
            result = subprocess.run(
                ['cargo', 'audit', '--json'],
                cwd=cargo_file.parent,
                capture_output=True,
                text=True
            )
            
            audit_data = json.loads(result.stdout)
            for warning in audit_data.get('warnings', []):
                vulnerabilities.append({
                    'type': 'rust',
                    'package': warning['package']['name'],
                    'severity': warning['advisory']['severity'],
                    'title': warning['advisory']['title'],
                    'cve': warning['advisory'].get('id', ''),
                    'recommendation': warning['advisory']['description']
                })
        
        except Exception as e:
            logger.error(f"Failed to scan Rust dependencies: {e}")
        
        return vulnerabilities
    
    def scan_code(self, plugin_path: Path) -> List[Dict[str, Any]]:
        """
        Scan plugin code for security issues.
        
        Args:
            plugin_path: Path to plugin directory
            
        Returns:
            List of security findings
        """
        findings = []
        
        # Run static analysis tools
        analyzers = {
            '.py': self._analyze_python_code,
            '.js': self._analyze_javascript_code,
            '.ts': self._analyze_typescript_code,
            '.go': self._analyze_go_code
        }
        
        for file_path in plugin_path.rglob('*'):
            if file_path.is_file():
                suffix = file_path.suffix.lower()
                if suffix in analyzers:
                    file_findings = analyzers[suffix](file_path)
                    findings.extend(file_findings)
        
        return findings
    
    def _analyze_python_code(self, file_path: Path) -> List[Dict[str, Any]]:
        """Analyze Python code for security issues."""
        findings = []
        
        try:
            # Use bandit for security analysis
            result = subprocess.run(
                ['bandit', '-f', 'json', str(file_path)],
                capture_output=True,
                text=True
            )
            
            bandit_data = json.loads(result.stdout)
            for issue in bandit_data.get('results', []):
                findings.append({
                    'file': str(file_path),
                    'line': issue['line_number'],
                    'severity': issue['issue_severity'].lower(),
                    'confidence': issue['issue_confidence'].lower(),
                    'title': issue['issue_text'],
                    'cwe': issue.get('issue_cwe', {}).get('id', '')
                })
        
        except Exception as e:
            logger.error(f"Failed to analyze Python code: {e}")
        
        return findings
    
    def _analyze_javascript_code(self, file_path: Path) -> List[Dict[str, Any]]:
        """Analyze JavaScript code for security issues."""
        findings = []
        
        try:
            # Use ESLint with security plugin
            result = subprocess.run(
                ['eslint', '--format=json', '--plugin=security', str(file_path)],
                capture_output=True,
                text=True
            )
            
            eslint_data = json.loads(result.stdout)
            for file_result in eslint_data:
                for message in file_result.get('messages', []):
                    if 'security' in message.get('ruleId', ''):
                        findings.append({
                            'file': str(file_path),
                            'line': message['line'],
                            'severity': message['severity'],
                            'title': message['message'],
                            'rule': message['ruleId']
                        })
        
        except Exception as e:
            logger.error(f"Failed to analyze JavaScript code: {e}")
        
        return findings
    
    def _analyze_typescript_code(self, file_path: Path) -> List[Dict[str, Any]]:
        """Analyze TypeScript code for security issues."""
        # Similar to JavaScript analysis
        return self._analyze_javascript_code(file_path)
    
    def _analyze_go_code(self, file_path: Path) -> List[Dict[str, Any]]:
        """Analyze Go code for security issues."""
        findings = []
        
        try:
            # Use gosec
            result = subprocess.run(
                ['gosec', '-fmt=json', str(file_path)],
                capture_output=True,
                text=True
            )
            
            gosec_data = json.loads(result.stdout)
            for issue in gosec_data.get('issues', []):
                findings.append({
                    'file': issue['file'],
                    'line': issue['line'],
                    'severity': issue['severity'].lower(),
                    'confidence': issue['confidence'].lower(),
                    'title': issue['details'],
                    'cwe': issue.get('cwe', {}).get('id', '')
                })
        
        except Exception as e:
            logger.error(f"Failed to analyze Go code: {e}")
        
        return findings


class PluginSecurityManager:
    """Central manager for all plugin security features."""
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.STANDARD):
        """
        Initialize security manager.
        
        Args:
            security_level: Default security level
        """
        self.security_level = security_level
        self.sandbox = PluginSandbox(security_level)
        self.signer = PluginSigner()
        self.scanner = VulnerabilityScanner()
        self.permission_cache: Dict[str, Dict[str, List[str]]] = {}
    
    def validate_plugin_security(self, plugin_path: Path) -> Tuple[bool, List[str]]:
        """
        Perform comprehensive security validation.
        
        Args:
            plugin_path: Path to plugin directory
            
        Returns:
            Tuple of (is_secure, security_issues)
        """
        issues = []
        
        # Check signature
        is_signed, signer = self.signer.verify_plugin(plugin_path)
        if not is_signed:
            issues.append("Plugin is not signed or signature is invalid")
        
        # Scan for vulnerabilities
        dep_vulns = self.scanner.scan_dependencies(plugin_path)
        for vuln in dep_vulns:
            issues.append(
                f"{vuln['severity'].upper()} vulnerability in {vuln['package']}: {vuln['title']}"
            )
        
        # Scan code
        code_findings = self.scanner.scan_code(plugin_path)
        for finding in code_findings:
            if finding['severity'] in ['high', 'critical']:
                issues.append(
                    f"Security issue in {finding['file']}:{finding['line']}: {finding['title']}"
                )
        
        # Check permissions
        manifest_path = plugin_path / 'manifest.json'
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            permissions = manifest.get('permissions', {})
            dangerous_perms = self._check_dangerous_permissions(permissions)
            issues.extend(dangerous_perms)
        
        is_secure = len(issues) == 0
        return is_secure, issues
    
    def _check_dangerous_permissions(self, permissions: Dict[str, List[str]]) -> List[str]:
        """Check for dangerous permission combinations."""
        issues = []
        
        # Check for unrestricted network access
        if 'network' in permissions and 'any' in permissions['network']:
            if 'filesystem' in permissions and 'write' in permissions['filesystem']:
                issues.append("Plugin requests unrestricted network and filesystem write access")
        
        # Check for process spawning with network access
        if 'process' in permissions and any(p in permissions['process'] for p in ['spawn', 'exec']):
            if 'network' in permissions:
                issues.append("Plugin can spawn processes and access network")
        
        # Check for environment variable access
        if 'environment' in permissions and 'write' in permissions['environment']:
            issues.append("Plugin can modify environment variables")
        
        return issues
    
    def create_secure_context(self, plugin_id: str, requested_permissions: Dict[str, List[str]],
                            user_approved: bool = False) -> SandboxContext:
        """
        Create a secure execution context for a plugin.
        
        Args:
            plugin_id: Plugin identifier
            requested_permissions: Permissions requested by plugin
            user_approved: Whether user has approved permissions
            
        Returns:
            Sandbox context
        """
        if not user_approved and self.security_level != SecurityLevel.UNRESTRICTED:
            # Limit permissions to safe subset
            granted_permissions = {
                'filesystem': ['read'],
                'network': ['https']
            }
        else:
            granted_permissions = requested_permissions
        
        # Cache permissions
        self.permission_cache[plugin_id] = granted_permissions
        
        # Create sandbox
        return self.sandbox.create_sandbox(plugin_id, granted_permissions)
    
    def sign_plugin(self, plugin_path: Path, author_email: str,
                   passphrase: Optional[str] = None) -> bool:
        """
        Sign a plugin package.
        
        Args:
            plugin_path: Path to plugin directory
            author_email: Author's email for key selection
            passphrase: Key passphrase
            
        Returns:
            True if signed successfully
        """
        # Find or generate key
        keys = self.signer.gpg.list_keys(keys=[author_email])
        if not keys:
            logger.info(f"Generating new key for {author_email}")
            fingerprint = self.signer.generate_key_pair(
                plugin_path.name,
                author_email
            )
        else:
            fingerprint = keys[0]['fingerprint']
        
        # Sign plugin
        return self.signer.sign_plugin(plugin_path, fingerprint, passphrase)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create security manager
    security_manager = PluginSecurityManager(SecurityLevel.STANDARD)
    
    # Example plugin path
    plugin_path = Path("example_plugin")
    
    # Validate security
    is_secure, issues = security_manager.validate_plugin_security(plugin_path)
    if not is_secure:
        logger.warning(f"Security issues found: {issues}")
    
    # Create secure execution context
    with security_manager.create_secure_context(
        "example-plugin@1.0.0",
        {'filesystem': ['read'], 'network': ['https']},
        user_approved=True
    ) as sandbox:
        logger.info("Plugin executing in sandbox")
        
        # Plugin code would execute here
        try:
            # Test permission enforcement
            sandbox.enforce_permission(PermissionType.FILESYSTEM_READ)
            logger.info("Filesystem read permission granted")
            
            sandbox.enforce_permission(PermissionType.FILESYSTEM_WRITE)
        except PermissionError as e:
            logger.info(f"Permission denied as expected: {e}")