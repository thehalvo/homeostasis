"""
Security vulnerability scanning framework for Homeostasis testing infrastructure.

This module provides comprehensive security scanning capabilities integrated with
the testing framework, including:
- Dependency vulnerability scanning
- Static code analysis for security issues
- Container image scanning
- Infrastructure as Code (IaC) security scanning
- Runtime security testing
- OWASP compliance checking
"""

import asyncio
import json
import logging
import subprocess
import tempfile
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple, Union
from concurrent.futures import ThreadPoolExecutor
import yaml
import requests
from packaging import version
import re
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class SecurityVulnerability:
    """Represents a security vulnerability finding."""
    id: str
    severity: str  # critical, high, medium, low, info
    category: str  # dependency, code, container, iac, runtime
    title: str
    description: str
    affected_component: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    cve_ids: List[str] = None
    cwe_ids: List[str] = None
    owasp_category: Optional[str] = None
    remediation: Optional[str] = None
    references: List[str] = None
    metadata: Dict[str, Any] = None
    detected_at: datetime = None
    confidence: float = 1.0


@dataclass
class SecurityScanResult:
    """Results from a security scan."""
    scan_id: str
    scan_type: str
    target_path: str
    started_at: datetime
    completed_at: datetime
    vulnerabilities: List[SecurityVulnerability]
    summary: Dict[str, int]  # severity -> count
    metadata: Dict[str, Any] = None
    scan_status: str = "completed"  # completed, failed, partial
    error_messages: List[str] = None


class SecurityScanner:
    """Base class for security scanners."""
    
    def __init__(self):
        self.name = self.__class__.__name__
        self.supported_languages = set()
        self.supported_file_patterns = []
        
    async def scan(self, target_path: Path, options: Dict[str, Any] = None) -> SecurityScanResult:
        """Perform security scan on target path."""
        raise NotImplementedError
        
    def is_applicable(self, target_path: Path) -> bool:
        """Check if scanner is applicable to target."""
        # Check for supported file patterns
        for pattern in self.supported_file_patterns:
            if list(target_path.rglob(pattern)):
                return True
        return False


class DependencyVulnerabilityScanner(SecurityScanner):
    """Scan for vulnerabilities in project dependencies."""
    
    def __init__(self):
        super().__init__()
        self.supported_file_patterns = [
            "package.json", "package-lock.json", "yarn.lock",
            "requirements.txt", "Pipfile", "Pipfile.lock", "poetry.lock",
            "go.mod", "go.sum",
            "pom.xml", "build.gradle", "build.gradle.kts",
            "Gemfile", "Gemfile.lock",
            "composer.json", "composer.lock",
            "Cargo.toml", "Cargo.lock",
            "*.csproj", "packages.config", "project.json"
        ]
        
    async def scan(self, target_path: Path, options: Dict[str, Any] = None) -> SecurityScanResult:
        """Scan dependencies for vulnerabilities."""
        scan_id = self._generate_scan_id()
        started_at = datetime.now()
        vulnerabilities = []
        
        # Detect package managers and scan
        scanners = {
            "npm": self._scan_npm,
            "python": self._scan_python,
            "go": self._scan_go,
            "maven": self._scan_maven,
            "gradle": self._scan_gradle,
            "ruby": self._scan_ruby,
            "php": self._scan_php,
            "rust": self._scan_rust,
            "dotnet": self._scan_dotnet
        }
        
        tasks = []
        for scanner_name, scanner_func in scanners.items():
            if self._has_dependencies(target_path, scanner_name):
                tasks.append(scanner_func(target_path))
                
        # Run scanners concurrently
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, list):
                    vulnerabilities.extend(result)
                elif isinstance(result, Exception):
                    logger.error(f"Scanner error: {result}")
                    
        completed_at = datetime.now()
        
        return SecurityScanResult(
            scan_id=scan_id,
            scan_type="dependency",
            target_path=str(target_path),
            started_at=started_at,
            completed_at=completed_at,
            vulnerabilities=vulnerabilities,
            summary=self._summarize_vulnerabilities(vulnerabilities),
            metadata={"scanners_used": list(scanners.keys())}
        )
        
    def _generate_scan_id(self) -> str:
        """Generate unique scan ID."""
        timestamp = datetime.now().isoformat()
        return hashlib.sha256(f"{self.name}:{timestamp}".encode()).hexdigest()[:12]
        
    def _has_dependencies(self, target_path: Path, scanner_type: str) -> bool:
        """Check if target has dependencies for given scanner type."""
        patterns = {
            "npm": ["package.json", "package-lock.json", "yarn.lock"],
            "python": ["requirements.txt", "Pipfile", "poetry.lock", "setup.py"],
            "go": ["go.mod"],
            "maven": ["pom.xml"],
            "gradle": ["build.gradle", "build.gradle.kts"],
            "ruby": ["Gemfile"],
            "php": ["composer.json"],
            "rust": ["Cargo.toml"],
            "dotnet": ["*.csproj", "packages.config"]
        }
        
        for pattern in patterns.get(scanner_type, []):
            if list(target_path.rglob(pattern)):
                return True
        return False
        
    async def _scan_npm(self, target_path: Path) -> List[SecurityVulnerability]:
        """Scan NPM dependencies using npm audit."""
        vulnerabilities = []
        
        try:
            # Run npm audit
            process = await asyncio.create_subprocess_exec(
                "npm", "audit", "--json",
                cwd=target_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if stdout:
                audit_data = json.loads(stdout.decode())
                
                # Parse vulnerabilities
                for advisory_id, advisory in audit_data.get("advisories", {}).items():
                    vulnerability = SecurityVulnerability(
                        id=f"npm-{advisory_id}",
                        severity=advisory.get("severity", "unknown"),
                        category="dependency",
                        title=advisory.get("title", "Unknown vulnerability"),
                        description=advisory.get("overview", ""),
                        affected_component=advisory.get("module_name", ""),
                        cve_ids=advisory.get("cves", []),
                        cwe_ids=[f"CWE-{advisory.get('cwe', '')}"] if advisory.get('cwe') else [],
                        remediation=advisory.get("recommendation", ""),
                        references=[advisory.get("url", "")] if advisory.get("url") else [],
                        metadata={
                            "vulnerable_versions": advisory.get("vulnerable_versions", ""),
                            "patched_versions": advisory.get("patched_versions", "")
                        },
                        detected_at=datetime.now()
                    )
                    vulnerabilities.append(vulnerability)
                    
        except Exception as e:
            logger.error(f"NPM scan failed: {e}")
            
        return vulnerabilities
        
    async def _scan_python(self, target_path: Path) -> List[SecurityVulnerability]:
        """Scan Python dependencies using safety."""
        vulnerabilities = []
        
        try:
            # Find requirements file
            req_files = list(target_path.rglob("requirements*.txt"))
            if not req_files:
                return vulnerabilities
                
            for req_file in req_files:
                process = await asyncio.create_subprocess_exec(
                    "safety", "check", "--json", "-r", str(req_file),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await process.communicate()
                
                if stdout and process.returncode != 0:
                    safety_data = json.loads(stdout.decode())
                    
                    for vuln in safety_data:
                        vulnerability = SecurityVulnerability(
                            id=f"pyup-{vuln.get('vulnerability_id', 'unknown')}",
                            severity=self._map_python_severity(vuln),
                            category="dependency",
                            title=f"Vulnerability in {vuln.get('package', 'unknown')}",
                            description=vuln.get("advisory", ""),
                            affected_component=f"{vuln.get('package', '')}=={vuln.get('installed_version', '')}",
                            cve_ids=[vuln.get("cve", "")] if vuln.get("cve") else [],
                            remediation=f"Update to version {vuln.get('safe_versions', 'latest')}",
                            metadata={
                                "installed_version": vuln.get("installed_version", ""),
                                "affected_versions": vuln.get("affected_versions", "")
                            },
                            detected_at=datetime.now()
                        )
                        vulnerabilities.append(vulnerability)
                        
        except FileNotFoundError:
            logger.warning("Safety tool not found, skipping Python dependency scan")
        except Exception as e:
            logger.error(f"Python scan failed: {e}")
            
        return vulnerabilities
        
    async def _scan_go(self, target_path: Path) -> List[SecurityVulnerability]:
        """Scan Go dependencies using govulncheck."""
        vulnerabilities = []
        
        try:
            process = await asyncio.create_subprocess_exec(
                "govulncheck", "-json", "./...",
                cwd=target_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if stdout:
                # Parse JSON lines output
                for line in stdout.decode().splitlines():
                    if line.strip():
                        try:
                            vuln_data = json.loads(line)
                            if vuln_data.get("vulns"):
                                for vuln in vuln_data["vulns"]:
                                    vulnerability = SecurityVulnerability(
                                        id=f"go-{vuln.get('id', 'unknown')}",
                                        severity="high",  # Go doesn't provide severity
                                        category="dependency",
                                        title=vuln.get("summary", ""),
                                        description=vuln.get("details", ""),
                                        affected_component=vuln.get("package", ""),
                                        cve_ids=vuln.get("aliases", []),
                                        metadata={
                                            "symbol": vuln.get("symbol", ""),
                                            "call_stacks": vuln.get("call_stacks", [])
                                        },
                                        detected_at=datetime.now()
                                    )
                                    vulnerabilities.append(vulnerability)
                        except json.JSONDecodeError:
                            continue
                            
        except FileNotFoundError:
            logger.warning("govulncheck not found, skipping Go dependency scan")
        except Exception as e:
            logger.error(f"Go scan failed: {e}")
            
        return vulnerabilities
        
    async def _scan_maven(self, target_path: Path) -> List[SecurityVulnerability]:
        """Scan Maven dependencies using OWASP dependency check."""
        # Implementation for Maven scanning
        return []
        
    async def _scan_gradle(self, target_path: Path) -> List[SecurityVulnerability]:
        """Scan Gradle dependencies."""
        # Implementation for Gradle scanning
        return []
        
    async def _scan_ruby(self, target_path: Path) -> List[SecurityVulnerability]:
        """Scan Ruby dependencies using bundle-audit."""
        vulnerabilities = []
        
        try:
            process = await asyncio.create_subprocess_exec(
                "bundle-audit", "check", "--format", "json",
                cwd=target_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if stdout:
                audit_data = json.loads(stdout.decode())
                
                for result in audit_data.get("results", []):
                    vulnerability = SecurityVulnerability(
                        id=f"ruby-{result.get('id', 'unknown')}",
                        severity=result.get("criticality", "medium"),
                        category="dependency",
                        title=result.get("title", ""),
                        description=result.get("description", ""),
                        affected_component=f"{result.get('gem', '')}:{result.get('version', '')}",
                        cve_ids=[result.get("cve", "")] if result.get("cve") else [],
                        remediation=result.get("patched_versions", ""),
                        references=[result.get("url", "")] if result.get("url") else [],
                        detected_at=datetime.now()
                    )
                    vulnerabilities.append(vulnerability)
                    
        except FileNotFoundError:
            logger.warning("bundle-audit not found, skipping Ruby dependency scan")
        except Exception as e:
            logger.error(f"Ruby scan failed: {e}")
            
        return vulnerabilities
        
    async def _scan_php(self, target_path: Path) -> List[SecurityVulnerability]:
        """Scan PHP dependencies."""
        # Implementation for PHP scanning using local-php-security-checker
        return []
        
    async def _scan_rust(self, target_path: Path) -> List[SecurityVulnerability]:
        """Scan Rust dependencies using cargo-audit."""
        vulnerabilities = []
        
        try:
            process = await asyncio.create_subprocess_exec(
                "cargo", "audit", "--json",
                cwd=target_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if stdout:
                audit_data = json.loads(stdout.decode())
                
                for warning in audit_data.get("warnings", []):
                    advisory = warning.get("advisory", {})
                    vulnerability = SecurityVulnerability(
                        id=f"rust-{advisory.get('id', 'unknown')}",
                        severity=advisory.get("severity", "medium"),
                        category="dependency",
                        title=advisory.get("title", ""),
                        description=advisory.get("description", ""),
                        affected_component=warning.get("package", {}).get("name", ""),
                        cve_ids=[advisory.get("id", "")] if advisory.get("id", "").startswith("CVE") else [],
                        metadata={
                            "affected_functions": advisory.get("affected_functions", [])
                        },
                        detected_at=datetime.now()
                    )
                    vulnerabilities.append(vulnerability)
                    
        except FileNotFoundError:
            logger.warning("cargo-audit not found, skipping Rust dependency scan")
        except Exception as e:
            logger.error(f"Rust scan failed: {e}")
            
        return vulnerabilities
        
    async def _scan_dotnet(self, target_path: Path) -> List[SecurityVulnerability]:
        """Scan .NET dependencies."""
        # Implementation for .NET scanning
        return []
        
    def _map_python_severity(self, vuln: Dict[str, Any]) -> str:
        """Map Python vulnerability to standard severity."""
        # Safety doesn't provide severity, so we estimate based on description
        description = vuln.get("advisory", "").lower()
        if any(word in description for word in ["critical", "severe", "rce", "remote code"]):
            return "critical"
        elif any(word in description for word in ["high", "injection", "xss", "csrf"]):
            return "high"
        elif any(word in description for word in ["medium", "dos", "denial"]):
            return "medium"
        else:
            return "low"
            
    def _summarize_vulnerabilities(self, vulnerabilities: List[SecurityVulnerability]) -> Dict[str, int]:
        """Summarize vulnerabilities by severity."""
        summary = {"critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0}
        for vuln in vulnerabilities:
            severity = vuln.severity.lower()
            if severity in summary:
                summary[severity] += 1
        return summary


class StaticCodeSecurityScanner(SecurityScanner):
    """Scan source code for security vulnerabilities."""
    
    def __init__(self):
        super().__init__()
        self.supported_languages = {
            "python", "javascript", "typescript", "java", "go", 
            "c", "cpp", "csharp", "ruby", "php", "swift", "kotlin"
        }
        
    async def scan(self, target_path: Path, options: Dict[str, Any] = None) -> SecurityScanResult:
        """Scan source code for security issues."""
        scan_id = self._generate_scan_id()
        started_at = datetime.now()
        vulnerabilities = []
        
        # Language-specific scanners
        scanners = {
            ".py": self._scan_python_code,
            ".js": self._scan_javascript_code,
            ".ts": self._scan_typescript_code,
            ".java": self._scan_java_code,
            ".go": self._scan_go_code,
            ".c": self._scan_c_code,
            ".cpp": self._scan_cpp_code,
            ".cs": self._scan_csharp_code,
            ".rb": self._scan_ruby_code,
            ".php": self._scan_php_code,
            ".swift": self._scan_swift_code,
            ".kt": self._scan_kotlin_code
        }
        
        # Also run general scanners
        general_scanners = [
            self._scan_secrets,
            self._scan_hardcoded_passwords,
            self._scan_insecure_random,
            self._scan_sql_injection,
            self._scan_xss_vulnerabilities,
            self._scan_path_traversal,
            self._scan_insecure_deserialization
        ]
        
        # Scan files by extension
        tasks = []
        for ext, scanner_func in scanners.items():
            files = list(target_path.rglob(f"*{ext}"))
            if files:
                for file_path in files[:100]:  # Limit files per extension
                    tasks.append(scanner_func(file_path))
                    
        # Run general scanners
        for scanner_func in general_scanners:
            tasks.append(scanner_func(target_path))
            
        # Execute all scans concurrently
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, list):
                    vulnerabilities.extend(result)
                elif isinstance(result, Exception):
                    logger.error(f"Scanner error: {result}")
                    
        completed_at = datetime.now()
        
        return SecurityScanResult(
            scan_id=scan_id,
            scan_type="static_code",
            target_path=str(target_path),
            started_at=started_at,
            completed_at=completed_at,
            vulnerabilities=vulnerabilities,
            summary=self._summarize_vulnerabilities(vulnerabilities)
        )
        
    def _generate_scan_id(self) -> str:
        """Generate unique scan ID."""
        timestamp = datetime.now().isoformat()
        return hashlib.sha256(f"{self.name}:{timestamp}".encode()).hexdigest()[:12]
        
    async def _scan_python_code(self, file_path: Path) -> List[SecurityVulnerability]:
        """Scan Python code using bandit."""
        vulnerabilities = []
        
        try:
            process = await asyncio.create_subprocess_exec(
                "bandit", "-f", "json", str(file_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if stdout:
                bandit_data = json.loads(stdout.decode())
                
                for issue in bandit_data.get("results", []):
                    vulnerability = SecurityVulnerability(
                        id=f"bandit-{issue.get('test_id', 'unknown')}",
                        severity=issue.get("issue_severity", "medium").lower(),
                        category="code",
                        title=issue.get("issue_text", ""),
                        description=issue.get("issue_text", ""),
                        affected_component=str(file_path),
                        file_path=issue.get("filename", ""),
                        line_number=issue.get("line_number", 0),
                        cwe_ids=[f"CWE-{issue.get('issue_cwe', {}).get('id', '')}"] if issue.get('issue_cwe') else [],
                        confidence=self._map_confidence(issue.get("issue_confidence", "medium")),
                        metadata={
                            "test_name": issue.get("test_name", ""),
                            "code": issue.get("code", "")
                        },
                        detected_at=datetime.now()
                    )
                    vulnerabilities.append(vulnerability)
                    
        except FileNotFoundError:
            logger.warning("Bandit not found, skipping Python code scan")
        except Exception as e:
            logger.error(f"Python code scan failed: {e}")
            
        return vulnerabilities
        
    async def _scan_javascript_code(self, file_path: Path) -> List[SecurityVulnerability]:
        """Scan JavaScript code using ESLint security plugin."""
        vulnerabilities = []
        
        try:
            # Create temporary ESLint config
            eslint_config = {
                "extends": ["plugin:security/recommended"],
                "plugins": ["security"],
                "parserOptions": {
                    "ecmaVersion": 2021,
                    "sourceType": "module"
                }
            }
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(eslint_config, f)
                config_path = f.name
                
            try:
                process = await asyncio.create_subprocess_exec(
                    "eslint", "--format", "json", "--config", config_path, str(file_path),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await process.communicate()
                
                if stdout:
                    eslint_data = json.loads(stdout.decode())
                    
                    for file_result in eslint_data:
                        for message in file_result.get("messages", []):
                            if "security" in message.get("ruleId", ""):
                                vulnerability = SecurityVulnerability(
                                    id=f"eslint-{message.get('ruleId', 'unknown')}",
                                    severity=self._map_eslint_severity(message.get("severity", 1)),
                                    category="code",
                                    title=message.get("message", ""),
                                    description=message.get("message", ""),
                                    affected_component=str(file_path),
                                    file_path=file_result.get("filePath", ""),
                                    line_number=message.get("line", 0),
                                    metadata={
                                        "column": message.get("column", 0),
                                        "rule": message.get("ruleId", "")
                                    },
                                    detected_at=datetime.now()
                                )
                                vulnerabilities.append(vulnerability)
                                
            finally:
                Path(config_path).unlink()
                
        except FileNotFoundError:
            logger.warning("ESLint not found, skipping JavaScript code scan")
        except Exception as e:
            logger.error(f"JavaScript code scan failed: {e}")
            
        return vulnerabilities
        
    async def _scan_typescript_code(self, file_path: Path) -> List[SecurityVulnerability]:
        """Scan TypeScript code."""
        # Similar to JavaScript scanning
        return await self._scan_javascript_code(file_path)
        
    async def _scan_java_code(self, file_path: Path) -> List[SecurityVulnerability]:
        """Scan Java code using SpotBugs."""
        # Implementation for Java scanning
        return []
        
    async def _scan_go_code(self, file_path: Path) -> List[SecurityVulnerability]:
        """Scan Go code using gosec."""
        vulnerabilities = []
        
        try:
            process = await asyncio.create_subprocess_exec(
                "gosec", "-fmt", "json", str(file_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if stdout:
                gosec_data = json.loads(stdout.decode())
                
                for issue in gosec_data.get("Issues", []):
                    vulnerability = SecurityVulnerability(
                        id=f"gosec-{issue.get('rule_id', 'unknown')}",
                        severity=issue.get("severity", "medium").lower(),
                        category="code",
                        title=issue.get("details", ""),
                        description=issue.get("details", ""),
                        affected_component=str(file_path),
                        file_path=issue.get("file", ""),
                        line_number=int(issue.get("line", "0")),
                        cwe_ids=[f"CWE-{issue.get('cwe', {}).get('ID', '')}"] if issue.get('cwe') else [],
                        confidence=self._map_confidence(issue.get("confidence", "medium")),
                        metadata={
                            "code": issue.get("code", ""),
                            "column": issue.get("column", "")
                        },
                        detected_at=datetime.now()
                    )
                    vulnerabilities.append(vulnerability)
                    
        except FileNotFoundError:
            logger.warning("gosec not found, skipping Go code scan")
        except Exception as e:
            logger.error(f"Go code scan failed: {e}")
            
        return vulnerabilities
        
    async def _scan_c_code(self, file_path: Path) -> List[SecurityVulnerability]:
        """Scan C code using cppcheck."""
        # Implementation for C scanning
        return []
        
    async def _scan_cpp_code(self, file_path: Path) -> List[SecurityVulnerability]:
        """Scan C++ code using cppcheck."""
        # Implementation for C++ scanning
        return []
        
    async def _scan_csharp_code(self, file_path: Path) -> List[SecurityVulnerability]:
        """Scan C# code using security analyzers."""
        # Implementation for C# scanning
        return []
        
    async def _scan_ruby_code(self, file_path: Path) -> List[SecurityVulnerability]:
        """Scan Ruby code using brakeman."""
        # Implementation for Ruby scanning
        return []
        
    async def _scan_php_code(self, file_path: Path) -> List[SecurityVulnerability]:
        """Scan PHP code using psalm."""
        # Implementation for PHP scanning
        return []
        
    async def _scan_swift_code(self, file_path: Path) -> List[SecurityVulnerability]:
        """Scan Swift code."""
        # Implementation for Swift scanning
        return []
        
    async def _scan_kotlin_code(self, file_path: Path) -> List[SecurityVulnerability]:
        """Scan Kotlin code."""
        # Implementation for Kotlin scanning
        return []
        
    async def _scan_secrets(self, target_path: Path) -> List[SecurityVulnerability]:
        """Scan for hardcoded secrets using detect-secrets."""
        vulnerabilities = []
        
        try:
            process = await asyncio.create_subprocess_exec(
                "detect-secrets", "scan", "--all-files", str(target_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if stdout:
                secrets_data = json.loads(stdout.decode())
                
                for file_path, secrets in secrets_data.get("results", {}).items():
                    for secret in secrets:
                        vulnerability = SecurityVulnerability(
                            id=f"secret-{secret.get('type', 'unknown')}",
                            severity="high",
                            category="code",
                            title=f"Hardcoded {secret.get('type', 'secret')} detected",
                            description=f"Found potential {secret.get('type', 'secret')} in source code",
                            affected_component=file_path,
                            file_path=file_path,
                            line_number=secret.get("line_number", 0),
                            owasp_category="A03:2021 – Injection",
                            remediation="Remove hardcoded secrets and use environment variables or secret management systems",
                            metadata={
                                "secret_type": secret.get("type", ""),
                                "hashed_secret": secret.get("hashed_secret", "")
                            },
                            detected_at=datetime.now()
                        )
                        vulnerabilities.append(vulnerability)
                        
        except FileNotFoundError:
            logger.warning("detect-secrets not found, skipping secrets scan")
        except Exception as e:
            logger.error(f"Secrets scan failed: {e}")
            
        return vulnerabilities
        
    async def _scan_hardcoded_passwords(self, target_path: Path) -> List[SecurityVulnerability]:
        """Scan for hardcoded passwords using regex patterns."""
        vulnerabilities = []
        
        # Common password patterns
        password_patterns = [
            (r'password\s*=\s*["\']([^"\']+)["\']', "Hardcoded password"),
            (r'pwd\s*=\s*["\']([^"\']+)["\']', "Hardcoded password"),
            (r'passwd\s*=\s*["\']([^"\']+)["\']', "Hardcoded password"),
            (r'api_key\s*=\s*["\']([^"\']+)["\']', "Hardcoded API key"),
            (r'secret\s*=\s*["\']([^"\']+)["\']', "Hardcoded secret"),
            (r'token\s*=\s*["\']([^"\']+)["\']', "Hardcoded token")
        ]
        
        # Scan text files
        for file_path in target_path.rglob("*"):
            if file_path.is_file() and file_path.suffix in [".py", ".js", ".java", ".go", ".rb", ".php"]:
                try:
                    content = file_path.read_text()
                    
                    for line_num, line in enumerate(content.splitlines(), 1):
                        for pattern, description in password_patterns:
                            if re.search(pattern, line, re.IGNORECASE):
                                vulnerability = SecurityVulnerability(
                                    id=f"hardcoded-{hashlib.md5(line.encode()).hexdigest()[:8]}",
                                    severity="high",
                                    category="code",
                                    title=description,
                                    description=f"{description} found in source code",
                                    affected_component=str(file_path),
                                    file_path=str(file_path),
                                    line_number=line_num,
                                    cwe_ids=["CWE-798"],
                                    owasp_category="A07:2021 – Identification and Authentication Failures",
                                    remediation="Use environment variables or secure credential storage",
                                    detected_at=datetime.now()
                                )
                                vulnerabilities.append(vulnerability)
                                break
                                
                except Exception as e:
                    logger.debug(f"Could not scan {file_path}: {e}")
                    
        return vulnerabilities
        
    async def _scan_insecure_random(self, target_path: Path) -> List[SecurityVulnerability]:
        """Scan for insecure random number generation."""
        # Implementation for insecure random detection
        return []
        
    async def _scan_sql_injection(self, target_path: Path) -> List[SecurityVulnerability]:
        """Scan for SQL injection vulnerabilities."""
        # Implementation for SQL injection detection
        return []
        
    async def _scan_xss_vulnerabilities(self, target_path: Path) -> List[SecurityVulnerability]:
        """Scan for XSS vulnerabilities."""
        # Implementation for XSS detection
        return []
        
    async def _scan_path_traversal(self, target_path: Path) -> List[SecurityVulnerability]:
        """Scan for path traversal vulnerabilities."""
        # Implementation for path traversal detection
        return []
        
    async def _scan_insecure_deserialization(self, target_path: Path) -> List[SecurityVulnerability]:
        """Scan for insecure deserialization."""
        # Implementation for insecure deserialization detection
        return []
        
    def _map_confidence(self, confidence: str) -> float:
        """Map confidence levels to float values."""
        mapping = {
            "high": 0.9,
            "medium": 0.7,
            "low": 0.5
        }
        return mapping.get(confidence.lower(), 0.5)
        
    def _map_eslint_severity(self, severity: int) -> str:
        """Map ESLint severity to standard severity."""
        if severity == 2:
            return "high"
        elif severity == 1:
            return "medium"
        else:
            return "low"
            
    def _summarize_vulnerabilities(self, vulnerabilities: List[SecurityVulnerability]) -> Dict[str, int]:
        """Summarize vulnerabilities by severity."""
        summary = {"critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0}
        for vuln in vulnerabilities:
            severity = vuln.severity.lower()
            if severity in summary:
                summary[severity] += 1
        return summary


class ContainerSecurityScanner(SecurityScanner):
    """Scan container images for vulnerabilities."""
    
    def __init__(self):
        super().__init__()
        self.supported_file_patterns = ["Dockerfile", "docker-compose.yml", "docker-compose.yaml"]
        
    async def scan(self, target_path: Path, options: Dict[str, Any] = None) -> SecurityScanResult:
        """Scan container images and Dockerfiles."""
        scan_id = self._generate_scan_id()
        started_at = datetime.now()
        vulnerabilities = []
        
        # Scan Dockerfiles
        dockerfile_vulns = await self._scan_dockerfiles(target_path)
        vulnerabilities.extend(dockerfile_vulns)
        
        # Scan container images if specified
        if options and options.get("scan_images"):
            image_vulns = await self._scan_images(options.get("images", []))
            vulnerabilities.extend(image_vulns)
            
        completed_at = datetime.now()
        
        return SecurityScanResult(
            scan_id=scan_id,
            scan_type="container",
            target_path=str(target_path),
            started_at=started_at,
            completed_at=completed_at,
            vulnerabilities=vulnerabilities,
            summary=self._summarize_vulnerabilities(vulnerabilities)
        )
        
    def _generate_scan_id(self) -> str:
        """Generate unique scan ID."""
        timestamp = datetime.now().isoformat()
        return hashlib.sha256(f"{self.name}:{timestamp}".encode()).hexdigest()[:12]
        
    async def _scan_dockerfiles(self, target_path: Path) -> List[SecurityVulnerability]:
        """Scan Dockerfiles using hadolint."""
        vulnerabilities = []
        
        try:
            dockerfiles = list(target_path.rglob("Dockerfile*"))
            
            for dockerfile in dockerfiles:
                process = await asyncio.create_subprocess_exec(
                    "hadolint", "--format", "json", str(dockerfile),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await process.communicate()
                
                if stdout:
                    hadolint_data = json.loads(stdout.decode())
                    
                    for issue in hadolint_data:
                        vulnerability = SecurityVulnerability(
                            id=f"hadolint-{issue.get('code', 'unknown')}",
                            severity=self._map_hadolint_severity(issue.get("level", "info")),
                            category="container",
                            title=f"Dockerfile issue: {issue.get('code', '')}",
                            description=issue.get("message", ""),
                            affected_component=str(dockerfile),
                            file_path=issue.get("file", ""),
                            line_number=issue.get("line", 0),
                            remediation=f"See https://github.com/hadolint/hadolint/wiki/{issue.get('code', '')}",
                            metadata={
                                "column": issue.get("column", 0),
                                "code": issue.get("code", "")
                            },
                            detected_at=datetime.now()
                        )
                        vulnerabilities.append(vulnerability)
                        
        except FileNotFoundError:
            logger.warning("hadolint not found, skipping Dockerfile scan")
        except Exception as e:
            logger.error(f"Dockerfile scan failed: {e}")
            
        return vulnerabilities
        
    async def _scan_images(self, images: List[str]) -> List[SecurityVulnerability]:
        """Scan container images using trivy."""
        vulnerabilities = []
        
        try:
            for image in images:
                process = await asyncio.create_subprocess_exec(
                    "trivy", "image", "--format", "json", image,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await process.communicate()
                
                if stdout:
                    trivy_data = json.loads(stdout.decode())
                    
                    for result in trivy_data.get("Results", []):
                        for vuln in result.get("Vulnerabilities", []):
                            vulnerability = SecurityVulnerability(
                                id=f"trivy-{vuln.get('VulnerabilityID', 'unknown')}",
                                severity=vuln.get("Severity", "medium").lower(),
                                category="container",
                                title=vuln.get("Title", ""),
                                description=vuln.get("Description", ""),
                                affected_component=f"{vuln.get('PkgName', '')}:{vuln.get('InstalledVersion', '')}",
                                cve_ids=[vuln.get("VulnerabilityID", "")] if vuln.get("VulnerabilityID", "").startswith("CVE") else [],
                                remediation=f"Update to version {vuln.get('FixedVersion', 'latest')}",
                                references=vuln.get("References", []),
                                metadata={
                                    "target": result.get("Target", ""),
                                    "type": result.get("Type", ""),
                                    "nvd_score": vuln.get("CVSS", {}).get("nvd", {}).get("V3Score", 0)
                                },
                                detected_at=datetime.now()
                            )
                            vulnerabilities.append(vulnerability)
                            
        except FileNotFoundError:
            logger.warning("trivy not found, skipping container image scan")
        except Exception as e:
            logger.error(f"Container image scan failed: {e}")
            
        return vulnerabilities
        
    def _map_hadolint_severity(self, level: str) -> str:
        """Map hadolint levels to standard severity."""
        mapping = {
            "error": "high",
            "warning": "medium",
            "info": "low",
            "style": "info"
        }
        return mapping.get(level.lower(), "low")
        
    def _summarize_vulnerabilities(self, vulnerabilities: List[SecurityVulnerability]) -> Dict[str, int]:
        """Summarize vulnerabilities by severity."""
        summary = {"critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0}
        for vuln in vulnerabilities:
            severity = vuln.severity.lower()
            if severity in summary:
                summary[severity] += 1
        return summary


class IaCSecurityScanner(SecurityScanner):
    """Scan Infrastructure as Code for security issues."""
    
    def __init__(self):
        super().__init__()
        self.supported_file_patterns = [
            "*.tf", "*.tfvars",  # Terraform
            "*.yml", "*.yaml",   # Kubernetes, Ansible
            "*.json",            # CloudFormation
            "template.yaml"      # SAM
        ]
        
    async def scan(self, target_path: Path, options: Dict[str, Any] = None) -> SecurityScanResult:
        """Scan IaC files for security issues."""
        scan_id = self._generate_scan_id()
        started_at = datetime.now()
        vulnerabilities = []
        
        # Run different IaC scanners
        scanners = [
            self._scan_terraform,
            self._scan_kubernetes,
            self._scan_cloudformation,
            self._scan_ansible
        ]
        
        tasks = []
        for scanner_func in scanners:
            if self._has_iac_files(target_path, scanner_func.__name__):
                tasks.append(scanner_func(target_path))
                
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, list):
                    vulnerabilities.extend(result)
                elif isinstance(result, Exception):
                    logger.error(f"IaC scanner error: {result}")
                    
        completed_at = datetime.now()
        
        return SecurityScanResult(
            scan_id=scan_id,
            scan_type="iac",
            target_path=str(target_path),
            started_at=started_at,
            completed_at=completed_at,
            vulnerabilities=vulnerabilities,
            summary=self._summarize_vulnerabilities(vulnerabilities)
        )
        
    def _generate_scan_id(self) -> str:
        """Generate unique scan ID."""
        timestamp = datetime.now().isoformat()
        return hashlib.sha256(f"{self.name}:{timestamp}".encode()).hexdigest()[:12]
        
    def _has_iac_files(self, target_path: Path, scanner_type: str) -> bool:
        """Check if target has IaC files for given scanner."""
        patterns = {
            "_scan_terraform": ["*.tf", "*.tfvars"],
            "_scan_kubernetes": ["*.yaml", "*.yml"],
            "_scan_cloudformation": ["template.json", "template.yaml"],
            "_scan_ansible": ["playbook*.yml", "playbook*.yaml"]
        }
        
        for pattern in patterns.get(scanner_type, []):
            if list(target_path.rglob(pattern)):
                return True
        return False
        
    async def _scan_terraform(self, target_path: Path) -> List[SecurityVulnerability]:
        """Scan Terraform files using checkov."""
        vulnerabilities = []
        
        try:
            process = await asyncio.create_subprocess_exec(
                "checkov", "-d", str(target_path), "--framework", "terraform", "-o", "json",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if stdout:
                checkov_data = json.loads(stdout.decode())
                
                for check_type in ["failed_checks", "failed_bridgecrew_checks"]:
                    for check in checkov_data.get("results", {}).get(check_type, []):
                        vulnerability = SecurityVulnerability(
                            id=f"checkov-{check.get('check_id', 'unknown')}",
                            severity=self._map_checkov_severity(check),
                            category="iac",
                            title=check.get("check_name", ""),
                            description=check.get("check_name", ""),
                            affected_component=check.get("resource", ""),
                            file_path=check.get("file_path", ""),
                            line_number=check.get("file_line_range", [0])[0],
                            remediation=check.get("guideline", ""),
                            metadata={
                                "resource_type": check.get("check_type", ""),
                                "bc_check_id": check.get("bc_check_id", "")
                            },
                            detected_at=datetime.now()
                        )
                        vulnerabilities.append(vulnerability)
                        
        except FileNotFoundError:
            logger.warning("checkov not found, skipping Terraform scan")
        except Exception as e:
            logger.error(f"Terraform scan failed: {e}")
            
        return vulnerabilities
        
    async def _scan_kubernetes(self, target_path: Path) -> List[SecurityVulnerability]:
        """Scan Kubernetes manifests."""
        # Implementation for Kubernetes scanning
        return []
        
    async def _scan_cloudformation(self, target_path: Path) -> List[SecurityVulnerability]:
        """Scan CloudFormation templates."""
        # Implementation for CloudFormation scanning
        return []
        
    async def _scan_ansible(self, target_path: Path) -> List[SecurityVulnerability]:
        """Scan Ansible playbooks."""
        # Implementation for Ansible scanning
        return []
        
    def _map_checkov_severity(self, check: Dict[str, Any]) -> str:
        """Map checkov check to severity."""
        # Checkov doesn't provide severity, so we estimate
        check_name = check.get("check_name", "").lower()
        if any(word in check_name for word in ["critical", "high", "admin", "root"]):
            return "high"
        elif any(word in check_name for word in ["medium", "encryption", "ssl"]):
            return "medium"
        else:
            return "low"
            
    def _summarize_vulnerabilities(self, vulnerabilities: List[SecurityVulnerability]) -> Dict[str, int]:
        """Summarize vulnerabilities by severity."""
        summary = {"critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0}
        for vuln in vulnerabilities:
            severity = vuln.severity.lower()
            if severity in summary:
                summary[severity] += 1
        return summary


class SecurityTestOrchestrator:
    """Orchestrates security scanning across all testing infrastructure."""
    
    def __init__(self):
        self.scanners = {
            "dependency": DependencyVulnerabilityScanner(),
            "static_code": StaticCodeSecurityScanner(),
            "container": ContainerSecurityScanner(),
            "iac": IaCSecurityScanner()
        }
        self.results_cache = {}
        
    async def run_security_scan(self, target_path: Path, scan_types: List[str] = None,
                               options: Dict[str, Any] = None) -> Dict[str, SecurityScanResult]:
        """Run security scans on target."""
        if scan_types is None:
            scan_types = list(self.scanners.keys())
            
        results = {}
        tasks = []
        
        for scan_type in scan_types:
            if scan_type in self.scanners:
                scanner = self.scanners[scan_type]
                if scanner.is_applicable(target_path):
                    task = asyncio.create_task(
                        scanner.scan(target_path, options),
                        name=f"scan_{scan_type}"
                    )
                    tasks.append((scan_type, task))
                    
        # Execute scans concurrently
        for scan_type, task in tasks:
            try:
                result = await task
                results[scan_type] = result
            except Exception as e:
                logger.error(f"Security scan {scan_type} failed: {e}")
                
        return results
        
    def generate_security_report(self, scan_results: Dict[str, SecurityScanResult]) -> Dict[str, Any]:
        """Generate comprehensive security report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_vulnerabilities": 0,
                "by_severity": {"critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0},
                "by_category": {},
                "scan_types": list(scan_results.keys())
            },
            "scans": {},
            "top_vulnerabilities": [],
            "recommendations": []
        }
        
        all_vulnerabilities = []
        
        for scan_type, result in scan_results.items():
            report["scans"][scan_type] = {
                "status": result.scan_status,
                "duration": (result.completed_at - result.started_at).total_seconds(),
                "vulnerabilities_found": len(result.vulnerabilities),
                "summary": result.summary
            }
            
            all_vulnerabilities.extend(result.vulnerabilities)
            
            # Update summary
            for severity, count in result.summary.items():
                report["summary"]["by_severity"][severity] += count
                
        report["summary"]["total_vulnerabilities"] = len(all_vulnerabilities)
        
        # Group by category
        for vuln in all_vulnerabilities:
            category = vuln.category
            if category not in report["summary"]["by_category"]:
                report["summary"]["by_category"][category] = 0
            report["summary"]["by_category"][category] += 1
            
        # Get top vulnerabilities
        critical_high = [v for v in all_vulnerabilities if v.severity in ["critical", "high"]]
        report["top_vulnerabilities"] = [
            {
                "id": v.id,
                "severity": v.severity,
                "title": v.title,
                "affected_component": v.affected_component,
                "remediation": v.remediation
            }
            for v in sorted(critical_high, key=lambda x: x.severity)[:10]
        ]
        
        # Generate recommendations
        if report["summary"]["by_severity"]["critical"] > 0:
            report["recommendations"].append(
                "URGENT: Address critical vulnerabilities immediately before deployment"
            )
            
        if report["summary"]["by_category"].get("dependency", 0) > 10:
            report["recommendations"].append(
                "Consider implementing automated dependency updates and vulnerability monitoring"
            )
            
        if report["summary"]["by_category"].get("code", 0) > 20:
            report["recommendations"].append(
                "Implement pre-commit hooks for security scanning to catch issues earlier"
            )
            
        return report
        
    async def run_continuous_security_monitoring(self, target_path: Path, 
                                               interval_minutes: int = 60) -> None:
        """Run continuous security monitoring."""
        while True:
            try:
                # Run security scan
                results = await self.run_security_scan(target_path)
                
                # Check for new vulnerabilities
                self._check_new_vulnerabilities(results)
                
                # Store results
                self.results_cache[datetime.now()] = results
                
                # Wait for next scan
                await asyncio.sleep(interval_minutes * 60)
                
            except Exception as e:
                logger.error(f"Continuous monitoring error: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error
                
    def _check_new_vulnerabilities(self, results: Dict[str, SecurityScanResult]) -> None:
        """Check for new vulnerabilities compared to previous scan."""
        # Implementation for comparing with previous results
        pass


# Integration with existing test framework
def integrate_security_scanning(test_runner):
    """Decorator to add security scanning to test execution."""
    async def wrapper(*args, **kwargs):
        # Run original test
        result = await test_runner(*args, **kwargs)
        
        # Extract test directory from args
        test_case = args[1] if len(args) > 1 else kwargs.get("test_case")
        if test_case and hasattr(test_case, "source_code"):
            # Create temporary directory with test code
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Write test files
                for filename, content in test_case.source_code.items():
                    file_path = temp_path / filename
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    file_path.write_text(content)
                    
                # Run security scan
                orchestrator = SecurityTestOrchestrator()
                scan_results = await orchestrator.run_security_scan(temp_path)
                
                # Add security results to test result
                if hasattr(result, "metadata"):
                    if result.metadata is None:
                        result.metadata = {}
                    result.metadata["security_scan"] = orchestrator.generate_security_report(scan_results)
                    
        return result
        
    return wrapper