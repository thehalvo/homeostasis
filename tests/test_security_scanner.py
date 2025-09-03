"""
Test suite for the security vulnerability scanning framework.
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

from modules.testing.security_scanner import (
    SecurityVulnerability,
    SecurityScanResult,
    DependencyVulnerabilityScanner,
    StaticCodeSecurityScanner,
    ContainerSecurityScanner,
    IaCSecurityScanner,
    SecurityTestOrchestrator
)


class TestSecurityVulnerability:
    """Test SecurityVulnerability dataclass."""
    
    def test_vulnerability_creation(self):
        """Test creating a vulnerability instance."""
        vuln = SecurityVulnerability(
            id="test-001",
            severity="high",
            category="dependency",
            title="Test vulnerability",
            description="Test description",
            affected_component="test-package==1.0.0",
            cve_ids=["CVE-2021-12345"],
            remediation="Update to version 2.0.0"
        )
        
        assert vuln.id == "test-001"
        assert vuln.severity == "high"
        assert vuln.category == "dependency"
        assert vuln.cve_ids == ["CVE-2021-12345"]


class TestDependencyVulnerabilityScanner:
    """Test dependency vulnerability scanning."""
    
    @pytest.fixture
    def scanner(self):
        return DependencyVulnerabilityScanner()
    
    @pytest.fixture
    def temp_project(self):
        """Create a temporary project structure."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    def test_scanner_initialization(self, scanner):
        """Test scanner initialization."""
        assert scanner.name == "DependencyVulnerabilityScanner"
        assert "package.json" in scanner.supported_file_patterns
        assert "requirements.txt" in scanner.supported_file_patterns
        assert "go.mod" in scanner.supported_file_patterns
    
    def test_is_applicable(self, scanner, temp_project):
        """Test checking if scanner is applicable."""
        # Create package.json
        (temp_project / "package.json").write_text('{"name": "test"}')
        assert scanner.is_applicable(temp_project)
        
        # Create requirements.txt
        (temp_project / "requirements.txt").write_text("flask==2.0.0")
        assert scanner.is_applicable(temp_project)
    
    @pytest.mark.asyncio
    async def test_scan_npm_dependencies(self, scanner, temp_project):
        """Test scanning NPM dependencies."""
        # Create vulnerable package.json
        package_json = {
            "name": "test-app",
            "version": "1.0.0",
            "dependencies": {
                "lodash": "4.17.11"  # Known vulnerable version
            }
        }
        (temp_project / "package.json").write_text(json.dumps(package_json))
        
        # Mock npm audit output
        mock_audit_output = {
            "advisories": {
                "1523": {
                    "module_name": "lodash",
                    "severity": "high",
                    "title": "Prototype Pollution",
                    "cves": ["CVE-2019-10744"],
                    "vulnerable_versions": "<4.17.19",
                    "patched_versions": ">=4.17.19",
                    "recommendation": "Upgrade to version 4.17.19 or later"
                }
            }
        }
        
        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (
                json.dumps(mock_audit_output).encode(),
                b""
            )
            mock_subprocess.return_value = mock_process
            
            result = await scanner.scan(temp_project)
            
            assert result.scan_type == "dependency"
            assert len(result.vulnerabilities) > 0
            
            vuln = result.vulnerabilities[0]
            assert vuln.severity == "high"
            assert vuln.affected_component == "lodash"
            assert "CVE-2019-10744" in vuln.cve_ids
    
    @pytest.mark.asyncio
    async def test_scan_python_dependencies(self, scanner, temp_project):
        """Test scanning Python dependencies."""
        # Create requirements.txt with vulnerable package
        (temp_project / "requirements.txt").write_text("django==2.2.0")
        
        # Mock safety output
        mock_safety_output = [{
            "package": "django",
            "installed_version": "2.2.0",
            "vulnerability_id": "38624",
            "advisory": "Django before 2.2.24 allows...",
            "cve": "CVE-2021-33571",
            "safe_versions": "2.2.24"
        }]
        
        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (
                json.dumps(mock_safety_output).encode(),
                b""
            )
            mock_process.returncode = 1  # Safety returns 1 when vulnerabilities found
            mock_subprocess.return_value = mock_process
            
            vulnerabilities = await scanner._scan_python(temp_project)
            
            assert len(vulnerabilities) > 0
            vuln = vulnerabilities[0]
            assert "django" in vuln.affected_component
            assert vuln.cve_ids == ["CVE-2021-33571"]


class TestStaticCodeSecurityScanner:
    """Test static code security scanning."""
    
    @pytest.fixture
    def scanner(self):
        return StaticCodeSecurityScanner()
    
    @pytest.fixture
    def temp_code(self):
        """Create temporary code files."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.mark.asyncio
    async def test_scan_python_code(self, scanner, temp_code):
        """Test scanning Python code for vulnerabilities."""
        # Create Python file with security issues
        vulnerable_code = '''
import pickle
import os

def load_data(filename):
    # Insecure deserialization
    with open(filename, 'rb') as f:
        return pickle.load(f)

def run_command(user_input):
    # Command injection vulnerability
    os.system(f"echo {user_input}")

password = "hardcoded_password123"
api_key = "sk-1234567890abcdef"
'''
        (temp_code / "vulnerable.py").write_text(vulnerable_code)
        
        # Mock bandit output
        mock_bandit_output = {
            "results": [
                {
                    "test_id": "B301",
                    "issue_severity": "HIGH",
                    "issue_confidence": "HIGH",
                    "issue_text": "Pickle library used",
                    "filename": str(temp_code / "vulnerable.py"),
                    "line_number": 7,
                    "issue_cwe": {"id": 502}
                },
                {
                    "test_id": "B602",
                    "issue_severity": "HIGH",
                    "issue_confidence": "HIGH",
                    "issue_text": "Use of os.system detected",
                    "filename": str(temp_code / "vulnerable.py"),
                    "line_number": 12,
                    "issue_cwe": {"id": 78}
                }
            ]
        }
        
        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (
                json.dumps(mock_bandit_output).encode(),
                b""
            )
            mock_subprocess.return_value = mock_process
            
            vulnerabilities = await scanner._scan_python_code(temp_code / "vulnerable.py")
            
            assert len(vulnerabilities) >= 2
            assert any(v.cwe_ids == ["CWE-502"] for v in vulnerabilities)
            assert any(v.cwe_ids == ["CWE-78"] for v in vulnerabilities)
    
    @pytest.mark.asyncio
    async def test_scan_hardcoded_passwords(self, scanner, temp_code):
        """Test scanning for hardcoded passwords."""
        # Create file with hardcoded credentials
        code_with_secrets = '''
database_password = "supersecret123"
api_key = "sk-prod-1234567890"
AWS_SECRET_KEY = "aws_secret_key_here"
token = "ghp_1234567890abcdef"
'''
        (temp_code / "config.py").write_text(code_with_secrets)
        
        vulnerabilities = await scanner._scan_hardcoded_passwords(temp_code)
        
        assert len(vulnerabilities) >= 3
        for vuln in vulnerabilities:
            assert vuln.severity == "high"
            assert vuln.cwe_ids == ["CWE-798"]
            assert "A07:2021" in vuln.owasp_category


class TestContainerSecurityScanner:
    """Test container security scanning."""
    
    @pytest.fixture
    def scanner(self):
        return ContainerSecurityScanner()
    
    @pytest.fixture
    def temp_dockerfile(self):
        """Create temporary Dockerfile."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.mark.asyncio
    async def test_scan_dockerfile(self, scanner, temp_dockerfile):
        """Test scanning Dockerfile for security issues."""
        # Create Dockerfile with security issues
        dockerfile_content = '''
FROM ubuntu:latest
USER root
RUN apt-get update && apt-get install -y curl
RUN curl -L https://example.com/script.sh | sh
EXPOSE 22
ADD . /app
RUN chmod 777 /app
CMD ["python", "app.py"]
'''
        (temp_dockerfile / "Dockerfile").write_text(dockerfile_content)
        
        # Mock hadolint output
        mock_hadolint_output = [
            {
                "line": 2,
                "code": "DL3002",
                "level": "warning",
                "message": "Last USER should not be root"
            },
            {
                "line": 4,
                "code": "DL3015",
                "level": "info",
                "message": "Avoid additional packages by specifying --no-install-recommends"
            },
            {
                "line": 5,
                "code": "DL3001",
                "level": "info",
                "message": "Avoid RUN with sudo"
            }
        ]
        
        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (
                json.dumps(mock_hadolint_output).encode(),
                b""
            )
            mock_subprocess.return_value = mock_process
            
            vulnerabilities = await scanner._scan_dockerfiles(temp_dockerfile)
            
            assert len(vulnerabilities) >= 3
            assert any("root" in v.description for v in vulnerabilities)


class TestIaCSecurityScanner:
    """Test Infrastructure as Code security scanning."""
    
    @pytest.fixture
    def scanner(self):
        return IaCSecurityScanner()
    
    @pytest.fixture
    def temp_iac(self):
        """Create temporary IaC files."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.mark.asyncio
    async def test_scan_terraform(self, scanner, temp_iac):
        """Test scanning Terraform files."""
        # Create Terraform file with security issues
        terraform_content = '''
resource "aws_s3_bucket" "data" {
  bucket = "my-data-bucket"
}

resource "aws_security_group_rule" "allow_all" {
  type              = "ingress"
  from_port         = 0
  to_port           = 65535
  protocol          = "tcp"
  cidr_blocks       = ["0.0.0.0/0"]
  security_group_id = aws_security_group.main.id
}
'''
        (temp_iac / "main.tf").write_text(terraform_content)
        
        # Mock checkov output
        mock_checkov_output = {
            "results": {
                "failed_checks": [
                    {
                        "check_id": "CKV_AWS_18",
                        "check_name": "Ensure S3 bucket has encryption",
                        "resource": "aws_s3_bucket.data",
                        "file_path": "main.tf",
                        "file_line_range": [1, 3],
                        "guideline": "Enable encryption for S3 bucket"
                    },
                    {
                        "check_id": "CKV_AWS_24",
                        "check_name": "Ensure no security groups allow ingress from 0.0.0.0/0",
                        "resource": "aws_security_group_rule.allow_all",
                        "file_path": "main.tf",
                        "file_line_range": [5, 12]
                    }
                ]
            }
        }
        
        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (
                json.dumps(mock_checkov_output).encode(),
                b""
            )
            mock_subprocess.return_value = mock_process
            
            vulnerabilities = await scanner._scan_terraform(temp_iac)
            
            assert len(vulnerabilities) >= 2
            assert any("encryption" in v.title.lower() for v in vulnerabilities)
            assert any("0.0.0.0/0" in v.title for v in vulnerabilities)


class TestSecurityTestOrchestrator:
    """Test security test orchestrator."""
    
    @pytest.fixture
    def orchestrator(self):
        return SecurityTestOrchestrator()
    
    @pytest.fixture
    def temp_project(self):
        """Create temporary project."""
        temp_dir = tempfile.mkdtemp()
        project_path = Path(temp_dir)
        
        # Create sample project structure
        (project_path / "src").mkdir()
        (project_path / "src" / "app.py").write_text("print('Hello World')")
        (project_path / "requirements.txt").write_text("flask==2.0.0")
        (project_path / "Dockerfile").write_text("FROM python:3.9")
        (project_path / "terraform").mkdir()
        (project_path / "terraform" / "main.tf").write_text('provider "aws" {}')
        
        yield project_path
        shutil.rmtree(temp_dir)
    
    @pytest.mark.asyncio
    async def test_run_security_scan(self, orchestrator, temp_project):
        """Test running complete security scan."""
        # Mock scanner results
        mock_scan_result = SecurityScanResult(
            scan_id="test-123",
            scan_type="dependency",
            target_path=str(temp_project),
            started_at=datetime.now(),
            completed_at=datetime.now(),
            vulnerabilities=[
                SecurityVulnerability(
                    id="test-001",
                    severity="high",
                    category="dependency",
                    title="Test vulnerability",
                    description="Test",
                    affected_component="test-package"
                )
            ],
            summary={"critical": 0, "high": 1, "medium": 0, "low": 0, "info": 0}
        )
        
        # Mock all scanners
        for scanner in orchestrator.scanners.values():
            scanner.scan = AsyncMock(return_value=mock_scan_result)
            scanner.is_applicable = Mock(return_value=True)
        
        results = await orchestrator.run_security_scan(temp_project)
        
        assert len(results) == len(orchestrator.scanners)
        for scan_type, result in results.items():
            assert result.scan_id == "test-123"
            assert len(result.vulnerabilities) == 1
    
    def test_generate_security_report(self, orchestrator):
        """Test generating security report."""
        scan_results = {
            "dependency": SecurityScanResult(
                scan_id="dep-123",
                scan_type="dependency",
                target_path="/test",
                started_at=datetime.now(),
                completed_at=datetime.now(),
                vulnerabilities=[
                    SecurityVulnerability(
                        id="dep-001",
                        severity="critical",
                        category="dependency",
                        title="Critical dependency vulnerability",
                        description="Test",
                        affected_component="vulnerable-package==1.0.0",
                        remediation="Update to version 2.0.0"
                    )
                ],
                summary={"critical": 1, "high": 0, "medium": 0, "low": 0, "info": 0}
            ),
            "static_code": SecurityScanResult(
                scan_id="code-123",
                scan_type="static_code",
                target_path="/test",
                started_at=datetime.now(),
                completed_at=datetime.now(),
                vulnerabilities=[
                    SecurityVulnerability(
                        id="code-001",
                        severity="high",
                        category="code",
                        title="SQL Injection",
                        description="Test",
                        affected_component="app.py",
                        file_path="app.py",
                        line_number=42
                    )
                ],
                summary={"critical": 0, "high": 1, "medium": 0, "low": 0, "info": 0}
            )
        }
        
        report = orchestrator.generate_security_report(scan_results)
        
        assert report["summary"]["total_vulnerabilities"] == 2
        assert report["summary"]["by_severity"]["critical"] == 1
        assert report["summary"]["by_severity"]["high"] == 1
        assert len(report["top_vulnerabilities"]) == 2
        assert len(report["recommendations"]) > 0
        assert "URGENT" in report["recommendations"][0]


@pytest.mark.integration
class TestSecurityScannerIntegration:
    """Integration tests for security scanner."""
    
    @pytest.mark.asyncio
    async def test_full_security_scan_workflow(self):
        """Test complete security scanning workflow."""
        # Create a test project
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir)
            
            # Create vulnerable Python code
            (project_path / "app.py").write_text('''
import os
password = "admin123"
os.system(f"echo {input()}")
''')
            
            # Create vulnerable dependencies
            (project_path / "requirements.txt").write_text("django==2.2.0")
            
            # Create vulnerable Dockerfile
            (project_path / "Dockerfile").write_text('''
FROM python:3.9
USER root
COPY . /app
''')
            
            # Mock security scanner subprocess calls
            with patch('asyncio.create_subprocess_exec') as mock_subprocess:
                # Mock safety check for Python dependencies
                safety_output = [{
                    "vulnerability_id": "12345",
                    "package": "django",
                    "installed_version": "2.2.0",
                    "safe_versions": ["3.2.0", "4.0.0"],
                    "advisory": "Django 2.2.0 has SQL injection vulnerability allowing remote code execution",
                    "cve": "CVE-2020-1234"
                }]
                
                # Mock bandit check for Python code
                bandit_output = {
                    "results": [{
                        "test_id": "B105",
                        "issue_severity": "high",
                        "issue_text": "Hardcoded password string detected",
                        "filename": str(project_path / "app.py"),
                        "line_number": 2,
                        "issue_confidence": "high"
                    }]
                }
                
                # Mock hadolint check for Dockerfile
                hadolint_output = [{
                    "rule": "DL3002",
                    "level": "error",
                    "message": "Last USER should not be root",
                    "file": str(project_path / "Dockerfile"),
                    "line": 2
                }]
                
                async def mock_communicate(*args, **kwargs):
                    cmd = args[0] if args else mock_subprocess.call_args[0][0]
                    if "safety" in cmd:
                        return (json.dumps(safety_output).encode(), b"")
                    elif "bandit" in cmd:
                        return (json.dumps(bandit_output).encode(), b"")
                    elif "hadolint" in cmd:
                        return (json.dumps(hadolint_output).encode(), b"")
                    else:
                        return (b"", b"")
                
                mock_process = AsyncMock()
                mock_process.communicate = mock_communicate
                mock_process.returncode = 1  # Non-zero for safety when vulnerabilities found
                mock_subprocess.return_value = mock_process
                
                # Run security scan
                orchestrator = SecurityTestOrchestrator()
                results = await orchestrator.run_security_scan(project_path)
                
                # Generate report
                report = orchestrator.generate_security_report(results)
            
            # Verify results
            assert report["summary"]["total_vulnerabilities"] > 0
            assert any(cat in report["summary"]["by_category"] 
                      for cat in ["dependency", "code", "container"])
            
            # Test severity filtering
            high_severity_vulns = []
            for result in results.values():
                high_severity_vulns.extend(
                    [v for v in result.vulnerabilities if v.severity in ["critical", "high"]]
                )
            
            assert len(high_severity_vulns) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])