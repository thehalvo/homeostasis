"""
LLM Security Manager for Homeostasis.

This module provides security, privacy, and ethical safeguards for LLM integration,
including sensitive data handling, data leakage detection, and compliance frameworks.
"""

import json
import logging
import re
import hashlib
import uuid
from typing import Dict, List, Optional, Any, Tuple, Set, Union, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path
import secrets
import base64

from ..security.security_config import SecurityConfig, get_security_config

if TYPE_CHECKING:
    from ..analysis.llm_context_manager import LLMContext


logger = logging.getLogger(__name__)


class SensitiveDataType(Enum):
    """Types of sensitive data that should be scrubbed."""
    API_KEY = "api_key"
    PASSWORD = "password"
    TOKEN = "token"
    SECRET = "secret"
    CREDENTIAL = "credential"
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    DATABASE_URL = "database_url"
    PRIVATE_KEY = "private_key"
    CERTIFICATE = "certificate"
    PERSONAL_NAME = "personal_name"
    ADDRESS = "address"
    MEDICAL_ID = "medical_id"
    FINANCIAL_ACCOUNT = "financial_account"


class ComplianceFramework(Enum):
    """Supported compliance frameworks."""
    HIPAA = "hipaa"
    PCI_DSS = "pci_dss"
    GDPR = "gdpr"
    SOX = "sox"
    FERPA = "ferpa"
    CCPA = "ccpa"
    SOC2 = "soc2"
    ISO27001 = "iso27001"


@dataclass
class SensitiveDataPattern:
    """Pattern for detecting sensitive data."""
    data_type: SensitiveDataType
    pattern: str
    description: str
    compliance_frameworks: List[ComplianceFramework] = field(default_factory=list)
    replacement_strategy: str = "mask"  # mask, remove, redact, hash
    confidence_threshold: float = 0.8


@dataclass
class DataLeakageDetection:
    """Result of data leakage detection."""
    detected: bool
    leakage_type: str
    confidence: float
    location: str
    suggested_action: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityViolation:
    """Security violation detected during LLM processing."""
    violation_id: str
    violation_type: str
    severity: str  # low, medium, high, critical
    timestamp: str
    context_id: Optional[str] = None
    description: str = ""
    remediation_action: str = ""
    compliance_impact: List[ComplianceFramework] = field(default_factory=list)


class LLMSecurityManager:
    """
    Security manager for LLM integration with comprehensive safeguards.
    
    Features:
    - Sensitive data scrubbing and anonymization
    - Data leakage detection
    - Compliance framework enforcement
    - Malicious injection detection
    - Security violation logging
    """
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        """Initialize the LLM security manager."""
        self.config = config or get_security_config()
        self.sensitive_patterns = self._initialize_sensitive_patterns()
        self.security_violations: List[SecurityViolation] = []
        self.anonymization_mapping: Dict[str, str] = {}
        
        # Initialize compliance settings
        self.active_compliance_frameworks = self._load_compliance_frameworks()
        
        logger.info(f"LLM Security Manager initialized with {len(self.sensitive_patterns)} patterns")
    
    def scrub_sensitive_data(self, 
                           text: str, 
                           context_id: Optional[str] = None,
                           preserve_anonymization: bool = True) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Scrub sensitive data from text before sending to LLM.
        
        Args:
            text: Text to scrub
            context_id: Context ID for logging
            preserve_anonymization: Whether to preserve consistent anonymization
            
        Returns:
            Tuple of (scrubbed_text, detections)
        """
        scrubbed_text = text
        detections = []
        
        for pattern in self.sensitive_patterns:
            matches = re.finditer(pattern.pattern, text, re.IGNORECASE | re.MULTILINE)
            
            for match in matches:
                original_value = match.group(0)
                
                # Apply replacement strategy
                if pattern.replacement_strategy == "mask":
                    replacement = self._mask_value(original_value)
                elif pattern.replacement_strategy == "remove":
                    replacement = ""
                elif pattern.replacement_strategy == "redact":
                    replacement = f"[REDACTED_{pattern.data_type.value.upper()}]"
                elif pattern.replacement_strategy == "hash":
                    replacement = self._hash_value(original_value)
                else:
                    replacement = "[SENSITIVE_DATA]"
                
                # Preserve consistent anonymization if requested
                if preserve_anonymization:
                    if original_value not in self.anonymization_mapping:
                        self.anonymization_mapping[original_value] = replacement
                    replacement = self.anonymization_mapping[original_value]
                
                scrubbed_text = scrubbed_text.replace(original_value, replacement)
                
                detections.append({
                    "type": pattern.data_type.value,
                    "original_length": len(original_value),
                    "position": match.start(),
                    "replacement": replacement,
                    "confidence": pattern.confidence_threshold,
                    "compliance_frameworks": [f.value for f in pattern.compliance_frameworks]
                })
        
        # Log scrubbing activity
        if detections:
            logger.info(f"Scrubbed {len(detections)} sensitive data items from text")
            if context_id:
                self._log_scrubbing_activity(context_id, detections)
        
        return scrubbed_text, detections
    
    def detect_data_leakage(self, 
                           prompt: str, 
                           response: str,
                           context_id: Optional[str] = None) -> List[DataLeakageDetection]:
        """
        Detect potential data leakage in LLM prompt/response cycles.
        
        Args:
            prompt: LLM prompt text
            response: LLM response text
            context_id: Context ID for logging
            
        Returns:
            List of detected data leakage incidents
        """
        detections = []
        
        # Check for sensitive data in response that wasn't in prompt
        prompt_sensitive = self._extract_sensitive_data(prompt)
        response_sensitive = self._extract_sensitive_data(response)
        
        # Look for new sensitive data in response
        for data_type, response_items in response_sensitive.items():
            prompt_items = prompt_sensitive.get(data_type, set())
            new_items = response_items - prompt_items
            
            for item in new_items:
                detections.append(DataLeakageDetection(
                    detected=True,
                    leakage_type=f"new_sensitive_data_{data_type}",
                    confidence=0.9,
                    location="llm_response",
                    suggested_action="review_and_redact",
                    details={
                        "data_type": data_type,
                        "leaked_item": self._mask_value(item),
                        "context_id": context_id
                    }
                ))
        
        # Check for potential injection patterns
        injection_detection = self._detect_injection_patterns(prompt, response)
        if injection_detection.detected:
            detections.append(injection_detection)
        
        # Check for unauthorized data disclosure
        disclosure_detection = self._detect_unauthorized_disclosure(response)
        if disclosure_detection.detected:
            detections.append(disclosure_detection)
        
        # Log detections
        if detections:
            logger.warning(f"Detected {len(detections)} potential data leakage incidents")
            if context_id:
                self._log_leakage_detections(context_id, detections)
        
        return detections
    
    def check_compliance_violations(self, 
                                  llm_context: "LLMContext",
                                  proposed_patch: str) -> List[SecurityViolation]:
        """
        Check for compliance framework violations.
        
        Args:
            llm_context: LLM context being processed
            proposed_patch: Proposed patch content
            
        Returns:
            List of compliance violations
        """
        violations = []
        
        for framework in self.active_compliance_frameworks:
            if framework == ComplianceFramework.HIPAA:
                violations.extend(self._check_hipaa_compliance(llm_context, proposed_patch))
            elif framework == ComplianceFramework.PCI_DSS:
                violations.extend(self._check_pci_compliance(llm_context, proposed_patch))
            elif framework == ComplianceFramework.GDPR:
                violations.extend(self._check_gdpr_compliance(llm_context, proposed_patch))
            elif framework == ComplianceFramework.SOX:
                violations.extend(self._check_sox_compliance(llm_context, proposed_patch))
        
        # Log violations
        if violations:
            logger.error(f"Detected {len(violations)} compliance violations")
            self.security_violations.extend(violations)
        
        return violations
    
    def sanitize_llm_context(self, llm_context: "LLMContext") -> "LLMContext":
        """
        Sanitize LLM context before processing.
        
        Args:
            llm_context: Original LLM context
            
        Returns:
            Sanitized LLM context
        """
        # Import here to avoid circular import
        from ..analysis.llm_context_manager import LLMContext
        
        # Create a copy to avoid modifying the original
        sanitized_context = LLMContext(
            context_id=llm_context.context_id,
            created_at=llm_context.created_at,
            updated_at=llm_context.updated_at,
            error_context=llm_context.error_context,
            error_classification=llm_context.error_classification
        )
        
        # Sanitize error message
        if llm_context.error_context.error_message:
            sanitized_message, _ = self.scrub_sensitive_data(
                llm_context.error_context.error_message,
                llm_context.context_id
            )
            sanitized_context.error_context.error_message = sanitized_message
        
        # Sanitize stack trace
        if llm_context.error_context.stack_trace:
            sanitized_trace, _ = self.scrub_sensitive_data(
                llm_context.error_context.stack_trace,
                llm_context.context_id
            )
            sanitized_context.error_context.stack_trace = sanitized_trace
        
        # Sanitize source code snippet
        if llm_context.error_context.source_code_snippet:
            sanitized_code, _ = self.scrub_sensitive_data(
                llm_context.error_context.source_code_snippet,
                llm_context.context_id
            )
            sanitized_context.error_context.source_code_snippet = sanitized_code
        
        # Copy other attributes
        sanitized_context.rule_based_analysis = llm_context.rule_based_analysis
        sanitized_context.ml_analysis = llm_context.ml_analysis
        sanitized_context.language_specific_analysis = llm_context.language_specific_analysis
        sanitized_context.compiler_diagnostics = llm_context.compiler_diagnostics
        sanitized_context.project_structure = llm_context.project_structure
        sanitized_context.related_files = llm_context.related_files
        sanitized_context.dependency_graph = llm_context.dependency_graph
        sanitized_context.similar_errors = llm_context.similar_errors
        sanitized_context.previous_fixes = llm_context.previous_fixes
        sanitized_context.performance_metrics = llm_context.performance_metrics
        sanitized_context.system_state = llm_context.system_state
        sanitized_context.prompt_template = llm_context.prompt_template
        sanitized_context.expected_output_format = llm_context.expected_output_format
        sanitized_context.provider_preferences = llm_context.provider_preferences
        
        return sanitized_context
    
    def validate_llm_response_safety(self, 
                                   response: str,
                                   context_id: Optional[str] = None) -> Tuple[bool, List[str]]:
        """
        Validate that LLM response is safe and doesn't contain harmful content.
        
        Args:
            response: LLM response to validate
            context_id: Context ID for logging
            
        Returns:
            Tuple of (is_safe, violations)
        """
        violations = []
        
        # Check for malicious code patterns
        malicious_patterns = [
            r'rm\s+-rf\s+/',  # Dangerous rm command
            r'sudo\s+rm',     # Sudo with rm
            r'eval\s*\(',     # Eval function calls
            r'exec\s*\(',     # Exec function calls
            r'import\s+os.*system',  # OS system calls
            r'subprocess\.call',      # Subprocess calls
            r'__import__',           # Dynamic imports
            r'globals\(\)',          # Global access
            r'locals\(\)',           # Local access
            r'open\s*\([^)]*["\']w["\']',  # File writing
            r'connect\s*\(',         # Network connections
            r'socket\.',             # Socket usage
            r'urllib\.request',      # HTTP requests
            r'requests\.get|requests\.post',  # HTTP library
        ]
        
        for pattern in malicious_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                violations.append(f"Potentially malicious code pattern detected: {pattern}")
        
        # Check for sensitive data leakage
        leakage_detections = self.detect_data_leakage("", response, context_id)
        for detection in leakage_detections:
            violations.append(f"Data leakage detected: {detection.leakage_type}")
        
        # Check for injection attempts
        injection_patterns = [
            r'<script.*?>',           # Script tags
            r'javascript:',           # JavaScript protocol
            r'on\w+\s*=',            # Event handlers
            r'expression\s*\(',       # CSS expressions
            r'\\x[0-9a-fA-F]{2}',     # Hex encoding
            r'%[0-9a-fA-F]{2}',       # URL encoding
        ]
        
        for pattern in injection_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                violations.append(f"Potential injection pattern detected: {pattern}")
        
        is_safe = len(violations) == 0
        
        if not is_safe:
            logger.warning(f"LLM response safety validation failed: {violations}")
            if context_id:
                self._log_safety_violation(context_id, violations)
        
        return is_safe, violations
    
    def get_security_summary(self) -> Dict[str, Any]:
        """
        Get a summary of security activities and violations.
        
        Returns:
            Security summary dictionary
        """
        return {
            "total_violations": len(self.security_violations),
            "violations_by_type": self._count_violations_by_type(),
            "active_compliance_frameworks": [f.value for f in self.active_compliance_frameworks],
            "sensitive_patterns_count": len(self.sensitive_patterns),
            "anonymization_entries": len(self.anonymization_mapping),
            "recent_violations": [
                {
                    "id": v.violation_id,
                    "type": v.violation_type,
                    "severity": v.severity,
                    "timestamp": v.timestamp
                }
                for v in self.security_violations[-10:]  # Last 10 violations
            ]
        }
    
    def _initialize_sensitive_patterns(self) -> List[SensitiveDataPattern]:
        """Initialize patterns for detecting sensitive data."""
        return [
            # API Keys and tokens
            SensitiveDataPattern(
                data_type=SensitiveDataType.API_KEY,
                pattern=r'(?i)(?:api[_-]?key|apikey|access[_-]?key|secret[_-]?key|private[_-]?key)["\']?\s*[:=]\s*["\']?([a-zA-Z0-9_-]{20,})',
                description="API keys and access tokens",
                compliance_frameworks=[ComplianceFramework.PCI_DSS, ComplianceFramework.SOC2],
                replacement_strategy="redact"
            ),
            SensitiveDataPattern(
                data_type=SensitiveDataType.PASSWORD,
                pattern=r'(?i)(?:password|passwd|pwd)["\']?\s*[:=]\s*["\']?([^\s\'"]{6,})',
                description="Passwords",
                compliance_frameworks=[ComplianceFramework.PCI_DSS, ComplianceFramework.SOC2, ComplianceFramework.GDPR],
                replacement_strategy="redact"
            ),
            SensitiveDataPattern(
                data_type=SensitiveDataType.TOKEN,
                pattern=r'(?i)(?:token|bearer|auth)["\']?\s*[:=]\s*["\']?([a-zA-Z0-9_.-]{20,})',
                description="Authentication tokens",
                compliance_frameworks=[ComplianceFramework.PCI_DSS, ComplianceFramework.SOC2],
                replacement_strategy="redact"
            ),
            
            # Personal information
            SensitiveDataPattern(
                data_type=SensitiveDataType.EMAIL,
                pattern=r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                description="Email addresses",
                compliance_frameworks=[ComplianceFramework.GDPR, ComplianceFramework.HIPAA, ComplianceFramework.CCPA],
                replacement_strategy="mask"
            ),
            SensitiveDataPattern(
                data_type=SensitiveDataType.PHONE,
                pattern=r'(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}',
                description="Phone numbers",
                compliance_frameworks=[ComplianceFramework.GDPR, ComplianceFramework.HIPAA, ComplianceFramework.CCPA],
                replacement_strategy="mask"
            ),
            SensitiveDataPattern(
                data_type=SensitiveDataType.SSN,
                pattern=r'\b(?:\d{3}-\d{2}-\d{4}|\d{9})\b',
                description="Social Security Numbers",
                compliance_frameworks=[ComplianceFramework.HIPAA, ComplianceFramework.SOX],
                replacement_strategy="redact"
            ),
            SensitiveDataPattern(
                data_type=SensitiveDataType.CREDIT_CARD,
                pattern=r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3[0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b',
                description="Credit card numbers",
                compliance_frameworks=[ComplianceFramework.PCI_DSS],
                replacement_strategy="redact"
            ),
            
            # Network and system information
            SensitiveDataPattern(
                data_type=SensitiveDataType.IP_ADDRESS,
                pattern=r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
                description="IP addresses",
                compliance_frameworks=[ComplianceFramework.GDPR, ComplianceFramework.SOC2],
                replacement_strategy="mask"
            ),
            SensitiveDataPattern(
                data_type=SensitiveDataType.DATABASE_URL,
                pattern=r'(?i)(?:postgresql|mysql|mongodb|redis)://[^\s\'"]+',
                description="Database connection URLs",
                compliance_frameworks=[ComplianceFramework.PCI_DSS, ComplianceFramework.SOC2],
                replacement_strategy="redact"
            ),
            
            # Medical and financial identifiers
            SensitiveDataPattern(
                data_type=SensitiveDataType.MEDICAL_ID,
                pattern=r'(?i)(?:mrn|medical[_-]?record|patient[_-]?id)["\']?\s*[:=]\s*["\']?([a-zA-Z0-9-]{6,})',
                description="Medical record numbers",
                compliance_frameworks=[ComplianceFramework.HIPAA],
                replacement_strategy="redact"
            ),
            SensitiveDataPattern(
                data_type=SensitiveDataType.FINANCIAL_ACCOUNT,
                pattern=r'(?i)(?:account[_-]?number|routing[_-]?number|bank[_-]?account)["\']?\s*[:=]\s*["\']?([0-9-]{8,})',
                description="Financial account numbers",
                compliance_frameworks=[ComplianceFramework.PCI_DSS, ComplianceFramework.SOX],
                replacement_strategy="redact"
            ),
        ]
    
    def _load_compliance_frameworks(self) -> List[ComplianceFramework]:
        """Load active compliance frameworks from configuration."""
        frameworks = []
        
        # Get from configuration
        config_frameworks = self.config.get("compliance_frameworks", [])
        
        for framework_name in config_frameworks:
            try:
                framework = ComplianceFramework(framework_name.lower())
                frameworks.append(framework)
            except ValueError:
                logger.warning(f"Unknown compliance framework: {framework_name}")
        
        # Default to basic security if none specified
        if not frameworks:
            frameworks = [ComplianceFramework.SOC2]
        
        return frameworks
    
    def _mask_value(self, value: str) -> str:
        """Mask a sensitive value."""
        if len(value) <= 4:
            return "*" * len(value)
        return value[:2] + "*" * (len(value) - 4) + value[-2:]
    
    def _hash_value(self, value: str) -> str:
        """Hash a sensitive value."""
        return hashlib.sha256(value.encode()).hexdigest()[:16]
    
    def _extract_sensitive_data(self, text: str) -> Dict[str, Set[str]]:
        """Extract sensitive data from text."""
        sensitive_data = {}
        
        for pattern in self.sensitive_patterns:
            matches = re.findall(pattern.pattern, text, re.IGNORECASE)
            if matches:
                data_type = pattern.data_type.value
                if data_type not in sensitive_data:
                    sensitive_data[data_type] = set()
                sensitive_data[data_type].update(matches)
        
        return sensitive_data
    
    def _detect_injection_patterns(self, prompt: str, response: str) -> DataLeakageDetection:
        """Detect potential injection patterns."""
        # Check for prompt injection attempts
        injection_indicators = [
            "ignore previous instructions",
            "forget everything above",
            "system prompt",
            "you are now",
            "pretend to be",
            "roleplay as",
            "act as if",
            "override your guidelines"
        ]
        
        for indicator in injection_indicators:
            if indicator.lower() in prompt.lower():
                return DataLeakageDetection(
                    detected=True,
                    leakage_type="prompt_injection",
                    confidence=0.8,
                    location="prompt",
                    suggested_action="block_request",
                    details={"indicator": indicator}
                )
        
        return DataLeakageDetection(
            detected=False,
            leakage_type="none",
            confidence=0.0,
            location="",
            suggested_action="none"
        )
    
    def _detect_unauthorized_disclosure(self, response: str) -> DataLeakageDetection:
        """Detect unauthorized information disclosure."""
        # Check for system information disclosure
        system_info_patterns = [
            r'(?i)version\s*[:=]\s*[0-9.]+',
            r'(?i)server\s*[:=]\s*[^\s]+',
            r'(?i)database\s*[:=]\s*[^\s]+',
            r'(?i)operating\s*system',
            r'(?i)file\s*path\s*[:=]\s*/[^\s]+',
        ]
        
        for pattern in system_info_patterns:
            if re.search(pattern, response):
                return DataLeakageDetection(
                    detected=True,
                    leakage_type="system_info_disclosure",
                    confidence=0.7,
                    location="response",
                    suggested_action="review_response",
                    details={"pattern": pattern}
                )
        
        return DataLeakageDetection(
            detected=False,
            leakage_type="none",
            confidence=0.0,
            location="",
            suggested_action="none"
        )
    
    def _check_hipaa_compliance(self, llm_context: "LLMContext", proposed_patch: str) -> List[SecurityViolation]:
        """Check HIPAA compliance violations."""
        violations = []
        
        # Check for PHI in context
        phi_patterns = [
            r'(?i)patient',
            r'(?i)medical',
            r'(?i)health',
            r'(?i)diagnosis',
            r'(?i)treatment',
            r'(?i)prescription'
        ]
        
        for pattern in phi_patterns:
            if re.search(pattern, str(llm_context.to_dict())):
                violations.append(SecurityViolation(
                    violation_id=str(uuid.uuid4()),
                    violation_type="hipaa_phi_exposure",
                    severity="high",
                    timestamp=datetime.now().isoformat(),
                    context_id=llm_context.context_id,
                    description=f"Potential PHI detected in context: {pattern}",
                    remediation_action="sanitize_context_before_processing",
                    compliance_impact=[ComplianceFramework.HIPAA]
                ))
        
        return violations
    
    def _check_pci_compliance(self, llm_context: "LLMContext", proposed_patch: str) -> List[SecurityViolation]:
        """Check PCI DSS compliance violations."""
        violations = []
        
        # Check for payment card data
        pci_patterns = [
            r'(?i)credit\s*card',
            r'(?i)payment',
            r'(?i)cardholder',
            r'(?i)cvv',
            r'(?i)expiry',
            r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13})\b'
        ]
        
        for pattern in pci_patterns:
            if re.search(pattern, str(llm_context.to_dict())):
                violations.append(SecurityViolation(
                    violation_id=str(uuid.uuid4()),
                    violation_type="pci_card_data_exposure",
                    severity="critical",
                    timestamp=datetime.now().isoformat(),
                    context_id=llm_context.context_id,
                    description=f"Potential card data detected: {pattern}",
                    remediation_action="immediate_sanitization_required",
                    compliance_impact=[ComplianceFramework.PCI_DSS]
                ))
        
        return violations
    
    def _check_gdpr_compliance(self, llm_context: "LLMContext", proposed_patch: str) -> List[SecurityViolation]:
        """Check GDPR compliance violations."""
        violations = []
        
        # Check for personal data
        gdpr_patterns = [
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'(?i)personal\s*data',
            r'(?i)individual',
            r'(?i)citizen',
            r'(?i)resident'
        ]
        
        for pattern in gdpr_patterns:
            if re.search(pattern, str(llm_context.to_dict())):
                violations.append(SecurityViolation(
                    violation_id=str(uuid.uuid4()),
                    violation_type="gdpr_personal_data_exposure",
                    severity="high",
                    timestamp=datetime.now().isoformat(),
                    context_id=llm_context.context_id,
                    description=f"Potential personal data detected: {pattern}",
                    remediation_action="ensure_data_subject_consent",
                    compliance_impact=[ComplianceFramework.GDPR]
                ))
        
        return violations
    
    def _check_sox_compliance(self, llm_context: "LLMContext", proposed_patch: str) -> List[SecurityViolation]:
        """Check SOX compliance violations."""
        violations = []
        
        # Check for financial data
        sox_patterns = [
            r'(?i)financial',
            r'(?i)revenue',
            r'(?i)earnings',
            r'(?i)accounting',
            r'(?i)audit',
            r'(?i)securities'
        ]
        
        for pattern in sox_patterns:
            if re.search(pattern, str(llm_context.to_dict())):
                violations.append(SecurityViolation(
                    violation_id=str(uuid.uuid4()),
                    violation_type="sox_financial_data_exposure",
                    severity="high",
                    timestamp=datetime.now().isoformat(),
                    context_id=llm_context.context_id,
                    description=f"Potential financial data detected: {pattern}",
                    remediation_action="ensure_financial_data_controls",
                    compliance_impact=[ComplianceFramework.SOX]
                ))
        
        return violations
    
    def _count_violations_by_type(self) -> Dict[str, int]:
        """Count violations by type."""
        counts = {}
        for violation in self.security_violations:
            counts[violation.violation_type] = counts.get(violation.violation_type, 0) + 1
        return counts
    
    def _log_scrubbing_activity(self, context_id: str, detections: List[Dict[str, Any]]):
        """Log data scrubbing activity."""
        logger.info(f"Data scrubbing activity for context {context_id}: {len(detections)} items scrubbed")
    
    def _log_leakage_detections(self, context_id: str, detections: List[DataLeakageDetection]):
        """Log data leakage detections."""
        logger.warning(f"Data leakage detections for context {context_id}: {len(detections)} incidents")
    
    def _log_safety_violation(self, context_id: str, violations: List[str]):
        """Log safety violations."""
        logger.error(f"Safety violations for context {context_id}: {violations}")


def create_llm_security_manager(config: Optional[SecurityConfig] = None) -> LLMSecurityManager:
    """Create and return a configured LLM security manager."""
    return LLMSecurityManager(config)