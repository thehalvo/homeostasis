"""
Security guardrails for LLM interactions.

This module provides comprehensive security measures including:
1. PII detection and scrubbing
2. Prompt injection detection
3. Data leakage prevention
4. Malicious content filtering
5. Compliance enforcement
"""

import re
import json
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set, Pattern
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import threading

from .logger import MonitoringLogger


class SecurityLevel(Enum):
    """Security enforcement levels."""
    PERMISSIVE = "permissive"    # Log violations but allow
    RESTRICTIVE = "restrictive"  # Block violations
    STRICT = "strict"           # Block and quarantine


class ViolationType(Enum):
    """Types of security violations."""
    PII_DETECTED = "pii_detected"
    PROMPT_INJECTION = "prompt_injection"
    DATA_LEAKAGE = "data_leakage"
    MALICIOUS_CONTENT = "malicious_content"
    UNSAFE_CODE = "unsafe_code"
    POLICY_VIOLATION = "policy_violation"
    CREDENTIAL_EXPOSURE = "credential_exposure"


@dataclass
class SecurityViolation:
    """Represents a security violation."""
    violation_type: ViolationType
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    detected_content: str
    scrubbed_content: str
    confidence: float
    timestamp: float
    source: str  # 'prompt' or 'response'
    patterns_matched: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComplianceRule:
    """Compliance rule configuration."""
    name: str
    description: str
    patterns: List[Pattern]
    violation_type: ViolationType
    severity: str
    action: str  # 'log', 'block', 'scrub'
    frameworks: List[str] = field(default_factory=list)  # HIPAA, PCI, GDPR, etc.


class PIIDetector:
    """Enhanced PII detection with multiple detection strategies."""
    
    def __init__(self):
        """Initialize PII detection patterns."""
        # Basic PII patterns
        self.patterns = {
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'phone_us': re.compile(r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b'),
            'phone_intl': re.compile(r'\+(?:[0-9] ?){6,14}[0-9]\b'),
            'ssn': re.compile(r'\b\d{3}-?\d{2}-?\d{4}\b'),
            'ssn_spaced': re.compile(r'\b\d{3}\s\d{2}\s\d{4}\b'),
            'credit_card': re.compile(r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3[0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b'),
            'passport': re.compile(r'\b[A-Z]{1,2}[0-9]{6,9}\b'),
            'drivers_license': re.compile(r'\b[A-Z]{1,2}[0-9]{7,8}\b'),
            'ip_address': re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'),
            'mac_address': re.compile(r'\b[0-9A-Fa-f]{2}[:-]?[0-9A-Fa-f]{2}[:-]?[0-9A-Fa-f]{2}[:-]?[0-9A-Fa-f]{2}[:-]?[0-9A-Fa-f]{2}[:-]?[0-9A-Fa-f]{2}\b'),
            'bank_account': re.compile(r'\b[0-9]{8,17}\b'),
            'routing_number': re.compile(r'\b[0-9]{9}\b'),
            'iban': re.compile(r'\b[A-Z]{2}[0-9]{2}[A-Z0-9]{4}[0-9]{7}([A-Z0-9]?){0,16}\b'),
            'date_of_birth': re.compile(r'\b(?:0[1-9]|1[0-2])[/-](?:0[1-9]|[12][0-9]|3[01])[/-](?:19|20)\d{2}\b'),
            'medical_record': re.compile(r'\bMRN?[-\s]?:?\s*[0-9]{6,10}\b', re.IGNORECASE),
            'national_id': re.compile(r'\b(?:NID|ID)[-\s]?:?\s*[A-Z0-9]{8,15}\b', re.IGNORECASE),
        }
        
        # Financial data patterns
        self.financial_patterns = {
            'account_number': re.compile(r'\b(?:account|acct)[-\s]?(?:number|num|#)?[-\s]?:?\s*[0-9]{6,18}\b', re.IGNORECASE),
            'sort_code': re.compile(r'\b[0-9]{2}-?[0-9]{2}-?[0-9]{2}\b'),
            'swift_code': re.compile(r'\b[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}([A-Z0-9]{3})?\b'),
            'tax_id': re.compile(r'\b(?:TIN|EIN)[-\s]?:?\s*[0-9]{2}-?[0-9]{7}\b', re.IGNORECASE),
        }
        
        # Context-based PII patterns
        self.context_patterns = {
            'personal_info': re.compile(r'(?i)\b(?:my|personal)\s+(?:address|phone|email|ssn|social\s+security)\b'),
            'confidential': re.compile(r'(?i)\b(?:confidential|private|sensitive)\s+(?:information|data|details)\b'),
            'healthcare': re.compile(r'(?i)\b(?:patient|medical|health)\s+(?:record|information|data)\b'),
            'financial_info': re.compile(r'(?i)\b(?:financial|banking|credit)\s+(?:information|details|account)\b'),
        }
        
        # Name patterns (more complex detection)
        self.name_patterns = [
            re.compile(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'),  # First Last
            re.compile(r'\b[A-Z][a-z]+ [A-Z]\. [A-Z][a-z]+\b'),  # First M. Last
            re.compile(r'\b(?:Mr|Mrs|Ms|Dr|Prof)\.? [A-Z][a-z]+ [A-Z][a-z]+\b'),  # Title First Last
        ]
        
        # Common name lists for better detection
        self.common_first_names = {
            'james', 'john', 'robert', 'michael', 'william', 'david', 'richard', 'joseph',
            'thomas', 'christopher', 'charles', 'daniel', 'matthew', 'anthony', 'mark',
            'mary', 'patricia', 'jennifer', 'linda', 'elizabeth', 'barbara', 'susan',
            'jessica', 'sarah', 'karen', 'nancy', 'lisa', 'betty', 'helen', 'sandra'
        }
        
        self.common_last_names = {
            'smith', 'johnson', 'williams', 'brown', 'jones', 'garcia', 'miller',
            'davis', 'rodriguez', 'martinez', 'hernandez', 'lopez', 'gonzalez',
            'wilson', 'anderson', 'thomas', 'taylor', 'moore', 'jackson', 'martin'
        }
    
    def detect_pii(self, text: str, context: str = '') -> Tuple[bool, List[str], Dict[str, List[str]]]:
        """
        Comprehensive PII detection.
        
        Args:
            text: Text to analyze
            context: Additional context for detection
            
        Returns:
            Tuple of (has_pii, detected_types, detailed_matches)
        """
        detected_types = []
        detailed_matches = defaultdict(list)
        
        # Check basic patterns
        for pii_type, pattern in self.patterns.items():
            matches = pattern.findall(text)
            if matches:
                detected_types.append(pii_type)
                detailed_matches[pii_type].extend(matches)
        
        # Check financial patterns
        for fin_type, pattern in self.financial_patterns.items():
            matches = pattern.findall(text)
            if matches:
                detected_types.append(f'financial_{fin_type}')
                detailed_matches[f'financial_{fin_type}'].extend(matches)
        
        # Check context patterns
        for ctx_type, pattern in self.context_patterns.items():
            if pattern.search(text):
                detected_types.append(f'context_{ctx_type}')
                detailed_matches[f'context_{ctx_type}'].append(pattern.pattern)
        
        # Check for names using patterns and common name lists
        name_matches = self._detect_names(text)
        if name_matches:
            detected_types.append('names')
            detailed_matches['names'].extend(name_matches)
        
        # Check for addresses
        address_matches = self._detect_addresses(text)
        if address_matches:
            detected_types.append('addresses')
            detailed_matches['addresses'].extend(address_matches)
        
        return len(detected_types) > 0, detected_types, dict(detailed_matches)
    
    def _detect_names(self, text: str) -> List[str]:
        """Detect potential names in text."""
        names = []
        
        # Use pattern matching
        for pattern in self.name_patterns:
            matches = pattern.findall(text)
            for match in matches:
                # Validate against common names
                parts = match.lower().split()
                if len(parts) >= 2:
                    first_name = parts[0]
                    last_name = parts[-1]
                    
                    if (first_name in self.common_first_names or 
                            last_name in self.common_last_names):
                        names.append(match)
        
        return names
    
    def _detect_addresses(self, text: str) -> List[str]:
        """Detect potential addresses in text."""
        address_patterns = [
            re.compile(r'\b\d+\s+[A-Z][a-z]+\s+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd)\b'),
            re.compile(r'\b[A-Z][a-z]+,\s*[A-Z]{2}\s*\d{5}(?:-\d{4})?\b'),  # City, ST ZIP
            re.compile(r'\b\d{5}(?:-\d{4})?\b'),  # ZIP codes
        ]
        
        addresses = []
        for pattern in address_patterns:
            matches = pattern.findall(text)
            addresses.extend(matches)
        
        return addresses
    
    def scrub_pii(self, text: str, detected_matches: Dict[str, List[str]]) -> str:
        """
        Scrub detected PII from text.
        
        Args:
            text: Original text
            detected_matches: PII matches from detect_pii
            
        Returns:
            Scrubbed text
        """
        scrubbed = text
        
        # Define replacement patterns
        replacements = {
            'email': '[EMAIL_REDACTED]',
            'phone_us': '[PHONE_REDACTED]',
            'phone_intl': '[PHONE_REDACTED]',
            'ssn': '[SSN_REDACTED]',
            'ssn_spaced': '[SSN_REDACTED]',
            'credit_card': '[CREDIT_CARD_REDACTED]',
            'passport': '[PASSPORT_REDACTED]',
            'drivers_license': '[LICENSE_REDACTED]',
            'ip_address': '[IP_REDACTED]',
            'mac_address': '[MAC_REDACTED]',
            'bank_account': '[ACCOUNT_REDACTED]',
            'routing_number': '[ROUTING_REDACTED]',
            'iban': '[IBAN_REDACTED]',
            'date_of_birth': '[DOB_REDACTED]',
            'medical_record': '[MRN_REDACTED]',
            'national_id': '[ID_REDACTED]',
            'names': '[NAME_REDACTED]',
            'addresses': '[ADDRESS_REDACTED]',
        }
        
        # Replace detected PII
        for pii_type, matches in detected_matches.items():
            if pii_type in replacements:
                replacement = replacements[pii_type]
                for match in matches:
                    if isinstance(match, str):
                        scrubbed = scrubbed.replace(match, replacement)
        
        # Apply pattern-based replacements for remaining items
        for pii_type, pattern in self.patterns.items():
            if pii_type in replacements:
                scrubbed = pattern.sub(replacements[pii_type], scrubbed)
        
        return scrubbed


class PromptInjectionDetector:
    """Detects prompt injection and manipulation attempts."""
    
    def __init__(self):
        """Initialize prompt injection detection patterns."""
        self.injection_patterns = {
            'ignore_instructions': [
                re.compile(r'(?i)ignore\s+(?:previous|all|the|your)\s+(?:instructions|prompts?|commands?)', re.IGNORECASE),
                re.compile(r'(?i)disregard\s+(?:previous|all|the|your)\s+(?:instructions|prompts?)', re.IGNORECASE),
                re.compile(r'(?i)forget\s+(?:previous|all|the|your)\s+(?:instructions|prompts?)', re.IGNORECASE),
            ],
            'role_manipulation': [
                re.compile(r'(?i)(?:act|pretend|behave)\s+(?:as|like)\s+(?:if\s+)?(?:you\s+(?:are|were)|a)', re.IGNORECASE),
                re.compile(r'(?i)you\s+are\s+now\s+(?:a|an)', re.IGNORECASE),
                re.compile(r'(?i)from\s+now\s+on,?\s+you\s+(?:are|will\s+be)', re.IGNORECASE),
            ],
            'system_override': [
                re.compile(r'(?i)system\s*[:.]?\s*(?:override|bypass|jailbreak)', re.IGNORECASE),
                re.compile(r'(?i)admin\s+(?:mode|access|override)', re.IGNORECASE),
                re.compile(r'(?i)developer\s+(?:mode|access)', re.IGNORECASE),
            ],
            'prompt_leakage': [
                re.compile(r'(?i)(?:show|reveal|display|tell)\s+me\s+your\s+(?:system\s+)?(?:prompt|instructions)', re.IGNORECASE),
                re.compile(r'(?i)what\s+(?:are|is)\s+your\s+(?:system\s+)?(?:prompt|instructions)', re.IGNORECASE),
                re.compile(r'(?i)repeat\s+your\s+(?:system\s+)?(?:prompt|instructions)', re.IGNORECASE),
            ],
            'boundary_testing': [
                re.compile(r'(?i)can\s+you\s+(?:help|assist)\s+me\s+(?:with|to)\s+(?:hacking|breaking)', re.IGNORECASE),
                re.compile(r'(?i)how\s+(?:do\s+i|can\s+i|to)\s+(?:hack|break|bypass)', re.IGNORECASE),
                re.compile(r'(?i)tell\s+me\s+how\s+to\s+(?:break|bypass|hack)', re.IGNORECASE),
            ],
            'data_extraction': [
                re.compile(r'(?i)(?:extract|dump|list|show)\s+(?:all\s+)?(?:data|information|files|users)', re.IGNORECASE),
                re.compile(r'(?i)give\s+me\s+(?:access\s+to|all\s+the)', re.IGNORECASE),
                re.compile(r'(?i)download\s+(?:all|the)\s+(?:data|files)', re.IGNORECASE),
            ]
        }
        
        # Advanced injection techniques
        self.advanced_patterns = [
            # Unicode and encoding attacks
            re.compile(r'\\u[0-9a-f]{4}', re.IGNORECASE),
            re.compile(r'%[0-9a-f]{2}', re.IGNORECASE),
            
            # Hidden characters and zero-width spaces
            re.compile(r'[\u200b-\u200d\ufeff]'),
            
            # Markdown/HTML injection
            re.compile(r'<script[^>]*>', re.IGNORECASE),
            re.compile(r'javascript:', re.IGNORECASE),
            re.compile(r'on\w+\s*=', re.IGNORECASE),
        ]
    
    def detect_injection(self, text: str) -> Tuple[bool, List[str], float]:
        """
        Detect prompt injection attempts.
        
        Args:
            text: Text to analyze
            
        Returns:
            Tuple of (has_injection, detected_types, confidence_score)
        """
        detected_types = []
        confidence_scores = []
        
        # Check standard injection patterns
        for injection_type, patterns in self.injection_patterns.items():
            type_matches = 0
            for pattern in patterns:
                if pattern.search(text):
                    type_matches += 1
            
            if type_matches > 0:
                detected_types.append(injection_type)
                # Higher confidence for more matches
                confidence_scores.append(min(0.9, 0.3 + (type_matches * 0.2)))
        
        # Check advanced patterns
        advanced_matches = 0
        for pattern in self.advanced_patterns:
            if pattern.search(text):
                advanced_matches += 1
        
        if advanced_matches > 0:
            detected_types.append('advanced_injection')
            confidence_scores.append(min(0.8, 0.4 + (advanced_matches * 0.1)))
        
        # Calculate overall confidence
        overall_confidence = max(confidence_scores) if confidence_scores else 0.0
        
        return len(detected_types) > 0, detected_types, overall_confidence


class ContentSafetyDetector:
    """Detects unsafe, malicious, or policy-violating content."""
    
    def __init__(self):
        """Initialize content safety detection."""
        self.unsafe_patterns = {
            'malware_generation': [
                re.compile(r'(?i)(?:create|generate|write|build)\s+(?:a\s+)?(?:virus|malware|trojan|backdoor)', re.IGNORECASE),
                re.compile(r'(?i)how\s+to\s+(?:create|make|build)\s+(?:malware|virus)', re.IGNORECASE),
            ],
            'illegal_activities': [
                re.compile(r'(?i)how\s+to\s+(?:sell|buy|make)\s+(?:drugs|illegal)', re.IGNORECASE),
                re.compile(r'(?i)(?:money\s+laundering|tax\s+evasion|fraud)', re.IGNORECASE),
            ],
            'violence_incitement': [
                re.compile(r'(?i)how\s+to\s+(?:kill|murder|harm|hurt)', re.IGNORECASE),
                re.compile(r'(?i)(?:bomb|explosive|weapon)\s+(?:making|creation|instructions)', re.IGNORECASE),
            ],
            'hate_speech': [
                re.compile(r'(?i)(?:hate|attack|target)\s+(?:people|groups?)\s+(?:because|based\s+on)', re.IGNORECASE),
            ],
            'privacy_violation': [
                re.compile(r'(?i)(?:spy|monitor|track)\s+(?:on\s+)?(?:someone|people)', re.IGNORECASE),
                re.compile(r'(?i)(?:steal|extract|access)\s+(?:personal|private)\s+(?:data|information)', re.IGNORECASE),
            ]
        }
        
        # Code safety patterns
        self.code_safety_patterns = {
            'dangerous_functions': [
                re.compile(r'\b(?:eval|exec|system|shell_exec|passthru)\s*\(', re.IGNORECASE),
                re.compile(r'\b(?:rm\s+-rf|del\s+/f|format\s+c:)', re.IGNORECASE),
            ],
            'network_exploitation': [
                re.compile(r'\b(?:nmap|metasploit|sqlmap|burpsuite)\b', re.IGNORECASE),
                re.compile(r'(?:sql\s+injection|xss|csrf|rce)', re.IGNORECASE),
            ]
        }
    
    def detect_unsafe_content(self, text: str) -> Tuple[bool, List[str], str]:
        """
        Detect unsafe content.
        
        Args:
            text: Text to analyze
            
        Returns:
            Tuple of (has_unsafe_content, detected_types, severity)
        """
        detected_types = []
        max_severity = 'low'
        
        # Check unsafe content patterns
        for content_type, patterns in self.unsafe_patterns.items():
            for pattern in patterns:
                if pattern.search(text):
                    detected_types.append(content_type)
                    # Set severity based on content type
                    if content_type in ['violence_incitement', 'illegal_activities']:
                        max_severity = 'critical'
                    elif content_type in ['malware_generation', 'privacy_violation']:
                        max_severity = 'high' if max_severity != 'critical' else max_severity
                    else:
                        max_severity = 'medium' if max_severity not in ['high', 'critical'] else max_severity
                    break
        
        # Check code safety
        for code_type, patterns in self.code_safety_patterns.items():
            for pattern in patterns:
                if pattern.search(text):
                    detected_types.append(f'code_{code_type}')
                    max_severity = 'high' if max_severity not in ['critical'] else max_severity
                    break
        
        return len(detected_types) > 0, detected_types, max_severity


class SecurityGuardrails:
    """Comprehensive security guardrails for LLM interactions."""
    
    def __init__(self, 
                 security_level: SecurityLevel = SecurityLevel.RESTRICTIVE,
                 compliance_rules: Optional[List[ComplianceRule]] = None,
                 storage_dir: Optional[Path] = None):
        """
        Initialize security guardrails.
        
        Args:
            security_level: Security enforcement level
            compliance_rules: Custom compliance rules
            storage_dir: Directory to store security logs
        """
        self.logger = MonitoringLogger("security_guardrails")
        self.security_level = security_level
        
        # Storage
        self.storage_dir = storage_dir or Path("logs/security")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize detectors
        self.pii_detector = PIIDetector()
        self.injection_detector = PromptInjectionDetector()
        self.content_safety_detector = ContentSafetyDetector()
        
        # Compliance rules
        self.compliance_rules = compliance_rules or self._get_default_compliance_rules()
        
        # Violation tracking
        self.violations_history = []
        self.quarantine_hashes: Set[str] = set()
        
        # Thread safety
        self._lock = threading.Lock()
        
        self.logger.info(f"Initialized security guardrails with {security_level.value} level")
    
    def _get_default_compliance_rules(self) -> List[ComplianceRule]:
        """Get default compliance rules for common frameworks."""
        rules = []
        
        # HIPAA rules
        rules.append(ComplianceRule(
            name="HIPAA_PHI_Detection",
            description="Detect Protected Health Information",
            patterns=[
                re.compile(r'(?i)(?:patient|medical|health)\s+(?:record|information|data)', re.IGNORECASE),
                re.compile(r'\bMRN?[-\s]?:?\s*[0-9]{6,10}\b', re.IGNORECASE),
            ],
            violation_type=ViolationType.PII_DETECTED,
            severity="high",
            action="block",
            frameworks=["HIPAA"]
        ))
        
        # PCI DSS rules
        rules.append(ComplianceRule(
            name="PCI_DSS_Card_Data",
            description="Detect payment card data",
            patterns=[
                re.compile(r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3[0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b'),
                re.compile(r'\b[0-9]{3,4}\b'),  # CVV
            ],
            violation_type=ViolationType.PII_DETECTED,
            severity="critical",
            action="block",
            frameworks=["PCI_DSS"]
        ))
        
        # GDPR rules
        rules.append(ComplianceRule(
            name="GDPR_Personal_Data",
            description="Detect personal data under GDPR",
            patterns=[
                re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
                re.compile(r'\b(?:\+?[1-9]\d{1,14})\b'),
            ],
            violation_type=ViolationType.PII_DETECTED,
            severity="medium",
            action="scrub",
            frameworks=["GDPR"]
        ))
        
        # SOX rules
        rules.append(ComplianceRule(
            name="SOX_Financial_Data",
            description="Detect financial data for SOX compliance",
            patterns=[
                re.compile(r'(?i)(?:financial|accounting|revenue|profit)\s+(?:statement|data|report)', re.IGNORECASE),
                re.compile(r'(?i)(?:insider|material)\s+(?:information|trading)', re.IGNORECASE),
            ],
            violation_type=ViolationType.POLICY_VIOLATION,
            severity="high",
            action="block",
            frameworks=["SOX"]
        ))
        
        return rules
    
    def _generate_content_hash(self, content: str) -> str:
        """Generate hash for content identification."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _apply_compliance_rules(self, text: str, source: str) -> List[SecurityViolation]:
        """Apply compliance rules to text."""
        violations = []
        
        for rule in self.compliance_rules:
            for pattern in rule.patterns:
                if pattern.search(text):
                    violation = SecurityViolation(
                        violation_type=rule.violation_type,
                        severity=rule.severity,
                        description=f"Compliance violation: {rule.description}",
                        detected_content=text[:200] + "..." if len(text) > 200 else text,
                        scrubbed_content="",
                        confidence=0.9,
                        timestamp=time.time(),
                        source=source,
                        patterns_matched=[rule.name],
                        metadata={
                            'compliance_frameworks': rule.frameworks,
                            'rule_action': rule.action
                        }
                    )
                    violations.append(violation)
                    break
        
        return violations
    
    def analyze_content(self, 
                       content: str, 
                       source: str = 'unknown',
                       context: Dict[str, Any] = None) -> Tuple[bool, List[SecurityViolation], str]:
        """
        Comprehensive content analysis for security violations.
        
        Args:
            content: Content to analyze
            source: Source of content ('prompt' or 'response')
            context: Additional context information
            
        Returns:
            Tuple of (is_safe, violations, scrubbed_content)
        """
        violations = []
        scrubbed_content = content
        context = context or {}
        
        with self._lock:
            # Check if content is quarantined
            content_hash = self._generate_content_hash(content)
            if content_hash in self.quarantine_hashes:
                violation = SecurityViolation(
                    violation_type=ViolationType.POLICY_VIOLATION,
                    severity="critical",
                    description="Content matches quarantined hash",
                    detected_content=content[:100] + "..." if len(content) > 100 else content,
                    scrubbed_content="[QUARANTINED_CONTENT]",
                    confidence=1.0,
                    timestamp=time.time(),
                    source=source
                )
                violations.append(violation)
                return False, violations, "[QUARANTINED_CONTENT]"
            
            # PII Detection
            has_pii, pii_types, pii_matches = self.pii_detector.detect_pii(content)
            if has_pii:
                scrubbed_content = self.pii_detector.scrub_pii(content, pii_matches)
                
                violation = SecurityViolation(
                    violation_type=ViolationType.PII_DETECTED,
                    severity="high",
                    description=f"PII detected: {', '.join(pii_types)}",
                    detected_content=content[:200] + "..." if len(content) > 200 else content,
                    scrubbed_content=scrubbed_content,
                    confidence=0.8,
                    timestamp=time.time(),
                    source=source,
                    patterns_matched=pii_types,
                    metadata={'pii_matches': pii_matches}
                )
                violations.append(violation)
            
            # Prompt Injection Detection
            has_injection, injection_types, injection_confidence = self.injection_detector.detect_injection(content)
            if has_injection:
                violation = SecurityViolation(
                    violation_type=ViolationType.PROMPT_INJECTION,
                    severity="high",
                    description=f"Prompt injection detected: {', '.join(injection_types)}",
                    detected_content=content[:200] + "..." if len(content) > 200 else content,
                    scrubbed_content="[POTENTIAL_INJECTION_BLOCKED]",
                    confidence=injection_confidence,
                    timestamp=time.time(),
                    source=source,
                    patterns_matched=injection_types
                )
                violations.append(violation)
            
            # Content Safety Detection
            has_unsafe, unsafe_types, unsafe_severity = self.content_safety_detector.detect_unsafe_content(content)
            if has_unsafe:
                violation = SecurityViolation(
                    violation_type=ViolationType.MALICIOUS_CONTENT,
                    severity=unsafe_severity,
                    description=f"Unsafe content detected: {', '.join(unsafe_types)}",
                    detected_content=content[:200] + "..." if len(content) > 200 else content,
                    scrubbed_content="[UNSAFE_CONTENT_BLOCKED]",
                    confidence=0.8,
                    timestamp=time.time(),
                    source=source,
                    patterns_matched=unsafe_types
                )
                violations.append(violation)
            
            # Apply compliance rules
            compliance_violations = self._apply_compliance_rules(content, source)
            violations.extend(compliance_violations)
            
            # Store violations
            for violation in violations:
                self.violations_history.append(violation)
                
                # Log violation
                self.logger.warning(f"Security violation: {violation.description}",
                                  violation_type=violation.violation_type.value,
                                  severity=violation.severity,
                                  source=violation.source,
                                  confidence=violation.confidence,
                                  patterns_matched=violation.patterns_matched)
                
                # Handle quarantine for strict mode
                if (self.security_level == SecurityLevel.STRICT and 
                    violation.severity in ['high', 'critical']):
                    self.quarantine_hashes.add(content_hash)
            
            # Determine if content is safe based on security level
            is_safe = True
            
            if self.security_level == SecurityLevel.RESTRICTIVE:
                # Block high and critical violations
                high_severity_violations = [v for v in violations if v.severity in ['high', 'critical']]
                if high_severity_violations:
                    is_safe = False
                    scrubbed_content = "[CONTENT_BLOCKED_BY_SECURITY_POLICY]"
            
            elif self.security_level == SecurityLevel.STRICT:
                # Block any violations
                if violations:
                    is_safe = False
                    scrubbed_content = "[CONTENT_BLOCKED_BY_SECURITY_POLICY]"
            
            # For PERMISSIVE mode, always allow but log violations
            
            return is_safe, violations, scrubbed_content
    
    def get_security_report(self, time_window: int = 86400) -> Dict[str, Any]:
        """
        Generate security report for specified time window.
        
        Args:
            time_window: Time window in seconds (default: 24 hours)
            
        Returns:
            Security report
        """
        current_time = time.time()
        cutoff_time = current_time - time_window
        
        with self._lock:
            recent_violations = [
                v for v in self.violations_history
                if v.timestamp >= cutoff_time
            ]
        
        # Group violations by type
        violations_by_type = defaultdict(int)
        violations_by_severity = defaultdict(int)
        violations_by_source = defaultdict(int)
        
        for violation in recent_violations:
            violations_by_type[violation.violation_type.value] += 1
            violations_by_severity[violation.severity] += 1
            violations_by_source[violation.source] += 1
        
        # Calculate trends
        total_violations = len(recent_violations)
        
        return {
            'report_timestamp': current_time,
            'time_window_hours': time_window / 3600,
            'security_level': self.security_level.value,
            'total_violations': total_violations,
            'quarantined_items': len(self.quarantine_hashes),
            'violations_by_type': dict(violations_by_type),
            'violations_by_severity': dict(violations_by_severity),
            'violations_by_source': dict(violations_by_source),
            'top_violations': [
                {
                    'type': v.violation_type.value,
                    'severity': v.severity,
                    'description': v.description,
                    'timestamp': v.timestamp,
                    'confidence': v.confidence
                }
                for v in sorted(recent_violations, key=lambda x: x.timestamp, reverse=True)[:10]
            ],
            'recommendations': self._generate_security_recommendations(recent_violations)
        }
    
    def _generate_security_recommendations(self, violations: List[SecurityViolation]) -> List[str]:
        """Generate security recommendations based on violations."""
        recommendations = []
        
        if not violations:
            return ["No security violations detected. System operating normally."]
        
        # Analyze violation patterns
        pii_violations = [v for v in violations if v.violation_type == ViolationType.PII_DETECTED]
        injection_violations = [v for v in violations if v.violation_type == ViolationType.PROMPT_INJECTION]
        unsafe_violations = [v for v in violations if v.violation_type == ViolationType.MALICIOUS_CONTENT]
        
        if len(pii_violations) > 5:
            recommendations.append("High PII detection rate. Consider implementing stronger input validation and user training.")
        
        if len(injection_violations) > 3:
            recommendations.append("Multiple prompt injection attempts detected. Consider implementing additional input sanitization.")
        
        if len(unsafe_violations) > 2:
            recommendations.append("Unsafe content detected. Review content policies and user access controls.")
        
        if len(violations) > 20:
            recommendations.append("High overall violation rate. Consider increasing security level or implementing additional controls.")
        
        # Check for critical violations
        critical_violations = [v for v in violations if v.severity == 'critical']
        if critical_violations:
            recommendations.append(f"URGENT: {len(critical_violations)} critical security violations detected. Immediate review required.")
        
        return recommendations
    
    def export_security_logs(self, output_file: Path, time_window: Optional[int] = None) -> None:
        """
        Export security logs to file.
        
        Args:
            output_file: Output file path
            time_window: Time window in seconds (None for all logs)
        """
        current_time = time.time()
        
        with self._lock:
            if time_window:
                cutoff_time = current_time - time_window
                violations_to_export = [v for v in self.violations_history if v.timestamp >= cutoff_time]
            else:
                violations_to_export = self.violations_history
        
        export_data = {
            'export_timestamp': current_time,
            'security_level': self.security_level.value,
            'time_window': time_window,
            'total_violations': len(violations_to_export),
            'violations': [
                {
                    'violation_type': v.violation_type.value,
                    'severity': v.severity,
                    'description': v.description,
                    'detected_content': v.detected_content,
                    'scrubbed_content': v.scrubbed_content,
                    'confidence': v.confidence,
                    'timestamp': v.timestamp,
                    'source': v.source,
                    'patterns_matched': v.patterns_matched,
                    'metadata': v.metadata
                }
                for v in violations_to_export
            ],
            'security_report': self.get_security_report(time_window or 86400)
        }
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"Exported {len(violations_to_export)} security violations to {output_file}")
    
    def update_security_level(self, new_level: SecurityLevel) -> None:
        """Update security enforcement level."""
        old_level = self.security_level
        self.security_level = new_level
        
        self.logger.info(f"Security level updated from {old_level.value} to {new_level.value}")
    
    def add_compliance_rule(self, rule: ComplianceRule) -> None:
        """Add a custom compliance rule."""
        self.compliance_rules.append(rule)
        self.logger.info(f"Added compliance rule: {rule.name} for frameworks: {', '.join(rule.frameworks)}")
    
    def clear_quarantine(self, content_hash: Optional[str] = None) -> None:
        """Clear quarantine for specific hash or all items."""
        with self._lock:
            if content_hash:
                if content_hash in self.quarantine_hashes:
                    self.quarantine_hashes.remove(content_hash)
                    self.logger.info(f"Cleared quarantine for hash: {content_hash}")
            else:
                self.quarantine_hashes.clear()
                self.logger.info("Cleared all quarantined items")