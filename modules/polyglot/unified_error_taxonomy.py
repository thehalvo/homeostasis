"""
Unified error taxonomy for polyglot systems.
Provides a common classification system for errors across all programming languages.
"""

import json
import logging
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from abc import ABC, abstractmethod


class ErrorCategory(Enum):
    """High-level error categories that apply across all languages."""
    SYNTAX = "syntax"
    RUNTIME = "runtime"
    COMPILATION = "compilation"
    CONFIGURATION = "configuration"
    DEPENDENCY = "dependency"
    NETWORK = "network"
    SECURITY = "security"
    PERFORMANCE = "performance"
    CONCURRENCY = "concurrency"
    MEMORY = "memory"
    IO = "io"
    DATABASE = "database"
    API = "api"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    VALIDATION = "validation"
    BUSINESS_LOGIC = "business_logic"
    FRAMEWORK = "framework"
    DEPLOYMENT = "deployment"
    INFRASTRUCTURE = "infrastructure"


class ErrorSeverity(Enum):
    """Severity levels for errors."""
    CRITICAL = "critical"  # System crash or data loss
    HIGH = "high"         # Major functionality broken
    MEDIUM = "medium"     # Partial functionality affected
    LOW = "low"          # Minor issues
    WARNING = "warning"   # Potential issues


class ErrorScope(Enum):
    """Scope of error impact."""
    LOCAL = "local"              # Single function/method
    MODULE = "module"            # Single module/class
    SERVICE = "service"          # Single service
    CROSS_SERVICE = "cross_service"  # Multiple services
    SYSTEM = "system"            # Entire system


@dataclass
class ErrorPattern:
    """Common error pattern that can occur across languages."""
    pattern_id: str
    name: str
    category: ErrorCategory
    description: str
    common_causes: List[str]
    language_specific_manifestations: Dict[str, str]
    detection_rules: Dict[str, Any]
    fix_strategies: List[str]


@dataclass
class UnifiedError:
    """Unified representation of an error across languages."""
    error_id: str
    timestamp: datetime
    category: ErrorCategory
    severity: ErrorSeverity
    scope: ErrorScope
    original_language: str
    original_error_type: str
    message: str
    patterns: List[ErrorPattern]
    language_mappings: Dict[str, Dict[str, Any]]
    context: Dict[str, Any] = field(default_factory=dict)
    related_errors: List[str] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)


@dataclass
class ErrorMapping:
    """Maps language-specific errors to unified taxonomy."""
    language: str
    native_error_type: str
    unified_category: ErrorCategory
    severity_mapping: Dict[str, ErrorSeverity]
    pattern_ids: List[str]
    extraction_rules: Dict[str, Any]


class LanguageErrorMapper(ABC):
    """Abstract base class for language-specific error mappers."""
    
    @abstractmethod
    def map_to_unified(self, error: Dict[str, Any]) -> UnifiedError:
        """Map a language-specific error to unified format."""
        pass
        
    @abstractmethod
    def map_from_unified(self, unified_error: UnifiedError) -> Dict[str, Any]:
        """Map a unified error back to language-specific format."""
        pass
        
    @abstractmethod
    def get_error_patterns(self) -> List[ErrorPattern]:
        """Get language-specific error patterns."""
        pass


class UnifiedErrorTaxonomy:
    """
    Manages the unified error taxonomy across all supported languages.
    Provides mapping, classification, and pattern matching capabilities.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.error_patterns: Dict[str, ErrorPattern] = {}
        self.language_mappers: Dict[str, LanguageErrorMapper] = {}
        self.error_mappings: Dict[str, List[ErrorMapping]] = {}
        
        # Initialize common error patterns
        self._initialize_common_patterns()
        
        # Initialize language mappers
        self._initialize_language_mappers()
        
    def _initialize_common_patterns(self):
        """Initialize common error patterns that occur across languages."""
        
        # Null/Nil reference errors
        self.register_pattern(ErrorPattern(
            pattern_id="null_reference",
            name="Null Reference Error",
            category=ErrorCategory.RUNTIME,
            description="Attempting to access member of null/nil/undefined reference",
            common_causes=[
                "Uninitialized variable",
                "Failed object creation",
                "Missing null check",
                "Race condition"
            ],
            language_specific_manifestations={
                "python": "AttributeError: 'NoneType' object has no attribute",
                "javascript": "TypeError: Cannot read property of undefined",
                "java": "NullPointerException",
                "csharp": "NullReferenceException",
                "go": "panic: runtime error: invalid memory address",
                "rust": "Option::unwrap() on None",
                "ruby": "NoMethodError: undefined method for nil:NilClass",
                "php": "Fatal error: Call to a member function on null"
            },
            detection_rules={
                "keywords": ["null", "nil", "undefined", "NoneType", "NullPointer"],
                "patterns": [
                    r"'NoneType' object",
                    r"Cannot read property.*of undefined",
                    r"NullPointerException",
                    r"invalid memory address"
                ]
            },
            fix_strategies=[
                "Add null/nil checks before access",
                "Initialize variables properly",
                "Use optional/maybe types",
                "Implement proper error handling"
            ]
        ))
        
        # Type errors
        self.register_pattern(ErrorPattern(
            pattern_id="type_mismatch",
            name="Type Mismatch Error",
            category=ErrorCategory.RUNTIME,
            description="Operation on incompatible types",
            common_causes=[
                "Incorrect type assumption",
                "Missing type conversion",
                "Dynamic typing issues",
                "API contract violation"
            ],
            language_specific_manifestations={
                "python": "TypeError: unsupported operand type(s)",
                "javascript": "TypeError: X is not a function",
                "java": "ClassCastException",
                "typescript": "Type 'X' is not assignable to type 'Y'",
                "go": "cannot use X (type Y) as type Z",
                "rust": "mismatched types",
                "ruby": "TypeError: no implicit conversion",
                "php": "TypeError: Argument must be of type"
            },
            detection_rules={
                "keywords": ["TypeError", "type mismatch", "ClassCastException", "not assignable"],
                "patterns": [
                    r"unsupported operand type",
                    r"is not a function",
                    r"cannot use.*type.*as type",
                    r"mismatched types"
                ]
            },
            fix_strategies=[
                "Add explicit type conversion",
                "Validate input types",
                "Use type guards/checks",
                "Update type annotations"
            ]
        ))
        
        # Concurrency errors
        self.register_pattern(ErrorPattern(
            pattern_id="race_condition",
            name="Race Condition",
            category=ErrorCategory.CONCURRENCY,
            description="Concurrent access to shared resource without proper synchronization",
            common_causes=[
                "Missing locks/mutexes",
                "Incorrect synchronization",
                "Shared mutable state",
                "Timing dependencies"
            ],
            language_specific_manifestations={
                "python": "RuntimeError: dictionary changed size during iteration",
                "java": "ConcurrentModificationException",
                "go": "fatal error: concurrent map writes",
                "rust": "cannot borrow as mutable more than once",
                "csharp": "Collection was modified; enumeration operation may not execute",
                "javascript": "Possible race condition detected"
            },
            detection_rules={
                "keywords": ["concurrent", "race", "deadlock", "synchronization"],
                "patterns": [
                    r"concurrent.*modification",
                    r"changed size during iteration",
                    r"concurrent map",
                    r"cannot borrow.*mutable"
                ]
            },
            fix_strategies=[
                "Add proper synchronization",
                "Use thread-safe data structures",
                "Implement locking mechanisms",
                "Avoid shared mutable state"
            ]
        ))
        
        # Memory errors
        self.register_pattern(ErrorPattern(
            pattern_id="memory_leak",
            name="Memory Leak",
            category=ErrorCategory.MEMORY,
            description="Memory not properly released leading to exhaustion",
            common_causes=[
                "Circular references",
                "Unclosed resources",
                "Event listener accumulation",
                "Cache without limits"
            ],
            language_specific_manifestations={
                "python": "MemoryError",
                "javascript": "JavaScript heap out of memory",
                "java": "OutOfMemoryError: Java heap space",
                "go": "runtime: out of memory",
                "csharp": "OutOfMemoryException",
                "rust": "memory allocation failed",
                "c": "malloc: Cannot allocate memory",
                "cpp": "std::bad_alloc"
            },
            detection_rules={
                "keywords": ["OutOfMemory", "heap", "memory", "allocation failed"],
                "patterns": [
                    r"out of memory",
                    r"heap.*space",
                    r"Cannot allocate memory",
                    r"bad_alloc"
                ]
            },
            fix_strategies=[
                "Profile memory usage",
                "Release resources explicitly",
                "Implement resource pooling",
                "Add memory limits/monitoring"
            ]
        ))
        
        # API errors
        self.register_pattern(ErrorPattern(
            pattern_id="api_contract_violation",
            name="API Contract Violation",
            category=ErrorCategory.API,
            description="API called with invalid parameters or in wrong state",
            common_causes=[
                "Missing required parameters",
                "Invalid parameter types",
                "Exceeded rate limits",
                "Invalid authentication"
            ],
            language_specific_manifestations={
                "http": "400 Bad Request, 422 Unprocessable Entity",
                "graphql": "GraphQL validation error",
                "grpc": "INVALID_ARGUMENT",
                "rest": "Missing required field",
                "soap": "SOAP Fault: Client"
            },
            detection_rules={
                "keywords": ["Bad Request", "validation", "required", "invalid argument"],
                "patterns": [
                    r"400 Bad Request",
                    r"422.*Unprocessable",
                    r"Missing required",
                    r"INVALID_ARGUMENT"
                ]
            },
            fix_strategies=[
                "Validate inputs before API call",
                "Update API client to match contract",
                "Add request validation",
                "Implement retry with backoff"
            ]
        ))
        
        # Database errors
        self.register_pattern(ErrorPattern(
            pattern_id="database_constraint_violation",
            name="Database Constraint Violation",
            category=ErrorCategory.DATABASE,
            description="Database operation violates defined constraints",
            common_causes=[
                "Duplicate key",
                "Foreign key violation",
                "Not null constraint",
                "Check constraint failure"
            ],
            language_specific_manifestations={
                "sql": "UNIQUE constraint violated",
                "postgresql": "duplicate key value violates unique constraint",
                "mysql": "Duplicate entry for key",
                "mongodb": "E11000 duplicate key error",
                "orm": "IntegrityError"
            },
            detection_rules={
                "keywords": ["constraint", "duplicate", "foreign key", "integrity"],
                "patterns": [
                    r"constraint.*violat",
                    r"duplicate.*key",
                    r"foreign key",
                    r"IntegrityError"
                ]
            },
            fix_strategies=[
                "Check existence before insert",
                "Use upsert operations",
                "Handle constraint violations",
                "Validate data before DB operation"
            ]
        ))
        
    def _initialize_language_mappers(self):
        """Initialize language-specific error mappers."""
        # Register built-in mappers
        self.register_language_mapper('python', PythonErrorMapper())
        self.register_language_mapper('javascript', JavaScriptErrorMapper())
        self.register_language_mapper('java', JavaErrorMapper())
        self.register_language_mapper('go', GoErrorMapper())
        self.register_language_mapper('rust', RustErrorMapper())
        self.register_language_mapper('csharp', CSharpErrorMapper())
        self.register_language_mapper('ruby', RubyErrorMapper())
        self.register_language_mapper('php', PHPErrorMapper())
        
    def register_pattern(self, pattern: ErrorPattern) -> None:
        """Register a new error pattern."""
        self.error_patterns[pattern.pattern_id] = pattern
        self.logger.info(f"Registered error pattern: {pattern.name}")
        
    def register_language_mapper(self, language: str, mapper: LanguageErrorMapper) -> None:
        """Register a language-specific error mapper."""
        self.language_mappers[language] = mapper
        
        # Get patterns from mapper
        patterns = mapper.get_error_patterns()
        for pattern in patterns:
            self.register_pattern(pattern)
            
        self.logger.info(f"Registered {language} error mapper")
        
    def classify_error(self, error_data: Dict[str, Any], language: str) -> UnifiedError:
        """
        Classify an error from any language into the unified taxonomy.
        """
        if language not in self.language_mappers:
            # Fallback to generic classification
            return self._generic_classification(error_data, language)
            
        mapper = self.language_mappers[language]
        unified_error = mapper.map_to_unified(error_data)
        
        # Match against known patterns
        matched_patterns = self._match_patterns(unified_error)
        unified_error.patterns = matched_patterns
        
        # Add tags based on patterns
        for pattern in matched_patterns:
            unified_error.tags.add(pattern.pattern_id)
            unified_error.tags.add(pattern.category.value)
            
        return unified_error
        
    def _match_patterns(self, error: UnifiedError) -> List[ErrorPattern]:
        """Match error against known patterns."""
        matched = []
        
        for pattern in self.error_patterns.values():
            # Check category match
            if pattern.category != error.category:
                continue
                
            # Check detection rules
            if self._matches_detection_rules(error, pattern.detection_rules):
                matched.append(pattern)
                
        return matched
        
    def _matches_detection_rules(
        self, 
        error: UnifiedError, 
        rules: Dict[str, Any]
    ) -> bool:
        """Check if error matches detection rules."""
        # Check keywords
        keywords = rules.get('keywords', [])
        error_text = f"{error.message} {error.original_error_type}".lower()
        
        for keyword in keywords:
            if keyword.lower() in error_text:
                return True
                
        # Check patterns
        import re
        patterns = rules.get('patterns', [])
        for pattern in patterns:
            if re.search(pattern, error_text, re.IGNORECASE):
                return True
                
        return False
        
    def _generic_classification(
        self, 
        error_data: Dict[str, Any], 
        language: str
    ) -> UnifiedError:
        """Generic classification for unsupported languages."""
        # Extract basic information
        message = str(error_data.get('message', ''))
        error_type = str(error_data.get('type', 'UnknownError'))
        
        # Attempt to determine category
        category = self._infer_category(message, error_type)
        severity = self._infer_severity(message, error_type)
        
        return UnifiedError(
            error_id=f"generic_{language}_{datetime.now().timestamp()}",
            timestamp=datetime.now(),
            category=category,
            severity=severity,
            scope=ErrorScope.LOCAL,
            original_language=language,
            original_error_type=error_type,
            message=message,
            patterns=[],
            language_mappings={language: error_data}
        )
        
    def _infer_category(self, message: str, error_type: str) -> ErrorCategory:
        """Infer error category from message and type."""
        text = f"{message} {error_type}".lower()
        
        category_keywords = {
            ErrorCategory.SYNTAX: ['syntax', 'parse', 'unexpected token'],
            ErrorCategory.RUNTIME: ['runtime', 'exception', 'error'],
            ErrorCategory.NETWORK: ['network', 'connection', 'timeout', 'socket'],
            ErrorCategory.DATABASE: ['database', 'sql', 'query', 'transaction'],
            ErrorCategory.MEMORY: ['memory', 'heap', 'stack', 'overflow'],
            ErrorCategory.CONCURRENCY: ['thread', 'concurrent', 'lock', 'deadlock'],
            ErrorCategory.SECURITY: ['security', 'permission', 'unauthorized'],
            ErrorCategory.API: ['api', 'endpoint', 'request', 'response'],
        }
        
        for category, keywords in category_keywords.items():
            if any(keyword in text for keyword in keywords):
                return category
                
        return ErrorCategory.RUNTIME
        
    def _infer_severity(self, message: str, error_type: str) -> ErrorSeverity:
        """Infer error severity from message and type."""
        text = f"{message} {error_type}".lower()
        
        if any(word in text for word in ['fatal', 'critical', 'crash', 'panic']):
            return ErrorSeverity.CRITICAL
        elif any(word in text for word in ['error', 'exception', 'failure']):
            return ErrorSeverity.HIGH
        elif any(word in text for word in ['warning', 'deprecated']):
            return ErrorSeverity.WARNING
        else:
            return ErrorSeverity.MEDIUM
            
    def get_similar_errors(
        self, 
        error: UnifiedError, 
        language: Optional[str] = None
    ) -> List[Tuple[str, ErrorPattern]]:
        """
        Find similar errors that might occur in other languages.
        """
        similar = []
        
        for pattern in error.patterns:
            manifestations = pattern.language_specific_manifestations
            
            for lang, manifestation in manifestations.items():
                if language and lang != language:
                    continue
                if lang != error.original_language:
                    similar.append((lang, pattern))
                    
        return similar
        
    def get_fix_recommendations(
        self, 
        error: UnifiedError, 
        target_language: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get fix recommendations for an error, optionally in a specific language.
        """
        recommendations = []
        
        for pattern in error.patterns:
            for strategy in pattern.fix_strategies:
                rec = {
                    'strategy': strategy,
                    'pattern': pattern.name,
                    'confidence': 0.8 if pattern in error.patterns else 0.5
                }
                
                # Add language-specific implementation hints
                if target_language:
                    rec['language_specific'] = self._get_language_specific_fix(
                        pattern,
                        strategy,
                        target_language
                    )
                    
                recommendations.append(rec)
                
        return recommendations
        
    def _get_language_specific_fix(
        self, 
        pattern: ErrorPattern, 
        strategy: str, 
        language: str
    ) -> Dict[str, Any]:
        """Get language-specific implementation of a fix strategy."""
        # This would contain language-specific code templates
        language_fixes = {
            'python': {
                'Add null/nil checks before access': 
                    'if obj is not None:\n    obj.method()',
                'Add explicit type conversion': 
                    'int(value) or str(value) or float(value)',
                'Add proper synchronization': 
                    'with threading.Lock():\n    # critical section',
            },
            'javascript': {
                'Add null/nil checks before access': 
                    'if (obj !== null && obj !== undefined) {\n    obj.method();\n}',
                'Add explicit type conversion': 
                    'Number(value) or String(value) or Boolean(value)',
                'Add proper synchronization': 
                    '// Use async/await or Promise chains',
            },
            # Add more languages...
        }
        
        return language_fixes.get(language, {}).get(strategy, {})
        
    def export_taxonomy(self, format: str = 'json') -> str:
        """Export the error taxonomy in various formats."""
        if format == 'json':
            taxonomy = {
                'categories': [cat.value for cat in ErrorCategory],
                'severities': [sev.value for sev in ErrorSeverity],
                'scopes': [scope.value for scope in ErrorScope],
                'patterns': {
                    pattern_id: {
                        'name': pattern.name,
                        'category': pattern.category.value,
                        'description': pattern.description,
                        'common_causes': pattern.common_causes,
                        'languages': list(pattern.language_specific_manifestations.keys()),
                        'fix_strategies': pattern.fix_strategies
                    }
                    for pattern_id, pattern in self.error_patterns.items()
                },
                'supported_languages': list(self.language_mappers.keys())
            }
            return json.dumps(taxonomy, indent=2)
            
        elif format == 'markdown':
            lines = ['# Unified Error Taxonomy\n']
            
            lines.append('## Error Categories\n')
            for cat in ErrorCategory:
                lines.append(f'- **{cat.value}**: {cat.name}\n')
                
            lines.append('\n## Error Patterns\n')
            for pattern in self.error_patterns.values():
                lines.append(f'### {pattern.name}\n')
                lines.append(f'- **Category**: {pattern.category.value}\n')
                lines.append(f'- **Description**: {pattern.description}\n')
                lines.append('- **Common Causes**:\n')
                for cause in pattern.common_causes:
                    lines.append(f'  - {cause}\n')
                lines.append(f'- **Supported Languages**: {", ".join(pattern.language_specific_manifestations.keys())}\n')
                lines.append('\n')
                
            return ''.join(lines)
            
        else:
            raise ValueError(f"Unsupported export format: {format}")


# Language-specific mapper implementations

class PythonErrorMapper(LanguageErrorMapper):
    """Maps Python errors to unified taxonomy."""
    
    def map_to_unified(self, error: Dict[str, Any]) -> UnifiedError:
        error_type = error.get('type', 'Exception')
        message = error.get('message', '')
        
        # Map Python error types to categories
        category_map = {
            'SyntaxError': ErrorCategory.SYNTAX,
            'TypeError': ErrorCategory.RUNTIME,
            'ValueError': ErrorCategory.VALIDATION,
            'AttributeError': ErrorCategory.RUNTIME,
            'KeyError': ErrorCategory.RUNTIME,
            'IndexError': ErrorCategory.RUNTIME,
            'MemoryError': ErrorCategory.MEMORY,
            'ImportError': ErrorCategory.DEPENDENCY,
            'RuntimeError': ErrorCategory.RUNTIME,
            'OSError': ErrorCategory.IO,
            'ConnectionError': ErrorCategory.NETWORK,
        }
        
        category = category_map.get(error_type, ErrorCategory.RUNTIME)
        
        return UnifiedError(
            error_id=f"python_{datetime.now().timestamp()}",
            timestamp=datetime.now(),
            category=category,
            severity=ErrorSeverity.HIGH,
            scope=ErrorScope.LOCAL,
            original_language='python',
            original_error_type=error_type,
            message=message,
            patterns=[],
            language_mappings={'python': error}
        )
        
    def map_from_unified(self, unified_error: UnifiedError) -> Dict[str, Any]:
        return unified_error.language_mappings.get('python', {
            'type': unified_error.original_error_type,
            'message': unified_error.message
        })
        
    def get_error_patterns(self) -> List[ErrorPattern]:
        return []


class JavaScriptErrorMapper(LanguageErrorMapper):
    """Maps JavaScript errors to unified taxonomy."""
    
    def map_to_unified(self, error: Dict[str, Any]) -> UnifiedError:
        error_type = error.get('name', 'Error')
        message = error.get('message', '')
        
        category_map = {
            'SyntaxError': ErrorCategory.SYNTAX,
            'TypeError': ErrorCategory.RUNTIME,
            'ReferenceError': ErrorCategory.RUNTIME,
            'RangeError': ErrorCategory.VALIDATION,
            'NetworkError': ErrorCategory.NETWORK,
        }
        
        category = category_map.get(error_type, ErrorCategory.RUNTIME)
        
        return UnifiedError(
            error_id=f"javascript_{datetime.now().timestamp()}",
            timestamp=datetime.now(),
            category=category,
            severity=ErrorSeverity.HIGH,
            scope=ErrorScope.LOCAL,
            original_language='javascript',
            original_error_type=error_type,
            message=message,
            patterns=[],
            language_mappings={'javascript': error}
        )
        
    def map_from_unified(self, unified_error: UnifiedError) -> Dict[str, Any]:
        return unified_error.language_mappings.get('javascript', {
            'name': unified_error.original_error_type,
            'message': unified_error.message
        })
        
    def get_error_patterns(self) -> List[ErrorPattern]:
        return []


class JavaErrorMapper(LanguageErrorMapper):
    """Maps Java errors to unified taxonomy."""
    
    def map_to_unified(self, error: Dict[str, Any]) -> UnifiedError:
        error_type = error.get('exception', 'Exception')
        message = error.get('message', '')
        
        category_map = {
            'NullPointerException': ErrorCategory.RUNTIME,
            'ClassCastException': ErrorCategory.RUNTIME,
            'IllegalArgumentException': ErrorCategory.VALIDATION,
            'OutOfMemoryError': ErrorCategory.MEMORY,
            'StackOverflowError': ErrorCategory.MEMORY,
            'IOException': ErrorCategory.IO,
            'SQLException': ErrorCategory.DATABASE,
            'SecurityException': ErrorCategory.SECURITY,
            'ConcurrentModificationException': ErrorCategory.CONCURRENCY,
        }
        
        category = category_map.get(error_type, ErrorCategory.RUNTIME)
        
        return UnifiedError(
            error_id=f"java_{datetime.now().timestamp()}",
            timestamp=datetime.now(),
            category=category,
            severity=ErrorSeverity.HIGH,
            scope=ErrorScope.LOCAL,
            original_language='java',
            original_error_type=error_type,
            message=message,
            patterns=[],
            language_mappings={'java': error}
        )
        
    def map_from_unified(self, unified_error: UnifiedError) -> Dict[str, Any]:
        return unified_error.language_mappings.get('java', {
            'exception': unified_error.original_error_type,
            'message': unified_error.message
        })
        
    def get_error_patterns(self) -> List[ErrorPattern]:
        return []


class GoErrorMapper(LanguageErrorMapper):
    """Maps Go errors to unified taxonomy."""
    
    def map_to_unified(self, error: Dict[str, Any]) -> UnifiedError:
        error_msg = error.get('error', '')
        
        # Go doesn't have typed exceptions, analyze message
        category = ErrorCategory.RUNTIME
        if 'panic:' in error_msg:
            category = ErrorCategory.RUNTIME
        elif 'syntax error' in error_msg:
            category = ErrorCategory.SYNTAX
        elif 'undefined:' in error_msg:
            category = ErrorCategory.COMPILATION
        elif 'cannot find package' in error_msg:
            category = ErrorCategory.DEPENDENCY
            
        return UnifiedError(
            error_id=f"go_{datetime.now().timestamp()}",
            timestamp=datetime.now(),
            category=category,
            severity=ErrorSeverity.HIGH,
            scope=ErrorScope.LOCAL,
            original_language='go',
            original_error_type='error',
            message=error_msg,
            patterns=[],
            language_mappings={'go': error}
        )
        
    def map_from_unified(self, unified_error: UnifiedError) -> Dict[str, Any]:
        return unified_error.language_mappings.get('go', {
            'error': unified_error.message
        })
        
    def get_error_patterns(self) -> List[ErrorPattern]:
        return []


class RustErrorMapper(LanguageErrorMapper):
    """Maps Rust errors to unified taxonomy."""
    
    def map_to_unified(self, error: Dict[str, Any]) -> UnifiedError:
        error_type = error.get('error_type', 'Error')
        message = error.get('message', '')
        
        category = ErrorCategory.COMPILATION  # Rust catches most at compile time
        if 'panic' in message:
            category = ErrorCategory.RUNTIME
        elif 'borrow' in message:
            category = ErrorCategory.MEMORY
            
        return UnifiedError(
            error_id=f"rust_{datetime.now().timestamp()}",
            timestamp=datetime.now(),
            category=category,
            severity=ErrorSeverity.HIGH,
            scope=ErrorScope.LOCAL,
            original_language='rust',
            original_error_type=error_type,
            message=message,
            patterns=[],
            language_mappings={'rust': error}
        )
        
    def map_from_unified(self, unified_error: UnifiedError) -> Dict[str, Any]:
        return unified_error.language_mappings.get('rust', {
            'error_type': unified_error.original_error_type,
            'message': unified_error.message
        })
        
    def get_error_patterns(self) -> List[ErrorPattern]:
        return []


class CSharpErrorMapper(LanguageErrorMapper):
    """Maps C# errors to unified taxonomy."""
    
    def map_to_unified(self, error: Dict[str, Any]) -> UnifiedError:
        error_type = error.get('exception', 'Exception')
        message = error.get('message', '')
        
        category_map = {
            'NullReferenceException': ErrorCategory.RUNTIME,
            'InvalidCastException': ErrorCategory.RUNTIME,
            'ArgumentException': ErrorCategory.VALIDATION,
            'OutOfMemoryException': ErrorCategory.MEMORY,
            'StackOverflowException': ErrorCategory.MEMORY,
            'IOException': ErrorCategory.IO,
            'SqlException': ErrorCategory.DATABASE,
            'UnauthorizedAccessException': ErrorCategory.SECURITY,
        }
        
        category = category_map.get(error_type, ErrorCategory.RUNTIME)
        
        return UnifiedError(
            error_id=f"csharp_{datetime.now().timestamp()}",
            timestamp=datetime.now(),
            category=category,
            severity=ErrorSeverity.HIGH,
            scope=ErrorScope.LOCAL,
            original_language='csharp',
            original_error_type=error_type,
            message=message,
            patterns=[],
            language_mappings={'csharp': error}
        )
        
    def map_from_unified(self, unified_error: UnifiedError) -> Dict[str, Any]:
        return unified_error.language_mappings.get('csharp', {
            'exception': unified_error.original_error_type,
            'message': unified_error.message
        })
        
    def get_error_patterns(self) -> List[ErrorPattern]:
        return []


class RubyErrorMapper(LanguageErrorMapper):
    """Maps Ruby errors to unified taxonomy."""
    
    def map_to_unified(self, error: Dict[str, Any]) -> UnifiedError:
        error_type = error.get('class', 'StandardError')
        message = error.get('message', '')
        
        category_map = {
            'SyntaxError': ErrorCategory.SYNTAX,
            'TypeError': ErrorCategory.RUNTIME,
            'NoMethodError': ErrorCategory.RUNTIME,
            'ArgumentError': ErrorCategory.VALIDATION,
            'LoadError': ErrorCategory.DEPENDENCY,
            'SecurityError': ErrorCategory.SECURITY,
        }
        
        category = category_map.get(error_type, ErrorCategory.RUNTIME)
        
        return UnifiedError(
            error_id=f"ruby_{datetime.now().timestamp()}",
            timestamp=datetime.now(),
            category=category,
            severity=ErrorSeverity.HIGH,
            scope=ErrorScope.LOCAL,
            original_language='ruby',
            original_error_type=error_type,
            message=message,
            patterns=[],
            language_mappings={'ruby': error}
        )
        
    def map_from_unified(self, unified_error: UnifiedError) -> Dict[str, Any]:
        return unified_error.language_mappings.get('ruby', {
            'class': unified_error.original_error_type,
            'message': unified_error.message
        })
        
    def get_error_patterns(self) -> List[ErrorPattern]:
        return []


class PHPErrorMapper(LanguageErrorMapper):
    """Maps PHP errors to unified taxonomy."""
    
    def map_to_unified(self, error: Dict[str, Any]) -> UnifiedError:
        error_type = error.get('type', 'Error')
        message = error.get('message', '')
        
        category_map = {
            'ParseError': ErrorCategory.SYNTAX,
            'TypeError': ErrorCategory.RUNTIME,
            'Error': ErrorCategory.RUNTIME,
            'Exception': ErrorCategory.RUNTIME,
            'PDOException': ErrorCategory.DATABASE,
        }
        
        category = category_map.get(error_type, ErrorCategory.RUNTIME)
        
        return UnifiedError(
            error_id=f"php_{datetime.now().timestamp()}",
            timestamp=datetime.now(),
            category=category,
            severity=ErrorSeverity.HIGH,
            scope=ErrorScope.LOCAL,
            original_language='php',
            original_error_type=error_type,
            message=message,
            patterns=[],
            language_mappings={'php': error}
        )
        
    def map_from_unified(self, unified_error: UnifiedError) -> Dict[str, Any]:
        return unified_error.language_mappings.get('php', {
            'type': unified_error.original_error_type,
            'message': unified_error.message
        })
        
    def get_error_patterns(self) -> List[ErrorPattern]:
        return []