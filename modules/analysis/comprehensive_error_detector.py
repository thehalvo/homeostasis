"""
Comprehensive Error Detection and Classification System for LLM Integration.

This module provides expanded error detection capabilities that goes beyond HTTP/rate-limit
errors to detect syntax errors, compilation failures, runtime exceptions, configuration
inconsistencies, and logical bugs across multiple languages and frameworks.
"""

import os
import re
import ast
import json
import logging
import subprocess
import tempfile
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union, Set
from dataclasses import dataclass, asdict
from collections import defaultdict

# Configure logging
logger = logging.getLogger(__name__)


class ErrorCategory(Enum):
    """Comprehensive error categories for classification."""
    SYNTAX = "syntax"
    COMPILATION = "compilation"
    RUNTIME = "runtime"
    CONFIGURATION = "configuration"
    LOGIC = "logic"
    ENVIRONMENT = "environment"
    CONCURRENCY = "concurrency"
    SECURITY = "security"
    PERFORMANCE = "performance"
    DEPENDENCY = "dependency"
    NETWORK = "network"
    DATABASE = "database"
    FILESYSTEM = "filesystem"
    MEMORY = "memory"
    UNKNOWN = "unknown"


class ErrorSeverity(Enum):
    """Error severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class LanguageType(Enum):
    """Supported programming languages."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    GO = "go"
    RUST = "rust"
    CSHARP = "csharp"
    RUBY = "ruby"
    PHP = "php"
    SWIFT = "swift"
    KOTLIN = "kotlin"
    SCALA = "scala"
    CLOJURE = "clojure"
    ELIXIR = "elixir"
    DART = "dart"
    CPP = "cpp"
    C = "c"
    # Additional languages from Phase 12.A
    ZIG = "zig"
    NIM = "nim"
    CRYSTAL = "crystal"
    HASKELL = "haskell"
    FSHARP = "fsharp"
    ERLANG = "erlang"
    SQL = "sql"
    BASH = "bash"
    POWERSHELL = "powershell"
    LUA = "lua"
    R = "r"
    MATLAB = "matlab"
    JULIA = "julia"
    TERRAFORM = "terraform"
    ANSIBLE = "ansible"
    YAML = "yaml"
    JSON = "json"
    DOCKERFILE = "dockerfile"
    UNKNOWN = "unknown"


@dataclass
class ErrorContext:
    """Comprehensive error context information."""
    
    # Basic error information
    error_message: str
    exception_type: Optional[str] = None
    stack_trace: Optional[List[str]] = None
    
    # Source code context
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    function_name: Optional[str] = None
    source_code_snippet: Optional[str] = None
    
    # Language and framework context
    language: LanguageType = LanguageType.UNKNOWN
    framework: Optional[str] = None
    
    # Environment context
    os_type: Optional[str] = None
    python_version: Optional[str] = None
    node_version: Optional[str] = None
    java_version: Optional[str] = None
    
    # Dependency context
    dependencies: Optional[Dict[str, str]] = None
    package_manager: Optional[str] = None
    
    # Configuration context
    config_files: Optional[List[str]] = None
    environment_variables: Optional[Dict[str, str]] = None
    
    # Project structure context
    project_type: Optional[str] = None
    build_system: Optional[str] = None
    
    # Additional metadata
    timestamp: Optional[str] = None
    service_name: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        # Convert enums to string values
        result['language'] = self.language.value
        return result


@dataclass
class ErrorClassification:
    """Result of error classification."""
    
    category: ErrorCategory
    severity: ErrorSeverity
    confidence: float
    description: str
    root_cause: str
    suggestions: List[str]
    
    # Additional classification details
    subcategory: Optional[str] = None
    affected_components: Optional[List[str]] = None
    potential_fix_complexity: Optional[str] = None  # simple, moderate, complex
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result['category'] = self.category.value
        result['severity'] = self.severity.value
        return result


class LanguageSpecificParser:
    """Base class for language-specific parsers."""
    
    def __init__(self, language: LanguageType):
        self.language = language
    
    def parse_syntax_error(self, error_message: str, source_code: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Parse syntax errors specific to the language."""
        raise NotImplementedError
    
    def parse_compilation_error(self, error_message: str, source_code: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Parse compilation errors specific to the language."""
        raise NotImplementedError
    
    def detect_runtime_issues(self, error_context: ErrorContext) -> List[Dict[str, Any]]:
        """Detect runtime issues specific to the language."""
        raise NotImplementedError


class PythonParser(LanguageSpecificParser):
    """Python-specific error parser."""
    
    def __init__(self):
        super().__init__(LanguageType.PYTHON)
        
        # Python syntax error patterns
        self.syntax_patterns = [
            (r"SyntaxError: invalid syntax", "invalid_syntax"),
            (r"SyntaxError: unexpected EOF while parsing", "unexpected_eof"),
            (r"SyntaxError: unmatched '\)'", "unmatched_parenthesis"),
            (r"SyntaxError: unmatched '\]'", "unmatched_bracket"),
            (r"SyntaxError: unmatched '\}'", "unmatched_brace"),
            (r"IndentationError: expected an indented block", "missing_indentation"),
            (r"IndentationError: unindent does not match any outer indentation level", "inconsistent_indentation"),
            (r"TabError: inconsistent use of tabs and spaces", "mixed_tabs_spaces"),
        ]
        
        # Python runtime error patterns
        self.runtime_patterns = [
            (r"NameError: name '([^']+)' is not defined", "undefined_variable"),
            (r"AttributeError: '([^']+)' object has no attribute '([^']+)'", "missing_attribute"),
            (r"KeyError: '([^']+)'", "missing_key"),
            (r"IndexError: list index out of range", "index_out_of_bounds"),
            (r"TypeError: '([^']+)' object is not (callable|subscriptable|iterable)", "type_error"),
            (r"ValueError: invalid literal for (\w+)\(\) with base (\d+): '([^']+)'", "conversion_error"),
            (r"ZeroDivisionError: division by zero", "division_by_zero"),
            (r"FileNotFoundError: \[Errno 2\] No such file or directory: '([^']+)'", "file_not_found"),
            (r"ImportError: No module named '([^']+)'", "missing_module"),
            (r"ModuleNotFoundError: No module named '([^']+)'", "missing_module"),
        ]
    
    def parse_syntax_error(self, error_message: str, source_code: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Parse Python syntax errors."""
        for pattern, error_type in self.syntax_patterns:
            match = re.search(pattern, error_message)
            if match:
                return {
                    "error_type": error_type,
                    "pattern": pattern,
                    "match_groups": match.groups(),
                    "category": ErrorCategory.SYNTAX,
                    "language": self.language
                }
        
        # Try to parse using AST if source code is available
        if source_code:
            try:
                ast.parse(source_code)
            except SyntaxError as e:
                return {
                    "error_type": "syntax_error",
                    "ast_error": str(e),
                    "line_number": e.lineno,
                    "offset": e.offset,
                    "category": ErrorCategory.SYNTAX,
                    "language": self.language
                }
        
        return None
    
    def parse_compilation_error(self, error_message: str, source_code: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Python doesn't have traditional compilation errors, but can have import-time errors."""
        # Check for import errors that occur during "compilation" phase
        if "ImportError" in error_message or "ModuleNotFoundError" in error_message:
            for pattern, error_type in self.runtime_patterns:
                if "ImportError" in pattern or "ModuleNotFoundError" in pattern:
                    match = re.search(pattern, error_message)
                    if match:
                        return {
                            "error_type": error_type,
                            "pattern": pattern,
                            "match_groups": match.groups(),
                            "category": ErrorCategory.DEPENDENCY,
                            "language": self.language
                        }
        
        return None
    
    def detect_runtime_issues(self, error_context: ErrorContext) -> List[Dict[str, Any]]:
        """Detect Python runtime issues."""
        issues = []
        
        for pattern, error_type in self.runtime_patterns:
            match = re.search(pattern, error_context.error_message)
            if match:
                issues.append({
                    "error_type": error_type,
                    "pattern": pattern,
                    "match_groups": match.groups(),
                    "category": self._categorize_runtime_error(error_type),
                    "language": self.language
                })
        
        return issues
    
    def _categorize_runtime_error(self, error_type: str) -> ErrorCategory:
        """Categorize Python runtime errors."""
        category_map = {
            "undefined_variable": ErrorCategory.LOGIC,
            "missing_attribute": ErrorCategory.LOGIC,
            "missing_key": ErrorCategory.LOGIC,
            "index_out_of_bounds": ErrorCategory.LOGIC,
            "type_error": ErrorCategory.LOGIC,
            "conversion_error": ErrorCategory.LOGIC,
            "division_by_zero": ErrorCategory.LOGIC,
            "file_not_found": ErrorCategory.FILESYSTEM,
            "missing_module": ErrorCategory.DEPENDENCY,
        }
        return category_map.get(error_type, ErrorCategory.RUNTIME)


class JavaScriptParser(LanguageSpecificParser):
    """JavaScript-specific error parser."""
    
    def __init__(self):
        super().__init__(LanguageType.JAVASCRIPT)
        
        # JavaScript syntax error patterns
        self.syntax_patterns = [
            (r"SyntaxError: Unexpected token '([^']+)'", "unexpected_token"),
            (r"SyntaxError: Unexpected end of input", "unexpected_eof"),
            (r"SyntaxError: Missing \) after argument list", "missing_parenthesis"),
            (r"SyntaxError: Unexpected identifier", "unexpected_identifier"),
            (r"SyntaxError: Invalid left-hand side in assignment", "invalid_assignment"),
        ]
        
        # JavaScript runtime error patterns
        self.runtime_patterns = [
            (r"ReferenceError: ([^']+) is not defined", "undefined_variable"),
            (r"TypeError: Cannot read propert(?:y|ies) '([^']+)' of (null|undefined)", "null_property_access"),
            (r"TypeError: ([^']+) is not a function", "not_a_function"),
            (r"TypeError: Cannot set propert(?:y|ies) '([^']+)' of (null|undefined)", "null_property_set"),
            (r"RangeError: Maximum call stack size exceeded", "stack_overflow"),
            (r"Error: Cannot find module '([^']+)'", "missing_module"),
        ]
    
    def parse_syntax_error(self, error_message: str, source_code: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Parse JavaScript syntax errors."""
        for pattern, error_type in self.syntax_patterns:
            match = re.search(pattern, error_message)
            if match:
                return {
                    "error_type": error_type,
                    "pattern": pattern,
                    "match_groups": match.groups(),
                    "category": ErrorCategory.SYNTAX,
                    "language": self.language
                }
        return None
    
    def parse_compilation_error(self, error_message: str, source_code: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """JavaScript compilation errors (from transpilers like TypeScript, Babel)."""
        # TypeScript compilation errors
        if "TS" in error_message and re.search(r"TS\d+:", error_message):
            return {
                "error_type": "typescript_compilation",
                "category": ErrorCategory.COMPILATION,
                "language": LanguageType.TYPESCRIPT
            }
        
        return None
    
    def detect_runtime_issues(self, error_context: ErrorContext) -> List[Dict[str, Any]]:
        """Detect JavaScript runtime issues."""
        issues = []
        
        for pattern, error_type in self.runtime_patterns:
            match = re.search(pattern, error_context.error_message)
            if match:
                issues.append({
                    "error_type": error_type,
                    "pattern": pattern,
                    "match_groups": match.groups(),
                    "category": self._categorize_runtime_error(error_type),
                    "language": self.language
                })
        
        return issues
    
    def _categorize_runtime_error(self, error_type: str) -> ErrorCategory:
        """Categorize JavaScript runtime errors."""
        category_map = {
            "undefined_variable": ErrorCategory.LOGIC,
            "null_property_access": ErrorCategory.LOGIC,
            "not_a_function": ErrorCategory.LOGIC,
            "null_property_set": ErrorCategory.LOGIC,
            "stack_overflow": ErrorCategory.LOGIC,
            "missing_module": ErrorCategory.DEPENDENCY,
        }
        return category_map.get(error_type, ErrorCategory.RUNTIME)


class ComprehensiveErrorDetector:
    """
    Comprehensive error detection and classification system.
    
    This detector goes beyond simple HTTP/rate-limit errors to detect:
    - Syntax errors across multiple languages
    - Compilation failures
    - Runtime exceptions
    - Configuration inconsistencies
    - Logical bugs
    - Environment issues
    - Concurrency problems
    - Security vulnerabilities
    """
    
    def __init__(self):
        """Initialize the comprehensive error detector."""
        
        # Initialize language-specific parsers
        self.parsers = {
            LanguageType.PYTHON: PythonParser(),
            LanguageType.JAVASCRIPT: JavaScriptParser(),
            # TODO: Add more language parsers
        }
        
        # Configuration error patterns
        self.config_patterns = [
            (r"(?i)configuration.*error", "config_error"),
            (r"(?i)missing.*configuration", "missing_config"),
            (r"(?i)invalid.*configuration", "invalid_config"),
            (r"(?i)environment.*variable.*not.*set", "missing_env_var"),
            (r"(?i)database.*connection.*failed", "db_connection_error"),
            (r"(?i)redis.*connection.*failed", "redis_connection_error"),
        ]
        
        # Environment error patterns
        self.environment_patterns = [
            (r"(?i)permission.*denied", "permission_error"),
            (r"(?i)disk.*space", "disk_space_error"),
            (r"(?i)memory.*error", "memory_error"),
            (r"(?i)timeout", "timeout_error"),
            (r"(?i)network.*unreachable", "network_error"),
            (r"(?i)port.*already.*in.*use", "port_conflict"),
        ]
        
        # Concurrency error patterns
        self.concurrency_patterns = [
            (r"(?i)deadlock", "deadlock"),
            (r"(?i)race.*condition", "race_condition"),
            (r"(?i)concurrent.*modification", "concurrent_modification"),
            (r"(?i)thread.*safety", "thread_safety"),
            (r"(?i)atomic.*operation.*failed", "atomic_operation_failed"),
        ]
        
        # Security error patterns
        self.security_patterns = [
            (r"(?i)sql.*injection", "sql_injection"),
            (r"(?i)cross.*site.*scripting", "xss"),
            (r"(?i)unauthorized.*access", "unauthorized_access"),
            (r"(?i)authentication.*failed", "auth_failed"),
            (r"(?i)certificate.*error", "certificate_error"),
            (r"(?i)insecure.*connection", "insecure_connection"),
        ]
    
    def detect_language(self, error_context: ErrorContext) -> LanguageType:
        """
        Detect the programming language from error context.
        
        Args:
            error_context: Error context information
            
        Returns:
            Detected language type
        """
        # Check explicit language setting
        if error_context.language != LanguageType.UNKNOWN:
            return error_context.language
        
        # Detect from file extension
        if error_context.file_path:
            file_path = Path(error_context.file_path)
            extension = file_path.suffix.lower()
            
            extension_map = {
                '.py': LanguageType.PYTHON,
                '.js': LanguageType.JAVASCRIPT,
                '.ts': LanguageType.TYPESCRIPT,
                '.java': LanguageType.JAVA,
                '.go': LanguageType.GO,
                '.rs': LanguageType.RUST,
                '.cs': LanguageType.CSHARP,
                '.rb': LanguageType.RUBY,
                '.php': LanguageType.PHP,
                '.swift': LanguageType.SWIFT,
                '.kt': LanguageType.KOTLIN,
                '.scala': LanguageType.SCALA,
                '.clj': LanguageType.CLOJURE,
                '.ex': LanguageType.ELIXIR,
                '.dart': LanguageType.DART,
                '.cpp': LanguageType.CPP,
                '.c': LanguageType.C,
            }
            
            if extension in extension_map:
                return extension_map[extension]
        
        # Detect from error message patterns
        error_msg = error_context.error_message.lower()
        
        if any(pattern in error_msg for pattern in ['traceback', 'nameerror', 'attributeerror', 'keyerror']):
            return LanguageType.PYTHON
        elif any(pattern in error_msg for pattern in ['referenceerror', 'typeerror', 'syntaxerror']):
            return LanguageType.JAVASCRIPT
        elif 'nullpointerexception' in error_msg or 'java.' in error_msg:
            return LanguageType.JAVA
        elif 'panic:' in error_msg or 'goroutine' in error_msg:
            return LanguageType.GO
        elif any(pattern in error_msg for pattern in ['fluttererror', 'nosuchmethoderror', 'rangeerror', 'formatexception']):
            return LanguageType.DART
        elif any(pattern in error_msg for pattern in ['xamarin', 'mono-rt', 'foundation.monotouch']):
            return LanguageType.CSHARP  # Xamarin
        elif any(pattern in error_msg for pattern in ['unity', 'unityengine', 'missingreferenceexception']):
            return LanguageType.CSHARP  # Unity
        elif any(pattern in error_msg for pattern in ['react native', 'metro bundler', 'red box', 'bridge module']):
            return LanguageType.JAVASCRIPT  # React Native
        
        return LanguageType.UNKNOWN
    
    def extract_context_from_environment(self) -> Dict[str, Any]:
        """
        Extract environment context information.
        
        Returns:
            Environment context dictionary
        """
        context = {}
        
        # OS information
        context['os_type'] = os.name
        
        # Python version
        try:
            import sys
            context['python_version'] = sys.version
        except:
            pass
        
        # Node.js version
        try:
            result = subprocess.run(['node', '--version'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                context['node_version'] = result.stdout.strip()
        except:
            pass
        
        # Java version
        try:
            result = subprocess.run(['java', '-version'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                context['java_version'] = result.stderr.strip()
        except:
            pass
        
        # Environment variables (selective)
        context['environment_variables'] = {
            key: os.environ.get(key, '')
            for key in ['PATH', 'PYTHONPATH', 'NODE_ENV', 'JAVA_HOME']
            if key in os.environ
        }
        
        return context
    
    def detect_dependencies(self, project_root: Optional[str] = None) -> Dict[str, Any]:
        """
        Detect project dependencies from various package managers.
        
        Args:
            project_root: Root directory of the project
            
        Returns:
            Dependencies information
        """
        dependencies = {}
        
        if not project_root:
            project_root = os.getcwd()
        
        project_path = Path(project_root)
        
        # Python dependencies
        requirements_txt = project_path / 'requirements.txt'
        pyproject_toml = project_path / 'pyproject.toml'
        pipfile = project_path / 'Pipfile'
        
        if requirements_txt.exists():
            dependencies['python_requirements'] = self._parse_requirements_txt(requirements_txt)
        elif pyproject_toml.exists():
            dependencies['python_pyproject'] = self._parse_pyproject_toml(pyproject_toml)
        elif pipfile.exists():
            dependencies['python_pipfile'] = 'detected'
        
        # Node.js dependencies
        package_json = project_path / 'package.json'
        if package_json.exists():
            dependencies['nodejs_package'] = self._parse_package_json(package_json)
        
        # Java dependencies
        pom_xml = project_path / 'pom.xml'
        build_gradle = project_path / 'build.gradle'
        
        if pom_xml.exists():
            dependencies['java_maven'] = 'detected'
        elif build_gradle.exists():
            dependencies['java_gradle'] = 'detected'
        
        return dependencies
    
    def _parse_requirements_txt(self, file_path: Path) -> List[str]:
        """Parse Python requirements.txt file."""
        try:
            with open(file_path, 'r') as f:
                return [line.strip() for line in f if line.strip() and not line.startswith('#')]
        except Exception as e:
            logger.warning(f"Error parsing requirements.txt: {e}")
            return []
    
    def _parse_pyproject_toml(self, file_path: Path) -> str:
        """Parse Python pyproject.toml file."""
        # For now, just return that it was detected
        # Could be enhanced to actually parse TOML content
        return 'detected'
    
    def _parse_package_json(self, file_path: Path) -> Dict[str, Any]:
        """Parse Node.js package.json file."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                return {
                    'dependencies': data.get('dependencies', {}),
                    'devDependencies': data.get('devDependencies', {}),
                    'name': data.get('name', ''),
                    'version': data.get('version', '')
                }
        except Exception as e:
            logger.warning(f"Error parsing package.json: {e}")
            return {}
    
    def classify_error(self, error_context: ErrorContext) -> ErrorClassification:
        """
        Classify an error based on comprehensive analysis.
        
        Args:
            error_context: Error context information
            
        Returns:
            Error classification result
        """
        # Detect language if not specified
        language = self.detect_language(error_context)
        error_context.language = language
        
        # Initialize classification with defaults
        category = ErrorCategory.UNKNOWN
        severity = ErrorSeverity.MEDIUM
        confidence = 0.0
        description = "Unknown error"
        root_cause = "unknown"
        suggestions = ["Manual investigation required"]
        subcategory = None
        
        # Try language-specific parsing first
        if language in self.parsers:
            parser = self.parsers[language]
            
            # Check for syntax errors
            syntax_result = parser.parse_syntax_error(error_context.error_message, error_context.source_code_snippet)
            if syntax_result:
                category = syntax_result["category"]
                severity = ErrorSeverity.HIGH
                confidence = 0.9
                description = f"Syntax error in {language.value} code"
                root_cause = syntax_result["error_type"]
                suggestions = self._get_syntax_suggestions(syntax_result["error_type"], language)
                subcategory = syntax_result["error_type"]
            
            # Check for compilation errors
            if category == ErrorCategory.UNKNOWN:
                compilation_result = parser.parse_compilation_error(error_context.error_message, error_context.source_code_snippet)
                if compilation_result:
                    category = compilation_result["category"]
                    severity = ErrorSeverity.HIGH
                    confidence = 0.85
                    description = f"Compilation error in {language.value} code"
                    root_cause = compilation_result["error_type"]
                    suggestions = self._get_compilation_suggestions(compilation_result["error_type"], language)
                    subcategory = compilation_result["error_type"]
            
            # Check for runtime issues
            if category == ErrorCategory.UNKNOWN:
                runtime_issues = parser.detect_runtime_issues(error_context)
                if runtime_issues:
                    issue = runtime_issues[0]  # Take the first match
                    category = issue["category"]
                    severity = ErrorSeverity.MEDIUM
                    confidence = 0.8
                    description = f"Runtime error in {language.value} code"
                    root_cause = issue["error_type"]
                    suggestions = self._get_runtime_suggestions(issue["error_type"], language)
                    subcategory = issue["error_type"]
        
        # Check for configuration errors
        if category == ErrorCategory.UNKNOWN:
            for pattern, error_type in self.config_patterns:
                if re.search(pattern, error_context.error_message):
                    category = ErrorCategory.CONFIGURATION
                    severity = ErrorSeverity.HIGH
                    confidence = 0.75
                    description = "Configuration error detected"
                    root_cause = error_type
                    suggestions = self._get_config_suggestions(error_type)
                    subcategory = error_type
                    break
        
        # Check for environment errors
        if category == ErrorCategory.UNKNOWN:
            for pattern, error_type in self.environment_patterns:
                if re.search(pattern, error_context.error_message):
                    category = ErrorCategory.ENVIRONMENT
                    severity = ErrorSeverity.HIGH
                    confidence = 0.7
                    description = "Environment error detected"
                    root_cause = error_type
                    suggestions = self._get_environment_suggestions(error_type)
                    subcategory = error_type
                    break
        
        # Check for concurrency errors
        if category == ErrorCategory.UNKNOWN:
            for pattern, error_type in self.concurrency_patterns:
                if re.search(pattern, error_context.error_message):
                    category = ErrorCategory.CONCURRENCY
                    severity = ErrorSeverity.CRITICAL
                    confidence = 0.8
                    description = "Concurrency error detected"
                    root_cause = error_type
                    suggestions = self._get_concurrency_suggestions(error_type)
                    subcategory = error_type
                    break
        
        # Check for security errors
        if category == ErrorCategory.UNKNOWN:
            for pattern, error_type in self.security_patterns:
                if re.search(pattern, error_context.error_message):
                    category = ErrorCategory.SECURITY
                    severity = ErrorSeverity.CRITICAL
                    confidence = 0.85
                    description = "Security vulnerability detected"
                    root_cause = error_type
                    suggestions = self._get_security_suggestions(error_type)
                    subcategory = error_type
                    break
        
        # Determine affected components
        affected_components = self._identify_affected_components(error_context)
        
        # Determine fix complexity
        fix_complexity = self._estimate_fix_complexity(category, root_cause)
        
        return ErrorClassification(
            category=category,
            severity=severity,
            confidence=confidence,
            description=description,
            root_cause=root_cause,
            suggestions=suggestions,
            subcategory=subcategory,
            affected_components=affected_components,
            potential_fix_complexity=fix_complexity
        )
    
    def _get_syntax_suggestions(self, error_type: str, language: LanguageType) -> List[str]:
        """Get suggestions for syntax errors."""
        suggestions_map = {
            "invalid_syntax": [
                "Check for missing colons, commas, or parentheses",
                "Verify proper indentation",
                "Look for unclosed quotes or brackets"
            ],
            "unexpected_eof": [
                "Check for unclosed parentheses, brackets, or braces",
                "Ensure all code blocks are properly closed"
            ],
            "missing_indentation": [
                "Add proper indentation after colons (:)",
                "Use consistent indentation (4 spaces recommended for Python)"
            ],
            "unexpected_token": [
                "Check for missing semicolons or commas",
                "Verify syntax is correct for the language version"
            ]
        }
        
        return suggestions_map.get(error_type, ["Review syntax documentation for " + language.value])
    
    def _get_compilation_suggestions(self, error_type: str, language: LanguageType) -> List[str]:
        """Get suggestions for compilation errors."""
        suggestions_map = {
            "missing_module": [
                "Install the missing dependency",
                "Check import/require statements",
                "Verify module name spelling"
            ],
            "typescript_compilation": [
                "Check TypeScript configuration",
                "Fix type annotations",
                "Install missing type definitions"
            ]
        }
        
        return suggestions_map.get(error_type, ["Check compilation configuration"])
    
    def _get_runtime_suggestions(self, error_type: str, language: LanguageType) -> List[str]:
        """Get suggestions for runtime errors."""
        suggestions_map = {
            "undefined_variable": [
                "Check variable name spelling",
                "Ensure variable is declared before use",
                "Check variable scope"
            ],
            "missing_attribute": [
                "Verify object has the expected attribute",
                "Check for typos in attribute name",
                "Add proper error handling"
            ],
            "missing_key": [
                "Check if dictionary key exists before accessing",
                "Use get() method with default value",
                "Add proper error handling"
            ],
            "null_property_access": [
                "Check for null/undefined values before property access",
                "Add null checks or optional chaining",
                "Initialize variables properly"
            ]
        }
        
        return suggestions_map.get(error_type, ["Add proper error handling"])
    
    def _get_config_suggestions(self, error_type: str) -> List[str]:
        """Get suggestions for configuration errors."""
        suggestions_map = {
            "missing_config": [
                "Create required configuration files",
                "Set missing configuration parameters",
                "Check configuration file paths"
            ],
            "invalid_config": [
                "Validate configuration syntax",
                "Check configuration values",
                "Review configuration documentation"
            ],
            "missing_env_var": [
                "Set required environment variables",
                "Check .env file",
                "Verify environment variable names"
            ]
        }
        
        return suggestions_map.get(error_type, ["Review configuration documentation"])
    
    def _get_environment_suggestions(self, error_type: str) -> List[str]:
        """Get suggestions for environment errors."""
        suggestions_map = {
            "permission_error": [
                "Check file/directory permissions",
                "Run with appropriate privileges",
                "Verify user access rights"
            ],
            "disk_space_error": [
                "Free up disk space",
                "Check available storage",
                "Clean temporary files"
            ],
            "memory_error": [
                "Increase available memory",
                "Optimize memory usage",
                "Check for memory leaks"
            ],
            "timeout_error": [
                "Increase timeout values",
                "Check network connectivity",
                "Optimize operation performance"
            ]
        }
        
        return suggestions_map.get(error_type, ["Check environment settings"])
    
    def _get_concurrency_suggestions(self, error_type: str) -> List[str]:
        """Get suggestions for concurrency errors."""
        suggestions_map = {
            "deadlock": [
                "Review locking order",
                "Implement timeout mechanisms",
                "Use deadlock detection tools"
            ],
            "race_condition": [
                "Add proper synchronization",
                "Use atomic operations",
                "Review shared resource access"
            ],
            "thread_safety": [
                "Use thread-safe data structures",
                "Add proper locking mechanisms",
                "Review concurrent access patterns"
            ]
        }
        
        return suggestions_map.get(error_type, ["Review concurrency patterns"])
    
    def _get_security_suggestions(self, error_type: str) -> List[str]:
        """Get suggestions for security errors."""
        suggestions_map = {
            "sql_injection": [
                "Use parameterized queries",
                "Validate input data",
                "Implement proper sanitization"
            ],
            "xss": [
                "Escape output data",
                "Validate and sanitize input",
                "Use Content Security Policy"
            ],
            "unauthorized_access": [
                "Implement proper authentication",
                "Check authorization logic",
                "Review access controls"
            ]
        }
        
        return suggestions_map.get(error_type, ["Review security best practices"])
    
    def _identify_affected_components(self, error_context: ErrorContext) -> List[str]:
        """Identify affected system components."""
        components = []
        
        if error_context.file_path:
            components.append(error_context.file_path)
        
        if error_context.function_name:
            components.append(f"function:{error_context.function_name}")
        
        if error_context.service_name:
            components.append(f"service:{error_context.service_name}")
        
        # Infer components from error message
        error_msg = error_context.error_message.lower()
        
        if 'database' in error_msg or 'sql' in error_msg:
            components.append('database')
        
        if 'redis' in error_msg:
            components.append('redis')
        
        if 'network' in error_msg or 'connection' in error_msg:
            components.append('network')
        
        if 'file' in error_msg or 'directory' in error_msg:
            components.append('filesystem')
        
        return components
    
    def _estimate_fix_complexity(self, category: ErrorCategory, root_cause: str) -> str:
        """Estimate the complexity of fixing the error."""
        
        # Simple fixes
        simple_fixes = {
            ErrorCategory.SYNTAX: ["invalid_syntax", "missing_indentation", "unexpected_token"],
            ErrorCategory.CONFIGURATION: ["missing_env_var", "invalid_config"],
            ErrorCategory.DEPENDENCY: ["missing_module"]
        }
        
        # Complex fixes
        complex_fixes = {
            ErrorCategory.CONCURRENCY: ["deadlock", "race_condition"],
            ErrorCategory.SECURITY: ["sql_injection", "xss"],
            ErrorCategory.LOGIC: ["algorithm_error"]
        }
        
        # Check for simple fixes
        for cat, causes in simple_fixes.items():
            if category == cat and root_cause in causes:
                return "simple"
        
        # Check for complex fixes
        for cat, causes in complex_fixes.items():
            if category == cat and root_cause in causes:
                return "complex"
        
        # Default to moderate
        return "moderate"
    
    def correlate_logs_and_traces(self, error_contexts: List[ErrorContext], 
                                  monitoring_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Correlate multiple error logs, stack traces, and monitoring data to pinpoint complex issues.
        
        Args:
            error_contexts: List of error contexts to correlate
            monitoring_data: Additional monitoring/metrics data
            
        Returns:
            Correlation analysis results
        """
        correlation_result = {
            "total_errors": len(error_contexts),
            "error_patterns": defaultdict(int),
            "affected_services": set(),
            "time_correlation": {},
            "cascading_failures": [],
            "root_cause_analysis": {}
        }
        
        # Analyze error patterns
        for context in error_contexts:
            classification = self.classify_error(context)
            correlation_result["error_patterns"][classification.category.value] += 1
            
            if context.service_name:
                correlation_result["affected_services"].add(context.service_name)
        
        # Convert sets to lists for JSON serialization
        correlation_result["affected_services"] = list(correlation_result["affected_services"])
        correlation_result["error_patterns"] = dict(correlation_result["error_patterns"])
        
        # Detect cascading failures
        if len(error_contexts) > 1:
            correlation_result["cascading_failures"] = self._detect_cascading_failures(error_contexts)
        
        # Correlate with monitoring data if available
        if monitoring_data:
            correlation_result["monitoring_correlation"] = self._correlate_with_monitoring(
                error_contexts, monitoring_data
            )
        
        return correlation_result
    
    def _detect_cascading_failures(self, error_contexts: List[ErrorContext]) -> List[Dict[str, Any]]:
        """Detect cascading failures in error sequence."""
        cascading = []
        
        # Sort by timestamp if available
        sorted_contexts = sorted(
            error_contexts,
            key=lambda x: x.timestamp or "0"
        )
        
        # Look for patterns that suggest cascading failures
        for i in range(len(sorted_contexts) - 1):
            current = sorted_contexts[i]
            next_error = sorted_contexts[i + 1]
            
            # Check if errors are related
            if (current.service_name == next_error.service_name or
                self._are_errors_related(current, next_error)):
                
                cascading.append({
                    "primary_error": {
                        "service": current.service_name,
                        "error": current.error_message[:100],
                        "timestamp": current.timestamp
                    },
                    "secondary_error": {
                        "service": next_error.service_name,
                        "error": next_error.error_message[:100],
                        "timestamp": next_error.timestamp
                    },
                    "relationship": "potential_cascade"
                })
        
        return cascading
    
    def _are_errors_related(self, error1: ErrorContext, error2: ErrorContext) -> bool:
        """Check if two errors are potentially related."""
        
        # Check if they involve similar components
        if error1.file_path and error2.file_path:
            if Path(error1.file_path).parent == Path(error2.file_path).parent:
                return True
        
        # Check for common keywords in error messages
        common_keywords = set(error1.error_message.lower().split()) & set(error2.error_message.lower().split())
        if len(common_keywords) >= 3:  # At least 3 common words
            return True
        
        return False
    
    def _correlate_with_monitoring(self, error_contexts: List[ErrorContext], 
                                   monitoring_data: Dict[str, Any]) -> Dict[str, Any]:
        """Correlate errors with monitoring data."""
        correlation = {
            "metrics_anomalies": [],
            "performance_impact": {},
            "resource_usage": {}
        }
        
        # This would integrate with actual monitoring systems
        # For now, provide a placeholder structure
        
        # Analyze CPU/Memory spikes around error times
        if "cpu_usage" in monitoring_data:
            correlation["resource_usage"]["cpu_spike_detected"] = (
                monitoring_data["cpu_usage"] > 80
            )
        
        if "memory_usage" in monitoring_data:
            correlation["resource_usage"]["memory_spike_detected"] = (
                monitoring_data["memory_usage"] > 90
            )
        
        return correlation


# Utility functions for integration with existing system

def create_error_context_from_log(log_entry: Dict[str, Any], 
                                  project_root: Optional[str] = None) -> ErrorContext:
    """
    Create an ErrorContext from a log entry.
    
    Args:
        log_entry: Log entry dictionary
        project_root: Root directory of the project
        
    Returns:
        ErrorContext object
    """
    # Extract basic information
    error_message = log_entry.get("message", "")
    exception_type = log_entry.get("exception_type")
    stack_trace = log_entry.get("traceback", [])
    
    # Extract source code context from error details
    error_details = log_entry.get("error_details", {})
    detailed_frames = error_details.get("detailed_frames", [])
    
    file_path = None
    line_number = None
    function_name = None
    
    if detailed_frames:
        frame = detailed_frames[0]  # Take the first frame
        file_path = frame.get("file")
        line_number = frame.get("line")
        function_name = frame.get("function")
    
    # Create error context
    context = ErrorContext(
        error_message=error_message,
        exception_type=exception_type,
        stack_trace=stack_trace,
        file_path=file_path,
        line_number=line_number,
        function_name=function_name,
        timestamp=log_entry.get("timestamp"),
        service_name=log_entry.get("service")
    )
    
    # Add environment context
    detector = ComprehensiveErrorDetector()
    env_context = detector.extract_context_from_environment()
    
    context.os_type = env_context.get("os_type")
    context.python_version = env_context.get("python_version")
    context.node_version = env_context.get("node_version")
    context.java_version = env_context.get("java_version")
    context.environment_variables = env_context.get("environment_variables")
    
    # Add dependency context
    if project_root:
        dependencies = detector.detect_dependencies(project_root)
        context.dependencies = dependencies
    
    return context


def analyze_comprehensive_error(log_entry: Dict[str, Any], 
                               project_root: Optional[str] = None) -> Dict[str, Any]:
    """
    Perform comprehensive error analysis on a log entry.
    
    Args:
        log_entry: Log entry dictionary
        project_root: Root directory of the project
        
    Returns:
        Comprehensive analysis results
    """
    # Create error context
    error_context = create_error_context_from_log(log_entry, project_root)
    
    # Initialize detector and classify error
    detector = ComprehensiveErrorDetector()
    classification = detector.classify_error(error_context)
    
    # Return combined results
    return {
        "error_context": error_context.to_dict(),
        "classification": classification.to_dict(),
        "analysis_method": "comprehensive_detection",
        "detector_version": "1.0.0"
    }


if __name__ == "__main__":
    # Example usage and testing
    print("Comprehensive Error Detection System")
    print("===================================")
    
    # Test with sample Python error
    python_error_context = ErrorContext(
        error_message="NameError: name 'undefined_variable' is not defined",
        exception_type="NameError",
        stack_trace=[
            "Traceback (most recent call last):",
            "  File 'test.py', line 10, in main",
            "    print(undefined_variable)",
            "NameError: name 'undefined_variable' is not defined"
        ],
        file_path="test.py",
        line_number=10,
        function_name="main",
        language=LanguageType.PYTHON
    )
    
    detector = ComprehensiveErrorDetector()
    classification = detector.classify_error(python_error_context)
    
    print(f"\nPython Error Classification:")
    print(f"Category: {classification.category.value}")
    print(f"Severity: {classification.severity.value}")
    print(f"Confidence: {classification.confidence:.2f}")
    print(f"Root Cause: {classification.root_cause}")
    print(f"Description: {classification.description}")
    print(f"Suggestions: {classification.suggestions}")
    print(f"Fix Complexity: {classification.potential_fix_complexity}")
    
    # Test with sample JavaScript error
    js_error_context = ErrorContext(
        error_message="ReferenceError: myFunction is not defined",
        exception_type="ReferenceError",
        file_path="app.js",
        line_number=25,
        language=LanguageType.JAVASCRIPT
    )
    
    js_classification = detector.classify_error(js_error_context)
    
    print(f"\nJavaScript Error Classification:")
    print(f"Category: {js_classification.category.value}")
    print(f"Severity: {js_classification.severity.value}")
    print(f"Confidence: {js_classification.confidence:.2f}")
    print(f"Root Cause: {js_classification.root_cause}")
    print(f"Suggestions: {js_classification.suggestions}")
    
    # Test correlation functionality
    print(f"\nTesting Error Correlation:")
    error_contexts = [python_error_context, js_error_context]
    correlation = detector.correlate_logs_and_traces(error_contexts)
    
    print(f"Total Errors: {correlation['total_errors']}")
    print(f"Error Patterns: {correlation['error_patterns']}")
    print(f"Affected Services: {correlation['affected_services']}")