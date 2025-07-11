#!/usr/bin/env python3
"""
Multi-Language and Framework Detection for LLM Patch Generation

This module provides comprehensive language and framework detection capabilities
to enable LLM-based patch generation across multiple programming languages and frameworks.
"""

import json
import re
import ast
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass
from enum import Enum


logger = logging.getLogger(__name__)


class LanguageType(Enum):
    """Supported programming languages."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    GO = "go"
    RUST = "rust"
    SWIFT = "swift"
    KOTLIN = "kotlin"
    CSHARP = "csharp"
    RUBY = "ruby"
    PHP = "php"
    SCALA = "scala"
    ELIXIR = "elixir"
    CLOJURE = "clojure"
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
class FrameworkInfo:
    """Information about a detected framework."""
    name: str
    language: LanguageType
    version: Optional[str] = None
    confidence: float = 0.0
    indicators: List[str] = None
    
    def __post_init__(self):
        if self.indicators is None:
            self.indicators = []


@dataclass
class LanguageInfo:
    """Information about detected language and frameworks."""
    language: LanguageType
    confidence: float
    frameworks: List[FrameworkInfo]
    file_patterns: List[str]
    language_features: Dict[str, Any]


class MultiLanguageFrameworkDetector:
    """
    Comprehensive detector for programming languages and frameworks.
    
    This detector can identify:
    1. Programming language from file extensions and content
    2. Frameworks and libraries being used
    3. Language-specific features and patterns
    4. Project structure and configuration files
    """

    def __init__(self):
        """Initialize the detector with language and framework patterns."""
        self.language_patterns = self._init_language_patterns()
        self.framework_patterns = self._init_framework_patterns()
        self.file_extension_map = self._init_file_extension_map()
        
    def _init_language_patterns(self) -> Dict[LanguageType, Dict[str, Any]]:
        """Initialize language-specific detection patterns."""
        return {
            LanguageType.PYTHON: {
                'imports': [
                    r'import\s+\w+',
                    r'from\s+\w+\s+import',
                    r'import\s+\w+\.\w+',
                ],
                'syntax': [
                    r'def\s+\w+\s*\(',
                    r'class\s+\w+\s*\(',
                    r'if\s+__name__\s*==\s*["\']__main__["\']',
                    r'@\w+',  # decorators
                    r':\s*$',  # colon at end of line
                ],
                'keywords': [
                    'def', 'class', 'import', 'from', 'if', 'elif', 'else',
                    'try', 'except', 'finally', 'with', 'as', 'lambda'
                ],
                'comment_style': '#',
                'indent_style': 'spaces',
                'typical_indent': 4
            },
            LanguageType.JAVASCRIPT: {
                'imports': [
                    r'import\s+.*\s+from\s+["\'].+["\']',
                    r'const\s+.*=\s*require\s*\(',
                    r'import\s*\(',  # dynamic imports
                ],
                'syntax': [
                    r'function\s+\w+\s*\(',
                    r'const\s+\w+\s*=',
                    r'let\s+\w+\s*=',
                    r'var\s+\w+\s*=',
                    r'=>\s*{',  # arrow functions
                    r'}\s*;?\s*$',
                ],
                'keywords': [
                    'function', 'const', 'let', 'var', 'if', 'else', 'for',
                    'while', 'return', 'import', 'export', 'class', 'extends'
                ],
                'comment_style': '//',
                'indent_style': 'spaces',
                'typical_indent': 2
            },
            LanguageType.TYPESCRIPT: {
                'imports': [
                    r'import\s+.*\s+from\s+["\'].+["\']',
                    r'import\s+type\s+',
                ],
                'syntax': [
                    r'interface\s+\w+\s*{',
                    r'type\s+\w+\s*=',
                    r':\s*\w+(\[\])?(\s*\|)?',  # type annotations
                    r'<\w+>',  # generics
                    r'as\s+\w+',  # type assertions
                ],
                'keywords': [
                    'interface', 'type', 'enum', 'namespace', 'declare',
                    'public', 'private', 'protected', 'readonly', 'abstract'
                ],
                'comment_style': '//',
                'indent_style': 'spaces',
                'typical_indent': 2
            },
            LanguageType.JAVA: {
                'imports': [
                    r'import\s+[\w\.]+;',
                    r'package\s+[\w\.]+;',
                ],
                'syntax': [
                    r'public\s+class\s+\w+',
                    r'public\s+static\s+void\s+main',
                    r'@\w+',  # annotations
                    r'}\s*$',
                    r';\s*$',
                ],
                'keywords': [
                    'public', 'private', 'protected', 'static', 'final',
                    'class', 'interface', 'extends', 'implements', 'package'
                ],
                'comment_style': '//',
                'indent_style': 'spaces',
                'typical_indent': 4
            },
            LanguageType.GO: {
                'imports': [
                    r'import\s+["\'][^"\']+["\']',
                    r'import\s+\(',
                    r'package\s+\w+',
                ],
                'syntax': [
                    r'func\s+\w+\s*\(',
                    r'type\s+\w+\s+struct',
                    r'type\s+\w+\s+interface',
                    r':=',  # short variable declaration
                    r'go\s+\w+\(',  # goroutines
                ],
                'keywords': [
                    'func', 'package', 'import', 'type', 'struct', 'interface',
                    'var', 'const', 'if', 'else', 'for', 'range', 'select', 'case'
                ],
                'comment_style': '//',
                'indent_style': 'tabs',
                'typical_indent': 1
            },
            LanguageType.RUST: {
                'imports': [
                    r'use\s+[\w:]+;',
                    r'extern\s+crate\s+\w+;',
                ],
                'syntax': [
                    r'fn\s+\w+\s*\(',
                    r'struct\s+\w+\s*{',
                    r'enum\s+\w+\s*{',
                    r'impl\s+\w+',
                    r'&\w+',  # references
                    r'->\s*\w+',  # return types
                ],
                'keywords': [
                    'fn', 'struct', 'enum', 'impl', 'trait', 'use', 'mod',
                    'let', 'mut', 'match', 'if', 'else', 'loop', 'while', 'for'
                ],
                'comment_style': '//',
                'indent_style': 'spaces',
                'typical_indent': 4
            },
            LanguageType.SWIFT: {
                'imports': [
                    r'import\s+\w+',
                ],
                'syntax': [
                    r'func\s+\w+\s*\(',
                    r'class\s+\w+\s*:',
                    r'struct\s+\w+\s*{',
                    r'enum\s+\w+\s*{',
                    r'var\s+\w+\s*:',
                    r'let\s+\w+\s*=',
                ],
                'keywords': [
                    'func', 'class', 'struct', 'enum', 'protocol', 'extension',
                    'var', 'let', 'if', 'else', 'guard', 'switch', 'case', 'for'
                ],
                'comment_style': '//',
                'indent_style': 'spaces',
                'typical_indent': 4
            }
        }

    def _init_framework_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize framework detection patterns."""
        return {
            # Python Frameworks
            'django': {
                'language': LanguageType.PYTHON,
                'patterns': [
                    r'from\s+django',
                    r'import\s+django',
                    r'DJANGO_SETTINGS_MODULE',
                    r'django\.db\.',
                    r'models\.Model',
                ],
                'files': ['manage.py', 'settings.py', 'urls.py'],
                'directories': ['migrations/'],
                'config_files': ['requirements.txt', 'Pipfile', 'setup.py']
            },
            'flask': {
                'language': LanguageType.PYTHON,
                'patterns': [
                    r'from\s+flask\s+import',
                    r'Flask\s*\(',
                    r'@app\.route',
                    r'flask\.Flask',
                ],
                'files': ['app.py', 'run.py'],
                'config_files': ['requirements.txt', 'Pipfile']
            },
            'fastapi': {
                'language': LanguageType.PYTHON,
                'patterns': [
                    r'from\s+fastapi\s+import',
                    r'FastAPI\s*\(',
                    r'@app\.(get|post|put|delete)',
                    r'Depends\s*\(',
                ],
                'files': ['main.py', 'app.py'],
                'config_files': ['requirements.txt', 'Pipfile']
            },
            
            # JavaScript/TypeScript Frameworks
            'react': {
                'language': LanguageType.JAVASCRIPT,
                'patterns': [
                    r'import\s+React',
                    r'from\s+["\']react["\']',
                    r'useState\s*\(',
                    r'useEffect\s*\(',
                    r'className=',
                    r'JSX\.Element',
                ],
                'files': [],
                'config_files': ['package.json']
            },
            'vue': {
                'language': LanguageType.JAVASCRIPT,
                'patterns': [
                    r'import\s+Vue',
                    r'from\s+["\']vue["\']',
                    r'<template>',
                    r'<script>',
                    r'Vue\.component',
                ],
                'files': [],
                'config_files': ['package.json', 'vue.config.js']
            },
            'angular': {
                'language': LanguageType.TYPESCRIPT,
                'patterns': [
                    r'@Component\s*\(',
                    r'@Injectable\s*\(',
                    r'@NgModule\s*\(',
                    r'from\s+["\']@angular/',
                ],
                'files': ['angular.json'],
                'config_files': ['package.json', 'tsconfig.json']
            },
            'nextjs': {
                'language': LanguageType.JAVASCRIPT,
                'patterns': [
                    r'from\s+["\']next/',
                    r'getStaticProps',
                    r'getServerSideProps',
                    r'pages/_app',
                ],
                'files': ['next.config.js'],
                'config_files': ['package.json']
            },
            
            # Java Frameworks
            'spring': {
                'language': LanguageType.JAVA,
                'patterns': [
                    r'@SpringBootApplication',
                    r'@RestController',
                    r'@Service',
                    r'@Repository',
                    r'org\.springframework\.',
                ],
                'files': ['application.properties', 'application.yml'],
                'config_files': ['pom.xml', 'build.gradle']
            },
            
            # Go Frameworks
            'gin': {
                'language': LanguageType.GO,
                'patterns': [
                    r'gin\.Default\s*\(',
                    r'gin\.New\s*\(',
                    r'c\s*\*gin\.Context',
                    r'github\.com/gin-gonic/gin',
                ],
                'files': [],
                'config_files': ['go.mod', 'go.sum']
            },
            'echo': {
                'language': LanguageType.GO,
                'patterns': [
                    r'echo\.New\s*\(',
                    r'e\s*\*echo\.Echo',
                    r'github\.com/labstack/echo',
                ],
                'files': [],
                'config_files': ['go.mod', 'go.sum']
            },
            
            # Rust Frameworks
            'actix': {
                'language': LanguageType.RUST,
                'patterns': [
                    r'actix_web::',
                    r'HttpServer::new',
                    r'App::new',
                    r'actix-web\s*=',
                ],
                'files': [],
                'config_files': ['Cargo.toml']
            },
            'rocket': {
                'language': LanguageType.RUST,
                'patterns': [
                    r'#\[macro_use\]\s*extern\s+crate\s+rocket',
                    r'rocket::',
                    r'#\[launch\]',
                    r'rocket\s*=',
                ],
                'files': [],
                'config_files': ['Cargo.toml']
            }
        }

    def _init_file_extension_map(self) -> Dict[str, LanguageType]:
        """Initialize file extension to language mapping."""
        return {
            '.py': LanguageType.PYTHON,
            '.js': LanguageType.JAVASCRIPT,
            '.jsx': LanguageType.JAVASCRIPT,
            '.ts': LanguageType.TYPESCRIPT,
            '.tsx': LanguageType.TYPESCRIPT,
            '.java': LanguageType.JAVA,
            '.go': LanguageType.GO,
            '.rs': LanguageType.RUST,
            '.swift': LanguageType.SWIFT,
            '.kt': LanguageType.KOTLIN,
            '.cs': LanguageType.CSHARP,
            '.rb': LanguageType.RUBY,
            '.php': LanguageType.PHP,
            '.scala': LanguageType.SCALA,
            '.ex': LanguageType.ELIXIR,
            '.exs': LanguageType.ELIXIR,
            '.clj': LanguageType.CLOJURE,
            '.cljs': LanguageType.CLOJURE,
            '.cpp': LanguageType.CPP,
            '.cc': LanguageType.CPP,
            '.cxx': LanguageType.CPP,
            '.c': LanguageType.C,
            '.h': LanguageType.C,
            '.hpp': LanguageType.CPP,
            # Additional languages from Phase 12.A
            '.zig': LanguageType.ZIG,
            '.nim': LanguageType.NIM,
            '.nims': LanguageType.NIM,
            '.cr': LanguageType.CRYSTAL,
            '.hs': LanguageType.HASKELL,
            '.lhs': LanguageType.HASKELL,
            '.fs': LanguageType.FSHARP,
            '.fsi': LanguageType.FSHARP,
            '.fsx': LanguageType.FSHARP,
            '.erl': LanguageType.ERLANG,
            '.hrl': LanguageType.ERLANG,
            '.sql': LanguageType.SQL,
            '.ddl': LanguageType.SQL,
            '.dml': LanguageType.SQL,
            '.sh': LanguageType.BASH,
            '.bash': LanguageType.BASH,
            '.ps1': LanguageType.POWERSHELL,
            '.psm1': LanguageType.POWERSHELL,
            '.psd1': LanguageType.POWERSHELL,
            '.lua': LanguageType.LUA,
            '.r': LanguageType.R,
            '.R': LanguageType.R,
            '.m': LanguageType.MATLAB,
            '.mat': LanguageType.MATLAB,
            '.jl': LanguageType.JULIA,
            '.tf': LanguageType.TERRAFORM,
            '.tfvars': LanguageType.TERRAFORM,
            '.yml': LanguageType.YAML,
            '.yaml': LanguageType.YAML,
            '.json': LanguageType.JSON,
            '.jsonc': LanguageType.JSON,
        }

    def detect_language_and_frameworks(self, 
                                     file_path: Optional[str] = None,
                                     source_code: Optional[str] = None,
                                     project_root: Optional[str] = None) -> LanguageInfo:
        """
        Detect programming language and frameworks from file or source code.
        
        Args:
            file_path: Path to the source file
            source_code: Source code content
            project_root: Root directory of the project
            
        Returns:
            LanguageInfo with detected language and frameworks
        """
        # First, try to detect language from file extension
        detected_language = LanguageType.UNKNOWN
        confidence = 0.0
        
        if file_path:
            file_ext = Path(file_path).suffix.lower()
            if file_ext in self.file_extension_map:
                detected_language = self.file_extension_map[file_ext]
                confidence = 0.8  # High confidence from file extension
        
        # If we have source code, analyze it for language patterns
        if source_code and detected_language != LanguageType.UNKNOWN:
            # Validate and refine language detection using content analysis
            content_confidence = self._analyze_language_content(source_code, detected_language)
            confidence = max(confidence, content_confidence)
        elif source_code:
            # Try to detect language from content alone
            detected_language, confidence = self._detect_language_from_content(source_code)
        
        # Detect frameworks
        frameworks = []
        if source_code:
            frameworks = self._detect_frameworks(source_code, detected_language, project_root)
        
        # Get language features
        language_features = self._get_language_features(detected_language, source_code)
        
        # Get file patterns for this language
        file_patterns = self._get_file_patterns(detected_language)
        
        return LanguageInfo(
            language=detected_language,
            confidence=confidence,
            frameworks=frameworks,
            file_patterns=file_patterns,
            language_features=language_features
        )

    def _analyze_language_content(self, source_code: str, suspected_language: LanguageType) -> float:
        """
        Analyze source code content to validate language detection.
        
        Args:
            source_code: Source code to analyze
            suspected_language: Language detected from file extension
            
        Returns:
            Confidence score for the language detection
        """
        if suspected_language not in self.language_patterns:
            return 0.0
        
        patterns = self.language_patterns[suspected_language]
        total_score = 0
        max_score = 0
        
        # Check syntax patterns
        for pattern in patterns.get('syntax', []):
            max_score += 1
            if re.search(pattern, source_code, re.MULTILINE):
                total_score += 1
        
        # Check import patterns
        for pattern in patterns.get('imports', []):
            max_score += 1
            if re.search(pattern, source_code, re.MULTILINE):
                total_score += 1
        
        # Check for keywords
        keywords = patterns.get('keywords', [])
        if keywords:
            max_score += len(keywords)
            for keyword in keywords:
                if re.search(r'\b' + keyword + r'\b', source_code):
                    total_score += 1
        
        return total_score / max_score if max_score > 0 else 0.0

    def _detect_language_from_content(self, source_code: str) -> Tuple[LanguageType, float]:
        """
        Detect language from source code content alone.
        
        Args:
            source_code: Source code to analyze
            
        Returns:
            Tuple of (detected_language, confidence)
        """
        best_language = LanguageType.UNKNOWN
        best_confidence = 0.0
        
        for language, patterns in self.language_patterns.items():
            confidence = self._analyze_language_content(source_code, language)
            if confidence > best_confidence:
                best_confidence = confidence
                best_language = language
        
        return best_language, best_confidence

    def _detect_frameworks(self, 
                          source_code: str, 
                          language: LanguageType,
                          project_root: Optional[str] = None) -> List[FrameworkInfo]:
        """
        Detect frameworks being used in the source code.
        
        Args:
            source_code: Source code to analyze
            language: Detected programming language
            project_root: Root directory of the project
            
        Returns:
            List of detected frameworks
        """
        frameworks = []
        
        for framework_name, framework_config in self.framework_patterns.items():
            # Skip if framework doesn't match the detected language
            if framework_config['language'] != language:
                continue
            
            confidence = 0.0
            indicators = []
            
            # Check source code patterns
            patterns = framework_config.get('patterns', [])
            pattern_matches = 0
            for pattern in patterns:
                if re.search(pattern, source_code, re.MULTILINE | re.IGNORECASE):
                    pattern_matches += 1
                    indicators.append(f"Pattern: {pattern}")
            
            if patterns:
                confidence += (pattern_matches / len(patterns)) * 0.7
            
            # Check for framework-specific files in project
            if project_root:
                file_matches = 0
                files = framework_config.get('files', [])
                for file_name in files:
                    file_path = Path(project_root) / file_name
                    if file_path.exists():
                        file_matches += 1
                        indicators.append(f"File: {file_name}")
                
                if files:
                    confidence += (file_matches / len(files)) * 0.3
            
            # Only include frameworks with reasonable confidence
            if confidence > 0.2:
                frameworks.append(FrameworkInfo(
                    name=framework_name,
                    language=language,
                    confidence=confidence,
                    indicators=indicators
                ))
        
        # Sort by confidence
        frameworks.sort(key=lambda f: f.confidence, reverse=True)
        return frameworks

    def _get_language_features(self, 
                              language: LanguageType, 
                              source_code: Optional[str] = None) -> Dict[str, Any]:
        """
        Get language-specific features and characteristics.
        
        Args:
            language: Detected language
            source_code: Source code to analyze
            
        Returns:
            Dictionary of language features
        """
        if language not in self.language_patterns:
            return {}
        
        patterns = self.language_patterns[language]
        features = {
            'comment_style': patterns.get('comment_style', '#'),
            'indent_style': patterns.get('indent_style', 'spaces'),
            'typical_indent': patterns.get('typical_indent', 4),
            'keywords': patterns.get('keywords', [])
        }
        
        # Analyze actual indentation if source code is provided
        if source_code:
            features['detected_indent'] = self._detect_indentation(source_code)
        
        return features

    def _detect_indentation(self, source_code: str) -> Dict[str, Any]:
        """
        Detect indentation style and size from source code.
        
        Args:
            source_code: Source code to analyze
            
        Returns:
            Dictionary with indentation information
        """
        lines = source_code.split('\n')
        space_indents = []
        tab_indents = []
        
        for line in lines:
            if line.strip():  # Skip empty lines
                leading_spaces = len(line) - len(line.lstrip(' '))
                leading_tabs = len(line) - len(line.lstrip('\t'))
                
                if leading_spaces > 0:
                    space_indents.append(leading_spaces)
                elif leading_tabs > 0:
                    tab_indents.append(leading_tabs)
        
        indent_info = {
            'uses_spaces': len(space_indents) > 0,
            'uses_tabs': len(tab_indents) > 0,
            'space_count': len(space_indents),
            'tab_count': len(tab_indents)
        }
        
        if space_indents:
            # Find most common indentation size
            from collections import Counter
            indent_sizes = [indent for indent in space_indents if indent > 0]
            if indent_sizes:
                most_common = Counter(indent_sizes).most_common(1)[0][0]
                indent_info['typical_spaces'] = most_common
        
        return indent_info

    def _get_file_patterns(self, language: LanguageType) -> List[str]:
        """
        Get typical file patterns for a language.
        
        Args:
            language: Programming language
            
        Returns:
            List of file patterns
        """
        patterns_map = {
            LanguageType.PYTHON: ['*.py', '*.pyw'],
            LanguageType.JAVASCRIPT: ['*.js', '*.jsx'],
            LanguageType.TYPESCRIPT: ['*.ts', '*.tsx'],
            LanguageType.JAVA: ['*.java'],
            LanguageType.GO: ['*.go'],
            LanguageType.RUST: ['*.rs'],
            LanguageType.SWIFT: ['*.swift'],
            LanguageType.KOTLIN: ['*.kt', '*.kts'],
            LanguageType.CSHARP: ['*.cs'],
            LanguageType.RUBY: ['*.rb'],
            LanguageType.PHP: ['*.php'],
            LanguageType.SCALA: ['*.scala'],
            LanguageType.ELIXIR: ['*.ex', '*.exs'],
            LanguageType.CLOJURE: ['*.clj', '*.cljs'],
            LanguageType.CPP: ['*.cpp', '*.cc', '*.cxx', '*.hpp'],
            LanguageType.C: ['*.c', '*.h'],
            # Additional languages from Phase 12.A
            LanguageType.ZIG: ['*.zig'],
            LanguageType.NIM: ['*.nim', '*.nims'],
            LanguageType.CRYSTAL: ['*.cr'],
            LanguageType.HASKELL: ['*.hs', '*.lhs'],
            LanguageType.FSHARP: ['*.fs', '*.fsi', '*.fsx'],
            LanguageType.ERLANG: ['*.erl', '*.hrl'],
            LanguageType.SQL: ['*.sql', '*.ddl', '*.dml'],
            LanguageType.BASH: ['*.sh', '*.bash'],
            LanguageType.POWERSHELL: ['*.ps1', '*.psm1', '*.psd1'],
            LanguageType.LUA: ['*.lua'],
            LanguageType.R: ['*.r', '*.R'],
            LanguageType.MATLAB: ['*.m', '*.mat'],
            LanguageType.JULIA: ['*.jl'],
            LanguageType.TERRAFORM: ['*.tf', '*.tfvars'],
            LanguageType.ANSIBLE: ['*.yml', '*.yaml', 'playbook.yml', 'site.yml'],
            LanguageType.YAML: ['*.yml', '*.yaml'],
            LanguageType.JSON: ['*.json', '*.jsonc'],
            LanguageType.DOCKERFILE: ['Dockerfile', 'Dockerfile.*', '*.dockerfile'],
        }
        
        return patterns_map.get(language, [])

    def get_llm_context_for_language(self, language_info: LanguageInfo) -> Dict[str, Any]:
        """
        Generate LLM context information for the detected language and frameworks.
        
        Args:
            language_info: Detected language information
            
        Returns:
            Context dictionary for LLM prompts
        """
        context = {
            'language': language_info.language.value,
            'confidence': language_info.confidence,
            'features': language_info.language_features,
            'file_patterns': language_info.file_patterns,
            'frameworks': []
        }
        
        # Add framework information
        for framework in language_info.frameworks:
            context['frameworks'].append({
                'name': framework.name,
                'confidence': framework.confidence,
                'indicators': framework.indicators
            })
        
        # Add language-specific guidance for LLMs
        language_guidance = self._get_language_guidance(language_info.language)
        context['llm_guidance'] = language_guidance
        
        return context

    def _get_language_guidance(self, language: LanguageType) -> Dict[str, Any]:
        """
        Get language-specific guidance for LLM patch generation.
        
        Args:
            language: Programming language
            
        Returns:
            Guidance dictionary
        """
        guidance_map = {
            LanguageType.PYTHON: {
                'style_guide': 'PEP 8',
                'common_patterns': [
                    'Use snake_case for variables and functions',
                    'Use PascalCase for classes',
                    'Prefer list comprehensions when appropriate',
                    'Use context managers (with statements) for resource management'
                ],
                'error_handling': 'Use try/except blocks with specific exception types',
                'imports': 'Place imports at top, group standard library, third-party, local'
            },
            LanguageType.JAVASCRIPT: {
                'style_guide': 'Airbnb or Standard',
                'common_patterns': [
                    'Use camelCase for variables and functions',
                    'Use PascalCase for constructors and classes',
                    'Prefer const/let over var',
                    'Use arrow functions for callbacks'
                ],
                'error_handling': 'Use try/catch blocks or Promise .catch()',
                'imports': 'Use ES6 import/export syntax'
            },
            LanguageType.JAVA: {
                'style_guide': 'Google Java Style Guide',
                'common_patterns': [
                    'Use camelCase for variables and methods',
                    'Use PascalCase for classes',
                    'Use ALL_CAPS for constants',
                    'Follow bean naming conventions'
                ],
                'error_handling': 'Use try/catch blocks with specific exception types',
                'imports': 'Organize imports and avoid wildcards'
            },
            LanguageType.GO: {
                'style_guide': 'Go official style guide',
                'common_patterns': [
                    'Use camelCase for exported functions',
                    'Use lowercase for package-private',
                    'Follow receiver naming conventions',
                    'Use short variable names in small scopes'
                ],
                'error_handling': 'Check errors explicitly, return error as last value',
                'imports': 'Group standard library, third-party, local'
            },
            LanguageType.RUST: {
                'style_guide': 'Rust official style guide',
                'common_patterns': [
                    'Use snake_case for variables and functions',
                    'Use PascalCase for types and traits',
                    'Use SCREAMING_SNAKE_CASE for constants',
                    'Prefer ? operator for error propagation'
                ],
                'error_handling': 'Use Result<T, E> and Option<T> types',
                'imports': 'Use explicit use statements'
            },
            LanguageType.ZIG: {
                'style_guide': 'Zig style guide',
                'common_patterns': [
                    'Use snake_case for variables and functions',
                    'Use PascalCase for types',
                    'Explicit error handling with error unions',
                    'Comptime for compile-time computation'
                ],
                'error_handling': 'Use error unions and try/catch',
                'imports': 'Use @import() for modules'
            },
            LanguageType.NIM: {
                'style_guide': 'Nim style guide',
                'common_patterns': [
                    'Use camelCase for procedures and variables',
                    'Use PascalCase for types',
                    'Prefer result types for error handling',
                    'Use pragmas for compiler hints'
                ],
                'error_handling': 'Use exceptions or Option/Result types',
                'imports': 'Use import statements'
            },
            LanguageType.CRYSTAL: {
                'style_guide': 'Crystal style guide (Ruby-like)',
                'common_patterns': [
                    'Use snake_case for methods and variables',
                    'Use PascalCase for classes and modules',
                    'Use SCREAMING_SNAKE_CASE for constants',
                    'Type inference with optional type annotations'
                ],
                'error_handling': 'Use exceptions with begin/rescue/end',
                'imports': 'Use require statements'
            },
            LanguageType.HASKELL: {
                'style_guide': 'Haskell style guide',
                'common_patterns': [
                    'Use camelCase for functions and variables',
                    'Use PascalCase for types and constructors',
                    'Pattern matching for control flow',
                    'Pure functions by default'
                ],
                'error_handling': 'Use Maybe, Either, or custom monads',
                'imports': 'Use import statements with qualified names'
            },
            LanguageType.FSHARP: {
                'style_guide': 'F# style guide',
                'common_patterns': [
                    'Use camelCase for values and functions',
                    'Use PascalCase for types and modules',
                    'Prefer immutability',
                    'Use pattern matching extensively'
                ],
                'error_handling': 'Use Result<\'T,\'TError> or Option types',
                'imports': 'Use open statements'
            },
            LanguageType.ERLANG: {
                'style_guide': 'Erlang style guide',
                'common_patterns': [
                    'Use snake_case for functions and atoms',
                    'Use PascalCase for variables',
                    'Actor model with message passing',
                    'Pattern matching in function heads'
                ],
                'error_handling': 'Use pattern matching on {ok, Result} or {error, Reason}',
                'imports': 'Use -include and -import directives'
            },
            LanguageType.SQL: {
                'style_guide': 'SQL style conventions',
                'common_patterns': [
                    'Use UPPERCASE for SQL keywords',
                    'Use snake_case for table and column names',
                    'Proper indentation for readability',
                    'Use meaningful aliases'
                ],
                'error_handling': 'Handle NULL values and use transactions',
                'imports': 'N/A - Use proper schema references'
            },
            LanguageType.BASH: {
                'style_guide': 'Bash style guide (Google)',
                'common_patterns': [
                    'Use lowercase with underscores for variables',
                    'Use uppercase for environment variables',
                    'Quote variables to prevent word splitting',
                    'Use [[ ]] for conditionals'
                ],
                'error_handling': 'Check exit codes, use set -e, trap errors',
                'imports': 'Use source or . to include scripts'
            },
            LanguageType.POWERSHELL: {
                'style_guide': 'PowerShell style guide',
                'common_patterns': [
                    'Use PascalCase for cmdlets (Verb-Noun)',
                    'Use camelCase for variables',
                    'Use approved verbs for functions',
                    'Explicit type declarations when needed'
                ],
                'error_handling': 'Use try/catch blocks and -ErrorAction',
                'imports': 'Use Import-Module'
            },
            LanguageType.LUA: {
                'style_guide': 'Lua style guide',
                'common_patterns': [
                    'Use snake_case for variables and functions',
                    'Use PascalCase for classes/modules',
                    'Local variables preferred over global',
                    'Tables for data structures'
                ],
                'error_handling': 'Use pcall/xpcall for protected calls',
                'imports': 'Use require() for modules'
            },
            LanguageType.R: {
                'style_guide': 'R style guide (tidyverse)',
                'common_patterns': [
                    'Use snake_case for variables and functions',
                    'Use <- for assignment',
                    'Vectorized operations preferred',
                    'Use explicit returns'
                ],
                'error_handling': 'Use tryCatch() blocks',
                'imports': 'Use library() or require()'
            },
            LanguageType.MATLAB: {
                'style_guide': 'MATLAB style guide',
                'common_patterns': [
                    'Use camelCase for variables and functions',
                    'Use uppercase for constants',
                    'Vectorize operations when possible',
                    'Preallocate arrays'
                ],
                'error_handling': 'Use try/catch blocks',
                'imports': 'Use addpath or import'
            },
            LanguageType.JULIA: {
                'style_guide': 'Julia style guide',
                'common_patterns': [
                    'Use snake_case for functions and variables',
                    'Use PascalCase for types and modules',
                    'Type annotations for performance',
                    'Multiple dispatch for polymorphism'
                ],
                'error_handling': 'Use try/catch blocks or @error macro',
                'imports': 'Use using or import'
            },
            LanguageType.TERRAFORM: {
                'style_guide': 'Terraform style conventions',
                'common_patterns': [
                    'Use snake_case for all names',
                    'Group related resources',
                    'Use meaningful resource names',
                    'Pin provider versions'
                ],
                'error_handling': 'Use validation blocks and preconditions',
                'imports': 'Use module blocks'
            },
            LanguageType.ANSIBLE: {
                'style_guide': 'Ansible best practices',
                'common_patterns': [
                    'Use snake_case for variables',
                    'Prefix role variables with role name',
                    'Use meaningful task names',
                    'YAML formatting with proper indentation'
                ],
                'error_handling': 'Use failed_when, ignore_errors, and block/rescue',
                'imports': 'Use include_tasks or import_playbook'
            },
            LanguageType.YAML: {
                'style_guide': 'YAML style guide',
                'common_patterns': [
                    'Consistent indentation (2 or 4 spaces)',
                    'Use hyphens for lists',
                    'Quote strings when necessary',
                    'Avoid tabs'
                ],
                'error_handling': 'Validate schema compliance',
                'imports': 'Use anchors and aliases for reuse'
            },
            LanguageType.JSON: {
                'style_guide': 'JSON formatting conventions',
                'common_patterns': [
                    'Use double quotes for strings',
                    'No trailing commas',
                    'Consistent indentation',
                    'Valid data types only'
                ],
                'error_handling': 'Ensure valid JSON syntax',
                'imports': 'N/A - Use references or includes at application level'
            },
            LanguageType.DOCKERFILE: {
                'style_guide': 'Dockerfile best practices',
                'common_patterns': [
                    'Use UPPERCASE for instructions',
                    'One instruction per line',
                    'Minimize layers',
                    'Use specific base image tags'
                ],
                'error_handling': 'Use HEALTHCHECK and proper error codes',
                'imports': 'Use FROM for base images, COPY for files'
            }
        }
        
        return guidance_map.get(language, {
            'style_guide': 'Follow language conventions',
            'common_patterns': [],
            'error_handling': 'Use appropriate error handling for the language',
            'imports': 'Organize imports properly'
        })


def create_multi_language_detector() -> MultiLanguageFrameworkDetector:
    """Factory function to create a multi-language framework detector."""
    return MultiLanguageFrameworkDetector()


if __name__ == "__main__":
    # Test the detector
    print("Testing Multi-Language Framework Detector")
    print("=========================================")
    
    detector = create_multi_language_detector()
    
    # Test Python code
    python_code = '''
import os
from django.db import models
from django.contrib.auth.models import User

class BlogPost(models.Model):
    title = models.CharField(max_length=200)
    content = models.TextField()
    author = models.ForeignKey(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.title
'''
    
    result = detector.detect_language_and_frameworks(
        file_path="blog/models.py",
        source_code=python_code
    )
    
    print(f"Language: {result.language.value}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Frameworks: {[f.name for f in result.frameworks]}")
    print(f"Features: {result.language_features}")
    
    # Test JavaScript code
    js_code = '''
import React, { useState, useEffect } from 'react';
import axios from 'axios';

const BlogList = () => {
    const [posts, setPosts] = useState([]);
    const [loading, setLoading] = useState(true);
    
    useEffect(() => {
        const fetchPosts = async () => {
            try {
                const response = await axios.get('/api/posts');
                setPosts(response.data);
            } catch (error) {
                console.error('Error fetching posts:', error);
            } finally {
                setLoading(false);
            }
        };
        
        fetchPosts();
    }, []);
    
    return (
        <div className="blog-list">
            {loading ? (
                <p>Loading...</p>
            ) : (
                posts.map(post => (
                    <div key={post.id} className="blog-post">
                        <h2>{post.title}</h2>
                        <p>{post.content}</p>
                    </div>
                ))
            )}
        </div>
    );
};

export default BlogList;
'''
    
    result2 = detector.detect_language_and_frameworks(
        file_path="src/components/BlogList.jsx",
        source_code=js_code
    )
    
    print(f"\nSecond test:")
    print(f"Language: {result2.language.value}")
    print(f"Confidence: {result2.confidence:.2f}")
    print(f"Frameworks: {[f.name for f in result2.frameworks]}")
    
    # Test LLM context generation
    context = detector.get_llm_context_for_language(result)
    print(f"\nLLM Context: {json.dumps(context, indent=2)}")