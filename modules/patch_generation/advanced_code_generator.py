#!/usr/bin/env python3
"""
Advanced Code Generator for Phase 13.3

This module implements advanced code generation capabilities including:
- Fine-tuned LLM code repair
- Context-aware code generation
- Style-preserving fix generation
- Multi-file coordinated changes
- Semantic understanding of codebase structure
"""

import ast
import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx

from ..analysis.llm_context_manager import LLMContextManager
from ..analysis.models.transformer_code_understanding import TransformerCodeAnalyzer as TransformerCodeUnderstanding
from ..llm_integration.provider_abstraction import LLMManager, LLMMessage, LLMRequest
from .code_style_analyzer import CodeStyleAnalyzer
from .multi_language_framework_detector import MultiLanguageFrameworkDetector

logger = logging.getLogger(__name__)


class CodeGenerationMode(Enum):
    """Different modes of code generation."""

    SINGLE_FILE = "single_file"
    MULTI_FILE = "multi_file"
    REFACTORING = "refactoring"
    FEATURE_ADDITION = "feature_addition"
    BUG_FIX = "bug_fix"


@dataclass
class CodeContext:
    """Comprehensive code context for generation."""

    target_file: str
    related_files: List[str] = field(default_factory=list)
    imports_map: Dict[str, List[str]] = field(default_factory=dict)
    call_graph: Optional[nx.DiGraph] = None
    class_hierarchy: Dict[str, List[str]] = field(default_factory=dict)
    function_signatures: Dict[str, str] = field(default_factory=dict)
    variable_scopes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    code_patterns: List[str] = field(default_factory=list)
    architectural_patterns: List[str] = field(default_factory=list)
    semantic_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResult:
    """Result of advanced code generation."""

    success: bool
    mode: CodeGenerationMode
    changes: List[Dict[str, Any]]
    multi_file_changes: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    semantic_analysis: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    reasoning: str = ""
    test_suggestions: List[str] = field(default_factory=list)
    dependency_updates: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


class AdvancedCodeGenerator:
    """
    Advanced code generator implementing Phase 13.3 capabilities.

    Features:
    - Fine-tuned LLM integration for code repair
    - Deep context analysis across multiple files
    - Style preservation with advanced pattern matching
    - Coordinated multi-file changes
    - Semantic understanding of codebase structure
    """

    def __init__(
        self,
        llm_manager: LLMManager,
        context_manager: LLMContextManager,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the advanced code generator.

        Args:
            llm_manager: LLM manager for model interactions
            context_manager: Context manager for code analysis
            config: Configuration dictionary
        """
        self.llm_manager = llm_manager
        self.context_manager = context_manager
        self.language_detector = MultiLanguageFrameworkDetector()
        self.style_analyzer = CodeStyleAnalyzer()
        self.transformer_analyzer = TransformerCodeUnderstanding()

        # Configuration
        self.config = config or {}
        self.max_context_window = self.config.get("max_context_window", 16000)
        self.enable_multi_file = self.config.get("enable_multi_file", True)
        self.semantic_depth = self.config.get("semantic_depth", 3)  # Levels of analysis
        self.confidence_threshold = self.config.get("confidence_threshold", 0.7)

        # Advanced prompting configuration
        self.use_chain_of_thought = self.config.get("use_chain_of_thought", True)
        self.use_few_shot = self.config.get("use_few_shot", True)
        self.temperature_by_mode = {
            CodeGenerationMode.BUG_FIX: 0.1,
            CodeGenerationMode.REFACTORING: 0.3,
            CodeGenerationMode.FEATURE_ADDITION: 0.5,
            CodeGenerationMode.SINGLE_FILE: 0.2,
            CodeGenerationMode.MULTI_FILE: 0.2,
        }

        logger.info("Initialized Advanced Code Generator with Phase 13.3 capabilities")

    def generate_with_context(
        self,
        error_context: Dict[str, Any],
        source_code: str,
        mode: CodeGenerationMode = CodeGenerationMode.BUG_FIX,
    ) -> GenerationResult:
        """
        Generate code with comprehensive context analysis.

        Args:
            error_context: Error or task context
            source_code: Source code to analyze
            mode: Generation mode

        Returns:
            Generation result with all changes and analysis
        """
        try:
            # Build comprehensive code context
            code_context = self._build_code_context(
                error_context.get("file_path", ""), source_code, error_context
            )

            # Perform semantic analysis
            semantic_analysis = self._perform_semantic_analysis(
                code_context, source_code, error_context
            )

            # Determine if multi-file changes are needed
            if self.enable_multi_file and self._requires_multi_file_changes(
                semantic_analysis, error_context
            ):
                return self._generate_multi_file_changes(
                    code_context, semantic_analysis, error_context, mode
                )
            else:
                return self._generate_single_file_changes(
                    code_context, semantic_analysis, error_context, source_code, mode
                )

        except Exception as e:
            logger.error(f"Error in advanced code generation: {e}")
            return GenerationResult(
                success=False,
                mode=mode,
                changes=[],
                reasoning=f"Generation failed: {str(e)}",
                warnings=[str(e)],
            )

    def _build_code_context(
        self, file_path: str, source_code: str, error_context: Dict[str, Any]
    ) -> CodeContext:
        """
        Build comprehensive code context for generation.

        Args:
            file_path: Path to target file
            source_code: Source code content
            error_context: Error context information

        Returns:
            Comprehensive code context
        """
        context = CodeContext(target_file=file_path)

        # Detect language and frameworks
        language_info = self.language_detector.detect_language_and_frameworks(
            file_path, source_code
        )

        # Extract imports and dependencies
        context.imports_map = self._extract_imports(
            source_code, language_info.language.value
        )

        # Find related files
        context.related_files = self._find_related_files(
            file_path, context.imports_map, language_info
        )

        # Build call graph if applicable
        if language_info.language.value in ["python", "javascript", "java", "go"]:
            context.call_graph = self._build_call_graph(
                source_code, language_info.language.value
            )

        # Extract class hierarchy
        context.class_hierarchy = self._extract_class_hierarchy(
            source_code, language_info.language.value
        )

        # Extract function signatures
        context.function_signatures = self._extract_function_signatures(
            source_code, language_info.language.value
        )

        # Analyze variable scopes
        context.variable_scopes = self._analyze_variable_scopes(
            source_code, language_info.language.value
        )

        # Detect code patterns
        context.code_patterns = self._detect_code_patterns(source_code)

        # Identify architectural patterns
        context.architectural_patterns = self._identify_architectural_patterns(
            source_code, context.related_files, language_info
        )

        return context

    def _perform_semantic_analysis(
        self, code_context: CodeContext, source_code: str, error_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform deep semantic analysis of the code.

        Args:
            code_context: Code context
            source_code: Source code
            error_context: Error context

        Returns:
            Semantic analysis results
        """
        # Use transformer model for code understanding
        transformer_analysis = self.transformer_analyzer.analyze_code(
            source_code, error_context.get("language", "python")
        )

        # Analyze data flow
        data_flow = self._analyze_data_flow(code_context, source_code)

        # Identify side effects
        side_effects = self._identify_side_effects(code_context, source_code)

        # Analyze error propagation paths
        error_paths = self._analyze_error_propagation(
            code_context, error_context, source_code
        )

        # Determine fix impact
        fix_impact = self._analyze_fix_impact(code_context, error_context, source_code)

        return {
            "transformer_analysis": transformer_analysis,
            "data_flow": data_flow,
            "side_effects": side_effects,
            "error_paths": error_paths,
            "fix_impact": fix_impact,
            "semantic_confidence": self._calculate_semantic_confidence(
                transformer_analysis, data_flow, error_paths
            ),
        }

    def _requires_multi_file_changes(
        self, semantic_analysis: Dict[str, Any], error_context: Dict[str, Any]
    ) -> bool:
        """
        Determine if multi-file changes are required.

        Args:
            semantic_analysis: Semantic analysis results
            error_context: Error context

        Returns:
            True if multi-file changes are needed
        """
        # Check if error involves cross-file dependencies
        if semantic_analysis.get("fix_impact", {}).get("cross_file_impact", False):
            return True

        # Check if error involves interface changes
        if semantic_analysis.get("fix_impact", {}).get("interface_changes", False):
            return True

        # Check if error propagates across files
        error_paths = semantic_analysis.get("error_paths", [])
        for path in error_paths:
            if len(set(node.get("file") for node in path)) > 1:
                return True

        return False

    def _generate_single_file_changes(
        self,
        code_context: CodeContext,
        semantic_analysis: Dict[str, Any],
        error_context: Dict[str, Any],
        source_code: str,
        mode: CodeGenerationMode,
    ) -> GenerationResult:
        """
        Generate changes for a single file with advanced context.

        Args:
            code_context: Code context
            semantic_analysis: Semantic analysis results
            error_context: Error context
            source_code: Source code
            mode: Generation mode

        Returns:
            Generation result
        """
        # Build advanced prompt with semantic understanding
        prompt = self._build_advanced_prompt(
            code_context, semantic_analysis, error_context, source_code, mode
        )

        # Generate fix with appropriate temperature
        temperature = self.temperature_by_mode.get(mode, 0.2)

        # Create LLM request
        request = LLMRequest(
            messages=[LLMMessage(role="user", content=prompt)],
            temperature=temperature,
            max_tokens=4000,
        )

        try:
            # Get LLM response
            response = self.llm_manager.complete(request)

            # Parse and validate response
            parsed_response = self._parse_generation_response(response.content)

            # Apply style preservation
            styled_changes = self._apply_style_preservation(
                parsed_response.get("changes", []), code_context, source_code
            )

            # Validate semantic correctness
            validation_result = self._validate_semantic_correctness(
                styled_changes, code_context, semantic_analysis
            )

            return GenerationResult(
                success=validation_result["valid"],
                mode=mode,
                changes=styled_changes,
                semantic_analysis=semantic_analysis,
                confidence=parsed_response.get("confidence", 0.5),
                reasoning=parsed_response.get("reasoning", ""),
                test_suggestions=parsed_response.get("test_suggestions", []),
                warnings=validation_result.get("warnings", []),
            )

        except Exception as e:
            logger.error(f"Error generating single file changes: {e}")
            return GenerationResult(
                success=False,
                mode=mode,
                changes=[],
                reasoning=f"Generation error: {str(e)}",
                warnings=[str(e)],
            )

    def _generate_multi_file_changes(
        self,
        code_context: CodeContext,
        semantic_analysis: Dict[str, Any],
        error_context: Dict[str, Any],
        mode: CodeGenerationMode,
    ) -> GenerationResult:
        """
        Generate coordinated changes across multiple files.

        Args:
            code_context: Code context
            semantic_analysis: Semantic analysis results
            error_context: Error context
            mode: Generation mode

        Returns:
            Generation result with multi-file changes
        """
        multi_file_changes = {}
        all_warnings = []

        # Identify all files that need changes
        affected_files = self._identify_affected_files(
            code_context, semantic_analysis, error_context
        )

        # Generate changes for each file with coordination
        for file_path in affected_files:
            file_content = self._read_file_content(file_path)
            if not file_content:
                continue

            # Build file-specific context with cross-file awareness
            file_context = self._build_file_specific_context(
                file_path, file_content, code_context, semantic_analysis
            )

            # Generate coordinated changes
            file_changes = self._generate_coordinated_changes(
                file_path,
                file_content,
                file_context,
                code_context,
                semantic_analysis,
                error_context,
                mode,
            )

            if file_changes["changes"]:
                multi_file_changes[file_path] = file_changes["changes"]
                all_warnings.extend(file_changes.get("warnings", []))

        # Validate cross-file consistency
        consistency_result = self._validate_cross_file_consistency(
            multi_file_changes, code_context, semantic_analysis
        )

        return GenerationResult(
            success=consistency_result["valid"] and bool(multi_file_changes),
            mode=CodeGenerationMode.MULTI_FILE,
            changes=[],  # Empty for multi-file mode
            multi_file_changes=multi_file_changes,
            semantic_analysis=semantic_analysis,
            confidence=consistency_result.get("confidence", 0.5),
            reasoning=self._generate_multi_file_reasoning(
                multi_file_changes, semantic_analysis
            ),
            test_suggestions=self._generate_multi_file_test_suggestions(
                multi_file_changes, code_context
            ),
            dependency_updates=self._identify_dependency_updates(
                multi_file_changes, code_context
            ),
            warnings=all_warnings + consistency_result.get("warnings", []),
        )

    def _build_advanced_prompt(
        self,
        code_context: CodeContext,
        semantic_analysis: Dict[str, Any],
        error_context: Dict[str, Any],
        source_code: str,
        mode: CodeGenerationMode,
    ) -> str:
        """
        Build an advanced prompt with semantic understanding.

        Args:
            code_context: Code context
            semantic_analysis: Semantic analysis results
            error_context: Error context
            source_code: Source code
            mode: Generation mode

        Returns:
            Advanced prompt for LLM
        """
        prompt_parts = []

        # System context
        prompt_parts.append(
            f"""You are an expert software engineer with deep understanding of code semantics and architecture.
You are tasked with {mode.value} that requires careful analysis and precise implementation."""
        )

        # Add chain-of-thought if enabled
        if self.use_chain_of_thought:
            prompt_parts.append(
                """
Think through this problem step by step:
1. Understand the root cause and context
2. Analyze the impact of potential changes
3. Design a solution that maintains code quality
4. Ensure the fix is minimal and focused
5. Preserve existing code style and patterns"""
            )

        # Add few-shot examples if enabled
        if self.use_few_shot:
            examples = self._get_few_shot_examples(
                error_context.get("error_type"), error_context.get("language")
            )
            if examples:
                prompt_parts.append("\nHere are some examples of similar fixes:")
                prompt_parts.extend(examples)

        # Add semantic context
        prompt_parts.append(
            f"""
SEMANTIC ANALYSIS:
- Code Understanding Confidence: {semantic_analysis.get('semantic_confidence', 0):.2f}
- Data Flow Analysis: {json.dumps(semantic_analysis.get('data_flow', {}), indent=2)}
- Side Effects Identified: {json.dumps(semantic_analysis.get('side_effects', []), indent=2)}
- Error Propagation Paths: {len(semantic_analysis.get('error_paths', []))} paths identified
- Fix Impact Analysis: {json.dumps(semantic_analysis.get('fix_impact', {}), indent=2)}"""
        )

        # Add code context
        prompt_parts.append(
            f"""
CODE CONTEXT:
- Target File: {code_context.target_file}
- Related Files: {', '.join(code_context.related_files[:5])}
- Imports: {json.dumps(list(code_context.imports_map.keys())[:10], indent=2)}
- Class Hierarchy: {json.dumps(code_context.class_hierarchy, indent=2)}
- Key Functions: {', '.join(list(code_context.function_signatures.keys())[:10])}
- Architectural Patterns: {', '.join(code_context.architectural_patterns)}"""
        )

        # Add the actual error/task
        prompt_parts.append(
            f"""
TASK DETAILS:
Error Type: {error_context.get('error_type', 'Unknown')}
Error Message: {error_context.get('error_message', '')}
File: {error_context.get('file_path', '')}
Line: {error_context.get('line_number', 'Unknown')}

SOURCE CODE:
```{error_context.get('language', 'python')}
{source_code}
```"""
        )

        # Add requirements
        prompt_parts.append(
            """
REQUIREMENTS:
1. Generate a precise fix that addresses the root cause
2. Maintain semantic correctness and program behavior
3. Preserve existing code style and patterns
4. Minimize changes to reduce risk
5. Consider all side effects and dependencies
6. Ensure the fix is compatible with the codebase architecture

OUTPUT FORMAT:
```json
{
    "reasoning": "Step-by-step reasoning for the fix",
    "changes": [
        {
            "line_start": <number>,
            "line_end": <number>,
            "original_code": "exact original code",
            "new_code": "replacement code",
            "explanation": "why this change is necessary"
        }
    ],
    "test_suggestions": ["test cases to verify the fix"],
    "confidence": 0.0-1.0,
    "potential_side_effects": ["list of potential impacts"]
}
```"""
        )

        return "\n".join(prompt_parts)

    def _extract_imports(self, source_code: str, language: str) -> Dict[str, List[str]]:
        """Extract imports from source code."""
        imports = defaultdict(list)

        if language == "python":
            try:
                tree = ast.parse(source_code)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports[alias.name].append("import")
                    elif isinstance(node, ast.ImportFrom):
                        module = node.module or ""
                        for alias in node.names:
                            imports[module].append(alias.name)
            except (SyntaxError, ValueError, AttributeError):
                pass

        elif language in ["javascript", "typescript"]:
            # Simple regex-based extraction
            import_pattern = (
                r'import\s+(?:{[^}]+}|[\w\s,]+)\s+from\s+[\'"]([^\'"]+)[\'"]'
            )
            for match in re.finditer(import_pattern, source_code):
                imports[match.group(1)].append("import")

        return dict(imports)

    def _find_related_files(
        self, file_path: str, imports_map: Dict[str, List[str]], language_info: Any
    ) -> List[str]:
        """Find files related to the target file."""
        related = []
        base_dir = Path(file_path).parent

        # Find imported files
        for import_name in imports_map:
            potential_paths = self._resolve_import_path(
                import_name, base_dir, language_info.language.value
            )
            related.extend(p for p in potential_paths if Path(p).exists())

        # Find files that import this file
        target_module = Path(file_path).stem
        for p in base_dir.rglob(f"*.{Path(file_path).suffix}"):
            if p != Path(file_path):
                try:
                    content = p.read_text()
                    if target_module in content:
                        related.append(str(p))
                except (IOError, UnicodeDecodeError):
                    pass

        return list(set(related))[:10]  # Limit to 10 most relevant

    def _build_call_graph(self, source_code: str, language: str) -> nx.DiGraph:
        """Build a call graph from source code."""
        graph = nx.DiGraph()

        if language == "python":
            try:
                tree = ast.parse(source_code)
                current_function = None

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        current_function = node.name
                        graph.add_node(current_function)
                    elif isinstance(node, ast.Call) and current_function:
                        if isinstance(node.func, ast.Name):
                            graph.add_edge(current_function, node.func.id)
                        elif isinstance(node.func, ast.Attribute):
                            graph.add_edge(current_function, node.func.attr)
            except (SyntaxError, ValueError, AttributeError):
                pass

        return graph

    def _extract_class_hierarchy(
        self, source_code: str, language: str
    ) -> Dict[str, List[str]]:
        """Extract class hierarchy from source code."""
        hierarchy = {}

        if language == "python":
            try:
                tree = ast.parse(source_code)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        bases = []
                        for base in node.bases:
                            if isinstance(base, ast.Name):
                                bases.append(base.id)
                        hierarchy[node.name] = bases
            except (SyntaxError, ValueError, AttributeError):
                pass

        return hierarchy

    def _extract_function_signatures(
        self, source_code: str, language: str
    ) -> Dict[str, str]:
        """Extract function signatures from source code."""
        signatures = {}

        if language == "python":
            try:
                tree = ast.parse(source_code)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        args = []
                        for arg in node.args.args:
                            args.append(arg.arg)
                        signatures[node.name] = f"({', '.join(args)})"
            except (SyntaxError, ValueError, AttributeError):
                pass

        return signatures

    def _analyze_variable_scopes(
        self, source_code: str, language: str
    ) -> Dict[str, Dict[str, Any]]:
        """Analyze variable scopes in the code."""
        scopes = defaultdict(dict)

        if language == "python":
            try:
                tree = ast.parse(source_code)
                self._analyze_python_scopes(tree, scopes)
            except (SyntaxError, ValueError, AttributeError):
                pass

        return dict(scopes)

    def _analyze_python_scopes(self, node, scopes, current_scope="global"):
        """Recursively analyze Python variable scopes."""
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef):
                new_scope = f"{current_scope}.{child.name}"
                scopes[new_scope] = {
                    "type": "function",
                    "parent": current_scope,
                    "variables": [],
                }
                self._analyze_python_scopes(child, scopes, new_scope)
            elif isinstance(child, ast.ClassDef):
                new_scope = f"{current_scope}.{child.name}"
                scopes[new_scope] = {
                    "type": "class",
                    "parent": current_scope,
                    "variables": [],
                }
                self._analyze_python_scopes(child, scopes, new_scope)
            elif isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Name):
                        scopes[current_scope].setdefault("variables", []).append(
                            target.id
                        )
            else:
                self._analyze_python_scopes(child, scopes, current_scope)

    def _detect_code_patterns(self, source_code: str) -> List[str]:
        """Detect common code patterns."""
        patterns = []

        # Check for common patterns
        pattern_checks = {
            "singleton": r"class\s+\w+.*:\s*\n\s*_instance\s*=\s*None",
            "factory": r"def\s+create_\w+|class\s+\w+Factory",
            "observer": r"def\s+notify|def\s+subscribe|def\s+observe",
            "decorator": r"@\w+|def\s+\w+\(.*\):\s*\n\s*def\s+wrapper",
            "context_manager": r"def\s+__enter__|def\s+__exit__",
            "iterator": r"def\s+__iter__|def\s+__next__",
            "async_await": r"async\s+def|await\s+",
            "generator": r"yield\s+",
            "comprehension": r"\[.*for.*in.*\]|\{.*for.*in.*\}",
            "error_handling": r"try:.*except",
        }

        for pattern_name, pattern_regex in pattern_checks.items():
            if re.search(pattern_regex, source_code, re.MULTILINE | re.DOTALL):
                patterns.append(pattern_name)

        return patterns

    def _identify_architectural_patterns(
        self, source_code: str, related_files: List[str], language_info: Any
    ) -> List[str]:
        """Identify architectural patterns in the codebase."""
        patterns = []

        # Check for MVC pattern
        if any("model" in f.lower() for f in related_files):
            if any("view" in f.lower() for f in related_files):
                if any("controller" in f.lower() for f in related_files):
                    patterns.append("MVC")

        # Check for layered architecture
        layers = ["presentation", "business", "data", "service", "repository"]
        if sum(any(layer in f.lower() for f in related_files) for layer in layers) >= 2:
            patterns.append("Layered Architecture")

        # Check for microservices
        if (
            any("service" in f.lower() for f in related_files)
            and len(related_files) > 5
        ):
            patterns.append("Microservices")

        # Framework-specific patterns
        for framework in language_info.frameworks:
            if framework.name.lower() == "django":
                patterns.append("Django MVT")
            elif framework.name.lower() == "flask":
                patterns.append("Flask Blueprints")
            elif framework.name.lower() == "spring":
                patterns.append("Spring MVC")

        return patterns

    def _analyze_data_flow(
        self, code_context: CodeContext, source_code: str
    ) -> Dict[str, Any]:
        """Analyze data flow in the code."""
        data_flow = {
            "inputs": [],
            "outputs": [],
            "transformations": [],
            "external_calls": [],
        }

        # This is a simplified analysis - in production, use proper data flow analysis
        # Look for function parameters (inputs)
        for func_name, signature in code_context.function_signatures.items():
            if signature != "()":
                data_flow["inputs"].append(
                    {"function": func_name, "parameters": signature}
                )

        # Look for return statements (outputs)
        return_pattern = r"return\s+(.+?)(?:\n|$)"
        for match in re.finditer(return_pattern, source_code):
            data_flow["outputs"].append(match.group(1).strip())

        # Look for external calls
        for import_module in code_context.imports_map:
            data_flow["external_calls"].append(import_module)

        return data_flow

    def _identify_side_effects(
        self, code_context: CodeContext, source_code: str
    ) -> List[Dict[str, Any]]:
        """Identify potential side effects in the code."""
        side_effects = []

        # File I/O operations
        io_patterns = [
            (r"open\s*\(", "file_io"),
            (r"\.write\s*\(", "file_write"),
            (r"\.read\s*\(", "file_read"),
        ]

        # Database operations
        db_patterns = [
            (r"\.execute\s*\(", "database_operation"),
            (r"\.save\s*\(", "database_save"),
            (r"\.delete\s*\(", "database_delete"),
        ]

        # Network operations
        network_patterns = [
            (r"requests\.\w+\s*\(", "http_request"),
            (r"urllib.*urlopen\s*\(", "url_request"),
            (r"socket\.\w+\s*\(", "socket_operation"),
        ]

        # State modifications
        state_patterns = [
            (r"global\s+\w+", "global_state_modification"),
            (r"self\.\w+\s*=", "instance_state_modification"),
            (r"cls\.\w+\s*=", "class_state_modification"),
        ]

        all_patterns = io_patterns + db_patterns + network_patterns + state_patterns

        for pattern, effect_type in all_patterns:
            matches = re.finditer(pattern, source_code)
            for match in matches:
                line_no = source_code[: match.start()].count("\n") + 1
                side_effects.append(
                    {"type": effect_type, "line": line_no, "code": match.group(0)}
                )

        return side_effects

    def _analyze_error_propagation(
        self, code_context: CodeContext, error_context: Dict[str, Any], source_code: str
    ) -> List[List[Dict[str, Any]]]:
        """Analyze how errors propagate through the code."""
        propagation_paths = []

        # Get error location
        error_line = error_context.get("line_number", 0)
        if not error_line:
            return propagation_paths

        # Find the function containing the error
        error_function = None
        if code_context.call_graph:
            # Simple heuristic: find function that contains the error line
            lines = source_code.split("\n")
            current_function = None
            for i, line in enumerate(lines):
                if re.match(r"def\s+(\w+)", line):
                    match = re.match(r"def\s+(\w+)", line)
                    current_function = match.group(1)
                if i + 1 == error_line:
                    error_function = current_function
                    break

        if error_function and code_context.call_graph:
            # Find all paths from error function
            try:
                for node in code_context.call_graph.nodes():
                    if node != error_function:
                        paths = list(
                            nx.all_simple_paths(
                                code_context.call_graph, error_function, node, cutoff=3
                            )
                        )
                        for path in paths:
                            propagation_paths.append(
                                [
                                    {"function": f, "file": code_context.target_file}
                                    for f in path
                                ]
                            )
            except (nx.NetworkXError, AttributeError, KeyError):
                pass

        return propagation_paths[:5]  # Limit to 5 paths

    def _analyze_fix_impact(
        self, code_context: CodeContext, error_context: Dict[str, Any], source_code: str
    ) -> Dict[str, Any]:
        """Analyze the potential impact of fixing the error."""
        impact = {
            "affected_functions": [],
            "affected_classes": [],
            "cross_file_impact": False,
            "interface_changes": False,
            "breaking_changes": False,
            "test_impact": [],
        }

        # Identify affected functions based on call graph
        if code_context.call_graph:
            # Find functions that might be affected
            for func in code_context.function_signatures:
                if func in code_context.call_graph:
                    impact["affected_functions"].append(func)

        # Check if error is in a public interface
        error_in_public_interface = False
        if error_context.get("error_type") in ["AttributeError", "TypeError"]:
            # Check if error involves public methods/attributes
            if not any(
                name.startswith("_")
                for name in error_context.get("root_cause", "").split(".")
            ):
                error_in_public_interface = True
                impact["interface_changes"] = True

        # Check for cross-file impact
        if len(code_context.related_files) > 0:
            impact["cross_file_impact"] = error_in_public_interface

        # Assess breaking changes
        if impact["interface_changes"]:
            impact["breaking_changes"] = True

        return impact

    def _calculate_semantic_confidence(
        self,
        transformer_analysis: Dict[str, Any],
        data_flow: Dict[str, Any],
        error_paths: List[Any],
    ) -> float:
        """Calculate confidence in semantic understanding."""
        confidence_factors = []

        # Transformer analysis confidence
        if transformer_analysis:
            confidence_factors.append(transformer_analysis.get("confidence", 0.5))

        # Data flow completeness
        data_flow_score = 0.0
        if data_flow.get("inputs"):
            data_flow_score += 0.25
        if data_flow.get("outputs"):
            data_flow_score += 0.25
        if data_flow.get("transformations"):
            data_flow_score += 0.25
        if data_flow.get("external_calls"):
            data_flow_score += 0.25
        confidence_factors.append(data_flow_score)

        # Error path clarity
        if error_paths:
            path_score = min(1.0, len(error_paths) / 3.0)
            confidence_factors.append(path_score)

        # Calculate overall confidence
        if confidence_factors:
            return sum(confidence_factors) / len(confidence_factors)
        return 0.5

    def _parse_generation_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM generation response."""
        try:
            # Try to extract JSON from response
            json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))

            # Try direct JSON parsing
            return json.loads(response)
        except (json.JSONDecodeError, ValueError, AttributeError):
            # Fallback parsing
            return {"changes": [], "reasoning": response, "confidence": 0.3}

    def _apply_style_preservation(
        self, changes: List[Dict[str, Any]], code_context: CodeContext, source_code: str
    ) -> List[Dict[str, Any]]:
        """Apply style preservation to generated changes."""
        # Get style conventions
        language = self.language_detector.detect_language_and_frameworks(
            code_context.target_file, source_code
        ).language.value

        style_conventions = self.style_analyzer.analyze_file_style(
            code_context.target_file, language, source_code
        )

        # Apply style to each change
        styled_changes = []
        for change in changes:
            styled_change = change.copy()
            if "new_code" in styled_change:
                styled_change["new_code"] = self.style_analyzer.format_code_to_style(
                    styled_change["new_code"], style_conventions, language
                )
            styled_changes.append(styled_change)

        return styled_changes

    def _validate_semantic_correctness(
        self,
        changes: List[Dict[str, Any]],
        code_context: CodeContext,
        semantic_analysis: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Validate semantic correctness of generated changes."""
        validation_result = {"valid": True, "warnings": []}

        # Check if changes preserve function signatures
        for change in changes:
            if "def " in change.get("original_code", ""):
                # Check if function signature is preserved
                original_sig = re.search(
                    r"def\s+(\w+)\s*\((.*?)\)", change.get("original_code", "")
                )
                new_sig = re.search(
                    r"def\s+(\w+)\s*\((.*?)\)", change.get("new_code", "")
                )

                if original_sig and new_sig:
                    if original_sig.group(1) != new_sig.group(1):
                        validation_result["warnings"].append(
                            f"Function name changed: {original_sig.group(1)} -> {new_sig.group(1)}"
                        )
                    if original_sig.group(2) != new_sig.group(2):
                        validation_result["warnings"].append(
                            f"Function signature changed for {original_sig.group(1)}"
                        )

        # Check if changes introduce new side effects
        new_side_effects = []
        for change in changes:
            new_code = change.get("new_code", "")
            # Simple check for new I/O operations
            if "open(" in new_code and "open(" not in change.get("original_code", ""):
                new_side_effects.append("File I/O")
            if "requests." in new_code and "requests." not in change.get(
                "original_code", ""
            ):
                new_side_effects.append("Network request")

        if new_side_effects:
            validation_result["warnings"].append(
                f"New side effects introduced: {', '.join(new_side_effects)}"
            )

        # Validate against semantic analysis results
        if semantic_analysis.get("semantic_confidence", 0) < self.confidence_threshold:
            validation_result["warnings"].append(
                f"Low semantic confidence: {semantic_analysis.get('semantic_confidence', 0):.2f}"
            )

        return validation_result

    def _identify_affected_files(
        self,
        code_context: CodeContext,
        semantic_analysis: Dict[str, Any],
        error_context: Dict[str, Any],
    ) -> List[str]:
        """Identify all files affected by the change."""
        affected = set([code_context.target_file])

        # Add files based on import relationships
        affected.update(code_context.related_files)

        # Add files based on error propagation
        for path in semantic_analysis.get("error_paths", []):
            for node in path:
                if "file" in node:
                    affected.add(node["file"])

        # Add files based on fix impact
        if semantic_analysis.get("fix_impact", {}).get("cross_file_impact"):
            # Add all related files if cross-file impact is detected
            affected.update(code_context.related_files)

        return list(affected)

    def _read_file_content(self, file_path: str) -> Optional[str]:
        """Read file content safely."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.warning(f"Could not read file {file_path}: {e}")
            return None

    def _build_file_specific_context(
        self,
        file_path: str,
        file_content: str,
        code_context: CodeContext,
        semantic_analysis: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build context specific to a file for multi-file changes."""
        return {
            "file_path": file_path,
            "imports_from_target": self._find_imports_from_target(
                file_content, code_context.target_file
            ),
            "exports_to_target": self._find_exports_to_target(
                file_content, code_context.target_file
            ),
            "shared_dependencies": self._find_shared_dependencies(
                file_path, code_context
            ),
            "change_requirements": self._determine_change_requirements(
                file_path, code_context, semantic_analysis
            ),
        }

    def _find_imports_from_target(
        self, file_content: str, target_file: str
    ) -> List[str]:
        """Find what this file imports from the target file."""
        imports = []
        target_module = Path(target_file).stem

        # Python imports
        import_patterns = [
            rf"from\s+\S*{target_module}\s+import\s+(\w+)",
            rf"import\s+\S*{target_module}",
        ]

        for pattern in import_patterns:
            matches = re.finditer(pattern, file_content)
            for match in matches:
                imports.append(match.group(0))

        return imports

    def _find_exports_to_target(self, file_content: str, target_file: str) -> List[str]:
        """Find what this file exports that the target might use."""
        # This is a simplified implementation
        exports = []

        # Find class and function definitions
        definition_patterns = [
            r"^class\s+(\w+)",
            r"^def\s+(\w+)",
            r"^(\w+)\s*=",  # Module-level variables
        ]

        for pattern in definition_patterns:
            matches = re.finditer(pattern, file_content, re.MULTILINE)
            for match in matches:
                exports.append(match.group(1))

        return exports

    def _find_shared_dependencies(
        self, file_path: str, code_context: CodeContext
    ) -> List[str]:
        """Find dependencies shared between files."""
        shared = []

        # Compare imports
        file_content = self._read_file_content(file_path)
        if file_content:
            file_imports = self._extract_imports(
                file_content,
                self.language_detector.detect_language_and_frameworks(
                    file_path, file_content
                ).language.value,
            )

            for imp in file_imports:
                if imp in code_context.imports_map:
                    shared.append(imp)

        return shared

    def _determine_change_requirements(
        self,
        file_path: str,
        code_context: CodeContext,
        semantic_analysis: Dict[str, Any],
    ) -> List[str]:
        """Determine what changes are required in this file."""
        requirements = []

        # Check if file imports changed interfaces
        if semantic_analysis.get("fix_impact", {}).get("interface_changes"):
            requirements.append("update_imports")
            requirements.append("update_method_calls")

        # Check if file extends changed classes
        file_content = self._read_file_content(file_path)
        if file_content:
            for class_name, bases in code_context.class_hierarchy.items():
                if class_name in file_content:
                    requirements.append("update_inheritance")

        return requirements

    def _generate_coordinated_changes(
        self,
        file_path: str,
        file_content: str,
        file_context: Dict[str, Any],
        code_context: CodeContext,
        semantic_analysis: Dict[str, Any],
        error_context: Dict[str, Any],
        mode: CodeGenerationMode,
    ) -> Dict[str, Any]:
        """Generate changes for a file with coordination awareness."""
        # Build coordinated prompt
        prompt = f"""Generate coordinated changes for {file_path} based on the following context:

ORIGINAL ERROR:
{json.dumps(error_context, indent=2)}

FILE CONTEXT:
{json.dumps(file_context, indent=2)}

COORDINATION REQUIREMENTS:
- This file imports from target: {file_context['imports_from_target']}
- This file exports to target: {file_context['exports_to_target']}
- Change requirements: {file_context['change_requirements']}

FILE CONTENT:
```
{file_content}
```

Generate minimal changes that maintain consistency with the fix in the target file.

OUTPUT FORMAT:
```json
{{
    "changes": [
        {{
            "line_start": <number>,
            "line_end": <number>,
            "original_code": "exact original code",
            "new_code": "replacement code",
            "explanation": "why this change is necessary"
        }}
    ],
    "warnings": ["any potential issues"]
}}
```"""

        # Get LLM response
        try:
            request = LLMRequest(
                messages=[LLMMessage(role="user", content=prompt)],
                temperature=0.1,
                max_tokens=2000,
            )
            response = self.llm_manager.complete(request)

            result = self._parse_generation_response(response.content)
            return {
                "changes": result.get("changes", []),
                "warnings": result.get("warnings", []),
            }
        except Exception as e:
            logger.error(f"Error generating coordinated changes for {file_path}: {e}")
            return {"changes": [], "warnings": [str(e)]}

    def _validate_cross_file_consistency(
        self,
        multi_file_changes: Dict[str, List[Dict[str, Any]]],
        code_context: CodeContext,
        semantic_analysis: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Validate consistency across multi-file changes."""
        validation_result = {"valid": True, "warnings": [], "confidence": 0.8}

        # Check for conflicting changes
        all_changes = []
        for file_path, changes in multi_file_changes.items():
            for change in changes:
                all_changes.append({"file": file_path, "change": change})

        # Validate interface consistency
        interface_changes = {}
        for change_info in all_changes:
            change = change_info["change"]
            # Look for function/class definition changes
            if "def " in change.get("original_code", "") or "class " in change.get(
                "original_code", ""
            ):
                # Extract interface changes
                original_match = re.search(
                    r"(?:def|class)\s+(\w+)", change.get("original_code", "")
                )
                new_match = re.search(
                    r"(?:def|class)\s+(\w+)", change.get("new_code", "")
                )

                if original_match and new_match:
                    if original_match.group(1) != new_match.group(1):
                        interface_changes[original_match.group(1)] = new_match.group(1)

        # Check if all references are updated
        for old_name, new_name in interface_changes.items():
            for file_path, changes in multi_file_changes.items():
                file_content = self._read_file_content(file_path)
                if file_content and old_name in file_content:
                    # Check if there's a corresponding change
                    has_update = any(
                        old_name in change.get("original_code", "")
                        and new_name in change.get("new_code", "")
                        for change in changes
                    )
                    if not has_update:
                        validation_result["warnings"].append(
                            f"File {file_path} may need updates for {old_name} -> {new_name}"
                        )
                        validation_result["confidence"] *= 0.9

        return validation_result

    def _generate_multi_file_reasoning(
        self,
        multi_file_changes: Dict[str, List[Dict[str, Any]]],
        semantic_analysis: Dict[str, Any],
    ) -> str:
        """Generate reasoning for multi-file changes."""
        reasoning_parts = []

        reasoning_parts.append(
            f"Generated coordinated changes across {len(multi_file_changes)} files:"
        )

        for file_path, changes in multi_file_changes.items():
            reasoning_parts.append(f"\n{file_path}:")
            for i, change in enumerate(changes, 1):
                reasoning_parts.append(
                    f"  {i}. {change.get('explanation', 'Update required')}"
                )

        if semantic_analysis.get("fix_impact", {}).get("interface_changes"):
            reasoning_parts.append(
                "\nInterface changes were detected and propagated across all dependent files."
            )

        if semantic_analysis.get("fix_impact", {}).get("breaking_changes"):
            reasoning_parts.append(
                "\nWARNING: These changes may be breaking. Update all external dependencies."
            )

        return "\n".join(reasoning_parts)

    def _generate_multi_file_test_suggestions(
        self,
        multi_file_changes: Dict[str, List[Dict[str, Any]]],
        code_context: CodeContext,
    ) -> List[str]:
        """Generate test suggestions for multi-file changes."""
        suggestions = []

        # Integration tests
        suggestions.append("Add integration tests to verify cross-file functionality")

        # Interface tests
        if any(
            "def " in str(changes) or "class " in str(changes)
            for changes in multi_file_changes.values()
        ):
            suggestions.append("Test all public interfaces to ensure compatibility")

        # Regression tests
        suggestions.append("Add regression tests for the original error scenario")

        # File-specific tests
        for file_path in multi_file_changes:
            file_name = Path(file_path).stem
            suggestions.append(f"Update tests for {file_name} module")

        return suggestions

    def _identify_dependency_updates(
        self,
        multi_file_changes: Dict[str, List[Dict[str, Any]]],
        code_context: CodeContext,
    ) -> Dict[str, Any]:
        """Identify required dependency updates."""
        updates = {"imports": [], "exports": [], "version_bumps": []}

        # Check for new imports
        for file_path, changes in multi_file_changes.items():
            for change in changes:
                new_code = change.get("new_code", "")
                original_code = change.get("original_code", "")

                # Check for new imports
                new_imports = re.findall(r"import\s+(\w+)", new_code)
                original_imports = re.findall(r"import\s+(\w+)", original_code)

                for imp in new_imports:
                    if imp not in original_imports:
                        updates["imports"].append({"file": file_path, "import": imp})

        # Suggest version bumps for interface changes
        if any(
            "class " in str(changes) or "def " in str(changes)
            for changes in multi_file_changes.values()
        ):
            updates["version_bumps"].append(
                {"type": "minor", "reason": "Interface changes detected"}
            )

        return updates

    def _get_few_shot_examples(self, error_type: str, language: str) -> List[str]:
        """Get few-shot examples for the given error type and language."""
        examples = []

        # This would be expanded with a proper example database
        example_map = {
            ("NameError", "python"): [
                """Example: NameError in Python
Original: result = undefined_var + 10
Fix: result = 0 + 10  # or define undefined_var before use"""
            ],
            ("TypeError", "python"): [
                """Example: TypeError in Python
Original: total = "10" + 5
Fix: total = int("10") + 5  # Convert string to int"""
            ],
            ("AttributeError", "python"): [
                """Example: AttributeError in Python
Original: value = obj.non_existent_attr
Fix: value = getattr(obj, 'non_existent_attr', default_value)  # Safe access with default"""
            ],
        }

        key = (error_type, language)
        if key in example_map:
            examples.extend(example_map[key])

        return examples

    def _resolve_import_path(
        self, import_name: str, base_dir: Path, language: str
    ) -> List[str]:
        """Resolve import name to potential file paths."""
        paths = []

        if language == "python":
            # Convert module name to path
            module_path = import_name.replace(".", "/")
            potential_paths = [
                base_dir / f"{module_path}.py",
                base_dir / module_path / "__init__.py",
                base_dir.parent / f"{module_path}.py",
                base_dir.parent / module_path / "__init__.py",
            ]
            paths.extend(str(p) for p in potential_paths if p.exists())

        elif language in ["javascript", "typescript"]:
            # Handle various import styles
            potential_paths = [
                base_dir / f"{import_name}.js",
                base_dir / f"{import_name}.ts",
                base_dir / f"{import_name}/index.js",
                base_dir / f"{import_name}/index.ts",
                base_dir / import_name,
            ]
            paths.extend(str(p) for p in potential_paths if p.exists())

        return paths


def create_advanced_code_generator(
    llm_manager: LLMManager,
    context_manager: LLMContextManager,
    config: Optional[Dict[str, Any]] = None,
) -> AdvancedCodeGenerator:
    """
    Factory function to create an advanced code generator.

    Args:
        llm_manager: LLM manager instance
        context_manager: Context manager instance
        config: Configuration dictionary

    Returns:
        Configured advanced code generator
    """
    return AdvancedCodeGenerator(llm_manager, context_manager, config)
