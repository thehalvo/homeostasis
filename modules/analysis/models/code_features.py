"""
Advanced feature extraction for code analysis in Homeostasis.

This module provides sophisticated feature extraction capabilities for:
- Multi-language code analysis
- Error pattern recognition
- Code complexity metrics
- Semantic embeddings
- Context-aware features
"""

import ast
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Language-specific imports
try:
    pass  # tree_sitter not actually used
    TREE_SITTER_AVAILABLE = False
except ImportError:
    TREE_SITTER_AVAILABLE = False

try:
    import torch
    from transformers import AutoModel, AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


@dataclass
class CodeFeatures:
    """Container for extracted code features."""

    # Basic features
    language: str
    file_path: str
    line_number: int
    function_name: str
    class_name: Optional[str]

    # Complexity metrics
    cyclomatic_complexity: int
    cognitive_complexity: int
    lines_of_code: int
    nesting_depth: int

    # Error-specific features
    error_type: str
    error_message: str
    error_context: str
    stack_depth: int

    # Code structure features
    ast_features: Dict[str, Any]
    token_features: Dict[str, int]
    dependency_features: Dict[str, Any]

    # Semantic features
    embeddings: Optional[np.ndarray] = None
    semantic_similarity: Optional[float] = None

    # Pattern features
    pattern_matches: Dict[str, bool] = field(default_factory=dict)
    anti_patterns: List[str] = field(default_factory=list)

    # Context features
    surrounding_code: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    globals: List[str] = field(default_factory=list)
    locals: Dict[str, Any] = field(default_factory=dict)

    # Framework-specific features
    framework: Optional[str] = None
    framework_features: Dict[str, Any] = field(default_factory=dict)


class LanguageAnalyzer:
    """Base class for language-specific analysis."""

    def __init__(self, language: str):
        self.language = language
        self.parser = self._initialize_parser()

    def _initialize_parser(self):
        """Initialize language-specific parser."""
        return None

    def extract_ast_features(self, code: str) -> Dict[str, Any]:
        """Extract AST-based features from code."""
        raise NotImplementedError

    def calculate_complexity(self, code: str) -> Dict[str, int]:
        """Calculate various complexity metrics."""
        raise NotImplementedError

    def extract_patterns(self, code: str) -> Dict[str, bool]:
        """Extract language-specific patterns."""
        raise NotImplementedError


class PythonAnalyzer(LanguageAnalyzer):
    """Python-specific code analyzer."""

    def __init__(self):
        super().__init__("python")

    def extract_ast_features(self, code: str) -> Dict[str, Any]:
        """Extract Python AST features."""
        try:
            tree = ast.parse(code)

            features = {
                "node_types": Counter(),
                "function_count": 0,
                "class_count": 0,
                "import_count": 0,
                "loop_count": 0,
                "conditional_count": 0,
                "exception_count": 0,
                "decorator_count": 0,
                "comprehension_count": 0,
                "lambda_count": 0,
                "async_count": 0,
            }

            for node in ast.walk(tree):
                node_type = type(node).__name__
                features["node_types"][node_type] += 1

                if isinstance(node, ast.FunctionDef):
                    features["function_count"] += 1
                    if node.decorator_list:
                        features["decorator_count"] += len(node.decorator_list)
                elif isinstance(node, ast.AsyncFunctionDef):
                    features["function_count"] += 1
                    features["async_count"] += 1
                elif isinstance(node, ast.ClassDef):
                    features["class_count"] += 1
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    features["import_count"] += 1
                elif isinstance(node, (ast.For, ast.While, ast.AsyncFor)):
                    features["loop_count"] += 1
                elif isinstance(node, (ast.If, ast.IfExp)):
                    features["conditional_count"] += 1
                elif isinstance(node, (ast.Try, ast.ExceptHandler)):
                    features["exception_count"] += 1
                elif isinstance(
                    node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
                ):
                    features["comprehension_count"] += 1
                elif isinstance(node, ast.Lambda):
                    features["lambda_count"] += 1
                elif isinstance(node, (ast.Await, ast.AsyncWith)):
                    features["async_count"] += 1

            # Convert Counter to dict for JSON serialization
            features["node_types"] = dict(features["node_types"])

            return features

        except SyntaxError:
            return {"parse_error": True, "node_types": {}}

    def calculate_complexity(self, code: str) -> Dict[str, int]:
        """Calculate Python code complexity metrics."""
        try:
            tree = ast.parse(code)

            # Cyclomatic complexity
            cyclomatic = 1  # Base complexity

            # Cognitive complexity
            cognitive = 0
            # TODO: nesting_level may be needed for future cognitive complexity calculations
            # nesting_level = 0

            # Maximum nesting depth
            max_nesting = 0

            class ComplexityVisitor(ast.NodeVisitor):
                def __init__(self):
                    self.nesting_stack = []

                def visit_If(self, node):
                    nonlocal cyclomatic, cognitive, max_nesting
                    cyclomatic += 1
                    cognitive += 1 + len(self.nesting_stack)
                    self.nesting_stack.append("if")
                    max_nesting = max(max_nesting, len(self.nesting_stack))
                    self.generic_visit(node)
                    self.nesting_stack.pop()

                def visit_For(self, node):
                    nonlocal cyclomatic, cognitive, max_nesting
                    cyclomatic += 1
                    cognitive += 1 + len(self.nesting_stack)
                    self.nesting_stack.append("for")
                    max_nesting = max(max_nesting, len(self.nesting_stack))
                    self.generic_visit(node)
                    self.nesting_stack.pop()

                def visit_While(self, node):
                    nonlocal cyclomatic, cognitive, max_nesting
                    cyclomatic += 1
                    cognitive += 1 + len(self.nesting_stack)
                    self.nesting_stack.append("while")
                    max_nesting = max(max_nesting, len(self.nesting_stack))
                    self.generic_visit(node)
                    self.nesting_stack.pop()

                def visit_ExceptHandler(self, node):
                    nonlocal cyclomatic, cognitive
                    cyclomatic += 1
                    cognitive += 1
                    self.generic_visit(node)

                def visit_BoolOp(self, node):
                    nonlocal cyclomatic, cognitive
                    # Each additional boolean operator adds complexity
                    cyclomatic += len(node.values) - 1
                    cognitive += len(node.values) - 1
                    self.generic_visit(node)

            visitor = ComplexityVisitor()
            visitor.visit(tree)

            # Count lines
            lines_of_code = len(
                [
                    line
                    for line in code.split("\n")
                    if line.strip() and not line.strip().startswith("#")
                ]
            )

            return {
                "cyclomatic_complexity": cyclomatic,
                "cognitive_complexity": cognitive,
                "lines_of_code": lines_of_code,
                "nesting_depth": max_nesting,
            }

        except SyntaxError:
            return {
                "cyclomatic_complexity": 0,
                "cognitive_complexity": 0,
                "lines_of_code": len(code.split("\n")),
                "nesting_depth": 0,
            }

    def extract_patterns(self, code: str) -> Dict[str, bool]:
        """Extract Python-specific patterns and anti-patterns."""
        patterns = {
            # Common patterns
            "uses_type_hints": bool(re.search(r"def\s+\w+\s*\([^)]*:\s*\w+", code)),
            "uses_f_strings": bool(re.search(r'f["\']', code)),
            "uses_comprehensions": bool(
                re.search(r"\[.*for.*in.*\]|\{.*for.*in.*\}", code)
            ),
            "uses_context_managers": bool(re.search(r"with\s+", code)),
            "uses_decorators": bool(re.search(r"@\w+", code)),
            "uses_generators": bool(re.search(r"yield\s+", code)),
            "uses_async": bool(re.search(r"async\s+def|await\s+", code)),
            # Anti-patterns
            "bare_except": bool(re.search(r"except\s*:", code)),
            "mutable_default_args": bool(
                re.search(r"def\s+\w+\s*\([^)]*=\s*(\[|\{)", code)
            ),
            "global_usage": bool(re.search(r"global\s+", code)),
            "eval_usage": bool(re.search(r"eval\s*\(", code)),
            "exec_usage": bool(re.search(r"exec\s*\(", code)),
            "star_imports": bool(re.search(r"from\s+\w+\s+import\s+\*", code)),
            "long_lines": any(len(line) > 120 for line in code.split("\n")),
            "todo_comments": bool(
                re.search(r"#\s*(TODO|FIXME|HACK)", code, re.IGNORECASE)
            ),
        }

        return patterns


class JavaScriptAnalyzer(LanguageAnalyzer):
    """JavaScript/TypeScript code analyzer."""

    def __init__(self):
        super().__init__("javascript")

    def extract_ast_features(self, code: str) -> Dict[str, Any]:
        """Extract JavaScript AST features using regex patterns."""
        features = {
            "function_count": len(
                re.findall(r"function\s+\w+\s*\(|const\s+\w+\s*=\s*\(.*?\)\s*=>", code)
            ),
            "class_count": len(re.findall(r"class\s+\w+", code)),
            "import_count": len(re.findall(r"import\s+.*from|require\s*\(", code)),
            "async_count": len(re.findall(r"async\s+", code)),
            "promise_count": len(
                re.findall(r"\.then\s*\(|\.catch\s*\(|Promise\.", code)
            ),
            "arrow_function_count": len(re.findall(r"=>", code)),
            "destructuring_count": len(
                re.findall(r"const\s*\{.*?\}|let\s*\{.*?\}|var\s*\{.*?\}", code)
            ),
        }

        return features

    def calculate_complexity(self, code: str) -> Dict[str, int]:
        """Calculate JavaScript code complexity."""
        # Simplified complexity calculation using pattern matching
        cyclomatic = 1
        cyclomatic += len(re.findall(r"\bif\s*\(", code))
        cyclomatic += len(re.findall(r"\belse\s+if\s*\(", code))
        cyclomatic += len(re.findall(r"\bfor\s*\(", code))
        cyclomatic += len(re.findall(r"\bwhile\s*\(", code))
        cyclomatic += len(re.findall(r"\bcatch\s*\(", code))
        cyclomatic += len(re.findall(r"\bcase\s+", code))
        cyclomatic += len(re.findall(r"\?\s*.*\s*:", code))  # Ternary operators

        lines_of_code = len(
            [
                line
                for line in code.split("\n")
                if line.strip() and not line.strip().startswith("//")
            ]
        )

        # Estimate nesting depth
        max_nesting = 0
        current_nesting = 0
        for char in code:
            if char == "{":
                current_nesting += 1
                max_nesting = max(max_nesting, current_nesting)
            elif char == "}":
                current_nesting = max(0, current_nesting - 1)

        return {
            "cyclomatic_complexity": cyclomatic,
            "cognitive_complexity": cyclomatic * 2,  # Rough estimate
            "lines_of_code": lines_of_code,
            "nesting_depth": min(max_nesting, 10),  # Cap at 10
        }

    def extract_patterns(self, code: str) -> Dict[str, bool]:
        """Extract JavaScript patterns."""
        patterns = {
            # Modern patterns
            "uses_const_let": bool(re.search(r"\b(const|let)\s+", code)),
            "uses_arrow_functions": bool(re.search(r"=>", code)),
            "uses_template_literals": bool(re.search(r"`[^`]*\$\{", code)),
            "uses_destructuring": bool(
                re.search(r"const\s*\{.*?\}|let\s*\{.*?\}", code)
            ),
            "uses_spread_operator": bool(re.search(r"\.\.\.", code)),
            "uses_async_await": bool(re.search(r"async\s+.*await\s+", code, re.DOTALL)),
            "uses_classes": bool(re.search(r"class\s+\w+", code)),
            # Anti-patterns
            "uses_var": bool(re.search(r"\bvar\s+", code)),
            "uses_eval": bool(re.search(r"eval\s*\(", code)),
            "uses_with": bool(re.search(r"\bwith\s*\(", code)),
            "callback_hell": code.count("})") > 5 and "}).then(" in code,
            "no_semicolons": ";" not in code and len(code) > 100,
            "console_log": bool(re.search(r"console\.(log|error|warn)", code)),
        }

        return patterns


class MultiLanguageFeatureExtractor:
    """Extract features from code in multiple languages."""

    def __init__(self):
        self.analyzers = {
            "python": PythonAnalyzer(),
            "javascript": JavaScriptAnalyzer(),
            "typescript": JavaScriptAnalyzer(),  # Reuse JS analyzer
        }

        # Initialize semantic model if available
        self.semantic_model = None
        self.tokenizer = None
        if TRANSFORMERS_AVAILABLE:
            try:
                # Use specific revision for security and reproducibility
                model_revision = "1b2e0bfe5003709471fb6e04c0943470cf4a5b30"
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "microsoft/codebert-base", revision=model_revision
                )
                self.semantic_model = AutoModel.from_pretrained(
                    "microsoft/codebert-base", revision=model_revision
                )
                self.semantic_model.eval()
            except Exception as e:
                print(f"Failed to load CodeBERT model: {e}")

    def detect_language(self, code: str, file_path: Optional[str] = None) -> str:
        """Detect the programming language of the code."""
        if file_path:
            ext = Path(file_path).suffix.lower()
            language_map = {
                ".py": "python",
                ".js": "javascript",
                ".jsx": "javascript",
                ".ts": "typescript",
                ".tsx": "typescript",
                ".java": "java",
                ".cpp": "cpp",
                ".c": "c",
                ".cs": "csharp",
                ".go": "go",
                ".rb": "ruby",
                ".php": "php",
                ".rs": "rust",
                ".swift": "swift",
                ".kt": "kotlin",
                ".scala": "scala",
                ".r": "r",
                ".m": "matlab",
                ".jl": "julia",
            }
            if ext in language_map:
                return language_map[ext]

        # Simple heuristic-based detection
        if "def " in code and "import " in code:
            return "python"
        elif "function" in code or "const " in code or "=>" in code:
            return "javascript"
        elif "public class" in code or "public static void" in code:
            return "java"
        elif "#include" in code:
            return "cpp" if "cout" in code or "class" in code else "c"

        return "unknown"

    def extract_error_context(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract context around an error."""
        context = {
            "error_type": error_data.get("exception_type", "unknown"),
            "error_message": error_data.get("message", ""),
            "stack_depth": 0,
            "error_location": {},
            "surrounding_code": [],
        }

        # Extract from traceback
        traceback = error_data.get("traceback", [])
        if isinstance(traceback, list):
            context["stack_depth"] = len([line for line in traceback if "File" in line])

        # Extract from detailed frames
        if (
            "error_details" in error_data
            and "detailed_frames" in error_data["error_details"]
        ):
            frames = error_data["error_details"]["detailed_frames"]
            if frames:
                last_frame = frames[-1]
                context["error_location"] = {
                    "file": last_frame.get("file", ""),
                    "line": last_frame.get("line", 0),
                    "function": last_frame.get("function", ""),
                    "code": last_frame.get("code", ""),
                }

                # Get surrounding lines if available
                if "line_context" in last_frame:
                    context["surrounding_code"] = last_frame["line_context"].split("\n")

        return context

    def extract_semantic_embeddings(
        self, code: str, language: str
    ) -> Optional[np.ndarray]:
        """Extract semantic embeddings using CodeBERT or similar models."""
        if not self.semantic_model or not self.tokenizer:
            return None

        try:
            # Prepare input
            inputs = self.tokenizer(
                code, return_tensors="pt", max_length=512, truncation=True, padding=True
            )

            # Get embeddings
            with torch.no_grad():
                outputs = self.semantic_model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1).numpy()

            return embeddings[0]

        except Exception as e:
            print(f"Failed to extract embeddings: {e}")
            return None

    def extract_dependency_features(self, code: str, language: str) -> Dict[str, Any]:
        """Extract dependency and import features."""
        features = {
            "import_count": 0,
            "external_dependencies": [],
            "internal_dependencies": [],
            "circular_dependencies": False,
        }

        if language == "python":
            # Extract Python imports
            import_pattern = r"(?:from\s+([\w.]+)\s+)?import\s+([\w,\s*]+)"
            for match in re.finditer(import_pattern, code):
                features["import_count"] += 1
                module = match.group(1) or match.group(2).split(",")[0].strip()
                if "." in module:
                    features["internal_dependencies"].append(module)
                else:
                    features["external_dependencies"].append(module)

        elif language in ["javascript", "typescript"]:
            # Extract JS/TS imports
            import_pattern = (
                r'import\s+.*?from\s+[\'"](.+?)[\'"]|require\s*\([\'"](.+?)[\'"]\)'
            )
            for match in re.finditer(import_pattern, code):
                features["import_count"] += 1
                module = match.group(1) or match.group(2)
                if module.startswith("."):
                    features["internal_dependencies"].append(module)
                else:
                    features["external_dependencies"].append(module)

        return features

    def extract_framework_features(self, code: str, language: str) -> Dict[str, Any]:
        """Extract framework-specific features."""
        features = {"framework": None, "framework_patterns": {}}

        if language == "python":
            # Django detection
            if "django" in code.lower() or "models.Model" in code:
                features["framework"] = "django"
                features["framework_patterns"] = {
                    "uses_models": bool(re.search(r"models\.Model", code)),
                    "uses_views": bool(re.search(r"django\.views|View", code)),
                    "uses_forms": bool(re.search(r"forms\.Form|ModelForm", code)),
                    "uses_serializers": bool(re.search(r"serializers\.", code)),
                }
            # Flask detection
            elif "flask" in code.lower() or "@app.route" in code:
                features["framework"] = "flask"
                features["framework_patterns"] = {
                    "uses_routes": bool(re.search(r"@app\.route", code)),
                    "uses_blueprints": bool(re.search(r"Blueprint", code)),
                    "uses_request": bool(re.search(r"request\.", code)),
                }
            # FastAPI detection
            elif "fastapi" in code.lower() or "APIRouter" in code:
                features["framework"] = "fastapi"
                features["framework_patterns"] = {
                    "uses_routers": bool(re.search(r"APIRouter", code)),
                    "uses_pydantic": bool(re.search(r"BaseModel", code)),
                    "uses_dependencies": bool(re.search(r"Depends", code)),
                }

        elif language in ["javascript", "typescript"]:
            # React detection
            if "react" in code.lower() or "useState" in code or "jsx" in code:
                features["framework"] = "react"
                features["framework_patterns"] = {
                    "uses_hooks": bool(re.search(r"use[A-Z]\w+", code)),
                    "uses_jsx": bool(re.search(r"<[A-Z]\w+", code)),
                    "uses_components": bool(
                        re.search(r"export.*function.*return.*<", code, re.DOTALL)
                    ),
                }
            # Express detection
            elif "express" in code.lower() or "app.get(" in code:
                features["framework"] = "express"
                features["framework_patterns"] = {
                    "uses_routes": bool(re.search(r"app\.(get|post|put|delete)", code)),
                    "uses_middleware": bool(re.search(r"app\.use", code)),
                    "uses_router": bool(re.search(r"Router\(\)", code)),
                }

        return features

    def extract_features(self, error_data: Dict[str, Any]) -> CodeFeatures:
        """Extract comprehensive features from error data."""
        # Extract error context
        error_context = self.extract_error_context(error_data)

        # Get code snippet
        code = error_context["error_location"].get("code", "")
        if not code and error_context["surrounding_code"]:
            code = "\n".join(error_context["surrounding_code"])

        # Detect language
        file_path = error_context["error_location"].get("file", "")
        language = self.detect_language(code, file_path)

        # Get language-specific analyzer
        analyzer = self.analyzers.get(language)

        # Extract features based on available analyzer
        if analyzer:
            ast_features = analyzer.extract_ast_features(code)
            complexity_metrics = analyzer.calculate_complexity(code)
            pattern_matches = analyzer.extract_patterns(code)
        else:
            # Fallback to basic analysis
            ast_features = {"language_not_supported": True}
            complexity_metrics = {
                "cyclomatic_complexity": 1,
                "cognitive_complexity": 1,
                "lines_of_code": len(code.split("\n")),
                "nesting_depth": 0,
            }
            pattern_matches = {}

        # Extract language-agnostic features
        dependency_features = self.extract_dependency_features(code, language)
        framework_features = self.extract_framework_features(code, language)

        # Extract token features
        tokens = re.findall(r"\b\w+\b", code)
        token_features = dict(Counter(tokens).most_common(50))

        # Extract semantic embeddings
        embeddings = self.extract_semantic_embeddings(code, language)

        # Create CodeFeatures object
        features = CodeFeatures(
            language=language,
            file_path=file_path,
            line_number=error_context["error_location"].get("line", 0),
            function_name=error_context["error_location"].get("function", ""),
            class_name=None,  # TODO: Extract from AST
            # Complexity metrics
            **complexity_metrics,
            # Error features
            error_type=error_context["error_type"],
            error_message=error_context["error_message"],
            error_context=code,
            stack_depth=error_context["stack_depth"],
            # Structure features
            ast_features=ast_features,
            token_features=token_features,
            dependency_features=dependency_features,
            # Semantic features
            embeddings=embeddings,
            # Pattern features
            pattern_matches=pattern_matches,
            anti_patterns=[
                k
                for k, v in pattern_matches.items()
                if k.startswith(("bare_", "uses_eval", "uses_var", "global_usage"))
                and v
            ],
            # Context features
            surrounding_code=error_context["surrounding_code"],
            imports=dependency_features["external_dependencies"]
            + dependency_features["internal_dependencies"],
            locals=error_data.get("error_details", {})
            .get("detailed_frames", [{}])[-1]
            .get("locals", {}),
            # Framework features
            framework=framework_features["framework"],
            framework_features=framework_features["framework_patterns"],
        )

        return features

    def extract_batch_features(
        self, error_data_list: List[Dict[str, Any]]
    ) -> List[CodeFeatures]:
        """Extract features from multiple errors in batch."""
        return [self.extract_features(error_data) for error_data in error_data_list]

    def features_to_vector(self, features: CodeFeatures) -> np.ndarray:
        """Convert CodeFeatures to a numerical feature vector."""
        vector_components = []

        # Numeric features
        vector_components.extend(
            [
                features.cyclomatic_complexity,
                features.cognitive_complexity,
                features.lines_of_code,
                features.nesting_depth,
                features.stack_depth,
                len(features.imports),
                len(features.anti_patterns),
            ]
        )

        # One-hot encode language
        languages = [
            "python",
            "javascript",
            "typescript",
            "java",
            "cpp",
            "c",
            "csharp",
            "go",
            "ruby",
            "php",
        ]
        lang_vector = [1 if features.language == lang else 0 for lang in languages]
        vector_components.extend(lang_vector)

        # One-hot encode error type (top 20 most common)
        error_types = [
            "KeyError",
            "ValueError",
            "TypeError",
            "AttributeError",
            "IndexError",
            "ImportError",
            "NameError",
            "SyntaxError",
            "RuntimeError",
            "ZeroDivisionError",
            "FileNotFoundError",
            "PermissionError",
            "ConnectionError",
            "TimeoutError",
            "AssertionError",
            "NotImplementedError",
            "MemoryError",
            "RecursionError",
            "StopIteration",
            "GeneratorExit",
        ]
        error_vector = [1 if features.error_type == et else 0 for et in error_types]
        vector_components.extend(error_vector)

        # Boolean pattern features
        pattern_values = [
            1 if features.pattern_matches.get(pattern, False) else 0
            for pattern in [
                "uses_type_hints",
                "uses_async",
                "bare_except",
                "uses_eval",
                "uses_var",
                "callback_hell",
                "long_lines",
                "todo_comments",
            ]
        ]
        vector_components.extend(pattern_values)

        # Framework indicator
        frameworks = [
            "django",
            "flask",
            "fastapi",
            "react",
            "express",
            "vue",
            "angular",
        ]
        framework_vector = [1 if features.framework == fw else 0 for fw in frameworks]
        vector_components.extend(framework_vector)

        # AST feature counts (normalized)
        if "node_types" in features.ast_features:
            for node_type in ["FunctionDef", "ClassDef", "If", "For", "Try", "Import"]:
                count = features.ast_features["node_types"].get(node_type, 0)
                vector_components.append(min(count / 10.0, 1.0))  # Normalize to [0, 1]
        else:
            vector_components.extend([0] * 6)

        # Add embeddings if available
        if features.embeddings is not None:
            # Use first 50 dimensions of embeddings
            vector_components.extend(features.embeddings[:50].tolist())
        else:
            vector_components.extend([0] * 50)

        return np.array(vector_components)


class FeaturePipeline:
    """End-to-end feature extraction pipeline."""

    def __init__(self, cache_features: bool = True):
        self.extractor = MultiLanguageFeatureExtractor()
        self.cache_features = cache_features
        self.feature_cache = {}

    def process_error(self, error_data: Dict[str, Any]) -> np.ndarray:
        """Process a single error and return feature vector."""
        # Generate cache key
        cache_key = self._generate_cache_key(error_data)

        # Check cache
        if self.cache_features and cache_key in self.feature_cache:
            return self.feature_cache[cache_key]

        # Extract features
        features = self.extractor.extract_features(error_data)

        # Convert to vector
        feature_vector = self.extractor.features_to_vector(features)

        # Cache result
        if self.cache_features:
            self.feature_cache[cache_key] = feature_vector

        return feature_vector

    def process_batch(self, error_data_list: List[Dict[str, Any]]) -> np.ndarray:
        """Process multiple errors and return feature matrix."""
        feature_vectors = [
            self.process_error(error_data) for error_data in error_data_list
        ]
        return np.vstack(feature_vectors)

    def _generate_cache_key(self, error_data: Dict[str, Any]) -> str:
        """Generate a cache key for error data."""
        key_components = [
            error_data.get("exception_type", ""),
            error_data.get("message", ""),
            str(
                error_data.get("error_details", {})
                .get("detailed_frames", [{}])[-1]
                .get("line", 0)
            ),
        ]
        key_string = "|".join(key_components)
        return hashlib.sha256(key_string.encode()).hexdigest()

    def save_feature_metadata(self, output_path: str):
        """Save feature metadata for interpretation."""
        metadata = {
            "feature_names": self._get_feature_names(),
            "feature_types": self._get_feature_types(),
            "feature_descriptions": self._get_feature_descriptions(),
        }

        with open(output_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def _get_feature_names(self) -> List[str]:
        """Get names of all features in the vector."""
        names = [
            "cyclomatic_complexity",
            "cognitive_complexity",
            "lines_of_code",
            "nesting_depth",
            "stack_depth",
            "import_count",
            "anti_pattern_count",
        ]

        # Language features
        languages = [
            "python",
            "javascript",
            "typescript",
            "java",
            "cpp",
            "c",
            "csharp",
            "go",
            "ruby",
            "php",
        ]
        names.extend([f"lang_{lang}" for lang in languages])

        # Error type features
        error_types = [
            "KeyError",
            "ValueError",
            "TypeError",
            "AttributeError",
            "IndexError",
            "ImportError",
            "NameError",
            "SyntaxError",
            "RuntimeError",
            "ZeroDivisionError",
            "FileNotFoundError",
            "PermissionError",
            "ConnectionError",
            "TimeoutError",
            "AssertionError",
            "NotImplementedError",
            "MemoryError",
            "RecursionError",
            "StopIteration",
            "GeneratorExit",
        ]
        names.extend([f"error_{et}" for et in error_types])

        # Pattern features
        patterns = [
            "uses_type_hints",
            "uses_async",
            "bare_except",
            "uses_eval",
            "uses_var",
            "callback_hell",
            "long_lines",
            "todo_comments",
        ]
        names.extend([f"pattern_{p}" for p in patterns])

        # Framework features
        frameworks = [
            "django",
            "flask",
            "fastapi",
            "react",
            "express",
            "vue",
            "angular",
        ]
        names.extend([f"framework_{fw}" for fw in frameworks])

        # AST features
        ast_nodes = ["FunctionDef", "ClassDef", "If", "For", "Try", "Import"]
        names.extend([f"ast_{node}" for node in ast_nodes])

        # Embedding features
        names.extend([f"embedding_{i}" for i in range(50)])

        return names

    def _get_feature_types(self) -> Dict[str, str]:
        """Get types of all features."""
        types = {}

        # Numeric features
        for name in [
            "cyclomatic_complexity",
            "cognitive_complexity",
            "lines_of_code",
            "nesting_depth",
            "stack_depth",
            "import_count",
            "anti_pattern_count",
        ]:
            types[name] = "numeric"

        # Categorical features (one-hot encoded)
        for prefix in ["lang_", "error_", "pattern_", "framework_"]:
            for name in self._get_feature_names():
                if name.startswith(prefix):
                    types[name] = "binary"

        # AST features (normalized counts)
        for name in self._get_feature_names():
            if name.startswith("ast_"):
                types[name] = "numeric"

        # Embedding features
        for name in self._get_feature_names():
            if name.startswith("embedding_"):
                types[name] = "numeric"

        return types

    def _get_feature_descriptions(self) -> Dict[str, str]:
        """Get descriptions of all features."""
        descriptions = {
            "cyclomatic_complexity": "McCabe cyclomatic complexity",
            "cognitive_complexity": "Cognitive complexity metric",
            "lines_of_code": "Number of non-empty, non-comment lines",
            "nesting_depth": "Maximum nesting depth in the code",
            "stack_depth": "Depth of the error stack trace",
            "import_count": "Number of import statements",
            "anti_pattern_count": "Number of detected anti-patterns",
        }

        # Add descriptions for other features
        for name in self._get_feature_names():
            if name.startswith("lang_"):
                descriptions[name] = f"Language is {name[5:]}"
            elif name.startswith("error_"):
                descriptions[name] = f"Error type is {name[6:]}"
            elif name.startswith("pattern_"):
                descriptions[name] = f"Code exhibits pattern: {name[8:]}"
            elif name.startswith("framework_"):
                descriptions[name] = f"Uses {name[10:]} framework"
            elif name.startswith("ast_"):
                descriptions[name] = f"Normalized count of {name[4:]} AST nodes"
            elif name.startswith("embedding_"):
                descriptions[name] = f"Semantic embedding dimension {name[10:]}"

        return descriptions


if __name__ == "__main__":
    # Example usage
    from .data_collector import get_sample_data

    # Create feature extractor
    extractor = MultiLanguageFeatureExtractor()

    # Get sample error data
    sample_errors = get_sample_data()

    # Extract features from first error
    features = extractor.extract_features(sample_errors[0])

    print(f"Language: {features.language}")
    print(f"Error Type: {features.error_type}")
    print(
        f"Complexity: Cyclomatic={features.cyclomatic_complexity}, Cognitive={features.cognitive_complexity}"
    )
    print(f"Anti-patterns: {features.anti_patterns}")
    print(f"Framework: {features.framework}")

    # Convert to vector
    vector = extractor.features_to_vector(features)
    print(f"\nFeature vector shape: {vector.shape}")

    # Create pipeline and process batch
    pipeline = FeaturePipeline()
    feature_matrix = pipeline.process_batch(sample_errors)
    print(f"\nBatch feature matrix shape: {feature_matrix.shape}")

    # Save feature metadata
    pipeline.save_feature_metadata("feature_metadata.json")
    print("\nFeature metadata saved to feature_metadata.json")
