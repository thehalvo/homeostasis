"""
Enhanced Python Analyzer Plugin

Main entry point for the Enhanced Python Analyzer plugin that provides
ML-powered insights and framework-specific optimizations for Python errors.
"""

import ast
import json
import logging
import re
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# USHS imports (these would be provided by the host system)
from ushs_core import PluginContext, PluginInput, PluginOutput, USHSPlugin
from ushs_core.analysis import (AnalysisResult, CodeContext, FixSuggestion,
                                Pattern)

logger = logging.getLogger(__name__)


class ErrorCategory(Enum):
    """Categories of Python errors."""

    SYNTAX = "syntax"
    RUNTIME = "runtime"
    LOGIC = "logic"
    PERFORMANCE = "performance"
    SECURITY = "security"
    STYLE = "style"


class FrameworkType(Enum):
    """Supported Python frameworks."""

    DJANGO = "django"
    FLASK = "flask"
    FASTAPI = "fastapi"
    TORNADO = "tornado"
    PYRAMID = "pyramid"
    GENERIC = "generic"


@dataclass
class MLInsight:
    """Machine learning insight for error analysis."""

    insight_type: str
    description: str
    confidence: float
    related_patterns: List[str]
    suggested_action: str


class EnhancedPythonAnalyzer(USHSPlugin):
    """Enhanced Python Analyzer plugin implementation."""

    # Plugin metadata (loaded from manifest.json)
    metadata = {
        "name": "enhanced-python-analyzer",
        "version": "1.0.0",
        "type": "analysis",
    }

    def __init__(self):
        """Initialize the plugin."""
        super().__init__()
        self.config = {}
        self.ml_client = None
        self.pattern_db = self._load_pattern_database()
        self.framework_detector = FrameworkDetector()

    async def initialize(self, context: PluginContext) -> None:
        """
        Initialize the plugin with context.

        Args:
            context: Plugin execution context
        """
        logger.info("Initializing Enhanced Python Analyzer")

        # Load configuration
        self.config = context.config

        # Initialize ML client if API key is provided
        if self.config.get("mlApiKey"):
            self.ml_client = MLClient(self.config["mlApiKey"])

        # Set analysis depth
        self.analysis_depth = self.config.get("analysisDepth", "standard")

        logger.info(f"Plugin initialized with analysis depth: {self.analysis_depth}")

    async def execute(self, input_data: PluginInput) -> PluginOutput:
        """
        Execute the plugin analysis.

        Args:
            input_data: Input data containing error information

        Returns:
            Analysis results with suggestions
        """
        try:
            # Extract error data
            error_data = input_data.data

            # Perform analysis
            analysis_result = await self.analyze_error(error_data)

            # Generate output
            return PluginOutput(
                success=True,
                data=asdict(analysis_result),
                metadata={
                    "plugin": self.metadata["name"],
                    "version": self.metadata["version"],
                    "analysis_depth": self.analysis_depth,
                },
            )

        except Exception as e:
            logger.error(f"Error during analysis: {e}")
            return PluginOutput(success=False, error=str(e), data={})

    async def analyze_error(self, error_data: Dict[str, Any]) -> AnalysisResult:
        """
        Analyze a Python error with ML-powered insights.

        Args:
            error_data: Error information

        Returns:
            Comprehensive analysis result
        """
        # Extract error details
        error_type = error_data.get("exception_type", "Unknown")
        error_message = error_data.get("message", "")
        traceback = error_data.get("traceback", [])
        code_context = self._extract_code_context(error_data)

        # Detect framework
        framework = self.framework_detector.detect(code_context)

        # Categorize error
        error_category = self._categorize_error(error_type, error_message)

        # Detect patterns
        patterns = await self.detect_patterns(code_context)

        # Get ML insights if available
        ml_insights = []
        if self.ml_client and self.analysis_depth in ["standard", "deep"]:
            ml_insights = await self._get_ml_insights(
                error_type, error_message, code_context, patterns
            )

        # Generate fix suggestions
        fix_suggestions = await self.suggest_fixes(
            error_type, error_message, code_context, patterns, ml_insights, framework
        )

        # Calculate confidence score
        confidence = self._calculate_confidence(patterns, ml_insights, fix_suggestions)

        # Build analysis result
        return AnalysisResult(
            error_type=error_type,
            error_category=error_category.value,
            confidence=confidence,
            root_cause=self._determine_root_cause(error_type, error_message, patterns),
            patterns_detected=[p.name for p in patterns],
            fix_suggestions=fix_suggestions,
            ml_insights=[asdict(i) for i in ml_insights],
            framework_specific={
                "framework": framework.value,
                "recommendations": self._get_framework_recommendations(
                    framework, error_type, patterns
                ),
            },
            metadata={
                "analysis_depth": self.analysis_depth,
                "patterns_checked": len(self.pattern_db),
                "ml_enabled": self.ml_client is not None,
            },
        )

    async def detect_patterns(self, code_context: CodeContext) -> List[Pattern]:
        """
        Detect code patterns and anti-patterns.

        Args:
            code_context: Code context around the error

        Returns:
            List of detected patterns
        """
        detected_patterns = []

        # Parse code if possible
        try:
            tree = ast.parse(code_context.code_snippet)

            # Check each pattern in the database
            for pattern in self.pattern_db:
                if self._check_pattern(tree, code_context, pattern):
                    detected_patterns.append(pattern)

            # Deep analysis if enabled
            if self.analysis_depth == "deep":
                # Additional AST analysis
                visitor = PatternVisitor()
                visitor.visit(tree)
                detected_patterns.extend(visitor.detected_patterns)

        except SyntaxError:
            # Fallback to regex-based pattern detection
            for pattern in self.pattern_db:
                if pattern.regex and re.search(
                    pattern.regex, code_context.code_snippet
                ):
                    detected_patterns.append(pattern)

        return detected_patterns

    async def suggest_fixes(
        self,
        error_type: str,
        error_message: str,
        code_context: CodeContext,
        patterns: List[Pattern],
        ml_insights: List[MLInsight],
        framework: FrameworkType,
    ) -> List[FixSuggestion]:
        """
        Generate intelligent fix suggestions.

        Args:
            error_type: Type of error
            error_message: Error message
            code_context: Code context
            patterns: Detected patterns
            ml_insights: ML-generated insights
            framework: Detected framework

        Returns:
            List of fix suggestions
        """
        suggestions = []

        # Error-specific fixes
        if error_type == "KeyError":
            suggestions.extend(
                self._suggest_keyerror_fixes(error_message, code_context)
            )
        elif error_type == "AttributeError":
            suggestions.extend(
                self._suggest_attributeerror_fixes(error_message, code_context)
            )
        elif error_type == "TypeError":
            suggestions.extend(
                self._suggest_typeerror_fixes(error_message, code_context)
            )

        # Pattern-based fixes
        for pattern in patterns:
            if pattern.fix_suggestion:
                suggestions.append(pattern.fix_suggestion)

        # ML-based fixes
        for insight in ml_insights:
            if insight.confidence > 0.8:
                suggestions.append(
                    FixSuggestion(
                        type="ml_suggested",
                        description=insight.suggested_action,
                        confidence=insight.confidence,
                        patch=self._generate_patch(code_context, insight),
                    )
                )

        # Framework-specific fixes
        framework_suggestions = self._get_framework_fixes(
            framework, error_type, code_context
        )
        suggestions.extend(framework_suggestions)

        # Sort by confidence and deduplicate
        suggestions = self._deduplicate_suggestions(suggestions)
        suggestions.sort(key=lambda s: s.confidence, reverse=True)

        return suggestions[:5]  # Return top 5 suggestions

    async def shutdown(self) -> None:
        """Clean up resources."""
        if self.ml_client:
            await self.ml_client.close()

    def _load_pattern_database(self) -> List[Pattern]:
        """Load the pattern database."""
        patterns = []

        # Common Python anti-patterns
        patterns.append(
            Pattern(
                name="unsafe_dict_access",
                description="Direct dictionary key access without checking",
                regex=r'(\w+)\[[\'"]\w+[\'"]\\](?!\s*=)',
                severity="medium",
                fix_suggestion=FixSuggestion(
                    type="code_change",
                    description="Use .get() method with default value",
                    confidence=0.95,
                    patch=None,  # Generated dynamically
                ),
            )
        )

        patterns.append(
            Pattern(
                name="bare_except",
                description="Bare except clause that catches all exceptions",
                regex=r"except\s*:",
                severity="high",
                fix_suggestion=FixSuggestion(
                    type="code_change",
                    description="Specify exception types to catch",
                    confidence=0.9,
                    patch=None,
                ),
            )
        )

        patterns.append(
            Pattern(
                name="mutable_default_argument",
                description="Mutable default argument in function definition",
                regex=r"def\s+\w+\([^)]*=\s*(\[|{)",
                severity="high",
                fix_suggestion=FixSuggestion(
                    type="code_change",
                    description="Use None as default and initialize in function body",
                    confidence=0.98,
                    patch=None,
                ),
            )
        )

        return patterns

    def _extract_code_context(self, error_data: Dict[str, Any]) -> CodeContext:
        """Extract code context from error data."""
        # This would normally extract actual code from the file
        # For this example, we'll create a simplified context
        return CodeContext(
            file_path=error_data.get("file_path", ""),
            line_number=error_data.get("line_number", 0),
            code_snippet=error_data.get("code_snippet", ""),
            function_name=error_data.get("function_name", ""),
            class_name=error_data.get("class_name", ""),
        )

    def _categorize_error(self, error_type: str, error_message: str) -> ErrorCategory:
        """Categorize the error type."""
        if error_type in ["SyntaxError", "IndentationError"]:
            return ErrorCategory.SYNTAX
        elif error_type in ["KeyError", "AttributeError", "IndexError", "TypeError"]:
            return ErrorCategory.RUNTIME
        elif "performance" in error_message.lower():
            return ErrorCategory.PERFORMANCE
        elif "security" in error_message.lower():
            return ErrorCategory.SECURITY
        else:
            return ErrorCategory.RUNTIME

    def _check_pattern(
        self, tree: ast.AST, context: CodeContext, pattern: Pattern
    ) -> bool:
        """Check if a pattern matches the code."""
        # Simplified pattern checking
        # In a real implementation, this would be more sophisticated
        if pattern.regex:
            return bool(re.search(pattern.regex, context.code_snippet))
        return False

    async def _get_ml_insights(
        self,
        error_type: str,
        error_message: str,
        code_context: CodeContext,
        patterns: List[Pattern],
    ) -> List[MLInsight]:
        """Get ML-powered insights."""
        if not self.ml_client:
            return []

        try:
            # Call ML service
            insights = await self.ml_client.analyze(
                error_type=error_type,
                error_message=error_message,
                code_context=code_context,
                patterns=[p.name for p in patterns],
            )

            return insights
        except Exception as e:
            logger.warning(f"ML analysis failed: {e}")
            return []

    def _determine_root_cause(
        self, error_type: str, error_message: str, patterns: List[Pattern]
    ) -> str:
        """Determine the root cause of the error."""
        # Simplified root cause analysis
        if error_type == "KeyError":
            return "Missing key validation before dictionary access"
        elif error_type == "AttributeError":
            return "Attempting to access attribute on None or incorrect type"
        elif patterns:
            return f"Code pattern issue: {patterns[0].description}"
        else:
            return "Error cause requires further investigation"

    def _calculate_confidence(
        self,
        patterns: List[Pattern],
        ml_insights: List[MLInsight],
        suggestions: List[FixSuggestion],
    ) -> float:
        """Calculate overall confidence score."""
        if not patterns and not ml_insights:
            return 0.5

        # Weight different factors
        pattern_confidence = sum(0.8 for _ in patterns) / max(len(patterns), 1)
        ml_confidence = sum(i.confidence for i in ml_insights) / max(
            len(ml_insights), 1
        )
        suggestion_confidence = sum(s.confidence for s in suggestions) / max(
            len(suggestions), 1
        )

        # Weighted average
        weights = [0.3, 0.4, 0.3]
        scores = [pattern_confidence, ml_confidence, suggestion_confidence]

        return sum(w * s for w, s in zip(weights, scores))

    def _suggest_keyerror_fixes(
        self, error_message: str, context: CodeContext
    ) -> List[FixSuggestion]:
        """Suggest fixes for KeyError."""
        suggestions = []

        # Extract key name from error message
        key_match = re.search(r"['\"](\w+)['\"]", error_message)
        if key_match:
            key_name = key_match.group(1)

            suggestions.append(
                FixSuggestion(
                    type="code_change",
                    description=f'Use .get() method with default value for key "{key_name}"',
                    confidence=0.95,
                    patch=f"value = data.get('{key_name}', None)",
                )
            )

            suggestions.append(
                FixSuggestion(
                    type="code_change",
                    description=f'Check if key "{key_name}" exists before accessing',
                    confidence=0.9,
                    patch=f"if '{key_name}' in data:\n    value = data['{key_name}']",
                )
            )

        return suggestions

    def _suggest_attributeerror_fixes(
        self, error_message: str, context: CodeContext
    ) -> List[FixSuggestion]:
        """Suggest fixes for AttributeError."""
        suggestions = []

        if "NoneType" in error_message:
            suggestions.append(
                FixSuggestion(
                    type="code_change",
                    description="Add None check before accessing attribute",
                    confidence=0.9,
                    patch="if obj is not None:\n    value = obj.attribute",
                )
            )

        return suggestions

    def _suggest_typeerror_fixes(
        self, error_message: str, context: CodeContext
    ) -> List[FixSuggestion]:
        """Suggest fixes for TypeError."""
        suggestions = []

        if "argument" in error_message:
            suggestions.append(
                FixSuggestion(
                    type="code_change",
                    description="Check function signature and argument types",
                    confidence=0.85,
                    patch=None,
                )
            )

        return suggestions

    def _get_framework_recommendations(
        self, framework: FrameworkType, error_type: str, patterns: List[Pattern]
    ) -> List[str]:
        """Get framework-specific recommendations."""
        recommendations = []

        if framework == FrameworkType.DJANGO:
            if error_type == "DoesNotExist":
                recommendations.append(
                    "Use get_object_or_404() for cleaner error handling"
                )
            elif error_type == "MultipleObjectsReturned":
                recommendations.append(
                    "Use filter().first() instead of get() when multiple objects possible"
                )

        elif framework == FrameworkType.FLASK:
            if error_type == "KeyError" and any(
                p.name == "unsafe_dict_access" for p in patterns
            ):
                recommendations.append(
                    "Use request.args.get() instead of direct access"
                )

        elif framework == FrameworkType.FASTAPI:
            if error_type == "ValidationError":
                recommendations.append(
                    "Define Pydantic models for request/response validation"
                )

        return recommendations

    def _get_framework_fixes(
        self, framework: FrameworkType, error_type: str, context: CodeContext
    ) -> List[FixSuggestion]:
        """Get framework-specific fix suggestions."""
        suggestions = []

        if framework == FrameworkType.DJANGO and error_type == "DoesNotExist":
            suggestions.append(
                FixSuggestion(
                    type="framework_specific",
                    description="Use Django's get_object_or_404",
                    confidence=0.95,
                    patch="from django.shortcuts import get_object_or_404\nobj = get_object_or_404(Model, pk=id)",
                )
            )

        return suggestions

    def _generate_patch(
        self, context: CodeContext, insight: MLInsight
    ) -> Optional[str]:
        """Generate a code patch based on ML insight."""
        # Simplified patch generation
        # In a real implementation, this would use sophisticated code generation
        return None

    def _deduplicate_suggestions(
        self, suggestions: List[FixSuggestion]
    ) -> List[FixSuggestion]:
        """Remove duplicate suggestions."""
        seen = set()
        unique = []

        for suggestion in suggestions:
            key = (suggestion.type, suggestion.description)
            if key not in seen:
                seen.add(key)
                unique.append(suggestion)

        return unique


class FrameworkDetector:
    """Detects Python framework from code context."""

    def detect(self, context: CodeContext) -> FrameworkType:
        """Detect the framework being used."""
        code = context.code_snippet.lower()
        file_path = context.file_path.lower()

        # Django detection
        if any(
            indicator in code for indicator in ["django", "models.model", "views.py"]
        ):
            return FrameworkType.DJANGO

        # Flask detection
        if any(indicator in code for indicator in ["flask", "app.route", "@app."]):
            return FrameworkType.FLASK

        # FastAPI detection
        if any(indicator in code for indicator in ["fastapi", "pydantic", "@app.get"]):
            return FrameworkType.FASTAPI

        # Tornado detection
        if "tornado" in code:
            return FrameworkType.TORNADO

        # Pyramid detection
        if "pyramid" in code:
            return FrameworkType.PYRAMID

        return FrameworkType.GENERIC


class PatternVisitor(ast.NodeVisitor):
    """AST visitor for detecting code patterns."""

    def __init__(self):
        self.detected_patterns = []

    def visit_FunctionDef(self, node):
        """Check function definitions for patterns."""
        # Check for mutable default arguments
        for arg in node.args.defaults:
            if isinstance(arg, (ast.List, ast.Dict, ast.Set)):
                self.detected_patterns.append(
                    Pattern(
                        name="mutable_default_argument",
                        description=f"Mutable default argument in function {node.name}",
                        severity="high",
                    )
                )

        self.generic_visit(node)

    def visit_ExceptHandler(self, node):
        """Check exception handlers."""
        if node.type is None:
            self.detected_patterns.append(
                Pattern(
                    name="bare_except",
                    description="Bare except clause found",
                    severity="high",
                )
            )

        self.generic_visit(node)


class MLClient:
    """Mock ML client for demonstration."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    async def analyze(self, **kwargs) -> List[MLInsight]:
        """Mock ML analysis."""
        # In a real implementation, this would call an ML service
        return [
            MLInsight(
                insight_type="pattern_recognition",
                description="Similar error pattern found in 85% of Django projects",
                confidence=0.85,
                related_patterns=["unsafe_dict_access"],
                suggested_action="Implement request data validation middleware",
            )
        ]

    async def close(self):
        """Close ML client connection."""
        pass


# Export the plugin class
__all__ = ["EnhancedPythonAnalyzer"]
