"""
Intelligent Error Classification System combining rule-based and ML-based approaches.

This module provides sophisticated error classification that recognizes error types
including syntax, logic, config, environment, concurrency, and security issues.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .comprehensive_error_detector import (ComprehensiveErrorDetector,
                                           ErrorCategory, ErrorClassification,
                                           ErrorContext, ErrorSeverity,
                                           LanguageType)
from .language_parsers import CompilerIntegration, create_language_parser

logger = logging.getLogger(__name__)


class ClassificationConfidence(Enum):
    """Classification confidence levels."""

    VERY_HIGH = "very_high"  # > 0.9
    HIGH = "high"  # 0.7 - 0.9
    MEDIUM = "medium"  # 0.5 - 0.7
    LOW = "low"  # 0.3 - 0.5
    VERY_LOW = "very_low"  # < 0.3


@dataclass
class ClassificationFeatures:
    """Features extracted for ML classification."""

    # Error message features
    error_message_length: int
    has_stack_trace: bool
    stack_trace_depth: int

    # Language features
    language_type: str
    framework_detected: bool

    # Error type indicators
    has_exception_type: bool
    has_line_number: bool
    has_file_path: bool

    # Content features
    has_import_error: bool
    has_syntax_keywords: bool
    has_null_reference: bool
    has_index_error: bool
    has_type_error: bool
    has_network_keywords: bool
    has_database_keywords: bool
    has_config_keywords: bool
    has_security_keywords: bool
    has_concurrency_keywords: bool

    # Environment features
    has_env_variables: bool
    has_dependency_info: bool

    def to_array(self) -> List[float]:
        """Convert features to array for ML model."""
        return [
            float(self.error_message_length),
            float(self.has_stack_trace),
            float(self.stack_trace_depth),
            float(self.framework_detected),
            float(self.has_exception_type),
            float(self.has_line_number),
            float(self.has_file_path),
            float(self.has_import_error),
            float(self.has_syntax_keywords),
            float(self.has_null_reference),
            float(self.has_index_error),
            float(self.has_type_error),
            float(self.has_network_keywords),
            float(self.has_database_keywords),
            float(self.has_config_keywords),
            float(self.has_security_keywords),
            float(self.has_concurrency_keywords),
            float(self.has_env_variables),
            float(self.has_dependency_info),
        ]


class FeatureExtractor:
    """Extract features from error contexts for ML classification."""

    def __init__(self):
        """Initialize feature extractor."""

        # Keyword sets for different error categories
        self.syntax_keywords = {
            "syntax",
            "parse",
            "unexpected",
            "invalid",
            "malformed",
            "indentation",
            "bracket",
            "parenthesis",
            "quote",
            "eof",
        }

        self.network_keywords = {
            "connection",
            "timeout",
            "network",
            "socket",
            "http",
            "https",
            "url",
            "dns",
            "ssl",
            "tls",
            "certificate",
            "unreachable",
        }

        self.database_keywords = {
            "database",
            "sql",
            "query",
            "connection",
            "table",
            "column",
            "constraint",
            "foreign",
            "key",
            "transaction",
            "rollback",
        }

        self.config_keywords = {
            "config",
            "configuration",
            "setting",
            "property",
            "parameter",
            "environment",
            "variable",
            "missing",
            "invalid",
        }

        self.security_keywords = {
            "security",
            "permission",
            "access",
            "denied",
            "unauthorized",
            "authentication",
            "authorization",
            "token",
            "credential",
        }

        self.concurrency_keywords = {
            "thread",
            "lock",
            "deadlock",
            "race",
            "concurrent",
            "atomic",
            "synchronization",
            "mutex",
            "semaphore",
            "parallel",
        }

    def extract_features(self, error_context: ErrorContext) -> ClassificationFeatures:
        """
        Extract classification features from error context.

        Args:
            error_context: Error context to extract features from

        Returns:
            Classification features
        """
        # Handle both dict and ErrorContext object
        if isinstance(error_context, dict):
            error_msg = error_context.get("message", "").lower()
            error_message_length = len(error_context.get("message", ""))
            has_stack_trace = bool(error_context.get("stack_trace"))
            stack_trace_depth = (
                len(error_context.get("stack_trace", ""))
                if error_context.get("stack_trace")
                else 0
            )

            # Language and framework features
            language_type = error_context.get("language", "unknown")
            framework_detected = bool(error_context.get("framework"))

            # Error metadata features
            has_exception_type = bool(error_context.get("exception_type"))
            has_line_number = bool(error_context.get("line_number"))
            has_file_path = bool(error_context.get("file_path"))
        else:
            error_msg = error_context.error_message.lower()

            # Basic message features
            error_message_length = len(error_context.error_message)
            has_stack_trace = bool(error_context.stack_trace)
            stack_trace_depth = (
                len(error_context.stack_trace) if error_context.stack_trace else 0
            )

            # Language and framework features
            language_type = error_context.language.value
            framework_detected = bool(error_context.framework)

            # Error metadata features
            has_exception_type = bool(error_context.exception_type)
            has_line_number = bool(error_context.line_number)
            has_file_path = bool(error_context.file_path)

        # Content-based features
        has_import_error = any(
            keyword in error_msg for keyword in ["import", "module", "package"]
        )
        has_syntax_keywords = any(
            keyword in error_msg for keyword in self.syntax_keywords
        )
        has_null_reference = any(
            keyword in error_msg for keyword in ["null", "none", "undefined", "nil"]
        )
        has_index_error = any(
            keyword in error_msg for keyword in ["index", "bounds", "range"]
        )
        has_type_error = any(
            keyword in error_msg for keyword in ["type", "cast", "conversion"]
        )
        has_network_keywords = any(
            keyword in error_msg for keyword in self.network_keywords
        )
        has_database_keywords = any(
            keyword in error_msg for keyword in self.database_keywords
        )
        has_config_keywords = any(
            keyword in error_msg for keyword in self.config_keywords
        )
        has_security_keywords = any(
            keyword in error_msg for keyword in self.security_keywords
        )
        has_concurrency_keywords = any(
            keyword in error_msg for keyword in self.concurrency_keywords
        )

        # Environment features
        if isinstance(error_context, dict):
            has_env_variables = bool(error_context.get("environment_variables"))
            has_dependency_info = bool(error_context.get("dependencies"))
        else:
            has_env_variables = bool(
                getattr(error_context, "environment_variables", None)
            )
            has_dependency_info = bool(getattr(error_context, "dependencies", None))

        return ClassificationFeatures(
            error_message_length=error_message_length,
            has_stack_trace=has_stack_trace,
            stack_trace_depth=stack_trace_depth,
            language_type=language_type,
            framework_detected=framework_detected,
            has_exception_type=has_exception_type,
            has_line_number=has_line_number,
            has_file_path=has_file_path,
            has_import_error=has_import_error,
            has_syntax_keywords=has_syntax_keywords,
            has_null_reference=has_null_reference,
            has_index_error=has_index_error,
            has_type_error=has_type_error,
            has_network_keywords=has_network_keywords,
            has_database_keywords=has_database_keywords,
            has_config_keywords=has_config_keywords,
            has_security_keywords=has_security_keywords,
            has_concurrency_keywords=has_concurrency_keywords,
            has_env_variables=has_env_variables,
            has_dependency_info=has_dependency_info,
        )


class RuleBasedClassifier:
    """Enhanced rule-based classifier for error categorization."""

    def __init__(self):
        """Initialize rule-based classifier."""
        self.rules = self._create_classification_rules()

    def _create_classification_rules(self) -> List[Dict[str, Any]]:
        """Create classification rules."""
        return [
            # Syntax errors
            {
                "category": ErrorCategory.SYNTAX,
                "conditions": [
                    {"feature": "has_syntax_keywords", "value": True},
                    {
                        "feature": "has_exception_type",
                        "value": True,
                        "patterns": ["SyntaxError", "ParseError"],
                    },
                ],
                "confidence": 0.9,
            },
            # Compilation errors
            {
                "category": ErrorCategory.COMPILATION,
                "conditions": [
                    {
                        "feature": "error_message",
                        "patterns": [
                            "compilation error",
                            "build failed",
                            "cannot compile",
                        ],
                    }
                ],
                "confidence": 0.85,
            },
            # Logic errors
            {
                "category": ErrorCategory.LOGIC,
                "conditions": [
                    {"feature": "has_null_reference", "value": True},
                    {"feature": "has_index_error", "value": True},
                ],
                "confidence": 0.8,
                "match_any": True,
            },
            # Configuration errors
            {
                "category": ErrorCategory.CONFIGURATION,
                "conditions": [{"feature": "has_config_keywords", "value": True}],
                "confidence": 0.75,
            },
            # Network errors
            {
                "category": ErrorCategory.NETWORK,
                "conditions": [{"feature": "has_network_keywords", "value": True}],
                "confidence": 0.8,
            },
            # Database errors
            {
                "category": ErrorCategory.DATABASE,
                "conditions": [{"feature": "has_database_keywords", "value": True}],
                "confidence": 0.8,
            },
            # Security errors
            {
                "category": ErrorCategory.SECURITY,
                "conditions": [{"feature": "has_security_keywords", "value": True}],
                "confidence": 0.85,
            },
            # Concurrency errors
            {
                "category": ErrorCategory.CONCURRENCY,
                "conditions": [{"feature": "has_concurrency_keywords", "value": True}],
                "confidence": 0.85,
            },
            # Dependency errors
            {
                "category": ErrorCategory.DEPENDENCY,
                "conditions": [{"feature": "has_import_error", "value": True}],
                "confidence": 0.7,
            },
        ]

    def classify(
        self, features: ClassificationFeatures, error_context: ErrorContext
    ) -> Tuple[Optional[ErrorCategory], float]:
        """
        Classify error using rule-based approach.

        Args:
            features: Extracted features
            error_context: Original error context

        Returns:
            Tuple of (category, confidence)
        """
        best_category = None
        best_confidence = 0.0

        for rule in self.rules:
            if self._rule_matches(rule, features, error_context):
                confidence = rule["confidence"]
                if confidence > best_confidence:
                    best_category = rule["category"]
                    best_confidence = confidence

        return best_category, best_confidence

    def _rule_matches(
        self,
        rule: Dict[str, Any],
        features: ClassificationFeatures,
        error_context: ErrorContext,
    ) -> bool:
        """Check if a rule matches the given features and context."""
        conditions = rule["conditions"]
        match_any = rule.get("match_any", False)

        matches = []

        for condition in conditions:
            feature_name = condition["feature"]

            if feature_name == "error_message":
                # Special handling for error message patterns
                patterns = condition.get("patterns", [])
                # Handle both dict and ErrorContext object
                if isinstance(error_context, dict):
                    error_msg = error_context.get("message", "").lower()
                else:
                    error_msg = error_context.error_message.lower()
                match = any(pattern.lower() in error_msg for pattern in patterns)
                matches.append(match)
            else:
                # Handle feature-based conditions
                if hasattr(features, feature_name):
                    feature_value = getattr(features, feature_name)
                    expected_value = condition.get("value")
                    patterns = condition.get("patterns", [])

                    if expected_value is not None:
                        # Direct value comparison
                        matches.append(feature_value == expected_value)
                    elif patterns:
                        # Pattern matching for exception types
                        if feature_name == "has_exception_type":
                            if isinstance(error_context, dict):
                                exception_type = error_context.get("exception_type")
                            else:
                                exception_type = error_context.exception_type
                            if exception_type:
                                match = any(
                                    pattern in exception_type for pattern in patterns
                                )
                            matches.append(match)
                        else:
                            matches.append(False)
                    else:
                        matches.append(bool(feature_value))
                else:
                    matches.append(False)

        # Return result based on match_any flag
        if match_any:
            return any(matches)
        else:
            return all(matches)


class MLBasedClassifier:
    """Machine learning-based error classifier."""

    def __init__(self, model_path: Optional[str] = None):
        """Initialize ML classifier."""
        self.model_path = model_path
        self.model = None
        self.categories = list(ErrorCategory)
        self.feature_extractor = FeatureExtractor()

        # Try to load pre-trained model
        if model_path and Path(model_path).exists():
            self._load_model()
        else:
            # Create a simple rule-based fallback model
            self._create_fallback_model()

    def _load_model(self):
        """Load pre-trained ML model."""
        try:
            # Placeholder for actual model loading
            # In a real implementation, this would load a trained scikit-learn,
            # TensorFlow, or PyTorch model
            logger.info(f"Loading ML model from {self.model_path}")
            # self.model = joblib.load(self.model_path)
        except Exception as e:
            logger.error(f"Error loading ML model: {e}")
            self._create_fallback_model()

    def _create_fallback_model(self):
        """Create a simple fallback model based on feature weights."""
        # Simple weighted scoring model as fallback
        self.model = {
            "type": "weighted_scoring",
            "weights": {
                ErrorCategory.SYNTAX: {
                    "has_syntax_keywords": 3.0,
                    "has_exception_type": 2.0,
                },
                ErrorCategory.LOGIC: {
                    "has_null_reference": 2.5,
                    "has_index_error": 2.5,
                    "has_type_error": 2.0,
                },
                ErrorCategory.NETWORK: {
                    "has_network_keywords": 3.0,
                },
                ErrorCategory.DATABASE: {
                    "has_database_keywords": 3.0,
                },
                ErrorCategory.CONFIGURATION: {
                    "has_config_keywords": 2.5,
                    "has_env_variables": 1.5,
                },
                ErrorCategory.SECURITY: {
                    "has_security_keywords": 3.0,
                },
                ErrorCategory.CONCURRENCY: {
                    "has_concurrency_keywords": 3.0,
                },
                ErrorCategory.DEPENDENCY: {
                    "has_import_error": 2.5,
                    "has_dependency_info": 1.0,
                },
            },
        }

    def classify(
        self, features: ClassificationFeatures
    ) -> Tuple[Optional[ErrorCategory], float]:
        """
        Classify error using ML approach.

        Args:
            features: Extracted features

        Returns:
            Tuple of (category, confidence)
        """
        if not self.model:
            return None, 0.0

        if self.model.get("type") == "weighted_scoring":
            return self._weighted_scoring_classify(features)
        else:
            # Placeholder for actual ML model prediction
            return self._ml_model_classify(features)

    def _weighted_scoring_classify(
        self, features: ClassificationFeatures
    ) -> Tuple[Optional[ErrorCategory], float]:
        """Classify using weighted scoring fallback model."""
        category_scores = {}

        for category, weights in self.model["weights"].items():
            score = 0.0

            for feature_name, weight in weights.items():
                if hasattr(features, feature_name):
                    feature_value = getattr(features, feature_name)
                    if isinstance(feature_value, bool):
                        score += weight if feature_value else 0.0
                    else:
                        # Normalize numeric features
                        normalized_value = min(float(feature_value) / 100.0, 1.0)
                        score += weight * normalized_value

            category_scores[category] = score

        if not category_scores:
            return None, 0.0

        # Find best category
        best_category = max(category_scores.keys(), key=lambda k: category_scores[k])
        best_score = category_scores[best_category]

        # Convert score to confidence (0-1)
        max_possible_score = sum(self.model["weights"][best_category].values())
        confidence = (
            min(best_score / max_possible_score, 1.0) if max_possible_score > 0 else 0.0
        )

        return best_category, confidence

    def _ml_model_classify(
        self, features: ClassificationFeatures
    ) -> Tuple[Optional[ErrorCategory], float]:
        """Classify using actual ML model (placeholder)."""
        # This would use the actual trained model
        # For now, return a placeholder
        return ErrorCategory.UNKNOWN, 0.5


class IntelligentClassifier:
    """
    Intelligent error classifier combining rule-based and ML-based approaches.

    This classifier uses multiple strategies to achieve high accuracy:
    1. Rule-based classification for well-defined patterns
    2. ML-based classification for complex patterns
    3. Hybrid approach combining both methods
    4. Language-specific parsing for detailed analysis
    5. Compiler integration for syntax validation
    """

    def __init__(
        self, ml_model_path: Optional[str] = None, use_compiler_integration: bool = True
    ):
        """
        Initialize intelligent classifier.

        Args:
            ml_model_path: Path to ML model file
            use_compiler_integration: Whether to use compiler integration
        """

        # Initialize components
        self.feature_extractor = FeatureExtractor()
        self.rule_classifier = RuleBasedClassifier()
        self.ml_classifier = MLBasedClassifier(ml_model_path)
        self.comprehensive_detector = ComprehensiveErrorDetector()

        # Initialize compiler integration if requested
        self.compiler_integration = None
        if use_compiler_integration:
            try:
                self.compiler_integration = CompilerIntegration()
            except Exception as e:
                logger.warning(f"Could not initialize compiler integration: {e}")

        logger.info("Initialized intelligent error classifier")

    def classify_error(
        self, error_context: ErrorContext, strategy: str = "hybrid"
    ) -> ErrorClassification:
        """
        Classify an error using the specified strategy.

        Args:
            error_context: Error context to classify
            strategy: Classification strategy ("rule_based", "ml_based", "hybrid")

        Returns:
            Error classification result
        """

        # Extract features
        features = self.feature_extractor.extract_features(error_context)

        # Get language-specific analysis
        language_analysis = self._get_language_specific_analysis(error_context)

        # Get compiler diagnostics if available
        compiler_diagnostics = self._get_compiler_diagnostics(error_context)

        # Apply classification strategy
        if strategy == "rule_based":
            classification = self._rule_based_classification(
                features, error_context, language_analysis
            )
        elif strategy == "ml_based":
            classification = self._ml_based_classification(features, error_context)
        else:  # hybrid
            classification = self._hybrid_classification(
                features, error_context, language_analysis, compiler_diagnostics
            )

        # Enhance classification with additional analysis
        self._enhance_classification(
            classification, error_context, language_analysis, compiler_diagnostics
        )

        return classification

    def _get_language_specific_analysis(
        self, error_context: ErrorContext
    ) -> Optional[Dict[str, Any]]:
        """Get language-specific error analysis."""
        try:
            # Handle both dict and ErrorContext object
            if isinstance(error_context, dict):
                language = error_context.get("language", "unknown")
                error_message = error_context.get("message", "")
                source_code_snippet = error_context.get("source_code_snippet")
            else:
                language = error_context.language
                error_message = error_context.error_message
                source_code_snippet = error_context.source_code_snippet

            language_parser = create_language_parser(language)
            if language_parser:
                # Try different types of analysis
                syntax_result = language_parser.parse_syntax_error(
                    error_message, source_code_snippet
                )

                compilation_result = language_parser.parse_compilation_error(
                    error_message, source_code_snippet
                )

                runtime_issues = language_parser.detect_runtime_issues(error_context)

                return {
                    "syntax_analysis": syntax_result,
                    "compilation_analysis": compilation_result,
                    "runtime_analysis": runtime_issues,
                }
        except Exception as e:
            logger.debug(f"Error in language-specific analysis: {e}")

        return None

    def _get_compiler_diagnostics(
        self, error_context: ErrorContext
    ) -> Optional[Dict[str, Any]]:
        """Get compiler diagnostics if available."""
        if not self.compiler_integration or not error_context.source_code_snippet:
            return None

        try:
            # Handle both dict and ErrorContext object
            if isinstance(error_context, dict):
                source_code_snippet = error_context.get("source_code_snippet")
                language = error_context.get("language", "unknown")
                file_path = error_context.get("file_path")
            else:
                source_code_snippet = error_context.source_code_snippet
                language = error_context.language
                file_path = error_context.file_path

            return self.compiler_integration.get_detailed_diagnostics(
                source_code_snippet, language, file_path
            )
        except Exception as e:
            logger.debug(f"Error getting compiler diagnostics: {e}")
            return None

    def _rule_based_classification(
        self,
        features: ClassificationFeatures,
        error_context: ErrorContext,
        language_analysis: Optional[Dict[str, Any]],
    ) -> ErrorClassification:
        """Perform rule-based classification."""

        # Try comprehensive detector first
        comprehensive_result = self.comprehensive_detector.classify_error(error_context)

        # If comprehensive detector has high confidence, use it
        if comprehensive_result.confidence >= 0.8:
            return comprehensive_result

        # Otherwise, use rule-based classifier
        category, confidence = self.rule_classifier.classify(features, error_context)

        if category:
            return ErrorClassification(
                category=category,
                severity=self._determine_severity(category),
                confidence=confidence,
                description=f"Rule-based classification: {category.value}",
                root_cause=self._determine_root_cause(category, error_context),
                suggestions=self._get_suggestions(category, error_context),
            )
        else:
            # Fallback to comprehensive detector result
            return comprehensive_result

    def _ml_based_classification(
        self, features: ClassificationFeatures, error_context: ErrorContext
    ) -> ErrorClassification:
        """Perform ML-based classification."""

        category, confidence = self.ml_classifier.classify(features)

        if category and confidence > 0.3:
            return ErrorClassification(
                category=category,
                severity=self._determine_severity(category),
                confidence=confidence,
                description=f"ML-based classification: {category.value}",
                root_cause=self._determine_root_cause(category, error_context),
                suggestions=self._get_suggestions(category, error_context),
            )
        else:
            # Fallback to rule-based
            return self._rule_based_classification(features, error_context, None)

    def _hybrid_classification(
        self,
        features: ClassificationFeatures,
        error_context: ErrorContext,
        language_analysis: Optional[Dict[str, Any]],
        compiler_diagnostics: Optional[Dict[str, Any]],
    ) -> ErrorClassification:
        """Perform hybrid classification combining rule-based and ML approaches."""

        # Get both classifications
        rule_category, rule_confidence = self.rule_classifier.classify(
            features, error_context
        )
        ml_category, ml_confidence = self.ml_classifier.classify(features)

        # Use language-specific analysis for high-confidence results
        if language_analysis:
            syntax_analysis = language_analysis.get("syntax_analysis")
            compilation_analysis = language_analysis.get("compilation_analysis")
            runtime_analysis = language_analysis.get("runtime_analysis")

            # Syntax errors have high priority
            if syntax_analysis:
                return ErrorClassification(
                    category=syntax_analysis["category"],
                    severity=ErrorSeverity.HIGH,
                    confidence=0.95,
                    description="Syntax error detected by language parser",
                    root_cause=syntax_analysis["error_type"],
                    suggestions=self._get_syntax_suggestions(
                        syntax_analysis["error_type"]
                    ),
                )

            # Compilation errors have high priority
            if compilation_analysis:
                return ErrorClassification(
                    category=compilation_analysis["category"],
                    severity=ErrorSeverity.HIGH,
                    confidence=0.9,
                    description="Compilation error detected by language parser",
                    root_cause=compilation_analysis["error_type"],
                    suggestions=self._get_compilation_suggestions(
                        compilation_analysis["error_type"]
                    ),
                )

            # Runtime errors
            if runtime_analysis:
                issue = runtime_analysis[0]  # Take first issue
                return ErrorClassification(
                    category=issue["category"],
                    severity=ErrorSeverity.MEDIUM,
                    confidence=0.8,
                    description="Runtime error detected by language parser",
                    root_cause=issue["error_type"],
                    suggestions=self._get_runtime_suggestions(issue["error_type"]),
                )

        # Use compiler diagnostics if available
        if compiler_diagnostics and compiler_diagnostics.get("syntax_valid") is False:
            return ErrorClassification(
                category=ErrorCategory.SYNTAX,
                severity=ErrorSeverity.HIGH,
                confidence=0.9,
                description="Syntax error detected by compiler",
                root_cause="compilation_failed",
                suggestions=["Fix syntax errors identified by compiler"]
                + compiler_diagnostics.get("compilation_errors", []),
            )

        # Combine rule-based and ML results
        if rule_confidence >= 0.7 and ml_confidence >= 0.7:
            # Both methods have high confidence
            if rule_category == ml_category:
                # Agreement - use higher confidence
                confidence = max(rule_confidence, ml_confidence)
                category = rule_category
            else:
                # Disagreement - use rule-based as it's more interpretable
                confidence = rule_confidence * 0.9  # Slight penalty for disagreement
                category = rule_category
        elif rule_confidence >= 0.7:
            # Rule-based has high confidence
            confidence = rule_confidence
            category = rule_category
        elif ml_confidence >= 0.7:
            # ML has high confidence
            confidence = ml_confidence
            category = ml_category
        else:
            # Both have low confidence - use comprehensive detector
            comprehensive_result = self.comprehensive_detector.classify_error(
                error_context
            )
            return comprehensive_result

        return ErrorClassification(
            category=category,
            severity=self._determine_severity(category),
            confidence=confidence,
            description=f"Hybrid classification: {category.value}",
            root_cause=self._determine_root_cause(category, error_context),
            suggestions=self._get_suggestions(category, error_context),
            subcategory=f"rule_conf:{rule_confidence:.2f}_ml_conf:{ml_confidence:.2f}",
        )

    def _enhance_classification(
        self,
        classification: ErrorClassification,
        error_context: ErrorContext,
        language_analysis: Optional[Dict[str, Any]],
        compiler_diagnostics: Optional[Dict[str, Any]],
    ):
        """Enhance classification with additional analysis."""

        # Add affected components
        affected_components = []
        # Handle both dict and ErrorContext object
        if isinstance(error_context, dict):
            file_path = error_context.get("file_path")
            service_name = error_context.get("service_name")
        else:
            file_path = error_context.file_path
            service_name = getattr(error_context, "service_name", None)

        if file_path:
            affected_components.append(file_path)
        if service_name:
            affected_components.append(f"service:{service_name}")

        classification.affected_components = affected_components

        # Determine fix complexity
        classification.potential_fix_complexity = self._estimate_fix_complexity(
            classification.category, classification.root_cause
        )

        # Add language-specific suggestions
        if language_analysis:
            additional_suggestions = self._get_language_specific_suggestions(
                language_analysis
            )
            classification.suggestions.extend(additional_suggestions)

        # Add compiler-specific suggestions
        if compiler_diagnostics and compiler_diagnostics.get("compilation_errors"):
            classification.suggestions.extend(
                ["Review compiler errors:"]
                + compiler_diagnostics["compilation_errors"][:3]
            )

    def _determine_severity(self, category: ErrorCategory) -> ErrorSeverity:
        """Determine error severity based on category."""
        severity_map = {
            ErrorCategory.SYNTAX: ErrorSeverity.HIGH,
            ErrorCategory.COMPILATION: ErrorSeverity.HIGH,
            ErrorCategory.SECURITY: ErrorSeverity.CRITICAL,
            ErrorCategory.CONCURRENCY: ErrorSeverity.CRITICAL,
            ErrorCategory.MEMORY: ErrorSeverity.HIGH,
            ErrorCategory.NETWORK: ErrorSeverity.MEDIUM,
            ErrorCategory.DATABASE: ErrorSeverity.MEDIUM,
            ErrorCategory.CONFIGURATION: ErrorSeverity.MEDIUM,
            ErrorCategory.LOGIC: ErrorSeverity.MEDIUM,
            ErrorCategory.RUNTIME: ErrorSeverity.MEDIUM,
            ErrorCategory.DEPENDENCY: ErrorSeverity.LOW,
            ErrorCategory.FILESYSTEM: ErrorSeverity.LOW,
            ErrorCategory.ENVIRONMENT: ErrorSeverity.MEDIUM,
        }
        return severity_map.get(category, ErrorSeverity.MEDIUM)

    def _determine_root_cause(
        self, category: ErrorCategory, error_context: ErrorContext
    ) -> str:
        """Determine root cause based on category and context."""
        # Handle both dict and ErrorContext object
        if isinstance(error_context, dict):
            exception_type = error_context.get("exception_type")
        else:
            exception_type = error_context.exception_type

        if exception_type:
            return f"{category.value}_{exception_type.lower()}"
        else:
            return f"{category.value}_error"

    def _get_suggestions(
        self, category: ErrorCategory, error_context: ErrorContext
    ) -> List[str]:
        """Get suggestions based on error category."""
        suggestions_map = {
            ErrorCategory.SYNTAX: [
                "Check syntax for missing brackets, parentheses, or semicolons",
                "Verify proper indentation",
                "Look for unclosed strings or comments",
            ],
            ErrorCategory.LOGIC: [
                "Add null/undefined checks",
                "Validate array indices before access",
                "Review algorithm logic",
            ],
            ErrorCategory.CONFIGURATION: [
                "Check configuration files",
                "Verify environment variables",
                "Review application settings",
            ],
            ErrorCategory.NETWORK: [
                "Check network connectivity",
                "Verify URLs and endpoints",
                "Review timeout settings",
            ],
            ErrorCategory.DATABASE: [
                "Check database connection",
                "Verify query syntax",
                "Review database permissions",
            ],
            ErrorCategory.SECURITY: [
                "Review security configurations",
                "Check authentication and authorization",
                "Validate input sanitization",
            ],
            ErrorCategory.CONCURRENCY: [
                "Review thread synchronization",
                "Check for race conditions",
                "Implement proper locking mechanisms",
            ],
        }

        return suggestions_map.get(category, ["Manual investigation required"])

    def _get_syntax_suggestions(self, error_type: str) -> List[str]:
        """Get syntax-specific suggestions."""
        return [
            f"Fix {error_type} syntax error",
            "Check language-specific syntax rules",
            "Use IDE syntax highlighting for guidance",
        ]

    def _get_compilation_suggestions(self, error_type: str) -> List[str]:
        """Get compilation-specific suggestions."""
        return [
            f"Resolve {error_type} compilation issue",
            "Check import statements",
            "Verify all dependencies are available",
        ]

    def _get_runtime_suggestions(self, error_type: str) -> List[str]:
        """Get runtime-specific suggestions."""
        return [
            f"Handle {error_type} runtime error",
            "Add appropriate error handling",
            "Validate input parameters",
        ]

    def _get_language_specific_suggestions(
        self, language_analysis: Dict[str, Any]
    ) -> List[str]:
        """Get language-specific suggestions from analysis."""
        suggestions = []

        if language_analysis.get("syntax_analysis"):
            suggestions.append("Review syntax according to language specifications")

        if language_analysis.get("compilation_analysis"):
            suggestions.append("Fix compilation errors before running")

        if language_analysis.get("runtime_analysis"):
            suggestions.append("Add runtime error handling")

        return suggestions

    def _estimate_fix_complexity(self, category: ErrorCategory, root_cause: str) -> str:
        """Estimate complexity of fixing the error."""

        simple_categories = {ErrorCategory.SYNTAX, ErrorCategory.CONFIGURATION}
        complex_categories = {ErrorCategory.CONCURRENCY, ErrorCategory.SECURITY}

        if category in simple_categories:
            return "simple"
        elif category in complex_categories:
            return "complex"
        else:
            return "moderate"

    def get_classification_confidence_level(
        self, confidence: float
    ) -> ClassificationConfidence:
        """Convert numeric confidence to confidence level enum."""
        if confidence > 0.9:
            return ClassificationConfidence.VERY_HIGH
        elif confidence > 0.7:
            return ClassificationConfidence.HIGH
        elif confidence > 0.5:
            return ClassificationConfidence.MEDIUM
        elif confidence > 0.3:
            return ClassificationConfidence.LOW
        else:
            return ClassificationConfidence.VERY_LOW


# Utility functions for integration
def classify_error_intelligently(
    error_context: ErrorContext,
    strategy: str = "hybrid",
    ml_model_path: Optional[str] = None,
) -> ErrorClassification:
    """
    Utility function to classify an error using the intelligent classifier.

    Args:
        error_context: Error context to classify
        strategy: Classification strategy
        ml_model_path: Path to ML model file

    Returns:
        Error classification result
    """
    classifier = IntelligentClassifier(ml_model_path=ml_model_path)
    return classifier.classify_error(error_context, strategy=strategy)


if __name__ == "__main__":
    # Test the intelligent classifier
    print("Intelligent Error Classifier Test")
    print("=================================")

    # Test with different types of errors
    test_contexts = [
        # Syntax error
        ErrorContext(
            error_message="SyntaxError: invalid syntax",
            exception_type="SyntaxError",
            language=LanguageType.PYTHON,
            file_path="test.py",
            line_number=10,
        ),
        # Logic error
        ErrorContext(
            error_message="NullPointerException: Cannot invoke method on null object",
            exception_type="NullPointerException",
            language=LanguageType.JAVA,
            file_path="Test.java",
            line_number=25,
        ),
        # Network error
        ErrorContext(
            error_message="Connection timeout: Unable to connect to database server",
            language=LanguageType.PYTHON,
            service_name="api_service",
        ),
        # Security error
        ErrorContext(
            error_message="PermissionError: Access denied to sensitive resource",
            exception_type="PermissionError",
            language=LanguageType.PYTHON,
        ),
    ]

    classifier = IntelligentClassifier()

    for i, context in enumerate(test_contexts):
        print(f"\nTest Case {i + 1}:")
        print(f"Error: {context.error_message}")

        # Test different strategies
        for strategy in ["rule_based", "ml_based", "hybrid"]:
            result = classifier.classify_error(context, strategy=strategy)

            print(f"\n{strategy.upper()} Strategy:")
            print(f"  Category: {result.category.value}")
            print(f"  Severity: {result.severity.value}")
            print(f"  Confidence: {result.confidence:.2f}")
            print(f"  Root Cause: {result.root_cause}")
            print(f"  Fix Complexity: {result.potential_fix_complexity}")

    print("\nTesting feature extraction:")
    feature_extractor = FeatureExtractor()
    test_features = feature_extractor.extract_features(test_contexts[0])
    print(f"Features for syntax error: {test_features}")
    print(f"Feature array: {test_features.to_array()}")
