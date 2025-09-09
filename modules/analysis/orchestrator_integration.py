"""
Integration module for connecting comprehensive error detection with the orchestrator.

This module provides enhanced error analysis capabilities to the orchestrator,
including comprehensive error detection, intelligent classification, and LLM context management.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .comprehensive_error_detector import (ComprehensiveErrorDetector,
                                           ErrorClassification, ErrorContext,
                                           create_error_context_from_log)
from .intelligent_classifier import IntelligentClassifier
from .language_parsers import CompilerIntegration, create_language_parser
from .llm_context_manager import LLMContextManager
from .mobile_framework_parsers import UnifiedFrameworkParserFactory

logger = logging.getLogger(__name__)


class EnhancedAnalysisEngine:
    """
    Enhanced analysis engine that provides comprehensive error detection and classification
    for the orchestrator with LLM integration capabilities.
    """

    def __init__(
        self,
        storage_dir: Optional[Path] = None,
        use_compiler_integration: bool = True,
        ml_model_path: Optional[str] = None,
    ):
        """
        Initialize the enhanced analysis engine.

        Args:
            storage_dir: Directory for storing LLM contexts
            use_compiler_integration: Whether to use compiler integration
            ml_model_path: Path to ML model for classification
        """

        # Initialize core components
        self.comprehensive_detector = ComprehensiveErrorDetector()
        self.intelligent_classifier = IntelligentClassifier(
            ml_model_path=ml_model_path,
            use_compiler_integration=use_compiler_integration,
        )
        self.context_manager = LLMContextManager(storage_dir=storage_dir)
        self.framework_factory = UnifiedFrameworkParserFactory()

        # Initialize compiler integration if enabled
        self.compiler_integration = None
        if use_compiler_integration:
            try:
                self.compiler_integration = CompilerIntegration()
                logger.info("Compiler integration enabled")
            except Exception as e:
                logger.warning(f"Could not initialize compiler integration: {e}")

        logger.info("Enhanced Analysis Engine initialized")

    def analyze_error_comprehensive(
        self,
        error_log: Dict[str, Any],
        project_root: Optional[str] = None,
        strategy: str = "hybrid",
        store_for_llm: bool = True,
    ) -> Dict[str, Any]:
        """
        Perform comprehensive error analysis.

        Args:
            error_log: Error log entry
            project_root: Project root directory for context
            strategy: Analysis strategy ("rule_based", "ml_based", "hybrid")
            store_for_llm: Whether to store context for LLM processing

        Returns:
            Comprehensive analysis results
        """

        # Create error context from log
        error_context = create_error_context_from_log(error_log, project_root)

        # Perform intelligent classification
        classification = self.intelligent_classifier.classify_error(
            error_context, strategy=strategy
        )

        # Get language-specific analysis
        language_analysis = self._get_language_analysis(error_context)

        # Get framework analysis (mobile and web frameworks)
        framework_analysis = self._get_framework_analysis(error_context, project_root)

        # Get compiler diagnostics if available
        compiler_diagnostics = self._get_compiler_diagnostics(error_context)

        # Correlate with monitoring data if available
        correlation_data = self._correlate_with_monitoring(error_context, error_log)

        # Create comprehensive analysis result
        analysis_result = {
            "analysis_method": "enhanced_comprehensive",
            "engine_version": "1.0.0",
            # Core analysis
            "error_context": error_context.to_dict(),
            "classification": classification.to_dict(),
            # Additional analysis
            "language_analysis": language_analysis,
            "framework_analysis": framework_analysis,
            "compiler_diagnostics": compiler_diagnostics,
            "correlation_data": correlation_data,
            # Analysis metadata
            "strategy_used": strategy,
            "confidence_level": self._determine_confidence_level(
                classification.confidence
            ),
            "complexity_assessment": self._assess_complexity(
                classification, language_analysis
            ),
            # Recommendations
            "recommended_actions": self._generate_recommendations(
                classification, language_analysis
            ),
            "priority_level": self._determine_priority(classification),
        }

        # Store context for LLM processing if requested
        if store_for_llm:
            try:
                context_id = self.context_manager.store_error_context(
                    error_context=error_context,
                    additional_analysis={
                        "classification": classification.to_dict(),
                        "language_specific": language_analysis,
                        "framework_analysis": framework_analysis,
                        "compiler_diagnostics": compiler_diagnostics,
                    },
                    project_context=self._extract_project_context(project_root),
                    monitoring_data=correlation_data,
                )
                analysis_result["llm_context_id"] = context_id
                logger.info(f"Stored LLM context: {context_id}")
            except Exception as e:
                logger.error(f"Error storing LLM context: {e}")
                analysis_result["llm_context_error"] = str(e)

        return analysis_result

    def prepare_llm_prompt(
        self, context_id: str, prompt_type: str = "comprehensive"
    ) -> Dict[str, Any]:
        """
        Prepare an LLM prompt for the given context.

        Args:
            context_id: LLM context ID
            prompt_type: Type of prompt to generate

        Returns:
            Prepared LLM prompt data
        """
        try:
            return self.context_manager.prepare_llm_prompt(context_id, prompt_type)
        except Exception as e:
            logger.error(f"Error preparing LLM prompt for context {context_id}: {e}")
            raise

    def update_llm_results(
        self,
        context_id: str,
        llm_response: Dict[str, Any],
        patch_result: Optional[Dict[str, Any]] = None,
        test_result: Optional[Dict[str, Any]] = None,
    ):
        """
        Update LLM context with results.

        Args:
            context_id: LLM context ID
            llm_response: LLM response data
            patch_result: Patch generation result
            test_result: Test execution result
        """
        try:
            self.context_manager.update_context_with_results(
                context_id, llm_response, patch_result, test_result
            )
            logger.info(f"Updated LLM context {context_id} with results")
        except Exception as e:
            logger.error(f"Error updating LLM context {context_id}: {e}")

    def correlate_multiple_errors(
        self,
        error_logs: List[Dict[str, Any]],
        project_root: Optional[str] = None,
        monitoring_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Correlate multiple errors to detect patterns and cascading failures.

        Args:
            error_logs: List of error log entries
            project_root: Project root directory
            monitoring_data: System monitoring data

        Returns:
            Correlation analysis results
        """

        # Convert logs to error contexts
        error_contexts = [
            create_error_context_from_log(log, project_root) for log in error_logs
        ]

        # Perform correlation analysis
        correlation_result = self.comprehensive_detector.correlate_logs_and_traces(
            error_contexts, monitoring_data
        )

        # Add intelligent classification for each error
        classifications = []
        for context in error_contexts:
            classification = self.intelligent_classifier.classify_error(context)
            classifications.append(classification.to_dict())

        correlation_result["error_classifications"] = classifications
        correlation_result["analysis_method"] = "enhanced_correlation"

        return correlation_result

    def get_analysis_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the analysis engine.

        Returns:
            Analysis engine statistics
        """

        # Get LLM context statistics
        context_stats = self.context_manager.get_context_statistics()

        # Get compiler integration status
        compiler_status = {
            "enabled": self.compiler_integration is not None,
            "available_compilers": {},
        }

        if self.compiler_integration:
            compiler_status["available_compilers"] = (
                self.compiler_integration.available_compilers
            )

        return {
            "engine_version": "1.0.0",
            "components": {
                "comprehensive_detector": True,
                "intelligent_classifier": True,
                "context_manager": True,
                "framework_factory": True,
                "compiler_integration": compiler_status["enabled"],
            },
            "compiler_status": compiler_status,
            "llm_context_stats": context_stats,
        }

    def export_training_data(self, output_file: Path) -> int:
        """
        Export context data for ML model training.

        Args:
            output_file: Output file path

        Returns:
            Number of samples exported
        """
        return self.context_manager.export_contexts_for_training(output_file)

    def _get_language_analysis(
        self, error_context: ErrorContext
    ) -> Optional[Dict[str, Any]]:
        """Get language-specific analysis."""
        try:
            parser = create_language_parser(error_context.language)
            if not parser:
                return None

            syntax_result = parser.parse_syntax_error(
                error_context.error_message, error_context.source_code_snippet
            )

            compilation_result = parser.parse_compilation_error(
                error_context.error_message, error_context.source_code_snippet
            )

            runtime_issues = parser.detect_runtime_issues(error_context)

            return {
                "parser_available": True,
                "language": error_context.language.value,
                "syntax_analysis": syntax_result,
                "compilation_analysis": compilation_result,
                "runtime_analysis": runtime_issues,
            }

        except Exception as e:
            logger.debug(f"Error in language analysis: {e}")
            return {"parser_available": False, "error": str(e)}

    def _get_framework_analysis(
        self, error_context: ErrorContext, project_root: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """Get framework-specific analysis (mobile and web frameworks)."""
        try:
            return self.framework_factory.analyze_error_with_framework_context(
                error_context, project_root
            )
        except Exception as e:
            logger.debug(f"Error in framework analysis: {e}")
            return {"error": str(e)}

    # Maintain backward compatibility
    def _get_mobile_framework_analysis(
        self, error_context: ErrorContext, project_root: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """Get mobile framework-specific analysis. (Backward compatibility method)"""
        return self._get_framework_analysis(error_context, project_root)

    def _get_compiler_diagnostics(
        self, error_context: ErrorContext
    ) -> Optional[Dict[str, Any]]:
        """Get compiler diagnostics if available."""
        if not self.compiler_integration or not error_context.source_code_snippet:
            return None

        try:
            return self.compiler_integration.get_detailed_diagnostics(
                error_context.source_code_snippet,
                error_context.language,
                error_context.file_path,
            )
        except Exception as e:
            logger.debug(f"Error getting compiler diagnostics: {e}")
            return {"error": str(e)}

    def _correlate_with_monitoring(
        self, error_context: ErrorContext, error_log: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Correlate error with monitoring data."""
        correlation = {}

        # Extract timing information
        if "timestamp" in error_log:
            correlation["error_timestamp"] = error_log["timestamp"]

        # Extract service information
        if error_context.service_name:
            correlation["affected_service"] = error_context.service_name

        # Add error frequency analysis (placeholder)
        correlation["frequency_analysis"] = {
            "is_recurring": False,  # Would be determined by actual monitoring data
            "similar_errors_count": 0,
            "time_since_last_occurrence": None,
        }

        return correlation if correlation else None

    def _extract_project_context(
        self, project_root: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """Extract project context information."""
        if not project_root:
            return None

        try:
            project_path = Path(project_root)

            # Detect project type
            project_type = "unknown"
            if (project_path / "package.json").exists():
                project_type = "nodejs"
            elif (project_path / "requirements.txt").exists() or (
                project_path / "pyproject.toml"
            ).exists():
                project_type = "python"
            elif (project_path / "pom.xml").exists():
                project_type = "java_maven"
            elif (project_path / "build.gradle").exists():
                project_type = "java_gradle"
            elif (project_path / "Cargo.toml").exists():
                project_type = "rust"
            elif (project_path / "go.mod").exists():
                project_type = "go"

            return {
                "type": project_type,
                "root_path": str(project_path),
                "structure": {
                    "has_tests": any(project_path.glob("**/test*")),
                    "has_docs": any(project_path.glob("**/doc*")),
                    "has_config": any(project_path.glob("**/*config*")),
                },
            }
        except Exception as e:
            logger.debug(f"Error extracting project context: {e}")
            return None

    def _determine_confidence_level(self, confidence: float) -> str:
        """Determine confidence level string."""
        if confidence >= 0.9:
            return "very_high"
        elif confidence >= 0.7:
            return "high"
        elif confidence >= 0.5:
            return "medium"
        elif confidence >= 0.3:
            return "low"
        else:
            return "very_low"

    def _assess_complexity(
        self,
        classification: ErrorClassification,
        language_analysis: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Assess the complexity of the error and potential fix."""

        complexity_factors = {
            "error_category": classification.category.value,
            "fix_complexity": classification.potential_fix_complexity,
            "language_specific": bool(
                language_analysis and language_analysis.get("parser_available")
            ),
            "has_compilation_errors": bool(
                language_analysis and language_analysis.get("compilation_analysis")
            ),
            "multiple_root_causes": len(classification.suggestions) > 3,
        }

        # Calculate overall complexity score
        complexity_score = 0
        if classification.potential_fix_complexity == "simple":
            complexity_score = 1
        elif classification.potential_fix_complexity == "moderate":
            complexity_score = 2
        else:
            complexity_score = 3

        # Adjust based on other factors
        if complexity_factors["has_compilation_errors"]:
            complexity_score += 1

        if complexity_factors["multiple_root_causes"]:
            complexity_score += 1

        return {
            "factors": complexity_factors,
            "score": min(complexity_score, 5),  # Cap at 5
            "level": ["trivial", "simple", "moderate", "complex", "very_complex"][
                min(complexity_score, 4)
            ],
        }

    def _generate_recommendations(
        self,
        classification: ErrorClassification,
        language_analysis: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Generate actionable recommendations."""
        recommendations = []

        # Primary recommendation based on classification
        recommendations.append(
            {
                "type": "primary_fix",
                "action": f"Address {classification.category.value} error",
                "priority": "high",
                "description": classification.description,
                "suggestions": classification.suggestions[:3],  # Top 3 suggestions
            }
        )

        # Language-specific recommendations
        if language_analysis and language_analysis.get("syntax_analysis"):
            recommendations.append(
                {
                    "type": "syntax_fix",
                    "action": "Fix syntax errors",
                    "priority": "critical",
                    "description": "Syntax errors prevent code execution",
                    "suggestions": [
                        "Review language syntax rules",
                        "Use IDE syntax checking",
                    ],
                }
            )

        # Prevention recommendations
        if classification.category.value in ["logic", "runtime"]:
            recommendations.append(
                {
                    "type": "prevention",
                    "action": "Add error handling",
                    "priority": "medium",
                    "description": "Prevent similar errors in the future",
                    "suggestions": [
                        "Add input validation",
                        "Implement error handling",
                        "Add unit tests for edge cases",
                    ],
                }
            )

        return recommendations

    def _determine_priority(self, classification: ErrorClassification) -> str:
        """Determine error priority level."""
        severity_priority_map = {
            "critical": "urgent",
            "high": "high",
            "medium": "normal",
            "low": "low",
            "info": "low",
        }

        return severity_priority_map.get(classification.severity.value, "normal")


# Factory function for orchestrator integration
def create_enhanced_analysis_engine(
    config: Optional[Dict[str, Any]] = None,
) -> EnhancedAnalysisEngine:
    """
    Create an enhanced analysis engine for orchestrator integration.

    Args:
        config: Configuration dictionary

    Returns:
        Configured enhanced analysis engine
    """

    if config is None:
        config = {}

    # Extract configuration parameters
    storage_dir = None
    if "llm_context_storage" in config:
        storage_dir = Path(config["llm_context_storage"])

    use_compiler_integration = config.get("use_compiler_integration", True)
    ml_model_path = config.get("ml_model_path")

    return EnhancedAnalysisEngine(
        storage_dir=storage_dir,
        use_compiler_integration=use_compiler_integration,
        ml_model_path=ml_model_path,
    )


# Backward compatibility function for existing orchestrator code
def analyze_error_enhanced(
    error_log: Dict[str, Any],
    project_root: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Enhanced error analysis function for backward compatibility.

    Args:
        error_log: Error log entry
        project_root: Project root directory
        config: Analysis configuration

    Returns:
        Enhanced analysis results
    """

    engine = create_enhanced_analysis_engine(config)
    return engine.analyze_error_comprehensive(error_log, project_root)


if __name__ == "__main__":
    # Test the enhanced analysis engine
    print("Enhanced Analysis Engine Test")
    print("============================")

    # Create test configuration
    import tempfile

    test_config = {
        "use_compiler_integration": True,
        "llm_context_storage": os.path.join(tempfile.gettempdir(), "test_llm_contexts"),
    }

    # Create engine
    engine = create_enhanced_analysis_engine(test_config)

    # Test mobile framework error logs
    test_error_logs = [
        # Python error
        {
            "timestamp": "2023-01-01T12:00:00",
            "service": "test_service",
            "level": "ERROR",
            "message": "NameError: name 'undefined_variable' is not defined",
            "exception_type": "NameError",
            "traceback": [
                "Traceback (most recent call last):",
                "  File 'test.py', line 10, in main",
                "    print(undefined_variable)",
                "NameError: name 'undefined_variable' is not defined",
            ],
            "error_details": {
                "exception_type": "NameError",
                "message": "'undefined_variable'",
                "detailed_frames": [
                    {
                        "file": "test.py",
                        "line": 10,
                        "function": "main",
                        "locals": {"other_var": "value"},
                    }
                ],
            },
        },
        # Flutter error
        {
            "timestamp": "2023-01-01T12:01:00",
            "service": "flutter_app",
            "level": "ERROR",
            "message": "FlutterError: RenderFlex overflowed by 15 pixels on the right",
            "exception_type": "FlutterError",
            "error_details": {
                "exception_type": "FlutterError",
                "message": "RenderFlex overflowed by 15 pixels on the right",
                "detailed_frames": [
                    {"file": "lib/main.dart", "line": 25, "function": "build"}
                ],
            },
        },
        # React Native error
        {
            "timestamp": "2023-01-01T12:02:00",
            "service": "rn_app",
            "level": "ERROR",
            "message": "Element type is invalid: expected a string but received undefined",
            "exception_type": "Error",
            "error_details": {
                "exception_type": "Error",
                "message": "Element type is invalid: expected a string but received undefined",
                "detailed_frames": [
                    {
                        "file": "src/components/MyComponent.js",
                        "line": 15,
                        "function": "render",
                    }
                ],
            },
        },
        # Unity error
        {
            "timestamp": "2023-01-01T12:03:00",
            "service": "unity_game",
            "level": "ERROR",
            "message": "NullReferenceException: Object reference not set to an instance of an object",
            "exception_type": "NullReferenceException",
            "error_details": {
                "exception_type": "NullReferenceException",
                "message": "Object reference not set to an instance of an object",
                "detailed_frames": [
                    {
                        "file": "Assets/Scripts/PlayerController.cs",
                        "line": 42,
                        "function": "Update",
                    }
                ],
            },
        },
        # React error
        {
            "timestamp": "2023-01-01T12:04:00",
            "service": "react_app",
            "level": "ERROR",
            "message": "Invalid hook call. Hooks can only be called inside the body of a function component",
            "exception_type": "Error",
            "error_details": {
                "exception_type": "Error",
                "message": "Invalid hook call. Hooks can only be called inside the body of a function component",
                "detailed_frames": [
                    {
                        "file": "src/components/MyComponent.jsx",
                        "line": 20,
                        "function": "handleClick",
                    }
                ],
            },
        },
        # Vue error
        {
            "timestamp": "2023-01-01T12:05:00",
            "service": "vue_app",
            "level": "ERROR",
            "message": "[Vue warn]: Property or method 'myProperty' is not defined on the instance",
            "exception_type": "Error",
            "error_details": {
                "exception_type": "Error",
                "message": "[Vue warn]: Property or method 'myProperty' is not defined on the instance",
                "detailed_frames": [
                    {
                        "file": "src/components/MyComponent.vue",
                        "line": 15,
                        "function": "mounted",
                    }
                ],
            },
        },
        # Angular error
        {
            "timestamp": "2023-01-01T12:06:00",
            "service": "angular_app",
            "level": "ERROR",
            "message": "No provider for HttpClient!",
            "exception_type": "NullInjectorError",
            "error_details": {
                "exception_type": "NullInjectorError",
                "message": "No provider for HttpClient!",
                "detailed_frames": [
                    {
                        "file": "src/app/app.component.ts",
                        "line": 25,
                        "function": "constructor",
                    }
                ],
            },
        },
    ]

    # Test each error type
    for i, test_error_log in enumerate(test_error_logs):
        print(f"\n--- Test Case {i + 1}: {test_error_log['service']} ---")

        # Perform analysis
        result = engine.analyze_error_comprehensive(test_error_log)

        print(f"Analysis method: {result['analysis_method']}")
        print(f"Classification: {result['classification']['category']}")
        print(f"Confidence: {result['classification']['confidence']:.2f}")
        print(f"Priority: {result['priority_level']}")

        # Show framework analysis if available
        if "framework_analysis" in result and result["framework_analysis"]:
            framework_analysis = result["framework_analysis"]
            print(
                f"Detected Framework: {framework_analysis.get('detected_framework', 'unknown')}"
            )
            print(f"Parser Type: {framework_analysis.get('parser_type', 'none')}")

            runtime_issues = framework_analysis.get("runtime_analysis", [])
            if runtime_issues:
                print(f"Framework Issues: {len(runtime_issues)} detected")
                for issue in runtime_issues[:2]:  # Show first 2
                    print(
                        f"  - {issue.get('error_type', 'unknown')}: {issue.get('framework', 'generic')}"
                    )

        print(f"LLM Context ID: {result.get('llm_context_id', 'Not stored')}")

        # Test LLM prompt preparation for first error
        if i == 0 and "llm_context_id" in result:
            try:
                prompt_data = engine.prepare_llm_prompt(result["llm_context_id"])
                print(f"LLM prompt prepared: {prompt_data['template_type']}")
                print(f"Prompt length: {len(prompt_data['user_prompt'])} characters")
            except Exception as e:
                print(f"Error preparing LLM prompt: {e}")

    # Get engine statistics
    print("\n--- Engine Statistics ---")
    stats = engine.get_analysis_statistics()
    print(f"Components: {stats['components']}")
    print(f"LLM contexts: {stats['llm_context_stats']['total_contexts']}")
    print(
        f"Supported frameworks: {engine.framework_factory.get_supported_frameworks()}"
    )
