"""
Enhanced analyzer interface for combining different analysis approaches.

This module provides a comprehensive interface for error analysis, supporting:
1. Rule-based analysis (pattern matching)
2. Machine learning-based analysis
3. LLM-based analysis
4. Hybrid approaches combining multiple methods
"""

import logging
import os
from typing import Any, Dict, List, Optional

from .ai_stub import (
    AVAILABLE_MODELS,
    AIAnalyzer,
    AIModelConfig,
    create_ensemble_analyzer,
    get_available_models,
)
from .ml_analyzer import HybridAnalyzer, MLAnalyzer, get_available_analysis_modes
from .rule_based import FASTAPI_ERROR_PATTERNS, RuleBasedAnalyzer

# Configure logging
logger = logging.getLogger(__name__)


class AnalysisStrategy:
    """Enum-like class for analysis strategies."""

    RULE_BASED_ONLY = "rule_based_only"  # Only use rule-based analysis
    AI_FALLBACK = "ai_fallback"  # Use AI when rule-based confidence is low
    AI_ENHANCED = "ai_enhanced"  # Always use AI and combine with rule-based
    AI_PRIMARY = "ai_primary"  # Use AI as primary, rule-based as fallback
    ENSEMBLE = "ensemble"  # Use ensemble of all available methods
    ML_BASED = "ml_based"  # Use machine learning-based analysis
    HYBRID = "hybrid"  # Use hybrid approach combining rule-based, ML, and LLM


class Analyzer:
    """
    Enhanced unified analyzer that combines different analysis approaches.
    """

    def __init__(
        self,
        strategy: str = AnalysisStrategy.RULE_BASED_ONLY,
        ai_model_type: str = "stub",
        api_key: Optional[str] = None,
        ml_mode: str = "parallel",
        use_llm: bool = False,
        use_ai: Optional[bool] = None,
    ):
        """
        Initialize the analyzer with enhanced options.

        Args:
            strategy: Analysis strategy to use
            ai_model_type: Type of AI model to use when AI is enabled
            api_key: API key for external AI services (if needed)
            ml_mode: ML analysis mode when ML-based or hybrid strategy is used
            use_llm: Whether to use LLM in hybrid strategy
            use_ai: Legacy parameter for backward compatibility. If True, uses AI_FALLBACK strategy
        """
        # Handle legacy use_ai parameter for backward compatibility
        if use_ai is not None:
            self.use_ai = use_ai
            if use_ai:
                strategy = AnalysisStrategy.AI_FALLBACK
            else:
                strategy = AnalysisStrategy.RULE_BASED_ONLY
        else:
            # Set use_ai based on strategy for backward compatibility
            self.use_ai = strategy != AnalysisStrategy.RULE_BASED_ONLY

        self.strategy = strategy

        # Initialize appropriate analyzer based on strategy
        if strategy == AnalysisStrategy.ML_BASED:
            # Use the ML analyzer
            self.analyzer = MLAnalyzer(mode=ml_mode)
            # Add rule_based_analyzer for backward compatibility
            self.rule_based_analyzer = RuleBasedAnalyzer(
                additional_patterns=FASTAPI_ERROR_PATTERNS
            )
            logger.info(f"Initialized ML analyzer with mode: {ml_mode}")
        elif strategy == AnalysisStrategy.HYBRID:
            # Use the hybrid analyzer
            self.analyzer = HybridAnalyzer(
                ml_mode=ml_mode, use_llm=use_llm, llm_api_key=api_key
            )
            # Add rule_based_analyzer for backward compatibility
            self.rule_based_analyzer = RuleBasedAnalyzer(
                additional_patterns=FASTAPI_ERROR_PATTERNS
            )
            logger.info(
                f"Initialized hybrid analyzer (ML mode: {ml_mode}, LLM: {use_llm})"
            )
        else:
            # Use the traditional analyzer
            # Initialize rule-based analyzer
            self.rule_analyzer = RuleBasedAnalyzer(
                additional_patterns=FASTAPI_ERROR_PATTERNS
            )
            # Also set as rule_based_analyzer for backward compatibility
            self.rule_based_analyzer = self.rule_analyzer

            # Store strategy and parameters
            self.ai_model_type = ai_model_type

            # Initialize AI analyzer if needed
            self.ai_analyzer = None
            if strategy != AnalysisStrategy.RULE_BASED_ONLY:
                # Use provided API key or try environment variable
                api_key = api_key or os.environ.get("AI_API_KEY")

                if strategy == AnalysisStrategy.ENSEMBLE:
                    # Create ensemble analyzer with multiple models
                    self.ai_analyzer = create_ensemble_analyzer()
                else:
                    # Create single model analyzer
                    model_config = AVAILABLE_MODELS.get(ai_model_type)
                    if model_config:
                        # Clone the config and update API key
                        config_dict = vars(model_config)
                        config_dict["api_key"] = api_key
                        model_config = AIModelConfig(**config_dict)

                        self.ai_analyzer = AIAnalyzer(
                            model_type=ai_model_type,
                            api_key=api_key,
                            endpoint=model_config.endpoint,
                            model_path=model_config.model_path,
                            parameters=model_config.parameters,
                        )
                    else:
                        # Fallback to stub model
                        self.ai_analyzer = AIAnalyzer()

            logger.info(f"Initialized traditional analyzer with strategy: {strategy}")

    def analyze_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze an error using the configured strategy.

        Args:
            error_data: Error data to analyze

        Returns:
            Analysis results
        """
        # Use ML or hybrid analyzer if applicable
        if self.strategy in [AnalysisStrategy.ML_BASED, AnalysisStrategy.HYBRID]:
            return self.analyzer.analyze_error(error_data)

        # For traditional strategies, use the original logic
        # Always run rule-based analysis for all strategies
        rule_analysis = self.rule_analyzer.analyze_error(error_data)
        rule_confidence = rule_analysis.get("confidence", "low")

        # Convert confidence to numeric for easier comparison
        confidence_map = {"high": 0.8, "medium": 0.5, "low": 0.3}
        if isinstance(rule_confidence, (int, float)):
            # Already numeric
            rule_confidence_score = rule_confidence
            # Determine string confidence based on numeric value
            if rule_confidence >= 0.8:
                rule_confidence_str = "high"
            elif rule_confidence >= 0.5:
                rule_confidence_str = "medium"
            else:
                rule_confidence_str = "low"
        else:
            # String confidence
            rule_confidence_str = rule_confidence
            rule_confidence_score = confidence_map.get(rule_confidence, 0.3)

        # If rule-based only, return rule analysis
        if self.strategy == AnalysisStrategy.RULE_BASED_ONLY or not self.ai_analyzer:
            return {
                **rule_analysis,
                "analysis_method": "rule_based",
                "confidence_score": rule_confidence_score,
            }

        # For other strategies, get AI analysis
        ai_analysis = self.ai_analyzer.analyze_error(error_data)
        ai_confidence_score = ai_analysis.get("confidence", 0.0) if ai_analysis else 0.0

        # Apply the selected strategy
        if self.strategy == AnalysisStrategy.AI_FALLBACK:
            # Use AI analysis only when rule-based confidence is low
            if (
                rule_confidence_str == "low"
                and ai_analysis
                and ai_confidence_score > 0.3
            ):
                return {
                    **ai_analysis,
                    "rule_analysis": rule_analysis,
                    "analysis_method": "ai_fallback",
                    "rule_confidence": rule_confidence_str,
                }
            else:
                return {
                    **rule_analysis,
                    "analysis_method": "rule_based",
                    "confidence_score": rule_confidence_score,
                }

        elif self.strategy == AnalysisStrategy.AI_PRIMARY:
            # Use AI as primary method, fallback to rule-based if AI confidence is low
            if ai_analysis and ai_confidence_score >= 0.5:
                return {
                    **ai_analysis,
                    "rule_analysis": rule_analysis,
                    "analysis_method": "ai_primary",
                    "rule_confidence": rule_confidence_str,
                }
            else:
                return {
                    **rule_analysis,
                    "ai_analysis": ai_analysis,
                    "analysis_method": "rule_based_fallback",
                    "confidence_score": rule_confidence_score,
                }

        elif (
            self.strategy == AnalysisStrategy.AI_ENHANCED
            or self.strategy == AnalysisStrategy.ENSEMBLE
        ):
            # Combine both analyses, using the more detailed information from each
            # For example, use rule-based root cause but AI descriptions and suggestions
            combined = {
                "error_data": error_data,
                "rule_analysis": rule_analysis,
                "ai_analysis": ai_analysis,
                "analysis_method": "combined",
            }

            # Select the analysis with higher confidence as the primary
            if ai_analysis and ai_confidence_score > rule_confidence_score:
                combined.update(
                    {
                        "root_cause": ai_analysis.get(
                            "root_cause", rule_analysis.get("root_cause", "unknown")
                        ),
                        "description": ai_analysis.get(
                            "description", rule_analysis.get("description", "")
                        ),
                        "suggestion": ai_analysis.get(
                            "suggestion", rule_analysis.get("suggestion", "")
                        ),
                        "confidence": ai_confidence_score,
                        "primary_method": "ai",
                    }
                )
            else:
                combined.update(
                    {
                        "root_cause": rule_analysis.get(
                            "root_cause",
                            (
                                ai_analysis.get("root_cause", "unknown")
                                if ai_analysis
                                else "unknown"
                            ),
                        ),
                        "description": rule_analysis.get(
                            "description",
                            ai_analysis.get("description", "") if ai_analysis else "",
                        ),
                        "suggestion": rule_analysis.get(
                            "suggestion",
                            ai_analysis.get("suggestion", "") if ai_analysis else "",
                        ),
                        "confidence": rule_confidence_score,
                        "primary_method": "rule_based",
                    }
                )

            return combined

        # Default fallback
        return {
            **rule_analysis,
            "analysis_method": "rule_based",
            "confidence_score": rule_confidence_score,
        }

    def analyze_errors(
        self, error_data_list: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple errors.

        Args:
            error_data_list: List of error data to analyze

        Returns:
            List of analysis results
        """
        return [self.analyze_error(error_data) for error_data in error_data_list]


def analyze_error_from_log(
    error_log: Dict[str, Any],
    strategy: str = AnalysisStrategy.RULE_BASED_ONLY,
    ai_model_type: str = "stub",
    ml_mode: str = "parallel",
    use_llm: bool = False,
) -> Dict[str, Any]:
    """
    Utility function to analyze a single error log entry with enhanced options.

    Args:
        error_log: Error log data
        strategy: Analysis strategy to use
        ai_model_type: Type of AI model to use when AI is enabled
        ml_mode: ML analysis mode when ML-based or hybrid strategy is used
        use_llm: Whether to use LLM in hybrid strategy

    Returns:
        Analysis results
    """
    analyzer = Analyzer(
        strategy=strategy, ai_model_type=ai_model_type, ml_mode=ml_mode, use_llm=use_llm
    )
    return analyzer.analyze_error(error_log)


def get_available_strategies() -> List[str]:
    """
    Get a list of available analysis strategies.

    Returns:
        List of available strategy names
    """
    return [
        AnalysisStrategy.RULE_BASED_ONLY,
        AnalysisStrategy.AI_FALLBACK,
        AnalysisStrategy.AI_ENHANCED,
        AnalysisStrategy.AI_PRIMARY,
        AnalysisStrategy.ENSEMBLE,
        AnalysisStrategy.ML_BASED,
        AnalysisStrategy.HYBRID,
    ]


def get_available_ai_models() -> List[str]:
    """
    Get a list of available AI models.

    Returns:
        List of available AI model names
    """
    return get_available_models()


def get_available_ml_modes() -> List[str]:
    """
    Get a list of available ML analysis modes.

    Returns:
        List of available ML mode names
    """
    return get_available_analysis_modes()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Example usage with enhanced options
    print("Analysis Module Demo")
    print("===================")

    # Create test error data
    error_data = {
        "timestamp": "2023-01-01T12:00:00",
        "service": "example_service",
        "level": "ERROR",
        "message": "KeyError: 'todo_id'",
        "exception_type": "KeyError",
        "traceback": [
            "Traceback (most recent call last):",
            "  ...",
            "KeyError: 'todo_id'",
        ],
        "error_details": {
            "exception_type": "KeyError",
            "message": "'todo_id'",
            "detailed_frames": [
                {
                    "file": "/app/services/example_service/app.py",
                    "line": 42,
                    "function": "get_todo",
                    "locals": {"todo_db": {"1": {"title": "Example"}}},
                }
            ],
        },
    }

    # Show available strategies and models
    print("\nAvailable Analysis Strategies:")
    for strategy in get_available_strategies():
        print(f"- {strategy}")

    print("\nAvailable AI Models:")
    for model in get_available_ai_models():
        print(f"- {model}")

    print("\nAvailable ML Modes:")
    for mode in get_available_ml_modes():
        print(f"- {mode}")

    # Demo each strategy
    strategies = [
        AnalysisStrategy.RULE_BASED_ONLY,
        AnalysisStrategy.AI_FALLBACK,
        AnalysisStrategy.AI_PRIMARY,
        AnalysisStrategy.ML_BASED,
        AnalysisStrategy.HYBRID,
    ]

    for strategy in strategies:
        print(f"\n\nStrategy: {strategy}")
        print("-" * (len(strategy) + 10))

        analyzer = Analyzer(strategy=strategy)
        result = analyzer.analyze_error(error_data)

        print(f"Analysis Method: {result.get('analysis_method', 'Unknown')}")

        if "root_cause" in result:
            print(f"Root Cause: {result.get('root_cause', 'Unknown')}")
        elif "error_type" in result:
            print(f"Error Type: {result.get('error_type', 'Unknown')}")

        print(
            f"Confidence: {result.get('confidence', result.get('confidence_score', 0.0))}"
        )

        if "primary_method" in result:
            print(f"Primary Method: {result.get('primary_method', 'Unknown')}")

    # Example of using utility function
    print("\n\nUsing utility function")
    print("---------------------")
    result = analyze_error_from_log(
        error_log=error_data,
        strategy=AnalysisStrategy.HYBRID,
        ml_mode="parallel",
        use_llm=False,
    )
    print(f"Analysis Method: {result.get('analysis_method', 'Unknown')}")
    print(f"Primary Method: {result.get('primary_method', 'Unknown')}")
    if "root_cause" in result:
        print(f"Root Cause: {result.get('root_cause', 'Unknown')}")
    elif "error_type" in result:
        print(f"Error Type: {result.get('error_type', 'Unknown')}")
    print(
        f"Confidence: {result.get('confidence', result.get('confidence_score', 0.0))}"
    )
