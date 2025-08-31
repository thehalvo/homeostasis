"""
ML-based analyzer for error classification and analysis.

This module integrates machine learning models with rule-based analysis,
providing a hybrid approach to error classification and root cause analysis.
"""
import os
import logging
from enum import Enum
from typing import Dict, List, Optional, Any, Union

from .rule_based import RuleBasedAnalyzer
from .rule_confidence import ConfidenceLevel
from .models.error_classifier import ErrorClassifierModel
from .ai_stub import AIAnalyzer, AIModelType

# Configure logging
logger = logging.getLogger(__name__)


class MLAnalysisMode(Enum):
    """Enum for different ML analysis modes."""
    
    DISABLED = "disabled"  # ML analysis is disabled
    FALLBACK = "fallback"  # Use ML when rule-based fails
    PARALLEL = "parallel"  # Run both and select the better one
    PRIMARY = "primary"    # Use ML as primary, rule-based as fallback
    ENSEMBLE = "ensemble"  # Combine results from both methods


class MLAnalyzer:
    """
    Enhanced analyzer that combines ML-based and rule-based approaches.
    """
    
    def __init__(self, 
                 mode: Union[MLAnalysisMode, str] = MLAnalysisMode.PARALLEL,
                 ml_model_path: Optional[str] = None,
                 rule_confidence_threshold: float = 0.5,
                 ml_confidence_threshold: float = 0.6):
        """
        Initialize the hybrid ML analyzer.
        
        Args:
            mode: Analysis mode to use
            ml_model_path: Path to the ML model file
            rule_confidence_threshold: Confidence threshold for rule-based results
            ml_confidence_threshold: Confidence threshold for ML-based results
        """
        # Initialize rule-based analyzer
        self.rule_analyzer = RuleBasedAnalyzer()
        
        # Set analysis mode
        self.mode = mode if isinstance(mode, MLAnalysisMode) else MLAnalysisMode(mode)
        
        # Set confidence thresholds
        self.rule_confidence_threshold = rule_confidence_threshold
        self.ml_confidence_threshold = ml_confidence_threshold
        
        # Initialize ML classifier
        self.ml_classifier = None
        if self.mode != MLAnalysisMode.DISABLED:
            self._initialize_ml_model(ml_model_path)
        
        logger.info(f"Initialized MLAnalyzer with mode: {self.mode.value}")
    
    def _initialize_ml_model(self, model_path: Optional[str] = None):
        """
        Initialize the ML classifier model.
        
        Args:
            model_path: Path to the model file
        """
        # Default model path if not provided
        if not model_path:
            model_path = os.path.join(
                os.path.dirname(__file__), 
                "models", 
                "error_classifier.pkl"
            )
        
        # Initialize the classifier
        self.ml_classifier = ErrorClassifierModel(model_path=model_path)
        
        # Try to load the model
        try:
            model_loaded = self.ml_classifier.load()
            if model_loaded:
                logger.info(f"Loaded ML model from {model_path}")
                logger.info(f"Model can classify {len(self.ml_classifier.classes)} error types")
            else:
                logger.warning(f"Failed to load ML model from {model_path}, ML analysis will be disabled")
                self.mode = MLAnalysisMode.DISABLED
        except Exception as e:
            logger.error(f"Error loading ML model: {str(e)}")
            self.mode = MLAnalysisMode.DISABLED
    
    def _convert_rule_confidence(self, confidence: str) -> float:
        """
        Convert rule-based confidence level to numeric value.
        
        Args:
            confidence: Confidence level as string
            
        Returns:
            Confidence as float
        """
        confidence_map = {
            ConfidenceLevel.HIGH.value: 0.85,
            ConfidenceLevel.MEDIUM.value: 0.6,
            ConfidenceLevel.LOW.value: 0.3,
            "high": 0.85,
            "medium": 0.6,
            "low": 0.3
        }
        
        return confidence_map.get(confidence, 0.3)
    
    def _rule_analysis(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform rule-based analysis.
        
        Args:
            error_data: Error data dictionary
            
        Returns:
            Rule-based analysis results
        """
        result = self.rule_analyzer.analyze_error(error_data)
        
        # Add numeric confidence score if not present
        if "confidence_score" not in result and "confidence" in result:
            result["confidence_score"] = self._convert_rule_confidence(result["confidence"])
        
        return result
    
    def _ml_analysis(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform ML-based analysis.
        
        Args:
            error_data: Error data dictionary
            
        Returns:
            ML-based analysis results
        """
        if not self.ml_classifier or self.mode == MLAnalysisMode.DISABLED:
            return {
                "success": False,
                "error": "ML classifier not available",
                "confidence_score": 0.0
            }
        
        try:
            # Get prediction from ML model
            prediction = self.ml_classifier.predict(error_data)
            
            if not prediction["success"]:
                return prediction
            
            # Map ML classification to a format similar to rule-based analysis
            return {
                "success": True,
                "error_type": prediction.get("error_type", "unknown"),
                "confidence_score": prediction.get("confidence", 0.0),
                "alternatives": prediction.get("alternatives", []),
                "analysis_method": "ml"
            }
        except Exception as e:
            logger.error(f"Error in ML analysis: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "confidence_score": 0.0
            }
    
    def _combine_results(self, rule_result: Dict[str, Any], ml_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combine results from rule-based and ML-based analysis.
        
        Args:
            rule_result: Rule-based analysis results
            ml_result: ML-based analysis results
            
        Returns:
            Combined analysis results
        """
        # Get confidence scores
        rule_confidence = rule_result.get("confidence_score", 0.0)
        ml_confidence = ml_result.get("confidence_score", 0.0)
        
        # Create combined result
        combined = {
            "rule_analysis": rule_result,
            "ml_analysis": ml_result,
            "analysis_method": "hybrid"
        }
        
        # Determine primary result based on confidence
        if ml_confidence >= self.ml_confidence_threshold and ml_confidence > rule_confidence:
            # Use ML result as primary
            combined.update({
                "error_type": ml_result.get("error_type", "unknown"),
                "confidence_score": ml_confidence,
                "primary_method": "ml"
            })
        elif rule_confidence >= self.rule_confidence_threshold:
            # Use rule-based result as primary
            combined.update({
                "error_type": rule_result.get("root_cause", "unknown"),
                "confidence_score": rule_confidence,
                "primary_method": "rule"
            })
        else:
            # Low confidence in both, use ensemble approach
            # For now, simply take the highest confidence one
            if ml_confidence > rule_confidence:
                combined.update({
                    "error_type": ml_result.get("error_type", "unknown"),
                    "confidence_score": ml_confidence,
                    "primary_method": "ml_low_confidence"
                })
            else:
                combined.update({
                    "error_type": rule_result.get("root_cause", "unknown"),
                    "confidence_score": rule_confidence,
                    "primary_method": "rule_low_confidence"
                })
        
        return combined
    
    def analyze_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze an error using the configured hybrid approach.
        
        Args:
            error_data: Error data dictionary
            
        Returns:
            Analysis results
        """
        # Always perform rule-based analysis
        rule_result = self._rule_analysis(error_data)
        rule_confidence = rule_result.get("confidence_score", 0.0)
        
        # For DISABLED mode, just return rule-based result
        if self.mode == MLAnalysisMode.DISABLED:
            return rule_result
        
        # Determine if we need ML analysis based on mode
        perform_ml_analysis = False
        
        if self.mode == MLAnalysisMode.FALLBACK:
            # Use ML only if rule confidence is below threshold
            perform_ml_analysis = rule_confidence < self.rule_confidence_threshold
        elif self.mode in [MLAnalysisMode.PARALLEL, MLAnalysisMode.PRIMARY, MLAnalysisMode.ENSEMBLE]:
            # Always perform ML analysis for other modes
            perform_ml_analysis = True
        
        # Perform ML analysis if needed
        ml_result = None
        if perform_ml_analysis:
            ml_result = self._ml_analysis(error_data)
            ml_confidence = ml_result.get("confidence_score", 0.0)
        
        # Return results based on mode
        if self.mode == MLAnalysisMode.FALLBACK:
            if perform_ml_analysis and ml_result and ml_result.get("success", False):
                ml_confidence = ml_result.get("confidence_score", 0.0)
                if ml_confidence >= self.ml_confidence_threshold:
                    # ML confidence is high enough to use
                    return {
                        **ml_result,
                        "rule_analysis": rule_result,
                        "fallback_reason": "low_rule_confidence"
                    }
            
            # Use rule-based result
            return rule_result
        
        elif self.mode == MLAnalysisMode.PRIMARY:
            if ml_result and ml_result.get("success", False):
                ml_confidence = ml_result.get("confidence_score", 0.0)
                if ml_confidence >= self.ml_confidence_threshold:
                    # ML confidence is high enough to use
                    return {
                        **ml_result,
                        "rule_analysis": rule_result
                    }
            
            # Fallback to rule-based result
            return {
                **rule_result,
                "ml_analysis": ml_result,
                "fallback_reason": "low_ml_confidence"
            }
        
        elif self.mode in [MLAnalysisMode.PARALLEL, MLAnalysisMode.ENSEMBLE]:
            # Combine results
            return self._combine_results(rule_result, ml_result)
        
        # Default to rule-based result if no other case matches
        return rule_result
    
    def analyze_errors(self, error_data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze multiple errors.
        
        Args:
            error_data_list: List of error data dictionaries
            
        Returns:
            List of analysis results
        """
        return [self.analyze_error(error_data) for error_data in error_data_list]


class HybridAnalyzer:
    """
    Enhanced hybrid analyzer that combines rule-based, ML, and LLM approaches.
    """
    
    def __init__(self, 
                 ml_mode: Union[MLAnalysisMode, str] = MLAnalysisMode.PARALLEL,
                 ml_model_path: Optional[str] = None,
                 use_llm: bool = False,
                 llm_api_key: Optional[str] = None,
                 llm_endpoint: Optional[str] = None):
        """
        Initialize the hybrid analyzer with multiple analysis methods.
        
        Args:
            ml_mode: ML analysis mode to use
            ml_model_path: Path to the ML model file
            use_llm: Whether to use LLM for analysis
            llm_api_key: API key for LLM service
            llm_endpoint: API endpoint for LLM service
        """
        # Initialize ML analyzer
        self.ml_analyzer = MLAnalyzer(
            mode=ml_mode,
            ml_model_path=ml_model_path
        )
        
        # Initialize LLM analyzer if enabled
        self.llm_analyzer = None
        self.use_llm = use_llm
        
        if use_llm:
            self._initialize_llm_analyzer(llm_api_key, llm_endpoint)
    
    def _initialize_llm_analyzer(self, api_key: Optional[str] = None, endpoint: Optional[str] = None):
        """
        Initialize the LLM analyzer.
        
        Args:
            api_key: API key for LLM service
            endpoint: API endpoint for LLM service
        """
        try:
            # Use environment variable for API key if not provided
            api_key = api_key or os.environ.get("LLM_API_KEY")
            
            if not api_key:
                logger.warning("No API key provided for LLM, disabling LLM analysis")
                self.use_llm = False
                return
            
            # Default endpoint if not provided
            endpoint = endpoint or "https://api.openai.com/v1/chat/completions"
            
            # Initialize LLM analyzer
            self.llm_analyzer = AIAnalyzer(
                model_type=AIModelType.LLM,
                api_key=api_key,
                endpoint=endpoint
            )
            
            logger.info("Initialized LLM analyzer")
        except Exception as e:
            logger.error(f"Error initializing LLM analyzer: {str(e)}")
            self.use_llm = False
    
    def analyze_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze an error using the hybrid approach.
        
        Args:
            error_data: Error data dictionary
            
        Returns:
            Analysis results
        """
        # Get ML analysis result
        ml_result = self.ml_analyzer.analyze_error(error_data)
        
        # Extract rule and ML confidences
        if "rule_analysis" in ml_result:
            rule_confidence = ml_result["rule_analysis"].get("confidence_score", 0.0)
        else:
            rule_confidence = ml_result.get("confidence_score", 0.0)
        
        if "ml_analysis" in ml_result:
            ml_confidence = ml_result["ml_analysis"].get("confidence_score", 0.0)
        else:
            ml_confidence = 0.0
        
        # If both rule and ML confidence are high, skip LLM analysis
        if max(rule_confidence, ml_confidence) >= 0.7 or not self.use_llm or not self.llm_analyzer:
            # Add source information
            ml_result["sources"] = ["rule_based", "ml"]
            return ml_result
        
        # Perform LLM analysis
        try:
            llm_result = self.llm_analyzer.analyze_error(error_data)
            
            # Combine results
            combined_result = {
                **ml_result,
                "llm_analysis": llm_result,
                "sources": ["rule_based", "ml", "llm"]
            }
            
            # If LLM has high confidence, override with LLM result
            llm_confidence = llm_result.get("confidence", 0.0)
            if llm_confidence > 0.7 and llm_confidence > max(rule_confidence, ml_confidence):
                combined_result["error_type"] = llm_result.get("root_cause", combined_result.get("error_type", "unknown"))
                combined_result["confidence_score"] = llm_confidence
                combined_result["primary_method"] = "llm"
            
            return combined_result
        except Exception as e:
            logger.error(f"Error in LLM analysis: {str(e)}")
            ml_result["llm_error"] = str(e)
            ml_result["sources"] = ["rule_based", "ml"]
            return ml_result
    
    def analyze_errors(self, error_data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze multiple errors.
        
        Args:
            error_data_list: List of error data dictionaries
            
        Returns:
            List of analysis results
        """
        return [self.analyze_error(error_data) for error_data in error_data_list]


# Utility functions
def get_available_analysis_modes() -> List[str]:
    """
    Get a list of available ML analysis modes.
    
    Returns:
        List of mode names
    """
    return [mode.value for mode in MLAnalysisMode]


def analyze_with_hybrid_system(error_data: Dict[str, Any], 
                              ml_mode: str = "parallel",
                              use_llm: bool = False) -> Dict[str, Any]:
    """
    Utility function to analyze an error with the hybrid system.
    
    Args:
        error_data: Error data dictionary
        ml_mode: ML analysis mode to use
        use_llm: Whether to use LLM for analysis
        
    Returns:
        Analysis results
    """
    analyzer = HybridAnalyzer(ml_mode=ml_mode, use_llm=use_llm)
    return analyzer.analyze_error(error_data)


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Test error data
    test_error = {
        "timestamp": "2023-01-01T12:00:00",
        "service": "example_service",
        "level": "ERROR",
        "message": "KeyError: 'todo_id'",
        "exception_type": "KeyError",
        "traceback": [
            "Traceback (most recent call last):",
            "  File '/app/services/example_service/app.py', line 42, in get_todo",
            "    todo = todo_db[todo_id]",
            "KeyError: 'todo_id'"
        ],
        "error_details": {
            "exception_type": "KeyError",
            "message": "'todo_id'",
            "detailed_frames": [
                {
                    "file": "/app/services/example_service/app.py",
                    "line": 42,
                    "function": "get_todo",
                    "locals": {"todo_db": {"1": {"title": "Example"}}}
                }
            ]
        }
    }
    
    # Test with different modes
    print("Testing Hybrid Analysis System\n")
    
    print("Available analysis modes:")
    for mode in get_available_analysis_modes():
        print(f"- {mode}")
    
    for mode in ["disabled", "fallback", "parallel", "primary"]:
        print(f"\n\nAnalysis mode: {mode}")
        print("-" * 50)
        
        analyzer = MLAnalyzer(mode=mode)
        result = analyzer.analyze_error(test_error)
        
        print(f"Primary analysis method: {result.get('primary_method', 'N/A')}")
        print(f"Error type: {result.get('error_type', result.get('root_cause', 'Unknown'))}")
        print(f"Confidence: {result.get('confidence_score', 0.0):.2f}")
        
        if "rule_analysis" in result:
            print(f"Rule confidence: {result['rule_analysis'].get('confidence_score', 0.0):.2f}")
        
        if "ml_analysis" in result:
            print(f"ML confidence: {result['ml_analysis'].get('confidence_score', 0.0):.2f}")
    
    # Test the full hybrid analyzer
    print("\n\nFull Hybrid Analyzer (ML + LLM)")
    print("-" * 50)
    
    hybrid_analyzer = HybridAnalyzer(ml_mode="parallel", use_llm=False)
    hybrid_result = hybrid_analyzer.analyze_error(test_error)
    
    print(f"Analysis sources: {hybrid_result.get('sources', [])}")
    print(f"Primary method: {hybrid_result.get('primary_method', 'N/A')}")
    print(f"Error type: {hybrid_result.get('error_type', hybrid_result.get('root_cause', 'Unknown'))}")
    print(f"Confidence: {hybrid_result.get('confidence_score', 0.0):.2f}")