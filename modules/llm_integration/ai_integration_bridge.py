#!/usr/bin/env python3
"""
AI Integration Bridge for Phase 13 Advanced AI Components.

Provides interfaces and hooks for collaboration between Phase 12 LLM integration
and Phase 13 advanced AI tasks including deep learning models, code understanding,
style preservation, and multi-file coordination.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Callable, Protocol
from pathlib import Path
from enum import Enum
import json

from ..analysis.code_style_analyzer import CodeStyleAnalyzer
from ..patch_generation.multi_language_framework_detector import LanguageFrameworkDetector
from .continuous_improvement import get_improvement_engine


class AICapability(Enum):
    """AI capabilities that can be integrated."""
    DEEP_LEARNING_CLASSIFICATION = "deep_learning_classification"
    TRANSFORMER_CODE_UNDERSTANDING = "transformer_code_understanding"
    FINE_TUNED_CODE_GENERATION = "fine_tuned_code_generation"
    STYLE_PRESERVING_GENERATION = "style_preserving_generation"
    SEMANTIC_CODE_ANALYSIS = "semantic_code_analysis"
    HIERARCHICAL_ERROR_CLASSIFICATION = "hierarchical_error_classification"
    ZERO_SHOT_ERROR_DETECTION = "zero_shot_error_detection"
    MULTI_FILE_COORDINATION = "multi_file_coordination"
    CONTINUOUS_LEARNING = "continuous_learning"
    HUMAN_IN_THE_LOOP = "human_in_the_loop"


@dataclass
class CodeContext:
    """Comprehensive code context for AI analysis."""
    file_path: str
    code_content: str
    language: str
    framework: Optional[str] = None
    error_info: Optional[Dict[str, Any]] = None
    style_preferences: Optional[Dict[str, Any]] = None
    related_files: List[str] = field(default_factory=list)
    dependency_graph: Optional[Dict[str, List[str]]] = None
    semantic_metadata: Optional[Dict[str, Any]] = None
    historical_context: Optional[Dict[str, Any]] = None


@dataclass
class AIAnalysisResult:
    """Result from AI analysis."""
    capability: AICapability
    confidence: float
    analysis_data: Dict[str, Any]
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0


class AIModelInterface(Protocol):
    """Protocol for AI model implementations."""
    
    def predict(self, context: CodeContext) -> AIAnalysisResult:
        """Analyze code context and return predictions."""
        ...
    
    def get_capabilities(self) -> List[AICapability]:
        """Get list of capabilities this model supports."""
        ...
    
    def update_model(self, feedback_data: List[Dict[str, Any]]) -> bool:
        """Update model with new feedback data."""
        ...


class StylePreservationInterface(ABC):
    """Interface for style preservation components."""
    
    @abstractmethod
    def analyze_style(self, code_context: CodeContext) -> Dict[str, Any]:
        """Analyze the style of given code."""
        pass
    
    @abstractmethod
    def preserve_style(self, original_code: str, generated_code: str, 
                      style_preferences: Dict[str, Any]) -> str:
        """Apply style preservation to generated code."""
        pass
    
    @abstractmethod
    def validate_style_consistency(self, code_snippets: List[str]) -> Dict[str, Any]:
        """Validate style consistency across multiple code snippets."""
        pass


class CodeUnderstandingInterface(ABC):
    """Interface for code understanding components."""
    
    @abstractmethod
    def extract_semantic_features(self, code_context: CodeContext) -> Dict[str, Any]:
        """Extract semantic features from code."""
        pass
    
    @abstractmethod
    def analyze_code_structure(self, code_context: CodeContext) -> Dict[str, Any]:
        """Analyze code structure and patterns."""
        pass
    
    @abstractmethod
    def identify_code_patterns(self, code_context: CodeContext) -> List[Dict[str, Any]]:
        """Identify common code patterns and idioms."""
        pass
    
    @abstractmethod
    def assess_code_quality(self, code_context: CodeContext) -> Dict[str, Any]:
        """Assess code quality metrics."""
        pass


class MultiFileCoordinationInterface(ABC):
    """Interface for multi-file coordination components."""
    
    @abstractmethod
    def analyze_dependencies(self, file_paths: List[str]) -> Dict[str, List[str]]:
        """Analyze dependencies between files."""
        pass
    
    @abstractmethod
    def coordinate_changes(self, changes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Coordinate changes across multiple files."""
        pass
    
    @abstractmethod
    def validate_change_consistency(self, changes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate consistency of changes across files."""
        pass


class ContinuousLearningInterface(ABC):
    """Interface for continuous learning components."""
    
    @abstractmethod
    def collect_feedback(self, analysis_result: AIAnalysisResult, 
                        actual_outcome: Dict[str, Any]) -> None:
        """Collect feedback from analysis results."""
        pass
    
    @abstractmethod
    def update_models(self, feedback_batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Update models based on feedback batch."""
        pass
    
    @abstractmethod
    def get_learning_metrics(self) -> Dict[str, Any]:
        """Get metrics about learning progress."""
        pass


class AIIntegrationBridge:
    """
    Bridge for integrating Phase 12 LLM capabilities with Phase 13 AI components.
    
    This class provides the integration layer between existing LLM functionality
    and future advanced AI capabilities.
    """
    
    def __init__(self):
        """Initialize the AI integration bridge."""
        self.logger = logging.getLogger(__name__)
        
        # Component registries
        self._ai_models: Dict[AICapability, List[AIModelInterface]] = {}
        self._style_preservers: List[StylePreservationInterface] = []
        self._code_understanders: List[CodeUnderstandingInterface] = []
        self._multi_file_coordinators: List[MultiFileCoordinationInterface] = []
        self._continuous_learners: List[ContinuousLearningInterface] = []
        
        # Integration hooks
        self._pre_analysis_hooks: List[Callable] = []
        self._post_analysis_hooks: List[Callable] = []
        self._style_enhancement_hooks: List[Callable] = []
        self._learning_feedback_hooks: List[Callable] = []
        
        # Initialize built-in components
        self._initialize_builtin_components()
    
    def _initialize_builtin_components(self) -> None:
        """Initialize built-in components that are available."""
        try:
            # Register existing style analyzer as a style preservation component
            style_analyzer = CodeStyleAnalyzer()
            builtin_style_preserver = BuiltinStylePreserver(style_analyzer)
            self.register_style_preserver(builtin_style_preserver)
            
            # Register existing language detector as part of code understanding
            language_detector = LanguageFrameworkDetector()
            builtin_code_understander = BuiltinCodeUnderstandingAdapter(language_detector)
            self.register_code_understander(builtin_code_understander)
            
            # Register continuous improvement engine
            improvement_engine = get_improvement_engine()
            builtin_continuous_learner = BuiltinContinuousLearningAdapter(improvement_engine)
            self.register_continuous_learner(builtin_continuous_learner)
            
            self.logger.info("Initialized built-in AI integration components")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize built-in components: {e}")
    
    def register_ai_model(self, model: AIModelInterface) -> None:
        """Register an AI model for specific capabilities."""
        capabilities = model.get_capabilities()
        for capability in capabilities:
            if capability not in self._ai_models:
                self._ai_models[capability] = []
            self._ai_models[capability].append(model)
        
        self.logger.info(f"Registered AI model with capabilities: {[c.value for c in capabilities]}")
    
    def register_style_preserver(self, preserver: StylePreservationInterface) -> None:
        """Register a style preservation component."""
        self._style_preservers.append(preserver)
        self.logger.info("Registered style preservation component")
    
    def register_code_understander(self, understander: CodeUnderstandingInterface) -> None:
        """Register a code understanding component."""
        self._code_understanders.append(understander)
        self.logger.info("Registered code understanding component")
    
    def register_multi_file_coordinator(self, coordinator: MultiFileCoordinationInterface) -> None:
        """Register a multi-file coordination component."""
        self._multi_file_coordinators.append(coordinator)
        self.logger.info("Registered multi-file coordination component")
    
    def register_continuous_learner(self, learner: ContinuousLearningInterface) -> None:
        """Register a continuous learning component."""
        self._continuous_learners.append(learner)
        self.logger.info("Registered continuous learning component")
    
    def add_analysis_hook(self, hook: Callable, phase: str = "pre") -> None:
        """Add a hook for analysis phases."""
        if phase == "pre":
            self._pre_analysis_hooks.append(hook)
        elif phase == "post":
            self._post_analysis_hooks.append(hook)
        elif phase == "style_enhancement":
            self._style_enhancement_hooks.append(hook)
        elif phase == "learning_feedback":
            self._learning_feedback_hooks.append(hook)
        
        self.logger.info(f"Added {phase} analysis hook")
    
    def analyze_with_ai(self, code_context: CodeContext, 
                       capabilities: List[AICapability]) -> List[AIAnalysisResult]:
        """
        Perform AI analysis using requested capabilities.
        
        Args:
            code_context: Code context to analyze
            capabilities: List of AI capabilities to use
            
        Returns:
            List of analysis results
        """
        # Run pre-analysis hooks
        for hook in self._pre_analysis_hooks:
            try:
                hook(code_context)
            except Exception as e:
                self.logger.error(f"Pre-analysis hook failed: {e}")
        
        results = []
        
        # Enhance context with style and semantic information
        enhanced_context = self._enhance_context(code_context)
        
        # Run AI models for each requested capability
        for capability in capabilities:
            if capability in self._ai_models:
                for model in self._ai_models[capability]:
                    try:
                        result = model.predict(enhanced_context)
                        results.append(result)
                    except Exception as e:
                        self.logger.error(f"AI model prediction failed for {capability}: {e}")
            else:
                self.logger.warning(f"No models available for capability: {capability}")
        
        # Run post-analysis hooks
        for hook in self._post_analysis_hooks:
            try:
                hook(results)
            except Exception as e:
                self.logger.error(f"Post-analysis hook failed: {e}")
        
        return results
    
    def preserve_and_enhance_style(self, original_code: str, generated_code: str, 
                                  code_context: CodeContext) -> str:
        """
        Preserve and enhance code style using available style preservers.
        
        Args:
            original_code: Original code
            generated_code: Generated code to style
            code_context: Code context for style analysis
            
        Returns:
            Style-enhanced code
        """
        if not self._style_preservers:
            return generated_code
        
        # Get style preferences from context or analyze them
        style_preferences = code_context.style_preferences
        if not style_preferences:
            style_preferences = self._analyze_style_preferences(code_context)
        
        enhanced_code = generated_code
        
        # Apply style preservation from all registered preservers
        for preserver in self._style_preservers:
            try:
                enhanced_code = preserver.preserve_style(
                    original_code, enhanced_code, style_preferences
                )
            except Exception as e:
                self.logger.error(f"Style preservation failed: {e}")
        
        # Run style enhancement hooks
        for hook in self._style_enhancement_hooks:
            try:
                enhanced_code = hook(enhanced_code, style_preferences)
            except Exception as e:
                self.logger.error(f"Style enhancement hook failed: {e}")
        
        return enhanced_code
    
    def coordinate_multi_file_changes(self, changes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Coordinate changes across multiple files.
        
        Args:
            changes: List of proposed changes
            
        Returns:
            Coordinated and validated changes
        """
        if not self._multi_file_coordinators:
            return changes
        
        coordinated_changes = changes
        
        # Apply coordination from all registered coordinators
        for coordinator in self._multi_file_coordinators:
            try:
                coordinated_changes = coordinator.coordinate_changes(coordinated_changes)
            except Exception as e:
                self.logger.error(f"Multi-file coordination failed: {e}")
        
        return coordinated_changes
    
    def provide_learning_feedback(self, analysis_results: List[AIAnalysisResult], 
                                 actual_outcomes: List[Dict[str, Any]]) -> None:
        """
        Provide feedback to continuous learning systems.
        
        Args:
            analysis_results: AI analysis results
            actual_outcomes: Actual outcomes from applying the analysis
        """
        # Provide feedback to all continuous learners
        for learner in self._continuous_learners:
            try:
                for result, outcome in zip(analysis_results, actual_outcomes):
                    learner.collect_feedback(result, outcome)
            except Exception as e:
                self.logger.error(f"Learning feedback failed: {e}")
        
        # Run learning feedback hooks
        for hook in self._learning_feedback_hooks:
            try:
                hook(analysis_results, actual_outcomes)
            except Exception as e:
                self.logger.error(f"Learning feedback hook failed: {e}")
    
    def _enhance_context(self, code_context: CodeContext) -> CodeContext:
        """Enhance code context with additional analysis."""
        enhanced_context = CodeContext(
            file_path=code_context.file_path,
            code_content=code_context.code_content,
            language=code_context.language,
            framework=code_context.framework,
            error_info=code_context.error_info,
            style_preferences=code_context.style_preferences,
            related_files=code_context.related_files.copy(),
            dependency_graph=code_context.dependency_graph,
            semantic_metadata=code_context.semantic_metadata,
            historical_context=code_context.historical_context
        )
        
        # Enhance with style analysis if not already present
        if not enhanced_context.style_preferences:
            enhanced_context.style_preferences = self._analyze_style_preferences(enhanced_context)
        
        # Enhance with semantic metadata if not already present
        if not enhanced_context.semantic_metadata:
            enhanced_context.semantic_metadata = self._extract_semantic_metadata(enhanced_context)
        
        return enhanced_context
    
    def _analyze_style_preferences(self, code_context: CodeContext) -> Dict[str, Any]:
        """Analyze style preferences from code context."""
        if not self._style_preservers:
            return {}
        
        try:
            # Use the first style preserver to analyze style
            return self._style_preservers[0].analyze_style(code_context)
        except Exception as e:
            self.logger.error(f"Style analysis failed: {e}")
            return {}
    
    def _extract_semantic_metadata(self, code_context: CodeContext) -> Dict[str, Any]:
        """Extract semantic metadata from code context."""
        if not self._code_understanders:
            return {}
        
        try:
            # Use the first code understander to extract semantic features
            return self._code_understanders[0].extract_semantic_features(code_context)
        except Exception as e:
            self.logger.error(f"Semantic metadata extraction failed: {e}")
            return {}
    
    def get_available_capabilities(self) -> List[AICapability]:
        """Get list of available AI capabilities."""
        return list(self._ai_models.keys())
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get status of AI integration components."""
        return {
            "ai_models": {
                capability.value: len(models) 
                for capability, models in self._ai_models.items()
            },
            "style_preservers": len(self._style_preservers),
            "code_understanders": len(self._code_understanders),
            "multi_file_coordinators": len(self._multi_file_coordinators),
            "continuous_learners": len(self._continuous_learners),
            "hooks": {
                "pre_analysis": len(self._pre_analysis_hooks),
                "post_analysis": len(self._post_analysis_hooks),
                "style_enhancement": len(self._style_enhancement_hooks),
                "learning_feedback": len(self._learning_feedback_hooks)
            }
        }


# Built-in adapter implementations for existing components

class BuiltinStylePreserver(StylePreservationInterface):
    """Adapter for existing CodeStyleAnalyzer."""
    
    def __init__(self, style_analyzer: CodeStyleAnalyzer):
        self.style_analyzer = style_analyzer
    
    def analyze_style(self, code_context: CodeContext) -> Dict[str, Any]:
        """Analyze style using existing CodeStyleAnalyzer."""
        return self.style_analyzer.analyze_style(
            code_context.code_content, 
            code_context.language
        )
    
    def preserve_style(self, original_code: str, generated_code: str, 
                      style_preferences: Dict[str, Any]) -> str:
        """Apply style preservation using existing analyzer."""
        return self.style_analyzer.apply_style_preferences(
            generated_code, 
            style_preferences
        )
    
    def validate_style_consistency(self, code_snippets: List[str]) -> Dict[str, Any]:
        """Validate style consistency across code snippets."""
        if not code_snippets:
            return {"consistent": True, "issues": []}
        
        # Analyze style of first snippet as reference
        reference_style = self.style_analyzer.analyze_style(code_snippets[0])
        
        issues = []
        for i, snippet in enumerate(code_snippets[1:], 1):
            snippet_style = self.style_analyzer.analyze_style(snippet)
            
            # Compare key style attributes
            for key in ["indentation_type", "indentation_size", "quote_style"]:
                if reference_style.get(key) != snippet_style.get(key):
                    issues.append(f"Snippet {i}: {key} mismatch")
        
        return {
            "consistent": len(issues) == 0,
            "issues": issues,
            "reference_style": reference_style
        }


class BuiltinCodeUnderstandingAdapter(CodeUnderstandingInterface):
    """Adapter for existing language detection and analysis components."""
    
    def __init__(self, language_detector: LanguageFrameworkDetector):
        self.language_detector = language_detector
    
    def extract_semantic_features(self, code_context: CodeContext) -> Dict[str, Any]:
        """Extract semantic features using existing detection."""
        file_path = Path(code_context.file_path)
        
        # Use existing detector to get language and framework info
        detected_info = self.language_detector.detect_language_and_framework(
            file_path, code_context.code_content
        )
        
        return {
            "language": detected_info.get("language"),
            "framework": detected_info.get("framework"),
            "confidence": detected_info.get("confidence", 0.0),
            "framework_version": detected_info.get("framework_version"),
            "additional_frameworks": detected_info.get("additional_frameworks", [])
        }
    
    def analyze_code_structure(self, code_context: CodeContext) -> Dict[str, Any]:
        """Basic code structure analysis."""
        code = code_context.code_content
        lines = code.split('\n')
        
        return {
            "line_count": len(lines),
            "non_empty_lines": len([line for line in lines if line.strip()]),
            "comment_lines": len([line for line in lines if line.strip().startswith('#')]),
            "function_count": code.count('def ') if code_context.language == 'python' else 0,
            "class_count": code.count('class ') if code_context.language == 'python' else 0
        }
    
    def identify_code_patterns(self, code_context: CodeContext) -> List[Dict[str, Any]]:
        """Identify basic code patterns."""
        patterns = []
        code = code_context.code_content
        
        # Basic pattern detection
        if 'try:' in code and 'except' in code:
            patterns.append({"pattern": "error_handling", "type": "try_except"})
        
        if 'import ' in code:
            patterns.append({"pattern": "imports", "type": "module_imports"})
        
        if code_context.language == 'python':
            if 'def ' in code:
                patterns.append({"pattern": "functions", "type": "function_definitions"})
            if 'class ' in code:
                patterns.append({"pattern": "classes", "type": "class_definitions"})
        
        return patterns
    
    def assess_code_quality(self, code_context: CodeContext) -> Dict[str, Any]:
        """Basic code quality assessment."""
        code = code_context.code_content
        lines = code.split('\n')
        
        # Basic quality metrics
        avg_line_length = sum(len(line) for line in lines) / len(lines) if lines else 0
        max_line_length = max(len(line) for line in lines) if lines else 0
        
        quality_score = 1.0
        issues = []
        
        # Check for long lines
        if max_line_length > 120:
            quality_score -= 0.1
            issues.append("Lines too long (>120 characters)")
        
        # Check for very short functions (potential code smell)
        if code_context.language == 'python':
            function_lines = [i for i, line in enumerate(lines) if line.strip().startswith('def ')]
            for func_line in function_lines:
                next_func = next((i for i in function_lines if i > func_line), len(lines))
                if next_func - func_line < 3:
                    quality_score -= 0.05
                    issues.append("Very short function detected")
        
        return {
            "score": max(0.0, quality_score),
            "issues": issues,
            "metrics": {
                "avg_line_length": avg_line_length,
                "max_line_length": max_line_length,
                "line_count": len(lines)
            }
        }


class BuiltinContinuousLearningAdapter(ContinuousLearningInterface):
    """Adapter for existing continuous improvement engine."""
    
    def __init__(self, improvement_engine):
        self.improvement_engine = improvement_engine
    
    def collect_feedback(self, analysis_result: AIAnalysisResult, 
                        actual_outcome: Dict[str, Any]) -> None:
        """Collect feedback using existing improvement engine."""
        from .continuous_improvement import PatchFeedback, FeedbackType, PatchOutcome
        import time
        
        # Convert analysis result to patch feedback
        outcome = PatchOutcome.SUCCESS if actual_outcome.get("success", False) else PatchOutcome.FAILURE
        
        feedback = PatchFeedback(
            patch_id=actual_outcome.get("patch_id", "unknown"),
            feedback_type=FeedbackType.SYSTEM,
            outcome=outcome,
            timestamp=time.time(),
            context={
                "capability": analysis_result.capability.value,
                "confidence": analysis_result.confidence,
                "analysis_data": analysis_result.analysis_data
            },
            metrics={"processing_time": analysis_result.processing_time},
            feedback_source="ai_integration_bridge",
            confidence_score=analysis_result.confidence
        )
        
        self.improvement_engine.record_patch_feedback(feedback)
    
    def update_models(self, feedback_batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Update models using improvement engine patterns."""
        # Trigger pattern analysis
        patterns = self.improvement_engine.analyze_patterns()
        
        return {
            "patterns_found": len(patterns),
            "timestamp": time.time()
        }
    
    def get_learning_metrics(self) -> Dict[str, Any]:
        """Get learning metrics from improvement engine."""
        return self.improvement_engine.get_statistics()


# Global instance
_ai_integration_bridge = None

def get_ai_integration_bridge() -> AIIntegrationBridge:
    """Get the global AI integration bridge instance."""
    global _ai_integration_bridge
    if _ai_integration_bridge is None:
        _ai_integration_bridge = AIIntegrationBridge()
    return _ai_integration_bridge