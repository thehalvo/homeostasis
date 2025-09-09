"""Self-Training Systems for Homeostasis Framework.

This module implements machine learning feedback loops, human-in-the-loop
annotation systems, and continuous learning capabilities that allow the
Homeostasis framework to improve itself over time.
"""

from .adaptive_confidence import (ConfidenceCalculator, ConfidenceContext,
                                  ContextualThresholds, FixComplexity,
                                  ReviewTrigger, SystemCriticality)
from .annotation_system import (AnnotationInterface, AnnotationQualityScorer,
                                AnnotationType, HumanFeedbackCollector)
from .continuous_learning import (DeploymentMonitor, LearningPipeline,
                                  OutcomeTracker)
from .feedback_loops import (AutomatedRetrainer, MLFeedbackLoop,
                             ModelPerformanceTracker, PredictionFeedback)
from .rule_extraction import (AutomatedRuleGenerator, PatternAnalyzer,
                              RuleExtractor)

__all__ = [
    "MLFeedbackLoop",
    "PredictionFeedback",
    "ModelPerformanceTracker",
    "AutomatedRetrainer",
    "AnnotationInterface",
    "AnnotationType",
    "HumanFeedbackCollector",
    "AnnotationQualityScorer",
    "RuleExtractor",
    "PatternAnalyzer",
    "AutomatedRuleGenerator",
    "DeploymentMonitor",
    "OutcomeTracker",
    "LearningPipeline",
    "ConfidenceCalculator",
    "ContextualThresholds",
    "ReviewTrigger",
    "ConfidenceContext",
    "SystemCriticality",
    "FixComplexity",
]
