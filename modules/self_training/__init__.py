"""Self-Training Systems for Homeostasis Framework.

This module implements machine learning feedback loops, human-in-the-loop
annotation systems, and continuous learning capabilities that allow the
Homeostasis framework to improve itself over time.
"""

from .feedback_loops import (
    MLFeedbackLoop,
    PredictionFeedback,
    ModelPerformanceTracker,
    AutomatedRetrainer
)
from .annotation_system import (
    AnnotationInterface,
    AnnotationType,
    HumanFeedbackCollector,
    AnnotationQualityScorer
)
from .rule_extraction import (
    RuleExtractor,
    PatternAnalyzer,
    AutomatedRuleGenerator
)
from .continuous_learning import (
    DeploymentMonitor,
    OutcomeTracker,
    LearningPipeline
)
from .adaptive_confidence import (
    ConfidenceCalculator,
    ContextualThresholds,
    ReviewTrigger,
    ConfidenceContext,
    SystemCriticality,
    FixComplexity
)

__all__ = [
    'MLFeedbackLoop',
    'PredictionFeedback',
    'ModelPerformanceTracker',
    'AutomatedRetrainer',
    'AnnotationInterface',
    'AnnotationType',
    'HumanFeedbackCollector',
    'AnnotationQualityScorer',
    'RuleExtractor',
    'PatternAnalyzer',
    'AutomatedRuleGenerator',
    'DeploymentMonitor',
    'OutcomeTracker',
    'LearningPipeline',
    'ConfidenceCalculator',
    'ContextualThresholds',
    'ReviewTrigger',
    'ConfidenceContext',
    'SystemCriticality',
    'FixComplexity'
]