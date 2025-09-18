"""Adaptive Confidence Thresholds Based on Context.

This module implements dynamic confidence scoring that adjusts thresholds
based on historical success rates, system criticality, and fix complexity.
"""

import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .continuous_learning import FixOutcome

logger = logging.getLogger(__name__)


class SystemCriticality(Enum):
    """Criticality levels for different system components."""

    CRITICAL = "critical"  # Core system functionality
    HIGH = "high"  # Important features
    MEDIUM = "medium"  # Standard functionality
    LOW = "low"  # Non-essential features
    EXPERIMENTAL = "experimental"  # Experimental or beta features


class FixComplexity(Enum):
    """Complexity levels for fixes."""

    TRIVIAL = "trivial"  # Single line changes
    SIMPLE = "simple"  # Few lines, single file
    MODERATE = "moderate"  # Multiple changes, single file
    COMPLEX = "complex"  # Multiple files
    VERY_COMPLEX = "very_complex"  # System-wide changes


@dataclass
class ConfidenceContext:
    """Context information for confidence calculation."""

    error_type: str
    error_severity: str
    affected_components: List[str]
    fix_complexity: FixComplexity
    system_criticality: SystemCriticality
    historical_success_rate: Optional[float] = None
    recent_failures: int = 0
    time_of_day: Optional[str] = None  # 'business_hours', 'after_hours', 'weekend'
    deployment_frequency: Optional[float] = None  # Deploys per day

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["fix_complexity"] = self.fix_complexity.value
        data["system_criticality"] = self.system_criticality.value
        return data


@dataclass
class ConfidenceThreshold:
    """Dynamic confidence threshold for a specific context."""

    base_threshold: float
    current_threshold: float
    adjustment_history: List[
        Tuple[float, float, str]
    ]  # (timestamp, adjustment, reason)
    context_hash: str
    last_updated: float

    def apply_adjustment(self, adjustment: float, reason: str) -> None:
        """Apply an adjustment to the threshold."""
        self.current_threshold = max(0.0, min(1.0, self.current_threshold + adjustment))
        self.adjustment_history.append((time.time(), adjustment, reason))
        self.last_updated = time.time()

    def decay_to_base(self, decay_rate: float = 0.01) -> None:
        """Gradually decay threshold back to base value."""
        if abs(self.current_threshold - self.base_threshold) > 0.001:
            diff = self.base_threshold - self.current_threshold
            adjustment = diff * decay_rate
            self.apply_adjustment(adjustment, "decay_to_base")


class ConfidenceCalculator:
    """Calculates adaptive confidence scores for fixes."""

    def __init__(
        self,
        storage_dir: Path = Path("data/confidence_thresholds"),
        base_threshold: float = 0.7,
        learning_rate: float = 0.1,
    ):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.base_threshold = base_threshold
        self.learning_rate = learning_rate

        self.thresholds: Dict[str, ConfidenceThreshold] = {}
        self.performance_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=100)
        )
        self.context_weights = {
            "historical_success": 0.3,
            "system_criticality": 0.25,
            "fix_complexity": 0.25,
            "recent_failures": 0.1,
            "time_context": 0.1,
        }

        # Load existing thresholds
        self._load_thresholds()

    def calculate_confidence(
        self, model_confidence: float, context: ConfidenceContext
    ) -> Dict[str, Any]:
        """Calculate adjusted confidence based on context."""
        # Get or create threshold for this context
        context_hash = self._hash_context(context)
        threshold = self._get_or_create_threshold(context_hash, context)

        # Calculate confidence adjustments
        adjustments = self._calculate_adjustments(context)

        # Apply adjustments to model confidence
        adjusted_confidence = model_confidence
        total_adjustment = 0.0

        for factor, adjustment in adjustments.items():
            weight = self.context_weights.get(factor, 0.1)
            weighted_adjustment = adjustment * weight
            adjusted_confidence *= 1 + weighted_adjustment
            total_adjustment += weighted_adjustment

        # Ensure confidence is in valid range
        adjusted_confidence = max(0.0, min(1.0, adjusted_confidence))

        # Determine if review is needed
        requires_review = adjusted_confidence < threshold.current_threshold

        result = {
            "original_confidence": model_confidence,
            "adjusted_confidence": adjusted_confidence,
            "threshold": threshold.current_threshold,
            "requires_review": requires_review,
            "adjustments": adjustments,
            "total_adjustment": total_adjustment,
            "context_hash": context_hash,
            "reasons": self._generate_reasoning(adjustments, context),
        }

        # Record for learning
        self._record_prediction(context_hash, adjusted_confidence)

        return result

    def update_from_outcome(
        self, context_hash: str, predicted_confidence: float, actual_outcome: FixOutcome
    ) -> None:
        """Update thresholds based on actual outcomes."""
        if context_hash not in self.thresholds:
            logger.warning(f"Unknown context hash: {context_hash}")
            return

        threshold = self.thresholds[context_hash]

        # Calculate prediction error
        actual_success = 1.0 if actual_outcome == FixOutcome.SUCCESS else 0.0
        error = predicted_confidence - actual_success

        # Update threshold based on error
        if abs(error) > 0.3:  # Significant error
            if error > 0:  # Overconfident
                adjustment = self.learning_rate * abs(error)
                threshold.apply_adjustment(adjustment, f"overconfident_by_{error:.2f}")
            else:  # Underconfident
                adjustment = -self.learning_rate * abs(error)
                threshold.apply_adjustment(adjustment, f"underconfident_by_{error:.2f}")

        # Update performance history
        self.performance_history[context_hash].append(
            {
                "predicted": predicted_confidence,
                "actual": actual_success,
                "error": error,
                "timestamp": time.time(),
            }
        )

        # Save updated threshold
        self._save_threshold(context_hash, threshold)

    def _calculate_adjustments(self, context: ConfidenceContext) -> Dict[str, float]:
        """Calculate confidence adjustments for each factor."""
        adjustments = {}

        # Historical success rate adjustment
        if context.historical_success_rate is not None:
            # Lower confidence if historical success is low
            if context.historical_success_rate < 0.5:
                adjustments["historical_success"] = -0.3 * (
                    0.5 - context.historical_success_rate
                )
            elif context.historical_success_rate > 0.8:
                adjustments["historical_success"] = 0.1 * (
                    context.historical_success_rate - 0.8
                )

        # System criticality adjustment
        criticality_adjustments = {
            SystemCriticality.CRITICAL: -0.3,  # Much more conservative
            SystemCriticality.HIGH: -0.15,  # More conservative
            SystemCriticality.MEDIUM: 0.0,  # Neutral
            SystemCriticality.LOW: 0.1,  # Slightly more permissive
            SystemCriticality.EXPERIMENTAL: 0.2,  # More permissive
        }
        adjustments["system_criticality"] = criticality_adjustments.get(
            context.system_criticality, 0.0
        )

        # Fix complexity adjustment
        complexity_adjustments = {
            FixComplexity.TRIVIAL: 0.2,  # Higher confidence for simple fixes
            FixComplexity.SIMPLE: 0.1,
            FixComplexity.MODERATE: 0.0,
            FixComplexity.COMPLEX: -0.2,
            FixComplexity.VERY_COMPLEX: -0.4,  # Much lower confidence for complex fixes
        }
        adjustments["fix_complexity"] = complexity_adjustments.get(
            context.fix_complexity, 0.0
        )

        # Recent failures adjustment
        if context.recent_failures > 0:
            adjustments["recent_failures"] = -0.1 * min(context.recent_failures, 5)

        # Time context adjustment
        if context.time_of_day:
            time_adjustments = {
                "business_hours": 0.0,  # Normal confidence during business hours
                "after_hours": -0.1,  # Slightly more conservative after hours
                "weekend": -0.15,  # More conservative on weekends
            }
            adjustments["time_context"] = time_adjustments.get(context.time_of_day, 0.0)

        return adjustments

    def _generate_reasoning(
        self, adjustments: Dict[str, float], context: ConfidenceContext
    ) -> List[str]:
        """Generate human-readable reasoning for confidence adjustments."""
        reasons = []

        for factor, adjustment in adjustments.items():
            if abs(adjustment) < 0.01:
                continue

            if factor == "historical_success":
                if adjustment < 0:
                    reasons.append(
                        f"Historical success rate is low ({context.historical_success_rate:.1%})"
                    )
                else:
                    reasons.append(
                        f"Historical success rate is high ({context.historical_success_rate:.1%})"
                    )

            elif factor == "system_criticality":
                if adjustment < 0:
                    reasons.append(
                        f"Affecting {context.system_criticality.value} system components"
                    )

            elif factor == "fix_complexity":
                if adjustment < 0:
                    reasons.append(f"Fix is {context.fix_complexity.value} complexity")
                elif adjustment > 0:
                    reasons.append(f"Fix is {context.fix_complexity.value} (low risk)")

            elif factor == "recent_failures":
                reasons.append(
                    f"Recent failures detected ({context.recent_failures} in last period)"
                )

            elif factor == "time_context":
                if adjustment < 0:
                    reasons.append(f"Deployment during {context.time_of_day}")

        return reasons

    def _hash_context(self, context: ConfidenceContext) -> str:
        """Generate a hash for the context."""
        # Create a stable hash from key context attributes
        key_parts = [
            context.error_type,
            context.system_criticality.value,
            context.fix_complexity.value,
            str(sorted(context.affected_components)),
        ]

        import hashlib

        context_str = "|".join(key_parts)
        return hashlib.sha256(context_str.encode()).hexdigest()[:16]

    def _get_or_create_threshold(
        self, context_hash: str, context: ConfidenceContext
    ) -> ConfidenceThreshold:
        """Get existing threshold or create new one."""
        if context_hash in self.thresholds:
            return self.thresholds[context_hash]

        # Create new threshold based on context
        base = self.base_threshold

        # Adjust base based on criticality
        if context.system_criticality == SystemCriticality.CRITICAL:
            base += 0.15
        elif context.system_criticality == SystemCriticality.HIGH:
            base += 0.1

        threshold = ConfidenceThreshold(
            base_threshold=base,
            current_threshold=base,
            adjustment_history=[],
            context_hash=context_hash,
            last_updated=time.time(),
        )

        self.thresholds[context_hash] = threshold
        self._save_threshold(context_hash, threshold)

        return threshold

    def _record_prediction(self, context_hash: str, confidence: float) -> None:
        """Record a prediction for later analysis."""
        record = {
            "context_hash": context_hash,
            "confidence": confidence,
            "timestamp": time.time(),
        }

        # Save to predictions file
        predictions_file = self.storage_dir / "predictions.jsonl"
        with open(predictions_file, "a") as f:
            f.write(json.dumps(record) + "\n")

    def _save_threshold(
        self, context_hash: str, threshold: ConfidenceThreshold
    ) -> None:
        """Save threshold to disk."""
        threshold_file = self.storage_dir / f"threshold_{context_hash}.json"

        data = {
            "base_threshold": threshold.base_threshold,
            "current_threshold": threshold.current_threshold,
            "adjustment_history": threshold.adjustment_history,
            "context_hash": threshold.context_hash,
            "last_updated": threshold.last_updated,
        }

        with open(threshold_file, "w") as f:
            json.dump(data, f, indent=2)

    def _load_thresholds(self) -> None:
        """Load existing thresholds from disk."""
        if self.storage_dir.exists():
            for threshold_file in self.storage_dir.glob("threshold_*.json"):
                with open(threshold_file, "r") as f:
                    data = json.load(f)
                    threshold = ConfidenceThreshold(**data)
                    self.thresholds[threshold.context_hash] = threshold


class ContextualThresholds:
    """Manages contextual thresholds across the system."""

    def __init__(self, confidence_calculator: ConfidenceCalculator):
        self.confidence_calculator = confidence_calculator
        self.threshold_groups: Dict[str, List[str]] = defaultdict(list)
        self.group_performance: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"success": 0, "total": 0, "success_rate": 0.0}
        )

    def get_threshold_for_context(
        self, error_type: str, system_component: str, deployment_context: Dict[str, Any]
    ) -> float:
        """Get the appropriate threshold for a given context."""
        # Determine system criticality
        criticality = self._determine_criticality(system_component)

        # Determine fix complexity
        complexity = self._estimate_complexity(deployment_context)

        # Get historical success rate
        success_rate = self._get_historical_success_rate(error_type)

        # Get recent failures
        recent_failures = self._count_recent_failures(error_type, hours=24)

        # Determine time context
        time_context = self._get_time_context()

        # Create context
        context = ConfidenceContext(
            error_type=error_type,
            error_severity=deployment_context.get("severity", "medium"),
            affected_components=[system_component],
            fix_complexity=complexity,
            system_criticality=criticality,
            historical_success_rate=success_rate,
            recent_failures=recent_failures,
            time_of_day=time_context,
        )

        # Calculate confidence
        result = self.confidence_calculator.calculate_confidence(
            deployment_context.get("model_confidence", 0.5), context
        )

        return float(result["threshold"])

    def update_group_performance(
        self, group_name: str, context_hash: str, success: bool
    ) -> None:
        """Update performance metrics for a threshold group."""
        self.threshold_groups[group_name].append(context_hash)

        group_stats = self.group_performance[group_name]
        group_stats["total"] += 1
        if success:
            group_stats["success"] += 1

        # Calculate success rate
        group_stats["success_rate"] = group_stats["success"] / group_stats["total"]

    def get_group_recommendations(self) -> List[Dict[str, Any]]:
        """Get recommendations for threshold adjustments by group."""
        recommendations = []

        for group_name, group_stats in self.group_performance.items():
            if group_stats["total"] < 10:
                continue

            success_rate = group_stats["success_rate"]

            if success_rate < 0.5:
                recommendations.append(
                    {
                        "group": group_name,
                        "action": "increase_thresholds",
                        "reason": f"Low success rate ({success_rate:.1%})",
                        "suggested_adjustment": 0.1,
                    }
                )
            elif success_rate > 0.9 and group_stats["total"] > 50:
                recommendations.append(
                    {
                        "group": group_name,
                        "action": "decrease_thresholds",
                        "reason": f"Very high success rate ({success_rate:.1%})",
                        "suggested_adjustment": -0.05,
                    }
                )

        return recommendations

    def _determine_criticality(self, component: str) -> SystemCriticality:
        """Determine system criticality from component name."""
        critical_patterns = ["auth", "payment", "security", "database", "api"]
        high_patterns = ["user", "core", "service", "controller"]
        low_patterns = ["test", "example", "demo", "sandbox"]
        experimental_patterns = ["beta", "experimental", "alpha"]

        component_lower = component.lower()

        if any(p in component_lower for p in critical_patterns):
            return SystemCriticality.CRITICAL
        elif any(p in component_lower for p in high_patterns):
            return SystemCriticality.HIGH
        elif any(p in component_lower for p in low_patterns):
            return SystemCriticality.LOW
        elif any(p in component_lower for p in experimental_patterns):
            return SystemCriticality.EXPERIMENTAL
        else:
            return SystemCriticality.MEDIUM

    def _estimate_complexity(self, deployment_context: Dict[str, Any]) -> FixComplexity:
        """Estimate fix complexity from deployment context."""
        num_files = len(deployment_context.get("affected_files", []))
        patch_size = len(deployment_context.get("patch_content", ""))

        if num_files == 1 and patch_size < 100:
            return FixComplexity.TRIVIAL
        elif num_files == 1 and patch_size < 500:
            return FixComplexity.SIMPLE
        elif num_files <= 3 and patch_size < 1000:
            return FixComplexity.MODERATE
        elif num_files <= 10:
            return FixComplexity.COMPLEX
        else:
            return FixComplexity.VERY_COMPLEX

    def _get_historical_success_rate(self, error_type: str) -> Optional[float]:
        """Get historical success rate for error type."""
        # This would query historical data
        # For now, return a simulated value
        return 0.75

    def _count_recent_failures(self, error_type: str, hours: int) -> int:
        """Count recent failures for error type."""
        # This would query recent deployment history
        # For now, return a simulated value
        return 0

    def _get_time_context(self) -> str:
        """Determine current time context."""
        from datetime import datetime

        now = datetime.now()

        if now.weekday() >= 5:  # Saturday or Sunday
            return "weekend"
        elif 9 <= now.hour < 17:  # Business hours
            return "business_hours"
        else:
            return "after_hours"


class ReviewTrigger:
    """Determines when human review is required."""

    def __init__(
        self,
        confidence_calculator: ConfidenceCalculator,
        contextual_thresholds: ContextualThresholds,
    ):
        self.confidence_calculator = confidence_calculator
        self.contextual_thresholds = contextual_thresholds
        self.review_history: deque = deque(maxlen=1000)
        self.bypass_conditions: List[Dict[str, Any]] = []

    def should_trigger_review(
        self,
        error_data: Dict[str, Any],
        fix_data: Dict[str, Any],
        model_confidence: float,
    ) -> Tuple[bool, List[str]]:
        """Determine if human review should be triggered."""
        reasons = []

        # Check bypass conditions first
        if self._check_bypass_conditions(error_data, fix_data):
            return False, ["Bypass condition met"]

        # Get context
        context = self._build_context(error_data, fix_data)

        # Calculate adjusted confidence
        confidence_result = self.confidence_calculator.calculate_confidence(
            model_confidence, context
        )

        # Primary review trigger: confidence below threshold
        if confidence_result["requires_review"]:
            reasons.append(
                f"Confidence ({confidence_result['adjusted_confidence']:.2f}) "
                f"below threshold ({confidence_result['threshold']:.2f})"
            )

        # Additional triggers
        additional_reasons = self._check_additional_triggers(
            error_data, fix_data, context, confidence_result
        )
        reasons.extend(additional_reasons)

        # Record decision
        self._record_review_decision(
            error_data.get("error_id", ""), bool(reasons), reasons, confidence_result
        )

        return bool(reasons), reasons

    def add_bypass_condition(self, condition: Dict[str, Any]) -> None:
        """Add a condition that bypasses review requirements."""
        self.bypass_conditions.append(condition)

    def get_review_statistics(self) -> Dict[str, Any]:
        """Get statistics on review triggers."""
        if not self.review_history:
            return {"total_decisions": 0}

        total = len(self.review_history)
        triggered = sum(1 for r in self.review_history if r["triggered"])

        # Analyze reasons
        reason_counts: Dict[str, int] = defaultdict(int)
        for record in self.review_history:
            for reason in record.get("reasons", []):
                # Extract reason type
                if "confidence" in reason.lower():
                    reason_counts["low_confidence"] += 1
                elif "critical" in reason.lower():
                    reason_counts["critical_system"] += 1
                elif "complex" in reason.lower():
                    reason_counts["high_complexity"] += 1
                else:
                    reason_counts["other"] += 1

        return {
            "total_decisions": total,
            "reviews_triggered": triggered,
            "trigger_rate": triggered / total if total > 0 else 0,
            "reason_distribution": dict(reason_counts),
            "avg_confidence_when_triggered": (
                np.mean(
                    [
                        r["confidence_result"]["adjusted_confidence"]
                        for r in self.review_history
                        if r["triggered"] and "confidence_result" in r
                    ]
                )
                if any(r["triggered"] for r in self.review_history)
                else 0
            ),
        }

    def _build_context(
        self, error_data: Dict[str, Any], fix_data: Dict[str, Any]
    ) -> ConfidenceContext:
        """Build context from error and fix data."""
        # Determine affected components
        affected_files = fix_data.get("affected_files", [])
        components = [self._extract_component(f) for f in affected_files]

        # Determine system criticality
        criticality = SystemCriticality.MEDIUM
        for component in components:
            comp_criticality = self.contextual_thresholds._determine_criticality(
                component
            )
            if comp_criticality.value < criticality.value:
                criticality = comp_criticality

        # Estimate complexity
        complexity = self.contextual_thresholds._estimate_complexity(fix_data)

        return ConfidenceContext(
            error_type=error_data.get("error_type", "unknown"),
            error_severity=error_data.get("severity", "medium"),
            affected_components=components,
            fix_complexity=complexity,
            system_criticality=criticality,
            historical_success_rate=self.contextual_thresholds._get_historical_success_rate(
                error_data.get("error_type", "unknown")
            ),
            recent_failures=self.contextual_thresholds._count_recent_failures(
                error_data.get("error_type", "unknown"), hours=24
            ),
            time_of_day=self.contextual_thresholds._get_time_context(),
        )

    def _check_bypass_conditions(
        self, error_data: Dict[str, Any], fix_data: Dict[str, Any]
    ) -> bool:
        """Check if any bypass conditions are met."""
        for condition in self.bypass_conditions:
            if self._evaluate_condition(condition, error_data, fix_data):
                return True
        return False

    def _evaluate_condition(
        self,
        condition: Dict[str, Any],
        error_data: Dict[str, Any],
        fix_data: Dict[str, Any],
    ) -> bool:
        """Evaluate a single bypass condition."""
        # Simple condition evaluation
        # In practice, this would be more sophisticated
        if "error_type" in condition:
            if error_data.get("error_type") != condition["error_type"]:
                return False

        if "max_files" in condition:
            if len(fix_data.get("affected_files", [])) > condition["max_files"]:
                return False

        if "min_confidence" in condition:
            if fix_data.get("confidence", 0) < condition["min_confidence"]:
                return False

        return True

    def _check_additional_triggers(
        self,
        error_data: Dict[str, Any],
        fix_data: Dict[str, Any],
        context: ConfidenceContext,
        confidence_result: Dict[str, Any],
    ) -> List[str]:
        """Check for additional review triggers beyond confidence."""
        reasons = []

        # Trigger for critical systems regardless of confidence
        if context.system_criticality == SystemCriticality.CRITICAL:
            if confidence_result["adjusted_confidence"] < 0.95:
                reasons.append("Critical system component requires review")

        # Trigger for very complex fixes
        if context.fix_complexity == FixComplexity.VERY_COMPLEX:
            reasons.append("Very complex fix requires review")

        # Trigger for high recent failure rate
        if context.recent_failures > 3:
            reasons.append(f"High recent failure count ({context.recent_failures})")

        # Trigger for specific error types that always need review
        always_review_errors = [
            "security_vulnerability",
            "data_corruption",
            "authentication_failure",
        ]
        if error_data.get("error_type") in always_review_errors:
            reasons.append(
                f"Error type '{error_data['error_type']}' always requires review"
            )

        # Trigger for fixes affecting multiple critical files
        critical_file_patterns = ["config", "security", "auth", "database"]
        critical_files = [
            f
            for f in fix_data.get("affected_files", [])
            if any(p in f.lower() for p in critical_file_patterns)
        ]
        if len(critical_files) > 1:
            reasons.append(f"Multiple critical files affected ({len(critical_files)})")

        return reasons

    def _extract_component(self, file_path: str) -> str:
        """Extract component name from file path."""
        parts = Path(file_path).parts

        # Try to identify the component from the path
        if "src" in parts:
            idx = parts.index("src")
            if idx + 1 < len(parts):
                return parts[idx + 1]

        # Fallback to parent directory
        return Path(file_path).parent.name

    def _record_review_decision(
        self,
        error_id: str,
        triggered: bool,
        reasons: List[str],
        confidence_result: Dict[str, Any],
    ) -> None:
        """Record review decision for analysis."""
        record = {
            "error_id": error_id,
            "triggered": triggered,
            "reasons": reasons,
            "confidence_result": confidence_result,
            "timestamp": time.time(),
        }

        self.review_history.append(record)
