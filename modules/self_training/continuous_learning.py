"""Continuous Learning from Deployment Results.

This module monitors the long-term success of deployed fixes and
feeds this information back into the training pipeline.
"""

import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Set

import numpy as np

from ..monitoring.health_checks import HealthChecker
from .feedback_loops import MLFeedbackLoop, PredictionFeedback
from .rule_extraction import RuleExtractor

logger = logging.getLogger(__name__)


class DeploymentStatus(Enum):
    """Status of a deployed fix."""

    ACTIVE = "active"
    ROLLED_BACK = "rolled_back"
    SUPERSEDED = "superseded"
    MONITORING = "monitoring"
    VALIDATED = "validated"


class FixOutcome(Enum):
    """Long-term outcome of a fix."""

    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    REGRESSION = "regression"
    UNKNOWN = "unknown"


@dataclass
class DeployedFix:
    """Represents a fix deployed to production."""

    fix_id: str
    deployment_time: float
    error_id: str
    error_type: str
    fix_type: str  # 'ml_generated', 'rule_based', 'human_approved'
    confidence: float
    affected_files: List[str]
    patch_content: str
    model_version: Optional[str] = None
    status: DeploymentStatus = DeploymentStatus.MONITORING

    # Monitoring data
    error_recurrence_count: int = 0
    performance_metrics: Optional[Dict[str, float]] = None
    stability_score: float = 1.0
    last_checked: Optional[float] = None

    def __post_init__(self):
        if self.performance_metrics is None:
            self.performance_metrics = {}
        if self.last_checked is None:
            self.last_checked = self.deployment_time


@dataclass
class OutcomeReport:
    """Report on the outcome of a deployed fix."""

    fix_id: str
    outcome: FixOutcome
    duration_hours: float
    error_recurrence_rate: float
    performance_impact: Dict[str, float]
    stability_trend: List[float]
    confidence_calibration: float  # How accurate was the confidence?
    lessons_learned: List[str]

    def to_training_feedback(self) -> Dict[str, Any]:
        """Convert outcome to training feedback format."""
        return {
            "fix_id": self.fix_id,
            "success": self.outcome in [FixOutcome.SUCCESS, FixOutcome.PARTIAL_SUCCESS],
            "outcome_score": self._calculate_outcome_score(),
            "duration": self.duration_hours,
            "metrics": {
                "error_recurrence": self.error_recurrence_rate,
                "performance_impact": self.performance_impact,
                "stability": (
                    np.mean(self.stability_trend) if self.stability_trend else 1.0
                ),
            },
        }

    def _calculate_outcome_score(self) -> float:
        """Calculate a numeric score for the outcome."""
        scores = {
            FixOutcome.SUCCESS: 1.0,
            FixOutcome.PARTIAL_SUCCESS: 0.7,
            FixOutcome.FAILURE: 0.0,
            FixOutcome.REGRESSION: -0.5,
            FixOutcome.UNKNOWN: 0.5,
        }
        return scores.get(self.outcome, 0.5)


class DeploymentMonitor:
    """Monitors deployed fixes and tracks their long-term outcomes."""

    def __init__(
        self,
        storage_dir: Path = Path("data/deployment_monitoring"),
        monitoring_duration_hours: int = 168,  # 1 week
        check_interval_hours: int = 1,
    ):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.monitoring_duration = timedelta(hours=monitoring_duration_hours)
        self.check_interval = timedelta(hours=check_interval_hours)

        self.active_deployments: Dict[str, DeployedFix] = {}
        self.completed_deployments: Dict[str, DeployedFix] = {}
        self.outcome_history: Deque[OutcomeReport] = deque(maxlen=1000)

        # Load existing deployments
        self._load_deployments()

    def track_deployment(
        self,
        fix_id: str,
        error_id: str,
        error_type: str,
        fix_type: str,
        confidence: float,
        affected_files: List[str],
        patch_content: str,
        model_version: Optional[str] = None,
    ) -> None:
        """Start tracking a newly deployed fix."""
        deployment = DeployedFix(
            fix_id=fix_id,
            deployment_time=time.time(),
            error_id=error_id,
            error_type=error_type,
            fix_type=fix_type,
            confidence=confidence,
            affected_files=affected_files,
            patch_content=patch_content,
            model_version=model_version,
        )

        self.active_deployments[fix_id] = deployment
        self._save_deployment(deployment)

        logger.info(
            f"Started monitoring deployment {fix_id} for error type {error_type}"
        )

    def check_deployments(self, health_checker: HealthChecker) -> List[OutcomeReport]:
        """Check all active deployments and generate outcome reports."""
        current_time = time.time()
        completed_reports = []

        for fix_id, deployment in list(self.active_deployments.items()):
            # Skip if recently checked
            if (
                deployment.last_checked is not None
                and current_time - deployment.last_checked
                < self.check_interval.total_seconds()
            ):
                continue

            # Check deployment status
            outcome = self._check_deployment_outcome(deployment, health_checker)
            deployment.last_checked = current_time

            # Check if monitoring period is complete
            deployment_duration = current_time - deployment.deployment_time
            if deployment_duration >= self.monitoring_duration.total_seconds():
                # Generate final report
                report = self._generate_outcome_report(deployment, outcome)
                completed_reports.append(report)

                # Move to completed
                deployment.status = DeploymentStatus.VALIDATED
                self.completed_deployments[fix_id] = deployment
                del self.active_deployments[fix_id]

                # Save outcome
                self._save_outcome(report)
            else:
                # Update deployment status
                self._save_deployment(deployment)

        return completed_reports

    def get_deployment_stats(self) -> Dict[str, Any]:
        """Get statistics on deployment outcomes."""
        outcome_distribution: defaultdict[str, int] = defaultdict(int)
        fix_type_success_rates: defaultdict[str, Dict[str, int]] = defaultdict(lambda: {"success": 0, "total": 0})

        stats = {
            "active_deployments": len(self.active_deployments),
            "completed_deployments": len(self.completed_deployments),
            "success_rate": 0.0,
            "avg_monitoring_duration": 0.0,
            "outcome_distribution": outcome_distribution,
            "fix_type_success_rates": fix_type_success_rates,
        }

        if self.outcome_history:
            successes = sum(
                1 for r in self.outcome_history if r.outcome == FixOutcome.SUCCESS
            )
            stats["success_rate"] = successes / len(self.outcome_history)

            total_duration = sum(r.duration_hours for r in self.outcome_history)
            stats["avg_monitoring_duration"] = total_duration / len(
                self.outcome_history
            )

            for report in self.outcome_history:
                outcome_distribution[report.outcome.value] += 1

        # Calculate success rates by fix type
        for deployment in self.completed_deployments.values():
            fix_type_stats = fix_type_success_rates[deployment.fix_type]
            fix_type_stats["total"] += 1

            # Find corresponding outcome
            outcome = self._find_outcome_for_deployment(deployment.fix_id)
            if outcome and outcome.outcome == FixOutcome.SUCCESS:
                fix_type_stats["success"] += 1

        # Convert to success rates
        for fix_type, counts in stats["fix_type_success_rates"].items():
            if counts["total"] > 0:
                counts["success_rate"] = counts["success"] / counts["total"]

        return dict(stats)

    def _check_deployment_outcome(
        self, deployment: DeployedFix, health_checker: HealthChecker
    ) -> FixOutcome:
        """Check the current outcome of a deployment."""
        # Check for error recurrence
        error_recurrence = self._check_error_recurrence(deployment)
        deployment.error_recurrence_count = error_recurrence

        # Check system health
        health_metrics = health_checker.check_health()

        # Check performance impact
        perf_impact = self._calculate_performance_impact(deployment, health_metrics)
        deployment.performance_metrics = perf_impact

        # Update stability score
        deployment.stability_score = self._calculate_stability_score(
            error_recurrence, perf_impact, health_metrics
        )

        # Determine outcome
        if error_recurrence > 5:
            return FixOutcome.FAILURE
        elif deployment.stability_score < 0.5:
            return FixOutcome.REGRESSION
        elif error_recurrence > 0:
            return FixOutcome.PARTIAL_SUCCESS
        elif deployment.stability_score > 0.8:
            return FixOutcome.SUCCESS
        else:
            return FixOutcome.UNKNOWN

    def _check_error_recurrence(self, deployment: DeployedFix) -> int:
        """Check if the same error has recurred."""
        # This would query the error monitoring system
        # For now, return a simulated value
        return 0

    def _calculate_performance_impact(
        self, deployment: DeployedFix, health_metrics: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate the performance impact of the fix."""
        # Compare current metrics with baseline
        # This is a simplified version
        return {
            "cpu_impact": health_metrics.get("cpu_usage", 0) - 50,  # Baseline 50%
            "memory_impact": health_metrics.get("memory_usage", 0) - 60,  # Baseline 60%
            "response_time_impact": health_metrics.get("avg_response_time", 100)
            - 100,  # Baseline 100ms
        }

    def _calculate_stability_score(
        self,
        error_recurrence: int,
        perf_impact: Dict[str, float],
        health_metrics: Dict[str, Any],
    ) -> float:
        """Calculate overall stability score."""
        score = 1.0

        # Penalize for error recurrence
        score -= error_recurrence * 0.1

        # Penalize for negative performance impact
        for metric, impact in perf_impact.items():
            if impact > 10:  # More than 10% degradation
                score -= 0.1

        # Consider overall health
        if health_metrics.get("status") != "healthy":
            score -= 0.2

        return max(0.0, min(1.0, score))

    def _generate_outcome_report(
        self, deployment: DeployedFix, final_outcome: FixOutcome
    ) -> OutcomeReport:
        """Generate a comprehensive outcome report."""
        duration_hours = (time.time() - deployment.deployment_time) / 3600

        # Calculate error recurrence rate
        recurrence_rate = deployment.error_recurrence_count / max(
            1, duration_hours / 24
        )

        # Extract lessons learned
        lessons = self._extract_lessons(deployment, final_outcome)

        # Calculate confidence calibration
        expected_success = deployment.confidence
        actual_success = 1.0 if final_outcome == FixOutcome.SUCCESS else 0.0
        confidence_calibration = 1.0 - abs(expected_success - actual_success)

        report = OutcomeReport(
            fix_id=deployment.fix_id,
            outcome=final_outcome,
            duration_hours=duration_hours,
            error_recurrence_rate=recurrence_rate,
            performance_impact=deployment.performance_metrics,
            stability_trend=[deployment.stability_score],  # Would have historical data
            confidence_calibration=confidence_calibration,
            lessons_learned=lessons,
        )

        self.outcome_history.append(report)
        return report

    def _extract_lessons(
        self, deployment: DeployedFix, outcome: FixOutcome
    ) -> List[str]:
        """Extract lessons learned from the deployment."""
        lessons = []

        if outcome == FixOutcome.FAILURE:
            lessons.append(
                f"Fix for {deployment.error_type} failed after {deployment.error_recurrence_count} recurrences"
            )
            if deployment.confidence > 0.8:
                lessons.append(
                    "High confidence prediction failed - need to recalibrate"
                )

        elif outcome == FixOutcome.REGRESSION:
            lessons.append("Fix caused performance regression")
            worst_metric = max(
                deployment.performance_metrics.items(), key=lambda x: abs(x[1])
            )
            lessons.append(f"Worst impact on {worst_metric[0]}: {worst_metric[1]:.1f}%")

        elif outcome == FixOutcome.SUCCESS:
            if deployment.fix_type == "ml_generated":
                lessons.append(
                    f"ML model {deployment.model_version} successful for {deployment.error_type}"
                )

        return lessons

    def _find_outcome_for_deployment(self, fix_id: str) -> Optional[OutcomeReport]:
        """Find the outcome report for a deployment."""
        for report in self.outcome_history:
            if report.fix_id == fix_id:
                return report
        return None

    def _save_deployment(self, deployment: DeployedFix) -> None:
        """Save deployment to disk."""
        deployment_file = self.storage_dir / "active" / f"{deployment.fix_id}.json"
        deployment_file.parent.mkdir(exist_ok=True)

        data = asdict(deployment)
        data["status"] = deployment.status.value

        with open(deployment_file, "w") as f:
            json.dump(data, f, indent=2)

    def _save_outcome(self, report: OutcomeReport) -> None:
        """Save outcome report to disk."""
        outcome_file = self.storage_dir / "outcomes" / f"{report.fix_id}.json"
        outcome_file.parent.mkdir(exist_ok=True)

        data = asdict(report)
        data["outcome"] = report.outcome.value

        with open(outcome_file, "w") as f:
            json.dump(data, f, indent=2)

    def _load_deployments(self) -> None:
        """Load existing deployments from disk."""
        # Load active deployments
        active_dir = self.storage_dir / "active"
        if active_dir.exists():
            for deployment_file in active_dir.glob("*.json"):
                with open(deployment_file, "r") as f:
                    data = json.load(f)
                    data["status"] = DeploymentStatus(data["status"])
                    deployment = DeployedFix(**data)
                    self.active_deployments[deployment.fix_id] = deployment

        # Load completed deployments and outcomes
        outcomes_dir = self.storage_dir / "outcomes"
        if outcomes_dir.exists():
            for outcome_file in outcomes_dir.glob("*.json"):
                with open(outcome_file, "r") as f:
                    data = json.load(f)
                    data["outcome"] = FixOutcome(data["outcome"])
                    report = OutcomeReport(**data)
                    self.outcome_history.append(report)


class OutcomeTracker:
    """Tracks and analyzes long-term outcomes of fixes."""

    def __init__(self, deployment_monitor: DeploymentMonitor):
        self.deployment_monitor = deployment_monitor
        self.outcome_patterns: Dict[str, List[Any]] = defaultdict(list)
        self.success_predictors: Dict[str, Any] = {}

    def analyze_outcome_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in fix outcomes."""
        patterns = {
            "error_type_success": defaultdict(lambda: {"success": 0, "total": 0}),
            "confidence_accuracy": self._analyze_confidence_accuracy(),
            "time_to_failure": self._analyze_time_to_failure(),
            "regression_indicators": self._identify_regression_indicators(),
        }

        # Analyze by error type
        for report in self.deployment_monitor.outcome_history:
            deployment = self._get_deployment_for_report(report)
            if deployment:
                error_stats = patterns["error_type_success"][deployment.error_type]
                error_stats["total"] += 1
                if report.outcome == FixOutcome.SUCCESS:
                    error_stats["success"] += 1

        # Calculate success rates
        for error_type, type_stats in patterns["error_type_success"].items():
            if type_stats["total"] > 0:
                type_stats["success_rate"] = type_stats["success"] / type_stats["total"]

        return patterns

    def predict_fix_success(
        self,
        error_type: str,
        fix_type: str,
        confidence: float,
        affected_files: List[str],
    ) -> float:
        """Predict the likelihood of fix success based on historical data."""
        # TODO: Use features for more sophisticated prediction model
        _ = {  # features
            "error_type": error_type,
            "fix_type": fix_type,
            "confidence": confidence,
            "num_files": len(affected_files),
            "file_types": self._extract_file_types(affected_files),
        }

        # Simple prediction based on historical success rates
        base_rate = 0.7  # Default success rate

        # Adjust based on error type history
        error_patterns = self.outcome_patterns.get(error_type, [])
        if error_patterns:
            error_success_rate = sum(1 for p in error_patterns if p["success"]) / len(
                error_patterns
            )
            base_rate = 0.3 * base_rate + 0.7 * error_success_rate

        # Adjust based on confidence calibration
        if confidence > 0.9 and fix_type == "ml_generated":
            base_rate *= 0.9  # High confidence ML fixes slightly less reliable

        # Adjust based on scope
        if len(affected_files) > 5:
            base_rate *= 0.8  # Large changes more risky

        return min(0.95, max(0.05, base_rate))

    def _analyze_confidence_accuracy(self) -> Dict[str, float]:
        """Analyze how accurate confidence scores are."""
        confidence_bins: Dict[float, Dict[str, List[float]]] = defaultdict(
            lambda: {"predicted": [], "actual": []}
        )

        for report in self.deployment_monitor.outcome_history:
            deployment = self._get_deployment_for_report(report)
            if deployment:
                bin_idx = int(deployment.confidence * 10) / 10  # Round to nearest 0.1
                confidence_bins[bin_idx]["predicted"].append(deployment.confidence)
                confidence_bins[bin_idx]["actual"].append(
                    1.0 if report.outcome == FixOutcome.SUCCESS else 0.0
                )

        accuracy = {}
        for bin_idx, data in confidence_bins.items():
            if data["actual"]:
                accuracy[f"{bin_idx:.1f}"] = {
                    "mean_predicted": np.mean(data["predicted"]),
                    "mean_actual": np.mean(data["actual"]),
                    "calibration_error": abs(
                        np.mean(data["predicted"]) - np.mean(data["actual"])
                    ),
                    "sample_size": len(data["actual"]),
                }

        return accuracy

    def _analyze_time_to_failure(self) -> Dict[str, float]:
        """Analyze how long fixes last before failing."""
        failure_times = []

        for report in self.deployment_monitor.outcome_history:
            if report.outcome in [FixOutcome.FAILURE, FixOutcome.REGRESSION]:
                failure_times.append(report.duration_hours)

        if failure_times:
            return {
                "mean_hours": float(np.mean(failure_times)),
                "median_hours": float(np.median(failure_times)),
                "min_hours": float(np.min(failure_times)),
                "max_hours": float(np.max(failure_times)),
                "std_hours": float(np.std(failure_times)),
            }

        return {"mean_hours": float("inf")}

    def _identify_regression_indicators(self) -> List[str]:
        """Identify common indicators of fixes that cause regressions."""
        indicators = []
        regression_reports = [
            r
            for r in self.deployment_monitor.outcome_history
            if r.outcome == FixOutcome.REGRESSION
        ]

        if len(regression_reports) >= 5:
            # Analyze common patterns
            file_types: Dict[str, int] = defaultdict(int)
            fix_sizes = []

            for report in regression_reports:
                deployment = self._get_deployment_for_report(report)
                if deployment:
                    for file in deployment.affected_files:
                        ext = Path(file).suffix
                        file_types[ext] += 1

                    fix_sizes.append(len(deployment.patch_content))

            # Identify indicators
            if file_types:
                most_common_type = max(file_types.items(), key=lambda x: x[1])
                indicators.append(f"Changes to {most_common_type[0]} files")

            if fix_sizes and np.mean(fix_sizes) > 1000:
                indicators.append("Large patches (>1000 characters)")

        return indicators

    def _get_deployment_for_report(
        self, report: OutcomeReport
    ) -> Optional[DeployedFix]:
        """Get deployment data for a report."""
        # Check completed deployments
        if report.fix_id in self.deployment_monitor.completed_deployments:
            return self.deployment_monitor.completed_deployments[report.fix_id]
        # Check active deployments
        if report.fix_id in self.deployment_monitor.active_deployments:
            return self.deployment_monitor.active_deployments[report.fix_id]
        return None

    def _extract_file_types(self, file_paths: List[str]) -> Set[str]:
        """Extract file types from paths."""
        return {Path(f).suffix for f in file_paths}


class LearningPipeline:
    """Orchestrates the continuous learning pipeline."""

    def __init__(
        self,
        deployment_monitor: DeploymentMonitor,
        outcome_tracker: OutcomeTracker,
        ml_feedback_loop: MLFeedbackLoop,
        rule_extractor: RuleExtractor,
    ):
        self.deployment_monitor = deployment_monitor
        self.outcome_tracker = outcome_tracker
        self.ml_feedback_loop = ml_feedback_loop
        self.rule_extractor = rule_extractor

        self.learning_cycles = 0
        self.last_learning_time = time.time()

    def run_learning_cycle(self, health_checker: HealthChecker) -> Dict[str, Any]:
        """Run a complete learning cycle."""
        logger.info(f"Starting learning cycle {self.learning_cycles + 1}")

        results = {
            "cycle": self.learning_cycles + 1,
            "timestamp": time.time(),
            "outcomes_processed": 0,
            "ml_feedback_sent": 0,
            "rules_extracted": 0,
            "insights": [],
        }

        # Check deployments and get outcomes
        outcome_reports = self.deployment_monitor.check_deployments(health_checker)
        results["outcomes_processed"] = len(outcome_reports)

        # Process each outcome
        for report in outcome_reports:
            # Send to ML feedback
            self._send_outcome_to_ml_feedback(report)
            results["ml_feedback_sent"] += 1

            # Extract rules if successful
            if report.outcome == FixOutcome.SUCCESS:
                self._extract_rules_from_success(report)
                results["rules_extracted"] += 1

        # Analyze patterns
        patterns = self.outcome_tracker.analyze_outcome_patterns()

        # Generate insights
        insights = self._generate_insights(patterns)
        results["insights"] = insights

        # Update learning metrics
        self.learning_cycles += 1
        self.last_learning_time = time.time()

        logger.info(
            f"Completed learning cycle {results['cycle']} with {results['outcomes_processed']} outcomes"
        )

        return results

    def _send_outcome_to_ml_feedback(self, report: OutcomeReport) -> None:
        """Send outcome data to ML feedback loop."""
        deployment = self._get_deployment_for_outcome(report)
        if not deployment:
            return

        # Create feedback based on outcome
        feedback = PredictionFeedback(
            prediction_id=f"fix_{report.fix_id}",
            model_name=f"{deployment.error_type}_fixer",
            model_version=deployment.model_version or "unknown",
            input_data={
                "error_type": deployment.error_type,
                "error_id": deployment.error_id,
                "affected_files": deployment.affected_files,
            },
            prediction=deployment.confidence,  # Predicted success probability
            actual_outcome=report.to_training_feedback()["outcome_score"],
            confidence=deployment.confidence,
            context={
                "fix_type": deployment.fix_type,
                "deployment_duration": report.duration_hours,
                "outcome": report.outcome.value,
            },
        )

        self.ml_feedback_loop.add_feedback(feedback)

    def _extract_rules_from_success(self, report: OutcomeReport) -> None:
        """Extract rules from successful fixes."""
        deployment = self._get_deployment_for_outcome(report)
        if not deployment:
            return

        # Create mock patch data for rule extraction
        # In real implementation, would retrieve actual patch data
        patch_data = type(
            "PatchData",
            (),
            {
                "file_path": (
                    deployment.affected_files[0] if deployment.affected_files else ""
                ),
                "get_diff": lambda: deployment.patch_content,
            },
        )()

        error_data = {
            "error_id": deployment.error_id,
            "error_type": deployment.error_type,
            "error_message": f"Error of type {deployment.error_type}",
        }

        fix_metadata = {
            "fix_id": deployment.fix_id,
            "success": True,
            "confidence": deployment.confidence,
            "outcome_score": report.to_training_feedback()["outcome_score"],
        }

        self.rule_extractor.analyze_successful_fix(error_data, patch_data, fix_metadata)

    def _generate_insights(self, patterns: Dict[str, Any]) -> List[str]:
        """Generate actionable insights from patterns."""
        insights = []

        # Confidence calibration insights
        conf_accuracy = patterns.get("confidence_accuracy", {})
        for conf_bin, data in conf_accuracy.items():
            if data["calibration_error"] > 0.2 and data["sample_size"] > 10:
                insights.append(
                    f"Model confidence around {conf_bin} is off by {data['calibration_error']:.1%} "
                    f"(predicted: {data['mean_predicted']:.1%}, actual: {data['mean_actual']:.1%})"
                )

        # Error type insights
        error_success = patterns.get("error_type_success", {})
        for error_type, error_stats in error_success.items():
            if (
                error_stats.get("total", 0) > 5
                and error_stats.get("success_rate", 0) < 0.5
            ):
                insights.append(
                    f"Low success rate ({error_stats['success_rate']:.1%}) for {error_type} errors - "
                    f"consider additional training or rule refinement"
                )

        # Regression indicators
        regression_indicators = patterns.get("regression_indicators", [])
        if regression_indicators:
            insights.append(
                f"Common regression indicators: {', '.join(regression_indicators[:3])}"
            )

        # Time to failure
        ttf = patterns.get("time_to_failure", {})
        if ttf.get("mean_hours", float("inf")) < 24:
            insights.append(
                f"Fixes failing quickly (mean: {ttf['mean_hours']:.1f} hours) - "
                f"consider longer monitoring periods"
            )

        return insights

    def _get_deployment_for_outcome(
        self, report: OutcomeReport
    ) -> Optional[DeployedFix]:
        """Get deployment data for an outcome report."""
        # Try completed deployments first
        if report.fix_id in self.deployment_monitor.completed_deployments:
            return self.deployment_monitor.completed_deployments[report.fix_id]
        # Then active deployments
        if report.fix_id in self.deployment_monitor.active_deployments:
            return self.deployment_monitor.active_deployments[report.fix_id]
        return None

    def get_learning_status(self) -> Dict[str, Any]:
        """Get the current status of the learning pipeline."""
        deployment_stats = self.deployment_monitor.get_deployment_stats()

        return {
            "learning_cycles_completed": self.learning_cycles,
            "last_cycle_time": datetime.fromtimestamp(
                self.last_learning_time
            ).isoformat(),
            "deployment_stats": deployment_stats,
            "ml_feedback_queue_size": len(self.ml_feedback_loop.feedback_buffer),
            "extracted_patterns": len(self.rule_extractor.patterns),
            "active_monitoring": len(self.deployment_monitor.active_deployments),
        }
