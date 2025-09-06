"""ML Feedback Loops for Model Improvement.

This module implements feedback collection and automated retraining systems
that allow ML models to improve based on real-world performance.
"""

import json
import logging
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)

from ..analysis.healing_metrics import HealingMetricsCollector as HealingMetrics
from ..analysis.models.data_collector import ErrorDataCollector
from ..analysis.models.serving import ModelServer
from ..analysis.models.trainer import ModelTrainer, TrainingConfig

logger = logging.getLogger(__name__)


@dataclass
class PredictionFeedback:
    """Represents feedback for a single prediction."""

    prediction_id: str
    model_name: str
    model_version: str
    input_data: Dict[str, Any]
    prediction: Any
    actual_outcome: Optional[Any] = None
    confidence: Optional[float] = None
    timestamp: float = None
    context: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PredictionFeedback":
        """Create from dictionary."""
        return cls(**data)


class MLFeedbackLoop:
    """Main feedback loop for ML model improvement."""

    def __init__(
        self,
        feedback_dir: Path = Path("data/ml_feedback"),
        min_feedback_for_retrain: int = 1000,
        retrain_interval_hours: int = 24,
        performance_threshold: float = 0.85,
    ):
        self.feedback_dir = Path(feedback_dir)
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
        self.min_feedback_for_retrain = min_feedback_for_retrain
        self.retrain_interval = timedelta(hours=retrain_interval_hours)
        self.performance_threshold = performance_threshold

        self.feedback_buffer = defaultdict(list)
        self.last_retrain = defaultdict(lambda: datetime.min)
        self.performance_tracker = ModelPerformanceTracker()
        self.data_collector = ErrorDataCollector()

        # Load existing feedback
        self._load_feedback()

    def add_feedback(self, feedback: PredictionFeedback) -> None:
        """Add prediction feedback to the system."""
        logger.info(
            f"Adding feedback for model {feedback.model_name} v{feedback.model_version}"
        )

        # Add to buffer
        model_key = f"{feedback.model_name}:{feedback.model_version}"
        self.feedback_buffer[model_key].append(feedback)

        # Persist feedback
        self._save_feedback(feedback)

        # Update performance metrics
        if feedback.actual_outcome is not None:
            self.performance_tracker.update(
                model_name=feedback.model_name,
                model_version=feedback.model_version,
                prediction=feedback.prediction,
                actual=feedback.actual_outcome,
                confidence=feedback.confidence,
            )

        # Check if retraining is needed
        self._check_retrain_trigger(model_key)

    def add_batch_feedback(self, feedback_list: List[PredictionFeedback]) -> None:
        """Add multiple feedback items at once."""
        for feedback in feedback_list:
            self.add_feedback(feedback)

    def get_model_performance(
        self, model_name: str, model_version: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get performance metrics for a model."""
        return self.performance_tracker.get_metrics(model_name, model_version)

    def _check_retrain_trigger(self, model_key: str) -> None:
        """Check if model should be retrained based on feedback."""
        feedback_count = len(self.feedback_buffer[model_key])
        time_since_retrain = datetime.now() - self.last_retrain[model_key]

        model_name, model_version = model_key.split(":")
        performance = self.performance_tracker.get_metrics(model_name, model_version)

        should_retrain = False
        reason = ""

        # Check feedback count threshold
        if feedback_count >= self.min_feedback_for_retrain:
            should_retrain = True
            reason = f"Feedback count ({feedback_count}) exceeded threshold"

        # Check time interval
        elif time_since_retrain >= self.retrain_interval:
            should_retrain = True
            reason = f"Time interval ({time_since_retrain}) exceeded"

        # Check performance degradation
        elif (
            performance
            and performance.get("accuracy", 1.0) < self.performance_threshold
        ):
            should_retrain = True
            reason = f"Performance ({performance['accuracy']:.3f}) below threshold"

        if should_retrain:
            logger.info(f"Triggering retrain for {model_key}: {reason}")
            self._trigger_retrain(model_key)

    def _trigger_retrain(self, model_key: str) -> None:
        """Trigger model retraining with collected feedback."""
        model_name, model_version = model_key.split(":")
        feedback_data = self.feedback_buffer[model_key]

        # Convert feedback to training data
        training_data = self._prepare_training_data(feedback_data)

        # Add to data collector
        for item in training_data:
            # Add error data with label in metadata
            error_data = item["input_data"].copy()
            error_data["label"] = item["actual_outcome"]
            error_data["metadata"] = {
                "source": "feedback_loop",
                "original_prediction": item["prediction"],
                "model_version": model_version,
                "actual_outcome": item["actual_outcome"],
            }
            self.data_collector.add_error(error_data=error_data, source="feedback_loop")

        # Clear feedback buffer for this model
        self.feedback_buffer[model_key] = []
        self.last_retrain[model_key] = datetime.now()

        # Notify retrainer
        retrainer = AutomatedRetrainer()
        retrainer.schedule_retrain(model_name, priority="high")

    def _prepare_training_data(
        self, feedback_list: List[PredictionFeedback]
    ) -> List[Dict[str, Any]]:
        """Convert feedback to training data format."""
        training_data = []

        for feedback in feedback_list:
            if feedback.actual_outcome is not None:
                training_data.append(
                    {
                        "input_data": feedback.input_data,
                        "prediction": feedback.prediction,
                        "actual_outcome": feedback.actual_outcome,
                        "confidence": feedback.confidence,
                        "context": feedback.context,
                    }
                )

        return training_data

    def _save_feedback(self, feedback: PredictionFeedback) -> None:
        """Save feedback to disk."""
        date_str = datetime.fromtimestamp(feedback.timestamp).strftime("%Y-%m-%d")
        feedback_file = self.feedback_dir / f"{feedback.model_name}_{date_str}.jsonl"

        with open(feedback_file, "a") as f:
            f.write(json.dumps(feedback.to_dict()) + "\n")

    def _load_feedback(self) -> None:
        """Load existing feedback from disk."""
        for feedback_file in self.feedback_dir.glob("*.jsonl"):
            with open(feedback_file, "r") as f:
                for line in f:
                    if line.strip():
                        feedback = PredictionFeedback.from_dict(json.loads(line))
                        model_key = f"{feedback.model_name}:{feedback.model_version}"
                        self.feedback_buffer[model_key].append(feedback)


class ModelPerformanceTracker:
    """Track model performance metrics over time."""

    def __init__(self):
        self.metrics = defaultdict(lambda: defaultdict(list))
        self.prediction_pairs = defaultdict(list)

    def update(
        self,
        model_name: str,
        model_version: str,
        prediction: Any,
        actual: Any,
        confidence: Optional[float] = None,
    ) -> None:
        """Update performance metrics with new prediction-outcome pair."""
        key = f"{model_name}:{model_version}"
        self.prediction_pairs[key].append(
            {
                "prediction": prediction,
                "actual": actual,
                "confidence": confidence,
                "timestamp": time.time(),
            }
        )

        # Calculate metrics periodically
        if len(self.prediction_pairs[key]) % 100 == 0:
            self._calculate_metrics(key)

    def get_metrics(
        self, model_name: str, model_version: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get current performance metrics for a model."""
        if model_version:
            key = f"{model_name}:{model_version}"
            self._calculate_metrics(key)
            return self.metrics[key]
        else:
            # Return metrics for all versions
            all_metrics = {}
            for key in self.metrics:
                if key.startswith(f"{model_name}:"):
                    version = key.split(":")[1]
                    all_metrics[version] = self.metrics[key]
            return all_metrics

    def _calculate_metrics(self, model_key: str) -> None:
        """Calculate performance metrics from prediction pairs."""
        pairs = self.prediction_pairs[model_key]
        if not pairs:
            return

        predictions = [p["prediction"] for p in pairs]
        actuals = [p["actual"] for p in pairs]

        # Determine if classification or regression
        is_classification = all(isinstance(p, (int, str, bool)) for p in predictions)

        metrics = {"sample_count": len(pairs), "timestamp": time.time()}

        if is_classification:
            # Classification metrics
            metrics["accuracy"] = accuracy_score(actuals, predictions)
            metrics["precision"] = precision_score(
                actuals, predictions, average="weighted", zero_division=0
            )
            metrics["recall"] = recall_score(
                actuals, predictions, average="weighted", zero_division=0
            )
            metrics["f1"] = f1_score(
                actuals, predictions, average="weighted", zero_division=0
            )
        else:
            # Regression metrics
            metrics["mse"] = mean_squared_error(actuals, predictions)
            metrics["mae"] = mean_absolute_error(actuals, predictions)
            metrics["rmse"] = np.sqrt(metrics["mse"])

        # Confidence analysis
        confidences = [p["confidence"] for p in pairs if p["confidence"] is not None]
        if confidences:
            metrics["avg_confidence"] = np.mean(confidences)
            metrics["confidence_std"] = np.std(confidences)

        self.metrics[model_key] = metrics


class AutomatedRetrainer:
    """Automated model retraining system."""

    def __init__(
        self,
        training_config_dir: Path = Path("configs/training"),
        model_registry_dir: Path = Path("models/registry"),
    ):
        self.training_config_dir = Path(training_config_dir)
        self.model_registry_dir = Path(model_registry_dir)
        self.retrain_queue = []
        self.active_retrains = {}

    def schedule_retrain(
        self,
        model_name: str,
        priority: str = "normal",
        custom_config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Schedule a model for retraining."""
        retrain_id = f"{model_name}_{int(time.time())}"

        retrain_task = {
            "id": retrain_id,
            "model_name": model_name,
            "priority": priority,
            "custom_config": custom_config,
            "scheduled_at": time.time(),
            "status": "queued",
        }

        # Add to queue based on priority
        if priority == "high":
            self.retrain_queue.insert(0, retrain_task)
        else:
            self.retrain_queue.append(retrain_task)

        logger.info(
            f"Scheduled retrain {retrain_id} for {model_name} with {priority} priority"
        )
        return retrain_id

    def process_retrain_queue(self) -> None:
        """Process queued retraining tasks."""
        while self.retrain_queue:
            task = self.retrain_queue.pop(0)
            self._execute_retrain(task)

    def _execute_retrain(self, task: Dict[str, Any]) -> None:
        """Execute a single retrain task."""
        logger.info(f"Starting retrain {task['id']} for {task['model_name']}")

        try:
            # Load training config
            config = self._load_training_config(task["model_name"])
            if task["custom_config"]:
                config.update(task["custom_config"])

            # Initialize trainer
            trainer = ModelTrainer(TrainingConfig(**config))

            # Get latest training data
            data_collector = ErrorDataCollector()
            dataset = data_collector.export_dataset(
                output_path=None,  # Return in memory
                filters={"model_name": task["model_name"]},
            )

            # Train model
            trained_model = trainer.train(
                X=dataset["features"],
                y=dataset["labels"],
                model_name=task["model_name"],
            )

            # Update model registry
            version = f"v{int(time.time())}"
            trainer.register_model(
                trained_model,
                model_name=task["model_name"],
                version=version,
                metadata={
                    "retrain_id": task["id"],
                    "trigger": "automated_feedback_loop",
                    "training_samples": len(dataset["features"]),
                },
            )

            logger.info(
                f"Completed retrain {task['id']}: {task['model_name']} {version}"
            )

        except Exception as e:
            logger.error(f"Failed to retrain {task['id']}: {str(e)}")
            raise

    def _load_training_config(self, model_name: str) -> Dict[str, Any]:
        """Load training configuration for a model."""
        config_file = self.training_config_dir / f"{model_name}_config.json"

        if config_file.exists():
            with open(config_file, "r") as f:
                return json.load(f)
        else:
            # Return default config
            return {
                "model_type": "classifier",
                "algorithm": "random_forest",
                "hyperparameter_search": "grid",
                "cross_validation": 5,
                "metrics": ["accuracy", "precision", "recall", "f1"],
            }


def integrate_with_existing_systems() -> None:
    """Integration function to connect feedback loops with existing Homeostasis components."""

    # Initialize feedback loop
    ml_feedback_loop = MLFeedbackLoop()

    # Extend ModelServer to collect feedback
    original_predict = ModelServer.predict

    def predict_with_feedback(
        self, data: Dict[str, Any], model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Enhanced predict method that collects feedback."""
        result = original_predict(self, data, model_name)

        # Create feedback entry
        feedback = PredictionFeedback(
            prediction_id=result.get("request_id", str(time.time())),
            model_name=model_name or self.default_model,
            model_version=result.get("model_version", "unknown"),
            input_data=data,
            prediction=result["prediction"],
            confidence=result.get("confidence"),
        )

        # Store for later outcome collection
        self._pending_feedback[feedback.prediction_id] = feedback

        return result

    ModelServer.predict = predict_with_feedback

    # Add outcome collection endpoint
    def add_prediction_outcome(self, prediction_id: str, actual_outcome: Any) -> None:
        """Add actual outcome for a prediction."""
        if prediction_id in self._pending_feedback:
            feedback = self._pending_feedback[prediction_id]
            feedback.actual_outcome = actual_outcome
            ml_feedback_loop.add_feedback(feedback)
            del self._pending_feedback[prediction_id]

    ModelServer.add_prediction_outcome = add_prediction_outcome
    ModelServer._pending_feedback = {}

    # Connect with healing metrics
    original_track_fix = HealingMetrics.track_fix_outcome

    def track_fix_with_ml_feedback(
        self, error_id: str, success: bool, fix_data: Dict[str, Any]
    ) -> None:
        """Enhanced fix tracking that includes ML feedback."""
        original_track_fix(self, error_id, success, fix_data)

        # If ML was involved in the fix
        if "ml_prediction" in fix_data:
            prediction_id = fix_data.get("ml_prediction_id")
            if prediction_id:
                # Add outcome to ML feedback
                ModelServer.add_prediction_outcome(
                    prediction_id=prediction_id,
                    actual_outcome="success" if success else "failure",
                )

    HealingMetrics.track_fix_outcome = track_fix_with_ml_feedback

    logger.info("ML feedback loops integrated with existing systems")
