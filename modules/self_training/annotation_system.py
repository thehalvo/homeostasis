"""Human-in-the-Loop Annotation System.

This module provides interfaces for developers to review and annotate
healing actions, creating high-quality training data for model improvement.
"""

import json
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..analysis.models.data_collector import ErrorDataCollector
from ..llm_integration.patch_generator import PatchData
from .feedback_loops import MLFeedbackLoop, PredictionFeedback

logger = logging.getLogger(__name__)


class AnnotationStatus(Enum):
    """Status of an annotation task."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    EXPIRED = "expired"


class AnnotationType(Enum):
    """Types of annotations."""

    ERROR_CLASSIFICATION = "error_classification"
    PATCH_REVIEW = "patch_review"
    ROOT_CAUSE_VALIDATION = "root_cause_validation"
    CONFIDENCE_CALIBRATION = "confidence_calibration"
    EDGE_CASE_LABELING = "edge_case_labeling"


@dataclass
class AnnotationTask:
    """Represents a single annotation task."""

    task_id: str
    task_type: AnnotationType
    data: Dict[str, Any]
    model_prediction: Optional[Any] = None
    model_confidence: Optional[float] = None
    created_at: float = None
    expires_at: Optional[float] = None
    status: AnnotationStatus = AnnotationStatus.PENDING
    assigned_to: Optional[str] = None
    priority: int = 5  # 1-10, higher is more important

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()
        if self.task_id is None:
            self.task_id = str(uuid.uuid4())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["task_type"] = self.task_type.value
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnnotationTask":
        """Create from dictionary."""
        data["task_type"] = AnnotationType(data["task_type"])
        data["status"] = AnnotationStatus(data["status"])
        return cls(**data)


@dataclass
class Annotation:
    """Represents a completed annotation."""

    annotation_id: str
    task_id: str
    annotator_id: str
    annotation_type: AnnotationType
    original_data: Dict[str, Any]
    model_prediction: Optional[Any]
    human_label: Any
    confidence: float  # Annotator's confidence in their label
    time_taken: float  # Seconds to complete
    notes: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: float = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()
        if self.annotation_id is None:
            self.annotation_id = str(uuid.uuid4())


class AnnotationInterface:
    """Main interface for human annotation tasks."""

    def __init__(
        self,
        storage_dir: Path = Path("data/annotations"),
        task_expiry_hours: int = 48,
        min_annotations_per_task: int = 1,
        agreement_threshold: float = 0.8,
    ):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.task_expiry_seconds = task_expiry_hours * 3600
        self.min_annotations_per_task = min_annotations_per_task
        self.agreement_threshold = agreement_threshold

        self.tasks = {}
        self.annotations = defaultdict(list)
        self.annotator_stats = defaultdict(dict)

        # Callbacks for different annotation types
        self.annotation_callbacks = {}

        # Load existing data
        self._load_data()

    def create_task(
        self,
        task_type: AnnotationType,
        data: Dict[str, Any],
        model_prediction: Optional[Any] = None,
        model_confidence: Optional[float] = None,
        priority: int = 5,
    ) -> str:
        """Create a new annotation task."""
        task = AnnotationTask(
            task_id=str(uuid.uuid4()),
            task_type=task_type,
            data=data,
            model_prediction=model_prediction,
            model_confidence=model_confidence,
            priority=priority,
            expires_at=time.time() + self.task_expiry_seconds,
        )

        self.tasks[task.task_id] = task
        self._save_task(task)

        logger.info(f"Created annotation task {task.task_id} of type {task_type.value}")
        return task.task_id

    def get_next_task(
        self, annotator_id: str, task_types: Optional[List[AnnotationType]] = None
    ) -> Optional[AnnotationTask]:
        """Get the next task for an annotator."""
        # Filter available tasks
        available_tasks = []
        current_time = time.time()

        for task in self.tasks.values():
            if (
                task.status == AnnotationStatus.PENDING
                and (task.expires_at is None or task.expires_at > current_time)
                and (task_types is None or task.task_type in task_types)
            ):
                available_tasks.append(task)

        if not available_tasks:
            return None

        # Sort by priority and age
        available_tasks.sort(key=lambda t: (-t.priority, t.created_at))

        # Assign task
        task = available_tasks[0]
        task.status = AnnotationStatus.IN_PROGRESS
        task.assigned_to = annotator_id
        self._save_task(task)

        return task

    def submit_annotation(
        self,
        task_id: str,
        annotator_id: str,
        human_label: Any,
        confidence: float,
        time_taken: float,
        notes: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Submit an annotation for a task."""
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")

        task = self.tasks[task_id]

        annotation = Annotation(
            annotation_id=str(uuid.uuid4()),
            task_id=task_id,
            annotator_id=annotator_id,
            annotation_type=task.task_type,
            original_data=task.data,
            model_prediction=task.model_prediction,
            human_label=human_label,
            confidence=confidence,
            time_taken=time_taken,
            notes=notes,
            metadata=metadata,
        )

        self.annotations[task_id].append(annotation)
        self._save_annotation(annotation)

        # Update annotator stats
        self._update_annotator_stats(annotator_id, annotation)

        # Check if task is complete
        if len(self.annotations[task_id]) >= self.min_annotations_per_task:
            task.status = AnnotationStatus.COMPLETED
            self._process_completed_task(task)
        else:
            task.status = AnnotationStatus.PENDING
            task.assigned_to = None

        self._save_task(task)

        logger.info(
            f"Received annotation {annotation.annotation_id} for task {task_id}"
        )
        return annotation.annotation_id

    def skip_task(self, task_id: str, annotator_id: str, reason: str) -> None:
        """Skip an annotation task."""
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")

        task = self.tasks[task_id]
        task.status = AnnotationStatus.PENDING
        task.assigned_to = None

        # Log skip reason
        skip_data = {
            "task_id": task_id,
            "annotator_id": annotator_id,
            "reason": reason,
            "timestamp": time.time(),
        }

        skip_file = self.storage_dir / "skipped_tasks.jsonl"
        with open(skip_file, "a") as f:
            f.write(json.dumps(skip_data) + "\n")

        self._save_task(task)

    def register_callback(
        self,
        annotation_type: AnnotationType,
        callback: Callable[[List[Annotation]], None],
    ) -> None:
        """Register a callback for completed annotations."""
        self.annotation_callbacks[annotation_type] = callback

    def get_annotator_stats(self, annotator_id: str) -> Dict[str, Any]:
        """Get statistics for an annotator."""
        return self.annotator_stats.get(
            annotator_id,
            {
                "total_annotations": 0,
                "avg_confidence": 0,
                "avg_time_taken": 0,
                "annotation_types": {},
                "quality_score": 0,
            },
        )

    def _process_completed_task(self, task: AnnotationTask) -> None:
        """Process a completed annotation task."""
        task_annotations = self.annotations[task.task_id]

        # Calculate agreement if multiple annotations
        if len(task_annotations) > 1:
            agreement = self._calculate_agreement(task_annotations)
            logger.info(f"Task {task.task_id} completed with {agreement:.2f} agreement")

        # Call registered callback
        if task.task_type in self.annotation_callbacks:
            self.annotation_callbacks[task.task_type](task_annotations)

        # Send to ML feedback loop
        self._send_to_ml_feedback(task, task_annotations)

    def _calculate_agreement(self, annotations: List[Annotation]) -> float:
        """Calculate inter-annotator agreement."""
        if len(annotations) < 2:
            return 1.0

        # Simple agreement: fraction of annotators with same label
        labels = [a.human_label for a in annotations]
        label_counts = defaultdict(int)
        for label in labels:
            label_counts[label] += 1

        max_agreement = max(label_counts.values()) / len(labels)
        return max_agreement

    def _send_to_ml_feedback(
        self, task: AnnotationTask, annotations: List[Annotation]
    ) -> None:
        """Send annotation data to ML feedback loop."""
        # Use majority vote or weighted average
        human_label = self._aggregate_annotations(annotations)

        if task.model_prediction is not None:
            feedback = PredictionFeedback(
                prediction_id=task.task_id,
                model_name=f"{task.task_type.value}_model",
                model_version="latest",
                input_data=task.data,
                prediction=task.model_prediction,
                actual_outcome=human_label,
                confidence=task.model_confidence,
                context={"annotation_count": len(annotations)},
            )

            ml_loop = MLFeedbackLoop()
            ml_loop.add_feedback(feedback)

    def _aggregate_annotations(self, annotations: List[Annotation]) -> Any:
        """Aggregate multiple annotations into a single label."""
        if not annotations:
            return None

        # For classification, use majority vote weighted by confidence
        label_scores = defaultdict(float)
        for annotation in annotations:
            label_scores[annotation.human_label] += annotation.confidence

        # Return label with highest score
        return max(label_scores.items(), key=lambda x: x[1])[0]

    def _update_annotator_stats(
        self, annotator_id: str, annotation: Annotation
    ) -> None:
        """Update statistics for an annotator."""
        stats = self.annotator_stats[annotator_id]

        # Initialize if needed
        if "total_annotations" not in stats:
            stats = {
                "total_annotations": 0,
                "total_confidence": 0,
                "total_time": 0,
                "annotation_types": defaultdict(int),
                "quality_scores": [],
            }

        # Update counts
        stats["total_annotations"] += 1
        stats["total_confidence"] += annotation.confidence
        stats["total_time"] += annotation.time_taken
        stats["annotation_types"][annotation.annotation_type.value] += 1

        # Calculate averages
        stats["avg_confidence"] = stats["total_confidence"] / stats["total_annotations"]
        stats["avg_time_taken"] = stats["total_time"] / stats["total_annotations"]

        self.annotator_stats[annotator_id] = stats

    def _save_task(self, task: AnnotationTask) -> None:
        """Save task to disk."""
        task_file = self.storage_dir / "tasks" / f"{task.task_id}.json"
        task_file.parent.mkdir(exist_ok=True)

        with open(task_file, "w") as f:
            json.dump(task.to_dict(), f, indent=2)

    def _save_annotation(self, annotation: Annotation) -> None:
        """Save annotation to disk."""
        date_str = datetime.fromtimestamp(annotation.created_at).strftime("%Y-%m-%d")
        annotation_file = self.storage_dir / "annotations" / f"{date_str}.jsonl"
        annotation_file.parent.mkdir(exist_ok=True)

        # Convert dataclass to dict and handle enum values
        annotation_dict = asdict(annotation)

        # Custom JSON encoder for enums
        def convert_enums(obj):
            if isinstance(obj, dict):
                return {k: convert_enums(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_enums(v) for v in obj]
            elif isinstance(obj, Enum):
                return obj.value
            else:
                return obj

        annotation_dict = convert_enums(annotation_dict)

        with open(annotation_file, "a") as f:
            f.write(json.dumps(annotation_dict) + "\n")

    def _load_data(self) -> None:
        """Load existing tasks and annotations from disk."""
        # Load tasks
        tasks_dir = self.storage_dir / "tasks"
        if tasks_dir.exists():
            for task_file in tasks_dir.glob("*.json"):
                with open(task_file, "r") as f:
                    task_data = json.load(f)
                    task = AnnotationTask.from_dict(task_data)
                    self.tasks[task.task_id] = task

        # Load annotations
        annotations_dir = self.storage_dir / "annotations"
        if annotations_dir.exists():
            for annotation_file in annotations_dir.glob("*.jsonl"):
                with open(annotation_file, "r") as f:
                    for line in f:
                        if line.strip():
                            ann_data = json.loads(line)
                            ann_data["annotation_type"] = AnnotationType(
                                ann_data["annotation_type"]
                            )
                            annotation = Annotation(**ann_data)
                            self.annotations[annotation.task_id].append(annotation)


class HumanFeedbackCollector:
    """Collects and processes human feedback on model predictions."""

    def __init__(self, annotation_interface: AnnotationInterface):
        self.annotation_interface = annotation_interface
        self.data_collector = ErrorDataCollector()

    def request_patch_review(
        self,
        error_data: Dict[str, Any],
        generated_patch: PatchData,
        model_confidence: float,
    ) -> str:
        """Request human review of a generated patch."""
        task_data = {
            "error": error_data,
            "patch": {
                "file_path": error_data.get("file_path", "unknown"),
                "original_code": generated_patch.original_code,
                "patched_code": generated_patch.patched_code,
                "patch_diff": generated_patch.get_diff(),
            },
            "context": {
                "error_type": error_data.get("error_type"),
                "stack_trace": error_data.get("stack_trace"),
                "affected_functions": error_data.get("affected_functions", []),
            },
        }

        return self.annotation_interface.create_task(
            task_type=AnnotationType.PATCH_REVIEW,
            data=task_data,
            model_prediction="approved",
            model_confidence=model_confidence,
            priority=self._calculate_priority(error_data, model_confidence),
        )

    def request_error_classification(
        self, error_data: Dict[str, Any], model_prediction: str, model_confidence: float
    ) -> str:
        """Request human classification of an error."""
        return self.annotation_interface.create_task(
            task_type=AnnotationType.ERROR_CLASSIFICATION,
            data=error_data,
            model_prediction=model_prediction,
            model_confidence=model_confidence,
            priority=self._calculate_priority(error_data, model_confidence),
        )

    def collect_edge_cases(
        self, min_confidence: float = 0.3, max_confidence: float = 0.7
    ) -> List[str]:
        """Identify and collect edge cases for annotation."""
        # Query recent predictions with medium confidence
        edge_cases: List[Any] = []

        # This would integrate with the model serving system
        # to identify predictions in the uncertainty range

        return edge_cases

    def _calculate_priority(self, error_data: Dict[str, Any], confidence: float) -> int:
        """Calculate annotation priority based on error and confidence."""
        priority = 5  # Base priority

        # Higher priority for lower confidence
        if confidence < 0.5:
            priority += 2
        elif confidence < 0.7:
            priority += 1

        # Higher priority for critical errors
        if error_data.get("severity") == "critical":
            priority += 2
        elif error_data.get("severity") == "high":
            priority += 1

        # Cap at 10
        return min(priority, 10)


class AnnotationQualityScorer:
    """Scores the quality of annotations for training data selection."""

    def __init__(self):
        self.scorer_models = {}

    def score_annotation(self, annotation: Annotation, task: AnnotationTask) -> float:
        """Score the quality of an annotation."""
        score = 1.0

        # Factor 1: Annotator confidence
        score *= annotation.confidence

        # Factor 2: Time taken (penalize too fast or too slow)
        ideal_time = self._get_ideal_time(task.task_type)
        time_factor = 1.0 - abs(annotation.time_taken - ideal_time) / (ideal_time * 2)
        score *= max(0.5, time_factor)

        # Factor 3: Agreement with other annotations
        task_annotations = self._get_task_annotations(task.task_id)
        if len(task_annotations) > 1:
            agreement = self._calculate_agreement_score(annotation, task_annotations)
            score *= 0.5 + 0.5 * agreement

        # Factor 4: Annotator historical quality
        annotator_quality = self._get_annotator_quality(annotation.annotator_id)
        score *= 0.7 + 0.3 * annotator_quality

        return min(1.0, score)

    def select_training_data(
        self,
        annotations: List[Annotation],
        min_quality: float = 0.7,
        max_samples: Optional[int] = None,
    ) -> List[Annotation]:
        """Select high-quality annotations for training."""
        # Score all annotations
        scored_annotations = [
            (ann, self.score_annotation(ann, self._get_task(ann.task_id)))
            for ann in annotations
        ]

        # Filter by minimum quality
        filtered = [
            (ann, score) for ann, score in scored_annotations if score >= min_quality
        ]

        # Sort by quality
        filtered.sort(key=lambda x: x[1], reverse=True)

        # Limit samples if requested
        if max_samples:
            filtered = filtered[:max_samples]

        return [ann for ann, _ in filtered]

    def _get_ideal_time(self, task_type: AnnotationType) -> float:
        """Get ideal time for task type in seconds."""
        ideal_times = {
            AnnotationType.ERROR_CLASSIFICATION: 30,
            AnnotationType.PATCH_REVIEW: 120,
            AnnotationType.ROOT_CAUSE_VALIDATION: 60,
            AnnotationType.CONFIDENCE_CALIBRATION: 20,
            AnnotationType.EDGE_CASE_LABELING: 45,
        }
        return ideal_times.get(task_type, 60)

    def _calculate_agreement_score(
        self, annotation: Annotation, other_annotations: List[Annotation]
    ) -> float:
        """Calculate how well this annotation agrees with others."""
        if not other_annotations:
            return 1.0

        agreements = []
        for other in other_annotations:
            if other.annotation_id != annotation.annotation_id:
                # Simple equality check for now
                agrees = annotation.human_label == other.human_label
                agreements.append(1.0 if agrees else 0.0)

        return sum(agreements) / len(agreements) if agreements else 1.0

    def _get_annotator_quality(self, annotator_id: str) -> float:
        """Get historical quality score for an annotator."""
        # This would look up historical performance
        # For now, return a default
        return 0.8

    def _get_task_annotations(self, task_id: str) -> List[Annotation]:
        """Get all annotations for a task."""
        # This would query the annotation storage
        # For now, return empty list
        return []

    def _get_task(self, task_id: str) -> AnnotationTask:
        """Get task by ID."""
        # This would query the task storage
        # For now, return a dummy task
        return AnnotationTask(
            task_id=task_id, task_type=AnnotationType.ERROR_CLASSIFICATION, data={}
        )
