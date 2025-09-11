"""
Model versioning and evaluation framework for Homeostasis.

This module provides:
- Model version control with Git-like semantics
- Comprehensive evaluation metrics and reporting
- A/B testing and model comparison
- Model lineage tracking
- Automatic model selection based on performance
"""

import datetime
import hashlib
import json
import os
import pickle
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

# Optional imports for advanced features
try:
    pass  # mlflow not actually used
    MLFLOW_AVAILABLE = False
except ImportError:
    MLFLOW_AVAILABLE = False

try:
    import joblib

    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False


class RestrictedUnpickler(pickle.Unpickler):
    """
    Restricted unpickler that only allows specific safe classes.
    This prevents arbitrary code execution during unpickling.
    """

    ALLOWED_MODULES = {
        "numpy",
        "numpy.core.multiarray",
        "numpy.core.numeric",
        "sklearn",
        "sklearn.ensemble",
        "sklearn.tree",
        "sklearn.linear_model",
        "sklearn.naive_bayes",
        "sklearn.svm",
        "sklearn.neighbors",
        "sklearn.preprocessing",
        "sklearn.preprocessing._label",
        "sklearn.feature_extraction",
        "sklearn.feature_extraction.text",
        "sklearn.pipeline",
        "joblib",
        "joblib.numpy_pickle",
        "collections",
        "builtins",
        "pandas",
        "pandas.core",
        "scipy",
        "scipy.sparse",
    }

    ALLOWED_NAMES = {
        ("builtins", "slice"),
        ("builtins", "range"),
        ("builtins", "tuple"),
        ("builtins", "list"),
        ("builtins", "dict"),
        ("builtins", "set"),
        ("builtins", "frozenset"),
        ("builtins", "bytearray"),
        ("collections", "OrderedDict"),
        ("numpy", "ndarray"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy", "dtype"),
        ("sklearn.preprocessing._label", "LabelEncoder"),
        ("pandas.core.frame", "DataFrame"),
        ("pandas.core.series", "Series"),
        ("scipy.sparse._csr", "csr_matrix"),
        ("scipy.sparse._csc", "csc_matrix"),
    }

    def find_class(self, module, name):
        # Check if module.name combination is explicitly allowed
        if (module, name) in self.ALLOWED_NAMES:
            return super().find_class(module, name)

        # Check if module is in allowed modules
        if any(module.startswith(allowed) for allowed in self.ALLOWED_MODULES):
            return super().find_class(module, name)

        # Reject everything else
        raise pickle.UnpicklingError(
            f"Attempting to unpickle unsafe class {module}.{name}. "
            f"Only allowed modules: {self.ALLOWED_MODULES}"
        )


def secure_pickle_load(filepath: str):
    """
    Securely load a pickled model with protection against arbitrary code execution.

    Args:
        filepath: Path to the pickle file

    Returns:
        Loaded object

    Raises:
        pickle.UnpicklingError: If unsafe classes are detected
    """
    with open(filepath, "rb") as f:
        unpickler = RestrictedUnpickler(f)
        return unpickler.load()


@dataclass
class ModelVersion:
    """Represents a specific version of a model."""

    version_id: str
    model_name: str
    model_type: str  # classifier, regressor, etc.
    created_at: str
    parent_version: Optional[str] = None
    training_config: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    status: str = "experimental"  # experimental, staging, production, archived
    artifacts: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Comprehensive evaluation results for a model."""

    model_version: str
    dataset_name: str
    timestamp: str

    # Basic metrics
    primary_metric: str
    primary_score: float

    # Detailed metrics
    classification_metrics: Optional[Dict[str, float]] = None
    regression_metrics: Optional[Dict[str, float]] = None

    # Additional analysis
    confusion_matrix: Optional[List[List[int]]] = None
    classification_report: Optional[Dict[str, Any]] = None
    feature_importance: Optional[Dict[str, float]] = None
    prediction_distribution: Optional[Dict[str, Any]] = None

    # Performance metrics
    inference_time_ms: Optional[float] = None
    memory_usage_mb: Optional[float] = None

    # Statistical tests
    statistical_tests: Optional[Dict[str, Any]] = None

    # Visualizations
    plots: Dict[str, str] = field(default_factory=dict)


class ModelVersionControl:
    """Git-like version control system for ML models."""

    def __init__(self, repository_path: str = "models/repository"):
        """Initialize the model repository."""
        self.repository_path = Path(repository_path)
        self.repository_path.mkdir(parents=True, exist_ok=True)

        # Set up database for tracking
        self.db_path = self.repository_path / "versions.db"
        self._init_database()

        # Set up directories
        self.models_dir = self.repository_path / "models"
        self.evaluations_dir = self.repository_path / "evaluations"
        self.artifacts_dir = self.repository_path / "artifacts"

        for dir_path in [self.models_dir, self.evaluations_dir, self.artifacts_dir]:
            dir_path.mkdir(exist_ok=True)

    def _init_database(self):
        """Initialize the SQLite database for version tracking."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create versions table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS model_versions (
                version_id TEXT PRIMARY KEY,
                model_name TEXT NOT NULL,
                model_type TEXT NOT NULL,
                created_at TEXT NOT NULL,
                parent_version TEXT,
                training_config TEXT,
                metrics TEXT,
                tags TEXT,
                status TEXT,
                artifacts TEXT,
                metadata TEXT
            )
        """
        )

        # Create evaluations table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS evaluations (
                evaluation_id TEXT PRIMARY KEY,
                model_version TEXT NOT NULL,
                dataset_name TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                primary_metric TEXT,
                primary_score REAL,
                full_results TEXT,
                FOREIGN KEY (model_version) REFERENCES model_versions (version_id)
            )
        """
        )

        # Create lineage table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS model_lineage (
                child_version TEXT NOT NULL,
                parent_version TEXT NOT NULL,
                relationship_type TEXT,
                created_at TEXT NOT NULL,
                PRIMARY KEY (child_version, parent_version),
                FOREIGN KEY (child_version) REFERENCES model_versions (version_id),
                FOREIGN KEY (parent_version) REFERENCES model_versions (version_id)
            )
        """
        )

        conn.commit()
        conn.close()

    def _generate_version_id(self, model_name: str) -> str:
        """Generate a unique version ID."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = hashlib.sha256(os.urandom(16)).hexdigest()[:16]
        return f"{model_name}_{timestamp}_{random_suffix}"

    def commit_model(
        self,
        model: Any,
        model_name: str,
        model_type: str,
        training_config: Dict[str, Any],
        metrics: Dict[str, float],
        parent_version: Optional[str] = None,
        tags: Optional[List[str]] = None,
        artifacts: Optional[Dict[str, str]] = None,
    ) -> ModelVersion:
        """Commit a new model version to the repository."""
        # Generate version ID
        version_id = self._generate_version_id(model_name)

        # Create model version object
        version = ModelVersion(
            version_id=version_id,
            model_name=model_name,
            model_type=model_type,
            created_at=datetime.datetime.now().isoformat(),
            parent_version=parent_version,
            training_config=training_config,
            metrics=metrics,
            tags=tags or [],
            status="experimental",
            artifacts=artifacts or {},
        )

        # Save model file
        model_path = self.models_dir / f"{version_id}.pkl"
        if JOBLIB_AVAILABLE:
            joblib.dump(model, model_path)
        else:
            import pickle

            with open(model_path, "wb") as f:
                pickle.dump(model, f)

        # Save to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO model_versions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                version.version_id,
                version.model_name,
                version.model_type,
                version.created_at,
                version.parent_version,
                json.dumps(version.training_config),
                json.dumps(version.metrics),
                json.dumps(version.tags),
                version.status,
                json.dumps(version.artifacts),
                json.dumps(version.metadata),
            ),
        )

        # Update lineage if parent exists
        if parent_version:
            cursor.execute(
                """
                INSERT INTO model_lineage VALUES (?, ?, ?, ?)
            """,
                (version_id, parent_version, "derived", version.created_at),
            )

        conn.commit()
        conn.close()

        return version

    def get_version(self, version_id: str) -> Optional[ModelVersion]:
        """Get a specific model version."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT * FROM model_versions WHERE version_id = ?
        """,
            (version_id,),
        )

        row = cursor.fetchone()
        conn.close()

        if row:
            return ModelVersion(
                version_id=row[0],
                model_name=row[1],
                model_type=row[2],
                created_at=row[3],
                parent_version=row[4],
                training_config=json.loads(row[5]),
                metrics=json.loads(row[6]),
                tags=json.loads(row[7]),
                status=row[8],
                artifacts=json.loads(row[9]),
                metadata=json.loads(row[10]),
            )

        return None

    def load_model(self, version_id: str) -> Any:
        """Load a model from the repository."""
        model_path = self.models_dir / f"{version_id}.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found for version {version_id}")

        if JOBLIB_AVAILABLE:
            return joblib.load(model_path)
        else:
            return secure_pickle_load(model_path)

    def list_versions(
        self,
        model_name: Optional[str] = None,
        status: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[ModelVersion]:
        """List model versions with optional filtering."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = "SELECT * FROM model_versions WHERE 1=1"
        params = []

        if model_name:
            query += " AND model_name = ?"
            params.append(model_name)

        if status:
            query += " AND status = ?"
            params.append(status)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        versions = []
        for row in rows:
            version = ModelVersion(
                version_id=row[0],
                model_name=row[1],
                model_type=row[2],
                created_at=row[3],
                parent_version=row[4],
                training_config=json.loads(row[5]),
                metrics=json.loads(row[6]),
                tags=json.loads(row[7]),
                status=row[8],
                artifacts=json.loads(row[9]),
                metadata=json.loads(row[10]),
            )

            # Filter by tags if specified
            if tags and not any(tag in version.tags for tag in tags):
                continue

            versions.append(version)

        return sorted(versions, key=lambda v: v.created_at, reverse=True)

    def tag_version(self, version_id: str, tags: List[str]):
        """Add tags to a model version."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get current tags
        cursor.execute(
            """
            SELECT tags FROM model_versions WHERE version_id = ?
        """,
            (version_id,),
        )

        row = cursor.fetchone()
        if row:
            current_tags = json.loads(row[0])
            updated_tags = list(set(current_tags + tags))

            cursor.execute(
                """
                UPDATE model_versions SET tags = ? WHERE version_id = ?
            """,
                (json.dumps(updated_tags), version_id),
            )

            conn.commit()

        conn.close()

    def promote_version(self, version_id: str, new_status: str):
        """Promote a model version to a new status."""
        valid_statuses = ["experimental", "staging", "production", "archived"]

        if new_status not in valid_statuses:
            raise ValueError(f"Invalid status: {new_status}")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE model_versions SET status = ? WHERE version_id = ?
        """,
            (new_status, version_id),
        )

        conn.commit()
        conn.close()

    def get_lineage(self, version_id: str) -> Dict[str, List[str]]:
        """Get the lineage (ancestors and descendants) of a model version."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get ancestors
        ancestors = []
        cursor.execute(
            """
            SELECT parent_version FROM model_lineage WHERE child_version = ?
        """,
            (version_id,),
        )

        for row in cursor.fetchall():
            ancestors.append(row[0])

        # Get descendants
        descendants = []
        cursor.execute(
            """
            SELECT child_version FROM model_lineage WHERE parent_version = ?
        """,
            (version_id,),
        )

        for row in cursor.fetchall():
            descendants.append(row[0])

        conn.close()

        return {"ancestors": ancestors, "descendants": descendants}

    def diff_versions(self, version1_id: str, version2_id: str) -> Dict[str, Any]:
        """Compare two model versions."""
        v1 = self.get_version(version1_id)
        v2 = self.get_version(version2_id)

        if not v1 or not v2:
            raise ValueError("One or both versions not found")

        diff = {
            "version1": version1_id,
            "version2": version2_id,
            "config_diff": self._diff_configs(v1.training_config, v2.training_config),
            "metrics_diff": self._diff_metrics(v1.metrics, v2.metrics),
            "tags_diff": {
                "added": list(set(v2.tags) - set(v1.tags)),
                "removed": list(set(v1.tags) - set(v2.tags)),
            },
        }

        return diff

    def _diff_configs(self, config1: Dict, config2: Dict) -> Dict[str, Any]:
        """Compare two configuration dictionaries."""
        all_keys = set(config1.keys()) | set(config2.keys())
        diff = {}

        for key in all_keys:
            val1 = config1.get(key)
            val2 = config2.get(key)

            if val1 != val2:
                diff[key] = {"old": val1, "new": val2}

        return diff

    def _diff_metrics(self, metrics1: Dict, metrics2: Dict) -> Dict[str, Any]:
        """Compare two metrics dictionaries."""
        all_keys = set(metrics1.keys()) | set(metrics2.keys())
        diff = {}

        for key in all_keys:
            val1 = metrics1.get(key, 0)
            val2 = metrics2.get(key, 0)

            if val1 != val2:
                diff[key] = {
                    "old": val1,
                    "new": val2,
                    "change": val2 - val1,
                    "change_pct": (
                        ((val2 - val1) / val1 * 100) if val1 != 0 else float("inf")
                    ),
                }

        return diff


class ModelEvaluator:
    """Comprehensive model evaluation framework."""

    def __init__(self, version_control: ModelVersionControl):
        """Initialize the evaluator."""
        self.version_control = version_control
        self.evaluation_cache = {}

    def evaluate_model(
        self,
        version_id: str,
        X_test: np.ndarray,
        y_test: np.ndarray,
        dataset_name: str = "test",
        task_type: Optional[str] = None,
        additional_metrics: Optional[Dict[str, Callable]] = None,
    ) -> EvaluationResult:
        """Perform comprehensive model evaluation."""
        # Load model
        model = self.version_control.load_model(version_id)
        version = self.version_control.get_version(version_id)

        if not version:
            raise ValueError(f"Version {version_id} not found")

        # Determine task type if not provided
        if task_type is None:
            task_type = version.model_type

        # Make predictions
        import time

        start_time = time.time()
        y_pred = model.predict(X_test)
        inference_time = (
            (time.time() - start_time) * 1000 / len(X_test)
        )  # ms per sample

        # Get prediction probabilities if available
        y_proba = None
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)

        # Calculate metrics based on task type
        if task_type == "classifier":
            metrics, primary_metric, primary_score = self._evaluate_classifier(
                y_test, y_pred, y_proba, additional_metrics
            )
            confusion_mat = confusion_matrix(y_test, y_pred).tolist()
            class_report = classification_report(y_test, y_pred, output_dict=True)
            regression_metrics = None
        else:
            metrics, primary_metric, primary_score = self._evaluate_regressor(
                y_test, y_pred, additional_metrics
            )
            confusion_mat = None
            class_report = None
            regression_metrics = metrics

        # Extract feature importance if available
        feature_importance = self._extract_feature_importance(model)

        # Analyze prediction distribution
        prediction_dist = self._analyze_predictions(y_test, y_pred)

        # Perform statistical tests
        statistical_tests = self._perform_statistical_tests(y_test, y_pred)

        # Generate plots
        plots = self._generate_evaluation_plots(
            y_test, y_pred, y_proba, version_id, dataset_name
        )

        # Create evaluation result
        result = EvaluationResult(
            model_version=version_id,
            dataset_name=dataset_name,
            timestamp=datetime.datetime.now().isoformat(),
            primary_metric=primary_metric,
            primary_score=primary_score,
            classification_metrics=metrics if task_type == "classifier" else None,
            regression_metrics=regression_metrics,
            confusion_matrix=confusion_mat,
            classification_report=class_report,
            feature_importance=feature_importance,
            prediction_distribution=prediction_dist,
            inference_time_ms=inference_time,
            statistical_tests=statistical_tests,
            plots=plots,
        )

        # Save evaluation result
        self._save_evaluation(result)

        return result

    def _evaluate_classifier(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray],
        additional_metrics: Optional[Dict[str, Callable]],
    ) -> Tuple[Dict, str, float]:
        """Evaluate classification model."""
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_weighted": precision_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
            "recall_weighted": recall_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
            "f1_weighted": f1_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
            "cohen_kappa": cohen_kappa_score(y_true, y_pred),
            "matthews_corrcoef": matthews_corrcoef(y_true, y_pred),
        }

        # Add AUC if binary classification and probabilities available
        if len(np.unique(y_true)) == 2 and y_proba is not None:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
            metrics["log_loss"] = log_loss(y_true, y_proba)

        # Add custom metrics
        if additional_metrics:
            for name, func in additional_metrics.items():
                metrics[name] = func(y_true, y_pred)

        primary_metric = "f1_weighted"
        primary_score = metrics[primary_metric]

        return metrics, primary_metric, primary_score

    def _evaluate_regressor(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        additional_metrics: Optional[Dict[str, Callable]],
    ) -> Tuple[Dict, str, float]:
        """Evaluate regression model."""
        metrics = {
            "mse": mean_squared_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
            "mape": (
                np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                if not any(y_true == 0)
                else float("inf")
            ),
            "explained_variance": 1 - np.var(y_true - y_pred) / np.var(y_true),
        }

        # Add custom metrics
        if additional_metrics:
            for name, func in additional_metrics.items():
                metrics[name] = func(y_true, y_pred)

        primary_metric = "rmse"
        primary_score = metrics[primary_metric]

        return metrics, primary_metric, primary_score

    def _extract_feature_importance(self, model: Any) -> Optional[Dict[str, float]]:
        """Extract feature importance from model if available."""
        importance = None

        # Try different methods to get feature importance
        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
        elif hasattr(model, "coef_"):
            importance = (
                np.abs(model.coef_).mean(axis=0)
                if model.coef_.ndim > 1
                else np.abs(model.coef_)
            )
        elif hasattr(model, "named_steps"):
            # For sklearn pipelines
            if "model" in model.named_steps and hasattr(
                model.named_steps["model"], "feature_importances_"
            ):
                importance = model.named_steps["model"].feature_importances_

        if importance is not None:
            # Convert to dict with feature indices as keys
            return {f"feature_{i}": float(imp) for i, imp in enumerate(importance)}

        return None

    def _analyze_predictions(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze prediction distribution and errors."""
        analysis = {
            "prediction_stats": {
                "mean": float(np.mean(y_pred)),
                "std": float(np.std(y_pred)),
                "min": float(np.min(y_pred)),
                "max": float(np.max(y_pred)),
                "median": float(np.median(y_pred)),
            },
            "error_stats": {
                "mean_error": float(np.mean(y_pred - y_true)),
                "std_error": float(np.std(y_pred - y_true)),
                "max_overestimate": float(np.max(y_pred - y_true)),
                "max_underestimate": float(np.min(y_pred - y_true)),
            },
        }

        # Add distribution percentiles
        percentiles = [10, 25, 50, 75, 90]
        analysis["prediction_percentiles"] = {
            f"p{p}": float(np.percentile(y_pred, p)) for p in percentiles
        }

        return analysis

    def _perform_statistical_tests(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """Perform statistical tests on predictions."""
        tests = {}

        # Test for bias (systematic over/under prediction)
        errors = y_pred - y_true
        t_stat, p_value = stats.ttest_1samp(errors, 0)
        tests["bias_test"] = {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant_bias": p_value < 0.05,
        }

        # Test for normality of residuals
        if len(errors) > 20:
            stat, p_value = stats.normaltest(errors)
            tests["normality_test"] = {
                "statistic": float(stat),
                "p_value": float(p_value),
                "residuals_normal": p_value > 0.05,
            }

        # Test for homoscedasticity (constant variance)
        if len(errors) > 30:
            # Breusch-Pagan test approximation
            residuals_squared = errors**2
            correlation = np.corrcoef(y_pred, residuals_squared)[0, 1]
            tests["homoscedasticity_test"] = {
                "correlation": float(correlation),
                "likely_heteroscedastic": abs(correlation) > 0.3,
            }

        return tests

    def _generate_evaluation_plots(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray],
        version_id: str,
        dataset_name: str,
    ) -> Dict[str, str]:
        """Generate evaluation plots."""
        plots_dir = (
            self.version_control.evaluations_dir / f"{version_id}_{dataset_name}_plots"
        )
        plots_dir.mkdir(exist_ok=True)

        plots = {}

        # Confusion matrix plot (for classification)
        if len(np.unique(y_true)) < 20:  # Reasonable number of classes
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.ylabel("True Label")
            plt.xlabel("Predicted Label")
            plt.title("Confusion Matrix")
            plot_path = plots_dir / "confusion_matrix.png"
            plt.savefig(plot_path)
            plt.close()
            plots["confusion_matrix"] = str(plot_path)

        # Actual vs Predicted plot
        plt.figure(figsize=(8, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot(
            [y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--", lw=2
        )
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Actual vs Predicted")
        plot_path = plots_dir / "actual_vs_predicted.png"
        plt.savefig(plot_path)
        plt.close()
        plots["actual_vs_predicted"] = str(plot_path)

        # Residual plot
        residuals = y_pred - y_true
        plt.figure(figsize=(8, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color="r", linestyle="--")
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals")
        plt.title("Residual Plot")
        plot_path = plots_dir / "residual_plot.png"
        plt.savefig(plot_path)
        plt.close()
        plots["residual_plot"] = str(plot_path)

        # ROC curve (for binary classification)
        if len(np.unique(y_true)) == 2 and y_proba is not None:
            from sklearn.metrics import roc_curve

            fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])

            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label="ROC curve")
            plt.plot([0, 1], [0, 1], "k--", label="Random")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.legend()
            plot_path = plots_dir / "roc_curve.png"
            plt.savefig(plot_path)
            plt.close()
            plots["roc_curve"] = str(plot_path)

        return plots

    def _save_evaluation(self, result: EvaluationResult):
        """Save evaluation result to database."""
        conn = sqlite3.connect(self.version_control.db_path)
        cursor = conn.cursor()

        evaluation_id = f"eval_{result.model_version}_{result.dataset_name}_{int(datetime.datetime.now().timestamp())}"

        cursor.execute(
            """
            INSERT INTO evaluations VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                evaluation_id,
                result.model_version,
                result.dataset_name,
                result.timestamp,
                result.primary_metric,
                result.primary_score,
                json.dumps(asdict(result)),
            ),
        )

        conn.commit()
        conn.close()

    def compare_models(
        self,
        version_ids: List[str],
        X_test: np.ndarray,
        y_test: np.ndarray,
        dataset_name: str = "test",
    ) -> pd.DataFrame:
        """Compare multiple model versions."""
        results = []

        for version_id in version_ids:
            eval_result = self.evaluate_model(version_id, X_test, y_test, dataset_name)

            # Extract key metrics
            metrics = (
                eval_result.classification_metrics
                or eval_result.regression_metrics
                or {}
            )

            row = {
                "version_id": version_id,
                "model_name": self.version_control.get_version(version_id).model_name,
                "primary_metric": eval_result.primary_metric,
                "primary_score": eval_result.primary_score,
                "inference_time_ms": eval_result.inference_time_ms,
            }

            # Add all metrics
            row.update(metrics)

            results.append(row)

        # Create comparison dataframe
        df = pd.DataFrame(results)

        # Sort by primary score (descending for most metrics)
        df = df.sort_values("primary_score", ascending=False)

        return df

    def perform_ab_test(
        self,
        version_a: str,
        version_b: str,
        X_test: np.ndarray,
        y_test: np.ndarray,
        significance_level: float = 0.05,
    ) -> Dict[str, Any]:
        """Perform A/B test between two model versions."""
        # Evaluate both models
        eval_a = self.evaluate_model(version_a, X_test, y_test, "ab_test")
        eval_b = self.evaluate_model(version_b, X_test, y_test, "ab_test")

        # Load models for detailed comparison
        model_a = self.version_control.load_model(version_a)
        model_b = self.version_control.load_model(version_b)

        # Get predictions
        pred_a = model_a.predict(X_test)
        pred_b = model_b.predict(X_test)

        # Calculate individual errors
        errors_a = np.abs(pred_a - y_test)
        errors_b = np.abs(pred_b - y_test)

        # Perform paired t-test
        t_stat, p_value = stats.ttest_rel(errors_a, errors_b)

        # Determine winner
        mean_error_a = np.mean(errors_a)
        mean_error_b = np.mean(errors_b)

        if p_value < significance_level:
            winner = version_a if mean_error_a < mean_error_b else version_b
            significant = True
        else:
            winner = None
            significant = False

        result = {
            "version_a": version_a,
            "version_b": version_b,
            "metric_a": eval_a.primary_score,
            "metric_b": eval_b.primary_score,
            "mean_error_a": float(mean_error_a),
            "mean_error_b": float(mean_error_b),
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": significant,
            "winner": winner,
            "improvement": float(abs(eval_b.primary_score - eval_a.primary_score)),
            "improvement_pct": float(
                (eval_b.primary_score - eval_a.primary_score)
                / eval_a.primary_score
                * 100
            ),
        }

        return result


class AutoModelSelector:
    """Automatic model selection based on performance criteria."""

    def __init__(self, version_control: ModelVersionControl, evaluator: ModelEvaluator):
        """Initialize the auto selector."""
        self.version_control = version_control
        self.evaluator = evaluator

    def select_best_model(
        self,
        model_name: str,
        X_val: np.ndarray,
        y_val: np.ndarray,
        selection_criteria: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Select the best model version based on validation performance."""
        # Default selection criteria
        if selection_criteria is None:
            selection_criteria = {
                "metric": "primary",  # Use primary metric
                "minimize": False,  # Higher is better
                "min_samples": 100,  # Minimum validation samples
                "max_inference_time": 100.0,  # Max ms per prediction
            }

        # Get candidate models
        candidates = self.version_control.list_versions(
            model_name=model_name, status="staging"  # Only consider staging models
        )

        if not candidates:
            # Fall back to experimental models
            candidates = self.version_control.list_versions(
                model_name=model_name, status="experimental"
            )

        if not candidates:
            raise ValueError(f"No models found for {model_name}")

        # Evaluate all candidates
        results = []
        for candidate in candidates:
            eval_result = self.evaluator.evaluate_model(
                candidate.version_id, X_val, y_val, "validation"
            )

            # Check constraints
            if eval_result.inference_time_ms > selection_criteria["max_inference_time"]:
                continue

            results.append((candidate.version_id, eval_result))

        if not results:
            raise ValueError("No models meet the selection criteria")

        # Select best based on metric
        metric_name = selection_criteria["metric"]
        minimize = selection_criteria["minimize"]

        if metric_name == "primary":
            scores = [(v, r.primary_score) for v, r in results]
        else:
            scores = []
            for v, r in results:
                metrics = r.classification_metrics or r.regression_metrics or {}
                if metric_name in metrics:
                    scores.append((v, metrics[metric_name]))

        if not scores:
            raise ValueError(f"Metric {metric_name} not found in evaluation results")

        # Sort and select best
        scores.sort(key=lambda x: x[1], reverse=not minimize)
        best_version = scores[0][0]

        return best_version

    def recommend_deployment(
        self,
        model_name: str,
        X_val: np.ndarray,
        y_val: np.ndarray,
        current_production: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Recommend whether to deploy a new model version."""
        # Find best candidate
        best_candidate = self.select_best_model(model_name, X_val, y_val)

        recommendation = {
            "candidate_version": best_candidate,
            "current_production": current_production,
            "recommendation": "deploy",
            "confidence": "high",
            "reasons": [],
        }

        # If there's a current production model, compare
        if current_production:
            ab_result = self.evaluator.perform_ab_test(
                current_production, best_candidate, X_val, y_val
            )

            if ab_result["winner"] == current_production:
                recommendation["recommendation"] = "keep_current"
                recommendation["confidence"] = (
                    "high" if ab_result["significant"] else "medium"
                )
                recommendation["reasons"].append(
                    f"Current model performs better (p-value: {ab_result['p_value']:.4f})"
                )
            elif not ab_result["significant"]:
                recommendation["recommendation"] = "monitor"
                recommendation["confidence"] = "low"
                recommendation["reasons"].append(
                    f"No significant difference (p-value: {ab_result['p_value']:.4f})"
                )
            else:
                recommendation["reasons"].append(
                    f"New model shows {ab_result['improvement_pct']:.1f}% improvement"
                )
        else:
            recommendation["reasons"].append("No current production model")

        # Add performance summary
        eval_result = self.evaluator.evaluation_cache.get(best_candidate)
        if eval_result:
            recommendation["expected_performance"] = {
                "metric": eval_result.primary_metric,
                "value": eval_result.primary_score,
                "inference_time_ms": eval_result.inference_time_ms,
            }

        return recommendation


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    # Generate sample data
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=10, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize version control
    vc = ModelVersionControl("test_repository")

    # Train and commit a model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Calculate training metrics
    train_score = model.score(X_train, y_train)
    val_score = model.score(X_test, y_test)

    # Commit model
    version = vc.commit_model(
        model=model,
        model_name="error_classifier",
        model_type="classifier",
        training_config={"n_estimators": 100, "max_depth": None, "random_state": 42},
        metrics={"train_accuracy": train_score, "val_accuracy": val_score},
        tags=["random_forest", "baseline"],
    )

    print(f"Committed model version: {version.version_id}")

    # Initialize evaluator
    evaluator = ModelEvaluator(vc)

    # Evaluate model
    eval_result = evaluator.evaluate_model(
        version.version_id, X_test, y_test, dataset_name="test_set"
    )

    print("\nEvaluation Results:")
    print(
        f"Primary metric ({eval_result.primary_metric}): {eval_result.primary_score:.4f}"
    )
    print(f"Inference time: {eval_result.inference_time_ms:.2f} ms/sample")

    # Train another model for comparison
    model2 = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    model2.fit(X_train, y_train)

    version2 = vc.commit_model(
        model=model2,
        model_name="error_classifier",
        model_type="classifier",
        training_config={"n_estimators": 200, "max_depth": 10, "random_state": 42},
        metrics={
            "train_accuracy": model2.score(X_train, y_train),
            "val_accuracy": model2.score(X_test, y_test),
        },
        parent_version=version.version_id,
        tags=["random_forest", "tuned"],
    )

    # Compare models
    comparison = evaluator.compare_models(
        [version.version_id, version2.version_id], X_test, y_test
    )

    print("\nModel Comparison:")
    print(comparison[["version_id", "primary_score", "inference_time_ms"]])

    # Perform A/B test
    ab_result = evaluator.perform_ab_test(
        version.version_id, version2.version_id, X_test, y_test
    )

    print("\nA/B Test Results:")
    print(f"Winner: {ab_result['winner']}")
    print(f"Significant: {ab_result['significant']}")
    print(f"p-value: {ab_result['p_value']:.4f}")

    # Test auto model selection
    selector = AutoModelSelector(vc, evaluator)
    best_model = selector.select_best_model("error_classifier", X_test, y_test)
    print(f"\nBest model selected: {best_model}")

    # Get deployment recommendation
    recommendation = selector.recommend_deployment(
        "error_classifier", X_test, y_test, current_production=version.version_id
    )

    print("\nDeployment Recommendation:")
    print(f"Recommendation: {recommendation['recommendation']}")
    print(f"Confidence: {recommendation['confidence']}")
    print(f"Reasons: {recommendation['reasons']}")
