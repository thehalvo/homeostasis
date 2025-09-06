"""
Model training infrastructure for Homeostasis ML components.

This module provides a comprehensive training pipeline with support for:
- Multiple model types (classifiers, regressors, embeddings, generative)
- Distributed training capabilities
- Hyperparameter optimization
- Cross-validation and evaluation
- Model versioning and artifact management
"""

import datetime
import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import joblib
import numpy as np
import optuna  # For advanced hyperparameter optimization
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    model_type: str  # classifier, regressor, embeddings, generative
    model_name: str
    data_path: str
    output_dir: str
    test_size: float = 0.2
    validation_size: float = 0.1
    random_state: int = 42
    n_jobs: int = -1  # Use all available cores
    cv_folds: int = 5
    optimization_method: str = "grid"  # grid, random, bayesian
    optimization_trials: int = 100
    early_stopping: bool = True
    early_stopping_patience: int = 10
    distributed: bool = False
    distributed_backend: str = "dask"  # dask, ray, spark
    experiment_tracking: str = "mlflow"  # mlflow, wandb, none
    versioning_enabled: bool = True
    auto_feature_engineering: bool = True
    ensemble_models: List[str] = field(default_factory=list)
    custom_metrics: Dict[str, Callable] = field(default_factory=dict)


@dataclass
class TrainingResult:
    """Results from model training."""

    model_id: str
    model_path: str
    metrics: Dict[str, float]
    best_params: Dict[str, Any]
    feature_importance: Optional[Dict[str, float]]
    training_time: float
    validation_scores: List[float]
    test_predictions: Optional[np.ndarray]
    confusion_matrix: Optional[np.ndarray]
    classification_report: Optional[Dict[str, Any]]
    artifacts: Dict[str, str]  # Additional artifacts (plots, reports, etc.)


class ModelRegistry:
    """Registry for managing trained models."""

    def __init__(self, registry_path: str = "models/registry"):
        """Initialize the model registry."""
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.registry_path / "models.json"
        self.models = self._load_registry()

    def _load_registry(self) -> Dict[str, Any]:
        """Load the model registry from disk."""
        if self.registry_file.exists():
            with open(self.registry_file, "r") as f:
                return json.load(f)
        return {"models": {}, "latest": {}}

    def _save_registry(self):
        """Save the model registry to disk."""
        with open(self.registry_file, "w") as f:
            json.dump(self.models, f, indent=2)

    def register_model(self, model_id: str, model_info: Dict[str, Any]):
        """Register a new model."""
        self.models["models"][model_id] = {
            **model_info,
            "registered_at": datetime.datetime.now().isoformat(),
        }

        # Update latest model for this type
        model_type = model_info.get("model_type", "unknown")
        self.models["latest"][model_type] = model_id

        self._save_registry()
        logger.info(f"Registered model {model_id} of type {model_type}")

    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model information by ID."""
        return self.models["models"].get(model_id)

    def get_latest_model(self, model_type: str) -> Optional[str]:
        """Get the latest model ID for a given type."""
        return self.models["latest"].get(model_type)

    def list_models(self, model_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all models, optionally filtered by type."""
        models = []
        for model_id, info in self.models["models"].items():
            if model_type is None or info.get("model_type") == model_type:
                models.append({"id": model_id, **info})
        return sorted(models, key=lambda x: x.get("registered_at", ""), reverse=True)


class FeatureEngineer:
    """Automated feature engineering for ML models."""

    def __init__(self, feature_config: Optional[Dict[str, Any]] = None):
        """Initialize the feature engineer."""
        self.feature_config = feature_config or {}
        self.transformers = {}
        self.feature_names = []

    def fit_transform(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Fit and transform features."""
        # This is a placeholder - implement actual feature engineering
        # based on the data type and problem
        logger.info("Performing automated feature engineering...")

        # Example: Add polynomial features for numeric data
        if self.feature_config.get("polynomial_features", False):
            from sklearn.preprocessing import PolynomialFeatures

            poly = PolynomialFeatures(degree=2, include_bias=False)
            X_poly = poly.fit_transform(X)
            self.transformers["polynomial"] = poly
            return X_poly

        return X

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features using fitted transformers."""
        X_transformed = X

        if "polynomial" in self.transformers:
            X_transformed = self.transformers["polynomial"].transform(X_transformed)

        return X_transformed


class DistributedTrainer:
    """Handle distributed training across multiple nodes/GPUs."""

    def __init__(self, backend: str = "dask"):
        """Initialize distributed trainer."""
        self.backend = backend
        self._setup_backend()

    def _setup_backend(self):
        """Set up the distributed computing backend."""
        if self.backend == "dask":
            try:
                from dask.distributed import Client

                self.client = Client()
                logger.info(f"Dask client initialized: {self.client}")
            except ImportError:
                logger.warning("Dask not installed, falling back to local training")
                self.client = None
        elif self.backend == "ray":
            try:
                import ray

                ray.init(ignore_reinit_error=True)
                logger.info("Ray initialized for distributed training")
            except ImportError:
                logger.warning("Ray not installed, falling back to local training")

    def train_distributed(self, train_func: Callable, *args, **kwargs):
        """Execute training in a distributed manner."""
        if self.backend == "dask" and self.client:
            future = self.client.submit(train_func, *args, **kwargs)
            return future.result()
        elif self.backend == "ray":
            try:
                import ray

                remote_func = ray.remote(train_func)
                return ray.get(remote_func.remote(*args, **kwargs))
            except Exception:
                pass

        # Fallback to local training
        return train_func(*args, **kwargs)


class ModelTrainer:
    """Main trainer class for all model types."""

    def __init__(self, config: TrainingConfig):
        """Initialize the model trainer."""
        self.config = config
        self.registry = ModelRegistry(os.path.join(config.output_dir, "registry"))
        self.feature_engineer = (
            FeatureEngineer() if config.auto_feature_engineering else None
        )
        self.distributed_trainer = (
            DistributedTrainer(config.distributed_backend)
            if config.distributed
            else None
        )
        self._setup_experiment_tracking()

    def _setup_experiment_tracking(self):
        """Set up experiment tracking system."""
        if self.config.experiment_tracking == "mlflow":
            try:
                import mlflow

                mlflow.set_tracking_uri(f"file://{self.config.output_dir}/mlruns")
                mlflow.set_experiment(f"homeostasis_{self.config.model_type}")
                logger.info("MLflow tracking initialized")
            except ImportError:
                logger.warning("MLflow not installed, experiment tracking disabled")
        elif self.config.experiment_tracking == "wandb":
            try:
                import wandb

                wandb.init(
                    project=f"homeostasis_{self.config.model_type}",
                    config=self.config.__dict__,
                )
                logger.info("Weights & Biases tracking initialized")
            except ImportError:
                logger.warning("wandb not installed, experiment tracking disabled")

    def _generate_model_id(self) -> str:
        """Generate a unique model ID."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        config_hash = hashlib.md5(
            json.dumps(self.config.__dict__, sort_keys=True).encode()
        ).hexdigest()[:8]
        return f"{self.config.model_name}_{timestamp}_{config_hash}"

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load training data from the specified path."""
        logger.info(f"Loading data from {self.config.data_path}")

        # This is a placeholder - implement actual data loading
        # based on the data format (jsonl, csv, parquet, etc.)

        # For now, assume we're loading from the data collector format
        from .data_collector import ErrorDataCollector

        collector = ErrorDataCollector(data_dir=os.path.dirname(self.config.data_path))
        training_data = collector.get_training_data()

        if not training_data:
            raise ValueError("No training data found")

        # Extract features and labels
        from .error_classifier import ErrorClassificationFeatures

        feature_extractor = ErrorClassificationFeatures()

        X_text = []
        y = []

        for error_data, label in training_data:
            X_text.append(feature_extractor.prepare_text_features(error_data))
            y.append(label)

        return np.array(X_text), np.array(y)

    def _split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Split data into train, validation, and test sets."""
        # First split: train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y if self.config.model_type == "classifier" else None,
        )

        # Second split: train and validation
        val_size_adjusted = self.config.validation_size / (1 - self.config.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=val_size_adjusted,
            random_state=self.config.random_state,
            stratify=y_temp if self.config.model_type == "classifier" else None,
        )

        logger.info(
            f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}"
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    def _get_model(self, model_name: str) -> Any:
        """Get a model instance by name."""
        # Import models based on type
        if self.config.model_type == "classifier":
            from lightgbm import LGBMClassifier
            from sklearn.ensemble import (
                GradientBoostingClassifier,
                RandomForestClassifier,
            )
            from sklearn.neural_network import MLPClassifier
            from sklearn.svm import SVC
            from xgboost import XGBClassifier

            models = {
                "random_forest": RandomForestClassifier(n_jobs=self.config.n_jobs),
                "gradient_boosting": GradientBoostingClassifier(),
                "svm": SVC(probability=True),
                "mlp": MLPClassifier(max_iter=1000),
                "xgboost": XGBClassifier(n_jobs=self.config.n_jobs),
                "lightgbm": LGBMClassifier(n_jobs=self.config.n_jobs),
            }
        elif self.config.model_type == "regressor":
            from lightgbm import LGBMRegressor
            from sklearn.ensemble import (
                GradientBoostingRegressor,
                RandomForestRegressor,
            )
            from sklearn.neural_network import MLPRegressor
            from sklearn.svm import SVR
            from xgboost import XGBRegressor

            models = {
                "random_forest": RandomForestRegressor(n_jobs=self.config.n_jobs),
                "gradient_boosting": GradientBoostingRegressor(),
                "svm": SVR(),
                "mlp": MLPRegressor(max_iter=1000),
                "xgboost": XGBRegressor(n_jobs=self.config.n_jobs),
                "lightgbm": LGBMRegressor(n_jobs=self.config.n_jobs),
            }
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")

        if model_name not in models:
            raise ValueError(f"Unknown model: {model_name}")

        return models[model_name]

    def _get_hyperparameter_space(self, model_name: str) -> Dict[str, Any]:
        """Get hyperparameter search space for a model."""
        if "random_forest" in model_name:
            return {
                "n_estimators": [50, 100, 200, 300],
                "max_depth": [None, 10, 20, 30, 50],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            }
        elif "gradient_boosting" in model_name:
            return {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.3],
                "max_depth": [3, 5, 7, 10],
                "subsample": [0.8, 0.9, 1.0],
            }
        elif "svm" in model_name:
            return {
                "C": [0.1, 1, 10, 100],
                "kernel": ["linear", "rbf", "poly"],
                "gamma": ["scale", "auto", 0.001, 0.01],
            }
        elif "mlp" in model_name:
            return {
                "hidden_layer_sizes": [(50,), (100,), (100, 50), (100, 100)],
                "activation": ["relu", "tanh"],
                "learning_rate": ["constant", "adaptive"],
                "alpha": [0.0001, 0.001, 0.01],
            }
        elif "xgboost" in model_name:
            return {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.01, 0.1, 0.3],
                "max_depth": [3, 5, 7, 10],
                "subsample": [0.8, 0.9, 1.0],
                "colsample_bytree": [0.8, 0.9, 1.0],
            }
        elif "lightgbm" in model_name:
            return {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.01, 0.1, 0.3],
                "num_leaves": [31, 50, 100],
                "feature_fraction": [0.8, 0.9, 1.0],
                "bagging_fraction": [0.8, 0.9, 1.0],
            }
        else:
            return {}

    def _optimize_hyperparameters(
        self, model: Any, X_train: np.ndarray, y_train: np.ndarray
    ) -> Tuple[Any, Dict[str, Any]]:
        """Optimize model hyperparameters."""
        logger.info(
            f"Optimizing hyperparameters using {self.config.optimization_method}"
        )

        param_space = self._get_hyperparameter_space(self.config.model_name)

        if not param_space:
            logger.warning("No hyperparameter space defined, using default parameters")
            return model, {}

        if self.config.optimization_method == "grid":
            search = GridSearchCV(
                model,
                param_space,
                cv=self.config.cv_folds,
                n_jobs=self.config.n_jobs,
                verbose=1,
            )
        elif self.config.optimization_method == "random":
            search = RandomizedSearchCV(
                model,
                param_space,
                n_iter=self.config.optimization_trials,
                cv=self.config.cv_folds,
                n_jobs=self.config.n_jobs,
                verbose=1,
            )
        elif self.config.optimization_method == "bayesian":
            # Use Optuna for Bayesian optimization
            return self._optimize_with_optuna(model, X_train, y_train, param_space)
        else:
            raise ValueError(
                f"Unknown optimization method: {self.config.optimization_method}"
            )

        search.fit(X_train, y_train)

        return search.best_estimator_, search.best_params_

    def _optimize_with_optuna(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        param_space: Dict[str, Any],
    ) -> Tuple[Any, Dict[str, Any]]:
        """Optimize hyperparameters using Optuna."""

        def objective(trial):
            params = {}
            for param_name, param_values in param_space.items():
                if isinstance(param_values[0], (int, float)):
                    if isinstance(param_values[0], int):
                        params[param_name] = trial.suggest_int(
                            param_name, min(param_values), max(param_values)
                        )
                    else:
                        params[param_name] = trial.suggest_float(
                            param_name, min(param_values), max(param_values)
                        )
                else:
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_values
                    )

            model_copy = model.__class__(**params)
            scores = cross_val_score(
                model_copy,
                X_train,
                y_train,
                cv=self.config.cv_folds,
                n_jobs=self.config.n_jobs,
            )
            return scores.mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.config.optimization_trials)

        best_params = study.best_params
        best_model = model.__class__(**best_params)
        best_model.fit(X_train, y_train)

        return best_model, best_params

    def _evaluate_model(
        self, model: Any, X_test: np.ndarray, y_test: np.ndarray
    ) -> Dict[str, Any]:
        """Evaluate the trained model."""
        logger.info("Evaluating model on test set")

        y_pred = model.predict(X_test)

        metrics = {}

        if self.config.model_type == "classifier":
            metrics["accuracy"] = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average="weighted"
            )
            metrics["precision"] = precision
            metrics["recall"] = recall
            metrics["f1_score"] = f1

            # Add custom metrics
            for metric_name, metric_func in self.config.custom_metrics.items():
                metrics[metric_name] = metric_func(y_test, y_pred)

            # Generate classification report and confusion matrix
            class_report = classification_report(y_test, y_pred, output_dict=True)
            conf_matrix = confusion_matrix(y_test, y_pred)

            return {
                "metrics": metrics,
                "classification_report": class_report,
                "confusion_matrix": conf_matrix.tolist(),
                "predictions": y_pred,
            }

        elif self.config.model_type == "regressor":
            metrics["mse"] = mean_squared_error(y_test, y_pred)
            metrics["mae"] = mean_absolute_error(y_test, y_pred)
            metrics["rmse"] = np.sqrt(metrics["mse"])

            # Add custom metrics
            for metric_name, metric_func in self.config.custom_metrics.items():
                metrics[metric_name] = metric_func(y_test, y_pred)

            return {"metrics": metrics, "predictions": y_pred}

    def _save_model(self, model: Any, model_id: str) -> str:
        """Save the trained model."""
        model_dir = Path(self.config.output_dir) / "models" / model_id
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / "model.pkl"

        # Save with joblib for better performance with numpy arrays
        joblib.dump(model, model_path)

        logger.info(f"Model saved to {model_path}")

        return str(model_path)

    def _save_artifacts(
        self, model_id: str, artifacts: Dict[str, Any]
    ) -> Dict[str, str]:
        """Save additional training artifacts."""
        artifact_dir = Path(self.config.output_dir) / "models" / model_id / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        artifact_paths = {}

        for name, artifact in artifacts.items():
            if name == "feature_importance" and artifact is not None:
                path = artifact_dir / "feature_importance.json"
                with open(path, "w") as f:
                    json.dump(artifact, f, indent=2)
                artifact_paths[name] = str(path)

            elif name == "training_history":
                path = artifact_dir / "training_history.json"
                with open(path, "w") as f:
                    json.dump(artifact, f, indent=2)
                artifact_paths[name] = str(path)

            elif name == "evaluation_report":
                path = artifact_dir / "evaluation_report.json"
                with open(path, "w") as f:
                    json.dump(artifact, f, indent=2)
                artifact_paths[name] = str(path)

        return artifact_paths

    def train(self) -> TrainingResult:
        """Execute the complete training pipeline."""
        start_time = datetime.datetime.now()

        # Generate model ID
        model_id = self._generate_model_id()
        logger.info(f"Starting training for model {model_id}")

        # Start experiment tracking
        if self.config.experiment_tracking == "mlflow":
            try:
                import mlflow

                mlflow.start_run(run_name=model_id)
                mlflow.log_params(self.config.__dict__)
            except Exception:
                pass

        # Load data
        X, y = self._load_data()

        # Feature engineering
        if self.feature_engineer:
            X = self.feature_engineer.fit_transform(X, y)

        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self._split_data(X, y)

        # Get base model
        base_model = self._get_model(self.config.model_name)

        # Create pipeline if we're dealing with text data
        if isinstance(X[0], str):
            from sklearn.feature_extraction.text import TfidfVectorizer

            pipeline = Pipeline(
                [
                    (
                        "vectorizer",
                        TfidfVectorizer(max_features=5000, ngram_range=(1, 2)),
                    ),
                    ("model", base_model),
                ]
            )
            model = pipeline
        else:
            model = base_model

        # Optimize hyperparameters
        if self.config.distributed and self.distributed_trainer:
            model, best_params = self.distributed_trainer.train_distributed(
                self._optimize_hyperparameters, model, X_train, y_train
            )
        else:
            model, best_params = self._optimize_hyperparameters(model, X_train, y_train)

        # Evaluate on validation set
        val_scores = cross_val_score(
            model, X_val, y_val, cv=self.config.cv_folds, n_jobs=self.config.n_jobs
        )

        # Final evaluation on test set
        evaluation = self._evaluate_model(model, X_test, y_test)

        # Extract feature importance if available
        feature_importance = None
        if hasattr(model, "feature_importances_"):
            feature_importance = dict(
                zip(
                    range(len(model.feature_importances_)),
                    model.feature_importances_.tolist(),
                )
            )
        elif hasattr(model, "named_steps") and hasattr(
            model.named_steps["model"], "feature_importances_"
        ):
            feature_importance = dict(
                zip(
                    range(len(model.named_steps["model"].feature_importances_)),
                    model.named_steps["model"].feature_importances_.tolist(),
                )
            )

        # Save model
        model_path = self._save_model(model, model_id)

        # Save artifacts
        artifacts = {
            "feature_importance": feature_importance,
            "training_history": {
                "validation_scores": val_scores.tolist(),
                "best_params": best_params,
            },
            "evaluation_report": evaluation,
        }
        artifact_paths = self._save_artifacts(model_id, artifacts)

        # Calculate training time
        training_time = (datetime.datetime.now() - start_time).total_seconds()

        # Create result
        result = TrainingResult(
            model_id=model_id,
            model_path=model_path,
            metrics=evaluation["metrics"],
            best_params=best_params,
            feature_importance=feature_importance,
            training_time=training_time,
            validation_scores=val_scores.tolist(),
            test_predictions=evaluation.get("predictions"),
            confusion_matrix=evaluation.get("confusion_matrix"),
            classification_report=evaluation.get("classification_report"),
            artifacts=artifact_paths,
        )

        # Register model
        model_info = {
            "model_type": self.config.model_type,
            "model_name": self.config.model_name,
            "model_path": model_path,
            "metrics": evaluation["metrics"],
            "best_params": best_params,
            "training_time": training_time,
            "config": self.config.__dict__,
        }
        self.registry.register_model(model_id, model_info)

        # End experiment tracking
        if self.config.experiment_tracking == "mlflow":
            try:
                import mlflow

                mlflow.log_metrics(evaluation["metrics"])
                mlflow.log_artifact(model_path)
                mlflow.end_run()
            except Exception:
                pass
        elif self.config.experiment_tracking == "wandb":
            try:
                import wandb

                wandb.log(evaluation["metrics"])
                wandb.save(model_path)
                wandb.finish()
            except Exception:
                pass

        logger.info(f"Training completed for model {model_id}")
        logger.info(f"Metrics: {evaluation['metrics']}")

        return result


class EnsembleTrainer(ModelTrainer):
    """Trainer for ensemble models combining multiple base models."""

    def __init__(self, config: TrainingConfig):
        """Initialize ensemble trainer."""
        super().__init__(config)
        if not config.ensemble_models:
            raise ValueError("Ensemble models list cannot be empty")

    def train(self) -> TrainingResult:
        """Train an ensemble of models."""
        logger.info(f"Training ensemble with models: {self.config.ensemble_models}")

        # Train individual models
        base_results = []
        base_models = []

        for model_name in self.config.ensemble_models:
            # Create config for base model
            base_config = TrainingConfig(
                model_type=self.config.model_type,
                model_name=model_name,
                data_path=self.config.data_path,
                output_dir=self.config.output_dir,
                test_size=self.config.test_size,
                validation_size=self.config.validation_size,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs,
                cv_folds=self.config.cv_folds,
                optimization_method=self.config.optimization_method,
                optimization_trials=self.config.optimization_trials
                // len(self.config.ensemble_models),
                distributed=self.config.distributed,
                distributed_backend=self.config.distributed_backend,
                experiment_tracking="none",  # Track only ensemble
                versioning_enabled=False,
                auto_feature_engineering=self.config.auto_feature_engineering,
            )

            # Train base model
            base_trainer = ModelTrainer(base_config)
            result = base_trainer.train()
            base_results.append(result)

            # Load trained model
            base_model = joblib.load(result.model_path)
            base_models.append(base_model)

        # Create ensemble model
        if self.config.model_type == "classifier":
            from sklearn.ensemble import VotingClassifier

            ensemble = VotingClassifier(
                estimators=[
                    (f"model_{i}", model) for i, model in enumerate(base_models)
                ],
                voting="soft",  # Use predicted probabilities
            )
        else:
            from sklearn.ensemble import VotingRegressor

            ensemble = VotingRegressor(
                estimators=[
                    (f"model_{i}", model) for i, model in enumerate(base_models)
                ]
            )

        # Load data for final training
        X, y = self._load_data()
        if self.feature_engineer:
            X = self.feature_engineer.fit_transform(X, y)

        X_train, X_val, X_test, y_train, y_val, y_test = self._split_data(X, y)

        # Train ensemble
        ensemble.fit(X_train, y_train)

        # Evaluate ensemble
        evaluation = self._evaluate_model(ensemble, X_test, y_test)

        # Save ensemble model
        model_id = self._generate_model_id()
        model_path = self._save_model(ensemble, model_id)

        # Create result
        result = TrainingResult(
            model_id=model_id,
            model_path=model_path,
            metrics=evaluation["metrics"],
            best_params={"base_models": self.config.ensemble_models},
            feature_importance=None,
            training_time=sum(r.training_time for r in base_results),
            validation_scores=[],
            test_predictions=evaluation.get("predictions"),
            confusion_matrix=evaluation.get("confusion_matrix"),
            classification_report=evaluation.get("classification_report"),
            artifacts={"base_models": [r.model_path for r in base_results]},
        )

        # Register ensemble model
        model_info = {
            "model_type": f"ensemble_{self.config.model_type}",
            "model_name": f"ensemble_{'-'.join(self.config.ensemble_models)}",
            "model_path": model_path,
            "metrics": evaluation["metrics"],
            "base_models": self.config.ensemble_models,
            "config": self.config.__dict__,
        }
        self.registry.register_model(model_id, model_info)

        logger.info(f"Ensemble training completed: {model_id}")

        return result


# Utility functions for common training scenarios
def train_error_classifier(
    data_path: str,
    output_dir: str = "models",
    model_name: str = "random_forest",
    distributed: bool = False,
) -> TrainingResult:
    """Train an error classification model."""
    config = TrainingConfig(
        model_type="classifier",
        model_name=model_name,
        data_path=data_path,
        output_dir=output_dir,
        distributed=distributed,
        optimization_method="bayesian",
        optimization_trials=50,
    )

    trainer = ModelTrainer(config)
    return trainer.train()


def train_ensemble_classifier(
    data_path: str,
    output_dir: str = "models",
    ensemble_models: List[str] = None,
    distributed: bool = False,
) -> TrainingResult:
    """Train an ensemble error classifier."""
    if ensemble_models is None:
        ensemble_models = ["random_forest", "xgboost", "lightgbm"]

    config = TrainingConfig(
        model_type="classifier",
        model_name="ensemble",
        data_path=data_path,
        output_dir=output_dir,
        distributed=distributed,
        ensemble_models=ensemble_models,
        optimization_method="random",
        optimization_trials=30,
    )

    trainer = EnsembleTrainer(config)
    return trainer.train()


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Train Homeostasis ML models")
    parser.add_argument("--data-path", required=True, help="Path to training data")
    parser.add_argument(
        "--output-dir", default="models", help="Output directory for models"
    )
    parser.add_argument(
        "--model-type", default="classifier", choices=["classifier", "regressor"]
    )
    parser.add_argument(
        "--model-name", default="random_forest", help="Model name to train"
    )
    parser.add_argument("--ensemble", action="store_true", help="Train ensemble model")
    parser.add_argument(
        "--distributed", action="store_true", help="Use distributed training"
    )

    args = parser.parse_args()

    if args.ensemble:
        result = train_ensemble_classifier(
            args.data_path, args.output_dir, distributed=args.distributed
        )
    else:
        result = train_error_classifier(
            args.data_path,
            args.output_dir,
            model_name=args.model_name,
            distributed=args.distributed,
        )

    print("\nTraining completed!")
    print(f"Model ID: {result.model_id}")
    print(f"Metrics: {result.metrics}")
    print(f"Model saved to: {result.model_path}")
