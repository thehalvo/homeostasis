"""
Error classification model for Homeostasis.

This module implements a machine learning model for classifying errors into types,
providing a more flexible and adaptive way to analyze errors beyond rule-based matching.
"""

import os
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Add scikit-learn as dependency for the project


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
        "sklearn.ensemble._forest",
        "collections",
        "builtins",
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
        ("sklearn.ensemble._forest", "RandomForestClassifier"),
        ("sklearn.feature_extraction.text", "TfidfVectorizer"),
        ("sklearn.pipeline", "Pipeline"),
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


class ErrorClassificationFeatures:
    """Extracts features from error data for classification."""

    def __init__(self):
        """Initialize the feature extractor."""
        self.common_exception_types = {
            "KeyError",
            "ValueError",
            "TypeError",
            "AttributeError",
            "IndexError",
            "ImportError",
            "NameError",
            "AssertionError",
            "RuntimeError",
            "SyntaxError",
            "FileNotFoundError",
            "ZeroDivisionError",
            "PermissionError",
            "ConnectionError",
            "TimeoutError",
        }

        # Common patterns to look for in error messages
        self.message_patterns = {
            "missing_key": r"KeyError:.*?['\"](.*?)['\"]",
            "undefined_variable": r"NameError:.*?name ['\"](.*?)['\"] is not defined",
            "type_mismatch": r"TypeError:.*?",
            "attribute_missing": r"AttributeError:.*?has no attribute ['\"](.*?)['\"]",
            "index_out_of_range": r"IndexError:.*?",
            "zero_division": r"ZeroDivisionError:.*?",
            "file_not_found": r"FileNotFoundError:.*?['\"](.*?)['\"]",
            "import_error": r"ImportError:.*?['\"](.*?)['\"]",
            "connection_error": r"ConnectionError:.*?",
            "timeout_error": r"TimeoutError:.*?",
            "permission_error": r"PermissionError:.*?['\"](.*?)['\"]",
            "assertion_failed": r"AssertionError:.*?",
            "database_error": r"(database|sql|db).*?error",
            "syntax_error": r"SyntaxError:.*?",
            "not_callable": r".*?object is not callable",
            "invalid_literal": r"invalid literal for.*? with base",
            "dict_access": r"\[.*?\]",
            "dict_key": r"\[['\"](.*?)['\"]\]",
            "NoneType": r"NoneType",
            "missing_argument": r"missing.*?argument",
            "expected_got": r"expected.*?got",
        }

    def extract_error_type(self, error_data: Dict[str, Any]) -> str:
        """
        Extract the exception type from error data.

        Args:
            error_data: Error data dictionary

        Returns:
            Exception type as string
        """
        # Try different fields where exception type might be found
        exception_type = error_data.get("exception_type", "")
        if not exception_type and "error_details" in error_data:
            exception_type = error_data["error_details"].get("exception_type", "")
        if not exception_type:
            # Try to extract from message
            message = error_data.get("message", "")
            if ":" in message:
                exception_type = message.split(":", 1)[0].strip()

        return exception_type

    def extract_message(self, error_data: Dict[str, Any]) -> str:
        """
        Extract the error message from error data.

        Args:
            error_data: Error data dictionary

        Returns:
            Error message as string
        """
        message = error_data.get("message", "")
        if not message and "error_details" in error_data:
            message = error_data["error_details"].get("message", "")
        return message

    def extract_traceback_info(
        self, error_data: Dict[str, Any]
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Extract key information from traceback.

        Args:
            error_data: Error data dictionary

        Returns:
            Tuple of (file_paths, function_names, line_contexts)
        """
        traceback = error_data.get("traceback", [])
        if isinstance(traceback, str):
            traceback = traceback.split("\n")

        detailed_frames = []
        if (
            "error_details" in error_data
            and "detailed_frames" in error_data["error_details"]
        ):
            detailed_frames = error_data["error_details"]["detailed_frames"]

        file_paths = []
        function_names = []
        line_contexts = []

        # Extract from formatted traceback
        for line in traceback:
            if 'File "' in line:
                match = re.search(r'File "(.*?)", line (\d+), in (.*)', line)
                if match:
                    file_path, _, func_name = match.groups()
                    file_paths.append(os.path.basename(file_path))
                    function_names.append(func_name)

        # Extract from detailed frames if available
        for frame in detailed_frames:
            if "file" in frame:
                file_paths.append(os.path.basename(frame["file"]))
            if "function" in frame:
                function_names.append(frame["function"])
            if "line_context" in frame:
                line_contexts.append(frame["line_context"])
            elif "line" in frame and "code" in frame:
                line_contexts.append(f"{frame['line']}: {frame['code']}")

        return file_paths, function_names, line_contexts

    def extract_features(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract all features from error data.

        Args:
            error_data: Error data dictionary

        Returns:
            Dictionary of extracted features
        """
        exception_type = self.extract_error_type(error_data)
        message = self.extract_message(error_data)
        file_paths, function_names, line_contexts = self.extract_traceback_info(
            error_data
        )

        # Check pattern matches in message
        pattern_matches = {
            pattern_name: 1 if re.search(pattern, message, re.IGNORECASE) else 0
            for pattern_name, pattern in self.message_patterns.items()
        }

        # Get local variables if available
        local_vars = []
        if (
            "error_details" in error_data
            and "detailed_frames" in error_data["error_details"]
        ):
            for frame in error_data["error_details"]["detailed_frames"]:
                if "locals" in frame:
                    local_vars.extend(list(frame["locals"].keys()))

        return {
            "exception_type": exception_type,
            "message": message,
            "file_paths": file_paths,
            "function_names": function_names,
            "line_contexts": line_contexts,
            "pattern_matches": pattern_matches,
            "has_local_vars": len(local_vars) > 0,
            "local_var_count": len(local_vars),
            "local_vars": local_vars,
        }

    def prepare_text_features(self, error_data: Dict[str, Any]) -> str:
        """
        Prepare text features for vectorization.

        Args:
            error_data: Error data or extracted features

        Returns:
            Concatenated text for vectorization
        """
        # Check if we're working with raw error data or extracted features
        if "exception_type" not in error_data or isinstance(
            error_data["exception_type"], dict
        ):
            features = self.extract_features(error_data)
        else:
            features = error_data

        # Combine all text features into one string
        texts = [
            features["exception_type"],
            features["message"],
            " ".join(features["file_paths"]),
            " ".join(features["function_names"]),
            " ".join(features["line_contexts"]),
        ]

        return " ".join(filter(None, texts))


class ErrorClassifierModel:
    """Machine learning model for error classification."""

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the error classifier model.

        Args:
            model_path: Path to saved model file
        """
        self.feature_extractor = ErrorClassificationFeatures()
        self.model_path = model_path
        self.pipeline = None
        self.classes = []
        self.trained = False

    def build_pipeline(self):
        """
        Build the classification pipeline.

        Returns:
            sklearn Pipeline for classification
        """
        # Create a pipeline with TF-IDF vectorization and Random Forest classification
        return Pipeline(
            [
                (
                    "vectorizer",
                    TfidfVectorizer(
                        max_features=5000, ngram_range=(1, 2), stop_words="english"
                    ),
                ),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=100,
                        max_depth=None,
                        min_samples_split=2,
                        random_state=42,
                    ),
                ),
            ]
        )

    def load(self) -> bool:
        """
        Load a trained model from disk.

        Returns:
            True if successful, False otherwise
        """
        if not self.model_path:
            print("No model path specified.")
            return False

        try:
            model_file = Path(self.model_path)
            if not model_file.exists():
                print(f"Model file not found: {self.model_path}")
                return False

            model_data = secure_pickle_load(model_file)
            self.pipeline = model_data["pipeline"]
            self.classes = model_data["classes"]
            self.trained = True

            print(f"Loaded model from {self.model_path}")
            print(f"Model can classify {len(self.classes)} error types")
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False

    def save(self, path: Optional[str] = None) -> bool:
        """
        Save the trained model to disk.

        Args:
            path: Path to save the model to, defaults to self.model_path

        Returns:
            True if successful, False otherwise
        """
        if not self.trained:
            print("Cannot save untrained model.")
            return False

        save_path = path or self.model_path
        if not save_path:
            print("No path specified for saving model.")
            return False

        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            # Save model data
            model_data = {"pipeline": self.pipeline, "classes": self.classes}

            with open(save_path, "wb") as f:
                pickle.dump(model_data, f)

            print(f"Model saved to {save_path}")
            return True
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return False

    def train(self, errors: List[Dict[str, Any]], labels: List[str]) -> Dict[str, Any]:
        """
        Train the error classification model.

        Args:
            errors: List of error data dictionaries
            labels: List of error category labels

        Returns:
            Dictionary with training results and metrics
        """
        if not errors or not labels or len(errors) != len(labels):
            raise ValueError(
                "Invalid training data: errors and labels must be non-empty lists of the same length"
            )

        # Prepare training data
        X_text = [
            self.feature_extractor.prepare_text_features(error) for error in errors
        ]
        y = labels

        # Create train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_text, y, test_size=0.2, random_state=42
        )

        # Build and train pipeline
        self.pipeline = self.build_pipeline()
        self.pipeline.fit(X_train, y_train)

        # Get unique classes
        self.classes = list(self.pipeline.classes_)

        # Evaluate on test set
        y_pred = self.pipeline.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)

        self.trained = True

        return {
            "samples": len(errors),
            "classes": len(self.classes),
            "accuracy": report["accuracy"],
            "class_report": report,
            "feature_importance": (
                self._get_feature_importance()
                if hasattr(self, "_get_feature_importance")
                else None
            ),
        }

    def predict(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify an error using the trained model.

        Args:
            error_data: Error data dictionary

        Returns:
            Dictionary with prediction results
        """
        if not self.trained:
            return {"error": "Model not trained", "success": False}

        # Extract text features
        X_text = self.feature_extractor.prepare_text_features(error_data)

        # Get prediction and probabilities
        error_type = self.pipeline.predict([X_text])[0]
        probabilities = self.pipeline.predict_proba([X_text])[0]

        # Get confidence score (probability of predicted class)
        confidence = max(probabilities)

        # Get top alternative predictions
        class_probs = list(zip(self.classes, probabilities))
        class_probs.sort(key=lambda x: x[1], reverse=True)
        alternatives = [
            {"class": cls, "probability": prob}
            for cls, prob in class_probs[1:4]
            if prob > 0.05  # Get next 3 highest probabilities
        ]

        return {
            "error_type": error_type,
            "confidence": float(confidence),
            "alternatives": alternatives,
            "success": True,
        }

    def batch_predict(
        self, error_data_list: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Classify multiple errors using the trained model.

        Args:
            error_data_list: List of error data dictionaries

        Returns:
            List of prediction results
        """
        return [self.predict(error_data) for error_data in error_data_list]


def get_sample_data() -> List[Dict[str, Any]]:
    """
    Get sample error data for testing.

    Returns:
        List of sample error data dictionaries
    """
    return [
        {
            "timestamp": "2023-01-01T12:00:00",
            "service": "example_service",
            "level": "ERROR",
            "message": "KeyError: 'todo_id'",
            "exception_type": "KeyError",
            "traceback": [
                "Traceback (most recent call last):",
                "  File '/app/services/example_service/app.py', line 42, in get_todo",
                "    todo = todo_db[todo_id]",
                "KeyError: 'todo_id'",
            ],
            "error_details": {
                "exception_type": "KeyError",
                "message": "'todo_id'",
                "detailed_frames": [
                    {
                        "file": "/app/services/example_service/app.py",
                        "line": 42,
                        "function": "get_todo",
                        "locals": {"todo_db": {"1": {"title": "Example"}}},
                    }
                ],
            },
        },
        {
            "timestamp": "2023-01-02T13:30:00",
            "service": "example_service",
            "level": "ERROR",
            "message": "TypeError: 'NoneType' object is not subscriptable",
            "exception_type": "TypeError",
            "traceback": [
                "Traceback (most recent call last):",
                "  File '/app/services/example_service/app.py', line 55, in process_data",
                "    result = data['value']",
                "TypeError: 'NoneType' object is not subscriptable",
            ],
            "error_details": {
                "exception_type": "TypeError",
                "message": "'NoneType' object is not subscriptable",
                "detailed_frames": [
                    {
                        "file": "/app/services/example_service/app.py",
                        "line": 55,
                        "function": "process_data",
                        "locals": {"data": None},
                    }
                ],
            },
        },
        {
            "timestamp": "2023-01-03T09:15:00",
            "service": "example_service",
            "level": "ERROR",
            "message": "IndexError: list index out of range",
            "exception_type": "IndexError",
            "traceback": [
                "Traceback (most recent call last):",
                "  File '/app/services/example_service/app.py', line 68, in get_latest",
                "    return items[5]",
                "IndexError: list index out of range",
            ],
            "error_details": {
                "exception_type": "IndexError",
                "message": "list index out of range",
                "detailed_frames": [
                    {
                        "file": "/app/services/example_service/app.py",
                        "line": 68,
                        "function": "get_latest",
                        "locals": {"items": [1, 2, 3]},
                    }
                ],
            },
        },
        {
            "timestamp": "2023-01-04T14:20:00",
            "service": "example_service",
            "level": "ERROR",
            "message": "AttributeError: 'User' object has no attribute 'email'",
            "exception_type": "AttributeError",
            "traceback": [
                "Traceback (most recent call last):",
                "  File '/app/services/example_service/app.py', line 81, in send_notification",
                "    send_email(user.email, 'Notification')",
                "AttributeError: 'User' object has no attribute 'email'",
            ],
            "error_details": {
                "exception_type": "AttributeError",
                "message": "'User' object has no attribute 'email'",
                "detailed_frames": [
                    {
                        "file": "/app/services/example_service/app.py",
                        "line": 81,
                        "function": "send_notification",
                        "locals": {"user": {"name": "Test User"}},
                    }
                ],
            },
        },
        {
            "timestamp": "2023-01-05T16:45:00",
            "service": "example_service",
            "level": "ERROR",
            "message": "ImportError: No module named 'missing_module'",
            "exception_type": "ImportError",
            "traceback": [
                "Traceback (most recent call last):",
                "  File '/app/services/example_service/app.py', line 94, in setup",
                "    import missing_module",
                "ImportError: No module named 'missing_module'",
            ],
            "error_details": {
                "exception_type": "ImportError",
                "message": "No module named 'missing_module'",
                "detailed_frames": [
                    {
                        "file": "/app/services/example_service/app.py",
                        "line": 94,
                        "function": "setup",
                        "locals": {},
                    }
                ],
            },
        },
    ]


def create_test_model():
    """
    Create and train a test model with sample data.

    Returns:
        Trained ErrorClassifierModel
    """
    # Get sample data
    sample_data = get_sample_data()

    # Create labels (in a real scenario, these would come from the training dataset)
    labels = [
        "missing_dictionary_key",
        "none_value_error",
        "index_out_of_bounds",
        "missing_attribute",
        "missing_module",
    ]

    # Create and train the model
    model = ErrorClassifierModel()
    results = model.train(sample_data, labels)

    print(f"Trained model with {results['samples']} samples")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Classes: {results['classes']}")

    return model


if __name__ == "__main__":
    # Test the model
    model = create_test_model()

    # Save the model
    model_path = os.path.join(os.path.dirname(__file__), "error_classifier.pkl")
    model.save(model_path)

    # Test prediction
    test_error = {
        "timestamp": "2023-01-06T10:00:00",
        "service": "example_service",
        "level": "ERROR",
        "message": "KeyError: 'user_id'",
        "exception_type": "KeyError",
        "traceback": [
            "Traceback (most recent call last):",
            "  File '/app/services/example_service/app.py', line 120, in get_user",
            "    return users[user_id]",
            "KeyError: 'user_id'",
        ],
        "error_details": {
            "exception_type": "KeyError",
            "message": "'user_id'",
            "detailed_frames": [
                {
                    "file": "/app/services/example_service/app.py",
                    "line": 120,
                    "function": "get_user",
                    "locals": {"users": {}},
                }
            ],
        },
    }

    prediction = model.predict(test_error)
    print("\nPrediction for test error:")
    print(f"Error type: {prediction['error_type']}")
    print(f"Confidence: {prediction['confidence']:.4f}")
    print("Alternatives:")
    for alt in prediction["alternatives"]:
        print(f"  - {alt['class']}: {alt['probability']:.4f}")
