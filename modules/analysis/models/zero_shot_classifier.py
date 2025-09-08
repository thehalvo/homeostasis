"""
Zero-shot learning module for classifying new and unseen error types.

This module implements zero-shot classification capabilities that can handle
error types not seen during training, using semantic embeddings and
similarity-based approaches.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ErrorPrototype:
    """Prototype representation of an error class."""

    name: str
    description: str
    examples: List[str]
    semantic_embedding: Optional[np.ndarray] = None
    syntactic_patterns: List[str] = None
    typical_causes: List[str] = None
    fix_strategies: List[str] = None

    def __post_init__(self):
        if self.syntactic_patterns is None:
            self.syntactic_patterns = []
        if self.typical_causes is None:
            self.typical_causes = []
        if self.fix_strategies is None:
            self.fix_strategies = []


class SemanticErrorEmbedder:
    """Create semantic embeddings for errors."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the semantic embedder.

        Args:
            model_name: Name of the sentence transformer model
        """
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def embed_error(self, error_data: Dict[str, Any]) -> np.ndarray:
        """
        Create semantic embedding for an error.

        Args:
            error_data: Error data dictionary

        Returns:
            Semantic embedding vector
        """
        # Combine relevant error information
        text_parts = []

        if "exception_type" in error_data:
            text_parts.append(f"Error type: {error_data['exception_type']}")

        if "message" in error_data:
            text_parts.append(f"Message: {error_data['message']}")

        # Add code context if available
        if ("error_details" in error_data and
                "detailed_frames" in error_data["error_details"]):
            frames = error_data["error_details"]["detailed_frames"]
            if frames and "code" in frames[-1]:
                text_parts.append(f"Code: {frames[-1]['code']}")

        # Add traceback summary
        if "traceback" in error_data:
            tb = error_data["traceback"]
            if isinstance(tb, list) and len(tb) > 0:
                text_parts.append(f"Context: {tb[-1]}")

        # Create embedding
        text = " ".join(text_parts)
        embedding = self.model.encode(text)

        return embedding

    def embed_prototype(self, prototype: ErrorPrototype) -> np.ndarray:
        """
        Create semantic embedding for an error prototype.

        Args:
            prototype: Error prototype

        Returns:
            Semantic embedding vector
        """
        # Combine prototype information
        text_parts = [
            f"Error class: {prototype.name}",
            f"Description: {prototype.description}",
        ]

        # Add examples
        if prototype.examples:
            text_parts.append(f"Examples: {'; '.join(prototype.examples[:3])}")

        # Add typical causes
        if prototype.typical_causes:
            text_parts.append(f"Causes: {'; '.join(prototype.typical_causes[:3])}")

        # Create embedding
        text = " ".join(text_parts)
        embedding = self.model.encode(text)

        return embedding


class ZeroShotErrorClassifier:
    """Zero-shot classifier for unseen error types."""

    def __init__(self):
        """Initialize the zero-shot classifier."""
        self.embedder = SemanticErrorEmbedder()
        self.prototypes = self._initialize_prototypes()
        self.prototype_index = None
        self.nlp_pipeline = pipeline(
            "zero-shot-classification", model="facebook/bart-large-mnli"
        )

        # Build prototype index
        self._build_prototype_index()

    def _initialize_prototypes(self) -> Dict[str, ErrorPrototype]:
        """Initialize error prototypes with descriptions."""
        prototypes = {
            # Common error types
            "configuration_error": ErrorPrototype(
                name="configuration_error",
                description="Error in application configuration or settings",
                examples=[
                    "Invalid configuration value",
                    "Missing required configuration",
                    "Configuration type mismatch",
                ],
                typical_causes=[
                    "Missing config file",
                    "Invalid config format",
                    "Environment variable not set",
                ],
                fix_strategies=[
                    "validate_config",
                    "provide_defaults",
                    "check_env_vars",
                ],
            ),
            "authentication_error": ErrorPrototype(
                name="authentication_error",
                description="Authentication or authorization failure",
                examples=[
                    "Invalid credentials",
                    "Token expired",
                    "Insufficient permissions",
                ],
                typical_causes=[
                    "Wrong username/password",
                    "Expired token",
                    "Missing permissions",
                ],
                fix_strategies=[
                    "refresh_token",
                    "check_credentials",
                    "verify_permissions",
                ],
            ),
            "data_validation_error": ErrorPrototype(
                name="data_validation_error",
                description="Data does not meet validation requirements",
                examples=[
                    "Invalid email format",
                    "Number out of range",
                    "Required field missing",
                ],
                typical_causes=[
                    "User input error",
                    "Data corruption",
                    "Schema mismatch",
                ],
                fix_strategies=["add_validation", "sanitize_input", "provide_feedback"],
            ),
            "concurrency_error": ErrorPrototype(
                name="concurrency_error",
                description="Error due to concurrent access or race conditions",
                examples=["Deadlock detected", "Race condition", "Resource contention"],
                typical_causes=[
                    "Simultaneous access",
                    "Missing locks",
                    "Incorrect synchronization",
                ],
                fix_strategies=[
                    "add_locking",
                    "use_atomic_operations",
                    "implement_queue",
                ],
            ),
            "serialization_error": ErrorPrototype(
                name="serialization_error",
                description="Error during data serialization or deserialization",
                examples=["JSON decode error", "Pickle error", "XML parsing error"],
                typical_causes=[
                    "Malformed data",
                    "Incompatible versions",
                    "Encoding issues",
                ],
                fix_strategies=[
                    "validate_format",
                    "handle_versions",
                    "specify_encoding",
                ],
            ),
            "rate_limit_error": ErrorPrototype(
                name="rate_limit_error",
                description="API rate limit exceeded",
                examples=[
                    "Too many requests",
                    "Rate limit exceeded",
                    "Quota exhausted",
                ],
                typical_causes=[
                    "Too frequent requests",
                    "Burst traffic",
                    "Insufficient quota",
                ],
                fix_strategies=["implement_backoff", "add_caching", "use_rate_limiter"],
            ),
            "dependency_error": ErrorPrototype(
                name="dependency_error",
                description="External dependency failure or incompatibility",
                examples=[
                    "Service unavailable",
                    "Version conflict",
                    "Missing dependency",
                ],
                typical_causes=["Service down", "Version mismatch", "Network issues"],
                fix_strategies=["add_fallback", "check_versions", "implement_retry"],
            ),
            "state_error": ErrorPrototype(
                name="state_error",
                description="Invalid state or state transition",
                examples=[
                    "Invalid state transition",
                    "State machine error",
                    "Workflow violation",
                ],
                typical_causes=[
                    "Incorrect flow",
                    "Missing state check",
                    "Concurrent modification",
                ],
                fix_strategies=["validate_state", "add_guards", "implement_fsm"],
            ),
        }

        # Add semantic embeddings to prototypes
        for name, prototype in prototypes.items():
            prototype.semantic_embedding = self.embedder.embed_prototype(prototype)

        return prototypes

    def _build_prototype_index(self):
        """Build FAISS index for efficient similarity search."""
        # Collect prototype embeddings
        embeddings = []
        self.prototype_names = []

        for name, prototype in self.prototypes.items():
            embeddings.append(prototype.semantic_embedding)
            self.prototype_names.append(name)

        embeddings = np.array(embeddings).astype("float32")

        # Build FAISS index
        self.prototype_index = faiss.IndexFlatIP(self.embedder.embedding_dim)
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.prototype_index.add(embeddings)

    def classify_zero_shot(
        self, error_data: Dict[str, Any], k: int = 3
    ) -> Dict[str, Any]:
        """
        Classify an error using zero-shot learning.

        Args:
            error_data: Error data dictionary
            k: Number of nearest prototypes to consider

        Returns:
            Classification results
        """
        # Method 1: Semantic similarity
        semantic_results = self._classify_by_similarity(error_data, k)

        # Method 2: NLI-based classification
        nli_results = self._classify_by_nli(error_data)

        # Method 3: Pattern matching
        pattern_results = self._classify_by_patterns(error_data)

        # Combine results
        combined_results = self._combine_results(
            semantic_results, nli_results, pattern_results
        )

        # Add explanations
        combined_results["explanation"] = self._generate_explanation(
            error_data, combined_results
        )

        return combined_results

    def _classify_by_similarity(
        self, error_data: Dict[str, Any], k: int
    ) -> Dict[str, Any]:
        """Classify using semantic similarity to prototypes."""
        # Get error embedding
        error_embedding = self.embedder.embed_error(error_data)
        error_embedding = error_embedding.reshape(1, -1).astype("float32")
        faiss.normalize_L2(error_embedding)

        # Search for nearest prototypes
        distances, indices = self.prototype_index.search(error_embedding, k)

        # Convert to similarity scores (cosine similarity)
        similarities = distances[0]  # Already normalized, so this is cosine similarity

        # Create results
        results = []
        for i, (idx, sim) in enumerate(zip(indices[0], similarities)):
            prototype_name = self.prototype_names[idx]
            prototype = self.prototypes[prototype_name]

            results.append(
                {
                    "class": prototype_name,
                    "score": float(sim),
                    "description": prototype.description,
                    "fix_strategies": prototype.fix_strategies,
                }
            )

        return {"method": "semantic_similarity", "predictions": results}

    def _classify_by_nli(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """Classify using Natural Language Inference."""
        # Prepare error text
        error_text = self._error_to_text(error_data)

        # Prepare candidate labels and hypotheses
        candidate_labels = []
        hypotheses = []

        for name, prototype in self.prototypes.items():
            candidate_labels.append(name)
            hypotheses.append(prototype.description)

        # Run zero-shot classification
        result = self.nlp_pipeline(
            error_text,
            candidate_labels=candidate_labels,
            hypothesis_template="This is an error about {}",
            multi_label=False,
        )

        # Format results
        predictions = []
        for label, score in zip(result["labels"][:3], result["scores"][:3]):
            prototype = self.prototypes[label]
            predictions.append(
                {
                    "class": label,
                    "score": float(score),
                    "description": prototype.description,
                    "fix_strategies": prototype.fix_strategies,
                }
            )

        return {"method": "nli_classification", "predictions": predictions}

    def _classify_by_patterns(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """Classify using pattern matching."""
        error_text = self._error_to_text(error_data).lower()

        # Score each prototype based on pattern matches
        scores = {}

        for name, prototype in self.prototypes.items():
            score = 0.0

            # Check examples
            for example in prototype.examples:
                if example.lower() in error_text:
                    score += 0.3

            # Check typical causes
            for cause in prototype.typical_causes:
                if any(word in error_text for word in cause.lower().split()):
                    score += 0.2

            # Check name components
            name_parts = name.replace("_", " ").split()
            for part in name_parts:
                if part in error_text:
                    score += 0.1

            scores[name] = min(score, 1.0)  # Cap at 1.0

        # Sort by score
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Format results
        predictions = []
        for name, score in sorted_scores[:3]:
            if score > 0:
                prototype = self.prototypes[name]
                predictions.append(
                    {
                        "class": name,
                        "score": score,
                        "description": prototype.description,
                        "fix_strategies": prototype.fix_strategies,
                    }
                )

        return {"method": "pattern_matching", "predictions": predictions}

    def _error_to_text(self, error_data: Dict[str, Any]) -> str:
        """Convert error data to text representation."""
        parts = []

        if "exception_type" in error_data:
            parts.append(f"{error_data['exception_type']}")

        if "message" in error_data:
            parts.append(error_data["message"])

        if ("error_details" in error_data and
                "detailed_frames" in error_data["error_details"]):
            frames = error_data["error_details"]["detailed_frames"]
            if frames and "code" in frames[-1]:
                parts.append(f"in code: {frames[-1]['code']}")

        return " ".join(parts)

    def _combine_results(
        self, semantic: Dict[str, Any], nli: Dict[str, Any], pattern: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Combine results from different methods."""
        # Aggregate scores by class
        class_scores = defaultdict(list)

        # Weight for each method
        weights = {
            "semantic_similarity": 0.4,
            "nli_classification": 0.4,
            "pattern_matching": 0.2,
        }

        # Collect scores
        for method_result, method_name in [
            (semantic, "semantic_similarity"),
            (nli, "nli_classification"),
            (pattern, "pattern_matching"),
        ]:
            weight = weights[method_name]
            for pred in method_result["predictions"]:
                class_scores[pred["class"]].append(pred["score"] * weight)

        # Calculate final scores
        final_predictions = []
        for class_name, scores in class_scores.items():
            avg_score = sum(scores) / len(scores)
            prototype = self.prototypes[class_name]

            final_predictions.append(
                {
                    "class": class_name,
                    "score": avg_score,
                    "description": prototype.description,
                    "fix_strategies": prototype.fix_strategies,
                    "confidence": self._calculate_confidence(scores),
                }
            )

        # Sort by score
        final_predictions.sort(key=lambda x: x["score"], reverse=True)

        return {
            "predictions": final_predictions[:3],
            "methods_used": {
                "semantic": (
                    semantic["predictions"][:1] if semantic["predictions"] else []
                ),
                "nli": nli["predictions"][:1] if nli["predictions"] else [],
                "pattern": pattern["predictions"][:1] if pattern["predictions"] else [],
            },
        }

    def _calculate_confidence(self, scores: List[float]) -> float:
        """Calculate confidence based on score agreement."""
        if len(scores) < 2:
            return scores[0] if scores else 0.0

        # High confidence if methods agree
        std_dev = np.std(scores)
        mean_score = np.mean(scores)

        # Lower confidence if high variance
        confidence = mean_score * (1 - std_dev)

        return float(confidence)

    def _generate_explanation(
        self, error_data: Dict[str, Any], results: Dict[str, Any]
    ) -> str:
        """Generate explanation for the classification."""
        if not results["predictions"]:
            return "Unable to classify error with confidence"

        top_prediction = results["predictions"][0]
        prototype = self.prototypes[top_prediction["class"]]

        explanation_parts = [
            f"This appears to be a {top_prediction['class'].replace('_', ' ')}.",
            prototype.description,
            f"Confidence: {top_prediction['confidence']:.2f}",
        ]

        # Add typical causes if relevant
        if prototype.typical_causes:
            explanation_parts.append(
                f"Common causes: {', '.join(prototype.typical_causes[:2])}"
            )

        return " ".join(explanation_parts)

    def add_new_prototype(self, prototype: ErrorPrototype):
        """
        Add a new error prototype dynamically.

        Args:
            prototype: New error prototype
        """
        # Add embedding
        prototype.semantic_embedding = self.embedder.embed_prototype(prototype)

        # Add to prototypes
        self.prototypes[prototype.name] = prototype

        # Rebuild index
        self._build_prototype_index()

        logger.info(f"Added new prototype: {prototype.name}")

    def learn_from_feedback(
        self, error_data: Dict[str, Any], correct_class: str, predicted_class: str
    ):
        """
        Learn from user feedback to improve classification.

        Args:
            error_data: The error that was classified
            correct_class: The correct classification
            predicted_class: What the system predicted
        """
        # This could be used to:
        # 1. Fine-tune embeddings
        # 2. Adjust prototype descriptions
        # 3. Add new examples to prototypes
        # 4. Create new prototypes if needed

        logger.info(
            f"Learning from feedback: predicted {predicted_class}, "
            f"correct was {correct_class}"
        )

        # Add example to correct prototype if it exists
        if correct_class in self.prototypes:
            error_text = self._error_to_text(error_data)
            self.prototypes[correct_class].examples.append(error_text[:100])


class AdaptiveZeroShotClassifier:
    """Adaptive zero-shot classifier that learns from usage."""

    def __init__(self):
        """Initialize adaptive classifier."""
        self.base_classifier = ZeroShotErrorClassifier()
        self.usage_history = []
        self.performance_metrics = defaultdict(lambda: {"correct": 0, "total": 0})

    def classify(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify error with adaptation.

        Args:
            error_data: Error data

        Returns:
            Classification results
        """
        # Get base classification
        results = self.base_classifier.classify_zero_shot(error_data)

        # Add usage tracking
        self.usage_history.append(
            {"error": error_data, "results": results, "timestamp": np.datetime64("now")}
        )

        # Adapt if we have enough history
        if len(self.usage_history) > 100:
            self._adapt_classifier()

        return results

    def _adapt_classifier(self):
        """Adapt classifier based on usage patterns."""
        # Analyze common misclassifications
        # Update prototype embeddings
        # Adjust classification thresholds
        pass

    def provide_feedback(self, error_id: int, correct_class: str):
        """Provide feedback on classification."""
        if 0 <= error_id < len(self.usage_history):
            entry = self.usage_history[error_id]
            predicted = entry["results"]["predictions"][0]["class"]

            # Update metrics
            self.performance_metrics[predicted]["total"] += 1
            if predicted == correct_class:
                self.performance_metrics[predicted]["correct"] += 1

            # Learn from feedback
            self.base_classifier.learn_from_feedback(
                entry["error"], correct_class, predicted
            )


def demonstrate_zero_shot_classification():
    """Demonstrate zero-shot classification capabilities."""
    classifier = ZeroShotErrorClassifier()

    # Test with various unseen errors
    test_errors = [
        {
            "exception_type": "CustomConfigError",
            "message": "Invalid configuration: database_url must be a valid URL",
            "error_details": {
                "detailed_frames": [{"code": 'db = Database(config["database_url"])'}]
            },
        },
        {
            "exception_type": "TokenExpiredError",
            "message": "Authentication token has expired",
            "error_details": {
                "detailed_frames": [{"code": "user = authenticate(token)"}]
            },
        },
        {
            "exception_type": "ValidationError",
            "message": "Email address is not valid: missing @ symbol",
            "error_details": {
                "detailed_frames": [{"code": 'validate_email(user_input["email"])'}]
            },
        },
        {
            "exception_type": "ConcurrentModificationError",
            "message": "Resource was modified by another process",
            "error_details": {"detailed_frames": [{"code": "resource.save()"}]},
        },
    ]

    for i, error in enumerate(test_errors):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Classifying unseen error {i + 1}/{len(test_errors)}")
        logger.info(f"Error: {error['exception_type']}: {error['message']}")

        # Classify
        results = classifier.classify_zero_shot(error)

        # Display results
        logger.info("\nClassification Results:")
        for j, pred in enumerate(results["predictions"]):
            logger.info(
                f"{j + 1}. {pred['class']} (score: {pred['score']:.3f}, "
                f"confidence: {pred['confidence']:.3f})"
            )
            logger.info(f"   Description: {pred['description']}")
            logger.info(f"   Suggested fixes: {', '.join(pred['fix_strategies'])}")

        logger.info(f"\nExplanation: {results['explanation']}")

        # Show method contributions
        logger.info("\nMethod contributions:")
        for method, preds in results["methods_used"].items():
            if preds:
                logger.info(
                    f"  {method}: {preds[0]['class']} ({preds[0]['score']:.3f})"
                )


if __name__ == "__main__":
    demonstrate_zero_shot_classification()
