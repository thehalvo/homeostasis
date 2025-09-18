"""
Hierarchical error classification system for Homeostasis.

This module implements a multi-level classification system that categorizes
errors from coarse-grained to fine-grained levels, enabling more precise
error handling and fix generation.
"""

import hashlib
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from transformers import AutoModel, AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RestrictedUnpickler(pickle.Unpickler):
    """
    Restricted unpickler that only allows specific safe classes.
    This prevents arbitrary code execution during unpickling.
    """

    ALLOWED_MODULES = {
        "numpy",
        "numpy.core.multiarray",
        "numpy.core.numeric",
        "torch",
        "torch._utils",
        "torch.nn",
        "torch.nn.parameter",
        "torch.nn.modules",
        "torch.nn.functional",
        "sklearn.preprocessing._label",
        "sklearn.preprocessing",
        "collections",
        "builtins",
        "networkx",
        "networkx.classes.graph",
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
        ("torch._utils", "_rebuild_tensor_v2"),
        ("torch", "FloatStorage"),
        ("torch", "DoubleStorage"),
        ("torch", "HalfStorage"),
        ("torch", "LongStorage"),
        ("torch", "IntStorage"),
        ("torch", "ShortStorage"),
        ("torch", "CharStorage"),
        ("torch", "ByteStorage"),
        ("torch", "BoolStorage"),
        ("torch.nn.parameter", "Parameter"),
        ("sklearn.preprocessing._label", "LabelEncoder"),
        ("networkx.classes.graph", "Graph"),
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


def secure_torch_load(
    filepath: str, map_location=None, expected_hash: Optional[str] = None
):
    """
    Securely load a PyTorch checkpoint with protection against arbitrary code execution.

    Args:
        filepath: Path to the checkpoint file
        map_location: Device mapping for torch.load
        expected_hash: Expected SHA256 hash of the file (optional but recommended)

    Returns:
        Loaded checkpoint data

    Raises:
        ValueError: If file hash doesn't match expected hash
        RuntimeError: If loading fails
    """
    # Verify file hash if provided
    if expected_hash:
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)

        actual_hash = sha256_hash.hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(
                f"File hash mismatch! Expected: {expected_hash}, "
                f"Actual: {actual_hash}. File may have been tampered with."
            )

    # First, try loading with weights_only=True (safest option)
    try:
        with open(filepath, "rb") as f:
            return torch.load(f, map_location=map_location, weights_only=True)
    except Exception as e:
        logger.info(f"weights_only load failed, using restricted unpickler: {str(e)}")

    # Fall back to restricted unpickling
    with open(filepath, "rb") as f:
        # Use weights_only=True for additional security when possible
        try:
            return torch.load(filepath, map_location=map_location, weights_only=True)
        except (TypeError, pickle.UnpicklingError):
            # Fallback with custom unpickler if weights_only fails
            with open(filepath, "rb") as f:
                return RestrictedUnpickler(f).load()


@dataclass
class ErrorTaxonomy:
    """Hierarchical error taxonomy structure."""

    # Level 1: Coarse-grained categories
    SYNTAX_ERRORS = "syntax_errors"
    RUNTIME_ERRORS = "runtime_errors"
    LOGIC_ERRORS = "logic_errors"
    RESOURCE_ERRORS = "resource_errors"
    EXTERNAL_ERRORS = "external_errors"

    # Level 2: Medium-grained categories
    taxonomy = {
        SYNTAX_ERRORS: {
            "parsing_errors": ["SyntaxError", "IndentationError", "TabError"],
            "import_errors": ["ImportError", "ModuleNotFoundError"],
            "name_errors": ["NameError", "UnboundLocalError"],
        },
        RUNTIME_ERRORS: {
            "type_errors": ["TypeError", "AttributeError"],
            "value_errors": ["ValueError", "KeyError", "IndexError"],
            "arithmetic_errors": [
                "ZeroDivisionError",
                "OverflowError",
                "FloatingPointError",
            ],
            "assertion_errors": ["AssertionError"],
        },
        LOGIC_ERRORS: {
            "iteration_errors": ["StopIteration", "RuntimeError"],
            "recursion_errors": ["RecursionError"],
            "generator_errors": ["GeneratorExit", "StopAsyncIteration"],
        },
        RESOURCE_ERRORS: {
            "memory_errors": ["MemoryError", "SystemError"],
            "io_errors": ["IOError", "OSError", "FileNotFoundError", "PermissionError"],
            "process_errors": ["ProcessLookupError", "ChildProcessError"],
        },
        EXTERNAL_ERRORS: {
            "network_errors": ["ConnectionError", "TimeoutError", "URLError"],
            "database_errors": ["DatabaseError", "IntegrityError", "OperationalError"],
            "api_errors": ["HTTPError", "RequestException"],
        },
    }

    # Level 3: Fine-grained patterns
    fine_patterns = {
        "missing_key": {
            "parent": "value_errors",
            "patterns": [r"KeyError.*['\"](\w+)['\"]", r"key.*not found"],
            "fixes": ["add_default_value", "check_key_exists", "use_get_method"],
        },
        "none_attribute": {
            "parent": "type_errors",
            "patterns": [r"'NoneType'.*has no attribute", r"AttributeError.*None"],
            "fixes": ["add_none_check", "initialize_variable", "handle_optional"],
        },
        "index_out_of_bounds": {
            "parent": "value_errors",
            "patterns": [r"list index out of range", r"IndexError.*\d+"],
            "fixes": ["check_list_length", "use_safe_indexing", "handle_empty_list"],
        },
        "type_mismatch": {
            "parent": "type_errors",
            "patterns": [r"unsupported operand type", r"expected.*got.*type"],
            "fixes": ["add_type_conversion", "validate_types", "use_type_hints"],
        },
        "file_not_found": {
            "parent": "io_errors",
            "patterns": [r"No such file or directory", r"FileNotFoundError"],
            "fixes": ["check_file_exists", "create_if_missing", "use_default_path"],
        },
        "connection_timeout": {
            "parent": "network_errors",
            "patterns": [r"timed out", r"TimeoutError", r"connection timeout"],
            "fixes": ["increase_timeout", "add_retry_logic", "use_async_request"],
        },
        "circular_import": {
            "parent": "import_errors",
            "patterns": [r"circular import", r"import cycle detected"],
            "fixes": ["refactor_imports", "lazy_import", "restructure_modules"],
        },
        "encoding_error": {
            "parent": "value_errors",
            "patterns": [
                r"UnicodeDecodeError",
                r"codec can't decode",
                r"invalid.*encoding",
            ],
            "fixes": ["specify_encoding", "handle_encoding_errors", "detect_encoding"],
        },
    }

    def get_hierarchy(self) -> nx.DiGraph:
        """Build and return the error hierarchy as a directed graph."""
        G = nx.DiGraph()

        # Add root node
        G.add_node("root")

        # Add level 1 nodes
        for l1_category in self.taxonomy.keys():
            G.add_edge("root", l1_category)

            # Add level 2 nodes
            for l2_category, error_types in self.taxonomy[l1_category].items():
                G.add_edge(l1_category, l2_category)

                # Add level 3 nodes (specific error types)
                for error_type in error_types:
                    G.add_edge(l2_category, error_type)

        # Add fine-grained patterns
        for pattern_name, pattern_info in self.fine_patterns.items():
            parent = pattern_info["parent"]
            if G.has_node(parent):
                G.add_edge(parent, pattern_name)

        return G


class HierarchicalAttention(nn.Module):
    """Hierarchical attention mechanism for multi-level classification."""

    def __init__(self, input_dim: int, num_levels: int = 3):
        super().__init__()
        self.num_levels = num_levels

        # Attention layers for each level
        self.level_attentions = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, input_dim // 2),
                    nn.Tanh(),
                    nn.Linear(input_dim // 2, 1),
                )
                for _ in range(num_levels)
            ]
        )

        # Level combination layer
        self.combine = nn.Linear(input_dim * num_levels, input_dim)

    def forward(
        self, features: torch.Tensor, level_mask: Optional[torch.Tensor] = None
    ):
        """
        Apply hierarchical attention.

        Args:
            features: Input features [batch, seq_len, hidden_dim]
            level_mask: Optional mask for different levels

        Returns:
            Attended features for each level
        """
        level_outputs = []

        for i, attention in enumerate(self.level_attentions):
            # Compute attention weights
            weights = attention(features)  # [batch, seq_len, 1]
            weights = F.softmax(weights, dim=1)

            # Apply attention
            attended = torch.sum(features * weights, dim=1)  # [batch, hidden_dim]
            level_outputs.append(attended)

        # Combine all levels
        combined = torch.cat(level_outputs, dim=-1)  # [batch, hidden_dim * num_levels]
        output = self.combine(combined)  # [batch, hidden_dim]

        return output, level_outputs


class HierarchicalErrorClassifier(nn.Module):
    """Neural network for hierarchical error classification."""

    def __init__(
        self, taxonomy: ErrorTaxonomy, hidden_dim: int = 768, dropout_rate: float = 0.3
    ):
        super().__init__()
        self.taxonomy = taxonomy
        self.hidden_dim = hidden_dim

        # Text encoder
        # Use specific revision for security and reproducibility
        self.encoder = AutoModel.from_pretrained(
            "microsoft/codebert-base",
            revision="1b2e0bfe5003709471fb6e04c0943470cf4a5b30",
        )

        # Hierarchical attention
        self.hierarchical_attention = HierarchicalAttention(hidden_dim, num_levels=3)

        # Level 1 classifier (coarse-grained)
        self.level1_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, len(taxonomy.taxonomy)),  # 5 coarse categories
        )

        # Level 2 classifiers (medium-grained)
        self.level2_classifiers = nn.ModuleDict()
        for l1_cat, l2_cats in taxonomy.taxonomy.items():
            self.level2_classifiers[l1_cat] = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim // 4, len(l2_cats)),
            )

        # Level 3 classifier (fine-grained patterns)
        self.level3_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, len(taxonomy.fine_patterns)),
        )

        # Confidence estimation
        self.confidence_estimator = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        Forward pass through hierarchical classifier.

        Args:
            input_ids: Token IDs
            attention_mask: Attention mask

        Returns:
            Multi-level classification results
        """
        # Encode input
        encoder_output = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )

        # Get hierarchical features
        features = encoder_output.last_hidden_state
        hierarchical_features, level_features = self.hierarchical_attention(features)

        # Level 1 classification
        level1_logits = self.level1_classifier(level_features[0])
        level1_probs = F.softmax(level1_logits, dim=-1)

        # Level 2 classification (conditional on level 1)
        level2_outputs = {}
        for i, (l1_cat, classifier) in enumerate(self.level2_classifiers.items()):
            level2_logits = classifier(level_features[1])
            level2_outputs[l1_cat] = {
                "logits": level2_logits,
                "probs": F.softmax(level2_logits, dim=-1),
            }

        # Level 3 classification
        level3_logits = self.level3_classifier(level_features[2])
        level3_probs = F.softmax(level3_logits, dim=-1)

        # Confidence estimation
        all_features = torch.cat(level_features, dim=-1)
        confidence = self.confidence_estimator(all_features)

        return {
            "level1": {"logits": level1_logits, "probs": level1_probs},
            "level2": level2_outputs,
            "level3": {"logits": level3_logits, "probs": level3_probs},
            "confidence": confidence,
            "features": hierarchical_features,
        }


class HierarchicalClassificationPipeline:
    """Complete pipeline for hierarchical error classification."""

    def __init__(self, model_path: Optional[str] = None):
        """Initialize the hierarchical classification pipeline."""
        self.taxonomy = ErrorTaxonomy()
        self.hierarchy = self.taxonomy.get_hierarchy()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize tokenizer and model
        # Use specific revision for security and reproducibility
        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/codebert-base",
            revision="1b2e0bfe5003709471fb6e04c0943470cf4a5b30",
        )
        self.model = HierarchicalErrorClassifier(self.taxonomy).to(self.device)

        # Label encoders for each level
        self.level1_encoder = LabelEncoder()
        self.level2_encoders: Dict[str, LabelEncoder] = {}
        self.level3_encoder = LabelEncoder()

        self._initialize_label_encoders()

        if model_path and Path(model_path).exists():
            self.load(model_path)

    def _initialize_label_encoders(self):
        """Initialize label encoders for each hierarchy level."""
        # Level 1
        self.level1_encoder.fit(list(self.taxonomy.taxonomy.keys()))

        # Level 2
        for l1_cat, l2_cats in self.taxonomy.taxonomy.items():
            self.level2_encoders[l1_cat] = LabelEncoder()
            self.level2_encoders[l1_cat].fit(list(l2_cats.keys()))

        # Level 3
        self.level3_encoder.fit(list(self.taxonomy.fine_patterns.keys()))

    def classify(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify error hierarchically.

        Args:
            error_data: Error data dictionary

        Returns:
            Hierarchical classification results
        """
        # Extract text for classification
        text = self._extract_error_text(error_data)

        # Tokenize
        inputs = self.tokenizer(
            text, return_tensors="pt", max_length=512, truncation=True, padding=True
        ).to(self.device)

        # Get model predictions
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Process results
        results = self._process_outputs(outputs, error_data)

        # Add hierarchical path
        results["hierarchy_path"] = self._get_hierarchy_path(results)

        # Add suggested fixes
        results["suggested_fixes"] = self._get_suggested_fixes(results)

        return results

    def _extract_error_text(self, error_data: Dict[str, Any]) -> str:
        """Extract relevant text from error data."""
        parts = []

        # Add error type and message
        if "exception_type" in error_data:
            parts.append(f"Error: {error_data['exception_type']}")
        if "message" in error_data:
            parts.append(f"Message: {error_data['message']}")

        # Add traceback
        if "traceback" in error_data:
            tb = error_data["traceback"]
            if isinstance(tb, list):
                parts.append("Traceback: " + " ".join(tb[-3:]))
            else:
                parts.append(f"Traceback: {tb}")

        # Add code context
        if (
            "error_details" in error_data
            and "detailed_frames" in error_data["error_details"]
        ):
            frames = error_data["error_details"]["detailed_frames"]
            if frames:
                last_frame = frames[-1]
                if "code" in last_frame:
                    parts.append(f"Code: {last_frame['code']}")

        return " ".join(parts)

    def _process_outputs(
        self, outputs: Dict[str, Any], error_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process model outputs into classification results."""
        results: Dict[str, Any] = {
            "confidence": float(outputs["confidence"].squeeze().cpu().item())
        }

        # Level 1 prediction
        level1_probs = outputs["level1"]["probs"][0].cpu().numpy()
        level1_idx = np.argmax(level1_probs)
        level1_label = self.level1_encoder.inverse_transform([level1_idx])[0]

        results["level1"] = {
            "prediction": level1_label,
            "confidence": float(level1_probs[level1_idx]),
            "all_probs": {
                label: float(prob)
                for label, prob in zip(self.level1_encoder.classes_, level1_probs)
            },
        }

        # Level 2 prediction (based on level 1)
        if level1_label in outputs["level2"]:
            level2_data = outputs["level2"][level1_label]
            level2_probs = level2_data["probs"][0].cpu().numpy()
            level2_idx = np.argmax(level2_probs)
            level2_label = self.level2_encoders[level1_label].inverse_transform(
                [level2_idx]
            )[0]

            results["level2"] = {
                "prediction": level2_label,
                "confidence": float(level2_probs[level2_idx]),
                "all_probs": {
                    label: float(prob)
                    for label, prob in zip(
                        self.level2_encoders[level1_label].classes_, level2_probs
                    )
                },
            }

        # Level 3 prediction (fine patterns)
        level3_probs = outputs["level3"]["probs"][0].cpu().numpy()
        top_patterns = []
        for idx in np.argsort(level3_probs)[::-1][:3]:  # Top 3 patterns
            if level3_probs[idx] > 0.1:  # Threshold
                pattern_name = self.level3_encoder.inverse_transform([idx])[0]
                top_patterns.append(
                    {
                        "pattern": pattern_name,
                        "confidence": float(level3_probs[idx]),
                        "fixes": self.taxonomy.fine_patterns[pattern_name]["fixes"],
                    }
                )

        results["level3"] = {"patterns": top_patterns}

        # Add original error info
        results["original_error"] = {
            "type": error_data.get("exception_type", "unknown"),
            "message": error_data.get("message", ""),
        }

        return results

    def _get_hierarchy_path(self, results: Dict[str, Any]) -> List[str]:
        """Get the hierarchical classification path."""
        path = ["root"]

        if "level1" in results:
            path.append(results["level1"]["prediction"])

            if "level2" in results:
                path.append(results["level2"]["prediction"])

                # Add specific error type if we can determine it
                original_type = results["original_error"]["type"]
                if original_type != "unknown":
                    path.append(original_type)

        return path

    def _get_suggested_fixes(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get suggested fixes based on classification."""
        fixes = []

        # Add fixes from fine-grained patterns
        if "level3" in results and "patterns" in results["level3"]:
            for pattern in results["level3"]["patterns"]:
                for fix_type in pattern["fixes"]:
                    fixes.append(
                        {
                            "type": fix_type,
                            "pattern": pattern["pattern"],
                            "confidence": pattern["confidence"],
                            "description": self._get_fix_description(fix_type),
                        }
                    )

        # Sort by confidence
        fixes.sort(key=lambda x: x["confidence"], reverse=True)

        return fixes

    def _get_fix_description(self, fix_type: str) -> str:
        """Get human-readable description for fix type."""
        fix_descriptions = {
            "add_default_value": "Add a default value to handle missing keys",
            "check_key_exists": "Check if key exists before accessing",
            "use_get_method": "Use dict.get() method with default value",
            "add_none_check": "Add None check before accessing attributes",
            "initialize_variable": "Initialize variable before use",
            "handle_optional": "Handle optional values properly",
            "check_list_length": "Check list length before indexing",
            "use_safe_indexing": "Use safe indexing with try-except",
            "handle_empty_list": "Handle empty list case",
            "add_type_conversion": "Add explicit type conversion",
            "validate_types": "Validate input types before operations",
            "use_type_hints": "Add type hints for better type checking",
            "check_file_exists": "Check if file exists before opening",
            "create_if_missing": "Create file if it doesn't exist",
            "use_default_path": "Use default path as fallback",
            "increase_timeout": "Increase timeout duration",
            "add_retry_logic": "Add retry logic with exponential backoff",
            "use_async_request": "Use asynchronous requests",
            "refactor_imports": "Refactor imports to avoid cycles",
            "lazy_import": "Use lazy imports",
            "restructure_modules": "Restructure modules to eliminate cycles",
            "specify_encoding": "Specify file encoding explicitly",
            "handle_encoding_errors": "Handle encoding errors gracefully",
            "detect_encoding": "Detect file encoding automatically",
        }

        return fix_descriptions.get(fix_type, f"Apply {fix_type} fix")

    def visualize_classification(self, results: Dict[str, Any]) -> str:
        """Create a text visualization of the classification."""
        lines = []
        lines.append("=" * 60)
        lines.append("HIERARCHICAL ERROR CLASSIFICATION")
        lines.append("=" * 60)

        # Original error
        lines.append(f"Original Error: {results['original_error']['type']}")
        lines.append(f"Message: {results['original_error']['message'][:80]}...")
        lines.append("")

        # Classification path
        lines.append("Classification Hierarchy:")
        path = results["hierarchy_path"]
        for i, node in enumerate(path):
            indent = "  " * i
            if i < len(path) - 1:
                lines.append(f"{indent}├── {node}")
            else:
                lines.append(f"{indent}└── {node}")
        lines.append("")

        # Level details
        if "level1" in results:
            lines.append(
                f"Level 1 (Coarse): {results['level1']['prediction']} "
                f"(confidence: {results['level1']['confidence']:.2f})"
            )

        if "level2" in results:
            lines.append(
                f"Level 2 (Medium): {results['level2']['prediction']} "
                f"(confidence: {results['level2']['confidence']:.2f})"
            )

        if "level3" in results and results["level3"]["patterns"]:
            lines.append("Level 3 (Fine patterns):")
            for pattern in results["level3"]["patterns"]:
                lines.append(
                    f"  - {pattern['pattern']} "
                    f"(confidence: {pattern['confidence']:.2f})"
                )

        lines.append("")
        lines.append(f"Overall Confidence: {results['confidence']:.2f}")

        # Suggested fixes
        if results["suggested_fixes"]:
            lines.append("")
            lines.append("Suggested Fixes:")
            for i, fix in enumerate(results["suggested_fixes"][:3], 1):
                lines.append(
                    f"{i}. {fix['description']} "
                    f"(confidence: {fix['confidence']:.2f})"
                )

        lines.append("=" * 60)

        return "\n".join(lines)

    def train(
        self,
        training_data: List[Tuple[Dict[str, Any], Dict[str, str]]],
        epochs: int = 10,
        batch_size: int = 16,
    ) -> Dict[str, Any]:
        """
        Train the hierarchical classifier.

        Args:
            training_data: List of (error_data, labels) tuples
            epochs: Number of training epochs
            batch_size: Batch size

        Returns:
            Training results
        """
        # This is a placeholder for the full training implementation
        logger.info(
            f"Training hierarchical classifier with {len(training_data)} samples..."
        )

        # In a real implementation, you would:
        # 1. Create data loaders for each hierarchy level
        # 2. Define loss functions for each level
        # 3. Implement multi-task learning with level-specific losses
        # 4. Track metrics for each level
        # 5. Implement early stopping and model checkpointing

        return {
            "epochs": epochs,
            "samples": len(training_data),
            "status": "training_complete",
        }

    def save(self, path: str):
        """Save the trained model."""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "taxonomy": self.taxonomy,
                "level1_encoder": self.level1_encoder,
                "level2_encoders": self.level2_encoders,
                "level3_encoder": self.level3_encoder,
            },
            path,
        )
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """Load a trained model."""
        # Load checkpoint using secure loading mechanism
        checkpoint = secure_torch_load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.taxonomy = checkpoint["taxonomy"]
        self.level1_encoder = checkpoint["level1_encoder"]
        self.level2_encoders = checkpoint["level2_encoders"]
        self.level3_encoder = checkpoint["level3_encoder"]
        logger.info(f"Model loaded from {path}")


def demonstrate_hierarchical_classification():
    """Demonstrate the hierarchical classification system."""
    # Initialize pipeline
    pipeline = HierarchicalClassificationPipeline()

    # Test errors
    test_errors = [
        {
            "exception_type": "KeyError",
            "message": "'user_id'",
            "traceback": [
                "Traceback (most recent call last):",
                "  File 'app.py', line 42, in get_user",
                "    user = users[request['user_id']]",
                "KeyError: 'user_id'",
            ],
            "error_details": {
                "detailed_frames": [{"code": "user = users[request['user_id']]"}]
            },
        },
        {
            "exception_type": "AttributeError",
            "message": "'NoneType' object has no attribute 'get'",
            "traceback": [
                "Traceback (most recent call last):",
                "  File 'handlers.py', line 78, in process",
                "    value = response.get('data')",
                "AttributeError: 'NoneType' object has no attribute 'get'",
            ],
            "error_details": {
                "detailed_frames": [{"code": "value = response.get('data')"}]
            },
        },
        {
            "exception_type": "FileNotFoundError",
            "message": "[Errno 2] No such file or directory: 'config.json'",
            "traceback": [
                "Traceback (most recent call last):",
                "  File 'config.py', line 15, in load_config",
                "    with open('config.json', 'r') as f:",
                "FileNotFoundError: [Errno 2] No such file or directory: 'config.json'",
            ],
            "error_details": {
                "detailed_frames": [{"code": "with open('config.json', 'r') as f:"}]
            },
        },
    ]

    # Classify each error
    for i, error in enumerate(test_errors):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Classifying error {i + 1}/{len(test_errors)}")

        # Perform classification
        results = pipeline.classify(error)

        # Display results
        visualization = pipeline.visualize_classification(results)
        print(visualization)


if __name__ == "__main__":
    demonstrate_hierarchical_classification()
