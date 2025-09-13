"""
Transformer-based code understanding module for Homeostasis.

This module implements transformer models for understanding code context,
relationships, and semantics to improve error analysis and patch generation.
"""

import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel,
    AutoTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Secure model loading configuration for production
# Maps model names to their secure revisions
SECURE_MODEL_REVISIONS = {
    "microsoft/codebert-base": "1b2e0bfe5003709471fb6e04c0943470cf4a5b30",
    "microsoft/graphcodebert-base": "fd47d4e93708a8dc6b5aab6a2b8a44a80e8af18f",
    "Salesforce/codet5-base": "3b7da1157cbbbbd699c4c00dc69b9fd9d1145a59",
    "Salesforce/codet5-small": "e1a7fc1dc96e0cf0e0fafab7f8aae07c7de2b2c9",
}


def get_secure_revision(model_name: str) -> Optional[str]:
    """Get secure revision for a model name, returns None if not in whitelist."""
    return SECURE_MODEL_REVISIONS.get(model_name)


@dataclass
class CodeContext:
    """Container for code context information."""

    code: str
    language: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    function_name: Optional[str] = None
    class_name: Optional[str] = None
    imports: List[str] = None
    variables: List[str] = None
    called_functions: List[str] = None

    def __post_init__(self):
        if self.imports is None:
            self.imports = []
        if self.variables is None:
            self.variables = []
        if self.called_functions is None:
            self.called_functions = []


class CodeContextExtractor:
    """Extract structured context from code snippets."""

    def __init__(self):
        self.language_patterns = {
            "python": {
                "import": r"^(?:from\s+\S+\s+)?import\s+.+",
                "function": r"^def\s+(\w+)\s*\(",
                "class": r"^class\s+(\w+)(?:\(.*?\))?:",
                "variable": r"^(\w+)\s*=",
                "call": r"(\w+)\s*\(",
            },
            "javascript": {
                "import": r'^(?:import|require)\s*\(?[\'"](.+?)[\'"]\)?',
                "function": r"^(?:function\s+(\w+)|const\s+(\w+)\s*=\s*(?:async\s+)?(?:\(.*?\)|function))",
                "class": r"^class\s+(\w+)",
                "variable": r"^(?:let|const|var)\s+(\w+)\s*=",
                "call": r"(\w+)\s*\(",
            },
            "java": {
                "import": r"^import\s+[\w\.]+;",
                "function": r"(?:public|private|protected)?\s*(?:static\s+)?[\w<>\[\]]+\s+(\w+)\s*\(",
                "class": r"^(?:public\s+)?class\s+(\w+)",
                "variable": r"(?:public|private|protected)?\s*(?:static\s+)?(?:final\s+)?[\w<>\[\]]+\s+(\w+)\s*[=;]",
                "call": r"(\w+)\s*\(",
            },
        }

    def extract_context(
        self, code: str, language: str, error_line: Optional[int] = None
    ) -> CodeContext:
        """Extract context from code snippet."""
        lines = code.split("\n")
        patterns = self.language_patterns.get(
            language, self.language_patterns["python"]
        )

        context = CodeContext(code=code, language=language, line_number=error_line)

        # Extract imports
        import_pattern = re.compile(patterns["import"], re.MULTILINE)
        context.imports = import_pattern.findall(code)

        # Extract function context
        if error_line:
            # Find the function containing the error line
            for i in range(error_line - 1, -1, -1):
                if i < len(lines):
                    func_match = re.match(patterns["function"], lines[i])
                    if func_match:
                        context.function_name = func_match.group(1)
                        break

                    class_match = re.match(patterns["class"], lines[i])
                    if class_match:
                        context.class_name = class_match.group(1)

        # Extract variables and function calls
        var_pattern = re.compile(patterns["variable"], re.MULTILINE)
        call_pattern = re.compile(patterns["call"])

        for line in lines:
            var_matches = var_pattern.findall(line)
            context.variables.extend(var_matches)

            call_matches = call_pattern.findall(line)
            context.called_functions.extend(call_matches)

        # Remove duplicates
        context.variables = list(set(context.variables))
        context.called_functions = list(set(context.called_functions))

        return context


class TransformerCodeAnalyzer:
    """Transformer-based code analysis for error understanding."""

    def __init__(self, model_name: str = "microsoft/codebert-base"):
        """
        Initialize the transformer code analyzer.

        Args:
            model_name: Name of the pre-trained model to use
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load tokenizer and model with secure revision
        revision = get_secure_revision(model_name)
        if revision is None:
            raise ValueError(f"Model {model_name} not in secure whitelist")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, revision=revision
        )  # nosec: B615 - Using secure revision
        self.model = AutoModel.from_pretrained(
            model_name, revision=revision
        ).to(  # nosec: B615 - Using secure revision
            self.device
        )

        # Code context extractor
        self.context_extractor = CodeContextExtractor()

        # Initialize code understanding heads
        self._initialize_understanding_heads()

    def _initialize_understanding_heads(self):
        """Initialize task-specific heads for code understanding."""
        hidden_size = self.model.config.hidden_size

        # Error localization head
        self.error_localization = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1),  # Score for each token
        ).to(self.device)

        # Code relationship head
        self.relationship_classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(
                hidden_size, 5
            ),  # Types: calls, defines, uses, imports, unrelated
        ).to(self.device)

        # Fix suggestion head
        self.fix_generator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
        ).to(self.device)

    def encode_code(
        self, code: str, language: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Encode code into transformer representations.

        Args:
            code: Code snippet
            language: Programming language

        Returns:
            Dictionary with encoded representations
        """
        # Add language tag if specified
        if language:
            code = f"<{language}> {code}"

        # Tokenize
        inputs = self.tokenizer(
            code, return_tensors="pt", max_length=512, truncation=True, padding=True
        ).to(self.device)

        # Get model outputs
        with torch.no_grad():
            outputs = self.model(**inputs)

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "hidden_states": outputs.last_hidden_state,
            "pooled_output": (
                outputs.pooler_output
                if hasattr(outputs, "pooler_output")
                else outputs.last_hidden_state.mean(dim=1)
            ),
        }

    def analyze_error_context(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze error context using transformer model.

        Args:
            error_data: Error data dictionary

        Returns:
            Analysis results
        """
        # Extract code context from error
        code_snippet = self._extract_code_from_error(error_data)
        language = error_data.get("language", "python")
        error_line = self._get_error_line(error_data)

        # Extract structured context
        context = self.context_extractor.extract_context(
            code_snippet, language, error_line
        )

        # Encode the code
        encoded = self.encode_code(code_snippet, language)

        # Localize error
        error_scores = self.error_localization(encoded["hidden_states"])
        error_probs = torch.sigmoid(error_scores.squeeze(-1))

        # Find most likely error location
        masked_probs = error_probs * encoded["attention_mask"].float()
        error_token_idx = torch.argmax(masked_probs, dim=-1)

        # Decode error location
        error_tokens = self.tokenizer.convert_ids_to_tokens(
            encoded["input_ids"][0][error_token_idx].tolist()
        )

        # Analyze code relationships
        relationships = self._analyze_relationships(encoded, context)

        # Generate fix suggestions
        fix_embeddings = self.fix_generator(encoded["pooled_output"])

        return {
            "context": {
                "function": context.function_name,
                "class": context.class_name,
                "imports": context.imports,
                "variables": context.variables,
                "called_functions": context.called_functions,
            },
            "error_location": {
                "tokens": (
                    error_tokens if isinstance(error_tokens, list) else [error_tokens]
                ),
                "confidence": float(torch.max(masked_probs).item()),
                "token_scores": error_probs[0].cpu().numpy().tolist(),
            },
            "relationships": relationships,
            "code_embedding": encoded["pooled_output"].cpu().numpy().tolist(),
            "fix_embedding": fix_embeddings.cpu().numpy().tolist(),
        }

    def _extract_code_from_error(self, error_data: Dict[str, Any]) -> str:
        """Extract code snippet from error data."""
        # Try to get code from detailed frames
        if (
            "error_details" in error_data
            and "detailed_frames" in error_data["error_details"]
        ):
            frames = error_data["error_details"]["detailed_frames"]
            if frames:
                # Get code from the last frame
                last_frame = frames[-1]
                if "code_context" in last_frame:
                    return "\n".join(last_frame["code_context"])
                elif "code" in last_frame:
                    return last_frame["code"]

        # Try to extract from traceback
        if "traceback" in error_data:
            tb = error_data["traceback"]
            if isinstance(tb, list):
                # Look for code lines in traceback
                code_lines = []
                for line in tb:
                    if (
                        line.strip()
                        and not line.startswith("Traceback")
                        and not line.startswith("File")
                    ):
                        code_lines.append(line.strip())
                if code_lines:
                    return "\n".join(code_lines)

        return ""

    def _get_error_line(self, error_data: Dict[str, Any]) -> Optional[int]:
        """Extract error line number from error data."""
        if (
            "error_details" in error_data
            and "detailed_frames" in error_data["error_details"]
        ):
            frames = error_data["error_details"]["detailed_frames"]
            if frames:
                return frames[-1].get("line")
        return None

    def _analyze_relationships(
        self, encoded: Dict[str, torch.Tensor], context: CodeContext
    ) -> Dict[str, List[str]]:
        """Analyze relationships between code elements."""
        relationships = defaultdict(list)

        # This is a simplified version - in practice, you'd analyze
        # relationships between different code elements
        if context.imports:
            relationships["imports"] = context.imports
        if context.called_functions:
            relationships["calls"] = context.called_functions
        if context.variables:
            relationships["defines"] = context.variables

        return dict(relationships)

    def compare_code_similarity(
        self, code1: str, code2: str, language: Optional[str] = None
    ) -> float:
        """
        Compare semantic similarity between two code snippets.

        Args:
            code1: First code snippet
            code2: Second code snippet
            language: Programming language

        Returns:
            Similarity score (0-1)
        """
        # Encode both snippets
        enc1 = self.encode_code(code1, language)
        enc2 = self.encode_code(code2, language)

        # Compute cosine similarity
        pooled1 = enc1["pooled_output"]
        pooled2 = enc2["pooled_output"]

        similarity = F.cosine_similarity(pooled1, pooled2, dim=-1)

        return float(similarity.item())

    def suggest_fixes(
        self, error_data: Dict[str, Any], candidate_fixes: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Rank candidate fixes based on code understanding.

        Args:
            error_data: Error data
            candidate_fixes: List of potential fixes

        Returns:
            Ranked list of fixes with scores
        """
        # Analyze error context
        error_analysis = self.analyze_error_context(error_data)
        error_embedding = torch.tensor(error_analysis["fix_embedding"]).to(self.device)

        # Score each candidate fix
        scored_fixes = []
        for fix in candidate_fixes:
            # Encode the fix
            fix_encoded = self.encode_code(fix, error_data.get("language", "python"))
            fix_embedding = self.fix_generator(fix_encoded["pooled_output"])

            # Compute similarity to error context
            similarity = F.cosine_similarity(error_embedding, fix_embedding, dim=-1)

            # Check if fix addresses the error location
            fix_tokens = self.tokenizer.tokenize(fix)
            error_tokens = error_analysis["error_location"]["tokens"]
            token_overlap = (
                len(set(fix_tokens) & set(error_tokens)) / len(error_tokens)
                if error_tokens
                else 0
            )

            # Combined score
            score = 0.7 * similarity.item() + 0.3 * token_overlap

            scored_fixes.append(
                {
                    "fix": fix,
                    "score": float(score),
                    "similarity": float(similarity.item()),
                    "token_overlap": token_overlap,
                }
            )

        # Sort by score
        scored_fixes.sort(key=lambda x: x["score"], reverse=True)

        return scored_fixes


class CodeT5Analyzer:
    """Code understanding using CodeT5 model for generation tasks."""

    def __init__(self, model_name: str = "Salesforce/codet5-base"):
        """Initialize CodeT5 analyzer."""
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model and tokenizer with secure revision
        revision = get_secure_revision(model_name)
        if revision is None:
            raise ValueError(f"Model {model_name} not in secure whitelist")

        self.tokenizer = T5Tokenizer.from_pretrained(
            model_name, revision=revision
        )  # nosec: B615 - Using secure revision
        self.model = T5ForConditionalGeneration.from_pretrained(  # nosec: B615 - Using secure revision
            model_name, revision=revision
        ).to(
            self.device
        )

    def generate_fix(self, error_context: str, max_length: int = 150) -> str:
        """
        Generate a fix for the given error context.

        Args:
            error_context: Error context including code and error message
            max_length: Maximum length of generated fix

        Returns:
            Generated fix
        """
        # Prepare input
        input_text = f"fix error: {error_context}"
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True,
        ).to(self.device)

        # Generate fix
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=5,
                early_stopping=True,
                temperature=0.7,
            )

        # Decode output
        fix = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return fix

    def explain_error(self, error_data: Dict[str, Any]) -> str:
        """
        Generate an explanation for the error.

        Args:
            error_data: Error data dictionary

        Returns:
            Error explanation
        """
        # Prepare context
        error_type = error_data.get("exception_type", "Error")
        error_msg = error_data.get("message", "")
        code = self._extract_relevant_code(error_data)

        input_text = f"explain error: {error_type}: {error_msg} in code: {code}"

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True,
        ).to(self.device)

        # Generate explanation
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_length=200, num_beams=3, early_stopping=True
            )

        explanation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return explanation

    def _extract_relevant_code(self, error_data: Dict[str, Any]) -> str:
        """Extract relevant code from error data."""
        if (
            "error_details" in error_data
            and "detailed_frames" in error_data["error_details"]
        ):
            frames = error_data["error_details"]["detailed_frames"]
            if frames:
                last_frame = frames[-1]
                if "code" in last_frame:
                    return last_frame["code"]
        return ""


class CodeUnderstandingPipeline:
    """Complete pipeline for transformer-based code understanding."""

    def __init__(self):
        """Initialize the code understanding pipeline."""
        self.analyzer = TransformerCodeAnalyzer()
        self.generator = CodeT5Analyzer()
        self.context_cache = {}

    def process_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an error through the complete understanding pipeline.

        Args:
            error_data: Error data dictionary

        Returns:
            Comprehensive analysis results
        """
        # Analyze error context
        logger.info("Analyzing error context...")
        context_analysis = self.analyzer.analyze_error_context(error_data)

        # Generate explanation
        logger.info("Generating error explanation...")
        explanation = self.generator.explain_error(error_data)

        # Generate fix
        logger.info("Generating potential fix...")
        error_context = self._build_error_context(error_data, context_analysis)
        suggested_fix = self.generator.generate_fix(error_context)

        # Find similar errors in cache
        similar_errors = self._find_similar_errors(error_data)

        return {
            "context_analysis": context_analysis,
            "explanation": explanation,
            "suggested_fix": suggested_fix,
            "similar_errors": similar_errors,
            "confidence": self._calculate_confidence(context_analysis),
            "metadata": {
                "language": error_data.get("language", "python"),
                "error_type": error_data.get("exception_type", "unknown"),
                "timestamp": error_data.get("timestamp"),
            },
        }

    def _build_error_context(
        self, error_data: Dict[str, Any], context_analysis: Dict[str, Any]
    ) -> str:
        """Build comprehensive error context for fix generation."""
        parts = []

        # Add error type and message
        if "exception_type" in error_data:
            parts.append(
                f"{error_data['exception_type']}: {error_data.get('message', '')}"
            )

        # Add code context
        if "context" in context_analysis:
            ctx = context_analysis["context"]
            if ctx["function"]:
                parts.append(f"in function {ctx['function']}")
            if ctx["class"]:
                parts.append(f"in class {ctx['class']}")

        # Add relevant code
        code = self.analyzer._extract_code_from_error(error_data)
        if code:
            parts.append(f"code: {code}")

        return " ".join(parts)

    def _find_similar_errors(
        self, error_data: Dict[str, Any], threshold: float = 0.8
    ) -> List[Dict[str, Any]]:
        """Find similar errors from cache."""
        similar = []

        # Get current error embedding
        current_analysis = self.analyzer.analyze_error_context(error_data)
        current_embedding = torch.tensor(current_analysis["code_embedding"])

        # Compare with cached errors
        for cached_id, cached_data in self.context_cache.items():
            if "code_embedding" in cached_data:
                cached_embedding = torch.tensor(cached_data["code_embedding"])
                similarity = F.cosine_similarity(
                    current_embedding.unsqueeze(0), cached_embedding.unsqueeze(0)
                ).item()

                if similarity > threshold:
                    similar.append(
                        {
                            "id": cached_id,
                            "similarity": similarity,
                            "error_type": cached_data.get("error_type"),
                            "fix_applied": cached_data.get("fix_applied"),
                        }
                    )

        # Sort by similarity
        similar.sort(key=lambda x: x["similarity"], reverse=True)

        return similar[:5]  # Return top 5

    def _calculate_confidence(self, context_analysis: Dict[str, Any]) -> float:
        """Calculate overall confidence score."""
        scores = []

        # Error location confidence
        if "error_location" in context_analysis:
            scores.append(context_analysis["error_location"]["confidence"])

        # Context completeness
        ctx = context_analysis.get("context", {})
        context_score = sum(
            [
                0.2 if ctx.get("function") else 0,
                0.2 if ctx.get("class") else 0,
                0.2 if ctx.get("imports") else 0,
                0.2 if ctx.get("variables") else 0,
                0.2 if ctx.get("called_functions") else 0,
            ]
        )
        scores.append(context_score)

        return np.mean(scores) if scores else 0.5

    def cache_result(self, error_id: str, result: Dict[str, Any]):
        """Cache analysis result for future reference."""
        self.context_cache[error_id] = {
            "code_embedding": result["context_analysis"]["code_embedding"],
            "error_type": result["metadata"]["error_type"],
            "fix_applied": result.get("suggested_fix"),
            "timestamp": result["metadata"]["timestamp"],
        }


def create_test_samples() -> List[Dict[str, Any]]:
    """Create test samples for demonstration."""
    return [
        {
            "exception_type": "AttributeError",
            "message": "'NoneType' object has no attribute 'get'",
            "language": "python",
            "error_details": {
                "detailed_frames": [
                    {
                        "file": "app.py",
                        "line": 42,
                        "function": "process_data",
                        "code": 'value = response.get("data")',
                        "code_context": [
                            "def process_data(response):",
                            "    # Process API response",
                            '    value = response.get("data")',
                            '    return value["id"]',
                        ],
                    }
                ]
            },
        },
        {
            "exception_type": "KeyError",
            "message": "'user_id'",
            "language": "python",
            "error_details": {
                "detailed_frames": [
                    {
                        "file": "handlers.py",
                        "line": 78,
                        "function": "get_user",
                        "code": 'user = users[request["user_id"]]',
                        "code_context": [
                            "def get_user(request, users):",
                            "    # Get user from dictionary",
                            '    user = users[request["user_id"]]',
                            "    return user",
                        ],
                    }
                ]
            },
        },
    ]


if __name__ == "__main__":
    # Test the transformer code understanding
    logger.info("Initializing code understanding pipeline...")
    pipeline = CodeUnderstandingPipeline()

    # Test with sample errors
    test_errors = create_test_samples()

    for i, error in enumerate(test_errors):
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Processing error {i + 1}/{len(test_errors)}")
        logger.info(f"Error type: {error['exception_type']}")
        logger.info(f"Message: {error['message']}")

        # Process error
        result = pipeline.process_error(error)

        # Display results
        logger.info("\nContext Analysis:")
        logger.info(f"  Function: {result['context_analysis']['context']['function']}")
        logger.info(
            f"  Variables: {result['context_analysis']['context']['variables']}"
        )
        logger.info(
            f"  Error location confidence: {result['context_analysis']['error_location']['confidence']:.2f}"
        )

        logger.info(f"\nExplanation: {result['explanation']}")
        logger.info(f"\nSuggested Fix: {result['suggested_fix']}")
        logger.info(f"\nOverall Confidence: {result['confidence']:.2f}")

        # Cache result
        pipeline.cache_result(f"error_{i}", result)
