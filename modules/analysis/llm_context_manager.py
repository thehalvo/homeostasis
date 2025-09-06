"""
LLM Context Manager for storing and preparing error context for LLM prompt generation.

This module provides functionality to store comprehensive error context in the orchestrator
and prepare it for LLM-based patch generation and analysis.
"""

import json
import logging
import shutil
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..security.llm_security_manager import create_llm_security_manager
from .comprehensive_error_detector import ErrorClassification, ErrorContext
from .intelligent_classifier import IntelligentClassifier


class FailureType(Enum):
    """Types of failures that can occur during LLM processing."""

    LLM_REQUEST_FAILED = "llm_request_failed"
    PATCH_GENERATION_FAILED = "patch_generation_failed"
    PATCH_APPLICATION_FAILED = "patch_application_failed"
    TEST_VALIDATION_FAILED = "test_validation_failed"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


@dataclass
class FailureLog:
    """Log entry for failures during LLM processing."""

    failure_id: str
    context_id: str
    failure_type: FailureType
    timestamp: float
    error_message: str
    provider: Optional[str] = None
    attempt_number: int = 1
    retry_delay: Optional[float] = None
    stack_trace: Optional[str] = None
    additional_context: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "failure_id": self.failure_id,
            "context_id": self.context_id,
            "failure_type": self.failure_type.value,
            "timestamp": self.timestamp,
            "error_message": self.error_message,
            "provider": self.provider,
            "attempt_number": self.attempt_number,
            "retry_delay": self.retry_delay,
            "stack_trace": self.stack_trace,
            "additional_context": self.additional_context,
        }


@dataclass
class RetryContext:
    """Context for tracking retry attempts."""

    original_context_id: str
    retry_attempt: int
    previous_failures: List[FailureLog] = field(default_factory=list)
    modified_prompts: List[str] = field(default_factory=list)
    provider_fallback_chain: List[str] = field(default_factory=list)
    accumulated_context: Dict[str, Any] = field(default_factory=dict)


logger = logging.getLogger(__name__)


@dataclass
class LLMContext:
    """Comprehensive context for LLM prompt generation."""

    # Unique identifier for this context
    context_id: str

    # Timestamp
    created_at: str
    updated_at: str

    # Original error information
    error_context: ErrorContext
    error_classification: ErrorClassification

    # Analysis results
    rule_based_analysis: Optional[Dict[str, Any]] = None
    ml_analysis: Optional[Dict[str, Any]] = None
    language_specific_analysis: Optional[Dict[str, Any]] = None
    compiler_diagnostics: Optional[Dict[str, Any]] = None

    # Project context
    project_structure: Optional[Dict[str, Any]] = None
    related_files: Optional[List[str]] = None
    dependency_graph: Optional[Dict[str, Any]] = None

    # Historical context
    similar_errors: Optional[List[Dict[str, Any]]] = None
    previous_fixes: Optional[List[Dict[str, Any]]] = None

    # Monitoring data
    performance_metrics: Optional[Dict[str, Any]] = None
    system_state: Optional[Dict[str, Any]] = None

    # LLM-specific metadata
    prompt_template: Optional[str] = None
    expected_output_format: Optional[str] = None
    provider_preferences: Optional[Dict[str, Any]] = None

    # Failure tracking
    llm_failures: Optional[List[FailureLog]] = None
    validation_failures: Optional[List[FailureLog]] = None
    retry_history: Optional[List[Dict[str, Any]]] = None
    common_failure_patterns: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)

        # Convert ErrorContext and ErrorClassification to dict
        if isinstance(result["error_context"], ErrorContext):
            result["error_context"] = result["error_context"].to_dict()

        if isinstance(result["error_classification"], ErrorClassification):
            result["error_classification"] = result["error_classification"].to_dict()

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMContext":
        """Create LLMContext from dictionary."""
        # Convert error_context back to ErrorContext object
        if isinstance(data.get("error_context"), dict):
            error_context_data = data["error_context"]
            # Handle enum conversion
            from .comprehensive_error_detector import LanguageType

            if "language" in error_context_data:
                error_context_data["language"] = LanguageType(
                    error_context_data["language"]
                )

            data["error_context"] = ErrorContext(**error_context_data)

        # Convert error_classification back to ErrorClassification object
        if isinstance(data.get("error_classification"), dict):
            classification_data = data["error_classification"]
            # Handle enum conversions
            from .comprehensive_error_detector import ErrorCategory, ErrorSeverity

            if "category" in classification_data:
                classification_data["category"] = ErrorCategory(
                    classification_data["category"]
                )
            if "severity" in classification_data:
                classification_data["severity"] = ErrorSeverity(
                    classification_data["severity"]
                )

            data["error_classification"] = ErrorClassification(**classification_data)

        return cls(**data)


class LLMContextManager:
    """
    Manager for storing and retrieving error contexts for LLM prompt generation.

    This class provides:
    1. Storage of comprehensive error contexts
    2. Context enrichment with additional analysis
    3. Context retrieval and search capabilities
    4. LLM prompt generation preparation
    5. Context cleanup and archival
    """

    def __init__(self, storage_dir: Optional[Path] = None, max_contexts: int = 1000):
        """
        Initialize LLM context manager.

        Args:
            storage_dir: Directory to store context files
            max_contexts: Maximum number of contexts to keep in memory
        """
        self.storage_dir = storage_dir or Path.cwd() / "llm_contexts"
        self.storage_dir.mkdir(exist_ok=True)

        self.max_contexts = max_contexts
        self.contexts: Dict[str, LLMContext] = {}

        # Initialize classifier for enhanced analysis
        self.classifier = IntelligentClassifier()

        # Initialize security manager for data protection
        self.security_manager = create_llm_security_manager()

        # Load existing contexts
        self._load_existing_contexts()

        logger.info(
            f"Initialized LLM Context Manager with storage at {self.storage_dir}"
        )

    def store_error_context(
        self,
        error_context: ErrorContext,
        additional_analysis: Optional[Dict[str, Any]] = None,
        project_context: Optional[Dict[str, Any]] = None,
        monitoring_data: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Store a comprehensive error context for LLM processing.

        Args:
            error_context: Basic error context
            additional_analysis: Additional analysis results
            project_context: Project-specific context
            monitoring_data: System monitoring data

        Returns:
            Context ID for the stored context
        """

        # Generate unique context ID
        context_id = str(uuid.uuid4())

        # Classify the error
        error_classification = self.classifier.classify_error(error_context)

        # Create LLM context
        llm_context = LLMContext(
            context_id=context_id,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            error_context=error_context,
            error_classification=error_classification,
        )

        # Add additional analysis if provided
        if additional_analysis:
            llm_context.rule_based_analysis = additional_analysis.get("rule_based")
            llm_context.ml_analysis = additional_analysis.get("ml_based")
            llm_context.language_specific_analysis = additional_analysis.get(
                "language_specific"
            )
            llm_context.compiler_diagnostics = additional_analysis.get(
                "compiler_diagnostics"
            )

        # Add project context
        if project_context:
            llm_context.project_structure = project_context.get("structure")
            llm_context.related_files = project_context.get("related_files")
            llm_context.dependency_graph = project_context.get("dependencies")

        # Add monitoring data
        if monitoring_data:
            llm_context.performance_metrics = monitoring_data.get("performance")
            llm_context.system_state = monitoring_data.get("system_state")

        # Enrich context with historical data
        self._enrich_with_historical_data(llm_context)

        # Set LLM-specific metadata
        self._set_llm_metadata(llm_context)

        # Store in memory and file
        self.contexts[context_id] = llm_context
        self._save_context_to_file(llm_context)

        # Clean up old contexts if necessary
        self._cleanup_old_contexts()

        logger.info(f"Stored error context {context_id} for LLM processing")
        return context_id

    def get_context(self, context_id: str) -> Optional[LLMContext]:
        """
        Retrieve a stored error context.

        Args:
            context_id: Context ID to retrieve

        Returns:
            LLM context or None if not found
        """
        return self.contexts.get(context_id)

    def search_similar_contexts(
        self, error_context: ErrorContext, max_results: int = 5
    ) -> List[LLMContext]:
        """
        Search for similar error contexts.

        Args:
            error_context: Error context to find similar contexts for
            max_results: Maximum number of results to return

        Returns:
            List of similar contexts
        """
        similar_contexts = []

        for context in self.contexts.values():
            similarity_score = self._calculate_similarity(
                error_context, context.error_context
            )
            if similarity_score > 0.5:  # Threshold for similarity
                similar_contexts.append((context, similarity_score))

        # Sort by similarity score and return top results
        similar_contexts.sort(key=lambda x: x[1], reverse=True)
        return [context for context, _ in similar_contexts[:max_results]]

    def prepare_llm_prompt(
        self, context_id: str, template_type: str = "comprehensive"
    ) -> Dict[str, Any]:
        """
        Prepare an LLM prompt from stored context with security safeguards.

        Args:
            context_id: Context ID to prepare prompt for
            template_type: Type of prompt template to use

        Returns:
            Prepared prompt data for LLM
        """
        context = self.get_context(context_id)
        if not context:
            raise ValueError(f"Context {context_id} not found")

        # Sanitize context before processing
        sanitized_context = self.security_manager.sanitize_llm_context(context)

        # Create prompt based on template type
        if template_type == "comprehensive":
            prompt_data = self._create_comprehensive_prompt(sanitized_context)
        elif template_type == "focused":
            prompt_data = self._create_focused_prompt(sanitized_context)
        elif template_type == "patch_generation":
            prompt_data = self._create_patch_generation_prompt(sanitized_context)
        else:
            raise ValueError(f"Unknown template type: {template_type}")

        # Scrub sensitive data from the final prompt
        scrubbed_prompt, scrubbing_detections = (
            self.security_manager.scrub_sensitive_data(
                prompt_data.get("user_prompt", ""), context_id
            )
        )
        prompt_data["user_prompt"] = scrubbed_prompt

        # Add security metadata
        prompt_data["security_metadata"] = {
            "context_sanitized": True,
            "sensitive_data_scrubbed": len(scrubbing_detections) > 0,
            "scrubbing_detections": len(scrubbing_detections),
            "compliance_frameworks": [
                f.value for f in self.security_manager.active_compliance_frameworks
            ],
        }

        logger.info(f"Prepared secure LLM prompt for context {context_id}")
        return prompt_data

    def update_context_with_results(
        self,
        context_id: str,
        llm_response: Dict[str, Any],
        patch_result: Optional[Dict[str, Any]] = None,
        test_result: Optional[Dict[str, Any]] = None,
    ):
        """
        Update context with LLM response and results after security validation.

        Args:
            context_id: Context ID to update
            llm_response: LLM response data
            patch_result: Patch generation result
            test_result: Test execution result
        """
        context = self.get_context(context_id)
        if not context:
            return

        # Validate LLM response for security
        response_text = str(llm_response.get("content", ""))
        is_safe, safety_violations = self.security_manager.validate_llm_response_safety(
            response_text, context_id
        )

        # Detect data leakage in the response
        prompt_text = ""  # We'd need to store the original prompt to compare
        leakage_detections = self.security_manager.detect_data_leakage(
            prompt_text, response_text, context_id
        )

        # Update context with results
        if not hasattr(context, "llm_responses"):
            context.llm_responses = []

        response_entry = {
            "timestamp": datetime.now().isoformat(),
            "response": llm_response,
            "patch_result": patch_result,
            "test_result": test_result,
            "security_validation": {
                "is_safe": is_safe,
                "safety_violations": safety_violations,
                "leakage_detections": [d.__dict__ for d in leakage_detections],
            },
        }

        context.llm_responses.append(response_entry)
        context.updated_at = datetime.now().isoformat()

        # Save updated context
        self._save_context_to_file(context)

        logger.info(f"Updated context {context_id} with LLM results (safe: {is_safe})")

    def get_context_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about stored contexts.

        Returns:
            Context statistics
        """
        stats = {
            "total_contexts": len(self.contexts),
            "categories": defaultdict(int),
            "languages": defaultdict(int),
            "severities": defaultdict(int),
            "avg_confidence": 0.0,
            "storage_size_mb": 0.0,
        }

        total_confidence = 0.0

        for context in self.contexts.values():
            # Count by category
            stats["categories"][context.error_classification.category.value] += 1

            # Count by language
            stats["languages"][context.error_context.language.value] += 1

            # Count by severity
            stats["severities"][context.error_classification.severity.value] += 1

            # Sum confidence for average
            total_confidence += context.error_classification.confidence

        # Calculate average confidence
        if self.contexts:
            stats["avg_confidence"] = total_confidence / len(self.contexts)

        # Calculate storage size
        storage_size = sum(f.stat().st_size for f in self.storage_dir.glob("*.json"))
        stats["storage_size_mb"] = storage_size / (1024 * 1024)

        # Convert defaultdicts to regular dicts
        stats["categories"] = dict(stats["categories"])
        stats["languages"] = dict(stats["languages"])
        stats["severities"] = dict(stats["severities"])

        return stats

    def export_contexts_for_training(
        self, output_file: Path, include_successful_fixes: bool = True
    ) -> int:
        """
        Export contexts for ML model training.

        Args:
            output_file: Output file path
            include_successful_fixes: Whether to include only successful fixes

        Returns:
            Number of contexts exported
        """
        training_data = []

        for context in self.contexts.values():
            # Skip contexts without fix results if requested
            if include_successful_fixes and not hasattr(context, "llm_responses"):
                continue

            # Create training sample
            sample = {
                "input": {
                    "error_message": context.error_context.error_message,
                    "exception_type": context.error_context.exception_type,
                    "language": context.error_context.language.value,
                    "file_path": context.error_context.file_path,
                    "stack_trace": context.error_context.stack_trace,
                },
                "classification": {
                    "category": context.error_classification.category.value,
                    "severity": context.error_classification.severity.value,
                    "confidence": context.error_classification.confidence,
                    "root_cause": context.error_classification.root_cause,
                },
            }

            # Add successful fix information if available
            if hasattr(context, "llm_responses") and context.llm_responses:
                successful_fixes = [
                    response
                    for response in context.llm_responses
                    if response.get("test_result", {}).get("success", False)
                ]
                if successful_fixes:
                    sample["successful_fix"] = successful_fixes[
                        -1
                    ]  # Latest successful fix

            training_data.append(sample)

        # Save training data
        with open(output_file, "w") as f:
            json.dump(training_data, f, indent=2)

        logger.info(
            f"Exported {len(training_data)} contexts for training to {output_file}"
        )
        return len(training_data)

    def _load_existing_contexts(self):
        """Load existing contexts from storage."""
        try:
            for context_file in self.storage_dir.glob("*.json"):
                with open(context_file, "r") as f:
                    context_data = json.load(f)
                    context = LLMContext.from_dict(context_data)
                    self.contexts[context.context_id] = context

            logger.info(f"Loaded {len(self.contexts)} existing contexts")
        except Exception as e:
            logger.warning(f"Error loading existing contexts: {e}")

    def _save_context_to_file(self, context: LLMContext):
        """Save context to file."""
        try:
            context_file = self.storage_dir / f"{context.context_id}.json"
            with open(context_file, "w") as f:
                json.dump(context.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Error saving context {context.context_id}: {e}")

    def _cleanup_old_contexts(self):
        """Clean up old contexts to maintain memory limits."""
        if len(self.contexts) <= self.max_contexts:
            return

        # Sort by creation time and remove oldest
        sorted_contexts = sorted(self.contexts.values(), key=lambda x: x.created_at)

        contexts_to_remove = sorted_contexts[: -self.max_contexts]

        for context in contexts_to_remove:
            # Remove from memory
            self.contexts.pop(context.context_id, None)

            # Keep file but archive it
            context_file = self.storage_dir / f"{context.context_id}.json"
            if context_file.exists():
                archive_dir = self.storage_dir / "archived"
                archive_dir.mkdir(exist_ok=True)
                shutil.move(str(context_file), str(archive_dir / context_file.name))

        logger.info(f"Archived {len(contexts_to_remove)} old contexts")

    def _enrich_with_historical_data(self, context: LLMContext):
        """Enrich context with historical data."""
        # Find similar previous errors
        similar_contexts = self.search_similar_contexts(
            context.error_context, max_results=3
        )

        if similar_contexts:
            context.similar_errors = [
                {
                    "context_id": ctx.context_id,
                    "error_message": ctx.error_context.error_message,
                    "classification": ctx.error_classification.to_dict(),
                    "created_at": ctx.created_at,
                }
                for ctx in similar_contexts
            ]

        # Find previous successful fixes for similar errors
        previous_fixes = []
        for similar_ctx in similar_contexts:
            if hasattr(similar_ctx, "llm_responses") and similar_ctx.llm_responses:
                successful_fixes = [
                    response
                    for response in similar_ctx.llm_responses
                    if response.get("test_result", {}).get("success", False)
                ]
                if successful_fixes:
                    previous_fixes.extend(successful_fixes)

        if previous_fixes:
            context.previous_fixes = previous_fixes[:5]  # Limit to 5 most recent

        # Add failure pattern analysis from similar contexts
        common_failures = self._analyze_common_failure_patterns(similar_contexts)
        if common_failures:
            context.common_failure_patterns = common_failures

    def _set_llm_metadata(self, context: LLMContext):
        """Set LLM-specific metadata for the context."""
        # Determine appropriate prompt template based on error type
        category = context.error_classification.category.value

        if category == "syntax":
            context.prompt_template = "syntax_fix"
        elif category == "logic":
            context.prompt_template = "logic_fix"
        elif category == "configuration":
            context.prompt_template = "config_fix"
        else:
            context.prompt_template = "general_fix"

        # Set expected output format
        context.expected_output_format = "structured_patch"

        # Set provider preferences based on error complexity
        complexity = context.error_classification.potential_fix_complexity

        if complexity == "simple":
            context.provider_preferences = {
                "preferred_providers": ["openai", "anthropic"],
                "model_size": "small",
                "max_tokens": 1000,
            }
        elif complexity == "complex":
            context.provider_preferences = {
                "preferred_providers": ["anthropic", "openai"],
                "model_size": "large",
                "max_tokens": 4000,
            }
        else:
            context.provider_preferences = {
                "preferred_providers": ["openai", "anthropic"],
                "model_size": "medium",
                "max_tokens": 2000,
            }

    def _calculate_similarity(
        self, context1: ErrorContext, context2: ErrorContext
    ) -> float:
        """Calculate similarity between two error contexts."""
        similarity_score = 0.0

        # Language similarity
        if context1.language == context2.language:
            similarity_score += 0.3

        # Exception type similarity
        if context1.exception_type and context2.exception_type:
            if context1.exception_type == context2.exception_type:
                similarity_score += 0.3

        # Error message similarity (simple keyword matching)
        if context1.error_message and context2.error_message:
            words1 = set(context1.error_message.lower().split())
            words2 = set(context2.error_message.lower().split())
            if words1 and words2:
                common_words = words1.intersection(words2)
                similarity_score += 0.4 * (
                    len(common_words) / len(words1.union(words2))
                )

        return min(similarity_score, 1.0)

    def _create_comprehensive_prompt(self, context: LLMContext) -> Dict[str, Any]:
        """Create a comprehensive prompt for general analysis."""
        return {
            "template_type": "comprehensive",
            "system_prompt": "You are an expert software engineer tasked with analyzing and fixing code errors.",
            "user_prompt": f"""
Analyze the following error and provide a comprehensive fix:

ERROR DETAILS:
- Message: {context.error_context.error_message}
- Type: {context.error_context.exception_type or 'Unknown'}
- Language: {context.error_context.language.value}
- File: {context.error_context.file_path or 'Unknown'}
- Line: {context.error_context.line_number or 'Unknown'}

CLASSIFICATION:
- Category: {context.error_classification.category.value}
- Severity: {context.error_classification.severity.value}
- Root Cause: {context.error_classification.root_cause}
- Fix Complexity: {context.error_classification.potential_fix_complexity}

{self._format_additional_context(context)}

Please provide:
1. Root cause analysis
2. Recommended fix with code changes
3. Prevention strategies
4. Testing recommendations
""",
            "context": context.to_dict(),
            "max_tokens": context.provider_preferences.get("max_tokens", 2000),
        }

    def _create_focused_prompt(self, context: LLMContext) -> Dict[str, Any]:
        """Create a focused prompt for specific fixes."""
        return {
            "template_type": "focused",
            "system_prompt": f"You are a {context.error_context.language.value} expert. Fix this specific error concisely.",
            "user_prompt": f"""
Fix this {context.error_classification.category.value} error:

{context.error_context.error_message}

File: {context.error_context.file_path or 'Unknown'}
Line: {context.error_context.line_number or 'Unknown'}

Provide only the corrected code with minimal explanation.
""",
            "context": context.to_dict(),
            "max_tokens": context.provider_preferences.get("max_tokens", 1000),
        }

    def _create_patch_generation_prompt(self, context: LLMContext) -> Dict[str, Any]:
        """Create a prompt specifically for patch generation."""
        return {
            "template_type": "patch_generation",
            "system_prompt": "Generate a precise code patch to fix the identified error.",
            "user_prompt": f"""
Generate a patch file to fix this error:

ERROR: {context.error_context.error_message}
FILE: {context.error_context.file_path}
LINE: {context.error_context.line_number}
LANGUAGE: {context.error_context.language.value}

{self._format_source_context(context)}

Return the patch in unified diff format.
""",
            "context": context.to_dict(),
            "max_tokens": context.provider_preferences.get("max_tokens", 1500),
        }

    def _format_additional_context(self, context: LLMContext) -> str:
        """Format additional context information for prompts."""
        sections = []

        if context.similar_errors:
            sections.append("SIMILAR PREVIOUS ERRORS:")
            for error in context.similar_errors[:2]:  # Limit to 2 most similar
                sections.append(f"- {error['error_message'][:100]}...")

        if context.previous_fixes:
            sections.append("\nPREVIOUS SUCCESSFUL FIXES:")
            for fix in context.previous_fixes[:2]:  # Limit to 2 most recent
                sections.append("- Applied patch with success")

        if context.project_structure:
            sections.append(
                f"\nPROJECT TYPE: {context.project_structure.get('type', 'Unknown')}"
            )

        return "\n".join(sections) if sections else ""

    def _format_source_context(self, context: LLMContext) -> str:
        """Format source code context for prompts."""
        if context.error_context.source_code_snippet:
            return f"SOURCE CODE:\n```{context.error_context.language.value}\n{context.error_context.source_code_snippet}\n```"
        return ""

    def log_llm_failure(
        self,
        context_id: str,
        provider: str,
        error_type: str,
        error_message: str,
        attempt_number: int = 1,
        retry_delay: float = 0.0,
        request_data: Optional[Dict[str, Any]] = None,
        stack_trace: Optional[str] = None,
    ) -> str:
        """
        Log an LLM operation failure.

        Args:
            context_id: Context ID
            provider: LLM provider name
            error_type: Type of error
            error_message: Error message
            attempt_number: Attempt number
            retry_delay: Delay before retry
            request_data: Request data that failed
            stack_trace: Stack trace if available

        Returns:
            Failure ID
        """
        failure_id = str(uuid.uuid4())

        failure_log = FailureLog(
            failure_id=failure_id,
            context_id=context_id,
            timestamp=datetime.now().isoformat(),
            provider=provider,
            error_type=error_type,
            error_message=error_message,
            attempt_number=attempt_number,
            retry_delay=retry_delay,
            request_data=request_data or {},
            stack_trace=stack_trace,
        )

        # Add to context if it exists
        if context_id in self.contexts:
            context = self.contexts[context_id]
            if context.llm_failures is None:
                context.llm_failures = []
            context.llm_failures.append(failure_log)

        logger.warning(f"Logged LLM failure for context {context_id}: {error_message}")
        return failure_id

    def log_validation_failure(
        self,
        context_id: str,
        patch_id: str,
        validation_type: str,
        failure_reason: str,
        test_output: str = "",
        error_output: str = "",
        test_counts: Optional[Dict[str, int]] = None,
        execution_time: float = 0.0,
        retry_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Log a validation failure.

        Args:
            context_id: Context ID
            patch_id: Patch ID that failed validation
            validation_type: Type of validation (test, compile, lint, etc.)
            failure_reason: Reason for failure
            test_output: Test output
            error_output: Error output
            test_counts: Test counts
            execution_time: Execution time
            retry_context: Additional retry context

        Returns:
            Failure ID
        """
        failure_id = str(uuid.uuid4())

        validation_failure = FailureLog(
            failure_id=failure_id,
            context_id=context_id,
            patch_id=patch_id,
            timestamp=datetime.now().isoformat(),
            validation_type=validation_type,
            failure_reason=failure_reason,
            test_output=test_output,
            error_output=error_output,
            test_counts=test_counts or {},
            execution_time=execution_time,
            retry_context=retry_context,
        )

        # Add to context if it exists
        if context_id in self.contexts:
            context = self.contexts[context_id]
            if context.validation_failures is None:
                context.validation_failures = []
            context.validation_failures.append(validation_failure)

        logger.warning(
            f"Logged validation failure for context {context_id}, patch {patch_id}: {failure_reason}"
        )
        return failure_id

    def _analyze_common_failure_patterns(
        self, similar_contexts: List[LLMContext]
    ) -> List[Dict[str, Any]]:
        """
        Analyze common failure patterns from similar contexts.

        Args:
            similar_contexts: List of similar contexts

        Returns:
            List of common failure patterns
        """
        if not similar_contexts:
            return []

        patterns = []

        # Analyze LLM failures
        llm_error_types = defaultdict(int)
        llm_providers = defaultdict(int)

        for ctx in similar_contexts:
            if ctx.llm_failures:
                for failure in ctx.llm_failures:
                    llm_error_types[failure.error_type] += 1
                    llm_providers[failure.provider] += 1

        if llm_error_types:
            patterns.append(
                {
                    "type": "llm_failures",
                    "common_error_types": dict(llm_error_types),
                    "problematic_providers": dict(llm_providers),
                }
            )

        # Analyze validation failures
        validation_types = defaultdict(int)
        validation_reasons = defaultdict(int)

        for ctx in similar_contexts:
            if ctx.validation_failures:
                for failure in ctx.validation_failures:
                    validation_types[failure.validation_type] += 1
                    validation_reasons[failure.failure_reason] += 1

        if validation_types:
            patterns.append(
                {
                    "type": "validation_failures",
                    "common_validation_types": dict(validation_types),
                    "common_failure_reasons": dict(validation_reasons),
                }
            )

        return patterns


# Integration functions for orchestrator
def create_context_manager(storage_dir: Optional[Path] = None) -> LLMContextManager:
    """Create and return a configured LLM context manager."""
    return LLMContextManager(storage_dir=storage_dir)


def store_error_for_llm(
    context_manager: LLMContextManager,
    error_log: Dict[str, Any],
    analysis_result: Optional[Dict[str, Any]] = None,
    project_root: Optional[str] = None,
) -> str:
    """
    Convenience function to store error context for LLM processing.

    Args:
        context_manager: LLM context manager instance
        error_log: Error log entry
        analysis_result: Analysis results from other modules
        project_root: Project root directory

    Returns:
        Context ID for the stored context
    """
    # Convert error log to error context
    from .comprehensive_error_detector import create_error_context_from_log

    error_context = create_error_context_from_log(error_log, project_root)

    # Prepare additional analysis
    additional_analysis = analysis_result or {}

    # Store context
    return context_manager.store_error_context(
        error_context=error_context, additional_analysis=additional_analysis
    )


if __name__ == "__main__":
    # Test the LLM context manager
    print("LLM Context Manager Test")
    print("=======================")

    # Create context manager
    manager = LLMContextManager()

    # Create test error context
    from .comprehensive_error_detector import ErrorContext, LanguageType

    test_context = ErrorContext(
        error_message="NameError: name 'undefined_variable' is not defined",
        exception_type="NameError",
        language=LanguageType.PYTHON,
        file_path="test.py",
        line_number=10,
        function_name="main",
        service_name="test_service",
    )

    # Store context
    context_id = manager.store_error_context(test_context)
    print(f"Stored context: {context_id}")

    # Retrieve context
    retrieved_context = manager.get_context(context_id)
    print(f"Retrieved context: {retrieved_context.context_id}")

    # Prepare LLM prompt
    prompt_data = manager.prepare_llm_prompt(context_id, "comprehensive")
    print(f"Prepared prompt type: {prompt_data['template_type']}")
    print(f"Prompt length: {len(prompt_data['user_prompt'])} characters")

    # Get statistics
    stats = manager.get_context_statistics()
    print(f"Context statistics: {stats}")

    # Test similarity search
    similar_contexts = manager.search_similar_contexts(test_context)
    print(f"Found {len(similar_contexts)} similar contexts")
