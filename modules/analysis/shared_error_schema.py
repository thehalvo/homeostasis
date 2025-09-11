"""
Shared Error Schema Adaptations

This module provides a unified approach to error schema handling across different
backend languages. It ensures consistent error representation, validation, and
conversion between language-specific formats.
"""

import json
import logging
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .language_adapters import ErrorSchemaValidator

logger = logging.getLogger(__name__)

# Path to the schemas directory
SCHEMA_DIR = Path(__file__).parent / "schemas"


class SharedErrorSchema:
    """
    A central class for handling error schema operations across languages.

    This class serves as the authority for error schema validation, conversion,
    and schema registry. It provides a unified way to work with error schemas
    across different backend languages.
    """

    # Standard fields that should be present in all normalized errors
    STANDARD_FIELDS = {"error_id", "timestamp", "language", "error_type", "message"}

    # Optional fields that may be present in normalized errors
    OPTIONAL_FIELDS = {
        "language_version",
        "stack_trace",
        "framework",
        "framework_version",
        "runtime",
        "runtime_version",
        "context",
        "request",
        "user",
        "process",
        "environment",
        "additional_data",
        "error_code",
        "severity",
        "tags",
        "related_errors",
        "handled",
        "recovery_action",
        "platform",
    }

    def __init__(self, schema_path: Optional[str] = None):
        """
        Initialize the shared error schema.

        Args:
            schema_path: Optional path to a custom schema file
        """
        if schema_path is None:
            schema_path = SCHEMA_DIR / "error_schema.json"

        self.validator = ErrorSchemaValidator(schema_path)
        self.language_configs = self._load_language_configs()

    def validate(self, error_data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate error data against the standard schema.

        Args:
            error_data: Error data to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        return self.validator.validate(error_data)

    def is_normalized(self, error_data: Dict[str, Any]) -> bool:
        """
        Check if error data is already in the standard normalized format.

        Args:
            error_data: Error data to check

        Returns:
            True if the error data is in the standard format
        """
        # Check if required fields are present
        for field in self.STANDARD_FIELDS:
            if field not in error_data:
                return False

        # Check if any fields are not in standard or optional set
        for field in error_data:
            if field not in self.STANDARD_FIELDS and field not in self.OPTIONAL_FIELDS:
                return False

        return True

    def detect_language(self, error_data: Dict[str, Any]) -> str:
        """
        Detect the language from error data using heuristics.

        Args:
            error_data: Error data to analyze

        Returns:
            Detected language identifier or "unknown"
        """
        # If language is explicitly specified
        if "language" in error_data:
            return error_data["language"].lower()

        # Apply language-specific detection rules
        for language, config in self.language_configs.items():
            if self._matches_language_patterns(
                error_data, config.get("detection_patterns", [])
            ):
                return language

        # Check for common language-specific fields
        if any(
            k in error_data for k in ["exception_type", "traceback", "python_version"]
        ):
            return "python"
        elif any(k in error_data for k in ["name", "stack", "nodejs"]):
            return "javascript"
        elif any(
            k in error_data
            for k in ["exception_class", "stacktrace", "java_version", "jvm"]
        ):
            return "java"
        elif any(k in error_data for k in ["goroutine_id", "go_version"]):
            return "go"
        elif any(k in error_data for k in ["csharp_version", "dotnet_version"]):
            return "csharp"
        elif any(k in error_data for k in ["ruby_version", "backtrace"]):
            return "ruby"
        elif any(
            k in error_data for k in ["scala_version", "akka_version", "play_version"]
        ):
            return "scala"

        # Check for Scala-specific patterns in error type
        if "error_type" in error_data:
            error_type = str(error_data["error_type"])
            if (
                error_type.startswith("scala.")
                or error_type.startswith("akka.")
                or error_type.startswith("play.api.")
                or "MatchError" in error_type
            ):
                return "scala"

        return "unknown"

    def normalize_error(
        self, error_data: Dict[str, Any], language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Normalize error data to the standard format.

        Args:
            error_data: Error data in a language-specific format
            language: Optional language identifier (auto-detected if not specified)

        Returns:
            Error data in the standard format

        Raises:
            ValueError: If language cannot be determined or is not supported
        """
        # If already normalized, return as is
        if self.is_normalized(error_data):
            return error_data

        # Detect language if not provided
        if language is None:
            language = self.detect_language(error_data)

            if language == "unknown":
                raise ValueError(
                    "Could not detect language from error data, please specify explicitly"
                )

        language = language.lower()

        # Get the configuration for this language
        language_config = self.language_configs.get(language)
        if not language_config:
            raise ValueError(f"Unsupported language: {language}")

        # Create a normalized error structure
        normalized = {
            "error_id": str(uuid.uuid4()),
            "timestamp": error_data.get("timestamp", datetime.now().isoformat()),
            "language": language,
        }

        # Apply field mappings from configuration
        field_mappings = language_config.get("field_mappings", {})
        for standard_field, lang_field in field_mappings.items():
            if isinstance(lang_field, list):
                # Try multiple possible field names
                for field_name in lang_field:
                    if field_name in error_data:
                        normalized[standard_field] = error_data[field_name]
                        break
            elif lang_field in error_data:
                normalized[standard_field] = error_data[lang_field]

        # Apply stack trace normalization
        if "stack_trace" not in normalized and "stack_trace_fields" in language_config:
            normalized["stack_trace"] = self._normalize_stack_trace(
                error_data, language_config["stack_trace_fields"]
            )

        # Ensure required fields are present
        if "error_type" not in normalized:
            normalized["error_type"] = error_data.get("type", "Unknown")

        if "message" not in normalized:
            normalized["message"] = error_data.get("message", "")

        # Add language version if available
        for version_field in language_config.get("version_fields", []):
            if version_field in error_data:
                normalized["language_version"] = error_data[version_field]
                break

        # Add framework information if available
        for framework_field in language_config.get("framework_fields", []):
            if framework_field in error_data:
                normalized["framework"] = error_data[framework_field]

                # Check for framework version
                for version_field in language_config.get(
                    "framework_version_fields", []
                ):
                    if version_field in error_data:
                        normalized["framework_version"] = error_data[version_field]
                        break
                break

        # Add request information if available
        if "request" in error_data and isinstance(error_data["request"], dict):
            normalized["request"] = error_data["request"]

        # Add context information if available
        if "context" in error_data and isinstance(error_data["context"], dict):
            normalized["context"] = error_data["context"]

        # Add severity if available
        if "severity" in error_data or "level" in error_data:
            severity = error_data.get("severity", error_data.get("level", ""))
            normalized["severity"] = self._normalize_severity(severity, language_config)

        # Add handled flag if available
        if "handled" in error_data:
            normalized["handled"] = bool(error_data["handled"])

        # Add additional language-specific data
        additional_data = {}
        for key, value in error_data.items():
            if key not in normalized and key not in [
                "stack_trace",
                "request",
                "context",
                "severity",
                "level",
                "handled",
            ]:
                additional_data[key] = value

        if additional_data:
            normalized["additional_data"] = additional_data

        return normalized

    def denormalize_error(
        self, normalized_error: Dict[str, Any], target_language: str
    ) -> Dict[str, Any]:
        """
        Convert standard normalized error data back to a language-specific format.

        Args:
            normalized_error: Error data in the standard format
            target_language: Target language for conversion

        Returns:
            Error data in the language-specific format

        Raises:
            ValueError: If target language is not supported
        """
        target_language = target_language.lower()

        # Get the configuration for the target language
        language_config = self.language_configs.get(target_language)
        if not language_config:
            raise ValueError(f"Unsupported target language: {target_language}")

        # Create a language-specific error structure
        lang_error = {}

        # Apply reverse field mappings
        field_mappings = language_config.get("field_mappings", {})
        for standard_field, lang_field in field_mappings.items():
            if standard_field in normalized_error:
                if isinstance(lang_field, list) and len(lang_field) > 0:
                    # Use the first field name in the list
                    lang_error[lang_field[0]] = normalized_error[standard_field]
                else:
                    lang_error[lang_field] = normalized_error[standard_field]

        # Convert stack trace to language-specific format
        if "stack_trace" in normalized_error:
            stack_trace_format = language_config.get("stack_trace_format", "string")
            if stack_trace_format == "string":
                # Join the stack trace into a string
                if isinstance(normalized_error["stack_trace"], list):
                    if all(
                        isinstance(frame, str)
                        for frame in normalized_error["stack_trace"]
                    ):
                        lang_error["stack_trace"] = "\n".join(
                            normalized_error["stack_trace"]
                        )
                    elif all(
                        isinstance(frame, dict)
                        for frame in normalized_error["stack_trace"]
                    ):
                        # Convert structured frames to string representation
                        frame_strings = []
                        for frame in normalized_error["stack_trace"]:
                            frame_str = self._format_stack_frame(frame, language_config)
                            if frame_str:
                                frame_strings.append(frame_str)

                        if frame_strings:
                            lang_error["stack_trace"] = "\n".join(frame_strings)
            else:
                # Keep as structured data
                lang_error["stack_trace"] = normalized_error["stack_trace"]

        # Convert severity to language-specific format
        if "severity" in normalized_error:
            severity_mappings = language_config.get("severity_mappings", {})
            standard_severity = normalized_error["severity"].lower()

            if standard_severity in severity_mappings:
                lang_error["level"] = severity_mappings[standard_severity]
            else:
                lang_error["level"] = standard_severity

        # Add request information if available
        if "request" in normalized_error:
            lang_error["request"] = normalized_error["request"]

        # Add context information if available
        if "context" in normalized_error:
            lang_error["context"] = normalized_error["context"]

        # Add language version if available
        if "language_version" in normalized_error:
            version_field = language_config.get("version_fields", ["language_version"])[
                0
            ]
            lang_error[version_field] = normalized_error["language_version"]

        # Add framework information if available
        if "framework" in normalized_error:
            framework_field = language_config.get("framework_fields", ["framework"])[0]
            lang_error[framework_field] = normalized_error["framework"]

            if "framework_version" in normalized_error:
                version_field = language_config.get(
                    "framework_version_fields", ["framework_version"]
                )[0]
                lang_error[version_field] = normalized_error["framework_version"]

        # Add handled flag if available
        if "handled" in normalized_error:
            lang_error["handled"] = normalized_error["handled"]

        # Add additional data if available
        if "additional_data" in normalized_error:
            for key, value in normalized_error["additional_data"].items():
                if key not in lang_error:
                    lang_error[key] = value

        return lang_error

    def get_supported_languages(self) -> List[str]:
        """
        Get the list of supported languages.

        Returns:
            List of supported language identifiers
        """
        return list(self.language_configs.keys())

    def _load_language_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        Load language-specific error schema configurations.

        Returns:
            Dictionary mapping language IDs to configurations
        """
        config_file = SCHEMA_DIR / "language_configs.json"
        if not config_file.exists():
            # Create a default configuration if it doesn't exist
            configs = self._create_default_configs()

            # Save the configuration
            try:
                with open(config_file, "w") as f:
                    json.dump(configs, f, indent=2)
            except Exception as e:
                logger.warning(f"Could not save language configurations: {e}")

            return configs

        try:
            with open(config_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading language configurations: {e}")
            return self._create_default_configs()

    def _create_default_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        Create default language configurations.

        Returns:
            Dictionary mapping language IDs to default configurations
        """
        return {
            "python": {
                "field_mappings": {
                    "error_type": ["exception_type", "type"],
                    "message": "message",
                    "stack_trace": ["traceback", "stack_trace"],
                },
                "stack_trace_fields": ["traceback", "stack_trace"],
                "stack_trace_format": "list",
                "version_fields": ["python_version", "language_version"],
                "framework_fields": ["framework"],
                "framework_version_fields": ["framework_version"],
                "detection_patterns": [
                    "(?:Traceback \\(most recent call last\\):)",
                    '(?:File "[^"]+", line \\d+, in [^\\n]+)',
                    "(?:\\w+Error:|\\w+Exception:)",
                ],
                "severity_mappings": {
                    "debug": "DEBUG",
                    "info": "INFO",
                    "warning": "WARNING",
                    "error": "ERROR",
                    "critical": "CRITICAL",
                    "fatal": "FATAL",
                },
            },
            "javascript": {
                "field_mappings": {
                    "error_type": ["name", "type"],
                    "message": "message",
                    "stack_trace": ["stack", "stacktrace", "stack_trace"],
                },
                "stack_trace_fields": ["stack", "stacktrace", "stack_trace"],
                "stack_trace_format": "string",
                "version_fields": ["node_version", "js_version", "language_version"],
                "framework_fields": ["framework"],
                "framework_version_fields": ["framework_version"],
                "detection_patterns": [
                    "(?:at \\w+ \\([^)]+\\))",
                    "(?:TypeError|ReferenceError|SyntaxError):",
                ],
                "severity_mappings": {
                    "debug": "debug",
                    "info": "info",
                    "warning": "warn",
                    "error": "error",
                    "critical": "error",
                    "fatal": "fatal",
                },
            },
            "java": {
                "field_mappings": {
                    "error_type": ["exception_class", "exception_type", "type"],
                    "message": "message",
                    "stack_trace": ["stack_trace", "stacktrace"],
                },
                "stack_trace_fields": ["stack_trace", "stacktrace"],
                "stack_trace_format": "string",
                "version_fields": ["java_version", "language_version"],
                "framework_fields": ["framework"],
                "framework_version_fields": ["framework_version"],
                "detection_patterns": [
                    "(?:java\\.\\w+\\.\\w+Exception:)",
                    "(?:at [\\w\\.$]+\\([^)]+\\.java:\\d+\\))",
                    "(?:Caused by:)",
                ],
                "severity_mappings": {
                    "debug": "fine",
                    "info": "info",
                    "warning": "warning",
                    "error": "severe",
                    "critical": "severe",
                    "fatal": "severe",
                },
            },
            "go": {
                "field_mappings": {
                    "error_type": ["error_type", "type"],
                    "message": "message",
                    "stack_trace": ["stack_trace", "stacktrace"],
                },
                "stack_trace_fields": ["stack_trace", "stacktrace"],
                "stack_trace_format": "string",
                "version_fields": ["go_version", "language_version"],
                "framework_fields": ["framework"],
                "framework_version_fields": ["framework_version"],
                "detection_patterns": [
                    "(?:goroutine \\d+ \\[[^\\]]+\\]:)",
                    "(?:panic:)",
                    "(?:\\s+at .+\\.go:\\d+)",
                ],
                "severity_mappings": {
                    "debug": "debug",
                    "info": "info",
                    "warning": "warn",
                    "error": "error",
                    "critical": "panic",
                    "fatal": "fatal",
                },
            },
            "csharp": {
                "field_mappings": {
                    "error_type": ["exception_type", "type"],
                    "message": "message",
                    "stack_trace": ["stack_trace", "stacktrace"],
                },
                "stack_trace_fields": ["stack_trace", "stacktrace"],
                "stack_trace_format": "string",
                "version_fields": [
                    "csharp_version",
                    "dotnet_version",
                    "language_version",
                ],
                "framework_fields": ["framework"],
                "framework_version_fields": ["framework_version"],
                "detection_patterns": [
                    "(?:System\\.\\w+Exception:)",
                    "(?:at [\\w\\.]+ in [^:]+:\\d+)",
                ],
                "severity_mappings": {
                    "debug": "Debug",
                    "info": "Information",
                    "warning": "Warning",
                    "error": "Error",
                    "critical": "Critical",
                    "fatal": "Fatal",
                },
            },
            "ruby": {
                "field_mappings": {
                    "error_type": ["exception_class", "type"],
                    "message": "message",
                    "stack_trace": ["backtrace", "stack_trace"],
                },
                "stack_trace_fields": ["backtrace", "stack_trace"],
                "stack_trace_format": "list",
                "version_fields": ["ruby_version", "language_version"],
                "framework_fields": ["framework"],
                "framework_version_fields": ["framework_version"],
                "detection_patterns": ["(?:\\w+Error:)", "(?:from [^:]+:\\d+:in `.+')"],
                "severity_mappings": {
                    "debug": "DEBUG",
                    "info": "INFO",
                    "warning": "WARN",
                    "error": "ERROR",
                    "critical": "FATAL",
                    "fatal": "FATAL",
                },
            },
        }

    def _matches_language_patterns(
        self, error_data: Dict[str, Any], patterns: List[str]
    ) -> bool:
        """
        Check if error data matches any of the language detection patterns.

        Args:
            error_data: Error data to check
            patterns: List of regex patterns for language detection

        Returns:
            True if error data matches any pattern
        """
        # Create a consolidated string for pattern matching
        match_text = ""

        # Add error type and message
        if "error_type" in error_data:
            match_text += str(error_data["error_type"]) + " "
        if "exception_type" in error_data:
            match_text += str(error_data["exception_type"]) + " "
        if "type" in error_data:
            match_text += str(error_data["type"]) + " "
        if "name" in error_data:
            match_text += str(error_data["name"]) + " "
        if "message" in error_data:
            match_text += str(error_data["message"]) + " "

        # Add stack trace
        for field in ["stack_trace", "stacktrace", "traceback", "stack", "backtrace"]:
            if field in error_data:
                if isinstance(error_data[field], list):
                    match_text += "\n".join(str(item) for item in error_data[field])
                else:
                    match_text += str(error_data[field])

        # Check each pattern
        for pattern in patterns:
            if re.search(pattern, match_text):
                return True

        return False

    def _normalize_stack_trace(
        self, error_data: Dict[str, Any], stack_fields: List[str]
    ) -> List:
        """
        Normalize stack trace from various formats to the standard format.

        Args:
            error_data: Error data
            stack_fields: List of possible stack trace field names

        Returns:
            Normalized stack trace as a list
        """
        # Find the first available stack trace field
        stack_trace = None
        for field in stack_fields:
            if field in error_data:
                stack_trace = error_data[field]
                break

        if stack_trace is None:
            return []

        # If already a list, return as is
        if isinstance(stack_trace, list):
            return stack_trace

        # If a string, split into lines
        if isinstance(stack_trace, str):
            return stack_trace.splitlines()

        return []

    def _normalize_severity(
        self, severity: str, language_config: Dict[str, Any]
    ) -> str:
        """
        Normalize a severity level to the standard format.

        Args:
            severity: Severity level
            language_config: Language configuration

        Returns:
            Normalized severity level
        """
        severity = str(severity).lower()

        # Reverse lookup in severity mappings
        severity_mappings = language_config.get("severity_mappings", {})
        for standard, lang_specific in severity_mappings.items():
            if lang_specific.lower() == severity:
                return standard

        # Map common levels
        level_map = {
            "debug": "debug",
            "info": "info",
            "information": "info",
            "warn": "warning",
            "warning": "warning",
            "error": "error",
            "err": "error",
            "critical": "critical",
            "crit": "critical",
            "fatal": "fatal",
            "severe": "error",
            "emerg": "fatal",
            "alert": "critical",
            "notice": "info",
            "trace": "debug",
            "fine": "debug",
            "finer": "debug",
            "finest": "debug",
            "verbose": "debug",
            "config": "info",
        }

        return level_map.get(severity, "error")

    def _format_stack_frame(
        self, frame: Dict[str, Any], language_config: Dict[str, Any]
    ) -> str:
        """
        Format a stack frame according to language-specific conventions.

        Args:
            frame: Stack frame data
            language_config: Language configuration

        Returns:
            Formatted stack frame as a string
        """
        language = language_config.get("language", "unknown")

        if language == "java":
            # Format like Java stack trace
            package = frame.get("package", "")
            class_name = frame.get("class", "Unknown")
            method = frame.get("function", "unknown")
            file = frame.get("file", "Unknown.java")
            line_num = frame.get("line", "?")

            full_class = f"{package}.{class_name}" if package else class_name

            return f"    at {full_class}.{method}({file}:{line_num})"

        elif language == "go":
            # Format like Go stack trace
            package = frame.get("package", "")
            func_name = frame.get("function", "")
            file_path = frame.get("file", "")
            line_num = frame.get("line", "")

            func_full = f"{package}.{func_name}" if package else func_name
            return f"{func_full}()\n\t{file_path}:{line_num}"

        elif language == "javascript":
            # Format like JavaScript stack trace
            file_path = frame.get("file", "<unknown>")
            line_num = frame.get("line", "?")
            column = frame.get("column", "?")
            function = frame.get("function", "<anonymous>")

            return f"    at {function} ({file_path}:{line_num}:{column})"

        elif language == "python":
            # Format like Python traceback
            file_path = frame.get("file", "<unknown>")
            line_num = frame.get("line", "?")
            function = frame.get("function", "<unknown>")

            line = f'  File "{file_path}", line {line_num}, in {function}'

            if "context" in frame:
                line += f"\n    {frame['context']}"

            return line

        # Default format for other languages
        file_path = frame.get("file", "<unknown>")
        line_num = frame.get("line", "?")
        function = frame.get("function", "<unknown>")

        return f"{function} ({file_path}:{line_num})"


# Create a singleton instance for global use
shared_error_schema = SharedErrorSchema()


def detect_language(error_data: Dict[str, Any]) -> str:
    """
    Detect the language from error data using the global shared schema.

    Args:
        error_data: Error data to analyze

    Returns:
        Detected language identifier or "unknown"
    """
    return shared_error_schema.detect_language(error_data)


def normalize_error(
    error_data: Dict[str, Any], language: Optional[str] = None
) -> Dict[str, Any]:
    """
    Normalize error data using the global shared schema.

    Args:
        error_data: Error data in a language-specific format
        language: Optional language identifier (auto-detected if not specified)

    Returns:
        Error data in the standard format
    """
    return shared_error_schema.normalize_error(error_data, language)


def denormalize_error(
    normalized_error: Dict[str, Any], target_language: str
) -> Dict[str, Any]:
    """
    Denormalize error data using the global shared schema.

    Args:
        normalized_error: Error data in the standard format
        target_language: Target language for conversion

    Returns:
        Error data in the language-specific format
    """
    return shared_error_schema.denormalize_error(normalized_error, target_language)


def validate_error(error_data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Validate error data using the global shared schema.

    Args:
        error_data: Error data to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    return shared_error_schema.validate(error_data)


def get_supported_languages() -> List[str]:
    """
    Get the list of supported languages from the global shared schema.

    Returns:
        List of supported language identifiers
    """
    return shared_error_schema.get_supported_languages()


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Create the language configurations file if it doesn't exist
    schema = SharedErrorSchema()

    # Print supported languages
    languages = schema.get_supported_languages()
    print(f"Supported languages: {', '.join(languages)}")

    # Example error data for different languages
    examples = {
        "python": {
            "exception_type": "KeyError",
            "message": "'user_id'",
            "traceback": [
                "Traceback (most recent call last):",
                '  File "app.py", line 42, in get_user',
                "    user_id = data['user_id']",
                "KeyError: 'user_id'",
            ],
            "level": "ERROR",
            "python_version": "3.9.7",
        },
        "javascript": {
            "name": "TypeError",
            "message": "Cannot read property 'id' of undefined",
            "stack": "TypeError: Cannot read property 'id' of undefined\n    at getUserId (/app/src/utils.js:45:20)\n    at processRequest (/app/src/controllers/user.js:23:15)",
            "level": "error",
        },
        "java": {
            "exception_class": "java.lang.NullPointerException",
            "message": 'Cannot invoke "String.length()" because "str" is null',
            "stack_trace": 'java.lang.NullPointerException: Cannot invoke "String.length()" because "str" is null\n    at com.example.StringProcessor.processString(StringProcessor.java:42)\n    at com.example.Main.main(Main.java:25)',
            "level": "SEVERE",
        },
        "go": {
            "error_type": "runtime error",
            "message": "nil pointer dereference",
            "stack_trace": "goroutine 1 [running]:\nmain.processValue()\n\t/app/main.go:25\nmain.main()\n\t/app/main.go:12",
            "level": "error",
        },
    }

    # Test normalization for each language
    for lang, error_data in examples.items():
        print(f"\nTesting {lang} error:")
        try:
            # Normalize to standard format
            normalized = schema.normalize_error(error_data, lang)
            print(f"  Normalized error type: {normalized.get('error_type')}")
            print(f"  Normalized message: {normalized.get('message')}")
            print(f"  Normalized severity: {normalized.get('severity')}")

            # Validate the normalized error
            is_valid, error_msg = schema.validate(normalized)
            print(f"  Validation: {'Valid' if is_valid else 'Invalid'}")
            if error_msg:
                print(f"  Validation error: {error_msg}")

            # Convert back to language-specific format
            denormalized = schema.denormalize_error(normalized, lang)
            print(
                f"  Denormalized error type: {denormalized.get('error_type') or denormalized.get('name') or denormalized.get('exception_class')}"
            )

        except Exception as e:
            print(f"  Error processing {lang} example: {e}")

    # Test language detection
    print("\nTesting language detection:")
    for lang, error_data in examples.items():
        detected = schema.detect_language(error_data)
        print(f"  Example {lang}: detected as {detected}")
