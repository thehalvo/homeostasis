"""
Enhanced rule-based error analysis module.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .rule_config import (
    DEFAULT_RULES_DIR,
    Rule,
    RuleCategory,
    RuleSet,
    convert_legacy_patterns,
    get_all_rule_sets,
    get_rules_for_category,
)

# Legacy error patterns for backward compatibility
ERROR_PATTERNS = [
    {
        "pattern": r"KeyError: '?([^']*)'?",
        "type": "KeyError",
        "description": "Accessing a dictionary key that doesn't exist",
        "root_cause": "dict_key_not_exists",
        "suggestion": "Check if the key exists before accessing it",
    },
    {
        "pattern": r"IndexError: list index out of range",
        "type": "IndexError",
        "description": "Accessing a list index that is out of bounds",
        "root_cause": "list_index_out_of_bounds",
        "suggestion": "Check the list length before accessing an index",
    },
    {
        "pattern": r"AttributeError: '([^']*)' object has no attribute '([^']*)'",
        "type": "AttributeError",
        "description": "Accessing an attribute that doesn't exist on an object",
        "root_cause": "attribute_not_exists",
        "suggestion": "Check if the attribute exists before accessing it",
    },
    {
        "pattern": r"TypeError: '([^']*)' object is not (subscriptable|iterable|callable)",
        "type": "TypeError",
        "description": "Using an object in a way that is not supported",
        "root_cause": "type_not_supported",
        "suggestion": "Ensure the object is of the expected type before using it",
    },
    {
        "pattern": r"ValueError: invalid literal for int\(\) with base (\d+): '([^']*)'",
        "type": "ValueError",
        "description": "Converting a string to an integer that is not a valid integer",
        "root_cause": "invalid_int_conversion",
        "suggestion": "Add error handling when converting strings to integers",
    },
    {
        "pattern": r"ZeroDivisionError: division by zero",
        "type": "ZeroDivisionError",
        "description": "Dividing by zero",
        "root_cause": "division_by_zero",
        "suggestion": "Check if the denominator is zero before dividing",
    },
    {
        "pattern": r"FileNotFoundError: \[Errno 2\] No such file or directory: '([^']*)'",
        "type": "FileNotFoundError",
        "description": "Trying to open a file that doesn't exist",
        "root_cause": "file_not_found",
        "suggestion": "Check if the file exists before trying to open it",
    },
]

# FastAPI-specific error patterns
FASTAPI_ERROR_PATTERNS = [
    {
        "pattern": r"KeyError: '([^']*)'",
        "type": "KeyError",
        "description": "Accessing a dictionary key that doesn't exist in a FastAPI endpoint",
        "root_cause": "dict_key_not_exists",
        "suggestion": "Add error handling to check if the key exists before accessing it",
    },
    {
        "pattern": r"pydantic.error_wrappers.ValidationError",
        "type": "ValidationError",
        "description": "Request data failed Pydantic validation",
        "root_cause": "request_validation_error",
        "suggestion": "Ensure the request data matches the expected schema",
    },
]


class RuleBasedAnalyzer:
    """
    Enhanced analyzer that uses predefined rules to identify error patterns.
    """

    def __init__(
        self,
        additional_patterns: Optional[List[Dict[str, str]]] = None,
        categories: Optional[List[Union[str, RuleCategory]]] = None,
        rules_dir: Optional[Path] = None,
        load_from_files: bool = True,
    ):
        """
        Initialize the analyzer with error patterns.

        Args:
            additional_patterns: Additional error patterns to use (legacy format)
            categories: Categories of rules to load (defaults to all)
            rules_dir: Directory containing rule files (defaults to DEFAULT_RULES_DIR)
            load_from_files: Whether to load rules from files
        """
        self.rules = []

        # Load legacy patterns if there are no rule files
        create_default_rule_files = False
        rules_dir = rules_dir or DEFAULT_RULES_DIR

        # Check if rules files exist
        rule_files_exist = False
        for category_dir in rules_dir.glob("*"):
            if category_dir.is_dir() and any(category_dir.glob("*.json")):
                rule_files_exist = True
                break

        if not rule_files_exist:
            # Create default rule files from legacy patterns
            create_default_rule_files = True

        # Load rules from files if requested
        if load_from_files:
            if categories:
                # Load specific categories
                for category in categories:
                    self.rules.extend(get_rules_for_category(category))
            else:
                # Load all rule sets
                rule_sets = get_all_rule_sets()
                for rule_set in rule_sets:
                    self.rules.extend(rule_set.rules)

        # If no rules were loaded, convert legacy patterns
        if not self.rules:
            python_rules = convert_legacy_patterns(ERROR_PATTERNS, RuleCategory.PYTHON)
            self.rules.extend(python_rules)

            # If legacy patterns need to be stored
            if create_default_rule_files:
                # Create rule sets
                python_rule_set = RuleSet(
                    name="Python Common Errors",
                    rules=python_rules,
                    description="Common Python exceptions and errors",
                )

                # Export as JSON
                rules_dir.mkdir(exist_ok=True)
                python_dir = rules_dir / RuleCategory.PYTHON.value
                python_dir.mkdir(exist_ok=True)

                python_rules_file = python_dir / "common_errors.json"
                with open(python_rules_file, "w") as f:
                    json.dump(python_rule_set.to_dict(), f, indent=2)

        # Add additional patterns if provided
        if additional_patterns:
            additional_rules = convert_legacy_patterns(additional_patterns)
            self.rules.extend(additional_rules)

    def analyze_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze an error log entry and identify the root cause.

        Args:
            error_data: Error log data

        Returns:
            Analysis results including root cause and suggestions
        """
        # Extract error message
        error_message = error_data.get("message", "")

        # Extract traceback if available
        traceback = error_data.get("traceback", [])
        if isinstance(traceback, list) and traceback:
            traceback_str = "".join(traceback)
        else:
            traceback_str = str(traceback)

        # Extract exception type
        exception_type = error_data.get("exception_type", "")

        # Extract additional info from enhanced error logs
        error_details = error_data.get("error_details", {})
        if error_details:
            error_message += " " + error_details.get("message", "")
            exception_type = error_details.get("exception_type", exception_type)

            # Add detailed frames to traceback if available
            detailed_frames = error_details.get("detailed_frames", [])
            for frame in detailed_frames:
                file_info = f"{frame.get('file', '')}:{frame.get('line', '')} in {frame.get('function', '')}"
                traceback_str += file_info + "\n"

        # Analyze error message and traceback
        matched_rules = []

        # Try to match rules against error message and traceback
        for rule in self.rules:
            message_match = rule.matches(error_message)
            traceback_match = rule.matches(traceback_str)

            if message_match or traceback_match:
                match = message_match or traceback_match
                matched_rules.append((rule, match))

        # Return the best match (if any)
        if matched_rules:
            # Sort by confidence (high > medium > low)
            matched_rules.sort(
                key=lambda x: ["low", "medium", "high"].index(x[0].confidence.value),
                reverse=True,
            )
            best_rule, best_match = matched_rules[0]

            return {
                "error_data": error_data,
                "matched_pattern": best_rule.pattern,
                "root_cause": best_rule.root_cause,
                "description": best_rule.description,
                "suggestion": best_rule.suggestion,
                "match_groups": best_match.groups() if best_match.groups() else None,
                "confidence": best_rule.confidence.value,
                "severity": best_rule.severity.value,
                "category": best_rule.category.value,
                "rule_id": best_rule.id,
                "tags": best_rule.tags,
            }

        # If no pattern matches, try to make a best guess based on exception type
        if exception_type:
            # Find rules with matching exception type
            type_rules = [rule for rule in self.rules if rule.type == exception_type]

            if type_rules:
                # Use the first rule with matching type
                rule = type_rules[0]
                return {
                    "error_data": error_data,
                    "matched_pattern": None,
                    "root_cause": rule.root_cause,
                    "description": rule.description,
                    "suggestion": rule.suggestion,
                    "match_groups": None,
                    "confidence": "medium",
                    "severity": rule.severity.value,
                    "category": rule.category.value,
                    "rule_id": rule.id,
                    "tags": rule.tags,
                }

        # If no match, return a generic analysis
        return {
            "error_data": error_data,
            "matched_pattern": None,
            "root_cause": "unknown",
            "description": "Unknown error type",
            "suggestion": "Manual investigation required",
            "match_groups": None,
            "confidence": "low",
            "severity": "medium",
            "category": "unknown",
        }

    def analyze_errors(
        self, error_data_list: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple error log entries.

        Args:
            error_data_list: List of error log data

        Returns:
            List of analysis results
        """
        return [self.analyze_error(error_data) for error_data in error_data_list]

    def add_rule(self, rule: Rule) -> None:
        """
        Add a new rule to the analyzer.

        Args:
            rule: Rule to add
        """
        self.rules.append(rule)

    def remove_rule(self, rule_id: str) -> bool:
        """
        Remove a rule from the analyzer.

        Args:
            rule_id: ID of the rule to remove

        Returns:
            True if the rule was removed, False otherwise
        """
        for i, rule in enumerate(self.rules):
            if rule.id == rule_id:
                self.rules.pop(i)
                return True
        return False

    def get_rules_by_category(self, category: Union[str, RuleCategory]) -> List[Rule]:
        """
        Get rules by category.

        Args:
            category: Category to filter by

        Returns:
            List of rules in the specified category
        """
        if isinstance(category, str):
            try:
                category = RuleCategory(category.lower())
            except ValueError:
                category = RuleCategory.CUSTOM

        return [rule for rule in self.rules if rule.category == category]

    def get_rules_by_tag(self, tag: str) -> List[Rule]:
        """
        Get rules by tag.

        Args:
            tag: Tag to filter by

        Returns:
            List of rules with the specified tag
        """
        return [rule for rule in self.rules if tag in rule.tags]

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the loaded rules.

        Returns:
            Dictionary with rule statistics
        """
        categories = {}
        tags = set()
        severities = {}

        for rule in self.rules:
            # Count by category
            category = rule.category.value
            categories[category] = categories.get(category, 0) + 1

            # Collect tags
            tags.update(rule.tags)

            # Count by severity
            severity = rule.severity.value
            severities[severity] = severities.get(severity, 0) + 1

        return {
            "total_rules": len(self.rules),
            "categories": categories,
            "tags": sorted(list(tags)),
            "severities": severities,
        }


def create_fastapi_analyzer() -> RuleBasedAnalyzer:
    """
    Create an analyzer with FastAPI-specific patterns.

    Returns:
        RuleBasedAnalyzer configured for FastAPI
    """
    # First try to load from rule files
    fastapi_analyzer = RuleBasedAnalyzer(categories=[RuleCategory.FASTAPI])

    # If no rules were loaded, use legacy patterns
    if not fastapi_analyzer.rules:
        fastapi_analyzer = RuleBasedAnalyzer(additional_patterns=FASTAPI_ERROR_PATTERNS)

    return fastapi_analyzer


if __name__ == "__main__":
    # Example usage with the enhanced rule system
    print("Rule-Based Analyzer Demo")
    print("=======================")

    # Initialize the analyzer
    analyzer = RuleBasedAnalyzer()

    # Print statistics
    stats = analyzer.get_stats()
    print(f"\nLoaded {stats['total_rules']} rules:")
    for category, count in stats.get("categories", {}).items():
        print(f"- {category}: {count} rules")

    # Print rules by severity
    print("\nRules by severity:")
    for severity, count in stats.get("severities", {}).items():
        print(f"- {severity}: {count} rules")

    # Test with a sample error
    print("\nAnalyzing sample error:")
    error_data = {
        "timestamp": "2023-01-01T12:00:00",
        "service": "example_service",
        "level": "ERROR",
        "message": "KeyError: 'todo_id'",
        "exception_type": "KeyError",
        "traceback": [
            "Traceback (most recent call last):",
            "  ...",
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
    }

    analysis = analyzer.analyze_error(error_data)
    print(f"Rule ID: {analysis.get('rule_id', 'None')}")
    print(f"Category: {analysis.get('category', 'unknown')}")
    print(f"Root Cause: {analysis['root_cause']}")
    print(f"Description: {analysis['description']}")
    print(f"Suggestion: {analysis['suggestion']}")
    print(f"Confidence: {analysis['confidence']}")
    print(f"Severity: {analysis.get('severity', 'unknown')}")

    # Demo loading from different categories
    print("\nDemo with different categories:")

    # FastAPI-specific analyzer
    fastapi_analyzer = create_fastapi_analyzer()
    fastapi_stats = fastapi_analyzer.get_stats()
    print(f"\nFastAPI Analyzer: {fastapi_stats['total_rules']} rules loaded")

    # Create a custom rule
    from .rule_config import RuleSeverity

    custom_rule = Rule(
        pattern=r"PermissionError: \[Errno 13\] Permission denied: '([^']*)'",
        type="PermissionError",
        description="Insufficient permissions to access a file or directory",
        root_cause="permission_denied",
        suggestion="Check file permissions or run the application with elevated privileges",
        category=RuleCategory.CUSTOM,
        severity=RuleSeverity.HIGH,
        tags=["filesystem", "permissions"],
    )

    # Add the rule to the analyzer
    analyzer.add_rule(custom_rule)
    print(f"\nAdded custom rule: {custom_rule.id}")

    # Test with a permission error
    permission_error = {
        "timestamp": "2023-01-01T12:00:00",
        "service": "example_service",
        "level": "ERROR",
        "message": "PermissionError: [Errno 13] Permission denied: '/etc/passwd'",
        "exception_type": "PermissionError",
    }

    analysis = analyzer.analyze_error(permission_error)
    print("\nAnalysis of Permission Error:")
    print(f"Rule ID: {analysis.get('rule_id', 'None')}")
    print(f"Category: {analysis.get('category', 'unknown')}")
    print(f"Root Cause: {analysis['root_cause']}")
    print(f"Description: {analysis['description']}")
    print(f"Suggestion: {analysis['suggestion']}")
    print(f"Severity: {analysis.get('severity', 'unknown')}")
