"""
Rule configuration system for extensible rule-based error analysis.
"""

import json
import os
import re
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class RuleFormat(Enum):
    """Supported rule configuration formats."""

    JSON = "json"
    YAML = "yaml"
    PYTHON = "python"


class RuleCategory(Enum):
    """Categories of error analysis rules."""

    PYTHON = "python"
    FASTAPI = "fastapi"
    FLASK = "flask"
    DJANGO = "django"
    DATABASE = "database"
    NETWORK = "network"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    VALIDATION = "validation"
    CUSTOM = "custom"


class RuleSeverity(Enum):
    """Severity levels for errors."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class RuleConfidence(Enum):
    """Confidence levels for rule matches."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Rule:
    """
    Representation of an error analysis rule.
    """

    def __init__(
        self,
        pattern: str,
        type: str,
        description: str,
        root_cause: str,
        suggestion: str,
        category: Union[str, RuleCategory] = RuleCategory.PYTHON,
        severity: Union[str, RuleSeverity] = RuleSeverity.MEDIUM,
        confidence: Union[str, RuleConfidence] = RuleConfidence.HIGH,
        id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        examples: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a rule.

        Args:
            pattern: Regular expression pattern to match against errors
            type: Error type (e.g., "KeyError", "ValidationError")
            description: Human-readable description of the error
            root_cause: Identifier for the root cause of the error
            suggestion: Suggestion for fixing the error
            category: Category of the rule
            severity: Severity level of the error
            confidence: Confidence level when this rule matches
            id: Unique identifier for the rule
            tags: List of tags for organizing rules
            examples: List of example errors that this rule would match
            metadata: Additional metadata about the rule
        """
        self.pattern = pattern
        self.type = type
        self.description = description
        self.root_cause = root_cause
        self.suggestion = suggestion

        # Convert string enum values to enum instances if needed
        if isinstance(category, str):
            try:
                self.category = RuleCategory(category.lower())
            except ValueError:
                self.category = RuleCategory.CUSTOM
        else:
            self.category = category

        if isinstance(severity, str):
            try:
                self.severity = RuleSeverity(severity.lower())
            except ValueError:
                self.severity = RuleSeverity.MEDIUM
        else:
            self.severity = severity

        if isinstance(confidence, str):
            try:
                self.confidence = RuleConfidence(confidence.lower())
            except ValueError:
                self.confidence = RuleConfidence.MEDIUM
        else:
            self.confidence = confidence

        # Other fields
        self.id = (
            id or f"{self.category.value}_{self.type}_{hash(pattern) & 0xFFFFFFFF}"
        )
        self.tags = tags or []
        self.examples = examples or []
        self.metadata = metadata or {}

        # Compile the pattern
        self._compiled_pattern = re.compile(pattern)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the rule to a dictionary.

        Returns:
            Dictionary representation of the rule
        """
        return {
            "id": self.id,
            "pattern": self.pattern,
            "type": self.type,
            "description": self.description,
            "root_cause": self.root_cause,
            "suggestion": self.suggestion,
            "category": self.category.value,
            "severity": self.severity.value,
            "confidence": self.confidence.value,
            "tags": self.tags,
            "examples": self.examples,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Rule":
        """
        Create a rule from a dictionary.

        Args:
            data: Dictionary representation of a rule

        Returns:
            Rule instance

        Raises:
            ValueError: If required fields are missing
        """
        required_fields = ["pattern", "type", "description", "root_cause", "suggestion"]
        missing_fields = [field for field in required_fields if field not in data]

        if missing_fields:
            raise ValueError(
                f"Missing required fields for rule: {', '.join(missing_fields)}"
            )

        return cls(
            pattern=data["pattern"],
            type=data["type"],
            description=data["description"],
            root_cause=data["root_cause"],
            suggestion=data["suggestion"],
            category=data.get("category", RuleCategory.PYTHON),
            severity=data.get("severity", RuleSeverity.MEDIUM),
            confidence=data.get("confidence", RuleConfidence.HIGH),
            id=data.get("id"),
            tags=data.get("tags", []),
            examples=data.get("examples", []),
            metadata=data.get("metadata", {}),
        )

    def matches(self, text: str) -> Optional[re.Match]:
        """
        Check if the rule matches the given text.

        Args:
            text: Text to match against the rule's pattern

        Returns:
            Match object if there's a match, None otherwise
        """
        return self._compiled_pattern.search(text)


class RuleSet:
    """
    A collection of rules for error analysis.
    """

    def __init__(
        self, name: str, rules: Optional[List[Rule]] = None, description: str = ""
    ):
        """
        Initialize a rule set.

        Args:
            name: Name of the rule set
            rules: List of rules in the set
            description: Description of the rule set
        """
        self.name = name
        self.rules = rules or []
        self.description = description

    def add_rule(self, rule: Rule) -> None:
        """
        Add a rule to the set.

        Args:
            rule: Rule to add
        """
        self.rules.append(rule)

    def remove_rule(self, rule_id: str) -> bool:
        """
        Remove a rule from the set by ID.

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

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the rule set to a dictionary.

        Returns:
            Dictionary representation of the rule set
        """
        return {
            "name": self.name,
            "description": self.description,
            "rules": [rule.to_dict() for rule in self.rules],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RuleSet":
        """
        Create a rule set from a dictionary.

        Args:
            data: Dictionary representation of a rule set

        Returns:
            RuleSet instance

        Raises:
            ValueError: If required fields are missing
        """
        if "name" not in data:
            raise ValueError("Rule set must have a name")

        rule_set = cls(name=data["name"], description=data.get("description", ""))

        for rule_data in data.get("rules", []):
            try:
                rule = Rule.from_dict(rule_data)
                rule_set.add_rule(rule)
            except ValueError as e:
                # Suppress warnings during testing
                if not os.environ.get("TESTING") and not os.environ.get(
                    "PYTEST_CURRENT_TEST"
                ):
                    print(f"Warning: Skipping invalid rule: {e}")

        return rule_set


class RuleLoader:
    """
    Utility for loading rules from different sources.
    """

    @staticmethod
    def load_from_file(
        file_path: Union[str, Path], format: Optional[Union[str, RuleFormat]] = None
    ) -> RuleSet:
        """
        Load rules from a file.

        Args:
            file_path: Path to the rule file
            format: Format of the rule file (auto-detected if not specified)

        Returns:
            RuleSet instance with the loaded rules

        Raises:
            ValueError: If the file format is not supported
        """
        file_path = Path(file_path)

        # Auto-detect format if not specified
        if format is None:
            extension = file_path.suffix.lower()[1:]  # Remove the leading dot
            try:
                format = RuleFormat(extension)
            except ValueError:
                raise ValueError(f"Unsupported file format: {extension}")
        elif isinstance(format, str):
            try:
                format = RuleFormat(format.lower())
            except ValueError:
                raise ValueError(f"Unsupported file format: {format}")

        # Load the file based on its format
        if format == RuleFormat.JSON:
            with open(file_path, "r") as f:
                data = json.load(f)
        elif format == RuleFormat.YAML:
            try:
                import yaml

                with open(file_path, "r") as f:
                    data = yaml.safe_load(f)
            except ImportError:
                raise ImportError(
                    "PyYAML is required to load YAML rule files. Install it with pip install pyyaml."
                )
        elif format == RuleFormat.PYTHON:
            # Dynamically import the Python module
            import importlib.util

            spec = importlib.util.spec_from_file_location("rules_module", file_path)
            if spec is None:
                raise ValueError(f"Could not load module spec from {file_path}")
            module = importlib.util.module_from_spec(spec)
            if spec.loader is None:
                raise ValueError(f"Module spec has no loader for {file_path}")
            spec.loader.exec_module(module)

            # Get rule set from the module
            rule_set = getattr(module, "RULE_SET", None)
            if isinstance(rule_set, RuleSet):
                return rule_set

            # If there's no RULE_SET, try to create one from individual rules
            rules = getattr(module, "RULES", [])
            name = getattr(module, "NAME", file_path.stem)
            description = getattr(module, "DESCRIPTION", "")

            return RuleSet(name=name, rules=rules, description=description)
        else:
            raise ValueError(f"Unsupported format: {format}")

        # Create a rule set from the loaded data
        return RuleSet.from_dict(data)

    @staticmethod
    def load_from_directory(
        directory_path: Union[str, Path], recursive: bool = False
    ) -> List[RuleSet]:
        """
        Load all rule files from a directory.

        Args:
            directory_path: Path to the directory
            recursive: Whether to recursively load files from subdirectories

        Returns:
            List of RuleSet instances
        """
        directory_path = Path(directory_path)
        rule_sets: List[RuleSet] = []

        # Get list of rule files
        if recursive:
            rule_files: List[Path] = []
            for format_enum in RuleFormat:
                extension = f".{format_enum.value}"
                rule_files.extend(directory_path.glob(f"**/*{extension}"))
        else:
            rule_files: List[Path] = []
            for format_enum in RuleFormat:
                extension = f".{format_enum.value}"
                rule_files.extend(directory_path.glob(f"*{extension}"))

        # Load each file
        for file_path in rule_files:
            try:
                rule_set = RuleLoader.load_from_file(file_path)
                rule_sets.append(rule_set)
            except Exception as e:
                # Suppress warnings during testing
                if not os.environ.get("TESTING") and not os.environ.get(
                    "PYTEST_CURRENT_TEST"
                ):
                    print(f"Warning: Failed to load rule file {file_path}: {e}")

        return rule_sets


# Default rules directory
DEFAULT_RULES_DIR = Path(__file__).parent / "rules"
DEFAULT_RULES_DIR.mkdir(exist_ok=True)

# Create a subdirectory for each category
for category in RuleCategory:
    category_dir = DEFAULT_RULES_DIR / category.value
    category_dir.mkdir(exist_ok=True)


# Utility function to get all available rule sets
def get_all_rule_sets() -> List[RuleSet]:
    """
    Get all available rule sets from the default rules directory.

    Returns:
        List of RuleSet instances
    """
    return RuleLoader.load_from_directory(DEFAULT_RULES_DIR, recursive=True)


# Utility function to get rules for a specific category
def get_rules_for_category(category: Union[str, RuleCategory]) -> List[Rule]:
    """
    Get all rules for a specific category.

    Args:
        category: Category to get rules for

    Returns:
        List of Rule instances
    """
    if isinstance(category, str):
        try:
            category = RuleCategory(category.lower())
        except ValueError:
            category = RuleCategory.CUSTOM

    category_dir = DEFAULT_RULES_DIR / category.value

    if not category_dir.exists():
        return []

    rule_sets = RuleLoader.load_from_directory(category_dir)

    # Combine all rules from all rule sets
    all_rules = []
    for rule_set in rule_sets:
        all_rules.extend(rule_set.rules)

    return all_rules


def load_rule_configs() -> List[Dict[str, Any]]:
    """
    Load rule configurations from JSON files.

    Returns:
        List of rule configurations
    """
    import glob

    rules = []

    # Look for rule files in the default rules directory
    rule_files = glob.glob(str(DEFAULT_RULES_DIR / "**" / "*.json"), recursive=True)

    for rule_file in rule_files:
        try:
            with open(rule_file, "r") as f:
                file_data = json.load(f)

                # If the file contains a list of rules, add them all
                if isinstance(file_data, list):
                    rules.extend(file_data)
                # If it's a rule set, extract the rules
                elif isinstance(file_data, dict) and "rules" in file_data:
                    rules.extend(file_data["rules"])
                # If it's a single rule, add it
                elif isinstance(file_data, dict) and "id" in file_data:
                    rules.append(file_data)

        except Exception as e:
            # Suppress warnings during testing
            if not os.environ.get("TESTING") and not os.environ.get(
                "PYTEST_CURRENT_TEST"
            ):
                print(f"Warning: Failed to load rule file {rule_file}: {e}")

    return rules


# Convert predefined error patterns to the new format
def convert_legacy_patterns(
    patterns: List[Dict[str, str]], category: RuleCategory = RuleCategory.PYTHON
) -> List[Rule]:
    """
    Convert legacy error patterns to Rule objects.

    Args:
        patterns: List of legacy pattern dictionaries
        category: Category for the rules

    Returns:
        List of Rule objects
    """
    rules = []

    for pattern_dict in patterns:
        rule = Rule(
            pattern=pattern_dict["pattern"],
            type=pattern_dict["type"],
            description=pattern_dict["description"],
            root_cause=pattern_dict["root_cause"],
            suggestion=pattern_dict["suggestion"],
            category=category,
        )
        rules.append(rule)

    return rules


if __name__ == "__main__":
    # Example usage
    from rule_based import ERROR_PATTERNS, FASTAPI_ERROR_PATTERNS

    # Convert legacy patterns to rules
    python_rules = convert_legacy_patterns(ERROR_PATTERNS, RuleCategory.PYTHON)
    fastapi_rules = convert_legacy_patterns(
        FASTAPI_ERROR_PATTERNS, RuleCategory.FASTAPI
    )

    # Create rule sets
    python_rule_set = RuleSet(
        name="Python Common Errors",
        rules=python_rules,
        description="Common Python exceptions and errors",
    )

    fastapi_rule_set = RuleSet(
        name="FastAPI Errors",
        rules=fastapi_rules,
        description="FastAPI-specific errors and exceptions",
    )

    # Export as JSON
    python_rules_file = (
        DEFAULT_RULES_DIR / RuleCategory.PYTHON.value / "common_errors.json"
    )
    with open(python_rules_file, "w") as f:
        json.dump(python_rule_set.to_dict(), f, indent=2)

    fastapi_rules_file = (
        DEFAULT_RULES_DIR / RuleCategory.FASTAPI.value / "common_errors.json"
    )
    with open(fastapi_rules_file, "w") as f:
        json.dump(fastapi_rule_set.to_dict(), f, indent=2)

    print(f"Exported Python rules to {python_rules_file}")
    print(f"Exported FastAPI rules to {fastapi_rules_file}")

    # Test loading rules
    loaded_python_rules = RuleLoader.load_from_file(python_rules_file)
    print(f"Loaded {len(loaded_python_rules.rules)} Python rules")

    # Print some rule information
    for i, rule in enumerate(loaded_python_rules.rules[:3]):
        print(f"\nRule {i + 1}:")
        print(f"  ID: {rule.id}")
        print(f"  Type: {rule.type}")
        print(f"  Pattern: {rule.pattern}")
        print(f"  Description: {rule.description}")
        print(f"  Suggestion: {rule.suggestion}")
        print(f"  Category: {rule.category.value}")
        print(f"  Confidence: {rule.confidence.value}")
