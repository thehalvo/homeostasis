"""
Enhanced rule categorization system for error analysis rules.
"""

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from .rule_config import Rule, RuleCategory, RuleConfidence, RuleSeverity


class RuleCriticality(Enum):
    """Criticality levels for rule violations."""

    CRITICAL = "critical"  # Requires immediate attention, could lead to data loss or security issues
    HIGH = "high"  # Requires prompt attention, affects core functionality
    MEDIUM = "medium"  # Should be fixed, but not urgent
    LOW = "low"  # Minor issue, can be addressed later


class RuleComplexity(Enum):
    """Complexity levels for rule fixes."""

    SIMPLE = "simple"  # Simple fix, usually one-line change
    MODERATE = "moderate"  # Moderate fix, multiple lines or logic changes
    COMPLEX = "complex"  # Complex fix, requires architectural changes
    UNKNOWN = "unknown"  # Complexity cannot be determined automatically


class RuleReliability(Enum):
    """Reliability levels for rule detection."""

    HIGH = "high"  # Consistently detects the issue
    MEDIUM = "medium"  # Detects most instances of the issue
    LOW = "low"  # May miss some instances or produce false positives


class RuleSource(Enum):
    """Sources of rule definitions."""

    BUILT_IN = "built_in"  # Built-in rule
    COMMUNITY = "community"  # Community-contributed rule
    USER = "user"  # User-defined rule
    THIRD_PARTY = "third_party"  # Third-party rule


class RuleType(Enum):
    """Types of rules."""

    ERROR_DETECTION = "error_detection"  # Detects errors in logs or exceptions
    PERFORMANCE_ISSUE = "performance_issue"  # Detects performance issues
    SECURITY_VULNERABILITY = (
        "security_vulnerability"  # Detects security vulnerabilities
    )
    CODE_SMELL = "code_smell"  # Detects code quality issues
    BEST_PRACTICE = "best_practice"  # Enforces best practices


class EnhancedRule(Rule):
    """
    Enhanced rule with additional metadata for categorization.
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
        criticality: Union[str, RuleCriticality] = RuleCriticality.MEDIUM,
        complexity: Union[str, RuleComplexity] = RuleComplexity.MODERATE,
        reliability: Union[str, RuleReliability] = RuleReliability.MEDIUM,
        source: Union[str, RuleSource] = RuleSource.BUILT_IN,
        rule_type: Union[str, RuleType] = RuleType.ERROR_DETECTION,
        dependencies: Optional[List[str]] = None,
    ):
        """
        Initialize an enhanced rule.

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
            criticality: How critical the rule violation is
            complexity: Complexity of the fix
            reliability: Reliability of rule detection
            source: Source of the rule
            rule_type: Type of rule
            dependencies: IDs of rules that must match before this rule
        """
        super().__init__(
            pattern=pattern,
            type=type,
            description=description,
            root_cause=root_cause,
            suggestion=suggestion,
            category=category,
            severity=severity,
            confidence=confidence,
            id=id,
            tags=tags,
            examples=examples,
            metadata=metadata,
        )

        # Convert string enum values to enum instances if needed
        if isinstance(criticality, str):
            try:
                self.criticality = RuleCriticality(criticality.lower())
            except ValueError:
                self.criticality = RuleCriticality.MEDIUM
        else:
            self.criticality = criticality

        if isinstance(complexity, str):
            try:
                self.complexity = RuleComplexity(complexity.lower())
            except ValueError:
                self.complexity = RuleComplexity.MODERATE
        else:
            self.complexity = complexity

        if isinstance(reliability, str):
            try:
                self.reliability = RuleReliability(reliability.lower())
            except ValueError:
                self.reliability = RuleReliability.MEDIUM
        else:
            self.reliability = reliability

        if isinstance(source, str):
            try:
                self.source = RuleSource(source.lower())
            except ValueError:
                self.source = RuleSource.USER
        else:
            self.source = source

        if isinstance(rule_type, str):
            try:
                self.rule_type = RuleType(rule_type.lower())
            except ValueError:
                self.rule_type = RuleType.ERROR_DETECTION
        else:
            self.rule_type = rule_type

        # Rule dependencies
        self.dependencies = dependencies or []

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the enhanced rule to a dictionary.

        Returns:
            Dictionary representation of the rule
        """
        base_dict = super().to_dict()

        # Add enhanced fields
        base_dict.update(
            {
                "criticality": self.criticality.value,
                "complexity": self.complexity.value,
                "reliability": self.reliability.value,
                "source": self.source.value,
                "rule_type": self.rule_type.value,
                "dependencies": self.dependencies,
            }
        )

        return base_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EnhancedRule":
        """
        Create an enhanced rule from a dictionary.

        Args:
            data: Dictionary representation of a rule

        Returns:
            EnhancedRule instance

        Raises:
            ValueError: If required fields are missing
        """
        # Check for required fields
        required_fields = ["pattern", "type", "description", "root_cause", "suggestion"]
        missing_fields = [field for field in required_fields if field not in data]

        if missing_fields:
            raise ValueError(
                f"Missing required fields for rule: {', '.join(missing_fields)}"
            )

        # Extract enhanced fields with defaults
        criticality = data.get("criticality", RuleCriticality.MEDIUM.value)
        complexity = data.get("complexity", RuleComplexity.MODERATE.value)
        reliability = data.get("reliability", RuleReliability.MEDIUM.value)
        source = data.get("source", RuleSource.BUILT_IN.value)
        rule_type = data.get("rule_type", RuleType.ERROR_DETECTION.value)
        dependencies = data.get("dependencies", [])

        # Create instance with all fields
        return cls(
            pattern=data["pattern"],
            type=data["type"],
            description=data["description"],
            root_cause=data["root_cause"],
            suggestion=data["suggestion"],
            category=data.get("category", RuleCategory.PYTHON.value),
            severity=data.get("severity", RuleSeverity.MEDIUM.value),
            confidence=data.get("confidence", RuleConfidence.HIGH.value),
            id=data.get("id"),
            tags=data.get("tags", []),
            examples=data.get("examples", []),
            metadata=data.get("metadata", {}),
            criticality=criticality,
            complexity=complexity,
            reliability=reliability,
            source=source,
            rule_type=rule_type,
            dependencies=dependencies,
        )

    def get_overall_score(self) -> float:
        """
        Calculate an overall score for the rule based on various factors.

        Returns:
            Score between 0.0 and 1.0
        """
        # Convert enums to numeric values
        severity_values = {
            RuleSeverity.CRITICAL: 1.0,
            RuleSeverity.HIGH: 0.8,
            RuleSeverity.MEDIUM: 0.6,
            RuleSeverity.LOW: 0.4,
            RuleSeverity.INFO: 0.2,
        }

        confidence_values = {
            RuleConfidence.HIGH: 1.0,
            RuleConfidence.MEDIUM: 0.6,
            RuleConfidence.LOW: 0.3,
        }

        criticality_values = {
            RuleCriticality.CRITICAL: 1.0,
            RuleCriticality.HIGH: 0.8,
            RuleCriticality.MEDIUM: 0.5,
            RuleCriticality.LOW: 0.3,
        }

        reliability_values = {
            RuleReliability.HIGH: 1.0,
            RuleReliability.MEDIUM: 0.7,
            RuleReliability.LOW: 0.4,
        }

        # Calculate weighted score
        severity_score = severity_values.get(self.severity, 0.5)
        confidence_score = confidence_values.get(self.confidence, 0.5)
        criticality_score = criticality_values.get(self.criticality, 0.5)
        reliability_score = reliability_values.get(self.reliability, 0.5)

        # Weights for different factors
        weights = {
            "severity": 0.3,
            "confidence": 0.3,
            "criticality": 0.2,
            "reliability": 0.2,
        }

        # Calculate weighted average
        weighted_score = (
            weights["severity"] * severity_score +
            weights["confidence"] * confidence_score +
            weights["criticality"] * criticality_score +
            weights["reliability"] * reliability_score
        )

        return weighted_score


class RuleDependency:
    """
    Handles rule dependencies and chain resolution.
    """

    def __init__(self, rules: List[Union[Rule, EnhancedRule]]):
        """
        Initialize with a list of rules.

        Args:
            rules: List of rules to analyze for dependencies
        """
        self.rules = rules
        self.rule_map = {rule.id: rule for rule in rules}

    def get_dependent_rules(self, rule_id: str) -> List[Union[Rule, EnhancedRule]]:
        """
        Get rules that depend on the given rule.

        Args:
            rule_id: ID of the rule to check dependencies for

        Returns:
            List of rules that depend on the specified rule
        """
        dependent_rules = []

        for rule in self.rules:
            if hasattr(rule, "dependencies") and rule_id in rule.dependencies:
                dependent_rules.append(rule)

        return dependent_rules

    def get_prerequisites(self, rule_id: str) -> List[Union[Rule, EnhancedRule]]:
        """
        Get rules that are prerequisites for the given rule.

        Args:
            rule_id: ID of the rule to check prerequisites for

        Returns:
            List of rules that are prerequisites for the specified rule
        """
        rule = self.rule_map.get(rule_id)
        if not rule or not hasattr(rule, "dependencies"):
            return []

        prerequisites = []
        for dep_id in rule.dependencies:
            if dep_id in self.rule_map:
                prerequisites.append(self.rule_map[dep_id])

        return prerequisites

    def detect_circular_dependencies(self) -> List[List[str]]:
        """
        Detect circular dependencies in the rule set.

        Returns:
            List of lists, where each inner list represents a circular dependency chain
        """
        circular_chains = []

        for rule in self.rules:
            if not hasattr(rule, "dependencies") or not rule.dependencies:
                continue

            # Check for circular dependencies starting from this rule
            visited = set()
            path = [rule.id]
            self._dfs_check_circular(rule.id, visited, path, circular_chains)

        return circular_chains

    def _dfs_check_circular(
        self,
        rule_id: str,
        visited: Set[str],
        path: List[str],
        circular_chains: List[List[str]],
    ):
        """
        Helper for circular dependency detection using DFS.

        Args:
            rule_id: Current rule ID
            visited: Set of visited rule IDs
            path: Current path of rule IDs
            circular_chains: List to store detected circular chains
        """
        visited.add(rule_id)

        rule = self.rule_map.get(rule_id)
        if not rule or not hasattr(rule, "dependencies"):
            return

        for dep_id in rule.dependencies:
            if dep_id not in self.rule_map:
                continue

            if dep_id in path:
                # Found a cycle
                cycle_start = path.index(dep_id)
                circular_chain = path[cycle_start:] + [dep_id]
                if circular_chain not in circular_chains:
                    circular_chains.append(circular_chain)
                continue

            if dep_id not in visited:
                path.append(dep_id)
                self._dfs_check_circular(dep_id, visited, path, circular_chains)
                path.pop()


def upgrade_rule_to_enhanced(rule: Rule) -> EnhancedRule:
    """
    Upgrade a regular Rule to an EnhancedRule.

    Args:
        rule: Regular Rule instance

    Returns:
        EnhancedRule instance with values copied from the regular rule
    """
    return EnhancedRule(
        pattern=rule.pattern,
        type=rule.type,
        description=rule.description,
        root_cause=rule.root_cause,
        suggestion=rule.suggestion,
        category=rule.category,
        severity=rule.severity,
        confidence=rule.confidence,
        id=rule.id,
        tags=rule.tags,
        examples=rule.examples,
        metadata=rule.metadata,
    )


def upgrade_rules_in_file(file_path: Union[str, Path]) -> None:
    """
    Upgrade rules in a JSON file from regular rules to enhanced rules.

    Args:
        file_path: Path to the rule file
    """
    import json

    from .rule_config import RuleLoader

    file_path = Path(file_path)

    # Load existing rules
    rule_set = RuleLoader.load_from_file(file_path)

    # Create an enhanced rule set
    enhanced_rules = []
    for rule in rule_set.rules:
        enhanced_rule = upgrade_rule_to_enhanced(rule)
        enhanced_rules.append(enhanced_rule)

    enhanced_rule_set = {
        "name": rule_set.name,
        "description": rule_set.description,
        "rules": [rule.to_dict() for rule in enhanced_rules],
    }

    # Write back to file
    with open(file_path, "w") as f:
        json.dump(enhanced_rule_set, f, indent=2)


def validate_rule_pattern(pattern: str) -> bool:
    """
    Validate a regular expression pattern.

    Args:
        pattern: Regular expression pattern to validate

    Returns:
        True if the pattern is valid, False otherwise
    """
    import re

    try:
        re.compile(pattern)
        return True
    except re.error:
        return False


def detect_rule_conflicts(
    rules: List[Union[Rule, EnhancedRule]],
) -> List[Dict[str, Any]]:
    """
    Detect potential conflicts between rules.

    Args:
        rules: List of rules to check for conflicts

    Returns:
        List of dictionaries describing conflicts, each with 'rule1', 'rule2', and 'reason' keys
    """
    conflicts = []

    # Check for duplicate patterns
    for i, rule1 in enumerate(rules):
        for j, rule2 in enumerate(rules[i + 1 :], i + 1):
            # Check for exact pattern matches
            if rule1.pattern == rule2.pattern:
                conflicts.append(
                    {
                        "rule1": rule1.id,
                        "rule2": rule2.id,
                        "type": "duplicate_pattern",
                        "reason": f"Rules have identical pattern: {rule1.pattern}",
                    }
                )
                continue

            # Check for overlapping patterns (more complex)
            # This is a simplistic check - might produce false positives
            if rule1.pattern.startswith("^") != rule2.pattern.startswith(
                "^"
            ) or rule1.pattern.endswith("$") != rule2.pattern.endswith("$"):
                continue  # Different anchoring, less likely to conflict

            # Check if one pattern is a subset of another
            pattern1_simplified = rule1.pattern.replace("\\", "").replace(".*", "")
            pattern2_simplified = rule2.pattern.replace("\\", "").replace(".*", "")

            if (
                pattern1_simplified in pattern2_simplified or
                pattern2_simplified in pattern1_simplified
            ):
                conflicts.append(
                    {
                        "rule1": rule1.id,
                        "rule2": rule2.id,
                        "type": "overlapping_patterns",
                        "reason": f"Rules may have overlapping patterns: '{rule1.pattern}' and '{rule2.pattern}'",
                    }
                )

    return conflicts


if __name__ == "__main__":
    # Example usage

    # Create a sample enhanced rule
    enhanced_rule = EnhancedRule(
        pattern=r"KeyError: '([^']*)'",
        type="KeyError",
        description="Dictionary key not found",
        root_cause="missing_dict_key",
        suggestion="Check if the key exists before accessing it",
        criticality=RuleCriticality.HIGH,
        complexity=RuleComplexity.SIMPLE,
        reliability=RuleReliability.HIGH,
    )

    # Print the rule details
    print(f"Rule ID: {enhanced_rule.id}")
    print(f"Pattern: {enhanced_rule.pattern}")
    print(f"Criticality: {enhanced_rule.criticality.value}")
    print(f"Complexity: {enhanced_rule.complexity.value}")
    print(f"Reliability: {enhanced_rule.reliability.value}")
    print(f"Overall Score: {enhanced_rule.get_overall_score():.2f}")

    # Create rules with dependencies
    rule1 = EnhancedRule(
        id="rule1",
        pattern=r"Error 1",
        type="Type1",
        description="Rule 1",
        root_cause="cause1",
        suggestion="Fix 1",
    )

    rule2 = EnhancedRule(
        id="rule2",
        pattern=r"Error 2",
        type="Type2",
        description="Rule 2",
        root_cause="cause2",
        suggestion="Fix 2",
        dependencies=["rule1"],
    )

    rule3 = EnhancedRule(
        id="rule3",
        pattern=r"Error 3",
        type="Type3",
        description="Rule 3",
        root_cause="cause3",
        suggestion="Fix 3",
        dependencies=["rule2"],
    )

    # Create a circular dependency
    rule4 = EnhancedRule(
        id="rule4",
        pattern=r"Error 4",
        type="Type4",
        description="Rule 4",
        root_cause="cause4",
        suggestion="Fix 4",
        dependencies=["rule3", "rule5"],
    )

    rule5 = EnhancedRule(
        id="rule5",
        pattern=r"Error 5",
        type="Type5",
        description="Rule 5",
        root_cause="cause5",
        suggestion="Fix 5",
        dependencies=["rule4"],
    )

    # Check for circular dependencies
    rules = [rule1, rule2, rule3, rule4, rule5]
    dependency_checker = RuleDependency(rules)
    circular_deps = dependency_checker.detect_circular_dependencies()

    print("\nCircular Dependencies:")
    for chain in circular_deps:
        print(" -> ".join(chain))

    # Check for rule conflicts
    conflict_rule1 = EnhancedRule(
        id="conflict1",
        pattern=r"Error: (.*)",
        type="Error",
        description="General error",
        root_cause="general_error",
        suggestion="Fix error",
    )

    conflict_rule2 = EnhancedRule(
        id="conflict2",
        pattern=r"Error: (.*)",
        type="Error",
        description="Specific error",
        root_cause="specific_error",
        suggestion="Fix specific error",
    )

    conflict_rules = [conflict_rule1, conflict_rule2]
    conflicts = detect_rule_conflicts(conflict_rules)

    print("\nRule Conflicts:")
    for conflict in conflicts:
        print(
            f"{conflict['rule1']} conflicts with {conflict['rule2']}: {conflict['reason']}"
        )
