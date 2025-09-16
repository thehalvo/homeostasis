#!/usr/bin/env python3
"""
Command-line interface for rule management and testing.
"""
import argparse
import json
import os
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .rule_categories import EnhancedRule, RuleDependency, detect_rule_conflicts
from .rule_confidence import ConfidenceScorer, ContextualRuleAnalyzer
from .rule_config import (
    DEFAULT_RULES_DIR,
    Rule,
    RuleCategory,
    RuleLoader,
    RuleSet,
    get_all_rule_sets,
)


class RuleStats:
    """
    Collects and analyzes statistics for rules.
    """

    def __init__(self, rules_dir: Optional[Path] = None):
        """
        Initialize with rules directory.

        Args:
            rules_dir: Directory containing rule files
        """
        self.rules_dir = rules_dir or DEFAULT_RULES_DIR
        self.stats = self._collect_stats()

    def _collect_stats(self) -> Dict[str, Any]:
        """
        Collect statistics about rules.

        Returns:
            Dictionary of statistics
        """
        all_rule_sets = get_all_rule_sets()
        all_rules = []

        for rule_set in all_rule_sets:
            all_rules.extend(rule_set.rules)

        # Basic counts
        by_category: Dict[str, int] = {}
        by_severity: Dict[str, int] = {}
        by_confidence: Dict[str, int] = {}
        tags = set()

        for rule in all_rules:
            # Count by category
            category = rule.category.value
            by_category[category] = by_category.get(category, 0) + 1

            # Count by severity
            severity = rule.severity.value
            by_severity[severity] = by_severity.get(severity, 0) + 1

            # Count by confidence
            confidence = rule.confidence.value
            by_confidence[confidence] = by_confidence.get(confidence, 0) + 1

            # Collect tags
            tags.update(rule.tags)

        # Enhanced stats for EnhancedRule instances
        enhanced_rules = [r for r in all_rules if isinstance(r, EnhancedRule)]
        by_criticality: Dict[str, int] = {}
        by_complexity: Dict[str, int] = {}
        by_reliability: Dict[str, int] = {}
        by_source: Dict[str, int] = {}
        by_type: Dict[str, int] = {}

        for rule in enhanced_rules:
            # Count by criticality
            if hasattr(rule, "criticality"):
                criticality = rule.criticality.value
                by_criticality[criticality] = by_criticality.get(criticality, 0) + 1

            # Count by complexity
            if hasattr(rule, "complexity"):
                complexity = rule.complexity.value
                by_complexity[complexity] = by_complexity.get(complexity, 0) + 1

            # Count by reliability
            if hasattr(rule, "reliability"):
                reliability = rule.reliability.value
                by_reliability[reliability] = by_reliability.get(reliability, 0) + 1

            # Count by source
            if hasattr(rule, "source"):
                source = rule.source.value
                by_source[source] = by_source.get(source, 0) + 1

            # Count by type
            if hasattr(rule, "rule_type"):
                rule_type = rule.rule_type.value
                by_type[rule_type] = by_type.get(rule_type, 0) + 1

        # Check for conflicts
        conflicts = detect_rule_conflicts(all_rules)

        # Check for circular dependencies in enhanced rules
        circular_deps = []
        if enhanced_rules:
            # Cast to proper type for RuleDependency
            rules_for_dependency: List[Union[Rule, EnhancedRule]] = list(enhanced_rules)
            dependency_checker = RuleDependency(rules_for_dependency)
            circular_deps = dependency_checker.detect_circular_dependencies()

        # Compile stats
        stats = {
            "total_rules": len(all_rules),
            "total_enhanced_rules": len(enhanced_rules),
            "total_rule_sets": len(all_rule_sets),
            "by_category": by_category,
            "by_severity": by_severity,
            "by_confidence": by_confidence,
            "tags": sorted(list(tags)),
            "conflicts": conflicts,
            "circular_dependencies": circular_deps,
        }

        # Add enhanced stats if available
        if enhanced_rules:
            stats.update(
                {
                    "by_criticality": by_criticality,
                    "by_complexity": by_complexity,
                    "by_reliability": by_reliability,
                    "by_source": by_source,
                    "by_type": by_type,
                }
            )

        return stats

    def print_summary(self) -> None:
        """Print a summary of rule statistics."""
        print("\n===== Rule Statistics =====")
        print(f"Total Rules: {self.stats['total_rules']}")
        print(f"Enhanced Rules: {self.stats['total_enhanced_rules']}")
        print(f"Rule Sets: {self.stats['total_rule_sets']}")

        print("\nRules by Category:")
        for category, count in sorted(self.stats["by_category"].items()):
            print(f"  {category}: {count}")

        print("\nRules by Severity:")
        for severity, count in sorted(self.stats["by_severity"].items()):
            print(f"  {severity}: {count}")

        print("\nRules by Confidence:")
        for confidence, count in sorted(self.stats["by_confidence"].items()):
            print(f"  {confidence}: {count}")

        # Print enhanced stats if available
        if self.stats["total_enhanced_rules"] > 0:
            print("\nEnhanced Rule Statistics:")

            if "by_criticality" in self.stats:
                print("Rules by Criticality:")
                for criticality, count in sorted(self.stats["by_criticality"].items()):
                    print(f"  {criticality}: {count}")

            if "by_complexity" in self.stats:
                print("\nRules by Complexity:")
                for complexity, count in sorted(self.stats["by_complexity"].items()):
                    print(f"  {complexity}: {count}")

            if "by_reliability" in self.stats:
                print("\nRules by Reliability:")
                for reliability, count in sorted(self.stats["by_reliability"].items()):
                    print(f"  {reliability}: {count}")

        # Show conflicts if any
        if self.stats["conflicts"]:
            print(f"\nPotential Rule Conflicts: {len(self.stats['conflicts'])}")
            for i, conflict in enumerate(self.stats["conflicts"][:5], 1):
                print(
                    f"  {i}. {conflict['rule1']} conflicts with {conflict['rule2']}: {conflict['reason']}"
                )

            if len(self.stats["conflicts"]) > 5:
                print(f"  ... and {len(self.stats['conflicts']) - 5} more conflicts.")

        # Show circular dependencies if any
        if self.stats["circular_dependencies"]:
            print(
                f"\nCircular Dependencies: {len(self.stats['circular_dependencies'])}"
            )
            for i, chain in enumerate(self.stats["circular_dependencies"][:3], 1):
                print(f"  {i}. {' -> '.join(chain)}")

            if len(self.stats["circular_dependencies"]) > 3:
                print(
                    f"  ... and {len(self.stats['circular_dependencies']) - 3} more circular dependency chains."
                )

        # Show top tags
        if self.stats["tags"]:
            print(f"\nTop Tags (total: {len(self.stats['tags'])}):")
            for tag in sorted(self.stats["tags"][:10]):
                print(f"  {tag}")

            if len(self.stats["tags"]) > 10:
                print(f"  ... and {len(self.stats['tags']) - 10} more tags.")

    def export_stats(self, output_path: Path) -> None:
        """
        Export statistics to a JSON file.

        Args:
            output_path: Path to write the JSON file
        """
        with open(output_path, "w") as f:
            json.dump(self.stats, f, indent=2)

        print(f"Statistics exported to {output_path}")


class RuleTestResult:
    """
    Results of testing a rule.
    """

    def __init__(
        self, rule: Union[Rule, EnhancedRule], test_string: str, expected_match: bool
    ):
        """
        Initialize with rule and test data.

        Args:
            rule: Rule to test
            test_string: String to test against the rule
            expected_match: Whether the rule is expected to match
        """
        self.rule = rule
        self.test_string = test_string
        self.expected_match = expected_match

        # Run the test
        self.match = rule.matches(test_string)
        self.actual_match = self.match is not None
        self.passed = self.actual_match == expected_match

        # Score the match if it exists
        self.confidence_score = None
        if self.match:
            rule_match = ConfidenceScorer.score_match(rule, self.match)
            self.confidence_score = rule_match.confidence_score


class RuleTester:
    """
    Tests rules against sample data.
    """

    def __init__(self, rules: Optional[List[Union[Rule, EnhancedRule]]] = None):
        """
        Initialize with optional rules.

        Args:
            rules: Rules to test
        """
        self.rules = rules or []

    def load_rules(
        self,
        rule_ids: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
    ) -> None:
        """
        Load rules for testing.

        Args:
            rule_ids: IDs of specific rules to load
            categories: Categories of rules to load
        """
        all_rule_sets = get_all_rule_sets()
        all_rules = []

        for rule_set in all_rule_sets:
            all_rules.extend(rule_set.rules)

        # Filter by ID if provided
        if rule_ids:
            self.rules = [r for r in all_rules if r.id in rule_ids]

        # Filter by category if provided
        elif categories:
            category_enums = []
            for cat in categories:
                try:
                    category_enums.append(RuleCategory(cat.lower()))
                except ValueError:
                    print(f"Warning: Unknown category '{cat}'")

            self.rules = [r for r in all_rules if r.category in category_enums]

        # Otherwise use all rules
        else:
            self.rules = all_rules

    def test_rule(
        self,
        rule: Union[Rule, EnhancedRule],
        test_string: str,
        expected_match: bool = True,
    ) -> RuleTestResult:
        """
        Test a rule against a test string.

        Args:
            rule: Rule to test
            test_string: String to test against the rule
            expected_match: Whether the rule is expected to match

        Returns:
            RuleTestResult with test results
        """
        return RuleTestResult(rule, test_string, expected_match)

    def test_with_examples(
        self, rules: Optional[List[Union[Rule, EnhancedRule]]] = None
    ) -> List[RuleTestResult]:
        """
        Test rules using their example strings.

        Args:
            rules: Rules to test, defaults to all loaded rules

        Returns:
            List of RuleTestResult objects
        """
        rules = rules or self.rules
        results = []

        for rule in rules:
            # Skip rules without examples
            if not hasattr(rule, "examples") or not rule.examples:
                continue

            # Test each example
            for example in rule.examples:
                result = self.test_rule(rule, example, True)
                results.append(result)

        return results

    def run_test_file(self, test_file: Path) -> List[RuleTestResult]:
        """
        Run tests from a test file.

        The test file should be a JSON file with the following format:
        {
            "tests": [
                {
                    "rule_id": "rule1",
                    "test_string": "Error: test",
                    "expected_match": true
                },
                ...
            ]
        }

        Args:
            test_file: Path to the test file

        Returns:
            List of RuleTestResult objects
        """
        # Load test file
        with open(test_file, "r") as f:
            test_data = json.load(f)

        results = []
        rule_map = {rule.id: rule for rule in self.rules}

        # Run tests
        for test in test_data.get("tests", []):
            rule_id = test.get("rule_id")
            test_string = test.get("test_string")
            expected_match = test.get("expected_match", True)

            if not rule_id or not test_string:
                print(f"Warning: Invalid test: {test}")
                continue

            rule = rule_map.get(rule_id)
            if not rule:
                print(f"Warning: Rule not found: {rule_id}")
                continue

            result = self.test_rule(rule, test_string, expected_match)
            results.append(result)

        return results

    def print_results(self, results: List[RuleTestResult]) -> None:
        """
        Print test results.

        Args:
            results: List of RuleTestResult objects
        """
        passed = sum(1 for r in results if r.passed)
        failed = len(results) - passed

        print("\n===== Rule Test Results =====")
        print(f"Total Tests: {len(results)}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")

        if failed > 0:
            print("\nFailed Tests:")
            for i, result in enumerate([r for r in results if not r.passed], 1):
                print(f"\n{i}. Rule: {result.rule.id}")
                print(f"   Pattern: {result.rule.pattern}")
                print(f"   Test String: {result.test_string}")
                print(f"   Expected Match: {result.expected_match}")
                print(f"   Actual Match: {result.actual_match}")

                if result.match:
                    print(f"   Match Groups: {result.match.groups()}")
                    if result.confidence_score is not None:
                        print(f"   Confidence Score: {result.confidence_score:.2f}")

    def export_results(self, results: List[RuleTestResult], output_path: Path) -> None:
        """
        Export test results to a JSON file.

        Args:
            results: List of RuleTestResult objects
            output_path: Path to write the JSON file
        """
        # Convert results to a serializable format
        serialized_results = []
        for result in results:
            serialized = {
                "rule_id": result.rule.id,
                "pattern": result.rule.pattern,
                "test_string": result.test_string,
                "expected_match": result.expected_match,
                "actual_match": result.actual_match,
                "passed": result.passed,
            }

            if result.match:
                serialized["match_groups"] = (
                    [g for g in result.match.groups()] if result.match.groups() else []
                )

                if result.confidence_score is not None:
                    serialized["confidence_score"] = result.confidence_score

            serialized_results.append(serialized)

        # Write to file
        with open(output_path, "w") as f:
            json.dump(
                {
                    "total_tests": len(results),
                    "passed": sum(1 for r in results if r.passed),
                    "failed": sum(1 for r in results if not r.passed),
                    "results": serialized_results,
                },
                f,
                indent=2,
            )

        print(f"Test results exported to {output_path}")


class RuleManager:
    """
    Manages rule creation, editing, and organization.
    """

    def __init__(self, rules_dir: Optional[Path] = None):
        """
        Initialize with rules directory.

        Args:
            rules_dir: Directory containing rule files
        """
        self.rules_dir = rules_dir or DEFAULT_RULES_DIR

    def show_rule(self, rule_id: str) -> Union[Rule, EnhancedRule, None]:
        """
        Show details of a specific rule.

        Args:
            rule_id: ID of the rule to show

        Returns:
            The rule if found, None otherwise
        """
        all_rule_sets = get_all_rule_sets()

        for rule_set in all_rule_sets:
            for rule in rule_set.rules:
                if rule.id == rule_id:
                    return rule

        return None

    def print_rule_details(self, rule: Union[Rule, EnhancedRule]) -> None:
        """
        Print details of a rule.

        Args:
            rule: Rule to print details for
        """
        print("\n===== Rule Details =====")
        print(f"ID: {rule.id}")
        print(f"Type: {rule.type}")
        print(f"Pattern: {rule.pattern}")
        print(f"Category: {rule.category.value}")
        print(f"Severity: {rule.severity.value}")
        print(f"Confidence: {rule.confidence.value}")
        print(f"\nDescription: {rule.description}")
        print(f"Root Cause: {rule.root_cause}")
        print(f"Suggestion: {rule.suggestion}")

        if rule.tags:
            print(f"\nTags: {', '.join(rule.tags)}")

        if hasattr(rule, "examples") and rule.examples:
            print("\nExamples:")
            for example in rule.examples:
                print(f"  - {example}")

        # Print enhanced rule details if available
        if isinstance(rule, EnhancedRule):
            print("\nEnhanced Properties:")
            if hasattr(rule, "criticality"):
                print(f"Criticality: {rule.criticality.value}")
            if hasattr(rule, "complexity"):
                print(f"Complexity: {rule.complexity.value}")
            if hasattr(rule, "reliability"):
                print(f"Reliability: {rule.reliability.value}")
            if hasattr(rule, "source"):
                print(f"Source: {rule.source.value}")
            if hasattr(rule, "rule_type"):
                print(f"Rule Type: {rule.rule_type.value}")
            if hasattr(rule, "dependencies") and rule.dependencies:
                print(f"Dependencies: {', '.join(rule.dependencies)}")

    def create_rule(
        self, rule_data: Dict[str, Any], rule_set_name: str, category: str
    ) -> Union[Rule, EnhancedRule]:
        """
        Create a new rule and add it to a rule set.

        Args:
            rule_data: Data for the new rule
            rule_set_name: Name of the rule set to add the rule to
            category: Category for the rule

        Returns:
            The newly created rule
        """
        # Ensure required fields are present
        required_fields = ["pattern", "type", "description", "root_cause", "suggestion"]
        missing_fields = [field for field in required_fields if field not in rule_data]

        if missing_fields:
            raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")

        # Set category
        try:
            category_enum = RuleCategory(category.lower())
        except ValueError:
            category_enum = RuleCategory.CUSTOM

        rule_data["category"] = category_enum

        # Create rule
        rule = EnhancedRule.from_dict(rule_data)

        # Determine rule set path
        category_dir = self.rules_dir / category_enum.value
        category_dir.mkdir(exist_ok=True)

        # Convert rule set name to a valid filename
        filename = rule_set_name.lower().replace(" ", "_").replace("-", "_")
        if not filename.endswith(".json"):
            filename += ".json"

        rule_set_path = category_dir / filename

        # Load existing rule set or create a new one
        if rule_set_path.exists():
            rule_set = RuleLoader.load_from_file(rule_set_path)

            # Check for duplicate rule IDs
            if any(r.id == rule.id for r in rule_set.rules):
                raise ValueError(f"Rule with ID '{rule.id}' already exists in rule set")

            rule_set.add_rule(rule)
        else:
            rule_set = RuleSet(
                name=rule_set_name,
                rules=[rule],
                description=f"Rules for {category} category",
            )

        # Save rule set
        with open(rule_set_path, "w") as f:
            json.dump(rule_set.to_dict(), f, indent=2)

        return rule

    def update_rule(
        self, rule_id: str, rule_data: Dict[str, Any]
    ) -> Union[Rule, EnhancedRule, None]:
        """
        Update an existing rule.

        Args:
            rule_id: ID of the rule to update
            rule_data: New data for the rule

        Returns:
            The updated rule if found, None otherwise
        """
        # Find the rule and its rule set
        all_rule_sets = get_all_rule_sets()

        for rule_set in all_rule_sets:
            for i, rule in enumerate(rule_set.rules):
                if rule.id == rule_id:
                    # Update rule data
                    updated_rule: Union[Rule, EnhancedRule]
                    if isinstance(rule, EnhancedRule):
                        updated_rule = EnhancedRule.from_dict(
                            {**rule.to_dict(), **rule_data}
                        )
                    else:
                        # Create a new Rule from the updated data
                        rule_dict = rule.to_dict()
                        rule_dict.update(rule_data)
                        updated_rule = Rule.from_dict(rule_dict)

                    # Replace rule in the rule set
                    rule_set.rules[i] = updated_rule

                    # Save the rule set
                    rule_set_file = None

                    # Find the rule set file
                    for category in RuleCategory:
                        category_dir = self.rules_dir / category.value
                        if not category_dir.exists():
                            continue

                        for file_path in category_dir.glob("*.json"):
                            try:
                                loaded_rule_set = RuleLoader.load_from_file(file_path)
                                if loaded_rule_set.name == rule_set.name:
                                    rule_set_file = file_path
                                    break
                            except Exception:
                                continue

                        if rule_set_file:
                            break

                    if rule_set_file:
                        with open(rule_set_file, "w") as f:
                            json.dump(rule_set.to_dict(), f, indent=2)

                        return updated_rule
                    else:
                        raise ValueError(
                            f"Could not find rule set file for rule ID '{rule_id}'"
                        )

        return None

    def delete_rule(self, rule_id: str) -> bool:
        """
        Delete a rule.

        Args:
            rule_id: ID of the rule to delete

        Returns:
            True if the rule was deleted, False otherwise
        """
        # Find the rule and its rule set
        all_rule_sets = get_all_rule_sets()

        for rule_set in all_rule_sets:
            for rule in rule_set.rules:
                if rule.id == rule_id:
                    # Remove rule from the rule set
                    rule_set.remove_rule(rule_id)

                    # Find the rule set file
                    rule_set_file = None

                    for category in RuleCategory:
                        category_dir = self.rules_dir / category.value
                        if not category_dir.exists():
                            continue

                        for file_path in category_dir.glob("*.json"):
                            try:
                                loaded_rule_set = RuleLoader.load_from_file(file_path)
                                if loaded_rule_set.name == rule_set.name:
                                    rule_set_file = file_path
                                    break
                            except Exception:
                                continue

                        if rule_set_file:
                            break

                    if rule_set_file:
                        # Save the rule set or delete it if empty
                        if rule_set.rules:
                            with open(rule_set_file, "w") as f:
                                json.dump(rule_set.to_dict(), f, indent=2)
                        else:
                            rule_set_file.unlink()

                        return True
                    else:
                        raise ValueError(
                            f"Could not find rule set file for rule ID '{rule_id}'"
                        )

        return False


class RuleAnalyzer:
    """
    Analyzes error messages using rules.
    """

    def __init__(self, rules: Optional[List[Union[Rule, EnhancedRule]]] = None):
        """
        Initialize with optional rules.

        Args:
            rules: Rules to use for analysis
        """
        if rules:
            self.analyzer = ContextualRuleAnalyzer(rules)
        else:
            # Load all rules
            all_rule_sets = get_all_rule_sets()
            all_rules = []

            for rule_set in all_rule_sets:
                all_rules.extend(rule_set.rules)

            self.analyzer = ContextualRuleAnalyzer(all_rules)

    def analyze_error(
        self, error_text: str, error_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze an error message.

        Args:
            error_text: Error message to analyze
            error_context: Additional context for the error

        Returns:
            Analysis results
        """
        return self.analyzer.analyze_error(error_text, error_context)

    def print_analysis(self, analysis: Dict[str, Any]) -> None:
        """
        Print analysis results.

        Args:
            analysis: Analysis results from analyze_error
        """
        print("\n===== Error Analysis =====")

        if not analysis.get("matched", False):
            print("No matching rule found.")
            print(f"Confidence: {analysis.get('confidence', 'low')}")
            return

        print(f"Rule ID: {analysis.get('rule_id')}")
        print(f"Type: {analysis.get('type')}")
        if "category" in analysis:
            print(f"Category: {analysis.get('category')}")
        print(
            f"Confidence: {analysis.get('confidence')} ({analysis.get('confidence_score', 0.0):.2f})"
        )

        print(f"\nMatched Text: {analysis.get('matched_text')}")
        if analysis.get("match_groups"):
            print(f"Match Groups: {analysis.get('match_groups')}")

        print(f"\nDescription: {analysis.get('description')}")
        print(f"Root Cause: {analysis.get('root_cause')}")
        print(f"Suggestion: {analysis.get('suggestion')}")

        if "severity" in analysis:
            print(f"\nSeverity: {analysis.get('severity')}")

        # Print confidence factors if available
        if "confidence_factors" in analysis:
            print("\nConfidence Factors:")
            for factor, score in analysis.get("confidence_factors", {}).items():
                print(f"  {factor}: {score:.2f}")

        # Print alternative matches if available
        if "alternative_matches" in analysis and analysis["alternative_matches"]:
            print("\nAlternative Matches:")
            for i, alt in enumerate(analysis["alternative_matches"], 1):
                print(
                    f"  {i}. {alt.get('rule_id')} - {alt.get('description')} (Confidence: {alt.get('confidence')})"
                )


def create_parser() -> argparse.ArgumentParser:
    """
    Create command-line argument parser.

    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(description="Rule management and testing CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # List rules command
    list_parser = subparsers.add_parser("list", help="List rules")
    list_parser.add_argument("--category", "-c", help="Filter by category")
    list_parser.add_argument("--tag", "-t", help="Filter by tag")
    list_parser.add_argument("--severity", "-s", help="Filter by severity")
    list_parser.add_argument("--confidence", help="Filter by confidence")
    list_parser.add_argument(
        "--format", "-f", choices=["text", "json"], default="text", help="Output format"
    )
    list_parser.add_argument("--output", "-o", help="Output file for JSON format")

    # Show rule command
    show_parser = subparsers.add_parser("show", help="Show rule details")
    show_parser.add_argument("rule_id", help="ID of the rule to show")

    # Create rule command
    create_parser = subparsers.add_parser("create", help="Create a new rule")
    create_parser.add_argument("--input", "-i", help="JSON file with rule data")
    create_parser.add_argument("--pattern", "-p", help="Regex pattern for the rule")
    create_parser.add_argument("--type", "-t", help="Error type")
    create_parser.add_argument("--description", "-d", help="Description of the error")
    create_parser.add_argument("--root-cause", "-r", help="Root cause identifier")
    create_parser.add_argument(
        "--suggestion", "-s", help="Suggestion for fixing the error"
    )
    create_parser.add_argument(
        "--category", "-c", default="custom", help="Category for the rule"
    )
    create_parser.add_argument("--severity", default="medium", help="Severity level")
    create_parser.add_argument(
        "--confidence", default="medium", help="Confidence level"
    )
    create_parser.add_argument("--tags", help="Comma-separated list of tags")
    create_parser.add_argument(
        "--examples", help="Comma-separated list of example errors"
    )
    create_parser.add_argument(
        "--rule-set",
        default="custom_rules",
        help="Name of the rule set to add the rule to",
    )

    # Update rule command
    update_parser = subparsers.add_parser("update", help="Update an existing rule")
    update_parser.add_argument("rule_id", help="ID of the rule to update")
    update_parser.add_argument("--input", "-i", help="JSON file with rule data")
    update_parser.add_argument("--pattern", "-p", help="Regex pattern for the rule")
    update_parser.add_argument("--type", "-t", help="Error type")
    update_parser.add_argument("--description", "-d", help="Description of the error")
    update_parser.add_argument("--root-cause", "-r", help="Root cause identifier")
    update_parser.add_argument(
        "--suggestion", "-s", help="Suggestion for fixing the error"
    )
    update_parser.add_argument("--category", "-c", help="Category for the rule")
    update_parser.add_argument("--severity", help="Severity level")
    update_parser.add_argument("--confidence", help="Confidence level")
    update_parser.add_argument("--tags", help="Comma-separated list of tags")
    update_parser.add_argument(
        "--examples", help="Comma-separated list of example errors"
    )

    # Delete rule command
    delete_parser = subparsers.add_parser("delete", help="Delete a rule")
    delete_parser.add_argument("rule_id", help="ID of the rule to delete")

    # Test rules command
    test_parser = subparsers.add_parser("test", help="Test rules")
    test_parser.add_argument("--rule-id", "-r", help="ID of the rule to test")
    test_parser.add_argument("--category", "-c", help="Category of rules to test")
    test_parser.add_argument("--test-file", "-f", help="JSON file with test data")
    test_parser.add_argument(
        "--error", "-e", help="Error string to test against all rules"
    )
    test_parser.add_argument(
        "--use-examples",
        "-x",
        action="store_true",
        help="Test rules using their example strings",
    )
    test_parser.add_argument("--output", "-o", help="Output file for test results")

    # Analyze error command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze an error message")
    analyze_parser.add_argument("error", help="Error message to analyze")
    analyze_parser.add_argument("--context", "-c", help="JSON file with error context")
    analyze_parser.add_argument(
        "--output", "-o", help="Output file for analysis results"
    )

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show rule statistics")
    stats_parser.add_argument("--output", "-o", help="Output file for statistics")

    return parser


def main() -> None:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        if args.command == "list":
            # Implement rule listing
            print("Listing rules...")
            # TODO: Implement rule listing

        elif args.command == "show":
            # Show rule details
            manager = RuleManager()
            rule = manager.show_rule(args.rule_id)

            if rule:
                manager.print_rule_details(rule)
            else:
                print(f"Rule with ID '{args.rule_id}' not found.")

        elif args.command == "create":
            # Create a new rule
            manager = RuleManager()

            # Get rule data from input file or command-line arguments
            if args.input:
                with open(args.input, "r") as f:
                    rule_data = json.load(f)
            else:
                # Ensure required fields are provided
                if (
                    not args.pattern
                    or not args.type
                    or not args.description
                    or not args.root_cause
                    or not args.suggestion
                ):
                    print(
                        "Error: Missing required fields. Use --input or provide --pattern, --type, --description, --root-cause, and --suggestion."
                    )
                    return

                rule_data = {
                    "pattern": args.pattern,
                    "type": args.type,
                    "description": args.description,
                    "root_cause": args.root_cause,
                    "suggestion": args.suggestion,
                    "severity": args.severity,
                    "confidence": args.confidence,
                }

                # Add optional fields
                if args.tags:
                    rule_data["tags"] = [tag.strip() for tag in args.tags.split(",")]

                if args.examples:
                    rule_data["examples"] = [
                        example.strip() for example in args.examples.split(",")
                    ]

            try:
                rule = manager.create_rule(rule_data, args.rule_set, args.category)
                print(f"Rule '{rule.id}' created successfully.")
            except ValueError as e:
                print(f"Error creating rule: {e}")

        elif args.command == "update":
            # Update an existing rule
            manager = RuleManager()

            # Get rule data from input file or command-line arguments
            if args.input:
                with open(args.input, "r") as f:
                    rule_data = json.load(f)
            else:
                rule_data = {}

                # Add provided fields
                if args.pattern:
                    rule_data["pattern"] = args.pattern
                if args.type:
                    rule_data["type"] = args.type
                if args.description:
                    rule_data["description"] = args.description
                if args.root_cause:
                    rule_data["root_cause"] = args.root_cause
                if args.suggestion:
                    rule_data["suggestion"] = args.suggestion
                if args.category:
                    rule_data["category"] = args.category
                if args.severity:
                    rule_data["severity"] = args.severity
                if args.confidence:
                    rule_data["confidence"] = args.confidence
                if args.tags:
                    rule_data["tags"] = [tag.strip() for tag in args.tags.split(",")]
                if args.examples:
                    rule_data["examples"] = [
                        example.strip() for example in args.examples.split(",")
                    ]

            try:
                rule = manager.update_rule(args.rule_id, rule_data)

                if rule:
                    print(f"Rule '{args.rule_id}' updated successfully.")
                else:
                    print(f"Rule with ID '{args.rule_id}' not found.")
            except ValueError as e:
                print(f"Error updating rule: {e}")

        elif args.command == "delete":
            # Delete a rule
            manager = RuleManager()

            try:
                success = manager.delete_rule(args.rule_id)

                if success:
                    print(f"Rule '{args.rule_id}' deleted successfully.")
                else:
                    print(f"Rule with ID '{args.rule_id}' not found.")
            except ValueError as e:
                print(f"Error deleting rule: {e}")

        elif args.command == "test":
            # Test rules
            tester = RuleTester()

            # Load rules
            if args.rule_id:
                tester.load_rules(rule_ids=[args.rule_id])
            elif args.category:
                tester.load_rules(categories=[args.category])
            else:
                tester.load_rules()

            # Run tests
            results = []

            if args.test_file:
                results = tester.run_test_file(Path(args.test_file))
            elif args.error:
                # Test against all rules
                for rule in tester.rules:
                    result = tester.test_rule(rule, args.error)
                    results.append(result)
            elif args.use_examples:
                results = tester.test_with_examples()
            else:
                print(
                    "Error: No test method specified. Use --test-file, --error, or --use-examples."
                )
                return

            # Print results
            tester.print_results(results)

            # Export results if requested
            if args.output:
                tester.export_results(results, Path(args.output))

        elif args.command == "analyze":
            # Analyze an error message
            analyzer = RuleAnalyzer()

            # Load error context if provided
            error_context = None
            if args.context:
                with open(args.context, "r") as f:
                    error_context = json.load(f)

            # Analyze the error
            analysis = analyzer.analyze_error(args.error, error_context or {})

            # Print analysis
            analyzer.print_analysis(analysis)

            # Export analysis if requested
            if args.output:
                with open(args.output, "w") as f:
                    json.dump(analysis, f, indent=2)

        elif args.command == "stats":
            # Show rule statistics
            stats = RuleStats()
            stats.print_summary()

            # Export statistics if requested
            if args.output:
                stats.export_stats(Path(args.output))

    except Exception as e:
        print(f"Error: {e}")
        if os.environ.get("DEBUG"):
            traceback.print_exc()


if __name__ == "__main__":
    main()
