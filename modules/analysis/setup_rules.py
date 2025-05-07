#!/usr/bin/env python3
"""
Setup script for initializing and verifying the rule system.
"""
import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Add parent directory to sys.path
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

try:
    # Try direct import first
    from rule_config import (
        RuleCategory, RuleLoader, get_all_rule_sets, DEFAULT_RULES_DIR
    )
    from rule_categories import (
        detect_rule_conflicts, RuleDependency, upgrade_rule_to_enhanced
    )
    from rule_confidence import ContextualRuleAnalyzer
except ImportError:
    try:
        # Try relative import
        sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
        from modules.analysis.rule_config import (
            RuleCategory, RuleLoader, get_all_rule_sets, DEFAULT_RULES_DIR
        )
        from modules.analysis.rule_categories import (
            detect_rule_conflicts, RuleDependency, upgrade_rule_to_enhanced
        )
        from modules.analysis.rule_confidence import ContextualRuleAnalyzer
    except ImportError as e:
        print(f"Error importing required modules: {e}")
        print("\nDependency problem detected. Try installing required packages:")
        print("pip install pyyaml")
        sys.exit(1)


def setup_rule_directories() -> None:
    """Create rule directories for all categories."""
    print(f"Setting up rule directories in {DEFAULT_RULES_DIR}")
    
    # Create base rules directory
    DEFAULT_RULES_DIR.mkdir(exist_ok=True)
    
    # Create a directory for each category
    for category in RuleCategory:
        category_dir = DEFAULT_RULES_DIR / category.value
        category_dir.mkdir(exist_ok=True)
        print(f"  Created directory: {category_dir}")


def verify_rule_conflicts() -> List[Dict[str, Any]]:
    """
    Verify rules for conflicts.
    
    Returns:
        List of conflict dictionaries
    """
    print("Checking for rule conflicts...")
    
    # Load all rules
    rule_sets = get_all_rule_sets()
    all_rules = []
    
    for rule_set in rule_sets:
        all_rules.extend(rule_set.rules)
    
    # Check for conflicts
    conflicts = detect_rule_conflicts(all_rules)
    
    if conflicts:
        print(f"  Found {len(conflicts)} potential conflicts:")
        for i, conflict in enumerate(conflicts[:5], 1):
            print(f"  {i}. {conflict['rule1']} conflicts with {conflict['rule2']}: {conflict['reason']}")
        
        if len(conflicts) > 5:
            print(f"  ... and {len(conflicts) - 5} more conflicts.")
    else:
        print("  No conflicts found.")
    
    return conflicts


def verify_circular_dependencies() -> List[List[str]]:
    """
    Verify rules for circular dependencies.
    
    Returns:
        List of circular dependency chains
    """
    print("Checking for circular dependencies...")
    
    # Load all rules
    rule_sets = get_all_rule_sets()
    all_rules = []
    
    for rule_set in rule_sets:
        all_rules.extend(rule_set.rules)
    
    # Check for circular dependencies
    dependency_checker = RuleDependency(all_rules)
    circular_deps = dependency_checker.detect_circular_dependencies()
    
    if circular_deps:
        print(f"  Found {len(circular_deps)} circular dependency chains:")
        for i, chain in enumerate(circular_deps[:5], 1):
            print(f"  {i}. {' -> '.join(chain)}")
        
        if len(circular_deps) > 5:
            print(f"  ... and {len(circular_deps) - 5} more circular dependency chains.")
    else:
        print("  No circular dependencies found.")
    
    return circular_deps


def count_rules() -> Tuple[int, Dict[str, int]]:
    """
    Count rules by category.
    
    Returns:
        Tuple of (total_count, count_by_category)
    """
    print("Counting rules...")
    
    # Load all rules
    rule_sets = get_all_rule_sets()
    all_rules = []
    
    for rule_set in rule_sets:
        all_rules.extend(rule_set.rules)
    
    # Count by category
    by_category = {}
    for rule in all_rules:
        category = rule.category.value
        by_category[category] = by_category.get(category, 0) + 1
    
    total = len(all_rules)
    print(f"  Found {total} rules:")
    for category, count in sorted(by_category.items()):
        print(f"  - {category}: {count}")
    
    return total, by_category


def upgrade_rules() -> int:
    """
    Upgrade regular rules to enhanced rules.
    
    Returns:
        Number of rules upgraded
    """
    print("Upgrading rules to enhanced format...")
    
    # Load all rule sets
    rule_sets = get_all_rule_sets()
    upgraded = 0
    
    for rule_set in rule_sets:
        for rule in rule_set.rules:
            if not hasattr(rule, 'criticality'):
                # This is a regular rule, not an enhanced rule
                upgraded += 1
    
    if upgraded > 0:
        print(f"  Found {upgraded} rules to upgrade.")
        
        # Upgrade rules in each rule file
        for category in RuleCategory:
            category_dir = DEFAULT_RULES_DIR / category.value
            if not category_dir.exists():
                continue
            
            for file_path in category_dir.glob("*.json"):
                try:
                    # Load the rule set from file
                    rule_set = RuleLoader.load_from_file(file_path)
                    
                    # Check if any rules need to be upgraded
                    needs_upgrade = False
                    for rule in rule_set.rules:
                        if not hasattr(rule, 'criticality'):
                            needs_upgrade = True
                            break
                    
                    if needs_upgrade:
                        # Convert to enhanced rules
                        enhanced_rules = [
                            upgrade_rule_to_enhanced(rule) if not hasattr(rule, 'criticality') else rule
                            for rule in rule_set.rules
                        ]
                        
                        # Save back to file
                        with open(file_path, 'w') as f:
                            json.dump({
                                "name": rule_set.name,
                                "description": rule_set.description,
                                "rules": [rule.to_dict() for rule in enhanced_rules]
                            }, f, indent=2)
                        
                        print(f"  Upgraded rules in {file_path}")
                except Exception as e:
                    print(f"  Error upgrading rules in {file_path}: {e}")
    else:
        print("  All rules are already in enhanced format.")
    
    return upgraded


def verify_analyzer() -> bool:
    """
    Verify analyzer functionality.
    
    Returns:
        True if successful, False otherwise
    """
    print("Verifying analyzer functionality...")
    
    try:
        # Load all rules
        rule_sets = get_all_rule_sets()
        all_rules = []
        
        for rule_set in rule_sets:
            all_rules.extend(rule_set.rules)
        
        # Create analyzer
        analyzer = ContextualRuleAnalyzer(all_rules)
        
        # Test with a sample error
        error_text = "KeyError: 'user_id'"
        result = analyzer.analyze_error(error_text)
        
        if result.get("matched", False):
            print(f"  Successfully matched error: {error_text}")
            print(f"  Rule ID: {result.get('rule_id')}")
            print(f"  Confidence: {result.get('confidence')} ({result.get('confidence_score', 0.0):.2f})")
            return True
        else:
            print(f"  Warning: Failed to match error: {error_text}")
            return False
    except Exception as e:
        print(f"  Error verifying analyzer: {e}")
        return False


def create_status_report(output_path: Optional[Path] = None) -> None:
    """
    Create a status report of the rule system.
    
    Args:
        output_path: Path to write the report, or None to print to console
    """
    print("Creating rule system status report...")
    
    # Collect information
    total_rules, by_category = count_rules()
    conflicts = verify_rule_conflicts()
    circular_deps = verify_circular_dependencies()
    analyzer_ok = verify_analyzer()
    
    # Generate report
    report = {
        "total_rules": total_rules,
        "rules_by_category": by_category,
        "conflicts": conflicts,
        "circular_dependencies": circular_deps,
        "analyzer_functional": analyzer_ok,
        "rules_directory": str(DEFAULT_RULES_DIR),
        "status": "ok" if analyzer_ok and not circular_deps else "warning"
    }
    
    # Output report
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Status report written to {output_path}")
    else:
        print("\nRule System Status Report:")
        print(f"  Total Rules: {total_rules}")
        print(f"  Rules by Category: {by_category}")
        print(f"  Conflicts: {len(conflicts)}")
        print(f"  Circular Dependencies: {len(circular_deps)}")
        print(f"  Analyzer Functional: {analyzer_ok}")
        print(f"  Rules Directory: {DEFAULT_RULES_DIR}")
        print(f"  Status: {report['status']}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Setup and verify the rule system"
    )
    parser.add_argument(
        "--setup-dirs", action="store_true",
        help="Set up rule directories"
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Verify rule integrity"
    )
    parser.add_argument(
        "--upgrade", action="store_true",
        help="Upgrade rules to enhanced format"
    )
    parser.add_argument(
        "--report", action="store_true",
        help="Generate a status report"
    )
    parser.add_argument(
        "--output", "-o", type=str,
        help="Output file for the status report"
    )
    parser.add_argument(
        "--all", "-a", action="store_true",
        help="Perform all actions"
    )
    
    args = parser.parse_args()
    
    # If no arguments specified, show help
    if not any([args.setup_dirs, args.verify, args.upgrade, args.report, args.all]):
        parser.print_help()
        return
    
    # Perform actions
    if args.all or args.setup_dirs:
        setup_rule_directories()
        print()
    
    if args.all or args.upgrade:
        upgrade_rules()
        print()
    
    if args.all or args.verify:
        verify_rule_conflicts()
        print()
        verify_circular_dependencies()
        print()
        verify_analyzer()
        print()
    
    if args.all or args.report:
        output_path = Path(args.output) if args.output else None
        create_status_report(output_path)


if __name__ == "__main__":
    main()