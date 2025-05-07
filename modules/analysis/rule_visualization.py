#!/usr/bin/env python3
"""
Visualization tools for rule coverage and effectiveness statistics.
"""
import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import textwrap
import math

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
        RuleCriticality, RuleComplexity, RuleReliability, RuleSource, RuleType,
        detect_rule_conflicts, RuleDependency
    )
except ImportError:
    try:
        # Try relative import
        sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
        from modules.analysis.rule_config import (
            RuleCategory, RuleLoader, get_all_rule_sets, DEFAULT_RULES_DIR
        )
        from modules.analysis.rule_categories import (
            RuleCriticality, RuleComplexity, RuleReliability, RuleSource, RuleType,
            detect_rule_conflicts, RuleDependency
        )
    except ImportError as e:
        print(f"Error importing required modules: {e}")
        print("\nDependency problem detected. Try installing required packages:")
        print("pip install pyyaml")
        sys.exit(1)


class ASCIIBar:
    """
    Generates ASCII bar charts for terminal display.
    """
    
    @staticmethod
    def generate_horizontal_bar(value: int, max_value: int, width: int = 40, 
                              fill_char: str = '█', empty_char: str = '░') -> str:
        """
        Generate a horizontal bar chart.
        
        Args:
            value: Current value
            max_value: Maximum value
            width: Width of the bar in characters
            fill_char: Character to use for filled portion
            empty_char: Character to use for empty portion
            
        Returns:
            ASCII bar chart string
        """
        if max_value <= 0:
            return empty_char * width
        
        filled_width = min(width, int((value / max_value) * width))
        empty_width = width - filled_width
        
        return fill_char * filled_width + empty_char * empty_width
    
    @staticmethod
    def generate_horizontal_bar_with_label(label: str, value: int, max_value: int, 
                                        width: int = 40, percentage: bool = True,
                                        fill_char: str = '█', empty_char: str = '░') -> str:
        """
        Generate a horizontal bar chart with a label.
        
        Args:
            label: Label for the bar
            value: Current value
            max_value: Maximum value
            width: Width of the bar in characters
            percentage: Whether to show percentage
            fill_char: Character to use for filled portion
            empty_char: Character to use for empty portion
            
        Returns:
            ASCII bar chart string with label
        """
        bar = ASCIIBar.generate_horizontal_bar(value, max_value, width, fill_char, empty_char)
        
        if percentage and max_value > 0:
            percent = (value / max_value) * 100
            return f"{label:15} [{bar}] {value:4d} ({percent:5.1f}%)"
        else:
            return f"{label:15} [{bar}] {value:4d}"
    
    @staticmethod
    def generate_stacked_bar(values: Dict[str, int], max_value: int, width: int = 60,
                           fills: Optional[Dict[str, str]] = None) -> str:
        """
        Generate a stacked horizontal bar chart.
        
        Args:
            values: Dictionary of category to value
            max_value: Maximum total value
            width: Width of the bar in characters
            fills: Dictionary of category to fill character
            
        Returns:
            ASCII stacked bar chart string
        """
        if max_value <= 0:
            return ' ' * width
        
        # Default fill characters
        default_fills = {
            'critical': '█',
            'high': '▓',
            'medium': '▒',
            'low': '░',
            'info': ' '
        }
        
        fills = fills or default_fills
        total_width = 0
        bar = ''
        
        # Generate proportional segments
        for category, value in values.items():
            segment_width = min(width - total_width, int((value / max_value) * width))
            if segment_width <= 0:
                continue
                
            fill_char = fills.get(category, '█')
            bar += fill_char * segment_width
            total_width += segment_width
        
        # Fill remaining space
        if total_width < width:
            bar += ' ' * (width - total_width)
        
        return bar
    
    @staticmethod
    def generate_stacked_bar_with_legend(values: Dict[str, int], max_value: int, 
                                       width: int = 60, title: str = '',
                                       fills: Optional[Dict[str, str]] = None) -> str:
        """
        Generate a stacked horizontal bar chart with a legend.
        
        Args:
            values: Dictionary of category to value
            max_value: Maximum total value
            width: Width of the bar in characters
            title: Title for the bar chart
            fills: Dictionary of category to fill character
            
        Returns:
            ASCII stacked bar chart string with legend
        """
        bar = ASCIIBar.generate_stacked_bar(values, max_value, width, fills)
        
        # Create legend
        legend = '  '.join([f"{fills.get(cat, '█')} {cat}: {val}" for cat, val in values.items() if val > 0])
        
        if title:
            return f"{title}\n[{bar}]\n{legend}"
        else:
            return f"[{bar}]\n{legend}"


class RuleVisualizer:
    """
    Visualizes rule statistics and coverage.
    """
    
    def __init__(self, rules_dir: Path = None):
        """
        Initialize with rules directory.
        
        Args:
            rules_dir: Directory containing rule files
        """
        self.rules_dir = rules_dir or DEFAULT_RULES_DIR
        self.rule_sets = get_all_rule_sets()
        self.rules = []
        
        # Collect all rules
        for rule_set in self.rule_sets:
            self.rules.extend(rule_set.rules)
        
        # Collect statistics
        self.stats = self._collect_stats()
    
    def _collect_stats(self) -> Dict[str, Any]:
        """
        Collect statistics about rules.
        
        Returns:
            Dictionary of statistics
        """
        # Basic counts
        by_category = {}
        by_severity = {}
        by_confidence = {}
        by_criticality = {}
        by_complexity = {}
        by_reliability = {}
        by_source = {}
        by_rule_type = {}
        tags = {}
        
        for rule in self.rules:
            # Count by category
            category = rule.category.value
            by_category[category] = by_category.get(category, 0) + 1
            
            # Count by severity
            severity = rule.severity.value
            by_severity[severity] = by_severity.get(severity, 0) + 1
            
            # Count by confidence
            confidence = rule.confidence.value
            by_confidence[confidence] = by_confidence.get(confidence, 0) + 1
            
            # Count by tag
            for tag in rule.tags:
                tags[tag] = tags.get(tag, 0) + 1
            
            # Enhanced rule stats
            if hasattr(rule, 'criticality'):
                criticality = rule.criticality.value
                by_criticality[criticality] = by_criticality.get(criticality, 0) + 1
            
            if hasattr(rule, 'complexity'):
                complexity = rule.complexity.value
                by_complexity[complexity] = by_complexity.get(complexity, 0) + 1
            
            if hasattr(rule, 'reliability'):
                reliability = rule.reliability.value
                by_reliability[reliability] = by_reliability.get(reliability, 0) + 1
            
            if hasattr(rule, 'source'):
                source = rule.source.value
                by_source[source] = by_source.get(source, 0) + 1
            
            if hasattr(rule, 'rule_type'):
                rule_type = rule.rule_type.value
                by_rule_type[rule_type] = by_rule_type.get(rule_type, 0) + 1
        
        # Check for conflicts
        conflicts = detect_rule_conflicts(self.rules)
        
        # Check for circular dependencies
        dependency_checker = RuleDependency(self.rules)
        circular_deps = dependency_checker.detect_circular_dependencies()
        
        # Example coverage calculation
        total_examples = 0
        rules_with_examples = 0
        
        for rule in self.rules:
            if hasattr(rule, 'examples') and rule.examples:
                rules_with_examples += 1
                total_examples += len(rule.examples)
        
        return {
            "total_rules": len(self.rules),
            "by_category": by_category,
            "by_severity": by_severity,
            "by_confidence": by_confidence,
            "by_criticality": by_criticality,
            "by_complexity": by_complexity,
            "by_reliability": by_reliability,
            "by_source": by_source,
            "by_rule_type": by_rule_type,
            "tags": tags,
            "conflicts": conflicts,
            "circular_dependencies": circular_deps,
            "example_coverage": {
                "total_examples": total_examples,
                "rules_with_examples": rules_with_examples,
                "coverage_percentage": (rules_with_examples / len(self.rules) * 100) if self.rules else 0
            }
        }
    
    def visualize_category_distribution(self, width: int = 60) -> str:
        """
        Visualize distribution of rules by category.
        
        Args:
            width: Width of the bar chart
            
        Returns:
            ASCII visualization
        """
        by_category = self.stats["by_category"]
        max_value = max(by_category.values()) if by_category else 0
        
        result = "Rule Distribution by Category:\n\n"
        
        for category, count in sorted(by_category.items(), key=lambda x: x[1], reverse=True):
            result += ASCIIBar.generate_horizontal_bar_with_label(
                category, count, max_value, width) + "\n"
        
        return result
    
    def visualize_severity_distribution(self, width: int = 60) -> str:
        """
        Visualize distribution of rules by severity.
        
        Args:
            width: Width of the bar chart
            
        Returns:
            ASCII visualization
        """
        by_severity = self.stats["by_severity"]
        max_value = max(by_severity.values()) if by_severity else 0
        
        # Order by severity
        ordered_severities = ["critical", "high", "medium", "low", "info"]
        
        result = "Rule Distribution by Severity:\n\n"
        
        for severity in ordered_severities:
            if severity in by_severity:
                result += ASCIIBar.generate_horizontal_bar_with_label(
                    severity, by_severity[severity], max_value, width) + "\n"
        
        # Stacked bar representation
        result += "\nSeverity Distribution (Stacked):\n"
        
        fills = {
            'critical': '█',
            'high': '▓',
            'medium': '▒',
            'low': '░',
            'info': ' '
        }
        
        filtered_severities = {k: v for k, v in by_severity.items() if k in ordered_severities}
        
        result += ASCIIBar.generate_stacked_bar_with_legend(
            filtered_severities, sum(filtered_severities.values()), width, fills=fills)
        
        return result
    
    def visualize_tag_distribution(self, top_n: int = 10, width: int = 60) -> str:
        """
        Visualize distribution of rules by tag.
        
        Args:
            top_n: Number of top tags to show
            width: Width of the bar chart
            
        Returns:
            ASCII visualization
        """
        tags = self.stats["tags"]
        
        if not tags:
            return "No tags found."
        
        # Get top N tags
        top_tags = sorted(tags.items(), key=lambda x: x[1], reverse=True)[:top_n]
        max_value = top_tags[0][1] if top_tags else 0
        
        result = f"Top {top_n} Tags:\n\n"
        
        for tag, count in top_tags:
            result += ASCIIBar.generate_horizontal_bar_with_label(
                tag, count, max_value, width) + "\n"
        
        return result
    
    def visualize_example_coverage(self, width: int = 60) -> str:
        """
        Visualize example coverage in rules.
        
        Args:
            width: Width of the bar chart
            
        Returns:
            ASCII visualization
        """
        example_coverage = self.stats["example_coverage"]
        total_rules = self.stats["total_rules"]
        rules_with_examples = example_coverage["rules_with_examples"]
        coverage_percentage = example_coverage["coverage_percentage"]
        
        result = "Example Coverage:\n\n"
        
        # Basic coverage bar
        result += ASCIIBar.generate_horizontal_bar_with_label(
            "Coverage", rules_with_examples, total_rules, width) + "\n\n"
        
        # Add summary
        result += f"Rules with examples: {rules_with_examples}/{total_rules} ({coverage_percentage:.1f}%)\n"
        result += f"Total examples: {example_coverage['total_examples']}\n"
        result += f"Examples per rule: {example_coverage['total_examples']/total_rules:.2f} (average)\n"
        
        return result
    
    def visualize_dependency_graph(self) -> str:
        """
        Visualize rule dependencies.
        
        Returns:
            ASCII visualization of dependencies
        """
        # Find rules with dependencies
        rules_with_deps = [r for r in self.rules if hasattr(r, 'dependencies') and r.dependencies]
        
        if not rules_with_deps:
            return "No rule dependencies found."
        
        result = "Rule Dependency Graph:\n\n"
        
        # Create dependency checker
        dependency_checker = RuleDependency(self.rules)
        
        # Generate ASCII graph
        visited = set()
        
        def generate_tree(rule_id, depth=0):
            if rule_id in visited:
                return "  " * depth + f"└── {rule_id} (circular reference)\n"
            
            visited.add(rule_id)
            output = "  " * depth + f"└── {rule_id}\n"
            
            # Get dependent rules
            dependent_rules = dependency_checker.get_dependent_rules(rule_id)
            
            for i, rule in enumerate(dependent_rules):
                output += generate_tree(rule.id, depth + 1)
            
            visited.remove(rule_id)
            return output
        
        # Find root nodes (rules with no prerequisites)
        root_rules = []
        for rule in rules_with_deps:
            prereqs = dependency_checker.get_prerequisites(rule.id)
            if not prereqs:
                root_rules.append(rule)
        
        # Generate graph for each root
        for rule in root_rules:
            result += f"{rule.id}\n"
            prereqs = dependency_checker.get_dependent_rules(rule.id)
            for dep_rule in prereqs:
                result += generate_tree(dep_rule.id, 1)
            result += "\n"
        
        # Show circular dependencies
        circular_deps = dependency_checker.detect_circular_dependencies()
        if circular_deps:
            result += "\nCircular Dependencies:\n\n"
            for i, chain in enumerate(circular_deps, 1):
                result += f"{i}. {' -> '.join(chain)}\n"
        
        return result
    
    def visualize_all(self, width: int = 60) -> str:
        """
        Generate all visualizations.
        
        Args:
            width: Width of the bar charts
            
        Returns:
            ASCII visualizations
        """
        result = "=== RULE SYSTEM VISUALIZATION ===\n\n"
        result += f"Total Rules: {self.stats['total_rules']}\n\n"
        
        result += self.visualize_category_distribution(width) + "\n\n"
        result += self.visualize_severity_distribution(width) + "\n\n"
        result += self.visualize_example_coverage(width) + "\n\n"
        result += self.visualize_tag_distribution(10, width) + "\n\n"
        
        # Add enhanced visualizations if available
        if self.stats["by_criticality"]:
            result += "Rule Distribution by Criticality:\n\n"
            by_criticality = self.stats["by_criticality"]
            max_value = max(by_criticality.values()) if by_criticality else 0
            
            ordered_criticalities = ["critical", "high", "medium", "low"]
            
            for criticality in ordered_criticalities:
                if criticality in by_criticality:
                    result += ASCIIBar.generate_horizontal_bar_with_label(
                        criticality, by_criticality[criticality], max_value, width) + "\n"
            
            result += "\n\n"
        
        if self.stats["by_complexity"]:
            result += "Rule Distribution by Complexity:\n\n"
            by_complexity = self.stats["by_complexity"]
            max_value = max(by_complexity.values()) if by_complexity else 0
            
            ordered_complexities = ["complex", "moderate", "simple", "unknown"]
            
            for complexity in ordered_complexities:
                if complexity in by_complexity:
                    result += ASCIIBar.generate_horizontal_bar_with_label(
                        complexity, by_complexity[complexity], max_value, width) + "\n"
            
            result += "\n\n"
        
        if self.stats["by_reliability"]:
            result += "Rule Distribution by Reliability:\n\n"
            by_reliability = self.stats["by_reliability"]
            max_value = max(by_reliability.values()) if by_reliability else 0
            
            ordered_reliabilities = ["high", "medium", "low"]
            
            for reliability in ordered_reliabilities:
                if reliability in by_reliability:
                    result += ASCIIBar.generate_horizontal_bar_with_label(
                        reliability, by_reliability[reliability], max_value, width) + "\n"
            
            result += "\n\n"
        
        # Add dependency graph
        result += self.visualize_dependency_graph() + "\n"
        
        return result
    
    def export_html(self, output_path: Path) -> None:
        """
        Export visualizations as HTML.
        
        Args:
            output_path: Path to write the HTML file
        """
        # Basic HTML template with embedded CSS
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Homeostasis Rule System Analysis</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 20px;
                    max-width: 1200px;
                    margin: 0 auto;
                    color: #333;
                }
                h1, h2, h3 {
                    color: #2c3e50;
                }
                .container {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 20px;
                    margin: 20px 0;
                }
                .card {
                    background-color: #f9f9f9;
                    border-radius: 5px;
                    padding: 15px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    flex: 1 1 calc(50% - 20px);
                    min-width: 300px;
                }
                .bar-container {
                    margin: 10px 0;
                    width: 100%;
                }
                .bar-label {
                    display: inline-block;
                    width: 120px;
                    margin-right: 10px;
                    text-align: right;
                }
                .bar-outer {
                    display: inline-block;
                    width: calc(100% - 220px);
                    background-color: #eee;
                    height: 20px;
                    border-radius: 2px;
                    vertical-align: middle;
                }
                .bar-inner {
                    height: 100%;
                    border-radius: 2px;
                }
                .bar-value {
                    display: inline-block;
                    width: 80px;
                    margin-left: 10px;
                }
                .severity-critical { background-color: #e74c3c; }
                .severity-high { background-color: #e67e22; }
                .severity-medium { background-color: #f1c40f; }
                .severity-low { background-color: #3498db; }
                .severity-info { background-color: #2ecc71; }
                
                .criticality-critical { background-color: #e74c3c; }
                .criticality-high { background-color: #e67e22; }
                .criticality-medium { background-color: #f1c40f; }
                .criticality-low { background-color: #3498db; }
                
                .complexity-complex { background-color: #e74c3c; }
                .complexity-moderate { background-color: #f1c40f; }
                .complexity-simple { background-color: #2ecc71; }
                .complexity-unknown { background-color: #95a5a6; }
                
                .reliability-high { background-color: #2ecc71; }
                .reliability-medium { background-color: #f1c40f; }
                .reliability-low { background-color: #e74c3c; }
                
                .tag-chart .bar-inner { background-color: #3498db; }
                
                .stacked-bar {
                    height: 30px;
                    width: 100%;
                    margin: 10px 0;
                    display: flex;
                    border-radius: 3px;
                    overflow: hidden;
                }
                .stacked-segment {
                    height: 100%;
                }
                .stacked-legend {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 10px;
                    margin: 10px 0;
                }
                .legend-item {
                    display: flex;
                    align-items: center;
                }
                .legend-color {
                    width: 15px;
                    height: 15px;
                    margin-right: 5px;
                    border-radius: 2px;
                }
                .coverage-summary {
                    margin-top: 20px;
                    padding: 10px;
                    background-color: #f8f9fa;
                    border-left: 4px solid #6c757d;
                }
                table {
                    width: 100%;
                    border-collapse: collapse;
                    margin: 15px 0;
                }
                th, td {
                    padding: 8px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }
                th {
                    background-color: #f2f2f2;
                }
                tr:hover {
                    background-color: #f5f5f5;
                }
                .dependency-graph {
                    font-family: monospace;
                    white-space: pre;
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-radius: 5px;
                    overflow-x: auto;
                }
                .circular-dependencies {
                    color: #e74c3c;
                }
            </style>
        </head>
        <body>
            <h1>Homeostasis Rule System Analysis</h1>
            
            <div class="container">
                <div class="card">
                    <h2>Overview</h2>
                    <p><strong>Total Rules:</strong> {{total_rules}}</p>
                    <p><strong>Rules with Examples:</strong> {{rules_with_examples}} ({{example_coverage_percent}}%)</p>
                    <p><strong>Total Examples:</strong> {{total_examples}}</p>
                    <p><strong>Examples per Rule:</strong> {{examples_per_rule}}</p>
                </div>
                
                <div class="card">
                    <h2>Rule Type Distribution</h2>
                    <div class="stacked-bar">
                        {{rule_type_segments}}
                    </div>
                    <div class="stacked-legend">
                        {{rule_type_legend}}
                    </div>
                </div>
            </div>
            
            <div class="container">
                <div class="card">
                    <h2>Category Distribution</h2>
                    {{category_bars}}
                </div>
                
                <div class="card">
                    <h2>Severity Distribution</h2>
                    {{severity_bars}}
                    
                    <h3>Stacked View</h3>
                    <div class="stacked-bar">
                        {{severity_segments}}
                    </div>
                    <div class="stacked-legend">
                        {{severity_legend}}
                    </div>
                </div>
            </div>
            
            <div class="container">
                <div class="card">
                    <h2>Criticality Distribution</h2>
                    {{criticality_bars}}
                </div>
                
                <div class="card">
                    <h2>Complexity Distribution</h2>
                    {{complexity_bars}}
                </div>
            </div>
            
            <div class="container">
                <div class="card">
                    <h2>Reliability Distribution</h2>
                    {{reliability_bars}}
                </div>
                
                <div class="card">
                    <h2>Top Tags</h2>
                    <div class="tag-chart">
                        {{tag_bars}}
                    </div>
                </div>
            </div>
            
            <div class="container">
                <div class="card">
                    <h2>Example Coverage</h2>
                    <div class="bar-container">
                        <span class="bar-label">Coverage</span>
                        <span class="bar-outer">
                            <div class="bar-inner severity-medium" style="width: {{example_coverage_percent}}%;"></div>
                        </span>
                        <span class="bar-value">{{rules_with_examples}}/{{total_rules}}</span>
                    </div>
                    
                    <div class="coverage-summary">
                        <p><strong>Rules with examples:</strong> {{rules_with_examples}} out of {{total_rules}} ({{example_coverage_percent}}%)</p>
                        <p><strong>Total examples:</strong> {{total_examples}}</p>
                        <p><strong>Average examples per rule:</strong> {{examples_per_rule}}</p>
                    </div>
                </div>
                
                <div class="card">
                    <h2>Potential Issues</h2>
                    <h3>Rule Conflicts</h3>
                    {{conflicts_table}}
                    
                    <h3>Circular Dependencies</h3>
                    {{circular_dependencies}}
                </div>
            </div>
            
            <div class="container">
                <div class="card" style="flex: 1 1 100%;">
                    <h2>Dependency Graph</h2>
                    <div class="dependency-graph">
                        {{dependency_graph}}
                    </div>
                </div>
            </div>
            
            <footer>
                <p>Generated by Homeostasis Rule Visualization Tool</p>
            </footer>
        </body>
        </html>
        """
        
        # Generate category bars
        category_bars = ""
        by_category = self.stats["by_category"]
        max_category = max(by_category.values()) if by_category else 0
        
        for category, count in sorted(by_category.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / max_category) * 100 if max_category else 0
            category_bars += f"""
            <div class="bar-container">
                <span class="bar-label">{category}</span>
                <span class="bar-outer">
                    <div class="bar-inner severity-medium" style="width: {percentage}%;"></div>
                </span>
                <span class="bar-value">{count}</span>
            </div>
            """
        
        # Generate severity bars
        severity_bars = ""
        by_severity = self.stats["by_severity"]
        max_severity = max(by_severity.values()) if by_severity else 0
        
        ordered_severities = ["critical", "high", "medium", "low", "info"]
        
        for severity in ordered_severities:
            if severity in by_severity:
                count = by_severity[severity]
                percentage = (count / max_severity) * 100 if max_severity else 0
                severity_bars += f"""
                <div class="bar-container">
                    <span class="bar-label">{severity}</span>
                    <span class="bar-outer">
                        <div class="bar-inner severity-{severity}" style="width: {percentage}%;"></div>
                    </span>
                    <span class="bar-value">{count}</span>
                </div>
                """
        
        # Generate severity stacked bar
        severity_segments = ""
        severity_total = sum(by_severity.values())
        
        severity_legend = ""
        for severity in ordered_severities:
            if severity in by_severity:
                count = by_severity[severity]
                percentage = (count / severity_total) * 100 if severity_total else 0
                severity_segments += f'<div class="stacked-segment severity-{severity}" style="width: {percentage}%;"></div>'
                
                severity_legend += f"""
                <div class="legend-item">
                    <div class="legend-color severity-{severity}"></div>
                    <span>{severity}: {count}</span>
                </div>
                """
        
        # Generate criticality bars
        criticality_bars = ""
        by_criticality = self.stats["by_criticality"]
        max_criticality = max(by_criticality.values()) if by_criticality else 0
        
        ordered_criticalities = ["critical", "high", "medium", "low"]
        
        for criticality in ordered_criticalities:
            if criticality in by_criticality:
                count = by_criticality[criticality]
                percentage = (count / max_criticality) * 100 if max_criticality else 0
                criticality_bars += f"""
                <div class="bar-container">
                    <span class="bar-label">{criticality}</span>
                    <span class="bar-outer">
                        <div class="bar-inner criticality-{criticality}" style="width: {percentage}%;"></div>
                    </span>
                    <span class="bar-value">{count}</span>
                </div>
                """
        
        # Generate complexity bars
        complexity_bars = ""
        by_complexity = self.stats["by_complexity"]
        max_complexity = max(by_complexity.values()) if by_complexity else 0
        
        ordered_complexities = ["complex", "moderate", "simple", "unknown"]
        
        for complexity in ordered_complexities:
            if complexity in by_complexity:
                count = by_complexity[complexity]
                percentage = (count / max_complexity) * 100 if max_complexity else 0
                complexity_bars += f"""
                <div class="bar-container">
                    <span class="bar-label">{complexity}</span>
                    <span class="bar-outer">
                        <div class="bar-inner complexity-{complexity}" style="width: {percentage}%;"></div>
                    </span>
                    <span class="bar-value">{count}</span>
                </div>
                """
        
        # Generate reliability bars
        reliability_bars = ""
        by_reliability = self.stats["by_reliability"]
        max_reliability = max(by_reliability.values()) if by_reliability else 0
        
        ordered_reliabilities = ["high", "medium", "low"]
        
        for reliability in ordered_reliabilities:
            if reliability in by_reliability:
                count = by_reliability[reliability]
                percentage = (count / max_reliability) * 100 if max_reliability else 0
                reliability_bars += f"""
                <div class="bar-container">
                    <span class="bar-label">{reliability}</span>
                    <span class="bar-outer">
                        <div class="bar-inner reliability-{reliability}" style="width: {percentage}%;"></div>
                    </span>
                    <span class="bar-value">{count}</span>
                </div>
                """
        
        # Generate rule type segments
        rule_type_segments = ""
        by_rule_type = self.stats["by_rule_type"]
        rule_type_total = sum(by_rule_type.values())
        
        rule_type_colors = {
            "error_detection": "#3498db",
            "performance_issue": "#e67e22",
            "security_vulnerability": "#e74c3c",
            "code_smell": "#f1c40f",
            "best_practice": "#2ecc71"
        }
        
        rule_type_legend = ""
        for rule_type, count in by_rule_type.items():
            percentage = (count / rule_type_total) * 100 if rule_type_total else 0
            color = rule_type_colors.get(rule_type, "#95a5a6")
            rule_type_segments += f'<div class="stacked-segment" style="width: {percentage}%; background-color: {color};"></div>'
            
            rule_type_legend += f"""
            <div class="legend-item">
                <div class="legend-color" style="background-color: {color};"></div>
                <span>{rule_type.replace('_', ' ')}: {count}</span>
            </div>
            """
        
        # Generate tag bars
        tags = self.stats["tags"]
        tag_bars = ""
        
        if tags:
            top_tags = sorted(tags.items(), key=lambda x: x[1], reverse=True)[:10]
            max_tag = top_tags[0][1] if top_tags else 0
            
            for tag, count in top_tags:
                percentage = (count / max_tag) * 100 if max_tag else 0
                tag_bars += f"""
                <div class="bar-container">
                    <span class="bar-label">{tag}</span>
                    <span class="bar-outer">
                        <div class="bar-inner" style="width: {percentage}%;"></div>
                    </span>
                    <span class="bar-value">{count}</span>
                </div>
                """
        
        # Generate conflicts table
        conflicts = self.stats["conflicts"]
        conflicts_table = "<p>No conflicts detected.</p>"
        
        if conflicts:
            conflicts_table = """
            <table>
                <tr>
                    <th>Rule 1</th>
                    <th>Rule 2</th>
                    <th>Type</th>
                    <th>Reason</th>
                </tr>
            """
            
            for conflict in conflicts:
                conflicts_table += f"""
                <tr>
                    <td>{conflict['rule1']}</td>
                    <td>{conflict['rule2']}</td>
                    <td>{conflict['type']}</td>
                    <td>{conflict['reason']}</td>
                </tr>
                """
            
            conflicts_table += "</table>"
        
        # Generate circular dependencies
        circular_deps = self.stats["circular_dependencies"]
        circular_dependencies = "<p>No circular dependencies detected.</p>"
        
        if circular_deps:
            circular_dependencies = "<div class='circular-dependencies'>"
            
            for i, chain in enumerate(circular_deps, 1):
                circular_dependencies += f"<p>{i}. {' → '.join(chain)}</p>"
            
            circular_dependencies += "</div>"
        
        # Generate dependency graph
        dependency_graph = self.visualize_dependency_graph()
        
        # Replace placeholders in template
        html_content = html_template
        html_content = html_content.replace("{{total_rules}}", str(self.stats["total_rules"]))
        
        example_coverage = self.stats["example_coverage"]
        html_content = html_content.replace("{{rules_with_examples}}", str(example_coverage["rules_with_examples"]))
        html_content = html_content.replace("{{example_coverage_percent}}", f"{example_coverage['coverage_percentage']:.1f}")
        html_content = html_content.replace("{{total_examples}}", str(example_coverage["total_examples"]))
        html_content = html_content.replace("{{examples_per_rule}}", f"{example_coverage['total_examples']/self.stats['total_rules']:.2f}" if self.stats['total_rules'] else "0")
        
        html_content = html_content.replace("{{category_bars}}", category_bars)
        html_content = html_content.replace("{{severity_bars}}", severity_bars)
        html_content = html_content.replace("{{severity_segments}}", severity_segments)
        html_content = html_content.replace("{{severity_legend}}", severity_legend)
        html_content = html_content.replace("{{criticality_bars}}", criticality_bars)
        html_content = html_content.replace("{{complexity_bars}}", complexity_bars)
        html_content = html_content.replace("{{reliability_bars}}", reliability_bars)
        html_content = html_content.replace("{{tag_bars}}", tag_bars)
        html_content = html_content.replace("{{rule_type_segments}}", rule_type_segments)
        html_content = html_content.replace("{{rule_type_legend}}", rule_type_legend)
        html_content = html_content.replace("{{conflicts_table}}", conflicts_table)
        html_content = html_content.replace("{{circular_dependencies}}", circular_dependencies)
        html_content = html_content.replace("{{dependency_graph}}", dependency_graph)
        
        # Write to file
        with open(output_path, "w") as f:
            f.write(html_content)
        
        print(f"HTML visualization exported to {output_path}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Visualize rule coverage and statistics"
    )
    parser.add_argument(
        "--text", "-t", action="store_true",
        help="Generate text visualization to stdout"
    )
    parser.add_argument(
        "--html", "-o", type=str,
        help="Generate HTML visualization to specified file"
    )
    parser.add_argument(
        "--width", "-w", type=int, default=60,
        help="Width of ASCII bars (text mode only)"
    )
    parser.add_argument(
        "--category", "-c", action="store_true",
        help="Show category distribution only"
    )
    parser.add_argument(
        "--severity", "-s", action="store_true",
        help="Show severity distribution only"
    )
    parser.add_argument(
        "--tags", "-g", action="store_true",
        help="Show tag distribution only"
    )
    parser.add_argument(
        "--examples", "-e", action="store_true",
        help="Show example coverage only"
    )
    parser.add_argument(
        "--dependencies", "-d", action="store_true",
        help="Show dependency graph only"
    )
    
    args = parser.parse_args()
    
    # If no arguments specified, show help
    if not any([args.text, args.html, args.category, args.severity, 
                args.tags, args.examples, args.dependencies]):
        parser.print_help()
        return
    
    # Create visualizer
    visualizer = RuleVisualizer()
    
    # Generate requested visualizations
    if args.text or args.category or args.severity or args.tags or args.examples or args.dependencies:
        # Show specific visualizations if requested
        if args.category:
            print(visualizer.visualize_category_distribution(args.width))
        
        if args.severity:
            print(visualizer.visualize_severity_distribution(args.width))
        
        if args.tags:
            print(visualizer.visualize_tag_distribution(10, args.width))
        
        if args.examples:
            print(visualizer.visualize_example_coverage(args.width))
        
        if args.dependencies:
            print(visualizer.visualize_dependency_graph())
        
        # Show all if text mode is requested and no specific visualization
        if args.text and not any([args.category, args.severity, args.tags, args.examples, args.dependencies]):
            print(visualizer.visualize_all(args.width))
    
    # Generate HTML visualization if requested
    if args.html:
        visualizer.export_html(Path(args.html))


if __name__ == "__main__":
    main()