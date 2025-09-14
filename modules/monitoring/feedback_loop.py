"""
Feedback loop for fix quality improvement.

This module provides utilities for:
1. Analyzing fix effectiveness data
2. Generating insights for future fixes
3. Improving templates and fix strategies based on historical data
"""

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional

from modules.monitoring.logger import MonitoringLogger
from modules.monitoring.metrics_collector import MetricsCollector

# Define project root
project_root = Path(__file__).parent.parent.parent


class FeedbackLoop:
    """
    Implements a feedback loop for improving fix quality.
    """

    def __init__(
        self,
        metrics_collector: Optional[MetricsCollector] = None,
        insights_file: Optional[Path] = None,
        log_level: str = "INFO",
    ):
        """
        Initialize the feedback loop.

        Args:
            metrics_collector: Optional metrics collector instance
            insights_file: File to store insights and recommendations
            log_level: Logging level
        """
        self.logger = MonitoringLogger("feedback_loop", log_level=log_level)

        # Set up metrics collector
        if metrics_collector:
            self.metrics_collector = metrics_collector
        else:
            self.metrics_collector = MetricsCollector(log_level=log_level)

        # Set up insights file
        self.insights_file = insights_file or (project_root / "logs" / "insights.json")

        # Load existing insights
        self.insights = self._load_insights()

        # Template effectiveness tracking
        self.template_stats: DefaultDict[str, Dict[str, int]] = defaultdict(
            lambda: {"uses": 0, "success": 0, "failure": 0}
        )

        # Bug type effectiveness tracking
        self.bug_type_stats: DefaultDict[str, Dict[str, int]] = defaultdict(
            lambda: {"fixes": 0, "success": 0, "failure": 0}
        )

        # Load template stats
        self._load_template_stats()

        self.logger.info("Initialized feedback loop")

    def _load_insights(self) -> Dict[str, Any]:
        """
        Load insights from file.

        Returns:
            Insights data
        """
        if not self.insights_file.exists():
            return {
                "templates": {},
                "bug_types": {},
                "recommendations": [],
                "updated_at": time.time(),
            }

        try:
            with open(self.insights_file, "r") as f:
                data: Dict[str, Any] = json.load(f)
                return data
        except Exception as e:
            self.logger.exception(
                e, message=f"Failed to load insights from {self.insights_file}"
            )
            return {
                "templates": {},
                "bug_types": {},
                "recommendations": [],
                "updated_at": time.time(),
            }

    def _save_insights(self) -> None:
        """Save insights to file."""
        try:
            # Update timestamp
            self.insights["updated_at"] = time.time()

            with open(self.insights_file, "w") as f:
                json.dump(self.insights, f, indent=2)

            self.logger.info(f"Saved insights to {self.insights_file}")

        except Exception as e:
            self.logger.exception(
                e, message=f"Failed to save insights to {self.insights_file}"
            )

    def _load_template_stats(self) -> None:
        """Load template and bug type statistics from insights."""
        # Load template stats
        for template_name, stats in self.insights.get("templates", {}).items():
            self.template_stats[template_name] = stats

        # Load bug type stats
        for bug_type, stats in self.insights.get("bug_types", {}).items():
            self.bug_type_stats[bug_type] = stats

    def record_fix_result(
        self,
        patch_id: str,
        template_name: str,
        bug_type: str,
        success: bool,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record a fix result for learning.

        Args:
            patch_id: ID of the patch
            template_name: Name of the template used
            bug_type: Type of bug fixed
            success: Whether the fix was successful
            metrics: Optional metrics data
        """
        # Update template stats
        self.template_stats[template_name]["uses"] += 1
        if success:
            self.template_stats[template_name]["success"] += 1
        else:
            self.template_stats[template_name]["failure"] += 1

        # Update bug type stats
        self.bug_type_stats[bug_type]["fixes"] += 1
        if success:
            self.bug_type_stats[bug_type]["success"] += 1
        else:
            self.bug_type_stats[bug_type]["failure"] += 1

        # Update insights
        self.insights["templates"] = dict(self.template_stats)
        self.insights["bug_types"] = dict(self.bug_type_stats)

        # Save insights
        self._save_insights()

        self.logger.info(
            f"Recorded fix result for template {template_name}, bug type {bug_type}: {'success' if success else 'failure'}",
            patch_id=patch_id,
        )

    def get_template_effectiveness(
        self, template_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get effectiveness statistics for templates.

        Args:
            template_name: Optional template name to get specific stats

        Returns:
            Template effectiveness statistics
        """
        if template_name:
            if template_name in self.template_stats:
                stats = self.template_stats[template_name]
                success_rate = (
                    stats["success"] / stats["uses"] if stats["uses"] > 0 else 0
                )
                return {
                    "template": template_name,
                    "uses": stats["uses"],
                    "success": stats["success"],
                    "failure": stats["failure"],
                    "success_rate": success_rate,
                }
            return {
                "template": template_name,
                "uses": 0,
                "success": 0,
                "failure": 0,
                "success_rate": 0,
            }

        # Get stats for all templates
        results = {}
        for template_name, stats in self.template_stats.items():
            success_rate = stats["success"] / stats["uses"] if stats["uses"] > 0 else 0
            results[template_name] = {
                "uses": stats["uses"],
                "success": stats["success"],
                "failure": stats["failure"],
                "success_rate": success_rate,
            }

        return results

    def get_bug_type_effectiveness(
        self, bug_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get effectiveness statistics for bug types.

        Args:
            bug_type: Optional bug type to get specific stats

        Returns:
            Bug type effectiveness statistics
        """
        if bug_type:
            if bug_type in self.bug_type_stats:
                stats = self.bug_type_stats[bug_type]
                success_rate = (
                    stats["success"] / stats["fixes"] if stats["fixes"] > 0 else 0
                )
                return {
                    "bug_type": bug_type,
                    "fixes": stats["fixes"],
                    "success": stats["success"],
                    "failure": stats["failure"],
                    "success_rate": success_rate,
                }
            return {
                "bug_type": bug_type,
                "fixes": 0,
                "success": 0,
                "failure": 0,
                "success_rate": 0,
            }

        # Get stats for all bug types
        results = {}
        for bug_type, stats in self.bug_type_stats.items():
            success_rate = (
                stats["success"] / stats["fixes"] if stats["fixes"] > 0 else 0
            )
            results[bug_type] = {
                "fixes": stats["fixes"],
                "success": stats["success"],
                "failure": stats["failure"],
                "success_rate": success_rate,
            }

        return results

    def analyze_fixes(self) -> Dict[str, Any]:
        """
        Analyze fix effectiveness and generate insights.

        Returns:
            Analysis results
        """
        # Get fix metrics
        fix_metrics = self.metrics_collector.get_metrics("fix")

        if not fix_metrics:
            return {"count": 0, "message": "No fix metrics available"}

        # Group by template and bug type
        template_data = defaultdict(list)
        bug_type_data = defaultdict(list)

        for metric in fix_metrics:
            template = metric.get("template")
            bug_type = metric.get("bug_type")

            if template:
                template_data[template].append(metric)

            if bug_type:
                bug_type_data[bug_type].append(metric)

        # Analyze templates
        template_analysis = {}
        for template, metrics in template_data.items():
            success_count = sum(1 for m in metrics if m.get("success", False))
            success_rate = success_count / len(metrics)

            template_analysis[template] = {
                "count": len(metrics),
                "success_count": success_count,
                "failure_count": len(metrics) - success_count,
                "success_rate": success_rate,
            }

        # Analyze bug types
        bug_type_analysis = {}
        for bug_type, metrics in bug_type_data.items():
            success_count = sum(1 for m in metrics if m.get("success", False))
            success_rate = success_count / len(metrics)

            bug_type_analysis[bug_type] = {
                "count": len(metrics),
                "success_count": success_count,
                "failure_count": len(metrics) - success_count,
                "success_rate": success_rate,
            }

        # Overall analysis
        success_count = sum(1 for m in fix_metrics if m.get("success", False))
        success_rate = success_count / len(fix_metrics)

        result = {
            "count": len(fix_metrics),
            "success_count": success_count,
            "failure_count": len(fix_metrics) - success_count,
            "success_rate": success_rate,
            "templates": template_analysis,
            "bug_types": bug_type_analysis,
        }

        return result

    def generate_recommendations(self) -> List[Dict[str, Any]]:
        """
        Generate recommendations for improving fix quality.

        Returns:
            List of recommendations
        """
        recommendations = []

        # Analyze template effectiveness
        template_stats = self.get_template_effectiveness()

        for template_name, stats in template_stats.items():
            # Only consider templates with enough data
            if stats["uses"] < 5:
                continue

            # Look for underperforming templates
            if stats["success_rate"] < 0.7:  # Less than 70% success rate
                recommendations.append(
                    {
                        "type": "template_improvement",
                        "template": template_name,
                        "success_rate": stats["success_rate"],
                        "message": f"Template {template_name} has a low success rate of {stats['success_rate']:.1%}",
                        "suggestion": "Consider reviewing and improving this template",
                    }
                )

        # Analyze bug type effectiveness
        bug_type_stats = self.get_bug_type_effectiveness()

        for bug_type, stats in bug_type_stats.items():
            # Only consider bug types with enough data
            if stats["fixes"] < 5:
                continue

            # Look for challenging bug types
            if stats["success_rate"] < 0.6:  # Less than 60% success rate
                recommendations.append(
                    {
                        "type": "bug_type_challenge",
                        "bug_type": bug_type,
                        "success_rate": stats["success_rate"],
                        "message": f"Bug type {bug_type} has a low fix success rate of {stats['success_rate']:.1%}",
                        "suggestion": "Consider developing specialized templates or strategies for this bug type",
                    }
                )

        # Analyze metrics for successful and failed fixes
        fix_metrics = self.metrics_collector.get_metrics("fix")

        if fix_metrics:
            # Split by success/failure
            success_metrics = [m for m in fix_metrics if m.get("success", False)]
            failure_metrics = [m for m in fix_metrics if not m.get("success", False)]

            # Look for patterns in test metrics
            test_metrics = self.metrics_collector.get_metrics("test")
            if test_metrics:
                # Check if more thorough testing correlates with success
                success_patch_ids = {
                    m.get("patch_id") for m in success_metrics if "patch_id" in m
                }
                failure_patch_ids = {
                    m.get("patch_id") for m in failure_metrics if "patch_id" in m
                }

                success_test_counts = [
                    sum(1 for t in test_metrics if t.get("patch_id") == pid)
                    for pid in success_patch_ids
                ]

                failure_test_counts = [
                    sum(1 for t in test_metrics if t.get("patch_id") == pid)
                    for pid in failure_patch_ids
                ]

                if success_test_counts and failure_test_counts:
                    avg_success_tests = sum(success_test_counts) / len(
                        success_test_counts
                    )
                    avg_failure_tests = sum(failure_test_counts) / len(
                        failure_test_counts
                    )

                    if (
                        avg_success_tests > avg_failure_tests * 1.5
                    ):  # 50% more tests on average
                        recommendations.append(
                            {
                                "type": "testing_insight",
                                "message": f"Successful fixes have {avg_success_tests:.1f} tests on average, compared to {avg_failure_tests:.1f} for failed fixes",
                                "suggestion": "Consider more thorough testing to improve fix success rate",
                            }
                        )

        # Update insights with recommendations
        self.insights["recommendations"] = recommendations
        self._save_insights()

        return recommendations

    def apply_feedback_to_template(
        self, template_name: str, template_path: Path, improvements: Dict[str, Any]
    ) -> Optional[Path]:
        """
        Apply feedback to improve a template.

        Args:
            template_name: Name of the template to improve
            template_path: Path to the template file
            improvements: Dictionary of improvements to apply

        Returns:
            Path to the improved template or None if failed
        """
        if not template_path.exists():
            self.logger.error(f"Template file not found: {template_path}")
            return None

        try:
            # Read the template
            with open(template_path, "r") as f:
                template_content = f.read()

            # Apply improvements
            new_content = template_content

            # Add comments
            if "comments" in improvements:
                for comment in improvements["comments"]:
                    new_content = f"# {comment}\n{new_content}"

            # Add additional checks
            if "additional_checks" in improvements:
                for check in improvements["additional_checks"]:
                    # Identify placeholder for checks
                    check_placeholder = "# Additional checks can be added here"
                    if check_placeholder in new_content:
                        new_content = new_content.replace(
                            check_placeholder, f"{check_placeholder}\n{check}"
                        )

            # Modify patterns
            if "pattern_replacements" in improvements:
                for old_pattern, new_pattern in improvements[
                    "pattern_replacements"
                ].items():
                    new_content = new_content.replace(old_pattern, new_pattern)

            # Create backup of original template
            backup_path = template_path.with_suffix(f".bak_{int(time.time())}")
            import shutil

            shutil.copy2(template_path, backup_path)

            # Write improved template
            with open(template_path, "w") as f:
                f.write(new_content)

            self.logger.info(
                f"Applied feedback to template {template_name}, backup saved to {backup_path}"
            )

            return template_path

        except Exception as e:
            self.logger.exception(
                e, message=f"Failed to apply feedback to template {template_name}"
            )
            return None

    def update_template_priorities(self) -> Dict[str, float]:
        """
        Update template priorities based on effectiveness.

        Returns:
            Dictionary of template names to priority scores
        """
        template_stats = self.get_template_effectiveness()

        # Calculate priorities based on success rate
        priorities = {}
        for template_name, stats in template_stats.items():
            # Only consider templates with enough data
            if stats["uses"] < 3:
                priorities[template_name] = 0.5  # Default priority
                continue

            # Use success rate as priority
            priorities[template_name] = stats["success_rate"]

        return priorities

    def get_best_template_for_bug_type(self, bug_type: str) -> Optional[str]:
        """
        Get the most effective template for a specific bug type.

        Args:
            bug_type: Type of bug

        Returns:
            Name of the most effective template or None if no data
        """
        # Get fix metrics
        fix_metrics = self.metrics_collector.get_metrics("fix")

        if not fix_metrics:
            return None

        # Filter by bug type
        bug_fixes = [m for m in fix_metrics if m.get("bug_type") == bug_type]

        if not bug_fixes:
            return None

        # Group by template
        template_success: DefaultDict[str, Dict[str, int]] = defaultdict(
            lambda: {"count": 0, "success": 0}
        )

        for fix in bug_fixes:
            template = fix.get("template")
            if not template:
                continue

            template_success[template]["count"] += 1
            if fix.get("success", False):
                template_success[template]["success"] += 1

        # Find the template with highest success rate (with at least 3 uses)
        best_template = None
        best_rate = 0.0

        for template, stats in template_success.items():
            if stats["count"] < 3:
                continue

            success_rate = stats["success"] / stats["count"]
            if success_rate > best_rate:
                best_rate = success_rate
                best_template = template

        return best_template

    def log_learning_event(
        self, event_type: str, description: str, data: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log a learning event for the feedback loop.

        Args:
            event_type: Type of learning event
            description: Description of the event
            data: Optional additional data
        """
        event = {
            "event_type": event_type,
            "description": description,
            "timestamp": time.time(),
            "data": data or {},
        }

        # Add to insights
        if "learning_events" not in self.insights:
            self.insights["learning_events"] = []

        self.insights["learning_events"].append(event)
        self._save_insights()

        self.logger.info(f"Logged learning event: {description}", event_type=event_type)


class FixImprovement:
    """
    Identifies patterns for improving fix quality.
    """

    def __init__(self, feedback_loop: FeedbackLoop, log_level: str = "INFO"):
        """
        Initialize the fix improvement engine.

        Args:
            feedback_loop: Feedback loop for data and insights
            log_level: Logging level
        """
        self.logger = MonitoringLogger("fix_improvement", log_level=log_level)
        self.feedback_loop = feedback_loop

        self.logger.info("Initialized fix improvement engine")

    def identify_patterns(self) -> Dict[str, Any]:
        """
        Identify patterns for improving fix quality.

        Returns:
            Identified patterns
        """
        patterns: Dict[str, Any] = {"templates": {}, "bug_types": {}, "general": []}

        # Analyze template effectiveness
        template_stats = self.feedback_loop.get_template_effectiveness()

        # Find most and least effective templates
        template_effectiveness = [
            (name, stats["success_rate"])
            for name, stats in template_stats.items()
            if stats["uses"] >= 5
        ]

        if template_effectiveness:
            # Sort by success rate (descending)
            template_effectiveness.sort(key=lambda x: x[1], reverse=True)

            # Most effective templates
            for name, rate in template_effectiveness[:3]:
                patterns["templates"][name] = {
                    "success_rate": rate,
                    "effectiveness": "high",
                    "suggestion": "Use as model for other templates",
                }

            # Least effective templates
            for name, rate in template_effectiveness[-3:]:
                patterns["templates"][name] = {
                    "success_rate": rate,
                    "effectiveness": "low",
                    "suggestion": "Consider revision or replacement",
                }

        # Analyze bug type effectiveness
        bug_type_stats = self.feedback_loop.get_bug_type_effectiveness()

        # Find challenging bug types
        bug_type_effectiveness = [
            (name, stats["success_rate"])
            for name, stats in bug_type_stats.items()
            if stats["fixes"] >= 5
        ]

        if bug_type_effectiveness:
            # Sort by success rate (ascending)
            bug_type_effectiveness.sort(key=lambda x: x[1])

            # Most challenging bug types
            for name, rate in bug_type_effectiveness[:3]:
                patterns["bug_types"][name] = {
                    "success_rate": rate,
                    "challenge": "high",
                    "suggestion": "Develop specialized approach",
                }

        # Generate general insights
        fix_metrics = self.feedback_loop.metrics_collector.get_metrics("fix")

        if fix_metrics:
            success_count = sum(1 for m in fix_metrics if m.get("success", False))
            success_rate = success_count / len(fix_metrics)

            patterns["general"].append(
                {
                    "metric": "overall_success_rate",
                    "value": success_rate,
                    "insight": f"Overall fix success rate is {success_rate:.1%}",
                }
            )

            # Check if fix success rate is improving over time
            if len(fix_metrics) >= 10:
                # Sort by timestamp
                sorted_metrics = sorted(
                    fix_metrics, key=lambda x: x.get("timestamp", 0)
                )

                # Calculate success rate for first and second half
                midpoint = len(sorted_metrics) // 2
                first_half = sorted_metrics[:midpoint]
                second_half = sorted_metrics[midpoint:]

                first_success_rate = sum(
                    1 for m in first_half if m.get("success", False)
                ) / len(first_half)
                second_success_rate = sum(
                    1 for m in second_half if m.get("success", False)
                ) / len(second_half)

                improvement = second_success_rate - first_success_rate

                patterns["general"].append(
                    {
                        "metric": "success_rate_trend",
                        "value": improvement,
                        "insight": f"Fix success rate has {'improved' if improvement > 0 else 'declined'} by {abs(improvement):.1%}",
                    }
                )

        return patterns

    def generate_template_improvements(self, template_name: str) -> Dict[str, Any]:
        """
        Generate improvements for a specific template.

        Args:
            template_name: Name of the template to improve

        Returns:
            Suggested improvements
        """
        # Get template effectiveness
        template_stats = self.feedback_loop.get_template_effectiveness(template_name)

        if template_stats["uses"] < 5:
            return {
                "template": template_name,
                "message": "Not enough data to generate improvements",
            }

        # Get fix metrics for this template
        fix_metrics = self.feedback_loop.metrics_collector.get_metrics("fix")
        template_metrics = [
            m for m in fix_metrics if m.get("template") == template_name
        ]

        if not template_metrics:
            return {
                "template": template_name,
                "message": "No fix metrics found for this template",
            }

        # Split by success/failure
        failure_metrics = [m for m in template_metrics if not m.get("success", False)]

        # Identify common patterns in failures
        failure_patterns = set()

        for metric in failure_metrics:
            # Extract specific error messages
            error_message = metric.get("error_message", "")
            if error_message:
                failure_patterns.add(error_message)

        # Generate improvements
        improvements = {
            "template": template_name,
            "success_rate": template_stats["success_rate"],
            "uses": template_stats["uses"],
            "comments": [
                f"Template effectiveness: {template_stats['success_rate']:.1%} success rate over {template_stats['uses']} uses",
                f"Last updated: {time.strftime('%Y-%m-%d')}",
            ],
            "additional_checks": [],
            "pattern_replacements": {},
        }

        # Add checks based on failure patterns
        for pattern in failure_patterns:
            if "null" in pattern.lower():
                improvements["additional_checks"].append(
                    "# Check for null/None values\nif value is None:"
                )
            elif "undefined" in pattern.lower():
                improvements["additional_checks"].append(
                    "# Check for undefined values\nif 'undefined' in str(value):"
                )
            elif "index" in pattern.lower():
                improvements["additional_checks"].append(
                    "# Check for index errors\nif index >= len(collection):"
                )
            elif "key" in pattern.lower():
                improvements["additional_checks"].append(
                    "# Check for missing keys\nif key not in dictionary:"
                )

        # Add pattern replacements
        # These would normally be derived from analysis of specific template content
        # but we'll include some generic examples
        if template_stats["success_rate"] < 0.5:
            improvements["pattern_replacements"] = {
                "{{ value }}": "{{ value|default('') }}",
                "items[key]": "items.get(key)",
                "list[index]": "list[index] if index < len(list) else None",
            }

        return improvements

    def apply_improvements(self, templates_dir: Path) -> List[Dict[str, Any]]:
        """
        Apply improvements to templates.

        Args:
            templates_dir: Directory containing templates

        Returns:
            List of improvement results
        """
        results = []

        # Get recommendations
        recommendations = self.feedback_loop.generate_recommendations()

        # Filter for template improvement recommendations
        template_recommendations = [
            r for r in recommendations if r["type"] == "template_improvement"
        ]

        for recommendation in template_recommendations:
            template_name = recommendation["template"]
            template_path = templates_dir / f"{template_name}"

            if not template_path.exists():
                # Try with extension
                template_path = templates_dir / f"{template_name}.template"

                if not template_path.exists():
                    results.append(
                        {
                            "template": template_name,
                            "success": False,
                            "message": "Template file not found",
                        }
                    )
                    continue

            # Generate improvements
            improvements = self.generate_template_improvements(template_name)

            # Apply improvements
            result = self.feedback_loop.apply_feedback_to_template(
                template_name, template_path, improvements
            )

            if result:
                results.append(
                    {
                        "template": template_name,
                        "success": True,
                        "improvements": improvements,
                    }
                )

                # Log learning event
                self.feedback_loop.log_learning_event(
                    "template_improvement",
                    f"Improved template {template_name}",
                    {"template": template_name, "improvements": improvements},
                )
            else:
                results.append(
                    {
                        "template": template_name,
                        "success": False,
                        "message": "Failed to apply improvements",
                    }
                )

        return results


if __name__ == "__main__":
    # Example usage
    feedback_loop = FeedbackLoop()

    # Record some fix results
    feedback_loop.record_fix_result(
        patch_id="patch-1",
        template_name="keyerror_fix.py.template",
        bug_type="KeyError",
        success=True,
    )

    feedback_loop.record_fix_result(
        patch_id="patch-2",
        template_name="keyerror_fix.py.template",
        bug_type="KeyError",
        success=True,
    )

    feedback_loop.record_fix_result(
        patch_id="patch-3",
        template_name="keyerror_fix.py.template",
        bug_type="KeyError",
        success=False,
    )

    feedback_loop.record_fix_result(
        patch_id="patch-4",
        template_name="indexerror_fix.py.template",
        bug_type="IndexError",
        success=False,
    )

    # Get template effectiveness
    template_stats = feedback_loop.get_template_effectiveness()
    print(f"Template effectiveness: {template_stats}")

    # Get bug type effectiveness
    bug_type_stats = feedback_loop.get_bug_type_effectiveness()
    print(f"Bug type effectiveness: {bug_type_stats}")

    # Generate recommendations
    recommendations = feedback_loop.generate_recommendations()
    print(f"Recommendations: {recommendations}")

    # Fix improvement engine
    improvement = FixImprovement(feedback_loop)

    # Identify patterns
    patterns = improvement.identify_patterns()
    print(f"Identified patterns: {patterns}")

    # Generate template improvements
    template_improvements = improvement.generate_template_improvements(
        "keyerror_fix.py.template"
    )
    print(f"Template improvements: {template_improvements}")

    # Example of applying improvements (using actual templates directory)
    templates_dir = project_root / "modules" / "patch_generation" / "templates"
    if templates_dir.exists():
        results = improvement.apply_improvements(templates_dir)
        print(f"Improvement results: {results}")
