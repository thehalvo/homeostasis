"""
Backend Language Healing Metrics

This module provides tools for collecting, analyzing, and reporting metrics on
backend language healing efficiency. It tracks success rates, performance, and
cross-language effectiveness for self-healing capabilities.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)

# Default metrics file location
METRICS_DIR = Path(__file__).parent.parent.parent / "metrics"
METRICS_DIR.mkdir(exist_ok=True, parents=True)
DEFAULT_METRICS_FILE = METRICS_DIR / "backend_healing_metrics.json"


class HealingMetricsCollector:
    """
    Collector for healing efficiency metrics across backend languages.

    This class tracks:
    1. Success rates for error detection and fixes
    2. Performance metrics for healing operations
    3. Cross-language healing effectiveness
    4. Long-term healing impact
    """

    def __init__(self, metrics_file: Optional[Union[str, Path]] = None):
        """
        Initialize the metrics collector.

        Args:
            metrics_file: Path to the metrics JSON file
        """
        self.metrics_file = Path(metrics_file) if metrics_file else DEFAULT_METRICS_FILE

        # Initialize metrics structure
        self.metrics = {
            "global": {
                "healing_attempts": 0,
                "successful_healing": 0,
                "healing_success_rate": 0.0,
                "avg_detection_time": 0.0,
                "avg_analysis_time": 0.0,
                "avg_fix_generation_time": 0.0,
                "avg_fix_application_time": 0.0,
                "avg_total_healing_time": 0.0,
            },
            "languages": {},
            "error_types": {},
            "cross_language": {
                "attempts": 0,
                "successful": 0,
                "success_rate": 0.0,
                "by_source_target": {},
            },
            "time_series": {"daily": {}, "weekly": {}, "monthly": {}},
            "regression": {
                "prevented_errors": 0,
                "repeat_errors": 0,
                "prevention_rate": 0.0,
            },
        }

        # Load existing metrics if available
        self._load_metrics()

    def track_healing_attempt(
        self,
        error_data: Dict[str, Any],
        language: str,
        successful: bool,
        timings: Dict[str, float],
        is_cross_language: bool = False,
        source_language: Optional[str] = None,
    ) -> None:
        """
        Track a healing attempt for metrics collection.

        Args:
            error_data: Error data
            language: Target language
            successful: Whether the healing was successful
            timings: Timing information for different healing phases
            is_cross_language: Whether this was a cross-language healing
            source_language: Source language for cross-language healing
        """
        # Extract error type
        error_type = self._get_error_type(error_data, language)

        # Update global metrics
        self.metrics["global"]["healing_attempts"] += 1
        if successful:
            self.metrics["global"]["successful_healing"] += 1

        # Calculate success rate
        self.metrics["global"]["healing_success_rate"] = (
            self.metrics["global"]["successful_healing"]
            / self.metrics["global"]["healing_attempts"]
            * 100
        )

        # Update timing metrics
        self._update_timing_metrics(timings)

        # Update language-specific metrics
        language = language.lower()
        if language not in self.metrics["languages"]:
            self.metrics["languages"][language] = {
                "attempts": 0,
                "successful": 0,
                "success_rate": 0.0,
                "avg_healing_time": 0.0,
                "by_error_type": {},
            }

        lang_metrics = self.metrics["languages"][language]
        lang_metrics["attempts"] += 1
        if successful:
            lang_metrics["successful"] += 1

        lang_metrics["success_rate"] = (
            lang_metrics["successful"] / lang_metrics["attempts"] * 100
        )

        # Update timing for this language
        total_time = sum(timings.values())
        lang_metrics["avg_healing_time"] = (
            lang_metrics["avg_healing_time"] * (lang_metrics["attempts"] - 1)
            + total_time
        ) / lang_metrics["attempts"]

        # Update metrics by error type
        if error_type not in lang_metrics["by_error_type"]:
            lang_metrics["by_error_type"][error_type] = {
                "attempts": 0,
                "successful": 0,
                "success_rate": 0.0,
            }

        type_metrics = lang_metrics["by_error_type"][error_type]
        type_metrics["attempts"] += 1
        if successful:
            type_metrics["successful"] += 1

        type_metrics["success_rate"] = (
            type_metrics["successful"] / type_metrics["attempts"] * 100
        )

        # Update global error type metrics
        if error_type not in self.metrics["error_types"]:
            self.metrics["error_types"][error_type] = {
                "attempts": 0,
                "successful": 0,
                "success_rate": 0.0,
                "by_language": {},
            }

        global_type_metrics = self.metrics["error_types"][error_type]
        global_type_metrics["attempts"] += 1
        if successful:
            global_type_metrics["successful"] += 1

        global_type_metrics["success_rate"] = (
            global_type_metrics["successful"] / global_type_metrics["attempts"] * 100
        )

        # Add language to error type metrics
        if language not in global_type_metrics["by_language"]:
            global_type_metrics["by_language"][language] = {
                "attempts": 0,
                "successful": 0,
                "success_rate": 0.0,
            }

        lang_type_metrics = global_type_metrics["by_language"][language]
        lang_type_metrics["attempts"] += 1
        if successful:
            lang_type_metrics["successful"] += 1

        lang_type_metrics["success_rate"] = (
            lang_type_metrics["successful"] / lang_type_metrics["attempts"] * 100
        )

        # Update cross-language metrics if applicable
        if is_cross_language and source_language:
            source_language = source_language.lower()
            self.metrics["cross_language"]["attempts"] += 1
            if successful:
                self.metrics["cross_language"]["successful"] += 1

            self.metrics["cross_language"]["success_rate"] = (
                self.metrics["cross_language"]["successful"]
                / self.metrics["cross_language"]["attempts"]
                * 100
            )

            # Track by source and target language pair
            pair_key = f"{source_language}->{language}"
            if pair_key not in self.metrics["cross_language"]["by_source_target"]:
                self.metrics["cross_language"]["by_source_target"][pair_key] = {
                    "attempts": 0,
                    "successful": 0,
                    "success_rate": 0.0,
                }

            pair_metrics = self.metrics["cross_language"]["by_source_target"][pair_key]
            pair_metrics["attempts"] += 1
            if successful:
                pair_metrics["successful"] += 1

            pair_metrics["success_rate"] = (
                pair_metrics["successful"] / pair_metrics["attempts"] * 100
            )

        # Update time series metrics
        today = datetime.now().strftime("%Y-%m-%d")
        self._update_time_series(today, successful)

        # Save updated metrics
        self._save_metrics()

    def track_regression_prevention(
        self, error_data: Dict[str, Any], language: str, prevented: bool
    ) -> None:
        """
        Track regression error prevention.

        Args:
            error_data: Error data
            language: Language
            prevented: Whether recurrence was prevented
        """
        # Update regression metrics
        if prevented:
            self.metrics["regression"]["prevented_errors"] += 1
        else:
            self.metrics["regression"]["repeat_errors"] += 1

        total_errors = (
            self.metrics["regression"]["prevented_errors"]
            + self.metrics["regression"]["repeat_errors"]
        )

        self.metrics["regression"]["prevention_rate"] = (
            self.metrics["regression"]["prevented_errors"] / total_errors * 100
            if total_errors > 0
            else 0.0
        )

        # Save updated metrics
        self._save_metrics()

    def get_healing_efficiency(
        self,
        language: Optional[str] = None,
        error_type: Optional[str] = None,
        time_period: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get healing efficiency metrics.

        Args:
            language: Optional language filter
            error_type: Optional error type filter
            time_period: Optional time period filter (daily, weekly, monthly)

        Returns:
            Filtered metrics
        """
        result = {}

        # Filter by time period
        if time_period in ["daily", "weekly", "monthly"]:
            result["time_series"] = self.metrics["time_series"].get(time_period, {})
            return result

        # Start with global metrics
        result["global"] = self.metrics["global"].copy()

        # Filter by language
        if language:
            language = language.lower()
            if language in self.metrics["languages"]:
                result["language"] = self.metrics["languages"][language].copy()
            else:
                result["language"] = {"warning": f"No metrics for language: {language}"}
        else:
            result["languages"] = self.metrics["languages"].copy()

        # Filter by error type
        if error_type:
            if error_type in self.metrics["error_types"]:
                result["error_type"] = self.metrics["error_types"][error_type].copy()
            else:
                result["error_type"] = {
                    "warning": f"No metrics for error type: {error_type}"
                }

        # Include cross-language metrics
        result["cross_language"] = self.metrics["cross_language"].copy()

        # Include regression metrics
        result["regression"] = self.metrics["regression"].copy()

        return result

    def get_language_comparison(self) -> Dict[str, Any]:
        """
        Get a comparison of healing metrics across languages.

        Returns:
            Language comparison metrics
        """
        comparison = {
            "success_rates": {},
            "performance": {},
            "error_coverage": {},
            "cross_language_effectiveness": {},
        }

        # Compare success rates
        for language, metrics in self.metrics["languages"].items():
            comparison["success_rates"][language] = {
                "success_rate": metrics["success_rate"],
                "attempts": metrics["attempts"],
                "successful": metrics["successful"],
            }

        # Compare performance
        for language, metrics in self.metrics["languages"].items():
            comparison["performance"][language] = {
                "avg_healing_time": metrics["avg_healing_time"]
            }

        # Compare error coverage
        for language, metrics in self.metrics["languages"].items():
            comparison["error_coverage"][language] = {
                "error_types_covered": len(metrics["by_error_type"])
            }

        # Compare cross-language effectiveness
        for pair, metrics in self.metrics["cross_language"]["by_source_target"].items():
            source, target = pair.split("->")
            if target not in comparison["cross_language_effectiveness"]:
                comparison["cross_language_effectiveness"][target] = {}

            comparison["cross_language_effectiveness"][target][source] = {
                "success_rate": metrics["success_rate"],
                "attempts": metrics["attempts"],
                "successful": metrics["successful"],
            }

        return comparison

    def get_trends(
        self, period: str = "weekly", last_n: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get healing efficiency trends over time.

        Args:
            period: Time period (daily, weekly, monthly)
            last_n: Optional limit to last N periods

        Returns:
            Trend metrics
        """
        if period not in ["daily", "weekly", "monthly"]:
            return {"error": f"Invalid period: {period}"}

        # Get time series data
        time_series = self.metrics["time_series"].get(period, {})

        # Sort by date
        sorted_data = sorted(time_series.items())

        # Limit to last N periods if specified
        if last_n and last_n > 0:
            sorted_data = sorted_data[-last_n:]

        # Extract metrics
        dates = [item[0] for item in sorted_data]
        attempts = [item[1]["attempts"] for item in sorted_data]
        successful = [item[1]["successful"] for item in sorted_data]
        success_rates = [item[1]["success_rate"] for item in sorted_data]

        # Calculate trends
        trends = {
            "dates": dates,
            "attempts": attempts,
            "successful": successful,
            "success_rates": success_rates,
            "trend_direction": "stable",
        }

        # Determine trend direction
        if len(success_rates) >= 2:
            start_rate = success_rates[0]
            end_rate = success_rates[-1]

            if end_rate > start_rate * 1.05:  # 5% improvement
                trends["trend_direction"] = "improving"
            elif end_rate < start_rate * 0.95:  # 5% decline
                trends["trend_direction"] = "declining"

        return trends

    def generate_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive healing efficiency report.

        Returns:
            Report data
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_healing_attempts": self.metrics["global"]["healing_attempts"],
                "overall_success_rate": self.metrics["global"]["healing_success_rate"],
                "languages_covered": len(self.metrics["languages"]),
                "error_types_covered": len(self.metrics["error_types"]),
                "cross_language_success_rate": self.metrics["cross_language"][
                    "success_rate"
                ],
                "regression_prevention_rate": self.metrics["regression"][
                    "prevention_rate"
                ],
            },
            "performance": {
                "avg_total_healing_time": self.metrics["global"][
                    "avg_total_healing_time"
                ],
                "avg_detection_time": self.metrics["global"]["avg_detection_time"],
                "avg_analysis_time": self.metrics["global"]["avg_analysis_time"],
                "avg_fix_generation_time": self.metrics["global"][
                    "avg_fix_generation_time"
                ],
                "avg_fix_application_time": self.metrics["global"][
                    "avg_fix_application_time"
                ],
            },
            "top_languages": self._get_top_performers("languages", 5),
            "top_error_types": self._get_top_performers("error_types", 5),
            "trends": self.get_trends(period="weekly", last_n=10),
            "language_comparison": self.get_language_comparison(),
        }

        return report

    def export_metrics(
        self, format: str = "json", output_file: Optional[Union[str, Path]] = None
    ) -> Optional[str]:
        """
        Export metrics to a file.

        Args:
            format: Export format (json or csv)
            output_file: Optional output file path

        Returns:
            Output file path or None if export failed
        """
        if format.lower() not in ["json", "csv"]:
            logger.error(f"Unsupported export format: {format}")
            return None

        # Default output file
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = (
                METRICS_DIR / f"healing_metrics_export_{timestamp}.{format.lower()}"
            )
        else:
            output_file = Path(output_file)

        try:
            if format.lower() == "json":
                with open(output_file, "w") as f:
                    json.dump(self.metrics, f, indent=2)
            elif format.lower() == "csv":
                self._export_to_csv(output_file)

            logger.info(f"Exported metrics to {output_file}")
            return str(output_file)
        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")
            return None

    def reset_metrics(self) -> None:
        """Reset all metrics to initial values."""
        self.__init__(self.metrics_file)
        logger.info("Metrics have been reset")

    def _load_metrics(self) -> None:
        """Load metrics from the metrics file."""
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, "r") as f:
                    loaded_metrics = json.load(f)

                    # Merge with default structure
                    self._update_nested_dict(self.metrics, loaded_metrics)

                logger.info(f"Loaded metrics from {self.metrics_file}")
            except Exception as e:
                logger.warning(f"Error loading metrics from {self.metrics_file}: {e}")

    def _save_metrics(self) -> None:
        """Save metrics to the metrics file."""
        try:
            with open(self.metrics_file, "w") as f:
                json.dump(self.metrics, f, indent=2)

            logger.debug(f"Saved metrics to {self.metrics_file}")
        except Exception as e:
            logger.error(f"Error saving metrics to {self.metrics_file}: {e}")

    def _update_nested_dict(self, d1: Dict, d2: Dict) -> None:
        """
        Update a nested dictionary with values from another nested dictionary.

        Args:
            d1: Target dictionary to update
            d2: Source dictionary with new values
        """
        for k, v in d2.items():
            if k in d1 and isinstance(v, dict) and isinstance(d1[k], dict):
                self._update_nested_dict(d1[k], v)
            else:
                d1[k] = v

    def _get_error_type(self, error_data: Dict[str, Any], language: str) -> str:
        """
        Get the error type from error data.

        Args:
            error_data: Error data
            language: Language identifier

        Returns:
            Error type
        """
        if language == "python":
            return error_data.get("exception_type", "Unknown")
        elif language == "javascript":
            return error_data.get("name", "Unknown")
        elif language == "java":
            return error_data.get("exception_class", "Unknown")
        elif language == "go":
            return error_data.get("error_type", "Unknown")
        else:
            return error_data.get("error_type", "Unknown")

    def _update_timing_metrics(self, timings: Dict[str, float]) -> None:
        """
        Update global timing metrics.

        Args:
            timings: Timing information for different healing phases
        """
        global_metrics = self.metrics["global"]
        attempts = global_metrics["healing_attempts"]

        # Update each timing metric
        for phase, default_metric in [
            ("detection", "avg_detection_time"),
            ("analysis", "avg_analysis_time"),
            ("fix_generation", "avg_fix_generation_time"),
            ("fix_application", "avg_fix_application_time"),
        ]:
            if phase in timings:
                current_avg = global_metrics[default_metric]
                time_value = timings[phase]

                # Calculate new average
                if attempts == 1:
                    global_metrics[default_metric] = time_value
                else:
                    global_metrics[default_metric] = (
                        current_avg * (attempts - 1) + time_value
                    ) / attempts

        # Calculate total healing time
        total_time = sum(timings.values())

        if attempts == 1:
            global_metrics["avg_total_healing_time"] = total_time
        else:
            global_metrics["avg_total_healing_time"] = (
                global_metrics["avg_total_healing_time"] * (attempts - 1) + total_time
            ) / attempts

    def _update_time_series(self, date_str: str, successful: bool) -> None:
        """
        Update time series metrics.

        Args:
            date_str: Date string (YYYY-MM-DD)
            successful: Whether healing was successful
        """
        # Update daily metrics
        daily = self.metrics["time_series"].setdefault("daily", {})
        if date_str not in daily:
            daily[date_str] = {"attempts": 0, "successful": 0, "success_rate": 0.0}

        daily[date_str]["attempts"] += 1
        if successful:
            daily[date_str]["successful"] += 1

        daily[date_str]["success_rate"] = (
            daily[date_str]["successful"] / daily[date_str]["attempts"] * 100
        )

        # Update weekly metrics
        date = datetime.strptime(date_str, "%Y-%m-%d")
        week_start = (date - timedelta(days=date.weekday())).strftime("%Y-%m-%d")

        weekly = self.metrics["time_series"].setdefault("weekly", {})
        if week_start not in weekly:
            weekly[week_start] = {"attempts": 0, "successful": 0, "success_rate": 0.0}

        weekly[week_start]["attempts"] += 1
        if successful:
            weekly[week_start]["successful"] += 1

        weekly[week_start]["success_rate"] = (
            weekly[week_start]["successful"] / weekly[week_start]["attempts"] * 100
        )

        # Update monthly metrics
        month_start = date.strftime("%Y-%m-01")

        monthly = self.metrics["time_series"].setdefault("monthly", {})
        if month_start not in monthly:
            monthly[month_start] = {"attempts": 0, "successful": 0, "success_rate": 0.0}

        monthly[month_start]["attempts"] += 1
        if successful:
            monthly[month_start]["successful"] += 1

        monthly[month_start]["success_rate"] = (
            monthly[month_start]["successful"] / monthly[month_start]["attempts"] * 100
        )

    def _get_top_performers(self, category: str, limit: int = 5) -> Dict[str, Any]:
        """
        Get top performing items in a category by success rate.

        Args:
            category: Category to analyze (languages or error_types)
            limit: Maximum number of items to return

        Returns:
            Top performers with metrics
        """
        if category not in ["languages", "error_types"]:
            return {}

        items = self.metrics[category].items()

        # Filter to items with at least 5 attempts for statistical significance
        significant_items = [
            (name, metrics) for name, metrics in items if metrics["attempts"] >= 5
        ]

        # Sort by success rate (descending)
        sorted_items = sorted(
            significant_items, key=lambda x: x[1]["success_rate"], reverse=True
        )

        # Take top N
        top_items = sorted_items[:limit]

        # Format result
        result = {}
        for name, metrics in top_items:
            result[name] = {
                "success_rate": metrics["success_rate"],
                "attempts": metrics["attempts"],
                "successful": metrics["successful"],
            }

        return result

    def _export_to_csv(self, output_file: Path) -> None:
        """
        Export metrics to CSV format.

        Args:
            output_file: Output file path
        """
        # This is a simplified CSV export that focuses on the most important metrics
        with open(output_file, "w") as f:
            # Write header
            f.write("Category,Metric,Value\n")

            # Write global metrics
            for key, value in self.metrics["global"].items():
                f.write(f"Global,{key},{value}\n")

            # Write language metrics
            for language, metrics in self.metrics["languages"].items():
                f.write(f"Language_{language},success_rate,{metrics['success_rate']}\n")
                f.write(f"Language_{language},attempts,{metrics['attempts']}\n")
                f.write(f"Language_{language},successful,{metrics['successful']}\n")

            # Write cross-language metrics
            for key, value in self.metrics["cross_language"].items():
                if not isinstance(value, dict):
                    f.write(f"CrossLanguage,{key},{value}\n")

            # Write regression metrics
            for key, value in self.metrics["regression"].items():
                f.write(f"Regression,{key},{value}\n")


def track_healing_event(
    orchestrator,
    error_data: Dict[str, Any],
    language: str,
    fix_data: Optional[Dict[str, Any]] = None,
    successful: bool = False,
    timings: Optional[Dict[str, float]] = None,
    metrics_collector: Optional[HealingMetricsCollector] = None,
) -> None:
    """
    Track a healing event in the metrics system.

    Args:
        orchestrator: Orchestrator instance
        error_data: Error data
        language: Language identifier
        fix_data: Optional fix data
        successful: Whether healing was successful
        timings: Optional timing information
        metrics_collector: Optional metrics collector instance
    """
    # Create or get metrics collector
    if metrics_collector is None:
        metrics_collector = HealingMetricsCollector()

    # Default timings if not provided
    if timings is None:
        timings = {
            "detection": 0.1,
            "analysis": 0.5,
            "fix_generation": 1.0,
            "fix_application": 0.5,
        }

    # Determine if this was a cross-language fix
    is_cross_language = False
    source_language = None

    if (
        fix_data
        and fix_data.get("used_cross_language")
        and "source_languages" in fix_data
    ):
        is_cross_language = True
        # Use the first source language if multiple are provided
        source_languages = fix_data["source_languages"]
        if source_languages:
            source_language = source_languages[0]

    # Track the healing attempt
    metrics_collector.track_healing_attempt(
        error_data=error_data,
        language=language,
        successful=successful,
        timings=timings,
        is_cross_language=is_cross_language,
        source_language=source_language,
    )

    # Return the metrics collector for potential further use
    return metrics_collector


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Create metrics collector
    collector = HealingMetricsCollector()

    # Add some sample data for testing
    languages = ["python", "javascript", "java", "go"]
    error_types = {
        "python": ["KeyError", "TypeError", "AttributeError", "IndexError"],
        "javascript": ["TypeError", "ReferenceError", "SyntaxError"],
        "java": [
            "NullPointerException",
            "ClassCastException",
            "ArrayIndexOutOfBoundsException",
        ],
        "go": [
            "nil pointer dereference",
            "index out of range",
            "concurrent map writes",
        ],
    }

    # Generate sample metrics
    logger.info("Generating sample metrics data...")
    for _ in range(100):
        # Random language
        import random

        language = random.choice(languages)
        error_type = random.choice(error_types[language])

        # Create error data
        error_data = {"error_type": error_type}

        # Random success
        successful = random.random() < 0.7  # 70% success rate

        # Random timings
        timings = {
            "detection": random.uniform(0.1, 0.5),
            "analysis": random.uniform(0.3, 1.0),
            "fix_generation": random.uniform(0.5, 2.0),
            "fix_application": random.uniform(0.2, 1.0),
        }

        # Random cross-language (10% chance)
        is_cross_language = random.random() < 0.1
        source_language = None
        if is_cross_language:
            source_candidates = [lang for lang in languages if lang != language]
            source_language = random.choice(source_candidates)

            # Create fix data
            fix_data = {
                "used_cross_language": True,
                "source_languages": [source_language],
            }
        else:
            fix_data = {}

        # Track the metrics
        collector.track_healing_attempt(
            error_data=error_data,
            language=language,
            successful=successful,
            timings=timings,
            is_cross_language=is_cross_language,
            source_language=source_language,
        )

    # Generate report
    report = collector.generate_report()
    logger.info("\nHealing Efficiency Report:")
    logger.info(
        f"Total healing attempts: {report['summary']['total_healing_attempts']}"
    )
    logger.info(
        f"Overall success rate: {report['summary']['overall_success_rate']:.1f}%"
    )
    logger.info(f"Languages covered: {report['summary']['languages_covered']}")
    logger.info(f"Error types covered: {report['summary']['error_types_covered']}")

    logger.info("\nTop performing languages:")
    for lang, metrics in report["top_languages"].items():
        logger.info(
            f"  {lang}: {metrics['success_rate']:.1f}% success rate from {metrics['attempts']} attempts"
        )

    logger.info("\nTrend direction:", report["trends"]["trend_direction"])

    # Export metrics
    output_path = collector.export_metrics()
    if output_path:
        logger.info(f"\nMetrics exported to: {output_path}")
