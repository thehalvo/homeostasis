"""
Healing Effectiveness Calculator

This module provides tools to calculate and analyze the effectiveness of healing
operations in the Homeostasis framework.
"""

import json
import logging
import statistics
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class HealingMetrics:
    """Metrics for a single healing operation"""
    healing_id: str
    error_type: str
    language: str
    framework: Optional[str] = None
    success: bool = False
    time_to_detect: float = 0.0  # seconds
    time_to_analyze: float = 0.0
    time_to_generate_patch: float = 0.0
    time_to_test: float = 0.0
    time_to_apply: float = 0.0
    total_time: float = 0.0
    patches_generated: int = 0
    patches_tested: int = 0
    patches_applied: int = 0
    test_pass_rate: float = 0.0
    rollback_required: bool = False
    human_intervention_required: bool = False
    confidence_score: float = 0.0
    complexity_score: float = 0.0
    impact_score: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EffectivenessReport:
    """Overall effectiveness report"""
    total_healings: int
    successful_healings: int
    failed_healings: int
    success_rate: float
    average_healing_time: float
    median_healing_time: float
    fastest_healing_time: float
    slowest_healing_time: float
    average_patches_per_healing: float
    rollback_rate: float
    human_intervention_rate: float
    by_language: Dict[str, Dict[str, float]]
    by_error_type: Dict[str, Dict[str, float]]
    by_time_period: Dict[str, Dict[str, float]]
    confidence_distribution: Dict[str, int]
    complexity_distribution: Dict[str, int]
    recommendations: List[str]
    generated_at: datetime = field(default_factory=datetime.now)


class EffectivenessCalculator:
    """Calculate healing effectiveness metrics"""
    
    def __init__(self, data_path: Optional[Path] = None):
        self.data_path = data_path or Path.home() / ".homeostasis" / "metrics"
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.data_path / "healing_metrics.json"
        self.metrics: List[HealingMetrics] = []
        self._load_metrics()
    
    def _load_metrics(self):
        """Load existing metrics from file"""
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, 'r') as f:
                    data = json.load(f)
                    self.metrics = [
                        HealingMetrics(**m) for m in data.get("metrics", [])
                    ]
            except Exception as e:
                logger.error(f"Failed to load metrics: {e}")
    
    def _save_metrics(self):
        """Save metrics to file"""
        try:
            data = {
                "metrics": [
                    {**m.__dict__, "timestamp": m.timestamp.isoformat()}
                    for m in self.metrics
                ]
            }
            with open(self.metrics_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
    
    def add_metric(self, metric: HealingMetrics):
        """Add a new healing metric"""
        # Calculate derived metrics
        metric.total_time = (
            metric.time_to_detect +
            metric.time_to_analyze +
            metric.time_to_generate_patch +
            metric.time_to_test +
            metric.time_to_apply
        )
        
        if metric.patches_tested > 0:
            metric.test_pass_rate = metric.patches_applied / metric.patches_tested
        
        self.metrics.append(metric)
        self._save_metrics()
    
    def calculate_effectiveness(self, 
                               start_date: Optional[datetime] = None,
                               end_date: Optional[datetime] = None,
                               language: Optional[str] = None,
                               error_type: Optional[str] = None) -> EffectivenessReport:
        """Calculate overall effectiveness metrics"""
        
        # Filter metrics
        filtered_metrics = self._filter_metrics(
            start_date, end_date, language, error_type
        )
        
        if not filtered_metrics:
            return self._empty_report()
        
        # Basic counts
        total = len(filtered_metrics)
        successful = sum(1 for m in filtered_metrics if m.success)
        failed = total - successful
        
        # Time metrics
        healing_times = [m.total_time for m in filtered_metrics if m.success]
        avg_time = statistics.mean(healing_times) if healing_times else 0
        median_time = statistics.median(healing_times) if healing_times else 0
        fastest_time = min(healing_times) if healing_times else 0
        slowest_time = max(healing_times) if healing_times else 0
        
        # Patch metrics
        avg_patches = statistics.mean([m.patches_generated for m in filtered_metrics])
        
        # Intervention metrics
        rollback_count = sum(1 for m in filtered_metrics if m.rollback_required)
        human_intervention_count = sum(1 for m in filtered_metrics if m.human_intervention_required)
        
        # By language breakdown
        by_language = self._calculate_by_dimension(filtered_metrics, "language")
        
        # By error type breakdown
        by_error_type = self._calculate_by_dimension(filtered_metrics, "error_type")
        
        # By time period breakdown
        by_time_period = self._calculate_time_trends(filtered_metrics)
        
        # Confidence and complexity distributions
        confidence_dist = self._calculate_distribution(filtered_metrics, "confidence_score")
        complexity_dist = self._calculate_distribution(filtered_metrics, "complexity_score")
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            filtered_metrics, 
            successful / total if total > 0 else 0
        )
        
        return EffectivenessReport(
            total_healings=total,
            successful_healings=successful,
            failed_healings=failed,
            success_rate=successful / total if total > 0 else 0,
            average_healing_time=avg_time,
            median_healing_time=median_time,
            fastest_healing_time=fastest_time,
            slowest_healing_time=slowest_time,
            average_patches_per_healing=avg_patches,
            rollback_rate=rollback_count / total if total > 0 else 0,
            human_intervention_rate=human_intervention_count / total if total > 0 else 0,
            by_language=by_language,
            by_error_type=by_error_type,
            by_time_period=by_time_period,
            confidence_distribution=confidence_dist,
            complexity_distribution=complexity_dist,
            recommendations=recommendations
        )
    
    def _filter_metrics(self, start_date: Optional[datetime],
                       end_date: Optional[datetime],
                       language: Optional[str],
                       error_type: Optional[str]) -> List[HealingMetrics]:
        """Filter metrics based on criteria"""
        filtered = self.metrics
        
        if start_date:
            filtered = [m for m in filtered if m.timestamp >= start_date]
        
        if end_date:
            filtered = [m for m in filtered if m.timestamp <= end_date]
        
        if language:
            filtered = [m for m in filtered if m.language == language]
        
        if error_type:
            filtered = [m for m in filtered if m.error_type == error_type]
        
        return filtered
    
    def _calculate_by_dimension(self, metrics: List[HealingMetrics], 
                               dimension: str) -> Dict[str, Dict[str, float]]:
        """Calculate metrics grouped by a dimension"""
        grouped = defaultdict(list)
        
        for metric in metrics:
            key = getattr(metric, dimension)
            grouped[key].append(metric)
        
        results = {}
        for key, group_metrics in grouped.items():
            successful = sum(1 for m in group_metrics if m.success)
            total = len(group_metrics)
            
            results[key] = {
                "total": total,
                "successful": successful,
                "success_rate": successful / total if total > 0 else 0,
                "average_time": statistics.mean([m.total_time for m in group_metrics if m.success]) if successful > 0 else 0,
                "average_patches": statistics.mean([m.patches_generated for m in group_metrics])
            }
        
        return results
    
    def _calculate_time_trends(self, metrics: List[HealingMetrics]) -> Dict[str, Dict[str, float]]:
        """Calculate metrics over time periods"""
        # Group by day
        daily_groups = defaultdict(list)
        
        for metric in metrics:
            day_key = metric.timestamp.strftime("%Y-%m-%d")
            daily_groups[day_key].append(metric)
        
        results = {}
        for day, day_metrics in sorted(daily_groups.items()):
            successful = sum(1 for m in day_metrics if m.success)
            total = len(day_metrics)
            
            results[day] = {
                "total": total,
                "successful": successful,
                "success_rate": successful / total if total > 0 else 0,
                "average_time": statistics.mean([m.total_time for m in day_metrics if m.success]) if successful > 0 else 0
            }
        
        return results
    
    def _calculate_distribution(self, metrics: List[HealingMetrics], 
                               field: str) -> Dict[str, int]:
        """Calculate distribution of a numeric field"""
        values = [getattr(m, field) for m in metrics]
        
        if not values:
            return {}
        
        # Create bins
        bins = {
            "0-0.2": 0,
            "0.2-0.4": 0,
            "0.4-0.6": 0,
            "0.6-0.8": 0,
            "0.8-1.0": 0
        }
        
        for value in values:
            if value <= 0.2:
                bins["0-0.2"] += 1
            elif value <= 0.4:
                bins["0.2-0.4"] += 1
            elif value <= 0.6:
                bins["0.4-0.6"] += 1
            elif value <= 0.8:
                bins["0.6-0.8"] += 1
            else:
                bins["0.8-1.0"] += 1
        
        return bins
    
    def _generate_recommendations(self, metrics: List[HealingMetrics], 
                                 success_rate: float) -> List[str]:
        """Generate recommendations based on metrics"""
        recommendations = []
        
        # Success rate recommendations
        if success_rate < 0.7:
            recommendations.append(
                "Success rate is below 70%. Consider reviewing failed healings "
                "to identify common patterns and improve rule coverage."
            )
        
        # Time-based recommendations
        slow_healings = [m for m in metrics if m.total_time > 300]  # 5 minutes
        if len(slow_healings) > len(metrics) * 0.2:
            recommendations.append(
                "Over 20% of healings take more than 5 minutes. "
                "Consider optimizing analysis and patch generation algorithms."
            )
        
        # Rollback recommendations
        rollback_metrics = [m for m in metrics if m.rollback_required]
        if rollback_metrics:
            rollback_rate = len(rollback_metrics) / len(metrics)
            if rollback_rate > 0.1:
                recommendations.append(
                    f"Rollback rate is {rollback_rate:.1%}. "
                    "Review test coverage and patch validation processes."
                )
        
        # Language-specific recommendations
        by_language = self._calculate_by_dimension(metrics, "language")
        for lang, stats in by_language.items():
            if stats["success_rate"] < 0.6:
                recommendations.append(
                    f"{lang} has a low success rate ({stats['success_rate']:.1%}). "
                    f"Consider adding more {lang}-specific healing rules."
                )
        
        # Confidence recommendations
        low_confidence = [m for m in metrics if m.confidence_score < 0.5]
        if len(low_confidence) > len(metrics) * 0.3:
            recommendations.append(
                "Over 30% of healings have low confidence scores. "
                "Consider improving error analysis and pattern matching."
            )
        
        return recommendations
    
    def _empty_report(self) -> EffectivenessReport:
        """Return an empty report"""
        return EffectivenessReport(
            total_healings=0,
            successful_healings=0,
            failed_healings=0,
            success_rate=0.0,
            average_healing_time=0.0,
            median_healing_time=0.0,
            fastest_healing_time=0.0,
            slowest_healing_time=0.0,
            average_patches_per_healing=0.0,
            rollback_rate=0.0,
            human_intervention_rate=0.0,
            by_language={},
            by_error_type={},
            by_time_period={},
            confidence_distribution={},
            complexity_distribution={},
            recommendations=["No data available for analysis"]
        )
    
    def export_report(self, report: EffectivenessReport, 
                     format: str = "json") -> str:
        """Export effectiveness report"""
        if format == "json":
            return json.dumps(report.__dict__, indent=2, default=str)
        
        elif format == "html":
            return self._generate_html_report(report)
        
        elif format == "markdown":
            return self._generate_markdown_report(report)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _generate_html_report(self, report: EffectivenessReport) -> str:
        """Generate HTML report"""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Healing Effectiveness Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px; border: 1px solid #ddd; }}
        .metric-value {{ font-size: 24px; font-weight: bold; }}
        .metric-label {{ font-size: 14px; color: #666; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .recommendation {{ background-color: #f0f8ff; padding: 10px; margin: 5px 0; }}
    </style>
</head>
<body>
    <h1>Healing Effectiveness Report</h1>
    <p>Generated at: {report.generated_at}</p>
    
    <h2>Overall Metrics</h2>
    <div class="metric">
        <div class="metric-value">{report.total_healings}</div>
        <div class="metric-label">Total Healings</div>
    </div>
    <div class="metric">
        <div class="metric-value">{report.success_rate:.1%}</div>
        <div class="metric-label">Success Rate</div>
    </div>
    <div class="metric">
        <div class="metric-value">{report.average_healing_time:.1f}s</div>
        <div class="metric-label">Average Time</div>
    </div>
    
    <h2>Performance by Language</h2>
    <table>
        <tr>
            <th>Language</th>
            <th>Total</th>
            <th>Success Rate</th>
            <th>Avg Time</th>
        </tr>
        {"".join(f'''
        <tr>
            <td>{lang}</td>
            <td>{stats['total']}</td>
            <td>{stats['success_rate']:.1%}</td>
            <td>{stats['average_time']:.1f}s</td>
        </tr>
        ''' for lang, stats in report.by_language.items())}
    </table>
    
    <h2>Recommendations</h2>
    {"".join(f'<div class="recommendation">{rec}</div>' for rec in report.recommendations)}
</body>
</html>
"""
        return html
    
    def _generate_markdown_report(self, report: EffectivenessReport) -> str:
        """Generate Markdown report"""
        md = f"""# Healing Effectiveness Report

Generated at: {report.generated_at}

## Overall Metrics

- **Total Healings**: {report.total_healings}
- **Successful Healings**: {report.successful_healings}
- **Failed Healings**: {report.failed_healings}
- **Success Rate**: {report.success_rate:.1%}
- **Average Healing Time**: {report.average_healing_time:.1f}s
- **Median Healing Time**: {report.median_healing_time:.1f}s
- **Rollback Rate**: {report.rollback_rate:.1%}
- **Human Intervention Rate**: {report.human_intervention_rate:.1%}

## Performance by Language

| Language | Total | Success Rate | Avg Time |
|----------|-------|--------------|----------|
"""
        
        for lang, stats in report.by_language.items():
            md += f"| {lang} | {stats['total']} | {stats['success_rate']:.1%} | {stats['average_time']:.1f}s |\n"
        
        md += "\n## Recommendations\n\n"
        for rec in report.recommendations:
            md += f"- {rec}\n"
        
        return md
    
    def compare_periods(self, period1_start: datetime, period1_end: datetime,
                       period2_start: datetime, period2_end: datetime) -> Dict[str, Any]:
        """Compare effectiveness between two time periods"""
        report1 = self.calculate_effectiveness(period1_start, period1_end)
        report2 = self.calculate_effectiveness(period2_start, period2_end)
        
        comparison = {
            "period1": {
                "start": period1_start,
                "end": period1_end,
                "metrics": report1
            },
            "period2": {
                "start": period2_start,
                "end": period2_end,
                "metrics": report2
            },
            "changes": {
                "success_rate_change": report2.success_rate - report1.success_rate,
                "average_time_change": report2.average_healing_time - report1.average_healing_time,
                "rollback_rate_change": report2.rollback_rate - report1.rollback_rate
            }
        }
        
        return comparison
    
    def predict_effectiveness(self, error_type: str, language: str, 
                            complexity: float) -> Tuple[float, float]:
        """Predict success probability and expected time for a healing"""
        # Filter similar past healings
        similar_metrics = [
            m for m in self.metrics
            if m.error_type == error_type and m.language == language
            and abs(m.complexity_score - complexity) < 0.2
        ]
        
        if not similar_metrics:
            # Fallback to language-only prediction
            similar_metrics = [
                m for m in self.metrics
                if m.language == language
            ]
        
        if not similar_metrics:
            # No data available
            return 0.5, 60.0  # Default 50% success, 1 minute
        
        # Calculate predictions
        success_rate = sum(1 for m in similar_metrics if m.success) / len(similar_metrics)
        avg_time = statistics.mean([m.total_time for m in similar_metrics if m.success]) if any(m.success for m in similar_metrics) else 60.0
        
        # Adjust for complexity
        success_rate *= (1 - complexity * 0.3)  # Higher complexity reduces success rate
        avg_time *= (1 + complexity * 0.5)  # Higher complexity increases time
        
        return min(max(success_rate, 0.0), 1.0), avg_time