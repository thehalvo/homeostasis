"""
Advanced Telemetry and Rule Enhancement System for Homeostasis.

This module provides comprehensive telemetry collection, advanced analytics,
and intelligent rule enhancement capabilities that learn from error patterns,
patch success rates, LLM performance, and feed insights back into the system
to continuously improve detection, classification, and healing effectiveness.
"""

import json
import logging
import statistics
import uuid
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

from modules.monitoring.feedback_loop import FeedbackLoop
from modules.monitoring.logger import MonitoringLogger
from modules.monitoring.metrics_collector import MetricsCollector
from modules.security.audit import get_audit_logger, log_event

logger = logging.getLogger(__name__)


class TelemetryDataPoint:
    """Represents a telemetry data point with rich metadata."""
    
    def __init__(self, event_type: str, timestamp: datetime = None, **kwargs):
        """
        Initialize a telemetry data point.
        
        Args:
            event_type: Type of telemetry event
            timestamp: Timestamp of the event
            **kwargs: Additional telemetry data
        """
        self.event_type = event_type
        self.timestamp = timestamp or datetime.utcnow()
        self.data = kwargs
        self.id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'id': self.id,
            'event_type': self.event_type,
            'timestamp': self.timestamp.isoformat(),
            'data': self.data
        }


class ErrorPatternAnalyzer:
    """Analyzes error patterns to identify trends and insights."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the error pattern analyzer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.monitoring_logger = MonitoringLogger("error_pattern_analyzer")
        
        # Pattern storage
        self.error_patterns = defaultdict(list)
        self.temporal_patterns = defaultdict(list)
        self.contextual_patterns = defaultdict(lambda: defaultdict(list))
        
    def analyze_error_patterns(self, error_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze error patterns from historical data.
        
        Args:
            error_data: List of error events
            
        Returns:
            Analysis results with identified patterns
        """
        if not error_data:
            return {'patterns': [], 'insights': []}
        
        # Group errors by type, file, time, etc.
        error_types = Counter()
        file_patterns = Counter()
        temporal_patterns = defaultdict(list)
        severity_patterns = Counter()
        
        for error in error_data:
            error_type = error.get('error_type', 'unknown')
            file_path = error.get('file_path', 'unknown')
            timestamp = error.get('timestamp')
            severity = error.get('severity', 'medium')
            
            error_types[error_type] += 1
            file_patterns[file_path] += 1
            severity_patterns[severity] += 1
            
            if timestamp:
                try:
                    if isinstance(timestamp, str):
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    else:
                        dt = datetime.fromtimestamp(timestamp)
                    
                    hour = dt.hour
                    day_of_week = dt.weekday()
                    
                    temporal_patterns['hourly'][hour] = temporal_patterns['hourly'].get(hour, 0) + 1
                    temporal_patterns['daily'][day_of_week] = temporal_patterns['daily'].get(day_of_week, 0) + 1
                    
                except Exception as e:
                    self.monitoring_logger.warning(f"Failed to parse timestamp {timestamp}: {e}")
        
        # Identify significant patterns
        patterns = []
        insights = []
        
        # Most common error types
        top_errors = error_types.most_common(5)
        if top_errors:
            patterns.append({
                'type': 'frequent_errors',
                'data': top_errors,
                'description': f"Top error types: {', '.join([f'{err}({count})' for err, count in top_errors])}"
            })
            
            # Insight: Focus on most common errors
            most_common = top_errors[0]
            if most_common[1] > len(error_data) * 0.3:  # More than 30% of errors
                insights.append({
                    'type': 'dominant_error_type',
                    'message': f"Error type '{most_common[0]}' accounts for {most_common[1]/len(error_data)*100:.1f}% of all errors",
                    'recommendation': f"Consider creating specialized detection and fixing rules for '{most_common[0]}'"
                })
        
        # File hotspots
        top_files = file_patterns.most_common(5)
        if top_files:
            patterns.append({
                'type': 'error_hotspots',
                'data': top_files,
                'description': f"Files with most errors: {', '.join([f'{file}({count})' for file, count in top_files])}"
            })
            
            # Insight: Code quality hotspots
            hotspot = top_files[0]
            if hotspot[1] > 5:  # More than 5 errors in one file
                insights.append({
                    'type': 'code_quality_hotspot',
                    'message': f"File '{hotspot[0]}' has {hotspot[1]} errors - potential code quality issue",
                    'recommendation': f"Consider refactoring or adding additional validation to '{hotspot[0]}'"
                })
        
        # Temporal patterns
        if temporal_patterns:
            patterns.append({
                'type': 'temporal_patterns',
                'data': dict(temporal_patterns),
                'description': "Error occurrence patterns by time"
            })
            
            # Check for time-based patterns
            if 'hourly' in temporal_patterns:
                hourly_data = temporal_patterns['hourly']
                peak_hour = max(hourly_data.items(), key=lambda x: x[1])
                if peak_hour[1] > max(hourly_data.values()) * 0.8:  # Peak hour has 80%+ of max
                    insights.append({
                        'type': 'temporal_pattern',
                        'message': f"Errors peak at hour {peak_hour[0]} with {peak_hour[1]} occurrences",
                        'recommendation': "Consider load balancing or resource scaling during peak hours"
                    })
        
        # Severity distribution
        if severity_patterns:
            patterns.append({
                'type': 'severity_distribution',
                'data': dict(severity_patterns),
                'description': f"Error severity distribution: {dict(severity_patterns)}"
            })
            
            # Check for high severity concentration
            total_errors = sum(severity_patterns.values())
            high_severity = severity_patterns.get('high', 0) + severity_patterns.get('critical', 0)
            if high_severity > total_errors * 0.3:  # More than 30% high/critical
                insights.append({
                    'type': 'high_severity_concentration',
                    'message': f"{high_severity/total_errors*100:.1f}% of errors are high/critical severity",
                    'recommendation': "Consider implementing more proactive monitoring and faster response times"
                })
        
        return {
            'patterns': patterns,
            'insights': insights,
            'summary': {
                'total_errors': len(error_data),
                'unique_error_types': len(error_types),
                'unique_files_affected': len(file_patterns),
                'analysis_timestamp': datetime.utcnow().isoformat()
            }
        }


class LLMPerformanceAnalyzer:
    """Analyzes LLM performance metrics and identifies optimization opportunities."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the LLM performance analyzer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.monitoring_logger = MonitoringLogger("llm_performance_analyzer")
        
    def analyze_llm_performance(self, llm_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze LLM performance from interaction data.
        
        Args:
            llm_data: List of LLM interaction records
            
        Returns:
            Performance analysis results
        """
        if not llm_data:
            return {'performance_metrics': {}, 'insights': []}
        
        # Group by provider and model
        provider_performance = defaultdict(lambda: defaultdict(list))
        model_performance = defaultdict(list)
        token_usage = defaultdict(list)
        response_quality = defaultdict(list)
        cost_analysis = defaultdict(list)
        
        for interaction in llm_data:
            provider = interaction.get('provider', 'unknown')
            model = interaction.get('model', 'unknown')
            
            # Performance metrics
            response_time = interaction.get('response_time')
            tokens_used = interaction.get('tokens_used')
            cost = interaction.get('cost')
            success = interaction.get('success', False)
            confidence = interaction.get('confidence')
            
            if response_time is not None:
                provider_performance[provider]['response_times'].append(response_time)
                model_performance[f"{provider}:{model}"].append(response_time)
            
            if tokens_used is not None:
                token_usage[provider].append(tokens_used)
            
            if cost is not None:
                cost_analysis[provider].append(cost)
            
            if confidence is not None:
                response_quality[provider].append(confidence)
            
            # Success rate tracking
            provider_performance[provider]['successes'].append(success)
        
        # Calculate performance metrics
        performance_metrics = {}
        insights = []
        
        for provider, metrics in provider_performance.items():
            if 'response_times' in metrics and metrics['response_times']:
                avg_response_time = statistics.mean(metrics['response_times'])
                performance_metrics[provider] = {
                    'avg_response_time': avg_response_time,
                    'min_response_time': min(metrics['response_times']),
                    'max_response_time': max(metrics['response_times']),
                    'response_time_std': statistics.stdev(metrics['response_times']) if len(metrics['response_times']) > 1 else 0
                }
                
                # Success rate
                if 'successes' in metrics:
                    success_rate = sum(metrics['successes']) / len(metrics['successes'])
                    performance_metrics[provider]['success_rate'] = success_rate
                    
                    # Insight: Provider performance comparison
                    if success_rate < 0.8:  # Less than 80% success rate
                        insights.append({
                            'type': 'low_provider_success_rate',
                            'provider': provider,
                            'message': f"Provider {provider} has low success rate: {success_rate*100:.1f}%",
                            'recommendation': "Consider reviewing prompts or switching providers for better results"
                        })
                
                # Token usage analysis
                if provider in token_usage and token_usage[provider]:
                    avg_tokens = statistics.mean(token_usage[provider])
                    performance_metrics[provider]['avg_tokens_used'] = avg_tokens
                    
                    # Insight: High token usage
                    if avg_tokens > 2000:  # Arbitrary threshold
                        insights.append({
                            'type': 'high_token_usage',
                            'provider': provider,
                            'message': f"Provider {provider} uses high average tokens: {avg_tokens:.0f}",
                            'recommendation': "Consider optimizing prompts to reduce token usage and costs"
                        })
                
                # Cost analysis
                if provider in cost_analysis and cost_analysis[provider]:
                    total_cost = sum(cost_analysis[provider])
                    avg_cost = statistics.mean(cost_analysis[provider])
                    performance_metrics[provider]['total_cost'] = total_cost
                    performance_metrics[provider]['avg_cost_per_request'] = avg_cost
                    
                    # Insight: Cost optimization
                    if total_cost > 10:  # Arbitrary threshold
                        insights.append({
                            'type': 'high_cost_usage',
                            'provider': provider,
                            'message': f"Provider {provider} has high total cost: ${total_cost:.2f}",
                            'recommendation': "Consider cost optimization strategies or provider alternatives"
                        })
                
                # Response quality analysis
                if provider in response_quality and response_quality[provider]:
                    avg_confidence = statistics.mean(response_quality[provider])
                    performance_metrics[provider]['avg_confidence'] = avg_confidence
                    
                    # Insight: Low confidence responses
                    if avg_confidence < 0.7:  # Less than 70% average confidence
                        insights.append({
                            'type': 'low_response_confidence',
                            'provider': provider,
                            'message': f"Provider {provider} has low average confidence: {avg_confidence*100:.1f}%",
                            'recommendation': "Consider improving prompt engineering or model selection"
                        })
        
        # Provider comparison insights
        if len(performance_metrics) > 1:
            # Compare response times
            response_times = {p: m.get('avg_response_time', float('inf')) for p, m in performance_metrics.items()}
            fastest_provider = min(response_times, key=response_times.get)
            slowest_provider = max(response_times, key=response_times.get)
            
            time_diff = response_times[slowest_provider] - response_times[fastest_provider]
            if time_diff > 1.0:  # More than 1 second difference
                insights.append({
                    'type': 'provider_performance_comparison',
                    'message': f"Provider {fastest_provider} is {time_diff:.1f}s faster than {slowest_provider}",
                    'recommendation': f"Consider using {fastest_provider} for time-sensitive operations"
                })
            
            # Compare success rates
            success_rates = {p: m.get('success_rate', 0) for p, m in performance_metrics.items()}
            best_provider = max(success_rates, key=success_rates.get)
            worst_provider = min(success_rates, key=success_rates.get)
            
            rate_diff = success_rates[best_provider] - success_rates[worst_provider]
            if rate_diff > 0.1:  # More than 10% difference
                insights.append({
                    'type': 'provider_success_comparison',
                    'message': f"Provider {best_provider} has {rate_diff*100:.1f}% higher success rate than {worst_provider}",
                    'recommendation': f"Consider prioritizing {best_provider} for critical fixes"
                })
        
        return {
            'performance_metrics': performance_metrics,
            'insights': insights,
            'summary': {
                'total_interactions': len(llm_data),
                'providers_analyzed': len(performance_metrics),
                'analysis_timestamp': datetime.utcnow().isoformat()
            }
        }


class RuleOptimizer:
    """Optimizes detection and classification rules based on performance data."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the rule optimizer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.monitoring_logger = MonitoringLogger("rule_optimizer")
        
    def analyze_rule_performance(self, rule_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze rule performance and identify optimization opportunities.
        
        Args:
            rule_data: List of rule execution records
            
        Returns:
            Rule performance analysis
        """
        if not rule_data:
            return {'rule_metrics': {}, 'optimization_suggestions': []}
        
        # Group by rule
        rule_performance = defaultdict(lambda: {
            'executions': 0,
            'matches': 0,
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'execution_times': [],
            'confidence_scores': []
        })
        
        for execution in rule_data:
            rule_id = execution.get('rule_id', 'unknown')
            matched = execution.get('matched', False)
            correct = execution.get('correct_match', True)  # Whether the match was correct
            execution_time = execution.get('execution_time')
            confidence = execution.get('confidence')
            
            stats = rule_performance[rule_id]
            stats['executions'] += 1
            
            if matched:
                stats['matches'] += 1
                if correct:
                    stats['true_positives'] += 1
                else:
                    stats['false_positives'] += 1
            else:
                if not correct:  # Should have matched but didn't
                    stats['false_negatives'] += 1
            
            if execution_time is not None:
                stats['execution_times'].append(execution_time)
            
            if confidence is not None:
                stats['confidence_scores'].append(confidence)
        
        # Calculate metrics and identify optimization opportunities
        rule_metrics = {}
        optimization_suggestions = []
        
        for rule_id, stats in rule_performance.items():
            if stats['executions'] == 0:
                continue
            
            # Calculate performance metrics
            precision = 0
            recall = 0
            f1_score = 0
            
            if stats['true_positives'] + stats['false_positives'] > 0:
                precision = stats['true_positives'] / (stats['true_positives'] + stats['false_positives'])
            
            if stats['true_positives'] + stats['false_negatives'] > 0:
                recall = stats['true_positives'] / (stats['true_positives'] + stats['false_negatives'])
            
            if precision + recall > 0:
                f1_score = 2 * (precision * recall) / (precision + recall)
            
            match_rate = stats['matches'] / stats['executions']
            avg_execution_time = statistics.mean(stats['execution_times']) if stats['execution_times'] else 0
            avg_confidence = statistics.mean(stats['confidence_scores']) if stats['confidence_scores'] else 0
            
            rule_metrics[rule_id] = {
                'executions': stats['executions'],
                'match_rate': match_rate,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'avg_execution_time': avg_execution_time,
                'avg_confidence': avg_confidence,
                'true_positives': stats['true_positives'],
                'false_positives': stats['false_positives'],
                'false_negatives': stats['false_negatives']
            }
            
            # Generate optimization suggestions
            
            # Low precision (high false positives)
            if precision < 0.7 and stats['false_positives'] > 2:
                optimization_suggestions.append({
                    'rule_id': rule_id,
                    'type': 'precision_improvement',
                    'message': f"Rule {rule_id} has low precision ({precision*100:.1f}%) with {stats['false_positives']} false positives",
                    'recommendation': "Consider adding more specific conditions to reduce false positives"
                })
            
            # Low recall (high false negatives)
            if recall < 0.7 and stats['false_negatives'] > 2:
                optimization_suggestions.append({
                    'rule_id': rule_id,
                    'type': 'recall_improvement',
                    'message': f"Rule {rule_id} has low recall ({recall*100:.1f}%) with {stats['false_negatives']} false negatives",
                    'recommendation': "Consider broadening rule conditions to catch more cases"
                })
            
            # Slow execution
            if avg_execution_time > 1.0:  # More than 1 second
                optimization_suggestions.append({
                    'rule_id': rule_id,
                    'type': 'performance_optimization',
                    'message': f"Rule {rule_id} has slow average execution time: {avg_execution_time:.2f}s",
                    'recommendation': "Consider optimizing rule logic or adding early exit conditions"
                })
            
            # Low confidence
            if avg_confidence < 0.6 and stats['matches'] > 5:
                optimization_suggestions.append({
                    'rule_id': rule_id,
                    'type': 'confidence_improvement',
                    'message': f"Rule {rule_id} has low average confidence: {avg_confidence*100:.1f}%",
                    'recommendation': "Consider improving rule logic or adding confidence boosting factors"
                })
            
            # Underutilized rules
            if match_rate < 0.05 and stats['executions'] > 50:  # Less than 5% match rate with many executions
                optimization_suggestions.append({
                    'rule_id': rule_id,
                    'type': 'utilization_review',
                    'message': f"Rule {rule_id} has very low match rate: {match_rate*100:.1f}%",
                    'recommendation': "Consider reviewing if this rule is still relevant or needs adjustment"
                })
        
        return {
            'rule_metrics': rule_metrics,
            'optimization_suggestions': optimization_suggestions,
            'summary': {
                'total_rules_analyzed': len(rule_metrics),
                'total_executions': sum(stats['executions'] for stats in rule_performance.values()),
                'analysis_timestamp': datetime.utcnow().isoformat()
            }
        }
    
    def generate_rule_improvements(self, rule_id: str, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate specific improvements for a rule based on performance data.
        
        Args:
            rule_id: ID of the rule to improve
            performance_data: Performance metrics for the rule
            
        Returns:
            Specific improvement recommendations
        """
        improvements = {
            'rule_id': rule_id,
            'current_performance': performance_data,
            'suggested_changes': [],
            'priority': 'low'
        }
        
        precision = performance_data.get('precision', 1.0)
        recall = performance_data.get('recall', 1.0)
        f1_score = performance_data.get('f1_score', 1.0)
        false_positives = performance_data.get('false_positives', 0)
        false_negatives = performance_data.get('false_negatives', 0)
        
        # High priority if F1 score is very low
        if f1_score < 0.5:
            improvements['priority'] = 'high'
        elif f1_score < 0.7:
            improvements['priority'] = 'medium'
        
        # Specific improvement suggestions
        if precision < 0.8 and false_positives > 3:
            improvements['suggested_changes'].append({
                'type': 'add_specificity',
                'description': 'Add more specific conditions to reduce false positives',
                'examples': [
                    'Add file type restrictions',
                    'Include context validation',
                    'Add confidence thresholds'
                ]
            })
        
        if recall < 0.8 and false_negatives > 3:
            improvements['suggested_changes'].append({
                'type': 'broaden_coverage',
                'description': 'Expand rule conditions to catch more cases',
                'examples': [
                    'Add alternative patterns',
                    'Include similar error variations',
                    'Reduce overly restrictive conditions'
                ]
            })
        
        # Performance improvements
        avg_execution_time = performance_data.get('avg_execution_time', 0)
        if avg_execution_time > 0.5:
            improvements['suggested_changes'].append({
                'type': 'optimize_performance',
                'description': 'Improve rule execution speed',
                'examples': [
                    'Add early exit conditions',
                    'Optimize regex patterns',
                    'Cache expensive computations'
                ]
            })
        
        return improvements


class TelemetryRuleEnhancementSystem:
    """
    Main telemetry and rule enhancement system that coordinates all analytics
    and provides intelligent insights for system improvement.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the telemetry and rule enhancement system.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.monitoring_logger = MonitoringLogger("telemetry_rule_enhancement")
        self.audit_logger = get_audit_logger()
        
        # Initialize component analyzers
        self.error_analyzer = ErrorPatternAnalyzer(self.config.get('error_analysis', {}))
        self.llm_analyzer = LLMPerformanceAnalyzer(self.config.get('llm_analysis', {}))
        self.rule_optimizer = RuleOptimizer(self.config.get('rule_optimization', {}))
        
        # Initialize data collectors
        self.metrics_collector = MetricsCollector()
        self.feedback_loop = FeedbackLoop(self.metrics_collector)
        
        # Data storage
        self.telemetry_storage = Path(self.config.get('telemetry_storage', 'logs/telemetry'))
        self.telemetry_storage.mkdir(parents=True, exist_ok=True)
        
        # Analytics cache
        self.analytics_cache = {}
        self.cache_ttl = self.config.get('cache_ttl_minutes', 60)
        
        # Telemetry data buffer
        self.telemetry_buffer = []
        self.buffer_size = self.config.get('buffer_size', 1000)
        
    def collect_telemetry(self, event_type: str, **data) -> str:
        """
        Collect a telemetry data point.
        
        Args:
            event_type: Type of telemetry event
            **data: Additional telemetry data
            
        Returns:
            Telemetry point ID
        """
        telemetry_point = TelemetryDataPoint(event_type, **data)
        
        # Add to buffer
        self.telemetry_buffer.append(telemetry_point)
        
        # Flush buffer if it's full
        if len(self.telemetry_buffer) >= self.buffer_size:
            self._flush_telemetry_buffer()
        
        # Log the telemetry collection
        log_event(
            event_type='telemetry_collected',
            details={
                'telemetry_type': event_type,
                'telemetry_id': telemetry_point.id,
                **data
            }
        )
        
        return telemetry_point.id
    
    def _flush_telemetry_buffer(self) -> None:
        """Flush the telemetry buffer to storage."""
        if not self.telemetry_buffer:
            return
        
        try:
            # Create filename with timestamp
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            filename = f'telemetry_{timestamp}_{uuid.uuid4().hex[:8]}.json'
            file_path = self.telemetry_storage / filename
            
            # Convert to serializable format
            telemetry_data = [point.to_dict() for point in self.telemetry_buffer]
            
            # Save to file
            with open(file_path, 'w') as f:
                json.dump(telemetry_data, f, indent=2)
            
            self.monitoring_logger.info(f"Flushed {len(self.telemetry_buffer)} telemetry points to {file_path}")
            
            # Clear buffer
            self.telemetry_buffer.clear()
            
        except Exception as e:
            self.monitoring_logger.error(f"Failed to flush telemetry buffer: {e}")
    
    def load_telemetry_data(self, event_types: List[str] = None,
                           start_time: datetime = None,
                           end_time: datetime = None) -> List[Dict[str, Any]]:
        """
        Load telemetry data from storage.
        
        Args:
            event_types: Filter by event types
            start_time: Filter by start time
            end_time: Filter by end time
            
        Returns:
            List of telemetry data points
        """
        telemetry_data = []
        
        # Load from buffer first
        for point in self.telemetry_buffer:
            if event_types and point.event_type not in event_types:
                continue
            if start_time and point.timestamp < start_time:
                continue
            if end_time and point.timestamp > end_time:
                continue
            
            telemetry_data.append(point.to_dict())
        
        # Load from storage files
        try:
            for telemetry_file in self.telemetry_storage.glob('telemetry_*.json'):
                with open(telemetry_file, 'r') as f:
                    file_data = json.load(f)
                
                for point_data in file_data:
                    timestamp = datetime.fromisoformat(point_data['timestamp'])
                    
                    if event_types and point_data['event_type'] not in event_types:
                        continue
                    if start_time and timestamp < start_time:
                        continue
                    if end_time and timestamp > end_time:
                        continue
                    
                    telemetry_data.append(point_data)
                    
        except Exception as e:
            self.monitoring_logger.error(f"Failed to load telemetry data: {e}")
        
        return telemetry_data
    
    def analyze_error_patterns(self, days_back: int = 7) -> Dict[str, Any]:
        """
        Analyze error patterns from recent telemetry data.
        
        Args:
            days_back: Number of days to analyze
            
        Returns:
            Error pattern analysis results
        """
        cache_key = f"error_patterns_{days_back}"
        
        # Check cache
        if self._is_cache_valid(cache_key):
            return self.analytics_cache[cache_key]['data']
        
        # Load error data
        start_time = datetime.utcnow() - timedelta(days=days_back)
        error_data = self.load_telemetry_data(
            event_types=['error_detected', 'error_analyzed'],
            start_time=start_time
        )
        
        # Analyze patterns
        analysis = self.error_analyzer.analyze_error_patterns(error_data)
        
        # Cache results
        self._cache_result(cache_key, analysis)
        
        self.monitoring_logger.info(f"Analyzed {len(error_data)} error events from last {days_back} days")
        
        return analysis
    
    def analyze_llm_performance(self, days_back: int = 7) -> Dict[str, Any]:
        """
        Analyze LLM performance from recent telemetry data.
        
        Args:
            days_back: Number of days to analyze
            
        Returns:
            LLM performance analysis results
        """
        cache_key = f"llm_performance_{days_back}"
        
        # Check cache
        if self._is_cache_valid(cache_key):
            return self.analytics_cache[cache_key]['data']
        
        # Load LLM interaction data
        start_time = datetime.utcnow() - timedelta(days=days_back)
        llm_data = self.load_telemetry_data(
            event_types=['llm_interaction', 'llm_request', 'llm_response'],
            start_time=start_time
        )
        
        # Analyze performance
        analysis = self.llm_analyzer.analyze_llm_performance(llm_data)
        
        # Cache results
        self._cache_result(cache_key, analysis)
        
        self.monitoring_logger.info(f"Analyzed {len(llm_data)} LLM interactions from last {days_back} days")
        
        return analysis
    
    def analyze_rule_performance(self, days_back: int = 7) -> Dict[str, Any]:
        """
        Analyze rule performance from recent telemetry data.
        
        Args:
            days_back: Number of days to analyze
            
        Returns:
            Rule performance analysis results
        """
        cache_key = f"rule_performance_{days_back}"
        
        # Check cache
        if self._is_cache_valid(cache_key):
            return self.analytics_cache[cache_key]['data']
        
        # Load rule execution data
        start_time = datetime.utcnow() - timedelta(days=days_back)
        rule_data = self.load_telemetry_data(
            event_types=['rule_executed', 'rule_matched', 'rule_performance'],
            start_time=start_time
        )
        
        # Analyze performance
        analysis = self.rule_optimizer.analyze_rule_performance(rule_data)
        
        # Cache results
        self._cache_result(cache_key, analysis)
        
        self.monitoring_logger.info(f"Analyzed {len(rule_data)} rule executions from last {days_back} days")
        
        return analysis
    
    def generate_comprehensive_insights(self, days_back: int = 7) -> Dict[str, Any]:
        """
        Generate comprehensive insights from all analytics.
        
        Args:
            days_back: Number of days to analyze
            
        Returns:
            Comprehensive insights and recommendations
        """
        insights = {
            'analysis_period': f"{days_back} days",
            'generated_at': datetime.utcnow().isoformat(),
            'error_patterns': {},
            'llm_performance': {},
            'rule_performance': {},
            'cross_cutting_insights': [],
            'priority_recommendations': []
        }
        
        # Gather all analyses
        try:
            insights['error_patterns'] = self.analyze_error_patterns(days_back)
        except Exception as e:
            self.monitoring_logger.error(f"Failed to analyze error patterns: {e}")
            insights['error_patterns'] = {'error': str(e)}
        
        try:
            insights['llm_performance'] = self.analyze_llm_performance(days_back)
        except Exception as e:
            self.monitoring_logger.error(f"Failed to analyze LLM performance: {e}")
            insights['llm_performance'] = {'error': str(e)}
        
        try:
            insights['rule_performance'] = self.analyze_rule_performance(days_back)
        except Exception as e:
            self.monitoring_logger.error(f"Failed to analyze rule performance: {e}")
            insights['rule_performance'] = {'error': str(e)}
        
        # Generate cross-cutting insights
        insights['cross_cutting_insights'] = self._generate_cross_cutting_insights(insights)
        
        # Prioritize recommendations
        insights['priority_recommendations'] = self._prioritize_recommendations(insights)
        
        # Store comprehensive insights
        self._store_insights(insights)
        
        return insights
    
    def _generate_cross_cutting_insights(self, analysis_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate insights that span multiple analysis areas."""
        cross_cutting = []
        
        error_patterns = analysis_data.get('error_patterns', {})
        llm_performance = analysis_data.get('llm_performance', {})
        rule_performance = analysis_data.get('rule_performance', {})
        
        # Correlate error types with LLM success rates
        error_insights = error_patterns.get('insights', [])
        llm_insights = llm_performance.get('insights', [])
        rule_insights = rule_performance.get('optimization_suggestions', [])
        
        # Check for correlation between rule performance and error detection
        low_performing_rules = [
            insight for insight in rule_insights 
            if insight.get('type') in ['precision_improvement', 'recall_improvement']
        ]
        
        dominant_errors = [
            insight for insight in error_insights 
            if insight.get('type') == 'dominant_error_type'
        ]
        
        if low_performing_rules and dominant_errors:
            cross_cutting.append({
                'type': 'rule_error_correlation',
                'message': f"Found {len(low_performing_rules)} underperforming rules and {len(dominant_errors)} dominant error types",
                'recommendation': "Consider creating specialized rules for dominant error types to improve detection accuracy"
            })
        
        # Check for LLM performance vs fix success correlation
        llm_metrics = llm_performance.get('performance_metrics', {})
        
        low_confidence_providers = [
            provider for provider, metrics in llm_metrics.items() 
            if metrics.get('avg_confidence', 1.0) < 0.7
        ]
        
        if low_confidence_providers:
            cross_cutting.append({
                'type': 'llm_confidence_improvement',
                'message': f"Providers {', '.join(low_confidence_providers)} show low confidence scores",
                'recommendation': "Consider prompt engineering improvements or model fine-tuning to increase fix quality"
            })
        
        return cross_cutting
    
    def _prioritize_recommendations(self, analysis_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Prioritize recommendations based on impact and urgency."""
        all_recommendations = []
        
        # Collect all recommendations
        for area, data in analysis_data.items():
            if isinstance(data, dict):
                insights = data.get('insights', [])
                if area == 'rule_performance':
                    insights = data.get('optimization_suggestions', [])
                
                for insight in insights:
                    insight['source_area'] = area
                    all_recommendations.append(insight)
        
        # Add cross-cutting insights
        for insight in analysis_data.get('cross_cutting_insights', []):
            insight['source_area'] = 'cross_cutting'
            all_recommendations.append(insight)
        
        # Assign priority scores
        for rec in all_recommendations:
            rec['priority_score'] = self._calculate_priority_score(rec)
        
        # Sort by priority score (highest first)
        all_recommendations.sort(key=lambda x: x.get('priority_score', 0), reverse=True)
        
        # Return top 10 recommendations
        return all_recommendations[:10]
    
    def _calculate_priority_score(self, recommendation: Dict[str, Any]) -> float:
        """Calculate priority score for a recommendation."""
        score = 0.0
        
        # Base score by type
        type_scores = {
            'dominant_error_type': 8.0,
            'high_severity_concentration': 9.0,
            'low_provider_success_rate': 7.0,
            'precision_improvement': 6.0,
            'recall_improvement': 6.0,
            'rule_error_correlation': 8.5,
            'llm_confidence_improvement': 7.5,
            'high_cost_usage': 5.0,
            'performance_optimization': 4.0
        }
        
        rec_type = recommendation.get('type', 'unknown')
        score += type_scores.get(rec_type, 3.0)
        
        # Boost score for cross-cutting insights
        if recommendation.get('source_area') == 'cross_cutting':
            score += 2.0
        
        # Boost score for high-impact areas
        if any(keyword in recommendation.get('message', '').lower() 
               for keyword in ['critical', 'high', 'severe', 'urgent']):
            score += 1.5
        
        return score
    
    def _store_insights(self, insights: Dict[str, Any]) -> None:
        """Store comprehensive insights for historical analysis."""
        try:
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            filename = f'insights_{timestamp}.json'
            file_path = self.telemetry_storage / filename
            
            with open(file_path, 'w') as f:
                json.dump(insights, f, indent=2)
            
            self.monitoring_logger.info(f"Stored comprehensive insights to {file_path}")
            
        except Exception as e:
            self.monitoring_logger.error(f"Failed to store insights: {e}")
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached result is still valid."""
        if cache_key not in self.analytics_cache:
            return False
        
        cache_time = self.analytics_cache[cache_key]['timestamp']
        age_minutes = (datetime.utcnow() - cache_time).total_seconds() / 60
        
        return age_minutes < self.cache_ttl
    
    def _cache_result(self, cache_key: str, data: Any) -> None:
        """Cache analysis result."""
        self.analytics_cache[cache_key] = {
            'data': data,
            'timestamp': datetime.utcnow()
        }
    
    def cleanup_old_data(self, days_to_keep: int = 30) -> int:
        """
        Clean up old telemetry data.
        
        Args:
            days_to_keep: Number of days of data to keep
            
        Returns:
            Number of files cleaned up
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
        cleaned_count = 0
        
        try:
            for telemetry_file in self.telemetry_storage.glob('telemetry_*.json'):
                # Extract timestamp from filename
                try:
                    filename_parts = telemetry_file.stem.split('_')
                    if len(filename_parts) >= 2:
                        timestamp_str = filename_parts[1]
                        file_date = datetime.strptime(timestamp_str, '%Y%m%d')
                        
                        if file_date < cutoff_date:
                            telemetry_file.unlink()
                            cleaned_count += 1
                            
                except (ValueError, IndexError):
                    # Skip files with unexpected naming
                    continue
            
            # Clean up insights files too
            for insights_file in self.telemetry_storage.glob('insights_*.json'):
                try:
                    filename_parts = insights_file.stem.split('_')
                    if len(filename_parts) >= 2:
                        timestamp_str = filename_parts[1]
                        file_date = datetime.strptime(timestamp_str, '%Y%m%d')
                        
                        if file_date < cutoff_date:
                            insights_file.unlink()
                            cleaned_count += 1
                            
                except (ValueError, IndexError):
                    continue
            
            if cleaned_count > 0:
                self.monitoring_logger.info(f"Cleaned up {cleaned_count} old telemetry files")
            
        except Exception as e:
            self.monitoring_logger.error(f"Failed to cleanup old data: {e}")
        
        return cleaned_count


# Singleton instance
_telemetry_system = None

def get_telemetry_system(config: Dict[str, Any] = None) -> TelemetryRuleEnhancementSystem:
    """
    Get or create the singleton TelemetryRuleEnhancementSystem instance.
    
    Args:
        config: Optional configuration
        
    Returns:
        TelemetryRuleEnhancementSystem: The telemetry system instance
    """
    global _telemetry_system
    if _telemetry_system is None:
        _telemetry_system = TelemetryRuleEnhancementSystem(config)
    return _telemetry_system


# Convenience functions
def collect_telemetry(event_type: str, **data) -> str:
    """Collect a telemetry data point."""
    return get_telemetry_system().collect_telemetry(event_type, **data)


def analyze_error_patterns(days_back: int = 7) -> Dict[str, Any]:
    """Analyze error patterns from recent data."""
    return get_telemetry_system().analyze_error_patterns(days_back)


def analyze_llm_performance(days_back: int = 7) -> Dict[str, Any]:
    """Analyze LLM performance from recent data."""
    return get_telemetry_system().analyze_llm_performance(days_back)


def generate_comprehensive_insights(days_back: int = 7) -> Dict[str, Any]:
    """Generate comprehensive insights from all analytics."""
    return get_telemetry_system().generate_comprehensive_insights(days_back)