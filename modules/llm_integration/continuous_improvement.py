#!/usr/bin/env python3
"""
Continuous improvement system for LLM integration.

Provides mechanisms for collecting feedback, analyzing patch success/failure patterns,
and improving the system through data-driven insights.
"""

import hashlib
import json
import logging
import sqlite3
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


class PatchOutcome(Enum):
    """Possible outcomes for patch application."""

    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL_SUCCESS = "partial_success"
    REJECTED = "rejected"
    UNKNOWN = "unknown"


class FeedbackType(Enum):
    """Types of feedback that can be collected."""

    AUTOMATIC = "automatic"  # From test results, monitoring
    HUMAN = "human"  # From user reviews, ratings
    SYSTEM = "system"  # From system metrics, performance


@dataclass
class PatchFeedback:
    """Feedback data for a patch application."""

    patch_id: str
    feedback_type: FeedbackType
    outcome: PatchOutcome
    timestamp: float
    context: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    feedback_source: str = "unknown"
    confidence_score: float = 0.0
    error_patterns: List[str] = field(default_factory=list)
    success_indicators: List[str] = field(default_factory=list)
    improvement_suggestions: List[str] = field(default_factory=list)


@dataclass
class LearningPattern:
    """A learned pattern from patch feedback analysis."""

    pattern_id: str
    pattern_type: str  # error_type, code_pattern, success_factor
    pattern_data: Dict[str, Any]
    confidence: float
    sample_count: int
    success_rate: float
    last_updated: float
    tags: Set[str] = field(default_factory=set)


@dataclass
class ImprovementRecommendation:
    """Recommendation for system improvement."""

    recommendation_id: str
    category: str  # prompt_template, provider_selection, error_detection
    description: str
    priority: str  # high, medium, low
    evidence: List[str]
    implementation_effort: str  # low, medium, high
    expected_impact: str  # low, medium, high
    data_support: Dict[str, Any]
    created_at: float


class ContinuousImprovementEngine:
    """Engine for continuous improvement of the LLM integration system."""

    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize the continuous improvement engine.

        Args:
            data_dir: Directory for storing improvement data
        """
        self.logger = logging.getLogger(__name__)

        if data_dir is None:
            data_dir = Path.home() / ".homeostasis" / "improvement_data"

        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self.db_path = self.data_dir / "improvement.db"
        self._init_database()

        # Learning patterns cache
        self._patterns_cache: Dict[str, LearningPattern] = {}
        self._cache_last_updated = 0.0
        self._cache_ttl = 300.0  # 5 minutes

        # Load existing patterns
        self._load_patterns()

    def _init_database(self) -> None:
        """Initialize the SQLite database for storing feedback and patterns."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Patch feedback table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS patch_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patch_id TEXT NOT NULL,
                    feedback_type TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    context TEXT,
                    metrics TEXT,
                    feedback_source TEXT,
                    confidence_score REAL,
                    error_patterns TEXT,
                    success_indicators TEXT,
                    improvement_suggestions TEXT
                )
            """
            )

            # Learning patterns table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS learning_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_id TEXT UNIQUE NOT NULL,
                    pattern_type TEXT NOT NULL,
                    pattern_data TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    sample_count INTEGER NOT NULL,
                    success_rate REAL NOT NULL,
                    last_updated REAL NOT NULL,
                    tags TEXT
                )
            """
            )

            # Improvement recommendations table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS improvement_recommendations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    recommendation_id TEXT UNIQUE NOT NULL,
                    category TEXT NOT NULL,
                    description TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    evidence TEXT,
                    implementation_effort TEXT,
                    expected_impact TEXT,
                    data_support TEXT,
                    created_at REAL NOT NULL,
                    status TEXT DEFAULT 'pending'
                )
            """
            )

            # Create indexes
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_patch_feedback_patch_id ON patch_feedback(patch_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_patch_feedback_timestamp ON patch_feedback(timestamp)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_patterns_type ON learning_patterns(pattern_type)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_recommendations_category ON improvement_recommendations(category)"
            )

            conn.commit()

    def record_patch_feedback(self, feedback: PatchFeedback) -> None:
        """
        Record feedback for a patch application.

        Args:
            feedback: Patch feedback data
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO patch_feedback (
                        patch_id, feedback_type, outcome, timestamp, context,
                        metrics, feedback_source, confidence_score, error_patterns,
                        success_indicators, improvement_suggestions
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        feedback.patch_id,
                        feedback.feedback_type.value,
                        feedback.outcome.value,
                        feedback.timestamp,
                        json.dumps(feedback.context),
                        json.dumps(feedback.metrics),
                        feedback.feedback_source,
                        feedback.confidence_score,
                        json.dumps(feedback.error_patterns),
                        json.dumps(feedback.success_indicators),
                        json.dumps(feedback.improvement_suggestions),
                    ),
                )
                conn.commit()

            self.logger.info(
                f"Recorded feedback for patch {feedback.patch_id}: {feedback.outcome.value}"
            )

            # Trigger pattern analysis if we have enough new data
            self._maybe_trigger_pattern_analysis()

        except Exception as e:
            self.logger.error(f"Failed to record patch feedback: {e}")

    def analyze_patterns(self, min_samples: int = 10) -> List[LearningPattern]:
        """
        Analyze feedback data to identify patterns and trends.

        Args:
            min_samples: Minimum number of samples required for pattern identification

        Returns:
            List of identified learning patterns
        """
        patterns = []

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Analyze error patterns
                patterns.extend(self._analyze_error_patterns(cursor, min_samples))

                # Analyze success patterns
                patterns.extend(self._analyze_success_patterns(cursor, min_samples))

                # Analyze provider performance patterns
                patterns.extend(self._analyze_provider_patterns(cursor, min_samples))

                # Analyze temporal patterns
                patterns.extend(self._analyze_temporal_patterns(cursor, min_samples))

                # Store patterns in database
                self._store_patterns(patterns)

                # Update cache
                self._update_patterns_cache(patterns)

            self.logger.info(f"Analyzed patterns, found {len(patterns)} new patterns")
            return patterns

        except Exception as e:
            self.logger.error(f"Failed to analyze patterns: {e}")
            return []

    def _analyze_error_patterns(
        self, cursor, min_samples: int
    ) -> List[LearningPattern]:
        """Analyze error patterns from feedback data."""
        patterns = []

        # Get error patterns grouped by context
        cursor.execute(
            """
            SELECT error_patterns, outcome, COUNT(*) as count
            FROM patch_feedback 
            WHERE error_patterns != '[]' AND error_patterns IS NOT NULL
            GROUP BY error_patterns, outcome
            HAVING count >= ?
        """,
            (min_samples,),
        )

        error_data = cursor.fetchall()

        for error_patterns_json, outcome, count in error_data:
            try:
                error_patterns = json.loads(error_patterns_json)
                if not error_patterns:
                    continue

                # Calculate success rate for this error pattern
                cursor.execute(
                    """
                    SELECT outcome, COUNT(*) 
                    FROM patch_feedback 
                    WHERE error_patterns = ?
                    GROUP BY outcome
                """,
                    (error_patterns_json,),
                )

                outcome_counts = dict(cursor.fetchall())
                total_samples = sum(outcome_counts.values())
                success_count = outcome_counts.get(PatchOutcome.SUCCESS.value, 0)
                success_rate = (
                    success_count / total_samples if total_samples > 0 else 0.0
                )

                # Create pattern
                pattern_id = self._generate_pattern_id("error_pattern", error_patterns)
                pattern = LearningPattern(
                    pattern_id=pattern_id,
                    pattern_type="error_pattern",
                    pattern_data={
                        "error_patterns": error_patterns,
                        "outcome_distribution": outcome_counts,
                        "most_common_outcome": outcome,
                    },
                    confidence=min(1.0, total_samples / 100.0),
                    sample_count=total_samples,
                    success_rate=success_rate,
                    last_updated=time.time(),
                    tags={"error_analysis", "pattern_recognition"},
                )
                patterns.append(pattern)

            except json.JSONDecodeError:
                continue

        return patterns

    def _analyze_success_patterns(
        self, cursor, min_samples: int
    ) -> List[LearningPattern]:
        """Analyze success patterns from feedback data."""
        patterns = []

        # Get success indicators grouped by context
        cursor.execute(
            """
            SELECT success_indicators, context, COUNT(*) as count
            FROM patch_feedback 
            WHERE outcome = ? AND success_indicators != '[]' AND success_indicators IS NOT NULL
            GROUP BY success_indicators, context
            HAVING count >= ?
        """,
            (PatchOutcome.SUCCESS.value, min_samples),
        )

        success_data = cursor.fetchall()

        for success_indicators_json, context_json, count in success_data:
            try:
                success_indicators = json.loads(success_indicators_json)
                context = json.loads(context_json) if context_json else {}

                if not success_indicators:
                    continue

                # Create pattern
                pattern_id = self._generate_pattern_id(
                    "success_pattern", success_indicators
                )
                pattern = LearningPattern(
                    pattern_id=pattern_id,
                    pattern_type="success_pattern",
                    pattern_data={
                        "success_indicators": success_indicators,
                        "context": context,
                        "frequency": count,
                    },
                    confidence=min(1.0, count / 50.0),
                    sample_count=count,
                    success_rate=1.0,  # By definition, these are success patterns
                    last_updated=time.time(),
                    tags={"success_analysis", "best_practices"},
                )
                patterns.append(pattern)

            except json.JSONDecodeError:
                continue

        return patterns

    def _analyze_provider_patterns(
        self, cursor, min_samples: int
    ) -> List[LearningPattern]:
        """Analyze provider performance patterns."""
        patterns = []

        # Get provider performance data
        cursor.execute(
            """
            SELECT feedback_source, outcome, context, COUNT(*) as count
            FROM patch_feedback 
            WHERE feedback_source LIKE '%provider%'
            GROUP BY feedback_source, outcome, context
            HAVING count >= ?
        """,
            (min_samples,),
        )

        provider_data = cursor.fetchall()

        # Group by provider
        provider_stats: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        for feedback_source, outcome, context_json, count in provider_data:
            try:
                context = json.loads(context_json) if context_json else {}
                provider = context.get("provider", "unknown")
                provider_stats[provider][outcome] += count
            except json.JSONDecodeError:
                continue

        # Create patterns for each provider
        for provider, outcomes in provider_stats.items():
            total_samples = sum(outcomes.values())
            if total_samples < min_samples:
                continue

            success_count = outcomes.get(PatchOutcome.SUCCESS.value, 0)
            success_rate = success_count / total_samples

            pattern_id = self._generate_pattern_id("provider_performance", provider)
            pattern = LearningPattern(
                pattern_id=pattern_id,
                pattern_type="provider_performance",
                pattern_data={
                    "provider": provider,
                    "outcome_distribution": dict(outcomes),
                    "performance_metrics": {
                        "success_rate": success_rate,
                        "total_requests": total_samples,
                    },
                },
                confidence=min(1.0, total_samples / 200.0),
                sample_count=total_samples,
                success_rate=success_rate,
                last_updated=time.time(),
                tags={"provider_analysis", "performance"},
            )
            patterns.append(pattern)

        return patterns

    def _analyze_temporal_patterns(
        self, cursor, min_samples: int
    ) -> List[LearningPattern]:
        """Analyze temporal patterns in patch success/failure."""
        patterns: List[LearningPattern] = []

        # Get time-series data
        cursor.execute(
            """
            SELECT timestamp, outcome, metrics
            FROM patch_feedback 
            ORDER BY timestamp
        """
        )

        time_data = cursor.fetchall()

        if len(time_data) < min_samples:
            return patterns

        # Analyze hourly patterns
        hourly_success: Dict[int, int] = defaultdict(int)
        hourly_total: Dict[int, int] = defaultdict(int)

        for timestamp, outcome, metrics_json in time_data:
            hour = datetime.fromtimestamp(timestamp).hour
            hourly_total[hour] += 1
            if outcome == PatchOutcome.SUCCESS.value:
                hourly_success[hour] += 1

        # Find time periods with significantly different success rates
        overall_success_rate = sum(hourly_success.values()) / sum(hourly_total.values())

        for hour, total in hourly_total.items():
            if total >= min_samples:
                success_rate = hourly_success[hour] / total
                # If success rate is significantly different from overall
                if abs(success_rate - overall_success_rate) > 0.1:
                    pattern_id = self._generate_pattern_id(
                        "temporal_pattern", f"hour_{hour}"
                    )
                    pattern = LearningPattern(
                        pattern_id=pattern_id,
                        pattern_type="temporal_pattern",
                        pattern_data={
                            "time_period": f"hour_{hour}",
                            "success_rate": success_rate,
                            "overall_success_rate": overall_success_rate,
                            "deviation": success_rate - overall_success_rate,
                        },
                        confidence=min(1.0, total / 100.0),
                        sample_count=total,
                        success_rate=success_rate,
                        last_updated=time.time(),
                        tags={"temporal_analysis", "scheduling"},
                    )
                    patterns.append(pattern)

        return patterns

    def _generate_pattern_id(self, pattern_type: str, data: Any) -> str:
        """Generate a unique pattern ID."""
        data_str = json.dumps(data, sort_keys=True)
        hash_obj = hashlib.sha256(f"{pattern_type}:{data_str}".encode())
        return f"{pattern_type}_{hash_obj.hexdigest()[:16]}"

    def _store_patterns(self, patterns: List[LearningPattern]) -> None:
        """Store patterns in the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            for pattern in patterns:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO learning_patterns (
                        pattern_id, pattern_type, pattern_data, confidence,
                        sample_count, success_rate, last_updated, tags
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        pattern.pattern_id,
                        pattern.pattern_type,
                        json.dumps(pattern.pattern_data),
                        pattern.confidence,
                        pattern.sample_count,
                        pattern.success_rate,
                        pattern.last_updated,
                        json.dumps(list(pattern.tags)),
                    ),
                )

            conn.commit()

    def _load_patterns(self) -> None:
        """Load patterns from database into cache."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT pattern_id, pattern_type, pattern_data, confidence,
                           sample_count, success_rate, last_updated, tags
                    FROM learning_patterns
                """
                )

                for row in cursor.fetchall():
                    (
                        pattern_id,
                        pattern_type,
                        pattern_data,
                        confidence,
                        sample_count,
                        success_rate,
                        last_updated,
                        tags,
                    ) = row

                    pattern = LearningPattern(
                        pattern_id=pattern_id,
                        pattern_type=pattern_type,
                        pattern_data=json.loads(pattern_data),
                        confidence=confidence,
                        sample_count=sample_count,
                        success_rate=success_rate,
                        last_updated=last_updated,
                        tags=set(json.loads(tags)),
                    )

                    self._patterns_cache[pattern_id] = pattern

            self._cache_last_updated = time.time()

        except Exception as e:
            self.logger.error(f"Failed to load patterns: {e}")

    def _update_patterns_cache(self, patterns: List[LearningPattern]) -> None:
        """Update patterns cache with new patterns."""
        for pattern in patterns:
            self._patterns_cache[pattern.pattern_id] = pattern
        self._cache_last_updated = time.time()

    def _maybe_trigger_pattern_analysis(self) -> None:
        """Trigger pattern analysis if conditions are met."""
        # Check if enough time has passed since last analysis
        if time.time() - self._cache_last_updated < 3600:  # 1 hour
            return

        # Check if we have enough new feedback
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT COUNT(*) FROM patch_feedback 
                    WHERE timestamp > ?
                """,
                    (self._cache_last_updated,),
                )

                new_feedback_count = cursor.fetchone()[0]

                if new_feedback_count >= 50:  # Threshold for triggering analysis
                    self.logger.info(
                        f"Triggering pattern analysis with {new_feedback_count} new feedback entries"
                    )
                    self.analyze_patterns()

        except Exception as e:
            self.logger.error(f"Failed to check for pattern analysis trigger: {e}")

    def generate_improvement_recommendations(self) -> List[ImprovementRecommendation]:
        """Generate recommendations for system improvements based on learned patterns."""
        recommendations = []

        try:
            # Ensure patterns are up to date
            if time.time() - self._cache_last_updated > self._cache_ttl:
                self._load_patterns()

            # Analyze error patterns for improvement opportunities
            recommendations.extend(self._generate_error_pattern_recommendations())

            # Analyze provider performance for optimization
            recommendations.extend(self._generate_provider_recommendations())

            # Analyze success patterns for best practices
            recommendations.extend(self._generate_success_pattern_recommendations())

            # Store recommendations
            self._store_recommendations(recommendations)

            self.logger.info(
                f"Generated {len(recommendations)} improvement recommendations"
            )
            return recommendations

        except Exception as e:
            self.logger.error(f"Failed to generate improvement recommendations: {e}")
            return []

    def _generate_error_pattern_recommendations(
        self,
    ) -> List[ImprovementRecommendation]:
        """Generate recommendations based on error patterns."""
        recommendations = []

        error_patterns = [
            p
            for p in self._patterns_cache.values()
            if p.pattern_type == "error_pattern" and p.success_rate < 0.5
        ]

        for pattern in error_patterns:
            if pattern.sample_count < 20:  # Skip patterns with insufficient data
                continue

            recommendation_id = f"error_pattern_{pattern.pattern_id}"

            # Determine recommendation category based on error patterns
            error_types = pattern.pattern_data.get("error_patterns", [])

            if any("timeout" in error.lower() for error in error_types):
                category = "provider_selection"
                description = f"Consider adjusting timeout settings or provider selection for error patterns: {error_types}"
            elif any("memory" in error.lower() for error in error_types):
                category = "resource_optimization"
                description = (
                    f"Implement memory optimization for error patterns: {error_types}"
                )
            else:
                category = "error_detection"
                description = (
                    f"Improve error detection and handling for patterns: {error_types}"
                )

            recommendation = ImprovementRecommendation(
                recommendation_id=recommendation_id,
                category=category,
                description=description,
                priority="high" if pattern.success_rate < 0.2 else "medium",
                evidence=[
                    f"Pattern observed in {pattern.sample_count} cases",
                    f"Success rate: {pattern.success_rate:.2%}",
                    f"Confidence: {pattern.confidence:.2f}",
                ],
                implementation_effort="medium",
                expected_impact="high" if pattern.success_rate < 0.2 else "medium",
                data_support=asdict(pattern),
                created_at=time.time(),
            )

            recommendations.append(recommendation)

        return recommendations

    def _generate_provider_recommendations(self) -> List[ImprovementRecommendation]:
        """Generate recommendations based on provider performance patterns."""
        recommendations: List[ImprovementRecommendation] = []

        provider_patterns = [
            p
            for p in self._patterns_cache.values()
            if p.pattern_type == "provider_performance"
        ]

        if len(provider_patterns) < 2:  # Need multiple providers to compare
            return recommendations

        # Find best and worst performing providers
        provider_patterns.sort(key=lambda p: p.success_rate, reverse=True)
        best_provider = provider_patterns[0]
        worst_provider = provider_patterns[-1]

        if best_provider.success_rate - worst_provider.success_rate > 0.2:
            recommendation_id = f"provider_optimization_{int(time.time())}"

            recommendation = ImprovementRecommendation(
                recommendation_id=recommendation_id,
                category="provider_selection",
                description=f"Consider prioritizing {best_provider.pattern_data['provider']} over {worst_provider.pattern_data['provider']} based on performance data",
                priority="medium",
                evidence=[
                    f"{best_provider.pattern_data['provider']} success rate: {best_provider.success_rate:.2%}",
                    f"{worst_provider.pattern_data['provider']} success rate: {worst_provider.success_rate:.2%}",
                    f"Performance gap: {best_provider.success_rate - worst_provider.success_rate:.2%}",
                ],
                implementation_effort="low",
                expected_impact="medium",
                data_support={
                    "best_provider": asdict(best_provider),
                    "worst_provider": asdict(worst_provider),
                },
                created_at=time.time(),
            )

            recommendations.append(recommendation)

        return recommendations

    def _generate_success_pattern_recommendations(
        self,
    ) -> List[ImprovementRecommendation]:
        """Generate recommendations based on success patterns."""
        recommendations = []

        success_patterns = [
            p
            for p in self._patterns_cache.values()
            if p.pattern_type == "success_pattern" and p.confidence > 0.7
        ]

        for pattern in success_patterns:
            if pattern.sample_count < 10:
                continue

            recommendation_id = f"success_pattern_{pattern.pattern_id}"

            success_indicators = pattern.pattern_data.get("success_indicators", [])

            recommendation = ImprovementRecommendation(
                recommendation_id=recommendation_id,
                category="prompt_template",
                description=f"Incorporate successful patterns into prompt templates: {success_indicators}",
                priority="low",
                evidence=[
                    f"Pattern observed in {pattern.sample_count} successful cases",
                    f"Confidence: {pattern.confidence:.2f}",
                    "Consistent success indicators identified",
                ],
                implementation_effort="low",
                expected_impact="low",
                data_support=asdict(pattern),
                created_at=time.time(),
            )

            recommendations.append(recommendation)

        return recommendations

    def _store_recommendations(
        self, recommendations: List[ImprovementRecommendation]
    ) -> None:
        """Store recommendations in the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            for rec in recommendations:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO improvement_recommendations (
                        recommendation_id, category, description, priority,
                        evidence, implementation_effort, expected_impact,
                        data_support, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        rec.recommendation_id,
                        rec.category,
                        rec.description,
                        rec.priority,
                        json.dumps(rec.evidence),
                        rec.implementation_effort,
                        rec.expected_impact,
                        json.dumps(rec.data_support),
                        rec.created_at,
                    ),
                )

            conn.commit()

    def get_patterns(
        self, pattern_type: Optional[str] = None, min_confidence: float = 0.0
    ) -> List[LearningPattern]:
        """
        Get learning patterns, optionally filtered by type and confidence.

        Args:
            pattern_type: Type of patterns to retrieve
            min_confidence: Minimum confidence threshold

        Returns:
            List of matching patterns
        """
        # Ensure cache is up to date
        if time.time() - self._cache_last_updated > self._cache_ttl:
            self._load_patterns()

        patterns = list(self._patterns_cache.values())

        if pattern_type:
            patterns = [p for p in patterns if p.pattern_type == pattern_type]

        if min_confidence > 0:
            patterns = [p for p in patterns if p.confidence >= min_confidence]

        return sorted(patterns, key=lambda p: p.confidence, reverse=True)

    def get_recommendations(
        self, category: Optional[str] = None, priority: Optional[str] = None
    ) -> List[ImprovementRecommendation]:
        """
        Get improvement recommendations, optionally filtered.

        Args:
            category: Category to filter by
            priority: Priority to filter by

        Returns:
            List of matching recommendations
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                query = """
                    SELECT recommendation_id, category, description, priority,
                           evidence, implementation_effort, expected_impact,
                           data_support, created_at
                    FROM improvement_recommendations
                    WHERE status = 'pending'
                """
                params = []

                if category:
                    query += " AND category = ?"
                    params.append(category)

                if priority:
                    query += " AND priority = ?"
                    params.append(priority)

                query += " ORDER BY created_at DESC"

                cursor.execute(query, params)

                recommendations = []
                for row in cursor.fetchall():
                    (
                        rec_id,
                        cat,
                        desc,
                        prio,
                        evidence,
                        impl_effort,
                        exp_impact,
                        data_support,
                        created_at,
                    ) = row

                    recommendation = ImprovementRecommendation(
                        recommendation_id=rec_id,
                        category=cat,
                        description=desc,
                        priority=prio,
                        evidence=json.loads(evidence),
                        implementation_effort=impl_effort,
                        expected_impact=exp_impact,
                        data_support=json.loads(data_support),
                        created_at=created_at,
                    )

                    recommendations.append(recommendation)

                return recommendations

        except Exception as e:
            self.logger.error(f"Failed to get recommendations: {e}")
            return []

    def get_statistics(self) -> Dict[str, Any]:
        """Get improvement engine statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Feedback statistics
                cursor.execute("SELECT COUNT(*) FROM patch_feedback")
                total_feedback = cursor.fetchone()[0]

                cursor.execute(
                    """
                    SELECT outcome, COUNT(*) 
                    FROM patch_feedback 
                    GROUP BY outcome
                """
                )
                outcome_distribution = dict(cursor.fetchall())

                # Pattern statistics
                cursor.execute("SELECT COUNT(*) FROM learning_patterns")
                total_patterns = cursor.fetchone()[0]

                cursor.execute(
                    """
                    SELECT pattern_type, COUNT(*) 
                    FROM learning_patterns 
                    GROUP BY pattern_type
                """
                )
                pattern_distribution = dict(cursor.fetchall())

                # Recommendation statistics
                cursor.execute(
                    "SELECT COUNT(*) FROM improvement_recommendations WHERE status = 'pending'"
                )
                pending_recommendations = cursor.fetchone()[0]

                cursor.execute(
                    """
                    SELECT priority, COUNT(*) 
                    FROM improvement_recommendations 
                    WHERE status = 'pending'
                    GROUP BY priority
                """
                )
                recommendation_priorities = dict(cursor.fetchall())

                return {
                    "feedback": {
                        "total_count": total_feedback,
                        "outcome_distribution": outcome_distribution,
                    },
                    "patterns": {
                        "total_count": total_patterns,
                        "type_distribution": pattern_distribution,
                    },
                    "recommendations": {
                        "pending_count": pending_recommendations,
                        "priority_distribution": recommendation_priorities,
                    },
                    "cache_status": {
                        "last_updated": self._cache_last_updated,
                        "cached_patterns": len(self._patterns_cache),
                    },
                }

        except Exception as e:
            self.logger.error(f"Failed to get statistics: {e}")
            return {}


# Global instance
_improvement_engine = None


def get_improvement_engine() -> ContinuousImprovementEngine:
    """Get the global continuous improvement engine instance."""
    global _improvement_engine
    if _improvement_engine is None:
        _improvement_engine = ContinuousImprovementEngine()
    return _improvement_engine
