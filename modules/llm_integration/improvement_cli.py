#!/usr/bin/env python3
"""
CLI interface for continuous improvement system.

Provides commands for analyzing patterns, generating recommendations, and monitoring system improvements.
"""

import json
import click
from datetime import datetime
from typing import Optional
from .continuous_improvement import (
    get_improvement_engine, PatchFeedback, PatchOutcome, FeedbackType
)


@click.group(name="improve")
def improvement_cli():
    """Continuous improvement and learning commands."""
    pass


@improvement_cli.command()
@click.option("--min-samples", type=int, default=10, help="Minimum samples for pattern identification")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def analyze():
    """Analyze feedback data to identify patterns and trends."""
    engine = get_improvement_engine()
    
    click.echo("🔍 Analyzing feedback patterns...")
    patterns = engine.analyze_patterns(min_samples=10)
    
    if not patterns:
        click.echo("No significant patterns found.")
        return
    
    click.echo(f"✅ Found {len(patterns)} learning patterns:")
    click.echo()
    
    # Group patterns by type
    pattern_groups = {}
    for pattern in patterns:
        if pattern.pattern_type not in pattern_groups:
            pattern_groups[pattern.pattern_type] = []
        pattern_groups[pattern.pattern_type].append(pattern)
    
    for pattern_type, type_patterns in pattern_groups.items():
        click.echo(f"📊 {pattern_type.upper()} PATTERNS ({len(type_patterns)}):")
        
        for pattern in sorted(type_patterns, key=lambda p: p.confidence, reverse=True):
            confidence_indicator = "🔴" if pattern.confidence < 0.3 else "🟡" if pattern.confidence < 0.7 else "🟢"
            click.echo(f"  {confidence_indicator} {pattern.pattern_id}")
            click.echo(f"     Confidence: {pattern.confidence:.2f}, Samples: {pattern.sample_count}")
            click.echo(f"     Success Rate: {pattern.success_rate:.2%}")
            
            if pattern.pattern_type == "error_pattern":
                errors = pattern.pattern_data.get("error_patterns", [])
                click.echo(f"     Errors: {', '.join(errors[:3])}")
            elif pattern.pattern_type == "success_pattern":
                indicators = pattern.pattern_data.get("success_indicators", [])
                click.echo(f"     Success Factors: {', '.join(indicators[:3])}")
            elif pattern.pattern_type == "provider_performance":
                provider = pattern.pattern_data.get("provider", "unknown")
                click.echo(f"     Provider: {provider}")
            
            click.echo()


@improvement_cli.command()
@click.option("--category", help="Filter by category")
@click.option("--priority", help="Filter by priority")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
def recommend(category: Optional[str], priority: Optional[str], output_json: bool):
    """Generate improvement recommendations based on learned patterns."""
    engine = get_improvement_engine()
    
    click.echo("💡 Generating improvement recommendations...")
    recommendations = engine.generate_improvement_recommendations()
    
    # Filter recommendations
    if category or priority:
        filtered_recs = engine.get_recommendations(category=category, priority=priority)
        recommendations = filtered_recs
    
    if not recommendations:
        click.echo("No recommendations found matching criteria.")
        return
    
    if output_json:
        rec_data = []
        for rec in recommendations:
            rec_data.append({
                "id": rec.recommendation_id,
                "category": rec.category,
                "description": rec.description,
                "priority": rec.priority,
                "evidence": rec.evidence,
                "implementation_effort": rec.implementation_effort,
                "expected_impact": rec.expected_impact,
                "created_at": rec.created_at
            })
        click.echo(json.dumps(rec_data, indent=2))
    else:
        click.echo(f"✅ Generated {len(recommendations)} recommendations:")
        click.echo()
        
        # Group by priority
        priority_groups = {"high": [], "medium": [], "low": []}
        for rec in recommendations:
            if rec.priority in priority_groups:
                priority_groups[rec.priority].append(rec)
        
        for prio in ["high", "medium", "low"]:
            if not priority_groups[prio]:
                continue
            
            priority_icon = "🔴" if prio == "high" else "🟡" if prio == "medium" else "🟢"
            click.echo(f"{priority_icon} {prio.upper()} PRIORITY ({len(priority_groups[prio])}):")
            
            for rec in priority_groups[prio]:
                click.echo(f"  📋 {rec.category}: {rec.description[:80]}...")
                click.echo(f"     Implementation: {rec.implementation_effort}, Impact: {rec.expected_impact}")
                click.echo(f"     Evidence: {len(rec.evidence)} data points")
                click.echo()


@improvement_cli.command()
@click.option("--type", "pattern_type", help="Filter by pattern type")
@click.option("--min-confidence", type=float, default=0.0, help="Minimum confidence threshold")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
def patterns(pattern_type: Optional[str], min_confidence: float, output_json: bool):
    """List learned patterns from feedback analysis."""
    engine = get_improvement_engine()
    patterns = engine.get_patterns(pattern_type=pattern_type, min_confidence=min_confidence)
    
    if not patterns:
        click.echo("No patterns found matching criteria.")
        return
    
    if output_json:
        pattern_data = []
        for pattern in patterns:
            pattern_data.append({
                "id": pattern.pattern_id,
                "type": pattern.pattern_type,
                "confidence": pattern.confidence,
                "sample_count": pattern.sample_count,
                "success_rate": pattern.success_rate,
                "data": pattern.pattern_data,
                "tags": list(pattern.tags)
            })
        click.echo(json.dumps(pattern_data, indent=2))
    else:
        click.echo(f"📈 Found {len(patterns)} learned patterns:")
        click.echo()
        
        for pattern in patterns:
            confidence_indicator = "🔴" if pattern.confidence < 0.3 else "🟡" if pattern.confidence < 0.7 else "🟢"
            click.echo(f"{confidence_indicator} {pattern.pattern_id} ({pattern.pattern_type})")
            click.echo(f"   Confidence: {pattern.confidence:.2f}, Samples: {pattern.sample_count}")
            click.echo(f"   Success Rate: {pattern.success_rate:.2%}")
            click.echo(f"   Tags: {', '.join(pattern.tags)}")
            
            # Show specific pattern data based on type
            if pattern.pattern_type == "error_pattern":
                errors = pattern.pattern_data.get("error_patterns", [])
                click.echo(f"   Error Types: {', '.join(errors)}")
            elif pattern.pattern_type == "success_pattern":
                indicators = pattern.pattern_data.get("success_indicators", [])
                click.echo(f"   Success Indicators: {', '.join(indicators)}")
            elif pattern.pattern_type == "provider_performance":
                provider = pattern.pattern_data.get("provider", "unknown")
                metrics = pattern.pattern_data.get("performance_metrics", {})
                click.echo(f"   Provider: {provider}")
                click.echo(f"   Metrics: {metrics}")
            
            click.echo()


@improvement_cli.command()
@click.argument("patch_id")
@click.argument("outcome", type=click.Choice([o.value for o in PatchOutcome]))
@click.option("--feedback-type", type=click.Choice([t.value for t in FeedbackType]), default="human")
@click.option("--source", help="Feedback source", default="cli")
@click.option("--confidence", type=float, default=0.8, help="Confidence score")
@click.option("--context", help="Additional context (JSON string)")
@click.option("--metrics", help="Performance metrics (JSON string)")
def feedback(patch_id: str, outcome: str, feedback_type: str, source: str, 
             confidence: float, context: Optional[str], metrics: Optional[str]):
    """Record feedback for a patch application."""
    engine = get_improvement_engine()
    
    # Parse context and metrics
    context_dict = {}
    if context:
        try:
            context_dict = json.loads(context)
        except json.JSONDecodeError:
            click.echo("❌ Invalid JSON in context parameter")
            return
    
    metrics_dict = {}
    if metrics:
        try:
            metrics_dict = json.loads(metrics)
        except json.JSONDecodeError:
            click.echo("❌ Invalid JSON in metrics parameter")
            return
    
    # Create feedback record
    feedback_record = PatchFeedback(
        patch_id=patch_id,
        feedback_type=FeedbackType(feedback_type),
        outcome=PatchOutcome(outcome),
        timestamp=datetime.now().timestamp(),
        context=context_dict,
        metrics=metrics_dict,
        feedback_source=source,
        confidence_score=confidence
    )
    
    # Record the feedback
    engine.record_patch_feedback(feedback_record)
    
    click.echo(f"✅ Recorded {outcome} feedback for patch {patch_id}")


@improvement_cli.command()
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
def stats(output_json):
    """Show improvement engine statistics."""
    engine = get_improvement_engine()
    statistics = engine.get_statistics()
    
    if output_json:
        click.echo(json.dumps(statistics, indent=2))
    else:
        click.echo("📊 Improvement Engine Statistics")
        click.echo("=" * 40)
        
        # Feedback statistics
        feedback_stats = statistics.get("feedback", {})
        click.echo(f"📝 Feedback Records: {feedback_stats.get('total_count', 0)}")
        
        outcome_dist = feedback_stats.get("outcome_distribution", {})
        if outcome_dist:
            click.echo("   Outcome Distribution:")
            for outcome, count in outcome_dist.items():
                click.echo(f"     {outcome}: {count}")
        
        click.echo()
        
        # Pattern statistics
        pattern_stats = statistics.get("patterns", {})
        click.echo(f"🧠 Learning Patterns: {pattern_stats.get('total_count', 0)}")
        
        type_dist = pattern_stats.get("type_distribution", {})
        if type_dist:
            click.echo("   Pattern Types:")
            for pattern_type, count in type_dist.items():
                click.echo(f"     {pattern_type}: {count}")
        
        click.echo()
        
        # Recommendation statistics
        rec_stats = statistics.get("recommendations", {})
        click.echo(f"💡 Pending Recommendations: {rec_stats.get('pending_count', 0)}")
        
        priority_dist = rec_stats.get("priority_distribution", {})
        if priority_dist:
            click.echo("   Priority Distribution:")
            for priority, count in priority_dist.items():
                click.echo(f"     {priority}: {count}")
        
        click.echo()
        
        # Cache status
        cache_stats = statistics.get("cache_status", {})
        last_updated = cache_stats.get("last_updated", 0)
        if last_updated:
            updated_time = datetime.fromtimestamp(last_updated)
            click.echo(f"🔄 Cache Last Updated: {updated_time.strftime('%Y-%m-%d %H:%M:%S')}")
            click.echo(f"   Cached Patterns: {cache_stats.get('cached_patterns', 0)}")


@improvement_cli.command()
@click.argument("recommendation_id")
@click.option("--status", type=click.Choice(["implemented", "rejected", "pending"]), 
              default="implemented", help="New status for the recommendation")
def mark_recommendation(recommendation_id: str, status: str):
    """Mark a recommendation as implemented, rejected, or pending."""
    engine = get_improvement_engine()
    
    try:
        import sqlite3
        with sqlite3.connect(engine.db_path) as conn:
            cursor = conn.cursor()
            
            # Update recommendation status
            cursor.execute("""
                UPDATE improvement_recommendations 
                SET status = ? 
                WHERE recommendation_id = ?
            """, (status, recommendation_id))
            
            if cursor.rowcount == 0:
                click.echo(f"❌ Recommendation '{recommendation_id}' not found")
                return
            
            conn.commit()
            click.echo(f"✅ Marked recommendation '{recommendation_id}' as {status}")
    
    except Exception as e:
        click.echo(f"❌ Failed to update recommendation: {e}")


@improvement_cli.command()
@click.argument("output_file", type=click.Path())
@click.option("--format", "output_format", type=click.Choice(["json", "csv"]), 
              default="json", help="Export format")
def export(output_file: str, output_format: str):
    """Export improvement data for external analysis."""
    engine = get_improvement_engine()
    
    try:
        import sqlite3
        import csv
        from pathlib import Path
        
        output_path = Path(output_file)
        
        if output_format == "json":
            # Export as JSON
            data = {
                "statistics": engine.get_statistics(),
                "patterns": [
                    {
                        "id": p.pattern_id,
                        "type": p.pattern_type,
                        "confidence": p.confidence,
                        "sample_count": p.sample_count,
                        "success_rate": p.success_rate,
                        "data": p.pattern_data,
                        "tags": list(p.tags),
                        "last_updated": p.last_updated
                    }
                    for p in engine.get_patterns()
                ],
                "recommendations": [
                    {
                        "id": r.recommendation_id,
                        "category": r.category,
                        "description": r.description,
                        "priority": r.priority,
                        "evidence": r.evidence,
                        "implementation_effort": r.implementation_effort,
                        "expected_impact": r.expected_impact,
                        "created_at": r.created_at
                    }
                    for r in engine.get_recommendations()
                ]
            }
            
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
        
        elif output_format == "csv":
            # Export feedback data as CSV
            with sqlite3.connect(engine.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT patch_id, feedback_type, outcome, timestamp, 
                           feedback_source, confidence_score
                    FROM patch_feedback
                    ORDER BY timestamp
                """)
                
                with open(output_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'patch_id', 'feedback_type', 'outcome', 'timestamp',
                        'feedback_source', 'confidence_score'
                    ])
                    writer.writerows(cursor.fetchall())
        
        click.echo(f"✅ Exported improvement data to {output_path}")
    
    except Exception as e:
        click.echo(f"❌ Failed to export data: {e}")


if __name__ == "__main__":
    improvement_cli()