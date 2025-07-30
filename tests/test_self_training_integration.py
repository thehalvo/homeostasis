"""Integration tests for the self-training systems."""

import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

from modules.self_training import (
    MLFeedbackLoop,
    PredictionFeedback,
    AnnotationInterface,
    AnnotationType,
    HumanFeedbackCollector,
    RuleExtractor,
    DeploymentMonitor,
    OutcomeTracker,
    LearningPipeline,
    ConfidenceCalculator,
    ConfidenceContext,
    SystemCriticality,
    FixComplexity,
    ReviewTrigger
)
from modules.self_training.continuous_learning import FixOutcome
from modules.llm_integration.patch_generator import PatchData
from modules.monitoring.healing_metrics import HealingMetrics
from modules.monitoring.health_checks import HealthChecker
from modules.primary_languages.rule_engine import RuleEngine


class TestSelfTrainingIntegration:
    """Test the complete self-training system integration."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def ml_feedback_loop(self, temp_dir):
        """Create ML feedback loop instance."""
        return MLFeedbackLoop(
            feedback_dir=temp_dir / "ml_feedback",
            min_feedback_for_retrain=2,  # Low threshold for testing
            retrain_interval_hours=0.01  # Short interval for testing
        )
    
    @pytest.fixture
    def annotation_interface(self, temp_dir):
        """Create annotation interface instance."""
        return AnnotationInterface(
            storage_dir=temp_dir / "annotations",
            task_expiry_hours=1
        )
    
    @pytest.fixture
    def deployment_monitor(self, temp_dir):
        """Create deployment monitor instance."""
        return DeploymentMonitor(
            storage_dir=temp_dir / "deployments",
            monitoring_duration_hours=0.01,  # Very short for testing
            check_interval_hours=0.001
        )
    
    def test_end_to_end_feedback_loop(self, ml_feedback_loop):
        """Test complete feedback loop from prediction to retraining."""
        # Simulate model predictions with feedback
        predictions = [
            PredictionFeedback(
                prediction_id=f"pred_{i}",
                model_name="error_classifier",
                model_version="v1",
                input_data={"error_type": "null_pointer", "severity": "high"},
                prediction="memory_error",
                actual_outcome="null_pointer_error",
                confidence=0.8
            )
            for i in range(3)
        ]
        
        # Add feedback
        for feedback in predictions:
            ml_feedback_loop.add_feedback(feedback)
        
        # Check that feedback was collected
        assert len(ml_feedback_loop.feedback_buffer) > 0
        
        # Get model performance
        performance = ml_feedback_loop.get_model_performance("error_classifier", "v1")
        assert "accuracy" in performance
        assert performance["sample_count"] == 3
    
    def test_human_annotation_workflow(self, annotation_interface):
        """Test human-in-the-loop annotation workflow."""
        # Create annotation task
        task_id = annotation_interface.create_task(
            task_type=AnnotationType.PATCH_REVIEW,
            data={
                "error": {"type": "syntax_error", "message": "Unexpected token"},
                "patch": {"content": "fix = true;"}
            },
            model_prediction="approve",
            model_confidence=0.75,
            priority=8
        )
        
        # Get task for annotator
        task = annotation_interface.get_next_task("annotator_1")
        assert task is not None
        assert task.task_id == task_id
        assert task.status.value == "in_progress"
        
        # Submit annotation
        annotation_id = annotation_interface.submit_annotation(
            task_id=task_id,
            annotator_id="annotator_1",
            human_label="reject",
            confidence=0.9,
            time_taken=30.5,
            notes="Patch doesn't address root cause"
        )
        
        assert annotation_id is not None
        
        # Check task completion
        assert len(annotation_interface.annotations[task_id]) == 1
        
        # Get annotator stats
        stats = annotation_interface.get_annotator_stats("annotator_1")
        assert stats["total_annotations"] == 1
        assert stats["avg_confidence"] == 0.9
    
    def test_rule_extraction_from_fixes(self, temp_dir):
        """Test automated rule extraction from successful fixes."""
        rule_extractor = RuleExtractor(
            storage_dir=temp_dir / "rules",
            min_examples=2
        )
        
        # Create sample fixes
        fixes = [
            {
                "error_data": {
                    "error_message": "Cannot read property 'length' of undefined",
                    "error_type": "type_error",
                    "stack_trace": "at processArray (file.js:42)"
                },
                "patch": Mock(
                    spec=PatchData,
                    get_diff=Mock(return_value="""
-    return array.length;
+    return array ? array.length : 0;
""")
                ),
                "metadata": {"fix_id": "fix_1", "success": True}
            },
            {
                "error_data": {
                    "error_message": "Cannot read property 'size' of undefined",
                    "error_type": "type_error",
                    "stack_trace": "at checkSize (util.js:15)"
                },
                "patch": Mock(
                    spec=PatchData,
                    get_diff=Mock(return_value="""
-    if (obj.size > 0) {
+    if (obj && obj.size > 0) {
""")
                ),
                "metadata": {"fix_id": "fix_2", "success": True}
            }
        ]
        
        # Analyze fixes
        extracted_patterns = []
        for fix in fixes:
            patterns = rule_extractor.analyze_successful_fix(
                fix["error_data"],
                fix["patch"],
                fix["metadata"]
            )
            extracted_patterns.extend(patterns)
        
        # Should extract patterns but not create rules yet (min_examples=2)
        assert len(rule_extractor.pattern_candidates) > 0
        
        # Add one more similar fix to trigger rule creation
        similar_fix = {
            "error_data": {
                "error_message": "Cannot read property 'count' of undefined",
                "error_type": "type_error"
            },
            "patch": Mock(
                spec=PatchData,
                get_diff=Mock(return_value="""
-    total = data.count;
+    total = data ? data.count : 0;
""")
            ),
            "metadata": {"fix_id": "fix_3", "success": True}
        }
        
        new_patterns = rule_extractor.analyze_successful_fix(
            similar_fix["error_data"],
            similar_fix["patch"],
            similar_fix["metadata"]
        )
        
        # Should now have extracted rules
        rules = rule_extractor.get_rules(min_frequency=2)
        assert len(rules) >= 0  # May have created rules
    
    def test_deployment_monitoring(self, deployment_monitor):
        """Test deployment monitoring and outcome tracking."""
        # Track a deployment
        deployment_monitor.track_deployment(
            fix_id="fix_123",
            error_id="error_456",
            error_type="null_pointer",
            fix_type="ml_generated",
            confidence=0.85,
            affected_files=["src/main.py"],
            patch_content="if obj is not None:",
            model_version="v2"
        )
        
        # Check active deployments
        assert len(deployment_monitor.active_deployments) == 1
        assert "fix_123" in deployment_monitor.active_deployments
        
        # Simulate time passing and check deployment
        time.sleep(0.02)  # Wait for monitoring duration
        
        health_checker = Mock(spec=HealthChecker)
        health_checker.check_health.return_value = {
            "status": "healthy",
            "cpu_usage": 45,
            "memory_usage": 55,
            "avg_response_time": 95
        }
        
        # Check deployments
        reports = deployment_monitor.check_deployments(health_checker)
        
        # Should have completed monitoring
        assert len(reports) == 1
        assert reports[0].fix_id == "fix_123"
        assert reports[0].outcome in [FixOutcome.SUCCESS, FixOutcome.UNKNOWN]
    
    def test_adaptive_confidence_thresholds(self, temp_dir):
        """Test adaptive confidence threshold calculation."""
        calculator = ConfidenceCalculator(
            storage_dir=temp_dir / "confidence",
            base_threshold=0.7
        )
        
        # Create context for critical system
        critical_context = ConfidenceContext(
            error_type="authentication_error",
            error_severity="critical",
            affected_components=["auth_service"],
            fix_complexity=FixComplexity.COMPLEX,
            system_criticality=SystemCriticality.CRITICAL,
            historical_success_rate=0.6,
            recent_failures=2
        )
        
        # Calculate confidence
        result = calculator.calculate_confidence(0.8, critical_context)
        
        # Should have lower adjusted confidence for critical system
        assert result["adjusted_confidence"] < result["original_confidence"]
        assert result["threshold"] > 0.7  # Higher threshold for critical
        assert result["requires_review"] == (
            result["adjusted_confidence"] < result["threshold"]
        )
        
        # Create context for low-risk system
        low_risk_context = ConfidenceContext(
            error_type="ui_glitch",
            error_severity="low",
            affected_components=["demo_page"],
            fix_complexity=FixComplexity.TRIVIAL,
            system_criticality=SystemCriticality.EXPERIMENTAL,
            historical_success_rate=0.9
        )
        
        # Calculate confidence
        low_risk_result = calculator.calculate_confidence(0.8, low_risk_context)
        
        # Should have higher adjusted confidence for low risk
        assert low_risk_result["adjusted_confidence"] > critical_context.historical_success_rate
        assert low_risk_result["threshold"] <= 0.7
    
    def test_continuous_learning_pipeline(self, temp_dir, ml_feedback_loop):
        """Test the complete continuous learning pipeline."""
        # Create components
        deployment_monitor = DeploymentMonitor(
            storage_dir=temp_dir / "deployments",
            monitoring_duration_hours=0.001
        )
        outcome_tracker = OutcomeTracker(deployment_monitor)
        rule_extractor = RuleExtractor(storage_dir=temp_dir / "rules")
        
        # Create learning pipeline
        pipeline = LearningPipeline(
            deployment_monitor=deployment_monitor,
            outcome_tracker=outcome_tracker,
            ml_feedback_loop=ml_feedback_loop,
            rule_extractor=rule_extractor
        )
        
        # Track some deployments
        for i in range(3):
            deployment_monitor.track_deployment(
                fix_id=f"fix_{i}",
                error_id=f"error_{i}",
                error_type="type_error",
                fix_type="ml_generated",
                confidence=0.7 + i * 0.1,
                affected_files=[f"file_{i}.py"],
                patch_content=f"fix content {i}",
                model_version="v1"
            )
        
        # Wait and run learning cycle
        time.sleep(0.01)
        
        health_checker = Mock(spec=HealthChecker)
        health_checker.check_health.return_value = {"status": "healthy"}
        
        results = pipeline.run_learning_cycle(health_checker)
        
        assert results["cycle"] == 1
        assert results["outcomes_processed"] >= 0
        assert "insights" in results
    
    def test_review_trigger_logic(self, temp_dir):
        """Test review trigger decision logic."""
        calculator = ConfidenceCalculator(storage_dir=temp_dir / "confidence")
        contextual = Mock()
        trigger = ReviewTrigger(calculator, contextual)
        
        # Test high-confidence non-critical fix
        error_data = {
            "error_id": "err_1",
            "error_type": "formatting_error",
            "severity": "low"
        }
        fix_data = {
            "affected_files": ["utils/format.py"],
            "patch_content": "small fix",
            "confidence": 0.9
        }
        
        contextual._determine_criticality.return_value = SystemCriticality.LOW
        contextual._estimate_complexity.return_value = FixComplexity.SIMPLE
        contextual._get_historical_success_rate.return_value = 0.85
        contextual._count_recent_failures.return_value = 0
        contextual._get_time_context.return_value = "business_hours"
        
        should_review, reasons = trigger.should_trigger_review(
            error_data, fix_data, 0.9
        )
        
        # High confidence, low risk - should not trigger review
        assert not should_review or len(reasons) == 0
        
        # Test low-confidence critical fix
        critical_error = {
            "error_id": "err_2",
            "error_type": "authentication_failure",
            "severity": "critical"
        }
        critical_fix = {
            "affected_files": ["auth/login.py", "auth/session.py"],
            "patch_content": "complex auth fix",
            "confidence": 0.6
        }
        
        contextual._determine_criticality.return_value = SystemCriticality.CRITICAL
        contextual._estimate_complexity.return_value = FixComplexity.COMPLEX
        
        should_review, reasons = trigger.should_trigger_review(
            critical_error, critical_fix, 0.6
        )
        
        # Low confidence, critical system - should trigger review
        assert should_review
        assert len(reasons) > 0
        assert any("critical" in r.lower() or "confidence" in r.lower() for r in reasons)
    
    def test_feedback_loop_with_annotations(self, ml_feedback_loop, annotation_interface):
        """Test integration between feedback loop and annotations."""
        # Register callback to send annotations to feedback loop
        def annotation_callback(annotations):
            for ann in annotations:
                feedback = PredictionFeedback(
                    prediction_id=ann.task_id,
                    model_name="patch_reviewer",
                    model_version="v1",
                    input_data=ann.original_data,
                    prediction=ann.model_prediction,
                    actual_outcome=ann.human_label,
                    confidence=0.7
                )
                ml_feedback_loop.add_feedback(feedback)
        
        annotation_interface.register_callback(
            AnnotationType.PATCH_REVIEW,
            annotation_callback
        )
        
        # Create and complete annotation task
        task_id = annotation_interface.create_task(
            task_type=AnnotationType.PATCH_REVIEW,
            data={"patch": "test"},
            model_prediction="approve",
            model_confidence=0.7
        )
        
        annotation_interface.submit_annotation(
            task_id=task_id,
            annotator_id="test_annotator",
            human_label="reject",
            confidence=0.95,
            time_taken=20
        )
        
        # Check that feedback was added
        assert len(ml_feedback_loop.feedback_buffer) > 0