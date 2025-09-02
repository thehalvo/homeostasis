"""
Predictive Healing for Critical Components.

This module provides predictive healing capabilities that anticipate and prevent
failures before they occur using machine learning, time series analysis, and
anomaly detection.
"""

import logging
import numpy as np
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class PredictionType(Enum):
    """Types of predictions."""
    FAILURE_PROBABILITY = "failure_probability"
    TIME_TO_FAILURE = "time_to_failure"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    ANOMALY_SCORE = "anomaly_score"
    CAPACITY_THRESHOLD = "capacity_threshold"


class ComponentType(Enum):
    """Types of system components."""
    SERVICE = "service"
    DATABASE = "database"
    CACHE = "cache"
    QUEUE = "queue"
    NETWORK = "network"
    STORAGE = "storage"
    COMPUTE = "compute"
    LOAD_BALANCER = "load_balancer"


class HealthStatus(Enum):
    """Component health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    AT_RISK = "at_risk"
    CRITICAL = "critical"
    FAILING = "failing"


@dataclass
class ComponentMetrics:
    """Metrics for a system component."""
    component_id: str
    component_type: ComponentType
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_io: float
    network_io: float
    request_rate: float
    error_rate: float
    response_time_p95: float
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_vector(self) -> np.ndarray:
        """Convert metrics to feature vector."""
        return np.array([
            self.cpu_usage,
            self.memory_usage,
            self.disk_io,
            self.network_io,
            self.request_rate,
            self.error_rate,
            self.response_time_p95
        ] + list(self.custom_metrics.values()))


@dataclass
class FailurePrediction:
    """Prediction of potential failure."""
    component_id: str
    prediction_type: PredictionType
    probability: float
    time_to_failure: Optional[timedelta] = None
    confidence: float = 0.0
    contributing_factors: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    severity: str = "medium"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "component_id": self.component_id,
            "prediction_type": self.prediction_type.value,
            "probability": self.probability,
            "time_to_failure": self.time_to_failure.total_seconds() if self.time_to_failure else None,
            "confidence": self.confidence,
            "contributing_factors": self.contributing_factors,
            "recommended_actions": self.recommended_actions,
            "severity": self.severity
        }


@dataclass
class HealingRecommendation:
    """Recommended healing action."""
    action_type: str
    target_component: str
    priority: int  # 1-10, higher is more urgent
    description: str
    expected_improvement: float
    risk_level: str
    prerequisites: List[str] = field(default_factory=list)
    estimated_duration: Optional[timedelta] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "action_type": self.action_type,
            "target_component": self.target_component,
            "priority": self.priority,
            "description": self.description,
            "expected_improvement": self.expected_improvement,
            "risk_level": self.risk_level,
            "prerequisites": self.prerequisites,
            "estimated_duration": self.estimated_duration.total_seconds() if self.estimated_duration else None
        }


class PredictiveModel(ABC):
    """Abstract base class for predictive models."""
    
    @abstractmethod
    def train(self, historical_data: List[ComponentMetrics], failure_events: List[Dict[str, Any]]) -> None:
        """Train the model on historical data."""
        pass
    
    @abstractmethod
    def predict(self, current_metrics: ComponentMetrics) -> FailurePrediction:
        """Make a prediction based on current metrics."""
        pass
    
    @abstractmethod
    def update(self, new_data: ComponentMetrics, outcome: Optional[bool] = None) -> None:
        """Update model with new data."""
        pass


class AnomalyDetectionModel(PredictiveModel):
    """Anomaly detection using Isolation Forest."""
    
    def __init__(self, contamination: float = 0.1):
        self.model = IsolationForest(contamination=contamination, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []
        self.training_data = deque(maxlen=10000)
    
    def train(self, historical_data: List[ComponentMetrics], failure_events: List[Dict[str, Any]]) -> None:
        """Train anomaly detection model."""
        if not historical_data:
            return
        
        # Extract features
        X = np.array([metrics.to_vector() for metrics in historical_data])
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled)
        self.is_trained = True
        
        # Store training data for updates
        self.training_data.extend(historical_data)
        
        logger.info(f"Trained anomaly detection model on {len(historical_data)} samples")
    
    def predict(self, current_metrics: ComponentMetrics) -> FailurePrediction:
        """Predict anomaly score."""
        if not self.is_trained:
            return FailurePrediction(
                component_id=current_metrics.component_id,
                prediction_type=PredictionType.ANOMALY_SCORE,
                probability=0.0,
                confidence=0.0
            )
        
        # Extract and scale features
        X = current_metrics.to_vector().reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        # Get anomaly score
        anomaly_score = self.model.decision_function(X_scaled)[0]
        is_anomaly = self.model.predict(X_scaled)[0] == -1
        
        # Convert to probability (0-1 range)
        probability = 1 / (1 + np.exp(anomaly_score))
        
        # Identify contributing factors
        contributing_factors = self._identify_contributing_factors(current_metrics, X_scaled[0])
        
        # Generate recommendations
        recommended_actions = self._generate_recommendations(current_metrics, contributing_factors)
        
        return FailurePrediction(
            component_id=current_metrics.component_id,
            prediction_type=PredictionType.ANOMALY_SCORE,
            probability=probability if is_anomaly else 0.0,
            confidence=0.8 if self.is_trained else 0.0,
            contributing_factors=contributing_factors,
            recommended_actions=recommended_actions,
            severity="high" if probability > 0.8 else "medium" if probability > 0.5 else "low"
        )
    
    def update(self, new_data: ComponentMetrics, outcome: Optional[bool] = None) -> None:
        """Update model with new data."""
        self.training_data.append(new_data)
        
        # Retrain periodically
        if len(self.training_data) % 100 == 0:
            self.train(list(self.training_data), [])
    
    def _identify_contributing_factors(self, metrics: ComponentMetrics, scaled_features: np.ndarray) -> List[str]:
        """Identify which factors contribute most to anomaly."""
        factors = []
        feature_names = ["cpu", "memory", "disk_io", "network_io", "request_rate", "error_rate", "response_time"]
        
        # Find features that deviate most from normal
        mean_features = np.mean([m.to_vector() for m in self.training_data], axis=0)
        deviations = np.abs(metrics.to_vector() - mean_features)
        
        # Get top 3 deviating features
        top_indices = np.argsort(deviations)[-3:]
        
        for idx in top_indices:
            if idx < len(feature_names):
                factors.append(f"{feature_names[idx]}_anomaly")
        
        return factors
    
    def _generate_recommendations(self, metrics: ComponentMetrics, factors: List[str]) -> List[str]:
        """Generate recommendations based on anomaly factors."""
        recommendations = []
        
        if "cpu_anomaly" in factors:
            recommendations.append("Scale up compute resources")
            recommendations.append("Investigate CPU-intensive processes")
        
        if "memory_anomaly" in factors:
            recommendations.append("Increase memory allocation")
            recommendations.append("Check for memory leaks")
        
        if "error_rate_anomaly" in factors:
            recommendations.append("Review recent deployments")
            recommendations.append("Check dependency health")
        
        if "response_time_anomaly" in factors:
            recommendations.append("Optimize slow queries")
            recommendations.append("Add caching layer")
        
        return recommendations


class TimeSeriesPredictor(PredictiveModel):
    """Time series prediction for failure forecasting."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.is_trained = False
        self.time_series_data = {}
    
    def train(self, historical_data: List[ComponentMetrics], failure_events: List[Dict[str, Any]]) -> None:
        """Train time series model."""
        if len(historical_data) < self.window_size:
            return
        
        # Group by component
        component_data = {}
        for metrics in historical_data:
            if metrics.component_id not in component_data:
                component_data[metrics.component_id] = []
            component_data[metrics.component_id].append(metrics)
        
        # Create time series features
        X, y = [], []
        
        for component_id, data in component_data.items():
            # Sort by timestamp
            data.sort(key=lambda x: x.timestamp)
            
            # Create sliding windows
            for i in range(len(data) - self.window_size):
                window = data[i:i + self.window_size]
                target = data[i + self.window_size]
                
                # Extract features from window
                window_features = self._extract_window_features(window)
                X.append(window_features)
                
                # Target is whether failure occurred
                failure_occurred = any(
                    event.get("component_id") == component_id and
                    event.get("timestamp") >= target.timestamp and
                    event.get("timestamp") < target.timestamp + timedelta(hours=1)
                    for event in failure_events
                )
                y.append(1.0 if failure_occurred else 0.0)
        
        if X:
            self.model.fit(X, y)
            self.is_trained = True
            logger.info(f"Trained time series model on {len(X)} windows")
    
    def predict(self, current_metrics: ComponentMetrics) -> FailurePrediction:
        """Predict time to failure."""
        if not self.is_trained:
            return FailurePrediction(
                component_id=current_metrics.component_id,
                prediction_type=PredictionType.TIME_TO_FAILURE,
                probability=0.0
            )
        
        # Get historical data for component
        if current_metrics.component_id not in self.time_series_data:
            self.time_series_data[current_metrics.component_id] = deque(maxlen=self.window_size)
        
        self.time_series_data[current_metrics.component_id].append(current_metrics)
        
        if len(self.time_series_data[current_metrics.component_id]) < self.window_size:
            return FailurePrediction(
                component_id=current_metrics.component_id,
                prediction_type=PredictionType.TIME_TO_FAILURE,
                probability=0.0,
                confidence=0.0
            )
        
        # Extract features
        window_features = self._extract_window_features(list(self.time_series_data[current_metrics.component_id]))
        
        # Predict failure probability
        failure_prob = self.model.predict([window_features])[0]
        
        # Estimate time to failure
        time_to_failure = self._estimate_time_to_failure(failure_prob)
        
        return FailurePrediction(
            component_id=current_metrics.component_id,
            prediction_type=PredictionType.TIME_TO_FAILURE,
            probability=failure_prob,
            time_to_failure=time_to_failure,
            confidence=0.7 if self.is_trained else 0.0,
            severity="critical" if failure_prob > 0.7 else "high" if failure_prob > 0.5 else "medium"
        )
    
    def update(self, new_data: ComponentMetrics, outcome: Optional[bool] = None) -> None:
        """Update model with new data."""
        if new_data.component_id not in self.time_series_data:
            self.time_series_data[new_data.component_id] = deque(maxlen=self.window_size)
        self.time_series_data[new_data.component_id].append(new_data)
    
    def _extract_window_features(self, window: List[ComponentMetrics]) -> np.ndarray:
        """Extract features from time window."""
        features = []
        
        # Statistical features for each metric
        for attr in ["cpu_usage", "memory_usage", "error_rate", "response_time_p95"]:
            values = [getattr(m, attr) for m in window]
            features.extend([
                np.mean(values),
                np.std(values),
                np.min(values),
                np.max(values),
                np.percentile(values, 95)
            ])
        
        # Trend features
        timestamps = [(m.timestamp - window[0].timestamp).total_seconds() for m in window]
        for attr in ["cpu_usage", "memory_usage"]:
            values = [getattr(m, attr) for m in window]
            if len(set(timestamps)) > 1:  # Avoid division by zero
                slope = np.polyfit(timestamps, values, 1)[0]
                features.append(slope)
            else:
                features.append(0.0)
        
        return np.array(features)
    
    def _estimate_time_to_failure(self, failure_prob: float) -> Optional[timedelta]:
        """Estimate time to failure based on probability."""
        if failure_prob < 0.3:
            return None
        elif failure_prob < 0.5:
            return timedelta(hours=24)
        elif failure_prob < 0.7:
            return timedelta(hours=6)
        else:
            return timedelta(hours=1)


class ResourceExhaustionPredictor:
    """Predict resource exhaustion events."""
    
    def __init__(self):
        self.models = {}
        self.thresholds = {
            "cpu_usage": 0.85,
            "memory_usage": 0.90,
            "disk_usage": 0.85,
            "connection_pool": 0.80
        }
    
    def predict_exhaustion(
        self,
        component_id: str,
        historical_metrics: List[ComponentMetrics],
        resource_type: str
    ) -> FailurePrediction:
        """Predict resource exhaustion."""
        if len(historical_metrics) < 10:
            return FailurePrediction(
                component_id=component_id,
                prediction_type=PredictionType.RESOURCE_EXHAUSTION,
                probability=0.0
            )
        
        # Extract resource values
        timestamps = [(m.timestamp - historical_metrics[0].timestamp).total_seconds() / 3600 
                     for m in historical_metrics]
        
        if resource_type == "cpu":
            values = [m.cpu_usage for m in historical_metrics]
        elif resource_type == "memory":
            values = [m.memory_usage for m in historical_metrics]
        else:
            values = [m.custom_metrics.get(resource_type, 0) for m in historical_metrics]
        
        # Fit linear trend
        if len(set(timestamps)) > 1:
            coeffs = np.polyfit(timestamps, values, 1)
            slope, intercept = coeffs
            
            # Predict when threshold will be reached
            threshold = self.thresholds.get(f"{resource_type}_usage", 0.9)
            if slope > 0:
                hours_to_threshold = (threshold - values[-1]) / slope
                if hours_to_threshold > 0:
                    time_to_failure = timedelta(hours=hours_to_threshold)
                    probability = min(1.0, 1.0 / (1.0 + hours_to_threshold / 24))
                else:
                    time_to_failure = timedelta(hours=0)
                    probability = 1.0
            else:
                time_to_failure = None
                probability = 0.0
            
            return FailurePrediction(
                component_id=component_id,
                prediction_type=PredictionType.RESOURCE_EXHAUSTION,
                probability=probability,
                time_to_failure=time_to_failure,
                confidence=0.8,
                contributing_factors=[f"{resource_type}_growth_rate"],
                recommended_actions=[
                    f"Increase {resource_type} allocation",
                    f"Optimize {resource_type} usage",
                    "Implement auto-scaling"
                ],
                severity="high" if probability > 0.7 else "medium"
            )
        
        return FailurePrediction(
            component_id=component_id,
            prediction_type=PredictionType.RESOURCE_EXHAUSTION,
            probability=0.0
        )


class PredictiveHealer:
    """Main predictive healing orchestrator."""
    
    def __init__(self, healing_system: Any):
        self.healing_system = healing_system
        self.anomaly_detector = AnomalyDetectionModel()
        self.time_series_predictor = TimeSeriesPredictor()
        self.resource_predictor = ResourceExhaustionPredictor()
        
        self.component_history: Dict[str, deque] = {}
        self.predictions: Dict[str, List[FailurePrediction]] = {}
        self.healing_history: List[Dict[str, Any]] = []
    
    def analyze_component(self, metrics: ComponentMetrics) -> List[FailurePrediction]:
        """Analyze component and generate predictions."""
        predictions = []
        
        # Store metrics history
        if metrics.component_id not in self.component_history:
            self.component_history[metrics.component_id] = deque(maxlen=1000)
        self.component_history[metrics.component_id].append(metrics)
        
        # Anomaly detection
        anomaly_prediction = self.anomaly_detector.predict(metrics)
        if anomaly_prediction.probability > 0.5:
            predictions.append(anomaly_prediction)
        
        # Time series prediction
        time_series_prediction = self.time_series_predictor.predict(metrics)
        if time_series_prediction.probability > 0.3:
            predictions.append(time_series_prediction)
        
        # Resource exhaustion prediction
        for resource in ["cpu", "memory"]:
            history = list(self.component_history[metrics.component_id])
            resource_prediction = self.resource_predictor.predict_exhaustion(
                metrics.component_id,
                history,
                resource
            )
            if resource_prediction.probability > 0.3:
                predictions.append(resource_prediction)
        
        # Store predictions
        self.predictions[metrics.component_id] = predictions
        
        return predictions
    
    def get_healing_recommendations(
        self,
        component_id: str,
        predictions: List[FailurePrediction]
    ) -> List[HealingRecommendation]:
        """Generate healing recommendations based on predictions."""
        recommendations = []
        
        for prediction in predictions:
            if prediction.prediction_type == PredictionType.ANOMALY_SCORE:
                recommendations.extend(self._get_anomaly_recommendations(component_id, prediction))
            elif prediction.prediction_type == PredictionType.TIME_TO_FAILURE:
                recommendations.extend(self._get_failure_recommendations(component_id, prediction))
            elif prediction.prediction_type == PredictionType.RESOURCE_EXHAUSTION:
                recommendations.extend(self._get_resource_recommendations(component_id, prediction))
        
        # Sort by priority
        recommendations.sort(key=lambda x: x.priority, reverse=True)
        
        return recommendations
    
    def execute_preventive_healing(
        self,
        component_id: str,
        recommendation: HealingRecommendation,
        auto_approve: bool = False
    ) -> Dict[str, Any]:
        """Execute preventive healing action."""
        result = {
            "component_id": component_id,
            "recommendation": recommendation.to_dict(),
            "executed": False,
            "timestamp": datetime.now().isoformat()
        }
        
        if not auto_approve and recommendation.risk_level == "high":
            result["reason"] = "High-risk action requires manual approval"
            return result
        
        try:
            # Execute healing action through healing system
            healing_result = self.healing_system.execute_healing(
                component_id=component_id,
                action_type=recommendation.action_type,
                parameters={
                    "priority": recommendation.priority,
                    "preventive": True
                }
            )
            
            result["executed"] = True
            result["healing_result"] = healing_result
            
            # Record in history
            self.healing_history.append(result)
            
        except Exception as e:
            logger.error(f"Failed to execute preventive healing: {e}")
            result["error"] = str(e)
        
        return result
    
    def train_models(
        self,
        historical_data: List[ComponentMetrics],
        failure_events: List[Dict[str, Any]]
    ) -> None:
        """Train all predictive models."""
        logger.info("Training predictive models...")
        
        self.anomaly_detector.train(historical_data, failure_events)
        self.time_series_predictor.train(historical_data, failure_events)
        
        logger.info("Predictive models trained successfully")
    
    def get_component_health(self, component_id: str) -> Dict[str, Any]:
        """Get overall health assessment for a component."""
        predictions = self.predictions.get(component_id, [])
        
        # Calculate overall risk score
        if not predictions:
            risk_score = 0.0
            status = HealthStatus.HEALTHY
        else:
            risk_score = max(p.probability for p in predictions)
            if risk_score > 0.8:
                status = HealthStatus.CRITICAL
            elif risk_score > 0.6:
                status = HealthStatus.AT_RISK
            elif risk_score > 0.4:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY
        
        return {
            "component_id": component_id,
            "status": status.value,
            "risk_score": risk_score,
            "predictions": [p.to_dict() for p in predictions],
            "last_analyzed": datetime.now().isoformat()
        }
    
    def _get_anomaly_recommendations(
        self,
        component_id: str,
        prediction: FailurePrediction
    ) -> List[HealingRecommendation]:
        """Get recommendations for anomaly predictions."""
        recommendations = []
        
        if "cpu_anomaly" in prediction.contributing_factors:
            recommendations.append(HealingRecommendation(
                action_type="scale_horizontal",
                target_component=component_id,
                priority=8,
                description="Add more instances to handle CPU load",
                expected_improvement=0.7,
                risk_level="low",
                estimated_duration=timedelta(minutes=5)
            ))
        
        if "memory_anomaly" in prediction.contributing_factors:
            recommendations.append(HealingRecommendation(
                action_type="increase_memory",
                target_component=component_id,
                priority=7,
                description="Increase memory allocation",
                expected_improvement=0.6,
                risk_level="medium",
                prerequisites=["backup_state"],
                estimated_duration=timedelta(minutes=10)
            ))
        
        return recommendations
    
    def _get_failure_recommendations(
        self,
        component_id: str,
        prediction: FailurePrediction
    ) -> List[HealingRecommendation]:
        """Get recommendations for failure predictions."""
        recommendations = []
        
        if prediction.time_to_failure and prediction.time_to_failure < timedelta(hours=2):
            recommendations.append(HealingRecommendation(
                action_type="failover_prepare",
                target_component=component_id,
                priority=10,
                description="Prepare failover instance",
                expected_improvement=0.9,
                risk_level="low",
                estimated_duration=timedelta(minutes=15)
            ))
        
        recommendations.append(HealingRecommendation(
            action_type="health_check_increase",
            target_component=component_id,
            priority=6,
            description="Increase health check frequency",
            expected_improvement=0.3,
            risk_level="low",
            estimated_duration=timedelta(minutes=1)
        ))
        
        return recommendations
    
    def _get_resource_recommendations(
        self,
        component_id: str,
        prediction: FailurePrediction
    ) -> List[HealingRecommendation]:
        """Get recommendations for resource predictions."""
        recommendations = []
        
        if prediction.time_to_failure and prediction.time_to_failure < timedelta(hours=6):
            recommendations.append(HealingRecommendation(
                action_type="auto_scale_trigger",
                target_component=component_id,
                priority=9,
                description="Trigger auto-scaling policy",
                expected_improvement=0.8,
                risk_level="low",
                estimated_duration=timedelta(minutes=3)
            ))
        
        recommendations.append(HealingRecommendation(
            action_type="resource_optimization",
            target_component=component_id,
            priority=5,
            description="Run resource optimization routine",
            expected_improvement=0.4,
            risk_level="medium",
            prerequisites=["performance_baseline"],
            estimated_duration=timedelta(minutes=20)
        ))
        
        return recommendations


# Example usage
def analyze_critical_component(component_id: str, metrics: ComponentMetrics, healer: PredictiveHealer) -> Dict[str, Any]:
    """Analyze a critical component and get recommendations."""
    # Get predictions
    predictions = healer.analyze_component(metrics)
    
    # Get healing recommendations
    recommendations = healer.get_healing_recommendations(component_id, predictions)
    
    # Get overall health
    health = healer.get_component_health(component_id)
    
    return {
        "health": health,
        "recommendations": [r.to_dict() for r in recommendations],
        "auto_healing_available": len([r for r in recommendations if r.risk_level == "low"]) > 0
    }