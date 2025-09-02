"""
Comprehensive Observability Hooks for Homeostasis Framework

This module provides extensive observability capabilities through:
- OpenTelemetry integration for distributed tracing
- Structured logging with context propagation
- Metrics collection with multiple backends
- Custom instrumentation for healing operations
- Performance profiling hooks
- Error tracking and alerting
"""

import functools
import json
import logging
import time
import traceback
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum

# OpenTelemetry imports
try:
    from opentelemetry import trace, metrics, baggage
    from opentelemetry.context import attach, detach
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.semconv.resource import ResourceAttributes
    from opentelemetry.trace import Status, StatusCode
    from opentelemetry.metrics import CallbackOptions, Observation
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = None
    metrics = None

from modules.monitoring.monitoring_adapters import (
    AlertSeverity, Event, Metric, MetricType,
    create_monitoring_adapter
)

logger = logging.getLogger(__name__)


class ObservabilityLevel(Enum):
    """Levels of observability detail"""
    MINIMAL = "minimal"
    STANDARD = "standard"
    DETAILED = "detailed"
    DEBUG = "debug"


class OperationType(Enum):
    """Types of operations for categorization"""
    ERROR_DETECTION = "error_detection"
    ROOT_CAUSE_ANALYSIS = "root_cause_analysis"
    PATCH_GENERATION = "patch_generation"
    PATCH_VALIDATION = "patch_validation"
    DEPLOYMENT = "deployment"
    ROLLBACK = "rollback"
    MONITORING = "monitoring"
    HEALING_CYCLE = "healing_cycle"


@dataclass
class ObservabilityContext:
    """Context for observability data"""
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    parent_span_id: Optional[str] = None
    tenant_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    environment: str = "development"
    service_name: str = "homeostasis"
    service_version: str = "1.0.0"
    attributes: Dict[str, Any] = field(default_factory=dict)
    baggage: Dict[str, str] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Performance metrics for operations"""
    duration_ms: float
    cpu_time_ms: Optional[float] = None
    memory_used_mb: Optional[float] = None
    io_operations: Optional[int] = None
    network_calls: Optional[int] = None
    cache_hits: Optional[int] = None
    cache_misses: Optional[int] = None


class ObservabilityHooks:
    """
    Comprehensive observability hooks for the Homeostasis framework.
    
    Provides:
    - Distributed tracing with OpenTelemetry
    - Structured logging with context
    - Metrics collection and export
    - Performance profiling
    - Error tracking
    - Custom instrumentation
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize observability hooks with configuration"""
        self.config = config
        self.enabled = config.get('enabled', True)
        self.level = ObservabilityLevel(config.get('level', 'standard'))
        
        # Initialize monitoring adapters
        self.monitoring_adapters = []
        for adapter_config in config.get('monitoring_adapters', []):
            adapter = create_monitoring_adapter(
                adapter_config['provider'],
                adapter_config['config']
            )
            if adapter:
                self.monitoring_adapters.append(adapter)
        
        # Initialize OpenTelemetry if available
        self.tracer = None
        self.meter = None
        if OTEL_AVAILABLE and config.get('opentelemetry', {}).get('enabled', True):
            self._init_opentelemetry()
        
        # Initialize metrics collectors
        self._init_metrics_collectors()
        
        # Context storage
        self._context_stack = []
        
        # Performance tracking
        self.performance_stats = {}
        
    def _init_opentelemetry(self):
        """Initialize OpenTelemetry components"""
        try:
            otel_config = self.config.get('opentelemetry', {})
            
            # Create resource
            resource = Resource.create({
                ResourceAttributes.SERVICE_NAME: self.config.get('service_name', 'homeostasis'),
                ResourceAttributes.SERVICE_VERSION: self.config.get('service_version', '1.0.0'),
                ResourceAttributes.DEPLOYMENT_ENVIRONMENT: self.config.get('environment', 'development'),
            })
            
            # Initialize tracing
            if otel_config.get('tracing', {}).get('enabled', True):
                trace_endpoint = otel_config['tracing'].get('endpoint', 'localhost:4317')
                
                provider = TracerProvider(resource=resource)
                processor = BatchSpanProcessor(
                    OTLPSpanExporter(endpoint=trace_endpoint)
                )
                provider.add_span_processor(processor)
                trace.set_tracer_provider(provider)
                
                self.tracer = trace.get_tracer(__name__)
                
                # Auto-instrument libraries
                RequestsInstrumentor().instrument()
            
            # Initialize metrics
            if otel_config.get('metrics', {}).get('enabled', True):
                metrics_endpoint = otel_config['metrics'].get('endpoint', 'localhost:4317')
                
                reader = PeriodicExportingMetricReader(
                    exporter=OTLPMetricExporter(endpoint=metrics_endpoint),
                    export_interval_millis=otel_config['metrics'].get('export_interval', 60000)
                )
                
                provider = MeterProvider(resource=resource, metric_readers=[reader])
                metrics.set_meter_provider(provider)
                
                self.meter = metrics.get_meter(__name__)
                
            logger.info("OpenTelemetry initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenTelemetry: {e}")
    
    def _init_metrics_collectors(self):
        """Initialize metric collectors"""
        if not self.meter:
            return
        
        # Counter metrics
        self.healing_cycles_counter = self.meter.create_counter(
            name="homeostasis.healing_cycles",
            description="Number of healing cycles executed",
            unit="1"
        )
        
        self.patches_generated_counter = self.meter.create_counter(
            name="homeostasis.patches_generated",
            description="Number of patches generated",
            unit="1"
        )
        
        self.errors_detected_counter = self.meter.create_counter(
            name="homeostasis.errors_detected",
            description="Number of errors detected",
            unit="1"
        )
        
        self.rollbacks_counter = self.meter.create_counter(
            name="homeostasis.rollbacks",
            description="Number of rollbacks performed",
            unit="1"
        )
        
        # Histogram metrics
        self.healing_duration_histogram = self.meter.create_histogram(
            name="homeostasis.healing_duration",
            description="Duration of healing cycles",
            unit="ms"
        )
        
        self.patch_generation_duration_histogram = self.meter.create_histogram(
            name="homeostasis.patch_generation_duration",
            description="Duration of patch generation",
            unit="ms"
        )
        
        # Gauge metrics (using callbacks)
        self.meter.create_observable_gauge(
            name="homeostasis.active_healing_cycles",
            callbacks=[self._get_active_healing_cycles],
            description="Number of active healing cycles",
            unit="1"
        )
        
        self.meter.create_observable_gauge(
            name="homeostasis.system_health_score",
            callbacks=[self._get_system_health_score],
            description="Overall system health score (0-100)",
            unit="1"
        )
    
    def _get_active_healing_cycles(self, options: CallbackOptions) -> List[Observation]:
        """Callback for active healing cycles gauge"""
        # This would be implemented to track actual active cycles
        return [Observation(value=0)]
    
    def _get_system_health_score(self, options: CallbackOptions) -> List[Observation]:
        """Callback for system health score gauge"""
        # This would calculate a health score based on various metrics
        return [Observation(value=100)]
    
    @contextmanager
    def operation_context(
        self,
        operation_type: OperationType,
        operation_name: str,
        attributes: Optional[Dict[str, Any]] = None
    ):
        """
        Context manager for tracking operations with full observability.
        
        Args:
            operation_type: Type of operation being performed
            operation_name: Name of the specific operation
            attributes: Additional attributes to attach to the operation
        """
        if not self.enabled:
            yield
            return
        
        start_time = time.time()
        context = self._get_current_context()
        span = None
        token = None
        
        try:
            # Start span if tracer is available
            if self.tracer:
                span = self.tracer.start_span(
                    name=operation_name,
                    attributes={
                        "operation.type": operation_type.value,
                        "tenant.id": context.tenant_id,
                        "environment": context.environment,
                        **(attributes or {})
                    }
                )
                
                # Set baggage for context propagation
                if context.baggage:
                    token = attach(baggage.set_baggage("context", json.dumps(context.baggage)))
            
            # Log operation start
            self._log_operation_start(operation_type, operation_name, attributes)
            
            # Push context
            self._context_stack.append(context)
            
            yield
            
            # Record success metrics
            duration_ms = (time.time() - start_time) * 1000
            self._record_operation_success(operation_type, operation_name, duration_ms)
            
            if span:
                span.set_status(Status(StatusCode.OK))
            
        except Exception as e:
            # Record failure metrics
            duration_ms = (time.time() - start_time) * 1000
            self._record_operation_failure(operation_type, operation_name, duration_ms, e)
            
            if span:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
            
            # Log error with context
            self._log_operation_error(operation_type, operation_name, e)
            
            raise
            
        finally:
            # Pop context
            if self._context_stack:
                self._context_stack.pop()
            
            # End span
            if span:
                span.end()
            
            # Detach baggage
            if token:
                detach(token)
    
    def track_healing_cycle(
        self,
        cycle_id: str,
        error_details: Dict[str, Any],
        tenant_id: Optional[str] = None
    ):
        """
        Track a complete healing cycle with detailed observability.
        
        Args:
            cycle_id: Unique identifier for the healing cycle
            error_details: Details about the error being healed
            tenant_id: Tenant identifier for multi-tenancy
        """
        attributes = {
            "healing.cycle_id": cycle_id,
            "error.type": error_details.get('type', 'unknown'),
            "error.severity": error_details.get('severity', 'unknown'),
            "tenant.id": tenant_id
        }
        
        with self.operation_context(
            OperationType.HEALING_CYCLE,
            f"healing_cycle_{cycle_id}",
            attributes
        ):
            # Update metrics
            if self.healing_cycles_counter:
                self.healing_cycles_counter.add(
                    1,
                    attributes={"tenant.id": tenant_id} if tenant_id else {}
                )
            
            # Send event to monitoring systems
            self._send_event(
                title=f"Healing Cycle Started: {cycle_id}",
                text=f"Healing error type: {error_details.get('type')}",
                event_type="healing_cycle_start",
                tags={"cycle_id": cycle_id, "tenant_id": tenant_id} if tenant_id else {"cycle_id": cycle_id}
            )
    
    def track_patch_generation(
        self,
        patch_id: str,
        file_path: str,
        patch_type: str,
        success: bool,
        duration_ms: float,
        tenant_id: Optional[str] = None
    ):
        """Track patch generation with metrics and traces"""
        if not self.enabled:
            return
        
        attributes = {
            "patch.id": patch_id,
            "patch.file_path": file_path,
            "patch.type": patch_type,
            "patch.success": success,
            "tenant.id": tenant_id
        }
        
        # Update metrics
        if self.patches_generated_counter:
            self.patches_generated_counter.add(
                1,
                attributes={
                    "success": str(success),
                    "patch_type": patch_type,
                    **({"tenant.id": tenant_id} if tenant_id else {})
                }
            )
        
        if self.patch_generation_duration_histogram:
            self.patch_generation_duration_histogram.record(
                duration_ms,
                attributes={"patch_type": patch_type}
            )
        
        # Send metric to monitoring adapters
        self._send_metric(
            name="patch_generation_duration",
            value=duration_ms,
            metric_type=MetricType.HISTOGRAM,
            tags={
                "patch_type": patch_type,
                "success": str(success),
                **({"tenant_id": tenant_id} if tenant_id else {})
            }
        )
    
    def track_error_detection(
        self,
        error_id: str,
        error_type: str,
        severity: str,
        source: str,
        tenant_id: Optional[str] = None
    ):
        """Track error detection events"""
        if not self.enabled:
            return
        
        # Update metrics
        if self.errors_detected_counter:
            self.errors_detected_counter.add(
                1,
                attributes={
                    "error_type": error_type,
                    "severity": severity,
                    "source": source,
                    **({"tenant.id": tenant_id} if tenant_id else {})
                }
            )
        
        # Send alert for critical errors
        if severity in ["critical", "high"]:
            self._send_alert(
                name=f"Critical Error Detected: {error_type}",
                severity=AlertSeverity.CRITICAL if severity == "critical" else AlertSeverity.ERROR,
                message=f"Error {error_id} detected from {source}",
                tags={
                    "error_id": error_id,
                    "error_type": error_type,
                    **({"tenant_id": tenant_id} if tenant_id else {})
                }
            )
    
    def track_deployment(
        self,
        deployment_id: str,
        environment: str,
        success: bool,
        duration_ms: float,
        rollback_triggered: bool = False,
        tenant_id: Optional[str] = None
    ):
        """Track deployment operations"""
        if not self.enabled:
            return
        
        # Send event
        self._send_event(
            title=f"Deployment {'Succeeded' if success else 'Failed'}: {deployment_id}",
            text=f"Environment: {environment}, Duration: {duration_ms}ms",
            event_type="deployment",
            tags={
                "deployment_id": deployment_id,
                "environment": environment,
                "success": str(success),
                "rollback_triggered": str(rollback_triggered),
                **({"tenant_id": tenant_id} if tenant_id else {})
            }
        )
        
        # Track rollback if triggered
        if rollback_triggered and self.rollbacks_counter:
            self.rollbacks_counter.add(
                1,
                attributes={
                    "environment": environment,
                    "trigger": "deployment_failure",
                    **({"tenant.id": tenant_id} if tenant_id else {})
                }
            )
    
    def create_span(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        """Create a new span for tracing"""
        if self.tracer:
            return self.tracer.start_span(name, attributes=attributes)
        return None
    
    def add_event_to_span(
        self,
        span: Any,
        name: str,
        attributes: Optional[Dict[str, Any]] = None
    ):
        """Add an event to an existing span"""
        if span and hasattr(span, 'add_event'):
            span.add_event(name, attributes=attributes or {})
    
    def set_span_attribute(self, span: Any, key: str, value: Any):
        """Set an attribute on a span"""
        if span and hasattr(span, 'set_attribute'):
            span.set_attribute(key, value)
    
    def record_exception(self, span: Any, exception: Exception):
        """Record an exception in a span"""
        if span and hasattr(span, 'record_exception'):
            span.record_exception(exception)
    
    def _get_current_context(self) -> ObservabilityContext:
        """Get the current observability context"""
        if self._context_stack:
            return self._context_stack[-1]
        
        # Create default context
        return ObservabilityContext(
            environment=self.config.get('environment', 'development'),
            service_name=self.config.get('service_name', 'homeostasis'),
            service_version=self.config.get('service_version', '1.0.0')
        )
    
    def _log_operation_start(
        self,
        operation_type: OperationType,
        operation_name: str,
        attributes: Optional[Dict[str, Any]]
    ):
        """Log the start of an operation"""
        if self.level == ObservabilityLevel.MINIMAL:
            return
        
        context = self._get_current_context()
        log_data = {
            "event": "operation_start",
            "operation_type": operation_type.value,
            "operation_name": operation_name,
            "trace_id": context.trace_id,
            "tenant_id": context.tenant_id,
            "environment": context.environment,
            **(attributes or {})
        }
        
        logger.info(f"Operation started: {operation_name}", extra={"structured": log_data})
    
    def _log_operation_error(
        self,
        operation_type: OperationType,
        operation_name: str,
        error: Exception
    ):
        """Log an operation error with full context"""
        context = self._get_current_context()
        log_data = {
            "event": "operation_error",
            "operation_type": operation_type.value,
            "operation_name": operation_name,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "trace_id": context.trace_id,
            "tenant_id": context.tenant_id,
            "environment": context.environment,
            "stack_trace": traceback.format_exc() if self.level == ObservabilityLevel.DEBUG else None
        }
        
        logger.error(f"Operation failed: {operation_name}", extra={"structured": log_data})
    
    def _record_operation_success(
        self,
        operation_type: OperationType,
        operation_name: str,
        duration_ms: float
    ):
        """Record successful operation metrics"""
        if self.healing_duration_histogram and operation_type == OperationType.HEALING_CYCLE:
            self.healing_duration_histogram.record(
                duration_ms,
                attributes={"operation": operation_name}
            )
        
        # Update performance stats
        if operation_type not in self.performance_stats:
            self.performance_stats[operation_type] = {
                "count": 0,
                "total_duration": 0,
                "min_duration": float('inf'),
                "max_duration": 0
            }
        
        stats = self.performance_stats[operation_type]
        stats["count"] += 1
        stats["total_duration"] += duration_ms
        stats["min_duration"] = min(stats["min_duration"], duration_ms)
        stats["max_duration"] = max(stats["max_duration"], duration_ms)
    
    def _record_operation_failure(
        self,
        operation_type: OperationType,
        operation_name: str,
        duration_ms: float,
        error: Exception
    ):
        """Record failed operation metrics"""
        # Send alert for critical operation failures
        if operation_type in [OperationType.HEALING_CYCLE, OperationType.DEPLOYMENT]:
            self._send_alert(
                name=f"Operation Failed: {operation_name}",
                severity=AlertSeverity.ERROR,
                message=f"{operation_type.value} failed: {str(error)}",
                tags={
                    "operation_type": operation_type.value,
                    "operation_name": operation_name,
                    "error_type": type(error).__name__
                }
            )
    
    def _send_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType = MetricType.GAUGE,
        tags: Optional[Dict[str, str]] = None,
        unit: Optional[str] = None,
        description: Optional[str] = None
    ):
        """Send metric to all configured monitoring adapters"""
        metric = Metric(
            name=f"homeostasis.{name}",
            value=value,
            timestamp=datetime.utcnow(),
            metric_type=metric_type,
            tags=tags or {},
            unit=unit,
            description=description
        )
        
        for adapter in self.monitoring_adapters:
            try:
                adapter.send_metric(metric)
            except Exception as e:
                logger.error(f"Failed to send metric to {type(adapter).__name__}: {e}")
    
    def _send_event(
        self,
        title: str,
        text: str,
        event_type: str = "info",
        tags: Optional[Dict[str, str]] = None,
        priority: Optional[str] = None
    ):
        """Send event to all configured monitoring adapters"""
        event = Event(
            event_id=f"{event_type}_{int(time.time() * 1000)}",
            title=title,
            text=text,
            timestamp=datetime.utcnow(),
            event_type=event_type,
            tags=tags or {},
            priority=priority
        )
        
        for adapter in self.monitoring_adapters:
            try:
                adapter.send_event(event)
            except Exception as e:
                logger.error(f"Failed to send event to {type(adapter).__name__}: {e}")
    
    def _send_alert(
        self,
        name: str,
        severity: AlertSeverity,
        message: str,
        tags: Optional[Dict[str, str]] = None
    ):
        """Send alert to all configured monitoring adapters"""
        # For now, we'll send as a high-priority event
        # In a real implementation, this would integrate with alerting systems
        priority = "critical" if severity == AlertSeverity.CRITICAL else "high"
        
        self._send_event(
            title=f"ALERT: {name}",
            text=message,
            event_type="alert",
            tags={
                "severity": severity.value,
                **(tags or {})
            },
            priority=priority
        )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of performance statistics"""
        summary = {}
        
        for op_type, stats in self.performance_stats.items():
            if stats["count"] > 0:
                summary[op_type.value] = {
                    "count": stats["count"],
                    "avg_duration_ms": stats["total_duration"] / stats["count"],
                    "min_duration_ms": stats["min_duration"],
                    "max_duration_ms": stats["max_duration"],
                    "total_duration_ms": stats["total_duration"]
                }
        
        return summary
    
    def export_telemetry_data(self, filepath: str):
        """Export collected telemetry data for analysis"""
        data = {
            "timestamp": datetime.utcnow().isoformat(),
            "performance_summary": self.get_performance_summary(),
            "configuration": {
                "level": self.level.value,
                "environment": self.config.get('environment', 'development'),
                "monitoring_adapters": [
                    type(adapter).__name__ for adapter in self.monitoring_adapters
                ]
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Telemetry data exported to {filepath}")


# Decorator for automatic operation tracking
def track_operation(
    operation_type: OperationType,
    extract_attributes: Optional[Callable] = None
):
    """
    Decorator to automatically track operations with observability.
    
    Args:
        operation_type: Type of operation being tracked
        extract_attributes: Optional function to extract attributes from function arguments
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get observability hooks from somewhere (e.g., global registry or first arg)
            hooks = kwargs.get('observability_hooks')
            if not hooks:
                # Try to get from first argument if it has the attribute
                if args and hasattr(args[0], 'observability_hooks'):
                    hooks = args[0].observability_hooks
            
            if not hooks or not hooks.enabled:
                return func(*args, **kwargs)
            
            # Extract attributes if function provided
            attributes = {}
            if extract_attributes:
                try:
                    attributes = extract_attributes(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Failed to extract attributes: {e}")
            
            # Track the operation
            with hooks.operation_context(
                operation_type,
                func.__name__,
                attributes
            ):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


# Global instance management
_observability_hooks = None

def init_observability_hooks(config: Dict[str, Any]) -> ObservabilityHooks:
    """Initialize global observability hooks"""
    global _observability_hooks
    _observability_hooks = ObservabilityHooks(config)
    return _observability_hooks

def get_observability_hooks() -> Optional[ObservabilityHooks]:
    """Get the global observability hooks instance"""
    return _observability_hooks