"""
SLA Monitoring and Alerting System

Provides comprehensive Service Level Agreement (SLA) monitoring, tracking,
and alerting capabilities for enterprise environments. Supports multiple
SLA types, real-time monitoring, and automated escalation.
"""

import asyncio
import datetime
import logging
import statistics
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from collections import defaultdict, deque

from modules.monitoring.alert_system import AlertManager
from modules.monitoring.observability_hooks import ObservabilityHooks
from modules.security.audit import get_audit_logger
from modules.monitoring.logger import MonitoringLogger

logger = logging.getLogger(__name__)


class SLAType(Enum):
    """Types of SLAs"""
    AVAILABILITY = "availability"
    RESPONSE_TIME = "response_time"
    RESOLUTION_TIME = "resolution_time"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    CUSTOM = "custom"


class SLAStatus(Enum):
    """SLA compliance status"""
    MEETING = "meeting"
    AT_RISK = "at_risk"
    BREACHED = "breached"
    UNKNOWN = "unknown"


class EscalationLevel(Enum):
    """Escalation levels for SLA breaches"""
    LEVEL_1 = "level_1"  # Team lead
    LEVEL_2 = "level_2"  # Manager
    LEVEL_3 = "level_3"  # Director
    LEVEL_4 = "level_4"  # Executive


@dataclass
class SLADefinition:
    """Service Level Agreement definition"""
    sla_id: str
    name: str
    description: str
    type: SLAType
    target_value: float
    measurement_window: int  # seconds
    calculation_method: str  # average, percentile, min, max
    business_hours_only: bool = False
    exclusion_periods: List[Dict[str, Any]] = field(default_factory=list)
    warning_threshold: float = 0.9  # Warn at 90% of target
    critical_threshold: float = 0.95  # Critical at 95% of target
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SLATarget:
    """SLA target configuration"""
    service: str
    endpoint: Optional[str] = None
    customer: Optional[str] = None
    tier: Optional[str] = None  # bronze, silver, gold, platinum
    custom_conditions: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SLAMetric:
    """SLA metric measurement"""
    timestamp: datetime.datetime
    value: float
    sample_count: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SLAViolation:
    """SLA violation record"""
    violation_id: str
    sla_id: str
    target: SLATarget
    start_time: datetime.datetime
    end_time: Optional[datetime.datetime]
    severity: str  # warning, critical, breach
    current_value: float
    target_value: float
    duration_seconds: Optional[float] = None
    escalation_level: Optional[EscalationLevel] = None
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    resolution_notes: Optional[str] = None


@dataclass
class EscalationPolicy:
    """Escalation policy for SLA violations"""
    policy_id: str
    name: str
    sla_ids: List[str]
    escalation_rules: List[Dict[str, Any]]  # List of escalation rules
    notification_channels: Dict[EscalationLevel, List[str]]
    enabled: bool = True


@dataclass
class SLAReport:
    """SLA compliance report"""
    report_id: str
    period_start: datetime.datetime
    period_end: datetime.datetime
    sla_summaries: Dict[str, Dict[str, Any]]
    overall_compliance: float
    violations: List[SLAViolation]
    recommendations: List[str]
    generated_at: datetime.datetime


class SLAMonitoringSystem:
    """
    Comprehensive SLA monitoring and alerting system.
    
    Tracks service level agreements, monitors compliance in real-time,
    and manages escalations for violations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize SLA monitoring system.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Storage
        self.storage_path = Path(config.get('storage_path', 'data/sla_monitoring'))
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Managers
        self.alert_manager = AlertManager(config.get('alert_config', {}))
        self.audit_logger = get_audit_logger()
        self.monitoring_logger = MonitoringLogger("sla_monitoring")
        self.observability = ObservabilityHooks(config)
        
        # SLA stores
        self.sla_definitions: Dict[str, SLADefinition] = {}
        self.sla_targets: Dict[str, List[SLATarget]] = defaultdict(list)
        self.escalation_policies: Dict[str, EscalationPolicy] = {}
        
        # Monitoring data
        self.metrics_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.active_violations: Dict[str, SLAViolation] = {}
        self.violation_history: List[SLAViolation] = []
        
        # Monitoring state
        self.monitoring_tasks: Dict[str, asyncio.Task] = {}
        self.monitoring_enabled = True
        
        # Metric collectors
        self.metric_collectors: Dict[str, Callable] = {}
        
        # Business hours configuration
        self.business_hours = config.get('business_hours', {
            'timezone': 'UTC',
            'start': '09:00',
            'end': '17:00',
            'days': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        })
        
        # Initialize default components
        self._register_default_collectors()
        self._load_sla_data()
        
        logger.info("Initialized SLA monitoring system")
    
    def _register_default_collectors(self):
        """Register default metric collectors"""
        # Availability collector
        self.register_metric_collector('availability', self._collect_availability_metric)
        
        # Response time collector
        self.register_metric_collector('response_time', self._collect_response_time_metric)
        
        # Error rate collector
        self.register_metric_collector('error_rate', self._collect_error_rate_metric)
        
        # Throughput collector
        self.register_metric_collector('throughput', self._collect_throughput_metric)
    
    def create_sla(self, name: str, description: str, type: SLAType,
                   target_value: float, measurement_window: int,
                   calculation_method: str = "average",
                   business_hours_only: bool = False,
                   warning_threshold: float = 0.9,
                   critical_threshold: float = 0.95) -> str:
        """Create a new SLA definition.
        
        Args:
            name: SLA name
            description: SLA description
            type: Type of SLA
            target_value: Target value for the SLA
            measurement_window: Measurement window in seconds
            calculation_method: How to calculate the metric
            business_hours_only: Whether to measure only during business hours
            warning_threshold: Warning threshold (fraction of target)
            critical_threshold: Critical threshold (fraction of target)
            
        Returns:
            SLA ID
        """
        sla_id = f"sla_{datetime.datetime.utcnow().timestamp()}"
        
        sla = SLADefinition(
            sla_id=sla_id,
            name=name,
            description=description,
            type=type,
            target_value=target_value,
            measurement_window=measurement_window,
            calculation_method=calculation_method,
            business_hours_only=business_hours_only,
            warning_threshold=warning_threshold,
            critical_threshold=critical_threshold
        )
        
        self.sla_definitions[sla_id] = sla
        self._save_sla_data()
        
        # Log creation
        self.audit_logger.log_event(
            event_type='sla_created',
            user='system',
            details={
                'sla_id': sla_id,
                'name': name,
                'type': type.value,
                'target': target_value
            }
        )
        
        return sla_id
    
    def add_sla_target(self, sla_id: str, service: str,
                      endpoint: Optional[str] = None,
                      customer: Optional[str] = None,
                      tier: Optional[str] = None,
                      custom_conditions: Optional[Dict[str, Any]] = None):
        """Add a target for SLA monitoring.
        
        Args:
            sla_id: SLA ID
            service: Service name
            endpoint: Optional endpoint
            customer: Optional customer
            tier: Optional service tier
            custom_conditions: Optional custom conditions
        """
        if sla_id not in self.sla_definitions:
            raise ValueError(f"SLA {sla_id} not found")
        
        target = SLATarget(
            service=service,
            endpoint=endpoint,
            customer=customer,
            tier=tier,
            custom_conditions=custom_conditions or {}
        )
        
        self.sla_targets[sla_id].append(target)
        
        # Start monitoring if not already running
        if sla_id not in self.monitoring_tasks:
            self._start_sla_monitoring(sla_id)
    
    def _start_sla_monitoring(self, sla_id: str):
        """Start monitoring for an SLA"""
        if not self.monitoring_enabled:
            return
        
        sla = self.sla_definitions.get(sla_id)
        if not sla or not sla.enabled:
            return
        
        async def monitor_loop():
            while sla_id in self.sla_definitions and self.monitoring_enabled:
                try:
                    # Collect metrics for all targets
                    for target in self.sla_targets.get(sla_id, []):
                        metric = await self._collect_metric(sla, target)
                        if metric:
                            self._store_metric(sla_id, target, metric)
                            await self._evaluate_sla_compliance(sla_id, target, metric)
                    
                    # Wait for next collection interval
                    await asyncio.sleep(min(30, sla.measurement_window / 10))
                    
                except Exception as e:
                    logger.error(f"SLA monitoring error for {sla_id}: {e}")
                    await asyncio.sleep(30)
        
        task = asyncio.create_task(monitor_loop())
        self.monitoring_tasks[sla_id] = task
    
    async def _collect_metric(self, sla: SLADefinition, target: SLATarget) -> Optional[SLAMetric]:
        """Collect metric for SLA target"""
        # Check if within business hours if required
        if sla.business_hours_only and not self._is_business_hours():
            return None
        
        # Check exclusion periods
        if self._is_excluded_period(sla):
            return None
        
        # Get appropriate collector
        collector = self.metric_collectors.get(sla.type.value)
        if not collector:
            logger.warning(f"No collector for SLA type {sla.type.value}")
            return None
        
        # Collect metric
        try:
            value = await collector(target, sla)
            return SLAMetric(
                timestamp=datetime.datetime.utcnow(),
                value=value,
                metadata={'target': target.__dict__}
            )
        except Exception as e:
            logger.error(f"Metric collection failed: {e}")
            return None
    
    def _store_metric(self, sla_id: str, target: SLATarget, metric: SLAMetric):
        """Store metric in buffer"""
        key = self._get_metric_key(sla_id, target)
        self.metrics_buffer[key].append(metric)
    
    async def _evaluate_sla_compliance(self, sla_id: str, target: SLATarget, latest_metric: SLAMetric):
        """Evaluate SLA compliance based on metrics"""
        sla = self.sla_definitions[sla_id]
        key = self._get_metric_key(sla_id, target)
        
        # Get metrics within measurement window
        window_start = datetime.datetime.utcnow() - datetime.timedelta(seconds=sla.measurement_window)
        window_metrics = [
            m for m in self.metrics_buffer[key]
            if m.timestamp >= window_start
        ]
        
        if not window_metrics:
            return
        
        # Calculate aggregate value
        values = [m.value for m in window_metrics]
        
        if sla.calculation_method == "average":
            current_value = statistics.mean(values)
        elif sla.calculation_method == "percentile_95":
            current_value = self._calculate_percentile(values, 95)
        elif sla.calculation_method == "percentile_99":
            current_value = self._calculate_percentile(values, 99)
        elif sla.calculation_method == "min":
            current_value = min(values)
        elif sla.calculation_method == "max":
            current_value = max(values)
        else:
            current_value = statistics.mean(values)
        
        # Determine compliance status
        status = self._determine_compliance_status(sla, current_value)
        
        # Handle status changes
        violation_key = f"{sla_id}_{self._get_target_key(target)}"
        
        if status in [SLAStatus.AT_RISK, SLAStatus.BREACHED]:
            # Create or update violation
            if violation_key not in self.active_violations:
                violation = SLAViolation(
                    violation_id=f"vio_{datetime.datetime.utcnow().timestamp()}",
                    sla_id=sla_id,
                    target=target,
                    start_time=datetime.datetime.utcnow(),
                    end_time=None,
                    severity="warning" if status == SLAStatus.AT_RISK else "critical",
                    current_value=current_value,
                    target_value=sla.target_value
                )
                self.active_violations[violation_key] = violation
                
                # Send alert
                await self._send_sla_alert(sla, violation, status)
                
                # Log violation
                self.audit_logger.log_event(
                    event_type='sla_violation_started',
                    user='system',
                    details={
                        'sla_id': sla_id,
                        'violation_id': violation.violation_id,
                        'severity': violation.severity,
                        'current_value': current_value,
                        'target_value': sla.target_value
                    }
                )
            else:
                # Update existing violation
                violation = self.active_violations[violation_key]
                violation.current_value = current_value
                
                # Check for escalation
                if status == SLAStatus.BREACHED and violation.severity != "breach":
                    violation.severity = "breach"
                    await self._escalate_violation(sla, violation)
        
        elif status == SLAStatus.MEETING:
            # Close any active violation
            if violation_key in self.active_violations:
                violation = self.active_violations[violation_key]
                violation.end_time = datetime.datetime.utcnow()
                violation.duration_seconds = (violation.end_time - violation.start_time).total_seconds()
                
                # Move to history
                self.violation_history.append(violation)
                del self.active_violations[violation_key]
                
                # Send recovery alert
                await self._send_recovery_alert(sla, violation)
                
                # Log recovery
                self.audit_logger.log_event(
                    event_type='sla_violation_resolved',
                    user='system',
                    details={
                        'sla_id': sla_id,
                        'violation_id': violation.violation_id,
                        'duration_seconds': violation.duration_seconds
                    }
                )
        
        # Update observability metrics
        self.observability.record_metric(
            f"sla.{sla.type.value}.current",
            current_value,
            tags={
                'sla_id': sla_id,
                'service': target.service,
                'status': status.value
            }
        )
    
    def _determine_compliance_status(self, sla: SLADefinition, current_value: float) -> SLAStatus:
        """Determine SLA compliance status"""
        # Different comparison based on SLA type
        if sla.type in [SLAType.AVAILABILITY, SLAType.THROUGHPUT]:
            # Higher is better
            if current_value >= sla.target_value:
                return SLAStatus.MEETING
            elif current_value >= sla.target_value * sla.critical_threshold:
                return SLAStatus.AT_RISK
            else:
                return SLAStatus.BREACHED
        else:
            # Lower is better (response time, error rate)
            if current_value <= sla.target_value:
                return SLAStatus.MEETING
            elif current_value <= sla.target_value / sla.warning_threshold:
                return SLAStatus.AT_RISK
            else:
                return SLAStatus.BREACHED
    
    async def _send_sla_alert(self, sla: SLADefinition, violation: SLAViolation, status: SLAStatus):
        """Send SLA violation alert"""
        message = (
            f"SLA Violation: {sla.name}\n"
            f"Status: {status.value}\n"
            f"Current: {violation.current_value:.2f}\n"
            f"Target: {violation.target_value:.2f}\n"
            f"Service: {violation.target.service}"
        )
        
        if violation.target.endpoint:
            message += f"\nEndpoint: {violation.target.endpoint}"
        if violation.target.customer:
            message += f"\nCustomer: {violation.target.customer}"
        
        self.alert_manager.send_alert(
            message=message,
            data={
                'sla_id': sla.sla_id,
                'violation_id': violation.violation_id,
                'status': status.value,
                'current_value': violation.current_value,
                'target_value': violation.target_value
            },
            level='critical' if status == SLAStatus.BREACHED else 'warning'
        )
    
    async def _send_recovery_alert(self, sla: SLADefinition, violation: SLAViolation):
        """Send SLA recovery alert"""
        message = (
            f"SLA Recovered: {sla.name}\n"
            f"Duration: {violation.duration_seconds:.0f} seconds\n"
            f"Service: {violation.target.service}"
        )
        
        self.alert_manager.send_alert(
            message=message,
            data={
                'sla_id': sla.sla_id,
                'violation_id': violation.violation_id,
                'duration_seconds': violation.duration_seconds
            },
            level='info'
        )
    
    async def _escalate_violation(self, sla: SLADefinition, violation: SLAViolation):
        """Escalate SLA violation"""
        # Find applicable escalation policy
        for policy in self.escalation_policies.values():
            if sla.sla_id in policy.sla_ids and policy.enabled:
                await self._apply_escalation_policy(policy, sla, violation)
    
    async def _apply_escalation_policy(self, policy: EscalationPolicy,
                                     sla: SLADefinition, violation: SLAViolation):
        """Apply escalation policy to violation"""
        # Determine escalation level based on duration
        duration = (datetime.datetime.utcnow() - violation.start_time).total_seconds()
        
        escalation_level = None
        for rule in policy.escalation_rules:
            if duration >= rule.get('after_seconds', 0):
                escalation_level = EscalationLevel(rule['level'])
        
        if escalation_level and escalation_level != violation.escalation_level:
            violation.escalation_level = escalation_level
            
            # Send escalation notifications
            channels = policy.notification_channels.get(escalation_level, [])
            
            message = (
                f"ESCALATION - {escalation_level.value}\n"
                f"SLA: {sla.name}\n"
                f"Violation Duration: {duration:.0f} seconds\n"
                f"Current Value: {violation.current_value:.2f}\n"
                f"Target: {violation.target_value:.2f}"
            )
            
            self.alert_manager.send_alert(
                message=message,
                data={
                    'escalation_level': escalation_level.value,
                    'sla_id': sla.sla_id,
                    'violation_id': violation.violation_id
                },
                channels=channels,
                level='critical'
            )
    
    def create_escalation_policy(self, name: str, sla_ids: List[str],
                               escalation_rules: List[Dict[str, Any]],
                               notification_channels: Dict[str, List[str]]) -> str:
        """Create an escalation policy.
        
        Args:
            name: Policy name
            sla_ids: SLAs this policy applies to
            escalation_rules: List of escalation rules
            notification_channels: Notification channels by level
            
        Returns:
            Policy ID
        """
        policy_id = f"policy_{datetime.datetime.utcnow().timestamp()}"
        
        # Convert string keys to EscalationLevel
        channels = {}
        for level_str, channel_list in notification_channels.items():
            try:
                level = EscalationLevel(level_str)
                channels[level] = channel_list
            except ValueError:
                logger.warning(f"Invalid escalation level: {level_str}")
        
        policy = EscalationPolicy(
            policy_id=policy_id,
            name=name,
            sla_ids=sla_ids,
            escalation_rules=escalation_rules,
            notification_channels=channels
        )
        
        self.escalation_policies[policy_id] = policy
        self._save_sla_data()
        
        return policy_id
    
    def acknowledge_violation(self, violation_id: str, acknowledged_by: str,
                            resolution_notes: Optional[str] = None) -> bool:
        """Acknowledge an SLA violation.
        
        Args:
            violation_id: Violation ID
            acknowledged_by: User acknowledging
            resolution_notes: Optional notes
            
        Returns:
            True if acknowledged successfully
        """
        # Find violation in active violations
        for violation in self.active_violations.values():
            if violation.violation_id == violation_id:
                violation.acknowledged = True
                violation.acknowledged_by = acknowledged_by
                violation.resolution_notes = resolution_notes
                
                # Log acknowledgment
                self.audit_logger.log_event(
                    event_type='sla_violation_acknowledged',
                    user=acknowledged_by,
                    details={
                        'violation_id': violation_id,
                        'notes': resolution_notes
                    }
                )
                
                return True
        
        # Check historical violations
        for violation in self.violation_history:
            if violation.violation_id == violation_id:
                violation.acknowledged = True
                violation.acknowledged_by = acknowledged_by
                violation.resolution_notes = resolution_notes
                return True
        
        return False
    
    def get_sla_status(self, sla_id: Optional[str] = None) -> Dict[str, Any]:
        """Get current SLA status.
        
        Args:
            sla_id: Optional specific SLA ID
            
        Returns:
            SLA status information
        """
        if sla_id:
            sla = self.sla_definitions.get(sla_id)
            if not sla:
                return {}
            
            return self._get_single_sla_status(sla_id, sla)
        else:
            # All SLAs
            status = {}
            for sla_id, sla in self.sla_definitions.items():
                status[sla_id] = self._get_single_sla_status(sla_id, sla)
            return status
    
    def _get_single_sla_status(self, sla_id: str, sla: SLADefinition) -> Dict[str, Any]:
        """Get status for a single SLA"""
        targets_status = []
        
        for target in self.sla_targets.get(sla_id, []):
            key = self._get_metric_key(sla_id, target)
            
            # Get recent metrics
            recent_metrics = list(self.metrics_buffer[key])[-100:]
            
            if recent_metrics:
                current_value = recent_metrics[-1].value
                status = self._determine_compliance_status(sla, current_value)
            else:
                current_value = None
                status = SLAStatus.UNKNOWN
            
            # Check for active violation
            violation_key = f"{sla_id}_{self._get_target_key(target)}"
            active_violation = self.active_violations.get(violation_key)
            
            targets_status.append({
                'target': target.__dict__,
                'current_value': current_value,
                'status': status.value,
                'active_violation': {
                    'violation_id': active_violation.violation_id,
                    'start_time': active_violation.start_time.isoformat(),
                    'duration_seconds': (datetime.datetime.utcnow() - active_violation.start_time).total_seconds(),
                    'severity': active_violation.severity,
                    'acknowledged': active_violation.acknowledged
                } if active_violation else None
            })
        
        return {
            'sla': {
                'name': sla.name,
                'type': sla.type.value,
                'target_value': sla.target_value,
                'enabled': sla.enabled
            },
            'targets': targets_status
        }
    
    def generate_sla_report(self, period_hours: int = 24,
                          sla_ids: Optional[List[str]] = None) -> SLAReport:
        """Generate SLA compliance report.
        
        Args:
            period_hours: Period to cover
            sla_ids: Optional specific SLA IDs
            
        Returns:
            SLA report
        """
        period_start = datetime.datetime.utcnow() - datetime.timedelta(hours=period_hours)
        period_end = datetime.datetime.utcnow()
        
        if not sla_ids:
            sla_ids = list(self.sla_definitions.keys())
        
        sla_summaries = {}
        all_violations = []
        
        for sla_id in sla_ids:
            sla = self.sla_definitions.get(sla_id)
            if not sla:
                continue
            
            # Calculate compliance for each target
            target_summaries = []
            
            for target in self.sla_targets.get(sla_id, []):
                summary = self._calculate_target_compliance(
                    sla_id, sla, target, period_start, period_end
                )
                target_summaries.append(summary)
            
            # Aggregate target summaries
            if target_summaries:
                sla_summaries[sla_id] = {
                    'name': sla.name,
                    'type': sla.type.value,
                    'target_value': sla.target_value,
                    'overall_compliance': statistics.mean([s['compliance_percentage'] for s in target_summaries]),
                    'targets': target_summaries
                }
        
        # Get violations in period
        for violation in self.violation_history:
            if violation.start_time >= period_start:
                all_violations.append(violation)
        
        # Add active violations
        for violation in self.active_violations.values():
            if violation.start_time >= period_start:
                all_violations.append(violation)
        
        # Calculate overall compliance
        if sla_summaries:
            overall_compliance = statistics.mean([
                s['overall_compliance'] for s in sla_summaries.values()
            ])
        else:
            overall_compliance = 100.0
        
        # Generate recommendations
        recommendations = self._generate_recommendations(sla_summaries, all_violations)
        
        report = SLAReport(
            report_id=f"report_{datetime.datetime.utcnow().timestamp()}",
            period_start=period_start,
            period_end=period_end,
            sla_summaries=sla_summaries,
            overall_compliance=overall_compliance,
            violations=all_violations,
            recommendations=recommendations,
            generated_at=datetime.datetime.utcnow()
        )
        
        return report
    
    def _calculate_target_compliance(self, sla_id: str, sla: SLADefinition,
                                   target: SLATarget, period_start: datetime.datetime,
                                   period_end: datetime.datetime) -> Dict[str, Any]:
        """Calculate compliance for a specific target"""
        key = self._get_metric_key(sla_id, target)
        
        # Get metrics in period
        period_metrics = [
            m for m in self.metrics_buffer[key]
            if period_start <= m.timestamp <= period_end
        ]
        
        if not period_metrics:
            return {
                'target': target.__dict__,
                'compliance_percentage': 0,
                'sample_count': 0,
                'meeting_count': 0,
                'violation_count': 0
            }
        
        # Count compliant samples
        meeting_count = 0
        violation_count = 0
        
        for metric in period_metrics:
            status = self._determine_compliance_status(sla, metric.value)
            if status == SLAStatus.MEETING:
                meeting_count += 1
            elif status == SLAStatus.BREACHED:
                violation_count += 1
        
        compliance_percentage = (meeting_count / len(period_metrics)) * 100
        
        return {
            'target': target.__dict__,
            'compliance_percentage': compliance_percentage,
            'sample_count': len(period_metrics),
            'meeting_count': meeting_count,
            'violation_count': violation_count,
            'average_value': statistics.mean([m.value for m in period_metrics])
        }
    
    def _generate_recommendations(self, sla_summaries: Dict[str, Dict[str, Any]],
                                violations: List[SLAViolation]) -> List[str]:
        """Generate recommendations based on SLA performance"""
        recommendations = []
        
        # Check for SLAs with low compliance
        for sla_id, summary in sla_summaries.items():
            if summary['overall_compliance'] < 95:
                recommendations.append(
                    f"SLA '{summary['name']}' has {summary['overall_compliance']:.1f}% compliance. "
                    f"Consider reviewing targets or improving service performance."
                )
        
        # Check for repeated violations
        violation_counts = defaultdict(int)
        for violation in violations:
            violation_counts[violation.sla_id] += 1
        
        for sla_id, count in violation_counts.items():
            if count > 5:
                sla = self.sla_definitions.get(sla_id)
                if sla:
                    recommendations.append(
                        f"SLA '{sla.name}' had {count} violations. "
                        f"Consider root cause analysis and corrective actions."
                    )
        
        # Check for long duration violations
        long_violations = [v for v in violations if v.duration_seconds and v.duration_seconds > 3600]
        if long_violations:
            recommendations.append(
                f"{len(long_violations)} violations lasted over 1 hour. "
                f"Review incident response procedures."
            )
        
        return recommendations
    
    def register_metric_collector(self, metric_type: str, collector: Callable):
        """Register a custom metric collector.
        
        Args:
            metric_type: Type of metric
            collector: Async function to collect metric
        """
        self.metric_collectors[metric_type] = collector
    
    # Default metric collectors
    async def _collect_availability_metric(self, target: SLATarget, sla: SLADefinition) -> float:
        """Collect availability metric"""
        # This would integrate with actual monitoring
        # For now, return sample data
        import random
        return 99.5 + random.uniform(-0.5, 0.5)
    
    async def _collect_response_time_metric(self, target: SLATarget, sla: SLADefinition) -> float:
        """Collect response time metric"""
        # This would integrate with actual monitoring
        import random
        return 50 + random.uniform(-10, 20)
    
    async def _collect_error_rate_metric(self, target: SLATarget, sla: SLADefinition) -> float:
        """Collect error rate metric"""
        # This would integrate with actual monitoring
        import random
        return random.uniform(0, 0.02)
    
    async def _collect_throughput_metric(self, target: SLATarget, sla: SLADefinition) -> float:
        """Collect throughput metric"""
        # This would integrate with actual monitoring
        import random
        return 1000 + random.uniform(-100, 100)
    
    def _get_metric_key(self, sla_id: str, target: SLATarget) -> str:
        """Generate unique key for metric storage"""
        return f"{sla_id}_{self._get_target_key(target)}"
    
    def _get_target_key(self, target: SLATarget) -> str:
        """Generate unique key for target"""
        parts = [target.service]
        if target.endpoint:
            parts.append(target.endpoint)
        if target.customer:
            parts.append(target.customer)
        if target.tier:
            parts.append(target.tier)
        return "_".join(parts)
    
    def _calculate_percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile value"""
        if not values:
            return 0
        
        sorted_values = sorted(values)
        index = int(len(sorted_values) * (percentile / 100))
        
        if index >= len(sorted_values):
            return sorted_values[-1]
        
        return sorted_values[index]
    
    def _is_business_hours(self) -> bool:
        """Check if current time is within business hours"""
        # This would use proper timezone handling
        # For now, simplified check
        now = datetime.datetime.utcnow()
        
        # Check day of week
        if now.strftime('%A') not in self.business_hours['days']:
            return False
        
        # Check time
        current_time = now.time()
        start_time = datetime.datetime.strptime(self.business_hours['start'], '%H:%M').time()
        end_time = datetime.datetime.strptime(self.business_hours['end'], '%H:%M').time()
        
        return start_time <= current_time <= end_time
    
    def _is_excluded_period(self, sla: SLADefinition) -> bool:
        """Check if current time is in an exclusion period"""
        now = datetime.datetime.utcnow()
        
        for exclusion in sla.exclusion_periods:
            start = datetime.datetime.fromisoformat(exclusion['start'])
            end = datetime.datetime.fromisoformat(exclusion['end'])
            
            if start <= now <= end:
                return True
        
        return False
    
    def _load_sla_data(self):
        """Load SLA data from storage"""
        # This would load persisted SLA definitions
        pass
    
    def _save_sla_data(self):
        """Save SLA data to storage"""
        # This would persist SLA definitions
        pass
    
    async def shutdown(self):
        """Shutdown monitoring tasks"""
        self.monitoring_enabled = False
        
        # Cancel all monitoring tasks
        for task in self.monitoring_tasks.values():
            task.cancel()
        
        # Wait for tasks to complete
        if self.monitoring_tasks:
            await asyncio.gather(*self.monitoring_tasks.values(), return_exceptions=True)


# Factory function
def create_sla_monitoring(config: Dict[str, Any]) -> SLAMonitoringSystem:
    """Create SLA monitoring system"""
    return SLAMonitoringSystem(config)