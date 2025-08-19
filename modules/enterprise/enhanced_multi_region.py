"""
Enhanced Multi-Region Failover Support

Extends the existing multi-region capabilities with advanced failover strategies,
automated health checks, data consistency guarantees, and seamless traffic management.
"""

import asyncio
import datetime
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from collections import defaultdict
import geopy.distance

from modules.deployment.multi_environment.multi_region import (
    MultiRegionResilienceStrategy, Region, RegionStatus, FailoverStrategy,
    ConsistencyModel, RegionHealth, FailoverEvent
)
from modules.monitoring.alert_system import AlertManager
from modules.security.audit import get_audit_logger
from modules.monitoring.observability_hooks import ObservabilityHooks

logger = logging.getLogger(__name__)


class TrafficDistributionStrategy(Enum):
    """Traffic distribution strategies"""
    WEIGHTED = "weighted"
    GEOGRAPHIC = "geographic"
    LATENCY_BASED = "latency_based"
    LEAST_CONNECTIONS = "least_connections"
    ROUND_ROBIN = "round_robin"
    CANARY = "canary"


class HealthCheckType(Enum):
    """Types of health checks"""
    HTTP = "http"
    TCP = "tcp"
    GRPC = "grpc"
    CUSTOM = "custom"


class DataSyncStrategy(Enum):
    """Data synchronization strategies"""
    SYNC_REPLICATION = "sync_replication"
    ASYNC_REPLICATION = "async_replication"
    EVENTUAL_CONSISTENCY = "eventual_consistency"
    CONFLICT_FREE = "conflict_free"  # CRDT-based


@dataclass
class EnhancedRegion(Region):
    """Enhanced region with additional capabilities"""
    weight: float = 1.0  # Traffic weight
    priority: int = 1  # Failover priority (lower = higher priority)
    data_sync_strategy: DataSyncStrategy = DataSyncStrategy.ASYNC_REPLICATION
    health_check_config: Dict[str, Any] = field(default_factory=dict)
    traffic_metrics: Dict[str, float] = field(default_factory=dict)
    sync_lag_seconds: float = 0.0
    last_sync_time: Optional[datetime.datetime] = None


@dataclass
class HealthCheckConfig:
    """Health check configuration"""
    check_type: HealthCheckType
    endpoint: str
    interval_seconds: int = 30
    timeout_seconds: int = 10
    healthy_threshold: int = 2
    unhealthy_threshold: int = 3
    custom_script: Optional[str] = None
    expected_response: Optional[Dict[str, Any]] = None


@dataclass
class TrafficPolicy:
    """Traffic distribution policy"""
    policy_id: str
    name: str
    strategy: TrafficDistributionStrategy
    region_weights: Dict[str, float] = field(default_factory=dict)
    geographic_rules: List[Dict[str, Any]] = field(default_factory=list)
    canary_config: Optional[Dict[str, Any]] = None
    sticky_sessions: bool = False
    session_affinity_ttl: int = 3600


@dataclass
class FailoverDecision:
    """Failover decision record"""
    decision_id: str
    timestamp: datetime.datetime
    from_region: str
    to_regions: List[str]
    reason: str
    confidence_score: float
    automatic: bool
    approved_by: Optional[str] = None
    rollback_plan: Optional[Dict[str, Any]] = None


@dataclass
class DataConsistencyCheck:
    """Data consistency check result"""
    check_id: str
    timestamp: datetime.datetime
    regions: List[str]
    consistent: bool
    lag_seconds: Dict[str, float]
    conflicts: List[Dict[str, Any]]
    resolution_actions: List[str]


class EnhancedMultiRegionFailover:
    """
    Enhanced multi-region failover system with advanced capabilities
    for enterprise environments.
    """
    
    def __init__(self, config: Dict[str, Any], base_strategy: MultiRegionResilienceStrategy):
        """Initialize enhanced multi-region failover.
        
        Args:
            config: Configuration dictionary
            base_strategy: Base multi-region strategy
        """
        self.config = config
        self.base_strategy = base_strategy
        
        # Enhanced regions
        self.enhanced_regions: Dict[str, EnhancedRegion] = {}
        self._enhance_regions()
        
        # Managers
        self.alert_manager = AlertManager(config.get('alert_config', {}))
        self.audit_logger = get_audit_logger()
        self.observability = ObservabilityHooks(config)
        
        # Traffic management
        self.traffic_policies: Dict[str, TrafficPolicy] = {}
        self.active_policy_id: Optional[str] = None
        
        # Health checking
        self.health_check_configs: Dict[str, HealthCheckConfig] = {}
        self.health_check_results: Dict[str, List[Tuple[datetime.datetime, bool]]] = defaultdict(list)
        self.health_check_tasks: Dict[str, asyncio.Task] = {}
        
        # Failover management
        self.failover_decisions: List[FailoverDecision] = []
        self.pending_failovers: Dict[str, FailoverDecision] = {}
        self.failover_in_progress = False
        
        # Data consistency
        self.consistency_checks: List[DataConsistencyCheck] = []
        self.sync_monitors: Dict[str, asyncio.Task] = {}
        
        # Configuration
        self.auto_failover_enabled = config.get('auto_failover_enabled', True)
        self.min_healthy_regions = config.get('min_healthy_regions', 2)
        self.failover_cooldown_seconds = config.get('failover_cooldown_seconds', 300)
        self.consistency_check_interval = config.get('consistency_check_interval', 60)
        
        # Initialize components
        self._initialize_default_policies()
        self._start_health_checks()
        
        logger.info("Initialized enhanced multi-region failover")
    
    def _enhance_regions(self):
        """Enhance base regions with additional capabilities"""
        for region in self.base_strategy.regions:
            enhanced = EnhancedRegion(
                id=region.id,
                name=region.name,
                location=region.location,
                environments=region.environments,
                status=region.status,
                capacity=region.capacity,
                metadata=region.metadata,
                weight=self.config.get('region_weights', {}).get(region.id, 1.0),
                priority=self.config.get('region_priorities', {}).get(region.id, 1),
                data_sync_strategy=DataSyncStrategy(
                    self.config.get('data_sync_strategies', {}).get(
                        region.id, 'async_replication'
                    )
                )
            )
            self.enhanced_regions[region.id] = enhanced
    
    def _initialize_default_policies(self):
        """Initialize default traffic policies"""
        # Geographic policy
        geo_policy = TrafficPolicy(
            policy_id='geo_default',
            name='Geographic Routing',
            strategy=TrafficDistributionStrategy.GEOGRAPHIC,
            geographic_rules=[
                {
                    'source_continent': 'NA',
                    'target_regions': ['us-east-1', 'us-west-2']
                },
                {
                    'source_continent': 'EU',
                    'target_regions': ['eu-west-1', 'eu-central-1']
                },
                {
                    'source_continent': 'AS',
                    'target_regions': ['ap-southeast-1', 'ap-northeast-1']
                }
            ]
        )
        self.traffic_policies[geo_policy.policy_id] = geo_policy
        
        # Weighted policy
        weights = {region_id: region.weight for region_id, region in self.enhanced_regions.items()}
        weighted_policy = TrafficPolicy(
            policy_id='weighted_default',
            name='Weighted Distribution',
            strategy=TrafficDistributionStrategy.WEIGHTED,
            region_weights=weights
        )
        self.traffic_policies[weighted_policy.policy_id] = weighted_policy
        
        # Set default active policy
        self.active_policy_id = 'geo_default'
    
    def _start_health_checks(self):
        """Start health checks for all regions"""
        for region_id, region in self.enhanced_regions.items():
            # Create default health check config if not exists
            if region_id not in self.health_check_configs:
                self.health_check_configs[region_id] = HealthCheckConfig(
                    check_type=HealthCheckType.HTTP,
                    endpoint=f"https://{region.name}.homeostasis.io/health",
                    interval_seconds=30,
                    timeout_seconds=10,
                    healthy_threshold=2,
                    unhealthy_threshold=3
                )
            
            # Start health check task
            self._start_region_health_check(region_id)
    
    def _start_region_health_check(self, region_id: str):
        """Start health check for a specific region"""
        async def health_check_loop():
            config = self.health_check_configs[region_id]
            consecutive_failures = 0
            consecutive_successes = 0
            
            while region_id in self.enhanced_regions:
                try:
                    # Perform health check
                    healthy = await self._perform_health_check(region_id, config)
                    
                    # Record result
                    self.health_check_results[region_id].append(
                        (datetime.datetime.utcnow(), healthy)
                    )
                    
                    # Limit history
                    if len(self.health_check_results[region_id]) > 1000:
                        self.health_check_results[region_id] = self.health_check_results[region_id][-1000:]
                    
                    # Update consecutive counts
                    if healthy:
                        consecutive_successes += 1
                        consecutive_failures = 0
                    else:
                        consecutive_failures += 1
                        consecutive_successes = 0
                    
                    # Determine region health status
                    region = self.enhanced_regions[region_id]
                    previous_status = region.status
                    
                    if consecutive_failures >= config.unhealthy_threshold:
                        region.status = RegionStatus.UNHEALTHY
                    elif consecutive_successes >= config.healthy_threshold:
                        region.status = RegionStatus.HEALTHY
                    
                    # Handle status change
                    if region.status != previous_status:
                        await self._handle_region_status_change(
                            region_id, previous_status, region.status
                        )
                    
                    # Record metrics
                    self.observability.record_metric(
                        f"region.health.{region_id}",
                        1 if healthy else 0,
                        tags={'region': region_id, 'status': region.status.value}
                    )
                    
                    # Wait for next check
                    await asyncio.sleep(config.interval_seconds)
                    
                except Exception as e:
                    logger.error(f"Health check error for region {region_id}: {e}")
                    await asyncio.sleep(config.interval_seconds)
        
        task = asyncio.create_task(health_check_loop())
        self.health_check_tasks[region_id] = task
    
    async def _perform_health_check(self, region_id: str, config: HealthCheckConfig) -> bool:
        """Perform actual health check"""
        if config.check_type == HealthCheckType.HTTP:
            return await self._http_health_check(region_id, config)
        elif config.check_type == HealthCheckType.TCP:
            return await self._tcp_health_check(region_id, config)
        elif config.check_type == HealthCheckType.GRPC:
            return await self._grpc_health_check(region_id, config)
        elif config.check_type == HealthCheckType.CUSTOM:
            return await self._custom_health_check(region_id, config)
        else:
            return False
    
    async def _http_health_check(self, region_id: str, config: HealthCheckConfig) -> bool:
        """Perform HTTP health check"""
        # This would make actual HTTP request
        # For now, simulate based on region status
        region = self.enhanced_regions[region_id]
        return region.status in [RegionStatus.HEALTHY, RegionStatus.DEGRADED]
    
    async def _tcp_health_check(self, region_id: str, config: HealthCheckConfig) -> bool:
        """Perform TCP health check"""
        # This would make actual TCP connection
        return True
    
    async def _grpc_health_check(self, region_id: str, config: HealthCheckConfig) -> bool:
        """Perform gRPC health check"""
        # This would make actual gRPC health check
        return True
    
    async def _custom_health_check(self, region_id: str, config: HealthCheckConfig) -> bool:
        """Perform custom health check"""
        # This would execute custom script
        return True
    
    async def _handle_region_status_change(self, region_id: str,
                                         previous_status: RegionStatus,
                                         new_status: RegionStatus):
        """Handle region status change"""
        logger.info(f"Region {region_id} status changed: {previous_status.value} -> {new_status.value}")
        
        # Log status change
        self.audit_logger.log_event(
            event_type='region_status_change',
            user='system',
            details={
                'region_id': region_id,
                'previous_status': previous_status.value,
                'new_status': new_status.value
            }
        )
        
        # Send alert
        if new_status == RegionStatus.UNHEALTHY:
            self.alert_manager.send_alert(
                f"Region {region_id} is unhealthy",
                {
                    'region_id': region_id,
                    'status': new_status.value
                },
                level='critical'
            )
        
        # Check if failover is needed
        if new_status == RegionStatus.UNHEALTHY and self.auto_failover_enabled:
            await self._evaluate_failover_need(region_id)
    
    async def _evaluate_failover_need(self, failed_region_id: str):
        """Evaluate if failover is needed"""
        # Check if already in failover
        if self.failover_in_progress:
            logger.info("Failover already in progress, skipping evaluation")
            return
        
        # Check cooldown
        if self._in_failover_cooldown():
            logger.info("In failover cooldown period")
            return
        
        # Count healthy regions
        healthy_regions = [
            r for r in self.enhanced_regions.values()
            if r.status == RegionStatus.HEALTHY and r.id != failed_region_id
        ]
        
        if len(healthy_regions) < self.min_healthy_regions:
            logger.error(f"Only {len(healthy_regions)} healthy regions available, "
                        f"minimum required: {self.min_healthy_regions}")
            return
        
        # Create failover decision
        decision = await self._create_failover_decision(failed_region_id, healthy_regions)
        
        # Execute failover if confidence is high
        if decision.confidence_score >= 0.8:
            await self.execute_enhanced_failover(decision)
        else:
            # Queue for manual approval
            self.pending_failovers[decision.decision_id] = decision
            self.alert_manager.send_alert(
                f"Failover pending approval for region {failed_region_id}",
                {
                    'decision_id': decision.decision_id,
                    'confidence_score': decision.confidence_score
                },
                level='warning'
            )
    
    async def _create_failover_decision(self, failed_region_id: str,
                                      healthy_regions: List[EnhancedRegion]) -> FailoverDecision:
        """Create failover decision"""
        # Select target regions based on priority and capacity
        target_regions = sorted(
            healthy_regions,
            key=lambda r: (r.priority, -self._calculate_available_capacity(r))
        )[:2]  # Select top 2 regions
        
        # Calculate confidence score
        confidence_score = self._calculate_failover_confidence(
            failed_region_id, [r.id for r in target_regions]
        )
        
        # Create rollback plan
        rollback_plan = {
            'original_region': failed_region_id,
            'original_weights': {r.id: r.weight for r in self.enhanced_regions.values()},
            'original_policy': self.active_policy_id
        }
        
        decision = FailoverDecision(
            decision_id=f"decision_{datetime.datetime.utcnow().timestamp()}",
            timestamp=datetime.datetime.utcnow(),
            from_region=failed_region_id,
            to_regions=[r.id for r in target_regions],
            reason=f"Region {failed_region_id} health check failures",
            confidence_score=confidence_score,
            automatic=True,
            rollback_plan=rollback_plan
        )
        
        self.failover_decisions.append(decision)
        
        return decision
    
    def _calculate_available_capacity(self, region: EnhancedRegion) -> float:
        """Calculate available capacity in a region"""
        used_capacity = sum(region.capacity.values()) / len(region.capacity)
        return 100 - used_capacity
    
    def _calculate_failover_confidence(self, failed_region: str, target_regions: List[str]) -> float:
        """Calculate confidence score for failover decision"""
        confidence = 1.0
        
        # Check target region health history
        for region_id in target_regions:
            recent_checks = self.health_check_results[region_id][-10:]
            if recent_checks:
                success_rate = sum(1 for _, healthy in recent_checks if healthy) / len(recent_checks)
                confidence *= success_rate
        
        # Check data sync status
        for region_id in target_regions:
            region = self.enhanced_regions[region_id]
            if region.sync_lag_seconds > 60:  # More than 1 minute lag
                confidence *= 0.8
        
        # Check recent failover history
        recent_failovers = [
            f for f in self.failover_decisions
            if (datetime.datetime.utcnow() - f.timestamp).total_seconds() < 3600
        ]
        if len(recent_failovers) > 2:
            confidence *= 0.7  # Too many recent failovers
        
        return confidence
    
    def _in_failover_cooldown(self) -> bool:
        """Check if in failover cooldown period"""
        if not self.failover_decisions:
            return False
        
        last_failover = self.failover_decisions[-1]
        elapsed = (datetime.datetime.utcnow() - last_failover.timestamp).total_seconds()
        
        return elapsed < self.failover_cooldown_seconds
    
    async def execute_enhanced_failover(self, decision: FailoverDecision):
        """Execute enhanced failover with traffic management"""
        self.failover_in_progress = True
        
        try:
            logger.info(f"Executing failover from {decision.from_region} to {decision.to_regions}")
            
            # Phase 1: Prepare target regions
            await self._prepare_target_regions(decision.to_regions)
            
            # Phase 2: Update traffic distribution
            await self._update_traffic_distribution(decision)
            
            # Phase 3: Drain connections from failed region
            await self._drain_region_connections(decision.from_region)
            
            # Phase 4: Verify data consistency
            consistency_ok = await self._verify_data_consistency(decision.to_regions)
            if not consistency_ok:
                logger.warning("Data consistency check failed, but proceeding with failover")
            
            # Phase 5: Execute base failover
            for target_region in decision.to_regions:
                event = await self.base_strategy.failover_orchestrators[
                    list(self.base_strategy.policies.keys())[0]
                ].execute_failover(
                    decision.from_region,
                    target_region,
                    decision.reason,
                    []  # All services
                )
            
            # Phase 6: Verify failover success
            success = await self._verify_failover_success(decision)
            
            if success:
                logger.info("Failover completed successfully")
                
                # Update metrics
                self.observability.record_metric(
                    "failover.success",
                    1,
                    tags={
                        'from_region': decision.from_region,
                        'to_regions': ','.join(decision.to_regions)
                    }
                )
            else:
                logger.error("Failover verification failed")
                await self._rollback_failover(decision)
            
        except Exception as e:
            logger.error(f"Failover failed: {e}")
            await self._rollback_failover(decision)
            raise
        finally:
            self.failover_in_progress = False
    
    async def _prepare_target_regions(self, region_ids: List[str]):
        """Prepare target regions for failover"""
        for region_id in region_ids:
            region = self.enhanced_regions[region_id]
            
            # Pre-warm caches
            logger.info(f"Pre-warming caches in region {region_id}")
            
            # Scale up resources if needed
            if sum(region.capacity.values()) / len(region.capacity) > 70:
                logger.info(f"Scaling up resources in region {region_id}")
                # This would trigger actual scaling
    
    async def _update_traffic_distribution(self, decision: FailoverDecision):
        """Update traffic distribution for failover"""
        # Create temporary traffic policy
        temp_policy = TrafficPolicy(
            policy_id=f"failover_{decision.decision_id}",
            name="Failover Traffic Policy",
            strategy=TrafficDistributionStrategy.WEIGHTED,
            region_weights={}
        )
        
        # Calculate new weights
        failed_region = self.enhanced_regions[decision.from_region]
        failed_weight = failed_region.weight
        
        # Distribute failed region's weight to target regions
        weight_per_target = failed_weight / len(decision.to_regions)
        
        for region_id, region in self.enhanced_regions.items():
            if region_id == decision.from_region:
                temp_policy.region_weights[region_id] = 0
            elif region_id in decision.to_regions:
                temp_policy.region_weights[region_id] = region.weight + weight_per_target
            else:
                temp_policy.region_weights[region_id] = region.weight
        
        # Apply policy
        self.traffic_policies[temp_policy.policy_id] = temp_policy
        self.active_policy_id = temp_policy.policy_id
        
        logger.info(f"Updated traffic distribution: {temp_policy.region_weights}")
    
    async def _drain_region_connections(self, region_id: str):
        """Gracefully drain connections from a region"""
        logger.info(f"Draining connections from region {region_id}")
        
        # This would implement connection draining
        # For now, simulate with delay
        await asyncio.sleep(5)
    
    async def _verify_data_consistency(self, region_ids: List[str]) -> bool:
        """Verify data consistency across regions"""
        check = DataConsistencyCheck(
            check_id=f"check_{datetime.datetime.utcnow().timestamp()}",
            timestamp=datetime.datetime.utcnow(),
            regions=region_ids,
            consistent=True,
            lag_seconds={},
            conflicts=[],
            resolution_actions=[]
        )
        
        # Check sync lag for each region
        for region_id in region_ids:
            region = self.enhanced_regions[region_id]
            check.lag_seconds[region_id] = region.sync_lag_seconds
            
            if region.sync_lag_seconds > 30:  # More than 30 seconds
                check.consistent = False
                check.conflicts.append({
                    'region': region_id,
                    'issue': 'high_sync_lag',
                    'lag_seconds': region.sync_lag_seconds
                })
        
        self.consistency_checks.append(check)
        
        return check.consistent
    
    async def _verify_failover_success(self, decision: FailoverDecision) -> bool:
        """Verify failover completed successfully"""
        # Check target regions are healthy
        for region_id in decision.to_regions:
            region = self.enhanced_regions[region_id]
            if region.status != RegionStatus.HEALTHY:
                return False
        
        # Check traffic is flowing to target regions
        # This would check actual traffic metrics
        
        return True
    
    async def _rollback_failover(self, decision: FailoverDecision):
        """Rollback a failed failover"""
        logger.info(f"Rolling back failover {decision.decision_id}")
        
        if not decision.rollback_plan:
            logger.error("No rollback plan available")
            return
        
        # Restore original traffic weights
        original_weights = decision.rollback_plan['original_weights']
        for region_id, weight in original_weights.items():
            if region_id in self.enhanced_regions:
                self.enhanced_regions[region_id].weight = weight
        
        # Restore original policy
        self.active_policy_id = decision.rollback_plan['original_policy']
        
        # Alert about rollback
        self.alert_manager.send_alert(
            f"Failover rolled back for decision {decision.decision_id}",
            {
                'decision_id': decision.decision_id,
                'reason': 'verification_failed'
            },
            level='warning'
        )
    
    def approve_pending_failover(self, decision_id: str, approved_by: str) -> bool:
        """Approve a pending failover decision.
        
        Args:
            decision_id: Decision ID to approve
            approved_by: User approving the decision
            
        Returns:
            True if approved and executed
        """
        if decision_id not in self.pending_failovers:
            return False
        
        decision = self.pending_failovers[decision_id]
        decision.approved_by = approved_by
        decision.automatic = False
        
        # Execute failover
        asyncio.create_task(self.execute_enhanced_failover(decision))
        
        # Remove from pending
        del self.pending_failovers[decision_id]
        
        # Log approval
        self.audit_logger.log_event(
            event_type='failover_approved',
            user=approved_by,
            details={
                'decision_id': decision_id,
                'from_region': decision.from_region,
                'to_regions': decision.to_regions
            }
        )
        
        return True
    
    def create_traffic_policy(self, name: str, strategy: TrafficDistributionStrategy,
                            **kwargs) -> str:
        """Create a custom traffic policy.
        
        Args:
            name: Policy name
            strategy: Distribution strategy
            **kwargs: Strategy-specific parameters
            
        Returns:
            Policy ID
        """
        policy_id = f"policy_{datetime.datetime.utcnow().timestamp()}"
        
        policy = TrafficPolicy(
            policy_id=policy_id,
            name=name,
            strategy=strategy,
            region_weights=kwargs.get('region_weights', {}),
            geographic_rules=kwargs.get('geographic_rules', []),
            canary_config=kwargs.get('canary_config'),
            sticky_sessions=kwargs.get('sticky_sessions', False),
            session_affinity_ttl=kwargs.get('session_affinity_ttl', 3600)
        )
        
        self.traffic_policies[policy_id] = policy
        
        return policy_id
    
    def activate_traffic_policy(self, policy_id: str) -> bool:
        """Activate a traffic policy.
        
        Args:
            policy_id: Policy ID to activate
            
        Returns:
            True if activated successfully
        """
        if policy_id not in self.traffic_policies:
            return False
        
        self.active_policy_id = policy_id
        
        # Log activation
        self.audit_logger.log_event(
            event_type='traffic_policy_activated',
            user='system',
            details={
                'policy_id': policy_id,
                'policy_name': self.traffic_policies[policy_id].name
            }
        )
        
        return True
    
    async def perform_canary_deployment(self, region_id: str, canary_percentage: float,
                                      duration_minutes: int = 30) -> Dict[str, Any]:
        """Perform canary deployment to a region.
        
        Args:
            region_id: Target region for canary
            canary_percentage: Percentage of traffic for canary
            duration_minutes: Canary duration
            
        Returns:
            Canary deployment results
        """
        # Create canary policy
        canary_policy = TrafficPolicy(
            policy_id=f"canary_{datetime.datetime.utcnow().timestamp()}",
            name=f"Canary deployment to {region_id}",
            strategy=TrafficDistributionStrategy.CANARY,
            canary_config={
                'target_region': region_id,
                'percentage': canary_percentage,
                'start_time': datetime.datetime.utcnow().isoformat(),
                'duration_minutes': duration_minutes
            }
        )
        
        self.traffic_policies[canary_policy.policy_id] = canary_policy
        previous_policy = self.active_policy_id
        self.active_policy_id = canary_policy.policy_id
        
        # Monitor canary
        results = await self._monitor_canary_deployment(
            region_id, canary_percentage, duration_minutes
        )
        
        # Restore previous policy
        self.active_policy_id = previous_policy
        
        return results
    
    async def _monitor_canary_deployment(self, region_id: str, percentage: float,
                                       duration_minutes: int) -> Dict[str, Any]:
        """Monitor canary deployment"""
        start_time = datetime.datetime.utcnow()
        end_time = start_time + datetime.timedelta(minutes=duration_minutes)
        
        metrics = {
            'error_rates': [],
            'response_times': [],
            'success_count': 0,
            'total_count': 0
        }
        
        while datetime.datetime.utcnow() < end_time:
            # Collect metrics (this would get real metrics)
            region = self.enhanced_regions[region_id]
            
            metrics['error_rates'].append(region.traffic_metrics.get('error_rate', 0))
            metrics['response_times'].append(region.traffic_metrics.get('response_time', 0))
            metrics['total_count'] += 100  # Simulated
            metrics['success_count'] += 98  # Simulated
            
            # Check for anomalies
            if metrics['error_rates'][-1] > 0.05:  # 5% error rate
                logger.warning(f"High error rate in canary region {region_id}")
                break
            
            await asyncio.sleep(30)  # Check every 30 seconds
        
        # Calculate summary
        import statistics
        
        return {
            'region_id': region_id,
            'percentage': percentage,
            'duration_seconds': (datetime.datetime.utcnow() - start_time).total_seconds(),
            'average_error_rate': statistics.mean(metrics['error_rates']) if metrics['error_rates'] else 0,
            'average_response_time': statistics.mean(metrics['response_times']) if metrics['response_times'] else 0,
            'success_rate': metrics['success_count'] / metrics['total_count'] if metrics['total_count'] > 0 else 0,
            'recommendation': 'proceed' if metrics['error_rates'][-1] < 0.02 else 'rollback'
        }
    
    def get_failover_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive failover dashboard data"""
        # Region health summary
        region_health = {}
        for region_id, region in self.enhanced_regions.items():
            recent_checks = self.health_check_results[region_id][-10:]
            if recent_checks:
                success_rate = sum(1 for _, healthy in recent_checks if healthy) / len(recent_checks)
            else:
                success_rate = 0
            
            region_health[region_id] = {
                'status': region.status.value,
                'health_score': success_rate * 100,
                'weight': region.weight,
                'priority': region.priority,
                'sync_lag_seconds': region.sync_lag_seconds,
                'capacity_usage': sum(region.capacity.values()) / len(region.capacity) if region.capacity else 0
            }
        
        # Active traffic policy
        active_policy = self.traffic_policies.get(self.active_policy_id)
        
        # Recent failovers
        recent_failovers = sorted(
            self.failover_decisions,
            key=lambda x: x.timestamp,
            reverse=True
        )[:10]
        
        # Pending failovers
        pending_count = len(self.pending_failovers)
        
        # Data consistency
        recent_consistency = self.consistency_checks[-1] if self.consistency_checks else None
        
        return {
            'regions': region_health,
            'active_traffic_policy': {
                'policy_id': self.active_policy_id,
                'name': active_policy.name if active_policy else None,
                'strategy': active_policy.strategy.value if active_policy else None
            },
            'recent_failovers': [
                {
                    'decision_id': f.decision_id,
                    'timestamp': f.timestamp.isoformat(),
                    'from_region': f.from_region,
                    'to_regions': f.to_regions,
                    'automatic': f.automatic,
                    'confidence_score': f.confidence_score
                }
                for f in recent_failovers
            ],
            'pending_failovers': pending_count,
            'data_consistency': {
                'last_check': recent_consistency.timestamp.isoformat() if recent_consistency else None,
                'consistent': recent_consistency.consistent if recent_consistency else None,
                'max_lag_seconds': max(recent_consistency.lag_seconds.values()) if recent_consistency and recent_consistency.lag_seconds else 0
            },
            'auto_failover_enabled': self.auto_failover_enabled,
            'healthy_regions_count': sum(1 for r in self.enhanced_regions.values() if r.status == RegionStatus.HEALTHY),
            'total_regions_count': len(self.enhanced_regions)
        }
    
    async def shutdown(self):
        """Shutdown enhanced failover system"""
        # Cancel health check tasks
        for task in self.health_check_tasks.values():
            task.cancel()
        
        # Cancel sync monitor tasks
        for task in self.sync_monitors.values():
            task.cancel()
        
        # Wait for tasks to complete
        all_tasks = list(self.health_check_tasks.values()) + list(self.sync_monitors.values())
        if all_tasks:
            await asyncio.gather(*all_tasks, return_exceptions=True)


# Factory function
def create_enhanced_multi_region(config: Dict[str, Any],
                               base_strategy: MultiRegionResilienceStrategy) -> EnhancedMultiRegionFailover:
    """Create enhanced multi-region failover system"""
    return EnhancedMultiRegionFailover(config, base_strategy)