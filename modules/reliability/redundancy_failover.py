"""
Redundancy and Failover Coordination for High-Reliability Systems.

This module provides comprehensive redundancy management and failover coordination
to ensure system availability and reliability through multiple failure scenarios.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class RedundancyType(Enum):
    """Types of redundancy configurations."""

    ACTIVE_ACTIVE = "active_active"  # All instances handle traffic
    ACTIVE_PASSIVE = "active_passive"  # Standby instances ready
    N_PLUS_ONE = "n_plus_one"  # N active + 1 spare
    N_PLUS_M = "n_plus_m"  # N active + M spares
    GEOGRAPHIC = "geographic"  # Cross-region redundancy
    HIERARCHICAL = "hierarchical"  # Primary, secondary, tertiary


class FailoverStrategy(Enum):
    """Failover strategies."""

    IMMEDIATE = "immediate"  # Instant failover
    GRACEFUL = "graceful"  # Drain connections first
    STAGED = "staged"  # Gradual traffic shift
    MANUAL = "manual"  # Require human approval
    AUTOMATIC = "automatic"  # Fully automated


class InstanceState(Enum):
    """State of a redundant instance."""

    ACTIVE = "active"
    STANDBY = "standby"
    FAILING = "failing"
    FAILED = "failed"
    RECOVERING = "recovering"
    DRAINING = "draining"
    MAINTENANCE = "maintenance"


class HealthCheckType(Enum):
    """Types of health checks."""

    HTTP = "http"
    TCP = "tcp"
    GRPC = "grpc"
    DATABASE = "database"
    CUSTOM = "custom"


@dataclass
class HealthCheck:
    """Health check configuration."""

    check_type: HealthCheckType
    endpoint: str
    interval: timedelta
    timeout: timedelta
    healthy_threshold: int = 3
    unhealthy_threshold: int = 2
    expected_response: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "check_type": self.check_type.value,
            "endpoint": self.endpoint,
            "interval": self.interval.total_seconds(),
            "timeout": self.timeout.total_seconds(),
            "healthy_threshold": self.healthy_threshold,
            "unhealthy_threshold": self.unhealthy_threshold,
        }


@dataclass
class Instance:
    """Redundant instance representation."""

    instance_id: str
    host: str
    port: int
    state: InstanceState
    region: Optional[str] = None
    zone: Optional[str] = None
    capacity: float = 1.0  # Relative capacity
    current_load: float = 0.0
    health_checks: List[HealthCheck] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_health_check: Optional[datetime] = None
    consecutive_failures: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "instance_id": self.instance_id,
            "host": self.host,
            "port": self.port,
            "state": self.state.value,
            "region": self.region,
            "zone": self.zone,
            "capacity": self.capacity,
            "current_load": self.current_load,
            "health_checks": [hc.to_dict() for hc in self.health_checks],
            "last_health_check": (
                self.last_health_check.isoformat() if self.last_health_check else None
            ),
            "consecutive_failures": self.consecutive_failures,
        }


@dataclass
class RedundancyGroup:
    """Group of redundant instances."""

    group_id: str
    service_name: str
    redundancy_type: RedundancyType
    instances: List[Instance]
    min_active_instances: int
    max_active_instances: int
    failover_strategy: FailoverStrategy
    health_check_config: HealthCheck
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_active_instances(self) -> List[Instance]:
        """Get all active instances."""
        return [i for i in self.instances if i.state == InstanceState.ACTIVE]

    def get_standby_instances(self) -> List[Instance]:
        """Get all standby instances."""
        return [i for i in self.instances if i.state == InstanceState.STANDBY]

    def get_healthy_instances(self) -> List[Instance]:
        """Get all healthy instances (active or standby)."""
        return [
            i
            for i in self.instances
            if i.state in [InstanceState.ACTIVE, InstanceState.STANDBY]
        ]


@dataclass
class FailoverEvent:
    """Record of a failover event."""

    event_id: str
    timestamp: datetime
    group_id: str
    failed_instance: Instance
    replacement_instance: Optional[Instance]
    strategy_used: FailoverStrategy
    duration: timedelta
    success: bool
    error_message: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "group_id": self.group_id,
            "failed_instance": self.failed_instance.instance_id,
            "replacement_instance": (
                self.replacement_instance.instance_id
                if self.replacement_instance
                else None
            ),
            "strategy_used": self.strategy_used.value,
            "duration": self.duration.total_seconds(),
            "success": self.success,
            "error_message": self.error_message,
            "metrics": self.metrics,
        }


class HealthChecker(ABC):
    """Abstract base class for health checking."""

    @abstractmethod
    async def check(self, instance: Instance, health_check: HealthCheck) -> bool:
        """Perform health check on instance."""
        pass


class HTTPHealthChecker(HealthChecker):
    """HTTP-based health checker."""

    async def check(self, instance: Instance, health_check: HealthCheck) -> bool:
        """Check instance health via HTTP."""
        import aiohttp

        url = f"http://{instance.host}:{instance.port}{health_check.endpoint}"

        try:
            timeout = aiohttp.ClientTimeout(total=health_check.timeout.total_seconds())
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as response:
                    if health_check.expected_response:
                        return response.status == health_check.expected_response
                    return response.status == 200
        except Exception as e:
            logger.warning(f"Health check failed for {instance.instance_id}: {e}")
            return False


class TCPHealthChecker(HealthChecker):
    """TCP-based health checker."""

    async def check(self, instance: Instance, health_check: HealthCheck) -> bool:
        """Check instance health via TCP connection."""
        import socket

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(health_check.timeout.total_seconds())
            result = sock.connect_ex((instance.host, instance.port))
            sock.close()
            return result == 0
        except Exception as e:
            logger.warning(f"TCP health check failed for {instance.instance_id}: {e}")
            return False


class LoadBalancer(ABC):
    """Abstract base class for load balancing."""

    @abstractmethod
    def select_instance(self, instances: List[Instance]) -> Optional[Instance]:
        """Select an instance for handling a request."""
        pass

    @abstractmethod
    def update_load(self, instance: Instance, delta: float) -> None:
        """Update instance load after request."""
        pass


class RoundRobinBalancer(LoadBalancer):
    """Round-robin load balancer."""

    def __init__(self):
        self.current_index = 0

    def select_instance(self, instances: List[Instance]) -> Optional[Instance]:
        """Select next instance in round-robin fashion."""
        if not instances:
            return None

        active_instances = [i for i in instances if i.state == InstanceState.ACTIVE]
        if not active_instances:
            return None

        instance = active_instances[self.current_index % len(active_instances)]
        self.current_index += 1
        return instance

    def update_load(self, instance: Instance, delta: float) -> None:
        """Update instance load."""
        instance.current_load += delta


class LeastConnectionsBalancer(LoadBalancer):
    """Least connections load balancer."""

    def select_instance(self, instances: List[Instance]) -> Optional[Instance]:
        """Select instance with least connections."""
        active_instances = [i for i in instances if i.state == InstanceState.ACTIVE]
        if not active_instances:
            return None

        return min(active_instances, key=lambda i: i.current_load / i.capacity)

    def update_load(self, instance: Instance, delta: float) -> None:
        """Update instance load."""
        instance.current_load += delta


class FailoverCoordinator:
    """Main failover coordination system."""

    def __init__(
        self,
        health_checkers: Dict[HealthCheckType, HealthChecker],
        load_balancer: LoadBalancer,
        notification_handler: Optional[Callable] = None,
    ):
        self.health_checkers = health_checkers
        self.load_balancer = load_balancer
        self.notification_handler = notification_handler

        self.redundancy_groups: Dict[str, RedundancyGroup] = {}
        self.failover_history: List[FailoverEvent] = []
        self.health_check_tasks: Dict[str, asyncio.Task] = {}
        self.failover_in_progress: Set[str] = set()

    def register_redundancy_group(self, group: RedundancyGroup) -> None:
        """Register a redundancy group."""
        self.redundancy_groups[group.group_id] = group
        logger.info(f"Registered redundancy group: {group.group_id}")

        # Start health monitoring
        if group.group_id not in self.health_check_tasks:
            task = asyncio.create_task(self._monitor_group_health(group))
            self.health_check_tasks[group.group_id] = task

    async def _monitor_group_health(self, group: RedundancyGroup) -> None:
        """Monitor health of all instances in a group."""
        while True:
            try:
                for instance in group.instances:
                    if instance.state in [InstanceState.ACTIVE, InstanceState.STANDBY]:
                        for health_check in instance.health_checks:
                            is_healthy = await self._perform_health_check(
                                instance, health_check
                            )
                            await self._update_instance_health(
                                group, instance, is_healthy
                            )

                # Check if failover is needed
                await self._check_failover_needed(group)

                # Sleep until next check
                await asyncio.sleep(group.health_check_config.interval.total_seconds())

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error monitoring group {group.group_id}: {e}")
                await asyncio.sleep(10)  # Back off on error

    async def _perform_health_check(
        self, instance: Instance, health_check: HealthCheck
    ) -> bool:
        """Perform a single health check."""
        checker = self.health_checkers.get(health_check.check_type)
        if not checker:
            logger.warning(f"No health checker for type: {health_check.check_type}")
            return True

        return await checker.check(instance, health_check)

    async def _update_instance_health(
        self, group: RedundancyGroup, instance: Instance, is_healthy: bool
    ) -> None:
        """Update instance health status."""
        instance.last_health_check = datetime.now()

        if is_healthy:
            instance.consecutive_failures = 0
            if instance.state == InstanceState.RECOVERING:
                # Promote back to active after recovery
                if instance.consecutive_failures == 0:
                    instance.state = InstanceState.ACTIVE
                    logger.info(f"Instance {instance.instance_id} recovered")
        else:
            instance.consecutive_failures += 1

            # Check if instance should be marked as failing
            if (
                instance.consecutive_failures >=
                group.health_check_config.unhealthy_threshold
            ):
                if instance.state == InstanceState.ACTIVE:
                    instance.state = InstanceState.FAILING
                    logger.warning(f"Instance {instance.instance_id} is failing")

    async def _check_failover_needed(self, group: RedundancyGroup) -> None:
        """Check if failover is needed for a group."""
        active_instances = group.get_active_instances()
        failing_instances = [
            i for i in group.instances if i.state == InstanceState.FAILING
        ]

        # Check if we have enough active instances
        if len(active_instances) < group.min_active_instances and failing_instances:
            for instance in failing_instances:
                if instance.instance_id not in self.failover_in_progress:
                    await self.initiate_failover(group.group_id, instance)

    async def initiate_failover(
        self, group_id: str, failed_instance: Instance
    ) -> FailoverEvent:
        """Initiate failover for a failed instance."""
        logger.info(f"Initiating failover for instance {failed_instance.instance_id}")

        group = self.redundancy_groups.get(group_id)
        if not group:
            raise ValueError(f"Unknown redundancy group: {group_id}")

        self.failover_in_progress.add(failed_instance.instance_id)

        event = FailoverEvent(
            event_id=f"failover_{int(time.time() * 1000)}",
            timestamp=datetime.now(),
            group_id=group_id,
            failed_instance=failed_instance,
            replacement_instance=None,
            strategy_used=group.failover_strategy,
            duration=timedelta(),
            success=False,
        )

        start_time = time.time()

        try:
            # Execute failover based on strategy
            if group.failover_strategy == FailoverStrategy.IMMEDIATE:
                event.replacement_instance = await self._immediate_failover(
                    group, failed_instance
                )
            elif group.failover_strategy == FailoverStrategy.GRACEFUL:
                event.replacement_instance = await self._graceful_failover(
                    group, failed_instance
                )
            elif group.failover_strategy == FailoverStrategy.STAGED:
                event.replacement_instance = await self._staged_failover(
                    group, failed_instance
                )
            elif group.failover_strategy == FailoverStrategy.MANUAL:
                event.replacement_instance = await self._manual_failover(
                    group, failed_instance
                )
            else:
                event.replacement_instance = await self._automatic_failover(
                    group, failed_instance
                )

            event.success = event.replacement_instance is not None
            event.duration = timedelta(seconds=time.time() - start_time)

            # Update instance states
            if event.success:
                failed_instance.state = InstanceState.FAILED
                logger.info(
                    f"Failover completed: {failed_instance.instance_id} -> {event.replacement_instance.instance_id}"
                )

            # Send notification
            if self.notification_handler:
                await self.notification_handler(event)

        except Exception as e:
            logger.error(f"Failover failed: {e}")
            event.error_message = str(e)
            event.duration = timedelta(seconds=time.time() - start_time)

        finally:
            self.failover_in_progress.discard(failed_instance.instance_id)
            self.failover_history.append(event)

        return event

    async def _immediate_failover(
        self, group: RedundancyGroup, failed_instance: Instance
    ) -> Optional[Instance]:
        """Perform immediate failover."""
        # Find standby instance
        standby_instances = group.get_standby_instances()

        if not standby_instances:
            logger.error("No standby instances available for failover")
            return None

        # Select best standby instance
        replacement = self._select_best_standby(standby_instances, failed_instance)

        # Promote to active
        replacement.state = InstanceState.ACTIVE

        # Update load balancer
        self.load_balancer.update_load(replacement, failed_instance.current_load)

        return replacement

    async def _graceful_failover(
        self, group: RedundancyGroup, failed_instance: Instance
    ) -> Optional[Instance]:
        """Perform graceful failover with connection draining."""
        # Mark instance as draining
        failed_instance.state = InstanceState.DRAINING

        # Find replacement
        standby_instances = group.get_standby_instances()
        if not standby_instances:
            return None

        replacement = self._select_best_standby(standby_instances, failed_instance)
        replacement.state = InstanceState.ACTIVE

        # Wait for connections to drain
        drain_timeout = 30  # seconds
        start_time = time.time()

        while (
            failed_instance.current_load > 0 and
            time.time() - start_time < drain_timeout
        ):
            await asyncio.sleep(1)
            # In real implementation, would check actual connection count
            failed_instance.current_load = max(0, failed_instance.current_load - 0.1)

        # Transfer remaining load
        self.load_balancer.update_load(replacement, failed_instance.current_load)
        failed_instance.current_load = 0

        return replacement

    async def _staged_failover(
        self, group: RedundancyGroup, failed_instance: Instance
    ) -> Optional[Instance]:
        """Perform staged failover with gradual traffic shift."""
        standby_instances = group.get_standby_instances()
        if not standby_instances:
            return None

        replacement = self._select_best_standby(standby_instances, failed_instance)
        replacement.state = InstanceState.ACTIVE

        # Gradually shift traffic
        stages = 5
        for i in range(stages):
            transfer_ratio = (i + 1) / stages
            transferred_load = failed_instance.current_load * transfer_ratio

            replacement.current_load = transferred_load
            failed_instance.current_load = failed_instance.current_load * (
                1 - transfer_ratio
            )

            await asyncio.sleep(2)  # Wait between stages

        return replacement

    async def _manual_failover(
        self, group: RedundancyGroup, failed_instance: Instance
    ) -> Optional[Instance]:
        """Manual failover requiring approval."""
        logger.warning(f"Manual failover required for {failed_instance.instance_id}")

        # In real implementation, would wait for manual approval
        # For now, simulate approval after delay
        await asyncio.sleep(5)

        return await self._immediate_failover(group, failed_instance)

    async def _automatic_failover(
        self, group: RedundancyGroup, failed_instance: Instance
    ) -> Optional[Instance]:
        """Fully automatic failover with optimization."""
        # Determine best strategy based on conditions
        active_count = len(group.get_active_instances())

        if active_count < group.min_active_instances:
            # Critical - use immediate failover
            return await self._immediate_failover(group, failed_instance)
        elif failed_instance.current_load < 0.1:
            # Low load - use immediate failover
            return await self._immediate_failover(group, failed_instance)
        else:
            # Normal conditions - use graceful failover
            return await self._graceful_failover(group, failed_instance)

    def _select_best_standby(
        self, standby_instances: List[Instance], failed_instance: Instance
    ) -> Instance:
        """Select the best standby instance for replacement."""
        # Prefer same zone/region
        same_zone = [i for i in standby_instances if i.zone == failed_instance.zone]
        if same_zone:
            return same_zone[0]

        same_region = [
            i for i in standby_instances if i.region == failed_instance.region
        ]
        if same_region:
            return same_region[0]

        # Otherwise, select instance with most capacity
        return max(standby_instances, key=lambda i: i.capacity - i.current_load)

    def get_group_status(self, group_id: str) -> Dict[str, Any]:
        """Get status of a redundancy group."""
        group = self.redundancy_groups.get(group_id)
        if not group:
            return {}

        active_instances = group.get_active_instances()
        standby_instances = group.get_standby_instances()
        failing_instances = [
            i for i in group.instances if i.state == InstanceState.FAILING
        ]

        total_capacity = sum(i.capacity for i in active_instances)
        total_load = sum(i.current_load for i in active_instances)

        return {
            "group_id": group_id,
            "service_name": group.service_name,
            "redundancy_type": group.redundancy_type.value,
            "active_instances": len(active_instances),
            "standby_instances": len(standby_instances),
            "failing_instances": len(failing_instances),
            "total_capacity": total_capacity,
            "total_load": total_load,
            "load_percentage": (
                (total_load / total_capacity * 100) if total_capacity > 0 else 0
            ),
            "health_status": "healthy" if len(failing_instances) == 0 else "degraded",
            "instances": [i.to_dict() for i in group.instances],
        }

    def get_failover_metrics(self) -> Dict[str, Any]:
        """Get failover performance metrics."""
        if not self.failover_history:
            return {"total_failovers": 0, "success_rate": 0.0, "average_duration": 0.0}

        successful = [e for e in self.failover_history if e.success]
        total_duration = sum(e.duration.total_seconds() for e in self.failover_history)

        return {
            "total_failovers": len(self.failover_history),
            "success_rate": len(successful) / len(self.failover_history),
            "average_duration": total_duration / len(self.failover_history),
            "by_strategy": self._get_strategy_metrics(),
            "recent_events": [e.to_dict() for e in self.failover_history[-10:]],
        }

    def _get_strategy_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get metrics grouped by strategy."""
        strategy_metrics = {}

        for strategy in FailoverStrategy:
            events = [e for e in self.failover_history if e.strategy_used == strategy]
            if events:
                successful = [e for e in events if e.success]
                avg_duration = sum(e.duration.total_seconds() for e in events) / len(
                    events
                )

                strategy_metrics[strategy.value] = {
                    "count": len(events),
                    "success_rate": len(successful) / len(events),
                    "average_duration": avg_duration,
                }

        return strategy_metrics


# Example redundancy configurations
def create_web_service_redundancy() -> RedundancyGroup:
    """Create redundancy group for web service."""
    instances = [
        Instance(
            instance_id="web-1",
            host="web1.example.com",
            port=8080,
            state=InstanceState.ACTIVE,
            region="us-east-1",
            zone="us-east-1a",
            capacity=1.0,
            health_checks=[
                HealthCheck(
                    check_type=HealthCheckType.HTTP,
                    endpoint="/health",
                    interval=timedelta(seconds=10),
                    timeout=timedelta(seconds=5),
                )
            ],
        ),
        Instance(
            instance_id="web-2",
            host="web2.example.com",
            port=8080,
            state=InstanceState.ACTIVE,
            region="us-east-1",
            zone="us-east-1b",
            capacity=1.0,
            health_checks=[
                HealthCheck(
                    check_type=HealthCheckType.HTTP,
                    endpoint="/health",
                    interval=timedelta(seconds=10),
                    timeout=timedelta(seconds=5),
                )
            ],
        ),
        Instance(
            instance_id="web-3",
            host="web3.example.com",
            port=8080,
            state=InstanceState.STANDBY,
            region="us-east-1",
            zone="us-east-1c",
            capacity=1.0,
            health_checks=[
                HealthCheck(
                    check_type=HealthCheckType.HTTP,
                    endpoint="/health",
                    interval=timedelta(seconds=10),
                    timeout=timedelta(seconds=5),
                )
            ],
        ),
    ]

    return RedundancyGroup(
        group_id="web-service-primary",
        service_name="web-service",
        redundancy_type=RedundancyType.N_PLUS_ONE,
        instances=instances,
        min_active_instances=2,
        max_active_instances=3,
        failover_strategy=FailoverStrategy.GRACEFUL,
        health_check_config=HealthCheck(
            check_type=HealthCheckType.HTTP,
            endpoint="/health",
            interval=timedelta(seconds=10),
            timeout=timedelta(seconds=5),
        ),
    )


def create_database_redundancy() -> RedundancyGroup:
    """Create redundancy group for database."""
    instances = [
        Instance(
            instance_id="db-primary",
            host="db1.example.com",
            port=5432,
            state=InstanceState.ACTIVE,
            region="us-east-1",
            zone="us-east-1a",
            capacity=1.0,
            metadata={"role": "primary"},
            health_checks=[
                HealthCheck(
                    check_type=HealthCheckType.TCP,
                    endpoint="",
                    interval=timedelta(seconds=5),
                    timeout=timedelta(seconds=2),
                )
            ],
        ),
        Instance(
            instance_id="db-secondary",
            host="db2.example.com",
            port=5432,
            state=InstanceState.STANDBY,
            region="us-east-1",
            zone="us-east-1b",
            capacity=1.0,
            metadata={"role": "secondary", "replication_lag": 0},
            health_checks=[
                HealthCheck(
                    check_type=HealthCheckType.TCP,
                    endpoint="",
                    interval=timedelta(seconds=5),
                    timeout=timedelta(seconds=2),
                )
            ],
        ),
    ]

    return RedundancyGroup(
        group_id="database-primary",
        service_name="postgresql",
        redundancy_type=RedundancyType.ACTIVE_PASSIVE,
        instances=instances,
        min_active_instances=1,
        max_active_instances=1,
        failover_strategy=FailoverStrategy.AUTOMATIC,
        health_check_config=HealthCheck(
            check_type=HealthCheckType.DATABASE,
            endpoint="",
            interval=timedelta(seconds=5),
            timeout=timedelta(seconds=2),
        ),
    )
