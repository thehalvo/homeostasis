"""
Multi-Region Resilience Strategies

Implements strategies for maintaining service health and healing
capabilities across multiple geographic regions, including failover,
data consistency, and coordinated healing.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import geopy.distance

from modules.deployment.multi_environment.hybrid_orchestrator import (
    Environment, HealingContext, HealingPlan, HealingStep)
from modules.monitoring.distributed_monitoring import DistributedMonitor
from modules.security.audit import AuditLogger


class RegionStatus(Enum):
    """Health status of a region"""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"


class FailoverStrategy(Enum):
    """Types of failover strategies"""

    ACTIVE_PASSIVE = "active_passive"
    ACTIVE_ACTIVE = "active_active"
    PRIMARY_SECONDARY = "primary_secondary"
    ROUND_ROBIN = "round_robin"
    GEOGRAPHIC = "geographic"
    LATENCY_BASED = "latency_based"


class ConsistencyModel(Enum):
    """Data consistency models"""

    STRONG = "strong"
    EVENTUAL = "eventual"
    BOUNDED_STALENESS = "bounded_staleness"
    SESSION = "session"
    CONSISTENT_PREFIX = "consistent_prefix"


@dataclass
class Region:
    """Represents a geographic region"""

    id: str
    name: str
    location: Tuple[float, float]  # latitude, longitude
    environments: List[Environment]
    status: RegionStatus
    capacity: Dict[str, float]  # resource type -> available capacity
    metadata: Dict[str, Any]


@dataclass
class RegionHealth:
    """Health metrics for a region"""

    region_id: str
    status: RegionStatus
    availability: float  # 0-100%
    latency_ms: float
    error_rate: float
    capacity_usage: Dict[str, float]
    last_updated: datetime
    incidents: List[Dict[str, Any]]


@dataclass
class FailoverEvent:
    """Records a failover event"""

    event_id: str
    timestamp: datetime
    from_region: Region
    to_region: Region
    reason: str
    affected_services: List[str]
    duration_ms: Optional[float]
    success: bool


@dataclass
class ResiliencePolicy:
    """Policy for multi-region resilience"""

    name: str
    failover_strategy: FailoverStrategy
    consistency_model: ConsistencyModel
    rpo_seconds: int  # Recovery Point Objective
    rto_seconds: int  # Recovery Time Objective
    health_check_interval: int
    failover_threshold: float
    auto_failback: bool
    geo_restrictions: Optional[List[str]]  # List of allowed regions


class RegionHealthMonitor:
    """Monitors health across regions"""

    def __init__(self, regions: List[Region]):
        self.regions = {r.id: r for r in regions}
        self.health_history: Dict[str, List[RegionHealth]] = {r.id: [] for r in regions}
        self.monitor = DistributedMonitor()
        self.logger = logging.getLogger(__name__)

    async def check_region_health(self, region: Region) -> RegionHealth:
        """Check health of a single region"""
        try:
            # Aggregate health from all environments in region
            env_healths = []
            for env in region.environments:
                health = await self._check_environment_health(env)
                env_healths.append(health)

            # Calculate region metrics
            availability = self._calculate_availability(env_healths)
            latency = self._calculate_average_latency(env_healths)
            error_rate = self._calculate_error_rate(env_healths)
            capacity_usage = self._calculate_capacity_usage(region, env_healths)

            # Determine overall status
            if availability < 50 or error_rate > 0.1:
                status = RegionStatus.UNHEALTHY
            elif availability < 90 or error_rate > 0.05:
                status = RegionStatus.DEGRADED
            else:
                status = RegionStatus.HEALTHY

            health = RegionHealth(
                region_id=region.id,
                status=status,
                availability=availability,
                latency_ms=latency,
                error_rate=error_rate,
                capacity_usage=capacity_usage,
                last_updated=datetime.utcnow(),
                incidents=[],
            )

            # Store health history
            self.health_history[region.id].append(health)
            if len(self.health_history[region.id]) > 1000:
                self.health_history[region.id] = self.health_history[region.id][-1000:]

            return health

        except Exception as e:
            self.logger.error(f"Failed to check health for region {region.id}: {e}")
            return RegionHealth(
                region_id=region.id,
                status=RegionStatus.OFFLINE,
                availability=0,
                latency_ms=99999,
                error_rate=1.0,
                capacity_usage={},
                last_updated=datetime.utcnow(),
                incidents=[{"error": str(e)}],
            )

    async def _check_environment_health(self, env: Environment) -> Dict[str, Any]:
        """Check health of an environment"""
        # In practice, would query actual metrics
        return {
            "availability": 95.0,
            "latency_ms": 50.0,
            "error_rate": 0.01,
            "cpu_usage": 60.0,
            "memory_usage": 70.0,
            "disk_usage": 40.0,
        }

    def _calculate_availability(self, env_healths: List[Dict[str, Any]]) -> float:
        """Calculate average availability across environments"""
        if not env_healths:
            return 0.0
        return sum(h.get("availability", 0) for h in env_healths) / len(env_healths)

    def _calculate_average_latency(self, env_healths: List[Dict[str, Any]]) -> float:
        """Calculate average latency across environments"""
        if not env_healths:
            return 99999.0
        return sum(h.get("latency_ms", 99999) for h in env_healths) / len(env_healths)

    def _calculate_error_rate(self, env_healths: List[Dict[str, Any]]) -> float:
        """Calculate average error rate across environments"""
        if not env_healths:
            return 1.0
        return sum(h.get("error_rate", 1.0) for h in env_healths) / len(env_healths)

    def _calculate_capacity_usage(
        self, region: Region, env_healths: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate capacity usage for region"""
        return {
            "cpu": (
                sum(h.get("cpu_usage", 0) for h in env_healths) / len(env_healths)
                if env_healths
                else 0
            ),
            "memory": (
                sum(h.get("memory_usage", 0) for h in env_healths) / len(env_healths)
                if env_healths
                else 0
            ),
            "disk": (
                sum(h.get("disk_usage", 0) for h in env_healths) / len(env_healths)
                if env_healths
                else 0
            ),
        }

    def get_region_trend(self, region_id: str, hours: int = 24) -> Dict[str, Any]:
        """Get health trend for a region"""
        if region_id not in self.health_history:
            return {}

        cutoff = datetime.utcnow() - timedelta(hours=hours)
        recent_health = [
            h for h in self.health_history[region_id] if h.last_updated >= cutoff
        ]

        if not recent_health:
            return {}

        return {
            "average_availability": sum(h.availability for h in recent_health) /
            len(recent_health),
            "average_latency": sum(h.latency_ms for h in recent_health) /
            len(recent_health),
            "average_error_rate": sum(h.error_rate for h in recent_health) /
            len(recent_health),
            "status_counts": self._count_statuses(recent_health),
            "trend": self._calculate_trend(recent_health),
        }

    def _count_statuses(self, healths: List[RegionHealth]) -> Dict[str, int]:
        """Count occurrences of each status"""
        counts = {status.value: 0 for status in RegionStatus}
        for health in healths:
            counts[health.status.value] += 1
        return counts

    def _calculate_trend(self, healths: List[RegionHealth]) -> str:
        """Calculate health trend (improving/stable/degrading)"""
        if len(healths) < 2:
            return "stable"

        # Compare first half with second half
        mid = len(healths) // 2
        first_half_avg = sum(h.availability for h in healths[:mid]) / mid
        second_half_avg = sum(h.availability for h in healths[mid:]) / len(
            healths[mid:]
        )

        if second_half_avg > first_half_avg + 5:
            return "improving"
        elif second_half_avg < first_half_avg - 5:
            return "degrading"
        else:
            return "stable"


class FailoverOrchestrator:
    """Orchestrates failover between regions"""

    def __init__(self, regions: List[Region], policy: ResiliencePolicy):
        self.regions = {r.id: r for r in regions}
        self.policy = policy
        self.primary_region: Optional[str] = None
        self.active_regions: Set[str] = set()
        self.failover_history: List[FailoverEvent] = []
        self.auditor = AuditLogger()
        self.logger = logging.getLogger(__name__)

    async def execute_failover(
        self,
        from_region_id: str,
        to_region_id: str,
        reason: str,
        affected_services: List[str],
    ) -> FailoverEvent:
        """Execute failover from one region to another"""
        self.logger.info(f"Executing failover from {from_region_id} to {to_region_id}")

        start_time = datetime.utcnow()
        event = FailoverEvent(
            event_id=f"failover_{start_time.timestamp()}",
            timestamp=start_time,
            from_region=self.regions[from_region_id],
            to_region=self.regions[to_region_id],
            reason=reason,
            affected_services=affected_services,
            duration_ms=None,
            success=False,
        )

        try:
            # Pre-failover checks
            await self._pre_failover_checks(from_region_id, to_region_id)

            # Execute failover steps
            if self.policy.failover_strategy == FailoverStrategy.ACTIVE_PASSIVE:
                await self._execute_active_passive_failover(
                    from_region_id, to_region_id
                )
            elif self.policy.failover_strategy == FailoverStrategy.ACTIVE_ACTIVE:
                await self._execute_active_active_failover(from_region_id, to_region_id)
            elif self.policy.failover_strategy == FailoverStrategy.PRIMARY_SECONDARY:
                await self._execute_primary_secondary_failover(
                    from_region_id, to_region_id
                )

            # Update routing
            await self._update_routing(from_region_id, to_region_id)

            # Verify failover
            success = await self._verify_failover(to_region_id, affected_services)

            event.success = success
            event.duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            # Record event
            self.failover_history.append(event)
            self.auditor.log_event(
                "region_failover",
                user="system",
                details={
                    "event_id": event.event_id,
                    "from_region": from_region_id,
                    "to_region": to_region_id,
                    "reason": reason,
                    "success": success,
                    "duration_ms": event.duration_ms,
                },
            )

            return event

        except Exception as e:
            self.logger.error(f"Failover failed: {e}")
            event.success = False
            event.duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.failover_history.append(event)
            raise

    async def _pre_failover_checks(self, from_region_id: str, to_region_id: str):
        """Perform pre-failover validation"""
        to_region = self.regions[to_region_id]

        # Check target region health
        if to_region.status != RegionStatus.HEALTHY:
            raise Exception(f"Target region {to_region_id} is not healthy")

        # Check capacity
        # In practice, would check actual resource availability

        # Check geo-restrictions
        if self.policy.geo_restrictions:
            if to_region_id not in self.policy.geo_restrictions:
                raise Exception(f"Region {to_region_id} not allowed by policy")

    async def _execute_active_passive_failover(
        self, from_region_id: str, to_region_id: str
    ):
        """Execute active-passive failover"""
        # Activate passive region
        await self._activate_region(to_region_id)

        # Wait for activation
        await asyncio.sleep(5)

        # Deactivate old region
        await self._deactivate_region(from_region_id)

    async def _execute_active_active_failover(
        self, from_region_id: str, to_region_id: str
    ):
        """Execute active-active failover (rebalancing)"""
        # Both regions remain active, just redistribute load
        await self._redistribute_load(from_region_id, to_region_id)

    async def _execute_primary_secondary_failover(
        self, from_region_id: str, to_region_id: str
    ):
        """Execute primary-secondary failover"""
        # Promote secondary to primary
        self.primary_region = to_region_id

        # Update configurations
        await self._update_primary_configs(to_region_id)

    async def _activate_region(self, region_id: str):
        """Activate a region for traffic"""
        self.active_regions.add(region_id)
        self.logger.info(f"Activated region {region_id}")

    async def _deactivate_region(self, region_id: str):
        """Deactivate a region from receiving traffic"""
        self.active_regions.discard(region_id)
        self.logger.info(f"Deactivated region {region_id}")

    async def _redistribute_load(self, from_region_id: str, to_region_id: str):
        """Redistribute load between regions"""
        # In practice, would update load balancer weights
        self.logger.info(f"Redistributed load from {from_region_id} to {to_region_id}")

    async def _update_primary_configs(self, region_id: str):
        """Update configurations for new primary region"""
        # In practice, would update database configs, etc.
        self.logger.info(f"Updated primary configurations for {region_id}")

    async def _update_routing(self, from_region_id: str, to_region_id: str):
        """Update routing configuration"""
        # In practice, would update DNS, load balancers, CDN, etc.
        self.logger.info(f"Updated routing from {from_region_id} to {to_region_id}")

    async def _verify_failover(self, region_id: str, services: List[str]) -> bool:
        """Verify failover completed successfully"""
        # In practice, would run health checks on services
        return True

    async def auto_failback(self, original_region_id: str) -> Optional[FailoverEvent]:
        """Attempt automatic failback to original region"""
        if not self.policy.auto_failback:
            return None

        # Check if original region is healthy
        original_region = self.regions[original_region_id]
        if original_region.status != RegionStatus.HEALTHY:
            return None

        # Find current primary
        current_primary = self.primary_region or next(iter(self.active_regions), None)
        if not current_primary or current_primary == original_region_id:
            return None

        # Execute failback
        return await self.execute_failover(
            current_primary, original_region_id, "auto_failback", []  # All services
        )


class MultiRegionResilienceStrategy:
    """
    Comprehensive strategy for multi-region resilience including
    health monitoring, failover orchestration, and healing coordination.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.regions = self._load_regions(config.get("regions", []))
        self.policies = self._load_policies(config.get("policies", []))
        self.health_monitor = RegionHealthMonitor(self.regions)
        self.failover_orchestrators: Dict[str, FailoverOrchestrator] = {}
        self.logger = logging.getLogger(__name__)

        # Initialize orchestrators for each policy
        for policy in self.policies.values():
            relevant_regions = self._get_policy_regions(policy)
            self.failover_orchestrators[policy.name] = FailoverOrchestrator(
                relevant_regions, policy
            )

    def _load_regions(self, region_configs: List[Dict[str, Any]]) -> List[Region]:
        """Load region configurations"""
        regions = []
        for config in region_configs:
            region = Region(
                id=config["id"],
                name=config["name"],
                location=tuple(config["location"]),
                environments=[],  # Would be populated from environment configs
                status=RegionStatus.HEALTHY,
                capacity=config.get("capacity", {}),
                metadata=config.get("metadata", {}),
            )
            regions.append(region)
        return regions

    def _load_policies(
        self, policy_configs: List[Dict[str, Any]]
    ) -> Dict[str, ResiliencePolicy]:
        """Load resilience policies"""
        policies = {}
        for config in policy_configs:
            policy = ResiliencePolicy(
                name=config["name"],
                failover_strategy=FailoverStrategy(config["failover_strategy"]),
                consistency_model=ConsistencyModel(config["consistency_model"]),
                rpo_seconds=config["rpo_seconds"],
                rto_seconds=config["rto_seconds"],
                health_check_interval=config.get("health_check_interval", 30),
                failover_threshold=config.get("failover_threshold", 0.5),
                auto_failback=config.get("auto_failback", True),
                geo_restrictions=config.get("geo_restrictions"),
            )
            policies[policy.name] = policy
        return policies

    def _get_policy_regions(self, policy: ResiliencePolicy) -> List[Region]:
        """Get regions applicable to a policy"""
        if policy.geo_restrictions:
            return [r for r in self.regions if r.id in policy.geo_restrictions]
        return self.regions

    async def monitor_health_loop(self):
        """Continuous health monitoring loop"""
        while True:
            try:
                # Check health of all regions
                health_results = await asyncio.gather(
                    *[
                        self.health_monitor.check_region_health(region)
                        for region in self.regions
                    ]
                )

                # Update region statuses
                for region, health in zip(self.regions, health_results):
                    region.status = health.status

                # Check for failover conditions
                await self._check_failover_conditions(health_results)

                # Wait before next check
                await asyncio.sleep(30)  # Default 30 second interval

            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(30)

    async def _check_failover_conditions(self, health_results: List[RegionHealth]):
        """Check if any regions need failover"""
        for policy_name, orchestrator in self.failover_orchestrators.items():
            policy = self.policies[policy_name]

            # Check each region's health against threshold
            for health in health_results:
                if health.availability < (100 * policy.failover_threshold):
                    # Find best target region
                    target = self._find_best_failover_target(
                        health.region_id, health_results, policy
                    )

                    if target:
                        self.logger.warning(
                            f"Region {health.region_id} below threshold, "
                            f"initiating failover to {target}"
                        )

                        try:
                            await orchestrator.execute_failover(
                                health.region_id,
                                target,
                                f"availability below threshold ({health.availability}%)",
                                [],  # Would identify affected services
                            )
                        except Exception as e:
                            self.logger.error(f"Failover failed: {e}")

    def _find_best_failover_target(
        self,
        failing_region_id: str,
        health_results: List[RegionHealth],
        policy: ResiliencePolicy,
    ) -> Optional[str]:
        """Find the best region to failover to"""
        candidates = []

        for health in health_results:
            if health.region_id == failing_region_id:
                continue

            if health.status != RegionStatus.HEALTHY:
                continue

            if (
                policy.geo_restrictions and
                health.region_id not in policy.geo_restrictions
            ):
                continue

            # Score based on availability and capacity
            score = health.availability - sum(health.capacity_usage.values()) / 3
            candidates.append((health.region_id, score))

        if not candidates:
            return None

        # Return region with highest score
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    async def handle_healing_across_regions(
        self, healing_context: HealingContext, healing_plan: HealingPlan
    ) -> Dict[str, Any]:
        """Coordinate healing actions across multiple regions"""
        affected_regions = set()

        # Identify affected regions
        for env in healing_context.affected_environments:
            for region in self.regions:
                if env in region.environments:
                    affected_regions.add(region.id)

        if len(affected_regions) <= 1:
            # Single region healing, no special coordination needed
            return {"status": "single_region", "regions": list(affected_regions)}

        # Multi-region healing coordination
        self.logger.info(f"Coordinating healing across regions: {affected_regions}")

        # Check consistency requirements
        consistency_model = self._determine_consistency_model(healing_context)

        if consistency_model == ConsistencyModel.STRONG:
            # Synchronous healing across all regions
            return await self._coordinate_synchronous_healing(
                healing_plan, affected_regions
            )
        else:
            # Asynchronous healing with eventual consistency
            return await self._coordinate_asynchronous_healing(
                healing_plan, affected_regions, consistency_model
            )

    def _determine_consistency_model(self, context: HealingContext) -> ConsistencyModel:
        """Determine required consistency model for healing"""
        # Check if any affected service requires strong consistency
        if context.constraints.get("data_consistency", True):
            return ConsistencyModel.STRONG

        # Check service types
        critical_services = ["payment", "auth", "inventory"]
        if any(svc in context.dependencies for svc in critical_services):
            return ConsistencyModel.STRONG

        return ConsistencyModel.EVENTUAL

    async def _coordinate_synchronous_healing(
        self, plan: HealingPlan, affected_regions: Set[str]
    ) -> Dict[str, Any]:
        """Coordinate synchronous healing across regions"""
        # Execute healing steps in lock-step across regions
        results = {"status": "synchronous", "regions": {}}

        # Group steps by phase
        phases = self._group_steps_by_phase(plan.steps)

        for phase, steps in phases.items():
            # Execute phase across all regions simultaneously
            phase_results = await asyncio.gather(
                *[
                    self._execute_regional_phase(region_id, steps)
                    for region_id in affected_regions
                ],
                return_exceptions=True,
            )

            # Check for failures
            for region_id, result in zip(affected_regions, phase_results):
                if isinstance(result, Exception):
                    # Rollback all regions
                    await self._rollback_all_regions(plan, affected_regions)
                    return {
                        "status": "failed",
                        "error": str(result),
                        "failed_region": region_id,
                        "phase": phase,
                    }
                results["regions"][region_id] = result

        return results

    async def _coordinate_asynchronous_healing(
        self,
        plan: HealingPlan,
        affected_regions: Set[str],
        consistency_model: ConsistencyModel,
    ) -> Dict[str, Any]:
        """Coordinate asynchronous healing with eventual consistency"""
        results = {
            "status": "asynchronous",
            "consistency_model": consistency_model.value,
            "regions": {},
        }

        # Execute healing independently in each region
        tasks = [
            self._execute_regional_healing(region_id, plan)
            for region_id in affected_regions
        ]

        region_results = await asyncio.gather(*tasks, return_exceptions=True)

        for region_id, result in zip(affected_regions, region_results):
            if isinstance(result, Exception):
                results["regions"][region_id] = {
                    "status": "failed",
                    "error": str(result),
                }
            else:
                results["regions"][region_id] = result

        # Handle consistency based on model
        if consistency_model == ConsistencyModel.BOUNDED_STALENESS:
            # Wait for regions to converge within time bound
            await self._wait_for_convergence(affected_regions, timeout=300)

        return results

    def _group_steps_by_phase(
        self, steps: List[HealingStep]
    ) -> Dict[str, List[HealingStep]]:
        """Group healing steps into phases based on dependencies"""
        phases = {}
        phase_num = 0

        remaining_steps = steps.copy()
        completed_steps = set()

        while remaining_steps:
            phase_steps = []

            for step in remaining_steps:
                # Check if all dependencies are completed
                if all(dep in completed_steps for dep in step.dependencies):
                    phase_steps.append(step)

            if not phase_steps:
                # Circular dependency or error
                raise Exception("Unable to resolve step dependencies")

            phases[f"phase_{phase_num}"] = phase_steps
            completed_steps.update(s.step_id for s in phase_steps)
            remaining_steps = [s for s in remaining_steps if s not in phase_steps]
            phase_num += 1

        return phases

    async def _execute_regional_phase(
        self, region_id: str, steps: List[HealingStep]
    ) -> Dict[str, Any]:
        """Execute a phase of healing steps in a region"""
        # In practice, would coordinate with regional orchestrator
        return {"status": "success", "steps_executed": len(steps)}

    async def _execute_regional_healing(
        self, region_id: str, plan: HealingPlan
    ) -> Dict[str, Any]:
        """Execute complete healing plan in a region"""
        # In practice, would delegate to regional orchestrator
        return {"status": "success", "plan_id": plan.plan_id}

    async def _rollback_all_regions(self, plan: HealingPlan, regions: Set[str]):
        """Rollback healing in all regions"""
        rollback_tasks = [
            self._execute_regional_rollback(region_id, plan) for region_id in regions
        ]
        await asyncio.gather(*rollback_tasks, return_exceptions=True)

    async def _execute_regional_rollback(self, region_id: str, plan: HealingPlan):
        """Execute rollback in a specific region"""
        # In practice, would coordinate with regional orchestrator
        self.logger.info(f"Rolling back healing in region {region_id}")

    async def _wait_for_convergence(self, regions: Set[str], timeout: int):
        """Wait for regions to converge to consistent state"""
        start_time = datetime.utcnow()

        while (datetime.utcnow() - start_time).total_seconds() < timeout:
            # Check if regions have converged
            converged = await self._check_convergence(regions)
            if converged:
                return

            await asyncio.sleep(5)

        raise Exception(f"Regions did not converge within {timeout} seconds")

    async def _check_convergence(self, regions: Set[str]) -> bool:
        """Check if regions have converged to consistent state"""
        # In practice, would check data version vectors, etc.
        return True

    def get_region_distance(self, region1_id: str, region2_id: str) -> float:
        """Calculate distance between two regions in kilometers"""
        region1 = next((r for r in self.regions if r.id == region1_id), None)
        region2 = next((r for r in self.regions if r.id == region2_id), None)

        if not region1 or not region2:
            return float("inf")

        return geopy.distance.distance(region1.location, region2.location).km

    def get_optimal_region_for_location(
        self,
        location: Tuple[float, float],
        available_regions: Optional[List[str]] = None,
    ) -> Optional[str]:
        """Find optimal region for a given geographic location"""
        if available_regions:
            regions = [r for r in self.regions if r.id in available_regions]
        else:
            regions = [r for r in self.regions if r.status == RegionStatus.HEALTHY]

        if not regions:
            return None

        # Find closest region
        distances = [
            (r.id, geopy.distance.distance(location, r.location).km) for r in regions
        ]
        distances.sort(key=lambda x: x[1])

        return distances[0][0] if distances else None

    async def get_resilience_status(self) -> Dict[str, Any]:
        """Get comprehensive resilience status across all regions"""
        status = {
            "regions": {},
            "policies": {},
            "failover_history": [],
            "health_trends": {},
            "recommendations": [],
        }

        # Region statuses
        for region in self.regions:
            health = await self.health_monitor.check_region_health(region)
            status["regions"][region.id] = {
                "name": region.name,
                "status": health.status.value,
                "availability": health.availability,
                "latency_ms": health.latency_ms,
                "error_rate": health.error_rate,
                "capacity_usage": health.capacity_usage,
            }

        # Policy statuses
        for policy_name, policy in self.policies.items():
            orchestrator = self.failover_orchestrators[policy_name]
            status["policies"][policy_name] = {
                "strategy": policy.failover_strategy.value,
                "consistency_model": policy.consistency_model.value,
                "primary_region": orchestrator.primary_region,
                "active_regions": list(orchestrator.active_regions),
                "auto_failback": policy.auto_failback,
            }

        # Recent failover history
        for orchestrator in self.failover_orchestrators.values():
            for event in orchestrator.failover_history[-10:]:  # Last 10 events
                status["failover_history"].append(
                    {
                        "timestamp": event.timestamp.isoformat(),
                        "from_region": event.from_region.id,
                        "to_region": event.to_region.id,
                        "reason": event.reason,
                        "duration_ms": event.duration_ms,
                        "success": event.success,
                    }
                )

        # Health trends
        for region in self.regions:
            trend = self.health_monitor.get_region_trend(region.id, hours=24)
            if trend:
                status["health_trends"][region.id] = trend

        # Generate recommendations
        status["recommendations"] = self._generate_recommendations(status)

        return status

    def _generate_recommendations(self, status: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on current status"""
        recommendations = []

        # Check for unhealthy regions
        for region_id, region_status in status["regions"].items():
            if region_status["status"] in ["unhealthy", "degraded"]:
                recommendations.append(
                    f"Consider failover from {region_status['name']} "
                    f"(current availability: {region_status['availability']}%)"
                )

            # Check capacity
            for resource, usage in region_status["capacity_usage"].items():
                if usage > 80:
                    recommendations.append(
                        f"High {resource} usage in {region_status['name']} ({usage}%). "
                        f"Consider scaling or load redistribution."
                    )

        # Check failover history
        recent_failures = [
            event for event in status["failover_history"] if not event["success"]
        ]
        if len(recent_failures) > 2:
            recommendations.append(
                "Multiple failover failures detected. Review failover procedures and regional health."
            )

        # Check trends
        for region_id, trend in status["health_trends"].items():
            if trend.get("trend") == "degrading":
                recommendations.append(
                    f"Region {region_id} showing degrading health trend. Investigate root cause."
                )

        return recommendations
