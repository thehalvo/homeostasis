"""
Enhanced Resource Quotas and Rate Limiting for Homeostasis Framework

This module provides comprehensive resource management:
- Hierarchical quota management (global, tenant, user)
- Dynamic quota adjustment based on usage patterns
- Resource pooling and sharing
- Burst capacity handling
- Quota enforcement with graceful degradation
- Integration with existing rate limiting
- Real-time usage monitoring and alerts
"""

import asyncio
import json
import logging
import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import uuid

from modules.core.multi_tenancy import get_current_tenant_id, get_multi_tenancy_manager
from modules.monitoring.observability_hooks import get_observability_hooks
from modules.security.healing_rate_limiter import HealingRateLimiter, HealingRateLimitExceededError

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Types of resources that can be quota-controlled"""
    CPU_SECONDS = "cpu_seconds"
    MEMORY_MB = "memory_mb"
    STORAGE_MB = "storage_mb"
    NETWORK_BANDWIDTH_MBPS = "network_bandwidth_mbps"
    API_CALLS = "api_calls"
    HEALING_CYCLES = "healing_cycles"
    PATCHES = "patches"
    DEPLOYMENTS = "deployments"
    FILE_OPERATIONS = "file_operations"
    CONCURRENT_OPERATIONS = "concurrent_operations"
    DATABASE_CONNECTIONS = "database_connections"
    CACHE_SIZE_MB = "cache_size_mb"
    LOG_ENTRIES = "log_entries"
    CUSTOM = "custom"


class QuotaLevel(Enum):
    """Levels at which quotas can be applied"""
    GLOBAL = "global"
    TENANT = "tenant"
    USER = "user"
    SERVICE = "service"
    OPERATION = "operation"


class QuotaEnforcementMode(Enum):
    """How quotas are enforced"""
    STRICT = "strict"  # Hard limit, reject when exceeded
    SOFT = "soft"  # Allow temporary overages with alerts
    BURST = "burst"  # Allow burst capacity
    THROTTLE = "throttle"  # Slow down instead of reject


@dataclass
class ResourceQuota:
    """Definition of a resource quota"""
    resource_type: ResourceType
    limit: float
    period_seconds: int  # Time window for the quota
    enforcement_mode: QuotaEnforcementMode = QuotaEnforcementMode.STRICT
    burst_limit: Optional[float] = None  # Additional burst capacity
    burst_duration_seconds: Optional[int] = None
    warning_threshold_percent: float = 80.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceUsage:
    """Track resource usage over time"""
    resource_type: ResourceType
    current_usage: float = 0.0
    period_start: datetime = field(default_factory=datetime.utcnow)
    peak_usage: float = 0.0
    total_requests: int = 0
    rejected_requests: int = 0
    burst_usage: float = 0.0
    burst_start: Optional[datetime] = None
    history: deque = field(default_factory=lambda: deque(maxlen=100))


@dataclass
class QuotaPolicy:
    """Complete quota policy for an entity"""
    policy_id: str
    name: str
    level: QuotaLevel
    entity_id: Optional[str]  # Tenant ID, user ID, etc.
    quotas: Dict[ResourceType, ResourceQuota] = field(default_factory=dict)
    enabled: bool = True
    priority: int = 0  # Higher priority policies override lower
    inherit_from: Optional[str] = None  # Parent policy ID
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QuotaAllocation:
    """Result of quota allocation request"""
    granted: bool
    resource_type: ResourceType
    requested_amount: float
    granted_amount: float
    remaining_quota: float
    reason: Optional[str] = None
    retry_after_seconds: Optional[int] = None
    suggestions: List[str] = field(default_factory=list)


class ResourceQuotaManager:
    """
    Manages resource quotas and rate limiting across the system.
    
    Features:
    - Hierarchical quota management
    - Dynamic quota adjustment
    - Burst capacity handling
    - Real-time monitoring
    - Integration with rate limiting
    - Graceful degradation
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the resource quota manager"""
        self.config = config
        self.enabled = config.get('enabled', True)
        
        # Storage
        self.storage_path = Path(config.get('storage_path', './data/quotas'))
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Quota policies
        self._policies: Dict[str, QuotaPolicy] = {}
        self._policy_lock = threading.RLock()
        
        # Usage tracking
        self._usage: Dict[str, Dict[ResourceType, ResourceUsage]] = defaultdict(dict)
        self._usage_lock = threading.Lock()
        
        # Active allocations (for concurrent operations)
        self._active_allocations: Dict[str, Set[str]] = defaultdict(set)
        
        # Rate limiter integration
        self._rate_limiters: Dict[str, HealingRateLimiter] = {}
        
        # Default quotas
        self._init_default_quotas()
        
        # Load saved policies
        self._load_policies()
        
        # Start monitoring
        self._monitoring_thread = threading.Thread(target=self._monitor_usage, daemon=True)
        self._monitoring_thread.start()
    
    def _init_default_quotas(self):
        """Initialize default quota policies"""
        # Global default policy
        global_policy = QuotaPolicy(
            policy_id="global_default",
            name="Global Default Policy",
            level=QuotaLevel.GLOBAL,
            entity_id=None,
            quotas={
                ResourceType.CPU_SECONDS: ResourceQuota(
                    resource_type=ResourceType.CPU_SECONDS,
                    limit=36000,  # 10 CPU hours
                    period_seconds=3600,  # Per hour
                    enforcement_mode=QuotaEnforcementMode.BURST,
                    burst_limit=7200,  # 2 hour burst
                    burst_duration_seconds=300  # 5 minute burst window
                ),
                ResourceType.MEMORY_MB: ResourceQuota(
                    resource_type=ResourceType.MEMORY_MB,
                    limit=8192,  # 8 GB
                    period_seconds=1,  # Instantaneous
                    enforcement_mode=QuotaEnforcementMode.STRICT
                ),
                ResourceType.API_CALLS: ResourceQuota(
                    resource_type=ResourceType.API_CALLS,
                    limit=10000,
                    period_seconds=60,  # Per minute
                    enforcement_mode=QuotaEnforcementMode.THROTTLE,
                    burst_limit=2000,
                    burst_duration_seconds=10
                ),
                ResourceType.HEALING_CYCLES: ResourceQuota(
                    resource_type=ResourceType.HEALING_CYCLES,
                    limit=100,
                    period_seconds=3600,  # Per hour
                    enforcement_mode=QuotaEnforcementMode.STRICT
                ),
                ResourceType.CONCURRENT_OPERATIONS: ResourceQuota(
                    resource_type=ResourceType.CONCURRENT_OPERATIONS,
                    limit=50,
                    period_seconds=1,  # Instantaneous
                    enforcement_mode=QuotaEnforcementMode.STRICT
                )
            },
            priority=0
        )
        
        self._policies[global_policy.policy_id] = global_policy
        
        # Default tenant policies by tier
        self._init_tenant_tier_policies()
    
    def _init_tenant_tier_policies(self):
        """Initialize default policies for tenant tiers"""
        tier_configs = {
            "free": {
                ResourceType.CPU_SECONDS: (600, 3600),  # 10 min/hour
                ResourceType.MEMORY_MB: (512, 1),  # 512 MB
                ResourceType.API_CALLS: (100, 60),  # 100/min
                ResourceType.HEALING_CYCLES: (5, 3600),  # 5/hour
                ResourceType.CONCURRENT_OPERATIONS: (2, 1)
            },
            "starter": {
                ResourceType.CPU_SECONDS: (3600, 3600),  # 1 hour/hour
                ResourceType.MEMORY_MB: (2048, 1),  # 2 GB
                ResourceType.API_CALLS: (1000, 60),  # 1000/min
                ResourceType.HEALING_CYCLES: (20, 3600),  # 20/hour
                ResourceType.CONCURRENT_OPERATIONS: (10, 1)
            },
            "professional": {
                ResourceType.CPU_SECONDS: (18000, 3600),  # 5 hours/hour
                ResourceType.MEMORY_MB: (8192, 1),  # 8 GB
                ResourceType.API_CALLS: (5000, 60),  # 5000/min
                ResourceType.HEALING_CYCLES: (100, 3600),  # 100/hour
                ResourceType.CONCURRENT_OPERATIONS: (50, 1)
            },
            "enterprise": {
                ResourceType.CPU_SECONDS: (72000, 3600),  # 20 hours/hour
                ResourceType.MEMORY_MB: (32768, 1),  # 32 GB
                ResourceType.API_CALLS: (50000, 60),  # 50000/min
                ResourceType.HEALING_CYCLES: (1000, 3600),  # 1000/hour
                ResourceType.CONCURRENT_OPERATIONS: (200, 1)
            }
        }
        
        for tier, limits in tier_configs.items():
            policy = QuotaPolicy(
                policy_id=f"tenant_tier_{tier}",
                name=f"Tenant Tier {tier.title()}",
                level=QuotaLevel.TENANT,
                entity_id=None,  # Template policy
                quotas={},
                priority=10
            )
            
            for resource_type, (limit, period) in limits.items():
                policy.quotas[resource_type] = ResourceQuota(
                    resource_type=resource_type,
                    limit=limit,
                    period_seconds=period,
                    enforcement_mode=QuotaEnforcementMode.STRICT
                )
            
            self._policies[policy.policy_id] = policy
    
    def allocate_resource(
        self,
        resource_type: ResourceType,
        amount: float,
        entity_id: Optional[str] = None,
        level: QuotaLevel = QuotaLevel.TENANT,
        operation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> QuotaAllocation:
        """
        Request allocation of a resource.
        
        Args:
            resource_type: Type of resource to allocate
            amount: Amount requested
            entity_id: ID of entity (tenant, user, etc.)
            level: Level at which to check quota
            operation_id: Optional ID for tracking concurrent operations
            metadata: Additional metadata
            
        Returns:
            QuotaAllocation result
        """
        if not self.enabled:
            return QuotaAllocation(
                granted=True,
                resource_type=resource_type,
                requested_amount=amount,
                granted_amount=amount,
                remaining_quota=float('inf')
            )
        
        # Get applicable policy
        policy = self._get_applicable_policy(entity_id, level)
        if not policy or resource_type not in policy.quotas:
            # No quota defined, allow by default
            return QuotaAllocation(
                granted=True,
                resource_type=resource_type,
                requested_amount=amount,
                granted_amount=amount,
                remaining_quota=float('inf')
            )
        
        quota = policy.quotas[resource_type]
        
        # Get or create usage tracking
        usage_key = f"{level.value}:{entity_id or 'default'}"
        with self._usage_lock:
            if usage_key not in self._usage:
                self._usage[usage_key] = {}
            
            if resource_type not in self._usage[usage_key]:
                self._usage[usage_key][resource_type] = ResourceUsage(
                    resource_type=resource_type
                )
            
            usage = self._usage[usage_key][resource_type]
            
            # Reset usage if period expired
            self._reset_usage_if_needed(usage, quota)
            
            # Check quota
            result = self._check_and_allocate(
                usage,
                quota,
                amount,
                operation_id
            )
            
            # Update usage if granted
            if result.granted:
                usage.current_usage += result.granted_amount
                usage.total_requests += 1
                usage.peak_usage = max(usage.peak_usage, usage.current_usage)
                
                # Track in history
                usage.history.append({
                    'timestamp': datetime.utcnow(),
                    'amount': result.granted_amount,
                    'total_usage': usage.current_usage
                })
                
                # Handle concurrent operations
                if resource_type == ResourceType.CONCURRENT_OPERATIONS and operation_id:
                    self._active_allocations[usage_key].add(operation_id)
            else:
                usage.rejected_requests += 1
            
            # Check warning threshold
            if usage.current_usage >= quota.limit * (quota.warning_threshold_percent / 100):
                self._send_quota_warning(entity_id, resource_type, usage, quota)
            
            return result
    
    def release_resource(
        self,
        resource_type: ResourceType,
        amount: float,
        entity_id: Optional[str] = None,
        level: QuotaLevel = QuotaLevel.TENANT,
        operation_id: Optional[str] = None
    ):
        """Release allocated resources"""
        if not self.enabled:
            return
        
        usage_key = f"{level.value}:{entity_id or 'default'}"
        
        with self._usage_lock:
            if usage_key in self._usage and resource_type in self._usage[usage_key]:
                usage = self._usage[usage_key][resource_type]
                usage.current_usage = max(0, usage.current_usage - amount)
                
                # Handle concurrent operations
                if resource_type == ResourceType.CONCURRENT_OPERATIONS and operation_id:
                    self._active_allocations[usage_key].discard(operation_id)
    
    def get_usage_summary(
        self,
        entity_id: Optional[str] = None,
        level: QuotaLevel = QuotaLevel.TENANT
    ) -> Dict[str, Any]:
        """Get usage summary for an entity"""
        usage_key = f"{level.value}:{entity_id or 'default'}"
        policy = self._get_applicable_policy(entity_id, level)
        
        summary = {
            "entity_id": entity_id,
            "level": level.value,
            "timestamp": datetime.utcnow().isoformat(),
            "resources": {}
        }
        
        with self._usage_lock:
            for resource_type in ResourceType:
                usage = None
                quota = None
                
                if usage_key in self._usage and resource_type in self._usage[usage_key]:
                    usage = self._usage[usage_key][resource_type]
                
                if policy and resource_type in policy.quotas:
                    quota = policy.quotas[resource_type]
                
                if usage or quota:
                    resource_summary = {
                        "current_usage": usage.current_usage if usage else 0,
                        "limit": quota.limit if quota else None,
                        "period_seconds": quota.period_seconds if quota else None,
                        "usage_percent": (usage.current_usage / quota.limit * 100) if usage and quota and quota.limit > 0 else 0,
                        "peak_usage": usage.peak_usage if usage else 0,
                        "total_requests": usage.total_requests if usage else 0,
                        "rejected_requests": usage.rejected_requests if usage else 0,
                        "enforcement_mode": quota.enforcement_mode.value if quota else None
                    }
                    
                    summary["resources"][resource_type.value] = resource_summary
        
        return summary
    
    def set_quota(
        self,
        entity_id: Optional[str],
        level: QuotaLevel,
        resource_type: ResourceType,
        limit: float,
        period_seconds: int = 3600,
        enforcement_mode: QuotaEnforcementMode = QuotaEnforcementMode.STRICT,
        burst_limit: Optional[float] = None,
        burst_duration_seconds: Optional[int] = None
    ) -> bool:
        """Set or update a quota for an entity"""
        policy_id = f"{level.value}:{entity_id or 'default'}"
        
        with self._policy_lock:
            if policy_id not in self._policies:
                # Create new policy
                policy = QuotaPolicy(
                    policy_id=policy_id,
                    name=f"Policy for {entity_id or 'default'}",
                    level=level,
                    entity_id=entity_id,
                    priority=20  # User-defined policies have higher priority
                )
                self._policies[policy_id] = policy
            else:
                policy = self._policies[policy_id]
            
            # Set quota
            policy.quotas[resource_type] = ResourceQuota(
                resource_type=resource_type,
                limit=limit,
                period_seconds=period_seconds,
                enforcement_mode=enforcement_mode,
                burst_limit=burst_limit,
                burst_duration_seconds=burst_duration_seconds
            )
            
            # Save policy
            self._save_policy(policy)
            
            logger.info(f"Set quota for {entity_id}: {resource_type.value} = {limit}")
            return True
    
    def remove_quota(
        self,
        entity_id: Optional[str],
        level: QuotaLevel,
        resource_type: ResourceType
    ) -> bool:
        """Remove a specific quota"""
        policy_id = f"{level.value}:{entity_id or 'default'}"
        
        with self._policy_lock:
            if policy_id in self._policies:
                policy = self._policies[policy_id]
                if resource_type in policy.quotas:
                    del policy.quotas[resource_type]
                    self._save_policy(policy)
                    logger.info(f"Removed quota for {entity_id}: {resource_type.value}")
                    return True
        
        return False
    
    def get_quota_recommendations(
        self,
        entity_id: Optional[str],
        level: QuotaLevel,
        lookback_days: int = 7
    ) -> Dict[ResourceType, Dict[str, Any]]:
        """Get quota recommendations based on historical usage"""
        usage_key = f"{level.value}:{entity_id or 'default'}"
        recommendations = {}
        
        with self._usage_lock:
            if usage_key not in self._usage:
                return recommendations
            
            for resource_type, usage in self._usage[usage_key].items():
                if not usage.history:
                    continue
                
                # Analyze usage patterns
                usage_values = [h['total_usage'] for h in usage.history]
                if usage_values:
                    avg_usage = sum(usage_values) / len(usage_values)
                    max_usage = max(usage_values)
                    
                    # Recommend quota based on usage patterns
                    recommended_limit = max_usage * 1.5  # 50% headroom
                    recommended_burst = max_usage * 2.0  # 100% burst capacity
                    
                    recommendations[resource_type] = {
                        "current_limit": self._get_current_limit(entity_id, level, resource_type),
                        "recommended_limit": recommended_limit,
                        "recommended_burst": recommended_burst,
                        "average_usage": avg_usage,
                        "peak_usage": max_usage,
                        "utilization_percent": (avg_usage / self._get_current_limit(entity_id, level, resource_type) * 100) if self._get_current_limit(entity_id, level, resource_type) > 0 else 0
                    }
        
        return recommendations
    
    def create_rate_limiter_from_quotas(
        self,
        entity_id: Optional[str],
        level: QuotaLevel = QuotaLevel.TENANT
    ) -> HealingRateLimiter:
        """Create a HealingRateLimiter configured from quotas"""
        policy = self._get_applicable_policy(entity_id, level)
        
        config = {
            'enabled': True,
            'limits': {}
        }
        
        if policy:
            # Map quota types to rate limiter types
            mapping = {
                ResourceType.HEALING_CYCLES: 'healing_cycle',
                ResourceType.PATCHES: 'patch_application',
                ResourceType.DEPLOYMENTS: 'deployment',
                ResourceType.FILE_OPERATIONS: 'file'
            }
            
            for resource_type, rate_limit_type in mapping.items():
                if resource_type in policy.quotas:
                    quota = policy.quotas[resource_type]
                    config['limits'][rate_limit_type] = (
                        int(quota.limit),
                        quota.period_seconds
                    )
        
        return HealingRateLimiter(config)
    
    def _get_applicable_policy(
        self,
        entity_id: Optional[str],
        level: QuotaLevel
    ) -> Optional[QuotaPolicy]:
        """Get the most applicable policy for an entity"""
        with self._policy_lock:
            # First try exact match
            policy_id = f"{level.value}:{entity_id or 'default'}"
            if policy_id in self._policies:
                return self._policies[policy_id]
            
            # Then try inherited policies
            # For tenants, check tier-based policies
            if level == QuotaLevel.TENANT and entity_id:
                mt_manager = get_multi_tenancy_manager()
                if mt_manager:
                    tenant = mt_manager.get_tenant(entity_id)
                    if tenant:
                        tier_policy_id = f"tenant_tier_{tenant.tier.value.lower()}"
                        if tier_policy_id in self._policies:
                            return self._policies[tier_policy_id]
            
            # Finally, fall back to global default
            return self._policies.get("global_default")
    
    def _check_and_allocate(
        self,
        usage: ResourceUsage,
        quota: ResourceQuota,
        amount: float,
        operation_id: Optional[str]
    ) -> QuotaAllocation:
        """Check quota and allocate if possible"""
        remaining = quota.limit - usage.current_usage
        
        # Handle different enforcement modes
        if quota.enforcement_mode == QuotaEnforcementMode.STRICT:
            if amount <= remaining:
                return QuotaAllocation(
                    granted=True,
                    resource_type=quota.resource_type,
                    requested_amount=amount,
                    granted_amount=amount,
                    remaining_quota=remaining - amount
                )
            else:
                return QuotaAllocation(
                    granted=False,
                    resource_type=quota.resource_type,
                    requested_amount=amount,
                    granted_amount=0,
                    remaining_quota=remaining,
                    reason=f"Quota exceeded: {amount} requested, {remaining} available"
                )
        
        elif quota.enforcement_mode == QuotaEnforcementMode.SOFT:
            # Allow overage with warning
            granted_amount = amount
            if amount > remaining:
                logger.warning(f"Soft quota exceeded: {amount} requested, {remaining} available")
            
            return QuotaAllocation(
                granted=True,
                resource_type=quota.resource_type,
                requested_amount=amount,
                granted_amount=granted_amount,
                remaining_quota=max(0, remaining - amount)
            )
        
        elif quota.enforcement_mode == QuotaEnforcementMode.BURST:
            # Check burst capacity
            if amount <= remaining:
                return QuotaAllocation(
                    granted=True,
                    resource_type=quota.resource_type,
                    requested_amount=amount,
                    granted_amount=amount,
                    remaining_quota=remaining - amount
                )
            elif quota.burst_limit and quota.burst_duration_seconds:
                # Check if we can use burst
                now = datetime.utcnow()
                
                # Reset burst if window expired
                if usage.burst_start and (now - usage.burst_start).seconds > quota.burst_duration_seconds:
                    usage.burst_usage = 0
                    usage.burst_start = None
                
                # Start burst window if needed
                if not usage.burst_start:
                    usage.burst_start = now
                
                burst_remaining = quota.burst_limit - usage.burst_usage
                total_available = remaining + burst_remaining
                
                if amount <= total_available:
                    # Use regular quota first, then burst
                    from_regular = min(amount, remaining)
                    from_burst = amount - from_regular
                    
                    usage.burst_usage += from_burst
                    
                    return QuotaAllocation(
                        granted=True,
                        resource_type=quota.resource_type,
                        requested_amount=amount,
                        granted_amount=amount,
                        remaining_quota=remaining - from_regular,
                        reason=f"Using burst capacity: {from_burst} from burst"
                    )
                else:
                    return QuotaAllocation(
                        granted=False,
                        resource_type=quota.resource_type,
                        requested_amount=amount,
                        granted_amount=0,
                        remaining_quota=remaining,
                        reason=f"Burst capacity exceeded: {amount} requested, {total_available} available"
                    )
            else:
                return QuotaAllocation(
                    granted=False,
                    resource_type=quota.resource_type,
                    requested_amount=amount,
                    granted_amount=0,
                    remaining_quota=remaining,
                    reason=f"Quota exceeded and no burst configured"
                )
        
        elif quota.enforcement_mode == QuotaEnforcementMode.THROTTLE:
            # Grant partial amount to stay within quota
            granted_amount = min(amount, remaining)
            
            if granted_amount < amount:
                # Calculate retry delay based on quota replenishment
                retry_after = self._calculate_retry_after(quota, amount - granted_amount)
                
                return QuotaAllocation(
                    granted=True,
                    resource_type=quota.resource_type,
                    requested_amount=amount,
                    granted_amount=granted_amount,
                    remaining_quota=0,
                    reason=f"Throttled: {granted_amount} of {amount} granted",
                    retry_after_seconds=retry_after,
                    suggestions=["Reduce request rate", "Request smaller amounts"]
                )
            else:
                return QuotaAllocation(
                    granted=True,
                    resource_type=quota.resource_type,
                    requested_amount=amount,
                    granted_amount=granted_amount,
                    remaining_quota=remaining - granted_amount
                )
        
        # Default deny
        return QuotaAllocation(
            granted=False,
            resource_type=quota.resource_type,
            requested_amount=amount,
            granted_amount=0,
            remaining_quota=remaining,
            reason="Unknown enforcement mode"
        )
    
    def _reset_usage_if_needed(self, usage: ResourceUsage, quota: ResourceQuota):
        """Reset usage if quota period has expired"""
        now = datetime.utcnow()
        elapsed = (now - usage.period_start).total_seconds()
        
        if elapsed >= quota.period_seconds:
            usage.current_usage = 0
            usage.period_start = now
            usage.burst_usage = 0
            usage.burst_start = None
    
    def _calculate_retry_after(self, quota: ResourceQuota, needed_amount: float) -> int:
        """Calculate seconds until enough quota is available"""
        # Simple calculation: assume linear replenishment
        if quota.limit <= 0:
            return quota.period_seconds
        
        replenishment_rate = quota.limit / quota.period_seconds
        seconds_needed = needed_amount / replenishment_rate
        
        return max(1, int(seconds_needed))
    
    def _get_current_limit(
        self,
        entity_id: Optional[str],
        level: QuotaLevel,
        resource_type: ResourceType
    ) -> float:
        """Get current limit for a resource"""
        policy = self._get_applicable_policy(entity_id, level)
        if policy and resource_type in policy.quotas:
            return policy.quotas[resource_type].limit
        return float('inf')
    
    def _send_quota_warning(
        self,
        entity_id: Optional[str],
        resource_type: ResourceType,
        usage: ResourceUsage,
        quota: ResourceQuota
    ):
        """Send warning when approaching quota limit"""
        hooks = get_observability_hooks()
        if hooks:
            usage_percent = (usage.current_usage / quota.limit * 100) if quota.limit > 0 else 0
            
            hooks._send_event(
                title=f"Quota Warning: {resource_type.value}",
                text=f"Entity {entity_id or 'default'} at {usage_percent:.1f}% of quota",
                event_type="quota_warning",
                tags={
                    "entity_id": entity_id or "default",
                    "resource_type": resource_type.value,
                    "usage_percent": str(usage_percent),
                    "current_usage": str(usage.current_usage),
                    "limit": str(quota.limit)
                },
                priority="high" if usage_percent >= 90 else "medium"
            )
    
    def _monitor_usage(self):
        """Background thread to monitor usage and enforce quotas"""
        while True:
            try:
                time.sleep(60)  # Check every minute
                
                # Clean up expired allocations
                with self._usage_lock:
                    for usage_key, allocations in list(self._active_allocations.items()):
                        # Remove stale allocations (older than 1 hour)
                        # In production, track allocation timestamps
                        pass
                
                # Check for quota violations
                self._check_quota_violations()
                
                # Save usage data periodically
                self._save_usage_data()
                
            except Exception as e:
                logger.error(f"Error in quota monitoring: {e}")
    
    def _check_quota_violations(self):
        """Check for any quota violations and take action"""
        with self._usage_lock:
            for usage_key, resources in self._usage.items():
                level, entity_id = usage_key.split(':', 1)
                entity_id = entity_id if entity_id != 'default' else None
                
                policy = self._get_applicable_policy(entity_id, QuotaLevel(level))
                if not policy:
                    continue
                
                for resource_type, usage in resources.items():
                    if resource_type not in policy.quotas:
                        continue
                    
                    quota = policy.quotas[resource_type]
                    
                    # Check for violations
                    if usage.current_usage > quota.limit * 1.1:  # 10% grace
                        logger.error(
                            f"Quota violation detected: {usage_key} {resource_type.value} "
                            f"using {usage.current_usage}/{quota.limit}"
                        )
                        
                        # Take action based on enforcement mode
                        if quota.enforcement_mode == QuotaEnforcementMode.STRICT:
                            # Force release resources
                            # In production, implement resource reclamation
                            pass
    
    def _save_policy(self, policy: QuotaPolicy):
        """Save policy to disk"""
        policy_file = self.storage_path / f"policy_{policy.policy_id.replace(':', '_')}.json"
        
        with open(policy_file, 'w') as f:
            json.dump({
                'policy_id': policy.policy_id,
                'name': policy.name,
                'level': policy.level.value,
                'entity_id': policy.entity_id,
                'enabled': policy.enabled,
                'priority': policy.priority,
                'inherit_from': policy.inherit_from,
                'metadata': policy.metadata,
                'quotas': {
                    rt.value: {
                        'limit': q.limit,
                        'period_seconds': q.period_seconds,
                        'enforcement_mode': q.enforcement_mode.value,
                        'burst_limit': q.burst_limit,
                        'burst_duration_seconds': q.burst_duration_seconds,
                        'warning_threshold_percent': q.warning_threshold_percent,
                        'metadata': q.metadata
                    }
                    for rt, q in policy.quotas.items()
                }
            }, f, indent=2)
    
    def _load_policies(self):
        """Load policies from disk"""
        for policy_file in self.storage_path.glob("policy_*.json"):
            try:
                with open(policy_file, 'r') as f:
                    data = json.load(f)
                
                policy = QuotaPolicy(
                    policy_id=data['policy_id'],
                    name=data['name'],
                    level=QuotaLevel(data['level']),
                    entity_id=data.get('entity_id'),
                    enabled=data.get('enabled', True),
                    priority=data.get('priority', 0),
                    inherit_from=data.get('inherit_from'),
                    metadata=data.get('metadata', {})
                )
                
                # Load quotas
                for rt_str, quota_data in data.get('quotas', {}).items():
                    resource_type = ResourceType(rt_str)
                    policy.quotas[resource_type] = ResourceQuota(
                        resource_type=resource_type,
                        limit=quota_data['limit'],
                        period_seconds=quota_data['period_seconds'],
                        enforcement_mode=QuotaEnforcementMode(quota_data['enforcement_mode']),
                        burst_limit=quota_data.get('burst_limit'),
                        burst_duration_seconds=quota_data.get('burst_duration_seconds'),
                        warning_threshold_percent=quota_data.get('warning_threshold_percent', 80.0),
                        metadata=quota_data.get('metadata', {})
                    )
                
                self._policies[policy.policy_id] = policy
                
            except Exception as e:
                logger.error(f"Failed to load policy from {policy_file}: {e}")
    
    def _save_usage_data(self):
        """Save usage data for persistence and analytics"""
        usage_file = self.storage_path / f"usage_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        
        with self._usage_lock:
            usage_data = {}
            
            for usage_key, resources in self._usage.items():
                usage_data[usage_key] = {}
                
                for resource_type, usage in resources.items():
                    usage_data[usage_key][resource_type.value] = {
                        'current_usage': usage.current_usage,
                        'period_start': usage.period_start.isoformat(),
                        'peak_usage': usage.peak_usage,
                        'total_requests': usage.total_requests,
                        'rejected_requests': usage.rejected_requests,
                        'burst_usage': usage.burst_usage,
                        'burst_start': usage.burst_start.isoformat() if usage.burst_start else None
                    }
            
            with open(usage_file, 'w') as f:
                json.dump(usage_data, f, indent=2)


# Context manager for resource allocation
@contextmanager
def allocate_resources(
    quota_manager: ResourceQuotaManager,
    allocations: List[Tuple[ResourceType, float]],
    entity_id: Optional[str] = None,
    level: QuotaLevel = QuotaLevel.TENANT
):
    """
    Context manager for allocating and releasing resources.
    
    Usage:
        with allocate_resources(manager, [(ResourceType.CPU_SECONDS, 100)]) as results:
            if all(r.granted for r in results):
                # Use resources
                pass
    """
    results = []
    allocated = []
    
    try:
        # Allocate all resources
        for resource_type, amount in allocations:
            result = quota_manager.allocate_resource(
                resource_type,
                amount,
                entity_id,
                level
            )
            results.append(result)
            
            if result.granted:
                allocated.append((resource_type, result.granted_amount))
            else:
                # Rollback previous allocations
                for rt, amt in allocated:
                    quota_manager.release_resource(rt, amt, entity_id, level)
                
                # Raise exception with details
                raise HealingRateLimitExceededError(
                    f"Failed to allocate {resource_type.value}: {result.reason}"
                )
        
        yield results
        
    finally:
        # Release all allocated resources
        for resource_type, amount in allocated:
            quota_manager.release_resource(resource_type, amount, entity_id, level)


# Global instance management
_resource_quota_manager = None

def init_resource_quotas(config: Dict[str, Any]) -> ResourceQuotaManager:
    """Initialize the global resource quota manager"""
    global _resource_quota_manager
    _resource_quota_manager = ResourceQuotaManager(config)
    return _resource_quota_manager

def get_resource_quota_manager() -> Optional[ResourceQuotaManager]:
    """Get the global resource quota manager instance"""
    return _resource_quota_manager