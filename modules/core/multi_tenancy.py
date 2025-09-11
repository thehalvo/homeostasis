"""
Multi-Tenancy Support for Homeostasis Framework

This module provides comprehensive multi-tenancy capabilities:
- Tenant isolation and data segregation
- Per-tenant configuration and customization
- Resource quotas and usage tracking
- Tenant-aware healing policies
- Secure tenant context propagation
- Tenant lifecycle management
"""

import hashlib
import json
import logging
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from modules.monitoring.observability_hooks import get_observability_hooks
from modules.security.healing_rate_limiter import HealingRateLimiter

logger = logging.getLogger(__name__)


class TenantStatus(Enum):
    """Status of a tenant"""

    ACTIVE = "active"
    SUSPENDED = "suspended"
    PENDING_ACTIVATION = "pending_activation"
    PENDING_DELETION = "pending_deletion"
    DELETED = "deleted"


class TenantTier(Enum):
    """Tenant tier levels with different capabilities"""

    FREE = "free"
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    CUSTOM = "custom"


class IsolationLevel(Enum):
    """Levels of tenant isolation"""

    SHARED = "shared"  # Shared resources with logical separation
    DEDICATED = "dedicated"  # Dedicated compute resources
    ISOLATED = "isolated"  # Complete isolation with separate infrastructure


@dataclass
class TenantQuota:
    """Resource quotas for a tenant"""

    max_healing_cycles_per_hour: int = 10
    max_patches_per_day: int = 100
    max_deployments_per_day: int = 20
    max_file_modifications_per_hour: int = 50
    max_concurrent_operations: int = 5
    max_storage_mb: int = 1000
    max_cpu_seconds_per_hour: int = 3600
    max_api_calls_per_minute: int = 100
    max_users: int = 10
    custom_limits: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TenantUsage:
    """Track resource usage for a tenant"""

    healing_cycles_count: int = 0
    patches_count: int = 0
    deployments_count: int = 0
    file_modifications_count: int = 0
    storage_used_mb: float = 0
    cpu_seconds_used: float = 0
    api_calls_count: int = 0
    last_reset_time: datetime = field(default_factory=datetime.utcnow)
    period_start_time: datetime = field(default_factory=datetime.utcnow)
    custom_usage: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TenantConfiguration:
    """Tenant-specific configuration"""

    healing_enabled: bool = True
    auto_patch_enabled: bool = True
    auto_deploy_enabled: bool = False
    allowed_languages: Set[str] = field(
        default_factory=lambda: {"python", "javascript", "java"}
    )
    blocked_file_patterns: List[str] = field(default_factory=list)
    custom_rules: Dict[str, Any] = field(default_factory=dict)
    notification_webhooks: List[str] = field(default_factory=list)
    preferred_monitoring_backend: Optional[str] = None
    environment_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Tenant:
    """Represents a tenant in the system"""

    id: str
    name: str
    status: TenantStatus = TenantStatus.ACTIVE
    tier: TenantTier = TenantTier.FREE
    isolation_level: IsolationLevel = IsolationLevel.SHARED
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    quota: TenantQuota = field(default_factory=TenantQuota)
    configuration: TenantConfiguration = field(default_factory=TenantConfiguration)
    usage: TenantUsage = field(default_factory=TenantUsage)
    owner_email: Optional[str] = None
    api_keys: Set[str] = field(default_factory=set)
    tags: Set[str] = field(default_factory=set)


class TenantContext:
    """Thread-local tenant context"""

    _thread_local = threading.local()

    @classmethod
    def get_current_tenant_id(cls) -> Optional[str]:
        """Get the current tenant ID from thread-local context"""
        return getattr(cls._thread_local, "tenant_id", None)

    @classmethod
    def set_current_tenant_id(cls, tenant_id: Optional[str]):
        """Set the current tenant ID in thread-local context"""
        cls._thread_local.tenant_id = tenant_id

    @classmethod
    def clear(cls):
        """Clear the current tenant context"""
        if hasattr(cls._thread_local, "tenant_id"):
            delattr(cls._thread_local, "tenant_id")


class MultiTenancyManager:
    """
    Manages multi-tenant operations for the Homeostasis framework.

    Provides:
    - Tenant lifecycle management
    - Resource isolation and quotas
    - Tenant-aware operation execution
    - Usage tracking and billing
    - Configuration management
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the multi-tenancy manager"""
        self.config = config
        self.enabled = config.get("enabled", True)
        self.storage_path = Path(config.get("storage_path", "./data/tenants"))
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # In-memory tenant cache
        self._tenants: Dict[str, Tenant] = {}
        self._tenant_lock = threading.RLock()

        # Per-tenant rate limiters
        self._rate_limiters: Dict[str, HealingRateLimiter] = {}

        # Usage tracking
        self._usage_lock = threading.Lock()
        self._last_usage_reset = datetime.utcnow()

        # Default quotas by tier
        self.tier_quotas = {
            TenantTier.FREE: TenantQuota(
                max_healing_cycles_per_hour=5,
                max_patches_per_day=20,
                max_deployments_per_day=5,
                max_file_modifications_per_hour=10,
                max_concurrent_operations=2,
                max_storage_mb=100,
                max_cpu_seconds_per_hour=600,
                max_api_calls_per_minute=20,
                max_users=3,
            ),
            TenantTier.STARTER: TenantQuota(
                max_healing_cycles_per_hour=20,
                max_patches_per_day=100,
                max_deployments_per_day=20,
                max_file_modifications_per_hour=50,
                max_concurrent_operations=5,
                max_storage_mb=1000,
                max_cpu_seconds_per_hour=3600,
                max_api_calls_per_minute=100,
                max_users=10,
            ),
            TenantTier.PROFESSIONAL: TenantQuota(
                max_healing_cycles_per_hour=100,
                max_patches_per_day=500,
                max_deployments_per_day=100,
                max_file_modifications_per_hour=200,
                max_concurrent_operations=20,
                max_storage_mb=10000,
                max_cpu_seconds_per_hour=36000,
                max_api_calls_per_minute=500,
                max_users=50,
            ),
            TenantTier.ENTERPRISE: TenantQuota(
                max_healing_cycles_per_hour=1000,
                max_patches_per_day=5000,
                max_deployments_per_day=1000,
                max_file_modifications_per_hour=2000,
                max_concurrent_operations=100,
                max_storage_mb=100000,
                max_cpu_seconds_per_hour=360000,
                max_api_calls_per_minute=5000,
                max_users=1000,
            ),
        }

        # Load existing tenants
        self._load_tenants()

        # Start background tasks
        self._start_background_tasks()

    def create_tenant(
        self,
        name: str,
        tier: TenantTier = TenantTier.FREE,
        owner_email: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tenant:
        """Create a new tenant"""
        tenant_id = str(uuid.uuid4())

        # Get default quota for tier
        quota = self.tier_quotas.get(tier, TenantQuota())

        tenant = Tenant(
            id=tenant_id,
            name=name,
            tier=tier,
            quota=quota,
            owner_email=owner_email,
            metadata=metadata or {},
        )

        # Generate initial API key
        api_key = self._generate_api_key(tenant_id)
        tenant.api_keys.add(api_key)

        with self._tenant_lock:
            self._tenants[tenant_id] = tenant
            self._save_tenant(tenant)

            # Create tenant-specific rate limiter
            self._create_rate_limiter(tenant)

        logger.info(f"Created tenant: {tenant_id} ({name})")

        # Track tenant creation
        hooks = get_observability_hooks()
        if hooks:
            hooks._send_event(
                title=f"Tenant Created: {name}",
                text=f"Tier: {tier.value}, ID: {tenant_id}",
                event_type="tenant_created",
                tags={"tenant_id": tenant_id, "tier": tier.value},
            )

        return tenant

    def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        """Get tenant by ID"""
        with self._tenant_lock:
            return self._tenants.get(tenant_id)

    def get_tenant_by_api_key(self, api_key: str) -> Optional[Tenant]:
        """Get tenant by API key"""
        with self._tenant_lock:
            for tenant in self._tenants.values():
                if api_key in tenant.api_keys:
                    return tenant
        return None

    def update_tenant(
        self, tenant_id: str, updates: Dict[str, Any]
    ) -> Optional[Tenant]:
        """Update tenant properties"""
        with self._tenant_lock:
            tenant = self._tenants.get(tenant_id)
            if not tenant:
                return None

            # Update allowed fields
            allowed_fields = {"name", "status", "tier", "metadata", "owner_email"}
            for field_name, value in updates.items():
                if field_name in allowed_fields:
                    setattr(tenant, field_name, value)

            # Update quota if tier changed
            if "tier" in updates and updates["tier"] in self.tier_quotas:
                tenant.quota = self.tier_quotas[updates["tier"]]
                # Recreate rate limiter with new limits
                self._create_rate_limiter(tenant)

            tenant.updated_at = datetime.utcnow()
            self._save_tenant(tenant)

            return tenant

    def delete_tenant(self, tenant_id: str) -> bool:
        """Delete a tenant (soft delete by default)"""
        with self._tenant_lock:
            tenant = self._tenants.get(tenant_id)
            if not tenant:
                return False

            # Soft delete - mark as deleted
            tenant.status = TenantStatus.DELETED
            tenant.updated_at = datetime.utcnow()
            self._save_tenant(tenant)

            # Remove from active tenants
            del self._tenants[tenant_id]

            # Remove rate limiter
            if tenant_id in self._rate_limiters:
                del self._rate_limiters[tenant_id]

            logger.info(f"Deleted tenant: {tenant_id}")
            return True

    def suspend_tenant(self, tenant_id: str, reason: str) -> bool:
        """Suspend a tenant"""
        return (
            self.update_tenant(
                tenant_id,
                {
                    "status": TenantStatus.SUSPENDED,
                    "metadata": {
                        "suspension_reason": reason,
                        "suspended_at": datetime.utcnow().isoformat(),
                    },
                },
            )
            is not None
        )

    def reactivate_tenant(self, tenant_id: str) -> bool:
        """Reactivate a suspended tenant"""
        return (
            self.update_tenant(tenant_id, {"status": TenantStatus.ACTIVE}) is not None
        )

    @contextmanager
    def tenant_context(self, tenant_id: str):
        """Context manager for tenant-aware operations"""
        previous_tenant = TenantContext.get_current_tenant_id()

        try:
            # Set tenant context
            TenantContext.set_current_tenant_id(tenant_id)

            # Validate tenant
            tenant = self.get_tenant(tenant_id)
            if not tenant:
                raise ValueError(f"Invalid tenant ID: {tenant_id}")

            if tenant.status != TenantStatus.ACTIVE:
                raise ValueError(
                    f"Tenant {tenant_id} is not active: {tenant.status.value}"
                )

            yield tenant

        finally:
            # Restore previous context
            if previous_tenant:
                TenantContext.set_current_tenant_id(previous_tenant)
            else:
                TenantContext.clear()

    def check_quota(
        self, tenant_id: str, resource_type: str, requested_amount: int = 1
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if tenant has quota for the requested resource.

        Returns:
            Tuple of (allowed, reason_if_denied)
        """
        tenant = self.get_tenant(tenant_id)
        if not tenant:
            return False, "Invalid tenant"

        if tenant.status != TenantStatus.ACTIVE:
            return False, f"Tenant is {tenant.status.value}"

        # Reset usage if needed
        self._reset_usage_if_needed(tenant)

        # Check specific resource types
        if resource_type == "healing_cycle":
            current = tenant.usage.healing_cycles_count
            limit = tenant.quota.max_healing_cycles_per_hour
            if current + requested_amount > limit:
                return (
                    False,
                    f"Healing cycle quota exceeded: {current}/{limit} per hour",
                )

        elif resource_type == "patch":
            current = tenant.usage.patches_count
            limit = tenant.quota.max_patches_per_day
            if current + requested_amount > limit:
                return False, f"Patch quota exceeded: {current}/{limit} per day"

        elif resource_type == "deployment":
            current = tenant.usage.deployments_count
            limit = tenant.quota.max_deployments_per_day
            if current + requested_amount > limit:
                return False, f"Deployment quota exceeded: {current}/{limit} per day"

        elif resource_type == "file_modification":
            current = tenant.usage.file_modifications_count
            limit = tenant.quota.max_file_modifications_per_hour
            if current + requested_amount > limit:
                return (
                    False,
                    f"File modification quota exceeded: {current}/{limit} per hour",
                )

        elif resource_type == "api_call":
            current = tenant.usage.api_calls_count
            limit = tenant.quota.max_api_calls_per_minute
            # Check if within the last minute
            if (datetime.utcnow() - tenant.usage.last_reset_time).seconds < 60:
                if current + requested_amount > limit:
                    return (
                        False,
                        f"API call quota exceeded: {current}/{limit} per minute",
                    )

        elif resource_type == "storage":
            current = tenant.usage.storage_used_mb
            limit = tenant.quota.max_storage_mb
            if current + requested_amount > limit:
                return False, f"Storage quota exceeded: {current}/{limit} MB"

        elif resource_type == "cpu_seconds":
            current = tenant.usage.cpu_seconds_used
            limit = tenant.quota.max_cpu_seconds_per_hour
            if current + requested_amount > limit:
                return False, f"CPU quota exceeded: {current}/{limit} seconds per hour"

        # Check custom limits
        elif resource_type in tenant.quota.custom_limits:
            current = tenant.usage.custom_usage.get(resource_type, 0)
            limit = tenant.quota.custom_limits[resource_type]
            if current + requested_amount > limit:
                return (
                    False,
                    f"Custom quota '{resource_type}' exceeded: {current}/{limit}",
                )

        return True, None

    def track_usage(
        self, tenant_id: str, resource_type: str, amount: float = 1.0
    ) -> bool:
        """Track resource usage for a tenant"""
        tenant = self.get_tenant(tenant_id)
        if not tenant:
            return False

        with self._usage_lock:
            # Reset usage if needed
            self._reset_usage_if_needed(tenant)

            # Update usage counters
            if resource_type == "healing_cycle":
                tenant.usage.healing_cycles_count += int(amount)
            elif resource_type == "patch":
                tenant.usage.patches_count += int(amount)
            elif resource_type == "deployment":
                tenant.usage.deployments_count += int(amount)
            elif resource_type == "file_modification":
                tenant.usage.file_modifications_count += int(amount)
            elif resource_type == "api_call":
                tenant.usage.api_calls_count += int(amount)
            elif resource_type == "storage":
                tenant.usage.storage_used_mb = (
                    amount  # Storage is absolute, not incremental
                )
            elif resource_type == "cpu_seconds":
                tenant.usage.cpu_seconds_used += amount
            else:
                # Track custom usage
                if resource_type not in tenant.usage.custom_usage:
                    tenant.usage.custom_usage[resource_type] = 0
                tenant.usage.custom_usage[resource_type] += amount

            # Save updated usage
            self._save_tenant(tenant)

        return True

    def get_tenant_rate_limiter(self, tenant_id: str) -> Optional[HealingRateLimiter]:
        """Get the rate limiter for a specific tenant"""
        tenant = self.get_tenant(tenant_id)
        if not tenant:
            return None

        if tenant_id not in self._rate_limiters:
            self._create_rate_limiter(tenant)

        return self._rate_limiters.get(tenant_id)

    def get_tenant_storage_path(self, tenant_id: str) -> Path:
        """Get the storage path for a tenant"""
        tenant_path = self.storage_path / tenant_id
        tenant_path.mkdir(parents=True, exist_ok=True)
        return tenant_path

    def list_tenants(
        self,
        status: Optional[TenantStatus] = None,
        tier: Optional[TenantTier] = None,
        tags: Optional[Set[str]] = None,
    ) -> List[Tenant]:
        """List tenants with optional filters"""
        with self._tenant_lock:
            tenants = list(self._tenants.values())

            if status:
                tenants = [t for t in tenants if t.status == status]

            if tier:
                tenants = [t for t in tenants if t.tier == tier]

            if tags:
                tenants = [t for t in tenants if tags.issubset(t.tags)]

            return tenants

    def get_usage_summary(self, tenant_id: str) -> Dict[str, Any]:
        """Get usage summary for a tenant"""
        tenant = self.get_tenant(tenant_id)
        if not tenant:
            return {}

        # Calculate usage percentages
        usage_pct = {
            "healing_cycles": (
                tenant.usage.healing_cycles_count /
                tenant.quota.max_healing_cycles_per_hour *
                100
            ),
            "patches": (
                tenant.usage.patches_count / tenant.quota.max_patches_per_day * 100
            ),
            "deployments": (
                tenant.usage.deployments_count /
                tenant.quota.max_deployments_per_day *
                100
            ),
            "file_modifications": (
                tenant.usage.file_modifications_count /
                tenant.quota.max_file_modifications_per_hour *
                100
            ),
            "storage": (
                tenant.usage.storage_used_mb / tenant.quota.max_storage_mb * 100
            ),
            "cpu_seconds": (
                tenant.usage.cpu_seconds_used /
                tenant.quota.max_cpu_seconds_per_hour *
                100
            ),
            "api_calls": (
                tenant.usage.api_calls_count /
                tenant.quota.max_api_calls_per_minute *
                100
            ),
        }

        return {
            "tenant_id": tenant_id,
            "tenant_name": tenant.name,
            "tier": tenant.tier.value,
            "status": tenant.status.value,
            "usage": {
                "healing_cycles": {
                    "used": tenant.usage.healing_cycles_count,
                    "limit": tenant.quota.max_healing_cycles_per_hour,
                    "percentage": usage_pct["healing_cycles"],
                },
                "patches": {
                    "used": tenant.usage.patches_count,
                    "limit": tenant.quota.max_patches_per_day,
                    "percentage": usage_pct["patches"],
                },
                "deployments": {
                    "used": tenant.usage.deployments_count,
                    "limit": tenant.quota.max_deployments_per_day,
                    "percentage": usage_pct["deployments"],
                },
                "storage_mb": {
                    "used": tenant.usage.storage_used_mb,
                    "limit": tenant.quota.max_storage_mb,
                    "percentage": usage_pct["storage"],
                },
                "cpu_seconds": {
                    "used": tenant.usage.cpu_seconds_used,
                    "limit": tenant.quota.max_cpu_seconds_per_hour,
                    "percentage": usage_pct["cpu_seconds"],
                },
            },
            "period_start": tenant.usage.period_start_time.isoformat(),
            "last_reset": tenant.usage.last_reset_time.isoformat(),
        }

    def _generate_api_key(self, tenant_id: str) -> str:
        """Generate a secure API key for a tenant"""
        # Include tenant ID and random data
        data = f"{tenant_id}:{uuid.uuid4()}:{time.time()}"
        # Create hash
        return f"hom_{hashlib.sha256(data.encode()).hexdigest()[:32]}"

    def _create_rate_limiter(self, tenant: Tenant):
        """Create a tenant-specific rate limiter"""
        config = {
            "enabled": True,
            "limits": {
                "healing_cycle": (tenant.quota.max_healing_cycles_per_hour, 3600),
                "patch_application": (tenant.quota.max_patches_per_day, 86400),
                "deployment": (tenant.quota.max_deployments_per_day, 86400),
                "file": (tenant.quota.max_file_modifications_per_hour, 3600),
                "critical_file": (1, 3600),  # Always limit critical files
            },
        }

        self._rate_limiters[tenant.id] = HealingRateLimiter(config)

    def _reset_usage_if_needed(self, tenant: Tenant):
        """Reset usage counters based on time windows"""
        now = datetime.utcnow()

        # Reset hourly counters
        if (now - tenant.usage.last_reset_time).seconds >= 3600:
            tenant.usage.healing_cycles_count = 0
            tenant.usage.file_modifications_count = 0
            tenant.usage.cpu_seconds_used = 0
            tenant.usage.last_reset_time = now

        # Reset daily counters
        if (now - tenant.usage.period_start_time).days >= 1:
            tenant.usage.patches_count = 0
            tenant.usage.deployments_count = 0
            tenant.usage.period_start_time = now

        # Reset minute counters (for API calls)
        if (now - tenant.usage.last_reset_time).seconds >= 60:
            tenant.usage.api_calls_count = 0

    def _save_tenant(self, tenant: Tenant):
        """Save tenant to persistent storage"""
        tenant_file = self.storage_path / f"{tenant.id}.json"

        # Convert to dict for serialization
        tenant_dict = {
            "id": tenant.id,
            "name": tenant.name,
            "status": tenant.status.value,
            "tier": tenant.tier.value,
            "isolation_level": tenant.isolation_level.value,
            "created_at": tenant.created_at.isoformat(),
            "updated_at": tenant.updated_at.isoformat(),
            "metadata": tenant.metadata,
            "owner_email": tenant.owner_email,
            "api_keys": list(tenant.api_keys),
            "tags": list(tenant.tags),
            "quota": {
                "max_healing_cycles_per_hour": tenant.quota.max_healing_cycles_per_hour,
                "max_patches_per_day": tenant.quota.max_patches_per_day,
                "max_deployments_per_day": tenant.quota.max_deployments_per_day,
                "max_file_modifications_per_hour": tenant.quota.max_file_modifications_per_hour,
                "max_concurrent_operations": tenant.quota.max_concurrent_operations,
                "max_storage_mb": tenant.quota.max_storage_mb,
                "max_cpu_seconds_per_hour": tenant.quota.max_cpu_seconds_per_hour,
                "max_api_calls_per_minute": tenant.quota.max_api_calls_per_minute,
                "max_users": tenant.quota.max_users,
                "custom_limits": tenant.quota.custom_limits,
            },
            "configuration": {
                "healing_enabled": tenant.configuration.healing_enabled,
                "auto_patch_enabled": tenant.configuration.auto_patch_enabled,
                "auto_deploy_enabled": tenant.configuration.auto_deploy_enabled,
                "allowed_languages": list(tenant.configuration.allowed_languages),
                "blocked_file_patterns": tenant.configuration.blocked_file_patterns,
                "custom_rules": tenant.configuration.custom_rules,
                "notification_webhooks": tenant.configuration.notification_webhooks,
                "preferred_monitoring_backend": tenant.configuration.preferred_monitoring_backend,
                "environment_config": tenant.configuration.environment_config,
            },
            "usage": {
                "healing_cycles_count": tenant.usage.healing_cycles_count,
                "patches_count": tenant.usage.patches_count,
                "deployments_count": tenant.usage.deployments_count,
                "file_modifications_count": tenant.usage.file_modifications_count,
                "storage_used_mb": tenant.usage.storage_used_mb,
                "cpu_seconds_used": tenant.usage.cpu_seconds_used,
                "api_calls_count": tenant.usage.api_calls_count,
                "last_reset_time": tenant.usage.last_reset_time.isoformat(),
                "period_start_time": tenant.usage.period_start_time.isoformat(),
                "custom_usage": tenant.usage.custom_usage,
            },
        }

        with open(tenant_file, "w") as f:
            json.dump(tenant_dict, f, indent=2)

    def _load_tenants(self):
        """Load tenants from persistent storage"""
        for tenant_file in self.storage_path.glob("*.json"):
            try:
                with open(tenant_file, "r") as f:
                    data = json.load(f)

                # Skip deleted tenants
                if data.get("status") == TenantStatus.DELETED.value:
                    continue

                # Reconstruct tenant object
                tenant = Tenant(
                    id=data["id"],
                    name=data["name"],
                    status=TenantStatus(data["status"]),
                    tier=TenantTier(data["tier"]),
                    isolation_level=IsolationLevel(
                        data.get("isolation_level", "shared")
                    ),
                    created_at=datetime.fromisoformat(data["created_at"]),
                    updated_at=datetime.fromisoformat(data["updated_at"]),
                    metadata=data.get("metadata", {}),
                    owner_email=data.get("owner_email"),
                    api_keys=set(data.get("api_keys", [])),
                    tags=set(data.get("tags", [])),
                )

                # Load quota
                quota_data = data.get("quota", {})
                tenant.quota = TenantQuota(**quota_data)

                # Load configuration
                config_data = data.get("configuration", {})
                tenant.configuration = TenantConfiguration(
                    healing_enabled=config_data.get("healing_enabled", True),
                    auto_patch_enabled=config_data.get("auto_patch_enabled", True),
                    auto_deploy_enabled=config_data.get("auto_deploy_enabled", False),
                    allowed_languages=set(config_data.get("allowed_languages", [])),
                    blocked_file_patterns=config_data.get("blocked_file_patterns", []),
                    custom_rules=config_data.get("custom_rules", {}),
                    notification_webhooks=config_data.get("notification_webhooks", []),
                    preferred_monitoring_backend=config_data.get(
                        "preferred_monitoring_backend"
                    ),
                    environment_config=config_data.get("environment_config", {}),
                )

                # Load usage
                usage_data = data.get("usage", {})
                tenant.usage = TenantUsage(
                    healing_cycles_count=usage_data.get("healing_cycles_count", 0),
                    patches_count=usage_data.get("patches_count", 0),
                    deployments_count=usage_data.get("deployments_count", 0),
                    file_modifications_count=usage_data.get(
                        "file_modifications_count", 0
                    ),
                    storage_used_mb=usage_data.get("storage_used_mb", 0),
                    cpu_seconds_used=usage_data.get("cpu_seconds_used", 0),
                    api_calls_count=usage_data.get("api_calls_count", 0),
                    last_reset_time=datetime.fromisoformat(
                        usage_data.get("last_reset_time", datetime.utcnow().isoformat())
                    ),
                    period_start_time=datetime.fromisoformat(
                        usage_data.get(
                            "period_start_time", datetime.utcnow().isoformat()
                        )
                    ),
                    custom_usage=usage_data.get("custom_usage", {}),
                )

                self._tenants[tenant.id] = tenant

                # Create rate limiter
                self._create_rate_limiter(tenant)

                logger.info(f"Loaded tenant: {tenant.id} ({tenant.name})")

            except Exception as e:
                logger.error(f"Failed to load tenant from {tenant_file}: {e}")

    def _start_background_tasks(self):
        """Start background maintenance tasks"""
        # This could include:
        # - Usage reset schedulers
        # - Tenant health checks
        # - Storage cleanup
        # - Quota enforcement
        pass


# Global instance management
_multi_tenancy_manager = None


def init_multi_tenancy(config: Dict[str, Any]) -> MultiTenancyManager:
    """Initialize the global multi-tenancy manager"""
    global _multi_tenancy_manager
    _multi_tenancy_manager = MultiTenancyManager(config)
    return _multi_tenancy_manager


def get_multi_tenancy_manager() -> Optional[MultiTenancyManager]:
    """Get the global multi-tenancy manager instance"""
    return _multi_tenancy_manager


def get_current_tenant_id() -> Optional[str]:
    """Get the current tenant ID from context"""
    return TenantContext.get_current_tenant_id()


def require_tenant(func: Callable) -> Callable:
    """Decorator to ensure a tenant context is present"""

    def wrapper(*args, **kwargs):
        tenant_id = get_current_tenant_id()
        if not tenant_id:
            raise ValueError(
                "No tenant context found. Operations must be executed within a tenant context."
            )

        # Add tenant_id to kwargs if not present
        if "tenant_id" not in kwargs:
            kwargs["tenant_id"] = tenant_id

        return func(*args, **kwargs)

    return wrapper
