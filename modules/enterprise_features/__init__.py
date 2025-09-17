"""
Enterprise Features Module

Provides enterprise-grade features for the Homeostasis self-healing system including:
- SAML/SSO authentication
- Advanced RBAC with custom roles
- Enhanced compliance reporting (SOC2, HIPAA)
- SLA monitoring and alerting
- Enhanced multi-region failover support
"""

from typing import Any, Dict, Optional

from modules.deployment.multi_environment.multi_region import (
    MultiRegionResilienceStrategy,
)
from modules.enterprise.advanced_rbac import (
    AdvancedRBACManager,
    create_advanced_rbac_manager,
)
from modules.enterprise.enhanced_compliance import (
    EnhancedComplianceReporting,
    create_enhanced_compliance,
)
from modules.enterprise.enhanced_multi_region import (
    EnhancedMultiRegionFailover,
    create_enhanced_multi_region,
)

# Import enterprise components
from modules.enterprise.saml_sso import SAMLAuthenticationManager, create_saml_manager
from modules.enterprise.sla_monitoring import SLAMonitoringSystem, create_sla_monitoring

# Import base components needed
from modules.security.auth import get_auth_manager
from modules.security.compliance_reporting import get_compliance_reporting

__all__ = [
    "EnterpriseFeatures",
    "create_enterprise_features",
    "SAMLAuthenticationManager",
    "AdvancedRBACManager",
    "EnhancedComplianceReporting",
    "SLAMonitoringSystem",
    "EnhancedMultiRegionFailover",
]


class EnterpriseFeatures:
    """
    Central manager for all enterprise features.

    Provides a unified interface to access and configure enterprise-grade
    capabilities for the Homeostasis system.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize enterprise features.

        Args:
            config: Enterprise configuration dictionary
        """
        self.config = config

        # Initialize base components
        self.auth_manager = get_auth_manager(config.get("auth", {}))
        self.base_compliance = get_compliance_reporting(config.get("compliance", {}))

        # Initialize enterprise components
        self.saml_manager: Optional[SAMLAuthenticationManager] = None
        self.advanced_rbac: Optional[AdvancedRBACManager] = None
        self.enhanced_compliance: Optional[EnhancedComplianceReporting] = None
        self.sla_monitoring: Optional[SLAMonitoringSystem] = None
        self.enhanced_multi_region: Optional[EnhancedMultiRegionFailover] = None

        # Enable components based on configuration
        self._initialize_components()

    def _initialize_components(self):
        """Initialize enabled enterprise components"""
        # SAML/SSO
        if self.config.get("saml_enabled", False):
            self.saml_manager = create_saml_manager(
                self.config.get("saml", {}), self.auth_manager
            )

        # Advanced RBAC
        if self.config.get("advanced_rbac_enabled", True):
            self.advanced_rbac = create_advanced_rbac_manager(
                self.config.get("rbac", {})
            )

        # Enhanced Compliance
        if self.config.get("enhanced_compliance_enabled", True):
            self.enhanced_compliance = create_enhanced_compliance(
                self.config.get("compliance", {}), self.base_compliance
            )

        # SLA Monitoring
        if self.config.get("sla_monitoring_enabled", True):
            self.sla_monitoring = create_sla_monitoring(self.config.get("sla", {}))

        # Enhanced Multi-Region
        if self.config.get("enhanced_multi_region_enabled", False):
            # This requires base multi-region strategy
            base_strategy_config = self.config.get("multi_region", {})
            if base_strategy_config:
                base_strategy = MultiRegionResilienceStrategy(base_strategy_config)
                self.enhanced_multi_region = create_enhanced_multi_region(
                    self.config.get("enhanced_multi_region", {}), base_strategy
                )

    def get_enterprise_status(self) -> Dict[str, Any]:
        """Get status of all enterprise features.

        Returns:
            Dictionary containing status of each enterprise component
        """
        status: Dict[str, Any] = {"enabled_features": [], "component_status": {}}

        # Check SAML
        if self.saml_manager:
            status["enabled_features"].append("saml_sso")
            status["component_status"]["saml_sso"] = {
                "enabled": True,
                "identity_providers": len(self.saml_manager.identity_providers),
                "active_sessions": len(self.saml_manager.saml_sessions),
            }

        # Check Advanced RBAC
        if self.advanced_rbac:
            status["enabled_features"].append("advanced_rbac")
            status["component_status"]["advanced_rbac"] = {
                "enabled": True,
                "custom_roles": len(
                    [
                        r
                        for r in self.advanced_rbac.roles.values()
                        if r.type.value == "custom"
                    ]
                ),
                "access_policies": len(self.advanced_rbac.access_policies),
                "active_assignments": sum(
                    len(assignments)
                    for assignments in self.advanced_rbac.role_assignments.values()
                ),
            }

        # Check Enhanced Compliance
        if self.enhanced_compliance:
            status["enabled_features"].append("enhanced_compliance")
            status["component_status"]["enhanced_compliance"] = {
                "enabled": True,
                "monitoring_rules": len(self.enhanced_compliance.monitoring_rules),
                "active_monitors": len(
                    [
                        t
                        for t in self.enhanced_compliance.monitoring_tasks.values()
                        if t and not t.done()
                    ]
                ),
                "compliance_artifacts": len(
                    self.enhanced_compliance.compliance_artifacts
                ),
            }

        # Check SLA Monitoring
        if self.sla_monitoring:
            status["enabled_features"].append("sla_monitoring")
            status["component_status"]["sla_monitoring"] = {
                "enabled": True,
                "active_slas": len(self.sla_monitoring.sla_definitions),
                "active_violations": len(self.sla_monitoring.active_violations),
                "monitoring_tasks": len(
                    [
                        t
                        for t in self.sla_monitoring.monitoring_tasks.values()
                        if t and not t.done()
                    ]
                ),
            }

        # Check Enhanced Multi-Region
        if self.enhanced_multi_region:
            status["enabled_features"].append("enhanced_multi_region")
            dashboard = self.enhanced_multi_region.get_failover_dashboard()
            status["component_status"]["enhanced_multi_region"] = {
                "enabled": True,
                "total_regions": dashboard["total_regions_count"],
                "healthy_regions": dashboard["healthy_regions_count"],
                "pending_failovers": dashboard["pending_failovers"],
                "auto_failover_enabled": dashboard["auto_failover_enabled"],
            }

        return status

    def configure_enterprise_security(self, security_config: Dict[str, Any]):
        """Configure enterprise security settings.

        Args:
            security_config: Security configuration including:
                - mfa_required_roles: List of roles requiring MFA
                - session_timeout: Session timeout in seconds
                - ip_whitelist: List of allowed IP addresses
                - password_policy: Password requirements
        """
        # Configure MFA requirements
        if "mfa_required_roles" in security_config and self.advanced_rbac:
            for role_name in security_config["mfa_required_roles"]:
                if role_name in self.advanced_rbac.roles:
                    self.advanced_rbac.roles[role_name].conditions["require_mfa"] = True

        # Configure session settings
        if "session_timeout" in security_config:
            self.auth_manager.config["session_timeout"] = security_config[
                "session_timeout"
            ]

        # Configure IP restrictions
        if "ip_whitelist" in security_config and self.advanced_rbac:
            self.advanced_rbac.create_access_policy(
                name="IP Whitelist Policy",
                description="Restrict access to whitelisted IPs",
                resource_pattern="*",
                conditions=[{"ip_whitelist": security_config["ip_whitelist"]}],
                effect="allow",
                priority=100,
            )

    def generate_enterprise_report(
        self, report_type: str = "executive"
    ) -> Dict[str, Any]:
        """Generate comprehensive enterprise report.

        Args:
            report_type: Type of report (executive, technical, compliance)

        Returns:
            Comprehensive report data
        """
        import datetime

        report: Dict[str, Any] = {
            "generated_at": datetime.datetime.utcnow().isoformat(),
            "report_type": report_type,
            "sections": {},
        }

        # Security section
        if self.saml_manager or self.advanced_rbac:
            report["sections"]["security"] = {
                "authentication": {
                    "sso_enabled": self.saml_manager is not None,
                    "identity_providers": (
                        len(self.saml_manager.identity_providers)
                        if self.saml_manager
                        else 0
                    ),
                    "active_sessions": (
                        len(self.saml_manager.saml_sessions) if self.saml_manager else 0
                    ),
                },
                "authorization": {
                    "rbac_enabled": self.advanced_rbac is not None,
                    "custom_roles": (
                        len(
                            [
                                r
                                for r in self.advanced_rbac.roles.values()
                                if r.type.value == "custom"
                            ]
                        )
                        if self.advanced_rbac
                        else 0
                    ),
                    "access_policies": (
                        len(self.advanced_rbac.access_policies)
                        if self.advanced_rbac
                        else 0
                    ),
                },
            }

        # Compliance section
        if self.enhanced_compliance:
            # Generate SOC2 summary
            soc2_report = self.enhanced_compliance.generate_soc2_type2_report(
                period_days=90
            )

            # Generate HIPAA summary
            hipaa_report = self.enhanced_compliance.generate_hipaa_audit_report(
                period_days=365
            )

            report["sections"]["compliance"] = {
                "soc2": {
                    "overall_status": soc2_report["overall_status"],
                    "trust_service_criteria": soc2_report["trust_service_criteria"],
                    "exceptions_count": len(soc2_report["exceptions"]),
                },
                "hipaa": {
                    "safeguards": hipaa_report["safeguards"],
                    "phi_access_summary": hipaa_report["phi_access_summary"],
                    "risk_analysis": hipaa_report["risk_analysis"],
                },
            }

        # Operations section
        if self.sla_monitoring:
            sla_report = self.sla_monitoring.generate_sla_report(
                period_hours=24 * 7
            )  # Weekly

            report["sections"]["operations"] = {
                "sla_compliance": {
                    "overall_compliance": sla_report.overall_compliance,
                    "active_violations": len(self.sla_monitoring.active_violations),
                    "recommendations": sla_report.recommendations[:3],  # Top 3
                }
            }

        # Infrastructure section
        if self.enhanced_multi_region:
            failover_dashboard = self.enhanced_multi_region.get_failover_dashboard()

            report["sections"]["infrastructure"] = {
                "multi_region": {
                    "total_regions": failover_dashboard["total_regions_count"],
                    "healthy_regions": failover_dashboard["healthy_regions_count"],
                    "recent_failovers": len(failover_dashboard["recent_failovers"]),
                    "auto_failover_enabled": failover_dashboard[
                        "auto_failover_enabled"
                    ],
                }
            }

        return report

    async def perform_enterprise_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check of enterprise features.

        Returns:
            Health check results
        """
        health: Dict[str, Any] = {"overall_status": "healthy", "components": {}, "issues": []}

        # Check each component
        if self.saml_manager:
            try:
                # Check IdP connectivity
                idp_count = len(self.saml_manager.identity_providers)
                health["components"]["saml_sso"] = {
                    "status": "healthy" if idp_count > 0 else "degraded",
                    "identity_providers": idp_count,
                }
                if idp_count == 0:
                    health["issues"].append("No identity providers configured")
            except Exception as e:
                health["components"]["saml_sso"] = {
                    "status": "unhealthy",
                    "error": str(e),
                }
                health["overall_status"] = "degraded"

        if self.advanced_rbac:
            try:
                # Check RBAC functionality
                roles_count = len(self.advanced_rbac.roles)
                health["components"]["advanced_rbac"] = {
                    "status": "healthy",
                    "roles_count": roles_count,
                    "policies_count": len(self.advanced_rbac.access_policies),
                }
            except Exception as e:
                health["components"]["advanced_rbac"] = {
                    "status": "unhealthy",
                    "error": str(e),
                }
                health["overall_status"] = "degraded"

        if self.enhanced_compliance:
            try:
                # Check monitoring rules
                active_monitors = len(
                    [
                        t
                        for t in self.enhanced_compliance.monitoring_tasks.values()
                        if t and not t.done()
                    ]
                )
                health["components"]["enhanced_compliance"] = {
                    "status": "healthy" if active_monitors > 0 else "degraded",
                    "active_monitors": active_monitors,
                }
                if (
                    active_monitors == 0
                    and len(self.enhanced_compliance.monitoring_rules) > 0
                ):
                    health["issues"].append(
                        "Compliance monitoring rules defined but not running"
                    )
            except Exception as e:
                health["components"]["enhanced_compliance"] = {
                    "status": "unhealthy",
                    "error": str(e),
                }
                health["overall_status"] = "degraded"

        if self.sla_monitoring:
            try:
                # Check SLA violations
                violations = len(self.sla_monitoring.active_violations)
                health["components"]["sla_monitoring"] = {
                    "status": "degraded" if violations > 0 else "healthy",
                    "active_violations": violations,
                }
                if violations > 0:
                    health["issues"].append(f"{violations} active SLA violations")
            except Exception as e:
                health["components"]["sla_monitoring"] = {
                    "status": "unhealthy",
                    "error": str(e),
                }
                health["overall_status"] = "degraded"

        if self.enhanced_multi_region:
            try:
                # Check region health
                dashboard = self.enhanced_multi_region.get_failover_dashboard()
                unhealthy_regions = (
                    dashboard["total_regions_count"]
                    - dashboard["healthy_regions_count"]
                )
                health["components"]["enhanced_multi_region"] = {
                    "status": "degraded" if unhealthy_regions > 0 else "healthy",
                    "unhealthy_regions": unhealthy_regions,
                    "pending_failovers": dashboard["pending_failovers"],
                }
                if unhealthy_regions > 0:
                    health["issues"].append(f"{unhealthy_regions} unhealthy regions")
            except Exception as e:
                health["components"]["enhanced_multi_region"] = {
                    "status": "unhealthy",
                    "error": str(e),
                }
                health["overall_status"] = "degraded"

        # Set overall status based on issues
        if health["overall_status"] == "healthy" and health["issues"]:
            health["overall_status"] = "degraded"

        return health

    async def shutdown(self):
        """Gracefully shutdown enterprise features"""
        # Shutdown SLA monitoring
        if self.sla_monitoring:
            await self.sla_monitoring.shutdown()

        # Shutdown enhanced multi-region
        if self.enhanced_multi_region:
            await self.enhanced_multi_region.shutdown()

        # Other components don't have async shutdown requirements


def create_enterprise_features(config: Dict[str, Any]) -> EnterpriseFeatures:
    """Factory function to create enterprise features manager.

    Args:
        config: Enterprise configuration with sections:
            - saml_enabled: Enable SAML/SSO
            - saml: SAML configuration
            - advanced_rbac_enabled: Enable advanced RBAC
            - rbac: RBAC configuration
            - enhanced_compliance_enabled: Enable enhanced compliance
            - compliance: Compliance configuration
            - sla_monitoring_enabled: Enable SLA monitoring
            - sla: SLA configuration
            - enhanced_multi_region_enabled: Enable enhanced multi-region
            - multi_region: Multi-region configuration

    Returns:
        Configured EnterpriseFeatures instance
    """
    return EnterpriseFeatures(config)
