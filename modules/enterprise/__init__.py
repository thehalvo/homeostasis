"""
Enterprise Integration Module for Homeostasis

This module provides integration with enterprise systems including:
- IT Service Management (ITSM) tools
- Configuration Management Databases (CMDB)
- Enterprise monitoring systems
- Large-scale deployment orchestration
- Multi-team collaboration features
"""

from .itsm_integration import (
    ITSMConnector,
    ServiceNowConnector,
    JiraServiceManagementConnector,
    ITSMIncident,
    ITSMChangeRequest
)

from .cmdb_sync import (
    CMDBSynchronizer,
    CMDBItem,
    CMDBRelationship,
    ServiceNowCMDB,
    DeviceCMDB
)

from .monitoring_adapters import (
    MonitoringAdapter,
    DatadogAdapter,
    PrometheusAdapter,
    SplunkAdapter,
    ElasticAdapter,
    NewRelicAdapter
)

from .orchestration import (
    EnterpriseOrchestrator,
    KubernetesOrchestrator,
    TerraformOrchestrator,
    AnsibleOrchestrator,
    DeploymentPlan
)

from .collaboration import (
    TeamCollaborationHub,
    TeamNotificationService,
    ChangeApprovalWorkflow,
    TeamMetricsCollector,
    SlackIntegration,
    MicrosoftTeamsIntegration
)

__all__ = [
    # ITSM
    'ITSMConnector',
    'ServiceNowConnector',
    'JiraServiceManagementConnector',
    'ITSMIncident',
    'ITSMChangeRequest',
    
    # CMDB
    'CMDBSynchronizer',
    'CMDBItem',
    'CMDBRelationship',
    'ServiceNowCMDB',
    'DeviceCMDB',
    
    # Monitoring
    'MonitoringAdapter',
    'DatadogAdapter',
    'PrometheusAdapter',
    'SplunkAdapter',
    'ElasticAdapter',
    'NewRelicAdapter',
    
    # Orchestration
    'EnterpriseOrchestrator',
    'KubernetesOrchestrator',
    'TerraformOrchestrator',
    'AnsibleOrchestrator',
    'DeploymentPlan',
    
    # Collaboration
    'TeamCollaborationHub',
    'TeamNotificationService',
    'ChangeApprovalWorkflow',
    'TeamMetricsCollector',
    'SlackIntegration',
    'MicrosoftTeamsIntegration'
]