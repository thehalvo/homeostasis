# Enterprise Integration Module

The Enterprise Integration module provides comprehensive integration capabilities for large-scale enterprise environments, enabling Homeostasis to work seamlessly with existing enterprise tools and processes.

## Overview

This module implements Phase 14.4 of the Homeostasis project, providing:

1. **IT Service Management (ITSM) Integration** - Connect with ServiceNow, Jira Service Management, and other ITSM tools
2. **CMDB Synchronization** - Sync configuration items with enterprise CMDBs
3. **Enterprise Monitoring Adapters** - Integrate with Datadog, Prometheus, Splunk, Elasticsearch, and New Relic
4. **Large-Scale Deployment Orchestration** - Orchestrate deployments using Kubernetes, Terraform, and Ansible
5. **Multi-Team Collaboration** - Enable team notifications, approval workflows, and metrics tracking

## Components

### 1. ITSM Integration (`itsm_integration.py`)

Provides integration with IT Service Management tools for incident and change management.

**Features:**
- Create and update incidents automatically from healing actions
- Submit change requests for approval
- Link healing actions to ITSM tickets
- Search and retrieve ticket information

**Supported Platforms:**
- ServiceNow
- Jira Service Management
- Extensible for other ITSM tools

**Example Usage:**
```python
from modules.enterprise import create_itsm_connector, ITSMIncident, IncidentPriority

# Create connector
connector = create_itsm_connector('servicenow', {
    'instance': 'mycompany',
    'username': 'api_user',
    'password': 'secure_password'
})

# Create incident
incident = ITSMIncident(
    short_description="Database connection timeout",
    description="Automated healing action required",
    priority=IncidentPriority.HIGH,
    healing_context={'action_id': 'heal_12345'}
)

success, incident_number = await connector.create_incident(incident)
```

### 2. CMDB Synchronization (`cmdb_sync.py`)

Synchronizes configuration items between Homeostasis and enterprise CMDBs.

**Features:**
- Bi-directional sync of configuration items
- Relationship mapping and impact analysis
- Change detection and versioning
- Support for multiple CMDB platforms

**Supported Platforms:**
- ServiceNow CMDB
- Device42
- Extensible for other CMDBs

**Example Usage:**
```python
from modules.enterprise import create_cmdb_synchronizer, CMDBItem, CIType

# Create synchronizer
sync = create_cmdb_synchronizer('servicenow', {
    'instance': 'mycompany',
    'username': 'cmdb_user',
    'password': 'secure_password'
})

# Sync configuration items
items = [
    {
        'name': 'web-server-01',
        'type': 'application',
        'environment': 'production',
        'healing_enabled': True
    }
]

results = await sync.sync_from_homeostasis(items)
```

### 3. Monitoring Adapters (`monitoring_adapters.py`)

Integrates with enterprise monitoring systems to collect metrics and alerts.

**Features:**
- Query metrics and time-series data
- Retrieve and acknowledge alerts
- Send custom metrics and events
- Register callbacks for real-time processing

**Supported Platforms:**
- Datadog
- Prometheus
- Splunk
- Elasticsearch/Elastic Stack
- New Relic

**Example Usage:**
```python
from modules.enterprise import create_monitoring_adapter
from datetime import datetime, timedelta

# Create adapter
adapter = create_monitoring_adapter('datadog', {
    'api_key': 'your_api_key',
    'app_key': 'your_app_key'
})

# Query metrics
metrics = await adapter.get_metrics(
    {'metric_name': 'system.cpu.usage'},
    datetime.utcnow() - timedelta(hours=1),
    datetime.utcnow()
)

# Get alerts
alerts = await adapter.get_alerts({'group_states': ['Alert', 'Warn']})
```

### 4. Deployment Orchestration (`orchestration.py`)

Orchestrates large-scale deployments across different infrastructure platforms.

**Features:**
- Multi-stage deployment planning
- Parallel and sequential execution
- Health checks and validation
- Rollback capabilities
- Dry-run support

**Supported Platforms:**
- Kubernetes
- Terraform
- Ansible
- Extensible for other orchestrators

**Example Usage:**
```python
from modules.enterprise import create_orchestrator, DeploymentPlan, DeploymentStrategy

# Create orchestrator
orchestrator = create_orchestrator('kubernetes', {
    'kubeconfig': '/path/to/kubeconfig',
    'namespace': 'production'
})

# Create deployment plan
plan = DeploymentPlan(
    plan_id='deploy_001',
    name='Microservices Update',
    strategy=DeploymentStrategy.ROLLING_UPDATE,
    resources=[...]  # Define resources
)

# Execute deployment
result = await orchestrator.execute_plan(plan)
```

### 5. Team Collaboration (`collaboration.py`)

Enables multi-team collaboration with notifications and approval workflows.

**Features:**
- Team and member management
- Multi-channel notifications (Slack, Teams, Email)
- Approval workflows with configurable rules
- Team performance metrics
- On-call rotation support

**Supported Platforms:**
- Slack
- Microsoft Teams
- Webhook-based integrations
- Email (when configured)

**Example Usage:**
```python
from modules.enterprise import TeamCollaborationHub, Team, TeamMember, ApprovalRequest

# Create collaboration hub
hub = TeamCollaborationHub({
    'slack': {'webhook_url': 'https://hooks.slack.com/...'},
    'teams': {'webhook_url': 'https://outlook.office.com/...'}
})

# Create team
team = Team(
    team_id='platform_team',
    name='Platform Engineering',
    description='Responsible for infrastructure',
    members=['user1', 'user2']
)

await hub.create_team(team)

# Request approval
request = ApprovalRequest(
    request_id='req_001',
    title='Deploy critical fix',
    description='Fix for database connection issue',
    approvers=['user1', 'user2'],
    minimum_approvals=2
)

await hub.request_approval(request)
```

## Configuration

### Basic Configuration Example

```python
enterprise_config = {
    'itsm': {
        'provider': 'servicenow',
        'instance': 'mycompany',
        'username': 'api_user',
        'password': 'secure_password'
    },
    'cmdb': {
        'provider': 'servicenow',
        'instance': 'mycompany',
        'username': 'cmdb_user',
        'password': 'secure_password',
        'sync_interval': 300
    },
    'monitoring': {
        'datadog': {
            'api_key': 'dd_api_key',
            'app_key': 'dd_app_key'
        },
        'prometheus': {
            'base_url': 'http://prometheus:9090',
            'alertmanager_url': 'http://alertmanager:9093'
        }
    },
    'orchestration': {
        'kubernetes': {
            'kubeconfig': '/path/to/kubeconfig',
            'namespace': 'default'
        },
        'terraform': {
            'terraform_path': '/usr/local/bin/terraform',
            'working_directory': '/infrastructure'
        }
    },
    'collaboration': {
        'slack': {
            'webhook_url': 'https://hooks.slack.com/services/...',
            'bot_token': 'xoxb-...'
        },
        'teams': {
            'webhook_url': 'https://outlook.office.com/webhook/...'
        },
        'approval_timeout': 3600,
        'auto_approve_low_risk': False
    }
}
```

## Integration with Homeostasis Core

The enterprise module integrates with the core Homeostasis system through several touchpoints:

1. **Healing Actions** - Automatically create ITSM tickets and request approvals for healing actions
2. **Configuration Discovery** - Sync discovered configuration items to CMDB
3. **Monitoring** - Use enterprise monitoring data to trigger healing actions
4. **Deployment** - Execute healing fixes through enterprise orchestration tools
5. **Governance** - Enforce approval workflows based on the governance framework

### Example Integration Flow

```python
# In the main orchestrator
from modules.enterprise import (
    create_itsm_connector,
    create_monitoring_adapter,
    TeamCollaborationHub
)

class EnterpriseAwareOrchestrator:
    def __init__(self, config):
        self.itsm = create_itsm_connector(
            config['itsm']['provider'],
            config['itsm']
        )
        self.monitoring = create_monitoring_adapter(
            'datadog',
            config['monitoring']['datadog']
        )
        self.collaboration = TeamCollaborationHub(config['collaboration'])
    
    async def handle_healing_action(self, action):
        # Create ITSM incident
        incident = ITSMIncident(
            short_description=f"Healing: {action['description']}",
            priority=self._map_priority(action['severity']),
            healing_context=action
        )
        success, ticket_id = await self.itsm.create_incident(incident)
        
        # Request approval if needed
        if action['requires_approval']:
            request = await self._create_approval_request(action)
            await self.collaboration.request_approval(request)
        
        # Send notification
        await self.collaboration.send_notification(
            self._create_notification(action, ticket_id)
        )
```

## Best Practices

1. **Security**
   - Store credentials securely using environment variables or secret management
   - Use service accounts with minimal required permissions
   - Enable SSL/TLS for all connections
   - Implement proper authentication for webhooks

2. **Reliability**
   - Implement retry logic for external API calls
   - Use circuit breakers for failing services
   - Cache CMDB data for performance
   - Implement proper error handling and logging

3. **Performance**
   - Batch operations where possible
   - Use async operations for I/O-bound tasks
   - Implement connection pooling
   - Monitor API rate limits

4. **Governance**
   - Define clear approval policies
   - Implement audit logging for all actions
   - Regular review of automated actions
   - Maintain documentation of integrations

## Troubleshooting

### Common Issues

1. **Authentication Failures**
   - Verify credentials and permissions
   - Check network connectivity
   - Ensure API endpoints are accessible
   - Verify SSL certificates

2. **Sync Issues**
   - Check CMDB field mappings
   - Verify data formats and types
   - Review sync logs for errors
   - Ensure unique identifiers are consistent

3. **Notification Failures**
   - Verify webhook URLs
   - Check channel permissions
   - Review message formatting
   - Monitor rate limits

4. **Orchestration Errors**
   - Verify orchestrator configurations
   - Check resource permissions
   - Review deployment logs
   - Ensure health checks are properly configured

## Future Enhancements

1. **Additional Integrations**
   - PagerDuty for incident management
   - HashiCorp Vault for secrets management
   - Jenkins/GitLab CI for deployment pipelines
   - Microsoft Azure DevOps

2. **Advanced Features**
   - Machine learning for approval recommendations
   - Automated runbook execution
   - Cost optimization analysis
   - Compliance reporting

3. **Performance Improvements**
   - GraphQL support for efficient queries
   - WebSocket connections for real-time updates
   - Distributed caching
   - Event streaming with Kafka/Pulsar

## Contributing

When adding new enterprise integrations:

1. Follow the existing pattern of abstract base classes
2. Implement comprehensive error handling
3. Add appropriate logging
4. Write unit and integration tests
5. Update documentation
6. Consider backward compatibility

## License

This module is part of the Homeostasis project and follows the same open-source license.