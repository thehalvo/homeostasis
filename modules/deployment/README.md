# Deployment Module for Homeostasis

This module provides deployment capabilities for the Homeostasis self-healing system, including:

1. **Canary Deployment**: Gradually roll out fixes to a subset of traffic
2. **Blue-Green Deployment**: Deploy fixes to a parallel environment and switch traffic
3. **Rollback**: Automated rollback of failed deployments
4. **Monitoring**: Post-deployment monitoring for deployed fixes
5. **Kubernetes Integration**: Deploy and manage fixes in Kubernetes environments

## Components

- `canary.py`: Canary deployment for gradually rolling out fixes
- `blue_green.py`: Blue-green deployment for zero-downtime deployments
- `rollback.py`: Automated rollback mechanisms for failed deployments
- `kubernetes/`: Kubernetes-specific deployment utilities
- `monitoring.py`: Post-deployment monitoring integration
- `cloud/`: Cloud provider-specific deployment adapters

## Configuration

Deployment features are configured in the main Homeostasis config file under the `deployment` section.

## Usage

Example usage of the canary deployment:

```python
from modules.deployment.canary import CanaryDeployment

# Initialize canary deployment
canary = CanaryDeployment(config={
    "percentage": 10,  # Start with 10% of traffic
    "increment": 10,   # Increase by 10% each step
    "interval": 300,   # 5 minutes between increments
    "max_percentage": 100  # Go to 100% if all is well
})

# Start canary deployment
canary.start(service_name="example_service", fix_id="fix_123")

# Check status
status = canary.get_status()
print(f"Canary status: {status['current_percentage']}% deployed")

# Promote or rollback based on metrics
if status["metrics"]["error_rate"] < 0.01:
    canary.promote()  # Move to next percentage or complete
else:
    canary.rollback() # Revert to previous deployment
```