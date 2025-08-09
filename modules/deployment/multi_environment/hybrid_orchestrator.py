"""
Hybrid Cloud/On-Premise Healing Coordination

Provides unified orchestration across cloud and on-premise environments,
enabling coordinated healing actions across hybrid infrastructure.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

from modules.enterprise.orchestration import EnterpriseOrchestrator
from modules.security.audit import SecurityAuditor
from modules.monitoring.distributed_monitoring import DistributedMonitor


class EnvironmentType(Enum):
    """Types of deployment environments"""
    CLOUD_AWS = "cloud_aws"
    CLOUD_GCP = "cloud_gcp"
    CLOUD_AZURE = "cloud_azure"
    ON_PREMISE = "on_premise"
    EDGE = "edge"
    HYBRID = "hybrid"


class HealingScope(Enum):
    """Scope of healing actions"""
    LOCAL = "local"
    ENVIRONMENT = "environment"
    CROSS_ENVIRONMENT = "cross_environment"
    GLOBAL = "global"


@dataclass
class Environment:
    """Represents a deployment environment"""
    id: str
    name: str
    type: EnvironmentType
    region: Optional[str]
    connection_info: Dict[str, Any]
    capabilities: List[str]
    health_status: str
    metadata: Dict[str, Any]


@dataclass
class HealingContext:
    """Context for cross-environment healing"""
    error_id: str
    source_environment: Environment
    affected_environments: List[Environment]
    scope: HealingScope
    dependencies: List[str]
    constraints: Dict[str, Any]
    priority: int
    timestamp: datetime


@dataclass
class HealingPlan:
    """Coordinated healing plan across environments"""
    plan_id: str
    context: HealingContext
    steps: List['HealingStep']
    rollback_steps: List['HealingStep']
    approval_required: bool
    estimated_duration: int
    risk_score: float


@dataclass
class HealingStep:
    """Individual step in a healing plan"""
    step_id: str
    environment: Environment
    action: str
    parameters: Dict[str, Any]
    dependencies: List[str]
    timeout: int
    can_fail: bool
    rollback_action: Optional[str]


class EnvironmentConnector(ABC):
    """Abstract interface for environment connections"""
    
    @abstractmethod
    async def connect(self, connection_info: Dict[str, Any]) -> bool:
        """Establish connection to environment"""
        pass
    
    @abstractmethod
    async def get_health_status(self) -> Dict[str, Any]:
        """Get environment health status"""
        pass
    
    @abstractmethod
    async def execute_action(self, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute healing action in environment"""
        pass
    
    @abstractmethod
    async def rollback_action(self, action: str, parameters: Dict[str, Any]) -> bool:
        """Rollback a previously executed action"""
        pass


class CloudConnector(EnvironmentConnector):
    """Connector for cloud environments"""
    
    def __init__(self, cloud_type: str):
        self.cloud_type = cloud_type
        self.client = None
        self.logger = logging.getLogger(f"{__name__}.CloudConnector")
    
    async def connect(self, connection_info: Dict[str, Any]) -> bool:
        """Connect to cloud provider"""
        try:
            if self.cloud_type == "aws":
                # AWS connection logic
                pass
            elif self.cloud_type == "gcp":
                # GCP connection logic
                pass
            elif self.cloud_type == "azure":
                # Azure connection logic
                pass
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to {self.cloud_type}: {e}")
            return False
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get cloud environment health"""
        return {
            "status": "healthy",
            "services": {},
            "metrics": {},
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def execute_action(self, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute action in cloud environment"""
        return {
            "status": "success",
            "action": action,
            "result": {}
        }
    
    async def rollback_action(self, action: str, parameters: Dict[str, Any]) -> bool:
        """Rollback cloud action"""
        return True


class OnPremiseConnector(EnvironmentConnector):
    """Connector for on-premise environments"""
    
    def __init__(self):
        self.connection = None
        self.logger = logging.getLogger(f"{__name__}.OnPremiseConnector")
    
    async def connect(self, connection_info: Dict[str, Any]) -> bool:
        """Connect to on-premise infrastructure"""
        try:
            # SSH/API connection logic
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to on-premise: {e}")
            return False
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get on-premise environment health"""
        return {
            "status": "healthy",
            "servers": {},
            "services": {},
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def execute_action(self, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute action in on-premise environment"""
        return {
            "status": "success",
            "action": action,
            "result": {}
        }
    
    async def rollback_action(self, action: str, parameters: Dict[str, Any]) -> bool:
        """Rollback on-premise action"""
        return True


class HybridCloudOrchestrator(EnterpriseOrchestrator):
    """
    Orchestrates healing across hybrid cloud and on-premise environments.
    Coordinates actions, manages dependencies, and ensures consistency.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.environments: Dict[str, Environment] = {}
        self.connectors: Dict[str, EnvironmentConnector] = {}
        self.active_plans: Dict[str, HealingPlan] = {}
        self.monitor = DistributedMonitor()
        self.auditor = SecurityAuditor()
        self.logger = logging.getLogger(__name__)
        
        # Load environment configurations
        self._load_environments(config.get('environments', []))
    
    def _load_environments(self, env_configs: List[Dict[str, Any]]):
        """Load environment configurations"""
        for env_config in env_configs:
            env = Environment(
                id=env_config['id'],
                name=env_config['name'],
                type=EnvironmentType(env_config['type']),
                region=env_config.get('region'),
                connection_info=env_config['connection'],
                capabilities=env_config.get('capabilities', []),
                health_status='unknown',
                metadata=env_config.get('metadata', {})
            )
            self.environments[env.id] = env
            
            # Create appropriate connector
            if env.type in [EnvironmentType.CLOUD_AWS, EnvironmentType.CLOUD_GCP, EnvironmentType.CLOUD_AZURE]:
                self.connectors[env.id] = CloudConnector(env.type.value.split('_')[1])
            elif env.type == EnvironmentType.ON_PREMISE:
                self.connectors[env.id] = OnPremiseConnector()
    
    async def initialize(self) -> bool:
        """Initialize connections to all environments"""
        tasks = []
        for env_id, connector in self.connectors.items():
            env = self.environments[env_id]
            tasks.append(self._init_environment(env, connector))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        success_count = sum(1 for r in results if r is True)
        self.logger.info(f"Initialized {success_count}/{len(tasks)} environments")
        
        return success_count > 0  # At least one environment must be available
    
    async def _init_environment(self, env: Environment, connector: EnvironmentConnector) -> bool:
        """Initialize a single environment"""
        try:
            connected = await connector.connect(env.connection_info)
            if connected:
                health = await connector.get_health_status()
                env.health_status = health.get('status', 'unknown')
                return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to initialize environment {env.name}: {e}")
            return False
    
    async def detect_cross_environment_issue(self, error_data: Dict[str, Any]) -> Optional[HealingContext]:
        """Detect issues that span multiple environments"""
        source_env_id = error_data.get('environment_id')
        if not source_env_id or source_env_id not in self.environments:
            return None
        
        source_env = self.environments[source_env_id]
        
        # Analyze error to determine scope and affected environments
        affected_envs = await self._analyze_error_impact(error_data)
        
        # Determine healing scope
        if len(affected_envs) > 1:
            scope = HealingScope.CROSS_ENVIRONMENT
        elif affected_envs:
            scope = HealingScope.ENVIRONMENT
        else:
            scope = HealingScope.LOCAL
        
        # Extract dependencies
        dependencies = await self._extract_dependencies(error_data, affected_envs)
        
        return HealingContext(
            error_id=error_data.get('error_id', 'unknown'),
            source_environment=source_env,
            affected_environments=affected_envs,
            scope=scope,
            dependencies=dependencies,
            constraints=self._extract_constraints(error_data),
            priority=self._calculate_priority(error_data, affected_envs),
            timestamp=datetime.utcnow()
        )
    
    async def _analyze_error_impact(self, error_data: Dict[str, Any]) -> List[Environment]:
        """Analyze which environments are affected by an error"""
        affected = []
        error_type = error_data.get('type')
        
        # Check for cross-environment patterns
        if error_type in ['network_connectivity', 'api_timeout', 'service_unavailable']:
            # These errors might affect connected environments
            source_env_id = error_data.get('environment_id')
            for env_id, env in self.environments.items():
                if env_id != source_env_id:
                    # Check if environments are connected
                    if await self._are_environments_connected(source_env_id, env_id):
                        affected.append(env)
        
        return affected
    
    async def _are_environments_connected(self, env1_id: str, env2_id: str) -> bool:
        """Check if two environments have dependencies"""
        # Check service mesh connections, API dependencies, etc.
        return True  # Simplified for now
    
    async def _extract_dependencies(self, error_data: Dict[str, Any], 
                                   affected_envs: List[Environment]) -> List[str]:
        """Extract service dependencies from error context"""
        dependencies = []
        
        # Extract from error data
        if 'service_dependencies' in error_data:
            dependencies.extend(error_data['service_dependencies'])
        
        # Extract from stack trace
        if 'stack_trace' in error_data:
            # Parse stack trace for service calls
            pass
        
        return list(set(dependencies))
    
    def _extract_constraints(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract healing constraints from error context"""
        return {
            'max_downtime': error_data.get('sla_max_downtime', 300),
            'require_approval': error_data.get('require_approval', False),
            'maintenance_window': error_data.get('maintenance_window'),
            'data_consistency': error_data.get('data_consistency_required', True)
        }
    
    def _calculate_priority(self, error_data: Dict[str, Any], 
                          affected_envs: List[Environment]) -> int:
        """Calculate healing priority based on impact and severity"""
        base_priority = error_data.get('severity', 5)
        
        # Increase priority for production environments
        prod_count = sum(1 for env in affected_envs if 'production' in env.metadata.get('tags', []))
        priority = base_priority - (prod_count * 2)
        
        # Increase priority for cross-environment issues
        if len(affected_envs) > 1:
            priority -= 1
        
        return max(1, min(10, priority))
    
    async def create_healing_plan(self, context: HealingContext, 
                                patch_data: Dict[str, Any]) -> HealingPlan:
        """Create a coordinated healing plan across environments"""
        steps = []
        rollback_steps = []
        
        # Determine healing strategy based on scope
        if context.scope == HealingScope.CROSS_ENVIRONMENT:
            steps, rollback_steps = await self._create_cross_env_steps(context, patch_data)
        elif context.scope == HealingScope.ENVIRONMENT:
            steps, rollback_steps = await self._create_single_env_steps(context, patch_data)
        else:
            steps, rollback_steps = await self._create_local_steps(context, patch_data)
        
        # Calculate risk and duration
        risk_score = self._calculate_risk_score(context, steps)
        duration = self._estimate_duration(steps)
        
        # Determine if approval is required
        approval_required = (
            risk_score > 0.7 or
            context.constraints.get('require_approval', False) or
            any('production' in env.metadata.get('tags', []) 
                for env in context.affected_environments)
        )
        
        plan = HealingPlan(
            plan_id=f"plan_{context.error_id}_{datetime.utcnow().timestamp()}",
            context=context,
            steps=steps,
            rollback_steps=rollback_steps,
            approval_required=approval_required,
            estimated_duration=duration,
            risk_score=risk_score
        )
        
        self.active_plans[plan.plan_id] = plan
        return plan
    
    async def _create_cross_env_steps(self, context: HealingContext, 
                                    patch_data: Dict[str, Any]) -> Tuple[List[HealingStep], List[HealingStep]]:
        """Create steps for cross-environment healing"""
        steps = []
        rollback_steps = []
        
        # Phase 1: Prepare environments
        for env in context.affected_environments:
            step = HealingStep(
                step_id=f"prepare_{env.id}",
                environment=env,
                action="prepare_healing",
                parameters={"patch": patch_data, "dependencies": context.dependencies},
                dependencies=[],
                timeout=300,
                can_fail=False,
                rollback_action="cleanup_preparation"
            )
            steps.append(step)
            
            rollback_steps.append(HealingStep(
                step_id=f"rollback_prepare_{env.id}",
                environment=env,
                action="cleanup_preparation",
                parameters={},
                dependencies=[],
                timeout=60,
                can_fail=True,
                rollback_action=None
            ))
        
        # Phase 2: Apply patches in dependency order
        ordered_envs = self._order_environments_by_dependency(context.affected_environments)
        
        for i, env in enumerate(ordered_envs):
            dependencies = [f"prepare_{e.id}" for e in context.affected_environments]
            if i > 0:
                dependencies.append(f"apply_{ordered_envs[i-1].id}")
            
            step = HealingStep(
                step_id=f"apply_{env.id}",
                environment=env,
                action="apply_patch",
                parameters={"patch": patch_data},
                dependencies=dependencies,
                timeout=600,
                can_fail=False,
                rollback_action="revert_patch"
            )
            steps.append(step)
            
            rollback_steps.insert(0, HealingStep(
                step_id=f"rollback_apply_{env.id}",
                environment=env,
                action="revert_patch",
                parameters={"patch": patch_data},
                dependencies=[],
                timeout=300,
                can_fail=False,
                rollback_action=None
            ))
        
        # Phase 3: Verify healing
        for env in context.affected_environments:
            step = HealingStep(
                step_id=f"verify_{env.id}",
                environment=env,
                action="verify_healing",
                parameters={"error_id": context.error_id},
                dependencies=[f"apply_{env.id}"],
                timeout=300,
                can_fail=False,
                rollback_action=None
            )
            steps.append(step)
        
        return steps, rollback_steps
    
    async def _create_single_env_steps(self, context: HealingContext, 
                                     patch_data: Dict[str, Any]) -> Tuple[List[HealingStep], List[HealingStep]]:
        """Create steps for single environment healing"""
        env = context.source_environment
        steps = []
        rollback_steps = []
        
        # Apply patch
        steps.append(HealingStep(
            step_id=f"apply_{env.id}",
            environment=env,
            action="apply_patch",
            parameters={"patch": patch_data},
            dependencies=[],
            timeout=600,
            can_fail=False,
            rollback_action="revert_patch"
        ))
        
        # Verify
        steps.append(HealingStep(
            step_id=f"verify_{env.id}",
            environment=env,
            action="verify_healing",
            parameters={"error_id": context.error_id},
            dependencies=[f"apply_{env.id}"],
            timeout=300,
            can_fail=False,
            rollback_action=None
        ))
        
        # Rollback
        rollback_steps.append(HealingStep(
            step_id=f"rollback_{env.id}",
            environment=env,
            action="revert_patch",
            parameters={"patch": patch_data},
            dependencies=[],
            timeout=300,
            can_fail=False,
            rollback_action=None
        ))
        
        return steps, rollback_steps
    
    async def _create_local_steps(self, context: HealingContext, 
                                patch_data: Dict[str, Any]) -> Tuple[List[HealingStep], List[HealingStep]]:
        """Create steps for local healing"""
        # Similar to single environment but with reduced scope
        return await self._create_single_env_steps(context, patch_data)
    
    def _order_environments_by_dependency(self, environments: List[Environment]) -> List[Environment]:
        """Order environments based on dependencies"""
        # Simplified ordering - in practice would use topological sort
        return sorted(environments, key=lambda e: e.metadata.get('dependency_level', 0))
    
    def _calculate_risk_score(self, context: HealingContext, steps: List[HealingStep]) -> float:
        """Calculate risk score for healing plan"""
        base_risk = 0.3
        
        # Increase risk for cross-environment healing
        if context.scope == HealingScope.CROSS_ENVIRONMENT:
            base_risk += 0.2
        
        # Increase risk for production environments
        prod_count = sum(1 for env in context.affected_environments 
                        if 'production' in env.metadata.get('tags', []))
        base_risk += (prod_count * 0.1)
        
        # Increase risk based on number of steps
        base_risk += min(0.2, len(steps) * 0.02)
        
        return min(1.0, base_risk)
    
    def _estimate_duration(self, steps: List[HealingStep]) -> int:
        """Estimate total duration for healing plan"""
        # Account for parallel execution
        max_duration = 0
        sequential_duration = 0
        
        processed_deps = set()
        for step in steps:
            if not step.dependencies:
                max_duration = max(max_duration, step.timeout)
            else:
                # Check if dependencies are sequential
                if all(dep in processed_deps for dep in step.dependencies):
                    sequential_duration += step.timeout
                else:
                    max_duration = max(max_duration, step.timeout)
            processed_deps.add(step.step_id)
        
        return max_duration + sequential_duration
    
    async def execute_healing_plan(self, plan: HealingPlan) -> Dict[str, Any]:
        """Execute a healing plan across environments"""
        self.logger.info(f"Executing healing plan {plan.plan_id}")
        
        # Check approval if required
        if plan.approval_required:
            approved = await self._get_approval(plan)
            if not approved:
                return {"status": "rejected", "plan_id": plan.plan_id}
        
        # Record start of healing
        await self.auditor.log_event("healing_started", {
            "plan_id": plan.plan_id,
            "error_id": plan.context.error_id,
            "environments": [env.name for env in plan.context.affected_environments],
            "risk_score": plan.risk_score
        })
        
        # Execute steps
        results = {}
        executed_steps = []
        
        try:
            for step in plan.steps:
                # Wait for dependencies
                await self._wait_for_dependencies(step, results)
                
                # Execute step
                result = await self._execute_step(step)
                results[step.step_id] = result
                executed_steps.append(step)
                
                # Check if step failed
                if result.get('status') != 'success' and not step.can_fail:
                    raise Exception(f"Step {step.step_id} failed: {result.get('error')}")
            
            # All steps completed successfully
            await self.auditor.log_event("healing_completed", {
                "plan_id": plan.plan_id,
                "duration": sum(r.get('duration', 0) for r in results.values())
            })
            
            return {
                "status": "success",
                "plan_id": plan.plan_id,
                "results": results
            }
            
        except Exception as e:
            self.logger.error(f"Healing plan {plan.plan_id} failed: {e}")
            
            # Execute rollback
            rollback_results = await self._execute_rollback(plan, executed_steps)
            
            await self.auditor.log_event("healing_failed", {
                "plan_id": plan.plan_id,
                "error": str(e),
                "rollback_results": rollback_results
            })
            
            return {
                "status": "failed",
                "plan_id": plan.plan_id,
                "error": str(e),
                "results": results,
                "rollback_results": rollback_results
            }
    
    async def _get_approval(self, plan: HealingPlan) -> bool:
        """Get approval for healing plan"""
        # In practice, would integrate with approval system
        self.logger.info(f"Approval required for plan {plan.plan_id}")
        return True
    
    async def _wait_for_dependencies(self, step: HealingStep, results: Dict[str, Any]):
        """Wait for step dependencies to complete"""
        for dep in step.dependencies:
            while dep not in results:
                await asyncio.sleep(1)
            
            # Check if dependency succeeded
            if results[dep].get('status') != 'success':
                raise Exception(f"Dependency {dep} failed")
    
    async def _execute_step(self, step: HealingStep) -> Dict[str, Any]:
        """Execute a single healing step"""
        start_time = datetime.utcnow()
        
        try:
            connector = self.connectors.get(step.environment.id)
            if not connector:
                raise Exception(f"No connector for environment {step.environment.id}")
            
            # Execute action with timeout
            result = await asyncio.wait_for(
                connector.execute_action(step.action, step.parameters),
                timeout=step.timeout
            )
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            
            return {
                "status": "success",
                "result": result,
                "duration": duration
            }
            
        except asyncio.TimeoutError:
            return {
                "status": "failed",
                "error": "timeout",
                "duration": step.timeout
            }
        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds()
            return {
                "status": "failed",
                "error": str(e),
                "duration": duration
            }
    
    async def _execute_rollback(self, plan: HealingPlan, 
                              executed_steps: List[HealingStep]) -> Dict[str, Any]:
        """Execute rollback for failed healing plan"""
        self.logger.info(f"Executing rollback for plan {plan.plan_id}")
        
        results = {}
        
        # Find rollback steps for executed steps
        executed_ids = {step.step_id for step in executed_steps}
        relevant_rollbacks = [
            rb for rb in plan.rollback_steps 
            if any(f"rollback_{exec_id}" in rb.step_id for exec_id in executed_ids)
        ]
        
        # Execute rollback steps in order
        for rollback_step in relevant_rollbacks:
            try:
                result = await self._execute_step(rollback_step)
                results[rollback_step.step_id] = result
            except Exception as e:
                self.logger.error(f"Rollback step {rollback_step.step_id} failed: {e}")
                results[rollback_step.step_id] = {"status": "failed", "error": str(e)}
        
        return results
    
    async def get_environment_status(self, env_id: str) -> Dict[str, Any]:
        """Get current status of an environment"""
        if env_id not in self.environments:
            return {"error": "Environment not found"}
        
        env = self.environments[env_id]
        connector = self.connectors.get(env_id)
        
        if not connector:
            return {"error": "No connector available"}
        
        try:
            health = await connector.get_health_status()
            return {
                "environment": env.name,
                "type": env.type.value,
                "health": health,
                "active_healings": [
                    plan.plan_id for plan in self.active_plans.values()
                    if env in plan.context.affected_environments
                ]
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def get_cross_environment_view(self) -> Dict[str, Any]:
        """Get comprehensive view of all environments and their relationships"""
        view = {
            "environments": {},
            "connections": [],
            "active_healings": [],
            "health_summary": {
                "healthy": 0,
                "degraded": 0,
                "unhealthy": 0
            }
        }
        
        # Gather environment statuses
        for env_id, env in self.environments.items():
            status = await self.get_environment_status(env_id)
            view["environments"][env_id] = {
                "name": env.name,
                "type": env.type.value,
                "region": env.region,
                "status": status
            }
            
            # Update health summary
            health_status = status.get('health', {}).get('status', 'unknown')
            if health_status == 'healthy':
                view["health_summary"]["healthy"] += 1
            elif health_status == 'degraded':
                view["health_summary"]["degraded"] += 1
            else:
                view["health_summary"]["unhealthy"] += 1
        
        # Add active healings
        for plan in self.active_plans.values():
            view["active_healings"].append({
                "plan_id": plan.plan_id,
                "error_id": plan.context.error_id,
                "affected_environments": [env.name for env in plan.context.affected_environments],
                "scope": plan.context.scope.value,
                "risk_score": plan.risk_score
            })
        
        return view