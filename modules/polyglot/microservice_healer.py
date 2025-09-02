"""
Microservice healing across language boundaries.
Enables healing of distributed microservices written in different languages.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..analysis.language_plugin_system import LanguagePluginRegistry
from ..analysis.cross_language_orchestrator import CrossLanguageOrchestrator


class ServiceCommunicationProtocol(Enum):
    """Supported service communication protocols."""
    REST = "rest"
    GRPC = "grpc"
    GRAPHQL = "graphql"
    WEBSOCKET = "websocket"
    MESSAGE_QUEUE = "message_queue"
    EVENT_STREAM = "event_stream"


class ServiceMeshType(Enum):
    """Supported service mesh technologies."""
    ISTIO = "istio"
    LINKERD = "linkerd"
    CONSUL_CONNECT = "consul_connect"
    APP_MESH = "app_mesh"
    KUMA = "kuma"
    NONE = "none"


@dataclass
class ServiceInfo:
    """Information about a microservice."""
    service_id: str
    name: str
    language: str
    version: str
    endpoints: List[str]
    dependencies: List[str]
    protocols: List[ServiceCommunicationProtocol]
    health_status: str = "healthy"
    last_error_time: Optional[datetime] = None
    error_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ServiceError:
    """Error that occurred in a microservice."""
    service_id: str
    error_id: str
    timestamp: datetime
    error_type: str
    message: str
    stack_trace: Optional[str]
    language: str
    related_services: List[str] = field(default_factory=list)
    transaction_id: Optional[str] = None
    request_trace: Optional[Dict[str, Any]] = None
    impact_score: float = 0.0


@dataclass
class CrossServiceError:
    """Error that spans multiple services."""
    error_id: str
    root_cause_service: str
    affected_services: List[str]
    error_chain: List[ServiceError]
    communication_protocol: ServiceCommunicationProtocol
    transaction_id: Optional[str] = None
    total_impact_score: float = 0.0


@dataclass
class HealingStrategy:
    """Strategy for healing a cross-service error."""
    strategy_id: str
    target_services: List[str]
    patches: Dict[str, List[Dict[str, Any]]]  # service_id -> patches
    coordination_required: bool
    estimated_downtime: float
    risk_score: float
    rollback_plan: Optional[Dict[str, Any]] = None


class MicroserviceHealer:
    """
    Handles healing of microservices across language boundaries.
    Coordinates with language plugins to generate appropriate fixes.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.plugin_registry = LanguagePluginRegistry()
        self.cross_language_orchestrator = CrossLanguageOrchestrator()
        self.services: Dict[str, ServiceInfo] = {}
        self.service_mesh_type = ServiceMeshType(
            config.get('service_mesh', 'none').lower()
        )
        self.healing_history: List[Dict[str, Any]] = []
        
    async def register_service(self, service_info: ServiceInfo) -> None:
        """Register a microservice for monitoring and healing."""
        self.services[service_info.service_id] = service_info
        self.logger.info(
            f"Registered service: {service_info.name} "
            f"(ID: {service_info.service_id}, Language: {service_info.language})"
        )
        
    async def analyze_cross_service_error(
        self, 
        error: ServiceError
    ) -> Optional[CrossServiceError]:
        """
        Analyze an error to determine if it affects multiple services.
        Traces error propagation across service boundaries.
        """
        affected_services = [error.service_id]
        error_chain = [error]
        
        # Trace error through dependencies
        if error.related_services:
            for service_id in error.related_services:
                if service_id in self.services:
                    # Check for cascading errors
                    related_errors = await self._find_related_errors(
                        service_id, 
                        error.timestamp,
                        error.transaction_id
                    )
                    if related_errors:
                        affected_services.extend([e.service_id for e in related_errors])
                        error_chain.extend(related_errors)
                        
        # Determine root cause service
        root_cause_service = await self._determine_root_cause(error_chain)
        
        # Calculate total impact
        total_impact = sum(e.impact_score for e in error_chain)
        
        # Identify communication protocol
        protocol = await self._identify_communication_protocol(error_chain)
        
        if len(affected_services) > 1:
            return CrossServiceError(
                error_id=f"cross_{error.error_id}",
                root_cause_service=root_cause_service,
                affected_services=list(set(affected_services)),
                error_chain=error_chain,
                communication_protocol=protocol,
                transaction_id=error.transaction_id,
                total_impact_score=total_impact
            )
        
        return None
        
    async def generate_healing_strategy(
        self,
        cross_service_error: CrossServiceError
    ) -> HealingStrategy:
        """
        Generate a coordinated healing strategy for cross-service errors.
        Considers language boundaries and service dependencies.
        """
        patches = {}
        
        for service_error in cross_service_error.error_chain:
            service_id = service_error.service_id
            service_info = self.services.get(service_id)
            
            if not service_info:
                continue
                
            # Get language-specific plugin
            plugin = self.plugin_registry.get_plugin(service_info.language)
            if not plugin:
                self.logger.warning(
                    f"No plugin found for language: {service_info.language}"
                )
                continue
                
            # Generate language-specific fix
            fix = plugin.generate_fix(self._convert_to_plugin_error(service_error))
            
            if fix:
                patches[service_id] = [{
                    'language': service_info.language,
                    'fix': fix,
                    'error': service_error,
                    'dependencies': await self._analyze_fix_dependencies(fix, service_info)
                }]
                
        # Determine if coordination is required
        coordination_required = self._requires_coordination(
            cross_service_error,
            patches
        )
        
        # Estimate impact and risk
        estimated_downtime = self._estimate_downtime(patches, coordination_required)
        risk_score = self._calculate_risk_score(
            cross_service_error,
            patches,
            coordination_required
        )
        
        # Generate rollback plan
        rollback_plan = self._generate_rollback_plan(patches)
        
        return HealingStrategy(
            strategy_id=f"strategy_{cross_service_error.error_id}",
            target_services=list(patches.keys()),
            patches=patches,
            coordination_required=coordination_required,
            estimated_downtime=estimated_downtime,
            risk_score=risk_score,
            rollback_plan=rollback_plan
        )
        
    async def execute_healing_strategy(
        self,
        strategy: HealingStrategy,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Execute a healing strategy across multiple services.
        Coordinates deployment to minimize downtime and risk.
        """
        execution_result = {
            'strategy_id': strategy.strategy_id,
            'success': True,
            'services_healed': [],
            'errors': [],
            'rollback_performed': False,
            'start_time': datetime.now(),
            'end_time': None
        }
        
        try:
            if strategy.coordination_required:
                # Execute coordinated deployment
                result = await self._execute_coordinated_healing(
                    strategy,
                    dry_run
                )
            else:
                # Execute parallel healing for independent services
                result = await self._execute_parallel_healing(
                    strategy,
                    dry_run
                )
                
            execution_result.update(result)
            
        except Exception as e:
            self.logger.error(f"Healing strategy execution failed: {e}")
            execution_result['success'] = False
            execution_result['errors'].append(str(e))
            
            # Attempt rollback if not in dry run
            if not dry_run and strategy.rollback_plan:
                rollback_result = await self._execute_rollback(strategy.rollback_plan)
                execution_result['rollback_performed'] = rollback_result['success']
                
        finally:
            execution_result['end_time'] = datetime.now()
            self.healing_history.append(execution_result)
            
        return execution_result
        
    async def _execute_coordinated_healing(
        self,
        strategy: HealingStrategy,
        dry_run: bool
    ) -> Dict[str, Any]:
        """Execute healing with coordination between services."""
        result = {'services_healed': [], 'errors': []}
        
        # Sort services by dependency order
        ordered_services = self._get_deployment_order(strategy.target_services)
        
        for service_id in ordered_services:
            if service_id not in strategy.patches:
                continue
                
            service_info = self.services[service_id]
            patches = strategy.patches[service_id]
            
            try:
                # Apply patches for this service
                for patch_info in patches:
                    if dry_run:
                        self.logger.info(
                            f"[DRY RUN] Would apply patch to {service_id}: "
                            f"{patch_info['fix']}"
                        )
                    else:
                        # Apply the patch
                        await self._apply_patch(
                            service_info,
                            patch_info['fix'],
                            patch_info['language']
                        )
                        
                        # Run tests
                        test_result = await self._run_service_tests(service_info)
                        
                        if not test_result['success']:
                            raise Exception(
                                f"Tests failed for {service_id}: "
                                f"{test_result['errors']}"
                            )
                            
                result['services_healed'].append(service_id)
                
                # Wait for service to stabilize before proceeding
                if not dry_run:
                    await self._wait_for_service_health(service_id)
                    
            except Exception as e:
                result['errors'].append({
                    'service_id': service_id,
                    'error': str(e)
                })
                # Stop coordinated deployment on failure
                break
                
        return result
        
    async def _execute_parallel_healing(
        self,
        strategy: HealingStrategy,
        dry_run: bool
    ) -> Dict[str, Any]:
        """Execute healing in parallel for independent services."""
        tasks = []
        
        for service_id in strategy.target_services:
            if service_id in strategy.patches:
                task = self._heal_single_service(
                    service_id,
                    strategy.patches[service_id],
                    dry_run
                )
                tasks.append(task)
                
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        services_healed = []
        errors = []
        
        for i, result in enumerate(results):
            service_id = strategy.target_services[i]
            if isinstance(result, Exception):
                errors.append({
                    'service_id': service_id,
                    'error': str(result)
                })
            elif result.get('success'):
                services_healed.append(service_id)
            else:
                errors.append({
                    'service_id': service_id,
                    'error': result.get('error', 'Unknown error')
                })
                
        return {
            'services_healed': services_healed,
            'errors': errors
        }
        
    async def _heal_single_service(
        self,
        service_id: str,
        patches: List[Dict[str, Any]],
        dry_run: bool
    ) -> Dict[str, Any]:
        """Heal a single service."""
        try:
            service_info = self.services[service_id]
            
            for patch_info in patches:
                if dry_run:
                    self.logger.info(
                        f"[DRY RUN] Would apply patch to {service_id}: "
                        f"{patch_info['fix']}"
                    )
                else:
                    await self._apply_patch(
                        service_info,
                        patch_info['fix'],
                        patch_info['language']
                    )
                    
                    # Run tests
                    test_result = await self._run_service_tests(service_info)
                    
                    if not test_result['success']:
                        return {
                            'success': False,
                            'error': f"Tests failed: {test_result['errors']}"
                        }
                        
            return {'success': True}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
            
    async def _find_related_errors(
        self,
        service_id: str,
        timestamp: datetime,
        transaction_id: Optional[str]
    ) -> List[ServiceError]:
        """Find errors in a service related to a specific time/transaction."""
        # This would integrate with monitoring/logging systems
        # For now, return empty list
        return []
        
    async def _determine_root_cause(
        self,
        error_chain: List[ServiceError]
    ) -> str:
        """Determine the root cause service from an error chain."""
        if not error_chain:
            return ""
            
        # Sort by timestamp to find earliest error
        sorted_errors = sorted(error_chain, key=lambda e: e.timestamp)
        return sorted_errors[0].service_id
        
    async def _identify_communication_protocol(
        self,
        error_chain: List[ServiceError]
    ) -> ServiceCommunicationProtocol:
        """Identify the communication protocol involved in the error."""
        # Analyze error messages and stack traces for protocol hints
        for error in error_chain:
            if error.request_trace:
                protocol = error.request_trace.get('protocol')
                if protocol:
                    try:
                        return ServiceCommunicationProtocol(protocol)
                    except ValueError:
                        pass
                        
        # Default to REST if unable to determine
        return ServiceCommunicationProtocol.REST
        
    def _convert_to_plugin_error(self, service_error: ServiceError) -> Dict[str, Any]:
        """Convert ServiceError to format expected by language plugins."""
        return {
            'error_type': service_error.error_type,
            'message': service_error.message,
            'stack_trace': service_error.stack_trace,
            'file_path': None,  # Would be extracted from stack trace
            'line_number': None,  # Would be extracted from stack trace
            'context': {
                'service_id': service_error.service_id,
                'transaction_id': service_error.transaction_id,
                'related_services': service_error.related_services
            }
        }
        
    async def _analyze_fix_dependencies(
        self,
        fix: Dict[str, Any],
        service_info: ServiceInfo
    ) -> List[str]:
        """Analyze dependencies that might be affected by a fix."""
        dependencies = []
        
        # Check if fix modifies API contracts
        if fix.get('modifies_api'):
            # All dependent services need to be considered
            dependencies.extend(service_info.dependencies)
            
        # Check if fix modifies data structures
        if fix.get('modifies_data_structures'):
            # Services sharing data need to be considered
            for service_id, info in self.services.items():
                if service_id != service_info.service_id:
                    if any(dep in info.dependencies for dep in service_info.dependencies):
                        dependencies.append(service_id)
                        
        return list(set(dependencies))
        
    def _requires_coordination(
        self,
        cross_service_error: CrossServiceError,
        patches: Dict[str, List[Dict[str, Any]]]
    ) -> bool:
        """Determine if coordinated deployment is required."""
        # Check for API contract changes
        for service_patches in patches.values():
            for patch in service_patches:
                if patch.get('dependencies'):
                    return True
                    
        # Check for shared state modifications
        if cross_service_error.communication_protocol in [
            ServiceCommunicationProtocol.MESSAGE_QUEUE,
            ServiceCommunicationProtocol.EVENT_STREAM
        ]:
            return True
            
        # Check for transaction-based errors
        if cross_service_error.transaction_id:
            return True
            
        return False
        
    def _estimate_downtime(
        self,
        patches: Dict[str, List[Dict[str, Any]]],
        coordination_required: bool
    ) -> float:
        """Estimate downtime in seconds for healing execution."""
        base_time_per_service = 30.0  # 30 seconds per service
        
        if coordination_required:
            # Sequential deployment takes longer
            return len(patches) * base_time_per_service
        else:
            # Parallel deployment only takes as long as slowest service
            return base_time_per_service
            
    def _calculate_risk_score(
        self,
        cross_service_error: CrossServiceError,
        patches: Dict[str, List[Dict[str, Any]]],
        coordination_required: bool
    ) -> float:
        """Calculate risk score for healing strategy (0.0 to 1.0)."""
        risk_score = 0.0
        
        # Base risk from number of services affected
        risk_score += len(patches) * 0.1
        
        # Risk from coordination requirements
        if coordination_required:
            risk_score += 0.2
            
        # Risk from impact score
        risk_score += min(cross_service_error.total_impact_score * 0.1, 0.3)
        
        # Risk from critical services
        for service_id in patches.keys():
            service = self.services.get(service_id)
            if service and service.metadata.get('critical', False):
                risk_score += 0.1
                
        return min(risk_score, 1.0)
        
    def _generate_rollback_plan(
        self,
        patches: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Generate a rollback plan for the patches."""
        rollback_plan = {
            'services': {},
            'order': []
        }
        
        for service_id, service_patches in patches.items():
            rollback_plan['services'][service_id] = {
                'backup_required': True,
                'patches_to_revert': len(service_patches),
                'estimated_time': 30.0  # seconds
            }
            rollback_plan['order'].append(service_id)
            
        # Reverse order for rollback
        rollback_plan['order'].reverse()
        
        return rollback_plan
        
    def _get_deployment_order(self, service_ids: List[str]) -> List[str]:
        """Get deployment order based on service dependencies."""
        # Simple topological sort based on dependencies
        ordered = []
        visited = set()
        
        def visit(service_id: str):
            if service_id in visited:
                return
            visited.add(service_id)
            
            service = self.services.get(service_id)
            if service:
                for dep in service.dependencies:
                    if dep in service_ids:
                        visit(dep)
                        
            if service_id not in ordered:
                ordered.append(service_id)
                
        for service_id in service_ids:
            visit(service_id)
            
        return ordered
        
    async def _apply_patch(
        self,
        service_info: ServiceInfo,
        fix: Dict[str, Any],
        language: str
    ) -> None:
        """Apply a patch to a service."""
        # This would integrate with deployment systems
        self.logger.info(
            f"Applying patch to {service_info.name} ({language}): {fix}"
        )
        
    async def _run_service_tests(
        self,
        service_info: ServiceInfo
    ) -> Dict[str, Any]:
        """Run tests for a service."""
        # This would integrate with testing frameworks
        return {'success': True, 'errors': []}
        
    async def _wait_for_service_health(
        self,
        service_id: str,
        timeout: float = 60.0
    ) -> bool:
        """Wait for a service to become healthy."""
        # This would integrate with health check endpoints
        await asyncio.sleep(2)  # Simulate waiting
        return True
        
    async def _execute_rollback(
        self,
        rollback_plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a rollback plan."""
        self.logger.warning("Executing rollback plan")
        
        success = True
        for service_id in rollback_plan['order']:
            try:
                # This would integrate with deployment systems
                self.logger.info(f"Rolling back service: {service_id}")
                await asyncio.sleep(1)  # Simulate rollback
            except Exception as e:
                self.logger.error(f"Rollback failed for {service_id}: {e}")
                success = False
                
        return {'success': success}
        
    async def get_healing_metrics(self) -> Dict[str, Any]:
        """Get metrics about healing operations."""
        total_healings = len(self.healing_history)
        successful_healings = sum(
            1 for h in self.healing_history if h['success']
        )
        
        services_by_language = {}
        for service in self.services.values():
            lang = service.language
            services_by_language[lang] = services_by_language.get(lang, 0) + 1
            
        return {
            'total_healings': total_healings,
            'successful_healings': successful_healings,
            'success_rate': successful_healings / total_healings if total_healings > 0 else 0,
            'registered_services': len(self.services),
            'services_by_language': services_by_language,
            'service_mesh_type': self.service_mesh_type.value
        }