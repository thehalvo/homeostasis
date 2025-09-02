"""
Healing Orchestrator

Orchestrates complex healing scenarios across multiple services.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, Future

from .healing_engine import HealingEngine

logger = logging.getLogger(__name__)


class HealingOrchestrator:
    """
    Orchestrates healing actions across multiple services and components.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the healing orchestrator.
        
        Args:
            config: Configuration for the orchestrator
        """
        self.config = config or {}
        self.healing_engines: Dict[str, HealingEngine] = {}
        self.max_concurrent_healings = self.config.get('max_concurrent_healings', 5)
        self.healing_dependencies = self.config.get('healing_dependencies', {})
        self.orchestration_history: List[Dict[str, Any]] = []
        
    def register_engine(self, service_name: str, engine: HealingEngine) -> None:
        """
        Register a healing engine for a service.
        
        Args:
            service_name: Name of the service
            engine: Healing engine instance
        """
        self.healing_engines[service_name] = engine
        logger.info(f"Registered healing engine for service: {service_name}")
        
    def orchestrate_healing(self, errors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Orchestrate healing for multiple errors across services.
        
        Args:
            errors: List of error analyses with service information
            
        Returns:
            Orchestration result
        """
        start_time = datetime.now()
        
        # Group errors by service
        errors_by_service = self._group_errors_by_service(errors)
        
        # Determine healing order based on dependencies
        healing_order = self._determine_healing_order(errors_by_service.keys())
        
        # Execute healing in order
        results = {}
        with ThreadPoolExecutor(max_workers=self.max_concurrent_healings) as executor:
            for service_batch in healing_order:
                # Heal services in batch (services without dependencies on each other)
                futures: Dict[str, Future] = {}
                
                for service in service_batch:
                    if service in errors_by_service:
                        engine = self.healing_engines.get(service)
                        if engine:
                            service_errors = errors_by_service[service]
                            future = executor.submit(
                                self._heal_service,
                                service,
                                engine,
                                service_errors
                            )
                            futures[service] = future
                        else:
                            logger.warning(f"No healing engine registered for service: {service}")
                            
                # Wait for batch to complete
                for service, future in futures.items():
                    try:
                        results[service] = future.result(timeout=300)  # 5 minute timeout
                    except Exception as e:
                        logger.error(f"Healing failed for service {service}: {e}")
                        results[service] = {
                            'success': False,
                            'error': str(e),
                            'results': []
                        }
                        
        # Record orchestration
        duration = (datetime.now() - start_time).total_seconds()
        orchestration_result = {
            'timestamp': datetime.now(),
            'duration_seconds': duration,
            'total_errors': len(errors),
            'services_healed': len(results),
            'success_rate': self._calculate_success_rate(results),
            'results': results
        }
        
        self.orchestration_history.append(orchestration_result)
        return orchestration_result
        
    def _group_errors_by_service(self, errors: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group errors by service.
        
        Args:
            errors: List of errors
            
        Returns:
            Dictionary mapping service names to their errors
        """
        errors_by_service = {}
        for error in errors:
            service = error.get('service', 'unknown')
            if service not in errors_by_service:
                errors_by_service[service] = []
            errors_by_service[service].append(error)
        return errors_by_service
        
    def _determine_healing_order(self, services: List[str]) -> List[List[str]]:
        """
        Determine order of healing based on dependencies.
        
        Args:
            services: List of services to heal
            
        Returns:
            List of service batches to heal in order
        """
        # Simple implementation - group services by dependency level
        # In production, this would use topological sorting
        
        if not self.healing_dependencies:
            # No dependencies, heal all at once
            return [list(services)]
            
        # Group by dependency level
        levels = {}
        for service in services:
            deps = self.healing_dependencies.get(service, [])
            level = 0
            for dep in deps:
                if dep in services:
                    level += 1
            if level not in levels:
                levels[level] = []
            levels[level].append(service)
            
        # Return in order of dependency level
        return [levels[level] for level in sorted(levels.keys())]
        
    def _heal_service(self, service_name: str, engine: HealingEngine, errors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Heal all errors for a service.
        
        Args:
            service_name: Name of the service
            engine: Healing engine to use
            errors: List of errors to heal
            
        Returns:
            Healing results for the service
        """
        logger.info(f"Healing {len(errors)} errors for service: {service_name}")
        
        results = []
        for error in errors:
            context = {
                'service_name': service_name,
                'orchestration': True
            }
            result = engine.heal(error, context)
            if result:
                results.append(result.to_dict())
                
        success_count = sum(1 for r in results if r.get('success', False))
        return {
            'success': success_count == len(errors),
            'total_errors': len(errors),
            'healed': success_count,
            'results': results
        }
        
    def _calculate_success_rate(self, results: Dict[str, Dict[str, Any]]) -> float:
        """
        Calculate overall success rate.
        
        Args:
            results: Healing results by service
            
        Returns:
            Success rate as percentage
        """
        if not results:
            return 0.0
            
        total_errors = sum(r.get('total_errors', 0) for r in results.values())
        total_healed = sum(r.get('healed', 0) for r in results.values())
        
        if total_errors == 0:
            return 0.0
            
        return (total_healed / total_errors) * 100
        
    def get_orchestration_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get orchestration history.
        
        Args:
            limit: Maximum number of results
            
        Returns:
            List of orchestration results
        """
        if limit:
            return self.orchestration_history[-limit:]
        return self.orchestration_history