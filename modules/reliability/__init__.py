"""
High-Reliability Systems Module for Homeostasis.

This module provides comprehensive reliability features including:
- Formal methods verification
- Chaos engineering integration
- Predictive healing for critical components
- Redundancy and failover coordination
- Hardware/software boundary healing
"""

from .formal_verification import (
    FormalVerifier,
    Z3Verifier,
    ContractVerifier,
    ModelChecker,
    CriticalSystemVerifier,
    SystemModel,
    VerificationProperty,
    VerificationResult,
    VerificationLevel,
    PropertyType,
    create_example_critical_system,
    verify_critical_healing
)

from .chaos_engineering import (
    ChaosMonkey,
    ChaosExperiment,
    FaultInjection,
    FaultType,
    ImpactLevel,
    ExperimentStatus,
    SteadyStateHypothesis,
    NetworkFaultInjector,
    ResourceFaultInjector,
    ServiceFaultInjector,
    create_network_chaos_experiment,
    create_resource_chaos_experiment
)

from .predictive_healing import (
    PredictiveHealer,
    PredictiveModel,
    AnomalyDetectionModel,
    TimeSeriesPredictor,
    ResourceExhaustionPredictor,
    ComponentMetrics,
    FailurePrediction,
    HealingRecommendation,
    PredictionType,
    ComponentType,
    HealthStatus,
    analyze_critical_component
)

from .redundancy_failover import (
    FailoverCoordinator,
    RedundancyGroup,
    Instance,
    HealthCheck,
    FailoverEvent,
    RedundancyType,
    FailoverStrategy,
    InstanceState,
    HealthCheckType,
    LoadBalancer,
    RoundRobinBalancer,
    LeastConnectionsBalancer,
    HTTPHealthChecker,
    TCPHealthChecker,
    create_web_service_redundancy,
    create_database_redundancy
)

from .hardware_boundary_healing import (
    HardwareHealer,
    HardwareComponent,
    HardwareFault,
    HealingResult,
    HardwareType,
    FaultType as HardwareFaultType,
    HealingAction,
    HardwareMonitor,
    CPUMonitor,
    MemoryMonitor,
    DiskMonitor,
    create_server_hardware,
    monitor_and_heal_hardware
)

__all__ = [
    # Formal Verification
    'FormalVerifier',
    'Z3Verifier',
    'ContractVerifier',
    'ModelChecker',
    'CriticalSystemVerifier',
    'SystemModel',
    'VerificationProperty',
    'VerificationResult',
    'VerificationLevel',
    'PropertyType',
    'create_example_critical_system',
    'verify_critical_healing',
    
    # Chaos Engineering
    'ChaosMonkey',
    'ChaosExperiment',
    'FaultInjection',
    'FaultType',
    'ImpactLevel',
    'ExperimentStatus',
    'SteadyStateHypothesis',
    'NetworkFaultInjector',
    'ResourceFaultInjector',
    'ServiceFaultInjector',
    'create_network_chaos_experiment',
    'create_resource_chaos_experiment',
    
    # Predictive Healing
    'PredictiveHealer',
    'PredictiveModel',
    'AnomalyDetectionModel',
    'TimeSeriesPredictor',
    'ResourceExhaustionPredictor',
    'ComponentMetrics',
    'FailurePrediction',
    'HealingRecommendation',
    'PredictionType',
    'ComponentType',
    'HealthStatus',
    'analyze_critical_component',
    
    # Redundancy and Failover
    'FailoverCoordinator',
    'RedundancyGroup',
    'Instance',
    'HealthCheck',
    'FailoverEvent',
    'RedundancyType',
    'FailoverStrategy',
    'InstanceState',
    'HealthCheckType',
    'LoadBalancer',
    'RoundRobinBalancer',
    'LeastConnectionsBalancer',
    'HTTPHealthChecker',
    'TCPHealthChecker',
    'create_web_service_redundancy',
    'create_database_redundancy',
    
    # Hardware Boundary Healing
    'HardwareHealer',
    'HardwareComponent',
    'HardwareFault',
    'HealingResult',
    'HardwareType',
    'HardwareFaultType',
    'HealingAction',
    'HardwareMonitor',
    'CPUMonitor',
    'MemoryMonitor',
    'DiskMonitor',
    'create_server_hardware',
    'monitor_and_heal_hardware'
]