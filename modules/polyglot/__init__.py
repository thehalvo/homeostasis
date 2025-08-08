"""
Polyglot Application Support Module for Homeostasis.

This module provides comprehensive support for healing distributed microservices
written in different programming languages. It includes:

1. Microservice Healing - Coordinated healing across language boundaries
2. Service Mesh Integration - Support for Istio, Linkerd, and other meshes  
3. Cross-Language Dependency Analysis - Understand dependencies between services
4. Unified Error Taxonomy - Common error classification across languages
5. Polyglot Stack Trace Analysis - Correlate errors across distributed systems
6. Distributed Transaction Healing - Recover from failed distributed transactions

Example usage:
    from modules.polyglot import MicroserviceHealer, UnifiedErrorTaxonomy
    
    # Create a microservice healer
    healer = MicroserviceHealer(config)
    
    # Register services
    await healer.register_service(service_info)
    
    # Analyze cross-service errors
    cross_error = await healer.analyze_cross_service_error(error)
    
    # Generate healing strategy
    strategy = await healer.generate_healing_strategy(cross_error)
    
    # Execute healing
    result = await healer.execute_healing_strategy(strategy)
"""

from .microservice_healer import (
    MicroserviceHealer,
    ServiceInfo,
    ServiceError,
    CrossServiceError,
    HealingStrategy,
    ServiceCommunicationProtocol,
    ServiceMeshType
)

from .service_mesh_integration import (
    ServiceMeshIntegration,
    ServiceMeshConfig,
    TrafficPolicy,
    ServiceMeshMetrics,
    IstioAdapter,
    LinkerdAdapter
)

from .cross_language_dependency_analyzer import (
    CrossLanguageDependencyAnalyzer,
    DependencyGraph,
    Dependency,
    DependencyType,
    APIContract,
    SharedDataStructure,
    APIProtocol
)

from .unified_error_taxonomy import (
    UnifiedErrorTaxonomy,
    UnifiedError,
    ErrorCategory,
    ErrorSeverity,
    ErrorScope,
    ErrorPattern,
    LanguageErrorMapper
)

from .polyglot_stack_trace_analyzer import (
    PolyglotStackTraceAnalyzer,
    StackTrace,
    StackFrame,
    CorrelatedStackTrace,
    FrameType
)

from .distributed_transaction_healer import (
    DistributedTransactionHealer,
    DistributedTransaction,
    TransactionParticipant,
    TransactionState,
    TransactionPattern,
    CompensationStrategy,
    TransactionRecoveryPlan
)

__all__ = [
    # Microservice Healer
    'MicroserviceHealer',
    'ServiceInfo',
    'ServiceError',
    'CrossServiceError', 
    'HealingStrategy',
    'ServiceCommunicationProtocol',
    'ServiceMeshType',
    
    # Service Mesh Integration
    'ServiceMeshIntegration',
    'ServiceMeshConfig',
    'TrafficPolicy',
    'ServiceMeshMetrics',
    'IstioAdapter',
    'LinkerdAdapter',
    
    # Dependency Analysis
    'CrossLanguageDependencyAnalyzer',
    'DependencyGraph',
    'Dependency',
    'DependencyType',
    'APIContract',
    'SharedDataStructure',
    'APIProtocol',
    
    # Error Taxonomy
    'UnifiedErrorTaxonomy',
    'UnifiedError',
    'ErrorCategory',
    'ErrorSeverity',
    'ErrorScope',
    'ErrorPattern',
    'LanguageErrorMapper',
    
    # Stack Trace Analysis
    'PolyglotStackTraceAnalyzer',
    'StackTrace',
    'StackFrame',
    'CorrelatedStackTrace',
    'FrameType',
    
    # Transaction Healing
    'DistributedTransactionHealer',
    'DistributedTransaction',
    'TransactionParticipant',
    'TransactionState',
    'TransactionPattern',
    'CompensationStrategy',
    'TransactionRecoveryPlan'
]

# Version info
__version__ = '1.0.0'
__author__ = 'Homeostasis Team'