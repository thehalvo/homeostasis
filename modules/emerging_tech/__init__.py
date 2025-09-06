"""
Emerging Technology Support Module

This module provides self-healing capabilities for emerging technologies including:
- Quantum computing error mitigation
- Blockchain and distributed ledger healing
- IoT and edge device support
- Augmented reality application resilience
"""

from .augmented_reality import ARResilienceManager
from .blockchain import BlockchainHealer
from .iot import IoTDeviceMonitor
from .quantum_computing import QuantumErrorMitigator

__all__ = [
    "QuantumErrorMitigator",
    "BlockchainHealer",
    "IoTDeviceMonitor",
    "ARResilienceManager",
]
