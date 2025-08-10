"""
Legacy system integration module for Homeostasis.

This module provides adapters and healing capabilities for mainframe and legacy language systems.
"""

from .mainframe_adapter import MainframeAdapter
from .cobol_healer import COBOLHealer
from .fortran_healer import FortranHealer
from .modernization_analyzer import ModernizationAnalyzer
from .hybrid_orchestrator import HybridOrchestrator
from .compatibility_layer import CompatibilityLayer

__all__ = [
    'MainframeAdapter',
    'COBOLHealer', 
    'FortranHealer',
    'ModernizationAnalyzer',
    'HybridOrchestrator',
    'CompatibilityLayer'
]