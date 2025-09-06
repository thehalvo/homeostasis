"""
Legacy system integration module for Homeostasis.

This module provides adapters and healing capabilities for mainframe and legacy language systems.
"""

from .cobol_healer import COBOLHealer
from .compatibility_layer import CompatibilityLayer
from .fortran_healer import FortranHealer
from .hybrid_orchestrator import HybridOrchestrator
from .mainframe_adapter import MainframeAdapter
from .modernization_analyzer import ModernizationAnalyzer

__all__ = [
    "MainframeAdapter",
    "COBOLHealer",
    "FortranHealer",
    "ModernizationAnalyzer",
    "HybridOrchestrator",
    "CompatibilityLayer",
]
