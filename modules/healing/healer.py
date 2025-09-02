"""
Core Healer Module

This module provides the main healing functionality for Homeostasis.
"""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class HealingStrategy(Enum):
    """Types of healing strategies."""
    RESTART = "restart"
    ROLLBACK = "rollback"
    SCALE = "scale"
    RECONFIGURE = "reconfigure"
    PATCH = "patch"
    FAILOVER = "failover"
    CUSTOM = "custom"


@dataclass
class HealingResult:
    """Result of a healing action."""
    success: bool
    strategy: HealingStrategy
    description: str
    timestamp: datetime
    duration_seconds: float
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'success': self.success,
            'strategy': self.strategy.value,
            'description': self.description,
            'timestamp': self.timestamp.isoformat(),
            'duration_seconds': self.duration_seconds,
            'error': self.error,
            'metadata': self.metadata
        }


class Healer(ABC):
    """
    Abstract base class for healers.
    
    Healers implement specific healing strategies for different types of errors.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the healer.
        
        Args:
            config: Configuration for the healer
        """
        self.config = config or {}
        self.name = self.__class__.__name__
        
    @abstractmethod
    def can_heal(self, error_analysis: Dict[str, Any]) -> bool:
        """
        Check if this healer can handle the given error.
        
        Args:
            error_analysis: Error analysis data
            
        Returns:
            True if this healer can handle the error
        """
        pass
    
    @abstractmethod
    def heal(self, error_analysis: Dict[str, Any], context: Dict[str, Any]) -> HealingResult:
        """
        Perform healing action.
        
        Args:
            error_analysis: Error analysis data
            context: Additional context for healing
            
        Returns:
            Healing result
        """
        pass
    
    def validate_preconditions(self, context: Dict[str, Any]) -> bool:
        """
        Validate preconditions before healing.
        
        Args:
            context: Healing context
            
        Returns:
            True if preconditions are met
        """
        return True
    
    def rollback(self, healing_result: HealingResult, context: Dict[str, Any]) -> bool:
        """
        Rollback a healing action if needed.
        
        Args:
            healing_result: Previous healing result
            context: Healing context
            
        Returns:
            True if rollback was successful
        """
        logger.warning(f"{self.name} does not implement rollback")
        return False