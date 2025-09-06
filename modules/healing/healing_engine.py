"""
Healing Engine

Coordinates healing actions and manages healing strategies.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from .healer import Healer, HealingResult, HealingStrategy

logger = logging.getLogger(__name__)


class HealingEngine:
    """
    Main engine for coordinating healing actions.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the healing engine.

        Args:
            config: Configuration for the healing engine
        """
        self.config = config or {}
        self.healers: List[Healer] = []
        self.healing_history: List[HealingResult] = []
        self.max_healing_attempts = self.config.get("max_healing_attempts", 3)
        self.healing_timeout = self.config.get("healing_timeout", 300)  # 5 minutes

    def register_healer(self, healer: Healer) -> None:
        """
        Register a healer with the engine.

        Args:
            healer: Healer instance to register
        """
        self.healers.append(healer)
        logger.info(f"Registered healer: {healer.name}")

    def heal(
        self, error_analysis: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> Optional[HealingResult]:
        """
        Attempt to heal an error using registered healers.

        Args:
            error_analysis: Error analysis data
            context: Additional context for healing

        Returns:
            Healing result if successful, None otherwise
        """
        context = context or {}
        start_time = datetime.now()

        # Find applicable healers
        applicable_healers = [h for h in self.healers if h.can_heal(error_analysis)]

        if not applicable_healers:
            logger.warning("No applicable healers found for error")
            return None

        # Try each healer
        for healer in applicable_healers:
            logger.info(f"Attempting healing with {healer.name}")

            # Check preconditions
            if not healer.validate_preconditions(context):
                logger.warning(f"Preconditions not met for {healer.name}")
                continue

            try:
                # Perform healing
                result = healer.heal(error_analysis, context)

                # Record result
                self.healing_history.append(result)

                if result.success:
                    logger.info(f"Healing successful with {healer.name}")
                    return result
                else:
                    logger.warning(f"Healing failed with {healer.name}: {result.error}")

            except Exception as e:
                logger.error(f"Error during healing with {healer.name}: {e}")

        # No successful healing
        duration = (datetime.now() - start_time).total_seconds()
        failed_result = HealingResult(
            success=False,
            strategy=HealingStrategy.CUSTOM,
            description="All healing attempts failed",
            timestamp=datetime.now(),
            duration_seconds=duration,
            error="No healer could successfully handle the error",
        )
        self.healing_history.append(failed_result)
        return failed_result

    def get_healing_history(self, limit: Optional[int] = None) -> List[HealingResult]:
        """
        Get healing history.

        Args:
            limit: Maximum number of results to return

        Returns:
            List of healing results
        """
        if limit:
            return self.healing_history[-limit:]
        return self.healing_history

    def get_success_rate(self) -> float:
        """
        Calculate healing success rate.

        Returns:
            Success rate as a percentage
        """
        if not self.healing_history:
            return 0.0

        successes = sum(1 for r in self.healing_history if r.success)
        return (successes / len(self.healing_history)) * 100

    def clear_history(self) -> None:
        """Clear healing history."""
        self.healing_history.clear()
        logger.info("Healing history cleared")
