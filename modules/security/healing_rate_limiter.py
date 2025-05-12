"""
Healing Rate Limiter module for Homeostasis.

Provides rate limiting and throttling capabilities for self-healing actions
to prevent excessive system changes and ensure stability.

This module implements several types of rate limiting for healing actions:

1. Healing Cycle Rate Limiting:
   - Limits how many self-healing cycles can be executed in a given time period
   - Enforces a minimum time interval between healing cycles

2. Patch Application Rate Limiting:
   - Limits the number of patches that can be applied in a given time period
   - Prevents excessive code modifications

3. Deployment Rate Limiting:
   - Limits how many deployments can occur in a given time period
   - Ensures stability by preventing frequent service restarts

4. File-specific Rate Limiting:
   - Limits how many modifications can be made to a specific file
   - Special protections for critical files
   - File cooldown mechanism to prevent repeated failed modifications

Usage:
    from modules.security.healing_rate_limiter import get_healing_rate_limiter

    # Get the rate limiter with specific configuration
    rate_limiter = get_healing_rate_limiter(config)

    # Check if a new healing cycle is allowed
    if rate_limiter.check_healing_cycle_limit():
        # Run healing cycle
        ...

    # Check if patch applications are allowed
    if rate_limiter.check_patch_application_limit(num_patches):
        # Apply patches
        ...

    # Check if a deployment is allowed
    if rate_limiter.check_deployment_limit():
        # Deploy changes
        ...

    # Check if modifications to a specific file are allowed
    if rate_limiter.check_file_limit(file_path):
        # Modify file
        ...

    # Place a file in cooldown after an issue
    rate_limiter.place_file_in_cooldown(file_path, duration)

    # Get current usage statistics
    stats = rate_limiter.get_usage_stats()

    # Reset all counters
    rate_limiter.reset_counters()
"""

import collections
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from modules.security.audit import get_audit_logger, log_event

logger = logging.getLogger(__name__)


class HealingRateLimitExceededError(Exception):
    """Exception raised when a healing action rate limit is exceeded."""
    pass


class HealingRateLimiter:
    """Rate limiter for self-healing actions."""

    def __init__(self, config: Dict = None):
        """Initialize the healing rate limiter.
        
        Args:
            config: Configuration dictionary for rate limiter settings
        """
        self.config = config or {}
        
        # Extract configuration
        self.enabled = self.config.get('enabled', True)
        
        # Default rate limits
        self.default_rate_limits = {
            'healing_cycle': (10, 3600),   # 10 healing cycles per hour
            'patch_application': (20, 3600),  # 20 patch applications per hour
            'deployment': (5, 3600),       # 5 deployments per hour
            'file': (3, 3600),            # 3 patches per file per hour
            'critical_file': (1, 3600),    # 1 patch per critical file per hour
        }
        
        # Critical files that need extra protection (paths are relative to project root)
        self.critical_files = self.config.get('critical_files', [
            'orchestrator/orchestrator.py',
            'modules/patch_generation/patcher.py',
            'modules/security/',
            'modules/deployment/'
        ])
        
        # Environment-specific limits
        if 'environment_limits' in self.config:
            env_limits = self.config.get('environment_limits', {})
            # Update with environment-specific limits if provided
            if env_limits:
                current_env = self.config.get('environment', 'development')
                if current_env in env_limits:
                    for key, value in env_limits[current_env].items():
                        self.default_rate_limits[key] = value
        
        # Override defaults with config values if provided
        if 'limits' in self.config:
            self.default_rate_limits.update(self.config['limits'])
        
        # Initialize storage for action counts and timestamps
        self.action_counts = {
            'healing_cycle': collections.defaultdict(int),
            'patch_application': collections.defaultdict(int),
            'deployment': collections.defaultdict(int),
            'file': collections.defaultdict(lambda: collections.defaultdict(int)),
        }
        
        # Store time windows for each action type
        self.time_windows = {
            'healing_cycle': collections.defaultdict(float),
            'patch_application': collections.defaultdict(float),
            'deployment': collections.defaultdict(float),
            'file': collections.defaultdict(lambda: collections.defaultdict(float)),
        }
        
        # Track accumulated actions by file for advanced throttling
        self.file_actions = collections.defaultdict(list)
        
        # Store the last healing cycle time for enforcing minimum intervals
        self.last_healing_cycle = 0
        
        # Cooldown tracking for files that have had issues
        self.file_cooldowns = {}
        
    def check_healing_cycle_limit(self) -> bool:
        """Check if a new healing cycle is within rate limits.
        
        Returns:
            bool: True if within rate limits, False if exceeded
        """
        if not self.enabled:
            return True
            
        now = time.time()
        
        # Check minimum interval between healing cycles
        min_interval = self.config.get('min_healing_cycle_interval', 60)  # Default: 60 seconds
        if now - self.last_healing_cycle < min_interval:
            logger.warning(f"Minimum interval between healing cycles not met. "
                         f"Last cycle was {now - self.last_healing_cycle:.2f} seconds ago, "
                         f"minimum interval is {min_interval} seconds.")
            return False
        
        # Check rate limit for healing cycles
        limit, window = self.default_rate_limits['healing_cycle']
        key = 'global'
        
        # If window has expired, reset counter
        if now - self.time_windows['healing_cycle'][key] > window:
            self.action_counts['healing_cycle'][key] = 1
            self.time_windows['healing_cycle'][key] = now
            self.last_healing_cycle = now
            return True
        
        # Increment counter
        self.action_counts['healing_cycle'][key] += 1
        
        # Check if limit exceeded
        if self.action_counts['healing_cycle'][key] > limit:
            logger.warning(f"Healing cycle rate limit exceeded: {limit} per {window} seconds.")
            return False
            
        # Update last cycle time
        self.last_healing_cycle = now
        return True
    
    def check_patch_application_limit(self, count: int = 1) -> bool:
        """Check if patch applications are within rate limits.
        
        Args:
            count: Number of patches to be applied
            
        Returns:
            bool: True if within rate limits, False if exceeded
        """
        if not self.enabled:
            return True
            
        now = time.time()
        
        # Check rate limit for patch applications
        limit, window = self.default_rate_limits['patch_application']
        key = 'global'
        
        # If window has expired, reset counter
        if now - self.time_windows['patch_application'][key] > window:
            self.action_counts['patch_application'][key] = count
            self.time_windows['patch_application'][key] = now
            return True
        
        # Add count to counter
        new_count = self.action_counts['patch_application'][key] + count
        
        # Check if limit exceeded
        if new_count > limit:
            logger.warning(f"Patch application rate limit exceeded: {limit} per {window} seconds.")
            return False
            
        # Update counter
        self.action_counts['patch_application'][key] = new_count
        return True
    
    def check_deployment_limit(self) -> bool:
        """Check if deployments are within rate limits.
        
        Returns:
            bool: True if within rate limits, False if exceeded
        """
        if not self.enabled:
            return True
            
        now = time.time()
        
        # Check rate limit for deployments
        limit, window = self.default_rate_limits['deployment']
        key = 'global'
        
        # If window has expired, reset counter
        if now - self.time_windows['deployment'][key] > window:
            self.action_counts['deployment'][key] = 1
            self.time_windows['deployment'][key] = now
            return True
        
        # Increment counter
        self.action_counts['deployment'][key] += 1
        
        # Check if limit exceeded
        if self.action_counts['deployment'][key] > limit:
            logger.warning(f"Deployment rate limit exceeded: {limit} per {window} seconds.")
            return False
            
        return True
    
    def check_file_limit(self, file_path: str) -> bool:
        """Check if modifications to a specific file are within rate limits.
        
        Args:
            file_path: Path to the file being modified
            
        Returns:
            bool: True if within rate limits, False if exceeded
        """
        if not self.enabled:
            return True
            
        now = time.time()
        
        # Check if file is in cooldown
        if file_path in self.file_cooldowns:
            cooldown_until = self.file_cooldowns[file_path]
            if now < cooldown_until:
                remaining = int(cooldown_until - now)
                logger.warning(f"File {file_path} is in cooldown for {remaining} more seconds.")
                return False
            else:
                # Remove from cooldown
                del self.file_cooldowns[file_path]
        
        # Determine if this is a critical file
        is_critical = False
        for critical_path in self.critical_files:
            if critical_path.endswith('/'):
                # This is a directory pattern
                if file_path.startswith(critical_path):
                    is_critical = True
                    break
            else:
                # This is a file pattern
                if file_path == critical_path:
                    is_critical = True
                    break
        
        # Determine appropriate limit based on file criticality
        if is_critical:
            limit, window = self.default_rate_limits['critical_file']
        else:
            limit, window = self.default_rate_limits['file']
        
        # If window has expired, reset counter
        if now - self.time_windows['file'][file_path]['default'] > window:
            self.action_counts['file'][file_path]['default'] = 1
            self.time_windows['file'][file_path]['default'] = now
            
            # Also reset the action history for this file
            self.file_actions[file_path] = []
            return True
        
        # Check if limit exceeded
        if self.action_counts['file'][file_path]['default'] >= limit:
            logger.warning(f"File modification rate limit exceeded for {file_path}: "
                         f"{limit} per {window} seconds.")
            return False
        
        # Increment counter
        self.action_counts['file'][file_path]['default'] += 1
        
        # Add this action to the file's history
        self.file_actions[file_path].append(now)
        
        return True
    
    def place_file_in_cooldown(self, file_path: str, duration: int = 3600) -> None:
        """Place a file in cooldown after an issue occurs.
        
        Args:
            file_path: Path to the file to be placed in cooldown
            duration: Cooldown duration in seconds (default is 1 hour)
        """
        if not self.enabled:
            return
            
        now = time.time()
        self.file_cooldowns[file_path] = now + duration
        logger.warning(f"Placed file {file_path} in cooldown for {duration} seconds.")
        
        # Log the event
        log_event(
            event_type='file_cooldown',
            details={
                'file_path': file_path,
                'duration': duration,
                'reason': 'healing_issue'
            }
        )
    
    def get_current_limits(self) -> Dict[str, Tuple[int, int]]:
        """Get the current rate limits.
        
        Returns:
            Dict[str, Tuple[int, int]]: Dictionary of rate limits
        """
        return self.default_rate_limits
        
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics.
        
        Returns:
            Dict[str, Any]: Dictionary of usage statistics
        """
        now = time.time()
        
        usage = {}
        
        # Global actions
        for action_type in ['healing_cycle', 'patch_application', 'deployment']:
            limit, window = self.default_rate_limits[action_type]
            
            # Calculate time left in current window
            time_passed = now - self.time_windows[action_type]['global']
            if time_passed > window:
                # Window has expired
                time_left = 0
                count = 0
            else:
                time_left = window - time_passed
                count = self.action_counts[action_type]['global']
                
            usage[action_type] = {
                'count': count,
                'limit': limit,
                'window': window,
                'window_remaining': time_left,
                'percent_used': (count / limit) * 100 if limit > 0 else 0
            }
            
        # Files in cooldown
        files_in_cooldown = {}
        for file_path, cooldown_until in self.file_cooldowns.items():
            remaining = cooldown_until - now
            if remaining > 0:
                files_in_cooldown[file_path] = {
                    'remaining': remaining,
                    'until': datetime.fromtimestamp(cooldown_until).isoformat()
                }
                
        usage['cooldowns'] = files_in_cooldown
        
        # Top files by modification count
        top_files = []
        for file_path, counts in self.action_counts['file'].items():
            top_files.append({
                'file_path': file_path,
                'count': counts['default'],
                'is_critical': any(file_path.startswith(c) or file_path == c for c in self.critical_files)
            })
            
        # Sort by count, descending
        top_files.sort(key=lambda x: x['count'], reverse=True)
        usage['top_files'] = top_files[:10]  # Return top 10
        
        return usage
        
    def reset_counters(self) -> None:
        """Reset all rate limit counters."""
        if not self.enabled:
            return
            
        self.action_counts = {
            'healing_cycle': collections.defaultdict(int),
            'patch_application': collections.defaultdict(int),
            'deployment': collections.defaultdict(int),
            'file': collections.defaultdict(lambda: collections.defaultdict(int)),
        }
        
        self.time_windows = {
            'healing_cycle': collections.defaultdict(float),
            'patch_application': collections.defaultdict(float),
            'deployment': collections.defaultdict(float),
            'file': collections.defaultdict(lambda: collections.defaultdict(float)),
        }
        
        self.file_actions = collections.defaultdict(list)
        self.last_healing_cycle = 0
        self.file_cooldowns = {}


# Singleton instance for app-wide use
_healing_rate_limiter = None

def get_healing_rate_limiter(config: Dict = None) -> HealingRateLimiter:
    """Get or create the singleton HealingRateLimiter instance.
    
    Args:
        config: Optional configuration for the rate limiter
        
    Returns:
        HealingRateLimiter: The healing rate limiter instance
    """
    global _healing_rate_limiter
    if _healing_rate_limiter is None:
        _healing_rate_limiter = HealingRateLimiter(config)
    return _healing_rate_limiter