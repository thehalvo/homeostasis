#!/usr/bin/env python3
"""
Tests for the HealingRateLimiter class.
"""
import time
import unittest

from modules.security.healing_rate_limiter import HealingRateLimiter


class TestHealingRateLimiter(unittest.TestCase):
    """Tests for the HealingRateLimiter class."""

    def setUp(self):
        """Set up for the tests."""
        self.config = {
            "enabled": True,
            "min_healing_cycle_interval": 10,
            "limits": {
                "healing_cycle": [3, 60],  # 3 healing cycles per 60 seconds
                "patch_application": [5, 60],  # 5 patch applications per 60 seconds
                "deployment": [2, 60],  # 2 deployments per 60 seconds
                "file": [2, 60],  # 2 patches per file per 60 seconds
                "critical_file": [1, 60],  # 1 patch per critical file per 60 seconds
            },
            "critical_files": ["critical_file.py", "critical_dir/"],
        }
        self.rate_limiter = HealingRateLimiter(self.config)

    def test_healing_cycle_limit(self):
        """Test healing cycle rate limiting."""
        # Temporarily disable minimum interval to test rate limiting only
        original_interval = self.config.get("min_healing_cycle_interval", 10)
        self.rate_limiter.config["min_healing_cycle_interval"] = 0

        # First cycle should be allowed
        self.assertTrue(self.rate_limiter.check_healing_cycle_limit())

        # Second cycle should be allowed
        self.assertTrue(self.rate_limiter.check_healing_cycle_limit())

        # Third cycle should be allowed
        self.assertTrue(self.rate_limiter.check_healing_cycle_limit())

        # Fourth cycle should be blocked (exceeds limit of 3)
        self.assertFalse(self.rate_limiter.check_healing_cycle_limit())

        # Restore original interval
        self.rate_limiter.config["min_healing_cycle_interval"] = original_interval

    def test_healing_cycle_minimum_interval(self):
        """Test minimum interval between healing cycles."""
        # First cycle should be allowed
        self.assertTrue(self.rate_limiter.check_healing_cycle_limit())

        # Second cycle should be blocked due to minimum interval
        self.assertFalse(self.rate_limiter.check_healing_cycle_limit())

        # Manually set the last healing cycle time to simulate waiting
        self.rate_limiter.last_healing_cycle = time.time() - 11  # 11 seconds ago

        # Now it should be allowed
        self.assertTrue(self.rate_limiter.check_healing_cycle_limit())

    def test_patch_application_limit(self):
        """Test patch application rate limiting."""
        # Apply 2 patches at once
        self.assertTrue(self.rate_limiter.check_patch_application_limit(2))

        # Apply 2 more patches at once
        self.assertTrue(self.rate_limiter.check_patch_application_limit(2))

        # Apply 2 more patches - should be blocked (exceeds limit of 5)
        self.assertFalse(self.rate_limiter.check_patch_application_limit(2))

        # Apply just 1 more patch - should be allowed
        self.assertTrue(self.rate_limiter.check_patch_application_limit(1))

        # Now we're at the limit, so any more should be blocked
        self.assertFalse(self.rate_limiter.check_patch_application_limit(1))

    def test_deployment_limit(self):
        """Test deployment rate limiting."""
        # First deployment should be allowed
        self.assertTrue(self.rate_limiter.check_deployment_limit())

        # Second deployment should be allowed
        self.assertTrue(self.rate_limiter.check_deployment_limit())

        # Third deployment should be blocked (exceeds limit of 2)
        self.assertFalse(self.rate_limiter.check_deployment_limit())

    def test_file_limit(self):
        """Test file-specific rate limiting."""
        # Regular file modifications
        test_file = "test_file.py"

        # First modification should be allowed
        self.assertTrue(self.rate_limiter.check_file_limit(test_file))

        # Second modification should be allowed
        self.assertTrue(self.rate_limiter.check_file_limit(test_file))

        # Third modification should be blocked (exceeds limit of 2)
        self.assertFalse(self.rate_limiter.check_file_limit(test_file))

        # Critical file modifications
        critical_file = "critical_file.py"

        # First modification should be allowed
        self.assertTrue(self.rate_limiter.check_file_limit(critical_file))

        # Second modification should be blocked (exceeds limit of 1 for critical files)
        self.assertFalse(self.rate_limiter.check_file_limit(critical_file))

        # File in critical directory
        critical_dir_file = "critical_dir/some_file.py"

        # First modification should be allowed
        self.assertTrue(self.rate_limiter.check_file_limit(critical_dir_file))

        # Second modification should be blocked (exceeds limit of 1 for critical files)
        self.assertFalse(self.rate_limiter.check_file_limit(critical_dir_file))

    def test_file_cooldown(self):
        """Test file cooldown functionality."""
        test_file = "test_file.py"

        # Place file in cooldown
        self.rate_limiter.place_file_in_cooldown(
            test_file, 1
        )  # 1 second cooldown for test

        # File should be blocked during cooldown
        self.assertFalse(self.rate_limiter.check_file_limit(test_file))

        # Wait for cooldown to expire
        time.sleep(1.5)

        # Now it should be allowed
        self.assertTrue(self.rate_limiter.check_file_limit(test_file))

    def test_reset_counters(self):
        """Test counter reset functionality."""
        # Set up some limits
        self.rate_limiter.check_healing_cycle_limit()
        self.rate_limiter.check_patch_application_limit(2)
        self.rate_limiter.check_deployment_limit()
        self.rate_limiter.check_file_limit("test_file.py")

        # Reset counters
        self.rate_limiter.reset_counters()

        # Verify counters are reset
        self.assertEqual(self.rate_limiter.action_counts["healing_cycle"]["global"], 0)
        self.assertEqual(
            self.rate_limiter.action_counts["patch_application"]["global"], 0
        )
        self.assertEqual(self.rate_limiter.action_counts["deployment"]["global"], 0)
        self.assertEqual(
            self.rate_limiter.action_counts["file"]["test_file.py"]["default"], 0
        )

    def test_get_current_limits(self):
        """Test getting current limits."""
        limits = self.rate_limiter.get_current_limits()

        # Verify limits match config
        self.assertEqual(limits["healing_cycle"], (3, 60))
        self.assertEqual(limits["patch_application"], (5, 60))
        self.assertEqual(limits["deployment"], (2, 60))
        self.assertEqual(limits["file"], (2, 60))
        self.assertEqual(limits["critical_file"], (1, 60))

    def test_get_usage_stats(self):
        """Test getting usage statistics."""
        # Set up some usage
        self.rate_limiter.check_healing_cycle_limit()
        self.rate_limiter.check_patch_application_limit(2)
        self.rate_limiter.check_deployment_limit()
        self.rate_limiter.check_file_limit("test_file.py")
        self.rate_limiter.check_file_limit("critical_file.py")

        # Place a file in cooldown
        self.rate_limiter.place_file_in_cooldown("cooldown_file.py", 60)

        # Get usage stats
        stats = self.rate_limiter.get_usage_stats()

        # Verify stats include expected data
        self.assertEqual(stats["healing_cycle"]["count"], 1)
        self.assertEqual(stats["patch_application"]["count"], 2)
        self.assertEqual(stats["deployment"]["count"], 1)
        self.assertIn("cooldown_file.py", stats["cooldowns"])
        self.assertEqual(
            len(stats["top_files"]), 2
        )  # test_file.py and critical_file.py


if __name__ == "__main__":
    unittest.main()
