"""
End-to-End Healing Scenario Tests

Comprehensive test suite for validating the complete self-healing workflow from
error detection through patch deployment and verification.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
