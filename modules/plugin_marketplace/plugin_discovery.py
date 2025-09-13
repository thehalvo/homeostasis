"""
Plugin Discovery and Registry System

This module provides the core functionality for discovering, loading, and managing
plugins in the USHS ecosystem. It includes manifest validation, dependency resolution,
and plugin lifecycle management.
"""

import hashlib
import json
import logging
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import jsonschema
import semver

logger = logging.getLogger(__name__)


class PluginType(Enum):
    """Plugin categories as defined in the USHS standard."""

    LANGUAGE = "language"
    ANALYSIS = "analysis"
    INTEGRATION = "integration"
    DEPLOYMENT = "deployment"
    MONITORING = "monitoring"


class PluginStatus(Enum):
    """Plugin lifecycle states."""

    DISCOVERED = "discovered"
    VALIDATED = "validated"
    INSTALLED = "installed"
    INITIALIZED = "initialized"
    ACTIVE = "active"
    DISABLED = "disabled"
    ERROR = "error"
    UPDATING = "updating"


class PluginCapability(Enum):
    """Standard plugin capabilities."""

    ERROR_ANALYZE = "error.analyze"
    PATCH_GENERATE = "patch.generate"
    PATCH_VALIDATE = "patch.validate"
    METRICS_EXPORT = "metrics.export"
    TRACE_CORRELATE = "trace.correlate"
    DEPLOY_EXECUTE = "deploy.execute"
    ROLLBACK_PERFORM = "rollback.perform"


class PluginManifest:
    """Represents a plugin manifest with validation."""

    MANIFEST_SCHEMA_PATH = (
        Path(__file__).parent.parent.parent
        / "standards/v1.0/schemas/plugin-manifest.json"
    )

    def __init__(
        self, manifest_data: Dict[str, Any], source_path: Optional[Path] = None
    ):
        """
        Initialize a plugin manifest.

        Args:
            manifest_data: Raw manifest data
            source_path: Path to the manifest file
        """
        self.data = manifest_data
        self.source_path = source_path
        self._validate()

    def _validate(self):
        """Validate manifest against schema."""
        try:
            with open(self.MANIFEST_SCHEMA_PATH, "r") as f:
                schema = json.load(f)

            jsonschema.validate(self.data, schema)
        except jsonschema.ValidationError as e:
            raise ValueError(f"Invalid plugin manifest: {e.message}")
        except FileNotFoundError:
            logger.warning("Manifest schema not found, skipping validation")

    @property
    def name(self) -> str:
        """Get plugin name."""
        return self.data["name"]

    @property
    def version(self) -> str:
        """Get plugin version."""
        return self.data["version"]

    @property
    def type(self) -> PluginType:
        """Get plugin type."""
        return PluginType(self.data["type"])

    @property
    def display_name(self) -> str:
        """Get display name."""
        return self.data["displayName"]

    @property
    def description(self) -> str:
        """Get description."""
        return self.data["description"]

    @property
    def required_capabilities(self) -> Set[str]:
        """Get required capabilities."""
        return set(self.data["capabilities"]["required"])

    @property
    def optional_capabilities(self) -> Set[str]:
        """Get optional capabilities."""
        return set(self.data["capabilities"].get("optional", []))

    @property
    def permissions(self) -> Dict[str, List[str]]:
        """Get permission requirements."""
        return self.data.get("permissions", {})

    @property
    def engines(self) -> Dict[str, str]:
        """Get engine requirements."""
        return self.data["engines"]

    def is_compatible_with_ushs(self, ushs_version: str) -> bool:
        """
        Check if plugin is compatible with USHS version.

        Args:
            ushs_version: Current USHS version

        Returns:
            True if compatible
        """
        required_range = self.engines.get("ushs", "*")
        if required_range == "*":
            return True

        try:
            # Parse version requirement
            return semver.match(ushs_version, required_range)
        except ValueError:
            logger.warning(f"Invalid version requirement: {required_range}")
            return False


class PluginInfo:
    """Complete information about a discovered plugin."""

    def __init__(self, manifest: PluginManifest, path: Path):
        """
        Initialize plugin info.

        Args:
            manifest: Plugin manifest
            path: Plugin directory path
        """
        self.manifest = manifest
        self.path = path
        self.status = PluginStatus.DISCOVERED
        self.loaded_at = None
        self.error = None
        self.instance = None
        self.metadata: Dict[str, Any] = {}

    @property
    def id(self) -> str:
        """Get unique plugin identifier."""
        return f"{self.manifest.name}@{self.manifest.version}"

    @property
    def checksum(self) -> str:
        """Calculate plugin checksum for integrity verification."""
        hasher = hashlib.sha256()

        # Hash all plugin files
        for file_path in sorted(self.path.rglob("*")):
            if file_path.is_file():
                hasher.update(str(file_path.relative_to(self.path)).encode())
                hasher.update(file_path.read_bytes())

        return hasher.hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "name": self.manifest.name,
            "version": self.manifest.version,
            "type": self.manifest.type.value,
            "displayName": self.manifest.display_name,
            "description": self.manifest.description,
            "status": self.status.value,
            "path": str(self.path),
            "checksum": self.checksum,
            "loadedAt": self.loaded_at.isoformat() if self.loaded_at else None,
            "error": self.error,
            "metadata": self.metadata,
        }


class PluginDiscovery:
    """Discovers plugins from various sources."""

    def __init__(self, search_paths: Optional[List[Path]] = None):
        """
        Initialize plugin discovery.

        Args:
            search_paths: List of paths to search for plugins
        """
        self.search_paths = search_paths or self._default_search_paths()

    def _default_search_paths(self) -> List[Path]:
        """Get default plugin search paths."""
        paths = []

        # Built-in plugins directory
        builtin_path = Path(__file__).parent.parent / "analysis/plugins"
        if builtin_path.exists():
            paths.append(builtin_path)

        # User plugins directory
        user_path = Path.home() / ".homeostasis/plugins"
        if user_path.exists():
            paths.append(user_path)

        # System plugins directory
        system_path = Path("/usr/local/share/homeostasis/plugins")
        if system_path.exists():
            paths.append(system_path)

        return paths

    def discover_all(self) -> List[PluginInfo]:
        """
        Discover all plugins in search paths.

        Returns:
            List of discovered plugins
        """
        plugins: List[PluginInfo] = []

        for search_path in self.search_paths:
            plugins.extend(self.discover_in_path(search_path))

        return plugins

    def discover_in_path(self, path: Path) -> List[PluginInfo]:
        """
        Discover plugins in a specific path.

        Args:
            path: Directory to search

        Returns:
            List of discovered plugins
        """
        plugins: List[PluginInfo] = []

        if not path.exists():
            return plugins

        # Look for plugin directories
        for item in path.iterdir():
            if item.is_dir():
                plugin = self._discover_plugin(item)
                if plugin:
                    plugins.append(plugin)

        return plugins

    def _discover_plugin(self, path: Path) -> Optional[PluginInfo]:
        """
        Discover a single plugin.

        Args:
            path: Plugin directory path

        Returns:
            Plugin info or None if not a valid plugin
        """
        manifest_path = path / "manifest.json"

        if not manifest_path.exists():
            return None

        try:
            with open(manifest_path, "r") as f:
                manifest_data = json.load(f)

            manifest = PluginManifest(manifest_data, manifest_path)
            return PluginInfo(manifest, path)

        except Exception as e:
            logger.error(f"Failed to load plugin from {path}: {e}")
            return None


class PluginRegistry:
    """Central registry for all discovered plugins."""

    def __init__(self, ushs_version: str = "1.0.0"):
        """
        Initialize plugin registry.

        Args:
            ushs_version: Current USHS version
        """
        self.ushs_version = ushs_version
        self.plugins: Dict[str, PluginInfo] = {}
        self.discovery = PluginDiscovery()
        self._capabilities_index: Dict[str, Set[str]] = {}
        self._type_index: Dict[PluginType, Set[str]] = {}

    def discover_and_register(self) -> int:
        """
        Discover and register all available plugins.

        Returns:
            Number of plugins registered
        """
        discovered = self.discovery.discover_all()
        count = 0

        for plugin_info in discovered:
            if self.register(plugin_info):
                count += 1

        return count

    def register(self, plugin_info: PluginInfo) -> bool:
        """
        Register a plugin.

        Args:
            plugin_info: Plugin to register

        Returns:
            True if registered successfully
        """
        # Check compatibility
        if not plugin_info.manifest.is_compatible_with_ushs(self.ushs_version):
            logger.warning(
                f"Plugin {plugin_info.id} is not compatible with USHS {self.ushs_version}"
            )
            return False

        # Check for conflicts
        if plugin_info.id in self.plugins:
            logger.warning(f"Plugin {plugin_info.id} is already registered")
            return False

        # Register plugin
        self.plugins[plugin_info.id] = plugin_info
        plugin_info.status = PluginStatus.VALIDATED

        # Update indexes
        self._update_indexes(plugin_info)

        logger.info(f"Registered plugin: {plugin_info.id}")
        return True

    def _update_indexes(self, plugin_info: PluginInfo):
        """Update capability and type indexes."""
        # Update capability index
        for capability in plugin_info.manifest.required_capabilities:
            if capability not in self._capabilities_index:
                self._capabilities_index[capability] = set()
            self._capabilities_index[capability].add(plugin_info.id)

        # Update type index
        plugin_type = plugin_info.manifest.type
        if plugin_type not in self._type_index:
            self._type_index[plugin_type] = set()
        self._type_index[plugin_type].add(plugin_info.id)

    def get_plugin(self, plugin_id: str) -> Optional[PluginInfo]:
        """
        Get a plugin by ID.

        Args:
            plugin_id: Plugin identifier (name@version)

        Returns:
            Plugin info or None
        """
        return self.plugins.get(plugin_id)

    def get_plugins_by_type(self, plugin_type: PluginType) -> List[PluginInfo]:
        """
        Get all plugins of a specific type.

        Args:
            plugin_type: Plugin type to filter by

        Returns:
            List of matching plugins
        """
        plugin_ids = self._type_index.get(plugin_type, set())
        return [self.plugins[pid] for pid in plugin_ids]

    def get_plugins_by_capability(self, capability: str) -> List[PluginInfo]:
        """
        Get all plugins providing a specific capability.

        Args:
            capability: Capability to search for

        Returns:
            List of matching plugins
        """
        plugin_ids = self._capabilities_index.get(capability, set())
        return [self.plugins[pid] for pid in plugin_ids]

    def find_plugins(self, **criteria) -> List[PluginInfo]:
        """
        Find plugins matching multiple criteria.

        Args:
            **criteria: Search criteria (type, capabilities, name, etc.)

        Returns:
            List of matching plugins
        """
        results = list(self.plugins.values())

        # Filter by type
        if "type" in criteria:
            plugin_type = PluginType(criteria["type"])
            results = [p for p in results if p.manifest.type == plugin_type]

        # Filter by capabilities
        if "capabilities" in criteria:
            required_caps = set(criteria["capabilities"])
            results = [
                p
                for p in results
                if required_caps.issubset(
                    p.manifest.required_capabilities | p.manifest.optional_capabilities
                )
            ]

        # Filter by name pattern
        if "name" in criteria:
            pattern = criteria["name"].lower()
            results = [
                p
                for p in results
                if pattern in p.manifest.name.lower()
                or pattern in p.manifest.display_name.lower()
            ]

        # Filter by status
        if "status" in criteria:
            status = PluginStatus(criteria["status"])
            results = [p for p in results if p.status == status]

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics."""
        stats: Dict[str, Any] = {
            "total": len(self.plugins),
            "by_type": {},
            "by_status": {},
            "capabilities": list(self._capabilities_index.keys()),
        }

        # Count by type
        for plugin_type in PluginType:
            count = len(self._type_index.get(plugin_type, set()))
            stats["by_type"][plugin_type.value] = count

        # Count by status
        for status in PluginStatus:
            count = sum(1 for p in self.plugins.values() if p.status == status)
            stats["by_status"][status.value] = count

        return stats


class PluginValidator:
    """Validates plugins for security and quality."""

    def __init__(self):
        """Initialize plugin validator."""
        self.security_checks = [
            self._check_permissions,
            self._check_dependencies,
            self._check_code_signatures,
            self._check_known_vulnerabilities,
        ]

        self.quality_checks = [
            self._check_documentation,
            self._check_tests,
            self._check_code_quality,
            self._check_performance,
        ]

    def validate(self, plugin_info: PluginInfo) -> Tuple[bool, List[str]]:
        """
        Validate a plugin.

        Args:
            plugin_info: Plugin to validate

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues: List[str] = []

        # Run security checks
        for check in self.security_checks:
            check_issues = check(plugin_info)
            issues.extend(check_issues)

        # Run quality checks (warnings only)
        for check in self.quality_checks:
            check_issues = check(plugin_info)
            for issue in check_issues:
                issues.append(f"[Quality] {issue}")

        # Only security issues make validation fail
        security_issues = [i for i in issues if not i.startswith("[Quality]")]
        is_valid = len(security_issues) == 0

        return is_valid, issues

    def _check_permissions(self, plugin_info: PluginInfo) -> List[str]:
        """Check permission requirements."""
        issues: List[str] = []
        permissions = plugin_info.manifest.permissions

        # Check for dangerous permissions
        if "filesystem" in permissions and "write" in permissions["filesystem"]:
            if "process" in permissions and "spawn" in permissions["process"]:
                issues.append(
                    "Plugin requests both write and process spawn permissions"
                )

        # Check network permissions
        if "network" in permissions:
            for endpoint in permissions["network"]:
                if not endpoint.startswith("https://"):
                    issues.append(f"Insecure network endpoint: {endpoint}")

        return issues

    def _check_dependencies(self, plugin_info: PluginInfo) -> List[str]:
        """Check dependency security."""
        issues: List[str] = []

        # This would integrate with vulnerability databases
        # For now, just a placeholder

        return issues

    def _check_code_signatures(self, plugin_info: PluginInfo) -> List[str]:
        """Check code signatures."""
        issues: List[str] = []

        # Check for signature file
        sig_file = plugin_info.path / "manifest.json.sig"
        if not sig_file.exists():
            issues.append("Plugin is not signed")

        return issues

    def _check_known_vulnerabilities(self, plugin_info: PluginInfo) -> List[str]:
        """Check for known vulnerabilities."""
        issues: List[str] = []

        # This would check against CVE databases
        # For now, just a placeholder

        return issues

    def _check_documentation(self, plugin_info: PluginInfo) -> List[str]:
        """Check documentation completeness."""
        issues: List[str] = []

        required_docs = ["README.md", "LICENSE", "CHANGELOG.md"]
        for doc in required_docs:
            if not (plugin_info.path / doc).exists():
                issues.append(f"Missing required documentation: {doc}")

        return issues

    def _check_tests(self, plugin_info: PluginInfo) -> List[str]:
        """Check test coverage."""
        issues: List[str] = []

        test_dir = plugin_info.path / "tests"
        if not test_dir.exists():
            issues.append("No tests directory found")

        return issues

    def _check_code_quality(self, plugin_info: PluginInfo) -> List[str]:
        """Check code quality metrics."""
        issues: List[str] = []

        # This would run linters and static analysis
        # For now, just a placeholder

        return issues

    def _check_performance(self, plugin_info: PluginInfo) -> List[str]:
        """Check performance requirements."""
        issues: List[str] = []

        # This would run performance benchmarks
        # For now, just a placeholder

        return issues


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Create registry
    registry = PluginRegistry()

    # Discover plugins
    count = registry.discover_and_register()
    logger.info(f"Discovered and registered {count} plugins")

    # Get statistics
    stats = registry.get_statistics()
    logger.info(f"Registry statistics: {json.dumps(stats, indent=2)}")

    # Find language plugins
    language_plugins = registry.get_plugins_by_type(PluginType.LANGUAGE)
    logger.info(f"Found {len(language_plugins)} language plugins")

    # Find plugins with specific capability
    analyze_plugins = registry.get_plugins_by_capability(
        PluginCapability.ERROR_ANALYZE.value
    )
    logger.info(f"Found {len(analyze_plugins)} plugins with error analysis capability")

    # Validate plugins
    validator = PluginValidator()
    for plugin_id, plugin_info in registry.plugins.items():
        is_valid, issues = validator.validate(plugin_info)
        if issues:
            logger.warning(f"Plugin {plugin_id} validation issues: {issues}")
