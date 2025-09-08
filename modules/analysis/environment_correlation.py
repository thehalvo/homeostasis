"""
Environmental factor correlation system for error analysis.

This module identifies relationships between errors and environmental factors,
helping to detect patterns in when and why errors occur.
"""

import datetime
import logging
import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class EnvironmentalFactor:
    """Represents an environmental factor that may correlate with errors."""

    name: str  # Name of the factor
    value: Any  # Value of the factor
    source: str  # Source of the factor (e.g., "system", "application", "network")
    importance: float = 0.0  # Importance score (0.0 to 1.0)
    description: str = ""  # Description of the factor
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata


class EnvironmentCollector:
    """
    Collects environmental factors from various sources.
    """

    def __init__(self):
        """Initialize the environment collector."""
        # Register factor collectors
        self.collectors = {
            "system": self._collect_system_factors,
            "application": self._collect_application_factors,
            "network": self._collect_network_factors,
            "database": self._collect_database_factors,
            "deployment": self._collect_deployment_factors,
            "custom": self._collect_custom_factors,
        }

    def collect_all_factors(
        self, custom_factors: Optional[Dict[str, Any]] = None
    ) -> Dict[str, EnvironmentalFactor]:
        """
        Collect all available environmental factors.

        Args:
            custom_factors: Custom factors to include

        Returns:
            Dictionary of environmental factors
        """
        factors = {}

        # Collect from each source
        for source, collector in self.collectors.items():
            try:
                source_factors = collector(custom_factors)
                factors.update(source_factors)
                logger.debug(f"Collected {len(source_factors)} factors from {source}")
            except Exception as e:
                logger.error(f"Error collecting {source} factors: {str(e)}")

        return factors

    def _collect_system_factors(
        self, custom_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, EnvironmentalFactor]:
        """
        Collect system-level environmental factors.

        Args:
            custom_data: Custom data to include

        Returns:
            Dictionary of environmental factors
        """
        factors = {}

        # Operating system information
        try:
            import platform

            os_name = platform.system()
            os_version = platform.release()
            os_platform = platform.platform()
            python_version = platform.python_version()

            factors["system.os"] = EnvironmentalFactor(
                name="Operating System",
                value=os_name,
                source="system",
                importance=0.7,
                description="Operating system type",
            )

            factors["system.os_version"] = EnvironmentalFactor(
                name="OS Version",
                value=os_version,
                source="system",
                importance=0.6,
                description="Operating system version",
            )

            factors["system.platform"] = EnvironmentalFactor(
                name="Platform",
                value=os_platform,
                source="system",
                importance=0.5,
                description="Full platform information",
            )

            factors["system.python_version"] = EnvironmentalFactor(
                name="Python Version",
                value=python_version,
                source="system",
                importance=0.8,
                description="Python interpreter version",
            )
        except Exception as e:
            logger.warning(f"Error collecting platform information: {str(e)}")

        # System resources
        try:
            import psutil

            cpu_count = os.cpu_count() or psutil.cpu_count()
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            disk = psutil.disk_usage("/")
            disk_percent = disk.percent

            factors["system.cpu_count"] = EnvironmentalFactor(
                name="CPU Count",
                value=cpu_count,
                source="system",
                importance=0.6,
                description="Number of CPU cores",
            )

            factors["system.cpu_usage"] = EnvironmentalFactor(
                name="CPU Usage",
                value=cpu_percent,
                source="system",
                importance=0.8,
                description="CPU utilization percentage",
            )

            factors["system.memory_usage"] = EnvironmentalFactor(
                name="Memory Usage",
                value=memory_percent,
                source="system",
                importance=0.8,
                description="Memory utilization percentage",
            )

            factors["system.disk_usage"] = EnvironmentalFactor(
                name="Disk Usage",
                value=disk_percent,
                source="system",
                importance=0.7,
                description="Disk utilization percentage",
            )
        except ImportError:
            logger.warning("psutil not available, skipping system resource metrics")
        except Exception as e:
            logger.warning(f"Error collecting system resources: {str(e)}")

        # Time-related factors
        now = datetime.datetime.now()
        factors["system.time.hour"] = EnvironmentalFactor(
            name="Hour of Day",
            value=now.hour,
            source="system",
            importance=0.5,
            description="Hour of the day (0-23)",
        )

        factors["system.time.day_of_week"] = EnvironmentalFactor(
            name="Day of Week",
            value=now.weekday(),
            source="system",
            importance=0.4,
            description="Day of the week (0=Monday, 6=Sunday)",
        )

        factors["system.time.is_weekend"] = EnvironmentalFactor(
            name="Is Weekend",
            value=now.weekday() >= 5,  # 5=Saturday, 6=Sunday
            source="system",
            importance=0.4,
            description="Whether it's a weekend",
        )

        factors["system.time.month"] = EnvironmentalFactor(
            name="Month",
            value=now.month,
            source="system",
            importance=0.3,
            description="Month of the year (1-12)",
        )

        return factors

    def _collect_application_factors(
        self, custom_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, EnvironmentalFactor]:
        """
        Collect application-level environmental factors.

        Args:
            custom_data: Custom data to include

        Returns:
            Dictionary of environmental factors
        """
        factors = {}

        # Check for environment variables with application info
        env_prefixes = ["APP_", "APPLICATION_", "SERVICE_"]

        for key, value in os.environ.items():
            for prefix in env_prefixes:
                if key.startswith(prefix):
                    # Found an application-related environment variable
                    factor_name = key.lower().replace(prefix.lower(), "")
                    factors[f"application.{factor_name}"] = EnvironmentalFactor(
                        name=factor_name,
                        value=value,
                        source="application",
                        importance=0.7,
                        description=f"Application {factor_name} from environment variable",
                    )

        # Application version if available
        try:
            # Check common version files
            version_files = ["VERSION", "version.txt", "../VERSION", "../version.txt"]

            for file_path in version_files:
                if os.path.exists(file_path):
                    with open(file_path, "r") as f:
                        version = f.read().strip()
                        factors["application.version"] = EnvironmentalFactor(
                            name="Application Version",
                            value=version,
                            source="application",
                            importance=0.9,
                            description="Application version number",
                        )
                        break
        except Exception as e:
            logger.warning(f"Error reading application version: {str(e)}")

        # Check for custom application factors
        if custom_data and "application" in custom_data:
            app_data = custom_data["application"]

            for key, value in app_data.items():
                factors[f"application.{key}"] = EnvironmentalFactor(
                    name=key,
                    value=value,
                    source="application",
                    importance=0.8,
                    description=f"Custom application {key}",
                )

        return factors

    def _collect_network_factors(
        self, custom_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, EnvironmentalFactor]:
        """
        Collect network-related environmental factors.

        Args:
            custom_data: Custom data to include

        Returns:
            Dictionary of environmental factors
        """
        factors = {}

        # Check for network connectivity
        try:
            import socket

            # Hostname
            hostname = socket.gethostname()
            factors["network.hostname"] = EnvironmentalFactor(
                name="Hostname",
                value=hostname,
                source="network",
                importance=0.6,
                description="System hostname",
            )

            # Try to resolve a domain as a connectivity check
            try:
                # Use a timeout to avoid blocking
                start_time = time.time()
                socket.gethostbyname("www.google.com")
                resolution_time = time.time() - start_time

                factors["network.dns_resolution"] = EnvironmentalFactor(
                    name="DNS Resolution",
                    value=True,
                    source="network",
                    importance=0.7,
                    description="DNS resolution availability",
                    metadata={"resolution_time": resolution_time},
                )
            except socket.gaierror:
                factors["network.dns_resolution"] = EnvironmentalFactor(
                    name="DNS Resolution",
                    value=False,
                    source="network",
                    importance=0.7,
                    description="DNS resolution availability",
                )

            # Check internet connectivity using a secure approach
            try:
                # Use a short timeout and validate the URL scheme
                start_time = time.time()
                
                # Use a well-known public DNS server (Google DNS) for connectivity check
                # This is safer than opening URLs
                test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                test_socket.settimeout(2)
                # Try to connect to Google's public DNS on port 53
                result = test_socket.connect_ex(("8.8.8.8", 53))
                test_socket.close()
                
                response_time = time.time() - start_time
                
                if result == 0:
                    factors["network.internet_connectivity"] = EnvironmentalFactor(
                        name="Internet Connectivity",
                        value=True,
                        source="network",
                        importance=0.8,
                        description="Internet connectivity status",
                        metadata={"response_time": response_time, "method": "dns_socket"},
                    )
                else:
                    raise socket.error("Connection failed")
            except Exception:
                factors["network.internet_connectivity"] = EnvironmentalFactor(
                    name="Internet Connectivity",
                    value=False,
                    source="network",
                    importance=0.8,
                    description="Internet connectivity status",
                )
        except Exception as e:
            logger.warning(f"Error collecting network factors: {str(e)}")

        # Check for custom network factors
        if custom_data and "network" in custom_data:
            net_data = custom_data["network"]

            for key, value in net_data.items():
                factors[f"network.{key}"] = EnvironmentalFactor(
                    name=key,
                    value=value,
                    source="network",
                    importance=0.8,
                    description=f"Custom network {key}",
                )

        return factors

    def _collect_database_factors(
        self, custom_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, EnvironmentalFactor]:
        """
        Collect database-related environmental factors.

        Args:
            custom_data: Custom data to include

        Returns:
            Dictionary of environmental factors
        """
        factors = {}

        # Check for database connectivity from environment variables
        db_vars = {
            k: v
            for k, v in os.environ.items()
            if "DB_" in k or "DATABASE_" in k or "SQL_" in k
        }

        if db_vars:
            # Extract database type if possible
            db_type = None
            for key in db_vars.keys():
                if "TYPE" in key or "ENGINE" in key:
                    db_type = db_vars[key]
                    break

            if not db_type:
                # Try to infer from other variables
                if any("POSTGRES" in k for k in db_vars.keys()):
                    db_type = "postgresql"
                elif any("MYSQL" in k for k in db_vars.keys()):
                    db_type = "mysql"
                elif any("SQLITE" in k for k in db_vars.keys()):
                    db_type = "sqlite"
                elif any("MONGO" in k for k in db_vars.keys()):
                    db_type = "mongodb"
                else:
                    db_type = "unknown"

            factors["database.type"] = EnvironmentalFactor(
                name="Database Type",
                value=db_type,
                source="database",
                importance=0.7,
                description="Type of database in use",
            )

            # Look for host information
            db_host = None
            for key in db_vars.keys():
                if "HOST" in key:
                    db_host = db_vars[key]
                    break

            if db_host:
                factors["database.host"] = EnvironmentalFactor(
                    name="Database Host",
                    value=db_host,
                    source="database",
                    importance=0.6,
                    description="Database host location",
                )

        # Check for custom database factors
        if custom_data and "database" in custom_data:
            db_data = custom_data["database"]

            for key, value in db_data.items():
                # Hide sensitive information
                if key.lower() in ["password", "secret", "key"]:
                    value = "[REDACTED]"

                factors[f"database.{key}"] = EnvironmentalFactor(
                    name=key,
                    value=value,
                    source="database",
                    importance=0.8,
                    description=f"Custom database {key}",
                )

        return factors

    def _collect_deployment_factors(
        self, custom_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, EnvironmentalFactor]:
        """
        Collect deployment-related environmental factors.

        Args:
            custom_data: Custom data to include

        Returns:
            Dictionary of environmental factors
        """
        factors = {}

        # Check for common deployment environment variables
        for env_var, factor_name in [
            ("ENVIRONMENT", "environment"),
            ("ENV", "environment"),
            ("STAGE", "stage"),
            ("DEPLOYMENT_ENV", "environment"),
            ("NODE_ENV", "environment"),
            ("FLASK_ENV", "environment"),
            ("DJANGO_ENV", "environment"),
            ("KUBERNETES_SERVICE_HOST", "kubernetes"),
            ("DOCKER", "docker"),
            ("CONTAINER", "container"),
        ]:
            if env_var in os.environ:
                factors[f"deployment.{factor_name}"] = EnvironmentalFactor(
                    name=factor_name.capitalize(),
                    value=os.environ[env_var],
                    source="deployment",
                    importance=0.8,
                    description=f"Deployment {factor_name}",
                )

        # Check for Git information
        try:
            import subprocess

            # Check if we're in a git repository
            try:
                git_result = subprocess.run(
                    ["git", "rev-parse", "--is-inside-work-tree"],
                    capture_output=True,
                    text=True,
                    check=False,
                )

                if git_result.returncode == 0 and git_result.stdout.strip() == "true":
                    # Get current branch
                    branch_result = subprocess.run(
                        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                        capture_output=True,
                        text=True,
                        check=False,
                    )

                    if branch_result.returncode == 0:
                        branch = branch_result.stdout.strip()
                        factors["deployment.git_branch"] = EnvironmentalFactor(
                            name="Git Branch",
                            value=branch,
                            source="deployment",
                            importance=0.7,
                            description="Git branch name",
                        )

                    # Get latest commit hash
                    commit_result = subprocess.run(
                        ["git", "rev-parse", "HEAD"],
                        capture_output=True,
                        text=True,
                        check=False,
                    )

                    if commit_result.returncode == 0:
                        commit = commit_result.stdout.strip()
                        factors["deployment.git_commit"] = EnvironmentalFactor(
                            name="Git Commit",
                            value=commit,
                            source="deployment",
                            importance=0.7,
                            description="Git commit hash",
                        )
            except Exception:
                pass  # Git commands failed, not in a git repo or git not installed
        except Exception as e:
            logger.warning(f"Error collecting git information: {str(e)}")

        # Check for custom deployment factors
        if custom_data and "deployment" in custom_data:
            deploy_data = custom_data["deployment"]

            for key, value in deploy_data.items():
                factors[f"deployment.{key}"] = EnvironmentalFactor(
                    name=key,
                    value=value,
                    source="deployment",
                    importance=0.8,
                    description=f"Custom deployment {key}",
                )

        return factors

    def _collect_custom_factors(
        self, custom_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, EnvironmentalFactor]:
        """
        Collect custom environmental factors.

        Args:
            custom_data: Custom data to include

        Returns:
            Dictionary of environmental factors
        """
        factors = {}

        if not custom_data or not isinstance(custom_data, dict):
            return factors

        # Process any custom factors that don't fit into other categories
        if "custom" in custom_data:
            custom_factors = custom_data["custom"]

            for key, value in custom_factors.items():
                # Extract importance if provided
                importance = 0.5
                description = f"Custom factor: {key}"

                if isinstance(value, dict) and "value" in value:
                    # Format: {"value": actual_value, "importance": importance, "description": description}
                    actual_value = value.get("value")
                    importance = value.get("importance", 0.5)
                    description = value.get("description", description)
                    metadata = {
                        k: v
                        for k, v in value.items()
                        if k not in ["value", "importance", "description"]
                    }

                    factors[f"custom.{key}"] = EnvironmentalFactor(
                        name=key,
                        value=actual_value,
                        source="custom",
                        importance=importance,
                        description=description,
                        metadata=metadata,
                    )
                else:
                    # Simple format: key -> value
                    factors[f"custom.{key}"] = EnvironmentalFactor(
                        name=key,
                        value=value,
                        source="custom",
                        importance=importance,
                        description=description,
                    )

        # Also process any top-level custom factors
        for key, value in custom_data.items():
            if key not in [
                "system",
                "application",
                "network",
                "database",
                "deployment",
                "custom",
            ]:
                factors[f"custom.{key}"] = EnvironmentalFactor(
                    name=key,
                    value=value,
                    source="custom",
                    importance=0.5,
                    description=f"Custom factor: {key}",
                )

        return factors


class EnvironmentCorrelator:
    """
    Correlates errors with environmental factors.
    """

    def __init__(self):
        """Initialize the environment correlator."""
        self.collector = EnvironmentCollector()
        self.error_occurrences = defaultdict(list)  # Maps error types to occurrences
        self.factor_values = defaultdict(list)  # Maps factor names to values over time
        self.error_factors = defaultdict(list)  # Maps error occurrences to factor sets

    def capture_environment(
        self, custom_factors: Optional[Dict[str, Any]] = None
    ) -> Dict[str, EnvironmentalFactor]:
        """
        Capture the current environment.

        Args:
            custom_factors: Custom factors to include

        Returns:
            Dictionary of environmental factors
        """
        return self.collector.collect_all_factors(custom_factors)

    def record_error(
        self,
        error_data: Dict[str, Any],
        environmental_factors: Optional[Dict[str, EnvironmentalFactor]] = None,
        custom_factors: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Record an error with its environmental context.

        Args:
            error_data: Error data dictionary
            environmental_factors: Pre-collected environmental factors
            custom_factors: Custom factors to include

        Returns:
            Dictionary with error and environment information
        """
        # Extract error type
        error_type = error_data.get("exception_type", "")
        if not error_type and "error_details" in error_data:
            error_type = error_data["error_details"].get("exception_type", "")
        if not error_type:
            error_type = "unknown"

        # Extract error ID
        error_id = error_data.get(
            "_id", f"error_{len(self.error_occurrences[error_type])}"
        )

        # Collect environment if not provided
        if environmental_factors is None:
            environmental_factors = self.collector.collect_all_factors(custom_factors)

        # Record error occurrence
        occurrence = {
            "error_id": error_id,
            "error_type": error_type,
            "timestamp": error_data.get(
                "timestamp", datetime.datetime.now().isoformat()
            ),
            "message": error_data.get("message", ""),
            "environment": {
                name: factor.value for name, factor in environmental_factors.items()
            },
        }

        self.error_occurrences[error_type].append(occurrence)

        # Record factor values
        for name, factor in environmental_factors.items():
            self.factor_values[name].append(factor.value)

        # Associate error with factors
        self.error_factors[error_id] = environmental_factors

        return occurrence

    def analyze_correlations(
        self, error_type: Optional[str] = None, min_occurrences: int = 5
    ) -> Dict[str, Any]:
        """
        Analyze correlations between errors and environmental factors.

        Args:
            error_type: Specific error type to analyze, or None for all
            min_occurrences: Minimum number of occurrences required for analysis

        Returns:
            Dictionary with correlation analysis
        """
        results = {
            "correlations": [],
            "error_counts": {},
            "factor_stats": {},
            "error_patterns": [],
        }

        # Count errors by type
        for etype, occurrences in self.error_occurrences.items():
            results["error_counts"][etype] = len(occurrences)

        # Filter by error type if specified
        if error_type:
            if error_type not in self.error_occurrences:
                return {
                    "error": f"No occurrences of error type: {error_type}",
                    "error_counts": results["error_counts"],
                }

            error_types = [error_type]
        else:
            error_types = list(self.error_occurrences.keys())

        # Calculate basic stats for each factor
        for factor_name, values in self.factor_values.items():
            try:
                # Try to calculate numeric stats
                numeric_values = [
                    float(v) for v in values if str(v).replace(".", "", 1).isdigit()
                ]

                if numeric_values:
                    min_val = min(numeric_values)
                    max_val = max(numeric_values)
                    avg_val = sum(numeric_values) / len(numeric_values)

                    results["factor_stats"][factor_name] = {
                        "min": min_val,
                        "max": max_val,
                        "avg": avg_val,
                        "count": len(values),
                        "numeric_count": len(numeric_values),
                    }
                else:
                    # For non-numeric values, count occurrences
                    counter = Counter(values)
                    top_values = counter.most_common(5)

                    results["factor_stats"][factor_name] = {
                        "count": len(values),
                        "unique_values": len(counter),
                        "top_values": top_values,
                    }
            except Exception:
                # For non-numeric or error cases
                try:
                    counter = Counter(values)
                    results["factor_stats"][factor_name] = {
                        "count": len(values),
                        "unique_values": len(counter),
                    }
                except Exception:
                    results["factor_stats"][factor_name] = {
                        "count": len(values),
                        "error": "Could not analyze factor",
                    }

        # Analyze correlations for each error type
        for etype in error_types:
            occurrences = self.error_occurrences[etype]

            if len(occurrences) < min_occurrences:
                continue

            # Collect factor values for this error type
            error_factor_values = defaultdict(list)

            for occurrence in occurrences:
                error_id = occurrence["error_id"]
                if error_id in self.error_factors:
                    for name, factor in self.error_factors[error_id].items():
                        error_factor_values[name].append(factor.value)

            # Compare distribution of factors during errors vs. overall
            for factor_name, error_values in error_factor_values.items():
                all_values = self.factor_values[factor_name]

                if len(all_values) <= len(error_values) or len(set(all_values)) <= 1:
                    # Not enough contrast or variance in the data
                    continue

                # For categorical factors, compare frequencies
                if not all(str(v).replace(".", "", 1).isdigit() for v in all_values):
                    # Categorical analysis
                    all_counter = Counter(all_values)
                    error_counter = Counter(error_values)

                    # Convert to frequencies
                    all_freq = {k: v / len(all_values) for k, v in all_counter.items()}
                    error_freq = {
                        k: v / len(error_values) for k, v in error_counter.items()
                    }

                    # Calculate correlation strength for each value
                    strong_correlations = []

                    for value in set(all_values):
                        if value in error_freq:
                            all_frequency = all_freq.get(value, 0)
                            error_frequency = error_freq.get(value, 0)

                            # Simple frequency ratio as correlation measure
                            if all_frequency > 0:
                                correlation = error_frequency / all_frequency
                            else:
                                correlation = 0

                            # Apply significance filter
                            if correlation > 1.5 and error_frequency > 0.2:
                                strong_correlations.append(
                                    {
                                        "value": value,
                                        "correlation": correlation,
                                        "error_frequency": error_frequency,
                                        "overall_frequency": all_frequency,
                                    }
                                )

                    if strong_correlations:
                        results["correlations"].append(
                            {
                                "error_type": etype,
                                "factor": factor_name,
                                "type": "categorical",
                                "occurrences": len(error_values),
                                "strong_value_correlations": sorted(
                                    strong_correlations,
                                    key=lambda x: x["correlation"],
                                    reverse=True,
                                ),
                            }
                        )
                else:
                    # Numeric analysis - compare distributions
                    try:
                        # Convert to numeric
                        numeric_all = [float(v) for v in all_values]
                        numeric_error = [float(v) for v in error_values]

                        # Calculate basic statistics
                        all_avg = sum(numeric_all) / len(numeric_all)
                        error_avg = sum(numeric_error) / len(numeric_error)

                        # Simple z-score-like measure
                        if len(set(numeric_all)) > 1:
                            all_std = (sum((x - all_avg) ** 2 for x in numeric_all) /
                                     len(numeric_all)) ** 0.5

                            if all_std > 0:
                                z_score = (error_avg - all_avg) / all_std

                                # Apply significance filter
                                if abs(z_score) > 1.0:
                                    results["correlations"].append(
                                        {
                                            "error_type": etype,
                                            "factor": factor_name,
                                            "type": "numeric",
                                            "occurrences": len(error_values),
                                            "error_avg": error_avg,
                                            "overall_avg": all_avg,
                                            "overall_std": all_std,
                                            "z_score": z_score,
                                            "correlation_strength": abs(z_score),
                                        }
                                    )
                    except Exception as e:
                        logger.warning(
                            f"Error in numeric correlation analysis for {factor_name}: {str(e)}"
                        )

        # Sort correlations by strength
        results["correlations"] = sorted(
            results["correlations"],
            key=lambda x: (
                x.get("correlation_strength", 0)
                if "correlation_strength" in x
                else (
                    x.get("strong_value_correlations", [{}])[0].get("correlation", 0)
                    if x.get("strong_value_correlations")
                    else 0
                )
            ),
            reverse=True,
        )

        # Find patterns of multiple factors
        results["error_patterns"] = self._find_factor_patterns(
            error_types, min_occurrences
        )

        return results

    def _find_factor_patterns(
        self, error_types: List[str], min_occurrences: int
    ) -> List[Dict[str, Any]]:
        """
        Find patterns of factors that commonly occur together during errors.

        Args:
            error_types: List of error types to analyze
            min_occurrences: Minimum number of occurrences required

        Returns:
            List of factor patterns
        """
        patterns = []

        for etype in error_types:
            occurrences = self.error_occurrences[etype]

            if len(occurrences) < min_occurrences:
                continue

            # Group factor values by occurrence
            factor_sets = []

            for occurrence in occurrences:
                error_id = occurrence["error_id"]
                if error_id in self.error_factors:
                    # Get important factors for this error
                    important_factors = {
                        name: factor.value
                        for name, factor in self.error_factors[error_id].items()
                        if factor.importance >= 0.7  # Only include important factors
                    }

                    if important_factors:
                        factor_sets.append(important_factors)

            if len(factor_sets) < min_occurrences:
                continue

            # Find common combinations (simplified association rule mining)
            factor_combinations = defaultdict(int)

            for factor_set in factor_sets:
                # Generate combinations of factors
                items = list(factor_set.items())

                # Count pairs of factors
                for i in range(len(items)):
                    for j in range(i + 1, len(items)):
                        key = ((items[i][0], items[i][1]), (items[j][0], items[j][1]))
                        factor_combinations[key] += 1

            # Filter significant combinations
            significant_combinations = []

            for combo, count in factor_combinations.items():
                if (
                    count >= min_occurrences / 2
                ):  # Require combination to appear in at least half of occurrences
                    confidence = count / len(factor_sets)

                    significant_combinations.append(
                        {
                            "factors": [
                                {"name": name, "value": value} for name, value in combo
                            ],
                            "count": count,
                            "confidence": confidence,
                        }
                    )

            if significant_combinations:
                patterns.append(
                    {
                        "error_type": etype,
                        "occurrences": len(occurrences),
                        "factor_combinations": sorted(
                            significant_combinations,
                            key=lambda x: x["confidence"],
                            reverse=True,
                        ),
                    }
                )

        return sorted(patterns, key=lambda x: x.get("occurrences", 0), reverse=True)

    def suggest_monitoring(self) -> Dict[str, Any]:
        """
        Suggest factors to monitor based on correlation analysis.

        Returns:
            Dictionary with monitoring suggestions
        """
        # Run correlation analysis if not enough data
        if (not self.error_occurrences or
                sum(len(occurrences) for occurrences in self.error_occurrences.values()) <
                5):
            return {
                "error": "Not enough error data for monitoring suggestions",
                "min_required": 5,
                "current_count": sum(
                    len(occurrences) for occurrences in self.error_occurrences.values()
                ),
            }

        correlations = self.analyze_correlations(min_occurrences=3)

        # Extract important factors
        important_factors = set()

        for correlation in correlations.get("correlations", []):
            important_factors.add(correlation["factor"])

        # Include factors from patterns
        for pattern in correlations.get("error_patterns", []):
            for combo in pattern.get("factor_combinations", []):
                for factor in combo.get("factors", []):
                    if "name" in factor:
                        important_factors.add(factor["name"])

        # Group by source
        factors_by_source = defaultdict(list)

        for factor_name in important_factors:
            source = factor_name.split(".")[0] if "." in factor_name else "other"
            factors_by_source[source].append(factor_name)

        # Create monitoring suggestions
        suggestions = []

        for source, factors in factors_by_source.items():
            suggestions.append(
                {
                    "source": source,
                    "factors": sorted(factors),
                    "importance": "high" if len(factors) > 1 else "medium",
                    "suggested_interval": (
                        "1m" if source in ["system", "network"] else "10m"
                    ),
                }
            )

        return {
            "suggestions": sorted(
                suggestions, key=lambda x: len(x["factors"]), reverse=True
            ),
            "important_factors": sorted(list(important_factors)),
            "correlation_count": len(correlations.get("correlations", [])),
            "pattern_count": len(correlations.get("error_patterns", [])),
        }


def collect_environment(
    custom_factors: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Utility function to collect the current environment.

    Args:
        custom_factors: Custom factors to include

    Returns:
        Dictionary with environmental factors
    """
    collector = EnvironmentCollector()
    factors = collector.collect_all_factors(custom_factors)

    # Convert to a serializable format
    return {
        name: {
            "name": factor.name,
            "value": factor.value,
            "source": factor.source,
            "importance": factor.importance,
            "description": factor.description,
        }
        for name, factor in factors.items()
    }


def analyze_error_environment(
    error_sequence: List[Dict[str, Any]],
    custom_factors: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Utility function to analyze error correlations with environmental factors.

    Args:
        error_sequence: List of error data dictionaries
        custom_factors: Custom factors to include

    Returns:
        Analysis results with correlations
    """
    correlator = EnvironmentCorrelator()

    # Record each error with its environment
    for error_data in error_sequence:
        env_factors = correlator.capture_environment(custom_factors)
        correlator.record_error(error_data, env_factors)

    # Analyze correlations
    return correlator.analyze_correlations()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Example usage
    print("Environment Correlation Demo")
    print("===========================")

    # Collect current environment
    collector = EnvironmentCollector()
    environment = collector.collect_all_factors(
        {
            "custom": {
                "api_version": "1.2.3",
                "feature_flags": {"new_ui": True, "beta_features": False},
            }
        }
    )

    print(f"\nCollected {len(environment)} environmental factors")

    for source in [
        "system",
        "application",
        "network",
        "database",
        "deployment",
        "custom",
    ]:
        source_factors = [f for f in environment.values() if f.source == source]
        if source_factors:
            print(f"\n{source.capitalize()} Factors ({len(source_factors)}):")
            for factor in source_factors:
                print(
                    f"- {factor.name}: {factor.value} (importance: {factor.importance:.1f})"
                )

    # Simulate some errors with environmental correlations
    print("\nSimulating errors with environmental correlations...")

    correlator = EnvironmentCorrelator()

    # Create custom factors that will correlate with errors
    factors1 = correlator.capture_environment(
        {
            "custom": {
                "cpu_load": 0.2,
                "memory_pressure": 0.3,
                "connection_count": 10,
                "cache_hit_ratio": 0.9,
            }
        }
    )

    factors2 = correlator.capture_environment(
        {
            "custom": {
                "cpu_load": 0.85,  # High CPU correlates with timeout errors
                "memory_pressure": 0.3,
                "connection_count": 12,
                "cache_hit_ratio": 0.9,
            }
        }
    )

    factors3 = correlator.capture_environment(
        {
            "custom": {
                "cpu_load": 0.2,
                "memory_pressure": 0.9,  # High memory correlates with out of memory errors
                "connection_count": 10,
                "cache_hit_ratio": 0.4,
            }
        }
    )

    factors4 = correlator.capture_environment(
        {
            "custom": {
                "cpu_load": 0.3,
                "memory_pressure": 0.4,
                "connection_count": 50,  # High connection count correlates with connection errors
                "cache_hit_ratio": 0.8,
            }
        }
    )

    # Record errors with their environments
    for i in range(10):
        # Timeout errors have high CPU
        timeout_error = {
            "timestamp": f"2023-01-01T12:0{i}:00",
            "service": "api_service",
            "level": "ERROR",
            "message": "TimeoutError: Request timed out",
            "exception_type": "TimeoutError",
        }
        correlator.record_error(timeout_error, factors2)

        # Memory errors have high memory pressure
        memory_error = {
            "timestamp": f"2023-01-01T13:0{i}:00",
            "service": "worker_service",
            "level": "ERROR",
            "message": "MemoryError: Out of memory",
            "exception_type": "MemoryError",
        }
        correlator.record_error(memory_error, factors3)

        # Connection errors have high connection count
        connection_error = {
            "timestamp": f"2023-01-01T14:0{i}:00",
            "service": "database_service",
            "level": "ERROR",
            "message": "ConnectionError: Too many connections",
            "exception_type": "ConnectionError",
        }
        correlator.record_error(connection_error, factors4)

        # Some random errors with normal environment
        if i % 3 == 0:
            random_error = {
                "timestamp": f"2023-01-01T15:0{i}:00",
                "service": "random_service",
                "level": "ERROR",
                "message": "ValueError: Invalid value",
                "exception_type": "ValueError",
            }
            correlator.record_error(random_error, factors1)

    # Analyze correlations
    analysis = correlator.analyze_correlations()

    print(
        f"\nFound {len(analysis['correlations'])} correlations across {len(analysis['error_counts'])} error types"
    )

    # Print top correlations
    if analysis["correlations"]:
        print("\nTop Correlations:")
        for i, correlation in enumerate(analysis["correlations"][:5]):
            print(
                f"\n{i + 1}. {correlation['error_type']} correlates with {correlation['factor']}"
            )

            if correlation["type"] == "numeric":
                print(f"   Error average: {correlation['error_avg']:.2f}")
                print(f"   Overall average: {correlation['overall_avg']:.2f}")
                print(f"   Z-score: {correlation['z_score']:.2f}")
                print(f"   Strength: {correlation['correlation_strength']:.2f}")
            else:
                print("   Strong value correlations:")
                for value_corr in correlation["strong_value_correlations"][:3]:
                    print(f"   - Value: {value_corr['value']}")
                    print(f"     Correlation: {value_corr['correlation']:.2f}")
                    print(f"     Error frequency: {value_corr['error_frequency']:.2f}")
                    print(
                        f"     Overall frequency: {value_corr['overall_frequency']:.2f}"
                    )

    # Print error patterns
    if analysis["error_patterns"]:
        print("\nError Patterns:")
        for i, pattern in enumerate(analysis["error_patterns"][:3]):
            print(
                f"\n{i + 1}. {pattern['error_type']} ({pattern['occurrences']} occurrences)"
            )

            for j, combo in enumerate(pattern["factor_combinations"][:3]):
                print(f"   Pattern {j + 1} (confidence: {combo['confidence']:.2f}):")
                for factor in combo["factors"]:
                    print(f"   - {factor['name']}: {factor['value']}")

    # Get monitoring suggestions
    suggestions = correlator.suggest_monitoring()

    print("\nMonitoring Suggestions:")
    for suggestion in suggestions["suggestions"]:
        print(
            f"\n{suggestion['source'].capitalize()} (importance: {suggestion['importance']}):"
        )
        print(f"  Suggested interval: {suggestion['suggested_interval']}")
        print("  Factors to monitor:")
        for factor in suggestion["factors"]:
            print(f"  - {factor}")
