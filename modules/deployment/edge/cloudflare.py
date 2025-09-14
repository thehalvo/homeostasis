"""
Cloudflare provider for edge deployment.

Provides functionality for deploying and managing applications on Cloudflare Workers
and other Cloudflare edge services.
"""

import json
import logging
import os
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Optional

from modules.security.audit import get_audit_logger

logger = logging.getLogger(__name__)


class CloudflareProvider:
    """
    Cloudflare provider for edge deployment.

    Manages the deployment, update, and monitoring of applications on
    Cloudflare Workers and other Cloudflare edge services.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Cloudflare provider.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

        # Set default values from config
        self.account_id = self.config.get("account_id")
        self.api_token = self.config.get("api_token")
        self.zone_id = self.config.get("zone_id")

        # Check if Cloudflare tools are available
        self.wrangler_available = self._check_wrangler_available()
        if not self.wrangler_available:
            logger.warning(
                "Wrangler CLI not found, Cloudflare operations will be simulated"
            )

    def _check_wrangler_available(self) -> bool:
        """Check if Wrangler CLI is available.

        Returns:
            bool: True if Wrangler CLI is available, False otherwise
        """
        try:
            result = subprocess.run(
                ["which", "wrangler"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            return result.returncode == 0
        except Exception:
            return False

    def _run_wrangler(
        self,
        args: List[str],
        input_data: Optional[str] = None,
        cwd: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run Wrangler CLI command.

        Args:
            args: List of arguments for Wrangler
            input_data: Optional input data
            cwd: Optional working directory

        Returns:
            Dict: Result of Wrangler CLI command
        """
        if not self.wrangler_available:
            logger.info(f"Simulating Wrangler CLI command: wrangler {' '.join(args)}")
            return {"success": True, "simulated": True}

        try:
            # Set environment variables for authentication
            env = os.environ.copy()
            if self.api_token:
                env["CLOUDFLARE_API_TOKEN"] = self.api_token

            # Account ID is required for many commands
            if self.account_id and "--account-id" not in args and "-a" not in args:
                args.extend(["--account-id", self.account_id])

            cmd = ["wrangler"] + args
            logger.debug(f"Running Wrangler CLI command: {' '.join(cmd)}")

            process = subprocess.run(
                cmd,
                input=input_data.encode() if input_data else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                cwd=cwd,
                check=False,
            )

            stdout = process.stdout.decode() if process.stdout else ""
            stderr = process.stderr.decode() if process.stderr else ""

            if process.returncode != 0:
                logger.error(f"Wrangler CLI command failed: {stderr}")
                return {
                    "success": False,
                    "returncode": process.returncode,
                    "stdout": stdout,
                    "stderr": stderr,
                }

            # Try to parse JSON output if possible
            result: Dict[str, Any]
            try:
                if stdout and stdout.strip().startswith("{"):
                    result = json.loads(stdout)
                else:
                    result = {"output": stdout}
            except json.JSONDecodeError:
                result = {"output": stdout}

            result["success"] = True
            result["returncode"] = process.returncode

            return result

        except Exception as e:
            logger.exception(f"Error running Wrangler CLI command: {str(e)}")
            return {"success": False, "error": str(e)}

    def _setup_worker_project(
        self, service_name: str, fix_id: str, source_path: str
    ) -> Optional[str]:
        """Setup a Cloudflare Worker project.

        Args:
            service_name: Name of the service
            fix_id: ID of the fix
            source_path: Path to the source code

        Returns:
            Optional[str]: Path to the worker project, or None if failed
        """
        try:
            # Create a temporary directory for the project
            project_dir = tempfile.mkdtemp()
            worker_name = f"{service_name}-{fix_id}"

            # Copy source files to the project directory
            if os.path.isdir(source_path):
                # If source_path is a directory, copy all files
                for item in os.listdir(source_path):
                    source_item = os.path.join(source_path, item)
                    dest_item = os.path.join(project_dir, item)
                    if os.path.isdir(source_item):
                        shutil.copytree(source_item, dest_item)
                    else:
                        shutil.copy2(source_item, dest_item)
            else:
                # If source_path is a file, copy just that file
                shutil.copy2(
                    source_path,
                    os.path.join(project_dir, os.path.basename(source_path)),
                )

            # Create wrangler.toml configuration file
            wrangler_toml = f"""
name = "{worker_name}"
type = "javascript"
account_id = "{self.account_id}"
workers_dev = true
route = ""
zone_id = "{self.zone_id or ''}"
compatibility_date = "2023-06-01"

[build]
command = "npm install && npm run build"
[build.upload]
format = "service-worker"
"""

            with open(os.path.join(project_dir, "wrangler.toml"), "w") as f:
                f.write(wrangler_toml)

            logger.info(f"Created Cloudflare Worker project at {project_dir}")
            return project_dir

        except Exception as e:
            logger.error(f"Error setting up Cloudflare Worker project: {str(e)}")
            return None

    def deploy(
        self, service_name: str, fix_id: str, source_path: str, **kwargs
    ) -> Dict[str, Any]:
        """Deploy a service to Cloudflare Workers.

        Args:
            service_name: Name of the service
            fix_id: ID of the fix
            source_path: Path to the source code
            **kwargs: Additional parameters

        Returns:
            Dict[str, Any]: Deployment information
        """
        worker_name = f"{service_name}-{fix_id}"

        try:
            # Setup the worker project
            project_dir = self._setup_worker_project(service_name, fix_id, source_path)
            if not project_dir:
                return {
                    "success": False,
                    "error": "Failed to setup Cloudflare Worker project",
                }

            # Deploy the worker
            deploy_args = ["deploy"]

            # Add environment variables if provided
            env_vars = kwargs.get("env_vars", {})
            for key, value in env_vars.items():
                deploy_args.extend(["--var", f"{key}:{value}"])

            # Deploy using Wrangler
            result = self._run_wrangler(deploy_args, cwd=project_dir)

            # Clean up the project directory
            try:
                shutil.rmtree(project_dir)
            except (OSError, IOError):
                pass

            # Log the deployment
            try:
                get_audit_logger().log_event(
                    event_type="cloudflare_worker_deployed",
                    details={
                        "service_name": service_name,
                        "worker_name": worker_name,
                        "fix_id": fix_id,
                        "success": result["success"],
                    },
                )
            except Exception as e:
                logger.debug(f"Could not log to audit log: {str(e)}")

            return {
                "success": result["success"],
                "worker_name": worker_name,
                "provider": "cloudflare",
                "deployment_result": result,
            }

        except Exception as e:
            logger.exception(f"Error deploying Cloudflare Worker: {str(e)}")
            return {"success": False, "error": str(e)}

    def update(
        self,
        service_name: str,
        fix_id: str,
        source_path: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Update a service on Cloudflare Workers.

        Args:
            service_name: Name of the service
            fix_id: ID of the fix
            source_path: Optional path to the updated source code
            **kwargs: Additional parameters

        Returns:
            Dict[str, Any]: Update information
        """
        # For Cloudflare Workers, an update is the same as a deployment
        if source_path:
            return self.deploy(service_name, fix_id, source_path, **kwargs)

        worker_name = f"{service_name}-{fix_id}"

        # If no source path is provided, just update environment variables
        try:
            # Update environment variables if provided
            env_vars = kwargs.get("env_vars", {})
            if env_vars:
                update_args = ["secret", "bulk", "--name", worker_name]

                # Create a JSON file with the variables
                with tempfile.NamedTemporaryFile(
                    mode="w+", suffix=".json", delete=False
                ) as f:
                    json.dump(env_vars, f)
                    env_file = f.name

                update_args.extend(["--json", env_file])

                # Update using Wrangler
                result = self._run_wrangler(update_args)

                # Clean up the temporary file
                try:
                    os.remove(env_file)
                except (OSError, IOError):
                    pass

                # Log the update
                try:
                    get_audit_logger().log_event(
                        event_type="cloudflare_worker_updated",
                        details={
                            "service_name": service_name,
                            "worker_name": worker_name,
                            "fix_id": fix_id,
                            "success": result["success"],
                        },
                    )
                except Exception as e:
                    logger.debug(f"Could not log to audit log: {str(e)}")

                return {
                    "success": result["success"],
                    "worker_name": worker_name,
                    "provider": "cloudflare",
                    "update_result": result,
                }
            else:
                return {
                    "success": True,
                    "worker_name": worker_name,
                    "message": "No updates required",
                }

        except Exception as e:
            logger.exception(f"Error updating Cloudflare Worker: {str(e)}")
            return {"success": False, "error": str(e)}

    def delete(self, service_name: str, fix_id: str) -> Dict[str, Any]:
        """Delete a service from Cloudflare Workers.

        Args:
            service_name: Name of the service
            fix_id: ID of the fix

        Returns:
            Dict[str, Any]: Deletion information
        """
        worker_name = f"{service_name}-{fix_id}"

        try:
            # Delete the worker
            delete_args = ["delete", worker_name]
            result = self._run_wrangler(delete_args)

            # Log the deletion
            try:
                get_audit_logger().log_event(
                    event_type="cloudflare_worker_deleted",
                    details={
                        "service_name": service_name,
                        "worker_name": worker_name,
                        "fix_id": fix_id,
                        "success": result["success"],
                    },
                )
            except Exception as e:
                logger.debug(f"Could not log to audit log: {str(e)}")

            return {
                "success": result["success"],
                "worker_name": worker_name,
                "provider": "cloudflare",
                "deletion_result": result,
            }

        except Exception as e:
            logger.exception(f"Error deleting Cloudflare Worker: {str(e)}")
            return {"success": False, "error": str(e)}

    def get_status(self, service_name: str, fix_id: str) -> Dict[str, Any]:
        """Get the status of a Cloudflare Worker.

        Args:
            service_name: Name of the service
            fix_id: ID of the fix

        Returns:
            Dict[str, Any]: Status information
        """
        worker_name = f"{service_name}-{fix_id}"

        try:
            # Get the worker status
            status_args = ["status", worker_name]
            result = self._run_wrangler(status_args)

            if result.get("simulated", False):
                # Return simulated status
                return {
                    "success": True,
                    "simulated": True,
                    "worker_name": worker_name,
                    "provider": "cloudflare",
                    "status": "active",
                    "url": f"https://{worker_name}.{self.account_id}.workers.dev",
                }

            return {
                "success": result["success"],
                "worker_name": worker_name,
                "provider": "cloudflare",
                "status_result": result,
            }

        except Exception as e:
            logger.exception(f"Error getting Cloudflare Worker status: {str(e)}")
            return {"success": False, "error": str(e)}

    def purge_cache(
        self, service_name: str, fix_id: str, paths: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Purge the cache for a Cloudflare Worker.

        Args:
            service_name: Name of the service
            fix_id: ID of the fix
            paths: Optional list of paths to purge (defaults to all)

        Returns:
            Dict[str, Any]: Purge information
        """
        worker_name = f"{service_name}-{fix_id}"

        try:
            # Skip if no zone ID
            if not self.zone_id:
                return {
                    "success": False,
                    "error": "No zone ID provided for cache purge",
                }

            # Purge cache
            purge_args = ["cache", "purge"]

            # Add paths if provided
            if paths:
                for path in paths:
                    purge_args.append(path)
            else:
                purge_args.append("--everything")

            # Add zone ID
            purge_args.extend(["--zone", self.zone_id])

            # Purge using Wrangler
            result = self._run_wrangler(purge_args)

            # Log the purge
            try:
                get_audit_logger().log_event(
                    event_type="cloudflare_cache_purged",
                    details={
                        "service_name": service_name,
                        "worker_name": worker_name,
                        "fix_id": fix_id,
                        "paths": paths,
                        "success": result["success"],
                    },
                )
            except Exception as e:
                logger.debug(f"Could not log to audit log: {str(e)}")

            return {
                "success": result["success"],
                "worker_name": worker_name,
                "provider": "cloudflare",
                "purge_result": result,
            }

        except Exception as e:
            logger.exception(f"Error purging Cloudflare cache: {str(e)}")
            return {"success": False, "error": str(e)}

    def setup_canary(
        self, service_name: str, fix_id: str, percentage: int = 10
    ) -> Dict[str, Any]:
        """Setup canary deployment for a Cloudflare Worker.

        Args:
            service_name: Name of the service
            fix_id: ID of the fix
            percentage: Percentage of traffic to route to canary

        Returns:
            Dict[str, Any]: Canary setup information
        """
        worker_name = f"{service_name}-{fix_id}"

        try:
            # For Cloudflare, we use the Traffic Management feature (A/B Testing)
            # This requires the Enterprise plan, so we'll simulate it

            # Log the canary setup
            try:
                get_audit_logger().log_event(
                    event_type="cloudflare_canary_setup",
                    details={
                        "service_name": service_name,
                        "worker_name": worker_name,
                        "fix_id": fix_id,
                        "percentage": percentage,
                        "success": True,
                    },
                )
            except Exception as e:
                logger.debug(f"Could not log to audit log: {str(e)}")

            return {
                "success": True,
                "simulated": True,
                "worker_name": worker_name,
                "provider": "cloudflare",
                "percentage": percentage,
                "message": "Canary deployment simulated (requires Cloudflare Enterprise)",
            }

        except Exception as e:
            logger.exception(f"Error setting up Cloudflare canary: {str(e)}")
            return {"success": False, "error": str(e)}

    def update_canary(
        self, service_name: str, fix_id: str, percentage: int
    ) -> Dict[str, Any]:
        """Update canary deployment for a Cloudflare Worker.

        Args:
            service_name: Name of the service
            fix_id: ID of the fix
            percentage: New percentage of traffic to route to canary

        Returns:
            Dict[str, Any]: Canary update information
        """
        worker_name = f"{service_name}-{fix_id}"

        try:
            # For Cloudflare, we use the Traffic Management feature (A/B Testing)
            # This requires the Enterprise plan, so we'll simulate it

            # Log the canary update
            try:
                get_audit_logger().log_event(
                    event_type="cloudflare_canary_updated",
                    details={
                        "service_name": service_name,
                        "worker_name": worker_name,
                        "fix_id": fix_id,
                        "percentage": percentage,
                        "success": True,
                    },
                )
            except Exception as e:
                logger.debug(f"Could not log to audit log: {str(e)}")

            return {
                "success": True,
                "simulated": True,
                "worker_name": worker_name,
                "provider": "cloudflare",
                "percentage": percentage,
                "message": "Canary update simulated (requires Cloudflare Enterprise)",
            }

        except Exception as e:
            logger.exception(f"Error updating Cloudflare canary: {str(e)}")
            return {"success": False, "error": str(e)}

    def complete_canary(self, service_name: str, fix_id: str) -> Dict[str, Any]:
        """Complete canary deployment for a Cloudflare Worker.

        Args:
            service_name: Name of the service
            fix_id: ID of the fix

        Returns:
            Dict[str, Any]: Canary completion information
        """
        worker_name = f"{service_name}-{fix_id}"

        try:
            # For Cloudflare, we use the Traffic Management feature (A/B Testing)
            # This requires the Enterprise plan, so we'll simulate it

            # Log the canary completion
            try:
                get_audit_logger().log_event(
                    event_type="cloudflare_canary_completed",
                    details={
                        "service_name": service_name,
                        "worker_name": worker_name,
                        "fix_id": fix_id,
                        "success": True,
                    },
                )
            except Exception as e:
                logger.debug(f"Could not log to audit log: {str(e)}")

            return {
                "success": True,
                "simulated": True,
                "worker_name": worker_name,
                "provider": "cloudflare",
                "percentage": 100,
                "message": "Canary completion simulated (requires Cloudflare Enterprise)",
            }

        except Exception as e:
            logger.exception(f"Error completing Cloudflare canary: {str(e)}")
            return {"success": False, "error": str(e)}

    def rollback_canary(self, service_name: str, fix_id: str) -> Dict[str, Any]:
        """Rollback canary deployment for a Cloudflare Worker.

        Args:
            service_name: Name of the service
            fix_id: ID of the fix

        Returns:
            Dict[str, Any]: Canary rollback information
        """
        worker_name = f"{service_name}-{fix_id}"

        try:
            # For Cloudflare, we use the Traffic Management feature (A/B Testing)
            # This requires the Enterprise plan, so we'll simulate it

            # Log the canary rollback
            try:
                get_audit_logger().log_event(
                    event_type="cloudflare_canary_rolled_back",
                    details={
                        "service_name": service_name,
                        "worker_name": worker_name,
                        "fix_id": fix_id,
                        "success": True,
                    },
                )
            except Exception as e:
                logger.debug(f"Could not log to audit log: {str(e)}")

            return {
                "success": True,
                "simulated": True,
                "worker_name": worker_name,
                "provider": "cloudflare",
                "percentage": 0,
                "message": "Canary rollback simulated (requires Cloudflare Enterprise)",
            }

        except Exception as e:
            logger.exception(f"Error rolling back Cloudflare canary: {str(e)}")
            return {"success": False, "error": str(e)}


# Singleton instance
_cloudflare_provider = None


def get_cloudflare_provider(config: Optional[Dict[str, Any]] = None) -> CloudflareProvider:
    """Get or create the singleton CloudflareProvider instance.

    Args:
        config: Optional configuration

    Returns:
        CloudflareProvider: Singleton instance
    """
    global _cloudflare_provider
    if _cloudflare_provider is None:
        _cloudflare_provider = CloudflareProvider(config)
    return _cloudflare_provider
