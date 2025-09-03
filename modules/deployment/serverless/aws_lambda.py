"""
AWS Lambda provider for Homeostasis.

Provides functionality for deploying and managing serverless functions on AWS Lambda.
"""

import json
import logging
import os
import subprocess
import tempfile
import zipfile
from datetime import datetime
from typing import Dict, List, Optional, Any

from modules.deployment.serverless.base_provider import ServerlessProvider
from modules.security.audit import get_audit_logger

logger = logging.getLogger(__name__)


class AWSLambdaProvider(ServerlessProvider):
    """
    AWS Lambda provider for serverless function deployment.
    
    Manages the deployment, update, and monitoring of functions on AWS Lambda.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize AWS Lambda provider.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # Set default values from config
        self.region = self.config.get("region", "us-east-1")
        self.role_arn = self.config.get("role_arn")
        self.timeout = self.config.get("timeout", 30)
        self.memory_size = self.config.get("memory_size", 128)
        
        # Check if AWS CLI is available
        self.aws_cli_available = self._check_aws_cli_available()
        if not self.aws_cli_available:
            logger.warning("AWS CLI not found, AWS Lambda operations will be simulated")
            
    def _check_aws_cli_available(self) -> bool:
        """Check if AWS CLI is available.
        
        Returns:
            bool: True if AWS CLI is available, False otherwise
        """
        try:
            result = subprocess.run(
                ["which", "aws"], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                check=False
            )
            return result.returncode == 0
        except Exception:
            return False
            
    def _run_aws_cli(self, service: str, operation: str, args: List[str] = None, 
                   input_data: Optional[str] = None) -> Dict[str, Any]:
        """Run AWS CLI command.
        
        Args:
            service: AWS service (e.g., "lambda")
            operation: AWS operation (e.g., "create-function")
            args: Additional arguments
            input_data: Optional input data
            
        Returns:
            Dict: Result of AWS CLI command
        """
        if not self.aws_cli_available:
            logger.info(f"Simulating AWS CLI command: aws {service} {operation}")
            return {"success": True, "simulated": True}
            
        try:
            cmd = ["aws", service, operation, "--region", self.region]
            if args:
                cmd.extend(args)
                
            logger.debug(f"Running AWS CLI command: {' '.join(cmd)}")
            
            process = subprocess.run(
                cmd,
                input=input_data.encode() if input_data else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False
            )
            
            stdout = process.stdout.decode() if process.stdout else ""
            stderr = process.stderr.decode() if process.stderr else ""
            
            if process.returncode != 0:
                logger.error(f"AWS CLI command failed: {stderr}")
                return {
                    "success": False,
                    "returncode": process.returncode,
                    "stdout": stdout,
                    "stderr": stderr
                }
                
            # Try to parse JSON output
            try:
                if stdout and stdout.strip():
                    result = json.loads(stdout)
                else:
                    result = {}
            except json.JSONDecodeError:
                result = {"output": stdout}
                
            result["success"] = True
            result["returncode"] = process.returncode
            
            return result
            
        except Exception as e:
            logger.exception(f"Error running AWS CLI command: {str(e)}")
            return {"success": False, "error": str(e)}
            
    def _create_deployment_package(self, source_path: str) -> Optional[str]:
        """Create a deployment package (ZIP) for an AWS Lambda function.
        
        Args:
            source_path: Path to the source code
            
        Returns:
            Optional[str]: Path to the deployment package, or None if failed
        """
        try:
            # Create a temporary directory for the deployment package
            temp_dir = tempfile.mkdtemp()
            zip_path = os.path.join(temp_dir, "function.zip")
            
            # Create the ZIP file
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
                # If source_path is a directory, add all files
                if os.path.isdir(source_path):
                    for root, _, files in os.walk(source_path):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, source_path)
                            zip_file.write(file_path, arcname)
                else:
                    # If source_path is a file, add just that file
                    zip_file.write(source_path, os.path.basename(source_path))
                    
            logger.info(f"Created deployment package at {zip_path}")
            return zip_path
            
        except Exception as e:
            logger.error(f"Error creating deployment package: {str(e)}")
            return None
            
    def is_available(self) -> bool:
        """Check if AWS Lambda is available.
        
        Returns:
            bool: True if AWS Lambda is available, False otherwise
        """
        if not self.aws_cli_available:
            return False
            
        # Try listing functions to check if Lambda is available
        result = self._run_aws_cli("lambda", "list-functions")
        return result["success"]
        
    def deploy_function(self, function_name: str, fix_id: str, 
                      source_path: str, handler: str,
                      runtime: str = "python3.9", **kwargs) -> Dict[str, Any]:
        """Deploy a serverless function to AWS Lambda.
        
        Args:
            function_name: Name of the function
            fix_id: ID of the fix
            source_path: Path to the source code
            handler: Function handler (e.g., "index.handler")
            runtime: Runtime environment (e.g., "python3.9")
            **kwargs: Additional parameters
            
        Returns:
            Dict[str, Any]: Deployment information
        """
        # Generate function name with fix ID
        full_function_name = f"{function_name}-{fix_id}"
        
        # Create deployment package
        zip_path = self._create_deployment_package(source_path)
        if not zip_path:
            return {"success": False, "error": "Failed to create deployment package"}
            
        try:
            # Read the deployment package
            with open(zip_path, "rb") as f:
                zip_bytes = f.read()
                
            # Check if the function already exists
            exists_result = self._run_aws_cli(
                "lambda", "get-function",
                args=["--function-name", full_function_name]
            )
            
            if exists_result["success"] and not exists_result.get("simulated", False):
                # Function exists, update it
                logger.info(f"Function {full_function_name} already exists, updating it")
                return self.update_function(function_name, fix_id, source_path, handler=handler)
                
            # Create the function
            role_arn = kwargs.get("role_arn", self.role_arn)
            if not role_arn:
                return {"success": False, "error": "No IAM role ARN provided for Lambda function"}
                
            timeout = kwargs.get("timeout", self.timeout)
            memory_size = kwargs.get("memory_size", self.memory_size)
            env_vars = kwargs.get("environment", {})
            
            # Format environment variables for CLI
            env_vars_str = json.dumps({"Variables": env_vars}) if env_vars else None
            
            # Prepare tags
            tags = {
                "FixId": fix_id,
                "CreatedBy": "Homeostasis",
                "CreatedAt": datetime.now().isoformat()
            }
            tags.update(kwargs.get("tags", {}))
            tags_args = ["--tags"] + [f"{k}={v}" for k, v in tags.items()] if tags else []
            
            # Deploy the function
            create_args = [
                "--function-name", full_function_name,
                "--runtime", runtime,
                "--role", role_arn,
                "--handler", handler,
                "--timeout", str(timeout),
                "--memory-size", str(memory_size),
                "--zip-file", f"fileb://{zip_path}",
            ]
            
            # Add environment variables if provided
            if env_vars_str:
                create_args.extend(["--environment", env_vars_str])
                
            # Add tags
            create_args.extend(tags_args)
            
            # Execute the create command
            result = self._run_aws_cli("lambda", "create-function", args=create_args)
            
            # Clean up
            try:
                os.remove(zip_path)
                os.rmdir(os.path.dirname(zip_path))
            except:
                pass
                
            # Log the deployment
            try:
                get_audit_logger().log_event(
                    event_type="lambda_function_deployed",
                    details={
                        "function_name": function_name,
                        "lambda_function_name": full_function_name,
                        "fix_id": fix_id,
                        "runtime": runtime,
                        "handler": handler,
                        "success": result["success"]
                    }
                )
            except Exception as e:
                logger.debug(f"Could not log to audit log: {str(e)}")
                
            return result
            
        except Exception as e:
            logger.exception(f"Error deploying Lambda function: {str(e)}")
            return {"success": False, "error": str(e)}
            
    def update_function(self, function_name: str, fix_id: str, 
                      source_path: str, handler: str = None, **kwargs) -> Dict[str, Any]:
        """Update a serverless function on AWS Lambda.
        
        Args:
            function_name: Name of the function
            fix_id: ID of the fix
            source_path: Path to the source code
            handler: Optional new handler
            **kwargs: Additional parameters
            
        Returns:
            Dict[str, Any]: Update information
        """
        # Generate function name with fix ID
        full_function_name = f"{function_name}-{fix_id}"
        
        # Create deployment package
        zip_path = self._create_deployment_package(source_path)
        if not zip_path:
            return {"success": False, "error": "Failed to create deployment package"}
            
        try:
            # Read the deployment package
            with open(zip_path, "rb") as f:
                zip_bytes = f.read()
                
            # Update the function code
            code_update_args = [
                "--function-name", full_function_name,
                "--zip-file", f"fileb://{zip_path}"
            ]
            
            code_result = self._run_aws_cli("lambda", "update-function-code", args=code_update_args)
            
            if not code_result["success"] and not code_result.get("simulated", False):
                logger.error(f"Failed to update Lambda function code: {code_result}")
                return code_result
                
            # Update function configuration if handler is provided
            if handler:
                config_update_args = [
                    "--function-name", full_function_name,
                    "--handler", handler
                ]
                
                # Add optional configuration parameters
                if "timeout" in kwargs:
                    config_update_args.extend(["--timeout", str(kwargs["timeout"])])
                    
                if "memory_size" in kwargs:
                    config_update_args.extend(["--memory-size", str(kwargs["memory_size"])])
                    
                if "environment" in kwargs:
                    env_vars_str = json.dumps({"Variables": kwargs["environment"]})
                    config_update_args.extend(["--environment", env_vars_str])
                    
                config_result = self._run_aws_cli("lambda", "update-function-configuration", args=config_update_args)
                
                if not config_result["success"] and not config_result.get("simulated", False):
                    logger.error(f"Failed to update Lambda function configuration: {config_result}")
                    return config_result
                    
            # Clean up
            try:
                os.remove(zip_path)
                os.rmdir(os.path.dirname(zip_path))
            except:
                pass
                
            # Log the update
            try:
                get_audit_logger().log_event(
                    event_type="lambda_function_updated",
                    details={
                        "function_name": function_name,
                        "lambda_function_name": full_function_name,
                        "fix_id": fix_id,
                        "handler": handler,
                        "success": code_result["success"]
                    }
                )
            except Exception as e:
                logger.debug(f"Could not log to audit log: {str(e)}")
                
            return {
                "success": True,
                "code_update": code_result,
                "config_update": config_result if handler else None
            }
            
        except Exception as e:
            logger.exception(f"Error updating Lambda function: {str(e)}")
            return {"success": False, "error": str(e)}
            
    def delete_function(self, function_name: str, fix_id: str, **kwargs) -> Dict[str, Any]:
        """Delete a serverless function from AWS Lambda.
        
        Args:
            function_name: Name of the function
            fix_id: ID of the fix
            **kwargs: Additional parameters
            
        Returns:
            Dict[str, Any]: Deletion information
        """
        # Generate function name with fix ID
        full_function_name = f"{function_name}-{fix_id}"
        
        try:
            # Delete the function
            delete_args = ["--function-name", full_function_name]
            result = self._run_aws_cli("lambda", "delete-function", args=delete_args)
            
            # Log the deletion
            try:
                get_audit_logger().log_event(
                    event_type="lambda_function_deleted",
                    details={
                        "function_name": function_name,
                        "lambda_function_name": full_function_name,
                        "fix_id": fix_id,
                        "success": result["success"]
                    }
                )
            except Exception as e:
                logger.debug(f"Could not log to audit log: {str(e)}")
                
            return result
            
        except Exception as e:
            logger.exception(f"Error deleting Lambda function: {str(e)}")
            return {"success": False, "error": str(e)}
            
    def get_function_status(self, function_name: str, fix_id: str, **kwargs) -> Dict[str, Any]:
        """Get status of a deployed function on AWS Lambda.
        
        Args:
            function_name: Name of the function
            fix_id: ID of the fix
            **kwargs: Additional parameters
            
        Returns:
            Dict[str, Any]: Function status information
        """
        # Generate function name with fix ID
        full_function_name = f"{function_name}-{fix_id}"
        
        try:
            # Get the function
            get_args = ["--function-name", full_function_name]
            result = self._run_aws_cli("lambda", "get-function", args=get_args)
            
            if result.get("simulated", False):
                # Return simulated status
                return {
                    "success": True,
                    "simulated": True,
                    "function_name": full_function_name,
                    "status": "Active",
                    "runtime": "python3.9",
                    "handler": "index.handler",
                    "last_modified": datetime.now().isoformat()
                }
                
            if not result["success"]:
                return result
                
            # Extract relevant status information
            function_info = result.get("Configuration", {})
            status = {
                "success": True,
                "function_name": full_function_name,
                "status": function_info.get("State", "Unknown"),
                "runtime": function_info.get("Runtime"),
                "handler": function_info.get("Handler"),
                "memory_size": function_info.get("MemorySize"),
                "timeout": function_info.get("Timeout"),
                "last_modified": function_info.get("LastModified"),
                "version": function_info.get("Version")
            }
            
            return status
            
        except Exception as e:
            logger.exception(f"Error getting Lambda function status: {str(e)}")
            return {"success": False, "error": str(e)}
            
    def get_function_logs(self, function_name: str, fix_id: str, 
                        since: Optional[str] = None, **kwargs) -> List[Dict[str, Any]]:
        """Get logs for a deployed function on AWS Lambda.
        
        Args:
            function_name: Name of the function
            fix_id: ID of the fix
            since: Optional timestamp to get logs since (ISO format)
            **kwargs: Additional parameters
            
        Returns:
            List[Dict[str, Any]]: Function logs
        """
        # Generate function name with fix ID
        full_function_name = f"{function_name}-{fix_id}"
        
        try:
            # Construct log group name
            log_group = f"/aws/lambda/{full_function_name}"
            
            # Prepare filter parameters
            filter_args = ["--log-group-name", log_group]
            
            if since:
                # Convert ISO timestamp to milliseconds since epoch
                try:
                    since_dt = datetime.fromisoformat(since.replace('Z', '+00:00'))
                    since_ms = int(since_dt.timestamp() * 1000)
                    filter_args.extend(["--start-time", str(since_ms)])
                except Exception as e:
                    logger.warning(f"Invalid timestamp format for 'since': {since}. Using default.")
                    
            # Add any limit parameter
            limit = kwargs.get("limit", 100)
            filter_args.extend(["--limit", str(limit)])
            
            # Execute the filter command
            result = self._run_aws_cli("logs", "filter-log-events", args=filter_args)
            
            if result.get("simulated", False):
                # Return simulated logs
                return [{
                    "timestamp": datetime.now().isoformat(),
                    "message": "Simulated log message",
                    "ingestionTime": datetime.now().isoformat()
                }]
                
            if not result["success"]:
                return []
                
            # Extract log events
            logs = result.get("events", [])
            return logs
            
        except Exception as e:
            logger.exception(f"Error getting Lambda function logs: {str(e)}")
            return []
            
    def invoke_function(self, function_name: str, fix_id: str, 
                      payload: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Invoke a serverless function on AWS Lambda.
        
        Args:
            function_name: Name of the function
            fix_id: ID of the fix
            payload: Function payload
            **kwargs: Additional parameters
            
        Returns:
            Dict[str, Any]: Function response
        """
        # Generate function name with fix ID
        full_function_name = f"{function_name}-{fix_id}"
        
        try:
            # Prepare invoke parameters
            invoke_args = ["--function-name", full_function_name]
            
            # Add invocation type
            invocation_type = kwargs.get("invocation_type", "RequestResponse")
            invoke_args.extend(["--invocation-type", invocation_type])
            
            # Create temporary file for payload
            with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as f:
                json.dump(payload, f)
                payload_file = f.name
                
            invoke_args.extend(["--payload", f"fileb://{payload_file}"])
            
            # Create temporary file for response
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
                response_file = f.name
                
            invoke_args.extend(["--cli-binary-format", "raw-in-base64-out"])
            invoke_args.extend(["--output", "json"])
            
            # Execute the invoke command
            result = self._run_aws_cli("lambda", "invoke", args=invoke_args + ["--output", "json", response_file])
            
            if result.get("simulated", False):
                # Return simulated response
                return {
                    "success": True,
                    "simulated": True,
                    "StatusCode": 200,
                    "ExecutedVersion": "$LATEST",
                    "payload": {"result": "Simulated function response"}
                }
                
            # Read the response
            try:
                with open(response_file, "r") as f:
                    response_data = json.load(f)
            except:
                response_data = {}
                
            # Clean up temporary files
            try:
                os.remove(payload_file)
                os.remove(response_file)
            except:
                pass
                
            if not result["success"]:
                return result
                
            # Combine metadata with payload
            result.update({"payload": response_data})
            return result
            
        except Exception as e:
            logger.exception(f"Error invoking Lambda function: {str(e)}")
            return {"success": False, "error": str(e)}
            
    def setup_canary_deployment(self, function_name: str, fix_id: str,
                              traffic_percentage: int = 10, **kwargs) -> Dict[str, Any]:
        """Setup canary deployment for a function on AWS Lambda.
        
        Args:
            function_name: Name of the function
            fix_id: ID of the fix
            traffic_percentage: Percentage of traffic to route to new version
            **kwargs: Additional parameters
            
        Returns:
            Dict[str, Any]: Canary deployment information
        """
        # Generate function name with fix ID
        full_function_name = f"{function_name}-{fix_id}"
        
        try:
            # First, publish a version of the function
            publish_args = ["--function-name", full_function_name]
            publish_result = self._run_aws_cli("lambda", "publish-version", args=publish_args)
            
            if not publish_result["success"] and not publish_result.get("simulated", False):
                logger.error(f"Failed to publish Lambda function version: {publish_result}")
                return publish_result
                
            # Get the version number
            version = publish_result.get("Version", "1") if not publish_result.get("simulated", False) else "1"
            
            # Create an alias for the canary deployment
            alias_name = "canary"
            routing_config = json.dumps({
                "AdditionalVersionWeights": {
                    version: traffic_percentage / 100.0
                }
            })
            
            alias_args = [
                "--function-name", full_function_name,
                "--name", alias_name,
                "--function-version", "$LATEST",  # Primary traffic goes to LATEST
                "--routing-config", routing_config
            ]
            
            alias_result = self._run_aws_cli("lambda", "create-alias", args=alias_args)
            
            # If the alias already exists, update it
            if not alias_result["success"] and "ResourceConflictException" in str(alias_result.get("stderr", "")):
                logger.info("Alias 'canary' already exists, updating it")
                return self.update_canary_percentage(function_name, fix_id, traffic_percentage)
                
            # Log the canary setup
            try:
                get_audit_logger().log_event(
                    event_type="lambda_canary_setup",
                    details={
                        "function_name": function_name,
                        "lambda_function_name": full_function_name,
                        "fix_id": fix_id,
                        "version": version,
                        "traffic_percentage": traffic_percentage,
                        "success": alias_result["success"]
                    }
                )
            except Exception as e:
                logger.debug(f"Could not log to audit log: {str(e)}")
                
            return {
                "success": alias_result["success"],
                "function_name": full_function_name,
                "version": version,
                "alias": alias_name,
                "traffic_percentage": traffic_percentage
            }
            
        except Exception as e:
            logger.exception(f"Error setting up Lambda canary deployment: {str(e)}")
            return {"success": False, "error": str(e)}
            
    def update_canary_percentage(self, function_name: str, fix_id: str,
                               traffic_percentage: int, **kwargs) -> Dict[str, Any]:
        """Update canary deployment traffic percentage on AWS Lambda.
        
        Args:
            function_name: Name of the function
            fix_id: ID of the fix
            traffic_percentage: New percentage of traffic to route to new version
            **kwargs: Additional parameters
            
        Returns:
            Dict[str, Any]: Updated canary deployment information
        """
        # Generate function name with fix ID
        full_function_name = f"{function_name}-{fix_id}"
        alias_name = "canary"
        
        try:
            # Get the current alias configuration
            get_alias_args = [
                "--function-name", full_function_name,
                "--name", alias_name
            ]
            
            alias_result = self._run_aws_cli("lambda", "get-alias", args=get_alias_args)
            
            if not alias_result["success"] and not alias_result.get("simulated", False):
                logger.error(f"Failed to get Lambda alias: {alias_result}")
                return alias_result
                
            # Get the version being routed to
            if alias_result.get("simulated", False):
                version = "1"
            else:
                # Extract the version from the routing configuration
                routing_config = alias_result.get("RoutingConfig", {})
                additional_weights = routing_config.get("AdditionalVersionWeights", {})
                version = next(iter(additional_weights.keys())) if additional_weights else "1"
                
            # Update the alias with new routing configuration
            routing_config = json.dumps({
                "AdditionalVersionWeights": {
                    version: traffic_percentage / 100.0
                }
            })
            
            update_args = [
                "--function-name", full_function_name,
                "--name", alias_name,
                "--routing-config", routing_config
            ]
            
            update_result = self._run_aws_cli("lambda", "update-alias", args=update_args)
            
            # Log the canary update
            try:
                get_audit_logger().log_event(
                    event_type="lambda_canary_updated",
                    details={
                        "function_name": function_name,
                        "lambda_function_name": full_function_name,
                        "fix_id": fix_id,
                        "version": version,
                        "traffic_percentage": traffic_percentage,
                        "success": update_result["success"]
                    }
                )
            except Exception as e:
                logger.debug(f"Could not log to audit log: {str(e)}")
                
            return {
                "success": update_result["success"],
                "function_name": full_function_name,
                "version": version,
                "alias": alias_name,
                "traffic_percentage": traffic_percentage
            }
            
        except Exception as e:
            logger.exception(f"Error updating Lambda canary percentage: {str(e)}")
            return {"success": False, "error": str(e)}
            
    def complete_canary_deployment(self, function_name: str, fix_id: str, **kwargs) -> Dict[str, Any]:
        """Complete canary deployment by promoting new version to 100% on AWS Lambda.
        
        Args:
            function_name: Name of the function
            fix_id: ID of the fix
            **kwargs: Additional parameters
            
        Returns:
            Dict[str, Any]: Completion information
        """
        # Generate function name with fix ID
        full_function_name = f"{function_name}-{fix_id}"
        alias_name = "canary"
        
        try:
            # Get the current alias configuration
            get_alias_args = [
                "--function-name", full_function_name,
                "--name", alias_name
            ]
            
            alias_result = self._run_aws_cli("lambda", "get-alias", args=get_alias_args)
            
            if not alias_result["success"] and not alias_result.get("simulated", False):
                logger.error(f"Failed to get Lambda alias: {alias_result}")
                return alias_result
                
            # Get the version being routed to
            if alias_result.get("simulated", False):
                version = "1"
            else:
                # Extract the version from the routing configuration
                routing_config = alias_result.get("RoutingConfig", {})
                additional_weights = routing_config.get("AdditionalVersionWeights", {})
                version = next(iter(additional_weights.keys())) if additional_weights else "1"
                
            # Update the alias to point directly to the canary version
            update_args = [
                "--function-name", full_function_name,
                "--name", alias_name,
                "--function-version", version
            ]
            
            update_result = self._run_aws_cli("lambda", "update-alias", args=update_args)
            
            # Log the canary completion
            try:
                get_audit_logger().log_event(
                    event_type="lambda_canary_completed",
                    details={
                        "function_name": function_name,
                        "lambda_function_name": full_function_name,
                        "fix_id": fix_id,
                        "version": version,
                        "success": update_result["success"]
                    }
                )
            except Exception as e:
                logger.debug(f"Could not log to audit log: {str(e)}")
                
            return {
                "success": update_result["success"],
                "function_name": full_function_name,
                "version": version,
                "alias": alias_name,
                "traffic_percentage": 100
            }
            
        except Exception as e:
            logger.exception(f"Error completing Lambda canary deployment: {str(e)}")
            return {"success": False, "error": str(e)}
            
    def rollback_canary_deployment(self, function_name: str, fix_id: str, **kwargs) -> Dict[str, Any]:
        """Rollback canary deployment to previous version on AWS Lambda.
        
        Args:
            function_name: Name of the function
            fix_id: ID of the fix
            **kwargs: Additional parameters
            
        Returns:
            Dict[str, Any]: Rollback information
        """
        # Generate function name with fix ID
        full_function_name = f"{function_name}-{fix_id}"
        alias_name = "canary"
        
        try:
            # Update the alias to point directly to $LATEST and remove routing config
            update_args = [
                "--function-name", full_function_name,
                "--name", alias_name,
                "--function-version", "$LATEST"
            ]
            
            update_result = self._run_aws_cli("lambda", "update-alias", args=update_args)
            
            # Log the canary rollback
            try:
                get_audit_logger().log_event(
                    event_type="lambda_canary_rolled_back",
                    details={
                        "function_name": function_name,
                        "lambda_function_name": full_function_name,
                        "fix_id": fix_id,
                        "success": update_result["success"]
                    }
                )
            except Exception as e:
                logger.debug(f"Could not log to audit log: {str(e)}")
                
            return {
                "success": update_result["success"],
                "function_name": full_function_name,
                "version": "$LATEST",
                "alias": alias_name,
                "traffic_percentage": 0
            }
            
        except Exception as e:
            logger.exception(f"Error rolling back Lambda canary deployment: {str(e)}")
            return {"success": False, "error": str(e)}


# Singleton instance
_lambda_provider = None

def get_lambda_provider(config: Dict[str, Any] = None) -> AWSLambdaProvider:
    """Get or create the singleton AWSLambdaProvider instance.
    
    Args:
        config: Optional configuration
        
    Returns:
        AWSLambdaProvider: Singleton instance
    """
    global _lambda_provider
    if _lambda_provider is None:
        _lambda_provider = AWSLambdaProvider(config)
    return _lambda_provider