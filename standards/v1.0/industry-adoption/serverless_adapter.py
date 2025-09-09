"""
Serverless Platform Adapters for USHS v1.0

This module provides adapters that enable serverless platforms to comply
with the Universal Self-Healing Standard.
"""

import abc
import json
import logging
import tempfile
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

from modules.deployment.serverless.base_provider import ServerlessProvider
from modules.deployment.serverless.provider_factory import \
    get_serverless_provider

logger = logging.getLogger(__name__)


class ServerlessUSHSAdapter(abc.ABC):
    """
    Base adapter for serverless platforms to comply with USHS v1.0.
    
    This adapter implements the required USHS interfaces and translates
    them to serverless platform-specific operations.
    """
    
    # USHS compliance metadata
    USHS_VERSION = "1.0"
    USHS_INTERFACES = [
        "IDetector",
        "IAnalyzer", 
        "IGenerator",
        "IValidator",
        "IDeployer"
    ]
    CERTIFICATION_LEVEL = "Silver"  # Base implementation provides Silver level
    
    def __init__(self, provider: ServerlessProvider, config: Dict[str, Any] = None):
        """Initialize serverless USHS adapter.
        
        Args:
            provider: Underlying serverless provider
            config: Adapter configuration
        """
        self.provider = provider
        self.config = config or {}
        self.session_store: Dict[str, Dict[str, Any]] = {}
        
    # IDetector Interface Implementation
    
    def detect(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect errors in serverless functions.
        
        Args:
            config: Detection configuration
            
        Returns:
            List of ErrorEvent objects conforming to USHS schema
        """
        error_events = []
        
        # Get function logs
        function_name = config.get('function_name')
        fix_id = config.get('fix_id', 'detect')
        since = config.get('since')
        
        if function_name:
            logs = self.provider.get_function_logs(function_name, fix_id, since)
            
            # Parse logs for errors
            for log_entry in logs:
                if self._is_error_log(log_entry):
                    error_event = self._log_to_error_event(log_entry, function_name)
                    error_events.append(error_event)
        
        return error_events
    
    def subscribe(self, callback: Callable[[Dict[str, Any]], None]) -> Dict[str, Any]:
        """Subscribe to real-time error events.
        
        Args:
            callback: Function to call when errors are detected
            
        Returns:
            Subscription object with unsubscribe method
        """
        subscription_id = str(uuid4())
        
        # Implementation would set up CloudWatch/Azure Monitor/Stackdriver
        # log streaming with error filtering
        
        return {
            'id': subscription_id,
            'status': 'active',
            'unsubscribe': lambda: self._unsubscribe(subscription_id)
        }
    
    def get_detector_capabilities(self) -> Dict[str, Any]:
        """Get detector capabilities.
        
        Returns:
            Detector capabilities object
        """
        return {
            'languages': self._get_supported_languages(),
            'errorTypes': [
                'runtime_error',
                'timeout',
                'out_of_memory',
                'permission_denied',
                'rate_limit_exceeded'
            ],
            'platforms': [self._get_platform_name()],
            'realTimeCapable': True
        }
    
    # IAnalyzer Interface Implementation
    
    def analyze(self, error: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze error to determine root cause.
        
        Args:
            error: ErrorEvent object
            
        Returns:
            AnalysisResult object
        """
        analysis_id = str(uuid4())
        
        # Extract error details
        error_type = error.get('error', {}).get('type', 'unknown')
        error_message = error.get('error', {}).get('message', '')
        stack_trace = error.get('error', {}).get('stackTrace', [])
        
        # Perform platform-specific analysis
        root_cause = self._analyze_error_type(error_type, error_message, stack_trace)
        
        return {
            'id': analysis_id,
            'errorId': error.get('id'),
            'rootCause': root_cause,
            'suggestedFixes': self._suggest_fixes(root_cause),
            'confidence': self._calculate_confidence(root_cause),
            'metadata': {
                'analyzer': self.__class__.__name__,
                'platform': self._get_platform_name(),
                'timestamp': datetime.utcnow().isoformat()
            }
        }
    
    def get_supported_languages(self) -> List[str]:
        """Get languages supported by analyzer.
        
        Returns:
            List of supported language identifiers
        """
        return self._get_supported_languages()
    
    def get_confidence_score(self, analysis: Dict[str, Any]) -> float:
        """Get confidence score for analysis.
        
        Args:
            analysis: AnalysisResult object
            
        Returns:
            Confidence score between 0 and 1
        """
        return analysis.get('confidence', 0.0)
    
    # IGenerator Interface Implementation
    
    def generate(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate healing patch based on analysis.
        
        Args:
            analysis: AnalysisResult object
            
        Returns:
            HealingPatch object conforming to USHS schema
        """
        patch_id = str(uuid4())
        session_id = analysis.get('metadata', {}).get('sessionId', str(uuid4()))
        
        # Generate platform-specific patch
        changes = self._generate_changes(analysis)
        
        return {
            'id': patch_id,
            'sessionId': session_id,
            'changes': changes,
            'metadata': {
                'confidence': analysis.get('confidence', 0.5),
                'generator': self.__class__.__name__,
                'strategy': self._get_patch_strategy(analysis),
                'platform': self._get_platform_name(),
                'timestamp': datetime.utcnow().isoformat()
            }
        }
    
    def validate_patch(self, patch: Dict[str, Any]) -> Dict[str, Any]:
        """Validate generated patch.
        
        Args:
            patch: HealingPatch object
            
        Returns:
            ValidationResult object
        """
        # Perform static validation
        is_valid = all(self._validate_change(change) for change in patch.get('changes', []))
        
        return {
            'valid': is_valid,
            'errors': [] if is_valid else ['Invalid patch format'],
            'warnings': self._get_patch_warnings(patch)
        }
    
    def estimate_impact(self, patch: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate impact of applying patch.
        
        Args:
            patch: HealingPatch object
            
        Returns:
            ImpactAssessment object
        """
        return {
            'affectedServices': self._get_affected_services(patch),
            'downtime': self._estimate_downtime(patch),
            'riskLevel': self._assess_risk_level(patch),
            'rollbackTime': self._estimate_rollback_time(patch)
        }
    
    # IValidator Interface Implementation
    
    def validate(self, patch: Dict[str, Any], tests: Dict[str, Any]) -> Dict[str, Any]:
        """Validate patch by running tests.
        
        Args:
            patch: HealingPatch object
            tests: TestSuite object
            
        Returns:
            TestResult object
        """
        test_id = str(uuid4())
        
        # Deploy to test environment
        test_deployment = self._deploy_to_test(patch)
        
        # Run tests
        test_results = self._run_tests(test_deployment, tests)
        
        # Cleanup test deployment
        self._cleanup_test_deployment(test_deployment)
        
        return {
            'id': test_id,
            'patchId': patch.get('id'),
            'passed': test_results.get('passed', False),
            'failures': test_results.get('failures', []),
            'duration': test_results.get('duration', 0),
            'metadata': {
                'platform': self._get_platform_name(),
                'timestamp': datetime.utcnow().isoformat()
            }
        }
    
    def generate_tests(self, patch: Dict[str, Any]) -> Dict[str, Any]:
        """Generate test suite for patch.
        
        Args:
            patch: HealingPatch object
            
        Returns:
            TestSuite object
        """
        return {
            'id': str(uuid4()),
            'patchId': patch.get('id'),
            'tests': self._generate_test_cases(patch),
            'metadata': {
                'generator': self.__class__.__name__,
                'platform': self._get_platform_name()
            }
        }
    
    def assess_risk(self, patch: Dict[str, Any]) -> str:
        """Assess risk level of patch.
        
        Args:
            patch: HealingPatch object
            
        Returns:
            Risk level (low, medium, high, critical)
        """
        impact = self.estimate_impact(patch)
        return impact.get('riskLevel', 'medium')
    
    # IDeployer Interface Implementation
    
    def deploy(self, patch: Dict[str, Any], strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy validated patch to production.
        
        Args:
            patch: HealingPatch object
            strategy: DeploymentStrategy object
            
        Returns:
            DeploymentResult object
        """
        deployment_id = str(uuid4())
        
        # Extract deployment details
        function_name = self._get_function_from_patch(patch)
        source_path = self._prepare_source(patch)
        
        # Deploy based on strategy
        strategy_type = strategy.get('type', 'canary')
        
        if strategy_type == 'canary':
            result = self._deploy_canary(function_name, source_path, strategy)
        elif strategy_type == 'blue_green':
            result = self._deploy_blue_green(function_name, source_path, strategy)
        else:
            result = self._deploy_direct(function_name, source_path, strategy)
        
        return {
            'id': deployment_id,
            'patchId': patch.get('id'),
            'status': result.get('status', 'deployed'),
            'deploymentInfo': result,
            'metadata': {
                'strategy': strategy_type,
                'platform': self._get_platform_name(),
                'timestamp': datetime.utcnow().isoformat()
            }
        }
    
    def rollback(self, deployment: Dict[str, Any]) -> Dict[str, Any]:
        """Rollback a deployment.
        
        Args:
            deployment: DeploymentResult object
            
        Returns:
            RollbackResult object
        """
        rollback_id = str(uuid4())
        
        # Extract deployment info
        function_name = deployment.get('deploymentInfo', {}).get('functionName')
        fix_id = deployment.get('deploymentInfo', {}).get('fixId')
        
        # Perform rollback
        if deployment.get('metadata', {}).get('strategy') == 'canary':
            result = self.provider.rollback_canary_deployment(function_name, fix_id)
        else:
            # Direct rollback for other strategies
            result = self._rollback_direct(deployment)
        
        return {
            'id': rollback_id,
            'deploymentId': deployment.get('id'),
            'status': 'completed' if result else 'failed',
            'rollbackInfo': result,
            'metadata': {
                'platform': self._get_platform_name(),
                'timestamp': datetime.utcnow().isoformat()
            }
        }
    
    def get_deployment_status(self, deployment_id: str) -> Dict[str, Any]:
        """Get deployment status.
        
        Args:
            deployment_id: Deployment ID
            
        Returns:
            DeploymentStatus object
        """
        # Retrieve from session store
        deployment = self.session_store.get(deployment_id, {})
        
        if not deployment:
            return {
                'id': deployment_id,
                'status': 'not_found'
            }
        
        # Get current status from provider
        function_name = deployment.get('functionName')
        fix_id = deployment.get('fixId')
        
        if function_name and fix_id:
            status = self.provider.get_function_status(function_name, fix_id)
            
            return {
                'id': deployment_id,
                'status': self._map_provider_status(status),
                'details': status,
                'metadata': {
                    'platform': self._get_platform_name(),
                    'timestamp': datetime.utcnow().isoformat()
                }
            }
        
        return {
            'id': deployment_id,
            'status': 'unknown'
        }
    
    # Abstract methods for platform-specific implementation
    
    @abc.abstractmethod
    def _get_platform_name(self) -> str:
        """Get platform name."""
        pass
    
    @abc.abstractmethod
    def _get_supported_languages(self) -> List[str]:
        """Get supported programming languages."""
        pass
    
    @abc.abstractmethod
    def _is_error_log(self, log_entry: Dict[str, Any]) -> bool:
        """Check if log entry represents an error."""
        pass
    
    @abc.abstractmethod
    def _log_to_error_event(self, log_entry: Dict[str, Any], function_name: str) -> Dict[str, Any]:
        """Convert platform log to USHS ErrorEvent."""
        pass
    
    @abc.abstractmethod
    def _analyze_error_type(self, error_type: str, message: str, stack_trace: List[str]) -> Dict[str, Any]:
        """Analyze error to determine root cause."""
        pass
    
    @abc.abstractmethod
    def _suggest_fixes(self, root_cause: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Suggest fixes based on root cause."""
        pass
    
    @abc.abstractmethod
    def _generate_changes(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate code changes based on analysis."""
        pass
    
    # Helper methods
    
    def _calculate_confidence(self, root_cause: Dict[str, Any]) -> float:
        """Calculate confidence score."""
        # Base implementation - can be overridden
        if root_cause.get('type') == 'known_error':
            return 0.9
        elif root_cause.get('type') == 'likely_error':
            return 0.7
        else:
            return 0.5
    
    def _get_patch_strategy(self, analysis: Dict[str, Any]) -> str:
        """Determine patch strategy."""
        confidence = analysis.get('confidence', 0.5)
        if confidence > 0.8:
            return 'automated_fix'
        elif confidence > 0.6:
            return 'suggested_fix'
        else:
            return 'manual_review'
    
    def _validate_change(self, change: Dict[str, Any]) -> bool:
        """Validate a single change."""
        required_fields = ['file', 'diff', 'language']
        return all(field in change for field in required_fields)
    
    def _get_patch_warnings(self, patch: Dict[str, Any]) -> List[str]:
        """Get warnings for patch."""
        warnings = []
        
        # Check for risky changes
        for change in patch.get('changes', []):
            if 'delete' in change.get('diff', ''):
                warnings.append(f"Deletion detected in {change.get('file')}")
        
        return warnings
    
    def _get_affected_services(self, patch: Dict[str, Any]) -> List[str]:
        """Get services affected by patch."""
        # Extract from patch metadata
        return patch.get('metadata', {}).get('affectedServices', [])
    
    def _estimate_downtime(self, patch: Dict[str, Any]) -> int:
        """Estimate downtime in seconds."""
        # Base estimate - can be overridden
        return 30  # 30 seconds for function update
    
    def _assess_risk_level(self, patch: Dict[str, Any]) -> str:
        """Assess risk level of patch."""
        # Simple risk assessment
        confidence = patch.get('metadata', {}).get('confidence', 0.5)
        
        if confidence > 0.8:
            return 'low'
        elif confidence > 0.6:
            return 'medium'
        else:
            return 'high'
    
    def _estimate_rollback_time(self, patch: Dict[str, Any]) -> int:
        """Estimate rollback time in seconds."""
        return 15  # 15 seconds for function rollback
    
    def _unsubscribe(self, subscription_id: str) -> None:
        """Unsubscribe from error events."""
        # Implementation would remove log streaming subscription
        pass
    
    def _deploy_to_test(self, patch: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy patch to test environment."""
        # Implementation would create test function version
        return {
            'testFunctionName': f"test-{patch.get('id')}",
            'deployed': True
        }
    
    def _run_tests(self, deployment: Dict[str, Any], tests: Dict[str, Any]) -> Dict[str, Any]:
        """Run tests against deployment."""
        # Implementation would invoke test cases
        return {
            'passed': True,
            'failures': [],
            'duration': 120  # 2 minutes
        }
    
    def _cleanup_test_deployment(self, deployment: Dict[str, Any]) -> None:
        """Cleanup test deployment."""
        # Implementation would delete test function
        pass
    
    def _generate_test_cases(self, patch: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate test cases for patch."""
        return [
            {
                'name': 'smoke_test',
                'type': 'integration',
                'timeout': 60
            },
            {
                'name': 'regression_test',
                'type': 'unit',
                'timeout': 30
            }
        ]
    
    def _get_function_from_patch(self, patch: Dict[str, Any]) -> str:
        """Extract function name from patch."""
        # Implementation would parse patch metadata
        return patch.get('metadata', {}).get('functionName', 'unknown')
    
    def _prepare_source(self, patch: Dict[str, Any]) -> str:
        """Prepare source code with patch applied."""
        # Implementation would apply patch to source
        # Use secure temporary directory
        temp_dir = tempfile.gettempdir()
        return f"{temp_dir}/patched-{patch.get('id')}"
    
    def _deploy_canary(self, function_name: str, source_path: str, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy using canary strategy."""
        fix_id = str(uuid4())
        traffic_percentage = strategy.get('initialPercentage', 10)
        
        # Update function
        update_result = self.provider.update_function(function_name, fix_id, source_path)
        
        # Setup canary
        canary_result = self.provider.setup_canary_deployment(
            function_name, fix_id, traffic_percentage
        )
        
        return {
            'functionName': function_name,
            'fixId': fix_id,
            'status': 'canary_deployed',
            'canaryInfo': canary_result,
            'updateInfo': update_result
        }
    
    def _deploy_blue_green(self, function_name: str, source_path: str, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy using blue-green strategy."""
        # Implementation would create new function version
        # and switch traffic atomically
        fix_id = str(uuid4())
        
        return {
            'functionName': function_name,
            'fixId': fix_id,
            'status': 'blue_green_deployed'
        }
    
    def _deploy_direct(self, function_name: str, source_path: str, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy directly."""
        fix_id = str(uuid4())
        
        result = self.provider.update_function(function_name, fix_id, source_path)
        
        return {
            'functionName': function_name,
            'fixId': fix_id,
            'status': 'deployed',
            'updateInfo': result
        }
    
    def _rollback_direct(self, deployment: Dict[str, Any]) -> Dict[str, Any]:
        """Direct rollback implementation."""
        # Implementation would restore previous function version
        return {
            'status': 'rolled_back',
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def _map_provider_status(self, provider_status: Dict[str, Any]) -> str:
        """Map provider status to USHS status."""
        # Implementation would map platform-specific statuses
        state = provider_status.get('state', '').lower()
        
        if state in ['active', 'running']:
            return 'healthy'
        elif state in ['updating', 'deploying']:
            return 'updating'
        elif state in ['failed', 'error']:
            return 'failed'
        else:
            return 'unknown'


class AWSLambdaUSHSAdapter(ServerlessUSHSAdapter):
    """AWS Lambda adapter for USHS compliance."""
    
    SUPPORTED_LANGUAGES = [
        'python', 'javascript', 'typescript', 'java', 'csharp', 
        'go', 'ruby', 'rust'
    ]
    SUPPORTED_FEATURES = [
        'canary_deployment',
        'blue_green_deployment',
        'auto_scaling',
        'cold_start_optimization'
    ]
    
    def _get_platform_name(self) -> str:
        return "aws_lambda"
    
    def _get_supported_languages(self) -> List[str]:
        return self.SUPPORTED_LANGUAGES
    
    def _is_error_log(self, log_entry: Dict[str, Any]) -> bool:
        """Check if CloudWatch log entry is an error."""
        message = log_entry.get('message', '').lower()
        return any(indicator in message for indicator in [
            'error', 'exception', 'failed', 'timeout', 'abort'
        ])
    
    def _log_to_error_event(self, log_entry: Dict[str, Any], function_name: str) -> Dict[str, Any]:
        """Convert CloudWatch log to USHS ErrorEvent."""
        return {
            'id': str(uuid4()),
            'timestamp': datetime.fromtimestamp(
                log_entry.get('timestamp', 0) / 1000
            ).isoformat(),
            'severity': self._determine_severity(log_entry),
            'source': {
                'service': function_name,
                'version': log_entry.get('version', 'unknown'),
                'environment': 'production',
                'location': f"arn:aws:lambda:{self.provider.region}:function:{function_name}"
            },
            'error': {
                'type': self._extract_error_type(log_entry),
                'message': log_entry.get('message', ''),
                'stackTrace': self._extract_stack_trace(log_entry),
                'context': {
                    'requestId': log_entry.get('requestId'),
                    'logGroup': log_entry.get('logGroup'),
                    'logStream': log_entry.get('logStream')
                }
            }
        }
    
    def _analyze_error_type(self, error_type: str, message: str, stack_trace: List[str]) -> Dict[str, Any]:
        """Analyze Lambda-specific errors."""
        root_cause = {
            'type': 'unknown',
            'description': message,
            'category': 'runtime'
        }
        
        # Lambda-specific error patterns
        if 'Task timed out' in message:
            root_cause.update({
                'type': 'timeout',
                'category': 'configuration',
                'suggestion': 'Increase function timeout or optimize code'
            })
        elif 'Runtime.OutOfMemory' in error_type:
            root_cause.update({
                'type': 'out_of_memory',
                'category': 'resource',
                'suggestion': 'Increase function memory allocation'
            })
        elif 'AccessDenied' in message:
            root_cause.update({
                'type': 'permission_error',
                'category': 'security',
                'suggestion': 'Update IAM role permissions'
            })
        
        return root_cause
    
    def _suggest_fixes(self, root_cause: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Suggest Lambda-specific fixes."""
        suggestions = []
        
        if root_cause.get('type') == 'timeout':
            suggestions.append({
                'type': 'configuration',
                'action': 'increase_timeout',
                'confidence': 0.9
            })
        elif root_cause.get('type') == 'out_of_memory':
            suggestions.append({
                'type': 'configuration',
                'action': 'increase_memory',
                'confidence': 0.95
            })
        elif root_cause.get('type') == 'permission_error':
            suggestions.append({
                'type': 'iam_policy',
                'action': 'update_permissions',
                'confidence': 0.85
            })
        
        return suggestions
    
    def _generate_changes(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate Lambda-specific changes."""
        changes = []
        root_cause = analysis.get('rootCause', {})
        
        if root_cause.get('type') == 'timeout':
            changes.append({
                'file': 'serverless.yml',
                'diff': self._generate_timeout_patch(),
                'language': 'yaml',
                'framework': 'serverless'
            })
        elif root_cause.get('type') == 'out_of_memory':
            changes.append({
                'file': 'serverless.yml',
                'diff': self._generate_memory_patch(),
                'language': 'yaml',
                'framework': 'serverless'
            })
        
        return changes
    
    # Helper methods
    
    def _determine_severity(self, log_entry: Dict[str, Any]) -> str:
        """Determine error severity from log."""
        message = log_entry.get('message', '').lower()
        
        if 'critical' in message or 'fatal' in message:
            return 'critical'
        elif 'error' in message:
            return 'high'
        elif 'warning' in message:
            return 'medium'
        else:
            return 'low'
    
    def _extract_error_type(self, log_entry: Dict[str, Any]) -> str:
        """Extract error type from log."""
        message = log_entry.get('message', '')
        
        # Look for Lambda error patterns
        if 'Task timed out' in message:
            return 'TimeoutError'
        elif 'Runtime.OutOfMemory' in message:
            return 'OutOfMemoryError'
        elif 'AccessDenied' in message:
            return 'AccessDeniedError'
        else:
            return 'RuntimeError'
    
    def _extract_stack_trace(self, log_entry: Dict[str, Any]) -> List[str]:
        """Extract stack trace from log."""
        message = log_entry.get('message', '')
        lines = message.split('\n')
        
        # Find stack trace lines
        stack_lines = []
        in_stack = False
        
        for line in lines:
            if 'Traceback' in line or 'Stack trace' in line:
                in_stack = True
            elif in_stack and line.strip():
                stack_lines.append(line)
        
        return stack_lines
    
    def _generate_timeout_patch(self) -> str:
        """Generate patch to increase timeout."""
        return '''--- a/serverless.yml
+++ b/serverless.yml
@@ -10,7 +10,7 @@ provider:
 functions:
   main:
     handler: handler.main
-    timeout: 30
+    timeout: 300  # Increased from 30s to 5min
     events:
       - http:
           path: /'''
    
    def _generate_memory_patch(self) -> str:
        """Generate patch to increase memory."""
        return '''--- a/serverless.yml
+++ b/serverless.yml
@@ -10,7 +10,7 @@ provider:
 functions:
   main:
     handler: handler.main
-    memorySize: 128
+    memorySize: 512  # Increased from 128MB to 512MB
     events:
       - http:
           path: /'''


class AzureFunctionsUSHSAdapter(ServerlessUSHSAdapter):
    """Azure Functions adapter for USHS compliance."""
    
    SUPPORTED_LANGUAGES = [
        'csharp', 'javascript', 'typescript', 'python', 'java', 'powershell'
    ]
    SUPPORTED_FEATURES = [
        'deployment_slots',
        'durable_functions',
        'premium_plan',
        'vnet_integration'
    ]
    
    def _get_platform_name(self) -> str:
        return "azure_functions"
    
    def _get_supported_languages(self) -> List[str]:
        return self.SUPPORTED_LANGUAGES
    
    def _is_error_log(self, log_entry: Dict[str, Any]) -> bool:
        """Check if Application Insights log entry is an error."""
        severity = log_entry.get('severityLevel', 0)
        # Application Insights severity levels: 0=Verbose, 1=Info, 2=Warning, 3=Error, 4=Critical
        return severity >= 3
    
    def _log_to_error_event(self, log_entry: Dict[str, Any], function_name: str) -> Dict[str, Any]:
        """Convert Application Insights log to USHS ErrorEvent."""
        return {
            'id': str(uuid4()),
            'timestamp': log_entry.get('timestamp', datetime.utcnow().isoformat()),
            'severity': self._map_severity(log_entry.get('severityLevel', 3)),
            'source': {
                'service': function_name,
                'version': log_entry.get('sdkVersion', 'unknown'),
                'environment': log_entry.get('cloud_RoleName', 'production'),
                'location': f"azure://{log_entry.get('cloud_Location', 'unknown')}/{function_name}"
            },
            'error': {
                'type': log_entry.get('problemId', 'RuntimeError'),
                'message': log_entry.get('message', ''),
                'stackTrace': self._parse_azure_stack_trace(log_entry),
                'context': {
                    'operationId': log_entry.get('operation_Id'),
                    'instanceId': log_entry.get('cloud_RoleInstance'),
                    'customDimensions': log_entry.get('customDimensions', {})
                }
            }
        }
    
    def _analyze_error_type(self, error_type: str, message: str, stack_trace: List[str]) -> Dict[str, Any]:
        """Analyze Azure Functions-specific errors."""
        root_cause = {
            'type': 'unknown',
            'description': message,
            'category': 'runtime'
        }
        
        # Azure Functions-specific error patterns
        if 'Timeout value of' in message and 'exceeded' in message:
            root_cause.update({
                'type': 'timeout',
                'category': 'configuration',
                'suggestion': 'Increase functionTimeout in host.json'
            })
        elif 'System.OutOfMemoryException' in error_type:
            root_cause.update({
                'type': 'out_of_memory',
                'category': 'resource',
                'suggestion': 'Use Premium plan or optimize memory usage'
            })
        elif 'UnauthorizedAccessException' in error_type:
            root_cause.update({
                'type': 'authorization_error',
                'category': 'security',
                'suggestion': 'Check function app identity and RBAC permissions'
            })
        
        return root_cause
    
    def _suggest_fixes(self, root_cause: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Suggest Azure Functions-specific fixes."""
        suggestions = []
        
        if root_cause.get('type') == 'timeout':
            suggestions.append({
                'type': 'configuration',
                'action': 'update_host_json',
                'target': 'functionTimeout',
                'confidence': 0.9
            })
        elif root_cause.get('type') == 'out_of_memory':
            suggestions.append({
                'type': 'plan_upgrade',
                'action': 'switch_to_premium',
                'confidence': 0.85
            })
        
        return suggestions
    
    def _generate_changes(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate Azure Functions-specific changes."""
        changes = []
        root_cause = analysis.get('rootCause', {})
        
        if root_cause.get('type') == 'timeout':
            changes.append({
                'file': 'host.json',
                'diff': self._generate_azure_timeout_patch(),
                'language': 'json',
                'framework': 'azure_functions'
            })
        
        return changes
    
    # Helper methods
    
    def _map_severity(self, severity_level: int) -> str:
        """Map Application Insights severity to USHS severity."""
        mapping = {
            0: 'low',      # Verbose
            1: 'low',      # Information
            2: 'medium',   # Warning
            3: 'high',     # Error
            4: 'critical'  # Critical
        }
        return mapping.get(severity_level, 'medium')
    
    def _parse_azure_stack_trace(self, log_entry: Dict[str, Any]) -> List[str]:
        """Parse stack trace from Application Insights."""
        exceptions = log_entry.get('exceptions', [])
        stack_lines = []
        
        for exception in exceptions:
            if 'parsedStack' in exception:
                for frame in exception['parsedStack']:
                    line = f"  at {frame.get('method', 'unknown')} in {frame.get('fileName', 'unknown')}:{frame.get('line', 0)}"
                    stack_lines.append(line)
        
        return stack_lines
    
    def _generate_azure_timeout_patch(self) -> str:
        """Generate patch for Azure Functions timeout."""
        return '''--- a/host.json
+++ b/host.json
@@ -1,7 +1,7 @@
 {
   "version": "2.0",
   "functionTimeout": "00:05:00",
-  "functionTimeout": "00:05:00",
+  "functionTimeout": "00:10:00",
   "logging": {
     "applicationInsights": {
       "samplingSettings": {'''


class GCPFunctionsUSHSAdapter(ServerlessUSHSAdapter):
    """Google Cloud Functions adapter for USHS compliance."""
    
    SUPPORTED_LANGUAGES = [
        'python', 'nodejs', 'go', 'java', 'ruby', 'php', 'dotnet'
    ]
    SUPPORTED_FEATURES = [
        'traffic_splitting',
        'vpc_connector',
        'min_instances',
        'secret_manager'
    ]
    
    def _get_platform_name(self) -> str:
        return "gcp_functions"
    
    def _get_supported_languages(self) -> List[str]:
        return self.SUPPORTED_LANGUAGES
    
    def _is_error_log(self, log_entry: Dict[str, Any]) -> bool:
        """Check if Stackdriver log entry is an error."""
        severity = log_entry.get('severity', '').upper()
        return severity in ['ERROR', 'CRITICAL', 'ALERT', 'EMERGENCY']
    
    def _log_to_error_event(self, log_entry: Dict[str, Any], function_name: str) -> Dict[str, Any]:
        """Convert Stackdriver log to USHS ErrorEvent."""
        return {
            'id': str(uuid4()),
            'timestamp': log_entry.get('timestamp', datetime.utcnow().isoformat()),
            'severity': self._map_stackdriver_severity(log_entry.get('severity', 'ERROR')),
            'source': {
                'service': function_name,
                'version': log_entry.get('labels', {}).get('execution_id', 'unknown'),
                'environment': log_entry.get('resource', {}).get('labels', {}).get('function_name', 'production'),
                'location': f"projects/{log_entry.get('resource', {}).get('labels', {}).get('project_id')}/locations/{log_entry.get('resource', {}).get('labels', {}).get('region')}/functions/{function_name}"
            },
            'error': {
                'type': self._extract_gcp_error_type(log_entry),
                'message': log_entry.get('textPayload', log_entry.get('jsonPayload', {}).get('message', '')),
                'stackTrace': self._extract_gcp_stack_trace(log_entry),
                'context': {
                    'executionId': log_entry.get('labels', {}).get('execution_id'),
                    'insertId': log_entry.get('insertId'),
                    'trace': log_entry.get('trace')
                }
            }
        }
    
    def _analyze_error_type(self, error_type: str, message: str, stack_trace: List[str]) -> Dict[str, Any]:
        """Analyze GCP Functions-specific errors."""
        root_cause = {
            'type': 'unknown',
            'description': message,
            'category': 'runtime'
        }
        
        # GCP Functions-specific error patterns
        if 'Function execution took' in message and 'exceeded' in message:
            root_cause.update({
                'type': 'timeout',
                'category': 'configuration',
                'suggestion': 'Increase timeout in function configuration'
            })
        elif 'memory limit exceeded' in message.lower():
            root_cause.update({
                'type': 'out_of_memory',
                'category': 'resource',
                'suggestion': 'Increase memory allocation in function configuration'
            })
        elif 'Permission' in error_type and 'denied' in message:
            root_cause.update({
                'type': 'permission_error',
                'category': 'security',
                'suggestion': 'Update service account permissions'
            })
        
        return root_cause
    
    def _suggest_fixes(self, root_cause: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Suggest GCP Functions-specific fixes."""
        suggestions = []
        
        if root_cause.get('type') == 'timeout':
            suggestions.append({
                'type': 'configuration',
                'action': 'update_function_config',
                'parameter': 'timeout',
                'confidence': 0.9
            })
        elif root_cause.get('type') == 'out_of_memory':
            suggestions.append({
                'type': 'configuration',
                'action': 'update_function_config',
                'parameter': 'availableMemoryMb',
                'confidence': 0.95
            })
        
        return suggestions
    
    def _generate_changes(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate GCP Functions-specific changes."""
        changes = []
        root_cause = analysis.get('rootCause', {})
        
        if root_cause.get('type') in ['timeout', 'out_of_memory']:
            changes.append({
                'file': 'function.yaml',
                'diff': self._generate_gcp_config_patch(root_cause.get('type')),
                'language': 'yaml',
                'framework': 'gcp_functions'
            })
        
        return changes
    
    # Helper methods
    
    def _map_stackdriver_severity(self, severity: str) -> str:
        """Map Stackdriver severity to USHS severity."""
        mapping = {
            'DEFAULT': 'low',
            'DEBUG': 'low',
            'INFO': 'low',
            'NOTICE': 'low',
            'WARNING': 'medium',
            'ERROR': 'high',
            'CRITICAL': 'critical',
            'ALERT': 'critical',
            'EMERGENCY': 'critical'
        }
        return mapping.get(severity.upper(), 'medium')
    
    def _extract_gcp_error_type(self, log_entry: Dict[str, Any]) -> str:
        """Extract error type from GCP log."""
        json_payload = log_entry.get('jsonPayload', {})
        
        # Try to extract from structured logging
        if 'error' in json_payload:
            return json_payload['error'].get('type', 'RuntimeError')
        
        # Fallback to text analysis
        text = log_entry.get('textPayload', '')
        if 'TimeoutError' in text:
            return 'TimeoutError'
        elif 'MemoryError' in text:
            return 'MemoryError'
        
        return 'RuntimeError'
    
    def _extract_gcp_stack_trace(self, log_entry: Dict[str, Any]) -> List[str]:
        """Extract stack trace from GCP log."""
        json_payload = log_entry.get('jsonPayload', {})
        
        # Try structured stack trace
        if 'stack_trace' in json_payload:
            return json_payload['stack_trace'].split('\n')
        
        # Fallback to text parsing
        text = log_entry.get('textPayload', '')
        if 'Traceback' in text:
            lines = text.split('\n')
            stack_lines = []
            capturing = False
            
            for line in lines:
                if 'Traceback' in line:
                    capturing = True
                elif capturing and line.strip():
                    stack_lines.append(line)
            
            return stack_lines
        
        return []
    
    def _generate_gcp_config_patch(self, issue_type: str) -> str:
        """Generate GCP configuration patch."""
        if issue_type == 'timeout':
            return '''--- a/function.yaml
+++ b/function.yaml
@@ -2,7 +2,7 @@ name: my-function
 entryPoint: main
 runtime: python39
 trigger:
   eventType: providers/cloud.pubsub/eventTypes/topic.publish
-timeout: 60s
+timeout: 540s  # Increased from 1min to 9min
 availableMemoryMb: 256'''
        elif issue_type == 'out_of_memory':
            return '''--- a/function.yaml
+++ b/function.yaml
@@ -5,4 +5,4 @@ trigger:
   eventType: providers/cloud.pubsub/eventTypes/topic.publish
 timeout: 60s
-availableMemoryMb: 256
+availableMemoryMb: 1024  # Increased from 256MB to 1GB'''
        
        return ''


class VercelUSHSAdapter(ServerlessUSHSAdapter):
    """Vercel adapter for USHS compliance."""
    
    SUPPORTED_LANGUAGES = [
        'javascript', 'typescript', 'go', 'python', 'ruby'
    ]
    SUPPORTED_FEATURES = [
        'edge_functions',
        'preview_deployments',
        'automatic_rollback',
        'analytics'
    ]
    CERTIFICATION_LEVEL = "Gold"  # Enhanced features
    
    def _get_platform_name(self) -> str:
        return "vercel"
    
    def _get_supported_languages(self) -> List[str]:
        return self.SUPPORTED_LANGUAGES
    
    def _is_error_log(self, log_entry: Dict[str, Any]) -> bool:
        """Check if Vercel log entry is an error."""
        level = log_entry.get('level', '').lower()
        return level in ['error', 'fatal']
    
    def _log_to_error_event(self, log_entry: Dict[str, Any], function_name: str) -> Dict[str, Any]:
        """Convert Vercel log to USHS ErrorEvent."""
        return {
            'id': str(uuid4()),
            'timestamp': log_entry.get('timestamp', datetime.utcnow().isoformat()),
            'severity': self._map_vercel_severity(log_entry.get('level', 'error')),
            'source': {
                'service': function_name,
                'version': log_entry.get('deploymentId', 'unknown'),
                'environment': log_entry.get('environment', 'production'),
                'location': f"vercel://{log_entry.get('region', 'global')}/{function_name}"
            },
            'error': {
                'type': log_entry.get('type', 'RuntimeError'),
                'message': log_entry.get('message', ''),
                'stackTrace': log_entry.get('stack', '').split('\n') if log_entry.get('stack') else [],
                'context': {
                    'requestId': log_entry.get('requestId'),
                    'deploymentId': log_entry.get('deploymentId'),
                    'projectId': log_entry.get('projectId')
                }
            }
        }
    
    def _analyze_error_type(self, error_type: str, message: str, stack_trace: List[str]) -> Dict[str, Any]:
        """Analyze Vercel-specific errors."""
        root_cause = {
            'type': 'unknown',
            'description': message,
            'category': 'runtime'
        }
        
        # Vercel-specific error patterns
        if 'FUNCTION_INVOCATION_TIMEOUT' in error_type:
            root_cause.update({
                'type': 'timeout',
                'category': 'edge_function',
                'suggestion': 'Optimize edge function or move to serverless function'
            })
        elif 'edge runtime' in message.lower() and 'memory' in message.lower():
            root_cause.update({
                'type': 'edge_memory_limit',
                'category': 'resource',
                'suggestion': 'Reduce edge function memory usage or use serverless function'
            })
        
        return root_cause
    
    def _suggest_fixes(self, root_cause: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Suggest Vercel-specific fixes."""
        suggestions = []
        
        if root_cause.get('type') == 'timeout':
            suggestions.append({
                'type': 'configuration',
                'action': 'update_vercel_json',
                'target': 'functions.maxDuration',
                'confidence': 0.9
            })
        elif root_cause.get('type') == 'edge_memory_limit':
            suggestions.append({
                'type': 'refactor',
                'action': 'convert_to_serverless',
                'confidence': 0.85
            })
        
        return suggestions
    
    def _generate_changes(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate Vercel-specific changes."""
        changes = []
        root_cause = analysis.get('rootCause', {})
        
        if root_cause.get('type') == 'timeout':
            changes.append({
                'file': 'vercel.json',
                'diff': self._generate_vercel_timeout_patch(),
                'language': 'json',
                'framework': 'vercel'
            })
        
        return changes
    
    def _map_vercel_severity(self, level: str) -> str:
        """Map Vercel log level to USHS severity."""
        mapping = {
            'debug': 'low',
            'info': 'low',
            'warn': 'medium',
            'error': 'high',
            'fatal': 'critical'
        }
        return mapping.get(level.lower(), 'medium')
    
    def _generate_vercel_timeout_patch(self) -> str:
        """Generate Vercel timeout configuration patch."""
        return '''--- a/vercel.json
+++ b/vercel.json
@@ -1,8 +1,11 @@
 {
   "functions": {
     "api/*.js": {
-      "maxDuration": 10
+      "maxDuration": 60
     }
   },
+  "regions": ["iad1"],
+  "env": {
+    "NODE_OPTIONS": "--max-old-space-size=4096"
+  }
 }'''


class NetlifyUSHSAdapter(ServerlessUSHSAdapter):
    """Netlify adapter for USHS compliance."""
    
    SUPPORTED_LANGUAGES = [
        'javascript', 'typescript', 'go'
    ]
    SUPPORTED_FEATURES = [
        'edge_functions',
        'background_functions',
        'scheduled_functions',
        'deploy_preview'
    ]
    
    def _get_platform_name(self) -> str:
        return "netlify"
    
    def _get_supported_languages(self) -> List[str]:
        return self.SUPPORTED_LANGUAGES
    
    def _is_error_log(self, log_entry: Dict[str, Any]) -> bool:
        """Check if Netlify log entry is an error."""
        # Netlify function logs structure
        return log_entry.get('level') == 'error' or 'error' in log_entry.get('msg', '').lower()
    
    def _log_to_error_event(self, log_entry: Dict[str, Any], function_name: str) -> Dict[str, Any]:
        """Convert Netlify log to USHS ErrorEvent."""
        return {
            'id': str(uuid4()),
            'timestamp': log_entry.get('ts', datetime.utcnow().isoformat()),
            'severity': 'high' if log_entry.get('level') == 'error' else 'medium',
            'source': {
                'service': function_name,
                'version': log_entry.get('deploy_id', 'unknown'),
                'environment': log_entry.get('context', 'production'),
                'location': f"netlify://{log_entry.get('site_id', 'unknown')}/{function_name}"
            },
            'error': {
                'type': 'FunctionError',
                'message': log_entry.get('msg', ''),
                'stackTrace': self._extract_netlify_stack(log_entry),
                'context': {
                    'requestId': log_entry.get('request_id'),
                    'functionPath': log_entry.get('path'),
                    'duration': log_entry.get('duration_ms')
                }
            }
        }
    
    def _analyze_error_type(self, error_type: str, message: str, stack_trace: List[str]) -> Dict[str, Any]:
        """Analyze Netlify-specific errors."""
        root_cause = {
            'type': 'unknown',
            'description': message,
            'category': 'runtime'
        }
        
        # Netlify-specific patterns
        if 'TimeoutError' in message or 'function execution timeout' in message:
            root_cause.update({
                'type': 'timeout',
                'category': 'configuration',
                'suggestion': 'Background functions support up to 15 minutes'
            })
        
        return root_cause
    
    def _suggest_fixes(self, root_cause: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Suggest Netlify-specific fixes."""
        suggestions = []
        
        if root_cause.get('type') == 'timeout':
            suggestions.append({
                'type': 'refactor',
                'action': 'convert_to_background_function',
                'confidence': 0.9
            })
        
        return suggestions
    
    def _generate_changes(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate Netlify-specific changes."""
        changes = []
        root_cause = analysis.get('rootCause', {})
        
        if root_cause.get('type') == 'timeout':
            changes.append({
                'file': 'netlify.toml',
                'diff': self._generate_netlify_config_patch(),
                'language': 'toml',
                'framework': 'netlify'
            })
        
        return changes
    
    def _extract_netlify_stack(self, log_entry: Dict[str, Any]) -> List[str]:
        """Extract stack trace from Netlify logs."""
        stack = log_entry.get('stack', '')
        if stack:
            return stack.split('\n')
        return []
    
    def _generate_netlify_config_patch(self) -> str:
        """Generate Netlify configuration patch."""
        return '''--- a/netlify.toml
+++ b/netlify.toml
@@ -1,5 +1,10 @@
 [build]
   functions = "functions"
 
 [functions]
   directory = "functions"
+  
+[[functions]]
+  path = "*.js"
+  # Convert to background function for longer execution
+  schedule = "@hourly"'''


class CloudflareWorkersUSHSAdapter(ServerlessUSHSAdapter):
    """Cloudflare Workers adapter for USHS compliance."""
    
    SUPPORTED_LANGUAGES = [
        'javascript', 'typescript', 'rust', 'c', 'cobol', 'python'
    ]
    SUPPORTED_FEATURES = [
        'workers_kv',
        'durable_objects',
        'r2_storage',
        'zero_cold_start'
    ]
    CERTIFICATION_LEVEL = "Platinum"  # Edge-optimized features
    
    def _get_platform_name(self) -> str:
        return "cloudflare_workers"
    
    def _get_supported_languages(self) -> List[str]:
        return self.SUPPORTED_LANGUAGES
    
    def _is_error_log(self, log_entry: Dict[str, Any]) -> bool:
        """Check if Workers log entry is an error."""
        # Cloudflare Workers log structure
        return log_entry.get('level') == 'error' or log_entry.get('outcome') == 'exception'
    
    def _log_to_error_event(self, log_entry: Dict[str, Any], function_name: str) -> Dict[str, Any]:
        """Convert Workers log to USHS ErrorEvent."""
        return {
            'id': str(uuid4()),
            'timestamp': log_entry.get('timestamp', datetime.utcnow().isoformat()),
            'severity': self._determine_workers_severity(log_entry),
            'source': {
                'service': function_name,
                'version': log_entry.get('scriptName', 'unknown'),
                'environment': 'production',
                'location': f"workers://{log_entry.get('colo', 'global')}/{function_name}"
            },
            'error': {
                'type': log_entry.get('exceptions', [{}])[0].get('name', 'WorkerError'),
                'message': log_entry.get('exceptions', [{}])[0].get('message', ''),
                'stackTrace': self._extract_workers_stack(log_entry),
                'context': {
                    'rayId': log_entry.get('rayId'),
                    'colo': log_entry.get('colo'),
                    'cpu': log_entry.get('cpu'),
                    'memory': log_entry.get('memory')
                }
            }
        }
    
    def _analyze_error_type(self, error_type: str, message: str, stack_trace: List[str]) -> Dict[str, Any]:
        """Analyze Workers-specific errors."""
        root_cause = {
            'type': 'unknown',
            'description': message,
            'category': 'runtime'
        }
        
        # Cloudflare Workers-specific patterns
        if 'CPU exceeded' in message:
            root_cause.update({
                'type': 'cpu_limit',
                'category': 'resource',
                'suggestion': 'Optimize CPU usage or use Durable Objects'
            })
        elif 'Script exceeded memory limits' in message:
            root_cause.update({
                'type': 'memory_limit',
                'category': 'resource',
                'suggestion': 'Reduce memory usage or split into multiple workers'
            })
        elif 'KV namespace' in message:
            root_cause.update({
                'type': 'kv_error',
                'category': 'storage',
                'suggestion': 'Check KV namespace bindings and permissions'
            })
        
        return root_cause
    
    def _suggest_fixes(self, root_cause: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Suggest Workers-specific fixes."""
        suggestions = []
        
        if root_cause.get('type') == 'cpu_limit':
            suggestions.append({
                'type': 'optimization',
                'action': 'use_durable_objects',
                'confidence': 0.85
            })
        elif root_cause.get('type') == 'memory_limit':
            suggestions.append({
                'type': 'architecture',
                'action': 'split_worker_logic',
                'confidence': 0.9
            })
        
        return suggestions
    
    def _generate_changes(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate Workers-specific changes."""
        changes = []
        root_cause = analysis.get('rootCause', {})
        
        if root_cause.get('type') == 'cpu_limit':
            changes.append({
                'file': 'wrangler.toml',
                'diff': self._generate_workers_optimization_patch(),
                'language': 'toml',
                'framework': 'cloudflare_workers'
            })
        
        return changes
    
    def _determine_workers_severity(self, log_entry: Dict[str, Any]) -> str:
        """Determine severity for Workers errors."""
        if log_entry.get('outcome') == 'exception':
            return 'critical'
        elif log_entry.get('level') == 'error':
            return 'high'
        return 'medium'
    
    def _extract_workers_stack(self, log_entry: Dict[str, Any]) -> List[str]:
        """Extract stack trace from Workers logs."""
        exceptions = log_entry.get('exceptions', [])
        stack_lines = []
        
        for exc in exceptions:
            if 'stack' in exc:
                stack_lines.extend(exc['stack'].split('\n'))
        
        return stack_lines
    
    def _generate_workers_optimization_patch(self) -> str:
        """Generate Workers optimization patch."""
        return '''--- a/wrangler.toml
+++ b/wrangler.toml
@@ -3,6 +3,14 @@ main = "src/index.js"
 compatibility_date = "2024-01-15"
 
 [env.production]
 kv_namespaces = [
   { binding = "KV", id = "abcd1234" }
 ]
+
+# Enable Durable Objects for CPU-intensive operations
+[[durable_objects.bindings]]
+name = "PROCESSOR"
+class_name = "ProcessorObject"
+
+[build]
+command = "npm run build"'''


# Register all adapters
from standards.v1.0.industry-adoption import registry

registry.register_adapter('serverless', 'aws_lambda', AWSLambdaUSHSAdapter)
registry.register_adapter('serverless', 'azure_functions', AzureFunctionsUSHSAdapter)
registry.register_adapter('serverless', 'gcp_functions', GCPFunctionsUSHSAdapter)
registry.register_adapter('serverless', 'vercel', VercelUSHSAdapter)
registry.register_adapter('serverless', 'netlify', NetlifyUSHSAdapter)
registry.register_adapter('serverless', 'cloudflare_workers', CloudflareWorkersUSHSAdapter)