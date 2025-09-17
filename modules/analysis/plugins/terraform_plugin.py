"""
Terraform Language Plugin for Homeostasis

This plugin enables Homeostasis to analyze and fix errors in Terraform configuration files.
It provides comprehensive error handling for Terraform syntax errors, resource configuration issues,
provider problems, and infrastructure-as-code best practices.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..language_plugin_system import LanguagePlugin, register_plugin

logger = logging.getLogger(__name__)


class TerraformExceptionHandler:
    """
    Handles Terraform exceptions with robust error detection and classification.

    This class provides logic for categorizing Terraform errors based on their type,
    message, and common infrastructure-as-code patterns.
    """

    def __init__(self):
        """Initialize the Terraform exception handler."""
        self.rule_categories = {
            "syntax": "Terraform HCL syntax errors",
            "resource": "Resource configuration errors",
            "provider": "Provider configuration and authentication errors",
            "variable": "Variable definition and usage errors",
            "module": "Module configuration and sourcing errors",
            "state": "State management and backend errors",
            "plan": "Plan generation and validation errors",
            "apply": "Apply operation errors",
            "dependency": "Resource dependency and lifecycle errors",
            "validation": "Input validation and constraint errors",
            "backend": "Backend configuration errors",
            "workspace": "Workspace management errors",
        }

        # Common Terraform error patterns
        self.terraform_error_patterns = {
            "syntax_error": [
                r"Invalid configuration syntax",
                r"Argument or block definition required",
                r"Missing required argument",
                r"Unsupported argument",
                r"Invalid function call",
                r"Invalid expression",
                r"Expected a closing delimiter",
                r"Missing value for required argument",
            ],
            "resource_error": [
                r"Error creating .+:",
                r"Error reading .+:",
                r"Error updating .+:",
                r"Error deleting .+:",
                r"Resource .+ does not exist",
                r"Invalid resource name",
                r"Invalid resource type",
                r"Duplicate resource",
            ],
            "provider_error": [
                r"Provider .+ not available",
                r"Error configuring provider",
                r"Invalid provider configuration",
                r"Provider authentication failed",
                r"Provider version constraint",
                r"Required provider .+ not specified",
                r"Failed to query available provider packages",
            ],
            "variable_error": [
                r"Variable .+ not defined",
                r"Invalid variable type",
                r"Variable validation failed",
                r"No value provided for required variable",
                r"Invalid default value for variable",
                r"Reference to undeclared variable",
            ],
            "module_error": [
                r"Module .+ not found",
                r"Error downloading module",
                r"Invalid module source",
                r"Module version constraint",
                r"Cyclic module dependency",
            ],
            "state_error": [
                r"Error acquiring (?:the )?state lock",
                r"Error reading state",
                r"Error writing state",
                r"State file corrupted",
                r"Backend initialization required",
            ],
            "dependency_error": [
                r"Cycle in dependencies",
                r"Resource depends on .+ which is not declared",
                r"Cannot determine order for resource",
                r"Dependency cycle detected",
            ],
            "validation_error": [
                r"Invalid value for variable",
                r"Validation failed",
                r"Value does not match constraint",
                r"Type constraint violation",
            ],
        }

        # Terraform command exit codes
        self.terraform_exit_codes = {
            0: "Success",
            1: "General error",
            2: "Plan differs (terraform plan)",
            3: "Configuration error",
            4: "Backend error",
            5: "State lock error",
        }

        # Common provider-specific error patterns
        self.provider_patterns = {
            "aws": {
                "authentication": [
                    r"NoCredentialsError",
                    r"InvalidUserID\.NotFound",
                    r"UnauthorizedOperation",
                    r"AccessDenied",
                ],
                "resources": [
                    r"InvalidInstanceID\.NotFound",
                    r"InvalidSubnetID\.NotFound",
                    r"InvalidVpcID\.NotFound",
                    r"InvalidSecurityGroupID\.NotFound",
                ],
            },
            "azurerm": {
                "authentication": [
                    r"AuthenticationFailed",
                    r"InvalidAuthenticationTokenTenant",
                    r"Forbidden",
                ],
                "resources": [
                    r"ResourceNotFound",
                    r"ResourceGroupNotFound",
                    r"SubscriptionNotFound",
                ],
            },
            "google": {
                "authentication": [
                    r"googleapi: Error 401",
                    r"googleapi: Error 403",
                    r"Application Default Credentials not available",
                ],
                "resources": [
                    r"googleapi: Error 404",
                    r"Resource .+ was not found",
                    r"Project .+ not found",
                ],
            },
        }

        # Load rules from different categories
        self.rules = self._load_rules()

        # Pre-compile regex patterns for better performance
        self._compile_patterns()

    def _load_rules(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load Terraform error rules from rule files."""
        rules = {}
        rules_dir = Path(__file__).parent.parent / "rules" / "terraform"

        try:
            # Load common Terraform rules
            common_rules_path = rules_dir / "terraform_common_errors.json"
            if common_rules_path.exists():
                with open(common_rules_path, "r") as f:
                    common_data = json.load(f)
                    rules["common"] = common_data.get("rules", [])
                    logger.info(f"Loaded {len(rules['common'])} common Terraform rules")

            # Load provider-specific rules
            for provider in ["aws", "azurerm", "google", "kubernetes", "helm"]:
                provider_rules_path = rules_dir / f"{provider}_errors.json"
                if provider_rules_path.exists():
                    with open(provider_rules_path, "r") as f:
                        provider_data = json.load(f)
                        rules[provider] = provider_data.get("rules", [])
                        logger.info(f"Loaded {len(rules[provider])} {provider} rules")

        except Exception as e:
            logger.error(f"Error loading Terraform rules: {e}")
            rules = {"common": []}

        return rules

    def _compile_patterns(self):
        """Pre-compile regex patterns for better performance."""
        self.compiled_patterns: Dict[
            str, List[tuple[re.Pattern[str], Dict[str, Any]]]
        ] = {}

        for category, rule_list in self.rules.items():
            self.compiled_patterns[category] = []
            for rule in rule_list:
                try:
                    pattern = rule.get("pattern", "")
                    if pattern:
                        compiled = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
                        self.compiled_patterns[category].append((compiled, rule))
                except re.error as e:
                    logger.warning(
                        f"Invalid regex pattern in rule {rule.get('id', 'unknown')}: {e}"
                    )

    def analyze_exception(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a Terraform exception and determine its type and potential fixes.

        Args:
            error_data: Terraform error data in standard format

        Returns:
            Analysis results with categorization and fix suggestions
        """
        message = error_data.get("message", "")
        command = error_data.get("command", "")
        exit_code = error_data.get("exit_code", 0)
        provider = error_data.get("provider", "")

        # Detect provider if not provided
        if not provider:
            provider = self._detect_provider(
                message, error_data.get("config_content", "")
            )

        # Analyze based on error patterns
        analysis = self._analyze_by_patterns(message, command, provider)

        # If no specific pattern matched, analyze by exit code
        if analysis.get("confidence", "low") == "low":
            exit_analysis = self._analyze_by_exit_code(exit_code, message)
            if exit_analysis.get("confidence", "low") != "low":
                analysis = exit_analysis

        # Find matching rules
        matches = self._find_matching_rules(message, error_data)

        if matches:
            # Use the best match (highest confidence)
            best_match = max(matches, key=lambda x: x.get("confidence_score", 0))
            # Map rule type to subcategory
            rule_type = best_match.get("type", "")
            subcategory = self._map_rule_type_to_subcategory(rule_type)
            if not subcategory:
                subcategory = analysis.get("subcategory", "unknown")

            analysis.update(
                {
                    "category": best_match.get(
                        "category", analysis.get("category", "unknown")
                    ),
                    "subcategory": subcategory,
                    "confidence": best_match.get("confidence", "medium"),
                    "suggested_fix": best_match.get(
                        "suggestion", analysis.get("suggested_fix", "")
                    ),
                    "root_cause": best_match.get(
                        "root_cause", analysis.get("root_cause", "")
                    ),
                    "severity": best_match.get("severity", "medium"),
                    "rule_id": best_match.get("id", ""),
                    "tags": best_match.get("tags", []),
                    "all_matches": matches,
                }
            )

        analysis["provider"] = provider
        analysis["command"] = command
        analysis["exit_code"] = exit_code
        return analysis

    def _map_rule_type_to_subcategory(self, rule_type: str) -> str:
        """Map rule type to expected subcategory."""
        type_mapping = {
            "ValidationError": "validation",
            "SyntaxError": "syntax",
            "ResourceError": "resource",
            "ProviderError": "provider",
            "VariableError": "variable",
            "StateError": "state",
            "ModuleError": "module",
            "DependencyError": "dependency",
            "BackendError": "backend",
            "WorkspaceError": "workspace",
            "ParameterError": "parameter",
            "AuthError": "provider_auth",
        }
        return type_mapping.get(rule_type, rule_type.lower() if rule_type else "")

    def _detect_provider(self, message: str, config_content: str) -> str:
        """Detect Terraform provider from error message or configuration."""
        message_lower = message.lower()
        config_lower = config_content.lower() if config_content else ""

        # Check message for provider indicators
        if any(
            indicator in message_lower
            for indicator in ["aws", "amazon", "s3", "ec2", "iam"]
        ):
            return "aws"
        elif any(
            indicator in message_lower
            for indicator in ["azure", "azurerm", "microsoft"]
        ):
            return "azurerm"
        elif any(
            indicator in message_lower for indicator in ["google", "gcp", "googleapis"]
        ):
            return "google"
        elif any(indicator in message_lower for indicator in ["kubernetes", "k8s"]):
            return "kubernetes"
        elif "helm" in message_lower:
            return "helm"

        # Check configuration content for provider blocks
        provider_matches = re.findall(r'provider\s+"([^"]+)"', config_lower)
        if provider_matches:
            return str(provider_matches[0])

        return "unknown"

    def _analyze_by_patterns(
        self, message: str, command: str, provider: str
    ) -> Dict[str, Any]:
        """Analyze error by matching against common patterns."""
        # Check syntax errors
        for pattern in self.terraform_error_patterns["syntax_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "terraform",
                    "subcategory": "syntax",
                    "confidence": "high",
                    "suggested_fix": "Fix Terraform HCL syntax errors",
                    "root_cause": "terraform_syntax_error",
                    "severity": "high",
                    "tags": ["terraform", "syntax", "hcl"],
                }

        # Check resource errors
        for pattern in self.terraform_error_patterns["resource_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "terraform",
                    "subcategory": "resource",
                    "confidence": "high",
                    "suggested_fix": "Fix resource configuration or check resource state",
                    "root_cause": "terraform_resource_error",
                    "severity": "high",
                    "tags": ["terraform", "resource"],
                }

        # Check provider errors
        for pattern in self.terraform_error_patterns["provider_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "terraform",
                    "subcategory": "provider",
                    "confidence": "high",
                    "suggested_fix": "Fix provider configuration and authentication",
                    "root_cause": "terraform_provider_error",
                    "severity": "high",
                    "tags": ["terraform", "provider", provider],
                }

        # Check variable errors
        for pattern in self.terraform_error_patterns["variable_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "terraform",
                    "subcategory": "variable",
                    "confidence": "high",
                    "suggested_fix": "Fix variable definition or provide required values",
                    "root_cause": "terraform_variable_error",
                    "severity": "medium",
                    "tags": ["terraform", "variable"],
                }

        # Check module errors
        for pattern in self.terraform_error_patterns["module_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "terraform",
                    "subcategory": "module",
                    "confidence": "high",
                    "suggested_fix": "Fix module configuration or check module source",
                    "root_cause": "terraform_module_error",
                    "severity": "medium",
                    "tags": ["terraform", "module"],
                }

        # Check state errors
        for pattern in self.terraform_error_patterns["state_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "terraform",
                    "subcategory": "state",
                    "confidence": "high",
                    "suggested_fix": "Fix state management or backend configuration",
                    "root_cause": "terraform_state_error",
                    "severity": "high",
                    "tags": ["terraform", "state", "backend"],
                }

        # Check dependency errors
        for pattern in self.terraform_error_patterns["dependency_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "terraform",
                    "subcategory": "dependency",
                    "confidence": "high",
                    "suggested_fix": "Fix resource dependencies and ordering",
                    "root_cause": "terraform_dependency_error",
                    "severity": "medium",
                    "tags": ["terraform", "dependency", "cycle"],
                }

        # Check validation errors
        for pattern in self.terraform_error_patterns["validation_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "terraform",
                    "subcategory": "validation",
                    "confidence": "high",
                    "suggested_fix": "Fix variable values or validation constraints",
                    "root_cause": "terraform_validation_error",
                    "severity": "medium",
                    "tags": ["terraform", "validation"],
                }

        # Check provider-specific errors
        if provider in self.provider_patterns:
            provider_analysis = self._analyze_provider_specific_error(message, provider)
            if provider_analysis.get("confidence", "low") != "low":
                return provider_analysis

        return {
            "category": "terraform",
            "subcategory": "unknown",
            "confidence": "low",
            "suggested_fix": "Review Terraform configuration and error details",
            "root_cause": "terraform_generic_error",
            "severity": "medium",
            "tags": ["terraform", "generic"],
        }

    def _analyze_provider_specific_error(
        self, message: str, provider: str
    ) -> Dict[str, Any]:
        """Analyze provider-specific Terraform errors."""
        if provider not in self.provider_patterns:
            return {"confidence": "low"}

        provider_config = self.provider_patterns[provider]

        # Check authentication errors
        for pattern in provider_config.get("authentication", []):
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "terraform",
                    "subcategory": "provider_auth",
                    "confidence": "high",
                    "suggested_fix": f"Fix {provider} provider authentication and credentials",
                    "root_cause": f"terraform_{provider}_auth_error",
                    "severity": "high",
                    "tags": ["terraform", "provider", provider, "authentication"],
                }

        # Check resource errors
        for pattern in provider_config.get("resources", []):
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "terraform",
                    "subcategory": "provider_resource",
                    "confidence": "high",
                    "suggested_fix": f"Fix {provider} resource configuration or check resource existence",
                    "root_cause": f"terraform_{provider}_resource_error",
                    "severity": "medium",
                    "tags": ["terraform", "provider", provider, "resource"],
                }

        return {"confidence": "low"}

    def _analyze_by_exit_code(self, exit_code: int, message: str) -> Dict[str, Any]:
        """Analyze error by Terraform exit code."""
        if exit_code in self.terraform_exit_codes:
            description = self.terraform_exit_codes[exit_code]

            if exit_code == 1:
                return {
                    "category": "terraform",
                    "subcategory": "general",
                    "confidence": "medium",
                    "suggested_fix": "Review Terraform error details and configuration",
                    "root_cause": "terraform_general_error",
                    "severity": "medium",
                    "tags": ["terraform", "general"],
                    "exit_code_description": description,
                }
            elif exit_code == 2:
                return {
                    "category": "terraform",
                    "subcategory": "plan",
                    "confidence": "high",
                    "suggested_fix": "Plan shows differences - review and apply changes if intended",
                    "root_cause": "terraform_plan_diff",
                    "severity": "low",
                    "tags": ["terraform", "plan", "diff"],
                    "exit_code_description": description,
                }
            elif exit_code == 3:
                return {
                    "category": "terraform",
                    "subcategory": "configuration",
                    "confidence": "high",
                    "suggested_fix": "Fix Terraform configuration errors",
                    "root_cause": "terraform_configuration_error",
                    "severity": "high",
                    "tags": ["terraform", "configuration"],
                    "exit_code_description": description,
                }
            elif exit_code == 4:
                return {
                    "category": "terraform",
                    "subcategory": "backend",
                    "confidence": "high",
                    "suggested_fix": "Fix backend configuration or initialization",
                    "root_cause": "terraform_backend_error",
                    "severity": "high",
                    "tags": ["terraform", "backend"],
                    "exit_code_description": description,
                }
            elif exit_code == 5:
                return {
                    "category": "terraform",
                    "subcategory": "state_lock",
                    "confidence": "high",
                    "suggested_fix": "Resolve state lock conflict or force unlock if safe",
                    "root_cause": "terraform_state_lock_error",
                    "severity": "high",
                    "tags": ["terraform", "state", "lock"],
                    "exit_code_description": description,
                }

        return {
            "category": "terraform",
            "subcategory": "unknown",
            "confidence": "low",
            "suggested_fix": f"Unknown exit code {exit_code}",
            "root_cause": f"terraform_unknown_exit_code_{exit_code}",
            "severity": "medium",
            "tags": ["terraform", "unknown"],
        }

    def _find_matching_rules(
        self, error_text: str, error_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Find all rules that match the given error."""
        matches = []

        for category, patterns in self.compiled_patterns.items():
            for compiled_pattern, rule in patterns:
                match = compiled_pattern.search(error_text)
                if match:
                    # Calculate confidence score based on match quality
                    confidence_score = self._calculate_confidence(
                        match, rule, error_data
                    )

                    match_info = rule.copy()
                    match_info["confidence_score"] = confidence_score
                    match_info["match_groups"] = (
                        match.groups() if match.groups() else []
                    )
                    matches.append(match_info)

        return matches

    def _calculate_confidence(
        self, match: re.Match, rule: Dict[str, Any], error_data: Dict[str, Any]
    ) -> float:
        """Calculate confidence score for a rule match."""
        base_confidence = 0.5

        # Boost confidence for exact provider matches
        rule_provider = rule.get("provider", "").lower()
        error_provider = error_data.get("provider", "").lower()
        if rule_provider and error_provider and rule_provider == error_provider:
            base_confidence += 0.2

        # Boost confidence based on rule reliability
        reliability = rule.get("reliability", "medium")
        reliability_boost = {"high": 0.2, "medium": 0.1, "low": 0.0}
        base_confidence += reliability_boost.get(reliability, 0.0)

        # Boost confidence for command matches
        rule_command = rule.get("command", "").lower()
        error_command = error_data.get("command", "").lower()
        if rule_command and error_command and rule_command in error_command:
            base_confidence += 0.1

        return min(base_confidence, 1.0)


class TerraformPatchGenerator:
    """
    Generates patches for Terraform errors based on analysis results.

    This class creates Terraform configuration fixes for common errors using templates
    and heuristics specific to infrastructure-as-code patterns.
    """

    def __init__(self):
        """Initialize the Terraform patch generator."""
        self.template_dir = (
            Path(__file__).parent.parent / "patch_generation" / "templates"
        )
        self.terraform_template_dir = self.template_dir / "terraform"

        # Ensure template directory exists
        self.terraform_template_dir.mkdir(parents=True, exist_ok=True)

        # Load patch templates
        self.templates: Dict[str, str] = self._load_templates()

    def _load_templates(self) -> Dict[str, str]:
        """Load Terraform patch templates."""
        templates: Dict[str, str] = {}

        if not self.terraform_template_dir.exists():
            logger.warning(
                f"Terraform templates directory not found: {self.terraform_template_dir}"
            )
            return templates

        for template_file in self.terraform_template_dir.glob("*.tf.template"):
            try:
                with open(template_file, "r") as f:
                    template_name = template_file.stem.replace(".tf", "")
                    templates[template_name] = f.read()
                    logger.debug(f"Loaded template: {template_name}")
            except Exception as e:
                logger.error(f"Error loading template {template_file}: {e}")

        return templates

    def generate_patch(
        self,
        error_data: Dict[str, Any],
        analysis: Dict[str, Any],
        config_content: str = "",
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a patch for the Terraform error.

        Args:
            error_data: The Terraform error data
            analysis: Analysis results from TerraformExceptionHandler
            config_content: The Terraform configuration content that caused the error

        Returns:
            Patch information or None if no patch can be generated
        """
        root_cause = analysis.get("root_cause", "")
        provider = analysis.get("provider", "")

        # Map root causes to patch strategies
        patch_strategies = {
            "terraform_syntax_error": self._fix_syntax_error,
            "terraform_resource_error": self._fix_resource_error,
            "terraform_provider_error": self._fix_provider_error,
            "terraform_variable_error": self._fix_variable_error,
            "terraform_module_error": self._fix_module_error,
            "terraform_state_error": self._fix_state_error,
            "terraform_dependency_error": self._fix_dependency_error,
            "terraform_backend_error": self._fix_backend_error,
            "terraform_state_lock_error": self._fix_state_lock_error,
        }

        # Try provider-specific patches first
        if provider != "unknown":
            provider_strategy = patch_strategies.get(f"{root_cause}_{provider}")
            if provider_strategy:
                try:
                    return provider_strategy(error_data, analysis, config_content)
                except Exception as e:
                    logger.error(
                        f"Error generating provider-specific patch for {root_cause}: {e}"
                    )

        # Try generic strategy
        strategy = patch_strategies.get(root_cause)
        if strategy:
            try:
                return strategy(error_data, analysis, config_content)
            except Exception as e:
                logger.error(f"Error generating patch for {root_cause}: {e}")

        # Try to use templates if no specific strategy matches
        return self._template_based_patch(error_data, analysis, config_content)

    def _fix_syntax_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], config_content: str
    ) -> Optional[Dict[str, Any]]:
        """Fix Terraform HCL syntax errors."""
        message = error_data.get("message", "")

        fixes = []

        if "missing required argument" in message.lower():
            # Extract argument name
            arg_match = re.search(
                r'argument "([^"]+)" is required', message, re.IGNORECASE
            )
            if arg_match:
                arg_name = arg_match.group(1)
                fixes.append(
                    {
                        "type": "suggestion",
                        "description": f"Add missing required argument '{arg_name}'",
                        "fix": f"Add the {arg_name} argument to the resource or data block",
                    }
                )

        if "unsupported argument" in message.lower():
            # Extract argument name
            arg_match = re.search(
                r'argument "([^"]+)" is not expected', message, re.IGNORECASE
            )
            if arg_match:
                arg_name = arg_match.group(1)
                fixes.append(
                    {
                        "type": "suggestion",
                        "description": f"Remove unsupported argument '{arg_name}'",
                        "fix": f"Remove the {arg_name} argument or check for typos",
                    }
                )

        if "invalid function call" in message.lower():
            fixes.append(
                {
                    "type": "suggestion",
                    "description": "Invalid function call",
                    "fix": "Check function name spelling and argument syntax",
                }
            )

        if "expected a closing delimiter" in message.lower():
            fixes.append(
                {
                    "type": "suggestion",
                    "description": "Missing closing delimiter",
                    "fix": "Add missing closing brace, bracket, or parenthesis",
                }
            )

        if fixes:
            return {
                "type": "multiple_suggestions",
                "fixes": fixes,
                "description": "Terraform HCL syntax error fixes",
            }

        return {
            "type": "suggestion",
            "description": "Terraform syntax error. Use 'terraform validate' to check configuration",
        }

    def _fix_resource_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], config_content: str
    ) -> Optional[Dict[str, Any]]:
        """Fix Terraform resource configuration errors."""
        message = error_data.get("message", "")
        provider = analysis.get("provider", "")

        # Extract resource type from error message
        resource_match = re.search(
            r"Error (?:creating|reading|updating|deleting) ([^:]+):", message
        )
        resource_type = resource_match.group(1) if resource_match else "resource"

        fixes = [
            f"Check {resource_type} configuration for required parameters",
            f"Verify {resource_type} exists and is accessible",
            f"Check {provider} provider permissions for {resource_type}",
            "Review resource dependencies and references",
        ]

        if "does not exist" in message.lower():
            fixes.insert(
                0,
                f"Create the {resource_type} or update references to existing resources",
            )

        if "duplicate resource" in message.lower():
            fixes.insert(0, "Remove duplicate resource definition or use unique names")

        return {
            "type": "suggestion",
            "description": f"Resource error: {resource_type}",
            "fixes": fixes,
        }

    def _fix_provider_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], config_content: str
    ) -> Optional[Dict[str, Any]]:
        """Fix Terraform provider configuration errors."""
        provider = analysis.get("provider", "")

        if provider == "aws":
            fixes = [
                "Configure AWS credentials using AWS CLI, environment variables, or IAM roles",
                "Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables",
                "Configure AWS profile: aws configure",
                "Check AWS region configuration",
                "Verify IAM permissions for required AWS services",
            ]
        elif provider == "azurerm":
            fixes = [
                "Configure Azure authentication using Azure CLI or service principal",
                "Run 'az login' to authenticate with Azure",
                "Set ARM_CLIENT_ID, ARM_CLIENT_SECRET, ARM_SUBSCRIPTION_ID, ARM_TENANT_ID",
                "Check Azure subscription and resource group access",
            ]
        elif provider == "google":
            fixes = [
                "Configure Google Cloud authentication using gcloud CLI or service account",
                "Run 'gcloud auth application-default login'",
                "Set GOOGLE_APPLICATION_CREDENTIALS environment variable",
                "Check Google Cloud project and service account permissions",
            ]
        else:
            fixes = [
                f"Configure {provider} provider authentication",
                f"Check {provider} provider version constraints",
                f"Verify {provider} provider configuration block",
                f"Review {provider} provider documentation",
            ]

        return {
            "type": "suggestion",
            "description": f"Provider error: {provider}",
            "fixes": fixes,
        }

    def _fix_variable_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], config_content: str
    ) -> Optional[Dict[str, Any]]:
        """Fix Terraform variable errors."""
        message = error_data.get("message", "")

        # Extract variable name
        var_match = re.search(r'variable "([^"]+)"', message, re.IGNORECASE)
        var_name = var_match.group(1) if var_match else "variable"

        if "not defined" in message.lower():
            return {
                "type": "suggestion",
                "description": f"Variable '{var_name}' not defined",
                "fixes": [
                    f"Define variable '{var_name}' in a .tf file",
                    f'Create variable block: variable "{var_name}" {{ type = string }}',
                    "Provide value in terraform.tfvars or use -var flag",
                    "Check variable name spelling",
                ],
            }

        if "no value provided" in message.lower():
            return {
                "type": "suggestion",
                "description": f"No value provided for variable '{var_name}'",
                "fixes": [
                    f'Set value in terraform.tfvars: {var_name} = "value"',
                    f"Use command line: terraform apply -var='{var_name}=value'",
                    "Add default value to variable definition",
                    "Set environment variable TF_VAR_" + var_name,
                ],
            }

        if "validation failed" in message.lower():
            return {
                "type": "suggestion",
                "description": f"Variable '{var_name}' validation failed",
                "fixes": [
                    "Check variable value against validation rules",
                    "Review variable validation block conditions",
                    "Update variable value to meet validation criteria",
                ],
            }

        return {
            "type": "suggestion",
            "description": "Variable error",
            "fixes": [
                "Check variable definitions and usage",
                "Verify variable values and types",
                "Review variable validation rules",
            ],
        }

    def _fix_module_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], config_content: str
    ) -> Optional[Dict[str, Any]]:
        """Fix Terraform module errors."""
        message = error_data.get("message", "")

        if "not found" in message.lower():
            return {
                "type": "suggestion",
                "description": "Module not found",
                "fixes": [
                    "Check module source path or URL",
                    "Run 'terraform init' to download modules",
                    "Verify module source accessibility",
                    "Check module version constraints",
                ],
            }

        if "cyclic module dependency" in message.lower():
            return {
                "type": "suggestion",
                "description": "Cyclic module dependency",
                "fixes": [
                    "Remove circular dependencies between modules",
                    "Restructure module hierarchy",
                    "Use data sources instead of module outputs where appropriate",
                ],
            }

        return {
            "type": "suggestion",
            "description": "Module error",
            "fixes": [
                "Run 'terraform init' to initialize modules",
                "Check module source and version",
                "Verify module input variables",
            ],
        }

    def _fix_state_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], config_content: str
    ) -> Optional[Dict[str, Any]]:
        """Fix Terraform state management errors."""
        return {
            "type": "suggestion",
            "description": "State management error",
            "fixes": [
                "Run 'terraform init' to initialize backend",
                "Check backend configuration",
                "Verify state file accessibility and permissions",
                "Check state lock status",
                "Consider state backup and recovery options",
            ],
        }

    def _fix_dependency_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], config_content: str
    ) -> Optional[Dict[str, Any]]:
        """Fix Terraform resource dependency errors."""
        return {
            "type": "suggestion",
            "description": "Resource dependency error",
            "fixes": [
                "Review resource dependencies and remove cycles",
                "Use explicit depends_on when needed",
                "Check implicit dependencies through resource references",
                "Consider breaking down complex dependencies",
            ],
        }

    def _fix_backend_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], config_content: str
    ) -> Optional[Dict[str, Any]]:
        """Fix Terraform backend configuration errors."""
        return {
            "type": "suggestion",
            "description": "Backend configuration error",
            "fixes": [
                "Run 'terraform init' to initialize backend",
                "Check backend configuration in terraform block",
                "Verify backend credentials and permissions",
                "Check backend service availability",
            ],
        }

    def _fix_state_lock_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], config_content: str
    ) -> Optional[Dict[str, Any]]:
        """Fix Terraform state lock errors."""
        return {
            "type": "suggestion",
            "description": "State lock error",
            "fixes": [
                "Wait for other Terraform operations to complete",
                "Check if lock is stale and force unlock if safe: terraform force-unlock <lock-id>",
                "Verify no other Terraform processes are running",
                "Check backend service status and connectivity",
            ],
        }

    def _template_based_patch(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], config_content: str
    ) -> Optional[Dict[str, Any]]:
        """Generate patch using templates."""
        root_cause = analysis.get("root_cause", "")
        provider = analysis.get("provider", "")

        # Map root causes to template names
        template_map = {
            "terraform_syntax_error": "syntax_fix",
            "terraform_resource_error": "resource_fix",
            "terraform_provider_error": (
                f"{provider}_provider_fix" if provider != "unknown" else "provider_fix"
            ),
            "terraform_variable_error": "variable_fix",
        }

        template_name = template_map.get(root_cause)
        if template_name and template_name in self.templates:
            template = self.templates[template_name]

            return {
                "type": "template",
                "template": template,
                "description": f"Applied template fix for {root_cause}",
            }

        return None


class TerraformLanguagePlugin(LanguagePlugin):
    """
    Main Terraform language plugin for Homeostasis.

    This plugin orchestrates Terraform error analysis and patch generation,
    supporting multiple providers and infrastructure-as-code patterns.
    """

    VERSION = "1.0.0"
    AUTHOR = "Homeostasis Team"

    def __init__(self):
        """Initialize the Terraform language plugin."""
        self.language = "terraform"
        self.supported_extensions = {".tf", ".tfvars", ".hcl"}
        self.supported_frameworks = [
            "terraform",
            "aws",
            "azurerm",
            "azure",
            "google",
            "gcp",
            "kubernetes",
            "helm",
            "vault",
            "consul",
            "nomad",
            "random",
            "local",
            "external",
            "http",
            "tls",
            "archive",
        ]

        # Initialize components
        self.exception_handler = TerraformExceptionHandler()
        self.patch_generator = TerraformPatchGenerator()

        logger.info("Terraform language plugin initialized")

    def get_language_id(self) -> str:
        """Get the unique identifier for this language."""
        return "terraform"

    def get_language_name(self) -> str:
        """Get the human-readable name of the language."""
        return "Terraform"

    def get_language_version(self) -> str:
        """Get the version of the language supported by this plugin."""
        return "1.0+"

    def get_supported_frameworks(self) -> List[str]:
        """Get the list of frameworks supported by this language plugin."""
        return self.supported_frameworks

    def normalize_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize error data to the standard Homeostasis format.

        Args:
            error_data: Error data in the Terraform-specific format

        Returns:
            Error data in the standard format
        """
        # Map Terraform-specific error fields to standard format
        normalized = {
            "error_type": error_data.get("error_type", "TerraformError"),
            "message": error_data.get("message", error_data.get("description", "")),
            "language": "terraform",
            "file_path": error_data.get("file_path", error_data.get("file", "")),
            "line_number": error_data.get("line_number", error_data.get("line", 0)),
            "column_number": error_data.get(
                "column_number", error_data.get("column", 0)
            ),
            "command": error_data.get("command", ""),
            "provider": error_data.get("provider", ""),
            "exit_code": error_data.get("exit_code", error_data.get("returncode", 0)),
            "config_file": error_data.get("config_file", error_data.get("file", "")),
            "config_content": error_data.get("config_content", ""),
            "stack_trace": error_data.get("stack_trace", []),
            "context": error_data.get("context", {}),
            "timestamp": error_data.get("timestamp"),
            "severity": error_data.get("severity", "medium"),
        }

        # Add any additional fields from the original error
        for key, value in error_data.items():
            if key not in normalized and value is not None:
                normalized[key] = value

        return normalized

    def denormalize_error(self, standard_error: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert standard format error data back to the Terraform-specific format.

        Args:
            standard_error: Error data in the standard format

        Returns:
            Error data in the Terraform-specific format
        """
        # Map standard fields back to Terraform-specific format
        terraform_error = {
            "error_type": standard_error.get("error_type", "TerraformError"),
            "message": standard_error.get("message", ""),
            "file_path": standard_error.get("file_path", ""),
            "line_number": standard_error.get("line_number", 0),
            "column_number": standard_error.get("column_number", 0),
            "command": standard_error.get("command", ""),
            "provider": standard_error.get("provider", ""),
            "exit_code": standard_error.get("exit_code", 0),
            "config_file": standard_error.get(
                "config_file", standard_error.get("file_path", "")
            ),
            "config_content": standard_error.get("config_content", ""),
            "description": standard_error.get("message", ""),
            "file": standard_error.get("file_path", ""),
            "line": standard_error.get("line_number", 0),
            "column": standard_error.get("column_number", 0),
            "returncode": standard_error.get("exit_code", 0),
            "stack_trace": standard_error.get("stack_trace", []),
            "context": standard_error.get("context", {}),
            "timestamp": standard_error.get("timestamp"),
            "severity": standard_error.get("severity", "medium"),
        }

        # Add any additional fields from the standard error
        for key, value in standard_error.items():
            if key not in terraform_error and value is not None:
                terraform_error[key] = value

        return terraform_error

    def analyze_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a Terraform error.

        Args:
            error_data: Terraform error data

        Returns:
            Analysis results
        """
        try:
            # Ensure error data is in standard format
            if not error_data.get("language"):
                standard_error = self.normalize_error(error_data)
            else:
                standard_error = error_data

            # Analyze the error
            analysis = self.exception_handler.analyze_exception(standard_error)

            # Add plugin metadata
            analysis["plugin"] = "terraform"
            analysis["language"] = "terraform"
            analysis["plugin_version"] = self.VERSION

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing Terraform error: {e}")
            return {
                "category": "terraform",
                "subcategory": "unknown",
                "confidence": "low",
                "suggested_fix": "Unable to analyze Terraform error",
                "error": str(e),
                "plugin": "terraform",
            }

    def generate_fix(
        self, analysis: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a fix for an error based on the analysis.

        Args:
            analysis: Error analysis
            context: Additional context for fix generation

        Returns:
            Generated fix data
        """
        error_data = context.get("error_data", {})
        config_content = context.get("config_content", context.get("source_code", ""))

        fix = self.patch_generator.generate_patch(error_data, analysis, config_content)

        if fix:
            return fix
        else:
            return {
                "type": "suggestion",
                "description": analysis.get(
                    "suggested_fix", "No specific fix available"
                ),
                "confidence": analysis.get("confidence", "low"),
            }


# Register the plugin
register_plugin(TerraformLanguagePlugin())
