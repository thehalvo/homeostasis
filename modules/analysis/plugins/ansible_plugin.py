"""
Ansible Language Plugin for Homeostasis

This plugin enables Homeostasis to analyze and fix errors in Ansible playbooks and configurations.
It provides comprehensive error handling for Ansible syntax errors, module issues,
variable problems, and configuration management best practices.
"""
import logging
import re
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union, Set

from ..language_plugin_system import LanguagePlugin, register_plugin

logger = logging.getLogger(__name__)


class AnsibleExceptionHandler:
    """
    Handles Ansible exceptions with robust error detection and classification.
    
    This class provides logic for categorizing Ansible errors based on their type,
    message, and common configuration management patterns.
    """
    
    def __init__(self):
        """Initialize the Ansible exception handler."""
        self.rule_categories = {
            "syntax": "YAML syntax and structure errors",
            "module": "Module usage and parameter errors",
            "variable": "Variable definition and templating errors",
            "inventory": "Inventory and host configuration errors",
            "task": "Task definition and execution errors",
            "role": "Role structure and dependency errors",
            "template": "Jinja2 templating errors",
            "connection": "SSH and connection errors",
            "privilege": "Privilege escalation errors",
            "handler": "Handler definition and notification errors",
            "vault": "Ansible Vault encryption errors",
            "galaxy": "Ansible Galaxy role/collection errors"
        }
        
        # Common Ansible error patterns
        self.ansible_error_patterns = {
            "syntax_error": [
                r"Syntax Error while loading YAML",
                r"could not find expected ':'",
                r"found character '\t' that cannot start any token",
                r"mapping values are not allowed here",
                r"expected <block end>, but found",
                r"found undefined tag handle"
            ],
            "module_error": [
                r"No module named",
                r"is not a legal parameter",
                r"Invalid/incorrect parameter",
                r"required together:",
                r"mutually exclusive:",
                r"one of the following is required:",
                r"module .* not found",
                r"couldn't resolve module/action"
            ],
            "variable_error": [
                r"'.*' is undefined",
                r"AnsibleUndefinedVariable:",
                r"'dict object' has no attribute",
                r"variable .* is not defined",
                r"undefined variable"
            ],
            "template_error": [
                r"template error while templating string",
                r"jinja2.*error",
                r"template syntax error",
                r"expected token.*got"
            ],
            "inventory_error": [
                r"Could not match supplied host pattern",
                r"provided hosts list is empty",
                r"Unable to retrieve inventory",
                r"parsing .*/inventory/ as an inventory source failed",
                r"inventory file but it was empty"
            ],
            "task_error": [
                r"The task includes an option with an undefined variable",
                r"conflicting action statements:",
                r"ERROR! no action detected in task",
                r"couldn't resolve module/action"
            ],
            "connection_error": [
                r"Failed to connect to the host",
                r"Connection timed out",
                r"Authentication failure",
                r"Permission denied",
                r"Host key verification failed",
                r"unreachable"
            ],
            "privilege_error": [
                r"sudo: sorry, you must have a tty",
                r"sudo: no tty present",
                r"is not in the sudoers file",
                r"incorrect password attempts"
            ],
            "template_error": [
                r"TemplateSyntaxError:",
                r"UndefinedError:",
                r"template error while templating string",
                r"unable to locate .* in expected paths"
            ]
        }
        
        # Common Ansible modules and their common issues
        self.module_issues = {
            "copy": ["src file not found", "dest permission denied", "backup failed"],
            "template": ["template not found", "variable undefined", "syntax error"],
            "file": ["path not found", "permission denied", "state conflict"],
            "service": ["service not found", "permission denied", "systemd not available"],
            "package": ["package not found", "repository not available", "permission denied"],
            "command": ["command not found", "permission denied", "working directory"],
            "shell": ["shell not available", "command failed", "environment issue"],
            "user": ["user exists", "permission denied", "group not found"],
            "group": ["group exists", "permission denied", "invalid gid"],
            "mount": ["device not found", "mount point not found", "permission denied"],
            "cron": ["crontab not available", "invalid syntax", "permission denied"],
            "lineinfile": ["file not found", "permission denied", "backup failed"],
            "replace": ["file not found", "pattern not found", "permission denied"],
            "uri": ["connection failed", "timeout", "authentication failed"],
            "get_url": ["URL not accessible", "permission denied", "timeout"],
            "unarchive": ["archive not found", "extraction failed", "permission denied"]
        }
        
        # Ansible-specific exit codes
        self.ansible_exit_codes = {
            0: "Success",
            1: "Generic error",
            2: "One or more hosts failed",
            3: "One or more hosts unreachable",
            4: "Parser error",
            5: "Bad or incomplete options",
            99: "User interrupted execution",
            250: "Unexpected error"
        }
        
        # Load rules from different categories
        self.rules = self._load_rules()
        
        # Pre-compile regex patterns for better performance
        self._compile_patterns()
    
    def _load_rules(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load Ansible error rules from rule files."""
        rules = {}
        rules_dir = Path(__file__).parent.parent / "rules" / "ansible"
        
        try:
            # Load common Ansible rules
            common_rules_path = rules_dir / "ansible_common_errors.json"
            if common_rules_path.exists():
                with open(common_rules_path, 'r') as f:
                    common_data = json.load(f)
                    rules["common"] = common_data.get("rules", [])
                    logger.info(f"Loaded {len(rules['common'])} common Ansible rules")
            
            # Load module-specific rules
            for module_name in ["core", "network", "cloud", "system", "files"]:
                module_rules_path = rules_dir / f"{module_name}_module_errors.json"
                if module_rules_path.exists():
                    with open(module_rules_path, 'r') as f:
                        module_data = json.load(f)
                        rules[module_name] = module_data.get("rules", [])
                        logger.info(f"Loaded {len(rules[module_name])} {module_name} module rules")
                        
        except Exception as e:
            logger.error(f"Error loading Ansible rules: {e}")
            rules = {"common": []}
        
        return rules
    
    def _compile_patterns(self):
        """Pre-compile regex patterns for better performance."""
        self.compiled_patterns = {}
        
        for category, rule_list in self.rules.items():
            self.compiled_patterns[category] = []
            for rule in rule_list:
                try:
                    pattern = rule.get("pattern", "")
                    if pattern:
                        compiled = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
                        self.compiled_patterns[category].append((compiled, rule))
                except re.error as e:
                    logger.warning(f"Invalid regex pattern in rule {rule.get('id', 'unknown')}: {e}")
    
    def analyze_exception(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze an Ansible exception and determine its type and potential fixes.
        
        Args:
            error_data: Ansible error data in standard format
            
        Returns:
            Analysis results with categorization and fix suggestions
        """
        error_type = error_data.get("error_type", "AnsibleError")
        message = error_data.get("message", "")
        task_name = error_data.get("task_name", "")
        module_name = error_data.get("module_name", "")
        playbook_path = error_data.get("playbook_path", "")
        exit_code = error_data.get("exit_code", 0)
        
        # Detect module if not provided
        if not module_name:
            module_name = self._detect_module(message, task_name)
        
        # Analyze based on error patterns
        analysis = self._analyze_by_patterns(message, module_name, task_name)
        
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
            analysis.update({
                "category": best_match.get("category", analysis.get("category", "unknown")),
                "subcategory": best_match.get("type", analysis.get("subcategory", "unknown")),
                "confidence": best_match.get("confidence", "medium"),
                "suggested_fix": best_match.get("suggestion", analysis.get("suggested_fix", "")),
                "root_cause": best_match.get("root_cause", analysis.get("root_cause", "")),
                "severity": best_match.get("severity", "medium"),
                "rule_id": best_match.get("id", ""),
                "tags": best_match.get("tags", []),
                "all_matches": matches
            })
        
        analysis["module_name"] = module_name
        analysis["task_name"] = task_name
        analysis["exit_code"] = exit_code
        return analysis
    
    def _detect_module(self, message: str, task_name: str) -> str:
        """Detect Ansible module from error message or task name."""
        # Check message for module indicators
        for module in self.module_issues.keys():
            if module in message.lower() or module in task_name.lower():
                return module
        
        # Check for common module patterns in messages
        module_patterns = {
            "copy": ["copying", "copied", "src", "dest"],
            "template": ["templating", "template", "jinja"],
            "service": ["systemd", "service", "daemon"],
            "package": ["yum", "apt", "dnf", "pip", "install"],
            "file": ["file", "directory", "path", "stat"],
            "command": ["command", "cmd", "execute"],
            "shell": ["shell", "bash", "/bin/sh"],
            "user": ["user", "useradd", "usermod"],
            "group": ["group", "groupadd"],
            "uri": ["http", "https", "url", "curl", "wget"],
            "get_url": ["download", "fetch", "url"]
        }
        
        message_lower = message.lower()
        for module, keywords in module_patterns.items():
            if any(keyword in message_lower for keyword in keywords):
                return module
        
        return "unknown"
    
    def _analyze_by_patterns(self, message: str, module_name: str, task_name: str) -> Dict[str, Any]:
        """Analyze error by matching against common patterns."""
        message_lower = message.lower()
        
        # Check syntax errors
        for pattern in self.ansible_error_patterns["syntax_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "ansible",
                    "subcategory": "syntax",
                    "confidence": "high",
                    "suggested_fix": "Fix YAML syntax errors in playbook",
                    "root_cause": "ansible_yaml_syntax_error",
                    "severity": "high",
                    "tags": ["ansible", "yaml", "syntax"]
                }
        
        # Check module errors
        for pattern in self.ansible_error_patterns["module_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "ansible",
                    "subcategory": "module",
                    "confidence": "high",
                    "suggested_fix": f"Fix module configuration for {module_name if module_name != 'unknown' else 'the module'}",
                    "root_cause": f"ansible_module_error_{module_name}",
                    "severity": "high",
                    "tags": ["ansible", "module", module_name]
                }
        
        # Check variable errors
        for pattern in self.ansible_error_patterns["variable_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "ansible",
                    "subcategory": "variable",
                    "confidence": "high",
                    "suggested_fix": "Define missing variables or fix variable references",
                    "root_cause": "ansible_variable_error",
                    "severity": "medium",
                    "tags": ["ansible", "variable", "template"]
                }
        
        # Check template errors
        for pattern in self.ansible_error_patterns["template_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "ansible",
                    "subcategory": "template",
                    "confidence": "high",
                    "suggested_fix": "Fix Jinja2 template syntax or variable references",
                    "root_cause": "ansible_template_error",
                    "severity": "medium",
                    "tags": ["ansible", "template", "jinja2"]
                }
        
        # Check inventory errors
        for pattern in self.ansible_error_patterns["inventory_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "ansible",
                    "subcategory": "inventory",
                    "confidence": "high",
                    "suggested_fix": "Fix inventory configuration and host patterns",
                    "root_cause": "ansible_inventory_error",
                    "severity": "high",
                    "tags": ["ansible", "inventory", "hosts"]
                }
        
        # Check task errors
        for pattern in self.ansible_error_patterns["task_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "ansible",
                    "subcategory": "task",
                    "confidence": "high",
                    "suggested_fix": "Fix task definition and action statements",
                    "root_cause": "ansible_task_error",
                    "severity": "high",
                    "tags": ["ansible", "task", "action"]
                }
        
        # Check connection errors
        for pattern in self.ansible_error_patterns["connection_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "ansible",
                    "subcategory": "connection",
                    "confidence": "high",
                    "suggested_fix": "Fix SSH connection and authentication issues",
                    "root_cause": "ansible_connection_error",
                    "severity": "high",
                    "tags": ["ansible", "connection", "ssh"]
                }
        
        # Check privilege errors
        for pattern in self.ansible_error_patterns["privilege_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "ansible",
                    "subcategory": "privilege",
                    "confidence": "high",
                    "suggested_fix": "Fix privilege escalation configuration",
                    "root_cause": "ansible_privilege_error",
                    "severity": "high",
                    "tags": ["ansible", "sudo", "privilege"]
                }
        
        # Check template errors
        for pattern in self.ansible_error_patterns["template_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "ansible",
                    "subcategory": "template",
                    "confidence": "high",
                    "suggested_fix": "Fix Jinja2 template syntax and variable references",
                    "root_cause": "ansible_template_error",
                    "severity": "medium",
                    "tags": ["ansible", "template", "jinja2"]
                }
        
        # Check module-specific issues
        if module_name in self.module_issues:
            module_analysis = self._analyze_module_specific_error(message, module_name)
            if module_analysis.get("confidence", "low") != "low":
                return module_analysis
        
        return {
            "category": "ansible",
            "subcategory": "unknown",
            "confidence": "low",
            "suggested_fix": "Review Ansible playbook configuration and error details",
            "root_cause": "ansible_generic_error",
            "severity": "medium",
            "tags": ["ansible", "generic"]
        }
    
    def _analyze_module_specific_error(self, message: str, module_name: str) -> Dict[str, Any]:
        """Analyze module-specific Ansible errors."""
        if module_name not in self.module_issues:
            return {"confidence": "low"}
        
        common_issues = self.module_issues[module_name]
        message_lower = message.lower()
        
        # Check for module-specific error patterns
        for issue in common_issues:
            if issue.lower() in message_lower:
                return {
                    "category": "ansible",
                    "subcategory": f"module_{module_name}",
                    "confidence": "high",
                    "suggested_fix": f"Fix {module_name} module issue: {issue}",
                    "root_cause": f"ansible_{module_name}_{issue.replace(' ', '_')}",
                    "severity": "medium",
                    "tags": ["ansible", "module", module_name]
                }
        
        return {"confidence": "low"}
    
    def _analyze_by_exit_code(self, exit_code: int, message: str) -> Dict[str, Any]:
        """Analyze error by Ansible exit code."""
        if exit_code in self.ansible_exit_codes:
            description = self.ansible_exit_codes[exit_code]
            
            if exit_code == 2:
                return {
                    "category": "ansible",
                    "subcategory": "host_failure",
                    "confidence": "high",
                    "suggested_fix": "Check host connectivity and task execution on failed hosts",
                    "root_cause": "ansible_host_failure",
                    "severity": "medium",
                    "tags": ["ansible", "host", "failure"],
                    "exit_code_description": description
                }
            elif exit_code == 3:
                return {
                    "category": "ansible",
                    "subcategory": "unreachable",
                    "confidence": "high",
                    "suggested_fix": "Fix host connectivity and SSH configuration",
                    "root_cause": "ansible_host_unreachable",
                    "severity": "high",
                    "tags": ["ansible", "unreachable", "connection"],
                    "exit_code_description": description
                }
            elif exit_code == 4:
                return {
                    "category": "ansible",
                    "subcategory": "parser",
                    "confidence": "high",
                    "suggested_fix": "Fix playbook syntax and YAML structure",
                    "root_cause": "ansible_parser_error",
                    "severity": "high",
                    "tags": ["ansible", "parser", "syntax"],
                    "exit_code_description": description
                }
            elif exit_code == 5:
                return {
                    "category": "ansible",
                    "subcategory": "options",
                    "confidence": "high",
                    "suggested_fix": "Check command line options and arguments",
                    "root_cause": "ansible_bad_options",
                    "severity": "medium",
                    "tags": ["ansible", "options", "cli"],
                    "exit_code_description": description
                }
            else:
                return {
                    "category": "ansible",
                    "subcategory": "unknown",
                    "confidence": "low",
                    "suggested_fix": f"Ansible failed with exit code {exit_code}: {description}",
                    "root_cause": f"ansible_exit_code_{exit_code}",
                    "severity": "medium",
                    "tags": ["ansible", "general"],
                    "exit_code_description": description
                }
        
        return {
            "category": "ansible",
            "subcategory": "unknown",
            "confidence": "low",
            "suggested_fix": f"Unknown exit code {exit_code}",
            "root_cause": f"ansible_unknown_exit_code_{exit_code}",
            "severity": "medium",
            "tags": ["ansible", "unknown"]
        }
    
    def _find_matching_rules(self, error_text: str, error_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find all rules that match the given error."""
        matches = []
        
        for category, patterns in self.compiled_patterns.items():
            for compiled_pattern, rule in patterns:
                match = compiled_pattern.search(error_text)
                if match:
                    # Calculate confidence score based on match quality
                    confidence_score = self._calculate_confidence(match, rule, error_data)
                    
                    match_info = rule.copy()
                    match_info["confidence_score"] = confidence_score
                    match_info["match_groups"] = match.groups() if match.groups() else []
                    matches.append(match_info)
        
        return matches
    
    def _calculate_confidence(self, match: re.Match, rule: Dict[str, Any], 
                             error_data: Dict[str, Any]) -> float:
        """Calculate confidence score for a rule match."""
        base_confidence = 0.5
        
        # Boost confidence for exact module matches
        rule_module = rule.get("module", "").lower()
        error_module = error_data.get("module_name", "").lower()
        if rule_module and error_module and rule_module == error_module:
            base_confidence += 0.2
        
        # Boost confidence based on rule reliability
        reliability = rule.get("reliability", "medium")
        reliability_boost = {"high": 0.2, "medium": 0.1, "low": 0.0}
        base_confidence += reliability_boost.get(reliability, 0.0)
        
        # Boost confidence for task name matches
        rule_tags = set(rule.get("tags", []))
        error_tags = set()
        
        task_name = error_data.get("task_name", "").lower()
        if "template" in task_name:
            error_tags.add("template")
        if "copy" in task_name:
            error_tags.add("copy")
        if "service" in task_name:
            error_tags.add("service")
        
        if error_tags & rule_tags:
            base_confidence += 0.1
        
        return min(base_confidence, 1.0)


class AnsiblePatchGenerator:
    """
    Generates patches for Ansible errors based on analysis results.
    
    This class creates Ansible playbook fixes for common errors using templates
    and heuristics specific to configuration management patterns.
    """
    
    def __init__(self):
        """Initialize the Ansible patch generator."""
        self.template_dir = Path(__file__).parent.parent / "patch_generation" / "templates"
        self.ansible_template_dir = self.template_dir / "ansible"
        
        # Ensure template directory exists
        self.ansible_template_dir.mkdir(parents=True, exist_ok=True)
        
        # Load patch templates
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, str]:
        """Load Ansible patch templates."""
        templates = {}
        
        if not self.ansible_template_dir.exists():
            logger.warning(f"Ansible templates directory not found: {self.ansible_template_dir}")
            return templates
        
        for template_file in self.ansible_template_dir.glob("*.yml.template"):
            try:
                with open(template_file, 'r') as f:
                    template_name = template_file.stem.replace('.yml', '')
                    templates[template_name] = f.read()
                    logger.debug(f"Loaded template: {template_name}")
            except Exception as e:
                logger.error(f"Error loading template {template_file}: {e}")
        
        return templates
    
    def generate_patch(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                      playbook_content: str = "") -> Optional[Dict[str, Any]]:
        """
        Generate a patch for the Ansible error.
        
        Args:
            error_data: The Ansible error data
            analysis: Analysis results from AnsibleExceptionHandler
            playbook_content: The playbook content that caused the error
            
        Returns:
            Patch information or None if no patch can be generated
        """
        root_cause = analysis.get("root_cause", "")
        module_name = analysis.get("module_name", "")
        
        # Map root causes to patch strategies
        patch_strategies = {
            "ansible_yaml_syntax_error": self._fix_yaml_syntax_error,
            "ansible_syntax_error": self._fix_yaml_syntax_error,  # Alias for compatibility
            "ansible_module_error": self._fix_module_error,
            "ansible_variable_error": self._fix_variable_error,
            "ansible_inventory_error": self._fix_inventory_error,
            "ansible_task_error": self._fix_task_error,
            "ansible_connection_error": self._fix_connection_error,
            "ansible_privilege_error": self._fix_privilege_error,
            "ansible_template_error": self._fix_template_error,
            "ansible_role_error": self._fix_role_error
        }
        
        # Try module-specific patches first
        if module_name != "unknown":
            module_strategy = patch_strategies.get(f"ansible_{module_name}_error")
            if module_strategy:
                try:
                    return module_strategy(error_data, analysis, playbook_content)
                except Exception as e:
                    logger.error(f"Error generating module-specific patch for {module_name}: {e}")
        
        # Try generic strategy
        strategy = patch_strategies.get(root_cause)
        if strategy:
            try:
                return strategy(error_data, analysis, playbook_content)
            except Exception as e:
                logger.error(f"Error generating patch for {root_cause}: {e}")
        
        # Try to use templates if no specific strategy matches
        return self._template_based_patch(error_data, analysis, playbook_content)
    
    def _fix_yaml_syntax_error(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                              playbook_content: str) -> Optional[Dict[str, Any]]:
        """Fix YAML syntax errors in Ansible playbooks."""
        message = error_data.get("message", "")
        
        fixes = []
        
        if "could not find expected ':'" in message.lower():
            fixes.append({
                "type": "suggestion",
                "description": "Missing colon in YAML mapping",
                "fix": "Add missing ':' after key names in YAML mappings"
            })
        
        if "found character '\\t'" in message.lower():
            fixes.append({
                "type": "suggestion",
                "description": "Tab characters not allowed in YAML",
                "fix": "Replace tabs with spaces (use 2 spaces for Ansible indentation)"
            })
        
        if "mapping values are not allowed here" in message.lower():
            fixes.append({
                "type": "suggestion",
                "description": "Invalid YAML mapping structure",
                "fix": "Check YAML indentation and mapping structure"
            })
        
        if "expected <block end>" in message.lower():
            fixes.append({
                "type": "suggestion",
                "description": "Missing block end or invalid indentation",
                "fix": "Check YAML block structure and indentation levels"
            })
        
        if fixes:
            return {
                "type": "multiple_suggestions",
                "fixes": fixes,
                "description": "Ansible YAML syntax error fixes"
            }
        
        return {
            "type": "suggestion",
            "description": "YAML syntax error in Ansible playbook. Use ansible-lint or yamllint to check syntax"
        }
    
    def _fix_module_error(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                         playbook_content: str) -> Optional[Dict[str, Any]]:
        """Fix Ansible module errors."""
        message = error_data.get("message", "")
        module_name = analysis.get("module_name", "")
        
        if "is not a legal parameter" in message.lower():
            # Extract parameter name
            param_match = re.search(r"'([^']+)' is not a legal parameter", message)
            param_name = param_match.group(1) if param_match else "parameter"
            
            return {
                "type": "suggestion",
                "description": f"Illegal parameter '{param_name}' for {module_name} module",
                "fixes": [
                    f"Remove '{param_name}' parameter from {module_name} module",
                    f"Check {module_name} module documentation for valid parameters",
                    f"Fix spelling of parameter name '{param_name}'",
                    "Verify module version compatibility with parameter"
                ]
            }
        
        if "required together:" in message.lower():
            return {
                "type": "suggestion",
                "description": "Missing required parameter combination",
                "fixes": [
                    "Add missing required parameters that must be used together",
                    "Check module documentation for parameter requirements",
                    "Ensure all interdependent parameters are specified"
                ]
            }
        
        if "mutually exclusive:" in message.lower():
            return {
                "type": "suggestion",
                "description": "Mutually exclusive parameters used",
                "fixes": [
                    "Remove one of the mutually exclusive parameters",
                    "Choose appropriate parameter for your use case",
                    "Check module documentation for parameter conflicts"
                ]
            }
        
        if "one of the following is required:" in message.lower():
            return {
                "type": "suggestion",
                "description": "Missing required parameter",
                "fixes": [
                    "Add one of the required parameters listed in the error",
                    "Check module documentation for parameter requirements",
                    "Specify appropriate parameter for your task"
                ]
            }
        
        if "module .* not found" in message.lower():
            return {
                "type": "suggestion",
                "description": f"Module '{module_name}' not found",
                "fixes": [
                    f"Check spelling of module name '{module_name}'",
                    "Install required Ansible collection containing the module",
                    "Use ansible-galaxy to install missing collections",
                    "Verify module is available in your Ansible version"
                ]
            }
        
        return {
            "type": "suggestion",
            "description": f"Module error for {module_name}",
            "fixes": [
                f"Check {module_name} module documentation",
                "Verify module parameters and syntax",
                "Ensure module is available and installed"
            ]
        }
    
    def _fix_variable_error(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                           playbook_content: str) -> Optional[Dict[str, Any]]:
        """Fix Ansible variable errors."""
        message = error_data.get("message", "")
        
        # Extract variable name
        var_match = re.search(r"'([^']+)' is undefined", message)
        if not var_match:
            var_match = re.search(r"variable ([^\s]+) is not defined", message)
        
        var_name = var_match.group(1) if var_match else "variable"
        
        if "'dict object' has no attribute" in message.lower():
            # Extract attribute name
            attr_match = re.search(r"has no attribute '([^']+)'", message)
            attr_name = attr_match.group(1) if attr_match else "attribute"
            
            return {
                "type": "suggestion",
                "description": f"Dictionary missing attribute '{attr_name}'",
                "fixes": [
                    f"Check if '{attr_name}' key exists in the dictionary",
                    f"Use default filter: variable.{attr_name} | default('default_value')",
                    f"Add '{attr_name}' key to the dictionary definition",
                    "Use 'when' condition to check if attribute exists"
                ]
            }
        
        return {
            "type": "suggestion",
            "description": f"Variable '{var_name}' is undefined",
            "fixes": [
                f"Define variable '{var_name}' in vars section or vars files",
                f"Set default value: {var_name} | default('default_value')",
                f"Pass variable via command line: -e '{var_name}=value'",
                f"Check variable name spelling: '{var_name}'",
                "Use 'when' condition to check if variable is defined"
            ]
        }
    
    def _fix_inventory_error(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                            playbook_content: str) -> Optional[Dict[str, Any]]:
        """Fix Ansible inventory errors."""
        message = error_data.get("message", "")
        
        if "could not match supplied host pattern" in message.lower():
            return {
                "type": "suggestion",
                "description": "Host pattern not found in inventory",
                "fixes": [
                    "Check host names in inventory file",
                    "Verify host patterns and group names",
                    "Use 'ansible-inventory --list' to check inventory",
                    "Check inventory file path and format",
                    "Add missing hosts to inventory"
                ]
            }
        
        if "provided hosts list is empty" in message.lower():
            return {
                "type": "suggestion",
                "description": "Empty hosts list",
                "fixes": [
                    "Add hosts to inventory file",
                    "Check inventory file exists and is readable",
                    "Verify inventory file format (INI or YAML)",
                    "Specify correct inventory path with -i option"
                ]
            }
        
        return {
            "type": "suggestion",
            "description": "Inventory configuration error",
            "fixes": [
                "Check inventory file format and syntax",
                "Verify host definitions and group structure",
                "Test inventory with ansible-inventory command"
            ]
        }
    
    def _fix_task_error(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                       playbook_content: str) -> Optional[Dict[str, Any]]:
        """Fix Ansible task errors."""
        message = error_data.get("message", "")
        
        if "no action detected in task" in message.lower():
            return {
                "type": "suggestion",
                "description": "Task missing action/module",
                "fixes": [
                    "Add module name and parameters to task",
                    "Check task structure and indentation",
                    "Ensure task has proper action specified",
                    "Remove empty tasks or add appropriate modules"
                ]
            }
        
        if "conflicting action statements" in message.lower():
            return {
                "type": "suggestion",
                "description": "Multiple actions in single task",
                "fixes": [
                    "Use only one action/module per task",
                    "Split task into multiple tasks",
                    "Remove duplicate or conflicting actions",
                    "Check task structure and syntax"
                ]
            }
        
        return {
            "type": "suggestion",
            "description": "Task definition error",
            "fixes": [
                "Check task structure and indentation",
                "Verify module name and parameters",
                "Ensure proper YAML syntax in tasks"
            ]
        }
    
    def _fix_connection_error(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                             playbook_content: str) -> Optional[Dict[str, Any]]:
        """Fix Ansible connection errors."""
        return {
            "type": "suggestion",
            "description": "SSH connection failed",
            "fixes": [
                "Check SSH connectivity: ssh user@host",
                "Verify SSH key authentication",
                "Check inventory host addresses and ports",
                "Configure SSH connection parameters",
                "Add host key to known_hosts file",
                "Check firewall and network connectivity"
            ]
        }
    
    def _fix_privilege_error(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                            playbook_content: str) -> Optional[Dict[str, Any]]:
        """Fix Ansible privilege escalation errors."""
        message = error_data.get("message", "")
        
        if "sudo: sorry, you must have a tty" in message.lower():
            return {
                "type": "suggestion",
                "description": "Sudo requires TTY",
                "fixes": [
                    "Add 'ansible_ssh_pipelining=false' to inventory",
                    "Configure sudoers with 'Defaults !requiretty'",
                    "Use 'ansible_become_flags=-tt' for force TTY",
                    "Configure SSH connection with pty: true"
                ]
            }
        
        if "is not in the sudoers file" in message.lower():
            return {
                "type": "suggestion",
                "description": "User not in sudoers",
                "fixes": [
                    "Add user to sudoers file or sudo group",
                    "Configure appropriate sudo permissions",
                    "Use different user with sudo privileges",
                    "Configure become_user and become_method"
                ]
            }
        
        return {
            "type": "suggestion",
            "description": "Privilege escalation error",
            "fixes": [
                "Check sudo/become configuration",
                "Verify user permissions and sudoers",
                "Configure become_method and become_user",
                "Test sudo access manually on target host"
            ]
        }
    
    def _fix_template_error(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                           playbook_content: str) -> Optional[Dict[str, Any]]:
        """Fix Jinja2 template errors."""
        message = error_data.get("message", "")
        
        if "TemplateSyntaxError" in message:
            return {
                "type": "suggestion",
                "description": "Jinja2 template syntax error",
                "fixes": [
                    "Check Jinja2 template syntax",
                    "Verify proper use of {{ }} and {% %}",
                    "Check filter and function syntax",
                    "Escape special characters if needed"
                ]
            }
        
        if "UndefinedError" in message:
            return {
                "type": "suggestion",
                "description": "Undefined variable in template",
                "fixes": [
                    "Define missing variables used in template",
                    "Use default filter for optional variables",
                    "Check variable name spelling in template",
                    "Add variable definitions to vars or defaults"
                ]
            }
        
        return {
            "type": "suggestion",
            "description": "Template processing error",
            "fixes": [
                "Check Jinja2 template syntax and variables",
                "Verify template file exists and is readable",
                "Test template rendering with ansible-template"
            ]
        }
    
    def _fix_role_error(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                       playbook_content: str) -> Optional[Dict[str, Any]]:
        """Fix role-related errors."""
        message = error_data.get("message", "")
        
        if "was not found" in message:
            # Extract role name
            import re
            match = re.search(r"role '([^']+)'", message)
            role_name = match.group(1) if match else "unknown"
            
            return {
                "type": "suggestion",
                "description": f"Role '{role_name}' not found",
                "fixes": [
                    f"Install the role with: ansible-galaxy install {role_name}",
                    "Check the roles_path configuration in ansible.cfg",
                    "Verify the role exists in the roles/ directory",
                    "Check for typos in the role name"
                ]
            }
        
        return {
            "type": "suggestion",
            "description": "Role-related error",
            "fixes": [
                "Verify role exists in roles/ directory or Galaxy",
                "Check role dependencies in meta/main.yml",
                "Use ansible-galaxy to install missing roles",
                "Verify role path configuration"
            ]
        }
    
    def _template_based_patch(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                            playbook_content: str) -> Optional[Dict[str, Any]]:
        """Generate patch using templates."""
        root_cause = analysis.get("root_cause", "")
        module_name = analysis.get("module_name", "")
        
        # Map root causes to template names
        template_map = {
            "ansible_yaml_syntax_error": "yaml_syntax_fix",
            "ansible_module_error": f"{module_name}_module_fix" if module_name != "unknown" else "module_fix",
            "ansible_variable_error": "variable_fix",
            "ansible_connection_error": "connection_fix"
        }
        
        template_name = template_map.get(root_cause)
        if template_name and template_name in self.templates:
            template = self.templates[template_name]
            
            return {
                "type": "template",
                "template": template,
                "description": f"Applied template fix for {root_cause}"
            }
        
        return None


class AnsibleLanguagePlugin(LanguagePlugin):
    """
    Main Ansible language plugin for Homeostasis.
    
    This plugin orchestrates Ansible error analysis and patch generation,
    supporting playbooks, roles, and configuration management patterns.
    """
    
    VERSION = "1.0.0"
    AUTHOR = "Homeostasis Team"
    
    def __init__(self):
        """Initialize the Ansible language plugin."""
        self.language = "ansible"
        self.supported_extensions = {".yml", ".yaml", ".ini"}
        self.supported_frameworks = [
            "ansible-core", "ansible", "ansible-playbook", "molecule", "ansible-lint",
            "ansible-galaxy", "ansible-vault", "ansible-runner"
        ]
        
        # Initialize components
        self.exception_handler = AnsibleExceptionHandler()
        self.patch_generator = AnsiblePatchGenerator()
        
        logger.info("Ansible language plugin initialized")
    
    def get_language_id(self) -> str:
        """Get the unique identifier for this language."""
        return "ansible"
    
    def get_language_name(self) -> str:
        """Get the human-readable name of the language."""
        return "Ansible"
    
    def get_language_version(self) -> str:
        """Get the version of the language supported by this plugin."""
        return "2.9+"
    
    def get_supported_frameworks(self) -> List[str]:
        """Get the list of frameworks supported by this language plugin."""
        return self.supported_frameworks
    
    def normalize_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize error data to the standard Homeostasis format.
        
        Args:
            error_data: Error data in the Ansible-specific format
            
        Returns:
            Error data in the standard format
        """
        # Map Ansible-specific error fields to standard format
        normalized = {
            "error_type": error_data.get("error_type", "AnsibleError"),
            "message": error_data.get("message", error_data.get("msg", "")),
            "language": "ansible",
            "command": error_data.get("command", ""),
            "file_path": error_data.get("file_path", error_data.get("file", "")),
            "task_name": error_data.get("task_name", error_data.get("task", "")),
            "module_name": error_data.get("module_name", error_data.get("module", "")),
            "playbook_path": error_data.get("playbook_path", error_data.get("playbook", "")),
            "role_name": error_data.get("role_name", error_data.get("role", "")),
            "line_number": error_data.get("line_number", error_data.get("line", 0)),
            "column_number": error_data.get("column_number", error_data.get("column", 0)),
            "exit_code": error_data.get("exit_code", error_data.get("rc", 0)),
            "host": error_data.get("host", ""),
            "inventory_path": error_data.get("inventory_path", ""),
            "stack_trace": error_data.get("stack_trace", []),
            "context": error_data.get("context", {}),
            "timestamp": error_data.get("timestamp"),
            "severity": error_data.get("severity", "medium")
        }
        
        # Add any additional fields from the original error
        for key, value in error_data.items():
            if key not in normalized and value is not None:
                normalized[key] = value
        
        return normalized
    
    def denormalize_error(self, standard_error: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert standard format error data back to the Ansible-specific format.
        
        Args:
            standard_error: Error data in the standard format
            
        Returns:
            Error data in the Ansible-specific format
        """
        # Map standard fields back to Ansible-specific format
        ansible_error = {
            "error_type": standard_error.get("error_type", "AnsibleError"),
            "message": standard_error.get("message", ""),
            "command": standard_error.get("command", ""),
            "task_name": standard_error.get("task_name", ""),
            "module_name": standard_error.get("module_name", ""),
            "playbook_path": standard_error.get("playbook_path", ""),
            "role_name": standard_error.get("role_name", ""),
            "line_number": standard_error.get("line_number", 0),
            "column_number": standard_error.get("column_number", 0),
            "file_path": standard_error.get("file_path", ""),
            "exit_code": standard_error.get("exit_code", 0),
            "host": standard_error.get("host", ""),
            "inventory_path": standard_error.get("inventory_path", ""),
            "msg": standard_error.get("message", ""),
            "task": standard_error.get("task_name", ""),
            "module": standard_error.get("module_name", ""),
            "playbook": standard_error.get("playbook_path", ""),
            "role": standard_error.get("role_name", ""),
            "file": standard_error.get("file_path", ""),
            "line": standard_error.get("line_number", 0),
            "column": standard_error.get("column_number", 0),
            "rc": standard_error.get("exit_code", 0),
            "stack_trace": standard_error.get("stack_trace", []),
            "context": standard_error.get("context", {}),
            "timestamp": standard_error.get("timestamp"),
            "severity": standard_error.get("severity", "medium")
        }
        
        # Add any additional fields from the standard error
        for key, value in standard_error.items():
            if key not in ansible_error and value is not None:
                ansible_error[key] = value
        
        return ansible_error
    
    def analyze_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze an Ansible error.
        
        Args:
            error_data: Ansible error data
            
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
            analysis["plugin"] = "ansible"
            analysis["language"] = "ansible"
            analysis["plugin_version"] = self.VERSION
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing Ansible error: {e}")
            return {
                "category": "ansible",
                "subcategory": "unknown",
                "confidence": "low",
                "suggested_fix": "Unable to analyze Ansible error",
                "error": str(e),
                "plugin": "ansible"
            }
    
    def generate_fix(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a fix for an error based on the analysis.
        
        Args:
            analysis: Error analysis
            context: Additional context for fix generation
            
        Returns:
            Generated fix data
        """
        error_data = context.get("error_data", {})
        playbook_content = context.get("playbook_content", context.get("source_code", ""))
        
        fix = self.patch_generator.generate_patch(error_data, analysis, playbook_content)
        
        if fix:
            return fix
        else:
            return {
                "type": "suggestion",
                "description": analysis.get("suggested_fix", "No specific fix available"),
                "confidence": analysis.get("confidence", "low")
            }


# Register the plugin
register_plugin(AnsibleLanguagePlugin())