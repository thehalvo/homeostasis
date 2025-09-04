"""
YAML/JSON Configuration Language Plugin for Homeostasis

This plugin enables Homeostasis to analyze and fix errors in YAML and JSON configuration files.
It provides comprehensive error handling for configuration syntax errors, schema validation,
and common configuration issues across various tools and frameworks.
"""
import logging
import re
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

from ..language_plugin_system import LanguagePlugin, register_plugin

logger = logging.getLogger(__name__)


class YAMLJSONExceptionHandler:
    """
    Handles YAML/JSON configuration exceptions with robust error detection and classification.
    
    This class provides logic for categorizing configuration file errors based on their type,
    message, and common configuration patterns across different tools and frameworks.
    """
    
    def __init__(self):
        """Initialize the YAML/JSON exception handler."""
        self.rule_categories = {
            "syntax": "YAML/JSON syntax errors",
            "schema": "Schema validation errors",
            "structure": "Document structure errors",
            "encoding": "Character encoding errors",
            "indentation": "YAML indentation errors",
            "type": "Data type errors",
            "reference": "Reference and anchor errors",
            "validation": "Content validation errors",
            "format": "Format-specific errors",
            "security": "Security-related configuration issues"
        }
        
        # Common configuration file patterns and their error types
        self.config_frameworks = {
            "kubernetes": {
                "patterns": ["apiVersion", "kind", "metadata", "spec"],
                "common_errors": ["missing required fields", "invalid apiVersion", "malformed selector"]
            },
            "docker": {
                "patterns": ["version", "services", "volumes", "networks"],
                "common_errors": ["invalid version", "missing service definition", "invalid port mapping"]
            },
            "ansible": {
                "patterns": ["hosts", "tasks", "vars", "handlers"],
                "common_errors": ["undefined variable", "invalid module", "malformed task"]
            },
            "github_actions": {
                "patterns": ["name", "on", "jobs", "steps"],
                "common_errors": ["invalid trigger", "missing step action", "invalid runner"]
            },
            "gitlab_ci": {
                "patterns": ["stages", "variables", "before_script", "script"],
                "common_errors": ["invalid stage", "undefined variable", "missing script"]
            },
            "terraform": {
                "patterns": ["terraform", "provider", "resource", "variable"],
                "common_errors": ["invalid provider", "missing required argument", "circular dependency"]
            },
            "eslint": {
                "patterns": ["extends", "rules", "env", "parser"],
                "common_errors": ["unknown rule", "invalid config value", "missing extends"]
            }
        }
        
        # YAML-specific error patterns
        self.yaml_error_patterns = {
            "indentation": [
                r"found character '\t' that cannot start any token",
                r"could not find expected ':'",
                r"mapping values are not allowed here",
                r"expected <block end>, but found"
            ],
            "syntax": [
                r"while parsing a block mapping",
                r"could not find expected ':'",
                r"found unexpected end of stream",
                r"expected ',' or '}' in flow mapping",
                r"found unknown escape character",
                r"found undefined tag handle"
            ],
            "structure": [
                r"found duplicate key",
                r"found duplicate anchor",
                r"found undefined alias",
                r"recursive objects are not allowed"
            ]
        }
        
        # JSON-specific error patterns
        self.json_error_patterns = {
            "syntax": [
                r"Expecting ',' delimiter",
                r"Expecting ':' delimiter",
                r"Expecting property name enclosed in double quotes",
                r"Unterminated string starting at",
                r"Invalid control character",
                r"Extra data"
            ],
            "structure": [
                r"Duplicate keys not allowed",
                r"Object must be str, bytes or os.PathLike object",
                r"JSON object must be str, bytes or bytearray"
            ]
        }
        
        # Load rules from different categories
        self.rules = self._load_rules()
        
        # Pre-compile regex patterns for better performance
        self._compile_patterns()
    
    def _load_rules(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load YAML/JSON error rules from rule files."""
        rules = {}
        rules_dir = Path(__file__).parent.parent / "rules" / "yaml_json"
        
        try:
            # Load common YAML/JSON rules
            common_rules_path = rules_dir / "yaml_json_common_errors.json"
            if common_rules_path.exists():
                with open(common_rules_path, 'r') as f:
                    common_data = json.load(f)
                    rules["common"] = common_data.get("rules", [])
                    logger.info(f"Loaded {len(rules['common'])} common YAML/JSON rules")
            
            # Load framework-specific rules
            for framework in self.config_frameworks.keys():
                framework_rules_path = rules_dir / f"{framework}_errors.json"
                if framework_rules_path.exists():
                    with open(framework_rules_path, 'r') as f:
                        framework_data = json.load(f)
                        rules[framework] = framework_data.get("rules", [])
                        logger.info(f"Loaded {len(rules[framework])} {framework} rules")
                        
        except Exception as e:
            logger.error(f"Error loading YAML/JSON rules: {e}")
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
        Analyze a YAML/JSON configuration exception and determine its type and potential fixes.
        
        Args:
            error_data: Configuration error data in standard format
            
        Returns:
            Analysis results with categorization and fix suggestions
        """
        error_type = error_data.get("error_type", "ConfigError")
        message = error_data.get("message", "")
        file_type = error_data.get("file_type", "").lower()
        file_path = error_data.get("file_path", "")
        content = error_data.get("content", "")
        
        # Handle specific error types
        if error_type == "SchemaError":
            return {
                "category": "yaml_json",
                "subcategory": "schema",
                "confidence": "high",
                "suggested_fix": "Fix schema validation error - ensure all required properties are present and correctly typed",
                "root_cause": "schema_error",
                "severity": "high",
                "tags": ["schema", "validation"]
            }
        elif error_type == "TypeError" and ("expected" in message.lower() or "got" in message.lower()):
            return {
                "category": "yaml_json",
                "subcategory": "type",
                "confidence": "high",
                "suggested_fix": "Fix type mismatch - ensure values match the expected types",
                "root_cause": "type_error",
                "severity": "medium",
                "tags": ["type", "validation"]
            }
        elif error_type == "YAMLError":
            # Check for specific YAML error patterns
            if "inconsistent indentation" in message.lower():
                return {
                    "category": "yaml_json",
                    "subcategory": "indentation",
                    "confidence": "high",
                    "suggested_fix": "Fix YAML indentation - use consistent spacing (2 or 4 spaces)",
                    "root_cause": "indentation_error",
                    "severity": "high",
                    "tags": ["indentation", "yaml"]
                }
            elif "found duplicate key" in message.lower():
                return {
                    "category": "yaml_json",
                    "subcategory": "duplicate_key",
                    "confidence": "high",
                    "suggested_fix": "Remove or rename duplicate keys in YAML document",
                    "root_cause": "duplicate_key_error",
                    "severity": "high",
                    "tags": ["duplicate", "yaml", "structure"]
                }
        
        # Detect file type if not provided
        if not file_type:
            file_type = self._detect_file_type(file_path, content)
        
        # Detect configuration framework if possible
        framework = self._detect_framework(content, file_path)
        
        # Analyze based on file type
        if file_type == "yaml":
            analysis = self._analyze_yaml_error(message, content, framework)
        elif file_type == "json":
            analysis = self._analyze_json_error(message, content, framework)
        else:
            analysis = self._analyze_generic_config_error(message, content, framework)
        
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
        
        analysis["file_type"] = file_type
        analysis["framework"] = framework
        return analysis
    
    def _detect_file_type(self, file_path: str, content: str) -> str:
        """Detect whether the file is YAML or JSON."""
        if file_path:
            file_path_lower = file_path.lower()
            if file_path_lower.endswith(('.yaml', '.yml')):
                return "yaml"
            elif file_path_lower.endswith('.json'):
                return "json"
        
        # Try to detect from content
        if content:
            content_stripped = content.strip()
            if content_stripped.startswith(('{', '[')):
                return "json"
            elif ':' in content and not content_stripped.startswith(('{', '[')):
                return "yaml"
        
        return "unknown"
    
    def _detect_framework(self, content: str, file_path: str) -> str:
        """Detect the configuration framework from content or file path."""
        file_path_lower = file_path.lower() if file_path else ""
        
        # Check file path patterns
        if "docker-compose" in file_path_lower or "compose" in file_path_lower:
            return "docker"
        elif "kubernetes" in file_path_lower or "k8s" in file_path_lower:
            return "kubernetes"
        elif ".github/workflows" in file_path_lower:
            return "github_actions"
        elif ".gitlab-ci" in file_path_lower:
            return "gitlab_ci"
        elif "ansible" in file_path_lower or "playbook" in file_path_lower:
            return "ansible"
        elif ".terraform" in file_path_lower or "terraform" in file_path_lower:
            return "terraform"
        elif "eslint" in file_path_lower:
            return "eslint"
        
        # Check content patterns
        if content:
            content_lower = content.lower()
            for framework, config in self.config_frameworks.items():
                pattern_matches = sum(1 for pattern in config["patterns"] if pattern in content_lower)
                if pattern_matches >= 2:  # At least 2 patterns must match
                    return framework
        
        return "generic"
    
    def _analyze_yaml_error(self, message: str, content: str, framework: str) -> Dict[str, Any]:
        """Analyze YAML-specific errors."""
        # Check for indentation errors
        for pattern in self.yaml_error_patterns["indentation"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "yaml_json",
                    "subcategory": "indentation",
                    "confidence": "high",
                    "suggested_fix": "Fix YAML indentation - use spaces instead of tabs and ensure consistent indentation",
                    "root_cause": "yaml_indentation_error",
                    "severity": "high",
                    "tags": ["yaml", "indentation", "syntax"]
                }
        
        # Check for syntax errors
        for pattern in self.yaml_error_patterns["syntax"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "yaml_json",
                    "subcategory": "yaml_syntax",
                    "confidence": "high",
                    "suggested_fix": "Fix YAML syntax errors - check for missing colons, quotes, or invalid characters",
                    "root_cause": "yaml_syntax_error",
                    "severity": "high",
                    "tags": ["yaml", "syntax"]
                }
        
        # Check for structure errors
        for pattern in self.yaml_error_patterns["structure"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "yaml_json",
                    "subcategory": "structure",
                    "confidence": "high",
                    "suggested_fix": "Fix YAML structure issues - check for duplicate keys or undefined references",
                    "root_cause": "yaml_structure_error",
                    "severity": "medium",
                    "tags": ["yaml", "structure"]
                }
        
        # Framework-specific analysis
        if framework in self.config_frameworks:
            return self._analyze_framework_specific_error(message, content, framework, "yaml")
        
        return {
            "category": "yaml_json",
            "subcategory": "unknown",
            "confidence": "low",
            "suggested_fix": "Review YAML syntax and structure",
            "root_cause": "yaml_generic_error",
            "severity": "medium",
            "tags": ["yaml", "generic"]
        }
    
    def _analyze_json_error(self, message: str, content: str, framework: str) -> Dict[str, Any]:
        """Analyze JSON-specific errors."""
        message_lower = message.lower()
        
        # Check for syntax errors
        for pattern in self.json_error_patterns["syntax"]:
            if re.search(pattern, message, re.IGNORECASE):
                if "expecting ',' delimiter" in message_lower:
                    return {
                        "category": "yaml_json",
                        "subcategory": "json_syntax",
                        "confidence": "high",
                        "suggested_fix": "Add missing comma between JSON object properties or array elements",
                        "root_cause": "json_missing_comma",
                        "severity": "high",
                        "tags": ["json", "syntax", "comma"]
                    }
                elif "expecting ':' delimiter" in message_lower:
                    return {
                        "category": "yaml_json",
                        "subcategory": "json_syntax",
                        "confidence": "high",
                        "suggested_fix": "Add missing colon between JSON property name and value",
                        "root_cause": "json_missing_colon",
                        "severity": "high",
                        "tags": ["json", "syntax", "colon"]
                    }
                elif "expecting property name" in message_lower:
                    return {
                        "category": "yaml_json",
                        "subcategory": "json_syntax",
                        "confidence": "high",
                        "suggested_fix": "Property names in JSON must be enclosed in double quotes",
                        "root_cause": "json_unquoted_property",
                        "severity": "high",
                        "tags": ["json", "syntax", "quotes"]
                    }
                elif "unterminated string" in message_lower:
                    return {
                        "category": "yaml_json",
                        "subcategory": "json_syntax",
                        "confidence": "high",
                        "suggested_fix": "Add missing closing quote for JSON string value",
                        "root_cause": "json_unterminated_string",
                        "severity": "high",
                        "tags": ["json", "syntax", "string"]
                    }
                else:
                    return {
                        "category": "yaml_json",
                        "subcategory": "json_syntax",
                        "confidence": "high",
                        "suggested_fix": "Fix JSON syntax errors - check for proper quotes, commas, and brackets",
                        "root_cause": "json_syntax_error",
                        "severity": "high",
                        "tags": ["json", "syntax"]
                    }
        
        # Check for structure errors
        for pattern in self.json_error_patterns["structure"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "yaml_json",
                    "subcategory": "structure",
                    "confidence": "high",
                    "suggested_fix": "Fix JSON structure issues - check for duplicate keys or invalid object types",
                    "root_cause": "json_structure_error",
                    "severity": "medium",
                    "tags": ["json", "structure"]
                }
        
        # Framework-specific analysis
        if framework in self.config_frameworks:
            return self._analyze_framework_specific_error(message, content, framework, "json")
        
        return {
            "category": "yaml_json",
            "subcategory": "unknown",
            "confidence": "low",
            "suggested_fix": "Review JSON syntax and structure",
            "root_cause": "json_generic_error",
            "severity": "medium",
            "tags": ["json", "generic"]
        }
    
    def _analyze_framework_specific_error(self, message: str, content: str, framework: str, file_type: str) -> Dict[str, Any]:
        """Analyze framework-specific configuration errors."""
        framework_config = self.config_frameworks.get(framework, {})
        common_errors = framework_config.get("common_errors", [])
        
        message_lower = message.lower()
        
        # Check for framework-specific error patterns
        for error_pattern in common_errors:
            if error_pattern.lower() in message_lower:
                return {
                    "category": "yaml_json",
                    "subcategory": framework,
                    "confidence": "high",
                    "suggested_fix": f"Fix {framework} configuration: {error_pattern}",
                    "root_cause": f"{file_type}_{framework}_{error_pattern.replace(' ', '_')}",
                    "severity": "medium",
                    "tags": [file_type, framework, "configuration"]
                }
        
        # Framework-specific fixes
        framework_fixes = {
            "kubernetes": "Check Kubernetes resource definition - verify apiVersion, kind, and required fields",
            "docker": "Check Docker Compose configuration - verify version, services, and port mappings",
            "ansible": "Check Ansible playbook - verify task definitions, variables, and module usage",
            "github_actions": "Check GitHub Actions workflow - verify triggers, jobs, and step definitions",
            "gitlab_ci": "Check GitLab CI configuration - verify stages, scripts, and variable definitions",
            "terraform": "Check Terraform configuration - verify provider, resource, and variable definitions",
            "eslint": "Check ESLint configuration - verify rules, extends, and parser settings"
        }
        
        return {
            "category": "yaml_json",
            "subcategory": framework,
            "confidence": "medium",
            "suggested_fix": framework_fixes.get(framework, f"Check {framework} configuration syntax and structure"),
            "root_cause": f"{file_type}_{framework}_configuration_error",
            "severity": "medium",
            "tags": [file_type, framework, "configuration"]
        }
    
    def _analyze_generic_config_error(self, message: str, content: str, framework: str) -> Dict[str, Any]:
        """Analyze generic configuration errors."""
        return {
            "category": "yaml_json",
            "subcategory": "unknown",
            "confidence": "low",
            "suggested_fix": "Review configuration file syntax and structure",
            "root_cause": "config_generic_error",
            "severity": "medium",
            "tags": ["config", "generic"]
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
        
        # Boost confidence for exact file type matches
        rule_file_type = rule.get("file_type", "").lower()
        error_file_type = error_data.get("file_type", "").lower()
        if rule_file_type and rule_file_type == error_file_type:
            base_confidence += 0.2
        
        # Boost confidence based on rule reliability
        reliability = rule.get("reliability", "medium")
        reliability_boost = {"high": 0.2, "medium": 0.1, "low": 0.0}
        base_confidence += reliability_boost.get(reliability, 0.0)
        
        # Boost confidence for framework matches
        rule_framework = rule.get("framework", "").lower()
        error_framework = error_data.get("framework", "").lower()
        if rule_framework and error_framework and rule_framework == error_framework:
            base_confidence += 0.2
        
        return min(base_confidence, 1.0)


class YAMLJSONPatchGenerator:
    """
    Generates patches for YAML/JSON configuration errors based on analysis results.
    
    This class creates configuration fixes for common errors using templates
    and heuristics specific to configuration file patterns.
    """
    
    def __init__(self):
        """Initialize the YAML/JSON patch generator."""
        self.template_dir = Path(__file__).parent.parent / "patch_generation" / "templates"
        self.yaml_json_template_dir = self.template_dir / "yaml_json"
        
        # Ensure template directory exists
        self.yaml_json_template_dir.mkdir(parents=True, exist_ok=True)
        
        # Load patch templates
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, str]:
        """Load YAML/JSON patch templates."""
        templates = {}
        
        if not self.yaml_json_template_dir.exists():
            logger.warning(f"YAML/JSON templates directory not found: {self.yaml_json_template_dir}")
            return templates
        
        for template_file in self.yaml_json_template_dir.glob("*.template"):
            try:
                with open(template_file, 'r') as f:
                    template_name = template_file.stem
                    templates[template_name] = f.read()
                    logger.debug(f"Loaded template: {template_name}")
            except Exception as e:
                logger.error(f"Error loading template {template_file}: {e}")
        
        return templates
    
    def generate_patch(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                      content: str = "") -> Optional[Dict[str, Any]]:
        """
        Generate a patch for the YAML/JSON error.
        
        Args:
            error_data: The configuration error data
            analysis: Analysis results from YAMLJSONExceptionHandler
            content: The configuration file content that caused the error
            
        Returns:
            Patch information or None if no patch can be generated
        """
        root_cause = analysis.get("root_cause", "")
        file_type = analysis.get("file_type", "")
        
        # Map root causes to patch strategies
        patch_strategies = {
            "yaml_indentation_error": self._fix_yaml_indentation,
            "yaml_syntax_error": self._fix_yaml_syntax,
            "yaml_structure_error": self._fix_yaml_structure,
            "json_syntax_error": self._fix_json_syntax,
            "json_missing_comma": self._fix_json_missing_comma,
            "json_missing_colon": self._fix_json_missing_colon,
            "json_unquoted_property": self._fix_json_unquoted_property,
            "json_unterminated_string": self._fix_json_unterminated_string,
            "json_structure_error": self._fix_json_structure,
            "schema_error": self._fix_schema_error,
            "indentation_error": self._fix_yaml_indentation,
            "type_error": self._fix_type_error
        }
        
        strategy = patch_strategies.get(root_cause)
        if strategy:
            try:
                return strategy(error_data, analysis, content)
            except Exception as e:
                logger.error(f"Error generating patch for {root_cause}: {e}")
        
        # Try framework-specific fixes
        if framework != "generic":
            framework_fix = self._fix_framework_specific(error_data, analysis, content)
            if framework_fix:
                return framework_fix
        
        # Try to use templates if no specific strategy matches
        return self._template_based_patch(error_data, analysis, content)
    
    def _fix_yaml_indentation(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                             content: str) -> Optional[Dict[str, Any]]:
        """Fix YAML indentation errors."""
        return {
            "type": "suggestion",
            "description": "YAML indentation error",
            "fixes": [
                "Use spaces instead of tabs for indentation",
                "Ensure consistent indentation (usually 2 or 4 spaces)",
                "Check that child elements are properly indented",
                "Verify list items use proper indentation with '-'",
                "Use a YAML linter or formatter to check indentation"
            ]
        }
    
    def _fix_yaml_syntax(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                        content: str) -> Optional[Dict[str, Any]]:
        """Fix YAML syntax errors."""
        message = error_data.get("message", "")
        
        if "could not find expected ':'" in message.lower():
            return {
                "type": "suggestion",
                "description": "Missing colon in YAML key-value pair",
                "fixes": [
                    "Add missing ':' after property name",
                    "Ensure there's a space after the colon: 'key: value'",
                    "Check for typos in property names"
                ]
            }
        
        if "found unknown escape character" in message.lower():
            return {
                "type": "suggestion",
                "description": "Invalid escape character in YAML string",
                "fixes": [
                    "Use proper YAML string escaping",
                    "Quote strings containing special characters",
                    "Use literal scalar style (|) or folded style (>) for complex strings"
                ]
            }
        
        return {
            "type": "suggestion",
            "description": "YAML syntax error",
            "fixes": [
                "Check YAML syntax with a validator",
                "Ensure proper quoting of strings",
                "Verify bracket and brace matching",
                "Check for invalid characters or escape sequences"
            ]
        }
    
    def _fix_yaml_structure(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                           content: str) -> Optional[Dict[str, Any]]:
        """Fix YAML structure errors."""
        message = error_data.get("message", "")
        
        if "found duplicate key" in message.lower():
            return {
                "type": "suggestion",
                "description": "Duplicate key in YAML document",
                "fixes": [
                    "Remove or rename duplicate keys",
                    "Merge values if keys should be combined",
                    "Check for copy-paste errors"
                ]
            }
        
        if "found undefined alias" in message.lower():
            return {
                "type": "suggestion",
                "description": "Undefined alias reference in YAML",
                "fixes": [
                    "Define the anchor before using the alias",
                    "Check spelling of anchor and alias names",
                    "Remove unused aliases"
                ]
            }
        
        return {
            "type": "suggestion",
            "description": "YAML structure error",
            "fixes": [
                "Check document structure and organization",
                "Verify anchor and alias definitions",
                "Ensure no duplicate keys exist"
            ]
        }
    
    def _fix_json_syntax(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                        content: str) -> Optional[Dict[str, Any]]:
        """Fix JSON syntax errors."""
        return {
            "type": "suggestion",
            "description": "JSON syntax error",
            "fixes": [
                "Check JSON syntax with a validator",
                "Ensure all strings are enclosed in double quotes",
                "Verify proper comma placement between elements",
                "Check bracket and brace matching"
            ]
        }
    
    def _fix_json_missing_comma(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                               content: str) -> Optional[Dict[str, Any]]:
        """Fix missing comma in JSON."""
        return {
            "type": "suggestion",
            "description": "Missing comma in JSON",
            "fixes": [
                "Add comma between object properties",
                "Add comma between array elements",
                "Remove trailing comma if it's the last element",
                "Check for proper JSON formatting"
            ]
        }
    
    def _fix_json_missing_colon(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                               content: str) -> Optional[Dict[str, Any]]:
        """Fix missing colon in JSON."""
        return {
            "type": "suggestion",
            "description": "Missing colon in JSON object property",
            "fixes": [
                "Add ':' between property name and value",
                "Ensure property names are followed by colons",
                "Check for typos in property definitions"
            ]
        }
    
    def _fix_json_unquoted_property(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                                   content: str) -> Optional[Dict[str, Any]]:
        """Fix unquoted property name in JSON."""
        return {
            "type": "suggestion",
            "description": "Property name must be quoted in JSON",
            "fixes": [
                "Enclose property names in double quotes",
                "Change single quotes to double quotes",
                "Ensure all object keys are properly quoted"
            ]
        }
    
    def _fix_json_unterminated_string(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                                     content: str) -> Optional[Dict[str, Any]]:
        """Fix unterminated string in JSON."""
        return {
            "type": "suggestion", 
            "description": "Unterminated string in JSON",
            "fixes": [
                "Add missing closing quote for string value",
                "Check for unescaped quotes within strings",
                "Escape special characters in strings"
            ]
        }
    
    def _fix_json_structure(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                           content: str) -> Optional[Dict[str, Any]]:
        """Fix JSON structure errors."""
        return {
            "type": "suggestion",
            "description": "JSON structure error", 
            "fixes": [
                "Check for duplicate object keys",
                "Verify proper nesting of objects and arrays",
                "Ensure valid JSON data types are used"
            ]
        }
    
    def _fix_schema_error(self, error_data: Dict[str, Any], analysis: Dict[str, Any],
                          content: str) -> Optional[Dict[str, Any]]:
        """Fix schema validation errors."""
        return {
            "type": "suggestion",
            "description": "Schema validation error - ensure required properties are present",
            "fixes": [
                "Add missing required properties to the configuration",
                "Ensure property types match schema requirements",
                "Validate against the schema documentation",
                "Check for additional required nested properties"
            ]
        }
    
    def _fix_type_error(self, error_data: Dict[str, Any], analysis: Dict[str, Any],
                        content: str) -> Optional[Dict[str, Any]]:
        """Fix type mismatch errors."""
        return {
            "type": "suggestion",
            "description": "Type mismatch error - ensure values have correct types",
            "fixes": [
                "Convert values to the expected type",
                "Check if strings should be numbers or booleans",
                "Verify array vs object type requirements",
                "Ensure proper quoting for string values"
            ]
        }
    
    def _fix_framework_specific(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                               content: str) -> Optional[Dict[str, Any]]:
        """Fix framework-specific configuration errors."""
        framework = analysis.get("framework", "")
        
        framework_fixes = {
            "kubernetes": {
                "type": "suggestion",
                "description": "Kubernetes configuration error",
                "fixes": [
                    "Verify apiVersion is correct for the resource type",
                    "Ensure required fields (kind, metadata, spec) are present",
                    "Check resource naming and labeling conventions",
                    "Validate selector and template labels match"
                ]
            },
            "docker": {
                "type": "suggestion",
                "description": "Docker Compose configuration error",
                "fixes": [
                    "Check version compatibility with Docker Compose",
                    "Verify service definitions are properly structured",
                    "Check port mapping format (host:container)",
                    "Ensure volume and network references are valid"
                ]
            },
            "ansible": {
                "type": "suggestion",
                "description": "Ansible playbook configuration error",
                "fixes": [
                    "Check task structure and module names",
                    "Verify variable definitions and usage",
                    "Ensure proper indentation for task lists",
                    "Check handler definitions and notifications"
                ]
            },
            "github_actions": {
                "type": "suggestion",
                "description": "GitHub Actions workflow error",
                "fixes": [
                    "Verify workflow triggers (on) are correctly defined",
                    "Check job and step structure",
                    "Ensure action references are valid",
                    "Verify runner specifications"
                ]
            },
            "terraform": {
                "type": "suggestion",
                "description": "Terraform configuration error",
                "fixes": [
                    "Check provider configuration",
                    "Verify resource and data source syntax",
                    "Ensure variable types and defaults are correct",
                    "Check for circular dependencies"
                ]
            }
        }
        
        return framework_fixes.get(framework)
    
    def _template_based_patch(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                            content: str) -> Optional[Dict[str, Any]]:
        """Generate patch using templates."""
        root_cause = analysis.get("root_cause", "")
        
        # Map root causes to template names
        template_map = {
            "yaml_syntax_error": "yaml_syntax_fix",
            "yaml_indentation_error": "yaml_indentation_fix",
            "json_syntax_error": "json_syntax_fix",
            "config_validation_error": "config_validation_fix"
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


class YAMLJSONLanguagePlugin(LanguagePlugin):
    """
    Main YAML/JSON configuration language plugin for Homeostasis.
    
    This plugin orchestrates configuration file error analysis and patch generation,
    supporting multiple configuration frameworks and file formats.
    """
    
    VERSION = "1.0.0"
    AUTHOR = "Homeostasis Team"
    
    def __init__(self):
        """Initialize the YAML/JSON language plugin."""
        self.language = "yaml_json"
        self.supported_extensions = {".yaml", ".yml", ".json"}
        self.supported_frameworks = [
            "yaml", "json", "jsonschema",
            "kubernetes", "docker", "ansible", "github_actions", "gitlab_ci",
            "terraform", "eslint", "prettier", "babel", "webpack", "rollup",
            "vite", "package", "composer", "npm", "yarn"
        ]
        
        # Initialize components
        self.exception_handler = YAMLJSONExceptionHandler()
        self.patch_generator = YAMLJSONPatchGenerator()
        
        logger.info("YAML/JSON language plugin initialized")
    
    def get_language_id(self) -> str:
        """Get the unique identifier for this language."""
        return "yaml_json"
    
    def get_language_name(self) -> str:
        """Get the human-readable name of the language."""
        return "YAML/JSON"
    
    def get_language_version(self) -> str:
        """Get the version of the language supported by this plugin."""
        return "1.2+"
    
    def get_supported_frameworks(self) -> List[str]:
        """Get the list of frameworks supported by this language plugin."""
        return self.supported_frameworks
    
    def normalize_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize error data to the standard Homeostasis format.
        
        Args:
            error_data: Error data in the YAML/JSON-specific format
            
        Returns:
            Error data in the standard format
        """
        # Map YAML/JSON-specific error fields to standard format
        normalized = {
            "error_type": error_data.get("error_type", "ConfigError"),
            "message": error_data.get("message", error_data.get("description", "")),
            "language": "yaml_json",
            "file_type": error_data.get("file_type", error_data.get("format", "")),
            "file_path": error_data.get("file_path", error_data.get("filename", error_data.get("file", ""))),
            "line_number": error_data.get("line_number", error_data.get("line", 0)),
            "column_number": error_data.get("column_number", error_data.get("column", 0)),
            "content": error_data.get("content", error_data.get("source", "")),
            "framework": error_data.get("framework", ""),
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
        Convert standard format error data back to the YAML/JSON-specific format.
        
        Args:
            standard_error: Error data in the standard format
            
        Returns:
            Error data in the YAML/JSON-specific format
        """
        # Map standard fields back to YAML/JSON-specific format
        config_error = {
            "error_type": standard_error.get("error_type", "ConfigError"),
            "message": standard_error.get("message", ""),
            "file_type": standard_error.get("file_type", ""),
            "file_path": standard_error.get("file_path", ""),
            "line_number": standard_error.get("line_number", 0),
            "column_number": standard_error.get("column_number", 0),
            "content": standard_error.get("content", ""),
            "framework": standard_error.get("framework", ""),
            "description": standard_error.get("message", ""),
            "format": standard_error.get("file_type", ""),
            "filename": standard_error.get("file_path", ""),
            "file": standard_error.get("file_path", ""),
            "line": standard_error.get("line_number", 0),
            "column": standard_error.get("column_number", 0),
            "source": standard_error.get("content", ""),
            "stack_trace": standard_error.get("stack_trace", []),
            "context": standard_error.get("context", {}),
            "timestamp": standard_error.get("timestamp"),
            "severity": standard_error.get("severity", "medium")
        }
        
        # Add any additional fields from the standard error
        for key, value in standard_error.items():
            if key not in config_error and value is not None:
                config_error[key] = value
        
        return config_error
    
    def analyze_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a YAML/JSON configuration error.
        
        Args:
            error_data: Configuration error data
            
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
            analysis["plugin"] = "yaml_json"
            analysis["language"] = "yaml_json"
            analysis["plugin_version"] = self.VERSION
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing YAML/JSON error: {e}")
            return {
                "category": "yaml_json",
                "subcategory": "unknown",
                "confidence": "low",
                "suggested_fix": "Unable to analyze configuration error",
                "error": str(e),
                "plugin": "yaml_json"
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
        content = context.get("content", context.get("source_code", ""))
        
        fix = self.patch_generator.generate_patch(error_data, analysis, content)
        
        if fix:
            return fix
        else:
            return {
                "type": "suggestion",
                "description": analysis.get("suggested_fix", "No specific fix available"),
                "confidence": analysis.get("confidence", "low")
            }


# Register the plugin
register_plugin(YAMLJSONLanguagePlugin())