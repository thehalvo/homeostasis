"""
Dockerfile Language Plugin for Homeostasis

This plugin enables Homeostasis to analyze and fix errors in Dockerfile configurations.
It provides comprehensive error handling for Docker build errors, syntax issues,
layer optimization problems, and container best practices.
"""
import logging
import re
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union, Set

from ..language_plugin_system import LanguagePlugin, register_plugin

logger = logging.getLogger(__name__)


class DockerfileExceptionHandler:
    """
    Handles Dockerfile exceptions with robust error detection and classification.
    
    This class provides logic for categorizing Dockerfile errors based on their type,
    message, and common Docker build patterns and best practices.
    """
    
    def __init__(self):
        """Initialize the Dockerfile exception handler."""
        self.rule_categories = {
            "syntax": "Dockerfile syntax errors",
            "instruction": "Invalid instruction usage",
            "build": "Build context and process errors",
            "layer": "Layer optimization and caching issues",
            "security": "Security and best practice violations",
            "performance": "Performance optimization issues",
            "network": "Network and connectivity errors",
            "filesystem": "File system and path errors",
            "base_image": "Base image and FROM instruction errors",
            "multi_stage": "Multi-stage build errors",
            "args": "Build argument and variable errors",
            "copy": "COPY and ADD instruction errors"
        }
        
        # Common Dockerfile error patterns
        self.dockerfile_error_patterns = {
            "syntax_error": [
                r"Unknown instruction:",
                r"Dockerfile parse error",
                r"Invalid instruction format",
                r"Missing instruction argument",
                r"Unexpected token",
                r"Invalid Dockerfile"
            ],
            "instruction_error": [
                r"COPY failed:",
                r"ADD failed:",
                r"RUN returned a non-zero code:",
                r"Unable to execute command",
                r"Invalid instruction",
                r"Bad instruction"
            ],
            "build_context_error": [
                r"unable to prepare context:",
                r"build context is empty",
                r"no such file or directory",
                r"forbidden path outside the build context",
                r"Dockerfile not found"
            ],
            "base_image_error": [
                r"pull access denied",
                r"repository does not exist",
                r"image not found",
                r"manifest unknown",
                r"unauthorized: authentication required",
                r"invalid reference format"
            ],
            "network_error": [
                r"network timeout",
                r"connection refused",
                r"temporary failure in name resolution",
                r"no route to host",
                r"connection timed out"
            ],
            "filesystem_error": [
                r"no space left on device",
                r"permission denied",
                r"operation not permitted",
                r"file exists",
                r"directory not empty"
            ],
            "security_error": [
                r"running as root",
                r"setuid",
                r"privileged",
                r"--privileged",
                r"unsafe operation"
            ]
        }
        
        # Dockerfile instruction validation patterns
        self.instruction_patterns = {
            "FROM": {
                "required": True,
                "format": r"FROM\s+(?:\w+/)?[\w.-]+(?::\w+)?(?:\s+AS\s+\w+)?",
                "common_errors": ["missing tag", "invalid image name", "missing FROM"]
            },
            "RUN": {
                "format": r"RUN\s+.+",
                "common_errors": ["command not found", "permission denied", "package not available"]
            },
            "COPY": {
                "format": r"COPY\s+(?:--from=\w+\s+)?\S+\s+\S+",
                "common_errors": ["source not found", "destination invalid", "permission denied"]
            },
            "ADD": {
                "format": r"ADD\s+\S+\s+\S+",
                "common_errors": ["URL not accessible", "archive extraction failed", "source not found"]
            },
            "WORKDIR": {
                "format": r"WORKDIR\s+\S+",
                "common_errors": ["path not absolute", "directory creation failed"]
            },
            "EXPOSE": {
                "format": r"EXPOSE\s+\d+(?:/\w+)?",
                "common_errors": ["invalid port number", "port out of range"]
            },
            "ENV": {
                "format": r"ENV\s+\w+=\S+(?:\s+\w+=\S+)*",
                "common_errors": ["invalid variable name", "missing value"]
            },
            "ARG": {
                "format": r"ARG\s+\w+(?:=\S+)?",
                "common_errors": ["invalid argument name", "undefined argument used"]
            }
        }
        
        # Best practices violations
        self.best_practices = {
            "security": [
                "Avoid running as root user",
                "Don't use sudo in containers",
                "Minimize attack surface",
                "Use specific image tags instead of latest",
                "Scan for vulnerabilities"
            ],
            "performance": [
                "Minimize layer count",
                "Use .dockerignore file",
                "Combine RUN commands",
                "Order instructions by change frequency",
                "Use multi-stage builds"
            ],
            "maintainability": [
                "Use LABEL for metadata",
                "Document exposed ports",
                "Set proper WORKDIR",
                "Use specific base images",
                "Keep Dockerfile readable"
            ]
        }
        
        # Load rules from different categories
        self.rules = self._load_rules()
        
        # Pre-compile regex patterns for better performance
        self._compile_patterns()
    
    def _load_rules(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load Dockerfile error rules from rule files."""
        rules = {}
        rules_dir = Path(__file__).parent.parent / "rules" / "dockerfile"
        
        try:
            # Load common Dockerfile rules
            common_rules_path = rules_dir / "dockerfile_common_errors.json"
            if common_rules_path.exists():
                with open(common_rules_path, 'r') as f:
                    common_data = json.load(f)
                    rules["common"] = common_data.get("rules", [])
                    logger.info(f"Loaded {len(rules['common'])} common Dockerfile rules")
            
            # Load best practices rules
            practices_rules_path = rules_dir / "dockerfile_best_practices.json"
            if practices_rules_path.exists():
                with open(practices_rules_path, 'r') as f:
                    practices_data = json.load(f)
                    rules["best_practices"] = practices_data.get("rules", [])
                    logger.info(f"Loaded {len(rules['best_practices'])} best practices rules")
                        
        except Exception as e:
            logger.error(f"Error loading Dockerfile rules: {e}")
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
        Analyze a Dockerfile exception and determine its type and potential fixes.
        
        Args:
            error_data: Dockerfile error data in standard format
            
        Returns:
            Analysis results with categorization and fix suggestions
        """
        error_type = error_data.get("error_type", "DockerError")
        message = error_data.get("message", "")
        build_step = error_data.get("build_step", "")
        dockerfile_content = error_data.get("dockerfile_content", "")
        command = error_data.get("command", "")
        
        # Analyze based on error patterns
        analysis = self._analyze_by_patterns(message, build_step, dockerfile_content)
        
        # Check for best practices violations
        if dockerfile_content:
            practices_analysis = self._analyze_best_practices(dockerfile_content)
            if practices_analysis.get("confidence", "low") != "low":
                # Merge best practices findings
                analysis.setdefault("recommendations", []).extend(
                    practices_analysis.get("recommendations", [])
                )
        
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
        
        analysis["build_step"] = build_step
        analysis["command"] = command
        return analysis
    
    def _analyze_by_patterns(self, message: str, build_step: str, dockerfile_content: str) -> Dict[str, Any]:
        """Analyze error by matching against common patterns."""
        message_lower = message.lower()
        
        # Check syntax errors
        for pattern in self.dockerfile_error_patterns["syntax_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "dockerfile",
                    "subcategory": "syntax",
                    "confidence": "high",
                    "suggested_fix": "Fix Dockerfile syntax errors",
                    "root_cause": "dockerfile_syntax_error",
                    "severity": "high",
                    "tags": ["dockerfile", "syntax"]
                }
        
        # Check instruction errors
        for pattern in self.dockerfile_error_patterns["instruction_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "dockerfile",
                    "subcategory": "instruction",
                    "confidence": "high",
                    "suggested_fix": "Fix Dockerfile instruction usage",
                    "root_cause": "dockerfile_instruction_error",
                    "severity": "high",
                    "tags": ["dockerfile", "instruction"]
                }
        
        # Check build context errors
        for pattern in self.dockerfile_error_patterns["build_context_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "dockerfile",
                    "subcategory": "build_context",
                    "confidence": "high",
                    "suggested_fix": "Fix build context configuration and file paths",
                    "root_cause": "dockerfile_build_context_error",
                    "severity": "high",
                    "tags": ["dockerfile", "build_context"]
                }
        
        # Check base image errors
        for pattern in self.dockerfile_error_patterns["base_image_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "dockerfile",
                    "subcategory": "base_image",
                    "confidence": "high",
                    "suggested_fix": "Fix base image reference and accessibility",
                    "root_cause": "dockerfile_base_image_error",
                    "severity": "high",
                    "tags": ["dockerfile", "base_image", "registry"]
                }
        
        # Check network errors
        for pattern in self.dockerfile_error_patterns["network_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "dockerfile",
                    "subcategory": "network",
                    "confidence": "high",
                    "suggested_fix": "Fix network connectivity issues",
                    "root_cause": "dockerfile_network_error",
                    "severity": "medium",
                    "tags": ["dockerfile", "network", "connectivity"]
                }
        
        # Check filesystem errors
        for pattern in self.dockerfile_error_patterns["filesystem_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "dockerfile",
                    "subcategory": "filesystem",
                    "confidence": "high",
                    "suggested_fix": "Fix filesystem permissions and space issues",
                    "root_cause": "dockerfile_filesystem_error",
                    "severity": "medium",
                    "tags": ["dockerfile", "filesystem", "permissions"]
                }
        
        # Check security errors
        for pattern in self.dockerfile_error_patterns["security_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "dockerfile",
                    "subcategory": "security",
                    "confidence": "high",
                    "suggested_fix": "Address security concerns and follow best practices",
                    "root_cause": "dockerfile_security_issue",
                    "severity": "high",
                    "tags": ["dockerfile", "security", "best_practices"]
                }
        
        return {
            "category": "dockerfile",
            "subcategory": "unknown",
            "confidence": "low",
            "suggested_fix": "Review Dockerfile configuration and build process",
            "root_cause": "dockerfile_generic_error",
            "severity": "medium",
            "tags": ["dockerfile", "generic"]
        }
    
    def _analyze_best_practices(self, dockerfile_content: str) -> Dict[str, Any]:
        """Analyze Dockerfile for best practices violations."""
        recommendations = []
        lines = dockerfile_content.split('\n')
        
        # Check for common best practices violations
        
        # 1. Using latest tag
        for line in lines:
            if line.strip().upper().startswith('FROM') and ':latest' in line:
                recommendations.append({
                    "type": "best_practice",
                    "category": "security",
                    "issue": "Using 'latest' tag for base image",
                    "fix": "Use specific version tags for reproducible builds"
                })
        
        # 2. Running as root
        has_user_instruction = any(line.strip().upper().startswith('USER') for line in lines)
        if not has_user_instruction:
            recommendations.append({
                "type": "best_practice",
                "category": "security",
                "issue": "No USER instruction found - container runs as root",
                "fix": "Add USER instruction to run as non-root user"
            })
        
        # 3. Multiple RUN instructions that could be combined
        run_count = sum(1 for line in lines if line.strip().upper().startswith('RUN'))
        if run_count > 3:
            recommendations.append({
                "type": "best_practice",
                "category": "performance",
                "issue": f"Multiple RUN instructions ({run_count}) create unnecessary layers",
                "fix": "Combine RUN instructions using && to reduce layer count"
            })
        
        # 4. Missing .dockerignore recommendation
        recommendations.append({
            "type": "best_practice",
            "category": "performance",
            "issue": "Consider using .dockerignore file",
            "fix": "Create .dockerignore to exclude unnecessary files from build context"
        })
        
        # 5. WORKDIR best practice
        has_workdir = any(line.strip().upper().startswith('WORKDIR') for line in lines)
        if not has_workdir:
            recommendations.append({
                "type": "best_practice",
                "category": "maintainability",
                "issue": "No WORKDIR instruction found",
                "fix": "Set explicit WORKDIR to avoid using root directory"
            })
        
        if recommendations:
            return {
                "confidence": "medium",
                "recommendations": recommendations
            }
        
        return {"confidence": "low"}
    
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
        
        # Boost confidence for exact instruction matches
        rule_instruction = rule.get("instruction", "").upper()
        build_step = error_data.get("build_step", "").upper()
        if rule_instruction and build_step and rule_instruction in build_step:
            base_confidence += 0.2
        
        # Boost confidence based on rule reliability
        reliability = rule.get("reliability", "medium")
        reliability_boost = {"high": 0.2, "medium": 0.1, "low": 0.0}
        base_confidence += reliability_boost.get(reliability, 0.0)
        
        # Boost confidence for error type matches
        rule_tags = set(rule.get("tags", []))
        context_tags = set()
        
        if "build" in error_data.get("command", "").lower():
            context_tags.add("build")
        if error_data.get("dockerfile_content"):
            context_tags.add("dockerfile")
        
        if context_tags & rule_tags:
            base_confidence += 0.1
        
        return min(base_confidence, 1.0)


class DockerfilePatchGenerator:
    """
    Generates patches for Dockerfile errors based on analysis results.
    
    This class creates Dockerfile fixes for common errors using templates
    and heuristics specific to Docker build patterns and best practices.
    """
    
    def __init__(self):
        """Initialize the Dockerfile patch generator."""
        self.template_dir = Path(__file__).parent.parent / "patch_generation" / "templates"
        self.dockerfile_template_dir = self.template_dir / "dockerfile"
        
        # Ensure template directory exists
        self.dockerfile_template_dir.mkdir(parents=True, exist_ok=True)
        
        # Load patch templates
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, str]:
        """Load Dockerfile patch templates."""
        templates = {}
        
        if not self.dockerfile_template_dir.exists():
            logger.warning(f"Dockerfile templates directory not found: {self.dockerfile_template_dir}")
            return templates
        
        for template_file in self.dockerfile_template_dir.glob("*.dockerfile.template"):
            try:
                with open(template_file, 'r') as f:
                    template_name = template_file.stem.replace('.dockerfile', '')
                    templates[template_name] = f.read()
                    logger.debug(f"Loaded template: {template_name}")
            except Exception as e:
                logger.error(f"Error loading template {template_file}: {e}")
        
        return templates
    
    def generate_patch(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                      dockerfile_content: str = "") -> Optional[Dict[str, Any]]:
        """
        Generate a patch for the Dockerfile error.
        
        Args:
            error_data: The Dockerfile error data
            analysis: Analysis results from DockerfileExceptionHandler
            dockerfile_content: The Dockerfile content that caused the error
            
        Returns:
            Patch information or None if no patch can be generated
        """
        root_cause = analysis.get("root_cause", "")
        build_step = analysis.get("build_step", "")
        
        # Map root causes to patch strategies
        patch_strategies = {
            "dockerfile_syntax_error": self._fix_syntax_error,
            "dockerfile_instruction_error": self._fix_instruction_error,
            "dockerfile_build_context_error": self._fix_build_context_error,
            "dockerfile_base_image_error": self._fix_base_image_error,
            "dockerfile_network_error": self._fix_network_error,
            "dockerfile_filesystem_error": self._fix_filesystem_error,
            "dockerfile_security_issue": self._fix_security_issue
        }
        
        strategy = patch_strategies.get(root_cause)
        if strategy:
            try:
                return strategy(error_data, analysis, dockerfile_content)
            except Exception as e:
                logger.error(f"Error generating patch for {root_cause}: {e}")
        
        # Generate best practices suggestions
        if analysis.get("recommendations"):
            return self._generate_best_practices_patch(error_data, analysis, dockerfile_content)
        
        # Try to use templates if no specific strategy matches
        return self._template_based_patch(error_data, analysis, dockerfile_content)
    
    def _fix_syntax_error(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                         dockerfile_content: str) -> Optional[Dict[str, Any]]:
        """Fix Dockerfile syntax errors."""
        message = error_data.get("message", "")
        
        fixes = []
        
        if "unknown instruction" in message.lower():
            # Extract instruction name
            inst_match = re.search(r'unknown instruction:\s*(\w+)', message, re.IGNORECASE)
            if inst_match:
                instruction = inst_match.group(1)
                fixes.append({
                    "type": "suggestion",
                    "description": f"Unknown instruction '{instruction}'",
                    "fix": f"Check spelling of '{instruction}' or use valid Dockerfile instruction"
                })
        
        if "missing instruction argument" in message.lower():
            fixes.append({
                "type": "suggestion",
                "description": "Missing instruction argument",
                "fix": "Add required argument to the instruction"
            })
        
        if "invalid instruction format" in message.lower():
            fixes.append({
                "type": "suggestion",
                "description": "Invalid instruction format",
                "fix": "Check instruction syntax and argument format"
            })
        
        if fixes:
            return {
                "type": "multiple_suggestions",
                "fixes": fixes,
                "description": "Dockerfile syntax error fixes"
            }
        
        return {
            "type": "suggestion",
            "description": "Dockerfile syntax error. Check instruction format and arguments"
        }
    
    def _fix_instruction_error(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                              dockerfile_content: str) -> Optional[Dict[str, Any]]:
        """Fix Dockerfile instruction errors."""
        message = error_data.get("message", "")
        build_step = analysis.get("build_step", "")
        
        if "copy failed" in message.lower():
            return {
                "type": "suggestion",
                "description": "COPY instruction failed",
                "fixes": [
                    "Check source file exists in build context",
                    "Verify file paths are correct (case-sensitive)",
                    "Ensure destination directory exists or will be created",
                    "Check file permissions on source files",
                    "Use absolute paths or paths relative to WORKDIR"
                ]
            }
        
        if "add failed" in message.lower():
            return {
                "type": "suggestion",
                "description": "ADD instruction failed",
                "fixes": [
                    "Check source URL accessibility (for remote files)",
                    "Verify archive extraction permissions and format",
                    "Consider using COPY instead of ADD for local files",
                    "Check network connectivity for remote resources"
                ]
            }
        
        if "run returned a non-zero code" in message.lower():
            # Extract exit code if available
            code_match = re.search(r'returned a non-zero code:\s*(\d+)', message)
            exit_code = code_match.group(1) if code_match else "non-zero"
            
            return {
                "type": "suggestion",
                "description": f"RUN command failed with exit code {exit_code}",
                "fixes": [
                    "Check command syntax and availability",
                    "Verify package names and repositories",
                    "Ensure proper permissions for command execution",
                    "Add error handling with || true if failure is acceptable",
                    "Check base image has required tools installed"
                ]
            }
        
        return {
            "type": "suggestion",
            "description": "Dockerfile instruction error",
            "fixes": [
                "Check instruction syntax and arguments",
                "Verify file paths and permissions",
                "Review command execution context"
            ]
        }
    
    def _fix_build_context_error(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                                dockerfile_content: str) -> Optional[Dict[str, Any]]:
        """Fix Docker build context errors."""
        message = error_data.get("message", "")
        
        if "unable to prepare context" in message.lower():
            return {
                "type": "suggestion",
                "description": "Unable to prepare build context",
                "fixes": [
                    "Check Docker daemon is running",
                    "Verify build context directory exists",
                    "Check file permissions on build context",
                    "Ensure Dockerfile exists in specified location",
                    "Review .dockerignore file for exclusions"
                ]
            }
        
        if "build context is empty" in message.lower():
            return {
                "type": "suggestion",
                "description": "Build context is empty",
                "fixes": [
                    "Add files to build context directory",
                    "Check .dockerignore is not excluding all files",
                    "Verify you're running docker build from correct directory",
                    "Ensure build context path is correct"
                ]
            }
        
        if "forbidden path outside the build context" in message.lower():
            return {
                "type": "suggestion",
                "description": "Path outside build context",
                "fixes": [
                    "Move files into build context directory",
                    "Use relative paths within build context",
                    "Avoid using .. to access parent directories",
                    "Copy required files to build context before building"
                ]
            }
        
        return {
            "type": "suggestion",
            "description": "Build context error",
            "fixes": [
                "Check build context configuration",
                "Verify file paths and permissions",
                "Review .dockerignore file"
            ]
        }
    
    def _fix_base_image_error(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                             dockerfile_content: str) -> Optional[Dict[str, Any]]:
        """Fix Docker base image errors."""
        message = error_data.get("message", "")
        
        if "pull access denied" in message.lower():
            return {
                "type": "suggestion",
                "description": "Pull access denied for base image",
                "fixes": [
                    "Login to Docker registry: docker login",
                    "Check image exists and spelling is correct",
                    "Verify registry permissions and access rights",
                    "Use public image or configure private registry access",
                    "Check image name format: registry/namespace/repository:tag"
                ]
            }
        
        if "repository does not exist" in message.lower():
            return {
                "type": "suggestion",
                "description": "Repository does not exist",
                "fixes": [
                    "Check image name spelling and case sensitivity",
                    "Verify registry and namespace are correct",
                    "Use docker search to find available images",
                    "Check if image has been moved or deprecated"
                ]
            }
        
        if "manifest unknown" in message.lower():
            return {
                "type": "suggestion",
                "description": "Image manifest unknown",
                "fixes": [
                    "Check image tag exists for the specified architecture",
                    "Verify image tag spelling",
                    "Use docker pull to test image accessibility",
                    "Try using 'latest' tag or check available tags"
                ]
            }
        
        if "unauthorized: authentication required" in message.lower():
            return {
                "type": "suggestion",
                "description": "Authentication required for base image",
                "fixes": [
                    "Login to registry: docker login <registry>",
                    "Configure registry credentials",
                    "Check if image requires authentication",
                    "Use public alternative if available"
                ]
            }
        
        return {
            "type": "suggestion",
            "description": "Base image error",
            "fixes": [
                "Check image name and tag",
                "Verify registry access and authentication",
                "Test image pull manually"
            ]
        }
    
    def _fix_network_error(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                          dockerfile_content: str) -> Optional[Dict[str, Any]]:
        """Fix Docker network errors."""
        return {
            "type": "suggestion",
            "description": "Network connectivity error",
            "fixes": [
                "Check internet connectivity",
                "Verify DNS resolution is working",
                "Check firewall and proxy settings",
                "Try using different DNS servers",
                "Retry build after network issues are resolved",
                "Use --network=host for troubleshooting if needed"
            ]
        }
    
    def _fix_filesystem_error(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                             dockerfile_content: str) -> Optional[Dict[str, Any]]:
        """Fix Docker filesystem errors."""
        message = error_data.get("message", "")
        
        if "no space left on device" in message.lower():
            return {
                "type": "suggestion",
                "description": "No space left on device",
                "fixes": [
                    "Clean up Docker system: docker system prune",
                    "Remove unused images: docker image prune",
                    "Free up disk space on host system",
                    "Use multi-stage build to reduce final image size",
                    "Optimize Dockerfile to minimize layer sizes"
                ]
            }
        
        if "permission denied" in message.lower():
            return {
                "type": "suggestion",
                "description": "Permission denied",
                "fixes": [
                    "Check file permissions in build context",
                    "Run Docker with appropriate permissions",
                    "Fix file ownership: chown in Dockerfile",
                    "Use USER instruction to set proper user context",
                    "Check Docker daemon permissions"
                ]
            }
        
        return {
            "type": "suggestion",
            "description": "Filesystem error",
            "fixes": [
                "Check disk space and permissions",
                "Verify file system access",
                "Review Docker daemon configuration"
            ]
        }
    
    def _fix_security_issue(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                           dockerfile_content: str) -> Optional[Dict[str, Any]]:
        """Fix Docker security issues."""
        return {
            "type": "suggestion",
            "description": "Security issue detected",
            "fixes": [
                "Avoid running containers as root user",
                "Remove unnecessary privileges and capabilities",
                "Use specific image tags instead of 'latest'",
                "Scan images for vulnerabilities",
                "Follow Docker security best practices",
                "Minimize attack surface by removing unnecessary packages"
            ]
        }
    
    def _generate_best_practices_patch(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                                      dockerfile_content: str) -> Optional[Dict[str, Any]]:
        """Generate patches for best practices violations."""
        recommendations = analysis.get("recommendations", [])
        
        if not recommendations:
            return None
        
        fixes = []
        for rec in recommendations:
            fixes.append({
                "type": "best_practice",
                "category": rec.get("category", "general"),
                "issue": rec.get("issue", ""),
                "fix": rec.get("fix", "")
            })
        
        return {
            "type": "best_practices",
            "fixes": fixes,
            "description": "Dockerfile best practices recommendations"
        }
    
    def _template_based_patch(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                            dockerfile_content: str) -> Optional[Dict[str, Any]]:
        """Generate patch using templates."""
        root_cause = analysis.get("root_cause", "")
        
        # Map root causes to template names
        template_map = {
            "dockerfile_syntax_error": "syntax_fix",
            "dockerfile_instruction_error": "instruction_fix",
            "dockerfile_base_image_error": "base_image_fix",
            "dockerfile_security_issue": "security_fix"
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


class DockerfileLanguagePlugin(LanguagePlugin):
    """
    Main Dockerfile language plugin for Homeostasis.
    
    This plugin orchestrates Dockerfile error analysis and patch generation,
    supporting Docker build processes and container best practices.
    """
    
    VERSION = "1.0.0"
    AUTHOR = "Homeostasis Team"
    
    def __init__(self):
        """Initialize the Dockerfile language plugin."""
        self.language = "dockerfile"
        self.supported_extensions = {"Dockerfile", ".dockerfile"}
        self.supported_frameworks = [
            "docker", "buildx", "compose", "swarm", "kubernetes",
            "podman", "buildah", "skopeo"
        ]
        
        # Initialize components
        self.exception_handler = DockerfileExceptionHandler()
        self.patch_generator = DockerfilePatchGenerator()
        
        logger.info("Dockerfile language plugin initialized")
    
    def get_language_id(self) -> str:
        """Get the unique identifier for this language."""
        return "dockerfile"
    
    def get_language_name(self) -> str:
        """Get the human-readable name of the language."""
        return "Dockerfile"
    
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
            error_data: Error data in the Dockerfile-specific format
            
        Returns:
            Error data in the standard format
        """
        # Map Dockerfile-specific error fields to standard format
        normalized = {
            "error_type": error_data.get("error_type", "DockerError"),
            "message": error_data.get("message", error_data.get("description", "")),
            "language": "dockerfile",
            "command": error_data.get("command", ""),
            "build_step": error_data.get("build_step", error_data.get("step", "")),
            "dockerfile_path": error_data.get("dockerfile_path", error_data.get("file", "")),
            "dockerfile_content": error_data.get("dockerfile_content", ""),
            "line_number": error_data.get("line_number", error_data.get("line", 0)),
            "instruction": error_data.get("instruction", ""),
            "build_context": error_data.get("build_context", ""),
            "image_id": error_data.get("image_id", ""),
            "exit_code": error_data.get("exit_code", error_data.get("returncode", 0)),
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
        Convert standard format error data back to the Dockerfile-specific format.
        
        Args:
            standard_error: Error data in the standard format
            
        Returns:
            Error data in the Dockerfile-specific format
        """
        # Map standard fields back to Dockerfile-specific format
        dockerfile_error = {
            "error_type": standard_error.get("error_type", "DockerError"),
            "message": standard_error.get("message", ""),
            "command": standard_error.get("command", ""),
            "build_step": standard_error.get("build_step", ""),
            "dockerfile_path": standard_error.get("dockerfile_path", ""),
            "dockerfile_content": standard_error.get("dockerfile_content", ""),
            "line_number": standard_error.get("line_number", 0),
            "instruction": standard_error.get("instruction", ""),
            "build_context": standard_error.get("build_context", ""),
            "image_id": standard_error.get("image_id", ""),
            "exit_code": standard_error.get("exit_code", 0),
            "description": standard_error.get("message", ""),
            "step": standard_error.get("build_step", ""),
            "file": standard_error.get("dockerfile_path", ""),
            "line": standard_error.get("line_number", 0),
            "returncode": standard_error.get("exit_code", 0),
            "stack_trace": standard_error.get("stack_trace", []),
            "context": standard_error.get("context", {}),
            "timestamp": standard_error.get("timestamp"),
            "severity": standard_error.get("severity", "medium")
        }
        
        # Add any additional fields from the standard error
        for key, value in standard_error.items():
            if key not in dockerfile_error and value is not None:
                dockerfile_error[key] = value
        
        return dockerfile_error
    
    def analyze_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a Dockerfile error.
        
        Args:
            error_data: Dockerfile error data
            
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
            analysis["plugin"] = "dockerfile"
            analysis["language"] = "dockerfile"
            analysis["plugin_version"] = self.VERSION
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing Dockerfile error: {e}")
            return {
                "category": "dockerfile",
                "subcategory": "unknown",
                "confidence": "low",
                "suggested_fix": "Unable to analyze Dockerfile error",
                "error": str(e),
                "plugin": "dockerfile"
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
        dockerfile_content = context.get("dockerfile_content", context.get("source_code", ""))
        
        fix = self.patch_generator.generate_patch(error_data, analysis, dockerfile_content)
        
        if fix:
            return fix
        else:
            return {
                "type": "suggestion",
                "description": analysis.get("suggested_fix", "No specific fix available"),
                "confidence": analysis.get("confidence", "low")
            }


# Register the plugin
register_plugin(DockerfileLanguagePlugin())