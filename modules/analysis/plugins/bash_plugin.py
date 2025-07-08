"""
Bash/Shell Language Plugin for Homeostasis

This plugin enables Homeostasis to analyze and fix errors in Bash/Shell scripts.
It provides comprehensive error handling for shell scripting errors, including
syntax errors, command failures, variable issues, and script execution problems.
"""
import logging
import re
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union, Set

from ..language_plugin_system import LanguagePlugin, register_plugin

logger = logging.getLogger(__name__)


class BashExceptionHandler:
    """
    Handles Bash/Shell exceptions with robust error detection and classification.
    
    This class provides logic for categorizing shell script errors based on their type,
    message, and exit codes. It supports Bash, Zsh, Fish, and other shell environments.
    """
    
    def __init__(self):
        """Initialize the Bash exception handler."""
        self.rule_categories = {
            "syntax": "Shell syntax errors",
            "command": "Command not found or execution errors",
            "variable": "Variable and parameter errors",
            "file": "File and directory access errors", 
            "permission": "Permission and access errors",
            "redirection": "Input/output redirection errors",
            "expansion": "Parameter and command expansion errors",
            "arithmetic": "Arithmetic expression errors",
            "conditional": "Conditional expression errors",
            "loop": "Loop and iteration errors",
            "function": "Function definition and call errors",
            "signal": "Signal and process errors"
        }
        
        # Common shell error patterns
        self.error_patterns = {
            "syntax_error": [
                r"syntax error near unexpected token",
                r"syntax error: unexpected end of file",
                r"unexpected EOF while looking for matching",
                r"unterminated quoted string",
                r"command substitution: line \d+ syntax error",
                r"syntax error in conditional expression",
                r"bad substitution"
            ],
            "command_not_found": [
                r"command not found",
                r"No such file or directory",
                r"Permission denied",
                r"cannot execute binary file",
                r"Exec format error"
            ],
            "variable_error": [
                r"unbound variable",
                r"parameter null or not set",
                r"bad substitution",
                r"invalid variable name",
                r"ambiguous redirect"
            ],
            "file_error": [
                r"No such file or directory",
                r"Is a directory",
                r"Not a directory",
                r"File exists",
                r"Directory not empty"
            ],
            "permission_error": [
                r"Permission denied",
                r"Operation not permitted",
                r"cannot create",
                r"cannot remove",
                r"cannot access"
            ],
            "redirection_error": [
                r"ambiguous redirect",
                r"No such file or directory",
                r"cannot create",
                r"Bad file descriptor"
            ]
        }
        
        # Common shell exit codes and their meanings
        self.exit_codes = {
            1: "General errors",
            2: "Misuse of shell builtins",
            126: "Command invoked cannot execute",
            127: "Command not found",
            128: "Invalid argument to exit",
            129: "Fatal error signal '1' (SIGHUP)",
            130: "Script terminated by Control-C",
            131: "Fatal error signal '3' (SIGQUIT)",
            132: "Fatal error signal '4' (SIGILL)",
            133: "Fatal error signal '5' (SIGTRAP)",
            134: "Fatal error signal '6' (SIGABRT)",
            135: "Fatal error signal '7' (SIGBUS)",
            136: "Fatal error signal '8' (SIGFPE)",
            137: "Fatal error signal '9' (SIGKILL)",
            138: "Fatal error signal '10' (SIGUSR1)",
            139: "Fatal error signal '11' (SIGSEGV)",
            140: "Fatal error signal '12' (SIGUSR2)",
            141: "Fatal error signal '13' (SIGPIPE)",
            142: "Fatal error signal '14' (SIGALRM)",
            143: "Fatal error signal '15' (SIGTERM)"
        }
        
        # Load rules from different categories
        self.rules = self._load_rules()
        
        # Pre-compile regex patterns for better performance
        self._compile_patterns()
    
    def _load_rules(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load Bash error rules from rule files."""
        rules = {}
        rules_dir = Path(__file__).parent.parent / "rules" / "bash"
        
        try:
            # Load common Bash rules
            common_rules_path = rules_dir / "bash_common_errors.json"
            if common_rules_path.exists():
                with open(common_rules_path, 'r') as f:
                    common_data = json.load(f)
                    rules["common"] = common_data.get("rules", [])
                    logger.info(f"Loaded {len(rules['common'])} common Bash rules")
            
            # Load shell-specific rules
            for shell_type in ["bash", "zsh", "fish", "sh"]:
                shell_rules_path = rules_dir / f"{shell_type}_errors.json"
                if shell_rules_path.exists():
                    with open(shell_rules_path, 'r') as f:
                        shell_data = json.load(f)
                        rules[shell_type] = shell_data.get("rules", [])
                        logger.info(f"Loaded {len(rules[shell_type])} {shell_type} rules")
                        
        except Exception as e:
            logger.error(f"Error loading Bash rules: {e}")
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
        Analyze a Bash/Shell exception and determine its type and potential fixes.
        
        Args:
            error_data: Shell error data in standard format
            
        Returns:
            Analysis results with categorization and fix suggestions
        """
        error_type = error_data.get("error_type", "ShellError")
        message = error_data.get("message", "")
        exit_code = error_data.get("exit_code", 0)
        command = error_data.get("command", "")
        shell_type = error_data.get("shell_type", "bash")
        
        # Analyze based on error patterns
        analysis = self._analyze_by_patterns(message, exit_code, command)
        
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
        
        analysis["shell_type"] = shell_type
        analysis["exit_code"] = exit_code
        return analysis
    
    def _analyze_by_patterns(self, message: str, exit_code: int, command: str) -> Dict[str, Any]:
        """Analyze error by matching against common patterns."""
        message_lower = message.lower()
        
        # Check syntax errors
        for pattern in self.error_patterns["syntax_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "bash",
                    "subcategory": "syntax",
                    "confidence": "high",
                    "suggested_fix": "Fix shell script syntax errors",
                    "root_cause": "bash_syntax_error",
                    "severity": "high",
                    "tags": ["bash", "syntax"]
                }
        
        # Check command not found errors
        for pattern in self.error_patterns["command_not_found"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "bash",
                    "subcategory": "command",
                    "confidence": "high",
                    "suggested_fix": "Check command availability and PATH",
                    "root_cause": "bash_command_not_found",
                    "severity": "high",
                    "tags": ["bash", "command"]
                }
        
        # Check variable errors
        for pattern in self.error_patterns["variable_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "bash",
                    "subcategory": "variable",
                    "confidence": "high",
                    "suggested_fix": "Check variable definition and usage",
                    "root_cause": "bash_variable_error",
                    "severity": "medium",
                    "tags": ["bash", "variable"]
                }
        
        # Check file errors
        for pattern in self.error_patterns["file_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "bash",
                    "subcategory": "file",
                    "confidence": "high",
                    "suggested_fix": "Check file paths and existence",
                    "root_cause": "bash_file_error",
                    "severity": "medium",
                    "tags": ["bash", "file"]
                }
        
        # Check permission errors
        for pattern in self.error_patterns["permission_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "bash",
                    "subcategory": "permission",
                    "confidence": "high",
                    "suggested_fix": "Check file permissions and user access",
                    "root_cause": "bash_permission_error",
                    "severity": "high",
                    "tags": ["bash", "permission"]
                }
        
        # Check redirection errors
        for pattern in self.error_patterns["redirection_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "bash",
                    "subcategory": "redirection",
                    "confidence": "high",
                    "suggested_fix": "Check input/output redirection syntax",
                    "root_cause": "bash_redirection_error",
                    "severity": "medium",
                    "tags": ["bash", "redirection"]
                }
        
        return {
            "category": "bash",
            "subcategory": "unknown",
            "confidence": "low",
            "suggested_fix": "Review shell script for errors",
            "root_cause": "bash_generic_error",
            "severity": "medium",
            "tags": ["bash", "generic"]
        }
    
    def _analyze_by_exit_code(self, exit_code: int, message: str) -> Dict[str, Any]:
        """Analyze error by exit code."""
        if exit_code in self.exit_codes:
            description = self.exit_codes[exit_code]
            
            if exit_code == 127:
                return {
                    "category": "bash",
                    "subcategory": "command",
                    "confidence": "high",
                    "suggested_fix": "Check if command exists and is in PATH",
                    "root_cause": "bash_command_not_found",
                    "severity": "high",
                    "tags": ["bash", "command", "path"],
                    "exit_code_description": description
                }
            elif exit_code == 126:
                return {
                    "category": "bash",
                    "subcategory": "permission",
                    "confidence": "high",
                    "suggested_fix": "Check file permissions and execution rights",
                    "root_cause": "bash_execution_permission_denied",
                    "severity": "high",
                    "tags": ["bash", "permission", "execution"],
                    "exit_code_description": description
                }
            elif exit_code == 2:
                return {
                    "category": "bash",
                    "subcategory": "builtin",
                    "confidence": "high",
                    "suggested_fix": "Check shell builtin command usage",
                    "root_cause": "bash_builtin_misuse",
                    "severity": "medium",
                    "tags": ["bash", "builtin"],
                    "exit_code_description": description
                }
            elif exit_code >= 128 and exit_code <= 143:
                signal_num = exit_code - 128
                return {
                    "category": "bash",
                    "subcategory": "signal",
                    "confidence": "high",
                    "suggested_fix": f"Process terminated by signal {signal_num}. Check for interruption or system issues",
                    "root_cause": f"bash_signal_{signal_num}",
                    "severity": "high",
                    "tags": ["bash", "signal", "process"],
                    "exit_code_description": description
                }
            else:
                return {
                    "category": "bash",
                    "subcategory": "general",
                    "confidence": "medium",
                    "suggested_fix": f"Command failed with exit code {exit_code}: {description}",
                    "root_cause": f"bash_exit_code_{exit_code}",
                    "severity": "medium",
                    "tags": ["bash", "exit_code"],
                    "exit_code_description": description
                }
        
        return {
            "category": "bash",
            "subcategory": "unknown",
            "confidence": "low",
            "suggested_fix": f"Unknown exit code {exit_code}",
            "root_cause": f"bash_unknown_exit_code_{exit_code}",
            "severity": "medium",
            "tags": ["bash", "unknown"]
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
        
        # Boost confidence for exact shell type matches
        rule_shell = rule.get("shell_type", "").lower()
        error_shell = error_data.get("shell_type", "").lower()
        if rule_shell and rule_shell == error_shell:
            base_confidence += 0.2
        
        # Boost confidence based on rule reliability
        reliability = rule.get("reliability", "medium")
        reliability_boost = {"high": 0.2, "medium": 0.1, "low": 0.0}
        base_confidence += reliability_boost.get(reliability, 0.0)
        
        # Boost confidence for exit code matches
        rule_exit_code = rule.get("exit_code")
        error_exit_code = error_data.get("exit_code")
        if rule_exit_code and error_exit_code and rule_exit_code == error_exit_code:
            base_confidence += 0.2
        
        return min(base_confidence, 1.0)


class BashPatchGenerator:
    """
    Generates patches for Bash/Shell errors based on analysis results.
    
    This class creates shell script fixes for common errors using templates
    and heuristics specific to shell scripting patterns.
    """
    
    def __init__(self):
        """Initialize the Bash patch generator."""
        self.template_dir = Path(__file__).parent.parent / "patch_generation" / "templates"
        self.bash_template_dir = self.template_dir / "bash"
        
        # Ensure template directory exists
        self.bash_template_dir.mkdir(parents=True, exist_ok=True)
        
        # Load patch templates
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, str]:
        """Load Bash patch templates."""
        templates = {}
        
        if not self.bash_template_dir.exists():
            logger.warning(f"Bash templates directory not found: {self.bash_template_dir}")
            return templates
        
        for template_file in self.bash_template_dir.glob("*.sh.template"):
            try:
                with open(template_file, 'r') as f:
                    template_name = template_file.stem.replace('.sh', '')
                    templates[template_name] = f.read()
                    logger.debug(f"Loaded template: {template_name}")
            except Exception as e:
                logger.error(f"Error loading template {template_file}: {e}")
        
        return templates
    
    def generate_patch(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                      script_content: str = "") -> Optional[Dict[str, Any]]:
        """
        Generate a patch for the Bash error.
        
        Args:
            error_data: The Bash error data
            analysis: Analysis results from BashExceptionHandler
            script_content: The shell script content that caused the error
            
        Returns:
            Patch information or None if no patch can be generated
        """
        root_cause = analysis.get("root_cause", "")
        shell_type = analysis.get("shell_type", "bash")
        
        # Map root causes to patch strategies
        patch_strategies = {
            "bash_syntax_error": self._fix_syntax_error,
            "bash_command_not_found": self._fix_command_not_found,
            "bash_variable_error": self._fix_variable_error,
            "bash_file_error": self._fix_file_error,
            "bash_permission_error": self._fix_permission_error,
            "bash_redirection_error": self._fix_redirection_error,
            "bash_builtin_misuse": self._fix_builtin_error
        }
        
        strategy = patch_strategies.get(root_cause)
        if strategy:
            try:
                return strategy(error_data, analysis, script_content)
            except Exception as e:
                logger.error(f"Error generating patch for {root_cause}: {e}")
        
        # Try to use templates if no specific strategy matches
        return self._template_based_patch(error_data, analysis, script_content)
    
    def _fix_syntax_error(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                         script_content: str) -> Optional[Dict[str, Any]]:
        """Fix shell syntax errors."""
        message = error_data.get("message", "")
        
        fixes = []
        
        # Check for common syntax issues
        if "unexpected token" in message.lower():
            # Extract the unexpected token
            token_match = re.search(r"unexpected token `([^']+)'", message)
            if token_match:
                token = token_match.group(1)
                fixes.append({
                    "type": "suggestion",
                    "description": f"Unexpected token '{token}' found",
                    "fix": f"Check syntax around '{token}' - verify proper quoting, brackets, or operators"
                })
        
        if "unexpected end of file" in message.lower():
            fixes.append({
                "type": "suggestion",
                "description": "Unexpected end of file",
                "fix": "Check for unclosed quotes, brackets, or incomplete statements"
            })
        
        if "unterminated quoted string" in message.lower():
            fixes.append({
                "type": "suggestion",
                "description": "Unterminated quoted string",
                "fix": "Add missing closing quote for string literal"
            })
        
        if "looking for matching" in message.lower():
            # Extract what's missing
            matching_match = re.search(r"looking for matching `([^']+)'", message)
            if matching_match:
                missing = matching_match.group(1)
                fixes.append({
                    "type": "suggestion",
                    "description": f"Missing closing '{missing}'",
                    "fix": f"Add missing closing '{missing}' to match opening delimiter"
                })
        
        if fixes:
            return {
                "type": "multiple_suggestions",
                "fixes": fixes,
                "description": "Shell syntax error fixes"
            }
        
        return {
            "type": "suggestion",
            "description": "Shell syntax error detected. Check script syntax with shellcheck or bash -n"
        }
    
    def _fix_command_not_found(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                              script_content: str) -> Optional[Dict[str, Any]]:
        """Fix command not found errors."""
        message = error_data.get("message", "")
        command = error_data.get("command", "")
        
        # Extract command name from error message if not provided
        if not command:
            cmd_match = re.search(r"([^\s:]+): command not found", message)
            if cmd_match:
                command = cmd_match.group(1)
        
        if command:
            fixes = [
                f"Check if '{command}' is installed: which {command} || command -v {command}",
                f"Install the package containing '{command}'",
                f"Check if '{command}' is in PATH: echo $PATH",
                f"Use full path to '{command}' if it exists outside PATH",
                f"Check spelling of command name '{command}'"
            ]
            
            # Add common package suggestions for well-known commands
            common_packages = {
                "git": "git",
                "curl": "curl",
                "wget": "wget", 
                "python": "python3",
                "pip": "python3-pip",
                "node": "nodejs",
                "npm": "npm",
                "docker": "docker.io",
                "kubectl": "kubectl"
            }
            
            if command in common_packages:
                package = common_packages[command]
                fixes.insert(1, f"Install {command}: sudo apt-get install {package} (Ubuntu/Debian) or brew install {package} (macOS)")
            
            return {
                "type": "suggestion",
                "description": f"Command '{command}' not found",
                "fixes": fixes
            }
        
        return {
            "type": "suggestion",
            "description": "Command not found. Check command availability and PATH"
        }
    
    def _fix_variable_error(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                           script_content: str) -> Optional[Dict[str, Any]]:
        """Fix variable-related errors."""
        message = error_data.get("message", "")
        
        fixes = []
        
        if "unbound variable" in message.lower():
            # Extract variable name
            var_match = re.search(r"([^:\s]+): unbound variable", message)
            if var_match:
                var_name = var_match.group(1)
                fixes.extend([
                    f"Initialize variable '{var_name}' before use: {var_name}=\"default_value\"",
                    f"Use parameter expansion with default: ${{var_name:-default_value}}",
                    f"Check if variable is set: [[ -n \"${var_name}\" ]] && echo \"Set\" || echo \"Unset\"",
                    "Add 'set +u' to disable unbound variable checking (not recommended)"
                ])
        
        if "bad substitution" in message.lower():
            fixes.extend([
                "Check parameter expansion syntax: ${var}, ${var:-default}, ${var/pattern/replacement}",
                "Verify proper use of braces in variable expansion",
                "Check for typos in parameter expansion operators"
            ])
        
        if "parameter null or not set" in message.lower():
            fixes.extend([
                "Initialize the parameter before use",
                "Use parameter expansion with default values: ${param:-default}",
                "Check if running with 'set -u' which requires all variables to be set"
            ])
        
        if fixes:
            return {
                "type": "suggestion",
                "description": "Variable or parameter error",
                "fixes": fixes
            }
        
        return {
            "type": "suggestion",
            "description": "Variable error detected. Check variable declarations and usage"
        }
    
    def _fix_file_error(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                       script_content: str) -> Optional[Dict[str, Any]]:
        """Fix file and directory errors."""
        message = error_data.get("message", "")
        
        if "no such file or directory" in message.lower():
            return {
                "type": "suggestion",
                "description": "File or directory not found",
                "fixes": [
                    "Check if the file path is correct",
                    "Verify the file exists: ls -la /path/to/file",
                    "Create the file if it should exist: touch /path/to/file",
                    "Create the directory if it should exist: mkdir -p /path/to/dir",
                    "Check current working directory: pwd",
                    "Use absolute paths instead of relative paths"
                ]
            }
        
        if "is a directory" in message.lower():
            return {
                "type": "suggestion", 
                "description": "Attempted to use directory as file",
                "fixes": [
                    "Specify a file within the directory",
                    "Use directory-appropriate commands (ls, cd, etc.)",
                    "Check if you meant to target a file inside the directory"
                ]
            }
        
        if "not a directory" in message.lower():
            return {
                "type": "suggestion",
                "description": "Attempted to use file as directory",
                "fixes": [
                    "Check the path - it points to a file, not a directory",
                    "Remove the file if a directory is expected: rm file && mkdir dir",
                    "Use the correct directory path"
                ]
            }
        
        return {
            "type": "suggestion",
            "description": "File system error. Check paths and permissions"
        }
    
    def _fix_permission_error(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                             script_content: str) -> Optional[Dict[str, Any]]:
        """Fix permission-related errors."""
        message = error_data.get("message", "")
        
        return {
            "type": "suggestion",
            "description": "Permission denied",
            "fixes": [
                "Check file permissions: ls -la /path/to/file",
                "Add execute permissions: chmod +x /path/to/file",
                "Change ownership if needed: sudo chown user:group /path/to/file",
                "Run with sudo if administrative access is required",
                "Check if you're in the correct user context",
                "Verify file is not immutable: lsattr /path/to/file"
            ]
        }
    
    def _fix_redirection_error(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                              script_content: str) -> Optional[Dict[str, Any]]:
        """Fix input/output redirection errors."""
        message = error_data.get("message", "")
        
        if "ambiguous redirect" in message.lower():
            return {
                "type": "suggestion",
                "description": "Ambiguous redirection",
                "fixes": [
                    "Check redirection syntax: > file, >> file, 2> file, &> file",
                    "Quote filenames with spaces: > \"file name.txt\"",
                    "Ensure variable expansions result in single filenames",
                    "Use explicit file descriptors: 1> stdout.log 2> stderr.log"
                ]
            }
        
        if "bad file descriptor" in message.lower():
            return {
                "type": "suggestion",
                "description": "Bad file descriptor",
                "fixes": [
                    "Check file descriptor numbers (0=stdin, 1=stdout, 2=stderr)",
                    "Ensure file descriptors are open before redirecting",
                    "Use correct redirection syntax: 2>&1, 1>&2",
                    "Close file descriptors properly: exec 3>&-"
                ]
            }
        
        return {
            "type": "suggestion",
            "description": "Redirection error. Check input/output redirection syntax"
        }
    
    def _fix_builtin_error(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                          script_content: str) -> Optional[Dict[str, Any]]:
        """Fix shell builtin command errors."""
        return {
            "type": "suggestion",
            "description": "Shell builtin command error",
            "fixes": [
                "Check builtin command syntax: help <command>",
                "Verify command options and arguments",
                "Check if using correct shell (bash vs sh vs zsh)",
                "Review shell manual: man bash",
                "Use 'type <command>' to verify it's a builtin"
            ]
        }
    
    def _template_based_patch(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                            script_content: str) -> Optional[Dict[str, Any]]:
        """Generate patch using templates."""
        root_cause = analysis.get("root_cause", "")
        
        # Map root causes to template names
        template_map = {
            "bash_syntax_error": "syntax_fix",
            "bash_command_not_found": "command_check",
            "bash_variable_error": "variable_fix",
            "bash_permission_error": "permission_fix"
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


class BashLanguagePlugin(LanguagePlugin):
    """
    Main Bash/Shell language plugin for Homeostasis.
    
    This plugin orchestrates shell script error analysis and patch generation,
    supporting multiple shell environments and scripting patterns.
    """
    
    VERSION = "1.0.0"
    AUTHOR = "Homeostasis Team"
    
    def __init__(self):
        """Initialize the Bash language plugin."""
        self.language = "bash"
        self.supported_extensions = {".sh", ".bash", ".zsh", ".fish", ".ksh", ".csh"}
        self.supported_frameworks = [
            "bash", "zsh", "fish", "dash", "ksh", "csh", "tcsh",
            "sh", "ash", "busybox"
        ]
        
        # Initialize components
        self.exception_handler = BashExceptionHandler()
        self.patch_generator = BashPatchGenerator()
        
        logger.info("Bash language plugin initialized")
    
    def get_language_id(self) -> str:
        """Get the unique identifier for this language."""
        return "bash"
    
    def get_language_name(self) -> str:
        """Get the human-readable name of the language."""
        return "Bash/Shell"
    
    def get_language_version(self) -> str:
        """Get the version of the language supported by this plugin."""
        return "POSIX+"
    
    def get_supported_frameworks(self) -> List[str]:
        """Get the list of frameworks supported by this language plugin."""
        return self.supported_frameworks
    
    def normalize_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize error data to the standard Homeostasis format.
        
        Args:
            error_data: Error data in the Bash-specific format
            
        Returns:
            Error data in the standard format
        """
        # Map Bash-specific error fields to standard format
        normalized = {
            "error_type": error_data.get("error_type", "ShellError"),
            "message": error_data.get("message", error_data.get("stderr", "")),
            "language": "bash",
            "shell_type": error_data.get("shell_type", error_data.get("shell", "bash")),
            "exit_code": error_data.get("exit_code", error_data.get("returncode", 0)),
            "command": error_data.get("command", ""),
            "script_path": error_data.get("script_path", error_data.get("file", "")),
            "line_number": error_data.get("line_number", error_data.get("line", 0)),
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
        Convert standard format error data back to the Bash-specific format.
        
        Args:
            standard_error: Error data in the standard format
            
        Returns:
            Error data in the Bash-specific format
        """
        # Map standard fields back to Bash-specific format
        bash_error = {
            "error_type": standard_error.get("error_type", "ShellError"),
            "message": standard_error.get("message", ""),
            "shell_type": standard_error.get("shell_type", "bash"),
            "exit_code": standard_error.get("exit_code", 0),
            "command": standard_error.get("command", ""),
            "script_path": standard_error.get("script_path", ""),
            "line_number": standard_error.get("line_number", 0),
            "stderr": standard_error.get("message", ""),
            "shell": standard_error.get("shell_type", "bash"),
            "returncode": standard_error.get("exit_code", 0),
            "file": standard_error.get("script_path", ""),
            "line": standard_error.get("line_number", 0),
            "stack_trace": standard_error.get("stack_trace", []),
            "context": standard_error.get("context", {}),
            "timestamp": standard_error.get("timestamp"),
            "severity": standard_error.get("severity", "medium")
        }
        
        # Add any additional fields from the standard error
        for key, value in standard_error.items():
            if key not in bash_error and value is not None:
                bash_error[key] = value
        
        return bash_error
    
    def analyze_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a Bash/Shell error.
        
        Args:
            error_data: Bash error data
            
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
            analysis["plugin"] = "bash"
            analysis["language"] = "bash"
            analysis["plugin_version"] = self.VERSION
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing Bash error: {e}")
            return {
                "category": "bash",
                "subcategory": "unknown",
                "confidence": "low",
                "suggested_fix": "Unable to analyze shell error",
                "error": str(e),
                "plugin": "bash"
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
        script_content = context.get("script_content", context.get("source_code", ""))
        
        fix = self.patch_generator.generate_patch(error_data, analysis, script_content)
        
        if fix:
            return fix
        else:
            return {
                "type": "suggestion",
                "description": analysis.get("suggested_fix", "No specific fix available"),
                "confidence": analysis.get("confidence", "low")
            }


# Register the plugin
register_plugin(BashLanguagePlugin())