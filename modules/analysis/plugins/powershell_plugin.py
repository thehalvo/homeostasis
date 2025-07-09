"""
PowerShell Language Plugin for Homeostasis

This plugin enables Homeostasis to analyze and fix errors in PowerShell programming language code.
It provides comprehensive error handling for PowerShell syntax, cmdlets, pipeline operations,
and Windows-specific scripting patterns.
"""
import logging
import re
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

from ..language_plugin_system import LanguagePlugin, register_plugin

logger = logging.getLogger(__name__)


class PowerShellExceptionHandler:
    """Handles PowerShell exceptions with robust error detection and classification."""
    
    def __init__(self):
        self.rule_categories = {
            "syntax": "PowerShell syntax and parsing errors",
            "cmdlet": "Cmdlet and function errors",
            "pipeline": "Pipeline operation errors",
            "variable": "Variable and scope errors",
            "parameter": "Parameter binding errors",
            "execution": "Execution policy and permission errors",
            "module": "Module and import errors",
            "wmi": "WMI and CIM errors",
            "registry": "Windows registry errors",
            "filesystem": "File system operation errors"
        }
        
        self.powershell_error_patterns = {
            "syntax_error": [
                r"UnexpectedToken", r"MissingEndCurlyBrace", r"MissingEndParenthesis",
                r"MissingEndSquareBracket", r"MissingExpression", r"MissingFunctionBody",
                r"MissingOpenParenthesisInFunctionParameterList", r"MissingCloseParenthesisInFunctionParameterList"
            ],
            "cmdlet_error": [
                r"CommandNotFoundException", r"ParameterNotFound", r"AmbiguousParameterSet",
                r"MissingMandatoryParameter", r"ParameterArgumentValidationError",
                r"InvalidOperation", r"MethodNotFound", r"PropertyNotFound"
            ],
            "pipeline_error": [
                r"PipelineStoppedException", r"InvalidPipelineInput", r"ObjectNotFound",
                r"FormatError", r"ConvertError", r"InvalidCastException"
            ],
            "execution_error": [
                r"ExecutionPolicyRestricted", r"UnauthorizedAccessException",
                r"SecurityException", r"RemoteException", r"PSSecurityException"
            ],
            "module_error": [
                r"ModuleNotFoundError", r"ImportModuleError", r"FileNotFound",
                r"PathNotFound", r"DirectoryNotFound", r"AccessDenied"
            ]
        }
        
        self.rules = self._load_rules()
        self._compile_patterns()
    
    def _load_rules(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load PowerShell error rules from rule files."""
        rules = {}
        rules_dir = Path(__file__).parent.parent / "rules" / "powershell"
        
        try:
            common_rules_path = rules_dir / "powershell_common_errors.json"
            if common_rules_path.exists():
                with open(common_rules_path, 'r') as f:
                    common_data = json.load(f)
                    rules["common"] = common_data.get("rules", [])
        except Exception as e:
            logger.error(f"Error loading PowerShell rules: {e}")
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
        """Analyze a PowerShell exception and determine its type and potential fixes."""
        message = error_data.get("message", "")
        
        # Check for PowerShell-specific error patterns
        for pattern in self.powershell_error_patterns["syntax_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "powershell",
                    "subcategory": "syntax",
                    "confidence": "high",
                    "suggested_fix": "Fix PowerShell syntax errors",
                    "root_cause": "powershell_syntax_error",
                    "severity": "high",
                    "tags": ["powershell", "syntax"]
                }
        
        for pattern in self.powershell_error_patterns["cmdlet_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "powershell",
                    "subcategory": "cmdlet",
                    "confidence": "high",
                    "suggested_fix": "Fix cmdlet usage and parameters",
                    "root_cause": "powershell_cmdlet_error",
                    "severity": "high",
                    "tags": ["powershell", "cmdlet"]
                }
        
        for pattern in self.powershell_error_patterns["execution_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "powershell",
                    "subcategory": "execution",
                    "confidence": "high",
                    "suggested_fix": "Fix execution policy and permissions",
                    "root_cause": "powershell_execution_error",
                    "severity": "high",
                    "tags": ["powershell", "execution", "security"]
                }
        
        return {
            "category": "powershell",
            "subcategory": "unknown",
            "confidence": "low",
            "suggested_fix": "Review PowerShell code and error details",
            "root_cause": "powershell_generic_error",
            "severity": "medium",
            "tags": ["powershell", "generic"]
        }


class PowerShellPatchGenerator:
    """Generates patches for PowerShell errors based on analysis results."""
    
    def __init__(self):
        self.template_dir = Path(__file__).parent.parent / "patch_generation" / "templates"
        self.powershell_template_dir = self.template_dir / "powershell"
        self.powershell_template_dir.mkdir(parents=True, exist_ok=True)
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, str]:
        """Load PowerShell patch templates."""
        templates = {}
        if not self.powershell_template_dir.exists():
            return templates
        
        for template_file in self.powershell_template_dir.glob("*.ps1.template"):
            try:
                with open(template_file, 'r') as f:
                    template_name = template_file.stem.replace('.ps1', '')
                    templates[template_name] = f.read()
            except Exception as e:
                logger.error(f"Error loading template {template_file}: {e}")
        
        return templates
    
    def generate_patch(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                      source_code: str = "") -> Optional[Dict[str, Any]]:
        """Generate a patch for the PowerShell error."""
        root_cause = analysis.get("root_cause", "")
        
        if root_cause == "powershell_syntax_error":
            return self._fix_syntax_error(error_data, analysis, source_code)
        elif root_cause == "powershell_cmdlet_error":
            return self._fix_cmdlet_error(error_data, analysis, source_code)
        elif root_cause == "powershell_execution_error":
            return self._fix_execution_error(error_data, analysis, source_code)
        
        return {
            "type": "suggestion",
            "description": "General PowerShell error",
            "fixes": [
                "Check PowerShell syntax and cmdlet usage",
                "Verify execution policy settings",
                "Check module imports and dependencies"
            ]
        }
    
    def _fix_syntax_error(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                         source_code: str) -> Dict[str, Any]:
        """Fix PowerShell syntax errors."""
        return {
            "type": "suggestion",
            "description": "PowerShell syntax error",
            "fixes": [
                "Check for missing closing braces, parentheses, or brackets",
                "Verify proper variable declarations with $",
                "Ensure correct cmdlet and parameter syntax",
                "Check for proper string quoting and escaping"
            ]
        }
    
    def _fix_cmdlet_error(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                         source_code: str) -> Dict[str, Any]:
        """Fix PowerShell cmdlet errors."""
        return {
            "type": "suggestion",
            "description": "PowerShell cmdlet error",
            "fixes": [
                "Check cmdlet name spelling and availability",
                "Verify all mandatory parameters are provided",
                "Use Get-Help to check cmdlet syntax",
                "Check parameter types and values"
            ]
        }
    
    def _fix_execution_error(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                            source_code: str) -> Dict[str, Any]:
        """Fix PowerShell execution errors."""
        return {
            "type": "suggestion",
            "description": "PowerShell execution error",
            "fixes": [
                "Set execution policy: Set-ExecutionPolicy RemoteSigned",
                "Run PowerShell as administrator",
                "Check file and directory permissions",
                "Verify module signing and trust"
            ]
        }


class PowerShellLanguagePlugin(LanguagePlugin):
    """Main PowerShell language plugin for Homeostasis."""
    
    VERSION = "1.0.0"
    AUTHOR = "Homeostasis Team"
    
    def __init__(self):
        self.language = "powershell"
        self.supported_extensions = {".ps1", ".psm1", ".psd1"}
        self.supported_frameworks = ["powershell", "pwsh", "windows-powershell"]
        self.exception_handler = PowerShellExceptionHandler()
        self.patch_generator = PowerShellPatchGenerator()
        logger.info("PowerShell language plugin initialized")
    
    def get_language_id(self) -> str:
        return "powershell"
    
    def get_language_name(self) -> str:
        return "PowerShell"
    
    def get_language_version(self) -> str:
        return "5.1+"
    
    def get_supported_frameworks(self) -> List[str]:
        return self.supported_frameworks
    
    def normalize_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize error data to the standard Homeostasis format."""
        normalized = {
            "error_type": error_data.get("error_type", "PowerShellError"),
            "message": error_data.get("message", error_data.get("description", "")),
            "language": "powershell",
            "file_path": error_data.get("file_path", error_data.get("file", "")),
            "line_number": error_data.get("line_number", error_data.get("line", 0)),
            "column_number": error_data.get("column_number", error_data.get("column", 0)),
            "powershell_version": error_data.get("powershell_version", ""),
            "execution_policy": error_data.get("execution_policy", ""),
            "source_code": error_data.get("source_code", ""),
            "stack_trace": error_data.get("stack_trace", []),
            "context": error_data.get("context", {}),
            "timestamp": error_data.get("timestamp"),
            "severity": error_data.get("severity", "medium")
        }
        
        for key, value in error_data.items():
            if key not in normalized and value is not None:
                normalized[key] = value
        
        return normalized
    
    def denormalize_error(self, standard_error: Dict[str, Any]) -> Dict[str, Any]:
        """Convert standard format error data back to the PowerShell-specific format."""
        powershell_error = {
            "error_type": standard_error.get("error_type", "PowerShellError"),
            "message": standard_error.get("message", ""),
            "file_path": standard_error.get("file_path", ""),
            "line_number": standard_error.get("line_number", 0),
            "column_number": standard_error.get("column_number", 0),
            "powershell_version": standard_error.get("powershell_version", ""),
            "execution_policy": standard_error.get("execution_policy", ""),
            "source_code": standard_error.get("source_code", ""),
            "description": standard_error.get("message", ""),
            "file": standard_error.get("file_path", ""),
            "line": standard_error.get("line_number", 0),
            "column": standard_error.get("column_number", 0),
            "stack_trace": standard_error.get("stack_trace", []),
            "context": standard_error.get("context", {}),
            "timestamp": standard_error.get("timestamp"),
            "severity": standard_error.get("severity", "medium")
        }
        
        for key, value in standard_error.items():
            if key not in powershell_error and value is not None:
                powershell_error[key] = value
        
        return powershell_error
    
    def analyze_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a PowerShell error."""
        try:
            if not error_data.get("language"):
                standard_error = self.normalize_error(error_data)
            else:
                standard_error = error_data
            
            analysis = self.exception_handler.analyze_exception(standard_error)
            analysis["plugin"] = "powershell"
            analysis["language"] = "powershell"
            analysis["plugin_version"] = self.VERSION
            
            return analysis
        except Exception as e:
            logger.error(f"Error analyzing PowerShell error: {e}")
            return {
                "category": "powershell",
                "subcategory": "unknown",
                "confidence": "low",
                "suggested_fix": "Unable to analyze PowerShell error",
                "error": str(e),
                "plugin": "powershell"
            }
    
    def generate_fix(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a fix for an error based on the analysis."""
        error_data = context.get("error_data", {})
        source_code = context.get("source_code", "")
        
        fix = self.patch_generator.generate_patch(error_data, analysis, source_code)
        
        if fix:
            return fix
        else:
            return {
                "type": "suggestion",
                "description": analysis.get("suggested_fix", "No specific fix available"),
                "confidence": analysis.get("confidence", "low")
            }


# Register the plugin
register_plugin(PowerShellLanguagePlugin())