"""
Erlang Language Plugin for Homeostasis

This plugin enables Homeostasis to analyze and fix errors in Erlang programming language code.
It provides comprehensive error handling for Erlang compilation errors, OTP patterns,
process management, and BEAM VM issues to complement the existing Elixir support.
"""
import logging
import re
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

from ..language_plugin_system import LanguagePlugin, register_plugin

logger = logging.getLogger(__name__)


class ErlangExceptionHandler:
    """
    Handles Erlang exceptions with robust error detection and classification.
    
    This class provides logic for categorizing Erlang errors based on their type,
    message, and common actor model and OTP patterns.
    """
    
    def __init__(self):
        """Initialize the Erlang exception handler."""
        self.rule_categories = {
            "syntax": "Erlang syntax and parsing errors",
            "compilation": "Compilation and build errors",
            "runtime": "Runtime errors and exceptions",
            "process": "Process management and actor model errors",
            "otp": "OTP (Open Telecom Platform) errors",
            "supervision": "Supervision tree errors",
            "genserver": "GenServer behavior errors",
            "gensm": "GenStateMachine behavior errors",
            "application": "Application lifecycle errors",
            "distribution": "Distributed Erlang errors",
            "ets": "ETS table errors",
            "mnesia": "Mnesia database errors",
            "beam": "BEAM VM errors",
            "nif": "Native Implemented Functions errors"
        }
        
        # Common Erlang error patterns
        self.erlang_error_patterns = {
            "syntax_error": [
                r"syntax error before:",
                r"syntax error at line",
                r"unexpected.*?at line",
                r"unterminated.*?at line",
                r"illegal.*?at line",
                r"bad.*?character.*?at line",
                r"unexpected.*?token",
                r"missing.*?separator"
            ],
            "compilation_error": [
                r"function.*?undefined",
                r"module.*?undefined",
                r"record.*?undefined",
                r"variable.*?unsafe",
                r"variable.*?unbound",
                r"export.*?undefined",
                r"attribute.*?undefined",
                r"bad.*?function",
                r"bad.*?arity"
            ],
            "runtime_error": [
                r"badarg",
                r"badarith",
                r"badmatch",
                r"case_clause",
                r"function_clause",
                r"if_clause",
                r"try_clause",
                r"undef",
                r"noproc",
                r"timeout",
                r"system_limit",
                r"badkey",
                r"badmap"
            ],
            "process_error": [
                r"noproc",
                r"killed",
                r"normal",
                r"shutdown",
                r"process.*?died",
                r"process.*?terminated",
                r"link.*?process",
                r"monitor.*?process",
                r"exit.*?reason",
                r"trap_exit"
            ],
            "otp_error": [
                r"gen_server.*?terminated",
                r"gen_statem.*?terminated",
                r"gen_event.*?terminated",
                r"supervisor.*?terminated",
                r"application.*?failed",
                r"callback.*?failed",
                r"init.*?failed",
                r"handle_call.*?failed",
                r"handle_cast.*?failed",
                r"handle_info.*?failed"
            ],
            "supervision_error": [
                r"supervisor.*?child.*?died",
                r"supervisor.*?restart.*?failed",
                r"supervisor.*?shutdown.*?failed",
                r"child.*?specification.*?invalid",
                r"restart.*?intensity.*?exceeded",
                r"max_restarts.*?exceeded",
                r"supervisor.*?bridge.*?failed"
            ],
            "genserver_error": [
                r"gen_server.*?call.*?failed",
                r"gen_server.*?cast.*?failed",
                r"gen_server.*?timeout",
                r"gen_server.*?init.*?failed",
                r"gen_server.*?handle_call.*?crashed",
                r"gen_server.*?handle_cast.*?crashed",
                r"gen_server.*?handle_info.*?crashed",
                r"gen_server.*?terminate.*?failed"
            ],
            "application_error": [
                r"application.*?start.*?failed",
                r"application.*?stop.*?failed",
                r"application.*?not.*?started",
                r"application.*?already.*?started",
                r"application.*?dependency.*?failed",
                r"application.*?master.*?failed"
            ],
            "distribution_error": [
                r"distribution.*?failed",
                r"node.*?down",
                r"node.*?not.*?responding",
                r"net_kernel.*?failed",
                r"connection.*?closed",
                r"nodedown",
                r"noconnection",
                r"net_adm.*?failed"
            ],
            "ets_error": [
                r"ets.*?table.*?not.*?found",
                r"ets.*?table.*?already.*?exists",
                r"ets.*?badarg",
                r"ets.*?owner.*?died",
                r"ets.*?access.*?denied",
                r"ets.*?no_table"
            ],
            "mnesia_error": [
                r"mnesia.*?not.*?running",
                r"mnesia.*?table.*?not.*?found",
                r"mnesia.*?transaction.*?failed",
                r"mnesia.*?lock.*?failed",
                r"mnesia.*?aborted",
                r"mnesia.*?backup.*?failed",
                r"mnesia.*?restore.*?failed"
            ],
            "beam_error": [
                r"beam.*?error",
                r"system_limit",
                r"memory.*?limit",
                r"process.*?limit",
                r"atom.*?limit",
                r"code.*?loading.*?failed",
                r"scheduler.*?error"
            ],
            "nif_error": [
                r"nif.*?not.*?loaded",
                r"nif.*?library.*?failed",
                r"nif.*?function.*?failed",
                r"nif.*?upgrade.*?failed",
                r"nif.*?resource.*?error",
                r"nif.*?badarg"
            ]
        }
        
        # Erlang-specific concepts and their common issues
        self.erlang_concepts = {
            "process": ["process", "spawn", "pid", "link", "monitor"],
            "message": ["message", "send", "receive", "!", "?"],
            "pattern": ["pattern", "match", "case", "if", "guard"],
            "tuple": ["tuple", "{", "}", "element", "setelement"],
            "list": ["list", "[", "]", "head", "tail", "cons"],
            "atom": ["atom", "ok", "error", "true", "false"],
            "binary": ["binary", "<<", ">>", "bit", "byte"],
            "record": ["record", "field", "#", "record_info"],
            "fun": ["fun", "lambda", "closure", "function"],
            "try": ["try", "catch", "throw", "exit", "error"],
            "supervision": ["supervisor", "child", "restart", "shutdown"],
            "otp": ["gen_server", "gen_statem", "gen_event", "application"]
        }
        
        # Load rules from different categories
        self.rules = self._load_rules()
        
        # Pre-compile regex patterns for better performance
        self._compile_patterns()
    
    def _load_rules(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load Erlang error rules from rule files."""
        rules = {}
        rules_dir = Path(__file__).parent.parent / "rules" / "erlang"
        
        try:
            # Load common Erlang rules
            common_rules_path = rules_dir / "erlang_common_errors.json"
            if common_rules_path.exists():
                with open(common_rules_path, 'r') as f:
                    common_data = json.load(f)
                    rules["common"] = common_data.get("rules", [])
                    logger.info(f"Loaded {len(rules['common'])} common Erlang rules")
            
            # Load concept-specific rules
            for concept in ["otp", "processes", "supervision", "distribution"]:
                concept_rules_path = rules_dir / f"erlang_{concept}_errors.json"
                if concept_rules_path.exists():
                    with open(concept_rules_path, 'r') as f:
                        concept_data = json.load(f)
                        rules[concept] = concept_data.get("rules", [])
                        logger.info(f"Loaded {len(rules[concept])} {concept} rules")
                        
        except Exception as e:
            logger.error(f"Error loading Erlang rules: {e}")
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
        Analyze an Erlang exception and determine its type and potential fixes.
        
        Args:
            error_data: Erlang error data in standard format
            
        Returns:
            Analysis results with categorization and fix suggestions
        """
        message = error_data.get("message", "")
        file_path = error_data.get("file_path", "")
        line_number = error_data.get("line_number", 0)
        column_number = error_data.get("column_number", 0)
        
        # Analyze based on error patterns
        analysis = self._analyze_by_patterns(message, file_path)
        
        # Check for concept-specific issues only if we don't have high confidence already
        if analysis.get("confidence", "low") != "high":
            concept_analysis = self._analyze_erlang_concepts(message)
            if concept_analysis.get("confidence", "low") != "low":
                # Merge concept-specific findings
                analysis.update(concept_analysis)
        
        # Find matching rules
        matches = self._find_matching_rules(message, error_data)
        
        if matches:
            # Use the best match (highest confidence)
            best_match = max(matches, key=lambda x: x.get("confidence_score", 0))
            
            # Map rule types to expected subcategories
            type_mapping = {
                "SyntaxError": "syntax",
                "RuntimeError": "runtime",
                "CompilationError": "compilation",
                "ImportError": "import",
                "TypeError": "type",
                "ValueError": "value",
                "AttributeError": "attribute",
                "NameError": "name",
                "IOError": "io",
                "FileNotFoundError": "file"
            }
            
            rule_type = best_match.get("type", "unknown")
            # First check if we can determine subcategory from root_cause
            root_cause = best_match.get("root_cause", "")
            message = error_data.get("message", "")
            
            # Check message-specific patterns first for accurate categorization
            if "function clause head cannot match" in message:
                subcategory = "function"
            elif "no case clause matching" in message:
                subcategory = "pattern"
            elif "undefined function" in message:
                subcategory = "module"
            elif "head mismatch" in message:
                subcategory = "compilation"
            elif "process" in root_cause:
                subcategory = "process"
            elif "otp" in root_cause:
                subcategory = "otp"
            elif "module" in root_cause:
                subcategory = "module"
            elif "pattern" in root_cause:
                subcategory = "pattern"
            elif "function" in root_cause:
                subcategory = "function"
            elif "supervision" in root_cause:
                subcategory = "supervision"
            elif "genserver" in root_cause:
                subcategory = "genserver"
            elif "distribution" in root_cause:
                subcategory = "distribution"
            elif "message" in root_cause:
                subcategory = "message"
            else:
                # Fall back to type mapping
                subcategory = type_mapping.get(rule_type, rule_type.lower())
            
            analysis.update({
                "category": best_match.get("category", analysis.get("category", "unknown")),
                "subcategory": subcategory,
                "confidence": best_match.get("confidence", "medium"),
                "suggested_fix": best_match.get("suggestion", analysis.get("suggested_fix", "")),
                "root_cause": best_match.get("root_cause", analysis.get("root_cause", "")),
                "severity": best_match.get("severity", "medium"),
                "rule_id": best_match.get("id", ""),
                "tags": best_match.get("tags", []),
                "all_matches": matches
            })
        
        analysis["file_path"] = file_path
        analysis["line_number"] = line_number
        analysis["column_number"] = column_number
        return analysis
    
    def _analyze_by_patterns(self, message: str, file_path: str) -> Dict[str, Any]:
        """Analyze error by matching against common patterns."""
        
        # Check for specific error types first
        if "** exception error" in message:
            return {
                "category": "erlang",
                "subcategory": "process",
                "confidence": "high",
                "suggested_fix": "Fix process management and actor model errors",
                "root_cause": "erlang_process_error",
                "severity": "high",
                "tags": ["erlang", "process", "exception"]
            }
        
        if "undefined function" in message:
            return {
                "category": "erlang",
                "subcategory": "module",
                "confidence": "high",
                "suggested_fix": "Fix undefined function or module errors",
                "root_cause": "erlang_module_error",
                "severity": "high",
                "tags": ["erlang", "module", "function"]
            }
        
        if "head mismatch" in message:
            return {
                "category": "erlang",
                "subcategory": "compilation",
                "confidence": "high",
                "suggested_fix": "Fix compilation and build errors",
                "root_cause": "erlang_compilation_error",
                "severity": "high",
                "tags": ["erlang", "compilation", "mismatch"]
            }
        
        # Check syntax errors
        for pattern in self.erlang_error_patterns["syntax_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "erlang",
                    "subcategory": "syntax",
                    "confidence": "high",
                    "suggested_fix": "Fix Erlang syntax errors",
                    "root_cause": "erlang_syntax_error",
                    "severity": "high",
                    "tags": ["erlang", "syntax", "parser"]
                }
        
        # Check compilation errors
        for pattern in self.erlang_error_patterns["compilation_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "erlang",
                    "subcategory": "compilation",
                    "confidence": "high",
                    "suggested_fix": "Fix compilation and build errors",
                    "root_cause": "erlang_compilation_error",
                    "severity": "high",
                    "tags": ["erlang", "compilation", "build"]
                }
        
        # Check runtime errors
        for pattern in self.erlang_error_patterns["runtime_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "erlang",
                    "subcategory": "runtime",
                    "confidence": "high",
                    "suggested_fix": "Fix runtime errors and exceptions",
                    "root_cause": "erlang_runtime_error",
                    "severity": "high",
                    "tags": ["erlang", "runtime", "exception"]
                }
        
        # Check process errors
        for pattern in self.erlang_error_patterns["process_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "erlang",
                    "subcategory": "process",
                    "confidence": "high",
                    "suggested_fix": "Fix process management and actor model errors",
                    "root_cause": "erlang_process_error",
                    "severity": "high",
                    "tags": ["erlang", "process", "actor"]
                }
        
        # Check OTP errors
        for pattern in self.erlang_error_patterns["otp_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "erlang",
                    "subcategory": "otp",
                    "confidence": "high",
                    "suggested_fix": "Fix OTP behavior and pattern errors",
                    "root_cause": "erlang_otp_error",
                    "severity": "high",
                    "tags": ["erlang", "otp", "behavior"]
                }
        
        # Check supervision errors
        for pattern in self.erlang_error_patterns["supervision_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "erlang",
                    "subcategory": "supervision",
                    "confidence": "high",
                    "suggested_fix": "Fix supervision tree and restart strategy errors",
                    "root_cause": "erlang_supervision_error",
                    "severity": "high",
                    "tags": ["erlang", "supervision", "restart"]
                }
        
        # Check GenServer errors
        for pattern in self.erlang_error_patterns["genserver_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "erlang",
                    "subcategory": "genserver",
                    "confidence": "high",
                    "suggested_fix": "Fix GenServer behavior and callback errors",
                    "root_cause": "erlang_genserver_error",
                    "severity": "high",
                    "tags": ["erlang", "genserver", "callback"]
                }
        
        # Check distribution errors
        for pattern in self.erlang_error_patterns["distribution_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "erlang",
                    "subcategory": "distribution",
                    "confidence": "high",
                    "suggested_fix": "Fix distributed Erlang and clustering errors",
                    "root_cause": "erlang_distribution_error",
                    "severity": "medium",
                    "tags": ["erlang", "distribution", "cluster"]
                }
        
        return {
            "category": "erlang",
            "subcategory": "unknown",
            "confidence": "low",
            "suggested_fix": "Review Erlang code and compiler error details",
            "root_cause": "erlang_generic_error",
            "severity": "medium",
            "tags": ["erlang", "generic"]
        }
    
    def _analyze_erlang_concepts(self, message: str) -> Dict[str, Any]:
        """Analyze Erlang-specific concept errors."""
        message_lower = message.lower()
        
        # Check for specific function errors first
        if "function clause" in message_lower:
            return {
                "category": "erlang",
                "subcategory": "function",
                "confidence": "high",
                "suggested_fix": "Handle function clauses and patterns properly",
                "root_cause": "erlang_function_error",
                "severity": "high",
                "tags": ["erlang", "function", "clause"]
            }
        
        # Check for process-related errors
        if any(keyword in message_lower for keyword in self.erlang_concepts["process"]):
            return {
                "category": "erlang",
                "subcategory": "process",
                "confidence": "high",
                "suggested_fix": "Handle process lifecycle and communication properly",
                "root_cause": "erlang_process_error",
                "severity": "high",
                "tags": ["erlang", "process", "actor"]
            }
        
        # Check for OTP-related errors
        if any(keyword in message_lower for keyword in self.erlang_concepts["otp"]):
            return {
                "category": "erlang",
                "subcategory": "otp",
                "confidence": "high",
                "suggested_fix": "Handle OTP behaviors and patterns properly",
                "root_cause": "erlang_otp_error",
                "severity": "high",
                "tags": ["erlang", "otp", "behavior"]
            }
        
        # Check for supervision-related errors
        if any(keyword in message_lower for keyword in self.erlang_concepts["supervision"]):
            return {
                "category": "erlang",
                "subcategory": "supervision",
                "confidence": "high",
                "suggested_fix": "Handle supervision trees and restart strategies",
                "root_cause": "erlang_supervision_error",
                "severity": "high",
                "tags": ["erlang", "supervision", "restart"]
            }
        
        # Check for pattern matching errors
        if any(keyword in message_lower for keyword in self.erlang_concepts["pattern"]):
            # Special case for "no case clause matching" to have high confidence
            confidence = "high" if "no case clause matching" in message else "medium"
            return {
                "category": "erlang",
                "subcategory": "pattern",
                "confidence": confidence,
                "suggested_fix": "Handle pattern matching and guards properly",
                "root_cause": "erlang_pattern_error",
                "severity": "medium",
                "tags": ["erlang", "pattern", "match"]
            }
        
        # Check for message passing errors (but avoid generic "message" word)
        message_keywords = [k for k in self.erlang_concepts["message"] if k != "message"]
        if any(keyword in message_lower for keyword in message_keywords) or \
           ("message" in message_lower and ("send" in message_lower or "receive" in message_lower or "!" in message)):
            return {
                "category": "erlang",
                "subcategory": "message",
                "confidence": "medium",
                "suggested_fix": "Handle message passing and receive patterns",
                "root_cause": "erlang_message_error",
                "severity": "medium",
                "tags": ["erlang", "message", "receive"]
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
        
        # Boost confidence for file extension matches
        file_path = error_data.get("file_path", "")
        if file_path.endswith(".erl") or file_path.endswith(".hrl"):
            base_confidence += 0.2
        
        # Boost confidence based on rule reliability
        reliability = rule.get("reliability", "medium")
        reliability_boost = {"high": 0.2, "medium": 0.1, "low": 0.0}
        base_confidence += reliability_boost.get(reliability, 0.0)
        
        # Boost confidence for concept matches
        rule_tags = set(rule.get("tags", []))
        context_tags = set()
        
        message = error_data.get("message", "").lower()
        if "process" in message:
            context_tags.add("process")
        if "gen_server" in message:
            context_tags.add("genserver")
        if "supervisor" in message:
            context_tags.add("supervision")
        
        if context_tags & rule_tags:
            base_confidence += 0.1
        
        return min(base_confidence, 1.0)


class ErlangPatchGenerator:
    """
    Generates patches for Erlang errors based on analysis results.
    
    This class creates Erlang code fixes for common errors using templates
    and heuristics specific to actor model and OTP patterns.
    """
    
    def __init__(self):
        """Initialize the Erlang patch generator."""
        self.template_dir = Path(__file__).parent.parent / "patch_generation" / "templates"
        self.erlang_template_dir = self.template_dir / "erlang"
        
        # Ensure template directory exists
        self.erlang_template_dir.mkdir(parents=True, exist_ok=True)
        
        # Load patch templates
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, str]:
        """Load Erlang patch templates."""
        templates = {}
        
        if not self.erlang_template_dir.exists():
            logger.warning(f"Erlang templates directory not found: {self.erlang_template_dir}")
            return templates
        
        for template_file in self.erlang_template_dir.glob("*.erl.template"):
            try:
                with open(template_file, 'r') as f:
                    template_name = template_file.stem.replace('.erl', '')
                    templates[template_name] = f.read()
                    logger.debug(f"Loaded template: {template_name}")
            except Exception as e:
                logger.error(f"Error loading template {template_file}: {e}")
        
        return templates
    
    def generate_patch(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                      source_code: str = "") -> Optional[Dict[str, Any]]:
        """
        Generate a patch for the Erlang error.
        
        Args:
            error_data: The Erlang error data
            analysis: Analysis results from ErlangExceptionHandler
            source_code: The Erlang source code that caused the error
            
        Returns:
            Patch information or None if no patch can be generated
        """
        root_cause = analysis.get("root_cause", "")
        
        # Map root causes to patch strategies
        patch_strategies = {
            "erlang_syntax_error": self._fix_syntax_error,
            "erlang_compilation_error": self._fix_compilation_error,
            "erlang_runtime_error": self._fix_runtime_error,
            "erlang_process_error": self._fix_process_error,
            "erlang_otp_error": self._fix_otp_error,
            "erlang_supervision_error": self._fix_supervision_error,
            "erlang_genserver_error": self._fix_genserver_error,
            "erlang_distribution_error": self._fix_distribution_error,
            "erlang_pattern_error": self._fix_pattern_error,
            "erlang_message_error": self._fix_message_error,
            "erlang_function_error": self._fix_function_error,
            "erlang_module_error": self._fix_module_error
        }
        
        strategy = patch_strategies.get(root_cause)
        if strategy:
            try:
                return strategy(error_data, analysis, source_code)
            except Exception as e:
                logger.error(f"Error generating patch for {root_cause}: {e}")
        
        # Try to use templates if no specific strategy matches
        return self._template_based_patch(error_data, analysis, source_code)
    
    def _fix_syntax_error(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                         source_code: str) -> Optional[Dict[str, Any]]:
        """Fix Erlang syntax errors."""
        message = error_data.get("message", "")
        
        if "syntax error" in message.lower():
            return {
                "type": "suggestion",
                "description": "Erlang syntax error",
                "fixes": [
                    "Check for missing commas, semicolons, or periods",
                    "Verify proper atom and variable naming",
                    "Ensure balanced parentheses and brackets",
                    "Check for proper string and binary syntax"
                ]
            }
        
        return {
            "type": "suggestion",
            "description": "Syntax error. Check Erlang syntax and structure"
        }
    
    def _fix_compilation_error(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                              source_code: str) -> Optional[Dict[str, Any]]:
        """Fix compilation errors."""
        message = error_data.get("message", "")
        
        if "function" in message.lower() and "undefined" in message.lower():
            return {
                "type": "suggestion",
                "description": "Undefined function",
                "fixes": [
                    "Define the missing function",
                    "Check function name spelling",
                    "Verify function arity matches call",
                    "Add function to module exports"
                ]
            }
        
        if "module" in message.lower() and "undefined" in message.lower():
            return {
                "type": "suggestion",
                "description": "Undefined module",
                "fixes": [
                    "Check module name spelling",
                    "Ensure module file exists",
                    "Verify module is in code path",
                    "Check for circular dependencies"
                ]
            }
        
        return {
            "type": "suggestion",
            "description": "Compilation error. Check function and module definitions"
        }
    
    def _fix_runtime_error(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                          source_code: str) -> Optional[Dict[str, Any]]:
        """Fix runtime errors."""
        message = error_data.get("message", "")
        
        if "badarg" in message.lower():
            return {
                "type": "suggestion",
                "description": "Bad argument error",
                "fixes": [
                    "Check function arguments and types",
                    "Validate input parameters",
                    "Use guards to check argument validity",
                    "Handle edge cases in function calls"
                ]
            }
        
        if "badmatch" in message.lower():
            return {
                "type": "suggestion",
                "description": "Pattern match failure",
                "fixes": [
                    "Check pattern matching expressions",
                    "Ensure patterns match expected values",
                    "Use case expressions for multiple patterns",
                    "Handle all possible match cases"
                ]
            }
        
        if "case_clause" in message.lower():
            return {
                "type": "suggestion",
                "description": "Case clause error",
                "fixes": [
                    "Add missing case clause patterns",
                    "Use catch-all pattern (_) for default case",
                    "Check case expression value types",
                    "Ensure all possible values are handled"
                ]
            }
        
        return {
            "type": "suggestion",
            "description": "Runtime error. Check pattern matching and function calls"
        }
    
    def _fix_process_error(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                          source_code: str) -> Optional[Dict[str, Any]]:
        """Fix process management errors."""
        message = error_data.get("message", "")
        
        if "noproc" in message.lower():
            return {
                "type": "suggestion",
                "description": "Process not found error",
                "fixes": [
                    "Check if process is still alive before sending messages",
                    "Use process monitoring to detect process death",
                    "Handle process termination gracefully",
                    "Verify process registration and naming"
                ]
            }
        
        return {
            "type": "suggestion",
            "description": "Process error. Check process lifecycle and communication"
        }
    
    def _fix_otp_error(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                      source_code: str) -> Optional[Dict[str, Any]]:
        """Fix OTP behavior errors."""
        return {
            "type": "suggestion",
            "description": "OTP behavior error",
            "fixes": [
                "Implement required callback functions",
                "Handle init/1 callback properly",
                "Check callback return values",
                "Follow OTP behavior patterns"
            ]
        }
    
    def _fix_supervision_error(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                              source_code: str) -> Optional[Dict[str, Any]]:
        """Fix supervision tree errors."""
        return {
            "type": "suggestion",
            "description": "Supervision error",
            "fixes": [
                "Check supervisor child specifications",
                "Verify restart strategies and intensities",
                "Handle supervisor initialization properly",
                "Ensure proper child process startup"
            ]
        }
    
    def _fix_genserver_error(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                            source_code: str) -> Optional[Dict[str, Any]]:
        """Fix GenServer behavior errors."""
        return {
            "type": "suggestion",
            "description": "GenServer error",
            "fixes": [
                "Implement required GenServer callbacks",
                "Handle init/1 return values properly",
                "Use proper gen_server:call/cast patterns",
                "Handle terminate/2 callback correctly"
            ]
        }
    
    def _fix_distribution_error(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                               source_code: str) -> Optional[Dict[str, Any]]:
        """Fix distributed Erlang errors."""
        return {
            "type": "suggestion",
            "description": "Distribution error",
            "fixes": [
                "Check node connectivity and naming",
                "Verify distributed Erlang configuration",
                "Handle node down events properly",
                "Use proper inter-node communication patterns"
            ]
        }
    
    def _fix_pattern_error(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                          source_code: str) -> Optional[Dict[str, Any]]:
        """Fix pattern matching errors."""
        return {
            "type": "suggestion",
            "description": "Pattern matching error",
            "fixes": [
                "Add missing pattern match cases",
                "Use guards for complex pattern conditions",
                "Handle all tuple and list patterns",
                "Use catch-all patterns for default cases"
            ]
        }
    
    def _fix_message_error(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                          source_code: str) -> Optional[Dict[str, Any]]:
        """Fix message passing errors."""
        return {
            "type": "suggestion",
            "description": "Message passing error",
            "fixes": [
                "Use proper message sending syntax (Process ! Message)",
                "Handle all expected message patterns in receive",
                "Use selective receive for specific messages",
                "Add timeout clauses to receive expressions"
            ]
        }
    
    def _fix_function_error(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                           source_code: str) -> Optional[Dict[str, Any]]:
        """Fix function errors."""
        message = error_data.get("message", "")
        
        if "function clause" in message.lower():
            return {
                "type": "suggestion",
                "description": "Function clause error",
                "fixes": [
                    "Check function clause patterns match the arguments",
                    "Add missing function clauses for all cases",
                    "Verify guards in function heads",
                    "Ensure function arity matches calls"
                ]
            }
        
        return {
            "type": "suggestion",
            "description": "Function error. Check function definitions and clauses"
        }
    
    def _fix_module_error(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                         source_code: str) -> Optional[Dict[str, Any]]:
        """Fix module errors."""
        message = error_data.get("message", "")
        
        if "undefined function" in message.lower():
            return {
                "type": "suggestion",
                "description": "Undefined function error",
                "fixes": [
                    "Define the missing function in the module",
                    "Check function name spelling and arity",
                    "Add the function to -export([...]) list",
                    "Import the function from the correct module"
                ]
            }
        
        return {
            "type": "suggestion",
            "description": "Module error. Check module and function definitions"
        }
    
    def _template_based_patch(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                            source_code: str) -> Optional[Dict[str, Any]]:
        """Generate patch using templates."""
        root_cause = analysis.get("root_cause", "")
        subcategory = analysis.get("subcategory", "")
        
        # Map root causes to template names
        template_map = {
            "erlang_syntax_error": "syntax_fix",
            "erlang_process_error": "process_fix",
            "erlang_otp_error": "otp_fix",
            "erlang_genserver_error": "genserver_fix",
            "erlang_supervision_error": "supervision_fix",
            "erlang_pattern_error": "pattern_fix"
        }
        
        template_name = template_map.get(root_cause)
        if template_name and template_name in self.templates:
            template = self.templates[template_name]
            
            return {
                "type": "template",
                "template": template,
                "description": f"Applied template fix for {root_cause}"
            }
        
        # Return a default suggestion if no template is found
        return {
            "type": "suggestion",
            "description": f"Fix {subcategory} error in Erlang code",
            "fixes": [
                f"Review the {subcategory} error details",
                "Check Erlang documentation for proper syntax",
                "Ensure code follows OTP principles"
            ]
        }


class ErlangLanguagePlugin(LanguagePlugin):
    """
    Main Erlang language plugin for Homeostasis.
    
    This plugin orchestrates Erlang error analysis and patch generation,
    supporting actor model and OTP patterns to complement existing Elixir support.
    """
    
    VERSION = "1.0.0"
    AUTHOR = "Homeostasis Team"
    
    def __init__(self):
        """Initialize the Erlang language plugin."""
        self.language = "erlang"
        self.supported_extensions = {".erl", ".hrl", ".app", ".app.src"}
        self.supported_frameworks = [
            "erlang", "otp", "rebar3", "erlang.mk", "mix",
            "cowboy", "ranch", "gen_server", "gen_statem", "mnesia"
        ]
        
        # Initialize components
        self.exception_handler = ErlangExceptionHandler()
        self.patch_generator = ErlangPatchGenerator()
        
        logger.info("Erlang language plugin initialized")
    
    def get_language_id(self) -> str:
        """Get the unique identifier for this language."""
        return "erlang"
    
    def get_language_name(self) -> str:
        """Get the human-readable name of the language."""
        return "Erlang"
    
    def get_language_version(self) -> str:
        """Get the version of the language supported by this plugin."""
        return "OTP 24+"
    
    def get_supported_frameworks(self) -> List[str]:
        """Get the list of frameworks supported by this language plugin."""
        return self.supported_frameworks
    
    def normalize_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize error data to the standard Homeostasis format.
        
        Args:
            error_data: Error data in the Erlang-specific format
            
        Returns:
            Error data in the standard format
        """
        # Map Erlang-specific error fields to standard format
        normalized = {
            "error_type": error_data.get("error_type", "ErlangError"),
            "message": error_data.get("message", error_data.get("description", "")),
            "language": "erlang",
            "file_path": error_data.get("file_path", error_data.get("file", "")),
            "line_number": error_data.get("line_number", error_data.get("line", 0)),
            "column_number": error_data.get("column_number", error_data.get("column", 0)),
            "otp_version": error_data.get("otp_version", ""),
            "beam_version": error_data.get("beam_version", ""),
            "node_name": error_data.get("node_name", ""),
            "process_id": error_data.get("process_id", ""),
            "source_code": error_data.get("source_code", ""),
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
        Convert standard format error data back to the Erlang-specific format.
        
        Args:
            standard_error: Error data in the standard format
            
        Returns:
            Error data in the Erlang-specific format
        """
        # Map standard fields back to Erlang-specific format
        erlang_error = {
            "error_type": standard_error.get("error_type", "ErlangError"),
            "message": standard_error.get("message", ""),
            "file_path": standard_error.get("file_path", ""),
            "line_number": standard_error.get("line_number", 0),
            "column_number": standard_error.get("column_number", 0),
            "otp_version": standard_error.get("otp_version", ""),
            "beam_version": standard_error.get("beam_version", ""),
            "node_name": standard_error.get("node_name", ""),
            "process_id": standard_error.get("process_id", ""),
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
        
        # Add any additional fields from the standard error
        for key, value in standard_error.items():
            if key not in erlang_error and value is not None:
                erlang_error[key] = value
        
        return erlang_error
    
    def analyze_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze an Erlang error.
        
        Args:
            error_data: Erlang error data
            
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
            analysis["plugin"] = "erlang"
            analysis["language"] = "erlang"
            analysis["plugin_version"] = self.VERSION
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing Erlang error: {e}")
            return {
                "category": "erlang",
                "subcategory": "unknown",
                "confidence": "low",
                "suggested_fix": "Unable to analyze Erlang error",
                "error": str(e),
                "plugin": "erlang"
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
register_plugin(ErlangLanguagePlugin())