"""
Lua Language Plugin for Homeostasis

This plugin enables Homeostasis to analyze and fix errors in Lua programming language code.
It provides comprehensive error handling for Lua syntax errors, runtime issues,
and embedded scripting best practices.
"""
import logging
import re
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

from ..language_plugin_system import LanguagePlugin, register_plugin

logger = logging.getLogger(__name__)


class LuaExceptionHandler:
    """
    Handles Lua exceptions with robust error detection and classification.
    
    This class provides logic for categorizing Lua errors based on their type,
    message, and common scripting patterns.
    """
    
    def __init__(self):
        """Initialize the Lua exception handler."""
        self.rule_categories = {
            "syntax": "Lua syntax and parsing errors",
            "runtime": "Runtime errors and exceptions",
            "table": "Table access and manipulation errors",
            "function": "Function definition and calling errors",
            "variable": "Variable scope and access errors",
            "metamethod": "Metamethod and metatable errors",
            "coroutine": "Coroutine and threading errors",
            "module": "Module loading and require errors",
            "io": "Input/output and file handling errors",
            "string": "String manipulation and pattern matching errors",
            "math": "Mathematical operations and errors",
            "memory": "Memory allocation and garbage collection errors"
        }
        
        # Common Lua error patterns
        self.lua_error_patterns = {
            "syntax_error": [
                r"syntax error",
                r"unexpected symbol",
                r"'end' expected",
                r"'=' expected",
                r"malformed number",
                r"unfinished string",
                r"unexpected.*?near",
                r"expected.*?near",
                r"missing.*?near"
            ],
            "runtime_error": [
                r"stack overflow",
                r"memory allocation error",
                r"C stack overflow",
                r"not enough memory"
            ],
            "table_error": [
                r"attempt to index.*?nil",
                r"attempt to index.*?number",
                r"attempt to index.*?string",
                r"attempt to index.*?boolean",
                r"table index is nil",
                r"invalid key to 'next'"
            ],
            "function_error": [
                r"attempt to call.*?nil",
                r"attempt to call.*?number",
                r"attempt to call.*?string",
                r"attempt to call.*?boolean",
                r"bad argument.*?to.*?expected",
                r"wrong number of arguments",
                r"function.*?not defined"
            ],
            "variable_error": [
                r"attempt to.*?global.*?nil",
                r"attempt to.*?local.*?nil",
                r"variable.*?not defined",
                r"global.*?not found",
                r"undefined variable"
            ],
            "metamethod_error": [
                r"attempt to.*?metamethod",
                r"metamethod.*?not found",
                r"invalid.*?metamethod",
                r"bad argument.*?metamethod"
            ],
            "coroutine_error": [
                r"cannot resume.*?coroutine",
                r"cannot yield.*?coroutine",
                r"attempt to yield.*?main thread",
                r"dead coroutine"
            ],
            "module_error": [
                r"module.*?not found",
                r"error loading module",
                r"loop in require",
                r"bad argument.*?require"
            ],
            "io_error": [
                r"cannot open.*?file",
                r"cannot read.*?file",
                r"cannot write.*?file",
                r"file.*?not found",
                r"permission denied"
            ],
            "string_error": [
                r"bad argument.*?string",
                r"invalid.*?pattern",
                r"malformed.*?pattern",
                r"string.*?too long"
            ],
            "math_error": [
                r"bad argument.*?math",
                r"domain error",
                r"argument.*?out of range",
                r"division by zero"
            ]
        }
        
        # Lua-specific concepts and their common issues
        self.lua_concepts = {
            "tables": ["table access", "table index", "table operations"],
            "functions": ["function call", "function definition", "closure"],
            "metatables": ["metamethod", "metatable", "__index", "__newindex"],
            "coroutines": ["coroutine", "yield", "resume", "thread"],
            "modules": ["require", "module", "package"],
            "upvalues": ["upvalue", "closure", "local function"],
            "weak_tables": ["weak table", "weak reference", "gc"],
            "patterns": ["pattern", "gsub", "match", "find"]
        }
        
        # Load rules from different categories
        self.rules = self._load_rules()
        
        # Pre-compile regex patterns for better performance
        self._compile_patterns()
    
    def _load_rules(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load Lua error rules from rule files."""
        rules = {}
        rules_dir = Path(__file__).parent.parent / "rules" / "lua"
        
        try:
            # Load common Lua rules
            common_rules_path = rules_dir / "lua_common_errors.json"
            if common_rules_path.exists():
                with open(common_rules_path, 'r') as f:
                    common_data = json.load(f)
                    rules["common"] = common_data.get("rules", [])
                    logger.info(f"Loaded {len(rules['common'])} common Lua rules")
            
            # Load concept-specific rules
            for concept in ["tables", "functions", "coroutines", "modules"]:
                concept_rules_path = rules_dir / f"lua_{concept}_errors.json"
                if concept_rules_path.exists():
                    with open(concept_rules_path, 'r') as f:
                        concept_data = json.load(f)
                        rules[concept] = concept_data.get("rules", [])
                        logger.info(f"Loaded {len(rules[concept])} {concept} rules")
                        
        except Exception as e:
            logger.error(f"Error loading Lua rules: {e}")
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
        Analyze a Lua exception and determine its type and potential fixes.
        
        Args:
            error_data: Lua error data in standard format
            
        Returns:
            Analysis results with categorization and fix suggestions
        """
        error_type = error_data.get("error_type", "LuaError")
        message = error_data.get("message", "")
        file_path = error_data.get("file_path", "")
        line_number = error_data.get("line_number", 0)
        
        # Analyze based on error patterns
        analysis = self._analyze_by_patterns(message, file_path)
        
        # Only check for concept-specific issues if no specific pattern matched
        if analysis.get("subcategory") == "unknown":
            concept_analysis = self._analyze_lua_concepts(message)
            if concept_analysis.get("confidence", "low") != "low":
                # Merge concept-specific findings
                analysis.update(concept_analysis)
        
        # Find matching rules only if we haven't already categorized well
        if analysis.get("confidence") == "low" or analysis.get("subcategory") == "unknown":
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
        
        analysis["file_path"] = file_path
        analysis["line_number"] = line_number
        return analysis
    
    def _analyze_by_patterns(self, message: str, file_path: str) -> Dict[str, Any]:
        """Analyze error by matching against common patterns."""
        message_lower = message.lower()
        
        # Check syntax errors
        for pattern in self.lua_error_patterns["syntax_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "lua",
                    "subcategory": "syntax",
                    "confidence": "high",
                    "suggested_fix": "Fix Lua syntax errors",
                    "root_cause": "lua_syntax_error",
                    "severity": "high",
                    "tags": ["lua", "syntax", "parser"]
                }
        
        # Check for more specific nil errors first
        if "attempt to index" in message_lower and "nil" in message_lower:
            return {
                "category": "lua",
                "subcategory": "nil",
                "confidence": "high",
                "suggested_fix": "Fix nil value access",
                "root_cause": "lua_nil_error",
                "severity": "high",
                "tags": ["lua", "nil"]
            }
        
        # Check for type errors (concatenation)
        if "attempt to concatenate" in message_lower and "nil" in message_lower:
            return {
                "category": "lua",
                "subcategory": "type",
                "confidence": "high",
                "suggested_fix": "Fix type conversion errors",
                "root_cause": "lua_type_error",
                "severity": "high",
                "tags": ["lua", "type"]
            }
        
        # Check for function errors
        if "attempt to call" in message_lower and "nil" in message_lower:
            return {
                "category": "lua",
                "subcategory": "function",
                "confidence": "high",
                "suggested_fix": "Fix function call errors",
                "root_cause": "lua_function_error",
                "severity": "high",
                "tags": ["lua", "function"]
            }
        
        # Check for arithmetic errors
        if "attempt to perform arithmetic" in message_lower:
            return {
                "category": "lua",
                "subcategory": "arithmetic",
                "confidence": "high",
                "suggested_fix": "Fix arithmetic operation errors",
                "root_cause": "lua_arithmetic_error",
                "severity": "high",
                "tags": ["lua", "arithmetic"]
            }
        
        # Check runtime errors (more general)
        for pattern in self.lua_error_patterns["runtime_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "lua",
                    "subcategory": "runtime",
                    "confidence": "high",
                    "suggested_fix": "Fix runtime errors",
                    "root_cause": "lua_runtime_error",
                    "severity": "high",
                    "tags": ["lua", "runtime"]
                }
        
        # Check for table index errors
        if "table index is nil" in message_lower:
            return {
                "category": "lua",
                "subcategory": "table",
                "confidence": "high",
                "suggested_fix": "Fix table indexing errors",
                "root_cause": "lua_table_error",
                "severity": "high",
                "tags": ["lua", "table"]
            }
        
        # Check for module errors
        if "module" in message_lower and "not found" in message_lower:
            return {
                "category": "lua",
                "subcategory": "module",
                "confidence": "high",
                "suggested_fix": "Fix module loading errors",
                "root_cause": "lua_module_error",
                "severity": "high",
                "tags": ["lua", "module"]
            }
        
        # Check table errors
        for pattern in self.lua_error_patterns["table_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "lua",
                    "subcategory": "table",
                    "confidence": "high",
                    "suggested_fix": "Fix table access and indexing errors",
                    "root_cause": "lua_table_error",
                    "severity": "high",
                    "tags": ["lua", "table", "index"]
                }
        
        # Check function errors
        for pattern in self.lua_error_patterns["function_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "lua",
                    "subcategory": "function",
                    "confidence": "high",
                    "suggested_fix": "Fix function definition and calling errors",
                    "root_cause": "lua_function_error",
                    "severity": "high",
                    "tags": ["lua", "function", "call"]
                }
        
        # Check variable errors
        for pattern in self.lua_error_patterns["variable_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "lua",
                    "subcategory": "variable",
                    "confidence": "high",
                    "suggested_fix": "Fix variable scope and access errors",
                    "root_cause": "lua_variable_error",
                    "severity": "high",
                    "tags": ["lua", "variable", "scope"]
                }
        
        # Check metamethod errors
        for pattern in self.lua_error_patterns["metamethod_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "lua",
                    "subcategory": "metamethod",
                    "confidence": "high",
                    "suggested_fix": "Fix metamethod and metatable errors",
                    "root_cause": "lua_metamethod_error",
                    "severity": "medium",
                    "tags": ["lua", "metamethod", "metatable"]
                }
        
        # Check coroutine errors
        for pattern in self.lua_error_patterns["coroutine_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "lua",
                    "subcategory": "coroutine",
                    "confidence": "high",
                    "suggested_fix": "Fix coroutine and threading errors",
                    "root_cause": "lua_coroutine_error",
                    "severity": "medium",
                    "tags": ["lua", "coroutine", "thread"]
                }
        
        # Check module errors
        for pattern in self.lua_error_patterns["module_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "lua",
                    "subcategory": "module",
                    "confidence": "high",
                    "suggested_fix": "Fix module loading and require errors",
                    "root_cause": "lua_module_error",
                    "severity": "high",
                    "tags": ["lua", "module", "require"]
                }
        
        # Check IO errors
        for pattern in self.lua_error_patterns["io_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "lua",
                    "subcategory": "io",
                    "confidence": "high",
                    "suggested_fix": "Fix input/output and file handling errors",
                    "root_cause": "lua_io_error",
                    "severity": "high",
                    "tags": ["lua", "io", "file"]
                }
        
        # Check string errors
        for pattern in self.lua_error_patterns["string_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "lua",
                    "subcategory": "string",
                    "confidence": "high",
                    "suggested_fix": "Fix string manipulation and pattern errors",
                    "root_cause": "lua_string_error",
                    "severity": "medium",
                    "tags": ["lua", "string", "pattern"]
                }
        
        # Check math errors
        for pattern in self.lua_error_patterns["math_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "lua",
                    "subcategory": "math",
                    "confidence": "high",
                    "suggested_fix": "Fix mathematical operations and errors",
                    "root_cause": "lua_math_error",
                    "severity": "medium",
                    "tags": ["lua", "math", "calculation"]
                }
        
        return {
            "category": "lua",
            "subcategory": "unknown",
            "confidence": "low",
            "suggested_fix": "Review Lua code and error details",
            "root_cause": "lua_generic_error",
            "severity": "medium",
            "tags": ["lua", "generic"]
        }
    
    def _analyze_lua_concepts(self, message: str) -> Dict[str, Any]:
        """Analyze Lua-specific concept errors."""
        message_lower = message.lower()
        
        # Check for nil access errors
        if "attempt to index" in message_lower and "nil" in message_lower:
            return {
                "category": "lua",
                "subcategory": "nil",
                "confidence": "high",
                "suggested_fix": "Add nil check before accessing values",
                "root_cause": "lua_nil_error",
                "severity": "high",
                "tags": ["lua", "nil", "safety"]
            }
        
        # Check for table-related errors
        if "table index" in message_lower:
            return {
                "category": "lua",
                "subcategory": "table",
                "confidence": "high",
                "suggested_fix": "Check table operations and key access",
                "root_cause": "lua_table_error",
                "severity": "high",
                "tags": ["lua", "table", "index"]
            }
        
        # Check for function-related errors
        if "attempt to call" in message_lower:
            return {
                "category": "lua",
                "subcategory": "function",
                "confidence": "high",
                "suggested_fix": "Check function definitions and calls",
                "root_cause": "lua_function_error",
                "severity": "high",
                "tags": ["lua", "function", "call"]
            }
        
        # Check for type errors (concatenation on nil)
        if "concatenate" in message_lower and "nil" in message_lower:
            return {
                "category": "lua",
                "subcategory": "type",
                "confidence": "high",
                "suggested_fix": "Check types before operations",
                "root_cause": "lua_type_error",
                "severity": "high",
                "tags": ["lua", "type", "conversion"]
            }
        
        # Check for module errors
        if "module" in message_lower and "not found" in message_lower:
            return {
                "category": "lua",
                "subcategory": "module",
                "confidence": "high",
                "suggested_fix": "Check module loading and require paths",
                "root_cause": "lua_module_error",
                "severity": "high",
                "tags": ["lua", "module", "require"]
            }
        
        # Check for arithmetic errors
        if "arithmetic" in message_lower:
            return {
                "category": "lua",
                "subcategory": "arithmetic",
                "confidence": "high",
                "suggested_fix": "Check arithmetic operations and type conversions",
                "root_cause": "lua_arithmetic_error",
                "severity": "high",
                "tags": ["lua", "arithmetic", "type"]
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
        if file_path.endswith(".lua"):
            base_confidence += 0.2
        
        # Boost confidence based on rule reliability
        reliability = rule.get("reliability", "medium")
        reliability_boost = {"high": 0.2, "medium": 0.1, "low": 0.0}
        base_confidence += reliability_boost.get(reliability, 0.0)
        
        # Boost confidence for concept matches
        rule_tags = set(rule.get("tags", []))
        context_tags = set()
        
        message = error_data.get("message", "").lower()
        if "nil" in message:
            context_tags.add("nil")
        if "table" in message:
            context_tags.add("table")
        if "function" in message:
            context_tags.add("function")
        
        if context_tags & rule_tags:
            base_confidence += 0.1
        
        return min(base_confidence, 1.0)


class LuaPatchGenerator:
    """
    Generates patches for Lua errors based on analysis results.
    
    This class creates Lua code fixes for common errors using templates
    and heuristics specific to embedded scripting patterns.
    """
    
    def __init__(self):
        """Initialize the Lua patch generator."""
        self.template_dir = Path(__file__).parent.parent / "patch_generation" / "templates"
        self.lua_template_dir = self.template_dir / "lua"
        
        # Ensure template directory exists
        self.lua_template_dir.mkdir(parents=True, exist_ok=True)
        
        # Load patch templates
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, str]:
        """Load Lua patch templates."""
        templates = {}
        
        if not self.lua_template_dir.exists():
            logger.warning(f"Lua templates directory not found: {self.lua_template_dir}")
            return templates
        
        for template_file in self.lua_template_dir.glob("*.lua.template"):
            try:
                with open(template_file, 'r') as f:
                    template_name = template_file.stem.replace('.lua', '')
                    templates[template_name] = f.read()
                    logger.debug(f"Loaded template: {template_name}")
            except Exception as e:
                logger.error(f"Error loading template {template_file}: {e}")
        
        return templates
    
    def generate_patch(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                      source_code: str = "") -> Optional[Dict[str, Any]]:
        """
        Generate a patch for the Lua error.
        
        Args:
            error_data: The Lua error data
            analysis: Analysis results from LuaExceptionHandler
            source_code: The Lua source code that caused the error
            
        Returns:
            Patch information or None if no patch can be generated
        """
        root_cause = analysis.get("root_cause", "")
        subcategory = analysis.get("subcategory", "")
        
        # Map root causes to patch strategies
        patch_strategies = {
            "lua_syntax_error": self._fix_syntax_error,
            "lua_runtime_error": self._fix_runtime_error,
            "lua_table_error": self._fix_table_error,
            "lua_function_error": self._fix_function_error,
            "lua_variable_error": self._fix_variable_error,
            "lua_metamethod_error": self._fix_metamethod_error,
            "lua_coroutine_error": self._fix_coroutine_error,
            "lua_module_error": self._fix_module_error,
            "lua_io_error": self._fix_io_error,
            "lua_string_error": self._fix_string_error,
            "lua_math_error": self._fix_math_error,
            "lua_nil_access": self._fix_nil_access_error,
            "lua_nil_error": self._fix_nil_access_error,  # Map nil_error to nil_access handler
            "lua_type_error": self._fix_type_error,
            "lua_arithmetic_error": self._fix_arithmetic_error
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
        """Fix Lua syntax errors."""
        message = error_data.get("message", "")
        
        fixes = []
        
        if "end" in message.lower() and "expected" in message.lower():
            fixes.append({
                "type": "suggestion",
                "description": "Missing 'end' keyword",
                "fix": "Add 'end' to close if/while/for/function blocks"
            })
        
        if "=" in message.lower() and "expected" in message.lower():
            fixes.append({
                "type": "suggestion",
                "description": "Missing '=' in assignment",
                "fix": "Add '=' for variable assignment: variable = value"
            })
        
        if "unexpected" in message.lower():
            fixes.append({
                "type": "suggestion",
                "description": "Unexpected symbol",
                "fix": "Check for typos, missing operators, or incorrect syntax"
            })
        
        if "unfinished string" in message.lower():
            fixes.append({
                "type": "suggestion",
                "description": "Unfinished string literal",
                "fix": "Add closing quote to string literal"
            })
        
        if fixes:
            # Return the first fix as a suggestion for simpler interface
            return {
                "type": "suggestion",
                "description": f"Lua syntax error: {fixes[0]['description']}",
                "fix": fixes[0]["fix"]
            }
        
        return {
            "type": "suggestion",
            "description": "Lua syntax error. Check code structure and syntax"
        }
    
    def _fix_runtime_error(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                          source_code: str) -> Optional[Dict[str, Any]]:
        """Fix runtime errors."""
        message = error_data.get("message", "")
        
        if "attempt to" in message.lower() and "nil" in message.lower():
            return {
                "type": "suggestion",
                "description": "Nil value access",
                "fixes": [
                    "Check if value is not nil before using: if value ~= nil then ... end",
                    "Use type() function to check type: if type(value) == 'table' then ... end",
                    "Initialize variables before use",
                    "Use assert() for required values: assert(value, 'value is required')"
                ]
            }
        
        if "stack overflow" in message.lower():
            return {
                "type": "suggestion",
                "description": "Stack overflow error",
                "fixes": [
                    "Check for infinite recursion in functions",
                    "Add base case to recursive functions",
                    "Reduce recursion depth",
                    "Use iterative approach instead of recursion"
                ]
            }
        
        if "memory" in message.lower():
            return {
                "type": "suggestion",
                "description": "Memory allocation error",
                "fixes": [
                    "Reduce memory usage in the program",
                    "Use collectgarbage() to force garbage collection",
                    "Check for memory leaks",
                    "Use more memory-efficient data structures"
                ]
            }
        
        return {
            "type": "suggestion",
            "description": "Runtime error. Check program logic and value handling"
        }
    
    def _fix_table_error(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                        source_code: str) -> Optional[Dict[str, Any]]:
        """Fix table access errors."""
        message = error_data.get("message", "")
        
        if "attempt to index" in message.lower():
            return {
                "type": "suggestion",
                "description": "Table indexing error",
                "fixes": [
                    "Check if table exists: if table ~= nil then ... end",
                    "Verify table is actually a table: if type(table) == 'table' then ... end",
                    "Initialize table before use: table = {} or table = table or {}",
                    "Use rawget() for safe access: rawget(table, key)",
                    "Check key validity before access"
                ]
            }
        
        if "table index is nil" in message.lower():
            return {
                "type": "suggestion",
                "description": "Nil table key",
                "fixes": [
                    "Check key is not nil: if key ~= nil then table[key] = value end",
                    "Use valid key types (string, number, boolean, etc.)",
                    "Avoid using nil as table key",
                    "Use pairs() or ipairs() for safe iteration"
                ]
            }
        
        return {
            "type": "suggestion",
            "description": "Table access error",
            "fixes": [
                "Check table and key validity",
                "Initialize tables before use",
                "Use safe table access patterns"
            ]
        }
    
    def _fix_function_error(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                           source_code: str) -> Optional[Dict[str, Any]]:
        """Fix function errors."""
        message = error_data.get("message", "")
        
        if "attempt to call" in message.lower():
            return {
                "type": "suggestion",
                "description": "Function call error",
                "fixes": [
                    "Check if function exists: if type(func) == 'function' then func() end",
                    "Verify function is defined before calling",
                    "Check function name spelling",
                    "Use pcall() for safe function calls: local ok, result = pcall(func)"
                ]
            }
        
        if "bad argument" in message.lower():
            return {
                "type": "suggestion",
                "description": "Invalid function argument",
                "fixes": [
                    "Check argument types before calling function",
                    "Provide correct number of arguments",
                    "Use type() to validate arguments",
                    "Add argument validation in function"
                ]
            }
        
        if "wrong number of arguments" in message.lower():
            return {
                "type": "suggestion",
                "description": "Incorrect argument count",
                "fixes": [
                    "Check function signature for required parameters",
                    "Provide all required arguments",
                    "Use varargs (...) for variable arguments",
                    "Add default parameter handling"
                ]
            }
        
        return {
            "type": "suggestion",
            "description": "Function error",
            "fixes": [
                "Check function definitions and calls",
                "Verify argument types and counts",
                "Use safe function calling patterns"
            ]
        }
    
    def _fix_variable_error(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                           source_code: str) -> Optional[Dict[str, Any]]:
        """Fix variable scope and access errors."""
        message = error_data.get("message", "")
        
        if "global" in message.lower() and "nil" in message.lower():
            return {
                "type": "suggestion",
                "description": "Global variable not found",
                "fixes": [
                    "Declare global variable before use",
                    "Check variable name spelling",
                    "Use local variables when possible: local variable = value",
                    "Initialize variables properly",
                    "Use _G table for explicit global access: _G.variable"
                ]
            }
        
        if "undefined variable" in message.lower():
            return {
                "type": "suggestion",
                "description": "Undefined variable",
                "fixes": [
                    "Declare variable before use",
                    "Check variable scope",
                    "Use local keyword for local variables",
                    "Initialize variables with proper values"
                ]
            }
        
        return {
            "type": "suggestion",
            "description": "Variable access error",
            "fixes": [
                "Check variable declarations and scope",
                "Initialize variables before use",
                "Use proper variable naming"
            ]
        }
    
    def _fix_metamethod_error(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                             source_code: str) -> Optional[Dict[str, Any]]:
        """Fix metamethod errors."""
        return {
            "type": "suggestion",
            "description": "Metamethod error",
            "fixes": [
                "Check metatable is properly set: setmetatable(table, metatable)",
                "Verify metamethod names (__index, __newindex, etc.)",
                "Ensure metamethods are functions",
                "Check for circular metatable references"
            ]
        }
    
    def _fix_coroutine_error(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                            source_code: str) -> Optional[Dict[str, Any]]:
        """Fix coroutine errors."""
        message = error_data.get("message", "")
        
        if "cannot resume" in message.lower():
            return {
                "type": "suggestion",
                "description": "Coroutine resume error",
                "fixes": [
                    "Check coroutine status: coroutine.status(co)",
                    "Ensure coroutine is suspended before resuming",
                    "Create coroutine properly: coroutine.create(func)",
                    "Handle coroutine errors with pcall"
                ]
            }
        
        if "cannot yield" in message.lower():
            return {
                "type": "suggestion",
                "description": "Coroutine yield error",
                "fixes": [
                    "Only yield from inside coroutine",
                    "Check if in coroutine context",
                    "Use proper coroutine patterns",
                    "Avoid yielding from main thread"
                ]
            }
        
        return {
            "type": "suggestion",
            "description": "Coroutine error",
            "fixes": [
                "Check coroutine creation and management",
                "Use proper yield/resume patterns",
                "Handle coroutine states correctly"
            ]
        }
    
    def _fix_module_error(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                         source_code: str) -> Optional[Dict[str, Any]]:
        """Fix module loading errors."""
        message = error_data.get("message", "")
        
        if "module not found" in message.lower():
            return {
                "type": "suggestion",
                "description": "Module not found",
                "fixes": [
                    "Check module path and filename",
                    "Verify module exists in package.path",
                    "Use correct require syntax: require('module')",
                    "Check for typos in module name",
                    "Add module directory to package.path"
                ]
            }
        
        if "loop in require" in message.lower():
            return {
                "type": "suggestion",
                "description": "Circular module dependency",
                "fixes": [
                    "Restructure modules to avoid circular dependencies",
                    "Use lazy loading for some modules",
                    "Move shared code to separate module",
                    "Use forward declarations where possible"
                ]
            }
        
        return {
            "type": "suggestion",
            "description": "Module loading error",
            "fixes": [
                "Check module paths and names",
                "Verify module structure",
                "Use proper require patterns"
            ]
        }
    
    def _fix_io_error(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                     source_code: str) -> Optional[Dict[str, Any]]:
        """Fix IO and file handling errors."""
        message = error_data.get("message", "")
        
        if "cannot open" in message.lower():
            return {
                "type": "suggestion",
                "description": "File open error",
                "fixes": [
                    "Check if file exists before opening",
                    "Verify file path is correct",
                    "Check file permissions",
                    "Use proper file open modes ('r', 'w', 'a')",
                    "Handle file errors with pcall or check return values"
                ]
            }
        
        if "permission denied" in message.lower():
            return {
                "type": "suggestion",
                "description": "File permission error",
                "fixes": [
                    "Check file permissions",
                    "Run with appropriate privileges",
                    "Use different file location",
                    "Change file permissions if needed"
                ]
            }
        
        return {
            "type": "suggestion",
            "description": "IO error",
            "fixes": [
                "Check file paths and permissions",
                "Handle file operations safely",
                "Use proper error checking"
            ]
        }
    
    def _fix_string_error(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                         source_code: str) -> Optional[Dict[str, Any]]:
        """Fix string manipulation errors."""
        message = error_data.get("message", "")
        
        if "invalid pattern" in message.lower():
            return {
                "type": "suggestion",
                "description": "Invalid pattern error",
                "fixes": [
                    "Check pattern syntax for string functions",
                    "Escape special characters in patterns",
                    "Use correct pattern syntax for match, gsub, etc.",
                    "Test patterns with simple cases first"
                ]
            }
        
        if "bad argument" in message.lower() and "string" in message.lower():
            return {
                "type": "suggestion",
                "description": "String function argument error",
                "fixes": [
                    "Check argument types for string functions",
                    "Convert values to strings with tostring()",
                    "Verify string function parameters",
                    "Use type() to validate arguments"
                ]
            }
        
        return {
            "type": "suggestion",
            "description": "String manipulation error",
            "fixes": [
                "Check string function usage",
                "Verify string patterns and arguments",
                "Use proper string conversion"
            ]
        }
    
    def _fix_math_error(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                       source_code: str) -> Optional[Dict[str, Any]]:
        """Fix mathematical operation errors."""
        message = error_data.get("message", "")
        
        if "domain error" in message.lower():
            return {
                "type": "suggestion",
                "description": "Math domain error",
                "fixes": [
                    "Check input values are within valid range",
                    "Validate arguments before math operations",
                    "Handle special cases (negative numbers for sqrt, etc.)",
                    "Use appropriate math functions"
                ]
            }
        
        if "division by zero" in message.lower():
            return {
                "type": "suggestion",
                "description": "Division by zero error",
                "fixes": [
                    "Check divisor is not zero before division",
                    "Add validation: if divisor ~= 0 then result = a / divisor end",
                    "Handle zero division case appropriately",
                    "Use math.huge for infinity if needed"
                ]
            }
        
        return {
            "type": "suggestion",
            "description": "Math operation error",
            "fixes": [
                "Check mathematical operation validity",
                "Validate numeric arguments",
                "Handle edge cases in calculations"
            ]
        }
    
    def _fix_nil_access_error(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                             source_code: str) -> Optional[Dict[str, Any]]:
        """Fix nil access errors."""
        return {
            "type": "suggestion",
            "description": "Nil access error",
            "fixes": [
                "Check for nil before use: if value ~= nil then ... end",
                "Use type() function: if type(value) == 'table' then ... end",
                "Initialize variables properly",
                "Use assert() for required values: assert(value, 'value required')",
                "Use logical operators: local result = value or default_value"
            ]
        }
    
    def _fix_type_error(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                       source_code: str) -> Optional[Dict[str, Any]]:
        """Fix type errors."""
        message = error_data.get("message", "")
        
        if "concatenate" in message.lower():
            return {
                "type": "suggestion",
                "description": "Type conversion error in concatenation",
                "fixes": [
                    "Convert to string before concatenation: tostring(value)",
                    "Check type before concatenation: if type(value) == 'string' then",
                    "Use string.format for complex formatting: string.format('%s', value)",
                    "Handle nil values: local str = value or ''",
                    "Use table.concat for multiple values"
                ]
            }
        
        return {
            "type": "suggestion",
            "description": "Type error - check value types",
            "fixes": [
                "Check type before operations: type(value)",
                "Convert types appropriately: tonumber(), tostring()",
                "Validate input types",
                "Use proper type coercion"
            ]
        }
    
    def _fix_arithmetic_error(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                             source_code: str) -> Optional[Dict[str, Any]]:
        """Fix arithmetic errors."""
        message = error_data.get("message", "")
        
        if "string" in message.lower():
            return {
                "type": "suggestion",
                "description": "Arithmetic operation on non-numeric value",
                "fixes": [
                    "Convert to number: tonumber(value)",
                    "Check if numeric: if tonumber(value) then",
                    "Validate numeric input: assert(tonumber(value), 'numeric value required')",
                    "Handle conversion failures: local num = tonumber(value) or 0",
                    "Use type checking: if type(value) == 'number' then"
                ]
            }
        
        return {
            "type": "suggestion",
            "description": "Arithmetic operation error",
            "fixes": [
                "Ensure numeric operands: tonumber(value)",
                "Check for nil values before arithmetic",
                "Validate numeric types",
                "Handle edge cases (division by zero, overflow)"
            ]
        }
    
    def _template_based_patch(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                            source_code: str) -> Optional[Dict[str, Any]]:
        """Generate patch using templates."""
        root_cause = analysis.get("root_cause", "")
        subcategory = analysis.get("subcategory", "")
        
        # Map root causes to template names
        template_map = {
            "lua_syntax_error": "syntax_fix",
            "lua_runtime_error": "runtime_fix",
            "lua_nil_access": "nil_fix",
            "lua_table_error": "table_fix",
            "lua_function_error": "function_fix",
            "lua_module_error": "module_fix"
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


class LuaLanguagePlugin(LanguagePlugin):
    """
    Main Lua language plugin for Homeostasis.
    
    This plugin orchestrates Lua error analysis and patch generation,
    supporting embedded scripting and automation patterns.
    """
    
    VERSION = "1.0.0"
    AUTHOR = "Homeostasis Team"
    
    def __init__(self):
        """Initialize the Lua language plugin."""
        self.language = "lua"
        self.supported_extensions = {".lua", ".luau"}
        self.supported_frameworks = [
            "lua", "luarocks", "love2d", "openresty", "nginx_lua", "wireshark", "world_of_warcraft"
        ]
        
        # Initialize components
        self.exception_handler = LuaExceptionHandler()
        self.patch_generator = LuaPatchGenerator()
        
        logger.info("Lua language plugin initialized")
    
    def get_language_id(self) -> str:
        """Get the unique identifier for this language."""
        return "lua"
    
    def get_language_name(self) -> str:
        """Get the human-readable name of the language."""
        return "Lua"
    
    def get_language_version(self) -> str:
        """Get the version of the language supported by this plugin."""
        return "5.4+"
    
    def get_supported_frameworks(self) -> List[str]:
        """Get the list of frameworks supported by this language plugin."""
        return self.supported_frameworks
    
    def normalize_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize error data to the standard Homeostasis format.
        
        Args:
            error_data: Error data in the Lua-specific format
            
        Returns:
            Error data in the standard format
        """
        # Map Lua-specific error fields to standard format
        normalized = {
            "error_type": error_data.get("error_type", "LuaError"),
            "message": error_data.get("message", error_data.get("description", "")),
            "language": "lua",
            "file_path": error_data.get("file_path", error_data.get("file", "")),
            "line_number": error_data.get("line_number", error_data.get("line", 0)),
            "column_number": error_data.get("column_number", error_data.get("column", 0)),
            "lua_version": error_data.get("lua_version", ""),
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
        Convert standard format error data back to the Lua-specific format.
        
        Args:
            standard_error: Error data in the standard format
            
        Returns:
            Error data in the Lua-specific format
        """
        # Map standard fields back to Lua-specific format
        lua_error = {
            "error_type": standard_error.get("error_type", "LuaError"),
            "message": standard_error.get("message", ""),
            "file_path": standard_error.get("file_path", ""),
            "line_number": standard_error.get("line_number", 0),
            "column_number": standard_error.get("column_number", 0),
            "lua_version": standard_error.get("lua_version", ""),
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
            if key not in lua_error and value is not None:
                lua_error[key] = value
        
        return lua_error
    
    def analyze_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a Lua error.
        
        Args:
            error_data: Lua error data
            
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
            analysis["plugin"] = "lua"
            analysis["language"] = "lua"
            analysis["plugin_version"] = self.VERSION
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing Lua error: {e}")
            return {
                "category": "lua",
                "subcategory": "unknown",
                "confidence": "low",
                "suggested_fix": "Unable to analyze Lua error",
                "error": str(e),
                "plugin": "lua"
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
register_plugin(LuaLanguagePlugin())