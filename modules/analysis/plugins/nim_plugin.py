"""
Nim Language Plugin for Homeostasis

This plugin enables Homeostasis to analyze and fix errors in Nim programming language code.
It provides comprehensive error handling for Nim compilation errors, runtime issues,
memory management problems, and systems programming best practices.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..language_plugin_system import LanguagePlugin, register_plugin

logger = logging.getLogger(__name__)


class NimExceptionHandler:
    """
    Handles Nim exceptions with robust error detection and classification.

    This class provides logic for categorizing Nim errors based on their type,
    message, and common systems programming patterns.
    """

    def __init__(self):
        """Initialize the Nim exception handler."""
        self.rule_categories = {
            "syntax": "Nim syntax and parsing errors",
            "compilation": "Compilation and build errors",
            "type": "Type system and type checking errors",
            "memory": "Memory management and allocation errors",
            "runtime": "Runtime errors and exceptions",
            "import": "Import and module system errors",
            "macro": "Macro and template system errors",
            "proc": "Procedure and function definition errors",
            "async": "Async and concurrent programming errors",
            "nim_check": "Nim static analysis errors",
            "ffi": "Foreign function interface errors",
            "generic": "Generic type and template errors",
        }

        # Common Nim error patterns
        self.nim_error_patterns = {
            "syntax_error": [
                r"Error: expected.*?, got",
                r"Error: invalid indentation",
                r"Error: unexpected token",
                r"Error: invalid syntax",
                r"Error: expression expected",
                r"Error: statement expected",
                r"Error: closing.*?expected",
            ],
            "undefined_error": [
                r"Error: undeclared identifier",
                r"Error: undefined reference",
                r"Error: symbol.*?not found",
                r"Error: unknown identifier",
            ],
            "type_error": [
                r"Error: type mismatch: got.*?but expected",
                r"Error: cannot convert.*?to",
                r"Error: type '.*?' doesn't have a correct constructor",
                r"Error: type '.*?' has no member named",
                r"Error: invalid type.*?in this context",
                r"Error: type '.*?' cannot be instantiated",
                r"Error: ambiguous call",
            ],
            "compilation_error": [
                r"Error: cannot open file",
                r"Error: module.*?not found",
                r"Error: undeclared identifier",
                r"Error: redefinition of",
                r"Error: '.*?' is not accessible",
                r"Error: recursive dependency",
                r"Error: internal error",
            ],
            "memory_error": [
                r"Error: out of memory",
                r"Error: access violation",
                r"Error: unhandled exception: NilAccessDefect",
                r"Error: index out of bounds",
                r"Error: stack overflow",
                r"Error: invalid memory access",
            ],
            "runtime_error": [
                r"Error: unhandled exception",
                r"Error: assertion failed",
                r"Error: system.*?error",
                r"Error: resource.*?not found",
                r"Error: division by zero",
                r"Error: invalid.*?operation",
            ],
            "import_error": [
                r"Error: cannot import",
                r"Error: module.*?not found",
                r"Error: cannot open.*?module",
                r"Error: circular dependency",
                r"Error: import.*?failed",
            ],
            "macro_error": [
                r"Error: macro.*?not found",
                r"Error: template.*?not found",
                r"Error: macro.*?instantiation",
                r"Error: invalid.*?macro",
                r"Error: template.*?instantiation",
            ],
            "proc_error": [
                r"Error: procedure.*?not found",
                r"Error: ambiguous call",
                r"Error: cannot call.*?procedure",
                r"Error: procedure.*?expects.*?arguments",
                r"Error: overloaded.*?procedure",
            ],
            "async_error": [
                r"Error: async.*?procedure",
                r"Error: cannot use.*?in async context",
                r"Error: future.*?not completed",
                r"Error: await.*?in non-async procedure",
            ],
        }

        # Nim-specific concepts and their common issues
        self.nim_concepts = {
            "nil_access": ["nil pointer", "nil dereference", "NilAccessDefect"],
            "ranges": ["index out of bounds", "range check", "invalid range"],
            "seqs": ["sequence index", "seq bounds", "sequence access"],
            "optionals": ["option type", "some/none", "optional access"],
            "result": ["result type", "error handling", "result access"],
            "references": ["ref type", "reference access", "ref assignment"],
            "strings": ["string index", "string access", "string conversion"],
            "arrays": ["array bounds", "array access", "array assignment"],
            "objects": ["object initialization", "object access", "field access"],
            "variants": ["variant access", "case object", "variant assignment"],
        }

        # Load rules from different categories
        self.rules = self._load_rules()

        # Pre-compile regex patterns for better performance
        self._compile_patterns()

    def _load_rules(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load Nim error rules from rule files."""
        rules = {}
        rules_dir = Path(__file__).parent.parent / "rules" / "nim"

        try:
            # Load common Nim rules
            common_rules_path = rules_dir / "nim_common_errors.json"
            if common_rules_path.exists():
                with open(common_rules_path, "r") as f:
                    common_data = json.load(f)
                    rules["common"] = common_data.get("rules", [])
                    logger.info(f"Loaded {len(rules['common'])} common Nim rules")

            # Load concept-specific rules
            for concept in ["memory", "types", "async", "macros", "pragma"]:
                concept_rules_path = rules_dir / f"nim_{concept}_errors.json"
                if concept_rules_path.exists():
                    with open(concept_rules_path, "r") as f:
                        concept_data = json.load(f)
                        rules[concept] = concept_data.get("rules", [])
                        logger.info(f"Loaded {len(rules[concept])} {concept} rules")

        except Exception as e:
            logger.error(f"Error loading Nim rules: {e}")
            rules = {"common": []}

        return rules

    def _compile_patterns(self):
        """Pre-compile regex patterns for better performance."""
        self.compiled_patterns: Dict[str, List[tuple[re.Pattern[str], Dict[str, Any]]]] = {}

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
        Analyze a Nim exception and determine its type and potential fixes.

        Args:
            error_data: Nim error data in standard format

        Returns:
            Analysis results with categorization and fix suggestions
        """
        message = error_data.get("message", "")
        file_path = error_data.get("file_path", "")
        line_number = error_data.get("line_number", 0)
        column_number = error_data.get("column_number", 0)

        # Analyze based on error patterns
        analysis = self._analyze_by_patterns(message, file_path)

        # Check for concept-specific issues
        concept_analysis = self._analyze_nim_concepts(message)
        if concept_analysis.get("confidence", "low") != "low":
            # Merge concept-specific findings
            analysis.update(concept_analysis)

        # Find matching rules
        matches = self._find_matching_rules(message, error_data)

        if matches:
            # Use the best match (highest confidence)
            best_match = max(matches, key=lambda x: x.get("confidence_score", 0))

            # Map rule types to subcategories
            rule_type = best_match.get("type", "")
            subcategory_map = {
                "TypeError": "type",
                "ImportError": "import",
                "SyntaxError": "syntax",
                "ProcedureError": "proc",
                "MacroError": "macro",
                "RuntimeError": "runtime",
                "CompilationError": "compilation",
                "PragmaError": "pragma",
            }

            # Check for specific subcategories based on root cause
            root_cause = best_match.get("root_cause", "")
            if "nil_access" in root_cause:
                subcategory = "nil_access"
            elif "undefined" in root_cause:
                subcategory = "undefined"
            elif "option_error" in root_cause:
                subcategory = "optionals"
            elif "result_error" in root_cause:
                subcategory = "result"
            elif "bounds_error" in root_cause:
                subcategory = "bounds"
            elif "pragma" in root_cause:
                subcategory = "pragma"
            else:
                subcategory = subcategory_map.get(
                    rule_type, analysis.get("subcategory", "unknown")
                )

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

        analysis["file_path"] = file_path
        analysis["line_number"] = line_number
        analysis["column_number"] = column_number
        return analysis

    def _analyze_by_patterns(self, message: str, file_path: str) -> Dict[str, Any]:
        """Analyze error by matching against common patterns."""
        # Check syntax errors
        for pattern in self.nim_error_patterns["syntax_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "nim",
                    "subcategory": "syntax",
                    "confidence": "high",
                    "suggested_fix": "Fix Nim syntax errors",
                    "root_cause": "nim_syntax_error",
                    "severity": "high",
                    "tags": ["nim", "syntax", "parser"],
                }

        # Check undefined identifier errors
        if "undefined_error" in self.nim_error_patterns:
            for pattern in self.nim_error_patterns["undefined_error"]:
                if re.search(pattern, message, re.IGNORECASE):
                    return {
                        "category": "nim",
                        "subcategory": "undefined",
                        "confidence": "high",
                        "suggested_fix": "Define the identifier or import required module",
                        "root_cause": "nim_undefined_identifier",
                        "severity": "high",
                        "tags": ["nim", "undefined", "identifier"],
                    }

        # Check type errors
        for pattern in self.nim_error_patterns["type_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "nim",
                    "subcategory": "type",
                    "confidence": "high",
                    "suggested_fix": "Fix type system errors and type mismatches",
                    "root_cause": "nim_type_error",
                    "severity": "high",
                    "tags": ["nim", "type", "casting"],
                }

        # Check compilation errors
        for pattern in self.nim_error_patterns["compilation_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "nim",
                    "subcategory": "compilation",
                    "confidence": "high",
                    "suggested_fix": "Fix compilation and build errors",
                    "root_cause": "nim_compilation_error",
                    "severity": "high",
                    "tags": ["nim", "compilation", "build"],
                }

        # Check memory errors
        for pattern in self.nim_error_patterns["memory_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "nim",
                    "subcategory": "memory",
                    "confidence": "high",
                    "suggested_fix": "Fix memory management and allocation issues",
                    "root_cause": "nim_memory_error",
                    "severity": "critical",
                    "tags": ["nim", "memory", "safety"],
                }

        # Check runtime errors
        for pattern in self.nim_error_patterns["runtime_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "nim",
                    "subcategory": "runtime",
                    "confidence": "high",
                    "suggested_fix": "Fix runtime errors and exceptions",
                    "root_cause": "nim_runtime_error",
                    "severity": "high",
                    "tags": ["nim", "runtime", "exception"],
                }

        # Check import errors
        for pattern in self.nim_error_patterns["import_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "nim",
                    "subcategory": "import",
                    "confidence": "high",
                    "suggested_fix": "Fix import paths and module dependencies",
                    "root_cause": "nim_import_error",
                    "severity": "high",
                    "tags": ["nim", "import", "module"],
                }

        # Check macro errors
        for pattern in self.nim_error_patterns["macro_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "nim",
                    "subcategory": "macro",
                    "confidence": "high",
                    "suggested_fix": "Fix macro and template errors",
                    "root_cause": "nim_macro_error",
                    "severity": "medium",
                    "tags": ["nim", "macro", "template"],
                }

        # Check procedure errors
        for pattern in self.nim_error_patterns["proc_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "nim",
                    "subcategory": "proc",
                    "confidence": "high",
                    "suggested_fix": "Fix procedure and function errors",
                    "root_cause": "nim_proc_error",
                    "severity": "high",
                    "tags": ["nim", "proc", "function"],
                }

        # Check async errors
        for pattern in self.nim_error_patterns["async_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "nim",
                    "subcategory": "async",
                    "confidence": "high",
                    "suggested_fix": "Fix async/await usage and concurrency issues",
                    "root_cause": "nim_async_error",
                    "severity": "medium",
                    "tags": ["nim", "async", "concurrency"],
                }

        return {
            "category": "nim",
            "subcategory": "unknown",
            "confidence": "low",
            "suggested_fix": "Review Nim code and compiler error details",
            "root_cause": "nim_generic_error",
            "severity": "medium",
            "tags": ["nim", "generic"],
        }

    def _analyze_nim_concepts(self, message: str) -> Dict[str, Any]:
        """Analyze Nim-specific concept errors."""
        message_lower = message.lower()

        # Check for nil access errors
        if any(
            keyword in message_lower for keyword in ["nil", "nilaccessdefect", "null"]
        ):
            return {
                "category": "nim",
                "subcategory": "nil_access",
                "confidence": "high",
                "suggested_fix": "Add nil check before accessing reference",
                "root_cause": "nim_nil_access",
                "severity": "high",
                "tags": ["nim", "nil", "safety"],
            }

        # Check for range/bounds errors
        if any(keyword in message_lower for keyword in ["index", "bounds", "range"]):
            return {
                "category": "nim",
                "subcategory": "bounds",
                "confidence": "high",
                "suggested_fix": "Check array/sequence bounds before access",
                "root_cause": "nim_bounds_error",
                "severity": "high",
                "tags": ["nim", "bounds", "safety"],
            }

        # Check for sequence-related errors
        if any(keyword in message_lower for keyword in ["seq", "sequence"]):
            return {
                "category": "nim",
                "subcategory": "seqs",
                "confidence": "medium",
                "suggested_fix": "Check sequence operations and bounds",
                "root_cause": "nim_seq_error",
                "severity": "medium",
                "tags": ["nim", "seq", "collection"],
            }

        # Check for option type errors (be more specific)
        if any(
            keyword in message_lower
            for keyword in ["option[", "option ", ".some", ".none", "issome", "isnone"]
        ):
            return {
                "category": "nim",
                "subcategory": "optionals",
                "confidence": "medium",
                "suggested_fix": "Handle option types properly with some/none checks",
                "root_cause": "nim_option_error",
                "severity": "medium",
                "tags": ["nim", "option", "safety"],
            }

        # Check for result type errors (be more specific to avoid matching generic "error" word)
        if any(
            keyword in message_lower
            for keyword in [
                "result[",
                "result type",
                "result value",
                ".error",
                ".iserr",
            ]
        ):
            return {
                "category": "nim",
                "subcategory": "result",
                "confidence": "medium",
                "suggested_fix": "Handle result types with proper error checking",
                "root_cause": "nim_result_error",
                "severity": "medium",
                "tags": ["nim", "result", "error_handling"],
            }

        # Check for reference type errors
        if any(keyword in message_lower for keyword in ["ref", "reference"]):
            return {
                "category": "nim",
                "subcategory": "references",
                "confidence": "medium",
                "suggested_fix": "Check reference type usage and assignments",
                "root_cause": "nim_ref_error",
                "severity": "medium",
                "tags": ["nim", "ref", "memory"],
            }

        return {"confidence": "low"}

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

        # Boost confidence for file extension matches
        file_path = error_data.get("file_path", "")
        if file_path.endswith(".nim"):
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
        if "async" in message:
            context_tags.add("async")
        if "option" in message:
            context_tags.add("option")

        if context_tags & rule_tags:
            base_confidence += 0.1

        return min(base_confidence, 1.0)


class NimPatchGenerator:
    """
    Generates patches for Nim errors based on analysis results.

    This class creates Nim code fixes for common errors using templates
    and heuristics specific to systems programming patterns.
    """

    def __init__(self):
        """Initialize the Nim patch generator."""
        self.template_dir = (
            Path(__file__).parent.parent / "patch_generation" / "templates"
        )
        self.nim_template_dir = self.template_dir / "nim"

        # Ensure template directory exists
        self.nim_template_dir.mkdir(parents=True, exist_ok=True)

        # Load patch templates
        self.templates: Dict[str, str] = self._load_templates()

    def _load_templates(self) -> Dict[str, str]:
        """Load Nim patch templates."""
        templates: Dict[str, str] = {}

        if not self.nim_template_dir.exists():
            logger.warning(
                f"Nim templates directory not found: {self.nim_template_dir}"
            )
            return templates

        for template_file in self.nim_template_dir.glob("*.nim.template"):
            try:
                with open(template_file, "r") as f:
                    template_name = template_file.stem.replace(".nim", "")
                    templates[template_name] = f.read()
                    logger.debug(f"Loaded template: {template_name}")
            except Exception as e:
                logger.error(f"Error loading template {template_file}: {e}")

        return templates

    def generate_patch(
        self,
        error_data: Dict[str, Any],
        analysis: Dict[str, Any],
        source_code: str = "",
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a patch for the Nim error.

        Args:
            error_data: The Nim error data
            analysis: Analysis results from NimExceptionHandler
            source_code: The Nim source code that caused the error

        Returns:
            Patch information or None if no patch can be generated
        """
        root_cause = analysis.get("root_cause", "")

        # Map root causes to patch strategies
        patch_strategies = {
            "nim_syntax_error": self._fix_syntax_error,
            "nim_type_error": self._fix_type_error,
            "nim_compilation_error": self._fix_compilation_error,
            "nim_memory_error": self._fix_memory_error,
            "nim_runtime_error": self._fix_runtime_error,
            "nim_import_error": self._fix_import_error,
            "nim_macro_error": self._fix_macro_error,
            "nim_proc_error": self._fix_proc_error,
            "nim_async_error": self._fix_async_error,
            "nim_nil_access": self._fix_nil_access_error,
            "nim_bounds_error": self._fix_bounds_error,
            "nim_option_error": self._fix_option_error,
            "nim_result_error": self._fix_result_error,
            "nim_undefined_identifier": self._fix_undefined_error,
            "nim_pragma_error": self._fix_pragma_error,
        }

        strategy = patch_strategies.get(root_cause)
        if strategy:
            try:
                return strategy(error_data, analysis, source_code)
            except Exception as e:
                logger.error(f"Error generating patch for {root_cause}: {e}")

        # Try to use templates if no specific strategy matches
        return self._template_based_patch(error_data, analysis, source_code)

    def _fix_syntax_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix Nim syntax errors."""
        message = error_data.get("message", "")

        fixes = []

        if "expected" in message.lower() and "got" in message.lower():
            # Extract expected and got tokens
            expected_match = re.search(r"expected (.+?), got", message)
            got_match = re.search(r"got (.+)", message)

            if expected_match and got_match:
                expected = expected_match.group(1)
                got = got_match.group(1)
                fixes.append(
                    {
                        "type": "suggestion",
                        "description": f"Expected {expected}, but got {got}",
                        "fix": f"Replace '{got}' with '{expected}' or fix the syntax structure",
                    }
                )

        if "invalid indentation" in message.lower():
            return {
                "type": "suggestion",
                "description": "Fix indentation - Nim uses significant whitespace (2 spaces per level)",
            }

        if "unexpected token" in message.lower():
            fixes.append(
                {
                    "type": "suggestion",
                    "description": "Unexpected token found",
                    "fix": "Check for syntax errors like missing colons, operators, or keywords",
                }
            )

        if fixes:
            return {
                "type": "multiple_suggestions",
                "fixes": fixes,
                "description": "Nim syntax error fixes",
            }

        return {
            "type": "suggestion",
            "description": "Nim syntax error. Check code structure and syntax",
        }

    def _fix_type_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix Nim type errors."""
        message = error_data.get("message", "")

        if "type mismatch" in message.lower():
            # Extract type information
            type_match = re.search(r"got (.+?) but expected (.+)", message)
            if type_match:
                got_type = type_match.group(1)
                expected_type = type_match.group(2)

                return {
                    "type": "suggestion",
                    "description": f"Type mismatch: got {got_type}, expected {expected_type}",
                    "fixes": [
                        f"Convert value to {expected_type} using appropriate conversion",
                        f"Change variable type to {got_type}",
                        f"Use type casting: {expected_type}(value)",
                        "Check if types are compatible or add explicit conversion",
                    ],
                }

        if "cannot convert" in message.lower():
            return {
                "type": "suggestion",
                "description": "Type conversion error",
                "fixes": [
                    "Use explicit type conversion functions",
                    "Check if conversion is valid for the types involved",
                    "Use $ for string conversion or parseInt/parseFloat for numbers",
                    "Consider using type casting with appropriate checks",
                ],
            }

        if "ambiguous call" in message.lower():
            return {
                "type": "suggestion",
                "description": "Ambiguous procedure call",
                "fixes": [
                    "Specify types explicitly in procedure call",
                    "Use fully qualified procedure names",
                    "Add type annotations to disambiguate",
                    "Check for overlapping procedure signatures",
                ],
            }

        return {
            "type": "suggestion",
            "description": "Type system error. Check type compatibility and conversions",
        }

    def _fix_compilation_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix compilation errors."""
        message = error_data.get("message", "")

        if "cannot open file" in message.lower():
            return {
                "type": "suggestion",
                "description": "File not found during compilation",
                "fixes": [
                    "Check if file path is correct",
                    "Ensure file exists and is accessible",
                    "Check file permissions",
                    "Verify import path is relative to current file",
                ],
            }

        if "undeclared identifier" in message.lower():
            # Extract identifier name
            identifier_match = re.search(r"undeclared identifier '(.+?)'", message)
            identifier = identifier_match.group(1) if identifier_match else "identifier"

            return {
                "type": "suggestion",
                "description": f"Undeclared identifier '{identifier}'",
                "fixes": [
                    f"Declare variable or procedure '{identifier}'",
                    f"Check spelling of '{identifier}'",
                    f"Import module containing '{identifier}'",
                    f"Add variable declaration: var {identifier}: Type = value",
                    "Check if identifier is in scope",
                ],
            }

        if "redefinition" in message.lower():
            return {
                "type": "suggestion",
                "description": "Redefinition of identifier",
                "fixes": [
                    "Remove duplicate definition",
                    "Rename one of the conflicting identifiers",
                    "Check for multiple imports of the same name",
                    "Use qualified imports to avoid name conflicts",
                ],
            }

        return {
            "type": "suggestion",
            "description": "Compilation error. Check code structure and dependencies",
        }

    def _fix_memory_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix memory management errors."""
        message = error_data.get("message", "")

        if "out of memory" in message.lower():
            return {
                "type": "suggestion",
                "description": "Out of memory error",
                "fixes": [
                    "Reduce memory usage in the program",
                    "Check for memory leaks or excessive allocations",
                    "Use more memory-efficient data structures",
                    "Consider using streaming or chunked processing",
                ],
            }

        if "access violation" in message.lower():
            return {
                "type": "suggestion",
                "description": "Memory access violation",
                "fixes": [
                    "Check for buffer overruns or underruns",
                    "Verify array/sequence bounds before access",
                    "Check pointer validity before dereferencing",
                    "Use safe array access with bounds checking",
                ],
            }

        return {
            "type": "suggestion",
            "description": "Memory management error",
            "fixes": [
                "Check memory allocation and deallocation",
                "Verify pointer and reference usage",
                "Use safe memory access patterns",
            ],
        }

    def _fix_runtime_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix runtime errors."""
        message = error_data.get("message", "")

        if "unhandled exception" in message.lower():
            return {
                "type": "suggestion",
                "description": "Unhandled exception",
                "fixes": [
                    "Add try-except block to handle exceptions",
                    "Use appropriate exception handling for the error type",
                    "Check for potential error conditions before they occur",
                    "Add proper error handling and recovery logic",
                ],
            }

        if "assertion failed" in message.lower():
            return {
                "type": "suggestion",
                "description": "Assertion failed",
                "fixes": [
                    "Check the condition that caused the assertion to fail",
                    "Verify input data meets the expected criteria",
                    "Add proper validation before the assertion",
                    "Consider using conditional checks instead of assertions",
                ],
            }

        return {
            "type": "suggestion",
            "description": "Runtime error. Check program logic and input validation",
        }

    def _fix_import_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix import and module errors."""
        message = error_data.get("message", "")

        if "cannot import" in message.lower():
            # Extract module name
            module_match = re.search(r"cannot import (.+)", message)
            module_name = module_match.group(1) if module_match else "module"

            return {
                "type": "suggestion",
                "description": f"Cannot import '{module_name}'",
                "fixes": [
                    f"Check if module '{module_name}' exists",
                    f"Verify the path to '{module_name}' is correct",
                    "Use relative imports from current module",
                    "Check if module is in Nim's module path",
                    "Install missing packages with nimble install",
                ],
            }

        if "circular dependency" in message.lower():
            return {
                "type": "suggestion",
                "description": "Circular dependency detected",
                "fixes": [
                    "Restructure modules to avoid circular imports",
                    "Move shared code to a separate module",
                    "Use forward declarations where possible",
                    "Consider using interfaces or abstract types",
                ],
            }

        return {
            "type": "suggestion",
            "description": "Import/module error",
            "fixes": [
                "Check import paths and module structure",
                "Verify module exists and is accessible",
                "Review module dependencies",
            ],
        }

    def _fix_macro_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix macro and template errors."""
        return {
            "type": "suggestion",
            "description": "Macro/template error",
            "fixes": [
                "Check macro/template syntax and parameters",
                "Verify macro is properly defined and accessible",
                "Check for correct macro instantiation",
                "Review template argument types and constraints",
            ],
        }

    def _fix_proc_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix procedure and function errors."""
        message = error_data.get("message", "")

        if "procedure not found" in message.lower():
            return {
                "type": "suggestion",
                "description": "Procedure not found",
                "fixes": [
                    "Check procedure name spelling",
                    "Verify procedure is defined and accessible",
                    "Import module containing the procedure",
                    "Check procedure signature matches the call",
                ],
            }

        if "expects" in message.lower() and "arguments" in message.lower():
            return {
                "type": "suggestion",
                "description": "Incorrect number of arguments",
                "fixes": [
                    "Check procedure signature for required parameters",
                    "Provide all required arguments",
                    "Use default parameters if available",
                    "Check for overloaded procedures with different signatures",
                ],
            }

        return {
            "type": "suggestion",
            "description": "Procedure/function error",
            "fixes": [
                "Check procedure definitions and calls",
                "Verify argument types and counts",
                "Review procedure accessibility and imports",
            ],
        }

    def _fix_async_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix async/await errors."""
        message = error_data.get("message", "")

        if "async procedure" in message.lower():
            return {
                "type": "suggestion",
                "description": "Async procedure error",
                "fixes": [
                    "Use await when calling async procedures",
                    "Call async procedures from async context",
                    "Check async procedure signatures",
                    "Handle Future types properly",
                ],
            }

        if "await" in message.lower() and "non-async" in message.lower():
            return {
                "type": "suggestion",
                "description": "Await in non-async procedure",
                "fixes": [
                    "Make procedure async: proc name(): Future[Type] {.async.}",
                    "Remove await and call procedure synchronously",
                    "Use waitFor for synchronous waiting",
                ],
            }

        return {
            "type": "suggestion",
            "description": "Async/concurrency error",
            "fixes": [
                "Check async/await usage and procedure signatures",
                "Ensure proper async context and Future handling",
                "Review concurrency patterns and threading",
            ],
        }

    def _fix_nil_access_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix nil access errors."""
        return {
            "type": "suggestion",
            "description": "Nil access error",
            "fixes": [
                "Check for nil before dereferencing: if obj != nil: ...",
                "Use safe dereference operators where available",
                "Initialize references properly before use",
                "Use Option types for nullable references",
            ],
        }

    def _fix_bounds_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix array/sequence bounds errors."""
        return {
            "type": "suggestion",
            "description": "Array/sequence bounds error",
            "fixes": [
                "Check index is within bounds: if index < len(seq): ...",
                "Use high() to get maximum valid index",
                "Validate indices before array/sequence access",
                "Use length checks to prevent out-of-bounds access",
            ],
        }

    def _fix_option_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix option type errors."""
        return {
            "type": "suggestion",
            "description": "Option type error",
            "fixes": [
                "Check option has value: if option.isSome: ...",
                "Use get() to extract value from Some",
                "Handle None case explicitly",
                "Use getOrDefault() for default values",
            ],
        }

    def _fix_result_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix result type errors."""
        return {
            "type": "suggestion",
            "description": "Result type error",
            "fixes": [
                "Check result is Ok: if result.isOk: ...",
                "Use get() to extract value from Ok",
                "Handle error case: if result.isErr: ...",
                "Use proper error handling patterns",
            ],
        }

    def _fix_undefined_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix undefined identifier errors."""
        message = error_data.get("message", "")

        # Extract identifier name
        identifier_match = re.search(r"undeclared identifier: '(.+?)'", message)
        identifier = identifier_match.group(1) if identifier_match else "identifier"

        return {
            "type": "suggestion",
            "description": f"Undefined identifier '{identifier}'",
            "fixes": [
                f"Declare variable '{identifier}': var {identifier} = value",
                f"Check spelling of '{identifier}'",
                f"Import module containing '{identifier}'",
                "Check if identifier is in scope",
            ],
        }

    def _fix_pragma_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix pragma errors."""
        return {
            "type": "suggestion",
            "description": "Pragma error",
            "fixes": [
                "Check pragma syntax and spelling",
                "Ensure pragma is valid for this context",
                "Remove invalid pragma or replace with valid one",
                "Check documentation for available pragmas",
            ],
        }

    def _template_based_patch(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Generate patch using templates."""
        root_cause = analysis.get("root_cause", "")

        # Map root causes to template names
        template_map = {
            "nim_syntax_error": "syntax_fix",
            "nim_type_error": "type_fix",
            "nim_nil_access": "nil_fix",
            "nim_bounds_error": "bounds_fix",
            "nim_option_error": "option_fix",
            "nim_result_error": "result_fix",
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


class NimLanguagePlugin(LanguagePlugin):
    """
    Main Nim language plugin for Homeostasis.

    This plugin orchestrates Nim error analysis and patch generation,
    supporting performance-focused systems programming patterns.
    """

    VERSION = "1.0.0"
    AUTHOR = "Homeostasis Team"

    def __init__(self):
        """Initialize the Nim language plugin."""
        self.language = "nim"
        self.supported_extensions = {".nim", ".nims", ".nimble"}
        self.supported_frameworks = [
            "nim",
            "nimble",
            "jester",
            "prologue",
            "karax",
            "nigui",
        ]

        # Initialize components
        self.exception_handler = NimExceptionHandler()
        self.patch_generator = NimPatchGenerator()

        logger.info("Nim language plugin initialized")

    def get_language_id(self) -> str:
        """Get the unique identifier for this language."""
        return "nim"

    def get_language_name(self) -> str:
        """Get the human-readable name of the language."""
        return "Nim"

    def get_language_version(self) -> str:
        """Get the version of the language supported by this plugin."""
        return "2.0+"

    def get_supported_frameworks(self) -> List[str]:
        """Get the list of frameworks supported by this language plugin."""
        return self.supported_frameworks

    def normalize_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize error data to the standard Homeostasis format.

        Args:
            error_data: Error data in the Nim-specific format

        Returns:
            Error data in the standard format
        """
        # Map Nim-specific error fields to standard format
        normalized = {
            "error_type": error_data.get("error_type", "NimError"),
            "message": error_data.get("message", error_data.get("description", "")),
            "language": "nim",
            "file_path": error_data.get("file_path", error_data.get("file", "")),
            "line_number": error_data.get("line_number", error_data.get("line", 0)),
            "column_number": error_data.get(
                "column_number", error_data.get("column", 0)
            ),
            "compiler_version": error_data.get("compiler_version", ""),
            "build_mode": error_data.get("build_mode", ""),
            "source_code": error_data.get("source_code", ""),
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
        Convert standard format error data back to the Nim-specific format.

        Args:
            standard_error: Error data in the standard format

        Returns:
            Error data in the Nim-specific format
        """
        # Map standard fields back to Nim-specific format
        nim_error = {
            "error_type": standard_error.get("error_type", "NimError"),
            "message": standard_error.get("message", ""),
            "file_path": standard_error.get("file_path", ""),
            "line_number": standard_error.get("line_number", 0),
            "column_number": standard_error.get("column_number", 0),
            "compiler_version": standard_error.get("compiler_version", ""),
            "build_mode": standard_error.get("build_mode", ""),
            "source_code": standard_error.get("source_code", ""),
            "description": standard_error.get("message", ""),
            "file": standard_error.get("file_path", ""),
            "line": standard_error.get("line_number", 0),
            "column": standard_error.get("column_number", 0),
            "stack_trace": standard_error.get("stack_trace", []),
            "context": standard_error.get("context", {}),
            "timestamp": standard_error.get("timestamp"),
            "severity": standard_error.get("severity", "medium"),
        }

        # Add any additional fields from the standard error
        for key, value in standard_error.items():
            if key not in nim_error and value is not None:
                nim_error[key] = value

        return nim_error

    def analyze_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a Nim error.

        Args:
            error_data: Nim error data

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
            analysis["plugin"] = "nim"
            analysis["language"] = "nim"
            analysis["plugin_version"] = self.VERSION

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing Nim error: {e}")
            return {
                "category": "nim",
                "subcategory": "unknown",
                "confidence": "low",
                "suggested_fix": "Unable to analyze Nim error",
                "error": str(e),
                "plugin": "nim",
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
        source_code = context.get("source_code", "")

        fix = self.patch_generator.generate_patch(error_data, analysis, source_code)

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
register_plugin(NimLanguagePlugin())
