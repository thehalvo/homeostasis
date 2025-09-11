"""
Crystal Language Plugin for Homeostasis

This plugin enables Homeostasis to analyze and fix errors in Crystal programming language code.
It provides comprehensive error handling for Crystal compilation errors, runtime issues,
type system errors, and Ruby-like syntax patterns with static typing.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..language_plugin_system import LanguagePlugin, register_plugin

logger = logging.getLogger(__name__)


class CrystalExceptionHandler:
    """
    Handles Crystal exceptions with robust error detection and classification.

    This class provides logic for categorizing Crystal errors based on their type,
    message, and common Ruby-like patterns with static typing.
    """

    def __init__(self):
        """Initialize the Crystal exception handler."""
        self.rule_categories = {
            "syntax": "Crystal syntax and parsing errors",
            "compilation": "Compilation and build errors",
            "type": "Type system and type checking errors",
            "runtime": "Runtime errors and exceptions",
            "memory": "Memory management and allocation errors",
            "macro": "Macro system errors",
            "generic": "Generic type and template errors",
            "fiber": "Fiber and concurrency errors",
            "constant": "Constant definition and access errors",
            "method": "Method definition and calling errors",
            "class": "Class and module definition errors",
            "io": "Input/output and file system errors",
        }

        # Common Crystal error patterns
        self.crystal_error_patterns = {
            "syntax_error": [
                r"syntax error in.*?unexpected token",
                r"syntax error.*?expecting",
                r"unexpected end of file",
                r"unexpected token.*?expected",
                r"unterminated string literal",
                r"unterminated char literal",
                r"invalid character",
                r"expected.*?but found",
            ],
            "type_error": [
                r"no overload matches",
                r"expected.*?but got",
                r"type must be",
                r"can't infer type of",
                r"can't use.*?as",
                r"type.*?doesn't have method",
                r"undefined method.*?for",
                r"wrong number of arguments",
            ],
            "compilation_error": [
                r"can't find file",
                r"can't resolve",
                r"undefined constant",
                r"undefined local variable or method",
                r"already defined",
                r"private method.*?called",
                r"protected method.*?called",
                r"uninitialized constant",
            ],
            "runtime_error": [
                r"unhandled exception",
                r"null reference",
                r"index out of bounds",
                r"division by zero",
                r"invalid memory access",
                r"stack overflow",
                r"KeyError",
                r"ArgumentError",
            ],
            "memory_error": [
                r"out of memory",
                r"memory allocation failed",
                r"invalid pointer",
                r"use after free",
                r"double free",
                r"buffer overflow",
            ],
            "macro_error": [
                r"macro.*?not found",
                r"wrong number of arguments for macro",
                r"macro.*?expansion failed",
                r"invalid macro",
                r"macro.*?already defined",
            ],
            "generic_error": [
                r"generic type.*?not found",
                r"wrong number of type arguments",
                r"type parameter.*?not found",
                r"can't instantiate generic",
                r"type.*?is not generic",
            ],
            "fiber_error": [
                r"fiber.*?error",
                r"channel.*?error",
                r"concurrent.*?error",
                r"deadlock detected",
                r"fiber.*?not started",
            ],
            "constant_error": [
                r"constant.*?not found",
                r"constant.*?already defined",
                r"invalid constant",
                r"constant.*?must be",
                r"uninitialized constant",
            ],
            "method_error": [
                r"method.*?not found",
                r"undefined method",
                r"wrong number of arguments",
                r"method.*?already defined",
                r"private method.*?called",
                r"protected method.*?called",
            ],
            "class_error": [
                r"class.*?not found",
                r"class.*?already defined",
                r"superclass.*?not found",
                r"can't inherit from",
                r"module.*?not found",
            ],
            "io_error": [
                r"file.*?not found",
                r"permission denied",
                r"IO.*?error",
                r"no such file or directory",
                r"connection.*?error",
            ],
        }

        # Crystal-specific concepts and their common issues
        self.crystal_concepts = {
            "nil": ["nil reference", "null pointer", "nil check"],
            "union": ["union type", "union access", "type union"],
            "pointer": ["pointer access", "pointer arithmetic", "null pointer"],
            "channel": ["channel operation", "channel blocking", "channel closed"],
            "fiber": ["fiber scheduling", "fiber communication", "fiber error"],
            "struct": ["struct field", "struct initialization", "struct access"],
            "enum": ["enum value", "enum case", "enum access"],
            "alias": ["type alias", "alias definition", "alias access"],
            "annotation": ["annotation error", "annotation syntax", "annotation usage"],
            "lib": ["lib binding", "C binding", "external library"],
        }

        # Load rules from different categories
        self.rules = self._load_rules()

        # Pre-compile regex patterns for better performance
        self._compile_patterns()

    def _load_rules(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load Crystal error rules from rule files."""
        rules = {}
        rules_dir = Path(__file__).parent.parent / "rules" / "crystal"

        try:
            # Load common Crystal rules
            common_rules_path = rules_dir / "crystal_common_errors.json"
            if common_rules_path.exists():
                with open(common_rules_path, "r") as f:
                    common_data = json.load(f)
                    rules["common"] = common_data.get("rules", [])
                    logger.info(f"Loaded {len(rules['common'])} common Crystal rules")

            # Load concept-specific rules
            for concept in ["types", "macros", "fibers", "memory"]:
                concept_rules_path = rules_dir / f"crystal_{concept}_errors.json"
                if concept_rules_path.exists():
                    with open(concept_rules_path, "r") as f:
                        concept_data = json.load(f)
                        rules[concept] = concept_data.get("rules", [])
                        logger.info(f"Loaded {len(rules[concept])} {concept} rules")

        except Exception as e:
            logger.error(f"Error loading Crystal rules: {e}")
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
                    logger.warning(
                        f"Invalid regex pattern in rule {rule.get('id', 'unknown')}: {e}"
                    )

    def analyze_exception(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a Crystal exception and determine its type and potential fixes.

        Args:
            error_data: Crystal error data in standard format

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
        concept_analysis = self._analyze_crystal_concepts(message)
        if concept_analysis.get("confidence", "low") != "low":
            # Merge concept-specific findings
            analysis.update(concept_analysis)

        # Find matching rules
        matches = self._find_matching_rules(message, error_data)

        if matches:
            # Use the best match (highest confidence)
            best_match = max(matches, key=lambda x: x.get("confidence_score", 0))

            # Only update if we don't already have a high confidence nil subcategory
            if not (
                analysis.get("subcategory") == "nil" and
                analysis.get("confidence") == "high"
            ):
                analysis.update(
                    {
                        "category": best_match.get(
                            "category", analysis.get("category", "unknown")
                        ),
                        "subcategory": self._get_subcategory_from_rule(
                            best_match, analysis
                        ),
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
        message_lower = message.lower()

        # Check nil errors first (priority over runtime)
        if "null reference" in message_lower or (
            "nil" in message_lower and "method" in message_lower
        ):
            return {
                "category": "crystal",
                "subcategory": "nil",
                "confidence": "high",
                "suggested_fix": "Add nil check or use safe navigation",
                "root_cause": "crystal_nil_error",
                "severity": "high",
                "tags": ["crystal", "nil", "runtime"],
            }

        # Check syntax errors
        for pattern in self.crystal_error_patterns["syntax_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "crystal",
                    "subcategory": "syntax",
                    "confidence": "high",
                    "suggested_fix": "Fix Crystal syntax errors",
                    "root_cause": "crystal_syntax_error",
                    "severity": "high",
                    "tags": ["crystal", "syntax", "parser"],
                }

        # Check type errors
        for pattern in self.crystal_error_patterns["type_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "crystal",
                    "subcategory": "type",
                    "confidence": "high",
                    "suggested_fix": "Fix type system errors and type mismatches",
                    "root_cause": "crystal_type_error",
                    "severity": "high",
                    "tags": ["crystal", "type", "static_typing"],
                }

        # Check compilation errors
        for pattern in self.crystal_error_patterns["compilation_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "crystal",
                    "subcategory": "compilation",
                    "confidence": "high",
                    "suggested_fix": "Fix compilation and build errors",
                    "root_cause": "crystal_compilation_error",
                    "severity": "high",
                    "tags": ["crystal", "compilation", "build"],
                }

        # Check runtime errors
        for pattern in self.crystal_error_patterns["runtime_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "crystal",
                    "subcategory": "runtime",
                    "confidence": "high",
                    "suggested_fix": "Fix runtime errors and exceptions",
                    "root_cause": "crystal_runtime_error",
                    "severity": "high",
                    "tags": ["crystal", "runtime", "exception"],
                }

        # Check memory errors
        for pattern in self.crystal_error_patterns["memory_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "crystal",
                    "subcategory": "memory",
                    "confidence": "high",
                    "suggested_fix": "Fix memory management and allocation issues",
                    "root_cause": "crystal_memory_error",
                    "severity": "critical",
                    "tags": ["crystal", "memory", "safety"],
                }

        # Check macro errors
        for pattern in self.crystal_error_patterns["macro_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "crystal",
                    "subcategory": "macro",
                    "confidence": "high",
                    "suggested_fix": "Fix macro system errors",
                    "root_cause": "crystal_macro_error",
                    "severity": "medium",
                    "tags": ["crystal", "macro", "metaprogramming"],
                }

        # Check generic errors
        for pattern in self.crystal_error_patterns["generic_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "crystal",
                    "subcategory": "generic",
                    "confidence": "high",
                    "suggested_fix": "Fix generic type and template errors",
                    "root_cause": "crystal_generic_error",
                    "severity": "medium",
                    "tags": ["crystal", "generic", "template"],
                }

        # Check fiber errors
        for pattern in self.crystal_error_patterns["fiber_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "crystal",
                    "subcategory": "fiber",
                    "confidence": "high",
                    "suggested_fix": "Fix fiber and concurrency errors",
                    "root_cause": "crystal_fiber_error",
                    "severity": "medium",
                    "tags": ["crystal", "fiber", "concurrency"],
                }

        # Check method errors
        for pattern in self.crystal_error_patterns["method_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "crystal",
                    "subcategory": "method",
                    "confidence": "high",
                    "suggested_fix": "Fix method definition and calling errors",
                    "root_cause": "crystal_method_error",
                    "severity": "high",
                    "tags": ["crystal", "method", "function"],
                }

        return {
            "category": "crystal",
            "subcategory": "unknown",
            "confidence": "low",
            "suggested_fix": "Review Crystal code and compiler error details",
            "root_cause": "crystal_generic_error",
            "severity": "medium",
            "tags": ["crystal", "generic"],
        }

    def _analyze_crystal_concepts(self, message: str) -> Dict[str, Any]:
        """Analyze Crystal-specific concept errors."""
        message_lower = message.lower()

        # Check for nil-related errors
        if any(
            keyword in message_lower for keyword in ["nil", "null", "null reference"]
        ):
            return {
                "category": "crystal",
                "subcategory": "nil",
                "confidence": "high",
                "suggested_fix": "Add nil check or use union types properly",
                "root_cause": "crystal_nil_error",
                "severity": "high",
                "tags": ["crystal", "nil", "safety"],
            }

        # Check for union type errors
        if any(keyword in message_lower for keyword in ["union", "union type"]):
            return {
                "category": "crystal",
                "subcategory": "union",
                "confidence": "high",
                "suggested_fix": "Handle union types with proper type checking",
                "root_cause": "crystal_union_error",
                "severity": "medium",
                "tags": ["crystal", "union", "type"],
            }

        # Check for pointer-related errors
        if any(keyword in message_lower for keyword in ["pointer", "ptr"]):
            return {
                "category": "crystal",
                "subcategory": "pointer",
                "confidence": "medium",
                "suggested_fix": "Check pointer operations and memory safety",
                "root_cause": "crystal_pointer_error",
                "severity": "high",
                "tags": ["crystal", "pointer", "memory"],
            }

        # Check for channel-related errors
        if any(keyword in message_lower for keyword in ["channel", "chan"]):
            return {
                "category": "crystal",
                "subcategory": "channel",
                "confidence": "medium",
                "suggested_fix": "Handle channel operations and blocking correctly",
                "root_cause": "crystal_channel_error",
                "severity": "medium",
                "tags": ["crystal", "channel", "concurrency"],
            }

        # Check for fiber-related errors
        if any(keyword in message_lower for keyword in ["fiber", "spawn"]):
            return {
                "category": "crystal",
                "subcategory": "fiber",
                "confidence": "medium",
                "suggested_fix": "Handle fiber scheduling and communication",
                "root_cause": "crystal_fiber_error",
                "severity": "medium",
                "tags": ["crystal", "fiber", "concurrency"],
            }

        # Check for struct-related errors
        if any(keyword in message_lower for keyword in ["struct", "record"]):
            return {
                "category": "crystal",
                "subcategory": "struct",
                "confidence": "medium",
                "suggested_fix": "Check struct field access and initialization",
                "root_cause": "crystal_struct_error",
                "severity": "medium",
                "tags": ["crystal", "struct", "field"],
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
        if file_path.endswith(".cr"):
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
        if "fiber" in message:
            context_tags.add("fiber")
        if "union" in message:
            context_tags.add("union")

        if context_tags & rule_tags:
            base_confidence += 0.1

        return min(base_confidence, 1.0)

    def _get_subcategory_from_rule(
        self, rule: Dict[str, Any], analysis: Dict[str, Any]
    ) -> str:
        """Extract subcategory from rule type."""
        rule_type = rule.get("type", "")

        # Map rule types to subcategories
        type_to_subcategory = {
            "FiberError": "fiber",
            "TypeError": "type",
            "SyntaxError": "syntax",
            "CompilationError": "compilation",
            "RuntimeError": "runtime",
            "MethodError": "method",
            "MemoryError": "memory",
            "MacroError": "macro",
            "GenericError": "generic",
            "ConstantError": "constant",
            "ClassError": "class",
            "IOError": "io",
        }

        subcategory = type_to_subcategory.get(rule_type, "")
        if subcategory:
            return subcategory

        # Fall back to analysis subcategory or rule type
        return analysis.get("subcategory", rule_type.lower().replace("error", ""))


class CrystalPatchGenerator:
    """
    Generates patches for Crystal errors based on analysis results.

    This class creates Crystal code fixes for common errors using templates
    and heuristics specific to Ruby-like syntax with static typing.
    """

    def __init__(self):
        """Initialize the Crystal patch generator."""
        self.template_dir = (
            Path(__file__).parent.parent / "patch_generation" / "templates"
        )
        self.crystal_template_dir = self.template_dir / "crystal"

        # Ensure template directory exists
        self.crystal_template_dir.mkdir(parents=True, exist_ok=True)

        # Load patch templates
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict[str, str]:
        """Load Crystal patch templates."""
        templates = {}

        if not self.crystal_template_dir.exists():
            logger.warning(
                f"Crystal templates directory not found: {self.crystal_template_dir}"
            )
            return templates

        for template_file in self.crystal_template_dir.glob("*.cr.template"):
            try:
                with open(template_file, "r") as f:
                    template_name = template_file.stem.replace(".cr", "")
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
        Generate a patch for the Crystal error.

        Args:
            error_data: The Crystal error data
            analysis: Analysis results from CrystalExceptionHandler
            source_code: The Crystal source code that caused the error

        Returns:
            Patch information or None if no patch can be generated
        """
        root_cause = analysis.get("root_cause", "")

        # Map root causes to patch strategies
        patch_strategies = {
            "crystal_syntax_error": self._fix_syntax_error,
            "crystal_type_error": self._fix_type_error,
            "crystal_compilation_error": self._fix_compilation_error,
            "crystal_runtime_error": self._fix_runtime_error,
            "crystal_memory_error": self._fix_memory_error,
            "crystal_macro_error": self._fix_macro_error,
            "crystal_generic_error": self._fix_generic_error,
            "crystal_fiber_error": self._fix_fiber_error,
            "crystal_method_error": self._fix_method_error,
            "crystal_nil_error": self._fix_nil_error,
            "crystal_union_error": self._fix_union_error,
            "crystal_pointer_error": self._fix_pointer_error,
            "crystal_channel_error": self._fix_channel_error,
            "crystal_struct_error": self._fix_struct_error,
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
        """Fix Crystal syntax errors."""
        message = error_data.get("message", "")

        fixes = []

        if "unexpected token" in message.lower():
            fixes.append(
                {
                    "type": "suggestion",
                    "description": "Unexpected token found",
                    "fix": "Check for syntax errors like missing operators, semicolons, or keywords",
                }
            )

        if "expecting" in message.lower():
            # Extract expected token
            expected_match = re.search(r"expecting (.+)", message)
            if expected_match:
                expected = expected_match.group(1)
                fixes.append(
                    {
                        "type": "suggestion",
                        "description": f"Expected {expected}",
                        "fix": f"Add missing {expected} or fix the syntax structure",
                    }
                )

        if "unterminated string" in message.lower():
            fixes.append(
                {
                    "type": "suggestion",
                    "description": "Unterminated string literal",
                    "fix": "Add missing closing quote for string literal",
                }
            )

        if "unexpected end of file" in message.lower():
            fixes.append(
                {
                    "type": "suggestion",
                    "description": "Unexpected end of file",
                    "fix": "Check for missing closing brackets, braces, or end statements",
                }
            )

        if fixes:
            return {
                "type": "multiple_suggestions",
                "fixes": fixes,
                "description": "Crystal syntax error fixes",
            }

        return {
            "type": "suggestion",
            "description": "Crystal syntax error. Check code structure and syntax",
        }

    def _fix_type_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix Crystal type errors."""
        message = error_data.get("message", "")

        if "no overload matches" in message.lower():
            return {
                "type": "suggestion",
                "description": "No matching method overload found",
                "fixes": [
                    "Check method arguments and their types",
                    "Ensure all required parameters are provided",
                    "Check for type compatibility",
                    "Add explicit type annotations if needed",
                ],
            }

        if "expected" in message.lower() and "but got" in message.lower():
            # Extract type information
            type_match = re.search(r"expected (.+?) but got (.+)", message)
            if type_match:
                expected_type = type_match.group(1)
                got_type = type_match.group(2)

                return {
                    "type": "suggestion",
                    "description": f"Type mismatch: expected {expected_type}, got {got_type}",
                    "fixes": [
                        f"Convert value to {expected_type}",
                        f"Change variable type to {got_type}",
                        f"Use type casting: value.as({expected_type})",
                        "Check if types are compatible or add explicit conversion",
                    ],
                }

        if "can't infer type" in message.lower():
            return {
                "type": "suggestion",
                "description": "Cannot infer type",
                "fixes": [
                    "Add explicit type annotation: var_name : Type",
                    "Initialize variable with a value",
                    "Use type assertion: value.as(Type)",
                    "Provide more context for type inference",
                ],
            }

        if "undefined method" in message.lower():
            # Extract method name
            method_match = re.search(r"undefined method '(.+?)'", message)
            method_name = method_match.group(1) if method_match else "method"

            return {
                "type": "suggestion",
                "description": f"Undefined method '{method_name}'",
                "fixes": [
                    f"Check spelling of method '{method_name}'",
                    f"Ensure method '{method_name}' is defined for this type",
                    "Import required modules or classes",
                    "Check method visibility (private/protected/public)",
                ],
            }

        return {
            "type": "suggestion",
            "description": "Type system error. Check type compatibility and method signatures",
        }

    def _fix_compilation_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix compilation errors."""
        message = error_data.get("message", "")

        if "can't find file" in message.lower():
            return {
                "type": "suggestion",
                "description": "File not found during compilation",
                "fixes": [
                    "Check if file path is correct",
                    "Ensure file exists and is accessible",
                    "Check file permissions",
                    "Verify require path is relative to current file",
                ],
            }

        if "undefined constant" in message.lower():
            # Extract constant name
            constant_match = re.search(r"undefined constant (.+)", message)
            constant = constant_match.group(1) if constant_match else "constant"

            return {
                "type": "suggestion",
                "description": f"Undefined constant '{constant}'",
                "fixes": [
                    f"Define constant '{constant}' before use",
                    f"Check spelling of constant '{constant}'",
                    "Ensure constant is in correct scope",
                    "Import module containing the constant",
                ],
            }

        if "already defined" in message.lower():
            return {
                "type": "suggestion",
                "description": "Identifier already defined",
                "fixes": [
                    "Remove duplicate definition",
                    "Rename one of the conflicting identifiers",
                    "Use different scopes for the identifiers",
                    "Check for multiple requires of the same module",
                ],
            }

        if "private method" in message.lower() and "called" in message.lower():
            return {
                "type": "suggestion",
                "description": "Private method called from outside class",
                "fixes": [
                    "Make method public or protected",
                    "Call method from within the class",
                    "Create a public wrapper method",
                    "Use proper encapsulation patterns",
                ],
            }

        return {
            "type": "suggestion",
            "description": "Compilation error. Check code structure and dependencies",
        }

    def _fix_runtime_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix runtime errors."""
        message = error_data.get("message", "")

        if "null reference" in message.lower():
            return {
                "type": "suggestion",
                "description": "Null reference error",
                "fixes": [
                    "Check for nil before accessing object",
                    "Use safe navigation operator: obj.try(&.method)",
                    "Initialize object before use",
                    "Use union types with nil handling",
                ],
            }

        if "index out of bounds" in message.lower():
            return {
                "type": "suggestion",
                "description": "Array index out of bounds",
                "fixes": [
                    "Check array bounds before accessing: if index < array.size",
                    "Use safe access methods: array[index]?",
                    "Validate index is within valid range",
                    "Handle empty arrays properly",
                ],
            }

        if "division by zero" in message.lower():
            return {
                "type": "suggestion",
                "description": "Division by zero error",
                "fixes": [
                    "Check denominator is not zero before division",
                    "Add conditional check: if denominator != 0",
                    "Use exception handling: begin/rescue",
                    "Provide default value for zero case",
                ],
            }

        if "unhandled exception" in message.lower():
            return {
                "type": "suggestion",
                "description": "Unhandled exception",
                "fixes": [
                    "Add exception handling: begin/rescue/end",
                    "Use specific exception types in rescue",
                    "Add proper error handling and recovery",
                    "Check for potential error conditions",
                ],
            }

        return {
            "type": "suggestion",
            "description": "Runtime error. Check program logic and input validation",
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
                    "Check for memory leaks",
                    "Use more memory-efficient data structures",
                    "Process data in chunks or streams",
                ],
            }

        if "invalid pointer" in message.lower():
            return {
                "type": "suggestion",
                "description": "Invalid pointer access",
                "fixes": [
                    "Check pointer validity before use",
                    "Avoid pointer arithmetic errors",
                    "Use safe pointer operations",
                    "Check for null pointers",
                ],
            }

        return {
            "type": "suggestion",
            "description": "Memory management error",
            "fixes": [
                "Check memory allocation and deallocation",
                "Use safe memory access patterns",
                "Avoid buffer overflows and underflows",
            ],
        }

    def _fix_macro_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix macro system errors."""
        return {
            "type": "suggestion",
            "description": "Macro system error",
            "fixes": [
                "Check macro syntax and parameters",
                "Ensure macro is properly defined",
                "Check macro expansion and usage",
                "Review macro argument types and constraints",
            ],
        }

    def _fix_generic_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix generic type errors."""
        return {
            "type": "suggestion",
            "description": "Generic type error",
            "fixes": [
                "Check generic type parameters",
                "Ensure proper type constraints",
                "Verify generic instantiation",
                "Add explicit type annotations",
            ],
        }

    def _fix_fiber_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix fiber and concurrency errors."""
        return {
            "type": "suggestion",
            "description": "Fiber/concurrency error",
            "fixes": [
                "Check fiber scheduling and communication",
                "Handle fiber exceptions properly",
                "Use proper synchronization mechanisms",
                "Avoid deadlocks and race conditions",
            ],
        }

    def _fix_method_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix method definition and calling errors."""
        message = error_data.get("message", "")

        if "wrong number of arguments" in message.lower():
            return {
                "type": "suggestion",
                "description": "Wrong number of arguments",
                "fixes": [
                    "Check method signature for required parameters",
                    "Provide all required arguments",
                    "Use default parameters if available",
                    "Check for method overloads",
                ],
            }

        return {
            "type": "suggestion",
            "description": "Method error",
            "fixes": [
                "Check method definitions and calls",
                "Verify argument types and counts",
                "Review method accessibility",
            ],
        }

    def _fix_nil_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix nil-related errors."""
        return {
            "type": "suggestion",
            "description": "Nil reference error",
            "fixes": [
                "Check for nil before accessing: if obj != nil",
                "Use safe navigation: obj.try(&.method)",
                "Handle nil in union types properly",
                "Initialize variables before use",
            ],
        }

    def _fix_union_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix union type errors."""
        return {
            "type": "suggestion",
            "description": "Union type error",
            "fixes": [
                "Check union type with is_a?: if var.is_a?(Type)",
                "Handle all possible types in union",
                "Use case statement for union types",
                "Add explicit type checks",
            ],
        }

    def _fix_pointer_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix pointer-related errors."""
        return {
            "type": "suggestion",
            "description": "Pointer error",
            "fixes": [
                "Check pointer validity before dereferencing",
                "Use safe pointer operations",
                "Avoid pointer arithmetic errors",
                "Handle null pointers properly",
            ],
        }

    def _fix_channel_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix channel-related errors."""
        return {
            "type": "suggestion",
            "description": "Channel error",
            "fixes": [
                "Check channel operations and blocking",
                "Handle channel close events",
                "Use proper channel synchronization",
                "Avoid deadlocks in channel operations",
            ],
        }

    def _fix_struct_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix struct-related errors."""
        return {
            "type": "suggestion",
            "description": "Struct error",
            "fixes": [
                "Check struct field access and initialization",
                "Ensure all required fields are set",
                "Use proper struct syntax",
                "Check field types and constraints",
            ],
        }

    def _template_based_patch(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Generate patch using templates."""
        root_cause = analysis.get("root_cause", "")

        # Map root causes to template names
        template_map = {
            "crystal_syntax_error": "syntax_fix",
            "crystal_type_error": "type_fix",
            "crystal_nil_error": "nil_fix",
            "crystal_union_error": "union_fix",
            "crystal_fiber_error": "fiber_fix",
            "crystal_channel_error": "channel_fix",
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


class CrystalLanguagePlugin(LanguagePlugin):
    """
    Main Crystal language plugin for Homeostasis.

    This plugin orchestrates Crystal error analysis and patch generation,
    supporting Ruby-like syntax with static typing.
    """

    VERSION = "1.0.0"
    AUTHOR = "Homeostasis Team"

    def __init__(self):
        """Initialize the Crystal language plugin."""
        self.language = "crystal"
        self.supported_extensions = {".cr"}
        self.supported_frameworks = [
            "crystal",
            "shards",
            "kemal",
            "amber",
            "lucky",
            "athena",
        ]

        # Initialize components
        self.exception_handler = CrystalExceptionHandler()
        self.patch_generator = CrystalPatchGenerator()

        logger.info("Crystal language plugin initialized")

    def get_language_id(self) -> str:
        """Get the unique identifier for this language."""
        return "crystal"

    def get_language_name(self) -> str:
        """Get the human-readable name of the language."""
        return "Crystal"

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
            error_data: Error data in the Crystal-specific format

        Returns:
            Error data in the standard format
        """
        # Map Crystal-specific error fields to standard format
        normalized = {
            "error_type": error_data.get("error_type", "CrystalError"),
            "message": error_data.get("message", error_data.get("description", "")),
            "language": "crystal",
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
        Convert standard format error data back to the Crystal-specific format.

        Args:
            standard_error: Error data in the standard format

        Returns:
            Error data in the Crystal-specific format
        """
        # Map standard fields back to Crystal-specific format
        crystal_error = {
            "error_type": standard_error.get("error_type", "CrystalError"),
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
            if key not in crystal_error and value is not None:
                crystal_error[key] = value

        return crystal_error

    def analyze_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a Crystal error.

        Args:
            error_data: Crystal error data

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
            analysis["plugin"] = "crystal"
            analysis["language"] = "crystal"
            analysis["plugin_version"] = self.VERSION

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing Crystal error: {e}")
            return {
                "category": "crystal",
                "subcategory": "unknown",
                "confidence": "low",
                "suggested_fix": "Unable to analyze Crystal error",
                "error": str(e),
                "plugin": "crystal",
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
register_plugin(CrystalLanguagePlugin())
