"""
Zig Language Plugin for Homeostasis

This plugin enables Homeostasis to analyze and fix errors in Zig programming language code.
It provides comprehensive error handling for Zig compilation errors, runtime issues,
memory management problems, and systems programming best practices.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..language_plugin_system import LanguagePlugin, register_plugin

logger = logging.getLogger(__name__)


class ZigExceptionHandler:
    """
    Handles Zig exceptions with robust error detection and classification.

    This class provides logic for categorizing Zig errors based on their type,
    message, and common systems programming patterns.
    """

    def __init__(self):
        """Initialize the Zig exception handler."""
        self.rule_categories = {
            "syntax": "Zig syntax and parsing errors",
            "compilation": "Compilation and build errors",
            "type": "Type system and type checking errors",
            "memory": "Memory management and allocation errors",
            "runtime": "Runtime errors and panics",
            "import": "Import and module system errors",
            "comptime": "Compile-time evaluation errors",
            "undefined": "Undefined behavior and safety errors",
            "async": "Async and concurrent programming errors",
            "build": "Build system and zig build errors",
            "linking": "Linking and external library errors",
            "cross_compilation": "Cross-compilation and target errors",
        }

        # Common Zig error patterns
        self.zig_error_patterns = {
            "syntax_error": [
                r"error: expected (?!type).*?, found",  # Exclude type-related expectations
                r"error: invalid token",
                r"error: unexpected token",
                r"error: expected expression",
                r"error: expected statement",
                r"error: unterminated string literal",
                r"error: unterminated character literal",
            ],
            "type_error": [
                r"error: expected type",
                r"error: type '.*?' cannot represent integer value",
                r"error: cannot cast '.*?' to '.*?'",
                r"error: type mismatch",
                r"error: incompatible types",
                r"error: integer overflow",
                r"error: integer underflow",
            ],
            "undefined_error": [
                r"error: use of undeclared identifier '.*?'",
                r"error: '.*?' not found",
                r"error: no member named '.*?'",
                r"error: undefined symbol",
                r"error: container '.*?' has no member named '.*?'",
            ],
            "memory_error": [
                r"error: out of memory",
                r"error: invalid pointer",
                r"error: null pointer dereference",
                r"error: use after free",
                r"error: double free",
                r"error: memory leak detected",
            ],
            "import_error": [
                r"error: unable to find '.*?'",
                r"error: import failure",
                r"error: file not found",
                r"error: circular dependency",
                r"error: cannot import '.*?'",
            ],
            "comptime_error": [
                r"error: unable to evaluate constant expression",
                r"error: comptime call of non-comptime function",
                r"error: comptime variable cannot be modified at runtime",
                r"error: expected compile-time constant",
            ],
            "build_error": [
                r"error: zig build failed",
                r"error: build.zig error",
                r"error: target not found",
                r"error: build step failed",
                r"error: dependency not found",
            ],
            "async_error": [
                r"error: async function cannot be called directly",
                r"error: async function called without await",
                r"error: suspend point not reachable",
                r"error: async frame too large",
                r"error: await in non-async function",
            ],
        }

        # Zig-specific concepts and their common issues
        self.zig_concepts = {
            "optionals": [
                "unwrapping null",
                "optional type mismatch",
                "missing null check",
            ],
            "error_unions": [
                "error not handled",
                "error set mismatch",
                "missing try/catch",
            ],
            "allocators": ["allocator not found", "allocation failed", "memory leak"],
            "slices": ["index out of bounds", "slice bounds check", "invalid slice"],
            "arrays": ["array bounds", "array size mismatch", "comptime array size"],
            "structs": [
                "field not found",
                "struct initialization",
                "field type mismatch",
            ],
            "enums": [
                "enum value not found",
                "enum switch not exhaustive",
                "invalid enum",
            ],
            "unions": [
                "union field access",
                "union tag mismatch",
                "tagged union error",
            ],
            "pointers": ["null pointer", "pointer arithmetic", "alignment error"],
            "generics": [
                "generic parameter",
                "type parameter",
                "generic instantiation",
            ],
        }

        # Load rules from different categories
        self.rules = self._load_rules()

        # Pre-compile regex patterns for better performance
        self._compile_patterns()

    def _load_rules(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load Zig error rules from rule files."""
        rules = {}
        rules_dir = Path(__file__).parent.parent / "rules" / "zig"

        try:
            # Load common Zig rules
            common_rules_path = rules_dir / "zig_common_errors.json"
            if common_rules_path.exists():
                with open(common_rules_path, "r") as f:
                    common_data = json.load(f)
                    rules["common"] = common_data.get("rules", [])
                    logger.info(f"Loaded {len(rules['common'])} common Zig rules")

            # Load concept-specific rules
            for concept in ["memory", "types", "async", "comptime"]:
                concept_rules_path = rules_dir / f"zig_{concept}_errors.json"
                if concept_rules_path.exists():
                    with open(concept_rules_path, "r") as f:
                        concept_data = json.load(f)
                        rules[concept] = concept_data.get("rules", [])
                        logger.info(f"Loaded {len(rules[concept])} {concept} rules")

        except Exception as e:
            logger.error(f"Error loading Zig rules: {e}")
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
        Analyze a Zig exception and determine its type and potential fixes.

        Args:
            error_data: Zig error data in standard format

        Returns:
            Analysis results with categorization and fix suggestions
        """
        error_type = error_data.get("error_type", "ZigError")
        message = error_data.get("message", "")
        file_path = error_data.get("file_path", "")
        line_number = error_data.get("line_number", 0)
        column_number = error_data.get("column_number", 0)

        # Analyze based on error patterns
        analysis = self._analyze_by_patterns(message, file_path)

        # Check for concept-specific issues
        concept_analysis = self._analyze_zig_concepts(message)
        if concept_analysis.get("confidence", "low") != "low":
            # Merge concept-specific findings
            analysis.update(concept_analysis)

        # Find matching rules
        matches = self._find_matching_rules(message, error_data)

        # Only update with rule-based matches if we don't have high confidence from patterns
        if matches and analysis.get("confidence", "low") != "high":
            # Use the best match (highest confidence)
            best_match = max(matches, key=lambda x: x.get("confidence_score", 0))
            # Map error type to subcategory
            error_type = best_match.get("type", "")
            subcategory_map = {
                "SyntaxError": "syntax",
                "TypeError": "type",
                "MemoryError": "memory",
                "UndefinedError": "undefined",
                "ComptimeError": "comptime",
                "ImportError": "import",
            }
            subcategory = subcategory_map.get(
                error_type, analysis.get("subcategory", "unknown")
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
        # Check type errors first (more specific) - must be before syntax check
        for pattern in self.zig_error_patterns["type_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "zig",
                    "subcategory": "type",
                    "confidence": "high",
                    "suggested_fix": "Fix type system errors and type mismatches",
                    "root_cause": "zig_type_error",
                    "severity": "high",
                    "tags": ["zig", "type", "casting"],
                }

        # Check syntax errors - but skip if it looks like a type error
        for pattern in self.zig_error_patterns["syntax_error"]:
            if (
                re.search(pattern, message, re.IGNORECASE) and
                "expected type" not in message
            ):
                return {
                    "category": "zig",
                    "subcategory": "syntax",
                    "confidence": "high",
                    "suggested_fix": "Fix Zig syntax errors",
                    "root_cause": "zig_syntax_error",
                    "severity": "high",
                    "tags": ["zig", "syntax", "parser"],
                }

        # Check undefined errors
        for pattern in self.zig_error_patterns["undefined_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "zig",
                    "subcategory": "undefined",
                    "confidence": "high",
                    "suggested_fix": "Define missing identifiers or check imports",
                    "root_cause": "zig_undefined_error",
                    "severity": "high",
                    "tags": ["zig", "undefined", "identifier"],
                }

        # Check memory errors
        for pattern in self.zig_error_patterns["memory_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "zig",
                    "subcategory": "memory",
                    "confidence": "high",
                    "suggested_fix": "Fix memory management and allocation issues",
                    "root_cause": "zig_memory_error",
                    "severity": "critical",
                    "tags": ["zig", "memory", "safety"],
                }

        # Check import errors
        for pattern in self.zig_error_patterns["import_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "zig",
                    "subcategory": "import",
                    "confidence": "high",
                    "suggested_fix": "Fix import paths and module dependencies",
                    "root_cause": "zig_import_error",
                    "severity": "high",
                    "tags": ["zig", "import", "module"],
                }

        # Check comptime errors
        for pattern in self.zig_error_patterns["comptime_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "zig",
                    "subcategory": "comptime",
                    "confidence": "high",
                    "suggested_fix": "Fix compile-time evaluation issues",
                    "root_cause": "zig_comptime_error",
                    "severity": "medium",
                    "tags": ["zig", "comptime", "evaluation"],
                }

        # Check build errors
        for pattern in self.zig_error_patterns["build_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "zig",
                    "subcategory": "build",
                    "confidence": "high",
                    "suggested_fix": "Fix build configuration and dependencies",
                    "root_cause": "zig_build_error",
                    "severity": "high",
                    "tags": ["zig", "build", "configuration"],
                }

        # Check async errors
        for pattern in self.zig_error_patterns["async_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "zig",
                    "subcategory": "async",
                    "confidence": "high",
                    "suggested_fix": "Fix async/await usage and frame management",
                    "root_cause": "zig_async_error",
                    "severity": "medium",
                    "tags": ["zig", "async", "concurrency"],
                }

        return {
            "category": "zig",
            "subcategory": "unknown",
            "confidence": "low",
            "suggested_fix": "Review Zig code and compiler error details",
            "root_cause": "zig_generic_error",
            "severity": "medium",
            "tags": ["zig", "generic"],
        }

    def _analyze_zig_concepts(self, message: str) -> Dict[str, Any]:
        """Analyze Zig-specific concept errors."""
        message_lower = message.lower()

        # Check for optional-related errors
        if any(
            keyword in message_lower for keyword in ["optional", "null", "?", "unwrap"]
        ):
            if "null" in message_lower and "unwrap" in message_lower:
                return {
                    "category": "zig",
                    "subcategory": "optionals",
                    "confidence": "high",
                    "suggested_fix": "Add null check before unwrapping optional value",
                    "root_cause": "zig_optional_unwrap_null",
                    "severity": "high",
                    "tags": ["zig", "optionals", "null_safety"],
                }

        # Check for error union issues
        if any(keyword in message_lower for keyword in ["error", "try", "catch", "!"]):
            if "error" in message_lower and (
                "not handled" in message_lower or "try" in message_lower
            ):
                return {
                    "category": "zig",
                    "subcategory": "error_unions",
                    "confidence": "high",
                    "suggested_fix": "Handle error cases with try/catch or propagate with try",
                    "root_cause": "zig_error_not_handled",
                    "severity": "high",
                    "tags": ["zig", "error_handling", "safety"],
                }

        # Check for allocator issues
        if any(
            keyword in message_lower
            for keyword in ["allocator", "allocation", "alloc", "free"]
        ):
            return {
                "category": "zig",
                "subcategory": "allocators",
                "confidence": "medium",
                "suggested_fix": "Check allocator usage and memory management",
                "root_cause": "zig_allocator_error",
                "severity": "medium",
                "tags": ["zig", "allocator", "memory"],
            }

        # Check for slice/array bounds issues
        if any(
            keyword in message_lower
            for keyword in ["bounds", "index", "slice", "array"]
        ):
            return {
                "category": "zig",
                "subcategory": "bounds",
                "confidence": "high",
                "suggested_fix": "Check array/slice bounds and indices",
                "root_cause": "zig_bounds_error",
                "severity": "high",
                "tags": ["zig", "bounds", "safety"],
            }

        # Check for struct/field issues
        if any(keyword in message_lower for keyword in ["struct", "field", "member"]):
            return {
                "category": "zig",
                "subcategory": "structs",
                "confidence": "medium",
                "suggested_fix": "Check struct field definitions and access",
                "root_cause": "zig_struct_error",
                "severity": "medium",
                "tags": ["zig", "struct", "field"],
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
        if file_path.endswith(".zig"):
            base_confidence += 0.2

        # Boost confidence based on rule reliability
        reliability = rule.get("reliability", "medium")
        reliability_boost = {"high": 0.2, "medium": 0.1, "low": 0.0}
        base_confidence += reliability_boost.get(reliability, 0.0)

        # Boost confidence for concept matches
        rule_tags = set(rule.get("tags", []))
        context_tags = set()

        message = error_data.get("message", "").lower()
        if "comptime" in message:
            context_tags.add("comptime")
        if "async" in message:
            context_tags.add("async")
        if "optional" in message or "null" in message:
            context_tags.add("optionals")

        if context_tags & rule_tags:
            base_confidence += 0.1

        return min(base_confidence, 1.0)


class ZigPatchGenerator:
    """
    Generates patches for Zig errors based on analysis results.

    This class creates Zig code fixes for common errors using templates
    and heuristics specific to systems programming patterns.
    """

    def __init__(self):
        """Initialize the Zig patch generator."""
        self.template_dir = (
            Path(__file__).parent.parent / "patch_generation" / "templates"
        )
        self.zig_template_dir = self.template_dir / "zig"

        # Ensure template directory exists
        self.zig_template_dir.mkdir(parents=True, exist_ok=True)

        # Load patch templates
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict[str, str]:
        """Load Zig patch templates."""
        templates = {}

        if not self.zig_template_dir.exists():
            logger.warning(
                f"Zig templates directory not found: {self.zig_template_dir}"
            )
            return templates

        for template_file in self.zig_template_dir.glob("*.zig.template"):
            try:
                with open(template_file, "r") as f:
                    template_name = template_file.stem.replace(".zig", "")
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
        Generate a patch for the Zig error.

        Args:
            error_data: The Zig error data
            analysis: Analysis results from ZigExceptionHandler
            source_code: The Zig source code that caused the error

        Returns:
            Patch information or None if no patch can be generated
        """
        root_cause = analysis.get("root_cause", "")

        # Map root causes to patch strategies
        patch_strategies = {
            "zig_syntax_error": self._fix_syntax_error,
            "zig_type_error": self._fix_type_error,
            "zig_undefined_error": self._fix_undefined_error,
            "zig_memory_error": self._fix_memory_error,
            "zig_import_error": self._fix_import_error,
            "zig_comptime_error": self._fix_comptime_error,
            "zig_async_error": self._fix_async_error,
            "zig_optional_unwrap_null": self._fix_optional_error,
            "zig_error_not_handled": self._fix_error_union_error,
            "zig_bounds_error": self._fix_bounds_error,
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
        """Fix Zig syntax errors."""
        message = error_data.get("message", "")

        fixes = []

        if "expected" in message.lower() and "found" in message.lower():
            # Extract expected and found tokens
            expected_match = re.search(r"expected (.+?), found", message)
            found_match = re.search(r"found (.+)", message)

            if expected_match and found_match:
                expected = expected_match.group(1)
                found = found_match.group(1)

                # Special handling for semicolon
                if expected == "';'":
                    return {
                        "type": "suggestion",
                        "description": "Missing semicolon at end of statement",
                        "fix": "Add semicolon after the statement",
                    }

                fixes.append(
                    {
                        "type": "suggestion",
                        "description": f"Expected {expected}, but found {found}",
                        "fix": f"Replace '{found}' with '{expected}' or fix the syntax structure",
                    }
                )

        if "unterminated string literal" in message.lower():
            fixes.append(
                {
                    "type": "suggestion",
                    "description": "Unterminated string literal",
                    "fix": "Add missing closing quote for string literal",
                }
            )

        if "unexpected token" in message.lower():
            fixes.append(
                {
                    "type": "suggestion",
                    "description": "Unexpected token found",
                    "fix": "Check for syntax errors like missing semicolons, brackets, or keywords",
                }
            )

        if fixes:
            return {
                "type": "multiple_suggestions",
                "fixes": fixes,
                "description": "Zig syntax error fixes",
            }

        return {
            "type": "suggestion",
            "description": "Zig syntax error. Check code structure and syntax",
        }

    def _fix_type_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix Zig type errors."""
        message = error_data.get("message", "")

        if "expected type" in message.lower() and "found" in message.lower():
            # Extract type information
            type_match = re.search(r"expected type '(.+?)', found '(.+?)'", message)
            if type_match:
                expected_type = type_match.group(1)
                found_type = type_match.group(2)

                return {
                    "type": "suggestion",
                    "description": f"Type mismatch: expected {expected_type}, found {found_type}",
                    "fixes": [
                        f"Cast value to {expected_type}: @as({expected_type}, value)",
                        f"Change variable type to {found_type}",
                        "Convert value to correct type using appropriate functions",
                        "Check if types are compatible or add explicit conversion",
                    ],
                }

        if "integer overflow" in message.lower():
            return {
                "type": "suggestion",
                "description": "Integer overflow detected",
                "fixes": [
                    "Use larger integer type (i32 -> i64, etc.)",
                    "Add overflow checking with @addWithOverflow, @mulWithOverflow",
                    "Use wrapping arithmetic: +% instead of +",
                    "Check value ranges before arithmetic operations",
                ],
            }

        if "cannot cast" in message.lower():
            return {
                "type": "suggestion",
                "description": "Invalid type cast",
                "fixes": [
                    "Use @as() for safe type coercion",
                    "Use @intCast() for integer conversions",
                    "Use @floatCast() for floating point conversions",
                    "Check if cast is necessary or find alternative approach",
                ],
            }

        return {
            "type": "suggestion",
            "description": "Type system error. Check type compatibility and conversions",
        }

    def _fix_undefined_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix undefined identifier errors."""
        message = error_data.get("message", "")

        # Extract identifier name
        identifier_match = re.search(r"use of undeclared identifier '(.+?)'", message)
        if not identifier_match:
            identifier_match = re.search(r"'(.+?)' not found", message)

        identifier = identifier_match.group(1) if identifier_match else "identifier"

        return {
            "type": "suggestion",
            "description": f"Undefined identifier '{identifier}'",
            "fixes": [
                f"Declare variable or function '{identifier}'",
                f"Check spelling of '{identifier}'",
                f"Import module containing '{identifier}'",
                f"Add const/var declaration: const {identifier} = value;",
                "Check if identifier is in scope",
            ],
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
                    "Check allocator has sufficient memory",
                    "Free unused allocations with allocator.free()",
                    "Use arena allocator for temporary allocations",
                    "Reduce memory usage or increase available memory",
                ],
            }

        if "null pointer dereference" in message.lower():
            return {
                "type": "suggestion",
                "description": "Null pointer dereference",
                "fixes": [
                    "Check pointer is not null before dereferencing",
                    "Use if (ptr) |p| { ... } for safe pointer access",
                    "Initialize pointer before use",
                    "Use optional types (?*T) for nullable pointers",
                ],
            }

        if "use after free" in message.lower():
            return {
                "type": "suggestion",
                "description": "Use after free detected",
                "fixes": [
                    "Don't access memory after calling allocator.free()",
                    "Set pointer to null after freeing: ptr = null;",
                    "Use arena allocator for automatic cleanup",
                    "Review memory management logic",
                ],
            }

        return {
            "type": "suggestion",
            "description": "Memory management error",
            "fixes": [
                "Check memory allocation and deallocation",
                "Use appropriate allocator for the use case",
                "Avoid memory leaks and use-after-free bugs",
            ],
        }

    def _fix_import_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix import and module errors."""
        message = error_data.get("message", "")

        if "unable to find" in message.lower():
            # Extract module name
            module_match = re.search(r"unable to find '(.+?)'", message)
            module_name = module_match.group(1) if module_match else "module"

            return {
                "type": "suggestion",
                "description": f"Import error - module '{module_name}' not found",
                "fixes": [
                    f"Check path to '{module_name}' is correct",
                    f"Ensure '{module_name}' file exists",
                    "Use relative paths from current file",
                    'Check import syntax: @import("path/to/file.zig")',
                    "Add module to build.zig dependencies if external",
                ],
            }

        if "circular dependency" in message.lower():
            return {
                "type": "suggestion",
                "description": "Circular dependency detected",
                "fixes": [
                    "Restructure modules to avoid circular imports",
                    "Move shared code to separate module",
                    "Use forward declarations where possible",
                    "Review module architecture and dependencies",
                ],
            }

        return {
            "type": "suggestion",
            "description": "Import/module error - check import paths",
            "fixes": [
                "Check import paths and module structure",
                "Verify file exists and is accessible",
                "Review module dependencies",
            ],
        }

    def _fix_comptime_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix compile-time evaluation errors."""
        return {
            "type": "suggestion",
            "description": "Comptime evaluation error - unable to evaluate constant expression",
            "fixes": [
                "Ensure comptime expressions use only compile-time known values",
                "Use comptime keyword for compile-time variables",
                "Check that functions called at comptime are comptime-compatible",
                "Avoid runtime operations in comptime context",
            ],
        }

    def _fix_async_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix async/await errors."""
        message = error_data.get("message", "")

        if "async function cannot be called directly" in message.lower():
            return {
                "type": "suggestion",
                "description": "Async function called directly",
                "fixes": [
                    "Use 'await' to call async function: await function()",
                    "Call from async context or use async/await pattern",
                    "Consider making caller function async",
                    "Use frame allocation for async calls",
                ],
            }

        if "await in non-async function" in message.lower():
            return {
                "type": "suggestion",
                "description": "Await used in non-async function",
                "fixes": [
                    "Make function async: async fn functionName()",
                    "Remove await and call function synchronously",
                    "Use suspend/resume pattern for custom async handling",
                ],
            }

        return {
            "type": "suggestion",
            "description": "Async/concurrency error",
            "fixes": [
                "Check async/await usage and function signatures",
                "Ensure proper async context and frame management",
                "Review concurrency patterns and suspend points",
            ],
        }

    def _fix_optional_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix optional type errors."""
        return {
            "type": "suggestion",
            "description": "Optional value unwrapping error",
            "fixes": [
                "Check for null before unwrapping: if (optional) |value| { ... }",
                "Use orelse for default value: optional orelse default_value",
                "Use try for error handling: try optional",
                "Handle null case explicitly with if/else",
            ],
        }

    def _fix_error_union_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix error union handling errors."""
        return {
            "type": "suggestion",
            "description": "Error not handled properly",
            "fixes": [
                "Use try to propagate error: try function_call()",
                "Handle error with catch: function_call() catch |err| { ... }",
                "Use if for error checking: if (function_call()) |value| { ... } else |err| { ... }",
                "Add error to function signature or handle locally",
            ],
        }

    def _fix_bounds_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix array/slice bounds errors."""
        return {
            "type": "suggestion",
            "description": "Array/slice bounds error",
            "fixes": [
                "Check index is within bounds: if (index < array.len)",
                "Use range-based access: array[start..end]",
                "Validate indices before array access",
                "Use length checks to prevent out-of-bounds access",
            ],
        }

    def _template_based_patch(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Generate patch using templates."""
        root_cause = analysis.get("root_cause", "")

        # Map root causes to template names
        template_map = {
            "zig_syntax_error": "syntax_fix",
            "zig_type_error": "type_fix",
            "zig_optional_unwrap_null": "optional_fix",
            "zig_error_not_handled": "error_union_fix",
            "zig_memory_error": "memory_fix",
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


class ZigLanguagePlugin(LanguagePlugin):
    """
    Main Zig language plugin for Homeostasis.

    This plugin orchestrates Zig error analysis and patch generation,
    supporting systems programming patterns and Zig-specific concepts.
    """

    VERSION = "1.0.0"
    AUTHOR = "Homeostasis Team"

    def __init__(self):
        """Initialize the Zig language plugin."""
        self.language = "zig"
        self.supported_extensions = {".zig"}
        self.supported_frameworks = ["zig", "build.zig", "zig-build", "zigmod", "gyro"]

        # Initialize components
        self.exception_handler = ZigExceptionHandler()
        self.patch_generator = ZigPatchGenerator()

        logger.info("Zig language plugin initialized")

    def get_language_id(self) -> str:
        """Get the unique identifier for this language."""
        return "zig"

    def get_language_name(self) -> str:
        """Get the human-readable name of the language."""
        return "Zig"

    def get_language_version(self) -> str:
        """Get the version of the language supported by this plugin."""
        return "0.11+"

    def get_supported_frameworks(self) -> List[str]:
        """Get the list of frameworks supported by this language plugin."""
        return self.supported_frameworks

    def normalize_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize error data to the standard Homeostasis format.

        Args:
            error_data: Error data in the Zig-specific format

        Returns:
            Error data in the standard format
        """
        # Map Zig-specific error fields to standard format
        normalized = {
            "error_type": error_data.get("error_type", "ZigError"),
            "message": error_data.get("message", error_data.get("description", "")),
            "language": "zig",
            "file_path": error_data.get("file_path", error_data.get("file", "")),
            "line_number": error_data.get("line_number", error_data.get("line", 0)),
            "column_number": error_data.get(
                "column_number", error_data.get("column", 0)
            ),
            "compiler_version": error_data.get("compiler_version", ""),
            "target": error_data.get("target", ""),
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
        Convert standard format error data back to the Zig-specific format.

        Args:
            standard_error: Error data in the standard format

        Returns:
            Error data in the Zig-specific format
        """
        # Map standard fields back to Zig-specific format
        zig_error = {
            "error_type": standard_error.get("error_type", "ZigError"),
            "message": standard_error.get("message", ""),
            "file_path": standard_error.get("file_path", ""),
            "line_number": standard_error.get("line_number", 0),
            "column_number": standard_error.get("column_number", 0),
            "compiler_version": standard_error.get("compiler_version", ""),
            "target": standard_error.get("target", ""),
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
            if key not in zig_error and value is not None:
                zig_error[key] = value

        return zig_error

    def analyze_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a Zig error.

        Args:
            error_data: Zig error data

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
            analysis["plugin"] = "zig"
            analysis["language"] = "zig"
            analysis["plugin_version"] = self.VERSION

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing Zig error: {e}")
            return {
                "category": "zig",
                "subcategory": "unknown",
                "confidence": "low",
                "suggested_fix": "Unable to analyze Zig error",
                "error": str(e),
                "plugin": "zig",
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
register_plugin(ZigLanguagePlugin())
