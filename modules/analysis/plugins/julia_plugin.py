"""
Julia Language Plugin for Homeostasis

This plugin enables Homeostasis to analyze and fix errors in Julia programming language code.
It provides comprehensive error handling for Julia syntax errors, runtime issues,
and high-performance scientific computing best practices.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..language_plugin_system import LanguagePlugin, register_plugin

logger = logging.getLogger(__name__)


class JuliaExceptionHandler:
    """
    Handles Julia exceptions with robust error detection and classification.

    This class provides logic for categorizing Julia errors based on their type,
    message, and common high-performance computing patterns.
    """

    def __init__(self):
        """Initialize the Julia exception handler."""
        self.rule_categories = {
            "syntax": "Julia syntax and parsing errors",
            "runtime": "Runtime errors and exceptions",
            "type": "Type system and method dispatch errors",
            "bounds": "Array bounds and indexing errors",
            "method": "Method definition and dispatch errors",
            "package": "Package loading and dependency errors",
            "performance": "Performance and optimization errors",
            "memory": "Memory allocation and garbage collection errors",
            "parallel": "Parallel computing and threading errors",
            "io": "Input/output and file handling errors",
            "macro": "Macro and metaprogramming errors",
            "interop": "Foreign function interface and interoperability errors",
        }

        # Common Julia error patterns
        self.julia_error_patterns = {
            "syntax_error": [
                r"syntax:.*unexpected.*end.*input",
                r"syntax:.*unexpected.*token",
                r"syntax:.*incomplete.*expression",
                r"syntax:.*invalid.*syntax",
                r"ParseError:.*unexpected.*token",
                r"syntax:.*missing.*comma",
                r"syntax:.*extra.*token",
                r"syntax:.*incomplete.*string",
                r"syntax:.*unmatched.*delimiter",
            ],
            "runtime_error": [
                r"UndefVarError:(?!.*@\w+).*not defined",
                r"ArgumentError:.*invalid.*argument",
                r"ErrorException:.*error.*occurred",
                r"InterruptException:.*interrupted",
                r"StackOverflowError:.*stack overflow",
                r"OutOfMemoryError:.*out.*memory",
            ],
            "type_error": [
                r"MethodError:.*no method matching",
                r"TypeError:.*cannot.*convert",
                r"InexactError:.*cannot.*convert",
                r"TypeError:.*expected.*got",
                r"MethodError:.*cannot.*promote",
                r"BoundsError:.*attempt.*access",
                r"ArgumentError:.*invalid.*type",
                r"DomainError:.*not.*in.*domain",
            ],
            "bounds_error": [
                r"BoundsError:.*attempt.*access",
                r"BoundsError:.*index.*out.*range",
                r"ArgumentError:.*dimension.*mismatch",
                r"DimensionMismatch:.*dimensions.*must.*match",
                r"BoundsError:.*index.*beyond.*end",
                r"BoundsError:.*invalid.*index",
            ],
            "method_error": [
                r"MethodError:.*no method matching",
                r"MethodError:.*ambiguous.*method",
                r"MethodError:.*method.*not.*defined",
                r"UndefVarError:.*function.*not defined",
                r"MethodError:.*cannot.*call",
                r"ArgumentError:.*wrong.*number.*arguments",
            ],
            "package_error": [
                r"ArgumentError:.*Package.*not found",
                r"PkgError:.*package.*not.*available",
                r"LoadError:.*failed.*to.*load",
                r"UndefVarError:.*Module.*not.*defined",
                r"SystemError:.*could.*not.*load.*library",
                r"InitError:.*failed.*to.*initialize",
            ],
            "performance_error": [
                r"MethodError:.*type.*not.*concrete",
                r"MethodError:.*cannot.*infer.*type",
                r"ArgumentError:.*abstract.*type",
                r"TypeError:.*type.*instability",
                r"MethodError:.*dynamic.*dispatch",
                r"MethodError:.*type.*inference.*failed",
            ],
            "memory_error": [
                r"OutOfMemoryError:.*out.*memory",
                r"MemoryError:.*cannot.*allocate",
                r"SystemError:.*memory.*exhausted",
                r"GC.*error:.*garbage.*collection",
                r"AllocationError:.*allocation.*failed",
            ],
            "parallel_error": [
                r"TaskFailedException:.*task.*failed",
                r"RemoteException:.*remote.*error",
                r"ProcessExitedException:.*process.*exited",
                r"CompositeException:.*multiple.*exceptions",
                r"InterruptException:.*interrupted",
                r"ConcurrencyViolationError:.*concurrency.*violation",
            ],
            "io_error": [
                r"SystemError:.*no.*such.*file",
                r"SystemError:.*permission.*denied",
                r"ArgumentError:.*invalid.*file",
                r"IOError:.*input.*output.*error",
                r"EOFError:.*end.*of.*file",
                r"SystemError:.*file.*not.*found",
            ],
            "macro_error": [
                r"LoadError:.*@\w+.*not.*defined",
                r"LoadError:.*macro.*error",
                r"UndefVarError:.*@\w+.*not.*defined",
                r"MethodError:.*macro.*not.*applicable",
                r"ArgumentError:.*invalid.*macro",
                r"ParseError:.*macro.*syntax",
            ],
            "interop_error": [
                r"ccall:.*library.*not.*found",
                r"ccall:.*symbol.*not.*found",
                r"ArgumentError:.*invalid.*ccall",
                r"SystemError:.*foreign.*function",
                r"TypeError:.*ccall.*argument",
                r"LoadError:.*shared.*library",
            ],
        }

        # Julia-specific concepts and their common issues
        self.julia_concepts = {
            "arrays": ["array", "vector", "matrix", "bounds", "index"],
            "types": ["type", "struct", "abstract", "concrete", "parametric"],
            "methods": ["method", "dispatch", "multiple", "function"],
            "packages": ["package", "using", "import", "module"],
            "performance": ["type", "inference", "concrete", "stable"],
            "macros": ["macro", "metaprogramming", "quote", "eval"],
            "parallel": ["task", "thread", "parallel", "distributed"],
            "interop": ["ccall", "foreign", "library", "external"],
            "memory": ["garbage", "collection", "allocation", "memory"],
            "io": ["file", "read", "write", "stream", "io"],
        }

        # Load rules from different categories
        self.rules = self._load_rules()

        # Pre-compile regex patterns for better performance
        self._compile_patterns()

    def _load_rules(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load Julia error rules from rule files."""
        rules = {}
        rules_dir = Path(__file__).parent.parent / "rules" / "julia"

        try:
            # Load common Julia rules
            common_rules_path = rules_dir / "julia_common_errors.json"
            if common_rules_path.exists():
                with open(common_rules_path, "r") as f:
                    common_data = json.load(f)
                    rules["common"] = common_data.get("rules", [])
                    logger.info(f"Loaded {len(rules['common'])} common Julia rules")

            # Load concept-specific rules
            for concept in ["types", "methods", "packages", "performance"]:
                concept_rules_path = rules_dir / f"julia_{concept}_errors.json"
                if concept_rules_path.exists():
                    with open(concept_rules_path, "r") as f:
                        concept_data = json.load(f)
                        rules[concept] = concept_data.get("rules", [])
                        logger.info(f"Loaded {len(rules[concept])} {concept} rules")

        except Exception as e:
            logger.error(f"Error loading Julia rules: {e}")
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
        Analyze a Julia exception and determine its type and potential fixes.

        Args:
            error_data: Julia error data in standard format

        Returns:
            Analysis results with categorization and fix suggestions
        """
        message = error_data.get("message", "")
        file_path = error_data.get("file_path", "")
        line_number = error_data.get("line_number", 0)

        # Check for concept-specific issues first (higher priority)
        concept_analysis = self._analyze_julia_concepts(message)
        if concept_analysis.get("confidence", "low") != "low":
            analysis = concept_analysis
        else:
            # Analyze based on error patterns
            analysis = self._analyze_by_patterns(message, file_path)

        # Find matching rules
        matches = self._find_matching_rules(message, error_data)

        if matches:
            # Use the best match (highest confidence)
            best_match = max(matches, key=lambda x: x.get("confidence_score", 0))
            # Map confidence strings to numeric values for comparison
            confidence_values = {"high": 3, "medium": 2, "low": 1}
            original_confidence_val = confidence_values.get(
                analysis.get("confidence", "low"), 1
            )
            rule_confidence_val = confidence_values.get(
                best_match.get("confidence", "low"), 1
            )

            # Use the higher confidence
            final_confidence = (
                analysis.get("confidence")
                if original_confidence_val >= rule_confidence_val
                else best_match.get("confidence")
            )

            # Merge tags from both sources
            original_tags = set(analysis.get("tags", []))
            rule_tags = set(best_match.get("tags", []))
            merged_tags = list(original_tags | rule_tags)

            analysis.update(
                {
                    "category": analysis.get(
                        "category", "unknown"
                    ),  # Keep original category
                    "subcategory": analysis.get(
                        "subcategory", best_match.get("category", "unknown")
                    ),  # Keep original subcategory or use rule category
                    "confidence": final_confidence,
                    "suggested_fix": best_match.get(
                        "suggestion", analysis.get("suggested_fix", "")
                    ),
                    "root_cause": best_match.get(
                        "root_cause", analysis.get("root_cause", "")
                    ),
                    "severity": best_match.get(
                        "severity", analysis.get("severity", "medium")
                    ),
                    "rule_id": best_match.get("id", ""),
                    "tags": merged_tags,
                    "all_matches": matches,
                }
            )

        analysis["file_path"] = file_path
        analysis["line_number"] = line_number
        return analysis

    def _analyze_by_patterns(self, message: str, file_path: str) -> Dict[str, Any]:
        """Analyze error by matching against common patterns."""
        # Check syntax errors
        for pattern in self.julia_error_patterns["syntax_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "julia",
                    "subcategory": "syntax",
                    "confidence": "high",
                    "suggested_fix": "Fix Julia syntax errors",
                    "root_cause": "julia_syntax_error",
                    "severity": "high",
                    "tags": ["julia", "syntax", "parser"],
                }

        # Check runtime errors
        for pattern in self.julia_error_patterns["runtime_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "julia",
                    "subcategory": "runtime",
                    "confidence": "high",
                    "suggested_fix": "Fix runtime errors and variable access",
                    "root_cause": "julia_runtime_error",
                    "severity": "high",
                    "tags": ["julia", "runtime", "variable"],
                }

        # Check type errors
        for pattern in self.julia_error_patterns["type_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "julia",
                    "subcategory": "type",
                    "confidence": "high",
                    "suggested_fix": "Fix type system and method dispatch errors",
                    "root_cause": "julia_type_error",
                    "severity": "high",
                    "tags": ["julia", "type", "dispatch"],
                }

        # Check bounds errors
        for pattern in self.julia_error_patterns["bounds_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "julia",
                    "subcategory": "bounds",
                    "confidence": "high",
                    "suggested_fix": "Fix array bounds and indexing errors",
                    "root_cause": "julia_bounds_error",
                    "severity": "high",
                    "tags": ["julia", "bounds", "array"],
                }

        # Check method errors
        for pattern in self.julia_error_patterns["method_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "julia",
                    "subcategory": "method",
                    "confidence": "high",
                    "suggested_fix": "Fix method definition and dispatch errors",
                    "root_cause": "julia_method_error",
                    "severity": "high",
                    "tags": ["julia", "method", "dispatch"],
                }

        # Check package errors
        for pattern in self.julia_error_patterns["package_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "julia",
                    "subcategory": "package",
                    "confidence": "high",
                    "suggested_fix": "Fix package loading and dependency errors",
                    "root_cause": "julia_package_error",
                    "severity": "high",
                    "tags": ["julia", "package", "dependency"],
                }

        # Check performance errors
        for pattern in self.julia_error_patterns["performance_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "julia",
                    "subcategory": "performance",
                    "confidence": "high",
                    "suggested_fix": "Fix performance and type stability errors",
                    "root_cause": "julia_performance_error",
                    "severity": "medium",
                    "tags": ["julia", "performance", "type"],
                }

        # Check memory errors
        for pattern in self.julia_error_patterns["memory_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "julia",
                    "subcategory": "memory",
                    "confidence": "high",
                    "suggested_fix": "Fix memory allocation and garbage collection errors",
                    "root_cause": "julia_memory_error",
                    "severity": "critical",
                    "tags": ["julia", "memory", "gc"],
                }

        # Check parallel errors
        for pattern in self.julia_error_patterns["parallel_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "julia",
                    "subcategory": "parallel",
                    "confidence": "high",
                    "suggested_fix": "Fix parallel computing and threading errors",
                    "root_cause": "julia_parallel_error",
                    "severity": "medium",
                    "tags": ["julia", "parallel", "thread"],
                }

        # Check IO errors
        for pattern in self.julia_error_patterns["io_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "julia",
                    "subcategory": "io",
                    "confidence": "high",
                    "suggested_fix": "Fix input/output and file handling errors",
                    "root_cause": "julia_io_error",
                    "severity": "high",
                    "tags": ["julia", "io", "file"],
                }

        # Check macro errors
        for pattern in self.julia_error_patterns["macro_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "julia",
                    "subcategory": "macro",
                    "confidence": "high",
                    "suggested_fix": "Fix macro and metaprogramming errors",
                    "root_cause": "julia_macro_error",
                    "severity": "medium",
                    "tags": ["julia", "macro", "metaprogramming"],
                }

        # Check interop errors
        for pattern in self.julia_error_patterns["interop_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "julia",
                    "subcategory": "interop",
                    "confidence": "high",
                    "suggested_fix": "Fix foreign function interface and interoperability errors",
                    "root_cause": "julia_interop_error",
                    "severity": "medium",
                    "tags": ["julia", "interop", "ffi"],
                }

        return {
            "category": "julia",
            "subcategory": "unknown",
            "confidence": "low",
            "suggested_fix": "Review Julia code and error details",
            "root_cause": "julia_generic_error",
            "severity": "medium",
            "tags": ["julia", "generic"],
        }

    def _analyze_julia_concepts(self, message: str) -> Dict[str, Any]:
        """Analyze Julia-specific concept errors."""
        message_lower = message.lower()

        # Check for macro errors first (before UndefVarError)
        if "@" in message and any(
            keyword in message_lower for keyword in ["macro", "not defined"]
        ):
            return {
                "category": "julia",
                "subcategory": "macro",
                "confidence": "high",
                "suggested_fix": "Check macro definition and usage",
                "root_cause": "julia_macro_error",
                "severity": "medium",
                "tags": ["julia", "macro", "metaprogramming"],
            }

        # Check for type-related MethodError (e.g., operations between incompatible types)
        if "methoderror" in message_lower and any(
            op in message for op in ["+", "-", "*", "/", "^", "%"]
        ):
            return {
                "category": "julia",
                "subcategory": "type",
                "confidence": "high",
                "suggested_fix": "Check type compatibility for the operation",
                "root_cause": "julia_type_error",
                "severity": "high",
                "tags": ["julia", "type", "operation"],
            }

        # Check for MethodError with dispatch
        if "methoderror" in message_lower and "no method matching" in message_lower:
            return {
                "category": "julia",
                "subcategory": "dispatch",
                "confidence": "high",
                "suggested_fix": "Check method signatures and type annotations",
                "root_cause": "julia_dispatch_error",
                "severity": "high",
                "tags": ["julia", "method", "dispatch"],
            }

        # Check for UndefVarError
        if (
            any(
                keyword in message_lower for keyword in ["undefvarerror", "not defined"]
            ) and
            "@" not in message
        ):
            return {
                "category": "julia",
                "subcategory": "undefined",
                "confidence": "high",
                "suggested_fix": "Check variable/function name and ensure it's defined",
                "root_cause": "julia_undefined_error",
                "severity": "high",
                "tags": ["julia", "undefined", "variable"],
            }

        # Check for MethodError
        if any(
            keyword in message_lower
            for keyword in ["methoderror", "no method matching"]
        ):
            return {
                "category": "julia",
                "subcategory": "method",
                "confidence": "high",
                "suggested_fix": "Check method signatures and type annotations",
                "root_cause": "julia_method_error",
                "severity": "high",
                "tags": ["julia", "method", "dispatch"],
            }

        # Check for BoundsError
        if any(keyword in message_lower for keyword in ["boundserror", "out of range"]):
            return {
                "category": "julia",
                "subcategory": "bounds",
                "confidence": "high",
                "suggested_fix": "Check array bounds and indexing",
                "root_cause": "julia_bounds_error",
                "severity": "high",
                "tags": ["julia", "bounds", "array"],
            }

        # Check for type errors
        if any(keyword in message_lower for keyword in ["typeerror", "cannot convert"]):
            return {
                "category": "julia",
                "subcategory": "type",
                "confidence": "medium",
                "suggested_fix": "Check type conversions and compatibility",
                "root_cause": "julia_type_error",
                "severity": "medium",
                "tags": ["julia", "type", "conversion"],
            }

        # Check for package errors
        if any(
            keyword in message_lower
            for keyword in ["package", "module", "using", "import"]
        ):
            return {
                "category": "julia",
                "subcategory": "import",
                "confidence": "medium",
                "suggested_fix": "Check package installation and imports",
                "root_cause": "julia_import_error",
                "severity": "medium",
                "tags": ["julia", "package", "import"],
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
        if file_path.endswith(".jl"):
            base_confidence += 0.2

        # Boost confidence based on rule reliability
        reliability = rule.get("reliability", "medium")
        reliability_boost = {"high": 0.2, "medium": 0.1, "low": 0.0}
        base_confidence += reliability_boost.get(reliability, 0.0)

        # Boost confidence for concept matches
        rule_tags = set(rule.get("tags", []))
        context_tags = set()

        message = error_data.get("message", "").lower()
        if "method" in message:
            context_tags.add("method")
        if "type" in message:
            context_tags.add("type")
        if "bounds" in message:
            context_tags.add("bounds")

        if context_tags & rule_tags:
            base_confidence += 0.1

        return min(base_confidence, 1.0)


class JuliaPatchGenerator:
    """
    Generates patches for Julia errors based on analysis results.

    This class creates Julia code fixes for common errors using templates
    and heuristics specific to high-performance computing patterns.
    """

    def __init__(self):
        """Initialize the Julia patch generator."""
        self.template_dir = (
            Path(__file__).parent.parent / "patch_generation" / "templates"
        )
        self.julia_template_dir = self.template_dir / "julia"

        # Ensure template directory exists
        self.julia_template_dir.mkdir(parents=True, exist_ok=True)

        # Load patch templates
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict[str, str]:
        """Load Julia patch templates."""
        templates = {}

        if not self.julia_template_dir.exists():
            logger.warning(
                f"Julia templates directory not found: {self.julia_template_dir}"
            )
            return templates

        for template_file in self.julia_template_dir.glob("*.jl.template"):
            try:
                with open(template_file, "r") as f:
                    template_name = template_file.stem.replace(".jl", "")
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
        Generate a patch for the Julia error.

        Args:
            error_data: The Julia error data
            analysis: Analysis results from JuliaExceptionHandler
            source_code: The Julia source code that caused the error

        Returns:
            Patch information or None if no patch can be generated
        """
        root_cause = analysis.get("root_cause", "")

        # Map root causes to patch strategies
        patch_strategies = {
            "julia_syntax_error": self._fix_syntax_error,
            "julia_runtime_error": self._fix_runtime_error,
            "julia_type_error": self._fix_type_error,
            "julia_bounds_error": self._fix_bounds_error,
            "julia_method_error": self._fix_method_error,
            "julia_package_error": self._fix_package_error,
            "julia_import_error": self._fix_package_error,  # Same as package error
            "julia_performance_error": self._fix_performance_error,
            "julia_memory_error": self._fix_memory_error,
            "julia_parallel_error": self._fix_parallel_error,
            "julia_io_error": self._fix_io_error,
            "julia_macro_error": self._fix_macro_error,
            "julia_interop_error": self._fix_interop_error,
            "julia_undefined_error": self._fix_undefined_error,
            "julia_dispatch_error": self._fix_method_error,  # Same as method error
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
        """Fix Julia syntax errors."""
        message = error_data.get("message", "")

        fixes = []

        if "unexpected token" in message.lower():
            fixes.append(
                {
                    "type": "suggestion",
                    "description": "Unexpected token in code",
                    "fix": "Check for missing operators, commas, or parentheses",
                }
            )

        if "incomplete expression" in message.lower():
            fixes.append(
                {
                    "type": "suggestion",
                    "description": "Incomplete expression",
                    "fix": "Complete the expression or add missing closing symbols",
                }
            )

        if "unmatched delimiter" in message.lower():
            fixes.append(
                {
                    "type": "suggestion",
                    "description": "Unmatched delimiter",
                    "fix": "Check for matching parentheses, brackets, or braces",
                }
            )

        if fixes:
            # Return single suggestion if only one fix
            if len(fixes) == 1:
                return {
                    "type": "suggestion",
                    "description": fixes[0]["description"],
                    "fix": fixes[0]["fix"],
                }
            return {
                "type": "multiple_suggestions",
                "fixes": fixes,
                "description": "Julia syntax error fixes",
            }

        return {
            "type": "suggestion",
            "description": "Julia syntax error. Check code structure and syntax",
        }

    def _fix_runtime_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix runtime errors."""
        message = error_data.get("message", "")

        if "UndefVarError" in message:
            # Extract variable name
            var_match = re.search(r"UndefVarError: `(.+?)` not defined", message)
            var_name = var_match.group(1) if var_match else "variable"

            return {
                "type": "suggestion",
                "description": f"Undefined variable '{var_name}'",
                "fixes": [
                    f"Define variable '{var_name}' before use",
                    f"Check spelling of '{var_name}'",
                    f"Import package containing '{var_name}' with using or import",
                    f"Use @isdefined({var_name}) to check if variable exists",
                    "Check variable scope in functions and let blocks",
                ],
            }

        if "MethodError" in message and "no method matching" in message:
            return {
                "type": "suggestion",
                "description": "Method not found for given arguments",
                "fixes": [
                    "Check method signatures and argument types",
                    "Define method for the specific argument types",
                    "Use methods() to see available methods",
                    "Check for typos in function name",
                    "Import package containing the method",
                ],
            }

        if "ArgumentError" in message:
            return {
                "type": "suggestion",
                "description": "Invalid argument error",
                "fixes": [
                    "Check argument values and types",
                    "Verify argument constraints and ranges",
                    "Use appropriate argument conversion",
                    "Check function documentation for valid arguments",
                ],
            }

        return {
            "type": "suggestion",
            "description": "Runtime error. Check variable and function definitions",
        }

    def _fix_type_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix type system errors."""
        message = error_data.get("message", "")

        if "TypeError" in message:
            return {
                "type": "suggestion",
                "description": "Type error",
                "fixes": [
                    "Check type annotations and constraints",
                    "Use convert() or appropriate type conversion",
                    "Verify type compatibility",
                    "Use typeof() to check variable types",
                    "Consider using Union types for multiple types",
                ],
            }

        if "InexactError" in message:
            return {
                "type": "suggestion",
                "description": "Inexact type conversion",
                "fixes": [
                    "Use appropriate type conversion functions",
                    "Check for precision loss in conversions",
                    "Use floor(), ceil(), or round() for numeric conversions",
                    "Consider using wider numeric types",
                ],
            }

        return {
            "type": "suggestion",
            "description": "Type system error",
            "fixes": [
                "Check type annotations and conversions",
                "Verify type compatibility",
                "Use appropriate type conversion functions",
            ],
        }

    def _fix_bounds_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix array bounds errors."""
        message = error_data.get("message", "")

        if "BoundsError" in message:
            return {
                "type": "suggestion",
                "description": "Array bounds error",
                "fixes": [
                    "Check array dimensions with size() or length()",
                    "Use valid indices within array bounds",
                    "Use eachindex() for safe iteration",
                    "Check bounds with checkbounds()",
                    "Use 1-based indexing (Julia arrays start at 1)",
                    "Consider using get() for safe access with defaults",
                ],
            }

        if "DimensionMismatch" in message:
            return {
                "type": "suggestion",
                "description": "Array dimension mismatch",
                "fixes": [
                    "Check array dimensions with size()",
                    "Use reshape() to adjust array dimensions",
                    "Verify array compatibility for operations",
                    "Use broadcasting (.) for element-wise operations",
                    "Check matrix multiplication requirements",
                ],
            }

        return {
            "type": "suggestion",
            "description": "Array bounds error",
            "fixes": [
                "Check array dimensions and indices",
                "Use appropriate array access methods",
                "Verify array bounds before access",
            ],
        }

    def _fix_method_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix method dispatch errors."""
        message = error_data.get("message", "")

        if "no method matching" in message.lower():
            return {
                "type": "suggestion",
                "description": "No method matching arguments",
                "fixes": [
                    "Define method for the specific argument types",
                    "Check argument types and convert if needed",
                    "Use methods() to see available methods",
                    "Check for typos in function name",
                    "Import required packages for methods",
                ],
            }

        if "ambiguous" in message.lower():
            return {
                "type": "suggestion",
                "description": "Ambiguous method call",
                "fixes": [
                    "Specify more concrete types in method definitions",
                    "Use type annotations to disambiguate",
                    "Resolve method ambiguities with more specific signatures",
                    "Check method ordering and specificity",
                ],
            }

        return {
            "type": "suggestion",
            "description": "Method dispatch error",
            "fixes": [
                "Check method signatures and argument types",
                "Define required methods for types",
                "Use appropriate type annotations",
            ],
        }

    def _fix_package_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix package loading errors."""
        message = error_data.get("message", "")

        if "package" in message.lower() and "not found" in message.lower():
            return {
                "type": "suggestion",
                "description": "Package not found",
                "fixes": [
                    "Install package using Pkg.add()",
                    "Check package name spelling",
                    "Update package registry with Pkg.update()",
                    "Use Pkg.status() to check installed packages",
                    "Check package availability in registries",
                ],
            }

        if "failed to load" in message.lower():
            return {
                "type": "suggestion",
                "description": "Failed to load package",
                "fixes": [
                    "Check package dependencies",
                    "Reinstall package with Pkg.add()",
                    "Check for package version compatibility",
                    "Use Pkg.resolve() to fix dependency issues",
                    "Check for conflicting package versions",
                ],
            }

        return {
            "type": "suggestion",
            "description": "Package loading error",
            "fixes": [
                "Check package installation and dependencies",
                "Verify package names and imports",
                "Update packages if needed",
            ],
        }

    def _fix_performance_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix performance and type stability errors."""
        message = error_data.get("message", "")

        if "type" in message.lower() and "not concrete" in message.lower():
            return {
                "type": "suggestion",
                "description": "Type not concrete",
                "fixes": [
                    "Use concrete types instead of abstract types",
                    "Add type annotations to function signatures",
                    "Use type parameters for generic functions",
                    "Check type stability with @code_warntype",
                    "Avoid type-unstable operations",
                ],
            }

        if "dynamic dispatch" in message.lower():
            return {
                "type": "suggestion",
                "description": "Dynamic dispatch detected",
                "fixes": [
                    "Use concrete types to enable static dispatch",
                    "Add type annotations to variables",
                    "Use function barriers for type-unstable code",
                    "Check performance with @benchmark",
                    "Optimize hot loops with type stability",
                ],
            }

        return {
            "type": "suggestion",
            "description": "Performance error",
            "fixes": [
                "Use concrete types for better performance",
                "Add type annotations where needed",
                "Check type stability and dispatch",
            ],
        }

    def _fix_memory_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix memory allocation errors."""
        message = error_data.get("message", "")

        if "OutOfMemoryError" in message:
            return {
                "type": "suggestion",
                "description": "Out of memory error",
                "fixes": [
                    "Reduce memory usage with smaller arrays",
                    "Process data in chunks",
                    "Use memory-efficient data structures",
                    "Call GC.gc() to force garbage collection",
                    "Use @allocated to track memory allocations",
                ],
            }

        return {
            "type": "suggestion",
            "description": "Memory error",
            "fixes": [
                "Optimize memory usage",
                "Use efficient data structures",
                "Manage memory allocation carefully",
            ],
        }

    def _fix_parallel_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix parallel computing errors."""
        message = error_data.get("message", "")

        if "TaskFailedException" in message:
            return {
                "type": "suggestion",
                "description": "Task failed in parallel execution",
                "fixes": [
                    "Check task code for errors",
                    "Use try-catch in parallel tasks",
                    "Handle exceptions in @spawn or @async",
                    "Check task dependencies",
                    "Use fetch() to get task results safely",
                ],
            }

        if "RemoteException" in message:
            return {
                "type": "suggestion",
                "description": "Remote worker error",
                "fixes": [
                    "Check remote worker status",
                    "Handle remote exceptions properly",
                    "Use @everywhere for code distribution",
                    "Check worker process health",
                    "Use proper error handling in distributed code",
                ],
            }

        return {
            "type": "suggestion",
            "description": "Parallel computing error",
            "fixes": [
                "Check parallel task execution",
                "Handle exceptions in parallel code",
                "Verify worker processes",
            ],
        }

    def _fix_io_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix IO and file handling errors."""
        message = error_data.get("message", "")

        if "no such file" in message.lower():
            return {
                "type": "suggestion",
                "description": "File not found",
                "fixes": [
                    "Check file path and name",
                    "Use isfile() to check file existence",
                    "Use pwd() to check current directory",
                    "Use abspath() for absolute paths",
                    "Check file permissions",
                ],
            }

        if "permission denied" in message.lower():
            return {
                "type": "suggestion",
                "description": "Permission denied",
                "fixes": [
                    "Check file permissions",
                    "Run with appropriate privileges",
                    "Use different file location",
                    "Check directory permissions",
                ],
            }

        return {
            "type": "suggestion",
            "description": "IO error",
            "fixes": [
                "Check file paths and permissions",
                "Verify file existence and accessibility",
                "Use appropriate file handling functions",
            ],
        }

    def _fix_macro_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix macro and metaprogramming errors."""
        message = error_data.get("message", "")

        if "macro" in message.lower():
            return {
                "type": "suggestion",
                "description": "Macro error",
                "fixes": [
                    "Check macro syntax and usage",
                    "Use @macroexpand to debug macros",
                    "Verify macro arguments and types",
                    "Check for macro namespace conflicts",
                    "Use proper quoting and escaping",
                ],
            }

        return {
            "type": "suggestion",
            "description": "Macro/metaprogramming error",
            "fixes": [
                "Check macro definitions and usage",
                "Use debugging tools for macros",
                "Verify metaprogramming syntax",
            ],
        }

    def _fix_interop_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix foreign function interface errors."""
        message = error_data.get("message", "")

        if "ccall" in message.lower():
            return {
                "type": "suggestion",
                "description": "Foreign function interface error",
                "fixes": [
                    "Check library path and name",
                    "Verify function signature in ccall",
                    "Check argument types and conversion",
                    "Use proper calling conventions",
                    "Check shared library availability",
                ],
            }

        return {
            "type": "suggestion",
            "description": "Interoperability error",
            "fixes": [
                "Check foreign function interface setup",
                "Verify library and function availability",
                "Use appropriate type conversions",
            ],
        }

    def _fix_undefined_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix undefined variable/function errors."""
        return {
            "type": "suggestion",
            "description": "Undefined variable or function",
            "fixes": [
                "Check variable/function name spelling",
                "Define variable/function before use",
                "Import required packages with using or import",
                "Check variable scope in functions",
                "Use @isdefined to check if variable exists",
            ],
        }

    def _template_based_patch(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Generate patch using templates."""
        root_cause = analysis.get("root_cause", "")

        # Map root causes to template names
        template_map = {
            "julia_syntax_error": "syntax_fix",
            "julia_runtime_error": "runtime_fix",
            "julia_type_error": "type_fix",
            "julia_method_error": "method_fix",
            "julia_package_error": "package_fix",
            "julia_bounds_error": "bounds_fix",
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


class JuliaLanguagePlugin(LanguagePlugin):
    """
    Main Julia language plugin for Homeostasis.

    This plugin orchestrates Julia error analysis and patch generation,
    supporting high-performance scientific computing patterns.
    """

    VERSION = "1.0.0"
    AUTHOR = "Homeostasis Team"

    def __init__(self):
        """Initialize the Julia language plugin."""
        self.language = "julia"
        self.supported_extensions = {".jl", ".jlmeta"}
        self.supported_frameworks = [
            "julia",
            "plots",
            "dataframes",
            "flux",
            "jumps",
            "differentialequations",
            "mlj",
            "pluto",
            "genie",
            "distributed",
            "cuda",
            "pkg",
        ]

        # Initialize components
        self.exception_handler = JuliaExceptionHandler()
        self.patch_generator = JuliaPatchGenerator()

        logger.info("Julia language plugin initialized")

    def get_language_id(self) -> str:
        """Get the unique identifier for this language."""
        return "julia"

    def get_language_name(self) -> str:
        """Get the human-readable name of the language."""
        return "Julia"

    def get_language_version(self) -> str:
        """Get the version of the language supported by this plugin."""
        return "1.8+"

    def get_supported_frameworks(self) -> List[str]:
        """Get the list of frameworks supported by this language plugin."""
        return self.supported_frameworks

    def normalize_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize error data to the standard Homeostasis format.

        Args:
            error_data: Error data in the Julia-specific format

        Returns:
            Error data in the standard format
        """
        # Map Julia-specific error fields to standard format
        normalized = {
            "error_type": error_data.get("error_type", "JuliaError"),
            "message": error_data.get("message", error_data.get("description", "")),
            "language": "julia",
            "file_path": error_data.get("file_path", error_data.get("file", "")),
            "line_number": error_data.get("line_number", error_data.get("line", 0)),
            "column_number": error_data.get(
                "column_number", error_data.get("column", 0)
            ),
            "julia_version": error_data.get("julia_version", ""),
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
        Convert standard format error data back to the Julia-specific format.

        Args:
            standard_error: Error data in the standard format

        Returns:
            Error data in the Julia-specific format
        """
        # Map standard fields back to Julia-specific format
        julia_error = {
            "error_type": standard_error.get("error_type", "JuliaError"),
            "message": standard_error.get("message", ""),
            "file_path": standard_error.get("file_path", ""),
            "line_number": standard_error.get("line_number", 0),
            "column_number": standard_error.get("column_number", 0),
            "julia_version": standard_error.get("julia_version", ""),
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
            if key not in julia_error and value is not None:
                julia_error[key] = value

        return julia_error

    def analyze_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a Julia error.

        Args:
            error_data: Julia error data

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
            analysis["plugin"] = "julia"
            analysis["language"] = "julia"
            analysis["plugin_version"] = self.VERSION

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing Julia error: {e}")
            return {
                "category": "julia",
                "subcategory": "unknown",
                "confidence": "low",
                "suggested_fix": "Unable to analyze Julia error",
                "error": str(e),
                "plugin": "julia",
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
register_plugin(JuliaLanguagePlugin())
