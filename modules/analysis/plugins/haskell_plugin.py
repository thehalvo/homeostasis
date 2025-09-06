"""
Haskell Language Plugin for Homeostasis

This plugin enables Homeostasis to analyze and fix errors in Haskell programming language code.
It provides comprehensive error handling for Haskell compilation errors, type system errors,
functional programming patterns, and lazy evaluation issues.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..language_plugin_system import LanguagePlugin, register_plugin

logger = logging.getLogger(__name__)


class HaskellExceptionHandler:
    """
    Handles Haskell exceptions with robust error detection and classification.

    This class provides logic for categorizing Haskell errors based on their type,
    message, and common functional programming patterns.
    """

    def __init__(self):
        """Initialize the Haskell exception handler."""
        self.rule_categories = {
            "syntax": "Haskell syntax and parsing errors",
            "type": "Type system and type checking errors",
            "compilation": "Compilation and build errors",
            "runtime": "Runtime errors and exceptions",
            "import": "Import and module system errors",
            "instance": "Type class instance errors",
            "pattern": "Pattern matching errors",
            "lazy": "Lazy evaluation and strictness errors",
            "io": "Input/output and monadic errors",
            "memory": "Memory and space leak errors",
            "cabal": "Cabal build system errors",
            "ghc": "GHC compiler specific errors",
        }

        # Common Haskell error patterns
        self.haskell_error_patterns = {
            "syntax_error": [
                r"parse error",
                r"Parse error",
                r"unexpected.*?expecting",
                r"missing.*?in.*?expression",
                r"empty.*?do.*?block",
                r"indent.*?error",
                r"lexical error",
                r"unterminated.*?string",
                r"invalid.*?character",
            ],
            "type_error": [
                r"Couldn't match type",
                r"No instance for",
                r"Ambiguous type",
                r"Could not deduce",
                r"Expected type.*?Actual type",
                r"Type constructor.*?used as a type",
                r"Not in scope.*?data constructor",
                r"Occurs check",
                r"Rigid type variable",
                r"Couldn't match expected type",
            ],
            "compilation_error": [
                r"Not in scope",
                r"duplicate.*?definition",
                r"orphan.*?instance",
                r"overlapping.*?instance",
            ],
            "runtime_error": [
                r"Exception",
                r"\*\*\* Exception",
                r"divide by zero",
                r"arithmetic.*?overflow",
                r"arithmetic.*?underflow",
                r"infinite.*?loop",
                r"stack.*?overflow",
            ],
            "import_error": [
                r"Could not find module",
                r"Module.*?not found",
                r"Ambiguous module name",
                r"hidden.*?module",
                r"Could not load module",
                r"circular.*?import",
                r"Package.*?not found",
            ],
            "instance_error": [
                r"No instance for",
                r"Overlapping instances",
                r"Orphan instance",
                r"Conflicting.*?instance",
                r"Missing.*?instance",
                r"Duplicate.*?instance",
                r"Incoherent.*?instance",
            ],
            "pattern_error": [
                r"Non-exhaustive patterns",
                r"Pattern.*?not matched",
                r"Couldn't match.*?pattern",
                r"Irrefutable.*?pattern",
                r"Pattern.*?binding",
            ],
            "lazy_error": [
                r"<<loop>>",
                r"infinite.*?loop",
                r"laziness.*?error",
                r"strictness.*?error",
                r"space.*?leak",
                r"memory.*?leak",
            ],
            "io_error": [
                r"IO.*?exception",
                r"file.*?not.*?found",
                r"permission.*?denied",
                r"resource.*?busy",
                r"end.*?of.*?file",
                r"invalid.*?argument",
            ],
            "cabal_error": [
                r"cabal.*?error",
                r"Setup.*?error",
                r"build.*?failed",
                r"dependency.*?conflict",
                r"package.*?not.*?found",
                r"configure.*?failed",
            ],
        }

        # Haskell-specific concepts and their common issues
        self.haskell_concepts = {
            "monad": ["monad", "do", "bind", "return", "lift"],
            "functor": ["fmap", "functor", "applicative", "pure"],
            "lazy": ["lazy", "strict", "seq", "force", "deepseq"],
            "type_class": ["instance", "class", "constraint", "typeclass"],
            "higher_order": ["higher order", "function", "lambda", "closure"],
            "list": ["list", "[]", "head", "tail", "cons"],
            "maybe": ["maybe", "just", "nothing", "optional"],
            "either": ["either", "left", "right"],
            "io": ["io", "monad", "print", "read", "file"],
            "recursive": ["recursive", "recursion", "fix", "loop"],
        }

        # Load rules from different categories
        self.rules = self._load_rules()

        # Pre-compile regex patterns for better performance
        self._compile_patterns()

    def _load_rules(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load Haskell error rules from rule files."""
        rules = {}
        rules_dir = Path(__file__).parent.parent / "rules" / "haskell"

        try:
            # Load common Haskell rules
            common_rules_path = rules_dir / "haskell_common_errors.json"
            if common_rules_path.exists():
                with open(common_rules_path, "r") as f:
                    common_data = json.load(f)
                    rules["common"] = common_data.get("rules", [])
                    logger.info(f"Loaded {len(rules['common'])} common Haskell rules")

            # Load concept-specific rules
            for concept in ["types", "monads", "patterns", "lazy"]:
                concept_rules_path = rules_dir / f"haskell_{concept}_errors.json"
                if concept_rules_path.exists():
                    with open(concept_rules_path, "r") as f:
                        concept_data = json.load(f)
                        rules[concept] = concept_data.get("rules", [])
                        logger.info(f"Loaded {len(rules[concept])} {concept} rules")

        except Exception as e:
            logger.error(f"Error loading Haskell rules: {e}")
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
        Analyze a Haskell exception and determine its type and potential fixes.

        Args:
            error_data: Haskell error data in standard format

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
        concept_analysis = self._analyze_haskell_concepts(message)
        if concept_analysis.get("confidence", "low") != "low":
            # Merge concept-specific findings
            analysis.update(concept_analysis)

        # Find matching rules
        matches = self._find_matching_rules(message, error_data)

        # Only update from rules if we don't have a high-confidence concept match
        if matches and not (concept_analysis.get("confidence", "low") == "high"):
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
                "FileNotFoundError": "file",
                "PatternError": "pattern",
            }

            rule_type = best_match.get("type", "unknown")
            subcategory = type_mapping.get(rule_type, rule_type.lower())

            # Override subcategory based on root_cause for specific error types
            root_cause = best_match.get("root_cause", "")
            if "monad_error" in root_cause:
                subcategory = "monad"
            elif "typeclass_error" in root_cause:
                subcategory = "typeclass"
            elif "lazy_error" in root_cause:
                subcategory = "lazy"

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

        # Special handling for lazy evaluation indicators (overrides rules)
        if "undefined" in message.lower() and (
            "exception" in message.lower() or "bottom" in message.lower()
        ):
            analysis["subcategory"] = "laziness"
            analysis["tags"] = analysis.get("tags", []) + ["laziness", "bottom"]
            analysis["root_cause"] = "haskell_lazy_error"

        analysis["file_path"] = file_path
        analysis["line_number"] = line_number
        analysis["column_number"] = column_number
        return analysis

    def _analyze_by_patterns(self, message: str, file_path: str) -> Dict[str, Any]:
        """Analyze error by matching against common patterns."""

        # Check syntax errors
        for pattern in self.haskell_error_patterns["syntax_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "haskell",
                    "subcategory": "syntax",
                    "confidence": "high",
                    "suggested_fix": "Fix Haskell syntax errors",
                    "root_cause": "haskell_syntax_error",
                    "severity": "high",
                    "tags": ["haskell", "syntax", "parser"],
                }

        # Check type errors
        for pattern in self.haskell_error_patterns["type_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "haskell",
                    "subcategory": "type",
                    "confidence": "high",
                    "suggested_fix": "Fix type system errors and type mismatches",
                    "root_cause": "haskell_type_error",
                    "severity": "high",
                    "tags": ["haskell", "type", "inference"],
                }

        # Check compilation errors
        for pattern in self.haskell_error_patterns["compilation_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "haskell",
                    "subcategory": "compilation",
                    "confidence": "high",
                    "suggested_fix": "Fix compilation and build errors",
                    "root_cause": "haskell_compilation_error",
                    "severity": "high",
                    "tags": ["haskell", "compilation", "build"],
                }

        # Check runtime errors
        for pattern in self.haskell_error_patterns["runtime_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "haskell",
                    "subcategory": "runtime",
                    "confidence": "high",
                    "suggested_fix": "Fix runtime errors and exceptions",
                    "root_cause": "haskell_runtime_error",
                    "severity": "high",
                    "tags": ["haskell", "runtime", "exception"],
                }

        # Check import errors
        for pattern in self.haskell_error_patterns["import_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "haskell",
                    "subcategory": "import",
                    "confidence": "high",
                    "suggested_fix": "Fix import and module errors",
                    "root_cause": "haskell_import_error",
                    "severity": "high",
                    "tags": ["haskell", "import", "module"],
                }

        # Check instance errors
        for pattern in self.haskell_error_patterns["instance_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "haskell",
                    "subcategory": "instance",
                    "confidence": "high",
                    "suggested_fix": "Fix type class instance errors",
                    "root_cause": "haskell_instance_error",
                    "severity": "medium",
                    "tags": ["haskell", "instance", "typeclass"],
                }

        # Check pattern errors
        for pattern in self.haskell_error_patterns["pattern_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "haskell",
                    "subcategory": "pattern",
                    "confidence": "high",
                    "suggested_fix": "Fix pattern matching errors",
                    "root_cause": "haskell_pattern_error",
                    "severity": "high",
                    "tags": ["haskell", "pattern", "match"],
                }

        # Check lazy evaluation errors
        for pattern in self.haskell_error_patterns["lazy_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "haskell",
                    "subcategory": "lazy",
                    "confidence": "high",
                    "suggested_fix": "Fix lazy evaluation and strictness issues",
                    "root_cause": "haskell_lazy_error",
                    "severity": "medium",
                    "tags": ["haskell", "lazy", "strictness"],
                }

        return {
            "category": "haskell",
            "subcategory": "unknown",
            "confidence": "low",
            "suggested_fix": "Review Haskell code and compiler error details",
            "root_cause": "haskell_generic_error",
            "severity": "medium",
            "tags": ["haskell", "generic"],
        }

    def _analyze_haskell_concepts(self, message: str) -> Dict[str, Any]:
        """Analyze Haskell-specific concept errors."""
        message_lower = message.lower()

        # Check for monad-related errors
        if any(keyword in message_lower for keyword in self.haskell_concepts["monad"]):
            return {
                "category": "haskell",
                "subcategory": "monad",
                "confidence": "high",
                "suggested_fix": "Handle monadic operations and do-notation properly",
                "root_cause": "haskell_monad_error",
                "severity": "medium",
                "tags": ["haskell", "monad", "do"],
            }

        # Check for type class errors
        if any(
            keyword in message_lower for keyword in self.haskell_concepts["type_class"]
        ):
            return {
                "category": "haskell",
                "subcategory": "typeclass",
                "confidence": "high",
                "suggested_fix": "Fix type class instances and constraints",
                "root_cause": "haskell_type_class_error",
                "severity": "medium",
                "tags": ["haskell", "typeclass", "instance"],
            }

        # Check for lazy evaluation errors
        if any(keyword in message_lower for keyword in self.haskell_concepts["lazy"]):
            return {
                "category": "haskell",
                "subcategory": "lazy",
                "confidence": "medium",
                "suggested_fix": "Handle lazy evaluation and strictness issues",
                "root_cause": "haskell_lazy_error",
                "severity": "medium",
                "tags": ["haskell", "lazy", "strictness"],
            }

        # Check for Maybe-related errors
        if any(keyword in message_lower for keyword in self.haskell_concepts["maybe"]):
            return {
                "category": "haskell",
                "subcategory": "maybe",
                "confidence": "medium",
                "suggested_fix": "Handle Maybe types and null values properly",
                "root_cause": "haskell_maybe_error",
                "severity": "medium",
                "tags": ["haskell", "maybe", "optional"],
            }

        # Check for Either-related errors
        if any(keyword in message_lower for keyword in self.haskell_concepts["either"]):
            return {
                "category": "haskell",
                "subcategory": "either",
                "confidence": "medium",
                "suggested_fix": "Handle Either types and error handling properly",
                "root_cause": "haskell_either_error",
                "severity": "medium",
                "tags": ["haskell", "either", "error"],
            }

        # Check for list-related errors
        if any(keyword in message_lower for keyword in self.haskell_concepts["list"]):
            return {
                "category": "haskell",
                "subcategory": "list",
                "confidence": "medium",
                "suggested_fix": "Handle list operations and pattern matching",
                "root_cause": "haskell_list_error",
                "severity": "medium",
                "tags": ["haskell", "list", "pattern"],
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
        if file_path.endswith(".hs") or file_path.endswith(".lhs"):
            base_confidence += 0.2

        # Boost confidence based on rule reliability
        reliability = rule.get("reliability", "medium")
        reliability_boost = {"high": 0.2, "medium": 0.1, "low": 0.0}
        base_confidence += reliability_boost.get(reliability, 0.0)

        # Boost confidence for concept matches
        rule_tags = set(rule.get("tags", []))
        context_tags = set()

        message = error_data.get("message", "").lower()
        if "monad" in message:
            context_tags.add("monad")
        if "instance" in message:
            context_tags.add("instance")
        if "pattern" in message:
            context_tags.add("pattern")

        if context_tags & rule_tags:
            base_confidence += 0.1

        return min(base_confidence, 1.0)


class HaskellPatchGenerator:
    """
    Generates patches for Haskell errors based on analysis results.

    This class creates Haskell code fixes for common errors using templates
    and heuristics specific to functional programming patterns.
    """

    def __init__(self):
        """Initialize the Haskell patch generator."""
        self.template_dir = (
            Path(__file__).parent.parent / "patch_generation" / "templates"
        )
        self.haskell_template_dir = self.template_dir / "haskell"

        # Ensure template directory exists
        self.haskell_template_dir.mkdir(parents=True, exist_ok=True)

        # Load patch templates
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict[str, str]:
        """Load Haskell patch templates."""
        templates = {}

        if not self.haskell_template_dir.exists():
            logger.warning(
                f"Haskell templates directory not found: {self.haskell_template_dir}"
            )
            return templates

        for template_file in self.haskell_template_dir.glob("*.hs.template"):
            try:
                with open(template_file, "r") as f:
                    template_name = template_file.stem.replace(".hs", "")
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
        Generate a patch for the Haskell error.

        Args:
            error_data: The Haskell error data
            analysis: Analysis results from HaskellExceptionHandler
            source_code: The Haskell source code that caused the error

        Returns:
            Patch information or None if no patch can be generated
        """
        root_cause = analysis.get("root_cause", "")

        # Map root causes to patch strategies
        patch_strategies = {
            "haskell_syntax_error": self._fix_syntax_error,
            "haskell_type_error": self._fix_type_error,
            "haskell_compilation_error": self._fix_compilation_error,
            "haskell_runtime_error": self._fix_runtime_error,
            "haskell_import_error": self._fix_import_error,
            "haskell_instance_error": self._fix_instance_error,
            "haskell_pattern_error": self._fix_pattern_error,
            "haskell_lazy_error": self._fix_lazy_error,
            "haskell_monad_error": self._fix_monad_error,
            "haskell_type_class_error": self._fix_type_class_error,
            "haskell_maybe_error": self._fix_maybe_error,
            "haskell_either_error": self._fix_either_error,
            "haskell_list_error": self._fix_list_error,
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
        """Fix Haskell syntax errors."""
        message = error_data.get("message", "")

        if "parse error" in message.lower():
            return {
                "type": "suggestion",
                "description": "Haskell syntax parse error",
                "fixes": [
                    "Check for missing parentheses or brackets",
                    "Verify proper indentation in do-blocks and let/where clauses",
                    "Ensure proper operator precedence",
                    "Check for missing function arguments or type signatures",
                ],
            }

        if "indent" in message.lower():
            return {
                "type": "suggestion",
                "description": "Indentation error",
                "fixes": [
                    "Use consistent indentation (spaces or tabs, not both)",
                    "Align code properly in do-blocks",
                    "Check let/where clause indentation",
                    "Ensure proper alignment of guards and patterns",
                ],
            }

        return {
            "type": "suggestion",
            "description": "Syntax error. Check Haskell syntax and structure",
        }

    def _fix_type_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix Haskell type errors."""
        message = error_data.get("message", "")

        if "couldn't match type" in message.lower():
            return {
                "type": "suggestion",
                "description": "Type mismatch error",
                "fixes": [
                    "Check function signatures and return types",
                    "Add explicit type annotations",
                    "Use proper type conversion functions",
                    "Verify function composition and argument types",
                ],
            }

        if "no instance for" in message.lower():
            return {
                "type": "suggestion",
                "description": "Missing type class instance",
                "fixes": [
                    "Derive the required type class instance",
                    "Import the module containing the instance",
                    "Add explicit instance declaration",
                    "Use a different type that has the required instance",
                ],
            }

        if "ambiguous type" in message.lower():
            return {
                "type": "suggestion",
                "description": "Ambiguous type error",
                "fixes": [
                    "Add explicit type signatures",
                    "Use type applications with TypeApplications",
                    "Provide more context for type inference",
                    "Use qualified imports to avoid ambiguity",
                ],
            }

        return {
            "type": "suggestion",
            "description": "Type system error. Check types and signatures",
        }

    def _fix_compilation_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix compilation errors."""
        message = error_data.get("message", "")

        if "not in scope" in message.lower():
            return {
                "type": "suggestion",
                "description": "Identifier not in scope",
                "fixes": [
                    "Import the required module",
                    "Check spelling of identifiers",
                    "Ensure functions are defined before use",
                    "Add qualified imports if needed",
                ],
            }

        if "module not found" in message.lower():
            return {
                "type": "suggestion",
                "description": "Module not found",
                "fixes": [
                    "Check module name spelling",
                    "Ensure module is in the package dependencies",
                    "Add module to cabal file or stack.yaml",
                    "Check import paths and module hierarchy",
                ],
            }

        return {
            "type": "suggestion",
            "description": "Compilation error. Check imports and dependencies",
        }

    def _fix_runtime_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix runtime errors."""
        message = error_data.get("message", "")

        if "non-exhaustive patterns" in message.lower():
            return {
                "type": "suggestion",
                "description": "Non-exhaustive pattern matching",
                "fixes": [
                    "Add missing pattern cases",
                    "Use wildcard pattern (_) for default case",
                    "Handle all constructors in pattern matching",
                    "Use Maybe or Either for partial functions",
                ],
            }

        if "undefined" in message.lower():
            return {
                "type": "suggestion",
                "description": "Undefined value error",
                "fixes": [
                    "Replace undefined with actual implementation",
                    "Use error with descriptive message",
                    "Handle partial functions properly",
                    "Add proper error handling",
                ],
            }

        return {
            "type": "suggestion",
            "description": "Runtime error. Check program logic and error handling",
        }

    def _fix_import_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix import errors."""
        return {
            "type": "suggestion",
            "description": "Import/module error",
            "fixes": [
                "Check module names and paths",
                "Add missing dependencies to cabal file",
                "Use qualified imports to avoid conflicts",
                "Verify module is exposed in package",
            ],
        }

    def _fix_instance_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix type class instance errors."""
        return {
            "type": "suggestion",
            "description": "Type class instance error",
            "fixes": [
                "Add deriving clause to data type",
                "Import module with instance definitions",
                "Create explicit instance declaration",
                "Use newtype to avoid overlapping instances",
            ],
        }

    def _fix_pattern_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix pattern matching errors."""
        return {
            "type": "suggestion",
            "description": "Pattern matching error",
            "fixes": [
                "Add all missing pattern cases",
                "Use wildcard pattern for default case",
                "Handle all data constructors",
                "Use guards for conditional patterns",
            ],
        }

    def _fix_lazy_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix lazy evaluation errors."""
        return {
            "type": "suggestion",
            "description": "Laziness evaluation error",
            "fixes": [
                "Use strict evaluation with seq or deepseq",
                "Add bang patterns for strictness",
                "Use strict data structures",
                "Profile for space leaks and fix accumulator parameters",
            ],
        }

    def _fix_monad_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix monad-related errors."""
        return {
            "type": "suggestion",
            "description": "Monad error",
            "fixes": [
                "Check do-notation syntax and indentation",
                "Use proper monadic bind (>>=) and return",
                "Handle monadic composition correctly",
                "Use lift for monad transformers",
            ],
        }

    def _fix_type_class_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix type class errors."""
        return {
            "type": "suggestion",
            "description": "Type class instance error",
            "fixes": [
                "Add required type class constraints",
                "Derive or implement missing instances",
                "Use explicit type signatures",
                "Import modules with required instances",
            ],
        }

    def _fix_maybe_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix Maybe type errors."""
        return {
            "type": "suggestion",
            "description": "Maybe type error",
            "fixes": [
                "Handle Nothing case in pattern matching",
                "Use maybe function for safe operations",
                "Use fromMaybe for default values",
                "Chain Maybe operations with bind (>>=)",
            ],
        }

    def _fix_either_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix Either type errors."""
        return {
            "type": "suggestion",
            "description": "Either type error",
            "fixes": [
                "Handle both Left and Right cases",
                "Use either function for pattern matching",
                "Chain Either operations with bind",
                "Use appropriate error types for Left values",
            ],
        }

    def _fix_list_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix list-related errors."""
        return {
            "type": "suggestion",
            "description": "List error",
            "fixes": [
                "Handle empty list case in pattern matching",
                "Use safe functions like headMay, tailMay",
                "Check for infinite lists in lazy evaluation",
                "Use proper list construction and deconstruction",
            ],
        }

    def _template_based_patch(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Generate patch using templates."""
        root_cause = analysis.get("root_cause", "")

        # Map root causes to template names
        template_map = {
            "haskell_syntax_error": "syntax_fix",
            "haskell_type_error": "type_fix",
            "haskell_pattern_error": "pattern_fix",
            "haskell_monad_error": "monad_fix",
            "haskell_maybe_error": "maybe_fix",
            "haskell_either_error": "either_fix",
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


class HaskellLanguagePlugin(LanguagePlugin):
    """
    Main Haskell language plugin for Homeostasis.

    This plugin orchestrates Haskell error analysis and patch generation,
    supporting functional programming patterns and lazy evaluation.
    """

    VERSION = "1.0.0"
    AUTHOR = "Homeostasis Team"

    def __init__(self):
        """Initialize the Haskell language plugin."""
        self.language = "haskell"
        self.supported_extensions = {".hs", ".lhs", ".cabal"}
        self.supported_frameworks = [
            "haskell",
            "ghc",
            "cabal",
            "stack",
            "haskell-platform",
            "yesod",
            "snap",
            "servant",
            "hakyll",
            "xmonad",
        ]

        # Initialize components
        self.exception_handler = HaskellExceptionHandler()
        self.patch_generator = HaskellPatchGenerator()

        logger.info("Haskell language plugin initialized")

    def get_language_id(self) -> str:
        """Get the unique identifier for this language."""
        return "haskell"

    def get_language_name(self) -> str:
        """Get the human-readable name of the language."""
        return "Haskell"

    def get_language_version(self) -> str:
        """Get the version of the language supported by this plugin."""
        return "GHC 9.0+"

    def get_supported_frameworks(self) -> List[str]:
        """Get the list of frameworks supported by this language plugin."""
        return self.supported_frameworks

    def normalize_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize error data to the standard Homeostasis format.

        Args:
            error_data: Error data in the Haskell-specific format

        Returns:
            Error data in the standard format
        """
        # Map Haskell-specific error fields to standard format
        normalized = {
            "error_type": error_data.get("error_type", "HaskellError"),
            "message": error_data.get("message", error_data.get("description", "")),
            "language": "haskell",
            "file_path": error_data.get("file_path", error_data.get("file", "")),
            "line_number": error_data.get("line_number", error_data.get("line", 0)),
            "column_number": error_data.get(
                "column_number", error_data.get("column", 0)
            ),
            "compiler_version": error_data.get("compiler_version", ""),
            "ghc_version": error_data.get("ghc_version", ""),
            "cabal_version": error_data.get("cabal_version", ""),
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
        Convert standard format error data back to the Haskell-specific format.

        Args:
            standard_error: Error data in the standard format

        Returns:
            Error data in the Haskell-specific format
        """
        # Map standard fields back to Haskell-specific format
        haskell_error = {
            "error_type": standard_error.get("error_type", "HaskellError"),
            "message": standard_error.get("message", ""),
            "file_path": standard_error.get("file_path", ""),
            "line_number": standard_error.get("line_number", 0),
            "column_number": standard_error.get("column_number", 0),
            "compiler_version": standard_error.get("compiler_version", ""),
            "ghc_version": standard_error.get("ghc_version", ""),
            "cabal_version": standard_error.get("cabal_version", ""),
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
            if key not in haskell_error and value is not None:
                haskell_error[key] = value

        return haskell_error

    def analyze_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a Haskell error.

        Args:
            error_data: Haskell error data

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
            analysis["plugin"] = "haskell"
            analysis["language"] = "haskell"
            analysis["plugin_version"] = self.VERSION

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing Haskell error: {e}")
            return {
                "category": "haskell",
                "subcategory": "unknown",
                "confidence": "low",
                "suggested_fix": "Unable to analyze Haskell error",
                "error": str(e),
                "plugin": "haskell",
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
register_plugin(HaskellLanguagePlugin())
