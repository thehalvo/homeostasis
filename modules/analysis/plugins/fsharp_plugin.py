"""
F# Language Plugin for Homeostasis

This plugin enables Homeostasis to analyze and fix errors in F# programming language code.
It provides comprehensive error handling for F# compilation errors, functional programming patterns,
.NET integration issues, and type system errors.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..language_plugin_system import LanguagePlugin, register_plugin

logger = logging.getLogger(__name__)


class FSharpExceptionHandler:
    """
    Handles F# exceptions with robust error detection and classification.

    This class provides logic for categorizing F# errors based on their type,
    message, and common functional .NET programming patterns.
    """

    def __init__(self):
        """Initialize the F# exception handler."""
        self.rule_categories = {
            "syntax": "F# syntax and parsing errors",
            "type": "Type system and type checking errors",
            "compilation": "Compilation and build errors",
            "runtime": "Runtime errors and exceptions",
            "dotnet": ".NET interop and framework errors",
            "pattern": "Pattern matching errors",
            "computation": "Computation expression errors",
            "async": "Async workflow errors",
            "option": "Option type errors",
            "result": "Result type errors",
            "sequence": "Sequence and collection errors",
            "record": "Record type errors",
            "discriminated": "Discriminated union errors",
            "unit": "Unit of measure errors",
        }

        # Common F# error patterns
        self.fsharp_error_patterns = {
            "syntax_error": [
                r"Syntax error",
                r"Unexpected.*?expecting",
                r"Incomplete.*?expression",
                r"Invalid.*?syntax",
                r"Expected.*?but found",
                r"Unmatched.*?delimiter",
                r"Unexpected.*?end.*?of.*?input",
                r"Block.*?following.*?this.*?let.*?is.*?unfinished",
            ],
            "type_error": [
                r"Type mismatch",
                r"The type.*?does not match",
                r"This expression was expected to have type",
                r"but here has type",
                r"The type.*?is not defined",
                r"The value.*?is not defined",
                r"Lookup on object of indeterminate type",
                r"Type constraint mismatch",
                r"The type.*?cannot be instantiated",
            ],
            "compilation_error": [
                r"The.*?is not defined",
                r"The namespace.*?is not defined",
                r"The type.*?is not defined",
                r"Assembly.*?not found",
                r"Module.*?not found",
                r"Namespace.*?not found",
                r"Reference.*?not found",
                r"The field.*?is not defined",
            ],
            "runtime_error": [
                r"System\..*?Exception",
                r"NullReferenceException",
                r"ArgumentException",
                r"InvalidOperationException",
                r"IndexOutOfRangeException",
                r"KeyNotFoundException",
                r"DivideByZeroException",
                r"OverflowException",
                r"StackOverflowException",
            ],
            "dotnet_error": [
                r"Assembly.*?could not be loaded",
                r"\.NET.*?error",
                r"CLR.*?error",
                r"Framework.*?error",
                r"System\..*?not found",
                r"Interop.*?error",
                r"P/Invoke.*?error",
                r"COM.*?error",
            ],
            "pattern_error": [
                r"Incomplete pattern matches",
                r"This pattern match.*?incomplete",
                r"Pattern.*?not matched",
                r"Unreachable.*?pattern",
                r"Pattern.*?binding.*?error",
                r"Active pattern.*?error",
            ],
            "computation_error": [
                r"Computation expression.*?error",
                r"Builder.*?not found",
                r"Invalid.*?computation.*?expression",
                r"Workflow.*?error",
                r"Builder.*?method.*?not found",
            ],
            "async_error": [
                r"Async.*?workflow.*?error",
                r"Async.*?operation.*?error",
                r"Task.*?error",
                r"Async.*?binding.*?error",
                r"Async.*?exception",
            ],
            "option_error": [
                r"Option.*?type.*?error",
                r"None.*?value.*?error",
                r"Some.*?value.*?error",
                r"Option.*?binding.*?error",
            ],
            "result_error": [
                r"Result.*?type.*?error",
                r"Error.*?value.*?error",
                r"Ok.*?value.*?error",
                r"Result.*?binding.*?error",
            ],
            "sequence_error": [
                r"Sequence.*?error",
                r"Collection.*?error",
                r"List.*?error",
                r"Array.*?error",
                r"Set.*?error",
                r"Map.*?error",
            ],
            "record_error": [
                r"Record.*?field.*?error",
                r"Record.*?type.*?error",
                r"Field.*?not.*?found",
                r"Record.*?initialization.*?error",
            ],
            "discriminated_error": [
                r"Discriminated.*?union.*?error",
                r"Union.*?case.*?error",
                r"Case.*?not.*?found",
                r"Union.*?type.*?error",
            ],
            "unit_error": [
                r"Unit.*?of.*?measure.*?error",
                r"Measure.*?mismatch",
                r"Unit.*?mismatch",
                r"Dimension.*?error",
            ],
        }

        # F#-specific concepts and their common issues
        self.fsharp_concepts = {
            "option": ["option", "some", "none", "optional"],
            "result": ["result", "ok", "error", "either"],
            "async": ["async", "await", "task", "workflow"],
            "computation": ["computation", "expression", "builder", "workflow"],
            "pattern": ["pattern", "match", "with", "when"],
            "pipe": ["pipe", "|>", ">>", "compose"],
            "discriminated": ["discriminated", "union", "case", "type"],
            "record": ["record", "field", "with", "mutable"],
            "sequence": ["seq", "list", "array", "collection"],
            "unit": ["unit", "measure", "dimension", "float<_>"],
            "active": ["active", "pattern", "|_|", "partial"],
            "quotation": ["quotation", "expr", "reflect", "code"],
        }

        # Load rules from different categories
        self.rules = self._load_rules()

        # Pre-compile regex patterns for better performance
        self._compile_patterns()

    def _load_rules(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load F# error rules from rule files."""
        rules = {}
        rules_dir = Path(__file__).parent.parent / "rules" / "fsharp"

        try:
            # Load common F# rules
            common_rules_path = rules_dir / "fsharp_common_errors.json"
            if common_rules_path.exists():
                with open(common_rules_path, "r") as f:
                    common_data = json.load(f)
                    rules["common"] = common_data.get("rules", [])
                    logger.info(f"Loaded {len(rules['common'])} common F# rules")

            # Load concept-specific rules
            for concept in ["types", "patterns", "async", "dotnet"]:
                concept_rules_path = rules_dir / f"fsharp_{concept}_errors.json"
                if concept_rules_path.exists():
                    with open(concept_rules_path, "r") as f:
                        concept_data = json.load(f)
                        rules[concept] = concept_data.get("rules", [])
                        logger.info(f"Loaded {len(rules[concept])} {concept} rules")

        except Exception as e:
            logger.error(f"Error loading F# rules: {e}")
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
        Analyze an F# exception and determine its type and potential fixes.

        Args:
            error_data: F# error data in standard format

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
            concept_analysis = self._analyze_fsharp_concepts(message)
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
                "FileNotFoundError": "file",
            }

            rule_type = best_match.get("type", "unknown")
            # First check if we can determine subcategory from root_cause
            root_cause = best_match.get("root_cause", "")
            message = error_data.get("message", "")

            # Check message-specific patterns first for accurate categorization
            if "unexpected" in message.lower() and "token" in message.lower():
                subcategory = "syntax"
            elif "union case" in message.lower():
                subcategory = "union"
            elif "type mismatch" in message.lower() or "expecting a" in message.lower():
                subcategory = "type"
            elif (
                "pattern match" in message.lower()
                or "incomplete pattern" in message.lower()
            ):
                subcategory = "pattern"
            elif "computation expression" in message.lower():
                subcategory = "computation"
            elif "object reference not set" in message.lower():
                subcategory = "null"
            elif "namespace" in message.lower() and "not defined" in message.lower():
                subcategory = "import"
            elif "syntax" in root_cause:
                subcategory = "syntax"
            elif "type" in root_cause:
                subcategory = "type"
            elif "pattern" in root_cause:
                subcategory = "pattern"
            elif "computation" in root_cause:
                subcategory = "computation"
            elif "option" in root_cause:
                subcategory = "option"
            elif "result" in root_cause:
                subcategory = "result"
            elif "discriminated" in root_cause or "union" in root_cause:
                subcategory = "union"
            elif "import" in root_cause:
                subcategory = "import"
            elif "null" in root_cause:
                subcategory = "null"
            else:
                # Fall back to type mapping
                subcategory = type_mapping.get(rule_type, rule_type.lower())

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
        message_lower = message.lower()

        # Check specific messages first
        if "unexpected token" in message_lower:
            return {
                "category": "fsharp",
                "subcategory": "syntax",
                "confidence": "high",
                "suggested_fix": "Fix F# syntax errors",
                "root_cause": "fsharp_syntax_error",
                "severity": "high",
                "tags": ["fsharp", "syntax", "parser"],
            }

        if "object reference not set" in message_lower:
            return {
                "category": "fsharp",
                "subcategory": "null",
                "confidence": "high",
                "suggested_fix": "Fix null reference errors",
                "root_cause": "fsharp_null_error",
                "severity": "high",
                "tags": ["fsharp", "null", "reference"],
            }

        if "namespace" in message_lower and "not defined" in message_lower:
            return {
                "category": "fsharp",
                "subcategory": "import",
                "confidence": "high",
                "suggested_fix": "Fix import and namespace errors",
                "root_cause": "fsharp_import_error",
                "severity": "high",
                "tags": ["fsharp", "import", "namespace"],
            }

        # Check syntax errors
        for pattern in self.fsharp_error_patterns["syntax_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "fsharp",
                    "subcategory": "syntax",
                    "confidence": "high",
                    "suggested_fix": "Fix F# syntax errors",
                    "root_cause": "fsharp_syntax_error",
                    "severity": "high",
                    "tags": ["fsharp", "syntax", "parser"],
                }

        # Check type errors
        for pattern in self.fsharp_error_patterns["type_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "fsharp",
                    "subcategory": "type",
                    "confidence": "high",
                    "suggested_fix": "Fix type system errors and type mismatches",
                    "root_cause": "fsharp_type_error",
                    "severity": "high",
                    "tags": ["fsharp", "type", "inference"],
                }

        # Check compilation errors
        for pattern in self.fsharp_error_patterns["compilation_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "fsharp",
                    "subcategory": "compilation",
                    "confidence": "high",
                    "suggested_fix": "Fix compilation and build errors",
                    "root_cause": "fsharp_compilation_error",
                    "severity": "high",
                    "tags": ["fsharp", "compilation", "build"],
                }

        # Check runtime errors
        for pattern in self.fsharp_error_patterns["runtime_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "fsharp",
                    "subcategory": "runtime",
                    "confidence": "high",
                    "suggested_fix": "Fix runtime errors and exceptions",
                    "root_cause": "fsharp_runtime_error",
                    "severity": "high",
                    "tags": ["fsharp", "runtime", "exception"],
                }

        # Check .NET errors
        for pattern in self.fsharp_error_patterns["dotnet_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "fsharp",
                    "subcategory": "dotnet",
                    "confidence": "high",
                    "suggested_fix": "Fix .NET interop and framework errors",
                    "root_cause": "fsharp_dotnet_error",
                    "severity": "high",
                    "tags": ["fsharp", "dotnet", "interop"],
                }

        # Check pattern errors
        for pattern in self.fsharp_error_patterns["pattern_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "fsharp",
                    "subcategory": "pattern",
                    "confidence": "high",
                    "suggested_fix": "Fix pattern matching errors",
                    "root_cause": "fsharp_pattern_error",
                    "severity": "high",
                    "tags": ["fsharp", "pattern", "match"],
                }

        # Check async errors
        for pattern in self.fsharp_error_patterns["async_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "fsharp",
                    "subcategory": "async",
                    "confidence": "high",
                    "suggested_fix": "Fix async workflow errors",
                    "root_cause": "fsharp_async_error",
                    "severity": "medium",
                    "tags": ["fsharp", "async", "workflow"],
                }

        return {
            "category": "fsharp",
            "subcategory": "unknown",
            "confidence": "low",
            "suggested_fix": "Review F# code and compiler error details",
            "root_cause": "fsharp_generic_error",
            "severity": "medium",
            "tags": ["fsharp", "generic"],
        }

    def _analyze_fsharp_concepts(self, message: str) -> Dict[str, Any]:
        """Analyze F#-specific concept errors."""
        message_lower = message.lower()

        # Check for option-related errors (but avoid generic "some" word in message)
        if "option" in message_lower or (
            "some(" in message_lower
            or "none(" in message_lower
            or "some " in message_lower
            and "value" in message_lower
        ):
            return {
                "category": "fsharp",
                "subcategory": "option",
                "confidence": "high",
                "suggested_fix": "Handle Option types properly with Some/None pattern matching",
                "root_cause": "fsharp_option_error",
                "severity": "medium",
                "tags": ["fsharp", "option", "optional"],
            }

        # Check for result-related errors (but avoid generic "error" word)
        if (
            "result" in message_lower
            or "ok(" in message_lower
            or ("error(" in message_lower and "result" in message_lower)
        ):
            return {
                "category": "fsharp",
                "subcategory": "result",
                "confidence": "high",
                "suggested_fix": "Handle Result types properly with Ok/Error pattern matching",
                "root_cause": "fsharp_result_error",
                "severity": "medium",
                "tags": ["fsharp", "result", "error"],
            }

        # Check for async-related errors
        if any(keyword in message_lower for keyword in self.fsharp_concepts["async"]):
            return {
                "category": "fsharp",
                "subcategory": "async",
                "confidence": "high",
                "suggested_fix": "Handle async workflows and tasks properly",
                "root_cause": "fsharp_async_error",
                "severity": "medium",
                "tags": ["fsharp", "async", "workflow"],
            }

        # Check for computation expression errors
        if any(
            keyword in message_lower for keyword in self.fsharp_concepts["computation"]
        ):
            return {
                "category": "fsharp",
                "subcategory": "computation",
                "confidence": "high",
                "suggested_fix": "Handle computation expressions and builders properly",
                "root_cause": "fsharp_computation_error",
                "severity": "medium",
                "tags": ["fsharp", "computation", "builder"],
            }

        # Check for pattern matching errors
        if any(keyword in message_lower for keyword in self.fsharp_concepts["pattern"]):
            return {
                "category": "fsharp",
                "subcategory": "pattern",
                "confidence": "high",
                "suggested_fix": "Handle pattern matching properly with all cases",
                "root_cause": "fsharp_pattern_error",
                "severity": "high",
                "tags": ["fsharp", "pattern", "match"],
            }

        # Check for discriminated union errors
        if any(
            keyword in message_lower
            for keyword in self.fsharp_concepts["discriminated"]
        ):
            return {
                "category": "fsharp",
                "subcategory": "union",
                "confidence": "high",
                "suggested_fix": "Handle discriminated unions properly with all cases",
                "root_cause": "fsharp_discriminated_error",
                "severity": "medium",
                "tags": ["fsharp", "discriminated", "union"],
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
        if (
            file_path.endswith(".fs")
            or file_path.endswith(".fsx")
            or file_path.endswith(".fsi")
        ):
            base_confidence += 0.2

        # Boost confidence based on rule reliability
        reliability = rule.get("reliability", "medium")
        reliability_boost = {"high": 0.2, "medium": 0.1, "low": 0.0}
        base_confidence += reliability_boost.get(reliability, 0.0)

        # Boost confidence for concept matches
        rule_tags = set(rule.get("tags", []))
        context_tags = set()

        message = error_data.get("message", "").lower()
        if "option" in message:
            context_tags.add("option")
        if "async" in message:
            context_tags.add("async")
        if "pattern" in message:
            context_tags.add("pattern")

        if context_tags & rule_tags:
            base_confidence += 0.1

        return min(base_confidence, 1.0)


class FSharpPatchGenerator:
    """
    Generates patches for F# errors based on analysis results.

    This class creates F# code fixes for common errors using templates
    and heuristics specific to functional .NET programming patterns.
    """

    def __init__(self):
        """Initialize the F# patch generator."""
        self.template_dir = (
            Path(__file__).parent.parent / "patch_generation" / "templates"
        )
        self.fsharp_template_dir = self.template_dir / "fsharp"

        # Ensure template directory exists
        self.fsharp_template_dir.mkdir(parents=True, exist_ok=True)

        # Load patch templates
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict[str, str]:
        """Load F# patch templates."""
        templates = {}

        if not self.fsharp_template_dir.exists():
            logger.warning(
                f"F# templates directory not found: {self.fsharp_template_dir}"
            )
            return templates

        for template_file in self.fsharp_template_dir.glob("*.fs.template"):
            try:
                with open(template_file, "r") as f:
                    template_name = template_file.stem.replace(".fs", "")
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
        Generate a patch for the F# error.

        Args:
            error_data: The F# error data
            analysis: Analysis results from FSharpExceptionHandler
            source_code: The F# source code that caused the error

        Returns:
            Patch information or None if no patch can be generated
        """
        root_cause = analysis.get("root_cause", "")

        # Map root causes to patch strategies
        patch_strategies = {
            "fsharp_syntax_error": self._fix_syntax_error,
            "fsharp_type_error": self._fix_type_error,
            "fsharp_compilation_error": self._fix_compilation_error,
            "fsharp_runtime_error": self._fix_runtime_error,
            "fsharp_dotnet_error": self._fix_dotnet_error,
            "fsharp_pattern_error": self._fix_pattern_error,
            "fsharp_async_error": self._fix_async_error,
            "fsharp_option_error": self._fix_option_error,
            "fsharp_result_error": self._fix_result_error,
            "fsharp_computation_error": self._fix_computation_error,
            "fsharp_discriminated_error": self._fix_discriminated_error,
            "fsharp_union_error": self._fix_union_error,
            "fsharp_import_error": self._fix_import_error,
            "fsharp_null_error": self._fix_null_error,
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
        """Fix F# syntax errors."""
        message = error_data.get("message", "")

        if "syntax error" in message.lower():
            return {
                "type": "suggestion",
                "description": "F# syntax error",
                "fixes": [
                    "Check for missing parentheses or brackets",
                    "Verify proper indentation and whitespace",
                    "Ensure proper operator precedence",
                    "Check for missing semicolons in sequences",
                ],
            }

        if "unexpected" in message.lower() and "expecting" in message.lower():
            return {
                "type": "suggestion",
                "description": "Unexpected token in F# code",
                "fixes": [
                    "Check for missing keywords or operators",
                    "Verify proper function and value definitions",
                    "Ensure correct use of let, match, and other constructs",
                    "Check for proper pipe operator usage",
                ],
            }

        return {
            "type": "suggestion",
            "description": "Syntax error. Check F# syntax and structure",
        }

    def _fix_type_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix F# type errors."""
        message = error_data.get("message", "")

        if "type mismatch" in message.lower():
            return {
                "type": "suggestion",
                "description": "Type mismatch error",
                "fixes": [
                    "Add explicit type annotations",
                    "Use appropriate type conversion functions",
                    "Check function signatures and return types",
                    "Verify generic type parameters",
                ],
            }

        if "was expected to have type" in message.lower():
            return {
                "type": "suggestion",
                "description": "Expected type mismatch",
                "fixes": [
                    "Cast or convert to expected type",
                    "Use type inference helpers",
                    "Check pipeline operator type flow",
                    "Verify function composition types",
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

        if "is not defined" in message.lower():
            return {
                "type": "suggestion",
                "description": "Undefined identifier",
                "fixes": [
                    "Open required namespaces",
                    "Add module or assembly references",
                    "Check identifier spelling",
                    "Ensure proper module order",
                ],
            }

        if "not found" in message.lower():
            return {
                "type": "suggestion",
                "description": "Reference not found",
                "fixes": [
                    "Add NuGet package references",
                    "Include required assemblies",
                    "Check project references",
                    "Verify namespace imports",
                ],
            }

        return {
            "type": "suggestion",
            "description": "Compilation error. Check references and imports",
        }

    def _fix_runtime_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix runtime errors."""
        message = error_data.get("message", "")

        if "nullreferenceexception" in message.lower():
            return {
                "type": "suggestion",
                "description": "Null reference exception",
                "fixes": [
                    "Use Option types instead of null",
                    "Add null checks before access",
                    "Use Option.bind for safe operations",
                    "Initialize values properly",
                ],
            }

        if "indexoutofrangeexception" in message.lower():
            return {
                "type": "suggestion",
                "description": "Index out of range",
                "fixes": [
                    "Check array/list bounds before access",
                    "Use Array.tryGet or List.tryItem",
                    "Validate indices before use",
                    "Use seq functions for safe access",
                ],
            }

        return {
            "type": "suggestion",
            "description": "Runtime error. Check program logic and error handling",
        }

    def _fix_dotnet_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix .NET interop errors."""
        return {
            "type": "suggestion",
            "description": ".NET interop error",
            "fixes": [
                "Check assembly references and versions",
                "Verify P/Invoke signatures",
                "Use proper .NET type conversions",
                "Handle COM interop properly",
            ],
        }

    def _fix_pattern_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix pattern matching errors."""
        message = error_data.get("message", "")

        if "incomplete pattern" in message.lower():
            return {
                "type": "suggestion",
                "description": "Incomplete pattern matching",
                "fixes": [
                    "Add missing pattern cases",
                    "Use wildcard pattern (_) for default",
                    "Handle all discriminated union cases",
                    "Use when guards for complex patterns",
                ],
            }

        return {
            "type": "suggestion",
            "description": "Pattern matching error",
            "fixes": [
                "Check all pattern cases are handled",
                "Use proper pattern syntax",
                "Handle nested patterns correctly",
                "Use active patterns where appropriate",
            ],
        }

    def _fix_async_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix async workflow errors."""
        return {
            "type": "suggestion",
            "description": "Async workflow error",
            "fixes": [
                "Use async { } computation expression",
                "Handle async binding with let!",
                "Use Async.RunSynchronously for sync calls",
                "Handle async exceptions properly",
            ],
        }

    def _fix_option_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix Option type errors."""
        return {
            "type": "suggestion",
            "description": "Option type error",
            "fixes": [
                "Use pattern matching with Some/None",
                "Use Option.bind for chaining",
                "Use Option.defaultValue for defaults",
                "Handle None cases explicitly",
            ],
        }

    def _fix_result_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix Result type errors."""
        return {
            "type": "suggestion",
            "description": "Result type error",
            "fixes": [
                "Use pattern matching with Ok/Error",
                "Use Result.bind for chaining",
                "Handle error cases explicitly",
                "Use Result.mapError for error transformation",
            ],
        }

    def _fix_computation_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix computation expression errors."""
        return {
            "type": "suggestion",
            "description": "Computation expression error",
            "fixes": [
                "Check builder methods are implemented",
                "Use proper computation expression syntax",
                "Handle binding and return correctly",
                "Use appropriate computation builders",
            ],
        }

    def _fix_discriminated_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix discriminated union errors."""
        return {
            "type": "suggestion",
            "description": "Discriminated union error",
            "fixes": [
                "Handle all union cases in pattern matching",
                "Use proper union case syntax",
                "Check case constructor parameters",
                "Use when guards for complex cases",
            ],
        }

    def _fix_union_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix union errors."""
        return {
            "type": "suggestion",
            "description": "Union type error",
            "fixes": [
                "Check union case constructor arguments",
                "Handle all union cases in pattern matching",
                "Use proper discriminated union syntax",
                "Verify case names match definition",
            ],
        }

    def _fix_import_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix import errors."""
        return {
            "type": "suggestion",
            "description": "Import or namespace error",
            "fixes": [
                "Check module or namespace is defined",
                "Use 'open' to import namespaces",
                "Check assembly references in project",
                "Verify module names and paths",
            ],
        }

    def _fix_null_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix null reference errors."""
        return {
            "type": "suggestion",
            "description": "Null reference error",
            "fixes": [
                "Use Option types instead of nulls",
                "Check for null before accessing members",
                "Use pattern matching with null checks",
                "Consider using F# non-null types",
            ],
        }

    def _template_based_patch(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Generate patch using templates."""
        root_cause = analysis.get("root_cause", "")
        subcategory = analysis.get("subcategory", "")

        # Map root causes to template names
        template_map = {
            "fsharp_syntax_error": "syntax_fix",
            "fsharp_type_error": "type_fix",
            "fsharp_pattern_error": "pattern_fix",
            "fsharp_option_error": "option_fix",
            "fsharp_result_error": "result_fix",
            "fsharp_async_error": "async_fix",
        }

        template_name = template_map.get(root_cause)
        if template_name and template_name in self.templates:
            template = self.templates[template_name]

            return {
                "type": "template",
                "template": template,
                "description": f"Applied template fix for {root_cause}",
            }

        # Return a default suggestion if no template is found
        return {
            "type": "suggestion",
            "description": f"Fix {subcategory} error in F# code",
            "fixes": [
                f"Review the {subcategory} error details",
                "Check F# documentation for proper syntax",
                "Ensure code follows F# functional patterns",
            ],
        }


class FSharpLanguagePlugin(LanguagePlugin):
    """
    Main F# language plugin for Homeostasis.

    This plugin orchestrates F# error analysis and patch generation,
    supporting functional .NET programming patterns.
    """

    VERSION = "1.0.0"
    AUTHOR = "Homeostasis Team"

    def __init__(self):
        """Initialize the F# language plugin."""
        self.language = "fsharp"
        self.supported_extensions = {".fs", ".fsx", ".fsi", ".fsproj"}
        self.supported_frameworks = [
            "dotnet",
            "fsharp",
            "mono",
            "net-framework",
            "giraffe",
            "suave",
            "websharper",
            "fable",
            "bolero",
        ]

        # Initialize components
        self.exception_handler = FSharpExceptionHandler()
        self.patch_generator = FSharpPatchGenerator()

        logger.info("F# language plugin initialized")

    def get_language_id(self) -> str:
        """Get the unique identifier for this language."""
        return "fsharp"

    def get_language_name(self) -> str:
        """Get the human-readable name of the language."""
        return "F#"

    def get_language_version(self) -> str:
        """Get the version of the language supported by this plugin."""
        return "7.0+"

    def get_supported_frameworks(self) -> List[str]:
        """Get the list of frameworks supported by this language plugin."""
        return self.supported_frameworks

    def normalize_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize error data to the standard Homeostasis format.

        Args:
            error_data: Error data in the F#-specific format

        Returns:
            Error data in the standard format
        """
        # Map F#-specific error fields to standard format
        normalized = {
            "error_type": error_data.get("error_type", "FSharpError"),
            "message": error_data.get("message", error_data.get("description", "")),
            "language": "fsharp",
            "file_path": error_data.get("file_path", error_data.get("file", "")),
            "line_number": error_data.get("line_number", error_data.get("line", 0)),
            "column_number": error_data.get(
                "column_number", error_data.get("column", 0)
            ),
            "compiler_version": error_data.get("compiler_version", ""),
            "dotnet_version": error_data.get("dotnet_version", ""),
            "framework_version": error_data.get("framework_version", ""),
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
        Convert standard format error data back to the F#-specific format.

        Args:
            standard_error: Error data in the standard format

        Returns:
            Error data in the F#-specific format
        """
        # Map standard fields back to F#-specific format
        fsharp_error = {
            "error_type": standard_error.get("error_type", "FSharpError"),
            "message": standard_error.get("message", ""),
            "file_path": standard_error.get("file_path", ""),
            "line_number": standard_error.get("line_number", 0),
            "column_number": standard_error.get("column_number", 0),
            "compiler_version": standard_error.get("compiler_version", ""),
            "dotnet_version": standard_error.get("dotnet_version", ""),
            "framework_version": standard_error.get("framework_version", ""),
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
            if key not in fsharp_error and value is not None:
                fsharp_error[key] = value

        return fsharp_error

    def analyze_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze an F# error.

        Args:
            error_data: F# error data

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
            analysis["plugin"] = "fsharp"
            analysis["language"] = "fsharp"
            analysis["plugin_version"] = self.VERSION

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing F# error: {e}")
            return {
                "category": "fsharp",
                "subcategory": "unknown",
                "confidence": "low",
                "suggested_fix": "Unable to analyze F# error",
                "error": str(e),
                "plugin": "fsharp",
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
register_plugin(FSharpLanguagePlugin())
