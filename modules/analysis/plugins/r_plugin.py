"""
R Language Plugin for Homeostasis

This plugin enables Homeostasis to analyze and fix errors in R programming language code.
It provides comprehensive error handling for R syntax errors, runtime issues,
and data science analysis best practices.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..language_plugin_system import LanguagePlugin, register_plugin

logger = logging.getLogger(__name__)


class RExceptionHandler:
    """
    Handles R exceptions with robust error detection and classification.

    This class provides logic for categorizing R errors based on their type,
    message, and common data science patterns.
    """

    def __init__(self):
        """Initialize the R exception handler."""
        self.rule_categories = {
            "syntax": "R syntax and parsing errors",
            "runtime": "Runtime errors and exceptions",
            "data": "Data manipulation and analysis errors",
            "function": "Function definition and calling errors",
            "package": "Package loading and dependency errors",
            "plot": "Plotting and visualization errors",
            "model": "Statistical modeling errors",
            "vector": "Vector and matrix operation errors",
            "dataframe": "Data frame manipulation errors",
            "io": "Input/output and file handling errors",
            "memory": "Memory allocation and performance errors",
            "type": "Type conversion and coercion errors",
        }

        # Common R error patterns
        self.r_error_patterns = {
            "syntax_error": [
                r"unexpected.*in",
                r"unexpected end of input",
                r"unexpected symbol",
                r"missing value where TRUE/FALSE needed",
                r"Error in parse",
                r"unexpected '.*'",
                r"incomplete expression",
                r"unexpected token",
            ],
            "runtime_error": [
                r"object.*not found",
                r"could not find function",
                r"subscript out of bounds",
                r"index.*out of bounds",
                r"invalid type.*argument",
                r"cannot open connection",
                r"argument.*missing",
            ],
            "type_error": [
                r"non-numeric argument",
                r"invalid type",
                r"cannot coerce",
                r"is not subsettable",
                r"attempt to apply non-function",
                r"invalid.*type",
                r"invalid.*coercion",
                r"wrong type.*argument",
                r"non-conformable.*types",
            ],
            "data_error": [
                r"replacement has.*rows",
                r"number of items to replace",
                r"data lengths differ",
                r"non-conformable.*",
                r"more columns than column names",
                r"missing values in object",
                r"NA/NaN/Inf in.*call",
                r"non-finite values",
                r"incorrect number of dimensions",
                r"missing values are not allowed",
            ],
            "function_error": [
                r"could not find function",
                r"unused argument",
                r"argument.*missing",
                r"formal argument.*matched",
                r"'.*' is not a function",
                r"wrong number of arguments",
                r"closure.*not subsettable",
            ],
            "package_error": [
                r"there is no package called",
                r"package.*not available",
                r"error.*loading.*package",
                r"namespace.*not available",
                r"package.*required but not available",
                r"failed to load.*package",
            ],
            "plot_error": [
                r"figure margins too large",
                r"plot.new has not been called",
                r"invalid graphics parameter",
                r"plot region too small",
                r"invalid.*device",
                r"unable to open.*device",
            ],
            "model_error": [
                r"matrix.*singular",
                r"system is exactly singular",
                r"rank-deficient fit",
                r"contrasts.*not defined",
                r"invalid.*formula",
                r"model.*not found",
                r"degrees of freedom",
            ],
            "vector_error": [
                r"subscript out of bounds",
                r"index.*out of bounds",
                r"non-conformable.*",
                r"invalid.*subscript",
                r"length.*differ",
                r"recycling.*length",
            ],
            "dataframe_error": [
                r"undefined columns selected",
                r"column.*not found",
                r"row names.*not allowed",
                r"duplicate.*names",
                r"more columns than column names",
                r"arguments imply differing number of rows",
            ],
            "io_error": [
                r"cannot open.*file",
                r"cannot open connection",
                r"file.*not found",
                r"permission denied",
                r"cannot read.*file",
                r"invalid.*file",
            ],
            "memory_error": [
                r"cannot allocate memory",
                r"memory.*exhausted",
                r"out of memory",
                r"R.*memory",
                r"allocation.*failed",
            ],
        }

        # R-specific concepts and their common issues
        self.r_concepts = {
            "vectors": ["vector", "c()", "length", "subscript"],
            "dataframes": ["data.frame", "column", "row", "subset"],
            "functions": ["function", "argument", "parameter", "call"],
            "packages": ["library", "require", "install.packages", "namespace"],
            "plotting": ["plot", "ggplot", "graphics", "device"],
            "models": ["lm", "glm", "model", "formula"],
            "factors": ["factor", "level", "categorical"],
            "lists": ["list", "[[", "[[]]", "lapply"],
            "matrices": ["matrix", "array", "dim", "nrow", "ncol"],
            "missing": ["NA", "NULL", "missing", "na.omit"],
        }

        # Load rules from different categories
        self.rules = self._load_rules()

        # Pre-compile regex patterns for better performance
        self._compile_patterns()

    def _load_rules(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load R error rules from rule files."""
        rules = {}
        rules_dir = Path(__file__).parent.parent / "rules" / "r"

        try:
            # Load common R rules
            common_rules_path = rules_dir / "r_common_errors.json"
            if common_rules_path.exists():
                with open(common_rules_path, "r") as f:
                    common_data = json.load(f)
                    rules["common"] = common_data.get("rules", [])
                    logger.info(f"Loaded {len(rules['common'])} common R rules")

            # Load concept-specific rules
            for concept in ["data", "functions", "packages", "plotting"]:
                concept_rules_path = rules_dir / f"r_{concept}_errors.json"
                if concept_rules_path.exists():
                    with open(concept_rules_path, "r") as f:
                        concept_data = json.load(f)
                        rules[concept] = concept_data.get("rules", [])
                        logger.info(f"Loaded {len(rules[concept])} {concept} rules")

        except Exception as e:
            logger.error(f"Error loading R rules: {e}")
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
        Analyze an R exception and determine its type and potential fixes.

        Args:
            error_data: R error data in standard format

        Returns:
            Analysis results with categorization and fix suggestions
        """
        message = error_data.get("message", "")
        file_path = error_data.get("file_path", "")
        line_number = error_data.get("line_number", 0)

        # Handle specific error types first
        message_lower = message.lower()
        if "object" in message_lower and "not found" in message_lower:
            analysis = {
                "category": "r",
                "subcategory": "object",
                "confidence": "high",
                "suggested_fix": "Check object name and ensure it's defined",
                "root_cause": "r_object_not_found",
                "severity": "high",
                "tags": ["r", "object", "variable"],
            }
        elif "could not find function" in message_lower:
            analysis = {
                "category": "r",
                "subcategory": "function",
                "confidence": "high",
                "suggested_fix": "Check function name and ensure package is loaded",
                "root_cause": "r_function_not_found",
                "severity": "high",
                "tags": ["r", "function"],
            }
        elif "non-numeric argument" in message_lower:
            analysis = {
                "category": "r",
                "subcategory": "type",
                "confidence": "high",
                "suggested_fix": "Ensure arguments are of correct numeric type",
                "root_cause": "r_type_error",
                "severity": "high",
                "tags": ["r", "type", "numeric"],
            }
        elif "there is no package called" in message_lower:
            analysis = {
                "category": "r",
                "subcategory": "package",
                "confidence": "high",
                "suggested_fix": "Install missing package with install.packages()",
                "root_cause": "r_package_not_found",
                "severity": "high",
                "tags": ["r", "package", "library"],
            }
        elif "incorrect number of dimensions" in message_lower:
            analysis = {
                "category": "r",
                "subcategory": "dimension",
                "confidence": "high",
                "suggested_fix": "Check data structure dimensions",
                "root_cause": "r_dimension_error",
                "severity": "high",
                "tags": ["r", "dimension", "data"],
            }
        elif "missing values are not allowed" in message_lower:
            analysis = {
                "category": "r",
                "subcategory": "na",
                "confidence": "high",
                "suggested_fix": "Handle missing values with na.rm=TRUE or na.omit()",
                "root_cause": "r_na_error",
                "severity": "high",
                "tags": ["r", "na", "missing"],
            }
        else:
            # Check for concept-specific issues first (more specific)
            concept_analysis = self._analyze_r_concepts(message)
            if concept_analysis.get("confidence", "low") != "low":
                # Use concept-specific analysis as base
                analysis = concept_analysis
            else:
                # Fall back to pattern analysis
                analysis = self._analyze_by_patterns(message, file_path)

        # Find matching rules
        matches = self._find_matching_rules(message, error_data)

        if matches:
            # Use the best match (highest confidence)
            best_match = max(matches, key=lambda x: x.get("confidence_score", 0))
            # Only update if we don't already have a high confidence result
            if analysis.get("confidence", "low") != "high":
                analysis.update(
                    {
                        "category": "r",  # Always keep category as "r" for R plugin
                        "subcategory": best_match.get(
                            "category", analysis.get("subcategory", "unknown")
                        ),  # Use rule category as subcategory
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
            else:
                # Just add rule information without overriding the main analysis
                analysis["rule_id"] = best_match.get("id", "")
                analysis["all_matches"] = matches

        analysis["file_path"] = file_path
        analysis["line_number"] = line_number
        return analysis

    def _analyze_by_patterns(self, message: str, file_path: str) -> Dict[str, Any]:
        """Analyze error by matching against common patterns."""
        # Check syntax errors
        for pattern in self.r_error_patterns["syntax_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "r",
                    "subcategory": "syntax",
                    "confidence": "high",
                    "suggested_fix": "Fix R syntax errors",
                    "root_cause": "r_syntax_error",
                    "severity": "high",
                    "tags": ["r", "syntax", "parser"],
                }

        # Check type errors first (more specific)
        for pattern in self.r_error_patterns["type_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "r",
                    "subcategory": "type",
                    "confidence": "high",
                    "suggested_fix": "Fix type mismatch or conversion errors",
                    "root_cause": "r_type_error",
                    "severity": "high",
                    "tags": ["r", "type", "conversion"],
                }

        # Check runtime errors
        for pattern in self.r_error_patterns["runtime_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                # Determine specific subcategory based on the error
                subcategory = "runtime"
                tags = ["r", "runtime"]

                if re.search(r"object.*not found", message, re.IGNORECASE):
                    subcategory = "object"
                    tags.append("object")
                elif "could not find function" in message:
                    subcategory = "function"
                    tags.append("function")

                return {
                    "category": "r",
                    "subcategory": subcategory,
                    "confidence": "high",
                    "suggested_fix": "Fix runtime errors and object access",
                    "root_cause": "r_runtime_error",
                    "severity": "high",
                    "tags": tags,
                }

        # Check data errors
        for pattern in self.r_error_patterns["data_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                # Determine specific subcategory based on the error
                subcategory = "data"
                tags = ["r", "data"]

                if "dimension" in message.lower():
                    subcategory = "dimension"
                    tags.append("dimension")
                elif (
                    "missing values" in message.lower() or
                    "na" in message.lower() or
                    "nan" in message.lower()
                ):
                    subcategory = "na"
                    tags.append("na")

                return {
                    "category": "r",
                    "subcategory": subcategory,
                    "confidence": "high",
                    "suggested_fix": "Fix data manipulation and dimension errors",
                    "root_cause": "r_data_error",
                    "severity": "high",
                    "tags": tags,
                }

        # Check function errors
        for pattern in self.r_error_patterns["function_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "r",
                    "subcategory": "function",
                    "confidence": "high",
                    "suggested_fix": "Fix function definition and calling errors",
                    "root_cause": "r_function_error",
                    "severity": "high",
                    "tags": ["r", "function", "argument"],
                }

        # Check package errors
        for pattern in self.r_error_patterns["package_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "r",
                    "subcategory": "package",
                    "confidence": "high",
                    "suggested_fix": "Fix package loading and dependency errors",
                    "root_cause": "r_package_error",
                    "severity": "high",
                    "tags": ["r", "package", "library"],
                }

        # Check plot errors
        for pattern in self.r_error_patterns["plot_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "r",
                    "subcategory": "plot",
                    "confidence": "high",
                    "suggested_fix": "Fix plotting and visualization errors",
                    "root_cause": "r_plot_error",
                    "severity": "medium",
                    "tags": ["r", "plot", "graphics"],
                }

        # Check model errors
        for pattern in self.r_error_patterns["model_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "r",
                    "subcategory": "model",
                    "confidence": "high",
                    "suggested_fix": "Fix statistical modeling errors",
                    "root_cause": "r_model_error",
                    "severity": "medium",
                    "tags": ["r", "model", "statistics"],
                }

        # Check vector errors
        for pattern in self.r_error_patterns["vector_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "r",
                    "subcategory": "vector",
                    "confidence": "high",
                    "suggested_fix": "Fix vector and matrix operation errors",
                    "root_cause": "r_vector_error",
                    "severity": "high",
                    "tags": ["r", "vector", "matrix"],
                }

        # Check dataframe errors
        for pattern in self.r_error_patterns["dataframe_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "r",
                    "subcategory": "dataframe",
                    "confidence": "high",
                    "suggested_fix": "Fix data frame manipulation errors",
                    "root_cause": "r_dataframe_error",
                    "severity": "high",
                    "tags": ["r", "dataframe", "column"],
                }

        # Check IO errors
        for pattern in self.r_error_patterns["io_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "r",
                    "subcategory": "io",
                    "confidence": "high",
                    "suggested_fix": "Fix input/output and file handling errors",
                    "root_cause": "r_io_error",
                    "severity": "high",
                    "tags": ["r", "io", "file"],
                }

        # Check memory errors
        for pattern in self.r_error_patterns["memory_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "r",
                    "subcategory": "memory",
                    "confidence": "high",
                    "suggested_fix": "Fix memory allocation and performance errors",
                    "root_cause": "r_memory_error",
                    "severity": "critical",
                    "tags": ["r", "memory", "performance"],
                }

        # Check type errors
        for pattern in self.r_error_patterns["type_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "r",
                    "subcategory": "type",
                    "confidence": "high",
                    "suggested_fix": "Fix type conversion and coercion errors",
                    "root_cause": "r_type_error",
                    "severity": "medium",
                    "tags": ["r", "type", "conversion"],
                }

        return {
            "category": "r",
            "subcategory": "unknown",
            "confidence": "low",
            "suggested_fix": "Review R code and error details",
            "root_cause": "r_generic_error",
            "severity": "medium",
            "tags": ["r", "generic"],
        }

    def _analyze_r_concepts(self, message: str) -> Dict[str, Any]:
        """Analyze R-specific concept errors."""
        message_lower = message.lower()

        # Check for object not found errors
        if "object" in message_lower and "not found" in message_lower:
            return {
                "category": "r",
                "subcategory": "object",
                "confidence": "high",
                "suggested_fix": "Check object name and ensure it's defined",
                "root_cause": "r_object_not_found",
                "severity": "high",
                "tags": ["r", "object", "variable"],
            }

        # Check for subscript out of bounds
        if any(keyword in message_lower for keyword in ["subscript", "out of bounds"]):
            return {
                "category": "r",
                "subcategory": "bounds",
                "confidence": "high",
                "suggested_fix": "Check vector/matrix indices and bounds",
                "root_cause": "r_bounds_error",
                "severity": "high",
                "tags": ["r", "bounds", "index"],
            }

        # Check for data frame column errors
        if any(
            keyword in message_lower for keyword in ["column", "undefined", "selected"]
        ):
            return {
                "category": "r",
                "subcategory": "dataframe",
                "confidence": "medium",
                "suggested_fix": "Check data frame column names and selection",
                "root_cause": "r_dataframe_error",
                "severity": "medium",
                "tags": ["r", "dataframe", "column"],
            }

        # Check for package errors
        if any(
            keyword in message_lower for keyword in ["package", "library", "namespace"]
        ):
            return {
                "category": "r",
                "subcategory": "package",
                "confidence": "medium",
                "suggested_fix": "Check package installation and loading",
                "root_cause": "r_package_error",
                "severity": "medium",
                "tags": ["r", "package", "library"],
            }

        # Check for function errors
        if any(
            keyword in message_lower for keyword in ["function", "argument", "unused"]
        ):
            return {
                "category": "r",
                "subcategory": "function",
                "confidence": "medium",
                "suggested_fix": "Check function definition and arguments",
                "root_cause": "r_function_error",
                "severity": "medium",
                "tags": ["r", "function", "argument"],
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
        if file_path.endswith(".R") or file_path.endswith(".r"):
            base_confidence += 0.2

        # Boost confidence based on rule reliability
        reliability = rule.get("reliability", "medium")
        reliability_boost = {"high": 0.2, "medium": 0.1, "low": 0.0}
        base_confidence += reliability_boost.get(reliability, 0.0)

        # Boost confidence for concept matches
        rule_tags = set(rule.get("tags", []))
        context_tags = set()

        message = error_data.get("message", "").lower()
        if "object" in message:
            context_tags.add("object")
        if "function" in message:
            context_tags.add("function")
        if "package" in message:
            context_tags.add("package")

        if context_tags & rule_tags:
            base_confidence += 0.1

        return min(base_confidence, 1.0)


class RPatchGenerator:
    """
    Generates patches for R errors based on analysis results.

    This class creates R code fixes for common errors using templates
    and heuristics specific to data science patterns.
    """

    def __init__(self):
        """Initialize the R patch generator."""
        self.template_dir = (
            Path(__file__).parent.parent / "patch_generation" / "templates"
        )
        self.r_template_dir = self.template_dir / "r"

        # Ensure template directory exists
        self.r_template_dir.mkdir(parents=True, exist_ok=True)

        # Load patch templates
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict[str, str]:
        """Load R patch templates."""
        templates = {}

        if not self.r_template_dir.exists():
            logger.warning(f"R templates directory not found: {self.r_template_dir}")
            return templates

        for template_file in self.r_template_dir.glob("*.R.template"):
            try:
                with open(template_file, "r") as f:
                    template_name = template_file.stem.replace(".R", "")
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
        Generate a patch for the R error.

        Args:
            error_data: The R error data
            analysis: Analysis results from RExceptionHandler
            source_code: The R source code that caused the error

        Returns:
            Patch information or None if no patch can be generated
        """
        root_cause = analysis.get("root_cause", "")

        # Map root causes to patch strategies
        patch_strategies = {
            "r_syntax_error": self._fix_syntax_error,
            "r_runtime_error": self._fix_runtime_error,
            "r_data_error": self._fix_data_error,
            "r_function_error": self._fix_function_error,
            "r_package_error": self._fix_package_error,
            "r_plot_error": self._fix_plot_error,
            "r_model_error": self._fix_model_error,
            "r_vector_error": self._fix_vector_error,
            "r_dataframe_error": self._fix_dataframe_error,
            "r_io_error": self._fix_io_error,
            "r_memory_error": self._fix_memory_error,
            "r_type_error": self._fix_type_error,
            "r_object_not_found": self._fix_object_not_found_error,
            "r_bounds_error": self._fix_bounds_error,
            "r_dimension_error": self._fix_data_error,  # Dimension errors are data errors
            "r_na_error": self._fix_data_error,  # NA errors are also data errors
            "r_function_not_found": self._fix_function_error,  # Function not found errors
            "r_package_not_found": self._fix_package_error,  # Package not found errors
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
        """Fix R syntax errors."""
        message = error_data.get("message", "")

        fixes = []

        if "unexpected" in message.lower():
            fixes.append(
                {
                    "type": "suggestion",
                    "description": "Unexpected symbol in code",
                    "fix": "Check for missing parentheses, brackets, or operators",
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

        if "unexpected end of input" in message.lower():
            fixes.append(
                {
                    "type": "suggestion",
                    "description": "Unexpected end of input",
                    "fix": "Check for missing closing parentheses or brackets",
                }
            )

        if fixes:
            return {
                "type": "multiple_suggestions",
                "fixes": fixes,
                "description": "R syntax error fixes",
            }

        return {
            "type": "suggestion",
            "description": "R syntax error. Check code structure and syntax",
        }

    def _fix_runtime_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix runtime errors."""
        message = error_data.get("message", "")

        if "object" in message.lower() and "not found" in message.lower():
            # Extract object name
            obj_match = re.search(r"object '(.+?)' not found", message)
            obj_name = obj_match.group(1) if obj_match else "object"

            return {
                "type": "suggestion",
                "description": f"Object '{obj_name}' not found",
                "fixes": [
                    f"Check spelling of '{obj_name}'",
                    f"Ensure '{obj_name}' is defined before use",
                    f"Load required package containing '{obj_name}'",
                    "Check workspace with ls() to see available objects",
                    f"Use exists('{obj_name}') to check if object exists",
                ],
            }

        if "could not find function" in message.lower():
            # Extract function name
            func_match = re.search(r"could not find function \"(.+?)\"", message)
            func_name = func_match.group(1) if func_match else "function"

            return {
                "type": "suggestion",
                "description": f"Function '{func_name}' not found",
                "fixes": [
                    f"Load required package containing '{func_name}'",
                    f"Check spelling of '{func_name}'",
                    "Install missing package with install.packages()",
                    f"Check function name with ?{func_name}",
                    f"Use apropos('{func_name}') to find similar functions",
                ],
            }

        return {
            "type": "suggestion",
            "description": "Runtime error. Check object and function definitions",
        }

    def _fix_data_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix data manipulation errors."""
        message = error_data.get("message", "")

        # Handle dimension-specific errors
        if "dimension" in message.lower():
            return {
                "type": "suggestion",
                "description": "Dimension error",
                "fixes": [
                    "Check array/matrix dimensions with dim()",
                    "Verify correct number of dimensions for operation",
                    "Use drop=FALSE to preserve dimensions",
                    "Check indexing operations",
                ],
            }

        # Handle NA/missing value errors
        if "missing values" in message.lower() or " na " in message.lower():
            return {
                "type": "suggestion",
                "description": "Missing values (NA) error",
                "fixes": [
                    "Use na.rm=TRUE in functions that support it",
                    "Remove missing values with na.omit()",
                    "Check for NAs with is.na()",
                    "Handle missing values appropriately for your analysis",
                ],
            }

        if "data lengths differ" in message.lower():
            return {
                "type": "suggestion",
                "description": "Data lengths differ",
                "fixes": [
                    "Check vector lengths with length()",
                    "Use rep() to repeat values to match lengths",
                    "Use recycling carefully with vectors of different lengths",
                    "Check data frame dimensions with dim()",
                    "Use head() or tail() to examine data structure",
                ],
            }

        if "non-conformable" in message.lower():
            return {
                "type": "suggestion",
                "description": "Non-conformable arrays/matrices",
                "fixes": [
                    "Check matrix dimensions with dim()",
                    "Use t() to transpose matrices if needed",
                    "Ensure matrices have compatible dimensions for operations",
                    "Use %*% for matrix multiplication",
                    "Check array dimensions with str()",
                ],
            }

        if "missing values" in message.lower():
            return {
                "type": "suggestion",
                "description": "Missing values in data",
                "fixes": [
                    "Remove missing values with na.omit()",
                    "Check for NA values with is.na()",
                    "Use complete.cases() to find complete rows",
                    "Handle missing values with na.rm = TRUE",
                    "Impute missing values appropriately",
                ],
            }

        return {
            "type": "suggestion",
            "description": "Data manipulation error",
            "fixes": [
                "Check data dimensions and types",
                "Verify data compatibility for operations",
                "Handle missing values appropriately",
            ],
        }

    def _fix_function_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix function errors."""
        message = error_data.get("message", "")

        if "unused argument" in message.lower():
            return {
                "type": "suggestion",
                "description": "Unused argument in function call",
                "fixes": [
                    "Remove unused arguments from function call",
                    "Check function documentation with ?function_name",
                    "Verify correct argument names",
                    "Use args(function_name) to see expected arguments",
                    "Check for typos in argument names",
                ],
            }

        if "argument" in message.lower() and "missing" in message.lower():
            return {
                "type": "suggestion",
                "description": "Missing required argument",
                "fixes": [
                    "Provide all required arguments to function",
                    "Check function documentation for required parameters",
                    "Use default values where appropriate",
                    "Verify argument order and names",
                ],
            }

        if "formal argument" in message.lower() and "matched" in message.lower():
            return {
                "type": "suggestion",
                "description": "Formal argument matching error",
                "fixes": [
                    "Check argument names for typos",
                    "Use positional arguments in correct order",
                    "Verify function signature with args()",
                    "Use partial matching carefully",
                ],
            }

        return {
            "type": "suggestion",
            "description": "Function error",
            "fixes": [
                "Check function arguments and names",
                "Verify function exists and is loaded",
                "Review function documentation",
            ],
        }

    def _fix_package_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix package loading errors."""
        message = error_data.get("message", "")

        if "no package called" in message.lower():
            # Extract package name
            pkg_match = re.search(r"no package called '(.+?)'", message)
            pkg_name = pkg_match.group(1) if pkg_match else "package"

            return {
                "type": "suggestion",
                "description": f"Package '{pkg_name}' not found",
                "fixes": [
                    f"Install package: install.packages('{pkg_name}')",
                    "Check package name spelling",
                    "Update package repositories with update.packages()",
                    f"Check CRAN availability for '{pkg_name}'",
                    "Use available.packages() to see available packages",
                ],
            }

        if "package" in message.lower() and "not available" in message.lower():
            return {
                "type": "suggestion",
                "description": "Package not available",
                "fixes": [
                    "Check if package exists on CRAN",
                    "Try installing from different repository",
                    "Check R version compatibility",
                    "Install dependencies first",
                    "Try installing from GitHub if available",
                ],
            }

        return {
            "type": "suggestion",
            "description": "Package loading error",
            "fixes": [
                "Check package installation and availability",
                "Verify package names and repositories",
                "Update R and packages if needed",
            ],
        }

    def _fix_plot_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix plotting errors."""
        message = error_data.get("message", "")

        if "margins too large" in message.lower():
            return {
                "type": "suggestion",
                "description": "Plot margins too large",
                "fixes": [
                    "Increase plot window size",
                    "Adjust margins with par(mar=c(...))",
                    "Use smaller margin values",
                    "Check plot device size",
                    "Use layout() or par(mfrow=...) for multiple plots",
                ],
            }

        if "plot.new has not been called" in message.lower():
            return {
                "type": "suggestion",
                "description": "Plot device not initialized",
                "fixes": [
                    "Call plot() or plot.new() first",
                    "Initialize plot device properly",
                    "Check plot sequence and order",
                    "Use dev.new() for new plot device",
                ],
            }

        return {
            "type": "suggestion",
            "description": "Plotting error",
            "fixes": [
                "Check plot device and margins",
                "Verify plot function calls and sequence",
                "Adjust plot parameters as needed",
            ],
        }

    def _fix_model_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix statistical modeling errors."""
        message = error_data.get("message", "")

        if "singular" in message.lower():
            return {
                "type": "suggestion",
                "description": "Singular matrix in model",
                "fixes": [
                    "Check for multicollinearity in predictors",
                    "Remove redundant variables",
                    "Use regularization techniques",
                    "Check data quality and completeness",
                    "Consider using ridge regression or other robust methods",
                ],
            }

        if "contrasts" in message.lower():
            return {
                "type": "suggestion",
                "description": "Contrasts not defined",
                "fixes": [
                    "Set contrasts for factors explicitly",
                    "Use contrasts() function to define contrasts",
                    "Check factor levels and structure",
                    "Consider using different contrast coding",
                ],
            }

        return {
            "type": "suggestion",
            "description": "Statistical modeling error",
            "fixes": [
                "Check model specification and data",
                "Verify variable types and structure",
                "Consider alternative modeling approaches",
            ],
        }

    def _fix_vector_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix vector and matrix operation errors."""
        message = error_data.get("message", "")

        if "subscript out of bounds" in message.lower():
            return {
                "type": "suggestion",
                "description": "Subscript out of bounds",
                "fixes": [
                    "Check vector/matrix dimensions with length() or dim()",
                    "Use valid indices within bounds",
                    "Check for off-by-one errors",
                    "Use seq_along() for safe indexing",
                    "Verify index calculations",
                ],
            }

        if "non-conformable" in message.lower():
            return {
                "type": "suggestion",
                "description": "Non-conformable operations",
                "fixes": [
                    "Check dimensions with dim() or length()",
                    "Use appropriate matrix operations",
                    "Transpose matrices if needed with t()",
                    "Use element-wise operations when appropriate",
                    "Verify matrix multiplication requirements",
                ],
            }

        return {
            "type": "suggestion",
            "description": "Vector/matrix operation error",
            "fixes": [
                "Check dimensions and indices",
                "Verify operation compatibility",
                "Use appropriate vector/matrix functions",
            ],
        }

    def _fix_dataframe_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix data frame manipulation errors."""
        message = error_data.get("message", "")

        if "undefined columns selected" in message.lower():
            return {
                "type": "suggestion",
                "description": "Undefined columns selected",
                "fixes": [
                    "Check column names with names() or colnames()",
                    "Use str() to examine data frame structure",
                    "Check for typos in column names",
                    "Use which(names(df) == 'column_name') to find columns",
                    "Use exists() to check if column exists",
                ],
            }

        if "more columns than column names" in message.lower():
            return {
                "type": "suggestion",
                "description": "More columns than column names",
                "fixes": [
                    "Provide column names for all columns",
                    "Check data import settings",
                    "Use proper column specifications",
                    "Set col.names parameter appropriately",
                ],
            }

        if "arguments imply differing number of rows" in message.lower():
            return {
                "type": "suggestion",
                "description": "Differing number of rows",
                "fixes": [
                    "Check vector lengths with length()",
                    "Ensure all columns have same length",
                    "Use data.frame() with proper recycling",
                    "Check data alignment and structure",
                ],
            }

        return {
            "type": "suggestion",
            "description": "Data frame manipulation error",
            "fixes": [
                "Check data frame structure and column names",
                "Verify data dimensions and types",
                "Use proper data frame operations",
            ],
        }

    def _fix_io_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix IO and file handling errors."""
        message = error_data.get("message", "")

        if "cannot open" in message.lower():
            return {
                "type": "suggestion",
                "description": "Cannot open file",
                "fixes": [
                    "Check file path and name",
                    "Verify file exists with file.exists()",
                    "Check working directory with getwd()",
                    "Use proper file path separators",
                    "Check file permissions",
                ],
            }

        if "file" in message.lower() and "not found" in message.lower():
            return {
                "type": "suggestion",
                "description": "File not found",
                "fixes": [
                    "Check file path and spelling",
                    "Use list.files() to see available files",
                    "Set working directory with setwd()",
                    "Use full file paths",
                    "Check file extension",
                ],
            }

        return {
            "type": "suggestion",
            "description": "IO error",
            "fixes": [
                "Check file paths and permissions",
                "Verify file existence and accessibility",
                "Use proper file handling functions",
            ],
        }

    def _fix_memory_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix memory allocation errors."""
        message = error_data.get("message", "")

        if "memory" in message.lower():
            return {
                "type": "suggestion",
                "description": "Memory allocation error",
                "fixes": [
                    "Reduce data size or use sampling",
                    "Process data in chunks",
                    "Use memory-efficient data structures",
                    "Clear unused objects with rm()",
                    "Use gc() to trigger garbage collection",
                    "Consider using data.table for large datasets",
                ],
            }

        return {
            "type": "suggestion",
            "description": "Memory error",
            "fixes": [
                "Optimize memory usage",
                "Use efficient data processing methods",
                "Clear unused objects regularly",
            ],
        }

    def _fix_type_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix type conversion errors."""
        message = error_data.get("message", "")

        if "coercion" in message.lower() or "coerce" in message.lower():
            return {
                "type": "suggestion",
                "description": "Type coercion error",
                "fixes": [
                    "Use explicit type conversion functions",
                    "Check data types with class() or typeof()",
                    "Use as.numeric(), as.character(), etc.",
                    "Handle factor to numeric conversion properly",
                    "Use suppressWarnings() for expected coercion warnings",
                ],
            }

        if "non-numeric argument" in message.lower():
            return {
                "type": "suggestion",
                "description": "Non-numeric argument",
                "fixes": [
                    "Convert to numeric with as.numeric()",
                    "Check for character values in numeric columns",
                    "Use is.numeric() to test for numeric type",
                    "Handle factor variables appropriately",
                    "Check for missing values that prevent conversion",
                ],
            }

        return {
            "type": "suggestion",
            "description": "Type conversion error",
            "fixes": [
                "Check data types and conversions",
                "Use appropriate type conversion functions",
                "Handle type mismatches carefully",
            ],
        }

    def _fix_object_not_found_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix object not found errors."""
        return {
            "type": "suggestion",
            "description": "Object not found",
            "fixes": [
                "Check object name spelling",
                "Ensure object is defined before use",
                "Check workspace with ls()",
                "Load required packages",
                "Use exists() to check if object exists",
            ],
        }

    def _fix_bounds_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix subscript bounds errors."""
        return {
            "type": "suggestion",
            "description": "Subscript out of bounds",
            "fixes": [
                "Check vector/matrix dimensions with length() or dim()",
                "Use valid indices within bounds",
                "Use seq_along() for safe indexing",
                "Verify index calculations",
                "Check for off-by-one errors",
            ],
        }

    def _template_based_patch(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Generate patch using templates."""
        root_cause = analysis.get("root_cause", "")

        # Map root causes to template names
        template_map = {
            "r_syntax_error": "syntax_fix",
            "r_runtime_error": "runtime_fix",
            "r_object_not_found": "object_fix",
            "r_function_error": "function_fix",
            "r_package_error": "package_fix",
            "r_data_error": "data_fix",
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


class RLanguagePlugin(LanguagePlugin):
    """
    Main R language plugin for Homeostasis.

    This plugin orchestrates R error analysis and patch generation,
    supporting data science and statistical analysis patterns.
    """

    VERSION = "1.0.0"
    AUTHOR = "Homeostasis Team"

    def __init__(self):
        """Initialize the R language plugin."""
        self.language = "r"
        self.supported_extensions = {".R", ".r", ".Rmd", ".rmd", ".Rnw"}
        self.supported_frameworks = [
            "r",
            "rstudio",
            "shiny",
            "rmarkdown",
            "ggplot2",
            "dplyr",
            "tidyr",
            "tidyverse",
            "caret",
            "mlr3",
        ]

        # Initialize components
        self.exception_handler = RExceptionHandler()
        self.patch_generator = RPatchGenerator()

        logger.info("R language plugin initialized")

    def get_language_id(self) -> str:
        """Get the unique identifier for this language."""
        return "r"

    def get_language_name(self) -> str:
        """Get the human-readable name of the language."""
        return "R"

    def get_language_version(self) -> str:
        """Get the version of the language supported by this plugin."""
        return "4.0+"

    def get_supported_frameworks(self) -> List[str]:
        """Get the list of frameworks supported by this language plugin."""
        return self.supported_frameworks

    def normalize_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize error data to the standard Homeostasis format.

        Args:
            error_data: Error data in the R-specific format

        Returns:
            Error data in the standard format
        """
        # Map R-specific error fields to standard format
        normalized = {
            "error_type": error_data.get("error_type", "RError"),
            "message": error_data.get("message", error_data.get("description", "")),
            "language": "r",
            "file_path": error_data.get("file_path", error_data.get("file", "")),
            "line_number": error_data.get("line_number", error_data.get("line", 0)),
            "column_number": error_data.get(
                "column_number", error_data.get("column", 0)
            ),
            "r_version": error_data.get("r_version", ""),
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
        Convert standard format error data back to the R-specific format.

        Args:
            standard_error: Error data in the standard format

        Returns:
            Error data in the R-specific format
        """
        # Map standard fields back to R-specific format
        r_error = {
            "error_type": standard_error.get("error_type", "RError"),
            "message": standard_error.get("message", ""),
            "file_path": standard_error.get("file_path", ""),
            "line_number": standard_error.get("line_number", 0),
            "column_number": standard_error.get("column_number", 0),
            "r_version": standard_error.get("r_version", ""),
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
            if key not in r_error and value is not None:
                r_error[key] = value

        return r_error

    def analyze_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze an R error.

        Args:
            error_data: R error data

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
            analysis["plugin"] = "r"
            analysis["language"] = "r"
            analysis["plugin_version"] = self.VERSION

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing R error: {e}")
            return {
                "category": "r",
                "subcategory": "unknown",
                "confidence": "low",
                "suggested_fix": "Unable to analyze R error",
                "error": str(e),
                "plugin": "r",
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
register_plugin(RLanguagePlugin())
