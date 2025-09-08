"""
TypeScript Language Plugin for Homeostasis

This plugin enables Homeostasis to analyze and fix errors in TypeScript applications.
It provides comprehensive error handling for TypeScript compilation, type checking,
transpilation, and runtime errors, supporting both pure TypeScript and TypeScript
frameworks like React, Angular, and Vue with TypeScript.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..language_adapters import TypeScriptErrorAdapter
from ..language_plugin_system import LanguagePlugin, register_plugin

logger = logging.getLogger(__name__)


class TypeScriptExceptionHandler:
    """
    Handles TypeScript exceptions with comprehensive error detection and classification.

    This class provides logic for categorizing TypeScript compilation errors, type errors,
    transpilation issues, and runtime errors in TypeScript applications.
    """

    def __init__(self):
        """Initialize the TypeScript exception handler."""
        self.rule_categories = {
            "type_errors": "TypeScript type system errors",
            "compilation": "TypeScript compilation and syntax errors",
            "advanced_features": "Advanced TypeScript features errors",
            "config": "TypeScript configuration errors",
            "runtime": "TypeScript runtime errors",
            "transpilation": "TypeScript transpilation errors",
            "jsx_tsx": "JSX/TSX related errors",
            "decorators": "TypeScript decorator errors",
            "generics": "Generic type errors",
            "modules": "Module resolution errors",
        }

        # Load rules from different categories
        self.rules = self._load_rules()

        # Pre-compile regex patterns for better performance
        self._compile_patterns()

    def _load_rules(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load TypeScript error rules from rule files."""
        rules = {}
        rules_dir = Path(__file__).parent.parent / "rules" / "typescript"

        try:
            # Load type checking rules
            type_rules_path = rules_dir / "type_errors.json"
            if type_rules_path.exists():
                with open(type_rules_path, "r") as f:
                    type_data = json.load(f)
                    rules["type_errors"] = type_data.get("rules", [])
                    logger.info(
                        f"Loaded {len(rules['type_errors'])} TypeScript type error rules"
                    )

            # Load compilation rules
            comp_rules_path = rules_dir / "compilation_errors.json"
            if comp_rules_path.exists():
                with open(comp_rules_path, "r") as f:
                    comp_data = json.load(f)
                    rules["compilation"] = comp_data.get("rules", [])
                    logger.info(
                        f"Loaded {len(rules['compilation'])} TypeScript compilation rules"
                    )

            # Load advanced features rules
            advanced_rules_path = rules_dir / "advanced_features.json"
            if advanced_rules_path.exists():
                with open(advanced_rules_path, "r") as f:
                    advanced_data = json.load(f)
                    rules["advanced_features"] = advanced_data.get("rules", [])
                    logger.info(
                        f"Loaded {len(rules['advanced_features'])} TypeScript advanced features rules"
                    )

        except Exception as e:
            logger.error(f"Error loading TypeScript rules: {e}")
            rules = {"type_errors": [], "compilation": [], "advanced_features": []}

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
                        f"Invalid regex pattern in TypeScript rule {rule.get('id', 'unknown')}: {e}"
                    )

    def analyze_exception(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a TypeScript exception and determine its type and potential fixes.

        Args:
            error_data: TypeScript error data in standard format

        Returns:
            Analysis results with categorization and fix suggestions
        """
        error_type = error_data.get("error_type", "TSError")
        message = error_data.get("message", "")
        stack_trace = error_data.get("stack_trace", [])

        # Convert stack trace to string for pattern matching
        stack_str = ""
        if isinstance(stack_trace, list):
            stack_str = "\n".join([str(frame) for frame in stack_trace])
        elif isinstance(stack_trace, str):
            stack_str = stack_trace

        # Combine error info for analysis
        full_error_text = f"{error_type}: {message}\n{stack_str}"

        # Find matching rules
        matches = self._find_matching_rules(full_error_text, error_data)

        if matches:
            # Use the best match (highest confidence)
            best_match = max(matches, key=lambda x: x.get("confidence_score", 0))

            # Enhance suggestion with captured groups for specific error codes
            suggestion = best_match.get("suggestion", "")
            if best_match.get("error_code") == "TS2304" and best_match.get(
                "match_groups"
            ):
                identifier = best_match["match_groups"][0]
                suggestion = f"Cannot find name '{identifier}' - check if the identifier is declared, imported correctly, or install missing type definitions"
            elif (best_match.get("error_code") == "TS2322" and
                    len(best_match.get("match_groups", [])) >= 2):
                source_type = best_match["match_groups"][0]
                target_type = best_match["match_groups"][1]
                suggestion = f"Type '{source_type}' is not assignable to type '{target_type}' - fix type compatibility"

            return {
                "category": best_match.get("category", "typescript"),
                "subcategory": best_match.get("subcategory", "unknown"),
                "confidence": best_match.get("confidence", "medium"),
                "suggested_fix": suggestion,
                "root_cause": best_match.get("root_cause", ""),
                "severity": best_match.get("severity", "medium"),
                "rule_id": best_match.get("id", ""),
                "error_code": best_match.get("error_code", ""),
                "tags": best_match.get("tags", []),
                "fix_commands": best_match.get("fix_commands", []),
                "all_matches": matches,
            }

        # If no rules matched, provide generic analysis
        return self._generic_analysis(error_data)

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

        # Boost confidence for TypeScript error code matches
        error_code = rule.get("error_code", "")
        error_type = error_data.get("error_type", "")

        if error_code and error_code in error_type:
            base_confidence += 0.4

        # Boost confidence based on rule reliability
        reliability = rule.get("reliability", "medium")
        reliability_boost = {"high": 0.2, "medium": 0.1, "low": 0.0}
        base_confidence += reliability_boost.get(reliability, 0.0)

        # Boost confidence for rules with specific tags that match context
        rule_tags = set(rule.get("tags", []))
        context_tags = set()

        # Infer context from error data
        if "typescript" in error_data.get("language", "").lower():
            context_tags.add("typescript")
        if error_data.get("framework"):
            context_tags.add(error_data["framework"].lower())
        if "jsx" in error_data.get("message", "").lower() or "tsx" in str(
            error_data.get("stack_trace", "")
        ):
            context_tags.add("jsx")

        if context_tags & rule_tags:
            base_confidence += 0.1

        return min(base_confidence, 1.0)

    def _generic_analysis(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """Provide generic analysis for unmatched errors."""
        error_type = error_data.get("error_type", "TSError")
        error_code = error_data.get("error_code", error_type)
        message = error_data.get("message", "")

        # Basic categorization based on error type
        if error_type.startswith("TS") or error_code.startswith("TS"):
            # TypeScript compiler error
            if error_code.startswith("TS2"):
                category = "type_error"
                suggestion = "Fix type-related issues in your TypeScript code"
                # Special handling for TS2304 - extract identifier
                if error_code == "TS2304":
                    name_match = re.search(r"Cannot find name '([^']+)'", message)
                    if name_match:
                        identifier = name_match.group(1)
                        suggestion = f"Cannot find name '{identifier}' - check if the identifier is declared, imported correctly, or install missing type definitions"
            elif error_code.startswith("TS1"):
                category = "syntax_error"
                suggestion = "Fix syntax errors in your TypeScript code"
            elif error_code.startswith("TS5"):
                category = "config_error"
                suggestion = "Check TypeScript configuration in tsconfig.json"
            elif error_code.startswith("TS6"):
                category = "warning"
                suggestion = "Address TypeScript compiler warnings"
            elif error_code.startswith("TS7"):
                category = "declaration_error"
                suggestion = "Fix type declaration issues"
            else:
                category = "compilation_error"
                suggestion = "Fix TypeScript compilation error"
        else:
            # Runtime error or other
            category = "runtime_error"
            suggestion = "Fix runtime error in TypeScript application"

        return {
            "category": "typescript",
            "subcategory": category,
            "confidence": "low",
            "suggested_fix": suggestion,
            "root_cause": f"typescript_{category}",
            "severity": "medium",
            "rule_id": "ts_generic_handler",
            "error_code": error_code,
            "tags": ["typescript", "generic"],
        }

    def analyze_compilation_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze TypeScript compilation errors.

        Args:
            error_data: Error data from TypeScript compiler

        Returns:
            Analysis results with compilation-specific fixes
        """
        message = error_data.get("message", "")
        error_type = error_data.get("error_type", "")
        error_code = error_data.get("error_code", "")

        # Extract TypeScript error code if present
        if not error_code:
            ts_error_match = re.search(r"TS(\d+)", message + error_type)
            if ts_error_match:
                error_code = f"TS{ts_error_match.group(1)}"

        # Common TypeScript compilation error patterns
        compilation_patterns = {
            "TS1005": {
                "category": "syntax",
                "fix": "Add the missing token indicated in the error message",
                "severity": "error",
            },
            "TS1127": {
                "category": "syntax",
                "fix": "Remove invalid characters from the source code",
                "severity": "error",
            },
            "TS5023": {
                "category": "config",
                "fix": "Remove or fix unknown compiler option in tsconfig.json",
                "severity": "error",
            },
            "TS5024": {
                "category": "config",
                "fix": "Provide a value for the compiler option",
                "severity": "error",
            },
            "TS6133": {
                "category": "unused",
                "fix": "Remove unused variable or prefix with underscore",
                "severity": "warning",
            },
        }

        if error_code in compilation_patterns:
            pattern_info = compilation_patterns[error_code]
            return {
                "category": "typescript",
                "subcategory": pattern_info["category"],
                "confidence": "high",
                "suggested_fix": pattern_info["fix"],
                "root_cause": f"typescript_compilation_{pattern_info['category']}",
                "severity": pattern_info["severity"],
                "error_code": error_code,
                "tags": ["typescript", "compilation", pattern_info["category"]],
            }

        # Generic compilation error
        return {
            "category": "typescript",
            "subcategory": "compilation",
            "confidence": "medium",
            "suggested_fix": "Fix TypeScript compilation error",
            "root_cause": "typescript_compilation_error",
            "severity": "error",
            "error_code": error_code,
            "tags": ["typescript", "compilation"],
        }

    def analyze_type_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze TypeScript type system errors.

        Args:
            error_data: Error data with type system issues

        Returns:
            Analysis results with type-specific fixes
        """
        message = error_data.get("message", "")
        error_code = error_data.get("error_code", "")

        # Common type error patterns
        type_patterns = {
            "TS2304": "Cannot find name - check imports and declarations",
            "TS2322": "Type assignment mismatch - fix type compatibility",
            "TS2339": "Property does not exist - check property names and types",
            "TS2307": "Cannot find module - install missing dependency or fix import path",
            "TS2345": "Argument type mismatch - fix function argument types",
            "TS2540": "Read-only property assignment - use immutable update patterns",
            "TS2571": "Unknown type access - use type guards or assertions",
            "TS2564": "Uninitialized property - add initialization or definite assignment assertion",
        }

        fix_suggestion = type_patterns.get(error_code, "Fix TypeScript type error")

        # For TS2304, extract identifier and provide more specific suggestion
        if error_code == "TS2304":
            name_match = re.search(r"Cannot find name '([^']+)'", message)
            if name_match:
                identifier = name_match.group(1)
                fix_suggestion = f"Cannot find name '{identifier}' - check if the identifier is declared, imported correctly, or install missing type definitions"
        # For TS2322, include types in suggestion
        elif error_code == "TS2322":
            type_match = re.search(
                r"Type '([^']+)' is not assignable to type '([^']+)'", message
            )
            if type_match:
                fix_suggestion = f"Type '{type_match.group(1)}' is not assignable to type '{type_match.group(2)}' - fix type compatibility"

        return {
            "category": "typescript",
            "subcategory": "type_error",
            "confidence": "high" if error_code in type_patterns else "medium",
            "suggested_fix": fix_suggestion,
            "root_cause": (
                f"typescript_type_error_{error_code.lower()}"
                if error_code
                else "typescript_type_error"
            ),
            "severity": "error",
            "error_code": error_code,
            "tags": ["typescript", "types", "type-checking"],
        }


class TypeScriptPatchGenerator:
    """
    Generates patches for TypeScript errors based on analysis results.

    This class creates code fixes for common TypeScript errors using templates
    and heuristics specific to TypeScript patterns and best practices.
    """

    def __init__(self):
        """Initialize the TypeScript patch generator."""
        self.template_dir = (
            Path(__file__).parent.parent / "patch_generation" / "templates"
        )
        self.ts_template_dir = self.template_dir / "typescript"

        # Ensure template directory exists
        self.ts_template_dir.mkdir(parents=True, exist_ok=True)

        # Load patch templates
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict[str, str]:
        """Load TypeScript patch templates."""
        templates = {}

        if not self.ts_template_dir.exists():
            logger.warning(
                f"TypeScript templates directory not found: {self.ts_template_dir}"
            )
            return templates

        for template_file in self.ts_template_dir.glob("*.ts.template"):
            try:
                with open(template_file, "r") as f:
                    template_name = template_file.stem.replace(".ts", "")
                    templates[template_name] = f.read()
                    logger.debug(f"Loaded TypeScript template: {template_name}")
            except Exception as e:
                logger.error(f"Error loading TypeScript template {template_file}: {e}")

        return templates

    def generate_patch(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a patch for the TypeScript error.

        Args:
            error_data: The TypeScript error data
            analysis: Analysis results from TypeScriptExceptionHandler
            source_code: The source code where the error occurred

        Returns:
            Patch information or None if no patch can be generated
        """
        root_cause = analysis.get("root_cause", "")

        # Map root causes to patch strategies
        patch_strategies = {
            "typescript_type_error_ts2304": self._fix_cannot_find_name,
            "typescript_type_error_ts2322": self._fix_type_assignment,
            "typescript_type_error_ts2339": self._fix_property_not_exist,
            "typescript_type_error_ts2307": self._fix_module_not_found,
            "typescript_compilation_syntax": self._fix_syntax_error,
            "typescript_compilation_config": self._fix_config_error,
            "typescript_compilation_unused": self._fix_unused_variable,
        }

        strategy = patch_strategies.get(root_cause)
        if strategy:
            try:
                return strategy(error_data, analysis, source_code)
            except Exception as e:
                logger.error(f"Error generating TypeScript patch for {root_cause}: {e}")

        # Try to use templates if no specific strategy matches
        return self._template_based_patch(error_data, analysis, source_code)

    def _fix_cannot_find_name(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix TS2304: Cannot find name errors."""
        message = error_data.get("message", "")

        # Extract identifier name from error message
        name_match = re.search(r"Cannot find name '([^']+)'", message)
        if not name_match:
            return None

        identifier = name_match.group(1)

        # Common fixes based on identifier patterns
        common_fixes = {
            "React": "import React from 'react';",
            "Component": "import { Component } from 'react';",
            "useState": "import { useState } from 'react';",
            "useEffect": "import { useEffect } from 'react';",
            "process": "// Install @types/node: npm install --save-dev @types/node",
            "Buffer": "// Install @types/node: npm install --save-dev @types/node",
            "console": "// This should be available globally. Check TypeScript configuration.",
            "document": "// This should be available in browser environment. Check lib in tsconfig.json",
            "window": "// This should be available in browser environment. Check lib in tsconfig.json",
        }

        if identifier in common_fixes:
            return {
                "type": "suggestion",
                "description": f"Add import or type declaration for '{identifier}'",
                "fix_code": common_fixes[identifier],
                "confidence": "high",
            }

        return {
            "type": "suggestion",
            "description": f"Cannot find name '{identifier}'. Add import statement, declare the variable, or install missing type definitions.",
        }

    def _fix_type_assignment(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix TS2322: Type assignment errors."""
        message = error_data.get("message", "")

        # Extract types from error message
        type_match = re.search(
            r"Type '([^']+)' is not assignable to type '([^']+)'", message
        )
        if not type_match:
            return None

        source_type = type_match.group(1)
        target_type = type_match.group(2)

        # Common type conversion suggestions
        conversions = {
            (
                "string",
                "number",
            ): "Fix type compatibility: Convert string to number using parseInt() or parseFloat()",
            (
                "number",
                "string",
            ): "Fix type compatibility: Convert number to string using .toString() or String()",
            (
                "null",
                "string",
            ): "Fix type compatibility: Add null check or use nullish coalescing (??) operator",
            (
                "undefined",
                "string",
            ): "Fix type compatibility: Add undefined check or provide default value",
            (
                "any",
                "specific_type",
            ): "Fix type compatibility: Use type assertion or improve type definitions",
        }

        conversion_key = (source_type, target_type)
        if conversion_key in conversions:
            return {
                "type": "suggestion",
                "description": conversions[conversion_key],
                "confidence": "high",
            }

        return {
            "type": "suggestion",
            "description": f"Type '{source_type}' is not assignable to type '{target_type}'. Use type assertion, type guards, or fix the type compatibility.",
        }

    def _fix_property_not_exist(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix TS2339: Property does not exist errors."""
        message = error_data.get("message", "")

        # Extract property and type from error message
        prop_match = re.search(
            r"Property '([^']+)' does not exist on type '([^']+)'", message
        )
        if not prop_match:
            return None

        property_name = prop_match.group(1)
        type_name = prop_match.group(2)

        return {
            "type": "suggestion",
            "description": f"Property '{property_name}' does not exist on type '{type_name}'. Options: 1) Check property spelling, 2) Extend the interface/type, 3) Use optional chaining (?.), 4) Use bracket notation for dynamic properties.",
        }

    def _fix_module_not_found(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix TS2307: Cannot find module errors."""
        message = error_data.get("message", "")

        # Extract module name from error message
        module_match = re.search(r"Cannot find module '([^']+)'", message)
        if not module_match:
            return None

        module_name = module_match.group(1)

        # Check if it's a relative import or npm package
        if module_name.startswith("."):
            return {
                "type": "suggestion",
                "description": f"Cannot find module '{module_name}'. Check if the file exists and the path is correct.",
            }
        else:
            return {
                "type": "suggestion",
                "description": f"Cannot find module '{module_name}'. Install the package: npm install {module_name}, and install types: npm install --save-dev @types/{module_name}",
            }

    def _fix_syntax_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix TypeScript syntax errors."""
        message = error_data.get("message", "")

        # Common syntax error patterns
        if "expected" in message.lower():
            expected_match = re.search(r"'([^']+)' expected", message)
            if expected_match:
                expected_token = expected_match.group(1)
                return {
                    "type": "suggestion",
                    "description": f"Add the missing '{expected_token}' token.",
                }

        return {
            "type": "suggestion",
            "description": "Fix TypeScript syntax error by checking for missing semicolons, brackets, or incorrect syntax.",
        }

    def _fix_config_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix TypeScript configuration errors."""
        return {
            "type": "suggestion",
            "description": "Fix TypeScript configuration in tsconfig.json. Check compiler options and their values.",
        }

    def _fix_unused_variable(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix unused variable warnings."""
        message = error_data.get("message", "")

        # Extract variable name
        var_match = re.search(
            r"'([^']+)' is declared but its value is never read", message
        )
        if var_match:
            var_name = var_match.group(1)
            return {
                "type": "suggestion",
                "description": f"Variable '{var_name}' is unused. Either remove it or prefix with underscore (_) to indicate intentional non-use.",
            }

        return {
            "type": "suggestion",
            "description": "Remove unused variables or prefix with underscore to indicate intentional non-use.",
        }

    def _template_based_patch(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Generate patch using templates."""
        root_cause = analysis.get("root_cause", "")

        # Map root causes to template names
        template_map = {
            "typescript_type_error": "type_error_fix",
            "typescript_compilation_error": "compilation_fix",
            "typescript_import_error": "import_fix",
        }

        template_name = template_map.get(root_cause)
        if template_name and template_name in self.templates:
            template = self.templates[template_name]

            # Simple template substitution
            return {
                "type": "template",
                "template": template,
                "description": f"Applied template fix for {root_cause}",
            }

        return None


class TypeScriptLanguagePlugin(LanguagePlugin):
    """
    Main TypeScript language plugin for Homeostasis.

    This plugin orchestrates TypeScript error analysis and patch generation,
    supporting TypeScript compilation, type checking, and runtime errors.
    """

    VERSION = "1.0.0"
    AUTHOR = "Homeostasis Team"

    def __init__(self):
        """Initialize the TypeScript language plugin."""
        self.language = "typescript"
        self.supported_extensions = {".ts", ".tsx", ".d.ts"}
        self.supported_frameworks = [
            "react",
            "vue",
            "angular",
            "svelte",
            "nextjs",
            "nuxtjs",
            "express",
            "nestjs",
            "fastify",
            "koa",
            "node",
            "deno",
        ]

        # Initialize components
        self.adapter = TypeScriptErrorAdapter()
        self.exception_handler = TypeScriptExceptionHandler()
        self.patch_generator = TypeScriptPatchGenerator()

        logger.info("TypeScript language plugin initialized")

    def get_language_id(self) -> str:
        """Get the unique identifier for this language."""
        return "typescript"

    def get_language_name(self) -> str:
        """Get the human-readable name of the language."""
        return "TypeScript"

    def get_language_version(self) -> str:
        """Get the version of the language supported by this plugin."""
        return "3.0+"

    def get_supported_frameworks(self) -> List[str]:
        """Get the list of frameworks supported by this language plugin."""
        return self.supported_frameworks

    def normalize_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize error data to the standard Homeostasis format.

        Args:
            error_data: Error data in the TypeScript-specific format

        Returns:
            Error data in the standard format
        """
        return self.adapter.to_standard_format(error_data)

    def denormalize_error(self, standard_error: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert standard format error data back to the TypeScript-specific format.

        Args:
            standard_error: Error data in the standard format

        Returns:
            Error data in the TypeScript-specific format
        """
        return self.adapter.from_standard_format(standard_error)

    def can_handle(self, error_data: Dict[str, Any]) -> bool:
        """
        Check if this plugin can handle the given error.

        Args:
            error_data: Error data to check

        Returns:
            True if this plugin can handle the error, False otherwise
        """
        # Check if language is explicitly set
        if error_data.get("language") == "typescript":
            return True

        # Check for TypeScript error codes
        error_type = error_data.get("error_type", "")
        error_code = error_data.get("error_code", "")
        message = error_data.get("message", "")

        if error_type.startswith("TS") or error_code.startswith("TS"):
            return True

        # Check for TypeScript-specific patterns in message
        ts_patterns = [
            r"TS\d+:",
            r"error TS\d+:",
            r"TypeScript compilation",
            r"tsc \(\d+,\d+\):",
            r"Type '[^']+' is not assignable to type",
            r"Cannot find name '[^']+'",
            r"Property '[^']+' does not exist on type",
            r"Cannot find module '[^']+'",
        ]

        for pattern in ts_patterns:
            if re.search(pattern, message):
                return True

        # Check file extensions in stack trace
        stack_str = str(error_data.get("stack_trace", ""))
        if re.search(r"\.(ts|tsx):", stack_str):
            return True

        # Check for TypeScript-specific runtime environment
        runtime = error_data.get("runtime", "").lower()
        if "typescript" in runtime or "tsc" in runtime:
            return True

        return False

    def analyze_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a TypeScript error.

        Args:
            error_data: TypeScript error data

        Returns:
            Analysis results
        """
        try:
            # Ensure error data is in standard format
            if not error_data.get("language"):
                standard_error = self.adapter.to_standard_format(error_data)
            else:
                standard_error = error_data

            # Check if it's a type error first (more specific)
            if self._is_type_error(standard_error):
                analysis = self.exception_handler.analyze_type_error(standard_error)

            # Check if it's a compilation error
            elif self._is_compilation_error(standard_error):
                analysis = self.exception_handler.analyze_compilation_error(
                    standard_error
                )

            # Default error analysis
            else:
                analysis = self.exception_handler.analyze_exception(standard_error)

            # Add plugin metadata
            analysis["plugin"] = "typescript"
            analysis["language"] = "typescript"
            analysis["plugin_version"] = self.VERSION

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing TypeScript error: {e}")
            return {
                "category": "typescript",
                "subcategory": "unknown",
                "confidence": "low",
                "suggested_fix": "Unable to analyze TypeScript error",
                "error": str(e),
                "plugin": "typescript",
            }

    def _is_compilation_error(self, error_data: Dict[str, Any]) -> bool:
        """Check if this is a TypeScript compilation error."""
        error_type = error_data.get("error_type", "")
        error_code = error_data.get("error_code", "")
        message = error_data.get("message", "")

        # TypeScript compilation errors typically have TS error codes
        if error_type.startswith("TS") or error_code.startswith("TS"):
            return True

        # Check for compilation-specific patterns
        compilation_patterns = [
            "error TS",
            "TypeScript compilation",
            "tsc (",
            "compiler option",
        ]

        return any(pattern in message for pattern in compilation_patterns)

    def _is_type_error(self, error_data: Dict[str, Any]) -> bool:
        """Check if this is a TypeScript type system error."""
        error_code = error_data.get("error_code", "")
        message = error_data.get("message", "")

        # Common type error codes
        type_error_codes = [
            "TS2304",
            "TS2322",
            "TS2339",
            "TS2307",
            "TS2345",
            "TS2540",
            "TS2571",
            "TS2564",
        ]

        if error_code in type_error_codes:
            return True

        # Check for type-specific patterns in message
        type_patterns = [
            "Type .* is not assignable",
            "Cannot find name",
            "Property .* does not exist",
            "Cannot find module",
        ]

        return any(re.search(pattern, message) for pattern in type_patterns)

    def generate_fix(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a fix for the TypeScript error.

        Args:
            error_data: The TypeScript error data
            analysis: Analysis results
            source_code: Source code where the error occurred

        Returns:
            Fix information or None if no fix can be generated
        """
        try:
            return self.patch_generator.generate_patch(
                error_data, analysis, source_code
            )
        except Exception as e:
            logger.error(f"Error generating TypeScript fix: {e}")
            return None

    def get_language_info(self) -> Dict[str, Any]:
        """
        Get information about this language plugin.

        Returns:
            Language plugin information
        """
        return {
            "language": self.language,
            "version": self.VERSION,
            "supported_extensions": list(self.supported_extensions),
            "supported_frameworks": list(self.supported_frameworks),
            "features": [
                "TypeScript compilation error handling",
                "Type system error detection and fixes",
                "Advanced TypeScript features support",
                "Generic type error resolution",
                "TSX/JSX error handling",
                "Module resolution error detection",
                "TypeScript configuration validation",
                "Transpilation error detection",
            ],
            "environments": ["node", "browser", "deno", "electron", "react-native"],
        }


# Register the plugin
register_plugin(TypeScriptLanguagePlugin())
