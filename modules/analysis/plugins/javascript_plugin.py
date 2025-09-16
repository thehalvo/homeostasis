"""
JavaScript Language Plugin for Homeostasis

This plugin enables Homeostasis to analyze and fix errors in JavaScript applications.
It provides comprehensive error handling for both browser and Node.js environments,
including support for modern JavaScript features, async/await patterns, and popular
frameworks and libraries.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..language_adapters import JavaScriptErrorAdapter
from ..language_plugin_system import LanguagePlugin, register_plugin
from .javascript_dependency_analyzer import JavaScriptDependencyAnalyzer

logger = logging.getLogger(__name__)


class JavaScriptExceptionHandler:
    """
    Handles JavaScript exceptions with a robust error detection and classification system.

    This class provides logic for categorizing JavaScript exceptions based on their type,
    message, and stack trace patterns. It supports both browser and Node.js environments.
    """

    def __init__(self):
        """Initialize the JavaScript exception handler."""
        self.rule_categories = {
            "core": "Core JavaScript exceptions",
            "async": "Asynchronous operation exceptions",
            "dom": "DOM and browser API exceptions",
            "network": "Network and HTTP exceptions",
            "module": "Module loading and import exceptions",
            "nodejs": "Node.js specific exceptions",
            "framework": "Framework-specific exceptions",
            "bundler": "Build tool and bundler exceptions",
            "transpilation": "Transpilation and compilation exceptions",
            "memory": "Memory and performance exceptions",
            "security": "Security-related exceptions",
            "json": "JSON parsing exceptions",
        }

        # Load rules from different categories
        self.rules = self._load_rules()

        # Pre-compile regex patterns for better performance
        self._compile_patterns()

    def _load_rules(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load JavaScript error rules from rule files."""
        rules = {}
        rules_dir = Path(__file__).parent.parent / "rules" / "javascript"

        try:
            # Load common JavaScript rules
            common_rules_path = rules_dir / "js_common_errors.json"
            if common_rules_path.exists():
                with open(common_rules_path, "r") as f:
                    common_data = json.load(f)
                    rules["common"] = common_data.get("rules", [])
                    logger.info(
                        f"Loaded {len(rules['common'])} common JavaScript rules"
                    )

            # Load Node.js specific rules
            nodejs_rules_path = rules_dir / "nodejs_errors.json"
            if nodejs_rules_path.exists():
                with open(nodejs_rules_path, "r") as f:
                    nodejs_data = json.load(f)
                    rules["nodejs"] = nodejs_data.get("rules", [])
                    logger.info(f"Loaded {len(rules['nodejs'])} Node.js rules")

        except Exception as e:
            logger.error(f"Error loading JavaScript rules: {e}")
            rules = {"common": [], "nodejs": []}

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
        Analyze a JavaScript exception and determine its type and potential fixes.

        Args:
            error_data: JavaScript error data in standard format

        Returns:
            Analysis results with categorization and fix suggestions
        """
        error_type = error_data.get("error_type", "Error")
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
            return {
                "category": best_match.get("category", "unknown"),
                "subcategory": best_match.get("type", "unknown"),
                "confidence": best_match.get("confidence", "medium"),
                "suggested_fix": best_match.get("suggestion", ""),
                "root_cause": best_match.get("root_cause", ""),
                "severity": best_match.get("severity", "medium"),
                "rule_id": best_match.get("id", ""),
                "tags": best_match.get("tags", []),
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

        # Boost confidence for exact error type matches
        rule_type = rule.get("type", "").lower()
        error_type = error_data.get("error_type", "").lower()
        if rule_type and rule_type in error_type:
            base_confidence += 0.3

        # Boost confidence based on rule reliability
        reliability = rule.get("reliability", "medium")
        reliability_boost = {"high": 0.2, "medium": 0.1, "low": 0.0}
        base_confidence += reliability_boost.get(reliability, 0.0)

        # Boost confidence for rules with specific tags that match context
        rule_tags = set(rule.get("tags", []))
        context_tags = set()

        # Infer context from error data
        if "nodejs" in error_data.get("runtime", "").lower():
            context_tags.add("nodejs")
        if "browser" in error_data.get("runtime", "").lower():
            context_tags.add("browser")
        if error_data.get("framework"):
            context_tags.add(error_data["framework"].lower())

        if context_tags & rule_tags:
            base_confidence += 0.1

        return min(base_confidence, 1.0)

    def _generic_analysis(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """Provide generic analysis for unmatched errors."""
        error_type = error_data.get("error_type", "Error")

        # Basic categorization based on error type
        category_map = {
            "TypeError": "type",
            "ReferenceError": "reference",
            "SyntaxError": "syntax",
            "RangeError": "range",
            "EvalError": "eval",
            "URIError": "uri",
            "AggregateError": "aggregate",
        }

        category = category_map.get(error_type, "unknown")

        return {
            "category": "javascript",
            "subcategory": category,
            "confidence": "low",
            "suggested_fix": f"Review the {error_type} and check the surrounding code logic",
            "root_cause": f"js_{category}_error",
            "severity": "medium",
            "rule_id": "js_generic_handler",
            "tags": ["javascript", "generic"],
        }

    def analyze_transpilation_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze transpilation errors from Babel, TypeScript, etc.

        Args:
            error_data: Error data from transpilation process

        Returns:
            Analysis results with transpilation-specific fixes
        """
        message = error_data.get("message", "")
        error_type = error_data.get("error_type", "")
        stack_trace = str(error_data.get("stack_trace", ""))

        # Babel errors
        if "babel" in message.lower() or "babel" in stack_trace.lower():
            return self._analyze_babel_error(message, error_data)

        # TypeScript errors
        if (
            "typescript" in message.lower()
            or "tsc" in message.lower()
            or error_type.startswith("TS")
        ):
            return self._analyze_typescript_error(message, error_data)

        # Webpack errors
        if "webpack" in message.lower() or "webpack" in stack_trace.lower():
            return self._analyze_webpack_error(message, error_data)

        # Rollup errors
        if "rollup" in message.lower() or "rollup" in stack_trace.lower():
            return self._analyze_rollup_error(message, error_data)

        # Vite errors
        if "vite" in message.lower() or "vite" in stack_trace.lower():
            return self._analyze_vite_error(message, error_data)

        # Parcel errors
        if "parcel" in message.lower() or "parcel" in stack_trace.lower():
            return self._analyze_parcel_error(message, error_data)

        # Generic transpilation error
        return {
            "category": "transpilation",
            "subcategory": "unknown",
            "confidence": "medium",
            "suggested_fix": "Check transpilation configuration and source code syntax",
            "root_cause": "js_transpilation_error",
            "severity": "medium",
            "tags": ["javascript", "transpilation"],
        }

    def _analyze_babel_error(
        self, message: str, error_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze Babel transpilation errors."""
        # Common Babel error patterns
        if "Unexpected token" in message:
            return {
                "category": "transpilation",
                "subcategory": "babel_syntax",
                "confidence": "high",
                "suggested_fix": "Check for unsupported syntax or missing Babel presets/plugins",
                "root_cause": "babel_unexpected_token",
                "severity": "high",
                "tags": ["javascript", "babel", "syntax"],
                "fix_commands": [
                    "Check .babelrc or babel.config.js configuration",
                    "Ensure required presets are installed (@babel/preset-env, @babel/preset-react, etc.)",
                    "Verify syntax is supported by your Babel configuration",
                ],
            }

        if "Cannot find module" in message and "@babel" in message:
            return {
                "category": "transpilation",
                "subcategory": "babel_dependency",
                "confidence": "high",
                "suggested_fix": "Install missing Babel dependency",
                "root_cause": "babel_missing_dependency",
                "severity": "high",
                "tags": ["javascript", "babel", "dependency"],
                "fix_commands": [
                    "npm install --save-dev @babel/core @babel/cli",
                    "Install missing Babel presets/plugins",
                    "Check package.json devDependencies",
                ],
            }

        return {
            "category": "transpilation",
            "subcategory": "babel_general",
            "confidence": "medium",
            "suggested_fix": "Check Babel configuration and dependencies",
            "root_cause": "babel_error",
            "severity": "medium",
            "tags": ["javascript", "babel"],
        }

    def _analyze_typescript_error(
        self, message: str, error_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze TypeScript compilation errors."""
        # Extract TypeScript error code if present
        ts_error_match = re.search(r"TS(\d+):", message)
        ts_error_code = ts_error_match.group(1) if ts_error_match else None

        # Common TypeScript errors
        type_error_patterns = {
            "2304": {
                "pattern": "Cannot find name",
                "fix": "Check if the variable/type is declared and imported correctly",
                "commands": [
                    "Verify imports",
                    "Check type definitions",
                    "Install @types packages if needed",
                ],
            },
            "2322": {
                "pattern": "Type .* is not assignable to type",
                "fix": "Fix type mismatch by adjusting types or using type assertions",
                "commands": [
                    "Check type compatibility",
                    "Use type assertions if necessary",
                    "Update type definitions",
                ],
            },
            "2339": {
                "pattern": "Property .* does not exist on type",
                "fix": "Check if property exists or extend the type definition",
                "commands": [
                    "Verify property names",
                    "Extend interfaces",
                    "Use optional chaining",
                ],
            },
            "2307": {
                "pattern": "Cannot find module",
                "fix": "Install missing module or add type definitions",
                "commands": [
                    "npm install <module>",
                    "npm install --save-dev @types/<module>",
                    "Check import paths",
                ],
            },
        }

        if ts_error_code and ts_error_code in type_error_patterns:
            error_info = type_error_patterns[ts_error_code]
            return {
                "category": "transpilation",
                "subcategory": "typescript_type",
                "confidence": "high",
                "suggested_fix": error_info["fix"],
                "root_cause": f"typescript_error_{ts_error_code}",
                "severity": "high",
                "tags": ["javascript", "typescript", "types"],
                "error_code": f"TS{ts_error_code}",
                "fix_commands": error_info["commands"],
            }

        # Generic TypeScript error
        return {
            "category": "transpilation",
            "subcategory": "typescript_general",
            "confidence": "medium",
            "suggested_fix": "Fix TypeScript compilation errors",
            "root_cause": "typescript_error",
            "severity": "medium",
            "tags": ["javascript", "typescript"],
            "fix_commands": [
                "Check tsconfig.json configuration",
                "Verify type definitions",
                "Run tsc --noEmit to check types only",
            ],
        }

    def _analyze_webpack_error(
        self, message: str, error_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze Webpack bundling errors."""
        if "Module not found" in message:
            return {
                "category": "transpilation",
                "subcategory": "webpack_module",
                "confidence": "high",
                "suggested_fix": "Fix module resolution in webpack configuration",
                "root_cause": "webpack_module_not_found",
                "severity": "high",
                "tags": ["javascript", "webpack", "modules"],
                "fix_commands": [
                    "Check webpack resolve configuration",
                    "Verify import paths and aliases",
                    "Install missing dependencies",
                ],
            }

        if "Cannot resolve" in message:
            return {
                "category": "transpilation",
                "subcategory": "webpack_resolve",
                "confidence": "high",
                "suggested_fix": "Fix module resolution configuration",
                "root_cause": "webpack_resolve_error",
                "severity": "high",
                "tags": ["javascript", "webpack", "resolution"],
            }

        return {
            "category": "transpilation",
            "subcategory": "webpack_general",
            "confidence": "medium",
            "suggested_fix": "Check webpack configuration and build process",
            "root_cause": "webpack_error",
            "severity": "medium",
            "tags": ["javascript", "webpack"],
        }

    def _analyze_rollup_error(
        self, message: str, error_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze Rollup bundling errors."""
        return {
            "category": "transpilation",
            "subcategory": "rollup",
            "confidence": "medium",
            "suggested_fix": "Check Rollup configuration and plugins",
            "root_cause": "rollup_error",
            "severity": "medium",
            "tags": ["javascript", "rollup"],
        }

    def _analyze_vite_error(
        self, message: str, error_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze Vite build errors."""
        return {
            "category": "transpilation",
            "subcategory": "vite",
            "confidence": "medium",
            "suggested_fix": "Check Vite configuration and dependencies",
            "root_cause": "vite_error",
            "severity": "medium",
            "tags": ["javascript", "vite"],
        }

    def _analyze_parcel_error(
        self, message: str, error_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze Parcel bundling errors."""
        return {
            "category": "transpilation",
            "subcategory": "parcel",
            "confidence": "medium",
            "suggested_fix": "Check Parcel configuration and source files",
            "root_cause": "parcel_error",
            "severity": "medium",
            "tags": ["javascript", "parcel"],
        }


class JavaScriptPatchGenerator:
    """
    Generates patches for JavaScript errors based on analysis results.

    This class creates code fixes for common JavaScript errors using templates
    and heuristics specific to JavaScript patterns and best practices.
    """

    def __init__(self):
        """Initialize the JavaScript patch generator."""
        self.template_dir = (
            Path(__file__).parent.parent / "patch_generation" / "templates"
        )
        self.js_template_dir = self.template_dir / "javascript"

        # Ensure template directory exists
        self.js_template_dir.mkdir(parents=True, exist_ok=True)

        # Load patch templates
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict[str, str]:
        """Load JavaScript patch templates."""
        templates: Dict[str, str] = {}

        if not self.js_template_dir.exists():
            logger.warning(
                f"JavaScript templates directory not found: {self.js_template_dir}"
            )
            return templates

        for template_file in self.js_template_dir.glob("*.js.template"):
            try:
                with open(template_file, "r") as f:
                    template_name = template_file.stem.replace(".js", "")
                    templates[template_name] = f.read()
                    logger.debug(f"Loaded template: {template_name}")
            except Exception as e:
                logger.error(f"Error loading template {template_file}: {e}")

        return templates

    def generate_patch(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a patch for the JavaScript error.

        Args:
            error_data: The JavaScript error data
            analysis: Analysis results from JavaScriptExceptionHandler
            source_code: The source code where the error occurred

        Returns:
            Patch information or None if no patch can be generated
        """
        root_cause = analysis.get("root_cause", "")

        # Map root causes to patch strategies
        patch_strategies = {
            "js_property_access_on_undefined": self._fix_property_access,
            "js_not_a_function": self._fix_not_a_function,
            "js_undefined_reference": self._fix_undefined_reference,
            "js_syntax_error": self._fix_syntax_error,
            "js_unhandled_promise_rejection": self._fix_promise_rejection,
            "js_memory_limit_exceeded": self._fix_memory_issue,
            "js_async_operation_failure": self._fix_async_operation,
        }

        strategy = patch_strategies.get(root_cause)
        if strategy:
            try:
                return strategy(error_data, analysis, source_code)
            except Exception as e:
                logger.error(f"Error generating patch for {root_cause}: {e}")

        # Try to use templates if no specific strategy matches
        return self._template_based_patch(error_data, analysis, source_code)

    def _fix_property_access(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix property access on undefined/null."""
        message = error_data.get("message", "")

        # Extract property name from error message
        prop_match = re.search(r"Cannot (?:read|set) property '([^']+)' of", message)
        if not prop_match:
            return None

        property_name = prop_match.group(1)

        # Find the problematic line in source code
        stack_trace = error_data.get("stack_trace", [])
        if not stack_trace:
            return None

        # Extract line number from stack trace
        line_info = self._extract_line_info(stack_trace)
        if not line_info:
            return None

        lines = source_code.split("\n")
        if line_info["line"] > len(lines):
            return None

        problem_line = lines[line_info["line"] - 1]

        # Generate fix using optional chaining or null check
        if "?." in source_code:  # Modern JS environment supports optional chaining
            fixed_line = re.sub(
                rf"(\w+)\.{re.escape(property_name)}",
                rf"\1?.{property_name}",
                problem_line,
            )
        else:
            # Use traditional null check
            obj_match = re.search(rf"(\w+)\.{re.escape(property_name)}", problem_line)
            if obj_match:
                obj_name = obj_match.group(1)
                fixed_line = problem_line.replace(
                    f"{obj_name}.{property_name}",
                    f"({obj_name} && {obj_name}.{property_name})",
                )
            else:
                return None

        return {
            "type": "line_replacement",
            "file": line_info.get("file", ""),
            "line": line_info["line"],
            "original": problem_line.strip(),
            "replacement": fixed_line.strip(),
            "description": f"Added null check for property access on '{property_name}'",
        }

    def _fix_not_a_function(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix 'not a function' errors."""
        message = error_data.get("message", "")

        # Extract function name from error message
        func_match = re.search(r"([^\s]+) is not a function", message)
        if not func_match:
            return None

        function_ref = func_match.group(1)

        # Find the problematic line
        stack_trace = error_data.get("stack_trace", [])
        line_info = self._extract_line_info(stack_trace)
        if not line_info:
            return None

        lines = source_code.split("\n")
        if line_info["line"] > len(lines):
            return None

        problem_line = lines[line_info["line"] - 1]

        # Generate function check
        fixed_line = re.sub(
            rf"{re.escape(function_ref)}\s*\(",
            f"(typeof {function_ref} === 'function' ? {function_ref} : () => {{}})(",
            problem_line,
        )

        return {
            "type": "line_replacement",
            "file": line_info.get("file", ""),
            "line": line_info["line"],
            "original": problem_line.strip(),
            "replacement": fixed_line.strip(),
            "description": f"Added function type check for '{function_ref}'",
        }

    def _fix_undefined_reference(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix undefined reference errors."""
        message = error_data.get("message", "")

        # Extract variable name from error message
        var_match = re.search(r"([^\s]+) is not defined", message)
        if not var_match:
            return None

        var_name = var_match.group(1)

        # Common fixes based on variable name patterns
        if var_name in ["require", "module", "exports", "__dirname", "__filename"]:
            # Node.js environment issue
            return {
                "type": "suggestion",
                "description": f"'{var_name}' is not defined. This appears to be a Node.js environment issue. Consider using ES6 imports or ensure you're running in a Node.js environment.",
            }

        if var_name in ["window", "document", "console", "localStorage"]:
            # Browser environment issue
            return {
                "type": "suggestion",
                "description": f"'{var_name}' is not defined. This appears to be a browser environment issue. Ensure the code is running in a browser or add appropriate polyfills.",
            }

        # Generic undefined variable fix
        return {
            "type": "suggestion",
            "description": f"Variable '{var_name}' is not defined. Check spelling, ensure it's declared before use, or add the missing import/declaration.",
        }

    def _fix_promise_rejection(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix unhandled promise rejection errors."""
        stack_trace = error_data.get("stack_trace", [])
        line_info = self._extract_line_info(stack_trace)

        if not line_info:
            return {
                "type": "suggestion",
                "description": "Add .catch() handlers to Promises or use try/catch blocks with async/await",
            }

        lines = source_code.split("\n")
        if line_info["line"] > len(lines):
            return None

        problem_line = lines[line_info["line"] - 1]

        # Check if it's a Promise chain
        if ".then(" in problem_line and ".catch(" not in problem_line:
            fixed_line = (
                problem_line.rstrip()
                + ".catch(error => console.error('Promise rejected:', error));"
            )

            return {
                "type": "line_replacement",
                "file": line_info.get("file", ""),
                "line": line_info["line"],
                "original": problem_line.strip(),
                "replacement": fixed_line.strip(),
                "description": "Added .catch() handler for Promise",
            }

        return {
            "type": "suggestion",
            "description": "Add proper error handling for Promise rejections using .catch() or try/catch with async/await",
        }

    def _fix_memory_issue(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix memory-related issues."""
        return {
            "type": "suggestion",
            "description": "JavaScript heap out of memory. Consider: 1) Checking for memory leaks, 2) Processing data in chunks, 3) Increasing Node.js memory limit with --max-old-space-size flag",
        }

    def _fix_async_operation(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix async operation failures."""
        return {
            "type": "suggestion",
            "description": "Async operation failed. Consider adding retry logic, timeout handling, and proper error boundaries for network operations",
        }

    def _fix_syntax_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix syntax errors."""
        message = error_data.get("message", "")

        # Common syntax error patterns
        if "missing )" in message.lower():
            return {
                "type": "suggestion",
                "description": "Missing closing parenthesis. Check for unmatched parentheses in function calls or expressions.",
            }
        elif "unexpected token" in message.lower():
            return {
                "type": "suggestion",
                "description": "Unexpected token found. Check for missing commas, semicolons, or incorrect syntax.",
            }
        elif "missing ;" in message.lower():
            return {
                "type": "suggestion",
                "description": "Missing semicolon. Add semicolon at the end of the statement.",
            }

        return {
            "type": "suggestion",
            "description": "Syntax error detected. Review the code for proper JavaScript syntax.",
        }

    def _template_based_patch(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Generate patch using templates."""
        root_cause = analysis.get("root_cause", "")

        # Map root causes to template names
        template_map = {
            "js_property_access_on_undefined": "property_access_fix",
            "js_not_a_function": "function_check",
            "js_undefined_reference": "variable_check",
            "js_unhandled_promise_rejection": "promise_catch",
        }

        template_name = template_map.get(root_cause)
        if template_name and template_name in self.templates:
            template = self.templates[template_name]

            # Simple template substitution
            # In a real implementation, you'd use a proper template engine
            return {
                "type": "template",
                "template": template,
                "description": f"Applied template fix for {root_cause}",
            }

        return None

    def _extract_line_info(self, stack_trace: List) -> Optional[Dict[str, Any]]:
        """Extract file and line information from stack trace."""
        if not stack_trace:
            return None

        # Handle structured stack trace (dict format)
        if isinstance(stack_trace[0], dict):
            frame = stack_trace[0]
            if "line" in frame:
                return {
                    "file": frame.get("file", "unknown"),
                    "line": frame.get("line", 0),
                    "column": frame.get("column", 0),
                }

        # Look for line number in first frame (string format)
        first_frame = (
            stack_trace[0] if isinstance(stack_trace[0], str) else str(stack_trace[0])
        )

        # Common patterns for extracting line info
        patterns = [
            r"at .* \(([^:]+):(\d+):(\d+)\)",  # Node.js/Chrome format
            r"([^:]+):(\d+):(\d+)",  # Simple format
            r"@([^:]+):(\d+):(\d+)",  # Firefox format
        ]

        for pattern in patterns:
            match = re.search(pattern, first_frame)
            if match:
                return {
                    "file": match.group(1),
                    "line": int(match.group(2)),
                    "column": int(match.group(3)) if len(match.groups()) >= 3 else 0,
                }

        return None


class JavaScriptLanguagePlugin(LanguagePlugin):
    """
    Main JavaScript language plugin for Homeostasis.

    This plugin orchestrates JavaScript error analysis and patch generation,
    supporting both browser and Node.js environments.
    """

    VERSION = "1.0.0"
    AUTHOR = "Homeostasis Team"

    def __init__(self):
        """Initialize the JavaScript language plugin."""
        self.language = "javascript"
        self.supported_extensions = {".js", ".mjs", ".cjs", ".jsx"}
        self.supported_frameworks = [
            "express",
            "koa",
            "fastify",
            "nest",
            "next",
            "nuxt",
            "react",
            "vue",
            "angular",
            "svelte",
            "ember",
            "electron",
            "cordova",
            "ionic",
        ]

        # Initialize components
        self.adapter = JavaScriptErrorAdapter()
        self.exception_handler = JavaScriptExceptionHandler()
        self.patch_generator = JavaScriptPatchGenerator()
        self.dependency_analyzer = JavaScriptDependencyAnalyzer()

        logger.info("JavaScript language plugin initialized")

    def get_language_id(self) -> str:
        """Get the unique identifier for this language."""
        return "javascript"

    def get_language_name(self) -> str:
        """Get the human-readable name of the language."""
        return "JavaScript"

    def get_language_version(self) -> str:
        """Get the version of the language supported by this plugin."""
        return "ES5+"

    def get_supported_frameworks(self) -> List[str]:
        """Get the list of frameworks supported by this language plugin."""
        return self.supported_frameworks

    def normalize_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize error data to the standard Homeostasis format.

        Args:
            error_data: Error data in the JavaScript-specific format

        Returns:
            Error data in the standard format
        """
        return self.adapter.to_standard_format(error_data)

    def denormalize_error(self, standard_error: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert standard format error data back to the JavaScript-specific format.

        Args:
            standard_error: Error data in the standard format

        Returns:
            Error data in the JavaScript-specific format
        """
        return self.adapter.from_standard_format(standard_error)

    def generate_fix(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Generate a fix for an error based on the analysis.

        Flexible signature to handle both base class (analysis, context)
        and potentially other calling patterns.

        Returns:
            Generated fix data
        """
        # Handle different calling patterns
        if len(args) >= 2:
            analysis = args[0]
            context = args[1]
            source_code = args[2] if len(args) > 2 else None
        else:
            analysis = kwargs.get("analysis", {})
            context = kwargs.get("context", {})
            source_code = kwargs.get("source_code", None)

        if context is None:
            context = {}
        error_data = context.get("error_data", {})
        if source_code is None:
            source_code = context.get("source_code", "")

        try:
            fix = self.patch_generator.generate_patch(error_data, analysis, source_code)
        except Exception as e:
            logger.error(f"Error generating patch: {e}")
            fix = None

        if fix:
            return fix

        # Always return a suggestion if no specific fix is available
        result = {
            "type": "suggestion",
            "description": analysis.get("suggested_fix", "Use optional chaining"),
            "confidence": analysis.get("confidence", "low"),
        }
        return result

    def can_handle(self, error_data: Dict[str, Any]) -> bool:
        """
        Check if this plugin can handle the given error.

        Args:
            error_data: Error data to check

        Returns:
            True if this plugin can handle the error, False otherwise
        """
        # Check if language is explicitly set
        if error_data.get("language") == "javascript":
            return True

        # Check error type patterns
        error_type = error_data.get("error_type", "")
        js_error_types = {
            "TypeError",
            "ReferenceError",
            "SyntaxError",
            "RangeError",
            "EvalError",
            "URIError",
            "AggregateError",
            "Error",
        }

        if error_type in js_error_types:
            return True

        # Check JavaScript-specific patterns in message or stack trace
        message = error_data.get("message", "")
        stack_trace = str(error_data.get("stack_trace", ""))
        combined = message + stack_trace

        js_patterns = [
            r"at\s+\w+\s+\([^)]+\.js:\d+:\d+\)",  # Node.js stack trace
            r"\w+@[^:]+\.js:\d+:\d+",  # Firefox stack trace
            r"Cannot read property .* of undefined",  # Common JS error
            r"is not a function",  # Common JS error
            r"is not defined",  # Common JS error
            r"\.(js|mjs|cjs|jsx):",  # File extensions
        ]

        for pattern in js_patterns:
            if re.search(pattern, combined):
                return True

        # Check runtime environment
        runtime = error_data.get("runtime", "").lower()
        if "node" in runtime or "javascript" in runtime or "v8" in runtime:
            return True

        return False

    def _is_dependency_error(self, error_data: Dict[str, Any]) -> bool:
        """Check if this is a dependency-related error."""
        message = error_data.get("message", "").lower()

        dependency_patterns = [
            "cannot find module",
            "module not found",
            "eaddrinuse",
            "econnrefused",
            "enoent",
            "eacces",
            "peer dep",
            "eresolve",
        ]

        return any(pattern in message for pattern in dependency_patterns)

    def _is_transpilation_error(self, error_data: Dict[str, Any]) -> bool:
        """Check if this is a transpilation-related error."""
        message = error_data.get("message", "").lower()
        stack_trace = str(error_data.get("stack_trace", "")).lower()
        error_type = error_data.get("error_type", "")

        transpilation_patterns = [
            "babel",
            "typescript",
            "tsc",
            "webpack",
            "rollup",
            "vite",
            "parcel",
        ]

        return (
            any(pattern in message for pattern in transpilation_patterns)
            or any(pattern in stack_trace for pattern in transpilation_patterns)
            or error_type.startswith("TS")
        )

    def _is_bundler_error(self, error_data: Dict[str, Any]) -> bool:
        """Check if this is a bundler-related error."""
        message = error_data.get("message", "").lower()
        stack_trace = str(error_data.get("stack_trace", "")).lower()

        bundler_patterns = ["webpack", "rollup", "vite", "parcel", "esbuild", "swc"]

        return any(pattern in message for pattern in bundler_patterns) or any(
            pattern in stack_trace for pattern in bundler_patterns
        )

    def _analyze_bundler_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze bundler-specific errors."""
        # Use the existing transpilation analysis for bundler errors
        # since bundlers often involve transpilation
        return self.exception_handler.analyze_transpilation_error(error_data)

    def analyze_dependencies(self, project_path: str) -> Dict[str, Any]:
        """
        Analyze project dependencies.

        Args:
            project_path: Path to the JavaScript project root

        Returns:
            Dependency analysis results
        """
        try:
            return self.dependency_analyzer.analyze_project_dependencies(project_path)
        except Exception as e:
            logger.error(f"Error analyzing JavaScript dependencies: {e}")
            return {
                "error": str(e),
                "suggestions": ["Check project structure and package.json file"],
            }

    def analyze_dependency_error(
        self, error_data: Dict[str, Any], project_path: str
    ) -> Dict[str, Any]:
        """
        Analyze a dependency-related error.

        Args:
            error_data: Error data from JavaScript runtime
            project_path: Path to the project root

        Returns:
            Analysis results with fix suggestions
        """
        try:
            return self.dependency_analyzer.analyze_dependency_error(
                error_data, project_path
            )
        except Exception as e:
            logger.error(f"Error analyzing JavaScript dependency error: {e}")
            return {
                "category": "dependency",
                "subcategory": "unknown",
                "confidence": "low",
                "suggested_fix": "Check dependency configuration",
                "error": str(e),
            }

    def analyze_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a JavaScript error.

        Args:
            error_data: JavaScript error data

        Returns:
            Analysis results
        """
        try:
            # Ensure error data is in standard format
            if not error_data.get("language"):
                standard_error = self.adapter.to_standard_format(error_data)
            else:
                standard_error = error_data

            # Check if it's a dependency-related error
            if self._is_dependency_error(standard_error):
                project_path = standard_error.get("context", {}).get(
                    "project_path", "."
                )
                analysis = self.dependency_analyzer.analyze_dependency_error(
                    standard_error, project_path
                )

            # Check if it's a transpilation error
            elif self._is_transpilation_error(standard_error):
                analysis = self.exception_handler.analyze_transpilation_error(
                    standard_error
                )

            # Check if it's a bundler error
            elif self._is_bundler_error(standard_error):
                analysis = self._analyze_bundler_error(standard_error)

            # Default error analysis
            else:
                analysis = self.exception_handler.analyze_exception(standard_error)

            # Add plugin metadata
            analysis["plugin"] = "javascript"
            analysis["language"] = "javascript"
            analysis["plugin_version"] = "1.0.0"

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing JavaScript error: {e}")
            return {
                "category": "javascript",
                "subcategory": "unknown",
                "confidence": "low",
                "suggested_fix": "Unable to analyze error",
                "error": str(e),
                "plugin": "javascript",
            }

    def get_language_info(self) -> Dict[str, Any]:
        """
        Get information about this language plugin.

        Returns:
            Language plugin information
        """
        return {
            "language": self.language,
            "version": "1.0.0",
            "supported_extensions": list(self.supported_extensions),
            "supported_frameworks": list(self.supported_frameworks),
            "features": [
                "Browser JavaScript error handling",
                "Node.js error handling",
                "Async/await pattern support",
                "Promise rejection handling",
                "Module loading error detection",
                "Transpilation error detection",
                "Memory issue detection",
                "Framework-specific error handling",
            ],
            "environments": ["browser", "nodejs", "electron", "worker"],
        }


# Register the plugin
register_plugin(JavaScriptLanguagePlugin())
