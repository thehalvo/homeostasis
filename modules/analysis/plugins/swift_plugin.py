"""
Swift Language Plugin for Homeostasis

This plugin enables Homeostasis to analyze and fix errors in Swift applications.
It provides comprehensive error handling for iOS, macOS, watchOS, and tvOS applications,
including support for modern Swift features, concurrency, SwiftUI, UIKit, and Core Data.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..language_adapters import SwiftErrorAdapter
from ..language_plugin_system import LanguagePlugin, register_plugin
from .swift_dependency_analyzer import SwiftDependencyAnalyzer

logger = logging.getLogger(__name__)


class SwiftExceptionHandler:
    """
    Handles Swift exceptions with a robust error detection and classification system.

    This class provides logic for categorizing Swift exceptions based on their type,
    message, and stack trace patterns. It supports iOS, macOS, watchOS, and tvOS platforms.
    """

    def __init__(self):
        """Initialize the Swift exception handler."""
        self.rule_categories = {
            "core": "Core Swift language exceptions",
            "memory": "Memory management and ARC exceptions",
            "concurrency": "Async/await and concurrency exceptions",
            "uikit": "UIKit framework exceptions",
            "swiftui": "SwiftUI framework exceptions",
            "core_data": "Core Data persistence exceptions",
            "networking": "Network and URLSession exceptions",
            "foundation": "Foundation framework exceptions",
            "runtime": "Swift runtime exceptions",
            "optionals": "Optional unwrapping exceptions",
            "collections": "Array, Dictionary, Set exceptions",
            "threading": "Threading and dispatch queue exceptions",
            "spm": "Swift Package Manager exceptions",
        }

        # Load rules from different categories
        self.rules = self._load_rules()

        # Pre-compile regex patterns for better performance
        self._compile_patterns()

    def _load_rules(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load Swift error rules from rule files."""
        rules = {}
        rules_dir = Path(__file__).parent.parent / "rules" / "swift"

        try:
            # Load common Swift rules
            common_rules_path = rules_dir / "swift_common_errors.json"
            if common_rules_path.exists():
                with open(common_rules_path, "r") as f:
                    common_data = json.load(f)
                    rules["common"] = common_data.get("rules", [])
                    logger.info(f"Loaded {len(rules['common'])} common Swift rules")

            # Load UIKit specific rules
            uikit_rules_path = rules_dir / "swift_uikit_errors.json"
            if uikit_rules_path.exists():
                with open(uikit_rules_path, "r") as f:
                    uikit_data = json.load(f)
                    rules["uikit"] = uikit_data.get("rules", [])
                    logger.info(f"Loaded {len(rules['uikit'])} UIKit rules")

            # Load SwiftUI specific rules
            swiftui_rules_path = rules_dir / "swift_swiftui_errors.json"
            if swiftui_rules_path.exists():
                with open(swiftui_rules_path, "r") as f:
                    swiftui_data = json.load(f)
                    rules["swiftui"] = swiftui_data.get("rules", [])
                    logger.info(f"Loaded {len(rules['swiftui'])} SwiftUI rules")

            # Load Core Data specific rules
            coredata_rules_path = rules_dir / "swift_coredata_errors.json"
            if coredata_rules_path.exists():
                with open(coredata_rules_path, "r") as f:
                    coredata_data = json.load(f)
                    rules["coredata"] = coredata_data.get("rules", [])
                    logger.info(f"Loaded {len(rules['coredata'])} Core Data rules")

            # Load concurrency specific rules
            concurrency_rules_path = rules_dir / "swift_concurrency_errors.json"
            if concurrency_rules_path.exists():
                with open(concurrency_rules_path, "r") as f:
                    concurrency_data = json.load(f)
                    rules["concurrency"] = concurrency_data.get("rules", [])
                    logger.info(f"Loaded {len(rules['concurrency'])} concurrency rules")

        except Exception as e:
            logger.error(f"Error loading Swift rules: {e}")
            rules = {
                "common": [],
                "uikit": [],
                "swiftui": [],
                "coredata": [],
                "concurrency": [],
            }

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
        Analyze a Swift exception and determine its type and potential fixes.

        Args:
            error_data: Swift error data in standard format

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
        runtime = error_data.get("runtime", "").lower()
        if "ios" in runtime:
            context_tags.add("ios")
        if "macos" in runtime:
            context_tags.add("macos")
        if "watchos" in runtime:
            context_tags.add("watchos")
        if "tvos" in runtime:
            context_tags.add("tvos")
        if error_data.get("framework"):
            context_tags.add(error_data["framework"].lower())

        if context_tags & rule_tags:
            base_confidence += 0.1

        return min(base_confidence, 1.0)

    def _generic_analysis(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """Provide generic analysis for unmatched errors."""
        error_type = error_data.get("error_type", "Error")
        message = error_data.get("message", "")

        # Basic categorization based on error type and message patterns
        if "fatal error" in message.lower():
            category = "fatal"
            severity = "high"
        elif "unexpectedly found nil" in message.lower():
            category = "optionals"
            severity = "high"
        elif "index out of range" in message.lower():
            category = "collections"
            severity = "medium"
        elif "exc_bad_access" in message.lower():
            category = "memory"
            severity = "high"
        elif "deadlock" in message.lower():
            category = "concurrency"
            severity = "high"
        else:
            category = "unknown"
            severity = "medium"

        return {
            "category": "swift",
            "subcategory": category,
            "confidence": "low",
            "suggested_fix": f"Review the {error_type} and check the surrounding Swift code logic",
            "root_cause": f"swift_{category}_error",
            "severity": severity,
            "rule_id": "swift_generic_handler",
            "tags": ["swift", "generic"],
        }

    def analyze_fatal_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze Swift fatal errors.

        Args:
            error_data: Error data from Swift runtime

        Returns:
            Analysis results with fatal error specific fixes
        """
        message = error_data.get("message", "")

        # Common fatal error patterns
        if (
            "unexpectedly found nil while unwrapping an optional value"
            in message.lower()
        ):
            return {
                "category": "fatal",
                "subcategory": "force_unwrap",
                "confidence": "high",
                "suggested_fix": "Replace force unwrapping (!) with safe unwrapping (if let, guard let, or ??)",
                "root_cause": "swift_force_unwrap_nil",
                "severity": "high",
                "tags": ["swift", "optionals", "fatal"],
                "fix_commands": [
                    "Use if let binding for safe unwrapping",
                    "Use guard let for early exit patterns",
                    "Use nil coalescing operator (??) for default values",
                ],
            }

        if "index out of range" in message.lower():
            return {
                "category": "fatal",
                "subcategory": "array_bounds",
                "confidence": "high",
                "suggested_fix": "Check array bounds before accessing elements",
                "root_cause": "swift_array_index_out_of_bounds",
                "severity": "high",
                "tags": ["swift", "collections", "fatal"],
                "fix_commands": [
                    "Check array.count before accessing indices",
                    "Use array.indices.contains(index) for bounds checking",
                    "Use safe array access with indices validation",
                ],
            }

        if "attempting to load an asset" in message.lower():
            return {
                "category": "fatal",
                "subcategory": "resource_loading",
                "confidence": "high",
                "suggested_fix": "Verify resource exists in app bundle and check spelling",
                "root_cause": "swift_missing_resource",
                "severity": "medium",
                "tags": ["swift", "resources", "bundle"],
            }

        # Generic fatal error
        return {
            "category": "fatal",
            "subcategory": "unknown",
            "confidence": "medium",
            "suggested_fix": "Review fatal error message and check for nil unwrapping or bounds checking",
            "root_cause": "swift_fatal_error",
            "severity": "high",
            "tags": ["swift", "fatal"],
        }


class SwiftPatchGenerator:
    """
    Generates patches for Swift errors based on analysis results.

    This class creates code fixes for common Swift errors using templates
    and heuristics specific to Swift patterns and best practices.
    """

    def __init__(self):
        """Initialize the Swift patch generator."""
        self.template_dir = (
            Path(__file__).parent.parent / "patch_generation" / "templates"
        )
        self.swift_template_dir = self.template_dir / "swift"

        # Ensure template directory exists
        self.swift_template_dir.mkdir(parents=True, exist_ok=True)

        # Load patch templates
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict[str, str]:
        """Load Swift patch templates."""
        templates = {}

        if not self.swift_template_dir.exists():
            logger.warning(
                f"Swift templates directory not found: {self.swift_template_dir}"
            )
            return templates

        for template_file in self.swift_template_dir.glob("*.swift.template"):
            try:
                with open(template_file, "r") as f:
                    template_name = template_file.stem.replace(".swift", "")
                    templates[template_name] = f.read()
                    logger.debug(f"Loaded template: {template_name}")
            except Exception as e:
                logger.error(f"Error loading template {template_file}: {e}")

        return templates

    def generate_patch(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a patch for the Swift error.

        Args:
            error_data: The Swift error data
            analysis: Analysis results from SwiftExceptionHandler
            source_code: The source code where the error occurred

        Returns:
            Patch information or None if no patch can be generated
        """
        root_cause = analysis.get("root_cause", "")

        # Map root causes to patch strategies
        patch_strategies = {
            "swift_force_unwrap_nil": self._fix_force_unwrap,
            "swift_array_index_out_of_bounds": self._fix_array_bounds,
            "swift_weak_reference_cycle": self._fix_retain_cycle,
            "swift_main_thread_violation": self._fix_main_thread,
            "swift_core_data_context_error": self._fix_core_data_context,
            "swift_async_await_error": self._fix_async_await,
            "swift_swiftui_state_error": self._fix_swiftui_state,
        }

        strategy = patch_strategies.get(root_cause)
        if strategy:
            try:
                return strategy(error_data, analysis, source_code)
            except Exception as e:
                logger.error(f"Error generating patch for {root_cause}: {e}")

        # Try to use templates if no specific strategy matches
        return self._template_based_patch(error_data, analysis, source_code)

    def _fix_force_unwrap(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix force unwrapping of nil optionals."""
        stack_trace = error_data.get("stack_trace", [])

        # Extract line information
        line_info = self._extract_line_info(stack_trace)
        if not line_info:
            return None

        lines = source_code.split("\n")
        if line_info["line"] > len(lines):
            return None

        problem_line = lines[line_info["line"] - 1]

        # Look for force unwrapping patterns
        force_unwrap_pattern = r"(\w+)!"
        matches = re.finditer(force_unwrap_pattern, problem_line)

        fixed_line = problem_line
        for match in matches:
            var_name = match.group(1)
            # For simple cases, use nil coalescing
            if "return" in problem_line or "=" in problem_line:
                fixed_line = fixed_line.replace(
                    f"{var_name}!", f"{var_name} ?? defaultValue"
                )
            else:
                # Use if let for more complex cases
                fixed_line = f"if let {var_name} = {var_name} {{\n    {problem_line.replace(f'{var_name}!', var_name)}\n}}"

        return {
            "type": "line_replacement",
            "file": line_info.get("file", ""),
            "line": line_info["line"],
            "original": problem_line.strip(),
            "replacement": fixed_line.strip(),
            "description": "Replaced force unwrapping with safe optional handling",
        }

    def _fix_array_bounds(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix array index out of bounds errors."""
        stack_trace = error_data.get("stack_trace", [])
        line_info = self._extract_line_info(stack_trace)

        if not line_info:
            return None

        lines = source_code.split("\n")
        if line_info["line"] > len(lines):
            return None

        problem_line = lines[line_info["line"] - 1]

        # Look for array access patterns
        array_access_pattern = r"(\w+)\[(\w+|\d+)\]"
        match = re.search(array_access_pattern, problem_line)

        if match:
            array_name = match.group(1)
            index_expr = match.group(2)

            # Generate safe access
            safe_access = f"{array_name}.indices.contains({index_expr}) ? {array_name}[{index_expr}] : nil"
            fixed_line = problem_line.replace(
                f"{array_name}[{index_expr}]", safe_access
            )

            return {
                "type": "line_replacement",
                "file": line_info.get("file", ""),
                "line": line_info["line"],
                "original": problem_line.strip(),
                "replacement": fixed_line.strip(),
                "description": f"Added bounds checking for array access on '{array_name}'",
            }

        return None

    def _fix_retain_cycle(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix retain cycle issues."""
        return {
            "type": "suggestion",
            "description": "Potential retain cycle detected. Use [weak self] or [unowned self] in closures to break reference cycles.",
        }

    def _fix_main_thread(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix main thread checker violations."""
        return {
            "type": "suggestion",
            "description": "UI updates must be performed on the main thread. Use DispatchQueue.main.async { } or @MainActor for UI operations.",
        }

    def _fix_core_data_context(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix Core Data context errors."""
        return {
            "type": "suggestion",
            "description": "Core Data context error. Ensure context.save() is called on the correct queue and handle errors properly.",
        }

    def _fix_async_await(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix async/await concurrency errors."""
        return {
            "type": "suggestion",
            "description": "Async/await error. Ensure async functions are called from async context and handle potential throwing errors with try/catch.",
        }

    def _fix_swiftui_state(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix SwiftUI state management errors."""
        return {
            "type": "suggestion",
            "description": "SwiftUI state error. Ensure @State, @Binding, and @ObservableObject are used correctly and updates happen on main thread.",
        }

    def _template_based_patch(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Generate patch using templates."""
        root_cause = analysis.get("root_cause", "")

        # Map root causes to template names
        template_map = {
            "swift_force_unwrap_nil": "safe_optional_unwrapping",
            "swift_array_index_out_of_bounds": "safe_array_access",
            "swift_weak_reference_cycle": "weak_reference_fix",
            "swift_main_thread_violation": "main_thread_dispatch",
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

    def _extract_line_info(self, stack_trace: List) -> Optional[Dict[str, Any]]:
        """Extract file and line information from stack trace."""
        if not stack_trace:
            return None

        # Handle different stack trace formats
        first_frame = stack_trace[0]

        # If it's a dictionary with file/line info, return it directly
        if (
            isinstance(first_frame, dict)
            and "file" in first_frame
            and "line" in first_frame
        ):
            return {
                "file": first_frame.get("file", ""),
                "line": first_frame.get("line", 0),
                "column": first_frame.get("column", 0),
            }

        # Otherwise convert to string and parse
        first_frame = str(first_frame)

        # Common patterns for extracting line info
        patterns = [
            r"([^:]+):(\d+):(\d+)",  # File:line:column format
            r"(\w+\.swift):(\d+)",  # Swift file:line format
            r"([^:]+) line (\d+)",  # Alternative format
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


class SwiftLanguagePlugin(LanguagePlugin):
    """
    Main Swift language plugin for Homeostasis.

    This plugin orchestrates Swift error analysis and patch generation,
    supporting iOS, macOS, watchOS, and tvOS platforms.
    """

    VERSION = "1.0.0"
    AUTHOR = "Homeostasis Team"

    def __init__(self):
        """Initialize the Swift language plugin."""
        self.language = "swift"
        self.supported_extensions = {".swift"}
        self.supported_frameworks = [
            "uikit",
            "swiftui",
            "foundation",
            "core_data",
            "core_animation",
            "avfoundation",
            "mapkit",
            "core_location",
            "healthkit",
            "homekit",
            "cloudkit",
            "gameplaykit",
            "scenekit",
            "metal",
            "core_ml",
        ]

        # Initialize components
        self.adapter = SwiftErrorAdapter()
        self.exception_handler = SwiftExceptionHandler()
        self.patch_generator = SwiftPatchGenerator()
        self.dependency_analyzer = SwiftDependencyAnalyzer()

        logger.info("Swift language plugin initialized")

    def get_language_id(self) -> str:
        """Get the unique identifier for this language."""
        return "swift"

    def get_language_name(self) -> str:
        """Get the human-readable name of the language."""
        return "Swift"

    def get_language_version(self) -> str:
        """Get the version of the language supported by this plugin."""
        return "5.0+"

    def get_supported_frameworks(self) -> List[str]:
        """Get the list of frameworks supported by this language plugin."""
        return self.supported_frameworks

    def can_handle(self, error_data: Dict[str, Any]) -> bool:
        """
        Check if this plugin can handle the given error.

        Args:
            error_data: Error data to check

        Returns:
            True if this plugin can handle the error, False otherwise
        """
        # Check if language is explicitly set
        if error_data.get("language") == "swift":
            return True

        # Check runtime environment
        runtime = error_data.get("runtime", "").lower()
        if any(
            platform in runtime
            for platform in ["ios", "macos", "watchos", "tvos", "swift"]
        ):
            return True

        # Check error patterns specific to Swift
        message = error_data.get("message", "")
        stack_trace = str(error_data.get("stack_trace", ""))

        swift_patterns = [
            r"fatal error:",
            r"unexpectedly found nil while unwrapping",
            r"index out of range",
            r"EXC_BAD_ACCESS",
            r"Thread \d+: signal SIGABRT",
            r"\.swift:\d+",
            r"SwiftUI",
            r"UIKit",
            r"Foundation",
            r"Core Data",
        ]

        for pattern in swift_patterns:
            if re.search(pattern, message + stack_trace, re.IGNORECASE):
                return True

        # Check file extensions in stack trace
        if re.search(r"\.swift:", stack_trace):
            return True

        return False

    def _is_dependency_error(self, error_data: Dict[str, Any]) -> bool:
        """Check if this is a dependency-related error."""
        message = error_data.get("message", "").lower()

        dependency_patterns = [
            "no such module",
            "package.swift",
            "swift package",
            "dependency",
            "could not build",
            "missing package",
            "version conflict",
        ]

        return any(pattern in message for pattern in dependency_patterns)

    def analyze_dependencies(self, project_path: str) -> Dict[str, Any]:
        """
        Analyze project dependencies.

        Args:
            project_path: Path to the Swift project root

        Returns:
            Dependency analysis results
        """
        try:
            return self.dependency_analyzer.analyze_project_dependencies(project_path)
        except Exception as e:
            logger.error(f"Error analyzing Swift dependencies: {e}")
            return {
                "error": str(e),
                "suggestions": ["Check project structure and Package.swift file"],
            }

    def analyze_dependency_error(
        self, error_data: Dict[str, Any], project_path: str
    ) -> Dict[str, Any]:
        """
        Analyze a dependency-related error.

        Args:
            error_data: Error data from Swift compiler
            project_path: Path to the project root

        Returns:
            Analysis results with fix suggestions
        """
        try:
            return self.dependency_analyzer.analyze_dependency_error(
                error_data, project_path
            )
        except Exception as e:
            logger.error(f"Error analyzing Swift dependency error: {e}")
            return {
                "category": "dependency",
                "subcategory": "unknown",
                "confidence": "low",
                "suggested_fix": "Check dependency configuration",
                "error": str(e),
            }

    def analyze_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a Swift error.

        Args:
            error_data: Swift error data

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

            # Check for fatal errors first
            elif "fatal error" in standard_error.get("message", "").lower():
                analysis = self.exception_handler.analyze_fatal_error(standard_error)
            else:
                analysis = self.exception_handler.analyze_exception(standard_error)

            # Add plugin metadata
            analysis["plugin"] = "swift"
            analysis["language"] = "swift"
            analysis["plugin_version"] = self.VERSION

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing Swift error: {e}")
            return {
                "category": "swift",
                "subcategory": "unknown",
                "confidence": "low",
                "suggested_fix": "Unable to analyze error",
                "error": str(e),
                "plugin": "swift",
            }

    def generate_fix(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a fix for the Swift error.

        Args:
            error_data: The Swift error data
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
            logger.error(f"Error generating Swift fix: {e}")
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
                "iOS application error handling",
                "macOS application error handling",
                "watchOS application error handling",
                "tvOS application error handling",
                "SwiftUI error detection",
                "UIKit error detection",
                "Core Data error handling",
                "Async/await concurrency support",
                "Memory management error detection",
                "Optional unwrapping safety",
                "Array bounds checking",
            ],
            "platforms": ["ios", "macos", "watchos", "tvos"],
        }

    def normalize_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize error data to the standard Homeostasis format.

        Args:
            error_data: Language-specific error data

        Returns:
            Standardized error format
        """
        return {
            "language": self.get_language_id(),
            "type": error_data.get("type", "unknown"),
            "message": error_data.get("message", ""),
            "file": error_data.get("file", ""),
            "line": error_data.get("line", 0),
            "column": error_data.get("column", 0),
            "severity": error_data.get("severity", "error"),
            "context": error_data.get("context", {}),
            "raw_data": error_data,
        }

    def denormalize_error(self, standard_error: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert standard format error data back to the language-specific format.

        Args:
            standard_error: Standardized error data

        Returns:
            Language-specific error format
        """
        return {
            "type": standard_error.get("type", "unknown"),
            "message": standard_error.get("message", ""),
            "file": standard_error.get("file", ""),
            "line": standard_error.get("line", 0),
            "column": standard_error.get("column", 0),
            "severity": standard_error.get("severity", "error"),
            "context": standard_error.get("context", {}),
            "language_specific": standard_error.get("raw_data", {}),
        }


# Register the plugin
register_plugin(SwiftLanguagePlugin())
