"""
Web Components Language Plugin for Homeostasis

This plugin enables Homeostasis to analyze and fix errors in Web Components applications.
It provides comprehensive error handling for Custom Elements lifecycle, Shadow DOM encapsulation,
HTML Templates, and interoperability with popular frameworks like Lit and Stencil.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..language_adapters import JavaScriptErrorAdapter
from ..language_plugin_system import LanguagePlugin, register_plugin

logger = logging.getLogger(__name__)


class WebComponentExceptionHandler:
    """
    Handles Web Component exceptions with specialized error detection for Custom Elements,
    Shadow DOM, and other Web Component standards.

    This class provides pattern matching and analysis for Web Component-specific errors,
    categorizing them and suggesting appropriate fixes.
    """

    def __init__(self):
        """Initialize the Web Component exception handler."""
        self.rule_categories = {
            "lifecycle": "Custom Elements lifecycle errors",
            "shadow_dom": "Shadow DOM encapsulation issues",
            "templates": "HTML Template element errors",
            "interop": "Framework interoperability issues",
            "lit": "Lit library errors",
            "stencil": "Stencil framework errors",
            "registry": "Custom Elements registry errors",
            "css": "Shadow DOM CSS and styling issues",
            "events": "Custom element event handling issues",
            "slots": "Slot content and distribution errors",
        }

        # Load rules from different categories
        self.rules = self._load_rules()

        # Pre-compile regex patterns for better performance
        self._compile_patterns()

    def _load_rules(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load Web Component error rules from rule files."""
        rules = {}
        # In a real implementation, these rules would be loaded from JSON files
        # For now, we'll define them inline as a placeholder

        rules["lifecycle"] = [
            {
                "id": "ce_constructor_super",
                "pattern": "Failed to construct '\\w+': 1st argument is not an object, or super\\(\\) not called",
                "category": "lifecycle",
                "type": "constructor_error",
                "severity": "high",
                "suggestion": "Ensure the constructor calls super() before any other statements",
                "root_cause": "missing_super_call_in_constructor",
                "reliability": "high",
                "tags": ["webcomponents", "custom-elements", "constructor"],
            },
            {
                "id": "ce_connected_callback_error",
                "pattern": "uncaught (exception|error) in connectedCallback",
                "category": "lifecycle",
                "type": "callback_error",
                "severity": "high",
                "suggestion": "Check for errors in the connectedCallback method",
                "root_cause": "error_in_connected_callback",
                "reliability": "high",
                "tags": ["webcomponents", "custom-elements", "lifecycle"],
            },
            {
                "id": "ce_disconnected_callback_error",
                "pattern": "uncaught (exception|error) in disconnectedCallback",
                "category": "lifecycle",
                "type": "callback_error",
                "severity": "medium",
                "suggestion": "Check for errors in the disconnectedCallback method",
                "root_cause": "error_in_disconnected_callback",
                "reliability": "high",
                "tags": ["webcomponents", "custom-elements", "lifecycle"],
            },
            {
                "id": "ce_attribute_changed_callback_error",
                "pattern": "uncaught (exception|error) in attributeChangedCallback",
                "category": "lifecycle",
                "type": "callback_error",
                "severity": "medium",
                "suggestion": "Check for errors in the attributeChangedCallback method",
                "root_cause": "error_in_attribute_changed_callback",
                "reliability": "high",
                "tags": ["webcomponents", "custom-elements", "lifecycle"],
            },
            {
                "id": "ce_adopted_callback_error",
                "pattern": "uncaught (exception|error) in adoptedCallback",
                "category": "lifecycle",
                "type": "callback_error",
                "severity": "low",
                "suggestion": "Check for errors in the adoptedCallback method",
                "root_cause": "error_in_adopted_callback",
                "reliability": "high",
                "tags": ["webcomponents", "custom-elements", "lifecycle"],
            },
            {
                "id": "ce_observed_attributes_error",
                "pattern": "observedAttributes (must|should) return an array",
                "category": "lifecycle",
                "type": "observed_attributes_error",
                "severity": "medium",
                "suggestion": "Ensure observedAttributes returns an array of attribute names",
                "root_cause": "invalid_observed_attributes",
                "reliability": "high",
                "tags": ["webcomponents", "custom-elements", "attributes"],
            },
            {
                "id": "ce_define_before_customElements",
                "pattern": "(customElements is (undefined|not defined)|Cannot read property 'define' of (undefined|null))",
                "category": "registry",
                "type": "registry_error",
                "severity": "high",
                "suggestion": "Ensure customElements is available before defining components",
                "root_cause": "custom_elements_registry_unavailable",
                "reliability": "high",
                "tags": ["webcomponents", "custom-elements", "registry"],
            },
            {
                "id": "ce_already_defined",
                "pattern": "Failed to execute 'define' on 'CustomElementRegistry': the name '.+' has already been used with this registry",
                "category": "registry",
                "type": "registry_error",
                "severity": "medium",
                "suggestion": "Element with this name is already defined. Use a different name or check for duplicate registrations",
                "root_cause": "duplicate_element_definition",
                "reliability": "high",
                "tags": ["webcomponents", "custom-elements", "registry"],
            },
        ]

        rules["interop"] = [
            {
                "id": "react_event_handling",
                "pattern": "React event handler not firing on custom element",
                "category": "interop",
                "type": "react_integration",
                "severity": "medium",
                "suggestion": "Use synthetic event handlers or dispatch custom events that React can handle",
                "root_cause": "react_event_handler_issue",
                "reliability": "high",
                "tags": ["webcomponents", "react", "interop"],
            }
        ]

        rules["templates"] = [
            {
                "id": "template_not_cloned",
                "pattern": "Template content never cloned properly",
                "category": "templates",
                "type": "template_clone",
                "severity": "medium",
                "suggestion": "Use document.importNode(template.content, true) to properly clone template content",
                "root_cause": "template_content_not_cloned",
                "reliability": "high",
                "tags": ["webcomponents", "templates", "template"],
            }
        ]

        rules["shadow_dom"] = [
            {
                "id": "sd_closed_mode_access",
                "pattern": "Cannot read properties of null \\(reading 'querySelector'\\)",
                "category": "shadow_dom",
                "type": "closed_shadow_root",
                "severity": "medium",
                "suggestion": "Can't access elements inside a closed shadow root. Consider using open mode or storing references",
                "root_cause": "closed_shadow_root_access_attempt",
                "reliability": "medium",
                "tags": ["webcomponents", "shadow-dom", "encapsulation"],
            },
            {
                "id": "sd_style_leakage",
                "pattern": "CSS styles (leaking|affecting) (outside|external) elements",
                "category": "shadow_dom",
                "type": "style_encapsulation",
                "severity": "medium",
                "suggestion": "Use :host selector or create styles inside shadow DOM to prevent leakage",
                "root_cause": "css_style_leakage",
                "reliability": "medium",
                "tags": ["webcomponents", "shadow-dom", "css"],
            },
            {
                "id": "sd_event_retargeting",
                "pattern": "event.target (is|shows|points to) (shadow|internal) element",
                "category": "shadow_dom",
                "type": "event_retargeting",
                "severity": "low",
                "suggestion": "Use composedPath() to access original target in shadow DOM",
                "root_cause": "event_retargeting_confusion",
                "reliability": "medium",
                "tags": ["webcomponents", "shadow-dom", "events"],
            },
            {
                "id": "sd_slot_content",
                "pattern": "(slotted content not|slot content missing|slot not filled)",
                "category": "slots",
                "type": "slot_content",
                "severity": "medium",
                "suggestion": "Check that elements have the correct slot attribute or that default slot is available",
                "root_cause": "slot_content_distribution_issue",
                "reliability": "medium",
                "tags": ["webcomponents", "shadow-dom", "slots"],
            },
            {
                "id": "sd_part_styling",
                "pattern": "::part\\([^)]+\\) (not working|has no effect)",
                "category": "shadow_dom",
                "type": "part_styling",
                "severity": "low",
                "suggestion": "Ensure the part attribute is correctly set on the shadow DOM element",
                "root_cause": "part_attribute_styling_issue",
                "reliability": "medium",
                "tags": ["webcomponents", "shadow-dom", "css", "theming"],
            },
        ]

        # More rules would be loaded from actual JSON files in a real implementation
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
        Analyze a Web Component exception and determine its type and potential fixes.

        Args:
            error_data: Error data in standard format

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
        if "webcomponents" in error_data.get("tags", []):
            context_tags.add("webcomponents")
        if "custom-elements" in error_data.get("tags", []):
            context_tags.add("custom-elements")
        if "shadow-dom" in error_data.get("tags", []):
            context_tags.add("shadow-dom")
        if "lit" in error_data.get("framework", "").lower():
            context_tags.add("lit")
        if "stencil" in error_data.get("framework", "").lower():
            context_tags.add("stencil")

        if context_tags & rule_tags:
            base_confidence += 0.1

        return min(base_confidence, 1.0)

    def _generic_analysis(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """Provide generic analysis for unmatched errors."""
        error_type = error_data.get("error_type", "Error")
        message = error_data.get("message", "").lower()

        # Check for common Web Component related terms in the error
        if any(
            term in message
            for term in [
                "custom element",
                "web component",
                "shadow",
                "slot",
                "template",
            ]
        ):
            if "constructor" in message:
                return {
                    "category": "lifecycle",
                    "subcategory": "constructor",
                    "confidence": "medium",
                    "suggested_fix": "Check Custom Element constructor implementation. Ensure super() is called first.",
                    "root_cause": "custom_element_constructor_issue",
                    "severity": "high",
                    "tags": ["webcomponents", "custom-elements", "lifecycle"],
                }
            elif any(term in message for term in ["shadow", "root", "dom"]):
                return {
                    "category": "shadow_dom",
                    "subcategory": "general",
                    "confidence": "medium",
                    "suggested_fix": "Check Shadow DOM implementation and encapsulation boundaries.",
                    "root_cause": "shadow_dom_issue",
                    "severity": "medium",
                    "tags": ["webcomponents", "shadow-dom"],
                }
            elif any(term in message for term in ["slot", "assign", "slotted"]):
                return {
                    "category": "slots",
                    "subcategory": "general",
                    "confidence": "medium",
                    "suggested_fix": "Check slot assignments and content distribution.",
                    "root_cause": "slot_content_issue",
                    "severity": "medium",
                    "tags": ["webcomponents", "shadow-dom", "slots"],
                }
            else:
                return {
                    "category": "webcomponents",
                    "subcategory": "general",
                    "confidence": "low",
                    "suggested_fix": "Check Web Component implementation for standards compliance.",
                    "root_cause": "web_component_general_issue",
                    "severity": "medium",
                    "tags": ["webcomponents"],
                }

        # If no Web Component specific terms found, fall back to JavaScript general error
        category_map = {
            "TypeError": "type",
            "ReferenceError": "reference",
            "SyntaxError": "syntax",
            "RangeError": "range",
        }

        category = category_map.get(error_type, "unknown")

        return {
            "category": "javascript",
            "subcategory": category,
            "confidence": "low",
            "suggested_fix": f"Review the {error_type} in the context of Web Components",
            "root_cause": f"js_{category}_error",
            "severity": "medium",
            "rule_id": "web_component_generic_handler",
            "tags": ["javascript", "generic", "webcomponents"],
        }


class WebComponentPatchGenerator:
    """
    Generates patches for Web Component errors based on analysis results.

    This class creates code fixes for common Web Component errors using templates
    and heuristics specific to Custom Elements, Shadow DOM, and related standards.
    """

    def __init__(self):
        """Initialize the Web Component patch generator."""
        self.template_dir = (
            Path(__file__).parent.parent / "patch_generation" / "templates"
        )
        self.web_component_template_dir = self.template_dir / "web_components"

        # Ensure template directory exists
        self.web_component_template_dir.mkdir(parents=True, exist_ok=True)

        # Load patch templates
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict[str, str]:
        """Load Web Component patch templates."""
        templates = {}

        if not self.web_component_template_dir.exists():
            logger.warning(
                f"Web Component templates directory not found: {self.web_component_template_dir}"
            )
            return templates

        for template_file in self.web_component_template_dir.glob("*.js.template"):
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
        Generate a patch for the Web Component error.

        Args:
            error_data: The error data
            analysis: Analysis results from WebComponentExceptionHandler
            source_code: The source code where the error occurred

        Returns:
            Patch information or None if no patch can be generated
        """
        root_cause = analysis.get("root_cause", "")
        category = analysis.get("category", "")

        # Map root causes to patch strategies
        if category == "lifecycle":
            return self._fix_lifecycle_error(
                root_cause, error_data, analysis, source_code
            )
        elif category == "shadow_dom":
            return self._fix_shadow_dom_error(
                root_cause, error_data, analysis, source_code
            )
        elif category == "slots":
            return self._fix_slot_error(root_cause, error_data, analysis, source_code)
        elif category == "registry":
            return self._fix_registry_error(
                root_cause, error_data, analysis, source_code
            )

        # Try to use templates if no specific strategy matches
        return self._template_based_patch(error_data, analysis, source_code)

    def _fix_lifecycle_error(
        self,
        root_cause: str,
        error_data: Dict[str, Any],
        analysis: Dict[str, Any],
        source_code: str,
    ) -> Optional[Dict[str, Any]]:
        """Fix Custom Element lifecycle errors."""
        if root_cause == "missing_super_call_in_constructor":
            # Find the problematic constructor
            stack_trace = error_data.get("stack_trace", [])
            line_info = self._extract_line_info(stack_trace)

            if not line_info:
                return {
                    "type": "suggestion",
                    "description": "Add super() call at the beginning of your custom element constructor",
                    "suggestion": "Add super() call at the beginning of your custom element constructor",
                }

            lines = source_code.split("\n")
            if line_info["line"] > len(lines):
                return None

            # Find constructor start
            constructor_line = -1
            for i in range(line_info["line"], 0, -1):
                if "constructor" in lines[i - 1] and "{" in lines[i - 1]:
                    constructor_line = i
                    break

            if constructor_line == -1:
                return {
                    "type": "suggestion",
                    "description": "Make sure to call super() as the first statement in your custom element constructor",
                    "suggestion": "Make sure to call super() as the first statement in your custom element constructor",
                }

            # Find indentation
            indentation = ""
            for char in lines[constructor_line - 1]:
                if char.isspace():
                    indentation += char
                else:
                    break

            # Add additional indentation for constructor body
            body_indent = indentation + "  "

            # Check if there's a line after the constructor opening
            if constructor_line < len(lines) and "{" in lines[constructor_line - 1]:
                # Add super() call after the constructor line
                fixed_line = f"{body_indent}super();\n{lines[constructor_line]}"

                return {
                    "type": "line_replacement",
                    "file": line_info.get("file", ""),
                    "line": constructor_line,
                    "original": lines[constructor_line],
                    "replacement": fixed_line,
                    "description": "Added missing super() call to custom element constructor",
                }

        elif root_cause == "invalid_observed_attributes":
            # Find the problematic observedAttributes
            stack_trace = error_data.get("stack_trace", [])
            line_info = self._extract_line_info(stack_trace)

            if not line_info:
                return {
                    "type": "suggestion",
                    "description": "Ensure observedAttributes is implemented as a static getter that returns an array",
                    "suggestion": "Ensure observedAttributes is implemented as a static getter that returns an array",
                }

            return {
                "type": "suggestion",
                "description": "Implement observedAttributes as: static get observedAttributes() { return ['attribute-name']; }",
                "suggestion": "Implement observedAttributes as: static get observedAttributes() { return ['attribute-name']; }",
            }

        # Generic lifecycle error fix
        return {
            "type": "suggestion",
            "description": analysis.get(
                "suggested_fix",
                "Check lifecycle method implementation in your custom element",
            ),
            "suggestion": analysis.get(
                "suggested_fix",
                "Check lifecycle method implementation in your custom element",
            ),
        }

    def _fix_shadow_dom_error(
        self,
        root_cause: str,
        error_data: Dict[str, Any],
        analysis: Dict[str, Any],
        source_code: str,
    ) -> Optional[Dict[str, Any]]:
        """Fix Shadow DOM errors."""
        if root_cause == "closed_shadow_root_access_attempt":
            return {
                "type": "suggestion",
                "description": "Consider using 'open' mode for shadow root or store references to shadow DOM elements",
            }

        elif root_cause == "css_style_leakage":
            return {
                "type": "suggestion",
                "description": "Use :host selector for styles that should apply to the component, and ensure styles are added inside shadow DOM",
            }

        elif root_cause == "event_retargeting_confusion":
            return {
                "type": "suggestion",
                "description": "Use event.composedPath() to access the original target in shadow DOM",
            }

        # Generic shadow DOM error fix
        return {
            "type": "suggestion",
            "description": analysis.get(
                "suggested_fix",
                "Check shadow DOM implementation for proper encapsulation",
            ),
        }

    def _fix_slot_error(
        self,
        root_cause: str,
        error_data: Dict[str, Any],
        analysis: Dict[str, Any],
        source_code: str,
    ) -> Optional[Dict[str, Any]]:
        """Fix slot content distribution errors."""
        if root_cause == "slot_content_distribution_issue":
            return {
                "type": "suggestion",
                "description": "Check that elements have the correct slot attribute, or add a default (unnamed) slot if needed",
            }

        # Generic slot error fix
        return {
            "type": "suggestion",
            "description": analysis.get(
                "suggested_fix", "Check slot implementation and content distribution"
            ),
        }

    def _fix_registry_error(
        self,
        root_cause: str,
        error_data: Dict[str, Any],
        analysis: Dict[str, Any],
        source_code: str,
    ) -> Optional[Dict[str, Any]]:
        """Fix Custom Elements registry errors."""
        if root_cause == "custom_elements_registry_unavailable":
            return {
                "type": "suggestion",
                "description": "Ensure code runs after the DOM is fully loaded and customElements is available",
            }

        elif root_cause == "duplicate_element_definition":
            return {
                "type": "suggestion",
                "description": "Element is already defined. Check for duplicate calls to customElements.define() or use a different name",
            }

        # Generic registry error fix
        return {
            "type": "suggestion",
            "description": analysis.get(
                "suggested_fix", "Check Custom Elements registry usage"
            ),
        }

    def _template_based_patch(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Generate patch using templates."""
        root_cause = analysis.get("root_cause", "")

        # Map root causes to template names
        template_map = {
            "missing_super_call_in_constructor": "constructor_super",
            "invalid_observed_attributes": "observed_attributes",
            "closed_shadow_root_access_attempt": "shadow_dom_access",
            "slot_content_distribution_issue": "slot_content",
            "template_content_not_cloned": "template_optimization",
        }

        template_name = template_map.get(root_cause)
        if template_name and template_name in self.templates:
            template = self.templates[template_name]

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

        # Look for line number in first frame
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


class WebComponentLanguagePlugin(LanguagePlugin):
    """
    Main Web Components language plugin for Homeostasis.

    This plugin orchestrates Web Component error analysis and patch generation,
    supporting Custom Elements, Shadow DOM, HTML Templates, and integrations with
    popular libraries like Lit and Stencil.
    """

    VERSION = "1.0.0"
    AUTHOR = "Homeostasis Team"

    def __init__(self):
        """Initialize the Web Component language plugin."""
        self.language = "webcomponents"
        self.supported_extensions = {".js", ".mjs", ".html", ".jsx", ".ts", ".tsx"}
        self.supported_frameworks = [
            "lit",
            "stencil",
            "fast",
            "hybrids",
            "slim",
            "lwc",
            "haunted",
            "svelte",
            "skate",
            "polymer",
        ]

        # Initialize components
        self.adapter = JavaScriptErrorAdapter()  # Reuse JavaScript adapter as base
        self.exception_handler = WebComponentExceptionHandler()
        self.patch_generator = WebComponentPatchGenerator()

        logger.info("Web Component language plugin initialized")

    def get_language_id(self) -> str:
        """Get the unique identifier for this language."""
        return "webcomponents"

    def get_language_name(self) -> str:
        """Get the human-readable name of the language."""
        return "Web Components"

    def get_language_version(self) -> str:
        """Get the version of the language supported by this plugin."""
        return "v1 Spec+"

    def get_supported_frameworks(self) -> List[str]:
        """Get the list of frameworks supported by this language plugin."""
        return self.supported_frameworks

    def normalize_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize error data to the standard Homeostasis format.

        Args:
            error_data: Error data in the Web Component-specific format

        Returns:
            Error data in the standard format
        """
        # Extend JavaScript adapter normalization with Web Component specific data
        normalized = self.adapter.to_standard_format(error_data)

        # Add Web Component specific tags if detected
        if self._is_web_component_error(error_data):
            if "tags" not in normalized:
                normalized["tags"] = []
            normalized["tags"].append("webcomponents")

            # Add more specific tags based on error content
            message = str(error_data.get("message", "")).lower()
            if "custom element" in message or "customelements" in message:
                normalized["tags"].append("custom-elements")
            if "shadow" in message and ("dom" in message or "root" in message):
                normalized["tags"].append("shadow-dom")
            if "slot" in message or "slotted" in message:
                normalized["tags"].append("slots")
            if "template" in message:
                normalized["tags"].append("templates")

        return normalized

    def denormalize_error(self, standard_error: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert standard format error data back to the Web Component-specific format.

        Args:
            standard_error: Error data in the standard format

        Returns:
            Error data in the Web Component-specific format
        """
        # Web Components are JavaScript-based, so reuse JavaScript adapter
        return self.adapter.from_standard_format(standard_error)

    def analyze_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a Web Component error.

        Args:
            error_data: Web Component error data

        Returns:
            Analysis results
        """
        try:
            # Ensure error data is in standard format
            if not error_data.get("language"):
                standard_error = self.normalize_error(error_data)
            else:
                standard_error = error_data

            # Analyze using the Web Component exception handler
            analysis = self.exception_handler.analyze_exception(standard_error)

            # Add plugin metadata
            analysis["plugin"] = "webcomponents"
            analysis["language"] = "webcomponents"
            analysis["plugin_version"] = self.VERSION

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing Web Component error: {e}")
            return {
                "category": "webcomponents",
                "subcategory": "unknown",
                "confidence": "low",
                "suggested_fix": "Unable to analyze error",
                "error": str(e),
                "plugin": "webcomponents",
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

        try:
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
        except Exception as e:
            logger.error(f"Error generating Web Component fix: {e}")
            return {
                "type": "suggestion",
                "description": "Error generating fix: " + str(e),
                "confidence": "low",
            }

    def can_handle(self, error_data: Dict[str, Any]) -> bool:
        """
        Check if this plugin can handle the given error.

        Args:
            error_data: Error data to check

        Returns:
            True if this plugin can handle the error, False otherwise
        """
        # Check if language is explicitly set
        if error_data.get("language") == "webcomponents":
            return True

        # Check if the error is Web Component related
        if self._is_web_component_error(error_data):
            return True

        return False

    def _is_web_component_error(self, error_data: Dict[str, Any]) -> bool:
        """
        Check if this is a Web Component-related error.

        Args:
            error_data: Error data to check

        Returns:
            True if this is a Web Component error, False otherwise
        """
        # Check message content
        message = str(error_data.get("message", "")).lower()
        stack_trace = str(error_data.get("stack_trace", "")).lower()
        combined_text = message + "\n" + stack_trace

        web_component_patterns = [
            r"custom[ -]elements?",
            r"customelements\.define",
            r"customelementregistry",
            r"shadow[ -]dom",
            r"shadowroot",
            r"attachshadow",
            r"slot",
            r"slotchange",
            r"\btemplate\b",
            r"connectedcallback",
            r"disconnectedcallback",
            r"attributechangedcallback",
            r"adoptedcallback",
            r"observedattributes",
            r"extends htmlelement",
            r"lit-element",
            r"lithtml",
            r"litelement",
            r"stencil",
            r"\bslot\b",
            r"::part",
            r"::slotted",
        ]

        for pattern in web_component_patterns:
            if re.search(pattern, combined_text, re.IGNORECASE):
                return True

        # Check the framework field
        framework = str(error_data.get("framework", "")).lower()
        web_component_frameworks = [
            "lit",
            "stencil",
            "polymer",
            "fast",
            "slim",
            "skate",
            "hybrids",
            "haunted",
            "lwc",
        ]

        if any(fw in framework for fw in web_component_frameworks):
            return True

        return False

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
                "Custom Elements lifecycle error detection",
                "Shadow DOM encapsulation issue resolution",
                "HTML Template element optimization",
                "Web Component interoperability healing",
                "Lit and Stencil framework integration",
                "Web Component best practices enforcement",
            ],
            "environments": ["browser", "nodejs", "electron"],
        }


# Register the plugin
register_plugin(WebComponentLanguagePlugin())
