"""
Ember.js Framework Plugin for Homeostasis

This plugin enables Homeostasis to analyze and fix errors in Ember.js applications.
It provides comprehensive error handling for Ember components, templates, Ember Data store,
Ember Octane features, router, and testing environment issues.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..language_adapters import JavaScriptErrorAdapter
from ..language_plugin_system import LanguagePlugin, register_plugin

logger = logging.getLogger(__name__)


class EmberExceptionHandler:
    """
    Handles Ember.js-specific exceptions with comprehensive error detection and classification.

    This class provides logic for categorizing Ember component errors, template issues,
    Ember Data store problems, Octane features, and router/URL handling errors.
    """

    def __init__(self):
        """Initialize the Ember exception handler."""
        self.rule_categories = {
            "components": "Ember component related errors",
            "templates": "Handlebars template errors",
            "data": "Ember Data store errors",
            "octane": "Ember Octane features errors",
            "router": "Ember router and URL handling errors",
            "services": "Ember services errors",
            "testing": "Ember testing environment errors",
            "addons": "Ember addon integration errors",
            "lifecycle": "Component lifecycle hook errors",
            "modifiers": "Element modifiers errors",
            "glimmer": "Glimmer component errors",
            "tracking": "Tracked properties errors",
        }

        # Load rules from different categories
        self.rules = self._load_rules()

        # Pre-compile regex patterns for better performance
        self._compile_patterns()

    def _load_rules(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load Ember error rules from rule files."""
        rules = {}
        rules_dir = Path(__file__).parent.parent / "rules" / "ember"

        try:
            # Load common Ember rules
            common_rules_path = rules_dir / "ember_common_errors.json"
            if common_rules_path.exists():
                with open(common_rules_path, "r") as f:
                    common_data = json.load(f)
                    rules["common"] = common_data.get("rules", [])
                    logger.info(f"Loaded {len(rules['common'])} common Ember rules")

            # Load Ember template rules
            template_rules_path = rules_dir / "ember_template_errors.json"
            if template_rules_path.exists():
                with open(template_rules_path, "r") as f:
                    template_data = json.load(f)
                    rules["templates"] = template_data.get("rules", [])
                    logger.info(
                        f"Loaded {len(rules['templates'])} Ember template rules"
                    )

            # Load Ember Data rules
            data_rules_path = rules_dir / "ember_data_errors.json"
            if data_rules_path.exists():
                with open(data_rules_path, "r") as f:
                    data_data = json.load(f)
                    rules["data"] = data_data.get("rules", [])
                    logger.info(f"Loaded {len(rules['data'])} Ember Data rules")

            # Load Ember Octane rules
            octane_rules_path = rules_dir / "ember_octane_errors.json"
            if octane_rules_path.exists():
                with open(octane_rules_path, "r") as f:
                    octane_data = json.load(f)
                    rules["octane"] = octane_data.get("rules", [])
                    logger.info(f"Loaded {len(rules['octane'])} Ember Octane rules")

            # Load Ember Router rules
            router_rules_path = rules_dir / "ember_router_errors.json"
            if router_rules_path.exists():
                with open(router_rules_path, "r") as f:
                    router_data = json.load(f)
                    rules["router"] = router_data.get("rules", [])
                    logger.info(f"Loaded {len(rules['router'])} Ember Router rules")

        except Exception as e:
            logger.error(f"Error loading Ember rules: {e}")
            rules = {
                "common": [],
                "templates": [],
                "data": [],
                "octane": [],
                "router": [],
            }

        return rules

    def _compile_patterns(self):
        """Pre-compile regex patterns for better performance."""
        self.compiled_patterns: Dict[
            str, List[tuple[re.Pattern[str], Dict[str, Any]]]
        ] = {}

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
                        f"Invalid regex pattern in Ember rule {rule.get('id', 'unknown')}: {e}"
                    )

    def analyze_exception(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze an Ember exception and determine its type and potential fixes.

        Args:
            error_data: Ember error data in standard format

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
                "category": best_match.get("category", "ember"),
                "subcategory": best_match.get("subcategory", "unknown"),
                "confidence": best_match.get("confidence", "medium"),
                "suggested_fix": best_match.get("suggestion", ""),
                "root_cause": best_match.get("root_cause", ""),
                "severity": best_match.get("severity", "medium"),
                "rule_id": best_match.get("id", ""),
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

        # Boost confidence for Ember-specific patterns
        message = error_data.get("message", "").lower()
        if "ember" in message or "handlebars" in message or "glimmer" in message:
            base_confidence += 0.3

        # Boost confidence based on rule reliability
        reliability = rule.get("reliability", "medium")
        reliability_boost = {"high": 0.2, "medium": 0.1, "low": 0.0}
        base_confidence += reliability_boost.get(reliability, 0.0)

        # Boost confidence for rules with specific tags that match context
        rule_tags = set(rule.get("tags", []))
        context_tags = set()

        # Infer context from error data
        if "ember" in error_data.get("framework", "").lower():
            context_tags.add("ember")
        if "template" in message or "handlebars" in message:
            context_tags.add("templates")
        if "store" in message or "model" in message or "record" in message:
            context_tags.add("data")
        if "route" in message or "router" in message:
            context_tags.add("router")
        if "modifier" in message:
            context_tags.add("modifiers")
        if "octane" in message or "tracked" in message:
            context_tags.add("octane")

        if context_tags & rule_tags:
            base_confidence += 0.1

        return min(base_confidence, 1.0)

    def _generic_analysis(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """Provide generic analysis for unmatched errors."""
        message = error_data.get("message", "").lower()

        # Basic categorization based on error patterns
        if "template" in message or "handlebars" in message:
            category = "templates"
            suggestion = "Check Handlebars template syntax for errors in your component templates"
        elif "store" in message or "model" in message or "record" in message:
            category = "data"
            suggestion = "Check Ember Data store configuration and model relationships"
        elif "route" in message or "router" in message or "transition" in message:
            category = "router"
            suggestion = "Check Ember router configuration and URL handling"
        elif "component" in message or "element" in message:
            category = "components"
            suggestion = "Check Ember component definition and lifecycle hooks"
        elif "service" in message or "injection" in message:
            category = "services"
            suggestion = "Check Ember service injection and definition"
        elif "test" in message or "assertion" in message:
            category = "testing"
            suggestion = "Check Ember testing environment configuration"
        elif "tracked" in message or "modifier" in message or "glimmer" in message:
            category = "octane"
            suggestion = "Check Ember Octane features implementation"
        else:
            category = "unknown"
            suggestion = "Review Ember application implementation"

        return {
            "category": "ember",
            "subcategory": category,
            "confidence": "low",
            "suggested_fix": suggestion,
            "root_cause": f"ember_{category}_error",
            "severity": "medium",
            "rule_id": "ember_generic_handler",
            "tags": ["ember", "generic", category],
        }

    def analyze_template_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze Ember Handlebars template-specific errors.

        Args:
            error_data: Error data with template-related issues

        Returns:
            Analysis results with template-specific fixes
        """
        message = error_data.get("message", "").lower()

        # Common template error patterns
        template_patterns = {
            "syntax error": {
                "cause": "ember_template_syntax_error",
                "fix": "Fix syntax errors in your Handlebars template",
                "severity": "error",
            },
            "helper not found": {
                "cause": "ember_template_helper_not_found",
                "fix": "Register the helper or check for typos in helper name",
                "severity": "error",
            },
            "helper '": {
                "cause": "ember_template_helper_not_found",
                "fix": "Register the helper or check for typos in helper name",
                "severity": "error",
            },
            "component not found": {
                "cause": "ember_template_component_not_found",
                "fix": "Ensure the component is properly defined and registered",
                "severity": "error",
            },
            "unclosed element": {
                "cause": "ember_template_unclosed_element",
                "fix": "Close HTML elements properly in your templates",
                "severity": "error",
            },
            "modifier not found": {
                "cause": "ember_template_modifier_not_found",
                "fix": "Register the modifier or check for typos in modifier name",
                "severity": "error",
            },
            "block params": {
                "cause": "ember_template_block_params_error",
                "fix": "Check the block parameters in your each/let helpers",
                "severity": "error",
            },
        }

        for pattern, info in template_patterns.items():
            if pattern in message.lower():
                return {
                    "category": "ember",
                    "subcategory": "templates",
                    "confidence": "high",
                    "suggested_fix": info["fix"],
                    "root_cause": info["cause"],
                    "severity": info["severity"],
                    "tags": ["ember", "templates", "handlebars"],
                }

        # Generic template error
        return {
            "category": "ember",
            "subcategory": "templates",
            "confidence": "medium",
            "suggested_fix": "Check your Handlebars template syntax",
            "root_cause": "ember_template_error",
            "severity": "warning",
            "tags": ["ember", "templates"],
        }

    def analyze_data_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze Ember Data store errors.

        Args:
            error_data: Error data with Ember Data-related issues

        Returns:
            Analysis results with Ember Data-specific fixes
        """
        message = error_data.get("message", "").lower()
        stack_trace = str(error_data.get("stack_trace", "")).lower()

        # Ember Data specific error patterns
        data_patterns = {
            "record not found": {
                "cause": "ember_data_record_not_found",
                "fix": "Check that the record exists before attempting to access it",
                "severity": "error",
            },
            "cannot find record": {
                "cause": "ember_data_record_not_found",
                "fix": "Check that the record exists before attempting to access it",
                "severity": "error",
            },
            "is not loaded": {
                "cause": "ember_data_relationship_not_loaded",
                "fix": "Ensure relationships are properly defined and included in API responses",
                "severity": "error",
            },
            "adapter operation failed": {
                "cause": "ember_data_adapter_operation_failed",
                "fix": "Check your adapter configuration and API endpoint",
                "severity": "error",
            },
            "store is not injected": {
                "cause": "ember_data_store_not_injected",
                "fix": "Inject the store service into your component or route",
                "severity": "error",
            },
            "serializer could not": {
                "cause": "ember_data_serializer_error",
                "fix": "Check your serializer configuration for attribute mappings",
                "severity": "error",
            },
            "model not defined": {
                "cause": "ember_data_model_not_defined",
                "fix": "Define the model or check for typos in model name",
                "severity": "error",
            },
        }

        for pattern, info in data_patterns.items():
            if pattern in message or pattern in stack_trace:
                return {
                    "category": "ember",
                    "subcategory": "data",
                    "confidence": "high",
                    "suggested_fix": info["fix"],
                    "root_cause": info["cause"],
                    "severity": info["severity"],
                    "tags": ["ember", "ember-data", "store"],
                }

        # Generic Ember Data error
        return {
            "category": "ember",
            "subcategory": "data",
            "confidence": "medium",
            "suggested_fix": "Check Ember Data store configuration and model definitions",
            "root_cause": "ember_data_general_error",
            "severity": "medium",
            "tags": ["ember", "ember-data"],
        }

    def analyze_router_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze Ember Router navigation errors.

        Args:
            error_data: Error data with Router-related issues

        Returns:
            Analysis results with Router-specific fixes
        """
        message = error_data.get("message", "").lower()
        stack_trace = str(error_data.get("stack_trace", "")).lower()

        # Ember Router specific error patterns
        router_patterns = {
            "route not found": {
                "cause": "ember_router_route_not_found",
                "fix": "Define the route in your router.js file",
                "severity": "error",
            },
            "transition aborted": {
                "cause": "ember_router_transition_aborted",
                "fix": "Check transition hooks and ensure they don't abort unexpectedly",
                "severity": "warning",
            },
            "transition was aborted": {
                "cause": "ember_router_transition_aborted",
                "fix": "Check transition hooks and ensure they don't abort unexpectedly",
                "severity": "warning",
            },
            "transition was rejected": {
                "cause": "ember_router_transition_rejected",
                "fix": "Add proper error handling for transitions",
                "severity": "warning",
            },
            "dynamic segment": {
                "cause": "ember_router_dynamic_segment_error",
                "fix": "Ensure dynamic segments in routes have proper values",
                "severity": "error",
            },
            "router service is not available": {
                "cause": "ember_router_service_not_available",
                "fix": "Inject the router service properly",
                "severity": "error",
            },
        }

        for pattern, info in router_patterns.items():
            if pattern in message.lower() or pattern in stack_trace:
                return {
                    "category": "ember",
                    "subcategory": "router",
                    "confidence": "high",
                    "suggested_fix": info["fix"],
                    "root_cause": info["cause"],
                    "severity": info["severity"],
                    "tags": ["ember", "router", "transition"],
                }

        # Generic Router error
        return {
            "category": "ember",
            "subcategory": "router",
            "confidence": "medium",
            "suggested_fix": "Check Ember router configuration and transition handling",
            "root_cause": "ember_router_general_error",
            "severity": "medium",
            "tags": ["ember", "router"],
        }

    def analyze_octane_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze Ember Octane-specific errors.

        Args:
            error_data: Error data with Octane-related issues

        Returns:
            Analysis results with Octane-specific fixes
        """
        message = error_data.get("message", "").lower()
        stack_trace = str(error_data.get("stack_trace", "")).lower()

        # Ember Octane specific error patterns
        octane_patterns = {
            "tracked properties": {
                "cause": "ember_octane_tracked_properties_error",
                "fix": "Ensure properties are properly marked as @tracked",
                "severity": "error",
            },
            "@tracked": {
                "cause": "ember_octane_tracked_properties_error",
                "fix": "Ensure properties are properly marked as @tracked",
                "severity": "error",
            },
            "glimmer component": {
                "cause": "ember_octane_glimmer_component_error",
                "fix": "Check Glimmer component implementation",
                "severity": "error",
            },
            "modifier": {
                "cause": "ember_octane_modifier_error",
                "fix": "Check element modifier implementation",
                "severity": "error",
            },
            "args": {
                "cause": "ember_octane_args_error",
                "fix": "Access component arguments via this.args instead of this",
                "severity": "error",
            },
            "class-based": {
                "cause": "ember_octane_class_based_error",
                "fix": "Ensure you're using proper class-based component syntax",
                "severity": "error",
            },
        }

        for pattern, info in octane_patterns.items():
            if pattern in message or pattern in stack_trace:
                return {
                    "category": "ember",
                    "subcategory": "octane",
                    "confidence": "high",
                    "suggested_fix": info["fix"],
                    "root_cause": info["cause"],
                    "severity": info["severity"],
                    "tags": ["ember", "octane", "glimmer"],
                }

        # Generic Octane error
        return {
            "category": "ember",
            "subcategory": "octane",
            "confidence": "medium",
            "suggested_fix": "Check Ember Octane features implementation",
            "root_cause": "ember_octane_general_error",
            "severity": "medium",
            "tags": ["ember", "octane"],
        }


class EmberPatchGenerator:
    """
    Generates patches for Ember errors based on analysis results.

    This class creates code fixes for common Ember errors using templates
    and heuristics specific to Ember patterns and best practices.
    """

    def __init__(self):
        """Initialize the Ember patch generator."""
        self.template_dir = (
            Path(__file__).parent.parent / "patch_generation" / "templates"
        )
        self.ember_template_dir = self.template_dir / "ember"

        # Ensure template directory exists
        self.ember_template_dir.mkdir(parents=True, exist_ok=True)

        # Load patch templates
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict[str, str]:
        """Load Ember patch templates."""
        templates: Dict[str, str] = {}

        if not self.ember_template_dir.exists():
            logger.warning(
                f"Ember templates directory not found: {self.ember_template_dir}"
            )
            return templates

        for template_file in self.ember_template_dir.glob("*.hbs.template"):
            try:
                with open(template_file, "r") as f:
                    template_name = template_file.stem.replace(".hbs", "")
                    templates[template_name] = f.read()
                    logger.debug(f"Loaded Ember template: {template_name}")
            except Exception as e:
                logger.error(f"Error loading Ember template {template_file}: {e}")

        # Also load JS templates
        for template_file in self.ember_template_dir.glob("*.js.template"):
            try:
                with open(template_file, "r") as f:
                    template_name = template_file.stem.replace(".js", "")
                    templates[template_name] = f.read()
                    logger.debug(f"Loaded Ember JS template: {template_name}")
            except Exception as e:
                logger.error(f"Error loading Ember JS template {template_file}: {e}")

        return templates

    def generate_patch(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a patch for the Ember error.

        Args:
            error_data: The Ember error data
            analysis: Analysis results from EmberExceptionHandler
            source_code: The source code where the error occurred

        Returns:
            Patch information or None if no patch can be generated
        """
        root_cause = analysis.get("root_cause", "")

        # Map root causes to patch strategies
        patch_strategies = {
            "ember_template_syntax_error": self._fix_template_syntax,
            "ember_template_helper_not_found": self._fix_template_helper_not_found,
            "ember_template_component_not_found": self._fix_template_component_not_found,
            "ember_data_record_not_found": self._fix_data_record_not_found,
            "ember_data_relationship_not_loaded": self._fix_data_relationship_not_loaded,
            "ember_data_store_not_injected": self._fix_data_store_not_injected,
            "ember_router_route_not_found": self._fix_router_route_not_found,
            "ember_router_transition_aborted": self._fix_router_transition_aborted,
            "ember_octane_tracked_properties_error": self._fix_octane_tracked_properties,
            "ember_octane_args_error": self._fix_octane_args_access,
        }

        strategy = patch_strategies.get(root_cause)
        if strategy:
            try:
                return strategy(error_data, analysis, source_code)
            except Exception as e:
                logger.error(f"Error generating Ember patch for {root_cause}: {e}")

        # Try to use templates if no specific strategy matches
        return self._template_based_patch(error_data, analysis, source_code)

    def _fix_template_syntax(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix Handlebars template syntax errors."""
        return {
            "type": "suggestion",
            "description": "Fix syntax errors in your Handlebars template",
            "fix_commands": [
                "Check for unclosed curly braces {{}}",
                "Ensure block helpers have proper closing tags",
                "Fix malformed expressions",
                "Validate HTML element structure",
            ],
        }

    def _fix_template_helper_not_found(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix missing Handlebars helper."""
        message = error_data.get("message", "")

        # Try to extract helper name
        helper_match = re.search(
            r"helper ['\"]?([^'\"\s]+)['\"]? not found", message, re.IGNORECASE
        )
        helper_name = helper_match.group(1) if helper_match else "helper-name"

        return {
            "type": "suggestion",
            "description": f"Register the '{helper_name}' helper",
            "fix_code": f"""// app/helpers/{helper_name}.js
import {{ helper }} from '@ember/component/helper';

export function {helper_name}(params, hash) {{
  // Helper implementation
  return params;
}}

export default helper({helper_name});""",
            "fix_commands": [
                f"Create a helper file at app/helpers/{helper_name}.js",
                "Implement the helper function",
                "Ensure the helper is properly exported",
                "If using a third-party helper, install the addon",
            ],
        }

    def _fix_template_component_not_found(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix missing component in template."""
        message = error_data.get("message", "")

        # Try to extract component name
        component_match = re.search(
            r"component ['\"]?([^'\"\s]+)['\"]? not found", message, re.IGNORECASE
        )
        component_name = component_match.group(1) if component_match else "my-component"

        # Convert component name to file path format (kebab case)
        file_name = component_name.replace("::", "/")

        return {
            "type": "suggestion",
            "description": f"Create the '{component_name}' component",
            "fix_commands": [
                f"Generate the component with 'ember generate component {component_name}'",
                "Implement the component template and class",
                "Ensure the component is properly registered",
                "Check for typos in component invocation",
            ],
            "fix_code": f"""// app/components/{file_name}.js
import Component from '@glimmer/component';

export default class {component_name.replace('-', '_').title()}Component extends Component {{
  // Component implementation
}}""",
        }

    def _fix_data_record_not_found(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix Ember Data record not found errors."""
        return {
            "type": "suggestion",
            "description": "Handle missing records gracefully when record not found",
            "fix_commands": [
                "Add error handling when fetching records",
                "Use findRecord with options { reload: true } to refresh from backend",
                "Check if record exists before accessing properties",
                "Add proper error handling in route's model hook",
            ],
            "fix_code": """// In a route's model hook
model(params) {
  return this.store.findRecord('model-name', params.id)
    .catch(error => {
      // Handle record not found
      this.transitionTo('not-found');
      return null;
    });
}""",
        }

    def _fix_data_relationship_not_loaded(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix Ember Data relationship not loaded errors."""
        return {
            "type": "suggestion",
            "description": "Ensure relationships are properly loaded",
            "fix_commands": [
                "Use include when fetching records to include relationships",
                "Check relationship definitions in your models",
                "Add belongsTo/hasMany with proper inverse",
                "Use async: false for relationships that should be eager-loaded",
            ],
            "fix_code": """// In a route's model hook
model(params) {
  return this.store.findRecord('model-name', params.id, {
    include: 'relationship-name'
  });
}

// In your model definition
import Model, { belongsTo, hasMany } from '@ember-data/model';

export default class YourModel extends Model {
  @belongsTo('related-model', { async: true, inverse: 'relationshipName' })
  relationshipName;
}""",
        }

    def _fix_data_store_not_injected(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix Ember Data store not injected errors."""
        return {
            "type": "suggestion",
            "description": "Properly inject the store service",
            "fix_code": """import Component from '@glimmer/component';
import { inject as service } from '@ember/service';

export default class YourComponent extends Component {
  @service store;
  
  // Now you can use this.store
}""",
            "fix_commands": [
                "Import '@ember/service' and use @service decorator",
                "Inject the store in components, routes, or services",
                "Use this.store to access store methods",
                "In older Ember versions, use 'store: service()' syntax",
            ],
        }

    def _fix_router_route_not_found(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix route not found errors."""
        message = error_data.get("message", "")

        # Try to extract route name from various patterns
        route_match = re.search(
            r"[Rr]oute not found:?\s*['\"]?([^'\"\s]+)['\"]?", message
        )
        if not route_match:
            route_match = re.search(
                r"['\"]([^'\"]+)['\"]\s*(?:route)?\s*not found", message, re.IGNORECASE
            )
        route_name = route_match.group(1) if route_match else "route-name"

        return {
            "type": "suggestion",
            "description": f"Define the '{route_name}' route",
            "fix_code": f"""// app/router.js
Router.map(function() {{
  // Add your route definition
  this.route('{route_name}', {{ path: '/{route_name}' }});
}});

// Generate route file: ember generate route {route_name}""",
            "fix_commands": [
                "Add route definition to router.js",
                f"Generate route with 'ember generate route {route_name}'",
                "Implement route model hook if needed",
                "Create corresponding template file",
            ],
        }

    def _fix_router_transition_aborted(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix router transition aborted errors."""
        return {
            "type": "suggestion",
            "description": "Handle transition aborts properly",
            "fix_code": """// When transitioning
this.router.transitionTo('route-name')
  .catch(error => {
    // Handle transition error
    console.error('Transition failed:', error);
  });

// In a route
beforeModel(transition) {
  // If you need to abort, return a rejected promise with reason
  if (!this.session.isAuthenticated) {
    this.router.transitionTo('login');
    return reject(new Error('Not authenticated'));
  }
}""",
            "fix_commands": [
                "Add catch handlers to route transitions",
                "Return clear rejection reasons from route hooks",
                "Check for authorization in beforeModel hooks",
                "Use intermediateTransitionTo for non-URL changing transitions",
            ],
        }

    def _fix_octane_tracked_properties(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix Octane tracked properties errors."""
        return {
            "type": "suggestion",
            "description": "Use @tracked for reactive properties",
            "fix_code": """import Component from '@glimmer/component';
import { tracked } from '@glimmer/tracking';
import { action } from '@ember/object';

export default class YourComponent extends Component {
  @tracked count = 0;
  
  @action
  increment() {
    this.count++; // Will trigger re-rendering
  }
}""",
            "fix_commands": [
                "Import { tracked } from '@glimmer/tracking'",
                "Add @tracked decorator to properties that change",
                "Directly mutate tracked properties (no this.set needed)",
                "Use @action for event handlers",
            ],
        }

    def _fix_octane_args_access(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix Octane component args access errors."""
        return {
            "type": "suggestion",
            "description": "Access component arguments via this.args",
            "fix_code": """import Component from '@glimmer/component';

export default class YourComponent extends Component {
  // Incorrect: this.paramName
  // Correct:
  get computedValue() {
    return this.args.paramName * 2;
  }
  
  someMethod() {
    console.log(this.args.anotherParam);
  }
}""",
            "fix_commands": [
                "Access component arguments via this.args.paramName",
                "Don't destructure args in class body (use getters instead) - args are read-only",
                "For default values, use getters with nullish coalescing",
                "Remember args are read-only, don't modify them directly",
            ],
        }

    def _template_based_patch(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Generate patch using templates."""
        root_cause = analysis.get("root_cause", "")

        # Map root causes to template names
        template_map = {
            "ember_template_helper_not_found": "helper_definition",
            "ember_template_component_not_found": "component_definition",
            "ember_data_store_not_injected": "store_injection",
            "ember_router_route_not_found": "route_definition",
            "ember_octane_tracked_properties_error": "tracked_properties",
        }

        template_name = template_map.get(root_cause)
        if template_name and template_name in self.templates:
            template = self.templates[template_name]

            return {
                "type": "template",
                "template": template,
                "description": f"Applied Ember template fix for {root_cause}",
            }

        return None


class EmberLanguagePlugin(LanguagePlugin):
    """
    Main Ember.js framework plugin for Homeostasis.

    This plugin orchestrates Ember error analysis and patch generation,
    supporting Ember components, templates, Ember Data, Ember Octane features,
    router, and testing environment issues.
    """

    VERSION = "1.0.0"
    AUTHOR = "Homeostasis Team"

    def __init__(self):
        """Initialize the Ember language plugin."""
        self.language = "ember"
        self.supported_extensions = {".js", ".hbs", ".ts"}
        self.supported_frameworks = [
            "ember",
            "ember-data",
            "ember-octane",
            "ember-cli",
            "glimmer",
            "ember-engines",
            "empress",
            "ember-fastboot",
        ]

        # Initialize components
        self.adapter = JavaScriptErrorAdapter()  # Reuse JavaScript adapter
        self.exception_handler = EmberExceptionHandler()
        self.patch_generator = EmberPatchGenerator()

        logger.info("Ember.js framework plugin initialized")

    def get_language_id(self) -> str:
        """Get the unique identifier for this language."""
        return "ember"

    def get_language_name(self) -> str:
        """Get the human-readable name of the framework."""
        return "Ember.js"

    def get_language_version(self) -> str:
        """Get the version of the framework supported by this plugin."""
        return "3.x/4.x"

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
        # Check if framework is explicitly set
        framework = error_data.get("framework", "").lower()
        if "ember" in framework:
            return True

        # Check error message for Ember-specific patterns
        message = error_data.get("message", "").lower()
        stack_trace = str(error_data.get("stack_trace", "")).lower()

        ember_patterns = [
            r"ember",
            r"handlebars",
            r"hbs",
            r"glimmer",
            r"component",
            r"ember-data",
            r"store\.",
            r"model\(",
            r"record\.",
            r"ember-cli",
            r"route\.",
            r"router\.",
            r"transition",
            r"\.hbs:",
            r"tracked",
            r"octane",
            r"modifier",
            r"service\(",
            r"@service",
            r"this\.args",
            r"template",
        ]

        for pattern in ember_patterns:
            if re.search(pattern, message + stack_trace):
                return True

        # Check file extensions for Ember files
        if re.search(r"\.hbs:", stack_trace) or re.search(r"\.ember\.js:", stack_trace):
            return True

        # Check for Ember in package dependencies (if available)
        context = error_data.get("context", {})
        dependencies = context.get("dependencies", [])
        if any("ember" in dep.lower() for dep in dependencies):
            return True

        return False

    def analyze_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze an Ember error.

        Args:
            error_data: Ember error data

        Returns:
            Analysis results
        """
        try:
            # Ensure error data is in standard format
            if not error_data.get("language"):
                standard_error = self.adapter.to_standard_format(error_data)
            else:
                standard_error = error_data

            # Check if it's a template error
            if self._is_template_error(standard_error):
                analysis = self.exception_handler.analyze_template_error(standard_error)

            # Check if it's an Ember Data error
            elif self._is_data_error(standard_error):
                analysis = self.exception_handler.analyze_data_error(standard_error)

            # Check if it's a Router error
            elif self._is_router_error(standard_error):
                analysis = self.exception_handler.analyze_router_error(standard_error)

            # Check if it's an Octane features error
            elif self._is_octane_error(standard_error):
                analysis = self.exception_handler.analyze_octane_error(standard_error)

            # Default Ember error analysis
            else:
                analysis = self.exception_handler.analyze_exception(standard_error)

            # Add plugin metadata
            analysis["plugin"] = "ember"
            analysis["language"] = "ember"
            analysis["plugin_version"] = self.VERSION

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing Ember error: {e}")
            return {
                "category": "ember",
                "subcategory": "unknown",
                "confidence": "low",
                "suggested_fix": "Unable to analyze Ember error",
                "error": str(e),
                "plugin": "ember",
            }

    def _is_template_error(self, error_data: Dict[str, Any]) -> bool:
        """Check if this is a template-related error."""
        message = error_data.get("message", "").lower()
        stack_trace = str(error_data.get("stack_trace", "")).lower()

        template_patterns = [
            "template",
            "handlebars",
            "hbs",
            "helper",
            "component",
            "syntax error",
            "compile error",
            "unclosed element",
            "{{",
            "}}",
            ".hbs",
        ]

        return any(
            pattern in message or pattern in stack_trace
            for pattern in template_patterns
        )

    def _is_data_error(self, error_data: Dict[str, Any]) -> bool:
        """Check if this is an Ember Data related error."""
        message = error_data.get("message", "").lower()
        stack_trace = str(error_data.get("stack_trace", "")).lower()

        data_patterns = [
            "ember-data",
            "store",
            "model",
            "record",
            "relationship",
            "adapter",
            "serializer",
            "findrecord",
            "query",
            "peekrecord",
        ]

        return any(
            pattern in message or pattern in stack_trace for pattern in data_patterns
        )

    def _is_router_error(self, error_data: Dict[str, Any]) -> bool:
        """Check if this is a Router related error."""
        message = error_data.get("message", "").lower()
        stack_trace = str(error_data.get("stack_trace", "")).lower()

        router_patterns = [
            "router",
            "route",
            "transition",
            "url",
            "link",
            "redirect",
            "beforemodel",
            "aftermodel",
            "getroute",
            "transitionto",
        ]

        return any(
            pattern in message or pattern in stack_trace for pattern in router_patterns
        )

    def _is_octane_error(self, error_data: Dict[str, Any]) -> bool:
        """Check if this is an Octane features related error."""
        message = error_data.get("message", "").lower()
        stack_trace = str(error_data.get("stack_trace", "")).lower()

        octane_patterns = [
            "octane",
            "tracked",
            "glimmer",
            "modifier",
            "args",
            "decorator",
            "@tracked",
            "@action",
            "@service",
        ]

        return any(
            pattern in message or pattern in stack_trace for pattern in octane_patterns
        )

    def generate_fix(
        self, analysis: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a fix for the Ember error.

        Args:
            analysis: Analysis results
            context: Context information including error data and source code

        Returns:
            Fix information as a dictionary
        """
        try:
            error_data = context.get("error_data", {})
            source_code = context.get("source_code", "")

            if self.patch_generator:
                return self.patch_generator.generate_patch(
                    error_data, analysis, source_code
                )
            return {
                "type": "suggestion",
                "description": "Unable to generate automatic fix"
            }
        except Exception as e:
            logger.error(f"Error generating Ember fix: {e}")
            return {
                "type": "error",
                "description": f"Failed to generate fix: {str(e)}"
            }

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
                "Ember component and template error handling",
                "Handlebars template syntax error detection",
                "Ember Data store issue detection and fixes",
                "Ember Octane features support (tracked properties, modifiers)",
                "Ember Router and URL handling error resolution",
                "Ember testing environment debugging",
                "Ember services and dependency injection error handling",
                "Ember addon integration troubleshooting",
            ],
            "environments": ["browser", "node", "fastboot", "electron"],
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
register_plugin(EmberLanguagePlugin())
