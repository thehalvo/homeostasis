"""
Svelte Framework Plugin for Homeostasis

This plugin enables Homeostasis to analyze and fix errors in Svelte applications.
It provides comprehensive error handling for Svelte components, reactivity system,
SvelteKit routes and SSR, store management, transitions, animations, and compiler optimizations.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..language_adapters import JavaScriptErrorAdapter
from ..language_plugin_system import LanguagePlugin, register_plugin

logger = logging.getLogger(__name__)


class SvelteExceptionHandler:
    """
    Handles Svelte-specific exceptions with comprehensive error detection and classification.

    This class provides logic for categorizing Svelte component errors, reactivity issues,
    SvelteKit routing problems, store management errors, and compilation issues.
    """

    def __init__(self):
        """Initialize the Svelte exception handler."""
        self.rule_categories = {
            "reactivity": "Svelte reactivity system errors",
            "components": "Svelte component related errors",
            "stores": "Svelte store management errors",
            "sveltekit": "SvelteKit framework errors",
            "routing": "SvelteKit routing and navigation errors",
            "ssr": "Server-side rendering errors",
            "lifecycle": "Component lifecycle errors",
            "bindings": "Two-way binding errors",
            "transitions": "Svelte transition and animation errors",
            "compilation": "Svelte compiler errors",
            "actions": "Svelte action errors",
            "slots": "Slot and composition errors",
            "context": "Context API errors",
            "module": "Module script errors",
        }

        # Load rules from different categories
        self.rules = self._load_rules()

        # Pre-compile regex patterns for better performance
        self._compile_patterns()

    def _load_rules(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load Svelte error rules from rule files."""
        rules = {}
        rules_dir = Path(__file__).parent.parent / "rules" / "svelte"

        try:
            # Load common Svelte rules
            common_rules_path = rules_dir / "svelte_common_errors.json"
            if common_rules_path.exists():
                with open(common_rules_path, "r") as f:
                    common_data = json.load(f)
                    rules["common"] = common_data.get("rules", [])
                    logger.info(f"Loaded {len(rules['common'])} common Svelte rules")

            # Load reactivity rules
            reactivity_rules_path = rules_dir / "svelte_reactivity_errors.json"
            if reactivity_rules_path.exists():
                with open(reactivity_rules_path, "r") as f:
                    reactivity_data = json.load(f)
                    rules["reactivity"] = reactivity_data.get("rules", [])
                    logger.info(
                        f"Loaded {len(rules['reactivity'])} Svelte reactivity rules"
                    )

            # Load SvelteKit rules
            sveltekit_rules_path = rules_dir / "sveltekit_errors.json"
            if sveltekit_rules_path.exists():
                with open(sveltekit_rules_path, "r") as f:
                    sveltekit_data = json.load(f)
                    rules["sveltekit"] = sveltekit_data.get("rules", [])
                    logger.info(f"Loaded {len(rules['sveltekit'])} SvelteKit rules")

        except Exception as e:
            logger.error(f"Error loading Svelte rules: {e}")
            rules = {"common": [], "reactivity": [], "sveltekit": []}

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
                        f"Invalid regex pattern in Svelte rule {rule.get('id', 'unknown')}: {e}"
                    )

    def analyze_exception(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a Svelte exception and determine its type and potential fixes.

        Args:
            error_data: Svelte error data in standard format

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
                "category": best_match.get("category", "svelte"),
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

        # Boost confidence for Svelte-specific patterns
        message = error_data.get("message", "").lower()
        if "svelte" in message or "reactivity" in message or "sveltekit" in message:
            base_confidence += 0.3

        # Boost confidence based on rule reliability
        reliability = rule.get("reliability", "medium")
        reliability_boost = {"high": 0.2, "medium": 0.1, "low": 0.0}
        base_confidence += reliability_boost.get(reliability, 0.0)

        # Boost confidence for rules with specific tags that match context
        rule_tags = set(rule.get("tags", []))
        context_tags = set()

        # Infer context from error data
        if "svelte" in error_data.get("framework", "").lower():
            context_tags.add("svelte")
        if "sveltekit" in message:
            context_tags.add("sveltekit")
        if "reactive" in message or "$:" in message:
            context_tags.add("reactivity")
        if "store" in message or "writable" in message:
            context_tags.add("stores")
        if "transition" in message or "animate" in message:
            context_tags.add("transitions")
        if "compile" in message:
            context_tags.add("compilation")

        if context_tags & rule_tags:
            base_confidence += 0.1

        return min(base_confidence, 1.0)

    def _generic_analysis(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """Provide generic analysis for unmatched errors."""
        message = error_data.get("message", "").lower()

        # Basic categorization based on error patterns
        if "reactive" in message or "$:" in message:
            category = "reactivity"
            suggestion = "Check Svelte reactivity statements and variable assignments"
        elif "store" in message or "writable" in message or "readable" in message:
            category = "stores"
            suggestion = "Check Svelte store usage and subscription patterns"
        elif "sveltekit" in message or "kit" in message:
            category = "sveltekit"
            suggestion = "Check SvelteKit configuration and routing"
        elif "ssr" in message or "server" in message:
            category = "ssr"
            suggestion = "Check server-side rendering compatibility"
        elif "transition" in message or "animate" in message:
            category = "transitions"
            suggestion = "Check Svelte transition and animation usage"
        elif "compile" in message or "parser" in message:
            category = "compilation"
            suggestion = "Check Svelte component syntax and compilation"
        elif "bind:" in message or "binding" in message:
            category = "bindings"
            suggestion = "Check two-way binding syntax and usage"
        elif "slot" in message:
            category = "slots"
            suggestion = "Check slot usage and component composition"
        elif "context" in message:
            category = "context"
            suggestion = "Check Svelte context API usage"
        elif "action" in message:
            category = "actions"
            suggestion = "Check Svelte action implementation"
        else:
            category = "unknown"
            suggestion = "Review Svelte component implementation"

        return {
            "category": "svelte",
            "subcategory": category,
            "confidence": "low",
            "suggested_fix": suggestion,
            "root_cause": f"svelte_{category}_error",
            "severity": "medium",
            "rule_id": "svelte_generic_handler",
            "tags": ["svelte", "generic", category],
        }

    def analyze_reactivity_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze Svelte reactivity system specific errors.

        Args:
            error_data: Error data with reactivity-related issues

        Returns:
            Analysis results with reactivity-specific fixes
        """
        # First try using the loaded rules
        analysis = self.analyze_exception(error_data)

        # If we got a good match from rules, return it
        if (
            analysis.get("subcategory") == "reactivity" and
            analysis.get("confidence") in ["high", "medium"] and
            analysis.get("rule_id") != "svelte_generic_handler"
        ):
            return analysis

        # Fallback to hardcoded patterns for backward compatibility
        message = error_data.get("message", "")

        # Common reactivity error patterns
        reactivity_patterns = {
            "reactive statement ran more than 10 times": {
                "cause": "svelte_reactive_infinite_loop",
                "fix": "Avoid circular dependencies in reactive statements",
                "severity": "error",
            },
            "variable is not defined": {
                "cause": "svelte_reactive_undefined_variable",
                "fix": "Declare variables before using them in reactive statements",
                "severity": "error",
            },
            "undefinedvariable": {
                "cause": "svelte_reactive_undefined_variable",
                "fix": "Declare variables before using them in reactive statements",
                "severity": "error",
            },
            "cannot access before initialization": {
                "cause": "svelte_reactive_access_before_init",
                "fix": "Initialize variables before using them in reactive statements",
                "severity": "error",
            },
            "assignment to constant variable": {
                "cause": "svelte_reactive_const_assignment",
                "fix": "Use let instead of const for variables that need to change",
                "severity": "error",
            },
            "reactive statement has unused dependencies": {
                "cause": "svelte_reactive_unused_dependencies",
                "fix": "Remove unused variables from reactive statement dependencies",
                "severity": "warning",
            },
            "reactive statement has missing dependencies": {
                "cause": "svelte_reactive_missing_dependencies",
                "fix": "Include all referenced variables in reactive statement",
                "severity": "warning",
            },
        }

        for pattern, info in reactivity_patterns.items():
            if pattern in message.lower():
                return {
                    "category": "svelte",
                    "subcategory": "reactivity",
                    "confidence": "high",
                    "suggested_fix": info["fix"],
                    "root_cause": info["cause"],
                    "severity": info["severity"],
                    "tags": ["svelte", "reactivity", "reactive-statements"],
                }

        # Generic reactivity error
        return {
            "category": "svelte",
            "subcategory": "reactivity",
            "confidence": "medium",
            "suggested_fix": "Check Svelte reactivity statements and variable dependencies",
            "root_cause": "svelte_reactivity_error",
            "severity": "warning",
            "tags": ["svelte", "reactivity"],
        }

    def analyze_store_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze Svelte store management errors.

        Args:
            error_data: Error data with store-related issues

        Returns:
            Analysis results with store-specific fixes
        """
        message = error_data.get("message", "").lower()

        # Store specific error patterns
        store_patterns = {
            "writable is not defined": {
                "cause": "svelte_store_not_imported",
                "fix": "Import writable from 'svelte/store'",
                "severity": "error",
            },
            "readable is not defined": {
                "cause": "svelte_readable_not_imported",
                "fix": "Import readable from 'svelte/store'",
                "severity": "error",
            },
            "derived is not defined": {
                "cause": "svelte_derived_not_imported",
                "fix": "Import derived from 'svelte/store'",
                "severity": "error",
            },
            "cannot subscribe to undefined": {
                "cause": "svelte_store_undefined_subscription",
                "fix": "Ensure store is defined before subscribing",
                "severity": "error",
            },
            "store subscription leak": {
                "cause": "svelte_store_subscription_leak",
                "fix": "Unsubscribe from stores in onDestroy or use $ syntax for auto-subscription",
                "severity": "warning",
            },
            "store update outside of component": {
                "cause": "svelte_store_update_outside_component",
                "fix": "Update stores within component lifecycle or proper context",
                "severity": "warning",
            },
        }

        for pattern, info in store_patterns.items():
            if pattern in message:
                return {
                    "category": "svelte",
                    "subcategory": "stores",
                    "confidence": "high",
                    "suggested_fix": info["fix"],
                    "root_cause": info["cause"],
                    "severity": info["severity"],
                    "tags": ["svelte", "stores", "state-management"],
                }

        # Generic store error
        return {
            "category": "svelte",
            "subcategory": "stores",
            "confidence": "medium",
            "suggested_fix": "Check Svelte store usage and subscription patterns",
            "root_cause": "svelte_store_error",
            "severity": "medium",
            "tags": ["svelte", "stores"],
        }

    def analyze_sveltekit_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze SvelteKit framework errors.

        Args:
            error_data: Error data with SvelteKit-related issues

        Returns:
            Analysis results with SvelteKit-specific fixes
        """
        message = error_data.get("message", "").lower()

        # SvelteKit specific error patterns
        sveltekit_patterns = {
            "load function must return an object": {
                "cause": "sveltekit_load_return_type",
                "fix": "Load function must return an object with props or other data",
                "severity": "error",
            },
            "cannot use goto during ssr": {
                "cause": "sveltekit_goto_ssr_error",
                "fix": "Use goto only in browser context, check browser condition",
                "severity": "error",
            },
            "layout not found": {
                "cause": "sveltekit_layout_not_found",
                "fix": "Create __layout.svelte file or check layout file naming",
                "severity": "error",
            },
            "page not found": {
                "cause": "sveltekit_page_not_found",
                "fix": "Create page component or check route file naming",
                "severity": "error",
            },
            "endpoint must export a function": {
                "cause": "sveltekit_endpoint_export_error",
                "fix": "Export GET, POST, or other HTTP method functions from endpoint",
                "severity": "error",
            },
            "hydration mismatch": {
                "cause": "sveltekit_hydration_mismatch",
                "fix": "Ensure server and client render the same content",
                "severity": "error",
            },
            "prerender error": {
                "cause": "sveltekit_prerender_error",
                "fix": "Check prerender configuration and page dependencies",
                "severity": "warning",
            },
        }

        for pattern, info in sveltekit_patterns.items():
            if pattern in message:
                return {
                    "category": "svelte",
                    "subcategory": "sveltekit",
                    "confidence": "high",
                    "suggested_fix": info["fix"],
                    "root_cause": info["cause"],
                    "severity": info["severity"],
                    "tags": ["svelte", "sveltekit", "ssr"],
                }

        # Generic SvelteKit error
        return {
            "category": "svelte",
            "subcategory": "sveltekit",
            "confidence": "medium",
            "suggested_fix": "Check SvelteKit configuration and routing",
            "root_cause": "sveltekit_general_error",
            "severity": "medium",
            "tags": ["svelte", "sveltekit"],
        }


class SveltePatchGenerator:
    """
    Generates patches for Svelte errors based on analysis results.

    This class creates code fixes for common Svelte errors using templates
    and heuristics specific to Svelte patterns and best practices.
    """

    def __init__(self):
        """Initialize the Svelte patch generator."""
        self.template_dir = (
            Path(__file__).parent.parent / "patch_generation" / "templates"
        )
        self.svelte_template_dir = self.template_dir / "svelte"

        # Ensure template directory exists
        self.svelte_template_dir.mkdir(parents=True, exist_ok=True)

        # Load patch templates
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict[str, str]:
        """Load Svelte patch templates."""
        templates = {}

        if not self.svelte_template_dir.exists():
            logger.warning(
                f"Svelte templates directory not found: {self.svelte_template_dir}"
            )
            return templates

        for template_file in self.svelte_template_dir.glob("*.svelte.template"):
            try:
                with open(template_file, "r") as f:
                    template_name = template_file.stem.replace(".svelte", "")
                    templates[template_name] = f.read()
                    logger.debug(f"Loaded Svelte template: {template_name}")
            except Exception as e:
                logger.error(f"Error loading Svelte template {template_file}: {e}")

        return templates

    def generate_patch(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a patch for the Svelte error.

        Args:
            error_data: The Svelte error data
            analysis: Analysis results from SvelteExceptionHandler
            source_code: The source code where the error occurred

        Returns:
            Patch information or None if no patch can be generated
        """
        root_cause = analysis.get("root_cause", "")

        # Map root causes to patch strategies
        patch_strategies = {
            "svelte_reactive_infinite_loop": self._fix_reactive_infinite_loop,
            "svelte_reactive_undefined_variable": self._fix_reactive_undefined_variable,
            "svelte_reactive_access_before_init": self._fix_reactive_access_before_init,
            "svelte_store_not_imported": self._fix_store_not_imported,
            "svelte_store_subscription_leak": self._fix_store_subscription_leak,
            "sveltekit_load_return_type": self._fix_sveltekit_load_return_type,
            "sveltekit_goto_ssr_error": self._fix_sveltekit_goto_ssr_error,
            "sveltekit_hydration_mismatch": self._fix_sveltekit_hydration_mismatch,
            "svelte_transition_error": self._fix_transition_error,
            "svelte_binding_error": self._fix_binding_error,
        }

        strategy = patch_strategies.get(root_cause)
        if strategy:
            try:
                return strategy(error_data, analysis, source_code)
            except Exception as e:
                logger.error(f"Error generating Svelte patch for {root_cause}: {e}")

        # Try to use templates if no specific strategy matches
        return self._template_based_patch(error_data, analysis, source_code)

    def _fix_reactive_infinite_loop(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix reactive statement infinite loops."""
        return {
            "type": "suggestion",
            "description": "Break reactive statement circular dependencies",
            "fix_commands": [
                "Avoid assigning to variables that the reactive statement depends on",
                "Use intermediate variables to break circular dependencies",
                "Consider using stores for complex state management",
            ],
            "fix_code": """// Before (causes infinite loop):
// $: result = calculate(input);
// $: input = result + 1;

// After (breaks the loop):
let tempInput = 0;
$: result = calculate(tempInput);
$: if (someCondition) {
  tempInput = result + 1;
}""",
        }

    def _fix_reactive_undefined_variable(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix undefined variable in reactive statements."""
        return {
            "type": "suggestion",
            "description": "Declare variables before using in reactive statements",
            "fix_commands": [
                "Declare all variables with let before using them",
                "Initialize variables with default values",
                "Check variable scope and availability",
            ],
            "fix_code": """<script>
  // Declare variables first
  let count = 0;
  let doubledCount;
  
  // Then use in reactive statements
  $: doubledCount = count * 2;
</script>""",
        }

    def _fix_reactive_access_before_init(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix accessing variables before initialization."""
        return {
            "type": "suggestion",
            "description": "Initialize variables before accessing them",
            "fix_commands": [
                "Initialize variables at declaration",
                "Use conditional checks before accessing",
                "Move reactive statements after variable declarations",
            ],
            "fix_code": """<script>
  // Initialize at declaration
  let data = {};
  let isLoaded = false;
  
  // Safe reactive access
  $: processedData = isLoaded ? processData(data) : null;
</script>""",
        }

    def _fix_store_not_imported(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix missing store imports."""
        message = error_data.get("message", "")

        # Determine which store type is missing
        if "writable" in message.lower():
            store_type = "writable"
        elif "readable" in message.lower():
            store_type = "readable"
        elif "derived" in message.lower():
            store_type = "derived"
        else:
            store_type = "writable"

        return {
            "type": "line_addition",
            "description": f"Import {store_type} from svelte/store",
            "line_to_add": f"import {{ {store_type} }} from 'svelte/store';",
            "position": "top",
            "fix_code": f"""import {{ {store_type} }} from 'svelte/store';

// Create store
export const myStore = {store_type}(initialValue);""",
        }

    def _fix_store_subscription_leak(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix store subscription memory leaks."""
        return {
            "type": "suggestion",
            "description": "Properly manage store subscriptions",
            "fix_commands": [
                "Use $ syntax for automatic subscription management",
                "Unsubscribe in onDestroy if using manual subscriptions",
                "Store unsubscribe function and call it on component destruction",
            ],
            "fix_code": """<script>
  import { onDestroy } from 'svelte';
  import { myStore } from './stores.js';
  
  // Option 1: Use $ syntax (automatic cleanup)
  $: storeValue = $myStore;
  
  // Option 2: Manual subscription with cleanup
  let storeValue;
  const unsubscribe = myStore.subscribe(value => {
    storeValue = value;
  });
  
  onDestroy(() => {
    unsubscribe();
  });
</script>""",
        }

    def _fix_sveltekit_load_return_type(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix SvelteKit load function return type."""
        return {
            "type": "suggestion",
            "description": "Must return object from load function with proper structure",
            "fix_commands": [
                "Return object with props property for component data",
                "Return object with status and error for error handling",
                "Use proper TypeScript types for load function",
            ],
            "fix_code": """// In +page.js or +layout.js
export async function load({ params, url, fetch }) {
  try {
    const data = await fetchData(params.id);
    
    return {
      props: {
        data
      }
    };
  } catch (error) {
    return {
      status: 500,
      error: error.message
    };
  }
}""",
        }

    def _fix_sveltekit_goto_ssr_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix SvelteKit goto usage during SSR."""
        return {
            "type": "suggestion",
            "description": "Use goto only in browser context",
            "fix_commands": [
                "Check browser environment before using goto",
                "Use proper navigation methods for SSR",
                "Handle navigation in onMount or client-side events",
            ],
            "fix_code": """<script>
  import { browser } from '$app/environment';
  import { goto } from '$app/navigation';
  import { onMount } from 'svelte';
  
  function handleNavigation() {
    if (browser) {
      goto('/target-page');
    }
  }
  
  // Or use onMount for client-side navigation
  onMount(() => {
    if (shouldRedirect) {
      goto('/target-page');
    }
  });
</script>""",
        }

    def _fix_sveltekit_hydration_mismatch(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix SvelteKit hydration mismatches."""
        return {
            "type": "suggestion",
            "description": "Ensure server and client render the same content",
            "fix_commands": [
                "Use browser check for client-only content",
                "Ensure consistent data between server and client",
                "Use {#if mounted} for client-only components",
            ],
            "fix_code": """<script>
  import { browser } from '$app/environment';
  import { onMount } from 'svelte';
  
  let mounted = false;
  
  onMount(() => {
    mounted = true;
  });
</script>

<!-- For client-only content -->
{#if browser}
  <ClientOnlyComponent />
{/if}

<!-- Or use mounted flag -->
{#if mounted}
  <ComponentWithRandomContent />
{/if}""",
        }

    def _fix_transition_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix Svelte transition errors."""
        return {
            "type": "suggestion",
            "description": "Fix transition and animation usage",
            "fix_commands": [
                "Import transition functions from svelte/transition",
                "Use proper transition syntax on elements",
                "Check transition parameters and options",
            ],
            "fix_code": """<script>
  import { fade, slide, fly } from 'svelte/transition';
  
  let visible = true;
</script>

{#if visible}
  <div transition:fade>Fading content</div>
{/if}

{#if visible}
  <div transition:slide={{ duration: 300 }}>Sliding content</div>
{/if}

{#if visible}
  <div transition:fly={{ x: 200, duration: 500 }}>Flying content</div>
{/if}""",
        }

    def _fix_binding_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix two-way binding errors."""
        return {
            "type": "suggestion",
            "description": "Fix two-way binding syntax and usage",
            "fix_commands": [
                "Use bind: directive for two-way binding",
                "Ensure bound variable is writable (let, not const)",
                "Check binding target supports the property",
            ],
            "fix_code": """<script>
  let inputValue = '';
  let checked = false;
  let selectedValue = '';
</script>

<!-- Text input binding -->
<input type="text" bind:value={inputValue} />

<!-- Checkbox binding -->
<input type="checkbox" bind:checked={checked} />

<!-- Select binding -->
<select bind:value={selectedValue}>
  <option value="a">Option A</option>
  <option value="b">Option B</option>
</select>""",
        }

    def _template_based_patch(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Generate patch using templates."""
        root_cause = analysis.get("root_cause", "")

        # Map root causes to template names
        template_map = {
            "svelte_store_not_imported": "store_import_fix",
            "svelte_reactive_infinite_loop": "reactive_statement_fix",
            "sveltekit_load_return_type": "sveltekit_load_fix",
            "svelte_transition_error": "transition_fix",
        }

        template_name = template_map.get(root_cause)
        if template_name and template_name in self.templates:
            template = self.templates[template_name]

            return {
                "type": "template",
                "template": template,
                "description": f"Applied Svelte template fix for {root_cause}",
            }

        return None


class SvelteLanguagePlugin(LanguagePlugin):
    """
    Main Svelte framework plugin for Homeostasis.

    This plugin orchestrates Svelte error analysis and patch generation,
    supporting Svelte components, reactivity, stores, SvelteKit, and transitions.
    """

    VERSION = "1.0.0"
    AUTHOR = "Homeostasis Team"

    def __init__(self):
        """Initialize the Svelte language plugin."""
        self.language = "svelte"
        self.supported_extensions = {".svelte", ".js", ".ts"}
        self.supported_frameworks = [
            "svelte",
            "sveltekit",
            "@sveltejs/kit",
            "vite-svelte",
            "rollup-svelte",
            "webpack-svelte",
            "svelte-native",
        ]

        # Initialize components
        self.adapter = JavaScriptErrorAdapter()  # Reuse JavaScript adapter
        self.exception_handler = SvelteExceptionHandler()
        self.patch_generator = SveltePatchGenerator()

        logger.info("Svelte framework plugin initialized")

    def get_language_id(self) -> str:
        """Get the unique identifier for this language."""
        return "svelte"

    def get_language_name(self) -> str:
        """Get the human-readable name of the framework."""
        return "Svelte"

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
        if "svelte" in framework:
            return True

        # Check error message for Svelte-specific patterns
        message = error_data.get("message", "").lower()
        stack_trace = str(error_data.get("stack_trace", "")).lower()

        svelte_patterns = [
            r"svelte",
            r"sveltekit",
            r"@sveltejs",
            r"\$:",
            r"reactive statement",
            r"writable\(",
            r"readable\(",
            r"derived\(",
            r"writable.*not.*defined",
            r"readable.*not.*defined",
            r"derived.*not.*defined",
            r"\.svelte:",
            r"svelte/store",
            r"svelte/transition",
            r"svelte/animate",
            r"svelte/action",
            r"\+page\.",
            r"\+layout\.",
            r"\+error\.",
            r"app\.html",
            r"bind:",
            r"transition:",
            r"animate:",
            r"use:",
            r"slot=",
            r"createEventDispatcher",
            r"getContext",
            r"setContext",
            r"beforeUpdate",
            r"afterUpdate",
            r"tick\(\)",
        ]

        for pattern in svelte_patterns:
            if re.search(pattern, message + stack_trace):
                return True

        # Check file extensions for Svelte files
        if re.search(r"\.svelte:", stack_trace):
            return True

        # Check for Svelte in package dependencies (if available)
        context = error_data.get("context", {})
        dependencies = context.get("dependencies", [])
        if any("svelte" in dep.lower() for dep in dependencies):
            return True

        return False

    def analyze_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a Svelte error.

        Args:
            error_data: Svelte error data

        Returns:
            Analysis results
        """
        try:
            # Ensure error data is in standard format
            if not error_data.get("language"):
                standard_error = self.adapter.to_standard_format(error_data)
            else:
                standard_error = error_data

            # Check if it's a reactivity error
            if self._is_reactivity_error(standard_error):
                analysis = self.exception_handler.analyze_reactivity_error(
                    standard_error
                )

            # Check if it's a store error
            elif self._is_store_error(standard_error):
                analysis = self.exception_handler.analyze_store_error(standard_error)

            # Check if it's a SvelteKit error
            elif self._is_sveltekit_error(standard_error):
                analysis = self.exception_handler.analyze_sveltekit_error(
                    standard_error
                )

            # Default Svelte error analysis
            else:
                analysis = self.exception_handler.analyze_exception(standard_error)

            # Add plugin metadata
            analysis["plugin"] = "svelte"
            analysis["language"] = "svelte"
            analysis["plugin_version"] = self.VERSION

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing Svelte error: {e}")
            return {
                "category": "svelte",
                "subcategory": "unknown",
                "confidence": "low",
                "suggested_fix": "Unable to analyze Svelte error",
                "error": str(e),
                "plugin": "svelte",
            }

    def _is_reactivity_error(self, error_data: Dict[str, Any]) -> bool:
        """Check if this is a reactivity related error."""
        message = error_data.get("message", "").lower()

        reactivity_patterns = [
            "$:",
            "reactive statement",
            "reactive",
            "assignment",
            "variable is not defined",
            "cannot access before initialization",
            "ran more than 10 times",
        ]

        return any(pattern in message for pattern in reactivity_patterns)

    def _is_store_error(self, error_data: Dict[str, Any]) -> bool:
        """Check if this is a store related error."""
        message = error_data.get("message", "").lower()
        stack_trace = str(error_data.get("stack_trace", "")).lower()

        store_patterns = [
            "writable",
            "readable",
            "derived",
            "store",
            "subscribe",
            "unsubscribe",
            "svelte/store",
        ]

        return any(
            pattern in message or pattern in stack_trace for pattern in store_patterns
        )

    def _is_sveltekit_error(self, error_data: Dict[str, Any]) -> bool:
        """Check if this is a SvelteKit related error."""
        message = error_data.get("message", "").lower()
        stack_trace = str(error_data.get("stack_trace", "")).lower()

        sveltekit_patterns = [
            "sveltekit",
            "@sveltejs/kit",
            "load function",
            "goto",
            "page",
            "layout",
            "endpoint",
            "hydration",
            "prerender",
            "ssr",
            "+page.",
            "+layout.",
            "+error.",
        ]

        return any(
            pattern in message or pattern in stack_trace
            for pattern in sveltekit_patterns
        )

    def generate_fix(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a fix for the Svelte error.

        Args:
            error_data: The Svelte error data
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
            logger.error(f"Error generating Svelte fix: {e}")
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
                "Svelte component reactivity error detection",
                "SvelteKit route and SSR healing",
                "Svelte store management issue resolution",
                "Svelte transition and animation debugging",
                "Svelte compiler optimization support",
                "Two-way binding error fixes",
                "Svelte action error handling",
                "Context API error detection",
                "Slot composition error fixes",
                "Lifecycle hook error handling",
            ],
            "environments": ["browser", "node", "sveltekit", "electron"],
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
register_plugin(SvelteLanguagePlugin())
