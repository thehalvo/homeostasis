"""
Vue Framework Plugin for Homeostasis

This plugin enables Homeostasis to analyze and fix errors in Vue.js applications.
It provides comprehensive error handling for Vue components, directives, Composition API,
Vuex state management, Vue Router, and Vue 3 specific features.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..language_adapters import JavaScriptErrorAdapter
from ..language_plugin_system import LanguagePlugin, register_plugin

logger = logging.getLogger(__name__)


class VueExceptionHandler:
    """
    Handles Vue-specific exceptions with comprehensive error detection and classification.

    This class provides logic for categorizing Vue component errors, lifecycle issues,
    Composition API problems, Vuex state management errors, and Vue Router navigation issues.
    """

    def __init__(self):
        """Initialize the Vue exception handler."""
        self.rule_categories = {
            "components": "Vue component related errors",
            "directives": "Vue directive errors",
            "composition": "Composition API errors",
            "lifecycle": "Component lifecycle errors",
            "reactivity": "Vue reactivity system errors",
            "vuex": "Vuex state management errors",
            "router": "Vue Router navigation errors",
            "templates": "Template syntax and compilation errors",
            "props": "Props validation and usage errors",
            "events": "Event handling errors",
            "transitions": "Vue transition and animation errors",
            "ssr": "Server-side rendering errors",
        }

        # Load rules from different categories
        self.rules = self._load_rules()

        # Pre-compile regex patterns for better performance
        self._compile_patterns()

    def _load_rules(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load Vue error rules from rule files."""
        rules = {}
        rules_dir = Path(__file__).parent.parent / "rules" / "vue"

        try:
            # Load common Vue rules
            common_rules_path = rules_dir / "vue_common_errors.json"
            if common_rules_path.exists():
                with open(common_rules_path, "r") as f:
                    common_data = json.load(f)
                    rules["common"] = common_data.get("rules", [])
                    logger.info(f"Loaded {len(rules['common'])} common Vue rules")

            # Load Vue component rules
            component_rules_path = rules_dir / "vue_component_errors.json"
            if component_rules_path.exists():
                with open(component_rules_path, "r") as f:
                    component_data = json.load(f)
                    rules["components"] = component_data.get("rules", [])
                    logger.info(
                        f"Loaded {len(rules['components'])} Vue component rules"
                    )

            # Load Composition API rules
            composition_rules_path = rules_dir / "vue_composition_api_errors.json"
            if composition_rules_path.exists():
                with open(composition_rules_path, "r") as f:
                    composition_data = json.load(f)
                    rules["composition"] = composition_data.get("rules", [])
                    logger.info(
                        f"Loaded {len(rules['composition'])} Vue Composition API rules"
                    )

        except Exception as e:
            logger.error(f"Error loading Vue rules: {e}")
            rules = {"common": [], "components": [], "composition": []}

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
                        f"Invalid regex pattern in Vue rule {rule.get('id', 'unknown')}: {e}"
                    )

    def analyze_exception(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a Vue exception and determine its type and potential fixes.

        Args:
            error_data: Vue error data in standard format

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
                "category": best_match.get("category", "vue"),
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

        # Boost confidence for Vue-specific patterns
        message = error_data.get("message", "").lower()
        if "vue" in message or "composition" in message or "reactivity" in message:
            base_confidence += 0.3

        # Boost confidence based on rule reliability
        reliability = rule.get("reliability", "medium")
        reliability_boost = {"high": 0.2, "medium": 0.1, "low": 0.0}
        base_confidence += reliability_boost.get(reliability, 0.0)

        # Boost confidence for rules with specific tags that match context
        rule_tags = set(rule.get("tags", []))
        context_tags = set()

        # Infer context from error data
        if "vue" in error_data.get("framework", "").lower():
            context_tags.add("vue")
        if "vuex" in message:
            context_tags.add("vuex")
        if "composition" in message or "setup" in message:
            context_tags.add("composition")
        if "router" in message:
            context_tags.add("router")
        if "directive" in message:
            context_tags.add("directive")

        if context_tags & rule_tags:
            base_confidence += 0.1

        return min(base_confidence, 1.0)

    def _generic_analysis(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """Provide generic analysis for unmatched errors."""
        message = error_data.get("message", "").lower()

        # Basic categorization based on error patterns
        if "composition" in message or "setup" in message:
            category = "composition"
            suggestion = "Check Composition API usage - ensure reactive refs and computed are properly defined"
        elif "vuex" in message or "store" in message:
            category = "vuex"
            suggestion = "Check Vuex store configuration and mutations"
        elif "router" in message or "navigation" in message:
            category = "router"
            suggestion = "Check Vue Router configuration and navigation guards"
        elif "directive" in message:
            category = "directives"
            suggestion = "Check custom directive implementation and usage"
        elif "component" in message:
            category = "components"
            suggestion = "Check Vue component definition and lifecycle methods"
        elif "template" in message:
            category = "templates"
            suggestion = "Check Vue template syntax and bindings"
        elif "reactive" in message or "ref" in message:
            category = "reactivity"
            suggestion = "Check Vue reactivity system usage"
        else:
            category = "unknown"
            suggestion = "Review Vue application implementation"

        return {
            "category": "vue",
            "subcategory": category,
            "confidence": "low",
            "suggested_fix": suggestion,
            "root_cause": f"vue_{category}_error",
            "severity": "medium",
            "rule_id": "vue_generic_handler",
            "tags": ["vue", "generic", category],
        }

    def analyze_composition_api_error(
        self, error_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze Vue Composition API specific errors.

        Args:
            error_data: Error data with Composition API-related issues

        Returns:
            Analysis results with Composition API-specific fixes
        """
        message = error_data.get("message", "")

        # Common Composition API error patterns
        composition_patterns = {
            "cannot access before initialization": {
                "cause": "vue_composition_ref_access_before_init",
                "fix": "Initialize reactive references before accessing them in setup()",
                "severity": "error",
            },
            "ref is not defined": {
                "cause": "vue_composition_ref_not_defined",
                "fix": "Import ref from vue and define reactive references properly",
                "severity": "error",
            },
            "computed is not defined": {
                "cause": "vue_composition_computed_not_defined",
                "fix": "Import computed from vue and define computed properties properly",
                "severity": "error",
            },
            "watch is not defined": {
                "cause": "vue_composition_watch_not_defined",
                "fix": "Import watch from vue and define watchers properly",
                "severity": "error",
            },
            "onmounted is not defined": {
                "cause": "vue_composition_lifecycle_not_defined",
                "fix": "Import lifecycle hooks from vue (onMounted, onUpdated, etc.)",
                "severity": "error",
            },
            "setup must return an object": {
                "cause": "vue_composition_setup_return_type",
                "fix": "Return an object from setup() function with exposed properties and methods",
                "severity": "error",
            },
        }

        for pattern, info in composition_patterns.items():
            if pattern in message.lower():
                return {
                    "category": "vue",
                    "subcategory": "composition",
                    "confidence": "high",
                    "suggested_fix": info["fix"],
                    "root_cause": info["cause"],
                    "severity": info["severity"],
                    "tags": ["vue", "composition-api", "setup"],
                }

        # Generic Composition API error
        return {
            "category": "vue",
            "subcategory": "composition",
            "confidence": "medium",
            "suggested_fix": "Check Vue Composition API usage in setup() function",
            "root_cause": "vue_composition_api_error",
            "severity": "warning",
            "tags": ["vue", "composition-api"],
        }

    def analyze_vuex_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze Vuex state management errors.

        Args:
            error_data: Error data with Vuex-related issues

        Returns:
            Analysis results with Vuex-specific fixes
        """
        message = error_data.get("message", "").lower()

        # Vuex specific error patterns
        vuex_patterns = {
            "store is not defined": {
                "cause": "vuex_store_not_imported",
                "fix": "Import and inject Vuex store into your component or create a store instance",
                "severity": "error",
            },
            "cannot read property of undefined": {
                "cause": "vuex_undefined_state_property",
                "fix": "Check if the state property exists and is properly defined in Vuex modules",
                "severity": "error",
            },
            "mutation type is not defined": {
                "cause": "vuex_undefined_mutation",
                "fix": "Define the mutation in your Vuex store modules",
                "severity": "error",
            },
            "unknown mutation type": {
                "cause": "vuex_undefined_mutation",
                "fix": "Define the mutation in your Vuex store modules",
                "severity": "error",
            },
            "action type is not defined": {
                "cause": "vuex_undefined_action",
                "fix": "Define the action in your Vuex store modules",
                "severity": "error",
            },
            "do not mutate vuex store state outside mutation handlers": {
                "cause": "vuex_direct_state_mutation",
                "fix": "Use commit() with mutations to modify Vuex state instead of direct assignment",
                "severity": "error",
            },
            "module not found": {
                "cause": "vuex_module_not_found",
                "fix": "Register the Vuex module in your store configuration",
                "severity": "error",
            },
        }

        for pattern, info in vuex_patterns.items():
            if pattern in message:
                return {
                    "category": "vue",
                    "subcategory": "vuex",
                    "confidence": "high",
                    "suggested_fix": info["fix"],
                    "root_cause": info["cause"],
                    "severity": info["severity"],
                    "tags": ["vue", "vuex", "state-management"],
                }

        # Generic Vuex error
        return {
            "category": "vue",
            "subcategory": "vuex",
            "confidence": "medium",
            "suggested_fix": "Check Vuex store configuration and usage",
            "root_cause": "vuex_general_error",
            "severity": "medium",
            "tags": ["vue", "vuex"],
        }

    def analyze_router_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze Vue Router navigation errors.

        Args:
            error_data: Error data with Vue Router-related issues

        Returns:
            Analysis results with Vue Router-specific fixes
        """
        message = error_data.get("message", "").lower()

        # Vue Router specific error patterns
        router_patterns = {
            "router is not defined": {
                "cause": "vue_router_not_imported",
                "fix": "Import and configure Vue Router in your application",
                "severity": "error",
            },
            "route not found": {
                "cause": "vue_router_route_not_found",
                "fix": "Check route configuration and ensure the route path is defined",
                "severity": "error",
            },
            "no match found for location": {
                "cause": "vue_router_route_not_found",
                "fix": "Check route configuration and ensure the route path is defined",
                "severity": "error",
            },
            "navigation cancelled": {
                "cause": "vue_router_navigation_cancelled",
                "fix": "Check navigation guards and ensure they call next() appropriately",
                "severity": "warning",
            },
            "redirected when going from": {
                "cause": "vue_router_redirect_loop",
                "fix": "Check for infinite redirect loops in navigation guards",
                "severity": "warning",
            },
            "uncaught error during route navigation": {
                "cause": "vue_router_navigation_error",
                "fix": "Add error handling in navigation guards and route components",
                "severity": "error",
            },
        }

        for pattern, info in router_patterns.items():
            if pattern in message:
                return {
                    "category": "vue",
                    "subcategory": "router",
                    "confidence": "high",
                    "suggested_fix": info["fix"],
                    "root_cause": info["cause"],
                    "severity": info["severity"],
                    "tags": ["vue", "vue-router", "navigation"],
                }

        # Generic Vue Router error
        return {
            "category": "vue",
            "subcategory": "router",
            "confidence": "medium",
            "suggested_fix": "Check Vue Router configuration and navigation logic",
            "root_cause": "vue_router_general_error",
            "severity": "medium",
            "tags": ["vue", "vue-router"],
        }


class VuePatchGenerator:
    """
    Generates patches for Vue errors based on analysis results.

    This class creates code fixes for common Vue errors using templates
    and heuristics specific to Vue patterns and best practices.
    """

    def __init__(self):
        """Initialize the Vue patch generator."""
        self.template_dir = (
            Path(__file__).parent.parent / "patch_generation" / "templates"
        )
        self.vue_template_dir = self.template_dir / "vue"

        # Ensure template directory exists
        self.vue_template_dir.mkdir(parents=True, exist_ok=True)

        # Load patch templates
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict[str, str]:
        """Load Vue patch templates."""
        templates = {}

        if not self.vue_template_dir.exists():
            logger.warning(
                f"Vue templates directory not found: {self.vue_template_dir}"
            )
            return templates

        for template_file in self.vue_template_dir.glob("*.vue.template"):
            try:
                with open(template_file, "r") as f:
                    template_name = template_file.stem.replace(".vue", "")
                    templates[template_name] = f.read()
                    logger.debug(f"Loaded Vue template: {template_name}")
            except Exception as e:
                logger.error(f"Error loading Vue template {template_file}: {e}")

        return templates

    def generate_patch(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a patch for the Vue error.

        Args:
            error_data: The Vue error data
            analysis: Analysis results from VueExceptionHandler
            source_code: The source code where the error occurred

        Returns:
            Patch information or None if no patch can be generated
        """
        root_cause = analysis.get("root_cause", "")

        # Map root causes to patch strategies
        patch_strategies = {
            "vue_composition_ref_access_before_init": self._fix_ref_access_before_init,
            "vue_composition_ref_not_defined": self._fix_ref_not_defined,
            "vue_composition_computed_not_defined": self._fix_computed_not_defined,
            "vue_composition_watch_not_defined": self._fix_watch_not_defined,
            "vue_composition_lifecycle_not_defined": self._fix_lifecycle_not_defined,
            "vue_composition_setup_return_type": self._fix_setup_return_type,
            "vuex_store_not_imported": self._fix_vuex_store_not_imported,
            "vuex_undefined_mutation": self._fix_vuex_undefined_mutation,
            "vuex_direct_state_mutation": self._fix_vuex_direct_state_mutation,
            "vue_router_not_imported": self._fix_router_not_imported,
            "vue_router_route_not_found": self._fix_route_not_found,
        }

        strategy = patch_strategies.get(root_cause)
        if strategy:
            try:
                return strategy(error_data, analysis, source_code)
            except Exception as e:
                logger.error(f"Error generating Vue patch for {root_cause}: {e}")

        # Try to use templates if no specific strategy matches
        return self._template_based_patch(error_data, analysis, source_code)

    def _fix_ref_access_before_init(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix reactive ref access before initialization."""
        return {
            "type": "suggestion",
            "description": "Initialize reactive references before accessing them",
            "fix_commands": [
                "Move ref initialization to the beginning of setup()",
                "Ensure ref is defined before using it in computed or watchers",
                "Use nextTick() if you need to access DOM elements",
            ],
        }

    def _fix_ref_not_defined(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix undefined ref import."""
        return {
            "type": "line_addition",
            "description": "Add ref import from Vue",
            "line_to_add": "import { ref } from 'vue'",
            "position": "top",
            "fix_code": """import { ref } from 'vue'

export default {
  setup() {
    const myRef = ref(null)
    
    return {
      myRef
    }
  }
}""",
        }

    def _fix_computed_not_defined(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix undefined computed import."""
        return {
            "type": "line_addition",
            "description": "Add computed import from Vue",
            "line_to_add": "import { computed } from 'vue'",
            "position": "top",
            "fix_code": """import { computed, ref } from 'vue'

export default {
  setup() {
    const count = ref(0)
    const doubledCount = computed(() => count.value * 2)
    
    return {
      count,
      doubledCount
    }
  }
}""",
        }

    def _fix_watch_not_defined(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix undefined watch import."""
        return {
            "type": "line_addition",
            "description": "Add watch import from Vue",
            "line_to_add": "import { watch } from 'vue'",
            "position": "top",
            "fix_code": """import { watch, ref } from 'vue'

export default {
  setup() {
    const count = ref(0)
    
    watch(count, (newValue, oldValue) => {
      console.log('Count changed:', newValue)
    })
    
    return {
      count
    }
  }
}""",
        }

    def _fix_lifecycle_not_defined(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix undefined lifecycle hook imports."""
        message = error_data.get("message", "")

        # Extract lifecycle hook name
        lifecycle_match = re.search(
            r"(onMounted|onUpdated|onUnmounted|onBeforeMount|onBeforeUpdate|onBeforeUnmount)",
            message,
            re.IGNORECASE,
        )
        hook_name = lifecycle_match.group(1) if lifecycle_match else "onMounted"

        return {
            "type": "line_addition",
            "description": f"Add {hook_name} import from Vue",
            "line_to_add": f"import {{ {hook_name} }} from 'vue'",
            "position": "top",
            "fix_code": f"""import {{ {hook_name} }} from 'vue'

export default {{
  setup() {{
    {hook_name}(() => {{
      // Lifecycle hook logic here
    }})
    
    return {{}}
  }}
}}""",
        }

    def _fix_setup_return_type(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix setup function return type."""
        return {
            "type": "suggestion",
            "description": "Return an object from setup() function",
            "fix_commands": [
                "Return an object with all reactive data and methods you want to expose",
                "Use ref() for primitive values and reactive() for objects",
                "Export computed properties and methods that should be available in template",
            ],
            "fix_code": """export default {
  setup() {
    const count = ref(0)
    const user = reactive({ name: 'John', age: 30 })
    
    const increment = () => {
      count.value++
    }
    
    // Return object with exposed properties and methods
    return {
      count,
      user,
      increment
    }
  }
}""",
        }

    def _fix_vuex_store_not_imported(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix missing Vuex store import."""
        return {
            "type": "suggestion",
            "description": "Import and configure Vuex store",
            "fix_commands": [
                "Import { createStore } from 'vuex'",
                "Create store instance with state, mutations, actions, and getters",
                "Install store in Vue application using app.use(store)",
            ],
            "fix_code": """import { createStore } from 'vuex'

const store = createStore({
  state: {
    count: 0
  },
  mutations: {
    increment(state) {
      state.count++
    }
  },
  actions: {
    increment({ commit }) {
      commit('increment')
    }
  },
  getters: {
    doubleCount: state => state.count * 2
  }
})

export default store""",
        }

    def _fix_vuex_undefined_mutation(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix undefined Vuex mutation."""
        message = error_data.get("message", "")

        # Extract mutation name if possible
        mutation_match = re.search(r"mutation.*'([^']+)'", message)
        mutation_name = mutation_match.group(1) if mutation_match else "MUTATION_NAME"

        return {
            "type": "suggestion",
            "description": f"Define the '{mutation_name}' mutation in Vuex store",
            "fix_code": f"""mutations: {{
  {mutation_name}(state, payload) {{
    // Mutation logic here
    // Example: state.someProperty = payload
  }}
}}""",
            "fix_commands": [
                f"Add '{mutation_name}' to the mutations object in your Vuex store",
                "Mutations should be synchronous functions that modify state",
                "Use commit() to trigger mutations from components or actions",
            ],
        }

    def _fix_vuex_direct_state_mutation(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix direct Vuex state mutation."""
        return {
            "type": "suggestion",
            "description": "Use mutations to modify Vuex state",
            "fix_commands": [
                "Replace direct state assignment with commit() calls",
                "Define mutations for state modifications",
                "Use this.$store.commit('mutationName', payload) in components",
            ],
            "fix_code": """// Instead of: this.$store.state.count = 5
// Use: this.$store.commit('setCount', 5)

// In store:
mutations: {
  setCount(state, value) {
    state.count = value
  }
}""",
        }

    def _fix_router_not_imported(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix missing Vue Router import."""
        return {
            "type": "suggestion",
            "description": "Import and configure Vue Router",
            "fix_commands": [
                "Import { createRouter, createWebHistory } from 'vue-router'",
                "Define routes array with path and component mappings",
                "Install router in Vue application using app.use(router)",
            ],
            "fix_code": """import { createRouter, createWebHistory } from 'vue-router'
import Home from './components/Home.vue'
import About from './components/About.vue'

const routes = [
  { path: '/', component: Home },
  { path: '/about', component: About }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router""",
        }

    def _fix_route_not_found(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix route not found error."""
        message = error_data.get("message", "")

        # Extract route path if possible
        route_match = re.search(r"route.*'([^']+)'", message)
        route_path = route_match.group(1) if route_match else "/your-route"

        return {
            "type": "suggestion",
            "description": f"Define the '{route_path}' route in router configuration",
            "fix_code": f"""const routes = [
  // Add this route
  {{ path: '{route_path}', component: YourComponent }},
  // ... other routes
]""",
            "fix_commands": [
                f"Add route definition for '{route_path}' in routes array",
                "Ensure the component is imported and available",
                "Check for typos in route path and navigation calls",
            ],
        }

    def _template_based_patch(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Generate patch using templates."""
        root_cause = analysis.get("root_cause", "")

        # Map root causes to template names
        template_map = {
            "vue_composition_ref_not_defined": "composition_ref_import",
            "vue_composition_computed_not_defined": "composition_computed_import",
            "vuex_store_not_imported": "vuex_store_setup",
            "vue_router_not_imported": "vue_router_setup",
        }

        template_name = template_map.get(root_cause)
        if template_name and template_name in self.templates:
            template = self.templates[template_name]

            return {
                "type": "template",
                "template": template,
                "description": f"Applied Vue template fix for {root_cause}",
            }

        return None


class VueLanguagePlugin(LanguagePlugin):
    """
    Main Vue framework plugin for Homeostasis.

    This plugin orchestrates Vue error analysis and patch generation,
    supporting Vue components, Composition API, Vuex, Vue Router, and Vue 3 features.
    """

    VERSION = "1.0.0"
    AUTHOR = "Homeostasis Team"

    def __init__(self):
        """Initialize the Vue language plugin."""
        self.language = "vue"
        self.supported_extensions = {".vue", ".js", ".ts"}
        self.supported_frameworks = [
            "vue",
            "vue3",
            "nuxt",
            "vite-vue",
            "vue-cli",
            "quasar",
            "vuepress",
            "gridsome",
        ]

        # Initialize components
        self.adapter = JavaScriptErrorAdapter()  # Reuse JavaScript adapter
        self.exception_handler = VueExceptionHandler()
        self.patch_generator = VuePatchGenerator()

        logger.info("Vue framework plugin initialized")

    def get_language_id(self) -> str:
        """Get the unique identifier for this language."""
        return "vue"

    def get_language_name(self) -> str:
        """Get the human-readable name of the framework."""
        return "Vue.js"

    def get_language_version(self) -> str:
        """Get the version of the framework supported by this plugin."""
        return "2.x/3.x"

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
        if "vue" in framework:
            return True

        # Check error message for Vue-specific patterns
        message = error_data.get("message", "").lower()
        stack_trace = str(error_data.get("stack_trace", "")).lower()

        # Check for Vuex-specific patterns first
        if "vuex" in message or "mutation" in message or "store" in message:
            return True

        # Check for Vue Router patterns
        if ("router" in message or
                "route" in message or
                "navigation" in message or
                ("location" in message and "path" in message)):
            return True

        vue_patterns = [
            r"vue",
            r"composition",
            r"setup\(\)",
            r"ref\(",
            r"computed\(",
            r"watch\(",
            r"onmounted",
            r"onupdated",
            r"onunmounted",
            r"reactive\(",
            r"vuex",
            r"vue.*router",
            r"router.*vue",
            r"\.vue:",
            r"vue.*component",
            r"component.*vue",
            r"directive.*vue",
            r"vue.*directive",
            r"v-model",
            r"v-if",
            r"v-for",
            r"v-show",
            r"vue.*template",
        ]

        for pattern in vue_patterns:
            if re.search(pattern, message + stack_trace):
                return True

        # Check file extensions for Vue files
        if re.search(r"\.vue:", stack_trace):
            return True

        # Check for Vue in package dependencies (if available)
        context = error_data.get("context", {})
        dependencies = context.get("dependencies", [])
        if any("vue" in dep.lower() or "vuex" in dep.lower() for dep in dependencies):
            return True

        return False

    def analyze_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a Vue error.

        Args:
            error_data: Vue error data

        Returns:
            Analysis results
        """
        try:
            # Ensure error data is in standard format
            if not error_data.get("language"):
                standard_error = self.adapter.to_standard_format(error_data)
            else:
                standard_error = error_data

            # Check if it's a Composition API error
            if self._is_composition_api_error(standard_error):
                analysis = self.exception_handler.analyze_composition_api_error(
                    standard_error
                )

            # Check if it's a Vuex error
            elif self._is_vuex_error(standard_error):
                analysis = self.exception_handler.analyze_vuex_error(standard_error)

            # Check if it's a Vue Router error
            elif self._is_router_error(standard_error):
                analysis = self.exception_handler.analyze_router_error(standard_error)

            # Default Vue error analysis
            else:
                analysis = self.exception_handler.analyze_exception(standard_error)

            # Add plugin metadata
            analysis["plugin"] = "vue"
            analysis["language"] = "vue"
            analysis["plugin_version"] = self.VERSION

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing Vue error: {e}")
            return {
                "category": "vue",
                "subcategory": "unknown",
                "confidence": "low",
                "suggested_fix": "Unable to analyze Vue error",
                "error": str(e),
                "plugin": "vue",
            }

    def _is_composition_api_error(self, error_data: Dict[str, Any]) -> bool:
        """Check if this is a Composition API related error."""
        message = error_data.get("message", "").lower()

        composition_patterns = [
            "setup",
            "ref",
            "computed",
            "watch",
            "onmounted",
            "onupdated",
            "onunmounted",
            "reactive",
            "composition api",
        ]

        return any(pattern in message for pattern in composition_patterns)

    def _is_vuex_error(self, error_data: Dict[str, Any]) -> bool:
        """Check if this is a Vuex related error."""
        message = error_data.get("message", "").lower()
        stack_trace = str(error_data.get("stack_trace", "")).lower()

        vuex_patterns = [
            "vuex",
            "store",
            "mutation",
            "action",
            "getters",
            "commit",
            "dispatch",
        ]

        return any(
            pattern in message or pattern in stack_trace for pattern in vuex_patterns
        )

    def _is_router_error(self, error_data: Dict[str, Any]) -> bool:
        """Check if this is a Vue Router related error."""
        message = error_data.get("message", "").lower()
        stack_trace = str(error_data.get("stack_trace", "")).lower()

        router_patterns = [
            "router",
            "route",
            "navigation",
            "vue-router",
            "$router",
            "$route",
        ]

        return any(
            pattern in message or pattern in stack_trace for pattern in router_patterns
        )

    def generate_fix(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a fix for the Vue error.

        Args:
            error_data: The Vue error data
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
            logger.error(f"Error generating Vue fix: {e}")
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
                "Vue component lifecycle error handling",
                "Composition API error detection and fixes",
                "Vue template syntax error detection",
                "Vuex state management error handling",
                "Vue Router navigation error fixes",
                "Vue 3 features support",
                "Reactivity system error detection",
                "Custom directive error handling",
                "Props validation error fixes",
                "Event handling error detection",
            ],
            "environments": ["browser", "node", "nuxt", "electron"],
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
register_plugin(VueLanguagePlugin())
