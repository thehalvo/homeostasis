"""
Web Framework-Specific Error Parsers

This module provides comprehensive error detection and parsing capabilities for
modern web frameworks including React, Vue.js, Angular, Svelte, Next.js, Ember.js,
and Web Components.
"""

import re
import logging
from typing import Dict, List, Optional, Any

from .comprehensive_error_detector import ErrorContext, ErrorCategory, LanguageType
from .language_parsers import LanguageSpecificParser

logger = logging.getLogger(__name__)


class ReactParser(LanguageSpecificParser):
    """React framework-specific error parser."""
    
    def __init__(self):
        super().__init__(LanguageType.JAVASCRIPT)
        
        # React-specific error patterns
        self.react_patterns = [
            # Hooks errors
            (r"Invalid hook call\. Hooks can only be called inside the body of a function component", "invalid_hook_call"),
            (r"Cannot read propert(?:y|ies) '(.+)' of undefined.*useEffect", "useeffect_undefined"),
            (r"React Hook (.+) has missing dependencies: \[(.+)\]", "missing_dependencies"),
            (r"React Hook (.+) has an unnecessary dependency: (.+)", "unnecessary_dependency"),
            (r"React Hook (.+) cannot be called conditionally", "conditional_hook"),
            
            # Component errors
            (r"Element type is invalid: expected a string \(for built-in components\) or a class/function", "invalid_element_type"),
            (r"Cannot read propert(?:y|ies) '(.+)' of undefined.*render", "render_undefined_property"),
            (r"Cannot access before initialization.*const \[(.+), (.+)\] = useState", "usestate_before_init"),
            (r"Warning: Each child in a list should have a unique \"key\" prop", "missing_key_prop"),
            (r"Warning: Function components cannot be given refs", "function_component_ref"),
            
            # State management errors
            (r"Cannot call setState on a component that is not yet mounted", "setstate_unmounted"),
            (r"Cannot read propert(?:y|ies) '(.+)' of undefined.*setState", "setstate_undefined"),
            (r"Warning: Can't perform a React state update on an unmounted component", "state_update_unmounted"),
            
            # Redux errors
            (r"Error: Could not find \"store\" in the context of \"Connect\((.+)\)\"", "redux_missing_store"),
            (r"Actions must be plain objects", "redux_invalid_action"),
            (r"Reducer \"(.+)\" returned undefined during initialization", "redux_undefined_reducer"),
            
            # JSX errors
            (r"Adjacent JSX elements must be wrapped in an enclosing tag", "jsx_adjacent_elements"),
            (r"Unexpected token '<'", "jsx_syntax_error"),
            (r"'(.+)' is not defined.*JSX", "jsx_undefined_component"),
            
            # Props and PropTypes errors
            (r"Warning: Failed prop type: (.+)", "proptypes_validation"),
            (r"Warning: (.+): prop type `(.+)` is invalid", "invalid_proptype"),
            
            # Event handling errors
            (r"Cannot read propert(?:y|ies) '(.+)' of undefined.*onClick", "onclick_undefined"),
            (r"(.+) is not a function.*onClick", "onclick_not_function"),
            
            # Performance and rendering errors
            (r"Maximum update depth exceeded", "infinite_render_loop"),
            (r"Cannot update a component while rendering a different component", "render_update_component"),
        ]
        
        # React-specific compilation patterns
        self.compilation_patterns = [
            (r"Module not found: Error: Can't resolve '(.+)'", "module_not_found"),
            (r"export '(.+)' \(imported as '(.+)'\) was not found in '(.+)'", "export_not_found"),
            (r"TypeError: Cannot read propert(?:y|ies) '(.+)' of undefined.*babel", "babel_compilation_error"),
        ]
    
    def parse_syntax_error(self, error_message: str, source_code: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Parse React JSX syntax errors."""
        # Check for JSX-specific syntax issues
        for pattern, error_type in [
            (r"Adjacent JSX elements must be wrapped in an enclosing tag", "jsx_adjacent_elements"),
            (r"Unexpected token '<'", "jsx_syntax_error"),
            (r"Expected corresponding JSX closing tag for <(.+)>", "jsx_unclosed_tag"),
        ]:
            if re.search(pattern, error_message):
                return {
                    "error_type": error_type,
                    "pattern": pattern,
                    "category": ErrorCategory.SYNTAX,
                    "language": self.language,
                    "framework": "react"
                }
        
        return None
    
    def parse_compilation_error(self, error_message: str, source_code: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Parse React compilation errors."""
        for pattern, error_type in self.compilation_patterns:
            match = re.search(pattern, error_message)
            if match:
                return {
                    "error_type": error_type,
                    "pattern": pattern,
                    "match_groups": match.groups(),
                    "category": ErrorCategory.COMPILATION,
                    "language": self.language,
                    "framework": "react"
                }
        
        return None
    
    def detect_runtime_issues(self, error_context: ErrorContext) -> List[Dict[str, Any]]:
        """Detect React runtime issues."""
        issues = []
        
        for pattern, error_type in self.react_patterns:
            match = re.search(pattern, error_context.error_message)
            if match:
                issues.append({
                    "error_type": error_type,
                    "pattern": pattern,
                    "match_groups": match.groups(),
                    "category": self._categorize_react_error(error_type),
                    "language": self.language,
                    "framework": "react"
                })
        
        return issues
    
    def _categorize_react_error(self, error_type: str) -> ErrorCategory:
        """Categorize React errors."""
        category_map = {
            # Hooks
            "invalid_hook_call": ErrorCategory.LOGIC,
            "useeffect_undefined": ErrorCategory.LOGIC,
            "missing_dependencies": ErrorCategory.LOGIC,
            "unnecessary_dependency": ErrorCategory.LOGIC,
            "conditional_hook": ErrorCategory.LOGIC,
            
            # Components
            "invalid_element_type": ErrorCategory.LOGIC,
            "render_undefined_property": ErrorCategory.LOGIC,
            "usestate_before_init": ErrorCategory.LOGIC,
            "missing_key_prop": ErrorCategory.PERFORMANCE,
            "function_component_ref": ErrorCategory.LOGIC,
            
            # State
            "setstate_unmounted": ErrorCategory.LOGIC,
            "setstate_undefined": ErrorCategory.LOGIC,
            "state_update_unmounted": ErrorCategory.LOGIC,
            
            # Redux
            "redux_missing_store": ErrorCategory.CONFIGURATION,
            "redux_invalid_action": ErrorCategory.LOGIC,
            "redux_undefined_reducer": ErrorCategory.LOGIC,
            
            # JSX
            "jsx_adjacent_elements": ErrorCategory.SYNTAX,
            "jsx_syntax_error": ErrorCategory.SYNTAX,
            "jsx_undefined_component": ErrorCategory.DEPENDENCY,
            
            # Props
            "proptypes_validation": ErrorCategory.LOGIC,
            "invalid_proptype": ErrorCategory.LOGIC,
            
            # Events
            "onclick_undefined": ErrorCategory.LOGIC,
            "onclick_not_function": ErrorCategory.LOGIC,
            
            # Performance
            "infinite_render_loop": ErrorCategory.PERFORMANCE,
            "render_update_component": ErrorCategory.LOGIC,
        }
        return category_map.get(error_type, ErrorCategory.RUNTIME)


class VueParser(LanguageSpecificParser):
    """Vue.js framework-specific error parser."""
    
    def __init__(self):
        super().__init__(LanguageType.JAVASCRIPT)
        
        # Vue-specific error patterns
        self.vue_patterns = [
            # Component errors
            (r"\[Vue warn\]: Property or method \"(.+)\" is not defined on the instance", "undefined_property"),
            (r"\[Vue warn\]: Component template should contain exactly one root element", "multiple_root_elements"),
            (r"\[Vue warn\]: Do not use v-for index as key on <(.+)> tag", "vfor_index_key"),
            (r"\[Vue warn\]: Failed to mount component: template or render function not defined", "missing_template"),
            
            # Composition API errors
            (r"Cannot call (.+) outside of setup\(\)", "composition_outside_setup"),
            (r"\[Vue warn\]: ref\(\) should not be used as a reactive property", "ref_as_reactive"),
            (r"\[Vue warn\]: watch\(\) source should be", "invalid_watch_source"),
            (r"Cannot read propert(?:y|ies) '(.+)' of undefined.*computed", "computed_undefined"),
            
            # Reactivity errors
            (r"\[Vue warn\]: Avoid mutating a prop directly", "prop_mutation"),
            (r"\[Vue warn\]: You are setting a non-existent reactive property", "nonexistent_reactive_property"),
            (r"TypeError: Cannot read propert(?:y|ies) '(.+)' of undefined.*reactive", "reactive_undefined"),
            
            # Directive errors
            (r"\[Vue warn\]: Failed to resolve directive: (.+)", "unknown_directive"),
            (r"\[Vue warn\]: v-model is not supported on element type: <(.+)>", "unsupported_vmodel"),
            (r"\[Vue warn\]: Invalid prop: type check failed for prop \"(.+)\"", "prop_type_check_failed"),
            
            # Vue Router errors
            (r"NavigationDuplicated: Avoided redundant navigation to current location", "duplicate_navigation"),
            (r"\[vue-router\] Route with name '(.+)' does not exist", "route_not_found"),
            (r"\[vue-router\] missing param for named route \"(.+)\": Expected \"(.+)\"", "missing_route_param"),
            
            # Vuex errors
            (r"\[vuex\] unknown action type: (.+)", "unknown_vuex_action"),
            (r"\[vuex\] unknown mutation type: (.+)", "unknown_vuex_mutation"),
            (r"\[vuex\] module namespace not found in mapState\(\): (.+)", "vuex_namespace_not_found"),
            
            # Lifecycle errors
            (r"TypeError: Cannot read propert(?:y|ies) '(.+)' of undefined.*mounted", "mounted_undefined"),
            (r"TypeError: Cannot read propert(?:y|ies) '(.+)' of undefined.*beforeDestroy", "beforedestroy_undefined"),
            
            # Template errors
            (r"\[Vue warn\]: Error in render: \"TypeError: Cannot read propert(?:y|ies) '(.+)' of undefined\"", "template_render_error"),
            (r"\[Vue warn\]: Unknown custom element: <(.+)>", "unknown_custom_element"),
        ]
        
        # Vue compilation patterns
        self.compilation_patterns = [
            (r"Module not found: Error: Can't resolve '(.+)\.vue'", "vue_component_not_found"),
            (r"Vue packages version mismatch", "vue_version_mismatch"),
            (r"Templates should only be responsible for mapping", "template_logic_error"),
        ]
    
    def parse_syntax_error(self, error_message: str, source_code: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Parse Vue template syntax errors."""
        vue_syntax_patterns = [
            (r"Invalid expression: (.+) in", "template_expression_error"),
            (r"- invalid expression: (.+)", "invalid_template_expression"),
            (r"Unexpected token", "template_syntax_error"),
        ]
        
        for pattern, error_type in vue_syntax_patterns:
            if re.search(pattern, error_message):
                return {
                    "error_type": error_type,
                    "pattern": pattern,
                    "category": ErrorCategory.SYNTAX,
                    "language": self.language,
                    "framework": "vue"
                }
        
        return None
    
    def parse_compilation_error(self, error_message: str, source_code: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Parse Vue compilation errors."""
        for pattern, error_type in self.compilation_patterns:
            match = re.search(pattern, error_message)
            if match:
                return {
                    "error_type": error_type,
                    "pattern": pattern,
                    "match_groups": match.groups(),
                    "category": ErrorCategory.COMPILATION,
                    "language": self.language,
                    "framework": "vue"
                }
        
        return None
    
    def detect_runtime_issues(self, error_context: ErrorContext) -> List[Dict[str, Any]]:
        """Detect Vue runtime issues."""
        issues = []
        
        for pattern, error_type in self.vue_patterns:
            match = re.search(pattern, error_context.error_message)
            if match:
                issues.append({
                    "error_type": error_type,
                    "pattern": pattern,
                    "match_groups": match.groups(),
                    "category": self._categorize_vue_error(error_type),
                    "language": self.language,
                    "framework": "vue"
                })
        
        return issues
    
    def _categorize_vue_error(self, error_type: str) -> ErrorCategory:
        """Categorize Vue errors."""
        category_map = {
            # Component
            "undefined_property": ErrorCategory.LOGIC,
            "multiple_root_elements": ErrorCategory.SYNTAX,
            "vfor_index_key": ErrorCategory.PERFORMANCE,
            "missing_template": ErrorCategory.CONFIGURATION,
            
            # Composition API
            "composition_outside_setup": ErrorCategory.LOGIC,
            "ref_as_reactive": ErrorCategory.LOGIC,
            "invalid_watch_source": ErrorCategory.LOGIC,
            "computed_undefined": ErrorCategory.LOGIC,
            
            # Reactivity
            "prop_mutation": ErrorCategory.LOGIC,
            "nonexistent_reactive_property": ErrorCategory.LOGIC,
            "reactive_undefined": ErrorCategory.LOGIC,
            
            # Directives
            "unknown_directive": ErrorCategory.CONFIGURATION,
            "unsupported_vmodel": ErrorCategory.LOGIC,
            "prop_type_check_failed": ErrorCategory.LOGIC,
            
            # Router
            "duplicate_navigation": ErrorCategory.LOGIC,
            "route_not_found": ErrorCategory.CONFIGURATION,
            "missing_route_param": ErrorCategory.LOGIC,
            
            # Vuex
            "unknown_vuex_action": ErrorCategory.CONFIGURATION,
            "unknown_vuex_mutation": ErrorCategory.CONFIGURATION,
            "vuex_namespace_not_found": ErrorCategory.CONFIGURATION,
            
            # Lifecycle
            "mounted_undefined": ErrorCategory.LOGIC,
            "beforedestroy_undefined": ErrorCategory.LOGIC,
            
            # Template
            "template_render_error": ErrorCategory.RUNTIME,
            "unknown_custom_element": ErrorCategory.CONFIGURATION,
        }
        return category_map.get(error_type, ErrorCategory.RUNTIME)


class AngularParser(LanguageSpecificParser):
    """Angular framework-specific error parser."""
    
    def __init__(self):
        super().__init__(LanguageType.TYPESCRIPT)
        
        # Angular-specific error patterns
        self.angular_patterns = [
            # Dependency Injection errors
            (r"No provider for (.+)!", "no_provider"),
            (r"Can't resolve all parameters for (.+): \(\?\)", "unresolved_parameters"),
            (r"CIRCULAR_DEPENDENCY_ERROR", "circular_dependency"),
            (r"StaticInjectorError\[(.+)\]: (.+)", "static_injector_error"),
            
            # Template errors
            (r"Cannot read propert(?:y|ies) '(.+)' of undefined.*template", "template_undefined_property"),
            (r"Cannot bind to '(.+)' since it isn't a known property of '(.+)'", "unknown_property_binding"),
            (r"Can't bind to '(.+)' since it isn't a known directive of '(.+)'", "unknown_directive"),
            (r"'(.+)' is not a known element", "unknown_element"),
            
            # Component errors
            (r"Component (.+) is not included in a module", "component_not_in_module"),
            (r"Unexpected value '(.+)' declared by the module '(.+)'", "unexpected_module_value"),
            (r"ExpressionChangedAfterItHasBeenCheckedError", "expression_changed_after_check"),
            
            # Routing errors
            (r"Cannot match any routes for URL: (.+)", "route_not_found"),
            (r"Cannot activate the '(.+)' route as the '(.+)' outlet is not defined", "outlet_not_defined"),
            (r"InvalidRouterStateError: Invalid router state", "invalid_router_state"),
            
            # HttpClient errors
            (r"Http failure response for (.+): (\d+) (.+)", "http_error"),
            (r"Http failure during parsing for (.+)", "http_parsing_error"),
            (r"Unknown Error occurred during HTTP request", "unknown_http_error"),
            
            # NgRx errors
            (r"ngrx: Action creator '(.+)' is not a function", "ngrx_invalid_action_creator"),
            (r"ngrx: The feature name \"(.+)\" does not exist", "ngrx_feature_not_found"),
            (r"ngrx: Reducer (.+) returned undefined", "ngrx_undefined_reducer"),
            
            # Forms errors
            (r"Cannot find control with name: '(.+)'", "form_control_not_found"),
            (r"Cannot find control with path: '(.+)'", "form_control_path_not_found"),
            (r"No value accessor for form control with (.+)", "no_value_accessor"),
            
            # Change Detection errors
            (r"ExpressionChangedAfterItHasBeenCheckedError: Expression has changed after it was checked", "change_detection_error"),
            (r"Cannot read propert(?:y|ies) '(.+)' of null.*ngOnChanges", "lifecycle_null_property"),
        ]
        
        # Angular compilation patterns
        self.compilation_patterns = [
            (r"NG\d+: (.+)", "angular_compiler_error"),
            (r"Cannot determine the module for class (.+)", "module_determination_error"),
            (r"Component '(.+)' is not included in a module", "component_module_error"),
            (r"Can't resolve all parameters for (.+)", "parameter_resolution_error"),
        ]
    
    def parse_syntax_error(self, error_message: str, source_code: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Parse Angular template syntax errors."""
        angular_syntax_patterns = [
            (r"Template parse errors:", "template_parse_error"),
            (r"Parser Error: (.+) at column (\d+)", "template_parser_error"),
            (r"Unexpected token (.+) at column (\d+)", "template_unexpected_token"),
        ]
        
        for pattern, error_type in angular_syntax_patterns:
            match = re.search(pattern, error_message)
            if match:
                return {
                    "error_type": error_type,
                    "pattern": pattern,
                    "match_groups": match.groups(),
                    "category": ErrorCategory.SYNTAX,
                    "language": self.language,
                    "framework": "angular"
                }
        
        return None
    
    def parse_compilation_error(self, error_message: str, source_code: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Parse Angular compilation errors."""
        for pattern, error_type in self.compilation_patterns:
            match = re.search(pattern, error_message)
            if match:
                return {
                    "error_type": error_type,
                    "pattern": pattern,
                    "match_groups": match.groups(),
                    "category": ErrorCategory.COMPILATION,
                    "language": self.language,
                    "framework": "angular"
                }
        
        return None
    
    def detect_runtime_issues(self, error_context: ErrorContext) -> List[Dict[str, Any]]:
        """Detect Angular runtime issues."""
        issues = []
        
        for pattern, error_type in self.angular_patterns:
            match = re.search(pattern, error_context.error_message)
            if match:
                issues.append({
                    "error_type": error_type,
                    "pattern": pattern,
                    "match_groups": match.groups(),
                    "category": self._categorize_angular_error(error_type),
                    "language": self.language,
                    "framework": "angular"
                })
        
        return issues
    
    def _categorize_angular_error(self, error_type: str) -> ErrorCategory:
        """Categorize Angular errors."""
        category_map = {
            # DI
            "no_provider": ErrorCategory.CONFIGURATION,
            "unresolved_parameters": ErrorCategory.CONFIGURATION,
            "circular_dependency": ErrorCategory.LOGIC,
            "static_injector_error": ErrorCategory.CONFIGURATION,
            
            # Template
            "template_undefined_property": ErrorCategory.LOGIC,
            "unknown_property_binding": ErrorCategory.CONFIGURATION,
            "unknown_directive": ErrorCategory.CONFIGURATION,
            "unknown_element": ErrorCategory.CONFIGURATION,
            
            # Component
            "component_not_in_module": ErrorCategory.CONFIGURATION,
            "unexpected_module_value": ErrorCategory.CONFIGURATION,
            "expression_changed_after_check": ErrorCategory.LOGIC,
            
            # Routing
            "route_not_found": ErrorCategory.CONFIGURATION,
            "outlet_not_defined": ErrorCategory.CONFIGURATION,
            "invalid_router_state": ErrorCategory.LOGIC,
            
            # HTTP
            "http_error": ErrorCategory.NETWORK,
            "http_parsing_error": ErrorCategory.LOGIC,
            "unknown_http_error": ErrorCategory.NETWORK,
            
            # NgRx
            "ngrx_invalid_action_creator": ErrorCategory.LOGIC,
            "ngrx_feature_not_found": ErrorCategory.CONFIGURATION,
            "ngrx_undefined_reducer": ErrorCategory.LOGIC,
            
            # Forms
            "form_control_not_found": ErrorCategory.CONFIGURATION,
            "form_control_path_not_found": ErrorCategory.CONFIGURATION,
            "no_value_accessor": ErrorCategory.CONFIGURATION,
            
            # Change Detection
            "change_detection_error": ErrorCategory.LOGIC,
            "lifecycle_null_property": ErrorCategory.LOGIC,
        }
        return category_map.get(error_type, ErrorCategory.RUNTIME)


class SvelteParser(LanguageSpecificParser):
    """Svelte framework-specific error parser."""
    
    def __init__(self):
        super().__init__(LanguageType.JAVASCRIPT)
        
        # Svelte-specific error patterns
        self.svelte_patterns = [
            # Component errors
            (r"'(.+)' is not defined.*svelte", "undefined_variable"),
            (r"Cannot read propert(?:y|ies) '(.+)' of undefined.*\$:", "reactive_statement_undefined"),
            (r"Component has unused export '(.+)'", "unused_export"),
            
            # Reactivity errors
            (r"'(.+)' is not a store", "invalid_store"),
            (r"Assignment to (.+) is invalid", "invalid_assignment"),
            (r"Cannot subscribe to (.+)", "invalid_subscription"),
            
            # Template errors
            (r"<(.+)> component has unused prop '(.+)'", "unused_prop"),
            (r"<(.+)> was created with unknown prop '(.+)'", "unknown_prop"),
            (r"A component cannot have a slot and a contents at the same time", "slot_content_conflict"),
            
            # Binding errors
            (r"Cannot bind to (.+)", "invalid_binding"),
            (r"Binding to (.+) is invalid", "binding_error"),
            (r"Two-way binding to (.+) is invalid", "invalid_two_way_binding"),
            
            # Lifecycle errors
            (r"onMount callback must be called during component initialisation", "onmount_timing"),
            (r"Cannot call (.+) outside of component initialisation", "lifecycle_outside_init"),
            
            # Transition errors
            (r"Transition (.+) is not defined", "undefined_transition"),
            (r"Action (.+) is not defined", "undefined_action"),
            (r"Cannot apply transition to (.+)", "invalid_transition_target"),
            
            # Store errors
            (r"Cannot subscribe to (.+) - not a store", "not_a_store"),
            (r"Store value must be an object", "invalid_store_value"),
            
            # SvelteKit specific errors
            (r"Not found: (.+)", "sveltekit_not_found"),
            (r"Cannot load (.+)", "sveltekit_load_error"),
            (r"Preload function (.+) must return a promise or plain object", "invalid_preload_return"),
        ]
        
        # Svelte compilation patterns
        self.compilation_patterns = [
            (r"ParseError: (.+)", "svelte_parse_error"),
            (r"ValidationError: (.+)", "svelte_validation_error"),
            (r"Unexpected character '(.+)'", "svelte_unexpected_character"),
        ]
    
    def parse_syntax_error(self, error_message: str, source_code: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Parse Svelte template syntax errors."""
        svelte_syntax_patterns = [
            (r"ParseError: (.+)", "parse_error"),
            (r"Unexpected character '(.+)' at position (\d+)", "unexpected_character"),
            (r"Expected (.+) but found (.+)", "syntax_expectation_error"),
        ]
        
        for pattern, error_type in svelte_syntax_patterns:
            match = re.search(pattern, error_message)
            if match:
                return {
                    "error_type": error_type,
                    "pattern": pattern,
                    "match_groups": match.groups(),
                    "category": ErrorCategory.SYNTAX,
                    "language": self.language,
                    "framework": "svelte"
                }
        
        return None
    
    def parse_compilation_error(self, error_message: str, source_code: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Parse Svelte compilation errors."""
        for pattern, error_type in self.compilation_patterns:
            match = re.search(pattern, error_message)
            if match:
                return {
                    "error_type": error_type,
                    "pattern": pattern,
                    "match_groups": match.groups(),
                    "category": ErrorCategory.COMPILATION,
                    "language": self.language,
                    "framework": "svelte"
                }
        
        return None
    
    def detect_runtime_issues(self, error_context: ErrorContext) -> List[Dict[str, Any]]:
        """Detect Svelte runtime issues."""
        issues = []
        
        for pattern, error_type in self.svelte_patterns:
            match = re.search(pattern, error_context.error_message)
            if match:
                issues.append({
                    "error_type": error_type,
                    "pattern": pattern,
                    "match_groups": match.groups(),
                    "category": self._categorize_svelte_error(error_type),
                    "language": self.language,
                    "framework": "svelte"
                })
        
        return issues
    
    def _categorize_svelte_error(self, error_type: str) -> ErrorCategory:
        """Categorize Svelte errors."""
        category_map = {
            # Component
            "undefined_variable": ErrorCategory.LOGIC,
            "reactive_statement_undefined": ErrorCategory.LOGIC,
            "unused_export": ErrorCategory.LOGIC,
            
            # Reactivity
            "invalid_store": ErrorCategory.LOGIC,
            "invalid_assignment": ErrorCategory.LOGIC,
            "invalid_subscription": ErrorCategory.LOGIC,
            
            # Template
            "unused_prop": ErrorCategory.LOGIC,
            "unknown_prop": ErrorCategory.LOGIC,
            "slot_content_conflict": ErrorCategory.LOGIC,
            
            # Binding
            "invalid_binding": ErrorCategory.LOGIC,
            "binding_error": ErrorCategory.LOGIC,
            "invalid_two_way_binding": ErrorCategory.LOGIC,
            
            # Lifecycle
            "onmount_timing": ErrorCategory.LOGIC,
            "lifecycle_outside_init": ErrorCategory.LOGIC,
            
            # Transitions
            "undefined_transition": ErrorCategory.CONFIGURATION,
            "undefined_action": ErrorCategory.CONFIGURATION,
            "invalid_transition_target": ErrorCategory.LOGIC,
            
            # Store
            "not_a_store": ErrorCategory.LOGIC,
            "invalid_store_value": ErrorCategory.LOGIC,
            
            # SvelteKit
            "sveltekit_not_found": ErrorCategory.CONFIGURATION,
            "sveltekit_load_error": ErrorCategory.RUNTIME,
            "invalid_preload_return": ErrorCategory.LOGIC,
        }
        return category_map.get(error_type, ErrorCategory.RUNTIME)


class NextJSParser(LanguageSpecificParser):
    """Next.js framework-specific error parser (extends React)."""
    
    def __init__(self):
        super().__init__(LanguageType.JAVASCRIPT)
        
        # Next.js specific error patterns
        self.nextjs_patterns = [
            # Data fetching errors
            (r"getStaticProps cannot be used with getServerSideProps", "conflicting_data_fetching"),
            (r"getStaticPaths is required for dynamic SSG pages", "missing_static_paths"),
            (r"getStaticPaths did not return a paths array", "invalid_static_paths"),
            (r"notFound.*getStaticProps", "static_not_found"),
            
            # API route errors
            (r"API resolved without sending a response", "api_no_response"),
            (r"Cannot set headers after they are sent", "headers_already_sent"),
            (r"API route (.+) does not export a default function", "api_no_default_export"),
            
            # Routing errors
            (r"Dynamic route (.+) has more than one wildcard", "multiple_wildcards"),
            (r"Dynamic route (.+) cannot be used alongside a non-dynamic route", "dynamic_static_conflict"),
            (r"Cannot read propert(?:y|ies) '(.+)' of undefined.*router", "router_undefined"),
            
            # Image optimization errors
            (r"Invalid src prop (.+) on `next/image`", "invalid_image_src"),
            (r"next/image (.+) should be imported from", "incorrect_image_import"),
            
            # Build and compilation errors
            (r"Module not found: Can't resolve '(.+)' in '(.+)'", "module_resolution_error"),
            (r"webpack: Compilation failed", "webpack_compilation_failed"),
            (r"Error occurred prerendering page \"(.+)\"", "prerender_error"),
            
            # Middleware errors
            (r"Middleware (.+) must export a middleware function", "middleware_no_export"),
            (r"The edge runtime does not support (.+)", "edge_runtime_unsupported"),
            
            # App Router errors (Next.js 13+)
            (r"Cannot use (.+) in Server Components", "server_component_restriction"),
            (r"(.+) only works in Client Components", "client_component_only"),
            (r"Invalid segment \"(.+)\" in route", "invalid_route_segment"),
        ]
        
        # Next.js compilation patterns
        self.compilation_patterns = [
            (r"Build error occurred", "nextjs_build_error"),
            (r"Failed to compile", "nextjs_compilation_failed"),
            (r"Module parse failed: (.+)", "module_parse_failed"),
        ]
    
    def parse_syntax_error(self, error_message: str, source_code: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Parse Next.js syntax errors (delegates to React for JSX)."""
        # First check for Next.js specific syntax issues
        nextjs_syntax_patterns = [
            (r"Invalid segment \"(.+)\" in route", "invalid_route_segment"),
            (r"Dynamic route (.+) has invalid syntax", "invalid_dynamic_route_syntax"),
        ]
        
        for pattern, error_type in nextjs_syntax_patterns:
            match = re.search(pattern, error_message)
            if match:
                return {
                    "error_type": error_type,
                    "pattern": pattern,
                    "match_groups": match.groups(),
                    "category": ErrorCategory.SYNTAX,
                    "language": self.language,
                    "framework": "nextjs"
                }
        
        # Fallback to React parser for JSX syntax
        react_parser = ReactParser()
        result = react_parser.parse_syntax_error(error_message, source_code)
        if result:
            result["framework"] = "nextjs"
        return result
    
    def parse_compilation_error(self, error_message: str, source_code: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Parse Next.js compilation errors."""
        for pattern, error_type in self.compilation_patterns:
            match = re.search(pattern, error_message)
            if match:
                return {
                    "error_type": error_type,
                    "pattern": pattern,
                    "match_groups": match.groups(),
                    "category": ErrorCategory.COMPILATION,
                    "language": self.language,
                    "framework": "nextjs"
                }
        
        return None
    
    def detect_runtime_issues(self, error_context: ErrorContext) -> List[Dict[str, Any]]:
        """Detect Next.js runtime issues."""
        issues = []
        
        # Check Next.js specific patterns
        for pattern, error_type in self.nextjs_patterns:
            match = re.search(pattern, error_context.error_message)
            if match:
                issues.append({
                    "error_type": error_type,
                    "pattern": pattern,
                    "match_groups": match.groups(),
                    "category": self._categorize_nextjs_error(error_type),
                    "language": self.language,
                    "framework": "nextjs"
                })
        
        # Also check for React issues since Next.js extends React
        react_parser = ReactParser()
        react_issues = react_parser.detect_runtime_issues(error_context)
        for issue in react_issues:
            issue["framework"] = "nextjs"  # Mark as Next.js context
            issues.append(issue)
        
        return issues
    
    def _categorize_nextjs_error(self, error_type: str) -> ErrorCategory:
        """Categorize Next.js errors."""
        category_map = {
            # Data fetching
            "conflicting_data_fetching": ErrorCategory.LOGIC,
            "missing_static_paths": ErrorCategory.CONFIGURATION,
            "invalid_static_paths": ErrorCategory.LOGIC,
            "static_not_found": ErrorCategory.LOGIC,
            
            # API routes
            "api_no_response": ErrorCategory.LOGIC,
            "headers_already_sent": ErrorCategory.LOGIC,
            "api_no_default_export": ErrorCategory.CONFIGURATION,
            
            # Routing
            "multiple_wildcards": ErrorCategory.CONFIGURATION,
            "dynamic_static_conflict": ErrorCategory.CONFIGURATION,
            "router_undefined": ErrorCategory.LOGIC,
            
            # Images
            "invalid_image_src": ErrorCategory.LOGIC,
            "incorrect_image_import": ErrorCategory.DEPENDENCY,
            
            # Build
            "module_resolution_error": ErrorCategory.DEPENDENCY,
            "webpack_compilation_failed": ErrorCategory.COMPILATION,
            "prerender_error": ErrorCategory.RUNTIME,
            
            # Middleware
            "middleware_no_export": ErrorCategory.CONFIGURATION,
            "edge_runtime_unsupported": ErrorCategory.ENVIRONMENT,
            
            # App Router
            "server_component_restriction": ErrorCategory.LOGIC,
            "client_component_only": ErrorCategory.LOGIC,
            "invalid_route_segment": ErrorCategory.CONFIGURATION,
        }
        return category_map.get(error_type, ErrorCategory.RUNTIME)


class EmberParser(LanguageSpecificParser):
    """Ember.js framework-specific error parser."""
    
    def __init__(self):
        super().__init__(LanguageType.JAVASCRIPT)
        
        # Ember-specific error patterns
        self.ember_patterns = [
            # Component errors
            (r"Cannot read propert(?:y|ies) '(.+)' of undefined.*component", "component_undefined_property"),
            (r"Component (.+) is not defined", "undefined_component"),
            (r"You must pass a component name to the component helper", "missing_component_name"),
            
            # Template errors
            (r"An error occurred while compiling the template (.+)", "template_compilation_error"),
            (r"Cannot use (.+) as a modifier", "invalid_modifier"),
            (r"(.+) is not a helper", "invalid_helper"),
            
            # Model and data errors
            (r"Cannot read propert(?:y|ies) '(.+)' of undefined.*model", "model_undefined_property"),
            (r"Ember Data expected the primary data returned (.+) to be an object", "invalid_ember_data"),
            (r"No model was found for '(.+)'", "model_not_found"),
            
            # Router errors
            (r"There is no route named '(.+)'", "route_not_found"),
            (r"Cannot read propert(?:y|ies) '(.+)' of undefined.*transition", "transition_undefined"),
            (r"More context objects were passed than there are dynamic segments", "excess_route_context"),
            
            # Service errors
            (r"Cannot read propert(?:y|ies) '(.+)' of undefined.*service", "service_undefined"),
            (r"Attempting to inject an unknown injection: '(.+)'", "unknown_injection"),
            
            # Build errors
            (r"Build failed", "ember_build_failed"),
            (r"Missing addon (.+)", "missing_addon"),
            (r"Cannot resolve dependency '(.+)'", "dependency_resolution_error"),
        ]
        
        # Ember compilation patterns
        self.compilation_patterns = [
            (r"Template Compiler Error in (.+)", "template_compiler_error"),
            (r"Build Error: (.+)", "ember_build_error"),
            (r"Syntax error in (.+)\.hbs", "handlebars_syntax_error"),
        ]
    
    def parse_syntax_error(self, error_message: str, source_code: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Parse Ember template syntax errors."""
        ember_syntax_patterns = [
            (r"Syntax error in (.+)\.hbs: (.+)", "handlebars_syntax_error"),
            (r"Parse error on line (\d+): (.+)", "template_parse_error"),
            (r"Expecting (.+), got '(.+)'", "template_expectation_error"),
        ]
        
        for pattern, error_type in ember_syntax_patterns:
            match = re.search(pattern, error_message)
            if match:
                return {
                    "error_type": error_type,
                    "pattern": pattern,
                    "match_groups": match.groups(),
                    "category": ErrorCategory.SYNTAX,
                    "language": self.language,
                    "framework": "ember"
                }
        
        return None
    
    def parse_compilation_error(self, error_message: str, source_code: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Parse Ember compilation errors."""
        for pattern, error_type in self.compilation_patterns:
            match = re.search(pattern, error_message)
            if match:
                return {
                    "error_type": error_type,
                    "pattern": pattern,
                    "match_groups": match.groups(),
                    "category": ErrorCategory.COMPILATION,
                    "language": self.language,
                    "framework": "ember"
                }
        
        return None
    
    def detect_runtime_issues(self, error_context: ErrorContext) -> List[Dict[str, Any]]:
        """Detect Ember runtime issues."""
        issues = []
        
        for pattern, error_type in self.ember_patterns:
            match = re.search(pattern, error_context.error_message)
            if match:
                issues.append({
                    "error_type": error_type,
                    "pattern": pattern,
                    "match_groups": match.groups(),
                    "category": self._categorize_ember_error(error_type),
                    "language": self.language,
                    "framework": "ember"
                })
        
        return issues
    
    def _categorize_ember_error(self, error_type: str) -> ErrorCategory:
        """Categorize Ember errors."""
        category_map = {
            # Component
            "component_undefined_property": ErrorCategory.LOGIC,
            "undefined_component": ErrorCategory.CONFIGURATION,
            "missing_component_name": ErrorCategory.LOGIC,
            
            # Template
            "template_compilation_error": ErrorCategory.COMPILATION,
            "invalid_modifier": ErrorCategory.CONFIGURATION,
            "invalid_helper": ErrorCategory.CONFIGURATION,
            
            # Model
            "model_undefined_property": ErrorCategory.LOGIC,
            "invalid_ember_data": ErrorCategory.LOGIC,
            "model_not_found": ErrorCategory.CONFIGURATION,
            
            # Router
            "route_not_found": ErrorCategory.CONFIGURATION,
            "transition_undefined": ErrorCategory.LOGIC,
            "excess_route_context": ErrorCategory.LOGIC,
            
            # Service
            "service_undefined": ErrorCategory.CONFIGURATION,
            "unknown_injection": ErrorCategory.CONFIGURATION,
            
            # Build
            "ember_build_failed": ErrorCategory.COMPILATION,
            "missing_addon": ErrorCategory.DEPENDENCY,
            "dependency_resolution_error": ErrorCategory.DEPENDENCY,
        }
        return category_map.get(error_type, ErrorCategory.RUNTIME)


class WebComponentsParser(LanguageSpecificParser):
    """Web Components standard-specific error parser."""
    
    def __init__(self):
        super().__init__(LanguageType.JAVASCRIPT)
        
        # Web Components specific error patterns
        self.webcomponents_patterns = [
            # Custom Elements errors
            (r"Failed to construct 'CustomElementRegistry': (.+)", "custom_element_registry_error"),
            (r"Illegal constructor", "illegal_constructor"),
            (r"Cannot define (.+): this name has already been used", "duplicate_element_name"),
            (r"The result must not have children", "autonomous_element_children"),
            
            # Shadow DOM errors
            (r"Failed to execute 'attachShadow' on 'Element': (.+)", "attach_shadow_failed"),
            (r"Cannot read propert(?:y|ies) '(.+)' of (.+)shadowRoot", "shadow_root_undefined"),
            (r"Slotted content (.+) not found", "slot_content_not_found"),
            
            # Template errors
            (r"Cannot clone template: (.+)", "template_clone_error"),
            (r"Template content is not defined", "template_content_undefined"),
            (r"HTMLTemplateElement (.+) not found", "template_element_not_found"),
            
            # Lifecycle errors
            (r"Cannot call (.+) before element is connected", "lifecycle_before_connected"),
            (r"attributeChangedCallback called with undefined (.+)", "attribute_callback_undefined"),
            (r"observedAttributes must return an array", "invalid_observed_attributes"),
            
            # Module errors
            (r"Cannot import (.+) as a module", "module_import_error"),
            (r"Dynamic import of (.+) failed", "dynamic_import_failed"),
            
            # Polyfill errors
            (r"Web Components polyfill not loaded", "polyfill_not_loaded"),
            (r"customElements is not defined", "custom_elements_not_supported"),
        ]
        
        # Web Components compilation patterns
        self.compilation_patterns = [
            (r"Invalid Web Component definition: (.+)", "invalid_component_definition"),
            (r"Web Component (.+) cannot be compiled", "component_compilation_failed"),
            (r"Module resolution failed for (.+)", "module_resolution_failed"),
        ]
    
    def parse_syntax_error(self, error_message: str, source_code: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Parse Web Components syntax errors."""
        webcomponents_syntax_patterns = [
            (r"Invalid Web Component syntax: (.+)", "invalid_syntax"),
            (r"Template syntax error: (.+)", "template_syntax_error"),
            (r"Custom element name (.+) is invalid", "invalid_element_name"),
        ]
        
        for pattern, error_type in webcomponents_syntax_patterns:
            match = re.search(pattern, error_message)
            if match:
                return {
                    "error_type": error_type,
                    "pattern": pattern,
                    "match_groups": match.groups(),
                    "category": ErrorCategory.SYNTAX,
                    "language": self.language,
                    "framework": "webcomponents"
                }
        
        return None
    
    def parse_compilation_error(self, error_message: str, source_code: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Parse Web Components compilation errors."""
        for pattern, error_type in self.compilation_patterns:
            match = re.search(pattern, error_message)
            if match:
                return {
                    "error_type": error_type,
                    "pattern": pattern,
                    "match_groups": match.groups(),
                    "category": ErrorCategory.COMPILATION,
                    "language": self.language,
                    "framework": "webcomponents"
                }
        
        return None
    
    def detect_runtime_issues(self, error_context: ErrorContext) -> List[Dict[str, Any]]:
        """Detect Web Components runtime issues."""
        issues = []
        
        for pattern, error_type in self.webcomponents_patterns:
            match = re.search(pattern, error_context.error_message)
            if match:
                issues.append({
                    "error_type": error_type,
                    "pattern": pattern,
                    "match_groups": match.groups(),
                    "category": self._categorize_webcomponents_error(error_type),
                    "language": self.language,
                    "framework": "webcomponents"
                })
        
        return issues
    
    def _categorize_webcomponents_error(self, error_type: str) -> ErrorCategory:
        """Categorize Web Components errors."""
        category_map = {
            # Custom Elements
            "custom_element_registry_error": ErrorCategory.RUNTIME,
            "illegal_constructor": ErrorCategory.LOGIC,
            "duplicate_element_name": ErrorCategory.LOGIC,
            "autonomous_element_children": ErrorCategory.LOGIC,
            
            # Shadow DOM
            "attach_shadow_failed": ErrorCategory.RUNTIME,
            "shadow_root_undefined": ErrorCategory.LOGIC,
            "slot_content_not_found": ErrorCategory.LOGIC,
            
            # Template
            "template_clone_error": ErrorCategory.RUNTIME,
            "template_content_undefined": ErrorCategory.LOGIC,
            "template_element_not_found": ErrorCategory.CONFIGURATION,
            
            # Lifecycle
            "lifecycle_before_connected": ErrorCategory.LOGIC,
            "attribute_callback_undefined": ErrorCategory.LOGIC,
            "invalid_observed_attributes": ErrorCategory.LOGIC,
            
            # Module
            "module_import_error": ErrorCategory.DEPENDENCY,
            "dynamic_import_failed": ErrorCategory.DEPENDENCY,
            
            # Polyfill
            "polyfill_not_loaded": ErrorCategory.ENVIRONMENT,
            "custom_elements_not_supported": ErrorCategory.ENVIRONMENT,
        }
        return category_map.get(error_type, ErrorCategory.RUNTIME)


# Factory function for creating web framework parsers
def create_web_framework_parser(framework: str) -> Optional[LanguageSpecificParser]:
    """
    Create a web framework-specific parser.
    
    Args:
        framework: Framework name (react, vue, angular, svelte, nextjs, ember, webcomponents)
        
    Returns:
        Framework parser instance or None if not supported
    """
    framework_parsers = {
        "react": ReactParser,
        "vue": VueParser,
        "angular": AngularParser,
        "svelte": SvelteParser,
        "nextjs": NextJSParser,
        "ember": EmberParser,
        "webcomponents": WebComponentsParser,
    }
    
    parser_class = framework_parsers.get(framework.lower())
    if parser_class:
        try:
            return parser_class()
        except Exception as e:
            logger.error(f"Error creating {framework} parser: {e}")
    
    return None


if __name__ == "__main__":
    # Test the web framework parsers
    print("Web Framework Parsers Test")
    print("=========================")
    
    # Test React parser
    react_parser = ReactParser()
    react_error = "Invalid hook call. Hooks can only be called inside the body of a function component"
    react_context = ErrorContext(error_message=react_error, language=LanguageType.JAVASCRIPT)
    react_issues = react_parser.detect_runtime_issues(react_context)
    print("\nReact Runtime Issues:")
    print(f"Issues: {react_issues}")
    
    # Test Vue parser
    vue_parser = VueParser()
    vue_error = "[Vue warn]: Property or method \"myProperty\" is not defined on the instance"
    vue_context = ErrorContext(error_message=vue_error, language=LanguageType.JAVASCRIPT)
    vue_issues = vue_parser.detect_runtime_issues(vue_context)
    print("\nVue Runtime Issues:")
    print(f"Issues: {vue_issues}")
    
    # Test Angular parser
    angular_parser = AngularParser()
    angular_error = "No provider for HttpClient!"
    angular_context = ErrorContext(error_message=angular_error, language=LanguageType.TYPESCRIPT)
    angular_issues = angular_parser.detect_runtime_issues(angular_context)
    print("\nAngular Runtime Issues:")
    print(f"Issues: {angular_issues}")
    
    # Test Svelte parser
    svelte_parser = SvelteParser()
    svelte_error = "'myVariable' is not defined"
    svelte_context = ErrorContext(error_message=svelte_error, language=LanguageType.JAVASCRIPT)
    svelte_issues = svelte_parser.detect_runtime_issues(svelte_context)
    print("\nSvelte Runtime Issues:")
    print(f"Issues: {svelte_issues}")
    
    # Test Next.js parser
    nextjs_parser = NextJSParser()
    nextjs_error = "getStaticProps cannot be used with getServerSideProps"
    nextjs_context = ErrorContext(error_message=nextjs_error, language=LanguageType.JAVASCRIPT)
    nextjs_issues = nextjs_parser.detect_runtime_issues(nextjs_context)
    print("\nNext.js Runtime Issues:")
    print(f"Issues: {nextjs_issues}")
    
    # Test Ember parser
    ember_parser = EmberParser()
    ember_error = "There is no route named 'nonexistent'"
    ember_context = ErrorContext(error_message=ember_error, language=LanguageType.JAVASCRIPT)
    ember_issues = ember_parser.detect_runtime_issues(ember_context)
    print("\nEmber Runtime Issues:")
    print(f"Issues: {ember_issues}")
    
    # Test Web Components parser
    webcomponents_parser = WebComponentsParser()
    webcomponents_error = "Failed to construct 'CustomElementRegistry': this name has already been used"
    webcomponents_context = ErrorContext(error_message=webcomponents_error, language=LanguageType.JAVASCRIPT)
    webcomponents_issues = webcomponents_parser.detect_runtime_issues(webcomponents_context)
    print("\nWeb Components Runtime Issues:")
    print(f"Issues: {webcomponents_issues}")
    
    # Test factory function
    print("\nTesting Factory Function:")
    for framework in ["react", "vue", "angular", "svelte", "nextjs", "ember", "webcomponents"]:
        parser = create_web_framework_parser(framework)
        print(f"{framework}: {'✓' if parser else '✗'} {type(parser).__name__ if parser else 'None'}")