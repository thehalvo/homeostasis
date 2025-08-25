"""
Angular Framework Plugin for Homeostasis

This plugin enables Homeostasis to analyze and fix errors in Angular applications.
It provides comprehensive error handling for Angular components, dependency injection,
NgRx state management, template binding, module loading, and Angular Universal SSR.
"""
import logging
import re
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union, Set

from ..language_plugin_system import LanguagePlugin, register_plugin
from ..language_adapters import JavaScriptErrorAdapter

logger = logging.getLogger(__name__)


class AngularExceptionHandler:
    """
    Handles Angular-specific exceptions with comprehensive error detection and classification.
    
    This class provides logic for categorizing Angular component errors, dependency injection
    issues, NgRx state management problems, template binding errors, and SSR-related issues.
    """
    
    def __init__(self):
        """Initialize the Angular exception handler."""
        self.rule_categories = {
            "dependency_injection": "Angular dependency injection errors",
            "components": "Angular component related errors",
            "templates": "Template binding and syntax errors",
            "ngrx": "NgRx state management errors",
            "modules": "Angular module and lazy loading errors",
            "services": "Angular service errors",
            "routing": "Angular Router navigation errors",
            "forms": "Angular Forms (reactive/template-driven) errors",
            "pipes": "Angular pipe errors",
            "directives": "Angular directive errors",
            "lifecycle": "Component lifecycle errors",
            "ssr": "Angular Universal SSR errors",
            "animation": "Angular animation errors",
            "testing": "Angular testing errors"
        }
        
        # Load rules from different categories
        self.rules = self._load_rules()
        
        # Pre-compile regex patterns for better performance
        self._compile_patterns()
    
    def _load_rules(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load Angular error rules from rule files."""
        rules = {}
        rules_dir = Path(__file__).parent.parent / "rules" / "angular"
        
        try:
            # Load common Angular rules
            common_rules_path = rules_dir / "angular_common_errors.json"
            if common_rules_path.exists():
                with open(common_rules_path, 'r') as f:
                    common_data = json.load(f)
                    rules["common"] = common_data.get("rules", [])
                    logger.info(f"Loaded {len(rules['common'])} common Angular rules")
            
            # Load dependency injection rules
            di_rules_path = rules_dir / "angular_dependency_injection_errors.json"
            if di_rules_path.exists():
                with open(di_rules_path, 'r') as f:
                    di_data = json.load(f)
                    rules["dependency_injection"] = di_data.get("rules", [])
                    logger.info(f"Loaded {len(rules['dependency_injection'])} Angular DI rules")
            
            # Load NgRx rules
            ngrx_rules_path = rules_dir / "angular_ngrx_errors.json"
            if ngrx_rules_path.exists():
                with open(ngrx_rules_path, 'r') as f:
                    ngrx_data = json.load(f)
                    rules["ngrx"] = ngrx_data.get("rules", [])
                    logger.info(f"Loaded {len(rules['ngrx'])} Angular NgRx rules")
                    
        except Exception as e:
            logger.error(f"Error loading Angular rules: {e}")
            rules = {"common": [], "dependency_injection": [], "ngrx": []}
        
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
                    logger.warning(f"Invalid regex pattern in Angular rule {rule.get('id', 'unknown')}: {e}")
    
    def analyze_exception(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze an Angular exception and determine its type and potential fixes.
        
        Args:
            error_data: Angular error data in standard format
            
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
                "category": best_match.get("category", "angular"),
                "subcategory": best_match.get("subcategory", "unknown"),
                "confidence": best_match.get("confidence", "medium"),
                "suggested_fix": best_match.get("suggestion", ""),
                "root_cause": best_match.get("root_cause", ""),
                "severity": best_match.get("severity", "medium"),
                "rule_id": best_match.get("id", ""),
                "tags": best_match.get("tags", []),
                "fix_commands": best_match.get("fix_commands", []),
                "all_matches": matches
            }
        
        # If no rules matched, provide generic analysis
        return self._generic_analysis(error_data)
    
    def _find_matching_rules(self, error_text: str, error_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find all rules that match the given error."""
        matches = []
        
        for category, patterns in self.compiled_patterns.items():
            for compiled_pattern, rule in patterns:
                match = compiled_pattern.search(error_text)
                if match:
                    # Calculate confidence score based on match quality
                    confidence_score = self._calculate_confidence(match, rule, error_data)
                    
                    match_info = rule.copy()
                    match_info["confidence_score"] = confidence_score
                    match_info["match_groups"] = match.groups() if match.groups() else []
                    matches.append(match_info)
        
        return matches
    
    def _calculate_confidence(self, match: re.Match, rule: Dict[str, Any], 
                             error_data: Dict[str, Any]) -> float:
        """Calculate confidence score for a rule match."""
        base_confidence = 0.5
        
        # Boost confidence for Angular-specific patterns
        message = error_data.get("message", "").lower()
        if "angular" in message or "ng" in message or "dependency injection" in message:
            base_confidence += 0.3
        
        # Boost confidence based on rule reliability
        reliability = rule.get("reliability", "medium")
        reliability_boost = {"high": 0.2, "medium": 0.1, "low": 0.0}
        base_confidence += reliability_boost.get(reliability, 0.0)
        
        # Boost confidence for rules with specific tags that match context
        rule_tags = set(rule.get("tags", []))
        context_tags = set()
        
        # Infer context from error data
        if "angular" in error_data.get("framework", "").lower():
            context_tags.add("angular")
        if "ngrx" in message:
            context_tags.add("ngrx")
        if "dependency" in message or "injection" in message:
            context_tags.add("dependency-injection")
        if "template" in message or "binding" in message:
            context_tags.add("template")
        if "component" in message:
            context_tags.add("component")
        
        if context_tags & rule_tags:
            base_confidence += 0.1
        
        return min(base_confidence, 1.0)
    
    def _generic_analysis(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """Provide generic analysis for unmatched errors."""
        error_type = error_data.get("error_type", "Error")
        message = error_data.get("message", "").lower()
        
        # Basic categorization based on error patterns
        if "dependency" in message or "injection" in message or "injector" in message:
            category = "dependency_injection"
            suggestion = "Check Angular dependency injection - ensure providers are registered and tokens are correct"
        elif "ngrx" in message or "store" in message or "effect" in message:
            category = "ngrx"
            suggestion = "Check NgRx store configuration and action dispatching"
        elif "template" in message or "binding" in message:
            category = "templates"
            suggestion = "Check Angular template syntax and property binding"
        elif "module" in message or "lazy" in message:
            category = "modules"
            suggestion = "Check Angular module configuration and lazy loading setup"
        elif "component" in message:
            category = "components"
            suggestion = "Check Angular component definition and lifecycle"
        elif "router" in message or "navigation" in message:
            category = "routing"
            suggestion = "Check Angular Router configuration and navigation"
        elif "form" in message or "control" in message:
            category = "forms"
            suggestion = "Check Angular Forms configuration and validation"
        elif "pipe" in message:
            category = "pipes"
            suggestion = "Check Angular pipe implementation and usage"
        elif "directive" in message:
            category = "directives"
            suggestion = "Check Angular directive implementation"
        else:
            category = "unknown"
            suggestion = "Review Angular application implementation"
        
        return {
            "category": "angular",
            "subcategory": category,
            "confidence": "low",
            "suggested_fix": suggestion,
            "root_cause": f"angular_{category}_error",
            "severity": "medium",
            "rule_id": "angular_generic_handler",
            "tags": ["angular", "generic", category]
        }
    
    def analyze_dependency_injection_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze Angular dependency injection specific errors.
        
        Args:
            error_data: Error data with DI-related issues
            
        Returns:
            Analysis results with DI-specific fixes
        """
        message = error_data.get("message", "")
        
        # Common DI error patterns
        di_patterns = {
            "no provider for": {
                "cause": "angular_no_provider",
                "fix": "Add the service to providers array in module or component",
                "severity": "error"
            },
            "injectiontoken": {
                "cause": "angular_injection_token_error",
                "fix": "Provide value for InjectionToken or check token usage",
                "severity": "error"
            },
            "circular dependency": {
                "cause": "angular_circular_dependency",
                "fix": "Remove circular dependencies between services or use forwardRef()",
                "severity": "error"
            },
            "cannot resolve all parameters": {
                "cause": "angular_cannot_resolve_parameters",
                "fix": "Add @Injectable() decorator and ensure all dependencies are available",
                "severity": "error"
            },
            "invalid provider": {
                "cause": "angular_invalid_provider",
                "fix": "Check provider configuration in module or component",
                "severity": "error"
            },
            "injector not found": {
                "cause": "angular_injector_not_found",
                "fix": "Ensure component is properly instantiated within Angular context",
                "severity": "error"
            }
        }
        
        for pattern, info in di_patterns.items():
            if pattern in message.lower():
                return {
                    "category": "angular",
                    "subcategory": "dependency_injection",
                    "confidence": "high",
                    "suggested_fix": info["fix"],
                    "root_cause": info["cause"],
                    "severity": info["severity"],
                    "tags": ["angular", "dependency-injection", "di"]
                }
        
        # Generic DI error
        return {
            "category": "angular",
            "subcategory": "dependency_injection",
            "confidence": "medium",
            "suggested_fix": "Check Angular dependency injection configuration",
            "root_cause": "angular_di_error",
            "severity": "warning",
            "tags": ["angular", "dependency-injection"]
        }
    
    def analyze_ngrx_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze NgRx state management errors.
        
        Args:
            error_data: Error data with NgRx-related issues
            
        Returns:
            Analysis results with NgRx-specific fixes
        """
        message = error_data.get("message", "").lower()
        stack_trace = str(error_data.get("stack_trace", "")).lower()
        
        # NgRx specific error patterns
        ngrx_patterns = {
            "action must have a type": {
                "cause": "ngrx_action_no_type",
                "fix": "Ensure all NgRx actions have a 'type' property. Use createAction() from @ngrx/store to define actions with type safety",
                "severity": "error"
            },
            "store has not been provided": {
                "cause": "ngrx_store_not_provided",
                "fix": "Import StoreModule.forRoot() in your AppModule",
                "severity": "error"
            },
            "effects must be an array": {
                "cause": "ngrx_effects_not_array",
                "fix": "Ensure EffectsModule.forRoot() receives an array of effect classes",
                "severity": "error"
            },
            "selector function cannot return undefined": {
                "cause": "ngrx_selector_undefined",
                "fix": "Ensure selectors return a default value and handle undefined state",
                "severity": "error"
            },
            "reducer returned undefined": {
                "cause": "ngrx_reducer_undefined",
                "fix": "Ensure reducer always returns a state object, never undefined",
                "severity": "error"
            },
            "effect dispatched invalid action": {
                "cause": "ngrx_effect_invalid_action",
                "fix": "Ensure effects return valid action objects",
                "severity": "error"
            },
            "feature state not found": {
                "cause": "ngrx_feature_state_not_found",
                "fix": "Register feature state with StoreModule.forFeature()",
                "severity": "error"
            }
        }
        
        for pattern, info in ngrx_patterns.items():
            if pattern in message:
                return {
                    "category": "angular",
                    "subcategory": "ngrx",
                    "confidence": "high",
                    "suggested_fix": info["fix"],
                    "root_cause": info["cause"],
                    "severity": info["severity"],
                    "tags": ["angular", "ngrx", "state-management"]
                }
        
        # Generic NgRx error
        return {
            "category": "angular",
            "subcategory": "ngrx",
            "confidence": "medium",
            "suggested_fix": "Check NgRx store configuration and usage",
            "root_cause": "ngrx_general_error",
            "severity": "medium",
            "tags": ["angular", "ngrx"]
        }
    
    def analyze_template_binding_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze Angular template binding errors.
        
        Args:
            error_data: Error data with template binding issues
            
        Returns:
            Analysis results with template-specific fixes
        """
        message = error_data.get("message", "").lower()
        
        # Template binding error patterns
        template_patterns = {
            "cannot read property": {
                "cause": "angular_template_property_undefined",
                "fix": "Use safe navigation operator (?.) or *ngIf to check for undefined properties",
                "severity": "error"
            },
            "cannot bind to": {
                "cause": "angular_invalid_property_binding",
                "fix": "Check property binding syntax or add input property to component",
                "severity": "error"
            },
            "is not a known element": {
                "cause": "angular_unknown_element",
                "fix": "Import component module or check component selector",
                "severity": "error"
            },
            "is not a known property": {
                "cause": "angular_unknown_property",
                "fix": "Check property name or add @Input() decorator to component property",
                "severity": "error"
            },
            "expression has changed after it was checked": {
                "cause": "angular_expression_changed_after_checked",
                "fix": "Move logic to ngAfterViewInit or use setTimeout/Promise.resolve",
                "severity": "warning"
            },
            "cannot resolve all parameters for": {
                "cause": "angular_cannot_resolve_component_parameters",
                "fix": "Check component constructor dependencies and ensure they are provided",
                "severity": "error"
            }
        }
        
        for pattern, info in template_patterns.items():
            if pattern in message:
                return {
                    "category": "angular",
                    "subcategory": "templates",
                    "confidence": "high",
                    "suggested_fix": info["fix"],
                    "root_cause": info["cause"],
                    "severity": info["severity"],
                    "tags": ["angular", "template", "binding"]
                }
        
        # Generic template error
        return {
            "category": "angular",
            "subcategory": "templates",
            "confidence": "medium",
            "suggested_fix": "Check Angular template syntax and property bindings",
            "root_cause": "angular_template_error",
            "severity": "medium",
            "tags": ["angular", "template"]
        }


class AngularPatchGenerator:
    """
    Generates patches for Angular errors based on analysis results.
    
    This class creates code fixes for common Angular errors using templates
    and heuristics specific to Angular patterns and best practices.
    """
    
    def __init__(self):
        """Initialize the Angular patch generator."""
        self.template_dir = Path(__file__).parent.parent / "patch_generation" / "templates"
        self.angular_template_dir = self.template_dir / "angular"
        
        # Ensure template directory exists
        self.angular_template_dir.mkdir(parents=True, exist_ok=True)
        
        # Load patch templates
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, str]:
        """Load Angular patch templates."""
        templates = {}
        
        if not self.angular_template_dir.exists():
            logger.warning(f"Angular templates directory not found: {self.angular_template_dir}")
            return templates
        
        for template_file in self.angular_template_dir.glob("*.ts.template"):
            try:
                with open(template_file, 'r') as f:
                    template_name = template_file.stem.replace('.ts', '')
                    templates[template_name] = f.read()
                    logger.debug(f"Loaded Angular template: {template_name}")
            except Exception as e:
                logger.error(f"Error loading Angular template {template_file}: {e}")
        
        return templates
    
    def generate_patch(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                      source_code: str) -> Optional[Dict[str, Any]]:
        """
        Generate a patch for the Angular error.
        
        Args:
            error_data: The Angular error data
            analysis: Analysis results from AngularExceptionHandler
            source_code: The source code where the error occurred
            
        Returns:
            Patch information or None if no patch can be generated
        """
        root_cause = analysis.get("root_cause", "")
        
        # Map root causes to patch strategies
        patch_strategies = {
            "angular_no_provider": self._fix_no_provider,
            "angular_injection_token_error": self._fix_injection_token_error,
            "angular_circular_dependency": self._fix_circular_dependency,
            "angular_cannot_resolve_parameters": self._fix_cannot_resolve_parameters,
            "ngrx_store_not_provided": self._fix_ngrx_store_not_provided,
            "ngrx_action_no_type": self._fix_ngrx_action_no_type,
            "ngrx_reducer_undefined": self._fix_ngrx_reducer_undefined,
            "angular_template_property_undefined": self._fix_template_property_undefined,
            "angular_invalid_property_binding": self._fix_invalid_property_binding,
            "angular_unknown_element": self._fix_unknown_element,
            "angular_unknown_property": self._fix_unknown_property
        }
        
        strategy = patch_strategies.get(root_cause)
        if strategy:
            try:
                return strategy(error_data, analysis, source_code)
            except Exception as e:
                logger.error(f"Error generating Angular patch for {root_cause}: {e}")
        
        # Try to use templates if no specific strategy matches
        return self._template_based_patch(error_data, analysis, source_code)
    
    def _fix_no_provider(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                        source_code: str) -> Optional[Dict[str, Any]]:
        """Fix 'No provider for' errors."""
        message = error_data.get("message", "")
        
        # Extract service name if possible
        service_match = re.search(r"No provider for (\w+)", message)
        service_name = service_match.group(1) if service_match else "YourService"
        
        return {
            "type": "suggestion",
            "description": f"Add {service_name} to providers",
            "fix_commands": [
                f"Add {service_name} to the providers array in your module",
                f"Or add @Injectable({{providedIn: 'root'}}) to {service_name}",
                "Ensure the service is imported in the module"
            ],
            "fix_code": f"""// In your module:
@NgModule({{
  providers: [
    {service_name},
    // ... other providers
  ]
}})
export class YourModule {{ }}

// Or in your service:
@Injectable({{providedIn: 'root'}})
export class {service_name} {{ }}"""
        }
    
    def _fix_injection_token_error(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                                  source_code: str) -> Optional[Dict[str, Any]]:
        """Fix InjectionToken errors."""
        return {
            "type": "suggestion",
            "description": "Provide value for InjectionToken",
            "fix_commands": [
                "Create InjectionToken with new InjectionToken<Type>('description')",
                "Provide value in module providers array",
                "Use @Inject() decorator in constructor"
            ],
            "fix_code": """// Define token
export const MY_TOKEN = new InjectionToken<string>('My config token');

// Provide value
@NgModule({
  providers: [
    { provide: MY_TOKEN, useValue: 'my-value' }
  ]
})

// Inject in service/component
constructor(@Inject(MY_TOKEN) private config: string) {}"""
        }
    
    def _fix_circular_dependency(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                                source_code: str) -> Optional[Dict[str, Any]]:
        """Fix circular dependency errors."""
        return {
            "type": "suggestion",
            "description": "Remove circular dependencies between services",
            "fix_commands": [
                "Use forwardRef() for circular dependencies",
                "Extract shared dependencies to a separate service",
                "Restructure services to avoid circular references"
            ],
            "fix_code": """// Use forwardRef for circular dependencies
@Injectable()
export class ServiceA {
  constructor(@Inject(forwardRef(() => ServiceB)) private serviceB: ServiceB) {}
}

@Injectable()
export class ServiceB {
  constructor(@Inject(forwardRef(() => ServiceA)) private serviceA: ServiceA) {}
}"""
        }
    
    def _fix_cannot_resolve_parameters(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                                      source_code: str) -> Optional[Dict[str, Any]]:
        """Fix 'cannot resolve all parameters' errors."""
        return {
            "type": "suggestion",
            "description": "Add @Injectable() decorator and ensure dependencies are provided",
            "fix_commands": [
                "Add @Injectable() decorator to the service class",
                "Ensure all constructor parameters are provided or have @Inject() decorators",
                "Check that all dependencies are registered in the module"
            ],
            "fix_code": """@Injectable()
export class YourService {
  constructor(
    private http: HttpClient,
    @Inject(MY_TOKEN) private config: any
  ) {}
}"""
        }
    
    def _fix_ngrx_store_not_provided(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                                    source_code: str) -> Optional[Dict[str, Any]]:
        """Fix NgRx store not provided error."""
        return {
            "type": "suggestion",
            "description": "Import StoreModule in your AppModule",
            "fix_commands": [
                "Import StoreModule.forRoot() in AppModule",
                "Import reducers and initial state",
                "Ensure Store is imported where used"
            ],
            "fix_code": """import { StoreModule } from '@ngrx/store';
import { reducers, metaReducers } from './reducers';

@NgModule({
  imports: [
    StoreModule.forRoot(reducers, {
      metaReducers,
      runtimeChecks: {
        strictStateImmutability: true,
        strictActionImmutability: true,
      }
    })
  ]
})
export class AppModule {}"""
        }
    
    def _fix_ngrx_action_no_type(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                                source_code: str) -> Optional[Dict[str, Any]]:
        """Fix NgRx action without type property."""
        return {
            "type": "suggestion",
            "description": "Ensure all NgRx actions have a 'type' property",
            "fix_commands": [
                "Use createAction() helper to create actions",
                "Ensure all action objects have a 'type' property",
                "Follow NgRx action naming conventions"
            ],
            "fix_code": """import { createAction, props } from '@ngrx/store';

// Create actions with createAction
export const loadItems = createAction('[Item] Load Items');
export const loadItemsSuccess = createAction(
  '[Item] Load Items Success',
  props<{ items: Item[] }>()
);
export const loadItemsFailure = createAction(
  '[Item] Load Items Failure',
  props<{ error: any }>()
);"""
        }
    
    def _fix_ngrx_reducer_undefined(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                                   source_code: str) -> Optional[Dict[str, Any]]:
        """Fix NgRx reducer returning undefined."""
        return {
            "type": "suggestion",
            "description": "Ensure reducer always returns a state object",
            "fix_commands": [
                "Use createReducer() helper with initial state",
                "Ensure all switch cases return state",
                "Add default case that returns current state"
            ],
            "fix_code": """import { createReducer, on } from '@ngrx/store';
import * as ItemActions from './item.actions';

export interface ItemState {
  items: Item[];
  loading: boolean;
  error: any;
}

export const initialState: ItemState = {
  items: [],
  loading: false,
  error: null
};

export const itemReducer = createReducer(
  initialState,
  on(ItemActions.loadItems, state => ({ ...state, loading: true })),
  on(ItemActions.loadItemsSuccess, (state, { items }) => ({ 
    ...state, 
    items, 
    loading: false, 
    error: null 
  })),
  on(ItemActions.loadItemsFailure, (state, { error }) => ({ 
    ...state, 
    loading: false, 
    error 
  }))
);"""
        }
    
    def _fix_template_property_undefined(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                                        source_code: str) -> Optional[Dict[str, Any]]:
        """Fix template property access on undefined."""
        return {
            "type": "suggestion",
            "description": "Use safe navigation operator or null checks",
            "fix_commands": [
                "Use safe navigation operator (?.) for potentially undefined properties",
                "Use *ngIf to check if object exists before accessing properties",
                "Initialize properties in component constructor or ngOnInit"
            ],
            "fix_code": """<!-- Use safe navigation operator -->
<div>{{ user?.name }}</div>
<div>{{ user?.address?.street }}</div>

<!-- Use *ngIf to check existence -->
<div *ngIf="user">{{ user.name }}</div>

<!-- Initialize in component -->
ngOnInit() {
  this.user = this.user || {};
}"""
        }
    
    def _fix_invalid_property_binding(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                                     source_code: str) -> Optional[Dict[str, Any]]:
        """Fix invalid property binding errors."""
        message = error_data.get("message", "")
        
        # Extract property name if possible
        property_match = re.search(r"Can't bind to '([^']+)'", message)
        property_name = property_match.group(1) if property_match else "propertyName"
        
        return {
            "type": "suggestion",
            "description": f"Fix property binding for '{property_name}'",
            "fix_commands": [
                f"Add @Input() {property_name} to component if it's an input property",
                f"Check spelling of '{property_name}' property",
                "Ensure the property exists on the target element or component"
            ],
            "fix_code": f"""// In component:
export class YourComponent {{
  @Input() {property_name}: any;
}}

<!-- In template: -->
<app-child [{property_name}]="value"></app-child>"""
        }
    
    def _fix_unknown_element(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                            source_code: str) -> Optional[Dict[str, Any]]:
        """Fix unknown element errors."""
        message = error_data.get("message", "")
        
        # Extract element name if possible
        element_match = re.search(r"'([^']+)' is not a known element", message)
        element_name = element_match.group(1) if element_match else "app-component"
        
        return {
            "type": "suggestion",
            "description": f"Import component or module for '{element_name}'",
            "fix_commands": [
                f"Import the module containing '{element_name}' component",
                f"Add '{element_name}' to declarations in current module",
                f"Check the selector name of the component"
            ],
            "fix_code": f"""// Import component module
import {{ ComponentModule }} from './component.module';

@NgModule({{
  imports: [ComponentModule],
  // or declarations if component is in same module
  declarations: [{element_name.split('-')[1].title()}Component]
}})
export class YourModule {{ }}"""
        }
    
    def _fix_unknown_property(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                             source_code: str) -> Optional[Dict[str, Any]]:
        """Fix unknown property errors."""
        message = error_data.get("message", "")
        
        # Extract property name if possible
        property_match = re.search(r"'([^']+)' is not a known property", message)
        property_name = property_match.group(1) if property_match else "propertyName"
        
        return {
            "type": "suggestion",
            "description": f"Add @Input() decorator for property '{property_name}'",
            "fix_commands": [
                f"Add @Input() {property_name} to the target component",
                f"Check spelling of '{property_name}' property",
                "Ensure the property is meant to be an input property"
            ],
            "fix_code": f"""export class TargetComponent {{
  @Input() {property_name}: any;
}}"""
        }
    
    def _template_based_patch(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                            source_code: str) -> Optional[Dict[str, Any]]:
        """Generate patch using templates."""
        root_cause = analysis.get("root_cause", "")
        
        # Map root causes to template names
        template_map = {
            "angular_no_provider": "dependency_injection_provider",
            "ngrx_store_not_provided": "ngrx_store_setup",
            "angular_template_property_undefined": "safe_navigation",
            "angular_invalid_property_binding": "input_property"
        }
        
        template_name = template_map.get(root_cause)
        if template_name and template_name in self.templates:
            template = self.templates[template_name]
            
            return {
                "type": "template",
                "template": template,
                "description": f"Applied Angular template fix for {root_cause}"
            }
        
        return None


class AngularLanguagePlugin(LanguagePlugin):
    """
    Main Angular framework plugin for Homeostasis.
    
    This plugin orchestrates Angular error analysis and patch generation,
    supporting Angular components, dependency injection, NgRx, templates, and SSR.
    """
    
    VERSION = "1.0.0"
    AUTHOR = "Homeostasis Team"
    
    def __init__(self):
        """Initialize the Angular language plugin."""
        self.language = "angular"
        self.supported_extensions = {".ts", ".js", ".html"}
        self.supported_frameworks = [
            "angular", "@angular/core", "@angular/cli", "ionic", 
            "nativescript", "nx", "ngx", "primeng", "angular-material"
        ]
        
        # Initialize components
        self.adapter = JavaScriptErrorAdapter()  # Reuse JavaScript adapter
        self.exception_handler = AngularExceptionHandler()
        self.patch_generator = AngularPatchGenerator()
        
        logger.info("Angular framework plugin initialized")
    
    def get_language_id(self) -> str:
        """Get the unique identifier for this language."""
        return "angular"
    
    def get_language_name(self) -> str:
        """Get the human-readable name of the framework."""
        return "Angular"
    
    def get_language_version(self) -> str:
        """Get the version of the framework supported by this plugin."""
        return "2+"
    
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
        if "angular" in framework or "@angular" in framework:
            return True
        
        # Check error message for Angular-specific patterns
        message = error_data.get("message", "").lower()
        stack_trace = str(error_data.get("stack_trace", "")).lower()
        
        angular_patterns = [
            r"angular",
            r"@angular",
            r"ng\s",
            r"dependency injection",
            r"injector",
            r"no provider for",
            r"ngrx",
            r"@ngrx",
            r"@component",
            r"@injectable",
            r"@ngmodule",
            r"@input",
            r"@output",
            r"angular\.core",
            r"angular\.common",
            r"platformbrowserdynamic",
            r"component.*angular",
            r"angular.*component",
            r"template.*angular",
            r"angular.*template",
            r"cannot bind to",
            r"is not a known element",
            r"is not a known property",
            r"expression has changed after",
            r"cannot resolve all parameters",
            r"circular dependency",
            r"injection.*token"
        ]
        
        for pattern in angular_patterns:
            if re.search(pattern, message + stack_trace):
                return True
        
        # Check file paths for Angular project structure
        if re.search(r'@angular|angular\.json|ng\s|\.component\.|\.service\.|\.module\.', stack_trace):
            return True
        
        # Check for Angular in package dependencies (if available)
        context = error_data.get("context", {})
        dependencies = context.get("dependencies", [])
        if any("@angular" in dep or "angular" in dep or "ngrx" in dep for dep in dependencies):
            return True
        
        return False
    
    def analyze_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze an Angular error.
        
        Args:
            error_data: Angular error data
            
        Returns:
            Analysis results
        """
        try:
            # Ensure error data is in standard format
            if not error_data.get("language"):
                standard_error = self.adapter.to_standard_format(error_data)
            else:
                standard_error = error_data
            
            message = standard_error.get("message", "").lower()
            
            # Check if it's a dependency injection error
            if self._is_dependency_injection_error(standard_error):
                analysis = self.exception_handler.analyze_dependency_injection_error(standard_error)
            
            # Check if it's an NgRx error
            elif self._is_ngrx_error(standard_error):
                analysis = self.exception_handler.analyze_ngrx_error(standard_error)
            
            # Check if it's a template binding error
            elif self._is_template_binding_error(standard_error):
                analysis = self.exception_handler.analyze_template_binding_error(standard_error)
            
            # Default Angular error analysis
            else:
                analysis = self.exception_handler.analyze_exception(standard_error)
            
            # Add plugin metadata
            analysis["plugin"] = "angular"
            analysis["language"] = "angular"
            analysis["plugin_version"] = self.VERSION
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing Angular error: {e}")
            return {
                "category": "angular",
                "subcategory": "unknown",
                "confidence": "low",
                "suggested_fix": "Unable to analyze Angular error",
                "error": str(e),
                "plugin": "angular"
            }
    
    def _is_dependency_injection_error(self, error_data: Dict[str, Any]) -> bool:
        """Check if this is a dependency injection related error."""
        message = error_data.get("message", "").lower()
        
        di_patterns = [
            "dependency injection",
            "injector",
            "no provider for",
            "injection token",
            "circular dependency",
            "cannot resolve all parameters",
            "invalid provider"
        ]
        
        return any(pattern in message for pattern in di_patterns)
    
    def _is_ngrx_error(self, error_data: Dict[str, Any]) -> bool:
        """Check if this is an NgRx related error."""
        message = error_data.get("message", "").lower()
        stack_trace = str(error_data.get("stack_trace", "")).lower()
        
        ngrx_patterns = [
            "ngrx",
            "@ngrx",
            "store",
            "action",
            "reducer",
            "effect",
            "selector",
            "dispatch"
        ]
        
        return any(pattern in message or pattern in stack_trace for pattern in ngrx_patterns)
    
    def _is_template_binding_error(self, error_data: Dict[str, Any]) -> bool:
        """Check if this is a template binding related error."""
        message = error_data.get("message", "").lower()
        
        template_patterns = [
            "template",
            "binding",
            "cannot bind to",
            "is not a known element",
            "is not a known property",
            "expression has changed after",
            "cannot read property"
        ]
        
        return any(pattern in message for pattern in template_patterns)
    
    def generate_fix(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                    source_code: str) -> Optional[Dict[str, Any]]:
        """
        Generate a fix for the Angular error.
        
        Args:
            error_data: The Angular error data
            analysis: Analysis results
            source_code: Source code where the error occurred
            
        Returns:
            Fix information or None if no fix can be generated
        """
        try:
            return self.patch_generator.generate_patch(error_data, analysis, source_code)
        except Exception as e:
            logger.error(f"Error generating Angular fix: {e}")
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
                "Angular dependency injection error handling",
                "NgRx state management error detection and fixes",
                "Template binding error resolution",
                "Angular module and lazy loading optimization",
                "Component lifecycle error detection",
                "Angular Universal SSR error handling",
                "Forms validation error fixes",
                "Router navigation error detection",
                "Custom directive error handling",
                "Angular CLI integration"
            ],
            "environments": ["browser", "node", "ionic", "electron", "nativescript"]
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
            "raw_data": error_data
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
            "language_specific": standard_error.get("raw_data", {})
        }


# Register the plugin
register_plugin(AngularLanguagePlugin())