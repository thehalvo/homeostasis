"""
CSS Framework Plugin for Homeostasis

This plugin enables Homeostasis to analyze and fix errors in CSS frameworks and tools.
It provides comprehensive error handling for Tailwind CSS, CSS-in-JS libraries,
CSS Modules, SASS/LESS, CSS Grid, Flexbox, and CSS animations.
"""
import logging
import re
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union, Set

from ..language_plugin_system import LanguagePlugin, register_plugin

logger = logging.getLogger(__name__)


class CSSExceptionHandler:
    """
    Handles CSS framework-specific exceptions with comprehensive error detection and classification.
    
    This class provides logic for categorizing CSS framework errors, Tailwind CSS optimization,
    CSS-in-JS issues, CSS Module problems, SASS/LESS compilation errors, and layout debugging.
    """
    
    def __init__(self):
        """Initialize the CSS exception handler."""
        self.rule_categories = {
            "tailwind": "Tailwind CSS framework errors",
            "css_in_js": "CSS-in-JS library errors (Styled Components, Emotion)",
            "css_modules": "CSS Modules and preprocessing errors",
            "layout": "CSS Grid and Flexbox layout errors",
            "animations": "CSS animation and transition errors",
            "compilation": "CSS compilation and build errors",
            "performance": "CSS performance optimization",
            "syntax": "CSS syntax and validation errors",
            "responsive": "Responsive design and media query errors",
            "accessibility": "CSS accessibility issues"
        }
        
        # Load rules from different categories
        self.rules = self._load_rules()
        
        # Pre-compile regex patterns for better performance
        self._compile_patterns()
    
    def _load_rules(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load CSS error rules from rule files."""
        rules = {}
        rules_dir = Path(__file__).parent.parent / "rules" / "css"
        
        try:
            # Load Tailwind CSS rules
            tailwind_rules_path = rules_dir / "tailwind_errors.json"
            if tailwind_rules_path.exists():
                with open(tailwind_rules_path, 'r') as f:
                    tailwind_data = json.load(f)
                    rules["tailwind"] = tailwind_data.get("rules", [])
                    logger.info(f"Loaded {len(rules['tailwind'])} Tailwind CSS rules")
            
            # Load CSS-in-JS rules
            css_in_js_rules_path = rules_dir / "css_in_js_errors.json"
            if css_in_js_rules_path.exists():
                with open(css_in_js_rules_path, 'r') as f:
                    css_in_js_data = json.load(f)
                    rules["css_in_js"] = css_in_js_data.get("rules", [])
                    logger.info(f"Loaded {len(rules['css_in_js'])} CSS-in-JS rules")
            
            # Load CSS layout rules
            layout_rules_path = rules_dir / "css_layout_errors.json"
            if layout_rules_path.exists():
                with open(layout_rules_path, 'r') as f:
                    layout_data = json.load(f)
                    rules["layout"] = layout_data.get("rules", [])
                    logger.info(f"Loaded {len(rules['layout'])} CSS layout rules")
                    
        except Exception as e:
            logger.error(f"Error loading CSS rules: {e}")
            rules = {"tailwind": [], "css_in_js": [], "layout": []}
        
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
                    logger.warning(f"Invalid regex pattern in CSS rule {rule.get('id', 'unknown')}: {e}")
    
    def analyze_exception(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a CSS framework exception and determine its type and potential fixes.
        
        Args:
            error_data: CSS error data in standard format
            
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
            # Ensure category is always "css" for CSS plugin
            result = {
                "category": "css",
                "subcategory": best_match.get("subcategory", best_match.get("category", "unknown")),
                "confidence": best_match.get("confidence", "medium"),
                "suggested_fix": best_match.get("suggestion", ""),
                "root_cause": best_match.get("root_cause", ""),
                "severity": best_match.get("severity", "medium"),
                "rule_id": best_match.get("id", ""),
                "tags": best_match.get("tags", []),
                "fix_commands": best_match.get("fix_commands", []),
                "all_matches": matches
            }
            # If subcategory is a primary category like "css_in_js", keep it
            return result
        
        # If no rules matched, provide generic analysis
        return self._generic_analysis(error_data)
    
    def _find_matching_rules(self, error_text: str, error_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find all rules that match the given error."""
        matches = []
        
        # For truly generic CSS syntax errors, return no matches to force generic analysis
        message = error_data.get("message", "").lower()
        if "css syntax error" in message and not any(fw in message for fw in ["tailwind", "styled", "emotion", "sass", "less"]):
            return []
        
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
        
        # Boost confidence for CSS-specific patterns
        message = error_data.get("message", "").lower()
        if any(term in message for term in ["css", "style", "tailwind", "emotion", "styled"]):
            base_confidence += 0.3
        
        # Boost confidence based on rule reliability
        reliability = rule.get("reliability", "medium")
        reliability_boost = {"high": 0.2, "medium": 0.1, "low": 0.0}
        base_confidence += reliability_boost.get(reliability, 0.0)
        
        # Boost confidence for rules with specific tags that match context
        rule_tags = set(rule.get("tags", []))
        context_tags = set()
        
        # Infer context from error data
        if "tailwind" in message:
            context_tags.add("tailwind")
        if "styled" in message or "emotion" in message:
            context_tags.add("css-in-js")
        if "grid" in message or "flexbox" in message:
            context_tags.add("layout")
        if "animation" in message or "transition" in message:
            context_tags.add("animations")
        
        if context_tags & rule_tags:
            base_confidence += 0.1
        
        return min(base_confidence, 1.0)
    
    def _generic_analysis(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """Provide generic analysis for unmatched errors."""
        error_type = error_data.get("error_type", "Error")
        message = error_data.get("message", "").lower()
        
        # For generic CSS syntax errors without specific framework patterns
        if "css syntax error" in message and not any(fw in message for fw in ["tailwind", "styled", "emotion", "sass", "less"]):
            return {
                "category": "css",
                "subcategory": "unknown",
                "confidence": "low",
                "suggested_fix": "Review CSS syntax and validation",
                "root_cause": "css_unknown_error",
                "severity": "medium",
                "rule_id": "css_generic_handler",
                "tags": ["css", "generic", "unknown"]
            }
        
        # Basic categorization based on error patterns
        if "tailwind" in message:
            category = "tailwind"
            suggestion = "Check Tailwind CSS configuration and class usage"
        elif "styled" in message or "emotion" in message:
            category = "css_in_js"
            suggestion = "Check CSS-in-JS library configuration and styled component syntax"
        elif "grid" in message or "flexbox" in message or "flex" in message:
            category = "layout"
            suggestion = "Check CSS Grid or Flexbox layout properties"
        elif "animation" in message or "transition" in message:
            category = "animations"
            suggestion = "Check CSS animation and transition syntax"
        elif "sass" in message or "scss" in message or "less" in message:
            category = "css_modules"
            suggestion = "Check CSS preprocessing syntax and compilation"
        else:
            category = "unknown"
            suggestion = "Review CSS framework configuration and usage"
        
        # Calculate confidence based on pattern specificity
        confidence = "low"
        if category != "unknown":
            # Boost confidence if we matched a specific category
            confidence = "medium"
            # Further boost for specific framework mentions
            if any(fw in message for fw in ["tailwind", "styled-components", "emotion", "sass", "less"]):
                confidence = "high"
        
        return {
            "category": "css",
            "subcategory": category,
            "confidence": confidence,
            "suggested_fix": suggestion,
            "root_cause": f"css_{category}_error",
            "severity": "medium",
            "rule_id": "css_generic_handler",
            "tags": ["css", "generic", category]
        }
    
    def analyze_tailwind_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze Tailwind CSS specific errors.
        
        Args:
            error_data: Error data with Tailwind-related issues
            
        Returns:
            Analysis results with Tailwind-specific fixes
        """
        message = error_data.get("message", "")
        
        # Common Tailwind error patterns
        tailwind_patterns = [
            {
                "patterns": ["unknown utility class", "class not found", "invalid class", "unrecognized class"],
                "cause": "tailwind_unknown_class",
                "fix": "Check Tailwind CSS class name spelling and availability",
                "severity": "warning"
            },
            {
                "patterns": ["purged", "was purged", "purge configuration"],
                "cause": "tailwind_purge_error",
                "fix": "Check Tailwind CSS purge configuration - class may be incorrectly purged",
                "severity": "warning"
            },
            {
                "patterns": ["@apply", "apply directive"],
                "cause": "tailwind_apply_error",
                "fix": "Check @apply directive usage with valid Tailwind utilities",
                "severity": "error"
            },
            {
                "patterns": ["config", "configuration", "tailwind.config"],
                "cause": "tailwind_config_error",
                "fix": "Check tailwind.config.js configuration file",
                "severity": "error"
            },
            {
                "patterns": ["build", "postcss", "compilation"],
                "cause": "tailwind_build_error",
                "fix": "Check Tailwind CSS build process and PostCSS configuration",
                "severity": "error"
            }
        ]
        
        message_lower = message.lower()
        for pattern_info in tailwind_patterns:
            if any(pattern in message_lower for pattern in pattern_info["patterns"]):
                return {
                    "category": "css",
                    "subcategory": "tailwind",
                    "confidence": "high",
                    "suggested_fix": pattern_info["fix"],
                    "root_cause": pattern_info["cause"],
                    "severity": pattern_info["severity"],
                    "tags": ["css", "tailwind", "framework"]
                }
        
        # Generic Tailwind error
        return {
            "category": "css",
            "subcategory": "tailwind",
            "confidence": "medium",
            "suggested_fix": "Check Tailwind CSS configuration and class usage",
            "root_cause": "tailwind_general_error",
            "severity": "warning",
            "tags": ["css", "tailwind"]
        }
    
    def analyze_css_in_js_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze CSS-in-JS library errors (Styled Components, Emotion).
        
        Args:
            error_data: Error data with CSS-in-JS issues
            
        Returns:
            Analysis results with CSS-in-JS fixes
        """
        message = error_data.get("message", "").lower()
        stack_trace = str(error_data.get("stack_trace", "")).lower()
        
        # Styled Components specific errors
        if "styled" in message or "styled-components" in message:
            return self._analyze_styled_components_error(message, error_data)
        
        # Emotion specific errors
        if "emotion" in message:
            return self._analyze_emotion_error(message, error_data)
        
        # General CSS-in-JS errors
        css_in_js_patterns = {
            "template literal": {
                "cause": "css_in_js_template_literal_error",
                "fix": "Check CSS-in-JS template literal syntax",
                "severity": "error"
            },
            "theme": {
                "cause": "css_in_js_theme_error",
                "fix": "Check theme provider configuration and theme object structure",
                "severity": "error"
            },
            "props": {
                "cause": "css_in_js_props_error",
                "fix": "Check styled component props usage and TypeScript types",
                "severity": "warning"
            }
        }
        
        for pattern, info in css_in_js_patterns.items():
            if pattern in message:
                return {
                    "category": "css",
                    "subcategory": "css_in_js",
                    "confidence": "high",
                    "suggested_fix": info["fix"],
                    "root_cause": info["cause"],
                    "severity": info["severity"],
                    "tags": ["css", "css-in-js"]
                }
        
        return {
            "category": "css",
            "subcategory": "css_in_js",
            "confidence": "medium",
            "suggested_fix": "Check CSS-in-JS library configuration and usage",
            "root_cause": "css_in_js_general_error",
            "severity": "medium",
            "tags": ["css", "css-in-js"]
        }
    
    def _analyze_styled_components_error(self, message: str, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze Styled Components-specific errors."""
        if "babel" in message or "transform" in message:
            return {
                "category": "css",
                "subcategory": "css_in_js",
                "confidence": "high",
                "suggested_fix": "Check Styled Components Babel plugin configuration",
                "root_cause": "styled_components_babel_error",
                "severity": "error",
                "tags": ["css", "styled-components", "babel"]
            }
        
        if "ssr" in message or "server" in message:
            return {
                "category": "css",
                "subcategory": "css_in_js",
                "confidence": "high",
                "suggested_fix": "Configure Styled Components SSR with ServerStyleSheet",
                "root_cause": "styled_components_ssr_error",
                "severity": "error",
                "tags": ["css", "styled-components", "ssr"]
            }
        
        return {
            "category": "css",
            "subcategory": "css_in_js",
            "confidence": "medium",
            "suggested_fix": "Check Styled Components configuration and usage",
            "root_cause": "styled_components_general_error",
            "severity": "medium",
            "tags": ["css", "styled-components"]
        }
    
    def _analyze_emotion_error(self, message: str, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze Emotion-specific errors."""
        if "jsx" in message:
            return {
                "category": "css",
                "subcategory": "css_in_js",
                "confidence": "high",
                "suggested_fix": "Configure Emotion JSX pragma or use css prop with @emotion/react",
                "root_cause": "emotion_jsx_error",
                "severity": "error",
                "tags": ["css", "emotion", "jsx"]
            }
        
        if "cache" in message:
            return {
                "category": "css",
                "subcategory": "css_in_js",
                "confidence": "high",
                "suggested_fix": "Configure Emotion cache properly",
                "root_cause": "emotion_cache_error",
                "severity": "error",
                "tags": ["css", "emotion", "cache"]
            }
        
        return {
            "category": "css",
            "subcategory": "css_in_js",
            "confidence": "medium",
            "suggested_fix": "Check Emotion configuration and usage",
            "root_cause": "emotion_general_error",
            "severity": "medium",
            "tags": ["css", "emotion"]
        }
    
    def analyze_layout_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze CSS Grid and Flexbox layout errors.
        
        Args:
            error_data: Error data with layout issues
            
        Returns:
            Analysis results with layout fixes
        """
        message = error_data.get("message", "").lower()
        
        # Grid specific errors
        if "grid" in message:
            return {
                "category": "css",
                "subcategory": "layout",
                "confidence": "high",
                "suggested_fix": "Check CSS Grid properties: grid-template-columns, grid-template-rows, grid-area",
                "root_cause": "css_grid_layout_error",
                "severity": "warning",
                "tags": ["css", "grid", "layout"]
            }
        
        # Flexbox specific errors
        if "flex" in message or "flexbox" in message:
            return {
                "category": "css",
                "subcategory": "layout",
                "confidence": "high",
                "suggested_fix": "Check Flexbox properties: flex-direction, justify-content, align-items",
                "root_cause": "css_flexbox_layout_error",
                "severity": "warning",
                "tags": ["css", "flexbox", "layout"]
            }
        
        # General layout errors
        return {
            "category": "css",
            "subcategory": "layout",
            "confidence": "medium",
            "suggested_fix": "Check CSS layout properties and box model",
            "root_cause": "css_layout_general_error",
            "severity": "medium",
            "tags": ["css", "layout"]
        }


class CSSPatchGenerator:
    """
    Generates patches for CSS framework errors based on analysis results.
    
    This class creates code fixes for common CSS framework errors using templates
    and heuristics specific to CSS frameworks and best practices.
    """
    
    def __init__(self):
        """Initialize the CSS patch generator."""
        self.template_dir = Path(__file__).parent.parent / "patch_generation" / "templates"
        self.css_template_dir = self.template_dir / "css"
        
        # Ensure template directory exists
        self.css_template_dir.mkdir(parents=True, exist_ok=True)
        
        # Load patch templates
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, str]:
        """Load CSS patch templates."""
        templates = {}
        
        if not self.css_template_dir.exists():
            logger.warning(f"CSS templates directory not found: {self.css_template_dir}")
            return templates
        
        for template_file in self.css_template_dir.glob("*.css.template"):
            try:
                with open(template_file, 'r') as f:
                    template_name = template_file.stem.replace('.css', '')
                    templates[template_name] = f.read()
                    logger.debug(f"Loaded CSS template: {template_name}")
            except Exception as e:
                logger.error(f"Error loading CSS template {template_file}: {e}")
        
        return templates
    
    def generate_patch(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                      source_code: str) -> Optional[Dict[str, Any]]:
        """
        Generate a patch for the CSS framework error.
        
        Args:
            error_data: The CSS error data
            analysis: Analysis results from CSSExceptionHandler
            source_code: The source code where the error occurred
            
        Returns:
            Patch information or None if no patch can be generated
        """
        root_cause = analysis.get("root_cause", "")
        
        # Map root causes to patch strategies
        patch_strategies = {
            "tailwind_unknown_class": self._fix_tailwind_unknown_class,
            "tailwind_purge_error": self._fix_tailwind_purge_error,
            "tailwind_config_error": self._fix_tailwind_config_error,
            "styled_components_babel_error": self._fix_styled_components_babel,
            "styled_components_ssr_error": self._fix_styled_components_ssr,
            "emotion_jsx_error": self._fix_emotion_jsx,
            "css_grid_layout_error": self._fix_css_grid_layout,
            "css_flexbox_layout_error": self._fix_css_flexbox_layout
        }
        
        strategy = patch_strategies.get(root_cause)
        if strategy:
            try:
                return strategy(error_data, analysis, source_code)
            except Exception as e:
                logger.error(f"Error generating CSS patch for {root_cause}: {e}")
        
        # Try to use templates if no specific strategy matches
        return self._template_based_patch(error_data, analysis, source_code)
    
    def _fix_tailwind_unknown_class(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                                   source_code: str) -> Optional[Dict[str, Any]]:
        """Fix Tailwind unknown class errors."""
        return {
            "type": "suggestion",
            "description": "Check Tailwind CSS class name spelling and configuration",
            "fix_commands": [
                "Verify class name exists in Tailwind CSS documentation",
                "Check if class is included in your Tailwind build",
                "Ensure class is not purged by PurgeCSS configuration",
                "Add custom utility to tailwind.config.js if needed"
            ]
        }
    
    def _fix_tailwind_purge_error(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                                 source_code: str) -> Optional[Dict[str, Any]]:
        """Fix Tailwind purge configuration errors."""
        return {
            "type": "configuration",
            "description": "Update Tailwind CSS purge configuration",
            "fix_commands": [
                "Add file patterns to purge.content in tailwind.config.js",
                "Use safelist to preserve specific classes",
                "Check purge.options for custom extractors",
                "Disable purge in development mode"
            ],
            "template": "tailwind_purge_fix"
        }
    
    def _fix_tailwind_config_error(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                                  source_code: str) -> Optional[Dict[str, Any]]:
        """Fix Tailwind configuration errors."""
        return {
            "type": "configuration",
            "description": "Fix Tailwind CSS configuration file",
            "fix_commands": [
                "Check tailwind.config.js syntax and structure",
                "Verify theme customizations",
                "Ensure plugins are properly configured",
                "Check PostCSS configuration"
            ],
            "template": "tailwind_config_fix"
        }
    
    def _fix_styled_components_babel(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                                   source_code: str) -> Optional[Dict[str, Any]]:
        """Fix Styled Components Babel configuration."""
        return {
            "type": "configuration",
            "description": "Configure Styled Components Babel plugin",
            "fix_commands": [
                "Add babel-plugin-styled-components to .babelrc",
                "Configure displayName and fileName options",
                "Enable SSR and minification options",
                "Restart development server after configuration"
            ],
            "fix_code": """
{
  "plugins": [
    ["babel-plugin-styled-components", {
      "displayName": true,
      "fileName": true
    }]
  ]
}
"""
        }
    
    def _fix_styled_components_ssr(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                                  source_code: str) -> Optional[Dict[str, Any]]:
        """Fix Styled Components SSR configuration."""
        return {
            "type": "code_addition",
            "description": "Configure Styled Components for Server-Side Rendering",
            "fix_commands": [
                "Import ServerStyleSheet from styled-components",
                "Collect styles during SSR",
                "Inject styles into HTML head",
                "Clear styles after rendering"
            ],
            "fix_code": """
import { ServerStyleSheet } from 'styled-components'

const sheet = new ServerStyleSheet()
try {
  const html = renderToString(sheet.collectStyles(<App />))
  const styleTags = sheet.getStyleTags()
  // Inject styleTags into HTML head
} finally {
  sheet.seal()
}
"""
        }
    
    def _fix_emotion_jsx(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                        source_code: str) -> Optional[Dict[str, Any]]:
        """Fix Emotion JSX configuration."""
        return {
            "type": "configuration",
            "description": "Configure Emotion JSX pragma",
            "fix_commands": [
                "Add JSX pragma to files using css prop",
                "Configure Babel preset for automatic JSX",
                "Import jsx from @emotion/react",
                "Update TypeScript configuration for JSX"
            ],
            "fix_code": """
/** @jsx jsx */
import { jsx } from '@emotion/react'

// or use automatic JSX runtime
import { jsx } from '@emotion/react/jsx-runtime'
"""
        }
    
    def _fix_css_grid_layout(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                           source_code: str) -> Optional[Dict[str, Any]]:
        """Fix CSS Grid layout issues."""
        return {
            "type": "suggestion",
            "description": "Fix CSS Grid layout properties",
            "fix_commands": [
                "Check grid-template-columns and grid-template-rows",
                "Verify grid-area assignments",
                "Ensure proper grid container setup",
                "Check for grid item overflow issues",
                "Validate grid gap properties"
            ],
            "template": "css_grid_fix"
        }
    
    def _fix_css_flexbox_layout(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                               source_code: str) -> Optional[Dict[str, Any]]:
        """Fix CSS Flexbox layout issues."""
        return {
            "type": "suggestion",
            "description": "Fix CSS Flexbox layout properties",
            "fix_commands": [
                "Check flex-direction property",
                "Verify justify-content alignment",
                "Check align-items and align-content",
                "Validate flex-grow, flex-shrink, flex-basis",
                "Ensure proper flex container setup"
            ],
            "template": "css_flexbox_fix"
        }
    
    def _template_based_patch(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                            source_code: str) -> Optional[Dict[str, Any]]:
        """Generate patch using templates."""
        root_cause = analysis.get("root_cause", "")
        
        # Map root causes to template names
        template_map = {
            "tailwind_purge_error": "tailwind_purge_fix",
            "tailwind_config_error": "tailwind_config_fix",
            "css_grid_layout_error": "css_grid_fix",
            "css_flexbox_layout_error": "css_flexbox_fix"
        }
        
        template_name = template_map.get(root_cause)
        if template_name and template_name in self.templates:
            template = self.templates[template_name]
            
            return {
                "type": "template",
                "template": template,
                "description": f"Applied CSS template fix for {root_cause}"
            }
        
        return None


class CSSLanguagePlugin(LanguagePlugin):
    """
    Main CSS framework plugin for Homeostasis.
    
    This plugin orchestrates CSS framework error analysis and patch generation,
    supporting Tailwind CSS, CSS-in-JS libraries, CSS Modules, and layout debugging.
    """
    
    VERSION = "1.0.0"
    AUTHOR = "Homeostasis Team"
    
    def __init__(self):
        """Initialize the CSS language plugin."""
        self.language = "css"
        self.supported_extensions = {".css", ".scss", ".sass", ".less", ".styl", ".js", ".jsx", ".ts", ".tsx"}
        self.supported_frameworks = [
            "tailwindcss", "styled-components", "emotion", "css-modules", 
            "sass", "less", "stylus", "postcss", "bootstrap", "material-ui"
        ]
        
        # Initialize components
        self.exception_handler = CSSExceptionHandler()
        self.patch_generator = CSSPatchGenerator()
        
        logger.info("CSS framework plugin initialized")
    
    def get_language_id(self) -> str:
        """Get the unique identifier for this language."""
        return "css"
    
    def get_language_name(self) -> str:
        """Get the human-readable name of the framework."""
        return "CSS Frameworks"
    
    def get_language_version(self) -> str:
        """Get the version of the framework supported by this plugin."""
        return "CSS3+"
    
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
        if any(fw in framework for fw in ["tailwind", "styled", "emotion", "css"]):
            return True
        
        # Check error message for CSS-specific patterns
        message = error_data.get("message", "").lower()
        stack_trace = str(error_data.get("stack_trace", "")).lower()
        
        css_patterns = [
            r"tailwind",
            r"styled-components",
            r"emotion",
            r"css module",
            r"sass",
            r"scss",
            r"less",
            r"postcss",
            r"css.*error",
            r"style.*error",
            r"@apply",
            r"css-in-js",
            r"grid.*error",
            r"flex.*error",
            r"animation.*error",
            r"transition.*error"
        ]
        
        for pattern in css_patterns:
            if re.search(pattern, message + stack_trace):
                return True
        
        # Check file extensions for CSS files
        if re.search(r'\.(css|scss|sass|less|styl):', stack_trace):
            return True
        
        # Check for CSS frameworks in package dependencies (if available)
        context = error_data.get("context", {})
        dependencies = context.get("dependencies", [])
        css_deps = ["tailwindcss", "styled-components", "@emotion", "sass", "less", "stylus"]
        if any(dep in str(dependencies).lower() for dep in css_deps):
            return True
        
        return False
    
    def analyze_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a CSS framework error.
        
        Args:
            error_data: CSS error data
            
        Returns:
            Analysis results
        """
        try:
            message = error_data.get("message", "").lower()
            
            # Check if it's a Tailwind-related error
            if self._is_tailwind_error(error_data):
                analysis = self.exception_handler.analyze_tailwind_error(error_data)
            
            # Check if it's a CSS-in-JS error
            elif self._is_css_in_js_error(error_data):
                analysis = self.exception_handler.analyze_css_in_js_error(error_data)
            
            # Check if it's a layout error
            elif self._is_layout_error(error_data):
                analysis = self.exception_handler.analyze_layout_error(error_data)
            
            # Default CSS error analysis
            else:
                analysis = self.exception_handler.analyze_exception(error_data)
            
            # Add plugin metadata
            analysis["plugin"] = "css"
            analysis["language"] = "css"
            analysis["plugin_version"] = self.VERSION
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing CSS error: {e}")
            return {
                "category": "css",
                "subcategory": "unknown",
                "confidence": "low",
                "suggested_fix": "Unable to analyze CSS error",
                "error": str(e),
                "plugin": "css"
            }
    
    def _is_tailwind_error(self, error_data: Dict[str, Any]) -> bool:
        """Check if this is a Tailwind CSS related error."""
        message = error_data.get("message", "").lower()
        
        tailwind_patterns = [
            "tailwind",
            "@apply",
            "purge",
            "postcss.*tailwind",
            "utility class",
            "bg-",
            "text-",
            "flex-",
            "grid-",
            "p-",
            "m-",
            "w-",
            "h-"
        ]
        
        return any(pattern in message for pattern in tailwind_patterns)
    
    def _is_css_in_js_error(self, error_data: Dict[str, Any]) -> bool:
        """Check if this is a CSS-in-JS related error."""
        message = error_data.get("message", "").lower()
        stack_trace = str(error_data.get("stack_trace", "")).lower()
        
        css_in_js_patterns = [
            "styled-components",
            "emotion",
            "css-in-js",
            "styled",
            "theme"
        ]
        
        return any(pattern in message or pattern in stack_trace for pattern in css_in_js_patterns)
    
    def _is_layout_error(self, error_data: Dict[str, Any]) -> bool:
        """Check if this is a layout related error."""
        message = error_data.get("message", "").lower()
        
        layout_patterns = [
            "grid",
            "flexbox",
            "flex",
            "layout",
            "align",
            "justify"
        ]
        
        return any(pattern in message for pattern in layout_patterns)
    
    def generate_fix(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                    source_code: str) -> Optional[Dict[str, Any]]:
        """
        Generate a fix for the CSS framework error.
        
        Args:
            error_data: The CSS error data
            analysis: Analysis results
            source_code: Source code where the error occurred
            
        Returns:
            Fix information or None if no fix can be generated
        """
        try:
            return self.patch_generator.generate_patch(error_data, analysis, source_code)
        except Exception as e:
            logger.error(f"Error generating CSS fix: {e}")
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
                "Tailwind CSS optimization and error detection",
                "CSS-in-JS library healing (Styled Components, Emotion)",
                "CSS Module and SASS/LESS issue resolution",
                "CSS Grid and Flexbox layout debugging",
                "CSS animation and transition error handling",
                "CSS compilation and build error fixes",
                "CSS performance optimization suggestions",
                "Responsive design error detection"
            ],
            "environments": ["browser", "node", "webpack", "vite", "parcel"]
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
register_plugin(CSSLanguagePlugin())