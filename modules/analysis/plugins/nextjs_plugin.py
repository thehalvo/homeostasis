"""
Next.js Framework Plugin for Homeostasis

This plugin enables Homeostasis to analyze and fix errors in Next.js applications.
It provides comprehensive error handling for Next.js data fetching, API routes,
middleware, image optimization, static generation, server components, and Vercel deployment issues.
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


class NextjsExceptionHandler:
    """
    Handles Next.js-specific exceptions with comprehensive error detection and classification.
    
    This class provides logic for categorizing Next.js data fetching errors, API route issues,
    middleware problems, image optimization errors, static generation failures, and Vercel deployment issues.
    """
    
    def __init__(self):
        """Initialize the Next.js exception handler."""
        self.rule_categories = {
            "data_fetching": "Next.js data fetching errors",
            "api_routes": "Next.js API routes and serverless function errors",
            "middleware": "Next.js middleware errors",
            "image_optimization": "Next.js image component and optimization errors",
            "static_generation": "Static site generation (SSG) and incremental static regeneration (ISR) errors",
            "routing": "Next.js routing and navigation errors",
            "server_components": "React Server Components in Next.js errors",
            "app_dir": "Next.js App Router and app directory errors",
            "pages_dir": "Next.js Pages Router and pages directory errors",
            "deployment": "Vercel and other deployment errors",
            "configuration": "Next.js configuration errors",
            "styling": "CSS and styling related errors in Next.js",
            "typescript": "TypeScript integration errors in Next.js",
            "build": "Build-time errors in Next.js applications"
        }
        
        # Load rules from different categories
        self.rules = self._load_rules()
        
        # Pre-compile regex patterns for better performance
        self._compile_patterns()
    
    def _load_rules(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load Next.js error rules from rule files."""
        rules = {}
        rules_dir = Path(__file__).parent.parent / "rules" / "nextjs"
        
        try:
            # Ensure the rules directory exists
            if not rules_dir.exists():
                rules_dir.mkdir(parents=True, exist_ok=True)
                logger.warning(f"Created Next.js rules directory: {rules_dir}")
                rules = {"common": [], "data_fetching": [], "api_routes": [], "app_router": []}
                return rules
            
            # Load common Next.js rules
            common_rules_path = rules_dir / "nextjs_common_errors.json"
            if common_rules_path.exists():
                with open(common_rules_path, 'r') as f:
                    common_data = json.load(f)
                    rules["common"] = common_data.get("rules", [])
                    logger.info(f"Loaded {len(rules['common'])} common Next.js rules")
            
            # Load data fetching rules
            data_fetching_rules_path = rules_dir / "nextjs_data_fetching_errors.json"
            if data_fetching_rules_path.exists():
                with open(data_fetching_rules_path, 'r') as f:
                    data_fetching_data = json.load(f)
                    rules["data_fetching"] = data_fetching_data.get("rules", [])
                    logger.info(f"Loaded {len(rules['data_fetching'])} Next.js data fetching rules")
            
            # Load API routes rules
            api_routes_rules_path = rules_dir / "nextjs_api_routes_errors.json"
            if api_routes_rules_path.exists():
                with open(api_routes_rules_path, 'r') as f:
                    api_routes_data = json.load(f)
                    rules["api_routes"] = api_routes_data.get("rules", [])
                    logger.info(f"Loaded {len(rules['api_routes'])} Next.js API routes rules")
            
            # Load App Router rules
            app_router_rules_path = rules_dir / "nextjs_app_router_errors.json"
            if app_router_rules_path.exists():
                with open(app_router_rules_path, 'r') as f:
                    app_router_data = json.load(f)
                    rules["app_router"] = app_router_data.get("rules", [])
                    logger.info(f"Loaded {len(rules['app_router'])} Next.js App Router rules")
                    
        except Exception as e:
            logger.error(f"Error loading Next.js rules: {e}")
            rules = {"common": [], "data_fetching": [], "api_routes": [], "app_router": []}
        
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
                    logger.warning(f"Invalid regex pattern in Next.js rule {rule.get('id', 'unknown')}: {e}")
    
    def analyze_exception(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a Next.js exception and determine its type and potential fixes.
        
        Args:
            error_data: Next.js error data in standard format
            
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
                "category": best_match.get("category", "nextjs"),
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
        
        # Boost confidence for Next.js-specific patterns
        message = error_data.get("message", "").lower()
        if "next" in message or "nextjs" in message or "vercel" in message:
            base_confidence += 0.3
        
        # Boost confidence based on rule reliability
        reliability = rule.get("reliability", "medium")
        reliability_boost = {"high": 0.2, "medium": 0.1, "low": 0.0}
        base_confidence += reliability_boost.get(reliability, 0.0)
        
        # Boost confidence for rules with specific tags that match context
        rule_tags = set(rule.get("tags", []))
        context_tags = set()
        
        # Infer context from error data
        if "next" in error_data.get("framework", "").lower():
            context_tags.add("nextjs")
        if "api" in message or "/api/" in message:
            context_tags.add("api_routes")
        if "getServerSideProps" in message or "getStaticProps" in message:
            context_tags.add("data_fetching")
        if "middleware" in message:
            context_tags.add("middleware")
        if "image" in message or "img" in message:
            context_tags.add("image_optimization")
        if "vercel" in message:
            context_tags.add("deployment")
        if "app" in message and "directory" in message:
            context_tags.add("app_dir")
        if "pages" in message and "directory" in message:
            context_tags.add("pages_dir")
        if "server components" in message or "server-side" in message:
            context_tags.add("server_components")
        
        if context_tags & rule_tags:
            base_confidence += 0.1
        
        return min(base_confidence, 1.0)
    
    def _generic_analysis(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """Provide generic analysis for unmatched errors."""
        error_type = error_data.get("error_type", "Error")
        message = error_data.get("message", "").lower()
        
        # Basic categorization based on error patterns
        if "getstaticprops" in message or "getserversideprops" in message or "getstaticpaths" in message:
            category = "data_fetching"
            suggestion = "Check Next.js data fetching methods and their return values"
        elif "api" in message or "/api/" in message:
            category = "api_routes"
            suggestion = "Check API route handler signature and response formatting"
        elif "middleware" in message:
            category = "middleware"
            suggestion = "Verify Next.js middleware implementation and configuration"
        elif "image" in message or "next/image" in message:
            category = "image_optimization"
            suggestion = "Check Next.js Image component usage and configuration"
        elif "build" in message or "failed to compile" in message:
            category = "build"
            suggestion = "Check build configuration and compile-time errors"
        elif "route" in message or "404" in message:
            category = "routing"
            suggestion = "Verify Next.js routing configuration and file naming"
        elif "app" in message and ("directory" in message or "folder" in message):
            category = "app_dir"
            suggestion = "Check App Router implementation and file structure"
        elif "vercel" in message or "deploy" in message:
            category = "deployment"
            suggestion = "Verify Vercel deployment configuration and environment variables"
        elif "css" in message or "style" in message:
            category = "styling"
            suggestion = "Check CSS modules, styling imports, or global styles"
        elif "typescript" in message or "type" in message:
            category = "typescript"
            suggestion = "Check TypeScript types and Next.js type integration"
        else:
            category = "unknown"
            suggestion = "Review Next.js application structure and configuration"
        
        return {
            "category": "nextjs",
            "subcategory": category,
            "confidence": "low",
            "suggested_fix": suggestion,
            "root_cause": f"nextjs_{category}_error",
            "severity": "medium",
            "rule_id": "nextjs_generic_handler",
            "tags": ["nextjs", "generic", category]
        }
    
    def analyze_data_fetching_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze Next.js data fetching specific errors.
        
        Args:
            error_data: Error data with data fetching-related issues
            
        Returns:
            Analysis results with data fetching-specific fixes
        """
        message = error_data.get("message", "")
        
        # Common data fetching error patterns
        data_fetching_patterns = {
            "getServerSideProps.*(not.*return.*object|must.*return.*object)": {
                "cause": "nextjs_getserversideprops_return_type",
                "fix": "Ensure getServerSideProps returns an object with a props property",
                "severity": "error"
            },
            "getStaticProps.*not.*return.*object": {
                "cause": "nextjs_getstaticprops_return_type",
                "fix": "Ensure getStaticProps returns an object with a props property",
                "severity": "error"
            },
            "getStaticPaths.*not.*return.*paths": {
                "cause": "nextjs_getstaticpaths_return_type",
                "fix": "Ensure getStaticPaths returns an object with paths and fallback properties",
                "severity": "error"
            },
            "cannot.*fetch.*absolute URL": {
                "cause": "nextjs_absolute_url_fetch",
                "fix": "Use relative URLs for internal API routes or the full URL with protocol for external APIs",
                "severity": "error"
            },
            "revalidate.*must be a number": {
                "cause": "nextjs_revalidate_type",
                "fix": "The revalidate property must be a positive number or false",
                "severity": "error"
            },
            "can't be executed in the browser": {
                "cause": "nextjs_server_function_browser",
                "fix": "Server-side functions like getServerSideProps cannot be called in browser components",
                "severity": "error"
            },
            "notFound.*must be boolean": {
                "cause": "nextjs_notfound_type",
                "fix": "The notFound property must be a boolean value",
                "severity": "error"
            }
        }
        
        for pattern, info in data_fetching_patterns.items():
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "nextjs",
                    "subcategory": "data_fetching",
                    "confidence": "high",
                    "suggested_fix": info["fix"],
                    "root_cause": info["cause"],
                    "severity": info["severity"],
                    "tags": ["nextjs", "data-fetching", "server-side"]
                }
        
        # Generic data fetching error
        return {
            "category": "nextjs",
            "subcategory": "data_fetching",
            "confidence": "medium",
            "suggested_fix": "Check Next.js data fetching methods and their return values",
            "root_cause": "nextjs_data_fetching_error",
            "severity": "warning",
            "tags": ["nextjs", "data-fetching"]
        }
    
    def analyze_api_route_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze Next.js API route errors.
        
        Args:
            error_data: Error data with API route-related issues
            
        Returns:
            Analysis results with API route-specific fixes
        """
        message = error_data.get("message", "").lower()
        stack_trace = str(error_data.get("stack_trace", "")).lower()
        
        # API route specific error patterns
        api_patterns = {
            "api.*resolved without sending a response": {
                "cause": "nextjs_api_no_response",
                "fix": "Ensure API route handler sends a response using res.status().json() or similar",
                "severity": "error"
            },
            "body parsing failed": {
                "cause": "nextjs_api_body_parse",
                "fix": "Check request body format and content-type header",
                "severity": "error"
            },
            "method.*not allowed": {
                "cause": "nextjs_api_method_not_allowed",
                "fix": "Check that your API handler supports the HTTP method being used",
                "severity": "error"
            },
            "headers were already sent": {
                "cause": "nextjs_api_headers_already_sent",
                "fix": "Avoid sending multiple responses or modifying headers after sending response",
                "severity": "error"
            },
            "invalid status code": {
                "cause": "nextjs_api_invalid_status",
                "fix": "Use a valid HTTP status code between 100-599",
                "severity": "error"
            },
            "edge runtime.*not supported": {
                "cause": "nextjs_api_edge_runtime",
                "fix": "Check Edge Runtime compatibility with your API functionality",
                "severity": "error"
            }
        }
        
        for pattern, info in api_patterns.items():
            if re.search(pattern, message + stack_trace):
                return {
                    "category": "nextjs",
                    "subcategory": "api_routes",
                    "confidence": "high",
                    "suggested_fix": info["fix"],
                    "root_cause": info["cause"],
                    "severity": info["severity"],
                    "tags": ["nextjs", "api-routes", "serverless"]
                }
        
        # Generic API route error
        return {
            "category": "nextjs",
            "subcategory": "api_routes",
            "confidence": "medium",
            "suggested_fix": "Check API route handler signature and response formatting",
            "root_cause": "nextjs_api_route_error",
            "severity": "medium",
            "tags": ["nextjs", "api-routes"]
        }
    
    def analyze_app_router_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze Next.js App Router framework errors.
        
        Args:
            error_data: Error data with App Router-related issues
            
        Returns:
            Analysis results with App Router-specific fixes
        """
        message = error_data.get("message", "").lower()
        
        # App Router specific error patterns
        app_router_patterns = {
            "client component.*cannot.*import server component": {
                "cause": "nextjs_client_import_server",
                "fix": "Client components cannot import Server Components. Consider moving shared logic or using children prop",
                "severity": "error"
            },
            "client.*cannot.*use.*server.*hook": {
                "cause": "nextjs_client_server_hook",
                "fix": "Server hooks like cookies() can only be used in Server Components",
                "severity": "error"
            },
            "unstable_cache.*use error": {
                "cause": "nextjs_unstable_cache_error",
                "fix": "Check usage of unstable_cache and ensure it's used in a Server Component",
                "severity": "error"
            },
            "Route.*conflict.*app router": {
                "cause": "nextjs_app_route_conflict",
                "fix": "Resolve conflicting route definitions in app directory structure",
                "severity": "error"
            },
            "page\\.js.*layout\\.js.*conflict": {
                "cause": "nextjs_page_layout_conflict",
                "fix": "Resolve conflicts between page.js and layout.js files in the same directory",
                "severity": "error"
            },
            "invalid.*metadata.*export": {
                "cause": "nextjs_invalid_metadata",
                "fix": "Check metadata exports in layout.js or page.js file",
                "severity": "error"
            },
            "cannot use.*'use client'.*server component": {
                "cause": "nextjs_use_client_in_server",
                "fix": "Remove 'use client' directive from Server Component or move component to a separate file",
                "severity": "error"
            }
        }
        
        for pattern, info in app_router_patterns.items():
            if re.search(pattern, message):
                return {
                    "category": "nextjs",
                    "subcategory": "app_dir",
                    "confidence": "high",
                    "suggested_fix": info["fix"],
                    "root_cause": info["cause"],
                    "severity": info["severity"],
                    "tags": ["nextjs", "app-router", "server-components"]
                }
        
        # Generic App Router error
        return {
            "category": "nextjs",
            "subcategory": "app_dir",
            "confidence": "medium",
            "suggested_fix": "Check App Router implementation and directory structure",
            "root_cause": "nextjs_app_router_error",
            "severity": "medium",
            "tags": ["nextjs", "app-router"]
        }


class NextjsPatchGenerator:
    """
    Generates patches for Next.js errors based on analysis results.
    
    This class creates code fixes for common Next.js errors using templates
    and heuristics specific to Next.js patterns and best practices.
    """
    
    def __init__(self):
        """Initialize the Next.js patch generator."""
        self.template_dir = Path(__file__).parent.parent / "patch_generation" / "templates"
        self.nextjs_template_dir = self.template_dir / "nextjs"
        
        # Ensure template directory exists
        self.nextjs_template_dir.mkdir(parents=True, exist_ok=True)
        
        # Load patch templates
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, str]:
        """Load Next.js patch templates."""
        templates = {}
        
        if not self.nextjs_template_dir.exists():
            logger.warning(f"Next.js templates directory not found: {self.nextjs_template_dir}")
            return templates
        
        for template_file in self.nextjs_template_dir.glob("*.js.template"):
            try:
                with open(template_file, 'r') as f:
                    template_name = template_file.stem.replace('.js', '')
                    templates[template_name] = f.read()
                    logger.debug(f"Loaded Next.js template: {template_name}")
            except Exception as e:
                logger.error(f"Error loading Next.js template {template_file}: {e}")
        
        return templates
    
    def generate_patch(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                      source_code: str) -> Optional[Dict[str, Any]]:
        """
        Generate a patch for the Next.js error.
        
        Args:
            error_data: The Next.js error data
            analysis: Analysis results from NextjsExceptionHandler
            source_code: The source code where the error occurred
            
        Returns:
            Patch information or None if no patch can be generated
        """
        root_cause = analysis.get("root_cause", "")
        
        # Map root causes to patch strategies
        patch_strategies = {
            "nextjs_getserversideprops_return_type": self._fix_getserversideprops_return,
            "nextjs_getstaticprops_return_type": self._fix_getstaticprops_return,
            "nextjs_getstaticpaths_return_type": self._fix_getstaticpaths_return,
            "nextjs_absolute_url_fetch": self._fix_absolute_url_fetch,
            "nextjs_api_no_response": self._fix_api_no_response,
            "nextjs_api_method_not_allowed": self._fix_api_method_not_allowed,
            "nextjs_client_import_server": self._fix_client_import_server,
            "nextjs_use_client_in_server": self._fix_use_client_in_server,
            "nextjs_image_configuration": self._fix_image_configuration,
            "nextjs_middleware_config": self._fix_middleware_config
        }
        
        strategy = patch_strategies.get(root_cause)
        if strategy:
            try:
                return strategy(error_data, analysis, source_code)
            except Exception as e:
                logger.error(f"Error generating Next.js patch for {root_cause}: {e}")
        
        # Try to use templates if no specific strategy matches
        return self._template_based_patch(error_data, analysis, source_code)
    
    def _fix_getserversideprops_return(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                                      source_code: str) -> Optional[Dict[str, Any]]:
        """Fix getServerSideProps return value issues."""
        return {
            "type": "suggestion",
            "description": "Fix getServerSideProps return type",
            "fix_commands": [
                "Return an object with props property",
                "Ensure props is an object with data for the page",
                "Use notFound:true for 404 responses"
            ],
            "fix_code": """export async function getServerSideProps(context) {
  try {
    // Fetch data from external API or database
    const data = await fetchData(context.params.id);
    
    // Return data as props
    return {
      props: {
        data,
        // Additional props can be added here
      },
    };
  } catch (error) {
    // Return notFound for 404 response
    return {
      notFound: true,
    };
    
    // Or return error props
    // return {
    //   props: {
    //     error: error.message,
    //   },
    // };
  }
}"""
        }
    
    def _fix_getstaticprops_return(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                                  source_code: str) -> Optional[Dict[str, Any]]:
        """Fix getStaticProps return value issues."""
        return {
            "type": "suggestion",
            "description": "Fix getStaticProps return type",
            "fix_commands": [
                "Return an object with props property",
                "Add revalidate property for ISR if needed",
                "Use notFound:true for 404 responses"
            ],
            "fix_code": """export async function getStaticProps(context) {
  try {
    // Fetch data from external API or database
    const data = await fetchData(context.params.id);
    
    // Return data as props
    return {
      props: {
        data,
        // Additional props can be added here
      },
      // Revalidate the page every 60 seconds (optional)
      revalidate: 60,
    };
  } catch (error) {
    // Return notFound for 404 response
    return {
      notFound: true,
    };
    
    // Or return error props
    // return {
    //   props: {
    //     error: error.message,
    //   },
    // };
  }
}"""
        }
    
    def _fix_getstaticpaths_return(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                                  source_code: str) -> Optional[Dict[str, Any]]:
        """Fix getStaticPaths return value issues."""
        return {
            "type": "suggestion",
            "description": "Fix getStaticPaths return type",
            "fix_commands": [
                "Return an object with paths and fallback properties",
                "Ensure paths is an array of objects with params",
                "Set fallback to true, false, or 'blocking'"
            ],
            "fix_code": """export async function getStaticPaths() {
  // Fetch list of possible IDs
  const ids = await fetchIds();
  
  // Map to paths format
  const paths = ids.map(id => ({
    params: { id: id.toString() },
  }));
  
  return {
    paths,
    // Set fallback behavior:
    // false: 404 for ungenerated paths
    // true: shows fallback UI for ungenerated paths
    // 'blocking': server-renders pages on first request (like SSR)
    fallback: false,
  };
}"""
        }
    
    def _fix_absolute_url_fetch(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                               source_code: str) -> Optional[Dict[str, Any]]:
        """Fix absolute URL fetch issues."""
        return {
            "type": "suggestion",
            "description": "Fix absolute URL in fetch calls",
            "fix_commands": [
                "Use relative URLs for internal API routes",
                "Use absolute URLs with protocol for external APIs",
                "For API routes in getServerSideProps, use absolute URL with host from headers"
            ],
            "fix_code": """// For client-side fetches to internal API:
const data = await fetch('/api/endpoint');

// For server-side fetches to internal API:
export async function getServerSideProps({ req }) {
  // Create absolute URL from request headers
  const protocol = req.headers['x-forwarded-proto'] || 'http';
  const host = req.headers['x-forwarded-host'] || req.headers['host'];
  
  // Use absolute URL for internal API from server
  const data = await fetch(`${protocol}://${host}/api/endpoint`);
  
  // Or use relative URL
  // const data = await fetch('/api/endpoint', { 
  //   headers: { cookie: req.headers.cookie || '' } 
  // });
  
  return { props: { data: await data.json() } };
}

// For external API calls:
const externalData = await fetch('https://api.example.com/data');"""
        }
    
    def _fix_api_no_response(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                            source_code: str) -> Optional[Dict[str, Any]]:
        """Fix API route not sending response."""
        return {
            "type": "suggestion",
            "description": "Ensure API route sends a response",
            "fix_commands": [
                "Add res.status() and res.json() to send a proper response",
                "Ensure all code paths end with a response",
                "Check for early returns without responses"
            ],
            "fix_code": """// API route handler
export default async function handler(req, res) {
  try {
    // Process the request
    const data = await processRequest(req.body);
    
    // Send successful response
    return res.status(200).json({ data });
  } catch (error) {
    // Send error response
    return res.status(500).json({ error: error.message });
  }
  
  // Make sure all code paths send a response
  // This should never be reached if the try/catch works correctly
  return res.status(500).json({ error: 'Unknown error occurred' });
}"""
        }
    
    def _fix_api_method_not_allowed(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                                   source_code: str) -> Optional[Dict[str, Any]]:
        """Fix API route method not allowed."""
        return {
            "type": "suggestion",
            "description": "Handle different HTTP methods in API route",
            "fix_commands": [
                "Check request method with req.method",
                "Add handlers for each supported method",
                "Return 405 Method Not Allowed for unsupported methods"
            ],
            "fix_code": """// API route handler
export default async function handler(req, res) {
  // Check HTTP method
  const { method } = req;
  
  switch (method) {
    case 'GET':
      // Handle GET requests
      return res.status(200).json({ data: 'This is a GET response' });
      
    case 'POST':
      // Handle POST requests
      try {
        const result = await processData(req.body);
        return res.status(201).json({ result });
      } catch (error) {
        return res.status(400).json({ error: error.message });
      }
      
    case 'PUT':
      // Handle PUT requests
      // Similar to POST
      
    case 'DELETE':
      // Handle DELETE requests
      
    default:
      // Method not allowed for other methods
      res.setHeader('Allow', ['GET', 'POST', 'PUT', 'DELETE']);
      return res.status(405).json({ error: `Method ${method} Not Allowed` });
  }
}"""
        }
    
    def _fix_client_import_server(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                                 source_code: str) -> Optional[Dict[str, Any]]:
        """Fix client component importing server component."""
        return {
            "type": "suggestion",
            "description": "Fix client component importing server component",
            "fix_commands": [
                "Move shared logic to a separate utility file",
                "Use children or slots pattern to compose server and client components",
                "Use props to pass data from server to client components"
            ],
            "fix_code": """// Incorrect pattern:
// ClientComponent.js - 'use client'
// import ServerComponent from './ServerComponent'
//
// export default function ClientComponent() {
//   return <div><ServerComponent /></div>
// }

// Correct pattern 1: Use children prop
// ParentServerComponent.js
export default function ParentServerComponent() {
  // Fetch data in server component
  const data = fetchDataOnServer();
  
  return (
    <ClientComponent data={data}>
      <p>This content is rendered by the server</p>
    </ClientComponent>
  );
}

// ClientComponent.js - 'use client'
export default function ClientComponent({ data, children }) {
  // Client-side logic here
  const [state, setState] = useState(data);
  
  return (
    <div>
      <h1>Client component with server data: {state}</h1>
      {children} {/* Server-rendered content */}
    </div>
  );
}"""
        }
    
    def _fix_use_client_in_server(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                                 source_code: str) -> Optional[Dict[str, Any]]:
        """Fix 'use client' directive in server component."""
        return {
            "type": "suggestion",
            "description": "Fix 'use client' directive in server component",
            "fix_commands": [
                "Remove 'use client' directive from server component",
                "Split component into separate client and server components",
                "Move client-only logic to a dedicated client component"
            ],
            "fix_code": """// Instead of mixing server and client code:
// 'use client'  // This is causing the error
// export default function Component() {
//   // Server data fetching
//   // Client-side state and effects
// }

// Create separate files:

// ServerComponent.js (no directive)
import { ClientComponent } from './ClientComponent';

export default async function ServerComponent() {
  // Server-side data fetching
  const data = await fetchServerData();
  
  // Pass data to client component
  return <ClientComponent initialData={data} />;
}

// ClientComponent.js
'use client'
import { useState, useEffect } from 'react';

export function ClientComponent({ initialData }) {
  const [data, setData] = useState(initialData);
  const [clientState, setClientState] = useState(null);
  
  useEffect(() => {
    // Client-side effects
  }, []);
  
  return (
    <div>
      <h1>Server data: {data}</h1>
      <button onClick={() => setClientState('clicked')}>
        Client interaction
      </button>
    </div>
  );
}"""
        }
    
    def _fix_image_configuration(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                                source_code: str) -> Optional[Dict[str, Any]]:
        """Fix Next.js Image component configuration issues."""
        return {
            "type": "suggestion",
            "description": "Fix Next.js Image component configuration",
            "fix_commands": [
                "Configure Image domains in next.config.js",
                "Add width and height props to Image component",
                "Use loader or unoptimized prop for external images"
            ],
            "fix_code": """// In next.config.js
module.exports = {
  images: {
    domains: ['example.com', 'cdn.example.com'],
    // Optional: configure image sizes
    deviceSizes: [640, 750, 828, 1080, 1200, 1920, 2048],
    imageSizes: [16, 32, 48, 64, 96, 128, 256, 384],
  },
};

// In your component
import Image from 'next/image';

export default function MyComponent() {
  return (
    <div>
      {/* Local image with known dimensions */}
      <Image
        src="/images/profile.jpg"
        alt="Profile Picture"
        width={500}
        height={300}
        priority
      />
      
      {/* Remote image from configured domain */}
      <Image
        src="https://example.com/profile.jpg"
        alt="Remote Profile"
        width={500}
        height={300}
      />
      
      {/* Using fill for responsive images */}
      <div style={{ position: 'relative', width: '100%', height: '300px' }}>
        <Image
          src="/images/banner.jpg"
          alt="Banner"
          fill
          style={{ objectFit: 'cover' }}
        />
      </div>
      
      {/* For unconfigured external domains */}
      <Image
        src="https://unconfigured-domain.com/image.jpg"
        alt="External Image"
        width={500}
        height={300}
        unoptimized
      />
    </div>
  );
}"""
        }
    
    def _fix_middleware_config(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                              source_code: str) -> Optional[Dict[str, Any]]:
        """Fix Next.js middleware configuration issues."""
        return {
            "type": "suggestion",
            "description": "Fix Next.js middleware configuration",
            "fix_commands": [
                "Configure middleware matcher correctly",
                "Export middleware config with the correct format",
                "Ensure middleware returns a response or calls next()"
            ],
            "fix_code": """// middleware.js
import { NextResponse } from 'next/server';

// Middleware function
export function middleware(request) {
  // Get the pathname
  const { pathname } = request.nextUrl;
  
  // Example: redirect if not authenticated
  const token = request.cookies.get('token')?.value;
  
  // Check auth for protected routes
  if (pathname.startsWith('/dashboard') && !token) {
    // Redirect to login
    return NextResponse.redirect(new URL('/login', request.url));
  }
  
  // Example: rewrite for internationalization
  if (pathname.startsWith('/products')) {
    // Rewrite to internal route
    return NextResponse.rewrite(new URL('/api/products', request.url));
  }
  
  // Continue for other routes
  return NextResponse.next();
}

// Configure middleware matches
export const config = {
  // Matcher for specific paths
  matcher: [
    '/dashboard/:path*',
    '/products/:path*',
    '/api/:path*',
  ],
  
  // Or use more complex matchers
  // matcher: [
  //   {
  //     source: '/dashboard/:path*',
  //     has: [{ type: 'header', key: 'authorization' }],
  //   },
  //   {
  //     source: '/products/:path*',
  //     missing: [{ type: 'cookie', key: 'token' }],
  //   }
  // ]
};"""
        }
    
    def _template_based_patch(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                            source_code: str) -> Optional[Dict[str, Any]]:
        """Generate patch using templates."""
        root_cause = analysis.get("root_cause", "")
        
        # Map root causes to template names
        template_map = {
            "nextjs_getserversideprops_return_type": "getserversideprops_fix",
            "nextjs_getstaticprops_return_type": "getstaticprops_fix",
            "nextjs_getstaticpaths_return_type": "getstaticpaths_fix",
            "nextjs_api_no_response": "api_route_fix",
            "nextjs_api_method_not_allowed": "api_method_fix",
            "nextjs_image_configuration": "image_config_fix",
            "nextjs_middleware_config": "middleware_fix"
        }
        
        template_name = template_map.get(root_cause)
        if template_name and template_name in self.templates:
            template = self.templates[template_name]
            
            return {
                "type": "template",
                "template": template,
                "description": f"Applied Next.js template fix for {root_cause}"
            }
        
        return None


class NextjsLanguagePlugin(LanguagePlugin):
    """
    Main Next.js framework plugin for Homeostasis.
    
    This plugin orchestrates Next.js error analysis and patch generation,
    supporting data fetching, API routes, middleware, image optimization,
    static generation, and Vercel deployment issues.
    """
    
    VERSION = "1.0.0"
    AUTHOR = "Homeostasis Team"
    
    def __init__(self):
        """Initialize the Next.js language plugin."""
        self.language = "nextjs"
        self.supported_extensions = {".js", ".jsx", ".ts", ".tsx"}
        self.supported_frameworks = [
            "next", "nextjs", "next.js", "vercel", "create-next-app",
            "next-auth", "next-i18next", "next-seo", "next-pwa"
        ]
        
        # Initialize components
        self.adapter = JavaScriptErrorAdapter()  # Reuse JavaScript adapter
        self.exception_handler = NextjsExceptionHandler()
        self.patch_generator = NextjsPatchGenerator()
        
        logger.info("Next.js framework plugin initialized")
    
    def get_language_id(self) -> str:
        """Get the unique identifier for this language."""
        return "nextjs"
    
    def get_language_name(self) -> str:
        """Get the human-readable name of the framework."""
        return "Next.js"
    
    def get_language_version(self) -> str:
        """Get the version of the framework supported by this plugin."""
        return "12.x/13.x"
    
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
        if "next" in framework:
            return True
        
        # Check error message for Next.js-specific patterns
        message = error_data.get("message", "").lower()
        stack_trace = str(error_data.get("stack_trace", "")).lower()
        
        nextjs_patterns = [
            r"next",
            r"nextjs",
            r"next\.js",
            r"vercel",
            r"getserversideprops",
            r"getstaticprops",
            r"getstaticpaths",
            r"next/image",
            r"next/link",
            r"next/router",
            r"next/head",
            r"next/script",
            r"middleware",
            r"_app\.js",
            r"_document\.js",
            r"app/(.*)/page\.",
            r"app/(.*)/layout\.",
            r"pages/api/",
            r"pages/\[",
            r"server component",
            r"client component",
            r"incremental static",
            r"revalidate",
            r"notfound",
            r"fallback"
        ]
        
        for pattern in nextjs_patterns:
            if re.search(pattern, message + stack_trace):
                return True
        
        # Check file paths for Next.js-specific patterns
        file_path = error_data.get("file", "")
        nextjs_file_patterns = [
            r"/pages/",
            r"/app/",
            r"/components/",
            r"/middleware\.",
            r"/next\.config\.",
            r"/public/",
            r"/_app\.",
            r"/_document\.",
            r"/api/"
        ]
        
        for pattern in nextjs_file_patterns:
            if re.search(pattern, file_path):
                return True
        
        # Check for Next.js in package dependencies (if available)
        context = error_data.get("context", {})
        dependencies = context.get("dependencies", [])
        if any("next" in dep.lower() for dep in dependencies):
            return True
        
        return False
    
    def analyze_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a Next.js error.
        
        Args:
            error_data: Next.js error data
            
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
            
            # Check if it's a data fetching error
            if self._is_data_fetching_error(standard_error):
                analysis = self.exception_handler.analyze_data_fetching_error(standard_error)
            
            # Check if it's an API route error
            elif self._is_api_route_error(standard_error):
                analysis = self.exception_handler.analyze_api_route_error(standard_error)
            
            # Check if it's an App Router error
            elif self._is_app_router_error(standard_error):
                analysis = self.exception_handler.analyze_app_router_error(standard_error)
            
            # Default Next.js error analysis
            else:
                analysis = self.exception_handler.analyze_exception(standard_error)
            
            # Add plugin metadata
            analysis["plugin"] = "nextjs"
            analysis["language"] = "javascript"
            analysis["plugin_version"] = self.VERSION
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing Next.js error: {e}")
            return {
                "category": "nextjs",
                "subcategory": "unknown",
                "confidence": "low",
                "suggested_fix": "Unable to analyze Next.js error",
                "error": str(e),
                "plugin": "nextjs"
            }
    
    def _is_data_fetching_error(self, error_data: Dict[str, Any]) -> bool:
        """Check if this is a data fetching related error."""
        message = error_data.get("message", "").lower()
        
        data_fetching_patterns = [
            "getserversideprops",
            "getstaticprops",
            "getstaticpaths",
            "revalidate",
            "notfound",
            "fallback",
            "incremental static",
            "server-side props",
            "static generation"
        ]
        
        return any(pattern in message for pattern in data_fetching_patterns)
    
    def _is_api_route_error(self, error_data: Dict[str, Any]) -> bool:
        """Check if this is an API route related error."""
        message = error_data.get("message", "").lower()
        stack_trace = str(error_data.get("stack_trace", "")).lower()
        file_path = error_data.get("file", "").lower()
        
        api_patterns = [
            "api route",
            "api/",
            "pages/api",
            "app/api",
            "response",
            "request",
            "handler",
            "http method",
            "status code"
        ]
        
        # Check if file path contains API route patterns
        if "/api/" in file_path or "/pages/api/" in file_path or "/app/api/" in file_path:
            return True
        
        return any(pattern in message or pattern in stack_trace for pattern in api_patterns)
    
    def _is_app_router_error(self, error_data: Dict[str, Any]) -> bool:
        """Check if this is an App Router related error."""
        message = error_data.get("message", "").lower()
        stack_trace = str(error_data.get("stack_trace", "")).lower()
        file_path = error_data.get("file", "").lower()
        
        app_router_patterns = [
            "app/",
            "app router",
            "app directory",
            "layout.js",
            "page.js",
            "server component",
            "client component",
            "use client",
            "use server",
            "metadata",
            "route group"
        ]
        
        # Check if file path contains App Router patterns
        if "/app/" in file_path:
            return True
        
        return any(pattern in message or pattern in stack_trace for pattern in app_router_patterns)
    
    def generate_fix(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                    source_code: str) -> Optional[Dict[str, Any]]:
        """
        Generate a fix for the Next.js error.
        
        Args:
            error_data: The Next.js error data
            analysis: Analysis results
            source_code: Source code where the error occurred
            
        Returns:
            Fix information or None if no fix can be generated
        """
        try:
            return self.patch_generator.generate_patch(error_data, analysis, source_code)
        except Exception as e:
            logger.error(f"Error generating Next.js fix: {e}")
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
                "Next.js data fetching error detection",
                "API routes and serverless function healing",
                "Next.js middleware error resolution",
                "Image component and optimization issue detection",
                "Static generation and ISR error handling",
                "React Server Components in Next.js support",
                "App Router and app directory error resolution",
                "Pages Router and pages directory error handling",
                "Vercel deployment error detection",
                "Next.js configuration issue fixing"
            ],
            "environments": ["browser", "node", "vercel", "edge"]
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
register_plugin(NextjsLanguagePlugin())