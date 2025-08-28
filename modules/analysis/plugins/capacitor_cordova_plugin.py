"""
Capacitor/Cordova Framework Plugin for Homeostasis

This plugin enables Homeostasis to analyze and fix errors in Capacitor and Cordova applications.
It provides comprehensive error handling for hybrid mobile apps, plugin integration issues,
native bridge communication problems, and web-to-native interaction challenges.
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


class CapacitorCordovaExceptionHandler:
    """
    Handles Capacitor and Cordova-specific exceptions with comprehensive error detection and classification.
    
    This class provides logic for categorizing hybrid app errors, plugin integration issues,
    native bridge communication problems, and mobile deployment challenges.
    """
    
    def __init__(self):
        """Initialize the Capacitor/Cordova exception handler."""
        self.rule_categories = {
            "plugin_integration": "Plugin installation and integration errors",
            "native_bridge": "Native bridge communication and callback errors",
            "permissions": "Mobile permissions and capabilities errors",
            "platform_build": "Platform-specific build and deployment errors",
            "webview": "WebView and web-to-native communication errors",
            "lifecycle": "App lifecycle and background/foreground state errors",
            "device_apis": "Device API access and hardware integration errors",
            "config": "Configuration and platform-specific setup errors",
            "security": "Content Security Policy and security-related errors",
            "performance": "Performance and memory management in hybrid context",
            "navigation": "Navigation and deep linking errors",
            "storage": "Local storage and file system access errors"
        }
        
        # Load rules from different categories
        self.rules = self._load_rules()
        
        # Pre-compile regex patterns for better performance
        self._compile_patterns()
    
    def _load_rules(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load Capacitor/Cordova error rules from rule files."""
        rules = {}
        rules_dir = Path(__file__).parent.parent / "rules" / "capacitor_cordova"
        
        try:
            # Create rules directory if it doesn't exist
            rules_dir.mkdir(parents=True, exist_ok=True)
            
            # Load common Capacitor/Cordova rules
            common_rules_path = rules_dir / "capacitor_cordova_common_errors.json"
            if common_rules_path.exists():
                with open(common_rules_path, 'r') as f:
                    common_data = json.load(f)
                    rules["common"] = common_data.get("rules", [])
                    logger.info(f"Loaded {len(rules['common'])} common Capacitor/Cordova rules")
            else:
                rules["common"] = self._create_default_rules()
                self._save_default_rules(common_rules_path, rules["common"])
            
            # Load plugin-specific rules
            plugin_rules_path = rules_dir / "capacitor_cordova_plugin_errors.json"
            if plugin_rules_path.exists():
                with open(plugin_rules_path, 'r') as f:
                    plugin_data = json.load(f)
                    rules["plugins"] = plugin_data.get("rules", [])
                    logger.info(f"Loaded {len(rules['plugins'])} plugin-specific rules")
            else:
                rules["plugins"] = []
            
            # Load platform-specific rules
            platform_rules_path = rules_dir / "capacitor_cordova_platform_errors.json"
            if platform_rules_path.exists():
                with open(platform_rules_path, 'r') as f:
                    platform_data = json.load(f)
                    rules["platform"] = platform_data.get("rules", [])
                    logger.info(f"Loaded {len(rules['platform'])} platform-specific rules")
            else:
                rules["platform"] = []
                    
        except Exception as e:
            logger.error(f"Error loading Capacitor/Cordova rules: {e}")
            rules = {"common": self._create_default_rules(), "plugins": [], "platform": []}
        
        return rules
    
    def _create_default_rules(self) -> List[Dict[str, Any]]:
        """Create default Capacitor/Cordova error rules."""
        return [
            {
                "id": "capacitor_plugin_not_found",
                "pattern": r"Plugin.*not found|Plugin.*not installed|No such plugin",
                "category": "capacitor_cordova",
                "subcategory": "plugin_integration",
                "root_cause": "capacitor_plugin_not_found",
                "confidence": "high",
                "severity": "error",
                "suggestion": "Install the required plugin using npm/yarn and sync the platform",
                "tags": ["capacitor", "cordova", "plugins", "installation"],
                "reliability": "high"
            },
            {
                "id": "cordova_plugin_error",
                "pattern": r"cordova.*plugin.*error|Plugin.*failed to install",
                "category": "capacitor_cordova",
                "subcategory": "plugin_integration",
                "root_cause": "cordova_plugin_error",
                "confidence": "high",
                "severity": "error",
                "suggestion": "Check plugin compatibility and installation process",
                "tags": ["cordova", "plugins", "installation"],
                "reliability": "high"
            },
            {
                "id": "native_bridge_error",
                "pattern": r"Native.*bridge.*error|Bridge.*communication.*failed|Callback.*not found",
                "category": "capacitor_cordova",
                "subcategory": "native_bridge",
                "root_cause": "native_bridge_communication_error",
                "confidence": "medium",
                "severity": "error",
                "suggestion": "Check native bridge configuration and callback implementations",
                "tags": ["capacitor", "cordova", "native-bridge", "communication"],
                "reliability": "medium"
            },
            {
                "id": "permission_denied_mobile",
                "pattern": r"Permission denied|User denied.*permission|Permission.*not granted",
                "category": "capacitor_cordova",
                "subcategory": "permissions",
                "root_cause": "mobile_permission_denied",
                "confidence": "medium",
                "severity": "warning",
                "suggestion": "Request permissions properly and handle permission denials gracefully",
                "tags": ["capacitor", "cordova", "permissions", "mobile"],
                "reliability": "medium"
            },
            {
                "id": "webview_error",
                "pattern": r"WebView.*error|Failed to load.*in WebView|WebView.*crashed",
                "category": "capacitor_cordova",
                "subcategory": "webview",
                "root_cause": "webview_error",
                "confidence": "medium",
                "severity": "error",
                "suggestion": "Check WebView configuration and content loading",
                "tags": ["capacitor", "cordova", "webview", "loading"],
                "reliability": "medium"
            },
            {
                "id": "csp_violation",
                "pattern": r"Content Security Policy.*violation|CSP.*violation|unsafe.*inline",
                "category": "capacitor_cordova",
                "subcategory": "security",
                "root_cause": "csp_violation",
                "confidence": "high",
                "severity": "warning",
                "suggestion": "Update Content Security Policy or modify code to comply with CSP",
                "tags": ["capacitor", "cordova", "security", "csp"],
                "reliability": "high"
            },
            {
                "id": "platform_build_error",
                "pattern": r"Platform.*build.*failed|Build.*error.*android|Build.*error.*ios",
                "category": "capacitor_cordova",
                "subcategory": "platform_build",
                "root_cause": "platform_build_error",
                "confidence": "medium",
                "severity": "error",
                "suggestion": "Check platform-specific build configuration and dependencies",
                "tags": ["capacitor", "cordova", "build", "platform"],
                "reliability": "medium"
            }
        ]
    
    def _save_default_rules(self, file_path: Path, rules: List[Dict[str, Any]]):
        """Save default rules to file."""
        try:
            with open(file_path, 'w') as f:
                json.dump({"rules": rules}, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving default Capacitor/Cordova rules: {e}")
    
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
                    logger.warning(f"Invalid regex pattern in Capacitor/Cordova rule {rule.get('id', 'unknown')}: {e}")
    
    def analyze_exception(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a Capacitor/Cordova exception and determine its type and potential fixes.
        
        Args:
            error_data: Capacitor/Cordova error data in standard format
            
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
                "category": best_match.get("category", "capacitor_cordova"),
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
        
        # Boost confidence for Capacitor/Cordova-specific patterns
        message = error_data.get("message", "").lower()
        framework = error_data.get("framework", "").lower()
        
        if any(term in message or term in framework for term in ["capacitor", "cordova", "ionic"]):
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
        if "capacitor" in runtime:
            context_tags.add("capacitor")
        if "cordova" in runtime:
            context_tags.add("cordova")
        if "ionic" in runtime:
            context_tags.add("ionic")
        
        # Check for mobile platform context
        context = error_data.get("context", {})
        platform = context.get("platform", "").lower()
        if "android" in platform:
            context_tags.add("android")
        if "ios" in platform:
            context_tags.add("ios")
        if "mobile" in platform:
            context_tags.add("mobile")
        
        if context_tags & rule_tags:
            base_confidence += 0.1
        
        return min(base_confidence, 1.0)
    
    def _generic_analysis(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """Provide generic analysis for unmatched errors."""
        error_type = error_data.get("error_type", "Error")
        message = error_data.get("message", "").lower()
        
        # Basic categorization based on error patterns
        if "plugin" in message:
            category = "plugin_integration"
            suggestion = "Check plugin installation and configuration"
        elif "permission" in message:
            category = "permissions"
            suggestion = "Check app permissions and request permissions properly"
        elif "build" in message:
            category = "platform_build"
            suggestion = "Check platform-specific build configuration"
        elif "webview" in message:
            category = "webview"
            suggestion = "Check WebView configuration and content loading"
        elif "bridge" in message or "native" in message:
            category = "native_bridge"
            suggestion = "Check native bridge communication and callbacks"
        elif "csp" in message or "security" in message:
            category = "security"
            suggestion = "Check Content Security Policy configuration"
        elif "config" in message:
            category = "config"
            suggestion = "Check Capacitor/Cordova configuration files"
        else:
            category = "unknown"
            suggestion = "Review hybrid app implementation and check documentation"
        
        return {
            "category": "capacitor_cordova",
            "subcategory": category,
            "confidence": "low",
            "suggested_fix": suggestion,
            "root_cause": f"capacitor_cordova_{category}_error",
            "severity": "medium",
            "rule_id": "capacitor_cordova_generic_handler",
            "tags": ["capacitor", "cordova", "generic", category]
        }
    
    def analyze_plugin_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze Capacitor/Cordova plugin errors.
        
        Args:
            error_data: Error data with plugin-related issues
            
        Returns:
            Analysis results with plugin-specific fixes
        """
        message = error_data.get("message", "").lower()
        framework = error_data.get("framework", "").lower()
        
        # Plugin not found errors
        if "plugin" in message and ("not found" in message or "not installed" in message):
            return {
                "category": "capacitor_cordova",
                "subcategory": "plugin_integration",
                "confidence": "high",
                "suggested_fix": "Install the required plugin and sync platforms",
                "root_cause": "plugin_not_installed",
                "severity": "error",
                "tags": ["capacitor", "cordova", "plugins", "installation"],
                "fix_commands": [
                    "Install plugin: npm install @capacitor/plugin-name",
                    "Sync platforms: npx cap sync",
                    "For Cordova: cordova plugin add plugin-name",
                    "Check plugin compatibility with your platform versions"
                ]
            }
        
        # Plugin configuration errors
        if "plugin" in message and ("config" in message or "configuration" in message):
            return {
                "category": "capacitor_cordova",
                "subcategory": "plugin_integration",
                "confidence": "medium",
                "suggested_fix": "Check plugin configuration in capacitor.config.ts or config.xml",
                "root_cause": "plugin_configuration_error",
                "severity": "warning",
                "tags": ["capacitor", "cordova", "plugins", "configuration"]
            }
        
        # Plugin method not available
        if "method" in message and ("not available" in message or "not supported" in message):
            return {
                "category": "capacitor_cordova",
                "subcategory": "plugin_integration",
                "confidence": "medium",
                "suggested_fix": "Check if method is supported on current platform or plugin version",
                "root_cause": "plugin_method_not_available",
                "severity": "warning",
                "tags": ["capacitor", "cordova", "plugins", "compatibility"]
            }
        
        # Generic plugin error
        return {
            "category": "capacitor_cordova",
            "subcategory": "plugin_integration",
            "confidence": "medium",
            "suggested_fix": "Check plugin installation, configuration, and compatibility",
            "root_cause": "plugin_error",
            "severity": "error",
            "tags": ["capacitor", "cordova", "plugins"]
        }
    
    def analyze_platform_build_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze platform build errors.
        
        Args:
            error_data: Error data with platform build issues
            
        Returns:
            Analysis results with platform build specific fixes
        """
        message = error_data.get("message", "").lower()
        context = error_data.get("context", {})
        platform = context.get("platform", "").lower()
        
        # Android build errors
        if "android" in platform or "android" in message:
            if "gradle" in message:
                return {
                    "category": "capacitor_cordova",
                    "subcategory": "platform_build",
                    "confidence": "high",
                    "suggested_fix": "Check Android Gradle configuration and dependencies",
                    "root_cause": "android_gradle_build_error",
                    "severity": "error",
                    "tags": ["capacitor", "cordova", "android", "gradle"],
                    "fix_commands": [
                        "Clean and rebuild: npx cap run android",
                        "Check Android SDK and build tools versions",
                        "Update Gradle wrapper version",
                        "Check plugin compatibility with Android"
                    ]
                }
            
            if "sdk" in message:
                return {
                    "category": "capacitor_cordova",
                    "subcategory": "platform_build",
                    "confidence": "high",
                    "suggested_fix": "Configure Android SDK paths and versions",
                    "root_cause": "android_sdk_error",
                    "severity": "error",
                    "tags": ["capacitor", "cordova", "android", "sdk"]
                }
        
        # iOS build errors
        if "ios" in platform or "ios" in message:
            if "xcode" in message:
                return {
                    "category": "capacitor_cordova",
                    "subcategory": "platform_build",
                    "confidence": "high",
                    "suggested_fix": "Check Xcode project configuration and provisioning",
                    "root_cause": "ios_xcode_build_error",
                    "severity": "error",
                    "tags": ["capacitor", "cordova", "ios", "xcode"],
                    "fix_commands": [
                        "Open iOS project in Xcode and check for issues",
                        "Update provisioning profiles and certificates",
                        "Check iOS deployment target version",
                        "Sync Capacitor: npx cap sync ios"
                    ]
                }
            
            if "cocoapods" in message or "pod" in message:
                return {
                    "category": "capacitor_cordova",
                    "subcategory": "platform_build",
                    "confidence": "high",
                    "suggested_fix": "Run pod install and update CocoaPods dependencies",
                    "root_cause": "ios_cocoapods_error",
                    "severity": "error",
                    "tags": ["capacitor", "cordova", "ios", "cocoapods"]
                }
        
        # Generic build error
        return {
            "category": "capacitor_cordova",
            "subcategory": "platform_build",
            "confidence": "medium",
            "suggested_fix": "Check platform-specific build configuration and dependencies",
            "root_cause": "platform_build_error",
            "severity": "error",
            "tags": ["capacitor", "cordova", "build"]
        }


class CapacitorCordovaPatchGenerator:
    """
    Generates patches for Capacitor/Cordova errors based on analysis results.
    
    This class creates code fixes for common hybrid mobile app issues, plugin integration
    problems, and platform-specific deployment challenges.
    """
    
    def __init__(self):
        """Initialize the Capacitor/Cordova patch generator."""
        self.template_dir = Path(__file__).parent.parent / "patch_generation" / "templates"
        self.cap_cordova_template_dir = self.template_dir / "capacitor_cordova"
        
        # Ensure template directory exists
        self.cap_cordova_template_dir.mkdir(parents=True, exist_ok=True)
        
        # Load patch templates
        self.templates = self._load_templates()
        
        # Create default templates if they don't exist
        self._create_default_templates()
    
    def _load_templates(self) -> Dict[str, str]:
        """Load Capacitor/Cordova patch templates."""
        templates = {}
        
        if not self.cap_cordova_template_dir.exists():
            logger.warning(f"Capacitor/Cordova templates directory not found: {self.cap_cordova_template_dir}")
            return templates
        
        for template_file in self.cap_cordova_template_dir.glob("*.template"):
            try:
                with open(template_file, 'r') as f:
                    template_name = template_file.stem
                    templates[template_name] = f.read()
                    logger.debug(f"Loaded Capacitor/Cordova template: {template_name}")
            except Exception as e:
                logger.error(f"Error loading Capacitor/Cordova template {template_file}: {e}")
        
        return templates
    
    def _create_default_templates(self):
        """Create default Capacitor/Cordova templates if they don't exist."""
        default_templates = {
            "plugin_installation_fix.js.template": """
// Fix for plugin installation and usage
import { Capacitor } from '@capacitor/core';

// Check if plugin is available before using
async function safePluginUsage() {
  // For Capacitor
  if (Capacitor.isPluginAvailable('PluginName')) {
    const { PluginName } = await import('@capacitor/plugin-name');
    try {
      const result = await PluginName.someMethod();
      return result;
    } catch (error) {
      console.error('Plugin method failed:', error);
      // Provide fallback or graceful degradation
    }
  } else {
    console.warn('Plugin not available on this platform');
    // Provide web fallback if possible
  }
}

// For Cordova plugins
function safeCordovaPlugin() {
  if (window.cordova && window.cordova.plugins.SomePlugin) {
    window.cordova.plugins.SomePlugin.someMethod(
      (success) => {
        console.log('Plugin success:', success);
      },
      (error) => {
        console.error('Plugin error:', error);
      }
    );
  } else {
    console.warn('Cordova plugin not available');
  }
}
""",
            "permission_handling.js.template": """
// Safe permission handling for hybrid apps
import { Capacitor } from '@capacitor/core';

// Generic permission request helper
async function requestPermission(permissionType) {
  try {
    if (Capacitor.getPlatform() === 'web') {
      // Web permissions (if applicable)
      return await requestWebPermission(permissionType);
    }
    
    // For mobile platforms
    const { Permissions } = await import('@capacitor/permissions');
    
    // Check current permission status
    const status = await Permissions.query({ name: permissionType });
    
    if (status.state === 'granted') {
      return true;
    }
    
    if (status.state === 'denied') {
      // Permission was denied, show explanation
      showPermissionExplanation(permissionType);
      return false;
    }
    
    // Request permission
    const result = await Permissions.request({ name: permissionType });
    return result.state === 'granted';
    
  } catch (error) {
    console.error('Permission request failed:', error);
    return false;
  }
}

async function requestWebPermission(permissionType) {
  // Handle web-specific permissions
  if (permissionType === 'geolocation') {
    return new Promise((resolve) => {
      navigator.geolocation.getCurrentPosition(
        () => resolve(true),
        () => resolve(false)
      );
    });
  }
  return false;
}

function showPermissionExplanation(permissionType) {
  // Show user-friendly explanation for why permission is needed
  console.log(`Permission ${permissionType} is required for this feature.`);
}
""",
            "webview_csp_fix.html.template": """
<!-- Content Security Policy fix for hybrid apps -->
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>App</title>
    
    <!-- Updated CSP for Capacitor/Cordova apps -->
    <meta http-equiv="Content-Security-Policy" 
          content="default-src 'self' data: https://ssl.gstatic.com 'unsafe-eval' 'unsafe-inline'; 
                   object-src 'none'; 
                   style-src 'self' 'unsafe-inline'; 
                   script-src 'self' 'unsafe-inline' 'unsafe-eval'; 
                   media-src 'self' data: content:; 
                   img-src 'self' data: content: blob:; 
                   connect-src 'self' https: wss:; 
                   frame-src 'self';">
    
    <!-- For older Cordova apps -->
    <meta http-equiv="Content-Security-Policy" 
          content="default-src 'self' data: gap: https://ssl.gstatic.com 'unsafe-eval'; 
                   style-src 'self' 'unsafe-inline'; 
                   media-src *; 
                   img-src 'self' data: content:;">
    
    <meta name="format-detection" content="telephone=no">
    <meta name="msapplication-tap-highlight" content="no">
    <meta name="viewport" content="initial-scale=1, width=device-width, viewport-fit=cover">
    <meta name="color-scheme" content="light dark">
</head>
<body>
    <div id="app"></div>
    
    <script>
        // Safe script loading for hybrid apps
        function loadScript(src) {
            return new Promise((resolve, reject) => {
                const script = document.createElement('script');
                script.src = src;
                script.onload = resolve;
                script.onerror = reject;
                document.head.appendChild(script);
            });
        }
        
        // Initialize app safely
        document.addEventListener('DOMContentLoaded', function() {
            // App initialization code
        });
        
        // Handle Cordova device ready
        document.addEventListener('deviceready', function() {
            console.log('Cordova device ready');
        }, false);
    </script>
</body>
</html>
""",
            "platform_detection.js.template": """
// Platform detection and feature availability
import { Capacitor } from '@capacitor/core';

class PlatformManager {
  constructor() {
    this.platform = Capacitor.getPlatform();
    this.isNative = Capacitor.isNativePlatform();
  }
  
  // Check if running on specific platform
  isWeb() {
    return this.platform === 'web';
  }
  
  isIOS() {
    return this.platform === 'ios';
  }
  
  isAndroid() {
    return this.platform === 'android';
  }
  
  // Feature availability checks
  async isFeatureAvailable(featureName) {
    switch (featureName) {
      case 'camera':
        return this.isNative && Capacitor.isPluginAvailable('Camera');
      case 'geolocation':
        return this.isNative ? 
          Capacitor.isPluginAvailable('Geolocation') : 
          'geolocation' in navigator;
      case 'filesystem':
        return this.isNative && Capacitor.isPluginAvailable('Filesystem');
      default:
        return false;
    }
  }
  
  // Safe feature usage with fallbacks
  async useFeature(featureName, nativeImpl, webImpl) {
    if (await this.isFeatureAvailable(featureName)) {
      try {
        return await nativeImpl();
      } catch (error) {
        console.error(`Native ${featureName} failed:`, error);
        if (webImpl && this.isWeb()) {
          return await webImpl();
        }
        throw error;
      }
    } else if (webImpl && this.isWeb()) {
      return await webImpl();
    } else {
      throw new Error(`Feature ${featureName} not available`);
    }
  }
}

export default PlatformManager;
""",
            "build_configuration_fix.json.template": """
{
  "_comment": "Capacitor configuration fixes",
  "capacitor_config": {
    "appId": "com.example.app",
    "appName": "MyApp",
    "webDir": "dist",
    "server": {
      "androidScheme": "https"
    },
    "plugins": {
      "SplashScreen": {
        "launchShowDuration": 3000,
        "launchAutoHide": true
      },
      "Keyboard": {
        "resize": "body",
        "style": "dark",
        "resizeOnFullScreen": true
      }
    },
    "ios": {
      "scheme": "MyApp"
    },
    "android": {
      "allowMixedContent": true,
      "captureInput": true
    }
  },
  
  "_comment": "Cordova configuration fixes",
  "cordova_config": {
    "widget": {
      "id": "com.example.app",
      "version": "1.0.0",
      "preference": [
        {
          "name": "DisallowOverscroll",
          "value": "true"
        },
        {
          "name": "android-minSdkVersion",
          "value": "22"
        },
        {
          "name": "BackupWebStorage",
          "value": "none"
        }
      ],
      "platform": {
        "android": {
          "allow-intent": {
            "href": "market:*"
          }
        },
        "ios": {
          "allow-intent": {
            "href": "itms:*"
          }
        }
      }
    }
  }
}
"""
        }
        
        for template_name, template_content in default_templates.items():
            template_path = self.cap_cordova_template_dir / template_name
            if not template_path.exists():
                try:
                    with open(template_path, 'w') as f:
                        f.write(template_content)
                    logger.debug(f"Created default Capacitor/Cordova template: {template_name}")
                except Exception as e:
                    logger.error(f"Error creating default Capacitor/Cordova template {template_name}: {e}")
    
    def generate_patch(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                      source_code: str) -> Optional[Dict[str, Any]]:
        """
        Generate a patch for the Capacitor/Cordova error.
        
        Args:
            error_data: The Capacitor/Cordova error data
            analysis: Analysis results from CapacitorCordovaExceptionHandler
            source_code: The source code where the error occurred
            
        Returns:
            Patch information or None if no patch can be generated
        """
        root_cause = analysis.get("root_cause", "")
        
        # Map root causes to patch strategies
        patch_strategies = {
            "capacitor_plugin_not_found": self._fix_plugin_installation,
            "plugin_not_installed": self._fix_plugin_installation,  # Add this mapping
            "cordova_plugin_error": self._fix_cordova_plugin,
            "native_bridge_communication_error": self._fix_bridge_communication,
            "mobile_permission_denied": self._fix_permission_handling,
            "webview_error": self._fix_webview_issues,
            "csp_violation": self._fix_csp_violation,
            "platform_build_error": self._fix_platform_build,
            "android_gradle_build_error": self._fix_android_build,
            "ios_xcode_build_error": self._fix_ios_build
        }
        
        strategy = patch_strategies.get(root_cause)
        if strategy:
            try:
                return strategy(error_data, analysis, source_code)
            except Exception as e:
                logger.error(f"Error generating Capacitor/Cordova patch for {root_cause}: {e}")
        
        # Try to use templates if no specific strategy matches
        return self._template_based_patch(error_data, analysis, source_code)
    
    def _fix_plugin_installation(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                                source_code: str) -> Optional[Dict[str, Any]]:
        """Fix plugin installation and availability issues."""
        message = error_data.get("message", "")
        
        # Try to extract plugin name
        plugin_match = re.search(r"plugin[:\s]+['\"]?([^'\"\\s]+)['\"]?", message, re.IGNORECASE)
        plugin_name = plugin_match.group(1) if plugin_match else "PluginName"
        
        return {
            "type": "suggestion",
            "description": f"Install and configure {plugin_name} plugin",
            "fix_commands": [
                f"Install plugin: npm install @capacitor/{plugin_name.lower()}",
                "Sync platforms: npx cap sync",
                "For iOS: npx cap run ios",
                "For Android: npx cap run android",
                "Check plugin availability before usage"
            ],
            "template": "plugin_installation_fix",
            "code_example": f"""
// Safe plugin usage
import {{ Capacitor }} from '@capacitor/core';

if (Capacitor.isPluginAvailable('{plugin_name}')) {{
  const {{ {plugin_name} }} = await import('@capacitor/{plugin_name.lower()}');
  // Use plugin safely
}}
"""
        }
    
    def _fix_cordova_plugin(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                           source_code: str) -> Optional[Dict[str, Any]]:
        """Fix Cordova plugin issues."""
        return {
            "type": "suggestion",
            "description": "Fix Cordova plugin installation and configuration",
            "fix_commands": [
                "Install plugin: cordova plugin add plugin-name",
                "Check plugin compatibility with Cordova version",
                "Verify config.xml plugin configuration",
                "Build platforms: cordova build android/ios"
            ],
            "code_example": """
// Safe Cordova plugin usage
document.addEventListener('deviceready', function() {
  if (window.cordova && window.cordova.plugins.SomePlugin) {
    // Plugin is available
  }
}, false);
"""
        }
    
    def _fix_bridge_communication(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                                 source_code: str) -> Optional[Dict[str, Any]]:
        """Fix native bridge communication issues."""
        return {
            "type": "suggestion",
            "description": "Fix native bridge communication and callbacks",
            "fix_commands": [
                "Check if callback functions are properly defined",
                "Verify plugin method signatures match documentation",
                "Add error handling for bridge communication",
                "Ensure proper async/await usage with native calls"
            ],
            "code_example": """
// Safe bridge communication
try {
  const result = await NativePlugin.someMethod();
  // Handle success
} catch (error) {
  console.error('Bridge communication failed:', error);
  // Handle error gracefully
}
"""
        }
    
    def _fix_permission_handling(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                                source_code: str) -> Optional[Dict[str, Any]]:
        """Fix mobile permission handling."""
        return {
            "type": "suggestion",
            "description": "Implement proper permission handling",
            "fix_commands": [
                "Request permissions before using protected features",
                "Handle permission denials gracefully",
                "Provide user-friendly explanations for permissions",
                "Check permission status before each use"
            ],
            "template": "permission_handling",
            "code_example": """
// Request permission safely
const hasPermission = await requestPermission('camera');
if (hasPermission) {
  // Use camera
} else {
  // Show fallback or explanation
}
"""
        }
    
    def _fix_webview_issues(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                           source_code: str) -> Optional[Dict[str, Any]]:
        """Fix WebView loading and configuration issues."""
        return {
            "type": "suggestion",
            "description": "Fix WebView configuration and content loading",
            "fix_commands": [
                "Check WebView configuration in capacitor.config.ts",
                "Verify Content Security Policy settings",
                "Ensure all resources are properly loaded",
                "Test on different devices and WebView versions"
            ]
        }
    
    def _fix_csp_violation(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                          source_code: str) -> Optional[Dict[str, Any]]:
        """Fix Content Security Policy violations."""
        return {
            "type": "suggestion",
            "description": "Update Content Security Policy for hybrid apps",
            "fix_commands": [
                "Add 'unsafe-inline' and 'unsafe-eval' for hybrid apps if needed",
                "Include gap: protocol for Cordova",
                "Allow data: and blob: URLs for local resources",
                "Configure CSP in index.html meta tag"
            ],
            "template": "webview_csp_fix",
            "code_example": """
<!-- Updated CSP for hybrid apps -->
<meta http-equiv="Content-Security-Policy" 
      content="default-src 'self' data: gap: https://ssl.gstatic.com 'unsafe-eval' 'unsafe-inline';">
"""
        }
    
    def _fix_platform_build(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                           source_code: str) -> Optional[Dict[str, Any]]:
        """Fix general platform build issues."""
        return {
            "type": "suggestion",
            "description": "Fix platform-specific build configuration",
            "fix_commands": [
                "Clean and rebuild: npx cap sync && npx cap run [platform]",
                "Check platform-specific configuration files",
                "Update dependencies and build tools",
                "Verify SDK versions and paths"
            ]
        }
    
    def _fix_android_build(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                          source_code: str) -> Optional[Dict[str, Any]]:
        """Fix Android build issues."""
        return {
            "type": "suggestion",
            "description": "Fix Android build configuration",
            "fix_commands": [
                "Update Android SDK and build tools",
                "Check Gradle version compatibility",
                "Verify Android manifest permissions",
                "Clean Android build: cd android && ./gradlew clean"
            ]
        }
    
    def _fix_ios_build(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                      source_code: str) -> Optional[Dict[str, Any]]:
        """Fix iOS build issues."""
        return {
            "type": "suggestion",
            "description": "Fix iOS build configuration",
            "fix_commands": [
                "Update Xcode and iOS SDK",
                "Check provisioning profiles and certificates",
                "Run pod install in ios/ directory",
                "Verify iOS deployment target compatibility"
            ]
        }
    
    def _template_based_patch(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                            source_code: str) -> Optional[Dict[str, Any]]:
        """Generate patch using templates."""
        root_cause = analysis.get("root_cause", "")
        
        # Map root causes to template names
        template_map = {
            "capacitor_plugin_not_found": "plugin_installation_fix",
            "mobile_permission_denied": "permission_handling",
            "csp_violation": "webview_csp_fix",
            "platform_build_error": "build_configuration_fix"
        }
        
        template_name = template_map.get(root_cause)
        if template_name and template_name in self.templates:
            template = self.templates[template_name]
            
            return {
                "type": "template",
                "template": template,
                "description": f"Applied Capacitor/Cordova template fix for {root_cause}"
            }
        
        return None


class CapacitorCordovaLanguagePlugin(LanguagePlugin):
    """
    Main Capacitor/Cordova framework plugin for Homeostasis.
    
    This plugin orchestrates hybrid mobile app error analysis and patch generation,
    supporting both Capacitor and Cordova frameworks.
    """
    
    VERSION = "1.0.0"
    AUTHOR = "Homeostasis Team"
    
    def __init__(self):
        """Initialize the Capacitor/Cordova language plugin."""
        self.language = "capacitor_cordova"
        self.supported_extensions = {".js", ".ts", ".json", ".xml", ".html"}
        self.supported_frameworks = [
            "capacitor", "cordova", "ionic", "@capacitor", "@ionic"
        ]
        
        # Initialize components
        self.adapter = JavaScriptErrorAdapter()  # Reuse JavaScript adapter
        self.exception_handler = CapacitorCordovaExceptionHandler()
        self.patch_generator = CapacitorCordovaPatchGenerator()
        
        logger.info("Capacitor/Cordova framework plugin initialized")
    
    def get_language_id(self) -> str:
        """Get the unique identifier for this language."""
        return "capacitor_cordova"
    
    def get_language_name(self) -> str:
        """Get the human-readable name of the framework."""
        return "Capacitor/Cordova"
    
    def get_language_version(self) -> str:
        """Get the version of the framework supported by this plugin."""
        return "3.0+"
    
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
        if any(fw in framework for fw in ["capacitor", "cordova", "ionic"]):
            return True
        
        # Check runtime environment
        runtime = error_data.get("runtime", "").lower()
        if any(rt in runtime for rt in ["capacitor", "cordova", "ionic"]):
            return True
        
        # Check error message for Capacitor/Cordova-specific patterns
        message = error_data.get("message", "").lower()
        stack_trace = str(error_data.get("stack_trace", "")).lower()
        
        cap_cordova_patterns = [
            r"capacitor",
            r"cordova",
            r"ionic",
            r"@capacitor",
            r"@ionic",
            r"plugin.*not.*found",
            r"native.*bridge",
            r"webview",
            r"deviceready",
            r"phonegap",
            r"hybrid.*app",
            r"mobile.*app",
            r"native.*plugin",
            r"platform.*plugin"
        ]
        
        for pattern in cap_cordova_patterns:
            if re.search(pattern, message + stack_trace):
                return True
        
        # Check project structure indicators
        context = error_data.get("context", {})
        project_files = context.get("project_files", [])
        
        # Look for Capacitor/Cordova project files
        hybrid_project_indicators = [
            "capacitor.config.ts",
            "capacitor.config.js",
            "config.xml",
            "ionic.config.json",
            "package.json",
            "android/app/src/main/",
            "ios/App/",
            "www/",
            "src/",
            "platforms/"
        ]
        
        project_files_str = " ".join(project_files).lower()
        if any(indicator in project_files_str for indicator in hybrid_project_indicators):
            # Additional check for hybrid dependencies
            dependencies = context.get("dependencies", [])
            hybrid_dependencies = ["@capacitor", "@ionic", "cordova", "phonegap"]
            if any(any(hybrid_dep in dep.lower() for hybrid_dep in hybrid_dependencies) for dep in dependencies):
                return True
        
        return False
    
    def analyze_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a Capacitor/Cordova error.
        
        Args:
            error_data: Capacitor/Cordova error data
            
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
            
            # Check if it's a plugin-related error
            if self._is_plugin_error(standard_error):
                analysis = self.exception_handler.analyze_plugin_error(standard_error)
            
            # Check if it's a platform build error
            elif self._is_platform_build_error(standard_error):
                analysis = self.exception_handler.analyze_platform_build_error(standard_error)
            
            # Default Capacitor/Cordova error analysis
            else:
                analysis = self.exception_handler.analyze_exception(standard_error)
            
            # Add plugin metadata
            analysis["plugin"] = "capacitor_cordova"
            analysis["language"] = "capacitor_cordova"
            analysis["plugin_version"] = self.VERSION
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing Capacitor/Cordova error: {e}")
            return {
                "category": "capacitor_cordova",
                "subcategory": "unknown",
                "confidence": "low",
                "suggested_fix": "Unable to analyze Capacitor/Cordova error",
                "error": str(e),
                "plugin": "capacitor_cordova"
            }
    
    def _is_plugin_error(self, error_data: Dict[str, Any]) -> bool:
        """Check if this is a plugin related error."""
        message = error_data.get("message", "").lower()
        
        plugin_patterns = [
            "plugin",
            "not found",
            "not installed",
            "not available",
            "method",
            "callback"
        ]
        
        return any(pattern in message for pattern in plugin_patterns)
    
    def _is_platform_build_error(self, error_data: Dict[str, Any]) -> bool:
        """Check if this is a platform build related error."""
        message = error_data.get("message", "").lower()
        
        build_patterns = [
            "build",
            "gradle",
            "xcode",
            "android",
            "ios",
            "platform",
            "deployment",
            "compilation"
        ]
        
        return any(pattern in message for pattern in build_patterns)
    
    def generate_fix(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                    source_code: str) -> Optional[Dict[str, Any]]:
        """
        Generate a fix for the Capacitor/Cordova error.
        
        Args:
            error_data: The Capacitor/Cordova error data
            analysis: Analysis results
            source_code: Source code where the error occurred
            
        Returns:
            Fix information or None if no fix can be generated
        """
        try:
            return self.patch_generator.generate_patch(error_data, analysis, source_code)
        except Exception as e:
            logger.error(f"Error generating Capacitor/Cordova fix: {e}")
            return None
    
    def normalize_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize Capacitor/Cordova error data to standard format.
        
        Args:
            error_data: Capacitor/Cordova-specific error data
            
        Returns:
            Normalized error data
        """
        return self.adapter.to_standard_format(error_data)
    
    def denormalize_error(self, standard_error: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert standard error format back to Capacitor/Cordova format.
        
        Args:
            standard_error: Standard error data
            
        Returns:
            Capacitor/Cordova-specific error data
        """
        return self.adapter.from_standard_format(standard_error)
    
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
                "Capacitor plugin installation and configuration fixes",
                "Cordova plugin integration error resolution",
                "Native bridge communication debugging",
                "Mobile permissions and capabilities handling",
                "WebView configuration and CSP issue resolution",
                "Platform-specific build error fixes (Android/iOS)",
                "Hybrid app lifecycle and state management",
                "Device API access and hardware integration",
                "Content Security Policy optimization for mobile",
                "Cross-platform deployment issue resolution",
                "Ionic framework integration support"
            ],
            "platforms": ["mobile", "ios", "android", "hybrid"],
            "environments": ["capacitor", "cordova", "ionic", "hybrid-mobile"]
        }


# Register the plugin
register_plugin(CapacitorCordovaLanguagePlugin())