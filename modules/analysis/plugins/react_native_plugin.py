"""
React Native Framework Plugin for Homeostasis

This plugin enables Homeostasis to analyze and fix errors in React Native applications.
It provides comprehensive error handling for React Native components, native module issues,
platform-specific errors, navigation problems, and Metro bundler issues.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..language_adapters import JavaScriptErrorAdapter
from ..language_plugin_system import LanguagePlugin, register_plugin

logger = logging.getLogger(__name__)


class ReactNativeExceptionHandler:
    """
    Handles React Native-specific exceptions with comprehensive error detection and classification.

    This class provides logic for categorizing React Native errors including native modules,
    platform-specific issues, navigation errors, bundle loading issues, and bridge communication.
    """

    def __init__(self):
        """Initialize the React Native exception handler."""
        self.rule_categories = {
            "native_modules": "Native module integration errors",
            "bridge": "React Native bridge communication errors",
            "navigation": "React Navigation and routing errors",
            "platform": "Platform-specific iOS/Android errors",
            "bundler": "Metro bundler and build errors",
            "permissions": "Mobile permissions and capabilities errors",
            "styling": "React Native styling and layout errors",
            "lifecycle": "App lifecycle and state management errors",
            "networking": "Network and API communication errors",
            "storage": "AsyncStorage and data persistence errors",
            "animation": "Animation and gesture handling errors",
            "debugging": "Development and debugging errors",
        }

        # Load rules from different categories
        self.rules = self._load_rules()

        # Pre-compile regex patterns for better performance
        self._compile_patterns()

    def _load_rules(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load React Native error rules from rule files."""
        rules = {}
        rules_dir = Path(__file__).parent.parent / "rules" / "react_native"

        try:
            # Create rules directory if it doesn't exist
            rules_dir.mkdir(parents=True, exist_ok=True)

            # Load common React Native rules
            common_rules_path = rules_dir / "react_native_common_errors.json"
            if common_rules_path.exists():
                with open(common_rules_path, "r") as f:
                    common_data = json.load(f)
                    rules["common"] = common_data.get("rules", [])
                    logger.info(
                        f"Loaded {len(rules['common'])} common React Native rules"
                    )
            else:
                rules["common"] = self._create_default_rules()
                self._save_default_rules(common_rules_path, rules["common"])

            # Load native modules rules
            native_rules_path = rules_dir / "react_native_native_modules_errors.json"
            if native_rules_path.exists():
                with open(native_rules_path, "r") as f:
                    native_data = json.load(f)
                    rules["native"] = native_data.get("rules", [])
                    logger.info(f"Loaded {len(rules['native'])} native module rules")
            else:
                rules["native"] = []

            # Load navigation rules
            navigation_rules_path = rules_dir / "react_native_navigation_errors.json"
            if navigation_rules_path.exists():
                with open(navigation_rules_path, "r") as f:
                    navigation_data = json.load(f)
                    rules["navigation"] = navigation_data.get("rules", [])
                    logger.info(f"Loaded {len(rules['navigation'])} navigation rules")
            else:
                rules["navigation"] = []

        except Exception as e:
            logger.error(f"Error loading React Native rules: {e}")
            rules = {
                "common": self._create_default_rules(),
                "native": [],
                "navigation": [],
            }

        return rules

    def _create_default_rules(self) -> List[Dict[str, Any]]:
        """Create default React Native error rules."""
        return [
            {
                "id": "rn_red_screen_error",
                "pattern": r"(RedBox|Red screen|Fatal Exception).*React Native",
                "category": "react_native",
                "subcategory": "red_screen",
                "root_cause": "react_native_red_screen_error",
                "confidence": "high",
                "severity": "error",
                "suggestion": "Check the red screen error details and stack trace",
                "tags": ["react-native", "redbox", "fatal"],
                "reliability": "high",
            },
            {
                "id": "rn_native_module_not_found",
                "pattern": r"Native module.*cannot be null|NativeModule.*not found",
                "category": "react_native",
                "subcategory": "native_modules",
                "root_cause": "react_native_native_module_missing",
                "confidence": "high",
                "severity": "error",
                "suggestion": "Check if native module is properly linked and installed",
                "tags": ["react-native", "native-modules", "linking"],
                "reliability": "high",
            },
            {
                "id": "rn_metro_bundler_error",
                "pattern": r"Metro.*error|Unable to resolve module|Module.*does not exist",
                "category": "react_native",
                "subcategory": "bundler",
                "root_cause": "react_native_metro_bundler_error",
                "confidence": "high",
                "severity": "error",
                "suggestion": "Check Metro bundler configuration and module paths",
                "tags": ["react-native", "metro", "bundler"],
                "reliability": "high",
            },
            {
                "id": "rn_bridge_communication_error",
                "pattern": r"RCTBridge|Bridge.*error|Native.*bridge",
                "category": "react_native",
                "subcategory": "bridge",
                "root_cause": "react_native_bridge_error",
                "confidence": "medium",
                "severity": "error",
                "suggestion": "Check React Native bridge communication",
                "tags": ["react-native", "bridge", "communication"],
                "reliability": "medium",
            },
            {
                "id": "rn_permission_denied",
                "pattern": r"Permission denied|User denied.*permission",
                "category": "react_native",
                "subcategory": "permissions",
                "root_cause": "react_native_permission_denied",
                "confidence": "medium",
                "severity": "warning",
                "suggestion": "Check app permissions and request permissions properly",
                "tags": ["react-native", "permissions", "mobile"],
                "reliability": "medium",
            },
        ]

    def _save_default_rules(self, file_path: Path, rules: List[Dict[str, Any]]):
        """Save default rules to file."""
        try:
            with open(file_path, "w") as f:
                json.dump({"rules": rules}, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving default rules: {e}")

    def _compile_patterns(self):
        """Pre-compile regex patterns for better performance."""
        self.compiled_patterns: Dict[str, List[tuple[re.Pattern[str], Dict[str, Any]]]] = {}

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
                        f"Invalid regex pattern in React Native rule {rule.get('id', 'unknown')}: {e}"
                    )

    def analyze_exception(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a React Native exception and determine its type and potential fixes.

        Args:
            error_data: React Native error data in standard format

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
                "category": best_match.get("category", "react_native"),
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

        # Boost confidence for React Native-specific patterns
        message = error_data.get("message", "").lower()
        if "react native" in message or "react-native" in message:
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
        if "android" in runtime:
            context_tags.add("android")
        if "ios" in runtime:
            context_tags.add("ios")
        if "react-native" in runtime or "react native" in runtime:
            context_tags.add("react-native")

        framework = error_data.get("framework", "").lower()
        if "react-native" in framework:
            context_tags.add("react-native")

        if context_tags & rule_tags:
            base_confidence += 0.1

        return min(base_confidence, 1.0)

    def _generic_analysis(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """Provide generic analysis for unmatched errors."""
        message = error_data.get("message", "").lower()

        # Basic categorization based on error patterns
        if "native" in message and "module" in message:
            category = "native_modules"
            suggestion = "Check native module linking and installation"
        elif "metro" in message or "bundler" in message:
            category = "bundler"
            suggestion = "Check Metro bundler configuration and dependencies"
        elif "permission" in message:
            category = "permissions"
            suggestion = "Check app permissions configuration"
        elif "navigation" in message or "navigate" in message:
            category = "navigation"
            suggestion = "Check React Navigation configuration and route definitions"
        elif "bridge" in message:
            category = "bridge"
            suggestion = "Check React Native bridge communication"
        elif "style" in message or "layout" in message:
            category = "styling"
            suggestion = "Check React Native styling and layout properties"
        else:
            category = "unknown"
            suggestion = "Review React Native implementation and check documentation"

        return {
            "category": "react_native",
            "subcategory": category,
            "confidence": "low",
            "suggested_fix": suggestion,
            "root_cause": f"react_native_{category}_error",
            "severity": "medium",
            "rule_id": "react_native_generic_handler",
            "tags": ["react-native", "generic", category],
        }

    def analyze_native_module_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze React Native native module errors.

        Args:
            error_data: Error data with native module issues

        Returns:
            Analysis results with native module specific fixes
        """
        message = error_data.get("message", "").lower()

        # Common native module error patterns
        if "native module" in message and (
            "cannot be null" in message or "not found" in message
        ):
            return {
                "category": "react_native",
                "subcategory": "native_modules",
                "confidence": "high",
                "suggested_fix": "Check if native module is properly linked. Run 'react-native link' or configure autolinking",
                "root_cause": "react_native_native_module_missing",
                "severity": "error",
                "tags": ["react-native", "native-modules", "linking"],
                "fix_commands": [
                    "Check package.json for the dependency",
                    "Run 'npx react-native run-ios' or 'npx react-native run-android'",
                    "Verify native module is compatible with React Native version",
                    "Check iOS/Android native project configuration",
                ],
            }

        if "cocoapods" in message or "podfile" in message:
            return {
                "category": "react_native",
                "subcategory": "native_modules",
                "confidence": "high",
                "suggested_fix": "Run 'cd ios && pod install' to install iOS dependencies",
                "root_cause": "react_native_ios_pods_error",
                "severity": "error",
                "tags": ["react-native", "ios", "cocoapods"],
                "fix_commands": [
                    "cd ios && pod install",
                    "Check Podfile configuration",
                    "Update CocoaPods if needed",
                ],
            }

        if "gradle" in message or "android" in message:
            return {
                "category": "react_native",
                "subcategory": "native_modules",
                "confidence": "high",
                "suggested_fix": "Check Android Gradle configuration and dependencies",
                "root_cause": "react_native_android_gradle_error",
                "severity": "error",
                "tags": ["react-native", "android", "gradle"],
                "fix_commands": [
                    "Clean and rebuild Android project",
                    "Check android/build.gradle configuration",
                    "Verify Android SDK and build tools versions",
                ],
            }

        # Generic native module error
        return {
            "category": "react_native",
            "subcategory": "native_modules",
            "confidence": "medium",
            "suggested_fix": "Check native module installation and linking",
            "root_cause": "react_native_native_module_error",
            "severity": "error",
            "tags": ["react-native", "native-modules"],
        }

    def analyze_metro_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze React Native Metro bundler errors.

        Args:
            error_data: Error data with Metro bundler issues

        Returns:
            Analysis results with Metro bundler specific fixes
        """
        message = error_data.get("message", "").lower()

        if "unable to resolve module" in message:
            return {
                "category": "react_native",
                "subcategory": "bundler",
                "confidence": "high",
                "suggested_fix": "Check Metro bundler module resolution and ensure dependency is installed",
                "root_cause": "react_native_module_not_found",
                "severity": "error",
                "tags": ["react-native", "metro", "modules"],
                "fix_commands": [
                    "Check import/require statement spelling",
                    "Verify package is installed in node_modules",
                    "Clear Metro cache: npx react-native start --reset-cache",
                    "Check metro.config.js configuration",
                ],
            }

        if "module does not exist" in message or "cannot resolve" in message:
            return {
                "category": "react_native",
                "subcategory": "bundler",
                "confidence": "high",
                "suggested_fix": "Verify module exists and path is correct",
                "root_cause": "react_native_module_resolution_error",
                "severity": "error",
                "tags": ["react-native", "metro", "resolution"],
            }

        return {
            "category": "react_native",
            "subcategory": "bundler",
            "confidence": "medium",
            "suggested_fix": "Check Metro bundler configuration and clear cache",
            "root_cause": "react_native_metro_error",
            "severity": "error",
            "tags": ["react-native", "metro"],
        }


class ReactNativePatchGenerator:
    """
    Generates patches for React Native errors based on analysis results.

    This class creates code fixes for common React Native errors using templates
    and heuristics specific to React Native patterns and best practices.
    """

    def __init__(self):
        """Initialize the React Native patch generator."""
        self.template_dir = (
            Path(__file__).parent.parent / "patch_generation" / "templates"
        )
        self.rn_template_dir = self.template_dir / "react_native"

        # Ensure template directory exists
        self.rn_template_dir.mkdir(parents=True, exist_ok=True)

        # Load patch templates
        self.templates: Dict[str, str] = self._load_templates()

        # Create default templates if they don't exist
        self._create_default_templates()

    def _load_templates(self) -> Dict[str, str]:
        """Load React Native patch templates."""
        templates: Dict[str, str] = {}

        if not self.rn_template_dir.exists():
            logger.warning(
                f"React Native templates directory not found: {self.rn_template_dir}"
            )
            return templates

        for template_file in self.rn_template_dir.glob("*.template"):
            try:
                with open(template_file, "r") as f:
                    template_name = template_file.stem
                    templates[template_name] = f.read()
                    logger.debug(f"Loaded React Native template: {template_name}")
            except Exception as e:
                logger.error(
                    f"Error loading React Native template {template_file}: {e}"
                )

        return templates

    def _create_default_templates(self):
        """Create default React Native templates if they don't exist."""
        default_templates = {
            "native_module_import_fix.js.template": """
// Fix for missing native module import
import { NativeModules } from 'react-native';

// Check if the native module exists before using it
const { {{MODULE_NAME}} } = NativeModules;

if (!{{MODULE_NAME}}) {
  console.warn('{{MODULE_NAME}} native module not found. Make sure it is properly linked.');
  // Provide fallback or graceful degradation
}

// Safe usage of native module
if ({{MODULE_NAME}}) {
  {{MODULE_NAME}}.{{METHOD_NAME}}({{PARAMS}});
}
""",
            "metro_module_resolution_fix.js.template": """
// Fix for Metro module resolution error
// Check the import path and ensure the module exists

// Instead of:
// import SomeModule from './path/that/doesnt/exist';

// Use correct path:
// import SomeModule from './correct/path/to/module';

// Or install missing dependency:
// npm install missing-package
// yarn add missing-package

// Then import:
// import SomeModule from 'missing-package';
""",
            "permission_handling_fix.js.template": """
import { Platform, PermissionsAndroid } from 'react-native';

// Safe permission handling for React Native
const requestPermission = async () => {
  if (Platform.OS === 'android') {
    try {
      const granted = await PermissionsAndroid.request(
        PermissionsAndroid.PERMISSIONS.{{PERMISSION_NAME}},
        {
          title: '{{PERMISSION_TITLE}}',
          message: '{{PERMISSION_MESSAGE}}',
          buttonNeutral: 'Ask Me Later',
          buttonNegative: 'Cancel',
          buttonPositive: 'OK',
        }
      );
      return granted === PermissionsAndroid.RESULTS.GRANTED;
    } catch (err) {
      console.warn('Permission request error:', err);
      return false;
    }
  }
  return true; // iOS permissions handled differently
};
""",
            "styling_fix.js.template": """
import { StyleSheet, Platform } from 'react-native';

// Fix for React Native styling issues
const styles = StyleSheet.create({
  container: {
    flex: 1,
    // Use specific values instead of undefined or invalid CSS properties
    backgroundColor: '#ffffff',
    // Platform-specific styling
    ...Platform.select({
      ios: {
        paddingTop: 20, // iOS status bar
      },
      android: {
        paddingTop: 0,
      },
    }),
  },
});
""",
        }

        for template_name, template_content in default_templates.items():
            template_path = self.rn_template_dir / template_name
            if not template_path.exists():
                try:
                    with open(template_path, "w") as f:
                        f.write(template_content)
                    logger.debug(f"Created default template: {template_name}")
                except Exception as e:
                    logger.error(
                        f"Error creating default template {template_name}: {e}"
                    )

    def generate_patch(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a patch for the React Native error.

        Args:
            error_data: The React Native error data
            analysis: Analysis results from ReactNativeExceptionHandler
            source_code: The source code where the error occurred

        Returns:
            Patch information or None if no patch can be generated
        """
        root_cause = analysis.get("root_cause", "")

        # Map root causes to patch strategies
        patch_strategies = {
            "react_native_native_module_missing": self._fix_native_module_missing,
            "react_native_metro_bundler_error": self._fix_metro_bundler_error,
            "react_native_module_not_found": self._fix_module_not_found,
            "react_native_permission_denied": self._fix_permission_denied,
            "react_native_bridge_error": self._fix_bridge_error,
            "react_native_ios_pods_error": self._fix_ios_pods_error,
            "react_native_android_gradle_error": self._fix_android_gradle_error,
            "react_native_styling_error": self._fix_styling_error,
        }

        strategy = patch_strategies.get(root_cause)
        if strategy:
            try:
                return strategy(error_data, analysis, source_code)
            except Exception as e:
                logger.error(
                    f"Error generating React Native patch for {root_cause}: {e}"
                )

        # Try to use templates if no specific strategy matches
        return self._template_based_patch(error_data, analysis, source_code)

    def _fix_native_module_missing(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix missing native module errors."""
        message = error_data.get("message", "")

        # Extract module name if possible
        module_match = re.search(
            r"'([^']+)'|\"([^\"]+)\"|(\w+)\s*native module", message, re.IGNORECASE
        )
        module_name = None
        if module_match:
            module_name = (
                module_match.group(1) or module_match.group(2) or module_match.group(3)
            )

        return {
            "type": "suggestion",
            "description": f"Native module {module_name or 'unknown'} is not properly linked",
            "fix_commands": [
                "Check if the native module package is installed: npm list <package-name>",
                "For React Native 0.60+: Ensure autolinking is working properly",
                "For older versions: Run 'react-native link <package-name>'",
                "For iOS: Run 'cd ios && pod install'",
                "For Android: Check MainApplication.java for manual linking",
                "Rebuild the app: 'npx react-native run-ios' or 'npx react-native run-android'",
            ],
            "template": "native_module_import_fix",
            "template_variables": {
                "MODULE_NAME": module_name or "YourNativeModule",
                "METHOD_NAME": "someMethod",
                "PARAMS": "param1, param2",
            },
        }

    def _fix_metro_bundler_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix Metro bundler errors."""
        return {
            "type": "suggestion",
            "description": "Metro bundler configuration or caching issue",
            "fix_commands": [
                "Clear Metro cache: 'npx react-native start --reset-cache'",
                "Clear node_modules and reinstall: 'rm -rf node_modules && npm install'",
                "Check metro.config.js configuration",
                "Restart Metro bundler: 'npx react-native start'",
                "Check for conflicting Metro configurations",
            ],
        }

    def _fix_module_not_found(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix module resolution errors."""
        message = error_data.get("message", "")

        # Extract module name from error message
        module_match = re.search(r"'([^']+)'|\"([^\"]+)\"", message)
        module_name = (
            module_match.group(1) or module_match.group(2)
            if module_match
            else "unknown"
        )

        return {
            "type": "suggestion",
            "description": f"Module '{module_name}' cannot be resolved",
            "fix_commands": [
                f"Check if '{module_name}' is installed: npm list {module_name}",
                f"Install the module: npm install {module_name}",
                "Check the import/require path spelling",
                "Verify the module exists in node_modules",
                "Clear Metro cache and restart: npx react-native start --reset-cache",
            ],
            "template": "metro_module_resolution_fix",
        }

    def _fix_permission_denied(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix permission denied errors."""
        return {
            "type": "suggestion",
            "description": "App permission not granted or properly requested",
            "fix_commands": [
                "Add permission to android/app/src/main/AndroidManifest.xml",
                "Add permission to ios/YourApp/Info.plist",
                "Request permission at runtime using PermissionsAndroid",
                "Check permission request timing and user flow",
            ],
            "template": "permission_handling_fix",
            "template_variables": {
                "PERMISSION_NAME": "CAMERA",
                "PERMISSION_TITLE": "Camera Permission",
                "PERMISSION_MESSAGE": "This app needs access to camera to take photos",
            },
        }

    def _fix_bridge_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix React Native bridge communication errors."""
        return {
            "type": "suggestion",
            "description": "React Native bridge communication issue",
            "fix_commands": [
                "Check if the app is running in debug or release mode",
                "Verify Metro bundler is running and accessible",
                "Check network connectivity between app and Metro",
                "Restart the app and Metro bundler",
                "Check for JavaScript errors that might crash the bridge",
            ],
        }

    def _fix_ios_pods_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix iOS CocoaPods errors."""
        return {
            "type": "suggestion",
            "description": "iOS CocoaPods installation or configuration issue",
            "fix_commands": [
                "Run 'cd ios && pod install' to install iOS dependencies",
                "Update CocoaPods: 'sudo gem install cocoapods'",
                "Clean pods: 'cd ios && pod deintegrate && pod install'",
                "Check Podfile for syntax errors",
                "Verify iOS deployment target compatibility",
            ],
        }

    def _fix_android_gradle_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix Android Gradle build errors."""
        return {
            "type": "suggestion",
            "description": "Android Gradle build configuration issue",
            "fix_commands": [
                "Clean Android build: 'cd android && ./gradlew clean'",
                "Check android/build.gradle and android/app/build.gradle",
                "Verify Android SDK and build tools versions",
                "Check compileSdkVersion and targetSdkVersion compatibility",
                "Update Gradle wrapper if needed",
            ],
        }

    def _fix_styling_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix React Native styling errors."""
        return {
            "type": "suggestion",
            "description": "React Native styling issue",
            "fix_commands": [
                "Use valid React Native style properties (not CSS)",
                "Check for undefined or null style values",
                "Use StyleSheet.create() for better performance",
                "Avoid using CSS-specific properties like 'display: block'",
                "Use platform-specific styling where needed",
            ],
            "template": "styling_fix",
        }

    def _template_based_patch(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Generate patch using templates."""
        root_cause = analysis.get("root_cause", "")

        # Map root causes to template names
        template_map = {
            "react_native_native_module_missing": "native_module_import_fix",
            "react_native_module_not_found": "metro_module_resolution_fix",
            "react_native_permission_denied": "permission_handling_fix",
            "react_native_styling_error": "styling_fix",
        }

        template_name = template_map.get(root_cause)
        if template_name and template_name in self.templates:
            template = self.templates[template_name]

            return {
                "type": "template",
                "template": template,
                "description": f"Applied React Native template fix for {root_cause}",
            }

        return None


class ReactNativeLanguagePlugin(LanguagePlugin):
    """
    Main React Native framework plugin for Homeostasis.

    This plugin orchestrates React Native error analysis and patch generation,
    supporting both iOS and Android React Native applications.
    """

    VERSION = "1.0.0"
    AUTHOR = "Homeostasis Team"

    def __init__(self):
        """Initialize the React Native language plugin."""
        self.language = "react_native"
        self.supported_extensions = {".js", ".jsx", ".ts", ".tsx"}
        self.supported_frameworks = [
            "react-native",
            "expo",
            "@react-native-community",
            "react-native-cli",
            "expo-cli",
        ]

        # Initialize components
        self.adapter = JavaScriptErrorAdapter()  # Reuse JavaScript adapter
        self.exception_handler = ReactNativeExceptionHandler()
        self.patch_generator = ReactNativePatchGenerator()

        logger.info("React Native framework plugin initialized")

    def get_language_id(self) -> str:
        """Get the unique identifier for this language."""
        return "react_native"

    def get_language_name(self) -> str:
        """Get the human-readable name of the framework."""
        return "React Native"

    def get_language_version(self) -> str:
        """Get the version of the framework supported by this plugin."""
        return "0.60+"

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
        if (
            "react-native" in framework
            or "react native" in framework
            or "expo" in framework
        ):
            return True

        # Check runtime environment
        runtime = error_data.get("runtime", "").lower()
        if any(
            platform in runtime
            for platform in ["react-native", "react native", "expo", "android", "ios"]
        ):
            # Additional check to ensure it's React Native, not just Android/iOS
            message = error_data.get("message", "").lower()
            stack_trace = str(error_data.get("stack_trace", "")).lower()
            if any(
                rn_indicator in message + stack_trace
                for rn_indicator in [
                    "react-native",
                    "react native",
                    "metro",
                    "rctbridge",
                ]
            ):
                return True

        # Check error message for React Native-specific patterns
        message = error_data.get("message", "").lower()
        stack_trace = str(error_data.get("stack_trace", "")).lower()

        react_native_patterns = [
            r"react-native",
            r"react native",
            r"metro",
            r"rctbridge",
            r"native module",
            r"redbox",
            r"red screen",
            r"expo",
            r"@react-native-community",
            r"react-native-",
            r"\.native\.",
            r"android.*react",
            r"ios.*react",
            r"cocoapods.*react",
            r"gradle.*react",
        ]

        for pattern in react_native_patterns:
            if re.search(pattern, message + stack_trace):
                return True

        # Check project structure indicators
        context = error_data.get("context", {})
        project_files = context.get("project_files", [])

        # Look for React Native project files
        rn_project_indicators = [
            "package.json",
            "metro.config.js",
            "react-native.config.js",
            "app.json",  # Expo
            "eas.json",  # Expo
            "android/app/build.gradle",
            "ios/Podfile",
        ]

        project_files_str = " ".join(project_files).lower()
        if any(indicator in project_files_str for indicator in rn_project_indicators):
            # Check if it's actually React Native by looking for RN dependencies
            dependencies = context.get("dependencies", [])
            rn_dependencies = ["react-native", "@react-native", "expo"]
            if any(
                any(rn_dep in dep.lower() for rn_dep in rn_dependencies)
                for dep in dependencies
            ):
                return True

        return False

    def analyze_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a React Native error.

        Args:
            error_data: React Native error data

        Returns:
            Analysis results
        """
        try:
            # Ensure error data is in standard format
            if not error_data.get("language"):
                standard_error = self.adapter.to_standard_format(error_data)
            else:
                standard_error = error_data

            # Check if it's a native module error
            if self._is_native_module_error(standard_error):
                analysis = self.exception_handler.analyze_native_module_error(
                    standard_error
                )

            # Check if it's a Metro bundler error
            elif self._is_metro_error(standard_error):
                analysis = self.exception_handler.analyze_metro_error(standard_error)

            # Default React Native error analysis
            else:
                analysis = self.exception_handler.analyze_exception(standard_error)

            # Add plugin metadata
            analysis["plugin"] = "react_native"
            analysis["language"] = "react_native"
            analysis["plugin_version"] = self.VERSION

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing React Native error: {e}")
            return {
                "category": "react_native",
                "subcategory": "unknown",
                "confidence": "low",
                "suggested_fix": "Unable to analyze React Native error",
                "error": str(e),
                "plugin": "react_native",
            }

    def _is_native_module_error(self, error_data: Dict[str, Any]) -> bool:
        """Check if this is a native module related error."""
        message = error_data.get("message", "").lower()

        native_module_patterns = [
            "native module",
            "nativemodules",
            "cannot be null",
            "not found",
            "cocoapods",
            "podfile",
            "gradle",
            "android.*link",
            "ios.*link",
        ]

        return any(pattern in message for pattern in native_module_patterns)

    def _is_metro_error(self, error_data: Dict[str, Any]) -> bool:
        """Check if this is a Metro bundler related error."""
        message = error_data.get("message", "").lower()

        metro_patterns = [
            "metro",
            "unable to resolve module",
            "module does not exist",
            "cannot resolve",
            "bundler",
            "bundle.*error",
        ]

        return any(pattern in message for pattern in metro_patterns)

    def generate_fix(
        self, analysis: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a fix for the React Native error.

        Args:
            analysis: Analysis results
            context: Additional context containing error_data and source_code

        Returns:
            Fix information or empty dict if no fix can be generated
        """
        try:
            # Extract error data and source code from context
            error_data = context.get("error_data", {})
            source_code = context.get("source_code", "")

            result = self.patch_generator.generate_patch(
                error_data, analysis, source_code
            )
            return result if result is not None else {}
        except Exception as e:
            logger.error(f"Error generating React Native fix: {e}")
            return {}

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
                "React Native component error handling",
                "Native module linking and integration fixes",
                "Metro bundler configuration and caching issues",
                "iOS CocoaPods dependency management",
                "Android Gradle build error resolution",
                "Platform-specific error detection (iOS/Android)",
                "React Native bridge communication debugging",
                "Permission handling and mobile capabilities",
                "Navigation and routing error fixes",
                "Styling and layout issue detection",
                "Expo integration support",
            ],
            "platforms": ["ios", "android", "mobile"],
            "environments": ["react-native", "expo", "mobile"],
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
register_plugin(ReactNativeLanguagePlugin())
