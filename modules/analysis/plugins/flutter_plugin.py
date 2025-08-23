"""
Flutter/Dart Framework Plugin for Homeostasis

This plugin enables Homeostasis to analyze and fix errors in Flutter applications.
It provides comprehensive error handling for Flutter widgets, Dart language errors,
state management issues, and Flutter-specific development problems.
"""
import logging
import re
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union, Set

from ..language_plugin_system import LanguagePlugin, register_plugin

logger = logging.getLogger(__name__)


class DartErrorAdapter:
    """
    Adapter for converting Dart/Flutter errors to the standard error format.
    """
    
    def to_standard_format(self, dart_error: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert Dart/Flutter error to standard format.
        
        Args:
            dart_error: Raw Dart/Flutter error data
            
        Returns:
            Standardized error format
        """
        # Extract common fields
        error_type = dart_error.get("type", dart_error.get("error_type", "Error"))
        message = dart_error.get("message", dart_error.get("description", ""))
        stack_trace = dart_error.get("stackTrace", dart_error.get("stack_trace", []))
        
        # Handle Flutter-specific error fields
        widget_info = dart_error.get("widget", {})
        library_error_message = dart_error.get("libraryErrorMessage", "")
        
        # Combine messages if we have library error message
        if library_error_message and library_error_message != message:
            message = f"{message}\nLibrary Error: {library_error_message}"
        
        # Extract file and line information from stack trace
        file_info = self._extract_file_info(stack_trace)
        
        return {
            "error_type": error_type,
            "message": message,
            "stack_trace": stack_trace,
            "language": "dart",
            "framework": "flutter",
            "runtime": dart_error.get("runtime", "flutter"),
            "timestamp": dart_error.get("timestamp"),
            "file": file_info.get("file"),
            "line": file_info.get("line"),
            "column": file_info.get("column"),
            "widget_type": widget_info.get("type") if widget_info else None,
            "widget_properties": widget_info.get("properties", {}) if widget_info else {},
            "context": {
                "flutter_version": dart_error.get("flutterVersion"),
                "dart_version": dart_error.get("dartVersion"),
                "platform": dart_error.get("platform"),
                "debug_mode": dart_error.get("debugMode", True),
                "project_path": dart_error.get("projectPath")
            }
        }
    
    def _extract_file_info(self, stack_trace: Union[List, str]) -> Dict[str, Any]:
        """Extract file, line, and column information from stack trace."""
        if not stack_trace:
            return {}
        
        # Convert to string if it's a list
        if isinstance(stack_trace, list):
            stack_str = "\n".join([str(frame) for frame in stack_trace])
        else:
            stack_str = str(stack_trace)
        
        # Common Dart/Flutter stack trace patterns
        patterns = [
            r'([^:\s]+\.dart):(\d+):(\d+)',  # file.dart:line:column
            r'([^:\s]+\.dart) (\d+):(\d+)',  # file.dart line:column
            r'at ([^:\s]+\.dart):(\d+)',     # at file.dart:line
        ]
        
        for pattern in patterns:
            match = re.search(pattern, stack_str)
            if match:
                if len(match.groups()) >= 3:
                    return {
                        "file": match.group(1),
                        "line": int(match.group(2)),
                        "column": int(match.group(3)) if match.group(3).isdigit() else 0
                    }
                else:
                    return {
                        "file": match.group(1),
                        "line": int(match.group(2)) if len(match.groups()) >= 2 else 0,
                        "column": 0
                    }
        
        return {}


class FlutterExceptionHandler:
    """
    Handles Flutter/Dart-specific exceptions with comprehensive error detection and classification.
    
    This class provides logic for categorizing Flutter widget errors, Dart language issues,
    state management problems, and Flutter development-specific errors.
    """
    
    def __init__(self):
        """Initialize the Flutter exception handler."""
        self.rule_categories = {
            "common": "Common Flutter framework errors",
            "widgets": "Flutter widget and UI errors",
            "navigation": "Flutter navigation and routing errors",
            "state_management": "State management and lifecycle errors",
            "performance": "Flutter performance and rendering issues",
            "platform": "Platform-specific errors and integrations",
            "async": "Dart async/await and Future errors",
            "null_safety": "Dart null safety violations",
            "build": "Flutter build and compilation errors",
            "plugins": "Flutter plugin and platform channel errors",
            "animation": "Flutter animation and transition errors",
            "layout": "Flutter layout and constraint errors",
            "testing": "Flutter testing and widget testing errors",
            "dart_core": "Core Dart language errors"
        }
        
        # Load rules from different categories
        self.rules = self._load_rules()
        
        # Pre-compile regex patterns for better performance
        self._compile_patterns()
    
    def _load_rules(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load Flutter/Dart error rules from rule files."""
        rules = {}
        rules_dir = Path(__file__).parent.parent / "rules" / "flutter"
        
        try:
            # Create rules directory if it doesn't exist
            rules_dir.mkdir(parents=True, exist_ok=True)
            
            # Load all Flutter rule files
            rule_files = {
                "common": "flutter_common_errors.json",
                "widgets": "flutter_widget_errors.json", 
                "navigation": "flutter_navigation_errors.json",
                "state_management": "flutter_state_management_errors.json",
                "performance": "flutter_performance_errors.json",
                "platform": "flutter_platform_errors.json"
            }
            
            for category, filename in rule_files.items():
                file_path = rules_dir / filename
                if file_path.exists():
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                            rules[category] = data.get("rules", [])
                            logger.info(f"Loaded {len(rules[category])} Flutter {category} rules")
                    except Exception as e:
                        logger.error(f"Error loading Flutter {category} rules from {filename}: {e}")
                        rules[category] = []
                else:
                    rules[category] = []
                    logger.warning(f"Flutter {category} rules file not found: {filename}")
            
            # Load default rules if no common rules found
            if not rules.get("common"):
                rules["common"] = self._create_default_rules()
                self._save_default_rules(rules_dir / rule_files["common"], rules["common"])
                    
        except Exception as e:
            logger.error(f"Error loading Flutter rules: {e}")
            rules = {"common": self._create_default_rules(), "widgets": [], "navigation": [], 
                    "state_management": [], "performance": [], "platform": []}
        
        return rules
    
    def _create_default_rules(self) -> List[Dict[str, Any]]:
        """Create default Flutter/Dart error rules."""
        return [
            {
                "id": "flutter_widget_build_error",
                "pattern": r"Error.*build.*widget|Widget.*build.*error",
                "category": "flutter",
                "subcategory": "widgets",
                "root_cause": "flutter_widget_build_error",
                "confidence": "high",
                "severity": "error",
                "suggestion": "Check widget build method for null values or incorrect widget structure",
                "tags": ["flutter", "widgets", "build"],
                "reliability": "high"
            },
            {
                "id": "flutter_render_overflow",
                "pattern": r"RenderFlex overflowed|overflow.*pixels",
                "category": "flutter",
                "subcategory": "layout",
                "root_cause": "flutter_render_overflow",
                "confidence": "high",
                "severity": "warning",
                "suggestion": "Use Expanded, Flexible, or adjust widget constraints to fix overflow",
                "tags": ["flutter", "layout", "overflow"],
                "reliability": "high"
            },
            {
                "id": "dart_null_check_error",
                "pattern": r"Null check operator.*null value|type 'Null'.*not.*subtype",
                "category": "dart",
                "subcategory": "null_safety",
                "root_cause": "dart_null_check_error",
                "confidence": "high",
                "severity": "error",
                "suggestion": "Add null checks or use null-aware operators (?., ??)",
                "tags": ["dart", "null-safety", "runtime"],
                "reliability": "high"
            },
            {
                "id": "flutter_setstate_error",
                "pattern": r"setState.*called.*disposed|setState.*unmounted",
                "category": "flutter",
                "subcategory": "state",
                "root_cause": "flutter_setstate_disposed",
                "confidence": "high",
                "severity": "error",
                "suggestion": "Check if widget is mounted before calling setState",
                "tags": ["flutter", "state", "lifecycle"],
                "reliability": "high"
            },
            {
                "id": "dart_cast_error",
                "pattern": r"type.*is not a subtype.*in type cast|Unable to cast",
                "category": "dart",
                "subcategory": "types",
                "root_cause": "dart_type_cast_error",
                "confidence": "high",
                "severity": "error",
                "suggestion": "Use safe casting with 'as?' or check types before casting",
                "tags": ["dart", "types", "casting"],
                "reliability": "high"
            },
            {
                "id": "flutter_async_error",
                "pattern": r"Future.*completed with an error|Unhandled.*exception.*async",
                "category": "flutter",
                "subcategory": "async",
                "root_cause": "flutter_async_error",
                "confidence": "medium",
                "severity": "error",
                "suggestion": "Add proper error handling with try-catch or .catchError()",
                "tags": ["flutter", "async", "futures"],
                "reliability": "medium"
            }
        ]
    
    def _save_default_rules(self, file_path: Path, rules: List[Dict[str, Any]]):
        """Save default rules to file."""
        try:
            with open(file_path, 'w') as f:
                json.dump({"rules": rules}, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving default Flutter rules: {e}")
    
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
                    logger.warning(f"Invalid regex pattern in Flutter rule {rule.get('id', 'unknown')}: {e}")
    
    def analyze_exception(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a Flutter/Dart exception and determine its type and potential fixes.
        
        Args:
            error_data: Flutter/Dart error data in standard format
            
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
                "category": best_match.get("category", "flutter"),
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
        
        # Boost confidence for Flutter/Dart-specific patterns
        message = error_data.get("message", "").lower()
        if "flutter" in message or "dart" in message or "widget" in message:
            base_confidence += 0.3
        
        # Boost confidence based on rule reliability
        reliability = rule.get("reliability", "medium")
        reliability_boost = {"high": 0.2, "medium": 0.1, "low": 0.0}
        base_confidence += reliability_boost.get(reliability, 0.0)
        
        # Boost confidence for rules with specific tags that match context
        rule_tags = set(rule.get("tags", []))
        context_tags = set()
        
        # Infer context from error data
        framework = error_data.get("framework", "").lower()
        if "flutter" in framework:
            context_tags.add("flutter")
        
        language = error_data.get("language", "").lower()
        if "dart" in language:
            context_tags.add("dart")
        
        # Check for widget-related context
        if error_data.get("widget_type"):
            context_tags.add("widgets")
        
        # Check runtime context
        context = error_data.get("context", {})
        if context.get("debug_mode"):
            context_tags.add("debug")
        
        if context_tags & rule_tags:
            base_confidence += 0.1
        
        return min(base_confidence, 1.0)
    
    def _generic_analysis(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """Provide generic analysis for unmatched errors."""
        error_type = error_data.get("error_type", "Error")
        message = error_data.get("message", "").lower()
        
        # Basic categorization based on error patterns
        if "widget" in message or "build" in message:
            category = "widgets"
            suggestion = "Check widget build method and widget tree structure"
        elif "state" in message or "setstate" in message:
            category = "state"
            suggestion = "Check state management and widget lifecycle"
        elif "null" in message:
            category = "null_safety"
            suggestion = "Add null checks and use null-safe operators"
        elif "overflow" in message or "layout" in message:
            category = "layout"
            suggestion = "Check widget constraints and layout configuration"
        elif "async" in message or "future" in message:
            category = "async"
            suggestion = "Add proper async error handling"
        elif "navigation" in message or "route" in message:
            category = "navigation"
            suggestion = "Check navigation configuration and route definitions"
        else:
            category = "unknown"
            suggestion = "Review Flutter/Dart implementation and check documentation"
        
        return {
            "category": "flutter",
            "subcategory": category,
            "confidence": "low",
            "suggested_fix": suggestion,
            "root_cause": f"flutter_{category}_error",
            "severity": "medium",
            "rule_id": "flutter_generic_handler",
            "tags": ["flutter", "generic", category]
        }
    
    def analyze_widget_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze Flutter widget-specific errors.
        
        Args:
            error_data: Error data with widget-related issues
            
        Returns:
            Analysis results with widget-specific fixes
        """
        message = error_data.get("message", "").lower()
        widget_type = error_data.get("widget_type", "")
        
        # Common widget error patterns
        if "overflow" in message:
            return {
                "category": "flutter",
                "subcategory": "layout",
                "confidence": "high",
                "suggested_fix": "Use Expanded, Flexible, or SingleChildScrollView to handle overflow",
                "root_cause": "flutter_widget_overflow",
                "severity": "warning",
                "tags": ["flutter", "widgets", "layout", "overflow"],
                "fix_commands": [
                    "Wrap widget with Expanded or Flexible",
                    "Use SingleChildScrollView for scrollable content",
                    "Check parent widget constraints",
                    "Consider using MediaQuery for responsive design"
                ]
            }
        
        if "build" in message and "error" in message:
            return {
                "category": "flutter",
                "subcategory": "widgets",
                "confidence": "high",
                "suggested_fix": "Check widget build method for null values or incorrect return types",
                "root_cause": "flutter_widget_build_error",
                "severity": "error",
                "tags": ["flutter", "widgets", "build"],
                "fix_commands": [
                    "Ensure build method returns a valid Widget",
                    "Check for null values in widget properties",
                    "Verify all required parameters are provided",
                    "Add null checks for dynamic data"
                ]
            }
        
        if "setstate" in message and ("disposed" in message or "unmounted" in message):
            return {
                "category": "flutter",
                "subcategory": "state",
                "confidence": "high",
                "suggested_fix": "Check if widget is mounted before calling setState",
                "root_cause": "flutter_setstate_disposed",
                "severity": "error",
                "tags": ["flutter", "state", "lifecycle"],
                "fix_commands": [
                    "Add 'if (mounted) setState(() { ... });'",
                    "Cancel async operations in dispose()",
                    "Use StatefulWidget lifecycle methods properly"
                ]
            }
        
        # Generic widget error
        return {
            "category": "flutter",
            "subcategory": "widgets",
            "confidence": "medium",
            "suggested_fix": f"Check {widget_type or 'widget'} implementation and properties",
            "root_cause": "flutter_widget_error",
            "severity": "medium",
            "tags": ["flutter", "widgets"]
        }
    
    def analyze_dart_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze Dart language-specific errors.
        
        Args:
            error_data: Error data with Dart language issues
            
        Returns:
            Analysis results with Dart-specific fixes
        """
        message = error_data.get("message", "").lower()
        
        # Dart null safety errors
        if "null check operator" in message and "null value" in message:
            return {
                "category": "dart",
                "subcategory": "null_safety",
                "confidence": "high",
                "suggested_fix": "Use null-aware operators (?., ??, !) or add null checks",
                "root_cause": "dart_null_check_error",
                "severity": "error",
                "tags": ["dart", "null-safety", "runtime"],
                "fix_commands": [
                    "Use safe navigation: object?.property",
                    "Use null coalescing: value ?? defaultValue", 
                    "Add null checks: if (value != null) { ... }",
                    "Use late keyword for non-null variables initialized later"
                ]
            }
        
        # Type casting errors
        if "type" in message and ("not a subtype" in message or "cast" in message):
            return {
                "category": "dart",
                "subcategory": "types",
                "confidence": "high",
                "suggested_fix": "Use safe casting with 'as?' or check types before casting",
                "root_cause": "dart_type_cast_error",
                "severity": "error",
                "tags": ["dart", "types", "casting"],
                "fix_commands": [
                    "Use safe cast: object as? TargetType",
                    "Check type first: if (object is TargetType) { ... }",
                    "Use is operator for type checking",
                    "Verify generic types match expected types"
                ]
            }
        
        # Async/Future errors
        if "future" in message and ("completed with an error" in message or "unhandled" in message):
            return {
                "category": "dart",
                "subcategory": "async",
                "confidence": "high",
                "suggested_fix": "Add proper error handling with try-catch or .catchError()",
                "root_cause": "dart_async_error",
                "severity": "error",
                "tags": ["dart", "async", "futures"],
                "fix_commands": [
                    "Wrap async code in try-catch blocks",
                    "Use .catchError() on Futures",
                    "Handle exceptions in async functions",
                    "Use await with proper error handling"
                ]
            }
        
        # Generic Dart error
        return {
            "category": "dart",
            "subcategory": "language",
            "confidence": "medium",
            "suggested_fix": "Check Dart language syntax and semantics",
            "root_cause": "dart_language_error",
            "severity": "medium",
            "tags": ["dart", "language"]
        }


class FlutterPatchGenerator:
    """
    Generates patches for Flutter/Dart errors based on analysis results.
    
    This class creates code fixes for common Flutter widget issues and Dart
    language errors using templates and heuristics.
    """
    
    def __init__(self):
        """Initialize the Flutter patch generator."""
        self.template_dir = Path(__file__).parent.parent / "patch_generation" / "templates"
        self.flutter_template_dir = self.template_dir / "flutter"
        
        # Ensure template directory exists
        self.flutter_template_dir.mkdir(parents=True, exist_ok=True)
        
        # Load patch templates
        self.templates = self._load_templates()
        
        # Create default templates if they don't exist
        self._create_default_templates()
    
    def _load_templates(self) -> Dict[str, str]:
        """Load Flutter patch templates."""
        templates = {}
        
        if not self.flutter_template_dir.exists():
            logger.warning(f"Flutter templates directory not found: {self.flutter_template_dir}")
            return templates
        
        for template_file in self.flutter_template_dir.glob("*.dart.template"):
            try:
                with open(template_file, 'r') as f:
                    template_name = template_file.stem.replace('.dart', '')
                    templates[template_name] = f.read()
                    logger.debug(f"Loaded Flutter template: {template_name}")
            except Exception as e:
                logger.error(f"Error loading Flutter template {template_file}: {e}")
        
        return templates
    
    def _create_default_templates(self):
        """Create default Flutter templates if they don't exist."""
        default_templates = {
            "null_safety_fix.dart.template": """
// Fix for null safety issues
// Use null-aware operators and null checks

// Instead of:
// String value = nullableValue!; // This can throw

// Use:
String? value = nullableValue; // Nullable type
String safeValue = nullableValue ?? 'default'; // Null coalescing
String? propertyValue = object?.property; // Safe navigation

// For late initialization:
late String lateValue; // Will be initialized later

// Null checks:
if (nullableValue != null) {
  // Safe to use nullableValue here
  print(nullableValue);
}
""",
            "widget_overflow_fix.dart.template": """
// Fix for widget overflow issues
import 'package:flutter/material.dart';

// Instead of:
// Row(children: [Widget1(), Widget2(), Widget3()]) // Can overflow

// Use Expanded or Flexible:
Row(
  children: [
    Expanded(child: Widget1()),
    Flexible(child: Widget2()),
    Widget3(),
  ],
)

// Or use SingleChildScrollView:
SingleChildScrollView(
  scrollDirection: Axis.horizontal,
  child: Row(
    children: [Widget1(), Widget2(), Widget3()],
  ),
)

// For Column overflow:
Expanded(
  child: SingleChildScrollView(
    child: Column(
      children: [...widgets],
    ),
  ),
)
""",
            "setstate_lifecycle_fix.dart.template": """
// Fix for setState after dispose errors
import 'package:flutter/material.dart';

class SafeStatefulWidget extends StatefulWidget {
  @override
  _SafeStatefulWidgetState createState() => _SafeStatefulWidgetState();
}

class _SafeStatefulWidgetState extends State<SafeStatefulWidget> {
  
  void safeSetState(VoidCallback fn) {
    if (mounted) {
      setState(fn);
    }
  }
  
  void performAsyncOperation() async {
    // Do async work
    await Future.delayed(Duration(seconds: 1));
    
    // Safe setState call
    safeSetState(() {
      // Update state here
    });
  }
  
  @override
  void dispose() {
    // Cancel any ongoing operations
    super.dispose();
  }
  
  @override
  Widget build(BuildContext context) {
    return Container();
  }
}
""",
            "async_error_handling.dart.template": """
// Fix for async error handling
import 'dart:async';

// Proper async error handling
Future<void> handleAsyncOperation() async {
  try {
    await riskyAsyncOperation();
  } catch (error) {
    print('Error occurred: $error');
    // Handle error appropriately
  }
}

// Using catchError with Futures
Future<String> fetchData() {
  return httpRequest()
    .then((response) => response.body)
    .catchError((error) {
      print('Network error: $error');
      return 'Default data';
    });
}

// Stream error handling
StreamSubscription? subscription;

void listenToStream() {
  subscription = dataStream.listen(
    (data) {
      // Handle data
    },
    onError: (error) {
      print('Stream error: $error');
    },
  );
}

@override
void dispose() {
  subscription?.cancel();
  super.dispose();
}
"""
        }
        
        for template_name, template_content in default_templates.items():
            template_path = self.flutter_template_dir / template_name
            if not template_path.exists():
                try:
                    with open(template_path, 'w') as f:
                        f.write(template_content)
                    logger.debug(f"Created default Flutter template: {template_name}")
                except Exception as e:
                    logger.error(f"Error creating default Flutter template {template_name}: {e}")
    
    def generate_patch(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                      source_code: str) -> Optional[Dict[str, Any]]:
        """
        Generate a patch for the Flutter/Dart error.
        
        Args:
            error_data: The Flutter/Dart error data
            analysis: Analysis results from FlutterExceptionHandler
            source_code: The source code where the error occurred
            
        Returns:
            Patch information or None if no patch can be generated
        """
        root_cause = analysis.get("root_cause", "")
        
        # Map root causes to patch strategies
        patch_strategies = {
            "dart_null_check_error": self._fix_null_safety,
            "flutter_widget_overflow": self._fix_widget_overflow,
            "flutter_setstate_disposed": self._fix_setstate_lifecycle,
            "dart_type_cast_error": self._fix_type_casting,
            "dart_async_error": self._fix_async_error,
            "flutter_widget_build_error": self._fix_widget_build,
            "flutter_render_overflow": self._fix_render_overflow
        }
        
        strategy = patch_strategies.get(root_cause)
        if strategy:
            try:
                return strategy(error_data, analysis, source_code)
            except Exception as e:
                logger.error(f"Error generating Flutter patch for {root_cause}: {e}")
        
        # Try to use templates if no specific strategy matches
        return self._template_based_patch(error_data, analysis, source_code)
    
    def _fix_null_safety(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                        source_code: str) -> Optional[Dict[str, Any]]:
        """Fix null safety violations."""
        return {
            "type": "suggestion",
            "description": "Add null safety checks and use null-aware operators",
            "fix_commands": [
                "Use null-aware operators: object?.property",
                "Use null coalescing: value ?? defaultValue",
                "Add null checks: if (value != null) { ... }",
                "Use late keyword for non-null variables initialized later"
            ],
            "template": "null_safety_fix",
            "code_example": """
// Safe null handling
String? nullableString = getValue();
String result = nullableString ?? 'default';
print(nullableString?.length); // Safe navigation
"""
        }
    
    def _fix_widget_overflow(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                           source_code: str) -> Optional[Dict[str, Any]]:
        """Fix widget overflow issues."""
        return {
            "type": "suggestion",
            "description": "Fix widget overflow using Expanded, Flexible, or scrollable widgets",
            "fix_commands": [
                "Wrap flexible widgets with Expanded or Flexible",
                "Use SingleChildScrollView for scrollable content",
                "Check parent widget constraints",
                "Use MediaQuery for responsive sizing"
            ],
            "template": "widget_overflow_fix",
            "code_example": """
// Fix overflow with Expanded
Row(
  children: [
    Expanded(child: Text('Long text that might overflow')),
    Icon(Icons.star),
  ],
)
"""
        }
    
    def _fix_setstate_lifecycle(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                               source_code: str) -> Optional[Dict[str, Any]]:
        """Fix setState after dispose errors."""
        return {
            "type": "suggestion",
            "description": "Check widget mount state before calling setState",
            "fix_commands": [
                "Add mounted check: if (mounted) setState(() { ... })",
                "Cancel async operations in dispose()",
                "Use proper StatefulWidget lifecycle"
            ],
            "template": "setstate_lifecycle_fix",
            "code_example": """
void updateState() {
  if (mounted) {
    setState(() {
      // Safe state update
    });
  }
}
"""
        }
    
    def _fix_type_casting(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                         source_code: str) -> Optional[Dict[str, Any]]:
        """Fix type casting errors."""
        return {
            "type": "suggestion",
            "description": "Use safe type casting and type checking",
            "fix_commands": [
                "Use safe cast: object as? TargetType",
                "Check type first: if (object is TargetType) { ... }",
                "Use is operator for type checking",
                "Verify generic types"
            ],
            "code_example": """
// Safe type checking and casting
if (value is String) {
  String stringValue = value; // Safe
}

// Safe casting
String? stringValue = value as String?;
"""
        }
    
    def _fix_async_error(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                        source_code: str) -> Optional[Dict[str, Any]]:
        """Fix async/await and Future errors."""
        return {
            "type": "suggestion",
            "description": "Add proper async error handling",
            "fix_commands": [
                "Wrap async code in try-catch blocks",
                "Use .catchError() on Futures",
                "Handle exceptions in async functions",
                "Cancel async operations in dispose"
            ],
            "template": "async_error_handling",
            "code_example": """
Future<void> fetchData() async {
  try {
    final result = await apiCall();
    // Handle success
  } catch (error) {
    // Handle error
    print('Error: $error');
  }
}
"""
        }
    
    def _fix_widget_build(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                         source_code: str) -> Optional[Dict[str, Any]]:
        """Fix widget build method errors."""
        return {
            "type": "suggestion",
            "description": "Fix widget build method issues",
            "fix_commands": [
                "Ensure build method returns a valid Widget",
                "Check for null values in widget properties",
                "Verify all required parameters are provided",
                "Add debugging with debugPrint or flutter inspector"
            ]
        }
    
    def _fix_render_overflow(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                           source_code: str) -> Optional[Dict[str, Any]]:
        """Fix render overflow (RenderFlex overflow)."""
        return {
            "type": "suggestion",
            "description": "Fix layout overflow issues",
            "fix_commands": [
                "Use Expanded or Flexible widgets",
                "Add SingleChildScrollView for scrollable content",
                "Check MainAxisSize and CrossAxisAlignment",
                "Consider using Wrap widget for flowing layout"
            ],
            "template": "widget_overflow_fix"
        }
    
    def _template_based_patch(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                            source_code: str) -> Optional[Dict[str, Any]]:
        """Generate patch using templates."""
        root_cause = analysis.get("root_cause", "")
        
        # Map root causes to template names
        template_map = {
            "dart_null_check_error": "null_safety_fix",
            "flutter_widget_overflow": "widget_overflow_fix",
            "flutter_setstate_disposed": "setstate_lifecycle_fix",
            "dart_async_error": "async_error_handling"
        }
        
        template_name = template_map.get(root_cause)
        if template_name and template_name in self.templates:
            template = self.templates[template_name]
            
            return {
                "type": "template",
                "template": template,
                "description": f"Applied Flutter/Dart template fix for {root_cause}"
            }
        
        return None


class FlutterLanguagePlugin(LanguagePlugin):
    """
    Main Flutter framework plugin for Homeostasis.
    
    This plugin orchestrates Flutter/Dart error analysis and patch generation,
    supporting Flutter mobile, web, and desktop applications.
    """
    
    VERSION = "1.0.0"
    AUTHOR = "Homeostasis Team"
    
    def __init__(self):
        """Initialize the Flutter language plugin."""
        self.language = "flutter"
        self.supported_extensions = {".dart"}
        self.supported_frameworks = [
            "flutter", "flutter_web", "flutter_desktop", 
            "flutter_mobile", "flet", "dart"
        ]
        
        # Initialize components
        self.adapter = DartErrorAdapter()
        self.exception_handler = FlutterExceptionHandler()
        self.patch_generator = FlutterPatchGenerator()
        
        logger.info("Flutter framework plugin initialized")
    
    def normalize_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize error data to the standard Homeostasis format."""
        return self.adapter.to_standard_format(error_data)
    
    def denormalize_error(self, standard_error: Dict[str, Any]) -> Dict[str, Any]:
        """Convert standard format error data back to the language-specific format."""
        # Convert back to Dart/Flutter format
        return {
            "type": standard_error.get("error_type", "Error"),
            "message": standard_error.get("message", ""),
            "stackTrace": standard_error.get("stack_trace", []),
            "widget": {
                "type": standard_error.get("widget_type"),
                "properties": standard_error.get("widget_properties", {})
            } if standard_error.get("widget_type") else None,
            "flutterVersion": standard_error.get("context", {}).get("flutter_version"),
            "dartVersion": standard_error.get("context", {}).get("dart_version"),
            "platform": standard_error.get("context", {}).get("platform"),
            "debugMode": standard_error.get("context", {}).get("debug_mode", True),
            "projectPath": standard_error.get("context", {}).get("project_path")
        }
    
    def get_language_id(self) -> str:
        """Get the unique identifier for this language."""
        return "flutter"
    
    def get_language_name(self) -> str:
        """Get the human-readable name of the framework."""
        return "Flutter"
    
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
        if "flutter" in framework or "dart" in framework:
            return True
        
        # Check language
        language = error_data.get("language", "").lower()
        if "dart" in language or "flutter" in language:
            return True
        
        # Check runtime environment
        runtime = error_data.get("runtime", "").lower()
        if any(indicator in runtime for indicator in ["flutter", "dart", "dartvm"]):
            return True
        
        # Check error message for Flutter/Dart-specific patterns
        message = error_data.get("message", "").lower()
        stack_trace = str(error_data.get("stack_trace", "")).lower()
        
        flutter_patterns = [
            r"flutter",
            r"dart",
            r"widget",
            r"renderflex",
            r"renderbox",
            r"build.*context",
            r"setstate",
            r"stateful.*widget",
            r"stateless.*widget",
            r"mounted",
            r"dispose",
            r"scaffold",
            r"material.*app",
            r"cupertino",
            r"\.dart:",
            r"dartvm",
            r"isolate.*dart"
        ]
        
        for pattern in flutter_patterns:
            if re.search(pattern, message + stack_trace):
                return True
        
        # Check file extensions in stack trace
        if re.search(r'\.dart:\d+', stack_trace):
            return True
        
        # Check project structure indicators
        context = error_data.get("context", {})
        project_files = context.get("project_files", [])
        
        # Look for Flutter project files
        flutter_project_indicators = [
            "pubspec.yaml",
            "analysis_options.yaml",
            "lib/main.dart",
            "android/app/build.gradle",
            "ios/Runner.xcodeproj",
            "web/index.html"
        ]
        
        project_files_str = " ".join(project_files).lower()
        if any(indicator in project_files_str for indicator in flutter_project_indicators):
            # Check dependencies for Flutter
            dependencies = context.get("dependencies", [])
            if any("flutter" in dep.lower() or "dart" in dep.lower() for dep in dependencies):
                return True
        
        return False
    
    def analyze_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a Flutter/Dart error.
        
        Args:
            error_data: Flutter/Dart error data
            
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
            
            # Check if it's a widget-related error
            if self._is_widget_error(standard_error):
                analysis = self.exception_handler.analyze_widget_error(standard_error)
            
            # Check if it's a Dart language error
            elif self._is_dart_error(standard_error):
                analysis = self.exception_handler.analyze_dart_error(standard_error)
            
            # Default Flutter error analysis
            else:
                analysis = self.exception_handler.analyze_exception(standard_error)
            
            # Add plugin metadata
            analysis["plugin"] = "flutter"
            analysis["language"] = "flutter"
            analysis["plugin_version"] = self.VERSION
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing Flutter error: {e}")
            return {
                "category": "flutter",
                "subcategory": "unknown",
                "confidence": "low",
                "suggested_fix": "Unable to analyze Flutter error",
                "error": str(e),
                "plugin": "flutter"
            }
    
    def _is_widget_error(self, error_data: Dict[str, Any]) -> bool:
        """Check if this is a Flutter widget related error."""
        message = error_data.get("message", "").lower()
        
        widget_patterns = [
            "widget",
            "build",
            "renderflex",
            "renderbox",
            "overflow",
            "setstate",
            "disposed",
            "mounted",
            "scaffold",
            "material",
            "cupertino"
        ]
        
        return any(pattern in message for pattern in widget_patterns)
    
    def _is_dart_error(self, error_data: Dict[str, Any]) -> bool:
        """Check if this is a Dart language related error."""
        message = error_data.get("message", "").lower()
        error_type = error_data.get("error_type", "").lower()
        
        dart_patterns = [
            "null check operator",
            "type.*not.*subtype",
            "cast",
            "future.*completed.*error",
            "async",
            "isolate",
            "dart:",
            "_TypeError",
            "_CastError"
        ]
        
        return any(pattern in message or pattern in error_type for pattern in dart_patterns)
    
    def generate_fix(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a fix for an error based on the analysis.
        
        Args:
            analysis: Error analysis
            context: Additional context for fix generation
            
        Returns:
            Generated fix data
        """
        try:
            # Extract error data and source code from context if available
            error_data = analysis.get("error_data", {})
            source_code = context.get("source_code", "")
            
            # Generate patch
            patch_result = self.patch_generator.generate_patch(error_data, analysis, source_code)
            
            if patch_result:
                return patch_result
            
            # Return empty dict if no patch generated (as per abstract method)
            return {}
        except Exception as e:
            logger.error(f"Error generating Flutter fix: {e}")
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
                "Flutter widget error detection and fixes",
                "Dart null safety violation handling",
                "Widget overflow and layout issue resolution",
                "State management and lifecycle error handling",
                "Async/await and Future error management",
                "Type casting and type checking fixes",
                "Performance and rendering issue detection",
                "Navigation and routing error handling",
                "Platform channel and plugin error resolution",
                "Animation and transition error fixes",
                "Flutter testing error support"
            ],
            "platforms": ["mobile", "web", "desktop", "embedded"],
            "environments": ["flutter", "dart", "dartvm"]
        }


# Register the plugin
register_plugin(FlutterLanguagePlugin())