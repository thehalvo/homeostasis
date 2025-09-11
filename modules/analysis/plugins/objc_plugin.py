"""
Objective-C Plugin for Homeostasis

This plugin enables Homeostasis to analyze and fix errors in Objective-C applications.
It provides comprehensive error handling for iOS/macOS development including memory management,
runtime issues, UIKit/AppKit errors, and framework-specific problems.
"""

import json
import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..language_plugin_system import LanguagePlugin, register_plugin

logger = logging.getLogger(__name__)


class ObjCExceptionHandler:
    """
    Handles Objective-C-specific exceptions with comprehensive error detection and classification.

    This class provides logic for categorizing Objective-C compilation errors, runtime issues,
    memory management problems, and iOS/macOS framework-specific challenges.
    """

    def __init__(self):
        """Initialize the Objective-C exception handler."""
        self.rule_categories = {
            "compilation": "Objective-C compilation errors and syntax issues",
            "runtime": "Runtime exceptions and crashes",
            "memory": "Memory management and ARC-related errors",
            "uikit": "UIKit framework errors for iOS development",
            "appkit": "AppKit framework errors for macOS development",
            "foundation": "Foundation framework errors",
            "core_data": "Core Data persistence framework errors",
            "networking": "NSURLSession and networking errors",
            "threading": "Grand Central Dispatch and threading errors",
            "xcode": "Xcode build system and configuration errors",
            "interface_builder": "Interface Builder and Storyboard errors",
            "app_store": "App Store submission and validation errors",
            "permissions": "iOS/macOS permissions and entitlements errors",
            "cocoapods": "CocoaPods dependency management errors",
            "swift_interop": "Swift-Objective-C interoperability errors",
            "categories": "Objective-C categories and extensions errors",
            "protocols": "Protocol implementation and delegation errors",
            "blocks": "Blocks and closure-related errors",
            "kvo": "Key-Value Observing implementation errors",
        }

        # Load rules from different categories
        self.rules = self._load_rules()

        # Pre-compile regex patterns for better performance
        self._compile_patterns()

    def _load_rules(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load Objective-C error rules from rule files."""
        rules = {}
        rules_dir = Path(__file__).parent.parent / "rules" / "objc"

        try:
            # Create rules directory if it doesn't exist
            rules_dir.mkdir(parents=True, exist_ok=True)

            # Load common Objective-C rules
            common_rules_path = rules_dir / "objc_common_errors.json"
            if common_rules_path.exists():
                with open(common_rules_path, "r") as f:
                    common_data = json.load(f)
                    rules["common"] = common_data.get("rules", [])
                    logger.info(
                        f"Loaded {len(rules['common'])} common Objective-C rules"
                    )
            else:
                rules["common"] = self._create_default_rules()
                self._save_default_rules(common_rules_path, rules["common"])

            # Load iOS-specific rules
            ios_rules_path = rules_dir / "objc_ios_errors.json"
            if ios_rules_path.exists():
                with open(ios_rules_path, "r") as f:
                    ios_data = json.load(f)
                    rules["ios"] = ios_data.get("rules", [])
                    logger.info(f"Loaded {len(rules['ios'])} Objective-C iOS rules")
            else:
                rules["ios"] = []

            # Load memory-specific rules
            memory_rules_path = rules_dir / "objc_memory_errors.json"
            if memory_rules_path.exists():
                with open(memory_rules_path, "r") as f:
                    memory_data = json.load(f)
                    rules["memory"] = memory_data.get("rules", [])
                    logger.info(
                        f"Loaded {len(rules['memory'])} Objective-C memory rules"
                    )
            else:
                rules["memory"] = []

        except Exception as e:
            logger.error(f"Error loading Objective-C rules: {e}")
            rules = {"common": self._create_default_rules(), "ios": [], "memory": []}

        return rules

    def _create_default_rules(self) -> List[Dict[str, Any]]:
        """Create default Objective-C error detection rules."""
        return [
            {
                "id": "objc_exc_bad_access",
                "pattern": r"EXC_BAD_ACCESS|KERN_INVALID_ADDRESS|signal SIGABRT",
                "category": "memory",
                "severity": "critical",
                "description": "Bad memory access - likely null pointer or deallocated object",
                "fix_suggestions": [
                    "Check for nil object access",
                    "Verify object lifetime and ARC management",
                    "Use weak references to break retain cycles",
                    "Enable zombie objects for debugging: NSZombieEnabled",
                ],
            },
            {
                "id": "objc_unrecognized_selector",
                "pattern": r"unrecognized selector sent to|does not respond to selector",
                "category": "runtime",
                "severity": "high",
                "description": "Method not found on object",
                "fix_suggestions": [
                    "Check method name spelling and case",
                    "Verify object type is correct",
                    "Ensure method is declared in header file",
                    "Check if method exists in the target class",
                ],
            },
            {
                "id": "objc_property_not_found",
                "pattern": r"property '.*' not found on object|undeclared identifier",
                "category": "compilation",
                "severity": "high",
                "description": "Property or identifier not found",
                "fix_suggestions": [
                    "Check property declaration in header file",
                    "Verify correct import statements",
                    "Ensure property name is spelled correctly",
                    "Check if property should be declared as IBOutlet",
                ],
            },
            {
                "id": "objc_arc_error",
                "pattern": r"ARC forbids|autoreleasing object|unsafe_unretained",
                "category": "memory",
                "severity": "high",
                "description": "Automatic Reference Counting (ARC) error",
                "fix_suggestions": [
                    "Use proper ARC annotations (__strong, __weak, __unsafe_unretained)",
                    "Avoid manual retain/release calls in ARC code",
                    "Use weak references for delegates and parent objects",
                    "Check for retain cycles in block captures",
                ],
            },
            {
                "id": "objc_nib_loading_error",
                "pattern": r"Could not load NIB|nib but didn't get a UIView|NSNib",
                "category": "interface_builder",
                "severity": "medium",
                "description": "Interface Builder NIB/Storyboard loading error",
                "fix_suggestions": [
                    "Check NIB file exists in bundle",
                    "Verify correct class name in Interface Builder",
                    "Ensure outlets are properly connected",
                    "Check for circular references in view hierarchy",
                ],
            },
            {
                "id": "objc_kvo_error",
                "pattern": r"NSKeyValueObserving|KVO|addObserver.*forKeyPath",
                "category": "kvo",
                "severity": "medium",
                "description": "Key-Value Observing implementation error",
                "fix_suggestions": [
                    "Ensure observer is removed before deallocation",
                    "Check key path is valid and observable",
                    "Implement proper KVO change notification",
                    "Use NSKeyValueObservingOptionNew/Old for context",
                ],
            },
            {
                "id": "objc_threading_error",
                "pattern": r"UIKit is only safe to use from main thread|NSThread|dispatch_",
                "category": "threading",
                "severity": "high",
                "description": "Threading and UIKit main thread violation",
                "fix_suggestions": [
                    "Perform UI updates on main thread using dispatch_async",
                    "Use NSOperationQueue for background tasks",
                    "Implement proper thread synchronization",
                    "Check UIKit usage is limited to main thread",
                ],
            },
            {
                "id": "objc_cocoapods_error",
                "pattern": r"CocoaPods|pod install|Podfile|workspace",
                "category": "cocoapods",
                "severity": "medium",
                "description": "CocoaPods dependency management error",
                "fix_suggestions": [
                    "Run 'pod install' to update dependencies",
                    "Check Podfile syntax and version constraints",
                    "Clean and rebuild project after pod changes",
                    "Ensure proper workspace usage instead of project file",
                ],
            },
        ]

    def _save_default_rules(self, rules_path: Path, rules: List[Dict[str, Any]]):
        """Save default rules to JSON file."""
        try:
            rules_data = {
                "version": "1.0",
                "description": "Default Objective-C error detection rules",
                "rules": rules,
            }
            with open(rules_path, "w") as f:
                json.dump(rules_data, f, indent=2)
            logger.info(f"Saved {len(rules)} default Objective-C rules to {rules_path}")
        except Exception as e:
            logger.error(f"Error saving default Objective-C rules: {e}")

    def _compile_patterns(self):
        """Pre-compile regex patterns for better performance."""
        self.compiled_patterns = {}
        for category, rule_list in self.rules.items():
            self.compiled_patterns[category] = []
            for rule in rule_list:
                try:
                    compiled_pattern = re.compile(
                        rule["pattern"], re.IGNORECASE | re.MULTILINE
                    )
                    self.compiled_patterns[category].append((compiled_pattern, rule))
                except re.error as e:
                    logger.warning(
                        f"Invalid regex pattern in rule {rule.get('id', 'unknown')}: {e}"
                    )

    def analyze_error(
        self, error_message: str, context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Analyze an Objective-C error message and provide categorization and suggestions.

        Args:
            error_message: The error message to analyze
            context: Additional context information

        Returns:
            Analysis results with error type, category, and suggestions
        """
        if context is None:
            context = {}

        results = {
            "language": "objc",
            "error_message": error_message,
            "matches": [],
            "primary_category": "unknown",
            "severity": "medium",
            "fix_suggestions": [],
            "platform_info": context.get("platform_info", {}),
            "xcode_version": context.get("xcode_version", "unknown"),
            "ios_version": context.get("ios_version", "unknown"),
            "additional_context": {},
        }

        # Check each category of rules
        for category, pattern_list in self.compiled_patterns.items():
            for compiled_pattern, rule in pattern_list:
                match = compiled_pattern.search(error_message)
                if match:
                    match_info = {
                        "rule_id": rule["id"],
                        "category": rule["category"],
                        "severity": rule["severity"],
                        "description": rule["description"],
                        "fix_suggestions": rule["fix_suggestions"],
                        "matched_text": match.group(0),
                        "match_groups": match.groups(),
                    }
                    results["matches"].append(match_info)

        # Determine primary category and severity
        if results["matches"]:
            # Sort by severity priority
            severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
            sorted_matches = sorted(
                results["matches"], key=lambda x: severity_order.get(x["severity"], 4)
            )

            primary_match = sorted_matches[0]
            results["primary_category"] = primary_match["category"]
            results["severity"] = primary_match["severity"]

            # Collect all fix suggestions
            all_suggestions = []
            for match in results["matches"]:
                all_suggestions.extend(match["fix_suggestions"])
            results["fix_suggestions"] = list(set(all_suggestions))  # Remove duplicates

        # Add platform-specific analysis
        results["additional_context"] = self._analyze_platform_context(
            error_message, context
        )

        return results

    def _analyze_platform_context(
        self, error_message: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze platform-specific context and provide additional insights."""
        additional_context = {
            "detected_platform": "unknown",
            "framework_detected": [],
            "arc_related": False,
            "swift_interop": False,
            "ui_related": False,
        }

        # Detect platform
        if re.search(r"iOS|UIKit|iPhone|iPad", error_message, re.IGNORECASE):
            additional_context["detected_platform"] = "iOS"
        elif re.search(r"macOS|AppKit|NSApplication", error_message, re.IGNORECASE):
            additional_context["detected_platform"] = "macOS"
        elif re.search(r"watchOS|WatchKit", error_message, re.IGNORECASE):
            additional_context["detected_platform"] = "watchOS"
        elif re.search(r"tvOS|TVUIKit", error_message, re.IGNORECASE):
            additional_context["detected_platform"] = "tvOS"

        # Detect frameworks
        frameworks = [
            ("UIKit", r"UIKit|UI[A-Z]"),
            ("Foundation", r"Foundation|NS[A-Z]"),
            ("Core Data", r"Core ?Data|NSManagedObject|NSPersistentStore"),
            ("Core Graphics", r"Core ?Graphics|CG[A-Z]|Quartz"),
            ("AVFoundation", r"AVFoundation|AVPlayer|AVAudio"),
            ("MapKit", r"MapKit|MKMap"),
            ("Core Location", r"Core ?Location|CLLocation"),
            ("Photos", r"Photos|PHAsset"),
            ("CloudKit", r"CloudKit|CKRecord"),
        ]

        for framework_name, pattern in frameworks:
            if re.search(pattern, error_message, re.IGNORECASE):
                additional_context["framework_detected"].append(framework_name)

        # Check for ARC-related issues
        if re.search(
            r"ARC|retain|release|autorelease|__weak|__strong",
            error_message,
            re.IGNORECASE,
        ):
            additional_context["arc_related"] = True

        # Check for Swift interop
        if re.search(r"@objc|swift|bridging", error_message, re.IGNORECASE):
            additional_context["swift_interop"] = True

        # Check for UI-related issues
        if re.search(
            r"UI|view|controller|storyboard|segue|outlet", error_message, re.IGNORECASE
        ):
            additional_context["ui_related"] = True

        return additional_context


class ObjCPatchGenerator:
    """
    Generates patches and fixes for Objective-C code issues.

    This class provides automated patch generation for common Objective-C errors,
    memory management issues, and iOS/macOS framework problems.
    """

    def __init__(self):
        """Initialize the Objective-C patch generator."""
        self.patch_templates = self._load_patch_templates()

    def _load_patch_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load patch templates for different types of Objective-C errors."""
        return {
            "memory_management": {
                "description": "Fix ARC and memory management issues",
                "template": """
// Original problematic code:
// {original_code}

// Fixed code with proper ARC management:
{fixed_code}

// ARC Guidelines:
// 1. Use __weak for delegates and parent references
// 2. Use __strong for child objects (default)
// 3. Avoid retain cycles in blocks with weakSelf pattern
// 4. Don't mix ARC with manual retain/release
""",
            },
            "nil_check": {
                "description": "Add nil checks before method calls",
                "template": """
// Add nil check before method invocation:
if ({object_name} != nil) {
    // Original code here
    {original_code}
} else {
    // Handle nil case appropriately
    NSLog(@"Warning: {object_name} is nil");
    // Return early or provide default behavior
}
""",
            },
            "selector_fix": {
                "description": "Fix unrecognized selector errors",
                "template": """
// Check if object responds to selector before calling:
if ([{object_name} respondsToSelector:@selector({method_name})]) {
    // Original method call
    {original_code}
} else {
    NSLog(@"Warning: %@ does not respond to %@", {object_name}, NSStringFromSelector(@selector({method_name})));
    // Handle missing method case
}
""",
            },
            "threading_fix": {
                "description": "Fix main thread violations",
                "template": """
// Ensure UI updates happen on main thread:
dispatch_async(dispatch_get_main_queue(), ^{
    // UI update code here
    {ui_code}
});

// For background work, use global queue:
dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
    // Background work here
    {background_code}
    
    // Return to main thread for UI updates
    dispatch_async(dispatch_get_main_queue(), ^{
        // Update UI with results
        {ui_update_code}
    });
});
""",
            },
            "kvo_fix": {
                "description": "Fix KVO implementation",
                "template": """
// Proper KVO implementation:

// In viewDidLoad or initialization:
[self addObserver:self
       forKeyPath:@"{key_path}"
          options:NSKeyValueObservingOptionNew | NSKeyValueObservingOptionOld
          context:NULL];

// Implement observer method:
- (void)observeValueForKeyPath:(NSString *)keyPath 
                      ofObject:(id)object 
                        change:(NSDictionary<NSKeyValueChangeKey,id> *)change 
                       context:(void *)context {
    if ([keyPath isEqualToString:@"{key_path}"]) {
        // Handle change
        id newValue = change[NSKeyValueChangeNewKey];
        id oldValue = change[NSKeyValueChangeOldKey];
        
        // Update UI or perform logic
    } else {
        [super observeValueForKeyPath:keyPath ofObject:object change:change context:context];
    }
}

// In dealloc:
- (void)dealloc {
    [self removeObserver:self forKeyPath:@"{key_path}"];
}
""",
            },
            "outlet_connection": {
                "description": "Fix Interface Builder outlet connections",
                "template": """
// Ensure proper outlet connection:

// 1. In header file (.h):
@property (nonatomic, weak) IBOutlet UILabel *{outlet_name};

// 2. In implementation file (.m):
- (void)viewDidLoad {
    [super viewDidLoad];
    
    // Check if outlet is connected
    if (self.{outlet_name} == nil) {
        NSLog(@"Error: {outlet_name} outlet is not connected in Interface Builder");
        // Handle missing outlet gracefully
        return;
    }
    
    // Configure outlet
    self.{outlet_name}.text = @"Default Text";
}

// 3. Connect outlet in Interface Builder:
// - Control+drag from File's Owner to the UI element
// - Select the outlet name from the popup
""",
            },
        }

    def generate_patch(
        self, error_analysis: Dict[str, Any], source_code: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a patch for the identified Objective-C error.

        Args:
            error_analysis: Analysis results from ObjCExceptionHandler
            source_code: Optional source code context

        Returns:
            Generated patch information
        """
        patch_info = {
            "patch_type": "unknown",
            "confidence": 0.0,
            "patch_content": "",
            "explanation": "",
            "additional_steps": [],
            "risks": [],
        }

        if not error_analysis.get("matches"):
            return patch_info

        primary_match = error_analysis["matches"][0]
        category = primary_match["category"]

        # Generate patch based on error category
        if category == "memory":
            patch_info = self._generate_memory_patch(error_analysis, source_code)
        elif category == "runtime":
            patch_info = self._generate_runtime_patch(error_analysis, source_code)
        elif category == "threading":
            patch_info = self._generate_threading_patch(error_analysis, source_code)
        elif category == "interface_builder":
            patch_info = self._generate_ib_patch(error_analysis, source_code)
        elif category == "kvo":
            patch_info = self._generate_kvo_patch(error_analysis, source_code)
        else:
            patch_info = self._generate_generic_patch(error_analysis, source_code)

        return patch_info

    def _generate_memory_patch(
        self, error_analysis: Dict[str, Any], source_code: Optional[str]
    ) -> Dict[str, Any]:
        """Generate patch for memory management errors."""
        template = self.patch_templates["memory_management"]

        return {
            "patch_type": "memory_management",
            "confidence": 0.8,
            "patch_content": template["template"],
            "explanation": template["description"],
            "additional_steps": [
                "Enable NSZombieEnabled for debugging",
                "Use Instruments to analyze memory usage",
                "Review all delegate assignments for weak references",
                "Check block capture lists for retain cycles",
            ],
            "risks": [
                "Changing memory management may affect object lifetimes",
                "Ensure all code paths handle weak references properly",
            ],
        }

    def _generate_runtime_patch(
        self, error_analysis: Dict[str, Any], source_code: Optional[str]
    ) -> Dict[str, Any]:
        """Generate patch for runtime errors."""
        if "unrecognized selector" in error_analysis["error_message"]:
            template = self.patch_templates["selector_fix"]
        else:
            template = self.patch_templates["nil_check"]

        return {
            "patch_type": "runtime_safety",
            "confidence": 0.7,
            "patch_content": template["template"],
            "explanation": template["description"],
            "additional_steps": [
                "Add runtime checks for object types",
                "Implement proper error handling",
                "Use respondsToSelector: before method calls",
                "Validate input parameters",
            ],
            "risks": [
                "Additional checks may impact performance",
                "Ensure error handling doesn't mask underlying issues",
            ],
        }

    def _generate_threading_patch(
        self, error_analysis: Dict[str, Any], source_code: Optional[str]
    ) -> Dict[str, Any]:
        """Generate patch for threading errors."""
        template = self.patch_templates["threading_fix"]

        return {
            "patch_type": "threading_fix",
            "confidence": 0.8,
            "patch_content": template["template"],
            "explanation": template["description"],
            "additional_steps": [
                "Audit all UI updates for main thread usage",
                "Use NSOperationQueue for complex background tasks",
                "Implement proper synchronization for shared data",
                "Test thoroughly on different devices and iOS versions",
            ],
            "risks": [
                "Threading changes may affect app performance",
                "Deadlocks possible with improper synchronization",
            ],
        }

    def _generate_ib_patch(
        self, error_analysis: Dict[str, Any], source_code: Optional[str]
    ) -> Dict[str, Any]:
        """Generate patch for Interface Builder errors."""
        template = self.patch_templates["outlet_connection"]

        return {
            "patch_type": "interface_builder_fix",
            "confidence": 0.6,
            "patch_content": template["template"],
            "explanation": template["description"],
            "additional_steps": [
                "Verify all outlets are connected in Interface Builder",
                "Check for orphaned connections",
                "Ensure correct class names in IB",
                "Validate Storyboard file integrity",
            ],
            "risks": ["Manual IB fixes required", "May need to recreate connections"],
        }

    def _generate_kvo_patch(
        self, error_analysis: Dict[str, Any], source_code: Optional[str]
    ) -> Dict[str, Any]:
        """Generate patch for KVO errors."""
        template = self.patch_templates["kvo_fix"]

        return {
            "patch_type": "kvo_fix",
            "confidence": 0.7,
            "patch_content": template["template"],
            "explanation": template["description"],
            "additional_steps": [
                "Ensure observers are removed in dealloc",
                "Use proper KVO context for disambiguation",
                "Validate key paths at compile time if possible",
                "Consider using modern observation patterns",
            ],
            "risks": [
                "KVO can impact performance if overused",
                "Improper removal can cause crashes",
            ],
        }

    def _generate_generic_patch(
        self, error_analysis: Dict[str, Any], source_code: Optional[str]
    ) -> Dict[str, Any]:
        """Generate generic patch for other error types."""
        return {
            "patch_type": "generic_fix",
            "confidence": 0.3,
            "patch_content": "// Generic fix based on error analysis",
            "explanation": "Address the identified issue",
            "additional_steps": error_analysis.get("fix_suggestions", []),
            "risks": [
                "Manual review required for generic fixes",
                "Test thoroughly on target platforms",
            ],
        }


class ObjCLanguagePlugin(LanguagePlugin):
    """
    Objective-C Language Plugin for Homeostasis.

    This plugin provides comprehensive Objective-C error analysis and fixing capabilities
    for iOS/macOS development in the Homeostasis self-healing software system.
    """

    def __init__(self):
        """Initialize the Objective-C language plugin."""
        super().__init__()
        self.name = "objc_plugin"
        self.version = "1.0.0"
        self.description = (
            "Objective-C error analysis and fixing plugin for iOS/macOS development"
        )
        self.supported_languages = ["objc", "objective-c", "objectivec"]
        self.supported_extensions = [".m", ".mm", ".h"]

        # Initialize components
        self.exception_handler = ObjCExceptionHandler()
        self.patch_generator = ObjCPatchGenerator()

        logger.info(f"Initialized {self.name} v{self.version}")

    def can_handle(self, language: str, file_path: Optional[str] = None) -> bool:
        """
        Check if this plugin can handle the given language or file.

        Args:
            language: Programming language identifier
            file_path: Optional path to the source file

        Returns:
            True if this plugin can handle the language/file, False otherwise
        """
        # Check language
        if language.lower() in self.supported_languages:
            return True

        # Check file extension
        if file_path:
            file_path_obj = Path(file_path)
            if file_path_obj.suffix.lower() in self.supported_extensions:
                return True

        return False

    def analyze_error(
        self, error_message: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze an Objective-C error and provide comprehensive information.

        Args:
            error_message: The error message to analyze
            context: Optional context information

        Returns:
            Comprehensive error analysis results
        """
        if context is None:
            context = {}

        # Use the exception handler to analyze the error
        analysis = self.exception_handler.analyze_error(error_message, context)

        # Add plugin metadata
        analysis.update(
            {
                "plugin_name": self.name,
                "plugin_version": self.version,
                "analysis_timestamp": self._get_timestamp(),
                "confidence_score": self._calculate_confidence(analysis),
            }
        )

        return analysis

    def generate_fix(
        self, error_analysis: Dict[str, Any], source_code: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a fix for the analyzed Objective-C error.

        Args:
            error_analysis: Results from analyze_error
            source_code: Optional source code context

        Returns:
            Generated fix information
        """
        # Use the patch generator to create a fix
        patch_info = self.patch_generator.generate_patch(error_analysis, source_code)

        # Add plugin metadata
        patch_info.update(
            {
                "plugin_name": self.name,
                "plugin_version": self.version,
                "generation_timestamp": self._get_timestamp(),
                "error_analysis": error_analysis,
            }
        )

        return patch_info

    def test_fix(
        self,
        original_code: str,
        fixed_code: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Test a generated fix by attempting compilation.

        Args:
            original_code: The original problematic code
            fixed_code: The proposed fixed code
            context: Optional context information

        Returns:
            Test results
        """
        if context is None:
            context = {}

        test_results = {
            "success": False,
            "compilation_successful": False,
            "errors": [],
            "warnings": [],
            "execution_test": None,
        }

        try:
            # Test compilation of fixed code
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".m", delete=False
            ) as temp_file:
                temp_file.write(fixed_code)
                temp_file_path = temp_file.name

            # Attempt compilation with clang
            compile_command = [
                "clang",
                "-c",
                temp_file_path,
                "-o",
                "/dev/null",
                "-fobjc-arc",  # Enable ARC
                "-framework",
                "Foundation",
            ]

            # Add iOS/macOS specific flags based on context
            platform = context.get("platform", "ios")
            if platform == "ios":
                compile_command.extend(
                    [
                        "-isysroot",
                        "/Applications/Xcode.app/Contents/Developer/Platforms/iPhoneSimulator.platform/Developer/SDKs/iPhoneSimulator.sdk",
                    ]
                )

            result = subprocess.run(
                compile_command, capture_output=True, text=True, timeout=30
            )

            if result.returncode == 0:
                test_results["compilation_successful"] = True
                test_results["success"] = True
            else:
                test_results["errors"] = result.stderr.split("\n")

            # Clean up
            os.unlink(temp_file_path)

        except subprocess.TimeoutExpired:
            test_results["errors"] = ["Compilation timeout"]
        except Exception as e:
            test_results["errors"] = [f"Test execution error: {str(e)}"]

        return test_results

    def get_recommendations(self, error_analysis: Dict[str, Any]) -> List[str]:
        """
        Get recommendations for preventing similar Objective-C errors.

        Args:
            error_analysis: Results from analyze_error

        Returns:
            List of prevention recommendations
        """
        recommendations = []

        if error_analysis.get("primary_category") == "memory":
            recommendations.extend(
                [
                    "Always use ARC (Automatic Reference Counting) for new projects",
                    "Use weak references for delegates and parent objects",
                    "Implement proper dealloc methods to clean up resources",
                    "Use NSZombieEnabled during development for debugging",
                    "Analyze memory usage with Instruments",
                ]
            )
        elif error_analysis.get("primary_category") == "runtime":
            recommendations.extend(
                [
                    "Check for nil before calling methods on objects",
                    "Use respondsToSelector: before calling optional methods",
                    "Implement proper error handling and validation",
                    "Use defensive programming practices",
                    "Add runtime type checking where appropriate",
                ]
            )
        elif error_analysis.get("primary_category") == "threading":
            recommendations.extend(
                [
                    "Always update UI on the main thread",
                    "Use Grand Central Dispatch for background processing",
                    "Implement proper synchronization for shared resources",
                    "Avoid blocking the main thread with long operations",
                    "Use NSOperationQueue for complex task management",
                ]
            )

        # Platform-specific recommendations
        if (
            error_analysis.get("additional_context", {}).get("detected_platform") ==
            "iOS"
        ):
            recommendations.extend(
                [
                    "Follow iOS Human Interface Guidelines",
                    "Test on multiple device sizes and orientations",
                    "Handle app lifecycle events properly",
                    "Implement proper memory warnings handling",
                    "Use Auto Layout for responsive design",
                ]
            )

        # General Objective-C recommendations
        recommendations.extend(
            [
                "Use modern Objective-C syntax and features",
                "Follow Apple's coding conventions and style guides",
                "Implement comprehensive unit and integration tests",
                "Use static analysis tools (Clang Static Analyzer)",
                "Keep up with iOS/macOS version deprecations",
                "Document public APIs with proper headers",
                "Use version control and code review processes",
            ]
        )

        return recommendations

    def _calculate_confidence(self, analysis: Dict[str, Any]) -> float:
        """Calculate confidence score for the analysis."""
        if not analysis.get("matches"):
            return 0.0

        # Base confidence on number and quality of matches
        match_count = len(analysis["matches"])
        severity_weight = {"critical": 1.0, "high": 0.8, "medium": 0.6, "low": 0.4}

        total_weight = sum(
            severity_weight.get(match["severity"], 0.2) for match in analysis["matches"]
        )
        confidence = min(total_weight / match_count, 1.0) if match_count > 0 else 0.0

        return round(confidence, 2)

    def _get_timestamp(self) -> str:
        """Get current timestamp string."""
        import datetime

        return datetime.datetime.now().isoformat()


# Register the plugin
@register_plugin
def create_objc_plugin():
    """Factory function to create Objective-C plugin instance."""
    return ObjCLanguagePlugin()


# Export the plugin class for direct usage
__all__ = ["ObjCLanguagePlugin", "ObjCExceptionHandler", "ObjCPatchGenerator"]
