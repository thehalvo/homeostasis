"""
Java Android Plugin for Homeostasis

This plugin enables Homeostasis to analyze and fix errors in Android Java applications.
It provides comprehensive error handling for Android-specific Java errors including
Activities, Services, ContentProviders, BroadcastReceivers, Fragments, Views, and SDK APIs.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..language_adapters import JavaErrorAdapter
from ..language_plugin_system import LanguagePlugin, register_plugin

logger = logging.getLogger(__name__)


class AndroidJavaExceptionHandler:
    """
    Handles Android Java-specific exceptions with comprehensive error detection and classification.

    This class provides logic for categorizing Android Java component errors, lifecycle issues,
    memory problems, permission issues, and platform-specific challenges.
    """

    def __init__(self):
        """Initialize the Android Java exception handler."""
        self.rule_categories = {
            "activity": "Activity lifecycle and component errors",
            "fragment": "Fragment lifecycle and transaction errors",
            "service": "Service lifecycle and background processing errors",
            "broadcast": "BroadcastReceiver and intent handling errors",
            "content_provider": "ContentProvider and data access errors",
            "view": "View hierarchy and UI component errors",
            "layout": "Layout inflation and constraint errors",
            "resources": "Resource access and configuration errors",
            "permissions": "Android permissions and security errors",
            "manifest": "Android manifest configuration errors",
            "lifecycle": "Android component lifecycle errors",
            "memory": "Memory management and OutOfMemory errors",
            "threading": "Android threading and AsyncTask errors",
            "database": "SQLite and Room database errors",
            "networking": "Network and connectivity errors",
            "intent": "Intent and component communication errors",
            "gradle": "Android build and Gradle configuration errors",
            "sdk": "Android SDK and API level compatibility errors",
            "proguard": "ProGuard and code obfuscation errors",
            "hardware": "Hardware access and sensor errors",
            "media": "Media playback and camera errors",
        }

        # Load rules from different categories
        self.rules = self._load_rules()

        # Pre-compile regex patterns for better performance
        self._compile_patterns()

    def _load_rules(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load Android Java error rules from rule files."""
        rules = {}
        rules_dir = Path(__file__).parent.parent / "rules" / "java_android"
        java_rules_dir = Path(__file__).parent.parent / "rules" / "java"

        try:
            # Create rules directory if it doesn't exist
            rules_dir.mkdir(parents=True, exist_ok=True)

            # Load Android rules from java directory first
            android_rules_path = java_rules_dir / "android_errors.json"
            if android_rules_path.exists():
                with open(android_rules_path, "r") as f:
                    android_data = json.load(f)
                    rules["android"] = android_data.get("rules", [])
                    logger.info(
                        f"Loaded {len(rules['android'])} Android rules from java directory"
                    )

            # Load common Android Java rules
            common_rules_path = rules_dir / "java_android_common_errors.json"
            if common_rules_path.exists():
                with open(common_rules_path, "r") as f:
                    common_data = json.load(f)
                    rules["common"] = common_data.get("rules", [])
                    logger.info(
                        f"Loaded {len(rules['common'])} common Android Java rules"
                    )
            else:
                rules["common"] = self._create_default_rules()
                self._save_default_rules(common_rules_path, rules["common"])

            # Load activity-specific rules
            activity_rules_path = rules_dir / "java_android_activity_errors.json"
            if activity_rules_path.exists():
                with open(activity_rules_path, "r") as f:
                    activity_data = json.load(f)
                    rules["activity"] = activity_data.get("rules", [])
                    logger.info(
                        f"Loaded {len(rules['activity'])} Android activity rules"
                    )
            else:
                rules["activity"] = []

            # Load memory-specific rules
            memory_rules_path = rules_dir / "java_android_memory_errors.json"
            if memory_rules_path.exists():
                with open(memory_rules_path, "r") as f:
                    memory_data = json.load(f)
                    rules["memory"] = memory_data.get("rules", [])
                    logger.info(f"Loaded {len(rules['memory'])} Android memory rules")
            else:
                rules["memory"] = []

        except Exception as e:
            logger.error(f"Error loading Android Java rules: {e}")
            rules = {
                "common": self._create_default_rules(),
                "activity": [],
                "memory": [],
            }

        return rules

    def _create_default_rules(self) -> List[Dict[str, Any]]:
        """Create default Android Java error rules."""
        return [
            {
                "id": "activity_not_found",
                "pattern": r"Unable to find explicit activity class|ActivityNotFoundException",
                "category": "java_android",
                "subcategory": "activity",
                "root_cause": "activity_not_found",
                "confidence": "high",
                "severity": "error",
                "suggestion": "Declare the activity in AndroidManifest.xml or check the intent action",
                "tags": ["android", "activity", "manifest", "intent"],
                "reliability": "high",
            },
            {
                "id": "permission_denied",
                "pattern": r"Permission denied|SecurityException.*permission|java.lang.SecurityException",
                "category": "java_android",
                "subcategory": "permissions",
                "root_cause": "android_permission_denied",
                "confidence": "high",
                "severity": "error",
                "suggestion": "Add required permission to AndroidManifest.xml and request runtime permissions",
                "tags": ["android", "permissions", "security", "manifest"],
                "reliability": "high",
            },
            {
                "id": "out_of_memory",
                "pattern": r"OutOfMemoryError|GC.*overhead.*limit.*exceeded|java.lang.OutOfMemoryError",
                "category": "java_android",
                "subcategory": "memory",
                "root_cause": "android_out_of_memory",
                "confidence": "high",
                "severity": "critical",
                "suggestion": "Optimize memory usage, reduce bitmap sizes, implement proper lifecycle management",
                "tags": ["android", "memory", "performance", "lifecycle"],
                "reliability": "high",
            },
            {
                "id": "illegal_state_activity",
                "pattern": r"IllegalStateException.*Activity.*destroyed|IllegalStateException.*Can not perform.*after onSaveInstanceState",
                "category": "java_android",
                "subcategory": "lifecycle",
                "root_cause": "activity_illegal_state",
                "confidence": "high",
                "severity": "error",
                "suggestion": "Check activity lifecycle state before performing UI operations",
                "tags": ["android", "activity", "lifecycle", "state"],
                "reliability": "high",
            },
            {
                "id": "fragment_not_attached",
                "pattern": r"Fragment.*not attached to activity|IllegalStateException.*Fragment.*not attached",
                "category": "java_android",
                "subcategory": "fragment",
                "root_cause": "java_android_fragment_not_attached",
                "confidence": "high",
                "severity": "error",
                "suggestion": "Check if fragment is attached before accessing activity or context",
                "tags": ["android", "fragment", "lifecycle", "activity"],
                "reliability": "high",
            },
            {
                "id": "view_not_found",
                "pattern": r"findViewById.*returned null|NullPointerException.*findViewById",
                "category": "java_android",
                "subcategory": "view",
                "root_cause": "view_not_found",
                "confidence": "medium",
                "severity": "error",
                "suggestion": "Check view ID exists in layout and setContentView is called",
                "tags": ["android", "view", "layout", "ui"],
                "reliability": "medium",
            },
            {
                "id": "network_on_main_thread",
                "pattern": r"NetworkOnMainThreadException|StrictMode policy violation.*network",
                "category": "java_android",
                "subcategory": "threading",
                "root_cause": "network_on_main_thread",
                "confidence": "high",
                "severity": "error",
                "suggestion": "Move network operations to background thread using AsyncTask or other mechanisms",
                "tags": ["android", "networking", "threading", "main-thread"],
                "reliability": "high",
            },
            {
                "id": "resource_not_found",
                "pattern": r"Resources\\$NotFoundException|android.content.res.Resources\\$NotFoundException",
                "category": "java_android",
                "subcategory": "resources",
                "root_cause": "android_resource_not_found",
                "confidence": "high",
                "severity": "error",
                "suggestion": "Check resource exists in appropriate density/configuration folder",
                "tags": ["android", "resources", "configuration", "assets"],
                "reliability": "high",
            },
            {
                "id": "api_compatibility_issue",
                "pattern": r"NoSuchMethodError.*android|java.lang.NoSuchMethodError.*Landroid",
                "category": "java_android",
                "subcategory": "sdk",
                "root_cause": "java_android_api_compatibility",
                "confidence": "high",
                "severity": "error",
                "suggestion": "Check minimum SDK version and use Build.VERSION.SDK_INT for API compatibility",
                "tags": ["android", "api", "compatibility", "sdk"],
                "reliability": "high",
            },
            {
                "id": "background_limit_violation",
                "pattern": r"Background start not allowed|IllegalStateException.*startService.*background|Not allowed to start service Intent.*app is in background",
                "category": "java_android",
                "subcategory": "service",
                "root_cause": "java_android_background_limit_violation",
                "confidence": "high",
                "severity": "error",
                "suggestion": "Use foreground service or WorkManager for background operations on Android O+",
                "tags": ["android", "service", "background", "api26"],
                "reliability": "high",
            },
            {
                "id": "activity_lifecycle_violation",
                "pattern": r"IllegalStateException.*Activity.*destroyed|Can not perform this action after onSaveInstanceState",
                "category": "java_android",
                "subcategory": "lifecycle",
                "root_cause": "java_android_lifecycle_violation",
                "confidence": "high",
                "severity": "error",
                "suggestion": "Check if (!isDestroyed() && !isFinishing()) before UI operations",
                "tags": ["android", "activity", "lifecycle", "state"],
                "reliability": "high",
            },
            {
                "id": "view_not_found_npe",
                "pattern": r"NullPointerException.*findViewById|attempt to invoke virtual method.*on a null object reference.*findViewById",
                "category": "java_android",
                "subcategory": "view",
                "root_cause": "java_android_view_not_found",
                "confidence": "high",
                "severity": "error",
                "suggestion": "Check view ID exists in layout and setContentView is called before findViewById",
                "tags": ["android", "view", "layout", "null"],
                "reliability": "high",
            },
            {
                "id": "bad_token_exception",
                "pattern": r"BadTokenException.*token.*null|Unable to add window.*token null|WindowManager\$BadTokenException",
                "category": "java_android",
                "subcategory": "window",
                "root_cause": "java_android_bad_token",
                "confidence": "high",
                "severity": "error",
                "suggestion": "Check activity context is valid and not null when showing dialogs or windows. Use valid context/token for window operations",
                "tags": ["android", "window", "dialog", "token"],
                "reliability": "high",
            },
        ]

    def _save_default_rules(self, file_path: Path, rules: List[Dict[str, Any]]):
        """Save default rules to file."""
        try:
            with open(file_path, "w") as f:
                json.dump({"rules": rules}, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving default Android Java rules: {e}")

    def _compile_patterns(self):
        """Pre-compile regex patterns for better performance."""
        self.compiled_patterns: Dict[
            str, List[tuple[re.Pattern[str], Dict[str, Any]]]
        ] = {}

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
                        f"Invalid regex pattern in Android Java rule {rule.get('id', 'unknown')}: {e}"
                    )

    def analyze_exception(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze an Android Java exception and determine its type and potential fixes.

        Args:
            error_data: Android Java error data in standard format

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
                "category": "android",
                "subcategory": best_match.get("subcategory", "unknown"),
                "confidence": best_match.get("confidence", "medium"),
                "suggested_fix": best_match.get("suggestion", ""),
                "suggestion": best_match.get("suggestion", ""),
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

        # Boost confidence for Android-specific patterns
        message = error_data.get("message", "").lower()
        stack_trace = str(error_data.get("stack_trace", "")).lower()

        if any(
            term in message or term in stack_trace
            for term in ["android", "activity", "fragment", "service"]
        ):
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
        if "android" in framework:
            context_tags.add("android")

        # Check for Android package indicators in stack trace
        if "android." in stack_trace or "androidx." in stack_trace:
            context_tags.add("android")

        # Check for specific Android components
        if "activity" in stack_trace:
            context_tags.add("activity")
        if "fragment" in stack_trace:
            context_tags.add("fragment")
        if "service" in stack_trace:
            context_tags.add("service")

        if context_tags & rule_tags:
            base_confidence += 0.1

        return min(base_confidence, 1.0)

    def _generic_analysis(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """Provide generic analysis for unmatched errors."""
        message = error_data.get("message", "").lower()
        stack_trace = str(error_data.get("stack_trace", "")).lower()

        # Basic categorization based on error patterns
        if "activity" in message or "activity" in stack_trace:
            category = "activity"
            suggestion = "Check activity lifecycle and manifest declarations"
        elif "fragment" in message or "fragment" in stack_trace:
            category = "fragment"
            suggestion = "Check fragment lifecycle and transaction management"
        elif "permission" in message or "security" in message:
            category = "permissions"
            suggestion = "Check app permissions and runtime permission requests"
        elif "memory" in message or "outofmemory" in message:
            category = "memory"
            suggestion = (
                "Optimize memory usage and implement proper resource management"
            )
        elif "view" in message or "layout" in message:
            category = "view"
            suggestion = "Check view hierarchy and layout inflation"
        elif "resource" in message:
            category = "resources"
            suggestion = "Check resource availability and configuration"
        elif "network" in message:
            category = "networking"
            suggestion = "Check network operations and background threading"
        elif "database" in message or "sqlite" in message:
            category = "database"
            suggestion = "Check database operations and schema"
        else:
            category = "unknown"
            suggestion = "Review Android component implementation and lifecycle"

        return {
            "category": "android",
            "subcategory": category,
            "confidence": "low",
            "suggested_fix": suggestion,
            "suggestion": suggestion,
            "root_cause": f"android_{category}_error",
            "severity": "medium",
            "rule_id": "android_java_generic_handler",
            "tags": ["android", "java", "generic", category],
        }

    def analyze_activity_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze Android Activity specific errors.

        Args:
            error_data: Error data with activity-related issues

        Returns:
            Analysis results with activity-specific fixes
        """
        message = error_data.get("message", "").lower()

        # Activity lifecycle errors
        if "illegalstateexception" in message and (
            "activity" in message or "destroyed" in message
        ):
            return {
                "category": "android",
                "subcategory": "activity",
                "confidence": "high",
                "suggested_fix": "Check activity lifecycle state before performing operations",
                "suggestion": "Check activity lifecycle state before performing operations",
                "root_cause": "activity_lifecycle_violation",
                "severity": "error",
                "tags": ["android", "activity", "lifecycle"],
                "fix_commands": [
                    "Check if (!isDestroyed() && !isFinishing()) before UI operations",
                    "Move long-running operations to background services",
                    "Use lifecycle-aware components",
                    "Handle configuration changes properly",
                ],
            }

        # Activity not found errors
        if (
            "activitynotfoundexception" in message
            or "unable to find explicit activity" in message
        ):
            return {
                "category": "android",
                "subcategory": "activity",
                "confidence": "high",
                "suggested_fix": "Declare activity in AndroidManifest.xml or check intent",
                "suggestion": "Declare activity in AndroidManifest.xml or check intent",
                "root_cause": "activity_not_declared",
                "severity": "error",
                "tags": ["android", "activity", "manifest", "intent"],
            }

        # Activity launch errors
        if (
            "android.util.androidruntimeexception" in message
            and "calling startactivity" in message
        ):
            return {
                "category": "android",
                "subcategory": "activity",
                "confidence": "medium",
                "suggested_fix": "Check intent validity and activity declaration",
                "suggestion": "Check intent validity and activity declaration",
                "root_cause": "activity_launch_error",
                "severity": "error",
                "tags": ["android", "activity", "intent"],
            }

        # Generic activity error
        return {
            "category": "android",
            "subcategory": "activity",
            "confidence": "medium",
            "suggested_fix": "Check activity implementation and lifecycle management",
            "suggestion": "Check activity implementation and lifecycle management",
            "root_cause": "activity_error",
            "severity": "warning",
            "tags": ["android", "activity"],
        }

    def analyze_memory_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze Android memory management errors.

        Args:
            error_data: Error data with memory issues

        Returns:
            Analysis results with memory-specific fixes
        """
        message = error_data.get("message", "").lower()

        # OutOfMemoryError
        if "outofmemoryerror" in message:
            if "bitmap" in message:
                return {
                    "category": "android",
                    "subcategory": "memory",
                    "confidence": "high",
                    "suggested_fix": "Optimize bitmap handling and implement image caching",
                    "suggestion": "Optimize bitmap handling and implement image caching",
                    "root_cause": "bitmap_out_of_memory",
                    "severity": "critical",
                    "tags": ["android", "memory", "bitmap", "image"],
                    "fix_commands": [
                        "Use BitmapFactory.Options to scale images",
                        "Implement LRU cache for bitmaps",
                        "Recycle bitmaps when no longer needed",
                        "Use Glide or Picasso for image loading",
                        "Enable largeHeap in manifest if absolutely necessary",
                    ],
                }
            else:
                return {
                    "category": "android",
                    "subcategory": "memory",
                    "confidence": "high",
                    "suggested_fix": "Analyze memory usage and implement proper resource management",
                    "suggestion": "Analyze memory usage and implement proper resource management",
                    "root_cause": "general_out_of_memory",
                    "severity": "critical",
                    "tags": ["android", "memory", "performance"],
                }

        # GC overhead limit exceeded
        if "gc overhead limit exceeded" in message:
            return {
                "category": "android",
                "subcategory": "memory",
                "confidence": "high",
                "suggested_fix": "Reduce memory allocations and optimize garbage collection",
                "suggestion": "Reduce memory allocations and optimize garbage collection",
                "root_cause": "gc_overhead_limit",
                "severity": "critical",
                "tags": ["android", "memory", "gc", "performance"],
            }

        # Generic memory error
        return {
            "category": "android",
            "subcategory": "memory",
            "confidence": "medium",
            "suggested_fix": "Implement proper memory management practices",
            "suggestion": "Implement proper memory management practices",
            "root_cause": "memory_management_error",
            "severity": "error",
            "tags": ["android", "memory"],
        }


class AndroidJavaPatchGenerator:
    """
    Generates patches for Android Java errors based on analysis results.

    This class creates code fixes for common Android Java issues using templates
    and heuristics specific to Android development patterns and best practices.
    """

    def __init__(self):
        """Initialize the Android Java patch generator."""
        self.template_dir = (
            Path(__file__).parent.parent / "patch_generation" / "templates"
        )
        self.android_java_template_dir = self.template_dir / "java_android"

        # Ensure template directory exists
        self.android_java_template_dir.mkdir(parents=True, exist_ok=True)

        # Load patch templates
        self.templates = self._load_templates()

        # Create default templates if they don't exist
        self._create_default_templates()

    def _load_templates(self) -> Dict[str, str]:
        """Load Android Java patch templates."""
        templates: Dict[str, str] = {}

        if not self.android_java_template_dir.exists():
            logger.warning(
                f"Android Java templates directory not found: {self.android_java_template_dir}"
            )
            return templates

        for template_file in self.android_java_template_dir.glob("*.java.template"):
            try:
                with open(template_file, "r") as f:
                    template_name = template_file.stem.replace(".java", "")
                    templates[template_name] = f.read()
                    logger.debug(f"Loaded Android Java template: {template_name}")
            except Exception as e:
                logger.error(
                    f"Error loading Android Java template {template_file}: {e}"
                )

        return templates

    def _create_default_templates(self):
        """Create default Android Java templates if they don't exist."""
        default_templates = {
            "activity_lifecycle_safe.java.template": """
// Safe activity lifecycle management
public class SafeActivity extends AppCompatActivity {
    private boolean isActivityDestroyed = false;
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        isActivityDestroyed = false;
    }
    
    @Override
    protected void onDestroy() {
        isActivityDestroyed = true;
        super.onDestroy();
    }
    
    // Safe method to check if activity can perform UI operations
    private boolean canPerformUIOperations() {
        return !isActivityDestroyed && !isFinishing() && !isDestroyed();
    }
    
    // Safe fragment transaction
    private void safeFragmentTransaction(Fragment fragment, String tag) {
        if (canPerformUIOperations()) {
            getSupportFragmentManager()
                .beginTransaction()
                .replace(R.id.fragment_container, fragment, tag)
                .commitAllowingStateLoss(); // Use commitAllowingStateLoss to avoid IllegalStateException
        }
    }
    
    // Safe UI update from background thread
    private void safeUIUpdate(Runnable updateTask) {
        if (canPerformUIOperations()) {
            runOnUiThread(updateTask);
        }
    }
}
""",
            "fragment_lifecycle_safe.java.template": """
// Safe fragment lifecycle management
public class SafeFragment extends Fragment {
    
    @Override
    public void onAttach(@NonNull Context context) {
        super.onAttach(context);
    }
    
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, ViewGroup container, Bundle savedInstanceState) {
        return inflater.inflate(R.layout.fragment_layout, container, false);
    }
    
    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        // Safe to access views here
        initViews(view);
    }
    
    // Safe method to check if fragment is properly attached
    private boolean isFragmentSafe() {
        return isAdded() && !isDetached() && !isRemoving() && getActivity() != null;
    }
    
    // Safe context access
    private Context getSafeContext() {
        if (isFragmentSafe()) {
            return requireContext();
        }
        return null;
    }
    
    // Safe activity access
    private Activity getSafeActivity() {
        if (isFragmentSafe()) {
            return requireActivity();
        }
        return null;
    }
    
    // Safe method call that requires context
    private void performActionSafely() {
        Context context = getSafeContext();
        if (context != null) {
            // Perform action with context
        }
    }
    
    private void initViews(View view) {
        // Initialize views safely
    }
}
""",
            "permission_handling.java.template": """
// Safe permission handling for Android
public class PermissionHelper {
    private static final int PERMISSION_REQUEST_CODE = 1001;
    
    // Check if permission is granted
    public static boolean hasPermission(Context context, String permission) {
        return ContextCompat.checkSelfPermission(context, permission) == PackageManager.PERMISSION_GRANTED;
    }
    
    // Request permission with rationale
    public static void requestPermission(Activity activity, String permission, String rationale) {
        if (ActivityCompat.shouldShowRequestPermissionRationale(activity, permission)) {
            // Show rationale dialog
            showPermissionRationale(activity, permission, rationale);
        } else {
            // Request permission directly
            ActivityCompat.requestPermissions(activity, new String[]{permission}, PERMISSION_REQUEST_CODE);
        }
    }
    
    private static void showPermissionRationale(Activity activity, String permission, String rationale) {
        new AlertDialog.Builder(activity)
            .setTitle("Permission Required")
            .setMessage(rationale)
            .setPositiveButton("Grant", (dialog, which) -> {
                ActivityCompat.requestPermissions(activity, new String[]{permission}, PERMISSION_REQUEST_CODE);
            })
            .setNegativeButton("Cancel", (dialog, which) -> dialog.dismiss())
            .show();
    }
    
    // Handle permission request result
    public static boolean handlePermissionResult(int requestCode, String[] permissions, int[] grantResults) {
        if (requestCode == PERMISSION_REQUEST_CODE) {
            return grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED;
        }
        return false;
    }
    
    // Example usage for camera permission
    public static void requestCameraPermission(Activity activity) {
        if (!hasPermission(activity, Manifest.permission.CAMERA)) {
            requestPermission(activity, Manifest.permission.CAMERA, 
                "Camera permission is required to take photos.");
        }
    }
}
""",
            "memory_optimization.java.template": """
// Memory optimization patterns for Android
public class MemoryOptimizer {
    
    // Bitmap handling with proper recycling and caching
    public static class BitmapHelper {
        private static LruCache<String, Bitmap> bitmapCache;
        
        static {
            // Get max available VM memory
            final int maxMemory = (int) (Runtime.getRuntime().maxMemory() / 1024);
            // Use 1/8th of the available memory for this cache
            final int cacheSize = maxMemory / 8;
            
            bitmapCache = new LruCache<String, Bitmap>(cacheSize) {
                @Override
                protected int sizeOf(String key, Bitmap bitmap) {
                    return bitmap.getByteCount() / 1024;
                }
                
                @Override
                protected void entryRemoved(boolean evicted, String key, Bitmap oldValue, Bitmap newValue) {
                    if (oldValue != null && !oldValue.isRecycled()) {
                        oldValue.recycle();
                    }
                }
            };
        }
        
        // Load bitmap with proper scaling
        public static Bitmap loadScaledBitmap(String path, int reqWidth, int reqHeight) {
            String cacheKey = path + "_" + reqWidth + "_" + reqHeight;
            Bitmap cached = bitmapCache.get(cacheKey);
            if (cached != null) {
                return cached;
            }
            
            // First decode with inJustDecodeBounds=true to check dimensions
            final BitmapFactory.Options options = new BitmapFactory.Options();
            options.inJustDecodeBounds = true;
            BitmapFactory.decodeFile(path, options);
            
            // Calculate inSampleSize
            options.inSampleSize = calculateInSampleSize(options, reqWidth, reqHeight);
            
            // Decode bitmap with inSampleSize set
            options.inJustDecodeBounds = false;
            Bitmap bitmap = BitmapFactory.decodeFile(path, options);
            
            if (bitmap != null) {
                bitmapCache.put(cacheKey, bitmap);
            }
            
            return bitmap;
        }
        
        private static int calculateInSampleSize(BitmapFactory.Options options, int reqWidth, int reqHeight) {
            final int height = options.outHeight;
            final int width = options.outWidth;
            int inSampleSize = 1;
            
            if (height > reqHeight || width > reqWidth) {
                final int halfHeight = height / 2;
                final int halfWidth = width / 2;
                
                while ((halfHeight / inSampleSize) >= reqHeight && (halfWidth / inSampleSize) >= reqWidth) {
                    inSampleSize *= 2;
                }
            }
            
            return inSampleSize;
        }
        
        public static void clearCache() {
            if (bitmapCache != null) {
                bitmapCache.evictAll();
            }
        }
    }
    
    // Weak reference pattern to avoid memory leaks
    public static class WeakReferenceHelper {
        private WeakReference<Activity> activityRef;
        
        public WeakReferenceHelper(Activity activity) {
            this.activityRef = new WeakReference<>(activity);
        }
        
        public Activity getActivity() {
            return activityRef != null ? activityRef.get() : null;
        }
        
        public boolean isActivityValid() {
            Activity activity = getActivity();
            return activity != null && !activity.isFinishing() && !activity.isDestroyed();
        }
    }
    
    // Memory-efficient list operations
    public static void optimizeListView(ListView listView) {
        // Enable view recycling
        listView.setRecyclerListener(new AbsListView.RecyclerListener() {
            @Override
            public void onMovedToScrapHeap(View view) {
                // Clean up any resources when view is recycled
                ImageView imageView = view.findViewById(R.id.image);
                if (imageView != null) {
                    imageView.setImageDrawable(null);
                }
            }
        });
    }
}
""",
            "background_threading.java.template": """
// Safe background threading patterns for Android
public class BackgroundTaskHelper {
    
    // Modern approach using ExecutorService
    public static class ExecutorHelper {
        private static final ExecutorService backgroundExecutor = Executors.newFixedThreadPool(4);
        private static final Handler mainHandler = new Handler(Looper.getMainLooper());
        
        public static void executeInBackground(Runnable task) {
            backgroundExecutor.execute(task);
        }
        
        public static void executeOnMainThread(Runnable task) {
            if (Looper.myLooper() == Looper.getMainLooper()) {
                task.run();
            } else {
                mainHandler.post(task);
            }
        }
        
        public static <T> void executeAsync(Callable<T> backgroundTask, Consumer<T> onResult) {
            backgroundExecutor.execute(() -> {
                try {
                    T result = backgroundTask.call();
                    executeOnMainThread(() -> onResult.accept(result));
                } catch (Exception e) {
                    Log.e("BackgroundTask", "Error in background task", e);
                }
            });
        }
    }
    
    // SafeAsyncTask with weak reference to prevent memory leaks
    public static abstract class SafeAsyncTask<T> {
        private WeakReference<Activity> activityRef;
        
        public SafeAsyncTask(Activity activity) {
            this.activityRef = new WeakReference<>(activity);
        }
        
        public void execute() {
            ExecutorHelper.executeAsync(
                this::doInBackground,
                this::onPostExecute
            );
        }
        
        protected abstract T doInBackground();
        
        protected abstract void onPostExecute(T result);
        
        protected Activity getActivity() {
            return activityRef != null ? activityRef.get() : null;
        }
        
        protected boolean isActivityValid() {
            Activity activity = getActivity();
            return activity != null && !activity.isFinishing() && !activity.isDestroyed();
        }
    }
    
    // Network request helper
    public static void performNetworkRequest(String url, NetworkCallback callback) {
        ExecutorHelper.executeInBackground(() -> {
            try {
                // Perform network request
                String result = performHttpRequest(url);
                ExecutorHelper.executeOnMainThread(() -> callback.onSuccess(result));
            } catch (Exception e) {
                ExecutorHelper.executeOnMainThread(() -> callback.onError(e));
            }
        });
    }
    
    private static String performHttpRequest(String url) throws Exception {
        // Implementation of HTTP request
        // This is a placeholder - use your preferred HTTP library
        return "Response from " + url;
    }
    
    public interface NetworkCallback {
        void onSuccess(String result);
        void onError(Exception error);
    }
}
""",
            "safe_view_operations.java.template": """
// Safe view operations to prevent common Android UI errors
public class SafeViewHelper {
    
    // Safe findViewById with null checking
    public static <T extends View> T safeFindViewById(Activity activity, @IdRes int id, Class<T> clazz) {
        if (activity == null || activity.isFinishing() || activity.isDestroyed()) {
            return null;
        }
        
        View view = activity.findViewById(id);
        if (view != null && clazz.isInstance(view)) {
            return clazz.cast(view);
        }
        
        Log.w("SafeViewHelper", "View with id " + id + " not found or wrong type");
        return null;
    }
    
    // Safe findViewById for fragments
    public static <T extends View> T safeFindViewById(View parentView, @IdRes int id, Class<T> clazz) {
        if (parentView == null) {
            return null;
        }
        
        View view = parentView.findViewById(id);
        if (view != null && clazz.isInstance(view)) {
            return clazz.cast(view);
        }
        
        Log.w("SafeViewHelper", "View with id " + id + " not found or wrong type");
        return null;
    }
    
    // Safe text setting with null checks
    public static void safeSetText(TextView textView, String text) {
        if (textView != null) {
            textView.setText(text != null ? text : "");
        }
    }
    
    // Safe image setting with null checks
    public static void safeSetImageResource(ImageView imageView, @DrawableRes int resId) {
        if (imageView != null && resId != 0) {
            try {
                imageView.setImageResource(resId);
            } catch (Resources.NotFoundException e) {
                Log.e("SafeViewHelper", "Resource not found: " + resId, e);
            }
        }
    }
    
    // Safe visibility setting
    public static void safeSetVisibility(View view, int visibility) {
        if (view != null) {
            view.setVisibility(visibility);
        }
    }
    
    // Safe click listener setting
    public static void safeSetOnClickListener(View view, View.OnClickListener listener) {
        if (view != null) {
            view.setOnClickListener(listener);
        }
    }
    
    // Safe layout parameter modification
    public static void safeUpdateLayoutParams(View view, Function<ViewGroup.LayoutParams, ViewGroup.LayoutParams> updater) {
        if (view != null && view.getLayoutParams() != null) {
            ViewGroup.LayoutParams params = view.getLayoutParams();
            ViewGroup.LayoutParams updatedParams = updater.apply(params);
            if (updatedParams != null) {
                view.setLayoutParams(updatedParams);
            }
        }
    }
    
    // Safe view animation
    public static void safeAnimateView(View view, Animator animator) {
        if (view != null && animator != null) {
            // Check if view is attached to window
            if (view.isAttachedToWindow()) {
                animator.setTarget(view);
                animator.start();
            }
        }
    }
}
""",
        }

        for template_name, template_content in default_templates.items():
            template_path = self.android_java_template_dir / template_name
            if not template_path.exists():
                try:
                    with open(template_path, "w") as f:
                        f.write(template_content)
                    logger.debug(
                        f"Created default Android Java template: {template_name}"
                    )
                except Exception as e:
                    logger.error(
                        f"Error creating default Android Java template {template_name}: {e}"
                    )

    def generate_patch(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a patch for the Android Java error.

        Args:
            error_data: The Android Java error data
            analysis: Analysis results from AndroidJavaExceptionHandler
            source_code: The source code where the error occurred

        Returns:
            Patch information or None if no patch can be generated
        """
        root_cause = analysis.get("root_cause", "")

        # Map root causes to patch strategies
        patch_strategies = {
            "activity_not_found": self._fix_activity_not_found,
            "android_permission_denied": self._fix_permission_denied,
            "android_out_of_memory": self._fix_out_of_memory,
            "activity_illegal_state": self._fix_activity_lifecycle,
            "java_android_lifecycle_violation": self._fix_activity_lifecycle,
            "java_android_fragment_not_attached": self._fix_fragment_lifecycle,
            "fragment_not_attached": self._fix_fragment_lifecycle,
            "view_not_found": self._fix_view_not_found,
            "java_android_view_not_found": self._fix_view_not_found,
            "network_on_main_thread": self._fix_network_main_thread,
            "java_android_main_thread_violation": self._fix_network_main_thread,
            "android_resource_not_found": self._fix_resource_not_found,
            "java_android_bad_token": self._fix_bad_token,
            "android_unknown_error": self._fix_generic_error,
        }

        strategy = patch_strategies.get(root_cause)
        if strategy:
            try:
                return strategy(error_data, analysis, source_code)
            except Exception as e:
                logger.error(
                    f"Error generating Android Java patch for {root_cause}: {e}"
                )

        # Try to use templates if no specific strategy matches
        return self._template_based_patch(error_data, analysis, source_code)

    def _fix_activity_not_found(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix activity not found errors."""
        return {
            "type": "suggestion",
            "description": "Add activity declaration to AndroidManifest.xml",
            "fix_commands": [
                "Add activity to AndroidManifest.xml inside <application> tag",
                "Check intent action and category are correct",
                "Verify activity class name and package",
                "Ensure exported=true if launching from external apps",
            ],
            "code_example": """
<!-- Add to AndroidManifest.xml -->
<activity
    android:name=".YourActivity"
    android:exported="false" />
    
<!-- For intent filters -->
<activity
    android:name=".YourActivity"
    android:exported="true">
    <intent-filter>
        <action android:name="android.intent.action.VIEW" />
        <category android:name="android.intent.category.DEFAULT" />
    </intent-filter>
</activity>
""",
        }

    def _fix_permission_denied(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix permission denied errors."""
        return {
            "type": "suggestion",
            "description": "Add permissions and implement runtime permission requests",
            "fix_commands": [
                "Add permission to AndroidManifest.xml",
                "Check for runtime permissions on API 23+",
                "Request permissions before using protected features",
                "Handle permission denial gracefully",
            ],
            "template": "permission_handling",
            "code_example": """
// Add to AndroidManifest.xml
<uses-permission android:name="android.permission.CAMERA" />

// Runtime permission check
if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) 
    != PackageManager.PERMISSION_GRANTED) {
    ActivityCompat.requestPermissions(this, 
        new String[]{Manifest.permission.CAMERA}, REQUEST_CODE);
}
""",
        }

    def _fix_out_of_memory(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix out of memory errors."""
        return {
            "type": "suggestion",
            "description": "Implement memory optimization strategies",
            "fix_commands": [
                "Scale down large bitmaps before loading",
                "Implement LRU cache for images",
                "Use weak references for activity references",
                "Recycle bitmaps when no longer needed",
                "Profile memory usage with Android Studio",
            ],
            "template": "memory_optimization",
            "code_example": """
// Scale bitmap to reduce memory usage
BitmapFactory.Options options = new BitmapFactory.Options();
options.inSampleSize = 2; // Scale down by factor of 2
Bitmap bitmap = BitmapFactory.decodeFile(path, options);

// Use LRU cache
LruCache<String, Bitmap> cache = new LruCache<>(cacheSize);
""",
        }

    def _fix_activity_lifecycle(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix activity lifecycle violations."""
        return {
            "type": "suggestion",
            "description": "Check activity state before performing operations",
            "fix_commands": [
                "Check if (!isDestroyed() && !isFinishing()) before UI operations",
                "Use commitAllowingStateLoss() for fragment transactions",
                "Move background operations to services",
                "Implement proper activity lifecycle awareness",
            ],
            "template": "activity_lifecycle_safe",
        }

    def _fix_fragment_lifecycle(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix fragment lifecycle issues."""
        return {
            "type": "suggestion",
            "description": "Check fragment attachment before accessing context",
            "fix_commands": [
                "Check isAdded() before using fragment",
                "Use getContext() with null checks",
                "Avoid storing activity references in fragments",
                "Use lifecycle-aware components",
            ],
            "template": "fragment_lifecycle_safe",
        }

    def _fix_view_not_found(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix view not found errors."""
        return {
            "type": "suggestion",
            "description": "Ensure view exists and setContentView is called",
            "fix_commands": [
                "Check view ID exists in layout file",
                "Ensure setContentView() is called before findViewById()",
                "Verify correct layout is inflated",
                "Add null checks after findViewById()",
            ],
            "template": "safe_view_operations",
        }

    def _fix_network_main_thread(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix network on main thread errors."""
        return {
            "type": "suggestion",
            "description": "Move network operations to background thread",
            "fix_commands": [
                "Use AsyncTask, ExecutorService, or networking library",
                "Never perform network operations on main thread",
                "Use callbacks to update UI from background thread",
                "Consider using Retrofit or OkHttp for networking",
            ],
            "template": "background_threading",
        }

    def _fix_resource_not_found(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix resource not found errors."""
        return {
            "type": "suggestion",
            "description": "Check resource exists in appropriate folders",
            "fix_commands": [
                "Verify resource exists in res/ folder",
                "Check resource naming follows conventions",
                "Ensure resource is in correct density folder",
                "Add try-catch around resource access",
            ],
        }

    def _fix_bad_token(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix bad token errors."""
        return {
            "type": "suggestion",
            "description": "Fix window token errors",
            "fix_commands": [
                "Check activity/context is valid before showing dialogs",
                "Use getApplicationContext() for non-UI operations",
                "Pass valid activity context/token to dialogs and windows",
                "Check if (!isFinishing() && !isDestroyed()) before showing dialogs",
            ],
            "code_example": """
// Safe dialog showing
if (!isFinishing() && !isDestroyed()) {
    AlertDialog dialog = new AlertDialog.Builder(this)
        .setTitle("Title")
        .setMessage("Message")
        .create();
    dialog.show();
}

// Or use DialogFragment for better lifecycle handling
DialogFragment dialogFragment = new MyDialogFragment();
dialogFragment.show(getSupportFragmentManager(), "dialog");
""",
            "suggestion": "Check activity/context is valid before showing dialogs. Pass valid activity context/token to dialogs and windows",
        }

    def _fix_generic_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix generic Android errors."""
        # Use error_data from analysis if not passed directly
        if not error_data.get("error_type") and analysis.get("error_data"):
            error_data = analysis["error_data"]

        error_type = error_data.get("error_type", "")
        message = error_data.get("message", "")

        # Check if it contains token-related keywords
        if "token" in message.lower() or "BadTokenException" in error_type:
            return self._fix_bad_token(error_data, analysis, source_code)

        return {
            "type": "suggestion",
            "description": "General Android error handling",
            "fix_commands": [
                "Check Android lifecycle and component state",
                "Verify context validity",
                "Review Android best practices",
                "Add proper error handling",
            ],
            "suggestion": "Check Android lifecycle and component state. Verify context validity",
        }

    def _template_based_patch(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Generate patch using templates."""
        root_cause = analysis.get("root_cause", "")

        # Map root causes to template names
        template_map = {
            "activity_lifecycle_violation": "activity_lifecycle_safe",
            "fragment_not_attached": "fragment_lifecycle_safe",
            "android_permission_denied": "permission_handling",
            "android_out_of_memory": "memory_optimization",
            "network_on_main_thread": "background_threading",
            "view_not_found": "safe_view_operations",
        }

        template_name = template_map.get(root_cause)
        if template_name and template_name in self.templates:
            template = self.templates[template_name]

            return {
                "type": "template",
                "template": template,
                "description": f"Applied Android Java template fix for {root_cause}",
            }

        return None


class AndroidJavaLanguagePlugin(LanguagePlugin):
    """
    Main Android Java framework plugin for Homeostasis.

    This plugin orchestrates Android Java error analysis and patch generation,
    supporting Android-specific Java development patterns and best practices.
    """

    VERSION = "1.0.0"
    AUTHOR = "Homeostasis Team"

    def __init__(self):
        """Initialize the Android Java language plugin."""
        self.language = "java_android"
        self.supported_extensions = {".java", ".kt", ".xml"}
        self.supported_frameworks = [
            "android",
            "androidx",
            "android.support",
            "android-sdk",
        ]

        # Initialize components
        self.adapter = JavaErrorAdapter()
        self.exception_handler = AndroidJavaExceptionHandler()
        self.patch_generator = AndroidJavaPatchGenerator()

        logger.info("Android Java framework plugin initialized")

    def get_language_id(self) -> str:
        """Get the unique identifier for this language."""
        return "java_android"

    def get_language_name(self) -> str:
        """Get the human-readable name of the framework."""
        return "Java Android"

    def get_language_version(self) -> str:
        """Get the version of the framework supported by this plugin."""
        return "API 16+"

    def get_supported_frameworks(self) -> List[str]:
        """Get the list of frameworks supported by this language plugin."""
        return self.supported_frameworks

    def normalize_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize Android error data to standard format.

        Args:
            error_data: Raw Android error data

        Returns:
            Normalized error data
        """
        # Map common field names to standard format
        normalized = {
            "language": "java",
            "error_type": error_data.get(
                "exception_class", error_data.get("error_type", "")
            ),
            "message": error_data.get("message", ""),
            "stack_trace": [],
            "framework": error_data.get(
                "platform", error_data.get("framework", "android")
            ),
        }

        # Convert stacktrace string to list
        stacktrace = error_data.get("stacktrace", error_data.get("stack_trace", ""))
        if isinstance(stacktrace, str):
            normalized["stack_trace"] = [
                line.strip() for line in stacktrace.split("\n") if line.strip()
            ]
        elif isinstance(stacktrace, list):
            normalized["stack_trace"] = stacktrace

        # Copy over other fields
        for key, value in error_data.items():
            if key not in normalized and value is not None:
                normalized[key] = value

        return normalized

    def denormalize_error(self, standard_error: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert standard format error data back to Android-specific format.

        Args:
            standard_error: Error data in the standard format

        Returns:
            Error data in the Android-specific format
        """
        # Map standard fields back to Android-specific format
        android_error = {
            "exception_class": standard_error.get("error_type", ""),
            "message": standard_error.get("message", ""),
            "stacktrace": "",
            "platform": standard_error.get("framework", "android"),
        }

        # Convert stack trace list to string
        stack_trace = standard_error.get("stack_trace", [])
        if isinstance(stack_trace, list):
            android_error["stacktrace"] = "\n".join(stack_trace)
        elif isinstance(stack_trace, str):
            android_error["stacktrace"] = stack_trace

        # Copy over other fields
        for key, value in standard_error.items():
            if key not in android_error and value is not None:
                android_error[key] = value

        return android_error

    def can_handle(self, error_data: Dict[str, Any]) -> bool:
        """
        Check if this plugin can handle the given error.

        Args:
            error_data: Error data to check

        Returns:
            True if this plugin can handle the error, False otherwise
        """
        # Check if framework is explicitly set to Android
        framework = error_data.get("framework", "").lower()
        if "android" in framework:
            return True

        # Check error message for Android-specific patterns
        message = error_data.get("message", "").lower()
        stack_trace = str(error_data.get("stack_trace", "")).lower()

        android_patterns = [
            r"android\.",
            r"androidx\.",
            r"android\.support\.",
            r"activity",
            r"fragment",
            r"service",
            r"broadcastreceiver",
            r"contentprovider",
            r"androidmanifest",
            r"android\.content\.",
            r"android\.app\.",
            r"android\.os\.",
            r"android\.view\.",
            r"android\.widget\.",
            r"com\.android\.",
            r"dalvik\.",
            r"art\.",
        ]

        for pattern in android_patterns:
            if re.search(pattern, message + stack_trace):
                return True

        # Check project structure for Android indicators
        context = error_data.get("context", {})
        project_files = context.get("project_files", [])

        android_project_indicators = [
            "androidmanifest.xml",
            "build.gradle",
            "app/src/main/",
            "res/layout/",
            "res/values/",
            "proguard-rules.pro",
            "gradle.properties",
        ]

        project_files_str = " ".join(project_files).lower()
        if any(
            indicator in project_files_str for indicator in android_project_indicators
        ):
            return True

        # Check dependencies for Android SDK
        dependencies = context.get("dependencies", [])
        android_dependencies = ["com.android.", "androidx.", "android.support."]
        if any(
            any(android_dep in dep for android_dep in android_dependencies)
            for dep in dependencies
        ):
            return True

        return False

    def analyze_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze an Android Java error.

        Args:
            error_data: Android Java error data

        Returns:
            Analysis results
        """
        try:
            # Ensure error data is in standard format
            if not error_data.get("language"):
                standard_error = self.adapter.to_standard_format(error_data)
            else:
                standard_error = error_data

            # Check if it's an activity-related error
            if self._is_activity_error(standard_error):
                analysis = self.exception_handler.analyze_activity_error(standard_error)

            # Check if it's a memory-related error
            elif self._is_memory_error(standard_error):
                analysis = self.exception_handler.analyze_memory_error(standard_error)

            # Default Android Java error analysis
            else:
                analysis = self.exception_handler.analyze_exception(standard_error)

            # Add plugin metadata
            analysis["plugin"] = "java_android"
            analysis["language"] = "java_android"
            analysis["plugin_version"] = self.VERSION

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing Android Java error: {e}")
            return {
                "category": "java_android",
                "subcategory": "unknown",
                "confidence": "low",
                "suggested_fix": "Unable to analyze Android Java error",
                "error": str(e),
                "plugin": "java_android",
            }

    def _is_activity_error(self, error_data: Dict[str, Any]) -> bool:
        """Check if this is an activity related error."""
        message = error_data.get("message", "").lower()
        stack_trace = str(error_data.get("stack_trace", "")).lower()

        activity_patterns = [
            "activity",
            "activitynotfoundexception",
            "illegalstateexception",
            "destroyed",
            "finishing",
        ]

        return any(
            pattern in message or pattern in stack_trace
            for pattern in activity_patterns
        )

    def _is_memory_error(self, error_data: Dict[str, Any]) -> bool:
        """Check if this is a memory related error."""
        message = error_data.get("message", "").lower()

        memory_patterns = ["outofmemoryerror", "memory", "gc overhead limit", "bitmap"]

        return any(pattern in message for pattern in memory_patterns)

    def generate_fix(
        self, analysis: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a fix for the Android Java error.

        Args:
            analysis: Analysis results
            context: Additional context containing error_data and source_code

        Returns:
            Fix information or empty dict if no fix can be generated
        """
        try:
            # Extract error data and source code from context
            error_data = context.get("error_data", analysis.get("error_data", {}))
            source_code = context.get("source_code", context.get("code_snippet", ""))

            # Generate the patch
            patch = self.patch_generator.generate_patch(
                error_data, analysis, source_code
            )

            if patch is None:
                return {}

            # Add expected fields
            patch["language"] = "java"
            patch["framework"] = "android"

            # Map the patch content to expected fields
            if "description" in patch:
                patch["suggestion"] = patch["description"]
            if "fix_commands" in patch:
                patch["suggestion"] = "\n".join(patch["fix_commands"])
            elif "code_example" in patch:
                patch["suggestion"] = patch["code_example"]

            return patch

        except Exception as e:
            logger.error(f"Error generating Android Java fix: {e}")
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
                "Android Activity lifecycle error handling",
                "Fragment lifecycle and transaction management",
                "Android Service and BroadcastReceiver debugging",
                "ContentProvider and database error resolution",
                "Android permissions and security error handling",
                "Memory optimization and OutOfMemoryError prevention",
                "Android threading and main thread violation fixes",
                "View hierarchy and layout inflation error resolution",
                "Android manifest configuration issue detection",
                "Gradle build and dependency error handling",
                "Android SDK and API compatibility checking",
                "Resource management and configuration error fixes",
                "ProGuard and code obfuscation issue resolution",
            ],
            "environments": ["android", "mobile", "dalvik", "art"],
        }


# Register the plugin
register_plugin(AndroidJavaLanguagePlugin())
