"""
Unity Framework Plugin for Homeostasis

This plugin enables Homeostasis to analyze and fix errors in Unity applications.
It provides comprehensive error handling for Unity mobile games, C# scripting issues,
platform deployment problems, and Unity-specific development challenges.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..language_plugin_system import LanguagePlugin, register_plugin

logger = logging.getLogger(__name__)


class UnityErrorAdapter:
    """
    Adapter for converting Unity errors to the standard error format.
    """

    def to_standard_format(self, unity_error: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert Unity error to standard format.

        Args:
            unity_error: Raw Unity error data

        Returns:
            Standardized error format
        """
        # Extract common fields with Unity-specific naming
        error_type = unity_error.get(
            "condition", unity_error.get("error_type", "Exception")
        )
        message = unity_error.get("stackTrace", unity_error.get("message", ""))
        stack_trace = unity_error.get("stackTrace", unity_error.get("stack_trace", []))

        # Handle Unity console log specific fields
        log_type = unity_error.get("type", "")  # Error, Warning, Log, etc.

        # Extract file and line information from Unity stack trace
        file_info = self._extract_file_info(stack_trace)

        return {
            "error_type": error_type,
            "message": message,
            "stack_trace": stack_trace,
            "language": "csharp",
            "framework": "unity",
            "runtime": unity_error.get("runtime", "unity"),
            "timestamp": unity_error.get("timestamp"),
            "file": file_info.get("file"),
            "line": file_info.get("line"),
            "column": file_info.get("column"),
            "log_type": log_type,
            "unity_object": unity_error.get(
                "object"
            ),  # Unity object that caused the error
            "scene": unity_error.get("scene"),
            "context": {
                "unity_version": unity_error.get("unityVersion"),
                "platform": unity_error.get("platform"),
                "build_target": unity_error.get("buildTarget"),
                "editor_mode": unity_error.get("editorMode", False),
                "project_path": unity_error.get("projectPath"),
            },
        }

    def _extract_file_info(self, stack_trace: Union[List, str]) -> Dict[str, Any]:
        """Extract file, line, and column information from Unity stack trace."""
        if not stack_trace:
            return {}

        # Convert to string if it's a list
        if isinstance(stack_trace, list):
            stack_str = "\n".join([str(frame) for frame in stack_trace])
        else:
            stack_str = str(stack_trace)

        # Unity stack trace patterns
        patterns = [
            r"([^:]+\.cs):(\d+)",  # file.cs:line
            r"at ([^:]+\.cs):(\d+)",  # at file.cs:line
            r"([^:]+\.cs)\(at line (\d+)\)",  # file.cs(at line number)
            r"UnityEngine\.Debug:Log.*\n.*([^:]+\.cs):(\d+)",  # Debug.Log followed by file:line
        ]

        for pattern in patterns:
            match = re.search(pattern, stack_str)
            if match:
                return {
                    "file": match.group(1),
                    "line": int(match.group(2)) if match.group(2).isdigit() else 0,
                    "column": 0,
                }

        return {}


class UnityExceptionHandler:
    """
    Handles Unity-specific exceptions with comprehensive error detection and classification.

    This class provides logic for categorizing Unity scripting errors, mobile deployment issues,
    performance problems, and game development-specific challenges.
    """

    def __init__(self):
        """Initialize the Unity exception handler."""
        self.rule_categories = {
            "scripting": "Unity C# scripting and MonoBehaviour errors",
            "null_reference": "Null reference exceptions in Unity context",
            "mobile_build": "Mobile platform build and deployment errors",
            "performance": "Performance and optimization issues",
            "ui": "Unity UI (uGUI) and Canvas errors",
            "physics": "Physics and collision system errors",
            "animation": "Animation system and Animator errors",
            "audio": "Audio system and AudioSource errors",
            "networking": "Unity networking and multiplayer errors",
            "resources": "Asset loading and resource management errors",
            "platform": "Platform-specific integration errors",
            "rendering": "Graphics and rendering pipeline errors",
        }

        # Load rules from different categories
        self.rules = self._load_rules()

        # Pre-compile regex patterns for better performance
        self._compile_patterns()

    def _load_rules(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load Unity error rules from rule files."""
        rules = {}
        rules_dir = Path(__file__).parent.parent / "rules" / "unity"

        try:
            # Create rules directory if it doesn't exist
            rules_dir.mkdir(parents=True, exist_ok=True)

            # Load common Unity rules
            common_rules_path = rules_dir / "unity_common_errors.json"
            if common_rules_path.exists():
                with open(common_rules_path, "r") as f:
                    common_data = json.load(f)
                    rules["common"] = common_data.get("rules", [])
                    logger.info(f"Loaded {len(rules['common'])} common Unity rules")
            else:
                rules["common"] = self._create_default_rules()
                self._save_default_rules(common_rules_path, rules["common"])

            # Load mobile-specific rules
            mobile_rules_path = rules_dir / "unity_mobile_errors.json"
            if mobile_rules_path.exists():
                with open(mobile_rules_path, "r") as f:
                    mobile_data = json.load(f)
                    rules["mobile"] = mobile_data.get("rules", [])
                    logger.info(f"Loaded {len(rules['mobile'])} Unity mobile rules")
            else:
                rules["mobile"] = []

            # Load scripting rules
            scripting_rules_path = rules_dir / "unity_scripting_errors.json"
            if scripting_rules_path.exists():
                with open(scripting_rules_path, "r") as f:
                    scripting_data = json.load(f)
                    rules["scripting"] = scripting_data.get("rules", [])
                    logger.info(
                        f"Loaded {len(rules['scripting'])} Unity scripting rules"
                    )
            else:
                rules["scripting"] = []

        except Exception as e:
            logger.error(f"Error loading Unity rules: {e}")
            rules = {
                "common": self._create_default_rules(),
                "mobile": [],
                "scripting": [],
            }

        return rules

    def _create_default_rules(self) -> List[Dict[str, Any]]:
        """Create default Unity error rules."""
        return [
            {
                "id": "unity_null_reference_exception",
                "pattern": r"NullReferenceException.*Object reference not set",
                "category": "unity",
                "subcategory": "null_reference",
                "root_cause": "unity_null_reference_error",
                "confidence": "high",
                "severity": "error",
                "suggestion": "Check for null GameObject or Component references before accessing them",
                "tags": ["unity", "null-reference", "scripting"],
                "reliability": "high",
            },
            {
                "id": "unity_missing_component",
                "pattern": r"MissingComponentException|GetComponent.*returned null",
                "category": "unity",
                "subcategory": "scripting",
                "root_cause": "unity_missing_component",
                "confidence": "high",
                "severity": "error",
                "suggestion": "Ensure required component is attached to GameObject or use null checks",
                "tags": ["unity", "components", "scripting"],
                "reliability": "high",
            },
            {
                "id": "unity_destroyed_object_access",
                "pattern": r"MissingReferenceException|has been destroyed.*still trying to access",
                "category": "unity",
                "subcategory": "scripting",
                "root_cause": "unity_destroyed_object_access",
                "confidence": "high",
                "severity": "error",
                "suggestion": "Check if object exists before accessing it or use proper object lifecycle management",
                "tags": ["unity", "object-lifecycle", "scripting"],
                "reliability": "high",
            },
            {
                "id": "unity_coroutine_error",
                "pattern": r"Coroutine.*couldn't be started|StopCoroutine.*error",
                "category": "unity",
                "subcategory": "scripting",
                "root_cause": "unity_coroutine_error",
                "confidence": "medium",
                "severity": "error",
                "suggestion": "Ensure GameObject is active and MonoBehaviour is enabled when starting coroutines",
                "tags": ["unity", "coroutines", "scripting"],
                "reliability": "medium",
            },
            {
                "id": "unity_mobile_build_error",
                "pattern": r"BuildFailedException|Build failed.*mobile|Android.*build.*error|iOS.*build.*error",
                "category": "unity",
                "subcategory": "mobile_build",
                "root_cause": "unity_mobile_build_error",
                "confidence": "high",
                "severity": "error",
                "suggestion": "Check platform-specific build settings and dependencies",
                "tags": ["unity", "mobile", "build", "deployment"],
                "reliability": "high",
            },
            {
                "id": "unity_ui_null_reference",
                "pattern": r"UnityEngine\.UI.*NullReferenceException|UI.*component.*null",
                "category": "unity",
                "subcategory": "ui",
                "root_cause": "unity_ui_null_reference",
                "confidence": "high",
                "severity": "error",
                "suggestion": "Check UI component references and ensure they are properly assigned",
                "tags": ["unity", "ui", "ugui", "null-reference"],
                "reliability": "high",
            },
        ]

    def _save_default_rules(self, file_path: Path, rules: List[Dict[str, Any]]):
        """Save default rules to file."""
        try:
            with open(file_path, "w") as f:
                json.dump({"rules": rules}, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving default Unity rules: {e}")

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
                        f"Invalid regex pattern in Unity rule {rule.get('id', 'unknown')}: {e}"
                    )

    def analyze_exception(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a Unity exception and determine its type and potential fixes.

        Args:
            error_data: Unity error data in standard format

        Returns:
            Analysis results with categorization and fix suggestions
        """
        error_type = error_data.get("error_type", "Exception")
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
                "category": best_match.get("category", "unity"),
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

        # Boost confidence for Unity-specific patterns
        message = (error_data.get("message") or "").lower()
        framework = (error_data.get("framework") or "").lower()

        if "unity" in message or "unity" in framework or "unityengine" in message:
            base_confidence += 0.3

        # Boost confidence based on rule reliability
        reliability = rule.get("reliability", "medium")
        reliability_boost = {"high": 0.2, "medium": 0.1, "low": 0.0}
        base_confidence += reliability_boost.get(reliability, 0.0)

        # Boost confidence for rules with specific tags that match context
        rule_tags = set(rule.get("tags", []))
        context_tags = set()

        # Infer context from error data
        context = error_data.get("context", {})
        platform = (context.get("platform") or "").lower()
        build_target = (context.get("build_target") or "").lower()

        if "android" in platform or "android" in build_target:
            context_tags.add("android")
        if "ios" in platform or "ios" in build_target:
            context_tags.add("ios")
        if "mobile" in platform or "mobile" in build_target:
            context_tags.add("mobile")
        if context.get("editor_mode"):
            context_tags.add("editor")

        # Check stack trace for specific Unity components
        stack_str = str(error_data.get("stack_trace", "")).lower()
        if "unityengine.ui" in stack_str:
            context_tags.add("ui")
        if "coroutine" in stack_str:
            context_tags.add("coroutines")
        if "monobehaviour" in stack_str:
            context_tags.add("scripting")

        if context_tags & rule_tags:
            base_confidence += 0.1

        return min(base_confidence, 1.0)

    def _generic_analysis(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """Provide generic analysis for unmatched errors."""
        error_type = error_data.get("error_type", "Exception")
        message = (error_data.get("message") or "").lower()

        # Basic categorization based on error patterns
        if "nullreferenceexception" in error_type.lower():
            category = "null_reference"
            suggestion = "Add null checks before accessing GameObjects or Components"
        elif "missing" in message and "component" in message:
            category = "scripting"
            suggestion = "Ensure required components are attached to GameObjects"
        elif "destroyed" in message or "missing" in message:
            category = "scripting"
            suggestion = "Check object lifecycle and avoid accessing destroyed objects"
        elif "coroutine" in message:
            category = "scripting"
            suggestion = "Ensure GameObject is active when working with coroutines"
        elif "build" in message:
            category = "mobile_build"
            suggestion = "Check build settings and platform-specific configuration"
        elif "ui" in message or "canvas" in message:
            category = "ui"
            suggestion = "Check Unity UI components and Canvas setup"
        elif "audio" in message:
            category = "audio"
            suggestion = "Check AudioSource and AudioClip configuration"
        elif "physics" in message or "collision" in message:
            category = "physics"
            suggestion = "Check Collider and Rigidbody setup"
        else:
            category = "unknown"
            suggestion = (
                "Review Unity implementation and check Unity Console for more details"
            )

        return {
            "category": "unity",
            "subcategory": category,
            "confidence": "low",
            "suggested_fix": suggestion,
            "root_cause": f"unity_{category}_error",
            "severity": "medium",
            "rule_id": "unity_generic_handler",
            "tags": ["unity", "generic", category],
        }

    def analyze_mobile_build_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze Unity mobile build errors.

        Args:
            error_data: Error data with mobile build issues

        Returns:
            Analysis results with mobile build specific fixes
        """
        message = (error_data.get("message") or "").lower()
        context = error_data.get("context", {})
        platform = (context.get("platform") or "").lower()

        # Android-specific build errors
        if "android" in platform or "android" in message:
            if "gradle" in message:
                return {
                    "category": "unity",
                    "subcategory": "mobile_build",
                    "confidence": "high",
                    "suggested_fix": "Check Android Gradle build configuration and dependencies",
                    "root_cause": "unity_android_gradle_error",
                    "severity": "error",
                    "tags": ["unity", "mobile", "android", "gradle"],
                    "fix_commands": [
                        "Check Unity Android Build Settings",
                        "Verify Android SDK and NDK versions",
                        "Update Gradle version in Unity preferences",
                        "Check for conflicting Android plugins",
                    ],
                }

            if "sdk" in message or "ndk" in message:
                return {
                    "category": "unity",
                    "subcategory": "mobile_build",
                    "confidence": "high",
                    "suggested_fix": "Configure Android SDK/NDK paths in Unity preferences",
                    "root_cause": "unity_android_sdk_error",
                    "severity": "error",
                    "tags": ["unity", "mobile", "android", "sdk"],
                }

        # iOS-specific build errors
        if "ios" in platform or "ios" in message:
            if "xcode" in message:
                return {
                    "category": "unity",
                    "subcategory": "mobile_build",
                    "confidence": "high",
                    "suggested_fix": "Check Xcode project settings and provisioning profiles",
                    "root_cause": "unity_ios_xcode_error",
                    "severity": "error",
                    "tags": ["unity", "mobile", "ios", "xcode"],
                    "fix_commands": [
                        "Check Unity iOS Build Settings",
                        "Verify iOS deployment target version",
                        "Update Xcode and iOS SDK",
                        "Check provisioning profiles and certificates",
                    ],
                }

            if "provisioning" in message or "certificate" in message:
                return {
                    "category": "unity",
                    "subcategory": "mobile_build",
                    "confidence": "high",
                    "suggested_fix": "Configure iOS provisioning profiles and certificates",
                    "root_cause": "unity_ios_provisioning_error",
                    "severity": "error",
                    "tags": ["unity", "mobile", "ios", "provisioning"],
                }

        # Generic mobile build error
        return {
            "category": "unity",
            "subcategory": "mobile_build",
            "confidence": "medium",
            "suggested_fix": "Check Android/iOS platform-specific build settings and dependencies",
            "root_cause": "unity_mobile_build_error",
            "severity": "error",
            "tags": ["unity", "mobile", "build"],
        }

    def analyze_scripting_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze Unity C# scripting errors.

        Args:
            error_data: Error data with Unity scripting issues

        Returns:
            Analysis results with scripting specific fixes
        """
        message = (error_data.get("message") or "").lower()
        error_type = error_data.get("error_type", "")

        # Null reference errors specific to Unity
        if "nullreferenceexception" in error_type.lower():
            return {
                "category": "unity",
                "subcategory": "null_reference",
                "confidence": "high",
                "suggested_fix": "Add null checks before accessing GameObjects or Components",
                "root_cause": "unity_null_reference_error",
                "severity": "error",
                "tags": ["unity", "scripting", "null-reference"],
                "fix_commands": [
                    "Use null conditional operator: gameObject?.GetComponent<T>()",
                    "Add null checks: if (gameObject != null)",
                    "Initialize references in Awake() or Start()",
                    "Use FindObjectOfType with null checks",
                ],
            }

        # Component access errors
        if "missing" in message and "component" in message:
            return {
                "category": "unity",
                "subcategory": "scripting",
                "confidence": "high",
                "suggested_fix": "Ensure required component is attached or use RequireComponent attribute",
                "root_cause": "unity_missing_component",
                "severity": "error",
                "tags": ["unity", "scripting", "components"],
                "fix_commands": [
                    "Add [RequireComponent(typeof(ComponentType))] attribute",
                    "Attach component manually in Inspector",
                    "Use TryGetComponent for safe component access",
                    "Add component programmatically: gameObject.AddComponent<T>()",
                ],
            }

        # Destroyed object access
        if "destroyed" in message or "missingreferenceexception" in error_type.lower():
            return {
                "category": "unity",
                "subcategory": "scripting",
                "confidence": "high",
                "suggested_fix": "Check if object exists before accessing it",
                "root_cause": "unity_destroyed_object_access",
                "severity": "error",
                "tags": ["unity", "scripting", "object-lifecycle"],
                "fix_commands": [
                    "Check if object is not null and not destroyed",
                    "Use proper object lifecycle management",
                    "Unsubscribe from events before destroying objects",
                    "Use weak references for event callbacks",
                ],
            }

        # Generic scripting error
        return {
            "category": "unity",
            "subcategory": "scripting",
            "confidence": "medium",
            "suggested_fix": "Check Unity C# scripting implementation",
            "root_cause": "unity_scripting_error",
            "severity": "medium",
            "tags": ["unity", "scripting"],
        }


class UnityPatchGenerator:
    """
    Generates patches for Unity errors based on analysis results.

    This class creates code fixes for common Unity scripting issues, mobile deployment
    problems, and game development-specific challenges.
    """

    def __init__(self):
        """Initialize the Unity patch generator."""
        self.template_dir = (
            Path(__file__).parent.parent / "patch_generation" / "templates"
        )
        self.unity_template_dir = self.template_dir / "unity"

        # Ensure template directory exists
        self.unity_template_dir.mkdir(parents=True, exist_ok=True)

        # Load patch templates
        self.templates = self._load_templates()

        # Create default templates if they don't exist
        self._create_default_templates()

    def _load_templates(self) -> Dict[str, str]:
        """Load Unity patch templates."""
        templates = {}

        if not self.unity_template_dir.exists():
            logger.warning(
                f"Unity templates directory not found: {self.unity_template_dir}"
            )
            return templates

        for template_file in self.unity_template_dir.glob("*.cs.template"):
            try:
                with open(template_file, "r") as f:
                    template_name = template_file.stem.replace(".cs", "")
                    templates[template_name] = f.read()
                    logger.debug(f"Loaded Unity template: {template_name}")
            except Exception as e:
                logger.error(f"Error loading Unity template {template_file}: {e}")

        return templates

    def _create_default_templates(self):
        """Create default Unity templates if they don't exist."""
        default_templates = {
            "null_safety_unity.cs.template": """
// Unity-specific null safety patterns
using UnityEngine;

public class SafeUnityPatterns : MonoBehaviour
{
    [SerializeField] private GameObject targetObject;
    [SerializeField] private Transform targetTransform;
    
    void Start()
    {
        // Safe GameObject access
        if (targetObject != null)
        {
            targetObject.SetActive(true);
        }
        
        // Safe component access
        var rigidbody = GetComponent<Rigidbody>();
        if (rigidbody != null)
        {
            rigidbody.velocity = Vector3.zero;
        }
        
        // Using TryGetComponent (Unity 2019.2+)
        if (TryGetComponent<Collider>(out var collider))
        {
            collider.enabled = true;
        }
        
        // Safe FindObjectOfType usage
        var gameManager = FindObjectOfType<GameManager>();
        if (gameManager != null)
        {
            gameManager.Initialize();
        }
    }
    
    void Update()
    {
        // Null conditional operator with Unity objects
        targetTransform?.Rotate(Vector3.up * Time.deltaTime * 90);
    }
}
""",
            "component_requirements.cs.template": """
// Fix for missing component errors
using UnityEngine;

// Use RequireComponent to ensure dependencies
[RequireComponent(typeof(Rigidbody))]
[RequireComponent(typeof(Collider))]
public class ComponentSafetyExample : MonoBehaviour
{
    private Rigidbody rb;
    private Collider col;
    
    void Awake()
    {
        // Get required components (guaranteed to exist)
        rb = GetComponent<Rigidbody>();
        col = GetComponent<Collider>();
    }
    
    void Start()
    {
        // Safe to use without null checks
        rb.velocity = Vector3.forward * 10f;
        col.enabled = true;
    }
    
    // Alternative: Safe component access
    void SafeComponentAccess()
    {
        // For optional components, use TryGetComponent
        if (TryGetComponent<AudioSource>(out var audioSource))
        {
            audioSource.Play();
        }
        
        // Or traditional null check
        var animator = GetComponent<Animator>();
        if (animator != null)
        {
            animator.SetTrigger("Start");
        }
    }
}
""",
            "coroutine_safety.cs.template": """
// Safe coroutine patterns in Unity
using System.Collections;
using UnityEngine;

public class SafeCoroutinePatterns : MonoBehaviour
{
    private Coroutine currentCoroutine;
    
    void Start()
    {
        // Start coroutine safely
        if (gameObject.activeInHierarchy && enabled)
        {
            currentCoroutine = StartCoroutine(SafeCoroutineExample());
        }
    }
    
    IEnumerator SafeCoroutineExample()
    {
        // Check if object is still valid during coroutine
        while (this != null && gameObject.activeInHierarchy)
        {
            // Do work
            yield return new WaitForSeconds(1f);
            
            // Check again before continuing
            if (this == null) yield break;
            
            Debug.Log("Coroutine running safely");
        }
    }
    
    void StopSafely()
    {
        // Stop coroutine safely
        if (currentCoroutine != null)
        {
            StopCoroutine(currentCoroutine);
            currentCoroutine = null;
        }
    }
    
    void OnDisable()
    {
        // Always stop coroutines when disabled
        StopSafely();
    }
    
    void OnDestroy()
    {
        // Stop coroutines before destruction
        StopSafely();
    }
}
""",
            "unity_ui_safety.cs.template": """
// Safe Unity UI patterns
using UnityEngine;
using UnityEngine.UI;

public class SafeUIPatterns : MonoBehaviour
{
    [SerializeField] private Button myButton;
    [SerializeField] private Text myText;
    [SerializeField] private Image myImage;
    
    void Start()
    {
        // Safe UI component access
        if (myButton != null)
        {
            myButton.onClick.AddListener(OnButtonClick);
        }
        
        // Safe text update
        UpdateText("Hello World");
        
        // Safe image update
        UpdateImage(null); // This should handle null gracefully
    }
    
    void UpdateText(string newText)
    {
        if (myText != null)
        {
            myText.text = newText ?? string.Empty;
        }
    }
    
    void UpdateImage(Sprite newSprite)
    {
        if (myImage != null)
        {
            myImage.sprite = newSprite;
            // Handle null sprite gracefully
            myImage.enabled = newSprite != null;
        }
    }
    
    void OnButtonClick()
    {
        Debug.Log("Button clicked safely");
    }
    
    void OnDestroy()
    {
        // Unsubscribe from UI events
        if (myButton != null)
        {
            myButton.onClick.RemoveListener(OnButtonClick);
        }
    }
}
""",
            "mobile_performance.cs.template": """
// Unity mobile performance patterns
using UnityEngine;
using System.Collections;

public class MobilePerformancePatterns : MonoBehaviour
{
    [Header("Performance Settings")]
    [SerializeField] private bool enableVSync = true;
    [SerializeField] private int targetFrameRate = 60;
    
    void Awake()
    {
        // Mobile-specific settings
        SetupMobilePerformance();
    }
    
    void SetupMobilePerformance()
    {
        // Set target frame rate for mobile
        Application.targetFrameRate = targetFrameRate;
        
        // VSync setting
        QualitySettings.vSyncCount = enableVSync ? 1 : 0;
        
        // Mobile-specific quality settings
        if (IsMobilePlatform())
        {
            // Reduce quality for mobile
            QualitySettings.antiAliasing = 0;
            QualitySettings.anisotropicFiltering = AnisotropicFiltering.Disable;
        }
    }
    
    bool IsMobilePlatform()
    {
        return Application.platform == RuntimePlatform.Android ||
               Application.platform == RuntimePlatform.IPhonePlayer;
    }
    
    void OnApplicationPause(bool pauseStatus)
    {
        // Handle mobile app lifecycle
        if (pauseStatus)
        {
            // App is being paused (going to background)
            Time.timeScale = 0f;
        }
        else
        {
            // App is being resumed
            Time.timeScale = 1f;
        }
    }
    
    void OnApplicationFocus(bool hasFocus)
    {
        // Handle focus changes on mobile
        AudioListener.pause = !hasFocus;
    }
}
""",
        }

        for template_name, template_content in default_templates.items():
            template_path = self.unity_template_dir / template_name
            if not template_path.exists():
                try:
                    with open(template_path, "w") as f:
                        f.write(template_content)
                    logger.debug(f"Created default Unity template: {template_name}")
                except Exception as e:
                    logger.error(
                        f"Error creating default Unity template {template_name}: {e}"
                    )

    def generate_patch(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a patch for the Unity error.

        Args:
            error_data: The Unity error data
            analysis: Analysis results from UnityExceptionHandler
            source_code: The source code where the error occurred

        Returns:
            Patch information or None if no patch can be generated
        """
        root_cause = analysis.get("root_cause", "")

        # Map root causes to patch strategies
        patch_strategies = {
            "unity_null_reference_error": self._fix_null_reference,
            "unity_missing_component": self._fix_missing_component,
            "unity_destroyed_object_access": self._fix_destroyed_object,
            "unity_coroutine_error": self._fix_coroutine_error,
            "unity_mobile_build_error": self._fix_mobile_build,
            "unity_ui_null_reference": self._fix_ui_null_reference,
            "unity_android_gradle_error": self._fix_android_gradle,
            "unity_android_sdk_error": self._fix_android_sdk,
            "unity_ios_xcode_error": self._fix_ios_xcode,
            "unity_ios_provisioning_error": self._fix_ios_provisioning,
        }

        strategy = patch_strategies.get(root_cause)
        if strategy:
            try:
                return strategy(error_data, analysis, source_code)
            except Exception as e:
                logger.error(f"Error generating Unity patch for {root_cause}: {e}")

        # Try to use templates if no specific strategy matches
        return self._template_based_patch(error_data, analysis, source_code)

    def _fix_null_reference(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix Unity null reference exceptions."""
        return {
            "type": "suggestion",
            "description": "Add null checks for Unity GameObjects and Components",
            "fix_commands": [
                "Use null conditional operator: gameObject?.GetComponent<T>()",
                "Add null checks: if (gameObject != null)",
                "Initialize references in Awake() or Start()",
                "Use RequireComponent attribute for dependencies",
            ],
            "template": "null_safety_unity",
            "code_example": """
// Safe Unity object access
if (gameObject != null)
{
    var component = gameObject.GetComponent<Rigidbody>();
    if (component != null)
    {
        component.velocity = Vector3.zero;
    }
}
""",
        }

    def _fix_missing_component(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix missing component errors."""
        return {
            "type": "suggestion",
            "description": "Ensure required components are attached or use RequireComponent",
            "fix_commands": [
                "Add [RequireComponent(typeof(ComponentType))] attribute",
                "Attach component manually in Inspector",
                "Use TryGetComponent for safe component access",
                "Add component programmatically if needed",
            ],
            "template": "component_requirements",
            "code_example": """
[RequireComponent(typeof(Rigidbody))]
public class MyScript : MonoBehaviour
{
    private Rigidbody rb;
    
    void Awake()
    {
        rb = GetComponent<Rigidbody>(); // Guaranteed to exist
    }
}
""",
        }

    def _fix_destroyed_object(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix destroyed object access errors."""
        return {
            "type": "suggestion",
            "description": "Check object lifecycle and avoid accessing destroyed objects",
            "fix_commands": [
                "Check if object is not null and not destroyed",
                "Use proper object lifecycle management",
                "Unsubscribe from events before destroying objects",
                "Use weak references for callbacks",
            ],
            "code_example": """
// Check before accessing
if (this != null && gameObject != null)
{
    // Safe to access object
    transform.position = Vector3.zero;
}
""",
        }

    def _fix_coroutine_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix Unity coroutine errors."""
        return {
            "type": "suggestion",
            "description": "Ensure GameObject is active when working with coroutines",
            "fix_commands": [
                "Check gameObject.activeInHierarchy before starting coroutines",
                "Stop coroutines in OnDisable() and OnDestroy()",
                "Check object validity during coroutine execution",
                "Store coroutine references for proper cleanup",
            ],
            "template": "coroutine_safety",
            "code_example": """
if (gameObject.activeInHierarchy && enabled)
{
    StartCoroutine(MyCoroutine());
}
""",
        }

    def _fix_mobile_build(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix Unity mobile build errors."""
        context = error_data.get("context", {})
        platform = (context.get("platform") or "").lower()

        if "android" in platform:
            return {
                "type": "suggestion",
                "description": "Fix Android build configuration",
                "fix_commands": [
                    "Check Unity Android Build Settings",
                    "Verify Android SDK and NDK paths",
                    "Update Gradle version in Unity preferences",
                    "Check minimum API level compatibility",
                ],
            }
        elif "ios" in platform:
            return {
                "type": "suggestion",
                "description": "Fix iOS build configuration",
                "fix_commands": [
                    "Check Unity iOS Build Settings",
                    "Verify iOS deployment target",
                    "Update Xcode and iOS SDK",
                    "Check provisioning profiles",
                ],
            }

        return {
            "type": "suggestion",
            "description": "Check mobile platform build settings",
            "fix_commands": [
                "Verify platform-specific settings",
                "Check SDK versions and paths",
                "Update build tools if needed",
            ],
        }

    def _fix_ui_null_reference(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix Unity UI null reference errors."""
        return {
            "type": "suggestion",
            "description": "Fix Unity UI component references",
            "fix_commands": [
                "Assign UI components in Inspector",
                "Add null checks before accessing UI elements",
                "Use proper UI lifecycle management",
                "Unsubscribe from UI events on destroy",
            ],
            "template": "unity_ui_safety",
        }

    def _fix_android_gradle(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix Android Gradle build errors."""
        return {
            "type": "suggestion",
            "description": "Fix Android Gradle configuration",
            "fix_commands": [
                "Update Android Gradle Plugin in Unity",
                "Check Android SDK and NDK versions",
                "Verify Gradle wrapper version",
                "Check for conflicting Android dependencies",
            ],
        }

    def _fix_ios_xcode(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix iOS Xcode build errors."""
        return {
            "type": "suggestion",
            "description": "Fix iOS Xcode configuration",
            "fix_commands": [
                "Update Xcode to latest version",
                "Check iOS deployment target compatibility",
                "Verify provisioning profiles and certificates",
                "Check iOS SDK version in Unity",
            ],
        }

    def _fix_android_sdk(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix Android SDK related errors."""
        return {
            "type": "suggestion",
            "description": "Configure Android SDK paths in Unity",
            "fix_commands": [
                "Open Unity Preferences/External Tools",
                "Set Android SDK path to valid SDK location",
                "Verify Android SDK minimum API level",
                "Install required Android SDK components",
                "Check Android NDK path if using native plugins",
            ],
        }

    def _fix_ios_provisioning(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Fix iOS provisioning profile errors."""
        return {
            "type": "suggestion",
            "description": "Fix iOS provisioning profiles and certificates",
            "fix_commands": [
                "Open Xcode and update provisioning profiles",
                "Check Apple Developer account certificates",
                "Ensure correct team ID in Unity iOS settings",
                "Update iOS bundle identifier to match provisioning profile",
            ],
        }

    def _template_based_patch(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Generate patch using templates."""
        root_cause = analysis.get("root_cause", "")

        # Map root causes to template names
        template_map = {
            "unity_null_reference_error": "null_safety_unity",
            "unity_missing_component": "component_requirements",
            "unity_coroutine_error": "coroutine_safety",
            "unity_ui_null_reference": "unity_ui_safety",
        }

        template_name = template_map.get(root_cause)
        if template_name and template_name in self.templates:
            template = self.templates[template_name]

            return {
                "type": "template",
                "template": template,
                "description": f"Applied Unity template fix for {root_cause}",
            }

        return None


class UnityLanguagePlugin(LanguagePlugin):
    """
    Main Unity framework plugin for Homeostasis.

    This plugin orchestrates Unity error analysis and patch generation,
    supporting Unity mobile games, C# scripting, and cross-platform deployment.
    """

    VERSION = "1.0.0"
    AUTHOR = "Homeostasis Team"

    def __init__(self):
        """Initialize the Unity language plugin."""
        self.language = "unity"
        self.supported_extensions = {".cs", ".js", ".boo"}  # Unity script file types
        self.supported_frameworks = [
            "unity",
            "unity3d",
            "unityengine",
            "unity.mobile",
            "unity.android",
            "unity.ios",
        ]

        # Initialize components
        self.adapter = UnityErrorAdapter()
        self.exception_handler = UnityExceptionHandler()
        self.patch_generator = UnityPatchGenerator()

        logger.info("Unity framework plugin initialized")

    def get_language_id(self) -> str:
        """Get the unique identifier for this language."""
        return "unity"

    def get_language_name(self) -> str:
        """Get the human-readable name of the framework."""
        return "Unity"

    def get_language_version(self) -> str:
        """Get the version of the framework supported by this plugin."""
        return "2019.4+"

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
        framework = (error_data.get("framework") or "").lower()
        if "unity" in framework:
            return True

        # Check runtime environment
        runtime = (error_data.get("runtime") or "").lower()
        if "unity" in runtime or "unityengine" in runtime:
            return True

        # Check error message for Unity-specific patterns
        message = (error_data.get("message") or "").lower()
        stack_trace = str(error_data.get("stack_trace") or "").lower()

        unity_patterns = [
            r"unityengine",
            r"unity3d",
            r"monobehaviour",
            r"gameobject",
            r"transform",
            r"component",
            r"rigidbody",
            r"collider",
            r"renderer",
            r"camera",
            r"light",
            r"audio.*source",
            r"canvas",
            r"ui.*text",
            r"ui.*button",
            r"ui.*image",
            r"animator",
            r"animation",
            r"coroutine",
            r"awake\(\)",
            r"start\(\)",
            r"update\(\)",
            r"fixedupdate\(\)",
            r"lateupdate\(\)",
            r"ondestroy\(\)",
            r"onapplicationpause",
            r"unity.*console",
            r"debug\.log",
        ]

        for pattern in unity_patterns:
            if re.search(pattern, message + stack_trace):
                return True

        # Check project structure indicators
        context = error_data.get("context", {})
        project_files = context.get("project_files", [])

        # Look for Unity project files
        unity_project_indicators = [
            "projectsettings/projectversion.txt",
            "assets/",
            "library/",
            "packages/",
            "userSettings/",
            "*.unity",
            "*.asset",
            "*.prefab",
            "*.mat",
            "*.physicMaterial",
            "*.anim",
            "*.controller",
        ]

        project_files_str = " ".join(project_files).lower()
        if any(
            indicator in project_files_str for indicator in unity_project_indicators
        ):
            return True

        # Check for Unity-specific error types
        error_type = error_data.get("error_type", "")
        unity_error_types = [
            "MissingComponentException",
            "MissingReferenceException",
            "UnityException",
        ]

        if any(unity_type in error_type for unity_type in unity_error_types):
            return True

        return False

    def analyze_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a Unity error.

        Args:
            error_data: Unity error data

        Returns:
            Analysis results
        """
        try:
            # Ensure error data is in standard format
            if not error_data.get("language"):
                standard_error = self.adapter.to_standard_format(error_data)
            else:
                standard_error = error_data

            # Check if it's a mobile build error
            if self._is_mobile_build_error(standard_error):
                analysis = self.exception_handler.analyze_mobile_build_error(
                    standard_error
                )

            # Check if it's a Unity scripting error
            elif self._is_scripting_error(standard_error):
                analysis = self.exception_handler.analyze_scripting_error(
                    standard_error
                )

            # Default Unity error analysis
            else:
                analysis = self.exception_handler.analyze_exception(standard_error)

            # Add plugin metadata
            analysis["plugin"] = "unity"
            analysis["language"] = "unity"
            analysis["plugin_version"] = self.VERSION

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing Unity error: {e}")
            return {
                "category": "unity",
                "subcategory": "unknown",
                "confidence": "low",
                "suggested_fix": "Unable to analyze Unity error",
                "error": str(e),
                "plugin": "unity",
            }

    def _is_mobile_build_error(self, error_data: Dict[str, Any]) -> bool:
        """Check if this is a Unity mobile build related error."""
        message = (error_data.get("message") or "").lower()
        context = error_data.get("context", {})
        platform = (context.get("platform") or "").lower()
        build_target = (context.get("build_target") or "").lower()

        mobile_patterns = [
            "build",
            "gradle",
            "xcode",
            "android",
            "ios",
            "deployment",
            "provisioning",
            "sdk",
            "ndk",
        ]

        return (any(pattern in message for pattern in mobile_patterns) or
                "mobile" in platform or
                "android" in build_target or
                "ios" in build_target)

    def _is_scripting_error(self, error_data: Dict[str, Any]) -> bool:
        """Check if this is a Unity C# scripting related error."""
        message = (error_data.get("message") or "").lower()
        stack_trace = str(error_data.get("stack_trace") or "").lower()
        error_type = error_data.get("error_type", "")

        scripting_patterns = [
            "nullreferenceexception",
            "missingcomponentexception",
            "missingreferenceexception",
            "component",
            "gameobject",
            "monobehaviour",
            "coroutine",
        ]

        return any(
            pattern in message or
            pattern in stack_trace or
            pattern in error_type.lower()
            for pattern in scripting_patterns
        )

    def generate_fix(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], source_code: str
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a fix for the Unity error.

        Args:
            error_data: The Unity error data
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
            logger.error(f"Error generating Unity fix: {e}")
            return None

    def normalize_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize Unity error data to standard format.

        Args:
            error_data: Unity-specific error data

        Returns:
            Normalized error data
        """
        return self.adapter.to_standard_format(error_data)

    def denormalize_error(self, standard_error: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert standard error format back to Unity format.

        Args:
            standard_error: Standard error data

        Returns:
            Unity-specific error data
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
                "Unity C# scripting error detection and fixes",
                "Mobile platform build error resolution (Android/iOS)",
                "GameObject and Component null reference handling",
                "Unity UI (uGUI) error detection and fixes",
                "Coroutine and async operation error handling",
                "Physics and collision system error resolution",
                "Animation and Animator error fixes",
                "Audio system error detection",
                "Performance optimization suggestions for mobile",
                "Resource loading and asset management error handling",
                "Platform-specific deployment issue resolution",
                "Unity lifecycle method error detection",
            ],
            "platforms": ["mobile", "desktop", "console", "web"],
            "environments": ["unity", "unityengine", "game-development"],
        }


# Register the plugin
register_plugin(UnityLanguagePlugin())
