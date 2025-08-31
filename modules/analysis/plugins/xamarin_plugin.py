"""
Xamarin Framework Plugin for Homeostasis

This plugin enables Homeostasis to analyze and fix errors in Xamarin applications.
It provides comprehensive error handling for Xamarin.Forms, Xamarin.iOS, Xamarin.Android,
platform-specific binding issues, and cross-platform development challenges.
"""
import logging
import re
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

from ..language_plugin_system import LanguagePlugin, register_plugin

logger = logging.getLogger(__name__)


class XamarinErrorAdapter:
    """
    Adapter for converting Xamarin errors to the standard error format.
    """
    
    def to_standard_format(self, xamarin_error: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert Xamarin error to standard format.
        
        Args:
            xamarin_error: Raw Xamarin error data
            
        Returns:
            Standardized error format
        """
        # Extract common fields
        error_type = xamarin_error.get("Type", xamarin_error.get("error_type", "Exception"))
        message = xamarin_error.get("Message", xamarin_error.get("message", ""))
        stack_trace = xamarin_error.get("StackTrace", xamarin_error.get("stack_trace", []))
        
        # Handle Xamarin-specific error fields
        inner_exception = xamarin_error.get("InnerException", {})
        platform = xamarin_error.get("Platform", xamarin_error.get("platform", ""))
        
        # Combine messages if we have inner exception
        if inner_exception and isinstance(inner_exception, dict):
            inner_message = inner_exception.get("Message", "")
            if inner_message and inner_message != message:
                message = f"{message}\nInner Exception: {inner_message}"
        
        # Extract file and line information from stack trace
        file_info = self._extract_file_info(stack_trace)
        
        return {
            "error_type": error_type,
            "message": message,
            "stack_trace": stack_trace,
            "language": "csharp",
            "framework": "xamarin",
            "runtime": xamarin_error.get("runtime", "xamarin"),
            "timestamp": xamarin_error.get("timestamp"),
            "file": file_info.get("file"),
            "line": file_info.get("line"),
            "column": file_info.get("column"),
            "platform": platform,
            "inner_exception": inner_exception,
            "context": {
                "xamarin_version": xamarin_error.get("XamarinVersion"),
                "target_framework": xamarin_error.get("TargetFramework"),
                "platform_version": xamarin_error.get("PlatformVersion"),
                "device_type": xamarin_error.get("DeviceType"),
                "project_path": xamarin_error.get("ProjectPath")
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
        
        # Common Xamarin/C# stack trace patterns
        patterns = [
            r'at ([^:]+\.cs):line (\d+)',  # at file.cs:line number
            r'in ([^:]+\.cs):(\d+)',       # in file.cs:line
            r'([^:]+\.cs)\((\d+),(\d+)\)', # file.cs(line,column)
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


class XamarinExceptionHandler:
    """
    Handles Xamarin-specific exceptions with comprehensive error detection and classification.
    
    This class provides logic for categorizing Xamarin Forms errors, platform binding issues,
    cross-platform development problems, and mobile-specific concerns.
    """
    
    def __init__(self):
        """Initialize the Xamarin exception handler."""
        self.rule_categories = {
            "forms": "Xamarin.Forms UI and binding errors",
            "platform_binding": "Platform-specific binding and native integration errors",
            "dependency_service": "Xamarin DependencyService and IOC errors",
            "navigation": "Xamarin navigation and page lifecycle errors",
            "data_binding": "MVVM data binding and property notification errors",
            "rendering": "Custom renderer and effect errors",
            "permissions": "Mobile permissions and capabilities errors",
            "lifecycle": "Application and page lifecycle errors",
            "async": "Async/await and threading errors in mobile context",
            "resources": "Resource loading and platform-specific asset errors",
            "packaging": "App packaging and deployment errors",
            "performance": "Performance and memory management errors"
        }
        
        # Load rules from different categories
        self.rules = self._load_rules()
        
        # Pre-compile regex patterns for better performance
        self._compile_patterns()
    
    def _load_rules(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load Xamarin error rules from rule files."""
        rules = {}
        rules_dir = Path(__file__).parent.parent / "rules" / "xamarin"
        
        try:
            # Create rules directory if it doesn't exist
            rules_dir.mkdir(parents=True, exist_ok=True)
            
            # Load common Xamarin rules
            common_rules_path = rules_dir / "xamarin_common_errors.json"
            if common_rules_path.exists():
                with open(common_rules_path, 'r') as f:
                    common_data = json.load(f)
                    rules["common"] = common_data.get("rules", [])
                    logger.info(f"Loaded {len(rules['common'])} common Xamarin rules")
            else:
                rules["common"] = self._create_default_rules()
                self._save_default_rules(common_rules_path, rules["common"])
            
            # Load Forms-specific rules
            forms_rules_path = rules_dir / "xamarin_forms_errors.json"
            if forms_rules_path.exists():
                with open(forms_rules_path, 'r') as f:
                    forms_data = json.load(f)
                    rules["forms"] = forms_data.get("rules", [])
                    logger.info(f"Loaded {len(rules['forms'])} Xamarin.Forms rules")
            else:
                rules["forms"] = []
            
            # Load platform-specific rules
            platform_rules_path = rules_dir / "xamarin_platform_errors.json"
            if platform_rules_path.exists():
                with open(platform_rules_path, 'r') as f:
                    platform_data = json.load(f)
                    rules["platform"] = platform_data.get("rules", [])
                    logger.info(f"Loaded {len(rules['platform'])} platform-specific rules")
            else:
                rules["platform"] = []
                    
        except Exception as e:
            logger.error(f"Error loading Xamarin rules: {e}")
            rules = {"common": self._create_default_rules(), "forms": [], "platform": []}
        
        return rules
    
    def _create_default_rules(self) -> List[Dict[str, Any]]:
        """Create default Xamarin error rules."""
        return [
            {
                "id": "xamarin_null_reference_exception",
                "pattern": r"NullReferenceException.*Object reference not set",
                "category": "xamarin",
                "subcategory": "null_reference",
                "root_cause": "xamarin_null_reference_error",
                "confidence": "high",
                "severity": "error",
                "suggestion": "Check for null values before accessing objects and properties",
                "tags": ["xamarin", "null-reference", "runtime"],
                "reliability": "high"
            },
            {
                "id": "xamarin_forms_binding_error",
                "pattern": r"Binding.*error|BindingExpression.*path error",
                "category": "xamarin",
                "subcategory": "data_binding",
                "root_cause": "xamarin_forms_binding_error",
                "confidence": "high",
                "severity": "warning",
                "suggestion": "Check XAML binding paths and ensure BindingContext is set properly",
                "tags": ["xamarin", "forms", "binding", "mvvm"],
                "reliability": "high"
            },
            {
                "id": "xamarin_dependency_service_error",
                "pattern": r"DependencyService.*could not.*resolve|No implementation.*registered",
                "category": "xamarin",
                "subcategory": "dependency_service",
                "root_cause": "xamarin_dependency_service_missing",
                "confidence": "high",
                "severity": "error",
                "suggestion": "Register implementation with DependencyService.Register<T>() in platform projects",
                "tags": ["xamarin", "dependency-service", "ioc"],
                "reliability": "high"
            },
            {
                "id": "xamarin_platform_not_supported",
                "pattern": r"PlatformNotSupportedException|not supported.*platform",
                "category": "xamarin",
                "subcategory": "platform_binding",
                "root_cause": "xamarin_platform_not_supported",
                "confidence": "high",
                "severity": "error",
                "suggestion": "Implement platform-specific version or use conditional compilation",
                "tags": ["xamarin", "platform", "compatibility"],
                "reliability": "high"
            },
            {
                "id": "xamarin_renderer_error",
                "pattern": r"Renderer.*not found|Custom.*renderer.*error",
                "category": "xamarin",
                "subcategory": "rendering",
                "root_cause": "xamarin_renderer_error",
                "confidence": "medium",
                "severity": "error",
                "suggestion": "Check custom renderer implementation and registration",
                "tags": ["xamarin", "renderer", "custom-controls"],
                "reliability": "medium"
            },
            {
                "id": "xamarin_navigation_error",
                "pattern": r"Navigation.*error|Page.*not found|PopAsync.*error",
                "category": "xamarin",
                "subcategory": "navigation",
                "root_cause": "xamarin_navigation_error",
                "confidence": "medium",
                "severity": "error",
                "suggestion": "Check navigation stack and page lifecycle management",
                "tags": ["xamarin", "navigation", "pages"],
                "reliability": "medium"
            }
        ]
    
    def _save_default_rules(self, file_path: Path, rules: List[Dict[str, Any]]):
        """Save default rules to file."""
        try:
            with open(file_path, 'w') as f:
                json.dump({"rules": rules}, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving default Xamarin rules: {e}")
    
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
                    logger.warning(f"Invalid regex pattern in Xamarin rule {rule.get('id', 'unknown')}: {e}")
    
    def analyze_exception(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a Xamarin exception and determine its type and potential fixes.
        
        Args:
            error_data: Xamarin error data in standard format
            
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
                "category": best_match.get("category", "xamarin"),
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
        
        # Boost confidence for Xamarin-specific patterns
        message = error_data.get("message", "").lower()
        framework = error_data.get("framework", "").lower()
        
        if "xamarin" in message or "xamarin" in framework:
            base_confidence += 0.3
        
        # Boost confidence based on rule reliability
        reliability = rule.get("reliability", "medium")
        reliability_boost = {"high": 0.2, "medium": 0.1, "low": 0.0}
        base_confidence += reliability_boost.get(reliability, 0.0)
        
        # Boost confidence for rules with specific tags that match context
        rule_tags = set(rule.get("tags", []))
        context_tags = set()
        
        # Infer context from error data
        platform = error_data.get("platform", "").lower()
        if "ios" in platform:
            context_tags.add("ios")
        if "android" in platform:
            context_tags.add("android")
        if "xamarin" in framework:
            context_tags.add("xamarin")
        
        # Check stack trace for specific Xamarin components
        stack_str = str(error_data.get("stack_trace", "")).lower()
        if "xamarin.forms" in stack_str:
            context_tags.add("forms")
        if "dependencyservice" in stack_str:
            context_tags.add("dependency-service")
        if "renderer" in stack_str:
            context_tags.add("renderer")
        
        if context_tags & rule_tags:
            base_confidence += 0.1
        
        return min(base_confidence, 1.0)
    
    def _generic_analysis(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """Provide generic analysis for unmatched errors."""
        error_type = error_data.get("error_type", "Exception")
        message = error_data.get("message", "").lower()
        
        # Basic categorization based on error patterns
        if "nullreferenceexception" in error_type.lower():
            category = "null_reference"
            suggestion = "Add null checks before accessing objects and properties"
        elif "binding" in message:
            category = "data_binding"
            suggestion = "Check XAML binding paths and BindingContext configuration"
        elif "dependency" in message or "service" in message:
            category = "dependency_service"
            suggestion = "Check DependencyService registration and implementation"
        elif "navigation" in message or "page" in message:
            category = "navigation"
            suggestion = "Check page navigation and lifecycle management"
        elif "renderer" in message or "custom" in message:
            category = "rendering"
            suggestion = "Check custom renderer implementation and registration"
        elif "platform" in message or "not supported" in message:
            category = "platform_binding"
            suggestion = "Implement platform-specific code or use conditional compilation"
        elif "permission" in message:
            category = "permissions"
            suggestion = "Check app permissions and runtime permission requests"
        else:
            category = "unknown"
            suggestion = "Review Xamarin implementation and check documentation"
        
        return {
            "category": "xamarin",
            "subcategory": category,
            "confidence": "low",
            "suggested_fix": suggestion,
            "root_cause": f"xamarin_{category}_error",
            "severity": "medium",
            "rule_id": "xamarin_generic_handler",
            "tags": ["xamarin", "generic", category]
        }
    
    def analyze_forms_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze Xamarin.Forms specific errors.
        
        Args:
            error_data: Error data with Xamarin.Forms issues
            
        Returns:
            Analysis results with Forms-specific fixes
        """
        message = error_data.get("message", "").lower()
        stack_trace = str(error_data.get("stack_trace", "")).lower()
        
        # Common Xamarin.Forms error patterns
        if "binding" in message and ("error" in message or "path" in message):
            return {
                "category": "xamarin",
                "subcategory": "data_binding",
                "confidence": "high",
                "suggested_fix": "Check XAML binding paths and ensure BindingContext is properly set",
                "root_cause": "xamarin_forms_binding_error",
                "severity": "warning",
                "tags": ["xamarin", "forms", "binding"],
                "fix_commands": [
                    "Verify binding path spelling and case sensitivity",
                    "Ensure BindingContext is set before binding evaluation",
                    "Check if bound property implements INotifyPropertyChanged",
                    "Use x:Name for code-behind access instead of binding if needed"
                ]
            }
        
        if "renderer" in stack_trace and ("not found" in message or "error" in message):
            return {
                "category": "xamarin",
                "subcategory": "rendering",
                "confidence": "high",
                "suggested_fix": "Register custom renderer in platform-specific projects",
                "root_cause": "xamarin_renderer_missing",
                "severity": "error",
                "tags": ["xamarin", "forms", "renderer"],
                "fix_commands": [
                    "Add [assembly: ExportRenderer] attribute in platform projects",
                    "Ensure renderer inherits from correct base class",
                    "Check renderer namespace and assembly references",
                    "Verify target control type matches renderer"
                ]
            }
        
        if "page" in message and ("navigation" in message or "not found" in message):
            return {
                "category": "xamarin",
                "subcategory": "navigation",
                "confidence": "medium",
                "suggested_fix": "Check page registration and navigation stack management",
                "root_cause": "xamarin_navigation_error",
                "severity": "error",
                "tags": ["xamarin", "forms", "navigation"]
            }
        
        # Generic Forms error
        return {
            "category": "xamarin",
            "subcategory": "forms",
            "confidence": "medium",
            "suggested_fix": "Check Xamarin.Forms implementation and XAML structure",
            "root_cause": "xamarin_forms_error",
            "severity": "medium",
            "tags": ["xamarin", "forms"]
        }
    
    def analyze_dependency_service_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze Xamarin DependencyService errors.
        
        Args:
            error_data: Error data with DependencyService issues
            
        Returns:
            Analysis results with DependencyService specific fixes
        """
        message = error_data.get("message", "").lower()
        
        if "could not resolve" in message or "no implementation" in message:
            return {
                "category": "xamarin",
                "subcategory": "dependency_service",
                "confidence": "high",
                "suggested_fix": "Register interface implementation with DependencyService in platform projects",
                "root_cause": "xamarin_dependency_service_missing",
                "severity": "error",
                "tags": ["xamarin", "dependency-service", "ioc"],
                "fix_commands": [
                    "Add DependencyService.Register<IInterface, Implementation>() in platform startup",
                    "Use [assembly: Dependency] attribute on implementation class",
                    "Ensure interface is in shared project",
                    "Check implementation is in correct platform project"
                ]
            }
        
        return {
            "category": "xamarin",
            "subcategory": "dependency_service",
            "confidence": "medium",
            "suggested_fix": "Check DependencyService registration and interface implementation",
            "root_cause": "xamarin_dependency_service_error",
            "severity": "error",
            "tags": ["xamarin", "dependency-service"]
        }


class XamarinPatchGenerator:
    """
    Generates patches for Xamarin errors based on analysis results.
    
    This class creates code fixes for common Xamarin Forms issues, platform binding
    problems, and cross-platform development challenges.
    """
    
    def __init__(self):
        """Initialize the Xamarin patch generator."""
        self.template_dir = Path(__file__).parent.parent / "patch_generation" / "templates"
        self.xamarin_template_dir = self.template_dir / "xamarin"
        
        # Ensure template directory exists
        self.xamarin_template_dir.mkdir(parents=True, exist_ok=True)
        
        # Load patch templates
        self.templates = self._load_templates()
        
        # Create default templates if they don't exist
        self._create_default_templates()
    
    def _load_templates(self) -> Dict[str, str]:
        """Load Xamarin patch templates."""
        templates = {}
        
        if not self.xamarin_template_dir.exists():
            logger.warning(f"Xamarin templates directory not found: {self.xamarin_template_dir}")
            return templates
        
        for template_file in self.xamarin_template_dir.glob("*.cs.template"):
            try:
                with open(template_file, 'r') as f:
                    template_name = template_file.stem.replace('.cs', '')
                    templates[template_name] = f.read()
                    logger.debug(f"Loaded Xamarin template: {template_name}")
            except Exception as e:
                logger.error(f"Error loading Xamarin template {template_file}: {e}")
        
        return templates
    
    def _create_default_templates(self):
        """Create default Xamarin templates if they don't exist."""
        default_templates = {
            "null_safety_fix.cs.template": """
// Fix for null reference exceptions in Xamarin
using System;

// Safe null checking patterns
public class SafeNullHandling
{
    public void SafePropertyAccess(MyObject obj)
    {
        // Null conditional operator (C# 6.0+)
        var result = obj?.Property?.SubProperty;
        
        // Null coalescing
        var value = obj?.Property ?? "default";
        
        // Traditional null check
        if (obj != null && obj.Property != null)
        {
            // Safe to use obj.Property
        }
        
        // Null conditional with method calls
        obj?.Method()?.SubMethod();
    }
}
""",
            "dependency_service_fix.cs.template": """
// Fix for DependencyService registration
using Xamarin.Forms;

// In shared project - interface definition
public interface IMyService
{
    void DoSomething();
}

// In platform project (iOS/Android) - implementation
[assembly: Dependency(typeof(MyServiceImplementation))]
namespace MyApp.iOS
{
    public class MyServiceImplementation : IMyService
    {
        public void DoSomething()
        {
            // Platform-specific implementation
        }
    }
}

// Usage in shared code
public class MyViewModel
{
    public void CallService()
    {
        var service = DependencyService.Get<IMyService>();
        service?.DoSomething();
    }
}
""",
            "forms_binding_fix.xaml.template": """
<!-- Fix for Xamarin.Forms binding issues -->
<?xml version="1.0" encoding="utf-8" ?>
<ContentPage xmlns="http://xamarin.com/schemas/2014/forms"
             xmlns:x="http://schemas.microsoft.com/winfx/2009/xaml"
             x:Class="MyApp.MyPage">
    
    <!-- Ensure BindingContext is set -->
    <ContentPage.BindingContext>
        <local:MyViewModel />
    </ContentPage.BindingContext>
    
    <StackLayout>
        <!-- Correct binding syntax -->
        <Label Text="{Binding PropertyName}" />
        
        <!-- With fallback value -->
        <Label Text="{Binding PropertyName, FallbackValue='Default Text'}" />
        
        <!-- With string format -->
        <Label Text="{Binding NumberProperty, StringFormat='Value: {0:F2}'}" />
        
        <!-- Mode specifications -->
        <Entry Text="{Binding InputProperty, Mode=TwoWay}" />
    </StackLayout>
</ContentPage>
""",
            "custom_renderer_fix.cs.template": """
// Fix for custom renderer registration
using Xamarin.Forms;
using Xamarin.Forms.Platform.iOS; // or .Android

// Custom control in shared project
public class MyCustomControl : View
{
    // Custom properties and behavior
}

// Custom renderer in platform project
[assembly: ExportRenderer(typeof(MyCustomControl), typeof(MyCustomControlRenderer))]
namespace MyApp.iOS.Renderers
{
    public class MyCustomControlRenderer : ViewRenderer<MyCustomControl, UIView>
    {
        protected override void OnElementChanged(ElementChangedEventArgs<MyCustomControl> e)
        {
            base.OnElementChanged(e);
            
            if (e.NewElement != null)
            {
                // Create and configure native control
                var nativeControl = new UIView();
                SetNativeControl(nativeControl);
            }
        }
        
        protected override void OnElementPropertyChanged(object sender, PropertyChangedEventArgs e)
        {
            base.OnElementPropertyChanged(sender, e);
            
            // Handle property changes
            if (e.PropertyName == MyCustomControl.SomeProperty.PropertyName)
            {
                // Update native control
            }
        }
    }
}
""",
            "async_safe_patterns.cs.template": """
// Safe async patterns in Xamarin
using System;
using System.Threading.Tasks;
using Xamarin.Forms;

public class AsyncSafePatterns : ContentPage
{
    private bool _isLoading = false;
    
    // Safe async event handler
    private async void OnButtonClicked(object sender, EventArgs e)
    {
        if (_isLoading) return;
        
        try
        {
            _isLoading = true;
            await PerformAsyncOperation();
        }
        catch (Exception ex)
        {
            // Handle error
            await DisplayAlert("Error", ex.Message, "OK");
        }
        finally
        {
            _isLoading = false;
        }
    }
    
    // Safe async operation with cancellation
    private async Task PerformAsyncOperation()
    {
        using (var cts = new CancellationTokenSource(TimeSpan.FromSeconds(30)))
        {
            try
            {
                await SomeAsyncMethod(cts.Token);
            }
            catch (OperationCanceledException)
            {
                // Handle cancellation
            }
        }
    }
    
    // ConfigureAwait usage
    private async Task BackgroundOperation()
    {
        await Task.Run(() => {
            // Background work
        }).ConfigureAwait(false);
        
        // Switch back to UI thread for UI updates
        Device.BeginInvokeOnMainThread(() => {
            // UI updates
        });
    }
}
"""
        }
        
        for template_name, template_content in default_templates.items():
            template_path = self.xamarin_template_dir / template_name
            if not template_path.exists():
                try:
                    with open(template_path, 'w') as f:
                        f.write(template_content)
                    logger.debug(f"Created default Xamarin template: {template_name}")
                except Exception as e:
                    logger.error(f"Error creating default Xamarin template {template_name}: {e}")
    
    def generate_patch(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                      source_code: str) -> Optional[Dict[str, Any]]:
        """
        Generate a patch for the Xamarin error.
        
        Args:
            error_data: The Xamarin error data
            analysis: Analysis results from XamarinExceptionHandler
            source_code: The source code where the error occurred
            
        Returns:
            Patch information or None if no patch can be generated
        """
        root_cause = analysis.get("root_cause", "")
        
        # Map root causes to patch strategies
        patch_strategies = {
            "xamarin_null_reference_error": self._fix_null_reference,
            "xamarin_forms_binding_error": self._fix_forms_binding,
            "xamarin_dependency_service_missing": self._fix_dependency_service,
            "xamarin_renderer_missing": self._fix_custom_renderer,
            "xamarin_navigation_error": self._fix_navigation,
            "xamarin_platform_not_supported": self._fix_platform_support,
            "xamarin_async_error": self._fix_async_patterns
        }
        
        strategy = patch_strategies.get(root_cause)
        if strategy:
            try:
                return strategy(error_data, analysis, source_code)
            except Exception as e:
                logger.error(f"Error generating Xamarin patch for {root_cause}: {e}")
        
        # Try to use templates if no specific strategy matches
        return self._template_based_patch(error_data, analysis, source_code)
    
    def _fix_null_reference(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                           source_code: str) -> Optional[Dict[str, Any]]:
        """Fix null reference exceptions."""
        return {
            "type": "suggestion",
            "description": "Add null checks and use null-conditional operators",
            "fix_commands": [
                "Use null conditional operator: obj?.Property",
                "Use null coalescing: value ?? defaultValue",
                "Add traditional null checks: if (obj != null)",
                "Consider using nullable reference types (C# 8.0+)"
            ],
            "template": "null_safety_fix",
            "code_example": """
// Safe null handling
var result = myObject?.Property?.SubProperty;
var safeValue = result ?? "default";
"""
        }
    
    def _fix_forms_binding(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                          source_code: str) -> Optional[Dict[str, Any]]:
        """Fix Xamarin.Forms binding errors."""
        return {
            "type": "suggestion",
            "description": "Fix XAML binding paths and BindingContext configuration",
            "fix_commands": [
                "Verify binding path spelling and case sensitivity",
                "Ensure BindingContext is set before binding evaluation",
                "Implement INotifyPropertyChanged on view models",
                "Use FallbackValue for robust binding"
            ],
            "template": "forms_binding_fix",
            "code_example": """
<!-- Correct binding syntax -->
<Label Text="{Binding PropertyName, FallbackValue='Default'}" />
"""
        }
    
    def _fix_dependency_service(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                               source_code: str) -> Optional[Dict[str, Any]]:
        """Fix DependencyService registration issues."""
        return {
            "type": "suggestion",
            "description": "Register interface implementation with DependencyService",
            "fix_commands": [
                "Add [assembly: Dependency(typeof(Implementation))] in platform project",
                "Ensure interface is accessible from shared code",
                "Use DependencyService.Register<I, T>() programmatically",
                "Check implementation is in correct platform project"
            ],
            "template": "dependency_service_fix",
            "code_example": """
[assembly: Dependency(typeof(MyServiceImplementation))]
// Or programmatically:
DependencyService.Register<IMyService, MyServiceImplementation>();
"""
        }
    
    def _fix_custom_renderer(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                            source_code: str) -> Optional[Dict[str, Any]]:
        """Fix custom renderer registration issues."""
        return {
            "type": "suggestion",
            "description": "Register custom renderer in platform projects",
            "fix_commands": [
                "Add [assembly: ExportRenderer] attribute",
                "Ensure renderer inherits from correct base class",
                "Check namespace and assembly references",
                "Verify target control type matches"
            ],
            "template": "custom_renderer_fix",
            "code_example": """
[assembly: ExportRenderer(typeof(MyControl), typeof(MyControlRenderer))]
"""
        }
    
    def _fix_navigation(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                       source_code: str) -> Optional[Dict[str, Any]]:
        """Fix navigation and page lifecycle issues."""
        return {
            "type": "suggestion",
            "description": "Fix page navigation and lifecycle management",
            "fix_commands": [
                "Check navigation stack state before navigation",
                "Use await with navigation methods",
                "Handle page lifecycle events properly",
                "Verify page is properly registered"
            ]
        }
    
    def _fix_platform_support(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                             source_code: str) -> Optional[Dict[str, Any]]:
        """Fix platform not supported exceptions."""
        return {
            "type": "suggestion",
            "description": "Implement platform-specific code or conditional compilation",
            "fix_commands": [
                "Use Device.RuntimePlatform for platform detection",
                "Implement platform-specific interfaces",
                "Use conditional compilation directives",
                "Check API availability before usage"
            ],
            "code_example": """
if (Device.RuntimePlatform == Device.iOS)
{
    // iOS-specific code
}
else if (Device.RuntimePlatform == Device.Android)
{
    // Android-specific code
}
"""
        }
    
    def _fix_async_patterns(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                           source_code: str) -> Optional[Dict[str, Any]]:
        """Fix async/await patterns in mobile context."""
        return {
            "type": "suggestion",
            "description": "Use safe async patterns for mobile development",
            "fix_commands": [
                "Use ConfigureAwait(false) for library code",
                "Handle OperationCanceledException",
                "Use Device.BeginInvokeOnMainThread for UI updates",
                "Implement proper cancellation tokens"
            ],
            "template": "async_safe_patterns"
        }
    
    def _template_based_patch(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                            source_code: str) -> Optional[Dict[str, Any]]:
        """Generate patch using templates."""
        root_cause = analysis.get("root_cause", "")
        
        # Map root causes to template names
        template_map = {
            "xamarin_null_reference_error": "null_safety_fix",
            "xamarin_forms_binding_error": "forms_binding_fix",
            "xamarin_dependency_service_missing": "dependency_service_fix",
            "xamarin_renderer_missing": "custom_renderer_fix",
            "xamarin_async_error": "async_safe_patterns"
        }
        
        template_name = template_map.get(root_cause)
        if template_name and template_name in self.templates:
            template = self.templates[template_name]
            
            return {
                "type": "template",
                "template": template,
                "description": f"Applied Xamarin template fix for {root_cause}"
            }
        
        return None


class XamarinLanguagePlugin(LanguagePlugin):
    """
    Main Xamarin framework plugin for Homeostasis.
    
    This plugin orchestrates Xamarin error analysis and patch generation,
    supporting Xamarin.Forms, Xamarin.iOS, and Xamarin.Android applications.
    """
    
    VERSION = "1.0.0"
    AUTHOR = "Homeostasis Team"
    
    def __init__(self):
        """Initialize the Xamarin language plugin."""
        self.language = "xamarin"
        self.supported_extensions = {".cs", ".xaml"}
        self.supported_frameworks = [
            "xamarin", "xamarin.forms", "xamarin.ios", "xamarin.android",
            "xamarin.mac", "xamarin.essentials"
        ]
        
        # Initialize components
        self.adapter = XamarinErrorAdapter()
        self.exception_handler = XamarinExceptionHandler()
        self.patch_generator = XamarinPatchGenerator()
        
        logger.info("Xamarin framework plugin initialized")
    
    def get_language_id(self) -> str:
        """Get the unique identifier for this language."""
        return "xamarin"
    
    def get_language_name(self) -> str:
        """Get the human-readable name of the framework."""
        return "Xamarin"
    
    def get_language_version(self) -> str:
        """Get the version of the framework supported by this plugin."""
        return "4.0+"
    
    def get_supported_frameworks(self) -> List[str]:
        """Get the list of frameworks supported by this language plugin."""
        return self.supported_frameworks
    
    def normalize_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize error data to the standard Homeostasis format."""
        return self.adapter.to_standard_format(error_data)
    
    def denormalize_error(self, standard_error: Dict[str, Any]) -> Dict[str, Any]:
        """Convert standard format error data back to the language-specific format."""
        # Convert back to Xamarin/C# format
        return {
            "Type": standard_error.get("error_type", "Exception"),
            "Message": standard_error.get("message", ""),
            "StackTrace": standard_error.get("stack_trace", []),
            "Source": standard_error.get("source", ""),
            "TargetSite": standard_error.get("target_site", ""),
            "InnerException": standard_error.get("inner_exception"),
            "HResult": standard_error.get("hresult", 0),
            "Data": standard_error.get("additional_data", {}),
            "HelpLink": standard_error.get("help_link", ""),
            "Platform": standard_error.get("platform", ""),
            "DeviceInfo": standard_error.get("device_info", {})
        }
    
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
        if "xamarin" in framework:
            return True
        
        # Check runtime environment
        runtime = error_data.get("runtime", "").lower()
        if "xamarin" in runtime or "mono" in runtime:
            return True
        
        # Check error message for Xamarin-specific patterns
        message = error_data.get("message", "").lower()
        stack_trace = str(error_data.get("stack_trace", "")).lower()
        
        xamarin_patterns = [
            r"xamarin",
            r"xamarin\.forms",
            r"xamarin\.ios",
            r"xamarin\.android",
            r"dependencyservice",
            r"exportrenderer",
            r"bindingcontext",
            r"contentpage",
            r"stacklayout",
            r"bindable.*property",
            r"renderer",
            r"effect",
            r"platform.*specific",
            r"device\.runtime",
            r"assembly.*dependency"
        ]
        
        for pattern in xamarin_patterns:
            if re.search(pattern, message + stack_trace):
                return True
        
        # Check file extensions and project structure
        context = error_data.get("context", {})
        project_files = context.get("project_files", [])
        
        # Look for Xamarin project indicators
        xamarin_project_indicators = [
            "packages.config",
            "app.xaml",
            "mainpage.xaml", 
            "assemblyinfo.cs",
            "appdelegate.cs",
            "mainactivity.cs",
            "info.plist",
            "androidmanifest.xml"
        ]
        
        project_files_str = " ".join(project_files).lower()
        if any(indicator in project_files_str for indicator in xamarin_project_indicators):
            # Additional check for Xamarin-specific dependencies
            dependencies = context.get("dependencies", [])
            xamarin_dependencies = ["xamarin.forms", "xamarin.essentials", "xamarin.android", "xamarin.ios"]
            if any(any(xam_dep in dep.lower() for xam_dep in xamarin_dependencies) for dep in dependencies):
                return True
        
        return False
    
    def analyze_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a Xamarin error.
        
        Args:
            error_data: Xamarin error data
            
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
            stack_trace = str(standard_error.get("stack_trace", "")).lower()
            
            # Check if it's a Xamarin.Forms error
            if self._is_forms_error(standard_error):
                analysis = self.exception_handler.analyze_forms_error(standard_error)
            
            # Check if it's a DependencyService error
            elif self._is_dependency_service_error(standard_error):
                analysis = self.exception_handler.analyze_dependency_service_error(standard_error)
            
            # Default Xamarin error analysis
            else:
                analysis = self.exception_handler.analyze_exception(standard_error)
            
            # Add plugin metadata
            analysis["plugin"] = "xamarin"
            analysis["language"] = "xamarin"
            analysis["plugin_version"] = self.VERSION
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing Xamarin error: {e}")
            return {
                "category": "xamarin",
                "subcategory": "unknown",
                "confidence": "low",
                "suggested_fix": "Unable to analyze Xamarin error",
                "error": str(e),
                "plugin": "xamarin"
            }
    
    def _is_forms_error(self, error_data: Dict[str, Any]) -> bool:
        """Check if this is a Xamarin.Forms related error."""
        message = error_data.get("message", "").lower()
        stack_trace = str(error_data.get("stack_trace", "")).lower()
        framework = error_data.get("framework", "").lower()
        error_type = error_data.get("error_type", "").lower()
        
        # Check framework explicitly
        if "xamarin.forms" in framework:
            return True
            
        # Check error type for Xamarin.Forms specific exceptions
        if "bindingexception" in error_type:
            return True
        
        forms_patterns = [
            "xamarin.forms",
            "binding",
            "bindingcontext",
            "renderer",
            "contentpage",
            "stacklayout",
            "navigation",
            "xaml"
        ]
        
        return any(pattern in message or pattern in stack_trace for pattern in forms_patterns)
    
    def _is_dependency_service_error(self, error_data: Dict[str, Any]) -> bool:
        """Check if this is a DependencyService related error."""
        message = error_data.get("message", "").lower()
        stack_trace = str(error_data.get("stack_trace", "")).lower()
        
        dependency_patterns = [
            "dependencyservice",
            "could not resolve",
            "no implementation",
            "dependency",
            "ioc"
        ]
        
        return any(pattern in message or pattern in stack_trace for pattern in dependency_patterns)
    
    def generate_fix(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                    source_code: str = "") -> Optional[Dict[str, Any]]:
        """
        Generate a fix for an error based on the analysis.
        
        Args:
            error_data: Original error data
            analysis: Error analysis
            source_code: Source code context (optional)
            
        Returns:
            Generated fix data
        """
        try:
            
            # Generate patch
            patch_result = self.patch_generator.generate_patch(error_data, analysis, source_code)
            
            if patch_result:
                return patch_result
            
            # If no specific patch was generated, return a generic fix with binding info
            if "binding" in error_data.get("message", "").lower():
                return {
                    "type": "suggestion",
                    "description": "Fix XAML binding issue",
                    "fix_commands": [
                        "Check binding path spelling",
                        "Ensure BindingContext is set",
                        "Verify property exists on view model"
                    ]
                }
            
            # Return empty dict if no patch generated (as per abstract method)
            return {}
        except Exception as e:
            logger.error(f"Error generating Xamarin fix: {e}")
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
                "Xamarin.Forms UI and data binding error handling",
                "Platform-specific code integration fixes",
                "DependencyService and IOC container resolution",
                "Custom renderer and effect error detection",
                "Cross-platform navigation and lifecycle management",
                "MVVM pattern and data binding fixes", 
                "Mobile permissions and capabilities handling",
                "Async/await patterns for mobile development",
                "Resource loading and platform asset management",
                "App packaging and deployment error resolution"
            ],
            "platforms": ["ios", "android", "mobile"],
            "environments": ["xamarin", "mono", "mobile"]
        }


# Register the plugin
register_plugin(XamarinLanguagePlugin())