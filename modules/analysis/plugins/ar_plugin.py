"""
Augmented Reality Plugin for Homeostasis

Provides AR/VR-specific error detection and healing capabilities
"""

import os
import json
import re
from typing import Dict, List, Optional, Any
from ..language_plugin_system import LanguagePlugin
from ...emerging_tech.augmented_reality import (
    ARResilienceManager, ARPlatform, ARError, ARPerformanceMetrics
)


class ARPlugin(LanguagePlugin):
    """Plugin for AR/VR platforms and frameworks"""
    
    def __init__(self):
        super().__init__()
        self.name = "ar"
        self.version = "0.1.0"
        self.supported_extensions = [
            ".cs",      # Unity C#
            ".swift",   # ARKit
            ".java",    # ARCore Java
            ".kt",      # ARCore Kotlin
            ".js",      # WebXR
            ".ts",      # WebXR TypeScript
            ".cpp",     # Unreal C++
            ".hlsl",    # Shaders
            ".glsl"     # Shaders
        ]
        self.supported_platforms = [
            "arcore", "arkit", "unity_ar", "unreal_ar",
            "webxr", "vuforia", "openxr", "hololens",
            "oculus", "magic_leap"
        ]
        self.resilience_manager = ARResilienceManager()
        self._load_rules()
    
    def _load_rules(self):
        """Load AR-specific error rules"""
        rules_path = os.path.join(
            os.path.dirname(__file__),
            "../rules/ar/ar_errors.json"
        )
        
        if os.path.exists(rules_path):
            with open(rules_path, 'r') as f:
                self.rules = json.load(f)
        else:
            self.rules = {"rules": [], "platform_specific": {}}
    
    def detect_errors(self, code: str, file_path: str = None) -> List[Dict[str, Any]]:
        """Detect AR-specific errors in code"""
        errors = []
        
        # Detect platform
        platform = self.resilience_manager.detect_platform(code, file_path or "")
        
        # Apply rule-based detection
        for rule in self.rules.get("rules", []):
            if self._rule_applies(rule, code, platform.value):
                errors.append({
                    "type": rule["error_type"],
                    "rule_id": rule["id"],
                    "description": rule["description"],
                    "severity": rule["severity"],
                    "platform": platform.value,
                    "healing_options": rule.get("healing_options", [])
                })
        
        # Platform-specific checks
        if platform != ARPlatform.UNKNOWN:
            errors.extend(self._check_platform_specific_issues(code, platform))
        
        # Check for general AR best practices
        errors.extend(self._check_ar_best_practices(code))
        
        return errors
    
    def _rule_applies(self, rule: Dict, code: str, platform: str) -> bool:
        """Check if a rule applies to the code"""
        # Check platform compatibility
        rule_platforms = rule.get("platform", [])
        if "all" not in rule_platforms and platform not in rule_platforms:
            return False
        
        # Check pattern
        pattern = rule.get("pattern")
        if pattern and re.search(pattern, code, re.IGNORECASE | re.MULTILINE):
            return True
        
        return False
    
    def _check_platform_specific_issues(self, code: str, 
                                       platform: ARPlatform) -> List[Dict[str, Any]]:
        """Check for platform-specific issues"""
        issues = []
        
        if platform == ARPlatform.UNITY_AR:
            issues.extend(self._check_unity_ar_issues(code))
        elif platform == ARPlatform.ARCORE:
            issues.extend(self._check_arcore_issues(code))
        elif platform == ARPlatform.ARKIT:
            issues.extend(self._check_arkit_issues(code))
        elif platform == ARPlatform.WEBXR:
            issues.extend(self._check_webxr_issues(code))
        
        return issues
    
    def _check_unity_ar_issues(self, code: str) -> List[Dict[str, Any]]:
        """Check for Unity AR specific issues"""
        issues = []
        
        # Check for missing null checks on AR components
        if re.search(r"ARRaycastManager.*Raycast.*\(.*\)", code) and \
           not re.search(r"if\s*\(\s*\w+\s*!=\s*null", code):
            issues.append({
                "type": "MissingNullCheck",
                "description": "AR component used without null check",
                "severity": "medium",
                "suggestion": "Add null checks for AR components"
            })
        
        # Check for performance issues
        if re.search(r"Update\s*\(\s*\).*ARRaycast|FixedUpdate.*plane.*detection", code):
            issues.append({
                "type": "PerformanceIssue",
                "description": "Heavy AR operations in Update loop",
                "severity": "high",
                "suggestion": "Move AR operations to coroutines or less frequent updates"
            })
        
        return issues
    
    def _check_arcore_issues(self, code: str) -> List[Dict[str, Any]]:
        """Check for ARCore specific issues"""
        issues = []
        
        # Check for missing session checks
        if "frame.getCamera()" in code and "TrackingState.TRACKING" not in code:
            issues.append({
                "type": "TrackingStateCheck",
                "description": "Camera used without checking tracking state",
                "severity": "high",
                "suggestion": "Check TrackingState before using camera"
            })
        
        # Check for anchor limits
        anchor_count = len(re.findall(r"createAnchor\(", code))
        if anchor_count > 20:
            issues.append({
                "type": "ExcessiveAnchors",
                "description": f"Creating {anchor_count} anchors may impact performance",
                "severity": "medium",
                "suggestion": "Limit anchor count or implement anchor pooling"
            })
        
        return issues
    
    def _check_arkit_issues(self, code: str) -> List[Dict[str, Any]]:
        """Check for ARKit specific issues"""
        issues = []
        
        # Check for configuration issues
        if "ARWorldTrackingConfiguration" in code and \
           not re.search(r"isSupported|checkAvailability", code):
            issues.append({
                "type": "ConfigurationCheck",
                "description": "AR configuration used without availability check",
                "severity": "high",
                "suggestion": "Check configuration.isSupported before use"
            })
        
        # Check for memory intensive operations
        if re.search(r"getCurrentWorldMap.*Update|worldMap.*frequent", code):
            issues.append({
                "type": "MemoryIntensive",
                "description": "Frequent world map operations can cause memory issues",
                "severity": "medium",
                "suggestion": "Limit world map saves to key moments"
            })
        
        return issues
    
    def _check_webxr_issues(self, code: str) -> List[Dict[str, Any]]:
        """Check for WebXR specific issues"""
        issues = []
        
        # Check for missing feature detection
        if "navigator.xr.requestSession" in code and \
           "isSessionSupported" not in code:
            issues.append({
                "type": "FeatureDetection",
                "description": "XR session requested without feature detection",
                "severity": "high",
                "suggestion": "Check isSessionSupported before requestSession"
            })
        
        # Check for missing error handling
        if re.search(r"requestSession.*\(.*\)(?!.*catch)", code):
            issues.append({
                "type": "ErrorHandling",
                "description": "XR session request without error handling",
                "severity": "medium",
                "suggestion": "Add try-catch for session requests"
            })
        
        return issues
    
    def _check_ar_best_practices(self, code: str) -> List[Dict[str, Any]]:
        """Check for general AR best practices"""
        practices = []
        
        # Check for hardcoded values that should be configurable
        if re.search(r"(fov|FOV)\s*=\s*\d+|fieldOfView\s*=\s*\d+", code):
            practices.append({
                "type": "HardcodedFOV",
                "description": "Hardcoded field of view value",
                "severity": "low",
                "suggestion": "Make FOV configurable for different devices"
            })
        
        # Check for missing comfort settings
        if re.search(r"smooth.*locomotion|continuous.*movement", code, re.IGNORECASE) and \
           not re.search(r"comfort|vignette|tunnel.*vision", code, re.IGNORECASE):
            practices.append({
                "type": "ComfortSettings",
                "description": "Smooth locomotion without comfort options",
                "severity": "medium",
                "suggestion": "Add comfort settings to prevent motion sickness"
            })
        
        return practices
    
    def analyze_error(self, error_message: str, code_context: str,
                     file_path: str = None, 
                     performance_metrics: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
        """Analyze AR error and suggest fixes"""
        # Convert metrics dict to ARPerformanceMetrics if provided
        metrics = None
        if performance_metrics:
            metrics = ARPerformanceMetrics(
                fps=performance_metrics.get("fps", 60),
                frame_time_ms=performance_metrics.get("frame_time_ms", 16.67),
                cpu_usage=performance_metrics.get("cpu_usage", 0),
                gpu_usage=performance_metrics.get("gpu_usage", 0),
                memory_usage=performance_metrics.get("memory_usage", 0),
                battery_drain=performance_metrics.get("battery_drain", 0),
                thermal_state=performance_metrics.get("thermal_state", "nominal"),
                tracking_quality=performance_metrics.get("tracking_quality", 1.0),
                rendering_latency_ms=performance_metrics.get("rendering_latency_ms", 0),
                motion_to_photon_latency_ms=performance_metrics.get("motion_to_photon_latency_ms")
            )
        
        ar_error = self.resilience_manager.analyze_ar_error(
            error_message, code_context, file_path or "", metrics
        )
        
        if not ar_error:
            return None
        
        healing_strategies = self.resilience_manager.suggest_healing(ar_error)
        
        return {
            "error_type": ar_error.error_type.value,
            "platform": ar_error.platform.value,
            "description": ar_error.description,
            "confidence": ar_error.confidence,
            "suggested_fix": ar_error.suggested_fix,
            "healing_strategies": healing_strategies,
            "tracking_state": ar_error.tracking_state,
            "performance_metrics": ar_error.performance_metrics,
            "severity": ar_error.severity
        }
    
    def generate_fix(self, error_analysis: Dict[str, Any],
                    code_context: str) -> Optional[str]:
        """Generate fix code for AR error"""
        if not error_analysis or "healing_strategies" not in error_analysis:
            return None
        
        strategies = error_analysis["healing_strategies"]
        if not strategies:
            return None
        
        # Use the first applicable strategy
        strategy = strategies[0]
        
        # Create an ARError object for the manager
        from ...emerging_tech.augmented_reality import ARErrorType
        ar_error = ARError(
            error_type=ARErrorType(error_analysis["error_type"]),
            platform=ARPlatform(error_analysis["platform"]),
            description=error_analysis["description"],
            tracking_state=error_analysis.get("tracking_state"),
            performance_metrics=error_analysis.get("performance_metrics")
        )
        
        return self.resilience_manager.generate_healing_code(ar_error, strategy)
    
    def validate_fix(self, original_code: str, fixed_code: str,
                    error_analysis: Dict[str, Any]) -> bool:
        """Validate that fix addresses the AR error"""
        if not fixed_code or not fixed_code.strip():
            return False
        
        # Check for expected patterns based on strategy
        validation_patterns = {
            "tracking_recovery_ui": ["tracking", "overlay", "coach"],
            "dynamic_lod": ["LOD", "level", "detail"],
            "comfort_mode": ["comfort", "vignette", "motion"],
            "thermal_management": ["thermal", "quality", "temperature"],
            "anchor_update": ["anchor", "stabiliz", "drift"]
        }
        
        strategies = error_analysis.get("healing_strategies", [])
        if strategies:
            strategy_name = strategies[0].get("name", "")
            patterns = validation_patterns.get(strategy_name, [])
            return any(pattern in fixed_code for pattern in patterns)
        
        return True
    
    def analyze_performance(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Analyze AR application performance"""
        performance_metrics = ARPerformanceMetrics(
            fps=metrics.get("fps", 60),
            frame_time_ms=metrics.get("frame_time_ms", 16.67),
            cpu_usage=metrics.get("cpu_usage", 0),
            gpu_usage=metrics.get("gpu_usage", 0),
            memory_usage=metrics.get("memory_usage", 0),
            battery_drain=metrics.get("battery_drain", 0),
            thermal_state=metrics.get("thermal_state", "nominal"),
            tracking_quality=metrics.get("tracking_quality", 1.0),
            rendering_latency_ms=metrics.get("rendering_latency_ms", 0),
            motion_to_photon_latency_ms=metrics.get("motion_to_photon_latency_ms")
        )
        
        return self.resilience_manager.analyze_performance(performance_metrics)
    
    def check_comfort_violations(self, motion_data: Dict[str, float]) -> List[str]:
        """Check for comfort violations"""
        return self.resilience_manager.check_comfort_violations(motion_data)
    
    def get_platform_info(self, code: str, file_path: str = None) -> Dict[str, Any]:
        """Get information about the AR platform being used"""
        platform = self.resilience_manager.detect_platform(code, file_path or "")
        
        platform_info = {
            "platform": platform.value,
            "detected_features": self._detect_platform_features(code, platform),
            "rendering_api": self._detect_rendering_api(code),
            "tracking_capabilities": self._detect_tracking_capabilities(code, platform)
        }
        
        # Add platform-specific info from rules
        if platform.value in self.rules.get("platform_specific", {}):
            platform_info["requirements"] = self.rules["platform_specific"][platform.value]
        
        return platform_info
    
    def _detect_platform_features(self, code: str, platform: ARPlatform) -> List[str]:
        """Detect platform-specific features being used"""
        features = []
        
        feature_patterns = {
            "plane_detection": r"plane|surface|floor|wall|ceiling",
            "image_tracking": r"image.*track|reference.*image|marker",
            "face_tracking": r"face.*track|facial|ARFace",
            "object_tracking": r"object.*track|3d.*track",
            "cloud_anchors": r"cloud.*anchor|persistent.*anchor",
            "occlusion": r"occlusion|depth|people.*occlusion",
            "light_estimation": r"light.*estimat|illumination|environment.*probe",
            "motion_capture": r"motion.*capture|body.*track",
            "hand_tracking": r"hand.*track|finger|gesture",
            "eye_tracking": r"eye.*track|gaze"
        }
        
        for feature, pattern in feature_patterns.items():
            if re.search(pattern, code, re.IGNORECASE):
                features.append(feature)
        
        return features
    
    def _detect_rendering_api(self, code: str) -> Optional[str]:
        """Detect which rendering API is being used"""
        rendering_apis = {
            "metal": r"Metal|MTK|MTL",
            "vulkan": r"Vulkan|VK_",
            "opengl": r"OpenGL|GL_|glsl",
            "directx": r"DirectX|D3D|hlsl",
            "webgl": r"WebGL|THREE\.js|gl\.",
            "scenekit": r"SceneKit|SCN",
            "filament": r"Filament|gltfio"
        }
        
        for api, pattern in rendering_apis.items():
            if re.search(pattern, code):
                return api
        
        return None
    
    def _detect_tracking_capabilities(self, code: str, 
                                    platform: ARPlatform) -> List[str]:
        """Detect tracking capabilities being used"""
        capabilities = []
        
        tracking_patterns = {
            "6dof": r"world.*tracking|6.*dof|six.*degree",
            "3dof": r"orientation.*only|3.*dof|rotation.*only",
            "slam": r"slam|simultaneous.*localization",
            "marker": r"marker.*based|fiducial",
            "markerless": r"markerless|natural.*feature",
            "hybrid": r"hybrid.*tracking"
        }
        
        for capability, pattern in tracking_patterns.items():
            if re.search(pattern, code, re.IGNORECASE):
                capabilities.append(capability)
        
        # Platform defaults
        if not capabilities:
            if platform in [ARPlatform.ARCORE, ARPlatform.ARKIT]:
                capabilities.append("6dof")
                capabilities.append("markerless")
            elif platform == ARPlatform.VUFORIA:
                capabilities.append("marker")
        
        return capabilities
    
    def generate_platform_config(self, platform: str, 
                                device_capabilities: Dict[str, Any]) -> Dict[str, Any]:
        """Generate optimized AR configuration"""
        try:
            platform_enum = ARPlatform(platform)
            return self.resilience_manager.generate_platform_config(
                platform_enum, device_capabilities
            )
        except ValueError:
            return {"error": f"Unknown platform: {platform}"}
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return plugin capabilities"""
        return {
            "name": self.name,
            "version": self.version,
            "supported_platforms": self.supported_platforms,
            "supported_extensions": self.supported_extensions,
            "features": [
                "error_detection",
                "performance_analysis",
                "tracking_recovery",
                "comfort_optimization",
                "thermal_management",
                "multi_platform_support"
            ],
            "healing_strategies": [
                "tracking_recovery_ui",
                "dynamic_lod",
                "render_scale_adjustment",
                "comfort_mode",
                "thermal_management",
                "anchor_stabilization"
            ],
            "comfort_features": [
                "motion_smoothing",
                "vignetting",
                "teleportation",
                "snap_rotation",
                "horizon_locking"
            ]
        }