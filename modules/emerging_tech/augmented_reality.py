"""
Augmented Reality Application Resilience Module

Provides error detection, healing, and resilience for AR/VR applications
including tracking, rendering, performance, and user experience issues.
"""

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ARPlatform(Enum):
    """Supported AR/VR platforms and frameworks"""

    ARCORE = "arcore"  # Google ARCore
    ARKIT = "arkit"  # Apple ARKit
    UNITY_AR = "unity_ar"  # Unity AR Foundation
    UNREAL_AR = "unreal_ar"  # Unreal Engine AR
    WEBXR = "webxr"  # WebXR API
    VUFORIA = "vuforia"  # Vuforia SDK
    OPENXR = "openxr"  # OpenXR Standard
    HOLOLENS = "hololens"  # Microsoft HoloLens
    OCULUS = "oculus"  # Meta Quest/Oculus
    MAGIC_LEAP = "magic_leap"  # Magic Leap
    UNKNOWN = "unknown"


class ARErrorType(Enum):
    """Types of AR/VR application errors"""

    TRACKING_LOST = "tracking_lost"
    RENDERING_ERROR = "rendering_error"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    CALIBRATION_ERROR = "calibration_error"
    SENSOR_FUSION_ERROR = "sensor_fusion_error"
    OCCLUSION_ERROR = "occlusion_error"
    LIGHTING_ESTIMATION_ERROR = "lighting_estimation_error"
    ANCHOR_DRIFT = "anchor_drift"
    PLANE_DETECTION_FAILURE = "plane_detection_failure"
    MOTION_SICKNESS_RISK = "motion_sickness_risk"
    MEMORY_PRESSURE = "memory_pressure"
    THERMAL_THROTTLING = "thermal_throttling"
    NETWORK_LATENCY = "network_latency"
    CLOUD_ANCHOR_ERROR = "cloud_anchor_error"


@dataclass
class ARError:
    """Represents an AR/VR application error"""

    error_type: ARErrorType
    platform: ARPlatform
    description: str
    tracking_state: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, float]] = None
    sensor_data: Optional[Dict[str, Any]] = None
    rendering_info: Optional[Dict[str, Any]] = None
    suggested_fix: Optional[str] = None
    confidence: float = 0.0
    severity: str = "medium"
    timestamp: Optional[datetime] = None


@dataclass
class ARPerformanceMetrics:
    """AR application performance metrics"""

    fps: float
    frame_time_ms: float
    cpu_usage: float
    gpu_usage: float
    memory_usage: float
    battery_drain: float
    thermal_state: str
    tracking_quality: float
    rendering_latency_ms: float
    motion_to_photon_latency_ms: Optional[float] = None


class ARResilienceManager:
    """Handles AR/VR application resilience and error recovery"""

    def __init__(self):
        self.error_patterns = self._load_error_patterns()
        self.healing_strategies = self._load_healing_strategies()
        self.platform_detectors = self._initialize_platform_detectors()
        self.performance_thresholds = self._initialize_performance_thresholds()
        self.comfort_parameters = self._initialize_comfort_parameters()

    def _load_error_patterns(self) -> Dict[str, List[Dict]]:
        """Load AR error patterns for different platforms"""
        return {
            "arcore": [
                {
                    "pattern": r"TrackingState.*PAUSED|Tracking.*lost",
                    "type": ARErrorType.TRACKING_LOST,
                    "description": "ARCore tracking lost",
                    "fix": "Show tracking recovery UI and guide user",
                },
                {
                    "pattern": r"Insufficient.*features|Not.*enough.*feature.*points",
                    "type": ARErrorType.TRACKING_LOST,
                    "description": "Insufficient visual features for tracking",
                    "fix": "Guide user to better lit area with more texture",
                },
                {
                    "pattern": r"Cloud.*anchor.*failed|Anchor.*hosting.*error",
                    "type": ARErrorType.CLOUD_ANCHOR_ERROR,
                    "description": "Cloud anchor operation failed",
                    "fix": "Retry with better network or fallback to local anchor",
                },
            ],
            "arkit": [
                {
                    "pattern": r"ARSession.*interrupted|session.*was.*interrupted",
                    "type": ARErrorType.TRACKING_LOST,
                    "description": "ARKit session interrupted",
                    "fix": "Resume session and restore anchors",
                },
                {
                    "pattern": r"worldTrackingFailed|configuration.*unsupported",
                    "type": ARErrorType.TRACKING_LOST,
                    "description": "World tracking failed",
                    "fix": "Fallback to simpler tracking configuration",
                },
                {
                    "pattern": r"insufficientFeatures|limitedTracking",
                    "type": ARErrorType.TRACKING_LOST,
                    "description": "Limited tracking due to insufficient features",
                    "fix": "Show visual guidance for better tracking",
                },
            ],
            "unity_ar": [
                {
                    "pattern": r"XR.*Subsystem.*failed|AR.*Session.*failed.*start",
                    "type": ARErrorType.TRACKING_LOST,
                    "description": "Unity AR subsystem failure",
                    "fix": "Restart AR session with fallback configuration",
                },
                {
                    "pattern": r"Plane.*detection.*timeout|No.*planes.*detected",
                    "type": ARErrorType.PLANE_DETECTION_FAILURE,
                    "description": "Plane detection failed",
                    "fix": "Guide user to scan flat surfaces",
                },
                {
                    "pattern": r"Performance.*warning|FPS.*dropped|Frame.*budget.*exceeded",
                    "type": ARErrorType.PERFORMANCE_DEGRADATION,
                    "description": "Performance degradation detected",
                    "fix": "Reduce rendering quality or object count",
                },
            ],
            "webxr": [
                {
                    "pattern": r"XRSession.*ended|immersive.*session.*lost",
                    "type": ARErrorType.TRACKING_LOST,
                    "description": "WebXR session ended unexpectedly",
                    "fix": "Request new immersive session",
                },
                {
                    "pattern": r"WebGL.*context.*lost|GPU.*process.*crashed",
                    "type": ARErrorType.RENDERING_ERROR,
                    "description": "WebGL context lost",
                    "fix": "Restore WebGL context and reload assets",
                },
                {
                    "pattern": r"Permission.*denied|getUserMedia.*failed",
                    "type": ARErrorType.CALIBRATION_ERROR,
                    "description": "Camera permission denied",
                    "fix": "Request camera permissions",
                },
            ],
        }

    def _load_healing_strategies(self) -> Dict[ARErrorType, List[Dict]]:
        """Load healing strategies for different error types"""
        return {
            ARErrorType.TRACKING_LOST: [
                {
                    "name": "tracking_recovery_ui",
                    "description": "Show UI guidance for tracking recovery",
                    "applicable_platforms": ["all"],
                    "implementation": "Display visual hints and instructions",
                },
                {
                    "name": "relocalization",
                    "description": "Attempt to relocalize using saved map",
                    "applicable_platforms": ["arcore", "arkit", "hololens"],
                    "implementation": "Use persistent cloud anchors or saved world map",
                },
                {
                    "name": "fallback_tracking",
                    "description": "Switch to simpler tracking mode",
                    "applicable_platforms": ["all"],
                    "implementation": "Use rotation-only or 3DOF tracking",
                },
            ],
            ARErrorType.PERFORMANCE_DEGRADATION: [
                {
                    "name": "dynamic_lod",
                    "description": "Adjust level of detail dynamically",
                    "applicable_platforms": ["all"],
                    "implementation": "Reduce polygon count based on performance",
                },
                {
                    "name": "render_scale_adjustment",
                    "description": "Dynamically adjust render resolution",
                    "applicable_platforms": ["all"],
                    "implementation": "Lower resolution when performance drops",
                },
                {
                    "name": "occlusion_culling",
                    "description": "Aggressive occlusion culling",
                    "applicable_platforms": ["all"],
                    "implementation": "Hide objects not in view frustum",
                },
            ],
            ARErrorType.MOTION_SICKNESS_RISK: [
                {
                    "name": "comfort_mode",
                    "description": "Enable comfort settings",
                    "applicable_platforms": ["all"],
                    "implementation": "Add vignetting, reduce FOV, stabilize horizon",
                },
                {
                    "name": "motion_smoothing",
                    "description": "Smooth sudden movements",
                    "applicable_platforms": ["all"],
                    "implementation": "Apply motion damping and prediction",
                },
                {
                    "name": "teleportation",
                    "description": "Replace smooth locomotion with teleportation",
                    "applicable_platforms": ["vr"],
                    "implementation": "Use blink or fade transitions",
                },
            ],
            ARErrorType.THERMAL_THROTTLING: [
                {
                    "name": "thermal_management",
                    "description": "Reduce workload to manage temperature",
                    "applicable_platforms": ["mobile"],
                    "implementation": "Lower quality settings when device is hot",
                },
                {
                    "name": "frame_rate_limiting",
                    "description": "Cap frame rate to reduce heat",
                    "applicable_platforms": ["all"],
                    "implementation": "Limit to 30fps when thermal throttling detected",
                },
            ],
            ARErrorType.ANCHOR_DRIFT: [
                {
                    "name": "anchor_update",
                    "description": "Periodically update anchor positions",
                    "applicable_platforms": ["all"],
                    "implementation": "Recalculate anchor positions based on tracking",
                },
                {
                    "name": "multi_anchor_fusion",
                    "description": "Use multiple anchors for stability",
                    "applicable_platforms": ["all"],
                    "implementation": "Average positions from multiple nearby anchors",
                },
            ],
        }

    def _initialize_platform_detectors(self) -> Dict[ARPlatform, Dict]:
        """Initialize platform-specific detectors"""
        return {
            ARPlatform.ARCORE: {
                "imports": ["com.google.ar.core", "Google.XR.ARCoreExtensions"],
                "classes": ["Session", "Frame", "Anchor", "Trackable"],
                "file_extensions": [".java", ".kt", ".cs"],
                "keywords": ["ArCore", "CloudAnchor", "AugmentedImage"],
            },
            ARPlatform.ARKIT: {
                "imports": ["ARKit", "RealityKit"],
                "classes": ["ARSession", "ARWorldTrackingConfiguration", "ARAnchor"],
                "file_extensions": [".swift", ".m"],
                "keywords": ["ARKit", "ARWorldMap", "ARFaceTracking"],
            },
            ARPlatform.UNITY_AR: {
                "imports": [
                    "UnityEngine.XR.ARFoundation",
                    "Unity.XR.ARSubsystems",
                    "UnityEngine",
                ],
                "classes": [
                    "ARSession",
                    "ARSessionOrigin",
                    "ARRaycastManager",
                    "ARRaycastHit",
                ],
                "file_extensions": [".cs"],
                "keywords": [
                    "ARFoundation",
                    "XROrigin",
                    "ARPlaneManager",
                    "raycastManager",
                    "planeManager",
                    "trackables",
                    "MonoBehaviour",
                ],
            },
            ARPlatform.WEBXR: {
                "imports": ["webxr", "three.js", "aframe"],
                "keywords": [
                    "navigator.xr",
                    "XRSession",
                    "requestSession",
                    "immersive-ar",
                ],
                "file_extensions": [".js", ".ts"],
                "apis": ["requestAnimationFrame", "XRFrame", "XRReferenceSpace"],
            },
            ARPlatform.HOLOLENS: {
                "imports": ["Windows.UI.Input.Spatial", "Microsoft.MixedReality"],
                "classes": ["SpatialInteractionManager", "HolographicFrame"],
                "keywords": ["HoloLens", "MixedReality", "SpatialMapping"],
            },
        }

    def _initialize_performance_thresholds(self) -> Dict[str, Dict[str, Any]]:
        """Initialize performance thresholds for different platforms"""
        return {
            "mobile_ar": {
                "min_fps": 30,
                "target_fps": 60,
                "max_frame_time_ms": 33.3,
                "max_cpu_usage": 0.80,
                "max_gpu_usage": 0.85,
                "max_memory_usage": 0.75,
                "max_thermal_state_numeric": 1.0,  # 0=idle, 1=nominal, 2=serious, 3=critical
                "min_tracking_quality": 0.7,
            },
            "standalone_vr": {
                "min_fps": 72,
                "target_fps": 90,
                "max_frame_time_ms": 11.1,
                "max_motion_to_photon_ms": 20,
                "max_cpu_usage": 0.85,
                "max_gpu_usage": 0.90,
                "max_memory_usage": 0.80,
            },
            "pc_vr": {
                "min_fps": 90,
                "target_fps": 120,
                "max_frame_time_ms": 8.3,
                "max_motion_to_photon_ms": 15,
                "max_cpu_usage": 0.90,
                "max_gpu_usage": 0.95,
            },
        }

    def _initialize_comfort_parameters(self) -> Dict[str, Any]:
        """Initialize comfort parameters to prevent motion sickness"""
        return {
            "max_angular_velocity": 180,  # degrees per second
            "max_linear_acceleration": 5.0,  # m/s²
            "min_fov_comfort": 60,  # degrees
            "vignette_intensity": 0.3,
            "smooth_locomotion_speed": 3.0,  # m/s
            "teleport_fade_duration": 0.3,  # seconds
            "horizon_lock_strength": 0.8,
            "motion_blur_threshold": 90,  # degrees/second
        }

    def detect_platform(self, code_content: str, file_path: str) -> ARPlatform:
        """Detect which AR platform is being used"""
        platform_scores = {}

        for platform, detector in self.platform_detectors.items():
            score = 0

            # Check imports (highest weight)
            for import_pattern in detector.get("imports", []):
                if import_pattern in code_content:
                    score += 5

            # Check class names
            for class_name in detector.get("classes", []):
                if re.search(r"\b" + re.escape(class_name) + r"\b", code_content):
                    score += 2

            # Check keywords
            for keyword in detector.get("keywords", []):
                if keyword in code_content:
                    score += 1

            # Check file extension
            if file_path and any(
                file_path.endswith(ext) for ext in detector.get("file_extensions", [])
            ):
                score += 2

            platform_scores[platform] = score

        # Return the platform with highest score if it's above threshold
        best_platform = max(platform_scores, key=lambda k: platform_scores[k])
        if platform_scores[best_platform] >= 4:
            return best_platform

        return ARPlatform.UNKNOWN

    def analyze_ar_error(
        self,
        error_message: str,
        code_content: str,
        file_path: str,
        performance_metrics: Optional[ARPerformanceMetrics] = None,
    ) -> Optional[ARError]:
        """Analyze error and determine AR-specific issues"""
        platform = self.detect_platform(code_content, file_path)

        # Check performance metrics first if provided
        if performance_metrics:
            perf_error = self._check_performance_issues(
                platform if platform != ARPlatform.UNKNOWN else ARPlatform.ARCORE,
                performance_metrics,
            )
            if perf_error:
                return perf_error

        if platform == ARPlatform.UNKNOWN:
            return self._check_generic_ar_errors(error_message, performance_metrics)

        # Check platform-specific patterns
        platform_patterns = self.error_patterns.get(platform.value, [])

        for pattern_info in platform_patterns:
            if re.search(pattern_info["pattern"], error_message, re.IGNORECASE):
                return ARError(
                    error_type=pattern_info["type"],
                    platform=platform,
                    description=pattern_info["description"],
                    suggested_fix=pattern_info.get("fix"),
                    confidence=0.9,
                    timestamp=datetime.now(),
                )

        return self._check_generic_ar_errors(error_message, performance_metrics)

    def _check_generic_ar_errors(
        self,
        error_message: str,
        performance_metrics: Optional[ARPerformanceMetrics] = None,
    ) -> Optional[ARError]:
        """Check for generic AR errors"""
        generic_patterns = {
            r"tracking.*lost|lost.*tracking|tracking.*failed": ARErrorType.TRACKING_LOST,
            r"render.*error|draw.*call.*failed|shader.*error": ARErrorType.RENDERING_ERROR,
            r"calibrat.*fail|camera.*calibrat": ARErrorType.CALIBRATION_ERROR,
            r"anchor.*drift|anchor.*moved|position.*drift": ARErrorType.ANCHOR_DRIFT,
            r"plane.*not.*found|surface.*detection.*fail|no.*planes.*detected": ARErrorType.PLANE_DETECTION_FAILURE,
            r"lighting.*estimat|illumination.*fail": ARErrorType.LIGHTING_ESTIMATION_ERROR,
            r"occlusion.*error|depth.*fail": ARErrorType.OCCLUSION_ERROR,
            r"motion.*sick|nausea|dizzy|comfort": ARErrorType.MOTION_SICKNESS_RISK,
            r"memory.*pressure|low.*memory": ARErrorType.MEMORY_PRESSURE,
            r"thermal.*throttl|overheat|temperature.*high": ARErrorType.THERMAL_THROTTLING,
        }

        for pattern, error_type in generic_patterns.items():
            if re.search(pattern, error_message, re.IGNORECASE):
                return ARError(
                    error_type=error_type,
                    platform=ARPlatform.UNKNOWN,
                    description=f"Generic {error_type.value} detected",
                    confidence=0.7,
                    timestamp=datetime.now(),
                )

        return None

    def _check_performance_issues(
        self, platform: ARPlatform, metrics: ARPerformanceMetrics
    ) -> Optional[ARError]:
        """Check for performance-related issues"""
        # Determine threshold category
        if platform in [ARPlatform.ARCORE, ARPlatform.ARKIT, ARPlatform.WEBXR]:
            thresholds = self.performance_thresholds["mobile_ar"]
        elif platform in [ARPlatform.OCULUS, ARPlatform.MAGIC_LEAP]:
            thresholds = self.performance_thresholds["standalone_vr"]
        else:
            thresholds = self.performance_thresholds["pc_vr"]

        # Check FPS
        if metrics.fps < thresholds["min_fps"]:
            return ARError(
                error_type=ARErrorType.PERFORMANCE_DEGRADATION,
                platform=platform,
                description=f"Low frame rate: {metrics.fps} FPS",
                performance_metrics={"fps": metrics.fps},
                suggested_fix="Reduce rendering quality or optimize scene",
                severity="high",
                confidence=0.95,
                timestamp=datetime.now(),
            )

        # Check thermal state
        if metrics.thermal_state in ["serious", "critical"]:
            return ARError(
                error_type=ARErrorType.THERMAL_THROTTLING,
                platform=platform,
                description=f"Device thermal throttling: {metrics.thermal_state}",
                performance_metrics={},
                suggested_fix="Reduce workload to prevent overheating",
                severity="critical",
                confidence=0.9,
                timestamp=datetime.now(),
            )

        # Check motion-to-photon latency for VR
        if (
            metrics.motion_to_photon_latency_ms
            and metrics.motion_to_photon_latency_ms
            > thresholds.get("max_motion_to_photon_ms", 20)
        ):
            return ARError(
                error_type=ARErrorType.MOTION_SICKNESS_RISK,
                platform=platform,
                description="High motion-to-photon latency",
                performance_metrics={"latency_ms": metrics.motion_to_photon_latency_ms},
                suggested_fix="Optimize render pipeline to reduce latency",
                severity="high",
                confidence=0.85,
                timestamp=datetime.now(),
            )

        return None

    def suggest_healing(self, ar_error: ARError) -> List[Dict[str, Any]]:
        """Suggest healing strategies for AR error"""
        strategies = self.healing_strategies.get(ar_error.error_type, [])

        applicable_strategies = []
        for strategy in strategies:
            platforms = strategy["applicable_platforms"]
            if ar_error.platform.value in platforms or "all" in platforms:
                applicable_strategies.append(strategy)

        return applicable_strategies

    def generate_healing_code(
        self, ar_error: ARError, strategy: Dict[str, Any]
    ) -> Optional[str]:
        """Generate code for implementing healing strategy"""
        if ar_error.platform == ARPlatform.ARCORE:
            return self._generate_arcore_healing(ar_error, strategy)
        elif ar_error.platform == ARPlatform.ARKIT:
            return self._generate_arkit_healing(ar_error, strategy)
        elif ar_error.platform == ARPlatform.UNITY_AR:
            return self._generate_unity_healing(ar_error, strategy)
        elif ar_error.platform == ARPlatform.WEBXR:
            return self._generate_webxr_healing(ar_error, strategy)

        return self._generate_generic_ar_healing(ar_error, strategy)

    def _generate_arcore_healing(
        self, error: ARError, strategy: Dict[str, Any]
    ) -> Optional[str]:
        """Generate ARCore-specific healing code"""
        healing_templates = {
            "tracking_recovery_ui": """
// ARCore tracking recovery UI
private void showTrackingRecoveryUI(TrackingFailureReason reason) {
    runOnUiThread(() -> {
        trackingOverlay.setVisibility(View.VISIBLE);
        
        switch (reason) {
            case INSUFFICIENT_LIGHT:
                instructionText.setText("Move to a brighter area");
                instructionIcon.setImageResource(R.drawable.ic_brightness);
                break;
            case INSUFFICIENT_FEATURES:
                instructionText.setText("Point camera at textured surfaces");
                instructionIcon.setImageResource(R.drawable.ic_texture);
                break;
            case EXCESSIVE_MOTION:
                instructionText.setText("Move device slowly");
                instructionIcon.setImageResource(R.drawable.ic_slow_motion);
                break;
        }
        
        // Animate guidance
        animateTrackingHint();
    });
}

private void onTrackingRecovered() {
    runOnUiThread(() -> {
        trackingOverlay.animate()
            .alpha(0f)
            .setDuration(300)
            .withEndAction(() -> trackingOverlay.setVisibility(View.GONE));
    });
}

// Monitor tracking state
private void updateTrackingState(Frame frame) {
    Camera camera = frame.getCamera();
    TrackingState trackingState = camera.getTrackingState();
    
    if (trackingState == TrackingState.PAUSED) {
        TrackingFailureReason reason = camera.getTrackingFailureReason();
        showTrackingRecoveryUI(reason);
    } else if (trackingState == TrackingState.TRACKING) {
        onTrackingRecovered();
    }
}
""",
            "relocalization": """
// ARCore relocalization using Cloud Anchors
private void attemptRelocalization() {
    if (savedCloudAnchorId != null) {
        showProgressDialog("Relocalizing...");
        
        Anchor.CloudAnchorState cloudState = arSession.resolveCloudAnchor(savedCloudAnchorId);
        
        new Handler().postDelayed(new Runnable() {
            @Override
            public void run() {
                checkCloudAnchorState(cloudState);
            }
        }, 500);
    } else {
        // Fallback to local saved anchors
        restoreLocalAnchors();
    }
}

private void restoreLocalAnchors() {
    SharedPreferences prefs = getSharedPreferences("ar_anchors", MODE_PRIVATE);
    Set<String> anchorData = prefs.getStringSet("local_anchors", new HashSet<>());
    
    for (String data : anchorData) {
        try {
            // Parse saved pose data
            String[] parts = data.split(",");
            float[] translation = new float[]{
                Float.parseFloat(parts[0]),
                Float.parseFloat(parts[1]),
                Float.parseFloat(parts[2])
            };
            float[] rotation = new float[]{
                Float.parseFloat(parts[3]),
                Float.parseFloat(parts[4]),
                Float.parseFloat(parts[5]),
                Float.parseFloat(parts[6])
            };
            
            // Create anchor at saved pose
            Pose pose = new Pose(translation, rotation);
            Anchor anchor = arSession.createAnchor(pose);
            savedAnchors.add(anchor);
            
        } catch (Exception e) {
            Log.e(TAG, "Failed to restore anchor", e);
        }
    }
}
""",
            "dynamic_lod": """
// Dynamic Level of Detail for ARCore
public class DynamicLODManager {
    private static final int HIGH_POLY_THRESHOLD = 60;
    private static final int MEDIUM_POLY_THRESHOLD = 45;
    private static final int LOW_POLY_THRESHOLD = 30;
    
    private ModelRenderable highPolyModel;
    private ModelRenderable mediumPolyModel;
    private ModelRenderable lowPolyModel;
    
    public void updateLOD(Node node, float currentFPS) {
        ModelRenderable currentModel = (ModelRenderable) node.getRenderable();
        ModelRenderable targetModel = null;
        
        if (currentFPS >= HIGH_POLY_THRESHOLD) {
            targetModel = highPolyModel;
        } else if (currentFPS >= MEDIUM_POLY_THRESHOLD) {
            targetModel = mediumPolyModel;
        } else {
            targetModel = lowPolyModel;
        }
        
        if (targetModel != null && targetModel != currentModel) {
            node.setRenderable(targetModel);
        }
    }
    
    public void adjustRenderingQuality(float currentFPS) {
        // Adjust shadow quality
        if (currentFPS < LOW_POLY_THRESHOLD) {
            arSceneView.getScene().getSunlight().setShadowCastingEnabled(false);
        }
        
        // Adjust texture resolution
        int textureScale = currentFPS < MEDIUM_POLY_THRESHOLD ? 2 : 1;
        updateTextureScale(textureScale);
    }
}
""",
        }

        strategy_name = strategy.get("name")
        if strategy_name is None:
            return None
        return healing_templates.get(strategy_name)

    def _generate_arkit_healing(
        self, error: ARError, strategy: Dict[str, Any]
    ) -> Optional[str]:
        """Generate ARKit-specific healing code"""
        healing_templates = {
            "tracking_recovery_ui": """
// ARKit tracking recovery UI
class TrackingStateManager {
    weak var coachingOverlay: ARCoachingOverlayView?
    weak var session: ARSession?
    
    func setupCoachingOverlay(_ overlay: ARCoachingOverlayView) {
        coachingOverlay = overlay
        coachingOverlay?.session = session
        coachingOverlay?.goal = .tracking
        coachingOverlay?.activatesAutomatically = true
    }
    
    func handleTrackingState(_ camera: ARCamera) {
        switch camera.trackingState {
        case .normal:
            hideTrackingUI()
            
        case .notAvailable:
            showTrackingUI(message: "Tracking unavailable")
            
        case .limited(let reason):
            handleLimitedTracking(reason)
        }
    }
    
    private func handleLimitedTracking(_ reason: ARCamera.TrackingState.Reason) {
        var message = ""
        var icon = UIImage()
        
        switch reason {
        case .initializing:
            message = "Initializing AR..."
            icon = UIImage(systemName: "hourglass")!
            
        case .excessiveMotion:
            message = "Slow down your movement"
            icon = UIImage(systemName: "hare")!
            
        case .insufficientFeatures:
            message = "Point at a textured surface"
            icon = UIImage(systemName: "eye.slash")!
            
        case .relocalizing:
            message = "Relocalizing..."
            icon = UIImage(systemName: "location.magnifyingglass")!
            
        @unknown default:
            message = "Tracking limited"
        }
        
        showTrackingUI(message: message, icon: icon)
    }
}
""",
            "relocalization": """
// ARKit relocalization using world map
class WorldMapManager {
    private var worldMapData: Data?
    
    func saveWorldMap() {
        session?.getCurrentWorldMap { worldMap, error in
            guard let map = worldMap else {
                print("Error saving world map: \\(error?.localizedDescription ?? "Unknown")")
                return
            }
            
            do {
                self.worldMapData = try NSKeyedArchiver.archivedData(
                    withRootObject: map,
                    requiringSecureCoding: true
                )
                
                // Save to persistent storage
                UserDefaults.standard.set(self.worldMapData, forKey: "ARWorldMap")
                
                // Save anchor identifiers
                let anchorIDs = map.anchors.compactMap { $0.identifier }
                UserDefaults.standard.set(anchorIDs, forKey: "AnchorIdentifiers")
                
            } catch {
                print("Error archiving world map: \\(error)")
            }
        }
    }
    
    func loadWorldMap() {
        guard let data = UserDefaults.standard.data(forKey: "ARWorldMap"),
              let worldMap = try? NSKeyedUnarchiver.unarchivedObject(
                ofClass: ARWorldMap.self,
                from: data
              ) else {
            print("No saved world map found")
            return
        }
        
        let configuration = ARWorldTrackingConfiguration()
        configuration.initialWorldMap = worldMap
        configuration.planeDetection = [.horizontal, .vertical]
        
        // Relocalization configuration
        if #available(iOS 14.0, *) {
            configuration.sceneReconstruction = .meshWithClassification
        }
        
        session?.run(configuration, options: [.resetTracking, .removeExistingAnchors])
    }
}
""",
            "fallback_tracking": """
// ARKit fallback tracking modes
class FallbackTrackingManager {
    enum TrackingMode {
        case worldTracking
        case bodyTracking
        case imageTracking
        case orientationOnly
    }
    
    private var currentMode: TrackingMode = .worldTracking
    
    func switchToFallbackTracking() {
        switch currentMode {
        case .worldTracking:
            // Try body tracking if available
            if ARBodyTrackingConfiguration.isSupported {
                switchToBodyTracking()
            } else {
                switchToOrientationTracking()
            }
            
        case .bodyTracking:
            switchToImageTracking()
            
        case .imageTracking:
            switchToOrientationTracking()
            
        case .orientationOnly:
            print("Already in minimal tracking mode")
        }
    }
    
    private func switchToOrientationTracking() {
        let configuration = AROrientationTrackingConfiguration()
        session?.run(configuration, options: [.resetTracking])
        currentMode = .orientationOnly
        
        showNotification("Switched to rotation-only tracking")
    }
    
    private func switchToImageTracking() {
        guard let referenceImages = ARReferenceImage.referenceImages(
            inGroupNamed: "AR Resources",
            bundle: nil
        ) else { return }
        
        let configuration = ARImageTrackingConfiguration()
        configuration.trackingImages = referenceImages
        configuration.maximumNumberOfTrackedImages = 1
        
        session?.run(configuration, options: [.resetTracking])
        currentMode = .imageTracking
    }
}
""",
        }

        strategy_name = strategy.get("name")
        if strategy_name is None:
            return None
        return healing_templates.get(strategy_name)

    def _generate_unity_healing(
        self, error: ARError, strategy: Dict[str, Any]
    ) -> Optional[str]:
        """Generate Unity AR-specific healing code"""
        healing_templates = {
            "tracking_recovery_ui": """
// Unity AR Foundation tracking recovery UI
using UnityEngine;
using UnityEngine.XR.ARFoundation;
using UnityEngine.UI;

public class TrackingRecoveryUI : MonoBehaviour
{
    [SerializeField] private GameObject trackingLostPanel;
    [SerializeField] private Text instructionText;
    [SerializeField] private Image visualGuide;
    [SerializeField] private ARSession arSession;
    
    private ARSessionState previousState;
    
    void Start()
    {
        ARSession.stateChanged += OnSessionStateChanged;
    }
    
    void OnSessionStateChanged(ARSessionStateChangedEventArgs args)
    {
        if (args.state != previousState)
        {
            previousState = args.state;
            UpdateTrackingUI(args.state);
        }
    }
    
    void UpdateTrackingUI(ARSessionState state)
    {
        switch (state)
        {
            case ARSessionState.None:
            case ARSessionState.Unsupported:
                ShowError("AR not supported on this device");
                break;
                
            case ARSessionState.SessionInitializing:
                ShowMessage("Initializing AR...");
                break;
                
            case ARSessionState.SessionTracking:
                HideTrackingUI();
                break;
                
            case ARSessionState.NeedsInstall:
                ShowError("Please install AR services");
                break;
                
            default:
                ShowMessage("Tracking lost. Move device slowly");
                break;
        }
    }
    
    void ShowMessage(string message)
    {
        trackingLostPanel.SetActive(true);
        instructionText.text = message;
        
        // Animate visual guide
        visualGuide.GetComponent<Animator>().SetTrigger("ShowHint");
    }
    
    void HideTrackingUI()
    {
        trackingLostPanel.SetActive(false);
    }
}
""",
            "dynamic_lod": """
// Unity AR dynamic LOD system
using UnityEngine;
using UnityEngine.Rendering;
using Unity.Profiling;

public class ARDynamicLOD : MonoBehaviour
{
    [System.Serializable]
    public class LODModel
    {
        public GameObject model;
        public int maxVertices;
        public float distanceThreshold;
    }
    
    [SerializeField] private LODModel[] lodModels;
    [SerializeField] private Camera arCamera;
    
    private ProfilerRecorder frameTimeRecorder;
    private float targetFrameTime = 16.67f; // 60 FPS
    private int currentLODIndex = 0;
    
    void Start()
    {
        frameTimeRecorder = ProfilerRecorder.StartNew(
            ProfilerCategory.Internal,
            "Main Thread",
            15
        );
    }
    
    void Update()
    {
        UpdateLODBasedOnPerformance();
        UpdateLODBasedOnDistance();
    }
    
    void UpdateLODBasedOnPerformance()
    {
        if (!frameTimeRecorder.Valid)
            return;
            
        float avgFrameTime = frameTimeRecorder.LastValue / 1000000f; // Convert to ms
        
        if (avgFrameTime > targetFrameTime * 1.2f) // 20% over budget
        {
            // Decrease quality
            SwitchToLowerLOD();
        }
        else if (avgFrameTime < targetFrameTime * 0.8f) // 20% under budget
        {
            // Increase quality
            SwitchToHigherLOD();
        }
    }
    
    void UpdateLODBasedOnDistance()
    {
        float distance = Vector3.Distance(
            transform.position,
            arCamera.transform.position
        );
        
        for (int i = 0; i < lodModels.Length; i++)
        {
            if (distance <= lodModels[i].distanceThreshold)
            {
                SetLOD(i);
                break;
            }
        }
    }
    
    void SetLOD(int index)
    {
        if (index == currentLODIndex)
            return;
            
        // Disable current LOD
        if (currentLODIndex < lodModels.Length)
            lodModels[currentLODIndex].model.SetActive(false);
        
        // Enable new LOD
        currentLODIndex = Mathf.Clamp(index, 0, lodModels.Length - 1);
        lodModels[currentLODIndex].model.SetActive(true);
    }
    
    void SwitchToLowerLOD()
    {
        SetLOD(currentLODIndex + 1);
    }
    
    void SwitchToHigherLOD()
    {
        SetLOD(currentLODIndex - 1);
    }
}
""",
            "render_scale_adjustment": """
// Unity AR render scale adjustment
using UnityEngine;
using UnityEngine.XR.ARFoundation;
using UnityEngine.Rendering.Universal;

public class AdaptiveRenderScale : MonoBehaviour
{
    [SerializeField] private UniversalRenderPipelineAsset urpAsset;
    [SerializeField] private float minRenderScale = 0.5f;
    [SerializeField] private float maxRenderScale = 1.0f;
    [SerializeField] private float adjustmentSpeed = 0.1f;
    
    private float currentRenderScale = 1.0f;
    private float targetFrameTime = 16.67f; // 60 FPS
    private float[] frameTimeHistory = new float[30];
    private int frameTimeIndex = 0;
    
    void Update()
    {
        // Record frame time
        frameTimeHistory[frameTimeIndex] = Time.deltaTime * 1000f;
        frameTimeIndex = (frameTimeIndex + 1) % frameTimeHistory.Length;
        
        // Calculate average frame time
        float avgFrameTime = 0f;
        foreach (float time in frameTimeHistory)
            avgFrameTime += time;
        avgFrameTime /= frameTimeHistory.Length;
        
        // Adjust render scale based on performance
        if (avgFrameTime > targetFrameTime * 1.1f)
        {
            // Reduce render scale
            currentRenderScale = Mathf.Max(
                currentRenderScale - adjustmentSpeed * Time.deltaTime,
                minRenderScale
            );
        }
        else if (avgFrameTime < targetFrameTime * 0.9f)
        {
            // Increase render scale
            currentRenderScale = Mathf.Min(
                currentRenderScale + adjustmentSpeed * Time.deltaTime,
                maxRenderScale
            );
        }
        
        // Apply render scale
        urpAsset.renderScale = currentRenderScale;
        
        // Update UI if needed
        UpdatePerformanceUI(avgFrameTime, currentRenderScale);
    }
    
    void UpdatePerformanceUI(float frameTime, float renderScale)
    {
        // Update performance indicators
        if (performanceText != null)
        {
            performanceText.text = $"FPS: {1000f/frameTime:F1} Scale: {renderScale:F2}";
        }
    }
}
""",
        }

        strategy_name = strategy.get("name")
        if strategy_name is None:
            return None
        return healing_templates.get(strategy_name)

    def _generate_webxr_healing(
        self, error: ARError, strategy: Dict[str, Any]
    ) -> Optional[str]:
        """Generate WebXR-specific healing code"""
        healing_templates = {
            "tracking_recovery_ui": """
// WebXR tracking recovery UI
class XRTrackingRecovery {
    constructor(scene, camera) {
        this.scene = scene;
        this.camera = camera;
        this.overlay = this.createTrackingOverlay();
        this.isTracking = true;
    }
    
    createTrackingOverlay() {
        const overlay = document.createElement('div');
        overlay.id = 'xr-tracking-overlay';
        overlay.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 20px;
            border-radius: 10px;
            display: none;
            z-index: 9999;
        `;
        
        overlay.innerHTML = `
            <div class="tracking-message">
                <img src="tracking-hint.svg" width="100" />
                <p id="tracking-instruction">Move device slowly</p>
            </div>
        `;
        
        document.body.appendChild(overlay);
        return overlay;
    }
    
    updateTrackingState(frame, referenceSpace) {
        const pose = frame.getViewerPose(referenceSpace);
        
        if (!pose) {
            this.showTrackingLost('Unable to determine position');
            return;
        }
        
        // Check pose confidence if available
        if (pose.emulatedPosition) {
            this.showTrackingLost('Limited tracking - move to textured area');
        } else if (this.isTracking === false) {
            this.hideTrackingOverlay();
        }
    }
    
    showTrackingLost(message) {
        this.isTracking = false;
        this.overlay.style.display = 'block';
        document.getElementById('tracking-instruction').textContent = message;
        
        // Add visual hints to the scene
        this.addVisualHints();
    }
    
    hideTrackingOverlay() {
        this.isTracking = true;
        this.overlay.style.display = 'none';
        this.removeVisualHints();
    }
    
    addVisualHints() {
        // Add grid or other visual aids to help tracking
        const gridHelper = new THREE.GridHelper(10, 10);
        gridHelper.name = 'trackingGrid';
        this.scene.add(gridHelper);
    }
    
    removeVisualHints() {
        const grid = this.scene.getObjectByName('trackingGrid');
        if (grid) {
            this.scene.remove(grid);
        }
    }
}
""",
            "fallback_tracking": """
// WebXR fallback tracking modes
class XRFallbackTracking {
    constructor(renderer) {
        this.renderer = renderer;
        this.currentMode = 'immersive-ar';
        this.session = null;
    }
    
    async initializeXR() {
        if (!navigator.xr) {
            console.error('WebXR not supported');
            return;
        }
        
        // Try different modes in order of preference
        const modes = [
            'immersive-ar',
            'immersive-vr',
            'inline'
        ];
        
        for (const mode of modes) {
            if (await this.trySessionMode(mode)) {
                this.currentMode = mode;
                break;
            }
        }
    }
    
    async trySessionMode(mode) {
        try {
            const supported = await navigator.xr.isSessionSupported(mode);
            if (!supported) return false;
            
            const sessionInit = {
                requiredFeatures: this.getRequiredFeatures(mode),
                optionalFeatures: this.getOptionalFeatures(mode)
            };
            
            this.session = await navigator.xr.requestSession(mode, sessionInit);
            await this.renderer.xr.setSession(this.session);
            
            this.setupSessionHandlers();
            return true;
            
        } catch (error) {
            console.warn(`Failed to start ${mode} session:`, error);
            return false;
        }
    }
    
    getRequiredFeatures(mode) {
        switch (mode) {
            case 'immersive-ar':
                return ['local-floor'];
            case 'immersive-vr':
                return ['local-floor'];
            default:
                return [];
        }
    }
    
    getOptionalFeatures(mode) {
        return [
            'bounded-floor',
            'hand-tracking',
            'hit-test',
            'dom-overlay',
            'anchors',
            'depth-sensing',
            'light-estimation'
        ];
    }
    
    setupSessionHandlers() {
        this.session.addEventListener('end', () => {
            console.log('XR session ended');
            this.handleSessionEnd();
        });
        
        this.session.addEventListener('visibilitychange', (event) => {
            console.log('Visibility changed:', event.session.visibilityState);
            this.handleVisibilityChange(event.session.visibilityState);
        });
    }
    
    async handleSessionEnd() {
        // Try to restart with fallback mode
        const fallbackModes = {
            'immersive-ar': 'immersive-vr',
            'immersive-vr': 'inline',
            'inline': null
        };
        
        const nextMode = fallbackModes[this.currentMode];
        if (nextMode) {
            console.log(`Falling back to ${nextMode} mode`);
            await this.trySessionMode(nextMode);
        }
    }
}
""",
            "comfort_mode": """
// WebXR comfort mode for motion sickness prevention
class XRComfortMode {
    constructor(scene, camera) {
        this.scene = scene;
        this.camera = camera;
        this.settings = {
            vignette: true,
            vignetteIntensity: 0.4,
            tunnelVision: false,
            snapRotation: true,
            snapAngle: 30,
            teleportation: true,
            reducedMotion: false
        };
        
        this.vignetteEffect = this.createVignette();
        this.motionData = {
            lastPosition: new THREE.Vector3(),
            lastRotation: new THREE.Quaternion(),
            velocity: new THREE.Vector3(),
            angularVelocity: 0
        };
    }
    
    createVignette() {
        const geometry = new THREE.PlaneGeometry(2, 2);
        const material = new THREE.ShaderMaterial({
            uniforms: {
                intensity: { value: 0.0 },
                color: { value: new THREE.Color(0x000000) }
            },
            vertexShader: `
                varying vec2 vUv;
                void main() {
                    vUv = uv;
                    gl_Position = vec4(position.xy, 0.0, 1.0);
                }
            `,
            fragmentShader: `
                uniform float intensity;
                uniform vec3 color;
                varying vec2 vUv;
                
                void main() {
                    vec2 center = vec2(0.5, 0.5);
                    float dist = distance(vUv, center);
                    float vignette = smoothstep(0.3, 0.7, dist) * intensity;
                    gl_FragColor = vec4(color, vignette);
                }
            `,
            transparent: true,
            depthTest: false,
            depthWrite: false
        });
        
        const mesh = new THREE.Mesh(geometry, material);
        mesh.frustumCulled = false;
        mesh.renderOrder = 9999;
        
        // Add to camera
        const vignetteCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
        const vignetteScene = new THREE.Scene();
        vignetteScene.add(mesh);
        
        return { mesh, material, camera: vignetteCamera, scene: vignetteScene };
    }
    
    updateComfort(deltaTime) {
        // Calculate motion
        const currentPosition = this.camera.position.clone();
        const currentRotation = this.camera.quaternion.clone();
        
        // Linear velocity
        this.motionData.velocity.subVectors(
            currentPosition,
            this.motionData.lastPosition
        ).divideScalar(deltaTime);
        
        // Angular velocity
        const angleDiff = currentRotation.angleTo(this.motionData.lastRotation);
        this.motionData.angularVelocity = (angleDiff * 180 / Math.PI) / deltaTime;
        
        // Update vignette based on motion
        if (this.settings.vignette) {
            const speed = this.motionData.velocity.length();
            const rotSpeed = this.motionData.angularVelocity;
            
            const speedThreshold = 3.0; // m/s
            const rotThreshold = 90; // degrees/s
            
            const linearIntensity = Math.min(speed / speedThreshold, 1.0);
            const rotIntensity = Math.min(rotSpeed / rotThreshold, 1.0);
            
            const targetIntensity = Math.max(linearIntensity, rotIntensity) * 
                                  this.settings.vignetteIntensity;
            
            // Smooth transition
            const currentIntensity = this.vignetteEffect.material.uniforms.intensity.value;
            this.vignetteEffect.material.uniforms.intensity.value = THREE.MathUtils.lerp(
                currentIntensity,
                targetIntensity,
                0.1
            );
        }
        
        // Store current values
        this.motionData.lastPosition.copy(currentPosition);
        this.motionData.lastRotation.copy(currentRotation);
    }
    
    enableSnapRotation() {
        let accumRotation = 0;
        
        this.camera.parent.addEventListener('rotate', (event) => {
            if (!this.settings.snapRotation) return;
            
            accumRotation += event.detail.angle;
            
            if (Math.abs(accumRotation) >= this.settings.snapAngle) {
                const snaps = Math.floor(accumRotation / this.settings.snapAngle);
                const snapRotation = snaps * this.settings.snapAngle;
                
                // Apply snap rotation
                this.camera.parent.rotateY(THREE.MathUtils.degToRad(snapRotation));
                accumRotation -= snapRotation;
                
                // Add comfort fade
                this.addComfortFade();
            }
        });
    }
    
    addComfortFade() {
        // Brief fade to black during rotation
        const fadeOverlay = document.getElementById('comfort-fade');
        fadeOverlay.style.opacity = '1';
        setTimeout(() => {
            fadeOverlay.style.opacity = '0';
        }, 100);
    }
}
""",
        }

        strategy_name = strategy.get("name")
        if strategy_name is None:
            return None
        return healing_templates.get(strategy_name)

    def _generate_generic_ar_healing(
        self, error: ARError, strategy: Dict[str, Any]
    ) -> Optional[str]:
        """Generate generic AR healing code"""
        healing_templates = {
            "motion_smoothing": """
// Generic motion smoothing for AR/VR
class MotionSmoothingFilter {
    constructor(smoothingFactor = 0.1) {
        this.smoothingFactor = smoothingFactor;
        this.positionBuffer = [];
        this.rotationBuffer = [];
        this.maxBufferSize = 10;
    }
    
    smoothPosition(newPosition) {
        this.positionBuffer.push(newPosition.clone());
        
        if (this.positionBuffer.length > this.maxBufferSize) {
            this.positionBuffer.shift();
        }
        
        // Weighted average with more weight on recent positions
        let smoothedPosition = new THREE.Vector3();
        let totalWeight = 0;
        
        for (let i = 0; i < this.positionBuffer.length; i++) {
            const weight = (i + 1) / this.positionBuffer.length;
            smoothedPosition.add(
                this.positionBuffer[i].clone().multiplyScalar(weight)
            );
            totalWeight += weight;
        }
        
        smoothedPosition.divideScalar(totalWeight);
        
        // Apply additional exponential smoothing
        if (this.lastSmoothedPosition) {
            smoothedPosition.lerp(this.lastSmoothedPosition, this.smoothingFactor);
        }
        
        this.lastSmoothedPosition = smoothedPosition.clone();
        return smoothedPosition;
    }
    
    smoothRotation(newRotation) {
        this.rotationBuffer.push(newRotation.clone());
        
        if (this.rotationBuffer.length > this.maxBufferSize) {
            this.rotationBuffer.shift();
        }
        
        // Spherical linear interpolation for rotations
        let smoothedRotation = new THREE.Quaternion();
        
        if (this.rotationBuffer.length === 1) {
            return newRotation;
        }
        
        // SLERP between consecutive rotations
        smoothedRotation.copy(this.rotationBuffer[0]);
        for (let i = 1; i < this.rotationBuffer.length; i++) {
            const t = (i + 1) / this.rotationBuffer.length;
            smoothedRotation.slerp(this.rotationBuffer[i], t);
        }
        
        return smoothedRotation;
    }
    
    predictNextPosition(deltaTime) {
        if (this.positionBuffer.length < 2) {
            return this.lastSmoothedPosition || new THREE.Vector3();
        }
        
        // Simple linear prediction
        const recent = this.positionBuffer[this.positionBuffer.length - 1];
        const previous = this.positionBuffer[this.positionBuffer.length - 2];
        const velocity = recent.clone().sub(previous);
        
        return recent.clone().add(velocity.multiplyScalar(deltaTime));
    }
}
""",
            "thermal_management": """
// Generic thermal management for AR devices
class ThermalManager {
    constructor() {
        this.qualityLevels = {
            ultra: { renderScale: 1.0, shadowQuality: 'high', effects: true },
            high: { renderScale: 0.9, shadowQuality: 'medium', effects: true },
            medium: { renderScale: 0.75, shadowQuality: 'low', effects: false },
            low: { renderScale: 0.5, shadowQuality: 'none', effects: false }
        };
        
        this.currentLevel = 'high';
        this.thermalState = 'nominal';
        this.temperatureHistory = [];
    }
    
    updateThermalState(temperature, batteryTemp) {
        this.temperatureHistory.push({
            device: temperature,
            battery: batteryTemp,
            timestamp: Date.now()
        });
        
        // Keep last 30 seconds of data
        const cutoff = Date.now() - 30000;
        this.temperatureHistory = this.temperatureHistory.filter(
            entry => entry.timestamp > cutoff
        );
        
        // Determine thermal state
        if (temperature > 45 || batteryTemp > 40) {
            this.thermalState = 'critical';
        } else if (temperature > 40 || batteryTemp > 37) {
            this.thermalState = 'serious';
        } else if (temperature > 35 || batteryTemp > 35) {
            this.thermalState = 'fair';
        } else {
            this.thermalState = 'nominal';
        }
        
        this.adjustQualityForThermal();
    }
    
    adjustQualityForThermal() {
        const oldLevel = this.currentLevel;
        
        switch (this.thermalState) {
            case 'critical':
                this.currentLevel = 'low';
                this.enableAggressiveCooling();
                break;
            case 'serious':
                this.currentLevel = 'medium';
                break;
            case 'fair':
                this.currentLevel = 'high';
                break;
            case 'nominal':
                this.currentLevel = 'ultra';
                break;
        }
        
        if (oldLevel !== this.currentLevel) {
            this.applyQualitySettings(this.qualityLevels[this.currentLevel]);
        }
    }
    
    enableAggressiveCooling() {
        // Reduce CPU/GPU load
        if (typeof requestIdleCallback !== 'undefined') {
            requestIdleCallback(() => {
                // Defer non-critical tasks
                console.log('Deferring non-critical tasks for cooling');
            });
        }
        
        // Reduce frame rate
        this.targetFrameRate = 30;
        
        // Disable non-essential features
        this.disableParticles();
        this.disablePostProcessing();
        this.reduceLightingQuality();
    }
    
    applyQualitySettings(settings) {
        // Apply render scale
        if (window.renderer) {
            window.renderer.setPixelRatio(
                window.devicePixelRatio * settings.renderScale
            );
        }
        
        // Apply shadow quality
        this.updateShadowQuality(settings.shadowQuality);
        
        // Toggle effects
        this.toggleEffects(settings.effects);
        
        console.log(`Thermal management: Switched to ${this.currentLevel} quality`);
    }
}
""",
            "anchor_update": """
// Generic anchor stabilization for AR
class AnchorStabilizer {
    constructor() {
        this.anchors = new Map();
        this.updateInterval = 1000; // ms
        this.stabilizationThreshold = 0.01; // meters
        this.lastUpdateTime = 0;
    }
    
    addAnchor(id, anchor, object3D) {
        this.anchors.set(id, {
            anchor: anchor,
            object: object3D,
            originalPose: this.getPose(anchor),
            lastPose: this.getPose(anchor),
            driftHistory: [],
            stabilized: false
        });
    }
    
    updateAnchors(currentTime) {
        if (currentTime - this.lastUpdateTime < this.updateInterval) {
            return;
        }
        
        this.lastUpdateTime = currentTime;
        
        for (const [id, anchorData] of this.anchors) {
            this.updateAnchor(id, anchorData);
        }
    }
    
    updateAnchor(id, anchorData) {
        const currentPose = this.getPose(anchorData.anchor);
        if (!currentPose) return;
        
        // Calculate drift
        const drift = this.calculateDrift(anchorData.lastPose, currentPose);
        anchorData.driftHistory.push({
            drift: drift,
            timestamp: Date.now()
        });
        
        // Keep last 10 samples
        if (anchorData.driftHistory.length > 10) {
            anchorData.driftHistory.shift();
        }
        
        // Check if stabilization needed
        if (this.needsStabilization(anchorData)) {
            this.stabilizeAnchor(anchorData);
        }
        
        anchorData.lastPose = currentPose;
    }
    
    calculateDrift(pose1, pose2) {
        const positionDrift = Math.sqrt(
            Math.pow(pose2.position.x - pose1.position.x, 2) +
            Math.pow(pose2.position.y - pose1.position.y, 2) +
            Math.pow(pose2.position.z - pose1.position.z, 2)
        );
        
        const rotationDrift = pose1.rotation.angleTo(pose2.rotation);
        
        return { position: positionDrift, rotation: rotationDrift };
    }
    
    needsStabilization(anchorData) {
        if (anchorData.driftHistory.length < 3) return false;
        
        // Check recent drift
        const recentDrifts = anchorData.driftHistory.slice(-3);
        const avgDrift = recentDrifts.reduce(
            (sum, entry) => sum + entry.drift.position, 0
        ) / recentDrifts.length;
        
        return avgDrift > this.stabilizationThreshold;
    }
    
    stabilizeAnchor(anchorData) {
        // Apply smoothing to anchor position
        const smoothedPose = this.calculateSmoothedPose(anchorData);
        
        // Update object position
        if (anchorData.object) {
            anchorData.object.position.copy(smoothedPose.position);
            anchorData.object.quaternion.copy(smoothedPose.rotation);
        }
        
        anchorData.stabilized = true;
    }
    
    calculateSmoothedPose(anchorData) {
        // Use exponential moving average
        const alpha = 0.2;
        const currentPose = this.getPose(anchorData.anchor);
        
        const smoothedPosition = anchorData.lastPose.position.clone().lerp(
            currentPose.position,
            alpha
        );
        
        const smoothedRotation = anchorData.lastPose.rotation.clone().slerp(
            currentPose.rotation,
            alpha
        );
        
        return {
            position: smoothedPosition,
            rotation: smoothedRotation
        };
    }
    
    getPose(anchor) {
        // Platform-specific pose extraction
        if (anchor.pose) {
            return {
                position: new THREE.Vector3().fromArray(anchor.pose.position),
                rotation: new THREE.Quaternion().fromArray(anchor.pose.orientation)
            };
        }
        return null;
    }
}
""",
        }

        strategy_name = strategy.get("name")
        if strategy_name is None:
            return None
        return healing_templates.get(strategy_name)

    def analyze_performance(self, metrics: ARPerformanceMetrics) -> Dict[str, Any]:
        """Analyze AR application performance"""
        analysis: Dict[str, Any] = {
            "overall_status": "healthy",
            "issues": [],
            "recommendations": [],
        }

        # Determine platform category
        if metrics.motion_to_photon_latency_ms is not None:
            category = "pc_vr"
        elif metrics.thermal_state:
            category = "mobile_ar"
        else:
            category = "standalone_vr"

        thresholds = self.performance_thresholds[category]

        # Check FPS
        if metrics.fps < thresholds["min_fps"]:
            analysis["issues"].append(f"Low FPS: {metrics.fps}")
            analysis["overall_status"] = "critical"
            analysis["recommendations"].append(
                "Reduce scene complexity or rendering quality"
            )

        # Check frame time
        if metrics.frame_time_ms > thresholds["max_frame_time_ms"]:
            analysis["issues"].append(f"High frame time: {metrics.frame_time_ms}ms")
            analysis["recommendations"].append("Optimize render pipeline")

        # Check thermal state
        if metrics.thermal_state in ["serious", "critical"]:
            analysis["issues"].append(f"Thermal throttling: {metrics.thermal_state}")
            analysis["overall_status"] = (
                "warning"
                if analysis["overall_status"] == "healthy"
                else analysis["overall_status"]
            )
            analysis["recommendations"].append(
                "Reduce quality settings to manage temperature"
            )

        # Check motion-to-photon latency
        if (
            metrics.motion_to_photon_latency_ms
            and metrics.motion_to_photon_latency_ms
            > thresholds.get("max_motion_to_photon_ms", 20)
        ):
            analysis["issues"].append(
                f"High latency: {metrics.motion_to_photon_latency_ms}ms"
            )
            analysis["recommendations"].append(
                "Optimize tracking and rendering pipeline"
            )

        return analysis

    def check_comfort_violations(self, motion_data: Dict[str, float]) -> List[str]:
        """Check for comfort violations that might cause motion sickness"""
        violations = []
        comfort = self.comfort_parameters

        if motion_data.get("angular_velocity", 0) > comfort["max_angular_velocity"]:
            violations.append("Excessive rotation speed")

        if (
            motion_data.get("linear_acceleration", 0)
            > comfort["max_linear_acceleration"]
        ):
            violations.append("High acceleration detected")

        if motion_data.get("fov", 90) < comfort["min_fov_comfort"]:
            violations.append("Field of view too narrow")

        return violations

    def generate_platform_config(
        self, platform: ARPlatform, device_capabilities: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate optimized AR configuration for platform"""
        config: Dict[str, Any] = {
            "platform": platform.value,
            "generated_at": datetime.now().isoformat(),
            "settings": {},
        }

        if platform in [ARPlatform.ARCORE, ARPlatform.ARKIT]:
            config["settings"] = {
                "plane_detection": device_capabilities.get("supports_planes", True),
                "light_estimation": device_capabilities.get("supports_lighting", True),
                "cloud_anchors": device_capabilities.get("has_internet", True),
                "image_tracking": device_capabilities.get("cpu_score", 50) > 40,
                "face_tracking": platform == ARPlatform.ARKIT
                and device_capabilities.get("has_truedepth", False),
                "max_tracked_images": (
                    4 if device_capabilities.get("ram_gb", 2) >= 4 else 1
                ),
            }

        elif platform == ARPlatform.WEBXR:
            config["settings"] = {
                "required_features": ["local-floor"],
                "optional_features": ["bounded-floor", "anchors", "hit-test"],
                "framebuffer_scale": (
                    1.0 if device_capabilities.get("gpu_tier", 1) >= 2 else 0.75
                ),
                "antialiasing": device_capabilities.get("gpu_tier", 1) >= 2,
            }

        return config
