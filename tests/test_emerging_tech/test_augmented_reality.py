"""
Test cases for augmented reality application resilience
"""

import unittest
from unittest.mock import Mock, patch
from datetime import datetime
from modules.emerging_tech.augmented_reality import (
    ARResilienceManager, ARPlatform, ARErrorType, ARError, ARPerformanceMetrics
)
from modules.analysis.plugins.ar_plugin import ARPlugin


class TestARResilienceManager(unittest.TestCase):
    """Test AR resilience management functionality"""
    
    def setUp(self):
        self.manager = ARResilienceManager()
    
    def test_platform_detection_arcore(self):
        """Test ARCore platform detection"""
        code = """
import com.google.ar.core.*;

public class ARActivity extends AppCompatActivity {
    private Session arSession;
    private Frame frame;
    
    public void onDrawFrame() {
        frame = arSession.update();
        Camera camera = frame.getCamera();
    }
}
        """
        
        platform = self.manager.detect_platform(code, "ARActivity.java")
        self.assertEqual(platform, ARPlatform.ARCORE)
    
    def test_platform_detection_arkit(self):
        """Test ARKit platform detection"""
        code = """
import ARKit
import RealityKit

class ARViewController: UIViewController {
    @IBOutlet var arView: ARView!
    
    override func viewDidLoad() {
        let configuration = ARWorldTrackingConfiguration()
        configuration.planeDetection = [.horizontal, .vertical]
        arView.session.run(configuration)
    }
}
        """
        
        platform = self.manager.detect_platform(code, "ARViewController.swift")
        self.assertEqual(platform, ARPlatform.ARKIT)
    
    def test_platform_detection_unity(self):
        """Test Unity AR platform detection"""
        code = """
using UnityEngine;
using UnityEngine.XR.ARFoundation;

public class ARController : MonoBehaviour
{
    private ARSession arSession;
    private ARRaycastManager raycastManager;
    
    void Start()
    {
        arSession = GetComponent<ARSession>();
        raycastManager = GetComponent<ARRaycastManager>();
    }
}
        """
        
        platform = self.manager.detect_platform(code, "ARController.cs")
        self.assertEqual(platform, ARPlatform.UNITY_AR)
    
    def test_platform_detection_webxr(self):
        """Test WebXR platform detection"""
        code = """
async function startAR() {
    if (!navigator.xr) {
        console.error('WebXR not supported');
        return;
    }
    
    const session = await navigator.xr.requestSession('immersive-ar', {
        requiredFeatures: ['local-floor'],
        optionalFeatures: ['dom-overlay', 'hit-test']
    });
}
        """
        
        platform = self.manager.detect_platform(code, "ar-app.js")
        self.assertEqual(platform, ARPlatform.WEBXR)
    
    def test_tracking_lost_error_detection(self):
        """Test tracking lost error detection"""
        error_msg = "TrackingState changed to PAUSED: Insufficient features"
        code = "import com.google.ar.core.*;"
        
        error = self.manager.analyze_ar_error(error_msg, code, "app.java")
        
        self.assertIsNotNone(error)
        self.assertEqual(error.error_type, ARErrorType.TRACKING_LOST)
        self.assertEqual(error.platform, ARPlatform.ARCORE)
    
    def test_performance_degradation_detection(self):
        """Test performance degradation detection"""
        metrics = ARPerformanceMetrics(
            fps=25,  # Below 30 fps threshold
            frame_time_ms=40,
            cpu_usage=0.85,
            gpu_usage=0.90,
            memory_usage=0.70,
            battery_drain=15,
            thermal_state="nominal",
            tracking_quality=0.8,
            rendering_latency_ms=25
        )
        
        error = self.manager.analyze_ar_error("", "", "", metrics)
        
        self.assertIsNotNone(error)
        self.assertEqual(error.error_type, ARErrorType.PERFORMANCE_DEGRADATION)
    
    def test_thermal_throttling_detection(self):
        """Test thermal throttling detection"""
        metrics = ARPerformanceMetrics(
            fps=45,
            frame_time_ms=22,
            cpu_usage=0.90,
            gpu_usage=0.95,
            memory_usage=0.80,
            battery_drain=20,
            thermal_state="critical",
            tracking_quality=1.0,
            rendering_latency_ms=15
        )
        
        error = self.manager.analyze_ar_error("", "", "", metrics)
        
        self.assertIsNotNone(error)
        self.assertEqual(error.error_type, ARErrorType.THERMAL_THROTTLING)
        self.assertEqual(error.severity, "critical")
    
    def test_motion_sickness_detection(self):
        """Test motion sickness risk detection"""
        # High motion-to-photon latency
        metrics = ARPerformanceMetrics(
            fps=90,
            frame_time_ms=11,
            cpu_usage=0.60,
            gpu_usage=0.70,
            memory_usage=0.50,
            battery_drain=10,
            thermal_state="nominal",
            tracking_quality=1.0,
            rendering_latency_ms=10,
            motion_to_photon_latency_ms=25  # Above threshold
        )
        
        error = self.manager.analyze_ar_error("", "", "", metrics)
        
        self.assertIsNotNone(error)
        self.assertEqual(error.error_type, ARErrorType.MOTION_SICKNESS_RISK)
    
    def test_healing_strategy_suggestion(self):
        """Test healing strategy suggestions"""
        error = ARError(
            error_type=ARErrorType.TRACKING_LOST,
            platform=ARPlatform.ARCORE,
            description="Tracking lost",
            confidence=0.9
        )
        
        strategies = self.manager.suggest_healing(error)
        
        self.assertTrue(len(strategies) > 0)
        strategy_names = [s["name"] for s in strategies]
        self.assertIn("tracking_recovery_ui", strategy_names)
    
    def test_arcore_healing_code_generation(self):
        """Test ARCore healing code generation"""
        error = ARError(
            error_type=ARErrorType.TRACKING_LOST,
            platform=ARPlatform.ARCORE,
            description="Tracking lost"
        )
        
        strategy = {
            "name": "tracking_recovery_ui",
            "description": "Show tracking recovery UI"
        }
        
        code = self.manager.generate_healing_code(error, strategy)
        
        self.assertIsNotNone(code)
        self.assertIn("TrackingState", code)
        self.assertIn("showTrackingRecoveryUI", code)
    
    def test_unity_lod_code_generation(self):
        """Test Unity LOD code generation"""
        error = ARError(
            error_type=ARErrorType.PERFORMANCE_DEGRADATION,
            platform=ARPlatform.UNITY_AR,
            description="Low FPS"
        )
        
        strategy = {
            "name": "dynamic_lod",
            "description": "Dynamic level of detail"
        }
        
        code = self.manager.generate_healing_code(error, strategy)
        
        self.assertIsNotNone(code)
        self.assertIn("LOD", code)
        self.assertIn("SetLOD", code)
    
    def test_comfort_violation_check(self):
        """Test comfort violation detection"""
        motion_data = {
            "angular_velocity": 200,  # Above 180 deg/s threshold
            "linear_acceleration": 6.0,  # Above 5 m/s² threshold
            "fov": 50  # Below 60 degree minimum
        }
        
        violations = self.manager.check_comfort_violations(motion_data)
        
        self.assertEqual(len(violations), 3)
        self.assertIn("Excessive rotation speed", violations)
        self.assertIn("High acceleration detected", violations)
        self.assertIn("Field of view too narrow", violations)
    
    def test_performance_analysis(self):
        """Test performance analysis"""
        metrics = ARPerformanceMetrics(
            fps=60,
            frame_time_ms=16.67,
            cpu_usage=0.45,
            gpu_usage=0.60,
            memory_usage=0.55,
            battery_drain=8,
            thermal_state="nominal",
            tracking_quality=0.95,
            rendering_latency_ms=10
        )
        
        analysis = self.manager.analyze_performance(metrics)
        
        self.assertEqual(analysis["overall_status"], "healthy")
        self.assertEqual(len(analysis["issues"]), 0)


class TestARPlugin(unittest.TestCase):
    """Test AR plugin functionality"""
    
    def setUp(self):
        self.plugin = ARPlugin()
    
    def test_plugin_initialization(self):
        """Test plugin initialization"""
        self.assertEqual(self.plugin.name, "ar")
        self.assertIn(".cs", self.plugin.supported_extensions)
        self.assertIn("arcore", self.plugin.supported_platforms)
    
    def test_unity_performance_issue_detection(self):
        """Test Unity AR performance issue detection"""
        code = """
void Update()
{
    // Heavy AR operations in Update
    var hits = new List<ARRaycastHit>();
    raycastManager.Raycast(screenPoint, hits, TrackableType.Planes);
    
    foreach (var plane in planeManager.trackables)
    {
        // Process every plane every frame
        ProcessPlane(plane);
    }
}
        """
        
        errors = self.plugin.detect_errors(code, "ARScript.cs")
        
        self.assertTrue(any(e["type"] == "PerformanceIssue" for e in errors))
    
    def test_arcore_tracking_state_check(self):
        """Test ARCore tracking state check detection"""
        code = """
public void onUpdate(Frame frame) {
    Camera camera = frame.getCamera();
    Pose pose = camera.getPose();  // Using without checking state
    updateARObject(pose);
}
        """
        
        errors = self.plugin.detect_errors(code, "ARCore.java")
        
        self.assertTrue(any(e["type"] == "TrackingStateCheck" for e in errors))
    
    def test_webxr_feature_detection(self):
        """Test WebXR feature detection"""
        code = """
async function startXR() {
    // Missing feature detection
    const session = await navigator.xr.requestSession('immersive-ar');
    session.updateRenderState({ baseLayer: new XRWebGLLayer(session, gl) });
}
        """
        
        errors = self.plugin.detect_errors(code, "webxr.js")
        
        self.assertTrue(any(e["type"] == "FeatureDetection" for e in errors))
    
    def test_comfort_settings_detection(self):
        """Test comfort settings detection"""
        code = """
// Smooth locomotion without comfort options
void UpdateMovement() {
    transform.position += moveDirection * speed * Time.deltaTime;
    transform.rotation = Quaternion.Euler(0, rotationY, 0);
}
        """
        
        errors = self.plugin.detect_errors(code, "Movement.cs")
        
        self.assertTrue(any(e["type"] == "ComfortSettings" for e in errors))
    
    def test_error_analysis_with_metrics(self):
        """Test error analysis with performance metrics"""
        error_msg = "Frame rate dropped below target"
        code = "using UnityEngine.XR.ARFoundation;"
        metrics = {
            "fps": 22,
            "frame_time_ms": 45,
            "cpu_usage": 0.88,
            "gpu_usage": 0.92,
            "thermal_state": "serious"
        }
        
        analysis = self.plugin.analyze_error(error_msg, code, "app.cs", metrics)
        
        self.assertIsNotNone(analysis)
        self.assertEqual(analysis["error_type"], "performance_degradation")
        self.assertIn("healing_strategies", analysis)
    
    def test_platform_feature_detection(self):
        """Test platform feature detection"""
        code = """
// Unity AR with multiple features
ARPlaneManager planeManager;
ARFaceManager faceManager;
AROcclusionManager occlusionManager;

void Start() {
    planeManager.planesChanged += OnPlanesChanged;
    
    var lightEstimation = cameraManager.currentLightEstimation;
    
    if (ARSession.state == ARSessionState.SessionTracking) {
        StartCloudAnchors();
    }
}
        """
        
        info = self.plugin.get_platform_info(code, "MultiFeature.cs")
        
        self.assertEqual(info["platform"], "unity_ar")
        features = info["detected_features"]
        self.assertIn("plane_detection", features)
        self.assertIn("face_tracking", features)
        self.assertIn("occlusion", features)
        self.assertIn("light_estimation", features)
        self.assertIn("cloud_anchors", features)
    
    def test_rendering_api_detection(self):
        """Test rendering API detection"""
        code = """
#include <metal/metal.h>
#import <MetalKit/MetalKit.h>

@interface ARMetalRenderer : NSObject<MTKViewDelegate>
@property (nonatomic, strong) id<MTLDevice> device;
@end
        """
        
        info = self.plugin.get_platform_info(code, "Renderer.m")
        
        self.assertEqual(info["rendering_api"], "metal")
    
    def test_fix_generation_and_validation(self):
        """Test fix generation and validation"""
        error_analysis = {
            "error_type": "tracking_lost",
            "platform": "arkit",
            "description": "Tracking lost",
            "healing_strategies": [{
                "name": "tracking_recovery_ui",
                "description": "Show recovery UI"
            }]
        }
        
        fix_code = self.plugin.generate_fix(error_analysis, "")
        
        self.assertIsNotNone(fix_code)
        self.assertIn("tracking", fix_code.lower())
        
        # Validate fix
        is_valid = self.plugin.validate_fix("", fix_code, error_analysis)
        self.assertTrue(is_valid)


class TestARErrorScenarios(unittest.TestCase):
    """Test specific AR error scenarios"""
    
    def setUp(self):
        self.manager = ARResilienceManager()
    
    def test_plane_detection_failure(self):
        """Test plane detection failure"""
        error_msg = "No planes detected after 30 seconds"
        code = "ARPlaneManager planeManager;"
        
        error = self.manager.analyze_ar_error(error_msg, code, "app.cs")
        
        self.assertIsNotNone(error)
        self.assertEqual(error.error_type, ARErrorType.PLANE_DETECTION_FAILURE)
    
    def test_anchor_drift_detection(self):
        """Test anchor drift detection"""
        error_msg = "Anchor position drifted by 0.5 meters"
        code = "// AR anchor code"
        
        error = self.manager.analyze_ar_error(error_msg, code, "anchor.cs")
        
        self.assertIsNotNone(error)
        self.assertEqual(error.error_type, ARErrorType.ANCHOR_DRIFT)
    
    def test_cloud_anchor_error(self):
        """Test cloud anchor error"""
        error_msg = "Cloud anchor hosting failed: Network timeout"
        code = "Session.hostCloudAnchor(anchor);"
        
        error = self.manager.analyze_ar_error(error_msg, code, "cloud.java")
        
        self.assertIsNotNone(error)
        self.assertEqual(error.error_type, ARErrorType.CLOUD_ANCHOR_ERROR)
    
    def test_occlusion_error(self):
        """Test occlusion error detection"""
        error_msg = "People occlusion depth map generation failed"
        code = "configuration.peopleOcclusion = .enabled"
        
        error = self.manager.analyze_ar_error(error_msg, code, "occlusion.swift")
        
        self.assertIsNotNone(error)
        self.assertEqual(error.error_type, ARErrorType.OCCLUSION_ERROR)
    
    def test_lighting_estimation_error(self):
        """Test lighting estimation error"""
        error_msg = "Environment probe illumination estimation failed"
        code = "lightEstimation.ambientIntensity"
        
        error = self.manager.analyze_ar_error(error_msg, code, "lighting.swift")
        
        self.assertIsNotNone(error)
        self.assertEqual(error.error_type, ARErrorType.LIGHTING_ESTIMATION_ERROR)


if __name__ == "__main__":
    unittest.main()