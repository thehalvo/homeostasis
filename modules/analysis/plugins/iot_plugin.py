"""
IoT and Edge Device Plugin for Homeostasis

Provides IoT-specific error detection and healing capabilities
"""

import os
import json
import re
from typing import Dict, List, Optional, Any
from ..language_plugin_system import LanguagePlugin
from ...emerging_tech.iot import (
    IoTDeviceMonitor, IoTPlatform, IoTError, DeviceMetrics
)


class IoTPlugin(LanguagePlugin):
    """Plugin for IoT and edge device platforms"""
    
    def __init__(self):
        super().__init__()
        self.name = "iot"
        self.version = "0.1.0"
        self.supported_extensions = [
            ".ino",     # Arduino
            ".cpp",     # C++ (ESP32, Arduino)
            ".c",       # C
            ".py",      # Python (Raspberry Pi, MicroPython)
            ".js",      # JavaScript (Node.js IoT)
            ".yaml",    # Configuration files
            ".json"     # Configuration files
        ]
        self.supported_platforms = [
            "arduino", "esp32", "raspberry_pi", "mqtt",
            "coap", "lorawan", "zigbee", "edge_computing"
        ]
        self.monitor = IoTDeviceMonitor()
        self._load_rules()
    
    def get_language_id(self) -> str:
        """Get the unique identifier for this language."""
        return "iot"
    
    def get_language_name(self) -> str:
        """Get the human-readable name of the language."""
        return "IoT and Edge Computing"
    
    def get_language_version(self) -> str:
        """Get the version of the language supported by this plugin."""
        return "1.0"
    
    def analyze_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze an IoT-specific error."""
        error_message = error_data.get("message", "")
        code = error_data.get("code", "")
        file_path = error_data.get("file_path", "")
        device_metrics = error_data.get("device_metrics")
        
        # Convert metrics if provided
        metrics = None
        if device_metrics:
            metrics = DeviceMetrics(
                cpu_usage=device_metrics.get("cpu_usage", 0),
                memory_usage=device_metrics.get("memory_usage", 0),
                temperature=device_metrics.get("temperature", 25),
                battery_level=device_metrics.get("battery_level", 100),
                network_latency=device_metrics.get("network_latency", 0),
                signal_strength=device_metrics.get("signal_strength", -50)
            )
        
        iot_error = self.monitor.analyze_iot_error(
            error_message, code, file_path, metrics
        )
        
        if iot_error:
            return {
                "error_type": iot_error.error_type.value,
                "platform": iot_error.platform.value,
                "description": iot_error.description,
                "suggested_fix": iot_error.suggested_fix,
                "severity": iot_error.severity,
                "resource_impact": iot_error.resource_impact
            }
        
        return {"error_type": "unknown", "description": "Could not analyze IoT error"}
    
    def normalize_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize error data to the standard Homeostasis format."""
        return {
            "type": error_data.get("type", "error"),
            "message": error_data.get("message", ""),
            "severity": error_data.get("severity", "medium"),
            "platform": error_data.get("platform", "unknown"),
            "device_id": error_data.get("device_id"),
            "timestamp": error_data.get("timestamp")
        }
    
    def denormalize_error(self, standard_error: Dict[str, Any]) -> Dict[str, Any]:
        """Convert standard format error data back to IoT-specific format."""
        return {
            "type": standard_error.get("type", "error"),
            "message": standard_error.get("message", ""),
            "severity": standard_error.get("severity", "medium"),
            "platform": standard_error.get("platform", "unknown"),
            "device_id": standard_error.get("device_id")
        }
    
    def generate_fix(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a fix for an IoT error based on the analysis."""
        error_type = analysis.get("error_type")
        platform = analysis.get("platform")
        
        # Generate platform-specific fixes
        if error_type == "memory_constraint":
            return {
                "type": "optimization",
                "description": "Optimize memory usage",
                "suggestions": [
                    "Use PROGMEM for constant data (Arduino)",
                    "Avoid dynamic memory allocation",
                    "Use smaller data types",
                    "Free unused resources"
                ]
            }
        elif error_type == "connectivity_error":
            return {
                "type": "network_fix",
                "description": "Improve network connectivity",
                "code": self._get_connectivity_handler(platform)
            }
        elif error_type == "power_management":
            return {
                "type": "power_optimization",
                "description": "Optimize power consumption",
                "suggestions": [
                    "Use deep sleep modes",
                    "Reduce transmission frequency",
                    "Lower CPU frequency",
                    "Disable unused peripherals"
                ]
            }
        
        return {
            "type": "suggestion",
            "description": "Review IoT best practices for your platform"
        }
    
    def get_supported_frameworks(self) -> List[str]:
        """Get the list of frameworks supported by this language plugin."""
        return self.supported_platforms
    
    def _load_rules(self):
        """Load IoT-specific error rules"""
        rules_path = os.path.join(
            os.path.dirname(__file__),
            "../rules/iot/iot_errors.json"
        )
        
        if os.path.exists(rules_path):
            with open(rules_path, 'r') as f:
                self.rules = json.load(f)
        else:
            self.rules = {"rules": [], "platform_specific": {}}
    
    def detect_errors(self, code: str, file_path: str = None) -> List[Dict[str, Any]]:
        """Detect IoT-specific errors in code"""
        errors = []
        
        # Detect platform
        platform = self.monitor.detect_platform(code, file_path or "")
        
        # Always check for generic IoT issues
        errors.extend(self._detect_generic_iot_issues(code))
        
        if platform != IoTPlatform.UNKNOWN:
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
        if platform == IoTPlatform.ARDUINO:
            errors.extend(self._check_arduino_specific(code))
        elif platform == IoTPlatform.ESP32:
            errors.extend(self._check_esp32_specific(code))
        elif platform == IoTPlatform.MQTT:
            errors.extend(self._check_mqtt_specific(code))
        
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
    
    def _detect_generic_iot_issues(self, code: str) -> List[Dict[str, Any]]:
        """Detect generic IoT issues"""
        issues = []
        
        # Check for blocking operations in IoT code
        if re.search(r"delay\(\d{4,}\)|sleep\(\d+\)|time\.sleep\([^)]*\)", code):
            issues.append({
                "type": "BlockingOperation",
                "description": "Long blocking operation detected in IoT code",
                "severity": "medium",
                "suggestion": "Use non-blocking delays or async operations"
            })
        
        # Check for hardcoded credentials
        if re.search(r'(password|passwd|pwd|key|token)\s*=\s*["\'][^"\']+["\']', code, re.IGNORECASE):
            issues.append({
                "type": "HardcodedCredentials",
                "description": "Hardcoded credentials detected",
                "severity": "critical",
                "suggestion": "Use secure credential storage"
            })
        
        return issues
    
    def _check_arduino_specific(self, code: str) -> List[Dict[str, Any]]:
        """Check for Arduino-specific issues"""
        issues = []
        
        # Check for String usage (memory intensive)
        if re.search(r'\bString\s+\w+|String\(', code):
            issues.append({
                "type": "ArduinoStringUsage",
                "description": "String class usage can cause memory fragmentation",
                "severity": "medium",
                "suggestion": "Use char arrays for better memory management"
            })
        
        # Check for missing pinMode
        digital_operations = re.findall(r'digitalWrite\((\w+),', code)
        pin_modes = re.findall(r'pinMode\((\w+),', code)
        
        for pin in digital_operations:
            if pin not in pin_modes and not pin.isdigit():
                issues.append({
                    "type": "MissingPinMode",
                    "description": f"Pin {pin} used without pinMode initialization",
                    "severity": "high",
                    "suggestion": f"Add pinMode({pin}, OUTPUT) in setup()"
                })
        
        return issues
    
    def _check_esp32_specific(self, code: str) -> List[Dict[str, Any]]:
        """Check for ESP32-specific issues"""
        issues = []
        
        # Check for task creation without adequate stack
        task_creates = re.findall(r'xTaskCreate[^;]+', code)
        for task in task_creates:
            # Look for stack size parameter (third parameter in xTaskCreate)
            # xTaskCreate(function, "name", stackSize, ...)
            parts = task.split(',')
            if len(parts) >= 3:
                stack_param = parts[2].strip()
                stack_match = re.search(r'(\d+)', stack_param)
                if stack_match and int(stack_match.group(1)) < 2048:
                    issues.append({
                        "type": "InsufficientTaskStack",
                        "description": "Task created with insufficient stack size",
                        "severity": "high",
                        "suggestion": "Increase stack size to at least 2048 bytes"
                    })
        
        # Check for WiFi without event handlers
        if "WiFi.begin" in code and not re.search(r"WiFi\.on|WiFiEvent", code):
            issues.append({
                "type": "MissingWiFiEventHandlers",
                "description": "WiFi used without event handlers",
                "severity": "medium",
                "suggestion": "Add WiFi event handlers for robust connectivity"
            })
        
        return issues
    
    def _check_mqtt_specific(self, code: str) -> List[Dict[str, Any]]:
        """Check for MQTT-specific issues"""
        issues = []
        
        # Check for missing error handling in callbacks
        if re.search(r"def\s+on_message|def\s+on_connect", code):
            # Look for function definitions and check if they have try-except
            callback_match = re.search(r"def\s+on_\w+\([^)]+\):\s*\n(.*?)(?=\ndef|\Z)", code, re.DOTALL)
            if callback_match:
                callback_body = callback_match.group(1)
                if "try:" not in callback_body:
                    issues.append({
                        "type": "UnprotectedMQTTCallback",
                        "description": "MQTT callback without error handling",
                        "severity": "high",
                        "suggestion": "Add try-except blocks in MQTT callbacks"
                    })
        
        # Check for missing QoS specification
        if re.search(r"publish\([^)]+\)", code) and not re.search(r"qos\s*=", code):
            issues.append({
                "type": "DefaultQoSUsage",
                "description": "MQTT publish without explicit QoS",
                "severity": "low",
                "suggestion": "Specify QoS level based on message importance"
            })
        
        return issues
    
    def analyze_error(self, error_message: str, code_context: str,
                     file_path: str = None, device_metrics: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
        """Analyze IoT error and suggest fixes"""
        # Convert metrics dict to DeviceMetrics if provided
        metrics = None
        if device_metrics:
            metrics = DeviceMetrics(
                cpu_usage=device_metrics.get("cpu_usage", 0),
                memory_usage=device_metrics.get("memory_usage", 0),
                disk_usage=device_metrics.get("disk_usage", 0),
                battery_level=device_metrics.get("battery_level"),
                temperature=device_metrics.get("temperature"),
                network_latency=device_metrics.get("network_latency"),
                packet_loss=device_metrics.get("packet_loss")
            )
        
        iot_error = self.monitor.analyze_iot_error(
            error_message, code_context, file_path or "", metrics
        )
        
        if not iot_error:
            return None
        
        healing_strategies = self.monitor.suggest_healing(iot_error)
        
        result = {
            "error_type": iot_error.error_type.value,
            "platform": iot_error.platform.value,
            "description": iot_error.description,
            "confidence": iot_error.confidence,
            "suggested_fix": iot_error.suggested_fix,
            "healing_strategies": healing_strategies,
            "device_info": iot_error.device_info,
            "resource_usage": iot_error.resource_usage if iot_error.resource_usage else {"cpu": 0, "memory": 0},
            "severity": iot_error.severity
        }
        
        # Add resource usage from metrics if available
        if metrics:
            result["resource_usage"] = {
                "cpu_usage": metrics.cpu_usage,
                "memory_usage": metrics.memory_usage,
                "temperature": metrics.temperature
            }
        
        return result
    
    def generate_fix(self, error_analysis: Dict[str, Any],
                    code_context: str) -> Optional[str]:
        """Generate fix code for IoT error"""
        if not error_analysis or "healing_strategies" not in error_analysis:
            return None
        
        strategies = error_analysis["healing_strategies"]
        if not strategies:
            return None
        
        # Use the first applicable strategy
        strategy = strategies[0]
        
        # Create an IoTError object for the monitor
        from ...emerging_tech.iot import IoTErrorType
        iot_error = IoTError(
            error_type=IoTErrorType(error_analysis["error_type"]),
            platform=IoTPlatform(error_analysis["platform"]),
            description=error_analysis["description"],
            device_info=error_analysis.get("device_info"),
            resource_usage=error_analysis.get("resource_usage")
        )
        
        return self.monitor.generate_healing_code(iot_error, strategy)
    
    def validate_fix(self, original_code: str, fixed_code: str,
                    error_analysis: Dict[str, Any]) -> bool:
        """Validate that fix addresses the IoT error"""
        if not fixed_code or not fixed_code.strip():
            return False
        
        # Check for expected patterns based on strategy
        validation_patterns = {
            "auto_reconnect": ["reconnect", "retry", "backoff"],
            "resource_optimization": ["memory", "optimize", "free"],
            "power_saving_mode": ["sleep", "power", "low"],
            "sensor_redundancy": ["redundan", "backup", "fallback"],
            "offline_mode": ["offline", "queue", "store"]
        }
        
        strategies = error_analysis.get("healing_strategies", [])
        if strategies:
            strategy_name = strategies[0].get("name", "")
            patterns = validation_patterns.get(strategy_name, [])
            return any(pattern in fixed_code.lower() for pattern in patterns)
        
        return True
    
    def analyze_device_health(self, device_id: str, metrics: Dict[str, float],
                            historical_data: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Analyze device health status"""
        device_metrics = DeviceMetrics(
            cpu_usage=metrics.get("cpu_usage", 0),
            memory_usage=metrics.get("memory_usage", 0),
            disk_usage=metrics.get("disk_usage", 0),
            battery_level=metrics.get("battery_level"),
            temperature=metrics.get("temperature"),
            network_latency=metrics.get("network_latency"),
            packet_loss=metrics.get("packet_loss")
        )
        
        # Convert historical data if provided
        historical_metrics = None
        if historical_data:
            historical_metrics = [
                DeviceMetrics(**data) for data in historical_data
            ]
        
        return self.monitor.analyze_device_health(
            device_id, device_metrics, historical_metrics
        )
    
    def get_platform_info(self, code: str, file_path: str = None) -> Dict[str, Any]:
        """Get information about the IoT platform being used"""
        platform = self.monitor.detect_platform(code, file_path or "")
        
        platform_info = {
            "platform": platform.value,
            "detected_features": self._detect_platform_features(code, platform),
            "communication_protocols": self._detect_protocols(code),
            "resource_constraints": self._get_platform_constraints(platform)
        }
        
        return platform_info
    
    def _detect_platform_features(self, code: str, platform: IoTPlatform) -> List[str]:
        """Detect platform-specific features being used"""
        features = []
        
        if platform == IoTPlatform.ESP32:
            feature_patterns = {
                "wifi": r"WiFi\.|esp_wifi",
                "bluetooth": r"BLE|Bluetooth|esp_bt",
                "deep_sleep": r"esp_deep_sleep|esp_sleep",
                "touch": r"touchRead|touch_pad",
                "hall_sensor": r"hallRead",
                "dac": r"dacWrite",
                "adc": r"analogRead|adc1_"
            }
            
            for feature, pattern in feature_patterns.items():
                if re.search(pattern, code):
                    features.append(feature)
        
        elif platform == IoTPlatform.ARDUINO:
            feature_patterns = {
                "serial": r"Serial\.",
                "i2c": r"Wire\.",
                "spi": r"SPI\.",
                "servo": r"Servo",
                "interrupts": r"attachInterrupt",
                "eeprom": r"EEPROM\."
            }
            
            for feature, pattern in feature_patterns.items():
                if re.search(pattern, code):
                    features.append(feature)
        
        return features
    
    def _detect_protocols(self, code: str) -> List[str]:
        """Detect communication protocols being used"""
        protocols = []
        
        protocol_patterns = {
            "mqtt": r"mqtt|paho|mosquitto",
            "http": r"http|requests|fetch",
            "coap": r"coap|aiocoap",
            "websocket": r"websocket|ws://|wss://",
            "modbus": r"modbus|pymodbus",
            "can": r"can|canbus",
            "zigbee": r"zigbee|xbee",
            "lorawan": r"lora|lorawan|sx1276"
        }
        
        for protocol, pattern in protocol_patterns.items():
            if re.search(pattern, code, re.IGNORECASE):
                protocols.append(protocol)
        
        return protocols
    
    def _get_platform_constraints(self, platform: IoTPlatform) -> Dict[str, Any]:
        """Get resource constraints for platform"""
        constraints = self.rules.get("platform_specific", {}).get(platform.value, {})
        
        # Add default constraints if not in rules
        if platform == IoTPlatform.ARDUINO and "memory_limits" not in constraints:
            constraints["memory_limits"] = {
                "default": {"ram": 2048, "flash": 32768}
            }
        
        return constraints
    
    def generate_device_config(self, platform: str, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Generate optimized device configuration"""
        try:
            platform_enum = IoTPlatform(platform)
            return self.monitor.generate_device_config(platform_enum, requirements)
        except ValueError:
            return {"error": f"Unknown platform: {platform}"}
    
    def _get_connectivity_handler(self, platform: str) -> str:
        """Get platform-specific connectivity handler code"""
        handlers = {
            "arduino": """// WiFi reconnection handler
void reconnectWiFi() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("Reconnecting to WiFi...");
    WiFi.disconnect();
    delay(1000);
    WiFi.begin(ssid, password);
    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 10) {
      delay(500);
      attempts++;
    }
  }
}""",
            "esp32": """// ESP32 WiFi event handler
void WiFiEvent(WiFiEvent_t event) {
  switch(event) {
    case SYSTEM_EVENT_STA_DISCONNECTED:
      esp_wifi_connect();
      break;
    case SYSTEM_EVENT_STA_GOT_IP:
      // Connection successful
      break;
  }
}""",
            "mqtt": """// MQTT reconnection handler
void reconnectMQTT() {
  while (!client.connected()) {
    if (client.connect(clientId, username, password)) {
      // Resubscribe to topics
      client.subscribe(topic);
    } else {
      delay(5000);
    }
  }
}"""
        }
        return handlers.get(platform, "// Add connectivity handler for your platform")
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return plugin capabilities"""
        return {
            "name": self.name,
            "version": self.version,
            "supported_platforms": self.supported_platforms,
            "supported_extensions": self.supported_extensions,
            "features": [
                "error_detection",
                "resource_monitoring",
                "connectivity_healing",
                "power_optimization",
                "sensor_redundancy",
                "edge_computing_support"
            ],
            "healing_strategies": [
                "auto_reconnect",
                "resource_optimization",
                "power_saving_mode",
                "sensor_redundancy",
                "offline_mode",
                "edge_offloading"
            ]
        }