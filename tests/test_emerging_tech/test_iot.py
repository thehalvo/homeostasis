"""
Test cases for IoT and edge device support
"""

import unittest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
from modules.emerging_tech.iot import (
    IoTDeviceMonitor, IoTPlatform, IoTErrorType, IoTError, DeviceMetrics
)
from modules.analysis.plugins.iot_plugin import IoTPlugin


class TestIoTDeviceMonitor(unittest.TestCase):
    """Test IoT device monitoring functionality"""
    
    def setUp(self):
        self.monitor = IoTDeviceMonitor()
    
    def test_platform_detection_arduino(self):
        """Test Arduino platform detection"""
        code = """
#include <Arduino.h>
#include <Wire.h>

void setup() {
    Serial.begin(9600);
    pinMode(LED_BUILTIN, OUTPUT);
}

void loop() {
    digitalWrite(LED_BUILTIN, HIGH);
    delay(1000);
}
        """
        
        platform = self.monitor.detect_platform(code, "blink.ino")
        self.assertEqual(platform, IoTPlatform.ARDUINO)
    
    def test_platform_detection_esp32(self):
        """Test ESP32 platform detection"""
        code = """
#include <WiFi.h>
#include <esp_system.h>

const char* ssid = "MyNetwork";

void setup() {
    Serial.begin(115200);
    WiFi.begin(ssid, password);
    
    esp_sleep_enable_timer_wakeup(60 * 1000000);
}
        """
        
        platform = self.monitor.detect_platform(code, "esp32_wifi.ino")
        self.assertEqual(platform, IoTPlatform.ESP32)
    
    def test_platform_detection_raspberry_pi(self):
        """Test Raspberry Pi platform detection"""
        code = """
import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)
GPIO.setup(18, GPIO.OUT)

while True:
    GPIO.output(18, GPIO.HIGH)
    time.sleep(1)
    GPIO.output(18, GPIO.LOW)
    time.sleep(1)
        """
        
        platform = self.monitor.detect_platform(code, "gpio_control.py")
        self.assertEqual(platform, IoTPlatform.RASPBERRY_PI)
    
    def test_platform_detection_mqtt(self):
        """Test MQTT platform detection"""
        code = """
import paho.mqtt.client as mqtt

def on_connect(client, userdata, flags, rc):
    print(f"Connected with result code {rc}")
    client.subscribe("sensors/temperature")

client = mqtt.Client()
client.on_connect = on_connect
client.connect("mqtt.broker.com", 1883, 60)
        """
        
        platform = self.monitor.detect_platform(code, "mqtt_client.py")
        self.assertEqual(platform, IoTPlatform.MQTT)
    
    def test_connectivity_error_detection(self):
        """Test connectivity error detection"""
        error_msg = "WiFi connection lost"
        code = "#include <WiFi.h>"
        
        error = self.monitor.analyze_iot_error(error_msg, code, "wifi.ino")
        
        self.assertIsNotNone(error)
        self.assertEqual(error.error_type, IoTErrorType.CONNECTIVITY_ERROR)
        self.assertEqual(error.platform, IoTPlatform.ESP32)
    
    def test_memory_overflow_detection(self):
        """Test memory overflow detection"""
        error_msg = "Error: Out of memory"
        code = "void setup() { Serial.begin(9600); }"
        
        error = self.monitor.analyze_iot_error(error_msg, code, "test.ino")
        
        self.assertIsNotNone(error)
        self.assertEqual(error.error_type, IoTErrorType.MEMORY_OVERFLOW)
    
    def test_sensor_failure_detection(self):
        """Test sensor failure detection"""
        error_msg = "Sensor reading failed: Invalid value"
        code = "int sensorValue = analogRead(A0);"
        
        error = self.monitor.analyze_iot_error(error_msg, code, "sensor.ino")
        
        self.assertIsNotNone(error)
        self.assertEqual(error.error_type, IoTErrorType.SENSOR_FAILURE)
    
    def test_resource_constraint_detection(self):
        """Test resource constraint detection based on metrics"""
        metrics = DeviceMetrics(
            cpu_usage=0.95,
            memory_usage=0.92,
            disk_usage=0.50,
            temperature=65.0
        )
        
        error = self.monitor.analyze_iot_error("", "", "", metrics)
        
        self.assertIsNotNone(error)
        self.assertEqual(error.error_type, IoTErrorType.RESOURCE_CONSTRAINT)
        self.assertEqual(error.severity, "critical")
    
    def test_healing_strategy_suggestion(self):
        """Test healing strategy suggestions"""
        error = IoTError(
            error_type=IoTErrorType.CONNECTIVITY_ERROR,
            platform=IoTPlatform.ESP32,
            description="WiFi disconnected",
            confidence=0.9
        )
        
        strategies = self.monitor.suggest_healing(error)
        
        self.assertTrue(len(strategies) > 0)
        strategy_names = [s["name"] for s in strategies]
        self.assertIn("auto_reconnect", strategy_names)
    
    def test_arduino_healing_code_generation(self):
        """Test Arduino healing code generation"""
        error = IoTError(
            error_type=IoTErrorType.RESOURCE_CONSTRAINT,
            platform=IoTPlatform.ARDUINO,
            description="Low memory"
        )
        
        strategy = {
            "name": "resource_optimization",
            "description": "Optimize memory usage"
        }
        
        code = self.monitor.generate_healing_code(error, strategy)
        
        self.assertIsNotNone(code)
        self.assertIn("PROGMEM", code)  # Arduino memory optimization
    
    def test_esp32_power_management_code(self):
        """Test ESP32 power management code generation"""
        error = IoTError(
            error_type=IoTErrorType.POWER_MANAGEMENT,
            platform=IoTPlatform.ESP32,
            description="Battery low"
        )
        
        strategy = {
            "name": "power_saving_mode",
            "description": "Enable power saving"
        }
        
        code = self.monitor.generate_healing_code(error, strategy)
        
        self.assertIsNotNone(code)
        self.assertIn("esp_pm_config", code)
        self.assertIn("esp_sleep", code)
    
    def test_device_health_analysis(self):
        """Test device health analysis"""
        metrics = DeviceMetrics(
            cpu_usage=0.45,
            memory_usage=0.60,
            disk_usage=0.30,
            battery_level=85.0,
            temperature=45.0
        )
        
        health = self.monitor.analyze_device_health("device_001", metrics)
        
        self.assertEqual(health["overall_health"], "healthy")
        self.assertEqual(len(health["issues"]), 0)
    
    def test_device_health_with_issues(self):
        """Test device health analysis with issues"""
        metrics = DeviceMetrics(
            cpu_usage=0.90,
            memory_usage=0.95,
            disk_usage=0.30,
            battery_level=15.0,
            temperature=82.0
        )
        
        health = self.monitor.analyze_device_health("device_002", metrics)
        
        self.assertEqual(health["overall_health"], "critical")
        self.assertIn("Critical memory usage", health["issues"])
        self.assertIn("High CPU usage", health["issues"])
        self.assertIn("Low battery", health["issues"])
        self.assertIn("High temperature", health["issues"])
    
    def test_device_config_generation(self):
        """Test device configuration generation"""
        requirements = {
            "low_power": True,
            "battery_powered": True,
            "simple_task": True
        }
        
        config = self.monitor.generate_device_config(IoTPlatform.ESP32, requirements)
        
        self.assertEqual(config["platform"], "esp32")
        self.assertEqual(config["settings"]["cpu_freq"], 160)  # Lower for power saving
        self.assertTrue(config["settings"]["deep_sleep_enabled"])


class TestIoTPlugin(unittest.TestCase):
    """Test IoT plugin functionality"""
    
    def setUp(self):
        self.plugin = IoTPlugin()
    
    def test_plugin_initialization(self):
        """Test plugin initialization"""
        self.assertEqual(self.plugin.name, "iot")
        self.assertIn(".ino", self.plugin.supported_extensions)
        self.assertIn("arduino", self.plugin.supported_platforms)
    
    def test_blocking_operation_detection(self):
        """Test detection of blocking operations"""
        code = """
void loop() {
    readSensor();
    delay(5000);  // 5 second blocking delay
    sendData();
}
        """
        
        errors = self.plugin.detect_errors(code, "test.ino")
        
        self.assertTrue(any(e["type"] == "BlockingOperation" for e in errors))
    
    def test_hardcoded_credentials_detection(self):
        """Test detection of hardcoded credentials"""
        code = """
const char* password = "my_secret_password";
const char* apiKey = "abcd1234efgh5678";
        """
        
        errors = self.plugin.detect_errors(code, "config.h")
        
        self.assertTrue(any(e["type"] == "HardcodedCredentials" for e in errors))
    
    def test_arduino_string_usage_detection(self):
        """Test Arduino String class usage detection"""
        code = """
String message = "Hello World";
String sensorData = String(analogRead(A0));
        """
        
        errors = self.plugin.detect_errors(code, "arduino_code.ino")
        
        self.assertTrue(any(e["type"] == "ArduinoStringUsage" for e in errors))
    
    def test_missing_pinmode_detection(self):
        """Test missing pinMode detection"""
        code = """
void setup() {
    Serial.begin(9600);
    pinMode(LED_PIN, OUTPUT);
}

void loop() {
    digitalWrite(LED_PIN, HIGH);
    digitalWrite(BUZZER_PIN, LOW);  // BUZZER_PIN not initialized
}
        """
        
        errors = self.plugin.detect_errors(code, "test.ino")
        
        self.assertTrue(any(e["type"] == "MissingPinMode" for e in errors))
    
    def test_esp32_task_stack_detection(self):
        """Test ESP32 task stack size detection"""
        code = """
#include <WiFi.h>  // ESP32 WiFi library

void setup() {
    xTaskCreate(sensorTask, "Sensor", 1024, NULL, 1, NULL);  // Small stack
}
        """
        
        errors = self.plugin.detect_errors(code, "esp32_tasks.ino")
        
        self.assertTrue(any(e["type"] == "InsufficientTaskStack" for e in errors))
    
    def test_mqtt_callback_error_handling(self):
        """Test MQTT callback error handling detection"""
        code = """
import paho.mqtt.client as mqtt

def on_message(client, userdata, msg):
    data = json.loads(msg.payload)  # No error handling
    process_data(data)
    
client = mqtt.Client()
client.connect("broker", 1883)
        """
        
        errors = self.plugin.detect_errors(code, "mqtt_handler.py")
        
        self.assertTrue(any(e["type"] == "UnprotectedMQTTCallback" for e in errors))
    
    def test_error_analysis_with_metrics(self):
        """Test error analysis with device metrics"""
        error_msg = "System overload detected"
        code = "void loop() { /* sensor code */ }"
        metrics = {
            "cpu_usage": 0.92,
            "memory_usage": 0.88,
            "temperature": 78.5
        }
        
        # This is the custom analyze_error method defined at line 317 in iot_plugin.py
        analysis = self.plugin.analyze_error(error_msg, code, "sensor.ino", metrics)
        
        self.assertIsNotNone(analysis)
        self.assertIn("resource_usage", analysis)
    
    def test_platform_info_extraction(self):
        """Test platform information extraction"""
        code = """
#include <WiFi.h>
#include <Wire.h>
#include <SPI.h>
#include <esp_deep_sleep.h>

void setup() {
    WiFi.begin(ssid, password);
    Wire.begin();
    esp_sleep_enable_timer_wakeup(60000000);
}
        """
        
        info = self.plugin.get_platform_info(code, "esp32_multi.ino")
        
        self.assertEqual(info["platform"], "esp32")
        self.assertIn("wifi", info["detected_features"])
        self.assertIn("deep_sleep", info["detected_features"])
    
    def test_protocol_detection(self):
        """Test communication protocol detection"""
        code = """
import paho.mqtt.client as mqtt
import requests
from websocket import create_connection

# MQTT setup
client = mqtt.Client()

# HTTP request
response = requests.get("http://api.server.com")

# WebSocket
ws = create_connection("ws://localhost:8080")
        """
        
        info = self.plugin.get_platform_info(code, "multi_protocol.py")
        
        protocols = info["communication_protocols"]
        self.assertIn("mqtt", protocols)
        self.assertIn("http", protocols)
        self.assertIn("websocket", protocols)
    
    def test_fix_generation_and_validation(self):
        """Test fix generation and validation"""
        error_analysis = {
            "error_type": "connectivity_error",
            "platform": "esp32",
            "description": "WiFi connection lost",
            "healing_strategies": [{
                "name": "auto_reconnect",
                "description": "Automatic reconnection"
            }]
        }
        
        fix_code = self.plugin.generate_fix(error_analysis, "")
        
        self.assertIsNotNone(fix_code)
        self.assertIn("reconnect", fix_code.lower())
        
        # Validate fix
        is_valid = self.plugin.validate_fix("", fix_code, error_analysis)
        self.assertTrue(is_valid)


class TestIoTEdgeScenarios(unittest.TestCase):
    """Test IoT edge computing scenarios"""
    
    def setUp(self):
        self.monitor = IoTDeviceMonitor()
    
    def test_edge_processing_error(self):
        """Test edge processing error detection"""
        error_msg = "Model inference timeout after 30s"
        code = "import tensorflow.lite as tflite"
        
        error = self.monitor.analyze_iot_error(error_msg, code, "edge_ml.py")
        
        self.assertIsNotNone(error)
        self.assertEqual(error.error_type, IoTErrorType.EDGE_PROCESSING_ERROR)
    
    def test_synchronization_error(self):
        """Test synchronization error detection"""
        error_msg = "Time sync failed: NTP server unreachable"
        code = "// IoT device code"
        
        error = self.monitor.analyze_iot_error(error_msg, code, "device.c")
        
        self.assertIsNotNone(error)
        self.assertEqual(error.error_type, IoTErrorType.SYNCHRONIZATION_ERROR)
    
    def test_security_breach_detection(self):
        """Test security breach detection"""
        error_msg = "Unauthorized access attempt detected from 192.168.1.100"
        code = "// Security monitoring code"
        
        error = self.monitor.analyze_iot_error(error_msg, code, "security.c")
        
        self.assertIsNotNone(error)
        self.assertEqual(error.error_type, IoTErrorType.SECURITY_BREACH)
        self.assertEqual(error.severity, "critical")
    
    def test_firmware_error_detection(self):
        """Test firmware error detection"""
        error_msg = "Firmware verification failed: Invalid signature"
        code = "// OTA update code"
        
        error = self.monitor.analyze_iot_error(error_msg, code, "ota.c")
        
        self.assertIsNotNone(error)
        self.assertEqual(error.error_type, IoTErrorType.FIRMWARE_ERROR)
        self.assertEqual(error.severity, "critical")


if __name__ == "__main__":
    unittest.main()