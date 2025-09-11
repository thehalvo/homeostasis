"""
IoT and Edge Device Support Module

Provides error detection, healing, and resilience for IoT and edge computing
applications including device management, connectivity, and resource constraints.
"""

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class IoTPlatform(Enum):
    """Supported IoT platforms and protocols"""

    ARDUINO = "arduino"
    RASPBERRY_PI = "raspberry_pi"
    ESP32 = "esp32"
    AWS_IOT = "aws_iot"
    AZURE_IOT = "azure_iot"
    GOOGLE_IOT = "google_iot"
    MQTT = "mqtt"
    COAP = "coap"
    LORAWAN = "lorawan"
    ZIGBEE = "zigbee"
    EDGE_COMPUTING = "edge_computing"
    INDUSTRIAL_IOT = "industrial_iot"
    UNKNOWN = "unknown"


class IoTErrorType(Enum):
    """Types of IoT and edge device errors"""

    CONNECTIVITY_ERROR = "connectivity_error"
    RESOURCE_CONSTRAINT = "resource_constraint"
    SENSOR_FAILURE = "sensor_failure"
    ACTUATOR_FAILURE = "actuator_failure"
    POWER_MANAGEMENT = "power_management"
    MEMORY_OVERFLOW = "memory_overflow"
    NETWORK_TIMEOUT = "network_timeout"
    PROTOCOL_ERROR = "protocol_error"
    SECURITY_BREACH = "security_breach"
    FIRMWARE_ERROR = "firmware_error"
    SYNCHRONIZATION_ERROR = "synchronization_error"
    EDGE_PROCESSING_ERROR = "edge_processing_error"
    DATA_CORRUPTION = "data_corruption"
    DEVICE_OFFLINE = "device_offline"


@dataclass
class IoTError:
    """Represents an IoT or edge device error"""

    error_type: IoTErrorType
    platform: IoTPlatform
    description: str
    device_info: Optional[Dict[str, Any]] = None
    sensor_data: Optional[Dict[str, Any]] = None
    network_info: Optional[Dict[str, Any]] = None
    resource_usage: Optional[Dict[str, float]] = None
    suggested_fix: Optional[str] = None
    confidence: float = 0.0
    severity: str = "medium"
    timestamp: datetime = None


@dataclass
class DeviceMetrics:
    """Device resource metrics"""

    cpu_usage: float
    memory_usage: float
    disk_usage: float
    battery_level: Optional[float] = None
    temperature: Optional[float] = None
    network_latency: Optional[float] = None
    packet_loss: Optional[float] = None


class IoTDeviceMonitor:
    """Handles IoT and edge device monitoring and healing"""

    def __init__(self):
        self.error_patterns = self._load_error_patterns()
        self.healing_strategies = self._load_healing_strategies()
        self.platform_detectors = self._initialize_platform_detectors()
        self.resource_thresholds = self._initialize_resource_thresholds()
        self.protocol_handlers = self._initialize_protocol_handlers()

    def _load_error_patterns(self) -> Dict[str, List[Dict]]:
        """Load IoT error patterns for different platforms"""
        return {
            "arduino": [
                {
                    "pattern": r"Serial.*not.*available|Serial.*timeout",
                    "type": IoTErrorType.CONNECTIVITY_ERROR,
                    "description": "Serial communication timeout",
                    "fix": "Check serial connection and baud rate",
                },
                {
                    "pattern": r"Out.*of.*memory|heap.*overflow|stack.*overflow",
                    "type": IoTErrorType.MEMORY_OVERFLOW,
                    "description": "Memory overflow on Arduino",
                    "fix": "Optimize memory usage or use PROGMEM for constants",
                },
                {
                    "pattern": r"analogRead.*failed|digitalWrite.*error",
                    "type": IoTErrorType.SENSOR_FAILURE,
                    "description": "GPIO operation failed",
                    "fix": "Check pin connections and initialization",
                },
            ],
            "esp32": [
                {
                    "pattern": r"WiFi.*connection.*failed|WiFi.*disconnected",
                    "type": IoTErrorType.CONNECTIVITY_ERROR,
                    "description": "WiFi connection lost",
                    "fix": "Implement WiFi reconnection logic",
                },
                {
                    "pattern": r"Brownout.*detected|voltage.*low",
                    "type": IoTErrorType.POWER_MANAGEMENT,
                    "description": "Power supply insufficient",
                    "fix": "Use adequate power supply or reduce power consumption",
                },
                {
                    "pattern": r"Task.*watchdog.*triggered|WDT.*reset",
                    "type": IoTErrorType.FIRMWARE_ERROR,
                    "description": "Watchdog timer reset",
                    "fix": "Add task yields or increase WDT timeout",
                },
            ],
            "mqtt": [
                {
                    "pattern": r"MQTT.*connection.*lost|broker.*unreachable",
                    "type": IoTErrorType.CONNECTIVITY_ERROR,
                    "description": "MQTT broker connection lost",
                    "fix": "Implement automatic reconnection with backoff",
                },
                {
                    "pattern": r"QoS.*delivery.*failed|message.*not.*acknowledged",
                    "type": IoTErrorType.PROTOCOL_ERROR,
                    "description": "MQTT QoS delivery failure",
                    "fix": "Implement message persistence and retry",
                },
                {
                    "pattern": r"topic.*subscription.*failed|unauthorized.*topic",
                    "type": IoTErrorType.SECURITY_BREACH,
                    "description": "MQTT authorization error",
                    "fix": "Check topic permissions and credentials",
                },
            ],
            "edge_computing": [
                {
                    "pattern": r"inference.*timeout|model.*loading.*failed",
                    "type": IoTErrorType.EDGE_PROCESSING_ERROR,
                    "description": "Edge ML inference failure",
                    "fix": "Optimize model or increase timeout",
                },
                {
                    "pattern": r"cache.*full|storage.*exhausted",
                    "type": IoTErrorType.RESOURCE_CONSTRAINT,
                    "description": "Edge storage exhausted",
                    "fix": "Implement data pruning or offloading",
                },
                {
                    "pattern": r"sync.*failed|data.*inconsistent",
                    "type": IoTErrorType.SYNCHRONIZATION_ERROR,
                    "description": "Edge-cloud sync failure",
                    "fix": "Implement conflict resolution and retry",
                },
            ],
        }

    def _load_healing_strategies(self) -> Dict[IoTErrorType, List[Dict]]:
        """Load healing strategies for different error types"""
        return {
            IoTErrorType.CONNECTIVITY_ERROR: [
                {
                    "name": "auto_reconnect",
                    "description": "Automatic reconnection with exponential backoff",
                    "applicable_platforms": ["all"],
                    "implementation": "Implement reconnection loop with increasing delays",
                },
                {
                    "name": "connection_pooling",
                    "description": "Maintain multiple connections for failover",
                    "applicable_platforms": ["edge_computing", "industrial_iot"],
                    "implementation": "Use connection pool with health checks",
                },
                {
                    "name": "offline_mode",
                    "description": "Store and forward when offline",
                    "applicable_platforms": ["all"],
                    "implementation": "Buffer data locally and sync when connected",
                },
            ],
            IoTErrorType.RESOURCE_CONSTRAINT: [
                {
                    "name": "resource_optimization",
                    "description": "Optimize memory and CPU usage",
                    "applicable_platforms": ["arduino", "esp32", "raspberry_pi"],
                    "implementation": "Profile and optimize resource-intensive operations",
                },
                {
                    "name": "edge_offloading",
                    "description": "Offload computation to edge or cloud",
                    "applicable_platforms": ["edge_computing"],
                    "implementation": "Distribute workload based on resource availability",
                },
                {
                    "name": "adaptive_sampling",
                    "description": "Adjust sampling rate based on resources",
                    "applicable_platforms": ["all"],
                    "implementation": "Dynamically adjust data collection frequency",
                },
            ],
            IoTErrorType.SENSOR_FAILURE: [
                {
                    "name": "sensor_redundancy",
                    "description": "Use redundant sensors for critical measurements",
                    "applicable_platforms": ["industrial_iot"],
                    "implementation": "Average multiple sensor readings",
                },
                {
                    "name": "graceful_degradation",
                    "description": "Continue with reduced functionality",
                    "applicable_platforms": ["all"],
                    "implementation": "Provide estimates when sensors fail",
                },
                {
                    "name": "sensor_recalibration",
                    "description": "Automatic sensor recalibration",
                    "applicable_platforms": ["all"],
                    "implementation": "Periodic calibration routines",
                },
            ],
            IoTErrorType.POWER_MANAGEMENT: [
                {
                    "name": "power_saving_mode",
                    "description": "Enable low-power modes",
                    "applicable_platforms": ["esp32", "arduino"],
                    "implementation": "Use deep sleep and wake on interrupt",
                },
                {
                    "name": "dynamic_frequency_scaling",
                    "description": "Adjust CPU frequency based on workload",
                    "applicable_platforms": ["raspberry_pi", "edge_computing"],
                    "implementation": "Scale frequency with demand",
                },
                {
                    "name": "battery_monitoring",
                    "description": "Monitor and predict battery life",
                    "applicable_platforms": ["all"],
                    "implementation": "Track discharge rates and alert on low battery",
                },
            ],
            IoTErrorType.SECURITY_BREACH: [
                {
                    "name": "secure_boot",
                    "description": "Verify firmware integrity on boot",
                    "applicable_platforms": ["esp32", "industrial_iot"],
                    "implementation": "Use cryptographic signatures",
                },
                {
                    "name": "encrypted_communication",
                    "description": "Use TLS/DTLS for all communications",
                    "applicable_platforms": ["all"],
                    "implementation": "Implement end-to-end encryption",
                },
                {
                    "name": "intrusion_detection",
                    "description": "Detect anomalous behavior",
                    "applicable_platforms": ["edge_computing", "industrial_iot"],
                    "implementation": "Monitor for unusual patterns",
                },
            ],
        }

    def _initialize_platform_detectors(self) -> Dict[IoTPlatform, Dict]:
        """Initialize platform-specific detectors"""
        return {
            IoTPlatform.ARDUINO: {
                "includes": ["Arduino.h", "Wire.h", "SPI.h"],
                "functions": ["pinMode", "digitalWrite", "analogRead", "Serial.begin"],
                "file_extensions": [".ino"],
                "keywords": ["void setup()", "void loop()"],
            },
            IoTPlatform.ESP32: {
                "includes": ["WiFi.h", "esp_system.h", "freertos/FreeRTOS.h"],
                "functions": ["WiFi.begin", "esp_restart", "xTaskCreate"],
                "file_extensions": [".ino", ".cpp"],
                "keywords": ["ESP32", "esp_err_t", "CONFIG_"],
            },
            IoTPlatform.RASPBERRY_PI: {
                "imports": ["RPi.GPIO", "gpiozero", "picamera"],
                "functions": ["GPIO.setup", "GPIO.output"],
                "file_extensions": [".py"],
                "keywords": ["raspberry", "raspbian", "/dev/"],
            },
            IoTPlatform.MQTT: {
                "imports": ["paho.mqtt", "mosquitto", "aws-iot-device-sdk"],
                "functions": ["client.connect", "client.publish", "client.subscribe"],
                "keywords": ["mqtt", "broker", "QoS", "topic"],
            },
            IoTPlatform.AWS_IOT: {
                "imports": ["boto3", "AWSIoTPythonSDK", "aws-iot-device-sdk"],
                "keywords": ["thing", "shadow", "greengrass", "aws-iot"],
            },
            IoTPlatform.EDGE_COMPUTING: {
                "imports": ["tensorflow.lite", "onnxruntime", "edge-impulse"],
                "keywords": ["inference", "edge", "local processing", "fog computing"],
            },
        }

    def _initialize_resource_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Initialize resource usage thresholds"""
        return {
            "arduino": {
                "memory_critical": 0.90,
                "memory_warning": 0.80,
                "cpu_critical": 0.95,
                "cpu_warning": 0.85,
            },
            "esp32": {
                "memory_critical": 0.85,
                "memory_warning": 0.75,
                "cpu_critical": 0.90,
                "cpu_warning": 0.80,
                "temperature_critical": 80.0,  # Celsius
                "temperature_warning": 70.0,
            },
            "raspberry_pi": {
                "memory_critical": 0.90,
                "memory_warning": 0.80,
                "cpu_critical": 0.85,
                "cpu_warning": 0.75,
                "disk_critical": 0.95,
                "disk_warning": 0.85,
                "temperature_critical": 85.0,
                "temperature_warning": 75.0,
            },
            "edge_server": {
                "memory_critical": 0.95,
                "memory_warning": 0.85,
                "cpu_critical": 0.90,
                "cpu_warning": 0.80,
                "disk_critical": 0.90,
                "disk_warning": 0.80,
            },
        }

    def _initialize_protocol_handlers(self) -> Dict[str, Dict]:
        """Initialize protocol-specific handlers"""
        return {
            "mqtt": {
                "qos_levels": [0, 1, 2],
                "default_port": 1883,
                "secure_port": 8883,
                "keep_alive": 60,
                "reconnect_delay": [1, 2, 4, 8, 16, 30],
            },
            "coap": {
                "default_port": 5683,
                "secure_port": 5684,
                "methods": ["GET", "POST", "PUT", "DELETE"],
                "content_formats": [
                    "text/plain",
                    "application/json",
                    "application/cbor",
                ],
            },
            "lorawan": {
                "classes": ["A", "B", "C"],
                "data_rates": list(range(6)),
                "spreading_factors": [7, 8, 9, 10, 11, 12],
                "bandwidth": [125, 250, 500],
            },
            "http": {
                "methods": ["GET", "POST", "PUT", "DELETE", "PATCH"],
                "timeouts": {"connect": 5, "read": 30},
                "retry_count": 3,
            },
        }

    def detect_platform(self, code_content: str, file_path: str) -> IoTPlatform:
        """Detect which IoT platform is being used"""
        # Check file extensions first
        if file_path:
            if file_path.endswith(".ino"):
                # Could be Arduino or ESP32
                if any(
                    esp_keyword in code_content
                    for esp_keyword in ["ESP32", "WiFi.h", "<WiFi.h>", "esp_"]
                ):
                    return IoTPlatform.ESP32
                return IoTPlatform.ARDUINO

        # Check for platform-specific patterns
        for platform, detector in self.platform_detectors.items():
            score = 0

            # Check includes/imports
            for include in detector.get("includes", []) + detector.get("imports", []):
                if include in code_content:
                    score += 2

            # Check functions
            for func in detector.get("functions", []):
                if func in code_content:
                    score += 1

            # Check keywords
            for keyword in detector.get("keywords", []):
                if keyword in code_content:
                    score += 1

            if score >= 3:
                return platform

        # Check for protocol-specific code
        if any(
            proto in code_content.lower() for proto in ["mqtt", "mosquitto", "paho"]
        ):
            return IoTPlatform.MQTT

        if "coap" in code_content.lower():
            return IoTPlatform.COAP

        return IoTPlatform.UNKNOWN

    def analyze_iot_error(
        self,
        error_message: str,
        code_content: str,
        file_path: str,
        device_metrics: Optional[DeviceMetrics] = None,
    ) -> Optional[IoTError]:
        """Analyze error and determine IoT-specific issues"""
        platform = self.detect_platform(code_content, file_path)

        if platform == IoTPlatform.UNKNOWN:
            # Check resource constraints first if metrics provided
            if device_metrics:
                resource_error = self._check_resource_constraints(
                    platform, device_metrics
                )
                if resource_error:
                    return resource_error
            # Try generic IoT error detection
            return self._check_generic_iot_errors(
                error_message, device_metrics, platform
            )

        # Check platform-specific patterns
        platform_patterns = self.error_patterns.get(platform.value, [])

        for pattern_info in platform_patterns:
            if re.search(pattern_info["pattern"], error_message, re.IGNORECASE):
                return IoTError(
                    error_type=pattern_info["type"],
                    platform=platform,
                    description=pattern_info["description"],
                    suggested_fix=pattern_info.get("fix"),
                    confidence=0.9,
                    timestamp=datetime.now(),
                )

        # Check resource constraints if metrics provided
        if device_metrics:
            resource_error = self._check_resource_constraints(platform, device_metrics)
            if resource_error:
                return resource_error

        return self._check_generic_iot_errors(error_message, device_metrics, platform)

    def _check_generic_iot_errors(
        self,
        error_message: str,
        device_metrics: Optional[DeviceMetrics] = None,
        platform: IoTPlatform = IoTPlatform.UNKNOWN,
    ) -> Optional[IoTError]:
        """Check for generic IoT errors"""
        generic_patterns = {
            r"connection.*lost|network.*unreachable|WiFi.*timeout": IoTErrorType.CONNECTIVITY_ERROR,
            r"sensor.*fail|reading.*invalid|measurement.*error": IoTErrorType.SENSOR_FAILURE,
            r"actuator.*fail|control.*error|output.*fail": IoTErrorType.ACTUATOR_FAILURE,
            r"memory.*full|heap.*exhausted|stack.*overflow|out.*of.*memory": IoTErrorType.MEMORY_OVERFLOW,
            r"power.*fail|battery.*low|voltage.*drop": IoTErrorType.POWER_MANAGEMENT,
            r"sync.*fail|time.*drift|clock.*error": IoTErrorType.SYNCHRONIZATION_ERROR,
            r"security.*breach|unauthorized|certificate.*invalid": IoTErrorType.SECURITY_BREACH,
            r"firmware.*corrupt|update.*fail|boot.*error|firmware.*verification.*fail": IoTErrorType.FIRMWARE_ERROR,
            r"model.*inference|edge.*processing|ml.*timeout": IoTErrorType.EDGE_PROCESSING_ERROR,
            r"system.*overload|resource.*exhausted|cpu.*high|memory.*high": IoTErrorType.RESOURCE_CONSTRAINT,
        }

        # Define severity levels for different error types
        severity_map = {
            IoTErrorType.SECURITY_BREACH: "critical",
            IoTErrorType.FIRMWARE_ERROR: "critical",
            IoTErrorType.RESOURCE_CONSTRAINT: "critical",
            IoTErrorType.MEMORY_OVERFLOW: "high",
            IoTErrorType.POWER_MANAGEMENT: "high",
            IoTErrorType.EDGE_PROCESSING_ERROR: "medium",
            IoTErrorType.CONNECTIVITY_ERROR: "medium",
            IoTErrorType.SENSOR_FAILURE: "medium",
            IoTErrorType.ACTUATOR_FAILURE: "medium",
            IoTErrorType.SYNCHRONIZATION_ERROR: "low",
        }

        for pattern, error_type in generic_patterns.items():
            if re.search(pattern, error_message, re.IGNORECASE):
                return IoTError(
                    error_type=error_type,
                    platform=platform,
                    description=f"Generic {error_type.value} detected",
                    severity=severity_map.get(error_type, "medium"),
                    confidence=0.7,
                    timestamp=datetime.now(),
                )

        return None

    def _check_resource_constraints(
        self, platform: IoTPlatform, metrics: DeviceMetrics
    ) -> Optional[IoTError]:
        """Check for resource constraint violations"""
        platform_name = platform.value
        thresholds = self.resource_thresholds.get(
            platform_name, self.resource_thresholds.get("edge_server", {})
        )

        # Check memory
        if metrics.memory_usage > thresholds.get("memory_critical", 0.95):
            return IoTError(
                error_type=IoTErrorType.RESOURCE_CONSTRAINT,
                platform=platform,
                description="Critical memory usage detected",
                resource_usage={"memory": metrics.memory_usage},
                suggested_fix="Optimize memory usage or increase device memory",
                severity="critical",
                confidence=0.95,
                timestamp=datetime.now(),
            )

        # Check CPU
        if metrics.cpu_usage > thresholds.get("cpu_critical", 0.90):
            return IoTError(
                error_type=IoTErrorType.RESOURCE_CONSTRAINT,
                platform=platform,
                description="Critical CPU usage detected",
                resource_usage={"cpu": metrics.cpu_usage},
                suggested_fix="Optimize processing or scale resources",
                severity="critical",
                confidence=0.9,
                timestamp=datetime.now(),
            )

        # Check temperature if available
        if metrics.temperature and metrics.temperature > thresholds.get(
            "temperature_critical", 85.0
        ):
            return IoTError(
                error_type=IoTErrorType.POWER_MANAGEMENT,
                platform=platform,
                description="Device overheating detected",
                resource_usage={"temperature": metrics.temperature},
                suggested_fix="Improve cooling or reduce workload",
                severity="critical",
                confidence=0.9,
                timestamp=datetime.now(),
            )

        return None

    def suggest_healing(self, iot_error: IoTError) -> List[Dict[str, Any]]:
        """Suggest healing strategies for IoT error"""
        strategies = self.healing_strategies.get(iot_error.error_type, [])

        applicable_strategies = []
        for strategy in strategies:
            platforms = strategy["applicable_platforms"]
            if iot_error.platform.value in platforms or "all" in platforms:
                applicable_strategies.append(strategy)

        return applicable_strategies

    def generate_healing_code(
        self, iot_error: IoTError, strategy: Dict[str, Any]
    ) -> Optional[str]:
        """Generate code for implementing healing strategy"""
        if iot_error.platform == IoTPlatform.ARDUINO:
            return self._generate_arduino_healing(iot_error, strategy)
        elif iot_error.platform == IoTPlatform.ESP32:
            return self._generate_esp32_healing(iot_error, strategy)
        elif iot_error.platform == IoTPlatform.RASPBERRY_PI:
            return self._generate_raspberrypi_healing(iot_error, strategy)
        elif iot_error.platform == IoTPlatform.MQTT:
            return self._generate_mqtt_healing(iot_error, strategy)

        return self._generate_generic_iot_healing(iot_error, strategy)

    def _generate_arduino_healing(
        self, error: IoTError, strategy: Dict[str, Any]
    ) -> Optional[str]:
        """Generate Arduino-specific healing code"""
        healing_templates = {
            "resource_optimization": """
// Memory optimization for Arduino
#include <avr/pgmspace.h>

// Store constants in program memory
const char errorMsg[] PROGMEM = "Error occurred";
const int sensorData[] PROGMEM = {100, 200, 300, 400, 500};

// Use F() macro for string literals
void logMessage() {
    Serial.println(F("This string is stored in flash"));
}

// Free memory check
int freeMemory() {
    extern int __heap_start, *__brkval;
    int v;
    return (int) &v - (__brkval == 0 ? (int) &__heap_start : (int) __brkval);
}

// Periodic memory monitoring
void checkMemory() {
    int free = freeMemory();
    if (free < 100) {
        Serial.println(F("WARNING: Low memory!"));
        // Trigger cleanup or reset
    }
}
""",
            "sensor_redundancy": """
// Sensor redundancy implementation
#define NUM_SENSORS 3
#define SENSOR_PINS {A0, A1, A2}

int sensorPins[] = SENSOR_PINS;
bool sensorStatus[NUM_SENSORS] = {true, true, true};

float readSensorWithRedundancy() {
    float readings[NUM_SENSORS];
    int validReadings = 0;
    float sum = 0;
    
    // Read from all sensors
    for (int i = 0; i < NUM_SENSORS; i++) {
        if (sensorStatus[i]) {
            int raw = analogRead(sensorPins[i]);
            
            // Validate reading
            if (raw >= 0 && raw <= 1023) {
                readings[validReadings] = raw * (5.0 / 1023.0);
                sum += readings[validReadings];
                validReadings++;
            } else {
                sensorStatus[i] = false;
                Serial.print(F("Sensor "));
                Serial.print(i);
                Serial.println(F(" failed"));
            }
        }
    }
    
    // Return average of valid readings
    if (validReadings > 0) {
        return sum / validReadings;
    } else {
        // All sensors failed - return last known good value
        return lastKnownGoodValue;
    }
}
""",
            "power_saving_mode": """
// Power saving for Arduino
#include <avr/sleep.h>
#include <avr/power.h>
#include <avr/wdt.h>

// Watchdog interrupt
ISR(WDT_vect) {
    // Wake up by watchdog interrupt
}

void enterSleep() {
    // Disable ADC
    ADCSRA &= ~(1 << ADEN);
    
    // Clear various "reset" flags
    MCUSR = 0;
    
    // Allow changes, disable reset
    WDTCSR = bit(WDCE) | bit(WDE);
    
    // Set interrupt mode and an interval
    WDTCSR = bit(WDIE) | bit(WDP3) | bit(WDP0); // 8 seconds
    
    wdt_reset(); // Reset the WDT
    
    set_sleep_mode(SLEEP_MODE_PWR_DOWN);
    sleep_enable();
    
    // Turn off brown-out enable in software
    MCUCR = bit(BODS) | bit(BODSE);
    MCUCR = bit(BODS);
    
    // Sleep
    sleep_cpu();
    
    // Resume here after wakeup
    sleep_disable();
    
    // Re-enable ADC
    ADCSRA |= (1 << ADEN);
}

void setup() {
    // Disable unused peripherals
    power_adc_disable();
    power_spi_disable();
    power_timer1_disable();
    power_timer2_disable();
    power_twi_disable();
}
""",
        }

        strategy_name = strategy.get("name")
        return healing_templates.get(strategy_name)

    def _generate_esp32_healing(
        self, error: IoTError, strategy: Dict[str, Any]
    ) -> Optional[str]:
        """Generate ESP32-specific healing code"""
        healing_templates = {
            "auto_reconnect": """
// Auto-reconnect for ESP32 WiFi
#include <WiFi.h>

const char* ssid = "your-ssid";
const char* password = "your-password";

unsigned long previousMillis = 0;
unsigned long interval = 30000; // Check every 30 seconds

void setupWiFi() {
    WiFi.mode(WIFI_STA);
    WiFi.begin(ssid, password);
    WiFi.setAutoReconnect(true);
    WiFi.persistent(true);
}

void checkWiFiConnection() {
    unsigned long currentMillis = millis();
    
    // Only check periodically
    if (currentMillis - previousMillis >= interval) {
        previousMillis = currentMillis;
        
        if (WiFi.status() != WL_CONNECTED) {
            Serial.println("WiFi lost connection. Attempting reconnect...");
            WiFi.disconnect();
            WiFi.reconnect();
            
            // Wait with timeout
            int attempts = 0;
            while (WiFi.status() != WL_CONNECTED && attempts < 20) {
                delay(500);
                Serial.print(".");
                attempts++;
            }
            
            if (WiFi.status() == WL_CONNECTED) {
                Serial.println("\\nReconnected to WiFi");
                Serial.println(WiFi.localIP());
            } else {
                Serial.println("\\nFailed to reconnect");
                // Consider deep sleep or alternative connection
                enterDeepSleep();
            }
        }
    }
}

void enterDeepSleep() {
    Serial.println("Entering deep sleep for 5 minutes");
    esp_sleep_enable_timer_wakeup(5 * 60 * 1000000); // 5 minutes
    esp_deep_sleep_start();
}
""",
            "power_saving_mode": """
// ESP32 power management
#include "esp_pm.h"
#include "esp_wifi.h"
#include "driver/rtc_io.h"

void configurePowerManagement() {
    // Configure dynamic frequency scaling
    esp_pm_config_esp32_t pm_config = {
        .max_freq_mhz = 240,
        .min_freq_mhz = 10,
        .light_sleep_enable = true
    };
    esp_pm_configure(&pm_config);
    
    // Configure WiFi power saving
    esp_wifi_set_ps(WIFI_PS_MAX_MODEM);
    
    // Disable Bluetooth if not needed
    btStop();
    
    // Configure GPIO for low power
    gpio_deep_sleep_hold_en();
}

// Task with power-aware delays
void sensorTask(void *pvParameters) {
    while (1) {
        // Read sensor
        float value = readSensor();
        
        // Process data
        processData(value);
        
        // Use vTaskDelay for FreeRTOS-aware delay
        // This allows CPU to sleep
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}

// Deep sleep with external wake
void configureWakeupSources() {
    // Wake on GPIO
    esp_sleep_enable_ext0_wakeup(GPIO_NUM_33, 1); // Wake on HIGH
    
    // Wake on timer
    esp_sleep_enable_timer_wakeup(60 * 1000000); // 60 seconds
    
    // Wake on touch
    touchAttachInterrupt(T0, touchCallback, 40);
    esp_sleep_enable_touchpad_wakeup();
}
""",
            "edge_offloading": """
// Edge computation offloading for ESP32
#include <HTTPClient.h>
#include <ArduinoJson.h>

const char* edgeServer = "http://edge-server.local:8080";
const int LOCAL_THRESHOLD = 100; // Process locally if less than threshold

struct ComputeTask {
    float* data;
    size_t dataSize;
    int complexity;
};

bool shouldOffload(ComputeTask& task) {
    // Check available memory
    size_t freeHeap = ESP.getFreeHeap();
    size_t requiredMemory = task.dataSize * sizeof(float) * 2; // Conservative estimate
    
    if (freeHeap < requiredMemory + 10000) { // Keep 10KB buffer
        return true; // Offload due to memory constraints
    }
    
    // Check complexity
    if (task.complexity > LOCAL_THRESHOLD) {
        return true; // Offload complex tasks
    }
    
    return false;
}

String offloadComputation(ComputeTask& task) {
    HTTPClient http;
    http.begin(edgeServer + String("/compute"));
    http.addHeader("Content-Type", "application/json");
    
    // Prepare JSON payload
    StaticJsonDocument<1024> doc;
    JsonArray dataArray = doc.createNestedArray("data");
    
    for (size_t i = 0; i < min(task.dataSize, 50); i++) {
        dataArray.add(task.data[i]);
    }
    
    doc["complexity"] = task.complexity;
    
    String requestBody;
    serializeJson(doc, requestBody);
    
    int httpCode = http.POST(requestBody);
    String response = "";
    
    if (httpCode == HTTP_CODE_OK) {
        response = http.getString();
    } else {
        Serial.printf("Offload failed: %d\\n", httpCode);
        // Fall back to local processing
        response = processLocally(task);
    }
    
    http.end();
    return response;
}

String processLocally(ComputeTask& task) {
    // Simple local processing
    float sum = 0;
    for (size_t i = 0; i < task.dataSize; i++) {
        sum += task.data[i];
    }
    return String(sum / task.dataSize);
}
""",
        }

        strategy_name = strategy.get("name")
        return healing_templates.get(strategy_name)

    def _generate_raspberrypi_healing(
        self, error: IoTError, strategy: Dict[str, Any]
    ) -> Optional[str]:
        """Generate Raspberry Pi specific healing code"""
        healing_templates = {
            "resource_optimization": '''
# Resource optimization for Raspberry Pi
import os
import psutil
import gc
import resource

class ResourceMonitor:
    def __init__(self, memory_threshold=80, cpu_threshold=85):
        self.memory_threshold = memory_threshold
        self.cpu_threshold = cpu_threshold
        self.last_gc_time = time.time()
    
    def check_resources(self):
        """Monitor and optimize resource usage"""
        # Memory check
        memory = psutil.virtual_memory()
        if memory.percent > self.memory_threshold:
            self.optimize_memory()
        
        # CPU check
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > self.cpu_threshold:
            self.optimize_cpu()
        
        # Disk check
        disk = psutil.disk_usage('/')
        if disk.percent > 90:
            self.cleanup_disk()
    
    def optimize_memory(self):
        """Free up memory"""
        print(f"Memory usage high: {psutil.virtual_memory().percent}%")
        
        # Force garbage collection
        gc.collect()
        
        # Clear caches
        if hasattr(os, 'sync'):
            os.sync()
        
        # Drop caches (requires sudo)
        try:
            os.system('sudo sh -c "echo 3 > /proc/sys/vm/drop_caches"')
        except:
            pass
        
        print(f"Memory after optimization: {psutil.virtual_memory().percent}%")
    
    def optimize_cpu(self):
        """Reduce CPU usage"""
        # Lower process priority
        os.nice(10)
        
        # Enable CPU frequency scaling
        try:
            os.system('echo "powersave" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor')
        except:
            pass
    
    def cleanup_disk(self):
        """Clean up disk space"""
        # Remove old log files
        log_dirs = ['/var/log', '/tmp']
        for log_dir in log_dirs:
            os.system(f'sudo find {log_dir} -type f -mtime +7 -delete')

# Usage
monitor = ResourceMonitor()

def resource_aware_task(func):
    """Decorator for resource-aware execution"""
    def wrapper(*args, **kwargs):
        monitor.check_resources()
        result = func(*args, **kwargs)
        return result
    return wrapper
''',
            "sensor_recalibration": '''
# Automatic sensor calibration for Raspberry Pi
import numpy as np
import time
from collections import deque
import RPi.GPIO as GPIO

class SensorCalibrator:
    def __init__(self, sensor_pin, window_size=100):
        self.sensor_pin = sensor_pin
        self.window_size = window_size
        self.baseline_samples = deque(maxlen=window_size)
        self.calibration_offset = 0
        self.calibration_scale = 1.0
        
    def collect_baseline(self, duration=10):
        """Collect baseline readings for calibration"""
        print(f"Collecting baseline for {duration} seconds...")
        start_time = time.time()
        
        while time.time() - start_time < duration:
            reading = self.read_raw_sensor()
            self.baseline_samples.append(reading)
            time.sleep(0.1)
        
        # Calculate statistics
        baseline_array = np.array(self.baseline_samples)
        self.baseline_mean = np.mean(baseline_array)
        self.baseline_std = np.std(baseline_array)
        
        print(f"Baseline: mean={self.baseline_mean:.2f}, std={self.baseline_std:.2f}")
    
    def auto_calibrate(self, reference_value=None):
        """Automatically calibrate sensor"""
        current_readings = []
        
        # Collect current readings
        for _ in range(20):
            current_readings.append(self.read_raw_sensor())
            time.sleep(0.05)
        
        current_mean = np.mean(current_readings)
        
        if reference_value is not None:
            # Calibrate to match reference
            self.calibration_offset = reference_value - current_mean
            print(f"Calibrated with offset: {self.calibration_offset:.2f}")
        else:
            # Self-calibration based on drift detection
            drift = current_mean - self.baseline_mean
            if abs(drift) > 3 * self.baseline_std:
                self.calibration_offset = -drift
                print(f"Drift detected and corrected: {drift:.2f}")
    
    def read_calibrated(self):
        """Read calibrated sensor value"""
        raw_value = self.read_raw_sensor()
        calibrated = (raw_value + self.calibration_offset) * self.calibration_scale
        
        # Detect anomalies
        if abs(raw_value - self.baseline_mean) > 5 * self.baseline_std:
            print(f"Anomaly detected: {raw_value}")
            # Trigger recalibration if needed
            
        return calibrated
    
    def read_raw_sensor(self):
        """Read raw sensor value (implement based on sensor type)"""
        # Example for analog sensor via ADC
        # return adc.read_voltage(self.sensor_pin)
        pass

# Usage
calibrator = SensorCalibrator(sensor_pin=0)
calibrator.collect_baseline()
calibrator.auto_calibrate()
''',
        }

        strategy_name = strategy.get("name")
        return healing_templates.get(strategy_name)

    def _generate_mqtt_healing(
        self, error: IoTError, strategy: Dict[str, Any]
    ) -> Optional[str]:
        """Generate MQTT-specific healing code"""
        healing_templates = {
            "auto_reconnect": '''
# MQTT auto-reconnect with exponential backoff
import paho.mqtt.client as mqtt
import time
import threading

class ResilientMQTTClient:
    def __init__(self, client_id, broker_host, broker_port=1883):
        self.client_id = client_id
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.client = mqtt.Client(client_id)
        self.is_connected = False
        self.reconnect_delay = 1
        self.max_reconnect_delay = 300  # 5 minutes
        self.reconnect_thread = None
        
        # Configure callbacks
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect
        self.client.on_message = self.on_message
        
        # Message queue for offline storage
        self.offline_queue = []
        self.max_queue_size = 1000
    
    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print(f"Connected to MQTT broker at {self.broker_host}")
            self.is_connected = True
            self.reconnect_delay = 1  # Reset delay
            
            # Resubscribe to topics
            self.resubscribe()
            
            # Publish queued messages
            self.publish_offline_queue()
        else:
            print(f"Connection failed with code {rc}")
            self.is_connected = False
    
    def on_disconnect(self, client, userdata, rc):
        self.is_connected = False
        if rc != 0:
            print(f"Unexpected disconnection (rc={rc}). Will reconnect...")
            self.start_reconnect_thread()
    
    def start_reconnect_thread(self):
        if self.reconnect_thread is None or not self.reconnect_thread.is_alive():
            self.reconnect_thread = threading.Thread(target=self.reconnect_loop)
            self.reconnect_thread.daemon = True
            self.reconnect_thread.start()
    
    def reconnect_loop(self):
        while not self.is_connected:
            try:
                print(f"Attempting reconnection in {self.reconnect_delay}s...")
                time.sleep(self.reconnect_delay)
                
                self.client.reconnect()
                
                # Wait a bit to see if connection succeeds
                time.sleep(2)
                
                if not self.is_connected:
                    # Exponential backoff
                    self.reconnect_delay = min(
                        self.reconnect_delay * 2,
                        self.max_reconnect_delay
                    )
            except Exception as e:
                print(f"Reconnection failed: {e}")
    
    def publish(self, topic, payload, qos=1, retain=False):
        """Publish with offline queueing"""
        if self.is_connected:
            try:
                result = self.client.publish(topic, payload, qos, retain)
                return result
            except Exception as e:
                print(f"Publish failed: {e}")
                self.queue_message(topic, payload, qos, retain)
        else:
            self.queue_message(topic, payload, qos, retain)
    
    def queue_message(self, topic, payload, qos, retain):
        """Queue messages when offline"""
        if len(self.offline_queue) < self.max_queue_size:
            self.offline_queue.append({
                'topic': topic,
                'payload': payload,
                'qos': qos,
                'retain': retain,
                'timestamp': time.time()
            })
            print(f"Message queued for {topic} (queue size: {len(self.offline_queue)})")
        else:
            print("Offline queue full! Dropping oldest message.")
            self.offline_queue.pop(0)
            self.queue_message(topic, payload, qos, retain)
    
    def publish_offline_queue(self):
        """Publish all queued messages"""
        print(f"Publishing {len(self.offline_queue)} queued messages...")
        
        while self.offline_queue and self.is_connected:
            msg = self.offline_queue.pop(0)
            try:
                self.client.publish(
                    msg['topic'],
                    msg['payload'],
                    msg['qos'],
                    msg['retain']
                )
                time.sleep(0.1)  # Rate limiting
            except Exception as e:
                print(f"Failed to publish queued message: {e}")
                # Put it back
                self.offline_queue.insert(0, msg)
                break

# Usage
mqtt_client = ResilientMQTTClient("iot_device_001", "mqtt.broker.com")
mqtt_client.connect()
''',
            "offline_mode": '''
# MQTT offline mode with store-and-forward
import json
import os
import sqlite3
from datetime import datetime

class OfflineCapableMQTT:
    def __init__(self, db_path="mqtt_offline.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database for offline storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS offline_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT NOT NULL,
                payload TEXT NOT NULL,
                qos INTEGER DEFAULT 1,
                retain INTEGER DEFAULT 0,
                timestamp REAL NOT NULL,
                attempts INTEGER DEFAULT 0,
                status TEXT DEFAULT 'pending'
            )
        """)
        
        conn.commit()
        conn.close()
    
    def store_offline_message(self, topic, payload, qos=1, retain=False):
        """Store message in offline database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO offline_messages (topic, payload, qos, retain, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (topic, json.dumps(payload), qos, int(retain), datetime.now().timestamp()))
        
        conn.commit()
        conn.close()
    
    def get_pending_messages(self, limit=100):
        """Retrieve pending messages for transmission"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, topic, payload, qos, retain
            FROM offline_messages
            WHERE status = 'pending'
            ORDER BY timestamp ASC
            LIMIT ?
        """, (limit,))
        
        messages = cursor.fetchall()
        conn.close()
        
        return messages
    
    def mark_message_sent(self, message_id):
        """Mark message as sent"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE offline_messages
            SET status = 'sent'
            WHERE id = ?
        """, (message_id,))
        
        conn.commit()
        conn.close()
    
    def cleanup_old_messages(self, days=7):
        """Remove old sent messages"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_time = (datetime.now().timestamp() - (days * 24 * 3600))
        
        cursor.execute("""
            DELETE FROM offline_messages
            WHERE status = 'sent' AND timestamp < ?
        """, (cutoff_time,))
        
        conn.commit()
        conn.close()
''',
        }

        strategy_name = strategy.get("name")
        return healing_templates.get(strategy_name)

    def _generate_generic_iot_healing(
        self, error: IoTError, strategy: Dict[str, Any]
    ) -> Optional[str]:
        """Generate generic IoT healing code"""
        healing_templates = {
            "connection_pooling": '''
# Generic connection pooling for IoT devices
import queue
import threading
import time

class ConnectionPool:
    def __init__(self, connection_factory, pool_size=5):
        self.connection_factory = connection_factory
        self.pool_size = pool_size
        self.connections = queue.Queue(maxsize=pool_size)
        self.all_connections = []
        self.lock = threading.Lock()
        
        # Initialize pool
        self.initialize_pool()
    
    def initialize_pool(self):
        """Create initial connections"""
        for _ in range(self.pool_size):
            conn = self.create_connection()
            if conn:
                self.connections.put(conn)
                self.all_connections.append(conn)
    
    def create_connection(self):
        """Create a new connection with retry logic"""
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                conn = self.connection_factory()
                # Test connection
                if hasattr(conn, 'ping'):
                    conn.ping()
                return conn
            except Exception as e:
                print(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
        
        return None
    
    def get_connection(self, timeout=10):
        """Get a connection from the pool"""
        try:
            conn = self.connections.get(timeout=timeout)
            
            # Validate connection
            if self.validate_connection(conn):
                return conn
            else:
                # Connection is bad, create a new one
                new_conn = self.create_connection()
                if new_conn:
                    with self.lock:
                        # Replace bad connection
                        self.all_connections.remove(conn)
                        self.all_connections.append(new_conn)
                    return new_conn
                else:
                    raise Exception("Failed to create replacement connection")
                    
        except queue.Empty:
            raise Exception("No connections available")
    
    def return_connection(self, conn):
        """Return a connection to the pool"""
        if self.validate_connection(conn):
            self.connections.put(conn)
        else:
            # Connection is bad, create replacement
            self.create_and_add_connection()
    
    def validate_connection(self, conn):
        """Check if connection is still valid"""
        try:
            if hasattr(conn, 'is_connected'):
                return conn.is_connected()
            elif hasattr(conn, 'ping'):
                conn.ping()
                return True
            return True
        except:
            return False

# Usage example
def mqtt_connection_factory():
    import paho.mqtt.client as mqtt
    client = mqtt.Client()
    client.connect("broker.mqtt.com", 1883, 60)
    return client

pool = ConnectionPool(mqtt_connection_factory, pool_size=3)
''',
            "adaptive_sampling": '''
# Adaptive sampling rate based on resource availability
import time
import psutil
import threading

class AdaptiveSampler:
    def __init__(self, base_rate=1.0, min_rate=0.1, max_rate=10.0):
        self.base_rate = base_rate  # Hz
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.current_rate = base_rate
        self.resource_threshold = {
            'cpu': 80,
            'memory': 80,
            'battery': 20
        }
        self.adjustment_factor = 0.1
        
    def get_sampling_interval(self):
        """Get current sampling interval in seconds"""
        return 1.0 / self.current_rate
    
    def adjust_rate(self):
        """Adjust sampling rate based on resources"""
        resources = self.check_resources()
        
        # CPU constrained - reduce rate
        if resources['cpu'] > self.resource_threshold['cpu']:
            self.current_rate *= (1 - self.adjustment_factor)
        
        # Memory constrained - reduce rate  
        elif resources['memory'] > self.resource_threshold['memory']:
            self.current_rate *= (1 - self.adjustment_factor)
        
        # Battery low - reduce rate significantly
        elif resources.get('battery', 100) < self.resource_threshold['battery']:
            self.current_rate *= 0.5
        
        # Resources available - increase rate
        else:
            self.current_rate *= (1 + self.adjustment_factor)
        
        # Enforce limits
        self.current_rate = max(self.min_rate, min(self.current_rate, self.max_rate))
        
        return self.current_rate
    
    def check_resources(self):
        """Check system resources"""
        return {
            'cpu': psutil.cpu_percent(interval=0.1),
            'memory': psutil.virtual_memory().percent,
            'battery': psutil.sensors_battery().percent if psutil.sensors_battery() else 100
        }
    
    def adaptive_sampling_loop(self, sensor_func, callback):
        """Main sampling loop with adaptive rate"""
        while True:
            start_time = time.time()
            
            # Read sensor
            try:
                data = sensor_func()
                callback(data, self.current_rate)
            except Exception as e:
                print(f"Sensor error: {e}")
            
            # Adjust rate periodically
            if int(time.time()) % 10 == 0:
                old_rate = self.current_rate
                new_rate = self.adjust_rate()
                if abs(old_rate - new_rate) > 0.1:
                    print(f"Sampling rate adjusted: {old_rate:.1f} -> {new_rate:.1f} Hz")
            
            # Sleep for remainder of interval
            elapsed = time.time() - start_time
            sleep_time = self.get_sampling_interval() - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

# Usage
sampler = AdaptiveSampler(base_rate=5.0)

def read_sensor():
    # Simulate sensor reading
    return {'temperature': 25.5, 'humidity': 60}

def process_data(data, rate):
    print(f"Data: {data}, Rate: {rate:.1f} Hz")

# Start adaptive sampling
sampler.adaptive_sampling_loop(read_sensor, process_data)
''',
        }

        strategy_name = strategy.get("name")
        return healing_templates.get(strategy_name)

    def analyze_device_health(
        self,
        device_id: str,
        metrics: DeviceMetrics,
        historical_data: List[DeviceMetrics] = None,
    ) -> Dict[str, Any]:
        """Analyze overall device health"""
        health_status = {
            "device_id": device_id,
            "timestamp": datetime.now(),
            "overall_health": "healthy",
            "issues": [],
            "recommendations": [],
        }

        # Check current metrics
        if metrics.memory_usage > 0.9:
            health_status["issues"].append("Critical memory usage")
            health_status["overall_health"] = "critical"
            health_status["recommendations"].append(
                "Restart device or optimize memory usage"
            )

        if metrics.cpu_usage > 0.85:
            health_status["issues"].append("High CPU usage")
            if health_status["overall_health"] == "healthy":
                health_status["overall_health"] = "warning"
            health_status["recommendations"].append("Review running processes")

        if metrics.battery_level and metrics.battery_level < 20:
            health_status["issues"].append("Low battery")
            health_status["recommendations"].append("Connect to power source")

        if metrics.temperature and metrics.temperature > 75:
            health_status["issues"].append("High temperature")
            health_status["recommendations"].append(
                "Improve ventilation or reduce load"
            )

        # Analyze historical trends if available
        if historical_data and len(historical_data) > 10:
            health_status["trends"] = self._analyze_trends(historical_data)

        return health_status

    def _analyze_trends(self, historical_data: List[DeviceMetrics]) -> Dict[str, str]:
        """Analyze trends in device metrics"""
        trends = {}

        # Simple trend analysis (could be enhanced with more sophisticated methods)
        if len(historical_data) >= 2:
            recent_cpu = [m.cpu_usage for m in historical_data[-5:]]
            older_cpu = [m.cpu_usage for m in historical_data[-10:-5]]

            if (
                sum(recent_cpu) / len(recent_cpu)
                > sum(older_cpu) / len(older_cpu) * 1.2
            ):
                trends["cpu"] = "increasing"

            recent_memory = [m.memory_usage for m in historical_data[-5:]]
            older_memory = [m.memory_usage for m in historical_data[-10:-5]]

            if (
                sum(recent_memory) / len(recent_memory)
                > sum(older_memory) / len(older_memory) * 1.1
            ):
                trends["memory"] = "increasing"

        return trends

    def generate_device_config(
        self, platform: IoTPlatform, requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate optimized device configuration"""
        config = {
            "platform": platform.value,
            "generated_at": datetime.now().isoformat(),
            "settings": {},
        }

        if platform == IoTPlatform.ESP32:
            config["settings"] = {
                "cpu_freq": 160 if requirements.get("low_power") else 240,
                "wifi_mode": (
                    "WIFI_PS_MAX_MODEM"
                    if requirements.get("low_power")
                    else "WIFI_PS_MIN_MODEM"
                ),
                "deep_sleep_enabled": requirements.get("battery_powered", False),
                "watchdog_timeout": 30000,
                "stack_size": 4096 if requirements.get("simple_task") else 8192,
            }

        elif platform == IoTPlatform.ARDUINO:
            config["settings"] = {
                "baud_rate": 9600 if requirements.get("low_power") else 115200,
                "use_sleep": requirements.get("battery_powered", False),
                "analog_reference": "DEFAULT",
                "pwm_frequency": 490,
            }

        elif platform == IoTPlatform.RASPBERRY_PI:
            config["settings"] = {
                "governor": (
                    "powersave" if requirements.get("low_power") else "performance"
                ),
                "gpu_mem_split": 16 if requirements.get("headless") else 64,
                "enable_uart": requirements.get("serial_needed", False),
                "watchdog": True,
            }

        return config
