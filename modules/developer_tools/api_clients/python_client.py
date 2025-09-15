"""
Homeostasis Python API Client

A comprehensive Python client library for interacting with the Homeostasis
self-healing framework.
"""

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union
from urllib.parse import urljoin

import requests
import websocket
from websocket import WebSocketApp

logger = logging.getLogger(__name__)


class HealingStatus(Enum):
    """Healing operation status"""

    PENDING = "pending"
    ANALYZING = "analyzing"
    GENERATING_PATCH = "generating_patch"
    TESTING = "testing"
    APPLYING = "applying"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class ErrorSeverity(Enum):
    """Error severity levels"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorReport:
    """Error report structure"""

    error_message: str
    stack_trace: str
    language: str
    framework: Optional[str] = None
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    context: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        if self.timestamp:
            data["timestamp"] = self.timestamp.isoformat()
        if self.severity:
            data["severity"] = self.severity.value
        return data


@dataclass
class HealingResult:
    """Result of a healing operation"""

    healing_id: str
    status: HealingStatus
    error_id: str
    patches_generated: int
    patches_applied: int
    success: bool
    duration_seconds: float
    logs: List[str]
    rollback_available: bool
    metrics: Dict[str, Any]


@dataclass
class SystemHealth:
    """System health status"""

    status: str
    uptime_seconds: float
    active_healings: int
    total_errors_processed: int
    success_rate: float
    average_healing_time: float
    components: Dict[str, str]


class HomeostasisClient:
    """Main API client for Homeostasis"""

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: int = 30,
        verify_ssl: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        self.session = requests.Session()
        self._setup_session()
        self.ws_client: Optional[WebSocketApp] = None
        self._callbacks: Dict[str, List[Callable]] = {}

    def _setup_session(self):
        """Configure session with authentication and headers"""
        self.session.headers.update(
            {
                "Content-Type": "application/json",
                "User-Agent": "Homeostasis-Python-Client/1.0",
            }
        )

        if self.api_key:
            self.session.headers["Authorization"] = f"Bearer {self.api_key}"

    def _make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """Make HTTP request to API"""
        url = urljoin(self.base_url, endpoint)

        kwargs.setdefault("timeout", self.timeout)
        kwargs.setdefault("verify", self.verify_ssl)

        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise

    # Core API Methods

    def report_error(self, error: Union[ErrorReport, Dict[str, Any]]) -> Dict[str, Any]:
        """Report an error for healing"""
        if isinstance(error, ErrorReport):
            data = error.to_dict()
        else:
            data = error

        response = self._make_request("POST", "/api/v1/errors", json=data)
        result: Dict[str, Any] = response.json()
        return result

    def get_healing_status(self, healing_id: str) -> HealingResult:
        """Get status of a healing operation"""
        response = self._make_request("GET", f"/api/v1/healings/{healing_id}")
        data = response.json()

        return HealingResult(
            healing_id=data["healing_id"],
            status=HealingStatus(data["status"]),
            error_id=data["error_id"],
            patches_generated=data["patches_generated"],
            patches_applied=data["patches_applied"],
            success=data["success"],
            duration_seconds=data["duration_seconds"],
            logs=data["logs"],
            rollback_available=data["rollback_available"],
            metrics=data["metrics"],
        )

    def trigger_healing(
        self, error_id: str, options: Optional[Dict[str, Any]] = None
    ) -> str:
        """Manually trigger healing for an error"""
        data: Dict[str, Any] = {"error_id": error_id}
        if options:
            data["options"] = options

        response = self._make_request("POST", "/api/v1/healings", json=data)
        result = response.json()
        return str(result["healing_id"])

    def list_errors(
        self,
        filters: Optional[Dict[str, Any]] = None,
        page: int = 1,
        per_page: int = 50,
    ) -> Dict[str, Any]:
        """List reported errors with optional filters"""
        params = {"page": page, "per_page": per_page}

        if filters:
            params.update(filters)

        response = self._make_request("GET", "/api/v1/errors", params=params)
        result: Dict[str, Any] = response.json()
        return result

    def get_system_health(self) -> SystemHealth:
        """Get system health status"""
        response = self._make_request("GET", "/api/v1/health")
        data = response.json()

        return SystemHealth(
            status=data["status"],
            uptime_seconds=data["uptime_seconds"],
            active_healings=data["active_healings"],
            total_errors_processed=data["total_errors_processed"],
            success_rate=data["success_rate"],
            average_healing_time=data["average_healing_time"],
            components=data["components"],
        )

    def get_metrics(self, metric_type: str, time_range: str = "1h") -> Dict[str, Any]:
        """Get system metrics"""
        params = {"type": metric_type, "range": time_range}

        response = self._make_request("GET", "/api/v1/metrics", params=params)
        result: Dict[str, Any] = response.json()
        return result

    def rollback_healing(self, healing_id: str, reason: Optional[str] = None) -> bool:
        """Rollback a healing operation"""
        data = {"reason": reason} if reason else {}

        response = self._make_request(
            "POST", f"/api/v1/healings/{healing_id}/rollback", json=data
        )
        result = response.json()
        return bool(result["success"])

    def approve_healing(self, healing_id: str, approved_by: str) -> bool:
        """Approve a healing operation (for systems with approval workflow)"""
        data = {"approved_by": approved_by}

        response = self._make_request(
            "POST", f"/api/v1/healings/{healing_id}/approve", json=data
        )
        result = response.json()
        return bool(result["success"])

    def get_patches(self, healing_id: str) -> List[Dict[str, Any]]:
        """Get patches generated for a healing"""
        response = self._make_request("GET", f"/api/v1/healings/{healing_id}/patches")
        result = response.json()
        patches: List[Dict[str, Any]] = result["patches"]
        return patches

    def test_patch(
        self, patch_id: str, test_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Test a specific patch"""
        data = {"test_config": test_config} if test_config else {}

        response = self._make_request(
            "POST", f"/api/v1/patches/{patch_id}/test", json=data
        )
        result: Dict[str, Any] = response.json()
        return result

    # Configuration Management

    def get_config(self, component: Optional[str] = None) -> Dict[str, Any]:
        """Get system configuration"""
        endpoint = f"/api/v1/config/{component}" if component else "/api/v1/config"
        response = self._make_request("GET", endpoint)
        result: Dict[str, Any] = response.json()
        return result

    def update_config(self, component: str, config: Dict[str, Any]) -> bool:
        """Update system configuration"""
        response = self._make_request("PUT", f"/api/v1/config/{component}", json=config)
        result = response.json()
        return bool(result["success"])

    # Rule Management

    def list_rules(
        self, language: Optional[str] = None, category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List available healing rules"""
        params = {}
        if language:
            params["language"] = language
        if category:
            params["category"] = category

        response = self._make_request("GET", "/api/v1/rules", params=params)
        result = response.json()
        rules: List[Dict[str, Any]] = result["rules"]
        return rules

    def get_rule(self, rule_id: str) -> Dict[str, Any]:
        """Get details of a specific rule"""
        response = self._make_request("GET", f"/api/v1/rules/{rule_id}")
        result: Dict[str, Any] = response.json()
        return result

    def create_custom_rule(self, rule_data: Dict[str, Any]) -> str:
        """Create a custom healing rule"""
        response = self._make_request("POST", "/api/v1/rules", json=rule_data)
        result = response.json()
        return str(result["rule_id"])

    def update_rule(self, rule_id: str, rule_data: Dict[str, Any]) -> bool:
        """Update an existing rule"""
        response = self._make_request("PUT", f"/api/v1/rules/{rule_id}", json=rule_data)
        result = response.json()
        return bool(result["success"])

    def delete_rule(self, rule_id: str) -> bool:
        """Delete a custom rule"""
        response = self._make_request("DELETE", f"/api/v1/rules/{rule_id}")
        result = response.json()
        return bool(result["success"])

    # WebSocket Support for Real-time Updates

    def connect_websocket(self, on_message=None, on_error=None, on_close=None):
        """Connect to WebSocket for real-time updates"""
        ws_url = self.base_url.replace("http://", "ws://").replace("https://", "wss://")
        ws_url = f"{ws_url}/ws"

        def _on_message(ws, message):
            data = json.loads(message)
            event_type = data.get("type")

            # Call registered callbacks
            if event_type in self._callbacks:
                for callback in self._callbacks[event_type]:
                    callback(data)

            # Call general message handler
            if on_message:
                on_message(data)

        def _on_error(ws, error):
            logger.error(f"WebSocket error: {error}")
            if on_error:
                on_error(error)

        def _on_close(ws, close_status_code, close_msg):
            logger.info(f"WebSocket connection closed: {close_status_code} - {close_msg}")
            if on_close:
                on_close()

        def _on_open(ws):
            # Authenticate
            if self.api_key:
                ws.send(json.dumps({"type": "auth", "api_key": self.api_key}))

        self.ws_client = websocket.WebSocketApp(
            ws_url,
            on_message=_on_message,
            on_error=_on_error,
            on_close=_on_close,
            on_open=_on_open,
        )

        # Run in separate thread
        ws_thread = threading.Thread(target=self.ws_client.run_forever)
        ws_thread.daemon = True
        ws_thread.start()

    def subscribe(self, event_type: str, callback):
        """Subscribe to specific WebSocket events"""
        if event_type not in self._callbacks:
            self._callbacks[event_type] = []
        self._callbacks[event_type].append(callback)

    def unsubscribe(self, event_type: str, callback):
        """Unsubscribe from WebSocket events"""
        if event_type in self._callbacks:
            self._callbacks[event_type].remove(callback)

    def disconnect_websocket(self):
        """Disconnect WebSocket"""
        if self.ws_client:
            self.ws_client.close()

    # Batch Operations

    def batch_report_errors(self, errors: List[ErrorReport]) -> Dict[str, Any]:
        """Report multiple errors in batch"""
        data = {"errors": [error.to_dict() for error in errors]}

        response = self._make_request("POST", "/api/v1/errors/batch", json=data)
        result: Dict[str, Any] = response.json()
        return result

    def batch_get_status(self, healing_ids: List[str]) -> Dict[str, HealingResult]:
        """Get status of multiple healing operations"""
        data = {"healing_ids": healing_ids}

        response = self._make_request(
            "POST", "/api/v1/healings/batch/status", json=data
        )
        results = {}

        for healing_id, data in response.json()["results"].items():
            results[healing_id] = HealingResult(
                healing_id=data["healing_id"],
                status=HealingStatus(data["status"]),
                error_id=data["error_id"],
                patches_generated=data["patches_generated"],
                patches_applied=data["patches_applied"],
                success=data["success"],
                duration_seconds=data["duration_seconds"],
                logs=data["logs"],
                rollback_available=data["rollback_available"],
                metrics=data["metrics"],
            )

        return results

    # Utility Methods

    def wait_for_healing(
        self, healing_id: str, timeout: int = 300, poll_interval: int = 5
    ) -> HealingResult:
        """Wait for a healing operation to complete"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            result = self.get_healing_status(healing_id)

            if result.status in [
                HealingStatus.COMPLETED,
                HealingStatus.FAILED,
                HealingStatus.ROLLED_BACK,
            ]:
                return result

            time.sleep(poll_interval)

        raise TimeoutError(
            f"Healing {healing_id} did not complete within {timeout} seconds"
        )

    def export_logs(self, healing_id: str, output_format: str = "json") -> str:
        """Export healing logs"""
        params = {"format": output_format}

        response = self._make_request(
            "GET", f"/api/v1/healings/{healing_id}/logs", params=params
        )

        if output_format == "json":
            return json.dumps(response.json(), indent=2)
        else:
            return response.text

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.session.close()
        if self.ws_client:
            self.disconnect_websocket()


# Convenience functions


def create_client(
    base_url: str, api_key: Optional[str] = None, **kwargs
) -> HomeostasisClient:
    """Factory function to create a client"""
    return HomeostasisClient(base_url, api_key, **kwargs)


def quick_report_error(
    base_url: str,
    error_message: str,
    stack_trace: str,
    language: str,
    api_key: Optional[str] = None,
) -> str:
    """Quick function to report an error and get healing ID"""
    with create_client(base_url, api_key) as client:
        error = ErrorReport(
            error_message=error_message,
            stack_trace=stack_trace,
            language=language,
            timestamp=datetime.now(),
        )

        result = client.report_error(error)
        return str(result["error_id"])
