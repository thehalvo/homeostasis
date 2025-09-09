"""
USHS Python Reference Implementation
Universal Self-Healing Standard v1.0 Client Library
"""

import asyncio
import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urljoin

import aiohttp
import websockets


class Severity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class SessionStatus(Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class DeploymentStrategy(Enum):
    IMMEDIATE = "immediate"
    CANARY = "canary"
    BLUE_GREEN = "blue-green"
    ROLLING = "rolling"


@dataclass
class ErrorEvent:
    """USHS Error Event"""

    severity: Severity
    service: str
    type: str
    message: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    version: Optional[str] = None
    environment: Optional[str] = None
    location: Optional[str] = None
    hostname: Optional[str] = None
    container_id: Optional[str] = None
    code: Optional[str] = None
    stack_trace: Optional[List[Dict[str, Any]]] = None
    context: Optional[Dict[str, Any]] = None
    correlation_id: Optional[str] = None
    user_impact: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to USHS-compliant dictionary"""
        data = {
            "id": self.id,
            "timestamp": self.timestamp,
            "severity": self.severity.value,
            "source": {"service": self.service},
            "error": {"type": self.type, "message": self.message},
        }

        # Add optional source fields
        if self.version:
            data["source"]["version"] = self.version
        if self.environment:
            data["source"]["environment"] = self.environment
        if self.location:
            data["source"]["location"] = self.location
        if self.hostname:
            data["source"]["hostname"] = self.hostname
        if self.container_id:
            data["source"]["containerID"] = self.container_id

        # Add optional error fields
        if self.code:
            data["error"]["code"] = self.code
        if self.stack_trace:
            data["error"]["stackTrace"] = self.stack_trace
        if self.context:
            data["error"]["context"] = self.context

        # Add optional top-level fields
        if self.correlation_id:
            data["correlationId"] = self.correlation_id
        if self.user_impact:
            data["userImpact"] = self.user_impact
        if self.metadata:
            data["metadata"] = self.metadata

        return data


@dataclass
class HealingPatch:
    """USHS Healing Patch"""

    session_id: str
    error_id: str
    changes: List[Dict[str, Any]]
    confidence: float
    generator: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    generator_version: Optional[str] = None
    strategy: Optional[str] = None
    reasoning: Optional[str] = None
    alternatives: Optional[List[Dict[str, Any]]] = None
    estimated_impact: Optional[Dict[str, Any]] = None
    approvals: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to USHS-compliant dictionary"""
        data = {
            "id": self.id,
            "sessionId": self.session_id,
            "errorId": self.error_id,
            "changes": self.changes,
            "metadata": {"confidence": self.confidence, "generator": self.generator},
        }

        # Add optional metadata fields
        if self.generator_version:
            data["metadata"]["generatorVersion"] = self.generator_version
        if self.strategy:
            data["metadata"]["strategy"] = self.strategy
        if self.reasoning:
            data["metadata"]["reasoning"] = self.reasoning
        if self.alternatives:
            data["metadata"]["alternatives"] = self.alternatives
        if self.estimated_impact:
            data["metadata"]["estimatedImpact"] = self.estimated_impact

        # Add optional top-level fields
        if self.approvals:
            data["approvals"] = self.approvals

        return data


class USHSClient:
    """USHS Reference Client Implementation"""

    def __init__(
        self,
        base_url: str,
        auth_token: Optional[str] = None,
        api_key: Optional[str] = None,
        verify_ssl: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.auth_token = auth_token
        self.api_key = api_key
        self.verify_ssl = verify_ssl
        self.logger = logging.getLogger(__name__)
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._ws_handlers: Dict[str, List[Callable]] = {}

    async def __aenter__(self):
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def _ensure_session(self):
        """Ensure HTTP session exists"""
        if not self._session:
            headers = {}
            if self.auth_token:
                headers["Authorization"] = f"Bearer {self.auth_token}"
            elif self.api_key:
                headers["X-API-Key"] = self.api_key

            connector = aiohttp.TCPConnector(ssl=self.verify_ssl)
            self._session = aiohttp.ClientSession(headers=headers, connector=connector)

    async def close(self):
        """Close all connections"""
        if self._ws:
            await self._ws.close()
            self._ws = None
        if self._session:
            await self._session.close()
            self._session = None

    # Error Management

    async def report_error(self, error: ErrorEvent) -> Dict[str, str]:
        """Report a new error to the healing system"""
        await self._ensure_session()

        url = urljoin(self.base_url, "/errors")
        async with self._session.post(url, json=error.to_dict()) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def get_error(self, error_id: str) -> Dict[str, Any]:
        """Get error details"""
        await self._ensure_session()

        url = urljoin(self.base_url, f"/errors/{error_id}")
        async with self._session.get(url) as resp:
            resp.raise_for_status()
            return await resp.json()

    # Session Management

    async def start_session(
        self,
        error_id: str,
        policy: Optional[str] = None,
        priority: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Start a new healing session"""
        await self._ensure_session()

        data = {"errorId": error_id}
        if policy:
            data["policy"] = policy
        if priority:
            data["priority"] = priority

        url = urljoin(self.base_url, "/sessions")
        async with self._session.post(url, json=data) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def get_session(self, session_id: str) -> Dict[str, Any]:
        """Get session details"""
        await self._ensure_session()

        url = urljoin(self.base_url, f"/sessions/{session_id}")
        async with self._session.get(url) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def list_sessions(
        self, status: Optional[SessionStatus] = None, limit: int = 20, offset: int = 0
    ) -> Dict[str, Any]:
        """List healing sessions"""
        await self._ensure_session()

        params = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status.value

        url = urljoin(self.base_url, "/sessions")
        async with self._session.get(url, params=params) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def cancel_session(self, session_id: str) -> None:
        """Cancel a healing session"""
        await self._ensure_session()

        url = urljoin(self.base_url, f"/sessions/{session_id}")
        async with self._session.delete(url) as resp:
            resp.raise_for_status()

    # Patch Management

    async def get_session_patches(self, session_id: str) -> List[Dict[str, Any]]:
        """Get patches for a session"""
        await self._ensure_session()

        url = urljoin(self.base_url, f"/sessions/{session_id}/patches")
        async with self._session.get(url) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def submit_patch(
        self, session_id: str, patch: HealingPatch
    ) -> Dict[str, Any]:
        """Submit a patch for a session"""
        await self._ensure_session()

        url = urljoin(self.base_url, f"/sessions/{session_id}/patches")
        async with self._session.post(url, json=patch.to_dict()) as resp:
            resp.raise_for_status()
            return await resp.json()

    # Validation

    async def validate_patch(
        self,
        patch_id: str,
        tests: Optional[List[str]] = None,
        environment: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Validate a patch"""
        await self._ensure_session()

        data = {}
        if tests:
            data["tests"] = tests
        if environment:
            data["environment"] = environment

        url = urljoin(self.base_url, f"/patches/{patch_id}/validate")
        async with self._session.post(url, json=data) as resp:
            resp.raise_for_status()
            return await resp.json()

    # Deployment

    async def deploy_patch(
        self,
        patch_id: str,
        strategy: DeploymentStrategy,
        environment: str,
        approvals: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Deploy a patch"""
        await self._ensure_session()

        data = {"strategy": strategy.value, "environment": environment}
        if approvals:
            data["approvals"] = approvals

        url = urljoin(self.base_url, f"/patches/{patch_id}/deploy")
        async with self._session.post(url, json=data) as resp:
            resp.raise_for_status()
            return await resp.json()

    # WebSocket Support

    async def connect_websocket(
        self,
        subscribe: Optional[List[str]] = None,
        session: Optional[str] = None,
        service: Optional[str] = None,
    ):
        """Connect to WebSocket for real-time events"""
        ws_url = self.base_url.replace("https://", "wss://").replace("http://", "ws://")
        ws_url = urljoin(ws_url, "/ws")

        # Build query params
        params = []
        if subscribe:
            params.append(f"subscribe={','.join(subscribe)}")
        if session:
            params.append(f"session={session}")
        if service:
            params.append(f"service={service}")
        if self.api_key:
            params.append(f"apikey={self.api_key}")

        if params:
            ws_url += "?" + "&".join(params)

        # Build headers
        headers = {}
        if self.auth_token and not self.api_key:
            headers["Authorization"] = f"Bearer {self.auth_token}"

        self._ws = await websockets.connect(ws_url, extra_headers=headers)

        # Start message handler
        asyncio.create_task(self._handle_ws_messages())

    async def _handle_ws_messages(self):
        """Handle incoming WebSocket messages"""
        try:
            async for message in self._ws:
                try:
                    event = json.loads(message)
                    event_type = event.get("type", "")

                    # Call registered handlers
                    handlers = self._ws_handlers.get(event_type, [])
                    handlers.extend(self._ws_handlers.get("*", []))

                    for handler in handlers:
                        try:
                            await handler(event)
                        except Exception as e:
                            self.logger.error(f"Error in WebSocket handler: {e}")

                except json.JSONDecodeError:
                    self.logger.error(f"Invalid WebSocket message: {message}")

        except websockets.exceptions.ConnectionClosed:
            self.logger.info("WebSocket connection closed")
        except Exception as e:
            self.logger.error(f"WebSocket error: {e}")

    def on_event(self, event_type: str):
        """Decorator to register WebSocket event handlers"""

        def decorator(func):
            if event_type not in self._ws_handlers:
                self._ws_handlers[event_type] = []
            self._ws_handlers[event_type].append(func)
            return func

        return decorator

    async def subscribe(
        self, event_types: List[str], filters: Optional[Dict[str, Any]] = None
    ):
        """Subscribe to additional event types"""
        if not self._ws:
            raise RuntimeError("WebSocket not connected")

        command = {"command": "subscribe", "eventTypes": event_types}
        if filters:
            command["filters"] = filters

        await self._ws.send(json.dumps(command))

    async def unsubscribe(self, event_types: List[str]):
        """Unsubscribe from event types"""
        if not self._ws:
            raise RuntimeError("WebSocket not connected")

        command = {"command": "unsubscribe", "eventTypes": event_types}

        await self._ws.send(json.dumps(command))

    # Health Check

    async def health_check(self) -> Dict[str, Any]:
        """Check system health"""
        await self._ensure_session()

        url = urljoin(self.base_url, "/health")
        async with self._session.get(url) as resp:
            resp.raise_for_status()
            return await resp.json()


# Example usage
async def example_usage():
    """Example of using the USHS client"""

    # Create client
    async with USHSClient(
        base_url="https://api.example.com/ushs/v1", auth_token="your-auth-token"
    ) as client:

        # Report an error
        error = ErrorEvent(
            severity=Severity.HIGH,
            service="api-gateway",
            type="NullPointerException",
            message="Cannot read property 'id' of null",
            environment="production",
            stack_trace=[{"file": "app.js", "function": "getUser", "line": 42}],
        )

        result = await client.report_error(error)
        error_id = result["errorId"]
        session_id = result["sessionId"]

        # Check session status
        session = await client.get_session(session_id)
        print(f"Session status: {session['status']}")

        # Connect WebSocket for real-time updates
        await client.connect_websocket(subscribe=["session", "patch"])

        # Register event handlers
        @client.on_event("org.ushs.patch.generated")
        async def on_patch_generated(event):
            print(f"Patch generated: {event['data']['patchId']}")

        @client.on_event("org.ushs.session.completed")
        async def on_session_completed(event):
            print(f"Session completed: {event['data']['sessionId']}")

        # Wait for healing to complete
        await asyncio.sleep(30)


if __name__ == "__main__":
    asyncio.run(example_usage())
