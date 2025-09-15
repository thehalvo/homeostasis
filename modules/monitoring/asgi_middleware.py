"""
ASGI Middleware for Homeostasis

This module provides ASGI middleware components for integrating with the
Homeostasis monitoring and self-healing system. It works with ASGI frameworks
like Starlette, FastAPI, Quart, and others.
"""

import json
import logging
import time
import traceback
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, Optional

# from .extractor import extract_error_context  # TODO: implement extract_error_context
from .logger import MonitoringLogger

logger = logging.getLogger(__name__)


class HomeostasisASGIMiddleware:
    """
    ASGI middleware for integrating with Homeostasis monitoring and self-healing.

    This middleware captures and processes exceptions that occur during request
    handling, logs them in the Homeostasis format, and optionally attempts to
    recover from known error patterns.
    """

    config: Dict[str, Any]

    def __init__(self, app, **config):
        """
        Initialize the middleware.

        Args:
            app: ASGI application
            **config: Configuration options
        """
        self.app = app

        # Configuration defaults
        self.config = {
            "ENABLED": True,
            "LOG_LEVEL": "INFO",
            "LOG_REQUESTS": True,
            "LOG_RESPONSES": True,
            "LOG_REQUEST_BODY": False,
            "LOG_RESPONSE_BODY": False,
            "ATTEMPT_RECOVERY": True,
            "INCLUDE_SENSITIVE_DATA": False,
            "IGNORE_PATHS": ["/static/", "/favicon.ico"],
            "IGNORE_HEADERS": [],
            "MONITOR_PERFORMANCE": True,
            "SLOW_REQUEST_THRESHOLD": 1.0,  # seconds
            "ANALYZE_ERRORS": True,
            "FRAMEWORK": "asgi",  # Can be "fastapi", "starlette", "quart", etc.
        }

        # Update config with provided values
        self.config.update(config)

        # Initialize monitoring logger
        try:
            self.monitoring_logger = MonitoringLogger(
                self.config["FRAMEWORK"], log_level=self.config["LOG_LEVEL"]
            )

            logger.info(
                f"Homeostasis ASGI middleware initialized for {self.config['FRAMEWORK']}"
            )

            if not self.config["ENABLED"]:
                logger.info("Homeostasis ASGI middleware is disabled")

        except Exception as e:
            logger.exception(f"Error initializing Homeostasis ASGI middleware: {e}")

    async def __call__(self, scope, receive, send):
        """
        Process the ASGI request.

        Args:
            scope: ASGI scope
            receive: ASGI receive function
            send: ASGI send function
        """
        if not self.config["ENABLED"]:
            await self.app(scope, receive, send)
            return

        # Only handle HTTP requests
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Check if we should ignore this path
        path = scope.get("path", "")
        if self._should_ignore_path(path):
            await self.app(scope, receive, send)
            return

        # Generate a unique request ID
        request_id = str(uuid.uuid4())

        # Add request ID to scope state if supported
        if "state" in scope:
            scope["state"]["homeostasis_request_id"] = request_id

        # Start timing the request
        start_time = time.time()

        # Log the incoming request if enabled
        if self.config["LOG_REQUESTS"]:
            await self._log_request(scope, request_id)

        # Intercept the send function to capture the response
        response_status = [None]
        response_headers = [None]
        response_body = []

        async def send_wrapper(message):
            # Capture response information
            message_type = message["type"]

            if message_type == "http.response.start":
                response_status[0] = message.get("status", 0)
                response_headers[0] = message.get("headers", [])

            elif (
                message_type == "http.response.body"
                and self.config["LOG_RESPONSE_BODY"]
            ):
                if "body" in message:
                    response_body.append(message["body"])

            # Pass the message to the original send function
            await send(message)

            # Log the response after sending
            if message_type == "http.response.body" and "more_body" not in message:
                # This is the last message, log the response
                if self.config["LOG_RESPONSES"]:
                    duration = time.time() - start_time
                    await self._log_response(
                        scope,
                        request_id,
                        response_status[0],
                        response_headers[0],
                        b"".join(response_body) if response_body else None,
                        duration,
                    )

        # Intercept the receive function to capture the request body
        request_body = []

        async def receive_wrapper():
            message = await receive()

            # Capture request body if needed
            if message["type"] == "http.request" and self.config["LOG_REQUEST_BODY"]:
                if "body" in message:
                    request_body.append(message["body"])

            return message

        # Process the request
        try:
            # Call the application with our wrapped functions
            await self.app(scope, receive_wrapper, send_wrapper)

        except Exception as exc:
            # Handle and log the exception
            duration = time.time() - start_time

            # Build full request body
            full_body = b"".join(request_body) if request_body else None

            # Log the exception
            error_data = await self._capture_exception(
                scope, request_id, exc, duration, full_body
            )

            # Attempt to recover if enabled
            if self.config["ATTEMPT_RECOVERY"]:
                recovery_response = await self._attempt_recovery(
                    scope, request_id, exc, error_data, send
                )

                if recovery_response:
                    # The recovery function already sent the response
                    return

            # Re-raise the exception for the framework's error handling
            raise

    async def _log_request(self, scope: Dict[str, Any], request_id: str) -> None:
        """
        Log information about an incoming request.

        Args:
            scope: ASGI scope
            request_id: Request ID
        """
        try:
            # Get request method
            method = scope.get("method", "UNKNOWN")

            # Get request path
            path = scope.get("path", "")

            # Get request query string
            query_string = scope.get("query_string", b"").decode(
                "utf-8", errors="ignore"
            )

            # Get request headers
            headers = self._get_safe_headers(scope)

            # Get client address
            client = scope.get("client", ("unknown", 0))
            client_addr = f"{client[0]}:{client[1]}" if client else "unknown"

            # Build request data
            request_data = {
                "request_id": request_id,
                "method": method,
                "path": path,
                "query_string": query_string,
                "headers": headers,
                "remote_addr": client_addr,
                "timestamp": datetime.now().isoformat(),
            }

            # Add server name if available
            if "server" in scope:
                server = scope["server"]
                request_data["server"] = (
                    f"{server[0]}:{server[1]}" if server else "unknown"
                )

            # Add request body if enabled (collected later)

            # Log the request
            self.monitoring_logger.info(
                f"Request: {method} {path}", request=request_data
            )

        except Exception as e:
            logger.exception(f"Error logging request: {e}")

    async def _log_response(
        self,
        scope: Dict[str, Any],
        request_id: str,
        status: Optional[int],
        headers: Optional[list],
        body: Optional[bytes],
        duration: float,
    ) -> None:
        """
        Log information about a response.

        Args:
            scope: ASGI scope
            request_id: Request ID
            status: HTTP status code
            headers: Response headers
            body: Response body (may be None)
            duration: Request duration in seconds
        """
        try:
            # Build response data
            response_data = {
                "request_id": request_id,
                "status_code": status,
                "duration": duration,
                "timestamp": datetime.now().isoformat(),
            }

            # Add headers if available
            if headers:
                header_dict = {}
                for key, value in headers:
                    key_str = key.decode("utf-8", errors="ignore")
                    value_str = value.decode("utf-8", errors="ignore")
                    header_dict[key_str] = value_str
                response_data["headers"] = header_dict

            # Add body if available and enabled
            if body and self.config["LOG_RESPONSE_BODY"]:
                try:
                    # Check content type from headers
                    content_type = None
                    if headers:
                        for key, value in headers:
                            if key.lower() == b"content-type":
                                content_type = value.decode("utf-8", errors="ignore")
                                break

                    if content_type and "application/json" in content_type:
                        # JSON response
                        json_body = json.loads(body.decode("utf-8"))
                        if not self.config["INCLUDE_SENSITIVE_DATA"]:
                            json_body = self._sanitize_data(json_body)
                        response_data["body"] = json_body
                    elif content_type and "text/html" in content_type:
                        # HTML response - just log the size
                        response_data["body_size"] = len(body)
                    elif content_type and "text/plain" in content_type:
                        # Plain text response
                        response_data["body"] = body.decode("utf-8", errors="ignore")
                    else:
                        # Other types - just log the size
                        response_data["body_size"] = len(body)
                except Exception as e:
                    response_data["body_error"] = str(e)

            # Log the response
            log_level = "INFO" if status and status < 400 else "WARNING"
            log_method = getattr(self.monitoring_logger, log_level.lower())

            log_method(f"Response: {status} ({duration:.3f}s)", response=response_data)

            # Log performance metrics if enabled
            if (
                self.config["MONITOR_PERFORMANCE"]
                and duration > self.config["SLOW_REQUEST_THRESHOLD"]
            ):
                await self._log_slow_request(scope, request_id, duration)

        except Exception as e:
            logger.exception(f"Error logging response: {e}")

    async def _log_slow_request(
        self, scope: Dict[str, Any], request_id: str, duration: float
    ) -> None:
        """
        Log information about a slow request.

        Args:
            scope: ASGI scope
            request_id: Request ID
            duration: Request duration in seconds
        """
        try:
            # Get path for context
            path = scope.get("path", "")
            method = scope.get("method", "UNKNOWN")

            # Build slow request data
            slow_request_data = {
                "request_id": request_id,
                "method": method,
                "path": path,
                "duration": duration,
                "threshold": self.config["SLOW_REQUEST_THRESHOLD"],
                "timestamp": datetime.now().isoformat(),
            }

            # Get client address
            client = scope.get("client", ("unknown", 0))
            client_addr = f"{client[0]}:{client[1]}" if client else "unknown"
            slow_request_data["remote_addr"] = client_addr

            # Log the slow request
            self.monitoring_logger.warning(
                f"Slow request: {method} {path} ({duration:.3f}s)",
                slow_request=slow_request_data,
            )

        except Exception as e:
            logger.exception(f"Error logging slow request: {e}")

    async def _capture_exception(
        self,
        scope: Dict[str, Any],
        request_id: str,
        exc: Exception,
        duration: float,
        body: Optional[bytes] = None,
    ) -> Dict[str, Any]:
        """
        Capture and log an exception that occurred during request processing.

        Args:
            scope: ASGI scope
            request_id: Request ID
            exc: The exception that was raised
            duration: Request duration in seconds
            body: Request body (may be None)

        Returns:
            Dictionary with error data
        """
        try:
            # Generate exception information
            exc_type = type(exc).__name__
            exc_message = str(exc)
            exc_traceback = traceback.format_exception(
                type(exc), exc, exc.__traceback__
            )

            # Get path for context
            path = scope.get("path", "")
            method = scope.get("method", "UNKNOWN")

            # Create error data structure
            error_data = {
                "request_id": request_id,
                "timestamp": datetime.now().isoformat(),
                "exception_type": exc_type,
                "message": exc_message,
                "traceback": exc_traceback,
                "duration": duration,
                "method": method,
                "path": path,
                "service": self.config["FRAMEWORK"],
            }

            # Add request information
            query_string = scope.get("query_string", b"").decode(
                "utf-8", errors="ignore"
            )

            error_data["request"] = {
                "method": method,
                "path": path,
                "query_string": query_string,
                "headers": self._get_safe_headers(scope),
            }

            # Add client information
            client = scope.get("client", ("unknown", 0))
            client_addr = f"{client[0]}:{client[1]}" if client else "unknown"
            error_data["request"]["remote_addr"] = client_addr

            # Add server information
            if "server" in scope:
                server = scope["server"]
                error_data["request"]["server"] = (
                    f"{server[0]}:{server[1]}" if server else "unknown"
                )

            # Add request body if available and enabled
            if body and self.config["LOG_REQUEST_BODY"]:
                try:
                    # Check content type from headers
                    content_type = None
                    for key, value in scope.get("headers", []):
                        if key.lower() == b"content-type":
                            content_type = value.decode("utf-8", errors="ignore")
                            break

                    if content_type and "application/json" in content_type:
                        # JSON body
                        json_body = json.loads(body.decode("utf-8"))
                        if not self.config["INCLUDE_SENSITIVE_DATA"]:
                            json_body = self._sanitize_data(json_body)
                        error_data["request"]["body"] = json_body
                    elif (
                        content_type
                        and "application/x-www-form-urlencoded" in content_type
                    ):
                        # Form data - just log the size
                        error_data["request"]["body_size"] = len(body)
                    else:
                        # Other types - just log the size
                        error_data["request"]["body_size"] = len(body)
                except Exception as e:
                    error_data["request"]["body_error"] = str(e)

            # Extract additional error context
            # TODO: Implement extract_error_context function
            # error_context = extract_error_context(exc, error_data)
            # if error_context:
            #     error_data["error_details"] = error_context

            # Log the exception
            self.monitoring_logger.error(
                f"Exception in request: {exc_type}: {exc_message}", exception=error_data
            )

            # Analyze the error if enabled
            if self.config["ANALYZE_ERRORS"]:
                analysis = await self._analyze_error(error_data)
                error_data["analysis"] = analysis

            return error_data

        except Exception as e:
            logger.exception(f"Error capturing exception: {e}")
            return {
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "meta_error": str(e),
            }

    async def _analyze_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze an error using the Homeostasis analyzer.

        Args:
            error_data: Error data dictionary

        Returns:
            Analysis results
        """
        try:
            # Import here to avoid circular imports
            from ..analysis.analyzer import AnalysisStrategy, Analyzer

            # Analyze the error
            analyzer = Analyzer(strategy=AnalysisStrategy.HYBRID)
            analysis = analyzer.analyze_error(error_data)

            return analysis

        except Exception as e:
            logger.exception(f"Error analyzing exception: {e}")
            return {"error": str(e), "status": "failed"}

    async def _attempt_recovery(
        self,
        scope: Dict[str, Any],
        request_id: str,
        exc: Exception,
        error_data: Dict[str, Any],
        send: Callable,
    ) -> bool:
        """
        Attempt to recover from the exception.

        Args:
            scope: ASGI scope
            request_id: Request ID
            exc: The exception that was raised
            error_data: Error data dictionary
            send: ASGI send function

        Returns:
            True if recovery was successful, False otherwise
        """
        try:
            analysis = error_data.get("analysis", {})

            if analysis.get("can_recover", False):
                # Create a recovery response
                recovery = analysis.get("recovery", {})
                status_code = recovery.get("status_code", 500)
                error_message = recovery.get("error_message", "An error occurred")

                # Log the recovery attempt
                self.monitoring_logger.info(
                    f"Recovered from {type(exc).__name__} with status {status_code}",
                    recovery=recovery,
                    analysis=analysis,
                )

                # Create a JSON response
                body = json.dumps(
                    {
                        "error": error_message,
                        "recovered": True,
                        "type": type(exc).__name__,
                        "request_id": request_id,
                    }
                ).encode("utf-8")

                # Send the response
                await send(
                    {
                        "type": "http.response.start",
                        "status": status_code,
                        "headers": [
                            (b"content-type", b"application/json"),
                            (b"content-length", str(len(body)).encode("utf-8")),
                            (b"x-homeostasis-request-id", request_id.encode("utf-8")),
                        ],
                    }
                )

                await send({"type": "http.response.body", "body": body})

                return True

            return False

        except Exception as e:
            logger.exception(f"Error attempting recovery: {e}")
            return False

    def _get_safe_headers(self, scope: Dict[str, Any]) -> Dict[str, str]:
        """
        Extract safe headers from the scope, removing sensitive information.

        Args:
            scope: ASGI scope

        Returns:
            Dictionary of headers
        """
        headers = {}

        for key, value in scope.get("headers", []):
            key_str = key.decode("utf-8", errors="ignore").lower()
            value_str = value.decode("utf-8", errors="ignore")

            # Skip ignored headers
            if key_str in self.config["IGNORE_HEADERS"]:
                continue

            # Remove sensitive headers if needed
            if not self.config["INCLUDE_SENSITIVE_DATA"]:
                sensitive_headers = {
                    "authorization",
                    "cookie",
                    "x-api-key",
                    "api-key",
                    "x-auth-token",
                    "x-csrf-token",
                }

                if key_str in sensitive_headers:
                    value_str = "[REDACTED]"

            headers[key_str] = value_str

        return headers

    def _sanitize_data(self, data: Any) -> Any:
        """
        Sanitize data by removing sensitive information.

        Args:
            data: Data to sanitize

        Returns:
            Sanitized data
        """
        sensitive_fields = {
            "password",
            "token",
            "secret",
            "key",
            "auth",
            "credential",
            "credit_card",
            "card_number",
            "cvv",
            "ssn",
            "social_security",
            "passport",
            "license",
        }

        if isinstance(data, dict):
            return {
                k: (
                    "[REDACTED]"
                    if any(s in k.lower() for s in sensitive_fields)
                    else self._sanitize_data(v)
                )
                for k, v in data.items()
            }
        elif isinstance(data, list):
            return [self._sanitize_data(item) for item in data]
        else:
            return data

    def _should_ignore_path(self, path: str) -> bool:
        """
        Check if a path should be ignored.

        Args:
            path: URL path

        Returns:
            True if the path should be ignored, False otherwise
        """
        # Check exact matches
        if path in self.config["IGNORE_PATHS"]:
            return True

        # Check prefixes
        for ignore_path in self.config["IGNORE_PATHS"]:
            if ignore_path.endswith("*") and path.startswith(ignore_path[:-1]):
                return True

        return False


from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from tornado.web import RequestHandler

@runtime_checkable
class TornadoHandlerProtocol(Protocol):
    """Protocol defining the expected interface for Tornado RequestHandler."""
    request: Any
    _headers: Any

    def get_status(self) -> int: ...
    def set_status(self, status_code: int) -> None: ...
    def set_header(self, name: str, value: str) -> None: ...
    def write(self, chunk: Any) -> None: ...
    def finish(self) -> None: ...

class TornadoMonitoringMixin:
    """
    Mixin for integrating Homeostasis monitoring into Tornado request handlers.

    This mixin captures and processes exceptions that occur during request
    handling, logs them in the Homeostasis format, and optionally attempts to
    recover from known error patterns.

    Note: This mixin assumes it will be used with tornado.web.RequestHandler
    as a base class. The super() calls are to RequestHandler methods.
    """

    def initialize(self, **kwargs):
        """Initialize the request handler."""
        super().initialize(**kwargs)  # type: ignore[misc]

        # Get or create a Homeostasis configuration
        self.homeostasis_config = getattr(self, "homeostasis_config", {})

        # Set defaults if not already set
        self.homeostasis_config.setdefault("ENABLED", True)
        self.homeostasis_config.setdefault("LOG_LEVEL", "INFO")
        self.homeostasis_config.setdefault("LOG_REQUESTS", True)
        self.homeostasis_config.setdefault("LOG_RESPONSES", True)
        self.homeostasis_config.setdefault("LOG_REQUEST_BODY", False)
        self.homeostasis_config.setdefault("LOG_RESPONSE_BODY", False)
        self.homeostasis_config.setdefault("ATTEMPT_RECOVERY", True)
        self.homeostasis_config.setdefault("INCLUDE_SENSITIVE_DATA", False)
        self.homeostasis_config.setdefault("MONITOR_PERFORMANCE", True)
        self.homeostasis_config.setdefault("SLOW_REQUEST_THRESHOLD", 1.0)  # seconds
        self.homeostasis_config.setdefault("ANALYZE_ERRORS", True)

        # Initialize monitoring logger if not already set
        if not hasattr(self, "homeostasis_logger"):
            try:
                self.homeostasis_logger = MonitoringLogger(
                    "tornado", log_level=self.homeostasis_config["LOG_LEVEL"]
                )
            except Exception as e:
                logger.exception(f"Error initializing Homeostasis logger: {e}")

        # Generate a unique request ID
        self.homeostasis_request_id = str(uuid.uuid4())

        # Set start time
        self.homeostasis_start_time = time.time()

    def prepare(self):
        """Called at the beginning of a request before get/post/etc."""
        super().prepare()  # type: ignore[misc]

        if not self.homeostasis_config.get("ENABLED", True):
            return

        # Log the request if enabled
        if self.homeostasis_config.get("LOG_REQUESTS", True):
            self._log_request()

    def on_finish(self):
        """Called after the end of a request."""
        super().on_finish()  # type: ignore[misc]

        if not self.homeostasis_config.get("ENABLED", True):
            return

        # Calculate request duration
        duration = time.time() - self.homeostasis_start_time

        # Log performance metrics if enabled
        if self.homeostasis_config.get(
            "MONITOR_PERFORMANCE", True
        ) and duration > self.homeostasis_config.get("SLOW_REQUEST_THRESHOLD", 1.0):
            self._log_slow_request(duration)

        # Log the response if enabled
        if self.homeostasis_config.get("LOG_RESPONSES", True):
            self._log_response(duration)

    def write_error(self, status_code, **kwargs):
        """Override to customize the error response."""
        if not self.homeostasis_config.get("ENABLED", True):
            return super().write_error(status_code, **kwargs)  # type: ignore[misc]

        exc_info = kwargs.get("exc_info")
        if exc_info:
            exc_type, exc_value, exc_tb = exc_info

            # Calculate request duration
            duration = time.time() - self.homeostasis_start_time

            # Log the exception
            error_data = self._capture_exception(exc_value, duration)

            # Attempt to recover if enabled
            if self.homeostasis_config.get("ATTEMPT_RECOVERY", True):
                recovery_response = self._attempt_recovery(exc_value, error_data)
                if recovery_response:
                    return

        # Fall back to default error handling
        return super().write_error(status_code, **kwargs)  # type: ignore[misc]

    def _log_request(self):
        """Log information about the incoming request."""
        try:
            # Extract basic request information
            request_data = {
                "request_id": self.homeostasis_request_id,
                "method": self.request.method,
                "path": self.request.path,
                "query": self.request.query,
                "headers": self._get_safe_headers(),
                "remote_ip": self.request.remote_ip,
                "timestamp": datetime.now().isoformat(),
            }

            # Add protocol version if available
            request_data["protocol"] = self.request.version

            # Add request body if enabled and available
            if self.homeostasis_config.get("LOG_REQUEST_BODY", False):
                try:
                    content_type = self.request.headers.get("Content-Type", "")
                    if "application/json" in content_type:
                        # JSON body
                        try:
                            body = json.loads(self.request.body.decode("utf-8"))
                            if not self.homeostasis_config.get(
                                "INCLUDE_SENSITIVE_DATA", False
                            ):
                                body = self._sanitize_data(body)
                            request_data["body"] = body
                        except Exception:
                            request_data["body"] = self.request.body.decode(
                                "utf-8", errors="ignore"
                            )
                    elif "application/x-www-form-urlencoded" in content_type:
                        # Form data
                        request_data["body"] = {
                            k: v[0] if len(v) == 1 else v
                            for k, v in self.request.arguments.items()
                        }
                except Exception as e:
                    request_data["body_error"] = str(e)

            # Log the request
            self.homeostasis_logger.info(
                f"Request: {self.request.method} {self.request.path}",
                request=request_data,
            )

        except Exception as e:
            logger.exception(f"Error logging request: {e}")

    def _log_response(self, duration):
        """
        Log information about the response.

        Args:
            duration: Request duration in seconds
        """
        try:
            # Extract basic response information
            response_data = {
                "request_id": self.homeostasis_request_id,
                "status_code": self.get_status(),
                "duration": duration,
                "headers": {k: v for k, v in self._headers.items()},
                "timestamp": datetime.now().isoformat(),
            }

            # Add response body if enabled and possible
            if self.homeostasis_config.get("LOG_RESPONSE_BODY", False):
                try:
                    # Note: In Tornado we can't easily access the response body
                    # after it's been sent, so this is limited
                    response_data["body_available"] = False
                except Exception as e:
                    response_data["body_error"] = str(e)

            # Log the response
            log_level = "INFO" if self.get_status() < 400 else "WARNING"
            log_method = getattr(self.homeostasis_logger, log_level.lower())

            log_method(
                f"Response: {self.get_status()} ({duration:.3f}s)",
                response=response_data,
            )

        except Exception as e:
            logger.exception(f"Error logging response: {e}")

    def _log_slow_request(self, duration):
        """
        Log information about a slow request.

        Args:
            duration: Request duration in seconds
        """
        try:
            # Extract basic request information
            request_data = {
                "request_id": self.homeostasis_request_id,
                "method": self.request.method,
                "path": self.request.path,
                "duration": duration,
                "threshold": self.homeostasis_config.get("SLOW_REQUEST_THRESHOLD", 1.0),
                "timestamp": datetime.now().isoformat(),
            }

            # Log the slow request
            self.homeostasis_logger.warning(
                f"Slow request: {self.request.method} {self.request.path} ({duration:.3f}s)",
                slow_request=request_data,
            )

        except Exception as e:
            logger.exception(f"Error logging slow request: {e}")

    def _capture_exception(self, exc, duration):
        """
        Capture and log an exception.

        Args:
            exc: The exception that was raised
            duration: Request duration in seconds

        Returns:
            Dictionary with error data
        """
        try:
            # Generate exception information
            exc_type = type(exc).__name__
            exc_message = str(exc)
            exc_traceback = traceback.format_exception(
                type(exc), exc, exc.__traceback__
            )

            # Create error data structure
            error_data = {
                "request_id": self.homeostasis_request_id,
                "timestamp": datetime.now().isoformat(),
                "exception_type": exc_type,
                "message": exc_message,
                "traceback": exc_traceback,
                "duration": duration,
                "method": self.request.method,
                "path": self.request.path,
                "service": "tornado",
            }

            # Add request information
            error_data["request"] = {
                "method": self.request.method,
                "path": self.request.path,
                "query": self.request.query,
                "headers": self._get_safe_headers(),
                "remote_ip": self.request.remote_ip,
            }

            # Extract additional error context
            # TODO: Implement extract_error_context function
            # error_context = extract_error_context(exc, error_data)
            # if error_context:
            #     error_data["error_details"] = error_context

            # Log the exception
            self.homeostasis_logger.error(
                f"Exception in request: {exc_type}: {exc_message}", exception=error_data
            )

            # Analyze the error if enabled
            if self.homeostasis_config.get("ANALYZE_ERRORS", True):
                analysis = self._analyze_error(error_data)
                error_data["analysis"] = analysis

            return error_data

        except Exception as e:
            logger.exception(f"Error capturing exception: {e}")
            return {
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "meta_error": str(e),
            }

    def _analyze_error(self, error_data):
        """
        Analyze an error using the Homeostasis analyzer.

        Args:
            error_data: Error data dictionary

        Returns:
            Analysis results
        """
        try:
            # Import here to avoid circular imports
            from ..analysis.analyzer import AnalysisStrategy, Analyzer

            # Analyze the error
            analyzer = Analyzer(strategy=AnalysisStrategy.HYBRID)
            analysis = analyzer.analyze_error(error_data)

            return analysis

        except Exception as e:
            logger.exception(f"Error analyzing exception: {e}")
            return {"error": str(e), "status": "failed"}

    def _attempt_recovery(self, exc, error_data):
        """
        Attempt to recover from the exception.

        Args:
            exc: The exception that was raised
            error_data: Error data dictionary

        Returns:
            True if recovery was successful, False otherwise
        """
        try:
            analysis = error_data.get("analysis", {})

            if analysis.get("can_recover", False):
                # Create a recovery response
                recovery = analysis.get("recovery", {})
                status_code = recovery.get("status_code", 500)
                error_message = recovery.get("error_message", "An error occurred")

                # Log the recovery attempt
                self.homeostasis_logger.info(
                    f"Recovered from {type(exc).__name__} with status {status_code}",
                    recovery=recovery,
                    analysis=analysis,
                )

                # Set the status code
                self.set_status(status_code)

                # Set JSON content type
                self.set_header("Content-Type", "application/json")

                # Write the response
                self.write(
                    {
                        "error": error_message,
                        "recovered": True,
                        "type": type(exc).__name__,
                        "request_id": self.homeostasis_request_id,
                    }
                )

                # Finish the request
                self.finish()

                return True

            return False

        except Exception as e:
            logger.exception(f"Error attempting recovery: {e}")
            return False

    def _get_safe_headers(self):
        """
        Extract safe headers from the request, removing sensitive information.

        Returns:
            Dictionary of headers
        """
        headers = {k: v for k, v in self.request.headers.items()}

        # Remove sensitive headers if needed
        if not self.homeostasis_config.get("INCLUDE_SENSITIVE_DATA", False):
            sensitive_headers = {
                "Authorization",
                "Cookie",
                "X-Api-Key",
                "Api-Key",
                "X-Auth-Token",
                "X-Csrf-Token",
            }

            for header in sensitive_headers:
                if header in headers:
                    headers[header] = "[REDACTED]"

        return headers

    def _sanitize_data(self, data):
        """
        Sanitize data by removing sensitive information.

        Args:
            data: Data to sanitize

        Returns:
            Sanitized data
        """
        sensitive_fields = {
            "password",
            "token",
            "secret",
            "key",
            "auth",
            "credential",
            "credit_card",
            "card_number",
            "cvv",
            "ssn",
            "social_security",
            "passport",
            "license",
        }

        if isinstance(data, dict):
            return {
                k: (
                    "[REDACTED]"
                    if any(s in k.lower() for s in sensitive_fields)
                    else self._sanitize_data(v)
                )
                for k, v in data.items()
            }
        elif isinstance(data, list):
            return [self._sanitize_data(item) for item in data]
        else:
            return data


# Usage examples
"""
# ASGI middleware with Starlette/FastAPI
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from modules.monitoring.asgi_middleware import HomeostasisASGIMiddleware

async def homepage(request):
    return JSONResponse({"message": "Hello, world!"})

routes = [
    Route("/", endpoint=homepage)
]

app = Starlette(routes=routes)
app = HomeostasisASGIMiddleware(app, FRAMEWORK="starlette")

# FastAPI example
from fastapi import FastAPI
from modules.monitoring.asgi_middleware import HomeostasisASGIMiddleware

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello, world!"}

app = HomeostasisASGIMiddleware(app, FRAMEWORK="fastapi")

# Tornado example
import tornado.web
import tornado.ioloop
from modules.monitoring.asgi_middleware import TornadoMonitoringMixin

class MainHandler(TornadoMonitoringMixin, tornado.web.RequestHandler):
    def get(self):
        self.write("Hello, world!")

app = tornado.web.Application([
    (r"/", MainHandler),
])

if __name__ == "__main__":
    app.listen(8888)
    tornado.ioloop.IOLoop.current().start()
"""
