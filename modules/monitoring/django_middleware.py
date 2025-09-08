"""
Django Middleware for Homeostasis

This module provides middleware components for Django applications to integrate
with the Homeostasis monitoring and self-healing system.
"""

import json
import logging
import time
import traceback
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from .extractor import extract_error_context
from .logger import MonitoringLogger

# Check if Django is available
try:
    from django.conf import settings
    from django.core.exceptions import MiddlewareNotUsed
    from django.http import HttpRequest, HttpResponse, JsonResponse
    from django.urls import resolve

    DJANGO_AVAILABLE = True
except ImportError:
    DJANGO_AVAILABLE = False
    # Create dummy classes for type hints

    class HttpRequest:
        pass

    class HttpResponse:
        pass


logger = logging.getLogger(__name__)


class HomeostasisMiddleware:
    """
    Django middleware for integrating with Homeostasis monitoring and self-healing.

    This middleware captures and processes exceptions that occur during request
    handling, logs them in the Homeostasis format, and optionally attempts to
    recover from known error patterns.
    """

    def __init__(self, get_response):
        """
        Initialize the middleware.

        Args:
            get_response: The next middleware or view function in the chain
        """
        if not DJANGO_AVAILABLE:
            raise MiddlewareNotUsed("Django is not available")

        self.get_response = get_response

        # Initialize monitoring logger
        try:
            # Check if homeostasis config is available in Django settings
            homeostasis_config = getattr(settings, "HOMEOSTASIS", {})
            log_level = homeostasis_config.get("LOG_LEVEL", "INFO")

            self.monitoring_logger = MonitoringLogger("django", log_level=log_level)

            # Configure middleware options
            self.enabled = homeostasis_config.get("ENABLED", True)
            self.log_requests = homeostasis_config.get("LOG_REQUESTS", False)
            self.log_responses = homeostasis_config.get("LOG_RESPONSES", False)
            self.log_request_body = homeostasis_config.get("LOG_REQUEST_BODY", False)
            self.log_response_body = homeostasis_config.get("LOG_RESPONSE_BODY", False)
            self.attempt_recovery = homeostasis_config.get("ATTEMPT_RECOVERY", False)
            self.include_sensitive_data = homeostasis_config.get(
                "INCLUDE_SENSITIVE_DATA", False
            )

            # Get list of paths to ignore
            self.ignore_paths = set(homeostasis_config.get("IGNORE_PATHS", []))

            # Get list of exception types to ignore
            self.ignore_exceptions = set(
                homeostasis_config.get("IGNORE_EXCEPTIONS", [])
            )

            # Performance monitoring options
            self.monitor_performance = homeostasis_config.get(
                "MONITOR_PERFORMANCE", False
            )
            self.slow_request_threshold = homeostasis_config.get(
                "SLOW_REQUEST_THRESHOLD", 1.0
            )  # seconds

            logger.info("Homeostasis Django middleware initialized")

            if not self.enabled:
                raise MiddlewareNotUsed(
                    "Homeostasis middleware is disabled in settings"
                )

        except Exception as e:
            logger.exception(f"Error initializing Homeostasis middleware: {e}")
            raise MiddlewareNotUsed("Failed to initialize Homeostasis middleware")

    def __call__(self, request):
        """
        Process the request and response.

        Args:
            request: Django HTTP request

        Returns:
            Django HTTP response
        """
        # Check if we should ignore this path
        path = request.path
        if self._should_ignore_path(path):
            return self.get_response(request)

        # Generate a unique request ID for tracking
        request_id = str(uuid.uuid4())
        request.homeostasis_request_id = request_id

        # Start timing the request
        start_time = time.time()

        # Log the incoming request if enabled
        if self.log_requests:
            self._log_request(request)

        # Process the request
        try:
            # Get the response from the next middleware or view
            response = self.get_response(request)

            # Record request duration
            duration = time.time() - start_time

            # Log performance metrics if enabled
            if self.monitor_performance and duration > self.slow_request_threshold:
                self._log_slow_request(request, duration)

            # Log the response if enabled
            if self.log_responses:
                self._log_response(request, response, duration)

            return response

        except Exception as exc:
            # Handle and log the exception
            duration = time.time() - start_time

            # Check if this exception type should be ignored
            if self._should_ignore_exception(exc):
                # Re-raise the exception for Django's normal exception handling
                raise

            # Log the exception
            error_data = self._capture_exception(request, exc, duration)

            # Attempt to recover if enabled
            if self.attempt_recovery:
                recovery_response = self._attempt_recovery(request, exc, error_data)
                if recovery_response:
                    return recovery_response

            # Re-raise the exception for Django's normal exception handling
            raise

    def _log_request(self, request: HttpRequest) -> None:
        """
        Log information about an incoming request.

        Args:
            request: Django HTTP request
        """
        try:
            # Extract basic request information
            request_data = {
                "request_id": getattr(
                    request, "homeostasis_request_id", str(uuid.uuid4())
                ),
                "method": request.method,
                "path": request.path,
                "query_params": dict(request.GET),
                "headers": dict(self._get_safe_headers(request)),
                "remote_addr": self._get_client_ip(request),
                "timestamp": datetime.now().isoformat(),
            }

            # Add user information if available
            if hasattr(request, "user") and request.user.is_authenticated:
                request_data["user"] = {
                    "id": request.user.id,
                    "username": request.user.username,
                }

            # Add request body if enabled and available
            if self.log_request_body and request.method in ["POST", "PUT", "PATCH"]:
                try:
                    content_type = request.headers.get("Content-Type", "")
                    if "application/json" in content_type:
                        # JSON body
                        if hasattr(request, "body") and request.body:
                            body = json.loads(request.body.decode("utf-8"))
                            if not self.include_sensitive_data:
                                body = self._sanitize_data(body)
                            request_data["body"] = body
                    elif "application/x-www-form-urlencoded" in content_type:
                        # Form data
                        form_data = dict(request.POST)
                        if not self.include_sensitive_data:
                            form_data = self._sanitize_data(form_data)
                        request_data["body"] = form_data
                except Exception as e:
                    request_data["body_error"] = str(e)

            # Try to get the view name
            try:
                resolver_match = resolve(request.path)
                request_data["view"] = resolver_match.view_name
                request_data["url_name"] = resolver_match.url_name
                request_data["url_kwargs"] = resolver_match.kwargs
            except Exception:
                pass

            # Log the request
            self.monitoring_logger.info(
                f"Request: {request.method} {request.path}", request=request_data
            )

        except Exception as e:
            logger.exception(f"Error logging request: {e}")

    def _log_response(
        self, request: HttpRequest, response: HttpResponse, duration: float
    ) -> None:
        """
        Log information about a response.

        Args:
            request: Django HTTP request
            response: Django HTTP response
            duration: Request duration in seconds
        """
        try:
            # Extract basic response information
            response_data = {
                "request_id": getattr(
                    request, "homeostasis_request_id", str(uuid.uuid4())
                ),
                "status_code": response.status_code,
                "duration": duration,
                "content_type": response.get("Content-Type", ""),
                "content_length": response.get("Content-Length", ""),
                "headers": dict(response.headers.items()),
                "timestamp": datetime.now().isoformat(),
            }

            # Add response body if enabled and possible
            if self.log_response_body and hasattr(response, "content"):
                try:
                    content_type = response.get("Content-Type", "")
                    if "application/json" in content_type:
                        # JSON response
                        content = json.loads(response.content.decode("utf-8"))
                        if not self.include_sensitive_data:
                            content = self._sanitize_data(content)
                        response_data["body"] = content
                    elif "text/html" in content_type:
                        # HTML response - just log the size
                        response_data["body_size"] = len(response.content)
                    elif "text/plain" in content_type:
                        # Plain text response
                        response_data["body"] = response.content.decode("utf-8")
                except Exception as e:
                    response_data["body_error"] = str(e)

            # Log the response
            log_level = "INFO" if response.status_code < 400 else "WARNING"
            log_method = getattr(self.monitoring_logger, log_level.lower())

            log_method(
                f"Response: {response.status_code} ({duration:.3f}s)",
                response=response_data,
            )

        except Exception as e:
            logger.exception(f"Error logging response: {e}")

    def _log_slow_request(self, request: HttpRequest, duration: float) -> None:
        """
        Log information about a slow request.

        Args:
            request: Django HTTP request
            duration: Request duration in seconds
        """
        try:
            # Extract basic request information
            request_data = {
                "request_id": getattr(
                    request, "homeostasis_request_id", str(uuid.uuid4())
                ),
                "method": request.method,
                "path": request.path,
                "duration": duration,
                "threshold": self.slow_request_threshold,
                "timestamp": datetime.now().isoformat(),
            }

            # Try to get the view name
            try:
                resolver_match = resolve(request.path)
                request_data["view"] = resolver_match.view_name
            except Exception:
                pass

            # Log the slow request
            self.monitoring_logger.warning(
                f"Slow request: {request.method} {request.path} ({duration:.3f}s)",
                slow_request=request_data,
            )

        except Exception as e:
            logger.exception(f"Error logging slow request: {e}")

    def _capture_exception(
        self, request: HttpRequest, exc: Exception, duration: float
    ) -> Dict[str, Any]:
        """
        Capture and log an exception that occurred during request processing.

        Args:
            request: Django HTTP request
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
                "request_id": getattr(
                    request, "homeostasis_request_id", str(uuid.uuid4())
                ),
                "timestamp": datetime.now().isoformat(),
                "exception_type": exc_type,
                "message": exc_message,
                "traceback": exc_traceback,
                "duration": duration,
                "method": request.method,
                "path": request.path,
                "service": "django",
            }

            # Add request information
            error_data["request"] = {
                "method": request.method,
                "path": request.path,
                "query_params": dict(request.GET),
                "headers": dict(self._get_safe_headers(request)),
                "remote_addr": self._get_client_ip(request),
            }

            # Try to get the view name
            try:
                resolver_match = resolve(request.path)
                error_data["request"]["view"] = resolver_match.view_name
                error_data["request"]["url_name"] = resolver_match.url_name
                error_data["request"]["url_kwargs"] = resolver_match.kwargs
            except Exception:
                pass

            # Add user information if available
            if hasattr(request, "user") and request.user.is_authenticated:
                error_data["user"] = {
                    "id": request.user.id,
                    "username": request.user.username,
                }

            # Extract additional error context
            error_context = extract_error_context(exc, error_data)
            if error_context:
                error_data["error_details"] = error_context

            # Log the exception
            self.monitoring_logger.error(
                f"Exception in request: {exc_type}: {exc_message}", exception=error_data
            )

            return error_data

        except Exception as e:
            logger.exception(f"Error capturing exception: {e}")
            return {
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "meta_error": str(e),
            }

    def _attempt_recovery(
        self, request: HttpRequest, exc: Exception, error_data: Dict[str, Any]
    ) -> Optional[HttpResponse]:
        """
        Attempt to recover from the exception.

        Args:
            request: Django HTTP request
            exc: The exception that was raised
            error_data: Error data dictionary

        Returns:
            HttpResponse if recovery was successful, None otherwise
        """
        # Placeholder for future self-healing capabilities
        # This could look up known error patterns and apply recovery strategies
        try:
            # Import here to avoid circular imports
            from ..analysis.analyzer import AnalysisStrategy, Analyzer

            # Analyze the error
            analyzer = Analyzer(strategy=AnalysisStrategy.HYBRID)
            analysis = analyzer.analyze_error(error_data)

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

                # Return a JSON response with recovery information
                return JsonResponse(
                    {
                        "error": error_message,
                        "recovered": True,
                        "type": type(exc).__name__,
                        "request_id": error_data["request_id"],
                    },
                    status=status_code,
                )

            return None

        except Exception as e:
            logger.exception(f"Error attempting recovery: {e}")
            return None

    def _get_client_ip(self, request: HttpRequest) -> str:
        """
        Extract the client IP address from the request.

        Args:
            request: Django HTTP request

        Returns:
            Client IP address string
        """
        x_forwarded_for = request.META.get("HTTP_X_FORWARDED_FOR")
        if x_forwarded_for:
            # Get the first address in case of proxy chains
            ip = x_forwarded_for.split(",")[0].strip()
        else:
            ip = request.META.get("REMOTE_ADDR", "unknown")
        return ip

    def _get_safe_headers(self, request: HttpRequest) -> Dict[str, str]:
        """
        Extract safe headers from the request, removing sensitive information.

        Args:
            request: Django HTTP request

        Returns:
            Dictionary of headers
        """
        if not hasattr(request, "headers"):
            # Django < 2.2
            headers = {k: v for k, v in request.META.items() if k.startswith("HTTP_")}
            # Convert HTTP_X_HEADER to X-Header
            headers = {k[5:].replace("_", "-").title(): v for k, v in headers.items()}
        else:
            # Django >= 2.2
            headers = dict(request.headers.items())

        # Remove sensitive headers if needed
        if not self.include_sensitive_data:
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
        if path in self.ignore_paths:
            return True

        # Check prefixes
        for ignore_path in self.ignore_paths:
            if ignore_path.endswith("*") and path.startswith(ignore_path[:-1]):
                return True

        # Static and media file check
        if hasattr(settings, "STATIC_URL") and path.startswith(settings.STATIC_URL):
            return True

        if hasattr(settings, "MEDIA_URL") and path.startswith(settings.MEDIA_URL):
            return True

        return False

    def _should_ignore_exception(self, exc: Exception) -> bool:
        """
        Check if an exception should be ignored.

        Args:
            exc: Exception instance

        Returns:
            True if the exception should be ignored, False otherwise
        """
        exc_class = type(exc)
        exc_name = exc_class.__name__

        # Check exact matches by name
        if exc_name in self.ignore_exceptions:
            return True

        # Check exact matches by qualified name
        qualified_name = f"{exc_class.__module__}.{exc_name}"
        if qualified_name in self.ignore_exceptions:
            return True

        # Check if it's a Django 404 or permission error
        from django.core.exceptions import PermissionDenied
        from django.http import Http404

        if isinstance(exc, Http404) and "Http404" in self.ignore_exceptions:
            return True

        if (
            isinstance(exc, PermissionDenied) and
            "PermissionDenied" in self.ignore_exceptions
        ):
            return True

        return False


class HomeostasisDebugToolbarPanel:
    """
    Django Debug Toolbar panel for Homeostasis.

    This class can be used to integrate Homeostasis with the Django Debug Toolbar.
    """

    template = "homeostasis_debug_panel.html"

    def __init__(self, toolbar):
        self.toolbar = toolbar
        self.records = []

    def nav_title(self):
        return "Homeostasis"

    def nav_subtitle(self):
        # Show the number of errors if any
        error_count = sum(1 for r in self.records if r.get("level") == "ERROR")
        if error_count:
            return f"{error_count} errors"
        return ""

    def title(self):
        return "Homeostasis Monitoring"

    def url(self):
        return ""

    def process_request(self, request):
        # This method is called for each request
        pass

    def process_response(self, request, response):
        # This method is called for each response
        # Here we would collect monitoring data
        pass

    def enable_instrumentation(self):
        # Enable instrumentation when the panel is active
        # This should connect to the Homeostasis logging system
        pass

    def disable_instrumentation(self):
        # Disable instrumentation when the panel is deactivated
        pass

    def record_stats(self, stats):
        # Record statistics for display in the panel
        self.stats = stats


# Example Django settings for Homeostasis middleware
"""
# Add to your Django settings.py:

HOMEOSTASIS = {
    'ENABLED': True,
    'LOG_LEVEL': 'INFO',
    'LOG_REQUESTS': True,
    'LOG_RESPONSES': True,
    'LOG_REQUEST_BODY': False,
    'LOG_RESPONSE_BODY': False,
    'ATTEMPT_RECOVERY': True,
    'INCLUDE_SENSITIVE_DATA': False,
    'IGNORE_PATHS': ['/health/', '/admin/jsi18n/', '/static/*', '/media/*'],
    'IGNORE_EXCEPTIONS': ['Http404', 'PermissionDenied', 'django.core.exceptions.ValidationError'],
    'MONITOR_PERFORMANCE': True,
    'SLOW_REQUEST_THRESHOLD': 1.0,  # seconds
}

MIDDLEWARE = [
    # Other middleware...
    'modules.monitoring.django_middleware.HomeostasisMiddleware',
]
"""


# Example of how to integrate with Django Debug Toolbar
"""
# Add to your Django settings.py:

DEBUG_TOOLBAR_PANELS = [
    # Default panels...
    'modules.monitoring.django_middleware.HomeostasisDebugToolbarPanel',
]
"""
