"""
FastAPI middleware for enhanced logging of requests and exceptions.
Provides detailed tracking of requests, responses, and exceptions with rich context.
"""

import json
import time
import uuid
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import parse_qs

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from .logger import MonitoringLogger


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Enhanced middleware for logging requests, responses, and exceptions with rich context.
    """

    def __init__(
        self,
        app: ASGIApp,
        service_name: str = "example_service",
        log_level: str = "INFO",
        exclude_paths: Optional[List[str]] = None,
        sensitive_headers: Optional[List[str]] = None,
    ):
        """
        Initialize the middleware with enhanced options.

        Args:
            app: The ASGI app
            service_name: Name of the service
            log_level: Logging level
            exclude_paths: List of paths to exclude from logging (e.g., ["/health", "/metrics"])
            sensitive_headers: List of headers to exclude from logging (in addition to defaults)
        """
        super().__init__(app)
        self.logger = MonitoringLogger(service_name, log_level=log_level)
        self.exclude_paths = exclude_paths or []

        # Default sensitive headers + custom ones
        self.sensitive_headers = set(
            [
                "authorization",
                "cookie",
                "set-cookie",
                "x-api-key",
                "api-key",
                "token",
                "session",
                "password",
                "secret",
                "credential",
                "apikey",
            ]
        )

        if sensitive_headers:
            self.sensitive_headers.update([h.lower() for h in sensitive_headers])

    async def _get_request_body(self, request: Request) -> Optional[Dict[str, Any]]:
        """
        Attempt to parse and return the request body.

        Args:
            request: The incoming request

        Returns:
            Parsed request body if possible, None otherwise
        """
        try:
            # Attempt to get the body, may not be available for streaming requests
            # or if already consumed
            body = await request.body()
            content_type = request.headers.get("content-type", "").lower()

            if not body:
                return None

            if "application/json" in content_type:
                return json.loads(body)
            elif "application/x-www-form-urlencoded" in content_type:
                form_data = parse_qs(body.decode("utf-8"))
                # Convert lists with single items to just the item
                return {k: v[0] if len(v) == 1 else v for k, v in form_data.items()}
            else:
                # For other content types, just return the length
                return {"body_size_bytes": len(body)}
        except Exception:
            # If we can't get the body, just return None
            return None

    def _sanitize_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """
        Sanitize headers to remove sensitive information.

        Args:
            headers: Dictionary of headers

        Returns:
            Sanitized headers
        """
        sanitized = {}
        for header_name, header_value in headers.items():
            if header_name.lower() in self.sensitive_headers:
                sanitized[header_name] = "[REDACTED]"
            else:
                sanitized[header_name] = header_value
        return sanitized

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process the request, log relevant information, and handle exceptions with enhanced context.

        Args:
            request: The incoming request
            call_next: The next middleware/route handler

        Returns:
            The response
        """
        # Skip logging for excluded paths
        path = request.url.path
        for exclude_path in self.exclude_paths:
            if path.startswith(exclude_path):
                return await call_next(request)

        # Generate a unique request ID
        request_id = str(uuid.uuid4())
        start_time = time.time()

        # Prepare full path with query params
        full_path = path
        if request.query_params:
            full_path += f"?{request.query_params}"

        # Try to get request headers (sanitized)
        headers_dict = dict(request.headers.items())
        sanitized_headers = self._sanitize_headers(headers_dict)

        # Prepare detailed request info
        request_info = {
            "request_id": request_id,
            "method": request.method,
            "path": path,
            "full_path": full_path,
            "client_ip": request.client.host if request.client else None,
            "user_agent": request.headers.get("user-agent"),
            "headers": sanitized_headers,
            "query_params": dict(request.query_params),
        }

        # Skip body logging for now to avoid consuming the stream
        # This prevents the timeout issue with FastAPI/Starlette
        # TODO: Implement proper body logging with stream wrapper if needed

        # Log request start
        self.logger.info(
            f"Request started: {request.method} {full_path}",
            include_call_location=True,
            request_info=request_info,
            tags=["request", "start", request.method.lower()],
        )

        try:
            # Process request
            response = await call_next(request)

            # Calculate processing time
            process_time = time.time() - start_time
            duration_ms = round(process_time * 1000, 2)

            # Get response status and determine log level
            status_code = response.status_code
            is_error = status_code >= 400

            # Prepare response info
            response_info = {
                "status_code": status_code,
                "duration_ms": duration_ms,
            }

            # Log the appropriate level based on status code
            if is_error:
                log_msg = f"Request error: {request.method} {full_path} - {status_code}"
                self.logger.error(
                    log_msg,
                    include_call_location=True,
                    request_info=request_info,
                    response_info=response_info,
                    tags=[
                        "request",
                        "error",
                        f"status-{status_code}",
                        request.method.lower(),
                    ],
                )
            else:
                log_msg = (
                    f"Request completed: {request.method} {full_path} - {status_code}"
                )
                self.logger.info(
                    log_msg,
                    include_call_location=True,
                    request_info=request_info,
                    response_info=response_info,
                    tags=[
                        "request",
                        "success",
                        f"status-{status_code}",
                        request.method.lower(),
                    ],
                )

            return response

        except Exception as e:
            # Calculate processing time
            process_time = time.time() - start_time
            duration_ms = round(process_time * 1000, 2)

            # Add the failed request information to the error
            error_context = {
                "request_info": request_info,
                "duration_ms": duration_ms,
                "tags": ["request", "exception", request.method.lower()],
            }

            # Log the exception with enhanced details
            self.logger.exception(
                e, include_locals=True, include_globals=False, **error_context
            )

            # Re-raise the exception
            raise


def add_logging_middleware(
    app: FastAPI,
    service_name: str = "example_service",
    log_level: str = "INFO",
    exclude_paths: Optional[List[str]] = None,
    sensitive_headers: Optional[List[str]] = None,
) -> None:
    """
    Add enhanced logging middleware to a FastAPI application.

    Args:
        app: The FastAPI application
        service_name: Name of the service
        log_level: Logging level
        exclude_paths: Paths to exclude from logging (e.g., ["/health", "/metrics"])
        sensitive_headers: Additional headers to exclude from logging
    """
    app.add_middleware(
        LoggingMiddleware,
        service_name=service_name,
        log_level=log_level,
        exclude_paths=exclude_paths,
        sensitive_headers=sensitive_headers,
    )
