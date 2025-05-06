"""
FastAPI middleware for logging requests and exceptions.
"""
import json
import sys
import time
import traceback
import uuid
from typing import Callable, Dict, Any

from fastapi import FastAPI, Request, Response
from fastapi.routing import APIRoute
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from .logger import MonitoringLogger


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for logging requests, responses, and exceptions.
    """

    def __init__(self, app: ASGIApp, service_name: str = "example_service"):
        """
        Initialize the middleware.

        Args:
            app: The ASGI app
            service_name: Name of the service
        """
        super().__init__(app)
        self.logger = MonitoringLogger(service_name)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process the request, log relevant information, and handle exceptions.

        Args:
            request: The incoming request
            call_next: The next middleware/route handler
        
        Returns:
            The response
        """
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Log request
        path = request.url.path
        if request.query_params:
            path += f"?{request.query_params}"
        
        self.logger.info(
            f"Request started: {request.method} {path}",
            request_id=request_id,
            method=request.method,
            path=path,
            client=request.client.host if request.client else None
        )
        
        try:
            # Process request
            response = await call_next(request)
            
            # Log response
            process_time = time.time() - start_time
            status_code = response.status_code
            
            log_method = (
                self.logger.info if 100 <= status_code < 400 else self.logger.error
            )
            
            log_method(
                f"Request completed: {request.method} {path} - {status_code}",
                request_id=request_id,
                method=request.method,
                path=path,
                status_code=status_code,
                duration=round(process_time * 1000, 2)  # ms
            )
            
            return response
        
        except Exception as e:
            # Log exception
            process_time = time.time() - start_time
            
            self.logger.exception(
                e,
                request_id=request_id,
                method=request.method,
                path=path,
                duration=round(process_time * 1000, 2)  # ms
            )
            
            # Re-raise the exception
            raise


def add_logging_middleware(app: FastAPI, service_name: str = "example_service") -> None:
    """
    Add logging middleware to a FastAPI application.

    Args:
        app: The FastAPI application
        service_name: Name of the service
    """
    app.add_middleware(LoggingMiddleware, service_name=service_name)