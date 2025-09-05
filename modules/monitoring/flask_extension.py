"""
Flask Extension for Homeostasis

This module provides a Flask extension that integrates with the Homeostasis
monitoring and self-healing system, with special support for Flask blueprints.
"""
import json
import logging
import time
import traceback
import uuid
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, Optional

from .logger import MonitoringLogger
from .extractor import extract_error_context

# Check if Flask is available
try:
    from flask import Blueprint, Flask, Request, Response, current_app, g, request
    from werkzeug.exceptions import HTTPException
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    # Create dummy classes for type hints
    
    class Flask:
        pass
    
    class Blueprint:
        pass
    
    class Request:
        pass
    
    class Response:
        pass

logger = logging.getLogger(__name__)


class Homeostasis:
    """
    Flask extension for integrating with Homeostasis monitoring and self-healing.
    
    This extension captures and processes exceptions that occur during request
    handling, logs them in the Homeostasis format, and optionally attempts to 
    recover from known error patterns.
    """
    
    def __init__(self, app=None):
        """
        Initialize the extension.
        
        Args:
            app: Optional Flask application instance
        """
        self.app = app
        self.monitoring_logger = None
        
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
            "IGNORE_EXCEPTIONS": ["NotFound", "werkzeug.exceptions.NotFound"],
            "MONITOR_PERFORMANCE": True,
            "SLOW_REQUEST_THRESHOLD": 1.0,  # seconds
            "BLUEPRINT_SPECIFIC_HANDLERS": True,
            "ANALYZE_ERRORS": True
        }
        
        # Stores blueprint-specific error handlers
        self.blueprint_handlers = {}
        
        if app is not None:
            self.init_app(app)
    
    def init_app(self, app: Flask):
        """
        Initialize the extension with a Flask application.
        
        Args:
            app: Flask application instance
        """
        if not FLASK_AVAILABLE:
            logger.error("Flask is not available")
            return
        
        # Store the app reference
        self.app = app
        
        # Prefix for configuration keys
        prefix = "HOMEOSTASIS_"
        
        # Update config from app config
        for key, default_value in self.config.items():
            prefixed_key = f"{prefix}{key}"
            if prefixed_key in app.config:
                self.config[key] = app.config[prefixed_key]
        
        # Store configuration in app
        app.config.setdefault("HOMEOSTASIS", self.config)
        
        # Initialize monitoring logger
        try:
            self.monitoring_logger = MonitoringLogger(
                "flask",
                log_level=self.config["LOG_LEVEL"]
            )
            
            logger.info("Homeostasis Flask extension initialized")
            
            if not self.config["ENABLED"]:
                logger.info("Homeostasis Flask extension is disabled")
                return
            
        except Exception as e:
            logger.exception(f"Error initializing Homeostasis Flask extension: {e}")
            return
        
        # Register before_request handler
        app.before_request(self._before_request)
        
        # Register after_request handler
        app.after_request(self._after_request)
        
        # Register error handlers
        if self.config["BLUEPRINT_SPECIFIC_HANDLERS"]:
            # Register a blueprint-specific handler setup function
            app.register_blueprint_handlers = self.register_blueprint_handlers
            
            # Update any existing blueprints
            for bp_name, bp in app.blueprints.items():
                self._setup_blueprint_handlers(bp)
        
        # Register general error handler
        app.errorhandler(Exception)(self._handle_exception)
    
    def register_blueprint_handlers(self, blueprint: Blueprint):
        """
        Register Homeostasis error handlers for a blueprint.
        
        Args:
            blueprint: Flask blueprint instance
        """
        if not self.config["ENABLED"] or not self.config["BLUEPRINT_SPECIFIC_HANDLERS"]:
            return
        
        self._setup_blueprint_handlers(blueprint)
    
    def _setup_blueprint_handlers(self, blueprint: Blueprint):
        """
        Set up error handlers for a blueprint.
        
        Args:
            blueprint: Flask blueprint instance
        """
        # Store a reference to the blueprint
        self.blueprint_handlers[blueprint.name] = {
            "blueprint": blueprint,
            "handlers": {}
        }
        
        # Register the general exception handler for this blueprint
        blueprint.errorhandler(Exception)(self._make_blueprint_handler(blueprint))
    
    def _make_blueprint_handler(self, blueprint: Blueprint) -> Callable:
        """
        Create a blueprint-specific error handler.
        
        Args:
            blueprint: Flask blueprint instance
            
        Returns:
            Error handler function for the blueprint
        """
        def blueprint_error_handler(error):
            return self._handle_exception(error, blueprint=blueprint)
        
        return blueprint_error_handler
    
    def _before_request(self):
        """Handle before_request actions."""
        if not self.config["ENABLED"]:
            return
        
        try:
            # Generate a unique request ID
            request_id = str(uuid.uuid4())
            g.homeostasis_request_id = request_id
            
            # Store the start time for performance monitoring
            g.homeostasis_start_time = time.time()
            
            # Log the request if enabled
            if self.config["LOG_REQUESTS"]:
                self._log_request(request)
                
        except Exception as e:
            logger.exception(f"Error in Homeostasis before_request: {e}")
    
    def _after_request(self, response: Response) -> Response:
        """
        Handle after_request actions.
        
        Args:
            response: Flask response object
            
        Returns:
            Modified or original response
        """
        if not self.config["ENABLED"]:
            return response
        
        try:
            # Calculate request duration
            if hasattr(g, "homeostasis_start_time"):
                duration = time.time() - g.homeostasis_start_time
                
                # Log performance metrics if enabled
                if self.config["MONITOR_PERFORMANCE"] and duration > self.config["SLOW_REQUEST_THRESHOLD"]:
                    self._log_slow_request(request, duration)
                
                # Log the response if enabled
                if self.config["LOG_RESPONSES"]:
                    self._log_response(request, response, duration)
                    
        except Exception as e:
            logger.exception(f"Error in Homeostasis after_request: {e}")
            
        return response
    
    def _handle_exception(self, error: Exception, blueprint: Optional[Blueprint] = None) -> Response:
        """
        Handle an exception.
        
        Args:
            error: The exception that was raised
            blueprint: Optional blueprint that raised the exception
            
        Returns:
            Flask response object
        """
        if not self.config["ENABLED"]:
            # Re-raise the exception for Flask's normal exception handling
            if not isinstance(error, HTTPException):
                raise error
            return error
        
        try:
            # Calculate request duration
            duration = 0.0
            if hasattr(g, "homeostasis_start_time"):
                duration = time.time() - g.homeostasis_start_time
            
            # Check if we should ignore this exception
            if self._should_ignore_exception(error):
                # Let Flask handle it
                if not isinstance(error, HTTPException):
                    raise error
                return error
            
            # Get the request ID
            request_id = getattr(g, "homeostasis_request_id", str(uuid.uuid4()))
            
            # Capture the exception
            error_data = self._capture_exception(request, error, duration, blueprint)
            
            # Analyze the error if enabled
            if self.config["ANALYZE_ERRORS"]:
                analysis = self._analyze_error(error_data)
                error_data["analysis"] = analysis
            
            # Attempt to recover if enabled
            if self.config["ATTEMPT_RECOVERY"]:
                recovery_response = self._attempt_recovery(request, error, error_data)
                if recovery_response:
                    return recovery_response
            
            # If it's a HTTP exception, let Flask handle it
            if isinstance(error, HTTPException):
                # Add Homeostasis request ID to response headers
                original_response = error.get_response()
                original_response.headers["X-Homeostasis-Request-ID"] = request_id
                return original_response
            
            # Re-raise the exception for Flask's normal exception handling
            raise error
            
        except Exception as e:
            logger.exception(f"Error handling exception in Homeostasis: {e}")
            
            # If this is our own error, we need to re-raise the original error
            if error is not e:
                raise error
            
            # Otherwise, re-raise our error
            raise
    
    def _log_request(self, request: Request) -> None:
        """
        Log information about an incoming request.
        
        Args:
            request: Flask request object
        """
        try:
            # Check if we should ignore this path
            if self._should_ignore_path(request.path):
                return
                
            # Extract basic request information
            request_data = {
                "request_id": getattr(g, "homeostasis_request_id", str(uuid.uuid4())),
                "method": request.method,
                "path": request.path,
                "url": request.url,
                "query_string": request.query_string.decode("utf-8") if request.query_string else "",
                "headers": dict(self._get_safe_headers(request)),
                "remote_addr": request.remote_addr,
                "blueprint": request.blueprint,
                "endpoint": request.endpoint,
                "timestamp": datetime.now().isoformat()
            }
            
            # Add rule and view arguments if available
            if request.url_rule:
                request_data["url_rule"] = request.url_rule.rule
                request_data["view_args"] = request.view_args
            
            # Add request body if enabled and available
            if self.config["LOG_REQUEST_BODY"] and request.method in ["POST", "PUT", "PATCH"]:
                try:
                    content_type = request.headers.get("Content-Type", "")
                    if "application/json" in content_type:
                        # JSON body
                        if request.is_json and request.json:
                            body = request.json
                            if not self.config["INCLUDE_SENSITIVE_DATA"]:
                                body = self._sanitize_data(body)
                            request_data["body"] = body
                    elif "application/x-www-form-urlencoded" in content_type:
                        # Form data
                        form_data = dict(request.form)
                        if not self.config["INCLUDE_SENSITIVE_DATA"]:
                            form_data = self._sanitize_data(form_data)
                        request_data["body"] = form_data
                except Exception as e:
                    request_data["body_error"] = str(e)
            
            # Log the request
            self.monitoring_logger.info(
                f"Request: {request.method} {request.path}",
                request=request_data
            )
            
        except Exception as e:
            logger.exception(f"Error logging request: {e}")
    
    def _log_response(self, request: Request, response: Response, 
                     duration: float) -> None:
        """
        Log information about a response.
        
        Args:
            request: Flask request object
            response: Flask response object
            duration: Request duration in seconds
        """
        try:
            # Check if we should ignore this path
            if self._should_ignore_path(request.path):
                return
                
            # Extract basic response information
            response_data = {
                "request_id": getattr(g, "homeostasis_request_id", str(uuid.uuid4())),
                "status_code": response.status_code,
                "status": response.status,
                "duration": duration,
                "content_type": response.content_type,
                "content_length": response.content_length,
                "headers": dict(response.headers),
                "timestamp": datetime.now().isoformat()
            }
            
            # Add response body if enabled and possible
            if self.config["LOG_RESPONSE_BODY"] and response.data:
                try:
                    content_type = response.content_type or ""
                    if "application/json" in content_type:
                        # JSON response
                        content = json.loads(response.data.decode("utf-8"))
                        if not self.config["INCLUDE_SENSITIVE_DATA"]:
                            content = self._sanitize_data(content)
                        response_data["body"] = content
                    elif "text/html" in content_type:
                        # HTML response - just log the size
                        response_data["body_size"] = len(response.data)
                    elif "text/plain" in content_type:
                        # Plain text response
                        response_data["body"] = response.data.decode("utf-8")
                except Exception as e:
                    response_data["body_error"] = str(e)
            
            # Log the response
            log_level = "INFO" if response.status_code < 400 else "WARNING"
            log_method = getattr(self.monitoring_logger, log_level.lower())
            
            log_method(
                f"Response: {response.status_code} ({duration:.3f}s)",
                response=response_data
            )
            
        except Exception as e:
            logger.exception(f"Error logging response: {e}")
    
    def _log_slow_request(self, request: Request, duration: float) -> None:
        """
        Log information about a slow request.
        
        Args:
            request: Flask request object
            duration: Request duration in seconds
        """
        try:
            # Check if we should ignore this path
            if self._should_ignore_path(request.path):
                return
                
            # Extract basic request information
            request_data = {
                "request_id": getattr(g, "homeostasis_request_id", str(uuid.uuid4())),
                "method": request.method,
                "path": request.path,
                "blueprint": request.blueprint,
                "endpoint": request.endpoint,
                "duration": duration,
                "threshold": self.config["SLOW_REQUEST_THRESHOLD"],
                "timestamp": datetime.now().isoformat()
            }
            
            # Log the slow request
            self.monitoring_logger.warning(
                f"Slow request: {request.method} {request.path} ({duration:.3f}s)",
                slow_request=request_data
            )
            
        except Exception as e:
            logger.exception(f"Error logging slow request: {e}")
    
    def _capture_exception(self, request: Request, exc: Exception, 
                          duration: float, blueprint: Optional[Blueprint] = None) -> Dict[str, Any]:
        """
        Capture and log an exception that occurred during request processing.
        
        Args:
            request: Flask request object
            exc: The exception that was raised
            duration: Request duration in seconds
            blueprint: Optional blueprint that raised the exception
            
        Returns:
            Dictionary with error data
        """
        try:
            # Generate exception information
            exc_type = type(exc).__name__
            exc_message = str(exc)
            exc_traceback = traceback.format_exception(type(exc), exc, exc.__traceback__)
            
            # Create error data structure
            error_data = {
                "request_id": getattr(g, "homeostasis_request_id", str(uuid.uuid4())),
                "timestamp": datetime.now().isoformat(),
                "exception_type": exc_type,
                "message": exc_message,
                "traceback": exc_traceback,
                "duration": duration,
                "method": request.method,
                "path": request.path,
                "service": "flask"
            }
            
            # Add blueprint information if available
            if blueprint:
                error_data["blueprint"] = blueprint.name
            elif request.blueprint:
                error_data["blueprint"] = request.blueprint
                
            # Add endpoint information if available
            if request.endpoint:
                error_data["endpoint"] = request.endpoint
            
            # Add status code for HTTP exceptions
            if isinstance(exc, HTTPException):
                error_data["status_code"] = exc.code
            
            # Add request information
            error_data["request"] = {
                "method": request.method,
                "path": request.path,
                "url": request.url,
                "query_string": request.query_string.decode("utf-8") if request.query_string else "",
                "headers": dict(self._get_safe_headers(request)),
                "remote_addr": request.remote_addr
            }
            
            # Add rule and view arguments if available
            if request.url_rule:
                error_data["request"]["url_rule"] = request.url_rule.rule
                error_data["request"]["view_args"] = request.view_args
            
            # Extract additional error context
            error_context = extract_error_context(exc, error_data)
            if error_context:
                error_data["error_details"] = error_context
            
            # Log the exception
            self.monitoring_logger.error(
                f"Exception in request: {exc_type}: {exc_message}",
                exception=error_data
            )
            
            return error_data
            
        except Exception as e:
            logger.exception(f"Error capturing exception: {e}")
            return {
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "meta_error": str(e)
            }
    
    def _analyze_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze an error using the Homeostasis analyzer.
        
        Args:
            error_data: Error data dictionary
            
        Returns:
            Analysis results
        """
        try:
            # Import here to avoid circular imports
            from ..analysis.analyzer import Analyzer, AnalysisStrategy
            
            # Analyze the error
            analyzer = Analyzer(strategy=AnalysisStrategy.HYBRID)
            analysis = analyzer.analyze_error(error_data)
            
            return analysis
            
        except Exception as e:
            logger.exception(f"Error analyzing exception: {e}")
            return {
                "error": str(e),
                "status": "failed"
            }
    
    def _attempt_recovery(self, request: Request, exc: Exception, 
                         error_data: Dict[str, Any]) -> Optional[Response]:
        """
        Attempt to recover from the exception.
        
        Args:
            request: Flask request object
            exc: The exception that was raised
            error_data: Error data dictionary
            
        Returns:
            Response if recovery was successful, None otherwise
        """
        # This is where we would apply recovery strategies
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
                    analysis=analysis
                )
                
                # Import Flask response creation function here to avoid circular imports
                from flask import jsonify
                
                # Return a JSON response with recovery information
                response = jsonify({
                    "error": error_message,
                    "recovered": True,
                    "type": type(exc).__name__,
                    "request_id": error_data["request_id"]
                })
                response.status_code = status_code
                return response
            
            return None
            
        except Exception as e:
            logger.exception(f"Error attempting recovery: {e}")
            return None
    
    def _get_safe_headers(self, request: Request) -> Dict[str, str]:
        """
        Extract safe headers from the request, removing sensitive information.
        
        Args:
            request: Flask request object
            
        Returns:
            Dictionary of headers
        """
        headers = dict(request.headers)
        
        # Remove sensitive headers if needed
        if not self.config["INCLUDE_SENSITIVE_DATA"]:
            sensitive_headers = {
                "Authorization", "Cookie", "X-Api-Key", "Api-Key",
                "X-Auth-Token", "X-Csrf-Token"
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
            "password", "token", "secret", "key", "auth", "credential",
            "credit_card", "card_number", "cvv", "ssn", "social_security",
            "passport", "license"
        }
        
        if isinstance(data, dict):
            return {
                k: "[REDACTED]" if any(s in k.lower() for s in sensitive_fields)
                else self._sanitize_data(v)
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
        if exc_name in self.config["IGNORE_EXCEPTIONS"]:
            return True
        
        # Check exact matches by qualified name
        qualified_name = f"{exc_class.__module__}.{exc_name}"
        if qualified_name in self.config["IGNORE_EXCEPTIONS"]:
            return True
        
        return False


# Decorator for monitoring specific functions or routes
def monitor(func=None, *, include_args=False, include_result=False, 
           performance_threshold=None):
    """
    Decorator to monitor functions or routes.
    
    Args:
        func: Function to decorate
        include_args: Whether to include function arguments in monitoring
        include_result: Whether to include function result in monitoring
        performance_threshold: Custom performance threshold for this function
        
    Returns:
        Decorated function
    """
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            # Initialize
            try:
                # Get the Homeostasis extension
                app = current_app
                homeostasis = app.extensions.get("homeostasis")
                if not homeostasis or not homeostasis.config["ENABLED"]:
                    return f(*args, **kwargs)
                    
                start_time = time.time()
                
                # Generate an ID for this execution
                exec_id = str(uuid.uuid4())
                
                # Determine logging context
                context = {
                    "function": f.__name__,
                    "module": f.__module__,
                    "exec_id": exec_id
                }
                
                # Include arguments if enabled
                if include_args:
                    if len(args) > 0 and hasattr(args[0], "__class__") and args[0].__class__.__name__ == "MethodView":
                        # Skip 'self' for MethodViews
                        context["args"] = args[1:] if len(args) > 1 else []
                    else:
                        context["args"] = args
                    
                    context["kwargs"] = kwargs
                
                # Log function entry
                homeostasis.monitoring_logger.debug(
                    f"Function entry: {f.__name__}",
                    function_entry=context
                )
            except Exception as e:
                logger.exception(f"Error in Homeostasis function monitor setup: {e}")
                return f(*args, **kwargs)
            
            # Execute the function
            try:
                result = f(*args, **kwargs)
                duration = time.time() - start_time
                
                # Determine threshold
                threshold = performance_threshold
                if threshold is None and homeostasis:
                    threshold = homeostasis.config["SLOW_REQUEST_THRESHOLD"]
                
                # Build result context
                result_context = {
                    "function": f.__name__,
                    "module": f.__module__,
                    "exec_id": exec_id,
                    "duration": duration
                }
                
                # Include result if enabled
                if include_result:
                    result_context["result"] = result
                
                # Log performance if slow
                if threshold and duration > threshold:
                    homeostasis.monitoring_logger.warning(
                        f"Slow function: {f.__name__} ({duration:.3f}s)",
                        slow_function=result_context
                    )
                else:
                    homeostasis.monitoring_logger.debug(
                        f"Function exit: {f.__name__} ({duration:.3f}s)",
                        function_exit=result_context
                    )
                
                return result
                
            except Exception as e:
                # Log the exception
                duration = time.time() - start_time
                
                # Create error context
                error_context = {
                    "function": f.__name__,
                    "module": f.__module__,
                    "exec_id": exec_id,
                    "duration": duration,
                    "exception_type": type(e).__name__,
                    "message": str(e),
                    "traceback": traceback.format_exception(type(e), e, e.__traceback__)
                }
                
                # Log the exception
                homeostasis.monitoring_logger.error(
                    f"Exception in function: {f.__name__}: {type(e).__name__}: {str(e)}",
                    function_exception=error_context
                )
                
                # Re-raise the exception
                raise
                
        return wrapper
    
    if func is None:
        return decorator
    return decorator(func)


# Usage examples
"""
# Basic usage with Flask application
from flask import Flask
from modules.monitoring.flask_extension import Homeostasis

app = Flask(__name__)
homeostasis = Homeostasis(app)

# Or with factory pattern
homeostasis = Homeostasis()

def create_app():
    app = Flask(__name__)
    app.config["HOMEOSTASIS_LOG_REQUESTS"] = True
    homeostasis.init_app(app)
    return app

# Blueprint-specific error handling
from flask import Blueprint

bp = Blueprint("api", __name__, url_prefix="/api")

@bp.route("/")
def index():
    return {"status": "ok"}

# Register handlers for the blueprint
app.register_blueprint_handlers(bp)

# Function monitoring
from modules.monitoring.flask_extension import monitor

@monitor(include_args=True)
def my_function(arg1, arg2):
    return arg1 + arg2

# Route monitoring
@app.route("/monitored")
@monitor(performance_threshold=0.5)
def monitored_route():
    return {"status": "ok"}
"""