#!/usr/bin/env python3
"""
Homeostasis Dashboard

Web dashboard for monitoring and managing Homeostasis self-healing activities.
"""

import argparse
import json
import logging
import os
import signal
import sys
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import flask
import yaml
from flask import Flask, jsonify, redirect, render_template, request, url_for
from flask_cors import CORS
from flask_socketio import SocketIO, emit

# Add project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from modules.deployment.canary import CanaryStatus, get_canary_deployment
from modules.llm_integration.api_key_manager import (APIKeyManager,
                                                     KeyValidationError)
from modules.monitoring.cost_tracker import (Budget, BudgetPeriod, CostTracker,
                                             default_alert_callback)
from modules.monitoring.llm_metrics import (AlertConfig, LLMMetricsCollector,
                                            UsageQuota)
# Import Homeostasis modules
from modules.monitoring.metrics_collector import MetricsCollector
from modules.monitoring.security_guardrails import (SecurityGuardrails,
                                                    SecurityLevel)
from modules.security.api_security import (get_api_security_manager,
                                           secure_endpoint)
from modules.security.auth import authenticate, get_auth_manager
from modules.security.rbac import get_rbac_manager, has_permission
from modules.suggestion.diff_viewer import create_diff as generate_diff
from modules.suggestion.suggestion_manager import (SuggestionStatus,
                                                   get_suggestion_manager)

# Initialize Flask app
app = Flask(__name__, static_folder="static", template_folder="templates")

# Setup CORS
CORS(app)

# Setup SocketIO for real-time updates
socketio = SocketIO(app, cors_allowed_origins="*")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("homeostasis.dashboard")


class DashboardServer:
    """
    Main server class for the Homeostasis Dashboard.
    """

    def __init__(self, config_path: Optional[Path] = None, debug: bool = False):
        """
        Initialize the dashboard server.

        Args:
            config_path: Path to the configuration file
            debug: Whether to run in debug mode
        """
        # Set debug mode
        self.debug = debug

        # Load configuration
        self.config_path = config_path or Path(__file__).parent / "config.yaml"
        self.config = self._load_config()

        # Store Flask app reference
        self.app = app
        self.socketio = socketio

        # Apply configuration to Flask app
        app.secret_key = self.config.get("secret_key", os.urandom(24))
        app.config["SESSION_TYPE"] = "filesystem"
        app.config["DEBUG"] = debug

        # Initialize components
        self.metrics_collector = MetricsCollector()

        # Initialize LLM observability components
        llm_config = self.config.get("llm_observability", {})
        quota_config = UsageQuota(
            tokens_per_hour=llm_config.get("tokens_per_hour"),
            tokens_per_day=llm_config.get("tokens_per_day"),
            cost_per_hour=llm_config.get("cost_per_hour"),
            cost_per_day=llm_config.get("cost_per_day"),
            cost_per_month=llm_config.get("cost_per_month"),
            requests_per_minute=llm_config.get("requests_per_minute"),
            requests_per_hour=llm_config.get("requests_per_hour"),
        )
        alert_config = AlertConfig()

        self.llm_metrics_collector = LLMMetricsCollector(
            quota_config=quota_config, alert_config=alert_config
        )

        # Initialize cost tracker
        budgets = []
        if llm_config.get("daily_budget"):
            budgets.append(
                Budget(
                    amount=llm_config["daily_budget"],
                    period=BudgetPeriod.DAY,
                    hard_limit=llm_config.get("enforce_daily_limit", False),
                )
            )
        if llm_config.get("monthly_budget"):
            budgets.append(
                Budget(
                    amount=llm_config["monthly_budget"],
                    period=BudgetPeriod.MONTH,
                    hard_limit=llm_config.get("enforce_monthly_limit", False),
                )
            )

        self.cost_tracker = CostTracker(budgets=budgets)
        self.cost_tracker.add_alert_callback(default_alert_callback)

        # Initialize security guardrails
        security_level = SecurityLevel(llm_config.get("security_level", "restrictive"))
        self.security_guardrails = SecurityGuardrails(security_level=security_level)

        # Initialize suggestion manager
        suggestion_config = self.config.get("suggestion", {})
        storage_dir = suggestion_config.get("storage_dir")
        self.suggestion_manager = get_suggestion_manager(storage_dir, suggestion_config)
        logger.info("Suggestion manager initialized")

        # Initialize authentication if enabled
        if self.config.get("auth", {}).get("enabled", False):
            auth_config = self.config.get("auth", {})
            self.auth_manager = get_auth_manager(auth_config)
            self.rbac_manager = get_rbac_manager(auth_config)
            logger.info("Authentication initialized")
        else:
            self.auth_manager = None
            self.rbac_manager = None
            logger.info("Authentication disabled")

        # Initialize API security if enabled
        if self.config.get("api_security", {}).get("enabled", False):
            api_security_config = self.config.get("api_security", {})
            self.api_security_manager = get_api_security_manager(api_security_config)
            logger.info("API security initialized")
        else:
            self.api_security_manager = None
            logger.info("API security disabled")

        # Register routes and error handlers
        self._register_routes()
        self._register_error_handlers()
        self._register_socketio_handlers()

        logger.info("Dashboard server initialized")

    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file.

        Returns:
            Dictionary of configuration values
        """
        try:
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)
                logger.info(f"Loaded configuration from {self.config_path}")
                return config
        except FileNotFoundError:
            logger.warning(
                f"Configuration file {self.config_path} not found, using defaults"
            )
            # Create default configuration
            default_config = {
                "server": {"host": "127.0.0.1", "port": 5000, "debug": self.debug},
                "auth": {"enabled": False},
                "api_security": {"enabled": False},
                "homeostasis": {
                    "orchestrator_host": "127.0.0.1",
                    "orchestrator_port": 8000,
                },
            }

            # Save default configuration
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, "w") as f:
                yaml.dump(default_config, f, default_flow_style=False)
                logger.info(f"Created default configuration at {self.config_path}")

            return default_config

    def _register_routes(self) -> None:
        """Register Flask routes for the dashboard."""
        # Check if routes are already registered
        if hasattr(app, "_routes_registered") and app._routes_registered:
            return

        # Main dashboard route
        @app.route("/")
        def index():
            return render_template("index.html")

        # LLM Observability dashboard route
        @app.route("/llm-observability")
        def llm_observability():
            return render_template("llm_observability.html")

        # Suggestion interface routes
        @app.route("/suggestions/<error_id>")
        def suggestions(error_id):
            # Get the error details
            # In a real implementation, this would come from the database
            error_data = {
                "id": error_id,
                "timestamp": (datetime.now() - timedelta(minutes=15)).isoformat(),
                "service": "example_service",
                "error_type": "AttributeError",
                "message": "AttributeError: 'NoneType' object has no attribute 'get'",
                "file_path": "services/example_service/app.py:42",
                "status": "analyzing",
            }

            # Get suggestions from the suggestion manager
            suggestions = self.suggestion_manager.get_suggestions(error_id)

            # If no suggestions yet, generate them
            if not suggestions:
                # Simulate error data
                error_info = {
                    "file_path": "services/example_service/app.py",
                    "error_type": "AttributeError",
                    "error_message": "AttributeError: 'NoneType' object has no attribute 'get'",
                    "line_number": 42,
                }
                suggestions = self.suggestion_manager.generate_suggestions(
                    error_id, error_info
                )

            # Generate diffs for each suggestion
            suggestion_diffs = {}
            for suggestion in suggestions:
                diff_html = generate_diff(
                    suggestion.original_code, suggestion.suggested_code
                )
                suggestion_diffs[suggestion.suggestion_id] = diff_html

            return render_template(
                "suggestion.html",
                error=error_data,
                suggestions=suggestions,
                diff_html=suggestion_diffs,
            )

        # Configuration management UI route
        @app.route("/config")
        def configuration():
            # Simulated configuration data
            config = {
                "system_name": "Homeostasis",
                "environment": "development",
                "log_level": "INFO",
                "data_dir": "/var/lib/homeostasis",
                "auto_healing": True,
                "monitoring": {
                    "interval": 60,
                    "retention_days": 30,
                    "enable_apm": False,
                    "apm_provider": "none",
                    "apm_endpoint": "",
                    "apm_api_key": "",
                },
                "analysis": {
                    "confidence_threshold": 80,
                    "enable_ml": True,
                    "enable_rule_based": True,
                    "collect_training_data": True,
                    "languages": ["python", "javascript", "java"],
                },
                "deployment": {
                    "strategy": "canary",
                    "canary_percentage": 10,
                    "canary_interval": 15,
                    "canary_step": 20,
                    "auto_rollback": True,
                    "kubernetes": {
                        "enabled": True,
                        "namespace": "homeostasis",
                        "context": "default",
                    },
                    "cloud": {
                        "aws": {"enabled": True, "region": "us-east-1"},
                        "gcp": {"enabled": False},
                        "azure": {"enabled": False},
                    },
                    "edge": {"enabled": True, "provider": "cloudflare"},
                    "serverless": {"enabled": True, "provider": "aws-lambda"},
                },
                "security": {
                    "auth_enabled": True,
                    "auth_provider": "local",
                    "rbac_enabled": True,
                    "audit_enabled": True,
                    "audit_retention_days": 90,
                    "approvals": {
                        "enabled": True,
                        "required_approvers": 1,
                        "approval_timeout_hours": 24,
                    },
                    "rate_limiting": {
                        "enabled": True,
                        "healing_cycle_limit": 10,
                        "patch_limit": 20,
                        "deployment_limit": 5,
                    },
                },
            }

            # Simulated rules data
            rules = [
                {
                    "id": "rule-001",
                    "name": "AttributeError for NoneType",
                    "category": "python",
                    "enabled": True,
                },
                {
                    "id": "rule-002",
                    "name": "KeyError in dictionary access",
                    "category": "python",
                    "enabled": True,
                },
                {
                    "id": "rule-003",
                    "name": "TypeError in function call",
                    "category": "python",
                    "enabled": True,
                },
                {
                    "id": "rule-004",
                    "name": "Database connection failure",
                    "category": "database",
                    "enabled": True,
                },
                {
                    "id": "rule-005",
                    "name": "Missing JWT token",
                    "category": "authentication",
                    "enabled": True,
                },
            ]

            # Simulated templates data
            templates = [
                {
                    "id": "template-001",
                    "name": "None check before attribute access",
                    "category": "python",
                },
                {
                    "id": "template-002",
                    "name": "Dictionary get with default",
                    "category": "python",
                },
                {
                    "id": "template-003",
                    "name": "Type check before function call",
                    "category": "python",
                },
                {
                    "id": "template-004",
                    "name": "Database connection retry",
                    "category": "database",
                },
                {
                    "id": "template-005",
                    "name": "JWT token validation",
                    "category": "authentication",
                },
            ]

            # Simulated plugins data
            plugins = [
                {
                    "id": "plugin-001",
                    "name": "Java Language Support",
                    "version": "0.1.0",
                    "enabled": True,
                },
                {
                    "id": "plugin-002",
                    "name": "JavaScript Language Support",
                    "version": "0.1.0",
                    "enabled": True,
                },
                {
                    "id": "plugin-003",
                    "name": "Datadog Integration",
                    "version": "0.1.0",
                    "enabled": False,
                },
                {
                    "id": "plugin-004",
                    "name": "Prometheus Integration",
                    "version": "0.1.0",
                    "enabled": False,
                },
            ]

            return render_template(
                "config.html",
                config=config,
                rules=rules,
                templates=templates,
                plugins=plugins,
            )

        # Rule Editor Routes
        @app.route("/rules/editor/<rule_id>")
        def rule_editor(rule_id):
            """Edit an existing rule."""
            # Fetch rule data by ID (simulated for now)
            rule = next(
                (
                    r
                    for r in [
                        {
                            "id": "rule-001",
                            "name": "AttributeError for NoneType",
                            "category": "python",
                            "description": "Detects AttributeError exceptions when accessing attributes on None",
                            "error_type": "AttributeError",
                            "pattern": r"AttributeError: 'NoneType' object has no attribute '(\w+)'",
                            "enabled": True,
                            "template_id": "template-001",
                            "confidence": 95,
                            "examples": "AttributeError: 'NoneType' object has no attribute 'get'\nAttributeError: 'NoneType' object has no attribute 'name'",
                            "template_parameters": {
                                "attribute": "{0}",
                                "variable": "obj",
                                "default_value": "None",
                            },
                            "tags": ["python", "attribute", "none-check"],
                            "approval_required": False,
                            "auto_test": True,
                        },
                        {
                            "id": "rule-002",
                            "name": "KeyError in dictionary access",
                            "category": "python",
                            "description": "Detects KeyError exceptions when accessing dictionary keys",
                            "error_type": "KeyError",
                            "pattern": r"KeyError: ['\"']?(\w+)['\"']?",
                            "enabled": True,
                            "template_id": "template-002",
                            "confidence": 90,
                            "examples": "KeyError: 'user_id'\nKeyError: 'name'",
                            "template_parameters": {
                                "key": "{0}",
                                "dictionary": "data",
                                "default_value": "None",
                            },
                            "tags": ["python", "dictionary", "key-check"],
                            "approval_required": False,
                            "auto_test": True,
                        },
                    ]
                    if r["id"] == rule_id
                ),
                None,
            )

            # If rule not found, create a new rule template
            if not rule:
                rule = {
                    "id": "",
                    "name": "",
                    "category": "",
                    "description": "",
                    "error_type": "",
                    "pattern": "",
                    "enabled": True,
                    "template_id": "",
                    "confidence": 90,
                    "examples": "",
                    "template_parameters": {},
                    "tags": [],
                    "approval_required": False,
                    "auto_test": True,
                }

            # Fetch templates for selection
            templates = [
                {
                    "id": "template-001",
                    "name": "None check before attribute access",
                    "category": "python",
                    "content": "if {variable} is not None:\n    return {variable}.{attribute}\nreturn {default_value}",
                },
                {
                    "id": "template-002",
                    "name": "Dictionary get with default",
                    "category": "python",
                    "content": 'return {dictionary}.get("{key}", {default_value})',
                },
                {
                    "id": "template-003",
                    "name": "Type check before function call",
                    "category": "python",
                    "content": "if not isinstance({variable}, {expected_type}):\n    {variable} = {type_conversion}({variable})\nreturn {function}({variable})",
                },
            ]

            # Get the selected template if any
            selected_template = next(
                (t for t in templates if t["id"] == rule.get("template_id")), None
            )

            # Simulated rule history
            rule_history = (
                [
                    {
                        "timestamp": "2023-06-15 10:45:30",
                        "user": "admin",
                        "action": "created",
                        "details": "Rule created",
                    },
                    {
                        "timestamp": "2023-06-15 14:22:15",
                        "user": "john",
                        "action": "modified",
                        "details": "Updated pattern and parameters",
                    },
                    {
                        "timestamp": "2023-06-16 09:15:05",
                        "user": "admin",
                        "action": "modified",
                        "details": 'Changed template to "None check before attribute access"',
                    },
                ]
                if rule.get("id")
                else []
            )

            return render_template(
                "rule_editor.html",
                rule=rule,
                templates=templates,
                selected_template=selected_template,
                rule_history=rule_history,
            )

        @app.route("/rules/editor")
        def rule_editor_new():
            """Create a new rule."""
            return redirect(url_for("rule_editor", rule_id="new"))

        # Template Editor Routes
        @app.route("/templates/editor/<template_id>")
        def template_editor(template_id):
            """Edit an existing template."""
            # Fetch template data by ID (simulated for now)
            template = next(
                (
                    t
                    for t in [
                        {
                            "id": "template-001",
                            "name": "None check before attribute access",
                            "category": "python",
                            "description": "Adds a None check before accessing an attribute on an object, returning a default value if the object is None.",
                            "content": "if {variable} is not None:\n    return {variable}.{attribute}\nreturn {default_value}",
                            "parameters": {
                                "variable": "obj",
                                "variable_description": "The object to check for None",
                                "attribute": "attribute_name",
                                "attribute_description": "The name of the attribute to access",
                                "default_value": "None",
                                "default_value_description": "The value to return if the object is None",
                            },
                            "tags": ["python", "attribute", "none-check"],
                            "parent_id": None,
                        },
                        {
                            "id": "template-002",
                            "name": "Dictionary get with default",
                            "category": "python",
                            "description": "Uses dictionary.get() method to access a key with a default value if the key is not present.",
                            "content": 'return {dictionary}.get("{key}", {default_value})',
                            "parameters": {
                                "dictionary": "data",
                                "dictionary_description": "The dictionary to access",
                                "key": "key_name",
                                "key_description": "The key to access in the dictionary",
                                "default_value": "None",
                                "default_value_description": "The value to return if the key is not in the dictionary",
                            },
                            "tags": ["python", "dictionary", "key-check"],
                            "parent_id": None,
                        },
                    ]
                    if t["id"] == template_id
                ),
                None,
            )

            # If template not found, create a new template
            if not template:
                template = {
                    "id": "",
                    "name": "",
                    "category": "",
                    "description": "",
                    "content": "",
                    "parameters": {},
                    "tags": [],
                    "parent_id": None,
                }

            # Fetch parent templates for selection
            parent_templates = [
                {
                    "id": "template-001",
                    "name": "None check before attribute access",
                    "category": "python",
                },
                {
                    "id": "template-002",
                    "name": "Dictionary get with default",
                    "category": "python",
                },
                {
                    "id": "template-003",
                    "name": "Type check before function call",
                    "category": "python",
                },
            ]

            # Get the parent template if any
            parent_template = next(
                (t for t in parent_templates if t["id"] == template.get("parent_id")),
                None,
            )

            # Simulated template usage (rules using this template)
            template_usage = (
                [
                    {
                        "id": "rule-001",
                        "name": "AttributeError for NoneType",
                        "category": "python",
                        "parameters": {
                            "attribute": "{0}",
                            "variable": "obj",
                            "default_value": "None",
                        },
                        "last_used": "2023-06-16 15:30:45",
                    },
                    {
                        "id": "rule-006",
                        "name": "Object property access",
                        "category": "python",
                        "parameters": {
                            "attribute": "config",
                            "variable": "app",
                            "default_value": "{}",
                        },
                        "last_used": "2023-06-15 09:12:30",
                    },
                ]
                if template.get("id")
                else []
            )

            # Simulated child templates (templates inheriting from this one)
            child_templates = (
                [
                    {
                        "id": "template-007",
                        "name": "FastAPI request attribute access",
                        "category": "fastapi",
                    },
                    {
                        "id": "template-008",
                        "name": "Django model attribute access",
                        "category": "django",
                    },
                ]
                if template.get("id") == "template-001"
                else []
            )

            # Simulated template history
            template_history = (
                [
                    {
                        "timestamp": "2023-06-14 11:30:15",
                        "user": "admin",
                        "action": "created",
                        "details": "Template created",
                    },
                    {
                        "timestamp": "2023-06-15 09:45:20",
                        "user": "admin",
                        "action": "modified",
                        "details": "Updated parameter descriptions",
                    },
                    {
                        "timestamp": "2023-06-16 14:22:35",
                        "user": "john",
                        "action": "modified",
                        "details": "Improved content formatting",
                    },
                ]
                if template.get("id")
                else []
            )

            return render_template(
                "template_editor.html",
                template=template,
                parent_templates=parent_templates,
                parent_template=parent_template,
                template_usage=template_usage,
                child_templates=child_templates,
                template_history=template_history,
            )

        @app.route("/templates/editor")
        def template_editor_new():
            """Create a new template."""
            return redirect(url_for("template_editor", template_id="new"))

        # Performance and impact reporting route
        @app.route("/reports")
        def reports():
            # Simulated data for performance and impact reports

            # Overall metrics
            overall_metrics = {
                "health_score": 94,
                "healing_success_rate": 88,
                "total_services": 5,
                "healthy_services": 4,
                "total_errors": 28,
                "total_healed": 24,
            }

            # Recent activities
            recent_activities = [
                {
                    "time": "2023-06-15 10:30:45",
                    "service": "example_service",
                    "event": "Error detected: KeyError",
                    "status": "Error",
                    "status_class": "danger",
                },
                {
                    "time": "2023-06-15 10:30:50",
                    "service": "example_service",
                    "event": "Fix generated for KeyError",
                    "status": "Success",
                    "status_class": "success",
                },
                {
                    "time": "2023-06-15 10:31:05",
                    "service": "example_service",
                    "event": "Fix deployed: add dictionary get() with default",
                    "status": "Success",
                    "status_class": "success",
                },
                {
                    "time": "2023-06-15 10:35:22",
                    "service": "auth_service",
                    "event": "Error detected: AttributeError",
                    "status": "Error",
                    "status_class": "danger",
                },
                {
                    "time": "2023-06-15 10:35:30",
                    "service": "auth_service",
                    "event": "Fix generated for AttributeError",
                    "status": "Success",
                    "status_class": "success",
                },
                {
                    "time": "2023-06-15 10:36:15",
                    "service": "auth_service",
                    "event": "Fix deployed: add None check before attribute access",
                    "status": "Success",
                    "status_class": "success",
                },
            ]

            # Service performance data
            service_performance = [
                {
                    "name": "API Service",
                    "response_time": 110,
                    "error_rate": 1.2,
                    "cpu_usage": 38,
                    "memory_usage": 345,
                    "throughput": 125,
                },
                {
                    "name": "Auth Service",
                    "response_time": 85,
                    "error_rate": 0.8,
                    "cpu_usage": 25,
                    "memory_usage": 290,
                    "throughput": 65,
                },
                {
                    "name": "Database Service",
                    "response_time": 45,
                    "error_rate": 2.5,
                    "cpu_usage": 55,
                    "memory_usage": 420,
                    "throughput": 210,
                },
                {
                    "name": "Cache Service",
                    "response_time": 12,
                    "error_rate": 0.5,
                    "cpu_usage": 18,
                    "memory_usage": 180,
                    "throughput": 350,
                },
                {
                    "name": "Worker Service",
                    "response_time": 180,
                    "error_rate": 1.8,
                    "cpu_usage": 42,
                    "memory_usage": 310,
                    "throughput": 45,
                },
            ]

            # Healing activities data
            healing_activities = [
                {
                    "time": "2023-06-15 10:30:50",
                    "service": "example_service",
                    "error_type": "KeyError",
                    "fix_type": "Dictionary get with default",
                    "result": "Success",
                    "result_class": "success",
                    "duration": "1.2s",
                },
                {
                    "time": "2023-06-15 10:35:30",
                    "service": "auth_service",
                    "error_type": "AttributeError",
                    "fix_type": "None check before attribute access",
                    "result": "Success",
                    "result_class": "success",
                    "duration": "0.9s",
                },
                {
                    "time": "2023-06-15 09:15:20",
                    "service": "database_service",
                    "error_type": "ConnectionError",
                    "fix_type": "Connection retry with backoff",
                    "result": "Success",
                    "result_class": "success",
                    "duration": "3.5s",
                },
                {
                    "time": "2023-06-15 08:42:10",
                    "service": "api_service",
                    "error_type": "ValidationError",
                    "fix_type": "Add validation check",
                    "result": "Success",
                    "result_class": "success",
                    "duration": "1.8s",
                },
                {
                    "time": "2023-06-15 08:12:45",
                    "service": "worker_service",
                    "error_type": "MemoryError",
                    "fix_type": "Add chunked processing",
                    "result": "Partial",
                    "result_class": "warning",
                    "duration": "4.2s",
                },
            ]

            # Recent deployments data
            recent_deployments = [
                {
                    "time": "2023-06-15 10:31:05",
                    "service": "example_service",
                    "fix_id": "fix_001",
                    "type": "Canary",
                    "status": "Success",
                    "status_class": "success",
                    "impact": "Positive",
                    "impact_class": "success",
                },
                {
                    "time": "2023-06-15 10:36:15",
                    "service": "auth_service",
                    "fix_id": "fix_002",
                    "type": "Blue-Green",
                    "status": "Success",
                    "status_class": "success",
                    "impact": "Positive",
                    "impact_class": "success",
                },
                {
                    "time": "2023-06-15 09:22:30",
                    "service": "database_service",
                    "fix_id": "fix_003",
                    "type": "Rolling",
                    "status": "Success",
                    "status_class": "success",
                    "impact": "Positive",
                    "impact_class": "success",
                },
                {
                    "time": "2023-06-15 08:50:15",
                    "service": "api_service",
                    "fix_id": "fix_004",
                    "type": "Canary",
                    "status": "Success",
                    "status_class": "success",
                    "impact": "Neutral",
                    "impact_class": "secondary",
                },
                {
                    "time": "2023-06-15 08:20:10",
                    "service": "worker_service",
                    "fix_id": "fix_005",
                    "type": "Canary",
                    "status": "Rolled Back",
                    "status_class": "danger",
                    "impact": "Negative",
                    "impact_class": "danger",
                },
            ]

            # Service health data
            service_health = [
                {
                    "name": "API Service",
                    "status": "Healthy",
                    "status_class": "success",
                    "response_time": 110,
                    "error_rate": 1.2,
                    "uptime": "99.98%",
                    "health_score": 95,
                    "score_class": "success",
                },
                {
                    "name": "Auth Service",
                    "status": "Healthy",
                    "status_class": "success",
                    "response_time": 85,
                    "error_rate": 0.8,
                    "uptime": "99.99%",
                    "health_score": 98,
                    "score_class": "success",
                },
                {
                    "name": "Database Service",
                    "status": "Warning",
                    "status_class": "warning",
                    "response_time": 45,
                    "error_rate": 2.5,
                    "uptime": "99.95%",
                    "health_score": 90,
                    "score_class": "warning",
                },
                {
                    "name": "Cache Service",
                    "status": "Healthy",
                    "status_class": "success",
                    "response_time": 12,
                    "error_rate": 0.5,
                    "uptime": "100.00%",
                    "health_score": 99,
                    "score_class": "success",
                },
                {
                    "name": "Worker Service",
                    "status": "Healthy",
                    "status_class": "success",
                    "response_time": 180,
                    "error_rate": 1.8,
                    "uptime": "99.90%",
                    "health_score": 92,
                    "score_class": "success",
                },
            ]

            # Audit events data
            audit_events = [
                {
                    "id": "event-001",
                    "time": "2023-06-15 10:30:45",
                    "user": "system",
                    "event_type": "error_detected",
                    "service": "example_service",
                    "severity": "warning",
                    "severity_class": "warning",
                },
                {
                    "id": "event-002",
                    "time": "2023-06-15 10:30:50",
                    "user": "system",
                    "event_type": "fix_generated",
                    "service": "example_service",
                    "severity": "info",
                    "severity_class": "info",
                },
                {
                    "id": "event-003",
                    "time": "2023-06-15 10:31:05",
                    "user": "system",
                    "event_type": "fix_deployed",
                    "service": "example_service",
                    "severity": "info",
                    "severity_class": "info",
                },
                {
                    "id": "event-004",
                    "time": "2023-06-15 10:35:22",
                    "user": "system",
                    "event_type": "error_detected",
                    "service": "auth_service",
                    "severity": "warning",
                    "severity_class": "warning",
                },
                {
                    "id": "event-005",
                    "time": "2023-06-15 10:35:30",
                    "user": "system",
                    "event_type": "fix_generated",
                    "service": "auth_service",
                    "severity": "info",
                    "severity_class": "info",
                },
                {
                    "id": "event-006",
                    "time": "2023-06-15 10:36:15",
                    "user": "system",
                    "event_type": "fix_deployed",
                    "service": "auth_service",
                    "severity": "info",
                    "severity_class": "info",
                },
                {
                    "id": "event-007",
                    "time": "2023-06-15 10:40:12",
                    "user": "admin",
                    "event_type": "login",
                    "service": "dashboard",
                    "severity": "info",
                    "severity_class": "info",
                },
                {
                    "id": "event-008",
                    "time": "2023-06-15 10:42:30",
                    "user": "admin",
                    "event_type": "config_change",
                    "service": "dashboard",
                    "severity": "info",
                    "severity_class": "info",
                },
            ]

            # Recent reports data
            recent_reports = [
                {
                    "time": "2023-06-15 10:45:30",
                    "type": "System Overview",
                    "format": "PDF",
                    "time_range": "Last 24 Hours",
                    "url": "#",
                },
                {
                    "time": "2023-06-15 09:30:15",
                    "type": "Performance Metrics",
                    "format": "Excel",
                    "time_range": "Last 7 Days",
                    "url": "#",
                },
                {
                    "time": "2023-06-15 08:15:45",
                    "type": "Healing Activities",
                    "format": "PDF",
                    "time_range": "Last 24 Hours",
                    "url": "#",
                },
                {
                    "time": "2023-06-14 16:20:10",
                    "type": "Complete Report",
                    "format": "PDF",
                    "time_range": "Last 7 Days",
                    "url": "#",
                },
                {
                    "time": "2023-06-14 10:10:30",
                    "type": "Audit Events",
                    "format": "CSV",
                    "time_range": "Last 30 Days",
                    "url": "#",
                },
            ]

            # Available services for filtering
            services = [
                {"id": "api_service", "name": "API Service"},
                {"id": "auth_service", "name": "Auth Service"},
                {"id": "database_service", "name": "Database Service"},
                {"id": "cache_service", "name": "Cache Service"},
                {"id": "worker_service", "name": "Worker Service"},
            ]

            return render_template(
                "reports.html",
                overall_metrics=overall_metrics,
                recent_activities=recent_activities,
                service_performance=service_performance,
                healing_activities=healing_activities,
                recent_deployments=recent_deployments,
                service_health=service_health,
                audit_events=audit_events,
                recent_reports=recent_reports,
                services=services,
            )

        # Authentication routes
        @app.route("/login", methods=["GET", "POST"])
        def login():
            if request.method == "POST":
                username = request.form.get("username")
                password = request.form.get("password")

                if self.auth_manager:
                    user = self.auth_manager.authenticate(username, password)
                    if user:
                        # Generate token
                        access_token, refresh_token = self.auth_manager.generate_token(
                            user
                        )

                        # Set token in cookies or session
                        response = redirect(url_for("index"))
                        response.set_cookie("access_token", access_token, httponly=True)

                        return response
                    else:
                        return render_template(
                            "login.html", error="Invalid username or password"
                        )
                else:
                    # Authentication disabled, just redirect to index
                    return redirect(url_for("index"))
            else:
                return render_template("login.html")

        @app.route("/logout")
        def logout():
            # If using tokens, blacklist the current token
            if self.auth_manager:
                auth_header = request.headers.get("Authorization", "")
                token = (
                    auth_header.replace("Bearer ", "")
                    if auth_header.startswith("Bearer ")
                    else None
                )

                if token:
                    self.auth_manager.revoke_token(token)

            # Clear token cookie
            response = redirect(url_for("login"))
            response.delete_cookie("access_token")

            return response

        # API routes
        @app.route("/api/status")
        def api_status():
            return jsonify({"status": "ok", "version": "0.1.0"})

        @app.route("/api/errors")
        def api_errors():
            # Simulate error data for now
            # In a real implementation, this would query the Homeostasis system
            errors = [
                {
                    "id": "err_001",
                    "timestamp": (datetime.now() - timedelta(minutes=30)).isoformat(),
                    "service": "example_service",
                    "error_type": "KeyError",
                    "message": "KeyError: 'user_id'",
                    "status": "fixed",
                    "fix_id": "fix_001",
                },
                {
                    "id": "err_002",
                    "timestamp": (datetime.now() - timedelta(minutes=15)).isoformat(),
                    "service": "example_service",
                    "error_type": "AttributeError",
                    "message": "AttributeError: 'NoneType' object has no attribute 'get'",
                    "status": "analyzing",
                    "fix_id": None,
                },
            ]

            return jsonify({"errors": errors})

        @app.route("/api/fixes")
        def api_fixes():
            # Simulate fix data for now
            # In a real implementation, this would query the Homeostasis system
            fixes = [
                {
                    "id": "fix_001",
                    "timestamp": (datetime.now() - timedelta(minutes=25)).isoformat(),
                    "service": "example_service",
                    "error_id": "err_001",
                    "status": "deployed",
                    "confidence": 0.95,
                    "file_path": "services/example_service/app.py",
                    "description": "Added key check before accessing user_id",
                }
            ]

            return jsonify({"fixes": fixes})

        @app.route("/api/approvals")
        def api_approvals():
            # Simulate approval data for now
            # In a real implementation, this would query the Homeostasis system
            approvals = [
                {
                    "id": "apr_001",
                    "timestamp": (datetime.now() - timedelta(minutes=25)).isoformat(),
                    "fix_id": "fix_001",
                    "status": "approved",
                    "approver": "admin",
                    "comment": "LGTM",
                }
            ]

            return jsonify({"approvals": approvals})

        @app.route("/api/metrics")
        def api_metrics():
            # Get metrics from metrics collector
            # For now, simulate metrics data
            metrics = {
                "error_rate": {
                    "current": 0.02,
                    "history": [0.05, 0.04, 0.03, 0.02, 0.02],
                },
                "response_time": {"current": 125, "history": [140, 135, 130, 128, 125]},
                "fix_success_rate": {
                    "current": 0.95,
                    "history": [0.90, 0.92, 0.94, 0.95, 0.95],
                },
            }

            return jsonify({"metrics": metrics})

        @app.route("/api/canary")
        def api_canary():
            # Get canary deployment status
            # For now, simulate canary data
            canary = {
                "active": True,
                "service": "example_service",
                "fix_id": "fix_001",
                "status": "in_progress",
                "current_percentage": 50,
                "metrics": {
                    "error_rate": 0.01,
                    "success_rate": 0.99,
                    "response_time": 120,
                },
            }

            return jsonify({"canary": canary})

        # Suggestion API routes
        @app.route("/api/suggestions/<error_id>")
        def api_suggestions(error_id):
            """Get fix suggestions for an error."""
            suggestions = self.suggestion_manager.get_suggestions(error_id)
            suggestion_data = [suggestion.to_dict() for suggestion in suggestions]
            return jsonify({"suggestions": suggestion_data})

        @app.route("/api/suggestions/<suggestion_id>/approve", methods=["POST"])
        def api_approve_suggestion(suggestion_id):
            """Approve a suggestion."""
            data = request.json or {}
            reviewer = data.get("reviewer", "admin")
            comments = data.get("comments")

            suggestion = self.suggestion_manager.review_suggestion(
                suggestion_id, SuggestionStatus.APPROVED, reviewer, comments
            )

            if suggestion:
                return jsonify({"success": True, "suggestion": suggestion.to_dict()})
            else:
                return (
                    jsonify(
                        {
                            "success": False,
                            "message": f"Failed to approve suggestion {suggestion_id}",
                        }
                    ),
                    404,
                )

        @app.route("/api/suggestions/<suggestion_id>/reject", methods=["POST"])
        def api_reject_suggestion(suggestion_id):
            """Reject a suggestion."""
            data = request.json or {}
            reviewer = data.get("reviewer", "admin")
            comments = data.get("comments")

            suggestion = self.suggestion_manager.review_suggestion(
                suggestion_id, SuggestionStatus.REJECTED, reviewer, comments
            )

            if suggestion:
                return jsonify({"success": True, "suggestion": suggestion.to_dict()})
            else:
                return (
                    jsonify(
                        {
                            "success": False,
                            "message": f"Failed to reject suggestion {suggestion_id}",
                        }
                    ),
                    404,
                )

        @app.route("/api/suggestions/<suggestion_id>/modify", methods=["POST"])
        def api_modify_suggestion(suggestion_id):
            """Modify a suggestion."""
            data = request.json or {}
            suggested_code = data.get("suggested_code")
            reviewer = data.get("reviewer", "admin")
            comments = data.get("comments")

            if not suggested_code:
                return (
                    jsonify(
                        {
                            "success": False,
                            "message": "Missing required field: suggested_code",
                        }
                    ),
                    400,
                )

            suggestion = self.suggestion_manager.modify_suggestion(
                suggestion_id, suggested_code, reviewer, comments
            )

            if suggestion:
                return jsonify({"success": True, "suggestion": suggestion.to_dict()})
            else:
                return (
                    jsonify(
                        {
                            "success": False,
                            "message": f"Failed to modify suggestion {suggestion_id}",
                        }
                    ),
                    404,
                )

        @app.route("/api/suggestions/<suggestion_id>/deploy", methods=["POST"])
        def api_deploy_suggestion(suggestion_id):
            """Deploy a suggestion."""
            suggestion = self.suggestion_manager.mark_deployed(suggestion_id)

            if suggestion:
                return jsonify({"success": True, "suggestion": suggestion.to_dict()})
            else:
                return (
                    jsonify(
                        {
                            "success": False,
                            "message": f"Failed to deploy suggestion {suggestion_id}",
                        }
                    ),
                    404,
                )

        # LLM Key Management API Endpoints
        @app.route("/api/llm-keys", methods=["GET"])
        def api_get_llm_keys():
            """Get the status of all LLM API keys (without revealing the actual keys)."""
            try:
                key_manager = APIKeyManager()
                keys_status = key_manager.list_keys()

                # Get available secrets managers
                secrets_managers = key_manager.get_available_secrets_managers()

                return jsonify(
                    {
                        "success": True,
                        "providers": keys_status,
                        "secrets_managers": secrets_managers,
                    }
                )
            except Exception as e:
                return (
                    jsonify(
                        {
                            "success": False,
                            "message": f"Failed to get LLM keys status: {str(e)}",
                        }
                    ),
                    500,
                )

        @app.route("/api/llm-keys/<provider>", methods=["POST"])
        def api_set_llm_key(provider):
            """Set API key for a specific provider."""
            try:
                data = request.get_json()
                if not data or "api_key" not in data:
                    return (
                        jsonify({"success": False, "message": "API key is required"}),
                        400,
                    )

                api_key = data["api_key"].strip()
                if not api_key:
                    return (
                        jsonify(
                            {"success": False, "message": "API key cannot be empty"}
                        ),
                        400,
                    )

                key_manager = APIKeyManager()

                # Validate the key
                try:
                    is_valid = key_manager.validate_key(provider, api_key)
                    if not is_valid:
                        return (
                            jsonify(
                                {
                                    "success": False,
                                    "message": f"Invalid {provider} API key",
                                }
                            ),
                            400,
                        )
                except KeyValidationError as e:
                    return jsonify({"success": False, "message": str(e)}), 400

                # Store the key
                success = key_manager.set_key(provider, api_key)
                if success:
                    return jsonify(
                        {
                            "success": True,
                            "message": f"{provider.title()} API key set successfully",
                        }
                    )
                else:
                    return (
                        jsonify(
                            {
                                "success": False,
                                "message": f"Failed to store {provider} API key",
                            }
                        ),
                        500,
                    )

            except Exception as e:
                return (
                    jsonify(
                        {
                            "success": False,
                            "message": f"Failed to set {provider} API key: {str(e)}",
                        }
                    ),
                    500,
                )

        @app.route("/api/llm-keys/<provider>/test", methods=["POST"])
        def api_test_llm_key(provider):
            """Test an API key for a specific provider."""
            try:
                data = request.get_json() or {}
                api_key = data.get("api_key", "").strip()

                key_manager = APIKeyManager()

                # If no key is provided in the request, use the stored key
                if not api_key:
                    api_key = key_manager.get_key(provider)
                    if not api_key:
                        return (
                            jsonify(
                                {
                                    "success": False,
                                    "message": f"No {provider} API key found to test",
                                }
                            ),
                            400,
                        )

                # Test the key
                try:
                    is_valid = key_manager.validate_key(provider, api_key)
                    if is_valid:
                        return jsonify(
                            {
                                "success": True,
                                "message": f"{provider.title()} API key is valid",
                                "provider": provider,
                            }
                        )
                    else:
                        return (
                            jsonify(
                                {
                                    "success": False,
                                    "message": f"{provider.title()} API key validation failed",
                                }
                            ),
                            400,
                        )
                except KeyValidationError as e:
                    return jsonify({"success": False, "message": str(e)}), 400

            except Exception as e:
                return (
                    jsonify(
                        {
                            "success": False,
                            "message": f"Failed to test {provider} API key: {str(e)}",
                        }
                    ),
                    500,
                )

        @app.route("/api/llm-keys/<provider>", methods=["DELETE"])
        def api_remove_llm_key(provider):
            """Remove API key for a specific provider."""
            try:
                key_manager = APIKeyManager()
                success = key_manager.remove_key(provider)

                if success:
                    return jsonify(
                        {
                            "success": True,
                            "message": f"{provider.title()} API key removed successfully",
                        }
                    )
                else:
                    return (
                        jsonify(
                            {
                                "success": False,
                                "message": f"No {provider} API key found to remove",
                            }
                        ),
                        404,
                    )

            except Exception as e:
                return (
                    jsonify(
                        {
                            "success": False,
                            "message": f"Failed to remove {provider} API key: {str(e)}",
                        }
                    ),
                    500,
                )

        @app.route("/api/llm-keys/test-all", methods=["POST"])
        def api_test_all_llm_keys():
            """Test all configured LLM API keys."""
            try:
                key_manager = APIKeyManager()
                results = {}

                for provider in key_manager.SUPPORTED_PROVIDERS:
                    api_key = key_manager.get_key(provider)
                    if api_key:
                        try:
                            is_valid = key_manager.validate_key(provider, api_key)
                            results[provider] = {
                                "success": is_valid,
                                "message": (
                                    f"{provider.title()} API key is valid"
                                    if is_valid
                                    else f"{provider.title()} API key is invalid"
                                ),
                            }
                        except KeyValidationError as e:
                            results[provider] = {"success": False, "message": str(e)}
                        except Exception as e:
                            results[provider] = {
                                "success": False,
                                "message": f"Error testing {provider}: {str(e)}",
                            }
                    else:
                        results[provider] = {
                            "success": False,
                            "message": f"No {provider} API key configured",
                        }

                return jsonify({"success": True, "results": results})

            except Exception as e:
                return (
                    jsonify(
                        {
                            "success": False,
                            "message": f"Failed to test API keys: {str(e)}",
                        }
                    ),
                    500,
                )

        # LLM Observability API Endpoints
        @app.route("/api/llm/metrics", methods=["GET"])
        def api_llm_metrics():
            """Get LLM usage metrics."""
            try:
                time_window = request.args.get("time_window", 3600, type=int)
                usage_stats = self.llm_metrics_collector.get_current_usage()
                provider_performance = (
                    self.llm_metrics_collector.get_provider_performance(time_window)
                )

                return jsonify(
                    {
                        "success": True,
                        "usage_statistics": usage_stats,
                        "provider_performance": provider_performance,
                        "time_window": time_window,
                    }
                )
            except Exception as e:
                return (
                    jsonify(
                        {
                            "success": False,
                            "message": f"Failed to get LLM metrics: {str(e)}",
                        }
                    ),
                    500,
                )

        @app.route("/api/llm/costs", methods=["GET"])
        def api_llm_costs():
            """Get LLM cost breakdown and budget usage."""
            try:
                time_window = request.args.get("time_window", 86400, type=int)

                # Get cost breakdown
                cost_breakdown = self.llm_metrics_collector.get_cost_breakdown(
                    time_window
                )

                # Get budget usage
                budget_usage = self.cost_tracker.get_current_usage()

                # Get cost trends
                cost_trends = self.cost_tracker.get_cost_trends(days=7)

                # Get optimization recommendations
                recommendations = self.cost_tracker.get_optimization_recommendations()

                return jsonify(
                    {
                        "success": True,
                        "cost_breakdown": cost_breakdown,
                        "budget_usage": budget_usage,
                        "cost_trends": cost_trends,
                        "recommendations": recommendations,
                        "time_window": time_window,
                    }
                )
            except Exception as e:
                return (
                    jsonify(
                        {
                            "success": False,
                            "message": f"Failed to get LLM costs: {str(e)}",
                        }
                    ),
                    500,
                )

        @app.route("/api/llm/security", methods=["GET"])
        def api_llm_security():
            """Get LLM security report."""
            try:
                time_window = request.args.get("time_window", 86400, type=int)
                security_report = self.security_guardrails.get_security_report(
                    time_window
                )

                return jsonify(
                    {
                        "success": True,
                        "security_report": security_report,
                        "time_window": time_window,
                    }
                )
            except Exception as e:
                return (
                    jsonify(
                        {
                            "success": False,
                            "message": f"Failed to get LLM security report: {str(e)}",
                        }
                    ),
                    500,
                )

        @app.route("/api/llm/security/analyze", methods=["POST"])
        def api_llm_security_analyze():
            """Analyze content for security violations."""
            try:
                data = request.get_json()
                if not data or "content" not in data:
                    return (
                        jsonify({"success": False, "message": "Content is required"}),
                        400,
                    )

                content = data["content"]
                source = data.get("source", "unknown")
                context = data.get("context", {})

                is_safe, violations, scrubbed_content = (
                    self.security_guardrails.analyze_content(content, source, context)
                )

                violation_data = [
                    {
                        "type": v.violation_type.value,
                        "severity": v.severity,
                        "description": v.description,
                        "confidence": v.confidence,
                        "patterns_matched": v.patterns_matched,
                    }
                    for v in violations
                ]

                return jsonify(
                    {
                        "success": True,
                        "is_safe": is_safe,
                        "violations": violation_data,
                        "scrubbed_content": scrubbed_content,
                        "original_length": len(content),
                        "scrubbed_length": len(scrubbed_content),
                    }
                )
            except Exception as e:
                return (
                    jsonify(
                        {
                            "success": False,
                            "message": f"Failed to analyze content: {str(e)}",
                        }
                    ),
                    500,
                )

        @app.route("/api/llm/budgets", methods=["GET"])
        def api_llm_budgets():
            """Get budget configurations."""
            try:
                budget_usage = self.cost_tracker.get_current_usage()
                return jsonify({"success": True, "budgets": budget_usage})
            except Exception as e:
                return (
                    jsonify(
                        {
                            "success": False,
                            "message": f"Failed to get budgets: {str(e)}",
                        }
                    ),
                    500,
                )

        @app.route("/api/llm/budgets", methods=["POST"])
        def api_llm_budget_create():
            """Create a new budget."""
            try:
                data = request.get_json()
                if not data or "amount" not in data or "period" not in data:
                    return (
                        jsonify(
                            {
                                "success": False,
                                "message": "Amount and period are required",
                            }
                        ),
                        400,
                    )

                budget = Budget(
                    amount=float(data["amount"]),
                    period=BudgetPeriod(data["period"]),
                    alert_thresholds=data.get(
                        "alert_thresholds", [50.0, 75.0, 90.0, 100.0]
                    ),
                    hard_limit=data.get("hard_limit", False),
                    rollover=data.get("rollover", False),
                )

                budget_id = f"budget_{int(time.time())}"
                self.cost_tracker.add_budget(budget_id, budget)

                return jsonify(
                    {
                        "success": True,
                        "budget_id": budget_id,
                        "message": "Budget created successfully",
                    }
                )
            except Exception as e:
                return (
                    jsonify(
                        {
                            "success": False,
                            "message": f"Failed to create budget: {str(e)}",
                        }
                    ),
                    500,
                )

        @app.route("/api/llm/budgets/<budget_id>", methods=["DELETE"])
        def api_llm_budget_delete(budget_id):
            """Delete a budget."""
            try:
                self.cost_tracker.remove_budget(budget_id)
                return jsonify(
                    {
                        "success": True,
                        "message": f"Budget {budget_id} deleted successfully",
                    }
                )
            except Exception as e:
                return (
                    jsonify(
                        {
                            "success": False,
                            "message": f"Failed to delete budget: {str(e)}",
                        }
                    ),
                    500,
                )

        @app.route("/api/llm/export/metrics", methods=["POST"])
        def api_llm_export_metrics():
            """Export LLM metrics to file."""
            try:
                data = request.get_json() or {}
                time_window = data.get("time_window", 86400)
                filename = data.get("filename", f"llm_metrics_{int(time.time())}.json")

                # Use secure temporary directory
                output_file = Path(tempfile.gettempdir()) / filename
                self.llm_metrics_collector.export_metrics(output_file, time_window)

                return jsonify(
                    {
                        "success": True,
                        "filename": filename,
                        "message": "Metrics exported successfully",
                    }
                )
            except Exception as e:
                return (
                    jsonify(
                        {
                            "success": False,
                            "message": f"Failed to export metrics: {str(e)}",
                        }
                    ),
                    500,
                )

        @app.route("/api/llm/export/security", methods=["POST"])
        def api_llm_export_security():
            """Export security logs to file."""
            try:
                data = request.get_json() or {}
                time_window = data.get("time_window", 86400)
                filename = data.get("filename", f"llm_security_{int(time.time())}.json")

                # Use secure temporary directory
                output_file = Path(tempfile.gettempdir()) / filename
                self.security_guardrails.export_security_logs(output_file, time_window)

                return jsonify(
                    {
                        "success": True,
                        "filename": filename,
                        "message": "Security logs exported successfully",
                    }
                )
            except Exception as e:
                return (
                    jsonify(
                        {
                            "success": False,
                            "message": f"Failed to export security logs: {str(e)}",
                        }
                    ),
                    500,
                )

        @app.route("/api/llm/export/costs", methods=["POST"])
        def api_llm_export_costs():
            """Export cost report to file."""
            try:
                data = request.get_json() or {}
                days = data.get("days", 30)
                filename = data.get("filename", f"llm_costs_{int(time.time())}.json")

                # Use secure temporary directory
                output_file = Path(tempfile.gettempdir()) / filename
                self.cost_tracker.export_cost_report(output_file, days)

                return jsonify(
                    {
                        "success": True,
                        "filename": filename,
                        "message": "Cost report exported successfully",
                    }
                )
            except Exception as e:
                return (
                    jsonify(
                        {
                            "success": False,
                            "message": f"Failed to export cost report: {str(e)}",
                        }
                    ),
                    500,
                )

        # Mark routes as registered to prevent duplicate registration
        app._routes_registered = True

    def _register_error_handlers(self) -> None:
        """Register error handlers for the dashboard."""
        # Check if error handlers are already registered
        if (
            hasattr(app, "_error_handlers_registered")
            and app._error_handlers_registered
        ):
            return

        @app.errorhandler(404)
        def page_not_found(e):
            return render_template("error.html", error=e), 404

        @app.errorhandler(500)
        def server_error(e):
            return render_template("error.html", error=e), 500

        # Mark error handlers as registered to prevent duplicate registration
        app._error_handlers_registered = True

    def _register_socketio_handlers(self) -> None:
        """Register SocketIO event handlers for real-time updates."""
        # Check if socketio handlers are already registered
        if hasattr(socketio, "_handlers_registered") and socketio._handlers_registered:
            return

        @socketio.on("connect")
        def handle_connect():
            logger.info(f"Client connected: {request.sid}")

        @socketio.on("disconnect")
        def handle_disconnect():
            logger.info(f"Client disconnected: {request.sid}")

        @socketio.on("subscribe")
        def handle_subscribe(data):
            """
            Handle subscription requests for real-time updates.

            Args:
                data: Dictionary with subscription topics
            """
            topics = data.get("topics", [])
            logger.info(f"Client {request.sid} subscribed to topics: {topics}")

            # Join rooms for each topic
            for topic in topics:
                flask.session[f"subscribed_{topic}"] = True

            # Send initial data for subscribed topics
            self._send_initial_data(topics)

    def _send_initial_data(self, topics: List[str]) -> None:
        """
        Send initial data for subscribed topics.

        Args:
            topics: List of topics to send data for
        """
        # Send initial data for each topic
        for topic in topics:
            if topic == "errors":
                # Simulate error data
                errors = [
                    {
                        "id": "err_001",
                        "timestamp": (
                            datetime.now() - timedelta(minutes=30)
                        ).isoformat(),
                        "service": "example_service",
                        "error_type": "KeyError",
                        "message": "KeyError: 'user_id'",
                        "status": "fixed",
                        "fix_id": "fix_001",
                    },
                    {
                        "id": "err_002",
                        "timestamp": (
                            datetime.now() - timedelta(minutes=15)
                        ).isoformat(),
                        "service": "example_service",
                        "error_type": "AttributeError",
                        "message": "AttributeError: 'NoneType' object has no attribute 'get'",
                        "status": "analyzing",
                        "fix_id": None,
                    },
                ]

                emit("errors_update", {"errors": errors})

            elif topic == "fixes":
                # Simulate fix data
                fixes = [
                    {
                        "id": "fix_001",
                        "timestamp": (
                            datetime.now() - timedelta(minutes=25)
                        ).isoformat(),
                        "service": "example_service",
                        "error_id": "err_001",
                        "status": "deployed",
                        "confidence": 0.95,
                        "file_path": "services/example_service/app.py",
                        "description": "Added key check before accessing user_id",
                    }
                ]

                emit("fixes_update", {"fixes": fixes})

            elif topic == "metrics":
                # Simulate metrics data
                metrics = {
                    "error_rate": {
                        "current": 0.02,
                        "history": [0.05, 0.04, 0.03, 0.02, 0.02],
                    },
                    "response_time": {
                        "current": 125,
                        "history": [140, 135, 130, 128, 125],
                    },
                    "fix_success_rate": {
                        "current": 0.95,
                        "history": [0.90, 0.92, 0.94, 0.95, 0.95],
                    },
                }

                emit("metrics_update", {"metrics": metrics})

        # Mark socketio handlers as registered to prevent duplicate registration
        socketio._handlers_registered = True

    def run(self) -> None:
        """Run the dashboard server."""
        host = self.config.get("server", {}).get("host", "127.0.0.1")
        port = self.config.get("server", {}).get("port", 5000)

        logger.info(f"Starting dashboard server on {host}:{port}")
        socketio.run(app, host=host, port=port, debug=self.debug)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Homeostasis Dashboard")
    parser.add_argument("--config", "-c", type=str, help="Path to configuration file")
    parser.add_argument("--debug", "-d", action="store_true", help="Run in debug mode")

    args = parser.parse_args()

    # Determine the config path
    config_path = Path(args.config) if args.config else None

    # Create and run the dashboard server
    server = DashboardServer(config_path=config_path, debug=args.debug)
    server.run()


if __name__ == "__main__":
    main()
