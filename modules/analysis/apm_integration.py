"""
APM (Application Performance Monitoring) integration for Homeostasis.

This module provides integration with popular APM tools to extend the monitoring
and analysis capabilities of Homeostasis with external data sources.
"""

import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import requests

# Configure logging
logger = logging.getLogger(__name__)


class APMProvider(Enum):
    """Supported APM providers."""

    DATADOG = "datadog"
    NEW_RELIC = "new_relic"
    DYNATRACE = "dynatrace"
    ELASTIC_APM = "elastic_apm"
    PROMETHEUS = "prometheus"
    GRAFANA = "grafana"
    APP_DYNAMICS = "app_dynamics"
    INSTANA = "instana"
    PINPOINT = "pinpoint"
    SKYWALKING = "skywalking"
    CUSTOM = "custom"


@dataclass
class APMConfig:
    """Configuration for APM integration."""

    provider: APMProvider  # APM provider
    api_key: str  # API key for authentication
    api_endpoint: Optional[str] = None  # API endpoint URL
    app_id: Optional[str] = None  # Application ID
    environment: Optional[str] = None  # Environment (prod, staging, etc.)
    service_name: Optional[str] = None  # Service name
    additional_params: Dict[str, Any] = field(
        default_factory=dict
    )  # Additional parameters
    timeout: int = 30  # Request timeout in seconds


class APMIntegration(ABC):
    """Base class for APM integrations."""

    def __init__(self, config: APMConfig):
        """
        Initialize the APM integration.

        Args:
            config: APM configuration
        """
        self.config = config
        self.session = None
        self._initialize_session()

    def _initialize_session(self):
        """Initialize the HTTP session for API requests."""
        self.session = requests.Session()

        # Set default headers
        self.session.headers.update(
            {"Content-Type": "application/json", "Accept": "application/json"}
        )

        # Set up authentication based on provider
        if self.config.provider == APMProvider.DATADOG:
            self.session.headers.update(
                {
                    "DD-API-KEY": self.config.api_key,
                    "DD-APPLICATION-KEY": self.config.additional_params.get(
                        "app_key", ""
                    ),
                }
            )
        elif self.config.provider == APMProvider.NEW_RELIC:
            self.session.headers.update({"X-Api-Key": self.config.api_key})
        elif self.config.provider == APMProvider.DYNATRACE:
            self.session.headers.update(
                {"Authorization": f"Api-Token {self.config.api_key}"}
            )
        elif self.config.provider == APMProvider.ELASTIC_APM:
            self.session.headers.update(
                {"Authorization": f"ApiKey {self.config.api_key}"}
            )
        else:
            # Generic API key header
            self.session.headers.update(
                {"Authorization": f"Bearer {self.config.api_key}"}
            )

    @abstractmethod
    def get_metrics(
        self,
        metric_names: List[str],
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        interval: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get metrics from the APM provider.

        Args:
            metric_names: List of metric names to retrieve
            start_time: Start timestamp in epoch seconds
            end_time: End timestamp in epoch seconds
            interval: Time interval for aggregation

        Returns:
            Dictionary with metric data
        """
        pass

    @abstractmethod
    def get_traces(
        self,
        filter_params: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Get distributed traces from the APM provider.

        Args:
            filter_params: Parameters to filter traces
            limit: Maximum number of traces to retrieve
            start_time: Start timestamp in epoch seconds
            end_time: End timestamp in epoch seconds

        Returns:
            Dictionary with trace data
        """
        pass

    @abstractmethod
    def get_errors(
        self,
        filter_params: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Get error events from the APM provider.

        Args:
            filter_params: Parameters to filter errors
            limit: Maximum number of errors to retrieve
            start_time: Start timestamp in epoch seconds
            end_time: End timestamp in epoch seconds

        Returns:
            Dictionary with error data
        """
        pass

    def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Make an HTTP request to the APM API.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            params: Query parameters
            data: Request body data

        Returns:
            API response as dictionary
        """
        url = self._get_full_url(endpoint)

        try:
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                json=data,
                timeout=self.config.timeout,
            )

            response.raise_for_status()

            if response.content:
                return response.json()
            return {}
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            return {"error": str(e)}

    def _get_full_url(self, endpoint: str) -> str:
        """
        Get the full URL for an API endpoint.

        Args:
            endpoint: API endpoint

        Returns:
            Full URL
        """
        base_url = self.config.api_endpoint

        # Strip trailing slash from base URL if present
        if base_url and base_url.endswith("/"):
            base_url = base_url[:-1]

        # Strip leading slash from endpoint if present
        if endpoint.startswith("/"):
            endpoint = endpoint[1:]

        return f"{base_url}/{endpoint}"

    def test_connection(self) -> Dict[str, Any]:
        """
        Test the connection to the APM provider.

        Returns:
            Dictionary with test results
        """
        try:
            # This method should be overridden by subclasses
            # Default implementation uses a basic health check
            start_time = time.time()
            response = self._make_request("GET", "health")
            elapsed = time.time() - start_time

            return {
                "success": "error" not in response,
                "latency_ms": int(elapsed * 1000),
                "response": response,
            }
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return {"success": False, "error": str(e)}


class DatadogIntegration(APMIntegration):
    """Integration with Datadog APM."""

    def get_metrics(
        self,
        metric_names: List[str],
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        interval: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get metrics from Datadog.

        Args:
            metric_names: List of metric names to retrieve
            start_time: Start timestamp in epoch seconds
            end_time: End timestamp in epoch seconds
            interval: Time interval for aggregation

        Returns:
            Dictionary with metric data
        """
        # Default time range if not specified
        if not end_time:
            end_time = int(time.time())
        if not start_time:
            start_time = end_time - 3600  # Last hour

        params = {"from": start_time, "to": end_time, "query": " ".join(metric_names)}

        if interval:
            params["interval"] = interval

        return self._make_request(method="GET", endpoint="api/v1/query", params=params)

    def get_traces(
        self,
        filter_params: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Get distributed traces from Datadog.

        Args:
            filter_params: Parameters to filter traces
            limit: Maximum number of traces to retrieve
            start_time: Start timestamp in epoch seconds
            end_time: End timestamp in epoch seconds

        Returns:
            Dictionary with trace data
        """
        # Default time range if not specified
        if not end_time:
            end_time = int(time.time())
        if not start_time:
            start_time = end_time - 3600  # Last hour

        params = {"start": start_time, "end": end_time, "limit": limit}

        if filter_params:
            if "service" in filter_params:
                params["service"] = filter_params["service"]
            if "operation_name" in filter_params:
                params["operation_name"] = filter_params["operation_name"]
            if "resource_name" in filter_params:
                params["resource_name"] = filter_params["resource_name"]
            if "status" in filter_params:
                params["status"] = filter_params["status"]

        return self._make_request(method="GET", endpoint="api/v1/traces", params=params)

    def get_errors(
        self,
        filter_params: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Get error events from Datadog.

        Args:
            filter_params: Parameters to filter errors
            limit: Maximum number of errors to retrieve
            start_time: Start timestamp in epoch seconds
            end_time: End timestamp in epoch seconds

        Returns:
            Dictionary with error data
        """
        # Default time range if not specified
        if not end_time:
            end_time = int(time.time())
        if not start_time:
            start_time = end_time - 3600  # Last hour

        # Build query
        query_parts = ["@error:true"]

        if filter_params:
            if "service" in filter_params:
                query_parts.append(f"service:{filter_params['service']}")
            if "status" in filter_params:
                query_parts.append(f"status:{filter_params['status']}")
            if "error_type" in filter_params:
                query_parts.append(f"@error.type:{filter_params['error_type']}")
            if "environment" in filter_params:
                query_parts.append(f"env:{filter_params['environment']}")
        elif self.config.service_name:
            query_parts.append(f"service:{self.config.service_name}")

        if self.config.environment:
            query_parts.append(f"env:{self.config.environment}")

        params = {
            "start": start_time,
            "end": end_time,
            "limit": limit,
            "query": " ".join(query_parts),
        }

        return self._make_request(method="GET", endpoint="api/v1/events", params=params)

    def test_connection(self) -> Dict[str, Any]:
        """
        Test the connection to Datadog.

        Returns:
            Dictionary with test results
        """
        try:
            start_time = time.time()
            response = self._make_request("GET", "api/v1/validate")
            elapsed = time.time() - start_time

            return {
                "success": response.get("valid", False),
                "latency_ms": int(elapsed * 1000),
                "response": response,
            }
        except Exception as e:
            logger.error(f"Datadog connection test failed: {str(e)}")
            return {"success": False, "error": str(e)}


class NewRelicIntegration(APMIntegration):
    """Integration with New Relic APM."""

    def get_metrics(
        self,
        metric_names: List[str],
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        interval: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get metrics from New Relic.

        Args:
            metric_names: List of metric names to retrieve
            start_time: Start timestamp in epoch seconds
            end_time: End timestamp in epoch seconds
            interval: Time interval for aggregation

        Returns:
            Dictionary with metric data
        """
        # Default time range if not specified
        if not end_time:
            end_time = int(time.time() * 1000)  # New Relic uses milliseconds
        if not start_time:
            start_time = end_time - 3600000  # Last hour

        # Validate metric names to prevent SQL injection
        metric_pattern = re.compile(r"^[a-zA-Z0-9._-]+$")
        validated_metrics = []
        for name in metric_names:
            if not metric_pattern.match(name):
                logger.warning(f"Invalid metric name skipped: {name}")
                continue
            validated_metrics.append(name)

        if not validated_metrics:
            return {"error": "No valid metric names provided"}

        # Convert metric names to NRQL format
        metrics_clause = ", ".join([f"average({name})" for name in validated_metrics])

        # Build NRQL query with validated inputs
        query = f"SELECT {metrics_clause} FROM Metric"

        if self.config.app_id:
            # Ensure app_id is an integer to prevent injection
            try:
                app_id_int = int(self.config.app_id)
                query += f" WHERE appId = {app_id_int}"
            except (ValueError, TypeError):
                logger.error(f"Invalid app_id: {self.config.app_id}")
                return {"error": "Invalid app_id"}

        if interval:
            # Validate interval format
            interval_pattern = re.compile(r"^[0-9]+(s|m|h|d)$")
            if interval_pattern.match(interval):
                query += f" TIMESERIES {interval}"
            else:
                logger.warning(f"Invalid interval format: {interval}")

        # Ensure timestamps are integers
        try:
            start_time_int = int(start_time)
            end_time_int = int(end_time)
            query += f" SINCE {start_time_int} UNTIL {end_time_int}"
        except (ValueError, TypeError):
            logger.error("Invalid timestamp values")
            return {"error": "Invalid timestamp values"}

        data = {"query": query}

        return self._make_request(method="POST", endpoint="v2/nrql", data=data)

    def get_traces(
        self,
        filter_params: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Get distributed traces from New Relic.

        Args:
            filter_params: Parameters to filter traces
            limit: Maximum number of traces to retrieve
            start_time: Start timestamp in epoch seconds
            end_time: End timestamp in epoch seconds

        Returns:
            Dictionary with trace data
        """
        # Default time range if not specified
        if not end_time:
            end_time = int(time.time() * 1000)  # New Relic uses milliseconds
        if not start_time:
            start_time = end_time - 3600000  # Last hour

        # Build filter conditions
        conditions = []

        if filter_params:
            if "service" in filter_params:
                conditions.append(f"serviceName = '{filter_params['service']}'")
            if "operation_name" in filter_params:
                conditions.append(f"name = '{filter_params['operation_name']}'")
            if "duration_min" in filter_params:
                conditions.append(f"duration >= {filter_params['duration_min']}")
            if "status" in filter_params:
                conditions.append(f"http.statusCode = '{filter_params['status']}'")

        if self.config.app_id:
            conditions.append(f"appId = {self.config.app_id}")

        # Build NRQL query
        query = "SELECT * FROM DistributedTraceData"

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += f" SINCE {start_time} UNTIL {end_time} LIMIT {limit}"

        data = {"query": query}

        return self._make_request(method="POST", endpoint="v2/nrql", data=data)

    def get_errors(
        self,
        filter_params: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Get error events from New Relic.

        Args:
            filter_params: Parameters to filter errors
            limit: Maximum number of errors to retrieve
            start_time: Start timestamp in epoch seconds
            end_time: End timestamp in epoch seconds

        Returns:
            Dictionary with error data
        """
        # Default time range if not specified
        if not end_time:
            end_time = int(time.time() * 1000)  # New Relic uses milliseconds
        if not start_time:
            start_time = end_time - 3600000  # Last hour

        # Build filter conditions
        conditions = ["error IS TRUE"]

        if filter_params:
            if "service" in filter_params:
                conditions.append(f"serviceName = '{filter_params['service']}'")
            if "error_type" in filter_params:
                conditions.append(f"errorType = '{filter_params['error_type']}'")
            if "error_message" in filter_params:
                conditions.append(
                    f"errorMessage LIKE '%{filter_params['error_message']}%'"
                )

        if self.config.app_id:
            conditions.append(f"appId = {self.config.app_id}")

        # Build NRQL query
        query = "SELECT * FROM TransactionError"

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += f" SINCE {start_time} UNTIL {end_time} LIMIT {limit}"

        data = {"query": query}

        return self._make_request(method="POST", endpoint="v2/nrql", data=data)

    def test_connection(self) -> Dict[str, Any]:
        """
        Test the connection to New Relic.

        Returns:
            Dictionary with test results
        """
        try:
            start_time = time.time()
            # Simple query to check if the API key is valid
            response = self._make_request(method="GET", endpoint="v2/applications")
            elapsed = time.time() - start_time

            return {
                "success": "applications" in response,
                "latency_ms": int(elapsed * 1000),
                "response": response,
            }
        except Exception as e:
            logger.error(f"New Relic connection test failed: {str(e)}")
            return {"success": False, "error": str(e)}


class DynatraceIntegration(APMIntegration):
    """Integration with Dynatrace APM."""

    def get_metrics(
        self,
        metric_names: List[str],
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        interval: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get metrics from Dynatrace.

        Args:
            metric_names: List of metric names to retrieve
            start_time: Start timestamp in epoch seconds
            end_time: End timestamp in epoch seconds
            interval: Time interval for aggregation

        Returns:
            Dictionary with metric data
        """
        # Default time range if not specified
        if not end_time:
            end_time = int(time.time() * 1000)  # Dynatrace uses milliseconds
        if not start_time:
            start_time = end_time - 3600000  # Last hour

        metrics_data = {}

        # Query each metric individually
        for metric_name in metric_names:
            params = {"metricSelector": metric_name, "from": start_time, "to": end_time}

            if interval:
                params["resolution"] = interval

            response = self._make_request(
                method="GET", endpoint="api/v2/metrics/query", params=params
            )

            if "error" not in response:
                metrics_data[metric_name] = response

        return {"metrics": metrics_data}

    def get_traces(
        self,
        filter_params: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Get distributed traces from Dynatrace.

        Args:
            filter_params: Parameters to filter traces
            limit: Maximum number of traces to retrieve
            start_time: Start timestamp in epoch seconds
            end_time: End timestamp in epoch seconds

        Returns:
            Dictionary with trace data
        """
        # Default time range if not specified
        if not end_time:
            end_time = int(time.time() * 1000)  # Dynatrace uses milliseconds
        if not start_time:
            start_time = end_time - 3600000  # Last hour

        params = {"from": start_time, "to": end_time, "top": limit}

        # Add filters
        if filter_params:
            for key, value in filter_params.items():
                if key == "service":
                    params["serviceFilter"] = value
                elif key == "application":
                    params["applicationFilter"] = value
                elif key == "tag":
                    params["tagFilter"] = value

        return self._make_request(method="GET", endpoint="api/v2/traces", params=params)

    def get_errors(
        self,
        filter_params: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Get error events from Dynatrace.

        Args:
            filter_params: Parameters to filter errors
            limit: Maximum number of errors to retrieve
            start_time: Start timestamp in epoch seconds
            end_time: End timestamp in epoch seconds

        Returns:
            Dictionary with error data
        """
        # Default time range if not specified
        if not end_time:
            end_time = int(time.time() * 1000)  # Dynatrace uses milliseconds
        if not start_time:
            start_time = end_time - 3600000  # Last hour

        # Build query string for filtering errors
        query = 'status("ERROR")'

        if filter_params:
            if "service" in filter_params:
                query += f" AND entityName(\"{filter_params['service']}\")"
            if "error_type" in filter_params:
                query += f" AND errorType(\"{filter_params['error_type']}\")"

        params = {
            "from": start_time,
            "to": end_time,
            "eventType": "ERROR_EVENT",
            "entitySelector": query,
            "limit": limit,
        }

        return self._make_request(method="GET", endpoint="api/v2/events", params=params)

    def test_connection(self) -> Dict[str, Any]:
        """
        Test the connection to Dynatrace.

        Returns:
            Dictionary with test results
        """
        try:
            start_time = time.time()
            response = self._make_request(method="GET", endpoint="api/v2/metrics")
            elapsed = time.time() - start_time

            return {
                "success": "metrics" in response,
                "latency_ms": int(elapsed * 1000),
                "response": response,
            }
        except Exception as e:
            logger.error(f"Dynatrace connection test failed: {str(e)}")
            return {"success": False, "error": str(e)}


class ElasticAPMIntegration(APMIntegration):
    """Integration with Elastic APM."""

    def get_metrics(
        self,
        metric_names: List[str],
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        interval: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get metrics from Elastic APM.

        Args:
            metric_names: List of metric names to retrieve
            start_time: Start timestamp in epoch seconds
            end_time: End timestamp in epoch seconds
            interval: Time interval for aggregation

        Returns:
            Dictionary with metric data
        """
        # Default time range if not specified
        if not end_time:
            end_time = int(time.time()) * 1000  # Elastic uses milliseconds
        if not start_time:
            start_time = end_time - 3600000  # Last hour

        # Convert interval to Elastic format if provided
        interval_param = "1m"  # Default
        if interval:
            if interval.endswith("s"):
                interval_param = interval
            elif interval.endswith("m"):
                interval_param = interval
            elif interval.endswith("h"):
                interval_param = interval

        # Build query
        query = {
            "query": {
                "bool": {
                    "must": [
                        {"range": {"@timestamp": {"gte": start_time, "lte": end_time}}}
                    ]
                }
            },
            "aggs": {},
        }

        # Add service filter if available
        if self.config.service_name:
            query["query"]["bool"]["must"].append(
                {"term": {"service.name": self.config.service_name}}
            )

        # Add aggregations for each metric
        for metric_name in metric_names:
            query["aggs"][metric_name] = {
                "avg": {"field": metric_name},
                "date_histogram": {
                    "field": "@timestamp",
                    "fixed_interval": interval_param,
                },
            }

        data = query

        return self._make_request(method="POST", endpoint="apm-*/_search", data=data)

    def get_traces(
        self,
        filter_params: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Get distributed traces from Elastic APM.

        Args:
            filter_params: Parameters to filter traces
            limit: Maximum number of traces to retrieve
            start_time: Start timestamp in epoch seconds
            end_time: End timestamp in epoch seconds

        Returns:
            Dictionary with trace data
        """
        # Default time range if not specified
        if not end_time:
            end_time = int(time.time()) * 1000  # Elastic uses milliseconds
        if not start_time:
            start_time = end_time - 3600000  # Last hour

        # Build query
        query = {
            "query": {
                "bool": {
                    "must": [
                        {"range": {"@timestamp": {"gte": start_time, "lte": end_time}}},
                        {"exists": {"field": "trace.id"}},
                    ]
                }
            },
            "size": limit,
            "sort": [{"@timestamp": {"order": "desc"}}],
        }

        # Add filters
        if filter_params:
            if "service" in filter_params:
                query["query"]["bool"]["must"].append(
                    {"term": {"service.name": filter_params["service"]}}
                )
            if "transaction_name" in filter_params:
                query["query"]["bool"]["must"].append(
                    {"term": {"transaction.name": filter_params["transaction_name"]}}
                )
            if "duration_min" in filter_params:
                query["query"]["bool"]["must"].append(
                    {
                        "range": {
                            "transaction.duration.us": {
                                "gte": filter_params["duration_min"]
                                * 1000  # Convert to microseconds
                            }
                        }
                    }
                )
        elif self.config.service_name:
            query["query"]["bool"]["must"].append(
                {"term": {"service.name": self.config.service_name}}
            )

        data = query

        return self._make_request(
            method="POST", endpoint="apm-*-transaction*/_search", data=data
        )

    def get_errors(
        self,
        filter_params: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Get error events from Elastic APM.

        Args:
            filter_params: Parameters to filter errors
            limit: Maximum number of errors to retrieve
            start_time: Start timestamp in epoch seconds
            end_time: End timestamp in epoch seconds

        Returns:
            Dictionary with error data
        """
        # Default time range if not specified
        if not end_time:
            end_time = int(time.time()) * 1000  # Elastic uses milliseconds
        if not start_time:
            start_time = end_time - 3600000  # Last hour

        # Build query
        query = {
            "query": {
                "bool": {
                    "must": [
                        {"range": {"@timestamp": {"gte": start_time, "lte": end_time}}},
                        {"term": {"processor.event": "error"}},
                    ]
                }
            },
            "size": limit,
            "sort": [{"@timestamp": {"order": "desc"}}],
        }

        # Add filters
        if filter_params:
            if "service" in filter_params:
                query["query"]["bool"]["must"].append(
                    {"term": {"service.name": filter_params["service"]}}
                )
            if "error_type" in filter_params:
                query["query"]["bool"]["must"].append(
                    {"term": {"error.type": filter_params["error_type"]}}
                )
            if "error_message" in filter_params:
                query["query"]["bool"]["must"].append(
                    {"match_phrase": {"error.message": filter_params["error_message"]}}
                )
        elif self.config.service_name:
            query["query"]["bool"]["must"].append(
                {"term": {"service.name": self.config.service_name}}
            )

        data = query

        return self._make_request(
            method="POST", endpoint="apm-*-error*/_search", data=data
        )

    def test_connection(self) -> Dict[str, Any]:
        """
        Test the connection to Elastic APM.

        Returns:
            Dictionary with test results
        """
        try:
            start_time = time.time()
            response = self._make_request(method="GET", endpoint="_cluster/health")
            elapsed = time.time() - start_time

            return {
                "success": "status" in response,
                "latency_ms": int(elapsed * 1000),
                "response": response,
            }
        except Exception as e:
            logger.error(f"Elastic APM connection test failed: {str(e)}")
            return {"success": False, "error": str(e)}


class APMFactory:
    """Factory for creating APM integrations."""

    @staticmethod
    def create_integration(config: APMConfig) -> APMIntegration:
        """
        Create an APM integration based on the provider.

        Args:
            config: APM configuration

        Returns:
            APM integration instance

        Raises:
            ValueError: If the provider is not supported
        """
        if config.provider == APMProvider.DATADOG:
            return DatadogIntegration(config)
        elif config.provider == APMProvider.NEW_RELIC:
            return NewRelicIntegration(config)
        elif config.provider == APMProvider.DYNATRACE:
            return DynatraceIntegration(config)
        elif config.provider == APMProvider.ELASTIC_APM:
            return ElasticAPMIntegration(config)
        else:
            raise ValueError(f"Unsupported APM provider: {config.provider}")


class APMDataCollector:
    """
    Collector for retrieving and processing data from APM tools.
    """

    def __init__(self, config: APMConfig):
        """
        Initialize the APM data collector.

        Args:
            config: APM configuration
        """
        self.integration = APMFactory.create_integration(config)

    def collect_error_data(
        self,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        service_name: Optional[str] = None,
        error_type: Optional[str] = None,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """
        Collect error data from the APM tool.

        Args:
            start_time: Start timestamp in epoch seconds
            end_time: End timestamp in epoch seconds
            service_name: Service name to filter by
            error_type: Error type to filter by
            limit: Maximum number of errors to retrieve

        Returns:
            Dictionary with error data
        """
        filter_params = {}

        if service_name:
            filter_params["service"] = service_name

        if error_type:
            filter_params["error_type"] = error_type

        return self.integration.get_errors(
            filter_params=filter_params,
            limit=limit,
            start_time=start_time,
            end_time=end_time,
        )

    def collect_performance_metrics(
        self,
        metric_names: List[str],
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        interval: str = "1m",
    ) -> Dict[str, Any]:
        """
        Collect performance metrics from the APM tool.

        Args:
            metric_names: List of metric names to retrieve
            start_time: Start timestamp in epoch seconds
            end_time: End timestamp in epoch seconds
            interval: Time interval for aggregation

        Returns:
            Dictionary with metric data
        """
        return self.integration.get_metrics(
            metric_names=metric_names,
            start_time=start_time,
            end_time=end_time,
            interval=interval,
        )

    def convert_to_standard_format(
        self, apm_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Convert APM-specific error data to a standard format for analysis.

        Args:
            apm_data: APM-specific error data

        Returns:
            List of standardized error data dictionaries
        """
        provider = self.integration.config.provider
        standardized_errors = []

        try:
            if provider == APMProvider.DATADOG:
                self._convert_datadog_format(apm_data, standardized_errors)
            elif provider == APMProvider.NEW_RELIC:
                self._convert_newrelic_format(apm_data, standardized_errors)
            elif provider == APMProvider.DYNATRACE:
                self._convert_dynatrace_format(apm_data, standardized_errors)
            elif provider == APMProvider.ELASTIC_APM:
                self._convert_elastic_format(apm_data, standardized_errors)
            else:
                # Generic conversion - try common patterns
                self._convert_generic_format(apm_data, standardized_errors)
        except Exception as e:
            logger.error(f"Error converting APM data to standard format: {str(e)}")

        return standardized_errors

    def _convert_datadog_format(
        self, apm_data: Dict[str, Any], result: List[Dict[str, Any]]
    ) -> None:
        """
        Convert Datadog error data to standard format.

        Args:
            apm_data: Datadog error data
            result: List to append standardized errors to
        """
        if "events" in apm_data:
            for event in apm_data["events"]:
                error = {
                    "timestamp": event.get("date_happened", ""),
                    "service": event.get("service", ""),
                    "level": "ERROR",
                    "message": event.get("text", ""),
                    "exception_type": event.get("attributes", {}).get("error_type", ""),
                    "traceback": event.get("attributes", {}).get("traceback", []),
                    "error_details": {
                        "exception_type": event.get("attributes", {}).get(
                            "error_type", ""
                        ),
                        "message": event.get("text", ""),
                        "detailed_frames": [],
                    },
                    "_source": "datadog",
                }

                result.append(error)

    def _convert_newrelic_format(
        self, apm_data: Dict[str, Any], result: List[Dict[str, Any]]
    ) -> None:
        """
        Convert New Relic error data to standard format.

        Args:
            apm_data: New Relic error data
            result: List to append standardized errors to
        """
        if "results" in apm_data and apm_data["results"]:
            for item in apm_data["results"][0].get("events", []):
                error = {
                    "timestamp": item.get("timestamp", ""),
                    "service": item.get("serviceName", ""),
                    "level": "ERROR",
                    "message": item.get("errorMessage", ""),
                    "exception_type": item.get("errorType", ""),
                    "traceback": item.get("stackTrace", []),
                    "error_details": {
                        "exception_type": item.get("errorType", ""),
                        "message": item.get("errorMessage", ""),
                        "detailed_frames": [],
                    },
                    "_source": "newrelic",
                }

                result.append(error)

    def _convert_dynatrace_format(
        self, apm_data: Dict[str, Any], result: List[Dict[str, Any]]
    ) -> None:
        """
        Convert Dynatrace error data to standard format.

        Args:
            apm_data: Dynatrace error data
            result: List to append standardized errors to
        """
        if "events" in apm_data:
            for event in apm_data["events"]:
                error = {
                    "timestamp": event.get("startTime", ""),
                    "service": event.get("entityName", ""),
                    "level": "ERROR",
                    "message": event.get("title", ""),
                    "exception_type": event.get("properties", {}).get("errorType", ""),
                    "traceback": event.get("properties", {}).get("stackTrace", []),
                    "error_details": {
                        "exception_type": event.get("properties", {}).get(
                            "errorType", ""
                        ),
                        "message": event.get("title", ""),
                        "detailed_frames": [],
                    },
                    "_source": "dynatrace",
                }

                result.append(error)

    def _convert_elastic_format(
        self, apm_data: Dict[str, Any], result: List[Dict[str, Any]]
    ) -> None:
        """
        Convert Elastic APM error data to standard format.

        Args:
            apm_data: Elastic APM error data
            result: List to append standardized errors to
        """
        if "hits" in apm_data and "hits" in apm_data["hits"]:
            for hit in apm_data["hits"]["hits"]:
                source = hit.get("_source", {})
                error_info = source.get("error", {})
                service = source.get("service", {})

                # Extract stack trace
                stack_trace = []
                if (
                    "exception" in error_info
                    and "stacktrace" in error_info["exception"]
                ):
                    stack_frames = error_info["exception"]["stacktrace"]
                    for frame in stack_frames:
                        frame_str = f"  File \"{frame.get('filename', '')}\", line {frame.get('lineno', '')}, in {frame.get('function', '')}"
                        stack_trace.append(frame_str)

                # Extract detailed frames
                detailed_frames = []
                if (
                    "exception" in error_info
                    and "stacktrace" in error_info["exception"]
                ):
                    stack_frames = error_info["exception"]["stacktrace"]
                    for frame in stack_frames:
                        detailed_frames.append(
                            {
                                "file": frame.get("filename", ""),
                                "line": frame.get("lineno", ""),
                                "function": frame.get("function", ""),
                                "locals": frame.get("vars", {}),
                            }
                        )

                error = {
                    "timestamp": source.get("@timestamp", ""),
                    "service": service.get("name", ""),
                    "level": "ERROR",
                    "message": error_info.get("message", ""),
                    "exception_type": error_info.get("type", ""),
                    "traceback": stack_trace,
                    "error_details": {
                        "exception_type": error_info.get("type", ""),
                        "message": error_info.get("message", ""),
                        "detailed_frames": detailed_frames,
                    },
                    "_source": "elastic_apm",
                }

                result.append(error)

    def _convert_generic_format(
        self, apm_data: Dict[str, Any], result: List[Dict[str, Any]]
    ) -> None:
        """
        Convert generic error data to standard format.

        Args:
            apm_data: Generic error data
            result: List to append standardized errors to
        """
        # Try some common patterns
        error_items = None

        if "errors" in apm_data:
            error_items = apm_data["errors"]
        elif "events" in apm_data:
            error_items = apm_data["events"]
        elif "hits" in apm_data and "hits" in apm_data["hits"]:
            error_items = apm_data["hits"]["hits"]
        elif "results" in apm_data and apm_data["results"]:
            if "events" in apm_data["results"][0]:
                error_items = apm_data["results"][0]["events"]
            else:
                error_items = apm_data["results"]

        if not error_items:
            return

        for item in error_items:
            if isinstance(item, dict):
                # Extract meaningful fields
                timestamp = (
                    item.get("timestamp")
                    or item.get("date")
                    or item.get("time")
                    or item.get("startTime")
                    or item.get("eventTime")
                    or ""
                )

                service = (
                    item.get("service")
                    or item.get("serviceName")
                    or item.get("application")
                    or item.get("appName")
                    or item.get("entityName")
                    or ""
                )

                message = (
                    item.get("message")
                    or item.get("errorMessage")
                    or item.get("text")
                    or item.get("title")
                    or item.get("description")
                    or ""
                )

                exception_type = (
                    item.get("exceptionType")
                    or item.get("errorType")
                    or item.get("type")
                    or item.get("exception")
                    or ""
                )

                error = {
                    "timestamp": timestamp,
                    "service": service,
                    "level": "ERROR",
                    "message": message,
                    "exception_type": exception_type,
                    "traceback": item.get("stackTrace", []),
                    "error_details": {
                        "exception_type": exception_type,
                        "message": message,
                        "detailed_frames": [],
                    },
                    "_source": "apm",
                }

                result.append(error)


def create_apm_integration(
    provider_name: str,
    api_key: str,
    api_endpoint: Optional[str] = None,
    service_name: Optional[str] = None,
    environment: Optional[str] = None,
    additional_params: Optional[Dict[str, Any]] = None,
) -> APMIntegration:
    """
    Utility function to create an APM integration.

    Args:
        provider_name: Name of the APM provider
        api_key: API key for authentication
        api_endpoint: API endpoint URL
        service_name: Service name
        environment: Environment (prod, staging, etc.)
        additional_params: Additional parameters

    Returns:
        APM integration instance

    Raises:
        ValueError: If the provider is not supported
    """
    # Map provider name to enum
    provider_map = {
        "datadog": APMProvider.DATADOG,
        "new_relic": APMProvider.NEW_RELIC,
        "dynatrace": APMProvider.DYNATRACE,
        "elastic_apm": APMProvider.ELASTIC_APM,
        "prometheus": APMProvider.PROMETHEUS,
        "grafana": APMProvider.GRAFANA,
        "app_dynamics": APMProvider.APP_DYNAMICS,
        "instana": APMProvider.INSTANA,
        "pinpoint": APMProvider.PINPOINT,
        "skywalking": APMProvider.SKYWALKING,
        "custom": APMProvider.CUSTOM,
    }

    if provider_name.lower() not in provider_map:
        raise ValueError(f"Unsupported APM provider: {provider_name}")

    provider = provider_map[provider_name.lower()]

    # Create configuration
    config = APMConfig(
        provider=provider,
        api_key=api_key,
        api_endpoint=api_endpoint,
        service_name=service_name,
        environment=environment,
        additional_params=additional_params or {},
    )

    # Create integration
    return APMFactory.create_integration(config)


def collect_apm_errors(
    provider_name: str,
    api_key: str,
    api_endpoint: Optional[str] = None,
    service_name: Optional[str] = None,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """
    Utility function to collect errors from an APM tool.

    Args:
        provider_name: Name of the APM provider
        api_key: API key for authentication
        api_endpoint: API endpoint URL
        service_name: Service name
        start_time: Start timestamp in epoch seconds
        end_time: End timestamp in epoch seconds
        limit: Maximum number of errors to retrieve

    Returns:
        List of standardized error data dictionaries
    """
    # Create integration
    integration = create_apm_integration(
        provider_name=provider_name,
        api_key=api_key,
        api_endpoint=api_endpoint,
        service_name=service_name,
    )

    # Create collector
    collector = APMDataCollector(integration.config)

    # Collect error data
    error_data = collector.collect_error_data(
        start_time=start_time, end_time=end_time, service_name=service_name, limit=limit
    )

    # Convert to standard format
    return collector.convert_to_standard_format(error_data)


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Example usage
    print("APM Integration Demo")
    print("===================")

    # Test connection to DataDog (with fake API key)
    test_config = APMConfig(
        provider=APMProvider.DATADOG,
        api_key="fake_api_key",
        api_endpoint="https://api.datadoghq.com",
        service_name="example_service",
        environment="development",
    )

    # Test all supported APM providers
    test_providers = [
        APMProvider.DATADOG,
        APMProvider.NEW_RELIC,
        APMProvider.DYNATRACE,
        APMProvider.ELASTIC_APM,
    ]

    for provider in test_providers:
        print(f"\nTesting {provider.value} integration:")

        test_config.provider = provider
        test_config.api_key = f"fake_api_key_{provider.value}"

        if provider == APMProvider.DATADOG:
            test_config.api_endpoint = "https://api.datadoghq.com"
        elif provider == APMProvider.NEW_RELIC:
            test_config.api_endpoint = "https://api.newrelic.com"
        elif provider == APMProvider.DYNATRACE:
            test_config.api_endpoint = "https://example.live.dynatrace.com"
        elif provider == APMProvider.ELASTIC_APM:
            test_config.api_endpoint = "https://example.apm.us-central1.gcp.cloud.es.io"

        try:
            integration = APMFactory.create_integration(test_config)
            print(f"Created {provider.value} integration")
            print("(Note: Connection test will fail with fake API key)")
        except Exception as e:
            print(f"Error creating integration: {str(e)}")

    # Example of error data conversion
    print("\nExample of converting APM data to standard format:")

    # Create sample Datadog error data
    sample_datadog_data = {
        "events": [
            {
                "date_happened": "2023-01-01T12:00:00",
                "service": "example_service",
                "text": "KeyError: 'todo_id'",
                "attributes": {
                    "error_type": "KeyError",
                    "traceback": [
                        "Traceback (most recent call last):",
                        "  File '/app/services/example_service/app.py', line 42, in get_todo",
                        "    todo = todo_db[todo_id]",
                        "KeyError: 'todo_id'",
                    ],
                },
            }
        ]
    }

    test_config.provider = APMProvider.DATADOG
    collector = APMDataCollector(test_config)

    standard_errors = collector.convert_to_standard_format(sample_datadog_data)

    print(f"Converted {len(standard_errors)} errors to standard format:")
    for error in standard_errors:
        print(f"- {error['exception_type']}: {error['message']}")
        print(f"  Service: {error['service']}")
        print(f"  Timestamp: {error['timestamp']}")
        print(f"  Source: {error['_source']}")
