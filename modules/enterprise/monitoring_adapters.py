"""
Enterprise Monitoring System Adapters

Provides integration with popular enterprise monitoring tools to collect metrics,
events, and alerts that can be used by the Homeostasis healing system.
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics collected"""

    GAUGE = "gauge"
    COUNTER = "counter"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    RATE = "rate"


class AlertSeverity(Enum):
    """Alert severity levels"""

    CRITICAL = "critical"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    DEBUG = "debug"


class AlertStatus(Enum):
    """Alert status"""

    FIRING = "firing"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"
    SILENCED = "silenced"


@dataclass
class Metric:
    """Represents a metric data point"""

    name: str
    value: float
    timestamp: datetime
    metric_type: MetricType = MetricType.GAUGE
    tags: Dict[str, str] = field(default_factory=dict)
    unit: Optional[str] = None
    description: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "type": self.metric_type.value,
            "tags": self.tags,
            "unit": self.unit,
            "description": self.description,
        }


@dataclass
class Alert:
    """Represents a monitoring alert"""

    alert_id: str
    name: str
    severity: AlertSeverity
    status: AlertStatus
    message: str
    source: str
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    runbook_url: Optional[str] = None
    dashboard_url: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "alert_id": self.alert_id,
            "name": self.name,
            "severity": self.severity.value,
            "status": self.status.value,
            "message": self.message,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags,
            "metadata": self.metadata,
            "runbook_url": self.runbook_url,
            "dashboard_url": self.dashboard_url,
        }


@dataclass
class Event:
    """Represents a monitoring event"""

    event_id: str
    title: str
    text: str
    timestamp: datetime
    event_type: str = "info"
    tags: Dict[str, str] = field(default_factory=dict)
    source: Optional[str] = None
    priority: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "event_id": self.event_id,
            "title": self.title,
            "text": self.text,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "tags": self.tags,
            "source": self.source,
            "priority": self.priority,
        }


class MonitoringAdapter(ABC):
    """Abstract base class for monitoring system adapters"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.poll_interval = config.get("poll_interval", 60)  # seconds
        self.batch_size = config.get("batch_size", 100)
        self.metric_prefix = config.get("metric_prefix", "homeostasis")
        self._callbacks = {"metric": [], "alert": [], "event": []}

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to the monitoring system"""
        pass

    @abstractmethod
    def get_metrics(
        self, query: Dict[str, Any], start_time: datetime, end_time: datetime
    ) -> List[Metric]:
        """Query metrics from the monitoring system"""
        pass

    @abstractmethod
    def get_alerts(self, filters: Dict[str, Any]) -> List[Alert]:
        """Get current alerts from the monitoring system"""
        pass

    @abstractmethod
    def get_events(
        self, filters: Dict[str, Any], start_time: datetime, end_time: datetime
    ) -> List[Event]:
        """Get events from the monitoring system"""
        pass

    @abstractmethod
    def send_metric(self, metric: Metric) -> bool:
        """Send a metric to the monitoring system"""
        pass

    @abstractmethod
    def send_event(self, event: Event) -> bool:
        """Send an event to the monitoring system"""
        pass

    @abstractmethod
    def acknowledge_alert(self, alert_id: str, message: str = "") -> bool:
        """Acknowledge an alert in the monitoring system"""
        pass

    def register_callback(self, data_type: str, callback: Callable):
        """Register a callback for metric/alert/event processing"""
        if data_type in self._callbacks:
            self._callbacks[data_type].append(callback)

    def start_polling(self):
        """Start polling for metrics and alerts"""
        # This would typically run in a separate thread/process
        pass

    def stop_polling(self):
        """Stop polling"""
        pass

    def _process_data(self, data_type: str, data: Any):
        """Process data through registered callbacks"""
        for callback in self._callbacks.get(data_type, []):
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Error in {data_type} callback: {e}")


class DatadogAdapter(MonitoringAdapter):
    """Datadog monitoring adapter"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key", "")
        self.app_key = config.get("app_key", "")
        self.base_url = config.get("base_url", "https://api.datadoghq.com")
        self.site = config.get("site", "datadoghq.com")
        self._session = None

    def connect(self) -> bool:
        """Connect to Datadog API"""
        try:
            self._session = requests.Session()
            self._session.headers.update(
                {
                    "DD-API-KEY": self.api_key,
                    "DD-APPLICATION-KEY": self.app_key,
                    "Content-Type": "application/json",
                }
            )

            # Test connection
            test_url = f"{self.base_url}/api/v1/validate"
            response = self._session.get(test_url)
            response.raise_for_status()

            logger.info("Successfully connected to Datadog")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to Datadog: {e}")
            return False

    def get_metrics(
        self, query: Dict[str, Any], start_time: datetime, end_time: datetime
    ) -> List[Metric]:
        """Query metrics from Datadog"""
        try:
            url = f"{self.base_url}/api/v1/query"

            # Build query string
            metric_query = query.get("query", "")
            if not metric_query and query.get("metric_name"):
                # Build simple query
                metric_name = query["metric_name"]
                tags = query.get("tags", {})
                tag_filters = []
                for k, v in tags.items():
                    tag_filters.append(f"{k}:{v}")

                if tag_filters:
                    metric_query = f"{metric_name}{{{','.join(tag_filters)}}}"
                else:
                    metric_query = metric_name

            params = {
                "query": metric_query,
                "from": int(start_time.timestamp()),
                "to": int(end_time.timestamp()),
            }

            response = self._session.get(url, params=params)
            response.raise_for_status()

            data = response.json()
            metrics = []

            for series in data.get("series", []):
                metric_name = series.get("metric", "")
                points = series.get("points", [])
                scope = series.get("scope", "")

                # Parse tags from scope
                tags = self._parse_datadog_scope(scope)

                for timestamp, value in points:
                    if value is not None:
                        metric = Metric(
                            name=metric_name,
                            value=float(value),
                            timestamp=datetime.fromtimestamp(timestamp / 1000),
                            tags=tags,
                        )
                        metrics.append(metric)

            return metrics

        except Exception as e:
            logger.error(f"Failed to get metrics from Datadog: {e}")
            return []

    def get_alerts(self, filters: Dict[str, Any]) -> List[Alert]:
        """Get alerts from Datadog monitors"""
        try:
            url = f"{self.base_url}/api/v1/monitor"

            params = {}
            if filters.get("group_states"):
                params["group_states"] = ",".join(filters["group_states"])
            if filters.get("tags"):
                params["tags"] = ",".join(filters["tags"])
            if filters.get("monitor_ids"):
                params["monitor_ids"] = ",".join(map(str, filters["monitor_ids"]))

            response = self._session.get(url, params=params)
            response.raise_for_status()

            monitors = response.json()
            alerts = []

            for monitor in monitors:
                # Check monitor state
                state = monitor.get("overall_state", "OK")
                if state in ["Alert", "Warn", "No Data"]:
                    alert = Alert(
                        alert_id=str(monitor.get("id")),
                        name=monitor.get("name", ""),
                        severity=self._map_datadog_severity(state),
                        status=AlertStatus.FIRING,
                        message=monitor.get("message", ""),
                        source="datadog",
                        timestamp=datetime.fromisoformat(
                            monitor.get("modified").replace("Z", "+00:00")
                        ),
                        tags=monitor.get("tags", []),
                        metadata={
                            "monitor_type": monitor.get("type"),
                            "query": monitor.get("query"),
                            "thresholds": monitor.get("options", {}).get(
                                "thresholds", {}
                            ),
                        },
                    )
                    alerts.append(alert)

            return alerts

        except Exception as e:
            logger.error(f"Failed to get alerts from Datadog: {e}")
            return []

    def get_events(
        self, filters: Dict[str, Any], start_time: datetime, end_time: datetime
    ) -> List[Event]:
        """Get events from Datadog"""
        try:
            url = f"{self.base_url}/api/v1/events"

            params = {
                "start": int(start_time.timestamp()),
                "end": int(end_time.timestamp()),
            }

            if filters.get("tags"):
                params["tags"] = ",".join(filters["tags"])
            if filters.get("sources"):
                params["sources"] = ",".join(filters["sources"])
            if filters.get("priority"):
                params["priority"] = filters["priority"]

            response = self._session.get(url, params=params)
            response.raise_for_status()

            data = response.json()
            events = []

            for event_data in data.get("events", []):
                event = Event(
                    event_id=str(event_data.get("id")),
                    title=event_data.get("title", ""),
                    text=event_data.get("text", ""),
                    timestamp=datetime.fromtimestamp(
                        event_data.get("date_happened", 0)
                    ),
                    event_type=event_data.get("alert_type", "info"),
                    tags={
                        tag.split(":")[0]: tag.split(":")[1]
                        for tag in event_data.get("tags", [])
                        if ":" in tag
                    },
                    source=event_data.get("source", ""),
                    priority=event_data.get("priority"),
                )
                events.append(event)

            return events

        except Exception as e:
            logger.error(f"Failed to get events from Datadog: {e}")
            return []

    def send_metric(self, metric: Metric) -> bool:
        """Send metric to Datadog"""
        try:
            url = f"{self.base_url}/api/v1/series"

            # Format tags
            tags = []
            for k, v in metric.tags.items():
                tags.append(f"{k}:{v}")

            data = {
                "series": [
                    {
                        "metric": f"{self.metric_prefix}.{metric.name}",
                        "points": [[int(metric.timestamp.timestamp()), metric.value]],
                        "type": metric.metric_type.value,
                        "tags": tags,
                    }
                ]
            }

            response = self._session.post(url, json=data)
            response.raise_for_status()

            return True

        except Exception as e:
            logger.error(f"Failed to send metric to Datadog: {e}")
            return False

    def send_event(self, event: Event) -> bool:
        """Send event to Datadog"""
        try:
            url = f"{self.base_url}/api/v1/events"

            # Format tags
            tags = []
            for k, v in event.tags.items():
                tags.append(f"{k}:{v}")

            data = {
                "title": event.title,
                "text": event.text,
                "date_happened": int(event.timestamp.timestamp()),
                "priority": event.priority or "normal",
                "tags": tags,
                "alert_type": event.event_type,
                "source_type_name": event.source or "homeostasis",
            }

            response = self._session.post(url, json=data)
            response.raise_for_status()

            return True

        except Exception as e:
            logger.error(f"Failed to send event to Datadog: {e}")
            return False

    def acknowledge_alert(self, alert_id: str, message: str = "") -> bool:
        """Mute a Datadog monitor (equivalent to acknowledge)"""
        try:
            url = f"{self.base_url}/api/v1/monitor/{alert_id}/mute"

            data = {}
            if message:
                data["message"] = message

            response = self._session.post(url, json=data)
            response.raise_for_status()

            return True

        except Exception as e:
            logger.error(f"Failed to acknowledge Datadog alert: {e}")
            return False

    def _parse_datadog_scope(self, scope: str) -> Dict[str, str]:
        """Parse Datadog scope string into tags"""
        tags = {}
        if scope:
            parts = scope.split(",")
            for part in parts:
                if ":" in part:
                    key, value = part.split(":", 1)
                    tags[key] = value
        return tags

    def _map_datadog_severity(self, state: str) -> AlertSeverity:
        """Map Datadog monitor state to severity"""
        mapping = {
            "Alert": AlertSeverity.CRITICAL,
            "Warn": AlertSeverity.WARNING,
            "No Data": AlertSeverity.WARNING,
            "OK": AlertSeverity.INFO,
        }
        return mapping.get(state, AlertSeverity.INFO)


class PrometheusAdapter(MonitoringAdapter):
    """Prometheus monitoring adapter"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get("base_url", "http://localhost:9090")
        self.username = config.get("username", "")
        self.password = config.get("password", "")
        self.bearer_token = config.get("bearer_token", "")
        self._session = None

    def connect(self) -> bool:
        """Connect to Prometheus API"""
        try:
            self._session = requests.Session()

            # Set authentication
            if self.bearer_token:
                self._session.headers.update(
                    {"Authorization": f"Bearer {self.bearer_token}"}
                )
            elif self.username and self.password:
                self._session.auth = (self.username, self.password)

            # Test connection
            test_url = f"{self.base_url}/api/v1/query"
            params = {"query": "up"}
            response = self._session.get(test_url, params=params)
            response.raise_for_status()

            logger.info("Successfully connected to Prometheus")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to Prometheus: {e}")
            return False

    def get_metrics(
        self, query: Dict[str, Any], start_time: datetime, end_time: datetime
    ) -> List[Metric]:
        """Query metrics from Prometheus"""
        try:
            url = f"{self.base_url}/api/v1/query_range"

            # Build PromQL query
            promql = query.get("query", "")
            if not promql and query.get("metric_name"):
                # Build simple query
                metric_name = query["metric_name"]
                label_filters = []
                for k, v in query.get("labels", {}).items():
                    label_filters.append(f'{k}="{v}"')

                if label_filters:
                    promql = f"{metric_name}{{{','.join(label_filters)}}}"
                else:
                    promql = metric_name

            # Calculate step based on time range
            duration = (end_time - start_time).total_seconds()
            step = max(15, int(duration / 1000))  # Max 1000 data points

            params = {
                "query": promql,
                "start": int(start_time.timestamp()),
                "end": int(end_time.timestamp()),
                "step": f"{step}s",
            }

            response = self._session.get(url, params=params)
            response.raise_for_status()

            data = response.json()
            if data["status"] != "success":
                logger.error(
                    f"Prometheus query error: {data.get('error', 'Unknown error')}"
                )
                return []

            metrics = []
            for result in data["data"]["result"]:
                metric_info = result["metric"]
                metric_name = metric_info.get("__name__", "")

                # Extract labels as tags
                tags = {k: v for k, v in metric_info.items() if k != "__name__"}

                for timestamp, value in result["values"]:
                    metric = Metric(
                        name=metric_name,
                        value=float(value),
                        timestamp=datetime.fromtimestamp(timestamp),
                        tags=tags,
                    )
                    metrics.append(metric)

            return metrics

        except Exception as e:
            logger.error(f"Failed to get metrics from Prometheus: {e}")
            return []

    def get_alerts(self, filters: Dict[str, Any]) -> List[Alert]:
        """Get alerts from Prometheus AlertManager"""
        try:
            # Prometheus alerts are in AlertManager, not Prometheus itself
            alertmanager_url = self.config.get(
                "alertmanager_url", "http://localhost:9093"
            )
            url = f"{alertmanager_url}/api/v1/alerts"

            params = {}
            if filters.get("active"):
                params["active"] = "true"
            if filters.get("silenced"):
                params["silenced"] = filters["silenced"]
            if filters.get("inhibited"):
                params["inhibited"] = filters["inhibited"]

            response = self._session.get(url, params=params)
            response.raise_for_status()

            data = response.json()
            alerts = []

            for alert_data in data["data"]:
                # Map labels to tags
                tags = alert_data.get("labels", {})

                alert = Alert(
                    alert_id=alert_data.get("fingerprint", ""),
                    name=tags.get("alertname", ""),
                    severity=self._map_prometheus_severity(
                        tags.get("severity", "warning")
                    ),
                    status=self._map_prometheus_status(alert_data["status"]["state"]),
                    message=alert_data.get("annotations", {}).get("summary", ""),
                    source="prometheus",
                    timestamp=datetime.fromisoformat(
                        alert_data["startsAt"].replace("Z", "+00:00")
                    ),
                    tags=tags,
                    metadata={
                        "annotations": alert_data.get("annotations", {}),
                        "generator_url": alert_data.get("generatorURL", ""),
                    },
                    runbook_url=alert_data.get("annotations", {}).get("runbook_url"),
                    dashboard_url=alert_data.get("annotations", {}).get(
                        "dashboard_url"
                    ),
                )
                alerts.append(alert)

            return alerts

        except Exception as e:
            logger.error(f"Failed to get alerts from Prometheus: {e}")
            return []

    def get_events(
        self, filters: Dict[str, Any], start_time: datetime, end_time: datetime
    ) -> List[Event]:
        """Prometheus doesn't have native events - return empty list"""
        # Events could be extracted from specific metrics if configured
        return []

    def send_metric(self, metric: Metric) -> bool:
        """Send metric to Prometheus Pushgateway"""
        try:
            pushgateway_url = self.config.get(
                "pushgateway_url", "http://localhost:9091"
            )

            # Format metric for Pushgateway
            metric_name = f"{self.metric_prefix}_{metric.name}"
            labels = []
            for k, v in metric.tags.items():
                labels.append(f'{k}="{v}"')

            if labels:
                metric_line = f"{metric_name}{{{','.join(labels)}}} {metric.value}"
            else:
                metric_line = f"{metric_name} {metric.value}"

            # Push to gateway
            job_name = self.config.get("job_name", "homeostasis")
            url = f"{pushgateway_url}/metrics/job/{job_name}"

            response = requests.post(
                url, data=metric_line, headers={"Content-Type": "text/plain"}, timeout=30
            )
            response.raise_for_status()

            return True

        except Exception as e:
            logger.error(f"Failed to send metric to Prometheus: {e}")
            return False

    def send_event(self, event: Event) -> bool:
        """Prometheus doesn't support events - log and return"""
        logger.info(f"Prometheus doesn't support events. Event logged: {event.title}")
        return True

    def acknowledge_alert(self, alert_id: str, message: str = "") -> bool:
        """Silence alert in AlertManager"""
        try:
            alertmanager_url = self.config.get(
                "alertmanager_url", "http://localhost:9093"
            )
            url = f"{alertmanager_url}/api/v1/silences"

            # Create silence for the alert
            data = {
                "matchers": [
                    {"name": "fingerprint", "value": alert_id, "isRegex": False}
                ],
                "startsAt": datetime.utcnow().isoformat() + "Z",
                "endsAt": (datetime.utcnow() + timedelta(hours=4)).isoformat() + "Z",
                "createdBy": "homeostasis",
                "comment": message or "Acknowledged by Homeostasis",
            }

            response = self._session.post(url, json=data)
            response.raise_for_status()

            return True

        except Exception as e:
            logger.error(f"Failed to acknowledge Prometheus alert: {e}")
            return False

    def _map_prometheus_severity(self, severity: str) -> AlertSeverity:
        """Map Prometheus severity label to AlertSeverity"""
        mapping = {
            "critical": AlertSeverity.CRITICAL,
            "error": AlertSeverity.ERROR,
            "warning": AlertSeverity.WARNING,
            "info": AlertSeverity.INFO,
        }
        return mapping.get(severity.lower(), AlertSeverity.WARNING)

    def _map_prometheus_status(self, state: str) -> AlertStatus:
        """Map Prometheus alert state to AlertStatus"""
        mapping = {
            "active": AlertStatus.FIRING,
            "suppressed": AlertStatus.SILENCED,
            "unprocessed": AlertStatus.FIRING,
        }
        return mapping.get(state.lower(), AlertStatus.FIRING)


class SplunkAdapter(MonitoringAdapter):
    """Splunk monitoring adapter"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get("base_url", "https://localhost:8089")
        self.username = config.get("username", "")
        self.password = config.get("password", "")
        self.token = config.get("token", "")
        self._session = None
        self._session_key = None

    def connect(self) -> bool:
        """Connect to Splunk API"""
        try:
            self._session = requests.Session()
            self._session.verify = self.config.get("verify_ssl", True)

            if self.token:
                # Use token authentication
                self._session.headers.update({"Authorization": f"Splunk {self.token}"})
                self._session_key = self.token
            else:
                # Use username/password authentication
                auth_url = f"{self.base_url}/services/auth/login"
                data = {"username": self.username, "password": self.password}

                response = self._session.post(auth_url, data=data)
                response.raise_for_status()

                # Extract session key from response
                try:
                    import defusedxml.ElementTree as ET
                except ImportError:
                    import xml.etree.ElementTree as ET

                root = ET.fromstring(response.text)
                self._session_key = root.find(".//sessionKey").text

                self._session.headers.update(
                    {"Authorization": f"Splunk {self._session_key}"}
                )

            # Test connection
            test_url = f"{self.base_url}/services/server/info"
            response = self._session.get(test_url)
            response.raise_for_status()

            logger.info("Successfully connected to Splunk")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to Splunk: {e}")
            return False

    def get_metrics(
        self, query: Dict[str, Any], start_time: datetime, end_time: datetime
    ) -> List[Metric]:
        """Query metrics from Splunk"""
        try:
            # Use Splunk search to get metrics
            search_query = query.get("search", "")
            if not search_query:
                # Build basic metrics search
                metric_name = query.get("metric_name", "*")
                index = query.get("index", "metrics")
                search_query = f"search index={index} metric_name={metric_name}"

            # Add time range
            search_query += f" earliest={int(start_time.timestamp())} latest={int(end_time.timestamp())}"

            # Create search job
            url = f"{self.base_url}/services/search/jobs"
            data = {"search": search_query, "output_mode": "json"}

            response = self._session.post(url, data=data)
            response.raise_for_status()

            # Get job ID
            job_data = response.json()
            job_id = job_data["sid"]

            # Wait for job to complete
            self._wait_for_job(job_id)

            # Get results
            results_url = f"{self.base_url}/services/search/jobs/{job_id}/results"
            params = {"output_mode": "json"}

            response = self._session.get(results_url, params=params)
            response.raise_for_status()

            results = response.json()
            metrics = []

            for result in results.get("results", []):
                metric = Metric(
                    name=result.get("metric_name", ""),
                    value=float(result.get("_value", 0)),
                    timestamp=datetime.fromtimestamp(float(result.get("_time", 0))),
                    tags=self._extract_splunk_dimensions(result),
                )
                metrics.append(metric)

            return metrics

        except Exception as e:
            logger.error(f"Failed to get metrics from Splunk: {e}")
            return []

    def get_alerts(self, filters: Dict[str, Any]) -> List[Alert]:
        """Get alerts from Splunk"""
        try:
            url = f"{self.base_url}/services/alerts/fired_alerts"
            params = {"output_mode": "json"}

            response = self._session.get(url, params=params)
            response.raise_for_status()

            data = response.json()
            alerts = []

            for entry in data.get("entry", []):
                content = entry["content"]

                alert = Alert(
                    alert_id=entry["id"],
                    name=entry["name"],
                    severity=self._map_splunk_severity(content.get("severity", "3")),
                    status=AlertStatus.FIRING,
                    message=content.get("description", ""),
                    source="splunk",
                    timestamp=datetime.fromisoformat(
                        entry["updated"].replace("Z", "+00:00")
                    ),
                    metadata={
                        "savedsearch_name": content.get("savedsearch_name"),
                        "trigger_time": content.get("trigger_time"),
                        "result_count": content.get("triggered_alerts_count"),
                    },
                )
                alerts.append(alert)

            return alerts

        except Exception as e:
            logger.error(f"Failed to get alerts from Splunk: {e}")
            return []

    def get_events(
        self, filters: Dict[str, Any], start_time: datetime, end_time: datetime
    ) -> List[Event]:
        """Get events from Splunk"""
        try:
            # Build search query for events
            search_query = filters.get("search", "search *")
            if filters.get("index"):
                search_query = f'search index={filters["index"]} {search_query}'

            # Add time range
            search_query += f" earliest={int(start_time.timestamp())} latest={int(end_time.timestamp())}"

            # Create search job
            url = f"{self.base_url}/services/search/jobs"
            data = {"search": search_query, "output_mode": "json"}

            response = self._session.post(url, data=data)
            response.raise_for_status()

            # Get job ID
            job_data = response.json()
            job_id = job_data["sid"]

            # Wait for job to complete
            self._wait_for_job(job_id)

            # Get results
            results_url = f"{self.base_url}/services/search/jobs/{job_id}/results"
            params = {"output_mode": "json", "count": self.batch_size}

            response = self._session.get(results_url, params=params)
            response.raise_for_status()

            results = response.json()
            events = []

            for result in results.get("results", []):
                event = Event(
                    event_id=result.get("_cd", ""),
                    title=result.get("_raw", "")[:100],  # First 100 chars as title
                    text=result.get("_raw", ""),
                    timestamp=datetime.fromtimestamp(float(result.get("_time", 0))),
                    source=result.get("source", ""),
                    tags={
                        "host": result.get("host", ""),
                        "sourcetype": result.get("sourcetype", ""),
                        "index": result.get("index", ""),
                    },
                )
                events.append(event)

            return events

        except Exception as e:
            logger.error(f"Failed to get events from Splunk: {e}")
            return []

    def send_metric(self, metric: Metric) -> bool:
        """Send metric to Splunk metrics index"""
        try:
            url = f"{self.base_url}/services/collectors/http"

            # Format metric for Splunk
            data = {
                "time": int(metric.timestamp.timestamp()),
                "source": "homeostasis",
                "sourcetype": "metric",
                "index": self.config.get("metrics_index", "metrics"),
                "event": {
                    "metric_name": f"{self.metric_prefix}.{metric.name}",
                    "_value": metric.value,
                    **metric.tags,
                },
            }

            response = self._session.post(url, json=data)
            response.raise_for_status()

            return True

        except Exception as e:
            logger.error(f"Failed to send metric to Splunk: {e}")
            return False

    def send_event(self, event: Event) -> bool:
        """Send event to Splunk"""
        try:
            url = f"{self.base_url}/services/collectors/http"

            data = {
                "time": int(event.timestamp.timestamp()),
                "source": event.source or "homeostasis",
                "sourcetype": "homeostasis:event",
                "index": self.config.get("events_index", "main"),
                "event": {
                    "title": event.title,
                    "text": event.text,
                    "event_type": event.event_type,
                    "priority": event.priority,
                    **event.tags,
                },
            }

            response = self._session.post(url, json=data)
            response.raise_for_status()

            return True

        except Exception as e:
            logger.error(f"Failed to send event to Splunk: {e}")
            return False

    def acknowledge_alert(self, alert_id: str, message: str = "") -> bool:
        """Acknowledge alert in Splunk (mark as reviewed)"""
        # Splunk doesn't have a direct acknowledge mechanism
        # Log the acknowledgment as an event
        ack_event = Event(
            event_id=f"ack_{alert_id}",
            title=f"Alert Acknowledged: {alert_id}",
            text=message or "Acknowledged by Homeostasis",
            timestamp=datetime.utcnow(),
            event_type="alert_acknowledgment",
            tags={"alert_id": alert_id},
        )

        return self.send_event(ack_event)

    def _wait_for_job(self, job_id: str, timeout: int = 60):
        """Wait for Splunk search job to complete"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            url = f"{self.base_url}/services/search/jobs/{job_id}"
            params = {"output_mode": "json"}

            response = self._session.get(url, params=params)
            if response.status_code == 200:
                job_data = response.json()
                state = job_data["entry"][0]["content"]["dispatchState"]

                if state == "DONE":
                    return
                elif state == "FAILED":
                    raise Exception(f"Splunk job failed: {job_id}")

            time.sleep(1)

        raise Exception(f"Splunk job timeout: {job_id}")

    def _extract_splunk_dimensions(self, result: Dict[str, Any]) -> Dict[str, str]:
        """Extract dimensions/tags from Splunk result"""
        dimensions = {}

        # Common dimension fields
        dimension_fields = ["host", "source", "sourcetype", "index"]

        for field_name in dimension_fields:
            if field_name in result:
                dimensions[field_name] = result[field_name]

        # Extract custom dimensions (fields starting with dim_)
        for key, value in result.items():
            if key.startswith("dim_"):
                dimensions[key[4:]] = value

        return dimensions

    def _map_splunk_severity(self, severity: str) -> AlertSeverity:
        """Map Splunk severity to AlertSeverity"""
        mapping = {
            "1": AlertSeverity.INFO,
            "2": AlertSeverity.WARNING,
            "3": AlertSeverity.ERROR,
            "4": AlertSeverity.CRITICAL,
            "5": AlertSeverity.CRITICAL,
        }
        return mapping.get(severity, AlertSeverity.WARNING)


class ElasticAdapter(MonitoringAdapter):
    """Elasticsearch/Elastic Stack monitoring adapter"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get("base_url", "http://localhost:9200")
        self.username = config.get("username", "")
        self.password = config.get("password", "")
        self.api_key = config.get("api_key", "")
        self.cloud_id = config.get("cloud_id", "")
        self._session = None

    def connect(self) -> bool:
        """Connect to Elasticsearch"""
        try:
            self._session = requests.Session()

            # Set authentication
            if self.api_key:
                self._session.headers.update(
                    {"Authorization": f"ApiKey {self.api_key}"}
                )
            elif self.username and self.password:
                self._session.auth = (self.username, self.password)

            # Handle Elastic Cloud
            if self.cloud_id:
                import base64

                decoded = base64.b64decode(self.cloud_id).decode("utf-8")
                cluster_name, hosts = decoded.split(":")
                es_host = hosts.split("$")[0]
                self.base_url = (
                    f"https://{es_host}.{cluster_name}.cloud.elastic.co:9243"
                )

            # Test connection
            response = self._session.get(f"{self.base_url}/_cluster/health")
            response.raise_for_status()

            logger.info("Successfully connected to Elasticsearch")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to Elasticsearch: {e}")
            return False

    def get_metrics(
        self, query: Dict[str, Any], start_time: datetime, end_time: datetime
    ) -> List[Metric]:
        """Query metrics from Elasticsearch"""
        try:
            index = query.get("index", "metricbeat-*")

            # Build query
            es_query = {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "range": {
                                    "@timestamp": {
                                        "gte": start_time.isoformat(),
                                        "lte": end_time.isoformat(),
                                    }
                                }
                            }
                        ]
                    }
                },
                "size": self.batch_size,
                "sort": [{"@timestamp": {"order": "asc"}}],
            }

            # Add metric name filter if specified
            if query.get("metric_name"):
                es_query["query"]["bool"]["must"].append(
                    {"match": {"metricset.name": query["metric_name"]}}
                )

            # Add additional filters
            for field_name, value in query.get("filters", {}).items():
                es_query["query"]["bool"]["must"].append({"match": {field_name: value}})

            url = f"{self.base_url}/{index}/_search"
            response = self._session.post(url, json=es_query)
            response.raise_for_status()

            data = response.json()
            metrics = []

            for hit in data["hits"]["hits"]:
                source = hit["_source"]

                # Extract metric value (location varies by metricset)
                value = self._extract_metric_value(source)
                if value is not None:
                    metric = Metric(
                        name=source.get("metricset", {}).get("name", ""),
                        value=float(value),
                        timestamp=datetime.fromisoformat(
                            source["@timestamp"].replace("Z", "+00:00")
                        ),
                        tags=self._extract_elastic_tags(source),
                    )
                    metrics.append(metric)

            return metrics

        except Exception as e:
            logger.error(f"Failed to get metrics from Elasticsearch: {e}")
            return []

    def get_alerts(self, filters: Dict[str, Any]) -> List[Alert]:
        """Get alerts from Elasticsearch Watcher or detection rules"""
        try:
            # Query watcher history
            url = f"{self.base_url}/.watcher-history-*/_search"

            query = {
                "query": {"bool": {"must": [{"term": {"state": "active"}}]}},
                "size": self.batch_size,
                "sort": [{"trigger_event.triggered_time": {"order": "desc"}}],
            }

            response = self._session.post(url, json=query)
            response.raise_for_status()

            data = response.json()
            alerts = []

            for hit in data["hits"]["hits"]:
                source = hit["_source"]

                alert = Alert(
                    alert_id=hit["_id"],
                    name=source.get("watch_id", ""),
                    severity=self._determine_elastic_severity(source),
                    status=AlertStatus.FIRING,
                    message=source.get("result", {})
                    .get("condition", {})
                    .get("met", ""),
                    source="elasticsearch",
                    timestamp=datetime.fromisoformat(
                        source["trigger_event"]["triggered_time"].replace("Z", "+00:00")
                    ),
                    metadata={
                        "watch_id": source.get("watch_id"),
                        "node": source.get("node"),
                    },
                )
                alerts.append(alert)

            return alerts

        except Exception as e:
            logger.error(f"Failed to get alerts from Elasticsearch: {e}")
            return []

    def get_events(
        self, filters: Dict[str, Any], start_time: datetime, end_time: datetime
    ) -> List[Event]:
        """Get events from Elasticsearch"""
        try:
            index = filters.get("index", "logstash-*")

            query = {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "range": {
                                    "@timestamp": {
                                        "gte": start_time.isoformat(),
                                        "lte": end_time.isoformat(),
                                    }
                                }
                            }
                        ]
                    }
                },
                "size": self.batch_size,
                "sort": [{"@timestamp": {"order": "asc"}}],
            }

            # Add filters
            for field_name, value in filters.get("filters", {}).items():
                query["query"]["bool"]["must"].append({"match": {field_name: value}})

            url = f"{self.base_url}/{index}/_search"
            response = self._session.post(url, json=query)
            response.raise_for_status()

            data = response.json()
            events = []

            for hit in data["hits"]["hits"]:
                source = hit["_source"]

                event = Event(
                    event_id=hit["_id"],
                    title=source.get("message", "")[:100],
                    text=source.get("message", ""),
                    timestamp=datetime.fromisoformat(
                        source["@timestamp"].replace("Z", "+00:00")
                    ),
                    event_type=source.get("event.type", "info"),
                    tags=self._extract_elastic_tags(source),
                    source=source.get("event.module", ""),
                )
                events.append(event)

            return events

        except Exception as e:
            logger.error(f"Failed to get events from Elasticsearch: {e}")
            return []

    def send_metric(self, metric: Metric) -> bool:
        """Send metric to Elasticsearch"""
        try:
            index = f"homeostasis-metrics-{datetime.utcnow().strftime('%Y.%m.%d')}"

            doc = {
                "@timestamp": metric.timestamp.isoformat(),
                "metric": {
                    "name": f"{self.metric_prefix}.{metric.name}",
                    "value": metric.value,
                    "type": metric.metric_type.value,
                },
                "tags": metric.tags,
                "unit": metric.unit,
                "description": metric.description,
            }

            url = f"{self.base_url}/{index}/_doc"
            response = self._session.post(url, json=doc)
            response.raise_for_status()

            return True

        except Exception as e:
            logger.error(f"Failed to send metric to Elasticsearch: {e}")
            return False

    def send_event(self, event: Event) -> bool:
        """Send event to Elasticsearch"""
        try:
            index = f"homeostasis-events-{datetime.utcnow().strftime('%Y.%m.%d')}"

            doc = {
                "@timestamp": event.timestamp.isoformat(),
                "event": {
                    "id": event.event_id,
                    "title": event.title,
                    "text": event.text,
                    "type": event.event_type,
                    "priority": event.priority,
                    "module": event.source or "homeostasis",
                },
                "tags": event.tags,
            }

            url = f"{self.base_url}/{index}/_doc"
            response = self._session.post(url, json=doc)
            response.raise_for_status()

            return True

        except Exception as e:
            logger.error(f"Failed to send event to Elasticsearch: {e}")
            return False

    def acknowledge_alert(self, alert_id: str, message: str = "") -> bool:
        """Acknowledge alert by updating watcher history"""
        try:
            # Update the watcher history document
            url = f"{self.base_url}/.watcher-history-*/_update/{alert_id}"

            doc = {
                "doc": {
                    "state": "acknowledged",
                    "acknowledged_by": "homeostasis",
                    "acknowledged_at": datetime.utcnow().isoformat(),
                    "acknowledgment_message": message,
                }
            }

            response = self._session.post(url, json=doc)
            response.raise_for_status()

            return True

        except Exception as e:
            logger.error(f"Failed to acknowledge Elasticsearch alert: {e}")
            return False

    def _extract_metric_value(self, source: Dict[str, Any]) -> Optional[float]:
        """Extract metric value from various metricbeat structures"""
        # Try common metric locations

        # System metrics
        if "system" in source:
            if "cpu" in source["system"]:
                return source["system"]["cpu"].get("total", {}).get("pct")
            elif "memory" in source["system"]:
                return source["system"]["memory"].get("used", {}).get("pct")

        # Generic metric field
        if "metric" in source:
            if isinstance(source["metric"], dict):
                return source["metric"].get("value")
            else:
                return float(source["metric"])

        return None

    def _extract_elastic_tags(self, source: Dict[str, Any]) -> Dict[str, str]:
        """Extract tags from Elasticsearch document"""
        tags = {}

        # Common fields to use as tags
        tag_fields = [
            "host.name",
            "service.name",
            "environment",
            "agent.name",
            "cloud.provider",
            "cloud.region",
        ]

        for field_name in tag_fields:
            value = source
            for part in field_name.split("."):
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    value = None
                    break

            if value:
                tags[field_name.replace(".", "_")] = str(value)

        # Add custom tags if present
        if "tags" in source:
            if isinstance(source["tags"], dict):
                tags.update(source["tags"])
            elif isinstance(source["tags"], list):
                for i, tag in enumerate(source["tags"]):
                    tags[f"tag_{i}"] = tag

        return tags

    def _determine_elastic_severity(self, source: Dict[str, Any]) -> AlertSeverity:
        """Determine alert severity from Elasticsearch watcher data"""
        # Check for severity in metadata
        if "metadata" in source and "severity" in source["metadata"]:
            severity_map = {
                "critical": AlertSeverity.CRITICAL,
                "high": AlertSeverity.ERROR,
                "medium": AlertSeverity.WARNING,
                "low": AlertSeverity.INFO,
            }
            return severity_map.get(
                source["metadata"]["severity"].lower(), AlertSeverity.WARNING
            )

        # Default based on result
        if source.get("result", {}).get("condition", {}).get("met"):
            return AlertSeverity.WARNING

        return AlertSeverity.INFO


class NewRelicAdapter(MonitoringAdapter):
    """New Relic monitoring adapter"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key", "")
        self.account_id = config.get("account_id", "")
        self.region = config.get("region", "us")  # us or eu
        self.base_url = (
            f"https://api.{'eu.' if self.region == 'eu' else ''}newrelic.com"
        )
        self._session = None

    def connect(self) -> bool:
        """Connect to New Relic API"""
        try:
            self._session = requests.Session()
            self._session.headers.update(
                {"Api-Key": self.api_key, "Content-Type": "application/json"}
            )

            # Test connection with GraphQL API
            test_query = {"query": "{ actor { user { name email } } }"}

            url = f"{self.base_url}/graphql"
            response = self._session.post(url, json=test_query)
            response.raise_for_status()

            logger.info("Successfully connected to New Relic")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to New Relic: {e}")
            return False

    def get_metrics(
        self, query: Dict[str, Any], start_time: datetime, end_time: datetime
    ) -> List[Metric]:
        """Query metrics from New Relic using NRQL"""
        try:
            # Build NRQL query
            nrql = query.get("nrql", "")
            if not nrql:
                # Build basic query
                metric_name = query.get("metric_name", "*")
                # Validate metric name to prevent injection
                if not metric_name.replace("_", "").replace(".", "").replace("-", "").isalnum() and metric_name != "*":
                    raise ValueError(f"Invalid metric name: {metric_name}")
                nrql = f"SELECT average({metric_name}) FROM Metric"

                # Add WHERE clauses
                where_clauses = []
                for k, v in query.get("where", {}).items():
                    # Validate field name
                    if not k.replace("_", "").replace(".", "").isalnum():
                        raise ValueError(f"Invalid field name: {k}")
                    # Escape single quotes in values
                    escaped_value = str(v).replace("'", "''")
                    where_clauses.append(f"{k} = '{escaped_value}'")

                if where_clauses:
                    nrql += f" WHERE {' AND '.join(where_clauses)}"

            # Add time range
            nrql += f" SINCE {int(start_time.timestamp())} UNTIL {int(end_time.timestamp())}"

            # GraphQL query
            gql_query = {
                "query": f"""
                {{
                    actor {{
                        account(id: {int(self.account_id)}) {{
                            nrql(query: "{nrql}") {{
                                results
                            }}
                        }}
                    }}
                }}
                """
            }

            url = f"{self.base_url}/graphql"
            response = self._session.post(url, json=gql_query)
            response.raise_for_status()

            data = response.json()
            results = data["data"]["actor"]["account"]["nrql"]["results"]
            metrics = []

            for result in results:
                # Extract metric data
                for key, value in result.items():
                    if key not in ["timestamp", "beginTimeSeconds", "endTimeSeconds"]:
                        timestamp = result.get("timestamp") or result.get(
                            "beginTimeSeconds", 0
                        )

                        metric = Metric(
                            name=key,
                            value=float(value) if value is not None else 0,
                            timestamp=datetime.fromtimestamp(timestamp / 1000),
                            tags=self._extract_new_relic_tags(result),
                        )
                        metrics.append(metric)

            return metrics

        except Exception as e:
            logger.error(f"Failed to get metrics from New Relic: {e}")
            return []

    def get_alerts(self, filters: Dict[str, Any]) -> List[Alert]:
        """Get alerts from New Relic"""
        try:
            # Use REST API v2 for alerts
            url = "https://api.newrelic.com/v2/alerts_violations.json"

            params = {"only_open": filters.get("only_open", True), "page": 1}

            headers = {"X-Api-Key": self.api_key, "Content-Type": "application/json"}

            response = requests.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()

            data = response.json()
            alerts = []

            for violation in data.get("violations", []):
                alert = Alert(
                    alert_id=str(violation["id"]),
                    name=violation.get("label", ""),
                    severity=self._map_new_relic_severity(
                        violation.get("priority", "WARNING")
                    ),
                    status=(
                        AlertStatus.FIRING
                        if violation["opened_at"] and not violation.get("closed_at")
                        else AlertStatus.RESOLVED
                    ),
                    message=violation.get("violation_chart_url", ""),
                    source="newrelic",
                    timestamp=datetime.fromtimestamp(violation["opened_at"] / 1000),
                    metadata={
                        "policy_name": violation.get("policy_name"),
                        "condition_name": violation.get("condition_name"),
                        "entity": violation.get("entity"),
                    },
                )
                alerts.append(alert)

            return alerts

        except Exception as e:
            logger.error(f"Failed to get alerts from New Relic: {e}")
            return []

    def get_events(
        self, filters: Dict[str, Any], start_time: datetime, end_time: datetime
    ) -> List[Event]:
        """Get events from New Relic"""
        try:
            # Query events using NRQL
            # Validate timestamps
            start_ts = int(start_time.timestamp() * 1000)
            end_ts = int(end_time.timestamp() * 1000)
            batch_size = int(self.batch_size)
            
            nrql = f"""
            SELECT * FROM Transaction, SystemSample, ProcessSample 
            WHERE timestamp >= {start_ts} 
            AND timestamp <= {end_ts}
            LIMIT {batch_size}
            """

            gql_query = {
                "query": f"""
                {{
                    actor {{
                        account(id: {int(self.account_id)}) {{
                            nrql(query: "{nrql}") {{
                                results
                            }}
                        }}
                    }}
                }}
                """
            }

            url = f"{self.base_url}/graphql"
            response = self._session.post(url, json=gql_query)
            response.raise_for_status()

            data = response.json()
            results = data["data"]["actor"]["account"]["nrql"]["results"]
            events = []

            for result in results:
                event = Event(
                    event_id=str(result.get("timestamp", "")),
                    title=result.get("name", "New Relic Event"),
                    text=json.dumps(result),
                    timestamp=datetime.fromtimestamp(result.get("timestamp", 0) / 1000),
                    event_type=result.get("eventType", "info"),
                    tags=self._extract_new_relic_tags(result),
                )
                events.append(event)

            return events

        except Exception as e:
            logger.error(f"Failed to get events from New Relic: {e}")
            return []

    def send_metric(self, metric: Metric) -> bool:
        """Send metric to New Relic"""
        try:
            url = f"https://metric-api.{'eu.' if self.region == 'eu' else ''}newrelic.com/metric/v1"

            # Format metric for New Relic
            data = [
                {
                    "metrics": [
                        {
                            "name": f"{self.metric_prefix}.{metric.name}",
                            "type": metric.metric_type.value,
                            "value": metric.value,
                            "timestamp": int(metric.timestamp.timestamp()),
                            "attributes": metric.tags,
                        }
                    ]
                }
            ]

            headers = {"Api-Key": self.api_key, "Content-Type": "application/json"}

            response = requests.post(url, json=data, headers=headers, timeout=30)
            response.raise_for_status()

            return True

        except Exception as e:
            logger.error(f"Failed to send metric to New Relic: {e}")
            return False

    def send_event(self, event: Event) -> bool:
        """Send event to New Relic"""
        try:
            url = f"https://insights-collector.{'eu.' if self.region == 'eu' else ''}newrelic.com/v1/accounts/{self.account_id}/events"

            # Format event for New Relic
            data = [
                {
                    "eventType": "HomeostasisEvent",
                    "timestamp": int(event.timestamp.timestamp()),
                    "title": event.title,
                    "text": event.text,
                    "priority": event.priority,
                    "source": event.source,
                    **event.tags,
                }
            ]

            headers = {"X-Insert-Key": self.api_key, "Content-Type": "application/json"}

            response = requests.post(url, json=data, headers=headers, timeout=30)
            response.raise_for_status()

            return True

        except Exception as e:
            logger.error(f"Failed to send event to New Relic: {e}")
            return False

    def acknowledge_alert(self, alert_id: str, message: str = "") -> bool:
        """Acknowledge alert in New Relic"""
        try:
            # New Relic doesn't have direct alert acknowledgment
            # Create an annotation instead
            url = f"https://api.newrelic.com/v2/applications/{self.account_id}/deployments.json"

            data = {
                "deployment": {
                    "revision": f"alert_ack_{alert_id}",
                    "description": message
                    or f"Alert {alert_id} acknowledged by Homeostasis",
                    "user": "homeostasis",
                }
            }

            headers = {"X-Api-Key": self.api_key, "Content-Type": "application/json"}

            response = requests.post(url, json=data, headers=headers, timeout=30)
            response.raise_for_status()

            return True

        except Exception as e:
            logger.error(f"Failed to acknowledge New Relic alert: {e}")
            return False

    def _extract_new_relic_tags(self, result: Dict[str, Any]) -> Dict[str, str]:
        """Extract tags from New Relic result"""
        tags = {}

        # Common attribute fields
        tag_fields = [
            "appName",
            "host",
            "entityName",
            "entityType",
            "environment",
            "service",
            "cluster",
        ]

        for field_name in tag_fields:
            if field_name in result and result[field_name]:
                tags[field_name] = str(result[field_name])

        return tags

    def _map_new_relic_severity(self, priority: str) -> AlertSeverity:
        """Map New Relic priority to AlertSeverity"""
        mapping = {
            "CRITICAL": AlertSeverity.CRITICAL,
            "ERROR": AlertSeverity.ERROR,
            "WARNING": AlertSeverity.WARNING,
            "INFO": AlertSeverity.INFO,
        }
        return mapping.get(priority.upper(), AlertSeverity.WARNING)


# Factory function to create monitoring adapters
def create_monitoring_adapter(
    provider: str, config: Dict[str, Any]
) -> Optional[MonitoringAdapter]:
    """Factory function to create monitoring adapter instances"""
    providers = {
        "datadog": DatadogAdapter,
        "prometheus": PrometheusAdapter,
        "splunk": SplunkAdapter,
        "elasticsearch": ElasticAdapter,
        "elastic": ElasticAdapter,  # Alias
        "newrelic": NewRelicAdapter,
        # Add more providers as implemented
    }

    adapter_class = providers.get(provider.lower())
    if not adapter_class:
        logger.error(f"Unknown monitoring provider: {provider}")
        return None

    try:
        adapter = adapter_class(config)
        if adapter.connect():
            return adapter
        else:
            logger.error(f"Failed to connect to {provider}")
            return None
    except Exception as e:
        logger.error(f"Failed to create {provider} adapter: {e}")
        return None
