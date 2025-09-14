"""
Alerting system for unexpected behavior after fixes.

This module provides utilities for:
1. Monitoring service behavior after fixes
2. Detecting anomalies and unexpected behavior
3. Sending alerts through various channels
"""

import json
import smtplib
import socket
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Callable, Dict, List, Optional

import requests

from modules.monitoring.logger import MonitoringLogger


class AlertManager:
    """
    Manages alerts for unexpected behavior after fixes.
    """

    def __init__(
        self, config: Optional[Dict[str, Any]] = None, log_level: str = "INFO"
    ):
        """
        Initialize the alert manager.

        Args:
            config: Alert configuration dictionary
            log_level: Logging level
        """
        self.logger = MonitoringLogger("alert_manager", log_level=log_level)

        # Default configuration
        self.config: Dict[str, Any] = {
            "channels": {
                "console": True,
                "email": False,
                "slack": False,
                "webhook": False,
            },
            "thresholds": {
                "error_rate": 0.05,  # 5% error rate
                "response_time": 500,  # 500ms
                "memory_usage": 512,  # 512MB
            },
            "contacts": {"email": [], "slack": []},
            "email": {
                "smtp_server": "smtp.example.com",
                "smtp_port": 587,
                "smtp_user": "",
                "smtp_password": "",
                "from_address": "homeostasis@example.com",
            },
            "slack": {"webhook_url": ""},
            "webhook": {"url": ""},
        }

        # Update with provided config
        if config:
            self._update_config(config)

        # Alert history
        self.alert_history: List[Dict[str, Any]] = []

        # Alert handlers
        self.alert_handlers: Dict[str, Callable] = {}

        # Register default handlers
        self._register_default_handlers()

        self.logger.info("Initialized alert manager")

    def _update_config(self, config: Dict[str, Any]) -> None:
        """
        Update configuration with provided values.

        Args:
            config: New configuration values
        """

        # Deep merge configuration
        def merge_dicts(d1, d2):
            for key, value in d2.items():
                if key in d1 and isinstance(d1[key], dict) and isinstance(value, dict):
                    merge_dicts(d1[key], value)
                else:
                    d1[key] = value

        merge_dicts(self.config, config)

    def _register_default_handlers(self) -> None:
        """Register default alert handlers."""
        # Console handler
        self.register_alert_handler("console", self._console_alert_handler)

        # Email handler
        self.register_alert_handler("email", self._email_alert_handler)

        # Slack handler
        self.register_alert_handler("slack", self._slack_alert_handler)

        # Webhook handler
        self.register_alert_handler("webhook", self._webhook_alert_handler)

    def register_alert_handler(
        self, channel: str, handler: Callable[[str, Dict[str, Any]], None]
    ) -> None:
        """
        Register an alert handler for a channel.

        Args:
            channel: Alert channel name
            handler: Function that takes alert message and data
        """
        self.alert_handlers[channel] = handler
        self.logger.info(f"Registered alert handler for channel: {channel}")

    def _console_alert_handler(self, message: str, data: Dict[str, Any]) -> None:
        """
        Handle console alerts.

        Args:
            message: Alert message
            data: Alert data
        """
        formatted_time = time.strftime("%Y-%m-%d %H:%M:%S")
        self.logger.warning(f"[ALERT] {formatted_time} - {message}", data=data)

        # Print to console (for interactive sessions)
        print(f"\n[ALERT] {formatted_time} - {message}")
        if data:
            print(f"  Data: {json.dumps(data, indent=2)}")

    def _email_alert_handler(self, message: str, data: Dict[str, Any]) -> None:
        """
        Handle email alerts.

        Args:
            message: Alert message
            data: Alert data
        """
        email_config = self.config["email"]
        contacts = self.config["contacts"]["email"]

        if not email_config["smtp_server"] or not contacts:
            self.logger.warning("Email alert skipped - missing configuration")
            return

        try:
            # Create message
            for recipient in contacts:
                msg = MIMEMultipart()
                msg["From"] = email_config["from_address"]
                msg["To"] = recipient
                msg["Subject"] = f"[Homeostasis Alert] {message[:50]}..."

                # Create message body
                body = f"""
                <html>
                <body>
                <h2>Homeostasis Alert</h2>
                <p><strong>Message:</strong> {message}</p>
                <h3>Alert Data</h3>
                <pre>{json.dumps(data, indent=2)}</pre>
                <hr>
                <p><em>This is an automated alert from the Homeostasis self-healing system.</em></p>
                </body>
                </html>
                """

                msg.attach(MIMEText(body, "html"))

                # Send the message
                with smtplib.SMTP(
                    email_config["smtp_server"], email_config["smtp_port"]
                ) as server:
                    if email_config["smtp_user"] and email_config["smtp_password"]:
                        server.starttls()
                        server.login(
                            email_config["smtp_user"], email_config["smtp_password"]
                        )

                    server.send_message(msg)

                self.logger.info(f"Sent email alert to {recipient}")

        except Exception as e:
            self.logger.exception(e, message=f"Failed to send email alert: {str(e)}")

    def _slack_alert_handler(self, message: str, data: Dict[str, Any]) -> None:
        """
        Handle Slack alerts.

        Args:
            message: Alert message
            data: Alert data
        """
        slack_config = self.config["slack"]

        if not slack_config["webhook_url"]:
            self.logger.warning("Slack alert skipped - missing webhook URL")
            return

        try:
            # Format code block for data
            data_text = f"```{json.dumps(data, indent=2)}```" if data else ""

            # Create payload
            payload: Dict[str, Any] = {
                "text": "*[Homeostasis Alert]*",
                "blocks": [
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": f"*Alert:* {message}"},
                    }
                ],
            }

            # Add data if present
            if data_text:
                payload["blocks"].append(
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": f"*Data:*\n{data_text}"},
                    }
                )

            # Send to Slack
            response = requests.post(
                slack_config["webhook_url"],
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30,
            )

            if response.status_code != 200:
                self.logger.warning(f"Failed to send Slack alert: {response.text}")
            else:
                self.logger.info("Sent Slack alert")

        except Exception as e:
            self.logger.exception(e, message=f"Failed to send Slack alert: {str(e)}")

    def _webhook_alert_handler(self, message: str, data: Dict[str, Any]) -> None:
        """
        Handle webhook alerts.

        Args:
            message: Alert message
            data: Alert data
        """
        webhook_config = self.config["webhook"]

        if not webhook_config["url"]:
            self.logger.warning("Webhook alert skipped - missing URL")
            return

        try:
            # Create payload
            payload = {
                "message": message,
                "data": data,
                "timestamp": time.time(),
                "hostname": socket.gethostname(),
            }

            # Send to webhook
            response = requests.post(
                webhook_config["url"],
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30,
            )

            if response.status_code >= 400:
                self.logger.warning(f"Failed to send webhook alert: {response.text}")
            else:
                self.logger.info("Sent webhook alert")

        except Exception as e:
            self.logger.exception(e, message=f"Failed to send webhook alert: {str(e)}")

    def send_alert(
        self,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        level: str = "warning",
        channels: Optional[List[str]] = None,
    ) -> None:
        """
        Send an alert through configured channels.

        Args:
            message: Alert message
            data: Optional alert data
            level: Alert level (info, warning, error, critical)
            channels: Optional list of channels to use (defaults to all enabled)
        """
        # Use all enabled channels if not specified
        if channels is None:
            channels = [
                ch for ch, enabled in self.config["channels"].items() if enabled
            ]

        # Record alert
        alert_record = {
            "message": message,
            "data": data or {},
            "level": level,
            "timestamp": time.time(),
            "channels": channels,
        }

        self.alert_history.append(alert_record)

        # Send through each channel
        for channel in channels:
            if channel in self.alert_handlers:
                try:
                    self.alert_handlers[channel](message, data or {})
                except Exception as e:
                    self.logger.exception(
                        e, message=f"Error in alert handler for {channel}: {str(e)}"
                    )
            else:
                self.logger.warning(f"No handler registered for channel: {channel}")

        self.logger.info(f"Sent alert through {len(channels)} channels", level=level)

    def check_metric_thresholds(self, metrics: Dict[str, Any], patch_id: str) -> None:
        """
        Check metrics against thresholds and send alerts if needed.

        Args:
            metrics: Metrics data
            patch_id: ID of the patch being monitored
        """
        thresholds = self.config["thresholds"]

        for metric_name, threshold in thresholds.items():
            if metric_name in metrics:
                metric_value = metrics[metric_name]

                # Check if threshold is exceeded
                if metric_value > threshold:
                    self.send_alert(
                        f"Metric {metric_name} exceeded threshold for patch {patch_id}",
                        {
                            "patch_id": patch_id,
                            "metric": metric_name,
                            "value": metric_value,
                            "threshold": threshold,
                        },
                    )

    def check_for_anomalies(
        self,
        current_metrics: Dict[str, Any],
        historical_metrics: List[Dict[str, Any]],
        patch_id: str,
    ) -> None:
        """
        Check for anomalies in metrics compared to historical data.

        Args:
            current_metrics: Current metrics data
            historical_metrics: Historical metrics data
            patch_id: ID of the patch being monitored
        """
        if not historical_metrics:
            return

        anomalies = []

        # Calculate standard deviation for each metric
        for metric_name in current_metrics:
            # Skip non-numeric metrics
            if not isinstance(current_metrics[metric_name], (int, float)):
                continue

            # Get historical values
            historical_values: List[float] = [
                float(m.get(metric_name))
                for m in historical_metrics
                if metric_name in m and isinstance(m.get(metric_name), (int, float))
            ]

            if len(historical_values) < 5:
                continue

            # Calculate mean and standard deviation
            import statistics

            try:
                mean = statistics.mean(historical_values)
                stdev = statistics.stdev(historical_values)

                # Get current value
                current_value = current_metrics[metric_name]

                # Calculate Z-score
                if stdev > 0:
                    z_score = (current_value - mean) / stdev

                    # Check for anomaly (Z-score > 3 or < -3)
                    if abs(z_score) > 3:
                        anomalies.append(
                            {
                                "metric": metric_name,
                                "current_value": current_value,
                                "mean": mean,
                                "stdev": stdev,
                                "z_score": z_score,
                            }
                        )
            except statistics.StatisticsError:
                continue

        # Send alert if anomalies found
        if anomalies:
            self.send_alert(
                f"Anomalies detected in metrics for patch {patch_id}",
                {"patch_id": patch_id, "anomalies": anomalies},
            )

    def generate_alert_digest(self) -> Dict[str, Any]:
        """
        Generate a digest of recent alerts.

        Returns:
            Alert digest
        """
        if not self.alert_history:
            return {"count": 0, "message": "No alerts recorded"}

        # Group alerts by level
        alerts_by_level: Dict[str, List[Dict[str, Any]]] = {}
        for alert in self.alert_history:
            level = alert["level"]
            if level not in alerts_by_level:
                alerts_by_level[level] = []

            alerts_by_level[level].append(alert)

        # Count alerts by level
        counts = {level: len(alerts) for level, alerts in alerts_by_level.items()}

        # Get recent alerts
        recent_alerts = sorted(
            self.alert_history, key=lambda x: x["timestamp"], reverse=True
        )[:10]

        return {
            "count": len(self.alert_history),
            "counts_by_level": counts,
            "recent_alerts": recent_alerts,
        }


class AnomalyDetector:
    """
    Detects anomalies in service behavior after fixes.
    """

    def __init__(self, alert_manager: AlertManager, log_level: str = "INFO"):
        """
        Initialize the anomaly detector.

        Args:
            alert_manager: Alert manager for sending alerts
            log_level: Logging level
        """
        self.logger = MonitoringLogger("anomaly_detector", log_level=log_level)
        self.alert_manager = alert_manager

        # Baseline metrics
        self.baselines: Dict[str, Dict[str, Any]] = {}

        # Detection thresholds
        self.thresholds = {
            "error_rate_increase": 0.02,  # 2% increase in error rate
            "response_time_increase": 50,  # 50ms increase in response time
            "memory_usage_increase": 100,  # 100MB increase in memory usage
            "error_spike": 0.1,  # 10% error rate spike
            "response_time_spike": 200,  # 200ms response time spike
        }

        self.logger.info("Initialized anomaly detector")

    def establish_baseline(
        self, service_url: str, duration: int = 300, interval: int = 10
    ) -> Dict[str, Any]:
        """
        Establish a baseline for service behavior.

        Args:
            service_url: Base URL of the service
            duration: Monitoring duration in seconds
            interval: Check interval in seconds

        Returns:
            Baseline metrics
        """
        self.logger.info(f"Establishing baseline for {service_url} over {duration}s")

        baseline_metrics = []
        end_time = time.time() + duration

        try:
            while time.time() < end_time:
                # Collect metrics
                metrics = self._collect_metrics(service_url)
                baseline_metrics.append(metrics)

                # Wait for the next interval
                time.sleep(interval)

            # Calculate baseline
            baseline = self._calculate_baseline(baseline_metrics)

            # Store baseline
            self.baselines[service_url] = baseline

            self.logger.info(
                f"Established baseline for {service_url}", baseline=baseline
            )

            return baseline

        except Exception as e:
            self.logger.exception(
                e, message=f"Failed to establish baseline for {service_url}"
            )
            return {}

    def _collect_metrics(self, service_url: str) -> Dict[str, Any]:
        """
        Collect metrics from a service.

        Args:
            service_url: Base URL of the service

        Returns:
            Collected metrics
        """
        metrics = {"timestamp": time.time()}

        # Response time (health check)
        try:
            start_time = time.time()
            response = requests.get(f"{service_url}/health", timeout=5)
            elapsed = time.time() - start_time

            metrics["response_time"] = elapsed * 1000  # Convert to ms
            metrics["status_code"] = response.status_code

            # Try to parse response body if it's JSON
            try:
                response_json = response.json()
                if "memory_usage" in response_json:
                    metrics["memory_usage"] = response_json["memory_usage"]
            except Exception:
                pass

        except requests.RequestException as e:
            self.logger.warning(f"Health check failed: {str(e)}")
            metrics["error"] = str(e)

        # Error rate (sample requests to endpoints)
        error_count = 0
        request_count = 0

        # Try common endpoints
        for endpoint in ["/", "/api", "/status"]:
            try:
                response = requests.get(f"{service_url}{endpoint}", timeout=5)
                request_count += 1
                if response.status_code >= 400:
                    error_count += 1
            except requests.RequestException:
                request_count += 1
                error_count += 1

        if request_count > 0:
            metrics["error_rate"] = error_count / request_count

        return metrics

    def _calculate_baseline(self, metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate baseline from collected metrics.

        Args:
            metrics: List of metrics data points

        Returns:
            Baseline metrics
        """
        import statistics

        baseline = {
            "count": len(metrics),
            "start_time": metrics[0]["timestamp"] if metrics else time.time(),
            "end_time": metrics[-1]["timestamp"] if metrics else time.time(),
        }

        # Calculate statistics for each metric
        for metric_name in ["response_time", "error_rate", "memory_usage"]:
            # Extract values for this metric
            values = [m.get(metric_name) for m in metrics if metric_name in m]

            if not values:
                continue

            try:
                baseline[metric_name] = {
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "min": min(values),
                    "max": max(values),
                    "stddev": statistics.stdev(values) if len(values) > 1 else 0,
                }
            except statistics.StatisticsError:
                pass

        return baseline

    def monitor_for_anomalies(
        self, service_url: str, patch_id: str, duration: int = 3600, interval: int = 60
    ) -> Dict[str, Any]:
        """
        Monitor a service for anomalies after a fix.

        Args:
            service_url: Base URL of the service
            patch_id: ID of the patch to monitor
            duration: Monitoring duration in seconds
            interval: Check interval in seconds
        """
        self.logger.info(
            f"Starting anomaly monitoring for patch {patch_id} at {service_url}"
        )

        # Ensure we have a baseline
        if service_url not in self.baselines:
            self.logger.info(f"No baseline found for {service_url}, establishing one")
            self.establish_baseline(service_url)

        baseline = self.baselines.get(service_url, {})

        # Start monitoring
        end_time = time.time() + duration
        monitoring_metrics = []

        try:
            while time.time() < end_time:
                # Collect metrics
                metrics = self._collect_metrics(service_url)
                monitoring_metrics.append(metrics)

                # Check for anomalies
                if baseline:
                    self._check_for_anomalies(metrics, baseline, patch_id)

                # Check metric thresholds
                self.alert_manager.check_metric_thresholds(metrics, patch_id)

                # Wait for the next interval
                time.sleep(interval)

            self.logger.info(f"Completed anomaly monitoring for patch {patch_id}")

            # Generate summary
            summary = self._generate_monitoring_summary(
                monitoring_metrics, baseline, patch_id
            )

            # Log summary
            self.logger.info(
                f"Monitoring summary for patch {patch_id}", summary=summary
            )

            return summary

        except Exception as e:
            self.logger.exception(
                e, message=f"Failed to monitor {service_url} for anomalies"
            )
            return {}

    def _check_for_anomalies(
        self, current_metrics: Dict[str, Any], baseline: Dict[str, Any], patch_id: str
    ) -> None:
        """
        Check for anomalies in current metrics compared to baseline.

        Args:
            current_metrics: Current metrics data
            baseline: Baseline metrics
            patch_id: ID of the patch being monitored
        """
        anomalies = []

        # Check response time
        if "response_time" in current_metrics and "response_time" in baseline:
            current_rt = current_metrics["response_time"]
            baseline_rt = baseline["response_time"]["mean"]
            baseline_rt_stddev = baseline["response_time"].get("stddev", 0)

            # Calculate Z-score
            if baseline_rt_stddev > 0:
                z_score = (current_rt - baseline_rt) / baseline_rt_stddev

                # Check for anomaly (Z-score > 3)
                if z_score > 3:
                    anomalies.append(
                        {
                            "metric": "response_time",
                            "current_value": current_rt,
                            "baseline_value": baseline_rt,
                            "z_score": z_score,
                            "increase_percent": (
                                ((current_rt - baseline_rt) / baseline_rt) * 100
                                if baseline_rt > 0
                                else 0
                            ),
                        }
                    )

            # Check for significant increase
            rt_increase = current_rt - baseline_rt
            if rt_increase > self.thresholds["response_time_increase"]:
                anomalies.append(
                    {
                        "metric": "response_time_increase",
                        "current_value": current_rt,
                        "baseline_value": baseline_rt,
                        "increase": rt_increase,
                    }
                )

            # Check for spike
            if rt_increase > self.thresholds["response_time_spike"]:
                anomalies.append(
                    {
                        "metric": "response_time_spike",
                        "current_value": current_rt,
                        "baseline_value": baseline_rt,
                        "increase": rt_increase,
                    }
                )

        # Check error rate
        if "error_rate" in current_metrics and "error_rate" in baseline:
            current_er = current_metrics["error_rate"]
            baseline_er = baseline["error_rate"]["mean"]

            # Check for significant increase
            er_increase = current_er - baseline_er
            if er_increase > self.thresholds["error_rate_increase"]:
                anomalies.append(
                    {
                        "metric": "error_rate_increase",
                        "current_value": current_er,
                        "baseline_value": baseline_er,
                        "increase": er_increase,
                    }
                )

            # Check for spike
            if current_er > self.thresholds["error_spike"]:
                anomalies.append(
                    {
                        "metric": "error_rate_spike",
                        "current_value": current_er,
                        "baseline_value": baseline_er,
                        "increase": er_increase,
                    }
                )

        # Check memory usage
        if "memory_usage" in current_metrics and "memory_usage" in baseline:
            current_mu = current_metrics["memory_usage"]
            baseline_mu = baseline["memory_usage"]["mean"]

            # Check for significant increase
            mu_increase = current_mu - baseline_mu
            if mu_increase > self.thresholds["memory_usage_increase"]:
                anomalies.append(
                    {
                        "metric": "memory_usage_increase",
                        "current_value": current_mu,
                        "baseline_value": baseline_mu,
                        "increase": mu_increase,
                    }
                )

        # Send alert if anomalies found
        if anomalies:
            self.alert_manager.send_alert(
                f"Anomalies detected in service behavior after patch {patch_id}",
                {
                    "patch_id": patch_id,
                    "anomalies": anomalies,
                    "current_metrics": current_metrics,
                },
            )

    def _generate_monitoring_summary(
        self, metrics: List[Dict[str, Any]], baseline: Dict[str, Any], patch_id: str
    ) -> Dict[str, Any]:
        """
        Generate a summary of monitoring results.

        Args:
            metrics: Monitoring metrics
            baseline: Baseline metrics
            patch_id: ID of the patch

        Returns:
            Monitoring summary
        """
        if not metrics:
            return {"patch_id": patch_id, "count": 0, "message": "No metrics collected"}

        # Calculate statistics for monitoring period
        monitoring_stats = self._calculate_baseline(metrics)

        # Compare with baseline
        comparison = {}

        for metric_name in ["response_time", "error_rate", "memory_usage"]:
            if metric_name in monitoring_stats and metric_name in baseline:
                monitoring_mean = monitoring_stats[metric_name]["mean"]
                baseline_mean = baseline[metric_name]["mean"]

                change = monitoring_mean - baseline_mean
                percent_change = (
                    (change / baseline_mean) * 100 if baseline_mean > 0 else 0
                )

                comparison[metric_name] = {
                    "monitoring_mean": monitoring_mean,
                    "baseline_mean": baseline_mean,
                    "change": change,
                    "percent_change": percent_change,
                }

        # Overall assessment
        success = True
        for metric_name, data in comparison.items():
            # Consider significant degradation a failure
            if data["percent_change"] > 20 and data["change"] > 0:  # 20% degradation
                success = False
                break

        return {
            "patch_id": patch_id,
            "monitoring_period": {
                "start": metrics[0]["timestamp"] if metrics else 0,
                "end": metrics[-1]["timestamp"] if metrics else 0,
                "duration": (
                    (metrics[-1]["timestamp"] - metrics[0]["timestamp"])
                    if metrics
                    else 0
                ),
            },
            "metrics_count": len(metrics),
            "comparison": comparison,
            "success": success,
        }


if __name__ == "__main__":
    # Example usage
    alert_manager = AlertManager()

    # Register a custom alert handler
    def custom_handler(message, data):
        print(f"CUSTOM ALERT: {message}")
        print(f"Data: {data}")

    alert_manager.register_alert_handler("custom", custom_handler)

    # Send an alert
    alert_manager.send_alert(
        "This is a test alert", {"test": True}, channels=["console", "custom"]
    )

    # Anomaly detector
    detector = AnomalyDetector(alert_manager)

    # Example monitoring (if a service was running)
    try:
        # Establish baseline
        baseline = detector.establish_baseline(
            "http://localhost:8000", duration=10, interval=2
        )
        print(f"Established baseline: {baseline}")

        # Monitor for anomalies
        summary = detector.monitor_for_anomalies(
            "http://localhost:8000", "test-patch-1", duration=10, interval=2
        )

        print(f"Monitoring summary: {summary}")

    except Exception as e:
        print(f"Error during monitoring: {str(e)}")
