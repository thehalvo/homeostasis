"""
LLM metrics collection and monitoring module.

This module provides comprehensive observability for LLM operations including:
1. Request/response metrics tracking
2. Cost monitoring and quota management
3. PII/security guardrails
4. Provider performance analytics
"""

import hashlib
import json
import re
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

from .logger import MonitoringLogger
from .metrics_collector import MetricsCollector


@dataclass
class LLMRequestMetrics:
    """Metrics for a single LLM request."""

    request_id: str
    provider: str
    model: str
    timestamp: float
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cost: Optional[float] = None
    latency: Optional[float] = None
    success: bool = True
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    content_hash: Optional[str] = None  # Hash of content for deduplication
    pii_detected: bool = False
    unsafe_content_detected: bool = False


@dataclass
class UsageQuota:
    """Usage quota configuration."""

    tokens_per_hour: Optional[int] = None
    tokens_per_day: Optional[int] = None
    cost_per_hour: Optional[float] = None
    cost_per_day: Optional[float] = None
    cost_per_month: Optional[float] = None
    requests_per_minute: Optional[int] = None
    requests_per_hour: Optional[int] = None


@dataclass
class AlertConfig:
    """Alert configuration for usage monitoring."""

    cost_threshold_percentage: float = 80.0  # Alert at 80% of quota
    token_threshold_percentage: float = 90.0  # Alert at 90% of token quota
    request_threshold_percentage: float = 85.0  # Alert at 85% of request quota
    consecutive_failures_threshold: int = 5
    high_latency_threshold: float = 30.0  # seconds
    alert_cooldown: float = 300.0  # 5 minutes between alerts


class PIIDetector:
    """Detects potentially sensitive information in prompts and responses."""

    def __init__(self):
        """Initialize PII detection patterns."""
        self.patterns = {
            "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
            "phone": re.compile(
                r"\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b"
            ),
            "ssn": re.compile(r"\b\d{3}-?\d{2}-?\d{4}\b"),
            "credit_card": re.compile(
                r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3[0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b"
            ),
            "api_key": re.compile(r"\b(?:sk-|pk_|rk_)[a-zA-Z0-9]{20,}\b"),
            "jwt_token": re.compile(
                r"\beyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+\b"
            ),
            "password": re.compile(
                r'(?i)(?:password|passwd|pwd)[\s]*[:=][\s]*[\'"]?([^\s\'"]+)',
                re.IGNORECASE,
            ),
            "secret": re.compile(
                r'(?i)(?:secret|token|key)[\s]*[:=][\s]*[\'"]?([^\s\'"]+)',
                re.IGNORECASE,
            ),
            "private_key": re.compile(r"-----BEGIN (?:RSA |EC |DSA )?PRIVATE KEY-----"),
            "ip_address": re.compile(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b"),
            "url_with_credentials": re.compile(r"https?://[^:]+:[^@]+@[^\s]+"),
            "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
            "github_token": re.compile(r"\bghp_[a-zA-Z0-9]{36}\b"),
            "slack_token": re.compile(
                r"\bxox[baprs]-[0-9]{12}-[0-9]{12}-[0-9a-zA-Z]{24}\b"
            ),
            "stripe_key": re.compile(r"\b(?:sk|pk)_(?:test|live)_[a-zA-Z0-9]{24,}\b"),
        }

        # Additional context-based patterns
        self.context_patterns = [
            re.compile(r"(?i)my\s+(?:password|ssn|social\s+security)", re.IGNORECASE),
            re.compile(r"(?i)personal\s+(?:info|information|details)", re.IGNORECASE),
            re.compile(r"(?i)confidential", re.IGNORECASE),
        ]

    def detect_pii(self, text: str) -> Tuple[bool, List[str]]:
        """
        Detect PII in text.

        Args:
            text: Text to analyze

        Returns:
            Tuple of (has_pii, detected_types)
        """
        detected_types = []

        for pii_type, pattern in self.patterns.items():
            if pattern.search(text):
                detected_types.append(pii_type)

        # Check context patterns
        for pattern in self.context_patterns:
            if pattern.search(text):
                detected_types.append("contextual_pii")
                break

        return len(detected_types) > 0, detected_types

    def scrub_pii(self, text: str) -> str:
        """
        Scrub PII from text by replacing with placeholders.

        Args:
            text: Text to scrub

        Returns:
            Scrubbed text
        """
        scrubbed = text

        # Replace specific patterns
        replacements = {
            "email": "[EMAIL]",
            "phone": "[PHONE]",
            "ssn": "[SSN]",
            "credit_card": "[CREDIT_CARD]",
            "api_key": "[API_KEY]",
            "jwt_token": "[JWT_TOKEN]",
            "private_key": "[PRIVATE_KEY]",
            "ip_address": "[IP_ADDRESS]",
            "aws_access_key": "[AWS_KEY]",
            "github_token": "[GITHUB_TOKEN]",
            "slack_token": "[SLACK_TOKEN]",
            "stripe_key": "[STRIPE_KEY]",
        }

        for pii_type, replacement in replacements.items():
            if pii_type in self.patterns:
                scrubbed = self.patterns[pii_type].sub(replacement, scrubbed)

        # Handle password and secret patterns (more complex regex)
        scrubbed = re.sub(
            r'(?i)(?:password|passwd|pwd)[\s]*[:=][\s]*[\'"]?([^\s\'"]+)',
            r"password=[PASSWORD]",
            scrubbed,
            flags=re.IGNORECASE,
        )
        scrubbed = re.sub(
            r'(?i)(?:secret|token|key)[\s]*[:=][\s]*[\'"]?([^\s\'"]+)',
            r"secret=[SECRET]",
            scrubbed,
            flags=re.IGNORECASE,
        )

        return scrubbed


class UnsafeContentDetector:
    """Detects potentially unsafe content in prompts and responses."""

    def __init__(self):
        """Initialize unsafe content detection patterns."""
        self.patterns = {
            "malicious_injection": [
                re.compile(
                    r"(?i)ignore\s+(?:previous|all)\s+(?:instructions|prompts)",
                    re.IGNORECASE,
                ),
                re.compile(r"(?i)system\s*:\s*(?:jailbreak|override)", re.IGNORECASE),
                re.compile(r"(?i)act\s+as\s+if\s+you\s+are", re.IGNORECASE),
                re.compile(r"(?i)pretend\s+(?:you\s+are|to\s+be)", re.IGNORECASE),
            ],
            "prompt_leakage": [
                re.compile(
                    r"(?i)show\s+me\s+your\s+(?:system\s+)?(?:prompt|instructions)",
                    re.IGNORECASE,
                ),
                re.compile(
                    r"(?i)what\s+(?:are\s+your|is\s+your)\s+(?:system\s+)?(?:prompt|instructions)",
                    re.IGNORECASE,
                ),
                re.compile(
                    r"(?i)reveal\s+your\s+(?:system\s+)?(?:prompt|instructions)",
                    re.IGNORECASE,
                ),
            ],
            "data_extraction": [
                re.compile(
                    r"(?i)(?:extract|dump|show)\s+(?:all\s+)?(?:data|information|content)",
                    re.IGNORECASE,
                ),
                re.compile(
                    r"(?i)list\s+all\s+(?:files|users|secrets|keys)", re.IGNORECASE
                ),
            ],
        }

    def detect_unsafe_content(self, text: str) -> Tuple[bool, List[str]]:
        """
        Detect unsafe content in text.

        Args:
            text: Text to analyze

        Returns:
            Tuple of (has_unsafe_content, detected_types)
        """
        detected_types = []

        for content_type, patterns in self.patterns.items():
            for pattern in patterns:
                if pattern.search(text):
                    detected_types.append(content_type)
                    break

        return len(detected_types) > 0, detected_types


class CostCalculator:
    """Calculates costs for different LLM providers and models."""

    def __init__(self):
        """Initialize cost calculation data."""
        # Cost per 1K tokens (as of 2024, subject to change)
        self.pricing = {
            "openai": {
                "gpt-4": {"prompt": 0.03, "completion": 0.06},
                "gpt-4-turbo": {"prompt": 0.01, "completion": 0.03},
                "gpt-3.5-turbo": {"prompt": 0.0005, "completion": 0.0015},
                "gpt-3.5-turbo-16k": {"prompt": 0.003, "completion": 0.004},
            },
            "anthropic": {
                "claude-3-opus-20240229": {"prompt": 0.015, "completion": 0.075},
                "claude-3-sonnet-20240229": {"prompt": 0.003, "completion": 0.015},
                "claude-3-haiku-20240307": {"prompt": 0.00025, "completion": 0.00125},
                "claude-2.1": {"prompt": 0.008, "completion": 0.024},
                "claude-2.0": {"prompt": 0.008, "completion": 0.024},
            },
            "openrouter": {
                # OpenRouter pricing varies by model, using average estimates
                "anthropic/claude-3-opus": {"prompt": 0.015, "completion": 0.075},
                "anthropic/claude-3-sonnet": {"prompt": 0.003, "completion": 0.015},
                "anthropic/claude-3-haiku": {"prompt": 0.00025, "completion": 0.00125},
                "openai/gpt-4": {"prompt": 0.03, "completion": 0.06},
                "openai/gpt-3.5-turbo": {"prompt": 0.0005, "completion": 0.0015},
            },
        }

    def calculate_cost(
        self, provider: str, model: str, prompt_tokens: int, completion_tokens: int
    ) -> float:
        """
        Calculate cost for a request.

        Args:
            provider: Provider name
            model: Model name
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens

        Returns:
            Estimated cost in USD
        """
        provider = provider.lower()

        if provider not in self.pricing:
            return 0.0

        provider_pricing = self.pricing[provider]

        # Find model pricing (try exact match first, then partial match)
        model_pricing = None
        if model in provider_pricing:
            model_pricing = provider_pricing[model]
        else:
            # Try partial match for model names
            for pricing_model, pricing in provider_pricing.items():
                if (
                    model.lower() in pricing_model.lower()
                    or pricing_model.lower() in model.lower()
                ):
                    model_pricing = pricing
                    break

        if not model_pricing:
            # Use a default pricing if model not found
            avg_pricing = (
                list(provider_pricing.values())[0]
                if provider_pricing
                else {"prompt": 0.001, "completion": 0.002}
            )
            model_pricing = avg_pricing

        # Calculate cost (pricing is per 1K tokens)
        prompt_cost = (prompt_tokens / 1000.0) * model_pricing["prompt"]
        completion_cost = (completion_tokens / 1000.0) * model_pricing["completion"]

        return prompt_cost + completion_cost


class LLMMetricsCollector:
    """Comprehensive LLM metrics collection and monitoring."""

    def __init__(
        self,
        storage_dir: Optional[Path] = None,
        quota_config: Optional[UsageQuota] = None,
        alert_config: Optional[AlertConfig] = None,
    ):
        """
        Initialize LLM metrics collector.

        Args:
            storage_dir: Directory to store metrics
            quota_config: Usage quota configuration
            alert_config: Alert configuration
        """
        self.logger = MonitoringLogger("llm_metrics")
        self.metrics_collector = MetricsCollector(storage_dir)

        # Initialize components
        self.pii_detector = PIIDetector()
        self.unsafe_content_detector = UnsafeContentDetector()
        self.cost_calculator = CostCalculator()

        # Configuration
        self.quota_config = quota_config or UsageQuota()
        self.alert_config = alert_config or AlertConfig()

        # Metrics storage
        self.recent_requests: Deque[LLMRequestMetrics] = deque(
            maxlen=1000
        )  # Keep last 1000 requests in memory
        self.usage_windows: Dict[str, Deque[Any]] = {
            "minute": deque(maxlen=60),
            "hour": deque(maxlen=24),
            "day": deque(maxlen=30),
        }

        # Thread safety
        self._lock = threading.Lock()

        # Alert state
        self._last_alert_times: Dict[str, float] = defaultdict(float)

        self.logger.info("Initialized LLM metrics collector")

    def _generate_content_hash(self, content: str) -> str:
        """Generate hash for content deduplication."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _check_quota_usage(self, current_usage: Dict[str, Any]) -> List[str]:
        """
        Check if current usage exceeds quota thresholds.

        Args:
            current_usage: Current usage statistics

        Returns:
            List of quota violations
        """
        violations = []

        # Check token quotas
        if self.quota_config.tokens_per_hour:
            hourly_tokens = current_usage.get("tokens_last_hour", 0)
            threshold = self.quota_config.tokens_per_hour * (
                self.alert_config.token_threshold_percentage / 100.0
            )
            if hourly_tokens >= threshold:
                violations.append(
                    f"Hourly token usage ({hourly_tokens}) exceeds threshold ({threshold:.0f})"
                )

        if self.quota_config.tokens_per_day:
            daily_tokens = current_usage.get("tokens_last_day", 0)
            threshold = self.quota_config.tokens_per_day * (
                self.alert_config.token_threshold_percentage / 100.0
            )
            if daily_tokens >= threshold:
                violations.append(
                    f"Daily token usage ({daily_tokens}) exceeds threshold ({threshold:.0f})"
                )

        # Check cost quotas
        if self.quota_config.cost_per_hour:
            hourly_cost = current_usage.get("cost_last_hour", 0.0)
            threshold = self.quota_config.cost_per_hour * (
                self.alert_config.cost_threshold_percentage / 100.0
            )
            if hourly_cost >= threshold:
                violations.append(
                    f"Hourly cost (${hourly_cost:.4f}) exceeds threshold (${threshold:.4f})"
                )

        if self.quota_config.cost_per_day:
            daily_cost = current_usage.get("cost_last_day", 0.0)
            threshold = self.quota_config.cost_per_day * (
                self.alert_config.cost_threshold_percentage / 100.0
            )
            if daily_cost >= threshold:
                violations.append(
                    f"Daily cost (${daily_cost:.4f}) exceeds threshold (${threshold:.4f})"
                )

        # Check request rate quotas
        if self.quota_config.requests_per_minute:
            minute_requests = current_usage.get("requests_last_minute", 0)
            threshold = self.quota_config.requests_per_minute * (
                self.alert_config.request_threshold_percentage / 100.0
            )
            if minute_requests >= threshold:
                violations.append(
                    f"Requests per minute ({minute_requests}) exceeds threshold ({threshold:.0f})"
                )

        if self.quota_config.requests_per_hour:
            hourly_requests = current_usage.get("requests_last_hour", 0)
            threshold = self.quota_config.requests_per_hour * (
                self.alert_config.request_threshold_percentage / 100.0
            )
            if hourly_requests >= threshold:
                violations.append(
                    f"Hourly requests ({hourly_requests}) exceeds threshold ({threshold:.0f})"
                )

        return violations

    def _should_send_alert(self, alert_type: str) -> bool:
        """
        Check if an alert should be sent based on cooldown period.

        Args:
            alert_type: Type of alert

        Returns:
            True if alert should be sent
        """
        current_time = time.time()
        last_alert_time = self._last_alert_times.get(alert_type, 0)

        if current_time - last_alert_time >= self.alert_config.alert_cooldown:
            self._last_alert_times[alert_type] = current_time
            return True

        return False

    def record_request(
        self,
        request_id: str,
        provider: str,
        model: str,
        prompt_content: str,
        response_content: str,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        latency: Optional[float] = None,
        success: bool = True,
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> LLMRequestMetrics:
        """
        Record a complete LLM request with comprehensive metrics.

        Args:
            request_id: Unique request identifier
            provider: LLM provider name
            model: Model name
            prompt_content: Content of the prompt
            response_content: Content of the response
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            total_tokens: Total tokens used
            latency: Request latency in seconds
            success: Whether request was successful
            error_type: Type of error if unsuccessful
            error_message: Error message if unsuccessful

        Returns:
            LLM request metrics object
        """
        current_time = time.time()

        with self._lock:
            # Content analysis
            combined_content = f"{prompt_content}\n{response_content}"
            content_hash = self._generate_content_hash(combined_content)

            # PII detection
            pii_detected, pii_types = self.pii_detector.detect_pii(combined_content)

            # Unsafe content detection
            unsafe_detected, unsafe_types = (
                self.unsafe_content_detector.detect_unsafe_content(combined_content)
            )

            # Cost calculation
            cost = 0.0
            if prompt_tokens and completion_tokens and success:
                cost = self.cost_calculator.calculate_cost(
                    provider, model, prompt_tokens, completion_tokens
                )

            # Create metrics object
            metrics = LLMRequestMetrics(
                request_id=request_id,
                provider=provider,
                model=model,
                timestamp=current_time,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cost=cost,
                latency=latency,
                success=success,
                error_type=error_type,
                error_message=error_message,
                content_hash=content_hash,
                pii_detected=pii_detected,
                unsafe_content_detected=unsafe_detected,
            )

            # Store in recent requests
            self.recent_requests.append(metrics)

            # Record in metrics collector
            metric_data = {
                "request_id": request_id,
                "provider": provider,
                "model": model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "cost": cost,
                "latency": latency,
                "success": success,
                "error_type": error_type,
                "pii_detected": pii_detected,
                "unsafe_content_detected": unsafe_detected,
                "content_hash": content_hash,
            }

            self.metrics_collector.record_metric("llm_request", metric_data)

            # Log security issues
            if pii_detected:
                self.logger.warning(
                    f"PII detected in LLM request {request_id}: {', '.join(pii_types)}"
                )

            if unsafe_detected:
                self.logger.warning(
                    f"Unsafe content detected in LLM request {request_id}: {', '.join(unsafe_types)}"
                )

            # Check quotas and generate alerts
            current_usage = self.get_current_usage()
            quota_violations = self._check_quota_usage(current_usage)

            for violation in quota_violations:
                if self._should_send_alert(f"quota_{violation[:20]}"):
                    self.logger.error(
                        f"Quota violation: {violation}",
                        alert_type="quota_violation",
                        usage_stats=current_usage,
                    )

            # Check for consecutive failures
            if not success:
                recent_failures = sum(
                    1 for r in list(self.recent_requests)[-10:] if not r.success
                )
                if recent_failures >= self.alert_config.consecutive_failures_threshold:
                    if self._should_send_alert("consecutive_failures"):
                        self.logger.error(
                            f"High failure rate: {recent_failures} failures in last 10 requests",
                            alert_type="high_failure_rate",
                        )

            # Check for high latency
            if latency and latency > self.alert_config.high_latency_threshold:
                if self._should_send_alert("high_latency"):
                    self.logger.warning(
                        f"High latency detected: {latency:.2f}s for request {request_id}",
                        alert_type="high_latency",
                    )

            return metrics

    def get_current_usage(self) -> Dict[str, Any]:
        """
        Get current usage statistics across different time windows.

        Returns:
            Current usage statistics
        """
        current_time = time.time()

        # Time windows in seconds
        windows = {
            "minute": 60,
            "hour": 3600,
            "day": 86400,
            "week": 604800,
            "month": 2592000,
        }

        usage: Dict[str, Any] = {}

        with self._lock:
            for window_name, window_seconds in windows.items():
                cutoff_time = current_time - window_seconds

                # Filter requests in time window
                window_requests = [
                    r for r in self.recent_requests if r.timestamp >= cutoff_time
                ]

                # Calculate statistics
                total_requests = len(window_requests)
                successful_requests = sum(1 for r in window_requests if r.success)
                failed_requests = total_requests - successful_requests

                total_tokens = sum(r.total_tokens or 0 for r in window_requests)
                total_cost = sum(r.cost or 0.0 for r in window_requests)

                latencies = [
                    r.latency for r in window_requests if r.latency is not None
                ]
                avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

                pii_requests = sum(1 for r in window_requests if r.pii_detected)
                unsafe_requests = sum(
                    1 for r in window_requests if r.unsafe_content_detected
                )

                usage[f"requests_last_{window_name}"] = total_requests
                usage[f"successful_requests_last_{window_name}"] = successful_requests
                usage[f"failed_requests_last_{window_name}"] = failed_requests
                usage[f"tokens_last_{window_name}"] = total_tokens
                usage[f"cost_last_{window_name}"] = float(total_cost)
                usage[f"avg_latency_last_{window_name}"] = float(avg_latency)
                usage[f"pii_requests_last_{window_name}"] = pii_requests
                usage[f"unsafe_requests_last_{window_name}"] = unsafe_requests

                if total_requests > 0:
                    usage[f"success_rate_last_{window_name}"] = float(
                        successful_requests / total_requests
                    )
                    usage[f"pii_rate_last_{window_name}"] = float(
                        pii_requests / total_requests
                    )
                    usage[f"unsafe_rate_last_{window_name}"] = float(
                        unsafe_requests / total_requests
                    )
                else:
                    usage[f"success_rate_last_{window_name}"] = 1.0
                    usage[f"pii_rate_last_{window_name}"] = 0.0
                    usage[f"unsafe_rate_last_{window_name}"] = 0.0

        return usage

    def get_provider_performance(self, time_window: int = 3600) -> Dict[str, Any]:
        """
        Get provider performance statistics.

        Args:
            time_window: Time window in seconds (default: 1 hour)

        Returns:
            Provider performance statistics
        """
        current_time = time.time()
        cutoff_time = current_time - time_window

        provider_stats: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "total_tokens": 0,
                "total_cost": 0.0,
                "latencies": [],
                "errors": defaultdict(int),
            }
        )

        with self._lock:
            for request in self.recent_requests:
                if request.timestamp >= cutoff_time:
                    stats = provider_stats[request.provider]
                    stats["requests"] += 1

                    if request.success:
                        stats["successful_requests"] += 1
                    else:
                        stats["failed_requests"] += 1
                        if request.error_type:
                            stats["errors"][request.error_type] += 1

                    if request.total_tokens:
                        stats["total_tokens"] += request.total_tokens

                    if request.cost:
                        stats["total_cost"] += request.cost

                    if request.latency:
                        stats["latencies"].append(request.latency)

        # Calculate derived metrics
        result = {}
        for provider, stats in provider_stats.items():
            avg_latency = (
                sum(stats["latencies"]) / len(stats["latencies"])
                if stats["latencies"]
                else 0.0
            )
            success_rate = (
                stats["successful_requests"] / stats["requests"]
                if stats["requests"] > 0
                else 1.0
            )

            result[provider] = {
                "requests": stats["requests"],
                "success_rate": success_rate,
                "avg_latency": avg_latency,
                "total_tokens": stats["total_tokens"],
                "total_cost": stats["total_cost"],
                "error_breakdown": dict(stats["errors"]),
            }

        return result

    def generate_security_report(self, time_window: int = 86400) -> Dict[str, Any]:
        """
        Generate security report for PII and unsafe content detection.

        Args:
            time_window: Time window in seconds (default: 24 hours)

        Returns:
            Security report
        """
        current_time = time.time()
        cutoff_time = current_time - time_window

        with self._lock:
            recent_requests = [
                r for r in self.recent_requests if r.timestamp >= cutoff_time
            ]

        total_requests = len(recent_requests)
        pii_requests = [r for r in recent_requests if r.pii_detected]
        unsafe_requests = [r for r in recent_requests if r.unsafe_content_detected]

        return {
            "total_requests": total_requests,
            "pii_detections": {
                "count": len(pii_requests),
                "rate": (
                    len(pii_requests) / total_requests if total_requests > 0 else 0.0
                ),
                "by_provider": self._group_by_provider(pii_requests),
            },
            "unsafe_content_detections": {
                "count": len(unsafe_requests),
                "rate": (
                    len(unsafe_requests) / total_requests if total_requests > 0 else 0.0
                ),
                "by_provider": self._group_by_provider(unsafe_requests),
            },
            "recommendations": self._generate_security_recommendations(
                pii_requests, unsafe_requests
            ),
        }

    def _group_by_provider(self, requests: List[LLMRequestMetrics]) -> Dict[str, int]:
        """Group requests by provider."""
        provider_counts: Dict[str, int] = defaultdict(int)
        for request in requests:
            provider_counts[request.provider] += 1
        return dict(provider_counts)

    def _generate_security_recommendations(
        self,
        pii_requests: List[LLMRequestMetrics],
        unsafe_requests: List[LLMRequestMetrics],
    ) -> List[str]:
        """Generate security recommendations based on detected issues."""
        recommendations = []

        if pii_requests:
            recommendations.append(
                "Consider implementing stronger input sanitization to prevent PII in prompts"
            )
            recommendations.append(
                "Review data handling policies and user training on PII protection"
            )

        if unsafe_requests:
            recommendations.append(
                "Implement additional validation for prompt injection attempts"
            )
            recommendations.append(
                "Consider rate limiting for suspicious content patterns"
            )

        if len(pii_requests) > 5 or len(unsafe_requests) > 5:
            recommendations.append("Investigate potential systematic security issues")

        return recommendations

    def get_cost_breakdown(self, time_window: int = 86400) -> Dict[str, Any]:
        """
        Get detailed cost breakdown.

        Args:
            time_window: Time window in seconds (default: 24 hours)

        Returns:
            Cost breakdown analysis
        """
        current_time = time.time()
        cutoff_time = current_time - time_window

        with self._lock:
            recent_requests = [
                r
                for r in self.recent_requests
                if r.timestamp >= cutoff_time and r.cost is not None
            ]

        if not recent_requests:
            return {"total_cost": 0.0, "by_provider": {}, "by_model": {}}

        total_cost = sum(r.cost or 0.0 for r in recent_requests)

        # Group by provider
        provider_costs: Dict[str, float] = defaultdict(float)
        provider_tokens: Dict[str, int] = defaultdict(int)

        # Group by model
        model_costs: Dict[str, float] = defaultdict(float)
        model_tokens: Dict[str, int] = defaultdict(int)

        for request in recent_requests:
            provider_costs[request.provider] += request.cost or 0.0
            provider_tokens[request.provider] += request.total_tokens or 0

            model_key = f"{request.provider}/{request.model}"
            model_costs[model_key] += request.cost or 0.0
            model_tokens[model_key] += request.total_tokens or 0

        return {
            "total_cost": total_cost,
            "total_tokens": sum(r.total_tokens or 0 for r in recent_requests),
            "avg_cost_per_request": total_cost / len(recent_requests),
            "by_provider": {
                provider: {
                    "cost": cost,
                    "tokens": provider_tokens[provider],
                    "cost_per_token": (
                        cost / provider_tokens[provider]
                        if provider_tokens[provider] > 0
                        else 0
                    ),
                }
                for provider, cost in provider_costs.items()
            },
            "by_model": {
                model: {
                    "cost": cost,
                    "tokens": model_tokens[model],
                    "cost_per_token": (
                        cost / model_tokens[model] if model_tokens[model] > 0 else 0
                    ),
                }
                for model, cost in model_costs.items()
            },
        }

    def export_metrics(
        self, output_file: Path, time_window: Optional[int] = None
    ) -> None:
        """
        Export metrics to file.

        Args:
            output_file: Output file path
            time_window: Time window in seconds (None for all data)
        """
        current_time = time.time()

        with self._lock:
            if time_window:
                cutoff_time = current_time - time_window
                requests_to_export = [
                    r for r in self.recent_requests if r.timestamp >= cutoff_time
                ]
            else:
                requests_to_export = list(self.recent_requests)

        # Convert to serializable format
        export_data: Dict[str, Any] = {
            "export_timestamp": current_time,
            "time_window": time_window,
            "total_requests": len(requests_to_export),
            "requests": [],
        }

        for request in requests_to_export:
            export_data["requests"].append(
                {
                    "request_id": request.request_id,
                    "provider": request.provider,
                    "model": request.model,
                    "timestamp": request.timestamp,
                    "prompt_tokens": request.prompt_tokens,
                    "completion_tokens": request.completion_tokens,
                    "total_tokens": request.total_tokens,
                    "cost": request.cost,
                    "latency": request.latency,
                    "success": request.success,
                    "error_type": request.error_type,
                    "pii_detected": request.pii_detected,
                    "unsafe_content_detected": request.unsafe_content_detected,
                    "content_hash": request.content_hash,
                }
            )

        # Add aggregated statistics
        export_data["usage_statistics"] = self.get_current_usage()
        export_data["provider_performance"] = self.get_provider_performance(
            time_window or 86400
        )
        export_data["security_report"] = self.generate_security_report(
            time_window or 86400
        )
        export_data["cost_breakdown"] = self.get_cost_breakdown(time_window or 86400)

        with open(output_file, "w") as f:
            json.dump(export_data, f, indent=2)

        self.logger.info(
            f"Exported {len(requests_to_export)} LLM metrics to {output_file}"
        )
