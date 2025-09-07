"""
Cost tracking and quota management for LLM usage.

This module provides:
1. Real-time cost tracking across providers
2. Usage quota enforcement
3. Budget alerts and notifications
4. Cost optimization recommendations
"""

import json
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .logger import MonitoringLogger


class BudgetPeriod(Enum):
    """Budget period types."""

    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


@dataclass
class Budget:
    """Budget configuration for cost tracking."""

    amount: float
    period: BudgetPeriod
    alert_thresholds: List[float] = field(
        default_factory=lambda: [50.0, 75.0, 90.0, 100.0]
    )
    hard_limit: bool = False  # Whether to enforce hard limit
    rollover: bool = False  # Whether unused budget rolls over


@dataclass
class CostAlert:
    """Cost alert configuration."""

    budget_id: str
    threshold_percentage: float
    alert_type: str  # 'warning', 'critical', 'limit_reached'
    message: str
    timestamp: float
    current_usage: float
    budget_amount: float


@dataclass
class ProviderCostConfig:
    """Cost configuration for a specific provider."""

    provider: str
    daily_budget: Optional[float] = None
    monthly_budget: Optional[float] = None
    cost_per_token_limit: Optional[float] = None  # Max cost per token
    enabled: bool = True


class CostTracker:
    """Real-time cost tracking and budget management."""

    def __init__(
        self,
        storage_dir: Optional[Path] = None,
        budgets: Optional[List[Budget]] = None,
        provider_configs: Optional[Dict[str, ProviderCostConfig]] = None,
    ):
        """
        Initialize cost tracker.

        Args:
            storage_dir: Directory to store cost data
            budgets: List of budget configurations
            provider_configs: Provider-specific cost configurations
        """
        self.logger = MonitoringLogger("cost_tracker")

        # Storage
        self.storage_dir = storage_dir or Path("logs/cost_tracking")
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Budgets
        self.budgets = {f"budget_{i}": budget for i, budget in enumerate(budgets or [])}
        self.provider_configs = provider_configs or {}

        # Cost tracking
        self.cost_history = deque(maxlen=10000)  # Keep last 10k transactions
        self.budget_usage = defaultdict(float)  # Current usage per budget
        self.budget_periods = {}  # Track budget period start times

        # Alerts
        self.alert_callbacks: List[Callable[[CostAlert], None]] = []
        self.sent_alerts = defaultdict(set)  # Track sent alerts to avoid duplicates

        # Thread safety
        self._lock = threading.Lock()

        # Load existing data
        self._load_cost_data()

        self.logger.info(f"Initialized cost tracker with {len(self.budgets)} budgets")

    def add_budget(self, budget_id: str, budget: Budget) -> None:
        """Add a new budget."""
        with self._lock:
            self.budgets[budget_id] = budget
            self.budget_usage[budget_id] = 0.0
            self.budget_periods[budget_id] = time.time()

        self.logger.info(
            f"Added budget {budget_id}: ${budget.amount} per {budget.period.value}"
        )

    def remove_budget(self, budget_id: str) -> None:
        """Remove a budget."""
        with self._lock:
            if budget_id in self.budgets:
                del self.budgets[budget_id]
                del self.budget_usage[budget_id]
                if budget_id in self.budget_periods:
                    del self.budget_periods[budget_id]

                self.logger.info(f"Removed budget {budget_id}")

    def add_alert_callback(self, callback: Callable[[CostAlert], None]) -> None:
        """Add callback for cost alerts."""
        self.alert_callbacks.append(callback)

    def _load_cost_data(self) -> None:
        """Load existing cost data from storage."""
        try:
            cost_file = self.storage_dir / "cost_history.json"
            if cost_file.exists():
                with open(cost_file, "r") as f:
                    data = json.load(f)

                # Load recent transactions (last 7 days)
                cutoff_time = time.time() - (7 * 24 * 3600)
                recent_transactions = [
                    tx
                    for tx in data.get("transactions", [])
                    if tx["timestamp"] >= cutoff_time
                ]

                self.cost_history.extend(recent_transactions)
                self.logger.info(
                    f"Loaded {len(recent_transactions)} recent cost transactions"
                )

            # Load budget usage
            budget_file = self.storage_dir / "budget_usage.json"
            if budget_file.exists():
                with open(budget_file, "r") as f:
                    data = json.load(f)

                self.budget_usage.update(data.get("usage", {}))
                self.budget_periods.update(data.get("periods", {}))

                self.logger.info(
                    f"Loaded budget usage for {len(self.budget_usage)} budgets"
                )

        except Exception as e:
            self.logger.exception(e, message="Failed to load cost data")

    def _save_cost_data(self) -> None:
        """Save cost data to storage."""
        try:
            # Save cost history
            cost_file = self.storage_dir / "cost_history.json"
            with open(cost_file, "w") as f:
                json.dump(
                    {
                        "transactions": list(self.cost_history),
                        "last_updated": time.time(),
                    },
                    f,
                    indent=2,
                )

            # Save budget usage
            budget_file = self.storage_dir / "budget_usage.json"
            with open(budget_file, "w") as f:
                json.dump(
                    {
                        "usage": dict(self.budget_usage),
                        "periods": dict(self.budget_periods),
                        "last_updated": time.time(),
                    },
                    f,
                    indent=2,
                )

        except Exception as e:
            self.logger.exception(e, message="Failed to save cost data")

    def _get_period_duration(self, period: BudgetPeriod) -> float:
        """Get period duration in seconds."""
        durations = {
            BudgetPeriod.MINUTE: 60,
            BudgetPeriod.HOUR: 3600,
            BudgetPeriod.DAY: 86400,
            BudgetPeriod.WEEK: 604800,
            BudgetPeriod.MONTH: 2592000,  # Approximate 30 days
        }
        return durations[period]

    def _check_budget_period_reset(self, budget_id: str, budget: Budget) -> bool:
        """Check if budget period should be reset."""
        current_time = time.time()
        period_start = self.budget_periods.get(budget_id, current_time)
        period_duration = self._get_period_duration(budget.period)

        if current_time - period_start >= period_duration:
            # Reset budget period
            old_usage = self.budget_usage.get(budget_id, 0.0)

            if budget.rollover:
                # Calculate rollover amount
                rollover = max(0, budget.amount - old_usage)
                self.budget_usage[budget_id] = -rollover  # Negative for credit
            else:
                self.budget_usage[budget_id] = 0.0

            self.budget_periods[budget_id] = current_time

            self.logger.info(
                f"Reset budget period for {budget_id}. "
                f"Previous usage: ${old_usage:.4f}, "
                f"Rollover: ${-self.budget_usage[budget_id]:.4f}"
                if budget.rollover
                else "No rollover"
            )

            # Clear sent alerts for this budget
            if budget_id in self.sent_alerts:
                self.sent_alerts[budget_id].clear()

            return True

        return False

    def _check_budget_alerts(self, budget_id: str, budget: Budget, cost: float) -> None:
        """Check if budget alerts should be triggered."""
        current_usage = self.budget_usage.get(budget_id, 0.0) + cost
        usage_percentage = (current_usage / budget.amount) * 100.0

        for threshold in budget.alert_thresholds:
            if (
                usage_percentage >= threshold
                and threshold not in self.sent_alerts[budget_id]
            ):
                # Determine alert type
                if threshold >= 100.0:
                    alert_type = "limit_reached"
                elif threshold >= 90.0:
                    alert_type = "critical"
                else:
                    alert_type = "warning"

                # Create alert
                alert = CostAlert(
                    budget_id=budget_id,
                    threshold_percentage=threshold,
                    alert_type=alert_type,
                    message=f"Budget {budget_id} usage at {usage_percentage:.1f}% "
                    f"(${current_usage:.4f} of ${budget.amount:.4f})",
                    timestamp=time.time(),
                    current_usage=current_usage,
                    budget_amount=budget.amount,
                )

                # Send alert
                self._send_alert(alert)

                # Mark as sent
                self.sent_alerts[budget_id].add(threshold)

    def _send_alert(self, alert: CostAlert) -> None:
        """Send cost alert to registered callbacks."""
        self.logger.warning(
            f"Cost alert: {alert.message}",
            alert_type=alert.alert_type,
            budget_id=alert.budget_id,
            threshold=alert.threshold_percentage,
            current_usage=alert.current_usage,
            budget_amount=alert.budget_amount,
        )

        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.exception(e, message=f"Alert callback failed: {callback}")

    def record_cost(
        self,
        provider: str,
        model: str,
        cost: float,
        tokens: int,
        request_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Record a cost transaction.

        Args:
            provider: LLM provider name
            model: Model name
            cost: Cost in USD
            tokens: Number of tokens used
            request_id: Optional request ID
            metadata: Optional additional metadata

        Returns:
            True if transaction was allowed, False if blocked by budget limits
        """
        current_time = time.time()

        with self._lock:
            # Check provider configuration
            provider_config = self.provider_configs.get(provider)
            if provider_config and not provider_config.enabled:
                self.logger.warning(f"Cost tracking disabled for provider {provider}")
                return False

            # Check cost per token limit
            if provider_config and provider_config.cost_per_token_limit:
                cost_per_token = cost / tokens if tokens > 0 else 0
                if cost_per_token > provider_config.cost_per_token_limit:
                    self.logger.error(
                        f"Cost per token ({cost_per_token:.6f}) exceeds limit "
                        f"({provider_config.cost_per_token_limit:.6f}) for provider {provider}"
                    )
                    return False

            # Create transaction record
            transaction = {
                "timestamp": current_time,
                "provider": provider,
                "model": model,
                "cost": cost,
                "tokens": tokens,
                "request_id": request_id,
                "metadata": metadata or {},
            }

            # Check budgets and enforce limits
            blocked_by_budget = False

            for budget_id, budget in self.budgets.items():
                # Check if budget period needs reset
                self._check_budget_period_reset(budget_id, budget)

                # Calculate projected usage
                current_usage = self.budget_usage.get(budget_id, 0.0)
                projected_usage = current_usage + cost

                # Check hard limit
                if budget.hard_limit and projected_usage > budget.amount:
                    self.logger.error(
                        f"Transaction blocked by budget {budget_id}: "
                        f"${projected_usage:.4f} would exceed ${budget.amount:.4f}"
                    )
                    blocked_by_budget = True
                    continue

                # Check alerts
                self._check_budget_alerts(budget_id, budget, cost)

                # Update budget usage
                self.budget_usage[budget_id] = projected_usage

            if blocked_by_budget:
                return False

            # Record transaction
            self.cost_history.append(transaction)

            # Save data periodically
            if len(self.cost_history) % 100 == 0:
                self._save_cost_data()

            self.logger.info(
                f"Recorded cost: ${cost:.4f} for {provider}/{model} "
                f"({tokens} tokens, ${cost / tokens:.6f} per token)"
            )

            return True

    def get_current_usage(self, budget_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get current budget usage.

        Args:
            budget_id: Specific budget ID, or None for all budgets

        Returns:
            Current usage information
        """
        with self._lock:
            if budget_id:
                if budget_id not in self.budgets:
                    return {}

                budget = self.budgets[budget_id]
                usage = self.budget_usage.get(budget_id, 0.0)
                period_start = self.budget_periods.get(budget_id, time.time())

                return {
                    "budget_id": budget_id,
                    "amount": budget.amount,
                    "period": budget.period.value,
                    "current_usage": usage,
                    "remaining": budget.amount - usage,
                    "usage_percentage": (usage / budget.amount) * 100.0,
                    "period_start": period_start,
                    "time_remaining": self._get_period_duration(budget.period)
                    - (time.time() - period_start),
                }
            else:
                result = {}
                for bid, budget in self.budgets.items():
                    result[bid] = self.get_current_usage(bid)
                return result

    def get_cost_breakdown(
        self, time_window: int = 86400, group_by: str = "provider"
    ) -> Dict[str, Any]:
        """
        Get cost breakdown for specified time window.

        Args:
            time_window: Time window in seconds
            group_by: Group by 'provider', 'model', or 'day'

        Returns:
            Cost breakdown
        """
        current_time = time.time()
        cutoff_time = current_time - time_window

        with self._lock:
            recent_transactions = [
                tx for tx in self.cost_history if tx["timestamp"] >= cutoff_time
            ]

        if not recent_transactions:
            return {"total_cost": 0.0, "breakdown": {}}

        total_cost = sum(tx["cost"] for tx in recent_transactions)
        total_tokens = sum(tx["tokens"] for tx in recent_transactions)

        breakdown = defaultdict(lambda: {"cost": 0.0, "tokens": 0, "requests": 0})

        for tx in recent_transactions:
            if group_by == "provider":
                key = tx["provider"]
            elif group_by == "model":
                key = f"{tx['provider']}/{tx['model']}"
            elif group_by == "day":
                day = datetime.fromtimestamp(tx["timestamp"]).strftime("%Y-%m-%d")
                key = day
            else:
                key = "total"

            breakdown[key]["cost"] += tx["cost"]
            breakdown[key]["tokens"] += tx["tokens"]
            breakdown[key]["requests"] += 1

        # Calculate additional metrics
        for key, data in breakdown.items():
            if data["tokens"] > 0:
                data["cost_per_token"] = data["cost"] / data["tokens"]
            if data["requests"] > 0:
                data["avg_cost_per_request"] = data["cost"] / data["requests"]

        return {
            "total_cost": total_cost,
            "total_tokens": total_tokens,
            "total_requests": len(recent_transactions),
            "avg_cost_per_token": total_cost / total_tokens if total_tokens > 0 else 0,
            "breakdown": dict(breakdown),
        }

    def get_cost_trends(self, days: int = 7) -> Dict[str, Any]:
        """
        Get cost trends over specified number of days.

        Args:
            days: Number of days to analyze

        Returns:
            Cost trend analysis
        """
        current_time = time.time()
        cutoff_time = current_time - (days * 86400)

        with self._lock:
            recent_transactions = [
                tx for tx in self.cost_history if tx["timestamp"] >= cutoff_time
            ]

        # Group by day
        daily_costs = defaultdict(float)
        daily_tokens = defaultdict(int)
        daily_requests = defaultdict(int)

        for tx in recent_transactions:
            day = datetime.fromtimestamp(tx["timestamp"]).strftime("%Y-%m-%d")
            daily_costs[day] += tx["cost"]
            daily_tokens[day] += tx["tokens"]
            daily_requests[day] += 1

        # Calculate trends
        costs = list(daily_costs.values())
        tokens = list(daily_tokens.values())

        if len(costs) > 1:
            cost_trend = (costs[-1] - costs[0]) / len(costs) if costs[0] > 0 else 0
            token_trend = (tokens[-1] - tokens[0]) / len(tokens) if tokens[0] > 0 else 0
        else:
            cost_trend = 0
            token_trend = 0

        return {
            "daily_breakdown": {
                "costs": dict(daily_costs),
                "tokens": dict(daily_tokens),
                "requests": dict(daily_requests),
            },
            "trends": {
                "cost_trend": cost_trend,
                "token_trend": token_trend,
                "avg_daily_cost": sum(costs) / len(costs) if costs else 0,
                "avg_daily_tokens": sum(tokens) / len(tokens) if tokens else 0,
            },
            "projections": {
                "monthly_cost": sum(costs) * 30 / days if costs else 0,
                "monthly_tokens": sum(tokens) * 30 / days if tokens else 0,
            },
        }

    def get_optimization_recommendations(self) -> List[str]:
        """
        Get cost optimization recommendations based on usage patterns.

        Returns:
            List of optimization recommendations
        """
        recommendations = []

        # Analyze recent usage
        breakdown = self.get_cost_breakdown(time_window=86400 * 7)  # Last week
        trends = self.get_cost_trends(days=7)

        if not breakdown["breakdown"]:
            return ["No recent usage data available for analysis"]

        # Provider cost efficiency
        provider_costs = {
            k: v
            for k, v in breakdown["breakdown"].items()
            if isinstance(v, dict) and "cost_per_token" in v
        }

        if len(provider_costs) > 1:
            sorted_providers = sorted(
                provider_costs.items(), key=lambda x: x[1]["cost_per_token"]
            )

            cheapest = sorted_providers[0]
            most_expensive = sorted_providers[-1]

            if (
                most_expensive[1]["cost_per_token"]
                > cheapest[1]["cost_per_token"] * 1.5
            ):
                recommendations.append(
                    f"Consider using {cheapest[0]} more often - it's {most_expensive[1]['cost_per_token'] / cheapest[1]['cost_per_token']:.1f}x cheaper per token than {most_expensive[0]}"
                )

        # High cost alerts
        weekly_cost = breakdown["total_cost"]
        if weekly_cost > 50:  # Arbitrary threshold
            recommendations.append(
                f"Weekly cost of ${weekly_cost:.2f} is relatively high. Consider setting stricter budgets or usage limits."
            )

        # Usage pattern analysis
        if trends["trends"]["cost_trend"] > 0:
            recommendations.append(
                "Cost trend is increasing. Monitor usage closely and consider implementing cost controls."
            )

        # Token efficiency
        avg_cost_per_token = breakdown["avg_cost_per_token"]
        if avg_cost_per_token > 0.0001:  # High cost per token
            recommendations.append(
                f"Average cost per token (${avg_cost_per_token:.6f}) is high. Consider using more efficient models or providers."
            )

        # Budget utilization
        budget_usage = self.get_current_usage()
        for budget_id, usage_info in budget_usage.items():
            if isinstance(usage_info, dict):
                if usage_info["usage_percentage"] > 80:
                    recommendations.append(
                        f"Budget {budget_id} is {usage_info['usage_percentage']:.1f}% utilized. Consider increasing budget or optimizing usage."
                    )

        return recommendations or [
            "No specific optimization recommendations at this time"
        ]

    def export_cost_report(self, output_file: Path, days: int = 30) -> None:
        """
        Export comprehensive cost report.

        Args:
            output_file: Output file path
            days: Number of days to include in report
        """
        report_data = {
            "report_timestamp": time.time(),
            "report_period_days": days,
            "budget_usage": self.get_current_usage(),
            "cost_breakdown": self.get_cost_breakdown(time_window=days * 86400),
            "cost_trends": self.get_cost_trends(days=days),
            "optimization_recommendations": self.get_optimization_recommendations(),
            "budget_configurations": {
                bid: {
                    "amount": budget.amount,
                    "period": budget.period.value,
                    "alert_thresholds": budget.alert_thresholds,
                    "hard_limit": budget.hard_limit,
                    "rollover": budget.rollover,
                }
                for bid, budget in self.budgets.items()
            },
        }

        with open(output_file, "w") as f:
            json.dump(report_data, f, indent=2)

        self.logger.info(f"Exported cost report to {output_file}")


# Default alert callback that logs to the monitoring system
def default_alert_callback(alert: CostAlert) -> None:
    """Default alert callback that logs alerts."""
    logger = MonitoringLogger("cost_alerts")
    logger.warning(
        f"Cost Alert: {alert.message}",
        budget_id=alert.budget_id,
        alert_type=alert.alert_type,
        threshold=alert.threshold_percentage,
        current_usage=alert.current_usage,
        budget_amount=alert.budget_amount,
    )
