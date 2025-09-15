"""
Hybrid modern/legacy architecture orchestrator.

This module manages the coordination between modern microservices and legacy
systems during the modernization transition period.
"""

import asyncio
import json
import logging
import queue
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, cast

logger = logging.getLogger(__name__)


class SystemType(Enum):
    """System architecture types."""

    MAINFRAME = "mainframe"
    MONOLITH = "monolith"
    MICROSERVICE = "microservice"
    SERVERLESS = "serverless"
    LEGACY_API = "legacy_api"
    MODERN_API = "modern_api"


class RoutingStrategy(Enum):
    """Traffic routing strategies."""

    CANARY = "canary"
    BLUE_GREEN = "blue_green"
    GRADUAL = "gradual"
    SHADOW = "shadow"
    PERCENTAGE = "percentage"


class TransactionState(Enum):
    """Transaction processing states."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLBACK = "rollback"
    COMPENSATED = "compensated"


@dataclass
class SystemEndpoint:
    """Represents a system endpoint."""

    name: str
    type: SystemType
    host: str
    port: int
    protocol: str  # HTTP, MQ, FTP, etc.
    health_check: Optional[str] = None
    timeout: int = 30
    retry_count: int = 3
    circuit_breaker_threshold: int = 5
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HybridTransaction:
    """Represents a transaction spanning multiple systems."""

    transaction_id: str
    timestamp: datetime
    source_system: str
    target_systems: List[str]
    payload: Dict[str, Any]
    state: TransactionState
    results: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    compensation_actions: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class RoutingRule:
    """Traffic routing rule."""

    name: str
    source_pattern: str
    strategy: RoutingStrategy
    legacy_weight: float  # 0.0 to 1.0
    modern_weight: float  # 0.0 to 1.0
    conditions: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class HybridOrchestrator:
    """
    Orchestrates interactions between modern and legacy systems.

    Manages traffic routing, transaction coordination, data synchronization,
    and gradual migration from legacy to modern systems.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.systems: Dict[str, SystemEndpoint] = {}
        self.routing_rules: List[RoutingRule] = []
        self.active_transactions: Dict[str, HybridTransaction] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.data_sync_queue: queue.Queue[Dict[str, Any]] = queue.Queue()
        self.metrics = MetricsCollector()
        self._executor = ThreadPoolExecutor(max_workers=10)
        self._running = False

    def register_system(self, endpoint: SystemEndpoint):
        """Register a system endpoint."""
        self.systems[endpoint.name] = endpoint
        self.circuit_breakers[endpoint.name] = CircuitBreaker(
            endpoint.name, threshold=endpoint.circuit_breaker_threshold
        )
        logger.info(f"Registered system: {endpoint.name} ({endpoint.type.value})")

    def add_routing_rule(self, rule: RoutingRule):
        """Add traffic routing rule."""
        self.routing_rules.append(rule)
        logger.info(f"Added routing rule: {rule.name}")

    def start(self):
        """Start the orchestrator."""
        self._running = True

        # Start background tasks
        self._executor.submit(self._health_check_loop)
        self._executor.submit(self._data_sync_loop)
        self._executor.submit(self._metrics_collection_loop)

        logger.info("Hybrid orchestrator started")

    def stop(self):
        """Stop the orchestrator."""
        self._running = False
        self._executor.shutdown(wait=True)
        logger.info("Hybrid orchestrator stopped")

    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming request through hybrid architecture."""
        transaction = HybridTransaction(
            transaction_id=self._generate_transaction_id(),
            timestamp=datetime.now(),
            source_system=request.get("source", "unknown"),
            target_systems=[],
            payload=request,
            state=TransactionState.PENDING,
        )

        self.active_transactions[transaction.transaction_id] = transaction

        try:
            # Determine routing
            route = self._determine_route(request)
            transaction.target_systems = route["targets"]

            # Execute transaction
            transaction.state = TransactionState.PROCESSING

            if route["strategy"] == RoutingStrategy.SHADOW:
                # Shadow mode - send to both but only return legacy response
                results = await self._execute_shadow(transaction, route)
            elif route["strategy"] == RoutingStrategy.CANARY:
                # Canary mode - route percentage to modern
                results = await self._execute_canary(transaction, route)
            elif route["strategy"] == RoutingStrategy.BLUE_GREEN:
                # Blue-green - all traffic to one system
                results = await self._execute_blue_green(transaction, route)
            else:
                # Default percentage-based routing
                results = await self._execute_percentage(transaction, route)

            transaction.results = results
            transaction.state = TransactionState.COMPLETED

            # Record metrics
            self.metrics.record_transaction(transaction)

            return {
                "transaction_id": transaction.transaction_id,
                "status": "success",
                "results": results,
            }

        except Exception as e:
            logger.error(f"Transaction {transaction.transaction_id} failed: {e}")
            transaction.state = TransactionState.FAILED
            transaction.errors.append(str(e))

            # Attempt compensation if needed
            if transaction.compensation_actions:
                await self._compensate_transaction(transaction)

            return {
                "transaction_id": transaction.transaction_id,
                "status": "failed",
                "error": str(e),
            }

        finally:
            # Clean up completed transactions after delay
            self._executor.submit(
                self._cleanup_transaction,
                transaction.transaction_id,
                delay=300,  # 5 minutes
            )

    def _determine_route(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Determine routing based on rules."""
        for rule in self.routing_rules:
            if self._matches_pattern(request, rule.source_pattern):
                if self._check_conditions(request, rule.conditions):
                    return {
                        "strategy": rule.strategy,
                        "targets": self._select_targets(rule),
                        "weights": {
                            "legacy": rule.legacy_weight,
                            "modern": rule.modern_weight,
                        },
                    }

        # Default routing
        return {
            "strategy": RoutingStrategy.PERCENTAGE,
            "targets": ["legacy_system"],
            "weights": {"legacy": 1.0, "modern": 0.0},
        }

    def _matches_pattern(self, request: Dict[str, Any], pattern: str) -> bool:
        """Check if request matches pattern."""
        # Simple pattern matching - could be enhanced
        if pattern == "*":
            return True

        if "operation" in request:
            return bool(request["operation"] == pattern)

        if "path" in request:
            return bool(request["path"].startswith(pattern))

        return False

    def _check_conditions(
        self, request: Dict[str, Any], conditions: Dict[str, Any]
    ) -> bool:
        """Check routing conditions."""
        for key, value in conditions.items():
            if key == "user_type" and request.get("user_type") != value:
                return False
            elif key == "data_size" and request.get("size", 0) > value:
                return False
            elif key == "time_range":
                current_hour = datetime.now().hour
                if not (value["start"] <= current_hour < value["end"]):
                    return False

        return True

    def _select_targets(self, rule: RoutingRule) -> List[str]:
        """Select target systems based on rule."""
        targets = []

        if rule.legacy_weight > 0:
            # Find legacy systems
            legacy_systems = [
                name
                for name, system in self.systems.items()
                if system.type
                in [SystemType.MAINFRAME, SystemType.MONOLITH, SystemType.LEGACY_API]
            ]
            if legacy_systems:
                targets.extend(legacy_systems)

        if rule.modern_weight > 0:
            # Find modern systems
            modern_systems = [
                name
                for name, system in self.systems.items()
                if system.type
                in [
                    SystemType.MICROSERVICE,
                    SystemType.SERVERLESS,
                    SystemType.MODERN_API,
                ]
            ]
            if modern_systems:
                targets.extend(modern_systems)

        return targets

    async def _execute_shadow(
        self, transaction: HybridTransaction, route: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute in shadow mode - send to both but return legacy only."""
        tasks = []

        # Send to legacy (primary)
        legacy_system = self._get_legacy_system(route["targets"])
        if legacy_system:
            legacy_task = asyncio.create_task(
                self._send_to_system(transaction, legacy_system, is_primary=True)
            )
            tasks.append(("legacy", legacy_task))

        # Send to modern (shadow)
        modern_system = self._get_modern_system(route["targets"])
        if modern_system:
            modern_task = asyncio.create_task(
                self._send_to_system(transaction, modern_system, is_primary=False)
            )
            tasks.append(("modern", modern_task))

        # Wait for all to complete
        results = {}
        for name, task in tasks:
            try:
                results[name] = await task
            except Exception as e:
                results[name] = {"error": str(e)}

        # Compare results for monitoring
        if "legacy" in results and "modern" in results:
            self._compare_results(results["legacy"], results["modern"], transaction)

        # Return only legacy result
        return results.get("legacy", {"error": "Legacy system unavailable"})

    async def _execute_canary(
        self, transaction: HybridTransaction, route: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute canary deployment - route percentage to modern."""
        import random

        # Determine which system to use based on weights
        use_modern = random.random() < route["weights"]["modern"]

        if use_modern:
            system = self._get_modern_system(route["targets"])
            transaction.metadata["routed_to"] = "modern"
        else:
            system = self._get_legacy_system(route["targets"])
            transaction.metadata["routed_to"] = "legacy"

        if not system:
            raise ValueError(f"No system found for targets: {route['targets']}")

        return await self._send_to_system(transaction, system, is_primary=True)

    async def _execute_blue_green(
        self, transaction: HybridTransaction, route: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute blue-green deployment."""
        # Check which environment is active
        active_env = self.config.get("active_environment", "blue")

        if active_env == "blue":
            system = self._get_legacy_system(route["targets"])
        else:
            system = self._get_modern_system(route["targets"])

        if not system:
            raise ValueError(f"No system found for targets: {route['targets']}")

        return await self._send_to_system(transaction, system, is_primary=True)

    async def _execute_percentage(
        self, transaction: HybridTransaction, route: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute percentage-based routing."""
        import random

        # Similar to canary but with explicit percentage control
        use_modern = random.random() < route["weights"]["modern"]

        if use_modern and self._get_modern_system(route["targets"]):
            system = self._get_modern_system(route["targets"])
        else:
            system = self._get_legacy_system(route["targets"])

        if not system:
            raise ValueError(f"No system found for targets: {route['targets']}")

        return await self._send_to_system(transaction, system, is_primary=True)

    def _get_legacy_system(self, targets: List[str]) -> Optional[str]:
        """Get legacy system from targets."""
        for target in targets:
            if target in self.systems:
                system = self.systems[target]
                if system.type in [
                    SystemType.MAINFRAME,
                    SystemType.MONOLITH,
                    SystemType.LEGACY_API,
                ]:
                    return target
        return None

    def _get_modern_system(self, targets: List[str]) -> Optional[str]:
        """Get modern system from targets."""
        for target in targets:
            if target in self.systems:
                system = self.systems[target]
                if system.type in [
                    SystemType.MICROSERVICE,
                    SystemType.SERVERLESS,
                    SystemType.MODERN_API,
                ]:
                    return target
        return None

    async def _send_to_system(
        self, transaction: HybridTransaction, system_name: str, is_primary: bool
    ) -> Dict[str, Any]:
        """Send request to specific system."""
        if not system_name or system_name not in self.systems:
            raise ValueError(f"System {system_name} not found")

        system = self.systems[system_name]

        # Check circuit breaker
        if not self.circuit_breakers[system_name].is_available():
            raise Exception(f"Circuit breaker open for {system_name}")

        try:
            # Adapt request format based on system type
            adapted_request = self._adapt_request(transaction.payload, system)

            # Send request based on protocol
            if system.protocol.upper() == "HTTP":
                result = await self._http_request(system, adapted_request)
            elif system.protocol.upper() == "MQ":
                result = await self._mq_request(system, adapted_request)
            elif system.protocol.upper() == "FTP":
                result = await self._ftp_request(system, adapted_request)
            else:
                raise ValueError(f"Unsupported protocol: {system.protocol}")

            # Adapt response format
            adapted_result = self._adapt_response(result, system)

            # Record success
            self.circuit_breakers[system_name].record_success()

            # Add compensation action if this is a state-changing operation
            if is_primary and self._is_state_changing(transaction.payload):
                transaction.compensation_actions.append(
                    {
                        "system": system_name,
                        "action": self._create_compensation_action(
                            adapted_request, system
                        ),
                    }
                )

            return adapted_result

        except Exception:
            # Record failure
            self.circuit_breakers[system_name].record_failure()
            raise

    def _adapt_request(self, request: Dict[str, Any], system: SystemEndpoint) -> Any:
        """Adapt request format for target system."""
        if system.type == SystemType.MAINFRAME:
            # Convert to mainframe format (e.g., COBOL copybook)
            return self._convert_to_copybook(request)
        elif system.type == SystemType.LEGACY_API:
            # Convert to legacy API format (e.g., SOAP)
            return self._convert_to_soap(request)
        elif system.type in [SystemType.MICROSERVICE, SystemType.MODERN_API]:
            # Already in modern format (JSON)
            return request
        else:
            return request

    def _adapt_response(self, response: Any, system: SystemEndpoint) -> Dict[str, Any]:
        """Adapt response format from target system."""
        if system.type == SystemType.MAINFRAME:
            # Convert from mainframe format
            return self._convert_from_copybook(response)
        elif system.type == SystemType.LEGACY_API:
            # Convert from legacy API format
            return self._convert_from_soap(response)
        elif system.type in [SystemType.MICROSERVICE, SystemType.MODERN_API]:
            # Already in modern format
            return cast(Dict[str, Any], response)
        else:
            return cast(Dict[str, Any], response)

    def _convert_to_copybook(self, request: Dict[str, Any]) -> str:
        """Convert JSON to COBOL copybook format."""
        # Simplified example - real implementation would use proper copybook definitions
        copybook = ""

        # Fixed-width format
        if "customer_id" in request:
            copybook += str(request["customer_id"]).ljust(10)
        if "transaction_type" in request:
            copybook += str(request["transaction_type"]).ljust(2)
        if "amount" in request:
            amount_str = f"{request['amount']:.2f}".replace(".", "")
            copybook += amount_str.rjust(15, "0")

        return copybook

    def _convert_from_copybook(self, response: str) -> Dict[str, Any]:
        """Convert COBOL copybook format to JSON."""
        # Simplified example
        result = {}

        if len(response) >= 10:
            result["customer_id"] = response[0:10].strip()
        if len(response) >= 12:
            result["status_code"] = response[10:12].strip()
        if len(response) >= 27:
            amount_str = response[12:27]
            result["amount"] = float(amount_str) / 100

        return result

    def _convert_to_soap(self, request: Dict[str, Any]) -> str:
        """Convert JSON to SOAP XML."""
        # Simplified SOAP envelope
        soap = """<?xml version="1.0" encoding="UTF-8"?>
<soap:Envelope xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">
    <soap:Body>
        <ProcessRequest>"""

        for key, value in request.items():
            soap += f"\n            <{key}>{value}</{key}>"

        soap += """
        </ProcessRequest>
    </soap:Body>
</soap:Envelope>"""

        return soap

    def _convert_from_soap(self, response: str) -> Dict[str, Any]:
        """Convert SOAP XML to JSON."""
        # Simplified parsing - real implementation would use proper XML parsing
        import re

        result = {}

        # Extract values between tags
        patterns = {
            "status": r"<status>([^<]+)</status>",
            "message": r"<message>([^<]+)</message>",
            "result": r"<result>([^<]+)</result>",
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, response)
            if match:
                result[key] = match.group(1)

        return result

    async def _http_request(self, system: SystemEndpoint, request: Any) -> Any:
        """Make HTTP request to system."""
        # Simplified HTTP client - real implementation would use aiohttp or similar
        import aiohttp

        url = f"{system.protocol}://{system.host}:{system.port}/api/process"

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=request if isinstance(request, dict) else None,
                data=request if isinstance(request, str) else None,
                timeout=aiohttp.ClientTimeout(total=system.timeout),
            ) as response:
                if response.content_type == "application/json":
                    return await response.json()
                else:
                    return await response.text()

    async def _mq_request(self, system: SystemEndpoint, request: Any) -> Any:
        """Send message to message queue."""
        # Simplified MQ client - real implementation would use proper MQ library
        # This would integrate with IBM MQ, RabbitMQ, etc.

        # Simulate MQ request/response
        await asyncio.sleep(0.1)  # Simulate network delay

        return {
            "status": "processed",
            "correlation_id": self._generate_transaction_id(),
            "timestamp": datetime.now().isoformat(),
        }

    async def _ftp_request(self, system: SystemEndpoint, request: Any) -> Any:
        """Transfer file via FTP."""
        # Simplified FTP client - real implementation would use aioftp or similar

        # Simulate file transfer
        await asyncio.sleep(0.5)  # Simulate transfer delay

        return {
            "status": "transferred",
            "filename": f"TRANS.{datetime.now().strftime('%Y%m%d.%H%M%S')}.DAT",
            "size": len(str(request)),
        }

    def _is_state_changing(self, request: Dict[str, Any]) -> bool:
        """Check if request changes system state."""
        # Check for operations that modify data
        operation = request.get("operation", "").lower()
        method = request.get("method", "").upper()

        state_changing_ops = ["create", "update", "delete", "insert", "modify"]
        state_changing_methods = ["POST", "PUT", "DELETE", "PATCH"]

        return operation in state_changing_ops or method in state_changing_methods

    def _create_compensation_action(
        self, request: Any, system: SystemEndpoint
    ) -> Dict[str, Any]:
        """Create compensation action for rollback."""
        # Determine inverse operation
        operation = request.get("operation", "") if isinstance(request, dict) else ""

        compensation_map = {
            "create": "delete",
            "update": "restore",
            "delete": "recreate",
            "insert": "remove",
        }

        inverse_op = compensation_map.get(operation, "rollback")

        return {
            "operation": inverse_op,
            "original_request": request,
            "system_type": system.type.value,
            "timestamp": datetime.now().isoformat(),
        }

    async def _compensate_transaction(self, transaction: HybridTransaction):
        """Execute compensation actions for failed transaction."""
        logger.info(f"Compensating transaction {transaction.transaction_id}")

        for action in reversed(transaction.compensation_actions):
            try:
                system_name = action["system"]
                if system_name in self.systems:
                    # Execute compensation
                    await self._send_to_system(
                        transaction, system_name, is_primary=False
                    )

            except Exception as e:
                logger.error(f"Compensation failed for {system_name}: {e}")

        transaction.state = TransactionState.COMPENSATED

    def _compare_results(
        self, legacy_result: Any, modern_result: Any, transaction: HybridTransaction
    ):
        """Compare results from legacy and modern systems."""
        # Record comparison for analysis
        comparison = {
            "transaction_id": transaction.transaction_id,
            "timestamp": datetime.now().isoformat(),
            "legacy_result": legacy_result,
            "modern_result": modern_result,
            "match": legacy_result == modern_result,
        }

        if not comparison["match"]:
            # Log discrepancy for investigation
            logger.warning(
                f"Result mismatch for transaction {transaction.transaction_id}"
            )
            self.metrics.record_mismatch(comparison)

        # Queue for detailed analysis
        self.data_sync_queue.put(comparison)

    def _generate_transaction_id(self) -> str:
        """Generate unique transaction ID."""
        import uuid

        return f"TXN-{datetime.now().strftime('%Y%m%d%H%M%S')}-{str(uuid.uuid4())[:8]}"

    def _health_check_loop(self):
        """Background task for health checking."""
        while self._running:
            for name, system in self.systems.items():
                if system.health_check:
                    try:
                        # Perform health check
                        self._check_system_health(system)
                    except Exception as e:
                        logger.error(f"Health check failed for {name}: {e}")
                        self.circuit_breakers[name].record_failure()

            time.sleep(30)  # Check every 30 seconds

    def _check_system_health(self, system: SystemEndpoint):
        """Check health of a system."""
        # Simplified health check - real implementation would make actual requests
        if system.protocol.upper() == "HTTP":
            # Would make HTTP GET to health endpoint
            pass
        elif system.protocol.upper() == "MQ":
            # Would check MQ connection
            pass

    def _data_sync_loop(self):
        """Background task for data synchronization."""
        while self._running:
            try:
                # Process sync queue
                while not self.data_sync_queue.empty():
                    comparison = self.data_sync_queue.get_nowait()
                    self._analyze_discrepancy(comparison)

            except queue.Empty:
                pass
            except Exception as e:
                logger.error(f"Data sync error: {e}")

            time.sleep(5)

    def _analyze_discrepancy(self, comparison: Dict[str, Any]):
        """Analyze discrepancy between systems."""
        # Determine type of discrepancy
        legacy = comparison["legacy_result"]
        modern = comparison["modern_result"]

        # Check for data format differences
        if isinstance(legacy, dict) and isinstance(modern, dict):
            missing_in_modern = set(legacy.keys()) - set(modern.keys())
            missing_in_legacy = set(modern.keys()) - set(legacy.keys())

            if missing_in_modern:
                logger.info(f"Fields missing in modern: {missing_in_modern}")
            if missing_in_legacy:
                logger.info(f"Fields missing in legacy: {missing_in_legacy}")

    def _metrics_collection_loop(self):
        """Background task for metrics collection."""
        while self._running:
            # Collect and publish metrics
            metrics = self.metrics.get_metrics()

            # Log summary
            logger.info(f"Metrics: {json.dumps(metrics, indent=2)}")

            # Would publish to monitoring system

            time.sleep(60)  # Collect every minute

    def _cleanup_transaction(self, transaction_id: str, delay: int):
        """Clean up completed transaction after delay."""
        time.sleep(delay)

        if transaction_id in self.active_transactions:
            del self.active_transactions[transaction_id]


class CircuitBreaker:
    """Circuit breaker for system resilience."""

    def __init__(self, name: str, threshold: int = 5, timeout: int = 60):
        self.name = name
        self.threshold = threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "closed"  # closed, open, half-open

    def is_available(self) -> bool:
        """Check if circuit breaker allows requests."""
        if self.state == "closed":
            return True

        if self.state == "open":
            # Check if timeout has passed
            if not self.last_failure_time:
                return False
            elapsed = (datetime.now() - self.last_failure_time).seconds
            if elapsed > self.timeout:
                self.state = "half-open"
                return True
            return False

        return self.state == "half-open"

    def record_success(self):
        """Record successful request."""
        if self.state == "half-open":
            self.state = "closed"
            self.failure_count = 0

    def record_failure(self):
        """Record failed request."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if self.failure_count >= self.threshold:
            self.state = "open"
            logger.warning(f"Circuit breaker opened for {self.name}")


class MetricsCollector:
    """Collects and aggregates metrics."""

    def __init__(self):
        self.transactions = []
        self.mismatches = []
        self.system_metrics = {}

    def record_transaction(self, transaction: HybridTransaction):
        """Record transaction metrics."""
        self.transactions.append(
            {
                "transaction_id": transaction.transaction_id,
                "timestamp": transaction.timestamp,
                "duration": (datetime.now() - transaction.timestamp).total_seconds(),
                "state": transaction.state.value,
                "target_systems": transaction.target_systems,
                "errors": len(transaction.errors),
            }
        )

    def record_mismatch(self, comparison: Dict[str, Any]):
        """Record result mismatch."""
        self.mismatches.append(comparison)

    def get_metrics(self) -> Dict[str, Any]:
        """Get aggregated metrics."""
        if not self.transactions:
            return {}

        total = len(self.transactions)
        successful = len([t for t in self.transactions if t["state"] == "completed"])
        failed = len([t for t in self.transactions if t["state"] == "failed"])

        avg_duration = sum(t["duration"] for t in self.transactions) / total

        return {
            "total_transactions": total,
            "successful": successful,
            "failed": failed,
            "success_rate": (successful / total) * 100 if total > 0 else 0,
            "average_duration": avg_duration,
            "mismatches": len(self.mismatches),
            "mismatch_rate": (len(self.mismatches) / total) * 100 if total > 0 else 0,
        }
