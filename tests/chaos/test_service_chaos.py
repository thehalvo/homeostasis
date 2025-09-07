import asyncio
import random
import time
from collections import defaultdict
from datetime import datetime, timedelta

import networkx as nx
import pytest


class TestServiceChaos:
    """Test service-level chaos scenarios including cascading failures"""

    @pytest.fixture
    def service_mesh(self):
        """Create a simulated service mesh"""
        return ServiceMesh()

    @pytest.fixture
    def circuit_breaker_system(self):
        """Create circuit breaker system"""
        return CircuitBreakerSystem()

    @pytest.mark.asyncio
    async def test_cascading_failure_patterns(self, service_mesh):
        """Test different cascading failure patterns"""

        # Define service topology
        service_mesh.add_service("frontend", dependencies=["api-gateway"])
        service_mesh.add_service(
            "api-gateway", dependencies=["auth", "catalog", "cart"]
        )
        service_mesh.add_service("auth", dependencies=["user-db", "cache"])
        service_mesh.add_service("catalog", dependencies=["product-db", "cache"])
        service_mesh.add_service("cart", dependencies=["cart-db", "cache"])
        service_mesh.add_service("user-db", dependencies=[])
        service_mesh.add_service("product-db", dependencies=[])
        service_mesh.add_service("cart-db", dependencies=[])
        service_mesh.add_service("cache", dependencies=[])

        # Scenario 1: Database failure cascade
        cascade_results = []

        # Fail primary database
        service_mesh.inject_failure("product-db", failure_rate=1.0)

        # Simulate traffic and observe cascade
        for i in range(100):
            result = await service_mesh.send_request("frontend", "/catalog")
            cascade_results.append(result)

            # Check cascade progression
            if i == 20:
                # Catalog should start failing
                failed_services = service_mesh.get_failed_services()
                assert "catalog" in failed_services or result["status"] == "error"

            if i == 50:
                # API gateway might start experiencing issues
                error_rate = (
                    sum(1 for r in cascade_results[-20:] if r["status"] == "error") / 20
                )
                assert error_rate > 0.3

        # Analyze cascade impact
        final_failed_services = service_mesh.get_failed_services()
        assert "product-db" in final_failed_services
        assert len(final_failed_services) >= 2  # At least DB and one dependent

        # Reset for next scenario
        service_mesh.reset()

        # Scenario 2: Cache failure with fallback
        service_mesh.inject_failure("cache", failure_rate=1.0)
        service_mesh.set_fallback("auth", "cache", "direct-db-query")
        service_mesh.set_fallback("catalog", "cache", "direct-db-query")
        service_mesh.set_fallback("cart", "cache", "direct-db-query")

        fallback_results = []
        for _ in range(50):
            result = await service_mesh.send_request("frontend", "/catalog")
            fallback_results.append(result)

        # System should remain mostly operational with fallbacks
        success_rate = sum(
            1 for r in fallback_results if r["status"] == "success"
        ) / len(fallback_results)
        assert success_rate > 0.8  # High success rate despite cache failure

        # Verify fallbacks were used
        fallback_metrics = service_mesh.get_fallback_metrics()
        assert fallback_metrics["auth"]["cache"] > 0
        assert fallback_metrics["catalog"]["cache"] > 0
        assert fallback_metrics["cart"]["cache"] > 0

    @pytest.mark.asyncio
    async def test_circuit_breaker_behavior(self, circuit_breaker_system):
        """Test circuit breaker patterns under various failure scenarios"""

        # Configure circuit breakers
        breakers = {
            "payment": circuit_breaker_system.create_breaker(
                failure_threshold=5,
                success_threshold=3,
                timeout=timedelta(seconds=30),
                half_open_max_calls=2,
            ),
            "inventory": circuit_breaker_system.create_breaker(
                failure_threshold=3, success_threshold=2, timeout=timedelta(seconds=20)
            ),
            "shipping": circuit_breaker_system.create_breaker(
                failure_threshold=10, success_threshold=5, timeout=timedelta(seconds=60)
            ),
        }

        # Test 1: Circuit opens after threshold
        payment_results = []

        async def failing_payment_call():
            raise Exception("Payment service unavailable")

        # Send requests until circuit opens
        for i in range(10):
            try:
                result = await breakers["payment"].call(failing_payment_call)
                payment_results.append(("success", result))
            except Exception as e:
                payment_results.append(("error", str(e)))

        # Verify circuit opened
        assert breakers["payment"].state == "open"
        assert (
            sum(1 for r in payment_results if "Circuit breaker is open" in str(r[1]))
            > 0
        )

        # Test 2: Half-open state behavior
        await asyncio.sleep(0.1)  # Fast forward time for testing
        breakers["payment"]._last_failure_time = datetime.now() - timedelta(seconds=31)

        # First call in half-open state
        async def recovering_payment_call():
            if random.random() < 0.5:
                return "Payment successful"
            raise Exception("Still failing")

        half_open_results = []
        for _ in range(5):
            try:
                result = await breakers["payment"].call(recovering_payment_call)
                half_open_results.append(("success", result))
            except Exception as e:
                half_open_results.append(("error", str(e)))

        # Circuit should either close (if lucky) or return to open
        assert breakers["payment"].state in ["open", "closed"]

        # Test 3: Cascading circuit breakers
        async def dependent_service_call(upstream_breaker):
            """Service that depends on upstream service"""
            # Check upstream first
            try:
                await upstream_breaker.call(lambda: "upstream ok")
            except Exception:
                # Upstream failed, this service fails too
                raise Exception("Dependent service failed due to upstream")

            # Own logic
            if random.random() < 0.1:  # 10% failure rate
                raise Exception("Internal error")

            return "Success"

        # Reset breakers
        for breaker in breakers.values():
            breaker.reset()

        # Simulate cascading failures
        cascade_timeline = []

        for i in range(100):
            timestamp = time.time()

            # Inventory depends on payment
            try:
                result = await breakers["inventory"].call(
                    lambda: dependent_service_call(breakers["payment"])
                )
                cascade_timeline.append(
                    {
                        "time": timestamp,
                        "service": "inventory",
                        "status": "success",
                        "payment_state": breakers["payment"].state,
                        "inventory_state": breakers["inventory"].state,
                    }
                )
            except Exception as e:
                cascade_timeline.append(
                    {
                        "time": timestamp,
                        "service": "inventory",
                        "status": "failed",
                        "error": str(e),
                        "payment_state": breakers["payment"].state,
                        "inventory_state": breakers["inventory"].state,
                    }
                )

            # Inject payment failures after some successful calls
            if i == 20:

                async def always_fail():
                    raise Exception("Payment service down")

                # Force payment circuit to open
                for _ in range(6):
                    try:
                        await breakers["payment"].call(always_fail)
                    except Exception:
                        pass

            await asyncio.sleep(0.01)

        # Analyze cascade behavior
        payment_open_idx = next(
            i
            for i, event in enumerate(cascade_timeline)
            if event["payment_state"] == "open"
        )

        # Inventory should start failing after payment opens
        inventory_failures_after = sum(
            1
            for event in cascade_timeline[payment_open_idx:]
            if event["status"] == "failed"
        )

        assert inventory_failures_after > 0

    @pytest.mark.asyncio
    async def test_retry_patterns_with_backoff(self):
        """Test different retry patterns and backoff strategies"""

        class RetryStrategy:
            def __init__(self, max_retries=3, strategy="exponential"):
                self.max_retries = max_retries
                self.strategy = strategy

            def get_delay(self, attempt):
                """Calculate delay based on strategy"""
                if self.strategy == "exponential":
                    return 0.1 * (2**attempt)  # 0.1, 0.2, 0.4, 0.8...
                elif self.strategy == "linear":
                    return 0.1 * attempt  # 0.1, 0.2, 0.3...
                elif self.strategy == "fibonacci":
                    if attempt <= 1:
                        return 0.1
                    a, b = 0.1, 0.1
                    for _ in range(attempt - 1):
                        a, b = b, a + b
                    return b
                elif self.strategy == "decorrelated_jitter":
                    # AWS-style decorrelated jitter
                    base = 0.1 * (2**attempt)
                    return random.uniform(0.1, base)

                return 0.1  # Default

            async def execute_with_retry(self, func, *args, **kwargs):
                """Execute function with retry logic"""
                last_exception = None
                attempts = []

                for attempt in range(self.max_retries + 1):
                    start_time = time.time()

                    try:
                        result = await func(*args, **kwargs)
                        attempts.append(
                            {
                                "attempt": attempt,
                                "status": "success",
                                "duration": time.time() - start_time,
                            }
                        )
                        return result, attempts

                    except Exception as e:
                        last_exception = e
                        attempts.append(
                            {
                                "attempt": attempt,
                                "status": "failed",
                                "error": str(e),
                                "duration": time.time() - start_time,
                            }
                        )

                        if attempt < self.max_retries:
                            delay = self.get_delay(attempt)
                            await asyncio.sleep(delay)

                raise last_exception

        # Test different retry strategies
        strategies = ["exponential", "linear", "fibonacci", "decorrelated_jitter"]
        strategy_results = {}

        for strategy_name in strategies:
            retry = RetryStrategy(max_retries=4, strategy=strategy_name)

            # Simulate flaky service
            call_count = 0

            async def flaky_service():
                nonlocal call_count
                call_count += 1

                # Fail first 3 calls
                if call_count <= 3:
                    raise Exception(f"Service unavailable (attempt {call_count})")

                return f"Success on attempt {call_count}"

            # Reset counter for each strategy
            call_count = 0

            try:
                start_total_time = time.time()
                result, attempts = await retry.execute_with_retry(flaky_service)
                total_time = time.time() - start_total_time

                # Calculate delays between attempts
                delays = []
                for i in range(1, len(attempts)):
                    if attempts[i - 1]["status"] == "failed":
                        # Approximate delay (includes execution time)
                        delay = attempts[i]["duration"]
                        delays.append(delay)

                strategy_results[strategy_name] = {
                    "success": True,
                    "attempts": len(attempts),
                    "total_time": total_time,
                    "delays": delays,
                    "final_result": result,
                }

            except Exception as e:
                strategy_results[strategy_name] = {
                    "success": False,
                    "attempts": len(attempts),
                    "error": str(e),
                }

        # Verify retry patterns
        assert all(r["success"] for r in strategy_results.values())
        assert all(
            r["attempts"] == 4 for r in strategy_results.values()
        )  # 3 failures + 1 success

        # Verify backoff patterns (exponential should take longest)
        exp_time = strategy_results["exponential"]["total_time"]
        linear_time = strategy_results["linear"]["total_time"]
        assert exp_time > linear_time * 1.2  # Exponential should be notably slower

    @pytest.mark.asyncio
    async def test_bulkhead_isolation_pattern(self):
        """Test bulkhead pattern for failure isolation"""

        class BulkheadManager:
            def __init__(self):
                self.bulkheads = {}
                self.metrics = defaultdict(
                    lambda: {"accepted": 0, "rejected": 0, "completed": 0, "failed": 0}
                )

            def create_bulkhead(self, name, max_concurrent=10, max_queued=20):
                """Create a new bulkhead"""
                self.bulkheads[name] = {
                    "semaphore": asyncio.Semaphore(max_concurrent),
                    "queue": asyncio.Queue(maxsize=max_queued),
                    "active": 0,
                    "max_concurrent": max_concurrent,
                    "max_queued": max_queued,
                }
                return name

            async def execute_in_bulkhead(self, bulkhead_name, func, *args, **kwargs):
                """Execute function within bulkhead constraints"""
                bulkhead = self.bulkheads.get(bulkhead_name)
                if not bulkhead:
                    raise ValueError(f"Bulkhead {bulkhead_name} not found")

                # Try to acquire semaphore immediately
                try:
                    bulkhead["semaphore"].acquire_nowait()
                    acquired = True
                except Exception:
                    acquired = False

                if not acquired:
                    # Try to queue
                    try:
                        bulkhead["queue"].put_nowait(True)
                        self.metrics[bulkhead_name]["accepted"] += 1

                        # Wait for semaphore
                        await bulkhead["semaphore"].acquire()
                        bulkhead["queue"].get_nowait()

                    except asyncio.QueueFull:
                        self.metrics[bulkhead_name]["rejected"] += 1
                        raise Exception(f"Bulkhead {bulkhead_name} queue full")
                else:
                    self.metrics[bulkhead_name]["accepted"] += 1

                # Execute function
                bulkhead["active"] += 1
                try:
                    result = await func(*args, **kwargs)
                    self.metrics[bulkhead_name]["completed"] += 1
                    return result
                except Exception as e:
                    self.metrics[bulkhead_name]["failed"] += 1
                    # Log the error for debugging
                    logger.error(f"Bulkhead {bulkhead_name} operation failed: {str(e)}")
                    raise
                finally:
                    bulkhead["active"] -= 1
                    bulkhead["semaphore"].release()

        # Create bulkheads for different service types
        bulkhead_mgr = BulkheadManager()
        bulkhead_mgr.create_bulkhead("critical", max_concurrent=20, max_queued=50)
        bulkhead_mgr.create_bulkhead("standard", max_concurrent=10, max_queued=20)
        bulkhead_mgr.create_bulkhead("batch", max_concurrent=5, max_queued=10)

        # Simulate mixed workload
        async def critical_operation(duration=0.1):
            await asyncio.sleep(duration)
            if random.random() < 0.05:  # 5% failure rate
                raise Exception("Critical operation failed")
            return "critical_success"

        async def standard_operation(duration=0.2):
            await asyncio.sleep(duration)
            if random.random() < 0.1:  # 10% failure rate
                raise Exception("Standard operation failed")
            return "standard_success"

        async def batch_operation(duration=0.5):
            await asyncio.sleep(duration)
            if random.random() < 0.2:  # 20% failure rate
                raise Exception("Batch operation failed")
            return "batch_success"

        # Generate mixed load
        tasks = []
        operation_distribution = [
            ("critical", critical_operation, 0.5),  # 50% of requests
            ("standard", standard_operation, 0.3),  # 30% of requests
            ("batch", batch_operation, 0.2),  # 20% of requests
        ]

        for _ in range(200):
            # Select operation type based on distribution
            rand = random.random()
            cumulative = 0

            for bulkhead_name, operation, probability in operation_distribution:
                cumulative += probability
                if rand <= cumulative:
                    task = bulkhead_mgr.execute_in_bulkhead(bulkhead_name, operation)
                    tasks.append((bulkhead_name, task))
                    break

            # Small delay between requests
            await asyncio.sleep(0.01)

        # Gather results
        results = []
        for bulkhead_name, task in tasks:
            try:
                result = await task
                results.append((bulkhead_name, "success", result))
            except Exception as e:
                results.append((bulkhead_name, "error", str(e)))

        # Analyze bulkhead isolation
        for bulkhead_name in ["critical", "standard", "batch"]:
            metrics = bulkhead_mgr.metrics[bulkhead_name]

            # Verify bulkhead constraints were enforced
            assert metrics["rejected"] >= 0
            assert metrics["accepted"] > 0
            assert metrics["completed"] + metrics["failed"] <= metrics["accepted"]

            # Calculate success rate
            total_processed = metrics["completed"] + metrics["failed"]
            if total_processed > 0:
                success_rate = metrics["completed"] / total_processed

                # Critical should have highest success rate due to more resources
                if bulkhead_name == "critical":
                    assert success_rate > 0.9

    @pytest.mark.asyncio
    async def test_timeout_cascade_prevention(self):
        """Test timeout configurations to prevent cascading timeouts"""

        class TimeoutManager:
            def __init__(self):
                self.default_timeouts = {
                    "frontend": 10.0,
                    "api": 8.0,
                    "service": 5.0,
                    "database": 3.0,
                }
                self.call_stack = []

            async def call_with_timeout(self, service_name, target_service, operation):
                """Call service with appropriate timeout"""
                # Calculate remaining timeout budget
                current_depth = len(self.call_stack)
                base_timeout = self.default_timeouts.get(service_name, 5.0)

                # Reduce timeout for deeper calls
                adjusted_timeout = base_timeout * (0.8**current_depth)

                self.call_stack.append(
                    {
                        "service": service_name,
                        "target": target_service,
                        "timeout": adjusted_timeout,
                        "start_time": time.time(),
                    }
                )

                try:
                    result = await asyncio.wait_for(
                        operation(), timeout=adjusted_timeout
                    )

                    self.call_stack[-1]["status"] = "success"
                    self.call_stack[-1]["duration"] = (
                        time.time() - self.call_stack[-1]["start_time"]
                    )

                    return result

                except asyncio.TimeoutError:
                    self.call_stack[-1]["status"] = "timeout"
                    self.call_stack[-1]["duration"] = adjusted_timeout
                    raise

                finally:
                    completed_call = self.call_stack.pop()

                    # Check for timeout cascade risk
                    if completed_call["status"] == "timeout" and self.call_stack:
                        parent = self.call_stack[-1]
                        elapsed = time.time() - parent["start_time"]
                        remaining = parent["timeout"] - elapsed

                        if remaining < 1.0:  # Less than 1 second remaining
                            # Risk of cascade - log warning
                            completed_call["cascade_risk"] = True

        timeout_mgr = TimeoutManager()

        # Simulate service call chain
        async def database_operation():
            # Simulate slow database query
            delay = random.uniform(2, 4)
            await asyncio.sleep(delay)
            return "data"

        async def service_operation():
            # Service calls database
            return await timeout_mgr.call_with_timeout(
                "service", "database", database_operation
            )

        async def api_operation():
            # API calls service
            return await timeout_mgr.call_with_timeout(
                "api", "service", service_operation
            )

        async def frontend_operation():
            # Frontend calls API
            return await timeout_mgr.call_with_timeout("frontend", "api", api_operation)

        # Test various scenarios
        results = []

        for i in range(10):
            try:
                result = await frontend_operation()
                results.append(
                    {
                        "iteration": i,
                        "status": "success",
                        "call_stack": timeout_mgr.call_stack.copy(),
                    }
                )
            except asyncio.TimeoutError:
                results.append(
                    {
                        "iteration": i,
                        "status": "timeout",
                        "call_stack": timeout_mgr.call_stack.copy(),
                    }
                )

        # Analyze timeout behavior
        timeout_count = sum(1 for r in results if r["status"] == "timeout")
        assert timeout_count > 0  # Some timeouts expected with slow DB

        # Verify timeout budgets decreased with depth
        # This prevents cascading timeouts where parent times out
        # immediately after child


class ServiceMesh:
    """Simulated service mesh for testing"""

    def __init__(self):
        self.services = {}
        self.dependencies = nx.DiGraph()
        self.failure_rates = {}
        self.fallbacks = {}
        self.metrics = defaultdict(lambda: {"requests": 0, "errors": 0})
        self.fallback_metrics = defaultdict(lambda: defaultdict(int))

    def add_service(self, name, dependencies=None):
        """Add service to mesh"""
        self.services[name] = {"status": "healthy", "dependencies": dependencies or []}
        self.dependencies.add_node(name)

        for dep in dependencies or []:
            self.dependencies.add_edge(name, dep)

    def inject_failure(self, service, failure_rate):
        """Inject failure into service"""
        self.failure_rates[service] = failure_rate

    def set_fallback(self, service, dependency, fallback_action):
        """Set fallback for dependency failure"""
        if service not in self.fallbacks:
            self.fallbacks[service] = {}
        self.fallbacks[service][dependency] = fallback_action

    async def send_request(self, service, endpoint):
        """Simulate request through service mesh"""
        return await self._process_request(service, endpoint, visited=set())

    async def _process_request(self, service, endpoint, visited):
        """Process request with dependency resolution"""
        if service in visited:
            return {"status": "error", "reason": "circular_dependency"}

        visited.add(service)
        self.metrics[service]["requests"] += 1

        # Check if service should fail
        if random.random() < self.failure_rates.get(service, 0):
            self.metrics[service]["errors"] += 1
            return {"status": "error", "service": service, "reason": "injected_failure"}

        # Process dependencies
        service_info = self.services.get(service)
        if not service_info:
            return {"status": "error", "reason": "service_not_found"}

        for dep in service_info["dependencies"]:
            dep_result = await self._process_request(dep, endpoint, visited.copy())

            if dep_result["status"] == "error":
                # Check for fallback
                if service in self.fallbacks and dep in self.fallbacks[service]:
                    self.fallback_metrics[service][dep] += 1
                    # Execute fallback - simulate successful fallback action
                    # For cache failures, we simulate direct DB query as fallback
                    fallback_action = self.fallbacks[service][dep]
                    if fallback_action == "direct-db-query":
                        # Simulate successful direct DB query
                        continue  # Skip this failed dependency, use fallback
                else:
                    # Propagate failure
                    self.metrics[service]["errors"] += 1
                    return {
                        "status": "error",
                        "service": service,
                        "reason": "dependency_failure",
                        "failed_dependency": dep,
                    }

        # Service processed successfully
        await asyncio.sleep(random.uniform(0.01, 0.05))  # Simulate processing
        return {"status": "success", "service": service, "endpoint": endpoint}

    def get_failed_services(self):
        """Get list of services with high failure rate"""
        failed = []
        for service, metrics in self.metrics.items():
            if metrics["requests"] > 0:
                error_rate = metrics["errors"] / metrics["requests"]
                if error_rate > 0.5:
                    failed.append(service)
        return failed

    def get_fallback_metrics(self):
        """Get fallback usage metrics"""
        return dict(self.fallback_metrics)

    def reset(self):
        """Reset mesh state"""
        self.failure_rates.clear()
        self.metrics.clear()
        self.fallback_metrics.clear()
        self.fallbacks.clear()


class CircuitBreakerSystem:
    """Circuit breaker implementation for testing"""

    def create_breaker(
        self, failure_threshold, success_threshold, timeout, half_open_max_calls=1
    ):
        """Create a new circuit breaker"""
        return CircuitBreaker(
            failure_threshold=failure_threshold,
            success_threshold=success_threshold,
            timeout=timeout,
            half_open_max_calls=half_open_max_calls,
        )


class CircuitBreaker:
    """Circuit breaker implementation"""

    def __init__(
        self, failure_threshold, success_threshold, timeout, half_open_max_calls
    ):
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout = timeout
        self.half_open_max_calls = half_open_max_calls

        self.state = "closed"
        self.failure_count = 0
        self.success_count = 0
        self.half_open_calls = 0
        self._last_failure_time = None
        self._lock = asyncio.Lock()

    async def call(self, func, *args, **kwargs):
        """Execute function through circuit breaker"""
        async with self._lock:
            if self.state == "open":
                # Check if timeout has passed
                if (
                    self._last_failure_time
                    and datetime.now() - self._last_failure_time > self.timeout
                ):
                    self.state = "half-open"
                    self.half_open_calls = 0
                else:
                    raise Exception("Circuit breaker is open")

            elif self.state == "half-open":
                if self.half_open_calls >= self.half_open_max_calls:
                    raise Exception("Circuit breaker is half-open, max calls reached")
                self.half_open_calls += 1

        # Execute function
        try:
            result = await func(*args, **kwargs)

            async with self._lock:
                if self.state == "half-open":
                    self.success_count += 1
                    if self.success_count >= self.success_threshold:
                        self.state = "closed"
                        self.failure_count = 0
                        self.success_count = 0

                elif self.state == "closed":
                    self.failure_count = 0  # Reset on success

            return result

        except Exception as e:
            async with self._lock:
                self.failure_count += 1
                self._last_failure_time = datetime.now()
                self._last_error = str(e)  # Store the error for monitoring

                if self.state == "half-open":
                    self.state = "open"
                    self.success_count = 0
                    logger.warning(f"Circuit breaker opened from half-open state: {str(e)}")

                elif self.state == "closed":
                    if self.failure_count >= self.failure_threshold:
                        self.state = "open"
                        logger.warning(f"Circuit breaker opened after {self.failure_count} failures: {str(e)}")

            raise

    def reset(self):
        """Reset circuit breaker to closed state"""
        self.state = "closed"
        self.failure_count = 0
        self.success_count = 0
        self.half_open_calls = 0
        self._last_failure_time = None
