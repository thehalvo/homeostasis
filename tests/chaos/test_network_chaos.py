import asyncio
import random
import time

import pytest


class TestNetworkChaos:
    """Advanced network chaos engineering tests"""

    @pytest.fixture
    def network_simulator(self):
        """Create network simulation environment"""
        return NetworkSimulator()

    @pytest.mark.asyncio
    async def test_latency_distribution_patterns(self, network_simulator):
        """Test various latency distribution patterns"""
        patterns = [
            {
                "name": "uniform",
                "base_ms": 100,
                "variance_ms": 20,
                "distribution": lambda base, var: random.uniform(
                    base - var, base + var
                ),
            },
            {
                "name": "normal",
                "base_ms": 100,
                "std_dev_ms": 15,
                "distribution": lambda base, std: random.gauss(base, std),
            },
            {
                "name": "pareto",
                "base_ms": 50,
                "shape": 1.2,
                "distribution": lambda base, shape: base
                * (random.paretovariate(shape)),
            },
            {
                "name": "bimodal",
                "fast_ms": 10,
                "slow_ms": 500,
                "slow_probability": 0.1,
                "distribution": lambda fast, slow, prob: (
                    slow if random.random() < prob else fast
                ),
            },
        ]

        results = {}

        for pattern in patterns:
            samples = []

            # Generate latency samples
            for _ in range(1000):
                if pattern["name"] == "uniform":
                    latency = pattern["distribution"](
                        pattern["base_ms"], pattern["variance_ms"]
                    )
                elif pattern["name"] == "normal":
                    latency = max(
                        0,
                        pattern["distribution"](
                            pattern["base_ms"], pattern["std_dev_ms"]
                        ),
                    )
                elif pattern["name"] == "pareto":
                    latency = pattern["distribution"](
                        pattern["base_ms"], pattern["shape"]
                    )
                elif pattern["name"] == "bimodal":
                    latency = pattern["distribution"](
                        pattern["fast_ms"],
                        pattern["slow_ms"],
                        pattern["slow_probability"],
                    )

                samples.append(latency)

            # Calculate statistics
            samples.sort()
            results[pattern["name"]] = {
                "mean": sum(samples) / len(samples),
                "median": samples[len(samples) // 2],
                "p95": samples[int(len(samples) * 0.95)],
                "p99": samples[int(len(samples) * 0.99)],
                "min": samples[0],
                "max": samples[-1],
            }

        # Verify distribution characteristics
        assert 80 <= results["uniform"]["mean"] <= 120
        assert (
            results["normal"]["p99"] < results["normal"]["mean"] + 3 * 15
        )  # 3 std devs
        assert results["pareto"]["p99"] > results["pareto"]["p95"] * 1.5  # Long tail
        assert results["bimodal"]["p99"] > 400  # Slow mode captured

    @pytest.mark.asyncio
    async def test_packet_reordering_simulation(self, network_simulator):
        """Test packet reordering scenarios"""

        class PacketReorderer:
            def __init__(self, reorder_probability=0.1, max_delay_packets=3):
                self.reorder_probability = reorder_probability
                self.max_delay_packets = max_delay_packets
                self.delayed_packets = []
                self.sequence_number = 0

            def send_packet(self, data):
                """Simulate packet sending with potential reordering"""
                packet = {
                    "seq": self.sequence_number,
                    "data": data,
                    "timestamp": time.time(),
                }
                self.sequence_number += 1

                # Decide if packet should be delayed
                if random.random() < self.reorder_probability:
                    delay_count = random.randint(1, self.max_delay_packets)
                    self.delayed_packets.append((packet, delay_count))
                    return None

                # Send any packets whose delay has expired
                ready_packets = []
                remaining_delayed = []

                for delayed_packet, remaining_delay in self.delayed_packets:
                    if remaining_delay <= 1:
                        ready_packets.append(delayed_packet)
                    else:
                        remaining_delayed.append((delayed_packet, remaining_delay - 1))

                self.delayed_packets = remaining_delayed

                # Return current packet and any ready delayed packets
                return [packet] + ready_packets

        # Test packet reordering
        reorderer = PacketReorderer(reorder_probability=0.2)
        sent_packets = []
        received_packets = []

        # Send 100 packets
        for i in range(100):
            packet = f"data_{i}"
            sent_packets.append(packet)
            result = reorderer.send_packet(packet)
            if result:
                received_packets.extend(result)

        # Flush remaining delayed packets
        while reorderer.delayed_packets:
            result = reorderer.send_packet("flush")
            if result:
                received_packets.extend(result)

        # Verify all sent packets were received (excluding flush packets)
        received_data = [p["data"] for p in received_packets if p["data"] != "flush"]
        assert set(sent_packets) == set(received_data)
        assert len(sent_packets) == len(received_data)

        # Analyze reordering
        out_of_order_count = 0
        max_sequence_seen = -1

        for packet in received_packets:
            if packet["seq"] < max_sequence_seen:
                out_of_order_count += 1
            max_sequence_seen = max(max_sequence_seen, packet["seq"])

        # Verify reordering occurred
        assert out_of_order_count > 0
        assert out_of_order_count <= 50  # Not excessive reordering (allow for probabilistic variance)

    @pytest.mark.asyncio
    async def test_network_partition_scenarios(self):
        """Test various network partition scenarios"""

        class NetworkPartition:
            def __init__(self):
                self.partitions = {}
                self.nodes = set()

            def add_node(self, node_id):
                """Add a node to the network"""
                self.nodes.add(node_id)
                self.partitions[node_id] = {node_id}  # Initially isolated

            def merge_partitions(self, node1, node2):
                """Merge two network partitions"""
                partition1 = self.partitions[node1]
                partition2 = self.partitions[node2]

                # Create merged partition
                merged = partition1.union(partition2)

                # Update all nodes in merged partition
                for node in merged:
                    self.partitions[node] = merged

            def split_partition(self, nodes_group1, nodes_group2):
                """Split nodes into two partitions"""
                # Update partition membership
                for node in nodes_group1:
                    self.partitions[node] = set(nodes_group1)
                for node in nodes_group2:
                    self.partitions[node] = set(nodes_group2)

            def can_communicate(self, node1, node2):
                """Check if two nodes can communicate"""
                return node2 in self.partitions.get(node1, set())

        # Test various partition scenarios
        network = NetworkPartition()
        nodes = ["node_a", "node_b", "node_c", "node_d", "node_e"]

        for node in nodes:
            network.add_node(node)

        # Scenario 1: Full connectivity
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                network.merge_partitions(nodes[i], nodes[j])

        assert all(network.can_communicate(n1, n2) for n1 in nodes for n2 in nodes)

        # Scenario 2: Split brain (two equal partitions)
        network.split_partition(["node_a", "node_b"], ["node_c", "node_d", "node_e"])

        assert network.can_communicate("node_a", "node_b")
        assert not network.can_communicate("node_a", "node_c")
        assert network.can_communicate("node_c", "node_d")

        # Scenario 3: Isolated node
        network.split_partition(["node_a"], ["node_b", "node_c", "node_d", "node_e"])

        assert not any(
            network.can_communicate("node_a", other)
            for other in nodes
            if other != "node_a"
        )

    @pytest.mark.asyncio
    async def test_bandwidth_throttling(self):
        """Test bandwidth throttling simulation"""

        class BandwidthThrottler:
            def __init__(self, max_bytes_per_second):
                self.max_bytes_per_second = max_bytes_per_second
                self.tokens = (
                    0  # Start with no tokens to enforce rate from the beginning
                )
                self.last_update = time.time()

            async def send_data(self, data_size_bytes):
                """Attempt to send data with bandwidth throttling"""
                # Update token bucket
                current_time = time.time()
                elapsed = current_time - self.last_update
                self.tokens = min(
                    self.max_bytes_per_second,
                    self.tokens + elapsed * self.max_bytes_per_second,
                )
                self.last_update = current_time

                # Check if we can send
                if data_size_bytes <= self.tokens:
                    self.tokens -= data_size_bytes
                    return True, 0  # No delay
                else:
                    # Calculate required wait time
                    tokens_needed = data_size_bytes - self.tokens
                    wait_time = tokens_needed / self.max_bytes_per_second
                    await asyncio.sleep(wait_time)

                    # Update tokens after wait - we've waited long enough to send the data
                    self.tokens = 0  # All tokens were used
                    self.last_update = time.time()
                    return True, wait_time

        # Test throttling behavior
        throttler = BandwidthThrottler(max_bytes_per_second=1024 * 1024)  # 1MB/s

        send_times = []
        data_sizes = [512 * 1024, 1024 * 1024, 256 * 1024, 2048 * 1024]  # Various sizes

        start_time = time.time()
        for size in data_sizes:
            success, delay = await throttler.send_data(size)
            send_times.append(
                {"size": size, "delay": delay, "timestamp": time.time() - start_time}
            )

        # Verify throttling worked
        total_bytes = sum(data_sizes)
        total_time = send_times[-1]["timestamp"]
        effective_bandwidth = total_bytes / total_time

        # Should be close to but not exceed max bandwidth
        assert effective_bandwidth <= 1024 * 1024 * 1.1  # 10% tolerance

    @pytest.mark.asyncio
    async def test_connection_pool_exhaustion(self):
        """Test connection pool exhaustion scenarios"""

        class ConnectionPool:
            def __init__(self, max_connections=10):
                self.max_connections = max_connections
                self.active_connections = []
                self.waiting_queue = asyncio.Queue()
                self.metrics = {
                    "total_requests": 0,
                    "rejected_requests": 0,
                    "queue_timeouts": 0,
                    "max_queue_size": 0,
                }

            async def acquire_connection(self, timeout=5.0):
                """Acquire a connection from the pool"""
                self.metrics["total_requests"] += 1

                if len(self.active_connections) < self.max_connections:
                    conn = f"conn_{len(self.active_connections)}"
                    self.active_connections.append(conn)
                    return conn

                # Pool exhausted, queue the request
                try:
                    future = asyncio.Future()
                    await self.waiting_queue.put(future)
                    self.metrics["max_queue_size"] = max(
                        self.metrics["max_queue_size"], self.waiting_queue.qsize()
                    )

                    # Wait for connection with timeout
                    conn = await asyncio.wait_for(future, timeout=timeout)
                    return conn
                except asyncio.TimeoutError:
                    self.metrics["queue_timeouts"] += 1
                    raise

            async def release_connection(self, conn):
                """Release a connection back to the pool"""
                if conn in self.active_connections:
                    self.active_connections.remove(conn)

                    # Check if anyone is waiting
                    if not self.waiting_queue.empty():
                        future = await self.waiting_queue.get()
                        if not future.done():
                            future.set_result(conn)
                            self.active_connections.append(conn)

        # Test pool exhaustion
        pool = ConnectionPool(max_connections=5)

        async def use_connection(duration=0.1):
            """Simulate connection usage"""
            try:
                conn = await pool.acquire_connection(timeout=2.0)
                await asyncio.sleep(duration)
                await pool.release_connection(conn)
                return "success"
            except asyncio.TimeoutError:
                return "timeout"

        # Create burst of requests - overload the pool
        tasks = []
        for i in range(20):
            # First 10 connections hold for longer, causing timeouts for later requests
            duration = 2.5 if i < 10 else 0.1
            tasks.append(use_connection(duration))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Analyze results
        success_count = results.count("success")
        timeout_count = results.count("timeout")

        # Pool exhaustion behavior:
        # Some of the first tasks get connections and hold them for 2.5s
        # Waiting tasks timeout after 2s, before connections are released
        assert success_count >= 5  # Some requests succeed
        assert timeout_count >= 10  # Many timeout due to pool exhaustion
        assert pool.metrics["queue_timeouts"] == timeout_count
        assert pool.metrics["max_queue_size"] >= 10  # Queue was heavily used

    @pytest.mark.asyncio
    async def test_dns_failure_simulation(self):
        """Test DNS resolution failure scenarios"""

        class DNSResolver:
            def __init__(self):
                self.dns_cache = {}
                self.failure_domains = set()
                self.slow_domains = set()
                self.metrics = {
                    "queries": 0,
                    "cache_hits": 0,
                    "failures": 0,
                    "slow_queries": 0,
                }

            async def resolve(self, domain, use_cache=True):
                """Resolve domain with failure simulation"""
                self.metrics["queries"] += 1

                # Check cache first
                if use_cache and domain in self.dns_cache:
                    self.metrics["cache_hits"] += 1
                    return self.dns_cache[domain]

                # Simulate failures
                if domain in self.failure_domains:
                    self.metrics["failures"] += 1
                    raise Exception(f"DNS resolution failed for {domain}")

                # Simulate slow DNS
                if domain in self.slow_domains:
                    self.metrics["slow_queries"] += 1
                    await asyncio.sleep(random.uniform(2, 5))
                else:
                    await asyncio.sleep(random.uniform(0.01, 0.1))

                # Generate mock IP
                ip = f"10.0.{random.randint(0, 255)}.{random.randint(1, 254)}"
                self.dns_cache[domain] = ip
                return ip

            def inject_failure(self, domain):
                """Make a domain fail DNS resolution"""
                self.failure_domains.add(domain)
                self.dns_cache.pop(domain, None)

            def inject_slowness(self, domain):
                """Make a domain resolve slowly"""
                self.slow_domains.add(domain)
                self.dns_cache.pop(domain, None)  # Clear cache to force re-resolution

        # Test DNS failures
        resolver = DNSResolver()

        domains = [
            "service-a.internal",
            "service-b.internal",
            "service-c.internal",
            "database.internal",
            "cache.internal",
        ]

        # Resolve all domains initially
        initial_results = {}
        for domain in domains:
            ip = await resolver.resolve(domain)
            initial_results[domain] = ip

        # Inject failures and slowness
        resolver.inject_failure("database.internal")
        resolver.inject_slowness("cache.internal")

        # Test resolution with failures
        resolution_results = []

        for domain in domains:
            start_time = time.time()
            try:
                ip = await resolver.resolve(domain)
                duration = time.time() - start_time
                resolution_results.append(
                    {
                        "domain": domain,
                        "status": "success",
                        "duration": duration,
                        "ip": ip,
                    }
                )
            except Exception as e:
                duration = time.time() - start_time
                resolution_results.append(
                    {
                        "domain": domain,
                        "status": "failed",
                        "duration": duration,
                        "error": str(e),
                    }
                )

        # Verify behavior
        failed_domains = [
            r["domain"] for r in resolution_results if r["status"] == "failed"
        ]
        slow_domains = [
            r["domain"]
            for r in resolution_results
            if r["status"] == "success" and r["duration"] > 1.0
        ]

        assert "database.internal" in failed_domains
        assert "cache.internal" in slow_domains
        assert resolver.metrics["failures"] > 0
        assert resolver.metrics["cache_hits"] > 0


class NetworkSimulator:
    """Simulate various network conditions"""

    def __init__(self):
        self.latency_base = 0
        self.latency_variance = 0
        self.packet_loss_rate = 0
        self.bandwidth_limit = float("inf")

    def set_latency(self, base_ms, variance_ms):
        """Set network latency parameters"""
        self.latency_base = base_ms
        self.latency_variance = variance_ms

    def set_packet_loss(self, loss_rate):
        """Set packet loss rate (0-1)"""
        self.packet_loss_rate = loss_rate

    def set_bandwidth(self, bytes_per_second):
        """Set bandwidth limit"""
        self.bandwidth_limit = bytes_per_second

    async def simulate_request(self, payload_size):
        """Simulate a network request with current conditions"""
        # Simulate packet loss
        if random.random() < self.packet_loss_rate:
            raise Exception("Packet lost")

        # Simulate latency
        latency = self.latency_base + random.uniform(
            -self.latency_variance, self.latency_variance
        )
        await asyncio.sleep(latency / 1000)  # Convert to seconds

        # Simulate bandwidth limitation
        transfer_time = payload_size / self.bandwidth_limit
        await asyncio.sleep(transfer_time)

        return True
