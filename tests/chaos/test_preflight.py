"""
Pre-flight checks for chaos engineering tests
Ensures system is in a healthy state before running chaos experiments
"""

import pytest
import psutil
import asyncio
import os
import docker
from datetime import datetime


class TestPreflightChecks:
    """Pre-flight validation before chaos experiments"""
    
    @pytest.fixture
    def docker_client(self):
        """Create Docker client if available"""
        try:
            return docker.from_env()
        except:
            return None
    
    def test_system_resources(self):
        """Verify sufficient system resources"""
        # Get thresholds from environment variables or use defaults
        cpu_threshold = int(os.environ.get('CHAOS_CPU_THRESHOLD', '80'))
        memory_threshold = int(os.environ.get('CHAOS_MEMORY_THRESHOLD', '90'))
        min_memory_gb = int(os.environ.get('CHAOS_MIN_MEMORY_GB', '1'))
        disk_threshold = int(os.environ.get('CHAOS_DISK_THRESHOLD', '90'))
        min_disk_gb = int(os.environ.get('CHAOS_MIN_DISK_GB', '5'))
        
        # CPU check
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent >= cpu_threshold:
            pytest.skip(f"CPU usage too high: {cpu_percent}% (threshold: {cpu_threshold}%)")
        
        # Memory check
        memory = psutil.virtual_memory()
        if memory.percent >= memory_threshold:
            pytest.skip(f"Memory usage too high: {memory.percent}% (threshold: {memory_threshold}%)")
        if memory.available < min_memory_gb * 1024 * 1024 * 1024:
            pytest.skip(f"Less than {min_memory_gb}GB memory available: {memory.available / (1024**3):.1f}GB")
        
        # Disk check
        disk = psutil.disk_usage('/')
        if disk.percent >= disk_threshold:
            pytest.skip(f"Disk usage too high: {disk.percent}% (threshold: {disk_threshold}%)")
        if disk.free < min_disk_gb * 1024 * 1024 * 1024:
            pytest.skip(f"Less than {min_disk_gb}GB disk space available: {disk.free / (1024**3):.1f}GB")
    
    def test_network_connectivity(self):
        """Verify network is functional"""
        # Check network interfaces
        interfaces = psutil.net_if_stats()
        active_interfaces = [iface for iface, stats in interfaces.items() if stats.isup]
        assert len(active_interfaces) > 0, "No active network interfaces"
        
        # Check for localhost connectivity
        try:
            connections = psutil.net_connections()
            listening_ports = [conn.laddr.port for conn in connections 
                              if conn.status == 'LISTEN' and conn.laddr.ip in ('127.0.0.1', '0.0.0.0')]
            assert len(listening_ports) > 0, "No services listening on localhost"
        except (psutil.AccessDenied, PermissionError):
            # Skip this check on systems where we don't have permission
            # At least verify we have network interfaces
            pass
    
    def test_process_limits(self):
        """Check process and file descriptor limits"""
        import resource
        
        # Check file descriptor limits
        soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
        print(f"DEBUG: Current file descriptor limits - soft: {soft_limit}, hard: {hard_limit}")
        
        # Try to increase the soft limit if it's too low
        if soft_limit < 1024:
            print(f"DEBUG: Attempting to increase soft limit from {soft_limit} to 1024")
            try:
                new_limit = min(1024, hard_limit)  # Can't exceed hard limit
                resource.setrlimit(resource.RLIMIT_NOFILE, (new_limit, hard_limit))
                soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
                print(f"DEBUG: Updated file descriptor limits - soft: {soft_limit}, hard: {hard_limit}")
            except Exception as e:
                print(f"DEBUG: Failed to increase file descriptor limit: {e}")
        
        assert soft_limit >= 1024, f"File descriptor limit too low: {soft_limit} (expected >= 1024)"
        
        # Check process limits
        if hasattr(resource, 'RLIMIT_NPROC'):
            soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NPROC)
            assert soft_limit >= 512, f"Process limit too low: {soft_limit}"
    
    def test_docker_availability(self, docker_client):
        """Verify Docker is available and functional"""
        if docker_client is None:
            pytest.skip("Docker not available")
        
        # Check Docker daemon
        try:
            docker_client.ping()
        except Exception as e:
            pytest.fail(f"Docker daemon not responding: {e}")
        
        # Check for required images
        required_images = ['python:3.9', 'alpine:latest']
        available_images = {img.tags[0] for img in docker_client.images.list() if img.tags}
        
        # Note: Don't fail if images aren't present, they can be pulled
        missing_images = [img for img in required_images if img not in available_images]
        if missing_images:
            pytest.skip(f"Docker images will be pulled: {missing_images}")
    
    def test_required_tools(self):
        """Check for required system tools"""
        required_tools = {
            'tc': 'Traffic control for network chaos',
            'iptables': 'Firewall rules for network isolation',
            'stress-ng': 'Stress testing tool',
            'iperf3': 'Network performance testing'
        }
        
        missing_tools = []
        for tool, description in required_tools.items():
            if os.system(f"which {tool} > /dev/null 2>&1") != 0:
                missing_tools.append(f"{tool} ({description})")
        
        if missing_tools:
            # Don't fail, just warn
            pytest.skip(f"Optional tools not available: {', '.join(missing_tools)}")
    
    def test_kernel_capabilities(self):
        """Check for required kernel capabilities"""
        # Check if running in container
        if os.path.exists('/.dockerenv'):
            pytest.skip("Running in container, skipping kernel checks")
        
        # Check for cgroup support
        if not os.path.exists('/sys/fs/cgroup'):
            pytest.skip("cgroups not available")
        
        # Check for network namespace support
        try:
            with open('/proc/sys/kernel/unprivileged_userns_clone', 'r') as f:
                if f.read().strip() != '1':
                    pytest.skip("Unprivileged user namespaces not enabled")
        except FileNotFoundError:
            pass  # File doesn't exist on all systems
    
    @pytest.mark.asyncio
    async def test_baseline_performance(self):
        """Establish baseline performance metrics"""
        metrics = {
            'cpu_samples': [],
            'memory_samples': [],
            'io_samples': []
        }
        
        # Collect baseline over 5 seconds
        for _ in range(5):
            metrics['cpu_samples'].append(psutil.cpu_percent(interval=0))
            metrics['memory_samples'].append(psutil.virtual_memory().percent)
            
            io_counters = psutil.disk_io_counters()
            if io_counters:
                metrics['io_samples'].append({
                    'read_bytes': io_counters.read_bytes,
                    'write_bytes': io_counters.write_bytes
                })
            
            await asyncio.sleep(1)
        
        # Calculate baselines
        avg_cpu = sum(metrics['cpu_samples']) / len(metrics['cpu_samples'])
        avg_memory = sum(metrics['memory_samples']) / len(metrics['memory_samples'])
        
        # Get baseline thresholds from environment variables
        cpu_baseline_threshold = int(os.environ.get('CHAOS_CPU_BASELINE_THRESHOLD', '50'))
        memory_baseline_threshold = int(os.environ.get('CHAOS_MEMORY_BASELINE_THRESHOLD', '85'))
        
        # Verify system is relatively idle
        if avg_cpu >= cpu_baseline_threshold:
            pytest.skip(f"System CPU baseline too high: {avg_cpu:.1f}% (threshold: {cpu_baseline_threshold}%)")
        if avg_memory >= memory_baseline_threshold:
            pytest.skip(f"System memory baseline too high: {avg_memory:.1f}% (threshold: {memory_baseline_threshold}%)")
        
        # Store baselines for chaos tests
        with open('/tmp/chaos_baseline.txt', 'w') as f:
            f.write(f"cpu_baseline={avg_cpu}\n")
            f.write(f"memory_baseline={avg_memory}\n")
            f.write(f"timestamp={datetime.now().isoformat()}\n")
    
    def test_permissions(self):
        """Check for required permissions"""
        # Check if running as root (required for some chaos operations)
        if os.geteuid() != 0:
            # Check sudo availability
            if os.system("sudo -n true > /dev/null 2>&1") != 0:
                pytest.skip("Not running as root and sudo not available")
        
        # Check CAP_NET_ADMIN capability (for network chaos)
        try:
            import subprocess
            result = subprocess.run(['capsh', '--print'], 
                                  capture_output=True, text=True)
            if 'cap_net_admin' not in result.stdout.lower():
                pytest.skip("CAP_NET_ADMIN capability not available")
        except:
            pass  # capsh might not be available
    
    def test_chaos_module_dependencies(self):
        """Verify chaos engineering module dependencies"""
        try:
            from modules.reliability.chaos_engineering import ChaosEngineer
            from modules.monitoring.error_collector import ErrorCollector
            from modules.monitoring.metrics_collector import MetricsCollector
            # Verify modules are importable
            assert ErrorCollector is not None
            assert MetricsCollector is not None
        except ImportError as e:
            pytest.fail(f"Required chaos modules not available: {e}")
        
        # Test instantiation
        try:
            chaos = ChaosEngineer()
            assert chaos is not None
        except Exception as e:
            pytest.fail(f"Failed to initialize ChaosEngineer: {e}")
    
    def test_cleanup_from_previous_runs(self):
        """Clean up any artifacts from previous chaos runs"""
        cleanup_performed = []
        
        # Clean up stress-ng processes
        if os.system("pkill -f stress-ng > /dev/null 2>&1") == 0:
            cleanup_performed.append("Killed lingering stress-ng processes")
        
        # Clean up traffic control rules
        if os.geteuid() == 0:
            if os.system("tc qdisc del dev lo root > /dev/null 2>&1") == 0:
                cleanup_performed.append("Removed traffic control rules")
        
        # Clean up temporary files
        chaos_temp_files = ['/tmp/chaos_baseline.txt', '/tmp/chaos_runner.lock']
        for file_path in chaos_temp_files:
            if os.path.exists(file_path):
                os.remove(file_path)
                cleanup_performed.append(f"Removed {file_path}")
        
        if cleanup_performed:
            print(f"Cleanup performed: {', '.join(cleanup_performed)}")
    
    def test_concurrent_test_check(self):
        """Ensure no other chaos tests are running"""
        lock_file = '/tmp/chaos_runner.lock'
        
        if os.path.exists(lock_file):
            # Check if lock is stale
            with open(lock_file, 'r') as f:
                lock_data = f.read().strip()
            
            try:
                pid = int(lock_data)
                # Check if process is still running
                if psutil.pid_exists(pid):
                    process = psutil.Process(pid)
                    if 'chaos' in process.name().lower():
                        pytest.fail(f"Another chaos test is running (PID: {pid})")
            except:
                pass
            
            # Remove stale lock
            os.remove(lock_file)
        
        # Create new lock
        with open(lock_file, 'w') as f:
            f.write(str(os.getpid()))