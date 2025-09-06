#!/usr/bin/env python3
"""
Chaos Engineering Test Runner for Homeostasis

This runner orchestrates chaos experiments across the system, providing:
- Experiment scheduling and execution
- Real-time monitoring and metrics collection
- Safety checks and automatic rollback
- Comprehensive reporting
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from modules.monitoring.error_collector import ErrorCollector  # noqa: E402
from modules.monitoring.metrics_collector import MetricsCollector  # noqa: E402
from modules.reliability.chaos_engineering import (  # noqa: E402
    ChaosEngineer,
    ChaosExperiment,
    ChaosMonkey,
    FaultType,
)


class ChaosTestRunner:
    """Orchestrates chaos engineering tests"""

    def __init__(self, config_path: Optional[str] = None):
        self.config = self.load_config(config_path)
        self.chaos_engineer = ChaosEngineer()
        self.chaos_monkey = ChaosMonkey(self.chaos_engineer)
        self.metrics_collector = MetricsCollector()
        self.error_collector = ErrorCollector()

        self.experiments_run = 0
        self.experiments_passed = 0
        self.experiments_failed = 0
        self.start_time = None
        self.should_stop = False

        # Setup logging
        self.setup_logging()

        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        default_config = {
            "safety": {
                "max_error_rate": 0.5,
                "min_success_rate": 0.8,
                "max_latency_ms": 5000,
                "auto_rollback": True,
            },
            "monitoring": {"interval_seconds": 5, "metrics_window_minutes": 10},
            "experiments": {
                "categories": ["network", "resource", "service"],
                "intensity": "medium",  # low, medium, high
                "duration_minutes": 30,
            },
            "reporting": {
                "output_dir": "chaos_reports",
                "format": "json",  # json, yaml, html
                "include_metrics": True,
            },
        }

        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                if config_path.endswith(".yaml") or config_path.endswith(".yml"):
                    user_config = yaml.safe_load(f)
                else:
                    user_config = json.load(f)

            # Merge with defaults
            return self._merge_configs(default_config, user_config)

        return default_config

    def _merge_configs(self, default: Dict, user: Dict) -> Dict:
        """Recursively merge configurations"""
        result = default.copy()

        for key, value in user.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value

        return result

    def setup_logging(self):
        """Configure logging"""
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler("chaos_runner.log"),
            ],
        )
        self.logger = logging.getLogger("ChaosRunner")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.should_stop = True

    async def run(self, experiment_files: Optional[List[str]] = None):
        """Run chaos experiments"""
        self.start_time = datetime.now()
        self.logger.info("Starting Chaos Engineering Test Runner")

        try:
            # Load experiments
            experiments = await self.load_experiments(experiment_files)

            if not experiments:
                self.logger.warning("No experiments to run")
                return

            self.logger.info(f"Loaded {len(experiments)} experiments")

            # Start monitoring
            monitor_task = asyncio.create_task(self.monitor_system())

            # Run experiments
            for experiment in experiments:
                if self.should_stop:
                    self.logger.info("Stopping experiment execution")
                    break

                await self.run_experiment(experiment)

            # Stop monitoring
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

            # Generate report
            await self.generate_report()

        except Exception as e:
            self.logger.error(f"Fatal error in chaos runner: {e}", exc_info=True)
            raise

        finally:
            self.logger.info(
                f"Chaos tests completed. Run: {self.experiments_run}, "
                f"Passed: {self.experiments_passed}, Failed: {self.experiments_failed}"
            )

    async def load_experiments(
        self, experiment_files: Optional[List[str]] = None
    ) -> List[ChaosExperiment]:
        """Load experiments from files or generate based on config"""
        experiments = []

        if experiment_files:
            # Load from specified files
            for file_path in experiment_files:
                if os.path.exists(file_path):
                    with open(file_path, "r") as f:
                        if file_path.endswith(".yaml") or file_path.endswith(".yml"):
                            exp_data = yaml.safe_load(f)
                        else:
                            exp_data = json.load(f)

                    experiment = self._create_experiment_from_dict(exp_data)
                    experiments.append(experiment)
                else:
                    self.logger.warning(f"Experiment file not found: {file_path}")

        else:
            # Generate experiments based on configuration
            experiments = self._generate_experiments()

        return experiments

    def _create_experiment_from_dict(self, data: Dict[str, Any]) -> ChaosExperiment:
        """Create ChaosExperiment from dictionary"""
        return ChaosExperiment(
            name=data["name"],
            description=data.get("description", ""),
            hypothesis=data["hypothesis"],
            fault_type=FaultType[data["fault_type"]],
            target_service=data["target_service"],
            parameters=data.get("parameters", {}),
            duration=timedelta(minutes=data.get("duration_minutes", 5)),
            rollback_on_failure=data.get("rollback_on_failure", True),
        )

    def _generate_experiments(self) -> List[ChaosExperiment]:
        """Generate experiments based on configuration"""
        experiments = []
        categories = self.config["experiments"]["categories"]
        intensity = self.config["experiments"]["intensity"]

        # Network experiments
        if "network" in categories:
            experiments.extend(self._generate_network_experiments(intensity))

        # Resource experiments
        if "resource" in categories:
            experiments.extend(self._generate_resource_experiments(intensity))

        # Service experiments
        if "service" in categories:
            experiments.extend(self._generate_service_experiments(intensity))

        return experiments

    def _generate_network_experiments(self, intensity: str) -> List[ChaosExperiment]:
        """Generate network chaos experiments"""
        experiments = []

        # Latency injection
        latency_params = {
            "low": {"latency_ms": 100, "jitter_ms": 20, "affected_percentage": 25},
            "medium": {"latency_ms": 500, "jitter_ms": 100, "affected_percentage": 50},
            "high": {"latency_ms": 2000, "jitter_ms": 500, "affected_percentage": 75},
        }

        experiments.append(
            ChaosExperiment(
                name=f"Network Latency Test ({intensity})",
                description=f"Inject network latency with {intensity} intensity",
                hypothesis="System should maintain acceptable response times with degraded network",
                fault_type=FaultType.NETWORK_LATENCY,
                target_service="all",
                parameters=latency_params[intensity],
                duration=timedelta(minutes=10),
                rollback_on_failure=True,
            )
        )

        # Packet loss
        if intensity in ["medium", "high"]:
            loss_params = {
                "medium": {"packet_loss_percentage": 5, "correlation": 25},
                "high": {"packet_loss_percentage": 15, "correlation": 50},
            }

            experiments.append(
                ChaosExperiment(
                    name=f"Packet Loss Test ({intensity})",
                    description=f"Introduce packet loss with {intensity} intensity",
                    hypothesis="System should handle packet loss with retries",
                    fault_type=FaultType.NETWORK_PARTITION,
                    target_service="critical-services",
                    parameters=loss_params[intensity],
                    duration=timedelta(minutes=5),
                    rollback_on_failure=True,
                )
            )

        return experiments

    def _generate_resource_experiments(self, intensity: str) -> List[ChaosExperiment]:
        """Generate resource chaos experiments"""
        experiments = []

        # CPU pressure
        cpu_params = {
            "low": {"cpu_percentage": 50, "core_count": 1},
            "medium": {"cpu_percentage": 75, "core_count": 2},
            "high": {"cpu_percentage": 90, "core_count": 4},
        }

        experiments.append(
            ChaosExperiment(
                name=f"CPU Pressure Test ({intensity})",
                description=f"Apply CPU pressure with {intensity} intensity",
                hypothesis="System should handle CPU pressure through scaling or load shedding",
                fault_type=FaultType.RESOURCE_CPU,
                target_service="compute-intensive",
                parameters=cpu_params[intensity],
                duration=timedelta(minutes=5),
                rollback_on_failure=True,
            )
        )

        # Memory pressure
        if intensity in ["medium", "high"]:
            memory_params = {
                "medium": {"leak_rate_mb_per_min": 100, "max_leak_gb": 2},
                "high": {"leak_rate_mb_per_min": 500, "max_leak_gb": 5},
            }

            experiments.append(
                ChaosExperiment(
                    name=f"Memory Pressure Test ({intensity})",
                    description=f"Simulate memory leaks with {intensity} intensity",
                    hypothesis="System should detect and handle memory pressure",
                    fault_type=FaultType.RESOURCE_MEMORY,
                    target_service="memory-intensive",
                    parameters=memory_params[intensity],
                    duration=timedelta(minutes=10),
                    rollback_on_failure=True,
                )
            )

        return experiments

    def _generate_service_experiments(self, intensity: str) -> List[ChaosExperiment]:
        """Generate service chaos experiments"""
        experiments = []

        # Service failures
        failure_params = {
            "low": {"failure_probability": 0.1, "affected_dependencies": ["cache"]},
            "medium": {
                "failure_probability": 0.3,
                "affected_dependencies": ["cache", "secondary-db"],
            },
            "high": {
                "failure_probability": 0.5,
                "affected_dependencies": ["cache", "primary-db", "auth"],
            },
        }

        experiments.append(
            ChaosExperiment(
                name=f"Service Failure Test ({intensity})",
                description=f"Inject service failures with {intensity} intensity",
                hypothesis="Circuit breakers should prevent cascading failures",
                fault_type=FaultType.SERVICE_FAILURE,
                target_service="backend-services",
                parameters=failure_params[intensity],
                duration=timedelta(minutes=15),
                rollback_on_failure=True,
            )
        )

        return experiments

    async def run_experiment(self, experiment: ChaosExperiment):
        """Run a single chaos experiment"""
        self.logger.info(f"Starting experiment: {experiment.name}")
        self.experiments_run += 1

        # Collect baseline metrics
        baseline_metrics = await self.collect_baseline_metrics()

        # Check safety conditions before starting
        if not await self.check_safety_conditions(baseline_metrics):
            self.logger.warning(
                f"Safety conditions not met, skipping experiment: {experiment.name}"
            )
            self.experiments_failed += 1
            return

        # Run experiment
        start_time = datetime.now()

        try:
            # Execute experiment with monitoring
            result = await self.chaos_engineer.run_experiment(
                experiment, self.metrics_collector
            )

            # Check if hypothesis was validated
            if result.get("hypothesis_validated", False):
                self.logger.info(f"Experiment passed: {experiment.name}")
                self.experiments_passed += 1
            else:
                self.logger.warning(
                    f"Experiment failed: {experiment.name} - "
                    f"Hypothesis not validated"
                )
                self.experiments_failed += 1

            # Log detailed results
            self.logger.info(f"Experiment results: {json.dumps(result, indent=2)}")

        except Exception as e:
            self.logger.error(
                f"Error running experiment {experiment.name}: {e}", exc_info=True
            )
            self.experiments_failed += 1

            # Rollback if configured
            if experiment.rollback_on_failure:
                self.logger.info("Initiating rollback due to experiment failure")
                await self.chaos_engineer.rollback_experiment(experiment)

        finally:
            # Ensure cleanup
            duration = (datetime.now() - start_time).total_seconds()
            self.logger.info(
                f"Experiment {experiment.name} completed in {duration:.2f} seconds"
            )

    async def collect_baseline_metrics(self) -> Dict[str, Any]:
        """Collect baseline system metrics"""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "error_rate": 0.01,  # Simulated
            "latency_p50": 50,
            "latency_p95": 150,
            "latency_p99": 300,
            "success_rate": 0.99,
            "throughput": 1000,
            "cpu_usage": 30,
            "memory_usage": 45,
        }

        # In real implementation, collect from monitoring system
        return metrics

    async def check_safety_conditions(self, metrics: Dict[str, Any]) -> bool:
        """Check if system is in safe state for chaos testing"""
        safety = self.config["safety"]

        # Check error rate
        if metrics["error_rate"] > safety["max_error_rate"]:
            self.logger.warning(f"Error rate too high: {metrics['error_rate']}")
            return False

        # Check success rate
        if metrics["success_rate"] < safety["min_success_rate"]:
            self.logger.warning(f"Success rate too low: {metrics['success_rate']}")
            return False

        # Check latency
        if metrics["latency_p99"] > safety["max_latency_ms"]:
            self.logger.warning(f"Latency too high: {metrics['latency_p99']}ms")
            return False

        return True

    async def monitor_system(self):
        """Continuously monitor system health during experiments"""
        interval = self.config["monitoring"]["interval_seconds"]

        while not self.should_stop:
            try:
                # Collect current metrics
                metrics = await self.collect_baseline_metrics()

                # Check for critical issues
                if not await self.check_safety_conditions(metrics):
                    self.logger.error(
                        "Critical safety threshold breached during experiment!"
                    )

                    # Trigger emergency stop if configured
                    if self.config["safety"]["auto_rollback"]:
                        self.logger.info("Initiating emergency rollback")
                        await self.chaos_engineer.emergency_stop()

                await asyncio.sleep(interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(interval)

    async def generate_report(self):
        """Generate chaos testing report"""
        report_dir = Path(self.config["reporting"]["output_dir"])
        report_dir.mkdir(parents=True, exist_ok=True)

        duration = (datetime.now() - self.start_time).total_seconds() / 60

        report = {
            "summary": {
                "start_time": self.start_time.isoformat(),
                "duration_minutes": round(duration, 2),
                "total_experiments": self.experiments_run,
                "passed": self.experiments_passed,
                "failed": self.experiments_failed,
                "success_rate": (
                    self.experiments_passed / self.experiments_run * 100
                    if self.experiments_run > 0
                    else 0
                ),
            },
            "configuration": self.config,
            "experiments": [],  # Would include detailed results in production
        }

        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = report_dir / f"chaos_report_{timestamp}.json"

        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"Report saved to: {report_file}")

        # Print summary
        print("\n" + "=" * 60)
        print("CHAOS ENGINEERING TEST SUMMARY")
        print("=" * 60)
        print(f"Duration: {duration:.2f} minutes")
        print(f"Total Experiments: {self.experiments_run}")
        print(f"Passed: {self.experiments_passed}")
        print(f"Failed: {self.experiments_failed}")
        print(f"Success Rate: {report['summary']['success_rate']:.1f}%")
        print("=" * 60)


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Chaos Engineering Test Runner for Homeostasis"
    )

    parser.add_argument(
        "--config", "-c", help="Configuration file path (JSON or YAML)", default=None
    )

    parser.add_argument(
        "--experiments",
        "-e",
        nargs="+",
        help="Specific experiment files to run",
        default=None,
    )

    parser.add_argument(
        "--categories",
        nargs="+",
        choices=["network", "resource", "service"],
        help="Categories of experiments to run",
        default=None,
    )

    parser.add_argument(
        "--intensity",
        choices=["low", "medium", "high"],
        help="Intensity of chaos experiments",
        default="medium",
    )

    parser.add_argument(
        "--duration", type=int, help="Total duration in minutes", default=30
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate experiments without running them",
    )

    args = parser.parse_args()

    # Create runner
    runner = ChaosTestRunner(args.config)

    # Override config with command line args
    if args.categories:
        runner.config["experiments"]["categories"] = args.categories

    if args.intensity:
        runner.config["experiments"]["intensity"] = args.intensity

    if args.duration:
        runner.config["experiments"]["duration_minutes"] = args.duration

    # Run experiments
    if args.dry_run:
        print("DRY RUN MODE - Validating experiments only")
        experiments = await runner.load_experiments(args.experiments)
        print(f"Would run {len(experiments)} experiments:")
        for exp in experiments:
            print(f"  - {exp.name}: {exp.description}")
    else:
        await runner.run(args.experiments)


if __name__ == "__main__":
    asyncio.run(main())
