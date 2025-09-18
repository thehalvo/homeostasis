"""
Enhanced distributed training capabilities for Homeostasis ML models.

This module provides advanced distributed training features:
- Multi-GPU training with data parallelism
- Distributed data loading and preprocessing
- Gradient accumulation and synchronization
- Fault tolerance and checkpointing
- Support for multiple backends (Dask, Ray, Horovod, PyTorch DDP)
"""

import logging
import time
from concurrent.futures import as_completed
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import numpy as np

# Optional distributed computing imports
try:
    from dask.distributed import Client

    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

try:
    import ray

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

try:
    import horovod.torch as hvd

    HOROVOD_AVAILABLE = True
except ImportError:
    HOROVOD_AVAILABLE = False

try:
    import torch
    import torch.distributed as dist
    import torch.nn as nn
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data.distributed import DistributedSampler

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    pass  # MPI not actually used
    MPI_AVAILABLE = False
except ImportError:
    MPI_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DistributedConfig:
    """Configuration for distributed training."""

    backend: str = "auto"  # auto, dask, ray, horovod, torch_ddp, mpi
    num_workers: int = -1  # -1 for auto-detect
    num_gpus: int = 0  # Number of GPUs per worker
    batch_size_per_worker: int = 32
    gradient_accumulation_steps: int = 1
    mixed_precision: bool = False
    checkpoint_frequency: int = 10  # Checkpoint every N epochs
    fault_tolerance: bool = True
    communication_backend: str = "nccl"  # nccl, gloo, mpi
    data_parallel_strategy: str = "split"  # split, replicate
    model_parallel: bool = False
    pipeline_parallel: bool = False
    zero_optimization: int = 0  # ZeRO optimization level (0, 1, 2, 3)


class DistributedDataLoader:
    """Distributed data loader with efficient sharding."""

    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        batch_size: int,
        num_workers: int,
        rank: int,
    ):
        """Initialize distributed data loader."""
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.rank = rank

        # Calculate data shards
        self.total_samples = len(data)
        self.samples_per_worker = self.total_samples // num_workers
        self.start_idx = rank * self.samples_per_worker
        self.end_idx = (
            (rank + 1) * self.samples_per_worker
            if rank < num_workers - 1
            else self.total_samples
        )

        # Get local data shard
        self.local_data = data[self.start_idx : self.end_idx]
        self.local_labels = labels[self.start_idx : self.end_idx]
        self.local_samples = len(self.local_data)

        logger.info(
            f"Worker {rank}: Processing samples {self.start_idx} to {self.end_idx}"
        )

    def get_batches(self):
        """Generate batches for this worker."""
        indices = np.random.permutation(self.local_samples)

        for start_idx in range(0, self.local_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.local_samples)
            batch_indices = indices[start_idx:end_idx]

            yield (self.local_data[batch_indices], self.local_labels[batch_indices])

    def __len__(self):
        """Get number of batches."""
        return (self.local_samples + self.batch_size - 1) // self.batch_size


class DaskDistributedTrainer:
    """Distributed trainer using Dask."""

    def __init__(self, config: DistributedConfig):
        """Initialize Dask distributed trainer."""
        self.config = config
        self.client = None
        self._setup_client()

    def _setup_client(self):
        """Set up Dask client."""
        if not DASK_AVAILABLE:
            raise ImportError(
                "Dask not installed. Install with: pip install dask[distributed]"
            )

        # Set up client with appropriate configuration
        try:
            if self.config.num_workers == -1:
                # Use local cluster with auto-detected workers
                from dask.distributed import LocalCluster

                cluster = LocalCluster(
                    n_workers=None, threads_per_worker=1, processes=True  # Auto-detect
                )
                self.client = Client(cluster)
            else:
                # Assume scheduler is already running
                self.client = Client()

            logger.info(f"Dask client initialized: {self.client}")
            if self.client:
                logger.info(f"Dashboard: {self.client.dashboard_link}")
        except Exception as e:
            logger.error(f"Failed to initialize Dask client: {e}")
            self.client = None

    def train_distributed(
        self,
        train_func: Callable,
        data: np.ndarray,
        labels: np.ndarray,
        model_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute distributed training with Dask."""
        # Get number of workers
        if not self.client:
            raise RuntimeError("Dask client not initialized")

        num_workers = len(self.client.scheduler_info()["workers"])
        logger.info(f"Training on {num_workers} Dask workers")

        # Scatter data to workers
        data_futures = []
        label_futures = []

        chunk_size = len(data) // num_workers
        for i in range(num_workers):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size if i < num_workers - 1 else len(data)

            data_chunk = data[start_idx:end_idx]
            label_chunk = labels[start_idx:end_idx]

            # Scatter data chunks to workers
            data_future = self.client.scatter(data_chunk, broadcast=False)
            label_future = self.client.scatter(label_chunk, broadcast=False)

            data_futures.append(data_future)
            label_futures.append(label_future)

        # Submit training tasks
        futures = []
        for i, (data_future, label_future) in enumerate(
            zip(data_futures, label_futures)
        ):
            future = self.client.submit(
                train_func,
                data_future,
                label_future,
                model_params,
                i,  # worker rank
                num_workers,
                pure=False,
            )
            futures.append(future)

        # Gather results
        results = []
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Worker failed with error: {e}")
                if not self.config.fault_tolerance:
                    raise

        # Aggregate results
        aggregated_result = self._aggregate_results(results)

        return aggregated_result

    def _aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results from multiple workers."""
        if not results:
            return {}

        # Average metrics
        aggregated = {
            "loss": np.mean([r.get("loss", 0) for r in results]),
            "accuracy": np.mean([r.get("accuracy", 0) for r in results]),
            "num_samples": sum([r.get("num_samples", 0) for r in results]),
            "training_time": max([r.get("training_time", 0) for r in results]),
        }

        # Combine model parameters (average)
        if "model_state" in results[0]:
            model_states = [r["model_state"] for r in results]
            aggregated["model_state"] = self._average_model_states(model_states)

        return aggregated

    def _average_model_states(
        self, model_states: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Average model parameters across workers."""
        if not model_states:
            return {}

        averaged_state = {}

        # Get all parameter names
        param_names = list(model_states[0].keys())

        for param_name in param_names:
            # Stack parameters from all workers
            param_stack = [state[param_name] for state in model_states]

            # Average parameters
            if isinstance(param_stack[0], np.ndarray):
                averaged_state[param_name] = np.mean(param_stack, axis=0)
            elif hasattr(param_stack[0], "numpy"):  # PyTorch tensor
                param_array = np.stack([p.numpy() for p in param_stack])
                averaged_state[param_name] = np.mean(param_array, axis=0)
            else:
                # For non-array parameters, just use the first one
                averaged_state[param_name] = param_stack[0]

        return averaged_state

    def shutdown(self):
        """Shutdown Dask client."""
        if self.client:
            self.client.close()


class RayDistributedTrainer:
    """Distributed trainer using Ray."""

    def __init__(self, config: DistributedConfig):
        """Initialize Ray distributed trainer."""
        self.config = config
        self._setup_ray()

    def _setup_ray(self):
        """Set up Ray runtime."""
        if not RAY_AVAILABLE:
            raise ImportError("Ray not installed. Install with: pip install ray")

        # Initialize Ray
        if not ray.is_initialized():
            ray.init(
                num_cpus=(
                    None if self.config.num_workers == -1 else self.config.num_workers
                ),
                num_gpus=self.config.num_gpus if self.config.num_gpus > 0 else None,
                ignore_reinit_error=True,
            )

        logger.info(f"Ray initialized with resources: {ray.available_resources()}")

    @ray.remote
    def _train_worker(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        model_params: Dict[str, Any],
        rank: int,
        num_workers: int,
    ) -> Dict[str, Any]:
        """Training function for Ray worker."""
        # This would be replaced with actual training logic
        logger.info(f"Ray worker {rank} starting training")

        # Simulate training
        time.sleep(2)

        return {
            "loss": np.random.random(),
            "accuracy": np.random.random(),
            "num_samples": len(data),
            "training_time": 2.0,
            "rank": rank,
        }

    def train_distributed(
        self,
        train_func: Callable,
        data: np.ndarray,
        labels: np.ndarray,
        model_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute distributed training with Ray."""
        # Determine number of workers
        available_cpus = int(ray.available_resources().get("CPU", 1))
        num_workers = (
            self.config.num_workers if self.config.num_workers > 0 else available_cpus
        )

        logger.info(f"Training on {num_workers} Ray workers")

        # Put data in Ray object store
        data_id = ray.put(data)
        labels_id = ray.put(labels)
        model_params_id = ray.put(model_params)

        # Create remote training function
        @ray.remote
        def ray_train_func(data_chunk, labels_chunk, params, rank, total_workers):
            return train_func(data_chunk, labels_chunk, params, rank, total_workers)

        # Split data and submit tasks
        futures = []
        chunk_size = len(data) // num_workers

        for i in range(num_workers):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size if i < num_workers - 1 else len(data)

            # Get data chunks
            data_chunk = ray.get(data_id)[start_idx:end_idx]
            labels_chunk = ray.get(labels_id)[start_idx:end_idx]

            # Submit training task
            future = ray_train_func.remote(
                data_chunk, labels_chunk, model_params_id, i, num_workers
            )
            futures.append(future)

        # Wait for results
        results = ray.get(futures)

        # Aggregate results
        aggregated_result = self._aggregate_results(results)

        return aggregated_result

    def _aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results from Ray workers."""
        return {
            "loss": np.mean([r.get("loss", 0) for r in results]),
            "accuracy": np.mean([r.get("accuracy", 0) for r in results]),
            "num_samples": sum([r.get("num_samples", 0) for r in results]),
            "training_time": max([r.get("training_time", 0) for r in results]),
            "num_workers": len(results),
        }

    def shutdown(self):
        """Shutdown Ray."""
        if ray.is_initialized():
            ray.shutdown()


class HorovodDistributedTrainer:
    """Distributed trainer using Horovod."""

    def __init__(self, config: DistributedConfig):
        """Initialize Horovod distributed trainer."""
        self.config = config
        self._setup_horovod()

    def _setup_horovod(self):
        """Set up Horovod."""
        if not HOROVOD_AVAILABLE:
            raise ImportError(
                "Horovod not installed. Install with: pip install horovod"
            )

        # Initialize Horovod
        hvd.init()

        # Log setup info
        if hvd.rank() == 0:
            logger.info(f"Horovod initialized with {hvd.size()} processes")

        # Set GPU if available
        if self.config.num_gpus > 0 and TORCH_AVAILABLE:
            torch.cuda.set_device(hvd.local_rank())

    def train_distributed(
        self, model: Any, optimizer: Any, data_loader: Any, num_epochs: int
    ) -> Dict[str, Any]:
        """Execute distributed training with Horovod."""
        # Broadcast parameters
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        # Wrap optimizer with Horovod
        optimizer = hvd.DistributedOptimizer(
            optimizer,
            named_parameters=model.named_parameters(),
            compression=(
                hvd.Compression.fp16
                if self.config.mixed_precision
                else hvd.Compression.none
            ),
        )

        # Training loop
        results: Dict[str, List[float]] = {"losses": [], "accuracies": [], "epoch_times": []}

        for epoch in range(num_epochs):
            epoch_start = time.time()
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0

            for batch_idx, (data, target) in enumerate(data_loader):
                # Forward pass
                output = model(data)
                loss = nn.functional.cross_entropy(output, target)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Track metrics
                epoch_loss += loss.item()
                _, predicted = output.max(1)
                epoch_correct += predicted.eq(target).sum().item()
                epoch_total += target.size(0)

            # Average metrics across workers
            epoch_loss = hvd.allreduce(torch.tensor(epoch_loss), average=True)
            epoch_accuracy = hvd.allreduce(
                torch.tensor(epoch_correct / epoch_total), average=True
            )

            epoch_time = time.time() - epoch_start

            # Log progress (only on rank 0)
            if hvd.rank() == 0:
                logger.info(
                    f"Epoch {epoch}: Loss={epoch_loss:.4f}, "
                    f"Accuracy={epoch_accuracy:.4f}, Time={epoch_time:.2f}s"
                )

            results["losses"].append(
                epoch_loss.item() if hasattr(epoch_loss, "item") else epoch_loss
            )
            results["accuracies"].append(
                epoch_accuracy.item()
                if hasattr(epoch_accuracy, "item")
                else epoch_accuracy
            )
            results["epoch_times"].append(epoch_time)

            # Checkpoint
            if hvd.rank() == 0 and epoch % self.config.checkpoint_frequency == 0:
                self._save_checkpoint(model, optimizer, epoch)

        return results

    def _save_checkpoint(self, model: Any, optimizer: Any, epoch: int):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }

        checkpoint_path = f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")


class TorchDDPTrainer:
    """Distributed trainer using PyTorch DDP."""

    def __init__(self, config: DistributedConfig):
        """Initialize PyTorch DDP trainer."""
        self.config = config
        self._setup_ddp()

    def _setup_ddp(self):
        """Set up PyTorch distributed training."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not installed. Install with: pip install torch")

        # Initialize process group
        if not dist.is_initialized():
            dist.init_process_group(
                backend=self.config.communication_backend, init_method="env://"
            )

        # Get rank and world size
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        # Set device
        if self.config.num_gpus > 0 and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.rank % torch.cuda.device_count()}")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        logger.info(
            f"DDP initialized: rank={self.rank}, world_size={self.world_size}, device={self.device}"
        )

    def wrap_model(self, model: nn.Module) -> nn.Module:
        """Wrap model with DDP."""
        model = model.to(self.device)

        if self.config.num_gpus > 0:
            model = DDP(model, device_ids=[self.rank % torch.cuda.device_count()])
        else:
            model = DDP(model)

        return model

    def get_data_loader(self, dataset: Any, batch_size: int) -> Any:
        """Get distributed data loader."""
        sampler: DistributedSampler = DistributedSampler(
            dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True
        )

        from torch.utils.data import DataLoader

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=4,
            pin_memory=True if self.config.num_gpus > 0 else False,
        )

        return loader

    def train_epoch(
        self, model: nn.Module, loader: Any, optimizer: Any, epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch."""
        model.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(self.device), target.to(self.device)

            # Forward pass
            output = model(data)
            loss = nn.functional.cross_entropy(output, target)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                optimizer.step()

            # Track metrics
            total_loss += loss.item() * data.size(0)
            _, predicted = output.max(1)
            total_correct += predicted.eq(target).sum().item()
            total_samples += data.size(0)

        # All-reduce metrics
        metrics_tensor = torch.tensor(
            [total_loss, total_correct, total_samples],
            dtype=torch.float32,
            device=self.device,
        )
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)

        total_loss = metrics_tensor[0].item()
        total_correct = int(metrics_tensor[1].item())
        total_samples = int(metrics_tensor[2].item())

        return {
            "loss": total_loss / total_samples,
            "accuracy": total_correct / total_samples,
        }

    def save_checkpoint(
        self, model: nn.Module, optimizer: Any, epoch: int, metrics: Dict[str, float]
    ):
        """Save training checkpoint."""
        if self.rank == 0:  # Only save on rank 0
            # Get the actual model (unwrap DDP if needed)
            actual_model = model.module if hasattr(model, 'module') else model

            # Ensure we have a Module, not a Tensor
            if not isinstance(actual_model, nn.Module):
                logger.error(f"Expected nn.Module but got {type(actual_model)}")
                return

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": actual_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": metrics,
                "config": self.config.__dict__,
            }

            checkpoint_path = f"ddp_checkpoint_epoch_{epoch}.pt"
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")

    def cleanup(self):
        """Clean up distributed training."""
        if dist.is_initialized():
            dist.destroy_process_group()


class DistributedTrainingOrchestrator:
    """Main orchestrator for distributed training across different backends."""

    def __init__(self, config: DistributedConfig):
        """Initialize the orchestrator."""
        self.config = config
        self.backend = self._select_backend()
        self.trainer = self._create_trainer()

    def _select_backend(self) -> str:
        """Automatically select the best available backend."""
        if self.config.backend != "auto":
            return self.config.backend

        # Priority order based on availability and use case
        if TORCH_AVAILABLE and self.config.num_gpus > 0:
            if HOROVOD_AVAILABLE:
                return "horovod"
            else:
                return "torch_ddp"
        elif RAY_AVAILABLE:
            return "ray"
        elif DASK_AVAILABLE:
            return "dask"
        else:
            logger.warning(
                "No distributed backend available, falling back to local training"
            )
            return "local"

    def _create_trainer(self):
        """Create the appropriate trainer based on backend."""
        if self.backend == "dask":
            return DaskDistributedTrainer(self.config)
        elif self.backend == "ray":
            return RayDistributedTrainer(self.config)
        elif self.backend == "horovod":
            return HorovodDistributedTrainer(self.config)
        elif self.backend == "torch_ddp":
            return TorchDDPTrainer(self.config)
        else:
            return None  # Local training

    def train(
        self, model: Any, data: np.ndarray, labels: np.ndarray, num_epochs: int = 10
    ) -> Dict[str, Any]:
        """Execute distributed training."""
        logger.info(f"Starting distributed training with backend: {self.backend}")

        if self.trainer is None:
            # Fallback to local training
            return self._train_local(model, data, labels, num_epochs)

        # Execute distributed training based on backend
        if isinstance(self.trainer, (DaskDistributedTrainer, RayDistributedTrainer)):
            # These backends use functional training
            def train_func(data_chunk, labels_chunk, model_params, rank, num_workers):
                # This would be replaced with actual model training logic
                return {
                    "loss": np.random.random(),
                    "accuracy": np.random.random(),
                    "num_samples": len(data_chunk),
                    "training_time": 2.0,
                }

            return self.trainer.train_distributed(train_func, data, labels, {})

        else:
            # PyTorch-based backends
            # Convert numpy data to PyTorch dataset
            from torch.utils.data import TensorDataset

            dataset = TensorDataset(
                torch.from_numpy(data).float(), torch.from_numpy(labels).long()
            )

            # Create data loader
            if hasattr(self.trainer, "get_data_loader"):
                loader = self.trainer.get_data_loader(
                    dataset, self.config.batch_size_per_worker
                )
            else:
                from torch.utils.data import DataLoader

                loader = DataLoader(
                    dataset, batch_size=self.config.batch_size_per_worker, shuffle=True
                )

            # Wrap model if needed
            if hasattr(self.trainer, "wrap_model"):
                model = self.trainer.wrap_model(model)

            # Create optimizer
            optimizer = torch.optim.Adam(model.parameters())

            # Training loop
            results: Dict[str, List[float]] = {"losses": [], "accuracies": []}

            for epoch in range(num_epochs):
                if hasattr(self.trainer, "train_epoch"):
                    metrics = self.trainer.train_epoch(model, loader, optimizer, epoch)
                else:
                    # Generic training loop
                    metrics = self._train_epoch_generic(model, loader, optimizer)

                results["losses"].append(metrics["loss"])
                results["accuracies"].append(metrics["accuracy"])

                # Save checkpoint
                if epoch % self.config.checkpoint_frequency == 0:
                    if hasattr(self.trainer, "save_checkpoint"):
                        self.trainer.save_checkpoint(model, optimizer, epoch, metrics)

                logger.info(
                    f"Epoch {epoch}: Loss={metrics['loss']:.4f}, "
                    f"Accuracy={metrics['accuracy']:.4f}"
                )

            return results

    def _train_local(
        self, model: Any, data: np.ndarray, labels: np.ndarray, num_epochs: int
    ) -> Dict[str, Any]:
        """Fallback local training."""
        logger.info("Running local training (no distributed backend)")

        # Simple training loop
        from sklearn.model_selection import train_test_split

        X_train, X_val, y_train, y_val = train_test_split(
            data, labels, test_size=0.2, random_state=42
        )

        # Fit model (assuming sklearn-compatible interface)
        model.fit(X_train, y_train)

        # Evaluate
        train_score = model.score(X_train, y_train)
        val_score = model.score(X_val, y_val)

        return {
            "train_accuracy": train_score,
            "val_accuracy": val_score,
            "backend": "local",
        }

    def _train_epoch_generic(
        self, model: Any, loader: Any, optimizer: Any
    ) -> Dict[str, float]:
        """Generic training epoch implementation."""
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        model.train()

        for data, target in loader:
            # Forward pass
            output = model(data)
            loss = torch.nn.functional.cross_entropy(output, target)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Metrics
            total_loss += loss.item() * data.size(0)
            _, predicted = output.max(1)
            total_correct += predicted.eq(target).sum().item()
            total_samples += data.size(0)

        return {
            "loss": total_loss / total_samples,
            "accuracy": total_correct / total_samples,
        }

    def cleanup(self):
        """Clean up distributed resources."""
        if hasattr(self.trainer, "cleanup"):
            self.trainer.cleanup()
        elif hasattr(self.trainer, "shutdown"):
            self.trainer.shutdown()


# Utility functions
def estimate_optimal_workers(data_size: int, model_complexity: str = "medium") -> int:
    """Estimate optimal number of workers based on data size and model complexity."""
    # Simple heuristic
    complexity_factors = {"simple": 10000, "medium": 5000, "complex": 1000}

    factor = complexity_factors.get(model_complexity, 5000)
    optimal_workers = max(1, min(32, data_size // factor))

    # Check available resources
    if DASK_AVAILABLE:
        try:
            client = Client()
            available_workers = len(client.scheduler_info()["workers"])
            client.close()
            optimal_workers = min(optimal_workers, available_workers)
        except Exception:
            pass

    return optimal_workers


def benchmark_backends(
    data_size: int = 10000, num_features: int = 100
) -> Dict[str, float]:
    """Benchmark available distributed backends."""
    # Generate dummy data
    data = np.random.randn(data_size, num_features)
    labels = np.random.randint(0, 2, data_size)

    results = {}

    # Test each backend
    backends = []
    if DASK_AVAILABLE:
        backends.append("dask")
    if RAY_AVAILABLE:
        backends.append("ray")
    if HOROVOD_AVAILABLE and TORCH_AVAILABLE:
        backends.append("horovod")
    if TORCH_AVAILABLE:
        backends.append("torch_ddp")

    for backend in backends:
        try:
            config = DistributedConfig(backend=backend, num_workers=2)
            orchestrator = DistributedTrainingOrchestrator(config)

            # Time a simple training operation
            start_time = time.time()

            # Create a simple model
            if backend in ["horovod", "torch_ddp"]:
                model = torch.nn.Sequential(
                    torch.nn.Linear(num_features, 64),
                    torch.nn.ReLU(),
                    torch.nn.Linear(64, 2),
                )
            else:
                from sklearn.ensemble import RandomForestClassifier

                model = RandomForestClassifier(n_estimators=10, n_jobs=1)

            # Train for 1 epoch
            orchestrator.train(model, data, labels, num_epochs=1)

            elapsed_time = time.time() - start_time
            results[backend] = elapsed_time

            orchestrator.cleanup()

        except Exception as e:
            logger.error(f"Failed to benchmark {backend}: {e}")
            results[backend] = float("inf")

    return results


if __name__ == "__main__":
    # Example usage
    logger.info("Testing distributed training capabilities")

    # Check available backends
    logger.info(f"Dask available: {DASK_AVAILABLE}")
    logger.info(f"Ray available: {RAY_AVAILABLE}")
    logger.info(f"Horovod available: {HOROVOD_AVAILABLE}")
    logger.info(f"PyTorch available: {TORCH_AVAILABLE}")

    # Benchmark backends
    logger.info("\nBenchmarking available backends...")
    benchmark_results = benchmark_backends(data_size=5000, num_features=50)

    for backend, time_taken in sorted(benchmark_results.items(), key=lambda x: x[1]):
        logger.info(f"{backend}: {time_taken:.2f} seconds")

    # Test distributed training
    if any([DASK_AVAILABLE, RAY_AVAILABLE, HOROVOD_AVAILABLE, TORCH_AVAILABLE]):
        logger.info("\nTesting distributed training...")

        # Create configuration
        config = DistributedConfig(
            backend="auto",
            num_workers=2,
            batch_size_per_worker=32,
            checkpoint_frequency=5,
        )

        # Create orchestrator
        orchestrator = DistributedTrainingOrchestrator(config)

        # Generate dummy data
        data = np.random.randn(1000, 20)
        labels = np.random.randint(0, 2, 1000)

        # Create a simple model
        if TORCH_AVAILABLE:
            model = torch.nn.Sequential(
                torch.nn.Linear(20, 10), torch.nn.ReLU(), torch.nn.Linear(10, 2)
            )
        else:
            from sklearn.ensemble import RandomForestClassifier

            model = RandomForestClassifier(n_estimators=10)

        # Train
        results = orchestrator.train(model, data, labels, num_epochs=3)
        logger.info(f"Training results: {results}")

        # Cleanup
        orchestrator.cleanup()
    else:
        logger.warning("No distributed backends available for testing")
