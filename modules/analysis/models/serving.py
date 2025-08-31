"""
Model serving architecture for real-time inference in Homeostasis.

This module provides:
- High-performance model serving with caching and batching
- REST API and gRPC endpoints
- Model hot-swapping and A/B testing
- Request preprocessing and response postprocessing
- Monitoring and observability
- Auto-scaling based on load
"""
import json
import time
import asyncio
import threading
import queue
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import logging

# Web framework imports
try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from pydantic import BaseModel, Field
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

try:
    pass  # grpc not actually used
    GRPC_AVAILABLE = False
except ImportError:
    GRPC_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    from prometheus_client import Counter, Histogram, Gauge
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# Import other modules
from .versioning import ModelVersionControl
from .code_features import MultiLanguageFeatureExtractor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ServingConfig:
    """Configuration for model serving."""
    model_name: str
    model_version: Optional[str] = None  # None for latest
    
    # Performance settings
    batch_size: int = 32
    batch_timeout_ms: int = 100
    max_batch_wait_ms: int = 500
    num_worker_threads: int = 4
    
    # Caching
    enable_cache: bool = True
    cache_ttl_seconds: int = 3600
    cache_backend: str = "memory"  # memory, redis
    
    # API settings
    enable_rest_api: bool = True
    enable_grpc: bool = False
    api_port: int = 8080
    grpc_port: int = 50051
    
    # Monitoring
    enable_metrics: bool = True
    metrics_port: int = 9090
    
    # A/B testing
    enable_ab_testing: bool = False
    ab_test_versions: List[str] = field(default_factory=list)
    ab_test_weights: List[float] = field(default_factory=list)
    
    # Auto-scaling
    enable_autoscaling: bool = False
    min_replicas: int = 1
    max_replicas: int = 10
    target_qps_per_replica: int = 100


@dataclass
class InferenceRequest:
    """Request for model inference."""
    request_id: str
    data: Dict[str, Any]
    model_version: Optional[str] = None
    timeout_ms: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceResponse:
    """Response from model inference."""
    request_id: str
    prediction: Any
    model_version: str
    inference_time_ms: float
    cached: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModelCache:
    """Caching layer for model predictions."""
    
    def __init__(self, config: ServingConfig):
        """Initialize the cache."""
        self.config = config
        self.backend = config.cache_backend
        
        if self.backend == "memory":
            self.cache = {}
            self.cache_times = {}
        elif self.backend == "redis" and REDIS_AVAILABLE:
            self.redis_client = redis.Redis(
                host='localhost',
                port=6379,
                decode_responses=True
            )
        else:
            logger.warning(f"Cache backend {self.backend} not available, using memory cache")
            self.cache = {}
            self.cache_times = {}
            self.backend = "memory"
    
    def _generate_cache_key(self, data: Dict[str, Any], model_version: str) -> str:
        """Generate a cache key from request data."""
        import hashlib
        
        # Create a stable string representation
        data_str = json.dumps(data, sort_keys=True)
        combined = f"{model_version}:{data_str}"
        
        # Hash it
        return hashlib.md5(combined.encode()).hexdigest()
    
    def get(self, data: Dict[str, Any], model_version: str) -> Optional[Any]:
        """Get prediction from cache."""
        if not self.config.enable_cache:
            return None
        
        key = self._generate_cache_key(data, model_version)
        
        if self.backend == "memory":
            # Check if cached and not expired
            if key in self.cache:
                cache_time = self.cache_times.get(key, 0)
                if time.time() - cache_time < self.config.cache_ttl_seconds:
                    return self.cache[key]
                else:
                    # Expired, remove from cache
                    del self.cache[key]
                    del self.cache_times[key]
        
        elif self.backend == "redis":
            try:
                cached_value = self.redis_client.get(key)
                if cached_value:
                    return json.loads(cached_value)
            except Exception as e:
                logger.error(f"Redis cache get error: {e}")
        
        return None
    
    def set(self, data: Dict[str, Any], model_version: str, prediction: Any):
        """Set prediction in cache."""
        if not self.config.enable_cache:
            return
        
        key = self._generate_cache_key(data, model_version)
        
        if self.backend == "memory":
            self.cache[key] = prediction
            self.cache_times[key] = time.time()
            
            # Simple LRU eviction if cache gets too large
            if len(self.cache) > 10000:
                # Remove oldest entries
                sorted_keys = sorted(self.cache_times.items(), key=lambda x: x[1])
                for old_key, _ in sorted_keys[:1000]:
                    del self.cache[old_key]
                    del self.cache_times[old_key]
        
        elif self.backend == "redis":
            try:
                self.redis_client.setex(
                    key,
                    self.config.cache_ttl_seconds,
                    json.dumps(prediction)
                )
            except Exception as e:
                logger.error(f"Redis cache set error: {e}")
    
    def clear(self):
        """Clear the cache."""
        if self.backend == "memory":
            self.cache.clear()
            self.cache_times.clear()
        elif self.backend == "redis":
            try:
                self.redis_client.flushdb()
            except Exception as e:
                logger.error(f"Redis cache clear error: {e}")


class BatchProcessor:
    """Batch processing for efficient inference."""
    
    def __init__(self, model: Any, config: ServingConfig):
        """Initialize the batch processor."""
        self.model = model
        self.config = config
        self.request_queue = queue.Queue()
        self.processing = True
        self.executor = ThreadPoolExecutor(max_workers=config.num_worker_threads)
        
        # Start batch processing thread
        self.batch_thread = threading.Thread(target=self._batch_processing_loop)
        self.batch_thread.daemon = True
        self.batch_thread.start()
    
    def _batch_processing_loop(self):
        """Main batch processing loop."""
        while self.processing:
            batch = []
            batch_start_time = time.time()
            
            # Collect requests up to batch size or timeout
            while len(batch) < self.config.batch_size:
                timeout = self.config.batch_timeout_ms / 1000.0
                remaining_timeout = timeout - (time.time() - batch_start_time)
                
                if remaining_timeout <= 0:
                    break
                
                try:
                    request, future = self.request_queue.get(timeout=remaining_timeout)
                    batch.append((request, future))
                except queue.Empty:
                    break
                
                # Check max wait time
                if (time.time() - batch_start_time) * 1000 >= self.config.max_batch_wait_ms:
                    break
            
            # Process batch if not empty
            if batch:
                self._process_batch(batch)
    
    def _process_batch(self, batch: List[Tuple[InferenceRequest, asyncio.Future]]):
        """Process a batch of requests."""
        try:
            # Extract data from requests
            batch_data = [req.data for req, _ in batch]
            
            # Perform batch inference
            start_time = time.time()
            predictions = self._batch_predict(batch_data)
            inference_time = (time.time() - start_time) * 1000
            
            # Send responses
            for i, (request, future) in enumerate(batch):
                response = InferenceResponse(
                    request_id=request.request_id,
                    prediction=predictions[i],
                    model_version=self.model_version,
                    inference_time_ms=inference_time / len(batch),
                    cached=False
                )
                
                future.set_result(response)
                
        except Exception as e:
            # Send error to all requests in batch
            for request, future in batch:
                future.set_exception(e)
    
    def _batch_predict(self, batch_data: List[Dict[str, Any]]) -> List[Any]:
        """Perform batch prediction."""
        # This is a placeholder - actual implementation depends on model type
        # For sklearn models, we might need to vectorize the data first
        
        # Convert batch data to appropriate format
        if hasattr(self.model, 'predict_proba'):
            # Classification model
            return self.model.predict_proba(batch_data).tolist()
        else:
            # Regression or other model
            return self.model.predict(batch_data).tolist()
    
    async def process_request(self, request: InferenceRequest) -> InferenceResponse:
        """Process a single request (may be batched)."""
        future = asyncio.Future()
        self.request_queue.put((request, future))
        return await future
    
    def shutdown(self):
        """Shutdown the batch processor."""
        self.processing = False
        self.batch_thread.join()
        self.executor.shutdown()


class ModelServer:
    """Main model serving class."""
    
    def __init__(self, config: ServingConfig):
        """Initialize the model server."""
        self.config = config
        self.version_control = ModelVersionControl()
        self.feature_extractor = MultiLanguageFeatureExtractor()
        self.cache = ModelCache(config)
        
        # Load model(s)
        self.models = {}
        self.batch_processors = {}
        self._load_models()
        
        # Initialize metrics if enabled
        if config.enable_metrics and PROMETHEUS_AVAILABLE:
            self._init_metrics()
        
        # Request tracking
        self.active_requests = 0
        self.total_requests = 0
        self.request_history = deque(maxlen=1000)
    
    def _load_models(self):
        """Load models based on configuration."""
        if self.config.enable_ab_testing and self.config.ab_test_versions:
            # Load multiple versions for A/B testing
            for version in self.config.ab_test_versions:
                self._load_model_version(version)
        else:
            # Load single model version
            if self.config.model_version:
                self._load_model_version(self.config.model_version)
            else:
                # Load latest production model
                versions = self.version_control.list_versions(
                    model_name=self.config.model_name,
                    status="production"
                )
                if not versions:
                    versions = self.version_control.list_versions(
                        model_name=self.config.model_name
                    )
                
                if versions:
                    latest_version = versions[0].version_id
                    self._load_model_version(latest_version)
                else:
                    raise ValueError(f"No models found for {self.config.model_name}")
    
    def _load_model_version(self, version_id: str):
        """Load a specific model version."""
        logger.info(f"Loading model version: {version_id}")
        
        model = self.version_control.load_model(version_id)
        self.models[version_id] = model
        
        # Create batch processor
        self.batch_processors[version_id] = BatchProcessor(model, self.config)
    
    def _init_metrics(self):
        """Initialize Prometheus metrics."""
        self.request_counter = Counter(
            'model_requests_total',
            'Total number of requests',
            ['model_name', 'model_version', 'status']
        )
        
        self.request_latency = Histogram(
            'model_request_latency_seconds',
            'Request latency in seconds',
            ['model_name', 'model_version']
        )
        
        self.cache_hits = Counter(
            'model_cache_hits_total',
            'Total number of cache hits',
            ['model_name', 'model_version']
        )
        
        self.active_requests_gauge = Gauge(
            'model_active_requests',
            'Number of active requests',
            ['model_name']
        )
        
        self.model_errors = Counter(
            'model_errors_total',
            'Total number of errors',
            ['model_name', 'model_version', 'error_type']
        )
    
    def _select_model_version(self, request: InferenceRequest) -> str:
        """Select model version for request (A/B testing logic)."""
        if request.model_version and request.model_version in self.models:
            return request.model_version
        
        if self.config.enable_ab_testing and len(self.config.ab_test_versions) > 1:
            # Random selection based on weights
            import random
            
            if self.config.ab_test_weights:
                weights = self.config.ab_test_weights
            else:
                # Equal weights
                weights = [1.0 / len(self.config.ab_test_versions)] * len(self.config.ab_test_versions)
            
            return random.choices(self.config.ab_test_versions, weights=weights)[0]
        
        # Return first (or only) loaded model
        return list(self.models.keys())[0]
    
    async def predict(self, request: InferenceRequest) -> InferenceResponse:
        """Perform prediction for a request."""
        self.active_requests += 1
        self.total_requests += 1
        
        if self.config.enable_metrics:
            self.active_requests_gauge.labels(
                model_name=self.config.model_name
            ).set(self.active_requests)
        
        start_time = time.time()
        
        try:
            # Select model version
            model_version = self._select_model_version(request)
            
            # Check cache
            cached_result = self.cache.get(request.data, model_version)
            if cached_result is not None:
                if self.config.enable_metrics:
                    self.cache_hits.labels(
                        model_name=self.config.model_name,
                        model_version=model_version
                    ).inc()
                
                response = InferenceResponse(
                    request_id=request.request_id,
                    prediction=cached_result,
                    model_version=model_version,
                    inference_time_ms=0,
                    cached=True
                )
            else:
                # Process through batch processor
                batch_processor = self.batch_processors[model_version]
                response = await batch_processor.process_request(request)
                
                # Cache the result
                self.cache.set(request.data, model_version, response.prediction)
            
            # Record metrics
            if self.config.enable_metrics:
                elapsed_time = time.time() - start_time
                
                self.request_counter.labels(
                    model_name=self.config.model_name,
                    model_version=model_version,
                    status="success"
                ).inc()
                
                self.request_latency.labels(
                    model_name=self.config.model_name,
                    model_version=model_version
                ).observe(elapsed_time)
            
            # Track request
            self.request_history.append({
                "request_id": request.request_id,
                "timestamp": time.time(),
                "model_version": model_version,
                "cached": response.cached,
                "latency_ms": (time.time() - start_time) * 1000
            })
            
            return response
            
        except Exception as e:
            if self.config.enable_metrics:
                self.model_errors.labels(
                    model_name=self.config.model_name,
                    model_version=model_version if 'model_version' in locals() else "unknown",
                    error_type=type(e).__name__
                ).inc()
            
            logger.error(f"Prediction error: {e}")
            raise
        
        finally:
            self.active_requests -= 1
            if self.config.enable_metrics:
                self.active_requests_gauge.labels(
                    model_name=self.config.model_name
                ).set(self.active_requests)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        stats = {
            "total_requests": self.total_requests,
            "active_requests": self.active_requests,
            "loaded_models": list(self.models.keys()),
            "cache_enabled": self.config.enable_cache,
            "batch_size": self.config.batch_size
        }
        
        # Calculate recent request rate
        if self.request_history:
            recent_requests = [r for r in self.request_history 
                             if time.time() - r["timestamp"] < 60]
            stats["requests_per_minute"] = len(recent_requests)
            
            # Average latency
            if recent_requests:
                avg_latency = np.mean([r["latency_ms"] for r in recent_requests])
                stats["avg_latency_ms"] = float(avg_latency)
                
                # Cache hit rate
                cache_hits = sum(1 for r in recent_requests if r.get("cached", False))
                stats["cache_hit_rate"] = cache_hits / len(recent_requests)
        
        return stats
    
    def update_model(self, new_version: str):
        """Hot-swap to a new model version."""
        logger.info(f"Updating model to version: {new_version}")
        
        # Load new model
        self._load_model_version(new_version)
        
        # If not A/B testing, remove old versions
        if not self.config.enable_ab_testing:
            old_versions = [v for v in self.models.keys() if v != new_version]
            for old_version in old_versions:
                # Shutdown batch processor
                self.batch_processors[old_version].shutdown()
                
                # Remove model
                del self.models[old_version]
                del self.batch_processors[old_version]
        
        logger.info(f"Model updated successfully to version: {new_version}")
    
    def shutdown(self):
        """Shutdown the model server."""
        logger.info("Shutting down model server")
        
        # Shutdown all batch processors
        for processor in self.batch_processors.values():
            processor.shutdown()


# REST API implementation
if FASTAPI_AVAILABLE:
    class PredictionRequest(BaseModel):
        """API request model."""
        data: Dict[str, Any]
        model_version: Optional[str] = None
        timeout_ms: Optional[int] = Field(None, ge=0, le=60000)
    
    class PredictionResponse(BaseModel):
        """API response model."""
        request_id: str
        prediction: Any
        model_version: str
        inference_time_ms: float
        cached: bool
    
    class StatsResponse(BaseModel):
        """API stats response model."""
        total_requests: int
        active_requests: int
        loaded_models: List[str]
        requests_per_minute: Optional[float]
        avg_latency_ms: Optional[float]
        cache_hit_rate: Optional[float]
    
    def create_app(server: ModelServer) -> FastAPI:
        """Create FastAPI application."""
        app = FastAPI(title="Homeostasis Model Server")
        
        @app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {"status": "healthy", "models": list(server.models.keys())}
        
        @app.post("/predict", response_model=PredictionResponse)
        async def predict(request: PredictionRequest):
            """Prediction endpoint."""
            import uuid
            
            # Create internal request
            internal_request = InferenceRequest(
                request_id=str(uuid.uuid4()),
                data=request.data,
                model_version=request.model_version,
                timeout_ms=request.timeout_ms
            )
            
            try:
                # Get prediction
                response = await server.predict(internal_request)
                
                return PredictionResponse(
                    request_id=response.request_id,
                    prediction=response.prediction,
                    model_version=response.model_version,
                    inference_time_ms=response.inference_time_ms,
                    cached=response.cached
                )
            
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/stats", response_model=StatsResponse)
        async def get_stats():
            """Get server statistics."""
            stats = server.get_stats()
            return StatsResponse(**stats)
        
        @app.post("/update_model")
        async def update_model(new_version: str, background_tasks: BackgroundTasks):
            """Update model version."""
            background_tasks.add_task(server.update_model, new_version)
            return {"message": f"Model update to {new_version} initiated"}
        
        if server.config.enable_metrics and PROMETHEUS_AVAILABLE:
            @app.get("/metrics")
            async def metrics():
                """Prometheus metrics endpoint."""
                from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
                from fastapi.responses import Response
                
                return Response(
                    generate_latest(),
                    media_type=CONTENT_TYPE_LATEST
                )
        
        return app


class ModelServingOrchestrator:
    """Orchestrator for managing multiple model servers."""
    
    def __init__(self):
        """Initialize the orchestrator."""
        self.servers = {}
        self.app = None
    
    def add_model(self, config: ServingConfig):
        """Add a model to be served."""
        server = ModelServer(config)
        self.servers[config.model_name] = server
        
        logger.info(f"Added model server for {config.model_name}")
    
    def create_unified_api(self) -> FastAPI:
        """Create unified API for all models."""
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI not available")
        
        app = FastAPI(title="Homeostasis Model Serving")
        
        @app.get("/health")
        async def health_check():
            """Health check for all models."""
            health = {}
            for name, server in self.servers.items():
                health[name] = {
                    "status": "healthy",
                    "models": list(server.models.keys())
                }
            return health
        
        @app.post("/predict/{model_name}")
        async def predict(model_name: str, request: PredictionRequest):
            """Unified prediction endpoint."""
            if model_name not in self.servers:
                raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
            
            import uuid
            
            server = self.servers[model_name]
            internal_request = InferenceRequest(
                request_id=str(uuid.uuid4()),
                data=request.data,
                model_version=request.model_version,
                timeout_ms=request.timeout_ms
            )
            
            try:
                response = await server.predict(internal_request)
                
                return PredictionResponse(
                    request_id=response.request_id,
                    prediction=response.prediction,
                    model_version=response.model_version,
                    inference_time_ms=response.inference_time_ms,
                    cached=response.cached
                )
            
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/models")
        async def list_models():
            """List all available models."""
            models = {}
            for name, server in self.servers.items():
                models[name] = {
                    "versions": list(server.models.keys()),
                    "stats": server.get_stats()
                }
            return models
        
        self.app = app
        return app
    
    def run(self, host: str = "0.0.0.0", port: int = 8080):
        """Run the unified API server."""
        if not self.app:
            self.app = self.create_unified_api()
        
        uvicorn.run(self.app, host=host, port=port)
    
    def shutdown(self):
        """Shutdown all model servers."""
        for server in self.servers.values():
            server.shutdown()


# Auto-scaling manager
class AutoScaler:
    """Auto-scaling manager for model servers."""
    
    def __init__(self, orchestrator: ModelServingOrchestrator):
        """Initialize the auto-scaler."""
        self.orchestrator = orchestrator
        self.scaling_policies = {}
        self.monitoring = True
        self.monitor_thread = None
    
    def add_scaling_policy(self, model_name: str, config: ServingConfig):
        """Add scaling policy for a model."""
        if config.enable_autoscaling:
            self.scaling_policies[model_name] = {
                "config": config,
                "current_replicas": 1,
                "metrics_history": deque(maxlen=60)  # 1 minute of history
            }
    
    def start_monitoring(self):
        """Start monitoring loop."""
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            for model_name, policy in self.scaling_policies.items():
                if model_name in self.orchestrator.servers:
                    server = self.orchestrator.servers[model_name]
                    stats = server.get_stats()
                    
                    # Record metrics
                    policy["metrics_history"].append({
                        "timestamp": time.time(),
                        "qps": stats.get("requests_per_minute", 0) / 60,
                        "active_requests": stats.get("active_requests", 0),
                        "avg_latency_ms": stats.get("avg_latency_ms", 0)
                    })
                    
                    # Make scaling decision
                    self._make_scaling_decision(model_name, policy)
            
            time.sleep(10)  # Check every 10 seconds
    
    def _make_scaling_decision(self, model_name: str, policy: Dict[str, Any]):
        """Make scaling decision based on metrics."""
        if len(policy["metrics_history"]) < 3:
            return  # Not enough data
        
        config = policy["config"]
        current_replicas = policy["current_replicas"]
        
        # Calculate average QPS over last 30 seconds
        recent_metrics = [m for m in policy["metrics_history"] 
                         if time.time() - m["timestamp"] < 30]
        
        if recent_metrics:
            avg_qps = np.mean([m["qps"] for m in recent_metrics])
            
            # Calculate desired replicas
            desired_replicas = int(np.ceil(avg_qps / config.target_qps_per_replica))
            desired_replicas = max(config.min_replicas, 
                                  min(config.max_replicas, desired_replicas))
            
            # Scale if needed
            if desired_replicas != current_replicas:
                logger.info(f"Scaling {model_name} from {current_replicas} to {desired_replicas} replicas")
                self._scale_model(model_name, desired_replicas)
                policy["current_replicas"] = desired_replicas
    
    def _scale_model(self, model_name: str, replicas: int):
        """Scale model to desired number of replicas."""
        # This is a placeholder - actual implementation would depend on
        # the deployment platform (K8s, Docker Swarm, etc.)
        logger.info(f"Scaling {model_name} to {replicas} replicas")
    
    def stop_monitoring(self):
        """Stop monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()


if __name__ == "__main__":
    # Example usage
    import asyncio
    
    # Create serving configuration
    config = ServingConfig(
        model_name="error_classifier",
        model_version=None,  # Use latest
        batch_size=32,
        batch_timeout_ms=100,
        enable_cache=True,
        enable_metrics=True,
        enable_ab_testing=False
    )
    
    # Create model server
    server = ModelServer(config)
    
    # Test prediction
    async def test_prediction():
        request = InferenceRequest(
            request_id="test_123",
            data={
                "error_type": "KeyError",
                "message": "key 'user_id' not found",
                "file": "app.py",
                "line": 42
            }
        )
        
        response = await server.predict(request)
        print(f"Prediction: {response.prediction}")
        print(f"Model version: {response.model_version}")
        print(f"Inference time: {response.inference_time_ms:.2f} ms")
        print(f"Cached: {response.cached}")
    
    # Run test
    asyncio.run(test_prediction())
    
    # Create orchestrator for multiple models
    orchestrator = ModelServingOrchestrator()
    orchestrator.add_model(config)
    
    # Add another model
    config2 = ServingConfig(
        model_name="complexity_analyzer",
        enable_cache=True,
        enable_metrics=True
    )
    
    # orchestrator.add_model(config2)
    
    # Create and run API
    if FASTAPI_AVAILABLE:
        logger.info("Starting model serving API...")
        app = orchestrator.create_unified_api()
        
        # Run with: uvicorn serving:app --reload
        # orchestrator.run(port=8080)
    
    # Cleanup
    server.shutdown()