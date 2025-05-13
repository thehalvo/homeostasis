# Homeostasis Usage Examples

This document provides comprehensive examples of how to use Homeostasis in various real-world scenarios, from basic setup to advanced configurations.

## Basic Examples

### Example 1: Integrating with a Basic Flask Application

```python
# app.py
from flask import Flask, jsonify, request
from modules.monitoring.flask_extension import init_homeostasis

app = Flask(__name__)

# Initialize Homeostasis monitoring
init_homeostasis(
    app,
    service_name="flask_example",
    log_level="INFO",
    exclude_paths=["/health", "/metrics"],
    enable_healing=True
)

@app.route("/users/<user_id>")
def get_user(user_id):
    # Simulated database
    users = {
        "1": {"name": "Alice", "email": "alice@example.com"},
        "2": {"name": "Bob", "email": "bob@example.com"},
    }
    
    # This will raise a KeyError if user_id is not in users
    # Homeostasis will detect this and generate a fix
    return jsonify(users[user_id])

@app.route("/health")
def health_check():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(debug=True, port=8000)
```

#### Configuration (config.yaml)

```yaml
general:
  project_root: "."
  log_level: "INFO"
  environment: "development"

service:
  name: "flask_example"
  path: "./app.py"
  start_command: "python app.py"
  stop_command: "pkill -f 'python app.py'"
  health_check_url: "http://localhost:8000/health"
  health_check_timeout: 5
  log_file: "logs/flask_example.log"

monitoring:
  enabled: true
  log_level: "INFO"
  watch_patterns:
    - "logs/*.log"
  check_interval: 3

analysis:
  rule_based:
    enabled: true
    confidence_threshold: 0.7
  causal_chain:
    enabled: true
  ml_based:
    enabled: false

patch_generation:
  templates_dir: "modules/patch_generation/templates"
  generated_patches_dir: "logs/patches"
  backup_original_files: true

testing:
  enabled: true
  test_command: "pytest tests/"
  test_timeout: 30

deployment:
  enabled: true
  restart_service: true
  backup_before_deployment: true
  backup_dir: "logs/backups"

security:
  healing_rate_limiting:
    enabled: true
    max_healing_cycles_per_hour: 10
    min_interval_between_healing_seconds: 300
    max_patches_per_day: 20
```

#### How to Run

```bash
# Start the orchestrator
python orchestrator/orchestrator.py --config config.yaml

# In another terminal, trigger an error
curl http://localhost:8000/users/3
```

#### Expected Outcome

1. The request for a non-existent user ID will cause a KeyError
2. Homeostasis will detect this error and analyze it
3. A patch will be generated to add proper error handling
4. The patch will be tested and applied
5. After service restart, the endpoint will return a proper 404 response for missing users

### Example 2: Integrating with a FastAPI Application

```python
# fastapi_app.py
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, Optional
from modules.monitoring.asgi_middleware import HomeostasisASGIMiddleware

app = FastAPI(title="FastAPI Example")

# Add Homeostasis middleware
app.add_middleware(
    HomeostasisASGIMiddleware,
    service_name="fastapi_example",
    log_level="INFO",
    exclude_paths=["/health", "/docs", "/redoc", "/openapi.json"],
    include_request_body=True,
    include_response_body=True
)

# Simulated database
items_db: Dict[int, Dict] = {
    1: {"name": "Item 1", "price": 9.99},
    2: {"name": "Item 2", "price": 19.99}
}

class Item(BaseModel):
    name: str
    price: float
    description: Optional[str] = None

@app.get("/items/{item_id}")
async def get_item(item_id: int):
    # This will raise a KeyError if item_id is not in items_db
    # Homeostasis will detect and fix this
    return items_db[item_id]

@app.post("/items/")
async def create_item(item: Item):
    # Generate new ID
    new_id = max(items_db.keys()) + 1
    items_db[new_id] = item.dict()
    return {"id": new_id, **items_db[new_id]}

@app.get("/health")
async def health_check():
    return {"status": "ok"}
```

#### How to Run

```bash
# Run the FastAPI application
uvicorn fastapi_app:app --reload --port 8000

# In another terminal, start the orchestrator
python orchestrator/orchestrator.py --config fastapi_config.yaml

# Trigger an error
curl http://localhost:8000/items/999
```

## Advanced Examples

### Example 3: Complex Error Handling with Django

```python
# views.py in a Django application
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
import json
import logging

from modules.monitoring.django_middleware import HomeostasisRequestProcessor

# Initialize the request processor
processor = HomeostasisRequestProcessor(
    service_name="django_example",
    log_level="INFO",
    include_user_info=True,
    include_session_data=True
)

@csrf_exempt
@require_http_methods(["POST"])
@processor.monitor_view
def process_transaction(request):
    try:
        data = json.loads(request.body)
        
        # Simulate database operations
        user_id = data.get('user_id')
        amount = data.get('amount')
        
        # Convert amount to float - potential error if amount is not a number
        # Homeostasis will detect and fix this issue
        amount_float = float(amount)
        
        # Simulate database query - potential error if user doesn't exist
        # Homeostasis will detect and fix this issue
        user = get_user_from_db(user_id)
        
        # Process transaction logic...
        result = {
            'status': 'success',
            'transaction_id': '1234567890',
            'user': user['name'],
            'amount': amount_float
        }
        
        return JsonResponse(result)
    
    except Exception as e:
        logging.error(f"Transaction processing error: {str(e)}", exc_info=True)
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

def get_user_from_db(user_id):
    # Simulated database function
    users = {
        '1': {'name': 'John Doe', 'balance': 1000.00},
        '2': {'name': 'Jane Smith', 'balance': 2500.00}
    }
    
    # This will raise a KeyError if user_id is not in users
    return users[user_id]
```

### Example 4: Working with SQLAlchemy and Database Errors

```python
# db_example.py
from sqlalchemy import create_engine, Column, Integer, String, Float, MetaData, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from modules.monitoring.logger import MonitoringLogger

# Initialize Homeostasis logger
logger = MonitoringLogger(
    service_name="sqlalchemy_example",
    log_level="INFO",
    include_system_info=True,
    log_file_path="logs/db_example.log"
)

# Set up SQLAlchemy
Base = declarative_base()
engine = create_engine('sqlite:///example.db')
Session = sessionmaker(bind=engine)

class Product(Base):
    __tablename__ = 'products'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    price = Column(Float, nullable=False)
    stock = Column(Integer, default=0)
    
    def __repr__(self):
        return f"<Product(name='{self.name}', price={self.price}, stock={self.stock})>"

# Create tables
Base.metadata.create_all(engine)

def add_product(name, price, stock=0):
    session = Session()
    try:
        product = Product(name=name, price=price, stock=stock)
        session.add(product)
        session.commit()
        logger.info(f"Added product: {name}")
        return product.id
    except Exception as e:
        session.rollback()
        logger.error(f"Error adding product: {str(e)}", exc_info=True)
        raise
    finally:
        session.close()

def get_product(product_id):
    session = Session()
    try:
        # This can raise NoResultFound if product doesn't exist
        # Homeostasis will detect and fix this
        product = session.query(Product).filter(Product.id == product_id).one()
        return {
            'id': product.id,
            'name': product.name,
            'price': product.price,
            'stock': product.stock
        }
    except Exception as e:
        logger.error(f"Error retrieving product {product_id}: {str(e)}", exc_info=True)
        raise
    finally:
        session.close()

def update_stock(product_id, quantity):
    session = Session()
    try:
        # This can raise errors if the product doesn't exist
        # or if there's a transaction issue
        # Homeostasis will detect and fix this
        product = session.query(Product).filter(Product.id == product_id).one()
        product.stock = quantity
        session.commit()
        logger.info(f"Updated stock for product {product_id} to {quantity}")
        return True
    except Exception as e:
        session.rollback()
        logger.error(f"Error updating stock for product {product_id}: {str(e)}", exc_info=True)
        raise
    finally:
        session.close()
```

### Example 5: Handling Async Errors with Asyncio

```python
# asyncio_example.py
import asyncio
import aiohttp
import time
from modules.monitoring.logger import MonitoringLogger
from modules.monitoring.extractor import add_diagnostic_context

# Initialize Homeostasis logger
logger = MonitoringLogger(
    service_name="asyncio_example",
    log_level="INFO",
    include_system_info=True,
    log_file_path="logs/asyncio_example.log"
)

async def fetch_data(url, timeout=5):
    try:
        async with aiohttp.ClientSession() as session:
            with add_diagnostic_context(operation="fetch_data", url=url):
                async with session.get(url, timeout=timeout) as response:
                    if response.status != 200:
                        raise Exception(f"API returned {response.status}")
                    return await response.json()
    except asyncio.TimeoutError:
        logger.error(f"Timeout while fetching {url}", operation="fetch_data")
        raise
    except Exception as e:
        logger.error(f"Error fetching {url}: {str(e)}", exc_info=True, operation="fetch_data")
        raise

async def process_user_data(user_id):
    try:
        with add_diagnostic_context(user_id=user_id, operation="process_user_data"):
            # This can cause asyncio errors if APIs are unavailable
            # Homeostasis will detect and fix these errors
            user_data = await fetch_data(f"https://api.example.com/users/{user_id}")
            posts = await fetch_data(f"https://api.example.com/users/{user_id}/posts")
            
            # Process the data
            result = {
                "user": user_data,
                "posts_count": len(posts),
                "latest_posts": posts[:5]
            }
            
            logger.info(f"Processed data for user {user_id}", operation="process_user_data")
            return result
    except Exception as e:
        logger.error(f"Failed to process user {user_id}: {str(e)}", exc_info=True, operation="process_user_data")
        raise

async def main():
    user_ids = ["1", "2", "3", "invalid_id"]
    
    tasks = [process_user_data(user_id) for user_id in user_ids]
    
    # This pattern can lead to lost errors
    # Homeostasis will detect this issue and suggest a fix
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for user_id, result in zip(user_ids, results):
        if isinstance(result, Exception):
            logger.error(f"Processing failed for user {user_id}: {str(result)}")
        else:
            logger.info(f"Successfully processed user {user_id}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Integration Examples

### Example 6: Integration with APM Tools

```python
# apm_integration_example.py
from fastapi import FastAPI, HTTPException
from modules.monitoring.asgi_middleware import HomeostasisASGIMiddleware
from modules.analysis.apm_integration import APMIntegrationHandler
import json

app = FastAPI(title="APM Integration Example")

# Configure APM integration
apm_handler = APMIntegrationHandler(
    service_name="apm_example",
    apm_type="elastic",  # or "datadog", "newrelic", etc.
    correlation_enabled=True,
    config_path="apm_config.json"
)

# Add Homeostasis middleware with APM integration
app.add_middleware(
    HomeostasisASGIMiddleware,
    service_name="apm_example",
    log_level="INFO",
    exclude_paths=["/health"],
    apm_handler=apm_handler
)

# Load APM configuration
with open("apm_config.json", "r") as f:
    apm_config = json.load(f)

# Initialize APM based on provider type
if apm_config["provider"] == "elastic":
    from elasticapm.contrib.starlette import ElasticAPM
    app.add_middleware(
        ElasticAPM,
        service_name="apm_example",
        server_url=apm_config["server_url"],
        secret_token=apm_config["secret_token"]
    )
elif apm_config["provider"] == "datadog":
    # Datadog APM initialization
    from ddtrace.contrib.fastapi import FastAPIMiddleware
    app.add_middleware(FastAPIMiddleware)

@app.get("/items/{item_id}")
async def get_item(item_id: int):
    # Simulate database operation
    items = {1: "Item 1", 2: "Item 2"}
    if item_id not in items:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"id": item_id, "name": items[item_id]}

@app.get("/health")
async def health_check():
    return {"status": "ok"}
```

#### Configuration (apm_config.json)

```json
{
    "provider": "elastic",
    "server_url": "http://localhost:8200",
    "secret_token": "",
    "environment": "development",
    "correlation_id_header": "x-correlation-id",
    "transaction_sample_rate": 1.0
}
```

### Example 7: Causal Chain Analysis

This example demonstrates how to work with Homeostasis's causal chain analysis to diagnose complex, cascading errors.

```python
# causal_chain_example.py
from modules.monitoring.logger import MonitoringLogger
from modules.analysis.causal_chain import CausalChainAnalyzer
import time
import random
import os

# Initialize Homeostasis logger
logger = MonitoringLogger(
    service_name="causal_chain_example",
    log_level="INFO",
    include_system_info=True,
    log_file_path="logs/causal.log"
)

# Configure the causal chain analyzer
analyzer = CausalChainAnalyzer(
    correlation_window_seconds=60,
    max_chain_depth=5,
    probability_threshold=0.7,
    activate_on_error_count=3
)

class DatabaseSimulator:
    def __init__(self):
        self.connection_pool = []
        self.max_connections = 5
        logger.info("Database simulator initialized", component="database")
    
    def get_connection(self):
        if len(self.connection_pool) >= self.max_connections:
            logger.error("Connection pool exhausted", component="database")
            raise Exception("Connection pool exhausted")
        
        connection = {"id": random.randint(1000, 9999), "created_at": time.time()}
        self.connection_pool.append(connection)
        logger.info(f"Created new connection {connection['id']}", component="database")
        return connection
    
    def execute_query(self, query, connection=None):
        if connection is None:
            connection = self.get_connection()
        
        logger.info(f"Executing query with connection {connection['id']}", component="database", query=query)
        
        # Simulate different query execution issues
        if "product" in query and random.random() < 0.3:
            logger.error("Query timeout", component="database", connection_id=connection['id'], query=query)
            raise TimeoutError("Database query timed out")
        
        if "user" in query and random.random() < 0.3:
            logger.error("Query syntax error", component="database", connection_id=connection['id'], query=query)
            raise SyntaxError("SQL syntax error in query")
        
        return {"rows": random.randint(0, 100), "execution_time_ms": random.randint(10, 500)}
    
    def close_connection(self, connection):
        if connection in self.connection_pool:
            self.connection_pool.remove(connection)
            logger.info(f"Closed connection {connection['id']}", component="database")
        else:
            logger.warning(f"Attempted to close unknown connection {connection['id']}", component="database")

class ProductService:
    def __init__(self, db):
        self.db = db
        logger.info("Product service initialized", component="product_service")
    
    def get_product(self, product_id):
        logger.info(f"Getting product {product_id}", component="product_service")
        try:
            conn = self.db.get_connection()
            query = f"SELECT * FROM products WHERE id = {product_id}"
            result = self.db.execute_query(query, conn)
            self.db.close_connection(conn)
            return {"id": product_id, "data": result}
        except Exception as e:
            logger.error(f"Error getting product {product_id}: {str(e)}", exc_info=True, component="product_service")
            raise

class UserService:
    def __init__(self, db):
        self.db = db
        logger.info("User service initialized", component="user_service")
    
    def get_user(self, user_id):
        logger.info(f"Getting user {user_id}", component="user_service")
        try:
            conn = self.db.get_connection()
            query = f"SELECT * FROM users WHERE id = {user_id}"
            result = self.db.execute_query(query, conn)
            self.db.close_connection(conn)
            return {"id": user_id, "data": result}
        except Exception as e:
            logger.error(f"Error getting user {user_id}: {str(e)}", exc_info=True, component="user_service")
            raise

class OrderService:
    def __init__(self, product_service, user_service):
        self.product_service = product_service
        self.user_service = user_service
        logger.info("Order service initialized", component="order_service")
    
    def create_order(self, user_id, product_ids):
        logger.info(f"Creating order for user {user_id} with products {product_ids}", component="order_service")
        try:
            # Get user information - might trigger connection pool issue
            user = self.user_service.get_user(user_id)
            
            # Get product information - might trigger query timeout
            products = []
            for product_id in product_ids:
                product = self.product_service.get_product(product_id)
                products.append(product)
            
            # Create order - this will never execute if earlier steps fail
            logger.info(f"Order created for user {user_id}", component="order_service")
            return {"order_id": random.randint(10000, 99999), "user": user, "products": products}
        except Exception as e:
            logger.error(f"Failed to create order for user {user_id}: {str(e)}", exc_info=True, component="order_service")
            raise

# Run the example
if __name__ == "__main__":
    # Create logger directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Set up services
    db = DatabaseSimulator()
    product_service = ProductService(db)
    user_service = UserService(db)
    order_service = OrderService(product_service, user_service)
    
    # Process orders in a loop to generate causal chains
    for i in range(10):
        try:
            user_id = random.randint(1, 5)
            product_ids = [random.randint(1, 10) for _ in range(random.randint(1, 3))]
            order = order_service.create_order(user_id, product_ids)
            logger.info(f"Successfully created order {order['order_id']}")
        except Exception as e:
            logger.error(f"Order processing failed: {str(e)}")
        
        time.sleep(1)
    
    # Analyze error chains
    analyzer.analyze_logs("logs/causal.log")
    chains = analyzer.get_causal_chains()
    
    # Print detected chains
    for i, chain in enumerate(chains):
        print(f"\nCausal Chain #{i+1}:")
        for j, event in enumerate(chain):
            print(f"  {j+1}. [{event['level']}] {event['message']} (component: {event.get('component', 'unknown')})")
        print(f"  Root cause: {chain[0]['message']}")
```

## Production-Ready Configurations

### Example 8: Kubernetes Deployment with Homeostasis

This example demonstrates how to integrate Homeostasis with a service deployed in Kubernetes.

#### Kubernetes Deployment YAML (deployment.yaml)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: homeostasis-example
  labels:
    app: homeostasis-example
spec:
  replicas: 2
  selector:
    matchLabels:
      app: homeostasis-example
  template:
    metadata:
      labels:
        app: homeostasis-example
    spec:
      containers:
      - name: app
        image: my-registry/homeostasis-example:latest
        ports:
        - containerPort: 8000
        env:
        - name: HOMEOSTASIS_ENABLED
          value: "true"
        - name: HOMEOSTASIS_LOG_LEVEL
          value: "INFO"
        - name: HOMEOSTASIS_SERVICE_NAME
          value: "k8s-example"
        - name: HOMEOSTASIS_ENVIRONMENT
          value: "production"
        - name: HOMEOSTASIS_HEALING_MAX_CYCLES_PER_HOUR
          value: "5"
        - name: HOMEOSTASIS_APPROVAL_REQUIRED
          value: "true"
        volumeMounts:
        - name: homeostasis-config
          mountPath: /app/homeostasis-config
        - name: logs
          mountPath: /app/logs
      - name: homeostasis-sidecar
        image: my-registry/homeostasis-sidecar:latest
        env:
        - name: HOMEOSTASIS_SERVICE_NAME
          value: "k8s-example"
        - name: HOMEOSTASIS_LOG_PATH
          value: "/logs"
        - name: HOMEOSTASIS_CONFIG_PATH
          value: "/config/homeostasis-config.yaml"
        volumeMounts:
        - name: homeostasis-config
          mountPath: /config
        - name: logs
          mountPath: /logs
      volumes:
      - name: homeostasis-config
        configMap:
          name: homeostasis-config
      - name: logs
        emptyDir: {}
```

#### Homeostasis ConfigMap (configmap.yaml)

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: homeostasis-config
data:
  homeostasis-config.yaml: |
    general:
      project_root: "/app"
      log_level: "INFO"
      environment: "production"
    
    service:
      name: "k8s-example"
      health_check_url: "http://localhost:8000/health"
      health_check_timeout: 5
      log_file: "/logs/service.log"
    
    monitoring:
      enabled: true
      log_level: "INFO"
      watch_patterns:
        - "/logs/*.log"
      check_interval: 10
    
    analysis:
      rule_based:
        enabled: true
        confidence_threshold: 0.8
      causal_chain:
        enabled: true
      ml_based:
        enabled: true
        model_path: "/app/models/error_classifier.pkl"
    
    patch_generation:
      templates_dir: "/app/templates"
      generated_patches_dir: "/logs/patches"
      backup_original_files: true
    
    testing:
      enabled: true
      test_command: "python -m pytest /app/tests"
      test_timeout: 60
    
    deployment:
      enabled: true
      kubernetes:
        enabled: true
        context: "production"
        namespace: "default"
        deployment_name: "homeostasis-example"
      canary:
        enabled: true
        initial_weight: 0.1
        step_percentage: 0.1
        evaluation_period_minutes: 15
    
    security:
      healing_rate_limiting:
        enabled: true
        max_healing_cycles_per_hour: 5
        min_interval_between_healing_seconds: 600
        max_patches_per_day: 10
      
      approval:
        enabled: true
        required_for_critical: true
        webhook_url: "https://api.example.com/approval-webhook"
```

#### Python Flask Application Code

```python
# app.py
from flask import Flask, jsonify, request
import os
import logging
from modules.monitoring.flask_extension import init_homeostasis

app = Flask(__name__)

# Initialize Homeostasis if enabled
if os.environ.get("HOMEOSTASIS_ENABLED", "false").lower() == "true":
    init_homeostasis(
        app,
        service_name=os.environ.get("HOMEOSTASIS_SERVICE_NAME", "flask-app"),
        log_level=os.environ.get("HOMEOSTASIS_LOG_LEVEL", "INFO"),
        exclude_paths=["/health", "/metrics"],
        log_file_path=os.path.join("/app/logs", "service.log"),
        environment=os.environ.get("HOMEOSTASIS_ENVIRONMENT", "production")
    )

@app.route("/api/data")
def get_data():
    # Example endpoint with potential errors
    try:
        # Some logic that could fail
        data = process_data()
        return jsonify(data)
    except Exception as e:
        app.logger.error(f"Error processing data: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

def process_data():
    # Sample function that could have errors
    param = request.args.get("param")
    
    # This could cause errors if param is None or not convertible to int
    value = int(param) * 2
    
    return {"result": value}

if __name__ == "__main__":
    # Ensure logs directory exists
    os.makedirs("/app/logs", exist_ok=True)
    
    # Configure basic logging
    logging.basicConfig(
        filename="/app/logs/service.log",
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    app.run(host="0.0.0.0", port=8000)
```

## Advanced Integration Patterns

### Example 9: Cross-Language Orchestration

This example demonstrates how to use Homeostasis to handle errors across different programming languages.

#### Python FastAPI Service

```python
# python_service.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
from modules.monitoring.asgi_middleware import HomeostasisASGIMiddleware
from modules.analysis.cross_language_orchestrator import CrossLanguageAdapter

app = FastAPI()

# Initialize cross-language adapter
cross_lang_adapter = CrossLanguageAdapter(
    service_name="python_service",
    supported_languages=["python", "javascript", "java"],
    schema_path="/app/schema/error_schema.json"
)

# Add middleware
app.add_middleware(
    HomeostasisASGIMiddleware,
    service_name="python_service",
    cross_language_adapter=cross_lang_adapter
)

class Item(BaseModel):
    name: str
    quantity: int

@app.post("/process/")
async def process_item(item: Item):
    try:
        # Call JavaScript microservice
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://js-service:3000/validate",
                json=item.dict()
            )
            
            if response.status_code != 200:
                # Parse error using cross-language adapter
                error_data = response.json()
                normalized_error = cross_lang_adapter.normalize_error(
                    error_data, 
                    source_language="javascript"
                )
                
                # Log normalized error for Homeostasis to analyze
                app.logger.error(
                    f"JavaScript service error: {normalized_error['message']}", 
                    extra=normalized_error
                )
                
                raise HTTPException(status_code=400, detail=normalized_error['message'])
            
            return {"status": "success", "processed": response.json()}
    except Exception as e:
        app.logger.error(f"Processing error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
```

#### JavaScript Node.js Service

```javascript
// js_service.js
const express = require('express');
const { HomeostasisNode } = require('homeostasis-node');
const app = express();

// Initialize Homeostasis for Node.js
const homeostasis = new HomeostasisNode({
  serviceName: 'js_service',
  logLevel: 'info',
  logFilePath: '/logs/js_service.log',
  schemaPath: '/app/schema/error_schema.json'
});

// Add middleware
app.use(express.json());
app.use(homeostasis.middleware());

app.post('/validate', (req, res) => {
  const { name, quantity } = req.body;
  
  try {
    // Validation logic
    if (!name || name.trim() === '') {
      throw new Error('Name cannot be empty');
    }
    
    if (typeof quantity !== 'number' || quantity <= 0) {
      // Using Homeostasis specialized error
      throw homeostasis.createError({
        type: 'ValidationError',
        message: 'Quantity must be a positive number',
        code: 'INVALID_QUANTITY',
        context: {
          field: 'quantity',
          provided: quantity,
          expected: 'positive number'
        }
      });
    }
    
    // Process the item
    const result = {
      validated: true,
      name: name.toUpperCase(),
      quantity: quantity * 2
    };
    
    res.json(result);
  } catch (error) {
    // Log error with Homeostasis
    homeostasis.logError(error);
    
    // Format error response according to common schema
    res.status(400).json(homeostasis.formatError(error));
  }
});

app.listen(3000, () => {
  console.log('JavaScript service listening on port 3000');
});
```

#### Cross-Language Error Schema (error_schema.json)

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Homeostasis Cross-Language Error Schema",
  "description": "A standardized error format for cross-language error handling",
  "type": "object",
  "required": ["type", "message", "timestamp"],
  "properties": {
    "type": {
      "type": "string",
      "description": "Error type/class name"
    },
    "message": {
      "type": "string",
      "description": "Error message"
    },
    "code": {
      "type": "string",
      "description": "Error code"
    },
    "timestamp": {
      "type": "string",
      "format": "date-time",
      "description": "ISO timestamp when error occurred"
    },
    "source": {
      "type": "object",
      "properties": {
        "service": {
          "type": "string",
          "description": "Service name where error originated"
        },
        "file": {
          "type": "string",
          "description": "File where error occurred"
        },
        "line": {
          "type": "integer",
          "description": "Line number where error occurred"
        },
        "function": {
          "type": "string",
          "description": "Function where error occurred"
        },
        "language": {
          "type": "string",
          "description": "Programming language",
          "enum": ["python", "javascript", "java", "ruby", "go", "csharp", "php"]
        }
      }
    },
    "context": {
      "type": "object",
      "description": "Additional context about the error"
    },
    "traceback": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "file": {
            "type": "string"
          },
          "line": {
            "type": "integer"
          },
          "function": {
            "type": "string"
          },
          "code": {
            "type": "string"
          }
        }
      }
    }
  }
}
```

## Using Machine Learning for Error Analysis

### Example 10: ML-Enhanced Error Analysis

This example shows how to use Homeostasis's ML capabilities for more advanced error detection and classification.

```python
# ml_error_analyzer.py
from modules.monitoring.logger import MonitoringLogger
from modules.analysis.ml_analyzer import MLErrorAnalyzer
from modules.analysis.models.error_classifier import ErrorClassifier
from modules.analysis.models.data_collector import ErrorDataCollector
import pandas as pd
import os
import random
import time
from sklearn.model_selection import train_test_split

# Initialize the logger
logger = MonitoringLogger(
    service_name="ml_analyzer_example",
    log_level="INFO",
    log_file_path="logs/ml_analyzer.log"
)

# Initialize the error data collector
collector = ErrorDataCollector(
    storage_path="modules/analysis/models/training_data",
    feature_extractors=[
        "error_type_extractor",
        "stack_trace_features",
        "context_features",
        "message_features"
    ]
)

# Train or load the classifier
def get_or_train_classifier():
    model_path = "modules/analysis/models/error_classifier.pkl"
    
    # If model exists, load it
    if os.path.exists(model_path):
        logger.info("Loading existing error classifier model")
        return ErrorClassifier.load(model_path)
    
    # Otherwise, train a new model
    logger.info("Training new error classifier model")
    
    # Load or create training data
    if os.path.exists("modules/analysis/models/training_data/errors.csv"):
        df = pd.read_csv("modules/analysis/models/training_data/errors.csv")
    else:
        # If no data, generate synthetic data for this example
        logger.info("Generating synthetic training data")
        synthetic_data = generate_synthetic_data(500)
        df = pd.DataFrame(synthetic_data)
        os.makedirs("modules/analysis/models/training_data", exist_ok=True)
        df.to_csv("modules/analysis/models/training_data/errors.csv", index=False)
    
    # Split data
    X = df.drop(['error_category', 'root_cause'], axis=1)
    y_category = df['error_category']
    y_root_cause = df['root_cause']
    
    X_train, X_test, y_cat_train, y_cat_test, y_root_train, y_root_test = train_test_split(
        X, y_category, y_root_cause, test_size=0.2, random_state=42
    )
    
    # Create and train classifier
    classifier = ErrorClassifier()
    classifier.train(X_train, y_cat_train, y_root_train)
    
    # Evaluate
    accuracy_cat = classifier.evaluate(X_test, y_cat_test, target='category')
    accuracy_root = classifier.evaluate(X_test, y_root_test, target='root_cause')
    
    logger.info(f"Trained model with category accuracy: {accuracy_cat:.2f}, root cause accuracy: {accuracy_root:.2f}")
    
    # Save model
    classifier.save(model_path)
    return classifier

# Generate synthetic data for demonstration
def generate_synthetic_data(n_samples):
    error_types = ['ValueError', 'KeyError', 'TypeError', 'IndexError', 'AttributeError']
    error_categories = ['validation', 'database', 'configuration', 'api', 'concurrency']
    root_causes = ['missing_parameter', 'invalid_type', 'resource_not_found', 
                  'configuration_error', 'race_condition', 'timeout', 'permission_denied']
    
    components = ['web', 'database', 'auth', 'api', 'worker']
    
    data = []
    
    for _ in range(n_samples):
        error_type = random.choice(error_types)
        category = random.choice(error_categories)
        root_cause = random.choice(root_causes)
        component = random.choice(components)
        
        # Generate feature values based on the error type and root cause
        has_db_conn = 1 if category == 'database' else 0
        has_api_call = 1 if category == 'api' else 0
        has_config_access = 1 if category == 'configuration' else 0
        has_validation = 1 if category == 'validation' else 0
        has_await = 1 if category == 'concurrency' else 0
        
        num_lines = random.randint(5, 50)
        execution_time = random.randint(10, 5000)
        memory_usage = random.randint(50, 500)
        time_of_day = random.randint(0, 23)
        day_of_week = random.randint(0, 6)
        
        # Add some noise to make it realistic
        if random.random() < 0.1:
            has_db_conn = 1 - has_db_conn
        if random.random() < 0.1:
            has_api_call = 1 - has_api_call
        
        data.append({
            'error_type': error_type,
            'component': component,
            'has_db_connection': has_db_conn,
            'has_api_call': has_api_call,
            'has_config_access': has_config_access,
            'has_validation': has_validation,
            'has_await': has_await,
            'traceback_lines': num_lines,
            'execution_time_ms': execution_time,
            'memory_usage_mb': memory_usage,
            'time_of_day': time_of_day,
            'day_of_week': day_of_week,
            'error_category': category,
            'root_cause': root_cause
        })
    
    return data

# Initialize the ML analyzer
def initialize_ml_analyzer():
    classifier = get_or_train_classifier()
    
    analyzer = MLErrorAnalyzer(
        classifier=classifier,
        confidence_threshold=0.7,
        feature_extractors=[
            "error_type_extractor",
            "stack_trace_features",
            "context_features",
            "message_features"
        ],
        fallback_to_rules=True
    )
    
    return analyzer

# Simulate errors and analyze them
def simulate_errors(analyzer, num_errors=10):
    error_types = ['ValueError', 'KeyError', 'TypeError', 'IndexError', 'AttributeError']
    
    for i in range(num_errors):
        # Generate a random error
        error_type = random.choice(error_types)
        component = random.choice(['web', 'database', 'auth', 'api', 'worker'])
        
        # Create context for the error
        context = {
            'component': component,
            'execution_time_ms': random.randint(10, 5000),
            'memory_usage_mb': random.randint(50, 500),
            'has_db_connection': random.choice([0, 1]),
            'has_api_call': random.choice([0, 1]),
            'has_config_access': random.choice([0, 1]),
            'has_validation': random.choice([0, 1]),
            'has_await': random.choice([0, 1])
        }
        
        # Generate error message
        if error_type == 'ValueError':
            message = f"Invalid value for parameter '{random.choice(['id', 'name', 'email', 'age'])}'"
        elif error_type == 'KeyError':
            message = f"Key '{random.choice(['user_id', 'product_id', 'config', 'settings'])}' not found"
        elif error_type == 'TypeError':
            message = f"Cannot convert '{random.choice(['string', 'int', 'dict', 'list'])}' to '{random.choice(['int', 'str', 'bool', 'float'])}'"
        elif error_type == 'IndexError':
            message = f"List index {random.randint(0, 100)} out of range"
        else:  # AttributeError
            message = f"'{random.choice(['object', 'module', 'class', 'instance'])}' has no attribute '{random.choice(['save', 'process', 'validate', 'connect'])}'"
        
        # Log the error
        try:
            raise eval(f"{error_type}('{message}')")
        except Exception as e:
            logger.error(f"Error in {component}: {str(e)}", exc_info=True, **context)
        
        # Sleep briefly
        time.sleep(0.5)
    
    # Analyze the errors
    analyzer.analyze_log_file("logs/ml_analyzer.log")
    
    # Get the analysis results
    results = analyzer.get_analysis_results()
    
    return results

# Main execution
if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    os.makedirs("modules/analysis/models/training_data", exist_ok=True)
    
    analyzer = initialize_ml_analyzer()
    results = simulate_errors(analyzer, num_errors=15)
    
    # Print analysis results
    print("\nML Analysis Results:")
    print("====================")
    
    for i, result in enumerate(results):
        print(f"\nError #{i+1}:")
        print(f"  Type: {result['error_type']}")
        print(f"  Message: {result['message']}")
        print(f"  Predicted Category: {result['predicted_category']} (confidence: {result['category_confidence']:.2f})")
        print(f"  Predicted Root Cause: {result['predicted_root_cause']} (confidence: {result['root_cause_confidence']:.2f})")
        print(f"  Suggested Fix Template: {result['suggested_template']}")
```

## Conclusion

These examples demonstrate how to use Homeostasis in various scenarios, from basic error handling to complex production deployments. By following these patterns, you can implement self-healing capabilities in your applications and benefit from the automated error detection and correction that Homeostasis provides.

For more detailed information, refer to the API documentation and individual module documentation.