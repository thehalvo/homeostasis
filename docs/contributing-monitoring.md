# Contributing Monitoring Extensions to Homeostasis

This guide provides detailed instructions for extending the monitoring capabilities of the Homeostasis framework.

## Understanding the Monitoring Module

The Monitoring Module is Homeostasis' sensory system, responsible for:

1. Capturing logs, exceptions, and performance metrics
2. Standardizing error formats from different sources
3. Providing context for error analysis
4. Facilitating integration with various frameworks and platforms

## Extension Points

The Monitoring Module offers several extension points:

1. **Framework Integrations**: Adapters for web frameworks, ORMs, etc.
2. **Log Format Parsers**: Converters for different logging formats
3. **Context Providers**: Components that gather additional context
4. **Transport Mechanisms**: Ways to transmit error data

## Creating Framework Integrations

### 1. Identify Target Framework

Start by selecting a framework that would benefit from deeper integration:

- Web frameworks (Django, Flask, FastAPI, etc.)
- ORMs and database libraries
- Messaging systems
- Cloud provider SDKs

### 2. Create Middleware or Plugin

For web frameworks, create a middleware component in the `modules/monitoring/middleware.py` file:

```python
class NewFrameworkMiddleware:
    """Middleware for capturing errors from NewFramework."""
    
    def __init__(self, app, config=None):
        self.app = app
        self.config = config or {}
        
    def process_request(self, request):
        # Framework-specific request processing
        pass
        
    def process_response(self, request, response):
        # Framework-specific response processing
        return response
        
    def process_exception(self, request, exception):
        # Capture and format exception
        error_data = {
            "type": exception.__class__.__name__,
            "message": str(exception),
            "stack_trace": traceback.format_exc(),
            "context": {
                "url": request.path,
                "method": request.method,
                # Framework-specific context
            }
        }
        
        # Send to error processor
        from homeostasis.monitoring.logger import log_error
        log_error(error_data)
        
        # Framework-specific exception handling
        return None
```

### 3. Add Usage Documentation

Document how to integrate your middleware:

```python
"""
To use NewFrameworkMiddleware:

1. Import the middleware:
   from homeostasis.monitoring.middleware import NewFrameworkMiddleware

2. Add to your application:
   
   # For NewFramework
   app = NewFramework()
   app.middleware(NewFrameworkMiddleware(app))
   
   # Or with configuration
   app.middleware(NewFrameworkMiddleware(app, {
       "capture_404": True,
       "capture_request_body": False
   }))
"""
```

## Creating Log Format Parsers

### 1. Identify Target Log Format

Determine which log format you want to support:

- JSON logs
- Syslog
- Custom application formats
- Cloud provider log formats

### 2. Create Parser in the Extractor Module

Add a parser in `modules/monitoring/extractor.py`:

```python
def parse_custom_log_format(log_line):
    """
    Parse a log line in CustomFormat into the standard Homeostasis error format.
    
    Example input:
    [ERROR][2023-05-20 14:30:22] MyModule - KeyError: 'user_id' in process_user
    
    Returns standardized error dict.
    """
    # Parsing logic using regex or string processing
    match = re.match(r'\[(\w+)\]\[([^\]]+)\] (\w+) - (.+)', log_line)
    if not match:
        return None
        
    level, timestamp, module, message = match.groups()
    if level != "ERROR":
        return None
    
    # Extract error type and details
    error_parts = message.split(':', 1)
    error_type = error_parts[0].strip()
    error_message = error_parts[1].strip() if len(error_parts) > 1 else ""
    
    # Return standardized format
    return {
        "level": level,
        "timestamp": timestamp,
        "module": module,
        "type": error_type,
        "message": error_message,
        "raw_log": log_line
    }
```

### 3. Register the Parser

Add your parser to the extractor registry:

```python
LOG_PARSERS = {
    "json": parse_json_log,
    "syslog": parse_syslog,
    "custom_format": parse_custom_log_format
}

def extract_error_from_log(log_line, format_type=None):
    """Extract error information from various log formats."""
    # Auto-detect format if not specified
    if not format_type:
        format_type = detect_log_format(log_line)
    
    # Use appropriate parser
    if format_type in LOG_PARSERS:
        return LOG_PARSERS[format_type](log_line)
    
    # Fallback to default parser
    return parse_default_log(log_line)
```

## Creating Context Providers

Context providers gather additional information to help with error analysis.

### 1. Identify Useful Context

Determine what context would be valuable:

- System metrics (CPU, memory, disk)
- Application state (cache hit rates, queue sizes)
- Environment details (versions, configurations)
- Request data (for web applications)

### 2. Implement Context Provider

Create a context provider in `modules/monitoring/logger.py`:

```python
def gather_system_metrics():
    """Gather system metrics for additional context."""
    try:
        import psutil
        
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent,
            "open_files": len(psutil.Process().open_files()),
            "connections": len(psutil.Process().connections())
        }
    except ImportError:
        return {"error": "psutil not available"}
    except Exception as e:
        return {"error": f"Failed to gather metrics: {str(e)}"}
```

### 3. Register the Context Provider

Add your provider to the context gatherers:

```python
CONTEXT_PROVIDERS = [
    gather_basic_context,
    gather_request_context,
    gather_system_metrics
]

def gather_full_context(error_data, include_providers=None):
    """Gather context from all registered providers or specified ones."""
    context = {}
    
    providers = include_providers or CONTEXT_PROVIDERS
    for provider in providers:
        try:
            provider_context = provider()
            if provider_context:
                context.update(provider_context)
        except Exception as e:
            context[f"context_error_{provider.__name__}"] = str(e)
    
    return context
```

## Creating Transport Mechanisms

Transport mechanisms determine how error data is transmitted from the application to the analysis module.

### 1. Choose Transport Type

Select a transport method:

- Local file storage
- Message queue (RabbitMQ, Kafka)
- HTTP webhook
- Database storage

### 2. Implement Transport

Add a transport in `modules/monitoring/logger.py`:

```python
class KafkaTransport:
    """Transport for sending error data to Kafka."""
    
    def __init__(self, config):
        self.config = config
        self.topic = config.get("topic", "homeostasis-errors")
        self.producer = None
        
        # Setup Kafka producer
        try:
            from kafka import KafkaProducer
            import json
            
            self.producer = KafkaProducer(
                bootstrap_servers=config.get("bootstrap_servers", "localhost:9092"),
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
        except ImportError:
            print("Kafka transport requires kafka-python package")
    
    def send(self, error_data):
        """Send error data to Kafka topic."""
        if not self.producer:
            return False
            
        try:
            future = self.producer.send(self.topic, error_data)
            self.producer.flush()
            return True
        except Exception as e:
            print(f"Failed to send to Kafka: {str(e)}")
            return False
```

### 3. Register the Transport

Add your transport to the available transports:

```python
TRANSPORTS = {
    "file": FileTransport,
    "http": HttpTransport,
    "kafka": KafkaTransport
}

def get_transport(config):
    """Get transport instance based on configuration."""
    transport_type = config.get("transport", "file")
    
    if transport_type in TRANSPORTS:
        return TRANSPORTS[transport_type](config)
    
    # Fallback to file transport
    return FileTransport(config)
```

## Testing Your Extensions

### Unit Testing

Write unit tests for your extension in the `tests/` directory:

```python
# In tests/test_monitoring.py

def test_custom_log_parser():
    """Test custom log format parser."""
    from homeostasis.monitoring.extractor import parse_custom_log_format
    
    # Test with valid log line
    log_line = "[ERROR][2023-05-20 14:30:22] UserModule - KeyError: 'user_id' in process_user"
    result = parse_custom_log_format(log_line)
    
    assert result is not None
    assert result["type"] == "KeyError"
    assert result["module"] == "UserModule"
    assert "user_id" in result["message"]
    
    # Test with invalid format
    invalid_line = "Some random text that doesn't match the format"
    assert parse_custom_log_format(invalid_line) is None
```

### Integration Testing

Test integration with the full monitoring pipeline:

```python
def test_kafka_transport_integration():
    """Test Kafka transport integration."""
    from homeostasis.monitoring.logger import log_error
    from homeostasis.monitoring.extractor import extract_error_from_log
    
    # Mock Kafka producer
    mock_producer = mock.MagicMock()
    
    # Configure to use Kafka transport
    config = {
        "transport": "kafka",
        "bootstrap_servers": "mock:9092",
        "topic": "test-topic"
    }
    
    with mock.patch('kafka.KafkaProducer', return_value=mock_producer):
        # Log an error
        error_data = {
            "type": "ValueError",
            "message": "Invalid input",
            "stack_trace": "...",
            "context": {"test": True}
        }
        
        # This should use our transport
        log_error(error_data, config)
        
        # Verify producer was called
        mock_producer.send.assert_called_once()
        args, kwargs = mock_producer.send.call_args
        assert args[0] == "test-topic"
        assert "ValueError" in str(args[1])
```

## Extension Contribution Checklist

- [ ] Extension has clear purpose and scope
- [ ] Code follows project style and conventions
- [ ] Documentation is provided for usage
- [ ] Unit tests verify functionality
- [ ] Integration tests confirm compatibility
- [ ] Dependencies are clearly documented
- [ ] Error handling is robust
- [ ] Performance impact is considered

## Frequently Asked Questions

### How do I decide which extension point to use?
Choose based on what you're trying to achieve: framework integration (middleware), log parsing (extractor), additional context (context provider), or data transmission (transport).

### Can I extend multiple components at once?
Yes, complex integrations might touch multiple extension points. Keep them modular but coordinated.

### How do I handle sensitive data in monitoring?
Implement data scrubbing in your extension to remove or mask sensitive information.

---

By contributing monitoring extensions, you help Homeostasis capture a wider variety of errors across more platforms and frameworks. Thank you for your contribution!