# Homeostasis API Client Libraries

Official client libraries for integrating with the Homeostasis self-healing framework.

## Available Clients

- **Python** (`python_client.py`) - Full-featured Python client with async support
- **JavaScript/TypeScript** (`javascript_client.js`) - Browser and Node.js compatible
- **Go** (`go_client.go`) - Native Go client with context support

## Quick Start

### Python

```python
from homeostasis import create_client, quick_report_error

# Quick error reporting
error_id = quick_report_error(
    "http://localhost:8080",
    "Division by zero",
    "Traceback (most recent call last)...",
    "python",
    api_key="your-key"
)

# Full client usage
client = create_client("http://localhost:8080", api_key="your-key")
health = client.get_system_health()
print(f"System status: {health.status}")
```

### JavaScript

```javascript
import { createClient, quickReportError } from './javascript_client.js';

// Quick error reporting
const errorId = await quickReportError(
    'http://localhost:8080',
    'Cannot read property of undefined',
    'Error: Cannot read property...',
    'javascript',
    'your-key'
);

// Full client usage
const client = createClient('http://localhost:8080', 'your-key');
const health = await client.getSystemHealth();
console.log(`System status: ${health.status}`);
```

### Go

```go
import homeostasis "path/to/go_client"

// Quick error reporting
errorID, err := homeostasis.QuickReportError(
    context.Background(),
    "http://localhost:8080",
    "your-key",
    "nil pointer dereference",
    "goroutine 1 [running]...",
    "go"
)

// Full client usage
client := homeostasis.NewClient("http://localhost:8080",
    homeostasis.WithAPIKey("your-key"))
health, err := client.GetSystemHealth(context.Background())
fmt.Printf("System status: %s\n", health.Status)
```

## Common Features

All client libraries support:

- Error reporting and tracking
- Healing status monitoring
- System health checks
- Metrics retrieval
- Rule management
- Configuration updates
- WebSocket real-time updates
- Batch operations
- Automatic retries and error handling

## Authentication

Pass your API key when creating the client:

```python
# Python
client = HomeostasisClient(base_url, api_key="your-key")

# JavaScript
const client = new HomeostasisClient(baseUrl, 'your-key');

// Go
client := NewClient(baseURL, WithAPIKey("your-key"))
```

## Error Reporting

All clients use a similar structure for error reports:

```python
# Python
error = ErrorReport(
    error_message="Division by zero",
    stack_trace="Full stack trace...",
    language="python",
    framework="django",  # optional
    file_path="/app/views.py",  # optional
    line_number=42,  # optional
    severity=ErrorSeverity.HIGH,  # optional
    context={"user_id": 123}  # optional
)
```

## Real-time Updates

Subscribe to WebSocket events:

```python
# Python
client.connect_websocket()
client.subscribe("healing_started", lambda data: print(f"Started: {data}"))
client.subscribe("healing_completed", lambda data: print(f"Completed: {data}"))

# JavaScript
client.connectWebSocket();
client.subscribe('healing_started', (data) => console.log('Started:', data));
client.subscribe('healing_completed', (data) => console.log('Completed:', data));

// Go
client.ConnectWebSocket(ctx, onMessage, onError, onClose)
client.Subscribe("healing_started", func(data interface{}) {
    fmt.Println("Started:", data)
})
```

## Batch Operations

Report multiple errors at once:

```python
# Python
errors = [
    ErrorReport(error_message="Error 1", ...),
    ErrorReport(error_message="Error 2", ...),
]
results = client.batch_report_errors(errors)

# JavaScript
const errors = [
    new ErrorReport({ errorMessage: 'Error 1', ... }),
    new ErrorReport({ errorMessage: 'Error 2', ... }),
];
const results = await client.batchReportErrors(errors);

// Go
errors := []*ErrorReport{
    {ErrorMessage: "Error 1", ...},
    {ErrorMessage: "Error 2", ...},
}
results, err := client.BatchReportErrors(ctx, errors)
```

## Error Handling

All clients provide detailed error information:

```python
# Python
try:
    result = client.trigger_healing(error_id)
except requests.exceptions.HTTPError as e:
    print(f"HTTP error: {e}")
except requests.exceptions.Timeout as e:
    print(f"Request timeout: {e}")

# JavaScript
try {
    const result = await client.triggerHealing(errorId);
} catch (error) {
    if (error instanceof HomeostasisError) {
        console.error(`API error (${error.statusCode}): ${error.message}`);
    }
}

// Go
result, err := client.TriggerHealing(ctx, errorID, nil)
if err != nil {
    log.Printf("Error: %v", err)
}
```

## Configuration

Customize client behavior:

```python
# Python
client = HomeostasisClient(
    base_url,
    api_key="key",
    timeout=60,  # 60 seconds
    verify_ssl=True
)

# JavaScript
const client = new HomeostasisClient(baseUrl, apiKey, {
    timeout: 60000,  // 60 seconds
    wsMaxReconnectAttempts: 10,
    headers: { 'X-Custom-Header': 'value' }
});

// Go
client := NewClient(baseURL,
    WithAPIKey("key"),
    WithTimeout(60 * time.Second),
    WithMaxReconnectAttempts(10),
    WithHTTPClient(customHTTPClient))
```

## License

These client libraries are part of the Homeostasis project and follow the same license.