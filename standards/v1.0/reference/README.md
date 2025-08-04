# USHS Reference Implementations

This directory contains reference implementations of the Universal Self-Healing Standard (USHS) v1.0 client libraries in multiple programming languages.

## Available Implementations

### Python
- **Location**: `python/`
- **Package**: `ushs-client`
- **Requirements**: Python 3.7+, aiohttp, websockets

```python
from ushs_client import USHSClient, ErrorEvent, Severity

async with USHSClient(base_url="https://api.example.com/ushs/v1", auth_token="token") as client:
    error = ErrorEvent(
        severity=Severity.HIGH,
        service="my-service",
        type="RuntimeError",
        message="Something went wrong"
    )
    result = await client.report_error(error)
```

### TypeScript/JavaScript
- **Location**: `typescript/`
- **Package**: `@ushs/client`
- **Requirements**: Node.js 14+

```typescript
import { USHSClient, Severity } from '@ushs/client';

const client = new USHSClient({
  baseUrl: 'https://api.example.com/ushs/v1',
  authToken: 'token'
});

await client.reportError({
  severity: Severity.HIGH,
  source: { service: 'my-service' },
  error: { type: 'Error', message: 'Something went wrong' }
});
```

### Go
- **Location**: `go/`
- **Module**: `github.com/ushs/client-go`
- **Requirements**: Go 1.21+

```go
import "github.com/ushs/client-go/ushs"

client := ushs.NewClient(
    "https://api.example.com/ushs/v1",
    ushs.WithAuthToken("token"),
)

errorID, sessionID, err := client.ReportError(ctx, &ushs.ErrorEvent{
    Severity: ushs.SeverityHigh,
    Source:   ushs.ErrorSource{Service: "my-service"},
    Error:    ushs.ErrorDetails{Type: "Error", Message: "Something went wrong"},
})
```

## Features

All reference implementations provide:

### Core Functionality
- Error reporting and retrieval
- Healing session management
- Patch submission and validation
- Deployment orchestration
- Health checking

### Real-time Support
- WebSocket connection for live updates
- Event subscription and filtering
- Automatic reconnection with backoff

### Security
- Bearer token authentication
- API key authentication
- mTLS support (where applicable)
- TLS 1.3+ enforcement

### Developer Experience
- Type safety (TypeScript, Go)
- Async/await support
- Comprehensive error handling
- Extensive documentation
- Example code

## Installation

### Python
```bash
pip install ushs-client
```

### TypeScript/JavaScript
```bash
npm install @ushs/client
# or
yarn add @ushs/client
```

### Go
```bash
go get github.com/ushs/client-go
```

## Configuration

All clients support the following configuration options:

| Option | Description | Required |
|--------|-------------|----------|
| `baseUrl` | USHS API endpoint URL | Yes |
| `authToken` | Bearer authentication token | No* |
| `apiKey` | API key for authentication | No* |
| `timeout` | Request timeout in milliseconds | No |
| `verifySsl` | Whether to verify SSL certificates | No |

*At least one authentication method is required

## WebSocket Events

All implementations support the following event types:

### Error Events
- `org.ushs.error.detected` - New error detected

### Session Events  
- `org.ushs.session.started` - Healing session started
- `org.ushs.session.phaseCompleted` - Phase completed
- `org.ushs.session.failed` - Session failed
- `org.ushs.session.completed` - Session completed

### Patch Events
- `org.ushs.patch.generated` - Patch generated
- `org.ushs.patch.validating` - Validation started
- `org.ushs.patch.validated` - Validation completed
- `org.ushs.patch.approved` - Patch approved
- `org.ushs.patch.rejected` - Patch rejected

### Deployment Events
- `org.ushs.deployment.started` - Deployment started
- `org.ushs.deployment.progress` - Deployment progress
- `org.ushs.deployment.completed` - Deployment completed
- `org.ushs.deployment.failed` - Deployment failed
- `org.ushs.deployment.rollback` - Rollback initiated

## Examples

Each implementation includes example code demonstrating:
- Basic error reporting
- Session monitoring
- Real-time event handling
- Patch validation and deployment

See the `example` or example files in each language directory.

## Testing

Each implementation includes comprehensive test suites. Run tests with:

### Python
```bash
cd python
pytest
```

### TypeScript
```bash
cd typescript
npm test
```

### Go
```bash
cd go
go test ./...
```

## Contributing

When adding new language implementations, ensure they:
1. Implement all required USHS interfaces
2. Follow language-specific conventions
3. Include comprehensive documentation
4. Provide example code
5. Have >80% test coverage
6. Support all authentication methods
7. Handle WebSocket reconnection

## License

All reference implementations are licensed under the MIT License. See individual LICENSE files for details.