# USHS WebSocket Protocol Specification

## Overview

The USHS WebSocket protocol enables real-time communication between healing system components. This protocol is used for:

- Real-time error notifications
- Healing session status updates
- Live patch generation progress
- Deployment status streaming
- System health monitoring

## Connection Establishment

### Endpoint

```
wss://api.example.com/ushs/v1/ws
```

### Authentication

Clients MUST authenticate using one of:

1. **Bearer Token** in connection header:
   ```
   Authorization: Bearer <jwt-token>
   ```

2. **API Key** in query parameter:
   ```
   wss://api.example.com/ushs/v1/ws?apikey=<api-key>
   ```

3. **mTLS** client certificate

### Connection Parameters

Query parameters:
- `subscribe`: Comma-separated list of event types to subscribe to
- `session`: Specific session ID to monitor
- `service`: Service name filter

Example:
```
wss://api.example.com/ushs/v1/ws?subscribe=error,patch&service=api-gateway
```

## Message Format

All messages MUST follow the CloudEvents specification with USHS extensions:

```json
{
  "specversion": "1.0",
  "id": "uuid-v4",
  "source": "/healing/component/name",
  "type": "org.ushs.event.type",
  "datacontenttype": "application/json",
  "time": "2024-01-15T23:30:00Z",
  "subject": "optional/subject",
  "data": {
    // Event-specific data
  }
}
```

## Event Types

### Error Events

**Type**: `org.ushs.error.detected`

```json
{
  "type": "org.ushs.error.detected",
  "data": {
    "errorId": "uuid",
    "severity": "critical|high|medium|low",
    "service": "service-name",
    "summary": "Brief error description"
  }
}
```

### Session Events

**Type**: `org.ushs.session.*`

Session lifecycle events:
- `org.ushs.session.started`
- `org.ushs.session.phaseCompleted`
- `org.ushs.session.failed`
- `org.ushs.session.completed`

```json
{
  "type": "org.ushs.session.phaseCompleted",
  "subject": "session/abc123",
  "data": {
    "sessionId": "abc123",
    "phase": "analysis",
    "status": "completed",
    "duration": 1234,
    "nextPhase": "generation"
  }
}
```

### Patch Events

**Type**: `org.ushs.patch.*`

Patch lifecycle events:
- `org.ushs.patch.generated`
- `org.ushs.patch.validating`
- `org.ushs.patch.validated`
- `org.ushs.patch.approved`
- `org.ushs.patch.rejected`

```json
{
  "type": "org.ushs.patch.generated",
  "subject": "patch/xyz789",
  "data": {
    "patchId": "xyz789",
    "sessionId": "abc123",
    "confidence": 0.95,
    "changeCount": 3,
    "estimatedImpact": "low"
  }
}
```

### Deployment Events

**Type**: `org.ushs.deployment.*`

Deployment events:
- `org.ushs.deployment.started`
- `org.ushs.deployment.progress`
- `org.ushs.deployment.completed`
- `org.ushs.deployment.failed`
- `org.ushs.deployment.rollback`

```json
{
  "type": "org.ushs.deployment.progress",
  "subject": "deployment/dep456",
  "data": {
    "deploymentId": "dep456",
    "patchId": "xyz789",
    "strategy": "canary",
    "progress": 25,
    "currentPhase": "canary-10%",
    "metrics": {
      "errorRate": 0.01,
      "latency": 45
    }
  }
}
```

## Client Commands

Clients can send commands to the server:

### Subscribe Command

```json
{
  "command": "subscribe",
  "eventTypes": ["error", "session"],
  "filters": {
    "service": "api-gateway",
    "severity": ["critical", "high"]
  }
}
```

### Unsubscribe Command

```json
{
  "command": "unsubscribe",
  "eventTypes": ["error"]
}
```

### Ping Command

```json
{
  "command": "ping"
}
```

Server responds with:
```json
{
  "type": "pong",
  "timestamp": "2024-01-15T23:30:00Z"
}
```

## Connection Management

### Heartbeat

- Server sends ping frame every 30 seconds
- Client MUST respond with pong frame
- Connection closed after 3 missed pongs

### Reconnection

- Clients SHOULD implement exponential backoff
- Initial retry: 1 second
- Maximum retry interval: 60 seconds
- Include `Last-Event-ID` header for event replay

### Rate Limiting

- Maximum 100 messages per second per client
- Burst allowance: 1000 messages
- Rate limit headers in close frame

## Error Handling

### Error Codes

| Code | Description |
|------|-------------|
| 4000 | Invalid message format |
| 4001 | Authentication required |
| 4002 | Unauthorized |
| 4003 | Rate limit exceeded |
| 4004 | Invalid subscription |
| 4005 | Server error |

### Error Message Format

```json
{
  "type": "error",
  "code": 4000,
  "message": "Invalid message format",
  "details": {
    "field": "eventTypes",
    "error": "Must be an array"
  }
}
```

## Security Considerations

1. **TLS Required**: All connections MUST use TLS 1.3 or higher
2. **Authentication**: Valid authentication required before any subscriptions
3. **Authorization**: Event filtering based on client permissions
4. **Rate Limiting**: Prevent resource exhaustion
5. **Input Validation**: All client messages validated before processing

## Example Session

```javascript
// Client connects
const ws = new WebSocket('wss://api.example.com/ushs/v1/ws', {
  headers: {
    'Authorization': 'Bearer <token>'
  }
});

// Subscribe to events
ws.send(JSON.stringify({
  command: 'subscribe',
  eventTypes: ['error', 'session'],
  filters: {
    severity: ['critical', 'high']
  }
}));

// Receive error event
ws.on('message', (data) => {
  const event = JSON.parse(data);
  if (event.type === 'org.ushs.error.detected') {
    console.log('Error detected:', event.data);
  }
});

// Clean disconnect
ws.close(1000, 'Client disconnect');
```