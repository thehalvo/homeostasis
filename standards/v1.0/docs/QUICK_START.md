# USHS Quick Start Guide

Get up and running with the Universal Self-Healing Standard in 15 minutes.

## Prerequisites

- A running application or service
- Admin access to your infrastructure
- One of: Python 3.7+, Node.js 14+, or Go 1.21+

## Step 1: Install USHS Client (2 minutes)

### Python
```bash
pip install ushs-client
```

### Node.js
```bash
npm install @ushs/client
```

### Go
```bash
go get github.com/ushs/client-go
```

## Step 2: Deploy USHS Server (5 minutes)

### Option A: Docker Compose (Recommended for testing)

```yaml
# docker-compose.yml
version: '3.8'
services:
  ushs-orchestrator:
    image: ushs/orchestrator:latest
    ports:
      - "8080:8080"
      - "8081:8081"  # WebSocket
    environment:
      - USHS_AUTH_ENABLED=false  # Disable for quick start
      - USHS_LOG_LEVEL=info
    volumes:
      - ./data:/data

  ushs-ui:
    image: ushs/dashboard:latest
    ports:
      - "3000:3000"
    environment:
      - USHS_API_URL=http://ushs-orchestrator:8080
```

```bash
docker-compose up -d
```

### Option B: Kubernetes

```bash
kubectl apply -f k8s-manifests.yaml  # Use your own manifest file
```

## Step 3: Basic Integration (5 minutes)

### Python Example

```python
import asyncio
import os
from ushs_client import USHSClient, ErrorEvent, Severity

async def main():
    # Initialize client
    client = USHSClient(
        base_url="http://localhost:8080/ushs/v1",
        api_key="quickstart"  # Default for testing
    )
    
    # Simulate an error
    try:
        # Your application code
        result = 1 / 0  # Will cause ZeroDivisionError
    except Exception as e:
        # Report to USHS
        error_id, session_id = await client.report_error(ErrorEvent(
            severity=Severity.HIGH,
            service="quickstart-app",
            type=type(e).__name__,
            message=str(e),
            environment="development"
        ))
        
        print(f"Error reported: {error_id}")
        print(f"Healing session: {session_id}")
        
        # Monitor healing progress
        await monitor_healing(client, session_id)

async def monitor_healing(client, session_id):
    """Monitor healing session progress"""
    print("\nMonitoring healing progress...")
    
    for _ in range(30):  # Check for 30 seconds
        session = await client.get_session(session_id)
        print(f"Status: {session['status']}")
        
        if session['status'] in ['completed', 'failed']:
            print(f"\nHealing {session['status']}!")
            if session['status'] == 'completed':
                print("Your application has been healed! 🎉")
            break
            
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
```

### Node.js Example

```javascript
const { USHSClient, Severity } = require('@ushs/client');

async function main() {
    // Initialize client
    const client = new USHSClient({
        baseUrl: 'http://localhost:8080/ushs/v1',
        apiKey: 'quickstart'
    });
    
    try {
        // Your application code
        throw new Error('Database connection failed');
    } catch (error) {
        // Report to USHS
        const { errorId, sessionId } = await client.reportError({
            severity: Severity.HIGH,
            source: {
                service: 'quickstart-app',
                environment: 'development'
            },
            error: {
                type: error.name,
                message: error.message,
                stackTrace: error.stack.split('\n')
            }
        });
        
        console.log(`Error reported: ${errorId}`);
        console.log(`Healing session: ${sessionId}`);
        
        // Monitor healing
        await monitorHealing(client, sessionId);
    }
}

async function monitorHealing(client, sessionId) {
    console.log('\nMonitoring healing progress...');
    
    // Connect WebSocket for real-time updates
    client.connectWebSocket({ session: sessionId });
    
    client.on('org.ushs.session.phaseCompleted', (event) => {
        console.log(`Phase completed: ${event.data.phase}`);
    });
    
    client.on('org.ushs.session.completed', (event) => {
        console.log('\nHealing completed! 🎉');
        process.exit(0);
    });
    
    client.on('org.ushs.session.failed', (event) => {
        console.log('\nHealing failed:', event.data.reason);
        process.exit(1);
    });
}

main().catch(console.error);
```

### Go Example

```go
package main

import (
    "context"
    "fmt"
    "log"
    "time"
    
    "github.com/ushs/client-go/ushs"
)

func main() {
    // Initialize client
    client := ushs.NewClient(
        "http://localhost:8080/ushs/v1",
        ushs.WithAPIKey("quickstart"),
    )
    defer client.Close()
    
    ctx := context.Background()
    
    // Simulate an error
    err := simulateError()
    if err != nil {
        // Report to USHS
        errorID, sessionID, err := client.ReportError(ctx, &ushs.ErrorEvent{
            Severity: ushs.SeverityHigh,
            Source: ushs.ErrorSource{
                Service:     "quickstart-app",
                Environment: "development",
            },
            Error: ushs.ErrorDetails{
                Type:    "SimulatedError",
                Message: err.Error(),
            },
        })
        
        if err != nil {
            log.Fatalf("Failed to report error: %v", err)
        }
        
        fmt.Printf("Error reported: %s\n", errorID)
        fmt.Printf("Healing session: %s\n", sessionID)
        
        // Monitor healing
        monitorHealing(ctx, client, sessionID)
    }
}

func simulateError() error {
    return fmt.Errorf("database connection timeout")
}

func monitorHealing(ctx context.Context, client *ushs.Client, sessionID string) {
    fmt.Println("\nMonitoring healing progress...")
    
    ticker := time.NewTicker(1 * time.Second)
    defer ticker.Stop()
    
    for {
        select {
        case <-ticker.C:
            session, err := client.GetSession(ctx, sessionID)
            if err != nil {
                log.Printf("Error getting session: %v", err)
                continue
            }
            
            fmt.Printf("Status: %s\n", session.Status)
            
            if session.Status == ushs.SessionStatusCompleted {
                fmt.Println("\nHealing completed! 🎉")
                return
            } else if session.Status == ushs.SessionStatusFailed {
                fmt.Println("\nHealing failed!")
                return
            }
            
        case <-time.After(30 * time.Second):
            fmt.Println("\nTimeout waiting for healing")
            return
        }
    }
}
```

## Step 4: View Dashboard (1 minute)

Open http://localhost:3000 in your browser to see:

- Real-time healing sessions
- Error trends and patterns
- Healing success rates
- System health metrics

## Step 5: Create Your First Healing Rule (2 minutes)

Create a simple rule to handle database connection errors:

```yaml
# rules/database-timeout.yaml
apiVersion: healing.local/v1  # Your API version
kind: HealingRule
metadata:
  name: database-timeout-fix
spec:
  trigger:
    errorType: "DatabaseTimeout"
    service: "quickstart-app"
  
  actions:
    - type: configuration
      target: database.connection_pool
      changes:
        max_connections: "+10"
        timeout: "30s"
    
    - type: restart
      target: service
      strategy: rolling
  
  validation:
    - type: health_check
      endpoint: /health
      expected_status: 200
    
    - type: error_rate
      threshold: "< 0.01"
      duration: 5m
```

Apply the rule:
```bash
curl -X POST http://localhost:8080/ushs/v1/rules \
  -H "Content-Type: application/yaml" \
  -d @rules/database-timeout.yaml
```

## What's Next?

### 1. Enable Authentication

```yaml
# docker-compose.yml
environment:
  - USHS_AUTH_ENABLED=true
  - USHS_AUTH_TOKEN=your-secure-token
```

### 2. Add More Services

```python
# service-a.py
client_a = USHSClient(base_url=USHS_URL, api_key=API_KEY)

# service-b.py  
client_b = USHSClient(base_url=USHS_URL, api_key=API_KEY)

# Both services now have self-healing capabilities!
```

### 3. Configure Advanced Features

- **ML-based healing**: Enable pattern learning
- **Multi-language support**: Add parsers for your stack
- **Custom patches**: Write service-specific healers
- **Compliance**: Enable audit logging

### 4. Production Deployment

See our [Production Guide](./PRODUCTION_GUIDE.md) for:
- High availability setup
- Security hardening
- Performance tuning
- Monitoring and alerting

## Troubleshooting

### Server won't start
```bash
# Check logs
docker-compose logs ushs-orchestrator

# Common issue: Port already in use
# Solution: Change ports in docker-compose.yml
```

### Client can't connect
```bash
# Verify server is running
curl http://localhost:8080/ushs/v1/health

# Should return:
# {"status": "healthy", "components": {...}}
```

### No healing occurring
```bash
# Check if rules are loaded
curl http://localhost:8080/ushs/v1/rules

# Verify error format matches rule triggers
```

## Example Output

After running the quick start, you should see:

```
Error reported: 550e8400-e29b-41d4-a716-446655440000
Healing session: 650e8400-e29b-41d4-a716-446655440000

Monitoring healing progress...
Status: active
Status: active
Phase completed: analysis
Status: active
Phase completed: generation
Status: active
Phase completed: validation
Status: active
Phase completed: deployment
Status: completed

Healing completed!
```

## Next Steps

1. **Read the Docs**: Deep dive into [USHS concepts](./CONCEPTS.md)
3. **Run Compliance Tests**: Verify your implementation
4. **Contribute**: Share your healing patterns

---

**Congratulations!** You've successfully set up self-healing for your application. Your system is now more resilient and can automatically recover from common errors.
