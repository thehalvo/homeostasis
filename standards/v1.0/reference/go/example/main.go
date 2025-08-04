package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/ushs/client-go/ushs"
)

func main() {
	// Create client
	client := ushs.NewClient(
		"https://api.example.com/ushs/v1",
		ushs.WithAuthToken("your-auth-token"),
	)
	defer client.Close()

	ctx := context.Background()

	// Report an error
	event := &ushs.ErrorEvent{
		Severity: ushs.SeverityHigh,
		Source: ushs.ErrorSource{
			Service:     "api-gateway",
			Environment: "production",
		},
		Error: ushs.ErrorDetails{
			Type:    "NullPointerException",
			Message: "Cannot read property 'id' of null",
			StackTrace: []ushs.StackFrame{
				{
					File:     "app.go",
					Function: "getUser",
					Line:     42,
				},
			},
		},
	}

	errorID, sessionID, err := client.ReportError(ctx, event)
	if err != nil {
		log.Fatalf("Failed to report error: %v", err)
	}

	fmt.Printf("Error reported: %s\n", errorID)
	fmt.Printf("Session started: %s\n", sessionID)

	// Connect WebSocket for real-time updates
	err = client.ConnectWebSocket(ctx, []string{"session", "patch"}, sessionID, "")
	if err != nil {
		log.Printf("Failed to connect WebSocket: %v", err)
	}

	// Register event handlers
	client.OnEvent("org.ushs.patch.generated", func(event ushs.CloudEvent) {
		fmt.Printf("Patch generated: %v\n", event.Data)
	})

	client.OnEvent("org.ushs.session.completed", func(event ushs.CloudEvent) {
		fmt.Printf("Session completed: %v\n", event.Data)
	})

	// Check session status periodically
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	timeout := time.After(2 * time.Minute)

	for {
		select {
		case <-ticker.C:
			session, err := client.GetSession(ctx, sessionID)
			if err != nil {
				log.Printf("Failed to get session: %v", err)
				continue
			}

			fmt.Printf("Session status: %s\n", session.Status)

			if session.Status == ushs.SessionStatusCompleted ||
				session.Status == ushs.SessionStatusFailed {
				fmt.Printf("Session finished with status: %s\n", session.Status)
				return
			}

		case <-timeout:
			fmt.Println("Timeout waiting for session to complete")
			return
		}
	}
}