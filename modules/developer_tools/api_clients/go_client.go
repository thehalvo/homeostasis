// Package homeostasis provides a Go client library for interacting with the
// Homeostasis self-healing framework.
package homeostasis

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/websocket"
)

// HealingStatus represents the status of a healing operation
type HealingStatus string

const (
	HealingStatusPending        HealingStatus = "pending"
	HealingStatusAnalyzing      HealingStatus = "analyzing"
	HealingStatusGeneratingPatch HealingStatus = "generating_patch"
	HealingStatusTesting        HealingStatus = "testing"
	HealingStatusApplying       HealingStatus = "applying"
	HealingStatusCompleted      HealingStatus = "completed"
	HealingStatusFailed         HealingStatus = "failed"
	HealingStatusRolledBack     HealingStatus = "rolled_back"
)

// ErrorSeverity represents the severity level of an error
type ErrorSeverity string

const (
	ErrorSeverityLow      ErrorSeverity = "low"
	ErrorSeverityMedium   ErrorSeverity = "medium"
	ErrorSeverityHigh     ErrorSeverity = "high"
	ErrorSeverityCritical ErrorSeverity = "critical"
)

// ErrorReport represents an error to be reported for healing
type ErrorReport struct {
	ErrorMessage string                 `json:"error_message"`
	StackTrace   string                 `json:"stack_trace"`
	Language     string                 `json:"language"`
	Framework    string                 `json:"framework,omitempty"`
	FilePath     string                 `json:"file_path,omitempty"`
	LineNumber   int                    `json:"line_number,omitempty"`
	Severity     ErrorSeverity          `json:"severity,omitempty"`
	Context      map[string]interface{} `json:"context,omitempty"`
	Timestamp    time.Time              `json:"timestamp,omitempty"`
}

// HealingResult represents the result of a healing operation
type HealingResult struct {
	HealingID         string                 `json:"healing_id"`
	Status            HealingStatus          `json:"status"`
	ErrorID           string                 `json:"error_id"`
	PatchesGenerated  int                    `json:"patches_generated"`
	PatchesApplied    int                    `json:"patches_applied"`
	Success           bool                   `json:"success"`
	DurationSeconds   float64                `json:"duration_seconds"`
	Logs              []string               `json:"logs"`
	RollbackAvailable bool                   `json:"rollback_available"`
	Metrics           map[string]interface{} `json:"metrics"`
}

// SystemHealth represents the system health status
type SystemHealth struct {
	Status               string             `json:"status"`
	UptimeSeconds        float64            `json:"uptime_seconds"`
	ActiveHealings       int                `json:"active_healings"`
	TotalErrorsProcessed int                `json:"total_errors_processed"`
	SuccessRate          float64            `json:"success_rate"`
	AverageHealingTime   float64            `json:"average_healing_time"`
	Components           map[string]string  `json:"components"`
}

// Client is the main API client for Homeostasis
type Client struct {
	baseURL    string
	apiKey     string
	httpClient *http.Client
	wsConn     *websocket.Conn
	wsCallbacks map[string][]func(interface{})
	wsMutex    sync.RWMutex
	wsReconnectAttempts int
	wsMaxReconnectAttempts int
}

// ClientOption is a function that configures a Client
type ClientOption func(*Client)

// WithAPIKey sets the API key for authentication
func WithAPIKey(apiKey string) ClientOption {
	return func(c *Client) {
		c.apiKey = apiKey
	}
}

// WithHTTPClient sets a custom HTTP client
func WithHTTPClient(httpClient *http.Client) ClientOption {
	return func(c *Client) {
		c.httpClient = httpClient
	}
}

// WithTimeout sets the HTTP client timeout
func WithTimeout(timeout time.Duration) ClientOption {
	return func(c *Client) {
		c.httpClient.Timeout = timeout
	}
}

// WithMaxReconnectAttempts sets the maximum WebSocket reconnect attempts
func WithMaxReconnectAttempts(attempts int) ClientOption {
	return func(c *Client) {
		c.wsMaxReconnectAttempts = attempts
	}
}

// NewClient creates a new Homeostasis client
func NewClient(baseURL string, opts ...ClientOption) *Client {
	c := &Client{
		baseURL: strings.TrimRight(baseURL, "/"),
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
		},
		wsCallbacks: make(map[string][]func(interface{})),
		wsMaxReconnectAttempts: 5,
	}

	for _, opt := range opts {
		opt(c)
	}

	return c
}

// makeRequest performs an HTTP request to the API
func (c *Client) makeRequest(ctx context.Context, method, endpoint string, body interface{}) ([]byte, error) {
	url := c.baseURL + endpoint

	var bodyReader io.Reader
	if body != nil {
		jsonBody, err := json.Marshal(body)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal request body: %w", err)
		}
		bodyReader = bytes.NewReader(jsonBody)
	}

	req, err := http.NewRequestWithContext(ctx, method, url, bodyReader)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", "Homeostasis-Go-Client/1.0")
	
	if c.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	if resp.StatusCode >= 400 {
		var errorResp struct {
			Message string `json:"message"`
		}
		json.Unmarshal(respBody, &errorResp)
		return nil, fmt.Errorf("API error (status %d): %s", resp.StatusCode, errorResp.Message)
	}

	return respBody, nil
}

// Core API Methods

// ReportError reports an error for healing
func (c *Client) ReportError(ctx context.Context, error *ErrorReport) (string, error) {
	if error.Timestamp.IsZero() {
		error.Timestamp = time.Now()
	}

	resp, err := c.makeRequest(ctx, http.MethodPost, "/api/v1/errors", error)
	if err != nil {
		return "", err
	}

	var result struct {
		ErrorID string `json:"error_id"`
	}
	if err := json.Unmarshal(resp, &result); err != nil {
		return "", fmt.Errorf("failed to parse response: %w", err)
	}

	return result.ErrorID, nil
}

// GetHealingStatus gets the status of a healing operation
func (c *Client) GetHealingStatus(ctx context.Context, healingID string) (*HealingResult, error) {
	resp, err := c.makeRequest(ctx, http.MethodGet, "/api/v1/healings/"+healingID, nil)
	if err != nil {
		return nil, err
	}

	var result HealingResult
	if err := json.Unmarshal(resp, &result); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	return &result, nil
}

// TriggerHealing manually triggers healing for an error
func (c *Client) TriggerHealing(ctx context.Context, errorID string, options map[string]interface{}) (string, error) {
	body := map[string]interface{}{
		"error_id": errorID,
	}
	if options != nil {
		body["options"] = options
	}

	resp, err := c.makeRequest(ctx, http.MethodPost, "/api/v1/healings", body)
	if err != nil {
		return "", err
	}

	var result struct {
		HealingID string `json:"healing_id"`
	}
	if err := json.Unmarshal(resp, &result); err != nil {
		return "", fmt.Errorf("failed to parse response: %w", err)
	}

	return result.HealingID, nil
}

// ListErrors lists reported errors with optional filters
func (c *Client) ListErrors(ctx context.Context, filters map[string]string, page, perPage int) (map[string]interface{}, error) {
	params := url.Values{}
	params.Set("page", fmt.Sprintf("%d", page))
	params.Set("per_page", fmt.Sprintf("%d", perPage))
	
	for k, v := range filters {
		params.Set(k, v)
	}

	endpoint := "/api/v1/errors?" + params.Encode()
	resp, err := c.makeRequest(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return nil, err
	}

	var result map[string]interface{}
	if err := json.Unmarshal(resp, &result); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	return result, nil
}

// GetSystemHealth gets the system health status
func (c *Client) GetSystemHealth(ctx context.Context) (*SystemHealth, error) {
	resp, err := c.makeRequest(ctx, http.MethodGet, "/api/v1/health", nil)
	if err != nil {
		return nil, err
	}

	var health SystemHealth
	if err := json.Unmarshal(resp, &health); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	return &health, nil
}

// GetMetrics gets system metrics
func (c *Client) GetMetrics(ctx context.Context, metricType, timeRange string) (map[string]interface{}, error) {
	params := url.Values{}
	params.Set("type", metricType)
	params.Set("range", timeRange)

	endpoint := "/api/v1/metrics?" + params.Encode()
	resp, err := c.makeRequest(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return nil, err
	}

	var result map[string]interface{}
	if err := json.Unmarshal(resp, &result); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	return result, nil
}

// RollbackHealing rolls back a healing operation
func (c *Client) RollbackHealing(ctx context.Context, healingID string, reason string) (bool, error) {
	body := map[string]interface{}{}
	if reason != "" {
		body["reason"] = reason
	}

	resp, err := c.makeRequest(ctx, http.MethodPost, "/api/v1/healings/"+healingID+"/rollback", body)
	if err != nil {
		return false, err
	}

	var result struct {
		Success bool `json:"success"`
	}
	if err := json.Unmarshal(resp, &result); err != nil {
		return false, fmt.Errorf("failed to parse response: %w", err)
	}

	return result.Success, nil
}

// ApproveHealing approves a healing operation
func (c *Client) ApproveHealing(ctx context.Context, healingID, approvedBy string) (bool, error) {
	body := map[string]interface{}{
		"approved_by": approvedBy,
	}

	resp, err := c.makeRequest(ctx, http.MethodPost, "/api/v1/healings/"+healingID+"/approve", body)
	if err != nil {
		return false, err
	}

	var result struct {
		Success bool `json:"success"`
	}
	if err := json.Unmarshal(resp, &result); err != nil {
		return false, fmt.Errorf("failed to parse response: %w", err)
	}

	return result.Success, nil
}

// GetPatches gets patches generated for a healing
func (c *Client) GetPatches(ctx context.Context, healingID string) ([]map[string]interface{}, error) {
	resp, err := c.makeRequest(ctx, http.MethodGet, "/api/v1/healings/"+healingID+"/patches", nil)
	if err != nil {
		return nil, err
	}

	var result struct {
		Patches []map[string]interface{} `json:"patches"`
	}
	if err := json.Unmarshal(resp, &result); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	return result.Patches, nil
}

// TestPatch tests a specific patch
func (c *Client) TestPatch(ctx context.Context, patchID string, testConfig map[string]interface{}) (map[string]interface{}, error) {
	body := map[string]interface{}{}
	if testConfig != nil {
		body["test_config"] = testConfig
	}

	resp, err := c.makeRequest(ctx, http.MethodPost, "/api/v1/patches/"+patchID+"/test", body)
	if err != nil {
		return nil, err
	}

	var result map[string]interface{}
	if err := json.Unmarshal(resp, &result); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	return result, nil
}

// WebSocket Support

// ConnectWebSocket connects to the WebSocket for real-time updates
func (c *Client) ConnectWebSocket(ctx context.Context, onMessage func(interface{}), onError func(error), onClose func()) error {
	wsURL := strings.Replace(c.baseURL, "http://", "ws://", 1)
	wsURL = strings.Replace(wsURL, "https://", "wss://", 1)
	wsURL += "/ws"

	dialer := websocket.Dialer{
		HandshakeTimeout: 10 * time.Second,
	}

	header := http.Header{}
	if c.apiKey != "" {
		header.Set("Authorization", "Bearer "+c.apiKey)
	}

	conn, _, err := dialer.DialContext(ctx, wsURL, header)
	if err != nil {
		return fmt.Errorf("failed to connect to WebSocket: %w", err)
	}

	c.wsConn = conn
	c.wsReconnectAttempts = 0

	// Authentication
	if c.apiKey != "" {
		authMsg := map[string]string{
			"type":    "auth",
			"api_key": c.apiKey,
		}
		if err := conn.WriteJSON(authMsg); err != nil {
			conn.Close()
			return fmt.Errorf("failed to authenticate WebSocket: %w", err)
		}
	}

	// Start message handler
	go c.handleWebSocketMessages(onMessage, onError, onClose)

	return nil
}

func (c *Client) handleWebSocketMessages(onMessage func(interface{}), onError func(error), onClose func()) {
	defer func() {
		c.wsConn.Close()
		if onClose != nil {
			onClose()
		}
		c.attemptReconnect(onMessage, onError, onClose)
	}()

	for {
		var msg map[string]interface{}
		err := c.wsConn.ReadJSON(&msg)
		if err != nil {
			if onError != nil {
				onError(err)
			}
			break
		}

		eventType, ok := msg["type"].(string)
		if ok {
			c.wsMutex.RLock()
			callbacks := c.wsCallbacks[eventType]
			c.wsMutex.RUnlock()

			for _, callback := range callbacks {
				go callback(msg)
			}
		}

		if onMessage != nil {
			onMessage(msg)
		}
	}
}

func (c *Client) attemptReconnect(onMessage func(interface{}), onError func(error), onClose func()) {
	if c.wsReconnectAttempts >= c.wsMaxReconnectAttempts {
		return
	}

	c.wsReconnectAttempts++
	backoff := time.Duration(1<<uint(c.wsReconnectAttempts)) * time.Second
	if backoff > 30*time.Second {
		backoff = 30 * time.Second
	}

	time.Sleep(backoff)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if err := c.ConnectWebSocket(ctx, onMessage, onError, onClose); err != nil {
		fmt.Printf("WebSocket reconnection attempt %d/%d failed: %v\n", 
			c.wsReconnectAttempts, c.wsMaxReconnectAttempts, err)
	}
}

// Subscribe subscribes to specific WebSocket events
func (c *Client) Subscribe(eventType string, callback func(interface{})) {
	c.wsMutex.Lock()
	defer c.wsMutex.Unlock()

	c.wsCallbacks[eventType] = append(c.wsCallbacks[eventType], callback)
}

// Unsubscribe unsubscribes from WebSocket events
func (c *Client) Unsubscribe(eventType string, callback func(interface{})) {
	c.wsMutex.Lock()
	defer c.wsMutex.Unlock()

	callbacks := c.wsCallbacks[eventType]
	for i, cb := range callbacks {
		if &cb == &callback {
			c.wsCallbacks[eventType] = append(callbacks[:i], callbacks[i+1:]...)
			break
		}
	}
}

// DisconnectWebSocket disconnects the WebSocket connection
func (c *Client) DisconnectWebSocket() error {
	if c.wsConn != nil {
		c.wsReconnectAttempts = c.wsMaxReconnectAttempts // Prevent reconnection
		return c.wsConn.Close()
	}
	return nil
}

// Batch Operations

// BatchReportErrors reports multiple errors in batch
func (c *Client) BatchReportErrors(ctx context.Context, errors []*ErrorReport) (map[string]interface{}, error) {
	for _, err := range errors {
		if err.Timestamp.IsZero() {
			err.Timestamp = time.Now()
		}
	}

	body := map[string]interface{}{
		"errors": errors,
	}

	resp, err := c.makeRequest(ctx, http.MethodPost, "/api/v1/errors/batch", body)
	if err != nil {
		return nil, err
	}

	var result map[string]interface{}
	if err := json.Unmarshal(resp, &result); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	return result, nil
}

// BatchGetStatus gets status of multiple healing operations
func (c *Client) BatchGetStatus(ctx context.Context, healingIDs []string) (map[string]*HealingResult, error) {
	body := map[string]interface{}{
		"healing_ids": healingIDs,
	}

	resp, err := c.makeRequest(ctx, http.MethodPost, "/api/v1/healings/batch/status", body)
	if err != nil {
		return nil, err
	}

	var response struct {
		Results map[string]*HealingResult `json:"results"`
	}
	if err := json.Unmarshal(resp, &response); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	return response.Results, nil
}

// Utility Methods

// WaitForHealing waits for a healing operation to complete
func (c *Client) WaitForHealing(ctx context.Context, healingID string, timeout time.Duration, pollInterval time.Duration) (*HealingResult, error) {
	deadline := time.Now().Add(timeout)

	for time.Now().Before(deadline) {
		result, err := c.GetHealingStatus(ctx, healingID)
		if err != nil {
			return nil, err
		}

		switch result.Status {
		case HealingStatusCompleted, HealingStatusFailed, HealingStatusRolledBack:
			return result, nil
		}

		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(pollInterval):
			// Continue polling
		}
	}

	return nil, fmt.Errorf("healing %s did not complete within %v", healingID, timeout)
}

// ExportLogs exports healing logs
func (c *Client) ExportLogs(ctx context.Context, healingID string, format string) (interface{}, error) {
	params := url.Values{}
	params.Set("format", format)

	endpoint := "/api/v1/healings/" + healingID + "/logs?" + params.Encode()
	resp, err := c.makeRequest(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return nil, err
	}

	if format == "json" {
		var result interface{}
		if err := json.Unmarshal(resp, &result); err != nil {
			return nil, fmt.Errorf("failed to parse response: %w", err)
		}
		return result, nil
	}

	return string(resp), nil
}

// Close closes the client and releases resources
func (c *Client) Close() error {
	return c.DisconnectWebSocket()
}

// Convenience functions

// QuickReportError quickly reports an error and returns the error ID
func QuickReportError(ctx context.Context, baseURL, apiKey, errorMessage, stackTrace, language string) (string, error) {
	client := NewClient(baseURL, WithAPIKey(apiKey))
	defer client.Close()

	error := &ErrorReport{
		ErrorMessage: errorMessage,
		StackTrace:   stackTrace,
		Language:     language,
		Timestamp:    time.Now(),
	}

	return client.ReportError(ctx, error)
}