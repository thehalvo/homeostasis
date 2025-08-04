// Package ushs provides a Go client implementation for the Universal Self-Healing Standard v1.0
package ushs

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

	"github.com/google/uuid"
	"github.com/gorilla/websocket"
)

// Severity levels for errors
type Severity string

const (
	SeverityCritical Severity = "critical"
	SeverityHigh     Severity = "high"
	SeverityMedium   Severity = "medium"
	SeverityLow      Severity = "low"
)

// SessionStatus represents the status of a healing session
type SessionStatus string

const (
	SessionStatusActive    SessionStatus = "active"
	SessionStatusCompleted SessionStatus = "completed"
	SessionStatusFailed    SessionStatus = "failed"
	SessionStatusCancelled SessionStatus = "cancelled"
	SessionStatusTimeout   SessionStatus = "timeout"
)

// DeploymentStrategy represents deployment strategies
type DeploymentStrategy string

const (
	DeploymentStrategyImmediate  DeploymentStrategy = "immediate"
	DeploymentStrategyCanary     DeploymentStrategy = "canary"
	DeploymentStrategyBlueGreen  DeploymentStrategy = "blue-green"
	DeploymentStrategyRolling    DeploymentStrategy = "rolling"
)

// ErrorSource contains information about where an error originated
type ErrorSource struct {
	Service     string `json:"service"`
	Version     string `json:"version,omitempty"`
	Environment string `json:"environment,omitempty"`
	Location    string `json:"location,omitempty"`
	Hostname    string `json:"hostname,omitempty"`
	ContainerID string `json:"containerID,omitempty"`
}

// StackFrame represents a single frame in a stack trace
type StackFrame struct {
	File     string `json:"file,omitempty"`
	Function string `json:"function,omitempty"`
	Line     int    `json:"line,omitempty"`
	Column   int    `json:"column,omitempty"`
}

// ErrorDetails contains details about an error
type ErrorDetails struct {
	Type       string                 `json:"type"`
	Message    string                 `json:"message"`
	Code       string                 `json:"code,omitempty"`
	StackTrace []StackFrame           `json:"stackTrace,omitempty"`
	Context    map[string]interface{} `json:"context,omitempty"`
}

// UserImpact describes the impact on users
type UserImpact struct {
	Affected int    `json:"affected,omitempty"`
	Severity string `json:"severity,omitempty"`
}

// ErrorEvent represents an error in the USHS format
type ErrorEvent struct {
	ID            string                 `json:"id"`
	Timestamp     string                 `json:"timestamp"`
	Severity      Severity               `json:"severity"`
	Source        ErrorSource            `json:"source"`
	Error         ErrorDetails           `json:"error"`
	CorrelationID string                 `json:"correlationId,omitempty"`
	UserImpact    *UserImpact            `json:"userImpact,omitempty"`
	Metadata      map[string]interface{} `json:"metadata,omitempty"`
}

// PatchChange represents a single change in a patch
type PatchChange struct {
	Type       string                 `json:"type"`
	Target     string                 `json:"target"`
	Operation  string                 `json:"operation,omitempty"`
	Diff       string                 `json:"diff,omitempty"`
	Content    string                 `json:"content,omitempty"`
	Language   string                 `json:"language,omitempty"`
	Framework  string                 `json:"framework,omitempty"`
	Validation map[string]interface{} `json:"validation,omitempty"`
}

// PatchMetadata contains metadata about a patch
type PatchMetadata struct {
	Confidence       float64                  `json:"confidence"`
	Generator        string                   `json:"generator"`
	GeneratorVersion string                   `json:"generatorVersion,omitempty"`
	Strategy         string                   `json:"strategy,omitempty"`
	Reasoning        string                   `json:"reasoning,omitempty"`
	Alternatives     []map[string]interface{} `json:"alternatives,omitempty"`
	EstimatedImpact  map[string]string        `json:"estimatedImpact,omitempty"`
}

// HealingPatch represents a patch to fix an error
type HealingPatch struct {
	ID        string                   `json:"id"`
	SessionID string                   `json:"sessionId"`
	ErrorID   string                   `json:"errorId"`
	Changes   []PatchChange            `json:"changes"`
	Metadata  PatchMetadata            `json:"metadata"`
	Approvals []map[string]interface{} `json:"approvals,omitempty"`
}

// Phase represents a phase in the healing process
type Phase struct {
	Status    string                 `json:"status"`
	StartTime string                 `json:"startTime,omitempty"`
	EndTime   string                 `json:"endTime,omitempty"`
	Duration  int                    `json:"duration,omitempty"`
	Result    interface{}            `json:"result,omitempty"`
	Errors    []map[string]interface{} `json:"errors,omitempty"`
}

// HealingSession represents a healing session
type HealingSession struct {
	ID        string           `json:"id"`
	StartTime string           `json:"startTime"`
	EndTime   string           `json:"endTime,omitempty"`
	Status    SessionStatus    `json:"status"`
	ErrorID   string           `json:"errorId,omitempty"`
	Phases    map[string]Phase `json:"phases"`
}

// CloudEvent represents a CloudEvents-compliant event
type CloudEvent struct {
	SpecVersion     string      `json:"specversion"`
	ID              string      `json:"id"`
	Source          string      `json:"source"`
	Type            string      `json:"type"`
	DataContentType string      `json:"datacontenttype"`
	Time            string      `json:"time"`
	Subject         string      `json:"subject,omitempty"`
	Data            interface{} `json:"data"`
}

// Client represents a USHS client
type Client struct {
	baseURL    string
	httpClient *http.Client
	authToken  string
	apiKey     string
	ws         *websocket.Conn
	wsURL      string
	wsMutex    sync.Mutex
	handlers   map[string][]EventHandler
	handlerMu  sync.RWMutex
}

// ClientOption configures a Client
type ClientOption func(*Client)

// EventHandler handles WebSocket events
type EventHandler func(event CloudEvent)

// WithHTTPClient sets a custom HTTP client
func WithHTTPClient(client *http.Client) ClientOption {
	return func(c *Client) {
		c.httpClient = client
	}
}

// WithAuthToken sets the authentication token
func WithAuthToken(token string) ClientOption {
	return func(c *Client) {
		c.authToken = token
	}
}

// WithAPIKey sets the API key
func WithAPIKey(key string) ClientOption {
	return func(c *Client) {
		c.apiKey = key
	}
}

// NewClient creates a new USHS client
func NewClient(baseURL string, opts ...ClientOption) *Client {
	c := &Client{
		baseURL: strings.TrimRight(baseURL, "/"),
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
		},
		handlers: make(map[string][]EventHandler),
	}

	for _, opt := range opts {
		opt(c)
	}

	// Set WebSocket URL
	c.wsURL = strings.Replace(c.baseURL, "https://", "wss://", 1)
	c.wsURL = strings.Replace(c.wsURL, "http://", "ws://", 1)
	c.wsURL += "/ws"

	return c
}

// doRequest performs an HTTP request
func (c *Client) doRequest(ctx context.Context, method, path string, body interface{}, result interface{}) error {
	var bodyReader io.Reader
	if body != nil {
		jsonBody, err := json.Marshal(body)
		if err != nil {
			return fmt.Errorf("marshal request body: %w", err)
		}
		bodyReader = bytes.NewReader(jsonBody)
	}

	req, err := http.NewRequestWithContext(ctx, method, c.baseURL+path, bodyReader)
	if err != nil {
		return fmt.Errorf("create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")

	if c.authToken != "" {
		req.Header.Set("Authorization", "Bearer "+c.authToken)
	} else if c.apiKey != "" {
		req.Header.Set("X-API-Key", c.apiKey)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("do request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(body))
	}

	if resp.StatusCode == http.StatusNoContent {
		return nil
	}

	if result != nil {
		if err := json.NewDecoder(resp.Body).Decode(result); err != nil {
			return fmt.Errorf("decode response: %w", err)
		}
	}

	return nil
}

// ReportError reports a new error to the healing system
func (c *Client) ReportError(ctx context.Context, event *ErrorEvent) (errorID, sessionID string, err error) {
	// Set defaults
	if event.ID == "" {
		event.ID = uuid.New().String()
	}
	if event.Timestamp == "" {
		event.Timestamp = time.Now().UTC().Format(time.RFC3339)
	}

	var resp struct {
		ErrorID   string `json:"errorId"`
		SessionID string `json:"sessionId"`
	}

	err = c.doRequest(ctx, http.MethodPost, "/errors", event, &resp)
	if err != nil {
		return "", "", err
	}

	return resp.ErrorID, resp.SessionID, nil
}

// GetError retrieves error details
func (c *Client) GetError(ctx context.Context, errorID string) (*ErrorEvent, error) {
	var event ErrorEvent
	err := c.doRequest(ctx, http.MethodGet, "/errors/"+errorID, nil, &event)
	if err != nil {
		return nil, err
	}
	return &event, nil
}

// StartSession starts a new healing session
func (c *Client) StartSession(ctx context.Context, errorID string, policy *string, priority *string) (*HealingSession, error) {
	req := map[string]interface{}{
		"errorId": errorID,
	}
	if policy != nil {
		req["policy"] = *policy
	}
	if priority != nil {
		req["priority"] = *priority
	}

	var session HealingSession
	err := c.doRequest(ctx, http.MethodPost, "/sessions", req, &session)
	if err != nil {
		return nil, err
	}
	return &session, nil
}

// GetSession retrieves session details
func (c *Client) GetSession(ctx context.Context, sessionID string) (*HealingSession, error) {
	var session HealingSession
	err := c.doRequest(ctx, http.MethodGet, "/sessions/"+sessionID, nil, &session)
	if err != nil {
		return nil, err
	}
	return &session, nil
}

// ListSessionsOptions contains options for listing sessions
type ListSessionsOptions struct {
	Status *SessionStatus
	Limit  int
	Offset int
}

// ListSessionsResponse contains the response from listing sessions
type ListSessionsResponse struct {
	Sessions []HealingSession `json:"sessions"`
	Total    int              `json:"total"`
	Limit    int              `json:"limit"`
	Offset   int              `json:"offset"`
}

// ListSessions lists healing sessions
func (c *Client) ListSessions(ctx context.Context, opts *ListSessionsOptions) (*ListSessionsResponse, error) {
	query := url.Values{}
	if opts != nil {
		if opts.Status != nil {
			query.Set("status", string(*opts.Status))
		}
		if opts.Limit > 0 {
			query.Set("limit", fmt.Sprintf("%d", opts.Limit))
		}
		if opts.Offset > 0 {
			query.Set("offset", fmt.Sprintf("%d", opts.Offset))
		}
	}

	path := "/sessions"
	if len(query) > 0 {
		path += "?" + query.Encode()
	}

	var resp ListSessionsResponse
	err := c.doRequest(ctx, http.MethodGet, path, nil, &resp)
	if err != nil {
		return nil, err
	}
	return &resp, nil
}

// CancelSession cancels a healing session
func (c *Client) CancelSession(ctx context.Context, sessionID string) error {
	return c.doRequest(ctx, http.MethodDelete, "/sessions/"+sessionID, nil, nil)
}

// GetSessionPatches retrieves patches for a session
func (c *Client) GetSessionPatches(ctx context.Context, sessionID string) ([]HealingPatch, error) {
	var patches []HealingPatch
	err := c.doRequest(ctx, http.MethodGet, "/sessions/"+sessionID+"/patches", nil, &patches)
	if err != nil {
		return nil, err
	}
	return patches, nil
}

// SubmitPatch submits a patch for a session
func (c *Client) SubmitPatch(ctx context.Context, sessionID string, patch *HealingPatch) (*HealingPatch, error) {
	if patch.ID == "" {
		patch.ID = uuid.New().String()
	}

	var result HealingPatch
	err := c.doRequest(ctx, http.MethodPost, "/sessions/"+sessionID+"/patches", patch, &result)
	if err != nil {
		return nil, err
	}
	return &result, nil
}

// ValidatePatchOptions contains options for patch validation
type ValidatePatchOptions struct {
	Tests       []string `json:"tests,omitempty"`
	Environment string   `json:"environment,omitempty"`
}

// ValidationResult contains the result of patch validation
type ValidationResult struct {
	Valid   bool `json:"valid"`
	Results []struct {
		Test     string `json:"test"`
		Passed   bool   `json:"passed"`
		Duration int    `json:"duration"`
		Output   string `json:"output,omitempty"`
	} `json:"results"`
}

// ValidatePatch validates a patch
func (c *Client) ValidatePatch(ctx context.Context, patchID string, opts *ValidatePatchOptions) (*ValidationResult, error) {
	var result ValidationResult
	err := c.doRequest(ctx, http.MethodPost, "/patches/"+patchID+"/validate", opts, &result)
	if err != nil {
		return nil, err
	}
	return &result, nil
}

// DeployPatchOptions contains options for patch deployment
type DeployPatchOptions struct {
	Strategy    DeploymentStrategy `json:"strategy"`
	Environment string             `json:"environment"`
	Approvals   []struct {
		Approver  string `json:"approver"`
		Signature string `json:"signature"`
	} `json:"approvals,omitempty"`
}

// DeploymentResult contains the result of patch deployment
type DeploymentResult struct {
	DeploymentID string `json:"deploymentId"`
	Status       string `json:"status"`
	StartTime    string `json:"startTime"`
}

// DeployPatch deploys a patch
func (c *Client) DeployPatch(ctx context.Context, patchID string, opts *DeployPatchOptions) (*DeploymentResult, error) {
	var result DeploymentResult
	err := c.doRequest(ctx, http.MethodPost, "/patches/"+patchID+"/deploy", opts, &result)
	if err != nil {
		return nil, err
	}
	return &result, nil
}

// HealthCheckResult contains the result of a health check
type HealthCheckResult struct {
	Status     string `json:"status"`
	Components map[string]struct {
		Status  string `json:"status"`
		Message string `json:"message,omitempty"`
	} `json:"components"`
}

// HealthCheck checks system health
func (c *Client) HealthCheck(ctx context.Context) (*HealthCheckResult, error) {
	var result HealthCheckResult
	err := c.doRequest(ctx, http.MethodGet, "/health", nil, &result)
	if err != nil {
		return nil, err
	}
	return &result, nil
}

// ConnectWebSocket connects to the WebSocket for real-time events
func (c *Client) ConnectWebSocket(ctx context.Context, subscribe []string, session, service string) error {
	c.wsMutex.Lock()
	defer c.wsMutex.Unlock()

	if c.ws != nil {
		c.ws.Close()
	}

	// Build URL with query parameters
	u, err := url.Parse(c.wsURL)
	if err != nil {
		return fmt.Errorf("parse WebSocket URL: %w", err)
	}

	q := u.Query()
	if len(subscribe) > 0 {
		q.Set("subscribe", strings.Join(subscribe, ","))
	}
	if session != "" {
		q.Set("session", session)
	}
	if service != "" {
		q.Set("service", service)
	}
	if c.apiKey != "" {
		q.Set("apikey", c.apiKey)
	}
	u.RawQuery = q.Encode()

	// Set headers
	header := http.Header{}
	if c.authToken != "" && c.apiKey == "" {
		header.Set("Authorization", "Bearer "+c.authToken)
	}

	// Connect
	ws, _, err := websocket.DefaultDialer.DialContext(ctx, u.String(), header)
	if err != nil {
		return fmt.Errorf("dial WebSocket: %w", err)
	}

	c.ws = ws

	// Start message handler
	go c.handleWebSocketMessages()

	return nil
}

// handleWebSocketMessages handles incoming WebSocket messages
func (c *Client) handleWebSocketMessages() {
	defer func() {
		c.wsMutex.Lock()
		if c.ws != nil {
			c.ws.Close()
			c.ws = nil
		}
		c.wsMutex.Unlock()
	}()

	for {
		var event CloudEvent
		err := c.ws.ReadJSON(&event)
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				// Log error
			}
			return
		}

		c.handlerMu.RLock()
		handlers := make([]EventHandler, 0)
		if h, ok := c.handlers[event.Type]; ok {
			handlers = append(handlers, h...)
		}
		if h, ok := c.handlers["*"]; ok {
			handlers = append(handlers, h...)
		}
		c.handlerMu.RUnlock()

		for _, handler := range handlers {
			go handler(event)
		}
	}
}

// OnEvent registers an event handler
func (c *Client) OnEvent(eventType string, handler EventHandler) {
	c.handlerMu.Lock()
	defer c.handlerMu.Unlock()

	c.handlers[eventType] = append(c.handlers[eventType], handler)
}

// Subscribe subscribes to additional event types
func (c *Client) Subscribe(eventTypes []string, filters map[string]interface{}) error {
	c.wsMutex.Lock()
	defer c.wsMutex.Unlock()

	if c.ws == nil {
		return fmt.Errorf("WebSocket not connected")
	}

	cmd := map[string]interface{}{
		"command":    "subscribe",
		"eventTypes": eventTypes,
	}
	if filters != nil {
		cmd["filters"] = filters
	}

	return c.ws.WriteJSON(cmd)
}

// Unsubscribe unsubscribes from event types
func (c *Client) Unsubscribe(eventTypes []string) error {
	c.wsMutex.Lock()
	defer c.wsMutex.Unlock()

	if c.ws == nil {
		return fmt.Errorf("WebSocket not connected")
	}

	cmd := map[string]interface{}{
		"command":    "unsubscribe",
		"eventTypes": eventTypes,
	}

	return c.ws.WriteJSON(cmd)
}

// Close closes the client and cleans up resources
func (c *Client) Close() error {
	c.wsMutex.Lock()
	defer c.wsMutex.Unlock()

	if c.ws != nil {
		err := c.ws.Close()
		c.ws = nil
		return err
	}

	return nil
}