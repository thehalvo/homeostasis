/**
 * USHS TypeScript Reference Implementation
 * Universal Self-Healing Standard v1.0 Client Library
 */

import { EventEmitter } from 'events';
import WebSocket from 'ws';
import fetch, { RequestInit } from 'node-fetch';

// Enums
export enum Severity {
  CRITICAL = 'critical',
  HIGH = 'high',
  MEDIUM = 'medium',
  LOW = 'low'
}

export enum SessionStatus {
  ACTIVE = 'active',
  COMPLETED = 'completed',
  FAILED = 'failed',
  CANCELLED = 'cancelled',
  TIMEOUT = 'timeout'
}

export enum DeploymentStrategy {
  IMMEDIATE = 'immediate',
  CANARY = 'canary',
  BLUE_GREEN = 'blue-green',
  ROLLING = 'rolling'
}

// Interfaces
export interface ErrorSource {
  service: string;
  version?: string;
  environment?: string;
  location?: string;
  hostname?: string;
  containerID?: string;
}

export interface ErrorDetails {
  type: string;
  message: string;
  code?: string;
  stackTrace?: Array<{
    file?: string;
    function?: string;
    line?: number;
    column?: number;
  }>;
  context?: Record<string, any>;
}

export interface ErrorEvent {
  id?: string;
  timestamp?: string;
  severity: Severity;
  source: ErrorSource;
  error: ErrorDetails;
  correlationId?: string;
  userImpact?: {
    affected?: number;
    severity?: 'none' | 'degraded' | 'unavailable';
  };
  metadata?: Record<string, any>;
}

export interface PatchChange {
  type: 'file' | 'configuration' | 'dependency' | 'infrastructure';
  target: string;
  operation?: 'create' | 'update' | 'delete' | 'replace';
  diff?: string;
  content?: string;
  language?: string;
  framework?: string;
  validation?: {
    syntax?: boolean;
    semantics?: boolean;
    tests?: string[];
  };
}

export interface HealingPatch {
  id?: string;
  sessionId: string;
  errorId: string;
  changes: PatchChange[];
  metadata: {
    confidence: number;
    generator: string;
    generatorVersion?: string;
    strategy?: string;
    reasoning?: string;
    alternatives?: Array<{
      id?: string;
      confidence?: number;
      reason?: string;
    }>;
    estimatedImpact?: {
      performance?: 'improved' | 'neutral' | 'degraded';
      reliability?: 'improved' | 'neutral' | 'degraded';
      security?: 'improved' | 'neutral' | 'degraded';
    };
  };
  approvals?: Array<{
    type?: 'automatic' | 'manual' | 'policy';
    approver?: string;
    status?: 'pending' | 'approved' | 'rejected';
    timestamp?: string;
    comment?: string;
  }>;
}

export interface HealingSession {
  id: string;
  startTime: string;
  endTime?: string;
  status: SessionStatus;
  errorId?: string;
  phases: {
    [key: string]: {
      status: 'pending' | 'running' | 'completed' | 'failed' | 'skipped';
      startTime?: string;
      endTime?: string;
      duration?: number;
      result?: any;
      errors?: Array<{
        code?: string;
        message?: string;
        details?: any;
      }>;
    };
  };
}

export interface USHSClientOptions {
  baseUrl: string;
  authToken?: string;
  apiKey?: string;
  verifySsl?: boolean;
  timeout?: number;
}

export interface CloudEvent {
  specversion: string;
  id: string;
  source: string;
  type: string;
  datacontenttype: string;
  time: string;
  subject?: string;
  data: any;
}

/**
 * USHS Client Implementation
 */
export class USHSClient extends EventEmitter {
  private baseUrl: string;
  private authToken?: string;
  private apiKey?: string;
  private verifySsl: boolean;
  private timeout: number;
  private ws?: WebSocket;
  private wsReconnectTimer?: NodeJS.Timeout;
  private wsReconnectAttempts = 0;

  constructor(options: USHSClientOptions) {
    super();
    this.baseUrl = options.baseUrl.replace(/\/$/, '');
    this.authToken = options.authToken;
    this.apiKey = options.apiKey;
    this.verifySsl = options.verifySsl ?? true;
    this.timeout = options.timeout ?? 30000;
  }

  /**
   * Build request headers
   */
  private getHeaders(): Record<string, string> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      'Accept': 'application/json'
    };

    if (this.authToken) {
      headers['Authorization'] = `Bearer ${this.authToken}`;
    } else if (this.apiKey) {
      headers['X-API-Key'] = this.apiKey;
    }

    return headers;
  }

  /**
   * Make HTTP request
   */
  private async request<T>(
    method: string,
    path: string,
    data?: any,
    queryParams?: Record<string, any>
  ): Promise<T> {
    let url = `${this.baseUrl}${path}`;
    
    // Add query parameters
    if (queryParams) {
      const params = new URLSearchParams();
      Object.entries(queryParams).forEach(([key, value]) => {
        if (value !== undefined && value !== null) {
          params.append(key, String(value));
        }
      });
      const queryString = params.toString();
      if (queryString) {
        url += `?${queryString}`;
      }
    }

    const options: RequestInit = {
      method,
      headers: this.getHeaders(),
      timeout: this.timeout,
    };

    if (data && ['POST', 'PUT', 'PATCH'].includes(method)) {
      options.body = JSON.stringify(data);
    }

    const response = await fetch(url, options);

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`HTTP ${response.status}: ${error}`);
    }

    if (response.status === 204) {
      return {} as T;
    }

    return response.json() as Promise<T>;
  }

  /**
   * Generate UUID v4
   */
  private generateId(): string {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
      const r = (Math.random() * 16) | 0;
      const v = c === 'x' ? r : (r & 0x3) | 0x8;
      return v.toString(16);
    });
  }

  /**
   * Get current ISO timestamp
   */
  private getTimestamp(): string {
    return new Date().toISOString();
  }

  // Error Management

  /**
   * Report a new error
   */
  async reportError(error: ErrorEvent): Promise<{ errorId: string; sessionId: string }> {
    // Add defaults
    if (!error.id) {
      error.id = this.generateId();
    }
    if (!error.timestamp) {
      error.timestamp = this.getTimestamp();
    }

    return this.request<{ errorId: string; sessionId: string }>(
      'POST',
      '/errors',
      error
    );
  }

  /**
   * Get error details
   */
  async getError(errorId: string): Promise<ErrorEvent> {
    return this.request<ErrorEvent>('GET', `/errors/${errorId}`);
  }

  // Session Management

  /**
   * Start a new healing session
   */
  async startSession(
    errorId: string,
    policy?: string,
    priority?: string
  ): Promise<HealingSession> {
    const data: any = { errorId };
    if (policy) data.policy = policy;
    if (priority) data.priority = priority;

    return this.request<HealingSession>('POST', '/sessions', data);
  }

  /**
   * Get session details
   */
  async getSession(sessionId: string): Promise<HealingSession> {
    return this.request<HealingSession>('GET', `/sessions/${sessionId}`);
  }

  /**
   * List healing sessions
   */
  async listSessions(options?: {
    status?: SessionStatus;
    limit?: number;
    offset?: number;
  }): Promise<{
    sessions: HealingSession[];
    total: number;
    limit: number;
    offset: number;
  }> {
    return this.request('GET', '/sessions', undefined, options);
  }

  /**
   * Cancel a healing session
   */
  async cancelSession(sessionId: string): Promise<void> {
    await this.request<void>('DELETE', `/sessions/${sessionId}`);
  }

  // Patch Management

  /**
   * Get patches for a session
   */
  async getSessionPatches(sessionId: string): Promise<HealingPatch[]> {
    return this.request<HealingPatch[]>('GET', `/sessions/${sessionId}/patches`);
  }

  /**
   * Submit a patch for a session
   */
  async submitPatch(sessionId: string, patch: HealingPatch): Promise<HealingPatch> {
    // Add defaults
    if (!patch.id) {
      patch.id = this.generateId();
    }

    return this.request<HealingPatch>('POST', `/sessions/${sessionId}/patches`, patch);
  }

  // Validation

  /**
   * Validate a patch
   */
  async validatePatch(
    patchId: string,
    options?: {
      tests?: string[];
      environment?: string;
    }
  ): Promise<{
    valid: boolean;
    results: Array<{
      test: string;
      passed: boolean;
      duration: number;
      output?: string;
    }>;
  }> {
    return this.request('POST', `/patches/${patchId}/validate`, options);
  }

  // Deployment

  /**
   * Deploy a patch
   */
  async deployPatch(
    patchId: string,
    strategy: DeploymentStrategy,
    environment: string,
    approvals?: Array<{
      approver: string;
      signature: string;
    }>
  ): Promise<{
    deploymentId: string;
    status: string;
    startTime: string;
  }> {
    const data: any = { strategy, environment };
    if (approvals) data.approvals = approvals;

    return this.request('POST', `/patches/${patchId}/deploy`, data);
  }

  // Health Check

  /**
   * Check system health
   */
  async healthCheck(): Promise<{
    status: 'healthy' | 'degraded' | 'unhealthy';
    components: Record<string, {
      status: string;
      message?: string;
    }>;
  }> {
    return this.request('GET', '/health');
  }

  // WebSocket Support

  /**
   * Connect to WebSocket for real-time events
   */
  connectWebSocket(options?: {
    subscribe?: string[];
    session?: string;
    service?: string;
  }): void {
    let wsUrl = this.baseUrl.replace(/^https?:/, 'ws:').replace(/^https:/, 'wss:');
    wsUrl += '/ws';

    // Build query params
    const params: string[] = [];
    if (options?.subscribe) {
      params.push(`subscribe=${options.subscribe.join(',')}`);
    }
    if (options?.session) {
      params.push(`session=${options.session}`);
    }
    if (options?.service) {
      params.push(`service=${options.service}`);
    }
    if (this.apiKey) {
      params.push(`apikey=${this.apiKey}`);
    }

    if (params.length > 0) {
      wsUrl += '?' + params.join('&');
    }

    // Build headers
    const headers: Record<string, string> = {};
    if (this.authToken && !this.apiKey) {
      headers['Authorization'] = `Bearer ${this.authToken}`;
    }

    // Connect
    this.ws = new WebSocket(wsUrl, { headers });

    this.ws.on('open', () => {
      this.wsReconnectAttempts = 0;
      this.emit('ws:connected');
    });

    this.ws.on('message', (data: WebSocket.Data) => {
      try {
        const event = JSON.parse(data.toString()) as CloudEvent;
        this.emit('event', event);
        this.emit(event.type, event);
      } catch (error) {
        this.emit('ws:error', new Error(`Invalid message: ${data}`));
      }
    });

    this.ws.on('error', (error: Error) => {
      this.emit('ws:error', error);
    });

    this.ws.on('close', (code: number, reason: string) => {
      this.emit('ws:disconnected', { code, reason });
      this.attemptReconnect();
    });

    // Ping interval
    const pingInterval = setInterval(() => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        this.ws.ping();
      }
    }, 30000);

    this.ws.on('close', () => {
      clearInterval(pingInterval);
    });
  }

  /**
   * Attempt to reconnect WebSocket
   */
  private attemptReconnect(): void {
    if (this.wsReconnectAttempts >= 10) {
      this.emit('ws:reconnectFailed');
      return;
    }

    const delay = Math.min(1000 * Math.pow(2, this.wsReconnectAttempts), 60000);
    this.wsReconnectAttempts++;

    this.wsReconnectTimer = setTimeout(() => {
      this.emit('ws:reconnecting', { attempt: this.wsReconnectAttempts });
      this.connectWebSocket();
    }, delay);
  }

  /**
   * Subscribe to event types
   */
  async subscribe(eventTypes: string[], filters?: Record<string, any>): Promise<void> {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      throw new Error('WebSocket not connected');
    }

    const command: any = {
      command: 'subscribe',
      eventTypes
    };
    if (filters) {
      command.filters = filters;
    }

    this.ws.send(JSON.stringify(command));
  }

  /**
   * Unsubscribe from event types
   */
  async unsubscribe(eventTypes: string[]): Promise<void> {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      throw new Error('WebSocket not connected');
    }

    this.ws.send(JSON.stringify({
      command: 'unsubscribe',
      eventTypes
    }));
  }

  /**
   * Disconnect WebSocket
   */
  disconnectWebSocket(): void {
    if (this.wsReconnectTimer) {
      clearTimeout(this.wsReconnectTimer);
      this.wsReconnectTimer = undefined;
    }

    if (this.ws) {
      this.ws.close();
      this.ws = undefined;
    }
  }

  /**
   * Clean up resources
   */
  destroy(): void {
    this.disconnectWebSocket();
    this.removeAllListeners();
  }
}

// Example usage
async function exampleUsage() {
  // Create client
  const client = new USHSClient({
    baseUrl: 'https://api.example.com/ushs/v1',
    authToken: 'your-auth-token'
  });

  try {
    // Report an error
    const { errorId, sessionId } = await client.reportError({
      severity: Severity.HIGH,
      source: {
        service: 'api-gateway',
        environment: 'production'
      },
      error: {
        type: 'NullPointerException',
        message: "Cannot read property 'id' of null",
        stackTrace: [
          { file: 'app.js', function: 'getUser', line: 42 }
        ]
      }
    });

    console.log(`Error reported: ${errorId}`);
    console.log(`Session started: ${sessionId}`);

    // Connect WebSocket for real-time updates
    client.connectWebSocket({
      subscribe: ['session', 'patch'],
      session: sessionId
    });

    // Listen for events
    client.on('org.ushs.patch.generated', (event: CloudEvent) => {
      console.log(`Patch generated: ${event.data.patchId}`);
    });

    client.on('org.ushs.session.completed', (event: CloudEvent) => {
      console.log(`Session completed: ${event.data.sessionId}`);
    });

    // Check session status
    const session = await client.getSession(sessionId);
    console.log(`Session status: ${session.status}`);

    // Wait for healing to complete
    await new Promise(resolve => setTimeout(resolve, 30000));

  } finally {
    // Clean up
    client.destroy();
  }
}

// Run example if this is the main module
if (require.main === module) {
  exampleUsage().catch(console.error);
}

export default USHSClient;