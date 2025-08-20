/**
 * Homeostasis JavaScript/TypeScript API Client
 * 
 * A JavaScript client library for interacting with the Homeostasis
 * self-healing framework.
 */

class HomeostasisClient {
  constructor(baseUrl, apiKey = null, options = {}) {
    this.baseUrl = baseUrl.replace(/\/$/, '');
    this.apiKey = apiKey;
    this.timeout = options.timeout || 30000;
    this.headers = {
      'Content-Type': 'application/json',
      'User-Agent': 'Homeostasis-JS-Client/1.0',
      ...options.headers
    };
    
    if (this.apiKey) {
      this.headers['Authorization'] = `Bearer ${this.apiKey}`;
    }
    
    this.ws = null;
    this.wsCallbacks = {};
    this.wsReconnectAttempts = 0;
    this.wsMaxReconnectAttempts = options.wsMaxReconnectAttempts || 5;
  }

  // HTTP Request Helper
  async _makeRequest(method, endpoint, options = {}) {
    const url = `${this.baseUrl}${endpoint}`;
    const config = {
      method,
      headers: { ...this.headers, ...options.headers },
      ...options
    };

    if (options.body && typeof options.body === 'object') {
      config.body = JSON.stringify(options.body);
    }

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(url, {
        ...config,
        signal: controller.signal
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const error = await response.json().catch(() => ({ message: response.statusText }));
        throw new HomeostasisError(error.message || response.statusText, response.status);
      }

      return await response.json();
    } catch (error) {
      clearTimeout(timeoutId);
      
      if (error.name === 'AbortError') {
        throw new HomeostasisError('Request timeout', 408);
      }
      
      throw error;
    }
  }

  // Core API Methods

  async reportError(error) {
    const errorData = error instanceof ErrorReport ? error.toJSON() : error;
    return await this._makeRequest('POST', '/api/v1/errors', { body: errorData });
  }

  async getHealingStatus(healingId) {
    const data = await this._makeRequest('GET', `/api/v1/healings/${healingId}`);
    return new HealingResult(data);
  }

  async triggerHealing(errorId, options = {}) {
    const data = await this._makeRequest('POST', '/api/v1/healings', {
      body: { error_id: errorId, options }
    });
    return data.healing_id;
  }

  async listErrors(filters = {}, page = 1, perPage = 50) {
    const params = new URLSearchParams({
      page: page.toString(),
      per_page: perPage.toString(),
      ...filters
    });
    
    return await this._makeRequest('GET', `/api/v1/errors?${params}`);
  }

  async getSystemHealth() {
    const data = await this._makeRequest('GET', '/api/v1/health');
    return new SystemHealth(data);
  }

  async getMetrics(metricType, timeRange = '1h') {
    const params = new URLSearchParams({
      type: metricType,
      range: timeRange
    });
    
    return await this._makeRequest('GET', `/api/v1/metrics?${params}`);
  }

  async rollbackHealing(healingId, reason = null) {
    const body = reason ? { reason } : {};
    const data = await this._makeRequest('POST', `/api/v1/healings/${healingId}/rollback`, { body });
    return data.success;
  }

  async approveHealing(healingId, approvedBy) {
    const data = await this._makeRequest('POST', `/api/v1/healings/${healingId}/approve`, {
      body: { approved_by: approvedBy }
    });
    return data.success;
  }

  async getPatches(healingId) {
    const data = await this._makeRequest('GET', `/api/v1/healings/${healingId}/patches`);
    return data.patches;
  }

  async testPatch(patchId, testConfig = null) {
    const body = testConfig ? { test_config: testConfig } : {};
    return await this._makeRequest('POST', `/api/v1/patches/${patchId}/test`, { body });
  }

  // Configuration Management

  async getConfig(component = null) {
    const endpoint = component ? `/api/v1/config/${component}` : '/api/v1/config';
    return await this._makeRequest('GET', endpoint);
  }

  async updateConfig(component, config) {
    const data = await this._makeRequest('PUT', `/api/v1/config/${component}`, { body: config });
    return data.success;
  }

  // Rule Management

  async listRules(language = null, category = null) {
    const params = new URLSearchParams();
    if (language) params.append('language', language);
    if (category) params.append('category', category);
    
    const data = await this._makeRequest('GET', `/api/v1/rules?${params}`);
    return data.rules;
  }

  async getRule(ruleId) {
    return await this._makeRequest('GET', `/api/v1/rules/${ruleId}`);
  }

  async createCustomRule(ruleData) {
    const data = await this._makeRequest('POST', '/api/v1/rules', { body: ruleData });
    return data.rule_id;
  }

  async updateRule(ruleId, ruleData) {
    const data = await this._makeRequest('PUT', `/api/v1/rules/${ruleId}`, { body: ruleData });
    return data.success;
  }

  async deleteRule(ruleId) {
    const data = await this._makeRequest('DELETE', `/api/v1/rules/${ruleId}`);
    return data.success;
  }

  // WebSocket Support

  connectWebSocket(onMessage = null, onError = null, onClose = null) {
    const wsUrl = this.baseUrl.replace(/^http/, 'ws') + '/ws';
    
    this.ws = new WebSocket(wsUrl);
    
    this.ws.onopen = () => {
      this.wsReconnectAttempts = 0;
      
      // Authenticate if API key is present
      if (this.apiKey) {
        this.ws.send(JSON.stringify({
          type: 'auth',
          api_key: this.apiKey
        }));
      }
    };
    
    this.ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      const eventType = data.type;
      
      // Call registered callbacks
      if (this.wsCallbacks[eventType]) {
        this.wsCallbacks[eventType].forEach(callback => callback(data));
      }
      
      // Call general message handler
      if (onMessage) {
        onMessage(data);
      }
    };
    
    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      if (onError) {
        onError(error);
      }
    };
    
    this.ws.onclose = () => {
      console.log('WebSocket connection closed');
      if (onClose) {
        onClose();
      }
      
      // Attempt to reconnect
      if (this.wsReconnectAttempts < this.wsMaxReconnectAttempts) {
        this.wsReconnectAttempts++;
        setTimeout(() => {
          console.log(`Attempting WebSocket reconnection (${this.wsReconnectAttempts}/${this.wsMaxReconnectAttempts})`);
          this.connectWebSocket(onMessage, onError, onClose);
        }, Math.min(1000 * Math.pow(2, this.wsReconnectAttempts), 30000));
      }
    };
  }

  subscribe(eventType, callback) {
    if (!this.wsCallbacks[eventType]) {
      this.wsCallbacks[eventType] = [];
    }
    this.wsCallbacks[eventType].push(callback);
  }

  unsubscribe(eventType, callback) {
    if (this.wsCallbacks[eventType]) {
      this.wsCallbacks[eventType] = this.wsCallbacks[eventType].filter(cb => cb !== callback);
    }
  }

  disconnectWebSocket() {
    if (this.ws) {
      this.wsReconnectAttempts = this.wsMaxReconnectAttempts; // Prevent reconnection
      this.ws.close();
    }
  }

  // Batch Operations

  async batchReportErrors(errors) {
    const errorData = errors.map(error => 
      error instanceof ErrorReport ? error.toJSON() : error
    );
    
    return await this._makeRequest('POST', '/api/v1/errors/batch', {
      body: { errors: errorData }
    });
  }

  async batchGetStatus(healingIds) {
    const data = await this._makeRequest('POST', '/api/v1/healings/batch/status', {
      body: { healing_ids: healingIds }
    });
    
    const results = {};
    for (const [healingId, resultData] of Object.entries(data.results)) {
      results[healingId] = new HealingResult(resultData);
    }
    
    return results;
  }

  // Utility Methods

  async waitForHealing(healingId, timeout = 300000, pollInterval = 5000) {
    const startTime = Date.now();
    
    while (Date.now() - startTime < timeout) {
      const result = await this.getHealingStatus(healingId);
      
      if (['completed', 'failed', 'rolled_back'].includes(result.status)) {
        return result;
      }
      
      await new Promise(resolve => setTimeout(resolve, pollInterval));
    }
    
    throw new Error(`Healing ${healingId} did not complete within ${timeout}ms`);
  }

  async exportLogs(healingId, format = 'json') {
    const params = new URLSearchParams({ format });
    const response = await fetch(`${this.baseUrl}/api/v1/healings/${healingId}/logs?${params}`, {
      headers: this.headers
    });
    
    if (format === 'json') {
      return await response.json();
    } else {
      return await response.text();
    }
  }

  destroy() {
    this.disconnectWebSocket();
  }
}

// Data Classes

class ErrorReport {
  constructor({
    errorMessage,
    stackTrace,
    language,
    framework = null,
    filePath = null,
    lineNumber = null,
    severity = 'medium',
    context = null,
    timestamp = new Date()
  }) {
    this.errorMessage = errorMessage;
    this.stackTrace = stackTrace;
    this.language = language;
    this.framework = framework;
    this.filePath = filePath;
    this.lineNumber = lineNumber;
    this.severity = severity;
    this.context = context;
    this.timestamp = timestamp;
  }

  toJSON() {
    return {
      error_message: this.errorMessage,
      stack_trace: this.stackTrace,
      language: this.language,
      framework: this.framework,
      file_path: this.filePath,
      line_number: this.lineNumber,
      severity: this.severity,
      context: this.context,
      timestamp: this.timestamp.toISOString()
    };
  }
}

class HealingResult {
  constructor(data) {
    this.healingId = data.healing_id;
    this.status = data.status;
    this.errorId = data.error_id;
    this.patchesGenerated = data.patches_generated;
    this.patchesApplied = data.patches_applied;
    this.success = data.success;
    this.durationSeconds = data.duration_seconds;
    this.logs = data.logs;
    this.rollbackAvailable = data.rollback_available;
    this.metrics = data.metrics;
  }
}

class SystemHealth {
  constructor(data) {
    this.status = data.status;
    this.uptimeSeconds = data.uptime_seconds;
    this.activeHealings = data.active_healings;
    this.totalErrorsProcessed = data.total_errors_processed;
    this.successRate = data.success_rate;
    this.averageHealingTime = data.average_healing_time;
    this.components = data.components;
  }
}

class HomeostasisError extends Error {
  constructor(message, statusCode) {
    super(message);
    this.name = 'HomeostasisError';
    this.statusCode = statusCode;
  }
}

// TypeScript Support
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    HomeostasisClient,
    ErrorReport,
    HealingResult,
    SystemHealth,
    HomeostasisError
  };
}

// ES6 Export
export {
  HomeostasisClient,
  ErrorReport,
  HealingResult,
  SystemHealth,
  HomeostasisError
};

// Convenience functions

export function createClient(baseUrl, apiKey = null, options = {}) {
  return new HomeostasisClient(baseUrl, apiKey, options);
}

export async function quickReportError(baseUrl, errorMessage, stackTrace, language, apiKey = null) {
  const client = createClient(baseUrl, apiKey);
  const error = new ErrorReport({
    errorMessage,
    stackTrace,
    language,
    timestamp: new Date()
  });
  
  try {
    const result = await client.reportError(error);
    return result.error_id;
  } finally {
    client.destroy();
  }
}