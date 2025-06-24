package com.homeostasis.healing.services

/**
 * Service interface for telemetry and analytics
 */
interface TelemetryService {
    
    /**
     * Initialize the telemetry service
     */
    fun initialize()
    
    /**
     * Dispose the telemetry service
     */
    fun dispose()
    
    /**
     * Send an event to telemetry
     * @param event Event name
     * @param data Additional event data
     */
    fun sendEvent(event: String, data: Map<String, Any> = emptyMap())
    
    /**
     * Send an error to telemetry
     * @param error Error description
     * @param exception Exception details
     */
    fun sendError(error: String, exception: Throwable? = null)
    
    /**
     * Send performance metrics
     * @param operation Operation name
     * @param duration Duration in milliseconds
     * @param metadata Additional metadata
     */
    fun sendPerformanceMetric(operation: String, duration: Long, metadata: Map<String, Any> = emptyMap())
    
    /**
     * Check if telemetry is enabled
     * @return True if enabled
     */
    fun isEnabled(): Boolean
    
    /**
     * Clear all telemetry history and cached data
     */
    fun clearHistory()
}