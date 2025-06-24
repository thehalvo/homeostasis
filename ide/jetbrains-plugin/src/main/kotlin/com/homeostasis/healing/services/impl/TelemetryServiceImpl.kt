package com.homeostasis.healing.services.impl

import com.homeostasis.healing.services.ConfigurationService
import com.homeostasis.healing.services.TelemetryService
import com.intellij.openapi.application.ApplicationManager
import com.intellij.openapi.diagnostic.Logger
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.ConcurrentLinkedQueue

/**
 * Implementation of the TelemetryService interface
 */
class TelemetryServiceImpl : TelemetryService {
    
    companion object {
        private val LOG = Logger.getInstance(TelemetryServiceImpl::class.java)
    }
    
    private val configurationService: ConfigurationService by lazy {
        ApplicationManager.getApplication().getService(ConfigurationService::class.java)
    }
    
    // In-memory storage for telemetry data
    private val events = ConcurrentLinkedQueue<TelemetryEvent>()
    private val errors = ConcurrentLinkedQueue<TelemetryError>()
    private val performanceMetrics = ConcurrentLinkedQueue<PerformanceMetric>()
    private val sessionStats = ConcurrentHashMap<String, Any>()
    
    private var isInitialized = false
    
    override fun initialize() {
        LOG.info("Initializing TelemetryService")
        isInitialized = true
        
        // Initialize session statistics
        sessionStats["session_start"] = LocalDateTime.now()
        sessionStats["total_events"] = 0
        sessionStats["total_errors"] = 0
        sessionStats["total_healings"] = 0
    }
    
    override fun dispose() {
        LOG.info("Disposing TelemetryService")
        isInitialized = false
        
        // Clear all data
        clearHistory()
    }
    
    override fun sendEvent(event: String, data: Map<String, Any>) {
        if (!isEnabled() || !isInitialized) return
        
        try {
            val telemetryEvent = TelemetryEvent(
                name = event,
                timestamp = LocalDateTime.now(),
                data = data.toMap()
            )
            
            events.offer(telemetryEvent)
            updateSessionStats("total_events", 1)
            
            LOG.debug("Telemetry event sent: $event")
            
            // Special handling for healing events
            if (event.contains("healing") || event.contains("healed")) {
                updateSessionStats("total_healings", 1)
            }
            
        } catch (e: Exception) {
            LOG.error("Error sending telemetry event: $event", e)
        }
    }
    
    override fun sendError(error: String, exception: Throwable?) {
        if (!isEnabled() || !isInitialized) return
        
        try {
            val telemetryError = TelemetryError(
                description = error,
                timestamp = LocalDateTime.now(),
                stackTrace = exception?.stackTraceToString(),
                exceptionType = exception?.javaClass?.simpleName
            )
            
            errors.offer(telemetryError)
            updateSessionStats("total_errors", 1)
            
            LOG.debug("Telemetry error sent: $error")
            
        } catch (e: Exception) {
            LOG.error("Error sending telemetry error: $error", e)
        }
    }
    
    override fun sendPerformanceMetric(operation: String, duration: Long, metadata: Map<String, Any>) {
        if (!isEnabled() || !isInitialized) return
        
        try {
            val metric = PerformanceMetric(
                operation = operation,
                duration = duration,
                timestamp = LocalDateTime.now(),
                metadata = metadata.toMap()
            )
            
            performanceMetrics.offer(metric)
            
            LOG.debug("Performance metric sent: $operation took ${duration}ms")
            
        } catch (e: Exception) {
            LOG.error("Error sending performance metric: $operation", e)
        }
    }
    
    override fun isEnabled(): Boolean {
        return configurationService.isTelemetryEnabled()
    }
    
    override fun clearHistory() {
        LOG.info("Clearing telemetry history")
        
        events.clear()
        errors.clear()
        performanceMetrics.clear()
        
        // Reset session stats but keep session start time
        val sessionStart = sessionStats["session_start"]
        sessionStats.clear()
        sessionStats["session_start"] = sessionStart ?: LocalDateTime.now()
        sessionStats["total_events"] = 0
        sessionStats["total_errors"] = 0
        sessionStats["total_healings"] = 0
    }
    
    /**
     * Get all telemetry events
     */
    fun getEvents(): List<TelemetryEvent> {
        return events.toList()
    }
    
    /**
     * Get all telemetry errors
     */
    fun getErrors(): List<TelemetryError> {
        return errors.toList()
    }
    
    /**
     * Get all performance metrics
     */
    fun getPerformanceMetrics(): List<PerformanceMetric> {
        return performanceMetrics.toList()
    }
    
    /**
     * Get session statistics
     */
    fun getSessionStats(): Map<String, Any> {
        return sessionStats.toMap()
    }
    
    private fun updateSessionStats(key: String, increment: Int) {
        sessionStats.compute(key) { _, value ->
            (value as? Int ?: 0) + increment
        }
    }
}

/**
 * Data class for telemetry events
 */
data class TelemetryEvent(
    val name: String,
    val timestamp: LocalDateTime,
    val data: Map<String, Any>
)

/**
 * Data class for telemetry errors
 */
data class TelemetryError(
    val description: String,
    val timestamp: LocalDateTime,
    val stackTrace: String?,
    val exceptionType: String?
)

/**
 * Data class for performance metrics
 */
data class PerformanceMetric(
    val operation: String,
    val duration: Long,
    val timestamp: LocalDateTime,
    val metadata: Map<String, Any>
)