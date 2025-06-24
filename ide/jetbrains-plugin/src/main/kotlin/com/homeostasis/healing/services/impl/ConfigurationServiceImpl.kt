package com.homeostasis.healing.services.impl

import com.homeostasis.healing.services.ConfigurationService
import com.intellij.ide.util.PropertiesComponent
import com.intellij.openapi.diagnostic.Logger

/**
 * Implementation of the ConfigurationService interface using IntelliJ's PropertiesComponent
 */
class ConfigurationServiceImpl : ConfigurationService {
    
    companion object {
        private val LOG = Logger.getInstance(ConfigurationServiceImpl::class.java)
        
        // Configuration keys
        private const val SERVER_URL_KEY = "homeostasis.serverUrl"
        private const val API_KEY_KEY = "homeostasis.apiKey"
        private const val REAL_TIME_HEALING_KEY = "homeostasis.realTimeHealing"
        private const val HEALING_DELAY_KEY = "homeostasis.healingDelay"
        private const val ENABLED_LANGUAGES_KEY = "homeostasis.enabledLanguages"
        private const val CONFIDENCE_THRESHOLD_KEY = "homeostasis.confidenceThreshold"
        private const val TELEMETRY_ENABLED_KEY = "homeostasis.telemetryEnabled"
        private const val INLINE_HINTS_KEY = "homeostasis.inlineHints"
        private const val INSPECTIONS_ENABLED_KEY = "homeostasis.inspectionsEnabled"
        private const val NOTIFICATIONS_ENABLED_KEY = "homeostasis.notificationsEnabled"
        
        // Default values
        private const val DEFAULT_SERVER_URL = "http://localhost:8080"
        private const val DEFAULT_API_KEY = ""
        private const val DEFAULT_REAL_TIME_HEALING = true
        private const val DEFAULT_HEALING_DELAY = 2000
        private val DEFAULT_ENABLED_LANGUAGES = listOf(
            "python", "java", "javascript", "typescript", "go", "rust", 
            "csharp", "php", "ruby", "scala", "elixir", "clojure", 
            "swift", "kotlin", "dart"
        )
        private const val DEFAULT_CONFIDENCE_THRESHOLD = 0.7
        private const val DEFAULT_TELEMETRY_ENABLED = true
        private const val DEFAULT_INLINE_HINTS = true
        private const val DEFAULT_INSPECTIONS_ENABLED = true
        private const val DEFAULT_NOTIFICATIONS_ENABLED = true
    }
    
    private val properties = PropertiesComponent.getInstance()
    
    override fun initialize() {
        LOG.info("Initializing ConfigurationService")
        reloadConfiguration()
    }
    
    override fun getServerUrl(): String {
        return properties.getValue(SERVER_URL_KEY, DEFAULT_SERVER_URL)
    }
    
    override fun setServerUrl(url: String) {
        properties.setValue(SERVER_URL_KEY, url)
        LOG.info("Server URL updated to: $url")
    }
    
    override fun getApiKey(): String {
        return properties.getValue(API_KEY_KEY, DEFAULT_API_KEY)
    }
    
    override fun setApiKey(apiKey: String) {
        properties.setValue(API_KEY_KEY, apiKey)
        LOG.info("API key updated")
    }
    
    override fun isRealTimeHealingEnabled(): Boolean {
        return properties.getBoolean(REAL_TIME_HEALING_KEY, DEFAULT_REAL_TIME_HEALING)
    }
    
    override fun setRealTimeHealingEnabled(enabled: Boolean) {
        properties.setValue(REAL_TIME_HEALING_KEY, enabled)
        LOG.info("Real-time healing ${if (enabled) "enabled" else "disabled"}")
    }
    
    override fun getHealingDelay(): Int {
        return properties.getInt(HEALING_DELAY_KEY, DEFAULT_HEALING_DELAY)
    }
    
    override fun setHealingDelay(delay: Int) {
        properties.setValue(HEALING_DELAY_KEY, delay, DEFAULT_HEALING_DELAY)
        LOG.info("Healing delay set to: ${delay}ms")
    }
    
    override fun getEnabledLanguages(): List<String> {
        val languagesString = properties.getValue(ENABLED_LANGUAGES_KEY, "")
        return if (languagesString.isNotEmpty()) {
            languagesString.split(",").map { it.trim() }
        } else {
            DEFAULT_ENABLED_LANGUAGES
        }
    }
    
    override fun setEnabledLanguages(languages: List<String>) {
        val languagesString = languages.joinToString(",")
        properties.setValue(ENABLED_LANGUAGES_KEY, languagesString)
        LOG.info("Enabled languages updated: $languages")
    }
    
    override fun getConfidenceThreshold(): Double {
        val value = properties.getValue(CONFIDENCE_THRESHOLD_KEY, DEFAULT_CONFIDENCE_THRESHOLD.toString())
        return try {
            value.toDouble().coerceIn(0.0, 1.0)
        } catch (e: NumberFormatException) {
            LOG.warn("Invalid confidence threshold value: $value, using default")
            DEFAULT_CONFIDENCE_THRESHOLD
        }
    }
    
    override fun setConfidenceThreshold(threshold: Double) {
        val clampedThreshold = threshold.coerceIn(0.0, 1.0)
        properties.setValue(CONFIDENCE_THRESHOLD_KEY, clampedThreshold.toString())
        LOG.info("Confidence threshold set to: $clampedThreshold")
    }
    
    override fun isTelemetryEnabled(): Boolean {
        return properties.getBoolean(TELEMETRY_ENABLED_KEY, DEFAULT_TELEMETRY_ENABLED)
    }
    
    override fun setTelemetryEnabled(enabled: Boolean) {
        properties.setValue(TELEMETRY_ENABLED_KEY, enabled)
        LOG.info("Telemetry ${if (enabled) "enabled" else "disabled"}")
    }
    
    override fun isInlineHintsEnabled(): Boolean {
        return properties.getBoolean(INLINE_HINTS_KEY, DEFAULT_INLINE_HINTS)
    }
    
    override fun setInlineHintsEnabled(enabled: Boolean) {
        properties.setValue(INLINE_HINTS_KEY, enabled)
        LOG.info("Inline hints ${if (enabled) "enabled" else "disabled"}")
    }
    
    override fun isInspectionsEnabled(): Boolean {
        return properties.getBoolean(INSPECTIONS_ENABLED_KEY, DEFAULT_INSPECTIONS_ENABLED)
    }
    
    override fun setInspectionsEnabled(enabled: Boolean) {
        properties.setValue(INSPECTIONS_ENABLED_KEY, enabled)
        LOG.info("Inspections ${if (enabled) "enabled" else "disabled"}")
    }
    
    override fun isNotificationsEnabled(): Boolean {
        return properties.getBoolean(NOTIFICATIONS_ENABLED_KEY, DEFAULT_NOTIFICATIONS_ENABLED)
    }
    
    override fun setNotificationsEnabled(enabled: Boolean) {
        properties.setValue(NOTIFICATIONS_ENABLED_KEY, enabled)
        LOG.info("Notifications ${if (enabled) "enabled" else "disabled"}")
    }
    
    override fun reloadConfiguration() {
        LOG.info("Configuration reloaded")
    }
    
    override fun saveConfiguration() {
        LOG.info("Configuration saved")
    }
}