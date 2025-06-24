package com.homeostasis.healing.services

/**
 * Service interface for managing Homeostasis configuration
 */
interface ConfigurationService {
    
    /**
     * Initialize the configuration service
     */
    fun initialize()
    
    /**
     * Get the Homeostasis server URL
     * @return Server URL
     */
    fun getServerUrl(): String
    
    /**
     * Set the Homeostasis server URL
     * @param url Server URL
     */
    fun setServerUrl(url: String)
    
    /**
     * Get the API key for authentication
     * @return API key
     */
    fun getApiKey(): String
    
    /**
     * Set the API key for authentication
     * @param apiKey API key
     */
    fun setApiKey(apiKey: String)
    
    /**
     * Check if real-time healing is enabled
     * @return True if enabled
     */
    fun isRealTimeHealingEnabled(): Boolean
    
    /**
     * Enable or disable real-time healing
     * @param enabled True to enable, false to disable
     */
    fun setRealTimeHealingEnabled(enabled: Boolean)
    
    /**
     * Get the healing delay in milliseconds
     * @return Delay in milliseconds
     */
    fun getHealingDelay(): Int
    
    /**
     * Set the healing delay in milliseconds
     * @param delay Delay in milliseconds
     */
    fun setHealingDelay(delay: Int)
    
    /**
     * Get the list of enabled programming languages
     * @return List of enabled languages
     */
    fun getEnabledLanguages(): List<String>
    
    /**
     * Set the list of enabled programming languages
     * @param languages List of languages to enable
     */
    fun setEnabledLanguages(languages: List<String>)
    
    /**
     * Get the confidence threshold for auto-applying fixes
     * @return Confidence threshold (0.0 to 1.0)
     */
    fun getConfidenceThreshold(): Double
    
    /**
     * Set the confidence threshold for auto-applying fixes
     * @param threshold Confidence threshold (0.0 to 1.0)
     */
    fun setConfidenceThreshold(threshold: Double)
    
    /**
     * Check if telemetry is enabled
     * @return True if enabled
     */
    fun isTelemetryEnabled(): Boolean
    
    /**
     * Enable or disable telemetry
     * @param enabled True to enable, false to disable
     */
    fun setTelemetryEnabled(enabled: Boolean)
    
    /**
     * Check if inline hints are enabled
     * @return True if enabled
     */
    fun isInlineHintsEnabled(): Boolean
    
    /**
     * Enable or disable inline hints
     * @param enabled True to enable, false to disable
     */
    fun setInlineHintsEnabled(enabled: Boolean)
    
    /**
     * Check if inspections are enabled
     * @return True if enabled
     */
    fun isInspectionsEnabled(): Boolean
    
    /**
     * Enable or disable inspections
     * @param enabled True to enable, false to disable
     */
    fun setInspectionsEnabled(enabled: Boolean)
    
    /**
     * Check if notifications are enabled
     * @return True if enabled
     */
    fun isNotificationsEnabled(): Boolean
    
    /**
     * Enable or disable notifications
     * @param enabled True to enable, false to disable
     */
    fun setNotificationsEnabled(enabled: Boolean)
    
    /**
     * Reload configuration from storage
     */
    fun reloadConfiguration()
    
    /**
     * Save current configuration to storage
     */
    fun saveConfiguration()
}