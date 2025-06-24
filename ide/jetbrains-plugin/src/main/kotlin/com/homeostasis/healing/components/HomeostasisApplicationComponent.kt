package com.homeostasis.healing.components

import com.homeostasis.healing.services.ConfigurationService
import com.homeostasis.healing.services.HealingService
import com.homeostasis.healing.services.TelemetryService
import com.intellij.openapi.application.ApplicationManager
import com.intellij.openapi.components.ApplicationComponent
import com.intellij.openapi.diagnostic.Logger

/**
 * Application-level component that initializes the Homeostasis healing system
 */
class HomeostasisApplicationComponent : ApplicationComponent {
    
    companion object {
        private val LOG = Logger.getInstance(HomeostasisApplicationComponent::class.java)
        
        fun getInstance(): HomeostasisApplicationComponent {
            return ApplicationManager.getApplication().getComponent(HomeostasisApplicationComponent::class.java)
        }
    }
    
    private val configurationService: ConfigurationService by lazy {
        ApplicationManager.getApplication().getService(ConfigurationService::class.java)
    }
    
    private val telemetryService: TelemetryService by lazy {
        ApplicationManager.getApplication().getService(TelemetryService::class.java)
    }
    
    private val healingService: HealingService by lazy {
        ApplicationManager.getApplication().getService(HealingService::class.java)
    }
    
    override fun initComponent() {
        LOG.info("Initializing Homeostasis Self-Healing plugin")
        
        try {
            // Initialize configuration
            configurationService.initialize()
            
            // Initialize telemetry
            telemetryService.initialize()
            
            // Initialize healing service
            healingService.initialize()
            
            // Send activation event
            telemetryService.sendEvent("plugin.activated", mapOf(
                "ide" to "intellij",
                "version" to getPluginVersion()
            ))
            
            LOG.info("Homeostasis Self-Healing plugin initialized successfully")
            
        } catch (e: Exception) {
            LOG.error("Failed to initialize Homeostasis plugin", e)
        }
    }
    
    override fun disposeComponent() {
        LOG.info("Disposing Homeostasis Self-Healing plugin")
        
        try {
            // Send deactivation event
            telemetryService.sendEvent("plugin.deactivated", mapOf(
                "ide" to "intellij"
            ))
            
            // Dispose services
            healingService.dispose()
            telemetryService.dispose()
            
            LOG.info("Homeostasis Self-Healing plugin disposed successfully")
            
        } catch (e: Exception) {
            LOG.error("Error during plugin disposal", e)
        }
    }
    
    override fun getComponentName(): String = "HomeostasisApplicationComponent"
    
    private fun getPluginVersion(): String {
        return javaClass.`package`?.implementationVersion ?: "0.1.0"
    }
}