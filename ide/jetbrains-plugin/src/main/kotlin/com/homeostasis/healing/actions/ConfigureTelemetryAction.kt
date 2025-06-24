package com.homeostasis.healing.actions

import com.homeostasis.healing.services.ConfigurationService
import com.intellij.openapi.actionSystem.AnAction
import com.intellij.openapi.actionSystem.AnActionEvent
import com.intellij.openapi.application.ApplicationManager
import com.intellij.openapi.diagnostic.Logger
import com.intellij.openapi.options.ShowSettingsUtil
import com.intellij.openapi.project.Project

/**
 * Action to configure telemetry settings
 */
class ConfigureTelemetryAction : AnAction() {
    
    companion object {
        private val LOG = Logger.getInstance(ConfigureTelemetryAction::class.java)
    }
    
    private val configurationService: ConfigurationService by lazy {
        ApplicationManager.getApplication().getService(ConfigurationService::class.java)
    }
    
    override fun actionPerformed(e: AnActionEvent) {
        val project = e.project ?: return
        
        try {
            openTelemetrySettings(project)
        } catch (ex: Exception) {
            LOG.error("Error opening telemetry configuration", ex)
        }
    }
    
    override fun update(e: AnActionEvent) {
        val project = e.project
        e.presentation.isEnabledAndVisible = project != null
        
        // Update text based on current telemetry state
        val isTelemetryEnabled = configurationService.isTelemetryEnabled()
        e.presentation.text = if (isTelemetryEnabled) {
            "Disable Telemetry"
        } else {
            "Enable Telemetry"
        }
    }
    
    private fun openTelemetrySettings(project: Project) {
        // Open the Homeostasis settings page
        ShowSettingsUtil.getInstance().showSettingsDialog(project, "Homeostasis")
        LOG.info("Opened Homeostasis settings for telemetry configuration")
    }
}