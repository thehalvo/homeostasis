package com.homeostasis.healing.actions

import com.homeostasis.healing.services.TelemetryService
import com.homeostasis.healing.toolwindows.HealingToolWindow
import com.intellij.openapi.actionSystem.AnAction
import com.intellij.openapi.actionSystem.AnActionEvent
import com.intellij.openapi.application.ApplicationManager
import com.intellij.openapi.diagnostic.Logger
import com.intellij.openapi.project.Project
import com.intellij.openapi.ui.Messages
import com.intellij.openapi.wm.ToolWindowManager

/**
 * Action to clear healing history and statistics
 */
class ClearHistoryAction : AnAction() {
    
    companion object {
        private val LOG = Logger.getInstance(ClearHistoryAction::class.java)
    }
    
    private val telemetryService: TelemetryService by lazy {
        ApplicationManager.getApplication().getService(TelemetryService::class.java)
    }
    
    override fun actionPerformed(e: AnActionEvent) {
        val project = e.project ?: return
        
        try {
            if (confirmClearHistory(project)) {
                clearHistory(project)
            }
        } catch (ex: Exception) {
            LOG.error("Error clearing healing history", ex)
        }
    }
    
    override fun update(e: AnActionEvent) {
        val project = e.project
        e.presentation.isEnabledAndVisible = project != null && isToolWindowVisible(project)
    }
    
    private fun confirmClearHistory(project: Project): Boolean {
        val result = Messages.showYesNoDialog(
            project,
            "Are you sure you want to clear all healing history and statistics?\nThis action cannot be undone.",
            "Clear Healing History",
            "Clear",
            "Cancel",
            Messages.getQuestionIcon()
        )
        
        return result == Messages.YES
    }
    
    private fun clearHistory(project: Project) {
        // Clear telemetry data
        telemetryService.clearHistory()
        
        // Refresh the tool window
        val toolWindowManager = ToolWindowManager.getInstance(project)
        val toolWindow = toolWindowManager.getToolWindow("Homeostasis")
        
        if (toolWindow != null) {
            val contentManager = toolWindow.contentManager
            val content = contentManager.selectedContent
            
            if (content != null) {
                val component = content.component
                if (component is HealingToolWindow) {
                    component.clearHistory()
                    component.refreshStatistics()
                    LOG.info("Healing history cleared and statistics refreshed")
                } else {
                    LOG.warn("Tool window component is not a HealingToolWindow instance")
                }
            }
        }
        
        // Show confirmation
        Messages.showInfoMessage(
            project,
            "Healing history and statistics have been cleared.",
            "History Cleared"
        )
        
        LOG.info("Healing history cleared successfully")
    }
    
    private fun isToolWindowVisible(project: Project?): Boolean {
        if (project == null) return false
        
        val toolWindowManager = ToolWindowManager.getInstance(project)
        val toolWindow = toolWindowManager.getToolWindow("Homeostasis")
        
        return toolWindow?.isVisible == true
    }
}