package com.homeostasis.healing.actions

import com.homeostasis.healing.toolwindows.HealingToolWindow
import com.intellij.openapi.actionSystem.AnAction
import com.intellij.openapi.actionSystem.AnActionEvent
import com.intellij.openapi.diagnostic.Logger
import com.intellij.openapi.project.Project
import com.intellij.openapi.wm.ToolWindowManager

/**
 * Action to refresh healing statistics in the tool window
 */
class RefreshStatsAction : AnAction() {
    
    companion object {
        private val LOG = Logger.getInstance(RefreshStatsAction::class.java)
    }
    
    override fun actionPerformed(e: AnActionEvent) {
        val project = e.project ?: return
        
        try {
            refreshStatistics(project)
        } catch (ex: Exception) {
            LOG.error("Error refreshing healing statistics", ex)
        }
    }
    
    override fun update(e: AnActionEvent) {
        val project = e.project
        e.presentation.isEnabledAndVisible = project != null && isToolWindowVisible(project)
    }
    
    private fun refreshStatistics(project: Project) {
        val toolWindowManager = ToolWindowManager.getInstance(project)
        val toolWindow = toolWindowManager.getToolWindow("Homeostasis")
        
        if (toolWindow != null) {
            // Get the content component and refresh if it's our tool window
            val contentManager = toolWindow.contentManager
            val content = contentManager.selectedContent
            
            if (content != null) {
                val component = content.component
                if (component is HealingToolWindow) {
                    component.refreshStatistics()
                    LOG.info("Healing statistics refreshed")
                } else {
                    LOG.warn("Tool window component is not a HealingToolWindow instance")
                }
            }
        } else {
            LOG.warn("Homeostasis tool window not found")
        }
    }
    
    private fun isToolWindowVisible(project: Project?): Boolean {
        if (project == null) return false
        
        val toolWindowManager = ToolWindowManager.getInstance(project)
        val toolWindow = toolWindowManager.getToolWindow("Homeostasis")
        
        return toolWindow?.isVisible == true
    }
}