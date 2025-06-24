package com.homeostasis.healing.actions

import com.intellij.openapi.actionSystem.AnAction
import com.intellij.openapi.actionSystem.AnActionEvent
import com.intellij.openapi.project.Project
import com.intellij.openapi.wm.ToolWindowManager
import com.intellij.openapi.diagnostic.Logger

/**
 * Action to show the Homeostasis healing dashboard
 */
class ShowDashboardAction : AnAction() {
    
    companion object {
        private val LOG = Logger.getInstance(ShowDashboardAction::class.java)
    }
    
    override fun actionPerformed(e: AnActionEvent) {
        val project = e.project ?: return
        
        try {
            showDashboard(project)
        } catch (ex: Exception) {
            LOG.error("Error showing Homeostasis dashboard", ex)
        }
    }
    
    override fun update(e: AnActionEvent) {
        val project = e.project
        e.presentation.isEnabledAndVisible = project != null
    }
    
    private fun showDashboard(project: Project) {
        val toolWindowManager = ToolWindowManager.getInstance(project)
        val toolWindow = toolWindowManager.getToolWindow("Homeostasis")
        
        if (toolWindow != null) {
            toolWindow.show()
            toolWindow.activate(null)
            LOG.info("Homeostasis dashboard opened")
        } else {
            LOG.warn("Homeostasis tool window not found")
        }
    }
}