package com.homeostasis.healing.toolwindows

import com.intellij.openapi.project.Project
import com.intellij.openapi.wm.ToolWindow
import com.intellij.openapi.wm.ToolWindowFactory
import com.intellij.ui.content.ContentFactory

/**
 * Factory for creating the Homeostasis healing tool window
 */
class HealingToolWindowFactory : ToolWindowFactory {
    
    override fun createToolWindowContent(project: Project, toolWindow: ToolWindow) {
        val healingToolWindow = HealingToolWindow(project)
        val content = ContentFactory.SERVICE.getInstance()
            .createContent(healingToolWindow.getContent(), "", false)
        
        toolWindow.contentManager.addContent(content)
    }
    
    override fun shouldBeAvailable(project: Project): Boolean = true
}