package com.homeostasis.healing.actions

import com.homeostasis.healing.services.HealingService
import com.intellij.openapi.actionSystem.AnAction
import com.intellij.openapi.actionSystem.AnActionEvent
import com.intellij.openapi.application.ApplicationManager
import com.intellij.openapi.ui.Messages

/**
 * Action to toggle real-time healing on/off
 */
class ToggleRealTimeHealingAction : AnAction() {
    
    private val healingService: HealingService by lazy {
        ApplicationManager.getApplication().getService(HealingService::class.java)
    }
    
    override fun actionPerformed(e: AnActionEvent) {
        val project = e.project ?: return
        
        val currentlyEnabled = healingService.isRealTimeHealingEnabled()
        val newState = !currentlyEnabled
        
        healingService.setRealTimeHealingEnabled(newState)
        
        val message = if (newState) {
            "Real-time healing has been enabled"
        } else {
            "Real-time healing has been disabled"
        }
        
        Messages.showInfoMessage(project, message, "Homeostasis Real-time Healing")
    }
    
    override fun update(e: AnActionEvent) {
        val isEnabled = healingService.isRealTimeHealingEnabled()
        e.presentation.text = if (isEnabled) {
            "Disable Real-time Healing"
        } else {
            "Enable Real-time Healing"
        }
        
        e.presentation.description = if (isEnabled) {
            "Disable automatic healing as you type"
        } else {
            "Enable automatic healing as you type"
        }
    }
}