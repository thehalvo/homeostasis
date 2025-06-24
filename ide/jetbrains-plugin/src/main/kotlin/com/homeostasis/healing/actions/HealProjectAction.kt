package com.homeostasis.healing.actions

import com.homeostasis.healing.services.HealingService
import com.intellij.openapi.actionSystem.AnAction
import com.intellij.openapi.actionSystem.AnActionEvent
import com.intellij.openapi.application.ApplicationManager
import com.intellij.openapi.diagnostic.Logger
import com.intellij.openapi.progress.ProgressIndicator
import com.intellij.openapi.progress.ProgressManager
import com.intellij.openapi.progress.Task
import com.intellij.openapi.ui.Messages
import kotlinx.coroutines.runBlocking

/**
 * Action to heal the entire project
 */
class HealProjectAction : AnAction() {
    
    companion object {
        private val LOG = Logger.getInstance(HealProjectAction::class.java)
    }
    
    private val healingService: HealingService by lazy {
        ApplicationManager.getApplication().getService(HealingService::class.java)
    }
    
    override fun actionPerformed(e: AnActionEvent) {
        val project = e.project ?: return
        
        // Show confirmation dialog for project-wide healing
        val result = Messages.showYesNoDialog(
            project,
            "This will analyze and heal the entire project. This may take some time. Continue?",
            "Heal Entire Project",
            Messages.getQuestionIcon()
        )
        
        if (result != Messages.YES) {
            return
        }
        
        ProgressManager.getInstance().run(object : Task.Backgroundable(project, "Healing Project", true) {
            override fun run(indicator: ProgressIndicator) {
                try {
                    indicator.text = "Analyzing project for healing opportunities..."
                    indicator.isIndeterminate = false
                    indicator.fraction = 0.1
                    
                    val healingsApplied = runBlocking {
                        healingService.healProject(project)
                    }
                    
                    indicator.fraction = 1.0
                    
                    ApplicationManager.getApplication().invokeLater {
                        if (healingsApplied > 0) {
                            Messages.showInfoMessage(
                                project,
                                "Applied $healingsApplied healing(s) across the project",
                                "Homeostasis Project Healing Complete"
                            )
                        } else {
                            Messages.showInfoMessage(
                                project,
                                "No healings needed for this project",
                                "Homeostasis Project Healing Complete"
                            )
                        }
                    }
                    
                } catch (e: Exception) {
                    LOG.error("Error during project healing", e)
                    
                    ApplicationManager.getApplication().invokeLater {
                        Messages.showErrorDialog(
                            project,
                            "Error occurred during healing: ${e.message}",
                            "Homeostasis Project Healing Error"
                        )
                    }
                }
            }
        })
    }
    
    override fun update(e: AnActionEvent) {
        e.presentation.isEnabledAndVisible = e.project != null
    }
}