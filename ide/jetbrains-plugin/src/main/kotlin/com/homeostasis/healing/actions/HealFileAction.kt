package com.homeostasis.healing.actions

import com.homeostasis.healing.services.HealingService
import com.intellij.openapi.actionSystem.AnAction
import com.intellij.openapi.actionSystem.AnActionEvent
import com.intellij.openapi.actionSystem.CommonDataKeys
import com.intellij.openapi.application.ApplicationManager
import com.intellij.openapi.diagnostic.Logger
import com.intellij.openapi.progress.ProgressIndicator
import com.intellij.openapi.progress.ProgressManager
import com.intellij.openapi.progress.Task
import com.intellij.openapi.ui.Messages
import com.intellij.psi.PsiDocumentManager
import kotlinx.coroutines.runBlocking

/**
 * Action to heal the current file
 */
class HealFileAction : AnAction() {
    
    companion object {
        private val LOG = Logger.getInstance(HealFileAction::class.java)
    }
    
    private val healingService: HealingService by lazy {
        ApplicationManager.getApplication().getService(HealingService::class.java)
    }
    
    override fun actionPerformed(e: AnActionEvent) {
        val project = e.project ?: return
        val editor = e.getData(CommonDataKeys.EDITOR) ?: return
        val psiFile = PsiDocumentManager.getInstance(project).getPsiFile(editor.document) ?: return
        
        ProgressManager.getInstance().run(object : Task.Backgroundable(project, "Healing File", true) {
            override fun run(indicator: ProgressIndicator) {
                try {
                    indicator.text = "Analyzing file for healing opportunities..."
                    indicator.isIndeterminate = false
                    indicator.fraction = 0.1
                    
                    val healingsApplied = runBlocking {
                        healingService.healFile(psiFile)
                    }
                    
                    indicator.fraction = 1.0
                    
                    ApplicationManager.getApplication().invokeLater {
                        if (healingsApplied > 0) {
                            Messages.showInfoMessage(
                                project,
                                "Applied $healingsApplied healing(s) to ${psiFile.name}",
                                "Homeostasis Healing Complete"
                            )
                        } else {
                            Messages.showInfoMessage(
                                project,
                                "No healings needed for ${psiFile.name}",
                                "Homeostasis Healing Complete"
                            )
                        }
                    }
                    
                } catch (e: Exception) {
                    LOG.error("Error during file healing", e)
                    
                    ApplicationManager.getApplication().invokeLater {
                        Messages.showErrorDialog(
                            project,
                            "Error occurred during healing: ${e.message}",
                            "Homeostasis Healing Error"
                        )
                    }
                }
            }
        })
    }
    
    override fun update(e: AnActionEvent) {
        val project = e.project
        val editor = e.getData(CommonDataKeys.EDITOR)
        val psiFile = if (project != null && editor != null) {
            PsiDocumentManager.getInstance(project).getPsiFile(editor.document)
        } else null
        
        e.presentation.isEnabledAndVisible = psiFile != null && 
            healingService.getEnabledLanguages().contains(psiFile.language.id.lowercase())
    }
}