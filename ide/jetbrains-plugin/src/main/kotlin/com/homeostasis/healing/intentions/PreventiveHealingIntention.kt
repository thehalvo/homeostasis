package com.homeostasis.healing.intentions

import com.homeostasis.healing.services.HealingService
import com.intellij.codeInsight.intention.IntentionAction
import com.intellij.codeInsight.intention.PriorityAction
import com.intellij.openapi.application.ApplicationManager
import com.intellij.openapi.diagnostic.Logger
import com.intellij.openapi.editor.Editor
import com.intellij.openapi.project.Project
import com.intellij.psi.PsiFile
import com.intellij.util.IncorrectOperationException
import kotlinx.coroutines.runBlocking

/**
 * Intention action for preventive healing - applying optimizations and best practices
 */
class PreventiveHealingIntention : IntentionAction, PriorityAction {
    
    companion object {
        private val LOG = Logger.getInstance(PreventiveHealingIntention::class.java)
    }
    
    private val healingService: HealingService by lazy {
        ApplicationManager.getApplication().getService(HealingService::class.java)
    }
    
    override fun getText(): String = "Apply preventive healing optimizations"
    
    override fun getFamilyName(): String = "Homeostasis Preventive Healing"
    
    override fun isAvailable(project: Project, editor: Editor?, file: PsiFile?): Boolean {
        if (editor == null || file == null) return false
        
        val language = file.language.id.lowercase()
        return healingService.getEnabledLanguages().contains(language)
    }
    
    @Throws(IncorrectOperationException::class)
    override fun invoke(project: Project, editor: Editor?, file: PsiFile?) {
        if (editor == null || file == null) return
        
        try {
            val suggestions = runBlocking {
                healingService.analyzePsiFile(file)
            }
            
            // Filter for preventive/optimization suggestions (typically INFO level)
            val preventiveSuggestions = suggestions
                .filter { it.severity == com.homeostasis.healing.services.HealingSuggestion.Severity.INFO }
                .filter { it.confidence >= 0.5 } // Lower threshold for preventive suggestions
                .sortedByDescending { it.confidence }
            
            // Apply all preventive suggestions
            for (suggestion in preventiveSuggestions) {
                runBlocking {
                    healingService.applyHealing(suggestion, editor.document, project)
                }
            }
            
            if (preventiveSuggestions.isNotEmpty()) {
                LOG.info("Applied ${preventiveSuggestions.size} preventive healing suggestions")
            }
            
        } catch (e: Exception) {
            LOG.error("Error during preventive healing intention", e)
        }
    }
    
    override fun startInWriteAction(): Boolean = true
    
    override fun getPriority(): PriorityAction.Priority = PriorityAction.Priority.LOW
}