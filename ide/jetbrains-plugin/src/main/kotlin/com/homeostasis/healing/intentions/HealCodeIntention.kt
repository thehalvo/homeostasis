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
 * Intention action to heal code at the cursor position
 */
class HealCodeIntention : IntentionAction, PriorityAction {
    
    companion object {
        private val LOG = Logger.getInstance(HealCodeIntention::class.java)
    }
    
    private val healingService: HealingService by lazy {
        ApplicationManager.getApplication().getService(HealingService::class.java)
    }
    
    override fun getText(): String = "Heal code with Homeostasis"
    
    override fun getFamilyName(): String = "Homeostasis Healing"
    
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
            
            val caretOffset = editor.caretModel.offset
            
            // Find the most relevant suggestion near the cursor
            val relevantSuggestion = suggestions
                .filter { suggestion ->
                    caretOffset >= suggestion.startOffset && caretOffset <= suggestion.endOffset
                }
                .maxByOrNull { it.confidence }
            
            if (relevantSuggestion != null) {
                runBlocking {
                    healingService.applyHealing(relevantSuggestion, editor.document, project)
                }
            } else {
                // If no specific suggestion at cursor, heal the entire file
                runBlocking {
                    healingService.healFile(file)
                }
            }
            
        } catch (e: Exception) {
            LOG.error("Error during code healing intention", e)
        }
    }
    
    override fun startInWriteAction(): Boolean = true
    
    override fun getPriority(): PriorityAction.Priority = PriorityAction.Priority.NORMAL
}