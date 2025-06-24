package com.homeostasis.healing.inspections

import com.homeostasis.healing.services.HealingService
import com.homeostasis.healing.services.HealingSuggestion
import com.intellij.codeInspection.*
import com.intellij.openapi.application.ApplicationManager
import com.intellij.openapi.diagnostic.Logger
import com.intellij.openapi.project.Project
import com.intellij.psi.PsiElementVisitor
import com.intellij.psi.PsiFile
import kotlinx.coroutines.runBlocking

/**
 * Python-specific inspection that integrates with Homeostasis healing
 */
class PythonHealingInspection : LocalInspectionTool() {
    
    companion object {
        private val LOG = Logger.getInstance(PythonHealingInspection::class.java)
    }
    
    private val healingService: HealingService by lazy {
        ApplicationManager.getApplication().getService(HealingService::class.java)
    }
    
    override fun buildVisitor(holder: ProblemsHolder, isOnTheFly: Boolean): PsiElementVisitor {
        return object : PsiElementVisitor() {
            override fun visitFile(file: PsiFile) {
                if (file.language.id.lowercase() != "python") {
                    return
                }
                
                try {
                    // Get healing suggestions for the file
                    val suggestions = runBlocking {
                        healingService.analyzePsiFile(file)
                    }
                    
                    // Convert healing suggestions to IntelliJ problems
                    suggestions.forEach { suggestion ->
                        registerSuggestion(holder, file, suggestion)
                    }
                    
                } catch (e: Exception) {
                    LOG.error("Error during Python inspection", e)
                }
            }
        }
    }
    
    private fun registerSuggestion(holder: ProblemsHolder, file: PsiFile, suggestion: HealingSuggestion) {
        val startElement = file.findElementAt(suggestion.startOffset)
        val endElement = file.findElementAt(suggestion.endOffset - 1)
        
        if (startElement != null && endElement != null) {
            val highlightType = when (suggestion.severity) {
                HealingSuggestion.Severity.ERROR -> ProblemHighlightType.ERROR
                HealingSuggestion.Severity.WARNING -> ProblemHighlightType.WARNING
                HealingSuggestion.Severity.INFO -> ProblemHighlightType.INFORMATION
            }
            
            val fixes = mutableListOf<LocalQuickFix>()
            
            // Add quick fix if replacement is available
            if (suggestion.replacement != null) {
                fixes.add(HomeostasisQuickFix(suggestion))
            }
            
            holder.registerProblem(
                startElement,
                suggestion.description,
                highlightType,
                *fixes.toTypedArray()
            )
        }
    }
    
    /**
     * Quick fix implementation for Homeostasis suggestions
     */
    private inner class HomeostasisQuickFix(private val suggestion: HealingSuggestion) : LocalQuickFix {
        
        override fun getName(): String = "Apply Homeostasis Fix: ${suggestion.description}"
        
        override fun getFamilyName(): String = "Homeostasis Healing"
        
        override fun applyFix(project: Project, descriptor: ProblemDescriptor) {
            val document = descriptor.psiElement.containingFile.viewProvider.document
            if (document != null) {
                runBlocking {
                    healingService.applyHealing(suggestion, document, project)
                }
            }
        }
    }
}