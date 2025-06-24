package com.homeostasis.healing.annotators

import com.homeostasis.healing.services.HealingService
import com.homeostasis.healing.services.HealingSuggestion
import com.intellij.lang.annotation.AnnotationHolder
import com.intellij.lang.annotation.ExternalAnnotator
import com.intellij.lang.annotation.HighlightSeverity
import com.intellij.openapi.application.ApplicationManager
import com.intellij.openapi.diagnostic.Logger
import com.intellij.openapi.editor.Editor
import com.intellij.psi.PsiFile
import kotlinx.coroutines.runBlocking

/**
 * External annotator for JavaScript code healing
 */
class JavaScriptHealingAnnotator : ExternalAnnotator<PsiFile, List<HealingSuggestion>>() {
    
    companion object {
        private val LOG = Logger.getInstance(JavaScriptHealingAnnotator::class.java)
    }
    
    private val healingService: HealingService by lazy {
        ApplicationManager.getApplication().getService(HealingService::class.java)
    }
    
    override fun collectInformation(file: PsiFile, editor: Editor, hasErrors: Boolean): PsiFile? {
        // Only analyze JavaScript files
        if (!file.language.id.equals("JavaScript", ignoreCase = true)) {
            return null
        }
        
        // Skip if real-time healing is disabled
        if (!healingService.isRealTimeHealingEnabled()) {
            return null
        }
        
        return file
    }
    
    override fun doAnnotate(collectedInfo: PsiFile?): List<HealingSuggestion>? {
        if (collectedInfo == null) return null
        
        return try {
            runBlocking {
                healingService.analyzePsiFile(collectedInfo)
            }
        } catch (e: Exception) {
            LOG.error("Error analyzing JavaScript file: ${collectedInfo.name}", e)
            null
        }
    }
    
    override fun apply(
        file: PsiFile,
        annotationResult: List<HealingSuggestion>?,
        holder: AnnotationHolder
    ) {
        if (annotationResult == null) return
        
        for (suggestion in annotationResult) {
            try {
                val severity = when (suggestion.severity) {
                    HealingSuggestion.Severity.ERROR -> HighlightSeverity.ERROR
                    HealingSuggestion.Severity.WARNING -> HighlightSeverity.WARNING
                    HealingSuggestion.Severity.INFO -> HighlightSeverity.INFORMATION
                }
                
                val range = file.textRange.let { fileRange ->
                    if (suggestion.startOffset >= 0 && 
                        suggestion.endOffset <= fileRange.endOffset &&
                        suggestion.startOffset <= suggestion.endOffset) {
                        com.intellij.openapi.util.TextRange(suggestion.startOffset, suggestion.endOffset)
                    } else {
                        null
                    }
                }
                
                if (range != null) {
                    holder.newAnnotation(severity, suggestion.description)
                        .range(range)
                        .tooltip("${suggestion.description}\nRule: ${suggestion.ruleId}\nConfidence: ${(suggestion.confidence * 100).toInt()}%")
                        .create()
                }
                
            } catch (e: Exception) {
                LOG.error("Error creating annotation for JavaScript suggestion: ${suggestion.id}", e)
            }
        }
    }
}