package com.homeostasis.healing.services

import com.intellij.openapi.editor.Document
import com.intellij.openapi.project.Project
import com.intellij.psi.PsiFile

/**
 * Main service interface for code healing operations
 */
interface HealingService {
    
    /**
     * Initialize the healing service
     */
    fun initialize()
    
    /**
     * Dispose the healing service and clean up resources
     */
    fun dispose()
    
    /**
     * Analyze a document for potential healing opportunities
     * @param document The document to analyze
     * @param project The project context
     * @return List of healing suggestions
     */
    suspend fun analyzeDocument(document: Document, project: Project): List<HealingSuggestion>
    
    /**
     * Analyze a PSI file for potential healing opportunities
     * @param psiFile The PSI file to analyze
     * @return List of healing suggestions
     */
    suspend fun analyzePsiFile(psiFile: PsiFile): List<HealingSuggestion>
    
    /**
     * Apply a healing suggestion to a document
     * @param suggestion The healing suggestion to apply
     * @param document The document to modify
     * @param project The project context
     * @return True if the healing was successful
     */
    suspend fun applyHealing(suggestion: HealingSuggestion, document: Document, project: Project): Boolean
    
    /**
     * Heal an entire file
     * @param psiFile The PSI file to heal
     * @return Number of healings applied
     */
    suspend fun healFile(psiFile: PsiFile): Int
    
    /**
     * Heal an entire project
     * @param project The project to heal
     * @return Number of healings applied
     */
    suspend fun healProject(project: Project): Int
    
    /**
     * Check if real-time healing is enabled
     * @return True if real-time healing is enabled
     */
    fun isRealTimeHealingEnabled(): Boolean
    
    /**
     * Enable or disable real-time healing
     * @param enabled True to enable, false to disable
     */
    fun setRealTimeHealingEnabled(enabled: Boolean)
    
    /**
     * Get the confidence threshold for auto-applying fixes
     * @return Confidence threshold (0.0 to 1.0)
     */
    fun getConfidenceThreshold(): Double
    
    /**
     * Set the confidence threshold for auto-applying fixes
     * @param threshold Confidence threshold (0.0 to 1.0)
     */
    fun setConfidenceThreshold(threshold: Double)
}

/**
 * Data class representing a healing suggestion
 */
data class HealingSuggestion(
    val id: String,
    val description: String,
    val severity: Severity,
    val confidence: Double,
    val startOffset: Int,
    val endOffset: Int,
    val replacement: String?,
    val additionalChanges: List<AdditionalChange> = emptyList(),
    val ruleId: String,
    val language: String
) {
    enum class Severity {
        ERROR, WARNING, INFO
    }
}

/**
 * Data class representing additional changes that may be needed
 */
data class AdditionalChange(
    val filePath: String,
    val startOffset: Int,
    val endOffset: Int,
    val replacement: String
)