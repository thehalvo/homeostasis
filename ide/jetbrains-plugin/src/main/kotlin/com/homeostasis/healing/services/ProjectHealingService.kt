package com.homeostasis.healing.services

import com.intellij.psi.PsiFile

/**
 * Project-specific healing service interface
 */
interface ProjectHealingService {
    
    /**
     * Initialize the project healing service
     */
    fun initialize()
    
    /**
     * Dispose the project healing service
     */
    fun dispose()
    
    /**
     * Get healing statistics for the project
     * @return Healing statistics
     */
    fun getHealingStatistics(): HealingStatistics
    
    /**
     * Get healing history for the project
     * @return List of healing history entries
     */
    fun getHealingHistory(): List<HealingHistoryEntry>
    
    /**
     * Clear healing history for the project
     */
    fun clearHealingHistory()
    
    /**
     * Check if a file should be included in healing
     * @param psiFile The file to check
     * @return True if the file should be healed
     */
    fun shouldHealFile(psiFile: PsiFile): Boolean
    
    /**
     * Get project-specific healing configuration
     * @return Project configuration
     */
    fun getProjectConfiguration(): ProjectHealingConfiguration
}

/**
 * Data class for healing statistics
 */
data class HealingStatistics(
    val totalHealings: Int = 0,
    val sessionHealings: Int = 0,
    val successRate: Double = 100.0,
    val averageConfidence: Double = 0.0,
    val languageBreakdown: Map<String, Int> = emptyMap(),
    val ruleBreakdown: Map<String, Int> = emptyMap()
)

/**
 * Data class for healing history entries
 */
data class HealingHistoryEntry(
    val timestamp: Long,
    val filePath: String,
    val language: String,
    val ruleId: String,
    val confidence: Double,
    val applied: Boolean,
    val description: String
)

/**
 * Data class for project-specific healing configuration
 */
data class ProjectHealingConfiguration(
    val excludedPaths: List<String> = emptyList(),
    val customRules: List<String> = emptyList(),
    val languageSpecificSettings: Map<String, Map<String, Any>> = emptyMap()
)