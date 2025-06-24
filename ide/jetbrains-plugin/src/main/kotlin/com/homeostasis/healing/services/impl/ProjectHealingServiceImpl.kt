package com.homeostasis.healing.services.impl

import com.homeostasis.healing.services.*
import com.intellij.openapi.diagnostic.Logger
import com.intellij.psi.PsiFile
import java.util.concurrent.ConcurrentHashMap

/**
 * Implementation of ProjectHealingService
 */
class ProjectHealingServiceImpl : ProjectHealingService {
    
    companion object {
        private val LOG = Logger.getInstance(ProjectHealingServiceImpl::class.java)
    }
    
    private val healingHistory = mutableListOf<HealingHistoryEntry>()
    private val statistics = HealingStatistics()
    
    override fun initialize() {
        LOG.info("Initializing ProjectHealingService")
    }
    
    override fun dispose() {
        LOG.info("Disposing ProjectHealingService")
        healingHistory.clear()
    }
    
    override fun getHealingStatistics(): HealingStatistics {
        return statistics.copy(
            totalHealings = healingHistory.count { it.applied },
            sessionHealings = healingHistory.count { it.applied && isFromCurrentSession(it) },
            successRate = calculateSuccessRate(),
            averageConfidence = calculateAverageConfidence(),
            languageBreakdown = getLanguageBreakdown(),
            ruleBreakdown = getRuleBreakdown()
        )
    }
    
    override fun getHealingHistory(): List<HealingHistoryEntry> {
        return healingHistory.toList().sortedByDescending { it.timestamp }
    }
    
    override fun clearHealingHistory() {
        healingHistory.clear()
        LOG.info("Healing history cleared")
    }
    
    override fun shouldHealFile(psiFile: PsiFile): Boolean {
        val filePath = psiFile.virtualFile?.path ?: return false
        val config = getProjectConfiguration()
        
        // Check if file is in excluded paths
        return !config.excludedPaths.any { excludedPath ->
            filePath.contains(excludedPath)
        }
    }
    
    override fun getProjectConfiguration(): ProjectHealingConfiguration {
        // This would typically be loaded from project settings
        return ProjectHealingConfiguration(
            excludedPaths = listOf("node_modules", ".git", "target", "build"),
            customRules = emptyList(),
            languageSpecificSettings = emptyMap()
        )
    }
    
    private fun isFromCurrentSession(entry: HealingHistoryEntry): Boolean {
        // Consider entries from the last hour as current session
        val oneHourAgo = System.currentTimeMillis() - (60 * 60 * 1000)
        return entry.timestamp > oneHourAgo
    }
    
    private fun calculateSuccessRate(): Double {
        if (healingHistory.isEmpty()) return 100.0
        
        val applied = healingHistory.count { it.applied }
        return (applied.toDouble() / healingHistory.size) * 100.0
    }
    
    private fun calculateAverageConfidence(): Double {
        if (healingHistory.isEmpty()) return 0.0
        
        val totalConfidence = healingHistory.sumOf { it.confidence }
        return totalConfidence / healingHistory.size
    }
    
    private fun getLanguageBreakdown(): Map<String, Int> {
        return healingHistory
            .filter { it.applied }
            .groupBy { it.language }
            .mapValues { it.value.size }
    }
    
    private fun getRuleBreakdown(): Map<String, Int> {
        return healingHistory
            .filter { it.applied }
            .groupBy { it.ruleId }
            .mapValues { it.value.size }
    }
}