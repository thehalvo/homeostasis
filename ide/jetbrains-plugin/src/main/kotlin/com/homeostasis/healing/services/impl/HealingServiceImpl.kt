package com.homeostasis.healing.services.impl

import com.homeostasis.healing.services.*
import com.homeostasis.healing.utils.HomeostasisApiClient
import com.intellij.openapi.application.ApplicationManager
import com.intellij.openapi.command.WriteCommandAction
import com.intellij.openapi.diagnostic.Logger
import com.intellij.openapi.editor.Document
import com.intellij.openapi.project.Project
import com.intellij.psi.PsiDocumentManager
import com.intellij.psi.PsiFile
import com.intellij.psi.util.PsiTreeUtil
import kotlinx.coroutines.*
import java.util.concurrent.ConcurrentHashMap

/**
 * Implementation of the HealingService interface
 */
class HealingServiceImpl : HealingService {
    
    companion object {
        private val LOG = Logger.getInstance(HealingServiceImpl::class.java)
    }
    
    private val configurationService: ConfigurationService by lazy {
        ApplicationManager.getApplication().getService(ConfigurationService::class.java)
    }
    
    private val telemetryService: TelemetryService by lazy {
        ApplicationManager.getApplication().getService(TelemetryService::class.java)
    }
    
    private val apiClient = HomeostasisApiClient()
    private val scope = CoroutineScope(Dispatchers.IO + SupervisorJob())
    private val activeAnalyses = ConcurrentHashMap<String, Job>()
    
    override fun initialize() {
        LOG.info("Initializing HealingService")
        apiClient.initialize(configurationService.getServerUrl(), configurationService.getApiKey())
    }
    
    override fun dispose() {
        LOG.info("Disposing HealingService")
        scope.cancel()
        apiClient.dispose()
    }
    
    override suspend fun analyzeDocument(document: Document, project: Project): List<HealingSuggestion> {
        return withContext(Dispatchers.IO) {
            try {
                val psiFile = PsiDocumentManager.getInstance(project).getPsiFile(document)
                if (psiFile != null) {
                    analyzePsiFile(psiFile)
                } else {
                    emptyList()
                }
            } catch (e: Exception) {
                LOG.error("Error analyzing document", e)
                telemetryService.sendError("document_analysis_error", e)
                emptyList()
            }
        }
    }
    
    override suspend fun analyzePsiFile(psiFile: PsiFile): List<HealingSuggestion> {
        return withContext(Dispatchers.IO) {
            try {
                val language = psiFile.language.id.lowercase()
                val enabledLanguages = configurationService.getEnabledLanguages()
                
                if (!enabledLanguages.contains(language)) {
                    return@withContext emptyList()
                }
                
                val content = psiFile.text
                val filePath = psiFile.virtualFile?.path ?: "unknown"
                
                // Cancel any previous analysis for this file
                activeAnalyses[filePath]?.cancel()
                
                // Start new analysis
                val analysisJob = async {
                    apiClient.analyzeCode(content, language, filePath)
                }
                
                activeAnalyses[filePath] = analysisJob
                val result = analysisJob.await()
                activeAnalyses.remove(filePath)
                
                telemetryService.sendEvent("file_analyzed", mapOf(
                    "language" to language,
                    "suggestions_count" to result.size,
                    "file_size" to content.length
                ))
                
                result
                
            } catch (e: Exception) {
                LOG.error("Error analyzing PSI file: ${psiFile.name}", e)
                telemetryService.sendError("psi_analysis_error", e)
                emptyList()
            }
        }
    }
    
    override suspend fun applyHealing(
        suggestion: HealingSuggestion, 
        document: Document, 
        project: Project
    ): Boolean {
        return withContext(Dispatchers.EDT) {
            try {
                WriteCommandAction.runWriteCommandAction(project) {
                    if (suggestion.replacement != null) {
                        document.replaceString(
                            suggestion.startOffset, 
                            suggestion.endOffset, 
                            suggestion.replacement
                        )
                    }
                    
                    // Apply additional changes if any
                    suggestion.additionalChanges.forEach { change ->
                        // Handle additional file changes
                        applyAdditionalChange(change, project)
                    }
                }
                
                telemetryService.sendEvent("healing_applied", mapOf(
                    "rule_id" to suggestion.ruleId,
                    "language" to suggestion.language,
                    "confidence" to suggestion.confidence,
                    "severity" to suggestion.severity.name
                ))
                
                true
                
            } catch (e: Exception) {
                LOG.error("Error applying healing suggestion", e)
                telemetryService.sendError("healing_application_error", e)
                false
            }
        }
    }
    
    override suspend fun healFile(psiFile: PsiFile): Int {
        return withContext(Dispatchers.IO) {
            try {
                val suggestions = analyzePsiFile(psiFile)
                val document = PsiDocumentManager.getInstance(psiFile.project).getDocument(psiFile)
                
                if (document == null) {
                    LOG.warn("Could not get document for file: ${psiFile.name}")
                    return@withContext 0
                }
                
                var healingsApplied = 0
                val confidenceThreshold = getConfidenceThreshold()
                
                // Sort suggestions by confidence (highest first) and start offset (latest first to avoid offset conflicts)
                val sortedSuggestions = suggestions
                    .filter { it.confidence >= confidenceThreshold }
                    .sortedWith(compareByDescending<HealingSuggestion> { it.confidence }
                        .thenByDescending { it.startOffset })
                
                for (suggestion in sortedSuggestions) {
                    if (applyHealing(suggestion, document, psiFile.project)) {
                        healingsApplied++
                    }
                }
                
                telemetryService.sendEvent("file_healed", mapOf(
                    "language" to psiFile.language.id,
                    "healings_applied" to healingsApplied,
                    "total_suggestions" to suggestions.size
                ))
                
                healingsApplied
                
            } catch (e: Exception) {
                LOG.error("Error healing file: ${psiFile.name}", e)
                telemetryService.sendError("file_healing_error", e)
                0
            }
        }
    }
    
    override suspend fun healProject(project: Project): Int {
        return withContext(Dispatchers.IO) {
            try {
                var totalHealings = 0
                val enabledLanguages = configurationService.getEnabledLanguages()
                
                // Find all relevant files in the project
                val psiFiles = findHealableFiles(project, enabledLanguages)
                
                // Process files in parallel with limited concurrency
                val semaphore = kotlinx.coroutines.sync.Semaphore(4) // Max 4 concurrent healings
                
                val healingJobs = psiFiles.map { psiFile ->
                    async {
                        semaphore.withPermit {
                            healFile(psiFile)
                        }
                    }
                }
                
                totalHealings = healingJobs.awaitAll().sum()
                
                telemetryService.sendEvent("project_healed", mapOf(
                    "project_name" to project.name,
                    "files_processed" to psiFiles.size,
                    "total_healings" to totalHealings
                ))
                
                totalHealings
                
            } catch (e: Exception) {
                LOG.error("Error healing project: ${project.name}", e)
                telemetryService.sendError("project_healing_error", e)
                0
            }
        }
    }
    
    override fun isRealTimeHealingEnabled(): Boolean {
        return configurationService.isRealTimeHealingEnabled()
    }
    
    override fun setRealTimeHealingEnabled(enabled: Boolean) {
        configurationService.setRealTimeHealingEnabled(enabled)
    }
    
    override fun getConfidenceThreshold(): Double {
        return configurationService.getConfidenceThreshold()
    }
    
    override fun setConfidenceThreshold(threshold: Double) {
        configurationService.setConfidenceThreshold(threshold)
    }
    
    private fun applyAdditionalChange(change: AdditionalChange, project: Project) {
        // Implementation for applying additional changes to other files
        // This would involve finding the file, getting its document, and applying the change
        LOG.info("Applying additional change to: ${change.filePath}")
    }
    
    private fun findHealableFiles(project: Project, enabledLanguages: List<String>): List<PsiFile> {
        // Implementation to find all PSI files in the project that match enabled languages
        // This would use PsiManager to traverse the project structure
        return emptyList() // Placeholder
    }
}