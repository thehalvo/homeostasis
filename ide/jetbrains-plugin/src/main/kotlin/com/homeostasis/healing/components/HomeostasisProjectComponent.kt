package com.homeostasis.healing.components

import com.homeostasis.healing.services.ProjectHealingService
import com.intellij.openapi.components.ProjectComponent
import com.intellij.openapi.diagnostic.Logger
import com.intellij.openapi.project.Project

/**
 * Project-level component that manages healing for individual projects
 */
class HomeostasisProjectComponent(private val project: Project) : ProjectComponent {
    
    companion object {
        private val LOG = Logger.getInstance(HomeostasisProjectComponent::class.java)
        
        fun getInstance(project: Project): HomeostasisProjectComponent {
            return project.getComponent(HomeostasisProjectComponent::class.java)
        }
    }
    
    private val projectHealingService: ProjectHealingService by lazy {
        project.getService(ProjectHealingService::class.java)
    }
    
    override fun initComponent() {
        LOG.info("Initializing Homeostasis project component for: ${project.name}")
        
        try {
            // Initialize project-specific healing service
            projectHealingService.initialize()
            
            LOG.info("Homeostasis project component initialized for: ${project.name}")
            
        } catch (e: Exception) {
            LOG.error("Failed to initialize Homeostasis project component for: ${project.name}", e)
        }
    }
    
    override fun disposeComponent() {
        LOG.info("Disposing Homeostasis project component for: ${project.name}")
        
        try {
            // Dispose project-specific resources
            projectHealingService.dispose()
            
            LOG.info("Homeostasis project component disposed for: ${project.name}")
            
        } catch (e: Exception) {
            LOG.error("Error disposing Homeostasis project component for: ${project.name}", e)
        }
    }
    
    override fun getComponentName(): String = "HomeostasisProjectComponent"
}