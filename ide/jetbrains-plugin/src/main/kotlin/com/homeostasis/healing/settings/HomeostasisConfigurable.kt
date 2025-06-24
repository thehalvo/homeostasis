package com.homeostasis.healing.settings

import com.homeostasis.healing.services.ConfigurationService
import com.intellij.openapi.application.ApplicationManager
import com.intellij.openapi.options.Configurable
import com.intellij.openapi.ui.ComboBox
import com.intellij.ui.components.JBCheckBox
import com.intellij.ui.components.JBLabel
import com.intellij.ui.components.JBTextField
import com.intellij.util.ui.FormBuilder
import javax.swing.*

/**
 * Configuration panel for Homeostasis settings
 */
class HomeostasisConfigurable : Configurable {
    
    private val configurationService: ConfigurationService by lazy {
        ApplicationManager.getApplication().getService(ConfigurationService::class.java)
    }
    
    // UI Components
    private val serverUrlField = JBTextField()
    private val apiKeyField = JPasswordField()
    private val realTimeHealingCheckBox = JBCheckBox("Enable real-time healing")
    private val healingDelaySpinner = JSpinner(SpinnerNumberModel(2000, 500, 10000, 500))
    private val confidenceThresholdSpinner = JSpinner(SpinnerNumberModel(0.7, 0.0, 1.0, 0.1))
    private val telemetryCheckBox = JBCheckBox("Enable telemetry")
    private val inlineHintsCheckBox = JBCheckBox("Show inline hints")
    private val inspectionsCheckBox = JBCheckBox("Enable inspections")
    private val notificationsCheckBox = JBCheckBox("Enable notifications")
    
    // Language checkboxes
    private val languageCheckBoxes = mutableMapOf<String, JBCheckBox>()
    private val supportedLanguages = listOf(
        "python", "java", "javascript", "typescript", "go", "rust", 
        "csharp", "php", "ruby", "scala", "elixir", "clojure", 
        "swift", "kotlin", "dart"
    )
    
    private var mainPanel: JPanel? = null
    
    override fun getDisplayName(): String = "Homeostasis"
    
    override fun createComponent(): JComponent? {
        if (mainPanel == null) {
            mainPanel = createMainPanel()
        }
        return mainPanel
    }
    
    private fun createMainPanel(): JPanel {
        // Initialize language checkboxes
        supportedLanguages.forEach { language ->
            languageCheckBoxes[language] = JBCheckBox(language.capitalize())
        }
        
        val builder = FormBuilder.createFormBuilder()
        
        // Server Configuration
        builder.addLabeledComponent("Server URL:", serverUrlField)
        builder.addLabeledComponent("API Key:", apiKeyField)
        builder.addSeparator()
        
        // Healing Configuration
        builder.addComponent(realTimeHealingCheckBox)
        builder.addLabeledComponent("Healing delay (ms):", healingDelaySpinner)
        builder.addLabeledComponent("Confidence threshold:", confidenceThresholdSpinner)
        builder.addSeparator()
        
        // UI Configuration
        builder.addComponent(inlineHintsCheckBox)
        builder.addComponent(inspectionsCheckBox)
        builder.addComponent(notificationsCheckBox)
        builder.addSeparator()
        
        // Language Configuration
        builder.addComponent(JBLabel("Enabled Languages:"))
        val languagePanel = JPanel()
        languagePanel.layout = BoxLayout(languagePanel, BoxLayout.Y_AXIS)
        
        // Group languages in rows of 3
        val languageRows = languageCheckBoxes.values.chunked(3)
        languageRows.forEach { row ->
            val rowPanel = JPanel()
            rowPanel.layout = BoxLayout(rowPanel, BoxLayout.X_AXIS)
            row.forEach { checkbox ->
                rowPanel.add(checkbox)
                rowPanel.add(Box.createHorizontalStrut(10))
            }
            languagePanel.add(rowPanel)
        }
        
        builder.addComponent(languagePanel)
        builder.addSeparator()
        
        // Telemetry Configuration
        builder.addComponent(telemetryCheckBox)
        
        return builder.panel
    }
    
    override fun isModified(): Boolean {
        return serverUrlField.text != configurationService.getServerUrl() ||
                String(apiKeyField.password) != configurationService.getApiKey() ||
                realTimeHealingCheckBox.isSelected != configurationService.isRealTimeHealingEnabled() ||
                (healingDelaySpinner.value as Int) != configurationService.getHealingDelay() ||
                (confidenceThresholdSpinner.value as Double) != configurationService.getConfidenceThreshold() ||
                telemetryCheckBox.isSelected != configurationService.isTelemetryEnabled() ||
                inlineHintsCheckBox.isSelected != configurationService.isInlineHintsEnabled() ||
                inspectionsCheckBox.isSelected != configurationService.isInspectionsEnabled() ||
                notificationsCheckBox.isSelected != configurationService.isNotificationsEnabled() ||
                getSelectedLanguages() != configurationService.getEnabledLanguages()
    }
    
    override fun apply() {
        configurationService.setServerUrl(serverUrlField.text)
        configurationService.setApiKey(String(apiKeyField.password))
        configurationService.setRealTimeHealingEnabled(realTimeHealingCheckBox.isSelected)
        configurationService.setHealingDelay(healingDelaySpinner.value as Int)
        configurationService.setConfidenceThreshold(confidenceThresholdSpinner.value as Double)
        configurationService.setTelemetryEnabled(telemetryCheckBox.isSelected)
        configurationService.setInlineHintsEnabled(inlineHintsCheckBox.isSelected)
        configurationService.setInspectionsEnabled(inspectionsCheckBox.isSelected)
        configurationService.setNotificationsEnabled(notificationsCheckBox.isSelected)
        configurationService.setEnabledLanguages(getSelectedLanguages())
        configurationService.saveConfiguration()
    }
    
    override fun reset() {
        serverUrlField.text = configurationService.getServerUrl()
        apiKeyField.text = configurationService.getApiKey()
        realTimeHealingCheckBox.isSelected = configurationService.isRealTimeHealingEnabled()
        healingDelaySpinner.value = configurationService.getHealingDelay()
        confidenceThresholdSpinner.value = configurationService.getConfidenceThreshold()
        telemetryCheckBox.isSelected = configurationService.isTelemetryEnabled()
        inlineHintsCheckBox.isSelected = configurationService.isInlineHintsEnabled()
        inspectionsCheckBox.isSelected = configurationService.isInspectionsEnabled()
        notificationsCheckBox.isSelected = configurationService.isNotificationsEnabled()
        
        // Reset language selections
        val enabledLanguages = configurationService.getEnabledLanguages()
        languageCheckBoxes.forEach { (language, checkbox) ->
            checkbox.isSelected = enabledLanguages.contains(language)
        }
    }
    
    private fun getSelectedLanguages(): List<String> {
        return languageCheckBoxes
            .filter { it.value.isSelected }
            .map { it.key }
    }
}