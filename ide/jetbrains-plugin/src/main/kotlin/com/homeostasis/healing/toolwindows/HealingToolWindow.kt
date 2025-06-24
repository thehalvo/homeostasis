package com.homeostasis.healing.toolwindows

import com.homeostasis.healing.services.HealingService
import com.intellij.openapi.application.ApplicationManager
import com.intellij.openapi.project.Project
import com.intellij.ui.components.JBLabel
import com.intellij.ui.components.JBScrollPane
import com.intellij.ui.table.JBTable
import com.intellij.util.ui.FormBuilder
import java.awt.BorderLayout
import java.text.SimpleDateFormat
import java.util.*
import javax.swing.*
import javax.swing.table.DefaultTableModel

/**
 * Tool window showing Homeostasis healing statistics and history
 */
class HealingToolWindow(private val project: Project) {
    
    private val healingService: HealingService by lazy {
        ApplicationManager.getApplication().getService(HealingService::class.java)
    }
    
    private val mainPanel = JPanel(BorderLayout())
    private val statisticsPanel = createStatisticsPanel()
    private val historyTable = createHistoryTable()
    
    init {
        setupUI()
    }
    
    fun getContent(): JComponent = mainPanel
    
    private fun setupUI() {
        val tabbedPane = JTabbedPane()
        
        // Statistics tab
        tabbedPane.addTab("Statistics", statisticsPanel)
        
        // History tab
        val historyPanel = JPanel(BorderLayout())
        historyPanel.add(JBScrollPane(historyTable), BorderLayout.CENTER)
        tabbedPane.addTab("History", historyPanel)
        
        mainPanel.add(tabbedPane, BorderLayout.CENTER)
        
        // Add toolbar
        val toolbar = createToolbar()
        mainPanel.add(toolbar, BorderLayout.NORTH)
    }
    
    private fun createStatisticsPanel(): JPanel {
        val statsLabels = mapOf(
            "realTimeStatus" to JBLabel("Real-time Healing: Enabled"),
            "confidenceThreshold" to JBLabel("Confidence Threshold: 0.7"),
            "enabledLanguages" to JBLabel("Enabled Languages: 15"),
            "totalHealings" to JBLabel("Total Healings: 0"),
            "sessionHealings" to JBLabel("Session Healings: 0"),
            "successRate" to JBLabel("Success Rate: 100%"),
            "averageConfidence" to JBLabel("Average Confidence: 0.85")
        )
        
        val builder = FormBuilder.createFormBuilder()
        
        builder.addComponent(JBLabel("Homeostasis Healing Status"))
        builder.addSeparator()
        
        statsLabels.values.forEach { label ->
            builder.addComponent(label)
        }
        
        return builder.panel
    }
    
    private fun createHistoryTable(): JBTable {
        val columnNames = arrayOf("Time", "File", "Language", "Rule", "Confidence", "Status")
        val model = DefaultTableModel(columnNames, 0)
        
        // Add sample data
        val dateFormat = SimpleDateFormat("HH:mm:ss")
        model.addRow(arrayOf(
            dateFormat.format(Date()),
            "example.py",
            "Python",
            "null_reference_fix",
            "0.92",
            "Applied"
        ))
        
        model.addRow(arrayOf(
            dateFormat.format(Date(System.currentTimeMillis() - 60000)),
            "app.js",
            "JavaScript",
            "undefined_variable_fix",
            "0.78",
            "Applied"
        ))
        
        val table = JBTable(model)
        table.fillsViewportHeight = true
        
        return table
    }
    
    private fun createToolbar(): JComponent {
        val toolbar = JPanel()
        toolbar.layout = BoxLayout(toolbar, BoxLayout.X_AXIS)
        
        val refreshButton = JButton("Refresh")
        refreshButton.addActionListener {
            refreshStatistics()
        }
        
        val clearButton = JButton("Clear History")
        clearButton.addActionListener {
            clearHistory()
        }
        
        val configButton = JButton("Configure")
        configButton.addActionListener {
            openConfiguration()
        }
        
        toolbar.add(refreshButton)
        toolbar.add(Box.createHorizontalStrut(5))
        toolbar.add(clearButton)
        toolbar.add(Box.createHorizontalStrut(5))
        toolbar.add(configButton)
        toolbar.add(Box.createHorizontalGlue())
        
        return toolbar
    }
    
    private fun refreshStatistics() {
        // Update statistics labels with current values
        SwingUtilities.invokeLater {
            // This would be implemented to fetch actual statistics
            // For now, just trigger a refresh of the display
            mainPanel.repaint()
        }
    }
    
    private fun clearHistory() {
        val model = historyTable.model as DefaultTableModel
        model.rowCount = 0
    }
    
    private fun openConfiguration() {
        // Open the Homeostasis configuration dialog
        // This would integrate with the IDE's settings system
    }
}