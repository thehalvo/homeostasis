import * as vscode from 'vscode';
import { HealingService } from './services/healingService';
import { DiagnosticsProvider } from './providers/diagnosticsProvider';
import { CodeLensProvider } from './providers/codeLensProvider';
import { InlineFixProvider } from './providers/inlineFixProvider';
import { TelemetryService } from './services/telemetryService';
import { ConfigurationManager } from './services/configurationManager';
import { SettingsSyncService } from './services/settingsSyncService';

export function activate(context: vscode.ExtensionContext) {
    console.log('Homeostasis Self-Healing extension is now active!');

    // Initialize services
    const configManager = new ConfigurationManager();
    const telemetryService = new TelemetryService(configManager);
    const settingsSyncService = new SettingsSyncService(configManager, telemetryService);
    const healingService = new HealingService(configManager, telemetryService);
    const diagnosticsProvider = new DiagnosticsProvider(healingService);
    const codeLensProvider = new CodeLensProvider(healingService);
    const inlineFixProvider = new InlineFixProvider(healingService);

    // Register providers
    const supportedLanguages = configManager.getEnabledLanguages();
    
    // Register diagnostics provider for all supported languages
    supportedLanguages.forEach(language => {
        vscode.languages.registerCodeActionsProvider(
            { language: language },
            inlineFixProvider,
            {
                providedCodeActionKinds: [vscode.CodeActionKind.QuickFix]
            }
        );
    });

    // Register code lens provider if enabled
    if (configManager.isCodeLensEnabled()) {
        supportedLanguages.forEach(language => {
            vscode.languages.registerCodeLensProvider(
                { language: language },
                codeLensProvider
            );
        });
    }

    // Register commands
    const healFileCommand = vscode.commands.registerCommand('homeostasis.healFile', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showWarningMessage('No active editor found');
            return;
        }

        const document = editor.document;
        await healingService.healDocument(document);
    });

    const healWorkspaceCommand = vscode.commands.registerCommand('homeostasis.healWorkspace', async () => {
        if (!vscode.workspace.workspaceFolders) {
            vscode.window.showWarningMessage('No workspace found');
            return;
        }

        await healingService.healWorkspace();
    });

    const enableRealTimeCommand = vscode.commands.registerCommand('homeostasis.enableRealTimeHealing', () => {
        configManager.setRealTimeHealing(true);
        vscode.window.showInformationMessage('Real-time healing enabled');
    });

    const disableRealTimeCommand = vscode.commands.registerCommand('homeostasis.disableRealTimeHealing', () => {
        configManager.setRealTimeHealing(false);
        vscode.window.showInformationMessage('Real-time healing disabled');
    });

    const showDashboardCommand = vscode.commands.registerCommand('homeostasis.showHealingDashboard', () => {
        const serverUrl = configManager.getServerUrl();
        vscode.env.openExternal(vscode.Uri.parse(`${serverUrl}/dashboard`));
    });

    const configureTelemetryCommand = vscode.commands.registerCommand('homeostasis.configureTelemetry', async () => {
        const enabled = await vscode.window.showQuickPick(['Enable', 'Disable'], {
            placeHolder: 'Configure telemetry settings'
        });
        
        if (enabled) {
            const isEnabled = enabled === 'Enable';
            configManager.setTelemetryEnabled(isEnabled);
            vscode.window.showInformationMessage(`Telemetry ${isEnabled ? 'enabled' : 'disabled'}`);
        }
    });

    // Settings sync commands
    const syncSettingsCommand = vscode.commands.registerCommand('homeostasis.syncSettings', async () => {
        const success = await settingsSyncService.syncSettings();
        if (success) {
            vscode.window.showInformationMessage('Settings synchronized successfully');
        } else {
            vscode.window.showErrorMessage('Failed to synchronize settings');
        }
    });

    const createProfileCommand = vscode.commands.registerCommand('homeostasis.createSettingsProfile', async () => {
        const name = await vscode.window.showInputBox({
            prompt: 'Enter a name for this settings profile',
            placeHolder: 'My Development Settings'
        });
        
        if (name) {
            try {
                const profileId = await settingsSyncService.createSettingsProfile(name);
                vscode.window.showInformationMessage(`Settings profile "${name}" created with ID: ${profileId}`);
            } catch (error) {
                vscode.window.showErrorMessage(`Failed to create profile: ${(error as Error).message}`);
            }
        }
    });

    const loadProfileCommand = vscode.commands.registerCommand('homeostasis.loadSettingsProfile', async () => {
        try {
            const profiles = await settingsSyncService.listSettingsProfiles();
            if (profiles.length === 0) {
                vscode.window.showInformationMessage('No settings profiles found');
                return;
            }

            const items = profiles.map(profile => ({
                label: profile.name,
                description: `Last modified: ${new Date(profile.lastModified).toLocaleString()}`,
                detail: `Device: ${profile.deviceId}`,
                profileId: profile.id
            }));

            const selected = await vscode.window.showQuickPick(items, {
                placeHolder: 'Select a settings profile to load'
            });

            if (selected) {
                await settingsSyncService.loadSettingsProfile(selected.profileId);
            }
        } catch (error) {
            vscode.window.showErrorMessage(`Failed to load profiles: ${(error as Error).message}`);
        }
    });

    const exportSettingsCommand = vscode.commands.registerCommand('homeostasis.exportSettings', async () => {
        try {
            const exportData = await settingsSyncService.exportSettings();
            const doc = await vscode.workspace.openTextDocument({
                content: exportData,
                language: 'json'
            });
            await vscode.window.showTextDocument(doc);
            vscode.window.showInformationMessage('Settings exported to new document');
        } catch (error) {
            vscode.window.showErrorMessage(`Failed to export settings: ${(error as Error).message}`);
        }
    });

    const importSettingsCommand = vscode.commands.registerCommand('homeostasis.importSettings', async () => {
        const uris = await vscode.window.showOpenDialog({
            canSelectFiles: true,
            canSelectFolders: false,
            canSelectMany: false,
            filters: {
                'JSON Files': ['json'],
                'All Files': ['*']
            }
        });

        if (uris && uris.length > 0) {
            try {
                const document = await vscode.workspace.openTextDocument(uris[0]);
                const content = document.getText();
                await settingsSyncService.importSettings(content);
            } catch (error) {
                vscode.window.showErrorMessage(`Failed to import settings: ${(error as Error).message}`);
            }
        }
    });

    // Real-time healing setup
    let typingTimer: NodeJS.Timeout;
    const onDocumentChange = vscode.workspace.onDidChangeTextDocument((event) => {
        if (!configManager.isRealTimeHealingEnabled()) {
            return;
        }

        // Clear existing timer
        if (typingTimer) {
            clearTimeout(typingTimer);
        }

        // Set new timer
        typingTimer = setTimeout(async () => {
            await healingService.analyzeDocument(event.document);
        }, configManager.getHealingDelay());
    });

    // Configuration change listener
    const onConfigChange = vscode.workspace.onDidChangeConfiguration(event => {
        if (event.affectsConfiguration('homeostasis')) {
            configManager.reloadConfiguration();
            settingsSyncService.onConfigurationChanged();
            vscode.window.showInformationMessage('Homeostasis configuration reloaded');
        }
    });

    // Diagnostics collection
    const diagnosticCollection = vscode.languages.createDiagnosticCollection('homeostasis');
    diagnosticsProvider.setDiagnosticCollection(diagnosticCollection);

    // Status bar item
    const statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
    statusBarItem.text = "$(pulse) Homeostasis";
    statusBarItem.tooltip = "Homeostasis Self-Healing is active";
    statusBarItem.command = 'homeostasis.showHealingDashboard';
    statusBarItem.show();

    // Register disposables
    context.subscriptions.push(
        healFileCommand,
        healWorkspaceCommand,
        enableRealTimeCommand,
        disableRealTimeCommand,
        showDashboardCommand,
        configureTelemetryCommand,
        syncSettingsCommand,
        createProfileCommand,
        loadProfileCommand,
        exportSettingsCommand,
        importSettingsCommand,
        onDocumentChange,
        onConfigChange,
        diagnosticCollection,
        statusBarItem,
        settingsSyncService
    );

    // Initialize healing service connection
    healingService.initialize();

    // Send activation telemetry
    telemetryService.sendEvent('extension.activated', {
        version: context.extension.packageJSON.version,
        vscodeVersion: vscode.version
    });
}

export function deactivate() {
    console.log('Homeostasis Self-Healing extension is now deactivated');
}