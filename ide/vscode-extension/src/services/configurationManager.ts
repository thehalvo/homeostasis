import * as vscode from 'vscode';

export class ConfigurationManager {
    private configuration: vscode.WorkspaceConfiguration;

    constructor() {
        this.configuration = vscode.workspace.getConfiguration('homeostasis');
    }

    reloadConfiguration(): void {
        this.configuration = vscode.workspace.getConfiguration('homeostasis');
    }

    getServerUrl(): string {
        return this.configuration.get('serverUrl', 'http://localhost:8080');
    }

    getApiKey(): string {
        return this.configuration.get('apiKey', '');
    }

    isRealTimeHealingEnabled(): boolean {
        return this.configuration.get('realTimeHealing', true);
    }

    setRealTimeHealing(enabled: boolean): void {
        this.configuration.update('realTimeHealing', enabled, vscode.ConfigurationTarget.Global);
    }

    getHealingDelay(): number {
        return this.configuration.get('healingDelay', 2000);
    }

    getEnabledLanguages(): string[] {
        return this.configuration.get('enabledLanguages', ['python', 'javascript', 'typescript', 'java', 'go']);
    }

    getConfidenceThreshold(): number {
        return this.configuration.get('confidenceThreshold', 0.7);
    }

    isTelemetryEnabled(): boolean {
        return this.configuration.get('enableTelemetry', true);
    }

    setTelemetryEnabled(enabled: boolean): void {
        this.configuration.update('enableTelemetry', enabled, vscode.ConfigurationTarget.Global);
    }

    isInlineHintsEnabled(): boolean {
        return this.configuration.get('showInlineHints', true);
    }

    isCodeLensEnabled(): boolean {
        return this.configuration.get('enableCodeLens', true);
    }
}