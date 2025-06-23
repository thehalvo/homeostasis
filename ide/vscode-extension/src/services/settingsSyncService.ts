import * as vscode from 'vscode';
import axios, { AxiosInstance } from 'axios';
import { ConfigurationManager } from './configurationManager';
import { TelemetryService } from './telemetryService';

export interface SyncableSettings {
    realTimeHealing: boolean;
    healingDelay: number;
    enabledLanguages: string[];
    confidenceThreshold: number;
    showInlineHints: boolean;
    enableCodeLens: boolean;
    enableTelemetry: boolean;
}

export interface SettingsProfile {
    id: string;
    name: string;
    settings: SyncableSettings;
    lastModified: number;
    deviceId: string;
    userId?: string;
}

export class SettingsSyncService {
    private httpClient: AxiosInstance;
    private syncTimer?: NodeJS.Timeout;
    private deviceId: string;
    private readonly SYNC_INTERVAL = 300000; // 5 minutes
    private lastSyncTime: number = 0;

    constructor(
        private configManager: ConfigurationManager,
        private telemetryService: TelemetryService
    ) {
        this.deviceId = this.generateDeviceId();
        this.httpClient = axios.create({
            timeout: 10000,
            headers: {
                'Content-Type': 'application/json',
                'User-Agent': 'Homeostasis-VSCode-Extension'
            }
        });

        this.setupHttpClient();
        this.startSyncTimer();
    }

    private setupHttpClient(): void {
        this.httpClient.defaults.baseURL = this.configManager.getServerUrl();
        
        const apiKey = this.configManager.getApiKey();
        if (apiKey) {
            this.httpClient.defaults.headers.common['Authorization'] = `Bearer ${apiKey}`;
        }
    }

    private startSyncTimer(): void {
        this.syncTimer = setInterval(() => {
            this.syncSettings();
        }, this.SYNC_INTERVAL);
    }

    async syncSettings(): Promise<boolean> {
        try {
            const localSettings = this.getCurrentSettings();
            const remoteProfile = await this.fetchRemoteSettings();

            if (remoteProfile && this.shouldUseRemoteSettings(remoteProfile)) {
                await this.applyRemoteSettings(remoteProfile.settings);
                this.lastSyncTime = Date.now();
                
                this.telemetryService.sendEvent('settings.sync.downloaded', {
                    profileId: remoteProfile.id,
                    deviceId: this.deviceId
                });
                
                vscode.window.showInformationMessage('Settings synchronized from server');
                return true;
            } else {
                // Upload local settings if they're newer or remote doesn't exist
                await this.uploadLocalSettings(localSettings);
                this.lastSyncTime = Date.now();
                
                this.telemetryService.sendEvent('settings.sync.uploaded', {
                    deviceId: this.deviceId
                });
                
                return true;
            }
        } catch (error) {
            this.telemetryService.sendEvent('settings.sync.failed', {
                error: (error as Error).message,
                deviceId: this.deviceId
            });
            
            console.error('Settings sync failed:', error);
            return false;
        }
    }

    private async fetchRemoteSettings(): Promise<SettingsProfile | null> {
        try {
            const response = await this.httpClient.get('/api/settings/profile', {
                params: {
                    deviceId: this.deviceId
                }
            });
            return response.data;
        } catch (error: any) {
            if (error.response?.status === 404) {
                // No remote profile exists yet
                return null;
            }
            throw error;
        }
    }

    private async uploadLocalSettings(settings: SyncableSettings): Promise<void> {
        const profile: Omit<SettingsProfile, 'id'> = {
            name: `VSCode-${this.deviceId}`,
            settings,
            lastModified: Date.now(),
            deviceId: this.deviceId
        };

        await this.httpClient.post('/api/settings/profile', profile);
    }

    private shouldUseRemoteSettings(remoteProfile: SettingsProfile): boolean {
        // Use remote settings if they're newer than our last sync
        return remoteProfile.lastModified > this.lastSyncTime;
    }

    private getCurrentSettings(): SyncableSettings {
        return {
            realTimeHealing: this.configManager.isRealTimeHealingEnabled(),
            healingDelay: this.configManager.getHealingDelay(),
            enabledLanguages: this.configManager.getEnabledLanguages(),
            confidenceThreshold: this.configManager.getConfidenceThreshold(),
            showInlineHints: this.configManager.isInlineHintsEnabled(),
            enableCodeLens: this.configManager.isCodeLensEnabled(),
            enableTelemetry: this.configManager.isTelemetryEnabled()
        };
    }

    private async applyRemoteSettings(settings: SyncableSettings): Promise<void> {
        const config = vscode.workspace.getConfiguration('homeostasis');
        
        await Promise.all([
            config.update('realTimeHealing', settings.realTimeHealing, vscode.ConfigurationTarget.Global),
            config.update('healingDelay', settings.healingDelay, vscode.ConfigurationTarget.Global),
            config.update('enabledLanguages', settings.enabledLanguages, vscode.ConfigurationTarget.Global),
            config.update('confidenceThreshold', settings.confidenceThreshold, vscode.ConfigurationTarget.Global),
            config.update('showInlineHints', settings.showInlineHints, vscode.ConfigurationTarget.Global),
            config.update('enableCodeLens', settings.enableCodeLens, vscode.ConfigurationTarget.Global),
            config.update('enableTelemetry', settings.enableTelemetry, vscode.ConfigurationTarget.Global)
        ]);

        // Reload configuration in our manager
        this.configManager.reloadConfiguration();
    }

    async createSettingsProfile(name: string): Promise<string> {
        const settings = this.getCurrentSettings();
        const profile: Omit<SettingsProfile, 'id'> = {
            name,
            settings,
            lastModified: Date.now(),
            deviceId: this.deviceId
        };

        const response = await this.httpClient.post('/api/settings/profile', profile);
        return response.data.id;
    }

    async loadSettingsProfile(profileId: string): Promise<boolean> {
        try {
            const response = await this.httpClient.get(`/api/settings/profile/${profileId}`);
            const profile: SettingsProfile = response.data;
            
            await this.applyRemoteSettings(profile.settings);
            
            this.telemetryService.sendEvent('settings.profile.loaded', {
                profileId,
                deviceId: this.deviceId
            });
            
            vscode.window.showInformationMessage(`Settings profile "${profile.name}" loaded`);
            return true;
        } catch (error) {
            this.telemetryService.sendEvent('settings.profile.load_failed', {
                profileId,
                error: (error as Error).message,
                deviceId: this.deviceId
            });
            
            vscode.window.showErrorMessage(`Failed to load settings profile: ${(error as Error).message}`);
            return false;
        }
    }

    async listSettingsProfiles(): Promise<SettingsProfile[]> {
        try {
            const response = await this.httpClient.get('/api/settings/profiles', {
                params: {
                    deviceId: this.deviceId
                }
            });
            return response.data;
        } catch (error) {
            console.error('Failed to list settings profiles:', error);
            return [];
        }
    }

    async deleteSettingsProfile(profileId: string): Promise<boolean> {
        try {
            await this.httpClient.delete(`/api/settings/profile/${profileId}`);
            
            this.telemetryService.sendEvent('settings.profile.deleted', {
                profileId,
                deviceId: this.deviceId
            });
            
            return true;
        } catch (error) {
            console.error('Failed to delete settings profile:', error);
            return false;
        }
    }

    async exportSettings(): Promise<string> {
        const settings = this.getCurrentSettings();
        const exportData = {
            version: '1.0',
            timestamp: Date.now(),
            deviceId: this.deviceId,
            settings
        };

        return JSON.stringify(exportData, null, 2);
    }

    async importSettings(data: string): Promise<boolean> {
        try {
            const importData = JSON.parse(data);
            
            if (!importData.settings || !importData.version) {
                throw new Error('Invalid settings format');
            }

            await this.applyRemoteSettings(importData.settings);
            
            this.telemetryService.sendEvent('settings.imported', {
                version: importData.version,
                deviceId: this.deviceId
            });
            
            vscode.window.showInformationMessage('Settings imported successfully');
            return true;
        } catch (error) {
            vscode.window.showErrorMessage(`Failed to import settings: ${(error as Error).message}`);
            return false;
        }
    }

    private generateDeviceId(): string {
        // Generate a stable device identifier
        const machineId = vscode.env.machineId;
        const sessionId = vscode.env.sessionId;
        return `${machineId}-${sessionId}`.substring(0, 32);
    }

    onConfigurationChanged(): void {
        // Trigger sync when local settings change
        setTimeout(() => {
            this.syncSettings();
        }, 1000); // Debounce configuration changes
    }

    dispose(): void {
        if (this.syncTimer) {
            clearInterval(this.syncTimer);
        }
    }
}