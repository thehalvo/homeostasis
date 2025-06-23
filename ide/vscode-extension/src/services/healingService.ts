import * as vscode from 'vscode';
import axios, { AxiosInstance } from 'axios';
import { ConfigurationManager } from './configurationManager';
import { TelemetryService } from './telemetryService';

export interface HealingSuggestion {
    id: string;
    title: string;
    description: string;
    confidence: number;
    range: vscode.Range;
    fix: string;
    kind: string;
}

export interface AnalysisResult {
    errors: DiagnosticInfo[];
    suggestions: HealingSuggestion[];
    metadata: {
        analysisTime: number;
        language: string;
        rulesMatched: number;
    };
}

export interface DiagnosticInfo {
    message: string;
    severity: 'error' | 'warning' | 'info';
    range: vscode.Range;
    source: string;
    code?: string;
}

export class HealingService {
    private httpClient: AxiosInstance;
    private isConnected: boolean = false;
    private pendingAnalysis: Map<string, Promise<AnalysisResult>> = new Map();

    constructor(
        private configManager: ConfigurationManager,
        private telemetryService: TelemetryService
    ) {
        this.httpClient = axios.create({
            timeout: 10000,
            headers: {
                'Content-Type': 'application/json',
                'User-Agent': 'Homeostasis-VSCode-Extension'
            }
        });

        this.setupHttpClient();
    }

    private setupHttpClient(): void {
        this.httpClient.defaults.baseURL = this.configManager.getServerUrl();
        
        const apiKey = this.configManager.getApiKey();
        if (apiKey) {
            this.httpClient.defaults.headers.common['Authorization'] = `Bearer ${apiKey}`;
        }

        // Request interceptor for telemetry
        this.httpClient.interceptors.request.use(config => {
            this.telemetryService.sendEvent('api.request', {
                endpoint: config.url,
                method: config.method
            });
            return config;
        });

        // Response interceptor for error handling and telemetry
        this.httpClient.interceptors.response.use(
            response => {
                this.telemetryService.sendEvent('api.response', {
                    endpoint: response.config.url,
                    status: response.status,
                    responseTime: Date.now() - (response.config as any).startTime
                });
                return response;
            },
            error => {
                this.telemetryService.sendEvent('api.error', {
                    endpoint: error.config?.url,
                    error: error.message,
                    status: error.response?.status
                });
                return Promise.reject(error);
            }
        );
    }

    async initialize(): Promise<boolean> {
        try {
            const response = await this.httpClient.get('/health');
            this.isConnected = response.status === 200;
            
            if (this.isConnected) {
                vscode.window.showInformationMessage('Connected to Homeostasis healing server');
                this.telemetryService.sendEvent('service.connected');
            }
            
            return this.isConnected;
        } catch (error) {
            this.isConnected = false;
            vscode.window.showWarningMessage('Unable to connect to Homeostasis server. Real-time healing disabled.');
            this.telemetryService.sendEvent('service.connection_failed', {
                error: (error as Error).message
            });
            return false;
        }
    }

    async analyzeDocument(document: vscode.TextDocument): Promise<AnalysisResult | null> {
        if (!this.isConnected || !this.isLanguageSupported(document.languageId)) {
            return null;
        }

        const documentUri = document.uri.toString();
        
        // Check if analysis is already pending for this document
        if (this.pendingAnalysis.has(documentUri)) {
            return await this.pendingAnalysis.get(documentUri)!;
        }

        const analysisPromise = this.performAnalysis(document);
        this.pendingAnalysis.set(documentUri, analysisPromise);

        try {
            const result = await analysisPromise;
            return result;
        } finally {
            this.pendingAnalysis.delete(documentUri);
        }
    }

    private async performAnalysis(document: vscode.TextDocument): Promise<AnalysisResult> {
        const startTime = Date.now();
        
        try {
            const payload = {
                content: document.getText(),
                language: document.languageId,
                filename: document.fileName,
                workspace: vscode.workspace.getWorkspaceFolder(document.uri)?.uri.fsPath,
                options: {
                    confidenceThreshold: this.configManager.getConfidenceThreshold(),
                    includeWarnings: true,
                    maxSuggestions: 10
                }
            };

            const response = await this.httpClient.post('/api/analyze', payload);
            const data = response.data;

            // Convert API response to our internal format
            const result: AnalysisResult = {
                errors: data.errors?.map((error: any) => ({
                    message: error.message,
                    severity: error.severity,
                    range: new vscode.Range(
                        error.line - 1,
                        error.column,
                        error.endLine ? error.endLine - 1 : error.line - 1,
                        error.endColumn || error.column + error.length
                    ),
                    source: 'homeostasis',
                    code: error.code
                })) || [],
                suggestions: data.suggestions?.map((suggestion: any) => ({
                    id: suggestion.id,
                    title: suggestion.title,
                    description: suggestion.description,
                    confidence: suggestion.confidence,
                    range: new vscode.Range(
                        suggestion.line - 1,
                        suggestion.column,
                        suggestion.endLine ? suggestion.endLine - 1 : suggestion.line - 1,
                        suggestion.endColumn || suggestion.column + suggestion.length
                    ),
                    fix: suggestion.fix,
                    kind: suggestion.kind
                })) || [],
                metadata: {
                    analysisTime: Date.now() - startTime,
                    language: document.languageId,
                    rulesMatched: data.metadata?.rulesMatched || 0
                }
            };

            this.telemetryService.sendEvent('analysis.completed', {
                language: document.languageId,
                errorsFound: result.errors.length,
                suggestionsFound: result.suggestions.length,
                analysisTime: result.metadata.analysisTime,
                rulesMatched: result.metadata.rulesMatched
            });

            return result;
        } catch (error) {
            this.telemetryService.sendEvent('analysis.failed', {
                language: document.languageId,
                error: (error as Error).message
            });
            throw error;
        }
    }

    async healDocument(document: vscode.TextDocument): Promise<boolean> {
        const analysisResult = await this.analyzeDocument(document);
        if (!analysisResult || analysisResult.suggestions.length === 0) {
            vscode.window.showInformationMessage('No healing suggestions found for this document');
            return false;
        }

        const highConfidenceFixes = analysisResult.suggestions.filter(
            s => s.confidence >= this.configManager.getConfidenceThreshold()
        );

        if (highConfidenceFixes.length === 0) {
            vscode.window.showInformationMessage('No high-confidence fixes available');
            return false;
        }

        // Apply fixes in reverse order to maintain correct positions
        const editor = await vscode.window.showTextDocument(document);
        const workspaceEdit = new vscode.WorkspaceEdit();

        highConfidenceFixes
            .sort((a, b) => b.range.start.line - a.range.start.line)
            .forEach(suggestion => {
                workspaceEdit.replace(document.uri, suggestion.range, suggestion.fix);
            });

        const applied = await vscode.workspace.applyEdit(workspaceEdit);
        
        if (applied) {
            vscode.window.showInformationMessage(`Applied ${highConfidenceFixes.length} healing fixes`);
            this.telemetryService.sendEvent('fixes.applied', {
                language: document.languageId,
                fixesApplied: highConfidenceFixes.length
            });
        }

        return applied;
    }

    async healWorkspace(): Promise<void> {
        if (!vscode.workspace.workspaceFolders) {
            return;
        }

        const supportedFiles = await vscode.workspace.findFiles('**/*.{py,js,ts,java,go,rs,cs,php,rb,scala,ex,clj,swift,kt,dart}');
        const progressOptions: vscode.ProgressOptions = {
            location: vscode.ProgressLocation.Notification,
            title: 'Healing workspace...',
            cancellable: true
        };

        await vscode.window.withProgress(progressOptions, async (progress, token) => {
            let healedFiles = 0;
            const totalFiles = supportedFiles.length;

            for (let i = 0; i < totalFiles; i++) {
                if (token.isCancellationRequested) {
                    break;
                }

                const file = supportedFiles[i];
                progress.report({
                    increment: (100 / totalFiles),
                    message: `Processing ${file.fsPath}...`
                });

                try {
                    const document = await vscode.workspace.openTextDocument(file);
                    const healed = await this.healDocument(document);
                    if (healed) {
                        healedFiles++;
                    }
                } catch (error) {
                    console.error(`Failed to heal ${file.fsPath}:`, error);
                }
            }

            vscode.window.showInformationMessage(`Workspace healing completed. Healed ${healedFiles} files.`);
            this.telemetryService.sendEvent('workspace.healed', {
                totalFiles,
                healedFiles
            });
        });
    }

    private isLanguageSupported(languageId: string): boolean {
        return this.configManager.getEnabledLanguages().includes(languageId);
    }

    isServiceConnected(): boolean {
        return this.isConnected;
    }
}