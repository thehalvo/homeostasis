import * as vscode from 'vscode';
import { HealingService, DiagnosticInfo } from '../services/healingService';

export class DiagnosticsProvider {
    private diagnosticCollection?: vscode.DiagnosticCollection;
    private analysisCache: Map<string, { timestamp: number; diagnostics: vscode.Diagnostic[] }> = new Map();
    private readonly CACHE_TTL = 30000; // 30 seconds

    constructor(private healingService: HealingService) {
        // Listen to document changes to update diagnostics
        vscode.workspace.onDidChangeTextDocument(this.onDocumentChange, this);
        vscode.workspace.onDidSaveTextDocument(this.onDocumentSave, this);
        vscode.workspace.onDidCloseTextDocument(this.onDocumentClose, this);
    }

    setDiagnosticCollection(collection: vscode.DiagnosticCollection): void {
        this.diagnosticCollection = collection;
    }

    private async onDocumentChange(event: vscode.TextDocumentChangeEvent): Promise<void> {
        // Clear cache for changed document
        const uri = event.document.uri.toString();
        this.analysisCache.delete(uri);
        
        // Debounce updates - only update diagnostics after typing stops
        setTimeout(() => {
            this.updateDiagnostics(event.document);
        }, 1000);
    }

    private async onDocumentSave(document: vscode.TextDocument): Promise<void> {
        // Always update diagnostics on save
        await this.updateDiagnostics(document);
    }

    private onDocumentClose(document: vscode.TextDocument): void {
        // Clear diagnostics and cache for closed document
        const uri = document.uri.toString();
        this.analysisCache.delete(uri);
        
        if (this.diagnosticCollection) {
            this.diagnosticCollection.delete(document.uri);
        }
    }

    public async updateDiagnostics(document: vscode.TextDocument): Promise<void> {
        if (!this.diagnosticCollection || !this.healingService.isServiceConnected()) {
            return;
        }

        const uri = document.uri.toString();
        
        // Check cache first
        const cached = this.analysisCache.get(uri);
        if (cached && (Date.now() - cached.timestamp) < this.CACHE_TTL) {
            this.diagnosticCollection.set(document.uri, cached.diagnostics);
            return;
        }

        try {
            const analysisResult = await this.healingService.analyzeDocument(document);
            
            if (!analysisResult) {
                // Clear diagnostics if no analysis available
                this.diagnosticCollection.delete(document.uri);
                return;
            }

            const diagnostics = this.convertToDiagnostics(analysisResult.errors);
            
            // Cache the results
            this.analysisCache.set(uri, {
                timestamp: Date.now(),
                diagnostics
            });

            // Set diagnostics in VS Code
            this.diagnosticCollection.set(document.uri, diagnostics);

        } catch (error) {
            console.error('Failed to update diagnostics:', error);
            // Don't clear existing diagnostics on error - just log it
        }
    }

    private convertToDiagnostics(errors: DiagnosticInfo[]): vscode.Diagnostic[] {
        return errors.map(error => {
            const diagnostic = new vscode.Diagnostic(
                error.range,
                error.message,
                this.convertSeverity(error.severity)
            );

            diagnostic.source = error.source;
            diagnostic.code = error.code;

            // Add related information if available
            if (error.code) {
                diagnostic.relatedInformation = [
                    new vscode.DiagnosticRelatedInformation(
                        new vscode.Location(vscode.Uri.parse(''), error.range),
                        `Homeostasis rule: ${error.code}`
                    )
                ];
            }

            return diagnostic;
        });
    }

    private convertSeverity(severity: string): vscode.DiagnosticSeverity {
        switch (severity.toLowerCase()) {
            case 'error':
                return vscode.DiagnosticSeverity.Error;
            case 'warning':
                return vscode.DiagnosticSeverity.Warning;
            case 'info':
                return vscode.DiagnosticSeverity.Information;
            default:
                return vscode.DiagnosticSeverity.Hint;
        }
    }

    public async refreshAllDiagnostics(): Promise<void> {
        if (!this.diagnosticCollection) {
            return;
        }

        // Clear cache
        this.analysisCache.clear();

        // Update diagnostics for all open documents
        const openDocuments = vscode.workspace.textDocuments;
        const updatePromises = openDocuments.map(doc => this.updateDiagnostics(doc));
        
        await Promise.all(updatePromises);
    }

    public clearAllDiagnostics(): void {
        if (this.diagnosticCollection) {
            this.diagnosticCollection.clear();
        }
        this.analysisCache.clear();
    }

    dispose(): void {
        this.clearAllDiagnostics();
    }
}