import * as vscode from 'vscode';
import { HealingService } from '../services/healingService';

export class CodeLensProvider implements vscode.CodeLensProvider {
    private codeLenses: vscode.CodeLens[] = [];
    private regex: RegExp;
    private _onDidChangeCodeLenses: vscode.EventEmitter<void> = new vscode.EventEmitter<void>();
    public readonly onDidChangeCodeLenses: vscode.Event<void> = this._onDidChangeCodeLenses.event;

    constructor(private healingService: HealingService) {
        // Pattern to match common error-prone constructs
        this.regex = /\b(try|catch|finally|throw|assert|TODO|FIXME|BUG|HACK)\b/g;
        
        // Listen to document changes to update code lenses
        vscode.workspace.onDidChangeTextDocument(() => {
            this._onDidChangeCodeLenses.fire();
        });
    }

    public async provideCodeLenses(
        document: vscode.TextDocument,
        token: vscode.CancellationToken
    ): Promise<vscode.CodeLens[]> {
        
        if (!this.healingService.isServiceConnected()) {
            return [];
        }

        this.codeLenses = [];

        try {
            // Get analysis results for the document
            const analysisResult = await this.healingService.analyzeDocument(document);
            
            // Add code lenses for potential issues and suggestions
            if (analysisResult) {
                await this.addAnalysisCodeLenses(document, analysisResult);
            }

            // Add preventive code lenses for error-prone patterns
            await this.addPreventiveCodeLenses(document);

            return this.codeLenses;
        } catch (error) {
            console.error('Failed to provide code lenses:', error);
            return [];
        }
    }

    private async addAnalysisCodeLenses(document: vscode.TextDocument, analysisResult: any): Promise<void> {
        // Add code lens for each suggestion
        analysisResult.suggestions?.forEach((suggestion: any) => {
            const range = new vscode.Range(
                suggestion.range.start.line,
                0,
                suggestion.range.start.line,
                0
            );

            const confidenceText = `${Math.round(suggestion.confidence * 100)}%`;
            const codeLens = new vscode.CodeLens(range);
            
            codeLens.command = {
                title: `🔧 ${suggestion.title} (${confidenceText} confidence)`,
                command: 'homeostasis.applySuggestion',
                arguments: [document.uri, suggestion]
            };

            this.codeLenses.push(codeLens);
        });

        // Add summary code lens at the top of the file if there are multiple issues
        if (analysisResult.suggestions?.length > 1) {
            const topRange = new vscode.Range(0, 0, 0, 0);
            const summaryLens = new vscode.CodeLens(topRange);
            
            summaryLens.command = {
                title: `🏥 Heal ${analysisResult.suggestions.length} issues in this file`,
                command: 'homeostasis.healFile',
                arguments: []
            };

            this.codeLenses.unshift(summaryLens); // Add at the beginning
        }
    }

    private async addPreventiveCodeLenses(document: vscode.TextDocument): Promise<void> {
        const text = document.getText();
        let match;

        while ((match = this.regex.exec(text))) {
            const line = document.lineAt(document.positionAt(match.index).line);
            const indexOf = line.text.indexOf(match[0]);
            const position = new vscode.Position(line.lineNumber, indexOf);
            const range = document.getWordRangeAtPosition(position, new RegExp(this.regex));

            if (range) {
                const codeLens = this.createPreventiveCodeLens(range, match[0], line.text);
                if (codeLens) {
                    this.codeLenses.push(codeLens);
                }
            }
        }
    }

    private createPreventiveCodeLens(range: vscode.Range, keyword: string, lineText: string): vscode.CodeLens | null {
        const codeLens = new vscode.CodeLens(range);
        
        switch (keyword.toLowerCase()) {
            case 'try':
                codeLens.command = {
                    title: '💡 Improve error handling',
                    command: 'homeostasis.suggestErrorHandling',
                    arguments: [range]
                };
                break;
            case 'todo':
            case 'fixme':
            case 'bug':
            case 'hack':
                codeLens.command = {
                    title: '🎯 Track and resolve',
                    command: 'homeostasis.trackTechnicalDebt',
                    arguments: [range, keyword, lineText]
                };
                break;
            case 'assert':
                codeLens.command = {
                    title: '🛡️ Strengthen assertion',
                    command: 'homeostasis.improveAssertion',
                    arguments: [range]
                };
                break;
            default:
                return null;
        }

        return codeLens;
    }

    public resolveCodeLens(codeLens: vscode.CodeLens, token: vscode.CancellationToken): vscode.CodeLens {
        // Code lens resolution if needed for dynamic content
        return codeLens;
    }
}