import * as vscode from 'vscode';
import { HealingService, HealingSuggestion } from '../services/healingService';

export class InlineFixProvider implements vscode.CodeActionProvider {
    private static readonly providedCodeActionKinds = [
        vscode.CodeActionKind.QuickFix
    ];

    constructor(private healingService: HealingService) {}

    async provideCodeActions(
        document: vscode.TextDocument,
        range: vscode.Range | vscode.Selection,
        context: vscode.CodeActionContext,
        token: vscode.CancellationToken
    ): Promise<vscode.CodeAction[]> {
        
        // Only provide actions if the healing service is connected
        if (!this.healingService.isServiceConnected()) {
            return [];
        }

        try {
            const analysisResult = await this.healingService.analyzeDocument(document);
            if (!analysisResult || analysisResult.suggestions.length === 0) {
                return [];
            }

            const actions: vscode.CodeAction[] = [];

            // Filter suggestions that intersect with the current range
            const relevantSuggestions = analysisResult.suggestions.filter(suggestion =>
                suggestion.range.intersection(range) !== undefined
            );

            for (const suggestion of relevantSuggestions) {
                const action = this.createCodeAction(document, suggestion);
                if (action) {
                    actions.push(action);
                }
            }

            // Add a "Heal All Issues" action if there are multiple suggestions
            if (analysisResult.suggestions.length > 1) {
                const healAllAction = this.createHealAllAction(document, analysisResult.suggestions);
                if (healAllAction) {
                    actions.push(healAllAction);
                }
            }

            return actions;
        } catch (error) {
            console.error('Failed to provide code actions:', error);
            return [];
        }
    }

    private createCodeAction(document: vscode.TextDocument, suggestion: HealingSuggestion): vscode.CodeAction | null {
        const action = new vscode.CodeAction(
            `${suggestion.title} (${Math.round(suggestion.confidence * 100)}% confidence)`,
            vscode.CodeActionKind.QuickFix
        );

        action.edit = new vscode.WorkspaceEdit();
        action.edit.replace(document.uri, suggestion.range, suggestion.fix);
        
        // Set diagnostic information if available
        action.diagnostics = [];
        
        // Add detailed description
        action.tooltip = suggestion.description;
        
        // Set as preferred if high confidence
        action.isPreferred = suggestion.confidence > 0.8;

        return action;
    }

    private createHealAllAction(document: vscode.TextDocument, suggestions: HealingSuggestion[]): vscode.CodeAction | null {
        const highConfidenceSuggestions = suggestions.filter(s => s.confidence > 0.7);
        
        if (highConfidenceSuggestions.length === 0) {
            return null;
        }

        const action = new vscode.CodeAction(
            `Heal All Issues (${highConfidenceSuggestions.length} fixes)`,
            vscode.CodeActionKind.QuickFix
        );

        action.edit = new vscode.WorkspaceEdit();
        
        // Apply fixes in reverse order to maintain correct positions
        highConfidenceSuggestions
            .sort((a, b) => b.range.start.line - a.range.start.line)
            .forEach(suggestion => {
                action.edit!.replace(document.uri, suggestion.range, suggestion.fix);
            });

        action.tooltip = `Apply ${highConfidenceSuggestions.length} high-confidence healing fixes`;
        action.isPreferred = true;

        return action;
    }

    static get metadata(): vscode.CodeActionProviderMetadata {
        return {
            providedCodeActionKinds: InlineFixProvider.providedCodeActionKinds
        };
    }
}