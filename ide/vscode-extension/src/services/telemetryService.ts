import * as vscode from 'vscode';
import { ConfigurationManager } from './configurationManager';

export interface TelemetryEvent {
    event: string;
    properties?: Record<string, any>;
    timestamp: number;
    sessionId: string;
    userId?: string;
}

export class TelemetryService {
    private sessionId: string;
    private eventQueue: TelemetryEvent[] = [];
    private flushTimer?: NodeJS.Timeout;
    private readonly FLUSH_INTERVAL = 30000; // 30 seconds
    private readonly MAX_QUEUE_SIZE = 100;

    constructor(private configManager: ConfigurationManager) {
        this.sessionId = this.generateSessionId();
        this.startFlushTimer();
    }

    sendEvent(event: string, properties?: Record<string, any>): void {
        if (!this.configManager.isTelemetryEnabled()) {
            return;
        }

        const telemetryEvent: TelemetryEvent = {
            event,
            properties: {
                ...properties,
                vscodeVersion: vscode.version,
                extensionVersion: vscode.extensions.getExtension('homeostasis.homeostasis-healing')?.packageJSON.version,
                platform: process.platform,
                arch: process.arch
            },
            timestamp: Date.now(),
            sessionId: this.sessionId,
            userId: this.getUserId()
        };

        this.eventQueue.push(telemetryEvent);

        // Flush immediately if queue is full
        if (this.eventQueue.length >= this.MAX_QUEUE_SIZE) {
            this.flush();
        }
    }

    sendErrorEvent(error: Error, context?: Record<string, any>): void {
        this.sendEvent('error', {
            errorMessage: error.message,
            errorStack: error.stack,
            errorName: error.name,
            ...context
        });
    }

    sendPerformanceEvent(operation: string, duration: number, properties?: Record<string, any>): void {
        this.sendEvent('performance', {
            operation,
            duration,
            ...properties
        });
    }

    sendUsageEvent(feature: string, properties?: Record<string, any>): void {
        this.sendEvent('usage', {
            feature,
            ...properties
        });
    }

    private async flush(): Promise<void> {
        if (this.eventQueue.length === 0) {
            return;
        }

        const events = [...this.eventQueue];
        this.eventQueue = [];

        try {
            // In a real implementation, you would send this to your telemetry service
            // For now, we'll just log to console in development
            if (process.env.NODE_ENV === 'development') {
                console.log('Telemetry events:', events);
            }

            // Example of sending to a hypothetical telemetry endpoint
            // await this.sendToTelemetryService(events);
        } catch (error) {
            // If sending fails, add events back to queue (up to a limit)
            const remainingSpace = this.MAX_QUEUE_SIZE - this.eventQueue.length;
            if (remainingSpace > 0) {
                this.eventQueue.unshift(...events.slice(-remainingSpace));
            }
            
            console.error('Failed to send telemetry:', error);
        }
    }

    private startFlushTimer(): void {
        this.flushTimer = setInterval(() => {
            this.flush();
        }, this.FLUSH_INTERVAL);
    }

    private generateSessionId(): string {
        return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    private getUserId(): string | undefined {
        // In a real implementation, you might want to generate a stable anonymous user ID
        // For privacy, we don't collect actual user identification
        return undefined;
    }

    dispose(): void {
        if (this.flushTimer) {
            clearInterval(this.flushTimer);
        }
        this.flush(); // Final flush before disposal
    }
}