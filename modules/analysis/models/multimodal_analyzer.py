"""
Multimodal analysis module for combining logs, metrics, and code data.

This module implements models that can analyze multiple data modalities
simultaneously to provide more comprehensive error understanding.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging
from dataclasses import dataclass
from datetime import datetime
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MultimodalErrorData:
    """Container for multimodal error data."""
    # Text modality
    error_message: str
    traceback: List[str]
    code_context: Optional[str] = None
    
    # Metrics modality
    cpu_usage: Optional[List[float]] = None
    memory_usage: Optional[List[float]] = None
    response_times: Optional[List[float]] = None
    error_rate: Optional[float] = None
    
    # Log modality
    preceding_logs: Optional[List[str]] = None
    following_logs: Optional[List[str]] = None
    log_patterns: Optional[Dict[str, int]] = None
    
    # Temporal modality
    timestamp: Optional[datetime] = None
    time_of_day: Optional[float] = None
    day_of_week: Optional[int] = None
    is_peak_hours: Optional[bool] = None
    
    # System modality
    service_name: Optional[str] = None
    environment: Optional[str] = None
    version: Optional[str] = None
    dependencies: Optional[List[str]] = None


class MetricsEncoder(nn.Module):
    """Encode system metrics into embeddings."""
    
    def __init__(self, output_dim: int = 128):
        super().__init__()
        
        # Time series encoders for different metrics
        self.cpu_encoder = nn.LSTM(1, 64, batch_first=True)
        self.memory_encoder = nn.LSTM(1, 64, batch_first=True)
        self.response_encoder = nn.LSTM(1, 64, batch_first=True)
        
        # Aggregation layers
        self.metric_fusion = nn.Sequential(
            nn.Linear(192, 256),  # 64 * 3 metrics
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_dim)
        )
        
        # Static metrics encoder
        self.static_encoder = nn.Sequential(
            nn.Linear(10, 64),  # Placeholder for static metrics
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, metrics_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode metrics data.
        
        Args:
            metrics_data: Dictionary containing metric tensors
            
        Returns:
            Encoded metrics representation
        """
        encoded_metrics = []
        
        # Encode CPU usage
        if 'cpu_usage' in metrics_data:
            cpu_tensor = metrics_data['cpu_usage'].unsqueeze(-1)  # Add feature dim
            _, (cpu_hidden, _) = self.cpu_encoder(cpu_tensor)
            encoded_metrics.append(cpu_hidden.squeeze(0))
        else:
            encoded_metrics.append(torch.zeros(metrics_data['batch_size'], 64))
        
        # Encode memory usage
        if 'memory_usage' in metrics_data:
            mem_tensor = metrics_data['memory_usage'].unsqueeze(-1)
            _, (mem_hidden, _) = self.memory_encoder(mem_tensor)
            encoded_metrics.append(mem_hidden.squeeze(0))
        else:
            encoded_metrics.append(torch.zeros(metrics_data['batch_size'], 64))
        
        # Encode response times
        if 'response_times' in metrics_data:
            resp_tensor = metrics_data['response_times'].unsqueeze(-1)
            _, (resp_hidden, _) = self.response_encoder(resp_tensor)
            encoded_metrics.append(resp_hidden.squeeze(0))
        else:
            encoded_metrics.append(torch.zeros(metrics_data['batch_size'], 64))
        
        # Concatenate and fuse
        combined = torch.cat(encoded_metrics, dim=-1)
        fused = self.metric_fusion(combined)
        
        return fused


class LogPatternEncoder(nn.Module):
    """Encode log patterns and sequences."""
    
    def __init__(self, vocab_size: int = 10000, output_dim: int = 128):
        super().__init__()
        
        # Log embedding
        self.log_embedding = nn.Embedding(vocab_size, 256)
        
        # Sequential encoder for log sequences
        self.log_encoder = nn.LSTM(256, 128, num_layers=2, 
                                  bidirectional=True, batch_first=True)
        
        # Pattern frequency encoder
        self.pattern_encoder = nn.Sequential(
            nn.Linear(100, 128),  # Assuming 100 pattern types
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        
        # Output projection
        self.output_proj = nn.Linear(256, output_dim)  # Bidirectional LSTM
    
    def forward(self, log_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode log data.
        
        Args:
            log_data: Dictionary containing log tensors
            
        Returns:
            Encoded log representation
        """
        if 'log_sequences' in log_data:
            # Embed log tokens
            log_embeds = self.log_embedding(log_data['log_sequences'])
            
            # Encode sequences
            log_output, (hidden, _) = self.log_encoder(log_embeds)
            
            # Use last hidden state
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=-1)
            log_encoding = self.output_proj(hidden)
        else:
            batch_size = log_data.get('batch_size', 1)
            log_encoding = torch.zeros(batch_size, 128)
        
        # Encode pattern frequencies if available
        if 'pattern_frequencies' in log_data:
            pattern_encoding = self.pattern_encoder(log_data['pattern_frequencies'])
            log_encoding = log_encoding + pattern_encoding
        
        return log_encoding


class TemporalEncoder(nn.Module):
    """Encode temporal features."""
    
    def __init__(self, output_dim: int = 64):
        super().__init__()
        
        # Cyclical encoding for time features
        self.time_encoder = nn.Sequential(
            nn.Linear(8, 32),  # hour_sin, hour_cos, day_sin, day_cos, etc.
            nn.ReLU(),
            nn.Linear(32, output_dim // 2)
        )
        
        # Trend encoder
        self.trend_encoder = nn.Sequential(
            nn.Linear(4, 32),  # recent trend features
            nn.ReLU(),
            nn.Linear(32, output_dim // 2)
        )
    
    def forward(self, temporal_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode temporal features.
        
        Args:
            temporal_data: Dictionary containing temporal features
            
        Returns:
            Encoded temporal representation
        """
        batch_size = temporal_data.get('batch_size', 1)
        
        # Encode cyclical time features
        if 'time_features' in temporal_data:
            time_encoding = self.time_encoder(temporal_data['time_features'])
        else:
            time_encoding = torch.zeros(batch_size, 32)
        
        # Encode trend features
        if 'trend_features' in temporal_data:
            trend_encoding = self.trend_encoder(temporal_data['trend_features'])
        else:
            trend_encoding = torch.zeros(batch_size, 32)
        
        # Combine
        temporal_encoding = torch.cat([time_encoding, trend_encoding], dim=-1)
        
        return temporal_encoding


class CrossModalAttention(nn.Module):
    """Cross-modal attention mechanism."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.scale = dim ** -0.5
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, 
                value: torch.Tensor) -> torch.Tensor:
        """
        Apply cross-modal attention.
        
        Args:
            query: Query tensor from one modality
            key: Key tensor from another modality
            value: Value tensor from another modality
            
        Returns:
            Attended features
        """
        Q = self.query_proj(query)
        K = self.key_proj(key)
        V = self.value_proj(value)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention
        attended = torch.matmul(attention_weights, V)
        
        return attended, attention_weights


class MultimodalFusionNetwork(nn.Module):
    """Network for fusing multiple modalities."""
    
    def __init__(self, hidden_dim: int = 256, num_classes: int = 10):
        super().__init__()
        
        # Text encoder
        self.text_encoder = AutoModel.from_pretrained('microsoft/codebert-base')
        self.text_proj = nn.Linear(768, hidden_dim)
        
        # Modality-specific encoders
        self.metrics_encoder = MetricsEncoder(output_dim=hidden_dim)
        self.log_encoder = LogPatternEncoder(output_dim=hidden_dim)
        self.temporal_encoder = TemporalEncoder(output_dim=hidden_dim // 2)
        
        # Cross-modal attention
        self.text_to_metrics = CrossModalAttention(hidden_dim)
        self.metrics_to_logs = CrossModalAttention(hidden_dim)
        self.logs_to_text = CrossModalAttention(hidden_dim)
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3 + hidden_dim // 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Auxiliary heads for multi-task learning
        self.severity_head = nn.Linear(hidden_dim, 5)  # 5 severity levels
        self.category_head = nn.Linear(hidden_dim, 8)  # 8 error categories
        self.resolution_head = nn.Linear(hidden_dim, 3)  # 3 resolution types
    
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through multimodal network.
        
        Args:
            inputs: Dictionary containing all modality inputs
            
        Returns:
            Dictionary with predictions and features
        """
        # Encode text
        text_outputs = self.text_encoder(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )
        text_features = self.text_proj(text_outputs.pooler_output)
        
        # Encode other modalities
        metrics_features = self.metrics_encoder(inputs.get('metrics', {}))
        log_features = self.log_encoder(inputs.get('logs', {}))
        temporal_features = self.temporal_encoder(inputs.get('temporal', {}))
        
        # Apply cross-modal attention
        text_attended, _ = self.text_to_metrics(
            text_features.unsqueeze(1),
            metrics_features.unsqueeze(1),
            metrics_features.unsqueeze(1)
        )
        text_attended = text_attended.squeeze(1)
        
        metrics_attended, _ = self.metrics_to_logs(
            metrics_features.unsqueeze(1),
            log_features.unsqueeze(1),
            log_features.unsqueeze(1)
        )
        metrics_attended = metrics_attended.squeeze(1)
        
        logs_attended, _ = self.logs_to_text(
            log_features.unsqueeze(1),
            text_features.unsqueeze(1),
            text_features.unsqueeze(1)
        )
        logs_attended = logs_attended.squeeze(1)
        
        # Concatenate all features
        combined_features = torch.cat([
            text_attended,
            metrics_attended,
            logs_attended,
            temporal_features
        ], dim=-1)
        
        # Fuse features
        fused_features = self.fusion(combined_features)
        
        # Generate predictions
        main_logits = self.classifier(fused_features)
        severity_logits = self.severity_head(fused_features)
        category_logits = self.category_head(fused_features)
        resolution_logits = self.resolution_head(fused_features)
        
        return {
            'logits': main_logits,
            'severity_logits': severity_logits,
            'category_logits': category_logits,
            'resolution_logits': resolution_logits,
            'features': fused_features,
            'modality_features': {
                'text': text_features,
                'metrics': metrics_features,
                'logs': log_features,
                'temporal': temporal_features
            }
        }


class MultimodalErrorAnalyzer:
    """High-level interface for multimodal error analysis."""
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize the multimodal analyzer."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
        self.model = MultimodalFusionNetwork().to(self.device)
        self.scaler = StandardScaler()
        
        if model_path and Path(model_path).exists():
            self.load(model_path)
    
    def preprocess_multimodal_data(self, error_data: MultimodalErrorData) -> Dict[str, Any]:
        """
        Preprocess multimodal error data for model input.
        
        Args:
            error_data: Multimodal error data
            
        Returns:
            Preprocessed inputs
        """
        inputs = {'batch_size': 1}
        
        # Process text
        text = f"{error_data.error_message} {' '.join(error_data.traceback[:3])}"
        if error_data.code_context:
            text += f" Code: {error_data.code_context}"
        
        text_inputs = self.tokenizer(
            text,
            return_tensors='pt',
            max_length=512,
            truncation=True,
            padding=True
        )
        inputs.update(text_inputs)
        
        # Process metrics
        metrics = {}
        if error_data.cpu_usage:
            cpu_tensor = torch.tensor(error_data.cpu_usage[-100:], dtype=torch.float32)
            metrics['cpu_usage'] = cpu_tensor.unsqueeze(0)
        
        if error_data.memory_usage:
            mem_tensor = torch.tensor(error_data.memory_usage[-100:], dtype=torch.float32)
            metrics['memory_usage'] = mem_tensor.unsqueeze(0)
        
        if error_data.response_times:
            resp_tensor = torch.tensor(error_data.response_times[-100:], dtype=torch.float32)
            metrics['response_times'] = resp_tensor.unsqueeze(0)
        
        inputs['metrics'] = metrics
        
        # Process logs (simplified - in practice would use proper tokenization)
        log_data = {}
        if error_data.log_patterns:
            pattern_vector = torch.zeros(100)
            for i, (pattern, count) in enumerate(error_data.log_patterns.items()):
                if i < 100:
                    pattern_vector[i] = count
            log_data['pattern_frequencies'] = pattern_vector.unsqueeze(0)
        
        inputs['logs'] = log_data
        
        # Process temporal features
        temporal = {}
        if error_data.timestamp:
            hour = error_data.timestamp.hour
            day = error_data.timestamp.weekday()
            
            # Cyclical encoding
            hour_sin = np.sin(2 * np.pi * hour / 24)
            hour_cos = np.cos(2 * np.pi * hour / 24)
            day_sin = np.sin(2 * np.pi * day / 7)
            day_cos = np.cos(2 * np.pi * day / 7)
            
            time_features = torch.tensor([
                hour_sin, hour_cos, day_sin, day_cos,
                float(error_data.is_peak_hours or 0),
                0, 0, 0  # Placeholder for additional features
            ], dtype=torch.float32)
            
            temporal['time_features'] = time_features.unsqueeze(0)
        
        inputs['temporal'] = temporal
        
        return inputs
    
    def analyze(self, error_data: MultimodalErrorData) -> Dict[str, Any]:
        """
        Analyze error using multimodal data.
        
        Args:
            error_data: Multimodal error data
            
        Returns:
            Analysis results
        """
        # Preprocess data
        inputs = self.preprocess_multimodal_data(error_data)
        
        # Move to device
        for key in ['input_ids', 'attention_mask']:
            if key in inputs:
                inputs[key] = inputs[key].to(self.device)
        
        for modality in ['metrics', 'logs', 'temporal']:
            if modality in inputs:
                for tensor_key, tensor in inputs[modality].items():
                    if isinstance(tensor, torch.Tensor):
                        inputs[modality][tensor_key] = tensor.to(self.device)
        
        # Run model
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(inputs)
        
        # Process outputs
        results = self._process_outputs(outputs, error_data)
        
        # Add visualizations
        results['visualizations'] = self._create_visualizations(error_data, outputs)
        
        return results
    
    def _process_outputs(self, outputs: Dict[str, torch.Tensor], 
                        error_data: MultimodalErrorData) -> Dict[str, Any]:
        """Process model outputs into interpretable results."""
        # Main classification
        main_probs = F.softmax(outputs['logits'], dim=-1)[0].cpu().numpy()
        main_pred = np.argmax(main_probs)
        
        # Severity prediction
        severity_probs = F.softmax(outputs['severity_logits'], dim=-1)[0].cpu().numpy()
        severity_labels = ['critical', 'high', 'medium', 'low', 'info']
        severity_pred = severity_labels[np.argmax(severity_probs)]
        
        # Category prediction
        category_probs = F.softmax(outputs['category_logits'], dim=-1)[0].cpu().numpy()
        category_labels = ['system', 'network', 'database', 'application', 
                          'security', 'performance', 'configuration', 'unknown']
        category_pred = category_labels[np.argmax(category_probs)]
        
        # Resolution prediction
        resolution_probs = F.softmax(outputs['resolution_logits'], dim=-1)[0].cpu().numpy()
        resolution_labels = ['auto_fix', 'manual_intervention', 'escalate']
        resolution_pred = resolution_labels[np.argmax(resolution_probs)]
        
        # Feature importance (simplified)
        modality_features = outputs['modality_features']
        feature_norms = {
            modality: features.norm().item()
            for modality, features in modality_features.items()
            if features is not None
        }
        total_norm = sum(feature_norms.values())
        feature_importance = {
            modality: norm / total_norm if total_norm > 0 else 0
            for modality, norm in feature_norms.items()
        }
        
        return {
            'error_class': main_pred,
            'confidence': float(main_probs[main_pred]),
            'severity': severity_pred,
            'severity_confidence': float(severity_probs[np.argmax(severity_probs)]),
            'category': category_pred,
            'category_confidence': float(category_probs[np.argmax(category_probs)]),
            'resolution_type': resolution_pred,
            'resolution_confidence': float(resolution_probs[np.argmax(resolution_probs)]),
            'feature_importance': feature_importance,
            'insights': self._generate_insights(error_data, feature_importance)
        }
    
    def _generate_insights(self, error_data: MultimodalErrorData, 
                          feature_importance: Dict[str, float]) -> List[str]:
        """Generate insights based on multimodal analysis."""
        insights = []
        
        # Check metrics patterns
        if error_data.cpu_usage and max(error_data.cpu_usage) > 90:
            insights.append("High CPU usage detected before error")
        
        if error_data.memory_usage and max(error_data.memory_usage) > 85:
            insights.append("Memory pressure detected")
        
        if error_data.response_times:
            avg_response = np.mean(error_data.response_times)
            if avg_response > 1000:  # 1 second
                insights.append("Slow response times indicate performance issues")
        
        # Check temporal patterns
        if error_data.is_peak_hours:
            insights.append("Error occurred during peak hours")
        
        # Feature importance insights
        dominant_modality = max(feature_importance.items(), key=lambda x: x[1])
        if dominant_modality[1] > 0.5:
            insights.append(f"{dominant_modality[0].capitalize()} data was most informative")
        
        return insights
    
    def _create_visualizations(self, error_data: MultimodalErrorData, 
                             outputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Create visualizations for multimodal analysis."""
        visualizations = {}
        
        # Metrics visualization
        if error_data.cpu_usage or error_data.memory_usage:
            fig, axes = plt.subplots(2, 1, figsize=(10, 6))
            
            if error_data.cpu_usage:
                axes[0].plot(error_data.cpu_usage[-100:])
                axes[0].set_title('CPU Usage')
                axes[0].set_ylabel('Usage %')
            
            if error_data.memory_usage:
                axes[1].plot(error_data.memory_usage[-100:])
                axes[1].set_title('Memory Usage')
                axes[1].set_ylabel('Usage %')
            
            plt.tight_layout()
            visualizations['metrics_plot'] = fig
        
        # Feature importance visualization
        modality_features = outputs['modality_features']
        feature_names = list(modality_features.keys())
        feature_values = [
            modality_features[name].norm().item() if modality_features[name] is not None else 0
            for name in feature_names
        ]
        
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(feature_names, feature_values)
        ax.set_title('Modality Feature Importance')
        ax.set_ylabel('Feature Norm')
        visualizations['feature_importance_plot'] = fig
        
        return visualizations
    
    def explain_prediction(self, error_data: MultimodalErrorData, 
                          results: Dict[str, Any]) -> str:
        """Generate human-readable explanation of the prediction."""
        explanation_parts = [
            f"Error Classification: {results['error_class']} "
            f"(confidence: {results['confidence']:.2f})",
            f"Severity: {results['severity']}",
            f"Category: {results['category']}",
            f"Recommended Resolution: {results['resolution_type']}"
        ]
        
        # Add modality contributions
        important_modalities = sorted(
            results['feature_importance'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:2]
        
        explanation_parts.append(
            f"Analysis based primarily on {important_modalities[0][0]} "
            f"({important_modalities[0][1]:.1%}) and {important_modalities[1][0]} "
            f"({important_modalities[1][1]:.1%}) data."
        )
        
        # Add insights
        if results['insights']:
            explanation_parts.append("Key insights:")
            for insight in results['insights']:
                explanation_parts.append(f"- {insight}")
        
        return "\n".join(explanation_parts)


def demonstrate_multimodal_analysis():
    """Demonstrate multimodal error analysis."""
    analyzer = MultimodalErrorAnalyzer()
    
    # Create sample multimodal error data
    error_data = MultimodalErrorData(
        error_message="ConnectionError: Unable to connect to database",
        traceback=[
            "Traceback (most recent call last):",
            "  File 'app.py', line 42, in connect",
            "    conn = psycopg2.connect(dsn)",
            "ConnectionError: Unable to connect to database"
        ],
        code_context="conn = psycopg2.connect(dsn)",
        cpu_usage=[45, 48, 52, 78, 92, 95, 93, 85, 70, 65],
        memory_usage=[60, 62, 65, 68, 72, 78, 82, 85, 83, 80],
        response_times=[200, 250, 300, 500, 800, 1200, 1500, 2000, 1800, 1600],
        error_rate=0.15,
        log_patterns={
            'connection_timeout': 5,
            'retry_attempt': 3,
            'database_error': 8
        },
        timestamp=datetime.now(),
        time_of_day=14.5,
        day_of_week=2,
        is_peak_hours=True,
        service_name='api_service',
        environment='production'
    )
    
    # Analyze error
    logger.info("Analyzing multimodal error data...")
    results = analyzer.analyze(error_data)
    
    # Display results
    logger.info("\nAnalysis Results:")
    logger.info(f"Error Classification: {results['error_class']}")
    logger.info(f"Confidence: {results['confidence']:.2f}")
    logger.info(f"Severity: {results['severity']} ({results['severity_confidence']:.2f})")
    logger.info(f"Category: {results['category']} ({results['category_confidence']:.2f})")
    logger.info(f"Resolution: {results['resolution_type']} ({results['resolution_confidence']:.2f})")
    
    logger.info("\nFeature Importance:")
    for modality, importance in results['feature_importance'].items():
        logger.info(f"  {modality}: {importance:.2%}")
    
    logger.info("\nInsights:")
    for insight in results['insights']:
        logger.info(f"  - {insight}")
    
    # Generate explanation
    explanation = analyzer.explain_prediction(error_data, results)
    logger.info(f"\nExplanation:\n{explanation}")


if __name__ == "__main__":
    demonstrate_multimodal_analysis()