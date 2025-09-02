"""
Deep learning-based error classification model for Homeostasis.

This module implements neural network models for error pattern recognition,
providing more sophisticated pattern learning capabilities compared to
traditional ML approaches.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import logging
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ErrorDataset(Dataset):
    """PyTorch dataset for error data."""
    
    def __init__(self, errors: List[Dict[str, Any]], labels: Optional[List[str]] = None, 
                 tokenizer=None, max_length: int = 512):
        """
        Initialize the error dataset.
        
        Args:
            errors: List of error data dictionaries
            labels: List of error category labels (optional for inference)
            tokenizer: Tokenizer for text processing
            max_length: Maximum sequence length
        """
        self.errors = errors
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_encoder = LabelEncoder()
        
        if labels:
            self.label_encoder.fit(labels)
            self.encoded_labels = self.label_encoder.transform(labels)
        else:
            self.encoded_labels = None
    
    def __len__(self):
        return len(self.errors)
    
    def __getitem__(self, idx):
        error = self.errors[idx]
        
        # Extract text features
        text = self._extract_text_features(error)
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }
        
        if self.encoded_labels is not None:
            item['labels'] = torch.tensor(self.encoded_labels[idx], dtype=torch.long)
        
        return item
    
    def _extract_text_features(self, error: Dict[str, Any]) -> str:
        """Extract and combine text features from error data."""
        parts = []
        
        # Add exception type
        if 'exception_type' in error:
            parts.append(f"Exception: {error['exception_type']}")
        
        # Add error message
        if 'message' in error:
            parts.append(f"Message: {error['message']}")
        
        # Add traceback information
        if 'traceback' in error:
            tb = error['traceback']
            if isinstance(tb, list):
                tb_text = ' '.join(tb[-3:])  # Last 3 lines of traceback
            else:
                tb_text = str(tb)
            parts.append(f"Traceback: {tb_text}")
        
        # Add code context if available
        if 'error_details' in error and 'detailed_frames' in error['error_details']:
            frames = error['error_details']['detailed_frames']
            if frames:
                last_frame = frames[-1]
                if 'code' in last_frame:
                    parts.append(f"Code: {last_frame['code']}")
                if 'function' in last_frame:
                    parts.append(f"Function: {last_frame['function']}")
        
        return ' '.join(parts)


class AttentionLayer(nn.Module):
    """Self-attention layer for focusing on important error patterns."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)
        
    def forward(self, hidden_states):
        # Calculate attention weights
        attention_weights = torch.softmax(self.attention(hidden_states), dim=1)
        # Apply attention
        weighted_output = torch.sum(hidden_states * attention_weights, dim=1)
        return weighted_output, attention_weights


class DeepErrorClassifier(nn.Module):
    """Deep neural network for error classification."""
    
    def __init__(self, num_classes: int, hidden_dim: int = 768, 
                 dropout_rate: float = 0.3):
        """
        Initialize the deep error classifier.
        
        Args:
            num_classes: Number of error categories
            hidden_dim: Hidden dimension size
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()
        
        # Transformer encoder (using pre-trained model)
        self.encoder = AutoModel.from_pretrained('microsoft/codebert-base')
        
        # Freeze lower layers to prevent overfitting
        for param in self.encoder.embeddings.parameters():
            param.requires_grad = False
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.LayerNorm(hidden_dim)
        )
        
        # Attention mechanism
        self.attention = AttentionLayer(hidden_dim)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Pattern-specific heads for multi-task learning
        self.syntax_head = nn.Linear(hidden_dim, 2)  # Syntax error detection
        self.runtime_head = nn.Linear(hidden_dim, 2)  # Runtime error detection
        self.logic_head = nn.Linear(hidden_dim, 2)   # Logic error detection
        
    def forward(self, input_ids, attention_mask):
        # Encode input
        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Extract features
        hidden_states = encoder_output.last_hidden_state
        features = self.feature_extractor(hidden_states)
        
        # Apply attention
        attended_features, attention_weights = self.attention(features)
        
        # Main classification
        logits = self.classifier(attended_features)
        
        # Auxiliary classifications
        syntax_logits = self.syntax_head(attended_features)
        runtime_logits = self.runtime_head(attended_features)
        logic_logits = self.logic_head(attended_features)
        
        return {
            'logits': logits,
            'syntax_logits': syntax_logits,
            'runtime_logits': runtime_logits,
            'logic_logits': logic_logits,
            'attention_weights': attention_weights,
            'features': attended_features
        }


class HierarchicalErrorClassifier(nn.Module):
    """Hierarchical neural network for multi-level error classification."""
    
    def __init__(self, num_coarse_classes: int, num_fine_classes: Dict[int, int],
                 hidden_dim: int = 768, dropout_rate: float = 0.3):
        """
        Initialize hierarchical classifier.
        
        Args:
            num_coarse_classes: Number of coarse-grained categories
            num_fine_classes: Dict mapping coarse class to number of fine classes
            hidden_dim: Hidden dimension size
            dropout_rate: Dropout rate
        """
        super().__init__()
        
        self.encoder = AutoModel.from_pretrained('microsoft/codebert-base')
        
        # Coarse-grained classifier
        self.coarse_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_coarse_classes)
        )
        
        # Fine-grained classifiers for each coarse category
        self.fine_classifiers = nn.ModuleDict()
        for coarse_class, num_fine in num_fine_classes.items():
            self.fine_classifiers[str(coarse_class)] = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim // 4, num_fine)
            )
        
        # Global pooling
        self.pooler = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
    def forward(self, input_ids, attention_mask):
        # Encode input
        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Pool the output
        pooled_output = self.pooler(encoder_output.pooler_output)
        
        # Coarse classification
        coarse_logits = self.coarse_classifier(pooled_output)
        coarse_probs = F.softmax(coarse_logits, dim=-1)
        
        # Fine classification for each coarse category
        fine_logits = {}
        for coarse_class, classifier in self.fine_classifiers.items():
            fine_logits[coarse_class] = classifier(pooled_output)
        
        return {
            'coarse_logits': coarse_logits,
            'coarse_probs': coarse_probs,
            'fine_logits': fine_logits,
            'features': pooled_output
        }


class ErrorPatternRecognizer:
    """High-level interface for deep learning-based error pattern recognition."""
    
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize the error pattern recognizer.
        
        Args:
            model_path: Path to saved model
            device: Device to run model on (cuda/cpu)
        """
        self.model_path = model_path
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
        self.model = None
        self.label_encoder = None
        self.error_patterns = self._initialize_error_patterns()
        
    def _initialize_error_patterns(self) -> Dict[str, List[str]]:
        """Initialize known error patterns for pattern matching."""
        return {
            'syntax_errors': [
                r'SyntaxError:', r'IndentationError:', r'TabError:',
                r'unexpected EOF', r'invalid syntax', r'unexpected indent'
            ],
            'runtime_errors': [
                r'RuntimeError:', r'RecursionError:', r'StopIteration:',
                r'GeneratorExit:', r'SystemError:', r'MemoryError:'
            ],
            'type_errors': [
                r'TypeError:', r'AttributeError:', r'ValueError:',
                r'object is not', r'has no attribute', r'invalid literal'
            ],
            'index_errors': [
                r'IndexError:', r'KeyError:', r'out of range',
                r'list index', r'dictionary key', r'not found in'
            ],
            'io_errors': [
                r'IOError:', r'FileNotFoundError:', r'PermissionError:',
                r'OSError:', r'No such file', r'Permission denied'
            ],
            'import_errors': [
                r'ImportError:', r'ModuleNotFoundError:', r'No module named',
                r'cannot import name', r'circular import', r'failed to import'
            ],
            'network_errors': [
                r'ConnectionError:', r'TimeoutError:', r'URLError:',
                r'Connection refused', r'timed out', r'Network is unreachable'
            ],
            'database_errors': [
                r'DatabaseError:', r'IntegrityError:', r'OperationalError:',
                r'SQL', r'constraint violation', r'database is locked'
            ]
        }
    
    def extract_pattern_features(self, error_text: str) -> np.ndarray:
        """Extract pattern-based features from error text."""
        features = []
        
        for category, patterns in self.error_patterns.items():
            # Check if any pattern matches
            match_count = sum(1 for pattern in patterns 
                            if re.search(pattern, error_text, re.IGNORECASE))
            features.append(match_count)
        
        # Additional features
        features.extend([
            len(error_text.split()),  # Word count
            error_text.count('\n'),   # Line count
            error_text.count(':'),    # Colon count (common in errors)
            error_text.count('"'),    # Quote count
            error_text.count("'"),    # Single quote count
            1 if 'line' in error_text.lower() else 0,  # Contains line reference
            1 if 'file' in error_text.lower() else 0,  # Contains file reference
            1 if 'function' in error_text.lower() else 0,  # Contains function reference
        ])
        
        return np.array(features, dtype=np.float32)
    
    def train(self, errors: List[Dict[str, Any]], labels: List[str], 
              epochs: int = 10, batch_size: int = 16, learning_rate: float = 2e-5) -> Dict[str, Any]:
        """
        Train the deep learning model.
        
        Args:
            errors: List of error data
            labels: List of labels
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            
        Returns:
            Training results
        """
        # Initialize label encoder
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(labels)
        num_classes = len(self.label_encoder.classes_)
        
        # Create dataset and dataloader
        dataset = ErrorDataset(errors, labels, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize model
        self.model = DeepErrorClassifier(num_classes).to(self.device)
        
        # Setup optimizer and loss
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Training metrics
        train_losses = []
        train_accuracies = []
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            correct = 0
            total = 0
            
            for batch in dataloader:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs['logits'], labels)
                
                # Add auxiliary losses
                syntax_labels = self._get_syntax_labels(batch, dataset)
                runtime_labels = self._get_runtime_labels(batch, dataset)
                logic_labels = self._get_logic_labels(batch, dataset)
                
                if syntax_labels is not None:
                    loss += 0.1 * criterion(outputs['syntax_logits'], syntax_labels.to(self.device))
                if runtime_labels is not None:
                    loss += 0.1 * criterion(outputs['runtime_logits'], runtime_labels.to(self.device))
                if logic_labels is not None:
                    loss += 0.1 * criterion(outputs['logic_logits'], logic_labels.to(self.device))
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Update metrics
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs['logits'], 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            # Calculate epoch metrics
            avg_loss = epoch_loss / len(dataloader)
            accuracy = correct / total
            train_losses.append(avg_loss)
            train_accuracies.append(accuracy)
            
            logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        return {
            'epochs': epochs,
            'final_loss': train_losses[-1],
            'final_accuracy': train_accuracies[-1],
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'num_classes': num_classes,
            'classes': self.label_encoder.classes_.tolist()
        }
    
    def _get_syntax_labels(self, batch, dataset):
        """Get syntax error labels for auxiliary task."""
        # This would be implemented based on actual error patterns
        return None
    
    def _get_runtime_labels(self, batch, dataset):
        """Get runtime error labels for auxiliary task."""
        # This would be implemented based on actual error patterns
        return None
    
    def _get_logic_labels(self, batch, dataset):
        """Get logic error labels for auxiliary task."""
        # This would be implemented based on actual error patterns
        return None
    
    def predict(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict error category using the trained model.
        
        Args:
            error_data: Error data dictionary
            
        Returns:
            Prediction results
        """
        if self.model is None:
            return {
                'error': 'Model not trained',
                'success': False
            }
        
        self.model.eval()
        
        # Create dataset for single prediction
        dataset = ErrorDataset([error_data], tokenizer=self.tokenizer)
        item = dataset[0]
        
        # Prepare inputs
        input_ids = item['input_ids'].unsqueeze(0).to(self.device)
        attention_mask = item['attention_mask'].unsqueeze(0).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            logits = outputs['logits']
            probs = F.softmax(logits, dim=-1)
            
            # Get top prediction
            confidence, predicted_idx = torch.max(probs, dim=-1)
            predicted_class = self.label_encoder.inverse_transform([predicted_idx.cpu().item()])[0]
            
            # Get attention weights
            attention_weights = outputs['attention_weights'].cpu().numpy()
            
            # Get alternative predictions
            top_k = min(3, len(self.label_encoder.classes_))
            top_probs, top_indices = torch.topk(probs[0], top_k)
            alternatives = []
            for i in range(1, top_k):
                alt_class = self.label_encoder.inverse_transform([top_indices[i].cpu().item()])[0]
                alternatives.append({
                    'class': alt_class,
                    'probability': float(top_probs[i].cpu().item())
                })
        
        # Extract pattern features for interpretability
        error_text = dataset._extract_text_features(error_data)
        # TODO: pattern_features could be included in the return dict for enhanced interpretability
        # pattern_features = self.extract_pattern_features(error_text)
        
        return {
            'error_type': predicted_class,
            'confidence': float(confidence.cpu().item()),
            'alternatives': alternatives,
            'attention_weights': attention_weights.tolist(),
            'pattern_matches': self._get_pattern_matches(error_text),
            'features': outputs['features'].cpu().numpy().tolist(),
            'success': True
        }
    
    def _get_pattern_matches(self, error_text: str) -> Dict[str, bool]:
        """Get which error patterns matched."""
        matches = {}
        for category, patterns in self.error_patterns.items():
            matches[category] = any(re.search(pattern, error_text, re.IGNORECASE) 
                                  for pattern in patterns)
        return matches
    
    def save(self, path: Optional[str] = None) -> bool:
        """Save the trained model."""
        if self.model is None:
            logger.error("No model to save")
            return False
        
        save_path = Path(path or self.model_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model state and configuration
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'label_encoder': self.label_encoder,
            'model_config': {
                'num_classes': len(self.label_encoder.classes_),
                'hidden_dim': 768,
                'dropout_rate': 0.3
            }
        }, save_path)
        
        logger.info(f"Model saved to {save_path}")
        return True
    
    def load(self, path: Optional[str] = None) -> bool:
        """Load a trained model."""
        load_path = Path(path or self.model_path)
        
        if not load_path.exists():
            logger.error(f"Model file not found: {load_path}")
            return False
        
        try:
            checkpoint = torch.load(load_path, map_location=self.device)
            
            # Load label encoder
            self.label_encoder = checkpoint['label_encoder']
            
            # Initialize model
            config = checkpoint['model_config']
            self.model = DeepErrorClassifier(
                num_classes=config['num_classes'],
                hidden_dim=config.get('hidden_dim', 768),
                dropout_rate=config.get('dropout_rate', 0.3)
            ).to(self.device)
            
            # Load model weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            logger.info(f"Model loaded from {load_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False


def create_synthetic_training_data(num_samples: int = 1000) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Create synthetic training data for testing.
    
    Args:
        num_samples: Number of samples to generate
        
    Returns:
        Tuple of (errors, labels)
    """
    error_templates = {
        'missing_key': {
            'messages': [
                "KeyError: '{key}'",
                "KeyError: \"{key}\"",
                "dictionary key '{key}' not found"
            ],
            'tracebacks': [
                "File 'app.py', line {line}, in {func}\n    value = data['{key}']",
                "File 'utils.py', line {line}, in {func}\n    return config['{key}']"
            ]
        },
        'type_mismatch': {
            'messages': [
                "TypeError: unsupported operand type(s) for +: '{type1}' and '{type2}'",
                "TypeError: '{obj}' object is not {action}",
                "TypeError: expected {expected}, got {actual}"
            ],
            'tracebacks': [
                "File 'calc.py', line {line}, in {func}\n    result = a + b",
                "File 'process.py', line {line}, in {func}\n    obj.{method}()"
            ]
        },
        'index_error': {
            'messages': [
                "IndexError: list index out of range",
                "IndexError: tuple index out of range",
                "IndexError: index {idx} is out of bounds for axis 0"
            ],
            'tracebacks': [
                "File 'data.py', line {line}, in {func}\n    item = items[{idx}]",
                "File 'array.py', line {line}, in {func}\n    value = arr[{idx}]"
            ]
        },
        'import_error': {
            'messages': [
                "ImportError: No module named '{module}'",
                "ModuleNotFoundError: No module named '{module}'",
                "ImportError: cannot import name '{name}' from '{module}'"
            ],
            'tracebacks': [
                "File 'main.py', line {line}, in <module>\n    import {module}",
                "File 'app.py', line {line}, in <module>\n    from {module} import {name}"
            ]
        },
        'attribute_error': {
            'messages': [
                "AttributeError: '{cls}' object has no attribute '{attr}'",
                "AttributeError: module '{module}' has no attribute '{attr}'",
                "AttributeError: NoneType object has no attribute '{attr}'"
            ],
            'tracebacks': [
                "File 'model.py', line {line}, in {func}\n    obj.{attr}",
                "File 'service.py', line {line}, in {func}\n    result = data.{attr}()"
            ]
        }
    }
    
    # Common placeholders
    keys = ['user_id', 'config', 'data', 'value', 'token', 'session_id']
    types = ['int', 'str', 'list', 'dict', 'NoneType', 'float']
    objects = ['Response', 'User', 'Config', 'Database', 'Logger']
    actions = ['iterable', 'callable', 'subscriptable', 'hashable']
    modules = ['requests', 'pandas', 'numpy', 'custom_module', 'utils', 'helpers']
    attributes = ['get', 'post', 'save', 'load', 'process', 'validate']
    functions = ['process_data', 'handle_request', 'validate_input', 'save_result']
    
    errors = []
    labels = []
    
    for i in range(num_samples):
        # Randomly select error type
        error_type = np.random.choice(list(error_templates.keys()))
        template = error_templates[error_type]
        
        # Generate error message
        message_template = np.random.choice(template['messages'])
        message = message_template.format(
            key=np.random.choice(keys),
            type1=np.random.choice(types),
            type2=np.random.choice(types),
            obj=np.random.choice(objects),
            action=np.random.choice(actions),
            expected=np.random.choice(types),
            actual=np.random.choice(types),
            idx=np.random.randint(0, 100),
            module=np.random.choice(modules),
            name=np.random.choice(attributes),
            cls=np.random.choice(objects),
            attr=np.random.choice(attributes)
        )
        
        # Generate traceback
        traceback_template = np.random.choice(template['tracebacks'])
        traceback = traceback_template.format(
            line=np.random.randint(1, 1000),
            func=np.random.choice(functions),
            key=np.random.choice(keys),
            method=np.random.choice(attributes),
            idx=np.random.randint(0, 100),
            module=np.random.choice(modules),
            name=np.random.choice(attributes),
            attr=np.random.choice(attributes)
        )
        
        # Create error data
        error_data = {
            'timestamp': f'2024-01-{i % 30 + 1:02d}T{i % 24:02d}:00:00',
            'service': 'example_service',
            'level': 'ERROR',
            'message': message,
            'exception_type': message.split(':')[0],
            'traceback': [
                'Traceback (most recent call last):',
                traceback,
                message
            ],
            'error_details': {
                'exception_type': message.split(':')[0],
                'message': message.split(':', 1)[1].strip() if ':' in message else message,
                'detailed_frames': [
                    {
                        'file': '/app/' + traceback.split("'")[1],
                        'line': int(traceback.split('line ')[1].split(',')[0]),
                        'function': traceback.split('in ')[1].split('\n')[0],
                        'code': traceback.split('\n')[1].strip() if '\n' in traceback else ''
                    }
                ]
            }
        }
        
        errors.append(error_data)
        labels.append(error_type)
    
    return errors, labels


if __name__ == "__main__":
    # Test the deep learning classifier
    logger.info("Creating synthetic training data...")
    errors, labels = create_synthetic_training_data(1000)
    
    # Initialize and train model
    recognizer = ErrorPatternRecognizer()
    
    logger.info("Training deep learning model...")
    results = recognizer.train(errors[:800], labels[:800], epochs=5)
    
    logger.info(f"Training completed with accuracy: {results['final_accuracy']:.4f}")
    
    # Save model
    model_path = Path(__file__).parent / "deep_error_classifier.pt"
    recognizer.save(model_path)
    
    # Test predictions
    logger.info("\nTesting predictions...")
    for i in range(5):
        test_error = errors[800 + i]
        true_label = labels[800 + i]
        
        prediction = recognizer.predict(test_error)
        logger.info(f"\nTrue label: {true_label}")
        logger.info(f"Predicted: {prediction['error_type']} (confidence: {prediction['confidence']:.4f})")
        logger.info(f"Pattern matches: {prediction['pattern_matches']}")