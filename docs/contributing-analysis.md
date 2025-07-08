# Contributing to the Analysis Module

This guide provides detailed instructions for extending and improving the Analysis Module of Homeostasis.

## Understanding the Analysis Module

The Analysis Module serves as the "diagnostic" component of Homeostasis, responsible for:

1. Receiving standardized error data from the Monitoring Module
2. Analyzing errors using rules, patterns, and algorithms
3. Determining root causes of issues
4. Recommending appropriate fix templates
5. Providing context for the Patch Generation Module

## Extension Points

The Analysis Module offers several extension points:

1. **Rule Engines**: Different approaches to matching and analyzing errors
2. **Pattern Matchers**: Specialized components for recognizing error patterns
3. **Root Cause Analyzers**: Components that determine the underlying issue
4. **AI Integrations**: Machine learning components for error analysis
5. **Framework-Specific Analyzers**: Specialized analysis for specific technologies

## Creating a Custom Rule Engine

Rule engines define how rules are processed and matched against errors.

### 1. Plan Your Rule Engine

Determine what type of rule engine you want to create:

- Regex-based pattern matching
- Abstract syntax tree (AST) analysis
- Context-aware rule evaluator
- ML-based pattern recognition

### 2. Implement the Rule Engine

Create your rule engine in the `modules/analysis` directory:

```python
# In modules/analysis/my_rule_engine.py

class MyCustomRuleEngine:
    """
    A custom rule engine for specialized error analysis.
    
    This engine focuses on [specific type of analysis or errors].
    """
    
    def __init__(self, rules_config=None):
        """Initialize the rule engine with optional configuration."""
        self.rules = []
        self.config = rules_config or {}
        self._load_rules()
    
    def _load_rules(self):
        """Load rules from configuration or default locations."""
        # Load rule definitions
        # ...
        
    def analyze(self, error_data):
        """
        Analyze error data and determine matching rules.
        
        Args:
            error_data (dict): Standardized error data from monitoring module
            
        Returns:
            dict: Analysis results including matched rules and fix suggestions
        """
        results = {
            "matched_rules": [],
            "root_cause": None,
            "confidence": 0.0,
            "fix_suggestions": []
        }
        
        # Implement your analysis logic
        # ...
        
        return results
    
    def register_rule(self, rule):
        """Register a new rule with this engine."""
        # Validate and add the rule
        self.rules.append(rule)
        return True
```

### 3. Register Your Rule Engine

Add your engine to the analyzer registry in `modules/analysis/analyzer.py`:

```python
from .rule_based import RuleBasedEngine
from .ai_stub import AIAnalysisEngine
from .my_rule_engine import MyCustomRuleEngine

ANALYSIS_ENGINES = {
    "rule_based": RuleBasedEngine,
    "ai": AIAnalysisEngine,
    "my_custom": MyCustomRuleEngine
}

def get_analyzer(engine_type="rule_based", config=None):
    """Get the appropriate analyzer engine."""
    if engine_type in ANALYSIS_ENGINES:
        return ANALYSIS_ENGINES[engine_type](config)
    return ANALYSIS_ENGINES["rule_based"](config)
```

## Creating a Pattern Matcher

Pattern matchers are specialized components for recognizing specific error patterns.

### 1. Identify Pattern Types

Choose what patterns you want to match:

- Regular expressions for text logs
- JSON path expressions for structured logs
- Stack trace patterns
- AST patterns for code analysis

### 2. Implement the Pattern Matcher

Create a pattern matcher in `modules/analysis/patterns/`:

```python
# In modules/analysis/patterns/stack_trace_matcher.py

class StackTracePatternMatcher:
    """
    Pattern matcher specialized for stack trace analysis.
    """
    
    def __init__(self, config=None):
        self.config = config or {}
        self.patterns = self._load_patterns()
    
    def _load_patterns(self):
        """Load stack trace patterns from configuration."""
        # Load pattern definitions
        # ...
        return []
    
    def match(self, stack_trace):
        """
        Match patterns against a stack trace.
        
        Args:
            stack_trace (list): List of stack trace lines
            
        Returns:
            dict: Match results with confidence scores
        """
        results = {
            "matches": [],
            "top_match": None,
            "confidence": 0.0
        }
        
        # Implement matching logic
        # ...
        
        return results
```

### 3. Integrate with Rule Engine

Use your pattern matcher in a rule engine:

```python
# In your rule engine implementation

from .patterns.stack_trace_matcher import StackTracePatternMatcher

class EnhancedRuleEngine(RuleBasedEngine):
    """Enhanced rule engine with stack trace pattern matching."""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.stack_matcher = StackTracePatternMatcher(config)
    
    def analyze(self, error_data):
        """Analyze with enhanced pattern matching."""
        results = super().analyze(error_data)
        
        # Add stack trace analysis if available
        if "stack_trace" in error_data:
            stack_matches = self.stack_matcher.match(error_data["stack_trace"])
            if stack_matches["top_match"]:
                results["stack_analysis"] = stack_matches
                # Update confidence or add stack-based suggestions
                if stack_matches["confidence"] > results["confidence"]:
                    results["confidence"] = stack_matches["confidence"]
                    results["root_cause"] = stack_matches["top_match"]["cause"]
        
        return results
```

## Creating an AI-Based Analyzer

The Homeostasis framework includes a placeholder for AI-based analysis. You can extend this to implement actual AI capabilities.

### 1. Design Your AI Approach

Determine what AI techniques you want to use:

- Rule-based expert systems
- Machine learning classifiers
- Natural language processing
- Deep learning models

### 2. Implement AI Analyzer

Extend the AI stub in `modules/analysis/ai_stub.py`:

```python
# Update the existing ai_stub.py or create a new file

import json
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

class MLBasedAnalyzer:
    """
    Machine learning based analyzer for error classification.
    """
    
    def __init__(self, config=None):
        """Initialize the ML analyzer with trained model."""
        self.config = config or {}
        self.model_path = self.config.get("model_path", "models/error_classifier.pkl")
        self.model = self._load_model()
        
    def _load_model(self):
        """Load the trained model from disk."""
        try:
            import joblib
            if os.path.exists(self.model_path):
                return joblib.load(self.model_path)
            else:
                # Create a basic model if none exists
                return self._create_default_model()
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return self._create_default_model()
    
    def _create_default_model(self):
        """Create a default model if none exists."""
        # Simple text classification pipeline
        return Pipeline([
            ('vectorizer', TfidfVectorizer(max_features=1000)),
            ('classifier', MultinomialNB())
        ])
    
    def train(self, training_data):
        """
        Train the model with labeled error data.
        
        Args:
            training_data (list): List of dicts with 'error_text' and 'label'
        """
        # Extract features and labels
        texts = [item['error_text'] for item in training_data]
        labels = [item['label'] for item in training_data]
        
        # Train the model
        self.model.fit(texts, labels)
        
        # Save the model
        import joblib
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        
        return True
    
    def analyze(self, error_data):
        """
        Analyze error data using the ML model.
        
        Args:
            error_data (dict): Standardized error data
            
        Returns:
            dict: Analysis results
        """
        # Extract text features from error data
        error_text = self._extract_text_features(error_data)
        
        # Make prediction if model exists and text is available
        if self.model and error_text:
            try:
                # Get prediction probabilities
                proba = self.model.predict_proba([error_text])[0]
                classes = self.model.classes_
                
                # Get top predictions
                top_indices = np.argsort(proba)[::-1][:3]  # Top 3 predictions
                
                predictions = [
                    {"label": classes[i], "confidence": float(proba[i])}
                    for i in top_indices if proba[i] > 0.1  # Only include if confidence > 10%
                ]
                
                # Map predictions to fix suggestions if available
                fix_suggestions = self._map_to_fix_templates(predictions)
                
                return {
                    "matched_rules": [],  # AI doesn't use rules directly
                    "predictions": predictions,
                    "top_prediction": predictions[0] if predictions else None,
                    "confidence": float(max(proba)) if len(proba) > 0 else 0.0,
                    "fix_suggestions": fix_suggestions
                }
            except Exception as e:
                print(f"Prediction error: {str(e)}")
                
        # Fallback
        return {
            "matched_rules": [],
            "predictions": [],
            "confidence": 0.0,
            "fix_suggestions": []
        }
    
    def _extract_text_features(self, error_data):
        """Extract text features from error data for analysis."""
        features = []
        
        # Add error type and message
        if "type" in error_data:
            features.append(error_data["type"])
        if "message" in error_data:
            features.append(error_data["message"])
            
        # Add stack trace if available
        if "stack_trace" in error_data and error_data["stack_trace"]:
            if isinstance(error_data["stack_trace"], list):
                features.extend(error_data["stack_trace"][:5])  # First 5 lines
            elif isinstance(error_data["stack_trace"], str):
                features.append(error_data["stack_trace"])
        
        # Join all features into a single text
        return " ".join(features)
    
    def _map_to_fix_templates(self, predictions):
        """Map ML predictions to fix templates."""
        # Load mapping from configuration or predefined mappings
        mapping = self.config.get("label_to_template_mapping", {})
        
        suggestions = []
        for pred in predictions:
            label = pred["label"]
            if label in mapping:
                suggestions.append({
                    "template": mapping[label],
                    "confidence": pred["confidence"],
                    "parameters": {}  # Parameters would need extraction logic
                })
        
        return suggestions
```

### 3. Add Training Data Collection

Implement a component to collect training data:

```python
# In modules/analysis/training_data_collector.py

import json
import os
from datetime import datetime

class TrainingDataCollector:
    """
    Collects and stores training data for ML models.
    """
    
    def __init__(self, data_dir="data/training"):
        """Initialize the collector with a data directory."""
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
    def save_example(self, error_data, label, fix_template=None, success=None):
        """
        Save an error example for training.
        
        Args:
            error_data (dict): The error data
            label (str): The classification label
            fix_template (str, optional): The fix template used
            success (bool, optional): Whether the fix was successful
            
        Returns:
            bool: Whether the save was successful
        """
        try:
            # Create training example
            example = {
                "error_data": error_data,
                "label": label,
                "fix_template": fix_template,
                "success": success,
                "timestamp": datetime.now().isoformat()
            }
            
            # Generate filename
            filename = f"{int(datetime.now().timestamp())}_{label}.json"
            filepath = os.path.join(self.data_dir, filename)
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(example, f, indent=2)
                
            return True
        except Exception as e:
            print(f"Error saving training example: {str(e)}")
            return False
    
    def load_training_data(self, limit=None):
        """
        Load training data from the data directory.
        
        Args:
            limit (int, optional): Maximum number of examples to load
            
        Returns:
            list: Training examples
        """
        examples = []
        
        try:
            files = os.listdir(self.data_dir)
            json_files = [f for f in files if f.endswith('.json')]
            
            # Sort by timestamp (filename starts with timestamp)
            json_files.sort(reverse=True)
            
            # Apply limit if specified
            if limit:
                json_files = json_files[:limit]
            
            # Load each file
            for filename in json_files:
                filepath = os.path.join(self.data_dir, filename)
                with open(filepath, 'r') as f:
                    example = json.load(f)
                    
                    # Extract error text for training
                    error_text = self._extract_text(example["error_data"])
                    if error_text:
                        examples.append({
                            "error_text": error_text,
                            "label": example["label"],
                            "fix_template": example.get("fix_template"),
                            "success": example.get("success")
                        })
            
            return examples
        except Exception as e:
            print(f"Error loading training data: {str(e)}")
            return []
    
    def _extract_text(self, error_data):
        """Extract text representation from error data."""
        # Similar to the feature extraction in the ML analyzer
        features = []
        
        if "type" in error_data:
            features.append(error_data["type"])
        if "message" in error_data:
            features.append(error_data["message"])
            
        if "stack_trace" in error_data and error_data["stack_trace"]:
            if isinstance(error_data["stack_trace"], list):
                features.extend(error_data["stack_trace"][:5])
            elif isinstance(error_data["stack_trace"], str):
                features.append(error_data["stack_trace"])
        
        return " ".join(features)
```

## Testing Your Contributions

### Unit Testing

Create unit tests for your components:

```python
# In tests/test_analysis.py

import unittest
from homeostasis.modules.analysis.my_rule_engine import MyCustomRuleEngine

class TestMyRuleEngine(unittest.TestCase):
    """Tests for custom rule engine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = MyCustomRuleEngine()
        self.sample_error = {
            "type": "KeyError",
            "message": "'user_id'",
            "stack_trace": [
                "Traceback (most recent call last):",
                "  File \"app.py\", line 42, in get_user",
                "    user = users[user_id]",
                "KeyError: 'user_id'"
            ],
            "context": {
                "function": "get_user",
                "variables": {
                    "users": "{}"
                }
            }
        }
    
    def test_analyze(self):
        """Test basic analysis functionality."""
        result = self.engine.analyze(self.sample_error)
        
        # Verify result structure
        self.assertIn("matched_rules", result)
        self.assertIn("confidence", result)
        self.assertIn("fix_suggestions", result)
        
        # Verify analysis quality
        # These assertions will depend on your specific implementation
        self.assertGreater(len(result["matched_rules"]), 0)
        self.assertGreater(result["confidence"], 0.5)
        
    def test_rule_registration(self):
        """Test rule registration."""
        initial_count = len(self.engine.rules)
        
        # Register a test rule
        test_rule = {
            "id": "test_rule_001",
            "pattern": "TestError",
            "confidence": 0.9
        }
        
        success = self.engine.register_rule(test_rule)
        self.assertTrue(success)
        self.assertEqual(len(self.engine.rules), initial_count + 1)

# Add more test cases as needed
```

### Integration Testing

Test integration with the full analysis pipeline:

```python
# In tests/test_analysis_integration.py

import unittest
from homeostasis.modules.analysis.analyzer import get_analyzer
from homeostasis.modules.monitoring.logger import format_error

class TestAnalysisIntegration(unittest.TestCase):
    """Test integration of analysis components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.rule_engine = get_analyzer("rule_based")
        self.custom_engine = get_analyzer("my_custom")
        self.ai_engine = get_analyzer("ai")
        
        # Sample error for testing
        self.error = {
            "type": "KeyError",
            "message": "'user_id'",
            "stack_trace": [
                "Traceback (most recent call last):",
                "  File \"app.py\", line 42, in get_user",
                "    user = users[user_id]",
                "KeyError: 'user_id'"
            ]
        }
        
        # Format error for analysis
        self.formatted_error = format_error(self.error)
    
    def test_rule_engine_integration(self):
        """Test rule engine integration."""
        result = self.rule_engine.analyze(self.formatted_error)
        
        # Verify rule engine produces expected results
        self.assertIsNotNone(result)
        self.assertIn("matched_rules", result)
        
    def test_custom_engine_integration(self):
        """Test custom engine integration."""
        result = self.custom_engine.analyze(self.formatted_error)
        
        # Verify custom engine produces expected results
        self.assertIsNotNone(result)
        # Add assertions specific to your custom engine
        
    def test_ai_engine_integration(self):
        """Test AI engine integration."""
        result = self.ai_engine.analyze(self.formatted_error)
        
        # Verify AI engine produces expected results
        self.assertIsNotNone(result)
        self.assertIn("predictions", result)
        
    def test_analyzer_selection(self):
        """Test analyzer selection mechanism."""
        # Test default engine
        default_analyzer = get_analyzer()
        self.assertIsNotNone(default_analyzer)
        
        # Test custom engine selection
        custom_analyzer = get_analyzer("my_custom")
        self.assertIsNotNone(custom_analyzer)
        self.assertIsInstance(custom_analyzer, MyCustomRuleEngine)
        
        # Test invalid engine falls back to default
        fallback_analyzer = get_analyzer("non_existent")
        self.assertIsNotNone(fallback_analyzer)
        self.assertIsInstance(fallback_analyzer, RuleBasedEngine)
```

## Contribution Checklist

- [ ] Component has clear purpose and interfaces
- [ ] Code follows project style and conventions
- [ ] Documentation describes usage and extension points
- [ ] Unit tests verify component functionality
- [ ] Integration tests confirm compatibility with other modules
- [ ] Performance impact is considered
- [ ] Error handling is robust

## Best Practices

1. **Focus on Extensibility**: Design your components to be easily extended by others
2. **Balance Precision and Recall**: Error analysis should minimize both false positives and false negatives
3. **Consider Performance**: Analysis should be efficient enough for production use
4. **Document Clearly**: Provide clear documentation for how others can build on your work
5. **Support Incremental Improvement**: Design systems that can learn and improve over time

## Frequently Asked Questions

### How do I decide between rule-based and ML-based approaches?
Rule-based approaches work well for well-defined patterns with clear solutions. ML approaches excel at handling edge cases and complex patterns, but require training data.

### How can I contribute if I'm not an ML expert?
Focus on rule-based components, pattern matching, or improving the framework's architecture. You can also help collect and label training data.

### How do I test my analyzer with real-world errors?
Use the monitoring module to collect real error data, then feed it to your analyzer for testing.

---

By contributing to the Analysis Module, you help Homeostasis better identify and understand errors, leading to more accurate and effective self-healing. Thank you for your contribution!