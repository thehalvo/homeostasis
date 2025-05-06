# Homeostasis Analysis Module

This module analyzes errors and logs to identify the root cause of issues and suggest appropriate fixes.

## Features

- Rule-based detection system for common errors
- Pattern matching for identifying error sources
- Root cause analysis algorithms with confidence scoring
- Enhanced AI-based analysis framework with multiple model types
- Pluggable architecture for integrating new analysis strategies
- Hybrid analysis approach combining multiple techniques
- Extensible rule configuration system
- Category-based rule organization
- JSON/YAML rule definition format

## Components

### Rule-Based Analyzer

The rule-based analyzer uses predefined patterns to identify common error types and suggest fixes. It includes:

- Pattern matching for Python exceptions
- Framework-specific error patterns (e.g., FastAPI)
- Confidence scoring based on pattern specificity
- Error description and fix suggestion
- Enhanced pattern matching using detailed error context

### Rule Configuration System

The rule configuration system allows dynamic loading and management of error analysis rules:

- Organize rules by categories (Python, Framework, Database, etc.)
- Define rules in JSON or YAML formats
- Tag-based rule filtering and grouping
- Rule severity levels and confidence indicators
- Detailed rule metadata including examples and descriptions
- External rule file loading from configurable directories

### AI-Based Analysis

The AI-based analysis framework provides:

- Multiple model types (classifier, similarity, LLM, ensemble)
- Pluggable architecture for easy extension
- Confidence-based result selection
- Enhanced context extraction for better analysis

### Analysis Strategies

Multiple analysis strategies are supported:

- **Rule-Based Only**: Uses only rule-based analysis
- **AI Fallback**: Uses AI when rule-based confidence is low
- **AI Enhanced**: Combines rule-based and AI analysis
- **AI Primary**: Uses AI as primary, rule-based as fallback
- **Ensemble**: Uses multiple models and techniques together

## Usage

### Basic Error Analysis

```python
from modules.analysis.analyzer import Analyzer, AnalysisStrategy

# Initialize with preferred strategy
analyzer = Analyzer(
    strategy=AnalysisStrategy.AI_ENHANCED,
    ai_model_type="stub"  # or "classifier", "llm", etc.
)

# Analyze an error
error_data = {
    "timestamp": "2023-01-01T12:00:00", 
    "service": "example_service",
    "level": "ERROR",
    "message": "KeyError: 'todo_id'",
    "exception_type": "KeyError",
    "traceback": ["Traceback (most recent call last):", "...", "KeyError: 'todo_id'"]
}

# Get analysis results
analysis = analyzer.analyze_error(error_data)

# Extract results
print(f"Root Cause: {analysis['root_cause']}")
print(f"Description: {analysis['description']}")
print(f"Suggestion: {analysis['suggestion']}")
print(f"Confidence: {analysis['confidence']}")
print(f"Analysis Method: {analysis['analysis_method']}")

# Utility function for quick analysis
from modules.analysis.analyzer import analyze_error_from_log

quick_analysis = analyze_error_from_log(
    error_data,
    strategy=AnalysisStrategy.AI_FALLBACK
)
```

### Using Rule Configuration System

```python
from modules.analysis.rule_based import RuleBasedAnalyzer
from modules.analysis.rule_config import (
    Rule, RuleSet, RuleLoader, RuleCategory, RuleSeverity, RuleConfidence
)

# Initialize with specific rule categories
analyzer = RuleBasedAnalyzer(categories=[
    RuleCategory.PYTHON,
    RuleCategory.DATABASE,
    "django"  # String category names also work
])

# Get statistics about loaded rules
stats = analyzer.get_stats()
print(f"Loaded {stats['total_rules']} rules")
print(f"Categories: {stats['categories']}")
print(f"Tags: {stats['tags']}")

# Filter rules by category or tag
python_rules = analyzer.get_rules_by_category(RuleCategory.PYTHON)
database_rules = analyzer.get_rules_by_tag("database")

# Create and add a custom rule
custom_rule = Rule(
    pattern=r"PermissionError: \[Errno 13\] Permission denied: '([^']*)'",
    type="PermissionError",
    description="Insufficient permissions to access a file or directory",
    root_cause="permission_denied",
    suggestion="Check file permissions or run with elevated privileges",
    category=RuleCategory.CUSTOM,
    severity=RuleSeverity.HIGH,
    tags=["filesystem", "permissions"]
)

analyzer.add_rule(custom_rule)

# Create a rule set and export to file
from pathlib import Path
import json

rule_set = RuleSet(
    name="Custom Error Rules",
    rules=[custom_rule],
    description="Custom error rules for specific application"
)

rules_dir = Path("modules/analysis/rules/custom")
rules_dir.mkdir(exist_ok=True)
rules_file = rules_dir / "my_custom_rules.json"

with open(rules_file, "w") as f:
    json.dump(rule_set.to_dict(), f, indent=2)

# Load rules from a file
loaded_rule_set = RuleLoader.load_from_file(rules_file)
```

## AI Model Types

The following AI model types are available:

- **stub**: Simple placeholder implementation for testing
- **classifier**: Machine learning classifier (future implementation)
- **similarity**: Embedding-based similarity search (future implementation)
- **llm**: Large Language Model based analysis (future implementation)
- **ensemble**: Combination of multiple models (future implementation)

## Extending with New Analyzers

The AI-based framework can be extended with new analyzers by:

1. Subclassing the `AIModel` abstract base class
2. Implementing the required methods (`initialize` and `analyze`)
3. Registering the new model type in `AIModelType` and `AVAILABLE_MODELS`

```python
from modules.analysis.ai_stub import AIModel, AIModelConfig, AIModelType

class CustomModel(AIModel):
    def initialize(self) -> bool:
        # Initialize your model
        return True
        
    def analyze(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        # Analyze the error
        return {
            "root_cause": "...",
            "description": "...",
            "suggestion": "...", 
            "confidence": 0.7
        }
```