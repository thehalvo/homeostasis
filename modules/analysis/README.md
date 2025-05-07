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

## Recent Enhancements

- **Expanded Rule Sets**: Added 50+ new rules for common Python errors and exceptions
- **Enhanced Categorization System**: Added criticality, complexity, and reliability metrics
- **Confidence Scoring Algorithm**: Improved match accuracy with multi-factor scoring
- **Rule Management CLI**: New command-line tool for rule testing and management
- **Rule Dependency Resolution**: Support for complex error chains and dependencies
- **Conflict Detection**: Automatic detection of conflicting or overlapping rules
- **Visualization Support**: Tools for viewing rule coverage and effectiveness

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

### Rule Categories and Organization

Rules are organized into categories:
- **Python**: Common Python exceptions and errors
- **Framework-specific**: FastAPI, Flask, Django
- **Database**: SQL and ORM-related errors
- **Network**: Network-related errors
- **Authentication/Authorization**: Security-related errors
- **Custom**: User-defined rules

### Enhanced Rule Properties

Each rule now includes:
- Regular expression pattern to match against errors
- Error type identification
- Root cause classification
- Suggested fix
- Confidence and severity levels
- Tags for organization
- Example error messages

Enhanced rules also include:
- **Criticality levels**: critical, high, medium, low
- **Complexity levels**: simple, moderate, complex, unknown
- **Reliability levels**: high, medium, low
- **Dependency information**: for complex error chains

### Confidence Scoring System

The enhanced confidence scoring system evaluates rule matches based on:
- **Context relevance**: How well the rule matches the error context
- **Pattern specificity**: How specific and precise the rule pattern is
- **Match strength**: How strongly the pattern matches the error
- **Pattern quality**: Inherent quality of the rule pattern

This provides more accurate analysis and can handle ambiguous error cases by ranking possible matches.

### Rule Management CLI

The command-line interface provides tools for managing and testing rules:

```bash
# Show rule statistics
./rule_cli.py stats

# Show rule details
./rule_cli.py show <rule_id>

# Create a new rule
./rule_cli.py create --pattern "Error: (.*)" --type "Error" --description "Generic error" \
  --root-cause "unknown_error" --suggestion "Check the error message for details" \
  --category "custom" --rule-set "my_rules"

# Test rules
./rule_cli.py test --use-examples
./rule_cli.py test --error "KeyError: 'user_id'"

# Analyze an error
./rule_cli.py analyze "KeyError: 'user_id'"
```

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

### Using Enhanced Rules and Confidence Scoring

```python
from modules.analysis.rule_categories import (
    EnhancedRule, RuleCriticality, RuleComplexity, RuleReliability,
    RuleSource, RuleType
)
from modules.analysis.rule_confidence import (
    ContextualRuleAnalyzer, ConfidenceScorer
)

# Create an enhanced rule
enhanced_rule = EnhancedRule(
    pattern=r"KeyError: '([^']*)'",
    type="KeyError",
    description="Dictionary key not found",
    root_cause="missing_dict_key",
    suggestion="Check if the key exists before accessing it",
    criticality=RuleCriticality.HIGH,
    complexity=RuleComplexity.SIMPLE,
    reliability=RuleReliability.HIGH,
    tags=["python", "dictionary", "data-structure"]
)

# Use contextual analyzer with confidence scoring
analyzer = ContextualRuleAnalyzer([enhanced_rule])
error_text = "KeyError: 'user_id'"

error_context = {
    "exception_type": "KeyError",
    "detailed_frames": [
        {
            "file": "app.py",
            "line": 42,
            "function": "get_user",
            "locals": {
                "request_data": {"username": "test"}
            }
        }
    ]
}

analysis = analyzer.analyze_error(error_text, error_context)

print(f"Confidence Score: {analysis['confidence_score']:.2f}")
print(f"Factors: {analysis['confidence_factors']}")
if "alternative_matches" in analysis:
    print("Alternative matches found")
```

### Using Rule Dependencies

```python
from modules.analysis.rule_categories import RuleDependency

# Check for circular dependencies
dependency_checker = RuleDependency(rules)
circular_deps = dependency_checker.detect_circular_dependencies()

# Get dependent rules
dependent_rules = dependency_checker.get_dependent_rules("rule_id")

# Get prerequisite rules
prerequisites = dependency_checker.get_prerequisites("rule_id")
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