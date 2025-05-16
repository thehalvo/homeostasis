# Cross-Language Orchestration

This document explains the cross-language orchestration capabilities of Homeostasis, which enable the system to analyze and heal errors across different programming languages.

## Overview

Cross-language orchestration is a core feature of Homeostasis that allows it to:

1. Detect and normalize errors from multiple programming languages
2. Apply shared patterns and rules across language boundaries
3. Learn from fixes in one language and apply them to similar issues in other languages
4. Coordinate healing activities in polyglot applications
5. Provide consistent monitoring and metrics across the entire stack

## Architecture

The cross-language orchestration system is built around these key components:

### Enhanced Cross-Language Orchestrator

Located in `modules/analysis/enhanced_cross_language_orchestrator.py`, this component:

- Receives normalized errors from language-specific adapters
- Detects the language and routes to appropriate plugins
- Coordinates analysis and healing across languages
- Manages shared rules and patterns
- Collects and reports metrics

```python
class EnhancedCrossLanguageOrchestrator:
    """Orchestrates error handling across multiple languages"""
    
    def __init__(self):
        self.language_registry = LanguageRegistry()
        self.shared_rule_system = SharedRuleSystem()
        self.healing_metrics = HealingMetricsCollector()
        self.cache = CrossLanguageCache()
        
    def process_error(self, error_data):
        """Process an error from any supported language"""
        # Identify language and normalize if needed
        language = self._detect_language(error_data)
        normalized_error = self._ensure_normalized(error_data, language)
        
        # Find matching rules within language and across languages
        language_specific_matches = self._find_language_specific_matches(normalized_error)
        shared_matches = self._find_shared_matches(normalized_error)
        
        # Combine and rank potential fixes
        fixes = self._rank_fixes(language_specific_matches + shared_matches)
        
        # Apply the best fix
        result = self._apply_fix(normalized_error, fixes[0] if fixes else None)
        
        # Update metrics
        self.healing_metrics.track_healing_event(normalized_error, result)
        
        return result
```

### Language Adapters and Registry

The `LanguageRegistry` in `modules/analysis/language_plugin_system.py` manages the available language plugins:

```python
class LanguageRegistry:
    """Registry of available language plugins"""
    
    def __init__(self):
        self.plugins = {}
        self._load_plugins()
        
    def _load_plugins(self):
        """Load all available language plugins"""
        # Load built-in language plugins
        self.plugins["python"] = PythonLanguagePlugin()
        self.plugins["javascript"] = JavaScriptLanguagePlugin()
        self.plugins["java"] = JavaLanguagePlugin()
        self.plugins["go"] = GoLanguagePlugin()
        
        # Load any custom plugins
        self._load_custom_plugins()
        
    def get_plugin(self, language):
        """Get the plugin for a specific language"""
        if language in self.plugins:
            return self.plugins[language]
        return None
        
    def detect_language(self, error_data):
        """Detect the language from error data"""
        # Try explicit language field first
        if "language" in error_data:
            return error_data["language"]
            
        # Try to detect from error patterns
        for language, plugin in self.plugins.items():
            if plugin.can_handle(error_data):
                return language
                
        return "unknown"
```

### Shared Rule System

The `SharedRuleSystem` in `modules/analysis/shared_rule_system.py` provides language-agnostic rule matching:

```python
class SharedRuleSystem:
    """System for managing and applying shared rules across languages"""
    
    def __init__(self):
        self.shared_rules = []
        self._load_shared_rules()
        
    def _load_shared_rules(self):
        """Load all shared rules from rule files"""
        # Load core shared rules
        self._load_rules_from_file("modules/analysis/rules/shared/collection_errors_rules.json")
        self._load_rules_from_file("modules/analysis/rules/shared/reference_errors_rules.json")
        # Load additional rule files
        
    def find_matching_rules(self, normalized_error):
        """Find shared rules that match the given error"""
        matches = []
        
        # Check normalized classification first
        category = normalized_error["normalized_classification"]["category"]
        subcategory = normalized_error["normalized_classification"]["subcategory"]
        
        # Find rules that match this classification
        for rule in self.shared_rules:
            if self._rule_matches_classification(rule, category, subcategory):
                if self._rule_matches_language_pattern(rule, normalized_error):
                    matches.append(rule)
                    
        return matches
```

### Language-Specific Plugins

Each language has a dedicated plugin that handles language-specific analysis:

```python
class JavaLanguagePlugin(BaseLanguagePlugin):
    """Plugin for handling Java errors"""
    
    def __init__(self):
        super().__init__("java")
        self.adapter = JavaErrorAdapter()
        self._load_rules()
        
    def analyze_error(self, error_data):
        """Analyze a Java error and suggest fixes"""
        # Normalize if not already normalized
        normalized = self.adapter.ensure_normalized(error_data)
        
        # Find matching rules
        matches = self.find_matching_rules(normalized)
        
        # Generate potential fixes
        fixes = self.generate_fixes(normalized, matches)
        
        return {
            "normalized_error": normalized,
            "matches": matches,
            "fixes": fixes
        }
        
    def can_handle(self, error_data):
        """Check if this plugin can handle the given error"""
        # Check for Java-specific patterns
        if "error_type" in error_data:
            java_exceptions = [
                "NullPointerException", "ArrayIndexOutOfBoundsException",
                "ClassCastException", "IllegalArgumentException"
            ]
            if error_data["error_type"] in java_exceptions:
                return True
                
        # Check for Java stack trace patterns
        if "error_message" in error_data:
            if "at com.example" in error_data["error_message"] or \
               "at org.springframework" in error_data["error_message"]:
                return True
                
        return False
```

## Cross-Language Features

### Shared Error Schema

The shared error schema (documented in `error_schema.md`) provides a common representation for errors across languages. Key fields include:

- Common metadata (error ID, timestamp, etc.)
- Language-specific details (error type, message)
- Normalized classification (language-agnostic category)
- Stack trace in a standard format
- Environment information

### Cross-Language Rule Matching

Rules can match errors across language boundaries using several techniques:

1. **Normalized Classification**: The primary matching mechanism using language-agnostic categories
2. **Pattern Similarity**: Identifying similar error patterns across languages
3. **Contextual Matching**: Using code context to identify similar issues
4. **Historical Learning**: Applying successful fixes from one language to another

```python
def find_cross_language_matches(normalized_error, rule_system):
    """Find rules from other languages that might apply"""
    category = normalized_error["normalized_classification"]["category"]
    source_language = normalized_error["language"]
    
    # Find rules in other languages with the same category
    matches = []
    for rule in rule_system.all_rules:
        rule_category = rule.get("normalized_classification", {}).get("category")
        rule_language = rule.get("language")
        
        if rule_category == category and rule_language != source_language:
            # Check if this rule has a cross-language mapping
            if rule.has_cross_language_mapping(source_language):
                similarity = calculate_pattern_similarity(
                    normalized_error, rule.get_cross_language_pattern(source_language)
                )
                if similarity > 0.7:  # Threshold for cross-language matching
                    matches.append((rule, similarity))
    
    # Sort by similarity score
    matches.sort(key=lambda x: x[1], reverse=True)
    return [match[0] for match in matches]
```

### Fix Translation

When fixes are identified in one language, they can be adapted to other languages:

```python
class FixTranslator:
    """Translates fixes between languages"""
    
    def translate_fix(self, fix, source_language, target_language):
        """Translate a fix from source language to target language"""
        if fix["type"] == "parameter_check":
            return self._translate_parameter_check(fix, source_language, target_language)
        elif fix["type"] == "null_check":
            return self._translate_null_check(fix, source_language, target_language)
        elif fix["type"] == "bounds_check":
            return self._translate_bounds_check(fix, source_language, target_language)
        # Handle other fix types
        
        return None
    
    def _translate_null_check(self, fix, source_language, target_language):
        """Translate a null check fix between languages"""
        target_fix = copy.deepcopy(fix)
        
        # Map template path to target language
        if target_language == "python":
            target_fix["template_path"] = "templates/parameter_check.py.template"
        elif target_language == "javascript":
            target_fix["template_path"] = "templates/null_check.js.template"
        elif target_language == "java":
            target_fix["template_path"] = "templates/null_check.java.template"
        elif target_language == "go":
            target_fix["template_path"] = "templates/nil_check.go.template"
        
        # Translate parameters for the target language
        target_fix["parameters"] = self._translate_parameters(
            fix["parameters"], source_language, target_language
        )
        
        return target_fix
```

### Learning from Cross-Language Fixes

The system learns from successful fixes and improves over time:

```python
class CrossLanguageLearning:
    """Learns from fixes across languages"""
    
    def register_successful_fix(self, error, fix, result):
        """Register a successful fix for learning"""
        # Extract key patterns from the error
        error_category = error["normalized_classification"]["category"]
        language = error["language"]
        
        # Add to the pattern database
        self.patterns.add(error_category, language, {
            "error_pattern": self._extract_error_pattern(error),
            "fix_type": fix["type"],
            "fix_parameters": fix["parameters"],
            "result": result
        })
        
        # Generate cross-language mappings
        self._generate_cross_language_mappings(error_category, language)
    
    def _generate_cross_language_mappings(self, category, source_language):
        """Generate mappings for other languages based on patterns"""
        source_patterns = self.patterns.get(category, source_language)
        
        for target_language in self.supported_languages:
            if target_language == source_language:
                continue
                
            target_patterns = self.patterns.get(category, target_language)
            
            # Create mappings between source and target patterns
            for source_pattern in source_patterns:
                for target_pattern in target_patterns:
                    similarity = self._calculate_pattern_similarity(
                        source_pattern["error_pattern"],
                        target_pattern["error_pattern"]
                    )
                    
                    if similarity > 0.8:  # High similarity threshold
                        self.mappings.add(
                            category, source_language, target_language,
                            source_pattern, target_pattern, similarity
                        )
```

## Metrics and Monitoring

Cross-language metrics provide insights into healing effectiveness:

```python
class CrossLanguageMetrics:
    """Metrics for cross-language operations"""
    
    def record_cross_language_match(self, source_language, target_language, error_category, confidence):
        """Record a successful cross-language pattern match"""
        key = f"{source_language}_to_{target_language}.{error_category}"
        self.match_counts[key] = self.match_counts.get(key, 0) + 1
        self.confidence_sum[key] = self.confidence_sum.get(key, 0) + confidence
        
    def record_cross_language_fix(self, source_language, target_language, error_category, success):
        """Record a cross-language fix attempt"""
        key = f"{source_language}_to_{target_language}.{error_category}"
        self.fix_attempts[key] = self.fix_attempts.get(key, 0) + 1
        if success:
            self.fix_successes[key] = self.fix_successes.get(key, 0) + 1
            
    def get_success_rate(self, source_language, target_language, error_category=None):
        """Get the success rate for cross-language fixes"""
        if error_category:
            key = f"{source_language}_to_{target_language}.{error_category}"
            attempts = self.fix_attempts.get(key, 0)
            successes = self.fix_successes.get(key, 0)
        else:
            # Aggregate across all categories
            attempts = sum(count for key, count in self.fix_attempts.items() 
                         if key.startswith(f"{source_language}_to_{target_language}"))
            successes = sum(count for key, count in self.fix_successes.items() 
                          if key.startswith(f"{source_language}_to_{target_language}"))
        
        return successes / attempts if attempts > 0 else 0
```

## Testing

The backend testing framework validates cross-language capabilities:

```python
class CrossLanguageTest(LanguageTestCase):
    """Test case for cross-language functionality"""
    
    def __init__(self, test_id, source_error, target_language, expected_error, shared_rule_id):
        super().__init__(test_id)
        self.source_error = source_error
        self.target_language = target_language
        self.expected_error = expected_error
        self.shared_rule_id = shared_rule_id
        
    def run(self, orchestrator):
        """Run the cross-language test"""
        # First, process the source error
        source_result = orchestrator.process_error(self.source_error)
        
        # Check if the correct rule was identified
        if not self._check_rule_match(source_result, self.shared_rule_id):
            return TestResult(self.test_id, False, "Source error did not match expected shared rule")
        
        # Now check if the fix can be translated
        translator = FixTranslator()
        translated_fix = translator.translate_fix(
            source_result["fix"], 
            self.source_error["language"], 
            self.target_language
        )
        
        if not translated_fix:
            return TestResult(self.test_id, False, "Failed to translate fix to target language")
            
        # Verify the translated fix matches expectations
        if self._verify_translated_fix(translated_fix, self.expected_error):
            return TestResult(self.test_id, True, "Successfully translated fix across languages")
        else:
            return TestResult(self.test_id, False, "Translated fix did not match expectations")
```

## Implementation Examples

### Cross-Language Error Detection

```python
# Example: Detecting a null reference error across languages
def detect_null_reference_error(error_data):
    """Detect null reference errors in any language"""
    language = error_data.get("language", "unknown")
    error_type = error_data.get("error_type", "")
    error_message = error_data.get("error_message", "")
    
    patterns = {
        "python": {
            "types": ["AttributeError", "TypeError"],
            "messages": ["'NoneType' object has no attribute", "NoneType' object is not"]
        },
        "javascript": {
            "types": ["TypeError"],
            "messages": ["Cannot read property", "is undefined", "null is not an object"]
        },
        "java": {
            "types": ["NullPointerException"],
            "messages": []  # Any NullPointerException qualifies
        },
        "go": {
            "types": ["panic"],
            "messages": ["nil pointer dereference"]
        }
    }
    
    # Check if the error matches language-specific patterns
    if language in patterns:
        # Check error type
        if error_type in patterns[language]["types"]:
            # If type matches, check message patterns if any
            if not patterns[language]["messages"]:
                return True
            
            for pattern in patterns[language]["messages"]:
                if pattern in error_message:
                    return True
    
    return False
```

### Cross-Language Rule Application

```python
# Example: Applying a null check rule across languages
def apply_null_check_rule(error_data, fix_generator):
    """Apply a null check fix in any language"""
    language = error_data.get("language")
    variable_name = extract_variable_name(error_data)
    
    if not variable_name:
        return None
    
    templates = {
        "python": "if {variable} is None:\n    return {default_value}",
        "javascript": "if ({variable} === undefined || {variable} === null) {\n    return {default_value};\n}",
        "java": "if ({variable} == null) {\n    return {default_value};\n}",
        "go": "if {variable} == nil {\n    return {default_value}\n}"
    }
    
    default_values = {
        "python": "None",
        "javascript": "null",
        "java": "null",
        "go": "nil"
    }
    
    if language not in templates:
        return None
    
    template = templates[language]
    default_value = default_values[language]
    
    # Generate the fix using the template
    fix = template.format(variable=variable_name, default_value=default_value)
    
    return fix_generator.create_fix(
        error_data, 
        fix_type="null_check",
        code=fix,
        location=extract_location(error_data)
    )
```

## Configuration

Cross-language configuration is defined in `language_configs.json`:

```json
{
  "languages": [
    {
      "name": "python",
      "extensions": [".py"],
      "error_patterns": {
        "null_reference": {
          "error_types": ["AttributeError", "TypeError"],
          "message_patterns": ["'NoneType' object has no attribute", "NoneType' object is not"]
        },
        "index_error": {
          "error_types": ["IndexError", "KeyError"],
          "message_patterns": ["list index out of range", "key not found"]
        }
      }
    },
    {
      "name": "javascript",
      "extensions": [".js", ".ts", ".jsx", ".tsx"],
      "error_patterns": {
        "null_reference": {
          "error_types": ["TypeError"],
          "message_patterns": ["Cannot read property", "is undefined", "null is not an object"]
        },
        "index_error": {
          "error_types": ["TypeError", "RangeError"],
          "message_patterns": ["Cannot read property", "index out of bounds"]
        }
      }
    },
    {
      "name": "java",
      "extensions": [".java"],
      "error_patterns": {
        "null_reference": {
          "error_types": ["NullPointerException"],
          "message_patterns": []
        },
        "index_error": {
          "error_types": ["ArrayIndexOutOfBoundsException", "IndexOutOfBoundsException"],
          "message_patterns": []
        }
      }
    },
    {
      "name": "go",
      "extensions": [".go"],
      "error_patterns": {
        "null_reference": {
          "error_types": ["panic"],
          "message_patterns": ["nil pointer dereference"]
        },
        "index_error": {
          "error_types": ["panic"],
          "message_patterns": ["index out of range", "runtime error: index out of range"]
        }
      }
    }
  ],
  "shared_categories": [
    {
      "id": "null_reference",
      "description": "Null/nil/undefined reference errors",
      "fix_types": ["null_check", "default_value", "early_return"]
    },
    {
      "id": "index_error",
      "description": "Array/collection index out of bounds errors",
      "fix_types": ["bounds_check", "safe_access", "default_value"]
    }
  ]
}
```

## Best Practices

### Designing Cross-Language Rules

1. **Focus on Universal Patterns**: Target error patterns that exist across many languages
2. **Use Normalized Classifications**: Base rules on language-agnostic classifications
3. **Provide Language-Specific Implementations**: Create specialized templates for each language
4. **Consider Language Idioms**: Follow each language's best practices when generating fixes
5. **Test Across Languages**: Validate rules with tests in all supported languages

### Extending to New Languages

To add a new language:

1. Create a language adapter in `language_adapters.py`
2. Implement a language plugin in `plugins/`
3. Add language-specific rules in `rules/`
4. Update `language_configs.json`
5. Create templates for fixes
6. Add cross-language tests
7. Update metrics collection

## Conclusion

The cross-language orchestration capabilities of Homeostasis enable it to provide consistent error handling and healing across polyglot applications. By standardizing error representation, sharing patterns across languages, and translating fixes between languages, the system can leverage knowledge from one language to help fix issues in others.