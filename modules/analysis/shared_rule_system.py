"""
Shared Rule System for Backend Languages

This module provides a language-agnostic rule system that allows rules to be
shared and applied across different programming languages. It enables pattern
recognition and fix generation that can work across language boundaries.
"""
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple, Union

from .shared_error_schema import SharedErrorSchema, normalize_error
from .language_plugin_system import get_plugin, get_all_plugins

logger = logging.getLogger(__name__)

# Directory for shared rules
SHARED_RULES_DIR = Path(__file__).parent / "rules" / "shared"
SHARED_RULES_DIR.mkdir(exist_ok=True, parents=True)


class SharedRule:
    """
    A language-agnostic rule that can be applied across different programming languages.
    """
    
    def __init__(self, 
                 rule_id: str,
                 name: str,
                 description: str,
                 pattern: Union[str, List[str]],
                 root_cause: str,
                 suggestion: str,
                 applicable_languages: Optional[List[str]] = None,
                 language_specific_patterns: Optional[Dict[str, str]] = None,
                 language_specific_suggestions: Optional[Dict[str, str]] = None,
                 confidence: str = "medium",
                 severity: str = "medium",
                 category: str = "cross_language",
                 tags: Optional[List[str]] = None,
                 examples: Optional[Dict[str, Any]] = None):
        """
        Initialize a shared rule.
        
        Args:
            rule_id: Unique identifier for the rule
            name: Short name of the rule
            description: Detailed description of the error and when the rule applies
            pattern: Regex pattern(s) for matching the error 
            root_cause: Root cause identifier
            suggestion: Generic suggestion for fixing the error
            applicable_languages: List of languages this rule can be applied to
            language_specific_patterns: Language-specific patterns to override the generic pattern
            language_specific_suggestions: Language-specific suggestions to override the generic suggestion
            confidence: Confidence level (low, medium, high)
            severity: Severity level (low, medium, high, critical)
            category: Rule category
            tags: List of tags for categorizing the rule
            examples: Examples of errors for which this rule applies
        """
        self.rule_id = rule_id
        self.name = name
        self.description = description
        self.pattern = pattern
        self.root_cause = root_cause
        self.suggestion = suggestion
        self.applicable_languages = applicable_languages or []
        self.language_specific_patterns = language_specific_patterns or {}
        self.language_specific_suggestions = language_specific_suggestions or {}
        self.confidence = confidence
        self.severity = severity
        self.category = category
        self.tags = tags or []
        self.examples = examples or {}
        
        # Pre-compile regex patterns for efficiency
        self.compiled_patterns = {}
        self._compile_patterns()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the rule to a dictionary for serialization."""
        return {
            "id": self.rule_id,
            "name": self.name,
            "description": self.description,
            "pattern": self.pattern,
            "root_cause": self.root_cause,
            "suggestion": self.suggestion,
            "applicable_languages": self.applicable_languages,
            "language_specific_patterns": self.language_specific_patterns,
            "language_specific_suggestions": self.language_specific_suggestions,
            "confidence": self.confidence,
            "severity": self.severity,
            "category": self.category,
            "tags": self.tags,
            "examples": self.examples
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SharedRule':
        """Create a rule from a dictionary."""
        return cls(
            rule_id=data.get("id", ""),
            name=data.get("name", ""),
            description=data.get("description", ""),
            pattern=data.get("pattern", ""),
            root_cause=data.get("root_cause", ""),
            suggestion=data.get("suggestion", ""),
            applicable_languages=data.get("applicable_languages"),
            language_specific_patterns=data.get("language_specific_patterns"),
            language_specific_suggestions=data.get("language_specific_suggestions"),
            confidence=data.get("confidence", "medium"),
            severity=data.get("severity", "medium"),
            category=data.get("category", "cross_language"),
            tags=data.get("tags"),
            examples=data.get("examples")
        )
    
    def applies_to_language(self, language: str) -> bool:
        """
        Check if this rule applies to a specific language.
        
        Args:
            language: Language identifier
            
        Returns:
            True if the rule applies to the language
        """
        if not self.applicable_languages:
            # If no languages specified, applies to all
            return True
        
        return language.lower() in [lang.lower() for lang in self.applicable_languages]
    
    def get_pattern_for_language(self, language: str) -> Union[str, List[str]]:
        """
        Get the pattern to use for a specific language.
        
        Args:
            language: Language identifier
            
        Returns:
            Pattern (string or list of strings)
        """
        language = language.lower()
        
        if language in self.language_specific_patterns:
            return self.language_specific_patterns[language]
        
        return self.pattern
    
    def get_suggestion_for_language(self, language: str) -> str:
        """
        Get the suggestion to use for a specific language.
        
        Args:
            language: Language identifier
            
        Returns:
            Suggestion string
        """
        language = language.lower()
        
        if language in self.language_specific_suggestions:
            return self.language_specific_suggestions[language]
        
        return self.suggestion
    
    def matches(self, error_data: Dict[str, Any], language: str) -> Tuple[bool, Optional[Tuple]]:
        """
        Check if this rule matches an error.
        
        Args:
            error_data: Error data to check
            language: Language of the error
            
        Returns:
            Tuple of (matches, match_groups)
        """
        # Skip if rule doesn't apply to this language
        if not self.applies_to_language(language):
            return False, None
        
        # Get the pattern to use
        pattern = self.get_pattern_for_language(language)
        
        # Get compiled patterns
        if isinstance(pattern, list):
            compiled_patterns = [self._get_compiled_pattern(p, language) for p in pattern]
        else:
            compiled_patterns = [self._get_compiled_pattern(pattern, language)]
        
        # Normalize error data for consistent matching
        try:
            standard_error = normalize_error(error_data, language)
        except Exception as e:
            logger.warning(f"Error normalizing data for rule matching: {e}")
            standard_error = error_data
        
        # Create a consolidated text for pattern matching
        match_text = self._create_match_text(standard_error)
        
        # Try each pattern
        for compiled_pattern in compiled_patterns:
            if compiled_pattern:
                match = compiled_pattern.search(match_text)
                if match:
                    return True, match.groups() if match.groups() else tuple()
        
        return False, None
    
    def _compile_patterns(self):
        """Compile regex patterns for all languages."""
        # Compile generic pattern
        if isinstance(self.pattern, list):
            for i, pattern in enumerate(self.pattern):
                key = f"generic_{i}"
                try:
                    self.compiled_patterns[key] = re.compile(pattern, re.IGNORECASE | re.DOTALL)
                except Exception as e:
                    logger.warning(f"Invalid pattern in rule {self.rule_id}: {e}")
        else:
            try:
                self.compiled_patterns["generic"] = re.compile(self.pattern, re.IGNORECASE | re.DOTALL)
            except Exception as e:
                logger.warning(f"Invalid pattern in rule {self.rule_id}: {e}")
        
        # Compile language-specific patterns
        for lang, pattern in self.language_specific_patterns.items():
            if isinstance(pattern, list):
                for i, p in enumerate(pattern):
                    key = f"{lang}_{i}"
                    try:
                        self.compiled_patterns[key] = re.compile(p, re.IGNORECASE | re.DOTALL)
                    except Exception as e:
                        logger.warning(f"Invalid pattern for {lang} in rule {self.rule_id}: {e}")
            else:
                try:
                    self.compiled_patterns[lang] = re.compile(pattern, re.IGNORECASE | re.DOTALL)
                except Exception as e:
                    logger.warning(f"Invalid pattern for {lang} in rule {self.rule_id}: {e}")
    
    def _get_compiled_pattern(self, pattern: str, language: str):
        """Get a compiled regex pattern for a specific language."""
        lang_key = language.lower()
        
        # Try language-specific pattern first
        if lang_key in self.compiled_patterns:
            return self.compiled_patterns[lang_key]
        
        # Then check if pattern is in a list for this language
        for key in self.compiled_patterns:
            if key.startswith(f"{lang_key}_"):
                return self.compiled_patterns[key]
        
        # Fall back to generic pattern
        if "generic" in self.compiled_patterns:
            return self.compiled_patterns["generic"]
        
        # Or any pattern in the generic list
        for key in self.compiled_patterns:
            if key.startswith("generic_"):
                return self.compiled_patterns[key]
        
        return None
    
    def _create_match_text(self, error_data: Dict[str, Any]) -> str:
        """
        Create a consolidated text for pattern matching from error data.
        
        Args:
            error_data: Error data
            
        Returns:
            Consolidated text for pattern matching
        """
        match_text = ""
        
        # Add error type and message
        if "error_type" in error_data:
            match_text += f"{error_data['error_type']}: "
        
        if "message" in error_data:
            match_text += f"{error_data['message']}\n"
        
        # Add stack trace
        if "stack_trace" in error_data:
            stack_trace = error_data["stack_trace"]
            
            if isinstance(stack_trace, list):
                if all(isinstance(frame, str) for frame in stack_trace):
                    match_text += "\n".join(stack_trace)
                elif all(isinstance(frame, dict) for frame in stack_trace):
                    # Flatten structured frames
                    for frame in stack_trace:
                        frame_text = []
                        for key, value in frame.items():
                            frame_text.append(f"{key}:{value}")
                        match_text += " ".join(frame_text) + "\n"
            elif isinstance(stack_trace, str):
                match_text += stack_trace
        
        return match_text


class SharedRuleRegistry:
    """
    Registry of shared rules that can be applied across different programming languages.
    """
    
    def __init__(self):
        """Initialize the shared rule registry."""
        self.rules: Dict[str, SharedRule] = {}
        self.schema = SharedErrorSchema()
    
    def add_rule(self, rule: SharedRule) -> None:
        """
        Add a shared rule to the registry.
        
        Args:
            rule: Shared rule to add
        """
        self.rules[rule.rule_id] = rule
    
    def get_rule(self, rule_id: str) -> Optional[SharedRule]:
        """
        Get a shared rule by ID.
        
        Args:
            rule_id: Rule identifier
            
        Returns:
            Shared rule or None if not found
        """
        return self.rules.get(rule_id)
    
    def get_rules_for_language(self, language: str) -> List[SharedRule]:
        """
        Get all rules applicable to a specific language.
        
        Args:
            language: Language identifier
            
        Returns:
            List of applicable shared rules
        """
        return [rule for rule in self.rules.values() 
                if rule.applies_to_language(language)]
    
    def match_rules(self, error_data: Dict[str, Any], language: str) -> List[Dict[str, Any]]:
        """
        Find all rules that match an error.
        
        Args:
            error_data: Error data to check
            language: Language of the error
            
        Returns:
            List of rule match results
        """
        matches = []
        
        # Get rules for this language
        applicable_rules = self.get_rules_for_language(language)
        
        for rule in applicable_rules:
            is_match, match_groups = rule.matches(error_data, language)
            
            if is_match:
                matches.append({
                    "rule": rule,
                    "match_groups": match_groups
                })
        
        return matches
    
    def match_and_generate(self, error_data: Dict[str, Any], 
                          language: str) -> List[Dict[str, Any]]:
        """
        Match rules and generate fixes for an error.
        
        Args:
            error_data: Error data to check
            language: Language of the error
            
        Returns:
            List of rule matches with suggestions
        """
        # Match rules
        matches = self.match_rules(error_data, language)
        results = []
        
        for match in matches:
            rule = match["rule"]
            match_groups = match["match_groups"]
            
            # Get language-specific suggestion
            suggestion = rule.get_suggestion_for_language(language)
            
            # Replace placeholders in suggestion with match groups
            if match_groups:
                for i, group in enumerate(match_groups):
                    placeholder = f"${{{i+1}}}"
                    suggestion = suggestion.replace(placeholder, str(group))
            
            results.append({
                "rule_id": rule.rule_id,
                "name": rule.name,
                "description": rule.description,
                "root_cause": rule.root_cause,
                "suggestion": suggestion,
                "confidence": rule.confidence,
                "severity": rule.severity,
                "category": rule.category
            })
        
        return results
    
    def load_rules_from_directory(self, directory: Optional[Union[str, Path]] = None) -> int:
        """
        Load shared rules from a directory.
        
        Args:
            directory: Directory path (defaults to the shared rules directory)
            
        Returns:
            Number of rules loaded
        """
        if directory is None:
            directory = SHARED_RULES_DIR
        
        directory = Path(directory)
        
        if not directory.exists():
            logger.warning(f"Rule directory does not exist: {directory}")
            return 0
        
        # Count loaded rules
        loaded_count = 0
        
        # Load all JSON files in the directory
        for file_path in directory.glob("*.json"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                    if isinstance(data, dict) and "rules" in data:
                        # File contains a list of rules
                        for rule_data in data["rules"]:
                            try:
                                rule = SharedRule.from_dict(rule_data)
                                self.add_rule(rule)
                                loaded_count += 1
                            except Exception as e:
                                logger.warning(f"Error loading rule from {file_path}: {e}")
                    else:
                        # File contains a single rule
                        rule = SharedRule.from_dict(data)
                        self.add_rule(rule)
                        loaded_count += 1
            except Exception as e:
                logger.warning(f"Error loading rules from {file_path}: {e}")
        
        logger.info(f"Loaded {loaded_count} shared rules")
        return loaded_count
    
    def save_rules_to_directory(self, directory: Optional[Union[str, Path]] = None) -> None:
        """
        Save all rules to individual files in a directory.
        
        Args:
            directory: Directory path (defaults to the shared rules directory)
        """
        if directory is None:
            directory = SHARED_RULES_DIR
        
        directory = Path(directory)
        directory.mkdir(exist_ok=True, parents=True)
        
        # Organize rules by category
        rules_by_category = {}
        
        for rule in self.rules.values():
            category = rule.category
            if category not in rules_by_category:
                rules_by_category[category] = []
            
            rules_by_category[category].append(rule.to_dict())
        
        # Save each category to a separate file
        for category, rules in rules_by_category.items():
            file_path = directory / f"{category}_rules.json"
            
            try:
                with open(file_path, 'w') as f:
                    json.dump({
                        "category": category,
                        "rules": rules
                    }, f, indent=2)
                
                logger.info(f"Saved {len(rules)} rules to {file_path}")
            except Exception as e:
                logger.error(f"Error saving rules to {file_path}: {e}")


# Create initial shared rules for common error patterns
def create_default_shared_rules() -> List[SharedRule]:
    """
    Create a set of default shared rules for common error patterns.
    
    Returns:
        List of default shared rules
    """
    rules = []
    
    # Null/nil pointer dereference
    rules.append(SharedRule(
        rule_id="shared_null_pointer",
        name="Null/Nil Pointer Dereference",
        description="Attempted to dereference a null or nil pointer/reference",
        pattern=[
            "(?:null pointer dereference)",
            "(?:nil pointer dereference)",
            "(?:NullPointerException)",
            "(?:null reference|undefined is not an object)"
        ],
        root_cause="null_pointer_dereference",
        suggestion="Check if the object is null/nil before accessing its properties or methods",
        applicable_languages=["java", "go", "python", "javascript", "csharp"],
        language_specific_patterns={
            "java": "(?:java\\.lang\\.NullPointerException)",
            "javascript": "(?:TypeError: Cannot read propert(?:y|ies) '[^']+' of (null|undefined))",
            "python": "(?:AttributeError: 'NoneType' object has no attribute '[^']+')",
            "go": "(?:nil pointer dereference)",
            "csharp": "(?:System\\.NullReferenceException)"
        },
        language_specific_suggestions={
            "java": "Add a null check: if (object != null) { ... }",
            "javascript": "Use optional chaining (object?.property) or check if object exists: if (object) { ... }",
            "python": "Check if object is not None before accessing attributes: if object is not None: ...",
            "go": "Add a nil check: if object != nil { ... }",
            "csharp": "Use null conditional operator: object?.Property or check with: if (object != null) { ... }"
        },
        confidence="high",
        severity="high",
        category="reference_errors",
        tags=["null", "nil", "reference", "pointer"]
    ))
    
    # Index out of bounds
    rules.append(SharedRule(
        rule_id="shared_index_out_of_bounds",
        name="Index Out of Bounds",
        description="Attempted to access an array/list/slice element with an invalid index",
        pattern=[
            "(?:index out of range|out of bounds|IndexOutOfBoundsException)",
            "(?:IndexError: list index out of range)"
        ],
        root_cause="index_out_of_bounds",
        suggestion="Validate the index before accessing the collection element",
        applicable_languages=["java", "go", "python", "javascript", "csharp"],
        language_specific_patterns={
            "java": "(?:java\\.lang\\.(?:ArrayIndexOutOfBoundsException|IndexOutOfBoundsException|StringIndexOutOfBoundsException))",
            "javascript": "(?:TypeError: Cannot read property '\\d+' of|undefined is not an object)",
            "python": "(?:IndexError: (?:list|string|tuple) index out of range)",
            "go": "(?:index out of range \\[(\\d+)\\] with length (\\d+))",
            "csharp": "(?:System\\.IndexOutOfRangeException)"
        },
        language_specific_suggestions={
            "java": "Add bounds checking: if (index >= 0 && index < array.length) { ... }",
            "javascript": "Check array bounds: if (index >= 0 && index < array.length) { ... }",
            "python": "Verify the index is valid: if 0 <= index < len(list_obj): ...",
            "go": "Check that the index is within bounds: if index >= 0 && index < len(slice) { ... }",
            "csharp": "Validate index before access: if (index >= 0 && index < array.Length) { ... }"
        },
        confidence="high",
        severity="medium",
        category="collection_errors",
        tags=["array", "list", "slice", "index", "bounds"]
    ))
    
    # Key not found in map/dictionary
    rules.append(SharedRule(
        rule_id="shared_key_not_found",
        name="Key Not Found",
        description="Attempted to access a map/dictionary/object with a non-existent key",
        pattern=[
            "(?:key not found|KeyError|no such element|key doesn't exist)"
        ],
        root_cause="key_not_found",
        suggestion="Check if the key exists before accessing the map/dictionary",
        applicable_languages=["java", "go", "python", "javascript", "csharp"],
        language_specific_patterns={
            "java": "(?:java\\.util\\.NoSuchElementException|java\\.lang\\.NullPointerException.*\\.get\\([^\\)]+\\))",
            "javascript": "(?:TypeError: Cannot read propert(?:y|ies) '[^']+' of|undefined is not an object)",
            "python": "(?:KeyError: '([^']+)')",
            "go": "(?:key not found in map|map has no key|map\\[[^\\]]+\\])",
            "csharp": "(?:System\\.Collections\\.Generic\\.KeyNotFoundException)"
        },
        language_specific_suggestions={
            "java": "Use containsKey() to check if the key exists, or use getOrDefault() to provide a fallback",
            "javascript": "Use optional chaining (obj?.prop) or check with: if (key in object) or hasOwnProperty()",
            "python": "Use dict.get(key) with a default value or check with: if key in dict_obj: ...",
            "go": "Use value, ok := map[key]; if ok { ... }",
            "csharp": "Use TryGetValue() or check with ContainsKey() before accessing"
        },
        confidence="high",
        severity="medium",
        category="collection_errors",
        tags=["map", "dict", "dictionary", "key"]
    ))
    
    # Type error / Type mismatch
    rules.append(SharedRule(
        rule_id="shared_type_error",
        name="Type Error",
        description="Operation performed on a value of the wrong type",
        pattern=[
            "(?:TypeError|ClassCastException|type mismatch|type error)"
        ],
        root_cause="type_mismatch",
        suggestion="Verify the type of the value before performing operations on it",
        applicable_languages=["java", "go", "python", "javascript", "csharp"],
        language_specific_patterns={
            "java": "(?:java\\.lang\\.ClassCastException: ([^\\s]+) cannot be cast to ([^\\s]+))",
            "javascript": "(?:TypeError: (?!Cannot read)([^\\n]+))",
            "python": "(?:TypeError: ([^\\n]+))",
            "go": "(?:cannot use .+ \\((?:type|value) .+\\) as .+ value)",
            "csharp": "(?:System\\.InvalidCastException|Cannot convert type)"
        },
        language_specific_suggestions={
            "java": "Use instanceof to check types before casting: if (obj instanceof TargetType) { ... }",
            "javascript": "Check variable types using typeof operator or instanceof for objects",
            "python": "Verify types with isinstance() or handle different types explicitly",
            "go": "Use type assertions carefully with comma-ok syntax: value, ok := x.(T)",
            "csharp": "Use the 'as' operator with null check, or 'is' operator before casting"
        },
        confidence="high",
        severity="medium",
        category="type_errors",
        tags=["type", "cast", "convert"]
    ))
    
    # Concurrent modification
    rules.append(SharedRule(
        rule_id="shared_concurrent_modification",
        name="Concurrent Modification",
        description="Collection was modified during iteration",
        pattern=[
            "(?:ConcurrentModificationException|concurrent map writes|collection was modified)"
        ],
        root_cause="concurrent_modification",
        suggestion="Use thread-safe collections or proper synchronization when modifying collections concurrently",
        applicable_languages=["java", "go", "python", "csharp"],
        language_specific_patterns={
            "java": "(?:java\\.util\\.ConcurrentModificationException)",
            "go": "(?:concurrent map (?:read and map )?writes?)",
            "python": "(?:RuntimeError: dictionary changed size during iteration)",
            "csharp": "(?:System\\.InvalidOperationException: Collection was modified)"
        },
        language_specific_suggestions={
            "java": "Use CopyOnWriteArrayList, ConcurrentHashMap, or synchronize access. Alternatively, use Iterator.remove() to modify during iteration.",
            "go": "Use a mutex to protect map access: var mu sync.Mutex or use sync.Map for concurrent operations",
            "python": "Create a copy of the dictionary before iterating: for key in list(my_dict.keys()): ...",
            "csharp": "Use a concurrent collection like ConcurrentDictionary, or create a copy before iterating"
        },
        confidence="high",
        severity="high",
        category="concurrency_errors",
        tags=["concurrent", "thread", "collection", "iteration"]
    ))
    
    return rules


# Create a global shared rule registry
shared_rule_registry = SharedRuleRegistry()


def add_shared_rule(rule: SharedRule) -> None:
    """
    Add a shared rule to the global registry.
    
    Args:
        rule: Shared rule to add
    """
    shared_rule_registry.add_rule(rule)


def get_shared_rule(rule_id: str) -> Optional[SharedRule]:
    """
    Get a shared rule from the global registry.
    
    Args:
        rule_id: Rule identifier
        
    Returns:
        Shared rule or None if not found
    """
    return shared_rule_registry.get_rule(rule_id)


def get_rules_for_language(language: str) -> List[SharedRule]:
    """
    Get all rules applicable to a specific language from the global registry.
    
    Args:
        language: Language identifier
        
    Returns:
        List of applicable shared rules
    """
    return shared_rule_registry.get_rules_for_language(language)


def match_shared_rules(error_data: Dict[str, Any], language: str) -> List[Dict[str, Any]]:
    """
    Find all shared rules that match an error.
    
    Args:
        error_data: Error data to check
        language: Language of the error
        
    Returns:
        List of rule match results
    """
    return shared_rule_registry.match_rules(error_data, language)


def match_and_generate(error_data: Dict[str, Any], language: str) -> List[Dict[str, Any]]:
    """
    Match shared rules and generate fixes for an error.
    
    Args:
        error_data: Error data to check
        language: Language of the error
        
    Returns:
        List of rule matches with suggestions
    """
    return shared_rule_registry.match_and_generate(error_data, language)


def load_shared_rules() -> int:
    """
    Load shared rules from the rules directory.
    
    Returns:
        Number of rules loaded
    """
    return shared_rule_registry.load_rules_from_directory()


def initialize_shared_rules() -> None:
    """Initialize the shared rule system with default rules."""
    # First, try to load existing rules
    loaded = load_shared_rules()
    
    # If no rules were loaded, create default rules
    if loaded == 0:
        logger.info("No shared rules found, creating defaults")
        default_rules = create_default_shared_rules()
        
        for rule in default_rules:
            add_shared_rule(rule)
        
        # Save the default rules
        shared_rule_registry.save_rules_to_directory()


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize the shared rule system
    initialize_shared_rules()
    
    # Display loaded rules
    rules_count = len(shared_rule_registry.rules)
    logger.info(f"Loaded {rules_count} shared rules")
    
    # Test rules with example errors
    example_errors = {
        "python_keyerror": {
            "language": "python",
            "exception_type": "KeyError",
            "message": "'user_id'",
            "traceback": [
                "Traceback (most recent call last):",
                "  File \"app.py\", line 42, in get_user",
                "    user_id = data['user_id']",
                "KeyError: 'user_id'"
            ]
        },
        "java_npe": {
            "language": "java",
            "exception_class": "java.lang.NullPointerException",
            "message": "Cannot invoke \"String.length()\" because \"str\" is null",
            "stack_trace": "java.lang.NullPointerException: Cannot invoke \"String.length()\" because \"str\" is null\n    at com.example.StringProcessor.processString(StringProcessor.java:42)"
        },
        "go_nil_pointer": {
            "language": "go",
            "error_type": "runtime error",
            "message": "nil pointer dereference",
            "stack_trace": "goroutine 1 [running]:\nmain.processValue()\n\t/app/main.go:25"
        },
        "javascript_typeerror": {
            "language": "javascript",
            "name": "TypeError",
            "message": "Cannot read property 'id' of undefined",
            "stack": "TypeError: Cannot read property 'id' of undefined\n    at getUserId (/app/src/utils.js:45:20)"
        }
    }
    
    for name, error in example_errors.items():
        # Match rules
        language = error.get("language", "unknown")
        matches = match_shared_rules(error, language)
        
        logger.info(f"\nTesting {name} ({language}):")
        if matches:
            logger.info(f"  Matched {len(matches)} rules:")
            for match in matches:
                rule = match["rule"]
                logger.info(f"  - {rule.name} ({rule.rule_id})")
        else:
            logger.info("  No rules matched")
        
        # Generate suggestions
        suggestions = match_and_generate(error, language)
        
        if suggestions:
            logger.info(f"  Generated {len(suggestions)} suggestions:")
            for suggestion in suggestions:
                logger.info(f"  - {suggestion['name']}: {suggestion['suggestion']}")