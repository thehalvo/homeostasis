"""
Enhanced confidence scoring for rule-based error analysis.
"""
import re
from typing import Dict, List, Any, Union
from dataclasses import dataclass
from enum import Enum

from .rule_config import Rule, RuleConfidence
from .rule_categories import EnhancedRule


class ConfidenceLevel(Enum):
    """Enumeration for confidence levels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class RuleMatch:
    """
    Information about a rule match, including confidence scoring.
    """
    rule: Union[Rule, EnhancedRule]
    match: re.Match
    context_score: float = 1.0
    specificity_score: float = 1.0
    strength_score: float = 1.0
    pattern_quality_score: float = 1.0
    
    @property
    def confidence_score(self) -> float:
        """Calculate the overall confidence score."""
        # Weighted average of different factors
        weights = {
            "context": 0.3,
            "specificity": 0.3,
            "strength": 0.25,
            "pattern_quality": 0.15
        }
        
        score = (
            weights["context"] * self.context_score +
            weights["specificity"] * self.specificity_score +
            weights["strength"] * self.strength_score +
            weights["pattern_quality"] * self.pattern_quality_score
        )
        
        return score
    
    @property
    def confidence_level(self) -> RuleConfidence:
        """Convert score to a confidence level."""
        score = self.confidence_score
        
        if score >= 0.8:
            return RuleConfidence.HIGH
        elif score >= 0.5:
            return RuleConfidence.MEDIUM
        else:
            return RuleConfidence.LOW


class ConfidenceScorer:
    """
    Calculates confidence scores for rule matches.
    """
    
    @staticmethod
    def score_match(rule: Union[Rule, EnhancedRule], match: re.Match, 
                   error_context: Dict[str, Any] = None) -> RuleMatch:
        """
        Score a match based on various factors.
        
        Args:
            rule: The rule that matched
            match: The regex match object
            error_context: Additional context about the error
            
        Returns:
            RuleMatch with calculated scores
        """
        error_context = error_context or {}
        
        # Create base match object
        rule_match = RuleMatch(rule=rule, match=match)
        
        # Calculate individual scores
        rule_match.context_score = ConfidenceScorer._calculate_context_score(rule, error_context)
        rule_match.specificity_score = ConfidenceScorer._calculate_specificity_score(rule, match)
        rule_match.strength_score = ConfidenceScorer._calculate_strength_score(match)
        rule_match.pattern_quality_score = ConfidenceScorer._calculate_pattern_quality(rule)
        
        return rule_match
    
    @staticmethod
    def _calculate_context_score(rule: Union[Rule, EnhancedRule], 
                               error_context: Dict[str, Any]) -> float:
        """
        Calculate context-based confidence score.
        
        Args:
            rule: The rule being scored
            error_context: Information about the error's context
            
        Returns:
            Score between 0.0 and 1.0
        """
        score = 0.7  # Default medium-high score
        
        # If there's a type match between rule and error
        if "exception_type" in error_context and error_context["exception_type"] == rule.type:
            score += 0.2
        
        # If there's a frame/line context that can help verify
        if "detailed_frames" in error_context and error_context["detailed_frames"]:
            score += 0.1
            
            # Check if any frames contain information that strengthens the match
            frames = error_context["detailed_frames"]
            for frame in frames:
                # If there's local variable context that supports the rule
                if "locals" in frame and isinstance(frame["locals"], dict):
                    locals_dict = frame["locals"]
                    
                    # For KeyError, check if the key is mentioned in locals
                    if rule.type == "KeyError" and rule.match and rule.match.groups():
                        key_name = rule.match.group(1).strip("'\"")
                        # Check if key exists in any dict in locals
                        for var_name, var_value in locals_dict.items():
                            if isinstance(var_value, dict) and key_name in var_value:
                                score += 0.1
                                break
                
                # If function name gives context
                if "function" in frame:
                    func_name = frame["function"].lower()
                    relevant_keywords = ["get", "fetch", "load", "find", "retrieve", "access", "read"]
                    
                    if any(keyword in func_name for keyword in relevant_keywords):
                        score += 0.05
        
        # Cap the score at 1.0
        return min(score, 1.0)
    
    @staticmethod
    def _calculate_specificity_score(rule: Union[Rule, EnhancedRule], match: re.Match) -> float:
        """
        Calculate specificity-based confidence score.
        
        Args:
            rule: The rule being scored
            match: The regex match
            
        Returns:
            Score between 0.0 and 1.0
        """
        # Base score
        score = 0.6
        
        # Check if the rule captures specific data
        if match.groups():
            # More groups suggests a more specific pattern
            group_count = len(match.groups())
            score += min(0.1 * group_count, 0.3)
        
        # Check for anchors in the pattern that make it more specific
        pattern = rule.pattern
        if pattern.startswith('^'):
            score += 0.05
        if pattern.endswith('$'):
            score += 0.05
        
        # Check for word boundaries
        if r'\b' in pattern:
            score += 0.05
        
        # Calculate the ratio of literal text to pattern length
        # More literal text generally means higher specificity
        literal_text = re.sub(r'\\.|\\[dDwWsS]|\[[^\]]*\]|\([^)]*\)|[.+*?{}]', '', pattern)
        if len(pattern) > 0:
            literal_ratio = len(literal_text) / len(pattern)
            score += literal_ratio * 0.2
        
        # Cap the score at 1.0
        return min(score, 1.0)
    
    @staticmethod
    def _calculate_strength_score(match: re.Match) -> float:
        """
        Calculate strength-based confidence score.
        
        Args:
            match: The regex match
            
        Returns:
            Score between 0.0 and 1.0
        """
        # Start with a base score
        score = 0.5
        
        # Longer matches generally indicate stronger evidence
        match_length = len(match.group(0))
        if match_length > 50:
            score += 0.3
        elif match_length > 30:
            score += 0.2
        elif match_length > 10:
            score += 0.1
        
        # Full line matches are stronger signals
        full_string = match.string
        if match.group(0) == full_string:
            score += 0.2
        elif len(match.group(0)) / len(full_string) > 0.8:
            score += 0.1
        
        # Cap the score at 1.0
        return min(score, 1.0)
    
    @staticmethod
    def _calculate_pattern_quality(rule: Union[Rule, EnhancedRule]) -> float:
        """
        Calculate pattern quality based confidence score.
        
        Args:
            rule: The rule being scored
            
        Returns:
            Score between 0.0 and 1.0
        """
        # Start with a base score
        score = 0.7
        
        # Check if the rule has examples
        if hasattr(rule, 'examples') and rule.examples:
            # More examples suggest a better vetted pattern
            score += min(0.05 * len(rule.examples), 0.2)
        
        # Check for overly generic patterns that might lead to false positives
        pattern = rule.pattern
        if pattern in [r'.*', r'.+', r'Error', r'Exception']:
            score -= 0.5
        elif len(pattern) < 5:
            score -= 0.3
        
        # Very complex patterns might be error-prone
        complexity_indicators = ['(?:', '(?=', '(?!', '(?<=', '(?<!', '(?P<']
        complexity_count = sum(pattern.count(ind) for ind in complexity_indicators)
        if complexity_count > 3:
            score -= 0.1 * complexity_count
        
        # Cap the score at 1.0 and ensure it's not negative
        return max(0.1, min(score, 1.0))


class RuleMatchRanker:
    """
    Ranks and filters rule matches based on confidence and other factors.
    """
    
    @staticmethod
    def rank_matches(matches: List[RuleMatch]) -> List[RuleMatch]:
        """
        Rank matches by confidence score.
        
        Args:
            matches: List of rule matches
            
        Returns:
            Sorted list with highest confidence matches first
        """
        return sorted(matches, key=lambda m: m.confidence_score, reverse=True)
    
    @staticmethod
    def filter_low_confidence(matches: List[RuleMatch], 
                             min_score: float = 0.5) -> List[RuleMatch]:
        """
        Filter matches that don't meet the minimum confidence threshold.
        
        Args:
            matches: List of rule matches
            min_score: Minimum confidence score to include
            
        Returns:
            Filtered list of matches
        """
        return [m for m in matches if m.confidence_score >= min_score]
    
    @staticmethod
    def get_best_matches(matches: List[RuleMatch], top_n: int = 3) -> List[RuleMatch]:
        """
        Get the top N matches by confidence score.
        
        Args:
            matches: List of rule matches
            top_n: Number of top matches to return
            
        Returns:
            List of top N matches
        """
        ranked = RuleMatchRanker.rank_matches(matches)
        return ranked[:top_n]
    
    @staticmethod
    def resolve_conflicts(matches: List[RuleMatch]) -> List[RuleMatch]:
        """
        Resolve conflicts between matches covering the same parts of the input.
        
        Args:
            matches: List of rule matches
            
        Returns:
            List with conflicts resolved
        """
        # Sort by confidence
        ranked = RuleMatchRanker.rank_matches(matches)
        
        # Track which spans are covered by higher confidence matches
        covered_spans = []
        resolved_matches = []
        
        for match in ranked:
            match_span = match.match.span()
            
            # Check if this match significantly overlaps with a higher confidence match
            is_covered = False
            for span in covered_spans:
                # Calculate overlap percentage
                overlap_start = max(match_span[0], span[0])
                overlap_end = min(match_span[1], span[1])
                
                if overlap_start < overlap_end:
                    overlap_length = overlap_end - overlap_start
                    match_length = match_span[1] - match_span[0]
                    
                    # If more than 70% of this match is covered by a higher confidence match
                    if overlap_length / match_length > 0.7:
                        is_covered = True
                        break
            
            if not is_covered:
                resolved_matches.append(match)
                covered_spans.append(match_span)
        
        return resolved_matches


class ContextualRuleAnalyzer:
    """
    Analyzes errors with enhanced contextual analysis and confidence scoring.
    """
    
    def __init__(self, rules: List[Union[Rule, EnhancedRule]]):
        """
        Initialize with a list of rules.
        
        Args:
            rules: List of rules for error analysis
        """
        self.rules = rules
    
    def analyze_error(self, error_text: str, error_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyze an error with confidence scoring.
        
        Args:
            error_text: Text of the error message
            error_context: Additional context about the error
            
        Returns:
            Analysis results with confidence scores
        """
        error_context = error_context or {}
        matches = []
        
        # Look for rule matches
        for rule in self.rules:
            match = rule.matches(error_text)
            if match:
                # Score the match
                rule_match = ConfidenceScorer.score_match(rule, match, error_context)
                matches.append(rule_match)
        
        # No matches
        if not matches:
            return {
                "matched": False,
                "confidence": "low",
                "confidence_score": 0.0,
                "analysis": "No rule patterns matched this error"
            }
        
        # Rank and filter matches
        ranked_matches = RuleMatchRanker.rank_matches(matches)
        top_matches = RuleMatchRanker.get_best_matches(ranked_matches)
        resolved_matches = RuleMatchRanker.resolve_conflicts(top_matches)
        
        # Get the best match
        best_match = resolved_matches[0] if resolved_matches else ranked_matches[0]
        
        # Prepare alternative matches if available
        alternatives = []
        for match in resolved_matches[1:] if len(resolved_matches) > 1 else []:
            alternatives.append({
                "rule_id": match.rule.id,
                "description": match.rule.description,
                "confidence_score": match.confidence_score,
                "confidence": match.confidence_level.value
            })
        
        # Convert confidence score to level
        confidence_level = best_match.confidence_level.value
        
        # Prepare result
        result = {
            "matched": True,
            "rule_id": best_match.rule.id,
            "pattern": best_match.rule.pattern,
            "matched_text": best_match.match.group(0),
            "type": best_match.rule.type,
            "description": best_match.rule.description,
            "root_cause": best_match.rule.root_cause,
            "suggestion": best_match.rule.suggestion,
            "match_groups": best_match.match.groups() if best_match.match.groups() else None,
            "confidence": confidence_level,
            "confidence_score": best_match.confidence_score,
            "confidence_factors": {
                "context_score": best_match.context_score,
                "specificity_score": best_match.specificity_score,
                "strength_score": best_match.strength_score,
                "pattern_quality_score": best_match.pattern_quality_score
            }
        }
        
        # Add category and severity if available
        if hasattr(best_match.rule, 'category'):
            result["category"] = best_match.rule.category.value
        
        if hasattr(best_match.rule, 'severity'):
            result["severity"] = best_match.rule.severity.value
        
        # Add alternative matches if available
        if alternatives:
            result["alternative_matches"] = alternatives
        
        return result


if __name__ == "__main__":
    # Example usage
    from rule_config import Rule, RuleCategory, RuleSeverity, RuleConfidence
    
    # Create sample rules
    rule1 = Rule(
        pattern=r"KeyError: '([^']*)'",
        type="KeyError",
        description="Dictionary key not found",
        root_cause="dict_key_not_exists",
        suggestion="Check if the key exists before accessing it",
        category=RuleCategory.PYTHON,
        severity=RuleSeverity.MEDIUM,
        confidence=RuleConfidence.HIGH
    )
    
    rule2 = Rule(
        pattern=r"Error: could not find key '([^']*)'",
        type="KeyError",
        description="Dictionary key lookup failed",
        root_cause="dict_key_not_found",
        suggestion="Validate dictionary keys before access",
        category=RuleCategory.PYTHON,
        severity=RuleSeverity.MEDIUM,
        confidence=RuleConfidence.MEDIUM
    )
    
    # Create analyzer
    analyzer = ContextualRuleAnalyzer([rule1, rule2])
    
    # Test error
    error_text = "KeyError: 'user_id'"
    error_context = {
        "exception_type": "KeyError",
        "detailed_frames": [
            {
                "file": "app.py",
                "line": 42,
                "function": "get_user",
                "locals": {
                    "request_data": {"username": "test", "email": "test@example.com"}
                }
            }
        ]
    }
    
    # Analyze
    result = analyzer.analyze_error(error_text, error_context)
    
    # Print results
    print("Analysis Result:")
    print(f"Rule ID: {result.get('rule_id')}")
    print(f"Type: {result.get('type')}")
    print(f"Root Cause: {result.get('root_cause')}")
    print(f"Suggestion: {result.get('suggestion')}")
    print(f"Confidence: {result.get('confidence')}")
    print(f"Confidence Score: {result.get('confidence_score'):.2f}")
    
    print("\nConfidence Factors:")
    for factor, score in result.get('confidence_factors', {}).items():
        print(f"  {factor}: {score:.2f}")
    
    # Test with competing matches
    ambiguous_error = "Error: could not find key 'user_id'"
    
    # Analyze ambiguous error
    ambiguous_result = analyzer.analyze_error(ambiguous_error, error_context)
    
    print("\nAmbiguous Error Analysis:")
    print(f"Rule ID: {ambiguous_result.get('rule_id')}")
    print(f"Type: {ambiguous_result.get('type')}")
    print(f"Confidence: {ambiguous_result.get('confidence')}")
    print(f"Confidence Score: {ambiguous_result.get('confidence_score'):.2f}")