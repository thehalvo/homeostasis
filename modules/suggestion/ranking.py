"""
Ranking module for fix suggestions.

Provides algorithms for ranking and scoring fix suggestions based on various factors.
"""

import logging
import re

logger = logging.getLogger(__name__)


def rank_suggestions(suggestions):
    """Rank a list of fix suggestions.
    
    Args:
        suggestions: List of FixSuggestion objects
        
    Returns:
        List of FixSuggestion objects, sorted by ranking score (descending)
    """
    # Calculate ranking scores for each suggestion
    for suggestion in suggestions:
        suggestion.ranking_score = calculate_ranking_score(suggestion)
        
    # Sort by ranking score (descending)
    return sorted(suggestions, key=lambda s: s.ranking_score, reverse=True)


def calculate_ranking_score(suggestion) -> float:
    """Calculate a ranking score for a fix suggestion.
    
    Args:
        suggestion: FixSuggestion object
        
    Returns:
        float: Ranking score (0-1)
    """
    # Start with the confidence score as the base
    score = suggestion.confidence
    
    # Adjust based on various factors
    
    # 1. Adjust based on source (prefer human-validated fixes)
    source_factor = {
        "manual": 0.2,    # Human-created fixes get a bonus
        "hybrid": 0.1,    # Human-modified fixes get a small bonus
        "auto": 0.0       # No adjustment for purely automatic fixes
    }
    score += source_factor.get(suggestion.source, 0.0)
    
    # 2. Adjust based on code complexity - prefer simpler fixes
    complexity_penalty = calculate_complexity_penalty(suggestion)
    score -= complexity_penalty
    
    # 3. Adjust based on fix type - prefer certain types of fixes
    fix_type_factor = {
        # Some fix types might be more reliable than others
        "null_check": 0.05,          # Null checks are usually safe
        "parameter_check": 0.05,     # Parameter checks are usually safe
        "exception_handling": 0.03,  # Exception handling is useful
        "early_return": 0.02,        # Early returns can be good
        "retrying": -0.05,           # Retrying might mask deeper issues
    }
    score += fix_type_factor.get(suggestion.fix_type, 0.0)
    
    # 4. Cap the score at a maximum of 1.0
    score = min(score, 1.0)
    
    # 5. Ensure the score is at least 0.0
    score = max(score, 0.0)
    
    return score


def calculate_complexity_penalty(suggestion) -> float:
    """Calculate a complexity penalty for a fix suggestion.
    
    Args:
        suggestion: FixSuggestion object
        
    Returns:
        float: Complexity penalty (0-0.3)
    """
    # Calculate the size difference between original and suggested code
    original_lines = suggestion.original_code.count('\n')
    suggested_lines = suggestion.suggested_code.count('\n')
    
    # Penalize fixes that add too many lines
    lines_diff = suggested_lines - original_lines
    size_penalty = min(0.1, max(0, lines_diff / 20))
    
    # Penalize complex expressions
    complex_expressions = [
        r'lambda',                     # Lambda expressions
        r'if\s+.+?\s+else',            # Inline conditionals
        r'try\s*:',                    # Exception handling
        r'[^a-zA-Z0-9_]and[^a-zA-Z0-9_]|[^a-zA-Z0-9_]or[^a-zA-Z0-9_]',  # Complex boolean logic
        r'\[\s*for\s+.+?\s+in\s+',     # List comprehensions
        r'\{\s*(.+?)\s*:\s*(.+?)\s+for\s+',  # Dict comprehensions
    ]
    
    complexity_score = 0.0
    for pattern in complex_expressions:
        matches = len(re.findall(pattern, suggestion.suggested_code))
        complexity_score += matches * 0.02
        
    # Cap complexity score
    complexity_score = min(0.2, complexity_score)
    
    # Combine penalties
    total_penalty = size_penalty + complexity_score
    
    return min(0.3, total_penalty)  # Cap at 0.3 to avoid excessive penalties