"""
Fix Suggestion Module for Homeostasis.

This module provides functionality for human review and suggestion of fixes,
allowing users to review, modify, and provide feedback on automatically
generated fixes before they are deployed.
"""

from modules.suggestion.diff_viewer import create_diff, highlight_diff
from modules.suggestion.feedback import (FeedbackManager, FeedbackType,
                                         get_feedback_manager)
from modules.suggestion.knowledge_base import KnowledgeBase, get_knowledge_base
from modules.suggestion.ranking import rank_suggestions
from modules.suggestion.suggestion_manager import (FixSuggestion,
                                                   SuggestionManager,
                                                   SuggestionStatus,
                                                   get_suggestion_manager)

__all__ = [
    # Suggestion management
    "SuggestionManager",
    "FixSuggestion",
    "SuggestionStatus",
    "get_suggestion_manager",
    # Ranking
    "rank_suggestions",
    # Feedback
    "FeedbackManager",
    "FeedbackType",
    "get_feedback_manager",
    # Knowledge base
    "KnowledgeBase",
    "get_knowledge_base",
    # Diff viewing
    "create_diff",
    "highlight_diff",
]
