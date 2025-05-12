"""
Suggestion Manager for Homeostasis.

Manages the generation, ranking, and tracking of fix suggestions for human review.
"""

import datetime
import enum
import json
import logging
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union, Any

from modules.suggestion.ranking import rank_suggestions
from modules.patch_generation.patcher import PatchGenerator

logger = logging.getLogger(__name__)


class SuggestionStatus(enum.Enum):
    """Status of a fix suggestion."""
    GENERATED = "generated"  # Just generated, not yet reviewed
    REVIEWING = "reviewing"  # Under review by a human
    APPROVED = "approved"    # Approved by a human
    REJECTED = "rejected"    # Rejected by a human
    MODIFIED = "modified"    # Modified by a human
    DEPLOYED = "deployed"    # Deployed to the system
    FAILED = "failed"        # Failed after deployment


@dataclass
class FixSuggestion:
    """Represents a fix suggestion."""
    suggestion_id: str
    error_id: str
    fix_type: str
    confidence: float
    source: str  # "auto", "manual", "hybrid"
    file_path: str
    original_code: str
    suggested_code: str
    description: str
    metadata: Dict[str, Any]  # Additional metadata
    created_at: str
    modified_at: Optional[str] = None
    status: SuggestionStatus = SuggestionStatus.GENERATED
    reviewer: Optional[str] = None
    review_comments: Optional[str] = None
    ranking_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return {
            "suggestion_id": self.suggestion_id,
            "error_id": self.error_id,
            "fix_type": self.fix_type,
            "confidence": self.confidence,
            "source": self.source,
            "file_path": self.file_path,
            "original_code": self.original_code,
            "suggested_code": self.suggested_code,
            "description": self.description,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "modified_at": self.modified_at,
            "status": self.status.value,
            "reviewer": self.reviewer,
            "review_comments": self.review_comments,
            "ranking_score": self.ranking_score,
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FixSuggestion':
        """Create from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            FixSuggestion: New instance
        """
        # Convert status string to enum
        status_str = data.get("status", "generated")
        status = SuggestionStatus(status_str)
        
        return cls(
            suggestion_id=data["suggestion_id"],
            error_id=data["error_id"],
            fix_type=data["fix_type"],
            confidence=data["confidence"],
            source=data["source"],
            file_path=data["file_path"],
            original_code=data["original_code"],
            suggested_code=data["suggested_code"],
            description=data["description"],
            metadata=data.get("metadata", {}),
            created_at=data["created_at"],
            modified_at=data.get("modified_at"),
            status=status,
            reviewer=data.get("reviewer"),
            review_comments=data.get("review_comments"),
            ranking_score=data.get("ranking_score", 0.0),
        )


class SuggestionManager:
    """
    Manages fix suggestions for human review.
    
    Generates, stores, and manages fix suggestions, allowing humans to review,
    modify, and provide feedback on automatically generated fixes.
    """
    
    def __init__(self, storage_dir: Optional[str] = None, config: Dict[str, Any] = None):
        """Initialize the suggestion manager.
        
        Args:
            storage_dir: Directory to store suggestion data
            config: Configuration dictionary
        """
        self.config = config or {}
        self.storage_dir = Path(storage_dir or "logs/suggestions")
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Initialize patch generator
        self.patch_generator = PatchGenerator()
        
        # Load existing suggestions
        self.suggestions = {}  # error_id -> List[FixSuggestion]
        self._load_suggestions()
        
        logger.info(f"Initialized suggestion manager with storage directory: {self.storage_dir}")
        
    def _load_suggestions(self) -> None:
        """Load existing suggestions from storage."""
        try:
            # Look for suggestion files
            for file_path in self.storage_dir.glob("*.json"):
                try:
                    with open(file_path, "r") as f:
                        data = json.load(f)
                        
                    if "error_id" not in data or "suggestions" not in data:
                        logger.warning(f"Invalid suggestion file: {file_path}")
                        continue
                        
                    error_id = data["error_id"]
                    suggestions = []
                    
                    for suggestion_data in data["suggestions"]:
                        try:
                            suggestion = FixSuggestion.from_dict(suggestion_data)
                            suggestions.append(suggestion)
                        except (KeyError, ValueError) as e:
                            logger.warning(f"Error loading suggestion: {e}")
                            continue
                            
                    if suggestions:
                        self.suggestions[error_id] = suggestions
                        logger.debug(f"Loaded {len(suggestions)} suggestions for error {error_id}")
                        
                except Exception as e:
                    logger.warning(f"Error loading suggestion file {file_path}: {e}")
                    
            logger.info(f"Loaded suggestions for {len(self.suggestions)} errors")
            
        except Exception as e:
            logger.error(f"Error loading suggestions: {e}")
            
    def _save_suggestions(self, error_id: str) -> None:
        """Save suggestions for an error.
        
        Args:
            error_id: Error ID to save suggestions for
        """
        if error_id not in self.suggestions:
            return
            
        suggestions = self.suggestions[error_id]
        
        try:
            file_path = self.storage_dir / f"{error_id}.json"
            
            data = {
                "error_id": error_id,
                "suggestions": [suggestion.to_dict() for suggestion in suggestions]
            }
            
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)
                
            logger.debug(f"Saved {len(suggestions)} suggestions for error {error_id}")
            
        except Exception as e:
            logger.error(f"Error saving suggestions for error {error_id}: {e}")
            
    def generate_suggestions(self, error_id: str, error_data: Dict[str, Any],
                            max_suggestions: int = 3) -> List[FixSuggestion]:
        """Generate fix suggestions for an error.
        
        Args:
            error_id: Error ID
            error_data: Error data including stack trace, context, etc.
            max_suggestions: Maximum number of suggestions to generate
            
        Returns:
            List[FixSuggestion]: Generated suggestions
        """
        logger.info(f"Generating suggestions for error {error_id}")
        
        # Check if we already have suggestions for this error
        if error_id in self.suggestions:
            existing_suggestions = self.suggestions[error_id]
            if len(existing_suggestions) >= max_suggestions:
                logger.info(f"Already have {len(existing_suggestions)} suggestions for error {error_id}")
                return existing_suggestions
                
        # Generate patches using the patch generator
        try:
            file_path = error_data.get("file_path")
            error_type = error_data.get("error_type")
            error_message = error_data.get("error_message")
            line_number = error_data.get("line_number")
            
            if not file_path or not error_type:
                logger.error(f"Missing required error data for error {error_id}")
                return []
                
            # Get original code
            try:
                original_code = self._get_file_content(file_path)
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {e}")
                return []
                
            # Generate patches
            # In a real implementation, this would call the patch generator with the error details
            # For now, we'll simulate generating multiple fix suggestions
            suggestions = []
            
            # Create a simulated primary fix
            primary_fix = self._create_fix_suggestion(
                error_id=error_id,
                file_path=file_path,
                original_code=original_code,
                # In a real implementation, this would be the actual patched code
                suggested_code=original_code.replace("def process_data(data):", 
                                                    "def process_data(data=None):\n    if data is None:\n        data = {}\n"),
                fix_type="null_check",
                confidence=0.95,
                description="Added null check and default parameter value",
                source="auto"
            )
            suggestions.append(primary_fix)
            
            # Create alternative fixes with lower confidence
            if max_suggestions > 1:
                alt_fix1 = self._create_fix_suggestion(
                    error_id=error_id,
                    file_path=file_path,
                    original_code=original_code,
                    suggested_code=original_code.replace("def process_data(data):", 
                                                        "def process_data(data):\n    if data is None:\n        return None\n"),
                    fix_type="early_return",
                    confidence=0.8,
                    description="Added early return for null data",
                    source="auto"
                )
                suggestions.append(alt_fix1)
                
            if max_suggestions > 2:
                alt_fix2 = self._create_fix_suggestion(
                    error_id=error_id,
                    file_path=file_path,
                    original_code=original_code,
                    suggested_code=original_code.replace("def process_data(data):", 
                                                        "def process_data(data):\n    data = data or {}\n"),
                    fix_type="null_coalescing",
                    confidence=0.7,
                    description="Added null coalescing operator",
                    source="auto"
                )
                suggestions.append(alt_fix2)
                
            # Rank the suggestions
            ranked_suggestions = rank_suggestions(suggestions)
            
            # Store the suggestions
            self.suggestions[error_id] = ranked_suggestions
            self._save_suggestions(error_id)
            
            logger.info(f"Generated {len(ranked_suggestions)} suggestions for error {error_id}")
            return ranked_suggestions
            
        except Exception as e:
            logger.error(f"Error generating suggestions for error {error_id}: {e}")
            return []
            
    def _create_fix_suggestion(self, error_id: str, file_path: str, 
                              original_code: str, suggested_code: str,
                              fix_type: str, confidence: float, 
                              description: str, source: str) -> FixSuggestion:
        """Create a fix suggestion.
        
        Args:
            error_id: Error ID
            file_path: Path to the file being fixed
            original_code: Original code
            suggested_code: Suggested fixed code
            fix_type: Type of fix
            confidence: Confidence score (0-1)
            description: Description of the fix
            source: Source of the fix ("auto", "manual", "hybrid")
            
        Returns:
            FixSuggestion: New fix suggestion
        """
        suggestion_id = str(uuid.uuid4())
        now = datetime.datetime.utcnow().isoformat()
        
        return FixSuggestion(
            suggestion_id=suggestion_id,
            error_id=error_id,
            fix_type=fix_type,
            confidence=confidence,
            source=source,
            file_path=file_path,
            original_code=original_code,
            suggested_code=suggested_code,
            description=description,
            metadata={},
            created_at=now,
            status=SuggestionStatus.GENERATED,
            ranking_score=confidence,  # Initial ranking based on confidence
        )
        
    def _get_file_content(self, file_path: str) -> str:
        """Get the content of a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            str: File content
            
        Raises:
            FileNotFoundError: If file does not exist
        """
        with open(file_path, "r") as f:
            return f.read()
            
    def get_suggestions(self, error_id: str) -> List[FixSuggestion]:
        """Get all suggestions for an error.
        
        Args:
            error_id: Error ID
            
        Returns:
            List[FixSuggestion]: Suggestions for the error
        """
        return self.suggestions.get(error_id, [])
        
    def get_best_suggestion(self, error_id: str) -> Optional[FixSuggestion]:
        """Get the best suggestion for an error.
        
        Args:
            error_id: Error ID
            
        Returns:
            Optional[FixSuggestion]: Best suggestion or None
        """
        suggestions = self.get_suggestions(error_id)
        if not suggestions:
            return None
            
        # Sort by ranking score (descending)
        sorted_suggestions = sorted(suggestions, key=lambda s: s.ranking_score, reverse=True)
        return sorted_suggestions[0]
        
    def get_suggestion(self, suggestion_id: str) -> Optional[FixSuggestion]:
        """Get a specific suggestion by ID.
        
        Args:
            suggestion_id: Suggestion ID
            
        Returns:
            Optional[FixSuggestion]: Suggestion or None
        """
        for error_id, suggestions in self.suggestions.items():
            for suggestion in suggestions:
                if suggestion.suggestion_id == suggestion_id:
                    return suggestion
                    
        return None
        
    def update_suggestion(self, suggestion: FixSuggestion) -> bool:
        """Update a suggestion.
        
        Args:
            suggestion: Updated suggestion
            
        Returns:
            bool: True if updated successfully
        """
        suggestion_id = suggestion.suggestion_id
        error_id = suggestion.error_id
        
        if error_id not in self.suggestions:
            logger.warning(f"Error ID {error_id} not found")
            return False
            
        # Update suggestion in the list
        for i, existing in enumerate(self.suggestions[error_id]):
            if existing.suggestion_id == suggestion_id:
                suggestion.modified_at = datetime.datetime.utcnow().isoformat()
                self.suggestions[error_id][i] = suggestion
                self._save_suggestions(error_id)
                logger.info(f"Updated suggestion {suggestion_id} for error {error_id}")
                return True
                
        logger.warning(f"Suggestion ID {suggestion_id} not found for error {error_id}")
        return False
        
    def modify_suggestion(self, suggestion_id: str, suggested_code: str, 
                         reviewer: str, comments: Optional[str] = None) -> Optional[FixSuggestion]:
        """Modify a suggestion.
        
        Args:
            suggestion_id: Suggestion ID
            suggested_code: New suggested code
            reviewer: Username of the reviewer
            comments: Optional review comments
            
        Returns:
            Optional[FixSuggestion]: Updated suggestion or None
        """
        suggestion = self.get_suggestion(suggestion_id)
        if not suggestion:
            logger.warning(f"Suggestion ID {suggestion_id} not found")
            return None
            
        # Update the suggestion
        suggestion.suggested_code = suggested_code
        suggestion.reviewer = reviewer
        suggestion.review_comments = comments
        suggestion.status = SuggestionStatus.MODIFIED
        suggestion.source = "hybrid"  # Changed to hybrid since it's now human-modified
        suggestion.modified_at = datetime.datetime.utcnow().isoformat()
        
        # Update and save
        if self.update_suggestion(suggestion):
            return suggestion
        return None
        
    def review_suggestion(self, suggestion_id: str, status: SuggestionStatus,
                         reviewer: str, comments: Optional[str] = None) -> Optional[FixSuggestion]:
        """Review a suggestion.
        
        Args:
            suggestion_id: Suggestion ID
            status: New status (APPROVED, REJECTED)
            reviewer: Username of the reviewer
            comments: Optional review comments
            
        Returns:
            Optional[FixSuggestion]: Updated suggestion or None
        """
        if status not in [SuggestionStatus.APPROVED, SuggestionStatus.REJECTED]:
            logger.warning(f"Invalid review status: {status}")
            return None
            
        suggestion = self.get_suggestion(suggestion_id)
        if not suggestion:
            logger.warning(f"Suggestion ID {suggestion_id} not found")
            return None
            
        # Update the suggestion
        suggestion.status = status
        suggestion.reviewer = reviewer
        suggestion.review_comments = comments
        suggestion.modified_at = datetime.datetime.utcnow().isoformat()
        
        # Update and save
        if self.update_suggestion(suggestion):
            return suggestion
        return None
        
    def mark_deployed(self, suggestion_id: str) -> Optional[FixSuggestion]:
        """Mark a suggestion as deployed.
        
        Args:
            suggestion_id: Suggestion ID
            
        Returns:
            Optional[FixSuggestion]: Updated suggestion or None
        """
        suggestion = self.get_suggestion(suggestion_id)
        if not suggestion:
            logger.warning(f"Suggestion ID {suggestion_id} not found")
            return None
            
        # Update the suggestion
        suggestion.status = SuggestionStatus.DEPLOYED
        suggestion.modified_at = datetime.datetime.utcnow().isoformat()
        
        # Update and save
        if self.update_suggestion(suggestion):
            return suggestion
        return None
        
    def mark_failed(self, suggestion_id: str) -> Optional[FixSuggestion]:
        """Mark a suggestion as failed.
        
        Args:
            suggestion_id: Suggestion ID
            
        Returns:
            Optional[FixSuggestion]: Updated suggestion or None
        """
        suggestion = self.get_suggestion(suggestion_id)
        if not suggestion:
            logger.warning(f"Suggestion ID {suggestion_id} not found")
            return None
            
        # Update the suggestion
        suggestion.status = SuggestionStatus.FAILED
        suggestion.modified_at = datetime.datetime.utcnow().isoformat()
        
        # Update and save
        if self.update_suggestion(suggestion):
            return suggestion
        return None


# Singleton instance
_suggestion_manager = None

def get_suggestion_manager(storage_dir: Optional[str] = None, 
                          config: Dict[str, Any] = None) -> SuggestionManager:
    """Get or create the singleton SuggestionManager.
    
    Args:
        storage_dir: Directory to store suggestion data
        config: Configuration dictionary
        
    Returns:
        SuggestionManager: Singleton instance
    """
    global _suggestion_manager
    if _suggestion_manager is None:
        _suggestion_manager = SuggestionManager(storage_dir, config)
    return _suggestion_manager