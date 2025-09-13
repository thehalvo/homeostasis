"""
Feedback module for fix suggestions.

Collects and processes user feedback on fix suggestions to improve future fixes.
"""

import datetime
import enum
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class FeedbackType(enum.Enum):
    """Types of feedback."""

    RATING = "rating"  # Numeric rating (1-5)
    APPROVAL = "approval"  # Approved for deployment
    REJECTION = "rejection"  # Rejected, not suitable
    MODIFICATION = "modification"  # Suggestion was modified
    COMMENT = "comment"  # General comment


class Feedback:
    """Represents feedback on a fix suggestion."""

    def __init__(
        self,
        feedback_id: str,
        suggestion_id: str,
        feedback_type: FeedbackType,
        user_id: str,
        timestamp: str,
        content: Dict[str, Any],
    ):
        """Initialize feedback.

        Args:
            feedback_id: Unique feedback ID
            suggestion_id: ID of the suggestion feedback is for
            feedback_type: Type of feedback
            user_id: ID of the user providing feedback
            timestamp: Timestamp when feedback was given
            content: Feedback content (varies by type)
        """
        self.feedback_id = feedback_id
        self.suggestion_id = suggestion_id
        self.feedback_type = feedback_type
        self.user_id = user_id
        self.timestamp = timestamp
        self.content = content

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return {
            "feedback_id": self.feedback_id,
            "suggestion_id": self.suggestion_id,
            "feedback_type": self.feedback_type.value,
            "user_id": self.user_id,
            "timestamp": self.timestamp,
            "content": self.content,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Feedback":
        """Create from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            Feedback: New instance
        """
        # Convert feedback type string to enum
        feedback_type_str = data.get("feedback_type", "comment")
        feedback_type = FeedbackType(feedback_type_str)

        return cls(
            feedback_id=data["feedback_id"],
            suggestion_id=data["suggestion_id"],
            feedback_type=feedback_type,
            user_id=data["user_id"],
            timestamp=data["timestamp"],
            content=data.get("content", {}),
        )


class FeedbackManager:
    """
    Manages feedback on fix suggestions.

    Collects, stores, and analyzes user feedback to improve future fix generation.
    """

    def __init__(self, storage_dir: Optional[str] = None):
        """Initialize the feedback manager.

        Args:
            storage_dir: Directory to store feedback data
        """
        self.storage_dir = Path(storage_dir or "logs/feedback")
        os.makedirs(self.storage_dir, exist_ok=True)

        # Load existing feedback
        self.feedback: Dict[str, List[Feedback]] = {}  # suggestion_id -> List[Feedback]
        self._load_feedback()

        logger.info(
            f"Initialized feedback manager with storage directory: {self.storage_dir}"
        )

    def _load_feedback(self) -> None:
        """Load existing feedback from storage."""
        try:
            # Look for feedback files
            for file_path in self.storage_dir.glob("*.json"):
                try:
                    with open(file_path, "r") as f:
                        data = json.load(f)

                    if "suggestion_id" not in data or "feedback" not in data:
                        logger.warning(f"Invalid feedback file: {file_path}")
                        continue

                    suggestion_id = data["suggestion_id"]
                    feedback_list = []

                    for feedback_data in data["feedback"]:
                        try:
                            feedback = Feedback.from_dict(feedback_data)
                            feedback_list.append(feedback)
                        except (KeyError, ValueError) as e:
                            logger.warning(f"Error loading feedback: {e}")
                            continue

                    if feedback_list:
                        self.feedback[suggestion_id] = feedback_list
                        logger.debug(
                            f"Loaded {len(feedback_list)} feedback items for suggestion {suggestion_id}"
                        )

                except Exception as e:
                    logger.warning(f"Error loading feedback file {file_path}: {e}")

            logger.info(f"Loaded feedback for {len(self.feedback)} suggestions")

        except Exception as e:
            logger.error(f"Error loading feedback: {e}")

    def _save_feedback(self, suggestion_id: str) -> None:
        """Save feedback for a suggestion.

        Args:
            suggestion_id: Suggestion ID to save feedback for
        """
        if suggestion_id not in self.feedback:
            return

        feedback_list = self.feedback[suggestion_id]

        try:
            file_path = self.storage_dir / f"{suggestion_id}.json"

            data = {
                "suggestion_id": suggestion_id,
                "feedback": [feedback.to_dict() for feedback in feedback_list],
            }

            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)

            logger.debug(
                f"Saved {len(feedback_list)} feedback items for suggestion {suggestion_id}"
            )

        except Exception as e:
            logger.error(f"Error saving feedback for suggestion {suggestion_id}: {e}")

    def add_rating(
        self,
        suggestion_id: str,
        user_id: str,
        rating: int,
        comments: Optional[str] = None,
    ) -> Feedback:
        """Add a rating for a suggestion.

        Args:
            suggestion_id: Suggestion ID
            user_id: User ID
            rating: Rating (1-5)
            comments: Optional comments

        Returns:
            Feedback: Created feedback

        Raises:
            ValueError: If rating is invalid
        """
        # Validate rating
        if not isinstance(rating, int) or rating < 1 or rating > 5:
            raise ValueError("Rating must be an integer between 1 and 5")

        # Create feedback
        feedback_id = str(uuid.uuid4())
        now = datetime.datetime.utcnow().isoformat()

        content = {"rating": rating, "comments": comments}

        feedback = Feedback(
            feedback_id=feedback_id,
            suggestion_id=suggestion_id,
            feedback_type=FeedbackType.RATING,
            user_id=user_id,
            timestamp=now,
            content=content,
        )

        # Add to storage
        if suggestion_id not in self.feedback:
            self.feedback[suggestion_id] = []
        self.feedback[suggestion_id].append(feedback)

        # Save
        self._save_feedback(suggestion_id)

        logger.info(
            f"Added rating {rating} for suggestion {suggestion_id} by user {user_id}"
        )
        return feedback

    def add_approval(
        self, suggestion_id: str, user_id: str, comments: Optional[str] = None
    ) -> Feedback:
        """Add an approval for a suggestion.

        Args:
            suggestion_id: Suggestion ID
            user_id: User ID
            comments: Optional comments

        Returns:
            Feedback: Created feedback
        """
        # Create feedback
        feedback_id = str(uuid.uuid4())
        now = datetime.datetime.utcnow().isoformat()

        content = {"comments": comments}

        feedback = Feedback(
            feedback_id=feedback_id,
            suggestion_id=suggestion_id,
            feedback_type=FeedbackType.APPROVAL,
            user_id=user_id,
            timestamp=now,
            content=content,
        )

        # Add to storage
        if suggestion_id not in self.feedback:
            self.feedback[suggestion_id] = []
        self.feedback[suggestion_id].append(feedback)

        # Save
        self._save_feedback(suggestion_id)

        logger.info(f"Added approval for suggestion {suggestion_id} by user {user_id}")
        return feedback

    def add_rejection(self, suggestion_id: str, user_id: str, reason: str) -> Feedback:
        """Add a rejection for a suggestion.

        Args:
            suggestion_id: Suggestion ID
            user_id: User ID
            reason: Rejection reason

        Returns:
            Feedback: Created feedback
        """
        # Create feedback
        feedback_id = str(uuid.uuid4())
        now = datetime.datetime.utcnow().isoformat()

        content = {"reason": reason}

        feedback = Feedback(
            feedback_id=feedback_id,
            suggestion_id=suggestion_id,
            feedback_type=FeedbackType.REJECTION,
            user_id=user_id,
            timestamp=now,
            content=content,
        )

        # Add to storage
        if suggestion_id not in self.feedback:
            self.feedback[suggestion_id] = []
        self.feedback[suggestion_id].append(feedback)

        # Save
        self._save_feedback(suggestion_id)

        logger.info(f"Added rejection for suggestion {suggestion_id} by user {user_id}")
        return feedback

    def add_modification(
        self,
        suggestion_id: str,
        user_id: str,
        original_code: str,
        modified_code: str,
        comments: Optional[str] = None,
    ) -> Feedback:
        """Add a modification for a suggestion.

        Args:
            suggestion_id: Suggestion ID
            user_id: User ID
            original_code: Original code
            modified_code: Modified code
            comments: Optional comments

        Returns:
            Feedback: Created feedback
        """
        # Create feedback
        feedback_id = str(uuid.uuid4())
        now = datetime.datetime.utcnow().isoformat()

        content = {
            "original_code": original_code,
            "modified_code": modified_code,
            "comments": comments,
        }

        feedback = Feedback(
            feedback_id=feedback_id,
            suggestion_id=suggestion_id,
            feedback_type=FeedbackType.MODIFICATION,
            user_id=user_id,
            timestamp=now,
            content=content,
        )

        # Add to storage
        if suggestion_id not in self.feedback:
            self.feedback[suggestion_id] = []
        self.feedback[suggestion_id].append(feedback)

        # Save
        self._save_feedback(suggestion_id)

        logger.info(
            f"Added modification for suggestion {suggestion_id} by user {user_id}"
        )
        return feedback

    def add_comment(self, suggestion_id: str, user_id: str, comment: str) -> Feedback:
        """Add a comment for a suggestion.

        Args:
            suggestion_id: Suggestion ID
            user_id: User ID
            comment: Comment text

        Returns:
            Feedback: Created feedback
        """
        # Create feedback
        feedback_id = str(uuid.uuid4())
        now = datetime.datetime.utcnow().isoformat()

        content = {"comment": comment}

        feedback = Feedback(
            feedback_id=feedback_id,
            suggestion_id=suggestion_id,
            feedback_type=FeedbackType.COMMENT,
            user_id=user_id,
            timestamp=now,
            content=content,
        )

        # Add to storage
        if suggestion_id not in self.feedback:
            self.feedback[suggestion_id] = []
        self.feedback[suggestion_id].append(feedback)

        # Save
        self._save_feedback(suggestion_id)

        logger.info(f"Added comment for suggestion {suggestion_id} by user {user_id}")
        return feedback

    def get_feedback(self, suggestion_id: str) -> List[Feedback]:
        """Get all feedback for a suggestion.

        Args:
            suggestion_id: Suggestion ID

        Returns:
            List[Feedback]: Feedback for the suggestion
        """
        return self.feedback.get(suggestion_id, [])

    def get_average_rating(self, suggestion_id: str) -> Optional[float]:
        """Get the average rating for a suggestion.

        Args:
            suggestion_id: Suggestion ID

        Returns:
            Optional[float]: Average rating or None if no ratings
        """
        feedback_list = self.get_feedback(suggestion_id)
        if not feedback_list:
            return None

        # Filter to only include ratings
        ratings = [
            f.content["rating"]
            for f in feedback_list
            if f.feedback_type == FeedbackType.RATING and "rating" in f.content
        ]

        if not ratings:
            return None

        return sum(ratings) / len(ratings)

    def get_approval_count(self, suggestion_id: str) -> int:
        """Get the number of approvals for a suggestion.

        Args:
            suggestion_id: Suggestion ID

        Returns:
            int: Number of approvals
        """
        feedback_list = self.get_feedback(suggestion_id)
        return sum(1 for f in feedback_list if f.feedback_type == FeedbackType.APPROVAL)

    def get_rejection_count(self, suggestion_id: str) -> int:
        """Get the number of rejections for a suggestion.

        Args:
            suggestion_id: Suggestion ID

        Returns:
            int: Number of rejections
        """
        feedback_list = self.get_feedback(suggestion_id)
        return sum(
            1 for f in feedback_list if f.feedback_type == FeedbackType.REJECTION
        )

    def get_modification_count(self, suggestion_id: str) -> int:
        """Get the number of modifications for a suggestion.

        Args:
            suggestion_id: Suggestion ID

        Returns:
            int: Number of modifications
        """
        feedback_list = self.get_feedback(suggestion_id)
        return sum(
            1 for f in feedback_list if f.feedback_type == FeedbackType.MODIFICATION
        )

    def get_latest_modification(self, suggestion_id: str) -> Optional[Feedback]:
        """Get the latest modification for a suggestion.

        Args:
            suggestion_id: Suggestion ID

        Returns:
            Optional[Feedback]: Latest modification or None
        """
        feedback_list = self.get_feedback(suggestion_id)

        # Filter to only include modifications
        modifications = [
            f for f in feedback_list if f.feedback_type == FeedbackType.MODIFICATION
        ]

        if not modifications:
            return None

        # Sort by timestamp (descending) and return the first one
        return sorted(modifications, key=lambda f: f.timestamp, reverse=True)[0]


# Singleton instance
_feedback_manager = None


def get_feedback_manager(storage_dir: Optional[str] = None) -> FeedbackManager:
    """Get or create the singleton FeedbackManager.

    Args:
        storage_dir: Directory to store feedback data

    Returns:
        FeedbackManager: Singleton instance
    """
    global _feedback_manager
    if _feedback_manager is None:
        _feedback_manager = FeedbackManager(storage_dir)
    return _feedback_manager
