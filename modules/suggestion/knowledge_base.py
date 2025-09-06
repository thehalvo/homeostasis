"""
Knowledge Base for fix suggestions.

Stores and organizes validated fixes to improve future fix generation.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class FixTemplate:
    """Represents a fix template in the knowledge base."""

    def __init__(
        self,
        template_id: str,
        error_type: str,
        fix_type: str,
        pattern: str,
        template: str,
        metadata: Dict[str, Any],
    ):
        """Initialize a fix template.

        Args:
            template_id: Unique template ID
            error_type: Type of error this fixes
            fix_type: Type of fix
            pattern: Regex pattern to match code that needs fixing
            template: Template for the fix
            metadata: Additional metadata
        """
        self.template_id = template_id
        self.error_type = error_type
        self.fix_type = fix_type
        self.pattern = pattern
        self.template = template
        self.metadata = metadata

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return {
            "template_id": self.template_id,
            "error_type": self.error_type,
            "fix_type": self.fix_type,
            "pattern": self.pattern,
            "template": self.template,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FixTemplate":
        """Create from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            FixTemplate: New instance
        """
        return cls(
            template_id=data["template_id"],
            error_type=data["error_type"],
            fix_type=data["fix_type"],
            pattern=data["pattern"],
            template=data["template"],
            metadata=data.get("metadata", {}),
        )


class KnowledgeBase:
    """
    Stores and manages a knowledge base of validated fixes.

    Organizes fixes by error type and provides templates for generating new fixes.
    """

    def __init__(self, storage_dir: Optional[str] = None):
        """Initialize the knowledge base.

        Args:
            storage_dir: Directory to store knowledge base data
        """
        self.storage_dir = Path(storage_dir or "logs/knowledge_base")
        os.makedirs(self.storage_dir, exist_ok=True)

        # Load existing templates
        self.templates = {}  # error_type -> List[FixTemplate]
        self._load_templates()

        logger.info(
            f"Initialized knowledge base with storage directory: {self.storage_dir}"
        )

    def _load_templates(self) -> None:
        """Load existing templates from storage."""
        try:
            # Look for template files
            for file_path in self.storage_dir.glob("*.json"):
                try:
                    with open(file_path, "r") as f:
                        data = json.load(f)

                    if "error_type" not in data or "templates" not in data:
                        logger.warning(f"Invalid template file: {file_path}")
                        continue

                    error_type = data["error_type"]
                    templates = []

                    for template_data in data["templates"]:
                        try:
                            template = FixTemplate.from_dict(template_data)
                            templates.append(template)
                        except (KeyError, ValueError) as e:
                            logger.warning(f"Error loading template: {e}")
                            continue

                    if templates:
                        self.templates[error_type] = templates
                        logger.debug(
                            f"Loaded {len(templates)} templates for error type {error_type}"
                        )

                except Exception as e:
                    logger.warning(f"Error loading template file {file_path}: {e}")

            logger.info(f"Loaded templates for {len(self.templates)} error types")

        except Exception as e:
            logger.error(f"Error loading templates: {e}")

    def _save_templates(self, error_type: str) -> None:
        """Save templates for an error type.

        Args:
            error_type: Error type to save templates for
        """
        if error_type not in self.templates:
            return

        templates = self.templates[error_type]

        try:
            file_path = self.storage_dir / f"{error_type}.json"

            data = {
                "error_type": error_type,
                "templates": [template.to_dict() for template in templates],
            }

            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)

            logger.debug(
                f"Saved {len(templates)} templates for error type {error_type}"
            )

        except Exception as e:
            logger.error(f"Error saving templates for error type {error_type}: {e}")

    def add_template(
        self,
        error_type: str,
        fix_type: str,
        pattern: str,
        template: str,
        metadata: Dict[str, Any] = None,
    ) -> FixTemplate:
        """Add a new fix template.

        Args:
            error_type: Type of error this fixes
            fix_type: Type of fix
            pattern: Regex pattern to match code that needs fixing
            template: Template for the fix
            metadata: Additional metadata

        Returns:
            FixTemplate: Created template
        """
        # Validate pattern by trying to compile it
        try:
            re.compile(pattern)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {e}")

        # Create template
        template_id = (
            f"{error_type}_{fix_type}_{len(self.get_templates(error_type)) + 1}"
        )

        fix_template = FixTemplate(
            template_id=template_id,
            error_type=error_type,
            fix_type=fix_type,
            pattern=pattern,
            template=template,
            metadata=metadata or {},
        )

        # Add to storage
        if error_type not in self.templates:
            self.templates[error_type] = []
        self.templates[error_type].append(fix_template)

        # Save
        self._save_templates(error_type)

        logger.info(f"Added template {template_id} for error type {error_type}")
        return fix_template

    def get_templates(self, error_type: str) -> List[FixTemplate]:
        """Get all templates for an error type.

        Args:
            error_type: Error type

        Returns:
            List[FixTemplate]: Templates for the error type
        """
        return self.templates.get(error_type, [])

    def get_template(self, template_id: str) -> Optional[FixTemplate]:
        """Get a specific template by ID.

        Args:
            template_id: Template ID

        Returns:
            Optional[FixTemplate]: Template or None
        """
        for error_type, templates in self.templates.items():
            for template in templates:
                if template.template_id == template_id:
                    return template

        return None

    def update_template(self, template: FixTemplate) -> bool:
        """Update a template.

        Args:
            template: Updated template

        Returns:
            bool: True if updated successfully
        """
        template_id = template.template_id
        error_type = template.error_type

        if error_type not in self.templates:
            logger.warning(f"Error type {error_type} not found")
            return False

        # Update template in the list
        for i, existing in enumerate(self.templates[error_type]):
            if existing.template_id == template_id:
                self.templates[error_type][i] = template
                self._save_templates(error_type)
                logger.info(
                    f"Updated template {template_id} for error type {error_type}"
                )
                return True

        logger.warning(
            f"Template ID {template_id} not found for error type {error_type}"
        )
        return False

    def delete_template(self, template_id: str) -> bool:
        """Delete a template.

        Args:
            template_id: Template ID

        Returns:
            bool: True if deleted successfully
        """
        for error_type, templates in self.templates.items():
            for i, template in enumerate(templates):
                if template.template_id == template_id:
                    # Remove template
                    del self.templates[error_type][i]
                    self._save_templates(error_type)
                    logger.info(
                        f"Deleted template {template_id} for error type {error_type}"
                    )
                    return True

        logger.warning(f"Template ID {template_id} not found")
        return False

    def find_matching_templates(self, error_type: str, code: str) -> List[FixTemplate]:
        """Find templates that match code for an error type.

        Args:
            error_type: Error type
            code: Code to match against templates

        Returns:
            List[FixTemplate]: Matching templates
        """
        if error_type not in self.templates:
            return []

        matching_templates = []

        for template in self.templates[error_type]:
            try:
                if re.search(template.pattern, code):
                    matching_templates.append(template)
            except re.error:
                # Skip invalid patterns
                logger.warning(
                    f"Invalid regex pattern in template {template.template_id}"
                )
                continue

        return matching_templates

    def learn_from_fix(
        self,
        error_type: str,
        fix_type: str,
        original_code: str,
        fixed_code: str,
        metadata: Dict[str, Any] = None,
    ) -> Optional[FixTemplate]:
        """Learn a new template from a successful fix.

        Args:
            error_type: Type of error fixed
            fix_type: Type of fix applied
            original_code: Original buggy code
            fixed_code: Fixed code
            metadata: Additional metadata

        Returns:
            Optional[FixTemplate]: Created template or None
        """
        try:
            # Simplify code to create a pattern
            # This is a simplified implementation - in a real system,
            # you would use more sophisticated techniques

            # Create a simple pattern by escaping regex metacharacters
            # and replacing specific identifiers with wildcards
            pattern = re.escape(original_code)

            # Replace variable names with wildcards
            # This is just a simple example - real implementation would be more sophisticated
            var_names = set(re.findall(r"\b[a-zA-Z_]\w*\b", original_code))
            for var_name in var_names:
                pattern = pattern.replace(var_name, r"\w+")

            # Create a template from the fixed code
            template = fixed_code

            # Add the template to the knowledge base
            return self.add_template(
                error_type=error_type,
                fix_type=fix_type,
                pattern=pattern,
                template=template,
                metadata=metadata,
            )

        except Exception as e:
            logger.error(f"Error learning from fix: {e}")
            return None


# Singleton instance
_knowledge_base = None


def get_knowledge_base(storage_dir: Optional[str] = None) -> KnowledgeBase:
    """Get or create the singleton KnowledgeBase.

    Args:
        storage_dir: Directory to store knowledge base data

    Returns:
        KnowledgeBase: Singleton instance
    """
    global _knowledge_base
    if _knowledge_base is None:
        _knowledge_base = KnowledgeBase(storage_dir)
    return _knowledge_base
