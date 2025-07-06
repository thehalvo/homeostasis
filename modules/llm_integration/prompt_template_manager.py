#!/usr/bin/env python3
"""
Enhanced prompt template system for LLM integration.

Provides user-defined prompt engineering templates for specialized or domain-specific code fixes.
"""

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import re

from ..patch_generation.template_system import BaseTemplate


class PromptType(Enum):
    """Types of LLM prompts."""
    ERROR_ANALYSIS = "error_analysis"
    PATCH_GENERATION = "patch_generation"
    CODE_REVIEW = "code_review"
    REFACTORING = "refactoring"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    OPTIMIZATION = "optimization"
    SECURITY_REVIEW = "security_review"


@dataclass
class PromptMetadata:
    """Metadata for prompt templates."""
    name: str
    description: str
    prompt_type: PromptType
    domain: str  # e.g., "web_development", "data_science", "machine_learning"
    language: str  # Programming language
    framework: Optional[str] = None  # Framework (e.g., "django", "react")
    complexity_level: str = "intermediate"  # "beginner", "intermediate", "advanced"
    author: str = "unknown"
    version: str = "1.0.0"
    tags: Set[str] = field(default_factory=set)
    required_variables: List[str] = field(default_factory=list)
    optional_variables: List[str] = field(default_factory=list)
    example_usage: Optional[str] = None
    success_criteria: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    provider_preferences: Dict[str, float] = field(default_factory=dict)  # Provider name -> preference score


@dataclass
class PromptTemplate:
    """A prompt template for LLM interactions."""
    metadata: PromptMetadata
    system_prompt: str
    user_prompt_template: str
    context_template: Optional[str] = None
    examples: List[Dict[str, str]] = field(default_factory=list)
    validation_rules: List[str] = field(default_factory=list)
    post_processing_rules: List[str] = field(default_factory=list)
    
    def render(self, variables: Dict[str, Any]) -> Dict[str, str]:
        """
        Render the prompt template with variables.
        
        Args:
            variables: Variables to substitute in the template
            
        Returns:
            Dictionary with rendered prompts
        """
        # Create base template for processing
        system_template = BaseTemplate("temp_system", content=self.system_prompt)
        user_template = BaseTemplate("temp_user", content=self.user_prompt_template)
        
        rendered = {
            "system_prompt": system_template.render(variables),
            "user_prompt": user_template.render(variables)
        }
        
        if self.context_template:
            context_template = BaseTemplate("temp_context", content=self.context_template)
            rendered["context"] = context_template.render(variables)
        
        return rendered
    
    def validate_variables(self, variables: Dict[str, Any]) -> List[str]:
        """
        Validate that required variables are provided.
        
        Args:
            variables: Variables to validate
            
        Returns:
            List of validation errors
        """
        errors = []
        
        for required_var in self.metadata.required_variables:
            if required_var not in variables or variables[required_var] is None:
                errors.append(f"Required variable '{required_var}' is missing")
        
        return errors


class PromptTemplateManager:
    """Manager for prompt templates."""
    
    def __init__(self, templates_dir: Optional[Path] = None, user_templates_dir: Optional[Path] = None):
        """
        Initialize prompt template manager.
        
        Args:
            templates_dir: Directory containing built-in templates
            user_templates_dir: Directory containing user-defined templates
        """
        self.logger = logging.getLogger(__name__)
        
        # Set default directories
        if templates_dir is None:
            templates_dir = Path(__file__).parent / "prompt_templates"
        if user_templates_dir is None:
            user_templates_dir = Path.home() / ".homeostasis" / "prompt_templates"
        
        self.templates_dir = templates_dir
        self.user_templates_dir = user_templates_dir
        self.templates: Dict[str, PromptTemplate] = {}
        
        # Create directories if they don't exist
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        self.user_templates_dir.mkdir(parents=True, exist_ok=True)
        
        # Load templates
        self._load_built_in_templates()
        self._load_user_templates()
    
    def _load_built_in_templates(self) -> None:
        """Load built-in prompt templates."""
        self._load_templates_from_directory(self.templates_dir, is_user_template=False)
    
    def _load_user_templates(self) -> None:
        """Load user-defined prompt templates."""
        self._load_templates_from_directory(self.user_templates_dir, is_user_template=True)
    
    def _load_templates_from_directory(self, directory: Path, is_user_template: bool = False) -> None:
        """
        Load templates from a directory.
        
        Args:
            directory: Directory to load from
            is_user_template: Whether these are user-defined templates
        """
        if not directory.exists():
            return
        
        # Load YAML templates
        for template_file in directory.glob("*.yaml"):
            try:
                with open(template_file, 'r') as f:
                    data = yaml.safe_load(f)
                
                template = self._parse_template_data(data, template_file.stem)
                template_id = f"user:{template.metadata.name}" if is_user_template else template.metadata.name
                self.templates[template_id] = template
                
                self.logger.info(f"Loaded {'user' if is_user_template else 'built-in'} template: {template_id}")
                
            except Exception as e:
                self.logger.error(f"Failed to load template {template_file}: {e}")
        
        # Load JSON templates
        for template_file in directory.glob("*.json"):
            try:
                with open(template_file, 'r') as f:
                    data = json.load(f)
                
                template = self._parse_template_data(data, template_file.stem)
                template_id = f"user:{template.metadata.name}" if is_user_template else template.metadata.name
                self.templates[template_id] = template
                
                self.logger.info(f"Loaded {'user' if is_user_template else 'built-in'} template: {template_id}")
                
            except Exception as e:
                self.logger.error(f"Failed to load template {template_file}: {e}")
    
    def _parse_template_data(self, data: Dict[str, Any], default_name: str) -> PromptTemplate:
        """
        Parse template data from dictionary.
        
        Args:
            data: Template data
            default_name: Default name if not specified
            
        Returns:
            Parsed prompt template
        """
        # Parse metadata
        metadata_data = data.get("metadata", {})
        metadata = PromptMetadata(
            name=metadata_data.get("name", default_name),
            description=metadata_data.get("description", ""),
            prompt_type=PromptType(metadata_data.get("prompt_type", "patch_generation")),
            domain=metadata_data.get("domain", "general"),
            language=metadata_data.get("language", "python"),
            framework=metadata_data.get("framework"),
            complexity_level=metadata_data.get("complexity_level", "intermediate"),
            author=metadata_data.get("author", "unknown"),
            version=metadata_data.get("version", "1.0.0"),
            tags=set(metadata_data.get("tags", [])),
            required_variables=metadata_data.get("required_variables", []),
            optional_variables=metadata_data.get("optional_variables", []),
            example_usage=metadata_data.get("example_usage"),
            success_criteria=metadata_data.get("success_criteria", []),
            limitations=metadata_data.get("limitations", []),
            provider_preferences=metadata_data.get("provider_preferences", {})
        )
        
        # Parse template content
        return PromptTemplate(
            metadata=metadata,
            system_prompt=data.get("system_prompt", ""),
            user_prompt_template=data.get("user_prompt_template", ""),
            context_template=data.get("context_template"),
            examples=data.get("examples", []),
            validation_rules=data.get("validation_rules", []),
            post_processing_rules=data.get("post_processing_rules", [])
        )
    
    def get_template(self, template_id: str) -> Optional[PromptTemplate]:
        """
        Get a template by ID.
        
        Args:
            template_id: Template identifier
            
        Returns:
            Template if found, None otherwise
        """
        return self.templates.get(template_id)
    
    def list_templates(self, 
                      prompt_type: Optional[PromptType] = None,
                      domain: Optional[str] = None,
                      language: Optional[str] = None,
                      framework: Optional[str] = None,
                      complexity_level: Optional[str] = None,
                      tag: Optional[str] = None,
                      user_templates_only: bool = False) -> List[str]:
        """
        List templates with optional filtering.
        
        Args:
            prompt_type: Filter by prompt type
            domain: Filter by domain
            language: Filter by programming language
            framework: Filter by framework
            complexity_level: Filter by complexity level
            tag: Filter by tag
            user_templates_only: Only return user-defined templates
            
        Returns:
            List of template IDs
        """
        filtered_templates = []
        
        for template_id, template in self.templates.items():
            # Filter by user templates
            if user_templates_only and not template_id.startswith("user:"):
                continue
            
            # Filter by criteria
            if prompt_type and template.metadata.prompt_type != prompt_type:
                continue
            if domain and template.metadata.domain != domain:
                continue
            if language and template.metadata.language != language:
                continue
            if framework and template.metadata.framework != framework:
                continue
            if complexity_level and template.metadata.complexity_level != complexity_level:
                continue
            if tag and tag not in template.metadata.tags:
                continue
            
            filtered_templates.append(template_id)
        
        return sorted(filtered_templates)
    
    def find_best_template(self, 
                          prompt_type: PromptType,
                          language: str,
                          domain: Optional[str] = None,
                          framework: Optional[str] = None,
                          provider: Optional[str] = None) -> Optional[str]:
        """
        Find the best template for given criteria.
        
        Args:
            prompt_type: Type of prompt needed
            language: Programming language
            domain: Optional domain
            framework: Optional framework
            provider: LLM provider name
            
        Returns:
            Best matching template ID
        """
        candidates = []
        
        for template_id, template in self.templates.items():
            if template.metadata.prompt_type != prompt_type:
                continue
            if template.metadata.language != language:
                continue
            
            # Calculate matching score
            score = 0
            
            # Exact domain match
            if domain and template.metadata.domain == domain:
                score += 10
            elif domain and template.metadata.domain == "general":
                score += 1
            
            # Exact framework match
            if framework and template.metadata.framework == framework:
                score += 15
            elif not template.metadata.framework:  # Generic template
                score += 2
            
            # Provider preference
            if provider and provider in template.metadata.provider_preferences:
                score += template.metadata.provider_preferences[provider] * 5
            
            # User templates get slight preference
            if template_id.startswith("user:"):
                score += 1
            
            candidates.append((template_id, score))
        
        # Sort by score and return best match
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0] if candidates else None
    
    def create_user_template(self, template_data: Dict[str, Any], filename: Optional[str] = None) -> str:
        """
        Create a new user-defined template.
        
        Args:
            template_data: Template data
            filename: Optional filename (defaults to template name)
            
        Returns:
            Template ID of created template
        """
        template = self._parse_template_data(template_data, "unnamed")
        
        # Generate filename if not provided
        if filename is None:
            filename = f"{template.metadata.name}.yaml"
        
        # Save to user templates directory
        template_file = self.user_templates_dir / filename
        with open(template_file, 'w') as f:
            yaml.safe_dump(template_data, f, default_flow_style=False)
        
        # Add to templates registry
        template_id = f"user:{template.metadata.name}"
        self.templates[template_id] = template
        
        self.logger.info(f"Created user template: {template_id}")
        return template_id
    
    def update_user_template(self, template_id: str, template_data: Dict[str, Any]) -> bool:
        """
        Update an existing user template.
        
        Args:
            template_id: Template ID to update
            template_data: New template data
            
        Returns:
            True if successful, False otherwise
        """
        if not template_id.startswith("user:"):
            return False
        
        if template_id not in self.templates:
            return False
        
        try:
            template = self._parse_template_data(template_data, template_id.split(":", 1)[1])
            
            # Find the template file
            template_name = template.metadata.name
            for ext in [".yaml", ".json"]:
                template_file = self.user_templates_dir / f"{template_name}{ext}"
                if template_file.exists():
                    # Update the file
                    with open(template_file, 'w') as f:
                        if ext == ".yaml":
                            yaml.safe_dump(template_data, f, default_flow_style=False)
                        else:
                            json.dump(template_data, f, indent=2)
                    
                    # Update registry
                    self.templates[template_id] = template
                    self.logger.info(f"Updated user template: {template_id}")
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to update template {template_id}: {e}")
            return False
    
    def delete_user_template(self, template_id: str) -> bool:
        """
        Delete a user template.
        
        Args:
            template_id: Template ID to delete
            
        Returns:
            True if successful, False otherwise
        """
        if not template_id.startswith("user:"):
            return False
        
        if template_id not in self.templates:
            return False
        
        try:
            template = self.templates[template_id]
            template_name = template.metadata.name
            
            # Find and delete the template file
            for ext in [".yaml", ".json"]:
                template_file = self.user_templates_dir / f"{template_name}{ext}"
                if template_file.exists():
                    template_file.unlink()
                    break
            
            # Remove from registry
            del self.templates[template_id]
            self.logger.info(f"Deleted user template: {template_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete template {template_id}: {e}")
            return False
    
    def export_template(self, template_id: str, output_path: Path) -> bool:
        """
        Export a template to a file.
        
        Args:
            template_id: Template ID to export
            output_path: Output file path
            
        Returns:
            True if successful, False otherwise
        """
        template = self.get_template(template_id)
        if not template:
            return False
        
        try:
            template_data = {
                "metadata": {
                    "name": template.metadata.name,
                    "description": template.metadata.description,
                    "prompt_type": template.metadata.prompt_type.value,
                    "domain": template.metadata.domain,
                    "language": template.metadata.language,
                    "framework": template.metadata.framework,
                    "complexity_level": template.metadata.complexity_level,
                    "author": template.metadata.author,
                    "version": template.metadata.version,
                    "tags": list(template.metadata.tags),
                    "required_variables": template.metadata.required_variables,
                    "optional_variables": template.metadata.optional_variables,
                    "example_usage": template.metadata.example_usage,
                    "success_criteria": template.metadata.success_criteria,
                    "limitations": template.metadata.limitations,
                    "provider_preferences": template.metadata.provider_preferences
                },
                "system_prompt": template.system_prompt,
                "user_prompt_template": template.user_prompt_template,
                "context_template": template.context_template,
                "examples": template.examples,
                "validation_rules": template.validation_rules,
                "post_processing_rules": template.post_processing_rules
            }
            
            if output_path.suffix.lower() == ".yaml":
                with open(output_path, 'w') as f:
                    yaml.safe_dump(template_data, f, default_flow_style=False)
            else:
                with open(output_path, 'w') as f:
                    json.dump(template_data, f, indent=2)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export template {template_id}: {e}")
            return False
    
    def import_template(self, template_path: Path, as_user_template: bool = True) -> Optional[str]:
        """
        Import a template from a file.
        
        Args:
            template_path: Path to template file
            as_user_template: Whether to import as user template
            
        Returns:
            Template ID if successful, None otherwise
        """
        try:
            if template_path.suffix.lower() == ".yaml":
                with open(template_path, 'r') as f:
                    template_data = yaml.safe_load(f)
            else:
                with open(template_path, 'r') as f:
                    template_data = json.load(f)
            
            if as_user_template:
                return self.create_user_template(template_data, template_path.name)
            else:
                template = self._parse_template_data(template_data, template_path.stem)
                template_id = template.metadata.name
                self.templates[template_id] = template
                return template_id
                
        except Exception as e:
            self.logger.error(f"Failed to import template from {template_path}: {e}")
            return None
    
    def get_template_info(self, template_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a template.
        
        Args:
            template_id: Template ID
            
        Returns:
            Template information dictionary
        """
        template = self.get_template(template_id)
        if not template:
            return None
        
        return {
            "id": template_id,
            "name": template.metadata.name,
            "description": template.metadata.description,
            "prompt_type": template.metadata.prompt_type.value,
            "domain": template.metadata.domain,
            "language": template.metadata.language,
            "framework": template.metadata.framework,
            "complexity_level": template.metadata.complexity_level,
            "author": template.metadata.author,
            "version": template.metadata.version,
            "tags": list(template.metadata.tags),
            "required_variables": template.metadata.required_variables,
            "optional_variables": template.metadata.optional_variables,
            "example_usage": template.metadata.example_usage,
            "success_criteria": template.metadata.success_criteria,
            "limitations": template.metadata.limitations,
            "provider_preferences": template.metadata.provider_preferences,
            "is_user_template": template_id.startswith("user:")
        }


# Global instance
_prompt_template_manager = None

def get_prompt_template_manager() -> PromptTemplateManager:
    """Get the global prompt template manager instance."""
    global _prompt_template_manager
    if _prompt_template_manager is None:
        _prompt_template_manager = PromptTemplateManager()
    return _prompt_template_manager