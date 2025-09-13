"""
Hierarchical template system for code patch generation.

This module provides a robust template system that supports:
- Template inheritance and specialization
- Template composition and reuse
- Advanced variable substitution
- Conditional sections based on template variables
- Indentation preservation and normalization
"""

import json
import re
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional

# Import indentation utilities
from modules.patch_generation.indent_utils import (
    adjust_indentation_for_context,
    normalize_indentation,
    preserve_relative_indentation,
)


class BaseTemplate:
    """
    Base class for all patch templates, supporting inheritance and composition.
    """

    # Class-level variable for template registry
    templates_registry: ClassVar[Dict[str, "BaseTemplate"]] = {}

    def __init__(
        self,
        name: str,
        content: Optional[str] = None,
        file_path: Optional[Path] = None,
        parent: Optional[str] = None,
    ):
        """
        Initialize a template.

        Args:
            name: Unique identifier for this template
            content: The template content as a string (optional if file_path is provided)
            file_path: Path to the template file (optional if content is provided)
            parent: Name of the parent template (optional)
        """
        self.name = name
        self.parent_name = parent
        self.parent: Optional['BaseTemplate'] = None
        self.file_path = file_path

        # Store raw content before processing
        if content is not None:
            self._raw_content = content
        elif file_path is not None:
            self._raw_content = self._load_from_file(file_path)
        else:
            self._raw_content = ""

        # Process the template content
        self._process_template()

        # Register this template
        self.register()

    def _load_from_file(self, file_path: Path) -> str:
        """
        Load template content from a file.

        Args:
            file_path: Path to the template file

        Returns:
            The template content as a string
        """
        with open(file_path, "r") as f:
            return f.read()

    def _process_template(self) -> None:
        """
        Process the template content to extract blocks and metadata.
        """
        # Extract template metadata
        self.metadata = self._extract_metadata()

        # Extract template blocks
        self.blocks = self._extract_blocks()

        # Link to parent template if specified
        if self.parent_name:
            self._link_parent()

    def _extract_metadata(self) -> Dict[str, Any]:
        """
        Extract metadata from template comments.

        Returns:
            Dictionary of metadata
        """
        metadata: Dict[str, Any] = {"variables": [], "description": "", "applicability": []}

        # Extract variable definitions from comments
        var_pattern = r"#\s*-\s*\{\{\s*([\w_]+)\s*\}\}:\s*(.+)$"
        for line in self._raw_content.splitlines():
            var_match = re.match(var_pattern, line)
            if var_match:
                var_name = var_match.group(1)
                var_desc = var_match.group(2).strip()
                metadata["variables"].append(
                    {"name": var_name, "description": var_desc}
                )

        # Extract description
        desc_pattern = r"#\s*Template for (.+)$"
        desc_match = re.search(desc_pattern, self._raw_content)
        if desc_match:
            metadata["description"] = desc_match.group(1).strip()

        # Extract applicability information
        app_pattern = r"#\s*Applicable for:\s*(.+)$"
        app_match = re.search(app_pattern, self._raw_content)
        if app_match:
            applicability = app_match.group(1).strip()
            metadata["applicability"] = [
                item.strip() for item in applicability.split(",")
            ]

        return metadata

    def _extract_blocks(self) -> Dict[str, str]:
        """
        Extract template blocks from the content.

        Returns:
            Dictionary of block names to block content
        """
        blocks = {"main": self._raw_content}  # Default block is the entire content

        # Extract named blocks defined with {% block name %} content {% endblock %}
        block_pattern = r"\{\%\s*block\s+(\w+)\s*\%\}(.*?)\{\%\s*endblock\s*\%\}"
        for match in re.finditer(block_pattern, self._raw_content, re.DOTALL):
            block_name = match.group(1)
            block_content = match.group(2)
            blocks[block_name] = block_content.strip()

        return blocks

    def _link_parent(self) -> None:
        """
        Link this template to its parent.
        """
        if self.parent_name in self.templates_registry:
            self.parent = self.templates_registry[self.parent_name]

    def register(self) -> None:
        """
        Register this template in the global registry.
        """
        self.templates_registry[self.name] = self

    def get_block(self, block_name: str = "main") -> str:
        """
        Get a specific block from this template or its parent hierarchy.

        Args:
            block_name: Name of the block to retrieve

        Returns:
            The block content as a string
        """
        # Check if this template has the requested block
        if block_name in self.blocks:
            return self.blocks[block_name]

        # If not, check the parent template if available
        if self.parent:
            return self.parent.get_block(block_name)

        # If no parent or parent doesn't have the block, return empty string
        return ""

    def render(self, variables: Dict[str, str]) -> str:
        """
        Render the template with the provided variables.

        Args:
            variables: Dictionary of variable names and values to substitute

        Returns:
            Rendered template as a string
        """
        # Start with the main block
        result = self.get_block("main")

        # Perform template inheritance/composition first
        # Replace block references with actual block content
        result = self._process_block_includes(result)

        # Extract indentation information before variable substitution
        target_indent = variables.get("__indentation__", "")

        # Replace each variable
        result = self._substitute_variables(result, variables)

        # Process conditionals
        result = self._process_conditionals(result, variables)

        # Process loops
        result = self._process_loops(result, variables)

        # Apply any necessary indentation adjustments
        if (
            "code_block" in variables
            and variables.get("preserve_indentation", "true").lower() == "true"
        ):
            # If this template uses code_block, preserve relative indentation
            # Normalize the result first to remove any template-defined indentation
            normalized_result = normalize_indentation(result)

            # If a target indentation was specified, use it
            if target_indent:
                result = preserve_relative_indentation(normalized_result, target_indent)
            else:
                # Try to infer indentation from code_block
                code_block = variables.get("code_block", "")
                if code_block:
                    try:
                        # Use the context to calculate appropriate indentation
                        result = adjust_indentation_for_context(
                            normalized_result, code_block
                        )
                    except Exception:
                        # Fall back to normalized result if indentation adjustment fails
                        result = normalized_result
                else:
                    result = normalized_result

        return result

    def _process_block_includes(self, content: str) -> str:
        """
        Process block includes in the form {% include block_name %}.

        Args:
            content: Template content to process

        Returns:
            Processed content with block includes resolved
        """
        include_pattern = r"\{\%\s*include\s+(\w+)\s*\%\}"

        def replace_include(match):
            block_name = match.group(1)
            # Get the block content from this template or parent
            return self.get_block(block_name)

        # Replace all includes
        return re.sub(include_pattern, replace_include, content)

    def _substitute_variables(self, content: str, variables: Dict[str, str]) -> str:
        """
        Substitute variables in template.

        Args:
            content: Template content
            variables: Dictionary of variable values

        Returns:
            Template with variables substituted
        """
        result = content

        # Replace each variable
        for var_name, var_value in variables.items():
            # Match both "{{ var }}" and "{{var}}" patterns (with or without spaces)
            placeholder1 = "{{ " + var_name + " }}"
            placeholder2 = "{{" + var_name + "}}"

            if var_value is not None:
                result = result.replace(placeholder1, str(var_value))
                result = result.replace(placeholder2, str(var_value))

            # Handle default values in the format {{ var|default:"value" }}
            default_pattern = (
                r"\{\{\s*"
                + re.escape(var_name)
                + r'\s*\|\s*default\s*:\s*"([^"]+)"\s*\}\}'
            )

            def replace_default(match):
                default_value = match.group(1)
                return str(var_value) if var_value is not None else default_value

            result = re.sub(default_pattern, replace_default, result)

        return result

    def _process_conditionals(self, content: str, variables: Dict[str, str]) -> str:
        """
        Process conditional blocks in template.

        Args:
            content: Template content
            variables: Dictionary of variable values

        Returns:
            Template with conditionals processed
        """
        # Process if-else blocks first
        if_else_pattern = r"\{\%\s*if\s+([^\}]+)\s*\%\}(.*?)\{\%\s*else\s*\%\}(.*?)\{\%\s*endif\s*\%\}"

        def replace_if_else(match):
            condition = match.group(1).strip()
            if_content = match.group(2)
            else_content = match.group(3)

            # Evaluate the condition
            if self._evaluate_condition(condition, variables):
                return if_content
            else:
                return else_content

        # Replace all if-else conditionals
        result = re.sub(if_else_pattern, replace_if_else, content, flags=re.DOTALL)

        # Process simple if blocks
        if_pattern = r"\{\%\s*if\s+([^\}]+)\s*\%\}(.*?)\{\%\s*endif\s*\%\}"

        def replace_if(match):
            condition = match.group(1).strip()
            if_content = match.group(2)

            # Evaluate the condition
            if self._evaluate_condition(condition, variables):
                return if_content
            else:
                return ""

        # Replace all simple if conditionals
        result = re.sub(if_pattern, replace_if, result, flags=re.DOTALL)

        return result

    def _evaluate_condition(self, condition: str, variables: Dict[str, str]) -> bool:
        """
        Evaluate a condition string.

        Args:
            condition: Condition string (e.g., "var1 == 'value'" or "var2")
            variables: Dictionary of variable values

        Returns:
            True if the condition is met, False otherwise
        """
        # Check for equality condition (==)
        if "==" in condition:
            parts = condition.split("==")
            var_name = parts[0].strip()
            expected_value = parts[1].strip().strip('"').strip("'")

            return str(variables.get(var_name, "")) == expected_value

        # Check for inequality condition (!=)
        elif "!=" in condition:
            parts = condition.split("!=")
            var_name = parts[0].strip()
            expected_value = parts[1].strip().strip('"').strip("'")

            return str(variables.get(var_name, "")) != expected_value

        # Check for presence condition (just a variable name)
        else:
            var_name = condition.strip()

            # Check if variable exists and is truthy
            var_value = variables.get(var_name, "")
            return bool(var_value and var_value.lower() not in ("false", "0", ""))

    def _process_loops(self, content: str, variables: Dict[str, str]) -> str:
        """
        Process loop blocks in template.

        Args:
            content: Template content
            variables: Dictionary of variable values

        Returns:
            Template with loops processed
        """
        # Process for loops
        for_loop_pattern = (
            r"\{\%\s*for\s+(\w+)\s+in\s+(\w+)\s*\%\}(.*?)\{\%\s*endfor\s*\%\}"
        )

        def replace_for_loop(match):
            item_var = match.group(1)
            list_var = match.group(2)
            loop_content = match.group(3)

            # Get the list from variables
            items_list = variables.get(list_var, "[]")

            # Try to parse as JSON if it's a string representation of a list
            if isinstance(items_list, str):
                try:
                    items_list = json.loads(items_list)
                except json.JSONDecodeError:
                    # Not valid JSON, treat as a comma-separated list
                    items_list = [item.strip() for item in items_list.split(",")]

            # Ensure it's a list
            if not isinstance(items_list, list):
                return ""

            # Process the loop
            result = []
            for item in items_list:
                # Create a temporary variables dict with the loop variable
                temp_vars = variables.copy()
                temp_vars[item_var] = item

                # Replace variables in this iteration
                iteration_content = self._substitute_variables(loop_content, temp_vars)
                result.append(iteration_content)

            return "".join(result)

        # Replace all for loops
        result = re.sub(for_loop_pattern, replace_for_loop, content, flags=re.DOTALL)

        return result

    @classmethod
    def get_template(cls, name: str) -> Optional["BaseTemplate"]:
        """
        Get a template by name from the registry.

        Args:
            name: Name of the template

        Returns:
            The template if found, None otherwise
        """
        return cls.templates_registry.get(name)

    @classmethod
    def list_templates(cls) -> List[str]:
        """
        Get a list of all registered template names.

        Returns:
            List of template names
        """
        return list(cls.templates_registry.keys())


class TemplateManager:
    """
    Manager class for template loading, organization, and selection.
    """

    def __init__(self, templates_dir: Path):
        """
        Initialize the template manager.

        Args:
            templates_dir: Base directory containing template files
        """
        self.templates_dir = templates_dir
        self.framework_dirs = self._discover_framework_dirs()
        self.templates: Dict[str, BaseTemplate] = {}

        # Load all templates
        self._load_templates()

    def _discover_framework_dirs(self) -> Dict[str, Path]:
        """
        Discover framework-specific template directories.

        Returns:
            Dictionary mapping framework names to directory paths
        """
        framework_dirs = {}

        # Check for subdirectories in templates_dir
        for item in self.templates_dir.iterdir():
            if item.is_dir():
                framework_dirs[item.name] = item

        return framework_dirs

    def _load_templates(self) -> None:
        """
        Load all templates from the templates directory and framework subdirectories.
        """
        # Load base templates
        self._load_templates_from_dir(self.templates_dir, framework=None)

        # Load framework-specific templates
        for framework, framework_dir in self.framework_dirs.items():
            self._load_templates_from_dir(framework_dir, framework=framework)

    def _load_templates_from_dir(
        self, directory: Path, framework: Optional[str] = None
    ) -> None:
        """
        Load templates from a specific directory.

        Args:
            directory: Directory containing template files
            framework: Framework name if these are framework-specific templates
        """
        for template_file in directory.glob("*.template"):
            # Create a template ID incorporating the framework if applicable
            template_id = template_file.name
            if framework:
                # Add framework prefix to template ID
                template_id = f"{framework}:{template_id}"

            # Extract parent template from the first line if specified
            parent = None
            with open(template_file, "r") as f:
                first_line = f.readline().strip()
                parent_match = re.match(r"#\s*extends\s+(\S+)", first_line)
                if parent_match:
                    parent = parent_match.group(1)

            # Create and register the template
            BaseTemplate(template_id, file_path=template_file, parent=parent)

    def get_template(self, template_id: str) -> Optional[BaseTemplate]:
        """
        Get a template by ID.

        Args:
            template_id: Template identifier

        Returns:
            The template if found, None otherwise
        """
        return BaseTemplate.get_template(template_id)

    def find_template_for_error(
        self, error_type: str, framework: Optional[str] = None
    ) -> Optional[BaseTemplate]:
        """
        Find the most appropriate template for a given error type and framework.

        Args:
            error_type: Type of error to handle
            framework: Framework context (if applicable)

        Returns:
            The most appropriate template, or None if no suitable template is found
        """
        # First try framework-specific template if framework is specified
        if framework:
            framework_template_id = f"{framework}:{error_type}.py.template"
            framework_template = self.get_template(framework_template_id)
            if framework_template:
                return framework_template

        # Fall back to generic template
        generic_template_id = f"{error_type}.py.template"
        return self.get_template(generic_template_id)

    def list_templates(self, framework: Optional[str] = None) -> List[str]:
        """
        List available templates, optionally filtered by framework.

        Args:
            framework: Framework to filter by (optional)

        Returns:
            List of template IDs
        """
        all_templates = BaseTemplate.list_templates()

        # Filter by framework if specified
        if framework:
            return [t for t in all_templates if t.startswith(f"{framework}:")]

        return all_templates


# Example usage
if __name__ == "__main__":
    # Example template
    example_template = """
# Template for adding a try-except block around code
# The following variables will be replaced:
# - {{ code_block }}: The original code block to wrap in try-except
# - {{ exception_type }}: The exception type to catch
# - {{ error_message }}: The error message for the exception
# - {{ recovery_action }}: Code to run when the exception is caught
# - {{ log_error }}: Whether to include error logging (true/false)

{% block header %}
# Original code:
# {{ code_block }}

# Fixed code with try-except block:
{% endblock %}

try:
{{ code_block }}
except {{ exception_type }} as e:
{% if log_error == "true" %}
    logger.error(f"{{ error_message }}: {str(e)}")
{% endif %}
    {{ recovery_action }}
"""

    # Create a template
    template = BaseTemplate("try_except_block", content=example_template)

    # Render it with variables
    rendered = template.render(
        {
            "code_block": "    value = int(user_input)",
            "exception_type": "ValueError",
            "error_message": "Invalid user input",
            "recovery_action": "    value = 0  # Default value on error",
            "log_error": "true",
        }
    )

    print("Rendered template:")
    print(rendered)
