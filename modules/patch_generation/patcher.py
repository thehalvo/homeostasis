"""
Patch generation module for creating code fixes.
"""

import re
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Import diff utilities
from modules.patch_generation.diff_utils import (
    extract_code_block,
    generate_diff,
    identify_code_block,
)

# Import indentation utilities
from modules.patch_generation.indent_utils import (
    adjust_indentation_for_context,
    generate_line_indentation_map,
    get_line_indentation,
    normalize_indentation,
)

# Import LLM patch generator
from modules.patch_generation.llm_patch_generator import create_llm_patch_generator

# Import hierarchical template system
from modules.patch_generation.template_system import BaseTemplate, TemplateManager

# Templates directory
TEMPLATES_DIR = Path(__file__).parent / "templates"


class PatchTemplate:
    """
    Class representing a patch template for a specific error type.

    This class is maintained for backward compatibility. New code should use
    the BaseTemplate class from template_system.py.
    """

    def __init__(self, name: str, template_path: Path):
        """
        Initialize a patch template.

        Args:
            name: The name of the template
            template_path: Path to the template file
        """
        self.name = name
        self.template_path = template_path
        self.template = self._load_template()

        # Create a BaseTemplate instance for more advanced rendering
        self._base_template = BaseTemplate(name, file_path=template_path)

    def _load_template(self) -> str:
        """
        Load the template content from file.

        Returns:
            Template content as a string
        """
        with open(self.template_path, "r") as f:
            return f.read()

    def render(self, variables: Dict[str, str]) -> str:
        """
        Render the template with the provided variables.

        This method uses the hierarchical template system for rendering,
        falling back to the old implementation if needed.

        Args:
            variables: Dictionary of variable names and values to substitute

        Returns:
            Rendered template as a string
        """
        # Use the hierarchical template system if available
        if self._base_template:
            return self._base_template.render(variables)

        # Legacy implementation as fallback
        result = self.template

        # Replace each variable
        for var_name, var_value in variables.items():
            placeholder = "{{ " + var_name + " }}"
            result = result.replace(placeholder, var_value)

        # Process conditionals (simple implementation)
        # Format: {% if condition %} content {% endif %}
        pattern = r"\{\% if ([^\}]+) \%\}([^\{]+)\{\% endif \%\}"
        for match in re.finditer(pattern, result):
            condition = match.group(1).strip()
            content = match.group(2)

            # Evaluate the condition (very simple evaluation)
            # Only supports var_name == "value" pattern
            condition_met = False
            if "==" in condition:
                parts = condition.split("==")
                var_name = parts[0].strip()
                var_value = parts[1].strip().strip('"').strip("'")
                condition_met = variables.get(var_name) == var_value

            # Replace the conditional block
            if condition_met:
                result = result.replace(match.group(0), content)
            else:
                result = result.replace(match.group(0), "")

        return result


class PatchGenerator:
    """
    Class for generating code patches based on error analysis.
    """

    def __init__(self, templates_dir: Path = TEMPLATES_DIR, enable_llm: bool = True):
        """
        Initialize the patch generator.

        Args:
            templates_dir: Directory containing patch templates
            enable_llm: Whether to enable LLM-powered patch generation
        """
        self.templates_dir = templates_dir
        self.enable_llm = enable_llm

        # Initialize the template manager for hierarchical templates
        self.template_manager = TemplateManager(templates_dir)

        # Load legacy templates for backward compatibility
        self.templates = self._load_templates()

        # Initialize LLM patch generator if enabled
        self.llm_generator = None
        if enable_llm:
            try:
                self.llm_generator = create_llm_patch_generator()
                print("LLM patch generation enabled")
            except Exception as e:
                print(f"Warning: Could not initialize LLM patch generator: {e}")
                self.enable_llm = False

        # Mappings from error types to templates
        self.error_template_mapping = {
            "dict_key_not_exists": "keyerror_fix.py.template",
            "list_index_out_of_bounds": "list_index_error.py.template",
            "invalid_int_conversion": "int_conversion_error.py.template",
            "exception_handling_needed": "try_except_block.py.template",
            "function_needs_improvement": "function_replacement.py.template",
            "transaction_error": "transaction_error.py.template",
            "fastapi_async_mismatch": "fastapi_async_fix.py.template",
            "attribute_access_error": "attribute_error.py.template",
            "attribute_not_exists": "attribute_error.py.template",
            "type_mismatch_error": "type_error_fix.py.template",
        }

        # Framework detection patterns
        self.framework_patterns = {
            "fastapi": [r"from\s+fastapi\s+import", r"app\s*=\s*FastAPI\("],
            "django": [
                r"from\s+django",
                r"from\s+rest_framework",
                r"class\s+\w+View\(.*View\)",
            ],
            "sqlalchemy": [
                r"from\s+sqlalchemy",
                r"from\s+sqlalchemy\.orm",
                r"session\.commit\(",
                r"Base\s*=\s*declarative_base\(\)",
            ],
            "flask": [r"from\s+flask\s+import", r"app\s*=\s*Flask\("],
        }

        # Templates that support multi-line patching
        self.multiline_templates = {
            "try_except_block.py.template",
            "function_replacement.py.template",
            "transaction_error.py.template",
        }

        # Additional mappings specific to FastAPI bugs
        self.known_bugs_mapping = {
            "bug_1": {  # Missing error handling for non-existent IDs in get_todo
                "template": "fastapi:keyerror_fix.py.template",  # Now using framework-specific template
                "variables": {
                    "key_name": "todo_id",
                    "dict_name": "todo_db",
                    "status_code": "404",
                },
                "file_path": "services/example_service/app.py",
                "line_range": (73, 79),
                "framework": "fastapi",
            },
            "bug_2": {  # Missing completed field initialization
                "template": "missing_field_init.py.template",
                "variables": {
                    "dict_name": "todo_dict",
                    "field_name": "completed",
                    "default_value": "False",
                    "other_field": "id",
                    "other_value": "todo_id",
                },
                "file_path": "services/example_service/app.py",
                "line_range": (65, 70),
                "framework": "fastapi",
            },
            "bug_3": {  # Incorrect parameter in dict() method
                "template": "dict_missing_param.py.template",
                "variables": {"object": "todo", "dict_method": "dict"},
                "file_path": "services/example_service/app.py",
                "line_range": (90, 92),
                "framework": "fastapi",
            },
            "bug_4": {  # Unsafe list slicing
                "template": "list_index_error.py.template",
                "variables": {
                    "list_name": "todos",
                    "start_index": "skip",
                    "end_index": "skip + limit",
                },
                "file_path": "services/example_service/app.py",
                "line_range": (58, 61),
                "framework": "fastapi",
            },
            "bug_5": {  # Unsafe environment variable conversion
                "template": "int_conversion_error.py.template",
                "variables": {
                    "var_name": "port",
                    "default_value": "8000",
                    "env_var": "os.environ",
                    "env_var_name": "PORT",
                },
                "file_path": "services/example_service/app.py",
                "line_range": (115, 117),
                "framework": "fastapi",
            },
        }

    def _load_templates(self) -> Dict[str, PatchTemplate]:
        """
        Load legacy templates from the templates directory.

        Used for backward compatibility with existing code.

        Returns:
            Dictionary of template names to PatchTemplate objects
        """
        templates = {}

        for template_file in self.templates_dir.glob("*.template"):
            template_name = template_file.name
            templates[template_name] = PatchTemplate(template_name, template_file)

        return templates

    def detect_framework(self, file_path: Path) -> Optional[str]:
        """
        Detect which framework a file is using based on code patterns.

        Args:
            file_path: Path to the code file

        Returns:
            Framework name if detected, None otherwise
        """
        if not file_path.exists():
            return None

        try:
            with open(file_path, "r") as f:
                content = f.read()

            # Check each framework's patterns
            for framework, patterns in self.framework_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, content):
                        return framework
        except Exception as e:
            print(f"Error detecting framework: {e}")

        return None

    def generate_patch_from_analysis(
        self, analysis: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a patch based on error analysis.

        Args:
            analysis: Error analysis results

        Returns:
            Patch details if a suitable template is found, None otherwise
        """
        root_cause = analysis.get("root_cause")
        file_path = analysis.get("file_path")

        # Try LLM-powered patch generation first if enabled
        if self.enable_llm and self.llm_generator:
            try:
                llm_patch = self.llm_generator.generate_patch_from_analysis(analysis)
                if (
                    llm_patch and llm_patch.get("confidence", 0) > 0.3
                ):  # Minimum confidence threshold
                    print(
                        f"Generated LLM patch with confidence {llm_patch.get('confidence', 0):.2f}"
                    )
                    return llm_patch
                elif llm_patch:
                    print(
                        f"LLM patch confidence too low ({llm_patch.get('confidence', 0):.2f}), falling back to templates"
                    )
            except Exception as e:
                print(f"LLM patch generation failed: {e}, falling back to templates")

        # Fall back to template-based patch generation
        template_name = self.error_template_mapping.get(root_cause)
        if not template_name:
            # If no template and LLM is disabled/failed, return None
            return None

        # Detect the framework if a file path is provided
        framework = None
        if file_path:
            framework = self.detect_framework(Path(file_path))

        # Try to find the most appropriate template using the template manager
        template = None

        if framework:
            # Try framework-specific template first
            framework_template_id = f"{framework}:{template_name}"
            template = self.template_manager.get_template(framework_template_id)

        # Fall back to generic template if no framework-specific template was found
        if not template:
            template = self.template_manager.get_template(template_name)

        # If still no template found, try legacy templates
        if not template and template_name in self.templates:
            template = self.templates[template_name]
        elif not template:
            return None

        # Determine if this is a multi-line patch template
        is_multiline = template_name in self.multiline_templates

        # For a generic error, we can't determine the variables or code location
        # without more context, so return a template example with placeholders
        if isinstance(template, BaseTemplate):
            example_code = template.get_block("main")
            context_variables = [
                var["name"] for var in template.metadata.get("variables", [])
            ]
        else:
            # Legacy template handling
            example_code = template.template
            context_variables = self._get_template_variables(template_name)

        result = {
            "template_name": template_name,
            "root_cause": root_cause,
            "patch_type": "example",
            "example_code": example_code,
            "note": "This is an example patch. You need to manually apply it to the code.",
            "is_multiline": is_multiline,
            "framework": framework,
        }

        # Add additional information for multi-line patches
        if is_multiline:
            result["needs_code_context"] = True
            result["context_variables"] = context_variables

        return result

    def generate_patch_for_known_bug(self, bug_id: str) -> Optional[Dict[str, Any]]:
        """
        Generate a patch for a known bug.

        Args:
            bug_id: ID of the known bug

        Returns:
            Patch details if the bug ID is recognized, None otherwise
        """
        if bug_id not in self.known_bugs_mapping:
            return None

        bug_info = self.known_bugs_mapping[bug_id]
        template_name = bug_info["template"]
        framework = bug_info.get("framework")

        # Try to find the template using the hierarchical template system
        template = self.template_manager.get_template(template_name)

        # If not found and it's a legacy template name, try the legacy templates
        if (
            not template
            and ":" not in template_name
            and template_name in self.templates
        ):
            template = self.templates[template_name]
        elif not template:
            return None

        # Render the template with the provided variables
        if isinstance(template, BaseTemplate):
            rendered_code = template.render(bug_info["variables"])
        else:
            rendered_code = template.render(bug_info["variables"])

        # Build the patch information
        patch = {
            "bug_id": bug_id,
            "template_name": template_name,
            "file_path": bug_info["file_path"],
            "line_range": bug_info["line_range"],
            "variables": bug_info["variables"],
            "patch_type": "specific",
            "patch_code": rendered_code,
            "patch_id": str(uuid.uuid4()),
        }

        # Include framework information if available
        if framework:
            patch["framework"] = framework

        return patch

    def generate_patches_for_all_known_bugs(self) -> List[Dict[str, Any]]:
        """
        Generate patches for all known bugs.

        Returns:
            List of patch details for all known bugs
        """
        patches = []

        for bug_id in self.known_bugs_mapping:
            patch = self.generate_patch_for_known_bug(bug_id)
            if patch:
                patches.append(patch)

        return patches

    def apply_patch(self, patch: Dict[str, Any], project_root: Path) -> bool:
        """
        Apply a patch to the codebase.

        Args:
            patch: Patch details
            project_root: Root directory of the project

        Returns:
            True if the patch was applied successfully, False otherwise
        """
        # Handle LLM-generated patches
        if patch.get("patch_type") in ["llm_generated", "llm_generated_fallback"]:
            return self._apply_llm_patch(patch, project_root)

        if patch["patch_type"] != "specific":
            return False

        file_path = project_root / patch["file_path"]
        line_range = patch["line_range"]
        patch_code = patch["patch_code"]
        template_name = patch.get("template_name", "")

        # Check if this is a multi-line patch template
        is_multiline = template_name in self.multiline_templates

        # Read the file
        with open(file_path, "r") as f:
            original_content = f.read()

        lines = original_content.splitlines()

        # Extract the code comment lines that explain the bug
        # from the patch code (lines starting with #)
        comment_lines = []
        for line in patch_code.split("\n"):
            if line.strip().startswith("# "):
                comment_lines.append(line)

        # Keep only the actual code lines (not starting with #)
        code_lines = [
            line
            for line in patch_code.split("\n")
            if not line.strip().startswith("#") and line.strip()
        ]

        # Create a backup of the file
        backup_path = file_path.with_suffix(file_path.suffix + ".bak")
        shutil.copy2(file_path, backup_path)

        try:
            if not is_multiline:
                # Simple line replacement with improved indentation handling
                if line_range[0] <= len(lines):
                    # Extract the target line and surrounding context
                    start_idx = max(0, line_range[0] - 3)
                    end_idx = min(len(lines), line_range[1] + 3)
                    context_lines = lines[start_idx:end_idx]
                    context_block = "\n".join(context_lines)

                    # Get indentation from the context
                    target_line = lines[
                        line_range[0] - 1
                    ]  # Adjust for 0-based indexing
                    target_indentation = get_line_indentation(target_line)

                    # Join the code lines
                    code_block = "\n".join(code_lines)

                    # Apply appropriate indentation to the code block
                    formatted_code = adjust_indentation_for_context(
                        code_block,
                        context_block,
                        {0: target_indentation},  # Start with target line indentation
                    )

                    # Replace the lines in the file
                    new_lines = lines.copy()
                    new_lines[line_range[0] - 1 : line_range[1]] = (
                        formatted_code.splitlines()
                    )
                    new_content = "\n".join(new_lines)

                    # Write the modified content back to the file
                    with open(file_path, "w") as f:
                        f.write(new_content)

                    # Generate a diff for reference and include it in the patch metadata
                    diff = generate_diff(
                        original_content,
                        new_content,
                        filename=str(file_path.relative_to(project_root)),
                    )
                    patch["diff"] = diff

                    # Clean up the backup if successful
                    if backup_path.exists():
                        backup_path.unlink()

                    return True
            else:
                # Multi-line patch with advanced indentation handling
                # Extract the code block to be replaced
                start_line, end_line = line_range

                # If we need to expand the line range to get a complete code block
                if patch.get("expand_line_range", False):
                    # Identify the code block containing the specified line
                    focus_line = (
                        start_line + end_line
                    ) // 2  # Use middle line as focus
                    expanded_range = identify_code_block(original_content, focus_line)
                    start_line, end_line = expanded_range

                # Normalize the patch code to remove any indentation
                normalized_patch = normalize_indentation("\n".join(code_lines))

                # Generate an indentation map for the file
                line_indent_map = generate_line_indentation_map(file_path)

                # Get base indentation from the first line of the block
                base_indentation = line_indent_map.get(start_line, "")

                # Extract context for indentation analysis
                context_start = max(1, start_line - 5)
                context_end = min(len(lines), end_line + 5)
                context_block = "\n".join(lines[context_start - 1 : context_end])

                # Apply context-aware indentation to the patch code
                formatted_code = adjust_indentation_for_context(
                    normalized_patch,
                    context_block,
                    {0: base_indentation},  # Start with base indentation
                )

                # Apply the patch by replacing the entire block
                new_lines = lines.copy()
                new_lines[start_line - 1 : end_line] = formatted_code.splitlines()
                new_content = "\n".join(new_lines)

                # Write the modified content back to the file
                with open(file_path, "w") as f:
                    f.write(new_content)

                # Generate a diff for reference and include it in the patch metadata
                diff = generate_diff(
                    original_content,
                    new_content,
                    filename=str(file_path.relative_to(project_root)),
                )
                patch["diff"] = diff

                # Clean up the backup if successful
                if backup_path.exists():
                    backup_path.unlink()

                return True

        except Exception as e:
            # If anything goes wrong, restore from backup if it exists
            if backup_path.exists():
                shutil.copy2(backup_path, file_path)
                backup_path.unlink()
            print(f"Error applying patch: {e}")
            return False

        return False

    def _apply_llm_patch(self, patch: Dict[str, Any], project_root: Path) -> bool:
        """
        Apply an LLM-generated patch to the codebase.

        Args:
            patch: LLM-generated patch details
            project_root: Root directory of the project

        Returns:
            True if the patch was applied successfully, False otherwise
        """
        try:
            file_path = patch.get("file_path", "")
            if not file_path:
                print("No file path specified in LLM patch")
                return False

            target_file = project_root / file_path

            # Use the LLM generator's apply method if available
            if self.llm_generator:
                return self.llm_generator.apply_llm_patch(patch, str(target_file))
            else:
                # Fallback implementation
                return self._apply_llm_patch_fallback(patch, target_file)

        except Exception as e:
            print(f"Error applying LLM patch: {e}")
            return False

    def _apply_llm_patch_fallback(
        self, patch: Dict[str, Any], target_file: Path
    ) -> bool:
        """
        Fallback method to apply LLM patch without LLM generator.

        Args:
            patch: LLM patch details
            target_file: Target file path

        Returns:
            True if successful
        """
        try:
            if not target_file.exists():
                print(f"Target file does not exist: {target_file}")
                return False

            # Read current content
            with open(target_file, "r", encoding="utf-8") as f:
                original_content = f.read()

            # Extract changes from patch
            changes = patch.get("changes", [])
            if not changes:
                print("No changes specified in LLM patch")
                return False

            # Apply changes (simplified version)
            lines = original_content.split("\n")

            for change in reversed(
                sorted(changes, key=lambda x: x.get("line_start", 0))
            ):
                start_line = change.get("line_start", 1) - 1  # Convert to 0-based
                end_line = change.get("line_end", start_line + 1) - 1
                new_code = change.get("new_code", "").split("\n")

                if 0 <= start_line < len(lines):
                    lines[start_line : end_line + 1] = new_code

            # Write modified content
            backup_path = target_file.with_suffix(target_file.suffix + ".bak")
            shutil.copy2(target_file, backup_path)

            with open(target_file, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

            print(f"Applied LLM patch to {target_file}")
            return True

        except Exception as e:
            print(f"Error in LLM patch fallback: {e}")
            return False

    def _get_template_variables(self, template_name: str) -> List[str]:
        """
        Extract variable names from a template.

        Args:
            template_name: Name of the template

        Returns:
            List of variable names in the template
        """
        if template_name not in self.templates:
            return []

        template = self.templates[template_name]
        content = template.template

        # Find all variables in the format {{ variable_name }}
        variables = set()
        for match in re.finditer(r"\{\{\s*([\w_]+)\s*\}\}", content):
            variables.add(match.group(1))

        return list(variables)

    def generate_multiline_patch(
        self,
        error_type: str,
        file_path: Path,
        line_range: Tuple[int, int],
        variables: Dict[str, str],
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a multi-line patch for a specific error.

        Args:
            error_type: Type of error to fix
            file_path: Path to the file to patch
            line_range: Tuple of (start_line, end_line) to replace
            variables: Template variables to use

        Returns:
            Patch details if generation is successful, None otherwise
        """
        # Check if we have a template for this error type
        template_name = self.error_template_mapping.get(error_type)
        if not template_name:
            return None

        # Detect the framework if possible
        framework = self.detect_framework(file_path)

        # Try to get the most appropriate template using the hierarchical template system
        template = None
        template_id = template_name

        if framework:
            # Try framework-specific template first
            framework_template_id = f"{framework}:{template_name}"
            template = self.template_manager.get_template(framework_template_id)
            if template:
                template_id = framework_template_id

        # Fall back to generic template if no framework-specific template was found
        if not template:
            template = self.template_manager.get_template(template_name)

        # If still no template found, try legacy templates
        if not template and template_name in self.templates:
            template = self.templates[template_name]
        elif not template:
            return None

        # Check if this is a multi-line template
        if template_name not in self.multiline_templates:
            return None

        # If code_block variable is needed but not provided, extract it from the file
        required_variables = []
        if isinstance(template, BaseTemplate):
            required_variables = [
                var["name"] for var in template.metadata.get("variables", [])
            ]
        else:
            required_variables = self._get_template_variables(template_name)

        if "code_block" in required_variables and "code_block" not in variables:
            try:
                code_block = extract_code_block(file_path, line_range)
                # Normalize indentation for template using our utility
                normalized_code = normalize_indentation(code_block)
                variables["code_block"] = normalized_code
            except Exception as e:
                print(f"Error extracting code block: {e}")
                return None

        # Render the template with the provided variables
        if isinstance(template, BaseTemplate):
            rendered_code = template.render(variables)
        else:
            rendered_code = template.render(variables)

        # Create the patch
        patch = {
            "template_name": template_id,
            "file_path": str(file_path),
            "line_range": line_range,
            "variables": variables,
            "patch_type": "specific",
            "patch_code": rendered_code,
            "patch_id": str(uuid.uuid4()),
            "is_multiline": True,
            "expand_line_range": variables.get("expand_line_range", "false") == "true",
        }

        # Include framework information if available
        if framework:
            patch["framework"] = framework

        return patch


if __name__ == "__main__":
    # Example usage
    generator = PatchGenerator()

    # Check available templates
    print("Available templates:")
    for template_name in generator.templates:
        is_multiline = (
            "(multi-line)" if template_name in generator.multiline_templates else ""
        )
        print(f"- {template_name} {is_multiline}")

    # Generate a patch for a known bug
    patch = generator.generate_patch_for_known_bug("bug_1")
    if patch:
        print("\nGenerated patch for bug_1:")
        print(f"File: {patch['file_path']}")
        print(f"Line range: {patch['line_range']}")
        print(f"Patch code:\n{patch['patch_code']}")

    # Example of multi-line patch generation
    try:
        # Try to create a multi-line patch for a try-except block
        sample_file = Path("services/example_service/app.py")
        if sample_file.exists():
            multiline_patch = generator.generate_multiline_patch(
                "exception_handling_needed",
                sample_file,
                (140, 142),  # Line range for port conversion code
                {
                    "exception_type": "ValueError",
                    "error_message": "Invalid port value",
                    "recovery_action": "port = 8000  # Default to 8000 on error",
                    "log_error": "true",
                },
            )

            if multiline_patch:
                print("\nGenerated multi-line patch:")
                print(f"File: {multiline_patch['file_path']}")
                print(f"Line range: {multiline_patch['line_range']}")
                print(f"Patch code:\n{multiline_patch['patch_code']}")
    except Exception as e:
        print(f"Error generating multi-line patch: {e}")

    # Generate patches for all known bugs
    patches = generator.generate_patches_for_all_known_bugs()
    print(f"\nGenerated {len(patches)} patches for known bugs.")
