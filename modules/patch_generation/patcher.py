"""
Patch generation module for creating code fixes.
"""
import os
import re
import json
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union, Set
import uuid

# Import diff utilities
from modules.patch_generation.diff_utils import (
    generate_diff, parse_diff, apply_diff_to_file,
    identify_code_block, extract_code_block, get_code_context
)

# Templates directory
TEMPLATES_DIR = Path(__file__).parent / "templates"


class PatchTemplate:
    """
    Class representing a patch template for a specific error type.
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

        Args:
            variables: Dictionary of variable names and values to substitute

        Returns:
            Rendered template as a string
        """
        result = self.template
        
        # Replace each variable
        for var_name, var_value in variables.items():
            placeholder = "{{ " + var_name + " }}"
            result = result.replace(placeholder, var_value)
        
        # Process conditionals (simple implementation)
        # Format: {% if condition %} content {% endif %}
        pattern = r'\{\% if ([^\}]+) \%\}([^\{]+)\{\% endif \%\}'
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

    def __init__(self, templates_dir: Path = TEMPLATES_DIR):
        """
        Initialize the patch generator.

        Args:
            templates_dir: Directory containing patch templates
        """
        self.templates_dir = templates_dir
        self.templates = self._load_templates()
        
        # Mappings from error types to templates
        self.error_template_mapping = {
            "dict_key_not_exists": "keyerror_fix.py.template",
            "list_index_out_of_bounds": "list_index_error.py.template",
            "invalid_int_conversion": "int_conversion_error.py.template",
            "exception_handling_needed": "try_except_block.py.template",
            "function_needs_improvement": "function_replacement.py.template"
        }
        
        # Templates that support multi-line patching
        self.multiline_templates = {
            "try_except_block.py.template",
            "function_replacement.py.template"
        }
        
        # Additional mappings specific to FastAPI bugs
        self.known_bugs_mapping = {
            "bug_1": {  # Missing error handling for non-existent IDs in get_todo
                "template": "keyerror_fix.py.template",
                "variables": {
                    "key_name": "todo_id",
                    "dict_name": "todo_db"
                },
                "file_path": "services/example_service/app.py",
                "line_range": (73, 79)
            },
            "bug_2": {  # Missing completed field initialization
                "template": "missing_field_init.py.template",
                "variables": {
                    "dict_name": "todo_dict",
                    "field_name": "completed",
                    "default_value": "False",
                    "other_field": "id",
                    "other_value": "todo_id"
                },
                "file_path": "services/example_service/app.py",
                "line_range": (65, 70)
            },
            "bug_3": {  # Incorrect parameter in dict() method
                "template": "dict_missing_param.py.template",
                "variables": {
                    "object": "todo",
                    "dict_method": "dict"
                },
                "file_path": "services/example_service/app.py",
                "line_range": (90, 92)
            },
            "bug_4": {  # Unsafe list slicing
                "template": "list_index_error.py.template",
                "variables": {
                    "list_name": "todos",
                    "start_index": "skip",
                    "end_index": "skip + limit"
                },
                "file_path": "services/example_service/app.py",
                "line_range": (58, 61)
            },
            "bug_5": {  # Unsafe environment variable conversion
                "template": "int_conversion_error.py.template",
                "variables": {
                    "var_name": "port",
                    "default_value": "8000",
                    "env_var": "os.environ",
                    "env_var_name": "PORT"
                },
                "file_path": "services/example_service/app.py",
                "line_range": (115, 117)
            }
        }
    
    def _load_templates(self) -> Dict[str, PatchTemplate]:
        """
        Load all templates from the templates directory.

        Returns:
            Dictionary of template names to PatchTemplate objects
        """
        templates = {}
        
        for template_file in self.templates_dir.glob("*.template"):
            template_name = template_file.name
            templates[template_name] = PatchTemplate(template_name, template_file)
        
        return templates
    
    def generate_patch_from_analysis(self, analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generate a patch based on error analysis.

        Args:
            analysis: Error analysis results

        Returns:
            Patch details if a suitable template is found, None otherwise
        """
        root_cause = analysis.get("root_cause")
        
        # Try to find a matching template
        template_name = self.error_template_mapping.get(root_cause)
        if not template_name or template_name not in self.templates:
            return None
        
        # For a generic error, we can't determine the variables or code location
        # without more context, so return a template example with placeholders
        template = self.templates[template_name]
        
        # Determine if this is a multi-line patch template
        is_multiline = template_name in self.multiline_templates
        
        result = {
            "template_name": template_name,
            "root_cause": root_cause,
            "patch_type": "example",
            "example_code": template.template,
            "note": "This is an example patch. You need to manually apply it to the code.",
            "is_multiline": is_multiline
        }
        
        # Add additional information for multi-line patches
        if is_multiline:
            result["needs_code_context"] = True
            result["context_variables"] = self._get_template_variables(template_name)
        
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
        
        if template_name not in self.templates:
            return None
        
        template = self.templates[template_name]
        rendered_code = template.render(bug_info["variables"])
        
        return {
            "bug_id": bug_id,
            "template_name": template_name,
            "file_path": bug_info["file_path"],
            "line_range": bug_info["line_range"],
            "variables": bug_info["variables"],
            "patch_type": "specific",
            "patch_code": rendered_code,
            "patch_id": str(uuid.uuid4())
        }
    
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
            lines = f.readlines()
            original_content = "".join(lines)
        
        # Extract the code comment lines that explain the bug
        # from the patch code (lines starting with #)
        comment_lines = []
        for line in patch_code.split("\n"):
            if line.strip().startswith("# "):
                comment_lines.append(line)
        
        # Keep only the actual code lines (not starting with #)
        code_lines = [line for line in patch_code.split("\n") 
                    if not line.strip().startswith("#") and line.strip()]
        
        if not is_multiline:
            # Simple line replacement (original behavior)
            # Format the code to match the file's indentation
            if line_range[0] <= len(lines):
                # Get the indentation of the first line in the range
                first_line = lines[line_range[0] - 1]  # Adjust for 0-based indexing
                indent_match = re.match(r'^(\s+)', first_line)
                indentation = indent_match.group(1) if indent_match else ""
                
                # Apply indentation to each line of the patch code
                formatted_code_lines = [indentation + line for line in code_lines]
                formatted_code = "\n".join(formatted_code_lines) + "\n"
                
                # Replace the lines in the file with the patch
                lines[line_range[0] - 1:line_range[1]] = [formatted_code]
                
                # Write the modified file
                with open(file_path, "w") as f:
                    f.writelines(lines)
                
                return True
        else:
            # Multi-line patch with advanced handling
            try:
                # Extract the code block to be replaced
                start_line, end_line = line_range
                
                # If we need to expand the line range to get a complete code block
                if patch.get("expand_line_range", False):
                    # Identify the code block containing the specified line
                    with open(file_path, "r") as f:
                        file_content = f.read()
                    focus_line = (start_line + end_line) // 2  # Use middle line as focus
                    expanded_range = identify_code_block(file_content, focus_line)
                    start_line, end_line = expanded_range
                
                # Get the original code block
                original_block = extract_code_block(file_path, (start_line, end_line))
                
                # Format the patch code with proper indentation
                first_line = lines[start_line - 1]  # Adjust for 0-based indexing
                indent_match = re.match(r'^(\s+)', first_line)
                base_indentation = indent_match.group(1) if indent_match else ""
                
                # Apply base indentation to each line while preserving relative indentation
                formatted_code_lines = []
                for line in code_lines:
                    if line.strip():  # Not an empty line
                        # Preserve relative indentation beyond the base indentation
                        relative_indent_match = re.match(r'^(\s+)', line)
                        relative_indent = relative_indent_match.group(1) if relative_indent_match else ""
                        code_content = line.lstrip()
                        formatted_code_lines.append(base_indentation + code_content)
                    else:
                        formatted_code_lines.append(line)  # Keep empty lines as is
                
                formatted_code = "\n".join(formatted_code_lines)
                
                # Create a backup of the file
                backup_path = file_path.with_suffix(file_path.suffix + ".bak")
                shutil.copy2(file_path, backup_path)
                
                # Apply the patch by replacing the entire block
                with open(file_path, "r") as f:
                    content = f.read()
                
                # Replace the specific range of lines
                lines_before = content.splitlines()[:start_line - 1]
                lines_after = content.splitlines()[end_line:]
                new_content = "\n".join(lines_before + [formatted_code] + lines_after)
                
                # Write the modified content back to the file
                with open(file_path, "w") as f:
                    f.write(new_content)
                
                # Generate a diff for reference and include it in the patch metadata
                diff = generate_diff(original_content, new_content, 
                                   filename=str(file_path.relative_to(project_root)))
                patch["diff"] = diff
                
                # Clean up the backup if successful
                if backup_path.exists():
                    backup_path.unlink()
                
                return True
                
            except Exception as e:
                # If anything goes wrong, restore from backup if it exists
                backup_path = file_path.with_suffix(file_path.suffix + ".bak")
                if backup_path.exists():
                    shutil.copy2(backup_path, file_path)
                    backup_path.unlink()
                print(f"Error applying multi-line patch: {e}")
                return False
        
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
        for match in re.finditer(r'\{\{\s*([\w_]+)\s*\}\}', content):
            variables.add(match.group(1))
            
        return list(variables)
        
    def generate_multiline_patch(self, error_type: str, file_path: Path, 
                               line_range: Tuple[int, int], 
                               variables: Dict[str, str]) -> Optional[Dict[str, Any]]:
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
        if not template_name or template_name not in self.templates:
            return None
            
        # Check if this is a multi-line template
        if template_name not in self.multiline_templates:
            return None
            
        template = self.templates[template_name]
        
        # If code_block variable is needed but not provided, extract it from the file
        if "code_block" in self._get_template_variables(template_name) and "code_block" not in variables:
            try:
                code_block = extract_code_block(file_path, line_range)
                # Adjust indentation for template (remove common indent)
                code_lines = code_block.splitlines()
                if code_lines:
                    # Find minimum indentation across non-empty lines
                    indents = [len(line) - len(line.lstrip()) for line in code_lines if line.strip()]
                    if indents:
                        min_indent = min(indents)
                        # Remove that amount of indentation from each line
                        code_lines = [line[min_indent:] if line.strip() else line for line in code_lines]
                        code_block = "\n".join(code_lines)
                variables["code_block"] = code_block
            except Exception as e:
                print(f"Error extracting code block: {e}")
                return None
        
        # Render the template with the provided variables
        rendered_code = template.render(variables)
        
        # Create the patch
        return {
            "template_name": template_name,
            "file_path": str(file_path),
            "line_range": line_range,
            "variables": variables,
            "patch_type": "specific",
            "patch_code": rendered_code,
            "patch_id": str(uuid.uuid4()),
            "is_multiline": True,
            "expand_line_range": variables.get("expand_line_range", "false") == "true"
        }


if __name__ == "__main__":
    # Example usage
    generator = PatchGenerator()
    
    # Check available templates
    print("Available templates:")
    for template_name in generator.templates:
        is_multiline = "(multi-line)" if template_name in generator.multiline_templates else ""
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
                    "log_error": "true"
                }
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