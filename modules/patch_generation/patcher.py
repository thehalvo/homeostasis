"""
Patch generation module for creating code fixes.
"""
import os
import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import uuid

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
            "invalid_int_conversion": "int_conversion_error.py.template"
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
        
        return {
            "template_name": template_name,
            "root_cause": root_cause,
            "patch_type": "example",
            "example_code": template.template,
            "note": "This is an example patch. You need to manually apply it to the code."
        }
    
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
        
        # Read the file
        with open(file_path, "r") as f:
            lines = f.readlines()
        
        # Extract the code comment lines that explain the bug
        # from the patch code (lines starting with #)
        comment_lines = []
        for line in patch_code.split("\n"):
            if line.strip().startswith("# "):
                comment_lines.append(line)
        
        # Keep only the actual code lines (not starting with #)
        code_lines = [line for line in patch_code.split("\n") 
                     if not line.strip().startswith("#") and line.strip()]
        
        # Format the code to match the file's indentation
        if line_range[0] < len(lines):
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
        
        return False


if __name__ == "__main__":
    # Example usage
    generator = PatchGenerator()
    
    # Check available templates
    print("Available templates:")
    for template_name in generator.templates:
        print(f"- {template_name}")
    
    # Generate a patch for a known bug
    patch = generator.generate_patch_for_known_bug("bug_1")
    if patch:
        print("\nGenerated patch for bug_1:")
        print(f"File: {patch['file_path']}")
        print(f"Line range: {patch['line_range']}")
        print(f"Patch code:\n{patch['patch_code']}")
    
    # Generate patches for all known bugs
    patches = generator.generate_patches_for_all_known_bugs()
    print(f"\nGenerated {len(patches)} patches for known bugs.")