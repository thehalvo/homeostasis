"""
Tests for patch generator.
"""
import os
import sys
import pytest
import tempfile
import shutil
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.patch_generation.patcher import PatchGenerator, PatchTemplate
from modules.patch_generation.diff_utils import generate_diff, identify_code_block


def test_patch_template_loading():
    """Test loading a patch template."""
    generator = PatchGenerator()
    
    # Check if templates were loaded
    assert len(generator.templates) > 0
    
    # Check if specific templates exist
    assert "keyerror_fix.py.template" in generator.templates
    assert "missing_field_init.py.template" in generator.templates
    assert "dict_missing_param.py.template" in generator.templates
    assert "list_index_error.py.template" in generator.templates
    assert "int_conversion_error.py.template" in generator.templates


def test_template_rendering():
    """Test rendering a template with variables."""
    # Create a simple template
    template_content = "# Template\nvar1 = {{ var1 }}\nvar2 = {{ var2 }}"
    template_path = Path("test_template.txt")
    
    # Write the template to a file
    with open(template_path, "w") as f:
        f.write(template_content)
    
    try:
        # Create a template object
        template = PatchTemplate("test", template_path)
        
        # Render the template
        variables = {"var1": "value1", "var2": "value2"}
        rendered = template.render(variables)
        
        # Check the result
        assert "var1 = value1" in rendered
        assert "var2 = value2" in rendered
    
    finally:
        # Clean up
        if template_path.exists():
            template_path.unlink()


def test_generate_patch_for_known_bug():
    """Test generating a patch for a known bug."""
    generator = PatchGenerator()
    
    # Generate a patch for bug_1
    patch = generator.generate_patch_for_known_bug("bug_1")
    
    # Check the patch
    assert patch is not None
    assert patch["bug_id"] == "bug_1"
    assert patch["file_path"] == "services/example_service/app.py"
    assert "patch_code" in patch
    assert "HTTPException" in patch["patch_code"]
    assert "patch_id" in patch


def test_generate_patches_for_all_known_bugs():
    """Test generating patches for all known bugs."""
    generator = PatchGenerator()
    
    # Generate patches for all known bugs
    patches = generator.generate_patches_for_all_known_bugs()
    
    # Check the patches
    assert len(patches) == 5  # We have 5 known bugs
    
    # Check if all bugs are covered
    bug_ids = [patch["bug_id"] for patch in patches]
    assert "bug_1" in bug_ids
    assert "bug_2" in bug_ids
    assert "bug_3" in bug_ids
    assert "bug_4" in bug_ids
    assert "bug_5" in bug_ids


def test_template_variables_extraction():
    """Test extraction of template variables."""
    generator = PatchGenerator()
    
    # Check for template variables in keyerror_fix.py.template
    variables = generator._get_template_variables("keyerror_fix.py.template")
    assert "key_name" in variables
    assert "dict_name" in variables
    
    # Check for template variables in try_except_block.py.template
    variables = generator._get_template_variables("try_except_block.py.template")
    assert "code_block" in variables
    assert "exception_type" in variables
    assert "error_message" in variables
    assert "recovery_action" in variables


def test_multiline_patch_generation():
    """Test generating a multi-line patch."""
    # Create a temporary Python file for testing
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
        temp_file.write(b"""def sample_function():
    # This is a function that needs error handling
    value = int(input_value)
    return value * 2
""")
        temp_path = Path(temp_file.name)
    
    try:
        generator = PatchGenerator()
        
        # Generate a multi-line patch for adding try-except block
        patch = generator.generate_multiline_patch(
            "exception_handling_needed",
            temp_path,
            (2, 4),  # Line range for the code block
            {
                "exception_type": "ValueError",
                "error_message": "Invalid input value",
                "recovery_action": "return 0  # Return default value on error",
                "log_error": "true"
            }
        )
        
        # Verify the patch
        assert patch is not None
        assert patch["template_name"] == "try_except_block.py.template"
        assert patch["file_path"] == str(temp_path)
        assert patch["line_range"] == (2, 4)
        assert "try:" in patch["patch_code"]
        assert "except ValueError as e:" in patch["patch_code"]
        assert "Invalid input value" in patch["patch_code"]
        assert "return 0" in patch["patch_code"]
        assert patch["is_multiline"] is True
    
    finally:
        # Clean up the temporary file
        if temp_path.exists():
            temp_path.unlink()


def test_identify_code_block():
    """Test identifying a code block from a line number."""
    # Sample code with different blocks
    code = """def function_one():
    # This is the first function
    x = 1
    y = 2
    return x + y

def function_two():
    # This is the second function
    if condition:
        # This is a nested block
        do_something()
        if nested_condition:
            do_nested_thing()
    return result
"""
    
    # Test identifying the first function block
    block_range = identify_code_block(code, 3)  # Line with x = 1
    assert block_range == (1, 5)
    
    # Test identifying the nested block
    block_range = identify_code_block(code, 11)  # Line with do_nested_thing()
    assert block_range == (11, 12)


def test_multiline_patch_application():
    """Test applying a multi-line patch."""
    # Create a temporary directory for the test
    test_dir = Path(tempfile.mkdtemp())
    
    try:
        # Create a temporary file with code
        test_file = test_dir / "test_app.py"
        with open(test_file, "w") as f:
            f.write("""def process_data(data):
    # Process the input data
    result = data['value'] * 2
    return result
""")
        
        # Create a PatchGenerator instance
        generator = PatchGenerator()
        
        # Generate a multi-line patch
        patch = generator.generate_multiline_patch(
            "exception_handling_needed",
            test_file,
            (2, 3),  # Line range for the code to wrap in try-except
            {
                "exception_type": "KeyError",
                "error_message": "Missing 'value' key in data",
                "recovery_action": "return 0  # Default value",
                "log_error": "true"
            }
        )
        
        # Apply the patch
        result = generator.apply_patch(patch, test_dir)
        assert result is True
        
        # Read the patched file
        with open(test_file, "r") as f:
            patched_content = f.read()
        
        # Check that the patch was applied correctly
        assert "try:" in patched_content
        assert "except KeyError as e:" in patched_content
        assert "Missing 'value' key in data" in patched_content
        assert "return 0  # Default value" in patched_content
        
        # Also check for a diff in the patch metadata
        assert "diff" in patch
    
    finally:
        # Clean up the temporary directory
        shutil.rmtree(test_dir)