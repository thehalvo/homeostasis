"""
Tests for patch generator.
"""
import os
import sys
import pytest
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.patch_generation.patcher import PatchGenerator, PatchTemplate


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
    assert "KeyError" in patch["patch_code"]
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