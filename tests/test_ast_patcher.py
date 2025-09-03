"""
Tests for the AST-based patcher.
"""
import os
import unittest
import tempfile
from pathlib import Path

from modules.patch_generation.patcher import PatchGenerator
from modules.patch_generation.ast_patcher import ASTPatcher


class ASTPatcherTests(unittest.TestCase):
    """
    Test cases for the AST-based patcher.
    """
    
    def setUp(self):
        """
        Set up test fixtures.
        """
        # Path to templates directory
        self.templates_dir = Path(__file__).parent.parent / "modules" / "patch_generation" / "templates"
        
        # Initialize components
        self.patch_generator = PatchGenerator(self.templates_dir)
        self.ast_patcher = ASTPatcher(self.templates_dir, self.patch_generator)
        
        # Sample code for testing
        self.sample_code = """import os
from typing import List, Dict, Optional

def process_data(data_list, transform_func=None):
    \"\"\"
    Process a list of data items.
    
    Args:
        data_list: List of data items to process
        transform_func: Optional function to transform values
    
    Returns:
        Dict mapping keys to processed values
    \"\"\"
    result = {}
    
    for item in data_list:
        # This could raise an IndexError if item is empty
        key = item[0]
        
        # This could raise a KeyError if 'value' is not in item
        value = item['value']
        
        # This could raise a ValueError if value is not an integer
        processed = int(value)
        
        # This could raise an AttributeError if transform_func has no 'apply' method
        if transform_func:
            processed = transform_func.apply(processed)
            
        result[key] = processed
        
    return result

class Transformer:
    def __init__(self, factor=2):
        self.factor = factor
        
    def apply(self, value):
        return value * self.factor

# Example usage
data = [
    {'id': 1, 'value': '42'},
    {'id': 2, 'value': 'not-a-number'},
    {}  # Missing required keys
]

try:
    results = process_data(data, Transformer())
    print(results)
except Exception as e:
    print(f"Error: {e}")
"""
        
        # Create a temporary file with the sample code
        self.temp_file = tempfile.NamedTemporaryFile(mode='w+', suffix='.py', delete=False)
        self.temp_file.write(self.sample_code)
        self.temp_file.close()
        
    def tearDown(self):
        """
        Clean up after tests.
        """
        # Remove temporary file
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
            
    def test_analyze_file(self):
        """
        Test that file analysis works correctly.
        """
        result = self.ast_patcher.analyze_file(Path(self.temp_file.name))
        self.assertTrue(result, "File analysis should succeed")
        
        # Check that basic information was extracted
        self.assertIn('process_data', self.ast_patcher.analyzer.get_functions())
        self.assertIn('Transformer', self.ast_patcher.analyzer.get_classes())
        
    def test_generate_exception_handler(self):
        """
        Test generation of exception handlers.
        """
        # Analyze the file first
        self.ast_patcher.analyze_file(Path(self.temp_file.name))
        
        # Generate exception handler for KeyError at line 22 (value = item['value'])
        patch = self.ast_patcher.generate_exception_handler(22, "KeyError")
        
        # Verify the patch was generated
        self.assertIsNotNone(patch, "Exception handler patch should be generated")
        self.assertTrue("try:" in patch["patch_code"], "Patch should include try statement")
        self.assertTrue("except KeyError" in patch["patch_code"], "Patch should catch KeyError")
        
    def test_generate_parameter_check(self):
        """
        Test generation of parameter validation.
        """
        # Analyze the file first
        self.ast_patcher.analyze_file(Path(self.temp_file.name))
        
        # Generate parameter check for data_list
        patch = self.ast_patcher.generate_parameter_check("process_data", "data_list")
        
        # Verify the patch was generated
        self.assertIsNotNone(patch, "Parameter check patch should be generated")
        self.assertTrue("if data_list is None" in patch["patch_code"], 
                       "Patch should check if parameter is None")
        
    def test_generate_type_conversion(self):
        """
        Test generation of type conversion.
        """
        # Analyze the file first
        self.ast_patcher.analyze_file(Path(self.temp_file.name))
        
        # Generate type conversion for int at line 25 (processed = int(value))
        patch = self.ast_patcher.generate_type_conversion(25, "value", "int")
        
        # Verify the patch was generated
        self.assertIsNotNone(patch, "Type conversion patch should be generated")
        self.assertTrue("int" in patch["patch_code"], "Patch should include int conversion")
        
    def test_fix_attribute_error(self):
        """
        Test generation of attribute error fixes.
        """
        # Analyze the file first
        self.ast_patcher.analyze_file(Path(self.temp_file.name))
        
        # Generate attribute error fix for transform_func.apply
        patch = self.ast_patcher.fix_attribute_error(27, "transform_func", "apply")
        
        # Verify the patch was generated
        self.assertIsNotNone(patch, "Attribute error fix should be generated")
        self.assertTrue("hasattr(transform_func, \"apply\")" in patch["patch_code"], 
                       "Patch should check for attribute existence")
        
    def test_get_variables_in_scope(self):
        """
        Test getting variables in scope at a specific line.
        """
        # Analyze the file first
        self.ast_patcher.analyze_file(Path(self.temp_file.name))
        
        # Get variables in scope at line 22 (inside the for loop where 'item' is in scope)
        variables = self.ast_patcher.get_variables_in_scope(22)
        
        # Verify that function parameters and local variables are included
        self.assertIn("data_list", variables, "Parameter should be in scope")
        self.assertIn("transform_func", variables, "Parameter should be in scope")
        self.assertIn("result", variables, "Local variable should be in scope")
        # TODO: Loop variables like 'item' are not currently tracked by the AST analyzer
        # This would require enhancing the analyzer to track loop scopes


if __name__ == "__main__":
    unittest.main()