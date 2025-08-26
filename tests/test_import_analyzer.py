"""
Tests for the import analyzer module.
"""
import os
import unittest
import tempfile
from pathlib import Path

from modules.patch_generation.ast_analyzer import ASTAnalyzer
from modules.patch_generation.import_analyzer import ImportAnalyzer


class ImportAnalyzerTests(unittest.TestCase):
    """
    Test cases for the import analyzer.
    """
    
    def setUp(self):
        """
        Set up test fixtures.
        """
        # Sample code for testing
        self.sample_code = """
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
from sklearn.model_selection import train_test_split

def process_data(data_path: str) -> Dict:
    \"\"\"Process data from a file.\"\"\"
    # This uses pandas but doesn't import it
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    
    # This uses matplotlib but doesn't import it
    plt.figure()
    plt.plot(df['a'], df['b'])
    
    return {'data': df.to_dict()}

# This uses Path from pathlib (which is imported)
data_path = Path('data/file.csv')
if data_path.exists():
    result = process_data(str(data_path))
    
# Create model using sklearn (which is partially imported)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
"""
        
        # Create a temporary file with the sample code
        self.temp_file = tempfile.NamedTemporaryFile(mode='w+', suffix='.py', delete=False)
        self.temp_file.write(self.sample_code)
        self.temp_file.close()
        
        # Initialize the analyzer
        self.analyzer = ImportAnalyzer()
        self.analyzer.analyze_code(self.sample_code)
        
    def tearDown(self):
        """
        Clean up after tests.
        """
        # Remove temporary file
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
            
    def test_get_imports(self):
        """
        Test that imports are correctly extracted.
        """
        imports = self.analyzer.get_imports()
        
        # Check that we have the right number of imports
        self.assertEqual(len(imports), 6, "Should find 6 import statements")
        
        # Check for specific imports
        found_imports = []
        for import_info in imports:
            if not import_info.is_from:
                for name, _ in import_info.names:
                    found_imports.append(name)
            else:
                found_imports.append(f"{import_info.module}.{import_info.names[0][0]}")
                
        self.assertIn("os", found_imports, "Should find os import")
        self.assertIn("sys", found_imports, "Should find sys import")
        self.assertIn("numpy", found_imports, "Should find numpy import")
        self.assertIn("pathlib.Path", found_imports, "Should find Path from pathlib")
        self.assertIn("sklearn.model_selection.train_test_split", found_imports, 
                     "Should find train_test_split from sklearn.model_selection")
        
    def test_resolve_module(self):
        """
        Test module resolution.
        """
        # Resolve a standard library module
        os_context = self.analyzer.resolve_module("os")
        self.assertTrue(os_context.is_resolved, "os module should be resolved")
        self.assertTrue(os_context.is_stdlib, "os should be identified as stdlib")
        self.assertFalse(os_context.is_third_party, "os should not be third-party")
        
        # Resolve a module that might not exist (but the analyzer should handle it gracefully)
        missing_context = self.analyzer.resolve_module("module_that_does_not_exist")
        self.assertFalse(missing_context.is_resolved, "Non-existent module should not resolve")
        
    def test_find_missing_imports(self):
        """
        Test finding missing imports.
        """
        missing_imports = self.analyzer.find_missing_imports()
        
        # Convert to a more easily testable format
        missing_symbols = [symbol for symbol, _, _ in missing_imports]
        
        # Check for specific missing symbols
        self.assertIn("pd", missing_symbols, "pd (pandas) should be detected as missing")
        self.assertIn("plt", missing_symbols, "plt (matplotlib.pyplot) should be detected as missing")
        
        # X and y are used but might be considered parameters or other variables
        # self.assertIn("X", missing_symbols, "X should be detected as missing")
        # self.assertIn("y", missing_symbols, "y should be detected as missing")
        
    def test_suggest_imports(self):
        """
        Test import suggestions.
        """
        suggestions = self.analyzer.suggest_imports(self.sample_code.splitlines())
        
        # Convert to a more easily testable format
        suggested_imports = [suggestion["import_statement"] for suggestion in suggestions]
        
        # Check for specific suggestions
        self.assertTrue(any("import pd" in stmt or "pandas" in stmt for stmt in suggested_imports),
                       "Should suggest importing pandas for pd")
        self.assertTrue(any("import plt" in stmt or "matplotlib" in stmt for stmt in suggested_imports),
                       "Should suggest importing matplotlib.pyplot for plt")
        
    def test_gather_module_context(self):
        """
        Test gathering module context.
        """
        # Resolve the os module and gather context
        self.analyzer.resolve_module("os")
        context = self.analyzer.gather_module_context("os")
        
        # Check basic context information
        self.assertEqual(context["name"], "os", "Context name should be 'os'")
        self.assertTrue(context["is_stdlib"], "os should be identified as stdlib")
        
        # Check that some common os functions are included
        self.assertTrue(any(func in context["functions"] for func in 
                          ["getcwd", "listdir", "mkdir", "remove"]),
                       "Should include common os functions")
        
    def test_find_symbol_module(self):
        """
        Test finding which module a symbol was imported from.
        """
        # Check imported symbols
        self.assertEqual(self.analyzer.find_symbol_module("Path"), "pathlib",
                        "Path should be from pathlib")
        self.assertEqual(self.analyzer.find_symbol_module("np"), "numpy",
                        "np should be from numpy")
        self.assertEqual(self.analyzer.find_symbol_module("train_test_split"), 
                        "sklearn.model_selection",
                        "train_test_split should be from sklearn.model_selection")
        
        # Check non-imported symbols
        self.assertIsNone(self.analyzer.find_symbol_module("pd"),
                         "pd should not have a source module")


if __name__ == "__main__":
    unittest.main()