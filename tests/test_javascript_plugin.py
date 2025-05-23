"""
Test cases for JavaScript language plugin.

This module contains comprehensive test cases for the JavaScript plugin,
including error analysis, dependency handling, transpilation errors, and fix generation.
"""
import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from modules.analysis.plugins.javascript_plugin import (
    JavaScriptLanguagePlugin,
    JavaScriptExceptionHandler,
    JavaScriptPatchGenerator
)
from modules.analysis.plugins.javascript_dependency_analyzer import JavaScriptDependencyAnalyzer


class TestJavaScriptLanguagePlugin:
    """Test cases for the main JavaScript language plugin."""
    
    @pytest.fixture
    def plugin(self):
        """Create a JavaScript plugin instance for testing."""
        return JavaScriptLanguagePlugin()
    
    def test_plugin_initialization(self, plugin):
        """Test plugin initialization."""
        assert plugin.get_language_id() == "javascript"
        assert plugin.get_language_name() == "JavaScript"
        assert plugin.get_language_version() == "ES5+"
        assert "express" in plugin.get_supported_frameworks()
        assert "react" in plugin.get_supported_frameworks()
    
    def test_can_handle_javascript_errors(self, plugin):
        """Test plugin can handle JavaScript errors."""
        # Test with explicit language
        error_data = {"language": "javascript", "error_type": "TypeError"}
        assert plugin.can_handle(error_data) is True
        
        # Test with JavaScript error types
        js_errors = [
            {"error_type": "TypeError", "message": "Cannot read property 'x' of undefined"},
            {"error_type": "ReferenceError", "message": "x is not defined"},
            {"error_type": "SyntaxError", "message": "Unexpected token }"}
        ]
        
        for error in js_errors:
            assert plugin.can_handle(error) is True
    
    def test_can_handle_nodejs_patterns(self, plugin):
        """Test plugin can handle Node.js specific patterns."""
        nodejs_error = {
            "error_type": "Error",
            "message": "Cannot find module 'express'",
            "stack_trace": ["at require (internal/modules/cjs/loader.js:999:30)"]
        }
        assert plugin.can_handle(nodejs_error) is True
    
    def test_can_handle_browser_patterns(self, plugin):
        """Test plugin can handle browser-specific patterns."""
        browser_error = {
            "error_type": "TypeError",
            "message": "Cannot read property 'onclick' of null",
            "stack_trace": ["at HTMLElement.onclick (app.js:15:3)"]
        }
        assert plugin.can_handle(browser_error) is True
    
    def test_normalize_error(self, plugin):
        """Test error normalization."""
        js_error = {
            "name": "TypeError",
            "message": "Cannot read property 'id' of undefined",
            "stack": "TypeError: Cannot read property 'id' of undefined\\n    at app.js:10:5"
        }
        
        normalized = plugin.normalize_error(js_error)
        
        assert normalized["language"] == "javascript"
        assert normalized["error_type"] == "TypeError"
        assert normalized["message"] == "Cannot read property 'id' of undefined"
        assert "error_id" in normalized
        assert "timestamp" in normalized
    
    def test_denormalize_error(self, plugin):
        """Test error denormalization."""
        standard_error = {
            "language": "javascript",
            "error_type": "TypeError",
            "message": "Cannot read property 'id' of undefined",
            "stack_trace": ["TypeError: Cannot read property 'id' of undefined", "    at app.js:10:5"]
        }
        
        js_error = plugin.denormalize_error(standard_error)
        
        assert js_error["name"] == "TypeError"
        assert js_error["message"] == "Cannot read property 'id' of undefined"
        assert "stack" in js_error


class TestJavaScriptExceptionHandler:
    """Test cases for JavaScript exception handling."""
    
    @pytest.fixture
    def handler(self):
        """Create a JavaScript exception handler for testing."""
        return JavaScriptExceptionHandler()
    
    def test_analyze_type_error(self, handler):
        """Test analysis of TypeError."""
        error_data = {
            "error_type": "TypeError",
            "message": "Cannot read property 'id' of undefined",
            "stack_trace": ["at app.js:10:5"]
        }
        
        analysis = handler.analyze_exception(error_data)
        
        assert analysis["category"] == "javascript"
        assert analysis["confidence"] == "high"
        assert "property access" in analysis["suggested_fix"].lower()
        assert analysis["root_cause"] == "js_property_access_on_undefined"
    
    def test_analyze_reference_error(self, handler):
        """Test analysis of ReferenceError."""
        error_data = {
            "error_type": "ReferenceError", 
            "message": "someVar is not defined",
            "stack_trace": ["at app.js:15:3"]
        }
        
        analysis = handler.analyze_exception(error_data)
        
        assert analysis["category"] == "javascript"
        assert analysis["confidence"] == "high"
        assert "not defined" in analysis["suggested_fix"].lower()
        assert analysis["root_cause"] == "js_undefined_reference"
    
    def test_analyze_promise_rejection(self, handler):
        """Test analysis of Promise rejection."""
        error_data = {
            "error_type": "UnhandledPromiseRejection",
            "message": "Uncaught (in promise) TypeError: Cannot read property 'data' of undefined",
            "stack_trace": ["at Promise.then (app.js:25:8)"]
        }
        
        analysis = handler.analyze_exception(error_data)
        
        assert analysis["category"] == "javascript"
        assert analysis["confidence"] == "high"
        assert "catch" in analysis["suggested_fix"].lower()
        assert analysis["root_cause"] == "js_unhandled_promise_rejection"
    
    def test_analyze_transpilation_error(self, handler):
        """Test analysis of transpilation errors."""
        # Babel error
        babel_error = {
            "error_type": "SyntaxError",
            "message": "Babel: Unexpected token (15:8)",
            "stack_trace": ["at babel-core/lib/transformation/file.js:558:10"]
        }
        
        analysis = handler.analyze_transpilation_error(babel_error)
        
        assert analysis["category"] == "transpilation"
        assert analysis["subcategory"] == "babel_syntax"
        assert "babel" in analysis["tags"]
        
        # TypeScript error
        ts_error = {
            "error_type": "TS2304",
            "message": "TS2304: Cannot find name 'someVariable'",
            "stack_trace": ["at typescript compiler"]
        }
        
        analysis = handler.analyze_transpilation_error(ts_error)
        
        assert analysis["category"] == "transpilation"
        assert analysis["subcategory"] == "typescript_type"
        assert analysis["error_code"] == "TS2304"
    
    def test_generic_analysis(self, handler):
        """Test generic error analysis fallback."""
        unknown_error = {
            "error_type": "CustomError",
            "message": "Something went wrong",
            "stack_trace": []
        }
        
        analysis = handler.analyze_exception(unknown_error)
        
        assert analysis["category"] == "javascript"
        assert analysis["subcategory"] == "unknown"
        assert analysis["confidence"] == "low"


class TestJavaScriptPatchGenerator:
    """Test cases for JavaScript patch generation."""
    
    @pytest.fixture
    def generator(self):
        """Create a JavaScript patch generator for testing."""
        return JavaScriptPatchGenerator()
    
    def test_fix_property_access(self, generator):
        """Test fix generation for property access errors."""
        error_data = {
            "error_type": "TypeError",
            "message": "Cannot read property 'id' of undefined",
            "stack_trace": [{"function": "main", "file": "app.js", "line": 10, "column": 5}]
        }
        
        analysis = {
            "root_cause": "js_property_access_on_undefined",
            "confidence": "high"
        }
        
        source_code = \"\"\"function getUser() {
    const user = getDataFromAPI();
    return user.id;  // Error line
}\"\"\"\n        \n        patch = generator.generate_patch(error_data, analysis, source_code)\n        \n        assert patch is not None\n        assert patch[\"type\"] == \"line_replacement\"\n        assert \"?.\" in patch[\"replacement\"] or \"&&\" in patch[\"replacement\"]\n    \n    def test_fix_not_a_function(self, generator):\n        \"\"\"Test fix generation for 'not a function' errors.\"\"\"\n        error_data = {\n            \"error_type\": \"TypeError\",\n            \"message\": \"someFunction is not a function\",\n            \"stack_trace\": [{\"function\": \"main\", \"file\": \"app.js\", \"line\": 15, \"column\": 3}]\n        }\n        \n        analysis = {\n            \"root_cause\": \"js_not_a_function\",\n            \"confidence\": \"high\"\n        }\n        \n        source_code = \"\"\"function main() {\n    const result = someFunction();\n    return result;\n}\"\"\"\n        \n        patch = generator.generate_patch(error_data, analysis, source_code)\n        \n        assert patch is not None\n        assert patch[\"type\"] == \"line_replacement\"\n        assert \"typeof\" in patch[\"replacement\"]\n    \n    def test_fix_undefined_reference(self, generator):\n        \"\"\"Test fix generation for undefined reference errors.\"\"\"\n        error_data = {\n            \"error_type\": \"ReferenceError\",\n            \"message\": \"require is not defined\",\n            \"stack_trace\": []\n        }\n        \n        analysis = {\n            \"root_cause\": \"js_undefined_reference\",\n            \"confidence\": \"high\"\n        }\n        \n        source_code = \"\"\n        \n        patch = generator.generate_patch(error_data, analysis, source_code)\n        \n        assert patch is not None\n        assert patch[\"type\"] == \"suggestion\"\n        assert \"Node.js environment\" in patch[\"description\"]\n    \n    def test_fix_promise_rejection(self, generator):\n        \"\"\"Test fix generation for Promise rejection errors.\"\"\"\n        error_data = {\n            \"error_type\": \"UnhandledPromiseRejection\",\n            \"message\": \"Uncaught (in promise) Error: API failed\",\n            \"stack_trace\": [{\"function\": \"fetchData\", \"file\": \"app.js\", \"line\": 20, \"column\": 8}]\n        }\n        \n        analysis = {\n            \"root_cause\": \"js_unhandled_promise_rejection\",\n            \"confidence\": \"high\"\n        }\n        \n        source_code = \"\"\"async function fetchData() {\n    const response = fetch('/api/data').then(r => r.json());\n    return response;\n}\"\"\"\n        \n        patch = generator.generate_patch(error_data, analysis, source_code)\n        \n        assert patch is not None\n        assert \".catch\" in patch[\"replacement\"] or patch[\"type\"] == \"suggestion\"\n\n\nclass TestJavaScriptDependencyAnalyzer:\n    \"\"\"Test cases for JavaScript dependency analysis.\"\"\"\n    \n    @pytest.fixture\n    def analyzer(self):\n        \"\"\"Create a dependency analyzer for testing.\"\"\"\n        return JavaScriptDependencyAnalyzer()\n    \n    def test_analyze_module_not_found(self, analyzer):\n        \"\"\"Test analysis of module not found errors.\"\"\"\n        error_data = {\n            \"error_type\": \"Error\",\n            \"message\": \"Cannot find module 'express'\"\n        }\n        \n        analysis = analyzer.analyze_dependency_error(error_data, \"/fake/project\")\n        \n        assert analysis[\"category\"] == \"dependency\"\n        assert analysis[\"subcategory\"] == \"missing_package\"\n        assert analysis[\"package\"] == \"express\"\n        assert \"npm install express\" in analysis[\"fix_commands\"]\n    \n    def test_analyze_relative_import_error(self, analyzer):\n        \"\"\"Test analysis of relative import errors.\"\"\"\n        error_data = {\n            \"error_type\": \"Error\",\n            \"message\": \"Cannot find module './utils/helper'\"\n        }\n        \n        analysis = analyzer.analyze_dependency_error(error_data, \"/fake/project\")\n        \n        assert analysis[\"category\"] == \"dependency\"\n        assert analysis[\"subcategory\"] == \"relative_import\"\n        assert analysis[\"module\"] == \"./utils/helper\"\n    \n    def test_analyze_project_dependencies(self, analyzer):\n        \"\"\"Test project dependency analysis.\"\"\"\n        # Create a temporary package.json\n        with tempfile.TemporaryDirectory() as temp_dir:\n            temp_path = Path(temp_dir)\n            package_json = {\n                \"name\": \"test-project\",\n                \"version\": \"1.0.0\",\n                \"dependencies\": {\n                    \"express\": \"^4.18.0\",\n                    \"react\": \"*\"  # Wildcard version\n                },\n                \"devDependencies\": {\n                    \"webpack\": \"5.0.0-beta.1\"  # Pre-release version\n                }\n            }\n            \n            with open(temp_path / \"package.json\", \"w\") as f:\n                json.dump(package_json, f)\n            \n            analysis = analyzer.analyze_project_dependencies(str(temp_path))\n            \n            assert analysis[\"project_name\"] == \"test-project\"\n            assert analysis[\"dependencies\"][\"count\"] == 2\n            assert analysis[\"dev_dependencies\"][\"count\"] == 1\n            \n            # Check for issues\n            dep_issues = analysis[\"dependencies\"][\"issues\"]\n            assert any(issue[\"type\"] == \"wildcard_version\" for issue in dep_issues)\n            \n            dev_issues = analysis[\"dev_dependencies\"][\"issues\"]\n            assert any(issue[\"type\"] == \"prerelease_version\" for issue in dev_issues)\n\n\nclass TestIntegration:\n    \"\"\"Integration tests for the JavaScript plugin.\"\"\"\n    \n    @pytest.fixture\n    def plugin(self):\n        \"\"\"Create a JavaScript plugin instance for testing.\"\"\"\n        return JavaScriptLanguagePlugin()\n    \n    def test_full_error_analysis_flow(self, plugin):\n        \"\"\"Test the complete error analysis flow.\"\"\"\n        # Test a complex JavaScript error\n        error_data = {\n            \"name\": \"TypeError\",\n            \"message\": \"Cannot read property 'map' of undefined\",\n            \"stack\": \"\"\"TypeError: Cannot read property 'map' of undefined\n    at processData (app.js:25:8)\n    at fetch.then (app.js:15:12)\n    at process._tickCallback (internal/process/next_tick.js:68:7)\"\"\"\n        }\n        \n        # Analyze the error\n        analysis = plugin.analyze_error(error_data)\n        \n        assert analysis[\"plugin\"] == \"javascript\"\n        assert analysis[\"language\"] == \"javascript\"\n        assert analysis[\"category\"] == \"javascript\"\n        assert \"confidence\" in analysis\n        assert \"suggested_fix\" in analysis\n    \n    def test_dependency_error_flow(self, plugin):\n        \"\"\"Test dependency error analysis flow.\"\"\"\n        error_data = {\n            \"language\": \"javascript\",\n            \"error_type\": \"Error\",\n            \"message\": \"Cannot find module 'lodash'\",\n            \"context\": {\n                \"project_path\": \"/fake/project\"\n            }\n        }\n        \n        analysis = plugin.analyze_error(error_data)\n        \n        assert analysis[\"category\"] == \"dependency\"\n        assert analysis[\"subcategory\"] == \"missing_package\"\n        assert \"lodash\" in str(analysis)\n    \n    def test_transpilation_error_flow(self, plugin):\n        \"\"\"Test transpilation error analysis flow.\"\"\"\n        error_data = {\n            \"language\": \"javascript\",\n            \"error_type\": \"TS2304\",\n            \"message\": \"TS2304: Cannot find name 'React'\",\n            \"stack_trace\": [\"at typescript compiler\"]\n        }\n        \n        analysis = plugin.analyze_error(error_data)\n        \n        assert analysis[\"category\"] == \"transpilation\"\n        assert analysis[\"subcategory\"] == \"typescript_type\"\n        assert \"typescript\" in analysis[\"tags\"]\n    \n    def test_fix_generation_flow(self, plugin):\n        \"\"\"Test fix generation flow.\"\"\"\n        analysis = {\n            \"root_cause\": \"js_property_access_on_undefined\",\n            \"suggested_fix\": \"Use optional chaining\",\n            \"confidence\": \"high\"\n        }\n        \n        context = {\n            \"error_data\": {\n                \"error_type\": \"TypeError\",\n                \"message\": \"Cannot read property 'id' of undefined\"\n            },\n            \"source_code\": \"const id = user.id;\"\n        }\n        \n        fix = plugin.generate_fix(analysis, context)\n        \n        assert fix is not None\n        assert \"type\" in fix\n        assert \"description\" in fix or \"replacement\" in fix\n\n\nif __name__ == \"__main__\":\n    pytest.main([__file__])