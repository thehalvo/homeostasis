"""
Test cases for JavaScript language plugin.

This module contains comprehensive test cases for the JavaScript plugin,
including error analysis, dependency handling, transpilation errors, and fix generation.
"""
import pytest
import json
import tempfile
from pathlib import Path

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
            "stack_trace": [{"function": "getUser", "file": "app.js", "line": 3, "column": 12}]
        }
        
        analysis = {
            "root_cause": "js_property_access_on_undefined",
            "confidence": "high"
        }
        
        source_code = """function getUser() {
    const user = getDataFromAPI();
    return user.id;  // Error line
}"""
        
        null_patch = generator.generate_patch(error_data, analysis, source_code)
        
        assert null_patch is not None
        assert null_patch["type"] == "line_replacement"
        assert "?." in null_patch["replacement"] or "&&" in null_patch["replacement"]
    
    def test_fix_not_a_function(self, generator):
        """Test fix generation for 'not a function' errors."""
        error_data = {
            "error_type": "TypeError",
            "message": "someFunction is not a function",
            "stack_trace": [{"function": "main", "file": "app.js", "line": 2, "column": 19}]
        }
        
        analysis = {
            "root_cause": "js_not_a_function",
            "confidence": "high"
        }
        
        source_code = """function main() {
    const result = someFunction();
    return result;
}"""
        
        function_patch = generator.generate_patch(error_data, analysis, source_code)
        
        assert function_patch is not None
        assert function_patch["type"] == "line_replacement"
        assert "typeof" in function_patch["replacement"]
    
    def test_fix_undefined_reference(self, generator):
        """Test fix generation for undefined reference errors."""
        error_data = {
            "error_type": "ReferenceError",
            "message": "require is not defined",
            "stack_trace": []
        }
        
        analysis = {
            "root_cause": "js_undefined_reference",
            "confidence": "high"
        }
        
        source_code = ""
        
        undefined_patch = generator.generate_patch(error_data, analysis, source_code)
        
        assert undefined_patch is not None
        assert undefined_patch["type"] == "suggestion"
        assert "Node.js environment" in undefined_patch["description"]
    
    def test_fix_promise_rejection(self, generator):
        """Test fix generation for Promise rejection errors."""
        error_data = {
            "error_type": "UnhandledPromiseRejection",
            "message": "Uncaught (in promise) Error: API failed",
            "stack_trace": [{"function": "fetchData", "file": "app.js", "line": 2, "column": 20}]
        }
        
        analysis = {
            "root_cause": "js_unhandled_promise_rejection",
            "confidence": "high"
        }
        
        source_code = """async function fetchData() {
    const response = fetch('/api/data').then(r => r.json());
    return response;
}"""
        
        promise_patch = generator.generate_patch(error_data, analysis, source_code)
        
        assert promise_patch is not None
        assert ".catch" in promise_patch["replacement"] or promise_patch["type"] == "suggestion"


class TestJavaScriptDependencyAnalyzer:
    """Test cases for JavaScript dependency analysis."""
    
    @pytest.fixture
    def analyzer(self):
        """Create a dependency analyzer for testing."""
        return JavaScriptDependencyAnalyzer()
    
    def test_analyze_module_not_found(self, analyzer):
        """Test analysis of module not found errors."""
        error_data = {
            "error_type": "Error",
            "message": "Cannot find module 'express'"
        }
        
        analysis = analyzer.analyze_dependency_error(error_data, "/fake/project")
        
        assert analysis["category"] == "dependency"
        assert analysis["subcategory"] == "missing_package"
        assert analysis["package"] == "express"
        assert "npm install express" in analysis["fix_commands"]
    
    def test_analyze_relative_import_error(self, analyzer):
        """Test analysis of relative import errors."""
        error_data = {
            "error_type": "Error",
            "message": "Cannot find module './utils/helper'"
        }
        
        analysis = analyzer.analyze_dependency_error(error_data, "/fake/project")
        
        assert analysis["category"] == "dependency"
        assert analysis["subcategory"] == "relative_import"
        assert analysis["module"] == "./utils/helper"
    
    def test_analyze_project_dependencies(self, analyzer):
        """Test project dependency analysis."""
        # Create a temporary package.json
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            package_json = {
                "name": "test-project",
                "version": "1.0.0",
                "dependencies": {
                    "express": "^4.18.0",
                    "react": "*"  # Wildcard version
                },
                "devDependencies": {
                    "webpack": "5.0.0-beta.1"  # Pre-release version
                }
            }
            
            with open(temp_path / "package.json", "w") as f:
                json.dump(package_json, f)
            
            analysis = analyzer.analyze_project_dependencies(str(temp_path))
            
            assert analysis["project_name"] == "test-project"
            assert analysis["dependencies"]["count"] == 2
            assert analysis["dev_dependencies"]["count"] == 1
            
            # Check for issues
            dep_issues = analysis["dependencies"]["issues"]
            assert any(issue["type"] == "wildcard_version" for issue in dep_issues)
            
            dev_issues = analysis["dev_dependencies"]["issues"]
            assert any(issue["type"] == "prerelease_version" for issue in dev_issues)


class TestIntegration:
    """Integration tests for the JavaScript plugin."""
    
    @pytest.fixture
    def plugin(self):
        """Create a JavaScript plugin instance for testing."""
        return JavaScriptLanguagePlugin()
    
    def test_full_error_analysis_flow(self, plugin):
        """Test the complete error analysis flow."""
        # Test a complex JavaScript error
        error_data = {
            "name": "TypeError",
            "message": "Cannot read property 'map' of undefined",
            "stack": """TypeError: Cannot read property 'map' of undefined
    at processData (app.js:25:8)
    at fetch.then (app.js:15:12)
    at process._tickCallback (internal/process/next_tick.js:68:7)"""
        }
        
        # Analyze the error
        analysis = plugin.analyze_error(error_data)
        
        assert analysis["plugin"] == "javascript"
        assert analysis["language"] == "javascript"
        assert analysis["category"] == "javascript"
        assert "confidence" in analysis
        assert "suggested_fix" in analysis
    
    def test_dependency_error_flow(self, plugin):
        """Test dependency error analysis flow."""
        error_data = {
            "language": "javascript",
            "error_type": "Error",
            "message": "Cannot find module 'lodash'",
            "context": {
                "project_path": "/fake/project"
            }
        }
        
        analysis = plugin.analyze_error(error_data)
        
        assert analysis["category"] == "dependency"
        assert analysis["subcategory"] == "missing_package"
        assert "lodash" in str(analysis)
    
    def test_transpilation_error_flow(self, plugin):
        """Test transpilation error analysis flow."""
        error_data = {
            "language": "javascript",
            "error_type": "TS2304",
            "message": "TS2304: Cannot find name 'React'",
            "stack_trace": ["at typescript compiler"]
        }
        
        analysis = plugin.analyze_error(error_data)
        
        assert analysis["category"] == "transpilation"
        assert analysis["subcategory"] == "typescript_type"
        assert "typescript" in analysis["tags"]
    
    def test_fix_generation_flow(self, plugin):
        """Test fix generation flow."""
        analysis = {
            "root_cause": "js_property_access_on_undefined",
            "suggested_fix": "Use optional chaining",
            "confidence": "high"
        }
        
        context = {
            "error_data": {
                "error_type": "TypeError",
                "message": "Cannot read property 'id' of undefined",
                "stack_trace": [
                    {
                        "file": "test.js",
                        "line": 1,
                        "function": "main"
                    }
                ]
            },
            "source_code": "const id = user.id;"
        }
        
        # Pass source_code as third argument to match expected signature
        fix = plugin.generate_fix(analysis, context, context["source_code"])
        
        assert fix is not None
        assert "type" in fix
        assert "description" in fix or "replacement" in fix


if __name__ == "__main__":
    pytest.main([__file__])