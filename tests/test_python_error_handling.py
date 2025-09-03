"""
Test cases for Python error handling in the Homeostasis framework.

This module contains comprehensive test cases for Python error detection, analysis,
and patch generation including syntax errors, runtime errors, framework-specific errors,
async/await errors, type hint errors, and performance/security issues.
"""
import pytest
from unittest.mock import patch
import sys
import os

# Add the modules directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from modules.analysis.comprehensive_error_detector import PythonParser, ComprehensiveErrorDetector
from modules.analysis.language_parsers import CompilerIntegration
from modules.analysis.cross_language_orchestrator import CrossLanguageOrchestrator


class TestPythonParser:
    """Test cases for Python parser in comprehensive error detector."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = PythonParser()
    
    def test_parse_syntax_error_invalid_syntax(self):
        """Test parsing of invalid syntax errors."""
        error_string = """
  File "test.py", line 5
    if x = 5:
         ^
SyntaxError: invalid syntax
        """
        
        result = self.parser.parse(error_string)
        
        assert result is not None
        assert result["error_type"] == "SyntaxError"
        assert result["error_message"] == "invalid syntax"
        assert result["file"] == "test.py"
        assert result["line"] == 5
        assert result["category"] == "SYNTAX"
        assert "Use '==' for comparison" in result["suggestion"]
    
    def test_parse_indentation_error(self):
        """Test parsing of indentation errors."""
        error_string = """
  File "test.py", line 3
    print("hello")
    ^
IndentationError: expected an indented block
        """
        
        result = self.parser.parse(error_string)
        
        assert result is not None
        assert result["error_type"] == "IndentationError"
        assert result["error_message"] == "expected an indented block"
        assert result["file"] == "test.py"
        assert result["line"] == 3
        assert result["category"] == "SYNTAX"
        assert "proper indentation" in result["suggestion"].lower()
    
    def test_parse_tab_error(self):
        """Test parsing of tab/space mixing errors."""
        error_string = """
  File "test.py", line 8
    return result
                 ^
TabError: inconsistent use of tabs and spaces in indentation
        """
        
        result = self.parser.parse(error_string)
        
        assert result is not None
        assert result["error_type"] == "TabError"
        assert result["category"] == "SYNTAX"
        assert "Convert all indentation" in result["suggestion"]
    
    def test_parse_name_error(self):
        """Test parsing of name errors."""
        error_string = """
Traceback (most recent call last):
  File "test.py", line 10, in <module>
    print(undefined_variable)
NameError: name 'undefined_variable' is not defined
        """
        
        result = self.parser.parse(error_string)
        
        assert result is not None
        assert result["error_type"] == "NameError"
        assert result["error_message"] == "name 'undefined_variable' is not defined"
        assert result["category"] == "LOGIC"
        assert "Define 'undefined_variable'" in result["suggestion"]
    
    def test_parse_attribute_error(self):
        """Test parsing of attribute errors."""
        error_string = """
Traceback (most recent call last):
  File "test.py", line 15, in process
    result = obj.nonexistent_method()
AttributeError: 'MyClass' object has no attribute 'nonexistent_method'
        """
        
        result = self.parser.parse(error_string)
        
        assert result is not None
        assert result["error_type"] == "AttributeError"
        assert result["category"] == "LOGIC"
        assert "Check available attributes" in result["suggestion"]
    
    def test_parse_key_error(self):
        """Test parsing of key errors."""
        error_string = """
Traceback (most recent call last):
  File "test.py", line 20, in get_value
    return data['missing_key']
KeyError: 'missing_key'
        """
        
        result = self.parser.parse(error_string)
        
        assert result is not None
        assert result["error_type"] == "KeyError"
        assert result["error_message"] == "'missing_key'"
        assert result["category"] == "LOGIC"
        assert "Check if key exists" in result["suggestion"]
    
    def test_parse_index_error(self):
        """Test parsing of index errors."""
        error_string = """
Traceback (most recent call last):
  File "test.py", line 25, in get_item
    return items[10]
IndexError: list index out of range
        """
        
        result = self.parser.parse(error_string)
        
        assert result is not None
        assert result["error_type"] == "IndexError"
        assert result["category"] == "LOGIC"
        assert "Check list length" in result["suggestion"]
    
    def test_parse_type_error_not_callable(self):
        """Test parsing of type errors for non-callable objects."""
        error_string = """
Traceback (most recent call last):
  File "test.py", line 30, in call_func
    result = my_list()
TypeError: 'list' object is not callable
        """
        
        result = self.parser.parse(error_string)
        
        assert result is not None
        assert result["error_type"] == "TypeError"
        assert result["category"] == "LOGIC"
        assert "Remove parentheses" in result["suggestion"]
    
    def test_parse_value_error(self):
        """Test parsing of value errors."""
        error_string = """
Traceback (most recent call last):
  File "test.py", line 35, in convert
    num = int('abc')
ValueError: invalid literal for int() with base 10: 'abc'
        """
        
        result = self.parser.parse(error_string)
        
        assert result is not None
        assert result["error_type"] == "ValueError"
        assert result["category"] == "LOGIC"
        assert "Validate input" in result["suggestion"]
    
    def test_parse_zero_division_error(self):
        """Test parsing of zero division errors."""
        error_string = """
Traceback (most recent call last):
  File "test.py", line 40, in divide
    result = 10 / 0
ZeroDivisionError: division by zero
        """
        
        result = self.parser.parse(error_string)
        
        assert result is not None
        assert result["error_type"] == "ZeroDivisionError"
        assert result["category"] == "LOGIC"
        assert "Check for zero" in result["suggestion"]
    
    def test_parse_import_error(self):
        """Test parsing of import errors."""
        error_string = """
Traceback (most recent call last):
  File "test.py", line 1, in <module>
    import nonexistent_module
ImportError: No module named 'nonexistent_module'
        """
        
        result = self.parser.parse(error_string)
        
        assert result is not None
        assert result["error_type"] == "ImportError"
        assert result["category"] == "DEPENDENCY"
        assert "Install the module" in result["suggestion"]
    
    def test_parse_module_not_found_error(self):
        """Test parsing of module not found errors."""
        error_string = """
Traceback (most recent call last):
  File "test.py", line 1, in <module>
    from package import missing_module
ModuleNotFoundError: No module named 'package.missing_module'
        """
        
        result = self.parser.parse(error_string)
        
        assert result is not None
        assert result["error_type"] == "ModuleNotFoundError"
        assert result["category"] == "DEPENDENCY"
    
    def test_parse_file_not_found_error(self):
        """Test parsing of file not found errors."""
        error_string = """
Traceback (most recent call last):
  File "test.py", line 45, in read_file
    with open('missing.txt', 'r') as f:
FileNotFoundError: [Errno 2] No such file or directory: 'missing.txt'
        """
        
        result = self.parser.parse(error_string)
        
        assert result is not None
        assert result["error_type"] == "FileNotFoundError"
        assert result["category"] == "FILESYSTEM"
        assert "Check if file exists" in result["suggestion"]
    
    def test_parse_recursion_error(self):
        """Test parsing of recursion errors."""
        error_string = """
Traceback (most recent call last):
  File "test.py", line 50, in recursive_func
    return recursive_func(n-1)
  [Previous line repeated 996 more times]
RecursionError: maximum recursion depth exceeded
        """
        
        result = self.parser.parse(error_string)
        
        assert result is not None
        assert result["error_type"] == "RecursionError"
        assert result["category"] == "LOGIC"
        assert "base case" in result["suggestion"].lower()
    
    def test_parse_memory_error(self):
        """Test parsing of memory errors."""
        error_string = """
Traceback (most recent call last):
  File "test.py", line 55, in allocate_memory
    large_list = [0] * (10**10)
MemoryError
        """
        
        result = self.parser.parse(error_string)
        
        assert result is not None
        assert result["error_type"] == "MemoryError"
        assert result["category"] == "RESOURCES"
    
    def test_parse_overflow_error(self):
        """Test parsing of overflow errors."""
        error_string = """
Traceback (most recent call last):
  File "test.py", line 60, in calculate
    result = 10 ** 10000000
OverflowError: (34, 'Numerical result out of range')
        """
        
        result = self.parser.parse(error_string)
        
        assert result is not None
        assert result["error_type"] == "OverflowError"
        assert result["category"] == "LOGIC"
    
    def test_parse_unmatched_parentheses(self):
        """Test parsing of unmatched parentheses syntax error."""
        error_string = """
  File "test.py", line 65
    print("Hello"
                 ^
SyntaxError: unexpected EOF while parsing
        """
        
        result = self.parser.parse(error_string)
        
        assert result is not None
        assert result["error_type"] == "SyntaxError"
        assert result["error_message"] == "unexpected EOF while parsing"
        assert "unmatched parentheses" in result["suggestion"].lower()
    
    def test_parse_syntax_with_ast_validation(self):
        """Test syntax validation using AST when source code is available."""
        error_string = """
  File "test.py", line 70
    def func(
             ^
SyntaxError: unexpected EOF while parsing
        """
        
        # Mock source code context
        source_code = """
def func(
    # Missing closing parenthesis
"""
        
        result = self.parser.parse(error_string, context={"source_code": source_code})
        
        assert result is not None
        assert result["error_type"] == "SyntaxError"
        assert result["ast_validation_performed"] is True
    
    def test_parse_unknown_error_format(self):
        """Test parsing of unknown error format returns None."""
        error_string = "Some random text that is not an error"
        
        result = self.parser.parse(error_string)
        
        assert result is None


class TestPythonErrorDetection:
    """Test cases for Python error detection in comprehensive error detector."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = ComprehensiveErrorDetector()
    
    @patch('modules.analysis.comprehensive_error_detector.ComprehensiveErrorDetector._load_rules')
    def test_detect_python_syntax_error(self, mock_load_rules):
        """Test detection of Python syntax errors."""
        mock_load_rules.return_value = {}
        
        error_data = {
            "error": """
  File "test.py", line 5
    if x = 5:
         ^
SyntaxError: invalid syntax
            """,
            "language": "python"
        }
        
        result = self.detector.detect_error(error_data)
        
        assert result is not None
        assert result["language"] == "python"
        assert result["error_type"] == "SyntaxError"
        assert result["category"] == "SYNTAX"
    
    @patch('modules.analysis.comprehensive_error_detector.ComprehensiveErrorDetector._load_rules')
    def test_detect_python_runtime_error(self, mock_load_rules):
        """Test detection of Python runtime errors."""
        mock_load_rules.return_value = {}
        
        error_data = {
            "error": """
Traceback (most recent call last):
  File "test.py", line 10, in <module>
    print(undefined_var)
NameError: name 'undefined_var' is not defined
            """,
            "language": "python"
        }
        
        result = self.detector.detect_error(error_data)
        
        assert result is not None
        assert result["language"] == "python"
        assert result["error_type"] == "NameError"
        assert result["category"] == "LOGIC"


class TestPythonFrameworkErrors:
    """Test cases for Python framework-specific error handling."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = PythonParser()
    
    def test_parse_django_error(self):
        """Test parsing of Django-specific errors."""
        error_string = """
Traceback (most recent call last):
  File "/app/views.py", line 25, in get_user
    user = User.objects.get(pk=user_id)
  File "/django/db/models/manager.py", line 85, in manager_method
    return getattr(self.get_queryset(), name)(*args, **kwargs)
django.core.exceptions.ObjectDoesNotExist: User matching query does not exist.
        """
        
        # This would need Django-specific rules loaded
        result = self.parser.parse(error_string)
        
        # Basic parsing should still work
        assert result is not None
    
    def test_parse_flask_error(self):
        """Test parsing of Flask-specific errors."""
        error_string = """
Traceback (most recent call last):
  File "/app/app.py", line 30, in index
    return render_template('missing.html')
  File "/flask/templating.py", line 138, in render_template
    ctx.app.jinja_env.get_or_select_template(template_name_or_list)
jinja2.exceptions.TemplateNotFound: missing.html
        """
        
        result = self.parser.parse(error_string)
        
        assert result is not None
    
    def test_parse_asyncio_error(self):
        """Test parsing of asyncio-specific errors."""
        error_string = """
Traceback (most recent call last):
  File "async_app.py", line 15, in main
    await asyncio.gather(*tasks)
RuntimeError: This event loop is already running
        """
        
        result = self.parser.parse(error_string)
        
        assert result is not None
        assert result["error_type"] == "RuntimeError"
        assert "event loop" in result["error_message"]


class TestPythonTypeHintErrors:
    """Test cases for Python type hint related errors."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = PythonParser()
    
    def test_parse_type_hint_error(self):
        """Test parsing of type hint related errors."""
        error_string = """
Traceback (most recent call last):
  File "typed_app.py", line 20, in process
    result: List[str] = process_data(123)
TypeError: 'int' object is not iterable
        """
        
        result = self.parser.parse(error_string)
        
        assert result is not None
        assert result["error_type"] == "TypeError"


class TestPythonPerformanceAndSecurity:
    """Test cases for Python performance and security error patterns."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = PythonParser()
    
    def test_parse_performance_warning(self):
        """Test parsing of performance-related issues."""
        # Performance issues typically come as warnings or custom exceptions
        error_string = """
Warning: Detected inefficient list comprehension in nested loop at line 50
Performance impact: O(n²) complexity detected
        """
        
        # This would need custom performance rules
        result = self.parser.parse(error_string)
        
        # May return None if not matching standard error patterns
        # Real implementation would have performance-specific patterns
    
    def test_parse_security_issue(self):
        """Test parsing of security-related issues."""
        error_string = """
SecurityWarning: Detected SQL injection vulnerability
  File "db_handler.py", line 35, in query_user
    cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")
        """
        
        # This would need custom security rules
        result = self.parser.parse(error_string)


class TestCompilerIntegration:
    """Test cases for Python compiler integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.compiler = CompilerIntegration()
    
    def test_analyze_python_syntax(self):
        """Test Python syntax analysis using AST."""
        code = '''
def hello():
    print("Hello, World!")
'''
        
        result = self.compiler.analyze_python_code(code, "test.py")
        
        assert result["success"] is True
        assert len(result["errors"]) == 0
    
    def test_analyze_python_syntax_error(self):
        """Test Python syntax error detection using AST."""
        code = '''
def hello(:  # Missing parameter name
    print("Hello, World!")
'''
        
        result = self.compiler.analyze_python_code(code, "test.py")
        
        assert result["success"] is False
        assert len(result["errors"]) > 0
        assert result["errors"][0]["type"] == "SyntaxError"


class TestCrossLanguageOrchestratorPython:
    """Test cases for Python handling in cross-language orchestrator."""
    
    def setup_method(self, method):
        """Set up test fixtures."""
        self.orchestrator = CrossLanguageOrchestrator()
    
    def test_python_language_detection(self):
        """Test Python language detection in orchestrator."""
        # Test explicit language
        error_data = {"language": "python", "error": "SyntaxError"}
        language = self.orchestrator._detect_language(error_data)
        assert language == "python"
        
        # Test file extension
        error_data = {"file": "test.py", "error": "NameError"}
        language = self.orchestrator._detect_language(error_data)
        assert language == "python"
        
        # Test error pattern
        error_data = {
            "error": "NameError: name 'x' is not defined",
            "stack_trace": ["  File 'test.py', line 10"]
        }
        language = self.orchestrator._detect_language(error_data)
        assert language == "python"


class TestPythonEdgeCases:
    """Test cases for Python edge cases and corner cases."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = PythonParser()
    
    def test_parse_multiline_error_message(self):
        """Test parsing of errors with multiline messages."""
        error_string = """
Traceback (most recent call last):
  File "test.py", line 100, in complex_func
    validate_data(data)
ValueError: Invalid data format:
  - Missing required field 'name'
  - Field 'age' must be positive
  - Field 'email' is not a valid email address
        """
        
        result = self.parser.parse(error_string)
        
        assert result is not None
        assert result["error_type"] == "ValueError"
        assert "Invalid data format" in result["error_message"]
    
    def test_parse_chained_exceptions(self):
        """Test parsing of chained exceptions (Python 3+)."""
        error_string = """
Traceback (most recent call last):
  File "test.py", line 50, in process_data
    result = json.loads(data)
ValueError: Expecting property name enclosed in double quotes

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "test.py", line 60, in main
    processed = process_data(raw_input)
DataProcessingError: Failed to process input data
        """
        
        # Parser should handle the most recent exception
        result = self.parser.parse(error_string)
        
        assert result is not None
    
    def test_parse_unicode_in_error(self):
        """Test parsing of errors containing unicode characters."""
        error_string = """
Traceback (most recent call last):
  File "test.py", line 70, in print_message
    print(message)
UnicodeEncodeError: 'ascii' codec can't encode character '\\u2019' in position 10
        """
        
        result = self.parser.parse(error_string)
        
        assert result is not None
        assert result["error_type"] == "UnicodeEncodeError"
    
    def test_parse_very_long_stack_trace(self):
        """Test parsing of errors with very long stack traces."""
        stack_frames = []
        for i in range(50):
            stack_frames.append(f"  File 'module{i}.py', line {i*10}, in func{i}")
            stack_frames.append(f"    do_something_{i}()")
        
        error_string = f"""
Traceback (most recent call last):
{chr(10).join(stack_frames)}
RuntimeError: Maximum call stack exceeded
        """
        
        result = self.parser.parse(error_string)
        
        assert result is not None
        assert result["error_type"] == "RuntimeError"
    
    def test_parse_error_with_no_file_info(self):
        """Test parsing of errors without file information."""
        error_string = """
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'x' is not defined
        """
        
        result = self.parser.parse(error_string)
        
        assert result is not None
        assert result["error_type"] == "NameError"
        assert result["file"] == "<stdin>"


class TestPythonAsyncErrors:
    """Test cases for Python async/await specific errors."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = PythonParser()
    
    def test_parse_async_syntax_error(self):
        """Test parsing of async syntax errors."""
        error_string = """
  File "async_app.py", line 25
    result = await fetch_data()
             ^
SyntaxError: 'await' outside async function
        """
        
        result = self.parser.parse(error_string)
        
        assert result is not None
        assert result["error_type"] == "SyntaxError"
        assert "'await' outside async function" in result["error_message"]
    
    def test_parse_asyncio_timeout_error(self):
        """Test parsing of asyncio timeout errors."""
        error_string = """
Traceback (most recent call last):
  File "async_app.py", line 30, in fetch_with_timeout
    result = await asyncio.wait_for(fetch_data(), timeout=5.0)
asyncio.exceptions.TimeoutError
        """
        
        result = self.parser.parse(error_string)
        
        assert result is not None
        assert "TimeoutError" in result["error_type"]


class TestPythonPatchSuggestions:
    """Test cases for Python error patch suggestions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = PythonParser()
    
    def test_suggestion_for_assignment_in_condition(self):
        """Test suggestion for assignment in condition."""
        error_string = """
  File "test.py", line 5
    if x = 5:
         ^
SyntaxError: invalid syntax
        """
        
        result = self.parser.parse(error_string)
        
        assert result is not None
        assert "Use '==' for comparison" in result["suggestion"]
    
    def test_suggestion_for_missing_colon(self):
        """Test suggestion for missing colon."""
        error_string = """
  File "test.py", line 10
    if x == 5
            ^
SyntaxError: invalid syntax
        """
        
        result = self.parser.parse(error_string)
        
        assert result is not None
        # Parser should suggest adding a colon
    
    def test_suggestion_for_undefined_variable(self):
        """Test suggestion for undefined variable."""
        error_string = """
Traceback (most recent call last):
  File "test.py", line 15, in <module>
    print(undeclared_var)
NameError: name 'undeclared_var' is not defined
        """
        
        result = self.parser.parse(error_string)
        
        assert result is not None
        assert "Define 'undeclared_var'" in result["suggestion"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])