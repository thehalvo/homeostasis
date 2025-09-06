"""
Test case generator for creating tests for specific error types.

This module provides utilities for:
1. Creating test cases based on error patterns
2. Generating regression tests for fixed errors
3. Supporting different test frameworks (pytest, unittest)
"""

import ast
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from modules.analysis.rule_based import RuleBasedAnalyzer
from modules.monitoring.logger import MonitoringLogger


class TestGenerator:
    """
    Generates test cases for specific error types.
    """

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        log_level: str = "INFO",
        test_framework: str = "pytest",
    ):
        """
        Initialize the test generator.

        Args:
            output_dir: Directory to save generated tests
            log_level: Logging level
            test_framework: Test framework to use (pytest or unittest)
        """
        self.logger = MonitoringLogger("test_generator", log_level=log_level)
        self.output_dir = output_dir or (project_root / "tests" / "generated")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.test_framework = test_framework

        # Set up templates directory
        self.templates_dir = Path(__file__).parent / "templates"
        if not self.templates_dir.exists():
            self.templates_dir.mkdir(parents=True)
            self._create_default_templates()

        # Initialize rule analyzer for pattern matching
        self.rule_analyzer = RuleBasedAnalyzer()

        self.logger.info(f"Initialized test generator using {test_framework}")

    def _create_default_templates(self) -> None:
        """Create default test templates."""
        # Create pytest template
        pytest_template = """
# Generated test for {{ error_type }}
import pytest
from {{ module_path }} import {{ function_name }}

def test_{{ test_name }}():
    '''Test for {{ error_description }}'''
    # Setup
    {{ setup_code }}
    
    # Test
    {% if expect_exception %}
    with pytest.raises({{ exception_type }}):
        {{ test_code }}
    {% else %}
    result = {{ test_code }}
    assert {{ assertion }}
    {% endif %}
"""

        # Create unittest template
        unittest_template = """
# Generated test for {{ error_type }}
import unittest
from {{ module_path }} import {{ function_name }}

class Test{{ test_class_name }}(unittest.TestCase):
    def test_{{ test_name }}(self):
        '''Test for {{ error_description }}'''
        # Setup
        {{ setup_code }}
        
        # Test
        {% if expect_exception %}
        with self.assertRaises({{ exception_type }}):
            {{ test_code }}
        {% else %}
        result = {{ test_code }}
        self.assertTrue({{ assertion }})
        {% endif %}

if __name__ == "__main__":
    unittest.main()
"""

        # Save templates
        with open(self.templates_dir / "pytest_template.txt", "w") as f:
            f.write(pytest_template)

        with open(self.templates_dir / "unittest_template.txt", "w") as f:
            f.write(unittest_template)

    def _render_template(self, template_path: Path, context: Dict[str, Any]) -> str:
        """
        Render a template with the given context.

        Args:
            template_path: Path to the template file
            context: Dictionary of variables to use in the template

        Returns:
            Rendered template
        """
        try:
            # Simple template rendering
            with open(template_path, "r") as f:
                template = f.read()

            # Replace simple variables
            for key, value in context.items():
                template = template.replace("{{ " + key + " }}", str(value))

            # Handle if statements
            if_pattern = (
                r"{%\s*if\s+([^%]+)\s*%}(.*?){%\s*else\s*%}(.*?){%\s*endif\s*%}"
            )

            def replace_if(match):
                condition = match.group(1).strip()
                if_true = match.group(2)
                if_false = match.group(3)

                # Evaluate the condition
                try:
                    # Simple condition evaluation
                    result = eval(condition, {"__builtins__": {}}, context)
                    return if_true if result else if_false
                except Exception:
                    self.logger.warning(f"Failed to evaluate condition: {condition}")
                    return if_false

            template = re.sub(if_pattern, replace_if, template, flags=re.DOTALL)

            return template

        except Exception as e:
            self.logger.exception(
                e, message=f"Failed to render template {template_path}"
            )
            return ""

    def _get_test_name(self, function_name: str, error_type: str) -> str:
        """
        Generate a test name based on function name and error type.

        Args:
            function_name: Name of the function being tested
            error_type: Type of error being tested

        Returns:
            Test name
        """
        # Convert error type to a more readable form
        error_type = error_type.lower().replace("_", " ").replace("error", "").strip()
        error_parts = error_type.split()

        # Create a camel case name
        test_name = function_name
        for part in error_parts:
            if part:
                test_name += "_" + part

        return test_name

    def _get_function_details(
        self, file_path: Path, function_name: str
    ) -> Dict[str, Any]:
        """
        Get details about a function for test generation.

        Args:
            file_path: Path to the Python file
            function_name: Name of the function

        Returns:
            Function details
        """
        try:
            with open(file_path, "r") as f:
                code = f.read()

            tree = ast.parse(code)

            # Find the function
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == function_name:
                    # Get function arguments
                    args = []
                    for arg in node.args.args:
                        args.append(arg.arg)

                    # Get default values
                    defaults = []
                    for default in node.args.defaults:
                        if isinstance(default, (ast.Str, ast.Constant)):
                            if hasattr(default, "value"):
                                value = default.value
                            else:
                                value = default.s
                            defaults.append(repr(value))
                        elif isinstance(default, ast.Num):
                            defaults.append(str(default.n))
                        elif isinstance(default, ast.NameConstant):
                            defaults.append(str(default.value))
                        else:
                            defaults.append("None")

                    # Assign defaults to arguments
                    num_args = len(args)
                    num_defaults = len(defaults)
                    arg_defaults = {}

                    if num_defaults > 0:
                        for i in range(num_args - num_defaults, num_args):
                            arg_idx = i - (num_args - num_defaults)
                            if arg_idx < len(defaults):
                                arg_defaults[args[i]] = defaults[arg_idx]

                    # Extract docstring
                    docstring = ast.get_docstring(node) or ""

                    # Get function body
                    body_lines = []
                    for line in code.splitlines()[node.lineno : node.end_lineno]:
                        body_lines.append(line)

                    # Get module path for imports
                    rel_path = file_path.relative_to(project_root)
                    module_path = str(rel_path.with_suffix("")).replace(
                        os.path.sep, "."
                    )

                    return {
                        "name": function_name,
                        "args": args,
                        "defaults": arg_defaults,
                        "docstring": docstring,
                        "body": "\n".join(body_lines),
                        "module_path": module_path,
                    }

            return {
                "name": function_name,
                "args": [],
                "defaults": {},
                "docstring": "",
                "body": "",
                "module_path": "",
            }

        except Exception as e:
            self.logger.exception(
                e,
                message=f"Failed to get function details for {function_name} in {file_path}",
            )
            return {
                "name": function_name,
                "args": [],
                "defaults": {},
                "docstring": "",
                "body": "",
                "module_path": "",
            }

    def _generate_test_code(
        self,
        function_details: Dict[str, Any],
        error_type: str,
        params: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Generate test code for a function.

        Args:
            function_details: Function details
            error_type: Type of error
            params: Optional parameters for the test

        Returns:
            Test code components
        """
        function_name = function_details["name"]
        args = function_details["args"]
        defaults = function_details["defaults"]

        # Default params
        if params is None:
            params = {}

        # Generate setup code
        setup_lines = []

        # Generate test arguments
        test_args = []

        # Handle different error types
        expect_exception = True
        exception_type = "Exception"

        if error_type == "KeyError":
            exception_type = "KeyError"
            # Check if we're testing a dict access
            if "dict_name" in params:
                dict_name = params["dict_name"]
                key_name = params.get("key_name", "'missing_key'")
                setup_lines.append(f"{dict_name} = {{}}")
                test_args.append(f"{dict_name}[{key_name}]")
            else:
                # Default to function call with invalid arguments
                for arg in args:
                    if arg == "self":
                        continue
                    if arg in defaults:
                        test_args.append(f"{arg}={defaults[arg]}")
                    else:
                        test_args.append(f"{arg}={{}}")

        elif error_type == "IndexError":
            exception_type = "IndexError"
            # Check if we're testing a list access
            if "list_name" in params:
                list_name = params["list_name"]
                index = params.get("index", "0")
                setup_lines.append(f"{list_name} = []")
                test_args.append(f"{list_name}[{index}]")
            else:
                # Default to function call with invalid arguments
                for arg in args:
                    if arg == "self":
                        continue
                    if arg in defaults:
                        test_args.append(f"{arg}={defaults[arg]}")
                    else:
                        test_args.append(f"{arg}=[]")

        elif error_type == "TypeError":
            exception_type = "TypeError"
            # Default to function call with invalid argument types
            for arg in args:
                if arg == "self":
                    continue
                if arg in defaults:
                    test_args.append(f"{arg}={defaults[arg]}")
                else:
                    test_args.append(f"{arg}=None")

        elif error_type == "ValueError":
            exception_type = "ValueError"
            # Default to function call with invalid values
            for arg in args:
                if arg == "self":
                    continue
                if arg in defaults:
                    test_args.append(f"{arg}={defaults[arg]}")
                else:
                    test_args.append(f"{arg}='invalid_value'")

        elif error_type == "AttributeError":
            exception_type = "AttributeError"
            # Check if we're testing an attribute access
            if "obj_name" in params:
                obj_name = params["obj_name"]
                attr_name = params.get("attr_name", "missing_attr")
                setup_lines.append(f"{obj_name} = object()")
                test_args.append(f"{obj_name}.{attr_name}")
            else:
                # Default to function call with None for object arguments
                for arg in args:
                    if arg == "self":
                        continue
                    if arg in defaults:
                        test_args.append(f"{arg}={defaults[arg]}")
                    else:
                        test_args.append(f"{arg}=None")

        else:
            # Generic case
            exception_type = params.get("exception_type", "Exception")
            for arg in args:
                if arg == "self":
                    continue
                if arg in defaults:
                    test_args.append(f"{arg}={defaults[arg]}")
                else:
                    test_args.append(f"{arg}=None")

        # Override with custom parameters
        for key, value in params.items():
            # If param is an argument to the function, add it to test_args
            if key in args:
                # Replace existing argument
                for i, arg in enumerate(test_args):
                    if arg.startswith(f"{key}="):
                        test_args[i] = f"{key}={value}"
                        break
                else:
                    # Add new argument
                    test_args.append(f"{key}={value}")

        # Create test code
        test_code = f"{function_name}({', '.join(test_args)})"

        # Check if we should expect success instead of exception
        if "expect_exception" in params:
            expect_exception = params["expect_exception"]

        # Generate assertion
        assertion = "result is not None"
        if "assertion" in params:
            assertion = params["assertion"]

        # Combine setup lines
        setup_code = "\n    ".join(setup_lines)

        return {
            "setup_code": setup_code,
            "test_code": test_code,
            "expect_exception": expect_exception,
            "exception_type": exception_type,
            "assertion": assertion,
        }

    def generate_test_for_error(
        self,
        file_path: Path,
        function_name: str,
        error_type: str,
        output_path: Optional[Path] = None,
        params: Dict[str, Any] = None,
    ) -> Optional[Path]:
        """
        Generate a test for a specific error type.

        Args:
            file_path: Path to the Python file
            function_name: Name of the function
            error_type: Type of error
            output_path: Path to save the test
            params: Optional parameters for the test

        Returns:
            Path to the generated test or None if failed
        """
        self.logger.info(
            f"Generating test for {function_name} in {file_path} for error type {error_type}"
        )

        # Get function details
        function_details = self._get_function_details(file_path, function_name)

        if not function_details["module_path"]:
            self.logger.error(f"Failed to find function {function_name} in {file_path}")
            return None

        # Generate test name
        test_name = self._get_test_name(function_name, error_type)
        test_class_name = "".join(word.capitalize() for word in test_name.split("_"))

        # Generate test code
        test_code = self._generate_test_code(function_details, error_type, params)

        # Create context for template
        context = {
            "module_path": function_details["module_path"],
            "function_name": function_name,
            "test_name": test_name,
            "test_class_name": test_class_name,
            "error_type": error_type,
            "error_description": f"handling {error_type} in {function_name}",
            **test_code,
        }

        # Add custom params to context
        if params:
            context.update(params)

        # Choose template based on framework
        if self.test_framework == "pytest":
            template_path = self.templates_dir / "pytest_template.txt"
        else:
            template_path = self.templates_dir / "unittest_template.txt"

        # Render the template
        test_content = self._render_template(template_path, context)

        # Determine output path
        if output_path is None:
            output_filename = f"test_{test_name}.py"
            output_path = self.output_dir / output_filename

        # Ensure parent directories exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write the test
        try:
            with open(output_path, "w") as f:
                f.write(test_content)

            self.logger.info(f"Generated test at {output_path}")
            return output_path

        except Exception as e:
            self.logger.exception(e, message=f"Failed to write test to {output_path}")
            return None

    def generate_test_for_patch(
        self, patch: Dict[str, Any], error_info: Dict[str, Any] = None
    ) -> Optional[Path]:
        """
        Generate a test for a patch.

        Args:
            patch: Patch details
            error_info: Error information from analysis

        Returns:
            Path to the generated test or None if failed
        """
        file_path = patch.get("file_path")
        if not file_path:
            self.logger.error("Patch does not contain file_path")
            return None

        # Convert to Path
        file_path = project_root / file_path

        # Extract function name from patch
        function_name = patch.get("function_name")

        # Try to get function name from patch code
        if not function_name:
            # Look for function definition in the patch code
            patch_code = patch.get("patch_code", "")
            func_match = re.search(r"def\s+(\w+)\s*\(", patch_code)
            if func_match:
                function_name = func_match.group(1)
            else:
                # Try to extract from the file
                try:
                    with open(file_path, "r") as f:
                        content = f.read()

                    # Find the line range of the patch
                    line_range = patch.get("line_range")
                    if line_range:
                        # Extract portion of the file
                        lines = content.splitlines()
                        start_line, end_line = line_range

                        # Look backwards for function definition
                        for i in range(start_line, 0, -1):
                            line = lines[i - 1]  # Adjust for 0-based indexing
                            func_match = re.search(r"def\s+(\w+)\s*\(", line)
                            if func_match:
                                function_name = func_match.group(1)
                                break
                except Exception as e:
                    self.logger.exception(
                        e, message=f"Failed to extract function name from {file_path}"
                    )

        if not function_name:
            self.logger.error(
                f"Could not determine function name for patch in {file_path}"
            )
            return None

        # Determine error type
        error_type = "Exception"

        if error_info:
            error_type = error_info.get("error_type", "Exception")
        else:
            # Try to infer from patch
            patch_code = patch.get("patch_code", "")

            # Check common error patterns
            if "KeyError" in patch_code:
                error_type = "KeyError"
            elif "IndexError" in patch_code:
                error_type = "IndexError"
            elif "TypeError" in patch_code:
                error_type = "TypeError"
            elif "ValueError" in patch_code:
                error_type = "ValueError"
            elif "AttributeError" in patch_code:
                error_type = "AttributeError"

        # Extract parameters for test
        params = {}

        # Try to extract from error info
        if error_info:
            params = error_info.get("parameters", {})

        # Try to extract from patch code
        patch_code = patch.get("patch_code", "")

        # Look for dict access
        dict_match = re.search(r"([a-zA-Z][a-zA-Z0-9_]*)\[([^\]]+)\]", patch_code)
        if dict_match:
            dict_name = dict_match.group(1)
            key_name = dict_match.group(2)
            params["dict_name"] = dict_name
            params["key_name"] = key_name

        # Look for list access
        list_match = re.search(r"([a-zA-Z][a-zA-Z0-9_]*)\[(\d+)\]", patch_code)
        if list_match:
            list_name = list_match.group(1)
            index = list_match.group(2)
            params["list_name"] = list_name
            params["index"] = index

        # Look for attribute access
        attr_match = re.search(
            r"([a-zA-Z][a-zA-Z0-9_]*)\.([a-zA-Z][a-zA-Z0-9_]*)", patch_code
        )
        if attr_match:
            obj_name = attr_match.group(1)
            attr_name = attr_match.group(2)
            params["obj_name"] = obj_name
            params["attr_name"] = attr_name

        # Generate the test
        return self.generate_test_for_error(
            file_path, function_name, error_type, params=params
        )

    def generate_regression_tests_for_patches(
        self, patches: List[Dict[str, Any]]
    ) -> List[Path]:
        """
        Generate regression tests for multiple patches.

        Args:
            patches: List of patches

        Returns:
            List of paths to generated tests
        """
        generated_tests = []

        for patch in patches:
            test_path = self.generate_test_for_patch(patch)
            if test_path:
                generated_tests.append(test_path)

        return generated_tests


if __name__ == "__main__":
    # Example usage
    generator = TestGenerator()

    # Generate a test for KeyError
    test_path = generator.generate_test_for_error(
        project_root / "services" / "example_service" / "app.py",
        "get_item",
        "KeyError",
        params={"dict_name": "items", "key_name": "'non_existent'"},
    )

    print(f"Generated test: {test_path}")

    # Create a dummy patch
    patch = {
        "patch_id": "test-patch-1",
        "file_path": "services/example_service/app.py",
        "function_name": "get_item",
        "patch_code": "try:\n    return items[item_id]\nexcept KeyError:\n    return None",
        "line_range": (10, 12),
    }

    # Generate a test for the patch
    test_path = generator.generate_test_for_patch(patch)

    print(f"Generated regression test: {test_path}")
