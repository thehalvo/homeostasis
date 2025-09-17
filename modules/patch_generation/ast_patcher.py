"""
AST-based patch generation for more precise code fixes.

This module uses the AST analyzer to generate more accurate and context-aware patches
by understanding code structure, variable scopes, and function signatures.
"""

import ast
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from modules.patch_generation.ast_analyzer import ASTAnalyzer, ClassInfo, VariableInfo
from modules.patch_generation.patcher import PatchGenerator

logger = logging.getLogger(__name__)


class ASTPatcher:
    """
    Enhanced patch generator that uses AST analysis for more precise patches.
    """

    def __init__(self, templates_dir: Path, patch_generator: Optional[PatchGenerator] = None):
        """
        Initialize the AST patcher.

        Args:
            templates_dir: Directory containing patch templates
            patch_generator: Optional existing PatchGenerator to wrap
        """
        self.templates_dir = templates_dir
        self.patch_generator = patch_generator or PatchGenerator(templates_dir)
        self.template_manager = self.patch_generator.template_manager
        self.analyzer = ASTAnalyzer()

    def analyze_file(self, file_path: Path) -> bool:
        """
        Analyze a file to gather information for patching.

        Args:
            file_path: Path to the file to analyze

        Returns:
            True if analysis was successful, False otherwise
        """
        return self.analyzer.parse_file(file_path)

    def analyze_code(self, code: str) -> bool:
        """
        Analyze code string to gather information for patching.

        Args:
            code: Python code as a string

        Returns:
            True if analysis was successful, False otherwise
        """
        return self.analyzer.parse_code(code)

    def get_variables_in_scope(self, line_number: int) -> Dict[str, VariableInfo]:
        """
        Get variables available in the scope at a specific line.

        Args:
            line_number: Line number to check

        Returns:
            Dictionary of variable names to VariableInfo objects
        """
        if not self.analyzer.visitor:
            return {}

        # Get the function and class containing this line
        containing_function = self.analyzer.find_containing_function(line_number)
        containing_class = self.analyzer.find_containing_class(line_number)

        # Collect variables in scope
        variables_in_scope = {}

        # Module-level variables are always in scope
        for name, var_info in self.analyzer.get_variables().items():
            if var_info.defined_in_scope and isinstance(
                var_info.defined_in_scope, ast.Module
            ):
                variables_in_scope[name] = var_info

        # Add class attributes if in a class
        if containing_class:
            for name, attr_info in containing_class.attributes.items():
                variables_in_scope[name] = attr_info

            # Add 'self' if in a method
            if containing_function and containing_function.is_method:
                # Create a synthetic VariableInfo for 'self'
                self_var = VariableInfo(
                    name="self",
                    assignments=[],
                    usages=[],
                    defined_in_scope=containing_function.node,
                    value_type=containing_class.name,
                    is_parameter=True,
                )
                variables_in_scope["self"] = self_var

        # Add function parameters and local variables if in a function
        if containing_function:
            for name, var_info in self.analyzer.get_variables().items():
                if var_info.defined_in_scope is containing_function.node:
                    variables_in_scope[name] = var_info

        return variables_in_scope

    def generate_exception_handler(
        self, line_number: int, exception_type: str
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a try-except block around code at the specified line.

        Args:
            line_number: Line number to add exception handling to
            exception_type: Type of exception to catch

        Returns:
            Patch details if generation is successful, None otherwise
        """
        if not self.analyzer.visitor:
            logger.debug("No visitor - analyzer not initialized")
            return None

        # Find the containing function to get context
        containing_function = self.analyzer.find_containing_function(line_number)
        if not containing_function:
            logger.debug(f"No containing function found for line {line_number}")
            return None

        # Identify the statement to wrap in a try-except block
        nodes_at_line = self.analyzer.find_nodes_at_line(line_number)
        if not nodes_at_line:
            logger.debug(f"No nodes found at line {line_number}")
            return None

        # Get statement node and its context
        statement_node = nodes_at_line[0]
        parent_node = None

        # Find the actual statement and its parent
        # We need to traverse the AST to find the parent of our statement
        for node in ast.walk(containing_function.node):
            for child in ast.iter_child_nodes(node):
                if any(
                    child is n
                    or hasattr(child, "body")
                    and any(b is n for b in child.body)
                    for n in nodes_at_line
                ):
                    parent_node = node
                    break

        if not parent_node:
            return None

        # Find actual line range to wrap
        start_line = line_number
        end_line = line_number

        if hasattr(statement_node, "end_lineno"):
            end_line = statement_node.end_lineno

        # Extract the code block to wrap
        code_block = self.analyzer.extract_code_at_lines(start_line, end_line)

        # Get variables in scope for exception handling
        variables_in_scope = self.get_variables_in_scope(line_number)

        # Find the best recovery action based on context
        recovery_action = self._generate_recovery_action(
            exception_type, statement_node, variables_in_scope
        )

        # Detect if we need to import the exception type
        import_statement = None
        if "." in exception_type and exception_type not in str(
            self.analyzer.get_imports()
        ):
            # Likely need to import this exception
            module = exception_type.split(".")[0]
            import_statement = f"from {module} import {exception_type.split('.')[-1]}"

        # Generate patch using the try-except template
        template_name = "try_except_block.py.template"
        template = self.template_manager.get_template(template_name)

        if not template:
            return None

        variables = {
            "code_block": code_block,
            "exception_type": exception_type,
            "error_message": f"Error occurred during {containing_function.name}",
            "recovery_action": recovery_action,
            "log_error": "true",
            "import_statement": import_statement,
        }

        # Create the patch
        return {
            "template_name": template_name,
            "file_path": str(self.analyzer.file_path),
            "line_range": (start_line, end_line),
            "variables": variables,
            "patch_type": "specific",
            "patch_code": template.render({k: v for k, v in variables.items() if v is not None}),
            "is_multiline": True,
            "ast_analyzed": True,
        }

    def generate_parameter_check(
        self, function_name: str, parameter_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Generate code to check and handle a missing or invalid parameter.

        Args:
            function_name: Name of the function to patch
            parameter_name: Name of the parameter to check

        Returns:
            Patch details if generation is successful, None otherwise
        """
        if not self.analyzer.visitor:
            return None

        # Get function info
        function_info = self.analyzer.get_function_info(function_name)
        if not function_info:
            return None

        # Find the parameter
        param_info = None
        for param in function_info.parameters:
            if param["name"] == parameter_name:
                param_info = param
                break

        if not param_info:
            return None

        # Determine parameter type for proper checking
        param_type = param_info.get("annotation")

        # Generate appropriate check based on type
        check_template_name = "parameter_check.py.template"
        template = self.template_manager.get_template(check_template_name)

        if not template:
            # Fall back to regular dictionary check if template not found
            return self.patch_generator.generate_multiline_patch(
                "dict_key_not_exists",
                Path(self.analyzer.file_path) if self.analyzer.file_path else Path(),
                (function_info.node.lineno, function_info.node.lineno + 1),
                {
                    "key_name": parameter_name,
                    "dict_name": "kwargs" if param_info.get("is_kwarg") else "args",
                    "default_value": str(param_info.get("default", "None")),
                },
            )

        # Determine where to insert the check (after docstring if present)
        insert_line = function_info.node.lineno
        if function_info.docstring:
            # Skip the docstring to insert after it
            docstring_node = function_info.node.body[0]
            if isinstance(docstring_node, ast.Expr) and isinstance(
                docstring_node.value, ast.Constant
            ):
                insert_line = docstring_node.lineno + 1

        # Construct appropriate check based on parameter type
        if param_type == "str":
            check_code = f"if {parameter_name} is None or not {parameter_name}.strip():"
            default_value = (
                '""'
                if param_info.get("default") is None
                else repr(param_info.get("default", ""))
            )
        elif param_type in ("int", "float"):
            check_code = f"if {parameter_name} is None:"
            default_value = (
                "0"
                if param_info.get("default") is None
                else str(param_info.get("default", 0))
            )
        elif param_type in ("list", "List", "tuple", "Tuple", "set", "Set"):
            check_code = f"if {parameter_name} is None:"
            default_value = (
                "[]"
                if param_info.get("default") is None
                else str(param_info.get("default", []))
            )
        elif param_type in ("dict", "Dict", "mapping", "Mapping"):
            check_code = f"if {parameter_name} is None:"
            default_value = (
                "{}"
                if param_info.get("default") is None
                else str(param_info.get("default", {}))
            )
        else:
            # Generic check for any other type
            check_code = f"if {parameter_name} is None:"
            default_value = (
                "None"
                if param_info.get("default") is None
                else str(param_info.get("default", "None"))
            )

        variables = {
            "parameter_name": parameter_name,
            "check_code": check_code,
            "default_value": default_value,
            "parameter_type": param_type or "Any",
            "raise_error": "false",  # Default to not raising an error
            "error_type": "ValueError",
            "error_message": f"Missing required parameter: {parameter_name}",
        }

        # Determine if function body already has a parameter check
        has_existing_check = False
        for node in ast.walk(function_info.node):
            if isinstance(node, ast.If) and hasattr(node, "test"):
                test_code = self.analyzer.extract_code_at_lines(
                    node.test.lineno, node.test.lineno
                )
                if parameter_name in test_code and "None" in test_code:
                    has_existing_check = True
                    break

        if has_existing_check:
            # Don't add duplicate checks
            return None

        # Generate patch
        code_block = self.analyzer.extract_code_at_lines(insert_line, insert_line)
        match = re.match(r"^(\s*)", code_block)
        indentation = match.group(1) if match else ""

        variables["__indentation__"] = indentation

        rendered_code = template.render(variables)

        return {
            "template_name": check_template_name,
            "file_path": str(self.analyzer.file_path),
            "line_range": (insert_line, insert_line),
            "variables": variables,
            "patch_type": "specific",
            "patch_code": rendered_code,
            "is_multiline": True,
            "ast_analyzed": True,
            "insertion": True,  # This is an insertion, not a replacement
        }

    def generate_type_conversion(
        self, line_number: int, variable_name: str, target_type: str
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a type conversion patch for a variable.

        Args:
            line_number: Line number to add conversion to
            variable_name: Name of the variable to convert
            target_type: Type to convert to

        Returns:
            Patch details if generation is successful, None otherwise
        """
        if not self.analyzer.visitor:
            return None

        # Find the variable usage
        variables_in_scope = self.get_variables_in_scope(line_number)
        if variable_name not in variables_in_scope:
            return None

        # Find the node at this line
        nodes_at_line = self.analyzer.find_nodes_at_line(line_number)
        if not nodes_at_line:
            return None

        # Extract the code and indentation
        code_block = self.analyzer.extract_code_at_lines(line_number, line_number)
        match = re.match(r"^(\s*)", code_block)
        indentation = match.group(1) if match else ""

        # Prepare template variables
        template_variables = {
            "var_name": variable_name,
            "target_type": target_type,
            "error_message": f"Could not convert {variable_name} to {target_type}",
            "default_value": self._get_default_for_type(target_type),
            "__indentation__": indentation,
        }

        # Handle specific conversion scenarios
        if target_type == "int":
            template_name = "int_conversion_error.py.template"
        elif target_type == "float":
            template_name = "float_conversion_error.py.template"
        elif target_type == "str":
            template_name = "str_conversion_error.py.template"
        elif target_type == "bool":
            template_name = "bool_conversion_error.py.template"
        else:
            # Generic conversion
            template_name = "type_conversion.py.template"

        # Get the template
        template = self.template_manager.get_template(template_name)
        if not template:
            return None

        # Render the template
        rendered_code = template.render(template_variables)

        return {
            "template_name": template_name,
            "file_path": str(self.analyzer.file_path),
            "line_range": (line_number, line_number),
            "variables": template_variables,
            "patch_type": "specific",
            "patch_code": rendered_code,
            "is_multiline": False,
            "ast_analyzed": True,
        }

    def add_missing_import(
        self, module_name: str, name: str, alias: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a patch to add a missing import.

        Args:
            module_name: Name of the module to import from
            name: Name to import
            alias: Optional alias for the imported name

        Returns:
            Patch details if generation is successful, None otherwise
        """
        if not self.analyzer.visitor:
            return None

        # Check if import already exists
        for import_info in self.analyzer.get_imports():
            if import_info.is_from and import_info.module == module_name:
                for imported_name, imported_alias in import_info.names:
                    if imported_name == name and imported_alias == alias:
                        # Import already exists
                        return None
            elif not import_info.is_from and module_name == name:
                # Direct module import already exists
                return None

        # Find where to insert the import
        # Typically after other imports, before code
        import_line = 1  # Default to start of file
        last_import_line = 0

        for import_info in self.analyzer.get_imports():
            # ImportInfo doesn't have lineno, so we can't determine the line
            # We'll use line 1 as a fallback
            pass

        if last_import_line > 0:
            # Insert after the last import
            import_line = last_import_line + 1

        # Generate import statement
        if module_name:
            if alias:
                import_stmt = f"from {module_name} import {name} as {alias}"
            else:
                import_stmt = f"from {module_name} import {name}"
        else:
            if alias:
                import_stmt = f"import {name} as {alias}"
            else:
                import_stmt = f"import {name}"

        return {
            "template_name": "add_import.py.template",
            "file_path": str(self.analyzer.file_path),
            "line_range": (import_line, import_line),
            "variables": {"import_statement": import_stmt},
            "patch_type": "specific",
            "patch_code": import_stmt,
            "is_multiline": False,
            "ast_analyzed": True,
            "insertion": True,
        }

    def fix_attribute_error(
        self, line_number: int, obj_name: str, attr_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a patch to fix an AttributeError.

        Args:
            line_number: Line where the error occurs
            obj_name: Name of the object with missing attribute
            attr_name: Name of the missing attribute

        Returns:
            Patch details if generation is successful, None otherwise
        """
        if not self.analyzer.visitor:
            return None

        # Find the variable for the object
        variables_in_scope = self.get_variables_in_scope(line_number)
        if obj_name not in variables_in_scope:
            return None

        obj_info = variables_in_scope[obj_name]

        # Determine object type if possible
        obj_type = obj_info.value_type

        # Extract the code and indentation
        code_block = self.analyzer.extract_code_at_lines(line_number, line_number)
        match = re.match(r"^(\s*)", code_block)
        indentation = match.group(1) if match else ""

        # Find a class definition matching the object type if available
        class_info = None
        if obj_type and obj_type in self.analyzer.get_classes():
            class_info = self.analyzer.get_classes()[obj_type]

        # Generate a patch based on the context
        template_name = "attribute_error.py.template"
        template = self.template_manager.get_template(template_name)

        if not template:
            return None

        # Determine a default value for the attribute based on naming conventions
        default_value = self._infer_default_value(attr_name)

        # Prepare template variables
        template_variables = {
            "object_name": obj_name,
            "attribute_name": attr_name,
            "error_message": f"Object {obj_name} has no attribute {attr_name}",
            "default_value": default_value,
            "object_type": obj_type or "object",
            "class_has_attribute": "false",
            "__indentation__": indentation,
        }

        # Check if the class has a similar attribute (for typo fixes)
        if class_info:
            similar_attrs = self._find_similar_attributes(attr_name, class_info)
            if similar_attrs:
                template_variables["similar_attributes"] = similar_attrs
                template_variables["suggested_attribute"] = similar_attrs[0]

        # Render the template
        rendered_code = template.render(template_variables)

        return {
            "template_name": template_name,
            "file_path": str(self.analyzer.file_path),
            "line_range": (line_number, line_number),
            "variables": template_variables,
            "patch_type": "specific",
            "patch_code": rendered_code,
            "is_multiline": True,
            "ast_analyzed": True,
        }

    def _generate_recovery_action(
        self,
        exception_type: str,
        node: ast.AST,
        variables_in_scope: Dict[str, VariableInfo],
    ) -> str:
        """
        Generate an appropriate recovery action for an exception handler.

        Args:
            exception_type: Type of exception being caught
            node: AST node where the exception could occur
            variables_in_scope: Variables available in the current scope

        Returns:
            Code string for recovery action
        """
        # Default recovery based on exception type
        if exception_type == "KeyError" or exception_type == "IndexError":
            # For key/index errors, return a default value
            if isinstance(node, ast.Subscript):
                return "    # Return None or a default value when the key/index is not found\n    return None"

        elif exception_type == "ValueError":
            # For value errors, often related to conversions
            if isinstance(node, ast.Call):
                func_name = self._get_node_source(node.func)
                if func_name in ("int", "float", "complex"):
                    return (
                        "    # Use a default value when conversion fails\n    return 0"
                    )

        elif exception_type == "AttributeError":
            # For attribute errors, provide a dummy attribute
            if isinstance(node, ast.Attribute):
                return "    # Handle missing attribute\n    return None"

        elif exception_type == "TypeError":
            # Type errors often involve calling non-callables or wrong argument types
            return "    # Handle type mismatch\n    return None"

        elif exception_type == "ZeroDivisionError":
            # Division by zero
            return "    # Handle division by zero\n    return float('inf')  # or 0 depending on the context"

        # Generic recovery action
        if isinstance(node, ast.Return):
            # If in a return statement, return a sensible default
            return "    # Return a default value on error\n    return None"
        elif isinstance(node, ast.Assign):
            # For assignments, assign a default value
            return "    # Assign a default value on error\n    pass"
        elif isinstance(node, ast.Call):
            # For function calls, just pass
            return "    # Skip the function call on error\n    pass"

        # Very generic fallback
        return "    # Handle the error\n    pass"

    def _get_node_source(self, node: ast.AST) -> str:
        """
        Get source code for an AST node.

        Args:
            node: AST node

        Returns:
            Source code string
        """
        if hasattr(node, "lineno") and hasattr(node, "end_lineno"):
            return self.analyzer.extract_code_at_lines(node.lineno, node.end_lineno)
        return "unknown"

    def _get_default_for_type(self, type_name: str) -> str:
        """
        Get a default value string for a given type.

        Args:
            type_name: Name of the type

        Returns:
            Default value string
        """
        if type_name in ("int", "Int"):
            return "0"
        elif type_name in ("float", "Float"):
            return "0.0"
        elif type_name in ("str", "Str", "string", "String"):
            return '""'
        elif type_name in ("bool", "Bool", "boolean", "Boolean"):
            return "False"
        elif type_name in ("list", "List"):
            return "[]"
        elif type_name in ("dict", "Dict", "map", "Map"):
            return "{}"
        elif type_name in ("tuple", "Tuple"):
            return "()"
        elif type_name in ("set", "Set"):
            return "set()"
        return "None"

    def _infer_default_value(self, attr_name: str) -> str:
        """
        Infer a reasonable default value based on attribute name.

        Args:
            attr_name: Name of the attribute

        Returns:
            Default value string
        """
        # Check for common naming patterns
        if attr_name.startswith(("is_", "has_", "can_", "should_")):
            return "False"
        elif attr_name.endswith(("_count", "_size", "_length", "_index", "_id")):
            return "0"
        elif attr_name.endswith(("_name", "_key", "_path", "_url")):
            return '""'
        elif attr_name.endswith(("_list", "_array")):
            return "[]"
        elif attr_name.endswith(("_dict", "_map")):
            return "{}"
        elif attr_name.endswith(("_set",)):
            return "set()"
        return "None"

    def _find_similar_attributes(
        self, attr_name: str, class_info: ClassInfo
    ) -> List[str]:
        """
        Find attributes in a class that are similar to the given name.

        Args:
            attr_name: Attribute name to find similar matches for
            class_info: Class information

        Returns:
            List of similar attribute names
        """
        similar_attrs = []

        # Combine attributes and methods
        all_attrs = list(class_info.attributes.keys()) + list(class_info.methods.keys())

        # Look for exact matches first (ignoring case)
        for attr in all_attrs:
            if attr.lower() == attr_name.lower():
                similar_attrs.append(attr)

        # If no exact match, look for close matches
        if not similar_attrs:
            for attr in all_attrs:
                # Simple string distance: count characters in common
                if len(attr) > 2 and (attr in attr_name or attr_name in attr):
                    similar_attrs.append(attr)

        # Sort by similarity (length of common substring)
        similar_attrs.sort(
            key=lambda a: self._similarity_score(a, attr_name), reverse=True
        )

        return similar_attrs[:3]  # Return top 3 matches

    def _similarity_score(self, str1: str, str2: str) -> int:
        """
        Calculate a similarity score between two strings.

        Args:
            str1: First string
            str2: Second string

        Returns:
            Similarity score (higher means more similar)
        """
        # Simple Levenshtein distance-like score
        m, n = len(str1), len(str2)
        if m == 0 or n == 0:
            return 0

        # Count matching characters in correct positions
        matches = sum(1 for i in range(min(m, n)) if str1[i] == str2[i])

        # Penalize length difference
        length_diff_penalty = abs(m - n)

        # Reward longer common substrings
        common_len = 0
        for i in range(min(m, n)):
            if str1[i] == str2[i]:
                common_len += 1

        # Combine factors
        return matches + common_len - length_diff_penalty


if __name__ == "__main__":
    # Example usage
    from pathlib import Path

    # Simple test code
    test_code = """
import os
from typing import List, Dict, Optional

def process_data(data_list, transform_func=None):
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

# Example usage
data = [
    {'id': 1, 'value': '42'},
    {'id': 2, 'value': 'not-a-number'},
    {}  # Missing required keys
]

class Transformer:
    def apply(self, value):
        return value * 2

try:
    results = process_data(data, Transformer())
    print(results)
except Exception as e:
    print(f"Error: {e}")
"""

    # Set up the AST patcher
    templates_dir = Path(__file__).parent / "templates"
    patch_generator = PatchGenerator(templates_dir)
    ast_patcher = ASTPatcher(templates_dir, patch_generator)

    # Write the test code to a temporary file
    import tempfile

    test_file = Path(tempfile.gettempdir()) / "test_ast_patcher.py"
    with open(test_file, "w") as f:
        f.write(test_code)

    # Analyze the file
    success = ast_patcher.analyze_file(test_file)

    if success:
        print("AST analysis successful")

        # Generate a patch for the int conversion at line 17
        conversion_patch = ast_patcher.generate_type_conversion(17, "value", "int")
        if conversion_patch:
            print("\nType conversion patch:")
            print(conversion_patch["patch_code"])

        # Generate exception handler for the potential KeyError at line 14
        exception_patch = ast_patcher.generate_exception_handler(14, "KeyError")
        if exception_patch:
            print("\nException handler patch:")
            print(exception_patch["patch_code"])

        # Generate parameter check for the process_data function
        param_patch = ast_patcher.generate_parameter_check("process_data", "data_list")
        if param_patch:
            print("\nParameter check patch:")
            print(param_patch["patch_code"])

        # Generate AttributeError fix for the transform_func.apply call
        attr_patch = ast_patcher.fix_attribute_error(20, "transform_func", "apply")
        if attr_patch:
            print("\nAttributeError fix patch:")
            print(attr_patch["patch_code"])

    # Clean up
    if test_file.exists():
        test_file.unlink()
