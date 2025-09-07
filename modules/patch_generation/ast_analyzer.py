"""
AST-based code analysis for precise patch generation.

This module provides utilities for analyzing Python code using the Abstract Syntax Tree (AST),
which enables more accurate understanding of code structure, variable scopes, and function
signatures to generate better-targeted patches.
"""

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union


@dataclass
class VariableInfo:
    """Information about a variable in the code."""

    name: str
    assignments: List[ast.AST]  # AST nodes where this variable is assigned
    usages: List[ast.AST]  # AST nodes where this variable is used
    defined_in_scope: Optional[ast.AST] = None  # Scope where this variable is defined
    value_type: Optional[str] = None  # Type of the variable if it can be inferred
    is_parameter: bool = False  # Whether this is a function parameter
    is_imported: bool = False  # Whether this is an imported name
    import_source: Optional[str] = None  # Module from which this name is imported


@dataclass
class FunctionInfo:
    """Information about a function in the code."""

    name: str
    node: ast.FunctionDef
    parameters: List[Dict[str, Any]]  # Parameter information
    return_annotation: Optional[str] = None  # Return type annotation if present
    docstring: Optional[str] = None  # Function docstring if present
    calls: List[ast.Call] = None  # Calls to this function
    is_method: bool = False  # Whether this is a class method
    is_async: bool = False  # Whether this is an async function
    decorators: List[str] = None  # Function decorators
    parent_class: Optional[str] = None  # Parent class if this is a method


@dataclass
class ClassInfo:
    """Information about a class in the code."""

    name: str
    node: ast.ClassDef
    bases: List[str]  # Base classes
    methods: Dict[str, FunctionInfo]  # Class methods
    attributes: Dict[str, VariableInfo]  # Class attributes
    docstring: Optional[str] = None  # Class docstring if present
    decorators: List[str] = None  # Class decorators


class ImportInfo(NamedTuple):
    """Information about an import statement."""

    module: str
    names: List[Tuple[str, Optional[str]]]  # (name, alias) pairs
    is_from: bool = False  # Whether this is a 'from ... import' statement
    level: int = 0  # Relative import level (0 for absolute imports)


class CodeVisitor(ast.NodeVisitor):
    """
    AST visitor that collects information about code structure.
    """

    def __init__(self):
        super().__init__()
        self.variables = {}  # name -> VariableInfo
        self.functions = {}  # name -> FunctionInfo
        self.classes = {}  # name -> ClassInfo
        self.imports = []  # List of ImportInfo

        # Stack of scopes for tracking variable definitions
        self.scope_stack = []

        # Track current function or class
        self.current_function = None
        self.current_class = None

        # Track for condition structures
        self.if_stack = []
        self.loop_stack = []
        self.exception_handlers = []

    def visit_Module(self, node):
        """Visit a Module node."""
        self.scope_stack.append(node)
        self.generic_visit(node)
        self.scope_stack.pop()

    def visit_Import(self, node):
        """Visit an Import node."""
        names = [(alias.name, alias.asname) for alias in node.names]
        for name, asname in names:
            if asname:
                self._add_variable(asname, node, is_imported=True, import_source=name)
            else:
                self._add_variable(
                    name.split(".")[0], node, is_imported=True, import_source=name
                )

        self.imports.append(ImportInfo(module="", names=names))
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        """Visit an ImportFrom node."""
        names = [(alias.name, alias.asname) for alias in node.names]
        for name, asname in names:
            var_name = asname if asname else name
            self._add_variable(
                var_name,
                node,
                is_imported=True,
                import_source=f"{node.module}.{name}" if node.module else name,
            )

        self.imports.append(
            ImportInfo(
                module=node.module or "", names=names, is_from=True, level=node.level
            )
        )
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        """Visit a FunctionDef node."""
        # Extract docstring if present
        docstring = ast.get_docstring(node)

        # Process function parameters
        parameters = []
        for arg in node.args.args:
            arg_info = {
                "name": arg.arg,
                "annotation": self._get_annotation_name(arg.annotation),
                "default": None,
            }
            parameters.append(arg_info)

            # Register parameter as a variable
            self._add_variable(arg.arg, node, is_parameter=True)

        # Process defaults
        defaults = node.args.defaults
        if defaults:
            for i, default in enumerate(defaults):
                idx = len(parameters) - len(defaults) + i
                if idx < len(parameters):
                    parameters[idx]["default"] = self._get_node_value(default)

        # Process keyword-only arguments
        for arg in node.args.kwonlyargs:
            arg_info = {
                "name": arg.arg,
                "annotation": self._get_annotation_name(arg.annotation),
                "default": None,
                "keyword_only": True,
            }
            parameters.append(arg_info)
            self._add_variable(arg.arg, node, is_parameter=True)

        # Process keyword-only defaults
        kwdefaults = node.args.kw_defaults
        if kwdefaults:
            kw_start = len(parameters) - len(node.args.kwonlyargs)
            for i, default in enumerate(kwdefaults):
                idx = kw_start + i
                if default and idx < len(parameters):
                    parameters[idx]["default"] = self._get_node_value(default)

        # Process varargs and kwargs
        if node.args.vararg:
            parameters.append(
                {
                    "name": node.args.vararg.arg,
                    "annotation": self._get_annotation_name(
                        node.args.vararg.annotation
                    ),
                    "is_vararg": True,
                }
            )
            self._add_variable(node.args.vararg.arg, node, is_parameter=True)

        if node.args.kwarg:
            parameters.append(
                {
                    "name": node.args.kwarg.arg,
                    "annotation": self._get_annotation_name(node.args.kwarg.annotation),
                    "is_kwarg": True,
                }
            )
            self._add_variable(node.args.kwarg.arg, node, is_parameter=True)

        # Extract decorator list
        decorators = []
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                decorators.append(decorator.id)
            elif isinstance(decorator, ast.Attribute):
                decorators.append(self._get_attribute_name(decorator))
            elif isinstance(decorator, ast.Call):
                if isinstance(decorator.func, ast.Name):
                    decorators.append(f"{decorator.func.id}(...)")
                elif isinstance(decorator.func, ast.Attribute):
                    decorators.append(
                        f"{self._get_attribute_name(decorator.func)}(...)"
                    )

        # Create function info
        is_method = bool(self.current_class)
        parent_class = self.current_class.name if self.current_class else None

        function_info = FunctionInfo(
            name=node.name,
            node=node,
            parameters=parameters,
            return_annotation=self._get_annotation_name(node.returns),
            docstring=docstring,
            calls=[],
            is_method=is_method,
            is_async=isinstance(node, ast.AsyncFunctionDef),
            decorators=decorators,
            parent_class=parent_class,
        )

        # Store function info
        function_key = f"{parent_class}.{node.name}" if parent_class else node.name
        self.functions[function_key] = function_info

        # Add function to class if applicable
        if self.current_class and node.name != "__init__":
            self.current_class.methods[node.name] = function_info

        # Set as current function and push scope
        prev_function = self.current_function
        self.current_function = function_info
        self.scope_stack.append(node)

        # Visit function body
        self.generic_visit(node)

        # Restore previous state
        self.scope_stack.pop()
        self.current_function = prev_function

    def visit_AsyncFunctionDef(self, node):
        """Visit an AsyncFunctionDef node."""
        # Use the same implementation as FunctionDef
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node):
        """Visit a ClassDef node."""
        # Extract docstring if present
        docstring = ast.get_docstring(node)

        # Extract base classes
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(self._get_attribute_name(base))

        # Extract decorator list
        decorators = []
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                decorators.append(decorator.id)
            elif isinstance(decorator, ast.Attribute):
                decorators.append(self._get_attribute_name(decorator))

        # Create class info
        class_info = ClassInfo(
            name=node.name,
            node=node,
            bases=bases,
            methods={},
            attributes={},
            docstring=docstring,
            decorators=decorators,
        )

        # Store class info
        self.classes[node.name] = class_info

        # Set as current class and push scope
        prev_class = self.current_class
        self.current_class = class_info
        self.scope_stack.append(node)

        # Visit class body
        self.generic_visit(node)

        # Restore previous state
        self.scope_stack.pop()
        self.current_class = prev_class

    def visit_Assign(self, node):
        """Visit an Assign node."""
        # Process each target
        for target in node.targets:
            if isinstance(target, ast.Name):
                # Simple variable assignment
                self._add_variable(target.id, node)
                self._try_infer_type(target.id, node.value)
            elif isinstance(target, ast.Attribute) and isinstance(
                target.value, ast.Name
            ):
                # Class attribute assignment
                if (
                    target.value.id == "self"
                    and self.current_function
                    and self.current_class
                ):
                    if self.current_function.name == "__init__":
                        # Record as class attribute if in __init__ method
                        attr_name = target.attr
                        self._add_class_attribute(attr_name, node)
                        self._try_infer_type(attr_name, node.value, is_attribute=True)

        # Visit the value and other parts
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        """Visit an AnnAssign (annotated assignment) node."""
        if isinstance(node.target, ast.Name):
            # Add variable with type annotation
            var_name = node.target.id
            annotation = self._get_annotation_name(node.annotation)
            self._add_variable(var_name, node, value_type=annotation)

            # Also try to infer from the value if present
            if node.value:
                self._try_infer_type(var_name, node.value)
        elif isinstance(node.target, ast.Attribute) and isinstance(
            node.target.value, ast.Name
        ):
            # Annotated class attribute
            if node.target.value.id == "self" and self.current_class:
                attr_name = node.target.attr
                annotation = self._get_annotation_name(node.annotation)
                self._add_class_attribute(attr_name, node, value_type=annotation)

                # Also try to infer from the value if present
                if node.value:
                    self._try_infer_type(attr_name, node.value, is_attribute=True)

        self.generic_visit(node)

    def visit_Name(self, node):
        """Visit a Name node."""
        if isinstance(node.ctx, ast.Load):
            # This is a variable usage
            var_name = node.id
            if var_name in self.variables:
                self.variables[var_name].usages.append(node)

        self.generic_visit(node)

    def visit_Attribute(self, node):
        """Visit an Attribute node to track attribute access."""
        # Check if the base value is a Name node (e.g., 'pd' in 'pd.DataFrame')
        if isinstance(node.value, ast.Name) and isinstance(node.value.ctx, ast.Load):
            var_name = node.value.id
            # Create variable info if it doesn't exist
            if var_name not in self.variables:
                self.variables[var_name] = VariableInfo(
                    name=var_name,
                    assignments=[],
                    usages=[],
                    defined_in_scope=self.scope_stack[-1] if self.scope_stack else None,
                )
            # Add this as a usage
            self.variables[var_name].usages.append(node.value)

        self.generic_visit(node)

    def visit_Call(self, node):
        """Visit a Call node."""
        # Record function calls
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in self.functions:
                self.functions[func_name].calls.append(node)
        elif isinstance(node.func, ast.Attribute):
            # Method calls
            if isinstance(node.func.value, ast.Name):
                if node.func.value.id == "self" and self.current_class:
                    method_name = node.func.attr
                    class_method_key = f"{self.current_class.name}.{method_name}"
                    if class_method_key in self.functions:
                        self.functions[class_method_key].calls.append(node)

        self.generic_visit(node)

    def visit_If(self, node):
        """Visit an If node to track conditional blocks."""
        self.if_stack.append(node)
        self.generic_visit(node)
        self.if_stack.pop()

    def visit_For(self, node):
        """Visit a For node to track loop blocks."""
        self.loop_stack.append(node)
        self.generic_visit(node)
        self.loop_stack.pop()

    def visit_While(self, node):
        """Visit a While node to track loop blocks."""
        self.loop_stack.append(node)
        self.generic_visit(node)
        self.loop_stack.pop()

    def visit_ExceptHandler(self, node):
        """Visit an ExceptHandler node to track exception handling."""
        self.exception_handlers.append(node)
        self.generic_visit(node)
        self.exception_handlers.pop()

    def _add_variable(
        self,
        name: str,
        node: ast.AST,
        value_type: Optional[str] = None,
        is_parameter: bool = False,
        is_imported: bool = False,
        import_source: Optional[str] = None,
    ) -> None:
        """Add a variable to the tracked variables or update its information."""
        if name in self.variables:
            # Update existing variable
            self.variables[name].assignments.append(node)
            if value_type:
                self.variables[name].value_type = value_type
            if is_parameter:
                self.variables[name].is_parameter = True
            if is_imported:
                self.variables[name].is_imported = True
                self.variables[name].import_source = import_source
        else:
            # Create new variable info
            self.variables[name] = VariableInfo(
                name=name,
                assignments=[node],
                usages=[],
                defined_in_scope=self.scope_stack[-1] if self.scope_stack else None,
                value_type=value_type,
                is_parameter=is_parameter,
                is_imported=is_imported,
                import_source=import_source,
            )

    def _add_class_attribute(
        self, name: str, node: ast.AST, value_type: Optional[str] = None
    ) -> None:
        """Add a class attribute to the current class."""
        if not self.current_class:
            return

        if name in self.current_class.attributes:
            # Update existing attribute
            self.current_class.attributes[name].assignments.append(node)
            if value_type:
                self.current_class.attributes[name].value_type = value_type
        else:
            # Create new attribute info
            self.current_class.attributes[name] = VariableInfo(
                name=name,
                assignments=[node],
                usages=[],
                defined_in_scope=(
                    self.current_function.node if self.current_function else None
                ),
                value_type=value_type,
            )

    def _try_infer_type(
        self, name: str, value_node: ast.AST, is_attribute: bool = False
    ) -> None:
        """Try to infer the type of a variable from its assigned value."""
        inferred_type = None

        if isinstance(value_node, ast.Constant):
            # Direct constants
            if value_node.value is None:
                inferred_type = "None"
            elif isinstance(value_node.value, bool):
                inferred_type = "bool"
            elif isinstance(value_node.value, int):
                inferred_type = "int"
            elif isinstance(value_node.value, float):
                inferred_type = "float"
            elif isinstance(value_node.value, str):
                inferred_type = "str"
        elif isinstance(value_node, ast.List):
            inferred_type = "list"
        elif isinstance(value_node, ast.Dict):
            inferred_type = "dict"
        elif isinstance(value_node, ast.Set):
            inferred_type = "set"
        elif isinstance(value_node, ast.Tuple):
            inferred_type = "tuple"
        elif isinstance(value_node, ast.Call):
            if isinstance(value_node.func, ast.Name):
                inferred_type = value_node.func.id
            elif isinstance(value_node.func, ast.Attribute):
                inferred_type = self._get_attribute_name(value_node.func)

        # Apply the inferred type
        if inferred_type:
            if (
                is_attribute
                and self.current_class
                and name in self.current_class.attributes
            ):
                self.current_class.attributes[name].value_type = inferred_type
            elif name in self.variables:
                self.variables[name].value_type = inferred_type

    def _get_annotation_name(self, annotation: Optional[ast.AST]) -> Optional[str]:
        """Extract a string representation of a type annotation."""
        if annotation is None:
            return None

        if isinstance(annotation, ast.Name):
            return annotation.id
        elif isinstance(annotation, ast.Attribute):
            return self._get_attribute_name(annotation)
        elif isinstance(annotation, ast.Subscript):
            if isinstance(annotation.value, ast.Name):
                base = annotation.value.id
                # For simple subscripts, like List[str]
                if isinstance(annotation.slice, ast.Index):
                    index = self._get_annotation_name(annotation.slice.value)
                    return f"{base}[{index}]"
                elif isinstance(annotation.slice, ast.Name):
                    return f"{base}[{annotation.slice.id}]"
                return f"{base}[...]"
            elif isinstance(annotation.value, ast.Attribute):
                base = self._get_attribute_name(annotation.value)
                return f"{base}[...]"
        elif isinstance(annotation, ast.Constant) and annotation.value is None:
            return "None"

        return "Any"  # Default to Any for complex annotations

    def _get_attribute_name(self, node: ast.Attribute) -> str:
        """Get the full dotted name of an attribute."""
        if isinstance(node.value, ast.Name):
            return f"{node.value.id}.{node.attr}"
        elif isinstance(node.value, ast.Attribute):
            return f"{self._get_attribute_name(node.value)}.{node.attr}"
        return f"?.{node.attr}"

    def _get_node_value(self, node: ast.AST) -> Any:
        """Try to get a Python value from a simple AST node."""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.List):
            return [self._get_node_value(elt) for elt in node.elts]
        elif isinstance(node, ast.Dict):
            keys = [self._get_node_value(k) for k in node.keys]
            values = [self._get_node_value(v) for v in node.values]
            return dict(zip(keys, values))
        elif isinstance(node, ast.Name):
            # For simple names, just return the identifier as a string
            if node.id == "True":
                return True
            elif node.id == "False":
                return False
            elif node.id == "None":
                return None
            return node.id
        return None


class ASTAnalyzer:
    """
    Analyzer class for extracting information from Python code using AST.
    """

    def __init__(self):
        self.source_code = ""
        self.ast_tree = None
        self.visitor = None
        self.file_path = None

    def parse_file(self, file_path: Union[str, Path]) -> bool:
        """
        Parse a Python file into an AST and analyze it.

        Args:
            file_path: Path to the Python file to analyze

        Returns:
            True if parsing was successful, False otherwise
        """
        try:
            # Store the file path
            self.file_path = file_path

            with open(file_path, "r", encoding="utf-8") as f:
                self.source_code = f.read()

            self.ast_tree = ast.parse(self.source_code)
            self.visitor = CodeVisitor()
            self.visitor.visit(self.ast_tree)
            return True
        except Exception as e:
            print(f"Error parsing file {file_path}: {e}")
            return False

    def parse_code(self, code: str) -> bool:
        """
        Parse a string of Python code into an AST and analyze it.

        Args:
            code: Python code as a string

        Returns:
            True if parsing was successful, False otherwise
        """
        try:
            self.source_code = code
            self.ast_tree = ast.parse(code)
            self.visitor = CodeVisitor()
            self.visitor.visit(self.ast_tree)
            return True
        except Exception as e:
            print(f"Error parsing code: {e}")
            return False

    def get_variables(self) -> Dict[str, VariableInfo]:
        """
        Get all variables found in the code.

        Returns:
            Dictionary mapping variable names to VariableInfo objects
        """
        if not self.visitor:
            return {}
        return self.visitor.variables

    def get_functions(self) -> Dict[str, FunctionInfo]:
        """
        Get all functions found in the code.

        Returns:
            Dictionary mapping function names to FunctionInfo objects
        """
        if not self.visitor:
            return {}
        return self.visitor.functions

    def get_classes(self) -> Dict[str, ClassInfo]:
        """
        Get all classes found in the code.

        Returns:
            Dictionary mapping class names to ClassInfo objects
        """
        if not self.visitor:
            return {}
        return self.visitor.classes

    def get_imports(self) -> List[ImportInfo]:
        """
        Get all imports found in the code.

        Returns:
            List of ImportInfo objects
        """
        if not self.visitor:
            return []
        return self.visitor.imports

    def find_variable_scope(self, var_name: str) -> Optional[ast.AST]:
        """
        Find the scope in which a variable is defined.

        Args:
            var_name: Name of the variable to find

        Returns:
            The AST node representing the scope, or None if not found
        """
        if not self.visitor or var_name not in self.visitor.variables:
            return None
        return self.visitor.variables[var_name].defined_in_scope

    def find_undefined_variables(self) -> List[str]:
        """
        Find variables that are used but not defined in the current scope.

        Returns:
            List of undefined variable names
        """
        if not self.visitor:
            return []

        undefined = []
        for name, var_info in self.visitor.variables.items():
            if (
                var_info.usages
                and not var_info.assignments
                and not var_info.is_parameter
            ):
                undefined.append(name)

        return undefined

    def find_function_parameters(self, func_name: str) -> List[Dict[str, Any]]:
        """
        Get the parameters of a function.

        Args:
            func_name: Name of the function

        Returns:
            List of parameter information dictionaries, or empty list if function not found
        """
        if not self.visitor:
            return []

        # Try direct function lookup
        if func_name in self.visitor.functions:
            return self.visitor.functions[func_name].parameters

        # Try class method lookup (Class.method)
        if "." in func_name and func_name in self.visitor.functions:
            return self.visitor.functions[func_name].parameters

        return []

    def get_function_info(self, func_name: str) -> Optional[FunctionInfo]:
        """
        Get detailed information about a function.

        Args:
            func_name: Name of the function

        Returns:
            FunctionInfo object if found, None otherwise
        """
        if not self.visitor:
            return None

        # Try direct function lookup
        if func_name in self.visitor.functions:
            return self.visitor.functions[func_name]

        # Try class method lookup (Class.method)
        if "." in func_name and func_name in self.visitor.functions:
            return self.visitor.functions[func_name]

        return None

    def get_function_calls(self, func_name: str) -> List[ast.Call]:
        """
        Get all calls to a specific function.

        Args:
            func_name: Name of the function

        Returns:
            List of Call nodes, or empty list if function not found
        """
        func_info = self.get_function_info(func_name)
        if func_info:
            return func_info.calls
        return []

    def extract_code_at_lines(self, start_line: int, end_line: int) -> str:
        """
        Extract the code between the specified line numbers.

        Args:
            start_line: Starting line number (1-based)
            end_line: Ending line number (1-based)

        Returns:
            The extracted code as a string
        """
        if not self.source_code:
            return ""

        lines = self.source_code.splitlines()
        if start_line < 1:
            start_line = 1
        if end_line > len(lines):
            end_line = len(lines)

        return "\n".join(lines[start_line - 1:end_line])

    def find_nodes_at_line(self, line_number: int) -> List[ast.AST]:
        """
        Find AST nodes at a specific line number.

        Args:
            line_number: Line number to search (1-based)

        Returns:
            List of AST nodes at that line
        """
        if not self.ast_tree:
            return []

        nodes = []

        class LineVisitor(ast.NodeVisitor):
            def generic_visit(self, node):
                if hasattr(node, "lineno") and node.lineno == line_number:
                    nodes.append(node)
                super().generic_visit(node)

        LineVisitor().visit(self.ast_tree)
        return nodes

    def find_containing_function(self, line_number: int) -> Optional[FunctionInfo]:
        """
        Find the function containing a specific line of code.

        Args:
            line_number: Line number to check (1-based)

        Returns:
            FunctionInfo of the containing function, or None if not in a function
        """
        if not self.visitor:
            return None

        for func_name, func_info in self.visitor.functions.items():
            if hasattr(func_info.node, "lineno") and hasattr(
                func_info.node, "end_lineno"
            ):
                start = func_info.node.lineno
                end = func_info.node.end_lineno
                if start <= line_number <= end:
                    return func_info

        return None

    def find_containing_class(self, line_number: int) -> Optional[ClassInfo]:
        """
        Find the class containing a specific line of code.

        Args:
            line_number: Line number to check (1-based)

        Returns:
            ClassInfo of the containing class, or None if not in a class
        """
        if not self.visitor:
            return None

        for class_name, class_info in self.visitor.classes.items():
            if hasattr(class_info.node, "lineno") and hasattr(
                class_info.node, "end_lineno"
            ):
                start = class_info.node.lineno
                end = class_info.node.end_lineno
                if start <= line_number <= end:
                    return class_info

        return None


if __name__ == "__main__":
    # Example usage
    analyzer = ASTAnalyzer()

    # Simple test code
    test_code = """
import os
from typing import List, Dict, Optional

class ExampleClass:
    \"\"\"
    Example class docstring.
    \"\"\"
    
    def __init__(self, name: str, value: int = 0):
        self.name = name
        self.value = value
        self._private = True
        
    def process(self, data: List[str]) -> Dict[str, int]:
        \"\"\"Process the data and return results.\"\"\"
        result = {}
        for item in data:
            if item in self.name:
                result[item] = self.value
            else:
                result[item] = 0
        return result
        
def helper_function(x: int, y: Optional[int] = None) -> int:
    \"\"\"Helper function docstring.\"\"\"
    if y is None:
        y = 0
    return x + y

# Main code
example = ExampleClass("test", 42)
data_list = ["one", "two", "three"]
results = example.process(data_list)
print(results)
"""

    if analyzer.parse_code(test_code):
        # Print variables
        print("Variables:")
        for name, var_info in analyzer.get_variables().items():
            var_type = f": {var_info.value_type}" if var_info.value_type else ""
            print(f"  {name}{var_type}")

        # Print functions
        print("\nFunctions:")
        for name, func_info in analyzer.get_functions().items():
            params = ", ".join(
                f"{p['name']}: {p['annotation']}"
                for p in func_info.parameters
                if p.get("annotation")
            )
            returns = (
                f" -> {func_info.return_annotation}"
                if func_info.return_annotation
                else ""
            )
            print(f"  {name}({params}){returns}")

        # Print classes
        print("\nClasses:")
        for name, class_info in analyzer.get_classes().items():
            bases = f"({', '.join(class_info.bases)})" if class_info.bases else ""
            print(f"  {name}{bases}")

            print("    Methods:")
            for method_name, method_info in class_info.methods.items():
                print(f"      {method_name}")

            print("    Attributes:")
            for attr_name, attr_info in class_info.attributes.items():
                attr_type = f": {attr_info.value_type}" if attr_info.value_type else ""
                print(f"      {attr_name}{attr_type}")

        # Find function containing a line
        line_num = 23  # Line in the process method
        containing_func = analyzer.find_containing_function(line_num)
        containing_class = analyzer.find_containing_class(line_num)

        if containing_func:
            print(f"\nLine {line_num} is in function: {containing_func.name}")

        if containing_class:
            print(f"Line {line_num} is in class: {containing_class.name}")

        # Find undefined variables (should be empty for this example)
        undefined = analyzer.find_undefined_variables()
        if undefined:
            print("\nUndefined variables:", undefined)

        # Extract parameters of a function
        helper_params = analyzer.find_function_parameters("helper_function")
        print("\nhelper_function parameters:")
        for param in helper_params:
            default = (
                f" = {param['default']}"
                if "default" in param and param["default"] is not None
                else ""
            )
            annotation = f": {param['annotation']}" if param.get("annotation") else ""
            print(f"  {param['name']}{annotation}{default}")
