"""
Function signature analysis for detecting and fixing parameter errors.

This module provides specialized tools for analyzing function signatures, parameter usage,
and generating patches for common parameter-related issues like missing parameters,
type mismatches, and ordering problems.
"""
import ast
import inspect
import re
from typing import Dict, List, Optional, Any, Tuple, Union, Set, NamedTuple
from dataclasses import dataclass
from pathlib import Path

from modules.patch_generation.ast_analyzer import ASTAnalyzer, FunctionInfo


@dataclass
class ParameterIssue:
    """Information about a parameter issue."""
    function_name: str
    parameter_name: str
    issue_type: str  # 'missing', 'type_mismatch', 'order', 'unused', etc.
    line_number: int
    suggestion: Optional[str] = None
    severity: str = "medium"  # 'low', 'medium', 'high'


class FunctionCallMatch:
    """
    Information about a match between function call and definition.
    """
    def __init__(self, function_info: FunctionInfo, call_node: ast.Call, 
                line_number: int, file_path: Optional[str] = None):
        self.function_info = function_info
        self.call_node = call_node
        self.line_number = line_number
        self.file_path = file_path
        self.issues: List[ParameterIssue] = []
        
        # Match parameters between call and definition
        self.param_matches = {}  # param_name -> arg_value
        self.unmatched_args = []  # positional args that couldn't be matched
        self.unmatched_params = []  # parameters without corresponding args
        
        # Analyze the match
        self._analyze_match()
        
    def _analyze_match(self):
        """Analyze the match between call and definition."""
        # Get function parameters
        params = self.function_info.parameters
        
        # Track which parameters have been matched
        matched_params = set()
        
        # Match positional arguments
        for i, arg in enumerate(self.call_node.args):
            if i < len(params):
                param_name = params[i]['name']
                self.param_matches[param_name] = arg
                matched_params.add(param_name)
            else:
                self.unmatched_args.append(arg)
                
                # Create an issue for too many positional arguments
                self.issues.append(ParameterIssue(
                    function_name=self.function_info.name,
                    parameter_name=f"arg{i}",
                    issue_type="extra_positional",
                    line_number=self.line_number,
                    suggestion=f"Remove extra positional argument or update function signature"
                ))
        
        # Match keyword arguments
        for keyword in self.call_node.keywords:
            # Handle **kwargs special case
            if keyword.arg is None:
                # This is a **kwargs argument, can't analyze statically
                continue
                
            # Check if parameter exists
            param_found = False
            for param in params:
                if param['name'] == keyword.arg:
                    self.param_matches[keyword.arg] = keyword.value
                    matched_params.add(keyword.arg)
                    param_found = True
                    break
                    
            if not param_found:
                # Create an issue for unknown keyword argument
                self.issues.append(ParameterIssue(
                    function_name=self.function_info.name,
                    parameter_name=keyword.arg,
                    issue_type="unknown_keyword",
                    line_number=self.line_number,
                    suggestion=f"Remove unknown keyword argument or update function signature"
                ))
        
        # Check for missing required parameters
        for param in params:
            if param['name'] not in matched_params:
                # Skip parameters with default values
                if 'default' in param and param['default'] is not None:
                    continue
                    
                # Skip *args and **kwargs
                if param.get('is_vararg') or param.get('is_kwarg'):
                    continue
                    
                self.unmatched_params.append(param)
                
                # Create an issue for missing required parameter
                self.issues.append(ParameterIssue(
                    function_name=self.function_info.name,
                    parameter_name=param['name'],
                    issue_type="missing_required",
                    line_number=self.line_number,
                    suggestion=f"Add required parameter '{param['name']}'"
                ))
    
    def get_issues(self) -> List[ParameterIssue]:
        """Get all issues found in this call."""
        return self.issues
        
    def has_issues(self) -> bool:
        """Check if this call has any issues."""
        return len(self.issues) > 0


class FunctionSignatureAnalyzer:
    """
    Analyzer for function signatures and call sites.
    """
    
    def __init__(self, ast_analyzer: ASTAnalyzer = None):
        """
        Initialize the function signature analyzer.
        
        Args:
            ast_analyzer: Optional existing ASTAnalyzer to use
        """
        self.ast_analyzer = ast_analyzer or ASTAnalyzer()
        self.file_path = None
        
    def analyze_file(self, file_path: Path) -> bool:
        """
        Analyze a file to gather function signature information.
        
        Args:
            file_path: Path to the file to analyze
            
        Returns:
            True if analysis was successful, False otherwise
        """
        self.file_path = str(file_path)
        return self.ast_analyzer.parse_file(file_path)
        
    def analyze_code(self, code: str) -> bool:
        """
        Analyze code to gather function signature information.
        
        Args:
            code: Python code as a string
            
        Returns:
            True if analysis was successful, False otherwise
        """
        return self.ast_analyzer.parse_code(code)
        
    def find_function_calls(self, function_name: str) -> List[FunctionCallMatch]:
        """
        Find all calls to a specific function and analyze parameter usage.
        
        Args:
            function_name: Name of the function to find calls for
            
        Returns:
            List of FunctionCallMatch objects for each call site
        """
        # Get function info
        function_info = self.ast_analyzer.get_function_info(function_name)
        if not function_info:
            return []
            
        # Find all calls to this function
        call_nodes = self.ast_analyzer.get_function_calls(function_name)
        
        # Create match objects for each call
        matches = []
        for call_node in call_nodes:
            if hasattr(call_node, 'lineno'):
                matches.append(FunctionCallMatch(
                    function_info=function_info,
                    call_node=call_node,
                    line_number=call_node.lineno,
                    file_path=self.file_path
                ))
                
        return matches
        
    def find_all_function_calls(self) -> Dict[str, List[FunctionCallMatch]]:
        """
        Find all function calls in the analyzed code.
        
        Returns:
            Dictionary mapping function names to lists of FunctionCallMatch objects
        """
        result = {}
        
        # Get all defined functions
        functions = self.ast_analyzer.get_functions()
        
        # For each function, find its calls
        for func_name in functions:
            calls = self.find_function_calls(func_name)
            if calls:
                result[func_name] = calls
                
        return result
        
    def find_all_parameter_issues(self) -> List[ParameterIssue]:
        """
        Find all parameter issues in the analyzed code.
        
        Returns:
            List of ParameterIssue objects
        """
        all_issues = []
        
        # Find all function calls
        all_calls = self.find_all_function_calls()
        
        # Collect issues from all calls
        for func_name, calls in all_calls.items():
            for call_match in calls:
                all_issues.extend(call_match.get_issues())
                
        return all_issues
        
    def is_parameter_used(self, function_name: str, parameter_name: str) -> bool:
        """
        Check if a parameter is actually used in the function body.
        
        Args:
            function_name: Name of the function to check
            parameter_name: Name of the parameter to check
            
        Returns:
            True if the parameter is used, False otherwise
        """
        # Get function info
        function_info = self.ast_analyzer.get_function_info(function_name)
        if not function_info:
            return False
            
        # Get function body
        function_node = function_info.node
        if not function_node or not hasattr(function_node, 'body'):
            return False
            
        # Check if parameter is used in function body
        for node in ast.walk(function_node):
            if isinstance(node, ast.Name) and node.id == parameter_name:
                # Check if this is a parameter usage (not assignment)
                if isinstance(node.ctx, ast.Load):
                    return True
                    
        return False
        
    def find_unused_parameters(self) -> List[Tuple[str, str]]:
        """
        Find parameters that are defined but never used in their function bodies.
        
        Returns:
            List of (function_name, parameter_name) tuples for unused parameters
        """
        unused_params = []
        
        # Get all functions
        functions = self.ast_analyzer.get_functions()
        
        # Check each function's parameters
        for func_name, func_info in functions.items():
            for param in func_info.parameters:
                # Skip *args and **kwargs
                if param.get('is_vararg') or param.get('is_kwarg'):
                    continue
                    
                # Check if parameter is used
                if not self.is_parameter_used(func_name, param['name']):
                    unused_params.append((func_name, param['name']))
                    
        return unused_params
        
    def analyze_parameter_types(self, function_name: str) -> Dict[str, Dict[str, Any]]:
        """
        Analyze the types of values passed to a function's parameters.
        
        Args:
            function_name: Name of the function to analyze
            
        Returns:
            Dictionary mapping parameter names to type information
        """
        type_info = {}
        
        # Get function calls
        call_matches = self.find_function_calls(function_name)
        
        # Analyze each parameter
        for match in call_matches:
            for param_name, arg_node in match.param_matches.items():
                if param_name not in type_info:
                    type_info[param_name] = {
                        'types': set(),
                        'literals': [],
                        'complex': False
                    }
                    
                # Try to determine the arg type
                if isinstance(arg_node, ast.Constant):
                    # Direct constant value
                    arg_type = type(arg_node.value).__name__
                    type_info[param_name]['types'].add(arg_type)
                    type_info[param_name]['literals'].append(arg_node.value)
                elif isinstance(arg_node, ast.List):
                    type_info[param_name]['types'].add('list')
                    type_info[param_name]['complex'] = True
                elif isinstance(arg_node, ast.Dict):
                    type_info[param_name]['types'].add('dict')
                    type_info[param_name]['complex'] = True
                elif isinstance(arg_node, ast.Set):
                    type_info[param_name]['types'].add('set')
                    type_info[param_name]['complex'] = True
                elif isinstance(arg_node, ast.Name):
                    # Variable name, try to find its type from variable info
                    var_info = self.ast_analyzer.get_variables().get(arg_node.id)
                    if var_info and var_info.value_type:
                        type_info[param_name]['types'].add(var_info.value_type)
                    else:
                        type_info[param_name]['types'].add('unknown')
                else:
                    type_info[param_name]['types'].add('unknown')
                    type_info[param_name]['complex'] = True
                    
        return type_info
        
    def generate_parameter_checks(self, function_name: str) -> List[Dict[str, Any]]:
        """
        Generate parameter validation checks for a function.
        
        Args:
            function_name: Name of the function to generate checks for
            
        Returns:
            List of patch dictionaries for parameter checks
        """
        # Get function info
        function_info = self.ast_analyzer.get_function_info(function_name)
        if not function_info:
            return []
            
        patches = []
        
        # Analyze parameter types
        param_types = self.analyze_parameter_types(function_name)
        
        # Generate checks for each parameter
        for param in function_info.parameters:
            param_name = param['name']
            
            # Skip *args and **kwargs
            if param.get('is_vararg') or param.get('is_kwarg'):
                continue
                
            # Skip parameters with default values (they're already handled)
            if 'default' in param and param['default'] is not None:
                continue
                
            # Get parameter type information
            type_info = param_types.get(param_name, {})
            types = type_info.get('types', set())
            
            # Get parameter annotation if available
            annotation = param.get('annotation')
            
            # Determine which check to generate based on annotations and usage
            if annotation in ('str', 'String', 'str', 'Text'):
                check_code = f"if {param_name} is None or not isinstance({param_name}, str):"
                default_value = "''"
            elif annotation in ('int', 'Int', 'float', 'Float', 'number', 'Number'):
                check_code = f"if {param_name} is None or not isinstance({param_name}, (int, float)):"
                default_value = "0"
            elif annotation in ('list', 'List', 'sequence', 'Sequence'):
                check_code = f"if {param_name} is None or not hasattr({param_name}, '__iter__'):"
                default_value = "[]"
            elif annotation in ('dict', 'Dict', 'mapping', 'Mapping'):
                check_code = f"if {param_name} is None or not hasattr({param_name}, 'items'):"
                default_value = "{}"
            elif 'str' in types or 'unicode' in types:
                check_code = f"if {param_name} is None or not isinstance({param_name}, str):"
                default_value = "''"
            elif 'int' in types or 'float' in types:
                check_code = f"if {param_name} is None or not isinstance({param_name}, (int, float)):"
                default_value = "0"
            elif 'list' in types or 'tuple' in types:
                check_code = f"if {param_name} is None or not hasattr({param_name}, '__iter__'):"
                default_value = "[]"
            elif 'dict' in types:
                check_code = f"if {param_name} is None or not hasattr({param_name}, 'items'):"
                default_value = "{}"
            else:
                # Generic check
                check_code = f"if {param_name} is None:"
                default_value = "None"
                
            # Create patch dictionary
            patch = {
                "function_name": function_name,
                "parameter_name": param_name,
                "check_code": check_code,
                "default_value": default_value,
                "annotation": annotation or "Any",
                "line_number": function_info.node.lineno,
                "patch_type": "parameter_check"
            }
            
            patches.append(patch)
            
        return patches


if __name__ == "__main__":
    # Example usage
    test_code = """
def process_data(data_list, transform_func=None, options=None):
    \"\"\"
    Process the data with optional transformation.
    
    Args:
        data_list: List of data items to process
        transform_func: Optional function to transform the data
        options: Options for processing
    
    Returns:
        Processed data
    \"\"\"
    result = []
    
    # Check options (but options might be unused)
    if options is not None:
        print(f"Processing with options: {options}")
    
    # Process each item
    for item in data_list:
        # Transform the item if a transform function is provided
        if transform_func:
            item = transform_func(item)
        
        # Add to results
        result.append(item)
    
    return result

def double(x):
    return x * 2

# Various function calls with potential issues
process_data([1, 2, 3])  # Missing optional parameters
process_data([1, 2, 3], double)  # Correct call
process_data(transform_func=double, data_list=[4, 5, 6])  # Keyword arguments
process_data()  # Missing required parameter
process_data([1, 2, 3], double, {"verbose": True}, extra_arg="test")  # Too many arguments
"""
    
    # Create and use the function signature analyzer
    analyzer = FunctionSignatureAnalyzer()
    if analyzer.analyze_code(test_code):
        # Find all parameter issues
        issues = analyzer.find_all_parameter_issues()
        
        print(f"Found {len(issues)} parameter issues:")
        for issue in issues:
            print(f"- {issue.function_name}, {issue.parameter_name}: {issue.issue_type} (line {issue.line_number})")
            if issue.suggestion:
                print(f"  Suggestion: {issue.suggestion}")
                
        # Find unused parameters
        unused = analyzer.find_unused_parameters()
        if unused:
            print("\nUnused parameters:")
            for func_name, param_name in unused:
                print(f"- {func_name}.{param_name}")
                
        # Generate parameter checks
        checks = analyzer.generate_parameter_checks("process_data")
        
        if checks:
            print("\nGenerated parameter checks:")
            for check in checks:
                print(f"- {check['parameter_name']}: {check['check_code']}")
                print(f"  Default: {check['default_value']}")
                
        # Analyze parameter types
        types = analyzer.analyze_parameter_types("process_data")
        
        print("\nParameter type analysis:")
        for param_name, type_info in types.items():
            types_str = ", ".join(sorted(type_info['types'])) or "unknown"
            print(f"- {param_name}: {types_str}")
            if type_info.get('literals'):
                print(f"  Example values: {type_info['literals'][:3]}")