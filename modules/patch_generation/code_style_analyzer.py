#!/usr/bin/env python3
"""
Code Style Analyzer and Formatter

This module analyzes existing code style and formatting patterns to ensure
LLM-generated patches conform to the project's established conventions.
"""

import ast
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass
from collections import Counter, defaultdict


logger = logging.getLogger(__name__)


@dataclass
class StyleConventions:
    """Detected code style conventions."""
    indent_style: str  # 'spaces' or 'tabs'
    indent_size: int
    quote_style: str  # 'single', 'double', or 'mixed'
    line_length: int
    naming_conventions: Dict[str, str]
    spacing_patterns: Dict[str, Any]
    import_style: Dict[str, Any]
    comment_style: Dict[str, Any]
    architectural_patterns: List[str]
    confidence: float


class CodeStyleAnalyzer:
    """
    Analyzes code style and formatting patterns from existing codebase.
    
    This analyzer can detect:
    1. Indentation style (spaces vs tabs, size)
    2. Quote preferences (single vs double)
    3. Line length preferences
    4. Naming conventions
    5. Spacing around operators and brackets
    6. Import organization patterns
    7. Comment styles and patterns
    8. Architectural patterns (decorators, inheritance, etc.)
    """

    def __init__(self):
        """Initialize the style analyzer."""
        self.language_analyzers = {
            'python': self._analyze_python_style,
            'javascript': self._analyze_javascript_style,
            'typescript': self._analyze_typescript_style,
            'java': self._analyze_java_style,
            'go': self._analyze_go_style,
            'rust': self._analyze_rust_style,
            'swift': self._analyze_swift_style,
        }

    def analyze_file_style(self, 
                          file_path: str, 
                          language: str,
                          source_code: Optional[str] = None) -> StyleConventions:
        """
        Analyze the style conventions of a specific file.
        
        Args:
            file_path: Path to the file
            language: Programming language
            source_code: Source code content (if not provided, will read from file)
            
        Returns:
            Detected style conventions
        """
        if source_code is None:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    source_code = f.read()
            except Exception as e:
                logger.error(f"Could not read file {file_path}: {e}")
                return self._get_default_conventions(language)
        
        if not source_code.strip():
            return self._get_default_conventions(language)
        
        # Use language-specific analyzer
        analyzer = self.language_analyzers.get(language, self._analyze_generic_style)
        return analyzer(source_code, file_path)

    def analyze_project_style(self, 
                             project_root: str, 
                             language: str,
                             max_files: int = 20) -> StyleConventions:
        """
        Analyze style conventions across multiple files in a project.
        
        Args:
            project_root: Root directory of the project
            language: Programming language
            max_files: Maximum number of files to analyze
            
        Returns:
            Aggregated style conventions
        """
        file_extensions = self._get_file_extensions(language)
        project_path = Path(project_root)
        
        # Find relevant files
        source_files = []
        for ext in file_extensions:
            pattern = f"**/*{ext}"
            found_files = list(project_path.glob(pattern))
            source_files.extend(found_files[:max_files // len(file_extensions)])
        
        if not source_files:
            return self._get_default_conventions(language)
        
        # Analyze each file
        file_conventions = []
        for file_path in source_files[:max_files]:
            try:
                conventions = self.analyze_file_style(str(file_path), language)
                if conventions.confidence > 0.3:  # Only include high-confidence results
                    file_conventions.append(conventions)
            except Exception as e:
                logger.warning(f"Could not analyze {file_path}: {e}")
        
        if not file_conventions:
            return self._get_default_conventions(language)
        
        # Aggregate conventions
        return self._aggregate_conventions(file_conventions, language)

    def _analyze_python_style(self, source_code: str, file_path: str) -> StyleConventions:
        """Analyze Python-specific style conventions."""
        lines = source_code.split('\n')
        
        # Analyze indentation
        indent_style, indent_size = self._analyze_indentation(lines)
        
        # Analyze quotes
        quote_style = self._analyze_python_quotes(source_code)
        
        # Analyze line length
        line_length = self._analyze_line_length(lines)
        
        # Analyze naming conventions
        naming_conventions = self._analyze_python_naming(source_code)
        
        # Analyze spacing patterns
        spacing_patterns = self._analyze_python_spacing(lines)
        
        # Analyze import style
        import_style = self._analyze_python_imports(lines)
        
        # Analyze comment style
        comment_style = self._analyze_python_comments(lines)
        
        # Analyze architectural patterns
        architectural_patterns = self._analyze_python_architecture(source_code)
        
        # Calculate overall confidence
        confidence = self._calculate_confidence(source_code, 'python')
        
        return StyleConventions(
            indent_style=indent_style,
            indent_size=indent_size,
            quote_style=quote_style,
            line_length=line_length,
            naming_conventions=naming_conventions,
            spacing_patterns=spacing_patterns,
            import_style=import_style,
            comment_style=comment_style,
            architectural_patterns=architectural_patterns,
            confidence=confidence
        )

    def _analyze_javascript_style(self, source_code: str, file_path: str) -> StyleConventions:
        """Analyze JavaScript-specific style conventions."""
        lines = source_code.split('\n')
        
        # Analyze indentation
        indent_style, indent_size = self._analyze_indentation(lines)
        
        # Analyze quotes
        quote_style = self._analyze_javascript_quotes(source_code)
        
        # Analyze line length
        line_length = self._analyze_line_length(lines)
        
        # Analyze naming conventions
        naming_conventions = self._analyze_javascript_naming(source_code)
        
        # Analyze spacing patterns
        spacing_patterns = self._analyze_javascript_spacing(lines)
        
        # Analyze import style
        import_style = self._analyze_javascript_imports(lines)
        
        # Analyze comment style
        comment_style = self._analyze_javascript_comments(lines)
        
        # Analyze architectural patterns
        architectural_patterns = self._analyze_javascript_architecture(source_code)
        
        # Calculate confidence
        confidence = self._calculate_confidence(source_code, 'javascript')
        
        return StyleConventions(
            indent_style=indent_style,
            indent_size=indent_size,
            quote_style=quote_style,
            line_length=line_length,
            naming_conventions=naming_conventions,
            spacing_patterns=spacing_patterns,
            import_style=import_style,
            comment_style=comment_style,
            architectural_patterns=architectural_patterns,
            confidence=confidence
        )

    def _analyze_typescript_style(self, source_code: str, file_path: str) -> StyleConventions:
        """
        Analyze TypeScript code style conventions.
        
        TypeScript analysis includes all JavaScript patterns plus:
        - Type annotation style (inline vs separate)
        - Interface vs type alias preferences
        - Enum style conventions
        - Decorator usage patterns
        - Module/namespace conventions
        """
        # First perform JavaScript analysis as baseline
        js_conventions = self._analyze_javascript_style(source_code, file_path)
        
        lines = source_code.split('\n')
        
        # TypeScript-specific analysis
        type_annotation_style = self._analyze_typescript_type_annotations(source_code)
        interface_style = self._analyze_typescript_interfaces(source_code)
        enum_style = self._analyze_typescript_enums(source_code)
        decorator_style = self._analyze_typescript_decorators(source_code)
        module_style = self._analyze_typescript_modules(source_code)
        
        # Merge with JavaScript conventions
        architectural_patterns = js_conventions.architectural_patterns + [
            pattern for pattern in [
                f"type_annotations:{type_annotation_style}" if type_annotation_style else None,
                f"interface_style:{interface_style}" if interface_style else None,
                f"enum_style:{enum_style}" if enum_style else None,
                f"decorator_usage:{decorator_style}" if decorator_style else None,
                f"module_style:{module_style}" if module_style else None
            ] if pattern
        ]
        
        return StyleConventions(
            indent_style=js_conventions.indent_style,
            indent_size=js_conventions.indent_size,
            quote_style=js_conventions.quote_style,
            line_length=js_conventions.line_length,
            naming_conventions=self._merge_naming_conventions(
                js_conventions.naming_conventions,
                self._analyze_typescript_naming(source_code)
            ),
            spacing_patterns=js_conventions.spacing_patterns,
            import_style=js_conventions.import_style,
            comment_style=js_conventions.comment_style,
            architectural_patterns=architectural_patterns,
            confidence=min(js_conventions.confidence + 0.1, 1.0)  # Slightly higher confidence for TS
        )

    def _analyze_typescript_type_annotations(self, source_code: str) -> str:
        """Analyze TypeScript type annotation style."""
        inline_count = len(re.findall(r':\s*\w+(?:<[^>]+>)?(?:\[\])?', source_code))
        separate_type_count = len(re.findall(r'^type\s+\w+\s*=', source_code, re.MULTILINE))
        
        if inline_count > separate_type_count * 2:
            return "inline"
        elif separate_type_count > inline_count:
            return "separate_types"
        return "mixed"

    def _analyze_typescript_interfaces(self, source_code: str) -> str:
        """Analyze TypeScript interface vs type alias preferences."""
        interface_count = len(re.findall(r'^interface\s+\w+', source_code, re.MULTILINE))
        type_alias_count = len(re.findall(r'^type\s+\w+\s*=\s*\{', source_code, re.MULTILINE))
        
        if interface_count > type_alias_count * 2:
            return "interface_preferred"
        elif type_alias_count > interface_count * 2:
            return "type_alias_preferred"
        return "mixed"

    def _analyze_typescript_enums(self, source_code: str) -> str:
        """Analyze TypeScript enum style conventions."""
        const_enum_count = len(re.findall(r'const\s+enum\s+\w+', source_code))
        regular_enum_count = len(re.findall(r'(?<!const\s)enum\s+\w+', source_code))
        union_type_count = len(re.findall(r'type\s+\w+\s*=\s*[\'"][^\'"]+"?\s*\|', source_code))
        
        if const_enum_count > regular_enum_count:
            return "const_enum"
        elif regular_enum_count > 0:
            return "regular_enum"
        elif union_type_count > 0:
            return "string_literal_union"
        return None

    def _analyze_typescript_decorators(self, source_code: str) -> str:
        """Analyze TypeScript decorator usage patterns."""
        decorator_count = len(re.findall(r'@\w+(?:\([^)]*\))?', source_code))
        if decorator_count > 10:
            return "heavy"
        elif decorator_count > 0:
            return "moderate"
        return "none"

    def _analyze_typescript_modules(self, source_code: str) -> str:
        """Analyze TypeScript module/namespace conventions."""
        namespace_count = len(re.findall(r'namespace\s+\w+', source_code))
        module_count = len(re.findall(r'module\s+\w+', source_code))
        
        if namespace_count > 0:
            return "namespace"
        elif module_count > 0:
            return "module"
        return "es6_modules"

    def _analyze_typescript_naming(self, source_code: str) -> Dict[str, str]:
        """Analyze TypeScript-specific naming conventions."""
        conventions = {}
        
        # Interface naming (I prefix or not)
        interfaces = re.findall(r'interface\s+(\w+)', source_code)
        if interfaces:
            i_prefixed = sum(1 for name in interfaces if name.startswith('I') and len(name) > 1 and name[1].isupper())
            conventions['interfaces'] = 'i_prefix' if i_prefixed > len(interfaces) / 2 else 'pascal_case'
        
        # Type alias naming
        type_aliases = re.findall(r'type\s+(\w+)\s*=', source_code)
        if type_aliases:
            pascal_count = sum(1 for name in type_aliases if name[0].isupper())
            conventions['type_aliases'] = 'pascal_case' if pascal_count > len(type_aliases) / 2 else 'camel_case'
        
        # Enum naming
        enums = re.findall(r'enum\s+(\w+)', source_code)
        if enums:
            pascal_count = sum(1 for name in enums if name[0].isupper())
            conventions['enums'] = 'pascal_case' if pascal_count > len(enums) / 2 else 'camel_case'
        
        return conventions

    def _merge_naming_conventions(self, base: Dict[str, str], additional: Dict[str, str]) -> Dict[str, str]:
        """Merge two naming convention dictionaries."""
        merged = base.copy()
        merged.update(additional)
        return merged

    def _analyze_java_style(self, source_code: str, file_path: str) -> StyleConventions:
        """Analyze Java code style conventions."""
        # For now, return generic style
        # TODO: Implement full Java style analysis
        return self._analyze_generic_style(source_code, file_path)

    def _analyze_go_style(self, source_code: str, file_path: str) -> StyleConventions:
        """Analyze Go code style conventions."""
        # For now, return generic style  
        # TODO: Implement full Go style analysis
        return self._analyze_generic_style(source_code, file_path)

    def _analyze_rust_style(self, source_code: str, file_path: str) -> StyleConventions:
        """Analyze Rust code style conventions."""
        # For now, return generic style
        # TODO: Implement full Rust style analysis
        return self._analyze_generic_style(source_code, file_path)

    def _analyze_swift_style(self, source_code: str, file_path: str) -> StyleConventions:
        """Analyze Swift code style conventions."""
        # For now, return generic style
        # TODO: Implement full Swift style analysis
        return self._analyze_generic_style(source_code, file_path)

    def _analyze_generic_style(self, source_code: str, file_path: str) -> StyleConventions:
        """Generic style analysis for unsupported languages."""
        lines = source_code.split('\n')
        
        # Basic analysis
        indent_style, indent_size = self._analyze_indentation(lines)
        line_length = self._analyze_line_length(lines)
        
        return StyleConventions(
            indent_style=indent_style,
            indent_size=indent_size,
            quote_style='double',  # Default
            line_length=line_length,
            naming_conventions={},
            spacing_patterns={},
            import_style={},
            comment_style={},
            architectural_patterns=[],
            confidence=0.5
        )

    def _analyze_indentation(self, lines: List[str]) -> Tuple[str, int]:
        """Analyze indentation style and size."""
        space_indents = []
        tab_indents = []
        
        for line in lines:
            if line.strip():  # Skip empty lines
                leading_spaces = len(line) - len(line.lstrip(' '))
                leading_tabs = len(line) - len(line.lstrip('\t'))
                
                if leading_spaces > 0 and leading_tabs == 0:
                    space_indents.append(leading_spaces)
                elif leading_tabs > 0 and leading_spaces == 0:
                    tab_indents.append(leading_tabs)
        
        # Determine style
        if len(space_indents) > len(tab_indents):
            indent_style = 'spaces'
            # Find most common indentation level
            if space_indents:
                # Filter out 0 indentation and find GCD-like pattern
                non_zero_indents = [i for i in space_indents if i > 0]
                if non_zero_indents:
                    # Find the most likely indent size
                    min_indent = min(non_zero_indents)
                    possible_sizes = [2, 4, 8]
                    indent_size = min([s for s in possible_sizes if min_indent % s == 0], default=4)
                else:
                    indent_size = 4
            else:
                indent_size = 4
        elif len(tab_indents) > 0:
            indent_style = 'tabs'
            indent_size = 1  # Tabs are typically 1 tab = 1 level
        else:
            # Default
            indent_style = 'spaces'
            indent_size = 4
        
        return indent_style, indent_size

    def _analyze_python_quotes(self, source_code: str) -> str:
        """Analyze Python quote style preference."""
        single_quotes = len(re.findall(r"'[^']*'", source_code))
        double_quotes = len(re.findall(r'"[^"]*"', source_code))
        
        if single_quotes > double_quotes * 1.5:
            return 'single'
        elif double_quotes > single_quotes * 1.5:
            return 'double'
        else:
            return 'mixed'

    def _analyze_javascript_quotes(self, source_code: str) -> str:
        """Analyze JavaScript quote style preference."""
        single_quotes = len(re.findall(r"'[^']*'", source_code))
        double_quotes = len(re.findall(r'"[^"]*"', source_code))
        template_literals = len(re.findall(r'`[^`]*`', source_code))
        
        total = single_quotes + double_quotes + template_literals
        if total == 0:
            return 'single'  # Default for JS
        
        if single_quotes / total > 0.6:
            return 'single'
        elif double_quotes / total > 0.6:
            return 'double'
        else:
            return 'mixed'

    def _analyze_line_length(self, lines: List[str]) -> int:
        """Analyze preferred line length."""
        non_empty_lines = [line for line in lines if line.strip()]
        if not non_empty_lines:
            return 80
        
        line_lengths = [len(line) for line in non_empty_lines]
        
        # Find the 90th percentile as a reasonable max line length
        line_lengths.sort()
        percentile_90 = line_lengths[int(len(line_lengths) * 0.9)]
        
        # Round to common values
        common_lengths = [80, 100, 120, 140]
        for length in common_lengths:
            if percentile_90 <= length:
                return length
        
        return 120  # Default

    def _analyze_python_naming(self, source_code: str) -> Dict[str, str]:
        """Analyze Python naming conventions."""
        conventions = {}
        
        try:
            tree = ast.parse(source_code)
            
            # Analyze function names
            function_names = []
            class_names = []
            variable_names = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    function_names.append(node.name)
                elif isinstance(node, ast.ClassDef):
                    class_names.append(node.name)
                elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    variable_names.append(node.id)
            
            # Determine naming patterns
            if function_names:
                if all(self._is_snake_case(name) for name in function_names):
                    conventions['functions'] = 'snake_case'
                elif all(self._is_camel_case(name) for name in function_names):
                    conventions['functions'] = 'camelCase'
                else:
                    conventions['functions'] = 'mixed'
            
            if class_names:
                if all(self._is_pascal_case(name) for name in class_names):
                    conventions['classes'] = 'PascalCase'
                else:
                    conventions['classes'] = 'mixed'
            
            if variable_names:
                snake_case_vars = sum(1 for name in variable_names if self._is_snake_case(name))
                if snake_case_vars / len(variable_names) > 0.8:
                    conventions['variables'] = 'snake_case'
                else:
                    conventions['variables'] = 'mixed'
        
        except Exception as e:
            logger.warning(f"Could not parse Python AST: {e}")
        
        return conventions

    def _analyze_javascript_naming(self, source_code: str) -> Dict[str, str]:
        """Analyze JavaScript naming conventions."""
        conventions = {}
        
        # Function declarations
        function_pattern = r'function\s+([a-zA-Z_$][a-zA-Z0-9_$]*)'
        function_names = re.findall(function_pattern, source_code)
        
        # Variable declarations
        var_pattern = r'(?:const|let|var)\s+([a-zA-Z_$][a-zA-Z0-9_$]*)'
        variable_names = re.findall(var_pattern, source_code)
        
        # Class declarations
        class_pattern = r'class\s+([a-zA-Z_$][a-zA-Z0-9_$]*)'
        class_names = re.findall(class_pattern, source_code)
        
        # Analyze patterns
        if function_names:
            camel_case_functions = sum(1 for name in function_names if self._is_camel_case(name))
            if camel_case_functions / len(function_names) > 0.8:
                conventions['functions'] = 'camelCase'
            else:
                conventions['functions'] = 'mixed'
        
        if variable_names:
            camel_case_vars = sum(1 for name in variable_names if self._is_camel_case(name))
            if camel_case_vars / len(variable_names) > 0.8:
                conventions['variables'] = 'camelCase'
            else:
                conventions['variables'] = 'mixed'
        
        if class_names:
            pascal_case_classes = sum(1 for name in class_names if self._is_pascal_case(name))
            if pascal_case_classes / len(class_names) > 0.8:
                conventions['classes'] = 'PascalCase'
            else:
                conventions['classes'] = 'mixed'
        
        return conventions

    def _analyze_python_spacing(self, lines: List[str]) -> Dict[str, Any]:
        """Analyze Python spacing patterns."""
        patterns = {}
        
        # Analyze spacing around operators
        operator_spacing = defaultdict(int)
        for line in lines:
            # Check for spaces around = operator
            if ' = ' in line:
                operator_spacing['equals_spaced'] += 1
            elif '=' in line and ' = ' not in line:
                operator_spacing['equals_no_space'] += 1
        
        if operator_spacing:
            total = sum(operator_spacing.values())
            patterns['operator_spacing'] = max(operator_spacing.items(), key=lambda x: x[1])[0]
        
        return patterns

    def _analyze_javascript_spacing(self, lines: List[str]) -> Dict[str, Any]:
        """Analyze JavaScript spacing patterns."""
        patterns = {}
        
        # Analyze spacing around operators and braces
        brace_spacing = defaultdict(int)
        for line in lines:
            # Check for spaces in object literals
            if '{ ' in line and ' }' in line:
                brace_spacing['object_spaced'] += 1
            elif '{' in line and '}' in line and '{ ' not in line:
                brace_spacing['object_no_space'] += 1
        
        if brace_spacing:
            patterns['brace_spacing'] = max(brace_spacing.items(), key=lambda x: x[1])[0]
        
        return patterns

    def _analyze_python_imports(self, lines: List[str]) -> Dict[str, Any]:
        """Analyze Python import style."""
        imports = {}
        
        import_lines = [line.strip() for line in lines if line.strip().startswith(('import ', 'from '))]
        
        if import_lines:
            # Check for import grouping
            has_blank_lines = any(line == '' for line in lines[:20])  # Check first 20 lines
            imports['groups_imports'] = has_blank_lines
            
            # Check for relative imports
            relative_imports = sum(1 for line in import_lines if line.startswith('from .'))
            imports['uses_relative_imports'] = relative_imports > 0
        
        return imports

    def _analyze_javascript_imports(self, lines: List[str]) -> Dict[str, Any]:
        """Analyze JavaScript import style."""
        imports = {}
        
        import_lines = [line.strip() for line in lines if 'import' in line or 'require' in line]
        
        if import_lines:
            # Check ES6 vs CommonJS
            es6_imports = sum(1 for line in import_lines if line.startswith('import'))
            commonjs_imports = sum(1 for line in import_lines if 'require(' in line)
            
            if es6_imports > commonjs_imports:
                imports['style'] = 'es6'
            elif commonjs_imports > 0:
                imports['style'] = 'commonjs'
            else:
                imports['style'] = 'mixed'
        
        return imports

    def _analyze_python_comments(self, lines: List[str]) -> Dict[str, Any]:
        """Analyze Python comment style."""
        comments = {}
        
        comment_lines = [line for line in lines if line.strip().startswith('#')]
        
        if comment_lines:
            # Check for space after #
            spaced_comments = sum(1 for line in comment_lines if line.strip().startswith('# '))
            comments['space_after_hash'] = spaced_comments / len(comment_lines) > 0.8
        
        return comments

    def _analyze_javascript_comments(self, lines: List[str]) -> Dict[str, Any]:
        """Analyze JavaScript comment style."""
        comments = {}
        
        single_line_comments = [line for line in lines if '//' in line]
        
        if single_line_comments:
            # Check for space after //
            spaced_comments = sum(1 for line in single_line_comments if '// ' in line)
            comments['space_after_slash'] = spaced_comments / len(single_line_comments) > 0.8
        
        return comments

    def _analyze_python_architecture(self, source_code: str) -> List[str]:
        """Analyze Python architectural patterns."""
        patterns = []
        
        # Check for decorators
        if '@' in source_code:
            patterns.append('decorators')
        
        # Check for context managers
        if 'with ' in source_code:
            patterns.append('context_managers')
        
        # Check for list comprehensions
        if '[' in source_code and 'for' in source_code and 'in' in source_code:
            # Simple heuristic for list comprehensions
            if re.search(r'\[[^]]*\s+for\s+[^]]*\s+in\s+[^]]*\]', source_code):
                patterns.append('list_comprehensions')
        
        return patterns

    def _analyze_javascript_architecture(self, source_code: str) -> List[str]:
        """Analyze JavaScript architectural patterns."""
        patterns = []
        
        # Check for arrow functions
        if '=>' in source_code:
            patterns.append('arrow_functions')
        
        # Check for async/await
        if 'async' in source_code and 'await' in source_code:
            patterns.append('async_await')
        
        # Check for destructuring
        if re.search(r'const\s*{\s*\w+', source_code):
            patterns.append('destructuring')
        
        return patterns

    def _is_snake_case(self, name: str) -> bool:
        """Check if name follows snake_case convention."""
        return re.match(r'^[a-z_][a-z0-9_]*$', name) is not None

    def _is_camel_case(self, name: str) -> bool:
        """Check if name follows camelCase convention."""
        return re.match(r'^[a-z][a-zA-Z0-9]*$', name) is not None

    def _is_pascal_case(self, name: str) -> bool:
        """Check if name follows PascalCase convention."""
        return re.match(r'^[A-Z][a-zA-Z0-9]*$', name) is not None

    def _calculate_confidence(self, source_code: str, language: str) -> float:
        """Calculate confidence score for style analysis."""
        lines = source_code.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        if len(non_empty_lines) < 10:
            return 0.3  # Low confidence for short files
        elif len(non_empty_lines) < 50:
            return 0.7  # Medium confidence
        else:
            return 0.9  # High confidence for larger files

    def _get_file_extensions(self, language: str) -> List[str]:
        """Get file extensions for a language."""
        extension_map = {
            'python': ['.py'],
            'javascript': ['.js', '.jsx'],
            'typescript': ['.ts', '.tsx'],
            'java': ['.java'],
            'go': ['.go'],
            'rust': ['.rs'],
            'swift': ['.swift'],
        }
        
        return extension_map.get(language, [])

    def _get_default_conventions(self, language: str) -> StyleConventions:
        """Get default style conventions for a language."""
        defaults = {
            'python': StyleConventions(
                indent_style='spaces', indent_size=4, quote_style='single',
                line_length=88, naming_conventions={'functions': 'snake_case', 'classes': 'PascalCase'},
                spacing_patterns={}, import_style={}, comment_style={},
                architectural_patterns=[], confidence=0.5
            ),
            'javascript': StyleConventions(
                indent_style='spaces', indent_size=2, quote_style='single',
                line_length=80, naming_conventions={'functions': 'camelCase', 'classes': 'PascalCase'},
                spacing_patterns={}, import_style={}, comment_style={},
                architectural_patterns=[], confidence=0.5
            ),
            'java': StyleConventions(
                indent_style='spaces', indent_size=4, quote_style='double',
                line_length=100, naming_conventions={'functions': 'camelCase', 'classes': 'PascalCase'},
                spacing_patterns={}, import_style={}, comment_style={},
                architectural_patterns=[], confidence=0.5
            ),
        }
        
        return defaults.get(language, StyleConventions(
            indent_style='spaces', indent_size=4, quote_style='double',
            line_length=80, naming_conventions={}, spacing_patterns={},
            import_style={}, comment_style={}, architectural_patterns=[],
            confidence=0.5
        ))

    def _aggregate_conventions(self, conventions_list: List[StyleConventions], language: str) -> StyleConventions:
        """Aggregate conventions from multiple files."""
        if not conventions_list:
            return self._get_default_conventions(language)
        
        if len(conventions_list) == 1:
            return conventions_list[0]
        
        # Aggregate by taking most common values
        indent_styles = [c.indent_style for c in conventions_list]
        indent_sizes = [c.indent_size for c in conventions_list]
        quote_styles = [c.quote_style for c in conventions_list]
        line_lengths = [c.line_length for c in conventions_list]
        
        # Use Counter to find most common values
        most_common_indent_style = Counter(indent_styles).most_common(1)[0][0]
        most_common_indent_size = Counter(indent_sizes).most_common(1)[0][0]
        most_common_quote_style = Counter(quote_styles).most_common(1)[0][0]
        avg_line_length = int(sum(line_lengths) / len(line_lengths))
        
        # Aggregate naming conventions
        aggregated_naming = {}
        for key in ['functions', 'classes', 'variables']:
            values = [c.naming_conventions.get(key) for c in conventions_list if c.naming_conventions.get(key)]
            if values:
                aggregated_naming[key] = Counter(values).most_common(1)[0][0]
        
        # Calculate aggregate confidence
        avg_confidence = sum(c.confidence for c in conventions_list) / len(conventions_list)
        
        return StyleConventions(
            indent_style=most_common_indent_style,
            indent_size=most_common_indent_size,
            quote_style=most_common_quote_style,
            line_length=avg_line_length,
            naming_conventions=aggregated_naming,
            spacing_patterns={},  # Could be aggregated similarly
            import_style={},      # Could be aggregated similarly
            comment_style={},     # Could be aggregated similarly
            architectural_patterns=[], # Could be aggregated similarly
            confidence=avg_confidence
        )

    def format_code_to_style(self, 
                           code: str, 
                           conventions: StyleConventions, 
                           language: str) -> str:
        """
        Format code according to detected style conventions.
        
        Args:
            code: Code to format
            conventions: Detected style conventions
            language: Programming language
            
        Returns:
            Formatted code
        """
        # Apply basic formatting
        formatted = self._apply_indentation(code, conventions)
        formatted = self._apply_line_length(formatted, conventions)
        
        # Apply language-specific formatting
        if language == 'python':
            formatted = self._apply_python_formatting(formatted, conventions)
        elif language in ['javascript', 'typescript']:
            formatted = self._apply_javascript_formatting(formatted, conventions)
        
        return formatted

    def _apply_indentation(self, code: str, conventions: StyleConventions) -> str:
        """Apply consistent indentation."""
        lines = code.split('\n')
        formatted_lines = []
        
        for line in lines:
            stripped = line.lstrip()
            if not stripped:
                formatted_lines.append('')
                continue
            
            # Count current indentation level
            current_indent = len(line) - len(stripped)
            
            # Apply consistent indentation
            if conventions.indent_style == 'spaces':
                # Assume each 4 spaces or tab = 1 level, adjust to desired size
                indent_level = current_indent // 4  # Rough estimation
                new_indent = ' ' * (indent_level * conventions.indent_size)
            else:
                # Use tabs
                indent_level = current_indent // 4
                new_indent = '\t' * indent_level
            
            formatted_lines.append(new_indent + stripped)
        
        return '\n'.join(formatted_lines)

    def _apply_line_length(self, code: str, conventions: StyleConventions) -> str:
        """Apply line length constraints (basic implementation)."""
        # This is a simplified implementation
        # A full implementation would need language-aware line breaking
        return code

    def _apply_python_formatting(self, code: str, conventions: StyleConventions) -> str:
        """Apply Python-specific formatting."""
        if conventions.quote_style == 'single':
            # Convert double quotes to single (simplified)
            code = re.sub(r'"([^"]*)"', r"'\1'", code)
        elif conventions.quote_style == 'double':
            # Convert single quotes to double (simplified)
            code = re.sub(r"'([^']*)'", r'"\1"', code)
        
        return code

    def _apply_javascript_formatting(self, code: str, conventions: StyleConventions) -> str:
        """Apply JavaScript-specific formatting."""
        if conventions.quote_style == 'single':
            # Convert double quotes to single (simplified)
            code = re.sub(r'"([^"]*)"', r"'\1'", code)
        elif conventions.quote_style == 'double':
            # Convert single quotes to double (simplified)
            code = re.sub(r"'([^']*)'", r'"\1"', code)
        
        return code


def create_code_style_analyzer() -> CodeStyleAnalyzer:
    """Factory function to create a code style analyzer."""
    return CodeStyleAnalyzer()


if __name__ == "__main__":
    # Test the style analyzer
    print("Testing Code Style Analyzer")
    print("===========================")
    
    analyzer = create_code_style_analyzer()
    
    # Test Python code
    python_code = '''
import os
from django.db import models

class BlogPost(models.Model):
    title = models.CharField(max_length=200)
    content = models.TextField()
    
    def __str__(self):
        return self.title
    
    def get_absolute_url(self):
        return f"/posts/{self.id}/"
'''
    
    conventions = analyzer.analyze_file_style("test.py", "python", python_code)
    print(f"Python conventions:")
    print(f"  Indent: {conventions.indent_style} ({conventions.indent_size})")
    print(f"  Quotes: {conventions.quote_style}")
    print(f"  Line length: {conventions.line_length}")
    print(f"  Naming: {conventions.naming_conventions}")
    print(f"  Confidence: {conventions.confidence:.2f}")
    
    # Test formatting
    test_code = '''
def test_function( ):
    x=1+2
    return x
'''
    
    formatted = analyzer.format_code_to_style(test_code, conventions, "python")
    print(f"\nFormatted code:\n{formatted}")