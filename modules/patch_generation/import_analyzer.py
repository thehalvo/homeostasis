"""
Context gathering for imported modules and dependencies.

This module provides utilities for analyzing imports, resolving dependencies, and
gathering context from imported modules to improve patch generation.
"""

import importlib
import inspect
import logging
import os
import pkgutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from modules.patch_generation.ast_analyzer import ASTAnalyzer, ImportInfo

logger = logging.getLogger(__name__)


@dataclass
class ModuleContext:
    """Information about an imported module."""

    name: str
    full_path: Optional[str] = None
    is_resolved: bool = False
    is_stdlib: bool = False
    is_third_party: bool = False
    is_local: bool = False
    symbols: Dict[str, Any] = None
    functions: Set[str] = None
    classes: Set[str] = None
    constants: Dict[str, Any] = None
    doc: Optional[str] = None


class ImportAnalyzer:
    """
    Analyzer for imports and module dependencies.
    """

    def __init__(
        self, ast_analyzer: ASTAnalyzer = None, project_root: Optional[Path] = None
    ):
        """
        Initialize the import analyzer.

        Args:
            ast_analyzer: Optional existing ASTAnalyzer to use
            project_root: Optional root directory of the project for resolving local imports
        """
        self.ast_analyzer = ast_analyzer or ASTAnalyzer()
        self.project_root = project_root
        self.python_path = set(sys.path)

        if project_root:
            # Add project root to Python path for import resolution
            self.python_path.add(str(project_root))

        # Dictionary of module contexts
        self.module_contexts: Dict[str, ModuleContext] = {}

        # Standard library modules
        self.stdlib_modules = self._get_stdlib_modules()

    def _get_stdlib_modules(self) -> Set[str]:
        """
        Get the set of standard library module names.

        Returns:
            Set of standard library module names
        """
        stdlib_modules = set()

        # Common standard library modules
        stdlib_prefixes = {
            "abc",
            "argparse",
            "array",
            "ast",
            "asyncio",
            "base64",
            "collections",
            "concurrent",
            "configparser",
            "contextlib",
            "copy",
            "csv",
            "datetime",
            "decimal",
            "difflib",
            "dis",
            "enum",
            "errno",
            "fnmatch",
            "functools",
            "getopt",
            "getpass",
            "glob",
            "hashlib",
            "heapq",
            "hmac",
            "html",
            "http",
            "importlib",
            "inspect",
            "io",
            "ipaddress",
            "itertools",
            "json",
            "logging",
            "math",
            "multiprocessing",
            "netrc",
            "numbers",
            "operator",
            "os",
            "pathlib",
            "pickle",
            "platform",
            "pprint",
            "queue",
            "random",
            "re",
            "shutil",
            "signal",
            "socket",
            "socketserver",
            "sqlite3",
            "ssl",
            "statistics",
            "string",
            "subprocess",
            "sys",
            "tempfile",
            "threading",
            "time",
            "timeit",
            "tkinter",
            "traceback",
            "typing",
            "unittest",
            "urllib",
            "uuid",
            "venv",
            "warnings",
            "weakref",
            "xml",
            "xmlrpc",
            "zipfile",
            "zlib",
        }

        # Add standard modules and their submodules
        for prefix in stdlib_prefixes:
            stdlib_modules.add(prefix)

            # Try to identify submodules
            try:
                module = importlib.import_module(prefix)
                for _, name, is_pkg in pkgutil.iter_modules(
                    module.__path__, prefix + "."
                ):
                    stdlib_modules.add(name)
            except (ImportError, AttributeError):
                pass

        return stdlib_modules

    def analyze_file(self, file_path: Path) -> bool:
        """
        Analyze a file to gather import information.

        Args:
            file_path: Path to the file to analyze

        Returns:
            True if analysis was successful, False otherwise
        """
        return self.ast_analyzer.parse_file(file_path)

    def analyze_code(self, code: str) -> bool:
        """
        Analyze code to gather import information.

        Args:
            code: Python code as a string

        Returns:
            True if analysis was successful, False otherwise
        """
        return self.ast_analyzer.parse_code(code)

    def get_imports(self) -> List[ImportInfo]:
        """
        Get all imports from the analyzed code.

        Returns:
            List of ImportInfo objects
        """
        return self.ast_analyzer.get_imports()

    def resolve_module(self, module_name: str) -> Optional[ModuleContext]:
        """
        Resolve a module and gather information about it.

        Args:
            module_name: Name of the module to resolve

        Returns:
            ModuleContext object if resolution is successful, None otherwise
        """
        # Check if we've already resolved this module
        if module_name in self.module_contexts:
            return self.module_contexts[module_name]

        # Create a new context
        context = ModuleContext(
            name=module_name, symbols={}, functions=set(), classes=set(), constants={}
        )

        # Check if it's a standard library module
        if (
            module_name in self.stdlib_modules
            or module_name.split(".")[0] in self.stdlib_modules
        ):
            context.is_stdlib = True

        # Try to import the module
        try:
            module = importlib.import_module(module_name)
            context.is_resolved = True

            # Get module path
            if hasattr(module, "__file__"):
                context.full_path = module.__file__

                # Determine if module is local to project
                if self.project_root and context.full_path:
                    try:
                        if Path(context.full_path).is_relative_to(self.project_root):
                            context.is_local = True
                    except ValueError:
                        # Not relative to project root
                        pass

            # Get module docstring
            context.doc = module.__doc__

            # Collect module contents
            for name, obj in inspect.getmembers(module):
                # Skip private members
                if name.startswith("_"):
                    continue

                # Add to symbols
                context.symbols[name] = obj

                # Categorize by type
                if inspect.isfunction(obj) or inspect.isbuiltin(obj):
                    context.functions.add(name)
                elif inspect.isclass(obj):
                    context.classes.add(name)
                elif isinstance(obj, (int, float, str, bool)) or obj is None:
                    context.constants[name] = obj

        except (ImportError, AttributeError, ValueError) as e:
            logger.debug(f"Failed to resolve module {module_name}: {e}")

            # Still not resolved, check if it's a local module
            if self.project_root:
                module_path = self._find_local_module(module_name)
                if module_path:
                    context.full_path = str(module_path)
                    context.is_local = True

                    # Try to analyze the local module
                    try:
                        local_analyzer = ASTAnalyzer()
                        if local_analyzer.parse_file(module_path):
                            # Extract functions and classes
                            for func_name in local_analyzer.get_functions():
                                context.functions.add(func_name)
                                context.symbols[func_name] = "function"

                            for class_name in local_analyzer.get_classes():
                                context.classes.add(class_name)
                                context.symbols[class_name] = "class"

                            context.is_resolved = True
                    except Exception as e:
                        logger.debug(
                            f"Failed to analyze local module {module_name}: {e}"
                        )

        # Determine if it's a third-party module if not stdlib and not local
        if not context.is_stdlib and not context.is_local and context.is_resolved:
            context.is_third_party = True

        # Store the context
        self.module_contexts[module_name] = context

        return context

    def _find_local_module(self, module_name: str) -> Optional[Path]:
        """
        Find a local module file.

        Args:
            module_name: Name of the module to find

        Returns:
            Path to the module file if found, None otherwise
        """
        if not self.project_root:
            return None

        # Replace dots with path separators
        relative_path = module_name.replace(".", os.sep)

        # Check different possible file paths
        possible_paths = [
            self.project_root / f"{relative_path}.py",
            self.project_root / relative_path / "__init__.py",
        ]

        for path in possible_paths:
            if path.exists():
                return path

        return None

    def resolve_all_imports(self) -> Dict[str, ModuleContext]:
        """
        Resolve all imports in the analyzed code.

        Returns:
            Dictionary mapping module names to ModuleContext objects
        """
        imports = self.get_imports()

        for import_info in imports:
            if import_info.is_from:
                # from module import names
                self.resolve_module(import_info.module)
            else:
                # import module
                for name, _ in import_info.names:
                    # For import statements, only resolve the top-level module
                    top_module = name.split(".")[0]
                    self.resolve_module(top_module)

        return self.module_contexts

    def find_symbol_module(self, symbol_name: str) -> Optional[str]:
        """
        Find which module a symbol was imported from.

        Args:
            symbol_name: Name of the symbol to find

        Returns:
            Module name if found, None otherwise
        """
        imports = self.get_imports()

        for import_info in imports:
            if import_info.is_from:
                # Check if symbol was imported from this module
                for name, alias in import_info.names:
                    if (alias and alias == symbol_name) or name == symbol_name:
                        return import_info.module
            else:
                # Check if module itself was imported as this name
                for name, alias in import_info.names:
                    if alias == symbol_name:
                        return name
                    elif name == symbol_name:
                        return name
                    elif symbol_name.startswith(f"{name}."):
                        # Might be a submodule or attribute of this module
                        return name

        return None

    def find_module_symbols(self, module_name: str) -> Set[str]:
        """
        Find all symbols imported from a module.

        Args:
            module_name: Name of the module

        Returns:
            Set of symbol names imported from the module
        """
        imports = self.get_imports()
        symbols = set()

        for import_info in imports:
            if import_info.is_from and import_info.module == module_name:
                # Add symbols imported from this module
                for name, alias in import_info.names:
                    symbols.add(alias if alias else name)

        return symbols

    def get_import_graph(self) -> Dict[str, Set[str]]:
        """
        Generate a graph of import dependencies.

        Returns:
            Dictionary mapping module names to sets of imported module names
        """
        # Resolve all imports
        self.resolve_all_imports()

        # Create a graph of module dependencies
        graph = {}

        for module_name, context in self.module_contexts.items():
            if context.is_local:
                # For local modules, try to analyze their imports
                if context.full_path:
                    try:
                        local_analyzer = ASTAnalyzer()
                        if local_analyzer.parse_file(Path(context.full_path)):
                            local_imports = ImportAnalyzer(local_analyzer)
                            imports = local_imports.get_imports()

                            # Add dependencies to graph
                            dependencies = set()
                            for import_info in imports:
                                if import_info.is_from:
                                    dependencies.add(import_info.module)
                                else:
                                    for name, _ in import_info.names:
                                        dependencies.add(name.split(".")[0])

                            graph[module_name] = dependencies
                    except Exception as e:
                        logger.debug(
                            f"Failed to analyze imports for {module_name}: {e}"
                        )
                        graph[module_name] = set()
                else:
                    graph[module_name] = set()
            else:
                # For non-local modules, we don't analyze their imports
                graph[module_name] = set()

        return graph

    def find_missing_imports(self) -> List[Tuple[str, str, int]]:
        """
        Find symbols used in the code that might be missing imports.

        Returns:
            List of (symbol_name, suggested_module, line_number) tuples
        """
        missing_imports = []

        # Get all used variables
        variables = self.ast_analyzer.get_variables()

        # Get variables that are used but not defined in the current scope
        for name, var_info in variables.items():
            if (
                var_info.usages
                and not var_info.assignments
                and not var_info.is_parameter
            ):
                # This variable is used but not defined or imported, might need an import
                if not var_info.is_imported:
                    # Try to guess which module it might come from
                    suggested_module = self._suggest_module_for_symbol(name)
                    if suggested_module:
                        # Find the first usage of this variable
                        line_number = (
                            var_info.usages[0].lineno
                            if hasattr(var_info.usages[0], "lineno")
                            else 0
                        )
                        missing_imports.append((name, suggested_module, line_number))

        return missing_imports

    def _suggest_module_for_symbol(self, symbol_name: str) -> Optional[str]:
        """
        Suggest a module that might contain the given symbol.

        Args:
            symbol_name: Name of the symbol to find

        Returns:
            Suggested module name if found, None otherwise
        """
        # Common modules for standard types and functions
        common_symbols = {
            # Math and numbers
            "math": {"sin", "cos", "tan", "pi", "sqrt", "floor", "ceil"},
            "random": {"random", "randint", "choice", "shuffle", "sample"},
            "statistics": {"mean", "median", "mode", "stdev"},
            # Data structures
            "collections": {"defaultdict", "Counter", "namedtuple", "deque"},
            "itertools": {"chain", "combinations", "permutations", "product", "cycle"},
            "functools": {"partial", "reduce", "lru_cache", "wraps"},
            # IO and System
            "os": {"path", "environ", "getcwd", "mkdir", "remove", "listdir"},
            "sys": {"argv", "exit", "path", "platform", "stdin", "stdout"},
            "pathlib": {"Path", "PurePath"},
            "datetime": {"datetime", "date", "time", "timedelta"},
            # Web and Networking
            "requests": {"get", "post", "put", "delete", "Session"},
            "json": {"dumps", "loads", "JSONEncoder", "JSONDecoder"},
            "urllib": {"request", "parse", "error"},
            "http": {"client", "server"},
            # Typing
            "typing": {
                "List",
                "Dict",
                "Tuple",
                "Set",
                "Optional",
                "Union",
                "Any",
                "Callable",
            },
            # Testing
            "unittest": {"TestCase", "assertEqual", "assertTrue", "assertRaises"},
            "pytest": {"fixture", "mark", "raises", "approx"},
            # Web Frameworks
            "flask": {"Flask", "request", "Blueprint", "jsonify", "render_template"},
            "django": {"models", "views", "urls", "forms", "admin"},
            "fastapi": {"FastAPI", "Query", "Path", "Body", "Depends"},
            # Data Science
            "numpy": {"array", "zeros", "ones", "random", "matmul", "linspace"},
            "pandas": {"DataFrame", "Series", "read_csv", "concat", "merge"},
            "matplotlib": {"pyplot", "figure", "plot", "scatter", "hist"},
        }

        # Check if symbol matches a common module symbol
        for module, symbols in common_symbols.items():
            if symbol_name in symbols:
                return module

        # Check if symbol matches a class name (UpperCamelCase)
        if symbol_name[0].isupper() and not symbol_name.isupper():
            # Check for common classes
            class_modules = {
                "Exception": "builtins",
                "ValueError": "builtins",
                "TypeError": "builtins",
                "KeyError": "builtins",
                "IndexError": "builtins",
                "FileNotFoundError": "builtins",
                "Path": "pathlib",
                "DataFrame": "pandas",
                "Series": "pandas",
                "TestCase": "unittest",
                "Logger": "logging",
                "JSONEncoder": "json",
                "HTTPError": "requests",
                "Response": "requests",
                "Template": "string",
                "Process": "multiprocessing",
                "Thread": "threading",
                "Enum": "enum",
            }

            if symbol_name in class_modules:
                return class_modules[symbol_name]

        # Check for common third-party libraries
        if symbol_name.lower() in {
            "np",
            "pd",
            "plt",
            "sns",
            "tf",
            "torch",
            "sklearn",
            "requests",
            "bs4",
            "Flask",
            "Django",
            "FastAPI",
        }:
            # Common aliases
            aliases = {
                "np": "numpy",
                "pd": "pandas",
                "plt": "matplotlib.pyplot",
                "sns": "seaborn",
                "tf": "tensorflow",
                "sklearn": "scikit-learn",
                "bs4": "beautifulsoup4",
            }

            return aliases.get(symbol_name.lower(), symbol_name)

        # No suggestion found
        return None

    def gather_module_context(self, module_name: str) -> Dict[str, Any]:
        """
        Gather comprehensive context information about a module.

        Args:
            module_name: Name of the module

        Returns:
            Dictionary of context information
        """
        # Resolve the module if needed
        if module_name not in self.module_contexts:
            self.resolve_module(module_name)

        context = self.module_contexts.get(module_name)
        if not context or not context.is_resolved:
            return {}

        # Gather context information
        return {
            "name": context.name,
            "path": context.full_path,
            "is_stdlib": context.is_stdlib,
            "is_third_party": context.is_third_party,
            "is_local": context.is_local,
            "functions": sorted(list(context.functions)),
            "classes": sorted(list(context.classes)),
            "constants": context.constants,
            "doc": context.doc,
        }

    def gather_all_module_contexts(self) -> Dict[str, Dict[str, Any]]:
        """
        Gather context information for all resolved modules.

        Returns:
            Dictionary mapping module names to context information
        """
        # Resolve all imports
        self.resolve_all_imports()

        result = {}
        for module_name, context in self.module_contexts.items():
            if context.is_resolved:
                result[module_name] = self.gather_module_context(module_name)

        return result

    def suggest_imports(self, code_lines: List[str]) -> List[Dict[str, Any]]:
        """
        Suggest imports for symbols used in code.

        Args:
            code_lines: Lines of code to analyze

        Returns:
            List of import suggestions
        """
        # Parse the code
        if not self.ast_analyzer.parse_code("\n".join(code_lines)):
            return []

        # Find missing imports
        missing_imports = self.find_missing_imports()

        # Generate suggestions
        suggestions = []
        for symbol, module, line_number in missing_imports:
            suggestions.append(
                {
                    "symbol": symbol,
                    "module": module,
                    "line": line_number,
                    "import_statement": f"from {module} import {symbol}",
                    "confidence": "medium",
                }
            )

        return suggestions


if __name__ == "__main__":
    # Example usage
    test_code = """
import os
from typing import List, Dict, Optional
import numpy as np
from sklearn.model_selection import train_test_split

# This uses pandas but doesn't import it
df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

# This uses matplotlib but doesn't import it
plt.figure()
plt.plot(df['a'], df['b'])
plt.show()

# This uses Path but doesn't import it
data_path = Path('data/file.csv')
if data_path.exists():
    df = pd.read_csv(data_path)
    
# Create model using sklearn
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
"""

    # Create and use the import analyzer
    analyzer = ImportAnalyzer()
    if analyzer.analyze_code(test_code):
        # Find missing imports
        missing = analyzer.find_missing_imports()

        print("Missing imports:")
        for symbol, module, line in missing:
            print(f"- {symbol} from {module} (line {line})")

        # Resolve existing imports
        modules = analyzer.resolve_all_imports()

        print("\nResolved modules:")
        for name, context in modules.items():
            status = []
            if context.is_stdlib:
                status.append("stdlib")
            if context.is_third_party:
                status.append("third-party")
            if context.is_local:
                status.append("local")

            status_str = ", ".join(status) if status else "unresolved"
            print(f"- {name} ({status_str})")

            if context.is_resolved:
                if context.full_path:
                    print(f"  Path: {context.full_path}")
                if context.functions:
                    print(f"  Functions: {len(context.functions)}")
                if context.classes:
                    print(f"  Classes: {len(context.classes)}")

        # Make import suggestions
        suggestions = analyzer.suggest_imports(test_code.splitlines())

        print("\nImport suggestions:")
        for suggestion in suggestions:
            print(f"- {suggestion['import_statement']} (line {suggestion['line']})")
