"""
FastAPI Dependency Analyzer

This module provides analysis capabilities for FastAPI dependency chains and patterns.
It helps identify common issues in dependency design, potential performance bottlenecks,
and security concerns in FastAPI applications.
"""

import ast
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .rule_based import RuleBasedAnalyzer
from .rule_config import RuleCategory

logger = logging.getLogger(__name__)


class FastAPIDependencyVisitor(ast.NodeVisitor):
    """AST visitor for analyzing FastAPI dependencies."""

    def __init__(self):
        self.dependencies = {}
        self.routes = {}
        self.dependency_chains = {}
        self.potential_issues = []
        self.current_function = None

    def visit_FunctionDef(self, node):
        """Visit function definitions."""
        prev_function = self.current_function
        self.current_function = node.name

        # Check for FastAPI route decorators
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Call) and isinstance(
                decorator.func, ast.Attribute
            ):
                if decorator.func.attr in [
                    "get",
                    "post",
                    "put",
                    "delete",
                    "patch",
                    "head",
                    "options",
                ]:
                    self.routes[node.name] = {
                        "deps": [],
                        "line": node.lineno,
                        "method": decorator.func.attr,
                    }

                    # Extract dependencies from the function parameters
                    self._extract_dependencies(node)

        # Check if this is a dependency function (standalone without @app decorators)
        if not self._has_route_decorator(node) and self._might_be_dependency(node):
            self.dependencies[node.name] = {
                "deps": [],  # Dependencies that this dependency relies on
                "line": node.lineno,
                "yields": self._has_yield(node),
                "is_async": self._is_async(node),
            }

        # Visit children
        self.generic_visit(node)
        self.current_function = prev_function

    def visit_Call(self, node):
        """Visit function calls, looking for Depends() calls."""
        if self._is_depends_call(node):
            # Extract the dependency name or callable
            dep_name = self._extract_dependency_name(node)
            if dep_name and self.current_function:
                # Add to the current function's dependencies
                if self.current_function in self.routes:
                    self.routes[self.current_function]["deps"].append(dep_name)
                elif self.current_function in self.dependencies:
                    self.dependencies[self.current_function]["deps"].append(dep_name)

                    # Check for potential circular dependencies
                    self._check_circular_dependency(self.current_function, dep_name)

        # Visit children
        self.generic_visit(node)

    def _has_route_decorator(self, node):
        """Check if a function has FastAPI route decorators."""
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Call) and isinstance(
                decorator.func, ast.Attribute
            ):
                if decorator.func.attr in [
                    "get",
                    "post",
                    "put",
                    "delete",
                    "patch",
                    "head",
                    "options",
                ]:
                    return True
        return False

    def _might_be_dependency(self, node):
        """Check if a function might be a dependency."""
        # Dependencies often have Path, Query, Body, etc. parameters
        for arg in node.args.args:
            if arg.annotation and isinstance(arg.annotation, ast.Name):
                if arg.annotation.id in [
                    "Path",
                    "Query",
                    "Body",
                    "Header",
                    "Cookie",
                    "Depends",
                ]:
                    return True

            # Check for more complex annotations like Annotated[Type, Query(), etc.]
            if arg.annotation and isinstance(arg.annotation, ast.Subscript):
                if (
                    isinstance(arg.annotation.value, ast.Name)
                    and arg.annotation.value.id == "Annotated"
                ):
                    return True

        # Functions with no decorators that have parameters might be dependencies
        if not node.decorator_list and node.args.args:
            return True

        return False

    def _extract_dependencies(self, node):
        """Extract dependencies from a function's parameters."""
        for arg in node.args.args:
            if arg.annotation:
                # Check for Annotated[Type, Depends(...)]
                if (
                    isinstance(arg.annotation, ast.Subscript)
                    and isinstance(arg.annotation.value, ast.Name)
                    and arg.annotation.value.id == "Annotated"
                ):
                    # Extract Depends from slice
                    if isinstance(arg.annotation.slice, ast.Tuple):
                        for elt in arg.annotation.slice.elts[
                            1:
                        ]:  # Skip the first element (type)
                            if self._is_depends_call(elt):
                                dep_name = self._extract_dependency_name(elt)
                                if dep_name:
                                    self.routes[node.name]["deps"].append(dep_name)

    def _is_depends_call(self, node):
        """Check if an AST node is a Depends() call."""
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "Depends"
        ):
            return True
        return False

    def _extract_dependency_name(self, node):
        """Extract the dependency name from a Depends() call."""
        if node.args:
            if isinstance(node.args[0], ast.Name):
                return node.args[0].id
            # Handle other cases (lambdas, attributes, etc.)
            return "anonymous_dependency"
        return None

    def _check_circular_dependency(self, source, target):
        """Check for circular dependencies."""
        # Start a dependency chain if it doesn't exist
        if source not in self.dependency_chains:
            self.dependency_chains[source] = set([source])

        # Check if adding this dependency would create a circle
        if target in self.dependency_chains[source]:
            self.potential_issues.append(
                {
                    "type": "circular_dependency",
                    "message": f"Potential circular dependency detected: {source} -> {target}",
                    "location": {"function": source, "dependency": target},
                    "severity": "high",
                }
            )
        else:
            # Add the target to the chain
            self.dependency_chains[source].add(target)

            # If the target has its own dependencies, add those too
            if target in self.dependencies:
                for dep in self.dependencies[target]["deps"]:
                    if dep not in self.dependency_chains[source]:
                        self.dependency_chains[source].add(dep)

    def _has_yield(self, node):
        """Check if a function contains a yield statement."""
        for child in ast.walk(node):
            if isinstance(child, ast.Yield) or isinstance(child, ast.YieldFrom):
                return True
        return False

    def _is_async(self, node):
        """Check if a function is async."""
        return (
            isinstance(node, ast.AsyncFunctionDef)
            or hasattr(node, "is_async")
            and node.is_async
        )


class FastAPIDependencyAnalyzer:
    """
    Analyzer for FastAPI dependencies and dependency patterns.

    This class analyzes FastAPI code to identify dependency patterns, potential issues,
    and optimize dependency handling.
    """

    def __init__(self):
        """Initialize the FastAPI dependency analyzer."""
        self.rule_analyzer = RuleBasedAnalyzer(categories=[RuleCategory.FASTAPI])

    def analyze_code(
        self, file_path: str, code: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze FastAPI code for dependency patterns and issues.

        Args:
            file_path: Path to the FastAPI application file
            code: Optional source code string (if not provided, file_path will be read)

        Returns:
            Analysis results
        """
        if code is None:
            with open(file_path, "r") as f:
                code = f.read()

        try:
            # Parse the code into an AST
            tree = ast.parse(code)

            # Visit the AST to find dependencies
            visitor = FastAPIDependencyVisitor()
            visitor.visit(tree)

            # Analyze the results
            analysis_results = {
                "dependencies": visitor.dependencies,
                "routes": visitor.routes,
                "dependency_chains": visitor.dependency_chains,
                "potential_issues": visitor.potential_issues,
                "file_path": file_path,
            }

            # Add additional insights
            self._add_insights(analysis_results)

            return analysis_results

        except SyntaxError as e:
            logger.error(f"Failed to parse file {file_path}: {str(e)}")
            return {"error": "syntax_error", "message": str(e), "file_path": file_path}
        except Exception as e:
            logger.exception(f"Error analyzing file {file_path}: {str(e)}")
            return {
                "error": "analysis_error",
                "message": str(e),
                "file_path": file_path,
            }

    def analyze_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a FastAPI error related to dependencies.

        Args:
            error_data: Error data to analyze

        Returns:
            Analysis results
        """
        # First, try the rule-based analyzer for known patterns
        rule_analysis = self.rule_analyzer.analyze_error(error_data)

        # Check if it's a dependency-related error based on rule analysis
        if self._is_dependency_error(rule_analysis):
            # Enhance the analysis with more specific dependency insights
            enhanced_analysis = self._enhance_dependency_analysis(
                error_data, rule_analysis
            )
            return enhanced_analysis

        return rule_analysis

    def analyze_application(self, app_dir: str) -> Dict[str, Any]:
        """
        Analyze a complete FastAPI application for dependency patterns.

        Args:
            app_dir: Directory of the FastAPI application

        Returns:
            Complete analysis of the application's dependencies
        """
        app_dir_path = Path(app_dir)
        py_files = list(app_dir_path.glob("**/*.py"))

        # Analyze each Python file
        file_analyses = {}
        for file_path in py_files:
            try:
                with open(file_path, "r") as f:
                    code = f.read()

                # Only analyze if it looks like a FastAPI file
                if "fastapi" in code and (
                    "app = FastAPI" in code or "Depends(" in code
                ):
                    analysis = self.analyze_code(str(file_path), code)
                    if analysis.get("dependencies") or analysis.get("routes"):
                        file_analyses[str(file_path)] = analysis
            except Exception as e:
                logger.exception(f"Error analyzing file {file_path}: {str(e)}")

        # Consolidate the results
        consolidated_analysis = self._consolidate_analyses(file_analyses)
        return consolidated_analysis

    def _is_dependency_error(self, rule_analysis: Dict[str, Any]) -> bool:
        """Check if a rule analysis indicates a dependency-related error."""
        if rule_analysis.get("category") == "fastapi" and any(
            tag in rule_analysis.get("tags", []) for tag in ["dependency", "injection"]
        ):
            return True

        # Check root cause
        if rule_analysis.get("root_cause", "").startswith("fastapi_dependency"):
            return True

        return False

    def _enhance_dependency_analysis(
        self, error_data: Dict[str, Any], rule_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Enhance dependency error analysis with specific insights."""
        enhanced = rule_analysis.copy()

        # Extract the stack trace for more context
        stack_trace = error_data.get("traceback", [])

        # Try to identify the specific dependency causing the issue
        dependency_name = self._extract_dependency_from_traceback(stack_trace)
        if dependency_name:
            enhanced["dependency_name"] = dependency_name

        # Check for common dependency anti-patterns
        anti_patterns = self._identify_anti_patterns(error_data, rule_analysis)
        if anti_patterns:
            enhanced["anti_patterns"] = anti_patterns

        return enhanced

    def _extract_dependency_from_traceback(self, traceback: List[str]) -> Optional[str]:
        """Extract dependency name from traceback."""
        # Look for Depends() calls in the traceback
        depends_pattern = re.compile(r"Depends\(([^)]+)\)")

        for line in traceback:
            match = depends_pattern.search(line)
            if match:
                return match.group(1)

        return None

    def _identify_anti_patterns(
        self, error_data: Dict[str, Any], rule_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify common dependency anti-patterns from error data."""
        anti_patterns = []

        # Check for heavy dependencies that should be cached
        if (
            "TimeoutError" in error_data.get("exception_type", "")
            or "timeout" in error_data.get("message", "").lower()
        ):
            anti_patterns.append(
                {
                    "type": "heavy_dependency",
                    "message": "Potentially heavy dependency causing timeouts",
                    "suggestion": 'Use FastAPI dependency caching with "use_cache=True" parameter',
                }
            )

        # Check for mixing async/sync in dependencies
        if "asyncio" in " ".join(error_data.get("traceback", [])) and (
            "yield" in " ".join(error_data.get("traceback", []))
            or "await" in " ".join(error_data.get("traceback", []))
        ):
            anti_patterns.append(
                {
                    "type": "async_sync_mixing",
                    "message": "Potential mixing of async and sync code in dependencies",
                    "suggestion": "Ensure consistency in async/sync implementation for dependencies",
                }
            )

        return anti_patterns

    def _add_insights(self, analysis_results: Dict[str, Any]) -> None:
        """Add additional insights to the analysis results."""
        insights = []

        # Check for many dependencies on a single route
        for route, data in analysis_results.get("routes", {}).items():
            if len(data.get("deps", [])) > 5:
                insights.append(
                    {
                        "type": "many_dependencies",
                        "message": f'Route {route} has {len(data["deps"])} dependencies, which may impact performance',
                        "location": {"route": route, "line": data.get("line")},
                        "severity": "medium",
                    }
                )

        # Check for shared dependencies that could be cached
        dependency_usage_count: Dict[str, int] = {}
        for route_data in analysis_results.get("routes", {}).values():
            for dep in route_data.get("deps", []):
                dependency_usage_count[dep] = dependency_usage_count.get(dep, 0) + 1

        for dep, count in dependency_usage_count.items():
            if count > 3 and dep in analysis_results.get("dependencies", {}):
                insights.append(
                    {
                        "type": "cacheable_dependency",
                        "message": f"Dependency {dep} is used in {count} routes and could benefit from caching",
                        "location": {"dependency": dep},
                        "severity": "low",
                    }
                )

        # Check for yield dependencies that might be resource-heavy
        for dep, data in analysis_results.get("dependencies", {}).items():
            if data.get("yields") and len(analysis_results.get("routes", {})) > 3:
                insights.append(
                    {
                        "type": "resource_dependency",
                        "message": f"Dependency {dep} uses yield for resource management and is used across multiple routes",
                        "location": {"dependency": dep, "line": data.get("line")},
                        "severity": "low",
                    }
                )

        # Add the insights to the results
        analysis_results["insights"] = insights

    def _consolidate_analyses(
        self, file_analyses: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Consolidate analyses from multiple files."""
        consolidated: Dict[str, Any] = {
            "dependencies": {},
            "routes": {},
            "dependency_chains": {},
            "potential_issues": [],
            "insights": [],
            "files_analyzed": list(file_analyses.keys()),
        }

        for file_path, analysis in file_analyses.items():
            # Add file-specific prefix to names to avoid collisions
            file_prefix = Path(file_path).stem + "."

            # Add dependencies
            for dep, data in analysis.get("dependencies", {}).items():
                consolidated["dependencies"][file_prefix + dep] = {
                    **data,
                    "file_path": file_path,
                }

            # Add routes
            for route, data in analysis.get("routes", {}).items():
                consolidated["routes"][file_prefix + route] = {
                    **data,
                    "file_path": file_path,
                }

            # Add dependency chains
            for source, targets in analysis.get("dependency_chains", {}).items():
                consolidated["dependency_chains"][file_prefix + source] = {
                    "targets": [file_prefix + t for t in targets],
                    "file_path": file_path,
                }

            # Add issues and insights
            for issue in analysis.get("potential_issues", []):
                issue_copy = issue.copy()
                issue_copy["file_path"] = file_path
                consolidated["potential_issues"].append(issue_copy)

            for insight in analysis.get("insights", []):
                insight_copy = insight.copy()
                insight_copy["file_path"] = file_path
                consolidated["insights"].append(insight_copy)

        # Check for cross-file circular dependencies
        self._check_cross_file_circular_deps(consolidated)

        return consolidated

    def _check_cross_file_circular_deps(self, consolidated: Dict[str, Any]) -> None:
        """Check for circular dependencies across files."""
        # Build a complete dependency graph
        dependency_graph = {}

        for source, data in consolidated.get("dependency_chains", {}).items():
            dependency_graph[source] = set(data.get("targets", []))

        # Check for cycles in the graph
        visited = set()
        recursion_stack = set()

        def dfs_check_cycle(node, path=None):
            if path is None:
                path = []

            if node in recursion_stack:
                # Found a cycle
                cycle_path = path + [node]
                consolidated["potential_issues"].append(
                    {
                        "type": "cross_file_circular_dependency",
                        "message": f'Cross-file circular dependency detected: {" -> ".join(cycle_path)}',
                        "location": {"dependencies": cycle_path},
                        "severity": "high",
                    }
                )
                return True

            if node in visited:
                return False

            visited.add(node)
            recursion_stack.add(node)

            for neighbor in dependency_graph.get(node, set()):
                if dfs_check_cycle(neighbor, path + [node]):
                    return True

            recursion_stack.remove(node)
            return False

        # Check each node
        for node in dependency_graph:
            if node not in visited:
                dfs_check_cycle(node)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    analyzer = FastAPIDependencyAnalyzer()

    # Example error analysis
    error_data = {
        "timestamp": "2023-08-14T15:30:45",
        "level": "ERROR",
        "message": "Exception in dependency get_db",
        "exception_type": "SQLAlchemyError",
        "traceback": [
            "Traceback (most recent call last):",
            '  File "/app/dependencies.py", line 25, in get_db',
            "    db = SessionLocal()",
            '  File "/usr/local/lib/python3.9/site-packages/sqlalchemy/orm/session.py", line 3527, in __init__',
            "    self.connection = engine.connect()",
            '  File "/usr/local/lib/python3.9/site-packages/sqlalchemy/engine/base.py", line 3265, in connect',
            "    return self._connection_cls(self)",
            '  File "/usr/local/lib/python3.9/site-packages/sqlalchemy/engine/base.py", line 96, in __init__',
            "    else engine.raw_connection()",
            '  File "/usr/local/lib/python3.9/site-packages/sqlalchemy/engine/base.py", line 3339, in raw_connection',
            "    return self._wrap_pool_connect(self.pool.connect, _connection)",
            '  File "/usr/local/lib/python3.9/site-packages/sqlalchemy/engine/base.py", line 3309, in _wrap_pool_connect',
            "    return fn()",
            '  File "/usr/local/lib/python3.9/site-packages/sqlalchemy/pool/base.py", line 310, in connect',
            "    return _ConnectionFairy._checkout(self)",
            '  File "/usr/local/lib/python3.9/site-packages/sqlalchemy/pool/base.py", line 868, in _checkout',
            "    fairy = _ConnectionRecord.checkout(pool)",
            '  File "/usr/local/lib/python3.9/site-packages/sqlalchemy/pool/base.py", line 476, in checkout',
            "    rec = pool._do_get()",
            '  File "/usr/local/lib/python3.9/site-packages/sqlalchemy/pool/impl.py", line 146, in _do_get',
            "    self._dec_overflow()",
            '  File "/usr/local/lib/python3.9/site-packages/sqlalchemy/pool/impl.py", line 323, in _dec_overflow',
            "    self._overflow -= 1",
            '  File "/usr/local/lib/python3.9/site-packages/sqlalchemy/pool/base.py", line 307, in _checkout_existing',
            "    return self._checkout_impl()",
            '  File "/usr/local/lib/python3.9/site-packages/sqlalchemy/pool/base.py", line 454, in checkout',
            "    rec = pool._do_get()",
            '  File "/usr/local/lib/python3.9/site-packages/sqlalchemy/pool/impl.py", line 256, in _do_get',
            "    return self._create_connection()",
            '  File "/usr/local/lib/python3.9/site-packages/sqlalchemy/pool/base.py", line 273, in _create_connection',
            "    return _ConnectionRecord(self)",
            '  File "/usr/local/lib/python3.9/site-packages/sqlalchemy/pool/base.py", line 388, in __init__',
            "    self.__connect()",
            '  File "/usr/local/lib/python3.9/site-packages/sqlalchemy/pool/base.py", line 691, in __connect',
            '    pool.logger.debug("Error on connect(): %s", e)',
            '  File "/usr/local/lib/python3.9/site-packages/sqlalchemy/util/langhelpers.py", line 72, in __exit__',
            "    raise exc_value.with_traceback(exc_tb)",
            '  File "/usr/local/lib/python3.9/site-packages/sqlalchemy/pool/base.py", line 686, in __connect',
            "    self.dbapi_connection = connection = pool._invoke_creator(self)",
            '  File "/usr/local/lib/python3.9/site-packages/sqlalchemy/engine/create.py", line 578, in connect',
            "    return dialect.connect(*cargs, **cparams)",
            '  File "/usr/local/lib/python3.9/site-packages/sqlalchemy/engine/default.py", line 598, in connect',
            "    return self.dbapi.connect(*cargs, **cparams)",
            '  File "/usr/local/lib/python3.9/site-packages/psycopg2/__init__.py", line 122, in connect',
            "    conn = _connect(dsn, connection_factory=connection_factory, **kwasync)",
            "psycopg2.OperationalError: could not connect to server: Connection refused",
        ],
    }

    analysis = analyzer.analyze_error(error_data)
    logger.info(f"Error Analysis: {analysis}")

    # Example code analysis
    sample_code = """
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session

app = FastAPI()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_user_db(db: Session = Depends(get_db)):
    return UserRepository(db)

def require_admin(user_db = Depends(get_user_db), token: str = Depends(get_token)):
    user = user_db.get_user_by_token(token)
    if not user or not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    return user

@app.get("/admin/dashboard")
async def admin_dashboard(admin = Depends(require_admin)):
    return {"message": "Admin dashboard"}
    """

    code_analysis = analyzer.analyze_code("sample.py", sample_code)
    logger.info(f"Code Analysis: {code_analysis}")
