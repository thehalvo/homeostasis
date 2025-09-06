#!/usr/bin/env python3
"""
Contextual Code Analyzer

This module provides deep contextual analysis for code generation,
including multi-file dependency analysis, call graph construction,
and impact analysis for changes.
"""

import ast
import logging
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx

logger = logging.getLogger(__name__)


@dataclass
class DependencyNode:
    """Represents a node in the dependency graph."""

    name: str
    file_path: str
    node_type: str  # 'function', 'class', 'module', 'variable'
    line_number: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DependencyEdge:
    """Represents an edge in the dependency graph."""

    source: str
    target: str
    edge_type: str  # 'calls', 'imports', 'inherits', 'uses'
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ImpactAnalysis:
    """Results of impact analysis for a change."""

    direct_impacts: List[DependencyNode]
    indirect_impacts: List[DependencyNode]
    affected_tests: List[str]
    risk_level: str  # 'low', 'medium', 'high'
    breaking_changes: List[Dict[str, Any]]
    suggested_validations: List[str]


class ContextualAnalyzer:
    """
    Provides deep contextual analysis for code understanding and generation.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the contextual analyzer.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.dependency_graph = nx.DiGraph()
        self.file_cache = {}
        self.analysis_cache = {}

        # Configuration
        self.max_depth = self.config.get("max_analysis_depth", 5)
        self.include_tests = self.config.get("include_tests", True)
        self.track_data_flow = self.config.get("track_data_flow", True)

        logger.info("Initialized Contextual Analyzer")

    def analyze_codebase_context(
        self, target_file: str, root_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze the entire codebase context for a target file.

        Args:
            target_file: Path to the target file
            root_dir: Root directory of the project

        Returns:
            Comprehensive context analysis
        """
        if root_dir is None:
            root_dir = str(Path(target_file).parent)

        # Build dependency graph
        self._build_dependency_graph(root_dir, target_file)

        # Analyze module structure
        module_structure = self._analyze_module_structure(root_dir)

        # Find entry points
        entry_points = self._find_entry_points(root_dir)

        # Analyze test coverage
        test_coverage = self._analyze_test_coverage(target_file, root_dir)

        # Identify critical paths
        critical_paths = self._identify_critical_paths(target_file)

        return {
            "dependency_graph": self._serialize_graph(),
            "module_structure": module_structure,
            "entry_points": entry_points,
            "test_coverage": test_coverage,
            "critical_paths": critical_paths,
            "file_metrics": self._calculate_file_metrics(target_file),
        }

    def analyze_change_impact(
        self,
        file_path: str,
        change_type: str,
        changed_entity: str,
        change_details: Dict[str, Any],
    ) -> ImpactAnalysis:
        """
        Analyze the impact of a proposed change.

        Args:
            file_path: Path to file being changed
            change_type: Type of change ('function', 'class', 'interface')
            changed_entity: Name of the entity being changed
            change_details: Details about the change

        Returns:
            Impact analysis results
        """
        # Find direct dependencies
        direct_impacts = self._find_direct_dependencies(
            file_path, changed_entity, change_type
        )

        # Find indirect dependencies using graph traversal
        indirect_impacts = self._find_indirect_dependencies(
            direct_impacts, max_depth=self.max_depth
        )

        # Find affected tests
        affected_tests = self._find_affected_tests(
            file_path, changed_entity, direct_impacts + indirect_impacts
        )

        # Assess risk level
        risk_level = self._assess_risk_level(
            direct_impacts, indirect_impacts, change_details
        )

        # Identify breaking changes
        breaking_changes = self._identify_breaking_changes(
            change_type, change_details, direct_impacts
        )

        # Generate validation suggestions
        suggested_validations = self._generate_validation_suggestions(
            change_type, risk_level, affected_tests
        )

        return ImpactAnalysis(
            direct_impacts=direct_impacts,
            indirect_impacts=indirect_impacts,
            affected_tests=affected_tests,
            risk_level=risk_level,
            breaking_changes=breaking_changes,
            suggested_validations=suggested_validations,
        )

    def build_call_graph(self, file_path: str, content: str) -> nx.DiGraph:
        """
        Build a detailed call graph for a file.

        Args:
            file_path: Path to the file
            content: File content

        Returns:
            Call graph as NetworkX DiGraph
        """
        call_graph = nx.DiGraph()

        # Detect language
        language = self._detect_language(file_path)

        if language == "python":
            call_graph = self._build_python_call_graph(content, file_path)
        elif language in ["javascript", "typescript"]:
            call_graph = self._build_javascript_call_graph(content, file_path)
        elif language == "java":
            call_graph = self._build_java_call_graph(content, file_path)

        return call_graph

    def analyze_data_dependencies(self, file_path: str, content: str) -> Dict[str, Any]:
        """
        Analyze data dependencies and flow in the code.

        Args:
            file_path: Path to the file
            content: File content

        Returns:
            Data dependency analysis
        """
        language = self._detect_language(file_path)

        if language == "python":
            return self._analyze_python_data_flow(content)
        elif language in ["javascript", "typescript"]:
            return self._analyze_javascript_data_flow(content)
        else:
            return self._analyze_generic_data_flow(content, language)

    def find_usage_patterns(
        self, entity_name: str, entity_type: str, search_dir: str
    ) -> List[Dict[str, Any]]:
        """
        Find usage patterns of an entity across the codebase.

        Args:
            entity_name: Name of the entity
            entity_type: Type of entity ('function', 'class', 'variable')
            search_dir: Directory to search in

        Returns:
            List of usage patterns
        """
        usage_patterns = []

        # Search for entity usage
        for file_path in Path(search_dir).rglob(
            "*.py"
        ):  # TODO: Support multiple languages
            try:
                content = file_path.read_text()
                usages = self._find_entity_usages(
                    content, entity_name, entity_type, str(file_path)
                )
                usage_patterns.extend(usages)
            except Exception as e:
                logger.warning(f"Error analyzing {file_path}: {e}")

        return usage_patterns

    def _build_dependency_graph(self, root_dir: str, target_file: str):
        """Build comprehensive dependency graph for the codebase."""
        # Clear existing graph
        self.dependency_graph.clear()

        # Find all relevant files
        relevant_files = self._find_relevant_files(root_dir, target_file)

        for file_path in relevant_files:
            try:
                content = Path(file_path).read_text()
                self._analyze_file_dependencies(file_path, content)
            except Exception as e:
                logger.warning(f"Error analyzing {file_path}: {e}")

    def _find_relevant_files(self, root_dir: str, target_file: str) -> List[str]:
        """Find files relevant to the target file."""
        relevant = [target_file]
        root_path = Path(root_dir)

        # Add files in the same directory
        target_dir = Path(target_file).parent
        for file_path in target_dir.glob("*"):
            if file_path.is_file() and file_path.suffix in [
                ".py",
                ".js",
                ".java",
                ".go",
            ]:
                relevant.append(str(file_path))

        # Add files that might import the target
        target_module = Path(target_file).stem
        for file_path in root_path.rglob("*"):
            if file_path.is_file() and file_path.suffix in [
                ".py",
                ".js",
                ".java",
                ".go",
            ]:
                try:
                    content = file_path.read_text()
                    if target_module in content:
                        relevant.append(str(file_path))
                except (IOError, UnicodeDecodeError):
                    pass

        return list(set(relevant))[:50]  # Limit to 50 files

    def _analyze_file_dependencies(self, file_path: str, content: str):
        """Analyze dependencies in a single file."""
        language = self._detect_language(file_path)

        if language == "python":
            self._analyze_python_dependencies(file_path, content)
        elif language in ["javascript", "typescript"]:
            self._analyze_javascript_dependencies(file_path, content)
        elif language == "java":
            self._analyze_java_dependencies(file_path, content)

    def _analyze_python_dependencies(self, file_path: str, content: str):
        """Analyze Python file dependencies."""
        try:
            tree = ast.parse(content)

            # Add module node
            module_name = Path(file_path).stem
            self.dependency_graph.add_node(
                f"{file_path}:module:{module_name}",
                data=DependencyNode(
                    name=module_name,
                    file_path=file_path,
                    node_type="module",
                    line_number=1,
                ),
            )

            # Analyze imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self._add_import_dependency(file_path, alias.name, node.lineno)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for alias in node.names:
                        self._add_import_dependency(
                            file_path, f"{module}.{alias.name}", node.lineno
                        )
                elif isinstance(node, ast.FunctionDef):
                    self._add_function_node(file_path, node.name, node.lineno)
                elif isinstance(node, ast.ClassDef):
                    self._add_class_node(
                        file_path,
                        node.name,
                        node.lineno,
                        [base.id for base in node.bases if isinstance(base, ast.Name)],
                    )
        except Exception as e:
            logger.warning(f"Error parsing Python file {file_path}: {e}")

    def _analyze_javascript_dependencies(self, file_path: str, content: str):
        """Analyze JavaScript/TypeScript file dependencies."""
        # Simplified regex-based analysis

        # Find imports
        import_pattern = r'import\s+(?:{[^}]+}|[\w\s,]+)\s+from\s+[\'"]([^\'"]+)[\'"]'
        for match in re.finditer(import_pattern, content):
            self._add_import_dependency(
                file_path, match.group(1), content[: match.start()].count("\n") + 1
            )

        # Find function declarations
        func_pattern = r"(?:function\s+(\w+)|const\s+(\w+)\s*=\s*(?:async\s+)?(?:\([^)]*\)|[^=]*)=>)"
        for match in re.finditer(func_pattern, content):
            func_name = match.group(1) or match.group(2)
            if func_name:
                self._add_function_node(
                    file_path, func_name, content[: match.start()].count("\n") + 1
                )

        # Find class declarations
        class_pattern = r"class\s+(\w+)(?:\s+extends\s+(\w+))?"
        for match in re.finditer(class_pattern, content):
            class_name = match.group(1)
            base_class = match.group(2)
            self._add_class_node(
                file_path,
                class_name,
                content[: match.start()].count("\n") + 1,
                [base_class] if base_class else [],
            )

    def _analyze_java_dependencies(self, file_path: str, content: str):
        """Analyze Java file dependencies."""
        # Simplified regex-based analysis

        # Find imports
        import_pattern = r"import\s+(?:static\s+)?([a-zA-Z0-9_.]+);"
        for match in re.finditer(import_pattern, content):
            self._add_import_dependency(
                file_path, match.group(1), content[: match.start()].count("\n") + 1
            )

        # Find class declarations
        class_pattern = r"(?:public\s+)?(?:abstract\s+)?(?:final\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?(?:\s+implements\s+([^{]+))?"
        for match in re.finditer(class_pattern, content):
            class_name = match.group(1)
            extends = match.group(2)
            implements = match.group(3)

            bases = []
            if extends:
                bases.append(extends)
            if implements:
                bases.extend(impl.strip() for impl in implements.split(","))

            self._add_class_node(
                file_path, class_name, content[: match.start()].count("\n") + 1, bases
            )

        # Find method declarations
        method_pattern = r"(?:public\s+|private\s+|protected\s+)?(?:static\s+)?(?:final\s+)?(?:synchronized\s+)?(?:\w+(?:<[^>]+>)?)\s+(\w+)\s*\([^)]*\)"
        for match in re.finditer(method_pattern, content):
            method_name = match.group(1)
            if method_name not in ["if", "for", "while", "switch", "catch"]:
                self._add_function_node(
                    file_path, method_name, content[: match.start()].count("\n") + 1
                )

    def _add_import_dependency(
        self, file_path: str, import_name: str, line_number: int
    ):
        """Add an import dependency to the graph."""
        source_node = f"{file_path}:module:{Path(file_path).stem}"
        target_node = f"external:module:{import_name}"

        self.dependency_graph.add_node(
            target_node,
            data=DependencyNode(
                name=import_name,
                file_path="external",
                node_type="module",
                line_number=0,
            ),
        )

        self.dependency_graph.add_edge(
            source_node,
            target_node,
            data=DependencyEdge(
                source=source_node,
                target=target_node,
                edge_type="imports",
                metadata={"line": line_number},
            ),
        )

    def _add_function_node(self, file_path: str, func_name: str, line_number: int):
        """Add a function node to the graph."""
        node_id = f"{file_path}:function:{func_name}"

        self.dependency_graph.add_node(
            node_id,
            data=DependencyNode(
                name=func_name,
                file_path=file_path,
                node_type="function",
                line_number=line_number,
            ),
        )

        # Link to module
        module_node = f"{file_path}:module:{Path(file_path).stem}"
        if module_node in self.dependency_graph:
            self.dependency_graph.add_edge(
                module_node,
                node_id,
                data=DependencyEdge(
                    source=module_node, target=node_id, edge_type="contains"
                ),
            )

    def _add_class_node(
        self, file_path: str, class_name: str, line_number: int, base_classes: List[str]
    ):
        """Add a class node to the graph."""
        node_id = f"{file_path}:class:{class_name}"

        self.dependency_graph.add_node(
            node_id,
            data=DependencyNode(
                name=class_name,
                file_path=file_path,
                node_type="class",
                line_number=line_number,
                metadata={"base_classes": base_classes},
            ),
        )

        # Link to module
        module_node = f"{file_path}:module:{Path(file_path).stem}"
        if module_node in self.dependency_graph:
            self.dependency_graph.add_edge(
                module_node,
                node_id,
                data=DependencyEdge(
                    source=module_node, target=node_id, edge_type="contains"
                ),
            )

        # Add inheritance edges
        for base in base_classes:
            base_node = f"external:class:{base}"
            self.dependency_graph.add_node(
                base_node,
                data=DependencyNode(
                    name=base, file_path="external", node_type="class", line_number=0
                ),
            )
            self.dependency_graph.add_edge(
                node_id,
                base_node,
                data=DependencyEdge(
                    source=node_id, target=base_node, edge_type="inherits"
                ),
            )

    def _analyze_module_structure(self, root_dir: str) -> Dict[str, Any]:
        """Analyze the module structure of the project."""
        structure = {"packages": [], "modules": [], "standalone_files": []}

        root_path = Path(root_dir)

        # Find packages (directories with __init__.py)
        for path in root_path.rglob("__init__.py"):
            package_dir = path.parent
            structure["packages"].append(
                {
                    "name": package_dir.name,
                    "path": str(package_dir),
                    "submodules": [
                        f.stem
                        for f in package_dir.glob("*.py")
                        if f.name != "__init__.py"
                    ],
                }
            )

        # Find standalone modules
        for py_file in root_path.glob("*.py"):
            if py_file.name != "__init__.py":
                structure["standalone_files"].append(
                    {"name": py_file.stem, "path": str(py_file)}
                )

        return structure

    def _find_entry_points(self, root_dir: str) -> List[Dict[str, Any]]:
        """Find entry points in the project."""
        entry_points = []

        # Common entry point patterns
        entry_patterns = [
            'if __name__ == "__main__"',
            "def main(",
            "class Main",
            "@app.route",  # Flask
            "urlpatterns",  # Django
            "async def main",
        ]

        for file_path in Path(root_dir).rglob("*.py"):
            try:
                content = file_path.read_text()
                for pattern in entry_patterns:
                    if pattern in content:
                        entry_points.append(
                            {
                                "file": str(file_path),
                                "type": pattern,
                                "line": content[: content.find(pattern)].count("\n")
                                + 1,
                            }
                        )
                        break
            except (re.error, AttributeError):
                pass

        # Look for common entry point files
        common_entries = ["main.py", "app.py", "run.py", "manage.py", "__main__.py"]
        for entry_file in common_entries:
            for file_path in Path(root_dir).rglob(entry_file):
                if not any(ep["file"] == str(file_path) for ep in entry_points):
                    entry_points.append(
                        {"file": str(file_path), "type": "filename", "line": 1}
                    )

        return entry_points

    def _analyze_test_coverage(self, target_file: str, root_dir: str) -> Dict[str, Any]:
        """Analyze test coverage for the target file."""
        coverage = {
            "has_tests": False,
            "test_files": [],
            "test_functions": [],
            "coverage_percentage": 0.0,
        }

        # Find test files
        target_name = Path(target_file).stem
        test_patterns = [
            f"test_{target_name}.py",
            f"{target_name}_test.py",
            f"tests/test_{target_name}.py",
            f"tests/{target_name}_test.py",
        ]

        root_path = Path(root_dir)
        for pattern in test_patterns:
            for test_file in root_path.rglob(pattern):
                coverage["has_tests"] = True
                coverage["test_files"].append(str(test_file))

                # Find test functions
                try:
                    content = test_file.read_text()
                    test_funcs = re.findall(r"def\s+(test_\w+)", content)
                    coverage["test_functions"].extend(test_funcs)
                except (IOError, UnicodeDecodeError, re.error):
                    pass

        # Try to get actual coverage data
        try:
            result = subprocess.run(
                ["coverage", "report", "--include", target_file],
                capture_output=True,
                text=True,
                cwd=root_dir,
            )
            if result.returncode == 0:
                # Parse coverage output
                for line in result.stdout.split("\n"):
                    if target_file in line:
                        parts = line.split()
                        if len(parts) >= 4 and parts[-1].endswith("%"):
                            coverage["coverage_percentage"] = float(parts[-1][:-1])
        except (subprocess.SubprocessError, ValueError, IndexError):
            pass

        return coverage

    def _identify_critical_paths(self, target_file: str) -> List[List[str]]:
        """Identify critical execution paths involving the target file."""
        critical_paths = []

        # Find paths from entry points to target file
        target_nodes = [
            node
            for node in self.dependency_graph.nodes()
            if node.startswith(f"{target_file}:")
        ]

        entry_nodes = [
            node
            for node in self.dependency_graph.nodes()
            if "main" in node.lower() or "entry" in node.lower()
        ]

        for entry in entry_nodes:
            for target in target_nodes:
                try:
                    paths = list(
                        nx.all_simple_paths(
                            self.dependency_graph, entry, target, cutoff=self.max_depth
                        )
                    )
                    critical_paths.extend(paths[:3])  # Limit to 3 paths per pair
                except nx.NetworkXNoPath:
                    pass

        return critical_paths[:10]  # Limit total paths

    def _calculate_file_metrics(self, file_path: str) -> Dict[str, Any]:
        """Calculate metrics for a file."""
        metrics = {
            "lines_of_code": 0,
            "complexity": 0,
            "dependencies_in": 0,
            "dependencies_out": 0,
            "cohesion": 0.0,
        }

        try:
            content = Path(file_path).read_text()
            metrics["lines_of_code"] = len(content.split("\n"))

            # Count dependencies
            file_nodes = [
                node
                for node in self.dependency_graph.nodes()
                if node.startswith(f"{file_path}:")
            ]

            for node in file_nodes:
                metrics["dependencies_in"] += self.dependency_graph.in_degree(node)
                metrics["dependencies_out"] += self.dependency_graph.out_degree(node)

            # Simple complexity estimation
            metrics["complexity"] = (
                content.count("if ")
                + content.count("for ")
                + content.count("while ")
                + content.count("except:")
                + content.count("elif ")
            )

        except Exception:
            pass

        return metrics

    def _serialize_graph(self) -> Dict[str, Any]:
        """Serialize the dependency graph for output."""
        return {
            "nodes": [
                {
                    "id": node,
                    "data": self.dependency_graph.nodes[node].get("data", {}).__dict__,
                }
                for node in self.dependency_graph.nodes()
            ],
            "edges": [
                {
                    "source": u,
                    "target": v,
                    "data": self.dependency_graph.edges[u, v].get("data", {}).__dict__,
                }
                for u, v in self.dependency_graph.edges()
            ],
        }

    def _find_direct_dependencies(
        self, file_path: str, entity_name: str, entity_type: str
    ) -> List[DependencyNode]:
        """Find direct dependencies of an entity."""
        dependencies = []
        entity_node = f"{file_path}:{entity_type}:{entity_name}"

        if entity_node in self.dependency_graph:
            # Find all nodes that depend on this entity
            for node in self.dependency_graph.predecessors(entity_node):
                if "data" in self.dependency_graph.nodes[node]:
                    dependencies.append(self.dependency_graph.nodes[node]["data"])

        return dependencies

    def _find_indirect_dependencies(
        self, direct_deps: List[DependencyNode], max_depth: int
    ) -> List[DependencyNode]:
        """Find indirect dependencies up to max_depth."""
        indirect = []
        visited = set()

        def traverse(node_id: str, depth: int):
            if depth >= max_depth or node_id in visited:
                return

            visited.add(node_id)

            if node_id in self.dependency_graph:
                for pred in self.dependency_graph.predecessors(node_id):
                    if (
                        pred not in visited
                        and "data" in self.dependency_graph.nodes[pred]
                    ):
                        indirect.append(self.dependency_graph.nodes[pred]["data"])
                        traverse(pred, depth + 1)

        for dep in direct_deps:
            node_id = f"{dep.file_path}:{dep.node_type}:{dep.name}"
            traverse(node_id, 1)

        return indirect

    def _find_affected_tests(
        self, file_path: str, entity_name: str, all_impacts: List[DependencyNode]
    ) -> List[str]:
        """Find tests affected by changes."""
        affected_tests = []

        for impact in all_impacts:
            if "test" in impact.file_path.lower():
                affected_tests.append(impact.file_path)

        # Look for tests that import the changed file
        for node in self.dependency_graph.nodes():
            if "test" in node.lower() and file_path in str(
                self.dependency_graph.nodes[node]
            ):
                parts = node.split(":")
                if len(parts) > 0:
                    affected_tests.append(parts[0])

        return list(set(affected_tests))

    def _assess_risk_level(
        self,
        direct_impacts: List[DependencyNode],
        indirect_impacts: List[DependencyNode],
        change_details: Dict[str, Any],
    ) -> str:
        """Assess the risk level of a change."""
        # Count impacts
        total_impacts = len(direct_impacts) + len(indirect_impacts)

        # Check for critical indicators
        has_interface_change = change_details.get("interface_change", False)
        has_breaking_change = change_details.get("breaking_change", False)
        affects_public_api = any(not dep.name.startswith("_") for dep in direct_impacts)

        # Determine risk level
        if has_breaking_change or total_impacts > 20:
            return "high"
        elif has_interface_change or affects_public_api or total_impacts > 10:
            return "medium"
        else:
            return "low"

    def _identify_breaking_changes(
        self,
        change_type: str,
        change_details: Dict[str, Any],
        direct_impacts: List[DependencyNode],
    ) -> List[Dict[str, Any]]:
        """Identify potential breaking changes."""
        breaking_changes = []

        # Check for signature changes
        if change_type in ["function", "method"] and change_details.get(
            "signature_change"
        ):
            breaking_changes.append(
                {
                    "type": "signature_change",
                    "description": "Function/method signature changed",
                    "affected": [dep.name for dep in direct_impacts],
                }
            )

        # Check for removed entities
        if change_details.get("removed"):
            breaking_changes.append(
                {
                    "type": "removal",
                    "description": f"{change_type} removed",
                    "affected": [dep.name for dep in direct_impacts],
                }
            )

        # Check for type changes
        if change_details.get("type_change"):
            breaking_changes.append(
                {
                    "type": "type_change",
                    "description": "Return type or parameter type changed",
                    "affected": [dep.name for dep in direct_impacts],
                }
            )

        return breaking_changes

    def _generate_validation_suggestions(
        self, change_type: str, risk_level: str, affected_tests: List[str]
    ) -> List[str]:
        """Generate validation suggestions based on the change."""
        suggestions = []

        # Basic suggestions
        suggestions.append(f"Run all affected tests: {len(affected_tests)} test files")

        if risk_level == "high":
            suggestions.extend(
                [
                    "Perform full regression testing",
                    "Review all dependent modules for compatibility",
                    "Consider creating a feature flag for gradual rollout",
                    "Update API documentation if public interfaces changed",
                ]
            )
        elif risk_level == "medium":
            suggestions.extend(
                [
                    "Run integration tests for affected modules",
                    "Review direct dependencies for compatibility",
                    "Update relevant documentation",
                ]
            )
        else:
            suggestions.extend(
                [
                    "Run unit tests for the changed module",
                    "Perform smoke testing of main functionality",
                ]
            )

        # Type-specific suggestions
        if change_type == "class":
            suggestions.append("Verify inheritance hierarchy is maintained")
        elif change_type == "function":
            suggestions.append("Check all call sites for compatibility")

        return suggestions

    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension."""
        extension = Path(file_path).suffix.lower()
        language_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".java": "java",
            ".go": "go",
            ".rb": "ruby",
            ".php": "php",
            ".cs": "csharp",
            ".cpp": "cpp",
            ".c": "c",
            ".rs": "rust",
            ".swift": "swift",
            ".kt": "kotlin",
            ".scala": "scala",
            ".r": "r",
            ".jl": "julia",
        }
        return language_map.get(extension, "unknown")

    def _build_python_call_graph(self, content: str, file_path: str) -> nx.DiGraph:
        """Build call graph for Python code."""
        graph = nx.DiGraph()

        try:
            tree = ast.parse(content)

            class CallGraphVisitor(ast.NodeVisitor):
                def __init__(self, graph):
                    self.graph = graph
                    self.current_function = None
                    self.current_class = None

                def visit_ClassDef(self, node):
                    old_class = self.current_class
                    self.current_class = node.name
                    self.graph.add_node(node.name, type="class", line=node.lineno)
                    self.generic_visit(node)
                    self.current_class = old_class

                def visit_FunctionDef(self, node):
                    old_function = self.current_function

                    if self.current_class:
                        func_name = f"{self.current_class}.{node.name}"
                    else:
                        func_name = node.name

                    self.current_function = func_name
                    self.graph.add_node(func_name, type="function", line=node.lineno)

                    # If inside a class, add edge from class to method
                    if self.current_class:
                        self.graph.add_edge(
                            self.current_class, func_name, type="contains"
                        )

                    self.generic_visit(node)
                    self.current_function = old_function

                def visit_Call(self, node):
                    if self.current_function:
                        called_name = None

                        if isinstance(node.func, ast.Name):
                            called_name = node.func.id
                        elif isinstance(node.func, ast.Attribute):
                            if isinstance(node.func.value, ast.Name):
                                called_name = f"{node.func.value.id}.{node.func.attr}"
                            else:
                                called_name = node.func.attr

                        if called_name:
                            self.graph.add_edge(
                                self.current_function,
                                called_name,
                                type="calls",
                                line=node.lineno,
                            )

                    self.generic_visit(node)

            visitor = CallGraphVisitor(graph)
            visitor.visit(tree)

        except Exception as e:
            logger.warning(f"Error building Python call graph: {e}")

        return graph

    def _build_javascript_call_graph(self, content: str, file_path: str) -> nx.DiGraph:
        """Build call graph for JavaScript/TypeScript code."""
        graph = nx.DiGraph()

        # Function declarations
        func_pattern = r"(?:function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?(?:\([^)]*\)|[^=]*)=>)"
        for match in re.finditer(func_pattern, content):
            func_name = match.group(1) or match.group(2)
            if func_name:
                line = content[: match.start()].count("\n") + 1
                graph.add_node(func_name, type="function", line=line)

        # Class declarations
        class_pattern = r"class\s+(\w+)"
        for match in re.finditer(class_pattern, content):
            class_name = match.group(1)
            line = content[: match.start()].count("\n") + 1
            graph.add_node(class_name, type="class", line=line)

        # Simple call detection
        call_pattern = r"(\w+)\s*\("
        current_function = None

        lines = content.split("\n")
        for i, line in enumerate(lines):
            # Detect current function context
            func_match = re.match(
                r"(?:function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=)", line
            )
            if func_match:
                current_function = func_match.group(1) or func_match.group(2)

            # Find function calls
            for match in re.finditer(call_pattern, line):
                called = match.group(1)
                if current_function and called != current_function:
                    graph.add_edge(current_function, called, type="calls", line=i + 1)

        return graph

    def _build_java_call_graph(self, content: str, file_path: str) -> nx.DiGraph:
        """Build call graph for Java code."""
        graph = nx.DiGraph()

        # Class declarations
        class_pattern = r"(?:public\s+)?(?:abstract\s+)?(?:final\s+)?class\s+(\w+)"
        for match in re.finditer(class_pattern, content):
            class_name = match.group(1)
            line = content[: match.start()].count("\n") + 1
            graph.add_node(class_name, type="class", line=line)

        # Method declarations
        method_pattern = r"(?:public\s+|private\s+|protected\s+)?(?:static\s+)?(?:final\s+)?(?:\w+(?:<[^>]+>)?)\s+(\w+)\s*\([^)]*\)\s*(?:throws\s+[^{]+)?\s*\{"
        current_class = None
        current_method = None

        lines = content.split("\n")
        brace_count = 0

        for i, line in enumerate(lines):
            # Track class context
            class_match = re.search(class_pattern, line)
            if class_match:
                current_class = class_match.group(1)
                brace_count = 0

            # Track method context
            method_match = re.search(method_pattern, line)
            if method_match:
                method_name = method_match.group(1)
                if current_class:
                    full_method_name = f"{current_class}.{method_name}"
                else:
                    full_method_name = method_name

                current_method = full_method_name
                graph.add_node(full_method_name, type="method", line=i + 1)

                if current_class:
                    graph.add_edge(current_class, full_method_name, type="contains")

            # Track braces to determine scope
            brace_count += line.count("{") - line.count("}")

            # Find method calls (simplified)
            if current_method:
                call_pattern = r"(\w+)\s*\("
                for match in re.finditer(call_pattern, line):
                    called = match.group(1)
                    if called not in ["if", "for", "while", "switch", "catch", "new"]:
                        graph.add_edge(current_method, called, type="calls", line=i + 1)

        return graph

    def _analyze_python_data_flow(self, content: str) -> Dict[str, Any]:
        """Analyze data flow in Python code."""
        data_flow = {
            "variables": {},
            "parameters": {},
            "returns": {},
            "global_access": [],
            "side_effects": [],
        }

        try:
            tree = ast.parse(content)

            class DataFlowVisitor(ast.NodeVisitor):
                def __init__(self, data_flow):
                    self.data_flow = data_flow
                    self.current_function = None

                def visit_FunctionDef(self, node):
                    old_function = self.current_function
                    self.current_function = node.name

                    # Track parameters
                    params = []
                    for arg in node.args.args:
                        params.append(arg.arg)

                    self.data_flow["parameters"][node.name] = params

                    # Visit function body
                    self.generic_visit(node)
                    self.current_function = old_function

                def visit_Assign(self, node):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            var_name = target.id
                            scope = self.current_function or "module"

                            if scope not in self.data_flow["variables"]:
                                self.data_flow["variables"][scope] = []

                            self.data_flow["variables"][scope].append(
                                {"name": var_name, "line": node.lineno}
                            )

                    self.generic_visit(node)

                def visit_Return(self, node):
                    if self.current_function:
                        if self.current_function not in self.data_flow["returns"]:
                            self.data_flow["returns"][self.current_function] = []

                        # Simple representation of return value
                        return_type = "expression"
                        if isinstance(node.value, ast.Name):
                            return_type = f"variable:{node.value.id}"
                        elif isinstance(node.value, ast.Constant):
                            return_type = f"constant:{type(node.value.value).__name__}"

                        self.data_flow["returns"][self.current_function].append(
                            {"type": return_type, "line": node.lineno}
                        )

                    self.generic_visit(node)

                def visit_Global(self, node):
                    for name in node.names:
                        self.data_flow["global_access"].append(
                            {
                                "variable": name,
                                "function": self.current_function,
                                "line": node.lineno,
                            }
                        )

                    self.generic_visit(node)

            visitor = DataFlowVisitor(data_flow)
            visitor.visit(tree)

        except Exception as e:
            logger.warning(f"Error analyzing Python data flow: {e}")

        return data_flow

    def _analyze_javascript_data_flow(self, content: str) -> Dict[str, Any]:
        """Analyze data flow in JavaScript/TypeScript code."""
        data_flow = {
            "variables": {},
            "parameters": {},
            "returns": {},
            "global_access": [],
            "side_effects": [],
        }

        # Variable declarations
        var_patterns = [
            r"(?:const|let|var)\s+(\w+)",
            r"(?:const|let|var)\s+\{([^}]+)\}",  # Destructuring
            r"(?:const|let|var)\s+\[([^\]]+)\]",  # Array destructuring
        ]

        for pattern in var_patterns:
            for match in re.finditer(pattern, content):
                line = content[: match.start()].count("\n") + 1
                var_names = re.findall(r"\w+", match.group(1))

                for var_name in var_names:
                    if "module" not in data_flow["variables"]:
                        data_flow["variables"]["module"] = []

                    data_flow["variables"]["module"].append(
                        {"name": var_name, "line": line}
                    )

        # Function parameters
        func_pattern = r"(?:function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?)?\s*\(([^)]*)\)"
        for match in re.finditer(func_pattern, content):
            func_name = match.group(1) or match.group(2)
            if func_name and match.group(3):
                params = [p.strip() for p in match.group(3).split(",") if p.strip()]
                data_flow["parameters"][func_name] = params

        # Return statements
        return_pattern = r"return\s+([^;]+);"
        for match in re.finditer(return_pattern, content):
            line = content[: match.start()].count("\n") + 1
            # Find containing function (simplified)
            preceding = content[: match.start()]
            func_matches = list(re.finditer(func_pattern, preceding))
            if func_matches:
                last_func = func_matches[-1]
                func_name = last_func.group(1) or last_func.group(2)
                if func_name:
                    if func_name not in data_flow["returns"]:
                        data_flow["returns"][func_name] = []
                    data_flow["returns"][func_name].append(
                        {"type": "expression", "line": line}
                    )

        return data_flow

    def _analyze_generic_data_flow(self, content: str, language: str) -> Dict[str, Any]:
        """Generic data flow analysis for unsupported languages."""
        return {
            "variables": {},
            "parameters": {},
            "returns": {},
            "global_access": [],
            "side_effects": [],
            "language": language,
            "analysis": "generic",
        }

    def _find_entity_usages(
        self, content: str, entity_name: str, entity_type: str, file_path: str
    ) -> List[Dict[str, Any]]:
        """Find usages of an entity in code."""
        usages = []

        # Pattern based on entity type
        if entity_type == "function":
            patterns = [
                rf"{entity_name}\s*\(",  # Function call
                rf"\.{entity_name}\s*\(",  # Method call
                rf"=\s*{entity_name}(?:\s|$)",  # Assignment
            ]
        elif entity_type == "class":
            patterns = [
                rf"new\s+{entity_name}\s*\(",  # Instantiation
                rf"extends\s+{entity_name}",  # Inheritance
                rf":\s*{entity_name}",  # Type annotation
                rf"instanceof\s+{entity_name}",  # Type check
            ]
        else:  # variable
            patterns = [
                rf"\b{entity_name}\b",  # Any reference
            ]

        for pattern in patterns:
            for match in re.finditer(pattern, content):
                line_no = content[: match.start()].count("\n") + 1
                line_content = content.split("\n")[line_no - 1].strip()

                usages.append(
                    {
                        "file": file_path,
                        "line": line_no,
                        "type": entity_type,
                        "pattern": pattern,
                        "context": line_content,
                    }
                )

        return usages


def create_contextual_analyzer(
    config: Optional[Dict[str, Any]] = None,
) -> ContextualAnalyzer:
    """
    Factory function to create a contextual analyzer.

    Args:
        config: Configuration dictionary

    Returns:
        Configured contextual analyzer
    """
    return ContextualAnalyzer(config)
