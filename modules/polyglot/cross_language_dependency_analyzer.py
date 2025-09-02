"""
Cross-language dependency analysis for polyglot applications.
Analyzes dependencies across different programming languages and frameworks.
"""

import asyncio
import json
import logging
import re
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from enum import Enum

from ..analysis.language_plugin_system import LanguagePluginRegistry


class DependencyType(Enum):
    """Types of dependencies between services."""
    API_CALL = "api_call"
    DATABASE = "database"
    MESSAGE_QUEUE = "message_queue"
    SHARED_LIBRARY = "shared_library"
    CONFIGURATION = "configuration"
    PROTOCOL_BUFFER = "protocol_buffer"
    GRAPHQL_SCHEMA = "graphql_schema"
    EVENT_STREAM = "event_stream"


class APIProtocol(Enum):
    """API protocols used for communication."""
    REST = "rest"
    GRPC = "grpc"
    GRAPHQL = "graphql"
    WEBSOCKET = "websocket"
    THRIFT = "thrift"
    SOAP = "soap"


@dataclass
class Dependency:
    """Represents a dependency between components."""
    source_component: str
    target_component: str
    dependency_type: DependencyType
    source_language: str
    target_language: str
    protocol: Optional[APIProtocol] = None
    version_constraint: Optional[str] = None
    is_critical: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class APIContract:
    """Represents an API contract between services."""
    service_name: str
    endpoint: str
    method: str
    request_schema: Dict[str, Any]
    response_schema: Dict[str, Any]
    protocol: APIProtocol
    version: str
    language: str


@dataclass
class SharedDataStructure:
    """Represents a data structure shared across languages."""
    name: str
    definition_language: str
    consuming_languages: List[str]
    schema: Dict[str, Any]
    serialization_format: str  # json, protobuf, avro, etc.
    version: str


@dataclass
class DependencyGraph:
    """Dependency graph for the entire system."""
    nodes: Dict[str, Dict[str, Any]]  # component_id -> component_info
    edges: List[Dependency]
    api_contracts: Dict[str, APIContract]
    shared_structures: Dict[str, SharedDataStructure]
    analysis_timestamp: datetime = field(default_factory=datetime.now)


class CrossLanguageDependencyAnalyzer:
    """
    Analyzes dependencies across different programming languages.
    Builds a comprehensive dependency graph for polyglot systems.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.plugin_registry = LanguagePluginRegistry()
        self.dependency_graph = DependencyGraph(
            nodes={},
            edges=[],
            api_contracts={},
            shared_structures={}
        )
        
        # Language-specific analyzers
        self.language_analyzers = {
            'python': self._analyze_python_dependencies,
            'javascript': self._analyze_javascript_dependencies,
            'java': self._analyze_java_dependencies,
            'go': self._analyze_go_dependencies,
            'rust': self._analyze_rust_dependencies,
            'csharp': self._analyze_csharp_dependencies,
            'ruby': self._analyze_ruby_dependencies,
            'php': self._analyze_php_dependencies,
        }
        
    async def analyze_project(self, project_root: Path) -> DependencyGraph:
        """
        Analyze an entire project for cross-language dependencies.
        """
        self.logger.info(f"Starting cross-language dependency analysis for: {project_root}")
        
        # Discover all components
        components = await self._discover_components(project_root)
        
        # Analyze each component
        tasks = []
        for component in components:
            task = self._analyze_component(component)
            tasks.append(task)
            
        await asyncio.gather(*tasks)
        
        # Analyze cross-component dependencies
        await self._analyze_cross_dependencies()
        
        # Detect shared data structures
        await self._detect_shared_structures()
        
        # Validate dependency graph
        self._validate_dependency_graph()
        
        return self.dependency_graph
        
    async def _discover_components(self, project_root: Path) -> List[Dict[str, Any]]:
        """Discover all components in the project."""
        components = []
        
        # Look for common project markers
        markers = {
            'python': ['requirements.txt', 'setup.py', 'pyproject.toml', 'Pipfile'],
            'javascript': ['package.json'],
            'java': ['pom.xml', 'build.gradle', 'build.gradle.kts'],
            'go': ['go.mod'],
            'rust': ['Cargo.toml'],
            'csharp': ['*.csproj', '*.sln'],
            'ruby': ['Gemfile'],
            'php': ['composer.json'],
        }
        
        for path in project_root.rglob('*'):
            if path.is_file():
                for language, marker_files in markers.items():
                    if any(path.name == marker or path.match(marker) for marker in marker_files):
                        component_dir = path.parent
                        component = {
                            'path': component_dir,
                            'language': language,
                            'name': component_dir.name,
                            'marker_file': path.name
                        }
                        if component not in components:
                            components.append(component)
                            self.logger.info(
                                f"Discovered {language} component: {component['name']} "
                                f"at {component_dir}"
                            )
                            
        return components
        
    async def _analyze_component(self, component: Dict[str, Any]) -> None:
        """Analyze a single component for dependencies."""
        language = component['language']
        
        # Add component to graph
        component_id = f"{component['name']}_{language}"
        self.dependency_graph.nodes[component_id] = {
            'name': component['name'],
            'language': language,
            'path': str(component['path']),
            'dependencies': []
        }
        
        # Use language-specific analyzer
        if language in self.language_analyzers:
            analyzer = self.language_analyzers[language]
            await analyzer(component)
        else:
            self.logger.warning(f"No analyzer for language: {language}")
            
    async def _analyze_python_dependencies(self, component: Dict[str, Any]) -> None:
        """Analyze Python component dependencies."""
        component_path = Path(component['path'])
        component_id = f"{component['name']}_python"
        
        # Check for FastAPI/Flask REST endpoints
        for py_file in component_path.rglob('*.py'):
            content = py_file.read_text()
            
            # FastAPI endpoints
            fastapi_endpoints = re.findall(
                r'@app\.(get|post|put|delete|patch)\s*\(\s*["\']([^"\']+)["\']',
                content
            )
            for method, endpoint in fastapi_endpoints:
                self._add_api_endpoint(component_id, endpoint, method.upper(), 'python')
                
            # Flask endpoints
            flask_endpoints = re.findall(
                r'@app\.route\s*\(\s*["\']([^"\']+)["\'].*methods\s*=\s*\[([^\]]+)\]',
                content
            )
            for endpoint, methods in flask_endpoints:
                for method in re.findall(r'["\'](\w+)["\']', methods):
                    self._add_api_endpoint(component_id, endpoint, method, 'python')
                    
            # HTTP client calls
            http_calls = re.findall(
                r'requests\.(get|post|put|delete|patch)\s*\(\s*["\']([^"\']+)["\']',
                content
            )
            for method, url in http_calls:
                self._add_outgoing_api_call(component_id, url, method.upper())
                
            # gRPC imports
            if 'import grpc' in content or 'from grpc' in content:
                self._add_protocol_usage(component_id, APIProtocol.GRPC)
                
            # Database connections
            if 'sqlalchemy' in content or 'psycopg2' in content or 'pymongo' in content:
                self._add_database_dependency(component_id, 'python')
                
    async def _analyze_javascript_dependencies(self, component: Dict[str, Any]) -> None:
        """Analyze JavaScript/TypeScript component dependencies."""
        component_path = Path(component['path'])
        component_id = f"{component['name']}_javascript"
        
        # Read package.json
        package_json_path = component_path / 'package.json'
        if package_json_path.exists():
            package_data = json.loads(package_json_path.read_text())
            
            # Check for Express/Fastify/Koa
            deps = {**package_data.get('dependencies', {}), 
                   **package_data.get('devDependencies', {})}
            
            if 'express' in deps or '@types/express' in deps:
                # Look for Express routes
                for js_file in component_path.rglob('*.js'):
                    content = js_file.read_text()
                    express_routes = re.findall(
                        r'app\.(get|post|put|delete|patch)\s*\(\s*["\']([^"\']+)["\']',
                        content
                    )
                    for method, route in express_routes:
                        self._add_api_endpoint(component_id, route, method.upper(), 'javascript')
                        
            # Check for HTTP client usage
            if 'axios' in deps or 'fetch' in deps or 'request' in deps:
                for js_file in component_path.rglob('*.js'):
                    content = js_file.read_text()
                    # Axios calls
                    axios_calls = re.findall(
                        r'axios\.(get|post|put|delete|patch)\s*\(\s*["\']([^"\']+)["\']',
                        content
                    )
                    for method, url in axios_calls:
                        self._add_outgoing_api_call(component_id, url, method.upper())
                        
            # GraphQL
            if 'graphql' in deps or 'apollo-server' in deps:
                self._add_protocol_usage(component_id, APIProtocol.GRAPHQL)
                
            # Database clients
            if 'mongodb' in deps or 'mongoose' in deps:
                self._add_database_dependency(component_id, 'mongodb')
            if 'pg' in deps or 'mysql' in deps or 'mysql2' in deps:
                self._add_database_dependency(component_id, 'sql')
                
    async def _analyze_java_dependencies(self, component: Dict[str, Any]) -> None:
        """Analyze Java component dependencies."""
        component_path = Path(component['path'])
        component_id = f"{component['name']}_java"
        
        # Check for Spring Boot REST endpoints
        for java_file in component_path.rglob('*.java'):
            content = java_file.read_text()
            
            # Spring REST endpoints
            spring_endpoints = re.findall(
                r'@(GetMapping|PostMapping|PutMapping|DeleteMapping|PatchMapping|RequestMapping)\s*\(\s*["\']([^"\']+)["\']',
                content
            )
            for annotation, endpoint in spring_endpoints:
                method = annotation.replace('Mapping', '').upper()
                if method == 'REQUEST':
                    method = 'GET'  # Default for RequestMapping
                self._add_api_endpoint(component_id, endpoint, method, 'java')
                
            # RestTemplate or WebClient usage
            if 'RestTemplate' in content or 'WebClient' in content:
                rest_calls = re.findall(
                    r'\.(getForObject|postForObject|put|delete)\s*\(\s*["\']([^"\']+)["\']',
                    content
                )
                for method, url in rest_calls:
                    self._add_outgoing_api_call(component_id, url, method.upper())
                    
            # gRPC
            if 'import io.grpc' in content:
                self._add_protocol_usage(component_id, APIProtocol.GRPC)
                
            # Database
            if 'import javax.persistence' in content or 'import org.springframework.data' in content:
                self._add_database_dependency(component_id, 'jpa')
                
    async def _analyze_go_dependencies(self, component: Dict[str, Any]) -> None:
        """Analyze Go component dependencies."""
        component_path = Path(component['path'])
        component_id = f"{component['name']}_go"
        
        # Read go.mod for dependencies
        go_mod_path = component_path / 'go.mod'
        if go_mod_path.exists():
            content = go_mod_path.read_text()
            
            # Check for web frameworks
            if 'github.com/gin-gonic/gin' in content:
                # Look for Gin routes
                for go_file in component_path.rglob('*.go'):
                    file_content = go_file.read_text()
                    gin_routes = re.findall(
                        r'router\.(GET|POST|PUT|DELETE|PATCH)\s*\(\s*"([^"]+)"',
                        file_content
                    )
                    for method, route in gin_routes:
                        self._add_api_endpoint(component_id, route, method, 'go')
                        
            # gRPC
            if 'google.golang.org/grpc' in content:
                self._add_protocol_usage(component_id, APIProtocol.GRPC)
                
            # Database drivers
            if 'github.com/lib/pq' in content or 'github.com/go-sql-driver/mysql' in content:
                self._add_database_dependency(component_id, 'sql')
            if 'go.mongodb.org/mongo-driver' in content:
                self._add_database_dependency(component_id, 'mongodb')
                
    async def _analyze_rust_dependencies(self, component: Dict[str, Any]) -> None:
        """Analyze Rust component dependencies."""
        component_path = Path(component['path'])
        component_id = f"{component['name']}_rust"
        
        # Read Cargo.toml
        cargo_path = component_path / 'Cargo.toml'
        if cargo_path.exists():
            content = cargo_path.read_text()
            
            # Web frameworks
            if 'actix-web' in content or 'rocket' in content:
                # Would need to parse Rust source for routes
                pass
                
            # Database
            if 'diesel' in content or 'sqlx' in content:
                self._add_database_dependency(component_id, 'sql')
                
    async def _analyze_csharp_dependencies(self, component: Dict[str, Any]) -> None:
        """Analyze C# component dependencies."""
        component_path = Path(component['path'])
        component_id = f"{component['name']}_csharp"
        
        # Look for ASP.NET Core controllers
        for cs_file in component_path.rglob('*Controller.cs'):
            content = cs_file.read_text()
            
            # ASP.NET Core endpoints
            aspnet_endpoints = re.findall(
                r'\[(HttpGet|HttpPost|HttpPut|HttpDelete|HttpPatch)\s*\(\s*"([^"]+)"\s*\)\]',
                content
            )
            for method, endpoint in aspnet_endpoints:
                self._add_api_endpoint(
                    component_id, 
                    endpoint, 
                    method.replace('Http', '').upper(), 
                    'csharp'
                )
                
            # HttpClient usage
            if 'HttpClient' in content:
                http_calls = re.findall(
                    r'client\.(GetAsync|PostAsync|PutAsync|DeleteAsync)\s*\(\s*"([^"]+)"',
                    content
                )
                for method, url in http_calls:
                    self._add_outgoing_api_call(
                        component_id, 
                        url, 
                        method.replace('Async', '').upper()
                    )
                    
        # Entity Framework
        for cs_file in component_path.rglob('*.cs'):
            content = cs_file.read_text()
            if 'using Microsoft.EntityFrameworkCore' in content:
                self._add_database_dependency(component_id, 'ef_core')
                
    async def _analyze_ruby_dependencies(self, component: Dict[str, Any]) -> None:
        """Analyze Ruby component dependencies."""
        component_path = Path(component['path'])
        component_id = f"{component['name']}_ruby"
        
        # Read Gemfile
        gemfile_path = component_path / 'Gemfile'
        if gemfile_path.exists():
            content = gemfile_path.read_text()
            
            # Rails
            if 'rails' in content:
                # Look for Rails routes
                routes_file = component_path / 'config' / 'routes.rb'
                if routes_file.exists():
                    routes_content = routes_file.read_text()
                    # Simple pattern for Rails routes
                    rails_routes = re.findall(
                        r'(get|post|put|patch|delete)\s+["\']([^"\']+)["\']',
                        routes_content
                    )
                    for method, route in rails_routes:
                        self._add_api_endpoint(component_id, route, method.upper(), 'ruby')
                        
            # Database
            if 'pg' in content or 'mysql2' in content:
                self._add_database_dependency(component_id, 'sql')
                
    async def _analyze_php_dependencies(self, component: Dict[str, Any]) -> None:
        """Analyze PHP component dependencies."""
        component_path = Path(component['path'])
        component_id = f"{component['name']}_php"
        
        # Read composer.json
        composer_path = component_path / 'composer.json'
        if composer_path.exists():
            composer_data = json.loads(composer_path.read_text())
            requires = composer_data.get('require', {})
            
            # Laravel
            if 'laravel/framework' in requires:
                # Look for Laravel routes
                routes_path = component_path / 'routes' / 'api.php'
                if routes_path.exists():
                    content = routes_path.read_text()
                    laravel_routes = re.findall(
                        r'Route::(get|post|put|patch|delete)\s*\(\s*["\']([^"\']+)["\']',
                        content
                    )
                    for method, route in laravel_routes:
                        self._add_api_endpoint(component_id, route, method.upper(), 'php')
                        
            # Database
            if 'doctrine/orm' in requires:
                self._add_database_dependency(component_id, 'doctrine')
                
    async def _analyze_cross_dependencies(self) -> None:
        """Analyze dependencies between components."""
        # Match API calls to endpoints
        for node_id, node_data in self.dependency_graph.nodes.items():
            for call in node_data.get('outgoing_calls', []):
                # Find matching endpoint in other services
                for other_id, other_data in self.dependency_graph.nodes.items():
                    if node_id == other_id:
                        continue
                        
                    for endpoint in other_data.get('endpoints', []):
                        if self._matches_endpoint(call['url'], endpoint['path']):
                            dependency = Dependency(
                                source_component=node_id,
                                target_component=other_id,
                                dependency_type=DependencyType.API_CALL,
                                source_language=node_data['language'],
                                target_language=other_data['language'],
                                protocol=APIProtocol.REST,
                                metadata={
                                    'endpoint': endpoint['path'],
                                    'method': call['method']
                                }
                            )
                            self.dependency_graph.edges.append(dependency)
                            
    async def _detect_shared_structures(self) -> None:
        """Detect data structures shared across languages."""
        # Look for Protocol Buffer definitions
        for node_id, node_data in self.dependency_graph.nodes.items():
            path = Path(node_data['path'])
            
            # Protocol Buffers
            for proto_file in path.rglob('*.proto'):
                content = proto_file.read_text()
                messages = re.findall(r'message\s+(\w+)\s*{', content)
                
                for message_name in messages:
                    shared_struct = SharedDataStructure(
                        name=message_name,
                        definition_language='protobuf',
                        consuming_languages=self._find_proto_consumers(message_name),
                        schema={'type': 'protobuf', 'file': str(proto_file)},
                        serialization_format='protobuf',
                        version='1.0'
                    )
                    self.dependency_graph.shared_structures[message_name] = shared_struct
                    
            # OpenAPI/Swagger definitions
            for spec_file in path.rglob('*swagger*.json'):
                spec_data = json.loads(spec_file.read_text())
                if 'definitions' in spec_data:
                    for def_name, def_schema in spec_data['definitions'].items():
                        shared_struct = SharedDataStructure(
                            name=def_name,
                            definition_language='openapi',
                            consuming_languages=self._find_openapi_consumers(def_name),
                            schema=def_schema,
                            serialization_format='json',
                            version=spec_data.get('info', {}).get('version', '1.0')
                        )
                        self.dependency_graph.shared_structures[def_name] = shared_struct
                        
    def _add_api_endpoint(
        self, 
        component_id: str, 
        endpoint: str, 
        method: str, 
        language: str
    ) -> None:
        """Add an API endpoint to a component."""
        if 'endpoints' not in self.dependency_graph.nodes[component_id]:
            self.dependency_graph.nodes[component_id]['endpoints'] = []
            
        self.dependency_graph.nodes[component_id]['endpoints'].append({
            'path': endpoint,
            'method': method,
            'language': language
        })
        
    def _add_outgoing_api_call(
        self, 
        component_id: str, 
        url: str, 
        method: str
    ) -> None:
        """Add an outgoing API call from a component."""
        if 'outgoing_calls' not in self.dependency_graph.nodes[component_id]:
            self.dependency_graph.nodes[component_id]['outgoing_calls'] = []
            
        self.dependency_graph.nodes[component_id]['outgoing_calls'].append({
            'url': url,
            'method': method
        })
        
    def _add_protocol_usage(
        self, 
        component_id: str, 
        protocol: APIProtocol
    ) -> None:
        """Add protocol usage to a component."""
        if 'protocols' not in self.dependency_graph.nodes[component_id]:
            self.dependency_graph.nodes[component_id]['protocols'] = []
            
        if protocol.value not in self.dependency_graph.nodes[component_id]['protocols']:
            self.dependency_graph.nodes[component_id]['protocols'].append(protocol.value)
            
    def _add_database_dependency(
        self, 
        component_id: str, 
        db_type: str
    ) -> None:
        """Add database dependency to a component."""
        if 'databases' not in self.dependency_graph.nodes[component_id]:
            self.dependency_graph.nodes[component_id]['databases'] = []
            
        if db_type not in self.dependency_graph.nodes[component_id]['databases']:
            self.dependency_graph.nodes[component_id]['databases'].append(db_type)
            
    def _matches_endpoint(self, call_url: str, endpoint_path: str) -> bool:
        """Check if a call URL matches an endpoint path."""
        # Simple matching - could be enhanced with pattern matching
        call_path = call_url.split('/')[-1] if '/' in call_url else call_url
        endpoint_base = endpoint_path.split('/')[-1] if '/' in endpoint_path else endpoint_path
        
        # Remove parameters
        call_path = re.sub(r'\{[^}]+\}', '*', call_path)
        endpoint_base = re.sub(r'\{[^}]+\}', '*', endpoint_base)
        
        return call_path == endpoint_base
        
    def _find_proto_consumers(self, message_name: str) -> List[str]:
        """Find languages consuming a Protocol Buffer message."""
        consumers = []
        
        for node_id, node_data in self.dependency_graph.nodes.items():
            language = node_data['language']
            path = Path(node_data['path'])
            
            # Check for generated protobuf files
            patterns = {
                'python': f'*{message_name}_pb2.py',
                'javascript': f'*{message_name}_pb.js',
                'java': f'*{message_name}.java',
                'go': f'*{message_name}.pb.go',
                'csharp': f'*{message_name}.cs',
            }
            
            if language in patterns:
                pattern = patterns[language]
                if list(path.rglob(pattern)):
                    consumers.append(language)
                    
        return consumers
        
    def _find_openapi_consumers(self, definition_name: str) -> List[str]:
        """Find languages consuming an OpenAPI definition."""
        # This would check for generated client code
        return []
        
    def _validate_dependency_graph(self) -> None:
        """Validate the dependency graph for issues."""
        # Check for circular dependencies
        cycles = self._find_circular_dependencies()
        if cycles:
            self.logger.warning(f"Found circular dependencies: {cycles}")
            
        # Check for version mismatches
        mismatches = self._find_version_mismatches()
        if mismatches:
            self.logger.warning(f"Found version mismatches: {mismatches}")
            
    def _find_circular_dependencies(self) -> List[List[str]]:
        """Find circular dependencies in the graph."""
        cycles = []
        visited = set()
        rec_stack = set()
        
        def dfs(node: str, path: List[str]) -> None:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            # Get neighbors
            neighbors = [
                edge.target_component 
                for edge in self.dependency_graph.edges 
                if edge.source_component == node
            ]
            
            for neighbor in neighbors:
                if neighbor not in visited:
                    dfs(neighbor, path.copy())
                elif neighbor in rec_stack:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    cycles.append(cycle)
                    
            path.pop()
            rec_stack.remove(node)
            
        for node in self.dependency_graph.nodes:
            if node not in visited:
                dfs(node, [])
                
        return cycles
        
    def _find_version_mismatches(self) -> List[Dict[str, Any]]:
        """Find version mismatches in shared dependencies."""
        mismatches = []
        
        # Check API contract versions
        for contract_id, contract in self.dependency_graph.api_contracts.items():
            # Would check if consumers expect different versions
            pass
            
        return mismatches
        
    def export_dependency_graph(self, format: str = 'json') -> str:
        """Export the dependency graph in various formats."""
        if format == 'json':
            return json.dumps({
                'nodes': self.dependency_graph.nodes,
                'edges': [
                    {
                        'source': edge.source_component,
                        'target': edge.target_component,
                        'type': edge.dependency_type.value,
                        'source_language': edge.source_language,
                        'target_language': edge.target_language,
                        'protocol': edge.protocol.value if edge.protocol else None,
                        'metadata': edge.metadata
                    }
                    for edge in self.dependency_graph.edges
                ],
                'shared_structures': {
                    name: {
                        'name': struct.name,
                        'definition_language': struct.definition_language,
                        'consuming_languages': struct.consuming_languages,
                        'serialization_format': struct.serialization_format,
                        'version': struct.version
                    }
                    for name, struct in self.dependency_graph.shared_structures.items()
                },
                'analysis_timestamp': self.dependency_graph.analysis_timestamp.isoformat()
            }, indent=2)
            
        elif format == 'dot':
            # Export as Graphviz DOT format
            dot_lines = ['digraph dependencies {']
            
            # Add nodes
            for node_id, node_data in self.dependency_graph.nodes.items():
                label = f"{node_data['name']}\\n({node_data['language']})"
                dot_lines.append(f'  "{node_id}" [label="{label}"];')
                
            # Add edges
            for edge in self.dependency_graph.edges:
                label = f"{edge.dependency_type.value}"
                if edge.protocol:
                    label += f"\\n{edge.protocol.value}"
                dot_lines.append(
                    f'  "{edge.source_component}" -> "{edge.target_component}" '
                    f'[label="{label}"];'
                )
                
            dot_lines.append('}')
            return '\n'.join(dot_lines)
            
        else:
            raise ValueError(f"Unsupported export format: {format}")
            
    async def get_impact_analysis(
        self, 
        component_id: str, 
        change_type: str = 'api'
    ) -> Dict[str, Any]:
        """
        Analyze the impact of changes to a component.
        Returns affected components and risk assessment.
        """
        affected_components = set()
        impact_paths = []
        
        # Direct dependents
        direct_deps = [
            edge.source_component 
            for edge in self.dependency_graph.edges 
            if edge.target_component == component_id
        ]
        
        affected_components.update(direct_deps)
        
        # Transitive dependents
        visited = set()
        
        def find_transitive_deps(comp_id: str, path: List[str]):
            if comp_id in visited:
                return
            visited.add(comp_id)
            
            deps = [
                edge.source_component 
                for edge in self.dependency_graph.edges 
                if edge.target_component == comp_id
            ]
            
            for dep in deps:
                new_path = path + [dep]
                impact_paths.append(new_path)
                affected_components.add(dep)
                find_transitive_deps(dep, new_path)
                
        find_transitive_deps(component_id, [component_id])
        
        # Risk assessment
        risk_score = len(affected_components) * 0.1
        if change_type == 'api':
            risk_score *= 2  # API changes are higher risk
        elif change_type == 'database':
            risk_score *= 1.5
            
        return {
            'component_id': component_id,
            'change_type': change_type,
            'directly_affected': direct_deps,
            'total_affected': list(affected_components),
            'impact_paths': impact_paths,
            'risk_score': min(risk_score, 1.0),
            'recommendations': self._generate_impact_recommendations(
                component_id, 
                affected_components, 
                change_type
            )
        }
        
    def _generate_impact_recommendations(
        self, 
        component_id: str, 
        affected: Set[str], 
        change_type: str
    ) -> List[str]:
        """Generate recommendations based on impact analysis."""
        recommendations = []
        
        if len(affected) > 5:
            recommendations.append(
                "High impact change - consider phased rollout"
            )
            
        if change_type == 'api':
            recommendations.append(
                "Update API documentation and notify dependent teams"
            )
            recommendations.append(
                "Consider API versioning to maintain backward compatibility"
            )
            
        # Check for cross-language impacts
        languages = set()
        for comp_id in affected:
            if comp_id in self.dependency_graph.nodes:
                languages.add(self.dependency_graph.nodes[comp_id]['language'])
                
        if len(languages) > 2:
            recommendations.append(
                f"Change affects {len(languages)} different languages - "
                "ensure testing covers all language integrations"
            )
            
        return recommendations