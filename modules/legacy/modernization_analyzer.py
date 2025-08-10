"""
Modernization path analysis for legacy systems.

This module analyzes legacy codebases and provides recommendations for
gradual modernization while maintaining system stability.
"""

import json
import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import ast
import os

logger = logging.getLogger(__name__)


class ModernizationStrategy(Enum):
    """Modernization approach strategies."""
    STRANGLER_FIG = "Strangler Fig Pattern"
    BIG_BANG = "Big Bang Rewrite"
    INCREMENTAL = "Incremental Refactoring"
    HYBRID = "Hybrid Approach"
    LIFT_AND_SHIFT = "Lift and Shift"
    REPLATFORM = "Replatform"
    REFACTOR = "Refactor/Rearchitect"


class RiskLevel(Enum):
    """Risk levels for modernization efforts."""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


class TechnologyStack(Enum):
    """Technology stack categories."""
    MAINFRAME = "Mainframe"
    CLIENT_SERVER = "Client-Server"
    WEB_MONOLITH = "Web Monolith"
    MICROSERVICES = "Microservices"
    SERVERLESS = "Serverless"
    CLOUD_NATIVE = "Cloud Native"


@dataclass
class LegacyComponent:
    """Represents a component in the legacy system."""
    name: str
    type: str  # program, module, service, database, etc.
    language: str
    lines_of_code: int
    complexity_score: float
    dependencies: List[str] = field(default_factory=list)
    interfaces: List[str] = field(default_factory=list)
    business_criticality: str = "medium"
    technical_debt_score: float = 0.0
    last_modified: Optional[datetime] = None


@dataclass
class ModernizationPath:
    """Represents a modernization path for a component."""
    component: LegacyComponent
    strategy: ModernizationStrategy
    target_technology: str
    phases: List[Dict[str, Any]]
    estimated_effort: int  # person-days
    risk_level: RiskLevel
    dependencies: List[str]
    benefits: List[str]
    challenges: List[str]
    success_metrics: List[str]


@dataclass
class SystemAnalysis:
    """Complete system modernization analysis."""
    total_components: int
    total_loc: int
    languages: Dict[str, int]  # language -> line count
    complexity_distribution: Dict[str, int]
    risk_assessment: Dict[RiskLevel, int]
    recommended_strategy: ModernizationStrategy
    modernization_paths: List[ModernizationPath]
    roadmap: List[Dict[str, Any]]
    estimated_total_effort: int
    roi_analysis: Dict[str, Any]


class ModernizationAnalyzer:
    """
    Analyzes legacy systems and provides modernization recommendations.
    
    Evaluates code complexity, dependencies, business value, and technical
    debt to create actionable modernization roadmaps.
    """
    
    def __init__(self):
        self._complexity_patterns = self._load_complexity_patterns()
        self._modernization_templates = self._load_modernization_templates()
        self._technology_mappings = self._load_technology_mappings()
        
    def _load_complexity_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load patterns for complexity analysis."""
        return {
            "cobol": {
                "high_complexity": [
                    r"ALTER\s+", r"GO\s+TO\s+", r"PERFORM\s+.*THRU",
                    r"SECTION\s*\.", r"COPY\s+.*REPLACING"
                ],
                "medium_complexity": [
                    r"EVALUATE\s+", r"PERFORM\s+VARYING", r"EXEC\s+SQL",
                    r"CALL\s+", r"REDEFINES\s+"
                ],
                "technical_debt": [
                    r"GOTO\s+", r"ALTER\s+", r"COMMON\s+",
                    r"EQUIVALENCE\s+", r"\d{4}-\d{2}-\d{2}-"  # hardcoded dates
                ]
            },
            "fortran": {
                "high_complexity": [
                    r"GOTO\s+\d+", r"COMMON\s*/", r"EQUIVALENCE\s*\(",
                    r"ENTRY\s+", r"ASSIGN\s+\d+\s+TO"
                ],
                "medium_complexity": [
                    r"CALL\s+", r"SUBROUTINE\s+", r"FUNCTION\s+",
                    r"MODULE\s+", r"INTERFACE\s+"
                ],
                "technical_debt": [
                    r"GOTO\s+", r"COMMON\s+", r"IMPLICIT\s+",
                    r"PAUSE\s+", r"HOLLERITH"
                ]
            },
            "java": {
                "high_complexity": [
                    r"synchronized\s*\(", r"catch\s*\(\s*Exception\s*\)",
                    r"Class\.forName", r"\.newInstance\(\)",
                    r"Runtime\.getRuntime\(\)\.exec"
                ],
                "medium_complexity": [
                    r"implements\s+Serializable", r"@Deprecated",
                    r"Vector\s*<", r"Hashtable\s*<", r"StringBuffer\s+"
                ],
                "technical_debt": [
                    r"System\.out\.println", r"printStackTrace\(\)",
                    r"catch\s*\(\s*Exception\s*\)", r"@SuppressWarnings",
                    r"Date\s+\w+\s*=\s*new\s+Date\("  # old date API
                ]
            }
        }
        
    def _load_modernization_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load modernization approach templates."""
        return {
            "mainframe_to_cloud": {
                "strategies": [
                    ModernizationStrategy.STRANGLER_FIG,
                    ModernizationStrategy.INCREMENTAL,
                    ModernizationStrategy.REPLATFORM
                ],
                "phases": [
                    {
                        "name": "Assessment",
                        "activities": [
                            "Inventory all programs and dependencies",
                            "Analyze data flows and interfaces",
                            "Identify business-critical components",
                            "Assess technical debt"
                        ],
                        "duration": 30
                    },
                    {
                        "name": "Foundation",
                        "activities": [
                            "Set up cloud infrastructure",
                            "Establish CI/CD pipelines",
                            "Create data migration strategy",
                            "Build integration layer"
                        ],
                        "duration": 60
                    },
                    {
                        "name": "Pilot Migration",
                        "activities": [
                            "Select low-risk components",
                            "Migrate batch jobs first",
                            "Implement API facades",
                            "Validate data integrity"
                        ],
                        "duration": 90
                    },
                    {
                        "name": "Incremental Migration",
                        "activities": [
                            "Migrate by business domain",
                            "Maintain dual-run period",
                            "Gradually shift traffic",
                            "Decommission legacy components"
                        ],
                        "duration": 180
                    }
                ],
                "target_stack": [
                    "Containerized microservices",
                    "Cloud-native databases",
                    "Event-driven architecture",
                    "API-first design"
                ]
            },
            "monolith_to_microservices": {
                "strategies": [
                    ModernizationStrategy.STRANGLER_FIG,
                    ModernizationStrategy.INCREMENTAL
                ],
                "phases": [
                    {
                        "name": "Domain Analysis",
                        "activities": [
                            "Identify bounded contexts",
                            "Map data ownership",
                            "Define service boundaries",
                            "Plan API contracts"
                        ],
                        "duration": 30
                    },
                    {
                        "name": "Extract Services",
                        "activities": [
                            "Start with edge services",
                            "Extract authentication/authorization",
                            "Separate read-heavy services",
                            "Implement service mesh"
                        ],
                        "duration": 120
                    },
                    {
                        "name": "Data Decomposition",
                        "activities": [
                            "Implement event sourcing",
                            "Separate databases gradually",
                            "Ensure data consistency",
                            "Build data synchronization"
                        ],
                        "duration": 90
                    }
                ],
                "target_stack": [
                    "Container orchestration",
                    "Service mesh",
                    "API gateway",
                    "Distributed tracing"
                ]
            }
        }
        
    def _load_technology_mappings(self) -> Dict[str, Dict[str, Any]]:
        """Load technology migration mappings."""
        return {
            "cobol": {
                "target_languages": ["Java", "C#", "Python", "Go"],
                "migration_tools": [
                    "COBOL-to-Java converters",
                    "Micro Focus Visual COBOL",
                    "IBM Developer for z/OS",
                    "Rational Developer"
                ],
                "patterns": {
                    "batch_processing": "Spring Batch, Apache Spark",
                    "transaction_processing": "Spring Boot, .NET Core",
                    "file_handling": "Apache Camel, NiFi",
                    "report_generation": "JasperReports, Crystal Reports"
                }
            },
            "fortran": {
                "target_languages": ["C++", "Python", "Julia", "Rust"],
                "migration_tools": [
                    "F2C converter",
                    "Fortran-to-C++ translators",
                    "NumPy/SciPy for scientific computing",
                    "Modern Fortran compilers"
                ],
                "patterns": {
                    "numerical_computation": "NumPy, BLAS/LAPACK",
                    "scientific_modeling": "Julia, MATLAB, Python",
                    "high_performance": "C++, Rust, CUDA",
                    "data_analysis": "Pandas, R, Julia"
                }
            },
            "vb6": {
                "target_languages": ["C#", "VB.NET", "TypeScript", "Python"],
                "migration_tools": [
                    "Microsoft upgrade tools",
                    "VB6 to .NET converters",
                    "Third-party migration tools"
                ],
                "patterns": {
                    "desktop_apps": "WPF, WinForms, Electron",
                    "com_components": ".NET assemblies, REST APIs",
                    "database_access": "Entity Framework, Dapper",
                    "ui_forms": "Blazor, React, Angular"
                }
            }
        }
        
    def analyze_codebase(self, root_path: str) -> List[LegacyComponent]:
        """Analyze legacy codebase and identify components."""
        components = []
        
        for dirpath, dirnames, filenames in os.walk(root_path):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                
                # Determine language
                language = self._detect_language(filename)
                if not language:
                    continue
                    
                # Analyze file
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        
                    component = self._analyze_file(
                        file_path, filename, content, language
                    )
                    if component:
                        components.append(component)
                        
                except Exception as e:
                    logger.warning(f"Failed to analyze {file_path}: {e}")
                    
        # Analyze dependencies
        self._analyze_dependencies(components)
        
        return components
        
    def _detect_language(self, filename: str) -> Optional[str]:
        """Detect programming language from filename."""
        ext_to_lang = {
            '.cbl': 'cobol', '.cob': 'cobol', '.cobol': 'cobol',
            '.f': 'fortran', '.f77': 'fortran', '.f90': 'fortran',
            '.f95': 'fortran', '.f03': 'fortran', '.f08': 'fortran',
            '.for': 'fortran', '.ftn': 'fortran',
            '.java': 'java', '.jsp': 'java',
            '.vb': 'vb6', '.bas': 'vb6', '.frm': 'vb6',
            '.c': 'c', '.h': 'c',
            '.cpp': 'cpp', '.cc': 'cpp', '.cxx': 'cpp',
            '.cs': 'csharp',
            '.py': 'python',
            '.js': 'javascript', '.ts': 'typescript'
        }
        
        _, ext = os.path.splitext(filename.lower())
        return ext_to_lang.get(ext)
        
    def _analyze_file(self, file_path: str, filename: str, 
                     content: str, language: str) -> Optional[LegacyComponent]:
        """Analyze individual source file."""
        lines = content.split('\n')
        loc = len([l for l in lines if l.strip() and not l.strip().startswith(('*', 'C', '!', '//'))])
        
        # Calculate complexity
        complexity = self._calculate_complexity(content, language)
        
        # Detect component type
        component_type = self._detect_component_type(content, language)
        
        # Extract interfaces
        interfaces = self._extract_interfaces(content, language)
        
        # Calculate technical debt
        tech_debt = self._calculate_technical_debt(content, language)
        
        # Get file modification time
        try:
            mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
        except:
            mtime = None
            
        return LegacyComponent(
            name=filename,
            type=component_type,
            language=language,
            lines_of_code=loc,
            complexity_score=complexity,
            interfaces=interfaces,
            technical_debt_score=tech_debt,
            last_modified=mtime
        )
        
    def _calculate_complexity(self, content: str, language: str) -> float:
        """Calculate complexity score for code."""
        if language not in self._complexity_patterns:
            return 5.0  # Default medium complexity
            
        patterns = self._complexity_patterns[language]
        score = 1.0  # Base complexity
        
        # Count high complexity patterns
        for pattern in patterns.get("high_complexity", []):
            matches = len(re.findall(pattern, content, re.IGNORECASE))
            score += matches * 2.0
            
        # Count medium complexity patterns
        for pattern in patterns.get("medium_complexity", []):
            matches = len(re.findall(pattern, content, re.IGNORECASE))
            score += matches * 1.0
            
        # Normalize by lines of code
        loc = len(content.split('\n'))
        if loc > 0:
            score = min(score / (loc / 100), 10.0)  # Cap at 10
            
        return round(score, 2)
        
    def _detect_component_type(self, content: str, language: str) -> str:
        """Detect type of component."""
        content_upper = content.upper()
        
        if language in ["cobol", "fortran"]:
            if "PROGRAM-ID" in content_upper or "PROGRAM " in content_upper:
                if any(db in content_upper for db in ["EXEC SQL", "SELECT ", "INSERT "]):
                    return "database_program"
                elif any(ui in content_upper for ui in ["SCREEN", "DISPLAY", "ACCEPT"]):
                    return "interactive_program"
                else:
                    return "batch_program"
            elif "SUBROUTINE" in content_upper or "FUNCTION" in content_upper:
                return "library"
                
        elif language == "java":
            if "@RestController" in content or "@Controller" in content:
                return "web_controller"
            elif "@Service" in content:
                return "service"
            elif "@Repository" in content or "DAO" in content:
                return "data_access"
            elif "@Entity" in content:
                return "entity"
            elif "public static void main" in content:
                return "application"
            else:
                return "class"
                
        return "module"
        
    def _extract_interfaces(self, content: str, language: str) -> List[str]:
        """Extract external interfaces from code."""
        interfaces = []
        
        if language == "cobol":
            # CALL statements
            calls = re.findall(r"CALL\s+['\"](\w+)['\"]", content, re.IGNORECASE)
            interfaces.extend([f"CALL:{c}" for c in calls])
            
            # File operations
            files = re.findall(r"SELECT\s+(\w+)\s+ASSIGN", content, re.IGNORECASE)
            interfaces.extend([f"FILE:{f}" for f in files])
            
            # Database operations
            if "EXEC SQL" in content.upper():
                interfaces.append("DATABASE:SQL")
                
        elif language == "fortran":
            # Subroutine calls
            calls = re.findall(r"CALL\s+(\w+)", content, re.IGNORECASE)
            interfaces.extend([f"CALL:{c}" for c in calls])
            
            # Module usage
            modules = re.findall(r"USE\s+(\w+)", content, re.IGNORECASE)
            interfaces.extend([f"MODULE:{m}" for m in modules])
            
        elif language == "java":
            # Imports
            imports = re.findall(r"import\s+([\w\.]+);", content)
            interfaces.extend([f"IMPORT:{i}" for i in imports if not i.startswith("java.")])
            
            # REST endpoints
            endpoints = re.findall(r"@(?:Get|Post|Put|Delete)Mapping\(['\"]([^'\"]+)", content)
            interfaces.extend([f"REST:{e}" for e in endpoints])
            
        return list(set(interfaces))
        
    def _calculate_technical_debt(self, content: str, language: str) -> float:
        """Calculate technical debt score."""
        if language not in self._complexity_patterns:
            return 0.0
            
        patterns = self._complexity_patterns[language].get("technical_debt", [])
        debt_score = 0.0
        
        for pattern in patterns:
            matches = len(re.findall(pattern, content, re.IGNORECASE))
            debt_score += matches * 0.5
            
        # Check for code smells
        lines = content.split('\n')
        
        # Long methods/programs
        if language in ["cobol", "fortran"]:
            procedure_lines = 0
            for line in lines:
                if re.match(r"\s*PROCEDURE\s+DIVISION", line, re.IGNORECASE):
                    procedure_lines = len(lines) - lines.index(line)
                    break
            if procedure_lines > 500:
                debt_score += (procedure_lines / 100) * 0.5
                
        # Commented code
        commented_code = len([l for l in lines if re.match(r"^\s*[\*C!]\s*\w+", l)])
        if commented_code > 50:
            debt_score += (commented_code / 10) * 0.2
            
        return min(round(debt_score, 2), 10.0)
        
    def _analyze_dependencies(self, components: List[LegacyComponent]):
        """Analyze dependencies between components."""
        # Build dependency map
        component_map = {c.name: c for c in components}
        
        for component in components:
            for interface in component.interfaces:
                if interface.startswith("CALL:"):
                    called = interface.split(":")[1]
                    # Find called component
                    for other in components:
                        if called in other.name.upper():
                            component.dependencies.append(other.name)
                            break
                            
    def create_modernization_plan(self, components: List[LegacyComponent]) -> SystemAnalysis:
        """Create comprehensive modernization plan."""
        # Analyze system statistics
        total_loc = sum(c.lines_of_code for c in components)
        languages = {}
        complexity_dist = {"low": 0, "medium": 0, "high": 0}
        
        for comp in components:
            languages[comp.language] = languages.get(comp.language, 0) + comp.lines_of_code
            
            if comp.complexity_score < 3:
                complexity_dist["low"] += 1
            elif comp.complexity_score < 7:
                complexity_dist["medium"] += 1
            else:
                complexity_dist["high"] += 1
                
        # Determine overall strategy
        strategy = self._determine_strategy(components, languages)
        
        # Create modernization paths
        paths = []
        for component in components:
            path = self._create_component_path(component, strategy)
            paths.append(path)
            
        # Create roadmap
        roadmap = self._create_roadmap(paths)
        
        # Calculate effort
        total_effort = sum(p.estimated_effort for p in paths)
        
        # Risk assessment
        risk_dist = {level: 0 for level in RiskLevel}
        for path in paths:
            risk_dist[path.risk_level] += 1
            
        # ROI analysis
        roi = self._calculate_roi(components, paths, total_effort)
        
        return SystemAnalysis(
            total_components=len(components),
            total_loc=total_loc,
            languages=languages,
            complexity_distribution=complexity_dist,
            risk_assessment=risk_dist,
            recommended_strategy=strategy,
            modernization_paths=paths,
            roadmap=roadmap,
            estimated_total_effort=total_effort,
            roi_analysis=roi
        )
        
    def _determine_strategy(self, components: List[LegacyComponent], 
                          languages: Dict[str, int]) -> ModernizationStrategy:
        """Determine best modernization strategy."""
        # Check system characteristics
        total_components = len(components)
        avg_complexity = sum(c.complexity_score for c in components) / total_components
        avg_tech_debt = sum(c.technical_debt_score for c in components) / total_components
        
        # High complexity or technical debt suggests incremental approach
        if avg_complexity > 7 or avg_tech_debt > 5:
            return ModernizationStrategy.INCREMENTAL
            
        # Many interconnected components suggest strangler fig
        total_dependencies = sum(len(c.dependencies) for c in components)
        if total_dependencies > total_components * 2:
            return ModernizationStrategy.STRANGLER_FIG
            
        # Small, isolated systems might benefit from big bang
        if total_components < 10 and total_dependencies < total_components:
            return ModernizationStrategy.BIG_BANG
            
        # Legacy mainframe code usually needs careful migration
        if "cobol" in languages or "fortran" in languages:
            return ModernizationStrategy.STRANGLER_FIG
            
        return ModernizationStrategy.INCREMENTAL
        
    def _create_component_path(self, component: LegacyComponent, 
                              strategy: ModernizationStrategy) -> ModernizationPath:
        """Create modernization path for component."""
        # Determine target technology
        if component.language in self._technology_mappings:
            mapping = self._technology_mappings[component.language]
            target_tech = mapping["target_languages"][0]
            patterns = mapping.get("patterns", {})
        else:
            target_tech = "Java"  # Default
            patterns = {}
            
        # Create phases based on component type
        phases = self._create_phases(component, strategy)
        
        # Estimate effort
        base_effort = component.lines_of_code / 50  # Base: 50 LOC per day
        complexity_factor = 1 + (component.complexity_score / 10)
        debt_factor = 1 + (component.technical_debt_score / 10)
        effort = int(base_effort * complexity_factor * debt_factor)
        
        # Assess risk
        risk = self._assess_risk(component)
        
        # Benefits
        benefits = [
            "Improved maintainability",
            "Better performance",
            "Enhanced security",
            "Reduced operational costs"
        ]
        
        if component.language in ["cobol", "fortran"]:
            benefits.extend([
                "Access to modern developer talent",
                "Integration with modern systems",
                "Cloud deployment capability"
            ])
            
        # Challenges
        challenges = []
        if component.complexity_score > 7:
            challenges.append("High code complexity")
        if component.technical_debt_score > 5:
            challenges.append("Significant technical debt")
        if len(component.dependencies) > 5:
            challenges.append("Many dependencies to manage")
        if component.business_criticality == "high":
            challenges.append("Business-critical functionality")
            
        # Success metrics
        metrics = [
            "All tests passing",
            "Performance benchmarks met",
            "Zero data loss",
            "Successful parallel run"
        ]
        
        return ModernizationPath(
            component=component,
            strategy=strategy,
            target_technology=target_tech,
            phases=phases,
            estimated_effort=effort,
            risk_level=risk,
            dependencies=component.dependencies,
            benefits=benefits,
            challenges=challenges,
            success_metrics=metrics
        )
        
    def _create_phases(self, component: LegacyComponent, 
                      strategy: ModernizationStrategy) -> List[Dict[str, Any]]:
        """Create modernization phases for component."""
        phases = []
        
        # Analysis phase
        phases.append({
            "name": "Analysis",
            "activities": [
                f"Document {component.name} functionality",
                "Map data flows and dependencies",
                "Create test cases",
                "Design target architecture"
            ],
            "duration": max(5, component.lines_of_code // 1000),
            "deliverables": [
                "Functional specification",
                "Test plan",
                "Architecture design"
            ]
        })
        
        # Preparation phase
        if strategy == ModernizationStrategy.STRANGLER_FIG:
            phases.append({
                "name": "Facade Creation",
                "activities": [
                    "Build API facade",
                    "Implement adapter pattern",
                    "Set up routing logic",
                    "Create monitoring"
                ],
                "duration": 10,
                "deliverables": [
                    "API facade",
                    "Routing configuration",
                    "Monitoring dashboard"
                ]
            })
            
        # Migration phase
        phases.append({
            "name": "Migration",
            "activities": [
                f"Convert {component.language} to target language",
                "Implement modern patterns",
                "Refactor for cloud deployment",
                "Optimize performance"
            ],
            "duration": component.lines_of_code // 100,
            "deliverables": [
                "Migrated code",
                "Unit tests",
                "Performance benchmarks"
            ]
        })
        
        # Testing phase
        phases.append({
            "name": "Testing",
            "activities": [
                "Execute unit tests",
                "Run integration tests",
                "Perform parallel run",
                "Validate data integrity"
            ],
            "duration": max(10, component.lines_of_code // 500),
            "deliverables": [
                "Test results",
                "Performance comparison",
                "Sign-off documentation"
            ]
        })
        
        # Deployment phase
        phases.append({
            "name": "Deployment",
            "activities": [
                "Deploy to production",
                "Monitor performance",
                "Gradual traffic shift",
                "Decommission legacy"
            ],
            "duration": 5,
            "deliverables": [
                "Deployment documentation",
                "Monitoring reports",
                "Decommission plan"
            ]
        })
        
        return phases
        
    def _assess_risk(self, component: LegacyComponent) -> RiskLevel:
        """Assess modernization risk for component."""
        risk_score = 0
        
        # Complexity risk
        if component.complexity_score > 8:
            risk_score += 3
        elif component.complexity_score > 5:
            risk_score += 2
        else:
            risk_score += 1
            
        # Size risk
        if component.lines_of_code > 10000:
            risk_score += 3
        elif component.lines_of_code > 5000:
            risk_score += 2
        else:
            risk_score += 1
            
        # Dependency risk
        if len(component.dependencies) > 10:
            risk_score += 3
        elif len(component.dependencies) > 5:
            risk_score += 2
        else:
            risk_score += 1
            
        # Business criticality
        if component.business_criticality == "high":
            risk_score += 2
        elif component.business_criticality == "medium":
            risk_score += 1
            
        # Technical debt
        if component.technical_debt_score > 7:
            risk_score += 2
        elif component.technical_debt_score > 4:
            risk_score += 1
            
        # Map to risk level
        if risk_score >= 10:
            return RiskLevel.CRITICAL
        elif risk_score >= 7:
            return RiskLevel.HIGH
        elif risk_score >= 4:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
            
    def _create_roadmap(self, paths: List[ModernizationPath]) -> List[Dict[str, Any]]:
        """Create implementation roadmap."""
        roadmap = []
        
        # Sort paths by risk and dependencies
        sorted_paths = sorted(paths, key=lambda p: (
            p.risk_level.value,
            len(p.dependencies),
            p.component.business_criticality
        ))
        
        # Group into waves
        wave_size = max(1, len(sorted_paths) // 4)
        waves = [
            sorted_paths[i:i+wave_size] 
            for i in range(0, len(sorted_paths), wave_size)
        ]
        
        start_month = 0
        for i, wave in enumerate(waves):
            wave_effort = sum(p.estimated_effort for p in wave)
            wave_duration = max(3, wave_effort // 20)  # Assume 20 person-days per month
            
            roadmap.append({
                "wave": i + 1,
                "components": [p.component.name for p in wave],
                "start_month": start_month,
                "duration_months": wave_duration,
                "effort_days": wave_effort,
                "key_milestones": [
                    f"Complete analysis for {len(wave)} components",
                    f"Migrate {sum(p.component.lines_of_code for p in wave)} lines of code",
                    f"Deploy and validate wave {i + 1}"
                ],
                "risks": [
                    f"{sum(1 for p in wave if p.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL])} high-risk components",
                    f"Dependencies: {sum(len(p.dependencies) for p in wave)} total"
                ]
            })
            
            start_month += wave_duration
            
        return roadmap
        
    def _calculate_roi(self, components: List[LegacyComponent], 
                      paths: List[ModernizationPath], 
                      total_effort: int) -> Dict[str, Any]:
        """Calculate return on investment."""
        # Cost calculation
        avg_daily_rate = 800  # Average developer daily rate
        migration_cost = total_effort * avg_daily_rate
        
        # Current maintenance cost (estimated)
        total_loc = sum(c.lines_of_code for c in components)
        annual_maintenance_cost = total_loc * 10  # $10 per LOC per year (legacy)
        
        # Future maintenance cost (estimated)
        future_maintenance_cost = total_loc * 3  # $3 per LOC per year (modern)
        
        # Annual savings
        annual_savings = annual_maintenance_cost - future_maintenance_cost
        
        # Additional benefits
        productivity_gain = 0.3  # 30% productivity improvement
        additional_savings = annual_maintenance_cost * productivity_gain
        
        # Total annual benefit
        total_annual_benefit = annual_savings + additional_savings
        
        # Payback period
        payback_years = migration_cost / total_annual_benefit
        
        # 5-year NPV (assuming 10% discount rate)
        npv = 0
        for year in range(1, 6):
            npv += total_annual_benefit / ((1.1) ** year)
        npv -= migration_cost
        
        return {
            "migration_cost": migration_cost,
            "current_annual_maintenance": annual_maintenance_cost,
            "future_annual_maintenance": future_maintenance_cost,
            "annual_savings": annual_savings,
            "productivity_gains": additional_savings,
            "total_annual_benefit": total_annual_benefit,
            "payback_period_years": round(payback_years, 1),
            "five_year_npv": round(npv, 0),
            "roi_percentage": round((total_annual_benefit * 5 - migration_cost) / migration_cost * 100, 1)
        }
        
    def generate_report(self, analysis: SystemAnalysis) -> str:
        """Generate modernization analysis report."""
        report = []
        
        report.append("# Legacy System Modernization Analysis\n")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Executive Summary
        report.append("## Executive Summary\n")
        report.append(f"- Total Components: {analysis.total_components}")
        report.append(f"- Total Lines of Code: {analysis.total_loc:,}")
        report.append(f"- Recommended Strategy: {analysis.recommended_strategy.value}")
        report.append(f"- Estimated Effort: {analysis.estimated_total_effort:,} person-days")
        report.append(f"- Estimated Cost: ${analysis.roi_analysis['migration_cost']:,.0f}")
        report.append(f"- Payback Period: {analysis.roi_analysis['payback_period_years']} years")
        report.append(f"- 5-Year ROI: {analysis.roi_analysis['roi_percentage']}%\n")
        
        # Language Distribution
        report.append("## Language Distribution\n")
        for lang, loc in sorted(analysis.languages.items(), key=lambda x: x[1], reverse=True):
            percentage = (loc / analysis.total_loc) * 100
            report.append(f"- {lang.title()}: {loc:,} lines ({percentage:.1f}%)")
        report.append("")
        
        # Complexity Analysis
        report.append("## Complexity Distribution\n")
        for level, count in analysis.complexity_distribution.items():
            percentage = (count / analysis.total_components) * 100
            report.append(f"- {level.title()}: {count} components ({percentage:.1f}%)")
        report.append("")
        
        # Risk Assessment
        report.append("## Risk Assessment\n")
        for risk, count in sorted(analysis.risk_assessment.items(), key=lambda x: x[0].value):
            percentage = (count / analysis.total_components) * 100
            report.append(f"- {risk.value}: {count} components ({percentage:.1f}%)")
        report.append("")
        
        # Implementation Roadmap
        report.append("## Implementation Roadmap\n")
        for wave in analysis.roadmap:
            report.append(f"### Wave {wave['wave']} (Months {wave['start_month']}-{wave['start_month'] + wave['duration_months']})")
            report.append(f"- Components: {', '.join(wave['components'])}")
            report.append(f"- Effort: {wave['effort_days']} person-days")
            report.append("- Milestones:")
            for milestone in wave['key_milestones']:
                report.append(f"  - {milestone}")
            report.append("")
            
        # ROI Analysis
        report.append("## Return on Investment\n")
        roi = analysis.roi_analysis
        report.append(f"- Migration Investment: ${roi['migration_cost']:,.0f}")
        report.append(f"- Current Annual Maintenance: ${roi['current_annual_maintenance']:,.0f}")
        report.append(f"- Future Annual Maintenance: ${roi['future_annual_maintenance']:,.0f}")
        report.append(f"- Annual Savings: ${roi['annual_savings']:,.0f}")
        report.append(f"- Productivity Gains: ${roi['productivity_gains']:,.0f}")
        report.append(f"- Total Annual Benefit: ${roi['total_annual_benefit']:,.0f}")
        report.append(f"- 5-Year NPV: ${roi['five_year_npv']:,.0f}\n")
        
        # Recommendations
        report.append("## Key Recommendations\n")
        report.append("1. **Start with Low-Risk Components**: Begin modernization with components that have low complexity and few dependencies")
        report.append("2. **Establish Modern Infrastructure**: Set up CI/CD pipelines, automated testing, and cloud infrastructure early")
        report.append("3. **Maintain Parallel Systems**: Run legacy and modern systems in parallel during transition")
        report.append("4. **Invest in Training**: Ensure team has skills for both legacy maintenance and modern development")
        report.append("5. **Monitor Progress**: Track KPIs and adjust approach based on learnings")
        
        return '\n'.join(report)