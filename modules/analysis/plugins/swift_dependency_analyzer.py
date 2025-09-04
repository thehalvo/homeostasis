"""
Swift Package Manager Dependency Analyzer

This module provides dependency analysis for Swift projects using Swift Package Manager.
It can detect missing dependencies, version conflicts, and suggest fixes for common
dependency-related issues.
"""
import logging
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class SwiftDependencyAnalyzer:
    """
    Analyzes Swift Package Manager dependencies and provides fix suggestions.
    """
    
    def __init__(self):
        """Initialize the Swift dependency analyzer."""
        self.package_cache = {}
        self.known_packages = self._load_known_packages()
    
    def _load_known_packages(self) -> Dict[str, Dict[str, Any]]:
        """Load known Swift packages and their information."""
        # This would typically load from a database or API
        # For now, we'll include some common packages
        return {
            "Alamofire": {
                "url": "https://github.com/Alamofire/Alamofire.git",
                "description": "Elegant HTTP Networking in Swift",
                "common_imports": ["Alamofire"]
            },
            "SDWebImage": {
                "url": "https://github.com/SDWebImage/SDWebImage.git", 
                "description": "Asynchronous image downloader with cache support",
                "common_imports": ["SDWebImage"]
            },
            "SwiftyJSON": {
                "url": "https://github.com/SwiftyJSON/SwiftyJSON.git",
                "description": "The better way to deal with JSON data in Swift",
                "common_imports": ["SwiftyJSON"]
            },
            "RealmSwift": {
                "url": "https://github.com/realm/realm-swift.git",
                "description": "A mobile database that replaces Core Data & SQLite",
                "common_imports": ["RealmSwift", "Realm"]
            },
            "Kingfisher": {
                "url": "https://github.com/onevcat/Kingfisher.git",
                "description": "A lightweight, pure-Swift library for downloading and caching images",
                "common_imports": ["Kingfisher"]
            },
            "SnapKit": {
                "url": "https://github.com/SnapKit/SnapKit.git",
                "description": "A Swift Autolayout DSL for iOS & OS X",
                "common_imports": ["SnapKit"]
            },
            "RxSwift": {
                "url": "https://github.com/ReactiveX/RxSwift.git",
                "description": "Reactive Programming in Swift",
                "common_imports": ["RxSwift", "RxCocoa"]
            }
        }
    
    def analyze_project_dependencies(self, project_path: str) -> Dict[str, Any]:
        """
        Analyze dependencies for a Swift project.
        
        Args:
            project_path: Path to the Swift project root
            
        Returns:
            Analysis results with dependency information
        """
        project_path = Path(project_path)
        
        # Look for Package.swift file
        package_file = project_path / "Package.swift"
        if not package_file.exists():
            return {
                "has_spm": False,
                "error": "No Package.swift file found",
                "suggestions": [
                    "Initialize Swift Package Manager with: swift package init",
                    "Or check if this is an Xcode project with CocoaPods/Carthage"
                ]
            }
        
        try:
            # Parse Package.swift
            package_info = self._parse_package_swift(package_file)
            
            # Analyze dependencies
            dependency_analysis = self._analyze_dependencies(package_info, project_path)
            
            # Check for common issues
            issues = self._check_dependency_issues(package_info, project_path)
            
            return {
                "has_spm": True,
                "package_info": package_info,
                "dependency_analysis": dependency_analysis,
                "issues": issues,
                "suggestions": self._generate_suggestions(issues)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing Swift dependencies: {e}")
            return {
                "has_spm": True,
                "error": str(e),
                "suggestions": ["Check Package.swift syntax and Swift Package Manager setup"]
            }
    
    def _parse_package_swift(self, package_file: Path) -> Dict[str, Any]:
        """Parse Package.swift file to extract dependency information."""
        with open(package_file, 'r') as f:
            content = f.read()
        
        package_info = {
            "name": self._extract_package_name(content),
            "dependencies": self._extract_dependencies(content),
            "targets": self._extract_targets(content),
            "platforms": self._extract_platforms(content),
            "swift_tools_version": self._extract_swift_tools_version(content)
        }
        
        return package_info
    
    def _extract_package_name(self, content: str) -> Optional[str]:
        """Extract package name from Package.swift content."""
        match = re.search(r'name:\s*"([^"]+)"', content)
        return match.group(1) if match else None
    
    def _extract_dependencies(self, content: str) -> List[Dict[str, Any]]:
        """Extract dependencies from Package.swift content."""
        dependencies = []
        
        # Find dependencies array
        deps_match = re.search(r'dependencies:\s*\[(.*?)\]', content, re.DOTALL)
        if not deps_match:
            return dependencies
        
        deps_content = deps_match.group(1)
        
        # Extract individual dependency declarations
        dep_patterns = [
            r'\.package\(url:\s*"([^"]+)",\s*from:\s*"([^"]+)"\)',  # from version
            r'\.package\(url:\s*"([^"]+)",\s*"([^"]+)"\.\.<"([^"]+)"\)',  # version range
            r'\.package\(url:\s*"([^"]+)",\s*exact:\s*"([^"]+)"\)',  # exact version
            r'\.package\(url:\s*"([^"]+)",\s*branch:\s*"([^"]+)"\)',  # branch
            r'\.package\(url:\s*"([^"]+)",\s*revision:\s*"([^"]+)"\)',  # revision
            r'\.package\(path:\s*"([^"]+)"\)'  # local path
        ]
        
        for pattern in dep_patterns:
            matches = re.finditer(pattern, deps_content)
            for match in matches:
                if "path:" in pattern:
                    dependencies.append({
                        "type": "local",
                        "path": match.group(1)
                    })
                else:
                    dep = {
                        "type": "remote",
                        "url": match.group(1)
                    }
                    
                    if "from:" in pattern:
                        dep["version_requirement"] = f"from {match.group(2)}"
                    elif "..<" in pattern:
                        dep["version_requirement"] = f"{match.group(2)} ..< {match.group(3)}"
                    elif "exact:" in pattern:
                        dep["version_requirement"] = f"exact {match.group(2)}"
                    elif "branch:" in pattern:
                        dep["version_requirement"] = f"branch {match.group(2)}"
                    elif "revision:" in pattern:
                        dep["version_requirement"] = f"revision {match.group(2)}"
                    
                    dependencies.append(dep)
        
        return dependencies
    
    def _extract_targets(self, content: str) -> List[Dict[str, Any]]:
        """Extract targets from Package.swift content."""
        targets = []
        
        # Find targets array
        targets_match = re.search(r'targets:\s*\[(.*?)\]', content, re.DOTALL)
        if not targets_match:
            return targets
        
        targets_content = targets_match.group(1)
        
        # Extract individual target declarations
        target_pattern = r'\.target\(\s*name:\s*"([^"]+)"(?:,\s*dependencies:\s*\[([^\]]*)\])?\)'
        matches = re.finditer(target_pattern, targets_content, re.DOTALL)
        
        for match in matches:
            target = {
                "name": match.group(1),
                "dependencies": []
            }
            
            if match.group(2):
                # Extract target dependencies
                deps_str = match.group(2)
                dep_matches = re.findall(r'"([^"]+)"', deps_str)
                target["dependencies"] = dep_matches
            
            targets.append(target)
        
        return targets
    
    def _extract_platforms(self, content: str) -> List[str]:
        """Extract supported platforms from Package.swift content."""
        platforms = []
        
        platforms_match = re.search(r'platforms:\s*\[(.*?)\]', content, re.DOTALL)
        if not platforms_match:
            return platforms
        
        platforms_content = platforms_match.group(1)
        platform_matches = re.findall(r'\.(\w+)\([^)]*\)', platforms_content)
        
        return platform_matches
    
    def _extract_swift_tools_version(self, content: str) -> Optional[str]:
        """Extract Swift tools version from Package.swift content."""
        match = re.search(r'// swift-tools-version:\s*([^\s\n]+)', content)
        return match.group(1) if match else None
    
    def _analyze_dependencies(self, package_info: Dict[str, Any], project_path: Path) -> Dict[str, Any]:
        """Analyze the dependencies for potential issues."""
        dependencies = package_info.get("dependencies", [])
        
        analysis = {
            "total_dependencies": len(dependencies),
            "remote_dependencies": len([d for d in dependencies if d.get("type") == "remote"]),
            "local_dependencies": len([d for d in dependencies if d.get("type") == "local"]),
            "potential_issues": []
        }
        
        # Check for common dependency issues
        for dep in dependencies:
            if dep.get("type") == "remote":
                url = dep.get("url", "")
                
                # Check for deprecated or moved repositories
                if self._is_deprecated_package(url):
                    analysis["potential_issues"].append({
                        "type": "deprecated_package",
                        "url": url,
                        "severity": "medium"
                    })
                
                # Check for version requirements
                version_req = dep.get("version_requirement", "")
                if not version_req:
                    analysis["potential_issues"].append({
                        "type": "missing_version",
                        "url": url,
                        "severity": "low"
                    })
        
        return analysis
    
    def _check_dependency_issues(self, package_info: Dict[str, Any], project_path: Path) -> List[Dict[str, Any]]:
        """Check for specific dependency-related issues."""
        issues = []
        
        # Check if Package.resolved exists and is up to date
        resolved_file = project_path / "Package.resolved"
        if not resolved_file.exists():
            issues.append({
                "type": "missing_package_resolved",
                "severity": "low",
                "description": "Package.resolved file is missing - run 'swift package resolve'"
            })
        
        # Check for build directory
        build_dir = project_path / ".build"
        if not build_dir.exists():
            issues.append({
                "type": "missing_build_artifacts",
                "severity": "low", 
                "description": "Build artifacts missing - run 'swift build' to build dependencies"
            })
        
        # Check for conflicting dependencies
        conflicts = self._detect_dependency_conflicts(package_info)
        issues.extend(conflicts)
        
        return issues
    
    def _detect_dependency_conflicts(self, package_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect potential dependency conflicts."""
        conflicts = []
        dependencies = package_info.get("dependencies", [])
        
        # Group dependencies by package name
        package_groups = {}
        for dep in dependencies:
            if dep.get("type") == "remote":
                url = dep.get("url", "")
                package_name = self._extract_package_name_from_url(url)
                if package_name:
                    if package_name not in package_groups:
                        package_groups[package_name] = []
                    package_groups[package_name].append(dep)
        
        # Check for multiple versions of same package
        for package_name, deps in package_groups.items():
            if len(deps) > 1:
                conflicts.append({
                    "type": "duplicate_dependency",
                    "package": package_name,
                    "dependencies": deps,
                    "severity": "high",
                    "description": f"Multiple versions of {package_name} declared"
                })
        
        return conflicts
    
    def _extract_package_name_from_url(self, url: str) -> Optional[str]:
        """Extract package name from Git URL."""
        # Remove .git suffix and extract last path component
        clean_url = url.rstrip('.git')
        parts = clean_url.split('/')
        return parts[-1] if parts else None
    
    def _is_deprecated_package(self, url: str) -> bool:
        """Check if a package URL points to a deprecated package."""
        # This would typically check against a database of deprecated packages
        deprecated_patterns = [
            r"github\.com/[^/]+/[^/]+-deprecated",
            r"github\.com/[^/]+/deprecated-"
        ]
        
        for pattern in deprecated_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                return True
        
        return False
    
    def _generate_suggestions(self, issues: List[Dict[str, Any]]) -> List[str]:
        """Generate fix suggestions based on detected issues."""
        suggestions = []
        
        for issue in issues:
            issue_type = issue.get("type")
            
            if issue_type == "missing_package_resolved":
                suggestions.append("Run 'swift package resolve' to create Package.resolved")
            elif issue_type == "missing_build_artifacts":
                suggestions.append("Run 'swift build' to download and build dependencies")
            elif issue_type == "duplicate_dependency":
                package = issue.get("package", "unknown")
                suggestions.append(f"Remove duplicate dependency declarations for {package}")
            elif issue_type == "deprecated_package":
                url = issue.get("url", "")
                suggestions.append(f"Consider updating deprecated package: {url}")
            elif issue_type == "missing_version":
                url = issue.get("url", "")
                suggestions.append(f"Add version requirement for dependency: {url}")
        
        # Add general suggestions
        if not suggestions:
            suggestions.append("Dependencies appear to be properly configured")
        
        return suggestions
    
    def analyze_dependency_error(self, error_data: Dict[str, Any], project_path: str) -> Dict[str, Any]:
        """
        Analyze a dependency-related error and provide fix suggestions.
        
        Args:
            error_data: Error data from Swift compiler/runtime
            project_path: Path to the project root
            
        Returns:
            Analysis results with fix suggestions
        """
        message = error_data.get("message", "")
        # Check for common dependency error patterns
        if "No such module" in message:
            return self._analyze_missing_module_error(message, project_path)
        elif "Package.swift" in message and "error" in message.lower():
            return self._analyze_package_swift_error(message, project_path)
        elif "dependency" in message.lower() and "version" in message.lower():
            return self._analyze_version_conflict_error(message, project_path)
        elif "could not build" in message.lower() and "swift package" in message.lower():
            return self._analyze_build_error(message, project_path)
        else:
            return self._analyze_generic_dependency_error(message, project_path)
    
    def _analyze_missing_module_error(self, message: str, project_path: str) -> Dict[str, Any]:
        """Analyze 'No such module' errors."""
        # Extract module name
        module_match = re.search(r"No such module '([^']+)'", message)
        if not module_match:
            return self._generic_dependency_analysis("Missing module", message)
        
        module_name = module_match.group(1)
        
        # Check if it's a known package
        suggested_package = None
        for package_name, package_info in self.known_packages.items():
            if module_name in package_info.get("common_imports", []):
                suggested_package = package_name
                break
        
        suggestions = []
        if suggested_package:
            package_info = self.known_packages[suggested_package]
            suggestions.extend([
                f"Add {suggested_package} to Package.swift dependencies:",
                f'.package(url: "{package_info["url"]}", from: "latest_version")',
                f"Then add '{module_name}' to your target dependencies"
            ])
        else:
            suggestions.extend([
                f"Module '{module_name}' not found",
                "Check if the module name is spelled correctly",
                "Ensure the package is added to Package.swift dependencies",
                "Run 'swift build' to build dependencies"
            ])
        
        return {
            "category": "dependency",
            "subcategory": "missing_module",
            "confidence": "high",
            "suggested_fix": f"Add missing module '{module_name}' to dependencies",
            "root_cause": "swift_missing_module",
            "severity": "high",
            "module_name": module_name,
            "suggested_package": suggested_package,
            "suggestions": suggestions
        }
    
    def _analyze_package_swift_error(self, message: str, project_path: str) -> Dict[str, Any]:
        """Analyze Package.swift syntax errors."""
        return {
            "category": "dependency",
            "subcategory": "package_swift_syntax",
            "confidence": "high",
            "suggested_fix": "Fix Package.swift syntax error",
            "root_cause": "swift_package_swift_syntax",
            "severity": "high",
            "suggestions": [
                "Check Package.swift syntax",
                "Ensure all strings are properly quoted",
                "Verify dependency declarations format",
                "Run 'swift package describe' to validate Package.swift"
            ]
        }
    
    def _analyze_version_conflict_error(self, message: str, project_path: str) -> Dict[str, Any]:
        """Analyze dependency version conflict errors."""
        return {
            "category": "dependency",
            "subcategory": "version_conflict",
            "confidence": "high",
            "suggested_fix": "Resolve dependency version conflicts",
            "root_cause": "swift_dependency_version_conflict",
            "severity": "medium",
            "suggestions": [
                "Check for conflicting version requirements",
                "Update Package.swift with compatible version ranges",
                "Remove Package.resolved and run 'swift package resolve'",
                "Consider updating to latest compatible versions"
            ]
        }
    
    def _analyze_build_error(self, message: str, project_path: str) -> Dict[str, Any]:
        """Analyze Swift package build errors."""
        return {
            "category": "dependency",
            "subcategory": "build_error",
            "confidence": "medium",
            "suggested_fix": "Fix dependency build issues",
            "root_cause": "swift_dependency_build_error",
            "severity": "medium",
            "suggestions": [
                "Run 'swift package clean' and rebuild",
                "Check for platform compatibility issues",
                "Ensure all dependencies support your target platform",
                "Update to latest package versions"
            ]
        }
    
    def _analyze_generic_dependency_error(self, message: str, project_path: str) -> Dict[str, Any]:
        """Analyze generic dependency errors."""
        return self._generic_dependency_analysis("Generic dependency error", message)
    
    def _generic_dependency_analysis(self, error_type: str, message: str) -> Dict[str, Any]:
        """Provide generic dependency analysis."""
        return {
            "category": "dependency",
            "subcategory": "unknown",
            "confidence": "low",
            "suggested_fix": f"{error_type} - check dependency configuration",
            "root_cause": "swift_dependency_error",
            "severity": "medium",
            "suggestions": [
                "Verify Package.swift configuration",
                "Run 'swift package resolve' to update dependencies",
                "Check for dependency compatibility issues",
                "Review error message for specific details"
            ]
        }