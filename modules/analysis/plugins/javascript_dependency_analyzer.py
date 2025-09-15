"""
JavaScript Dependency Analyzer

This module provides dependency analysis for JavaScript projects, including
npm and yarn package management, version conflict detection, and dependency
tree analysis.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

from packaging import version as pkg_version

logger = logging.getLogger(__name__)


class JavaScriptDependencyAnalyzer:
    """
    Analyzer for JavaScript dependencies and package management.

    This class provides functionality to analyze package.json files, detect
    version conflicts, analyze dependency trees, and suggest dependency fixes.
    """

    def __init__(self):
        """Initialize the JavaScript dependency analyzer."""
        self.package_managers = ["npm", "yarn", "pnpm"]
        self.package_files = [
            "package.json",
            "package-lock.json",
            "yarn.lock",
            "pnpm-lock.yaml",
        ]

        # Cache for package information
        self.package_cache = {}

        # Common dependency patterns and their issues
        self.known_issues = {
            "peer_dependencies": {
                "react": ["react-dom"],
                "webpack": ["webpack-cli"],
                "@types/node": ["typescript"],
                "eslint": ["@eslint/config"],
            },
            "common_conflicts": [
                {"packages": ["react", "react-dom"], "reason": "version_mismatch"},
                {"packages": ["typescript", "@types/node"], "reason": "compatibility"},
                {
                    "packages": ["webpack", "webpack-dev-server"],
                    "reason": "version_compatibility",
                },
            ],
            "security_vulnerable": [
                # These would be loaded from a security database
                "node-sass@4.14.1",
                "lodash@4.17.20",
            ],
        }

    def analyze_project_dependencies(self, project_path: str) -> Dict[str, Any]:
        """
        Analyze dependencies for a JavaScript project.

        Args:
            project_path: Path to the JavaScript project root

        Returns:
            Analysis results including conflicts, missing dependencies, and suggestions
        """
        project_path_obj = Path(project_path)

        # Find package.json
        package_json_path = project_path_obj / "package.json"
        if not package_json_path.exists():
            return {
                "error": "package.json not found",
                "suggestions": ["Initialize project with 'npm init' or 'yarn init'"],
            }

        # Parse package.json
        try:
            with open(package_json_path, "r") as f:
                package_data = json.load(f)
        except json.JSONDecodeError as e:
            return {
                "error": f"Invalid package.json: {e}",
                "suggestions": ["Fix JSON syntax in package.json"],
            }

        # Perform analysis
        analysis = {
            "project_name": package_data.get("name", "unknown"),
            "project_version": package_data.get("version", "0.0.0"),
            "package_manager": self._detect_package_manager(project_path_obj),
            "dependencies": self._analyze_dependencies(package_data),
            "dev_dependencies": self._analyze_dev_dependencies(package_data),
            "peer_dependencies": self._analyze_peer_dependencies(package_data),
            "conflicts": self._detect_version_conflicts(package_data, project_path_obj),
            "missing_dependencies": self._detect_missing_dependencies(project_path_obj),
            "security_issues": self._detect_security_issues(package_data),
            "outdated_packages": self._detect_outdated_packages(package_data),
            "suggestions": [],
        }

        # Generate suggestions based on analysis
        analysis["suggestions"] = self._generate_suggestions(analysis)

        return analysis

    def analyze_dependency_error(
        self, error_data: Dict[str, Any], project_path: str
    ) -> Dict[str, Any]:
        """
        Analyze a dependency-related error.

        Args:
            error_data: Error data from JavaScript runtime
            project_path: Path to the project root

        Returns:
            Analysis results with fix suggestions
        """
        error_message = error_data.get("message", "")

        # Module not found errors
        if "Cannot find module" in error_message:
            return self._analyze_module_not_found(error_message, project_path)

        # Version conflict errors
        if "version" in error_message.lower() and (
            "conflict" in error_message.lower() or "mismatch" in error_message.lower()
        ):
            return self._analyze_version_conflict(error_message, project_path)

        # Peer dependency warnings
        if "peer dep" in error_message.lower() or "ERESOLVE" in error_message:
            return self._analyze_peer_dependency_issue(error_message, project_path)

        # Engine compatibility errors
        if "engine" in error_message.lower() and "node" in error_message.lower():
            return self._analyze_engine_compatibility(error_message, project_path)

        return {
            "category": "dependency",
            "subcategory": "unknown",
            "confidence": "low",
            "suggested_fix": "Check package.json and dependency configuration",
            "details": "Unable to analyze specific dependency error",
        }

    def _detect_package_manager(self, project_path: Path) -> str:
        """Detect which package manager is being used."""
        if (project_path / "yarn.lock").exists():
            return "yarn"
        elif (project_path / "pnpm-lock.yaml").exists():
            return "pnpm"
        elif (project_path / "package-lock.json").exists():
            return "npm"
        else:
            return "npm"  # default

    def _analyze_dependencies(self, package_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze production dependencies."""
        dependencies = package_data.get("dependencies", {})

        return {
            "count": len(dependencies),
            "packages": dependencies,
            "issues": self._find_dependency_issues(dependencies),
        }

    def _analyze_dev_dependencies(self, package_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze development dependencies."""
        dev_dependencies = package_data.get("devDependencies", {})

        return {
            "count": len(dev_dependencies),
            "packages": dev_dependencies,
            "issues": self._find_dependency_issues(dev_dependencies),
        }

    def _analyze_peer_dependencies(
        self, package_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze peer dependencies."""
        peer_dependencies = package_data.get("peerDependencies", {})

        return {
            "count": len(peer_dependencies),
            "packages": peer_dependencies,
            "missing": self._find_missing_peer_dependencies(package_data),
        }

    def _find_dependency_issues(
        self, dependencies: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """Find issues in dependency specifications."""
        issues = []

        for name, version_spec in dependencies.items():
            # Check for wildcard versions
            if version_spec in ["*", "latest"]:
                issues.append(
                    {
                        "type": "wildcard_version",
                        "package": name,
                        "version": version_spec,
                        "severity": "medium",
                        "message": f"Wildcard version '{version_spec}' can cause instability",
                    }
                )

            # Check for pre-release versions in production
            if re.search(r"(alpha|beta|rc|pre)", version_spec):
                issues.append(
                    {
                        "type": "prerelease_version",
                        "package": name,
                        "version": version_spec,
                        "severity": "high",
                        "message": f"Pre-release version '{version_spec}' may be unstable",
                    }
                )

            # Check for known security issues
            package_version = f"{name}@{version_spec}"
            if package_version in self.known_issues["security_vulnerable"]:
                issues.append(
                    {
                        "type": "security_vulnerability",
                        "package": name,
                        "version": version_spec,
                        "severity": "critical",
                        "message": f"Package {package_version} has known security vulnerabilities",
                    }
                )

        return issues

    def _detect_version_conflicts(
        self, package_data: Dict[str, Any], project_path: Path
    ) -> List[Dict[str, Any]]:
        """Detect version conflicts between dependencies."""
        conflicts = []

        # Get all dependencies
        all_deps = {}
        all_deps.update(package_data.get("dependencies", {}))
        all_deps.update(package_data.get("devDependencies", {}))
        all_deps.update(package_data.get("peerDependencies", {}))

        # Check for known conflicts
        common_conflicts = self.known_issues["common_conflicts"]
        if isinstance(common_conflicts, list):
            for conflict_def in common_conflicts:
                if (
                    isinstance(conflict_def, dict)
                    and "packages" in conflict_def
                    and "reason" in conflict_def
                ):
                    packages = conflict_def["packages"]
                    reason = conflict_def["reason"]

                    # Check if all packages in the conflict are present
                    present_packages = [pkg for pkg in packages if pkg in all_deps]

                    if len(present_packages) >= 2:
                        # Check version compatibility
                        versions = {pkg: all_deps[pkg] for pkg in present_packages}

                        if reason == "version_mismatch":
                            if not self._are_versions_compatible(versions):
                                conflicts.append(
                                    {
                                        "type": "version_mismatch",
                                        "packages": present_packages,
                                        "versions": versions,
                                        "severity": "high",
                                        "message": f"Version mismatch detected between {', '.join(present_packages)}",
                                    }
                                )

        return conflicts

    def _detect_missing_dependencies(self, project_path: Path) -> List[Dict[str, Any]]:
        """Detect missing dependencies by analyzing imports."""
        missing = []

        # Look for JavaScript files
        js_files: List[Path] = []
        for pattern in ["**/*.js", "**/*.mjs", "**/*.cjs", "**/*.jsx"]:
            js_files.extend(project_path.glob(pattern))

        # Analyze imports in JavaScript files
        imported_modules = set()

        for js_file in js_files[:10]:  # Limit to avoid performance issues
            try:
                with open(js_file, "r", encoding="utf-8") as f:
                    content = f.read()

                # Find require() calls
                require_pattern = r'require\([\'"`]([^\'"`]+)[\'"`]\)'
                requires = re.findall(require_pattern, content)
                imported_modules.update(requires)

                # Find ES6 imports
                import_pattern = r'import.*?from\s+[\'"`]([^\'"`]+)[\'"`]'
                imports = re.findall(import_pattern, content)
                imported_modules.update(imports)

            except Exception as e:
                logger.debug(f"Error reading {js_file}: {e}")

        # Check which imports are not in node_modules or package.json
        package_json_path = project_path / "package.json"
        if package_json_path.exists():
            with open(package_json_path, "r") as f:
                package_data = json.load(f)

            all_deps = set()
            all_deps.update(package_data.get("dependencies", {}).keys())
            all_deps.update(package_data.get("devDependencies", {}).keys())

            for module in imported_modules:
                # Skip relative imports and built-in modules
                if module.startswith(".") or module in [
                    "fs",
                    "path",
                    "http",
                    "https",
                    "url",
                    "crypto",
                ]:
                    continue

                # Extract package name (handle scoped packages)
                package_name = (
                    module.split("/")[0]
                    if not module.startswith("@")
                    else "/".join(module.split("/")[:2])
                )

                if package_name not in all_deps:
                    missing.append(
                        {
                            "type": "missing_dependency",
                            "package": package_name,
                            "imported_as": module,
                            "severity": "high",
                            "message": f"Package '{package_name}' is imported but not listed in dependencies",
                        }
                    )

        return missing

    def _detect_security_issues(
        self, package_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Detect security issues in dependencies."""
        security_issues = []

        # This would typically integrate with npm audit or Snyk
        # For now, we'll check against our known vulnerabilities

        all_deps = {}
        all_deps.update(package_data.get("dependencies", {}))
        all_deps.update(package_data.get("devDependencies", {}))

        for package, version_spec in all_deps.items():
            package_version = f"{package}@{version_spec}"
            if package_version in self.known_issues["security_vulnerable"]:
                security_issues.append(
                    {
                        "type": "security_vulnerability",
                        "package": package,
                        "version": version_spec,
                        "severity": "critical",
                        "message": f"Security vulnerability in {package_version}",
                    }
                )

        return security_issues

    def _detect_outdated_packages(
        self, package_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Detect outdated packages."""
        # This would typically run npm outdated or yarn outdated
        # For now, return empty list as this requires external calls
        return []

    def _analyze_module_not_found(
        self, error_message: str, project_path: str
    ) -> Dict[str, Any]:
        """Analyze module not found errors."""
        # Extract module name from error message
        module_match = re.search(
            r"Cannot find module ['\"]([^'\"]+)['\"]", error_message
        )

        if not module_match:
            return {
                "category": "dependency",
                "subcategory": "module_not_found",
                "confidence": "low",
                "suggested_fix": "Check module name and installation",
            }

        module_name = module_match.group(1)

        # Check if it's a relative import issue
        if module_name.startswith("."):
            return {
                "category": "dependency",
                "subcategory": "relative_import",
                "confidence": "high",
                "suggested_fix": f"Check if the file '{module_name}' exists relative to the importing file",
                "module": module_name,
                "fix_commands": [
                    f"Verify the path '{module_name}' exists",
                    "Check file extensions and case sensitivity",
                ],
            }

        # Check if it's a missing package
        package_name = (
            module_name.split("/")[0]
            if not module_name.startswith("@")
            else "/".join(module_name.split("/")[:2])
        )

        return {
            "category": "dependency",
            "subcategory": "missing_package",
            "confidence": "high",
            "suggested_fix": f"Install the missing package '{package_name}'",
            "module": module_name,
            "package": package_name,
            "fix_commands": [f"npm install {package_name}", f"yarn add {package_name}"],
        }

    def _analyze_version_conflict(
        self, error_message: str, project_path: str
    ) -> Dict[str, Any]:
        """Analyze version conflict errors."""
        return {
            "category": "dependency",
            "subcategory": "version_conflict",
            "confidence": "medium",
            "suggested_fix": "Resolve version conflicts in package.json",
            "fix_commands": [
                "npm ls to see dependency tree",
                "Update conflicting packages to compatible versions",
                "Use npm shrinkwrap or yarn.lock to lock versions",
            ],
        }

    def _analyze_peer_dependency_issue(
        self, error_message: str, project_path: str
    ) -> Dict[str, Any]:
        """Analyze peer dependency issues."""
        return {
            "category": "dependency",
            "subcategory": "peer_dependency",
            "confidence": "high",
            "suggested_fix": "Install missing peer dependencies",
            "fix_commands": [
                "npm info <package> peerDependencies to see requirements",
                "Install the required peer dependencies",
                "Update package.json with peer dependencies",
            ],
        }

    def _analyze_engine_compatibility(
        self, error_message: str, project_path: str
    ) -> Dict[str, Any]:
        """Analyze Node.js engine compatibility issues."""
        return {
            "category": "dependency",
            "subcategory": "engine_compatibility",
            "confidence": "high",
            "suggested_fix": "Update Node.js version or adjust engine requirements",
            "fix_commands": [
                "Check package.json engines field",
                "Update Node.js to required version",
                "Use nvm to manage Node.js versions",
            ],
        }

    def _find_missing_peer_dependencies(
        self, package_data: Dict[str, Any]
    ) -> List[str]:
        """Find missing peer dependencies."""
        missing = []

        dependencies = package_data.get("dependencies", {})
        peer_deps = package_data.get("peerDependencies", {})

        # Check if packages requiring peer dependencies are installed
        peer_dependencies = self.known_issues.get("peer_dependencies", {})
        if isinstance(peer_dependencies, dict):
            for dep_name in dependencies:
                if dep_name in peer_dependencies:
                    required_peers = peer_dependencies[dep_name]
                    if isinstance(required_peers, list):
                        for peer in required_peers:
                            if peer not in dependencies and peer not in peer_deps:
                                missing.append(peer)

        return missing

    def _are_versions_compatible(self, versions: Dict[str, str]) -> bool:
        """Check if package versions are compatible."""
        # Simplified compatibility check
        # In a real implementation, this would use semantic versioning rules

        # For now, just check if versions are not wildly different
        version_numbers = []
        for version_spec in versions.values():
            # Extract version number from spec
            version_match = re.search(r"(\d+\.\d+\.\d+)", version_spec)
            if version_match:
                try:
                    version_numbers.append(pkg_version.parse(version_match.group(1)))
                except Exception:
                    return True  # Can't parse, assume compatible

        if len(version_numbers) < 2:
            return True

        # Check if major versions are the same
        major_versions = {v.major for v in version_numbers}
        return len(major_versions) == 1

    def _generate_suggestions(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate suggestions based on analysis results."""
        suggestions = []

        # Suggestions for conflicts
        if analysis["conflicts"]:
            suggestions.append(
                "Resolve version conflicts by updating packages to compatible versions"
            )

        # Suggestions for missing dependencies
        if analysis["missing_dependencies"]:
            package_manager = analysis["package_manager"]
            missing_packages = [
                dep["package"] for dep in analysis["missing_dependencies"]
            ]
            suggestions.append(
                f"Install missing dependencies: {package_manager} install {' '.join(missing_packages)}"
            )

        # Suggestions for security issues
        if analysis["security_issues"]:
            suggestions.append("Update packages with security vulnerabilities")
            suggestions.append(
                "Run 'npm audit fix' or 'yarn audit --fix' to automatically fix issues"
            )

        # Suggestions for peer dependencies
        missing_peers = analysis["peer_dependencies"]["missing"]
        if missing_peers:
            package_manager = analysis["package_manager"]
            suggestions.append(
                f"Install missing peer dependencies: {package_manager} install {' '.join(missing_peers)}"
            )

        return suggestions
