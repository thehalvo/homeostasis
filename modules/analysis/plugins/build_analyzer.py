"""
Build file analyzer for Java projects.

This module provides functionality for analyzing and resolving issues in
Maven and Gradle build files.
"""

import logging
import re

import defusedxml.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class BuildFileAnalyzer:
    """
    Analyzes Maven and Gradle build files to detect and fix common issues.
    """

    def __init__(self):
        """Initialize the build file analyzer."""
        self.maven_repositories = {
            "central": "https://repo.maven.apache.org/maven2/",
            "jcenter": "https://jcenter.bintray.com/",
            "google": "https://maven.google.com/",
            "jitpack": "https://jitpack.io/",
        }

    def analyze_maven_pom(self, pom_path: str) -> Dict[str, Any]:
        """
        Analyze a Maven POM file for issues.

        Args:
            pom_path: Path to the pom.xml file

        Returns:
            Analysis results
        """
        try:
            tree = ET.parse(pom_path)
            root = tree.getroot()

            # Extract namespace if present
            ns = {"": root.tag.split("}")[0].strip("{")} if "}" in root.tag else {}

            # Create a namespace function to use in find/findall
            def ns_tag(tag):
                """Prepend namespace to tag if needed."""
                return f"{{{ns['']}}}{tag}" if ns else tag

            results = {
                "dependencies": [],
                "repositories": [],
                "plugins": [],
                "properties": {},
                "issues": [],
            }

            # Extract dependencies
            for dep in root.findall(f".//{ns_tag('dependency')}", ns):
                group_id = dep.find(f"{ns_tag('groupId')}", ns)
                artifact_id = dep.find(f"{ns_tag('artifactId')}", ns)
                version = dep.find(f"{ns_tag('version')}", ns)

                if group_id is not None and artifact_id is not None:
                    dependency = {
                        "groupId": group_id.text,
                        "artifactId": artifact_id.text,
                        "version": version.text if version is not None else "managed",
                    }

                    # Check for potential issues
                    if version is not None and "${" in version.text:
                        # Property reference
                        prop_name = version.text.strip("${}")
                        property_exists = False

                        # Check if property exists
                        for prop in root.findall(
                            f".//{ns_tag('properties')}/{ns_tag(prop_name)}", ns
                        ):
                            property_exists = True
                            break

                        if not property_exists:
                            results["issues"].append(
                                {
                                    "type": "missing_property",
                                    "description": f"Property {prop_name} referenced in {dependency['groupId']}:{dependency['artifactId']} is not defined",
                                    "severity": "medium",
                                    "fix_suggestion": f"Add <{prop_name}>VERSION</{prop_name}> to the <properties> section",
                                }
                            )

                    results["dependencies"].append(dependency)

            # Extract repositories
            for repo in root.findall(f".//{ns_tag('repository')}", ns):
                repo_id = repo.find(f"{ns_tag('id')}", ns)
                repo_url = repo.find(f"{ns_tag('url')}", ns)

                if repo_id is not None and repo_url is not None:
                    results["repositories"].append(
                        {"id": repo_id.text, "url": repo_url.text}
                    )

            # Extract plugins
            for plugin in root.findall(f".//{ns_tag('plugin')}", ns):
                group_id = plugin.find(f"{ns_tag('groupId')}", ns)
                artifact_id = plugin.find(f"{ns_tag('artifactId')}", ns)
                version = plugin.find(f"{ns_tag('version')}", ns)

                if group_id is not None and artifact_id is not None:
                    plugin_info = {
                        "groupId": group_id.text,
                        "artifactId": artifact_id.text,
                        "version": version.text if version is not None else "managed",
                    }

                    # Check for potential issues
                    if version is None:
                        results["issues"].append(
                            {
                                "type": "missing_plugin_version",
                                "description": f"Plugin {plugin_info['groupId']}:{plugin_info['artifactId']} has no version specified",
                                "severity": "low",
                                "fix_suggestion": "Add <version>X.Y.Z</version> to the plugin declaration",
                            }
                        )

                    results["plugins"].append(plugin_info)

            # Extract properties
            properties = root.find(f".//{ns_tag('properties')}", ns)
            if properties is not None:
                for prop in properties:
                    # Strip namespace if present
                    tag = prop.tag.split("}")[1] if "}" in prop.tag else prop.tag
                    results["properties"][tag] = prop.text

            return results

        except ET.ParseError as e:
            logger.error(f"Error parsing POM file {pom_path}: {e}")
            return {
                "dependencies": [],
                "repositories": [],
                "plugins": [],
                "properties": {},
                "issues": [
                    {
                        "type": "parse_error",
                        "description": f"XML parse error in POM file: {str(e)}",
                        "severity": "high",
                        "fix_suggestion": "Check XML syntax for errors, ensure tags are properly closed",
                    }
                ],
            }
        except Exception as e:
            logger.error(f"Error analyzing POM file {pom_path}: {e}")
            return {
                "dependencies": [],
                "repositories": [],
                "plugins": [],
                "properties": {},
                "issues": [
                    {
                        "type": "analysis_error",
                        "description": f"Error analyzing POM file: {str(e)}",
                        "severity": "medium",
                        "fix_suggestion": "Check file format and contents",
                    }
                ],
            }

    def analyze_gradle_build(self, build_path: str) -> Dict[str, Any]:
        """
        Analyze a Gradle build file for issues.

        Args:
            build_path: Path to the build.gradle file

        Returns:
            Analysis results
        """
        try:
            with open(build_path, "r") as f:
                content = f.read()

            results = {
                "dependencies": [],
                "repositories": [],
                "plugins": [],
                "properties": {},
                "issues": [],
            }

            # Extract dependencies using regex
            dep_pattern = r"(implementation|api|runtimeOnly|compileOnly|testImplementation|compile|runtime|testCompile|testRuntime)[\s\(]+'([^']+):([^']+):([^']+)'"
            for match in re.finditer(dep_pattern, content):
                scope, group_id, artifact_id, version = match.groups()

                dependency = {
                    "scope": scope,
                    "groupId": group_id,
                    "artifactId": artifact_id,
                    "version": version,
                }

                # Check for potential issues
                if "$" in version and "{" in version:
                    # Property reference
                    prop_name = version.strip("${}")
                    if not re.search(rf"{prop_name}\s*=", content):
                        results["issues"].append(
                            {
                                "type": "missing_property",
                                "description": f"Property {prop_name} referenced in {group_id}:{artifact_id} is not defined",
                                "severity": "medium",
                                "fix_suggestion": f"Add {prop_name}=VERSION to gradle.properties or ext section",
                            }
                        )

                results["dependencies"].append(dependency)

            # Extract repositories
            repo_pattern = r"repositories\s*\{([^}]+)\}"
            repo_matches = re.search(repo_pattern, content)
            if repo_matches:
                repo_block = repo_matches.group(1)

                # Check for common repositories
                if "mavenCentral()" in repo_block:
                    results["repositories"].append(
                        {
                            "id": "mavenCentral",
                            "url": "https://repo.maven.apache.org/maven2/",
                        }
                    )

                if "jcenter()" in repo_block:
                    results["repositories"].append(
                        {"id": "jcenter", "url": "https://jcenter.bintray.com/"}
                    )

                if "google()" in repo_block:
                    results["repositories"].append(
                        {"id": "google", "url": "https://maven.google.com/"}
                    )

                # Extract custom repositories
                custom_repo_pattern = r"maven\s*\{\s*url\s*[\"']([^\"']+)[\"']"
                for match in re.finditer(custom_repo_pattern, repo_block):
                    url = match.group(1)
                    results["repositories"].append({"id": "custom", "url": url})

            # Extract plugins
            plugin_pattern = r"(apply\s+plugin:\s*[\"']([^\"']+)[\"']|id\s*[\"']([^\"']+)[\"']\s*version\s*[\"']([^\"']+)[\"'])"
            for match in re.finditer(plugin_pattern, content):
                if match.group(2):  # apply plugin syntax
                    results["plugins"].append(
                        {"id": match.group(2), "version": "unspecified"}
                    )
                else:  # plugins DSL syntax
                    results["plugins"].append(
                        {"id": match.group(3), "version": match.group(4)}
                    )

            # Extract properties from ext block
            ext_pattern = r"ext\s*\{([^}]+)\}"
            ext_matches = re.search(ext_pattern, content)
            if ext_matches:
                ext_block = ext_matches.group(1)
                prop_pattern = r"(\w+)\s*=\s*[\"']([^\"']+)[\"']"
                for match in re.finditer(prop_pattern, ext_block):
                    name, value = match.groups()
                    results["properties"][name] = value

            return results

        except Exception as e:
            logger.error(f"Error analyzing Gradle file {build_path}: {e}")
            return {
                "dependencies": [],
                "repositories": [],
                "plugins": [],
                "properties": {},
                "issues": [
                    {
                        "type": "analysis_error",
                        "description": f"Error analyzing Gradle file: {str(e)}",
                        "severity": "medium",
                        "fix_suggestion": "Check file format and contents",
                    }
                ],
            }

    def find_dependency(
        self, group_id: str, artifact_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Search for a dependency in Maven repositories.

        Args:
            group_id: The Maven groupId
            artifact_id: The Maven artifactId

        Returns:
            Information about the dependency if found, None otherwise
        """
        # This would typically use Maven Central API or similar
        # For now, return a placeholder
        return {
            "groupId": group_id,
            "artifactId": artifact_id,
            "latestVersion": "x.y.z",
            "repository": "central",
        }

    def suggest_dependency_fix(
        self, group_id: str, artifact_id: str, version: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Suggest fixes for a dependency issue.

        Args:
            group_id: The Maven groupId
            artifact_id: The Maven artifactId
            version: Optional version to use

        Returns:
            Suggestion information
        """
        # Try to find the dependency
        dependency_info = self.find_dependency(group_id, artifact_id)

        if dependency_info is not None:
            version_to_use = version or dependency_info.get("latestVersion", "x.y.z")

            return {
                "found": True,
                "groupId": group_id,
                "artifactId": artifact_id,
                "suggestedVersion": version_to_use,
                "mavenSuggestion": f"<dependency>\n    <groupId>{group_id}</groupId>\n    <artifactId>{artifact_id}</artifactId>\n    <version>{version_to_use}</version>\n</dependency>",
                "gradleSuggestion": f"implementation '{group_id}:{artifact_id}:{version_to_use}'",
                "repository": dependency_info.get("repository", "central"),
            }
        else:
            # Dependency not found
            return {
                "found": False,
                "groupId": group_id,
                "artifactId": artifact_id,
                "mavenSuggestion": f"<dependency>\n    <groupId>{group_id}</groupId>\n    <artifactId>{artifact_id}</artifactId>\n    <version>VERSION</version>\n</dependency>",
                "gradleSuggestion": f"implementation '{group_id}:{artifact_id}:VERSION'",
                "additionalSuggestion": "Check the artifact coordinates and ensure the repository is correctly configured.",
            }

    def generate_dependency_report(self, project_dir: str) -> Dict[str, Any]:
        """
        Generate a dependency report for a Java project.

        Args:
            project_dir: Path to the project directory

        Returns:
            Dependency report data
        """
        project_path = Path(project_dir)

        report = {
            "projectType": "unknown",
            "buildFiles": [],
            "dependencies": {},
            "issues": [],
        }

        # Check for Maven
        pom_files = list(project_path.glob("**/pom.xml"))
        if pom_files:
            report["projectType"] = "maven"

            for pom_file in pom_files:
                analysis = self.analyze_maven_pom(str(pom_file))

                build_file = {
                    "path": str(pom_file.relative_to(project_path)),
                    "analysis": analysis,
                }

                report["buildFiles"].append(build_file)
                report["issues"].extend(analysis["issues"])

                # Track dependencies
                for dep in analysis["dependencies"]:
                    key = f"{dep['groupId']}:{dep['artifactId']}"
                    if key not in report["dependencies"]:
                        report["dependencies"][key] = []

                    report["dependencies"][key].append(
                        {
                            "version": dep["version"],
                            "buildFile": str(pom_file.relative_to(project_path)),
                        }
                    )

        # Check for Gradle
        gradle_files = list(project_path.glob("**/build.gradle"))
        if gradle_files:
            report["projectType"] = "gradle" if not pom_files else "hybrid"

            for gradle_file in gradle_files:
                analysis = self.analyze_gradle_build(str(gradle_file))

                build_file = {
                    "path": str(gradle_file.relative_to(project_path)),
                    "analysis": analysis,
                }

                report["buildFiles"].append(build_file)
                report["issues"].extend(analysis["issues"])

                # Track dependencies
                for dep in analysis["dependencies"]:
                    key = f"{dep['groupId']}:{dep['artifactId']}"
                    if key not in report["dependencies"]:
                        report["dependencies"][key] = []

                    report["dependencies"][key].append(
                        {
                            "version": dep["version"],
                            "scope": dep["scope"],
                            "buildFile": str(gradle_file.relative_to(project_path)),
                        }
                    )

        # Check for dependency version conflicts
        for dep_key, versions in report["dependencies"].items():
            if len(versions) > 1:
                # Check if versions differ
                unique_versions = set(
                    v["version"] for v in versions if v["version"] != "managed"
                )
                if len(unique_versions) > 1:
                    report["issues"].append(
                        {
                            "type": "version_conflict",
                            "description": f"Dependency {dep_key} has multiple versions: {', '.join(unique_versions)}",
                            "severity": "medium",
                            "fix_suggestion": "Use dependency management to ensure consistent versions",
                        }
                    )

        return report


# Create a build analyzer instance
build_analyzer = BuildFileAnalyzer()
