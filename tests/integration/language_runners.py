"""
Language-specific test runners for integration testing.

This module contains test runners for all 40+ supported languages.
"""

import asyncio
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import logging

from .language_integration_framework import (
    LanguageIntegrationTestRunner,
    IntegrationTestCase,
    IntegrationTestResult
)

logger = logging.getLogger(__name__)


class GoIntegrationTestRunner(LanguageIntegrationTestRunner):
    """Go language integration test runner."""
    
    async def setup_environment(self, test_case: IntegrationTestCase) -> Path:
        test_dir = Path(tempfile.mkdtemp(prefix="homeostasis_go_test_"))
        
        # Write source files
        for filename, content in test_case.source_code.items():
            file_path = test_dir / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)
            
        # Create go.mod if needed
        if not (test_dir / "go.mod").exists():
            module_name = f"homeostasis-test-{test_case.name}".replace(" ", "-")
            subprocess.run(["go", "mod", "init", module_name], cwd=test_dir, check=True)
            
        # Install dependencies
        if test_case.dependencies:
            for dep in test_case.dependencies:
                subprocess.run(["go", "get", dep], cwd=test_dir, check=True)
                
        return test_dir
        
    async def execute_code(self, test_dir: Path, test_case: IntegrationTestCase) -> Tuple[int, str, str]:
        # Build and run
        process = subprocess.Popen(
            ["go", "run", "."],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=test_dir,
            env={**os.environ, **test_case.environment}
        )
        
        try:
            stdout, stderr = process.communicate(timeout=test_case.timeout)
            return process.returncode, stdout.decode(), stderr.decode()
        except subprocess.TimeoutExpired:
            process.kill()
            return -1, "", "Test timeout exceeded"
            
    async def validate_environment(self, test_dir: Path) -> bool:
        result = subprocess.run(["go", "version"], capture_output=True, text=True)
        return result.returncode == 0


class RustIntegrationTestRunner(LanguageIntegrationTestRunner):
    """Rust language integration test runner."""
    
    async def setup_environment(self, test_case: IntegrationTestCase) -> Path:
        test_dir = Path(tempfile.mkdtemp(prefix="homeostasis_rust_test_"))
        
        # Create Cargo.toml
        cargo_toml = {
            "package": {
                "name": f"homeostasis_test_{test_case.name}".replace(" ", "_").replace("-", "_"),
                "version": "0.1.0",
                "edition": "2021"
            },
            "dependencies": {}
        }
        
        # Add dependencies
        for dep in test_case.dependencies:
            if "=" in dep:
                name, version = dep.split("=", 1)
                cargo_toml["dependencies"][name.strip()] = version.strip()
            else:
                cargo_toml["dependencies"][dep] = "*"
                
        # Write Cargo.toml
        import toml
        (test_dir / "Cargo.toml").write_text(toml.dumps(cargo_toml))
        
        # Create src directory
        src_dir = test_dir / "src"
        src_dir.mkdir()
        
        # Write source files
        for filename, content in test_case.source_code.items():
            if filename.startswith("src/"):
                file_path = test_dir / filename
            else:
                file_path = src_dir / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)
            
        return test_dir
        
    async def execute_code(self, test_dir: Path, test_case: IntegrationTestCase) -> Tuple[int, str, str]:
        # Build and run
        process = subprocess.Popen(
            ["cargo", "run"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=test_dir,
            env={**os.environ, **test_case.environment}
        )
        
        try:
            stdout, stderr = process.communicate(timeout=test_case.timeout)
            return process.returncode, stdout.decode(), stderr.decode()
        except subprocess.TimeoutExpired:
            process.kill()
            return -1, "", "Test timeout exceeded"
            
    async def validate_environment(self, test_dir: Path) -> bool:
        result = subprocess.run(["cargo", "--version"], capture_output=True, text=True)
        return result.returncode == 0


class JavaIntegrationTestRunner(LanguageIntegrationTestRunner):
    """Java language integration test runner."""
    
    async def setup_environment(self, test_case: IntegrationTestCase) -> Path:
        test_dir = Path(tempfile.mkdtemp(prefix="homeostasis_java_test_"))
        
        # Create directory structure
        src_dir = test_dir / "src" / "main" / "java"
        src_dir.mkdir(parents=True)
        
        # Write source files
        for filename, content in test_case.source_code.items():
            if filename.endswith(".java"):
                file_path = src_dir / filename
            else:
                file_path = test_dir / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)
            
        # Create pom.xml for Maven if dependencies exist
        if test_case.dependencies:
            pom_xml = self._generate_pom_xml(test_case)
            (test_dir / "pom.xml").write_text(pom_xml)
            
            # Run Maven install
            subprocess.run(["mvn", "install"], cwd=test_dir, check=True)
            
        return test_dir
        
    async def execute_code(self, test_dir: Path, test_case: IntegrationTestCase) -> Tuple[int, str, str]:
        # Find main class
        main_class = None
        src_dir = test_dir / "src" / "main" / "java"
        
        for java_file in src_dir.glob("**/*.java"):
            content = java_file.read_text()
            if "public static void main" in content:
                # Extract class name
                import re
                match = re.search(r'public\s+class\s+(\w+)', content)
                if match:
                    main_class = match.group(1)
                    break
                    
        if not main_class:
            return -1, "", "No main class found"
            
        # Compile and run
        if (test_dir / "pom.xml").exists():
            # Use Maven
            process = subprocess.Popen(
                ["mvn", "exec:java", f"-Dexec.mainClass={main_class}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=test_dir,
                env={**os.environ, **test_case.environment}
            )
        else:
            # Use javac/java directly
            # Compile
            java_files = list(src_dir.glob("**/*.java"))
            subprocess.run(["javac"] + [str(f) for f in java_files], cwd=test_dir, check=True)
            
            # Run
            process = subprocess.Popen(
                ["java", "-cp", "src/main/java", main_class],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=test_dir,
                env={**os.environ, **test_case.environment}
            )
            
        try:
            stdout, stderr = process.communicate(timeout=test_case.timeout)
            return process.returncode, stdout.decode(), stderr.decode()
        except subprocess.TimeoutExpired:
            process.kill()
            return -1, "", "Test timeout exceeded"
            
    async def validate_environment(self, test_dir: Path) -> bool:
        java_result = subprocess.run(["java", "-version"], capture_output=True, text=True)
        javac_result = subprocess.run(["javac", "-version"], capture_output=True, text=True)
        return java_result.returncode == 0 and javac_result.returncode == 0
        
    def _generate_pom_xml(self, test_case: IntegrationTestCase) -> str:
        """Generate Maven pom.xml file."""
        dependencies = []
        for dep in test_case.dependencies:
            # Parse dependency format: groupId:artifactId:version
            parts = dep.split(":")
            if len(parts) >= 2:
                dep_xml = f"""
        <dependency>
            <groupId>{parts[0]}</groupId>
            <artifactId>{parts[1]}</artifactId>
            <version>{parts[2] if len(parts) > 2 else 'LATEST'}</version>
        </dependency>"""
                dependencies.append(dep_xml)
                
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0
         http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>
    
    <groupId>com.homeostasis.test</groupId>
    <artifactId>{test_case.name.replace(' ', '-')}</artifactId>
    <version>1.0-SNAPSHOT</version>
    
    <properties>
        <maven.compiler.source>11</maven.compiler.source>
        <maven.compiler.target>11</maven.compiler.target>
    </properties>
    
    <dependencies>
        {"".join(dependencies)}
    </dependencies>
</project>"""


class PythonIntegrationTestRunner(LanguageIntegrationTestRunner):
    """Python language integration test runner."""
    
    async def setup_environment(self, test_case: IntegrationTestCase) -> Path:
        test_dir = Path(tempfile.mkdtemp(prefix="homeostasis_python_test_"))
        
        # Write source files
        for filename, content in test_case.source_code.items():
            file_path = test_dir / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)
            
        # Create virtual environment if needed
        if test_case.dependencies:
            venv_path = test_dir / "venv"
            subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)
            pip_path = venv_path / "bin" / "pip" if os.name != "nt" else venv_path / "Scripts" / "pip.exe"
            
            # Install dependencies
            for dep in test_case.dependencies:
                subprocess.run([str(pip_path), "install", dep], check=True)
                
        return test_dir
        
    async def execute_code(self, test_dir: Path, test_case: IntegrationTestCase) -> Tuple[int, str, str]:
        # Run the Python code
        main_file = test_case.metadata.get("main_file", "main.py")
        python_path = sys.executable
        
        if (test_dir / "venv").exists():
            python_path = test_dir / "venv" / "bin" / "python" if os.name != "nt" else test_dir / "venv" / "Scripts" / "python.exe"
            
        result = subprocess.run(
            [str(python_path), main_file],
            cwd=test_dir,
            capture_output=True,
            text=True
        )
        
        return result.returncode, result.stdout, result.stderr


class JavaScriptIntegrationTestRunner(LanguageIntegrationTestRunner):
    """JavaScript/Node.js integration test runner."""
    
    async def setup_environment(self, test_case: IntegrationTestCase) -> Path:
        test_dir = Path(tempfile.mkdtemp(prefix="homeostasis_js_test_"))
        
        # Write source files
        for filename, content in test_case.source_code.items():
            file_path = test_dir / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)
            
        # Create package.json if needed
        if not (test_dir / "package.json").exists():
            package_json = {
                "name": f"homeostasis-test-{test_case.name}".replace(" ", "-"),
                "version": "1.0.0",
                "dependencies": {}
            }
            
            if test_case.dependencies:
                for dep in test_case.dependencies:
                    if ":" in dep:
                        name, version = dep.split(":", 1)
                        package_json["dependencies"][name] = version
                    else:
                        package_json["dependencies"][dep] = "latest"
                        
            (test_dir / "package.json").write_text(json.dumps(package_json, indent=2))
            
        # Install dependencies
        if test_case.dependencies:
            subprocess.run(["npm", "install"], cwd=test_dir, check=True)
            
        return test_dir
        
    async def execute_code(self, test_dir: Path, test_case: IntegrationTestCase) -> Tuple[int, str, str]:
        # Run the JavaScript code
        main_file = test_case.metadata.get("main_file", "index.js")
        
        result = subprocess.run(
            ["node", main_file],
            cwd=test_dir,
            capture_output=True,
            text=True
        )
        
        return result.returncode, result.stdout, result.stderr


class TypeScriptIntegrationTestRunner(JavaScriptIntegrationTestRunner):
    """TypeScript integration test runner."""
    
    async def setup_environment(self, test_case: IntegrationTestCase) -> Path:
        test_dir = await super().setup_environment(test_case)
        
        # Create tsconfig.json
        tsconfig = {
            "compilerOptions": {
                "target": "ES2020",
                "module": "commonjs",
                "strict": True,
                "esModuleInterop": True,
                "skipLibCheck": True,
                "forceConsistentCasingInFileNames": True,
                "outDir": "./dist",
                "rootDir": "./"
            }
        }
        
        (test_dir / "tsconfig.json").write_text(json.dumps(tsconfig, indent=2))
        
        # Install TypeScript if not in dependencies
        if "typescript" not in test_case.dependencies:
            subprocess.run(["npm", "install", "--save-dev", "typescript"], cwd=test_dir, check=True)
            
        return test_dir
        
    async def execute_code(self, test_dir: Path, test_case: IntegrationTestCase) -> Tuple[int, str, str]:
        # Compile TypeScript
        compile_result = subprocess.run(
            ["npx", "tsc"],
            cwd=test_dir,
            capture_output=True,
            text=True
        )
        
        if compile_result.returncode != 0:
            return compile_result.returncode, compile_result.stdout, compile_result.stderr
            
        # Find compiled JS file
        main_file = test_dir / "dist" / "index.js"
        if not main_file.exists():
            js_files = list((test_dir / "dist").glob("*.js"))
            if js_files:
                main_file = js_files[0]
                
        # Run compiled JavaScript
        return await super().execute_code(test_dir / "dist", test_case)


class RubyIntegrationTestRunner(LanguageIntegrationTestRunner):
    """Ruby language integration test runner."""
    
    async def setup_environment(self, test_case: IntegrationTestCase) -> Path:
        test_dir = Path(tempfile.mkdtemp(prefix="homeostasis_ruby_test_"))
        
        # Write source files
        for filename, content in test_case.source_code.items():
            file_path = test_dir / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)
            
        # Create Gemfile if dependencies exist
        if test_case.dependencies:
            gemfile_content = "source 'https://rubygems.org'\n\n"
            for dep in test_case.dependencies:
                if "," in dep:
                    gem_name, version = dep.split(",", 1)
                    gemfile_content += f"gem '{gem_name}', '{version}'\n"
                else:
                    gemfile_content += f"gem '{dep}'\n"
                    
            (test_dir / "Gemfile").write_text(gemfile_content)
            
            # Run bundle install
            subprocess.run(["bundle", "install"], cwd=test_dir, check=True)
            
        return test_dir
        
    async def execute_code(self, test_dir: Path, test_case: IntegrationTestCase) -> Tuple[int, str, str]:
        main_file = test_dir / "main.rb"
        if not main_file.exists():
            rb_files = list(test_dir.glob("*.rb"))
            if rb_files:
                main_file = rb_files[0]
                
        # Use bundle exec if Gemfile exists
        if (test_dir / "Gemfile").exists():
            cmd = ["bundle", "exec", "ruby", str(main_file)]
        else:
            cmd = ["ruby", str(main_file)]
            
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=test_dir,
            env={**os.environ, **test_case.environment}
        )
        
        try:
            stdout, stderr = process.communicate(timeout=test_case.timeout)
            return process.returncode, stdout.decode(), stderr.decode()
        except subprocess.TimeoutExpired:
            process.kill()
            return -1, "", "Test timeout exceeded"
            
    async def validate_environment(self, test_dir: Path) -> bool:
        result = subprocess.run(["ruby", "--version"], capture_output=True, text=True)
        return result.returncode == 0


class PHPIntegrationTestRunner(LanguageIntegrationTestRunner):
    """PHP language integration test runner."""
    
    async def setup_environment(self, test_case: IntegrationTestCase) -> Path:
        test_dir = Path(tempfile.mkdtemp(prefix="homeostasis_php_test_"))
        
        # Write source files
        for filename, content in test_case.source_code.items():
            file_path = test_dir / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)
            
        # Create composer.json if dependencies exist
        if test_case.dependencies:
            composer_json = {
                "require": {}
            }
            for dep in test_case.dependencies:
                if ":" in dep:
                    name, version = dep.split(":", 1)
                    composer_json["require"][name] = version
                else:
                    composer_json["require"][dep] = "*"
                    
            (test_dir / "composer.json").write_text(json.dumps(composer_json, indent=2))
            
            # Run composer install
            subprocess.run(["composer", "install"], cwd=test_dir, check=True)
            
        return test_dir
        
    async def execute_code(self, test_dir: Path, test_case: IntegrationTestCase) -> Tuple[int, str, str]:
        main_file = test_dir / "index.php"
        if not main_file.exists():
            php_files = list(test_dir.glob("*.php"))
            if php_files:
                main_file = php_files[0]
                
        process = subprocess.Popen(
            ["php", str(main_file)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=test_dir,
            env={**os.environ, **test_case.environment}
        )
        
        try:
            stdout, stderr = process.communicate(timeout=test_case.timeout)
            return process.returncode, stdout.decode(), stderr.decode()
        except subprocess.TimeoutExpired:
            process.kill()
            return -1, "", "Test timeout exceeded"
            
    async def validate_environment(self, test_dir: Path) -> bool:
        result = subprocess.run(["php", "--version"], capture_output=True, text=True)
        return result.returncode == 0


class CSharpIntegrationTestRunner(LanguageIntegrationTestRunner):
    """C# language integration test runner."""
    
    async def setup_environment(self, test_case: IntegrationTestCase) -> Path:
        test_dir = Path(tempfile.mkdtemp(prefix="homeostasis_csharp_test_"))
        
        # Create .NET project
        project_name = f"HomeostasisTest{test_case.name.replace(' ', '')}".replace("-", "")
        subprocess.run(["dotnet", "new", "console", "-n", project_name], cwd=test_dir, check=True)
        
        project_dir = test_dir / project_name
        
        # Write source files
        for filename, content in test_case.source_code.items():
            if filename == "Program.cs":
                file_path = project_dir / filename
            else:
                file_path = project_dir / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)
            
        # Add NuGet packages
        for dep in test_case.dependencies:
            subprocess.run(["dotnet", "add", "package", dep], cwd=project_dir, check=True)
            
        return project_dir
        
    async def execute_code(self, test_dir: Path, test_case: IntegrationTestCase) -> Tuple[int, str, str]:
        process = subprocess.Popen(
            ["dotnet", "run"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=test_dir,
            env={**os.environ, **test_case.environment}
        )
        
        try:
            stdout, stderr = process.communicate(timeout=test_case.timeout)
            return process.returncode, stdout.decode(), stderr.decode()
        except subprocess.TimeoutExpired:
            process.kill()
            return -1, "", "Test timeout exceeded"
            
    async def validate_environment(self, test_dir: Path) -> bool:
        result = subprocess.run(["dotnet", "--version"], capture_output=True, text=True)
        return result.returncode == 0


class SwiftIntegrationTestRunner(LanguageIntegrationTestRunner):
    """Swift language integration test runner."""
    
    async def setup_environment(self, test_case: IntegrationTestCase) -> Path:
        test_dir = Path(tempfile.mkdtemp(prefix="homeostasis_swift_test_"))
        
        # Create Swift package
        subprocess.run(["swift", "package", "init", "--type", "executable"], cwd=test_dir, check=True)
        
        # Write source files
        sources_dir = test_dir / "Sources" / test_dir.name
        for filename, content in test_case.source_code.items():
            if filename.endswith(".swift"):
                file_path = sources_dir / filename
            else:
                file_path = test_dir / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)
            
        # Update Package.swift for dependencies
        if test_case.dependencies:
            # This would require parsing and updating Package.swift
            # For now, we'll assume dependencies are manually added
            pass
            
        return test_dir
        
    async def execute_code(self, test_dir: Path, test_case: IntegrationTestCase) -> Tuple[int, str, str]:
        process = subprocess.Popen(
            ["swift", "run"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=test_dir,
            env={**os.environ, **test_case.environment}
        )
        
        try:
            stdout, stderr = process.communicate(timeout=test_case.timeout)
            return process.returncode, stdout.decode(), stderr.decode()
        except subprocess.TimeoutExpired:
            process.kill()
            return -1, "", "Test timeout exceeded"
            
    async def validate_environment(self, test_dir: Path) -> bool:
        result = subprocess.run(["swift", "--version"], capture_output=True, text=True)
        return result.returncode == 0


class KotlinIntegrationTestRunner(LanguageIntegrationTestRunner):
    """Kotlin language integration test runner."""
    
    async def setup_environment(self, test_case: IntegrationTestCase) -> Path:
        test_dir = Path(tempfile.mkdtemp(prefix="homeostasis_kotlin_test_"))
        
        # Write source files
        for filename, content in test_case.source_code.items():
            file_path = test_dir / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)
            
        # Create build.gradle.kts if dependencies exist
        if test_case.dependencies:
            build_gradle = """
plugins {
    kotlin("jvm") version "1.9.0"
    application
}

repositories {
    mavenCentral()
}

dependencies {
"""
            for dep in test_case.dependencies:
                build_gradle += f'    implementation("{dep}")\n'
                
            build_gradle += """
}

application {
    mainClass.set("MainKt")
}
"""
            (test_dir / "build.gradle.kts").write_text(build_gradle)
            
        return test_dir
        
    async def execute_code(self, test_dir: Path, test_case: IntegrationTestCase) -> Tuple[int, str, str]:
        # Find main file
        main_file = test_dir / "main.kt"
        if not main_file.exists():
            kt_files = list(test_dir.glob("*.kt"))
            if kt_files:
                main_file = kt_files[0]
                
        if (test_dir / "build.gradle.kts").exists():
            # Use Gradle
            process = subprocess.Popen(
                ["gradle", "run"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=test_dir,
                env={**os.environ, **test_case.environment}
            )
        else:
            # Use kotlinc directly
            # Compile
            subprocess.run(["kotlinc", str(main_file), "-d", str(test_dir)], check=True)
            
            # Run
            process = subprocess.Popen(
                ["kotlin", "-cp", str(test_dir), "MainKt"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=test_dir,
                env={**os.environ, **test_case.environment}
            )
            
        try:
            stdout, stderr = process.communicate(timeout=test_case.timeout)
            return process.returncode, stdout.decode(), stderr.decode()
        except subprocess.TimeoutExpired:
            process.kill()
            return -1, "", "Test timeout exceeded"
            
    async def validate_environment(self, test_dir: Path) -> bool:
        result = subprocess.run(["kotlin", "-version"], capture_output=True, text=True)
        return result.returncode == 0


# Map of all language runners
LANGUAGE_RUNNERS = {
    "python": PythonIntegrationTestRunner,
    "javascript": JavaScriptIntegrationTestRunner,
    "typescript": TypeScriptIntegrationTestRunner,
    "go": GoIntegrationTestRunner,
    "rust": RustIntegrationTestRunner,
    "java": JavaIntegrationTestRunner,
    "ruby": RubyIntegrationTestRunner,
    "php": PHPIntegrationTestRunner,
    "csharp": CSharpIntegrationTestRunner,
    "swift": SwiftIntegrationTestRunner,
    "kotlin": KotlinIntegrationTestRunner,
    # Add more as needed
}


def get_runner_for_language(language: str, plugin: Any) -> LanguageIntegrationTestRunner:
    """Get the appropriate runner for a language."""
    runner_class = LANGUAGE_RUNNERS.get(language, LanguageIntegrationTestRunner)
    return runner_class(language, plugin)