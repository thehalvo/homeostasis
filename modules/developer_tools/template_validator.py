"""
Template Validation CLI Tool

This module provides a CLI tool for validating healing templates used in the
Homeostasis framework.
"""

import argparse
import ast
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from jinja2 import Environment, TemplateSyntaxError, meta

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of template validation"""

    template_path: Path
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_issues(self) -> bool:
        return len(self.errors) > 0 or len(self.warnings) > 0


@dataclass
class TemplateMetadata:
    """Metadata extracted from template"""

    language: str
    error_type: str
    description: str
    variables: List[str]
    required_imports: List[str] = field(default_factory=list)
    supported_frameworks: List[str] = field(default_factory=list)
    min_version: Optional[str] = None
    max_version: Optional[str] = None
    tags: List[str] = field(default_factory=list)


class TemplateValidator:
    """Validator for healing templates"""

    def __init__(self, templates_dir: Optional[Path] = None):
        self.templates_dir = (
            templates_dir or
            Path(__file__).parent.parent / "patch_generation" / "templates"
        )
        self.jinja_env = Environment(autoescape=True)
        self.language_validators = {
            "python": self._validate_python_template,
            "javascript": self._validate_javascript_template,
            "typescript": self._validate_typescript_template,
            "go": self._validate_go_template,
            "java": self._validate_java_template,
            "ruby": self._validate_ruby_template,
            "rust": self._validate_rust_template,
            "cpp": self._validate_cpp_template,
            "csharp": self._validate_csharp_template,
        }

    def validate_template(self, template_path: Path) -> ValidationResult:
        """Validate a single template"""
        result = ValidationResult(template_path=template_path, is_valid=True)

        # Check file exists
        if not template_path.exists():
            result.is_valid = False
            result.errors.append(f"Template file not found: {template_path}")
            return result

        # Read template content
        try:
            content = template_path.read_text()
        except Exception as e:
            result.is_valid = False
            result.errors.append(f"Failed to read template: {str(e)}")
            return result

        # Validate syntax
        self._validate_syntax(content, result)

        # Extract and validate metadata
        metadata = self._extract_metadata(content, template_path, result)
        if metadata:
            result.metadata = metadata.__dict__

            # Language-specific validation
            if metadata.language in self.language_validators:
                self.language_validators[metadata.language](content, metadata, result)

        # Validate template structure
        self._validate_structure(content, result)

        # Validate variable usage
        self._validate_variables(content, result)

        return result

    def _validate_syntax(self, content: str, result: ValidationResult):
        """Validate Jinja2 template syntax"""
        try:
            # Parse as Jinja2 template
            self.jinja_env.parse(content)
        except TemplateSyntaxError as e:
            result.is_valid = False
            result.errors.append(f"Template syntax error: {str(e)}")

    def _extract_metadata(
        self, content: str, template_path: Path, result: ValidationResult
    ) -> Optional[TemplateMetadata]:
        """Extract metadata from template"""
        # Try to extract from template comments
        metadata_match = re.search(r"{#\s*metadata:\s*(.+?)\s*#}", content, re.DOTALL)

        if metadata_match:
            try:
                metadata_str = metadata_match.group(1)
                metadata_dict = yaml.safe_load(metadata_str)

                # Validate required fields
                if "language" not in metadata_dict:
                    result.errors.append("Missing required metadata field: language")
                    return None

                if "error_type" not in metadata_dict:
                    result.errors.append("Missing required metadata field: error_type")
                    return None

                return TemplateMetadata(
                    language=metadata_dict["language"],
                    error_type=metadata_dict["error_type"],
                    description=metadata_dict.get("description", ""),
                    variables=metadata_dict.get("variables", []),
                    required_imports=metadata_dict.get("required_imports", []),
                    supported_frameworks=metadata_dict.get("supported_frameworks", []),
                    min_version=metadata_dict.get("min_version"),
                    max_version=metadata_dict.get("max_version"),
                    tags=metadata_dict.get("tags", []),
                )
            except Exception as e:
                result.warnings.append(f"Failed to parse metadata: {str(e)}")
        else:
            # Try to infer from path and content
            language = self._infer_language(template_path)
            error_type = self._infer_error_type(template_path)

            if not language:
                result.warnings.append("Could not determine template language")
                return None

            # Extract variables from template
            ast_tree = self.jinja_env.parse(content)
            variables = list(meta.find_undeclared_variables(ast_tree))

            return TemplateMetadata(
                language=language,
                error_type=error_type,
                description="",
                variables=variables,
            )

    def _infer_language(self, template_path: Path) -> Optional[str]:
        """Infer language from template path"""
        # Check file extension
        ext_mapping = {
            ".py.template": "python",
            ".js.template": "javascript",
            ".ts.template": "typescript",
            ".go.template": "go",
            ".java.template": "java",
            ".rb.template": "ruby",
            ".rs.template": "rust",
            ".cpp.template": "cpp",
            ".cs.template": "csharp",
        }

        for ext, lang in ext_mapping.items():
            if str(template_path).endswith(ext):
                return lang

        # Check parent directory
        parent_name = template_path.parent.name.lower()
        if parent_name in [
            "python",
            "javascript",
            "typescript",
            "go",
            "java",
            "ruby",
            "rust",
            "cpp",
            "csharp",
        ]:
            return parent_name

        return None

    def _infer_error_type(self, template_path: Path) -> str:
        """Infer error type from template filename"""
        filename = template_path.stem
        # Remove language extension if present
        filename = re.sub(r"\.(py|js|ts|go|java|rb|rs|cpp|cs)$", "", filename)
        return filename

    def _validate_structure(self, content: str, result: ValidationResult):
        """Validate template structure"""
        # Check for common template patterns
        if "{% if" in content and "{% endif" not in content:
            result.errors.append("Unclosed if statement")

        if "{% for" in content and "{% endfor" not in content:
            result.errors.append("Unclosed for loop")

        # Check for balanced braces
        open_braces = content.count("{%")
        close_braces = content.count("%}")
        if open_braces != close_braces:
            result.errors.append(
                f"Unbalanced template tags: {open_braces} open, {close_braces} close"
            )

    def _validate_variables(self, content: str, result: ValidationResult):
        """Validate variable usage in template"""
        try:
            ast_tree = self.jinja_env.parse(content)
            variables = list(meta.find_undeclared_variables(ast_tree))

            # Check for common required variables
            common_vars = ["error_message", "file_path", "line_number"]
            missing_common = [v for v in common_vars if v not in variables]

            if missing_common:
                result.warnings.append(
                    f"Template may be missing common variables: {', '.join(missing_common)}"
                )

            # Check for unused variables in metadata
            if "variables" in result.metadata:
                declared_vars = result.metadata["variables"]
                unused_vars = [v for v in declared_vars if v not in variables]
                if unused_vars:
                    result.warnings.append(
                        f"Declared variables not used in template: {', '.join(unused_vars)}"
                    )
        except Exception as e:
            result.warnings.append(f"Could not analyze variables: {str(e)}")

    def _validate_python_template(
        self, content: str, metadata: TemplateMetadata, result: ValidationResult
    ):
        """Validate Python-specific template"""
        # Render a test version
        test_vars = {
            "error_message": "Test error",
            "file_path": "test.py",
            "line_number": 10,
            "class_name": "TestClass",
            "function_name": "test_function",
            "variable_name": "test_var",
            "module_name": "test_module",
        }

        try:
            rendered = self.jinja_env.from_string(content).render(**test_vars)

            # Try to parse as Python
            ast.parse(rendered)
        except SyntaxError as e:
            result.errors.append(f"Generated Python code has syntax error: {str(e)}")
        except Exception as e:
            result.warnings.append(f"Could not validate Python syntax: {str(e)}")

        # Check for common Python patterns
        if "import" in content and "{% if" not in content:
            result.warnings.append(
                "Import statements should be conditional on requirements"
            )

    def _validate_javascript_template(
        self, content: str, metadata: TemplateMetadata, result: ValidationResult
    ):
        """Validate JavaScript-specific template"""
        # Check for common JS patterns
        if "const" not in content and "let" not in content and "var" not in content:
            result.warnings.append("Template doesn't declare any variables")

        # Check for proper error handling
        if "try" in content and "catch" not in content:
            result.errors.append("Try block without catch")

    def _validate_typescript_template(
        self, content: str, metadata: TemplateMetadata, result: ValidationResult
    ):
        """Validate TypeScript-specific template"""
        # Similar to JavaScript but check for type annotations
        self._validate_javascript_template(content, metadata, result)

        if ": any" in content:
            result.warnings.append(
                "Template uses 'any' type, consider more specific types"
            )

    def _validate_go_template(
        self, content: str, metadata: TemplateMetadata, result: ValidationResult
    ):
        """Validate Go-specific template"""
        # Check for error handling
        if "if err != nil" not in content and "error" in metadata.error_type.lower():
            result.warnings.append(
                "Go template for error handling should check 'err != nil'"
            )

        # Check for proper imports
        if "import" in content and "(" not in content:
            result.warnings.append(
                "Go imports should use parentheses for multiple imports"
            )

    def _validate_java_template(
        self, content: str, metadata: TemplateMetadata, result: ValidationResult
    ):
        """Validate Java-specific template"""
        # Check for class structure
        if "class" not in content and "interface" not in content:
            result.warnings.append("Java template should define a class or interface")

        # Check for exception handling
        if "Exception" in content and "try" not in content:
            result.warnings.append("Exception handling should use try-catch blocks")

    def _validate_ruby_template(
        self, content: str, metadata: TemplateMetadata, result: ValidationResult
    ):
        """Validate Ruby-specific template"""
        # Check for Ruby idioms
        if "begin" in content and "rescue" not in content:
            result.errors.append("Begin block without rescue")

    def _validate_rust_template(
        self, content: str, metadata: TemplateMetadata, result: ValidationResult
    ):
        """Validate Rust-specific template"""
        # Check for Result handling
        if "Result<" in content and "?" not in content and "unwrap" not in content:
            result.warnings.append("Result type should be handled with ? or unwrap")

    def _validate_cpp_template(
        self, content: str, metadata: TemplateMetadata, result: ValidationResult
    ):
        """Validate C++-specific template"""
        # Check for memory management
        if "new" in content and "delete" not in content:
            result.warnings.append(
                "Memory allocated with 'new' should be freed with 'delete'"
            )

    def _validate_csharp_template(
        self, content: str, metadata: TemplateMetadata, result: ValidationResult
    ):
        """Validate C#-specific template"""
        # Check for using statements
        if "IDisposable" in content and "using" not in content:
            result.warnings.append(
                "IDisposable objects should be used with 'using' statement"
            )

    def validate_directory(
        self, directory: Path, recursive: bool = True
    ) -> List[ValidationResult]:
        """Validate all templates in a directory"""
        results = []

        pattern = "**/*.template" if recursive else "*.template"
        for template_path in directory.glob(pattern):
            if template_path.is_file():
                result = self.validate_template(template_path)
                results.append(result)

        return results

    def generate_report(
        self, results: List[ValidationResult], format: str = "text"
    ) -> str:
        """Generate validation report"""
        if format == "text":
            return self._generate_text_report(results)
        elif format == "json":
            return self._generate_json_report(results)
        elif format == "markdown":
            return self._generate_markdown_report(results)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _generate_text_report(self, results: List[ValidationResult]) -> str:
        """Generate text format report"""
        report = ["Template Validation Report", "=" * 50, ""]

        total = len(results)
        valid = sum(1 for r in results if r.is_valid and not r.has_issues)
        with_warnings = sum(1 for r in results if r.is_valid and r.warnings)
        invalid = sum(1 for r in results if not r.is_valid)

        report.append(f"Total templates: {total}")
        report.append(f"Valid: {valid}")
        report.append(f"Valid with warnings: {with_warnings}")
        report.append(f"Invalid: {invalid}")
        report.append("")

        for result in results:
            if not result.is_valid or result.has_issues:
                report.append(f"Template: {result.template_path}")

                if result.errors:
                    report.append("  Errors:")
                    for error in result.errors:
                        report.append(f"    - {error}")

                if result.warnings:
                    report.append("  Warnings:")
                    for warning in result.warnings:
                        report.append(f"    - {warning}")

                report.append("")

        return "\n".join(report)

    def _generate_json_report(self, results: List[ValidationResult]) -> str:
        """Generate JSON format report"""
        data = {
            "summary": {
                "total": len(results),
                "valid": sum(1 for r in results if r.is_valid and not r.has_issues),
                "with_warnings": sum(1 for r in results if r.is_valid and r.warnings),
                "invalid": sum(1 for r in results if not r.is_valid),
            },
            "results": [
                {
                    "template": str(r.template_path),
                    "is_valid": r.is_valid,
                    "errors": r.errors,
                    "warnings": r.warnings,
                    "metadata": r.metadata,
                }
                for r in results
            ],
        }

        return json.dumps(data, indent=2)

    def _generate_markdown_report(self, results: List[ValidationResult]) -> str:
        """Generate Markdown format report"""
        report = ["# Template Validation Report", ""]

        total = len(results)
        valid = sum(1 for r in results if r.is_valid and not r.has_issues)
        with_warnings = sum(1 for r in results if r.is_valid and r.warnings)
        invalid = sum(1 for r in results if not r.is_valid)

        report.append("## Summary")
        report.append("")
        report.append(f"- **Total templates**: {total}")
        report.append(f"- **Valid**: {valid}")
        report.append(f"- **Valid with warnings**: {with_warnings}")
        report.append(f"- **Invalid**: {invalid}")
        report.append("")

        if any(not r.is_valid or r.has_issues for r in results):
            report.append("## Issues")
            report.append("")

            for result in results:
                if not result.is_valid or result.has_issues:
                    report.append(f"### {result.template_path}")
                    report.append("")

                    if result.errors:
                        report.append("**Errors:**")
                        for error in result.errors:
                            report.append(f"- {error}")
                        report.append("")

                    if result.warnings:
                        report.append("**Warnings:**")
                        for warning in result.warnings:
                            report.append(f"- {warning}")
                        report.append("")

        return "\n".join(report)


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Validate Homeostasis healing templates"
    )

    parser.add_argument("path", type=Path, help="Path to template file or directory")

    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Recursively validate templates in subdirectories",
    )

    parser.add_argument(
        "-f",
        "--format",
        choices=["text", "json", "markdown"],
        default="text",
        help="Output format (default: text)",
    )

    parser.add_argument(
        "-o", "--output", type=Path, help="Output file (default: stdout)"
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )

    parser.add_argument(
        "--fail-on-warning",
        action="store_true",
        help="Exit with non-zero code if warnings are found",
    )

    args = parser.parse_args()

    # Set up logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Create validator
    validator = TemplateValidator()

    # Validate templates
    if args.path.is_file():
        results = [validator.validate_template(args.path)]
    elif args.path.is_dir():
        results = validator.validate_directory(args.path, recursive=args.recursive)
    else:
        print(f"Error: {args.path} is not a valid file or directory")
        return 1

    # Generate report
    report = validator.generate_report(results, format=args.format)

    # Output report
    if args.output:
        args.output.write_text(report)
        print(f"Report written to {args.output}")
    else:
        print(report)

    # Determine exit code
    invalid_count = sum(1 for r in results if not r.is_valid)
    warning_count = sum(len(r.warnings) for r in results)

    if invalid_count > 0:
        return 1
    elif args.fail_on_warning and warning_count > 0:
        return 1
    else:
        return 0


if __name__ == "__main__":
    exit(main())
