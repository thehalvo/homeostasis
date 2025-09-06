"""
Test Suite Generator for All Supported Languages

This module automatically generates comprehensive test suites for all 40+ languages
supported by Homeostasis, including framework-specific and cross-language tests.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class CaseTemplate:
    """Template for generating test cases."""

    name: str
    description: str
    error_pattern: str
    fix_pattern: str
    tags: List[str]


# Common error patterns across languages
COMMON_ERROR_PATTERNS = {
    "null_safety": {
        "description": "Null/nil/undefined reference errors",
        "patterns": {
            "python": {
                "error": "AttributeError",
                "code": "obj.attr",
                "fix": "getattr(obj, 'attr', default)",
            },
            "javascript": {
                "error": "TypeError",
                "code": "obj.prop",
                "fix": "obj?.prop",
            },
            "java": {
                "error": "NullPointerException",
                "code": "obj.method()",
                "fix": "if (obj != null) obj.method()",
            },
            "go": {"error": "panic", "code": "*ptr", "fix": "if ptr != nil { *ptr }"},
            "rust": {
                "error": "panic",
                "code": ".unwrap()",
                "fix": ".unwrap_or_default()",
            },
            "ruby": {
                "error": "NoMethodError",
                "code": "obj.method",
                "fix": "obj&.method",
            },
            "csharp": {
                "error": "NullReferenceException",
                "code": "obj.Method()",
                "fix": "obj?.Method()",
            },
            "swift": {
                "error": "Fatal error",
                "code": "obj!",
                "fix": "obj ?? defaultValue",
            },
            "kotlin": {
                "error": "NullPointerException",
                "code": "obj!!.method()",
                "fix": "obj?.method()",
            },
            "php": {
                "error": "Fatal error",
                "code": "$obj->method()",
                "fix": "$obj?->method()",
            },
        },
    },
    "index_bounds": {
        "description": "Array/list index out of bounds errors",
        "patterns": {
            "python": {
                "error": "IndexError",
                "code": "arr[i]",
                "fix": "arr[i] if i < len(arr) else None",
            },
            "javascript": {
                "error": "undefined",
                "code": "arr[i]",
                "fix": "arr[i] ?? defaultValue",
            },
            "java": {
                "error": "ArrayIndexOutOfBoundsException",
                "code": "arr[i]",
                "fix": "i < arr.length ? arr[i] : null",
            },
            "go": {
                "error": "panic",
                "code": "slice[i]",
                "fix": "if i < len(slice) { slice[i] }",
            },
            "rust": {"error": "panic", "code": "vec[i]", "fix": "vec.get(i)"},
            "ruby": {"error": "nil", "code": "arr[i]", "fix": "arr[i] || default"},
            "csharp": {
                "error": "IndexOutOfRangeException",
                "code": "arr[i]",
                "fix": "i < arr.Length ? arr[i] : default",
            },
            "swift": {
                "error": "Fatal error",
                "code": "arr[i]",
                "fix": "i < arr.count ? arr[i] : nil",
            },
            "kotlin": {
                "error": "IndexOutOfBoundsException",
                "code": "list[i]",
                "fix": "list.getOrNull(i)",
            },
            "php": {
                "error": "Undefined offset",
                "code": "$arr[$i]",
                "fix": "$arr[$i] ?? $default",
            },
        },
    },
    "type_errors": {
        "description": "Type mismatch and casting errors",
        "patterns": {
            "python": {
                "error": "TypeError",
                "code": "int(value)",
                "fix": "int(value) if value.isdigit() else 0",
            },
            "javascript": {
                "error": "TypeError",
                "code": "parseInt(value)",
                "fix": "parseInt(value) || 0",
            },
            "java": {
                "error": "ClassCastException",
                "code": "(Type)obj",
                "fix": "obj instanceof Type ? (Type)obj : null",
            },
            "go": {"error": "panic", "code": "v.(Type)", "fix": "v, ok := v.(Type)"},
            "rust": {"error": "compile error", "code": "as Type", "fix": "try_into()"},
            "ruby": {
                "error": "TypeError",
                "code": "Integer(value)",
                "fix": "Integer(value) rescue 0",
            },
            "csharp": {
                "error": "InvalidCastException",
                "code": "(Type)obj",
                "fix": "obj as Type",
            },
            "swift": {"error": "Fatal error", "code": "as! Type", "fix": "as? Type"},
            "kotlin": {
                "error": "ClassCastException",
                "code": "as Type",
                "fix": "as? Type",
            },
            "typescript": {
                "error": "TypeError",
                "code": "value as Type",
                "fix": "value as Type | undefined",
            },
        },
    },
    "concurrency": {
        "description": "Race conditions and concurrency errors",
        "patterns": {
            "go": {
                "error": "race condition",
                "code": "map[key] = value",
                "fix": "mutex.Lock(); map[key] = value; mutex.Unlock()",
            },
            "java": {
                "error": "ConcurrentModificationException",
                "code": "list.add(item)",
                "fix": "synchronized(list) { list.add(item) }",
            },
            "rust": {
                "error": "compile error",
                "code": "data.push(item)",
                "fix": "let mut data = data.lock().unwrap(); data.push(item)",
            },
            "python": {
                "error": "RuntimeError",
                "code": "dict[key] = value",
                "fix": "with lock: dict[key] = value",
            },
            "csharp": {
                "error": "InvalidOperationException",
                "code": "list.Add(item)",
                "fix": "lock(list) { list.Add(item) }",
            },
            "swift": {
                "error": "EXC_BAD_ACCESS",
                "code": "array.append(item)",
                "fix": "queue.sync { array.append(item) }",
            },
            "kotlin": {
                "error": "ConcurrentModificationException",
                "code": "list.add(item)",
                "fix": "synchronized(list) { list.add(item) }",
            },
            "ruby": {
                "error": "ThreadError",
                "code": "array << item",
                "fix": "mutex.synchronize { array << item }",
            },
        },
    },
    "resource_handling": {
        "description": "Resource leaks and cleanup errors",
        "patterns": {
            "python": {
                "error": "ResourceWarning",
                "code": "f = open(file)",
                "fix": "with open(file) as f:",
            },
            "java": {
                "error": "Resource leak",
                "code": "new FileReader(file)",
                "fix": "try (FileReader fr = new FileReader(file))",
            },
            "go": {
                "error": "resource leak",
                "code": "file.Open()",
                "fix": "defer file.Close()",
            },
            "csharp": {
                "error": "Resource leak",
                "code": "new FileStream()",
                "fix": "using (var fs = new FileStream())",
            },
            "rust": {
                "error": "compile warning",
                "code": "File::open()",
                "fix": "let _file = File::open()?",
            },
            "swift": {
                "error": "Memory leak",
                "code": "FileHandle()",
                "fix": "defer { fileHandle.close() }",
            },
            "php": {
                "error": "Resource leak",
                "code": "fopen()",
                "fix": "try { $f = fopen(); } finally { fclose($f); }",
            },
        },
    },
}


class SuiteGenerator:
    """Generates comprehensive test suites for all supported languages."""

    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("test_suites")
        self.languages = self._get_all_languages()
        self.frameworks = self._get_frameworks_by_language()

    def _get_all_languages(self) -> List[str]:
        """Get list of all supported languages."""
        return [
            "python",
            "javascript",
            "typescript",
            "java",
            "go",
            "rust",
            "ruby",
            "php",
            "csharp",
            "swift",
            "kotlin",
            "scala",
            "elixir",
            "erlang",
            "clojure",
            "haskell",
            "fsharp",
            "lua",
            "r",
            "matlab",
            "julia",
            "nim",
            "crystal",
            "zig",
            "powershell",
            "bash",
            "sql",
            "yaml_json",
            "terraform",
            "dockerfile",
            "ansible",
            "cpp",
            "objc",
            "perl",
            "dart",
            "groovy",
            "vb",
            "fortran",
            "cobol",
            "pascal",
            "ada",
            "d",
            "ocaml",
            "scheme",
            "racket",
            "prolog",
        ]

    def _get_frameworks_by_language(self) -> Dict[str, List[str]]:
        """Get frameworks for each language."""
        return {
            "python": ["django", "flask", "fastapi", "sqlalchemy", "pytest"],
            "javascript": ["react", "vue", "angular", "express", "nextjs"],
            "typescript": ["angular", "react", "vue", "nestjs"],
            "java": ["spring", "hibernate", "junit", "maven", "gradle"],
            "go": ["gin", "echo", "gorm", "testify"],
            "rust": ["actix", "rocket", "diesel", "tokio"],
            "ruby": ["rails", "sinatra", "rspec"],
            "php": ["laravel", "symfony", "phpunit"],
            "csharp": ["aspnetcore", "entityframework", "nunit"],
            "swift": ["swiftui", "uikit", "combine", "coredata"],
            "kotlin": ["spring", "ktor", "exposed", "junit"],
            "elixir": ["phoenix", "ecto", "exunit"],
            "clojure": ["ring", "compojure", "core.async"],
        }

    def generate_all_test_suites(self):
        """Generate test suites for all supported languages."""
        for language in self.languages:
            logger.info(f"Generating test suite for {language}")

            # Create language directory
            lang_dir = self.output_dir / language
            lang_dir.mkdir(parents=True, exist_ok=True)

            # Generate basic error tests
            basic_tests = self._generate_basic_error_tests(language)
            if basic_tests:
                self._save_test_suite(lang_dir / "basic_errors.json", basic_tests)

            # Generate framework-specific tests
            if language in self.frameworks:
                framework_tests = self._generate_framework_tests(language)
                if framework_tests:
                    self._save_test_suite(
                        lang_dir / "framework_errors.json", framework_tests
                    )

            # Generate cross-language tests if applicable
            cross_lang_tests = self._generate_cross_language_tests(language)
            if cross_lang_tests:
                self._save_test_suite(
                    lang_dir / "cross_language_errors.json", cross_lang_tests
                )

    def _generate_basic_error_tests(self, language: str) -> List[Dict[str, Any]]:
        """Generate basic error test cases for a language."""
        tests = []

        for error_type, error_info in COMMON_ERROR_PATTERNS.items():
            if language in error_info["patterns"]:
                pattern = error_info["patterns"][language]

                test_case = {
                    "name": f"{language}_{error_type}",
                    "language": language,
                    "description": f"Test {error_info['description']} in {language}",
                    "test_type": "single",
                    "source_code": self._generate_error_code(
                        language, error_type, pattern
                    ),
                    "expected_errors": [
                        {"error_type": pattern["error"], "message": pattern["error"]}
                    ],
                    "expected_fixes": [
                        {
                            "fix_type": error_type,
                            "description": f"Apply {error_type} fix pattern",
                        }
                    ],
                    "environment": {},
                    "dependencies": [],
                    "frameworks": [],
                    "tags": ["basic", error_type],
                }

                tests.append(test_case)

        return tests

    def _generate_error_code(
        self, language: str, error_type: str, pattern: Dict[str, str]
    ) -> Dict[str, str]:
        """Generate source code that will produce the specified error."""
        templates = {
            "python": {
                "null_safety": """
def process_user(user):
    # This will cause AttributeError if user is None
    return user.name.upper()

user = None
result = process_user(user)
print(result)
""",
                "index_bounds": """
def get_item(items, index):
    # This will cause IndexError if index >= len(items)
    return items[index]

items = [1, 2, 3]
result = get_item(items, 10)
print(result)
""",
                "type_errors": """
def calculate(value):
    # This will cause TypeError if value is not numeric
    return int(value) * 2

result = calculate("not a number")
print(result)
""",
            },
            "javascript": {
                "null_safety": """
function getProperty(obj) {
    // This will cause TypeError if obj is null/undefined
    return obj.property.value;
}

const obj = null;
console.log(getProperty(obj));
""",
                "index_bounds": """
function getElement(arr, index) {
    // This may return undefined
    return arr[index].toString();
}

const arr = [1, 2, 3];
console.log(getElement(arr, 10));
""",
                "type_errors": """
function add(a, b) {
    // This will cause TypeError if a or b is not a number
    return a + b;
}

console.log(add(5, "not a number"));
""",
            },
            "java": {
                "null_safety": """
public class Main {
    public static void main(String[] args) {
        String str = null;
        // This will cause NullPointerException
        System.out.println(str.length());
    }
}
""",
                "index_bounds": """
public class Main {
    public static void main(String[] args) {
        int[] arr = {1, 2, 3};
        // This will cause ArrayIndexOutOfBoundsException
        System.out.println(arr[10]);
    }
}
""",
                "type_errors": """
public class Main {
    public static void main(String[] args) {
        Object obj = "string";
        // This will cause ClassCastException
        Integer num = (Integer) obj;
        System.out.println(num);
    }
}
""",
            },
            # Add more language templates as needed
        }

        # Get template for language and error type
        if language in templates and error_type in templates[language]:
            filename = self._get_main_filename(language)
            return {filename: templates[language][error_type]}
        else:
            # Generate generic code based on pattern
            filename = self._get_main_filename(language)
            code = self._generate_generic_error_code(language, error_type, pattern)
            return {filename: code}

    def _get_main_filename(self, language: str) -> str:
        """Get the main filename for a language."""
        extensions = {
            "python": "main.py",
            "javascript": "index.js",
            "typescript": "index.ts",
            "java": "Main.java",
            "go": "main.go",
            "rust": "main.rs",
            "ruby": "main.rb",
            "php": "index.php",
            "csharp": "Program.cs",
            "swift": "main.swift",
            "kotlin": "Main.kt",
            "scala": "Main.scala",
            "elixir": "main.ex",
            "erlang": "main.erl",
            "clojure": "main.clj",
            "haskell": "Main.hs",
            "fsharp": "Program.fs",
            "lua": "main.lua",
            "r": "main.R",
            "matlab": "main.m",
            "julia": "main.jl",
            "nim": "main.nim",
            "crystal": "main.cr",
            "zig": "main.zig",
            "powershell": "main.ps1",
            "bash": "main.sh",
            "sql": "main.sql",
            "yaml_json": "config.yaml",
            "terraform": "main.tf",
            "dockerfile": "Dockerfile",
            "ansible": "playbook.yml",
            "cpp": "main.cpp",
            "objc": "main.m",
            "perl": "main.pl",
            "dart": "main.dart",
        }
        return extensions.get(language, f"main.{language}")

    def _generate_generic_error_code(
        self, language: str, error_type: str, pattern: Dict[str, str]
    ) -> str:
        """Generate generic error code when no template exists."""
        # This is a simplified version - in practice, you'd want more sophisticated generation
        error_code = pattern.get("code", "")
        return f"""
# Auto-generated test case for {language} - {error_type}
# This code will produce: {pattern.get('error', 'error')}
# Error pattern: {error_code}
# Fix pattern: {pattern.get('fix', 'fix')}

# TODO: Implement actual {language} code that produces this error
"""

    def _generate_framework_tests(self, language: str) -> List[Dict[str, Any]]:
        """Generate framework-specific test cases."""
        tests = []
        frameworks = self.frameworks.get(language, [])

        for framework in frameworks:
            # Generate common framework errors
            test_case = {
                "name": f"{language}_{framework}_config_error",
                "language": language,
                "description": f"Test {framework} configuration error",
                "test_type": "framework",
                "source_code": self._generate_framework_error_code(language, framework),
                "expected_errors": [
                    {
                        "error_type": "ConfigurationError",
                        "message": f"{framework} configuration error",
                    }
                ],
                "expected_fixes": [
                    {
                        "fix_type": "framework_config",
                        "description": f"Fix {framework} configuration",
                    }
                ],
                "environment": {},
                "dependencies": [framework],
                "frameworks": [framework],
                "tags": ["framework", framework],
            }
            tests.append(test_case)

        return tests

    def _generate_framework_error_code(
        self, language: str, framework: str
    ) -> Dict[str, str]:
        """Generate framework-specific error code."""
        # Simplified - would need actual framework-specific patterns
        filename = self._get_main_filename(language)
        return {filename: f"# {framework} configuration error test for {language}\n"}

    def _generate_cross_language_tests(self, language: str) -> List[Dict[str, Any]]:
        """Generate cross-language interaction test cases."""
        tests = []

        # Define common cross-language scenarios
        cross_lang_scenarios = {
            "python": ["python-c", "python-javascript"],
            "javascript": ["javascript-wasm", "javascript-python"],
            "java": ["java-jni", "java-kotlin"],
            "go": ["go-c", "go-wasm"],
            "rust": ["rust-wasm", "rust-c"],
        }

        if language in cross_lang_scenarios:
            for scenario in cross_lang_scenarios[language]:
                test_case = {
                    "name": f"cross_lang_{scenario}",
                    "language": language,
                    "description": f"Test {scenario} interaction",
                    "test_type": "cross_language",
                    "source_code": {
                        self._get_main_filename(
                            language
                        ): f"# Cross-language test: {scenario}"
                    },
                    "expected_errors": [
                        {
                            "error_type": "InteropError",
                            "message": "Cross-language communication error",
                        }
                    ],
                    "expected_fixes": [
                        {
                            "fix_type": "interop_fix",
                            "description": "Fix cross-language communication",
                        }
                    ],
                    "environment": {},
                    "dependencies": [],
                    "frameworks": [],
                    "tags": ["cross_language", scenario],
                }
                tests.append(test_case)

        return tests

    def _save_test_suite(self, filepath: Path, tests: List[Dict[str, Any]]):
        """Save test suite to JSON file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(tests, f, indent=2)
        logger.info(f"Saved {len(tests)} tests to {filepath}")


def generate_all_test_suites():
    """Generate test suites for all supported languages."""
    generator = SuiteGenerator()
    generator.generate_all_test_suites()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_all_test_suites()
