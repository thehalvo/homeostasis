#!/usr/bin/env python3
"""
Test for LLM Language Detection and Context Generation for Phase 12.A.2 languages.

This test verifies that the multi-language framework detector and LLM integration
properly support all the new languages added in Phase 12.A.2.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.patch_generation.multi_language_framework_detector import (
    LanguageType,
    MultiLanguageFrameworkDetector,
)


class TestLLMLanguageDetection(unittest.TestCase):
    """Test LLM language detection and context generation for new languages."""

    def setUp(self):
        """Set up test fixtures."""
        self.detector = MultiLanguageFrameworkDetector()

    def test_zig_detection_and_context(self):
        """Test Zig language detection and LLM context generation."""
        # Sample Zig code
        zig_code = """
const std = @import("std");

pub fn main() !void {
    const stdout = std.io.getStdOut().writer();
    try stdout.print("Hello, {s}!\\n", .{"World"});
}

fn add(a: i32, b: i32) i32 {
    return a + b;
}
"""

        # Detect language
        result = self.detector.detect_language_and_frameworks(
            file_path="main.zig", source_code=zig_code
        )

        # Verify detection
        self.assertEqual(result.language, LanguageType.ZIG)
        self.assertGreater(result.confidence, 0.8)

        # Get LLM context
        llm_context = self.detector.get_llm_context_for_language(result)

        # Verify LLM context
        self.assertEqual(llm_context["language"], "zig")
        self.assertIn("llm_guidance", llm_context)
        self.assertIn("style_guide", llm_context["llm_guidance"])
        self.assertIn("error_handling", llm_context["llm_guidance"])
        self.assertIn("imports", llm_context["llm_guidance"])

    def test_nim_detection_and_context(self):
        """Test Nim language detection and LLM context generation."""
        nim_code = """
import strutils, sequtils

proc greet(name: string): string =
  result = "Hello, " & name & "!"

when isMainModule:
  let names = @["Alice", "Bob", "Charlie"]
  for name in names:
    echo greet(name)
"""

        result = self.detector.detect_language_and_frameworks(
            file_path="hello.nim", source_code=nim_code
        )

        self.assertEqual(result.language, LanguageType.NIM)
        self.assertIn("*.nim", result.file_patterns)

        llm_context = self.detector.get_llm_context_for_language(result)
        self.assertEqual(llm_context["language"], "nim")
        self.assertIn("camelCase", str(llm_context["llm_guidance"]["common_patterns"]))

    def test_sql_detection_and_context(self):
        """Test SQL language detection and LLM context generation."""
        sql_code = """
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

SELECT u.name, COUNT(o.id) as order_count
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
GROUP BY u.id, u.name
HAVING COUNT(o.id) > 5;
"""

        result = self.detector.detect_language_and_frameworks(
            file_path="schema.sql", source_code=sql_code
        )

        self.assertEqual(result.language, LanguageType.SQL)

        llm_context = self.detector.get_llm_context_for_language(result)
        self.assertEqual(llm_context["language"], "sql")
        self.assertIn("UPPERCASE", str(llm_context["llm_guidance"]["common_patterns"]))

    def test_terraform_detection_and_context(self):
        """Test Terraform language detection and LLM context generation."""
        terraform_code = """
resource "aws_instance" "example" {
  ami           = "ami-0c55b159cbfafe1f0"
  instance_type = "t2.micro"
  
  tags = {
    Name = "ExampleInstance"
    Environment = "Production"
  }
}

variable "region" {
  description = "AWS region"
  default     = "us-west-2"
}
"""

        result = self.detector.detect_language_and_frameworks(
            file_path="main.tf", source_code=terraform_code
        )

        self.assertEqual(result.language, LanguageType.TERRAFORM)

        llm_context = self.detector.get_llm_context_for_language(result)
        self.assertEqual(llm_context["language"], "terraform")
        self.assertIn("snake_case", str(llm_context["llm_guidance"]["common_patterns"]))

    def test_dockerfile_detection_and_context(self):
        """Test Dockerfile language detection and LLM context generation."""
        dockerfile_code = """
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "app.py"]
"""

        result = self.detector.detect_language_and_frameworks(
            file_path="Dockerfile", source_code=dockerfile_code
        )

        self.assertEqual(result.language, LanguageType.DOCKERFILE)

        llm_context = self.detector.get_llm_context_for_language(result)
        self.assertEqual(llm_context["language"], "dockerfile")
        self.assertIn("UPPERCASE", str(llm_context["llm_guidance"]["common_patterns"]))

    def test_all_languages_have_guidance(self):
        """Test that all Phase 12.A.2 languages have LLM guidance configured."""
        phase_12a_languages = [
            LanguageType.ZIG,
            LanguageType.NIM,
            LanguageType.CRYSTAL,
            LanguageType.HASKELL,
            LanguageType.FSHARP,
            LanguageType.ERLANG,
            LanguageType.SQL,
            LanguageType.BASH,
            LanguageType.POWERSHELL,
            LanguageType.LUA,
            LanguageType.R,
            LanguageType.MATLAB,
            LanguageType.JULIA,
            LanguageType.TERRAFORM,
            LanguageType.ANSIBLE,
            LanguageType.YAML,
            LanguageType.JSON,
            LanguageType.DOCKERFILE,
        ]

        for lang_type in phase_12a_languages:
            # Create a dummy language info
            from modules.patch_generation.multi_language_framework_detector import (
                LanguageInfo,
            )

            lang_info = LanguageInfo(
                language=lang_type,
                confidence=1.0,
                frameworks=[],
                file_patterns=[],
                language_features={},
            )

            # Get LLM context
            llm_context = self.detector.get_llm_context_for_language(lang_info)

            # Verify guidance exists
            self.assertIn("llm_guidance", llm_context)
            self.assertIn("style_guide", llm_context["llm_guidance"])
            self.assertIn("common_patterns", llm_context["llm_guidance"])
            self.assertIn("error_handling", llm_context["llm_guidance"])
            self.assertIn("imports", llm_context["llm_guidance"])

            # Verify it's not using default guidance
            self.assertNotEqual(
                llm_context["llm_guidance"]["style_guide"],
                "Follow language conventions",
                f"Language {lang_type.value} is using default guidance",
            )


if __name__ == "__main__":
    unittest.main()
