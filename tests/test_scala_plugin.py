"""
Tests for the Scala language plugin.

This file contains tests for verifying the functionality of the Scala language plugin,
including error normalization, analysis, and patch generation.
"""

import sys
import unittest
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.analysis.language_adapters import ScalaErrorAdapter
from modules.analysis.plugins.scala_plugin import ScalaLanguagePlugin


class TestScalaPlugin(unittest.TestCase):
    """Test cases for the Scala language plugin."""

    def setUp(self):
        """Set up test fixtures."""
        self.plugin = ScalaLanguagePlugin()
        self.adapter = ScalaErrorAdapter()

        # Sample Scala error data
        self.sample_error = {
            "error_type": "java.lang.NullPointerException",
            "message": 'Cannot invoke "String.length()" because "str" is null',
            "stack_trace": [
                "at com.example.MyClass.processString(MyClass.scala:42)",
                "at com.example.MyApp.main(MyApp.scala:15)",
            ],
            "scala_version": "2.13.6",
        }

        # Sample Akka error
        self.akka_error = {
            "error_type": "akka.actor.ActorNotFound",
            "message": "Actor not found for: ActorSelection[Anchor(akka://system/user/someActor#1990616746)]",
            "stack_trace": [
                "at akka.actor.ActorSelection.$bang(ActorSelection.scala:67)",
                "at com.example.AkkaExample.sendMessage(AkkaExample.scala:25)",
                "at com.example.AkkaExample.run(AkkaExample.scala:15)",
            ],
            "framework": "akka",
            "framework_version": "2.6.14",
            "scala_version": "2.13.6",
        }

        # Sample Play Framework error
        self.play_error = {
            "error_type": "play.api.libs.json.JsResultException",
            "message": "JsResultException(errors:List((,List(ValidationError(List(error.path.missing),ArraySeq())))))",
            "stack_trace": [
                "at play.api.libs.json.JsValue.$div(JsValue.scala:90)",
                "at controllers.MyController.processJson(MyController.scala:45)",
                "at play.core.routing.HandlerInvoker.invoke(HandlerInvoker.scala:126)",
            ],
            "framework": "play",
            "framework_version": "2.8.8",
            "scala_version": "2.13.6",
        }

        # Sample Scala match error
        self.match_error = {
            "error_type": "scala.MatchError",
            "message": "Some(Foo) (of class scala.Some)",
            "stack_trace": [
                "at com.example.PatternMatch.process(PatternMatch.scala:16)",
                "at com.example.MyApp.run(MyApp.scala:25)",
            ],
            "scala_version": "2.13.6",
        }

    def test_plugin_metadata(self):
        """Test plugin metadata is correctly set."""
        metadata = self.plugin.get_metadata()

        self.assertEqual(metadata["language_id"], "scala")
        self.assertEqual(metadata["language_name"], "Scala")
        self.assertEqual(metadata["language_version"], "2.12+")

        # Check supported frameworks
        supported_frameworks = self.plugin.get_supported_frameworks()
        expected_frameworks = ["akka", "play", "sbt", "cats", "zio", "base"]

        for framework in expected_frameworks:
            self.assertIn(framework, supported_frameworks)

    def test_error_normalization(self):
        """Test normalizing Scala errors to standard format."""
        standard_error = self.plugin.normalize_error(self.sample_error)

        # Check basic fields
        self.assertEqual(standard_error["language"], "scala")
        self.assertEqual(standard_error["error_type"], "java.lang.NullPointerException")
        self.assertEqual(
            standard_error["message"],
            'Cannot invoke "String.length()" because "str" is null',
        )
        self.assertEqual(standard_error["language_version"], "2.13.6")

        # Check stack trace is preserved
        self.assertEqual(len(standard_error["stack_trace"]), 2)

        # Verify it can be reversed
        denormalized = self.plugin.denormalize_error(standard_error)
        self.assertEqual(denormalized["error_type"], self.sample_error["error_type"])
        self.assertEqual(denormalized["message"], self.sample_error["message"])

    def test_null_pointer_analysis(self):
        """Test analysis of NullPointerException."""
        standard_error = self.plugin.normalize_error(self.sample_error)
        analysis = self.plugin.analyze_error(standard_error)

        # Check analysis results
        self.assertEqual(analysis["error_type"], "java.lang.NullPointerException")
        self.assertEqual(analysis["root_cause"], "scala_null_pointer")
        self.assertIn("Option", analysis["suggestion"])  # Should suggest using Option
        self.assertEqual(analysis["confidence"], "high")
        self.assertEqual(analysis["severity"], "high")

    def test_match_error_analysis(self):
        """Test analysis of MatchError."""
        standard_error = self.plugin.normalize_error(self.match_error)
        analysis = self.plugin.analyze_error(standard_error)

        # Check analysis results
        self.assertEqual(analysis["error_type"], "scala.MatchError")
        self.assertEqual(analysis["root_cause"], "scala_incomplete_match")
        self.assertIn(
            "exhaustive", analysis["suggestion"]
        )  # Should suggest exhaustive matches
        self.assertEqual(analysis["confidence"], "high")

    def test_akka_error_analysis(self):
        """Test analysis of Akka error."""
        standard_error = self.plugin.normalize_error(self.akka_error)
        analysis = self.plugin.analyze_error(standard_error)

        # Check analysis results
        self.assertEqual(analysis["error_type"], "akka.actor.ActorNotFound")
        self.assertEqual(analysis["root_cause"], "akka_actor_not_found")
        self.assertEqual(analysis["framework"], "akka")
        self.assertIn("actorSelection", analysis["suggestion"])

    def test_play_error_analysis(self):
        """Test analysis of Play Framework error."""
        standard_error = self.plugin.normalize_error(self.play_error)
        analysis = self.plugin.analyze_error(standard_error)

        # Check analysis results
        self.assertEqual(analysis["error_type"], "play.api.libs.json.JsResultException")
        self.assertEqual(analysis["framework"], "play")
        self.assertIn("json", analysis["suggestion"].lower())

    def test_patch_generation_null_pointer(self):
        """Test patch generation for null pointer."""
        standard_error = self.plugin.normalize_error(self.sample_error)
        analysis = self.plugin.analyze_error(standard_error)

        context = {
            "code_snippet": "val length = str.length()",
            "file": "MyClass.scala",
            "line": 42,
        }

        patch = self.plugin.generate_fix(analysis, context)

        # Check patch is generated
        self.assertEqual(patch["language"], "scala")
        self.assertEqual(patch["root_cause"], "scala_null_pointer")

        # Check the fix contains Option
        if "suggestion_code" in patch:
            self.assertIn("Option(", patch["suggestion_code"])
        if "patch_code" in patch:
            self.assertIn("Option(", patch["patch_code"])

    def test_patch_generation_match_error(self):
        """Test patch generation for match error."""
        standard_error = self.plugin.normalize_error(self.match_error)
        analysis = self.plugin.analyze_error(standard_error)

        context = {
            "code_snippet": "value match {\n  case Some(Type1) => handleType1()\n  case Some(Type2) => handleType2()\n}",
            "file": "PatternMatch.scala",
            "line": 16,
        }

        patch = self.plugin.generate_fix(analysis, context)

        # Check patch is generated
        self.assertEqual(patch["language"], "scala")
        self.assertEqual(patch["root_cause"], "scala_incomplete_match")

        # Check the fix contains wildcard pattern
        if "suggestion_code" in patch:
            self.assertIn("case _", patch["suggestion_code"])
        if "patch_code" in patch:
            self.assertIn("case _", patch["patch_code"])

    def test_stack_trace_parsing(self):
        """Test parsing Scala stack traces into structured frames."""
        stack_trace = [
            "at com.example.MyClass.processString(MyClass.scala:42)",
            "at com.example.MyApp.main(MyApp.scala:15)",
        ]

        parsed_frames = self.adapter._parse_scala_stack_trace(stack_trace)

        self.assertEqual(len(parsed_frames), 2)

        # Check first frame structure
        first_frame = parsed_frames[0]
        self.assertEqual(first_frame["package"], "com.example")
        self.assertEqual(first_frame["class"], "MyClass")
        self.assertEqual(first_frame["function"], "processString")
        self.assertEqual(first_frame["file"], "MyClass.scala")
        self.assertEqual(first_frame["line"], 42)


if __name__ == "__main__":
    unittest.main()
