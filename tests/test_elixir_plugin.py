"""
Tests for the Elixir/Erlang plugin functionality.
"""
import os
import sys
import json
import unittest
from pathlib import Path

# Add the parent directory to the path so we can import from modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.analysis.plugins.elixir_plugin import ElixirLanguagePlugin
from modules.analysis.language_adapters import ElixirErrorAdapter


class TestElixirPlugin(unittest.TestCase):
    """Test cases for Elixir/Erlang plugin."""

    def setUp(self):
        """Set up test fixtures."""
        self.plugin = ElixirLanguagePlugin()
        self.adapter = ElixirErrorAdapter()
        
        # Load sample error data
        self.key_error_data = {
            "exception": "KeyError",
            "message": "key :missing_key not found in: %{foo: \"bar\"}",
            "stacktrace": [
                "    (stdlib) :maps.find(:missing_key, %{foo: \"bar\"})",
                "    (elixir) lib/kernel.ex:1820: Kernel.map_get/2",
                "    (my_app) lib/my_app/service.ex:42: MyApp.Service.process_data/1",
                "    (my_app) lib/my_app/controller.ex:23: MyApp.Controller.action/2"
            ]
        }
        
        self.function_clause_error_data = {
            "exception": "FunctionClauseError",
            "message": "no function clause matching in MyApp.Service.process_data/1",
            "stacktrace": [
                "    (my_app) lib/my_app/service.ex:42: MyApp.Service.process_data/1",
                "    (my_app) lib/my_app/controller.ex:23: MyApp.Controller.action/2"
            ]
        }
        
        self.phoenix_error_data = {
            "exception": "Phoenix.Router.NoRouteError",
            "message": "no route found for GET /missing/path (MyApp.Router)",
            "stacktrace": [
                "    (phoenix) lib/phoenix/router.ex:352: Phoenix.Router.call/2",
                "    (my_app) lib/my_app_web/endpoint.ex:1: MyApp.Endpoint.plug_builder_call/2",
                "    (my_app) lib/plug/debugger.ex:136: MyApp.Endpoint.\"call (overridable 3)\"/2",
                "    (my_app) lib/phoenix/endpoint/cowboy2_handler.ex:42: MyApp.Endpoint.call/2"
            ],
            "framework": "phoenix"
        }
        
        self.ecto_error_data = {
            "exception": "Ecto.ChangesetError",
            "message": "could not perform insert because changeset is invalid",
            "stacktrace": [
                "    (ecto) lib/ecto/repo/schema.ex:212: Ecto.Repo.Schema.insert!/4",
                "    (my_app) lib/my_app/accounts.ex:65: MyApp.Accounts.create_user!/1",
                "    (my_app) lib/my_app_web/controllers/user_controller.ex:25: MyApp.UserController.create/2"
            ],
            "framework": "ecto"
        }
        
        self.otp_error_data = {
            "exception": "GenServer.CallError",
            "message": "call #{inspect(ServerPid)} timed out after 5000ms",
            "stacktrace": [
                "    (elixir) lib/gen_server.ex:1043: GenServer.call/3",
                "    (my_app) lib/my_app/server.ex:42: MyApp.Server.get_data/1",
                "    (my_app) lib/my_app/controller.ex:32: MyApp.Controller.action/2"
            ]
        }

    def test_plugin_basic_info(self):
        """Test basic plugin information."""
        self.assertEqual(self.plugin.get_language_id(), "elixir")
        self.assertEqual(self.plugin.get_language_name(), "Elixir")
        self.assertTrue(self.plugin.get_language_version().startswith("1."))
        
        supported_frameworks = self.plugin.get_supported_frameworks()
        self.assertIn("phoenix", supported_frameworks)
        self.assertIn("ecto", supported_frameworks)
        self.assertIn("otp", supported_frameworks)
        self.assertIn("base", supported_frameworks)

    def test_normalize_error(self):
        """Test error normalization."""
        standard_error = self.plugin.normalize_error(self.key_error_data)
        
        self.assertEqual(standard_error["language"], "elixir")
        self.assertEqual(standard_error["error_type"], "KeyError")
        self.assertEqual(standard_error["message"], "key :missing_key not found in: %{foo: \"bar\"}")
        self.assertTrue(isinstance(standard_error["stack_trace"], list))
        
        # Test that denormalization works
        denormalized = self.plugin.denormalize_error(standard_error)
        self.assertEqual(denormalized["exception"], "KeyError")
        self.assertEqual(denormalized["message"], "key :missing_key not found in: %{foo: \"bar\"}")

    def test_analyze_key_error(self):
        """Test analyzing a KeyError."""
        analysis = self.plugin.analyze_error(self.key_error_data)
        
        self.assertEqual(analysis["error_type"], "KeyError")
        self.assertEqual(analysis["root_cause"], "elixir_key_error")
        self.assertTrue(isinstance(analysis["description"], str))
        self.assertTrue(isinstance(analysis["suggestion"], str))
        self.assertIn("Map.get", analysis["suggestion"])

    def test_analyze_function_clause_error(self):
        """Test analyzing a FunctionClauseError."""
        analysis = self.plugin.analyze_error(self.function_clause_error_data)
        
        self.assertEqual(analysis["error_type"], "FunctionClauseError")
        self.assertEqual(analysis["root_cause"], "elixir_function_clause_error")
        self.assertTrue(isinstance(analysis["description"], str))
        self.assertTrue(isinstance(analysis["suggestion"], str))
        self.assertIn("pattern matching", analysis["suggestion"].lower())

    def test_analyze_phoenix_error(self):
        """Test analyzing a Phoenix error."""
        analysis = self.plugin.analyze_error(self.phoenix_error_data)
        
        self.assertEqual(analysis["error_type"], "Phoenix.Router.NoRouteError")
        self.assertEqual(analysis["category"], "phoenix")
        self.assertTrue(isinstance(analysis["description"], str))
        self.assertTrue(isinstance(analysis["suggestion"], str))
        self.assertIn("router", analysis["suggestion"].lower())

    def test_analyze_ecto_error(self):
        """Test analyzing an Ecto error."""
        analysis = self.plugin.analyze_error(self.ecto_error_data)
        
        self.assertEqual(analysis["error_type"], "Ecto.ChangesetError")
        self.assertEqual(analysis["category"], "ecto")
        self.assertTrue(isinstance(analysis["description"], str))
        self.assertTrue(isinstance(analysis["suggestion"], str))
        self.assertIn("changeset", analysis["suggestion"].lower())

    def test_generate_fix(self):
        """Test generating a fix for an error."""
        analysis = self.plugin.analyze_error(self.key_error_data)
        context = {"code_snippet": "map = %{foo: \"bar\"}\nvalue = map.missing_key"}
        
        fix = self.plugin.generate_fix(analysis, context)
        
        self.assertEqual(fix["language"], "elixir")
        self.assertEqual(fix["root_cause"], "elixir_key_error")
        self.assertTrue(isinstance(fix["suggestion"], str))
        
        # Either a code patch or suggestion code should be present
        self.assertTrue("patch_code" in fix or "suggestion_code" in fix)

    def test_integration(self):
        """Test the full integration flow of the plugin."""
        # Create an error, normalize, analyze, and generate fix
        error_data = self.key_error_data
        standard_error = self.plugin.normalize_error(error_data)
        analysis = self.plugin.analyze_error(standard_error)
        context = {"code_snippet": "map = %{foo: \"bar\"}\nvalue = map.missing_key"}
        fix = self.plugin.generate_fix(analysis, context)
        
        # Check that all steps work together
        self.assertEqual(analysis["error_type"], "KeyError")
        self.assertEqual(analysis["root_cause"], "elixir_key_error")
        self.assertEqual(fix["language"], "elixir")
        self.assertEqual(fix["root_cause"], "elixir_key_error")
        self.assertTrue("Map.get" in fix["suggestion"] or "Map.get" in fix.get("suggestion_code", ""))


if __name__ == '__main__':
    unittest.main()