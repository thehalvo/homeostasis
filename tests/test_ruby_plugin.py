import unittest

from modules.analysis.language_adapters import RubyErrorAdapter
from modules.analysis.plugins.ruby_plugin import (
    RubyExceptionHandler,
    RubyLanguagePlugin,
    RubyPatchGenerator,
)


class TestRubyPlugin(unittest.TestCase):
    def setUp(self):
        self.plugin = RubyLanguagePlugin()
        self.exception_handler = RubyExceptionHandler()
        self.adapter = RubyErrorAdapter()
        self.patch_generator = RubyPatchGenerator()

    def test_plugin_basic_info(self):
        """Test basic plugin information is correct"""
        self.assertEqual(self.plugin.get_language_id(), "ruby")
        self.assertEqual(self.plugin.get_language_name(), "Ruby")
        self.assertEqual(self.plugin.get_language_version(), "2.5+")

        frameworks = self.plugin.get_supported_frameworks()
        self.assertIn("rails", frameworks)
        self.assertIn("sinatra", frameworks)
        self.assertIn("rack", frameworks)
        self.assertIn("base", frameworks)

    def test_adapter_standardize_formats(self):
        """Test conversion to and from standard error format"""
        # Sample Ruby error
        ruby_error = {
            "exception_class": "NoMethodError",
            "message": "undefined method `name' for nil:NilClass",
            "backtrace": [
                "app/models/user.rb:25:in `display_name'",
                "app/controllers/users_controller.rb:10:in `show'",
                "actionpack-6.1.0/lib/action_controller/metal/basic_implicit_render.rb:6:in `send_action'",
            ],
            "ruby_version": "3.0.0",
            "framework": "Rails",
            "framework_version": "6.1.0",
        }

        # Convert to standard format
        standard_error = self.adapter.to_standard_format(ruby_error)

        # Verify standard format
        self.assertEqual(standard_error["language"], "ruby")
        self.assertEqual(standard_error["error_type"], "NoMethodError")
        self.assertEqual(
            standard_error["message"], "undefined method `name' for nil:NilClass"
        )
        self.assertEqual(standard_error["language_version"], "3.0.0")
        self.assertEqual(standard_error["framework"], "Rails")
        self.assertEqual(standard_error["framework_version"], "6.1.0")
        self.assertIsInstance(standard_error["stack_trace"], list)

        # Convert back to Ruby format
        ruby_error_roundtrip = self.adapter.from_standard_format(standard_error)

        # Verify roundtrip conversion
        self.assertEqual(ruby_error_roundtrip["exception_class"], "NoMethodError")
        self.assertEqual(
            ruby_error_roundtrip["message"], "undefined method `name' for nil:NilClass"
        )
        self.assertIn("backtrace", ruby_error_roundtrip)
        self.assertEqual(ruby_error_roundtrip["ruby_version"], "3.0.0")
        self.assertEqual(ruby_error_roundtrip["framework"], "Rails")
        self.assertEqual(ruby_error_roundtrip["framework_version"], "6.1.0")

    def test_exception_handler_nil_reference(self):
        """Test detection of nil reference errors"""
        error_data = {
            "error_type": "NoMethodError",
            "message": "undefined method `name' for nil:NilClass",
            "stack_trace": [
                "app/models/user.rb:25:in `display_name'",
                "app/controllers/users_controller.rb:10:in `show'",
            ],
        }

        analysis = self.exception_handler.analyze_error(error_data)

        self.assertEqual(analysis["rule_id"], "ruby_nil_reference")
        self.assertEqual(analysis["root_cause"], "ruby_nil_reference")
        self.assertEqual(analysis["confidence"], "high")
        self.assertEqual(analysis["category"], "core")

    def test_exception_handler_rails_record_not_found(self):
        """Test detection of ActiveRecord::RecordNotFound errors"""
        error_data = {
            "error_type": "ActiveRecord::RecordNotFound",
            "message": "Couldn't find User with ID 123",
            "stack_trace": [
                "app/controllers/users_controller.rb:10:in `show'",
                "actionpack-6.1.0/lib/action_controller/metal/basic_implicit_render.rb:6:in `send_action'",
            ],
            "framework": "rails",
        }

        analysis = self.exception_handler.analyze_error(error_data)

        self.assertEqual(analysis["rule_id"], "rails_record_not_found")
        self.assertEqual(analysis["root_cause"], "rails_record_not_found")
        self.assertEqual(analysis["confidence"], "high")
        self.assertEqual(analysis["category"], "activerecord")
        self.assertEqual(analysis["framework"], "rails")

    def test_exception_handler_sinatra_not_found(self):
        """Test detection of Sinatra::NotFound errors"""
        error_data = {
            "error_type": "Sinatra::NotFound",
            "message": "Sinatra::NotFound",
            "stack_trace": [
                "sinatra-2.2.0/lib/sinatra/base.rb:1185:in `block in route!'",
                "app.rb:25:in `block in <class:App>'",
            ],
            "framework": "sinatra",
        }

        analysis = self.exception_handler.analyze_error(error_data)

        self.assertEqual(analysis["rule_id"], "sinatra_not_found")
        self.assertEqual(analysis["root_cause"], "sinatra_not_found")
        self.assertEqual(analysis["confidence"], "high")
        self.assertEqual(analysis["category"], "sinatra")
        self.assertEqual(analysis["framework"], "sinatra")

    def test_exception_handler_missing_gem(self):
        """Test detection of LoadError for missing gems"""
        error_data = {
            "error_type": "LoadError",
            "message": "cannot load such file -- httparty",
            "stack_trace": ["/app.rb:5:in `require'", "/app.rb:5:in `<main>'"],
        }

        analysis = self.exception_handler.analyze_error(error_data)

        self.assertEqual(analysis["rule_id"], "ruby_gem_load_error")
        self.assertEqual(analysis["root_cause"], "ruby_missing_gem")
        self.assertEqual(analysis["confidence"], "high")
        self.assertEqual(analysis["category"], "gems")
        self.assertEqual(analysis["match_groups"][0], "httparty")

    def test_exception_handler_metaprogramming_error(self):
        """Test detection of metaprogramming errors"""
        error_data = {
            "error_type": "NoMethodError",
            "message": "undefined method `process_call' for #<User:0x00007f8b1a8b8a90> (method_missing)",
            "stack_trace": [
                "app/models/user.rb:25:in `method_missing'",
                "app/controllers/users_controller.rb:10:in `show'",
            ],
        }

        analysis = self.exception_handler.analyze_error(error_data)

        self.assertEqual(analysis["rule_id"], "ruby_method_missing_error")
        self.assertEqual(analysis["root_cause"], "ruby_method_missing")
        self.assertEqual(analysis["confidence"], "medium")
        self.assertEqual(analysis["category"], "metaprogramming")

    def test_patch_generation_nil_reference(self):
        """Test patch generation for nil reference"""
        analysis = {
            "rule_id": "ruby_nil_reference",
            "root_cause": "ruby_nil_reference",
            "confidence": "high",
            "severity": "medium",
            "error_data": {
                "error_type": "NoMethodError",
                "message": "undefined method `name' for nil:NilClass",
                "stack_trace": ["app/models/user.rb:25:in `display_name'"],
            },
            "match_groups": ("name", "nil:NilClass"),
        }

        context = {"code_snippet": "def display_name\n  user.name\nend"}

        patch = self.patch_generator.generate_patch(analysis, context)

        self.assertEqual(patch["patch_type"], "code")
        self.assertEqual(patch["language"], "ruby")
        self.assertEqual(patch["root_cause"], "ruby_nil_reference")
        self.assertIn("patch_code", patch)
        self.assertIn("if", patch["patch_code"])
        self.assertIn("nil", patch["patch_code"])

    def test_patch_generation_rails_record_not_found(self):
        """Test patch generation for ActiveRecord::RecordNotFound"""
        analysis = {
            "rule_id": "rails_record_not_found",
            "root_cause": "rails_record_not_found",
            "confidence": "high",
            "severity": "medium",
            "error_data": {
                "error_type": "ActiveRecord::RecordNotFound",
                "message": "Couldn't find User with ID 123",
                "stack_trace": ["app/controllers/users_controller.rb:10:in `show'"],
            },
            "match_groups": ("User",),
        }

        context = {
            "code_snippet": "def show\n  @user = User.find(params[:id])\nend",
            "framework": "rails",
        }

        patch = self.patch_generator.generate_patch(analysis, context)

        self.assertEqual(patch["patch_type"], "code")
        self.assertEqual(patch["language"], "ruby")
        self.assertEqual(patch["framework"], "rails")
        self.assertEqual(patch["root_cause"], "rails_record_not_found")
        self.assertIn("patch_code", patch)
        self.assertIn("find_by", patch["patch_code"])
        self.assertIn("redirect_to", patch["patch_code"])

    def test_patch_generation_missing_gem(self):
        """Test patch generation for missing gem"""
        analysis = {
            "rule_id": "ruby_gem_load_error",
            "root_cause": "ruby_missing_gem",
            "confidence": "high",
            "severity": "high",
            "error_data": {
                "error_type": "LoadError",
                "message": "cannot load such file -- httparty",
                "stack_trace": ["/app.rb:5:in `require'"],
            },
            "match_groups": ("httparty",),
        }

        context = {
            "code_snippet": "require 'httparty'\n\nresponse = HTTParty.get('https://example.com')"
        }

        patch = self.patch_generator.generate_patch(analysis, context)

        self.assertEqual(patch["patch_type"], "code")
        self.assertEqual(patch["language"], "ruby")
        self.assertEqual(patch["root_cause"], "ruby_missing_gem")
        self.assertIn("patch_code", patch)
        self.assertIn("gem 'httparty'", patch["patch_code"])
        self.assertIn("bundle install", patch["patch_code"])

    def test_plugin_integration(self):
        """Test full plugin integration flow"""
        # Original Ruby error
        ruby_error = {
            "exception_class": "NoMethodError",
            "message": "undefined method `name' for nil:NilClass",
            "backtrace": [
                "app/models/user.rb:25:in `display_name'",
                "app/controllers/users_controller.rb:10:in `show'",
            ],
            "ruby_version": "3.0.0",
            "framework": "Rails",
            "framework_version": "6.1.0",
        }

        # Expected context
        context = {
            "code_snippet": "def display_name\n  user.name\nend",
            "framework": "rails",
        }

        # Test the full flow
        standard_error = self.plugin.normalize_error(ruby_error)
        analysis = self.plugin.analyze_error(standard_error)
        fix = self.plugin.generate_fix(analysis, context)

        # Verify results
        self.assertEqual(analysis["rule_id"], "ruby_nil_reference")
        self.assertEqual(analysis["root_cause"], "ruby_nil_reference")
        self.assertEqual(fix["language"], "ruby")
        self.assertIn("suggestion", fix)


if __name__ == "__main__":
    unittest.main()
