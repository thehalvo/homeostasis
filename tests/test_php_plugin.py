import unittest

from modules.analysis.language_adapters import PHPErrorAdapter
from modules.analysis.plugins.php_plugin import (PHPExceptionHandler,
                                                 PHPLanguagePlugin,
                                                 PHPPatchGenerator)


class TestPHPPlugin(unittest.TestCase):
    def setUp(self):
        self.plugin = PHPLanguagePlugin()
        self.exception_handler = PHPExceptionHandler()
        self.adapter = PHPErrorAdapter()
        self.patch_generator = PHPPatchGenerator()

    def test_plugin_basic_info(self):
        """Test basic plugin information is correct"""
        self.assertEqual(self.plugin.get_language_id(), "php")
        self.assertEqual(self.plugin.get_language_name(), "PHP")
        self.assertEqual(self.plugin.get_language_version(), "7.0+")

        frameworks = self.plugin.get_supported_frameworks()
        self.assertIn("laravel", frameworks)
        self.assertIn("symfony", frameworks)
        self.assertIn("wordpress", frameworks)
        self.assertIn("codeigniter", frameworks)
        self.assertIn("base", frameworks)

    def test_adapter_standardize_formats(self):
        """Test conversion to and from standard error format"""
        # Sample PHP error
        php_error = {
            "type": "ErrorException",
            "message": "Undefined variable: user",
            "file": "/var/www/html/app/Controllers/UserController.php",
            "line": 25,
            "trace": [
                {
                    "file": "/var/www/html/app/Controllers/UserController.php",
                    "line": 25,
                    "function": "getUserProfile",
                    "class": "App\\Controllers\\UserController",
                },
                {
                    "file": "/var/www/html/routes/web.php",
                    "line": 16,
                    "function": "handle",
                    "class": "App\\Http\\Kernel",
                },
            ],
            "php_version": "8.1.0",
            "framework": "Laravel",
            "framework_version": "9.0.0",
        }

        # Convert to standard format
        standard_error = self.adapter.to_standard_format(php_error)

        # Verify standard format
        self.assertEqual(standard_error["language"], "php")
        self.assertEqual(standard_error["error_type"], "ErrorException")
        self.assertEqual(standard_error["message"], "Undefined variable: user")
        self.assertEqual(standard_error["language_version"], "8.1.0")
        self.assertEqual(standard_error["framework"], "Laravel")
        self.assertEqual(standard_error["framework_version"], "9.0.0")
        self.assertIsInstance(standard_error["stack_trace"], list)

        # Convert back to PHP format
        php_error_roundtrip = self.adapter.from_standard_format(standard_error)

        # Verify roundtrip conversion
        self.assertEqual(php_error_roundtrip["type"], "ErrorException")
        self.assertEqual(php_error_roundtrip["message"], "Undefined variable: user")
        self.assertIn("trace", php_error_roundtrip)
        self.assertEqual(php_error_roundtrip["php_version"], "8.1.0")
        self.assertEqual(php_error_roundtrip["framework"], "Laravel")
        self.assertEqual(php_error_roundtrip["framework_version"], "9.0.0")

    def test_exception_handler_undefined_variable(self):
        """Test detection of undefined variable errors"""
        error_data = {
            "error_type": "ErrorException",
            "message": "Undefined variable: user",
            "stack_trace": [
                {
                    "file": "/var/www/html/app/Controllers/UserController.php",
                    "line": 25,
                    "function": "getUserProfile",
                    "class": "App\\Controllers\\UserController",
                }
            ],
        }

        analysis = self.exception_handler.analyze_error(error_data)

        self.assertEqual(analysis["rule_id"], "php_undefined_variable")
        self.assertEqual(analysis["root_cause"], "php_undefined_variable")
        self.assertEqual(analysis["confidence"], "high")
        self.assertEqual(analysis["category"], "core")

    def test_exception_handler_null_reference(self):
        """Test detection of null reference errors"""
        error_data = {
            "error_type": "Error",
            "message": "Call to a member function getProfile() on null",
            "stack_trace": [
                {
                    "file": "/var/www/html/app/Controllers/UserController.php",
                    "line": 30,
                    "function": "getUserProfile",
                    "class": "App\\Controllers\\UserController",
                }
            ],
        }

        analysis = self.exception_handler.analyze_error(error_data)

        self.assertEqual(analysis["rule_id"], "php_null_reference")
        self.assertEqual(analysis["root_cause"], "php_null_reference")
        self.assertEqual(analysis["confidence"], "high")
        self.assertEqual(analysis["category"], "core")
        self.assertEqual(analysis["match_groups"][0], "getProfile")

    def test_exception_handler_laravel_model_not_found(self):
        """Test detection of Laravel ModelNotFoundException errors"""
        error_data = {
            "error_type": "Illuminate\\Database\\Eloquent\\ModelNotFoundException",
            "message": "No query results for model [App\\Models\\User] 123",
            "stack_trace": [
                {
                    "file": "/var/www/html/app/Controllers/UserController.php",
                    "line": 30,
                    "function": "findOrFail",
                    "class": "Illuminate\\Database\\Eloquent\\Builder",
                }
            ],
            "framework": "laravel",
        }

        analysis = self.exception_handler.analyze_error(error_data)

        self.assertEqual(analysis["rule_id"], "laravel_model_not_found")
        self.assertEqual(analysis["root_cause"], "laravel_model_not_found")
        self.assertEqual(analysis["confidence"], "high")
        self.assertEqual(analysis["category"], "database")
        self.assertEqual(analysis["framework"], "laravel")

    def test_exception_handler_symfony_container_exception(self):
        """Test detection of Symfony ServiceNotFoundException errors"""
        error_data = {
            "error_type": "Symfony\\Component\\DependencyInjection\\Exception\\ServiceNotFoundException",
            "message": 'You have requested a non-existent service "app.custom_service"',
            "stack_trace": [
                {
                    "file": "/var/www/html/src/Controller/DefaultController.php",
                    "line": 25,
                    "function": "get",
                    "class": "Symfony\\Component\\DependencyInjection\\Container",
                }
            ],
            "framework": "symfony",
        }

        analysis = self.exception_handler.analyze_error(error_data)

        self.assertEqual(analysis["rule_id"], "symfony_container_exception")
        self.assertEqual(analysis["root_cause"], "symfony_missing_service")
        self.assertEqual(analysis["confidence"], "high")
        self.assertEqual(analysis["category"], "dependency-injection")
        self.assertEqual(analysis["framework"], "symfony")
        self.assertEqual(analysis["match_groups"][0], "app.custom_service")

    def test_patch_generation_undefined_variable(self):
        """Test patch generation for undefined variable"""
        analysis = {
            "rule_id": "php_undefined_variable",
            "root_cause": "php_undefined_variable",
            "confidence": "high",
            "severity": "medium",
            "error_data": {
                "error_type": "ErrorException",
                "message": "Undefined variable: user",
                "stack_trace": [
                    {
                        "file": "/var/www/html/app/Controllers/UserController.php",
                        "line": 25,
                    }
                ],
            },
            "match_groups": ("user",),
        }

        context = {
            "code_snippet": "public function getUserProfile()\n{\n    return $user->profile;\n}"
        }

        patch = self.patch_generator.generate_patch(analysis, context)

        self.assertEqual(patch["patch_type"], "code")
        self.assertEqual(patch["language"], "php")
        self.assertEqual(patch["root_cause"], "php_undefined_variable")
        self.assertIn("patch_code", patch)
        self.assertIn("$user", patch["patch_code"])
        self.assertIn("isset", patch["patch_code"])

    def test_patch_generation_null_reference(self):
        """Test patch generation for null reference"""
        analysis = {
            "rule_id": "php_null_reference",
            "root_cause": "php_null_reference",
            "confidence": "high",
            "severity": "high",
            "error_data": {
                "error_type": "Error",
                "message": "Call to a member function getProfile() on null",
                "stack_trace": [
                    {
                        "file": "/var/www/html/app/Controllers/UserController.php",
                        "line": 30,
                    }
                ],
            },
            "match_groups": ("getProfile",),
        }

        context = {
            "code_snippet": "public function getUserProfile()\n{\n    return $user->getProfile();\n}"
        }

        patch = self.patch_generator.generate_patch(analysis, context)

        self.assertEqual(patch["patch_type"], "code")
        self.assertEqual(patch["language"], "php")
        self.assertEqual(patch["root_cause"], "php_null_reference")
        self.assertIn("patch_code", patch)
        self.assertIn("!== null", patch["patch_code"])
        self.assertIn("getProfile", patch["patch_code"])

    def test_patch_generation_laravel_model_not_found(self):
        """Test patch generation for Laravel ModelNotFoundException"""
        analysis = {
            "rule_id": "laravel_model_not_found",
            "root_cause": "laravel_model_not_found",
            "confidence": "high",
            "severity": "medium",
            "error_data": {
                "error_type": "Illuminate\\Database\\Eloquent\\ModelNotFoundException",
                "message": "No query results for model [App\\Models\\User] 123",
                "stack_trace": [
                    {
                        "file": "/var/www/html/app/Controllers/UserController.php",
                        "line": 30,
                    }
                ],
            },
            "match_groups": ("App\\Models\\User",),
        }

        context = {
            "code_snippet": "public function show($id)\n{\n    $user = User::findOrFail($id);\n    return view('users.show', compact('user'));\n}",
            "framework": "laravel",
        }

        patch = self.patch_generator.generate_patch(analysis, context)

        self.assertEqual(patch["patch_type"], "code")
        self.assertEqual(patch["language"], "php")
        self.assertEqual(patch["framework"], "laravel")
        self.assertEqual(patch["root_cause"], "laravel_model_not_found")
        self.assertIn("patch_code", patch)
        self.assertIn("try", patch["patch_code"])
        self.assertIn("findOrFail", patch["patch_code"])
        self.assertIn("catch", patch["patch_code"])
        self.assertIn("ModelNotFoundException", patch["patch_code"])

    def test_plugin_integration(self):
        """Test full plugin integration flow"""
        # Original PHP error
        php_error = {
            "type": "ErrorException",
            "message": "Undefined variable: user",
            "file": "/var/www/html/app/Controllers/UserController.php",
            "line": 25,
            "trace": [
                {
                    "file": "/var/www/html/app/Controllers/UserController.php",
                    "line": 25,
                    "function": "getUserProfile",
                    "class": "App\\Controllers\\UserController",
                }
            ],
            "php_version": "8.1.0",
            "framework": "Laravel",
            "framework_version": "9.0.0",
        }

        # Expected context
        context = {
            "code_snippet": "public function getUserProfile()\n{\n    return $user->profile;\n}",
            "framework": "laravel",
        }

        # Test the full flow
        standard_error = self.plugin.normalize_error(php_error)
        analysis = self.plugin.analyze_error(standard_error)
        fix = self.plugin.generate_fix(analysis, context)

        # Verify results
        self.assertEqual(analysis["rule_id"], "php_undefined_variable")
        self.assertEqual(analysis["root_cause"], "php_undefined_variable")
        self.assertEqual(fix["language"], "php")
        self.assertIn("patch_code", fix)


if __name__ == "__main__":
    unittest.main()
