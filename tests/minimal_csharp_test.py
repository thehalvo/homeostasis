"""
Minimal test for the C# plugin that doesn't require any external dependencies.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Basic classes needed from the plugin system


class LanguagePlugin:
    def get_language_id(self):
        pass

    def get_language_name(self):
        pass

    def get_language_version(self):
        pass

    def analyze_error(self, error_data):
        pass

    def normalize_error(self, error_data):
        pass

    def denormalize_error(self, standard_error):
        pass

    def generate_fix(self, analysis, context):
        pass

    def get_supported_frameworks(self):
        pass


# Mock registration function


def register_plugin(plugin):
    print(f"Registered plugin: {plugin.get_language_name()}")


# Import adapter
from modules.analysis.language_adapters import CSharpErrorAdapter


# Helper function to check if two dictionaries have the same keys and approximately the same values
def dict_similar(d1, d2, key):
    """Check if dictionaries have similar values for a specific key."""
    return key in d1 and key in d2 and d1[key] == d2[key]


# Mock minimal classes needed to test the plugin
# Create minimal rules directory structure needed for rule loading


def setup_test_rules():
    """Ensure rule directories exist."""
    rules_dir = Path(
        os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "modules",
                "analysis",
                "rules",
                "csharp",
            )
        )
    )
    if not rules_dir.exists():
        rules_dir.mkdir(parents=True, exist_ok=True)


# Create a minimal plugin implementation


class MinimalCSharpPlugin:
    """Minimal C# plugin for testing."""

    def __init__(self):
        """Initialize the test plugin."""
        self.adapter = CSharpErrorAdapter()

    def test_error_normalization(self):
        """Test error normalization from C# format to standard format."""
        null_reference_error = {
            "timestamp": "2023-08-15T12:34:56",
            "exception_type": "System.NullReferenceException",
            "message": "Object reference not set to an instance of an object",
            "stack_trace": [
                "   at MyCompany.MyApp.Services.UserService.GetUserProfile(Int32 userId) in D:\\Projects\\MyApp\\Services\\UserService.cs:line 42",
                "   at MyCompany.MyApp.Controllers.UserController.GetProfile(Int32 id) in D:\\Projects\\MyApp\\Controllers\\UserController.cs:line 28",
            ],
            "level": "error",
            "dotnet_version": "6.0.16",
            "framework": "ASP.NET Core",
            "framework_version": "6.0.16",
        }

        # Normalize to standard format
        standard_error = self.adapter.to_standard_format(null_reference_error)

        # Check that key fields were preserved
        success = True
        for key, expected in [
            ("language", "csharp"),
            ("error_type", "System.NullReferenceException"),
            ("message", "Object reference not set to an instance of an object"),
            ("language_version", "6.0.16"),
            ("framework", "ASP.NET Core"),
            ("framework_version", "6.0.16"),
        ]:
            if standard_error.get(key) != expected:
                success = False
                print(
                    f"❌ Field {key}: Expected '{expected}', got '{standard_error.get(key)}'"
                )
            else:
                print(f"✅ Field {key}: '{expected}'")

        # Check stack trace normalization
        if "stack_trace" in standard_error and isinstance(
            standard_error["stack_trace"], list
        ):
            if isinstance(standard_error["stack_trace"][0], dict):
                # If normalized to structured stack trace
                first_frame = standard_error["stack_trace"][0]
                for key, expected in [
                    ("namespace", "MyCompany.MyApp.Services"),
                    ("class", "UserService"),
                    ("function", "GetUserProfile"),
                    ("file", "D:\\Projects\\MyApp\\Services\\UserService.cs"),
                    ("line", 42),
                ]:
                    if first_frame.get(key) != expected:
                        success = False
                        print(
                            f"❌ Stack frame field {key}: Expected '{expected}', got '{first_frame.get(key)}'"
                        )
                    else:
                        print(f"✅ Stack frame field {key}: '{expected}'")
            else:
                # Not structured, but still a list
                print(
                    f"ℹ️ Stack trace normalized but not structured: {standard_error['stack_trace'][:2]}"
                )
        else:
            success = False
            print(
                f"❌ Stack trace field missing or not a list: {standard_error.get('stack_trace')}"
            )

        # Test denormalization
        csharp_error = self.adapter.from_standard_format(standard_error)

        # Check that key fields were preserved in the round trip
        for key, expected in [
            ("exception_type", "System.NullReferenceException"),
            ("message", "Object reference not set to an instance of an object"),
            ("dotnet_version", "6.0.16"),
            ("framework", "ASP.NET Core"),
        ]:
            if csharp_error.get(key) != expected:
                success = False
                print(
                    f"❌ Round-trip field {key}: Expected '{expected}', got '{csharp_error.get(key)}'"
                )
            else:
                print(f"✅ Round-trip field {key}: '{expected}'")

        # Check level capitalization
        if csharp_error.get("level") != "Error":
            success = False
            print(
                f"❌ Round-trip level capitalization: Expected 'Error', got '{csharp_error.get('level')}'"
            )
        else:
            print("✅ Round-trip level capitalization: 'Error'")

        # Check stack trace conversion
        if "stack_trace" in csharp_error and isinstance(
            csharp_error["stack_trace"], str
        ):
            if (
                "at MyCompany.MyApp.Services.UserService.GetUserProfile"
                in csharp_error["stack_trace"]
            ):
                print("✅ Round-trip stack trace contains expected method")
            else:
                success = False
                print(
                    f"❌ Round-trip stack trace missing expected method: {csharp_error['stack_trace'][:100]}"
                )
        else:
            success = False
            print(
                f"❌ Round-trip stack trace field missing or not a string: {csharp_error.get('stack_trace')}"
            )

        return success


def main():
    """Run the minimal test."""
    print("Running minimal C# error adapter test...\n")

    # Ensure rules directory exists
    setup_test_rules()

    # Create and run test
    test = MinimalCSharpPlugin()
    success = test.test_error_normalization()

    # Print final result
    print("\n" + "=" * 50)
    if success:
        print("✅ All C# error adapter tests passed!")
    else:
        print("❌ Some C# error adapter tests failed!")
    print("=" * 50)

    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
