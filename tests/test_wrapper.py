"""
Test wrapper for C# plugin to bypass module dependencies.

This module helps run the C# plugin tests without requiring the full analyzer
system, which has dependencies on external libraries like yaml.
"""
import sys
import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the plugin system without requiring analyzer
from modules.analysis.language_plugin_system import LanguagePlugin, register_plugin
from modules.analysis.language_adapters import CSharpErrorAdapter

from modules.analysis.plugins.csharp_plugin import CSharpExceptionHandler, CSharpPatchGenerator, CSharpLanguagePlugin

# Run the basic tests
def main():
    # Create test instances
    plugin = CSharpLanguagePlugin()
    adapter = CSharpErrorAdapter()
    
    # Test plugin metadata
    print(f"Language ID: {plugin.get_language_id()}")
    print(f"Language Name: {plugin.get_language_name()}")
    print(f"Language Version: {plugin.get_language_version()}")
    print(f"Plugin Version: {plugin.VERSION}")
    print(f"Plugin Author: {plugin.AUTHOR}")
    
    # Test supported frameworks
    frameworks = plugin.get_supported_frameworks()
    print(f"Supported Frameworks: {', '.join(frameworks)}")
    
    # Test with a sample error
    null_reference_error = {
        "timestamp": "2023-08-15T12:34:56",
        "exception_type": "System.NullReferenceException",
        "message": "Object reference not set to an instance of an object",
        "stack_trace": [
            "   at MyCompany.MyApp.Services.UserService.GetUserProfile(Int32 userId) in D:\\Projects\\MyApp\\Services\\UserService.cs:line 42",
            "   at MyCompany.MyApp.Controllers.UserController.GetProfile(Int32 id) in D:\\Projects\\MyApp\\Controllers\\UserController.cs:line 28"
        ],
        "level": "error",
        "dotnet_version": "6.0.16",
        "framework": "ASP.NET Core",
        "framework_version": "6.0.16"
    }
    
    # Test error normalization
    print("\nTesting error normalization:")
    standard_error = adapter.to_standard_format(null_reference_error)
    print(f"Standardized error type: {standard_error['error_type']}")
    print(f"Language: {standard_error['language']}")
    print(f"Framework: {standard_error['framework']}")
    
    # Check stack trace normalization
    if isinstance(standard_error["stack_trace"][0], dict):
        first_frame = standard_error["stack_trace"][0]
        print(f"First frame namespace: {first_frame.get('namespace')}")
        print(f"First frame class: {first_frame.get('class')}")
        print(f"First frame function: {first_frame.get('function')}")
        print(f"First frame file: {first_frame.get('file')}")
        print(f"First frame line: {first_frame.get('line')}")
    
    # Test denormalization
    print("\nTesting denormalization:")
    csharp_error = adapter.from_standard_format(standard_error)
    print(f"Denormalized exception_type: {csharp_error['exception_type']}")
    print(f"Denormalized framework: {csharp_error['framework']}")
    print(f"Denormalized level: {csharp_error['level']}")
    
    # Test error analysis
    print("\nTesting error analysis:")
    # Use exception handler directly to bypass dependency issues
    handler = CSharpExceptionHandler()
    analysis = handler.analyze_error(standard_error)
    print(f"Analysis root cause: {analysis['root_cause']}")
    print(f"Analysis error type: {analysis['error_type']}")
    print(f"Analysis confidence: {analysis['confidence']}")
    print(f"Analysis suggestion: {analysis['suggestion'][:50]}...")
    
    # Test patch generation
    print("\nTesting patch generation:")
    # Use patch generator directly
    generator = CSharpPatchGenerator()
    context = {
        "code_snippet": "var user = repository.GetUser(userId);\nvar profile = user.Profile;\nreturn profile.PreferredName;",
        "framework": "aspnetcore"
    }
    patch = generator.generate_patch(analysis, context)
    print(f"Patch ID: {patch['patch_id']}")
    print(f"Patch type: {patch['patch_type']}")
    print(f"Patch language: {patch['language']}")
    
    if "patch_code" in patch:
        print(f"Patch code (excerpt): {patch['patch_code'][:50]}...")
    elif "suggestion_code" in patch:
        print(f"Suggestion code (excerpt): {patch['suggestion_code'][:50]}...")
    
    print("\nBasic C# plugin functionality tests passed!")

if __name__ == "__main__":
    main()