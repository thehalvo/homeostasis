"""
Tests for the C# language plugin.

This test suite verifies the functionality of the C# error handling, analysis, and
patch generation capabilities of the Homeostasis framework.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

from modules.analysis.language_adapters import CSharpErrorAdapter
from modules.analysis.plugins.csharp_plugin import CSharpLanguagePlugin

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestCSharpPlugin(unittest.TestCase):
    """Test case for the C# language plugin."""

    def setUp(self):
        """Set up test case."""
        self.plugin = CSharpLanguagePlugin()
        self.adapter = CSharpErrorAdapter()

        # Sample C# exceptions for testing
        self.null_reference_error = {
            "timestamp": "2023-08-15T12:34:56",
            "exception_type": "System.NullReferenceException",
            "message": "Object reference not set to an instance of an object",
            "stack_trace": [
                "   at MyCompany.MyApp.Services.UserService.GetUserProfile(Int32 userId) in D:\\Projects\\MyApp\\Services\\UserService.cs:line 42",
                "   at MyCompany.MyApp.Controllers.UserController.GetProfile(Int32 id) in D:\\Projects\\MyApp\\Controllers\\UserController.cs:line 28",
                "   at lambda_method(Closure, Object, Object[])",
                "   at Microsoft.AspNetCore.Mvc.Infrastructure.ActionMethodExecutor.SyncActionResultExecutor.Execute(ActionContext actionContext, IActionResultTypeMapper mapper, ObjectMethodExecutor executor, Object controller, Object[] arguments)",
                "   at Microsoft.AspNetCore.Mvc.Infrastructure.ControllerActionInvoker.InvokeActionMethodAsync()",
            ],
            "level": "error",
            "dotnet_version": "6.0.16",
            "framework": "ASP.NET Core",
            "framework_version": "6.0.16",
        }

        self.db_update_error = {
            "timestamp": "2023-08-15T12:45:23",
            "exception_type": "Microsoft.EntityFrameworkCore.DbUpdateException",
            "message": "An error occurred while updating the entries. See the inner exception for details.",
            "stack_trace": [
                "   at Microsoft.EntityFrameworkCore.Update.ReaderModificationCommandBatch.ExecuteAsync(IRelationalConnection connection, CancellationToken cancellationToken)",
                "   at Microsoft.EntityFrameworkCore.Update.Internal.BatchExecutor.ExecuteAsync(DbContext _, ValueTuple`2 parameters, CancellationToken cancellationToken)",
                "   at Microsoft.EntityFrameworkCore.SqlServer.Storage.Internal.SqlServerExecutionStrategy.ExecuteAsync[TState,TResult](TState state, Func`4 operation, Func`4 verifySucceeded, CancellationToken cancellationToken)",
                "   at Microsoft.EntityFrameworkCore.ChangeTracking.Internal.StateManager.SaveChangesAsync(IList`1 entriesToSave, CancellationToken cancellationToken)",
                "   at Microsoft.EntityFrameworkCore.ChangeTracking.Internal.StateManager.SaveChangesAsync(StateManager stateManager, Boolean acceptAllChangesOnSuccess, CancellationToken cancellationToken)",
                "   at Microsoft.EntityFrameworkCore.DbContext.SaveChangesAsync(Boolean acceptAllChangesOnSuccess, CancellationToken cancellationToken)",
                "   at MyCompany.MyApp.Services.OrderService.CreateOrder(Order order) in D:\\Projects\\MyApp\\Services\\OrderService.cs:line 83",
                "   at MyCompany.MyApp.Controllers.OrderController.Post(OrderDto orderDto) in D:\\Projects\\MyApp\\Controllers\\OrderController.cs:line 45",
            ],
            "inner_exception": {
                "exception_type": "Microsoft.Data.SqlClient.SqlException",
                "message": "Violation of UNIQUE KEY constraint 'UQ_Orders_OrderNumber'. Cannot insert duplicate key in object 'dbo.Orders'. The duplicate key value is (ORD-2023-0001).",
            },
            "level": "error",
            "dotnet_version": "6.0.16",
            "framework": "ASP.NET Core",
            "framework_version": "6.0.16",
        }

        self.task_canceled_error = {
            "timestamp": "2023-08-15T13:02:17",
            "exception_type": "System.Threading.Tasks.TaskCanceledException",
            "message": "A task was canceled.",
            "stack_trace": [
                "   at System.Threading.Tasks.Task.ThrowIfExceptional(Boolean includeTaskCanceledExceptions)",
                "   at System.Threading.Tasks.Task.Wait(Int32 millisecondsTimeout, CancellationToken cancellationToken)",
                "   at System.Threading.Tasks.Task.Wait(CancellationToken cancellationToken)",
                "   at MyCompany.MyApp.Services.ApiClient.GetDataAsync(String endpoint, CancellationToken cancellationToken) in D:\\Projects\\MyApp\\Services\\ApiClient.cs:line 57",
                "   at MyCompany.MyApp.Controllers.DataController.Get(String id, CancellationToken cancellationToken) in D:\\Projects\\MyApp\\Controllers\\DataController.cs:line 32",
            ],
            "level": "warning",
            "dotnet_version": "6.0.16",
            "framework": "ASP.NET Core",
            "framework_version": "6.0.16",
        }

    def test_plugin_metadata(self):
        """Test plugin metadata and registration."""
        self.assertEqual(self.plugin.get_language_id(), "csharp")
        self.assertEqual(self.plugin.get_language_name(), "C#")
        self.assertEqual(self.plugin.get_language_version(), "7.0+")
        self.assertIsInstance(self.plugin.VERSION, str)
        self.assertIsInstance(self.plugin.AUTHOR, str)

        # Check supported frameworks
        frameworks = self.plugin.get_supported_frameworks()
        self.assertIn("aspnetcore", frameworks)
        self.assertIn("entityframework", frameworks)
        self.assertIn("azure", frameworks)

    def test_error_normalization(self):
        """Test error normalization from C# format to standard format."""
        # Test null reference exception
        standard_error = self.adapter.to_standard_format(self.null_reference_error)

        self.assertEqual(standard_error["language"], "csharp")
        self.assertEqual(standard_error["error_type"], "System.NullReferenceException")
        self.assertEqual(
            standard_error["message"],
            "Object reference not set to an instance of an object",
        )
        self.assertEqual(standard_error["language_version"], "6.0.16")
        self.assertEqual(standard_error["framework"], "ASP.NET Core")
        self.assertEqual(standard_error["framework_version"], "6.0.16")
        self.assertEqual(standard_error["severity"], "error")

        # Check stack trace normalization
        self.assertIsInstance(standard_error["stack_trace"], list)
        if isinstance(standard_error["stack_trace"][0], dict):
            # If normalized to structured stack trace
            first_frame = standard_error["stack_trace"][0]
            self.assertEqual(first_frame["namespace"], "MyCompany.MyApp.Services")
            self.assertEqual(first_frame["class"], "UserService")
            self.assertEqual(first_frame["function"], "GetUserProfile")
            self.assertEqual(
                first_frame["file"], "D:\\Projects\\MyApp\\Services\\UserService.cs"
            )
            self.assertEqual(first_frame["line"], 42)

    def test_denormalization(self):
        """Test converting from standard format back to C# format."""
        # First normalize to standard format
        standard_error = self.adapter.to_standard_format(self.null_reference_error)

        # Then denormalize back to C# format
        csharp_error = self.adapter.from_standard_format(standard_error)

        # Check that important fields are preserved
        self.assertEqual(
            csharp_error["exception_type"], "System.NullReferenceException"
        )
        self.assertEqual(
            csharp_error["message"],
            "Object reference not set to an instance of an object",
        )
        self.assertEqual(csharp_error["dotnet_version"], "6.0.16")
        self.assertEqual(csharp_error["framework"], "ASP.NET Core")
        self.assertEqual(
            csharp_error["level"], "Error"
        )  # Note: capitalized in C# format

        # Check stack trace conversion
        self.assertIsInstance(csharp_error["stack_trace"], str)
        self.assertIn(
            "at MyCompany.MyApp.Services.UserService.GetUserProfile",
            csharp_error["stack_trace"],
        )

    def test_error_analysis_null_reference(self):
        """Test error analysis for NullReferenceException."""
        # First normalize the error
        standard_error = self.adapter.to_standard_format(self.null_reference_error)

        # Analyze the error
        analysis = self.plugin.analyze_error(standard_error)

        # Check analysis results
        self.assertEqual(analysis["error_type"], "System.NullReferenceException")
        self.assertEqual(analysis["root_cause"], "csharp_null_reference")
        self.assertEqual(analysis["confidence"], "high")
        self.assertIsNotNone(analysis["suggestion"])
        self.assertIsNotNone(analysis["description"])

    def test_error_analysis_db_update(self):
        """Test error analysis for DbUpdateException."""
        # First normalize the error
        standard_error = self.adapter.to_standard_format(self.db_update_error)

        # Analyze the error
        analysis = self.plugin.analyze_error(standard_error)

        # Check analysis results
        self.assertEqual(
            analysis["error_type"], "Microsoft.EntityFrameworkCore.DbUpdateException"
        )
        self.assertEqual(analysis["root_cause"], "ef_db_update_failed")
        self.assertEqual(analysis["confidence"], "high")
        self.assertEqual(analysis["category"], "entityframework")
        self.assertEqual(analysis["framework"], "entityframework")
        self.assertIsNotNone(analysis["suggestion"])

    def test_error_analysis_task_canceled(self):
        """Test error analysis for TaskCanceledException."""
        # First normalize the error
        standard_error = self.adapter.to_standard_format(self.task_canceled_error)

        # Analyze the error
        analysis = self.plugin.analyze_error(standard_error)

        # Check analysis results
        self.assertEqual(
            analysis["error_type"], "System.Threading.Tasks.TaskCanceledException"
        )
        self.assertEqual(analysis["root_cause"], "csharp_task_canceled")
        self.assertEqual(analysis["category"], "async")
        self.assertIsNotNone(analysis["suggestion"])

    def test_patch_generation_null_reference(self):
        """Test patch generation for NullReferenceException."""
        # Setup - normalize and analyze error
        standard_error = self.adapter.to_standard_format(self.null_reference_error)
        analysis = self.plugin.analyze_error(standard_error)

        # Create context for patch generation
        context = {
            "code_snippet": "var user = repository.GetUser(userId);\nvar profile = user.Profile;\nreturn profile.PreferredName;",
            "framework": "aspnetcore",
        }

        # Generate patch
        patch = self.plugin.generate_fix(analysis, context)

        # Check patch results
        self.assertEqual(patch["patch_id"], f"csharp_{analysis['rule_id']}")
        self.assertEqual(patch["language"], "csharp")
        self.assertEqual(patch["framework"], "aspnetcore")
        self.assertEqual(patch["root_cause"], "csharp_null_reference")

        # Check patch content (either code or suggestion)
        self.assertTrue("patch_code" in patch or "suggestion_code" in patch)
        if "patch_code" in patch:
            self.assertIsInstance(patch["patch_code"], str)
            self.assertIsNotNone(patch["application_point"])
        else:
            self.assertIsInstance(patch["suggestion_code"], str)

    def test_cross_platform_conversion(self):
        """Test converting errors between languages."""
        # Normalize C# error to standard format
        standard_error = self.adapter.to_standard_format(self.null_reference_error)

        # Convert to Python format using the global converter
        with patch(
            "modules.analysis.language_adapters.ErrorAdapterFactory.get_adapter"
        ) as mock_adapter:
            # Mock Python adapter
            python_adapter = MagicMock()
            python_adapter.from_standard_format.return_value = {
                "timestamp": standard_error["timestamp"],
                "exception_type": "NullReferenceError",  # Python equivalent
                "message": standard_error["message"],
                "traceback": ["Traceback (most recent call last):", "  ..."],
            }
            mock_adapter.return_value = python_adapter

            # Convert to Python format
            from modules.analysis.language_adapters import \
                convert_from_standard_format

            python_error = convert_from_standard_format(standard_error, "python")

            # Verify conversion
            self.assertTrue(mock_adapter.called)
            self.assertTrue(python_adapter.from_standard_format.called)
            self.assertEqual(python_error["exception_type"], "NullReferenceError")


class TestCSharpStackTraceParsing(unittest.TestCase):
    """Test parsing C# stack traces into structured frames."""

    def setUp(self):
        """Set up test case."""
        self.adapter = CSharpErrorAdapter()

    def test_parse_standard_stack_trace(self):
        """Test parsing a standard C# stack trace."""
        stack_trace = [
            "   at MyCompany.MyApp.Services.UserService.GetUserProfile(Int32 userId) in D:\\Projects\\MyApp\\Services\\UserService.cs:line 42",
            "   at MyCompany.MyApp.Controllers.UserController.GetProfile(Int32 id) in D:\\Projects\\MyApp\\Controllers\\UserController.cs:line 28",
        ]

        frames = self.adapter._parse_csharp_stack_trace(stack_trace)

        self.assertEqual(len(frames), 2)

        # Check first frame
        self.assertEqual(frames[0]["namespace"], "MyCompany.MyApp.Services")
        self.assertEqual(frames[0]["class"], "UserService")
        self.assertEqual(frames[0]["function"], "GetUserProfile")
        self.assertEqual(frames[0]["parameters"], "Int32 userId")
        self.assertEqual(
            frames[0]["file"], "D:\\Projects\\MyApp\\Services\\UserService.cs"
        )
        self.assertEqual(frames[0]["line"], 42)

        # Check second frame
        self.assertEqual(frames[1]["namespace"], "MyCompany.MyApp.Controllers")
        self.assertEqual(frames[1]["class"], "UserController")
        self.assertEqual(frames[1]["function"], "GetProfile")
        self.assertEqual(frames[1]["parameters"], "Int32 id")
        self.assertEqual(
            frames[1]["file"], "D:\\Projects\\MyApp\\Controllers\\UserController.cs"
        )
        self.assertEqual(frames[1]["line"], 28)

    def test_parse_lambda_stack_trace(self):
        """Test parsing a stack trace with lambda expressions."""
        stack_trace = [
            "   at System.Linq.Enumerable.SelectEnumerableIterator`2.MoveNext()",
            "   at System.Linq.Enumerable.WhereSelectEnumerableIterator`2.MoveNext()",
            "   at MyCompany.MyApp.Services.DataService.<>c__DisplayClass5_0.<ProcessItems>b__0(Item item) in D:\\Projects\\MyApp\\Services\\DataService.cs:line 87",
        ]

        frames = self.adapter._parse_csharp_stack_trace(stack_trace)

        self.assertEqual(len(frames), 3)

        # Check lambda frame
        lambda_frame = frames[2]
        self.assertEqual(lambda_frame["namespace"], "MyCompany.MyApp.Services")
        self.assertEqual(lambda_frame["class"], "DataService")
        self.assertTrue("DisplayClass" in lambda_frame["function"])
        self.assertEqual(
            lambda_frame["file"], "D:\\Projects\\MyApp\\Services\\DataService.cs"
        )
        self.assertEqual(lambda_frame["line"], 87)

    def test_convert_frames_to_stack(self):
        """Test converting structured frames back to C# stack trace string."""
        frames = [
            {
                "namespace": "MyCompany.MyApp.Services",
                "class": "UserService",
                "function": "GetUserProfile",
                "parameters": "Int32 userId",
                "file": "D:\\Projects\\MyApp\\Services\\UserService.cs",
                "line": 42,
            },
            {
                "namespace": "MyCompany.MyApp.Controllers",
                "class": "UserController",
                "function": "GetProfile",
                "parameters": "Int32 id",
                "file": "D:\\Projects\\MyApp\\Controllers\\UserController.cs",
                "line": 28,
            },
        ]

        error_type = "System.NullReferenceException"
        message = "Object reference not set to an instance of an object"

        stack_trace = self.adapter._convert_frames_to_csharp_stack(
            error_type, message, frames
        )

        # Check the formatted stack trace
        self.assertTrue(stack_trace.startswith(f"{error_type}: {message}"))
        self.assertIn(
            "at MyCompany.MyApp.Services.UserService.GetUserProfile(Int32 userId) in D:\\Projects\\MyApp\\Services\\UserService.cs:line 42",
            stack_trace,
        )
        self.assertIn(
            "at MyCompany.MyApp.Controllers.UserController.GetProfile(Int32 id) in D:\\Projects\\MyApp\\Controllers\\UserController.cs:line 28",
            stack_trace,
        )


if __name__ == "__main__":
    unittest.main()
