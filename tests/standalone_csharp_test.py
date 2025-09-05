"""
Standalone test for C# error adapter without any external dependencies.
This test directly copies the relevant code from language_adapters.py to test
the C# error adapter functionality.
"""
import re
import uuid
from datetime import datetime
from typing import Dict, Any, List
import logging
import sys

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================================================================================
# Reduced version of the CSharpErrorAdapter from language_adapters.py
# ================================================================================


class StandaloneCSharpErrorAdapter:
    """Adapter for C# error formats."""
    
    def to_standard_format(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert C# error data to the standard format.
        
        Args:
            error_data: C# error data
            
        Returns:
            Error data in the standard format
        """
        # Create a standard error object
        standard_error = {
            "error_id": str(uuid.uuid4()),
            "timestamp": error_data.get("timestamp", datetime.now().isoformat()),
            "language": "csharp",
            "error_type": error_data.get("exception_type", ""),
            "message": error_data.get("message", "")
        }
        
        # Add .NET version if available
        if "dotnet_version" in error_data:
            standard_error["language_version"] = error_data["dotnet_version"]
        
        # Handle stack trace
        if "stack_trace" in error_data:
            # C# stack traces can be a string or a list
            if isinstance(error_data["stack_trace"], str):
                # Split into lines
                stack_lines = error_data["stack_trace"].split("\n")
                
                # Try to parse structured data from the stack trace
                parsed_frames = self._parse_csharp_stack_trace(stack_lines)
                
                if parsed_frames:
                    standard_error["stack_trace"] = parsed_frames
                else:
                    standard_error["stack_trace"] = stack_lines
            elif isinstance(error_data["stack_trace"], list):
                if all(isinstance(frame, dict) for frame in error_data["stack_trace"]):
                    # Already in structured format
                    standard_error["stack_trace"] = error_data["stack_trace"]
                else:
                    # List of strings
                    stack_lines = error_data["stack_trace"]
                    parsed_frames = self._parse_csharp_stack_trace(stack_lines)
                    
                    if parsed_frames:
                        standard_error["stack_trace"] = parsed_frames
                    else:
                        standard_error["stack_trace"] = stack_lines
        
        # Add framework information if available
        if "framework" in error_data:
            standard_error["framework"] = error_data["framework"]
            
            if "framework_version" in error_data:
                standard_error["framework_version"] = error_data["framework_version"]
        
        # Add request information if available
        if "request" in error_data:
            standard_error["request"] = error_data["request"]
        
        # Add any additional context
        if "context" in error_data:
            standard_error["context"] = error_data["context"]
        
        # Add inner exception if available
        if "inner_exception" in error_data:
            if "additional_data" not in standard_error:
                standard_error["additional_data"] = {}
            standard_error["additional_data"]["inner_exception"] = error_data["inner_exception"]
        
        # Add severity if available
        if "level" in error_data:
            # Map C# log levels to standard format
            level_map = {
                "trace": "debug",
                "debug": "debug",
                "information": "info",
                "warning": "warning",
                "error": "error",
                "critical": "critical",
                "fatal": "fatal"
            }
            standard_error["severity"] = level_map.get(error_data["level"].lower(), "error")
        
        # Add runtime if available
        if "runtime" in error_data:
            standard_error["runtime"] = error_data["runtime"]
            
            if "runtime_version" in error_data:
                standard_error["runtime_version"] = error_data["runtime_version"]
        
        # Add handled flag if available
        if "handled" in error_data:
            standard_error["handled"] = error_data["handled"]
        
        # Add additional C#-specific data
        csharp_specific = {}
        for key, value in error_data.items():
            if key not in standard_error and key not in ["stack_trace", "request", "context", "inner_exception"]:
                csharp_specific[key] = value
        
        if csharp_specific:
            if "additional_data" in standard_error:
                standard_error["additional_data"].update(csharp_specific)
            else:
                standard_error["additional_data"] = csharp_specific
        
        return standard_error
    
    def from_standard_format(self, standard_error: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert standard format error data to C#-specific format.
        
        Args:
            standard_error: Error data in the standard format
            
        Returns:
            Error data in the C#-specific format
        """
        # Create a C# error object
        csharp_error = {
            "timestamp": standard_error.get("timestamp", datetime.now().isoformat()),
            "exception_type": standard_error.get("error_type", "System.Exception"),
            "message": standard_error.get("message", "")
        }
        
        # Convert severity to C# logging level
        if "severity" in standard_error:
            level_map = {
                "debug": "Debug",
                "info": "Information",
                "warning": "Warning",
                "error": "Error",
                "critical": "Critical",
                "fatal": "Fatal"
            }
            csharp_error["level"] = level_map.get(standard_error["severity"].lower(), "Error")
        
        # Convert stack trace to C# format
        if "stack_trace" in standard_error:
            stack_trace = standard_error["stack_trace"]
            
            if isinstance(stack_trace, list):
                if all(isinstance(frame, str) for frame in stack_trace):
                    # Already in C# stack trace string format
                    csharp_error["stack_trace"] = "\n".join(stack_trace)
                elif all(isinstance(frame, dict) for frame in stack_trace):
                    # Convert structured frames to C# stack trace format
                    csharp_error["stack_trace"] = self._convert_frames_to_csharp_stack(
                        standard_error.get("error_type", "System.Exception"), 
                        standard_error.get("message", ""), 
                        stack_trace
                    )
                    # Also keep the structured version
                    csharp_error["structured_stack_trace"] = stack_trace
        
        # Add request information if available
        if "request" in standard_error:
            csharp_error["request"] = standard_error["request"]
        
        # Add context information if available
        if "context" in standard_error:
            csharp_error["context"] = standard_error["context"]
        
        # Add .NET version if available
        if "language_version" in standard_error:
            csharp_error["dotnet_version"] = standard_error["language_version"]
        
        # Add framework information if available
        if "framework" in standard_error:
            csharp_error["framework"] = standard_error["framework"]
            
            if "framework_version" in standard_error:
                csharp_error["framework_version"] = standard_error["framework_version"]
        
        # Add runtime information if available
        if "runtime" in standard_error:
            csharp_error["runtime"] = standard_error["runtime"]
            
            if "runtime_version" in standard_error:
                csharp_error["runtime_version"] = standard_error["runtime_version"]
        
        # Add handled flag if available
        if "handled" in standard_error:
            csharp_error["handled"] = standard_error["handled"]
        
        # Extract C#-specific data from additional_data
        if "additional_data" in standard_error:
            for key, value in standard_error["additional_data"].items():
                if key == "inner_exception":
                    csharp_error["inner_exception"] = value
                else:
                    csharp_error[key] = value
        
        return csharp_error
    
    def _parse_csharp_stack_trace(self, stack_lines: List[str]) -> List[Dict[str, Any]]:
        """
        Parse a C# stack trace into structured frames.
        
        Args:
            stack_lines: C# stack trace lines
            
        Returns:
            Structured frames or None if parsing fails
        """
        frames = []
        
        # C# stack trace patterns:
        # Standard format: at Namespace.Class.Method(parameters) in File:line number
        # Alternative: at Namespace.Class.Method(parameters) in c:\path\to\file.cs:line 42
        # Or simpler: at Namespace.Class.Method(parameters)
        
        # Regular patterns - simpler approach focused on the 'in file:line' pattern
        # First extract method signature
        method_pattern = r'\s*at\s+([^(]+)\(([^)]*)\)'
        # Then separately extract file location
        file_pattern = r'\s+in\s+(.+):line\s+(\d+)'
        
        try:
            for line in stack_lines:
                # Skip empty lines
                if not line.strip():
                    continue
                
                # Extract method first
                method_match = re.search(method_pattern, line)
                if not method_match:
                    continue
                    
                full_method = method_match.group(1)
                parameters = method_match.group(2)
                
                # Extract file and line if present
                file_match = re.search(file_pattern, line)
                file_path = file_match.group(1) if file_match else ""
                line_num = int(file_match.group(2)) if file_match else 0
                
                # Parse the full method name into namespace, class, and method
                parts = full_method.split('.')
                
                if len(parts) >= 3:
                    # Standard case: Namespace.Class.Method
                    namespace = '.'.join(parts[:-2])
                    class_name = parts[-2]
                    method = parts[-1]
                elif len(parts) == 2:
                    # Case: Class.Method
                    namespace = ""
                    class_name = parts[0]
                    method = parts[1]
                else:
                    # Case: Method (rare)
                    namespace = ""
                    class_name = ""
                    method = full_method
                
                # Handle special cases like lambda expressions
                if '<' in method and '>' in method:
                    # This is likely a lambda or anonymous method
                    pass  # We'll keep the method name with the angle brackets
                
                frames.append({
                    "namespace": namespace,
                    "class": class_name,
                    "function": method,
                    "parameters": parameters,
                    "file": file_path,
                    "line": line_num
                })
            
            return frames if frames else None
        except Exception as e:
            logger.debug(f"Failed to parse C# stack trace: {e}")
            return None
    
    def _convert_frames_to_csharp_stack(self, error_type: str, message: str, frames: List[Dict[str, Any]]) -> str:
        """
        Convert structured frames to a C# stack trace string.
        
        Args:
            error_type: Error type/name
            message: Error message
            frames: Structured frames
            
        Returns:
            C# stack trace string
        """
        stack_lines = [f"{error_type}: {message}"]
        
        for frame in frames:
            namespace = frame.get("namespace", "")
            class_name = frame.get("class", "")
            method = frame.get("function", "")
            parameters = frame.get("parameters", "")
            file = frame.get("file", "")
            line_num = frame.get("line", 0)
            
            # Build the full method name
            if namespace and class_name:
                full_method = f"{namespace}.{class_name}.{method}"
            elif class_name:
                full_method = f"{class_name}.{method}"
            else:
                full_method = method
            
            # Format the stack frame
            if file and line_num:
                stack_lines.append(f"   at {full_method}({parameters}) in {file}:line {line_num}")
            else:
                stack_lines.append(f"   at {full_method}({parameters})")
        
        return "\n".join(stack_lines)

# ================================================================================
# Test functions
# ================================================================================


def test_error_normalization():
    """Test error normalization from C# format to standard format."""
    print("Testing C# error normalization...")
    
    adapter = StandaloneCSharpErrorAdapter()
    
    # Sample null reference exception from C#
    null_reference_error = {
        "timestamp": "2023-08-15T12:34:56",
        "exception_type": "System.NullReferenceException",
        "message": "Object reference not set to an instance of an object",
        "stack_trace": [
            "   at MyCompany.MyApp.Services.UserService.GetUserProfile(Int32 userId) in D:\\Projects\\MyApp\\Services\\UserService.cs:line 42",
            "   at MyCompany.MyApp.Controllers.UserController.GetProfile(Int32 id) in D:\\Projects\\MyApp\\Controllers\\UserController.cs:line 28",
            "   at lambda_method(Closure, Object, Object[])",
            "   at Microsoft.AspNetCore.Mvc.Infrastructure.ActionMethodExecutor.SyncActionResultExecutor.Execute(ActionContext actionContext, IActionResultTypeMapper mapper, ObjectMethodExecutor executor, Object controller, Object[] arguments)",
            "   at Microsoft.AspNetCore.Mvc.Infrastructure.ControllerActionInvoker.InvokeActionMethodAsync()"
        ],
        "level": "error",
        "dotnet_version": "6.0.16",
        "framework": "ASP.NET Core",
        "framework_version": "6.0.16"
    }
    
    # Test normalization
    standard_error = adapter.to_standard_format(null_reference_error)
    
    # Check key fields
    success = True
    for key, expected in [
        ("language", "csharp"),
        ("error_type", "System.NullReferenceException"),
        ("message", "Object reference not set to an instance of an object"),
        ("language_version", "6.0.16"),
        ("framework", "ASP.NET Core"),
        ("framework_version", "6.0.16"),
        ("severity", "error")
    ]:
        if standard_error.get(key) != expected:
            print(f"❌ Field {key}: Expected '{expected}', got '{standard_error.get(key)}'")
            success = False
        else:
            print(f"✅ Field {key}: '{expected}'")
    
    # Check stack trace parsing
    if "stack_trace" in standard_error and isinstance(standard_error["stack_trace"], list):
        if isinstance(standard_error["stack_trace"][0], dict):
            first_frame = standard_error["stack_trace"][0]
            for key, expected in [
                ("namespace", "MyCompany.MyApp.Services"),
                ("class", "UserService"),
                ("function", "GetUserProfile"),
                ("file", "D:\\Projects\\MyApp\\Services\\UserService.cs"),
                ("line", 42)
            ]:
                if first_frame.get(key) != expected:
                    print(f"❌ Stack frame field {key}: Expected '{expected}', got '{first_frame.get(key)}'")
                    success = False
                else:
                    print(f"✅ Stack frame field {key}: '{expected}'")
        else:
            print(f"❌ Stack trace not normalized to structured frames: {standard_error['stack_trace']}")
            success = False
    else:
        print(f"❌ Stack trace missing or not a list: {standard_error.get('stack_trace')}")
        success = False
    
    return success


def test_error_denormalization():
    """Test denormalization from standard format back to C# format."""
    print("\nTesting C# error denormalization...")
    
    adapter = StandaloneCSharpErrorAdapter()
    
    # Create a standard error
    standard_error = {
        "error_id": str(uuid.uuid4()),
        "timestamp": "2023-08-15T12:34:56",
        "language": "csharp",
        "error_type": "System.NullReferenceException",
        "message": "Object reference not set to an instance of an object",
        "language_version": "6.0.16",
        "framework": "ASP.NET Core",
        "framework_version": "6.0.16",
        "severity": "error",
        "stack_trace": [
            {
                "namespace": "MyCompany.MyApp.Services",
                "class": "UserService",
                "function": "GetUserProfile",
                "parameters": "Int32 userId",
                "file": "D:\\Projects\\MyApp\\Services\\UserService.cs",
                "line": 42
            },
            {
                "namespace": "MyCompany.MyApp.Controllers",
                "class": "UserController",
                "function": "GetProfile",
                "parameters": "Int32 id",
                "file": "D:\\Projects\\MyApp\\Controllers\\UserController.cs",
                "line": 28
            }
        ]
    }
    
    # Denormalize to C# format
    csharp_error = adapter.from_standard_format(standard_error)
    
    # Check key fields
    success = True
    for key, expected in [
        ("exception_type", "System.NullReferenceException"),
        ("message", "Object reference not set to an instance of an object"),
        ("dotnet_version", "6.0.16"),
        ("framework", "ASP.NET Core"),
        ("framework_version", "6.0.16"),
        ("level", "Error")  # Note: capitalized in C# format
    ]:
        if csharp_error.get(key) != expected:
            print(f"❌ Field {key}: Expected '{expected}', got '{csharp_error.get(key)}'")
            success = False
        else:
            print(f"✅ Field {key}: '{expected}'")
    
    # Check stack trace formatting
    if "stack_trace" in csharp_error and isinstance(csharp_error["stack_trace"], str):
        expected_lines = [
            "System.NullReferenceException: Object reference not set to an instance of an object",
            "   at MyCompany.MyApp.Services.UserService.GetUserProfile(Int32 userId) in D:\\Projects\\MyApp\\Services\\UserService.cs:line 42",
            "   at MyCompany.MyApp.Controllers.UserController.GetProfile(Int32 id) in D:\\Projects\\MyApp\\Controllers\\UserController.cs:line 28"
        ]
        
        # Check each line is in the stack trace
        for line in expected_lines:
            if line not in csharp_error["stack_trace"]:
                print(f"❌ Stack trace missing line: '{line}'")
                success = False
            else:
                print(f"✅ Stack trace contains: '{line}'")
    else:
        print(f"❌ Stack trace missing or not a string: {csharp_error.get('stack_trace')}")
        success = False
    
    return success


def test_stack_trace_parsing():
    """Test C# stack trace parsing into structured frames."""
    print("\nTesting C# stack trace parsing...")
    
    adapter = StandaloneCSharpErrorAdapter()
    
    # Sample stack trace lines
    stack_trace = [
        "   at MyCompany.MyApp.Services.UserService.GetUserProfile(Int32 userId) in D:\\Projects\\MyApp\\Services\\UserService.cs:line 42",
        "   at MyCompany.MyApp.Controllers.UserController.GetProfile(Int32 id) in D:\\Projects\\MyApp\\Controllers\\UserController.cs:line 28",
        "   at System.Linq.Enumerable.SelectEnumerableIterator`2.MoveNext()",
        "   at MyCompany.MyApp.Services.DataService.<>c__DisplayClass5_0.<ProcessItems>b__0(Item item) in D:\\Projects\\MyApp\\Services\\DataService.cs:line 87"
    ]
    
    # Parse stack trace
    frames = adapter._parse_csharp_stack_trace(stack_trace)
    
    # Check that frames were parsed correctly
    success = True
    if not frames or not isinstance(frames, list):
        print(f"❌ Failed to parse frames: {frames}")
        return False
    
    # Check first standard frame
    first_frame = frames[0]
    for key, expected in [
        ("namespace", "MyCompany.MyApp.Services"),
        ("class", "UserService"),
        ("function", "GetUserProfile"),
        ("parameters", "Int32 userId"),
        ("file", "D:\\Projects\\MyApp\\Services\\UserService.cs"),
        ("line", 42)
    ]:
        if first_frame.get(key) != expected:
            print(f"❌ First frame field {key}: Expected '{expected}', got '{first_frame.get(key)}'")
            success = False
        else:
            print(f"✅ First frame field {key}: '{expected}'")
    
    # Check lambda frame
    lambda_frame = frames[3]
    if lambda_frame.get("file") != "D:\\Projects\\MyApp\\Services\\DataService.cs" or lambda_frame.get("line") != 87:
        print(f"❌ Lambda frame parsing failed: {lambda_frame}")
        success = False
    else:
        print(f"✅ Lambda frame parsing succeeded: file={lambda_frame.get('file')}, line={lambda_frame.get('line')}")
    
    return success


def main():
    """Run all tests."""
    print("Running standalone C# error adapter tests...\n")
    
    # Run tests
    norm_success = test_error_normalization()
    denorm_success = test_error_denormalization()
    parsing_success = test_stack_trace_parsing()
    
    # Print final results
    print("\n" + "=" * 50)
    print(f"Error normalization: {'✅ PASSED' if norm_success else '❌ FAILED'}")
    print(f"Error denormalization: {'✅ PASSED' if denorm_success else '❌ FAILED'}")
    print(f"Stack trace parsing: {'✅ PASSED' if parsing_success else '❌ FAILED'}")
    
    all_success = norm_success and denorm_success and parsing_success
    print(f"\nOverall result: {'✅ ALL TESTS PASSED' if all_success else '❌ SOME TESTS FAILED'}")
    print("=" * 50)
    
    return 0 if all_success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)