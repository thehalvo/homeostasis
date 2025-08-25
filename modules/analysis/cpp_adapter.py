"""
C/C++ Error Adapter for Homeostasis

This module provides the CPPErrorAdapter class for converting between
C/C++ error formats and the standard Homeostasis error format.
"""

from typing import Dict, Any, List, Union


class CPPErrorAdapter:
    """
    Adapter for converting C/C++ errors to/from standard format.
    
    This adapter handles the conversion between C/C++ compiler errors,
    runtime errors, and the standard error format used by Homeostasis.
    """
    
    def to_standard_format(self, cpp_error: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert C/C++ error to standard format.
        
        Args:
            cpp_error: C/C++ error data (compiler or runtime)
            
        Returns:
            Standardized error data
        """
        # Extract error information
        error_type = cpp_error.get("type", cpp_error.get("error_type", ""))
        message = cpp_error.get("message", cpp_error.get("what", ""))
        
        # Handle different stack trace formats
        stack_trace = self._extract_stack_trace(cpp_error)
        
        # Extract file and line information
        file_info = self._extract_file_info(cpp_error, stack_trace)
        
        # Determine error category
        category = self._categorize_error(error_type, message)
        
        standard_format = {
            "error_type": error_type or "UnknownError",
            "message": message,
            "stack_trace": stack_trace,
            "language": "cpp",
            "timestamp": cpp_error.get("timestamp"),
            "service_name": cpp_error.get("service_name"),
            "file": file_info.get("file"),
            "line": file_info.get("line"),
            "column": file_info.get("column"),
            "category": category,
            "compiler": cpp_error.get("compiler"),
            "compilation_flags": cpp_error.get("flags", []),
            "standard": cpp_error.get("standard"),  # Preserve standard field
            "additional_data": {
                "error_code": cpp_error.get("error_code"),
                "severity": cpp_error.get("severity", "error"),
                "phase": cpp_error.get("phase"),  # compile-time, link-time, runtime
                "notes": cpp_error.get("notes", []),
                "signal": cpp_error.get("signal"),
                "address": cpp_error.get("address"),
                "tool": cpp_error.get("tool"),
                "bytes_lost": cpp_error.get("bytes_lost"),
                "blocks_lost": cpp_error.get("blocks_lost")
            }
        }
        
        # Add template-specific fields if present
        if "instantiation_stack" in cpp_error:
            standard_format["instantiation_stack"] = cpp_error["instantiation_stack"]
        
        # Handle multiple errors (e.g., multiple compilation errors)
        if "errors" in cpp_error:
            standard_format["errors"] = cpp_error["errors"]
        
        # Handle warning as error flag
        if "warning_as_error" in cpp_error:
            standard_format["warning_as_error"] = cpp_error["warning_as_error"]
        
        # Handle linker error references
        if "references" in cpp_error:
            standard_format["references"] = cpp_error["references"]
            
        return standard_format
    
    def from_standard_format(self, standard_error: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert standard format error to C/C++ format.
        
        Args:
            standard_error: Error in standard format
            
        Returns:
            C/C++ formatted error data
        """
        additional_data = standard_error.get("additional_data", {})
        
        cpp_error = {
            "type": standard_error.get("error_type", "Error"),
            "message": standard_error.get("message", ""),
            "file": standard_error.get("file", ""),
            "line": standard_error.get("line", 0),
            "column": standard_error.get("column", 0),
            "timestamp": standard_error.get("timestamp"),
            "compiler": standard_error.get("compiler", "g++"),
            "severity": additional_data.get("severity", "error"),
            "phase": additional_data.get("phase", "runtime")
        }
        
        # Format stack trace
        stack_trace = standard_error.get("stack_trace", [])
        if stack_trace:
            cpp_error["backtrace"] = self._format_backtrace(stack_trace)
        
        # Add error code if available
        if "error_code" in additional_data:
            cpp_error["error_code"] = additional_data["error_code"]
        
        # Add compilation flags if available
        if "compilation_flags" in standard_error:
            cpp_error["flags"] = standard_error["compilation_flags"]
        
        # Add notes if available
        if "notes" in additional_data:
            cpp_error["notes"] = additional_data["notes"]
        
        return cpp_error
    
    def _extract_stack_trace(self, cpp_error: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract and parse stack trace from C/C++ error."""
        # Check for stacktrace field (commonly used)
        if "stacktrace" in cpp_error:
            trace = cpp_error["stacktrace"]
            if isinstance(trace, list) and trace and isinstance(trace[0], str):
                # Parse GDB-style stack trace
                return self._parse_gdb_stack_trace(trace)
            elif isinstance(trace, list):
                return trace  # Already structured
            
        # Check for backtrace field
        if "backtrace" in cpp_error:
            trace = cpp_error["backtrace"]
            if isinstance(trace, list) and trace and isinstance(trace[0], str):
                return self._parse_gdb_stack_trace(trace)
            elif isinstance(trace, list):
                return trace
        
        # Check for stack_trace field
        if "stack_trace" in cpp_error:
            trace = cpp_error["stack_trace"]
            if isinstance(trace, list) and trace and isinstance(trace[0], str):
                return self._parse_gdb_stack_trace(trace)
            elif isinstance(trace, list):
                return trace
        
        # Check for frames field (some debuggers use this)
        if "frames" in cpp_error:
            frames = cpp_error["frames"]
            if isinstance(frames, list):
                return frames
        
        # Try to extract from message (for segfaults and such)
        message = cpp_error.get("message", "")
        if "Segmentation fault" in message or "core dumped" in message:
            # Return a minimal stack trace indicator
            return [{"frame": 0, "signal": "SIGSEGV", "description": "Segmentation fault"}]
        
        return []
    
    def _parse_gdb_stack_trace(self, stack_trace: Union[str, List[str]]) -> List[Dict[str, Any]]:
        """
        Parse GDB-style stack trace into structured format.
        
        Args:
            stack_trace: GDB stack trace as string or list of lines
            
        Returns:
            List of parsed stack frames
        """
        import re
        
        if isinstance(stack_trace, str):
            lines = stack_trace.strip().split('\n')
        else:
            lines = stack_trace
            
        frames = []
        
        # GDB frame patterns
        # #0  0x0000000000400546 in function (args) at file.cpp:line
        # #0  function (args) at file.cpp:line
        # #0  0x0000000000400546 in function () from library.so
        gdb_pattern = re.compile(
            r'^#(\d+)\s+'                      # Frame number
            r'(?:0x[0-9a-fA-F]+\s+in\s+)?'    # Optional address
            r'([^\s(]+)'                       # Function name
            r'\s*\([^)]*\)'                    # Arguments in parentheses
            r'(?:\s+at\s+([^:]+):(\d+))?'     # Optional file:line
            r'(?:\s+from\s+(.+))?'            # Optional library
        )
        
        for line in lines:
            line = line.strip()
            match = gdb_pattern.match(line)
            if match:
                frame_num = int(match.group(1))
                function = match.group(2)
                file_path = match.group(3)
                line_num = match.group(4)
                library = match.group(5)
                
                frame = {
                    "frame": frame_num,
                    "function": function
                }
                
                if file_path:
                    frame["file"] = file_path
                if line_num:
                    frame["line"] = int(line_num)
                if library:
                    frame["library"] = library
                    
                frames.append(frame)
                
        return frames
    
    def _format_frame(self, frame: Union[str, Dict[str, Any]]) -> str:
        """Format a single stack frame."""
        if isinstance(frame, str):
            return frame
        
        # Format structured frame
        if isinstance(frame, dict):
            func = frame.get("function", "??")
            file = frame.get("file", "??")
            line = frame.get("line", 0)
            addr = frame.get("address", "")
            
            if addr:
                return f"[{addr}] {func} at {file}:{line}"
            else:
                return f"{func} at {file}:{line}"
        
        return str(frame)
    
    def _extract_file_info(self, cpp_error: Dict[str, Any], stack_trace: List[str]) -> Dict[str, Any]:
        """Extract file, line, and column information."""
        import re
        
        file_info = {}
        
        # Direct fields
        if "file" in cpp_error:
            file_info["file"] = cpp_error["file"]
        if "line" in cpp_error:
            file_info["line"] = cpp_error["line"]
        if "column" in cpp_error:
            file_info["column"] = cpp_error["column"]
        
        # If not found, try to extract from message or stack trace
        if not file_info.get("file"):
            # Common patterns: file.cpp:123:45: error
            pattern = r'([^:]+\.(c|cpp|cc|cxx|h|hpp)):(\d+):(\d+)'
            
            # Check message
            message = cpp_error.get("message", "")
            match = re.search(pattern, message)
            if match:
                file_info["file"] = match.group(1)
                file_info["line"] = int(match.group(3))
                file_info["column"] = int(match.group(4))
            
            # Check stack trace
            elif stack_trace:
                for frame in stack_trace[:3]:  # Check first few frames
                    match = re.search(pattern, str(frame))
                    if match:
                        file_info["file"] = match.group(1)
                        file_info["line"] = int(match.group(3))
                        break
        
        return file_info
    
    def _categorize_error(self, error_type: str, message: str) -> str:
        """Categorize the error based on type and message."""
        # Compilation errors
        if any(x in error_type.lower() for x in ["compilererror", "syntaxerror", "parseerror", "compilationwarning", "warning"]):
            return "compilation"
        
        # Linker errors
        if any(x in message.lower() for x in ["undefined reference", "linker error", "ld:"]):
            return "linker"
        
        # Memory errors
        if any(x in error_type.lower() + message.lower() for x in 
               ["segmentation", "sigsegv", "memory", "heap", "stack overflow"]):
            return "memory"
        
        # Runtime errors
        if any(x in error_type.lower() for x in ["runtime", "sigabrt", "sigfpe", "sigill"]):
            return "runtime"
        
        # Template errors
        if "template" in message.lower() or "instantiation" in message.lower():
            return "template"
        
        # STL errors
        if any(x in message for x in ["std::", "boost::", "<algorithm>", "<vector>", "<map>"]):
            return "stl"
        
        return "general"
    
    def _format_backtrace(self, stack_trace: List[Any]) -> List[str]:
        """Format stack trace for C/C++ output."""
        formatted = []
        
        for i, frame in enumerate(stack_trace):
            if isinstance(frame, str):
                formatted.append(f"#{i} {frame}")
            elif isinstance(frame, dict):
                func = frame.get("function", "??")
                file = frame.get("file", "??")
                line = frame.get("line", 0)
                formatted.append(f"#{i} {func} at {file}:{line}")
            else:
                formatted.append(f"#{i} {str(frame)}")
        
        return formatted