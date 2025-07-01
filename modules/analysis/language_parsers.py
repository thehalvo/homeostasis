"""
Additional language-specific parsers for comprehensive error detection.

This module extends the comprehensive error detector with parsers for
additional programming languages including Java, Go, Rust, C#, and more.
"""

import re
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

from .comprehensive_error_detector import (
    LanguageSpecificParser, LanguageType, ErrorCategory, ErrorContext
)

logger = logging.getLogger(__name__)


class JavaParser(LanguageSpecificParser):
    """Java-specific error parser with compilation and runtime error detection."""
    
    def __init__(self):
        super().__init__(LanguageType.JAVA)
        
        # Java compilation error patterns
        self.compilation_patterns = [
            (r"error: cannot find symbol", "symbol_not_found"),
            (r"error: class, interface, or enum expected", "syntax_error"),
            (r"error: reached end of file while parsing", "unexpected_eof"),
            (r"error: illegal start of expression", "illegal_expression"),
            (r"error: incompatible types", "type_mismatch"),
            (r"error: method .* in class .* cannot be applied", "method_signature_mismatch"),
            (r"error: package .* does not exist", "missing_package"),
        ]
        
        # Java runtime error patterns
        self.runtime_patterns = [
            (r"java\.lang\.NullPointerException", "null_pointer_exception"),
            (r"java\.lang\.IndexOutOfBoundsException", "index_out_of_bounds"),
            (r"java\.lang\.ClassCastException", "class_cast_exception"),
            (r"java\.lang\.IllegalArgumentException", "illegal_argument"),
            (r"java\.lang\.NumberFormatException", "number_format_exception"),
            (r"java\.lang\.OutOfMemoryError", "out_of_memory"),
            (r"java\.lang\.StackOverflowError", "stack_overflow"),
            (r"java\.io\.FileNotFoundException", "file_not_found"),
            (r"java\.net\.ConnectException", "connection_failed"),
            (r"java\.sql\.SQLException", "sql_exception"),
        ]
        
        # Framework-specific patterns
        self.framework_patterns = [
            (r"org\.springframework\.beans\.factory\.NoSuchBeanDefinitionException", "spring_bean_not_found"),
            (r"org\.hibernate\.LazyInitializationException", "hibernate_lazy_init"),
            (r"javax\.persistence\.EntityNotFoundException", "jpa_entity_not_found"),
        ]
    
    def parse_syntax_error(self, error_message: str, source_code: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Parse Java syntax errors."""
        # Java syntax errors are typically compilation errors
        return self.parse_compilation_error(error_message, source_code)
    
    def parse_compilation_error(self, error_message: str, source_code: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Parse Java compilation errors."""
        for pattern, error_type in self.compilation_patterns:
            if re.search(pattern, error_message):
                return {
                    "error_type": error_type,
                    "pattern": pattern,
                    "category": ErrorCategory.COMPILATION,
                    "language": self.language
                }
        
        # Try to compile if source code is available
        if source_code:
            try:
                compilation_result = self._test_compile_java(source_code)
                if compilation_result:
                    return compilation_result
            except Exception as e:
                logger.debug(f"Error testing Java compilation: {e}")
        
        return None
    
    def detect_runtime_issues(self, error_context: ErrorContext) -> List[Dict[str, Any]]:
        """Detect Java runtime issues."""
        issues = []
        
        # Check runtime patterns
        for pattern, error_type in self.runtime_patterns:
            if re.search(pattern, error_context.error_message):
                issues.append({
                    "error_type": error_type,
                    "pattern": pattern,
                    "category": self._categorize_java_error(error_type),
                    "language": self.language
                })
        
        # Check framework patterns
        for pattern, error_type in self.framework_patterns:
            if re.search(pattern, error_context.error_message):
                issues.append({
                    "error_type": error_type,
                    "pattern": pattern,
                    "category": self._categorize_java_error(error_type),
                    "language": self.language,
                    "framework": self._detect_java_framework(pattern)
                })
        
        return issues
    
    def _test_compile_java(self, source_code: str) -> Optional[Dict[str, Any]]:
        """Test compile Java code to detect compilation errors."""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.java', delete=False) as f:
                f.write(source_code)
                f.flush()
                
                # Try to compile
                result = subprocess.run(
                    ['javac', f.name],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if result.returncode != 0:
                    return {
                        "error_type": "compilation_error",
                        "compiler_output": result.stderr,
                        "category": ErrorCategory.COMPILATION,
                        "language": self.language
                    }
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # javac not available or timed out
            pass
        except Exception as e:
            logger.debug(f"Error in Java compilation test: {e}")
        
        return None
    
    def _categorize_java_error(self, error_type: str) -> ErrorCategory:
        """Categorize Java errors."""
        category_map = {
            "null_pointer_exception": ErrorCategory.LOGIC,
            "index_out_of_bounds": ErrorCategory.LOGIC,
            "class_cast_exception": ErrorCategory.LOGIC,
            "illegal_argument": ErrorCategory.LOGIC,
            "number_format_exception": ErrorCategory.LOGIC,
            "out_of_memory": ErrorCategory.MEMORY,
            "stack_overflow": ErrorCategory.LOGIC,
            "file_not_found": ErrorCategory.FILESYSTEM,
            "connection_failed": ErrorCategory.NETWORK,
            "sql_exception": ErrorCategory.DATABASE,
            "spring_bean_not_found": ErrorCategory.CONFIGURATION,
            "hibernate_lazy_init": ErrorCategory.LOGIC,
            "jpa_entity_not_found": ErrorCategory.DATABASE,
        }
        return category_map.get(error_type, ErrorCategory.RUNTIME)
    
    def _detect_java_framework(self, pattern: str) -> Optional[str]:
        """Detect Java framework from error pattern."""
        if "springframework" in pattern:
            return "Spring"
        elif "hibernate" in pattern:
            return "Hibernate"
        elif "javax.persistence" in pattern:
            return "JPA"
        return None


class GoParser(LanguageSpecificParser):
    """Go-specific error parser."""
    
    def __init__(self):
        super().__init__(LanguageType.GO)
        
        # Go compilation error patterns
        self.compilation_patterns = [
            (r"syntax error: unexpected (.+)", "syntax_error"),
            (r"undefined: (.+)", "undefined_identifier"),
            (r"cannot use .* as .* value", "type_mismatch"),
            (r"not enough arguments in call to (.+)", "insufficient_arguments"),
            (r"too many arguments in call to (.+)", "too_many_arguments"),
            (r"package (.+) is not in GOROOT", "missing_package"),
        ]
        
        # Go runtime error patterns
        self.runtime_patterns = [
            (r"panic: runtime error: invalid memory address or nil pointer dereference", "nil_pointer_dereference"),
            (r"panic: runtime error: index out of range", "index_out_of_range"),
            (r"panic: runtime error: slice bounds out of range", "slice_bounds_error"),
            (r"panic: (.+)", "panic"),
            (r"fatal error: concurrent map read and map write", "concurrent_map_access"),
            (r"fatal error: all goroutines are asleep - deadlock", "deadlock"),
        ]
    
    def parse_syntax_error(self, error_message: str, source_code: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Parse Go syntax errors."""
        return self.parse_compilation_error(error_message, source_code)
    
    def parse_compilation_error(self, error_message: str, source_code: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Parse Go compilation errors."""
        for pattern, error_type in self.compilation_patterns:
            match = re.search(pattern, error_message)
            if match:
                return {
                    "error_type": error_type,
                    "pattern": pattern,
                    "match_groups": match.groups(),
                    "category": ErrorCategory.COMPILATION,
                    "language": self.language
                }
        
        # Try to compile if source code is available
        if source_code:
            try:
                compilation_result = self._test_compile_go(source_code)
                if compilation_result:
                    return compilation_result
            except Exception as e:
                logger.debug(f"Error testing Go compilation: {e}")
        
        return None
    
    def detect_runtime_issues(self, error_context: ErrorContext) -> List[Dict[str, Any]]:
        """Detect Go runtime issues."""
        issues = []
        
        for pattern, error_type in self.runtime_patterns:
            match = re.search(pattern, error_context.error_message)
            if match:
                issues.append({
                    "error_type": error_type,
                    "pattern": pattern,
                    "match_groups": match.groups() if match.groups() else None,
                    "category": self._categorize_go_error(error_type),
                    "language": self.language
                })
        
        return issues
    
    def _test_compile_go(self, source_code: str) -> Optional[Dict[str, Any]]:
        """Test compile Go code to detect compilation errors."""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.go', delete=False) as f:
                f.write(source_code)
                f.flush()
                
                # Try to compile
                result = subprocess.run(
                    ['go', 'build', f.name],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if result.returncode != 0:
                    return {
                        "error_type": "compilation_error",
                        "compiler_output": result.stderr,
                        "category": ErrorCategory.COMPILATION,
                        "language": self.language
                    }
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # go not available or timed out
            pass
        except Exception as e:
            logger.debug(f"Error in Go compilation test: {e}")
        
        return None
    
    def _categorize_go_error(self, error_type: str) -> ErrorCategory:
        """Categorize Go errors."""
        category_map = {
            "nil_pointer_dereference": ErrorCategory.LOGIC,
            "index_out_of_range": ErrorCategory.LOGIC,
            "slice_bounds_error": ErrorCategory.LOGIC,
            "panic": ErrorCategory.RUNTIME,
            "concurrent_map_access": ErrorCategory.CONCURRENCY,
            "deadlock": ErrorCategory.CONCURRENCY,
        }
        return category_map.get(error_type, ErrorCategory.RUNTIME)


class RustParser(LanguageSpecificParser):
    """Rust-specific error parser."""
    
    def __init__(self):
        super().__init__(LanguageType.RUST)
        
        # Rust compilation error patterns
        self.compilation_patterns = [
            (r"error\[E\d+\]: (.+)", "compilation_error"),
            (r"error: cannot find (.+) in this scope", "undefined_identifier"),
            (r"error: mismatched types", "type_mismatch"),
            (r"error: use of moved value", "use_after_move"),
            (r"error: borrow checker", "borrow_checker_error"),
            (r"error: lifetime mismatch", "lifetime_error"),
        ]
        
        # Rust runtime error patterns (panics)
        self.runtime_patterns = [
            (r"thread '(.+)' panicked at '(.+)'", "panic"),
            (r"index out of bounds", "index_out_of_bounds"),
            (r"attempt to divide by zero", "division_by_zero"),
            (r"attempt to subtract with overflow", "arithmetic_overflow"),
        ]
    
    def parse_syntax_error(self, error_message: str, source_code: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Parse Rust syntax errors."""
        return self.parse_compilation_error(error_message, source_code)
    
    def parse_compilation_error(self, error_message: str, source_code: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Parse Rust compilation errors."""
        for pattern, error_type in self.compilation_patterns:
            match = re.search(pattern, error_message)
            if match:
                return {
                    "error_type": error_type,
                    "pattern": pattern,
                    "match_groups": match.groups(),
                    "category": ErrorCategory.COMPILATION,
                    "language": self.language
                }
        return None
    
    def detect_runtime_issues(self, error_context: ErrorContext) -> List[Dict[str, Any]]:
        """Detect Rust runtime issues."""
        issues = []
        
        for pattern, error_type in self.runtime_patterns:
            match = re.search(pattern, error_context.error_message)
            if match:
                issues.append({
                    "error_type": error_type,
                    "pattern": pattern,
                    "match_groups": match.groups(),
                    "category": self._categorize_rust_error(error_type),
                    "language": self.language
                })
        
        return issues
    
    def _categorize_rust_error(self, error_type: str) -> ErrorCategory:
        """Categorize Rust errors."""
        category_map = {
            "panic": ErrorCategory.RUNTIME,
            "index_out_of_bounds": ErrorCategory.LOGIC,
            "division_by_zero": ErrorCategory.LOGIC,
            "arithmetic_overflow": ErrorCategory.LOGIC,
        }
        return category_map.get(error_type, ErrorCategory.RUNTIME)


class CSharpParser(LanguageSpecificParser):
    """C#-specific error parser."""
    
    def __init__(self):
        super().__init__(LanguageType.CSHARP)
        
        # C# compilation error patterns
        self.compilation_patterns = [
            (r"error CS\d+: (.+)", "compilation_error"),
            (r"error CS0103: The name '(.+)' does not exist", "undefined_identifier"),
            (r"error CS0246: The type or namespace name '(.+)' could not be found", "type_not_found"),
            (r"error CS1002: ; expected", "syntax_error"),
            (r"error CS1513: } expected", "missing_brace"),
        ]
        
        # C# runtime error patterns
        self.runtime_patterns = [
            (r"System\.NullReferenceException", "null_reference_exception"),
            (r"System\.IndexOutOfRangeException", "index_out_of_range"),
            (r"System\.ArgumentException", "argument_exception"),
            (r"System\.InvalidOperationException", "invalid_operation"),
            (r"System\.OutOfMemoryException", "out_of_memory"),
            (r"System\.StackOverflowException", "stack_overflow"),
            (r"System\.IO\.FileNotFoundException", "file_not_found"),
            (r"System\.Net\.WebException", "web_exception"),
        ]
    
    def parse_syntax_error(self, error_message: str, source_code: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Parse C# syntax errors."""
        return self.parse_compilation_error(error_message, source_code)
    
    def parse_compilation_error(self, error_message: str, source_code: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Parse C# compilation errors."""
        for pattern, error_type in self.compilation_patterns:
            match = re.search(pattern, error_message)
            if match:
                return {
                    "error_type": error_type,
                    "pattern": pattern,
                    "match_groups": match.groups(),
                    "category": ErrorCategory.COMPILATION,
                    "language": self.language
                }
        return None
    
    def detect_runtime_issues(self, error_context: ErrorContext) -> List[Dict[str, Any]]:
        """Detect C# runtime issues."""
        issues = []
        
        for pattern, error_type in self.runtime_patterns:
            if re.search(pattern, error_context.error_message):
                issues.append({
                    "error_type": error_type,
                    "pattern": pattern,
                    "category": self._categorize_csharp_error(error_type),
                    "language": self.language
                })
        
        return issues
    
    def _categorize_csharp_error(self, error_type: str) -> ErrorCategory:
        """Categorize C# errors."""
        category_map = {
            "null_reference_exception": ErrorCategory.LOGIC,
            "index_out_of_range": ErrorCategory.LOGIC,
            "argument_exception": ErrorCategory.LOGIC,
            "invalid_operation": ErrorCategory.LOGIC,
            "out_of_memory": ErrorCategory.MEMORY,
            "stack_overflow": ErrorCategory.LOGIC,
            "file_not_found": ErrorCategory.FILESYSTEM,
            "web_exception": ErrorCategory.NETWORK,
        }
        return category_map.get(error_type, ErrorCategory.RUNTIME)


class TypeScriptParser(LanguageSpecificParser):
    """TypeScript-specific error parser."""
    
    def __init__(self):
        super().__init__(LanguageType.TYPESCRIPT)
        
        # TypeScript compilation error patterns
        self.compilation_patterns = [
            (r"error TS\d+: (.+)", "typescript_error"),
            (r"Cannot find name '(.+)'", "undefined_identifier"),
            (r"Type '(.+)' is not assignable to type '(.+)'", "type_mismatch"),
            (r"Property '(.+)' does not exist on type '(.+)'", "property_not_found"),
            (r"Cannot find module '(.+)'", "module_not_found"),
            (r"Expected (.+) arguments, but got (.+)", "argument_count_mismatch"),
        ]
    
    def parse_syntax_error(self, error_message: str, source_code: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Parse TypeScript syntax errors."""
        return self.parse_compilation_error(error_message, source_code)
    
    def parse_compilation_error(self, error_message: str, source_code: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Parse TypeScript compilation errors."""
        for pattern, error_type in self.compilation_patterns:
            match = re.search(pattern, error_message)
            if match:
                return {
                    "error_type": error_type,
                    "pattern": pattern,
                    "match_groups": match.groups(),
                    "category": ErrorCategory.COMPILATION,
                    "language": self.language
                }
        return None
    
    def detect_runtime_issues(self, error_context: ErrorContext) -> List[Dict[str, Any]]:
        """TypeScript runtime errors are typically JavaScript runtime errors."""
        # Delegate to JavaScript parser for runtime issues
        from .comprehensive_error_detector import JavaScriptParser
        js_parser = JavaScriptParser()
        return js_parser.detect_runtime_issues(error_context)


class DartParser(LanguageSpecificParser):
    """Dart/Flutter-specific error parser."""
    
    def __init__(self):
        super().__init__(LanguageType.DART)
        
        # Dart compilation error patterns
        self.compilation_patterns = [
            (r"Error: (.+)", "dart_error"),
            (r"Expected (.+) but found (.+)", "syntax_error"),
            (r"Undefined name '(.+)'", "undefined_identifier"),
            (r"The (.+) '(.+)' isn't defined", "undefined_reference"),
            (r"(.+) can't be assigned to (.+)", "type_mismatch"),
            (r"Too many positional arguments", "argument_error"),
            (r"Missing required argument", "missing_argument"),
            (r"dart:(.+) library not found", "missing_library"),
        ]
        
        # Flutter-specific error patterns
        self.flutter_patterns = [
            (r"FlutterError: (.+)", "flutter_error"),
            (r"RenderFlex overflowed by (.+) pixels", "overflow_error"),
            (r"Widget (\w+) was given an infinite size", "infinite_size_error"),
            (r"setState\(\) called after dispose\(\)", "lifecycle_error"),
            (r"Navigator operation requested with a context that does not include a Navigator", "navigation_error"),
            (r"MediaQuery\.of\(\) called with a context that does not contain a MediaQuery", "media_query_error"),
            (r"Provider\.of<(.+)>\(\) called with a context that does not contain a (.+)", "provider_error"),
        ]
        
        # Dart runtime error patterns
        self.runtime_patterns = [
            (r"NoSuchMethodError: (.+)", "no_such_method"),
            (r"RangeError \(index\): Invalid value: (.+)", "range_error"),
            (r"FormatException: (.+)", "format_exception"),
            (r"StateError: (.+)", "state_error"),
            (r"ArgumentError: (.+)", "argument_error"),
            (r"UnimplementedError: (.+)", "unimplemented_error"),
        ]
    
    def parse_syntax_error(self, error_message: str, source_code: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Parse Dart syntax errors."""
        return self.parse_compilation_error(error_message, source_code)
    
    def parse_compilation_error(self, error_message: str, source_code: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Parse Dart compilation errors."""
        for pattern, error_type in self.compilation_patterns:
            match = re.search(pattern, error_message)
            if match:
                return {
                    "error_type": error_type,
                    "pattern": pattern,
                    "match_groups": match.groups(),
                    "category": ErrorCategory.COMPILATION,
                    "language": self.language,
                    "framework": "dart"
                }
        
        # Check for Flutter-specific errors
        for pattern, error_type in self.flutter_patterns:
            match = re.search(pattern, error_message)
            if match:
                return {
                    "error_type": error_type,
                    "pattern": pattern,
                    "match_groups": match.groups(),
                    "category": ErrorCategory.RUNTIME,
                    "language": self.language,
                    "framework": "flutter"
                }
        
        return None
    
    def detect_runtime_issues(self, error_context: ErrorContext) -> List[Dict[str, Any]]:
        """Detect Dart/Flutter runtime issues."""
        issues = []
        
        for pattern, error_type in self.runtime_patterns:
            match = re.search(pattern, error_context.error_message)
            if match:
                issues.append({
                    "error_type": error_type,
                    "pattern": pattern,
                    "match_groups": match.groups(),
                    "category": self._categorize_dart_error(error_type),
                    "language": self.language,
                    "framework": "dart"
                })
        
        # Check Flutter-specific runtime issues
        for pattern, error_type in self.flutter_patterns:
            match = re.search(pattern, error_context.error_message)
            if match:
                issues.append({
                    "error_type": error_type,
                    "pattern": pattern,
                    "match_groups": match.groups(),
                    "category": self._categorize_flutter_error(error_type),
                    "language": self.language,
                    "framework": "flutter"
                })
        
        return issues
    
    def _categorize_dart_error(self, error_type: str) -> ErrorCategory:
        """Categorize Dart errors."""
        category_map = {
            "no_such_method": ErrorCategory.LOGIC,
            "range_error": ErrorCategory.LOGIC,
            "format_exception": ErrorCategory.LOGIC,
            "state_error": ErrorCategory.LOGIC,
            "argument_error": ErrorCategory.LOGIC,
            "unimplemented_error": ErrorCategory.LOGIC,
        }
        return category_map.get(error_type, ErrorCategory.RUNTIME)
    
    def _categorize_flutter_error(self, error_type: str) -> ErrorCategory:
        """Categorize Flutter errors."""
        category_map = {
            "flutter_error": ErrorCategory.RUNTIME,
            "overflow_error": ErrorCategory.LOGIC,
            "infinite_size_error": ErrorCategory.LOGIC,
            "lifecycle_error": ErrorCategory.LOGIC,
            "navigation_error": ErrorCategory.CONFIGURATION,
            "media_query_error": ErrorCategory.CONFIGURATION,
            "provider_error": ErrorCategory.CONFIGURATION,
        }
        return category_map.get(error_type, ErrorCategory.RUNTIME)


class ReactNativeParser(LanguageSpecificParser):
    """React Native-specific error parser (extends JavaScript)."""
    
    def __init__(self):
        super().__init__(LanguageType.JAVASCRIPT)
        
        # React Native specific error patterns
        self.react_native_patterns = [
            (r"Element type is invalid: (.+)", "invalid_element"),
            (r"Cannot read property '(.+)' of undefined", "undefined_property"),
            (r"Cannot read property '(.+)' of null", "null_property"),
            (r"Native module (.+) is null", "native_module_null"),
            (r"Red Box: (.+)", "red_box_error"),
            (r"Yellow Box: (.+)", "yellow_box_warning"),
            (r"Metro bundler error: (.+)", "bundler_error"),
            (r"Unable to resolve module (.+)", "module_resolution_error"),
            (r"Task :(.+) FAILED", "build_task_failed"),
            (r"INSTALL_FAILED_(.+)", "install_failed"),
            (r"Bridge module (.+) not found", "bridge_module_missing"),
        ]
        
        # Platform-specific patterns
        self.ios_patterns = [
            (r"dyld: Library not loaded: (.+)", "ios_library_missing"),
            (r"NSInvalidArgumentException", "ios_invalid_argument"),
            (r"EXC_BAD_ACCESS", "ios_memory_error"),
            (r"Terminating app due to uncaught exception", "ios_uncaught_exception"),
        ]
        
        self.android_patterns = [
            (r"java\.lang\.RuntimeException: (.+)", "android_runtime_exception"),
            (r"android\.view\.InflateException", "android_layout_inflation"),
            (r"ClassNotFoundException: (.+)", "android_class_not_found"),
            (r"ActivityNotFoundException", "android_activity_not_found"),
            (r"SecurityException: (.+)", "android_security_exception"),
        ]
    
    def parse_syntax_error(self, error_message: str, source_code: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Parse React Native JavaScript syntax errors."""
        # Delegate to JavaScript parser for syntax errors
        from .comprehensive_error_detector import JavaScriptParser
        js_parser = JavaScriptParser()
        result = js_parser.parse_syntax_error(error_message, source_code)
        if result:
            result["framework"] = "react_native"
        return result
    
    def parse_compilation_error(self, error_message: str, source_code: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Parse React Native bundler and build errors."""
        # Check for Metro bundler errors
        if "Metro bundler" in error_message or "metro" in error_message.lower():
            return {
                "error_type": "bundler_error",
                "category": ErrorCategory.COMPILATION,
                "language": self.language,
                "framework": "react_native"
            }
        
        # Check for native build errors
        for pattern, error_type in [("Task :", "build_task_failed"), ("INSTALL_FAILED", "install_failed")]:
            if pattern in error_message:
                return {
                    "error_type": error_type,
                    "category": ErrorCategory.COMPILATION,
                    "language": self.language,
                    "framework": "react_native"
                }
        
        return None
    
    def detect_runtime_issues(self, error_context: ErrorContext) -> List[Dict[str, Any]]:
        """Detect React Native runtime issues."""
        issues = []
        
        # Check React Native specific patterns
        for pattern, error_type in self.react_native_patterns:
            match = re.search(pattern, error_context.error_message)
            if match:
                issues.append({
                    "error_type": error_type,
                    "pattern": pattern,
                    "match_groups": match.groups(),
                    "category": self._categorize_rn_error(error_type),
                    "language": self.language,
                    "framework": "react_native"
                })
        
        # Check platform-specific patterns
        for pattern, error_type in self.ios_patterns:
            if re.search(pattern, error_context.error_message):
                issues.append({
                    "error_type": error_type,
                    "pattern": pattern,
                    "category": ErrorCategory.RUNTIME,
                    "language": self.language,
                    "framework": "react_native",
                    "platform": "ios"
                })
        
        for pattern, error_type in self.android_patterns:
            if re.search(pattern, error_context.error_message):
                issues.append({
                    "error_type": error_type,
                    "pattern": pattern,
                    "category": ErrorCategory.RUNTIME,
                    "language": self.language,
                    "framework": "react_native",
                    "platform": "android"
                })
        
        return issues
    
    def _categorize_rn_error(self, error_type: str) -> ErrorCategory:
        """Categorize React Native errors."""
        category_map = {
            "invalid_element": ErrorCategory.LOGIC,
            "undefined_property": ErrorCategory.LOGIC,
            "null_property": ErrorCategory.LOGIC,
            "native_module_null": ErrorCategory.CONFIGURATION,
            "red_box_error": ErrorCategory.RUNTIME,
            "yellow_box_warning": ErrorCategory.RUNTIME,
            "bundler_error": ErrorCategory.COMPILATION,
            "module_resolution_error": ErrorCategory.DEPENDENCY,
            "build_task_failed": ErrorCategory.COMPILATION,
            "install_failed": ErrorCategory.ENVIRONMENT,
            "bridge_module_missing": ErrorCategory.CONFIGURATION,
        }
        return category_map.get(error_type, ErrorCategory.RUNTIME)


class XamarinParser(LanguageSpecificParser):
    """Xamarin-specific error parser (extends C#)."""
    
    def __init__(self):
        super().__init__(LanguageType.CSHARP)
        
        # Xamarin compilation error patterns
        self.xamarin_compilation_patterns = [
            (r"XA\d+: (.+)", "xamarin_android_error"),
            (r"XI\d+: (.+)", "xamarin_ios_error"),
            (r"error: (.+) requires a reference to (.+)", "missing_reference"),
            (r"The type or namespace name '(.+)' does not exist in the namespace '(.+)'", "namespace_error"),
            (r"No resource found that matches the given name: '(.+)'", "resource_not_found"),
        ]
        
        # Xamarin runtime error patterns
        self.xamarin_runtime_patterns = [
            (r"Java\.Lang\.RuntimeException: (.+)", "xamarin_java_exception"),
            (r"Android\.Views\.InflateException", "xamarin_layout_inflation"),
            (r"System\.NotSupportedException: (.+)", "xamarin_not_supported"),
            (r"Xamarin\.Forms\.(.+)Exception: (.+)", "xamarin_forms_exception"),
            (r"Foundation\.MonoTouchException", "xamarin_ios_exception"),
        ]
        
        # Platform-specific patterns
        self.android_xamarin_patterns = [
            (r"at Android\.(.+)", "android_specific"),
            (r"at Java\.(.+)", "java_interop"),
            (r"mono-rt: (.+)", "mono_runtime"),
        ]
        
        self.ios_xamarin_patterns = [
            (r"at Foundation\.(.+)", "foundation_framework"),
            (r"at UIKit\.(.+)", "uikit_framework"),
            (r"at CoreFoundation\.(.+)", "core_foundation"),
        ]
    
    def parse_syntax_error(self, error_message: str, source_code: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Parse Xamarin C# syntax errors."""
        # Delegate to C# parser for syntax errors
        csharp_parser = CSharpParser()
        result = csharp_parser.parse_syntax_error(error_message, source_code)
        if result:
            result["framework"] = "xamarin"
        return result
    
    def parse_compilation_error(self, error_message: str, source_code: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Parse Xamarin compilation errors."""
        for pattern, error_type in self.xamarin_compilation_patterns:
            match = re.search(pattern, error_message)
            if match:
                return {
                    "error_type": error_type,
                    "pattern": pattern,
                    "match_groups": match.groups(),
                    "category": ErrorCategory.COMPILATION,
                    "language": self.language,
                    "framework": "xamarin"
                }
        
        return None
    
    def detect_runtime_issues(self, error_context: ErrorContext) -> List[Dict[str, Any]]:
        """Detect Xamarin runtime issues."""
        issues = []
        
        # Check Xamarin specific patterns
        for pattern, error_type in self.xamarin_runtime_patterns:
            match = re.search(pattern, error_context.error_message)
            if match:
                issues.append({
                    "error_type": error_type,
                    "pattern": pattern,
                    "match_groups": match.groups(),
                    "category": self._categorize_xamarin_error(error_type),
                    "language": self.language,
                    "framework": "xamarin"
                })
        
        # Check platform-specific patterns
        for pattern, error_type in self.android_xamarin_patterns:
            if re.search(pattern, error_context.error_message):
                issues.append({
                    "error_type": error_type,
                    "pattern": pattern,
                    "category": ErrorCategory.RUNTIME,
                    "language": self.language,
                    "framework": "xamarin",
                    "platform": "android"
                })
        
        for pattern, error_type in self.ios_xamarin_patterns:
            if re.search(pattern, error_context.error_message):
                issues.append({
                    "error_type": error_type,
                    "pattern": pattern,
                    "category": ErrorCategory.RUNTIME,
                    "language": self.language,
                    "framework": "xamarin",
                    "platform": "ios"
                })
        
        return issues
    
    def _categorize_xamarin_error(self, error_type: str) -> ErrorCategory:
        """Categorize Xamarin errors."""
        category_map = {
            "xamarin_java_exception": ErrorCategory.RUNTIME,
            "xamarin_layout_inflation": ErrorCategory.CONFIGURATION,
            "xamarin_not_supported": ErrorCategory.LOGIC,
            "xamarin_forms_exception": ErrorCategory.RUNTIME,
            "xamarin_ios_exception": ErrorCategory.RUNTIME,
        }
        return category_map.get(error_type, ErrorCategory.RUNTIME)


class UnityParser(LanguageSpecificParser):
    """Unity game engine error parser (C# with Unity-specific extensions)."""
    
    def __init__(self):
        super().__init__(LanguageType.CSHARP)
        
        # Unity compilation error patterns
        self.unity_compilation_patterns = [
            (r"Assets/(.+)\.cs\(\d+,\d+\): error CS\d+: (.+)", "unity_cs_error"),
            (r"The name '(.+)' does not exist in the current context", "unity_undefined_name"),
            (r"Cannot implicitly convert type '(.+)' to '(.+)'", "unity_type_conversion"),
            (r"'(.+)' does not contain a definition for '(.+)'", "unity_missing_member"),
            (r"Assembly '(.+)' not found", "unity_missing_assembly"),
        ]
        
        # Unity runtime error patterns
        self.unity_runtime_patterns = [
            (r"NullReferenceException: Object reference not set to an instance of an object", "unity_null_reference"),
            (r"MissingReferenceException: The object of type '(.+)' has been destroyed", "unity_missing_reference"),
            (r"ArgumentException: (.+)", "unity_argument_exception"),
            (r"InvalidOperationException: (.+)", "unity_invalid_operation"),
            (r"UnityException: (.+)", "unity_exception"),
            (r"ArgumentNullException: (.+)", "unity_argument_null"),
        ]
        
        # Unity-specific patterns
        self.unity_specific_patterns = [
            (r"Transform child out of bounds", "unity_transform_bounds"),
            (r"Camera component is required", "unity_missing_component"),
            (r"GameObject with tag '(.+)' not found", "unity_tag_not_found"),
            (r"Animation clip '(.+)' not found", "unity_animation_missing"),
            (r"Shader '(.+)' not found", "unity_shader_missing"),
            (r"Texture '(.+)' not found", "unity_texture_missing"),
            (r"Prefab '(.+)' not found", "unity_prefab_missing"),
            (r"Scene '(.+)' not found", "unity_scene_missing"),
        ]
    
    def parse_syntax_error(self, error_message: str, source_code: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Parse Unity C# syntax errors."""
        # Check for Unity-specific compilation errors first
        result = self.parse_compilation_error(error_message, source_code)
        if result:
            return result
        
        # Delegate to C# parser for general syntax errors
        csharp_parser = CSharpParser()
        result = csharp_parser.parse_syntax_error(error_message, source_code)
        if result:
            result["framework"] = "unity"
        return result
    
    def parse_compilation_error(self, error_message: str, source_code: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Parse Unity compilation errors."""
        for pattern, error_type in self.unity_compilation_patterns:
            match = re.search(pattern, error_message)
            if match:
                return {
                    "error_type": error_type,
                    "pattern": pattern,
                    "match_groups": match.groups(),
                    "category": ErrorCategory.COMPILATION,
                    "language": self.language,
                    "framework": "unity"
                }
        
        return None
    
    def detect_runtime_issues(self, error_context: ErrorContext) -> List[Dict[str, Any]]:
        """Detect Unity runtime issues."""
        issues = []
        
        # Check Unity runtime patterns
        for pattern, error_type in self.unity_runtime_patterns:
            match = re.search(pattern, error_context.error_message)
            if match:
                issues.append({
                    "error_type": error_type,
                    "pattern": pattern,
                    "match_groups": match.groups(),
                    "category": self._categorize_unity_error(error_type),
                    "language": self.language,
                    "framework": "unity"
                })
        
        # Check Unity-specific patterns
        for pattern, error_type in self.unity_specific_patterns:
            match = re.search(pattern, error_context.error_message)
            if match:
                issues.append({
                    "error_type": error_type,
                    "pattern": pattern,
                    "match_groups": match.groups(),
                    "category": self._categorize_unity_specific_error(error_type),
                    "language": self.language,
                    "framework": "unity"
                })
        
        return issues
    
    def _categorize_unity_error(self, error_type: str) -> ErrorCategory:
        """Categorize Unity runtime errors."""
        category_map = {
            "unity_null_reference": ErrorCategory.LOGIC,
            "unity_missing_reference": ErrorCategory.LOGIC,
            "unity_argument_exception": ErrorCategory.LOGIC,
            "unity_invalid_operation": ErrorCategory.LOGIC,
            "unity_exception": ErrorCategory.RUNTIME,
            "unity_argument_null": ErrorCategory.LOGIC,
        }
        return category_map.get(error_type, ErrorCategory.RUNTIME)
    
    def _categorize_unity_specific_error(self, error_type: str) -> ErrorCategory:
        """Categorize Unity-specific errors."""
        category_map = {
            "unity_transform_bounds": ErrorCategory.LOGIC,
            "unity_missing_component": ErrorCategory.CONFIGURATION,
            "unity_tag_not_found": ErrorCategory.CONFIGURATION,
            "unity_animation_missing": ErrorCategory.CONFIGURATION,
            "unity_shader_missing": ErrorCategory.CONFIGURATION,
            "unity_texture_missing": ErrorCategory.CONFIGURATION,
            "unity_prefab_missing": ErrorCategory.CONFIGURATION,
            "unity_scene_missing": ErrorCategory.CONFIGURATION,
        }
        return category_map.get(error_type, ErrorCategory.CONFIGURATION)


class CompilerIntegration:
    """
    Integration with various compilers and build systems for detailed diagnostics.
    """
    
    def __init__(self):
        """Initialize compiler integration."""
        self.available_compilers = self._detect_available_compilers()
    
    def _detect_available_compilers(self) -> Dict[str, bool]:
        """Detect which compilers are available on the system."""
        compilers = {
            'javac': False,
            'go': False,
            'rustc': False,
            'dotnet': False,
            'tsc': False,
            'gcc': False,
            'clang': False,
            'dart': False,
            'flutter': False,
            'msbuild': False,  # For Xamarin
            'unity': False,    # Unity Editor
        }
        
        for compiler in compilers.keys():
            try:
                result = subprocess.run(
                    [compiler, '--version'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                compilers[compiler] = (result.returncode == 0)
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
        
        return compilers
    
    def get_detailed_diagnostics(self, source_code: str, language: LanguageType, 
                                file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Get detailed diagnostics from language compilers/checkers.
        
        Args:
            source_code: Source code to analyze
            language: Programming language
            file_path: Optional file path for context
            
        Returns:
            Detailed diagnostic information
        """
        diagnostics = {
            "language": language.value,
            "compiler_available": False,
            "syntax_valid": None,
            "compilation_errors": [],
            "warnings": [],
            "suggestions": []
        }
        
        try:
            if language == LanguageType.JAVA and self.available_compilers.get('javac'):
                diagnostics.update(self._analyze_java_code(source_code))
            elif language == LanguageType.GO and self.available_compilers.get('go'):
                diagnostics.update(self._analyze_go_code(source_code))
            elif language == LanguageType.RUST and self.available_compilers.get('rustc'):
                diagnostics.update(self._analyze_rust_code(source_code))
            elif language == LanguageType.CSHARP and self.available_compilers.get('dotnet'):
                diagnostics.update(self._analyze_csharp_code(source_code))
            elif language == LanguageType.TYPESCRIPT and self.available_compilers.get('tsc'):
                diagnostics.update(self._analyze_typescript_code(source_code))
            elif language == LanguageType.DART and (self.available_compilers.get('dart') or self.available_compilers.get('flutter')):
                diagnostics.update(self._analyze_dart_code(source_code))
            elif language == LanguageType.PYTHON:
                diagnostics.update(self._analyze_python_code(source_code))
            
        except Exception as e:
            diagnostics["error"] = str(e)
            logger.error(f"Error in compiler integration: {e}")
        
        return diagnostics
    
    def _analyze_java_code(self, source_code: str) -> Dict[str, Any]:
        """Analyze Java code using javac."""
        result = {
            "compiler_available": True,
            "compilation_errors": [],
            "warnings": []
        }
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.java', delete=False) as f:
                f.write(source_code)
                f.flush()
                
                # Compile with verbose output
                compile_result = subprocess.run(
                    ['javac', '-Xlint:all', f.name],
                    capture_output=True,
                    text=True,
                    timeout=15
                )
                
                result["syntax_valid"] = (compile_result.returncode == 0)
                
                if compile_result.stderr:
                    # Parse compiler output
                    lines = compile_result.stderr.strip().split('\n')
                    for line in lines:
                        if 'error:' in line.lower():
                            result["compilation_errors"].append(line.strip())
                        elif 'warning:' in line.lower():
                            result["warnings"].append(line.strip())
                
                # Clean up
                Path(f.name).unlink(missing_ok=True)
                
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def _analyze_go_code(self, source_code: str) -> Dict[str, Any]:
        """Analyze Go code using go build."""
        result = {
            "compiler_available": True,
            "compilation_errors": [],
            "warnings": []
        }
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.go', delete=False) as f:
                # Add basic package declaration if missing
                if 'package ' not in source_code:
                    f.write('package main\n\n')
                f.write(source_code)
                f.flush()
                
                # Try to build
                build_result = subprocess.run(
                    ['go', 'build', f.name],
                    capture_output=True,
                    text=True,
                    timeout=15
                )
                
                result["syntax_valid"] = (build_result.returncode == 0)
                
                if build_result.stderr:
                    result["compilation_errors"] = build_result.stderr.strip().split('\n')
                
                # Clean up
                Path(f.name).unlink(missing_ok=True)
                
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def _analyze_rust_code(self, source_code: str) -> Dict[str, Any]:
        """Analyze Rust code using rustc."""
        result = {
            "compiler_available": True,
            "compilation_errors": [],
            "warnings": []
        }
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.rs', delete=False) as f:
                f.write(source_code)
                f.flush()
                
                # Try to compile
                compile_result = subprocess.run(
                    ['rustc', '--error-format=human', f.name],
                    capture_output=True,
                    text=True,
                    timeout=15
                )
                
                result["syntax_valid"] = (compile_result.returncode == 0)
                
                if compile_result.stderr:
                    lines = compile_result.stderr.strip().split('\n')
                    for line in lines:
                        if 'error:' in line.lower():
                            result["compilation_errors"].append(line.strip())
                        elif 'warning:' in line.lower():
                            result["warnings"].append(line.strip())
                
                # Clean up
                Path(f.name).unlink(missing_ok=True)
                
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def _analyze_csharp_code(self, source_code: str) -> Dict[str, Any]:
        """Analyze C# code using dotnet build."""
        result = {
            "compiler_available": True,
            "compilation_errors": [],
            "warnings": []
        }
        
        # C# analysis would require a full project structure
        # For now, return a placeholder
        result["note"] = "C# analysis requires full project context"
        
        return result
    
    def _analyze_typescript_code(self, source_code: str) -> Dict[str, Any]:
        """Analyze TypeScript code using tsc."""
        result = {
            "compiler_available": True,
            "compilation_errors": [],
            "warnings": []
        }
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.ts', delete=False) as f:
                f.write(source_code)
                f.flush()
                
                # Try to compile
                compile_result = subprocess.run(
                    ['tsc', '--noEmit', f.name],
                    capture_output=True,
                    text=True,
                    timeout=15
                )
                
                result["syntax_valid"] = (compile_result.returncode == 0)
                
                if compile_result.stdout:
                    result["compilation_errors"] = compile_result.stdout.strip().split('\n')
                
                # Clean up
                Path(f.name).unlink(missing_ok=True)
                
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def _analyze_python_code(self, source_code: str) -> Dict[str, Any]:
        """Analyze Python code using ast module."""
        result = {
            "compiler_available": True,
            "compilation_errors": [],
            "warnings": []
        }
        
        try:
            import ast
            ast.parse(source_code)
            result["syntax_valid"] = True
        except SyntaxError as e:
            result["syntax_valid"] = False
            result["compilation_errors"] = [f"SyntaxError: {e.msg} at line {e.lineno}"]
        except Exception as e:
            result["compilation_errors"] = [f"Error: {str(e)}"]
        
        return result
    
    def _analyze_dart_code(self, source_code: str) -> Dict[str, Any]:
        """Analyze Dart/Flutter code using dart analyze."""
        result = {
            "compiler_available": True,
            "compilation_errors": [],
            "warnings": []
        }
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.dart', delete=False) as f:
                f.write(source_code)
                f.flush()
                
                # Try dart analyze first
                analyze_result = subprocess.run(
                    ['dart', 'analyze', f.name],
                    capture_output=True,
                    text=True,
                    timeout=15
                )
                
                result["syntax_valid"] = (analyze_result.returncode == 0)
                
                if analyze_result.stdout:
                    lines = analyze_result.stdout.strip().split('\n')
                    for line in lines:
                        if 'error:' in line.lower():
                            result["compilation_errors"].append(line.strip())
                        elif 'warning:' in line.lower():
                            result["warnings"].append(line.strip())
                
                # If dart analyze not available, try flutter analyze
                if not result["syntax_valid"] and self.available_compilers.get('flutter'):
                    flutter_result = subprocess.run(
                        ['flutter', 'analyze', f.name],
                        capture_output=True,
                        text=True,
                        timeout=15
                    )
                    
                    if flutter_result.stdout:
                        result["compilation_errors"].extend(flutter_result.stdout.strip().split('\n'))
                
                # Clean up
                Path(f.name).unlink(missing_ok=True)
                
        except Exception as e:
            result["error"] = str(e)
        
        return result


# Factory function to create language parsers
def create_language_parser(language: LanguageType) -> Optional[LanguageSpecificParser]:
    """
    Create a language-specific parser.
    
    Args:
        language: Programming language type
        
    Returns:
        Language parser instance or None if not supported
    """
    parser_map = {
        LanguageType.PYTHON: lambda: __import__('comprehensive_error_detector', fromlist=['PythonParser']).PythonParser(),
        LanguageType.JAVASCRIPT: lambda: __import__('comprehensive_error_detector', fromlist=['JavaScriptParser']).JavaScriptParser(),
        LanguageType.JAVA: JavaParser,
        LanguageType.GO: GoParser,
        LanguageType.RUST: RustParser,
        LanguageType.CSHARP: CSharpParser,
        LanguageType.TYPESCRIPT: TypeScriptParser,
        LanguageType.DART: DartParser,
    }
    
    parser_factory = parser_map.get(language)
    if parser_factory:
        try:
            return parser_factory()
        except Exception as e:
            logger.error(f"Error creating parser for {language.value}: {e}")
    
    return None


if __name__ == "__main__":
    # Test the language parsers
    print("Language Parsers Test")
    print("====================")
    
    # Test Java parser
    java_parser = JavaParser()
    java_error = "error: cannot find symbol\n  symbol:   variable undefinedVar"
    java_result = java_parser.parse_compilation_error(java_error)
    print(f"\nJava Compilation Error:")
    print(f"Result: {java_result}")
    
    # Test Go parser
    go_parser = GoParser()
    go_error = "panic: runtime error: invalid memory address or nil pointer dereference"
    go_context = ErrorContext(error_message=go_error, language=LanguageType.GO)
    go_issues = go_parser.detect_runtime_issues(go_context)
    print(f"\nGo Runtime Issues:")
    print(f"Issues: {go_issues}")
    
    # Test Rust parser
    rust_parser = RustParser()
    rust_error = "error[E0425]: cannot find value `undefined_var` in this scope"
    rust_result = rust_parser.parse_compilation_error(rust_error)
    print(f"\nRust Compilation Error:")
    print(f"Result: {rust_result}")
    
    # Test Dart parser
    dart_parser = DartParser()
    dart_error = "FlutterError: RenderFlex overflowed by 15 pixels on the right"
    dart_context = ErrorContext(error_message=dart_error, language=LanguageType.DART)
    dart_issues = dart_parser.detect_runtime_issues(dart_context)
    print(f"\nDart/Flutter Runtime Issues:")
    print(f"Issues: {dart_issues}")
    
    # Test React Native parser  
    rn_parser = ReactNativeParser()
    rn_error = "Element type is invalid: expected a string but received undefined"
    rn_context = ErrorContext(error_message=rn_error, language=LanguageType.JAVASCRIPT)
    rn_issues = rn_parser.detect_runtime_issues(rn_context)
    print(f"\nReact Native Runtime Issues:")
    print(f"Issues: {rn_issues}")
    
    # Test Xamarin parser
    xamarin_parser = XamarinParser()
    xamarin_error = "XA0001: Java.Lang.RuntimeException: Unable to start activity"
    xamarin_result = xamarin_parser.parse_compilation_error(xamarin_error)
    print(f"\nXamarin Compilation Error:")
    print(f"Result: {xamarin_result}")
    
    # Test Unity parser
    unity_parser = UnityParser()
    unity_error = "NullReferenceException: Object reference not set to an instance of an object"
    unity_context = ErrorContext(error_message=unity_error, language=LanguageType.CSHARP)
    unity_issues = unity_parser.detect_runtime_issues(unity_context)
    print(f"\nUnity Runtime Issues:")
    print(f"Issues: {unity_issues}")
    
    # Test compiler integration
    print(f"\nCompiler Integration:")
    integration = CompilerIntegration()
    print(f"Available compilers: {integration.available_compilers}")
    
    # Test Python code analysis
    python_code = "def test():\n    print(undefined_variable)"
    python_diagnostics = integration.get_detailed_diagnostics(python_code, LanguageType.PYTHON)
    print(f"\nPython Code Diagnostics:")
    print(f"Syntax valid: {python_diagnostics.get('syntax_valid')}")
    print(f"Errors: {python_diagnostics.get('compilation_errors', [])}")
    
    # Test Dart code analysis if dart is available
    if integration.available_compilers.get('dart'):
        dart_code = "void main() { print(undefinedVariable); }"
        dart_diagnostics = integration.get_detailed_diagnostics(dart_code, LanguageType.DART)
        print(f"\nDart Code Diagnostics:")
        print(f"Syntax valid: {dart_diagnostics.get('syntax_valid')}")
        print(f"Errors: {dart_diagnostics.get('compilation_errors', [])}")