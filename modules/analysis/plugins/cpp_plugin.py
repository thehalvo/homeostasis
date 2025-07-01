"""
C/C++ Plugin for Homeostasis

This plugin enables Homeostasis to analyze and fix errors in C and C++ applications.
It provides comprehensive error handling for compilation errors, runtime issues, memory
management problems, and framework-specific issues.
"""
import logging
import re
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union, Set

from ..language_plugin_system import LanguagePlugin, register_plugin
from ..comprehensive_error_detector import ErrorCategory, ErrorSeverity, LanguageType

logger = logging.getLogger(__name__)


class CPPExceptionHandler:
    """
    Handles C/C++-specific exceptions with comprehensive error detection and classification.
    
    This class provides logic for categorizing C/C++ compilation errors, runtime issues,
    memory problems, and framework-specific challenges.
    """
    
    def __init__(self):
        """Initialize the C/C++ exception handler."""
        self.rule_categories = {
            "compilation": "C/C++ compilation errors and syntax issues",
            "linking": "Linker errors and library resolution issues",
            "runtime": "Runtime exceptions and crashes",
            "memory": "Memory management and segmentation fault errors",
            "threading": "Threading and concurrency errors",
            "stl": "Standard Template Library usage errors",
            "cmake": "CMake build system configuration errors",
            "makefile": "Makefile build system errors",
            "preprocessor": "Preprocessor macro and include errors",
            "templates": "C++ template instantiation errors",
            "syntax": "C/C++ syntax and semantic errors",
            "warnings": "Compiler warnings that may indicate issues",
            "undefined_behavior": "Undefined behavior and platform issues",
            "optimization": "Compiler optimization related issues"
        }
        
        # Load rules from different categories
        self.rules = self._load_rules()
        
        # Pre-compile regex patterns for better performance
        self._compile_patterns()
    
    def _load_rules(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load C/C++ error rules from rule files."""
        rules = {}
        rules_dir = Path(__file__).parent.parent / "rules" / "cpp"
        
        try:
            # Create rules directory if it doesn't exist
            rules_dir.mkdir(parents=True, exist_ok=True)
            
            # Load common C/C++ rules
            common_rules_path = rules_dir / "cpp_common_errors.json"
            if common_rules_path.exists():
                with open(common_rules_path, 'r') as f:
                    common_data = json.load(f)
                    rules["common"] = common_data.get("rules", [])
                    logger.info(f"Loaded {len(rules['common'])} common C/C++ rules")
            else:
                rules["common"] = self._create_default_rules()
                self._save_default_rules(common_rules_path, rules["common"])
            
            # Load compilation-specific rules
            compilation_rules_path = rules_dir / "cpp_compilation_errors.json"
            if compilation_rules_path.exists():
                with open(compilation_rules_path, 'r') as f:
                    compilation_data = json.load(f)
                    rules["compilation"] = compilation_data.get("rules", [])
                    logger.info(f"Loaded {len(rules['compilation'])} C/C++ compilation rules")
            else:
                rules["compilation"] = []
            
            # Load memory-specific rules
            memory_rules_path = rules_dir / "cpp_memory_errors.json"
            if memory_rules_path.exists():
                with open(memory_rules_path, 'r') as f:
                    memory_data = json.load(f)
                    rules["memory"] = memory_data.get("rules", [])
                    logger.info(f"Loaded {len(rules['memory'])} C/C++ memory rules")
            else:
                rules["memory"] = []
                    
        except Exception as e:
            logger.error(f"Error loading C/C++ rules: {e}")
            rules = {"common": self._create_default_rules(), "compilation": [], "memory": []}
        
        return rules
    
    def _create_default_rules(self) -> List[Dict[str, Any]]:
        """Create default C/C++ error detection rules."""
        return [
            {
                "id": "cpp_segfault",
                "pattern": r"segmentation fault|SIGSEGV|signal 11",
                "category": "memory",
                "severity": "critical",
                "description": "Segmentation fault - invalid memory access",
                "fix_suggestions": [
                    "Check for null pointer dereferences",
                    "Verify array bounds access",
                    "Ensure proper memory allocation/deallocation",
                    "Use debugging tools like valgrind or AddressSanitizer"
                ]
            },
            {
                "id": "cpp_undefined_symbol",
                "pattern": r"undefined reference to|undefined symbol",
                "category": "linking",
                "severity": "high",
                "description": "Undefined symbol error during linking",
                "fix_suggestions": [
                    "Check if all required libraries are linked",
                    "Verify function declarations match implementations",
                    "Ensure proper include paths are set",
                    "Check for missing object files in build"
                ]
            },
            {
                "id": "cpp_syntax_error",
                "pattern": r"error: expected|syntax error|parse error",
                "category": "compilation",
                "severity": "high",
                "description": "C/C++ syntax error",
                "fix_suggestions": [
                    "Check for missing semicolons or braces",
                    "Verify proper variable declarations",
                    "Ensure correct function syntax",
                    "Check for proper template syntax"
                ]
            },
            {
                "id": "cpp_include_error",
                "pattern": r"No such file or directory|cannot find|file not found",
                "category": "preprocessor",
                "severity": "high",
                "description": "Header file include error",
                "fix_suggestions": [
                    "Check include paths in build configuration",
                    "Verify header file exists and is accessible",
                    "Use proper relative/absolute paths",
                    "Install missing development packages"
                ]
            },
            {
                "id": "cpp_memory_leak",
                "pattern": r"memory leak|heap-use-after-free|double-free",
                "category": "memory",
                "severity": "high",
                "description": "Memory management error",
                "fix_suggestions": [
                    "Match every new with delete",
                    "Match every malloc with free",
                    "Use smart pointers (unique_ptr, shared_ptr)",
                    "Use RAII principles for resource management"
                ]
            },
            {
                "id": "cpp_template_error",
                "pattern": r"template instantiation|no matching function|candidate template",
                "category": "templates",
                "severity": "medium",
                "description": "C++ template instantiation error",
                "fix_suggestions": [
                    "Check template parameter constraints",
                    "Verify template specializations",
                    "Ensure proper template argument types",
                    "Check for SFINAE issues"
                ]
            }
        ]
    
    def _save_default_rules(self, rules_path: Path, rules: List[Dict[str, Any]]):
        """Save default rules to JSON file."""
        try:
            rules_data = {
                "version": "1.0",
                "description": "Default C/C++ error detection rules",
                "rules": rules
            }
            with open(rules_path, 'w') as f:
                json.dump(rules_data, f, indent=2)
            logger.info(f"Saved {len(rules)} default C/C++ rules to {rules_path}")
        except Exception as e:
            logger.error(f"Error saving default C/C++ rules: {e}")
    
    def _compile_patterns(self):
        """Pre-compile regex patterns for better performance."""
        self.compiled_patterns = {}
        for category, rule_list in self.rules.items():
            self.compiled_patterns[category] = []
            for rule in rule_list:
                try:
                    compiled_pattern = re.compile(rule["pattern"], re.IGNORECASE | re.MULTILINE)
                    self.compiled_patterns[category].append((compiled_pattern, rule))
                except re.error as e:
                    logger.warning(f"Invalid regex pattern in rule {rule.get('id', 'unknown')}: {e}")
    
    def analyze_error(self, error_message: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyze a C/C++ error message and provide categorization and suggestions.
        
        Args:
            error_message: The error message to analyze
            context: Additional context information
            
        Returns:
            Analysis results with error type, category, and suggestions
        """
        if context is None:
            context = {}
        
        results = {
            "language": "cpp",
            "error_message": error_message,
            "matches": [],
            "primary_category": "unknown",
            "severity": "medium",
            "fix_suggestions": [],
            "compiler_info": context.get("compiler_info", {}),
            "build_system": self._detect_build_system(context),
            "additional_context": {}
        }
        
        # Check each category of rules
        for category, pattern_list in self.compiled_patterns.items():
            for compiled_pattern, rule in pattern_list:
                match = compiled_pattern.search(error_message)
                if match:
                    match_info = {
                        "rule_id": rule["id"],
                        "category": rule["category"],
                        "severity": rule["severity"],
                        "description": rule["description"],
                        "fix_suggestions": rule["fix_suggestions"],
                        "matched_text": match.group(0),
                        "match_groups": match.groups()
                    }
                    results["matches"].append(match_info)
        
        # Determine primary category and severity
        if results["matches"]:
            # Sort by severity priority
            severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
            sorted_matches = sorted(results["matches"], 
                                  key=lambda x: severity_order.get(x["severity"], 4))
            
            primary_match = sorted_matches[0]
            results["primary_category"] = primary_match["category"]
            results["severity"] = primary_match["severity"]
            
            # Collect all fix suggestions
            all_suggestions = []
            for match in results["matches"]:
                all_suggestions.extend(match["fix_suggestions"])
            results["fix_suggestions"] = list(set(all_suggestions))  # Remove duplicates
        
        # Add compiler-specific analysis
        results["additional_context"] = self._analyze_compiler_context(error_message, context)
        
        return results
    
    def _detect_build_system(self, context: Dict[str, Any]) -> str:
        """Detect the build system being used."""
        file_path = context.get("file_path", "")
        
        if "CMakeLists.txt" in file_path or context.get("cmake_detected"):
            return "cmake"
        elif "Makefile" in file_path or context.get("make_detected"):
            return "make"
        elif "build.ninja" in file_path or context.get("ninja_detected"):
            return "ninja"
        elif ".vcxproj" in file_path or context.get("msbuild_detected"):
            return "msbuild"
        else:
            return "unknown"
    
    def _analyze_compiler_context(self, error_message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze compiler-specific context and provide additional insights."""
        additional_context = {
            "detected_compiler": "unknown",
            "compilation_stage": "unknown",
            "optimization_related": False,
            "standards_related": False
        }
        
        # Detect compiler type
        if re.search(r"gcc|g\+\+", error_message, re.IGNORECASE):
            additional_context["detected_compiler"] = "gcc"
        elif re.search(r"clang|clang\+\+", error_message, re.IGNORECASE):
            additional_context["detected_compiler"] = "clang"
        elif re.search(r"msvc|cl\.exe", error_message, re.IGNORECASE):
            additional_context["detected_compiler"] = "msvc"
        
        # Detect compilation stage
        if re.search(r"ld:|linker|undefined reference", error_message, re.IGNORECASE):
            additional_context["compilation_stage"] = "linking"
        elif re.search(r"preprocess|#include|macro", error_message, re.IGNORECASE):
            additional_context["compilation_stage"] = "preprocessing"
        elif re.search(r"syntax|parse|expected", error_message, re.IGNORECASE):
            additional_context["compilation_stage"] = "parsing"
        elif re.search(r"template|instantiation", error_message, re.IGNORECASE):
            additional_context["compilation_stage"] = "template_instantiation"
        
        # Check for optimization-related issues
        if re.search(r"optimization|O2|O3|unroll", error_message, re.IGNORECASE):
            additional_context["optimization_related"] = True
        
        # Check for C++ standards issues
        if re.search(r"c\+\+\d+|std=|standard", error_message, re.IGNORECASE):
            additional_context["standards_related"] = True
        
        return additional_context


class CPPPatchGenerator:
    """
    Generates patches and fixes for C/C++ code issues.
    
    This class provides automated patch generation for common C/C++ errors,
    memory management issues, and build configuration problems.
    """
    
    def __init__(self):
        """Initialize the C/C++ patch generator."""
        self.patch_templates = self._load_patch_templates()
    
    def _load_patch_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load patch templates for different types of C/C++ errors."""
        return {
            "memory_leak": {
                "description": "Fix memory leaks by adding proper cleanup",
                "template": """
// Original problematic code:
// {original_code}

// Fixed code with proper memory management:
{fixed_code}

// Additional recommendations:
// 1. Consider using smart pointers (std::unique_ptr, std::shared_ptr)
// 2. Follow RAII principles for automatic resource management
// 3. Use containers instead of raw arrays when possible
"""
            },
            "null_pointer": {
                "description": "Add null pointer checks",
                "template": """
// Add null pointer check before dereferencing:
if ({pointer_name} != nullptr) {
    // Original code here
    {original_code}
} else {
    // Handle null pointer case
    // Log error or return appropriate error code
    std::cerr << "Error: {pointer_name} is null" << std::endl;
    return; // or appropriate error handling
}
"""
            },
            "include_fix": {
                "description": "Fix missing include statements",
                "template": """
// Add missing include at the top of the file:
#include <{missing_header}>

// If it's a custom header, use quotes:
#include "{missing_header}"

// Common includes for different functionalities:
// - std::string: #include <string>
// - std::vector: #include <vector>
// - std::cout: #include <iostream>
// - std::shared_ptr: #include <memory>
"""
            },
            "template_fix": {
                "description": "Fix template instantiation issues",
                "template": """
// Template instantiation fix:
// Ensure template parameters meet requirements

// Original template:
{original_template}

// Fixed template with proper constraints:
template<typename T>
requires {constraints}  // C++20 concepts
// or use SFINAE for older standards:
// typename std::enable_if<{constraints}, void>::type* = nullptr
{fixed_template}
"""
            },
            "cmake_fix": {
                "description": "Fix CMake configuration issues",
                "template": """
# CMakeLists.txt fixes:

# Set minimum required version
cmake_minimum_required(VERSION 3.10)

# Set project name and version
project({project_name} VERSION 1.0.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD {cpp_standard})
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package({required_package} REQUIRED)

# Add executable or library
add_executable({target_name} {source_files})

# Link libraries
target_link_libraries({target_name} {libraries})

# Set include directories
target_include_directories({target_name} PRIVATE {include_dirs})
"""
            }
        }
    
    def generate_patch(self, error_analysis: Dict[str, Any], source_code: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a patch for the identified C/C++ error.
        
        Args:
            error_analysis: Analysis results from CPPExceptionHandler
            source_code: Optional source code context
            
        Returns:
            Generated patch information
        """
        patch_info = {
            "patch_type": "unknown",
            "confidence": 0.0,
            "patch_content": "",
            "explanation": "",
            "additional_steps": [],
            "risks": []
        }
        
        if not error_analysis.get("matches"):
            return patch_info
        
        primary_match = error_analysis["matches"][0]
        category = primary_match["category"]
        
        # Generate patch based on error category
        if category == "memory":
            patch_info = self._generate_memory_patch(error_analysis, source_code)
        elif category == "compilation":
            patch_info = self._generate_compilation_patch(error_analysis, source_code)
        elif category == "linking":
            patch_info = self._generate_linking_patch(error_analysis, source_code)
        elif category == "preprocessor":
            patch_info = self._generate_preprocessor_patch(error_analysis, source_code)
        elif category == "templates":
            patch_info = self._generate_template_patch(error_analysis, source_code)
        else:
            patch_info = self._generate_generic_patch(error_analysis, source_code)
        
        return patch_info
    
    def _generate_memory_patch(self, error_analysis: Dict[str, Any], source_code: Optional[str]) -> Dict[str, Any]:
        """Generate patch for memory-related errors."""
        template = self.patch_templates["memory_leak"]
        
        return {
            "patch_type": "memory_management",
            "confidence": 0.8,
            "patch_content": template["template"],
            "explanation": template["description"],
            "additional_steps": [
                "Compile with AddressSanitizer: -fsanitize=address",
                "Run static analysis tools like clang-static-analyzer",
                "Use valgrind for runtime memory checking",
                "Consider migrating to smart pointers"
            ],
            "risks": [
                "Changing memory management may affect performance",
                "Ensure all code paths are properly updated"
            ]
        }
    
    def _generate_compilation_patch(self, error_analysis: Dict[str, Any], source_code: Optional[str]) -> Dict[str, Any]:
        """Generate patch for compilation errors."""
        return {
            "patch_type": "compilation_fix",
            "confidence": 0.7,
            "patch_content": "// Fix compilation errors based on compiler messages",
            "explanation": "Address syntax and semantic errors",
            "additional_steps": [
                "Check C++ standard compatibility",
                "Verify all required headers are included",
                "Ensure proper namespace usage",
                "Check template instantiations"
            ],
            "risks": [
                "Syntax changes may affect code semantics",
                "Verify changes don't break other parts of code"
            ]
        }
    
    def _generate_linking_patch(self, error_analysis: Dict[str, Any], source_code: Optional[str]) -> Dict[str, Any]:
        """Generate patch for linking errors."""
        return {
            "patch_type": "linking_fix",
            "confidence": 0.6,
            "patch_content": "# Add required libraries and link flags",
            "explanation": "Fix undefined symbols and library linking issues",
            "additional_steps": [
                "Check if all required libraries are installed",
                "Verify library paths in build configuration",
                "Ensure proper symbol visibility",
                "Check for missing object files"
            ],
            "risks": [
                "Adding libraries may introduce new dependencies",
                "Version conflicts between libraries"
            ]
        }
    
    def _generate_preprocessor_patch(self, error_analysis: Dict[str, Any], source_code: Optional[str]) -> Dict[str, Any]:
        """Generate patch for preprocessor errors."""
        template = self.patch_templates["include_fix"]
        
        return {
            "patch_type": "preprocessor_fix",
            "confidence": 0.8,
            "patch_content": template["template"],
            "explanation": template["description"],
            "additional_steps": [
                "Check include paths in build system",
                "Verify header file installation",
                "Consider forward declarations where appropriate",
                "Check for circular dependencies"
            ],
            "risks": [
                "Adding includes may increase compilation time",
                "Potential for circular include dependencies"
            ]
        }
    
    def _generate_template_patch(self, error_analysis: Dict[str, Any], source_code: Optional[str]) -> Dict[str, Any]:
        """Generate patch for template errors."""
        template = self.patch_templates["template_fix"]
        
        return {
            "patch_type": "template_fix",
            "confidence": 0.5,
            "patch_content": template["template"],
            "explanation": template["description"],
            "additional_steps": [
                "Check template parameter constraints",
                "Verify template specializations",
                "Consider using concepts (C++20) for better error messages",
                "Simplify complex template hierarchies"
            ],
            "risks": [
                "Template changes may affect compilation time",
                "May break existing template instantiations"
            ]
        }
    
    def _generate_generic_patch(self, error_analysis: Dict[str, Any], source_code: Optional[str]) -> Dict[str, Any]:
        """Generate generic patch for other error types."""
        return {
            "patch_type": "generic_fix",
            "confidence": 0.3,
            "patch_content": "// Generic fix based on error analysis",
            "explanation": "Address the identified issue",
            "additional_steps": error_analysis.get("fix_suggestions", []),
            "risks": [
                "Manual review required for generic fixes",
                "Test thoroughly after applying changes"
            ]
        }


class CPPLanguagePlugin(LanguagePlugin):
    """
    C/C++ Language Plugin for Homeostasis.
    
    This plugin provides comprehensive C/C++ error analysis and fixing capabilities
    for the Homeostasis self-healing software system.
    """
    
    def __init__(self):
        """Initialize the C/C++ language plugin."""
        super().__init__()
        self.name = "cpp_plugin"
        self.version = "1.0.0"
        self.description = "C/C++ error analysis and fixing plugin"
        self.supported_languages = ["c", "cpp", "c++", "cc", "cxx"]
        self.supported_extensions = [".c", ".cpp", ".cc", ".cxx", ".c++", ".h", ".hpp", ".hxx"]
        
        # Initialize components
        self.exception_handler = CPPExceptionHandler()
        self.patch_generator = CPPPatchGenerator()
        
        logger.info(f"Initialized {self.name} v{self.version}")
    
    def can_handle(self, language: str, file_path: Optional[str] = None) -> bool:
        """
        Check if this plugin can handle the given language or file.
        
        Args:
            language: Programming language identifier
            file_path: Optional path to the source file
            
        Returns:
            True if this plugin can handle the language/file, False otherwise
        """
        # Check language
        if language.lower() in self.supported_languages:
            return True
        
        # Check file extension
        if file_path:
            file_path_obj = Path(file_path)
            if file_path_obj.suffix.lower() in self.supported_extensions:
                return True
        
        return False
    
    def analyze_error(self, error_message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze a C/C++ error and provide comprehensive information.
        
        Args:
            error_message: The error message to analyze
            context: Optional context information
            
        Returns:
            Comprehensive error analysis results
        """
        if context is None:
            context = {}
        
        # Use the exception handler to analyze the error
        analysis = self.exception_handler.analyze_error(error_message, context)
        
        # Add plugin metadata
        analysis.update({
            "plugin_name": self.name,
            "plugin_version": self.version,
            "analysis_timestamp": self._get_timestamp(),
            "confidence_score": self._calculate_confidence(analysis)
        })
        
        return analysis
    
    def generate_fix(self, error_analysis: Dict[str, Any], source_code: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a fix for the analyzed C/C++ error.
        
        Args:
            error_analysis: Results from analyze_error
            source_code: Optional source code context
            
        Returns:
            Generated fix information
        """
        # Use the patch generator to create a fix
        patch_info = self.patch_generator.generate_patch(error_analysis, source_code)
        
        # Add plugin metadata
        patch_info.update({
            "plugin_name": self.name,
            "plugin_version": self.version,
            "generation_timestamp": self._get_timestamp(),
            "error_analysis": error_analysis
        })
        
        return patch_info
    
    def test_fix(self, original_code: str, fixed_code: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Test a generated fix by attempting compilation.
        
        Args:
            original_code: The original problematic code
            fixed_code: The proposed fixed code
            context: Optional context information
            
        Returns:
            Test results
        """
        if context is None:
            context = {}
        
        test_results = {
            "success": False,
            "compilation_successful": False,
            "errors": [],
            "warnings": [],
            "execution_test": None
        }
        
        try:
            # Test compilation of fixed code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as temp_file:
                temp_file.write(fixed_code)
                temp_file_path = temp_file.name
            
            # Attempt compilation
            compiler = context.get("compiler", "g++")
            compile_command = [compiler, "-c", temp_file_path, "-o", "/dev/null"]
            
            result = subprocess.run(
                compile_command,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                test_results["compilation_successful"] = True
                test_results["success"] = True
            else:
                test_results["errors"] = result.stderr.split('\n')
            
            # Clean up
            os.unlink(temp_file_path)
            
        except subprocess.TimeoutExpired:
            test_results["errors"] = ["Compilation timeout"]
        except Exception as e:
            test_results["errors"] = [f"Test execution error: {str(e)}"]
        
        return test_results
    
    def get_recommendations(self, error_analysis: Dict[str, Any]) -> List[str]:
        """
        Get recommendations for preventing similar C/C++ errors.
        
        Args:
            error_analysis: Results from analyze_error
            
        Returns:
            List of prevention recommendations
        """
        recommendations = []
        
        if error_analysis.get("primary_category") == "memory":
            recommendations.extend([
                "Use smart pointers (std::unique_ptr, std::shared_ptr) instead of raw pointers",
                "Follow RAII principles for automatic resource management",
                "Enable compiler warnings: -Wall -Wextra -Werror",
                "Use static analysis tools like clang-static-analyzer",
                "Run with AddressSanitizer during development: -fsanitize=address"
            ])
        elif error_analysis.get("primary_category") == "compilation":
            recommendations.extend([
                "Use modern C++ standards (C++17 or later)",
                "Enable all compiler warnings and treat them as errors",
                "Use code formatting tools like clang-format",
                "Implement comprehensive unit tests",
                "Use continuous integration for build verification"
            ])
        elif error_analysis.get("primary_category") == "linking":
            recommendations.extend([
                "Use modern build systems like CMake",
                "Organize code into proper modules and libraries",
                "Document library dependencies clearly",
                "Use package managers like Conan or vcpkg",
                "Implement proper symbol visibility management"
            ])
        
        # General C/C++ recommendations
        recommendations.extend([
            "Use version control and code review processes",
            "Implement automated testing and continuous integration",
            "Follow established C++ coding standards (e.g., Google, LLVM)",
            "Use modern C++ features to avoid common pitfalls",
            "Regularly update compiler and development tools"
        ])
        
        return recommendations
    
    def _calculate_confidence(self, analysis: Dict[str, Any]) -> float:
        """Calculate confidence score for the analysis."""
        if not analysis.get("matches"):
            return 0.0
        
        # Base confidence on number and quality of matches
        match_count = len(analysis["matches"])
        severity_weight = {"critical": 1.0, "high": 0.8, "medium": 0.6, "low": 0.4}
        
        total_weight = sum(severity_weight.get(match["severity"], 0.2) for match in analysis["matches"])
        confidence = min(total_weight / match_count, 1.0) if match_count > 0 else 0.0
        
        return round(confidence, 2)
    
    def _get_timestamp(self) -> str:
        """Get current timestamp string."""
        import datetime
        return datetime.datetime.now().isoformat()


# Register the plugin
@register_plugin
def create_cpp_plugin():
    """Factory function to create CPP plugin instance."""
    return CPPLanguagePlugin()


# Export the plugin class for direct usage
__all__ = ['CPPLanguagePlugin', 'CPPExceptionHandler', 'CPPPatchGenerator']