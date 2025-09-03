"""
MATLAB Language Plugin for Homeostasis

This plugin enables Homeostasis to analyze and fix errors in MATLAB programming language code.
It provides comprehensive error handling for MATLAB syntax errors, runtime issues,
and scientific computing best practices.
"""
import logging
import re
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

from ..language_plugin_system import LanguagePlugin, register_plugin

logger = logging.getLogger(__name__)


class MATLABExceptionHandler:
    """
    Handles MATLAB exceptions with robust error detection and classification.
    
    This class provides logic for categorizing MATLAB errors based on their type,
    message, and common scientific computing patterns.
    """
    
    def __init__(self):
        """Initialize the MATLAB exception handler."""
        self.rule_categories = {
            "syntax": "MATLAB syntax and parsing errors",
            "runtime": "Runtime errors and exceptions",
            "matrix": "Matrix operations and linear algebra errors",
            "function": "Function definition and calling errors",
            "variable": "Variable scope and workspace errors",
            "file": "File I/O and data loading errors",
            "plot": "Plotting and visualization errors",
            "toolbox": "Toolbox and package errors",
            "memory": "Memory allocation and performance errors",
            "index": "Array indexing and dimension errors",
            "type": "Data type and conversion errors",
            "optimization": "Optimization and numerical computation errors"
        }
        
        # Common MATLAB error patterns
        self.matlab_error_patterns = {
            "syntax_error": [
                r"Parse error",
                r"Unexpected.*token",
                r"Expression or statement.*incorrect",
                r"Invalid.*syntax",
                r"Missing.*operator",
                r"Unbalanced.*parentheses",
                r"Expected.*end",
                r"Incomplete.*statement"
            ],
            "runtime_error": [
                r"Undefined.*variable",
                r"Undefined.*function",
                r"Not enough input arguments",
                r"Too many input arguments",
                r"Index exceeds.*bounds",
                r"Out of memory",
                r"Maximum recursion limit",
                r"Division by zero"
            ],
            "matrix_error": [
                r"Matrix dimensions.*not agree",
                r"Matrix.*singular",
                r"Matrix.*not square",
                r"Inner matrix dimensions.*not agree",
                r"Array indices.*positive integers",
                r"Subscript indices.*either.*positive",
                r"Matrix.*must be.*same size",
                r"Incompatible array sizes"
            ],
            "function_error": [
                r"Undefined.*function",
                r"Not enough input arguments",
                r"Too many input arguments",
                r"Expected.*input.*arguments",
                r"Function.*not found",
                r"Invalid.*function.*handle",
                r"Function.*requires.*toolbox"
            ],
            "variable_error": [
                r"Undefined.*variable",
                r"Variable.*not found",
                r"Attempt to reference.*nonexistent",
                r"Variable.*cleared",
                r"Variable.*out of scope",
                r"Global.*variable.*not found"
            ],
            "file_error": [
                r"Unable to read file",
                r"File.*not found",
                r"Permission denied",
                r"Invalid.*file.*format",
                r"Cannot open.*file",
                r"File.*already exists",
                r"Directory.*not found"
            ],
            "plot_error": [
                r"Invalid.*axes.*handle",
                r"Invalid.*figure.*handle",
                r"Graphics.*object.*not found",
                r"Invalid.*plot.*data",
                r"Figure.*window.*not available",
                r"OpenGL.*error"
            ],
            "toolbox_error": [
                r"Requires.*toolbox",
                r"Function.*not available",
                r"License.*not found",
                r"Toolbox.*not installed",
                r"Feature.*not available",
                r"Invalid.*license"
            ],
            "memory_error": [
                r"Out of memory",
                r"Cannot allocate.*memory",
                r"Memory.*allocation.*failed",
                r"Maximum.*variable.*size",
                r"Array.*too large",
                r"Insufficient.*memory"
            ],
            "index_error": [
                r"Index exceeds.*bounds",
                r"Array indices.*positive integers",
                r"Subscript indices.*positive",
                r"Index.*out of range",
                r"Invalid.*array.*index",
                r"Logical.*array.*wrong.*size"
            ],
            "type_error": [
                r"Conversion.*not possible",
                r"Cannot convert.*to",
                r"Invalid.*data.*type",
                r"Type.*mismatch",
                r"Expected.*numeric.*array",
                r"Cell.*array.*not.*supported"
            ],
            "optimization_error": [
                r"Optimization.*terminated",
                r"Function.*not.*decreasing",
                r"Line search.*failed",
                r"Gradient.*not.*finite",
                r"Hessian.*not.*positive.*definite",
                r"Constraint.*violation"
            ]
        }
        
        # MATLAB-specific concepts and their common issues
        self.matlab_concepts = {
            "matrices": ["matrix", "array", "dimension", "size"],
            "indexing": ["index", "subscript", "logical", "colon"],
            "functions": ["function", "handle", "anonymous", "nested"],
            "variables": ["variable", "workspace", "global", "persistent"],
            "plots": ["figure", "axes", "plot", "graphics"],
            "toolboxes": ["toolbox", "license", "feature"],
            "files": ["file", "load", "save", "import", "export"],
            "cell_arrays": ["cell", "array", "structure", "field"],
            "strings": ["string", "char", "text", "character"],
            "optimization": ["fmincon", "fminsearch", "optimization", "minimize"]
        }
        
        # Load rules from different categories
        self.rules = self._load_rules()
        
        # Pre-compile regex patterns for better performance
        self._compile_patterns()
    
    def _load_rules(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load MATLAB error rules from rule files."""
        rules = {}
        rules_dir = Path(__file__).parent.parent / "rules" / "matlab"
        
        try:
            # Load common MATLAB rules
            common_rules_path = rules_dir / "matlab_common_errors.json"
            if common_rules_path.exists():
                with open(common_rules_path, 'r') as f:
                    common_data = json.load(f)
                    rules["common"] = common_data.get("rules", [])
                    logger.info(f"Loaded {len(rules['common'])} common MATLAB rules")
            
            # Load concept-specific rules
            for concept in ["matrices", "functions", "plotting", "toolboxes"]:
                concept_rules_path = rules_dir / f"matlab_{concept}_errors.json"
                if concept_rules_path.exists():
                    with open(concept_rules_path, 'r') as f:
                        concept_data = json.load(f)
                        rules[concept] = concept_data.get("rules", [])
                        logger.info(f"Loaded {len(rules[concept])} {concept} rules")
                        
        except Exception as e:
            logger.error(f"Error loading MATLAB rules: {e}")
            rules = {"common": []}
        
        return rules
    
    def _compile_patterns(self):
        """Pre-compile regex patterns for better performance."""
        self.compiled_patterns = {}
        
        for category, rule_list in self.rules.items():
            self.compiled_patterns[category] = []
            for rule in rule_list:
                try:
                    pattern = rule.get("pattern", "")
                    if pattern:
                        compiled = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
                        self.compiled_patterns[category].append((compiled, rule))
                except re.error as e:
                    logger.warning(f"Invalid regex pattern in rule {rule.get('id', 'unknown')}: {e}")
    
    def analyze_exception(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a MATLAB exception and determine its type and potential fixes.
        
        Args:
            error_data: MATLAB error data in standard format
            
        Returns:
            Analysis results with categorization and fix suggestions
        """
        error_type = error_data.get("error_type", "MATLABError")
        message = error_data.get("message", "")
        file_path = error_data.get("file_path", "")
        line_number = error_data.get("line_number", 0)
        
        # Analyze based on error patterns
        analysis = self._analyze_by_patterns(message, file_path)
        
        # Check for concept-specific issues only if we don't have high confidence yet
        if analysis.get("confidence") != "high":
            concept_analysis = self._analyze_matlab_concepts(message)
            if concept_analysis.get("confidence", "low") != "low":
                # Merge concept-specific findings without overwriting existing good subcategory
                if analysis.get("subcategory") in ["unknown", "runtime"]:
                    analysis.update(concept_analysis)
                else:
                    # Keep existing subcategory but update other fields
                    for key, value in concept_analysis.items():
                        if key != "subcategory":
                            analysis[key] = value
        
        # Find matching rules
        matches = self._find_matching_rules(message, error_data)
        
        # Only use rule matches if we don't already have high confidence analysis
        if matches and analysis.get("confidence") != "high":
            # Use the best match (highest confidence)
            best_match = max(matches, key=lambda x: x.get("confidence_score", 0))
            # Only update if they have values - preserve existing category if rule doesn't specify
            if best_match.get("category"):
                analysis["category"] = best_match["category"]
            if best_match.get("type") and analysis.get("subcategory") == "unknown":
                analysis["subcategory"] = best_match["type"]
            analysis.update({
                "confidence": best_match.get("confidence", analysis.get("confidence", "medium")),
                "suggested_fix": best_match.get("suggestion", analysis.get("suggested_fix", "")),
                "root_cause": best_match.get("root_cause", analysis.get("root_cause", "")),
                "severity": best_match.get("severity", "medium"),
                "rule_id": best_match.get("id", ""),
                "tags": best_match.get("tags", analysis.get("tags", [])),
                "all_matches": matches
            })
        
        # Always ensure category is "matlab" for this plugin
        analysis["category"] = "matlab"
        analysis["file_path"] = file_path
        analysis["line_number"] = line_number
        return analysis
    
    def _analyze_by_patterns(self, message: str, file_path: str) -> Dict[str, Any]:
        """Analyze error by matching against common patterns."""
        message_lower = message.lower()
        
        # Check syntax errors
        for pattern in self.matlab_error_patterns["syntax_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "matlab",
                    "subcategory": "syntax",
                    "confidence": "high",
                    "suggested_fix": "Fix MATLAB syntax errors",
                    "root_cause": "matlab_syntax_error",
                    "severity": "high",
                    "tags": ["matlab", "syntax", "parser"]
                }
        
        # Check for specific undefined errors
        if "undefined function or variable" in message_lower or "undefined variable" in message_lower:
            return {
                "category": "matlab",
                "subcategory": "undefined",
                "confidence": "high",
                "suggested_fix": "Define the variable or function before use",
                "root_cause": "matlab_undefined_error",
                "severity": "high",
                "tags": ["matlab", "undefined", "variable"]
            }
        
        # Check for dimension errors
        if "matrix dimensions" in message_lower and ("must agree" in message_lower or "not agree" in message_lower):
            return {
                "category": "matlab",
                "subcategory": "dimension",
                "confidence": "high",
                "suggested_fix": "Check matrix dimensions for compatibility",
                "root_cause": "matlab_dimension_error",
                "severity": "high",
                "tags": ["matlab", "dimension", "matrix"]
            }
        
        # Check for index errors
        if "index exceeds" in message_lower or "indices must" in message_lower:
            return {
                "category": "matlab",
                "subcategory": "index",
                "confidence": "high",
                "suggested_fix": "Check array bounds and indices",
                "root_cause": "matlab_index_error",
                "severity": "high",
                "tags": ["matlab", "index", "bounds"]
            }
        
        # Check runtime errors
        for pattern in self.matlab_error_patterns["runtime_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "matlab",
                    "subcategory": "runtime",
                    "confidence": "high",
                    "suggested_fix": "Fix runtime errors and variable access",
                    "root_cause": "matlab_runtime_error",
                    "severity": "high",
                    "tags": ["matlab", "runtime", "variable"]
                }
        
        # Check matrix errors
        for pattern in self.matlab_error_patterns["matrix_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "matlab",
                    "subcategory": "matrix",
                    "confidence": "high",
                    "suggested_fix": "Fix matrix operations and dimensions",
                    "root_cause": "matlab_matrix_error",
                    "severity": "high",
                    "tags": ["matlab", "matrix", "dimension"]
                }
        
        # Check function errors
        for pattern in self.matlab_error_patterns["function_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "matlab",
                    "subcategory": "function",
                    "confidence": "high",
                    "suggested_fix": "Fix function definition and calling errors",
                    "root_cause": "matlab_function_error",
                    "severity": "high",
                    "tags": ["matlab", "function", "argument"]
                }
        
        # Check variable errors
        for pattern in self.matlab_error_patterns["variable_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "matlab",
                    "subcategory": "variable",
                    "confidence": "high",
                    "suggested_fix": "Fix variable scope and workspace errors",
                    "root_cause": "matlab_variable_error",
                    "severity": "high",
                    "tags": ["matlab", "variable", "workspace"]
                }
        
        # Check file errors
        for pattern in self.matlab_error_patterns["file_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "matlab",
                    "subcategory": "file",
                    "confidence": "high",
                    "suggested_fix": "Fix file I/O and data loading errors",
                    "root_cause": "matlab_file_error",
                    "severity": "high",
                    "tags": ["matlab", "file", "io"]
                }
        
        # Check plot errors
        for pattern in self.matlab_error_patterns["plot_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "matlab",
                    "subcategory": "plot",
                    "confidence": "high",
                    "suggested_fix": "Fix plotting and visualization errors",
                    "root_cause": "matlab_plot_error",
                    "severity": "medium",
                    "tags": ["matlab", "plot", "graphics"]
                }
        
        # Check toolbox errors
        for pattern in self.matlab_error_patterns["toolbox_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "matlab",
                    "subcategory": "toolbox",
                    "confidence": "high",
                    "suggested_fix": "Fix toolbox and licensing errors",
                    "root_cause": "matlab_toolbox_error",
                    "severity": "high",
                    "tags": ["matlab", "toolbox", "license"]
                }
        
        # Check memory errors
        for pattern in self.matlab_error_patterns["memory_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "matlab",
                    "subcategory": "memory",
                    "confidence": "high",
                    "suggested_fix": "Fix memory allocation and performance errors",
                    "root_cause": "matlab_memory_error",
                    "severity": "critical",
                    "tags": ["matlab", "memory", "performance"]
                }
        
        # Check index errors
        for pattern in self.matlab_error_patterns["index_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "matlab",
                    "subcategory": "index",
                    "confidence": "high",
                    "suggested_fix": "Fix array indexing and dimension errors",
                    "root_cause": "matlab_index_error",
                    "severity": "high",
                    "tags": ["matlab", "index", "array"]
                }
        
        # Check type errors
        for pattern in self.matlab_error_patterns["type_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "matlab",
                    "subcategory": "type",
                    "confidence": "high",
                    "suggested_fix": "Fix data type and conversion errors",
                    "root_cause": "matlab_type_error",
                    "severity": "medium",
                    "tags": ["matlab", "type", "conversion"]
                }
        
        # Check optimization errors
        for pattern in self.matlab_error_patterns["optimization_error"]:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "category": "matlab",
                    "subcategory": "optimization",
                    "confidence": "high",
                    "suggested_fix": "Fix optimization and numerical computation errors",
                    "root_cause": "matlab_optimization_error",
                    "severity": "medium",
                    "tags": ["matlab", "optimization", "numerical"]
                }
        
        return {
            "category": "matlab",
            "subcategory": "unknown",
            "confidence": "low",
            "suggested_fix": "Review MATLAB code and error details",
            "root_cause": "matlab_generic_error",
            "severity": "medium",
            "tags": ["matlab", "generic"]
        }
    
    def _analyze_matlab_concepts(self, message: str) -> Dict[str, Any]:
        """Analyze MATLAB-specific concept errors."""
        message_lower = message.lower()
        
        # Check for undefined variable/function errors
        if any(keyword in message_lower for keyword in ["undefined", "variable", "function"]):
            return {
                "category": "matlab",
                "subcategory": "undefined",
                "confidence": "high",
                "suggested_fix": "Check variable/function name and ensure it's defined",
                "root_cause": "matlab_undefined_error",
                "severity": "high",
                "tags": ["matlab", "undefined", "variable"]
            }
        
        # Check for matrix dimension errors
        if any(keyword in message_lower for keyword in ["matrix", "dimension", "size"]):
            return {
                "category": "matlab",
                "subcategory": "matrix",
                "confidence": "high",
                "suggested_fix": "Check matrix dimensions and operations",
                "root_cause": "matlab_matrix_error",
                "severity": "high",
                "tags": ["matlab", "matrix", "dimension"]
            }
        
        # Check for index errors
        if any(keyword in message_lower for keyword in ["index", "subscript", "bounds"]):
            return {
                "category": "matlab",
                "subcategory": "index",
                "confidence": "high",
                "suggested_fix": "Check array indices and bounds",
                "root_cause": "matlab_index_error",
                "severity": "high",
                "tags": ["matlab", "index", "bounds"]
            }
        
        # Check for toolbox errors
        if any(keyword in message_lower for keyword in ["toolbox", "license", "feature"]):
            return {
                "category": "matlab",
                "subcategory": "toolbox",
                "confidence": "medium",
                "suggested_fix": "Check toolbox availability and licensing",
                "root_cause": "matlab_toolbox_error",
                "severity": "medium",
                "tags": ["matlab", "toolbox", "license"]
            }
        
        # Check for file errors
        if any(keyword in message_lower for keyword in ["file", "load", "save", "read"]):
            return {
                "category": "matlab",
                "subcategory": "file",
                "confidence": "medium",
                "suggested_fix": "Check file paths and permissions",
                "root_cause": "matlab_file_error",
                "severity": "medium",
                "tags": ["matlab", "file", "io"]
            }
        
        return {"confidence": "low"}
    
    def _find_matching_rules(self, error_text: str, error_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find all rules that match the given error."""
        matches = []
        
        for category, patterns in self.compiled_patterns.items():
            for compiled_pattern, rule in patterns:
                match = compiled_pattern.search(error_text)
                if match:
                    # Calculate confidence score based on match quality
                    confidence_score = self._calculate_confidence(match, rule, error_data)
                    
                    match_info = rule.copy()
                    match_info["confidence_score"] = confidence_score
                    match_info["match_groups"] = match.groups() if match.groups() else []
                    matches.append(match_info)
        
        return matches
    
    def _calculate_confidence(self, match: re.Match, rule: Dict[str, Any], 
                             error_data: Dict[str, Any]) -> float:
        """Calculate confidence score for a rule match."""
        base_confidence = 0.5
        
        # Boost confidence for file extension matches
        file_path = error_data.get("file_path", "")
        if file_path.endswith(".m") or file_path.endswith(".mat"):
            base_confidence += 0.2
        
        # Boost confidence based on rule reliability
        reliability = rule.get("reliability", "medium")
        reliability_boost = {"high": 0.2, "medium": 0.1, "low": 0.0}
        base_confidence += reliability_boost.get(reliability, 0.0)
        
        # Boost confidence for concept matches
        rule_tags = set(rule.get("tags", []))
        context_tags = set()
        
        message = error_data.get("message", "").lower()
        if "matrix" in message:
            context_tags.add("matrix")
        if "function" in message:
            context_tags.add("function")
        if "index" in message:
            context_tags.add("index")
        
        if context_tags & rule_tags:
            base_confidence += 0.1
        
        return min(base_confidence, 1.0)


class MATLABPatchGenerator:
    """
    Generates patches for MATLAB errors based on analysis results.
    
    This class creates MATLAB code fixes for common errors using templates
    and heuristics specific to scientific computing patterns.
    """
    
    def __init__(self):
        """Initialize the MATLAB patch generator."""
        self.template_dir = Path(__file__).parent.parent / "patch_generation" / "templates"
        self.matlab_template_dir = self.template_dir / "matlab"
        
        # Ensure template directory exists
        self.matlab_template_dir.mkdir(parents=True, exist_ok=True)
        
        # Load patch templates
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, str]:
        """Load MATLAB patch templates."""
        templates = {}
        
        if not self.matlab_template_dir.exists():
            logger.warning(f"MATLAB templates directory not found: {self.matlab_template_dir}")
            return templates
        
        for template_file in self.matlab_template_dir.glob("*.m.template"):
            try:
                with open(template_file, 'r') as f:
                    template_name = template_file.stem.replace('.m', '')
                    templates[template_name] = f.read()
                    logger.debug(f"Loaded template: {template_name}")
            except Exception as e:
                logger.error(f"Error loading template {template_file}: {e}")
        
        return templates
    
    def generate_patch(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                      source_code: str = "") -> Optional[Dict[str, Any]]:
        """
        Generate a patch for the MATLAB error.
        
        Args:
            error_data: The MATLAB error data
            analysis: Analysis results from MATLABExceptionHandler
            source_code: The MATLAB source code that caused the error
            
        Returns:
            Patch information or None if no patch can be generated
        """
        root_cause = analysis.get("root_cause", "")
        subcategory = analysis.get("subcategory", "")
        
        # Map root causes to patch strategies
        patch_strategies = {
            "matlab_syntax_error": self._fix_syntax_error,
            "matlab_runtime_error": self._fix_runtime_error,
            "matlab_matrix_error": self._fix_matrix_error,
            "matlab_function_error": self._fix_function_error,
            "matlab_variable_error": self._fix_variable_error,
            "matlab_file_error": self._fix_file_error,
            "matlab_plot_error": self._fix_plot_error,
            "matlab_toolbox_error": self._fix_toolbox_error,
            "matlab_memory_error": self._fix_memory_error,
            "matlab_index_error": self._fix_index_error,
            "matlab_type_error": self._fix_type_error,
            "matlab_optimization_error": self._fix_optimization_error,
            "matlab_undefined_error": self._fix_undefined_error,
            "matlab_dimension_error": self._fix_dimension_error
        }
        
        strategy = patch_strategies.get(root_cause)
        if strategy:
            try:
                return strategy(error_data, analysis, source_code)
            except Exception as e:
                logger.error(f"Error generating patch for {root_cause}: {e}")
        
        # Try to use templates if no specific strategy matches
        return self._template_based_patch(error_data, analysis, source_code)
    
    def _fix_syntax_error(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                         source_code: str) -> Optional[Dict[str, Any]]:
        """Fix MATLAB syntax errors."""
        message = error_data.get("message", "")
        
        fixes = []
        
        if "unbalanced" in message.lower() and "parentheses" in message.lower():
            fixes.append({
                "type": "suggestion",
                "description": "Unbalanced parentheses",
                "fix": "Check for matching opening and closing parentheses"
            })
        
        if "expected" in message.lower() and "end" in message.lower():
            fixes.append({
                "type": "suggestion",
                "description": "Missing 'end' keyword",
                "fix": "Add 'end' to close function, if, for, while, or switch blocks"
            })
        
        if "incomplete" in message.lower() and "statement" in message.lower():
            fixes.append({
                "type": "suggestion",
                "description": "Incomplete statement",
                "fix": "Complete the statement or add missing semicolon"
            })
        
        if fixes:
            return {
                "type": "multiple_suggestions",
                "fixes": fixes,
                "description": "MATLAB syntax error fixes"
            }
        
        return {
            "type": "suggestion",
            "description": "MATLAB syntax error. Check code structure and syntax"
        }
    
    def _fix_runtime_error(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                          source_code: str) -> Optional[Dict[str, Any]]:
        """Fix runtime errors."""
        message = error_data.get("message", "")
        
        if "undefined" in message.lower() and "variable" in message.lower():
            # Extract variable name
            var_match = re.search(r"Undefined.*variable.*'(.+?)'", message)
            var_name = var_match.group(1) if var_match else "variable"
            
            return {
                "type": "suggestion",
                "description": f"Undefined variable '{var_name}'",
                "fixes": [
                    f"Define variable '{var_name}' before use",
                    f"Check spelling of '{var_name}'",
                    f"Use exist('{var_name}', 'var') to check if variable exists",
                    "Clear and reload workspace if needed",
                    "Check variable scope and function workspace"
                ]
            }
        
        if "undefined" in message.lower() and "function" in message.lower():
            # Extract function name
            func_match = re.search(r"Undefined.*function.*'(.+?)'", message)
            func_name = func_match.group(1) if func_match else "function"
            
            return {
                "type": "suggestion",
                "description": f"Undefined function '{func_name}'",
                "fixes": [
                    f"Check if function '{func_name}' is on the MATLAB path",
                    "Add function directory to path with addpath()",
                    f"Check spelling of '{func_name}'",
                    f"Install required toolbox containing '{func_name}'",
                    f"Use which('{func_name}') to locate function"
                ]
            }
        
        if "not enough input arguments" in message.lower():
            return {
                "type": "suggestion",
                "description": "Not enough input arguments",
                "fixes": [
                    "Provide all required function arguments",
                    "Check function documentation for required inputs",
                    "Use nargin to check argument count in function",
                    "Provide default values for optional arguments"
                ]
            }
        
        return {
            "type": "suggestion",
            "description": "Runtime error. Check variable and function definitions"
        }
    
    def _fix_matrix_error(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                         source_code: str) -> Optional[Dict[str, Any]]:
        """Fix matrix operation errors."""
        message = error_data.get("message", "")
        
        if "matrix dimensions" in message.lower() and "not agree" in message.lower():
            return {
                "type": "suggestion",
                "description": "Matrix dimensions do not agree",
                "fixes": [
                    "Check matrix dimensions with size()",
                    "Use transpose (') to match dimensions",
                    "Reshape matrices with reshape()",
                    "Use element-wise operations (.*) instead of matrix operations",
                    "Check that inner dimensions match for matrix multiplication"
                ]
            }
        
        if "matrix" in message.lower() and "singular" in message.lower():
            return {
                "type": "suggestion",
                "description": "Matrix is singular",
                "fixes": [
                    "Check matrix condition number with cond()",
                    "Use pinv() for pseudoinverse instead of inv()",
                    "Add regularization to avoid singularity",
                    "Use rank() to check matrix rank",
                    "Consider using SVD decomposition"
                ]
            }
        
        if "array indices" in message.lower() and "positive integers" in message.lower():
            return {
                "type": "suggestion",
                "description": "Array indices must be positive integers",
                "fixes": [
                    "Use positive integers for array indexing",
                    "Check index calculations and ensure they're positive",
                    "Use round() or floor() for non-integer indices",
                    "Use logical indexing for conditional selection",
                    "Check bounds with size() before indexing"
                ]
            }
        
        return {
            "type": "suggestion",
            "description": "Matrix operation error",
            "fixes": [
                "Check matrix dimensions and operations",
                "Verify matrix properties and conditioning",
                "Use appropriate linear algebra functions"
            ]
        }
    
    def _fix_function_error(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                           source_code: str) -> Optional[Dict[str, Any]]:
        """Fix function errors."""
        message = error_data.get("message", "")
        
        if "too many input arguments" in message.lower():
            return {
                "type": "suggestion",
                "description": "Too many input arguments",
                "fixes": [
                    "Check function signature and remove extra arguments",
                    "Use varargin for variable number of arguments",
                    "Check function documentation for correct usage",
                    "Use nargin to handle variable arguments"
                ]
            }
        
        if "function" in message.lower() and "not found" in message.lower():
            return {
                "type": "suggestion",
                "description": "Function not found",
                "fixes": [
                    "Check if function is on MATLAB path",
                    "Add function directory to path",
                    "Check function name spelling",
                    "Use which() to locate function",
                    "Install required toolbox"
                ]
            }
        
        return {
            "type": "suggestion",
            "description": "Function error",
            "fixes": [
                "Check function arguments and calling syntax",
                "Verify function availability and path",
                "Review function documentation"
            ]
        }
    
    def _fix_variable_error(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                           source_code: str) -> Optional[Dict[str, Any]]:
        """Fix variable scope and workspace errors."""
        message = error_data.get("message", "")
        
        if "variable" in message.lower() and "cleared" in message.lower():
            return {
                "type": "suggestion",
                "description": "Variable was cleared",
                "fixes": [
                    "Re-initialize cleared variables",
                    "Use persistent variables in functions",
                    "Check for clear commands that remove variables",
                    "Save/load workspace if needed",
                    "Use global variables for shared data"
                ]
            }
        
        if "out of scope" in message.lower():
            return {
                "type": "suggestion",
                "description": "Variable out of scope",
                "fixes": [
                    "Check variable scope in functions",
                    "Use global variables for shared access",
                    "Pass variables as function arguments",
                    "Use persistent variables for function-local storage",
                    "Check nested function variable access"
                ]
            }
        
        return {
            "type": "suggestion",
            "description": "Variable error",
            "fixes": [
                "Check variable definitions and scope",
                "Verify variable initialization",
                "Use appropriate variable scoping"
            ]
        }
    
    def _fix_file_error(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                       source_code: str) -> Optional[Dict[str, Any]]:
        """Fix file I/O errors."""
        message = error_data.get("message", "")
        
        if "file" in message.lower() and "not found" in message.lower():
            return {
                "type": "suggestion",
                "description": "File not found",
                "fixes": [
                    "Check file path and name",
                    "Use exist() to check if file exists",
                    "Use pwd to check current directory",
                    "Use fullfile() to construct file paths",
                    "Check file permissions"
                ]
            }
        
        if "unable to read file" in message.lower():
            return {
                "type": "suggestion",
                "description": "Unable to read file",
                "fixes": [
                    "Check file format and encoding",
                    "Use appropriate file reading function",
                    "Check file permissions and access",
                    "Verify file is not corrupted",
                    "Use try-catch for error handling"
                ]
            }
        
        return {
            "type": "suggestion",
            "description": "File I/O error",
            "fixes": [
                "Check file paths and permissions",
                "Verify file existence and format",
                "Use appropriate file handling functions"
            ]
        }
    
    def _fix_plot_error(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                       source_code: str) -> Optional[Dict[str, Any]]:
        """Fix plotting errors."""
        message = error_data.get("message", "")
        
        if "invalid" in message.lower() and "handle" in message.lower():
            return {
                "type": "suggestion",
                "description": "Invalid graphics handle",
                "fixes": [
                    "Check if figure/axes handles are valid",
                    "Use ishandle() to verify handle validity",
                    "Create new figure with figure()",
                    "Use gca for current axes",
                    "Check if figure was closed"
                ]
            }
        
        if "graphics" in message.lower() and "not found" in message.lower():
            return {
                "type": "suggestion",
                "description": "Graphics object not found",
                "fixes": [
                    "Create graphics objects before use",
                    "Check object handles and validity",
                    "Use findobj() to locate graphics objects",
                    "Verify plot data and parameters"
                ]
            }
        
        return {
            "type": "suggestion",
            "description": "Plotting error",
            "fixes": [
                "Check graphics handles and objects",
                "Verify plot data and parameters",
                "Use appropriate plotting functions"
            ]
        }
    
    def _fix_toolbox_error(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                          source_code: str) -> Optional[Dict[str, Any]]:
        """Fix toolbox and licensing errors."""
        message = error_data.get("message", "")
        
        if "requires" in message.lower() and "toolbox" in message.lower():
            # Extract toolbox name
            toolbox_match = re.search(r"requires.*?([A-Za-z\s]+).*?toolbox", message, re.IGNORECASE)
            toolbox_name = toolbox_match.group(1).strip() if toolbox_match else "toolbox"
            
            return {
                "type": "suggestion",
                "description": f"Requires {toolbox_name} toolbox",
                "fixes": [
                    f"Install {toolbox_name} toolbox",
                    "Check toolbox license with license('test', 'toolbox_name')",
                    "Use ver to check installed toolboxes",
                    "Find alternative functions that don't require toolbox",
                    "Contact administrator for toolbox access"
                ]
            }
        
        if "license" in message.lower() and "not found" in message.lower():
            return {
                "type": "suggestion",
                "description": "License not found",
                "fixes": [
                    "Check MATLAB license status",
                    "Contact administrator for license issues",
                    "Check network license server",
                    "Use license('inuse') to check active licenses",
                    "Try alternative functions"
                ]
            }
        
        return {
            "type": "suggestion",
            "description": "Toolbox/license error",
            "fixes": [
                "Check toolbox installation and licensing",
                "Verify required toolboxes are available",
                "Contact administrator for license issues"
            ]
        }
    
    def _fix_memory_error(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                         source_code: str) -> Optional[Dict[str, Any]]:
        """Fix memory allocation errors."""
        message = error_data.get("message", "")
        
        if "out of memory" in message.lower():
            return {
                "type": "suggestion",
                "description": "Out of memory",
                "fixes": [
                    "Reduce array sizes or use smaller data types",
                    "Process data in chunks",
                    "Use sparse matrices for large sparse data",
                    "Clear unused variables with clear",
                    "Use memory to check available memory",
                    "Consider using single precision instead of double"
                ]
            }
        
        return {
            "type": "suggestion",
            "description": "Memory error",
            "fixes": [
                "Optimize memory usage",
                "Use efficient data structures",
                "Clear unused variables"
            ]
        }
    
    def _fix_index_error(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                        source_code: str) -> Optional[Dict[str, Any]]:
        """Fix array indexing errors."""
        message = error_data.get("message", "")
        
        if "index exceeds" in message.lower() and "bounds" in message.lower():
            return {
                "type": "suggestion",
                "description": "Index exceeds array bounds",
                "fixes": [
                    "Check array size with size() or length()",
                    "Use valid indices within array bounds",
                    "Check index calculations",
                    "Use end for last element indexing",
                    "Use logical indexing for conditional selection"
                ]
            }
        
        return {
            "type": "suggestion",
            "description": "Array indexing error",
            "fixes": [
                "Check array dimensions and indices",
                "Verify index calculations",
                "Use appropriate indexing methods"
            ]
        }
    
    def _fix_type_error(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                       source_code: str) -> Optional[Dict[str, Any]]:
        """Fix data type errors."""
        message = error_data.get("message", "")
        
        if "conversion" in message.lower() and "not possible" in message.lower():
            return {
                "type": "suggestion",
                "description": "Type conversion not possible",
                "fixes": [
                    "Use appropriate conversion functions",
                    "Check data types with class() or isa()",
                    "Use str2num() for string to number conversion",
                    "Use double() or single() for numeric conversion",
                    "Handle cell arrays and structures appropriately"
                ]
            }
        
        return {
            "type": "suggestion",
            "description": "Data type error",
            "fixes": [
                "Check data types and conversions",
                "Use appropriate type conversion functions",
                "Verify data format and structure"
            ]
        }
    
    def _fix_optimization_error(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                               source_code: str) -> Optional[Dict[str, Any]]:
        """Fix optimization errors."""
        message = error_data.get("message", "")
        
        if "optimization terminated" in message.lower():
            return {
                "type": "suggestion",
                "description": "Optimization terminated",
                "fixes": [
                    "Check optimization settings and tolerances",
                    "Provide better initial guess",
                    "Check objective function and constraints",
                    "Use different optimization algorithm",
                    "Increase iteration limits"
                ]
            }
        
        return {
            "type": "suggestion",
            "description": "Optimization error",
            "fixes": [
                "Check optimization setup and parameters",
                "Verify objective function and constraints",
                "Try different optimization methods"
            ]
        }
    
    def _fix_undefined_error(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                            source_code: str) -> Optional[Dict[str, Any]]:
        """Fix undefined variable/function errors."""
        return {
            "type": "suggestion",
            "description": "Undefined variable or function",
            "fixes": [
                "Check variable/function name spelling",
                "Ensure variable/function is defined before use",
                "Check MATLAB path and function availability",
                "Use exist() to check if variable/function exists",
                "Clear and reload workspace if needed"
            ]
        }
    
    def _fix_dimension_error(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                            source_code: str) -> Optional[Dict[str, Any]]:
        """Fix matrix dimension mismatch errors."""
        message = error_data.get("message", "")
        
        if "must agree" in message.lower():
            return {
                "type": "suggestion",
                "description": "Matrix dimensions must agree",
                "fixes": [
                    "Check matrix sizes with size() before operations",
                    "Use element-wise operators (.*, ./, .^) for element operations",
                    "Use transpose (') or reshape() to align dimensions",
                    "Ensure compatible dimensions for matrix multiplication",
                    "Use bsxfun() for operations on different sized arrays"
                ]
            }
        
        return {
            "type": "suggestion",
            "description": "Matrix dimension error",
            "fixes": [
                "Check matrix dimensions with size() or length()",
                "Ensure matrices have compatible dimensions",
                "Use element-wise operations when appropriate",
                "Consider using repmat() or broadcast operations",
                "Debug with disp(size(variable)) to check dimensions"
            ]
        }
    
    def _template_based_patch(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                            source_code: str) -> Optional[Dict[str, Any]]:
        """Generate patch using templates."""
        root_cause = analysis.get("root_cause", "")
        subcategory = analysis.get("subcategory", "")
        
        # Map root causes to template names
        template_map = {
            "matlab_syntax_error": "syntax_fix",
            "matlab_runtime_error": "runtime_fix",
            "matlab_matrix_error": "matrix_fix",
            "matlab_function_error": "function_fix",
            "matlab_variable_error": "variable_fix",
            "matlab_file_error": "file_fix"
        }
        
        template_name = template_map.get(root_cause)
        if template_name and template_name in self.templates:
            template = self.templates[template_name]
            
            return {
                "type": "template",
                "template": template,
                "description": f"Applied template fix for {root_cause}"
            }
        
        return None


class MATLABLanguagePlugin(LanguagePlugin):
    """
    Main MATLAB language plugin for Homeostasis.
    
    This plugin orchestrates MATLAB error analysis and patch generation,
    supporting scientific computing and numerical analysis patterns.
    """
    
    VERSION = "1.0.0"
    AUTHOR = "Homeostasis Team"
    
    def __init__(self):
        """Initialize the MATLAB language plugin."""
        self.language = "matlab"
        self.supported_extensions = {".m", ".mat", ".mlx", ".mlapp"}
        self.supported_frameworks = [
            "matlab", "simulink", "signal_processing", "image_processing", 
            "statistics", "optimization", "control_systems", "neural_networks"
        ]
        
        # Initialize components
        self.exception_handler = MATLABExceptionHandler()
        self.patch_generator = MATLABPatchGenerator()
        
        logger.info("MATLAB language plugin initialized")
    
    def get_language_id(self) -> str:
        """Get the unique identifier for this language."""
        return "matlab"
    
    def get_language_name(self) -> str:
        """Get the human-readable name of the language."""
        return "MATLAB"
    
    def get_language_version(self) -> str:
        """Get the version of the language supported by this plugin."""
        return "R2021a+"
    
    def get_supported_frameworks(self) -> List[str]:
        """Get the list of frameworks supported by this language plugin."""
        return self.supported_frameworks
    
    def normalize_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize error data to the standard Homeostasis format.
        
        Args:
            error_data: Error data in the MATLAB-specific format
            
        Returns:
            Error data in the standard format
        """
        # Map MATLAB-specific error fields to standard format
        normalized = {
            "error_type": error_data.get("error_type", "MATLABError"),
            "message": error_data.get("message", error_data.get("description", "")),
            "language": "matlab",
            "file_path": error_data.get("file_path", error_data.get("file", "")),
            "line_number": error_data.get("line_number", error_data.get("line", 0)),
            "column_number": error_data.get("column_number", error_data.get("column", 0)),
            "matlab_version": error_data.get("matlab_version", ""),
            "source_code": error_data.get("source_code", ""),
            "stack_trace": error_data.get("stack_trace", []),
            "context": error_data.get("context", {}),
            "timestamp": error_data.get("timestamp"),
            "severity": error_data.get("severity", "medium")
        }
        
        # Add any additional fields from the original error
        for key, value in error_data.items():
            if key not in normalized and value is not None:
                normalized[key] = value
        
        return normalized
    
    def denormalize_error(self, standard_error: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert standard format error data back to the MATLAB-specific format.
        
        Args:
            standard_error: Error data in the standard format
            
        Returns:
            Error data in the MATLAB-specific format
        """
        # Map standard fields back to MATLAB-specific format
        matlab_error = {
            "error_type": standard_error.get("error_type", "MATLABError"),
            "message": standard_error.get("message", ""),
            "file_path": standard_error.get("file_path", ""),
            "line_number": standard_error.get("line_number", 0),
            "column_number": standard_error.get("column_number", 0),
            "matlab_version": standard_error.get("matlab_version", ""),
            "source_code": standard_error.get("source_code", ""),
            "description": standard_error.get("message", ""),
            "file": standard_error.get("file_path", ""),
            "line": standard_error.get("line_number", 0),
            "column": standard_error.get("column_number", 0),
            "stack_trace": standard_error.get("stack_trace", []),
            "context": standard_error.get("context", {}),
            "timestamp": standard_error.get("timestamp"),
            "severity": standard_error.get("severity", "medium")
        }
        
        # Add any additional fields from the standard error
        for key, value in standard_error.items():
            if key not in matlab_error and value is not None:
                matlab_error[key] = value
        
        return matlab_error
    
    def analyze_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a MATLAB error.
        
        Args:
            error_data: MATLAB error data
            
        Returns:
            Analysis results
        """
        try:
            # Ensure error data is in standard format
            if not error_data.get("language"):
                standard_error = self.normalize_error(error_data)
            else:
                standard_error = error_data
            
            # Analyze the error
            analysis = self.exception_handler.analyze_exception(standard_error)
            
            # Add plugin metadata
            analysis["plugin"] = "matlab"
            analysis["language"] = "matlab"
            analysis["plugin_version"] = self.VERSION
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing MATLAB error: {e}")
            return {
                "category": "matlab",
                "subcategory": "unknown",
                "confidence": "low",
                "suggested_fix": "Unable to analyze MATLAB error",
                "error": str(e),
                "plugin": "matlab"
            }
    
    def generate_fix(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a fix for an error based on the analysis.
        
        Args:
            analysis: Error analysis
            context: Additional context for fix generation
            
        Returns:
            Generated fix data
        """
        error_data = context.get("error_data", {})
        source_code = context.get("source_code", "")
        
        fix = self.patch_generator.generate_patch(error_data, analysis, source_code)
        
        if fix:
            return fix
        else:
            return {
                "type": "suggestion",
                "description": analysis.get("suggested_fix", "No specific fix available"),
                "confidence": analysis.get("confidence", "low")
            }


# Register the plugin
register_plugin(MATLABLanguagePlugin())