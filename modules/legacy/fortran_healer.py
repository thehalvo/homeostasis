"""
Fortran language healing implementation for Homeostasis.

Provides error detection and healing for Fortran programs including
both legacy FORTRAN 77 and modern Fortran 90/95/2003/2008/2018.
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class FortranStandard(Enum):
    """Fortran language standards."""

    F66 = "FORTRAN 66"
    F77 = "FORTRAN 77"
    F90 = "Fortran 90"
    F95 = "Fortran 95"
    F2003 = "Fortran 2003"
    F2008 = "Fortran 2008"
    F2018 = "Fortran 2018"


@dataclass
class FortranError:
    """Represents a Fortran compilation or runtime error."""

    error_code: str
    severity: str  # error, warning, info
    line_number: Optional[int]
    column_number: Optional[int]
    message: str
    source_line: Optional[str]
    compiler: str  # gfortran, ifort, pgfortran, etc.
    standard: FortranStandard
    category: str


class FortranHealer:
    """
    Healer for Fortran language errors.

    Handles compilation errors, runtime errors, and helps modernize
    legacy FORTRAN code to newer standards.
    """

    def __init__(self, standard: FortranStandard = FortranStandard.F2008):
        self.standard = standard
        self._error_patterns = self._compile_error_patterns()
        self._healing_rules = self._load_healing_rules()
        self._modernization_rules = self._load_modernization_rules()

    def _compile_error_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regular expressions for error parsing."""
        return {
            # GNU Fortran (gfortran) format
            "gfortran_error": re.compile(
                r"^([^:]+):(\d+):(\d+):\s*(Error|Warning|Info):\s+(.*)$"
            ),
            # Intel Fortran (ifort) format
            "ifort_error": re.compile(
                r"^([^(]+)\((\d+)\):\s*(error|warning|remark)\s+#(\d+):\s+(.*)$"
            ),
            # PGI/NVIDIA Fortran format
            "pgfortran_error": re.compile(r"^PGF90-([EWIS])-(\d+)-(.*)$"),
            # IBM XL Fortran format
            "xlf_error": re.compile(
                r"^\"([^\"]+)\",\s*line\s+(\d+)\.\d+:\s*(\d+)-(\d+)\s+\(([EWIS])\)\s+(.*)$"
            ),
            # Common patterns
            "line_continuation": re.compile(r"^\s{5}[^\s]"),
            "fixed_format": re.compile(r"^[cC*!]\s*.*$"),
            "statement_label": re.compile(r"^\s{0,4}\d+\s+"),
        }

    def _load_healing_rules(self) -> Dict[str, Dict[str, Any]]:
        """Load Fortran healing rules."""
        return {
            # Syntax errors
            "syntax_continuation": {
                "description": "Invalid continuation line",
                "category": "syntax",
                "healing": {
                    "action": "fix_continuation",
                    "f77_fix": "Add continuation character in column 6",
                    "f90_fix": "Add & at end of previous line",
                    "automated": True,
                },
            },
            "syntax_statement_order": {
                "description": "Incorrect statement order",
                "category": "syntax",
                "healing": {"action": "reorder_statements", "automated": False},
            },
            # Type errors
            "type_mismatch": {
                "description": "Type mismatch in assignment or call",
                "category": "type",
                "healing": {"action": "add_type_conversion", "automated": True},
            },
            "implicit_type": {
                "description": "Implicit typing used",
                "category": "type",
                "healing": {"action": "add_explicit_declaration", "automated": True},
            },
            # Array errors
            "array_bounds": {
                "description": "Array index out of bounds",
                "category": "array",
                "healing": {"action": "add_bounds_check", "automated": True},
            },
            "array_conformance": {
                "description": "Array shape mismatch",
                "category": "array",
                "healing": {"action": "check_array_shapes", "automated": False},
            },
            # I/O errors
            "io_unit": {
                "description": "Invalid I/O unit number",
                "category": "io",
                "healing": {"action": "fix_unit_number", "automated": True},
            },
            "io_format": {
                "description": "Format specification error",
                "category": "io",
                "healing": {"action": "fix_format_spec", "automated": False},
            },
            # Common block errors
            "common_alignment": {
                "description": "COMMON block alignment error",
                "category": "memory",
                "healing": {"action": "reorder_common_vars", "automated": True},
            },
            # Module errors
            "module_not_found": {
                "description": "Module not found",
                "category": "module",
                "healing": {"action": "add_use_statement", "automated": False},
            },
            # Floating point errors
            "floating_overflow": {
                "description": "Floating point overflow",
                "category": "numeric",
                "healing": {"action": "add_overflow_check", "automated": True},
            },
            "division_by_zero": {
                "description": "Division by zero",
                "category": "numeric",
                "healing": {"action": "add_zero_check", "automated": True},
            },
        }

    def _load_modernization_rules(self) -> Dict[str, Dict[str, Any]]:
        """Load rules for modernizing Fortran code."""
        return {
            "fixed_to_free": {
                "description": "Convert fixed-form to free-form",
                "from": FortranStandard.F77,
                "to": FortranStandard.F90,
                "automated": True,
            },
            "common_to_module": {
                "description": "Replace COMMON blocks with modules",
                "from": FortranStandard.F77,
                "to": FortranStandard.F90,
                "automated": True,
            },
            "goto_elimination": {
                "description": "Replace GOTO with structured constructs",
                "from": FortranStandard.F77,
                "to": FortranStandard.F90,
                "automated": False,
            },
            "implicit_none": {
                "description": "Add IMPLICIT NONE",
                "from": FortranStandard.F77,
                "to": FortranStandard.F90,
                "automated": True,
            },
            "arithmetic_if": {
                "description": "Replace arithmetic IF with block IF",
                "from": FortranStandard.F77,
                "to": FortranStandard.F90,
                "automated": True,
            },
            "do_loop_modernization": {
                "description": "Modernize DO loops",
                "from": FortranStandard.F77,
                "to": FortranStandard.F90,
                "automated": True,
            },
            "character_declarations": {
                "description": "Modernize CHARACTER declarations",
                "from": FortranStandard.F77,
                "to": FortranStandard.F90,
                "automated": True,
            },
            "array_syntax": {
                "description": "Use array syntax instead of loops",
                "from": FortranStandard.F77,
                "to": FortranStandard.F90,
                "automated": True,
            },
        }

    def parse_error(
        self, error_text: str, compiler: str = "gfortran"
    ) -> Optional[FortranError]:
        """Parse Fortran compiler error message."""
        error = None

        if compiler == "gfortran":
            match = self._error_patterns["gfortran_error"].match(error_text.strip())
            if match:
                line_num = int(match.group(2))
                col_num = int(match.group(3))
                severity = match.group(4).lower()
                message = match.group(5)

                error = FortranError(
                    error_code=self._extract_error_code(message),
                    severity=severity,
                    line_number=line_num,
                    column_number=col_num,
                    message=message,
                    source_line=None,
                    compiler=compiler,
                    standard=self.standard,
                    category=self._categorize_error(message),
                )

        elif compiler == "ifort":
            match = self._error_patterns["ifort_error"].match(error_text.strip())
            if match:
                line_num = int(match.group(2))
                severity = match.group(3)
                error_num = match.group(4)
                message = match.group(5)

                error = FortranError(
                    error_code=f"IFORT{error_num}",
                    severity=severity,
                    line_number=line_num,
                    column_number=None,
                    message=message,
                    source_line=None,
                    compiler=compiler,
                    standard=self.standard,
                    category=self._categorize_error(message),
                )

        return error

    def _extract_error_code(self, message: str) -> str:
        """Extract or generate error code from message."""
        # Look for common error patterns
        if "type mismatch" in message.lower():
            return "TYPE_MISMATCH"
        elif "undefined" in message.lower():
            return "UNDEFINED_VAR"
        elif "syntax error" in message.lower():
            return "SYNTAX_ERROR"
        elif "array" in message.lower():
            return "ARRAY_ERROR"
        else:
            return f"ERR_{hash(message) % 10000}"

    def _categorize_error(self, message: str) -> str:
        """Categorize error based on message content."""
        message_lower = message.lower()

        if "type" in message_lower or "mismatch" in message_lower:
            return "type"
        elif "array" in message_lower or "dimension" in message_lower:
            return "array"
        elif "undefined" in message_lower or "undeclared" in message_lower:
            return "declaration"
        elif "syntax" in message_lower:
            return "syntax"
        elif "i/o" in message_lower or "format" in message_lower:
            return "io"
        elif "module" in message_lower or "use" in message_lower:
            return "module"
        elif "common" in message_lower:
            return "memory"
        elif "float" in message_lower or "overflow" in message_lower:
            return "numeric"
        else:
            return "general"

    def analyze_source(self, source_code: str) -> List[Dict[str, Any]]:
        """Analyze Fortran source code for issues and modernization opportunities."""
        issues = []
        lines = source_code.split("\n")

        # Detect format (fixed vs free)
        is_fixed_format = self._is_fixed_format(lines)

        # Track program structure
        in_program = False
        in_subroutine = False
        in_function = False
        implicit_none_found = False
        common_blocks = []
        goto_count = 0

        for i, line in enumerate(lines, 1):
            # Skip comments
            if is_fixed_format and line and line[0] in "cC*!":
                continue
            elif not is_fixed_format and line.strip().startswith("!"):
                continue

            line_upper = line.upper()

            # Check for program units
            if re.search(r"\bPROGRAM\s+\w+", line_upper):
                in_program = True
            elif re.search(r"\bSUBROUTINE\s+\w+", line_upper):
                in_subroutine = True
            elif re.search(r"\b(FUNCTION|.*FUNCTION)\s+\w+", line_upper):
                in_function = True

            # Check for IMPLICIT NONE
            if "IMPLICIT NONE" in line_upper:
                implicit_none_found = True

            # Check for legacy features
            if not implicit_none_found and (in_program or in_subroutine or in_function):
                issues.append(
                    {
                        "line": i,
                        "type": "modernization",
                        "message": "Missing IMPLICIT NONE statement",
                        "severity": "warning",
                        "fix": "Add IMPLICIT NONE after program unit declaration",
                    }
                )
                implicit_none_found = False  # Reset for next program unit

            # Check for COMMON blocks
            if re.search(r"\bCOMMON\s*/?\w*/", line_upper):
                common_blocks.append(i)
                issues.append(
                    {
                        "line": i,
                        "type": "legacy",
                        "message": "COMMON block usage (consider using modules)",
                        "severity": "info",
                    }
                )

            # Check for GOTO statements
            if re.search(r"\bGOTO\s+\d+", line_upper):
                goto_count += 1
                issues.append(
                    {
                        "line": i,
                        "type": "legacy",
                        "message": "GOTO statement (use structured constructs)",
                        "severity": "warning",
                    }
                )

            # Check for arithmetic IF
            if re.search(r"\bIF\s*\([^)]+\)\s*\d+\s*,\s*\d+\s*,\s*\d+", line_upper):
                issues.append(
                    {
                        "line": i,
                        "type": "legacy",
                        "message": "Arithmetic IF statement (use block IF)",
                        "severity": "warning",
                    }
                )

            # Check for equivalence statements
            if "EQUIVALENCE" in line_upper:
                issues.append(
                    {
                        "line": i,
                        "type": "legacy",
                        "message": "EQUIVALENCE statement (error-prone)",
                        "severity": "warning",
                    }
                )

            # Check for fixed-form specific issues
            if is_fixed_format:
                # Check column 6 for continuation
                if len(line) > 5 and line[5] not in " 0":
                    if i > 1 and not self._is_valid_continuation(lines[i - 2], line):
                        issues.append(
                            {
                                "line": i,
                                "type": "syntax",
                                "message": "Invalid continuation line",
                                "severity": "error",
                            }
                        )

                # Check for code beyond column 72
                if len(line) > 72 and line[72:].strip():
                    issues.append(
                        {
                            "line": i,
                            "type": "format",
                            "message": "Code beyond column 72 (will be ignored)",
                            "severity": "warning",
                        }
                    )

            # Check for obsolete features
            if re.search(r"\bPAUSE\b", line_upper):
                issues.append(
                    {
                        "line": i,
                        "type": "obsolete",
                        "message": "PAUSE statement is obsolete",
                        "severity": "warning",
                    }
                )

            if re.search(r"\bASSIGN\s+\d+\s+TO", line_upper):
                issues.append(
                    {
                        "line": i,
                        "type": "obsolete",
                        "message": "ASSIGN statement is obsolete",
                        "severity": "warning",
                    }
                )

        return issues

    def _is_fixed_format(self, lines: List[str]) -> bool:
        """Detect if source is in fixed format."""
        # Look for typical fixed-format indicators
        for line in lines[:50]:  # Check first 50 lines
            if line:
                # Comment in column 1
                if line[0] in "cC*!":
                    return True
                # Statement label in columns 1-5
                if re.match(r"^\s{0,4}\d+\s+\w", line):
                    return True
                # Continuation in column 6
                if len(line) > 5 and line[5] not in " 0\t":
                    return True

        return False

    def _is_valid_continuation(self, prev_line: str, cont_line: str) -> bool:
        """Check if continuation line is valid."""
        # Previous line should have content
        if not prev_line or not prev_line.strip():
            return False

        # In fixed format, continuation char should be in column 6
        if len(cont_line) > 5 and cont_line[5] not in " 0":
            return True

        return False

    def generate_fix(self, error: FortranError, source_code: str) -> Optional[str]:
        """Generate fix for Fortran error."""
        lines = source_code.split("\n")

        if error.line_number and 0 < error.line_number <= len(lines):
            error_line_idx = error.line_number - 1

            # Fix based on error category
            if error.category == "type" and "implicit" in error.message.lower():
                # Add IMPLICIT NONE
                return self._add_implicit_none(lines, error_line_idx)

            elif error.category == "syntax" and "continuation" in error.message.lower():
                # Fix continuation line
                return self._fix_continuation(lines, error_line_idx)

            elif error.category == "numeric" and "division" in error.message.lower():
                # Add zero check
                return self._add_zero_check(lines, error_line_idx)

            elif error.category == "array" and "bounds" in error.message.lower():
                # Add bounds check
                return self._add_bounds_check(lines, error_line_idx)

        return None

    def _add_implicit_none(self, lines: List[str], error_idx: int) -> str:
        """Add IMPLICIT NONE statement."""
        # Find program unit start
        for i in range(error_idx, -1, -1):
            line_upper = lines[i].upper()
            if any(
                keyword in line_upper
                for keyword in ["PROGRAM", "SUBROUTINE", "FUNCTION", "MODULE"]
            ):
                # Insert IMPLICIT NONE after program unit declaration
                lines.insert(i + 1, "      IMPLICIT NONE")
                return "\n".join(lines)

        return None

    def _fix_continuation(self, lines: List[str], error_idx: int) -> str:
        """Fix continuation line issues."""
        if error_idx > 0:
            is_fixed = self._is_fixed_format(lines)

            if is_fixed:
                # Ensure continuation character in column 6
                cont_line = lines[error_idx]
                if len(cont_line) >= 6:
                    fixed_line = cont_line[:5] + "&" + cont_line[6:]
                else:
                    fixed_line = cont_line.ljust(5) + "&"
                lines[error_idx] = fixed_line
            else:
                # Free format - add & at end of previous line
                prev_line = lines[error_idx - 1].rstrip()
                if not prev_line.endswith("&"):
                    lines[error_idx - 1] = prev_line + " &"

            return "\n".join(lines)

        return None

    def _add_zero_check(self, lines: List[str], error_idx: int) -> str:
        """Add check for division by zero."""
        error_line = lines[error_idx]

        # Look for division operation
        div_match = re.search(r"(\w+)\s*=\s*([^/]+)/\s*(\w+)", error_line)
        if div_match:
            result_var = div_match.group(1)
            denominator = div_match.group(3)

            # Create IF block
            indent = len(error_line) - len(error_line.lstrip())
            check_lines = [
                " " * indent + f"IF ({denominator} /= 0.0) THEN",
                error_line,
                " " * indent + "ELSE",
                " " * indent + f"  {result_var} = 0.0  ! or appropriate error handling",
                " " * indent + "END IF",
            ]

            # Replace original line with IF block
            lines[error_idx:error_idx + 1] = check_lines
            return "\n".join(lines)

        return None

    def _add_bounds_check(self, lines: List[str], error_idx: int) -> str:
        """Add array bounds checking."""
        error_line = lines[error_idx]

        # Look for array access
        array_match = re.search(r"(\w+)\(([^)]+)\)", error_line)
        if array_match:
            array_name = array_match.group(1)
            index_expr = array_match.group(2)

            # Create bounds check
            indent = len(error_line) - len(error_line.lstrip())
            check_lines = [
                " " * indent + f"! Add bounds check for {array_name}",
                " " * indent + f"IF ({index_expr} >= LBOUND({array_name},1) .AND. &",
                " " * indent + f"    {index_expr} <= UBOUND({array_name},1)) THEN",
                error_line,
                " " * indent + "ELSE",
                " " * indent
                + f"  PRINT *, 'Array index out of bounds for {array_name}'",
                " " * indent + "  STOP",
                " " * indent + "END IF",
            ]

            lines[error_idx:error_idx + 1] = check_lines
            return "\n".join(lines)

        return None

    def modernize_code(
        self, source_code: str, target_standard: FortranStandard = None
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Modernize Fortran code to newer standard."""
        if target_standard is None:
            target_standard = FortranStandard.F2008

        modifications = []
        lines = source_code.split("\n")
        is_fixed = self._is_fixed_format(lines)

        # Convert fixed to free format if needed
        if is_fixed and target_standard.value >= FortranStandard.F90.value:
            lines, mods = self._convert_to_free_format(lines)
            modifications.extend(mods)

        # Apply modernization rules
        lines, mods = self._replace_common_blocks(lines)
        modifications.extend(mods)

        lines, mods = self._modernize_do_loops(lines)
        modifications.extend(mods)

        lines, mods = self._replace_arithmetic_if(lines)
        modifications.extend(mods)

        lines, mods = self._add_implicit_none_everywhere(lines)
        modifications.extend(mods)

        lines, mods = self._modernize_io_statements(lines)
        modifications.extend(mods)

        lines, mods = self._use_array_syntax(lines)
        modifications.extend(mods)

        return "\n".join(lines), modifications

    def _convert_to_free_format(
        self, lines: List[str]
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Convert fixed format to free format."""
        new_lines = []
        modifications = []

        for i, line in enumerate(lines):
            if not line:
                new_lines.append(line)
                continue

            # Handle comments
            if line[0] in "cC*!":
                new_line = "!" + line[1:]
                new_lines.append(new_line)
                modifications.append(
                    {
                        "line": i + 1,
                        "type": "format",
                        "original": line,
                        "modified": new_line,
                        "description": "Convert comment to free format",
                    }
                )
                continue

            # Handle statement labels
            label_match = re.match(r"^(\s{0,4})(\d+)\s+(.*)$", line)
            if label_match:
                label = label_match.group(2)
                statement = label_match.group(3)
                new_line = f"{label} {statement}"
                new_lines.append(new_line)
                modifications.append(
                    {
                        "line": i + 1,
                        "type": "format",
                        "original": line,
                        "modified": new_line,
                        "description": "Convert statement label",
                    }
                )
                continue

            # Handle continuation lines
            if len(line) > 5 and line[5] not in " 0":
                # Remove column 6 continuation and use &
                new_line = line[:5] + " " + line[6:]
                new_lines.append(new_line.strip())
                if i > 0 and not new_lines[-2].endswith("&"):
                    new_lines[-2] += " &"
                modifications.append(
                    {
                        "line": i + 1,
                        "type": "format",
                        "original": line,
                        "modified": new_line,
                        "description": "Convert continuation line",
                    }
                )
                continue

            # Regular statement - remove column restrictions
            if len(line) > 6:
                new_line = line[6:72].rstrip()
            else:
                new_line = line

            new_lines.append(new_line)

        return new_lines, modifications

    def _replace_common_blocks(
        self, lines: List[str]
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Replace COMMON blocks with modules."""
        modifications = []
        common_blocks = {}

        # First pass - collect COMMON blocks
        for i, line in enumerate(lines):
            common_match = re.match(r"\s*COMMON\s*/(\w+)/\s*(.+)$", line.upper())
            if common_match:
                block_name = common_match.group(1)
                var_list = common_match.group(2)

                if block_name not in common_blocks:
                    common_blocks[block_name] = []
                common_blocks[block_name].append({"line": i, "vars": var_list})

        # Generate module for each COMMON block
        module_lines = []
        for block_name, occurrences in common_blocks.items():
            module_lines.extend(
                [f"MODULE {block_name}_MODULE", "  IMPLICIT NONE", "  SAVE"]
            )

            # Collect all variables
            all_vars = set()
            for occ in occurrences:
                vars = [v.strip() for v in occ["vars"].split(",")]
                all_vars.update(vars)

            # Add variable declarations (simplified - would need type info)
            for var in sorted(all_vars):
                module_lines.append(f"  REAL :: {var}  ! Type needs to be determined")

            module_lines.extend([f"END MODULE {block_name}_MODULE", ""])

            modifications.append(
                {
                    "type": "modernization",
                    "description": f"Create module for COMMON block /{block_name}/",
                    "new_code": "\n".join(module_lines),
                }
            )

        return lines, modifications

    def _modernize_do_loops(
        self, lines: List[str]
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Modernize DO loops."""
        modifications = []
        new_lines = lines.copy()

        for i, line in enumerate(lines):
            # Old style: DO label var = start, end, step
            old_do_match = re.match(
                r"(\s*)DO\s+(\d+)\s+(\w+)\s*=\s*(.+)$", line, re.IGNORECASE
            )
            if old_do_match:
                indent = old_do_match.group(1)
                label = old_do_match.group(2)
                var = old_do_match.group(3)
                range_expr = old_do_match.group(4)

                # Convert to modern DO
                new_line = f"{indent}DO {var} = {range_expr}"
                new_lines[i] = new_line

                # Find corresponding labeled CONTINUE
                for j in range(i + 1, len(lines)):
                    if re.match(rf"^\s*{label}\s+CONTINUE", lines[j], re.IGNORECASE):
                        new_lines[j] = f"{indent}END DO"
                        break

                modifications.append(
                    {
                        "line": i + 1,
                        "type": "modernization",
                        "original": line,
                        "modified": new_line,
                        "description": "Modernize DO loop",
                    }
                )

        return new_lines, modifications

    def _replace_arithmetic_if(
        self, lines: List[str]
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Replace arithmetic IF with block IF."""
        modifications = []
        new_lines = lines.copy()

        for i, line in enumerate(lines):
            # Arithmetic IF: IF (expr) label1, label2, label3
            arith_if_match = re.match(
                r"(\s*)IF\s*\(([^)]+)\)\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)",
                line,
                re.IGNORECASE,
            )
            if arith_if_match:
                indent = arith_if_match.group(1)
                expr = arith_if_match.group(2)
                label_neg = arith_if_match.group(3)
                label_zero = arith_if_match.group(4)
                label_pos = arith_if_match.group(5)

                # Convert to block IF
                block_if = [
                    f"{indent}IF ({expr} < 0) THEN",
                    f"{indent}  GOTO {label_neg}",
                    f"{indent}ELSE IF ({expr} == 0) THEN",
                    f"{indent}  GOTO {label_zero}",
                    f"{indent}ELSE",
                    f"{indent}  GOTO {label_pos}",
                    f"{indent}END IF",
                ]

                # Note: This still uses GOTO, but structured
                # A better modernization would eliminate GOTOs entirely

                modifications.append(
                    {
                        "line": i + 1,
                        "type": "modernization",
                        "original": line,
                        "suggestion": "Replace arithmetic IF with block IF",
                        "new_code": "\n".join(block_if),
                    }
                )

        return new_lines, modifications

    def _add_implicit_none_everywhere(
        self, lines: List[str]
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Add IMPLICIT NONE to all program units."""
        modifications = []
        new_lines = lines.copy()

        i = 0
        while i < len(new_lines):
            line_upper = new_lines[i].upper()

            # Check for program unit start
            if any(
                re.search(rf"\b{keyword}\s+\w+", line_upper)
                for keyword in ["PROGRAM", "SUBROUTINE", "FUNCTION", "MODULE"]
            ):
                # Check if IMPLICIT NONE already exists
                has_implicit_none = False
                for j in range(i + 1, min(i + 10, len(new_lines))):
                    if "IMPLICIT NONE" in new_lines[j].upper():
                        has_implicit_none = True
                        break
                    # Stop at next program unit or executable statement
                    if any(
                        keyword in new_lines[j].upper()
                        for keyword in ["PROGRAM", "SUBROUTINE", "FUNCTION", "END"]
                    ):
                        break

                if not has_implicit_none:
                    # Insert IMPLICIT NONE
                    indent = len(new_lines[i]) - len(new_lines[i].lstrip())
                    new_lines.insert(i + 1, " " * indent + "  IMPLICIT NONE")
                    modifications.append(
                        {
                            "line": i + 1,
                            "type": "modernization",
                            "description": "Add IMPLICIT NONE",
                        }
                    )
                    i += 1  # Skip the inserted line

            i += 1

        return new_lines, modifications

    def _modernize_io_statements(
        self, lines: List[str]
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Modernize I/O statements."""
        modifications = []
        new_lines = lines.copy()

        for i, line in enumerate(lines):
            # Old style: WRITE(6,*) -> WRITE(*,*)
            old_write = re.sub(
                r"WRITE\s*\(\s*6\s*,", "WRITE(*,", line, flags=re.IGNORECASE
            )
            if old_write != line:
                new_lines[i] = old_write
                modifications.append(
                    {
                        "line": i + 1,
                        "type": "modernization",
                        "original": line,
                        "modified": old_write,
                        "description": "Use * for standard output",
                    }
                )

            # Old style: READ(5,*) -> READ(*,*)
            old_read = re.sub(
                r"READ\s*\(\s*5\s*,", "READ(*,", line, flags=re.IGNORECASE
            )
            if old_read != line:
                new_lines[i] = old_read
                modifications.append(
                    {
                        "line": i + 1,
                        "type": "modernization",
                        "original": line,
                        "modified": old_read,
                        "description": "Use * for standard input",
                    }
                )

        return new_lines, modifications

    def _use_array_syntax(
        self, lines: List[str]
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Suggest using array syntax instead of loops."""
        modifications = []

        i = 0
        while i < len(lines):
            # Look for simple array initialization loops
            if re.match(r"\s*DO\s+\w+\s*=", lines[i], re.IGNORECASE):
                # Check if it's a simple array assignment
                loop_var = None
                array_name = None

                do_match = re.match(
                    r"\s*DO\s+(\w+)\s*=\s*1\s*,\s*(\w+)", lines[i], re.IGNORECASE
                )
                if do_match:
                    loop_var = do_match.group(1)

                    # Look for array assignment in next lines
                    for j in range(i + 1, min(i + 5, len(lines))):
                        assign_match = re.match(
                            rf"\s*(\w+)\s*\(\s*{loop_var}\s*\)\s*=\s*(.+)",
                            lines[j],
                            re.IGNORECASE,
                        )
                        if assign_match:
                            array_name = assign_match.group(1)
                            value = assign_match.group(2)

                            # Suggest array syntax
                            modifications.append(
                                {
                                    "line": i + 1,
                                    "type": "modernization",
                                    "description": "Use array syntax",
                                    "original": f"DO loop for {array_name}",
                                    "suggestion": f"{array_name} = {value}  ! Array syntax",
                                }
                            )
                            break

            i += 1

        return lines, modifications

    def get_healing_suggestions(self, error: FortranError) -> List[Dict[str, Any]]:
        """Get healing suggestions for Fortran error."""
        suggestions = []

        # Category-specific suggestions
        if error.category == "type":
            suggestions.extend(
                [
                    {
                        "title": "Add Explicit Type Declaration",
                        "description": "Declare all variables explicitly",
                        "automated": True,
                        "action": "add_declaration",
                    },
                    {
                        "title": "Add IMPLICIT NONE",
                        "description": "Disable implicit typing",
                        "automated": True,
                        "action": "add_implicit_none",
                    },
                ]
            )

        elif error.category == "array":
            suggestions.extend(
                [
                    {
                        "title": "Add Bounds Checking",
                        "description": "Check array indices before access",
                        "automated": True,
                        "action": "add_bounds_check",
                    },
                    {
                        "title": "Use Allocatable Arrays",
                        "description": "Convert to allocatable for dynamic sizing",
                        "automated": False,
                        "action": "use_allocatable",
                    },
                ]
            )

        elif error.category == "io":
            suggestions.extend(
                [
                    {
                        "title": "Add IOSTAT Check",
                        "description": "Check I/O operation status",
                        "automated": True,
                        "action": "add_iostat",
                        "template": "IOSTAT=ios",
                    },
                    {
                        "title": "Modernize I/O",
                        "description": "Use modern I/O features",
                        "automated": True,
                        "action": "modernize_io",
                    },
                ]
            )

        elif error.category == "numeric":
            suggestions.extend(
                [
                    {
                        "title": "Add Overflow Protection",
                        "description": "Check for numeric overflow",
                        "automated": True,
                        "action": "add_overflow_check",
                    },
                    {
                        "title": "Use IEEE Arithmetic",
                        "description": "Enable IEEE exception handling",
                        "automated": False,
                        "action": "use_ieee_arithmetic",
                    },
                ]
            )

        # Add general modernization suggestions
        if self.standard <= FortranStandard.F77:
            suggestions.append(
                {
                    "title": "Modernize to Fortran 90+",
                    "description": "Update code to modern Fortran standards",
                    "automated": True,
                    "action": "modernize_code",
                }
            )

        return suggestions
