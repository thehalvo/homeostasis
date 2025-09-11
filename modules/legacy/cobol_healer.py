"""
COBOL language healing implementation for Homeostasis.

Provides error detection and healing for COBOL programs including
both mainframe and distributed COBOL variants.
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class COBOLDialect(Enum):
    """COBOL dialect variations."""

    IBM_ENTERPRISE = "IBM Enterprise COBOL"
    IBM_VS = "IBM VS COBOL II"
    MICRO_FOCUS = "Micro Focus COBOL"
    GNU_COBOL = "GnuCOBOL"
    FUJITSU = "Fujitsu COBOL"
    ACUCOBOL = "ACUCOBOL-GT"


@dataclass
class COBOLError:
    """Represents a COBOL compilation or runtime error."""

    error_code: str
    severity: str  # I=Informational, W=Warning, E=Error, S=Severe
    line_number: Optional[int]
    column_number: Optional[int]
    message: str
    source_line: Optional[str]
    program_id: Optional[str]
    dialect: COBOLDialect
    category: str


class COBOLHealer:
    """
    Healer for COBOL language errors.

    Handles compilation errors, runtime errors, and common COBOL
    programming issues across different dialects.
    """

    def __init__(self, dialect: COBOLDialect = COBOLDialect.IBM_ENTERPRISE):
        self.dialect = dialect
        self._error_patterns = self._compile_error_patterns()
        self._healing_rules = self._load_healing_rules()

    def _compile_error_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regular expressions for error parsing."""
        patterns = {
            # IBM Enterprise COBOL compiler messages
            "ibm_error": re.compile(r"^(IGY[A-Z]\d{4}[I|W|E|S])-(\d+)\s+(.*)$"),
            # Micro Focus error format
            "mf_error": re.compile(
                r"^\*\*\s+(\w+)-([IWES])-(\d+):\s+\((\d+),(\d+)\)\s+(.*)$"
            ),
            # GnuCOBOL error format
            "gnu_error": re.compile(r"^([^:]+):(\d+):\s+(warning|error):\s+(.*)$"),
            # Line number extraction
            "line_ref": re.compile(r"LINE\s+(\d+)"),
            # Program ID extraction
            "program_id": re.compile(r"PROGRAM-ID\.\s+(\w+)"),
        }
        return patterns

    def _load_healing_rules(self) -> Dict[str, Dict[str, Any]]:
        """Load COBOL healing rules."""
        return {
            # Data division errors
            "IGYDS1003": {
                "description": "PICTURE clause required",
                "category": "data_definition",
                "healing": {"action": "add_picture_clause", "template": "PIC X(30)"},
            },
            "IGYDS1082": {
                "description": "VALUE clause not allowed with OCCURS",
                "category": "data_definition",
                "healing": {"action": "remove_value_clause", "automated": True},
            },
            "IGYDS1089": {
                "description": "REDEFINES not valid at 01 level in FILE SECTION",
                "category": "data_definition",
                "healing": {"action": "move_to_working_storage", "automated": False},
            },
            # Procedure division errors
            "IGYPS2072": {
                "description": "Period required",
                "category": "syntax",
                "healing": {"action": "add_period", "automated": True},
            },
            "IGYPS2121": {
                "description": "Receiving field must be numeric",
                "category": "type_mismatch",
                "healing": {"action": "check_data_types", "automated": False},
            },
            "IGYPS2037": {
                "description": "Subscript out of range",
                "category": "runtime",
                "healing": {
                    "action": "add_bounds_check",
                    "template": "IF sub-var > table-size THEN...",
                },
            },
            # File handling errors
            "IGYPS2145": {
                "description": "File not open",
                "category": "file_handling",
                "healing": {"action": "add_open_statement", "automated": True},
            },
            # SQL errors in COBOL
            "DSNH103I": {
                "description": "SQL syntax error",
                "category": "embedded_sql",
                "healing": {"action": "fix_sql_syntax", "automated": False},
            },
            # Runtime errors
            "IGZ0006S": {
                "description": "Missing DD statement",
                "category": "jcl_error",
                "healing": {"action": "add_dd_statement", "automated": True},
            },
            "IGZ0035S": {
                "description": "File status 35 - file not found",
                "category": "file_error",
                "healing": {"action": "check_file_allocation", "automated": False},
            },
            "IGZ0201W": {
                "description": "Numeric overflow",
                "category": "arithmetic",
                "healing": {"action": "increase_field_size", "automated": True},
            },
        }

    def parse_error(self, error_text: str) -> Optional[COBOLError]:
        """Parse COBOL compiler or runtime error message."""
        error = None

        # Try IBM format
        if self.dialect in [COBOLDialect.IBM_ENTERPRISE, COBOLDialect.IBM_VS]:
            match = self._error_patterns["ibm_error"].match(error_text.strip())
            if match:
                error_code = match.group(1)
                line_num = int(match.group(2))
                message = match.group(3)
                severity = error_code[-1]  # Last character is severity

                error = COBOLError(
                    error_code=error_code[:-1],  # Remove severity suffix
                    severity=severity,
                    line_number=line_num,
                    column_number=None,
                    message=message,
                    source_line=None,
                    program_id=None,
                    dialect=self.dialect,
                    category=self._categorize_error(error_code),
                )

        # Try Micro Focus format
        elif self.dialect == COBOLDialect.MICRO_FOCUS:
            match = self._error_patterns["mf_error"].match(error_text.strip())
            if match:
                error_code = match.group(1)
                severity = match.group(2)
                line_num = int(match.group(4))
                col_num = int(match.group(5))
                message = match.group(6)

                error = COBOLError(
                    error_code=error_code,
                    severity=severity,
                    line_number=line_num,
                    column_number=col_num,
                    message=message,
                    source_line=None,
                    program_id=None,
                    dialect=self.dialect,
                    category=self._categorize_error(error_code),
                )

        # Try GnuCOBOL format
        elif self.dialect == COBOLDialect.GNU_COBOL:
            match = self._error_patterns["gnu_error"].match(error_text.strip())
            if match:
                filename = match.group(1)
                line_num = int(match.group(2))
                severity_text = match.group(3)
                message = match.group(4)

                severity = "E" if severity_text == "error" else "W"

                error = COBOLError(
                    error_code="GNU" + str(hash(message))[:4],
                    severity=severity,
                    line_number=line_num,
                    column_number=None,
                    message=message,
                    source_line=None,
                    program_id=filename,
                    dialect=self.dialect,
                    category=self._categorize_error_by_message(message),
                )

        return error

    def _categorize_error(self, error_code: str) -> str:
        """Categorize error based on error code."""
        if error_code.startswith("IGYDS"):
            return "data_definition"
        elif error_code.startswith("IGYPS"):
            return "procedure_division"
        elif error_code.startswith("IGYPG"):
            return "program_structure"
        elif error_code.startswith("IGYPP"):
            return "preprocessor"
        elif error_code.startswith("IGZ"):
            return "runtime"
        elif error_code.startswith("DSNH"):
            return "embedded_sql"
        else:
            return "general"

    def _categorize_error_by_message(self, message: str) -> str:
        """Categorize error based on message content."""
        message_lower = message.lower()

        if "syntax" in message_lower:
            return "syntax"
        elif "undefined" in message_lower or "not defined" in message_lower:
            return "undefined_reference"
        elif "type" in message_lower or "numeric" in message_lower:
            return "type_mismatch"
        elif "file" in message_lower:
            return "file_handling"
        elif "picture" in message_lower or "pic" in message_lower:
            return "data_definition"
        else:
            return "general"

    def analyze_source(self, source_code: str) -> List[Dict[str, Any]]:
        """Analyze COBOL source code for potential issues."""
        issues = []
        lines = source_code.split("\n")

        # Track program structure
        in_procedure = False

        # Common COBOL issues to check
        for i, line in enumerate(lines, 1):
            # Skip comment lines
            if len(line) > 6 and line[6] == "*":
                continue

            # Remove sequence area and indicator area
            if len(line) > 7:
                code_line = line[7:72]  # COBOL code area
            else:
                code_line = line

            code_upper = code_line.upper().strip()

            # Division tracking
            if "IDENTIFICATION DIVISION" in code_upper:
                in_procedure = False
            elif "DATA DIVISION" in code_upper:
                in_procedure = False
            elif "PROCEDURE DIVISION" in code_upper:
                in_procedure = True

            # Check for missing periods
            if (
                in_procedure
                and code_line.strip()
                and not code_line.rstrip().endswith(".")
            ):
                # Check if it's a statement that requires a period
                if any(
                    code_upper.startswith(stmt)
                    for stmt in ["MOVE", "COMPUTE", "PERFORM", "IF", "EVALUATE", "CALL"]
                ):
                    next_line_idx = i
                    while next_line_idx < len(lines):
                        next_line = lines[next_line_idx]
                        if len(next_line) > 7:
                            next_code = next_line[7:72].strip()
                            if next_code and not next_code.startswith(" "):
                                # New statement without period
                                issues.append(
                                    {
                                        "line": i,
                                        "type": "missing_period",
                                        "message": "Statement may be missing terminating period",
                                        "severity": "warning",
                                    }
                                )
                                break
                        next_line_idx += 1

            # Check for deprecated features
            if "ALTER" in code_upper:
                issues.append(
                    {
                        "line": i,
                        "type": "deprecated",
                        "message": "ALTER statement is deprecated",
                        "severity": "warning",
                        "suggestion": "Restructure logic to avoid ALTER",
                    }
                )

            if "GO TO" in code_upper and "DEPENDING ON" not in code_upper:
                issues.append(
                    {
                        "line": i,
                        "type": "style",
                        "message": "Unconditional GO TO should be avoided",
                        "severity": "info",
                        "suggestion": "Use structured programming constructs",
                    }
                )

            # Check for potential numeric issues
            if "MOVE" in code_upper:
                # Simple check for moving alphanumeric to numeric
                if re.search(r'MOVE\s+["\'].*["\']\s+TO\s+\w+-NBR', code_upper):
                    issues.append(
                        {
                            "line": i,
                            "type": "type_mismatch",
                            "message": "Moving alphanumeric literal to numeric field",
                            "severity": "error",
                        }
                    )

            # Check for unclosed strings
            quote_count = code_line.count('"') + code_line.count("'")
            if quote_count % 2 != 0:
                issues.append(
                    {
                        "line": i,
                        "type": "syntax",
                        "message": "Unclosed string literal",
                        "severity": "error",
                    }
                )

            # Check for SQL issues in EXEC SQL blocks
            if "EXEC SQL" in code_upper:
                if "END-EXEC" not in code_upper:
                    # Look for END-EXEC in following lines
                    found_end = False
                    for j in range(i, min(i + 20, len(lines))):
                        if "END-EXEC" in lines[j].upper():
                            found_end = True
                            break
                    if not found_end:
                        issues.append(
                            {
                                "line": i,
                                "type": "embedded_sql",
                                "message": "EXEC SQL block may be missing END-EXEC",
                                "severity": "error",
                            }
                        )

        return issues

    def generate_fix(self, error: COBOLError, source_code: str) -> Optional[str]:
        """Generate fix for COBOL error."""
        if error.error_code not in self._healing_rules:
            return None

        rule = self._healing_rules[error.error_code]
        if not rule["healing"].get("automated", False):
            return None

        lines = source_code.split("\n")

        if error.line_number and 0 < error.line_number <= len(lines):
            error_line_idx = error.line_number - 1
            error_line = lines[error_line_idx]

            action = rule["healing"]["action"]

            if action == "add_period":
                # Add period at end of statement
                if len(error_line) > 7:
                    code_area = error_line[7:72].rstrip()
                    if code_area and not code_area.endswith("."):
                        # Preserve original spacing
                        fixed_line = error_line[:7] + code_area + "."
                        if len(error_line) > 72:
                            fixed_line += error_line[72:]
                        lines[error_line_idx] = fixed_line
                        return "\n".join(lines)

            elif action == "add_picture_clause":
                # Add PICTURE clause to data definition
                if "PIC" not in error_line.upper():
                    template = rule["healing"].get("template", "PIC X(30)")
                    # Find position to insert
                    if len(error_line) > 7:
                        code_area = error_line[7:72]
                        # Look for level number and data name
                        match = re.match(r"^(\s*\d{2}\s+\w+)", code_area)
                        if match:
                            prefix = match.group(1)
                            fixed_line = error_line[:7] + prefix + f" {template}."
                            if len(error_line) > 72:
                                fixed_line += error_line[72:]
                            lines[error_line_idx] = fixed_line
                            return "\n".join(lines)

            elif action == "remove_value_clause":
                # Remove VALUE clause from OCCURS item
                if "VALUE" in error_line.upper():
                    # Simple removal of VALUE clause
                    value_pattern = re.compile(r"\s+VALUE\s+[^.]+", re.IGNORECASE)
                    if len(error_line) > 7:
                        code_area = error_line[7:72]
                        fixed_code = value_pattern.sub("", code_area)
                        fixed_line = error_line[:7] + fixed_code
                        if len(error_line) > 72:
                            fixed_line += error_line[72:]
                        lines[error_line_idx] = fixed_line
                        return "\n".join(lines)

            elif action == "add_open_statement":
                # Add file OPEN statement
                # This requires more context - find the file name and appropriate location
                # For now, return a suggestion
                return f"Add OPEN statement for file before line {error.line_number}"

            elif action == "increase_field_size":
                # Increase numeric field size
                if "PIC" in error_line.upper():
                    # Find and expand PICTURE clause
                    pic_pattern = re.compile(r"PIC\s+([S9]+)\((\d+)\)", re.IGNORECASE)
                    match = pic_pattern.search(error_line)
                    if match:
                        pic_type = match.group(1)
                        current_size = int(match.group(2))
                        new_size = min(
                            current_size * 2, 18
                        )  # Max 18 digits for numeric

                        fixed_line = pic_pattern.sub(
                            f"PIC {pic_type}({new_size})", error_line
                        )
                        lines[error_line_idx] = fixed_line
                        return "\n".join(lines)

        return None

    def get_healing_suggestions(self, error: COBOLError) -> List[Dict[str, Any]]:
        """Get healing suggestions for COBOL error."""
        suggestions = []

        # Check if we have a specific rule for this error
        if error.error_code in self._healing_rules:
            rule = self._healing_rules[error.error_code]
            suggestions.append(
                {
                    "title": rule["healing"]["action"].replace("_", " ").title(),
                    "description": rule["description"],
                    "automated": rule["healing"].get("automated", False),
                    "category": rule["category"],
                    "template": rule["healing"].get("template"),
                }
            )

        # Add general suggestions based on category
        if error.category == "data_definition":
            suggestions.extend(
                [
                    {
                        "title": "Verify PICTURE Clause",
                        "description": "Ensure all data items have appropriate PICTURE clauses",
                        "automated": False,
                        "category": "data_definition",
                    },
                    {
                        "title": "Check Level Numbers",
                        "description": "Verify level numbers are properly hierarchical",
                        "automated": False,
                        "category": "data_definition",
                    },
                ]
            )

        elif error.category == "type_mismatch":
            suggestions.extend(
                [
                    {
                        "title": "Add Type Conversion",
                        "description": "Use FUNCTION NUMVAL or NUMVAL-C for conversions",
                        "automated": False,
                        "category": "type_mismatch",
                        "template": "MOVE FUNCTION NUMVAL(source-field) TO target-field",
                    },
                    {
                        "title": "Initialize Numeric Fields",
                        "description": "Initialize numeric fields before use",
                        "automated": True,
                        "category": "type_mismatch",
                        "template": "INITIALIZE numeric-field",
                    },
                ]
            )

        elif error.category == "file_handling":
            suggestions.extend(
                [
                    {
                        "title": "Add File Status Check",
                        "description": "Check file status after each I/O operation",
                        "automated": False,
                        "category": "file_handling",
                        "template": "IF file-status NOT = '00' THEN...",
                    },
                    {
                        "title": "Verify File Declaration",
                        "description": "Ensure file is properly declared in FILE-CONTROL",
                        "automated": False,
                        "category": "file_handling",
                    },
                ]
            )

        elif error.category == "runtime":
            if "overflow" in error.message.lower():
                suggestions.append(
                    {
                        "title": "Add ON SIZE ERROR",
                        "description": "Handle arithmetic overflow conditions",
                        "automated": True,
                        "category": "runtime",
                        "template": "COMPUTE result = calculation ON SIZE ERROR...",
                    }
                )

        return suggestions

    def validate_fix(self, original_code: str, fixed_code: str) -> bool:
        """Validate that fix maintains COBOL structure."""
        # Basic validation - ensure we haven't broken COBOL format

        original_lines = original_code.split("\n")
        fixed_lines = fixed_code.split("\n")

        # Check line count hasn't changed dramatically
        if abs(len(original_lines) - len(fixed_lines)) > 10:
            logger.warning("Fix changed line count significantly")
            return False

        # Verify COBOL column structure is maintained
        for line in fixed_lines:
            if line and len(line) > 72:
                # Check if we have code beyond column 72
                if line[72:].strip() and not line[72:].isdigit():
                    logger.warning("Fix placed code beyond column 72")
                    return False

        # Check for balanced quotes
        for line in fixed_lines:
            if line.count('"') % 2 != 0 or line.count("'") % 2 != 0:
                logger.warning("Fix created unbalanced quotes")
                return False

        # Verify division structure
        divisions = [
            "IDENTIFICATION DIVISION",
            "ENVIRONMENT DIVISION",
            "DATA DIVISION",
            "PROCEDURE DIVISION",
        ]
        div_order = []

        for line in fixed_lines:
            line_upper = line.upper().strip()
            for div in divisions:
                if div in line_upper:
                    div_order.append(div)

        # Check divisions appear in correct order
        expected_order = [d for d in divisions if d in div_order]
        if div_order != expected_order:
            logger.warning("Fix disrupted division order")
            return False

        return True

    def modernize_code(self, source_code: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Modernize legacy COBOL code."""
        modifications = []
        lines = source_code.split("\n")

        for i, line in enumerate(lines):
            if len(line) > 7:
                code_area = line[7:72]
                code_upper = code_area.upper()

                # Replace deprecated features
                if "ALTER" in code_upper:
                    modifications.append(
                        {
                            "line": i + 1,
                            "type": "modernization",
                            "original": line,
                            "suggestion": "Replace ALTER with structured logic",
                            "automated": False,
                        }
                    )

                # Suggest replacing GO TO with PERFORM
                if re.search(r"\bGO\s+TO\s+\w+", code_upper):
                    modifications.append(
                        {
                            "line": i + 1,
                            "type": "modernization",
                            "original": line,
                            "suggestion": "Replace GO TO with PERFORM",
                            "automated": False,
                        }
                    )

                # Suggest using intrinsic functions
                if "CURRENT-DATE" in code_upper and "ACCEPT" in code_upper:
                    new_line = line.replace("ACCEPT", "MOVE FUNCTION CURRENT-DATE TO")
                    lines[i] = new_line
                    modifications.append(
                        {
                            "line": i + 1,
                            "type": "modernization",
                            "original": line,
                            "modified": new_line,
                            "automated": True,
                        }
                    )

                # Replace old-style file status checks
                if re.search(r'IF\s+\w+-STATUS\s+=\s+["\']\d+["\']', code_upper):
                    modifications.append(
                        {
                            "line": i + 1,
                            "type": "modernization",
                            "original": line,
                            "suggestion": "Use 88-level conditions for file status",
                            "automated": False,
                        }
                    )

        return "\n".join(lines), modifications
