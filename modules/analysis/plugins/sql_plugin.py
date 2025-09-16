"""
SQL Language Plugin for Homeostasis

This plugin enables Homeostasis to analyze and fix errors in SQL queries and database interactions.
It provides comprehensive error handling for database-related issues across different SQL dialects
including PostgreSQL, MySQL, SQLite, SQL Server, and Oracle.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..language_plugin_system import LanguagePlugin, register_plugin

logger = logging.getLogger(__name__)


class SQLExceptionHandler:
    """
    Handles SQL exceptions with robust error detection and classification.

    This class provides logic for categorizing SQL exceptions based on their type,
    message, and error codes. It supports multiple SQL dialects and database systems.
    """

    def __init__(self):
        """Initialize the SQL exception handler."""
        self.rule_categories = {
            "syntax": "SQL syntax errors",
            "constraint": "Constraint violation errors",
            "permission": "Permission and access errors",
            "connection": "Database connection errors",
            "performance": "Query performance issues",
            "transaction": "Transaction-related errors",
            "data_type": "Data type and casting errors",
            "index": "Index and optimization errors",
            "table": "Table and schema errors",
            "function": "Function and procedure errors",
        }

        # Database-specific error patterns
        self.db_error_patterns = {
            "postgresql": {
                "syntax_error": r"syntax error at or near",
                "relation_not_exist": r"relation \".*\" does not exist",
                "column_not_exist": r"column \".*\" does not exist",
                "constraint_violation": r"violates .*? constraint",
                "permission_denied": r"permission denied for",
                "connection_failed": r"could not connect to server",
                "duplicate_key": r"duplicate key value violates unique constraint",
            },
            "mysql": {
                "syntax_error": r"You have an error in your SQL syntax",
                "table_not_exist": r"Table '.*' doesn't exist",
                "column_not_exist": r"Unknown column '.*' in",
                "constraint_violation": r"Duplicate entry '.*' for key",
                "access_denied": r"Access denied for user",
                "connection_failed": r"Can't connect to MySQL server",
                "foreign_key": r"Cannot add or update a child row: a foreign key constraint fails",
            },
            "sqlite": {
                "syntax_error": r"near \".*\": syntax error",
                "table_not_exist": r"no such table:",
                "column_not_exist": r"no such column:",
                "constraint_violation": r"UNIQUE constraint failed:",
                "database_locked": r"database is locked",
                "foreign_key": r"FOREIGN KEY constraint failed",
            },
            "sqlserver": {
                "syntax_error": r"Incorrect syntax near",
                "object_not_exist": r"Invalid object name",
                "column_not_exist": r"Invalid column name",
                "constraint_violation": r"Violation of .* constraint",
                "login_failed": r"Login failed for user",
                "connection_failed": r"A network-related or instance-specific error",
            },
            "oracle": {
                "syntax_error": r"ORA-00936: missing expression",
                "table_not_exist": r"ORA-00942: table or view does not exist",
                "column_not_exist": r"ORA-00904: .* invalid identifier",
                "constraint_violation": r"ORA-00001: unique constraint .* violated",
                "permission_denied": r"ORA-01031: insufficient privileges",
                "connection_failed": r"ORA-12541: TNS:no listener",
            },
        }

        # Load rules from different categories
        self.rules = self._load_rules()

        # Pre-compile regex patterns for better performance
        self._compile_patterns()

    def _load_rules(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load SQL error rules from rule files."""
        rules = {}
        rules_dir = Path(__file__).parent.parent / "rules" / "sql"

        try:
            # Load common SQL rules
            common_rules_path = rules_dir / "sql_common_errors.json"
            if common_rules_path.exists():
                with open(common_rules_path, "r") as f:
                    common_data = json.load(f)
                    rules["common"] = common_data.get("rules", [])
                    logger.info(f"Loaded {len(rules['common'])} common SQL rules")

            # Load database-specific rules
            for db_type in ["postgresql", "mysql", "sqlite", "sqlserver", "oracle"]:
                db_rules_path = rules_dir / f"{db_type}_errors.json"
                if db_rules_path.exists():
                    with open(db_rules_path, "r") as f:
                        db_data = json.load(f)
                        rules[db_type] = db_data.get("rules", [])
                        logger.info(f"Loaded {len(rules[db_type])} {db_type} rules")

        except Exception as e:
            logger.error(f"Error loading SQL rules: {e}")
            rules = {"common": []}

        return rules

    def _compile_patterns(self):
        """Pre-compile regex patterns for better performance."""
        self.compiled_patterns: Dict[str, List[tuple[re.Pattern[str], Dict[str, Any]]]] = {}

        for category, rule_list in self.rules.items():
            self.compiled_patterns[category] = []
            for rule in rule_list:
                try:
                    pattern = rule.get("pattern", "")
                    if pattern:
                        compiled = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
                        self.compiled_patterns[category].append((compiled, rule))
                except re.error as e:
                    logger.warning(
                        f"Invalid regex pattern in rule {rule.get('id', 'unknown')}: {e}"
                    )

    def analyze_exception(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a SQL exception and determine its type and potential fixes.

        Args:
            error_data: SQL error data in standard format

        Returns:
            Analysis results with categorization and fix suggestions
        """
        message = error_data.get("message", "")
        error_code = error_data.get("error_code", "")
        database_type = error_data.get("database_type", "").lower()

        # Try to detect database type from error message if not provided
        if not database_type:
            database_type = self._detect_database_type(message, error_code)

        # Analyze based on database type
        if database_type in self.db_error_patterns:
            analysis = self._analyze_database_specific_error(
                message, error_code, database_type
            )
        else:
            analysis = self._analyze_generic_sql_error(message, error_code)

        # Find matching rules
        matches = self._find_matching_rules(message, error_data)

        if matches:
            # Use the best match (highest confidence)
            best_match = max(matches, key=lambda x: x.get("confidence_score", 0))

            # Map rule type to subcategory
            rule_type = best_match.get("type", "")
            subcategory = self._map_rule_type_to_subcategory(rule_type)
            if not subcategory:
                subcategory = analysis.get("subcategory", "unknown")

            # Get tags and ensure subcategory is included
            tags = best_match.get("tags", [])
            if subcategory and subcategory not in tags:
                tags = list(tags)  # Make a copy
                tags.append(subcategory)

            analysis.update(
                {
                    "category": best_match.get(
                        "category", analysis.get("category", "unknown")
                    ),
                    "subcategory": subcategory,
                    "confidence": best_match.get("confidence", "medium"),
                    "suggested_fix": best_match.get(
                        "suggestion", analysis.get("suggested_fix", "")
                    ),
                    "root_cause": best_match.get(
                        "root_cause", analysis.get("root_cause", "")
                    ),
                    "severity": best_match.get("severity", "medium"),
                    "rule_id": best_match.get("id", ""),
                    "tags": tags,
                    "all_matches": matches,
                }
            )

        analysis["database_type"] = database_type
        return analysis

    def _map_rule_type_to_subcategory(self, rule_type: str) -> str:
        """Map rule type to expected subcategory."""
        type_mapping = {
            "SyntaxError": "syntax",
            "ConstraintError": "constraint",
            "SchemaError": "schema",
            "PermissionError": "permission",
            "ConnectionError": "connection",
            "DataTypeError": "type",
            "TransactionError": "transaction",
            "FunctionError": "function",
            "IndexError": "index",
            "JoinError": "join",
        }
        return type_mapping.get(rule_type, rule_type.lower() if rule_type else "")

    def _detect_database_type(self, message: str, error_code: str) -> str:
        """Detect database type from error message or code."""
        message_lower = message.lower()

        # PostgreSQL indicators
        if any(
            indicator in message_lower
            for indicator in ["psql", "postgresql", "postgres"]
        ):
            return "postgresql"

        # MySQL indicators
        if any(indicator in message_lower for indicator in ["mysql", "mariadb"]):
            return "mysql"

        # SQLite indicators
        if "sqlite" in message_lower:
            return "sqlite"

        # SQL Server indicators
        if any(
            indicator in message_lower
            for indicator in ["sql server", "sqlserver", "mssql"]
        ):
            return "sqlserver"

        # Oracle indicators
        if any(
            indicator in message_lower for indicator in ["oracle", "ora-"]
        ) or error_code.startswith("ORA-"):
            return "oracle"

        # MongoDB (NoSQL but often included)
        if "mongodb" in message_lower or "mongo" in message_lower:
            return "mongodb"

        return "generic"

    def _analyze_database_specific_error(
        self, message: str, error_code: str, database_type: str
    ) -> Dict[str, Any]:
        """Analyze database-specific SQL errors."""
        patterns = self.db_error_patterns.get(database_type, {})

        # Check each pattern type
        for pattern_type, pattern in patterns.items():
            if re.search(pattern, message, re.IGNORECASE):
                return self._get_error_analysis(
                    pattern_type, database_type, message, error_code
                )

        # Default analysis for unmatched database-specific errors
        return {
            "category": "sql",
            "subcategory": "unknown",
            "confidence": "low",
            "suggested_fix": f"Check {database_type} documentation for error resolution",
            "root_cause": f"sql_{database_type}_error",
            "severity": "medium",
            "tags": ["sql", database_type],
        }

    def _analyze_generic_sql_error(
        self, message: str, error_code: str
    ) -> Dict[str, Any]:
        """Analyze generic SQL errors."""
        message_lower = message.lower()

        # Common SQL error patterns
        if "syntax error" in message_lower or "syntax" in message_lower:
            return {
                "category": "sql",
                "subcategory": "syntax",
                "confidence": "high",
                "suggested_fix": "Check SQL syntax for missing keywords, commas, or parentheses",
                "root_cause": "sql_syntax_error",
                "severity": "high",
                "tags": ["sql", "syntax"],
            }

        if "table" in message_lower and (
            "not exist" in message_lower or "doesn't exist" in message_lower
        ):
            return {
                "category": "sql",
                "subcategory": "table",
                "confidence": "high",
                "suggested_fix": "Verify table name spelling and existence in the database",
                "root_cause": "sql_table_not_exist",
                "severity": "high",
                "tags": ["sql", "table", "schema"],
            }

        if "column" in message_lower and (
            "not exist" in message_lower or "unknown" in message_lower
        ):
            return {
                "category": "sql",
                "subcategory": "column",
                "confidence": "high",
                "suggested_fix": "Check column name spelling and table schema",
                "root_cause": "sql_column_not_exist",
                "severity": "high",
                "tags": ["sql", "column", "schema"],
            }

        # Check for index errors before general constraint errors
        if (
            (
                "index" in message_lower
                and ("duplicate" in message_lower or "unique" in message_lower)
            )
            or ("cannot create index" in message_lower)
            or ("index" in message_lower and "already exists" in message_lower)
        ):
            return {
                "category": "sql",
                "subcategory": "index",
                "confidence": "high",
                "suggested_fix": "Check index constraints and ensure unique values where required",
                "root_cause": "sql_index_constraint_violation",
                "severity": "medium",
                "tags": ["sql", "index", "constraint"],
            }

        if "duplicate" in message_lower or "unique constraint" in message_lower:
            return {
                "category": "sql",
                "subcategory": "constraint",
                "confidence": "high",
                "suggested_fix": "Check for duplicate values or modify unique constraints",
                "root_cause": "sql_unique_constraint_violation",
                "severity": "medium",
                "tags": ["sql", "constraint", "unique"],
            }

        if "permission denied" in message_lower or "access denied" in message_lower:
            return {
                "category": "sql",
                "subcategory": "permission",
                "confidence": "high",
                "suggested_fix": "Check database user permissions and privileges",
                "root_cause": "sql_permission_denied",
                "severity": "high",
                "tags": ["sql", "permission", "access"],
            }

        if "ambiguous" in message_lower and (
            "column" in message_lower or "reference" in message_lower
        ):
            return {
                "category": "sql",
                "subcategory": "join",
                "confidence": "high",
                "suggested_fix": "Qualify column names with table aliases to resolve ambiguity",
                "root_cause": "sql_ambiguous_column",
                "severity": "high",
                "tags": ["sql", "join", "ambiguous"],
            }

        if "invalid input syntax" in message_lower and "type" in message_lower:
            return {
                "category": "sql",
                "subcategory": "type",
                "confidence": "high",
                "suggested_fix": "Check data types and ensure proper type conversions",
                "root_cause": "sql_type_mismatch",
                "severity": "medium",
                "tags": ["sql", "type", "conversion"],
            }

        if (
            ("could not connect" in message_lower)
            or (
                "connection" in message_lower
                and (
                    "failed" in message_lower
                    or "refused" in message_lower
                    or "timeout" in message_lower
                )
            )
            or ("unable to connect" in message_lower)
            or ("database server" in message_lower and "connect" in message_lower)
        ):
            return {
                "category": "sql",
                "subcategory": "connection",
                "confidence": "high",
                "suggested_fix": "Check database server status, network connectivity, and connection parameters",
                "root_cause": "sql_connection_error",
                "severity": "high",
                "tags": ["sql", "connection", "network"],
            }

        return {
            "category": "sql",
            "subcategory": "unknown",
            "confidence": "low",
            "suggested_fix": "Review SQL query and database configuration",
            "root_cause": "sql_generic_error",
            "severity": "medium",
            "tags": ["sql", "generic"],
        }

    def _get_error_analysis(
        self, pattern_type: str, database_type: str, message: str, error_code: str
    ) -> Dict[str, Any]:
        """Get detailed analysis for specific error patterns."""
        error_analyses = {
            "syntax_error": {
                "category": "sql",
                "subcategory": "syntax",
                "confidence": "high",
                "suggested_fix": "Fix SQL syntax errors - check for missing keywords, incorrect punctuation, or invalid syntax",
                "root_cause": f"sql_{database_type}_syntax_error",
                "severity": "high",
                "tags": ["sql", database_type, "syntax"],
            },
            "table_not_exist": {
                "category": "sql",
                "subcategory": "table",
                "confidence": "high",
                "suggested_fix": "Verify table exists and check spelling. Run CREATE TABLE if needed",
                "root_cause": f"sql_{database_type}_table_not_exist",
                "severity": "high",
                "tags": ["sql", database_type, "table", "schema"],
            },
            "relation_not_exist": {
                "category": "sql",
                "subcategory": "table",
                "confidence": "high",
                "suggested_fix": "Verify table/view exists and check spelling. Run CREATE TABLE/VIEW if needed",
                "root_cause": f"sql_{database_type}_relation_not_exist",
                "severity": "high",
                "tags": ["sql", database_type, "table", "view", "schema"],
            },
            "column_not_exist": {
                "category": "sql",
                "subcategory": "column",
                "confidence": "high",
                "suggested_fix": "Check column name spelling and table schema. Run ALTER TABLE ADD COLUMN if needed",
                "root_cause": f"sql_{database_type}_column_not_exist",
                "severity": "high",
                "tags": ["sql", database_type, "column", "schema"],
            },
            "constraint_violation": {
                "category": "sql",
                "subcategory": "constraint",
                "confidence": "high",
                "suggested_fix": "Check data values against constraints. Modify data or adjust constraints",
                "root_cause": f"sql_{database_type}_constraint_violation",
                "severity": "medium",
                "tags": ["sql", database_type, "constraint"],
            },
            "duplicate_key": {
                "category": "sql",
                "subcategory": "constraint",
                "confidence": "high",
                "suggested_fix": "Remove duplicate values or use INSERT ON CONFLICT/REPLACE INTO",
                "root_cause": f"sql_{database_type}_duplicate_key",
                "severity": "medium",
                "tags": ["sql", database_type, "constraint", "unique"],
            },
            "foreign_key": {
                "category": "sql",
                "subcategory": "constraint",
                "confidence": "high",
                "suggested_fix": "Ensure referenced values exist in parent table or disable foreign key checks temporarily",
                "root_cause": f"sql_{database_type}_foreign_key_violation",
                "severity": "medium",
                "tags": ["sql", database_type, "constraint", "foreign_key"],
            },
            "permission_denied": {
                "category": "sql",
                "subcategory": "permission",
                "confidence": "high",
                "suggested_fix": "Grant appropriate permissions to user or connect with sufficient privileges",
                "root_cause": f"sql_{database_type}_permission_denied",
                "severity": "high",
                "tags": ["sql", database_type, "permission", "access"],
            },
            "access_denied": {
                "category": "sql",
                "subcategory": "permission",
                "confidence": "high",
                "suggested_fix": "Check user credentials and database permissions",
                "root_cause": f"sql_{database_type}_access_denied",
                "severity": "high",
                "tags": ["sql", database_type, "permission", "access"],
            },
            "login_failed": {
                "category": "sql",
                "subcategory": "permission",
                "confidence": "high",
                "suggested_fix": "Verify username, password, and user exists in database",
                "root_cause": f"sql_{database_type}_login_failed",
                "severity": "high",
                "tags": ["sql", database_type, "permission", "authentication"],
            },
            "connection_failed": {
                "category": "sql",
                "subcategory": "connection",
                "confidence": "high",
                "suggested_fix": "Check database server status, network connectivity, and connection parameters",
                "root_cause": f"sql_{database_type}_connection_failed",
                "severity": "high",
                "tags": ["sql", database_type, "connection", "network"],
            },
            "database_locked": {
                "category": "sql",
                "subcategory": "transaction",
                "confidence": "high",
                "suggested_fix": "Wait for other transactions to complete or check for deadlocks",
                "root_cause": f"sql_{database_type}_database_locked",
                "severity": "medium",
                "tags": ["sql", database_type, "transaction", "lock"],
            },
        }

        return error_analyses.get(
            pattern_type,
            {
                "category": "sql",
                "subcategory": "unknown",
                "confidence": "medium",
                "suggested_fix": f"Consult {database_type} documentation for error resolution",
                "root_cause": f"sql_{database_type}_unknown",
                "severity": "medium",
                "tags": ["sql", database_type],
            },
        )

    def _find_matching_rules(
        self, error_text: str, error_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Find all rules that match the given error."""
        matches = []

        for category, patterns in self.compiled_patterns.items():
            for compiled_pattern, rule in patterns:
                match = compiled_pattern.search(error_text)
                if match:
                    # Calculate confidence score based on match quality
                    confidence_score = self._calculate_confidence(
                        match, rule, error_data
                    )

                    match_info = rule.copy()
                    match_info["confidence_score"] = confidence_score
                    match_info["match_groups"] = (
                        match.groups() if match.groups() else []
                    )
                    matches.append(match_info)

        return matches

    def _calculate_confidence(
        self, match: re.Match, rule: Dict[str, Any], error_data: Dict[str, Any]
    ) -> float:
        """Calculate confidence score for a rule match."""
        base_confidence = 0.5

        # Boost confidence for exact error type matches
        rule_type = rule.get("type", "").lower()
        error_type = error_data.get("error_type", "").lower()
        if rule_type and rule_type in error_type:
            base_confidence += 0.3

        # Boost confidence based on rule reliability
        reliability = rule.get("reliability", "medium")
        reliability_boost = {"high": 0.2, "medium": 0.1, "low": 0.0}
        base_confidence += reliability_boost.get(reliability, 0.0)

        # Boost confidence for database-specific rules
        rule_tags = set(rule.get("tags", []))
        database_type = error_data.get("database_type", "").lower()
        if database_type and database_type in rule_tags:
            base_confidence += 0.2

        return min(base_confidence, 1.0)


class SQLPatchGenerator:
    """
    Generates patches for SQL errors based on analysis results.

    This class creates SQL fixes for common database errors using templates
    and heuristics specific to different SQL dialects.
    """

    def __init__(self):
        """Initialize the SQL patch generator."""
        self.template_dir = (
            Path(__file__).parent.parent / "patch_generation" / "templates"
        )
        self.sql_template_dir = self.template_dir / "sql"

        # Ensure template directory exists
        self.sql_template_dir.mkdir(parents=True, exist_ok=True)

        # Load patch templates
        self.templates: Dict[str, str] = self._load_templates()

    def _load_templates(self) -> Dict[str, str]:
        """Load SQL patch templates."""
        templates: Dict[str, str] = {}

        if not self.sql_template_dir.exists():
            logger.warning(
                f"SQL templates directory not found: {self.sql_template_dir}"
            )
            return templates

        for template_file in self.sql_template_dir.glob("*.sql.template"):
            try:
                with open(template_file, "r") as f:
                    template_name = template_file.stem.replace(".sql", "")
                    templates[template_name] = f.read()
                    logger.debug(f"Loaded template: {template_name}")
            except Exception as e:
                logger.error(f"Error loading template {template_file}: {e}")

        return templates

    def generate_patch(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], query: str = ""
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a patch for the SQL error.

        Args:
            error_data: The SQL error data
            analysis: Analysis results from SQLExceptionHandler
            query: The SQL query that caused the error

        Returns:
            Patch information or None if no patch can be generated
        """
        root_cause = analysis.get("root_cause", "")
        database_type = analysis.get("database_type", "generic")

        # Map root causes to patch strategies
        patch_strategies = {
            "sql_syntax_error": self._fix_syntax_error,
            "sql_table_not_exist": self._fix_table_not_exist,
            "sql_column_not_exist": self._fix_column_not_exist,
            "sql_unique_constraint_violation": self._fix_unique_constraint,
            "sql_foreign_key_violation": self._fix_foreign_key_constraint,
            "sql_permission_denied": self._fix_permission_error,
            "sql_connection_failed": self._fix_connection_error,
            "sql_ambiguous_column": self._fix_ambiguous_column,
            "sql_type_mismatch": self._fix_type_mismatch,
            "sql_index_constraint_violation": self._fix_index_constraint,
        }

        # Try database-specific patches first
        specific_cause = root_cause.replace("sql_", f"sql_{database_type}_")
        strategy = patch_strategies.get(specific_cause) or patch_strategies.get(
            root_cause
        )

        if strategy:
            try:
                return strategy(error_data, analysis, query)
            except Exception as e:
                logger.error(f"Error generating patch for {root_cause}: {e}")

        # Try to use templates if no specific strategy matches
        return self._template_based_patch(error_data, analysis, query)

    def _fix_syntax_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], query: str
    ) -> Optional[Dict[str, Any]]:
        """Fix SQL syntax errors."""
        # Common syntax fixes
        fixes = []

        # Missing commas in SELECT statements
        if "select" in query.lower() and re.search(
            r"\w+\s+\w+\s+from", query, re.IGNORECASE
        ):
            fixes.append(
                {
                    "type": "suggestion",
                    "description": "Check for missing commas between column names in SELECT statement",
                    "example": "SELECT column1, column2 FROM table_name",
                }
            )

        # Missing FROM clause
        if "select" in query.lower() and "from" not in query.lower():
            fixes.append(
                {
                    "type": "suggestion",
                    "description": "Add FROM clause to specify the table",
                    "example": "SELECT * FROM table_name",
                }
            )

        # Unmatched quotes
        if query.count("'") % 2 != 0 or query.count('"') % 2 != 0:
            fixes.append(
                {
                    "type": "suggestion",
                    "description": "Check for unmatched quotes in string literals",
                    "fix": "Ensure all string literals are properly quoted",
                }
            )

        # Missing parentheses in functions
        if re.search(r"\w+\s*\(", query) and query.count("(") != query.count(")"):
            fixes.append(
                {
                    "type": "suggestion",
                    "description": "Check for unmatched parentheses in function calls",
                    "fix": "Ensure all opening parentheses have matching closing parentheses",
                }
            )

        if fixes:
            return {
                "type": "multiple_suggestions",
                "fixes": fixes,
                "description": "SQL syntax error fixes",
            }

        return {
            "type": "suggestion",
            "description": "Review SQL syntax for proper keywords, punctuation, and structure",
        }

    def _fix_table_not_exist(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], query: str
    ) -> Optional[Dict[str, Any]]:
        """Fix table not exist errors."""
        message = error_data.get("message", "")

        # Extract table name from error message
        table_match = re.search(
            r'(?:table|relation)\s+["\']?([^"\']+)["\']?\s+(?:does not exist|doesn\'t exist)',
            message,
            re.IGNORECASE,
        )
        if table_match:
            table_name = table_match.group(1)

            return {
                "type": "suggestion",
                "description": f"Table '{table_name}' does not exist",
                "fixes": [
                    f"Check if table name '{table_name}' is spelled correctly",
                    "Verify table exists in the database: SHOW TABLES or \\dt",
                    "Create the table if it doesn't exist",
                    "Check if you're connected to the correct database/schema",
                ],
            }

        return {
            "type": "suggestion",
            "description": "Table does not exist. Check table name spelling and database schema",
        }

    def _fix_column_not_exist(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], query: str
    ) -> Optional[Dict[str, Any]]:
        """Fix column not exist errors."""
        message = error_data.get("message", "")

        # Extract column name from error message
        column_match = re.search(
            r'(?:column|field)\s+["\']?([^"\']+)["\']?\s+(?:does not exist|doesn\'t exist|unknown)',
            message,
            re.IGNORECASE,
        )
        if column_match:
            column_name = column_match.group(1)

            return {
                "type": "suggestion",
                "description": f"Column '{column_name}' does not exist",
                "fixes": [
                    f"Check if column name '{column_name}' is spelled correctly",
                    "Verify column exists in the table: DESCRIBE table_name or \\d table_name",
                    "Add the column if it doesn't exist: ALTER TABLE table_name ADD COLUMN column_name datatype",
                    "Check if you're using the correct table alias",
                ],
            }

        return {
            "type": "suggestion",
            "description": "Column does not exist. Check column name spelling and table schema",
        }

    def _fix_unique_constraint(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], query: str
    ) -> Optional[Dict[str, Any]]:
        """Fix unique constraint violation errors."""
        database_type = analysis.get("database_type", "generic")

        fixes = [
            "Check for duplicate values in the data being inserted",
            "Use INSERT IGNORE (MySQL) or INSERT ... ON CONFLICT DO NOTHING (PostgreSQL) to skip duplicates",
            "Use REPLACE INTO (MySQL) or INSERT ... ON CONFLICT DO UPDATE (PostgreSQL) to update existing records",
            "Remove duplicate values before inserting",
        ]

        if database_type == "postgresql":
            fixes.append("Use INSERT ... ON CONFLICT (column) DO UPDATE SET ...")
        elif database_type == "mysql":
            fixes.append("Use INSERT ... ON DUPLICATE KEY UPDATE ...")
        elif database_type == "sqlite":
            fixes.append("Use INSERT OR REPLACE INTO ...")

        return {
            "type": "suggestion",
            "description": "Unique constraint violation - duplicate values detected",
            "fixes": fixes,
        }

    def _fix_foreign_key_constraint(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], query: str
    ) -> Optional[Dict[str, Any]]:
        """Fix foreign key constraint violation errors."""
        return {
            "type": "suggestion",
            "description": "Foreign key constraint violation",
            "fixes": [
                "Ensure the referenced value exists in the parent table",
                "Check if the foreign key references the correct column",
                "Insert the referenced record in the parent table first",
                "Temporarily disable foreign key checks if appropriate: SET foreign_key_checks = 0 (MySQL)",
            ],
        }

    def _fix_permission_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], query: str
    ) -> Optional[Dict[str, Any]]:
        """Fix permission denied errors."""
        database_type = analysis.get("database_type", "generic")

        fixes = [
            "Check if the user has appropriate permissions for the operation",
            "Connect with a user that has sufficient privileges",
            "Grant necessary permissions to the user",
        ]

        if database_type == "postgresql":
            fixes.extend(
                [
                    "GRANT SELECT, INSERT, UPDATE, DELETE ON table_name TO username",
                    "GRANT ALL PRIVILEGES ON DATABASE dbname TO username",
                ]
            )
        elif database_type == "mysql":
            fixes.extend(
                [
                    "GRANT SELECT, INSERT, UPDATE, DELETE ON database.table TO 'user'@'host'",
                    "GRANT ALL PRIVILEGES ON database.* TO 'user'@'host'",
                ]
            )

        return {
            "type": "suggestion",
            "description": "Permission denied - insufficient privileges",
            "fixes": fixes,
        }

    def _fix_connection_error(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], query: str
    ) -> Optional[Dict[str, Any]]:
        """Fix database connection errors."""
        return {
            "type": "suggestion",
            "description": "Database connection failed",
            "fixes": [
                "Check if the database server is running",
                "Verify connection parameters (host, port, database name)",
                "Check network connectivity to the database server",
                "Verify firewall settings allow database connections",
                "Check if the database service is started",
                "Verify SSL/TLS configuration if required",
            ],
        }

    def _fix_ambiguous_column(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], query: str
    ) -> Optional[Dict[str, Any]]:
        """Fix ambiguous column reference errors."""
        message = error_data.get("message", "")

        # Extract column name if possible
        column_match = re.search(r"column (?:reference )?'([^']+)'", message)
        column_name = column_match.group(1) if column_match else "column"

        return {
            "type": "suggestion",
            "description": f"Qualify the ambiguous column '{column_name}' with a table alias",
            "fix_steps": [
                "Add table aliases to your query (e.g., SELECT t1.column_name FROM table1 t1)",
                "Prefix the column with the correct table name (e.g., table1.column_name)",
                "If using joins, ensure all columns are qualified with their respective table aliases",
                "Consider using AS clause for column aliases to avoid conflicts",
            ],
            "example": "SELECT t1.column_name, t2.other_column\nFROM table1 t1\nJOIN table2 t2 ON t1.id = t2.id",
        }

    def _fix_type_mismatch(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], query: str
    ) -> Optional[Dict[str, Any]]:
        """Fix type mismatch errors."""
        message = error_data.get("message", "")

        # Try to extract type information
        type_match = re.search(r"type (\w+)", message)
        expected_type = type_match.group(1) if type_match else "expected type"

        return {
            "type": "suggestion",
            "description": f"Fix data type mismatch for {expected_type}",
            "fix_steps": [
                f"Cast the value to {expected_type} (e.g., CAST(value AS {expected_type.upper()}))",
                "Check if the value format matches the expected type",
                "For dates/timestamps, ensure proper format (YYYY-MM-DD)",
                "For numeric types, remove non-numeric characters",
                "Use appropriate conversion functions for your database",
            ],
            "example": "-- For PostgreSQL\nSELECT CAST('123' AS INTEGER);\n-- For MySQL\nSELECT CONVERT('123', SIGNED);",
        }

    def _fix_index_constraint(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], query: str
    ) -> Optional[Dict[str, Any]]:
        """Fix index constraint violations."""
        return {
            "type": "suggestion",
            "description": "Resolve index constraint violation from duplicate values",
            "fix_steps": [
                "Remove duplicate values from the column before creating the index",
                "Use a non-unique index if duplicate values are allowed",
                "Clean up existing data to satisfy unique constraints",
                "Consider using a partial unique index with a WHERE clause",
                "Drop and recreate the index with appropriate options",
            ],
            "example": "-- Remove duplicates before creating unique index\nDELETE FROM table_name t1\nUSING table_name t2\nWHERE t1.ctid < t2.ctid\n  AND t1.column_name = t2.column_name;",
        }

    def _template_based_patch(
        self, error_data: Dict[str, Any], analysis: Dict[str, Any], query: str
    ) -> Optional[Dict[str, Any]]:
        """Generate patch using templates."""
        root_cause = analysis.get("root_cause", "")

        # Map root causes to template names
        template_map = {
            "sql_syntax_error": "syntax_fix",
            "sql_table_not_exist": "table_check",
            "sql_column_not_exist": "column_check",
            "sql_constraint_violation": "constraint_fix",
        }

        template_name = template_map.get(root_cause)
        if template_name and template_name in self.templates:
            template = self.templates[template_name]

            return {
                "type": "template",
                "template": template,
                "description": f"Applied template fix for {root_cause}",
            }

        return None


class SQLLanguagePlugin(LanguagePlugin):
    """
    Main SQL language plugin for Homeostasis.

    This plugin orchestrates SQL error analysis and patch generation,
    supporting multiple SQL dialects and database systems.
    """

    VERSION = "1.0.0"
    AUTHOR = "Homeostasis Team"

    def __init__(self):
        """Initialize the SQL language plugin."""
        self.language = "sql"
        self.supported_extensions = {".sql", ".ddl", ".dml", ".psql", ".mysql"}
        self.supported_frameworks = [
            "postgresql",
            "mysql",
            "sqlite",
            "sqlserver",
            "oracle",
            "mariadb",
            "mongodb",
            "cassandra",
            "redis",
        ]

        # Initialize components
        self.exception_handler = SQLExceptionHandler()
        self.patch_generator = SQLPatchGenerator()

        logger.info("SQL language plugin initialized")

    def get_language_id(self) -> str:
        """Get the unique identifier for this language."""
        return "sql"

    def get_language_name(self) -> str:
        """Get the human-readable name of the language."""
        return "SQL"

    def get_language_version(self) -> str:
        """Get the version of the language supported by this plugin."""
        return "ANSI SQL"

    def get_supported_frameworks(self) -> List[str]:
        """Get the list of frameworks supported by this language plugin."""
        return self.supported_frameworks

    def normalize_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize error data to the standard Homeostasis format.

        Args:
            error_data: Error data in the SQL-specific format

        Returns:
            Error data in the standard format
        """
        # Map SQL-specific error fields to standard format
        normalized = {
            "error_type": error_data.get(
                "error_type", error_data.get("sqlstate", "SQLError")
            ),
            "message": error_data.get("message", error_data.get("description", "")),
            "language": "sql",
            "file_path": error_data.get("file_path", error_data.get("file", "")),
            "line_number": error_data.get("line_number", error_data.get("line", 0)),
            "column_number": error_data.get(
                "column_number", error_data.get("column", 0)
            ),
            "database_type": error_data.get(
                "database_type", error_data.get("driver", "")
            ),
            "error_code": error_data.get("error_code", error_data.get("code", "")),
            "query": error_data.get("query", error_data.get("sql", "")),
            "stack_trace": error_data.get("stack_trace", []),
            "context": error_data.get("context", {}),
            "timestamp": error_data.get("timestamp"),
            "severity": error_data.get("severity", "medium"),
        }

        # Add any additional fields from the original error
        for key, value in error_data.items():
            if key not in normalized and value is not None:
                normalized[key] = value

        return normalized

    def denormalize_error(self, standard_error: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert standard format error data back to the SQL-specific format.

        Args:
            standard_error: Error data in the standard format

        Returns:
            Error data in the SQL-specific format
        """
        # Map standard fields back to SQL-specific format
        sql_error = {
            "error_type": standard_error.get("error_type", "SQLError"),
            "message": standard_error.get("message", ""),
            "file": standard_error.get("file_path", ""),
            "line": standard_error.get("line_number", 0),
            "column": standard_error.get("column_number", 0),
            "database_type": standard_error.get("database_type", ""),
            "error_code": standard_error.get("error_code", ""),
            "query": standard_error.get("query", ""),
            "sqlstate": standard_error.get("error_type", ""),
            "code": standard_error.get("error_code", ""),
            "sql": standard_error.get("query", ""),
            "description": standard_error.get("message", ""),
            "driver": standard_error.get("database_type", ""),
            "stack_trace": standard_error.get("stack_trace", []),
            "context": standard_error.get("context", {}),
            "timestamp": standard_error.get("timestamp"),
            "severity": standard_error.get("severity", "medium"),
        }

        # Add any additional fields from the standard error
        for key, value in standard_error.items():
            if key not in sql_error and value is not None:
                sql_error[key] = value

        return sql_error

    def analyze_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a SQL error.

        Args:
            error_data: SQL error data

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
            analysis["plugin"] = "sql"
            analysis["language"] = "sql"
            analysis["plugin_version"] = self.VERSION

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing SQL error: {e}")
            return {
                "category": "sql",
                "subcategory": "unknown",
                "confidence": "low",
                "suggested_fix": "Unable to analyze SQL error",
                "error": str(e),
                "plugin": "sql",
            }

    def generate_fix(
        self, analysis: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a fix for an error based on the analysis.

        Args:
            analysis: Error analysis
            context: Additional context for fix generation

        Returns:
            Generated fix data
        """
        error_data = context.get("error_data", {})
        query = context.get("query", error_data.get("query", ""))

        fix = self.patch_generator.generate_patch(error_data, analysis, query)

        if fix:
            return fix
        else:
            return {
                "type": "suggestion",
                "description": analysis.get(
                    "suggested_fix", "No specific fix available"
                ),
                "confidence": analysis.get("confidence", "low"),
            }


# Register the plugin
register_plugin(SQLLanguagePlugin())
