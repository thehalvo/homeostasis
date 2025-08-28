"""
Tests for the SQL language plugin.
"""
import pytest
import json
from unittest.mock import Mock, patch

from modules.analysis.plugins.sql_plugin import (
    SQLLanguagePlugin, 
    SQLExceptionHandler, 
    SQLPatchGenerator
)


class TestSQLExceptionHandler:
    """Test the SQL exception handler."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.handler = SQLExceptionHandler()
    
    def test_analyze_syntax_error(self):
        """Test analysis of syntax errors."""
        error_data = {
            "error_type": "SQLError",
            "message": "Syntax error near 'FORM' at line 1",
            "file_path": "test.sql",
            "line_number": 1,
            "column_number": 5
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "sql"
        assert analysis["subcategory"] == "syntax"
        assert analysis["confidence"] == "high"
        assert "syntax" in analysis["tags"]
    
    def test_analyze_constraint_error(self):
        """Test analysis of constraint errors."""
        error_data = {
            "error_type": "SQLError",
            "message": "UNIQUE constraint failed: users.email",
            "file_path": "test.sql",
            "line_number": 15,
            "column_number": 8
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "sql"
        assert analysis["subcategory"] == "constraint"
        assert analysis["confidence"] == "high"
        assert "constraint" in analysis["tags"]
    
    def test_analyze_join_error(self):
        """Test analysis of join errors."""
        error_data = {
            "error_type": "SQLError",
            "message": "column reference 'id' is ambiguous",
            "file_path": "test.sql",
            "line_number": 20,
            "column_number": 12
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "sql"
        assert analysis["subcategory"] == "join"
        assert analysis["confidence"] == "high"
        assert "join" in analysis["tags"]
    
    def test_analyze_type_error(self):
        """Test analysis of type errors."""
        error_data = {
            "error_type": "SQLError",
            "message": "invalid input syntax for type integer",
            "file_path": "test.sql",
            "line_number": 25,
            "column_number": 15
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "sql"
        assert analysis["subcategory"] == "type"
        assert analysis["confidence"] == "high"
        assert "type" in analysis["tags"]
    
    def test_analyze_permission_error(self):
        """Test analysis of permission errors."""
        error_data = {
            "error_type": "SQLError",
            "message": "permission denied for table users",
            "file_path": "test.sql",
            "line_number": 30,
            "column_number": 10
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "sql"
        assert analysis["subcategory"] == "permission"
        assert analysis["confidence"] == "high"
        assert "permission" in analysis["tags"]
    
    def test_analyze_index_error(self):
        """Test analysis of index errors."""
        error_data = {
            "error_type": "SQLError",
            "message": "cannot create index on column with duplicate values",
            "file_path": "test.sql",
            "line_number": 35,
            "column_number": 1
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "sql"
        assert analysis["subcategory"] == "index"
        assert analysis["confidence"] == "high"
        assert "index" in analysis["tags"]
    
    def test_analyze_connection_error(self):
        """Test analysis of connection errors."""
        error_data = {
            "error_type": "SQLError",
            "message": "could not connect to database server",
            "file_path": "test.sql",
            "line_number": 1,
            "column_number": 1
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "sql"
        assert analysis["subcategory"] == "connection"
        assert analysis["confidence"] == "high"
        assert "connection" in analysis["tags"]
    
    def test_analyze_unknown_error(self):
        """Test analysis of unknown errors."""
        error_data = {
            "error_type": "SQLError",
            "message": "Some unknown error message",
            "file_path": "test.sql",
            "line_number": 45,
            "column_number": 1
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "sql"
        assert analysis["subcategory"] == "unknown"
        assert analysis["confidence"] == "low"
        assert "generic" in analysis["tags"]


class TestSQLPatchGenerator:
    """Test the SQL patch generator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = SQLPatchGenerator()
    
    def test_generate_syntax_fix(self):
        """Test generation of syntax fixes."""
        error_data = {
            "message": "Syntax error near 'FORM' at line 1",
            "file_path": "test.sql"
        }
        
        analysis = {
            "root_cause": "sql_syntax_error",
            "subcategory": "syntax",
            "confidence": "high"
        }
        
        patch = self.generator.generate_patch(error_data, analysis, "")
        
        assert patch is not None
        assert patch["type"] in ["suggestion", "multiple_suggestions"]
        assert "syntax" in patch["description"].lower()
    
    def test_generate_constraint_fix(self):
        """Test generation of constraint fixes."""
        error_data = {
            "message": "UNIQUE constraint failed: users.email",
            "file_path": "test.sql"
        }
        
        analysis = {
            "root_cause": "sql_unique_constraint_violation",
            "subcategory": "constraint",
            "confidence": "high"
        }
        
        patch = self.generator.generate_patch(error_data, analysis, "")
        
        assert patch is not None
        assert patch["type"] == "suggestion"
        assert "constraint" in patch["description"].lower() or "unique" in patch["description"].lower()
    
    def test_generate_join_fix(self):
        """Test generation of join fixes."""
        error_data = {
            "message": "column reference 'id' is ambiguous",
            "file_path": "test.sql"
        }
        
        analysis = {
            "root_cause": "sql_ambiguous_column",
            "subcategory": "join",
            "confidence": "high"
        }
        
        patch = self.generator.generate_patch(error_data, analysis, "")
        
        assert patch is not None
        assert patch["type"] == "suggestion"
        assert "ambiguous" in patch["description"].lower() or "qualify" in patch["description"].lower()
    
    def test_generate_type_fix(self):
        """Test generation of type fixes."""
        error_data = {
            "message": "invalid input syntax for type integer",
            "file_path": "test.sql"
        }
        
        analysis = {
            "root_cause": "sql_type_mismatch",
            "subcategory": "type",
            "confidence": "high"
        }
        
        patch = self.generator.generate_patch(error_data, analysis, "")
        
        assert patch is not None
        assert patch["type"] == "suggestion"
        assert "type" in patch["description"].lower() or "cast" in patch["description"].lower()
    
    def test_generate_permission_fix(self):
        """Test generation of permission fixes."""
        error_data = {
            "message": "permission denied for table users",
            "file_path": "test.sql"
        }
        
        analysis = {
            "root_cause": "sql_permission_denied",
            "subcategory": "permission",
            "confidence": "high"
        }
        
        patch = self.generator.generate_patch(error_data, analysis, "")
        
        assert patch is not None
        assert patch["type"] == "suggestion"
        assert "permission" in patch["description"].lower() or "grant" in patch["description"].lower()
    
    def test_generate_index_fix(self):
        """Test generation of index fixes."""
        error_data = {
            "message": "duplicate key value violates unique constraint",
            "file_path": "test.sql"
        }
        
        analysis = {
            "root_cause": "sql_index_constraint_violation",
            "subcategory": "index",
            "confidence": "high"
        }
        
        patch = self.generator.generate_patch(error_data, analysis, "")
        
        assert patch is not None
        assert patch["type"] == "suggestion"
        assert "duplicate" in patch["description"].lower() or "unique" in patch["description"].lower()
    
    def test_generate_connection_fix(self):
        """Test generation of connection fixes."""
        error_data = {
            "message": "could not connect to database server",
            "file_path": "test.sql"
        }
        
        analysis = {
            "root_cause": "sql_connection_failed",
            "subcategory": "connection",
            "confidence": "high"
        }
        
        patch = self.generator.generate_patch(error_data, analysis, "")
        
        assert patch is not None
        assert patch["type"] == "suggestion"
        assert "connection" in patch["description"].lower() or "database" in patch["description"].lower()


class TestSQLLanguagePlugin:
    """Test the SQL language plugin."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.plugin = SQLLanguagePlugin()
    
    def test_plugin_metadata(self):
        """Test plugin metadata."""
        assert self.plugin.get_language_id() == "sql"
        assert self.plugin.get_language_name() == "SQL"
        assert self.plugin.get_language_version() == "ANSI SQL"
        
        frameworks = self.plugin.get_supported_frameworks()
        assert "postgresql" in frameworks
        assert "mysql" in frameworks
        assert "sqlite" in frameworks
    
    def test_normalize_error(self):
        """Test error normalization."""
        sql_error = {
            "error_type": "SQLError",
            "message": "Test error",
            "file": "test.sql",
            "line": 10,
            "column": 5,
            "description": "Test error description"
        }
        
        normalized = self.plugin.normalize_error(sql_error)
        
        assert normalized["language"] == "sql"
        assert normalized["error_type"] == "SQLError"
        assert normalized["message"] == "Test error"
        assert normalized["file_path"] == "test.sql"
        assert normalized["line_number"] == 10
        assert normalized["column_number"] == 5
    
    def test_denormalize_error(self):
        """Test error denormalization."""
        standard_error = {
            "language": "sql",
            "error_type": "SQLError",
            "message": "Test error",
            "file_path": "test.sql",
            "line_number": 10,
            "column_number": 5,
            "severity": "high"
        }
        
        sql_error = self.plugin.denormalize_error(standard_error)
        
        assert sql_error["error_type"] == "SQLError"
        assert sql_error["message"] == "Test error"
        assert sql_error["file_path"] == "test.sql"
        assert sql_error["line_number"] == 10
        assert sql_error["column_number"] == 5
        assert sql_error["file"] == "test.sql"  # Alternative format
        assert sql_error["line"] == 10  # Alternative format
    
    def test_analyze_error(self):
        """Test error analysis."""
        error_data = {
            "error_type": "SQLError",
            "message": "UNIQUE constraint failed: users.email",
            "file_path": "test.sql",
            "line_number": 15,
            "column_number": 8
        }
        
        analysis = self.plugin.analyze_error(error_data)
        
        assert analysis["plugin"] == "sql"
        assert analysis["language"] == "sql"
        assert analysis["plugin_version"] == "1.0.0"
        assert analysis["category"] == "sql"
        assert analysis["subcategory"] == "constraint"
    
    def test_generate_fix(self):
        """Test fix generation."""
        analysis = {
            "root_cause": "sql_constraint_error",
            "subcategory": "constraint",
            "confidence": "high",
            "suggested_fix": "Fix constraint violation"
        }
        
        context = {
            "error_data": {
                "message": "UNIQUE constraint failed: users.email",
                "file_path": "test.sql"
            },
            "source_code": "INSERT INTO users (email) VALUES ('test@example.com');"
        }
        
        fix = self.plugin.generate_fix(analysis, context)
        
        assert fix is not None
        assert fix["type"] == "suggestion"
        assert "description" in fix
    
    def test_supported_extensions(self):
        """Test supported file extensions."""
        assert ".sql" in self.plugin.supported_extensions
        assert ".psql" in self.plugin.supported_extensions
        assert ".mysql" in self.plugin.supported_extensions
        assert ".ddl" in self.plugin.supported_extensions
        assert ".dml" in self.plugin.supported_extensions
    
    def test_error_analysis_with_invalid_data(self):
        """Test error analysis with invalid data."""
        error_data = {}
        
        analysis = self.plugin.analyze_error(error_data)
        
        assert analysis["plugin"] == "sql"
        assert analysis["language"] == "sql"
        # Should handle invalid data gracefully
        assert "category" in analysis
        assert "confidence" in analysis


if __name__ == "__main__":
    pytest.main([__file__])