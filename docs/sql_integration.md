# SQL Integration

The Homeostasis SQL Language Plugin provides error analysis and patch generation for SQL database queries and configurations. It supports multiple database systems and provides intelligent error detection for common SQL issues.

## Overview

The SQL plugin enables Homeostasis to:
- Analyze SQL syntax errors across multiple database dialects
- Detect and fix database schema issues
- Handle constraint violations and data type errors
- Provide intelligent suggestions for query optimization
- Support database-specific error patterns

## Supported Database Systems

- **PostgreSQL** - Full support for PostgreSQL-specific syntax and error patterns
- **MySQL/MariaDB** - MySQL dialect support
- **SQLite** - Lightweight database error handling
- **SQL Server** - Microsoft SQL Server specific patterns
- **Oracle** - Oracle database error detection
- **MongoDB** - Basic NoSQL query support

## Key Features

### Error Detection Categories

1. **Syntax Errors**
   - Invalid SQL syntax
   - Missing keywords or punctuation
   - Malformed queries

2. **Schema Errors**
   - Table not found
   - Column not found
   - Invalid data types

3. **Constraint Violations**
   - Primary key conflicts
   - Foreign key violations
   - Unique constraint errors
   - Check constraint failures

4. **Permission Errors**
   - Access denied
   - Insufficient privileges
   - Authentication failures

5. **Connection Errors**
   - Database connectivity issues
   - Timeout errors
   - Network problems

### Intelligent Patch Generation

The SQL plugin provides context-aware fixes including:

- **Query Corrections**: Automatic syntax fixing for common SQL errors
- **Schema Suggestions**: Recommendations for missing tables or columns
- **Constraint Handling**: Solutions for constraint violations
- **Performance Optimization**: Query optimization suggestions
- **Security Best Practices**: Secure query patterns

## Usage Examples

### Basic SQL Error Analysis

```python
from homeostasis import analyze_error

# Example SQL error
error_data = {
    "error_type": "SQLError",
    "message": "relation 'users' does not exist",
    "database_type": "postgresql",
    "query": "SELECT * FROM users WHERE id = 1"
}

analysis = analyze_error(error_data, language="sql")
print(analysis["suggested_fix"])
# Output: "Verify table name spelling and existence in the database"
```

### Database-Specific Error Handling

```python
# MySQL specific error
mysql_error = {
    "error_type": "MySQLError", 
    "message": "Table 'mydb.products' doesn't exist",
    "database_type": "mysql",
    "error_code": "1146"
}

analysis = analyze_error(mysql_error, language="sql")
```

### Constraint Violation Handling

```python
# Unique constraint violation
constraint_error = {
    "error_type": "IntegrityError",
    "message": "duplicate key value violates unique constraint 'users_email_key'",
    "database_type": "postgresql"
}

analysis = analyze_error(constraint_error, language="sql")
# Provides specific suggestions for handling duplicates
```

## Configuration

### Plugin Configuration

The SQL plugin can be configured in your `homeostasis.yaml`:

```yaml
plugins:
  sql:
    enabled: true
    database_types: [postgresql, mysql, sqlite, sqlserver, oracle]
    error_detection:
      syntax_checking: true
      schema_validation: true
      constraint_checking: true
      performance_analysis: true
    patch_generation:
      auto_suggest_indexes: true
      query_optimization: true
      security_fixes: true
```

### Database-Specific Settings

```yaml
plugins:
  sql:
    postgresql:
      version: "13+"
      extensions: [postgis, uuid-ossp]
    mysql:
      version: "8.0+"
      engine: innodb
    sqlite:
      pragma_settings: true
```

## Error Pattern Recognition

### Syntax Error Patterns

The plugin recognizes common SQL syntax errors:

```sql
-- Missing comma
SELECT column1 column2 FROM table_name;
-- Fix: Add comma between columns

-- Missing FROM clause  
SELECT column1, column2;
-- Fix: Add FROM table_name

-- Unmatched quotes
SELECT * FROM table WHERE name = 'John;
-- Fix: Close the quote
```

### Schema Error Patterns

```sql
-- Table not found
SELECT * FROM non_existent_table;
-- Suggestion: Check table name spelling or create table

-- Column not found
SELECT invalid_column FROM users;
-- Suggestion: Check column exists or fix spelling
```

### Constraint Error Patterns

```sql
-- Unique violation
INSERT INTO users (email) VALUES ('duplicate@example.com');
-- Suggestion: Use INSERT ... ON CONFLICT DO UPDATE (PostgreSQL)

-- Foreign key violation
INSERT INTO orders (user_id) VALUES (999);
-- Suggestion: Ensure referenced user exists
```

## Database-Specific Features

### PostgreSQL

- **Error Code Recognition**: Handles PostgreSQL-specific error codes
- **Extension Support**: Recognizes PostGIS and other extension errors
- **Advanced Types**: Support for arrays, JSON, and custom types
- **Constraint Types**: CHECK, EXCLUDE, and partial unique constraints

### MySQL

- **Engine-Specific Errors**: InnoDB vs MyISAM specific issues
- **Charset/Collation**: Character set and collation errors
- **Partition Support**: Partitioned table error handling
- **Strict Mode**: Handles strict SQL mode violations

### SQLite

- **Pragma Recognition**: SQLite-specific PRAGMA statements
- **Type Affinity**: SQLite's flexible typing system
- **WAL Mode**: Write-Ahead Logging related errors
- **Constraint Limitations**: SQLite constraint limitations

### SQL Server

- **T-SQL Syntax**: Microsoft-specific SQL extensions
- **System Functions**: SQL Server built-in functions
- **Identity Columns**: IDENTITY and sequence handling
- **Linked Servers**: Cross-server query errors

### Oracle

- **PL/SQL Integration**: Basic PL/SQL error detection
- **Package/Procedure**: Oracle package-related errors
- **Tablespace Issues**: Storage and tablespace errors
- **Sequence Objects**: Oracle sequence handling

## Best Practices

### Query Writing

1. **Use Explicit Column Names**: Avoid `SELECT *` in production
2. **Parameterized Queries**: Prevent SQL injection attacks
3. **Proper Indexing**: Create indexes for frequently queried columns
4. **Transaction Management**: Use appropriate transaction boundaries

### Error Handling

1. **Connection Pooling**: Implement proper connection management
2. **Retry Logic**: Handle transient connection failures
3. **Graceful Degradation**: Fallback strategies for database unavailability
4. **Logging**: Comprehensive error logging for debugging

### Performance Optimization

1. **Query Analysis**: Use EXPLAIN to understand query execution
2. **Index Optimization**: Regular index maintenance and optimization
3. **Query Caching**: Implement appropriate caching strategies
4. **Batch Operations**: Use batch inserts/updates for large datasets

## Integration Examples

### Django ORM Integration

```python
# Django model with SQL error handling
from django.db import models
from homeostasis.django import SQLErrorMiddleware

class User(models.Model):
    email = models.EmailField(unique=True)
    
    def save(self, *args, **kwargs):
        try:
            super().save(*args, **kwargs)
        except IntegrityError as e:
            # Homeostasis SQL plugin analyzes the error
            analysis = analyze_error({
                "error_type": "IntegrityError",
                "message": str(e),
                "database_type": "postgresql"
            })
            # Handle based on analysis
            if analysis["root_cause"] == "unique_constraint_violation":
                # Handle duplicate email
                pass
```

### SQLAlchemy Integration

```python
from sqlalchemy import create_engine, text
from homeostasis.sqlalchemy import sql_error_handler

@sql_error_handler
def execute_query(connection, query):
    try:
        result = connection.execute(text(query))
        return result.fetchall()
    except Exception as e:
        # Automatic error analysis and suggestions
        pass
```

### Raw SQL with Error Handling

```python
import psycopg2
from homeostasis import analyze_error

def safe_execute(cursor, query, params=None):
    try:
        cursor.execute(query, params)
        return cursor.fetchall()
    except psycopg2.Error as e:
        error_data = {
            "error_type": type(e).__name__,
            "message": str(e),
            "database_type": "postgresql",
            "query": query,
            "error_code": e.pgcode
        }
        analysis = analyze_error(error_data, language="sql")
        
        # Log suggested fix
        logger.error(f"SQL Error: {analysis['suggested_fix']}")
        
        # Implement retry logic or fallback
        if analysis["category"] == "connection":
            # Retry connection
            pass
        elif analysis["category"] == "syntax":
            # Log for manual review
            pass
        
        raise
```

## Troubleshooting

### Common Issues

1. **Database Type Detection**: Ensure `database_type` is correctly specified
2. **Error Message Parsing**: Verify error messages are complete and untruncated
3. **Context Information**: Provide as much context as possible for better analysis

### Debug Mode

Enable debug logging to see detailed analysis:

```python
import logging
logging.getLogger('homeostasis.sql').setLevel(logging.DEBUG)
```

### Custom Rules

Add custom SQL error rules:

```json
{
  "rules": [
    {
      "id": "custom_sql_rule",
      "pattern": "custom error pattern",
      "category": "custom",
      "suggestion": "Custom fix suggestion",
      "confidence": "high"
    }
  ]
}
```

## Performance Considerations

- **Rule Caching**: SQL rules are compiled and cached for performance
- **Pattern Matching**: Efficient regex compilation for error detection
- **Database-Specific Optimization**: Specialized handling per database type
- **Memory Usage**: Minimal memory footprint for rule storage

## Contributing

To extend the SQL plugin:

1. Add new error patterns to `rules/sql/`
2. Implement database-specific handlers
3. Add test cases for new error types
4. Update documentation with examples

## Related Documentation

- [Error Schema](error_schema.md) - Standard error format
- [Plugin Architecture](plugin_architecture.md) - Plugin development guide
- [Database Integration](integration_guides.md) - General database integration
- [Python Integration](python_integration.md) - Python-specific SQL frameworks