# ADR-005: Error Schema Standardization

Technical Story: #ARCH-005

## Context

Homeostasis needs to process errors from dozens of programming languages, frameworks, and runtime environments. Each source has its own error format, making it difficult to build a unified analysis and fix generation system. We need a standardized error schema that can represent errors from any source while preserving source-specific information for accurate fix generation.

## Decision Drivers

- Universality: Support all programming languages and frameworks
- Completeness: Capture all relevant error information
- Extensibility: Easy to add new error types and sources
- Performance: Efficient parsing and processing
- Searchability: Enable pattern matching and analysis
- Backwards Compatibility: Support schema evolution
- Tool Integration: Work with existing logging infrastructure

## Considered Options

1. **Custom JSON Schema** - Define our own error format
2. **OpenTelemetry Events** - Use OTel event specification
3. **CloudEvents** - Adopt CloudEvents standard
4. **Elastic Common Schema** - Use ECS format
5. **Hybrid Custom+Standard** - Custom schema with standard mappings

## Decision Outcome

Chosen option: "Hybrid Custom+Standard", creating a custom schema optimized for self-healing that can map to/from standard formats like OpenTelemetry and CloudEvents, because it provides the flexibility we need while maintaining compatibility with existing tools.

### Positive Consequences

- **Optimized Structure**: Schema designed for self-healing needs
- **Standard Compatibility**: Can integrate with existing tools
- **Language Agnostic**: Works across all platforms
- **Rich Context**: Captures detailed error context
- **Evolution Support**: Versioned schema allows changes
- **Type Safety**: Strong typing prevents data issues
- **Query Optimization**: Structure enables fast searches

### Negative Consequences

- **Mapping Complexity**: Need converters for each format
- **Maintenance Overhead**: Must maintain schema definitions
- **Storage Size**: Rich context means larger records
- **Learning Curve**: Developers must understand schema
- **Version Management**: Schema changes need migration
- **Validation Cost**: Schema validation adds overhead

## Implementation Details

### Core Error Schema

```json
{
  "$schema": "https://homeostasis.io/schemas/error/v1.0.0",
  "version": "1.0.0",
  "error": {
    "id": "uuid-v4",
    "timestamp": "2024-02-10T15:30:00Z",
    "severity": "error|warning|info",
    "type": {
      "category": "syntax|runtime|compilation|configuration|performance",
      "specific": "NullPointerException|SyntaxError|etc",
      "code": "NPE001"
    },
    "message": {
      "raw": "Original error message",
      "normalized": "Standardized message",
      "tokens": ["null", "pointer", "exception"]
    },
    "source": {
      "language": "java",
      "framework": "spring",
      "runtime": "jvm",
      "version": {
        "language": "11",
        "framework": "5.3.0",
        "runtime": "11.0.12"
      }
    },
    "location": {
      "file": "/src/main/java/UserService.java",
      "line": 42,
      "column": 15,
      "function": "getUserById",
      "class": "UserService",
      "module": "user-service"
    },
    "context": {
      "code": {
        "before": ["line 40", "line 41"],
        "error_line": "line 42 with error",
        "after": ["line 43", "line 44"]
      },
      "stack_trace": [
        {
          "file": "UserService.java",
          "line": 42,
          "function": "getUserById",
          "code": "user.getName()"
        }
      ],
      "variables": {
        "user": "null",
        "userId": "12345"
      },
      "recent_changes": [
        {
          "commit": "abc123",
          "timestamp": "2024-02-10T14:00:00Z",
          "files": ["UserService.java"]
        }
      ]
    },
    "environment": {
      "host": "prod-server-01",
      "container": "user-service-7d9b5c4f6-x2kjl",
      "cluster": "production-east",
      "deployment": "blue",
      "resources": {
        "cpu_usage": 0.75,
        "memory_usage": 0.82,
        "disk_usage": 0.45
      }
    },
    "metadata": {
      "correlation_id": "req-12345",
      "user_id": "user-67890",
      "session_id": "session-abcdef",
      "request_id": "req-123",
      "tags": ["critical", "customer-facing"],
      "custom": {
        "team": "backend",
        "service": "user-management"
      }
    },
    "analysis": {
      "root_cause": "Variable 'user' is null",
      "pattern_matches": ["NPE_PATTERN_001"],
      "confidence": 0.95,
      "similar_errors": ["error-id-1", "error-id-2"],
      "suggested_fix_ids": ["fix-001", "fix-002"]
    }
  }
}
```

### Schema Versioning

```python
class SchemaVersion:
    MAJOR = 1  # Breaking changes
    MINOR = 0  # New fields (backwards compatible)
    PATCH = 0  # Bug fixes
    
    @classmethod
    def is_compatible(cls, version: str) -> bool:
        major, minor, patch = map(int, version.split('.'))
        return major == cls.MAJOR and minor >= cls.MINOR
```

### Format Converters

```python
class ErrorSchemaConverter:
    @staticmethod
    def from_opentelemetry(otel_event: dict) -> dict:
        """Convert OpenTelemetry event to Homeostasis schema"""
        return {
            "version": "1.0.0",
            "error": {
                "id": otel_event.get("trace_id"),
                "timestamp": otel_event.get("timestamp"),
                "severity": map_otel_severity(otel_event.get("severity")),
                # ... mapping logic
            }
        }
    
    @staticmethod
    def to_cloudevents(error: dict) -> dict:
        """Convert Homeostasis error to CloudEvents format"""
        return {
            "specversion": "1.0",
            "type": f"io.homeostasis.error.{error['error']['type']['category']}",
            "source": error['error']['source']['module'],
            "id": error['error']['id'],
            "time": error['error']['timestamp'],
            "data": error['error']
        }
```

### Validation

```python
from jsonschema import validate, ValidationError

class ErrorValidator:
    def __init__(self):
        self.schema = load_schema("error-schema-v1.0.0.json")
    
    def validate(self, error_data: dict) -> bool:
        try:
            validate(instance=error_data, schema=self.schema)
            return True
        except ValidationError as e:
            logger.error(f"Schema validation failed: {e}")
            return False
```

### Storage Optimization

```python
class ErrorStorage:
    def store(self, error: dict):
        # Separate hot and cold data
        hot_data = extract_hot_fields(error)  # For quick queries
        cold_data = extract_cold_fields(error)  # Full context
        
        # Store in different backends
        redis.set(f"error:hot:{error['id']}", hot_data, ex=3600)
        s3.put_object(f"errors/cold/{error['id']}.json", cold_data)
        
        # Index for searching
        elasticsearch.index(
            index="errors",
            id=error['id'],
            body=create_search_document(error)
        )
```

### Query Interface

```python
class ErrorQuery:
    def find_similar_errors(self, error: dict) -> List[dict]:
        """Find errors with similar patterns"""
        query = {
            "bool": {
                "must": [
                    {"match": {"type.specific": error['type']['specific']}},
                    {"match": {"source.language": error['source']['language']}}
                ],
                "should": [
                    {"match": {"location.function": error['location']['function']}},
                    {"match": {"message.tokens": " ".join(error['message']['tokens'])}}
                ]
            }
        }
        return elasticsearch.search(index="errors", body={"query": query})
```

### Migration Strategy

1. **Version Detection**: Auto-detect schema version
2. **Lazy Migration**: Migrate on read when needed
3. **Batch Migration**: Background job for bulk updates
4. **Compatibility Mode**: Support multiple versions temporarily

## Links

- [Error Schema Documentation](../error_schema.md)
- [ADR-003: Language Plugin Architecture](003-language-plugin-architecture.md)
- [OpenTelemetry Specification](https://opentelemetry.io/docs/)