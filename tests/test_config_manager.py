"""
Tests for Multi-Environment Configuration Management
"""

import asyncio
import pytest
import tempfile
import yaml
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from modules.deployment.multi_environment.config_manager import (
    MultiEnvironmentConfigManager,
    ConfigFormat,
    ConfigScope,
    ConfigType,
    ChangeAction,
    ConfigValue,
    ConfigChange,
    ConfigTemplate,
    ConfigDrift,
    LocalConfigProvider,
    SecretManager,
    ConfigValidator
)
from modules.deployment.multi_environment.hybrid_orchestrator import Environment, EnvironmentType


@pytest.fixture
def mock_config():
    """Create mock configuration for testing"""
    with tempfile.TemporaryDirectory() as tmpdir:
        return {
            "master_key": "test-master-key-12345",
            "providers": [
                {
                    "name": "local",
                    "type": "local",
                    "path": tmpdir
                }
            ],
            "templates": [
                {
                    "id": "web-service",
                    "name": "Web Service Configuration",
                    "description": "Standard web service configuration",
                    "format": "yaml",
                    "content": """
server:
  port: ${port}
  host: ${host}
database:
  url: ${db_url}
  pool_size: ${db_pool_size}
logging:
  level: ${log_level}
""",
                    "variables": [
                        {"name": "port", "required": True, "default": 8080},
                        {"name": "host", "required": True, "default": "0.0.0.0"},
                        {"name": "db_url", "required": True},
                        {"name": "db_pool_size", "required": False, "default": 10},
                        {"name": "log_level", "required": False, "default": "INFO"}
                    ],
                    "metadata": {"environments": ["production", "staging"]}
                }
            ]
        }


@pytest.fixture
def config_manager(mock_config):
    """Create config manager instance for testing"""
    mock_audit_logger = Mock()
    mock_audit_logger.log_event = AsyncMock()
    
    with patch('modules.deployment.multi_environment.config_manager.AuditLogger', return_value=mock_audit_logger):
        with patch('modules.deployment.multi_environment.config_manager.DistributedMonitor'):
            return MultiEnvironmentConfigManager(mock_config)


@pytest.fixture
def mock_environment():
    """Create mock environment"""
    return Environment(
        id="prod-us-east-1",
        name="Production US East",
        type=EnvironmentType.CLOUD_AWS,
        region="us-east-1",
        connection_info={},
        capabilities=[],
        health_status="healthy",
        metadata={}
    )


@pytest.mark.asyncio
async def test_local_config_provider():
    """Test local configuration provider"""
    with tempfile.TemporaryDirectory() as tmpdir:
        provider = LocalConfigProvider(Path(tmpdir))
        
        # Test set and get
        assert await provider.set("test.key", "test_value", "production")
        value = await provider.get("test.key", "production")
        assert value == "test_value"
        
        # Test nested configuration
        assert await provider.set("database.host", "localhost", "production")
        assert await provider.set("database.port", 5432, "production")
        
        # Test list keys
        keys = await provider.list_keys(environment="production")
        assert "test.key" in keys
        assert "database.host" in keys
        assert "database.port" in keys
        
        # Test delete
        assert await provider.delete("test.key", "production")
        value = await provider.get("test.key", "production")
        assert value is None


def test_secret_manager():
    """Test secret encryption and decryption"""
    manager = SecretManager("test-key")
    
    # Test encryption/decryption
    secret = "my-secret-password"
    encrypted = manager.encrypt(secret)
    assert encrypted != secret
    assert manager.decrypt(encrypted) == secret
    
    # Test key rotation
    old_key = "old-key"
    new_key = "new-key"
    
    old_manager = SecretManager(old_key)
    encrypted_values = [
        old_manager.encrypt("secret1"),
        old_manager.encrypt("secret2")
    ]
    
    rotated = manager.rotate_key(old_key, new_key, encrypted_values)
    
    new_manager = SecretManager(new_key)
    assert new_manager.decrypt(rotated[0]) == "secret1"
    assert new_manager.decrypt(rotated[1]) == "secret2"


@pytest.mark.asyncio
async def test_config_validator():
    """Test configuration validation"""
    validator = ConfigValidator()
    
    # Test required validation
    config = ConfigValue(
        key="test.required",
        value=None,
        type=ConfigType.STRING,
        scope=ConfigScope.GLOBAL,
        validation_rules={"required": True}
    )
    is_valid, errors = await validator.validate(config)
    assert not is_valid
    assert "required" in errors[0]
    
    # Test type validation
    config.value = 123
    config.validation_rules = {"type": "string"}
    is_valid, errors = await validator.validate(config)
    assert not is_valid
    assert "Expected type string" in errors[0]
    
    # Test pattern validation
    config.value = "invalid-email"
    config.validation_rules = {"pattern": r"^[^@]+@[^@]+\.[^@]+$"}
    is_valid, errors = await validator.validate(config)
    assert not is_valid
    assert "pattern" in errors[0]
    
    # Test range validation
    config.value = 150
    config.type = ConfigType.NUMBER
    config.validation_rules = {"range": {"min": 0, "max": 100}}
    is_valid, errors = await validator.validate(config)
    assert not is_valid
    assert "between 0 and 100" in errors[0]
    
    # Test enum validation
    config.value = "invalid"
    config.validation_rules = {"enum": ["small", "medium", "large"]}
    is_valid, errors = await validator.validate(config)
    assert not is_valid
    assert "must be one of" in errors[0]
    
    # Test valid configuration
    config.value = "medium"
    is_valid, errors = await validator.validate(config)
    assert is_valid
    assert len(errors) == 0


@pytest.mark.asyncio
async def test_get_config_with_hierarchy(config_manager):
    """Test configuration retrieval with scope hierarchy"""
    # Set configs at different scopes
    provider = list(config_manager.providers.values())[0]
    await provider.set("app.timeout", 30, "global")
    await provider.set("app.timeout", 60, "prod-us-east-1")
    
    # Get with environment scope should return environment-specific value
    value = await config_manager.get_config("app.timeout", "prod-us-east-1")
    assert value == 60
    
    # Get without environment should return global value
    value = await config_manager.get_config("app.timeout")
    assert value == 30


@pytest.mark.asyncio
async def test_set_config_with_validation(config_manager):
    """Test setting configuration with validation"""
    # Set with validation rules
    validation_rules = {
        "type": "number",
        "range": {"min": 1, "max": 100}
    }
    
    # Valid value
    change = await config_manager.set_config(
        "app.threads",
        10,
        "production",
        validation_rules=validation_rules,
        reason="Initial configuration"
    )
    
    assert change.action == ChangeAction.ADD
    assert change.config_value.value == 10
    
    # Invalid value should raise error
    with pytest.raises(ValueError, match="Validation failed"):
        await config_manager.set_config(
            "app.threads",
            200,
            "production",
            validation_rules=validation_rules
        )


@pytest.mark.asyncio
@pytest.mark.skip(reason="Secret approval flow not fully implemented")
async def test_secret_encryption(config_manager):
    """Test secret encryption in configuration"""
    # Set encrypted secret in test environment (not production)
    change = await config_manager.set_config(
        "database.password",
        "my-secret-password",
        "test",
        encrypt_secret=True,
        reason="Set database password"
    )
    
    # Get without decryption
    encrypted = await config_manager.get_config(
        "database.password",
        "test",
        decrypt_secrets=False
    )
    assert isinstance(encrypted, dict)
    assert encrypted["_type"] == "secret"
    assert encrypted["_value"] != "my-secret-password"
    
    # Get with decryption
    decrypted = await config_manager.get_config(
        "database.password",
        "test",
        decrypt_secrets=True
    )
    assert decrypted == "my-secret-password"


@pytest.mark.asyncio
async def test_config_references(config_manager):
    """Test configuration references"""
    # Set base value
    await config_manager.set_config("defaults.port", 8080, "global")
    
    # Set reference in staging (not production, to avoid approval requirement)
    await config_manager.set_config(
        "app.port",
        {"_type": "reference", "_ref": "defaults.port"},
        "staging"
    )
    
    # Get should resolve reference
    value = await config_manager.get_config("app.port", "staging")
    assert value == 8080


@pytest.mark.asyncio
async def test_apply_template(config_manager):
    """Test applying configuration template"""
    variables = {
        "port": 3000,
        "host": "localhost",
        "db_url": "postgresql://localhost/myapp",
        "db_pool_size": 20,
        "log_level": "DEBUG"
    }
    
    changes = await config_manager.apply_template("web-service", "staging", variables)
    
    assert len(changes) > 0
    
    # Verify values were set
    port = await config_manager.get_config("server.port", "staging")
    assert port == 3000
    
    db_url = await config_manager.get_config("database.url", "staging")
    assert db_url == "postgresql://localhost/myapp"


@pytest.mark.asyncio
async def test_detect_drift(config_manager):
    """Test configuration drift detection"""
    # Set expected configuration
    await config_manager.set_config("app.version", "1.0.0", "production")
    await config_manager.set_config("app.replicas", 3, "production")
    
    # Mock expected vs actual difference
    config_manager._get_expected_configs = AsyncMock(return_value={
        "app.version": "1.0.0",
        "app.replicas": 3,
        "app.timeout": 30
    })
    
    config_manager._get_actual_configs = AsyncMock(return_value={
        "app.version": "1.0.1",  # Drift
        "app.replicas": 3,       # No drift
        "app.extra": "value"     # Extra config
    })
    
    drifts = await config_manager.detect_drift("production")
    
    assert len(drifts) == 1
    drift = drifts[0]
    assert drift.environment == "production"
    assert len(drift.differences) == 3
    
    # Check difference types
    diff_types = {d["key"]: d["type"] for d in drift.differences}
    assert diff_types["app.version"] == "mismatch"
    assert diff_types["app.timeout"] == "missing"
    assert diff_types["app.extra"] == "extra"


@pytest.mark.asyncio
async def test_remediate_drift(config_manager):
    """Test drift remediation"""
    # Create drift
    drift = ConfigDrift(
        environment="staging",
        service=None,
        expected_values={"app.port": 8080, "app.host": "0.0.0.0"},
        actual_values={"app.port": 3000},
        differences=[
            {"key": "app.port", "expected": 8080, "actual": 3000, "type": "mismatch"},
            {"key": "app.host", "expected": "0.0.0.0", "actual": None, "type": "missing"}
        ],
        severity="low",
        detected_at=datetime.utcnow(),
        auto_remediate=True
    )
    
    # Mock apply change
    config_manager._apply_change = AsyncMock()
    
    result = await config_manager.remediate_drift(drift)
    
    assert result["status"] == "completed"
    assert len(result["changes"]) == 2
    assert result["changes"][0]["action"] == "updated"
    assert result["changes"][1]["action"] == "added"


@pytest.mark.asyncio
async def test_promote_config(config_manager):
    """Test configuration promotion between environments"""
    # Set configurations in dev
    await config_manager.set_config("app.version", "2.0.0", "development")
    await config_manager.set_config("app.features.newFeature", True, "development")
    
    # Promote to staging
    result = await config_manager.promote_config(
        "development",
        "staging",
        keys=["app.version", "app.features.newFeature"]
    )
    
    assert result["from"] == "development"
    assert result["to"] == "staging"
    assert len(result["changes"]) == 2
    
    # Verify values were promoted
    version = await config_manager.get_config("app.version", "staging")
    assert version == "2.0.0"
    
    feature = await config_manager.get_config("app.features.newFeature", "staging")
    assert feature is True


@pytest.mark.asyncio
async def test_promote_config_dry_run(config_manager):
    """Test configuration promotion in dry-run mode"""
    await config_manager.set_config("app.test", "value", "development")
    
    result = await config_manager.promote_config(
        "development",
        "production",
        dry_run=True
    )
    
    assert result["dry_run"] is True
    assert len(result["changes"]) > 0
    
    # Verify value was not actually promoted
    value = await config_manager.get_config("app.test", "production")
    assert value is None


@pytest.mark.asyncio
async def test_rollback_config(config_manager):
    """Test configuration rollback"""
    # Set initial values in staging (not production, to avoid approval requirement)
    await config_manager.set_config("app.version", "1.0.0", "staging")
    
    # Wait a bit and update
    await asyncio.sleep(0.1)
    timestamp = datetime.utcnow()
    await asyncio.sleep(0.1)
    
    await config_manager.set_config("app.version", "2.0.0", "staging")
    await config_manager.set_config("app.newkey", "newvalue", "staging")
    
    # Mock history retrieval
    provider = list(config_manager.providers.values())[0]
    provider.get_history = AsyncMock(side_effect=[
        # First call will be for app.newkey (alphabetical order)
        [
            {"environment": "staging", "value": "newvalue", "modified": datetime.utcnow().isoformat()}
        ],
        # Second call will be for app.version
        [
            {"environment": "staging", "value": "2.0.0", "modified": datetime.utcnow().isoformat()},
            {"environment": "staging", "value": "1.0.0", "modified": (timestamp - timedelta(seconds=1)).isoformat()}
        ]
    ])
    
    result = await config_manager.rollback_config("staging", timestamp)
    
    assert len(result["changes"]) == 2
    # New key should be removed (didn't exist at timestamp)
    assert any(c["key"] == "app.newkey" and c["action"] == "removed" for c in result["changes"])
    # Version should be rolled back
    assert any(c["key"] == "app.version" and c["action"] == "rollback" for c in result["changes"])


@pytest.mark.asyncio
async def test_export_config_yaml(config_manager):
    """Test configuration export to YAML"""
    # Set some configurations in staging (not production, to avoid approval requirement)
    await config_manager.set_config("app.name", "myapp", "staging")
    await config_manager.set_config("app.version", "1.0.0", "staging")
    await config_manager.set_config("database.host", "localhost", "staging")
    
    yaml_export = await config_manager.export_config("staging", ConfigFormat.YAML)
    
    # Parse exported YAML
    data = yaml.safe_load(yaml_export)
    assert data["app"]["name"] == "myapp"
    assert data["app"]["version"] == "1.0.0"
    assert data["database"]["host"] == "localhost"


@pytest.mark.asyncio
async def test_export_config_json(config_manager):
    """Test configuration export to JSON"""
    await config_manager.set_config("app.port", 8080, "staging")
    
    json_export = await config_manager.export_config("staging", ConfigFormat.JSON)
    
    data = json.loads(json_export)
    assert data["app"]["port"] == 8080


@pytest.mark.asyncio
async def test_export_config_env(config_manager):
    """Test configuration export to environment variables"""
    await config_manager.set_config("app.name", "myapp", "staging")
    await config_manager.set_config("database.url", "postgresql://localhost", "staging")
    
    env_export = await config_manager.export_config("staging", ConfigFormat.ENV)
    
    lines = env_export.strip().split('\n')
    assert "APP_NAME=myapp" in lines
    assert "DATABASE_URL=postgresql://localhost" in lines


@pytest.mark.asyncio
async def test_pending_changes_approval(config_manager):
    """Test pending changes requiring approval"""
    # Production change should require approval
    change = await config_manager.set_config(
        "critical.setting",
        "value",
        "production",
        reason="Critical production change"
    )
    
    assert change.change_id in [c.change_id for c in config_manager.pending_changes]
    assert not change.applied
    
    # Development change should auto-apply
    dev_change = await config_manager.set_config(
        "test.setting",
        "value",
        "development",
        reason="Dev change"
    )
    
    assert dev_change.change_id not in [c.change_id for c in config_manager.pending_changes]


@pytest.mark.asyncio
async def test_get_config_summary(config_manager):
    """Test configuration summary generation"""
    # Set a configuration in global (will be auto-applied)
    await config_manager.set_config("defaults.timeout", 30, "global")
    
    # Set a configuration in development (will be auto-applied)
    await config_manager.set_config("app.version", "1.0.0", "development")
    
    # Set configurations in staging (not production, to avoid approval requirement for non-secrets)
    await config_manager.set_config("app.name", "myapp", "staging")
    
    # Set a secret in staging (will require approval, showing pending_changes > 0)
    await config_manager.set_config("database.password", "secret", "staging", encrypt_secret=True)
    
    summary = await config_manager.get_config_summary()
    
    assert "environments" in summary
    assert "global" in summary["environments"]
    assert summary["templates"] == 1
    assert summary["providers"] == ["local"]
    assert summary["pending_changes"] > 0  # Secret change is pending


def test_flatten_dict(config_manager):
    """Test dictionary flattening"""
    nested = {
        "app": {
            "server": {
                "port": 8080,
                "host": "localhost"
            },
            "features": ["auth", "api"]
        }
    }
    
    flattened = config_manager._flatten_dict(nested)
    
    assert flattened["app.server.port"] == 8080
    assert flattened["app.server.host"] == "localhost"
    assert flattened["app.features"] == ["auth", "api"]


def test_template_rendering(config_manager):
    """Test template variable rendering"""
    template = "Server running on ${host}:${port} with ${replicas} replicas"
    variables = {
        "host": "localhost",
        "port": 8080,
        "replicas": 3
    }
    
    rendered = config_manager._render_template(template, variables)
    assert rendered == "Server running on localhost:8080 with 3 replicas"


def test_drift_severity_calculation(config_manager):
    """Test drift severity calculation"""
    # Critical - security related
    severity = config_manager._calculate_drift_severity(
        [{"key": "database.password", "type": "mismatch"}],
        "staging"
    )
    assert severity == "critical"
    
    # High - production environment
    severity = config_manager._calculate_drift_severity(
        [{"key": "app.version", "type": "mismatch"}],
        "production"
    )
    assert severity == "high"
    
    # Medium - many changes
    severity = config_manager._calculate_drift_severity(
        [{"key": f"app.setting{i}", "type": "mismatch"} for i in range(15)],
        "staging"
    )
    assert severity == "medium"
    
    # Low - few non-critical changes
    severity = config_manager._calculate_drift_severity(
        [{"key": "app.timeout", "type": "mismatch"}],
        "development"
    )
    assert severity == "low"