"""
Multi-Environment Configuration Management

Provides centralized configuration management across multiple environments,
with support for hierarchical configs, secrets management, dynamic updates,
and configuration drift detection.
"""

import base64
import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from modules.monitoring.distributed_monitoring import DistributedMonitor
from modules.security.audit import AuditLogger


class ConfigFormat(Enum):
    """Configuration file formats"""

    JSON = "json"
    YAML = "yaml"
    PROPERTIES = "properties"
    INI = "ini"
    TOML = "toml"
    ENV = "env"
    XML = "xml"


class ConfigScope(Enum):
    """Configuration scope levels"""

    GLOBAL = "global"
    REGION = "region"
    ENVIRONMENT = "environment"
    SERVICE = "service"
    INSTANCE = "instance"


class ConfigType(Enum):
    """Types of configuration values"""

    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    SECRET = "secret"
    REFERENCE = "reference"


class ChangeAction(Enum):
    """Configuration change actions"""

    ADD = "add"
    UPDATE = "update"
    DELETE = "delete"
    RENAME = "rename"


@dataclass
class ConfigValue:
    """Represents a configuration value with metadata"""

    key: str
    value: Any
    type: ConfigType
    scope: ConfigScope
    environment: Optional[str] = None
    service: Optional[str] = None
    description: Optional[str] = None
    validation_rules: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_modified: datetime = field(default_factory=datetime.utcnow)
    modified_by: str = "system"
    version: int = 1
    encrypted: bool = False


@dataclass
class ConfigChange:
    """Represents a configuration change"""

    change_id: str
    action: ChangeAction
    config_value: ConfigValue
    old_value: Optional[Any] = None
    reason: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    approved: bool = False
    applied: bool = False
    rollback_value: Optional[ConfigValue] = None


@dataclass
class ConfigTemplate:
    """Template for configuration generation"""

    template_id: str
    name: str
    description: str
    format: ConfigFormat
    content: str
    variables: List[Dict[str, Any]]
    validation_schema: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConfigDrift:
    """Represents configuration drift detection result"""

    environment: str
    service: Optional[str]
    expected_values: Dict[str, Any]
    actual_values: Dict[str, Any]
    differences: List[Dict[str, Any]]
    severity: str  # low, medium, high, critical
    detected_at: datetime
    auto_remediate: bool = False


class ConfigProvider(ABC):
    """Abstract interface for configuration providers"""

    @abstractmethod
    async def get(self, key: str, environment: Optional[str] = None) -> Optional[Any]:
        """Get configuration value"""
        pass

    @abstractmethod
    async def set(
        self, key: str, value: Any, environment: Optional[str] = None
    ) -> bool:
        """Set configuration value"""
        pass

    @abstractmethod
    async def delete(self, key: str, environment: Optional[str] = None) -> bool:
        """Delete configuration value"""
        pass

    @abstractmethod
    async def list_keys(
        self, prefix: Optional[str] = None, environment: Optional[str] = None
    ) -> List[str]:
        """List configuration keys"""
        pass

    @abstractmethod
    async def get_history(self, key: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get configuration value history"""
        pass


class LocalConfigProvider(ConfigProvider):
    """Local file-based configuration provider"""

    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.configs: Dict[str, Dict[str, ConfigValue]] = {}
        self.logger = logging.getLogger(f"{__name__}.LocalConfig")
        self._load_configs()

    def _load_configs(self):
        """Load configurations from files"""
        if not self.base_path.exists():
            self.base_path.mkdir(parents=True, exist_ok=True)
            return

        for config_file in self.base_path.glob("*.yaml"):
            try:
                with open(config_file, "r") as f:
                    data = yaml.safe_load(f)
                    env_name = config_file.stem
                    self.configs[env_name] = self._parse_config_data(data, env_name)
            except Exception as e:
                self.logger.error(f"Failed to load config {config_file}: {e}")

    def _parse_config_data(
        self, data: Dict[str, Any], environment: str
    ) -> Dict[str, ConfigValue]:
        """Parse configuration data into ConfigValue objects"""
        configs = {}

        def parse_recursive(obj: Any, prefix: str = ""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    full_key = f"{prefix}.{key}" if prefix else key
                    if isinstance(value, dict) and not self._is_leaf_object(value):
                        parse_recursive(value, full_key)
                    else:
                        configs[full_key] = ConfigValue(
                            key=full_key,
                            value=value,
                            type=self._determine_type(value),
                            scope=ConfigScope.ENVIRONMENT,
                            environment=environment,
                        )

        parse_recursive(data)
        return configs

    def _is_leaf_object(self, obj: Dict[str, Any]) -> bool:
        """Check if object should be treated as a leaf value"""
        # Objects with _type field are leaf objects
        return "_type" in obj or "_value" in obj

    def _determine_type(self, value: Any) -> ConfigType:
        """Determine configuration value type"""
        if isinstance(value, bool):
            return ConfigType.BOOLEAN
        elif isinstance(value, (int, float)):
            return ConfigType.NUMBER
        elif isinstance(value, list):
            return ConfigType.ARRAY
        elif isinstance(value, dict):
            if value.get("_type") == "secret":
                return ConfigType.SECRET
            elif value.get("_type") == "reference":
                return ConfigType.REFERENCE
            return ConfigType.OBJECT
        else:
            return ConfigType.STRING

    async def get(self, key: str, environment: Optional[str] = None) -> Optional[Any]:
        """Get configuration value"""
        env = environment or "global"
        if env in self.configs and key in self.configs[env]:
            config_value = self.configs[env][key]
            return config_value.value
        return None

    async def set(
        self, key: str, value: Any, environment: Optional[str] = None
    ) -> bool:
        """Set configuration value"""
        env = environment or "global"
        if env not in self.configs:
            self.configs[env] = {}

        # Create or update ConfigValue
        if key in self.configs[env]:
            old_config = self.configs[env][key]
            old_config.value = value
            old_config.version += 1
            old_config.last_modified = datetime.utcnow()
        else:
            self.configs[env][key] = ConfigValue(
                key=key,
                value=value,
                type=self._determine_type(value),
                scope=ConfigScope.ENVIRONMENT,
                environment=env,
            )

        # Save to file
        await self._save_environment_config(env)
        return True

    async def delete(self, key: str, environment: Optional[str] = None) -> bool:
        """Delete configuration value"""
        env = environment or "global"
        if env in self.configs and key in self.configs[env]:
            del self.configs[env][key]
            await self._save_environment_config(env)
            return True
        return False

    async def list_keys(
        self, prefix: Optional[str] = None, environment: Optional[str] = None
    ) -> List[str]:
        """List configuration keys"""
        env = environment or "global"
        if env not in self.configs:
            return []

        keys = list(self.configs[env].keys())
        if prefix:
            keys = [k for k in keys if k.startswith(prefix)]

        return sorted(keys)

    async def get_history(self, key: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get configuration value history"""
        # Simple implementation - would need persistent storage for real history
        history = []
        for env, configs in self.configs.items():
            if key in configs:
                config = configs[key]
                history.append(
                    {
                        "environment": env,
                        "value": config.value,
                        "version": config.version,
                        "modified": config.last_modified.isoformat(),
                        "modified_by": config.modified_by,
                    }
                )
        return history[:limit]

    async def _save_environment_config(self, environment: str):
        """Save environment configuration to file"""
        if environment not in self.configs:
            return

        # Convert ConfigValues back to plain dict
        data = {}
        for key, config_value in self.configs[environment].items():
            # Convert dot notation back to nested dict
            parts = key.split(".")
            current = data
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = config_value.value

        # Save to file
        config_file = self.base_path / f"{environment}.yaml"
        with open(config_file, "w") as f:
            yaml.dump(data, f, default_flow_style=False)


class SecretManager:
    """Manages encryption and decryption of secrets"""

    def __init__(self, master_key: Optional[str] = None):
        if master_key:
            self.cipher = self._create_cipher(master_key)
        else:
            self.cipher = Fernet(Fernet.generate_key())
        self.logger = logging.getLogger(f"{__name__}.SecretManager")

    def _create_cipher(self, master_key: str) -> Fernet:
        """Create cipher from master key"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b"homeostasis-salt",  # In production, use random salt
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(master_key.encode()))
        return Fernet(key)

    def encrypt(self, value: str) -> str:
        """Encrypt a value"""
        return self.cipher.encrypt(value.encode()).decode()

    def decrypt(self, encrypted_value: str) -> str:
        """Decrypt a value"""
        try:
            return self.cipher.decrypt(encrypted_value.encode()).decode()
        except Exception as e:
            self.logger.error(f"Failed to decrypt value: {e}")
            raise

    def rotate_key(self, old_key: str, new_key: str, values: List[str]) -> List[str]:
        """Rotate encryption key for a list of values"""
        old_cipher = self._create_cipher(old_key)
        new_cipher = self._create_cipher(new_key)

        rotated_values = []
        for encrypted_value in values:
            try:
                # Decrypt with old key
                decrypted = old_cipher.decrypt(encrypted_value.encode()).decode()
                # Encrypt with new key
                rotated = new_cipher.encrypt(decrypted.encode()).decode()
                rotated_values.append(rotated)
            except Exception as e:
                self.logger.error(f"Failed to rotate key for value: {e}")
                rotated_values.append(
                    encrypted_value
                )  # Keep original if rotation fails

        return rotated_values


class ConfigValidator:
    """Validates configuration values against rules"""

    def __init__(self):
        self.validators = {
            "required": self._validate_required,
            "type": self._validate_type,
            "pattern": self._validate_pattern,
            "range": self._validate_range,
            "enum": self._validate_enum,
            "length": self._validate_length,
            "format": self._validate_format,
        }

    async def validate(self, config_value: ConfigValue) -> Tuple[bool, List[str]]:
        """Validate a configuration value"""
        errors = []

        for rule_name, rule_value in config_value.validation_rules.items():
            if rule_name in self.validators:
                is_valid, error = self.validators[rule_name](
                    config_value.value, rule_value
                )
                if not is_valid:
                    errors.append(f"{config_value.key}: {error}")

        return len(errors) == 0, errors

    def _validate_required(self, value: Any, required: bool) -> Tuple[bool, str]:
        """Validate required field"""
        if required and (value is None or value == ""):
            return False, "Value is required"
        return True, ""

    def _validate_type(self, value: Any, expected_type: str) -> Tuple[bool, str]:
        """Validate value type"""
        type_map = {
            "string": str,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
        }

        expected = type_map.get(expected_type)
        if expected and not isinstance(value, expected):
            return False, f"Expected type {expected_type}, got {type(value).__name__}"
        return True, ""

    def _validate_pattern(self, value: Any, pattern: str) -> Tuple[bool, str]:
        """Validate value against regex pattern"""
        if not isinstance(value, str):
            return True, ""

        if not re.match(pattern, value):
            return False, f"Value does not match pattern {pattern}"
        return True, ""

    def _validate_range(
        self, value: Any, range_spec: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Validate numeric range"""
        if not isinstance(value, (int, float)):
            return True, ""

        min_val = range_spec.get("min", float("-inf"))
        max_val = range_spec.get("max", float("inf"))

        if value < min_val or value > max_val:
            return False, f"Value must be between {min_val} and {max_val}"
        return True, ""

    def _validate_enum(self, value: Any, allowed_values: List[Any]) -> Tuple[bool, str]:
        """Validate against enumeration"""
        if value not in allowed_values:
            return False, f"Value must be one of {allowed_values}"
        return True, ""

    def _validate_length(
        self, value: Any, length_spec: Dict[str, int]
    ) -> Tuple[bool, str]:
        """Validate string or array length"""
        if not isinstance(value, (str, list)):
            return True, ""

        min_len = length_spec.get("min", 0)
        max_len = length_spec.get("max", float("inf"))

        if len(value) < min_len or len(value) > max_len:
            return False, f"Length must be between {min_len} and {max_len}"
        return True, ""

    def _validate_format(self, value: Any, format_type: str) -> Tuple[bool, str]:
        """Validate specific formats"""
        if not isinstance(value, str):
            return True, ""

        format_validators = {
            "email": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
            "url": r"^https?://[^\s]+$",
            "ipv4": r"^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$",
            "uuid": r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
        }

        pattern = format_validators.get(format_type)
        if pattern and not re.match(pattern, value):
            return False, f"Invalid {format_type} format"
        return True, ""


class MultiEnvironmentConfigManager:
    """
    Manages configuration across multiple environments with support for
    hierarchical configs, templating, validation, and drift detection.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.providers: Dict[str, ConfigProvider] = {}
        self.secret_manager = SecretManager(config.get("master_key"))
        self.validator = ConfigValidator()
        self.templates: Dict[str, ConfigTemplate] = {}
        self.pending_changes: List[ConfigChange] = []
        self.auditor = AuditLogger()
        self.monitor = DistributedMonitor()
        self.logger = logging.getLogger(__name__)

        # Initialize providers
        self._init_providers(config.get("providers", []))

        # Load templates
        self._load_templates(config.get("templates", []))

    def _init_providers(self, provider_configs: List[Dict[str, Any]]):
        """Initialize configuration providers"""
        for pconfig in provider_configs:
            provider_type = pconfig.get("type")
            if provider_type == "local":
                provider = LocalConfigProvider(Path(pconfig["path"]))
                self.providers[pconfig["name"]] = provider
            # Add more provider types as needed

    def _load_templates(self, template_configs: List[Dict[str, Any]]):
        """Load configuration templates"""
        for tconfig in template_configs:
            template = ConfigTemplate(
                template_id=tconfig["id"],
                name=tconfig["name"],
                description=tconfig.get("description", ""),
                format=ConfigFormat(tconfig["format"]),
                content=tconfig["content"],
                variables=tconfig.get("variables", []),
                validation_schema=tconfig.get("validation_schema"),
                metadata=tconfig.get("metadata", {}),
            )
            self.templates[template.template_id] = template

    async def get_config(
        self,
        key: str,
        environment: Optional[str] = None,
        scope: ConfigScope = ConfigScope.ENVIRONMENT,
        decrypt_secrets: bool = True,
    ) -> Optional[Any]:
        """Get configuration value with scope hierarchy"""
        # Try scopes in order: instance -> service -> environment -> region -> global
        scopes_to_try = self._get_scope_hierarchy(scope, environment)

        for provider_name, provider in self.providers.items():
            for env in scopes_to_try:
                value = await provider.get(key, env)
                if value is not None:
                    # Handle references
                    if isinstance(value, dict) and value.get("_type") == "reference":
                        ref_key = value.get("_ref")
                        value = await self.get_config(
                            ref_key, environment, scope, decrypt_secrets
                        )

                    # Handle secrets
                    if (
                        decrypt_secrets
                        and isinstance(value, dict)
                        and value.get("_type") == "secret"
                    ):
                        encrypted_value = value.get("_value")
                        if encrypted_value:
                            value = self.secret_manager.decrypt(encrypted_value)

                    return value

        return None

    def _get_scope_hierarchy(
        self, scope: ConfigScope, environment: Optional[str]
    ) -> List[str]:
        """Get scope hierarchy for configuration lookup"""
        hierarchy = []

        if environment:
            if scope == ConfigScope.INSTANCE:
                # Instance-specific config not implemented in this example
                pass
            if scope == ConfigScope.SERVICE:
                # Service-specific config not implemented in this example
                pass
            if scope in [
                ConfigScope.ENVIRONMENT,
                ConfigScope.SERVICE,
                ConfigScope.INSTANCE,
            ]:
                hierarchy.append(environment)

            # Extract region from environment name (e.g., "prod-us-east-1" -> "us-east-1")
            parts = environment.split("-")
            if len(parts) > 2:
                region = "-".join(parts[1:])
                hierarchy.append(f"region-{region}")

        hierarchy.append("global")
        return hierarchy

    async def set_config(
        self,
        key: str,
        value: Any,
        environment: Optional[str] = None,
        scope: ConfigScope = ConfigScope.ENVIRONMENT,
        encrypt_secret: bool = False,
        validation_rules: Optional[Dict[str, Any]] = None,
        reason: str = "",
    ) -> ConfigChange:
        """Set configuration value with change tracking"""
        # Get current value for change tracking
        old_value = await self.get_config(
            key, environment, scope, decrypt_secrets=False
        )

        # Handle secret encryption
        if encrypt_secret:
            encrypted_value = self.secret_manager.encrypt(str(value))
            value = {"_type": "secret", "_value": encrypted_value}

        # Create ConfigValue
        config_value = ConfigValue(
            key=key,
            value=value,
            type=self._determine_type(value),
            scope=scope,
            environment=environment,
            validation_rules=validation_rules or {},
            encrypted=encrypt_secret,
        )

        # Validate if rules provided
        if validation_rules:
            is_valid, errors = await self.validator.validate(config_value)
            if not is_valid:
                raise ValueError(f"Validation failed: {', '.join(errors)}")

        # Create change record
        change = ConfigChange(
            change_id=f"change_{datetime.utcnow().timestamp()}",
            action=ChangeAction.UPDATE if old_value is not None else ChangeAction.ADD,
            config_value=config_value,
            old_value=old_value,
            reason=reason,
        )

        # Apply change based on approval requirements
        if self._requires_approval(change):
            self.pending_changes.append(change)
            return change
        else:
            await self._apply_change(change)
            return change

    def _determine_type(self, value: Any) -> ConfigType:
        """Determine configuration value type"""
        if isinstance(value, dict):
            if value.get("_type") == "secret":
                return ConfigType.SECRET
            elif value.get("_type") == "reference":
                return ConfigType.REFERENCE

        if isinstance(value, bool):
            return ConfigType.BOOLEAN
        elif isinstance(value, (int, float)):
            return ConfigType.NUMBER
        elif isinstance(value, list):
            return ConfigType.ARRAY
        elif isinstance(value, dict):
            return ConfigType.OBJECT
        else:
            return ConfigType.STRING

    def _requires_approval(self, change: ConfigChange) -> bool:
        """Check if change requires approval"""
        # Production changes require approval
        if (
            change.config_value.environment
            and "prod" in change.config_value.environment
        ):
            return True

        # Secret changes require approval
        if change.config_value.type == ConfigType.SECRET:
            return True

        # Deletions require approval
        if change.action == ChangeAction.DELETE:
            return True

        return False

    async def _apply_change(self, change: ConfigChange):
        """Apply configuration change"""
        config_value = change.config_value

        # Find appropriate provider
        provider = list(self.providers.values())[0]  # Simple selection

        if change.action == ChangeAction.ADD or change.action == ChangeAction.UPDATE:
            success = await provider.set(
                config_value.key, config_value.value, config_value.environment
            )
        elif change.action == ChangeAction.DELETE:
            success = await provider.delete(config_value.key, config_value.environment)
        else:
            success = False

        change.applied = success

        # Audit log
        await self.auditor.log_event(
            "config_change",
            {
                "change_id": change.change_id,
                "action": change.action.value,
                "key": config_value.key,
                "environment": config_value.environment,
                "success": success,
                "reason": change.reason,
            },
        )

    async def apply_template(
        self, template_id: str, environment: str, variables: Dict[str, Any]
    ) -> List[ConfigChange]:
        """Apply configuration template to environment"""
        if template_id not in self.templates:
            raise ValueError(f"Template {template_id} not found")

        template = self.templates[template_id]

        # Validate variables
        missing_vars = [
            v["name"]
            for v in template.variables
            if v.get("required", False) and v["name"] not in variables
        ]
        if missing_vars:
            raise ValueError(f"Missing required variables: {missing_vars}")

        # Render template
        content = self._render_template(template.content, variables)

        # Parse rendered content based on format
        if template.format == ConfigFormat.YAML:
            configs = yaml.safe_load(content)
        elif template.format == ConfigFormat.JSON:
            configs = json.loads(content)
        else:
            raise ValueError(f"Unsupported template format: {template.format}")

        # Apply configurations
        changes = []
        for key, value in self._flatten_dict(configs).items():
            change = await self.set_config(
                key, value, environment, reason=f"Applied from template {template_id}"
            )
            changes.append(change)

        return changes

    def _render_template(self, template: str, variables: Dict[str, Any]) -> str:
        """Render template with variables"""
        # Simple variable substitution
        rendered = template
        for key, value in variables.items():
            rendered = rendered.replace(f"${{{key}}}", str(value))
            rendered = rendered.replace(f"${key}", str(value))  # Alternative syntax
        return rendered

    def _flatten_dict(self, d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        """Flatten nested dictionary to dot notation"""
        flattened = {}

        for key, value in d.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict) and not any(
                k.startswith("_") for k in value.keys()
            ):
                flattened.update(self._flatten_dict(value, full_key))
            else:
                flattened[full_key] = value

        return flattened

    async def detect_drift(
        self, environment: str, service: Optional[str] = None
    ) -> List[ConfigDrift]:
        """Detect configuration drift in environment"""
        drifts = []

        # Get expected configuration (from templates or baseline)
        expected_configs = await self._get_expected_configs(environment, service)

        # Get actual configuration
        actual_configs = await self._get_actual_configs(environment, service)

        # Compare configurations
        all_keys = set(expected_configs.keys()) | set(actual_configs.keys())

        differences = []
        for key in all_keys:
            expected = expected_configs.get(key)
            actual = actual_configs.get(key)

            if expected != actual:
                differences.append(
                    {
                        "key": key,
                        "expected": expected,
                        "actual": actual,
                        "type": (
                            "missing"
                            if actual is None
                            else "extra" if expected is None else "mismatch"
                        ),
                    }
                )

        if differences:
            # Calculate severity based on drift
            severity = self._calculate_drift_severity(differences, environment)

            drift = ConfigDrift(
                environment=environment,
                service=service,
                expected_values=expected_configs,
                actual_values=actual_configs,
                differences=differences,
                severity=severity,
                detected_at=datetime.utcnow(),
                auto_remediate=severity in ["low", "medium"]
                and not any(
                    d.get("key", "").endswith((".secret", ".password", ".key"))
                    for d in differences
                ),
            )
            drifts.append(drift)

        return drifts

    async def _get_expected_configs(
        self, environment: str, service: Optional[str]
    ) -> Dict[str, Any]:
        """Get expected configuration for environment"""
        # In practice, would retrieve from templates or baseline
        expected = {}

        # Get from templates
        for template in self.templates.values():
            if environment in template.metadata.get("environments", []):
                # Render template with default variables
                default_vars = {
                    v["name"]: v.get("default", "") for v in template.variables
                }
                content = self._render_template(template.content, default_vars)

                if template.format == ConfigFormat.YAML:
                    configs = yaml.safe_load(content)
                elif template.format == ConfigFormat.JSON:
                    configs = json.loads(content)
                else:
                    continue

                expected.update(self._flatten_dict(configs))

        return expected

    async def _get_actual_configs(
        self, environment: str, service: Optional[str]
    ) -> Dict[str, Any]:
        """Get actual configuration for environment"""
        actual = {}

        # Get from providers
        for provider in self.providers.values():
            keys = await provider.list_keys(environment=environment)
            for key in keys:
                value = await provider.get(key, environment)
                if value is not None:
                    actual[key] = value

        return actual

    def _calculate_drift_severity(
        self, differences: List[Dict[str, Any]], environment: str
    ) -> str:
        """Calculate severity of configuration drift"""
        # Critical if security-related configs changed
        if any(
            d["key"].endswith((".secret", ".password", ".key", ".token"))
            for d in differences
        ):
            return "critical"

        # High if production environment
        if "prod" in environment:
            return "high"

        # Medium if many changes
        if len(differences) > 10:
            return "medium"

        return "low"

    async def remediate_drift(self, drift: ConfigDrift) -> Dict[str, Any]:
        """Remediate configuration drift"""
        if not drift.auto_remediate:
            return {
                "status": "manual_intervention_required",
                "severity": drift.severity,
            }

        results = {"status": "remediating", "changes": []}

        for diff in drift.differences:
            key = diff["key"]
            expected = diff["expected"]
            actual = diff["actual"]

            try:
                if diff["type"] == "missing":
                    # Add missing configuration
                    change = await self.set_config(
                        key,
                        expected,
                        drift.environment,
                        reason="Drift remediation - adding missing config",
                    )
                    results["changes"].append(
                        {"key": key, "action": "added", "success": True}
                    )

                elif diff["type"] == "extra":
                    # Remove extra configuration
                    change = ConfigChange(
                        change_id=f"drift_{datetime.utcnow().timestamp()}",
                        action=ChangeAction.DELETE,
                        config_value=ConfigValue(
                            key=key,
                            value=actual,
                            type=ConfigType.STRING,
                            scope=ConfigScope.ENVIRONMENT,
                            environment=drift.environment,
                        ),
                        reason="Drift remediation - removing extra config",
                    )
                    await self._apply_change(change)
                    results["changes"].append(
                        {"key": key, "action": "removed", "success": True}
                    )

                elif diff["type"] == "mismatch":
                    # Update mismatched configuration
                    change = await self.set_config(
                        key,
                        expected,
                        drift.environment,
                        reason="Drift remediation - correcting mismatch",
                    )
                    results["changes"].append(
                        {"key": key, "action": "updated", "success": True}
                    )

            except Exception as e:
                self.logger.error(f"Failed to remediate drift for {key}: {e}")
                results["changes"].append(
                    {"key": key, "action": "failed", "error": str(e)}
                )

        results["status"] = "completed"
        return results

    async def promote_config(
        self,
        from_environment: str,
        to_environment: str,
        keys: Optional[List[str]] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Promote configuration from one environment to another"""
        results = {
            "from": from_environment,
            "to": to_environment,
            "changes": [],
            "dry_run": dry_run,
        }

        # Get keys to promote
        if keys is None:
            # Promote all keys
            provider = list(self.providers.values())[0]
            keys = await provider.list_keys(environment=from_environment)

        for key in keys:
            try:
                # Get value from source environment
                value = await self.get_config(
                    key, from_environment, decrypt_secrets=False
                )
                if value is None:
                    continue

                # Check if value exists in target environment
                existing_value = await self.get_config(
                    key, to_environment, decrypt_secrets=False
                )

                if existing_value != value:
                    change_info = {
                        "key": key,
                        "action": "update" if existing_value is not None else "add",
                        "old_value": existing_value,
                        "new_value": value,
                    }

                    if not dry_run:
                        # Apply change
                        change = await self.set_config(
                            key,
                            value,
                            to_environment,
                            reason=f"Promoted from {from_environment}",
                        )
                        change_info["applied"] = change.applied

                    results["changes"].append(change_info)

            except Exception as e:
                self.logger.error(f"Failed to promote {key}: {e}")
                results["changes"].append(
                    {"key": key, "action": "error", "error": str(e)}
                )

        return results

    async def rollback_config(
        self, environment: str, timestamp: datetime
    ) -> Dict[str, Any]:
        """Rollback configuration to a specific point in time"""
        results = {
            "environment": environment,
            "target_timestamp": timestamp.isoformat(),
            "changes": [],
        }

        # Get all keys in environment
        provider = list(self.providers.values())[0]
        keys = await provider.list_keys(environment=environment)

        for key in keys:
            try:
                # Get historical value at timestamp
                history = await provider.get_history(key)

                # Find value at or before timestamp
                historical_value = None
                for entry in history:
                    entry_time = datetime.fromisoformat(entry["modified"])
                    if entry_time <= timestamp and entry["environment"] == environment:
                        historical_value = entry["value"]
                        break

                # Get current value
                current_value = await provider.get(key, environment)

                if historical_value != current_value:
                    # Rollback to historical value
                    if historical_value is not None:
                        change = await self.set_config(
                            key,
                            historical_value,
                            environment,
                            reason=f"Rollback to {timestamp.isoformat()}",
                        )
                        results["changes"].append(
                            {
                                "key": key,
                                "action": "rollback",
                                "from": current_value,
                                "to": historical_value,
                            }
                        )
                    else:
                        # Key didn't exist at timestamp, remove it
                        change = ConfigChange(
                            change_id=f"rollback_{datetime.utcnow().timestamp()}",
                            action=ChangeAction.DELETE,
                            config_value=ConfigValue(
                                key=key,
                                value=current_value,
                                type=ConfigType.STRING,
                                scope=ConfigScope.ENVIRONMENT,
                                environment=environment,
                            ),
                            reason=f"Rollback - key didn't exist at {timestamp.isoformat()}",
                        )
                        await self._apply_change(change)
                        results["changes"].append({"key": key, "action": "removed"})

            except Exception as e:
                self.logger.error(f"Failed to rollback {key}: {e}")
                results["changes"].append(
                    {"key": key, "action": "error", "error": str(e)}
                )

        return results

    async def export_config(
        self, environment: str, format: ConfigFormat, include_secrets: bool = False
    ) -> str:
        """Export configuration in specified format"""
        # Get all configurations
        provider = list(self.providers.values())[0]
        keys = await provider.list_keys(environment=environment)

        configs = {}
        for key in keys:
            value = await self.get_config(
                key, environment, decrypt_secrets=include_secrets
            )
            if value is not None:
                # Convert dot notation to nested dict
                parts = key.split(".")
                current = configs
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = value

        # Convert to requested format
        if format == ConfigFormat.YAML:
            return yaml.dump(configs, default_flow_style=False)
        elif format == ConfigFormat.JSON:
            return json.dumps(configs, indent=2)
        elif format == ConfigFormat.ENV:
            # Flatten to environment variables
            env_vars = []
            for key, value in self._flatten_dict(configs).items():
                env_key = key.upper().replace(".", "_").replace("-", "_")
                env_vars.append(f"{env_key}={value}")
            return "\n".join(env_vars)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    async def get_config_summary(self) -> Dict[str, Any]:
        """Get summary of configuration state across environments"""
        summary = {
            "environments": {},
            "pending_changes": len(self.pending_changes),
            "templates": len(self.templates),
            "providers": list(self.providers.keys()),
        }

        # Get summary for each environment
        for provider_name, provider in self.providers.items():
            # Get all environments (simplified - assumes environment names in keys)
            environments = set()
            for key in await provider.list_keys():
                # Extract environment from provider structure
                environments.add("global")  # Simplified

            for env in environments:
                if env not in summary["environments"]:
                    summary["environments"][env] = {
                        "total_keys": 0,
                        "secrets": 0,
                        "references": 0,
                        "last_modified": None,
                    }

                keys = await provider.list_keys(environment=env)
                summary["environments"][env]["total_keys"] += len(keys)

                # Count secrets and references
                for key in keys:
                    value = await provider.get(key, env)
                    if isinstance(value, dict):
                        if value.get("_type") == "secret":
                            summary["environments"][env]["secrets"] += 1
                        elif value.get("_type") == "reference":
                            summary["environments"][env]["references"] += 1

        return summary
