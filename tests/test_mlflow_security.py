"""
Tests for MLflow security measures.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from modules.security.mlflow_security import (
    MLflowSecurityConfig,
    ModelSandbox,
    SecurityError,
    create_secure_mlflow_config,
    load_model_securely,
    secure_model_loader,
)


class TestMLflowSecurityConfig:
    """Test MLflow security configuration."""

    def test_init_default(self):
        """Test default initialization."""
        config = MLflowSecurityConfig()
        assert config.enable_model_validation is True
        assert config.enable_sandboxing is True
        assert config.max_model_size_mb == 1000

    def test_load_from_env(self):
        """Test loading trusted sources from environment."""
        with patch.dict(
            os.environ, {"MLFLOW_TRUSTED_SOURCES": "s3://bucket1,file:///trusted/"}
        ):
            config = MLflowSecurityConfig()
            assert "s3://bucket1" in config.trusted_model_sources
            assert "file:///trusted/" in config.trusted_model_sources

    def test_load_from_config_file(self):
        """Test loading configuration from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / "config"
            config_dir.mkdir()
            config_file = config_dir / "mlflow_security.json"

            config_data = {
                "trusted_sources": ["s3://secure-bucket/"],
                "allowed_hashes": {"model.pkl": "abc123"},
                "max_model_size_mb": 500,
            }

            with open(config_file, "w") as f:
                json.dump(config_data, f)

            with patch("modules.security.mlflow_security.Path") as mock_path:
                mock_path.return_value.exists.return_value = True
                mock_path.return_value.open = open(config_file)

                config = MLflowSecurityConfig()
                assert config.max_model_size_mb == 500

    def test_is_trusted_source(self):
        """Test trusted source validation."""
        config = MLflowSecurityConfig()
        config.trusted_model_sources = ["s3://trusted/", "file:///secure/"]

        assert config.is_trusted_source("s3://trusted/model1")
        assert config.is_trusted_source("file:///secure/models/prod")
        assert not config.is_trusted_source("s3://untrusted/model")
        assert not config.is_trusted_source("http://public/model")

    def test_is_trusted_source_empty(self):
        """Test behavior when no trusted sources configured."""
        config = MLflowSecurityConfig()
        config.trusted_model_sources = []

        assert not config.is_trusted_source("s3://any/model")

    def test_calculate_model_hash(self):
        """Test model hash calculation."""
        config = MLflowSecurityConfig()

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b"test model content")
            tmp.flush()

            hash1 = config.calculate_model_hash(tmp.name)
            assert len(hash1) == 64  # SHA256 hex length

            # Same content should produce same hash
            hash2 = config.calculate_model_hash(tmp.name)
            assert hash1 == hash2

            os.unlink(tmp.name)

    def test_validate_model_hash(self):
        """Test model hash validation."""
        config = MLflowSecurityConfig()

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b"test model content")
            tmp.flush()

            correct_hash = config.calculate_model_hash(tmp.name)

            # Test with explicit hash
            assert config.validate_model_hash(tmp.name, correct_hash)
            assert not config.validate_model_hash(tmp.name, "wrong_hash")

            # Test with allowed hashes
            config.allowed_model_hashes[Path(tmp.name).name] = correct_hash
            assert config.validate_model_hash(tmp.name)

            os.unlink(tmp.name)

    def test_check_model_size(self):
        """Test model size validation."""
        config = MLflowSecurityConfig()
        config.max_model_size_mb = 1

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            # Write 0.5 MB - should pass
            tmp.write(b"x" * (512 * 1024))
            tmp.flush()
            assert config.check_model_size(tmp.name)

            # Write 1.5 MB - should fail
            tmp.seek(0)
            tmp.write(b"x" * (1536 * 1024))
            tmp.flush()
            assert not config.check_model_size(tmp.name)

            os.unlink(tmp.name)


class TestSecureModelLoader:
    """Test secure model loader decorator."""

    def test_untrusted_source_blocked(self):
        """Test that untrusted sources are blocked."""
        config = MLflowSecurityConfig()
        config.trusted_model_sources = ["s3://trusted/"]

        @secure_model_loader(config)
        def load_model(uri):
            return "model_loaded"

        with pytest.raises(SecurityError) as exc:
            load_model("s3://untrusted/model")

        assert "not trusted" in str(exc.value)

    def test_trusted_source_allowed(self):
        """Test that trusted sources are allowed."""
        config = MLflowSecurityConfig()
        config.trusted_model_sources = ["s3://trusted/"]

        @secure_model_loader(config)
        def load_model(uri):
            return "model_loaded"

        result = load_model("s3://trusted/model1")
        assert result == "model_loaded"

    def test_local_file_size_check(self):
        """Test file size validation for local files."""
        config = MLflowSecurityConfig()
        config.trusted_model_sources = ["file://"]
        config.max_model_size_mb = 1

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            # Write 2 MB file
            tmp.write(b"x" * (2 * 1024 * 1024))
            tmp.flush()

            @secure_model_loader(config)
            def load_model(uri):
                return "model_loaded"

            with pytest.raises(SecurityError) as exc:
                load_model(f"file://{tmp.name}")

            assert "size exceeds limit" in str(exc.value)
            os.unlink(tmp.name)

    def test_local_file_hash_validation(self):
        """Test hash validation for local files."""
        config = MLflowSecurityConfig()
        config.trusted_model_sources = ["file://"]

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b"model content")
            tmp.flush()

            # Add wrong hash
            config.allowed_model_hashes[Path(tmp.name).name] = "wrong_hash"

            @secure_model_loader(config)
            def load_model(uri):
                return "model_loaded"

            with pytest.raises(SecurityError) as exc:
                load_model(f"file://{tmp.name}")

            assert "hash validation failed" in str(exc.value)
            os.unlink(tmp.name)

    def test_exception_handling(self):
        """Test exception handling during model loading."""
        config = MLflowSecurityConfig()
        config.trusted_model_sources = ["s3://trusted/"]

        @secure_model_loader(config)
        def load_model(uri):
            raise ValueError("Loading error")

        with pytest.raises(ValueError) as exc:
            load_model("s3://trusted/model")

        assert "Loading error" in str(exc.value)


class TestModelSandbox:
    """Test model sandboxing functionality."""

    def test_sandbox_init(self):
        """Test sandbox initialization."""
        sandbox = ModelSandbox(enable_network=False, memory_limit_mb=1024)
        assert sandbox.enable_network is False
        assert sandbox.memory_limit_mb == 1024

    def test_run_in_sandbox_success(self):
        """Test successful execution in sandbox."""
        sandbox = ModelSandbox()

        def model_func(x):
            return x * 2

        result = sandbox.run_in_sandbox(model_func, 5)
        assert result == 10

    def test_run_in_sandbox_failure(self):
        """Test exception handling in sandbox."""
        sandbox = ModelSandbox()

        def model_func():
            raise RuntimeError("Model error")

        with pytest.raises(RuntimeError) as exc:
            sandbox.run_in_sandbox(model_func)

        assert "Model error" in str(exc.value)


class TestSecurityHelpers:
    """Test security helper functions."""

    def test_create_secure_mlflow_config(self):
        """Test secure configuration generation."""
        config = create_secure_mlflow_config()

        assert config["mlflow.disable_auto_logging"] is True
        assert config["mlflow.authentication.enabled"] is True
        assert config["mlflow.models.validate_signature"] is True
        assert config["mlflow.recipes.enabled"] is False
        assert config["mlflow.server.audit_log_enabled"] is True

    @patch("modules.security.mlflow_security.mlflow.pyfunc")
    def test_load_model_securely(self, mock_pyfunc):
        """Test secure model loading helper."""
        mock_pyfunc.load_model.return_value = MagicMock()

        config = MLflowSecurityConfig()
        config.trusted_model_sources = ["s3://trusted/"]

        # Should work with trusted source
        load_model_securely("s3://trusted/model", config)
        assert mock_pyfunc.load_model.called

        # Should fail with untrusted source
        with pytest.raises(SecurityError):
            load_model_securely("s3://untrusted/model", config)


class TestIntegration:
    """Integration tests for MLflow security."""

    @patch("modules.security.mlflow_security.mlflow")
    def test_full_security_workflow(self, mock_mlflow):
        """Test complete security workflow."""
        # Setup
        config = MLflowSecurityConfig()
        config.trusted_model_sources = ["s3://prod-models/"]

        with tempfile.NamedTemporaryFile() as tmp:
            tmp.write(b"model data")
            tmp.flush()

            # Add model hash
            model_hash = config.calculate_model_hash(tmp.name)
            config.allowed_model_hashes["model.pkl"] = model_hash

            # Mock mlflow
            mock_model = MagicMock()
            mock_mlflow.pyfunc.load_model.return_value = mock_model

            # Test loading from trusted source
            @secure_model_loader(config)
            def load_and_predict(model_uri):
                model = mock_mlflow.pyfunc.load_model(model_uri)
                return model.predict([1, 2, 3])

            mock_model.predict.return_value = [4, 5, 6]
            result = load_and_predict("s3://prod-models/model1")
            assert result == [4, 5, 6]

            # Test untrusted source
            with pytest.raises(SecurityError):
                load_and_predict("https://evil.com/malicious-model")
