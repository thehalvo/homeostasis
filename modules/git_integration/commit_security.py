"""
Commit signing and verification for healing changes.

This module handles GPG signing of healing-generated commits and verification
of commit authenticity to maintain a secure audit trail for automated changes.
"""

import hashlib
import hmac
import json
import os
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from modules.monitoring.logger import HomeostasisLogger


class CommitSecurity:
    """Handles commit signing and verification for automated healing."""

    def __init__(self, repo_path: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize commit security manager.

        Args:
            repo_path: Path to the Git repository
            config: Configuration dictionary for commit security
        """
        self.repo_path = Path(repo_path)
        self.config = config or self._load_default_config()
        self.logger = HomeostasisLogger(__name__)

        # Initialize GPG configuration
        self.gpg_available = self._check_gpg_available()
        self.signing_enabled = self.config.get("enabled", False) and self.gpg_available

        # Homeostasis signature components
        self.homeostasis_key_id = self.config.get("homeostasis_key_id")
        self.healing_signature_prefix = "HOMEOSTASIS-HEALING"

        # Security settings
        self.require_verification = self.config.get("require_verification", False)
        self.allowed_signers = set(self.config.get("allowed_signers", []))

    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration for commit security."""
        return {
            "enabled": False,
            "gpg_key_id": None,
            "homeostasis_key_id": None,
            "require_verification": False,
            "allowed_signers": [],
            "signature_algorithm": "sha256",
            "healing_metadata": {
                "include_rule_id": True,
                "include_confidence": True,
                "include_timestamp": True,
                "include_checksum": True,
            },
            "audit_trail": {"enabled": True, "log_file": ".homeostasis-audit.log"},
        }

    def _check_gpg_available(self) -> bool:
        """Check if GPG is available on the system."""
        try:
            result = subprocess.run(
                ["gpg", "--version"], capture_output=True, text=True, timeout=10
            )
            return result.returncode == 0
        except Exception:
            return False

    def setup_gpg_key(self, key_id: str, passphrase: Optional[str] = None) -> bool:
        """
        Setup GPG key for signing healing commits.

        Args:
            key_id: GPG key ID to use for signing
            passphrase: Optional passphrase for the key

        Returns:
            True if key setup was successful
        """
        if not self.gpg_available:
            self.logger.error("GPG is not available on this system")
            return False

        try:
            # Check if key exists
            result = subprocess.run(
                ["gpg", "--list-secret-keys", key_id],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                self.logger.error(f"GPG key {key_id} not found")
                return False

            # Configure Git to use the key
            subprocess.run(
                ["git", "config", "user.signingkey", key_id],
                cwd=self.repo_path,
                check=True,
            )

            subprocess.run(
                ["git", "config", "commit.gpgsign", "true"],
                cwd=self.repo_path,
                check=True,
            )

            # Update configuration
            self.config["gpg_key_id"] = key_id
            self.signing_enabled = True

            self.logger.info(f"GPG key {key_id} configured for commit signing")
            return True

        except Exception as e:
            self.logger.error(f"Error setting up GPG key: {e}")
            return False

    def generate_homeostasis_key(self) -> Optional[str]:
        """
        Generate a dedicated GPG key for Homeostasis healing operations.

        Returns:
            Key ID if generation was successful, None otherwise
        """
        if not self.gpg_available:
            self.logger.error("GPG is not available on this system")
            return None

        try:
            # Create key generation configuration
            key_config = """
%echo Generating Homeostasis healing key
Key-Type: RSA
Key-Length: 4096
Subkey-Type: RSA
Subkey-Length: 4096
Name-Real: Homeostasis Healing Bot
Name-Email: homeostasis@localhost
Expire-Date: 2y
%no-protection
%commit
%echo done
"""

            # Write config to temporary file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".conf", delete=False
            ) as f:
                f.write(key_config.strip())
                config_file = f.name

            try:
                # Generate key
                result = subprocess.run(
                    ["gpg", "--batch", "--generate-key", config_file],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )

                if result.returncode != 0:
                    self.logger.error(f"Failed to generate GPG key: {result.stderr}")
                    return None

                # Get the generated key ID
                list_result = subprocess.run(
                    [
                        "gpg",
                        "--list-secret-keys",
                        "--with-colons",
                        "homeostasis@localhost",
                    ],
                    capture_output=True,
                    text=True,
                )

                if list_result.returncode == 0:
                    for line in list_result.stdout.split("\n"):
                        if line.startswith("sec:"):
                            key_id = line.split(":")[4]
                            self.config["homeostasis_key_id"] = key_id
                            self.logger.info(f"Generated Homeostasis key: {key_id}")
                            return key_id

                return None

            finally:
                # Clean up temporary file
                os.unlink(config_file)

        except Exception as e:
            self.logger.error(f"Error generating Homeostasis key: {e}")
            return None

    def sign_healing_commit(
        self, commit_message: str, healing_metadata: Dict[str, Any]
    ) -> str:
        """
        Create a signed commit message with healing metadata.

        Args:
            commit_message: Original commit message
            healing_metadata: Metadata about the healing operation

        Returns:
            Enhanced commit message with signature and metadata
        """
        # Add healing metadata to commit message
        enhanced_message = self._add_healing_metadata(commit_message, healing_metadata)

        if not self.signing_enabled:
            return enhanced_message

        # Add Homeostasis signature
        signature = self._create_homeostasis_signature(healing_metadata)
        enhanced_message += f"\n\n{self.healing_signature_prefix}: {signature}"

        return enhanced_message

    def _add_healing_metadata(self, message: str, metadata: Dict[str, Any]) -> str:
        """Add healing metadata to commit message."""
        metadata_config = self.config.get("healing_metadata", {})

        metadata_lines = []

        if metadata_config.get("include_rule_id") and "rule_id" in metadata:
            metadata_lines.append(f"Healing-Rule: {metadata['rule_id']}")

        if metadata_config.get("include_confidence") and "confidence" in metadata:
            metadata_lines.append(f"Healing-Confidence: {metadata['confidence']:.3f}")

        if metadata_config.get("include_timestamp"):
            timestamp = datetime.now().isoformat()
            metadata_lines.append(f"Healing-Timestamp: {timestamp}")

        if metadata_config.get("include_checksum") and "file_content" in metadata:
            checksum = hashlib.sha256(metadata["file_content"].encode()).hexdigest()[
                :16
            ]
            metadata_lines.append(f"Healing-Checksum: {checksum}")

        if metadata.get("branch"):
            metadata_lines.append(f"Healing-Branch: {metadata['branch']}")

        if metadata.get("original_error"):
            metadata_lines.append(
                f"Healing-Original-Error: {metadata['original_error']}"
            )

        if metadata_lines:
            return message + "\n\n" + "\n".join(metadata_lines)

        return message

    def _create_homeostasis_signature(self, metadata: Dict[str, Any]) -> str:
        """Create a Homeostasis-specific signature for healing commits."""
        # Create signature payload
        payload_data = {
            "timestamp": datetime.now().isoformat(),
            "rule_id": metadata.get("rule_id", ""),
            "confidence": metadata.get("confidence", 0.0),
            "file_path": metadata.get("file_path", ""),
            "healing_type": metadata.get("healing_type", "auto"),
        }

        payload_str = json.dumps(payload_data, sort_keys=True)

        # Create HMAC signature
        secret_key = self._get_or_create_secret_key()
        signature = hmac.new(
            secret_key.encode(), payload_str.encode(), hashlib.sha256
        ).hexdigest()

        return f"{signature}:{payload_str}"

    def _get_or_create_secret_key(self) -> str:
        """Get or create a secret key for HMAC signatures."""
        key_file = self.repo_path / ".homeostasis-key"

        if key_file.exists():
            try:
                with open(key_file, "r") as f:
                    return f.read().strip()
            except Exception:
                pass

        # Generate new key
        import secrets

        secret_key = secrets.token_hex(32)

        try:
            with open(key_file, "w") as f:
                f.write(secret_key)

            # Add to .gitignore
            gitignore_path = self.repo_path / ".gitignore"
            gitignore_content = ""

            if gitignore_path.exists():
                with open(gitignore_path, "r") as f:
                    gitignore_content = f.read()

            if ".homeostasis-key" not in gitignore_content:
                with open(gitignore_path, "a") as f:
                    f.write("\n.homeostasis-key\n")

            return secret_key

        except Exception as e:
            self.logger.error(f"Error creating secret key: {e}")
            return "default-key-insecure"

    def verify_healing_commit(self, commit_hash: str) -> Dict[str, Any]:
        """
        Verify a healing commit's authenticity and metadata.

        Args:
            commit_hash: Git commit hash to verify

        Returns:
            Verification results
        """
        try:
            # Get commit message
            result = subprocess.run(
                ["git", "show", "--format=%B", "--no-patch", commit_hash],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                return {"verified": False, "error": "Could not retrieve commit"}

            commit_message = result.stdout.strip()

            # Check if it's a healing commit
            if self.healing_signature_prefix not in commit_message:
                return {"verified": False, "error": "Not a healing commit"}

            # Extract and verify signature
            verification_result = self._verify_homeostasis_signature(commit_message)

            # Verify GPG signature if enabled
            gpg_verification = (
                self._verify_gpg_signature(commit_hash)
                if self.signing_enabled
                else None
            )

            # Extract metadata
            metadata = self._extract_healing_metadata(commit_message)

            return {
                "verified": verification_result["valid"],
                "homeostasis_signature": verification_result,
                "gpg_signature": gpg_verification,
                "metadata": metadata,
                "commit_hash": commit_hash,
            }

        except Exception as e:
            self.logger.error(f"Error verifying commit {commit_hash}: {e}")
            return {"verified": False, "error": str(e)}

    def _verify_homeostasis_signature(self, commit_message: str) -> Dict[str, Any]:
        """Verify Homeostasis signature in commit message."""
        try:
            # Find signature line
            lines = commit_message.split("\n")
            signature_line = None

            for line in lines:
                if line.startswith(f"{self.healing_signature_prefix}:"):
                    signature_line = line[
                        len(self.healing_signature_prefix) + 1 :
                    ].strip()
                    break

            if not signature_line:
                return {"valid": False, "error": "No Homeostasis signature found"}

            # Parse signature
            try:
                signature, payload_str = signature_line.split(":", 1)
                payload_data = json.loads(payload_str)
            except (ValueError, json.JSONDecodeError):
                return {"valid": False, "error": "Invalid signature format"}

            # Verify HMAC
            secret_key = self._get_or_create_secret_key()
            expected_signature = hmac.new(
                secret_key.encode(), payload_str.encode(), hashlib.sha256
            ).hexdigest()

            if not hmac.compare_digest(signature, expected_signature):
                return {"valid": False, "error": "Signature verification failed"}

            return {
                "valid": True,
                "payload": payload_data,
                "signature_algorithm": "hmac-sha256",
            }

        except Exception as e:
            return {"valid": False, "error": f"Verification error: {e}"}

    def _verify_gpg_signature(self, commit_hash: str) -> Optional[Dict[str, Any]]:
        """Verify GPG signature of a commit."""
        if not self.gpg_available:
            return None

        try:
            result = subprocess.run(
                ["git", "verify-commit", commit_hash],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                return {"valid": True, "output": result.stderr}
            else:
                return {"valid": False, "error": result.stderr}

        except Exception as e:
            return {"valid": False, "error": str(e)}

    def _extract_healing_metadata(self, commit_message: str) -> Dict[str, Any]:
        """Extract healing metadata from commit message."""
        metadata = {}
        lines = commit_message.split("\n")

        for line in lines:
            if line.startswith("Healing-"):
                try:
                    key, value = line.split(":", 1)
                    metadata_key = key.replace("Healing-", "").lower().replace("-", "_")
                    metadata[metadata_key] = value.strip()
                except ValueError:
                    continue

        # Convert specific fields
        if "confidence" in metadata:
            try:
                metadata["confidence"] = float(metadata["confidence"])
            except ValueError:
                pass

        return metadata

    def create_healing_commit(
        self, file_changes: Dict[str, str], healing_metadata: Dict[str, Any]
    ) -> bool:
        """
        Create a signed healing commit with proper metadata.

        Args:
            file_changes: Dictionary of file paths to new content
            healing_metadata: Metadata about the healing operation

        Returns:
            True if commit was created successfully
        """
        try:
            # Apply file changes
            for file_path, content in file_changes.items():
                full_path = self.repo_path / file_path
                full_path.parent.mkdir(parents=True, exist_ok=True)

                with open(full_path, "w") as f:
                    f.write(content)

                # Stage the file
                subprocess.run(
                    ["git", "add", file_path], cwd=self.repo_path, check=True
                )

            # Generate commit message
            from .commit_analyzer import CommitAnalyzer

            commit_analyzer = CommitAnalyzer(str(self.repo_path))
            base_message = commit_analyzer.generate_healing_commit_message(
                healing_metadata
            )

            # Sign the commit message
            signed_message = self.sign_healing_commit(base_message, healing_metadata)

            # Create commit
            env = os.environ.copy()
            if self.signing_enabled:
                env["GIT_COMMITTER_EMAIL"] = "homeostasis@localhost"
                env["GIT_COMMITTER_NAME"] = "Homeostasis Healing Bot"

            result = subprocess.run(
                ["git", "commit", "-m", signed_message],
                cwd=self.repo_path,
                env=env,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                self.logger.error(f"Failed to create commit: {result.stderr}")
                return False

            # Log to audit trail
            commit_hash = self._get_last_commit_hash()
            self._log_to_audit_trail(commit_hash, healing_metadata)

            self.logger.info(f"Created healing commit: {commit_hash}")
            return True

        except Exception as e:
            self.logger.error(f"Error creating healing commit: {e}")
            return False

    def _get_last_commit_hash(self) -> str:
        """Get the hash of the last commit."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                return result.stdout.strip()

            return "unknown"

        except Exception:
            return "unknown"

    def _log_to_audit_trail(
        self, commit_hash: str, healing_metadata: Dict[str, Any]
    ) -> None:
        """Log healing commit to audit trail."""
        if not self.config.get("audit_trail", {}).get("enabled", True):
            return

        try:
            log_file = self.repo_path / self.config["audit_trail"]["log_file"]

            audit_entry = {
                "timestamp": datetime.now().isoformat(),
                "commit_hash": commit_hash,
                "healing_metadata": healing_metadata,
                "signed": self.signing_enabled,
            }

            with open(log_file, "a") as f:
                f.write(json.dumps(audit_entry) + "\n")

        except Exception as e:
            self.logger.error(f"Error logging to audit trail: {e}")

    def get_audit_trail(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get audit trail of healing commits.

        Args:
            limit: Maximum number of entries to return

        Returns:
            List of audit trail entries
        """
        try:
            log_file = self.repo_path / self.config["audit_trail"]["log_file"]

            if not log_file.exists():
                return []

            entries = []
            with open(log_file, "r") as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        entries.append(entry)
                    except json.JSONDecodeError:
                        continue

            # Return most recent entries first
            entries.reverse()

            if limit:
                entries = entries[:limit]

            return entries

        except Exception as e:
            self.logger.error(f"Error reading audit trail: {e}")
            return []

    def verify_audit_trail_integrity(self) -> Dict[str, Any]:
        """
        Verify the integrity of the audit trail.

        Returns:
            Integrity verification results
        """
        audit_entries = self.get_audit_trail()

        results = {
            "total_entries": len(audit_entries),
            "verified_entries": 0,
            "failed_verifications": [],
            "missing_commits": [],
            "integrity_score": 0.0,
        }

        for entry in audit_entries:
            commit_hash = entry.get("commit_hash")
            if not commit_hash:
                continue

            # Verify commit exists
            try:
                subprocess.run(
                    ["git", "rev-parse", "--verify", commit_hash],
                    cwd=self.repo_path,
                    check=True,
                    capture_output=True,
                )
            except subprocess.CalledProcessError:
                results["missing_commits"].append(commit_hash)
                continue

            # Verify commit signature
            verification = self.verify_healing_commit(commit_hash)
            if verification["verified"]:
                results["verified_entries"] += 1
            else:
                results["failed_verifications"].append(
                    {
                        "commit_hash": commit_hash,
                        "error": verification.get("error", "Unknown error"),
                    }
                )

        # Calculate integrity score
        if results["total_entries"] > 0:
            results["integrity_score"] = (
                results["verified_entries"] / results["total_entries"]
            )

        return results
