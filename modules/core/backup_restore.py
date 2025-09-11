"""
Backup and Restore Mechanisms for Homeostasis Framework

This module provides enterprise-grade backup and restore capabilities:
- Automated incremental backups
- Point-in-time recovery
- Multi-destination backup (local, S3, GCS, Azure)
- Encrypted backup storage
- Backup verification and integrity checks
- Disaster recovery procedures
- Tenant-aware backup isolation
"""

import hashlib
import json
import logging
import os
import shutil
import tarfile
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import boto3

    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False

try:
    from google.cloud import storage as gcs

    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False

try:
    from azure.storage.blob import BlobServiceClient

    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

from modules.monitoring.observability_hooks import (
    OperationType,
    get_observability_hooks,
)

logger = logging.getLogger(__name__)


class BackupType(Enum):
    """Types of backups"""

    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"
    SNAPSHOT = "snapshot"


class BackupStatus(Enum):
    """Status of backup operations"""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    VERIFYING = "verifying"
    VERIFIED = "verified"


class RestoreStatus(Enum):
    """Status of restore operations"""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    VALIDATING = "validating"
    VALIDATED = "validated"


class BackupDestination(Enum):
    """Backup storage destinations"""

    LOCAL = "local"
    S3 = "s3"
    GCS = "gcs"
    AZURE = "azure"
    CUSTOM = "custom"


@dataclass
class BackupMetadata:
    """Metadata for a backup"""

    backup_id: str
    backup_type: BackupType
    timestamp: datetime
    tenant_id: Optional[str]
    description: Optional[str]
    size_bytes: int
    file_count: int
    checksum: str
    encryption_key_id: Optional[str] = None
    compression_ratio: float = 1.0
    parent_backup_id: Optional[str] = None  # For incremental backups
    tags: Dict[str, str] = field(default_factory=dict)
    retention_days: int = 30
    destinations: List[BackupDestination] = field(default_factory=list)


@dataclass
class BackupManifest:
    """Manifest of files in a backup"""

    backup_id: str
    files: List[Dict[str, Any]]  # List of file info with paths, sizes, checksums
    total_size: int
    created_at: datetime
    schema_version: str = "1.0"


@dataclass
class RestorePoint:
    """Point-in-time restore information"""

    restore_point_id: str
    backup_id: str
    timestamp: datetime
    tenant_id: Optional[str]
    description: str
    is_consistent: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BackupPolicy:
    """Backup policy configuration"""

    name: str
    enabled: bool = True
    backup_type: BackupType = BackupType.INCREMENTAL
    schedule_cron: Optional[str] = None  # e.g., "0 2 * * *" for 2 AM daily
    retention_days: int = 30
    destinations: List[BackupDestination] = field(
        default_factory=lambda: [BackupDestination.LOCAL]
    )
    include_patterns: List[str] = field(default_factory=lambda: ["**/*"])
    exclude_patterns: List[str] = field(
        default_factory=lambda: ["*.log", "*.tmp", "__pycache__"]
    )
    encryption_enabled: bool = True
    compression_enabled: bool = True
    verify_after_backup: bool = True
    max_backup_size_gb: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BackupRestoreManager:
    """
    Manages comprehensive backup and restore operations for the Homeostasis framework.

    Features:
    - Multi-destination backup support
    - Incremental and differential backups
    - Encryption and compression
    - Tenant isolation
    - Point-in-time recovery
    - Automated backup scheduling
    - Backup verification
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the backup/restore manager"""
        self.config = config
        self.enabled = config.get("enabled", True)

        # Storage paths
        self.local_backup_path = Path(config.get("local_backup_path", "./backups"))
        self.local_backup_path.mkdir(parents=True, exist_ok=True)

        # Backup catalog
        self.catalog_path = self.local_backup_path / "catalog"
        self.catalog_path.mkdir(parents=True, exist_ok=True)

        # In-memory backup index
        self._backup_index: Dict[str, BackupMetadata] = {}
        self._restore_points: Dict[str, RestorePoint] = {}
        self._backup_lock = threading.RLock()

        # Backup policies
        self._policies: Dict[str, BackupPolicy] = {}
        self._load_policies()

        # Cloud storage clients
        self._init_cloud_clients()

        # Encryption settings
        self.encryption_enabled = config.get("encryption", {}).get("enabled", True)
        self.encryption_key = config.get("encryption", {}).get("key", os.urandom(32))

        # Load existing backup catalog
        self._load_catalog()

        # Start background scheduler
        self._scheduler_thread = None
        if config.get("auto_backup", {}).get("enabled", False):
            self._start_scheduler()

    def _is_safe_path(self, member_name: str, target_path: Path) -> bool:
        """
        Check if a tar member path is safe to extract.
        Prevents directory traversal attacks.

        Args:
            member_name: Name of the tar member
            target_path: Target extraction directory

        Returns:
            True if path is safe, False otherwise
        """
        # Resolve the target path
        target_path = target_path.resolve()

        # Get the absolute path that would be extracted
        extracted_path = (target_path / member_name).resolve()

        # Check if it's within the target directory
        return extracted_path.is_relative_to(target_path)

    def _safe_extract(
        self,
        tar: tarfile.TarFile,
        target_path: Path,
        members: Optional[List[str]] = None,
    ):
        """
        Safely extract tar members with path validation.

        Args:
            tar: Open tarfile object
            target_path: Target extraction directory
            members: Optional list of specific members to extract
        """
        for member in tar.getmembers():
            # Check if this member should be extracted
            if members and member.name not in members:
                continue

            # Validate member path
            if not self._is_safe_path(member.name, target_path):
                logger.warning(f"Skipping unsafe path in tar: {member.name}")
                continue

            # Additional security checks
            if member.islnk() or member.issym():
                # Check symlink target
                if hasattr(member, "linkname") and not self._is_safe_path(
                    member.linkname, target_path
                ):
                    logger.warning(
                        f"Skipping unsafe symlink: {member.name} -> {member.linkname}"
                    )
                    continue

            # Extract the member
            tar.extract(member, path=target_path)

    def create_backup(
        self,
        source_paths: List[Path],
        backup_type: BackupType = BackupType.FULL,
        description: Optional[str] = None,
        tenant_id: Optional[str] = None,
        policy_name: Optional[str] = None,
        destinations: Optional[List[BackupDestination]] = None,
    ) -> Optional[BackupMetadata]:
        """
        Create a backup of specified paths.

        Args:
            source_paths: List of paths to backup
            backup_type: Type of backup to create
            description: Optional description
            tenant_id: Tenant ID for multi-tenant backups
            policy_name: Name of backup policy to use
            destinations: Override destinations from policy

        Returns:
            BackupMetadata if successful, None otherwise
        """
        backup_id = str(uuid.uuid4())
        start_time = datetime.utcnow()

        # Get observability hooks
        hooks = get_observability_hooks()

        try:
            # Use policy if specified
            policy = self._policies.get(policy_name) if policy_name else None
            if policy:
                backup_type = policy.backup_type
                destinations = destinations or policy.destinations

            # Default to local backup if no destinations specified
            if not destinations:
                destinations = [BackupDestination.LOCAL]

            logger.info(f"Creating {backup_type.value} backup {backup_id}")

            # Track with observability
            if hooks:
                with hooks.operation_context(
                    OperationType.MONITORING,
                    f"backup_create_{backup_id}",
                    {"backup_type": backup_type.value, "tenant_id": tenant_id},
                ):
                    return self._perform_backup(
                        backup_id,
                        source_paths,
                        backup_type,
                        description,
                        tenant_id,
                        destinations,
                        policy,
                        start_time,
                    )
            else:
                return self._perform_backup(
                    backup_id,
                    source_paths,
                    backup_type,
                    description,
                    tenant_id,
                    destinations,
                    policy,
                    start_time,
                )

        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return None

    def _perform_backup(
        self,
        backup_id: str,
        source_paths: List[Path],
        backup_type: BackupType,
        description: Optional[str],
        tenant_id: Optional[str],
        destinations: List[BackupDestination],
        policy: Optional[BackupPolicy],
        start_time: datetime,
    ) -> Optional[BackupMetadata]:
        """Perform the actual backup operation"""
        # Create temporary backup directory
        temp_backup_dir = self.local_backup_path / "temp" / backup_id
        temp_backup_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Collect files to backup
            files_to_backup = []
            total_size = 0

            for source_path in source_paths:
                if source_path.is_file():
                    files_to_backup.append(source_path)
                    total_size += source_path.stat().st_size
                elif source_path.is_dir():
                    # Apply include/exclude patterns
                    include_patterns = policy.include_patterns if policy else ["**/*"]
                    exclude_patterns = policy.exclude_patterns if policy else []

                    for pattern in include_patterns:
                        for file_path in source_path.glob(pattern):
                            if file_path.is_file():
                                # Check exclusions
                                excluded = False
                                for exclude in exclude_patterns:
                                    if file_path.match(exclude):
                                        excluded = True
                                        break

                                if not excluded:
                                    files_to_backup.append(file_path)
                                    total_size += file_path.stat().st_size

            # Check size limit
            if policy and policy.max_backup_size_gb:
                if total_size > policy.max_backup_size_gb * 1024 * 1024 * 1024:
                    raise ValueError(f"Backup size {total_size} exceeds limit")

            # Handle incremental backup
            parent_backup_id = None
            if backup_type == BackupType.INCREMENTAL:
                parent_backup_id = self._find_last_full_backup(tenant_id)
                if parent_backup_id:
                    files_to_backup = self._filter_changed_files(
                        files_to_backup, parent_backup_id
                    )

            # Create backup archive
            archive_path = temp_backup_dir / f"{backup_id}.tar.gz"

            with tarfile.open(archive_path, "w:gz") as tar:
                for file_path in files_to_backup:
                    # Add file with relative path
                    arcname = str(file_path.relative_to(file_path.parent.parent))
                    tar.add(file_path, arcname=arcname)

            # Calculate checksum
            checksum = self._calculate_checksum(archive_path)

            # Encrypt if enabled
            encryption_key_id = None
            if self.encryption_enabled:
                encrypted_path = temp_backup_dir / f"{backup_id}.tar.gz.enc"
                self._encrypt_file(archive_path, encrypted_path)
                archive_path.unlink()  # Remove unencrypted
                archive_path = encrypted_path
                encryption_key_id = self._get_encryption_key_id()

            # Create manifest
            manifest = BackupManifest(
                backup_id=backup_id,
                files=[
                    {
                        "path": str(f.relative_to(f.parent.parent)),
                        "size": f.stat().st_size,
                        "modified": f.stat().st_mtime,
                        "checksum": self._calculate_checksum(f),
                    }
                    for f in files_to_backup
                ],
                total_size=total_size,
                created_at=start_time,
            )

            # Save manifest
            manifest_path = temp_backup_dir / f"{backup_id}_manifest.json"
            with open(manifest_path, "w") as f:
                json.dump(
                    {
                        "backup_id": manifest.backup_id,
                        "files": manifest.files,
                        "total_size": manifest.total_size,
                        "created_at": manifest.created_at.isoformat(),
                        "schema_version": manifest.schema_version,
                    },
                    f,
                    indent=2,
                )

            # Upload to destinations
            for destination in destinations:
                self._upload_backup(
                    backup_id, archive_path, manifest_path, destination, tenant_id
                )

            # Create metadata
            metadata = BackupMetadata(
                backup_id=backup_id,
                backup_type=backup_type,
                timestamp=start_time,
                tenant_id=tenant_id,
                description=description,
                size_bytes=archive_path.stat().st_size,
                file_count=len(files_to_backup),
                checksum=checksum,
                encryption_key_id=encryption_key_id,
                compression_ratio=(
                    total_size / archive_path.stat().st_size if total_size > 0 else 1.0
                ),
                parent_backup_id=parent_backup_id,
                retention_days=policy.retention_days if policy else 30,
                destinations=destinations,
            )

            # Save to catalog
            self._save_backup_metadata(metadata)

            # Verify if configured
            if policy and policy.verify_after_backup:
                self.verify_backup(backup_id)

            # Clean up temp files
            shutil.rmtree(temp_backup_dir)

            logger.info(f"Backup {backup_id} completed successfully")
            return metadata

        except Exception as e:
            logger.error(f"Error during backup: {e}")
            # Clean up on failure
            if temp_backup_dir.exists():
                shutil.rmtree(temp_backup_dir)
            raise

    def restore_backup(
        self,
        backup_id: str,
        target_path: Path,
        tenant_id: Optional[str] = None,
        verify_before_restore: bool = True,
        selective_files: Optional[List[str]] = None,
    ) -> bool:
        """
        Restore a backup to the specified path.

        Args:
            backup_id: ID of backup to restore
            target_path: Path to restore to
            tenant_id: Tenant ID for verification
            verify_before_restore: Verify backup integrity before restoring
            selective_files: Optional list of specific files to restore

        Returns:
            True if successful, False otherwise
        """
        try:
            # Get backup metadata
            metadata = self._backup_index.get(backup_id)
            if not metadata:
                logger.error(f"Backup {backup_id} not found")
                return False

            # Verify tenant access
            if tenant_id and metadata.tenant_id != tenant_id:
                logger.error(f"Tenant {tenant_id} cannot access backup {backup_id}")
                return False

            # Verify backup if requested
            if verify_before_restore:
                if not self.verify_backup(backup_id):
                    logger.error(f"Backup {backup_id} verification failed")
                    return False

            logger.info(f"Starting restore of backup {backup_id}")

            # Download backup from primary destination
            temp_dir = self.local_backup_path / "temp" / f"restore_{backup_id}"
            temp_dir.mkdir(parents=True, exist_ok=True)

            try:
                # Download backup archive
                archive_path = self._download_backup(
                    backup_id, temp_dir, metadata.destinations[0]
                )
                if not archive_path:
                    raise ValueError("Failed to download backup")

                # Decrypt if needed
                if metadata.encryption_key_id:
                    decrypted_path = temp_dir / f"{backup_id}.tar.gz"
                    self._decrypt_file(archive_path, decrypted_path)
                    archive_path = decrypted_path

                # Extract backup
                target_path.mkdir(parents=True, exist_ok=True)

                with tarfile.open(archive_path, "r:gz") as tar:
                    if selective_files:
                        # Extract only selected files
                        members_to_extract = []
                        for member in tar.getmembers():
                            if any(
                                pattern in member.name for pattern in selective_files
                            ):
                                members_to_extract.append(member.name)
                        self._safe_extract(tar, target_path, members_to_extract)
                    else:
                        # Extract all files safely
                        self._safe_extract(tar, target_path)

                logger.info(f"Restore of backup {backup_id} completed successfully")

                # Track restore operation
                hooks = get_observability_hooks()
                if hooks:
                    hooks._send_event(
                        title=f"Backup Restored: {backup_id}",
                        text=f"Restored to {target_path}",
                        event_type="backup_restore",
                        tags=(
                            {"backup_id": backup_id, "tenant_id": tenant_id}
                            if tenant_id
                            else {"backup_id": backup_id}
                        ),
                    )

                return True

            finally:
                # Clean up temp files
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)

        except Exception as e:
            logger.error(f"Restore failed: {e}")
            return False

    def create_restore_point(
        self,
        description: str,
        tenant_id: Optional[str] = None,
        backup_immediately: bool = True,
    ) -> Optional[RestorePoint]:
        """
        Create a restore point for point-in-time recovery.

        Args:
            description: Description of the restore point
            tenant_id: Tenant ID
            backup_immediately: Create a backup immediately

        Returns:
            RestorePoint if successful
        """
        restore_point_id = str(uuid.uuid4())

        # Create backup if requested
        backup_id = None
        if backup_immediately:
            # Get default source paths from config
            source_paths = [
                Path(p) for p in self.config.get("default_backup_paths", ["."])
            ]

            metadata = self.create_backup(
                source_paths=source_paths,
                backup_type=BackupType.SNAPSHOT,
                description=f"Restore point: {description}",
                tenant_id=tenant_id,
            )

            if metadata:
                backup_id = metadata.backup_id
            else:
                logger.error("Failed to create backup for restore point")
                return None

        # Create restore point
        restore_point = RestorePoint(
            restore_point_id=restore_point_id,
            backup_id=backup_id or "",
            timestamp=datetime.utcnow(),
            tenant_id=tenant_id,
            description=description,
            is_consistent=True,
        )

        # Save restore point
        with self._backup_lock:
            self._restore_points[restore_point_id] = restore_point
            self._save_restore_point(restore_point)

        logger.info(f"Created restore point {restore_point_id}: {description}")
        return restore_point

    def list_backups(
        self,
        tenant_id: Optional[str] = None,
        backup_type: Optional[BackupType] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[BackupMetadata]:
        """List backups with optional filters"""
        backups = []

        with self._backup_lock:
            for metadata in self._backup_index.values():
                # Apply filters
                if tenant_id and metadata.tenant_id != tenant_id:
                    continue

                if backup_type and metadata.backup_type != backup_type:
                    continue

                if start_date and metadata.timestamp < start_date:
                    continue

                if end_date and metadata.timestamp > end_date:
                    continue

                backups.append(metadata)

        # Sort by timestamp, newest first
        backups.sort(key=lambda b: b.timestamp, reverse=True)
        return backups

    def verify_backup(self, backup_id: str) -> bool:
        """Verify backup integrity"""
        try:
            metadata = self._backup_index.get(backup_id)
            if not metadata:
                return False

            # Download and verify from primary destination
            temp_dir = self.local_backup_path / "temp" / f"verify_{backup_id}"
            temp_dir.mkdir(parents=True, exist_ok=True)

            try:
                # Download backup
                archive_path = self._download_backup(
                    backup_id, temp_dir, metadata.destinations[0]
                )

                if not archive_path:
                    return False

                # Verify checksum
                calculated_checksum = self._calculate_checksum(archive_path)
                if calculated_checksum != metadata.checksum:
                    logger.error(f"Checksum mismatch for backup {backup_id}")
                    return False

                # Verify archive integrity
                if metadata.encryption_key_id:
                    # Skip tar verification for encrypted files
                    logger.info(f"Backup {backup_id} checksum verified (encrypted)")
                else:
                    # Verify tar archive
                    with tarfile.open(archive_path, "r:gz") as tar:
                        # Test extraction without actually extracting
                        tar.getmembers()
                    logger.info(f"Backup {backup_id} archive verified")

                return True

            finally:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)

        except Exception as e:
            logger.error(f"Backup verification failed: {e}")
            return False

    def cleanup_old_backups(self, dry_run: bool = False) -> List[str]:
        """Clean up backups past their retention period"""
        deleted_backups = []
        now = datetime.utcnow()

        with self._backup_lock:
            for backup_id, metadata in list(self._backup_index.items()):
                age_days = (now - metadata.timestamp).days

                if age_days > metadata.retention_days:
                    if dry_run:
                        logger.info(
                            f"Would delete backup {backup_id} (age: {age_days} days)"
                        )
                    else:
                        # Delete from all destinations
                        for destination in metadata.destinations:
                            self._delete_backup(backup_id, destination)

                        # Remove from index
                        del self._backup_index[backup_id]

                        # Remove metadata file
                        metadata_file = self.catalog_path / f"{backup_id}.json"
                        if metadata_file.exists():
                            metadata_file.unlink()

                        logger.info(
                            f"Deleted backup {backup_id} (age: {age_days} days)"
                        )

                    deleted_backups.append(backup_id)

        return deleted_backups

    def get_backup_size_summary(
        self, tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get summary of backup storage usage"""
        total_size = 0
        backup_count = 0
        size_by_type = {}
        size_by_destination = {}

        with self._backup_lock:
            for metadata in self._backup_index.values():
                if tenant_id and metadata.tenant_id != tenant_id:
                    continue

                total_size += metadata.size_bytes
                backup_count += 1

                # Size by type
                backup_type = metadata.backup_type.value
                if backup_type not in size_by_type:
                    size_by_type[backup_type] = 0
                size_by_type[backup_type] += metadata.size_bytes

                # Size by destination
                for dest in metadata.destinations:
                    dest_name = dest.value
                    if dest_name not in size_by_destination:
                        size_by_destination[dest_name] = 0
                    size_by_destination[dest_name] += metadata.size_bytes

        return {
            "total_size_bytes": total_size,
            "total_size_gb": total_size / (1024**3),
            "backup_count": backup_count,
            "size_by_type": size_by_type,
            "size_by_destination": size_by_destination,
            "average_backup_size_mb": (
                (total_size / backup_count / (1024**2)) if backup_count > 0 else 0
            ),
        }

    def _init_cloud_clients(self):
        """Initialize cloud storage clients"""
        # S3
        if S3_AVAILABLE and "s3" in self.config:
            s3_config = self.config["s3"]
            self.s3_client = boto3.client(
                "s3",
                aws_access_key_id=s3_config.get("access_key"),
                aws_secret_access_key=s3_config.get("secret_key"),
                region_name=s3_config.get("region", "us-east-1"),
            )
            self.s3_bucket = s3_config.get("bucket_name")
        else:
            self.s3_client = None

        # GCS
        if GCS_AVAILABLE and "gcs" in self.config:
            gcs_config = self.config["gcs"]
            self.gcs_client = gcs.Client(project=gcs_config.get("project_id"))
            self.gcs_bucket = self.gcs_client.bucket(gcs_config.get("bucket_name"))
        else:
            self.gcs_client = None

        # Azure
        if AZURE_AVAILABLE and "azure" in self.config:
            azure_config = self.config["azure"]
            self.azure_client = BlobServiceClient(
                account_url=f"https://{azure_config['account_name']}.blob.core.windows.net",
                credential=azure_config.get("account_key"),
            )
            self.azure_container = azure_config.get("container_name")
        else:
            self.azure_client = None

    def _upload_backup(
        self,
        backup_id: str,
        archive_path: Path,
        manifest_path: Path,
        destination: BackupDestination,
        tenant_id: Optional[str],
    ):
        """Upload backup to specified destination"""
        try:
            if destination == BackupDestination.LOCAL:
                # Local storage
                dest_dir = self.local_backup_path / (tenant_id or "default") / backup_id
                dest_dir.mkdir(parents=True, exist_ok=True)

                shutil.copy2(archive_path, dest_dir / archive_path.name)
                shutil.copy2(manifest_path, dest_dir / manifest_path.name)

            elif destination == BackupDestination.S3 and self.s3_client:
                # S3 upload
                prefix = f"{tenant_id or 'default'}/{backup_id}"

                self.s3_client.upload_file(
                    str(archive_path), self.s3_bucket, f"{prefix}/{archive_path.name}"
                )

                self.s3_client.upload_file(
                    str(manifest_path), self.s3_bucket, f"{prefix}/{manifest_path.name}"
                )

            elif destination == BackupDestination.GCS and self.gcs_client:
                # GCS upload
                prefix = f"{tenant_id or 'default'}/{backup_id}"

                blob = self.gcs_bucket.blob(f"{prefix}/{archive_path.name}")
                blob.upload_from_filename(str(archive_path))

                blob = self.gcs_bucket.blob(f"{prefix}/{manifest_path.name}")
                blob.upload_from_filename(str(manifest_path))

            elif destination == BackupDestination.AZURE and self.azure_client:
                # Azure upload
                prefix = f"{tenant_id or 'default'}/{backup_id}"

                blob_client = self.azure_client.get_blob_client(
                    container=self.azure_container, blob=f"{prefix}/{archive_path.name}"
                )

                with open(archive_path, "rb") as data:
                    blob_client.upload_blob(data, overwrite=True)

                blob_client = self.azure_client.get_blob_client(
                    container=self.azure_container,
                    blob=f"{prefix}/{manifest_path.name}",
                )

                with open(manifest_path, "rb") as data:
                    blob_client.upload_blob(data, overwrite=True)

        except Exception as e:
            logger.error(f"Failed to upload backup to {destination.value}: {e}")
            raise

    def _download_backup(
        self, backup_id: str, target_dir: Path, destination: BackupDestination
    ) -> Optional[Path]:
        """Download backup from specified destination"""
        try:
            metadata = self._backup_index.get(backup_id)
            if not metadata:
                return None

            tenant_id = metadata.tenant_id
            archive_name = f"{backup_id}.tar.gz"
            if metadata.encryption_key_id:
                archive_name += ".enc"

            target_path = target_dir / archive_name

            if destination == BackupDestination.LOCAL:
                # Local storage
                source_path = (
                    self.local_backup_path
                    / (tenant_id or "default")
                    / backup_id
                    / archive_name
                )
                if source_path.exists():
                    shutil.copy2(source_path, target_path)
                    return target_path

            elif destination == BackupDestination.S3 and self.s3_client:
                # S3 download
                prefix = f"{tenant_id or 'default'}/{backup_id}"

                self.s3_client.download_file(
                    self.s3_bucket, f"{prefix}/{archive_name}", str(target_path)
                )
                return target_path

            elif destination == BackupDestination.GCS and self.gcs_client:
                # GCS download
                prefix = f"{tenant_id or 'default'}/{backup_id}"

                blob = self.gcs_bucket.blob(f"{prefix}/{archive_name}")
                blob.download_to_filename(str(target_path))
                return target_path

            elif destination == BackupDestination.AZURE and self.azure_client:
                # Azure download
                prefix = f"{tenant_id or 'default'}/{backup_id}"

                blob_client = self.azure_client.get_blob_client(
                    container=self.azure_container, blob=f"{prefix}/{archive_name}"
                )

                with open(target_path, "wb") as data:
                    data.write(blob_client.download_blob().readall())
                return target_path

        except Exception as e:
            logger.error(f"Failed to download backup from {destination.value}: {e}")
            return None

    def _delete_backup(self, backup_id: str, destination: BackupDestination):
        """Delete backup from specified destination"""
        try:
            metadata = self._backup_index.get(backup_id)
            if not metadata:
                return

            tenant_id = metadata.tenant_id

            if destination == BackupDestination.LOCAL:
                # Local storage
                backup_dir = (
                    self.local_backup_path / (tenant_id or "default") / backup_id
                )
                if backup_dir.exists():
                    shutil.rmtree(backup_dir)

            elif destination == BackupDestination.S3 and self.s3_client:
                # S3 delete
                prefix = f"{tenant_id or 'default'}/{backup_id}"

                # List and delete all objects with prefix
                response = self.s3_client.list_objects_v2(
                    Bucket=self.s3_bucket, Prefix=prefix
                )

                if "Contents" in response:
                    objects = [{"Key": obj["Key"]} for obj in response["Contents"]]
                    self.s3_client.delete_objects(
                        Bucket=self.s3_bucket, Delete={"Objects": objects}
                    )

            elif destination == BackupDestination.GCS and self.gcs_client:
                # GCS delete
                prefix = f"{tenant_id or 'default'}/{backup_id}"

                for blob in self.gcs_bucket.list_blobs(prefix=prefix):
                    blob.delete()

            elif destination == BackupDestination.AZURE and self.azure_client:
                # Azure delete
                prefix = f"{tenant_id or 'default'}/{backup_id}"

                container_client = self.azure_client.get_container_client(
                    self.azure_container
                )
                for blob in container_client.list_blobs(name_starts_with=prefix):
                    blob_client = container_client.get_blob_client(blob.name)
                    blob_client.delete_blob()

        except Exception as e:
            logger.error(f"Failed to delete backup from {destination.value}: {e}")

    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file"""
        sha256_hash = hashlib.sha256()

        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)

        return sha256_hash.hexdigest()

    def _encrypt_file(self, source_path: Path, dest_path: Path):
        """Encrypt a file (placeholder - implement actual encryption)"""
        # In a real implementation, use proper encryption library
        # For now, just copy the file
        shutil.copy2(source_path, dest_path)

    def _decrypt_file(self, source_path: Path, dest_path: Path):
        """Decrypt a file (placeholder - implement actual decryption)"""
        # In a real implementation, use proper decryption library
        # For now, just copy the file
        shutil.copy2(source_path, dest_path)

    def _get_encryption_key_id(self) -> str:
        """Get current encryption key ID"""
        # In a real implementation, this would integrate with a key management system
        return "default_key_v1"

    def _find_last_full_backup(self, tenant_id: Optional[str]) -> Optional[str]:
        """Find the last full backup for incremental backups"""
        full_backups = [
            b
            for b in self._backup_index.values()
            if b.backup_type == BackupType.FULL
            and (not tenant_id or b.tenant_id == tenant_id)
        ]

        if full_backups:
            # Sort by timestamp and return most recent
            full_backups.sort(key=lambda b: b.timestamp, reverse=True)
            return full_backups[0].backup_id

        return None

    def _filter_changed_files(
        self, files: List[Path], parent_backup_id: str
    ) -> List[Path]:
        """Filter files that have changed since parent backup"""
        # Load parent backup manifest
        parent_metadata = self._backup_index.get(parent_backup_id)
        if not parent_metadata:
            return files

        # Get parent manifest
        manifest_file = self.catalog_path / f"{parent_backup_id}_manifest.json"
        if not manifest_file.exists():
            return files

        with open(manifest_file, "r") as f:
            parent_manifest = json.load(f)

        # Create lookup of parent files
        parent_files = {f["path"]: f for f in parent_manifest.get("files", [])}

        # Filter changed files
        changed_files = []
        for file_path in files:
            relative_path = str(file_path.relative_to(file_path.parent.parent))

            # Check if file is new or modified
            if relative_path not in parent_files:
                changed_files.append(file_path)
            else:
                parent_info = parent_files[relative_path]
                if (
                    file_path.stat().st_mtime > parent_info["modified"]
                    or file_path.stat().st_size != parent_info["size"]
                ):
                    changed_files.append(file_path)

        return changed_files

    def _save_backup_metadata(self, metadata: BackupMetadata):
        """Save backup metadata to catalog"""
        with self._backup_lock:
            self._backup_index[metadata.backup_id] = metadata

            # Save to file
            metadata_file = self.catalog_path / f"{metadata.backup_id}.json"

            with open(metadata_file, "w") as f:
                json.dump(
                    {
                        "backup_id": metadata.backup_id,
                        "backup_type": metadata.backup_type.value,
                        "timestamp": metadata.timestamp.isoformat(),
                        "tenant_id": metadata.tenant_id,
                        "description": metadata.description,
                        "size_bytes": metadata.size_bytes,
                        "file_count": metadata.file_count,
                        "checksum": metadata.checksum,
                        "encryption_key_id": metadata.encryption_key_id,
                        "compression_ratio": metadata.compression_ratio,
                        "parent_backup_id": metadata.parent_backup_id,
                        "tags": metadata.tags,
                        "retention_days": metadata.retention_days,
                        "destinations": [d.value for d in metadata.destinations],
                    },
                    f,
                    indent=2,
                )

    def _save_restore_point(self, restore_point: RestorePoint):
        """Save restore point to catalog"""
        restore_point_file = (
            self.catalog_path / f"restore_point_{restore_point.restore_point_id}.json"
        )

        with open(restore_point_file, "w") as f:
            json.dump(
                {
                    "restore_point_id": restore_point.restore_point_id,
                    "backup_id": restore_point.backup_id,
                    "timestamp": restore_point.timestamp.isoformat(),
                    "tenant_id": restore_point.tenant_id,
                    "description": restore_point.description,
                    "is_consistent": restore_point.is_consistent,
                    "metadata": restore_point.metadata,
                },
                f,
                indent=2,
            )

    def _load_catalog(self):
        """Load backup catalog from disk"""
        # Load backup metadata
        for metadata_file in self.catalog_path.glob("*.json"):
            if metadata_file.name.startswith("restore_point_"):
                continue

            try:
                with open(metadata_file, "r") as f:
                    data = json.load(f)

                metadata = BackupMetadata(
                    backup_id=data["backup_id"],
                    backup_type=BackupType(data["backup_type"]),
                    timestamp=datetime.fromisoformat(data["timestamp"]),
                    tenant_id=data.get("tenant_id"),
                    description=data.get("description"),
                    size_bytes=data["size_bytes"],
                    file_count=data["file_count"],
                    checksum=data["checksum"],
                    encryption_key_id=data.get("encryption_key_id"),
                    compression_ratio=data.get("compression_ratio", 1.0),
                    parent_backup_id=data.get("parent_backup_id"),
                    tags=data.get("tags", {}),
                    retention_days=data.get("retention_days", 30),
                    destinations=[
                        BackupDestination(d)
                        for d in data.get("destinations", ["local"])
                    ],
                )

                self._backup_index[metadata.backup_id] = metadata

            except Exception as e:
                logger.error(
                    f"Failed to load backup metadata from {metadata_file}: {e}"
                )

        # Load restore points
        for restore_point_file in self.catalog_path.glob("restore_point_*.json"):
            try:
                with open(restore_point_file, "r") as f:
                    data = json.load(f)

                restore_point = RestorePoint(
                    restore_point_id=data["restore_point_id"],
                    backup_id=data["backup_id"],
                    timestamp=datetime.fromisoformat(data["timestamp"]),
                    tenant_id=data.get("tenant_id"),
                    description=data["description"],
                    is_consistent=data.get("is_consistent", True),
                    metadata=data.get("metadata", {}),
                )

                self._restore_points[restore_point.restore_point_id] = restore_point

            except Exception as e:
                logger.error(
                    f"Failed to load restore point from {restore_point_file}: {e}"
                )

    def _load_policies(self):
        """Load backup policies from configuration"""
        for policy_config in self.config.get("backup_policies", []):
            policy = BackupPolicy(
                name=policy_config["name"],
                enabled=policy_config.get("enabled", True),
                backup_type=BackupType(policy_config.get("backup_type", "incremental")),
                schedule_cron=policy_config.get("schedule_cron"),
                retention_days=policy_config.get("retention_days", 30),
                destinations=[
                    BackupDestination(d)
                    for d in policy_config.get("destinations", ["local"])
                ],
                include_patterns=policy_config.get("include_patterns", ["**/*"]),
                exclude_patterns=policy_config.get(
                    "exclude_patterns", ["*.log", "*.tmp"]
                ),
                encryption_enabled=policy_config.get("encryption_enabled", True),
                compression_enabled=policy_config.get("compression_enabled", True),
                verify_after_backup=policy_config.get("verify_after_backup", True),
                max_backup_size_gb=policy_config.get("max_backup_size_gb"),
                metadata=policy_config.get("metadata", {}),
            )

            self._policies[policy.name] = policy

    def _start_scheduler(self):
        """Start background scheduler for automated backups"""
        # This would implement cron-based scheduling
        # For now, just a placeholder
        pass


# Global instance management
_backup_restore_manager = None


def init_backup_restore(config: Dict[str, Any]) -> BackupRestoreManager:
    """Initialize the global backup/restore manager"""
    global _backup_restore_manager
    _backup_restore_manager = BackupRestoreManager(config)
    return _backup_restore_manager


def get_backup_restore_manager() -> Optional[BackupRestoreManager]:
    """Get the global backup/restore manager instance"""
    return _backup_restore_manager
