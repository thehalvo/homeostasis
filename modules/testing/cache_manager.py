"""
Test environment caching for faster validation.

This module provides utilities for:
1. Caching test environments to reduce startup time
2. Managing container snapshots
3. Storing and retrieving test environment state
"""
import os
import time
import json
import shutil
import hashlib
import threading
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

# Add project root to sys.path
project_root = Path(__file__).parent.parent.parent

from modules.monitoring.logger import MonitoringLogger


class CacheManager:
    """
    Manages caching of test environments.
    """
    
    def __init__(self, cache_dir: Optional[Path] = None, log_level: str = "INFO", max_cache_age: int = 24):
        """
        Initialize the cache manager.
        
        Args:
            cache_dir: Directory to store cache files, default is project_root/cache
            log_level: Logging level
            max_cache_age: Maximum age of cache entries in hours
        """
        self.logger = MonitoringLogger("cache_manager", log_level=log_level)
        self.cache_dir = cache_dir or (project_root / "modules" / "analysis" / "cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cache_age = max_cache_age
        self.cache_lock = threading.RLock()
        
        # Load the cache index
        self.cache_index_file = self.cache_dir / "cache_index.json"
        self.cache_index = self._load_cache_index()
        
        # Clean up expired cache entries
        self._cleanup_expired()
        
        self.logger.info(f"Initialized cache manager with {len(self.cache_index)} cache entries")
    
    def _load_cache_index(self) -> Dict[str, Any]:
        """
        Load the cache index from disk.
        
        Returns:
            Cache index
        """
        if not self.cache_index_file.exists():
            return {}
            
        try:
            with open(self.cache_index_file, "r") as f:
                return json.load(f)
        except Exception as e:
            self.logger.exception(e, message="Failed to load cache index")
            return {}
    
    def _save_cache_index(self) -> None:
        """Save the cache index to disk."""
        try:
            with open(self.cache_index_file, "w") as f:
                json.dump(self.cache_index, f, indent=2)
        except Exception as e:
            self.logger.exception(e, message="Failed to save cache index")
    
    def _cleanup_expired(self) -> None:
        """Clean up expired cache entries."""
        with self.cache_lock:
            now = datetime.now()
            expired_keys = []
            
            for key, entry in self.cache_index.items():
                timestamp = datetime.fromisoformat(entry["timestamp"])
                if now - timestamp > timedelta(hours=self.max_cache_age):
                    expired_keys.append(key)
            
            for key in expired_keys:
                self._remove_cache_entry(key)
                
            if expired_keys:
                self.logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def _remove_cache_entry(self, key: str) -> None:
        """
        Remove a cache entry.
        
        Args:
            key: Cache key
        """
        entry = self.cache_index.get(key)
        if not entry:
            return
            
        # Remove the cache directory
        cache_path = self.cache_dir / key
        if cache_path.exists():
            try:
                shutil.rmtree(cache_path)
            except Exception as e:
                self.logger.exception(e, message=f"Failed to remove cache directory: {cache_path}")
        
        # Remove from the index
        del self.cache_index[key]
        self._save_cache_index()
        
        self.logger.debug(f"Removed cache entry: {key}")
    
    def _generate_cache_key(self, **kwargs) -> str:
        """
        Generate a cache key from kwargs.
        
        Args:
            **kwargs: Key-value pairs to include in the cache key
            
        Returns:
            Cache key
        """
        # Create a sorted string representation of kwargs
        # Sort to ensure consistent keys for the same input
        items = sorted(kwargs.items(), key=lambda x: x[0])
        
        # Create a string representation
        key_str = "&".join(f"{k}={v}" for k, v in items)
        
        # Hash the string
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _docker_commit(self, container_id: str, repository: str, tag: str) -> Optional[str]:
        """
        Create a Docker image from a container.
        
        Args:
            container_id: Docker container ID
            repository: Repository for the image
            tag: Tag for the image
            
        Returns:
            Image ID or None if failed
        """
        try:
            # Commit the container to a new image
            result = subprocess.run(
                ["docker", "commit", container_id, f"{repository}:{tag}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            
            image_id = result.stdout.strip()
            self.logger.info(f"Created image {repository}:{tag} from container {container_id}")
            return image_id
            
        except subprocess.SubprocessError as e:
            self.logger.error(f"Failed to commit container: {str(e)}", stderr=e.stderr if hasattr(e, 'stderr') else None)
            return None
    
    def _docker_load(self, repository: str, tag: str) -> Optional[str]:
        """
        Load a Docker image and create a container.
        
        Args:
            repository: Repository of the image
            tag: Tag of the image
            
        Returns:
            Container ID or None if failed
        """
        try:
            # Create a container from the image
            result = subprocess.run(
                ["docker", "create", f"{repository}:{tag}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            
            container_id = result.stdout.strip()
            self.logger.info(f"Created container {container_id} from image {repository}:{tag}")
            return container_id
            
        except subprocess.SubprocessError as e:
            self.logger.error(f"Failed to load image: {str(e)}", stderr=e.stderr if hasattr(e, 'stderr') else None)
            return None
    
    def cache_container(self, 
                       container_id: str, 
                       test_type: str, 
                       patch_id: Optional[str] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Cache a Docker container for later reuse.
        
        Args:
            container_id: Docker container ID
            test_type: Type of test environment
            patch_id: Optional patch ID associated with this container
            metadata: Optional metadata to store with the cache entry
            
        Returns:
            Cache key
        """
        # Check if Docker is available
        try:
            subprocess.run(["docker", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        except (subprocess.SubprocessError, FileNotFoundError):
            self.logger.warning("Docker not available, skipping container caching")
            return ""
            
        # Generate a cache key
        cache_key = self._generate_cache_key(
            test_type=test_type,
            patch_id=patch_id or "none",
            container_id=container_id,
            timestamp=datetime.now().isoformat()
        )
        
        with self.cache_lock:
            # Create the cache directory
            cache_path = self.cache_dir / cache_key
            cache_path.mkdir(exist_ok=True)
            
            # Commit the container to a new image
            repository = "homeostasis-cache"
            tag = cache_key
            
            image_id = self._docker_commit(container_id, repository, tag)
            if not image_id:
                return ""
                
            # Store metadata
            entry = {
                "container_id": container_id,
                "image_id": image_id,
                "repository": repository,
                "tag": tag,
                "test_type": test_type,
                "patch_id": patch_id,
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata or {}
            }
            
            # Add to the cache index
            self.cache_index[cache_key] = entry
            self._save_cache_index()
            
            self.logger.info(f"Cached container {container_id} with key {cache_key}")
            
            return cache_key
    
    def load_cached_container(self, 
                            test_type: str, 
                            patch_id: Optional[str] = None) -> Optional[str]:
        """
        Load a cached container for a specific test type and patch.
        
        Args:
            test_type: Type of test environment
            patch_id: Optional patch ID
            
        Returns:
            Container ID or None if no suitable cache entry found
        """
        # Check if Docker is available
        try:
            subprocess.run(["docker", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        except (subprocess.SubprocessError, FileNotFoundError):
            self.logger.warning("Docker not available, skipping container loading")
            return None
            
        with self.cache_lock:
            # Find the newest matching cache entry
            matching_entries = []
            
            for key, entry in self.cache_index.items():
                if entry["test_type"] == test_type:
                    if patch_id is None or entry.get("patch_id") == patch_id:
                        matching_entries.append((key, entry))
            
            if not matching_entries:
                self.logger.info(f"No cached container found for test_type={test_type}, patch_id={patch_id}")
                return None
                
            # Sort by timestamp (newest first)
            matching_entries.sort(key=lambda x: x[1]["timestamp"], reverse=True)
            
            # Use the newest entry
            key, entry = matching_entries[0]
            
            # Load the image and create a container
            container_id = self._docker_load(entry["repository"], entry["tag"])
            
            if container_id:
                self.logger.info(f"Loaded cached container {container_id} from key {key}")
                return container_id
            
            # If loading failed, remove the entry
            self._remove_cache_entry(key)
            return None
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the cache.
        
        Returns:
            Cache statistics
        """
        with self.cache_lock:
            test_types = {}
            total_entries = len(self.cache_index)
            
            for key, entry in self.cache_index.items():
                test_type = entry["test_type"]
                if test_type not in test_types:
                    test_types[test_type] = 0
                test_types[test_type] += 1
            
            return {
                "total_entries": total_entries,
                "test_types": test_types,
                "cache_dir": str(self.cache_dir),
                "max_cache_age": self.max_cache_age
            }
    
    def clear_cache(self) -> None:
        """Clear all cache entries."""
        with self.cache_lock:
            keys = list(self.cache_index.keys())
            for key in keys:
                self._remove_cache_entry(key)
                
            self.logger.info(f"Cleared {len(keys)} cache entries")


class TestEnvironmentCache:
    """
    Higher-level cache for test environments, including file snapshots
    """
    
    def __init__(self, cache_dir: Optional[Path] = None, log_level: str = "INFO"):
        """
        Initialize the test environment cache.
        
        Args:
            cache_dir: Directory to store cache files
            log_level: Logging level
        """
        self.logger = MonitoringLogger("test_env_cache", log_level=log_level)
        self.cache_dir = cache_dir or (project_root / "modules" / "testing" / "env_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.container_cache = CacheManager(self.cache_dir / "containers", log_level=log_level)
        
        # Directory for file snapshots
        self.snapshot_dir = self.cache_dir / "snapshots"
        self.snapshot_dir.mkdir(exist_ok=True)
        
        self.logger.info("Initialized test environment cache")
    
    def _create_file_snapshot(self, snapshot_id: str, files: List[Path]) -> bool:
        """
        Create a snapshot of files.
        
        Args:
            snapshot_id: ID for the snapshot
            files: List of files to include in the snapshot
            
        Returns:
            True if successful, False otherwise
        """
        # Create a directory for this snapshot
        snapshot_path = self.snapshot_dir / snapshot_id
        snapshot_path.mkdir(exist_ok=True)
        
        try:
            # Copy files to the snapshot directory
            for file_path in files:
                if not file_path.exists():
                    self.logger.warning(f"File not found for snapshot: {file_path}")
                    continue
                    
                # Create parent directories
                rel_path = file_path.relative_to(project_root)
                dest_path = snapshot_path / rel_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy the file
                shutil.copy2(file_path, dest_path)
                
            self.logger.info(f"Created snapshot {snapshot_id} with {len(files)} files")
            return True
            
        except Exception as e:
            self.logger.exception(e, message=f"Failed to create snapshot {snapshot_id}")
            return False
    
    def _restore_file_snapshot(self, snapshot_id: str) -> bool:
        """
        Restore files from a snapshot.
        
        Args:
            snapshot_id: ID of the snapshot
            
        Returns:
            True if successful, False otherwise
        """
        snapshot_path = self.snapshot_dir / snapshot_id
        if not snapshot_path.exists():
            self.logger.warning(f"Snapshot not found: {snapshot_id}")
            return False
            
        try:
            # Walk through the snapshot directory
            for root, _, files in os.walk(snapshot_path):
                for file in files:
                    # Get the source path
                    src_path = Path(root) / file
                    
                    # Get the relative path
                    rel_path = src_path.relative_to(snapshot_path)
                    
                    # Get the destination path
                    dest_path = project_root / rel_path
                    
                    # Create parent directories
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Copy the file
                    shutil.copy2(src_path, dest_path)
            
            self.logger.info(f"Restored snapshot {snapshot_id}")
            return True
            
        except Exception as e:
            self.logger.exception(e, message=f"Failed to restore snapshot {snapshot_id}")
            return False
    
    def _delete_snapshot(self, snapshot_id: str) -> bool:
        """
        Delete a snapshot.
        
        Args:
            snapshot_id: ID of the snapshot
            
        Returns:
            True if successful, False otherwise
        """
        snapshot_path = self.snapshot_dir / snapshot_id
        if not snapshot_path.exists():
            return True
            
        try:
            shutil.rmtree(snapshot_path)
            self.logger.info(f"Deleted snapshot {snapshot_id}")
            return True
            
        except Exception as e:
            self.logger.exception(e, message=f"Failed to delete snapshot {snapshot_id}")
            return False
    
    def create_environment_snapshot(self, 
                                  environment_id: str,
                                  test_type: str,
                                  container_id: Optional[str] = None,
                                  files_to_snapshot: List[Path] = None,
                                  metadata: Dict[str, Any] = None) -> str:
        """
        Create a snapshot of a test environment.
        
        Args:
            environment_id: ID for the environment
            test_type: Type of test environment
            container_id: Optional Docker container ID
            files_to_snapshot: List of files to include in the snapshot
            metadata: Optional metadata
            
        Returns:
            Snapshot ID
        """
        # Generate a snapshot ID
        snapshot_id = hashlib.md5(f"{environment_id}_{test_type}_{int(time.time())}".encode()).hexdigest()
        
        # Create metadata if not provided
        if metadata is None:
            metadata = {}
            
        metadata.update({
            "environment_id": environment_id,
            "test_type": test_type,
            "timestamp": datetime.now().isoformat()
        })
        
        # Cache the container if provided
        container_cache_key = ""
        if container_id:
            container_cache_key = self.container_cache.cache_container(
                container_id,
                test_type,
                environment_id,
                metadata
            )
            metadata["container_cache_key"] = container_cache_key
        
        # Snapshot files if provided
        file_snapshot_created = False
        if files_to_snapshot:
            file_snapshot_created = self._create_file_snapshot(snapshot_id, files_to_snapshot)
            metadata["file_snapshot"] = file_snapshot_created
        
        # Save metadata
        metadata_file = self.cache_dir / f"{snapshot_id}.json"
        try:
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            self.logger.exception(e, message=f"Failed to save metadata for snapshot {snapshot_id}")
        
        self.logger.info(f"Created environment snapshot {snapshot_id} for {test_type}")
        return snapshot_id
    
    def restore_environment_snapshot(self, 
                                   snapshot_id: str,
                                   restore_files: bool = True) -> Dict[str, Any]:
        """
        Restore a test environment from a snapshot.
        
        Args:
            snapshot_id: ID of the snapshot
            restore_files: Whether to restore files
            
        Returns:
            Dictionary with restoration results
        """
        # Load metadata
        metadata_file = self.cache_dir / f"{snapshot_id}.json"
        if not metadata_file.exists():
            self.logger.warning(f"Metadata not found for snapshot {snapshot_id}")
            return {"success": False, "error": "Metadata not found"}
            
        try:
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
        except Exception as e:
            self.logger.exception(e, message=f"Failed to load metadata for snapshot {snapshot_id}")
            return {"success": False, "error": str(e)}
        
        # Restore container
        container_id = None
        if "container_cache_key" in metadata:
            container_id = self.container_cache.load_cached_container(
                metadata["test_type"],
                metadata.get("environment_id")
            )
        
        # Restore files
        files_restored = False
        if restore_files and metadata.get("file_snapshot", False):
            files_restored = self._restore_file_snapshot(snapshot_id)
        
        result = {
            "success": True,
            "snapshot_id": snapshot_id,
            "container_id": container_id,
            "files_restored": files_restored,
            "metadata": metadata
        }
        
        self.logger.info(f"Restored environment snapshot {snapshot_id}")
        return result
    
    def delete_environment_snapshot(self, snapshot_id: str) -> bool:
        """
        Delete a test environment snapshot.
        
        Args:
            snapshot_id: ID of the snapshot
            
        Returns:
            True if successful, False otherwise
        """
        # Load metadata
        metadata_file = self.cache_dir / f"{snapshot_id}.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                    
                # Remove container cache entry if exists
                if "container_cache_key" in metadata:
                    # Note: The container cache manager handles removal internally
                    pass
            except Exception as e:
                self.logger.exception(e, message=f"Failed to load metadata for snapshot {snapshot_id}")
        
        # Delete the file snapshot
        file_snapshot_deleted = self._delete_snapshot(snapshot_id)
        
        # Delete the metadata file
        try:
            metadata_file.unlink(missing_ok=True)
        except Exception as e:
            self.logger.exception(e, message=f"Failed to delete metadata for snapshot {snapshot_id}")
            return False
        
        self.logger.info(f"Deleted environment snapshot {snapshot_id}")
        return file_snapshot_deleted
    
    def list_snapshots(self, test_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List available snapshots.
        
        Args:
            test_type: Optional filter by test type
            
        Returns:
            List of snapshot metadata
        """
        snapshots = []
        
        # Look for metadata files
        for metadata_file in self.cache_dir.glob("*.json"):
            try:
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                    
                # Filter by test type if specified
                if test_type and metadata.get("test_type") != test_type:
                    continue
                    
                # Add to the list
                snapshots.append(metadata)
                
            except Exception as e:
                self.logger.exception(e, message=f"Failed to load metadata from {metadata_file}")
        
        # Sort by timestamp (newest first)
        snapshots.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        return snapshots
    
    def clear_cache(self) -> None:
        """Clear all cached environments."""
        # Clear container cache
        self.container_cache.clear_cache()
        
        # Clear file snapshots
        try:
            for snapshot_dir in self.snapshot_dir.glob("*"):
                if snapshot_dir.is_dir():
                    shutil.rmtree(snapshot_dir)
                    
            # Delete metadata files
            for metadata_file in self.cache_dir.glob("*.json"):
                metadata_file.unlink()
                
            self.logger.info("Cleared environment cache")
            
        except Exception as e:
            self.logger.exception(e, message="Failed to clear environment cache")


if __name__ == "__main__":
    # Example usage
    env_cache = TestEnvironmentCache()
    
    # List existing snapshots
    snapshots = env_cache.list_snapshots()
    print(f"Found {len(snapshots)} snapshots")
    
    # Create a snapshot (without container)
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w") as f:
        f.write("# Test file for snapshot\nprint('Hello world!')\n")
        f.flush()
        
        snapshot_id = env_cache.create_environment_snapshot(
            "test_env_1",
            "unit",
            files_to_snapshot=[Path(f.name)],
            metadata={"purpose": "testing"}
        )
        
        print(f"Created snapshot: {snapshot_id}")
    
    # Restore the snapshot
    result = env_cache.restore_environment_snapshot(snapshot_id)
    print(f"Restored snapshot: {result['success']}")
    
    # Delete the snapshot
    deleted = env_cache.delete_environment_snapshot(snapshot_id)
    print(f"Deleted snapshot: {deleted}")
    
    # Get container cache stats
    stats = env_cache.container_cache.get_cache_stats()
    print(f"Container cache stats: {stats}")