"""
Configuration Management Database (CMDB) Synchronization Module

Provides synchronization between Homeostasis healing system and enterprise CMDBs
to maintain accurate configuration item records and relationships.
"""

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import requests

logger = logging.getLogger(__name__)


class CIType(Enum):
    """Configuration Item types"""

    APPLICATION = "application"
    SERVICE = "service"
    SERVER = "server"
    DATABASE = "database"
    NETWORK_DEVICE = "network_device"
    CONTAINER = "container"
    KUBERNETES_CLUSTER = "kubernetes_cluster"
    MICROSERVICE = "microservice"
    API = "api"
    LOAD_BALANCER = "load_balancer"
    FUNCTION = "function"  # Serverless functions
    QUEUE = "queue"
    CACHE = "cache"
    STORAGE = "storage"


class CIStatus(Enum):
    """Configuration Item status"""

    ACTIVE = "active"
    INACTIVE = "inactive"
    MAINTENANCE = "maintenance"
    DECOMMISSIONED = "decommissioned"
    PLANNED = "planned"


class RelationshipType(Enum):
    """Types of relationships between CIs"""

    DEPENDS_ON = "depends_on"
    USED_BY = "used_by"
    RUNS_ON = "runs_on"
    HOSTED_BY = "hosted_by"
    CONNECTS_TO = "connects_to"
    MANAGES = "manages"
    CONTAINS = "contains"
    PART_OF = "part_of"
    BACKUP_FOR = "backup_for"
    CLUSTER_MEMBER = "cluster_member"


@dataclass
class CMDBItem:
    """Represents a Configuration Item in the CMDB"""

    ci_id: Optional[str] = None
    name: str = ""
    ci_type: CIType = CIType.APPLICATION
    status: CIStatus = CIStatus.ACTIVE
    description: str = ""
    owner: Optional[str] = None
    support_group: Optional[str] = None
    environment: str = "production"  # production, staging, development, test
    location: Optional[str] = None
    version: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    healing_enabled: bool = True
    healing_config: Dict[str, Any] = field(default_factory=dict)
    last_discovered: Optional[datetime] = None
    last_modified: Optional[datetime] = None
    checksum: Optional[str] = None  # For change detection

    def calculate_checksum(self) -> str:
        """Calculate checksum for change detection"""
        data = {
            "name": self.name,
            "ci_type": self.ci_type.value,
            "status": self.status.value,
            "version": self.version,
            "attributes": self.attributes,
            "tags": sorted(self.tags),
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


@dataclass
class CMDBRelationship:
    """Represents a relationship between Configuration Items"""

    relationship_id: Optional[str] = None
    source_ci_id: str = ""
    target_ci_id: str = ""
    relationship_type: RelationshipType = RelationshipType.DEPENDS_ON
    attributes: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "relationship_id": self.relationship_id,
            "source_ci_id": self.source_ci_id,
            "target_ci_id": self.target_ci_id,
            "relationship_type": self.relationship_type.value,
            "attributes": self.attributes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class CMDBSynchronizer(ABC):
    """Abstract base class for CMDB synchronizers"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sync_interval = config.get("sync_interval", 300)  # 5 minutes default
        self.batch_size = config.get("batch_size", 100)
        self.enable_auto_discovery = config.get("enable_auto_discovery", True)
        self.last_sync_time = None
        self._local_cache = {}  # Cache for performance

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to the CMDB"""
        pass

    @abstractmethod
    def get_ci(self, ci_id: str) -> Optional[CMDBItem]:
        """Retrieve a Configuration Item by ID"""
        pass

    @abstractmethod
    def create_ci(self, ci: CMDBItem) -> Tuple[bool, str]:
        """Create a new Configuration Item"""
        pass

    @abstractmethod
    def update_ci(self, ci_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing Configuration Item"""
        pass

    @abstractmethod
    def delete_ci(self, ci_id: str) -> bool:
        """Delete a Configuration Item"""
        pass

    @abstractmethod
    def search_cis(self, criteria: Dict[str, Any]) -> List[CMDBItem]:
        """Search for Configuration Items"""
        pass

    @abstractmethod
    def get_relationships(self, ci_id: str) -> List[CMDBRelationship]:
        """Get all relationships for a Configuration Item"""
        pass

    @abstractmethod
    def create_relationship(self, relationship: CMDBRelationship) -> bool:
        """Create a relationship between Configuration Items"""
        pass

    @abstractmethod
    def delete_relationship(self, relationship_id: str) -> bool:
        """Delete a relationship"""
        pass

    def sync_from_homeostasis(
        self, discovered_items: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Sync discovered items from Homeostasis to CMDB"""
        results = {
            "created": 0,
            "updated": 0,
            "unchanged": 0,
            "errors": 0,
            "details": [],
        }

        for item_data in discovered_items:
            try:
                ci = self._convert_to_cmdb_item(item_data)
                existing = self.get_ci(ci.ci_id) if ci.ci_id else None

                if existing:
                    # Check if update needed
                    new_checksum = ci.calculate_checksum()
                    if existing.checksum != new_checksum:
                        ci.checksum = new_checksum
                        if self.update_ci(ci.ci_id, ci.__dict__):
                            results["updated"] += 1
                            results["details"].append(
                                {
                                    "action": "updated",
                                    "ci_id": ci.ci_id,
                                    "name": ci.name,
                                }
                            )
                        else:
                            results["errors"] += 1
                    else:
                        results["unchanged"] += 1
                else:
                    # Create new CI
                    success, ci_id = self.create_ci(ci)
                    if success:
                        results["created"] += 1
                        results["details"].append(
                            {"action": "created", "ci_id": ci_id, "name": ci.name}
                        )
                    else:
                        results["errors"] += 1

            except Exception as e:
                logger.error(
                    f"Error syncing item {item_data.get('name', 'unknown')}: {e}"
                )
                results["errors"] += 1

        self.last_sync_time = datetime.utcnow()
        return results

    def sync_to_homeostasis(self) -> List[CMDBItem]:
        """Sync Configuration Items from CMDB to Homeostasis"""
        # Search for all items with healing enabled
        criteria = {
            "healing_enabled": True,
            "status": [CIStatus.ACTIVE.value, CIStatus.MAINTENANCE.value],
        }

        items = self.search_cis(criteria)

        # Update local cache
        for item in items:
            if item.ci_id:
                self._local_cache[item.ci_id] = item

        return items

    def get_impact_analysis(self, ci_id: str, depth: int = 3) -> Dict[str, Any]:
        """Analyze potential impact of changes to a CI"""
        visited = set()
        impact_map = {"direct": [], "indirect": [], "critical_paths": []}

        self._traverse_relationships(ci_id, impact_map, visited, 0, depth)

        # Identify critical paths
        critical_types = [CIType.DATABASE, CIType.API, CIType.SERVICE]
        for ci_info in impact_map["direct"] + impact_map["indirect"]:
            if ci_info.get("ci_type") in [t.value for t in critical_types]:
                impact_map["critical_paths"].append(ci_info)

        return impact_map

    def _traverse_relationships(
        self,
        ci_id: str,
        impact_map: Dict,
        visited: Set[str],
        current_depth: int,
        max_depth: int,
    ):
        """Recursively traverse CI relationships"""
        if ci_id in visited or current_depth > max_depth:
            return

        visited.add(ci_id)
        relationships = self.get_relationships(ci_id)

        for rel in relationships:
            if rel.relationship_type in [
                RelationshipType.USED_BY,
                RelationshipType.DEPENDS_ON,
            ]:
                target_ci = self.get_ci(rel.target_ci_id)
                if target_ci:
                    ci_info = {
                        "ci_id": target_ci.ci_id,
                        "name": target_ci.name,
                        "ci_type": target_ci.ci_type.value,
                        "relationship": rel.relationship_type.value,
                        "depth": current_depth + 1,
                    }

                    if current_depth == 0:
                        impact_map["direct"].append(ci_info)
                    else:
                        impact_map["indirect"].append(ci_info)

                    self._traverse_relationships(
                        target_ci.ci_id,
                        impact_map,
                        visited,
                        current_depth + 1,
                        max_depth,
                    )

    def _convert_to_cmdb_item(self, data: Dict[str, Any]) -> CMDBItem:
        """Convert discovered data to CMDB item"""
        ci_type = self._determine_ci_type(data)

        return CMDBItem(
            ci_id=data.get("id"),
            name=data.get("name", ""),
            ci_type=ci_type,
            status=CIStatus.ACTIVE,
            description=data.get("description", ""),
            owner=data.get("owner"),
            support_group=data.get("support_group"),
            environment=data.get("environment", "production"),
            location=data.get("location"),
            version=data.get("version"),
            attributes=data.get("attributes", {}),
            tags=data.get("tags", []),
            healing_enabled=data.get("healing_enabled", True),
            healing_config=data.get("healing_config", {}),
            last_discovered=datetime.utcnow(),
        )

    def _determine_ci_type(self, data: Dict[str, Any]) -> CIType:
        """Determine CI type from discovered data"""
        type_mapping = {
            "application": CIType.APPLICATION,
            "service": CIType.SERVICE,
            "microservice": CIType.MICROSERVICE,
            "database": CIType.DATABASE,
            "container": CIType.CONTAINER,
            "kubernetes": CIType.KUBERNETES_CLUSTER,
            "function": CIType.FUNCTION,
            "api": CIType.API,
        }

        detected_type = data.get("type", "").lower()
        for key, ci_type in type_mapping.items():
            if key in detected_type:
                return ci_type

        return CIType.APPLICATION  # Default


class ServiceNowCMDB(CMDBSynchronizer):
    """ServiceNow CMDB synchronizer implementation"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.instance = config.get("instance", "")
        self.username = config.get("username", "")
        self.password = config.get("password", "")
        self.base_url = f"https://{self.instance}.service-now.com"
        self._session = None

    def connect(self) -> bool:
        """Connect to ServiceNow CMDB"""
        try:
            self._session = requests.Session()
            self._session.auth = (self.username, self.password)
            self._session.headers.update(
                {"Accept": "application/json", "Content-Type": "application/json"}
            )

            # Test connection
            test_url = f"{self.base_url}/api/now/table/cmdb_ci?sysparm_limit=1"
            response = self._session.get(test_url)
            response.raise_for_status()

            logger.info("Successfully connected to ServiceNow CMDB")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to ServiceNow CMDB: {e}")
            return False

    def get_ci(self, ci_id: str) -> Optional[CMDBItem]:
        """Get CI from ServiceNow"""
        try:
            url = f"{self.base_url}/api/now/table/cmdb_ci/{ci_id}"
            response = self._session.get(url)
            response.raise_for_status()

            data = response.json()["result"]
            return self._parse_servicenow_ci(data)

        except Exception as e:
            logger.error(f"Failed to get CI from ServiceNow: {e}")
            return None

    def create_ci(self, ci: CMDBItem) -> Tuple[bool, str]:
        """Create CI in ServiceNow"""
        try:
            # Determine appropriate table based on CI type
            table = self._get_table_for_ci_type(ci.ci_type)
            url = f"{self.base_url}/api/now/table/{table}"

            data = {
                "name": ci.name,
                "short_description": ci.description,
                "operational_status": self._map_status_to_servicenow(ci.status),
                "environment": ci.environment,
                "version": ci.version,
                "owned_by": ci.owner,
                "support_group": ci.support_group,
                "location": ci.location,
                "attributes": json.dumps(ci.attributes),
                "u_healing_enabled": ci.healing_enabled,
                "u_healing_config": json.dumps(ci.healing_config),
            }

            response = self._session.post(url, json=data)
            response.raise_for_status()

            result = response.json()["result"]
            ci_id = result["sys_id"]

            logger.info(f"Created CI in ServiceNow: {ci_id}")
            return True, ci_id

        except Exception as e:
            logger.error(f"Failed to create CI in ServiceNow: {e}")
            return False, str(e)

    def update_ci(self, ci_id: str, updates: Dict[str, Any]) -> bool:
        """Update CI in ServiceNow"""
        try:
            # First get the CI to determine its table
            existing = self.get_ci(ci_id)
            if not existing:
                return False

            table = self._get_table_for_ci_type(existing.ci_type)
            url = f"{self.base_url}/api/now/table/{table}/{ci_id}"

            # Map updates to ServiceNow fields
            sn_updates = {}
            field_mapping = {
                "name": "name",
                "description": "short_description",
                "status": "operational_status",
                "environment": "environment",
                "version": "version",
                "owner": "owned_by",
                "support_group": "support_group",
                "location": "location",
            }

            for key, value in updates.items():
                if key in field_mapping:
                    sn_field = field_mapping[key]
                    if key == "status" and isinstance(value, CIStatus):
                        value = self._map_status_to_servicenow(value)
                    sn_updates[sn_field] = value

            if "attributes" in updates:
                sn_updates["attributes"] = json.dumps(updates["attributes"])

            if "healing_config" in updates:
                sn_updates["u_healing_config"] = json.dumps(updates["healing_config"])

            response = self._session.patch(url, json=sn_updates)
            response.raise_for_status()

            logger.info(f"Updated CI in ServiceNow: {ci_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to update CI in ServiceNow: {e}")
            return False

    def delete_ci(self, ci_id: str) -> bool:
        """Delete CI from ServiceNow (soft delete - set to decommissioned)"""
        try:
            return self.update_ci(ci_id, {"status": CIStatus.DECOMMISSIONED})

        except Exception as e:
            logger.error(f"Failed to delete CI from ServiceNow: {e}")
            return False

    def search_cis(self, criteria: Dict[str, Any]) -> List[CMDBItem]:
        """Search CIs in ServiceNow"""
        try:
            # Build query
            query_parts = []

            if "healing_enabled" in criteria:
                query_parts.append(f"u_healing_enabled={criteria['healing_enabled']}")

            if "status" in criteria:
                statuses = criteria["status"]
                if not isinstance(statuses, list):
                    statuses = [statuses]
                status_query = []
                for status in statuses:
                    sn_status = self._map_status_to_servicenow(CIStatus(status))
                    status_query.append(f"operational_status={sn_status}")
                if status_query:
                    query_parts.append(f"({' OR '.join(status_query)})")

            if "environment" in criteria:
                query_parts.append(f"environment={criteria['environment']}")

            if "ci_type" in criteria:
                # Search specific table
                table = self._get_table_for_ci_type(criteria["ci_type"])
            else:
                # Search base CI table
                table = "cmdb_ci"

            url = f"{self.base_url}/api/now/table/{table}"
            params = {"sysparm_limit": self.batch_size, "sysparm_offset": 0}

            if query_parts:
                params["sysparm_query"] = "^".join(query_parts)

            items = []
            while True:
                response = self._session.get(url, params=params)
                response.raise_for_status()

                results = response.json()["result"]
                if not results:
                    break

                for data in results:
                    ci = self._parse_servicenow_ci(data)
                    if ci:
                        items.append(ci)

                params["sysparm_offset"] += self.batch_size

                # Limit total results
                if len(items) >= 1000:
                    break

            return items

        except Exception as e:
            logger.error(f"Failed to search CIs in ServiceNow: {e}")
            return []

    def get_relationships(self, ci_id: str) -> List[CMDBRelationship]:
        """Get relationships for a CI from ServiceNow"""
        try:
            url = f"{self.base_url}/api/now/table/cmdb_rel_ci"
            params = {
                "sysparm_query": f"parent={ci_id}^ORchild={ci_id}",
                "sysparm_limit": 1000,
            }

            response = self._session.get(url, params=params)
            response.raise_for_status()

            relationships = []
            for data in response.json()["result"]:
                rel = CMDBRelationship(
                    relationship_id=data["sys_id"],
                    source_ci_id=data["parent"]["value"],
                    target_ci_id=data["child"]["value"],
                    relationship_type=self._parse_relationship_type(
                        data["type"]["value"]
                    ),
                )
                relationships.append(rel)

            return relationships

        except Exception as e:
            logger.error(f"Failed to get relationships from ServiceNow: {e}")
            return []

    def create_relationship(self, relationship: CMDBRelationship) -> bool:
        """Create relationship in ServiceNow"""
        try:
            url = f"{self.base_url}/api/now/table/cmdb_rel_ci"

            data = {
                "parent": relationship.source_ci_id,
                "child": relationship.target_ci_id,
                "type": self._map_relationship_to_servicenow(
                    relationship.relationship_type
                ),
            }

            response = self._session.post(url, json=data)
            response.raise_for_status()

            logger.info("Created relationship in ServiceNow")
            return True

        except Exception as e:
            logger.error(f"Failed to create relationship in ServiceNow: {e}")
            return False

    def delete_relationship(self, relationship_id: str) -> bool:
        """Delete relationship from ServiceNow"""
        try:
            url = f"{self.base_url}/api/now/table/cmdb_rel_ci/{relationship_id}"
            response = self._session.delete(url)
            response.raise_for_status()

            logger.info(f"Deleted relationship from ServiceNow: {relationship_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete relationship from ServiceNow: {e}")
            return False

    def _get_table_for_ci_type(self, ci_type: CIType) -> str:
        """Map CI type to ServiceNow table"""
        table_mapping = {
            CIType.APPLICATION: "cmdb_ci_appl",
            CIType.SERVICE: "cmdb_ci_service",
            CIType.SERVER: "cmdb_ci_server",
            CIType.DATABASE: "cmdb_ci_database",
            CIType.NETWORK_DEVICE: "cmdb_ci_netgear",
            CIType.CONTAINER: "cmdb_ci_docker_container",
            CIType.KUBERNETES_CLUSTER: "cmdb_ci_kubernetes_cluster",
        }
        return table_mapping.get(ci_type, "cmdb_ci")

    def _map_status_to_servicenow(self, status: CIStatus) -> str:
        """Map status to ServiceNow operational status"""
        mapping = {
            CIStatus.ACTIVE: "1",  # Operational
            CIStatus.INACTIVE: "2",  # Non-Operational
            CIStatus.MAINTENANCE: "3",  # Repair In Progress
            CIStatus.DECOMMISSIONED: "6",  # Retired
            CIStatus.PLANNED: "100",  # Pre-Production
        }
        return mapping.get(status, "1")

    def _map_relationship_to_servicenow(self, rel_type: RelationshipType) -> str:
        """Map relationship type to ServiceNow"""
        mapping = {
            RelationshipType.DEPENDS_ON: "d93304fb30854943",  # Depends on::Used by
            RelationshipType.RUNS_ON: "6b72318f47201200",  # Runs on::Runs
            RelationshipType.HOSTED_BY: "87a7a18f47201200",  # Hosted on::Hosts
            RelationshipType.CONTAINS: "55534dbe37002100",  # Contains::Contained by
        }
        return mapping.get(rel_type, "d93304fb30854943")

    def _parse_relationship_type(self, sn_type: str) -> RelationshipType:
        """Parse ServiceNow relationship type"""
        type_mapping = {
            "d93304fb30854943": RelationshipType.DEPENDS_ON,
            "6b72318f47201200": RelationshipType.RUNS_ON,
            "87a7a18f47201200": RelationshipType.HOSTED_BY,
            "55534dbe37002100": RelationshipType.CONTAINS,
        }
        return type_mapping.get(sn_type, RelationshipType.DEPENDS_ON)

    def _parse_servicenow_ci(self, data: Dict[str, Any]) -> Optional[CMDBItem]:
        """Parse ServiceNow CI data"""
        try:
            # Reverse map status
            status_map = {
                "1": CIStatus.ACTIVE,
                "2": CIStatus.INACTIVE,
                "3": CIStatus.MAINTENANCE,
                "6": CIStatus.DECOMMISSIONED,
                "100": CIStatus.PLANNED,
            }

            # Determine CI type from sys_class_name
            ci_type = self._parse_ci_type(data.get("sys_class_name", ""))

            # Parse attributes
            attributes = {}
            if data.get("attributes"):
                try:
                    attributes = json.loads(data["attributes"])
                except (json.JSONDecodeError, ValueError):
                    pass

            # Parse healing config
            healing_config = {}
            if data.get("u_healing_config"):
                try:
                    healing_config = json.loads(data["u_healing_config"])
                except (json.JSONDecodeError, ValueError):
                    pass

            return CMDBItem(
                ci_id=data.get("sys_id"),
                name=data.get("name", ""),
                ci_type=ci_type,
                status=status_map.get(data.get("operational_status"), CIStatus.ACTIVE),
                description=data.get("short_description", ""),
                owner=data.get("owned_by"),
                support_group=data.get("support_group"),
                environment=data.get("environment", "production"),
                location=data.get("location"),
                version=data.get("version"),
                attributes=attributes,
                healing_enabled=data.get("u_healing_enabled", True),
                healing_config=healing_config,
                last_modified=(
                    datetime.fromisoformat(data["sys_updated_on"])
                    if data.get("sys_updated_on")
                    else None
                ),
            )

        except Exception as e:
            logger.error(f"Failed to parse ServiceNow CI: {e}")
            return None

    def _parse_ci_type(self, sys_class_name: str) -> CIType:
        """Parse CI type from ServiceNow sys_class_name"""
        type_mapping = {
            "cmdb_ci_appl": CIType.APPLICATION,
            "cmdb_ci_service": CIType.SERVICE,
            "cmdb_ci_server": CIType.SERVER,
            "cmdb_ci_database": CIType.DATABASE,
            "cmdb_ci_netgear": CIType.NETWORK_DEVICE,
            "cmdb_ci_docker_container": CIType.CONTAINER,
            "cmdb_ci_kubernetes_cluster": CIType.KUBERNETES_CLUSTER,
        }
        return type_mapping.get(sys_class_name, CIType.APPLICATION)


class DeviceCMDB(CMDBSynchronizer):
    """Device42 CMDB synchronizer implementation"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get("base_url", "")
        self.username = config.get("username", "")
        self.password = config.get("password", "")
        self._session = None

    def connect(self) -> bool:
        """Connect to Device42 CMDB"""
        try:
            self._session = requests.Session()
            self._session.auth = (self.username, self.password)
            self._session.headers.update(
                {"Accept": "application/json", "Content-Type": "application/json"}
            )

            # Test connection
            test_url = f"{self.base_url}/api/1.0/devices/"
            response = self._session.get(test_url)
            response.raise_for_status()

            logger.info("Successfully connected to Device42 CMDB")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to Device42 CMDB: {e}")
            return False

    def get_ci(self, ci_id: str) -> Optional[CMDBItem]:
        """Get CI from Device42"""
        try:
            # Device42 uses different endpoints for different CI types
            # Try devices first
            url = f"{self.base_url}/api/1.0/devices/id/{ci_id}/"
            response = self._session.get(url)

            if response.status_code == 404:
                # Try applications
                url = f"{self.base_url}/api/1.0/appcomps/{ci_id}/"
                response = self._session.get(url)

            response.raise_for_status()
            data = response.json()

            return self._parse_device42_ci(data)

        except Exception as e:
            logger.error(f"Failed to get CI from Device42: {e}")
            return None

    def create_ci(self, ci: CMDBItem) -> Tuple[bool, str]:
        """Create CI in Device42"""
        try:
            # Determine endpoint based on CI type
            if ci.ci_type in [CIType.SERVER, CIType.NETWORK_DEVICE]:
                url = f"{self.base_url}/api/1.0/devices/"
                data = {
                    "name": ci.name,
                    "notes": ci.description,
                    "in_service": ci.status == CIStatus.ACTIVE,
                    "tags": ",".join(ci.tags),
                    "custom_fields": ci.attributes,
                }
            else:
                url = f"{self.base_url}/api/1.0/appcomps/"
                data = {
                    "name": ci.name,
                    "notes": ci.description,
                    "custom_fields": ci.attributes,
                }

            response = self._session.post(url, json=data)
            response.raise_for_status()

            result = response.json()
            ci_id = str(result.get("id", ""))

            logger.info(f"Created CI in Device42: {ci_id}")
            return True, ci_id

        except Exception as e:
            logger.error(f"Failed to create CI in Device42: {e}")
            return False, str(e)

    def update_ci(self, ci_id: str, updates: Dict[str, Any]) -> bool:
        """Update CI in Device42"""
        try:
            # Get existing CI to determine type
            existing = self.get_ci(ci_id)
            if not existing:
                return False

            # Determine endpoint
            if existing.ci_type in [CIType.SERVER, CIType.NETWORK_DEVICE]:
                url = f"{self.base_url}/api/1.0/devices/id/{ci_id}/"
            else:
                url = f"{self.base_url}/api/1.0/appcomps/{ci_id}/"

            # Map updates
            data = {}
            if "name" in updates:
                data["name"] = updates["name"]
            if "description" in updates:
                data["notes"] = updates["description"]
            if "status" in updates:
                data["in_service"] = updates["status"] == CIStatus.ACTIVE.value
            if "tags" in updates:
                data["tags"] = ",".join(updates["tags"])
            if "attributes" in updates:
                data["custom_fields"] = updates["attributes"]

            response = self._session.put(url, json=data)
            response.raise_for_status()

            logger.info(f"Updated CI in Device42: {ci_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to update CI in Device42: {e}")
            return False

    def delete_ci(self, ci_id: str) -> bool:
        """Delete CI from Device42"""
        try:
            # Get existing CI to determine type
            existing = self.get_ci(ci_id)
            if not existing:
                return False

            # Determine endpoint
            if existing.ci_type in [CIType.SERVER, CIType.NETWORK_DEVICE]:
                url = f"{self.base_url}/api/1.0/devices/{ci_id}/"
            else:
                url = f"{self.base_url}/api/1.0/appcomps/{ci_id}/"

            response = self._session.delete(url)
            response.raise_for_status()

            logger.info(f"Deleted CI from Device42: {ci_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete CI from Device42: {e}")
            return False

    def search_cis(self, criteria: Dict[str, Any]) -> List[CMDBItem]:
        """Search CIs in Device42"""
        try:
            items = []

            # Search devices
            url = f"{self.base_url}/api/1.0/devices/"
            params = {}

            if "name" in criteria:
                params["name"] = criteria["name"]
            if "tags" in criteria:
                params["tags"] = criteria["tags"]

            response = self._session.get(url, params=params)
            response.raise_for_status()

            for device in response.json().get("Devices", []):
                ci = self._parse_device42_ci(device)
                if ci and self._matches_criteria(ci, criteria):
                    items.append(ci)

            # Also search application components if needed
            if not criteria.get("ci_type") or criteria["ci_type"] == CIType.APPLICATION:
                url = f"{self.base_url}/api/1.0/appcomps/"
                response = self._session.get(url)
                response.raise_for_status()

                for app in response.json():
                    ci = self._parse_device42_ci(app)
                    if ci and self._matches_criteria(ci, criteria):
                        items.append(ci)

            return items

        except Exception as e:
            logger.error(f"Failed to search CIs in Device42: {e}")
            return []

    def get_relationships(self, ci_id: str) -> List[CMDBRelationship]:
        """Get relationships from Device42"""
        try:
            url = f"{self.base_url}/api/1.0/device_connections/"
            params = {"device_id": ci_id}

            response = self._session.get(url, params=params)
            response.raise_for_status()

            relationships = []
            for conn in response.json():
                rel = CMDBRelationship(
                    relationship_id=str(conn.get("connection_id")),
                    source_ci_id=str(conn.get("device_id")),
                    target_ci_id=str(conn.get("connected_device_id")),
                    relationship_type=RelationshipType.CONNECTS_TO,
                )
                relationships.append(rel)

            return relationships

        except Exception as e:
            logger.error(f"Failed to get relationships from Device42: {e}")
            return []

    def create_relationship(self, relationship: CMDBRelationship) -> bool:
        """Create relationship in Device42"""
        try:
            url = f"{self.base_url}/api/1.0/device_connections/"
            data = {
                "device_id": relationship.source_ci_id,
                "connected_device_id": relationship.target_ci_id,
            }

            response = self._session.post(url, json=data)
            response.raise_for_status()

            logger.info("Created relationship in Device42")
            return True

        except Exception as e:
            logger.error(f"Failed to create relationship in Device42: {e}")
            return False

    def delete_relationship(self, relationship_id: str) -> bool:
        """Delete relationship from Device42"""
        try:
            url = f"{self.base_url}/api/1.0/device_connections/{relationship_id}/"
            response = self._session.delete(url)
            response.raise_for_status()

            logger.info(f"Deleted relationship from Device42: {relationship_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete relationship from Device42: {e}")
            return False

    def _parse_device42_ci(self, data: Dict[str, Any]) -> Optional[CMDBItem]:
        """Parse Device42 data into CMDBItem"""
        try:
            # Determine CI type
            if "device_id" in data:
                ci_type = CIType.SERVER
                ci_id = str(data.get("device_id"))
            else:
                ci_type = CIType.APPLICATION
                ci_id = str(data.get("id"))

            # Parse tags
            tags = []
            if data.get("tags"):
                tags = [t.strip() for t in data["tags"].split(",")]

            return CMDBItem(
                ci_id=ci_id,
                name=data.get("name", ""),
                ci_type=ci_type,
                status=(
                    CIStatus.ACTIVE
                    if data.get("in_service", True)
                    else CIStatus.INACTIVE
                ),
                description=data.get("notes", ""),
                tags=tags,
                attributes=data.get("custom_fields", {}),
                last_modified=(
                    datetime.fromisoformat(data["last_updated"])
                    if data.get("last_updated")
                    else None
                ),
            )

        except Exception as e:
            logger.error(f"Failed to parse Device42 CI: {e}")
            return None

    def _matches_criteria(self, ci: CMDBItem, criteria: Dict[str, Any]) -> bool:
        """Check if CI matches search criteria"""
        if "status" in criteria:
            statuses = criteria["status"]
            if not isinstance(statuses, list):
                statuses = [statuses]
            if ci.status.value not in statuses:
                return False

        if "healing_enabled" in criteria:
            if ci.healing_enabled != criteria["healing_enabled"]:
                return False

        if "environment" in criteria:
            if ci.environment != criteria["environment"]:
                return False

        return True


# Factory function to create CMDB synchronizers
def create_cmdb_synchronizer(
    provider: str, config: Dict[str, Any]
) -> Optional[CMDBSynchronizer]:
    """Factory function to create CMDB synchronizer instances"""
    providers = {
        "servicenow": ServiceNowCMDB,
        "device42": DeviceCMDB,
        # Add more providers as implemented
    }

    sync_class = providers.get(provider.lower())
    if not sync_class:
        logger.error(f"Unknown CMDB provider: {provider}")
        return None

    try:
        synchronizer = sync_class(config)
        if synchronizer.connect():
            return synchronizer
        else:
            logger.error(f"Failed to connect to {provider} CMDB")
            return None
    except Exception as e:
        logger.error(f"Failed to create {provider} CMDB synchronizer: {e}")
        return None
