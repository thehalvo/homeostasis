"""
IT Service Management (ITSM) Integration Module

Provides integration with popular ITSM tools like ServiceNow, Jira Service Management,
BMC Remedy, and others for incident management, change requests, and problem tracking.
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)


class IncidentPriority(Enum):
    """Standard incident priority levels"""

    CRITICAL = "1"
    HIGH = "2"
    MEDIUM = "3"
    LOW = "4"
    PLANNING = "5"


class IncidentStatus(Enum):
    """Standard incident status values"""

    NEW = "new"
    IN_PROGRESS = "in_progress"
    ON_HOLD = "on_hold"
    RESOLVED = "resolved"
    CLOSED = "closed"
    CANCELLED = "cancelled"


class ChangeRequestStatus(Enum):
    """Standard change request status values"""

    NEW = "new"
    ASSESS = "assess"
    AUTHORIZE = "authorize"
    SCHEDULED = "scheduled"
    IMPLEMENT = "implement"
    REVIEW = "review"
    CLOSED = "closed"
    CANCELLED = "cancelled"


@dataclass
class ITSMIncident:
    """Represents an ITSM incident"""

    incident_id: Optional[str] = None
    short_description: str = ""
    description: str = ""
    priority: IncidentPriority = IncidentPriority.MEDIUM
    status: IncidentStatus = IncidentStatus.NEW
    assignment_group: Optional[str] = None
    assigned_to: Optional[str] = None
    category: Optional[str] = None
    subcategory: Optional[str] = None
    impact: Optional[str] = None
    urgency: Optional[str] = None
    caller_id: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    resolution_notes: Optional[str] = None
    custom_fields: Dict[str, Any] = field(default_factory=dict)
    healing_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ITSMChangeRequest:
    """Represents an ITSM change request"""

    change_id: Optional[str] = None
    short_description: str = ""
    description: str = ""
    justification: str = ""
    risk_assessment: str = ""
    implementation_plan: str = ""
    backout_plan: str = ""
    test_plan: str = ""
    status: ChangeRequestStatus = ChangeRequestStatus.NEW
    type: str = "standard"  # standard, normal, emergency
    assignment_group: Optional[str] = None
    assigned_to: Optional[str] = None
    requested_by: Optional[str] = None
    scheduled_start: Optional[datetime] = None
    scheduled_end: Optional[datetime] = None
    actual_start: Optional[datetime] = None
    actual_end: Optional[datetime] = None
    approval_status: Optional[str] = None
    approvers: List[str] = field(default_factory=list)
    custom_fields: Dict[str, Any] = field(default_factory=dict)
    healing_context: Dict[str, Any] = field(default_factory=dict)


class ITSMConnector(ABC):
    """Abstract base class for ITSM connectors"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_url = config.get("base_url", "")
        self.auth_method = config.get("auth_method", "basic")
        self.verify_ssl = config.get("verify_ssl", True)
        self._session: Optional[requests.Session] = None

    @abstractmethod
    def authenticate(self) -> bool:
        """Authenticate with the ITSM system"""
        pass

    @abstractmethod
    def create_incident(self, incident: ITSMIncident) -> Tuple[bool, str]:
        """Create a new incident"""
        pass

    @abstractmethod
    def update_incident(self, incident_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing incident"""
        pass

    @abstractmethod
    def get_incident(self, incident_id: str) -> Optional[ITSMIncident]:
        """Retrieve an incident by ID"""
        pass

    @abstractmethod
    def create_change_request(self, change: ITSMChangeRequest) -> Tuple[bool, str]:
        """Create a new change request"""
        pass

    @abstractmethod
    def update_change_request(self, change_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing change request"""
        pass

    @abstractmethod
    def get_change_request(self, change_id: str) -> Optional[ITSMChangeRequest]:
        """Retrieve a change request by ID"""
        pass

    @abstractmethod
    def search_incidents(self, query: Dict[str, Any]) -> List[ITSMIncident]:
        """Search for incidents based on criteria"""
        pass

    @abstractmethod
    def link_to_healing_action(
        self, ticket_id: str, healing_action_id: str, ticket_type: str = "incident"
    ) -> bool:
        """Link an ITSM ticket to a Homeostasis healing action"""
        pass

    def close_session(self):
        """Close the session if needed"""
        if self._session:
            self._session.close()
            self._session = None


class ServiceNowConnector(ITSMConnector):
    """ServiceNow ITSM connector implementation"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.instance = config.get("instance", "")
        self.username = config.get("username", "")
        self.password = config.get("password", "")
        self.client_id = config.get("client_id", "")
        self.client_secret = config.get("client_secret", "")
        self.refresh_token = config.get("refresh_token", "")
        self.access_token = None

    def authenticate(self) -> bool:
        """Authenticate with ServiceNow"""
        try:
            self._session = requests.Session()

            if self.auth_method == "oauth":
                # OAuth authentication
                token_url = f"{self.base_url}/oauth_token.do"
                data = {
                    "grant_type": "refresh_token" if self.refresh_token else "password",
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                }

                if self.refresh_token:
                    data["refresh_token"] = self.refresh_token
                else:
                    data["username"] = self.username
                    data["password"] = self.password

                if not self._session:
                    raise RuntimeError("Session not initialized")
                response = self._session.post(
                    token_url, data=data, verify=self.verify_ssl
                )
                response.raise_for_status()

                token_data = response.json()
                self.access_token = token_data.get("access_token")
                if self._session:
                    self._session.headers.update(
                        {"Authorization": f"Bearer {self.access_token}"}
                    )
            else:
                # Basic authentication
                if self._session:
                    self._session.auth = (self.username, self.password)

            # Test connection
            test_url = f"{self.base_url}/api/now/table/incident?sysparm_limit=1"
            if not self._session:
                raise RuntimeError("Session not initialized")
            response = self._session.get(test_url, verify=self.verify_ssl)
            response.raise_for_status()

            logger.info("Successfully authenticated with ServiceNow")
            return True

        except Exception as e:
            logger.error(f"Failed to authenticate with ServiceNow: {e}")
            return False

    def create_incident(self, incident: ITSMIncident) -> Tuple[bool, str]:
        """Create a new incident in ServiceNow"""
        try:
            url = f"{self.base_url}/api/now/table/incident"

            data = {
                "short_description": incident.short_description,
                "description": incident.description,
                "priority": incident.priority.value,
                "state": self._map_incident_status(incident.status),
                "category": incident.category,
                "subcategory": incident.subcategory,
                "impact": incident.impact or incident.priority.value,
                "urgency": incident.urgency or incident.priority.value,
                "caller_id": incident.caller_id,
                "assignment_group": incident.assignment_group,
                "assigned_to": incident.assigned_to,
                "work_notes": f"Created by Homeostasis healing system\n{json.dumps(incident.healing_context, indent=2)}",
            }

            # Add custom fields
            data.update(incident.custom_fields)

            response = self._session.post(url, json=data, verify=self.verify_ssl)
            response.raise_for_status()

            result = response.json()
            incident_number = result["result"]["number"]

            logger.info(f"Created ServiceNow incident: {incident_number}")
            return True, incident_number

        except Exception as e:
            logger.error(f"Failed to create ServiceNow incident: {e}")
            return False, str(e)

    def update_incident(self, incident_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing incident in ServiceNow"""
        try:
            # Handle both sys_id and incident number
            url = f"{self.base_url}/api/now/table/incident/{incident_id}"

            # Map any status updates
            if "status" in updates and isinstance(updates["status"], IncidentStatus):
                updates["state"] = self._map_incident_status(updates["status"])
                del updates["status"]

            response = self._session.patch(url, json=updates, verify=self.verify_ssl)
            response.raise_for_status()

            logger.info(f"Updated ServiceNow incident: {incident_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to update ServiceNow incident: {e}")
            return False

    def get_incident(self, incident_id: str) -> Optional[ITSMIncident]:
        """Retrieve an incident from ServiceNow"""
        try:
            url = f"{self.base_url}/api/now/table/incident/{incident_id}"

            response = self._session.get(url, verify=self.verify_ssl)
            response.raise_for_status()

            data = response.json()["result"]

            return self._parse_incident(data)

        except Exception as e:
            logger.error(f"Failed to retrieve ServiceNow incident: {e}")
            return None

    def create_change_request(self, change: ITSMChangeRequest) -> Tuple[bool, str]:
        """Create a new change request in ServiceNow"""
        try:
            url = f"{self.base_url}/api/now/table/change_request"

            data = {
                "short_description": change.short_description,
                "description": change.description,
                "justification": change.justification,
                "risk_assessment": change.risk_assessment,
                "implementation_plan": change.implementation_plan,
                "backout_plan": change.backout_plan,
                "test_plan": change.test_plan,
                "state": self._map_change_status(change.status),
                "type": change.type,
                "assignment_group": change.assignment_group,
                "assigned_to": change.assigned_to,
                "requested_by": change.requested_by,
                "start_date": (
                    change.scheduled_start.isoformat()
                    if change.scheduled_start
                    else None
                ),
                "end_date": (
                    change.scheduled_end.isoformat() if change.scheduled_end else None
                ),
                "work_notes": f"Created by Homeostasis healing system\n{json.dumps(change.healing_context, indent=2)}",
            }

            # Add custom fields
            data.update(change.custom_fields)

            response = self._session.post(url, json=data, verify=self.verify_ssl)
            response.raise_for_status()

            result = response.json()
            change_number = result["result"]["number"]

            logger.info(f"Created ServiceNow change request: {change_number}")
            return True, change_number

        except Exception as e:
            logger.error(f"Failed to create ServiceNow change request: {e}")
            return False, str(e)

    def update_change_request(self, change_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing change request in ServiceNow"""
        try:
            url = f"{self.base_url}/api/now/table/change_request/{change_id}"

            # Map any status updates
            if "status" in updates and isinstance(
                updates["status"], ChangeRequestStatus
            ):
                updates["state"] = self._map_change_status(updates["status"])
                del updates["status"]

            response = self._session.patch(url, json=updates, verify=self.verify_ssl)
            response.raise_for_status()

            logger.info(f"Updated ServiceNow change request: {change_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to update ServiceNow change request: {e}")
            return False

    def get_change_request(self, change_id: str) -> Optional[ITSMChangeRequest]:
        """Retrieve a change request from ServiceNow"""
        try:
            url = f"{self.base_url}/api/now/table/change_request/{change_id}"

            response = self._session.get(url, verify=self.verify_ssl)
            response.raise_for_status()

            data = response.json()["result"]

            return self._parse_change_request(data)

        except Exception as e:
            logger.error(f"Failed to retrieve ServiceNow change request: {e}")
            return None

    def search_incidents(self, query: Dict[str, Any]) -> List[ITSMIncident]:
        """Search for incidents in ServiceNow"""
        try:
            url = f"{self.base_url}/api/now/table/incident"

            # Build query parameters
            params = {
                "sysparm_limit": query.get("limit", 100),
                "sysparm_offset": query.get("offset", 0),
            }

            # Build query string
            query_parts = []
            if "status" in query:
                query_parts.append(
                    f"state={self._map_incident_status(query['status'])}"
                )
            if "priority" in query:
                query_parts.append(f"priority={query['priority'].value}")
            if "assignment_group" in query:
                query_parts.append(f"assignment_group={query['assignment_group']}")
            if "category" in query:
                query_parts.append(f"category={query['category']}")

            if query_parts:
                params["sysparm_query"] = "^".join(query_parts)

            response = self._session.get(url, params=params, verify=self.verify_ssl)
            response.raise_for_status()

            results = response.json()["result"]

            return [self._parse_incident(data) for data in results]

        except Exception as e:
            logger.error(f"Failed to search ServiceNow incidents: {e}")
            return []

    def link_to_healing_action(
        self, ticket_id: str, healing_action_id: str, ticket_type: str = "incident"
    ) -> bool:
        """Link a ServiceNow ticket to a Homeostasis healing action"""
        try:
            table = "incident" if ticket_type == "incident" else "change_request"
            url = f"{self.base_url}/api/now/table/{table}/{ticket_id}"

            # Add healing action reference to work notes
            work_note = f"Linked to Homeostasis healing action: {healing_action_id}"

            response = self._session.patch(
                url, json={"work_notes": work_note}, verify=self.verify_ssl
            )
            response.raise_for_status()

            logger.info(
                f"Linked ServiceNow {ticket_type} {ticket_id} to healing action {healing_action_id}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to link ServiceNow ticket: {e}")
            return False

    def _map_incident_status(self, status: IncidentStatus) -> str:
        """Map generic status to ServiceNow incident state"""
        mapping = {
            IncidentStatus.NEW: "1",  # New
            IncidentStatus.IN_PROGRESS: "2",  # In Progress
            IncidentStatus.ON_HOLD: "3",  # On Hold
            IncidentStatus.RESOLVED: "6",  # Resolved
            IncidentStatus.CLOSED: "7",  # Closed
            IncidentStatus.CANCELLED: "8",  # Cancelled
        }
        return mapping.get(status, "1")

    def _map_change_status(self, status: ChangeRequestStatus) -> str:
        """Map generic status to ServiceNow change state"""
        mapping = {
            ChangeRequestStatus.NEW: "-5",  # New
            ChangeRequestStatus.ASSESS: "-4",  # Assess
            ChangeRequestStatus.AUTHORIZE: "-3",  # Authorize
            ChangeRequestStatus.SCHEDULED: "-2",  # Scheduled
            ChangeRequestStatus.IMPLEMENT: "-1",  # Implement
            ChangeRequestStatus.REVIEW: "0",  # Review
            ChangeRequestStatus.CLOSED: "3",  # Closed
            ChangeRequestStatus.CANCELLED: "4",  # Cancelled
        }
        return mapping.get(status, "-5")

    def _parse_incident(self, data: Dict[str, Any]) -> ITSMIncident:
        """Parse ServiceNow incident data into ITSMIncident object"""
        # Reverse map status
        state_to_status = {
            "1": IncidentStatus.NEW,
            "2": IncidentStatus.IN_PROGRESS,
            "3": IncidentStatus.ON_HOLD,
            "6": IncidentStatus.RESOLVED,
            "7": IncidentStatus.CLOSED,
            "8": IncidentStatus.CANCELLED,
        }

        # Reverse map priority
        priority_map = {
            "1": IncidentPriority.CRITICAL,
            "2": IncidentPriority.HIGH,
            "3": IncidentPriority.MEDIUM,
            "4": IncidentPriority.LOW,
            "5": IncidentPriority.PLANNING,
        }

        return ITSMIncident(
            incident_id=data.get("sys_id"),
            short_description=data.get("short_description", ""),
            description=data.get("description", ""),
            priority=priority_map.get(data.get("priority"), IncidentPriority.MEDIUM),
            status=state_to_status.get(data.get("state"), IncidentStatus.NEW),
            assignment_group=data.get("assignment_group"),
            assigned_to=data.get("assigned_to"),
            category=data.get("category"),
            subcategory=data.get("subcategory"),
            impact=data.get("impact"),
            urgency=data.get("urgency"),
            caller_id=data.get("caller_id"),
            created_at=(
                datetime.fromisoformat(data["sys_created_on"])
                if data.get("sys_created_on")
                else None
            ),
            updated_at=(
                datetime.fromisoformat(data["sys_updated_on"])
                if data.get("sys_updated_on")
                else None
            ),
            resolved_at=(
                datetime.fromisoformat(data["resolved_at"])
                if data.get("resolved_at")
                else None
            ),
            resolution_notes=data.get("close_notes"),
        )

    def _parse_change_request(self, data: Dict[str, Any]) -> ITSMChangeRequest:
        """Parse ServiceNow change request data into ITSMChangeRequest object"""
        # Reverse map status
        state_to_status = {
            "-5": ChangeRequestStatus.NEW,
            "-4": ChangeRequestStatus.ASSESS,
            "-3": ChangeRequestStatus.AUTHORIZE,
            "-2": ChangeRequestStatus.SCHEDULED,
            "-1": ChangeRequestStatus.IMPLEMENT,
            "0": ChangeRequestStatus.REVIEW,
            "3": ChangeRequestStatus.CLOSED,
            "4": ChangeRequestStatus.CANCELLED,
        }

        return ITSMChangeRequest(
            change_id=data.get("sys_id"),
            short_description=data.get("short_description", ""),
            description=data.get("description", ""),
            justification=data.get("justification", ""),
            risk_assessment=data.get("risk_assessment", ""),
            implementation_plan=data.get("implementation_plan", ""),
            backout_plan=data.get("backout_plan", ""),
            test_plan=data.get("test_plan", ""),
            status=state_to_status.get(data.get("state"), ChangeRequestStatus.NEW),
            type=data.get("type", "standard"),
            assignment_group=data.get("assignment_group"),
            assigned_to=data.get("assigned_to"),
            requested_by=data.get("requested_by"),
            scheduled_start=(
                datetime.fromisoformat(data["start_date"])
                if data.get("start_date")
                else None
            ),
            scheduled_end=(
                datetime.fromisoformat(data["end_date"])
                if data.get("end_date")
                else None
            ),
            actual_start=(
                datetime.fromisoformat(data["work_start"])
                if data.get("work_start")
                else None
            ),
            actual_end=(
                datetime.fromisoformat(data["work_end"])
                if data.get("work_end")
                else None
            ),
            approval_status=data.get("approval"),
        )


class JiraServiceManagementConnector(ITSMConnector):
    """Jira Service Management (formerly Jira Service Desk) connector implementation"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_token = config.get("api_token", "")
        self.username = config.get("username", "")
        self.project_key = config.get("project_key", "")
        self.service_desk_id = config.get("service_desk_id", "")

    def authenticate(self) -> bool:
        """Authenticate with Jira Service Management"""
        try:
            self._session = requests.Session()
            self._session.auth = (self.username, self.api_token)
            self._session.headers.update(
                {"Accept": "application/json", "Content-Type": "application/json"}
            )

            # Test connection
            test_url = f"{self.base_url}/rest/api/3/myself"
            response = self._session.get(test_url, verify=self.verify_ssl)
            response.raise_for_status()

            logger.info("Successfully authenticated with Jira Service Management")
            return True

        except Exception as e:
            logger.error(f"Failed to authenticate with Jira Service Management: {e}")
            return False

    def create_incident(self, incident: ITSMIncident) -> Tuple[bool, str]:
        """Create a new incident in Jira Service Management"""
        try:
            url = f"{self.base_url}/rest/api/3/issue"

            # Map to Jira issue fields
            data = {
                "fields": {
                    "project": {"key": self.project_key},
                    "issuetype": {"name": "Incident"},
                    "summary": incident.short_description,
                    "description": {
                        "type": "doc",
                        "version": 1,
                        "content": [
                            {
                                "type": "paragraph",
                                "content": [
                                    {"type": "text", "text": incident.description}
                                ],
                            }
                        ],
                    },
                    "priority": {"name": self._map_priority_to_jira(incident.priority)},
                    "labels": ["homeostasis", "auto-healing"],
                }
            }

            # Add custom fields if configured
            if incident.custom_fields:
                data["fields"].update(incident.custom_fields)

            # Add healing context as a comment after creation
            response = self._session.post(url, json=data, verify=self.verify_ssl)
            response.raise_for_status()

            result = response.json()
            issue_key = result["key"]
            issue_id = result["id"]

            # Add healing context as comment
            if incident.healing_context:
                self._add_comment(
                    issue_id,
                    f"Healing Context:\n```json\n{json.dumps(incident.healing_context, indent=2)}\n```",
                )

            logger.info(f"Created Jira incident: {issue_key}")
            return True, issue_key

        except Exception as e:
            logger.error(f"Failed to create Jira incident: {e}")
            return False, str(e)

    def update_incident(self, incident_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing incident in Jira Service Management"""
        try:
            url = f"{self.base_url}/rest/api/3/issue/{incident_id}"

            # Build update data
            data = {"fields": {}}

            if "short_description" in updates:
                data["fields"]["summary"] = updates["short_description"]

            if "description" in updates:
                data["fields"]["description"] = {
                    "type": "doc",
                    "version": 1,
                    "content": [
                        {
                            "type": "paragraph",
                            "content": [
                                {"type": "text", "text": updates["description"]}
                            ],
                        }
                    ],
                }

            if "priority" in updates:
                priority = updates["priority"]
                if isinstance(priority, IncidentPriority):
                    data["fields"]["priority"] = {
                        "name": self._map_priority_to_jira(priority)
                    }

            if "status" in updates:
                # Status transitions require separate API call
                self._transition_issue(incident_id, updates["status"])

            if data["fields"]:
                response = self._session.put(url, json=data, verify=self.verify_ssl)
                response.raise_for_status()

            logger.info(f"Updated Jira incident: {incident_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to update Jira incident: {e}")
            return False

    def get_incident(self, incident_id: str) -> Optional[ITSMIncident]:
        """Retrieve an incident from Jira Service Management"""
        try:
            url = f"{self.base_url}/rest/api/3/issue/{incident_id}"

            response = self._session.get(url, verify=self.verify_ssl)
            response.raise_for_status()

            data = response.json()

            return self._parse_jira_incident(data)

        except Exception as e:
            logger.error(f"Failed to retrieve Jira incident: {e}")
            return None

    def create_change_request(self, change: ITSMChangeRequest) -> Tuple[bool, str]:
        """Create a new change request in Jira Service Management"""
        try:
            url = f"{self.base_url}/rest/api/3/issue"

            # Combine all change details into description
            description_parts = [
                change.description,
                f"\n\n**Justification:**\n{change.justification}",
                f"\n\n**Risk Assessment:**\n{change.risk_assessment}",
                f"\n\n**Implementation Plan:**\n{change.implementation_plan}",
                f"\n\n**Backout Plan:**\n{change.backout_plan}",
                f"\n\n**Test Plan:**\n{change.test_plan}",
            ]

            data = {
                "fields": {
                    "project": {"key": self.project_key},
                    "issuetype": {"name": "Change"},
                    "summary": change.short_description,
                    "description": {
                        "type": "doc",
                        "version": 1,
                        "content": [
                            {
                                "type": "paragraph",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": "\n".join(description_parts),
                                    }
                                ],
                            }
                        ],
                    },
                    "labels": ["homeostasis", "change-request", change.type],
                }
            }

            # Add scheduled dates if available
            if change.scheduled_start:
                data["fields"]["duedate"] = change.scheduled_start.date().isoformat()

            # Add custom fields if configured
            if change.custom_fields:
                data["fields"].update(change.custom_fields)

            response = self._session.post(url, json=data, verify=self.verify_ssl)
            response.raise_for_status()

            result = response.json()
            issue_key = result["key"]
            issue_id = result["id"]

            # Add healing context as comment
            if change.healing_context:
                self._add_comment(
                    issue_id,
                    f"Healing Context:\n```json\n{json.dumps(change.healing_context, indent=2)}\n```",
                )

            logger.info(f"Created Jira change request: {issue_key}")
            return True, issue_key

        except Exception as e:
            logger.error(f"Failed to create Jira change request: {e}")
            return False, str(e)

    def update_change_request(self, change_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing change request in Jira Service Management"""
        # Reuse incident update logic as Jira treats them similarly
        return self.update_incident(change_id, updates)

    def get_change_request(self, change_id: str) -> Optional[ITSMChangeRequest]:
        """Retrieve a change request from Jira Service Management"""
        try:
            url = f"{self.base_url}/rest/api/3/issue/{change_id}"

            response = self._session.get(url, verify=self.verify_ssl)
            response.raise_for_status()

            data = response.json()

            return self._parse_jira_change_request(data)

        except Exception as e:
            logger.error(f"Failed to retrieve Jira change request: {e}")
            return None

    def search_incidents(self, query: Dict[str, Any]) -> List[ITSMIncident]:
        """Search for incidents in Jira Service Management"""
        try:
            url = f"{self.base_url}/rest/api/3/search"

            # Build JQL query
            jql_parts = [f'project = "{self.project_key}"', 'issuetype = "Incident"']

            if "status" in query:
                status = query["status"]
                if isinstance(status, IncidentStatus):
                    jira_status = self._map_status_to_jira(status)
                    jql_parts.append(f'status = "{jira_status}"')

            if "priority" in query:
                priority = query["priority"]
                if isinstance(priority, IncidentPriority):
                    jira_priority = self._map_priority_to_jira(priority)
                    jql_parts.append(f'priority = "{jira_priority}"')

            if "assignment_group" in query:
                jql_parts.append(f'assignee = "{query["assignment_group"]}"')

            jql = " AND ".join(jql_parts)

            params = {
                "jql": jql,
                "maxResults": query.get("limit", 100),
                "startAt": query.get("offset", 0),
            }

            response = self._session.get(url, params=params, verify=self.verify_ssl)
            response.raise_for_status()

            results = response.json()["issues"]

            return [self._parse_jira_incident(issue) for issue in results]

        except Exception as e:
            logger.error(f"Failed to search Jira incidents: {e}")
            return []

    def link_to_healing_action(
        self, ticket_id: str, healing_action_id: str, ticket_type: str = "incident"
    ) -> bool:
        """Link a Jira ticket to a Homeostasis healing action"""
        try:
            comment = f"Linked to Homeostasis healing action: {healing_action_id}"
            return self._add_comment(ticket_id, comment)

        except Exception as e:
            logger.error(f"Failed to link Jira ticket: {e}")
            return False

    def _add_comment(self, issue_id: str, comment_text: str) -> bool:
        """Add a comment to a Jira issue"""
        try:
            url = f"{self.base_url}/rest/api/3/issue/{issue_id}/comment"

            data = {
                "body": {
                    "type": "doc",
                    "version": 1,
                    "content": [
                        {
                            "type": "paragraph",
                            "content": [{"type": "text", "text": comment_text}],
                        }
                    ],
                }
            }

            response = self._session.post(url, json=data, verify=self.verify_ssl)
            response.raise_for_status()

            return True

        except Exception as e:
            logger.error(f"Failed to add comment to Jira issue: {e}")
            return False

    def _transition_issue(self, issue_id: str, new_status: IncidentStatus) -> bool:
        """Transition a Jira issue to a new status"""
        try:
            # Get available transitions
            url = f"{self.base_url}/rest/api/3/issue/{issue_id}/transitions"
            response = self._session.get(url, verify=self.verify_ssl)
            response.raise_for_status()

            transitions = response.json()["transitions"]
            target_status = self._map_status_to_jira(new_status)

            # Find matching transition
            transition_id = None
            for transition in transitions:
                if transition["to"]["name"].lower() == target_status.lower():
                    transition_id = transition["id"]
                    break

            if not transition_id:
                logger.warning(f"No transition found to status: {target_status}")
                return False

            # Execute transition
            data = {"transition": {"id": transition_id}}
            response = self._session.post(url, json=data, verify=self.verify_ssl)
            response.raise_for_status()

            return True

        except Exception as e:
            logger.error(f"Failed to transition Jira issue: {e}")
            return False

    def _map_priority_to_jira(self, priority: IncidentPriority) -> str:
        """Map generic priority to Jira priority names"""
        mapping = {
            IncidentPriority.CRITICAL: "Highest",
            IncidentPriority.HIGH: "High",
            IncidentPriority.MEDIUM: "Medium",
            IncidentPriority.LOW: "Low",
            IncidentPriority.PLANNING: "Lowest",
        }
        return mapping.get(priority, "Medium")

    def _map_status_to_jira(self, status: IncidentStatus) -> str:
        """Map generic status to Jira status names"""
        mapping = {
            IncidentStatus.NEW: "To Do",
            IncidentStatus.IN_PROGRESS: "In Progress",
            IncidentStatus.ON_HOLD: "On Hold",
            IncidentStatus.RESOLVED: "Resolved",
            IncidentStatus.CLOSED: "Done",
            IncidentStatus.CANCELLED: "Cancelled",
        }
        return mapping.get(status, "To Do")

    def _parse_jira_incident(self, data: Dict[str, Any]) -> ITSMIncident:
        """Parse Jira issue data into ITSMIncident object"""
        fields = data.get("fields", {})

        # Extract description text from Jira's document format
        description = ""
        if fields.get("description"):
            desc_content = fields["description"].get("content", [])
            for block in desc_content:
                if block.get("type") == "paragraph":
                    for item in block.get("content", []):
                        if item.get("type") == "text":
                            description += item.get("text", "")

        # Map Jira priority
        priority_map = {
            "Highest": IncidentPriority.CRITICAL,
            "High": IncidentPriority.HIGH,
            "Medium": IncidentPriority.MEDIUM,
            "Low": IncidentPriority.LOW,
            "Lowest": IncidentPriority.PLANNING,
        }

        # Map Jira status
        status_map = {
            "To Do": IncidentStatus.NEW,
            "In Progress": IncidentStatus.IN_PROGRESS,
            "On Hold": IncidentStatus.ON_HOLD,
            "Resolved": IncidentStatus.RESOLVED,
            "Done": IncidentStatus.CLOSED,
            "Cancelled": IncidentStatus.CANCELLED,
        }

        return ITSMIncident(
            incident_id=data.get("key"),
            short_description=fields.get("summary", ""),
            description=description,
            priority=priority_map.get(
                fields.get("priority", {}).get("name"), IncidentPriority.MEDIUM
            ),
            status=status_map.get(
                fields.get("status", {}).get("name"), IncidentStatus.NEW
            ),
            assigned_to=(
                fields.get("assignee", {}).get("displayName")
                if fields.get("assignee")
                else None
            ),
            created_at=(
                datetime.fromisoformat(fields["created"].replace("Z", "+00:00"))
                if fields.get("created")
                else None
            ),
            updated_at=(
                datetime.fromisoformat(fields["updated"].replace("Z", "+00:00"))
                if fields.get("updated")
                else None
            ),
        )

    def _parse_jira_change_request(self, data: Dict[str, Any]) -> ITSMChangeRequest:
        """Parse Jira issue data into ITSMChangeRequest object"""
        fields = data.get("fields", {})

        # Extract description and parse sections
        description_text = ""
        if fields.get("description"):
            desc_content = fields["description"].get("content", [])
            for block in desc_content:
                if block.get("type") == "paragraph":
                    for item in block.get("content", []):
                        if item.get("type") == "text":
                            description_text += item.get("text", "")

        # Parse sections from description
        sections = self._parse_change_sections(description_text)

        # Map Jira status to change status
        status_map = {
            "To Do": ChangeRequestStatus.NEW,
            "In Progress": ChangeRequestStatus.IMPLEMENT,
            "In Review": ChangeRequestStatus.REVIEW,
            "Done": ChangeRequestStatus.CLOSED,
            "Cancelled": ChangeRequestStatus.CANCELLED,
        }

        return ITSMChangeRequest(
            change_id=data.get("key"),
            short_description=fields.get("summary", ""),
            description=sections.get("description", ""),
            justification=sections.get("justification", ""),
            risk_assessment=sections.get("risk_assessment", ""),
            implementation_plan=sections.get("implementation_plan", ""),
            backout_plan=sections.get("backout_plan", ""),
            test_plan=sections.get("test_plan", ""),
            status=status_map.get(
                fields.get("status", {}).get("name"), ChangeRequestStatus.NEW
            ),
            assigned_to=(
                fields.get("assignee", {}).get("displayName")
                if fields.get("assignee")
                else None
            ),
            scheduled_start=(
                datetime.fromisoformat(fields["duedate"])
                if fields.get("duedate")
                else None
            ),
        )

    def _parse_change_sections(self, text: str) -> Dict[str, str]:
        """Parse change request sections from description text"""
        sections = {
            "description": "",
            "justification": "",
            "risk_assessment": "",
            "implementation_plan": "",
            "backout_plan": "",
            "test_plan": "",
        }

        # Simple parsing logic - can be enhanced
        current_section = "description"
        lines = text.split("\n")

        for line in lines:
            line_lower = line.lower().strip()
            if "**justification:**" in line_lower:
                current_section = "justification"
            elif "**risk assessment:**" in line_lower:
                current_section = "risk_assessment"
            elif "**implementation plan:**" in line_lower:
                current_section = "implementation_plan"
            elif "**backout plan:**" in line_lower:
                current_section = "backout_plan"
            elif "**test plan:**" in line_lower:
                current_section = "test_plan"
            else:
                sections[current_section] += line + "\n"

        # Clean up sections
        for key in sections:
            sections[key] = sections[key].strip()

        return sections


# Factory function to create ITSM connectors
def create_itsm_connector(
    provider: str, config: Dict[str, Any]
) -> Optional[ITSMConnector]:
    """Factory function to create ITSM connector instances"""
    providers = {
        "servicenow": ServiceNowConnector,
        "jira": JiraServiceManagementConnector,
        # Add more providers as implemented
    }

    connector_class = providers.get(provider.lower())
    if not connector_class:
        logger.error(f"Unknown ITSM provider: {provider}")
        return None

    try:
        connector = connector_class(config)
        if connector.authenticate():
            return connector
        else:
            logger.error(f"Failed to authenticate with {provider}")
            return None
    except Exception as e:
        logger.error(f"Failed to create {provider} connector: {e}")
        return None
