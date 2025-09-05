"""
Approval workflow system for Homeostasis.

Provides functionality for managing approval workflows for critical changes
in production environments.
"""

import datetime
import json
import logging
import os
import uuid
from enum import Enum
from typing import Dict, List, Optional, Union

from modules.security.audit import get_audit_logger
from modules.security.rbac import get_rbac_manager

logger = logging.getLogger(__name__)


class ApprovalStatus(Enum):
    """Enumeration of approval statuses."""
    PENDING = 'pending'
    APPROVED = 'approved'
    REJECTED = 'rejected'
    EXPIRED = 'expired'
    CANCELLED = 'cancelled'


class ApprovalType(Enum):
    """Enumeration of approval types."""
    FIX_DEPLOYMENT = 'fix_deployment'
    CONFIG_CHANGE = 'config_change'
    USER_ROLE_CHANGE = 'user_role_change'
    CUSTOM = 'custom'


class ApprovalError(Exception):
    """Exception raised for approval workflow errors."""
    pass


class ApprovalRequest:
    """Represents an approval request."""
    
    def __init__(self, request_id: str, request_type: Union[ApprovalType, str],
                 requester: str, title: str, description: str,
                 data: Dict, required_approvers: int = 1,
                 expiry: Optional[int] = None):
        """Initialize an approval request.
        
        Args:
            request_id: Unique identifier for the request
            request_type: Type of approval request
            requester: Username of the requester
            title: Request title
            description: Request description
            data: Request data (varies by type)
            required_approvers: Number of approvers required
            expiry: Expiry time in seconds
        """
        self.request_id = request_id
        self.request_type = request_type
        self.requester = requester
        self.title = title
        self.description = description
        self.data = data
        self.required_approvers = required_approvers
        
        # Set timestamps
        now = datetime.datetime.utcnow()
        self.created_at = now
        self.updated_at = now
        
        # Set expiry time if provided
        self.expiry = now + datetime.timedelta(seconds=expiry) if expiry else None
        
        # Initialize status and approvals
        self.status = ApprovalStatus.PENDING
        self.approvals = {}  # {username: {'timestamp': datetime, 'comment': str}}
        self.rejections = {}  # {username: {'timestamp': datetime, 'reason': str}}
        self.comments = []  # [{username: str, timestamp: datetime, comment: str}]
        
    def to_dict(self) -> Dict:
        """Convert to dictionary.
        
        Returns:
            Dict: Dictionary representation of the approval request
        """
        return {
            'request_id': self.request_id,
            'request_type': self.request_type.value if isinstance(self.request_type, ApprovalType) else self.request_type,
            'requester': self.requester,
            'title': self.title,
            'description': self.description,
            'data': self.data,
            'required_approvers': self.required_approvers,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'expiry': self.expiry.isoformat() if self.expiry else None,
            'status': self.status.value,
            'approvals': self.approvals,
            'rejections': self.rejections,
            'comments': self.comments
        }
        
    @classmethod
    def from_dict(cls, data: Dict) -> 'ApprovalRequest':
        """Create an approval request from a dictionary.
        
        Args:
            data: Dictionary representation of an approval request
            
        Returns:
            ApprovalRequest: Approval request instance
        """
        # Create instance
        request = cls(
            request_id=data['request_id'],
            request_type=data['request_type'],
            requester=data['requester'],
            title=data['title'],
            description=data['description'],
            data=data['data'],
            required_approvers=data['required_approvers']
        )
        
        # Set timestamps
        request.created_at = datetime.datetime.fromisoformat(data['created_at'])
        request.updated_at = datetime.datetime.fromisoformat(data['updated_at'])
        
        # Set expiry
        if data.get('expiry'):
            request.expiry = datetime.datetime.fromisoformat(data['expiry'])
        else:
            request.expiry = None
            
        # Set status
        request.status = ApprovalStatus(data['status'])
        
        # Set approvals, rejections, and comments
        request.approvals = data.get('approvals', {})
        request.rejections = data.get('rejections', {})
        request.comments = data.get('comments', [])
        
        return request
        
    def is_expired(self) -> bool:
        """Check if the request has expired.
        
        Returns:
            bool: True if expired, False otherwise
        """
        if self.expiry is None:
            return False
            
        return datetime.datetime.utcnow() > self.expiry
        
    def is_approved(self) -> bool:
        """Check if the request is fully approved.
        
        Returns:
            bool: True if approved, False otherwise
        """
        return (len(self.approvals) >= self.required_approvers or
                self.status == ApprovalStatus.APPROVED)
                
    def is_rejected(self) -> bool:
        """Check if the request is rejected.
        
        Returns:
            bool: True if rejected, False otherwise
        """
        return (len(self.rejections) > 0 or
                self.status == ApprovalStatus.REJECTED)
                
    def update_status(self) -> None:
        """Update the request status based on approvals and rejections."""
        # Check for expiry
        if self.is_expired() and self.status == ApprovalStatus.PENDING:
            self.status = ApprovalStatus.EXPIRED
            self.updated_at = datetime.datetime.utcnow()
            return
            
        # Check for rejection
        if self.is_rejected() and self.status == ApprovalStatus.PENDING:
            self.status = ApprovalStatus.REJECTED
            self.updated_at = datetime.datetime.utcnow()
            return
            
        # Check for approval
        if self.is_approved() and self.status == ApprovalStatus.PENDING:
            self.status = ApprovalStatus.APPROVED
            self.updated_at = datetime.datetime.utcnow()
            return
            
    def approve(self, username: str, comment: str = None) -> bool:
        """Approve the request.
        
        Args:
            username: Username of the approver
            comment: Optional comment
            
        Returns:
            bool: True if approved, False otherwise
            
        Raises:
            ApprovalError: If request is not pending
        """
        # Check if request can be approved
        if self.status != ApprovalStatus.PENDING:
            raise ApprovalError(f"Cannot approve request with status: {self.status.value}")
            
        if self.is_expired():
            self.status = ApprovalStatus.EXPIRED
            raise ApprovalError("Cannot approve expired request")
            
        # Add approval
        now = datetime.datetime.utcnow()
        self.approvals[username] = {
            'timestamp': now.isoformat(),
            'comment': comment
        }
        self.updated_at = now
        
        # Add comment if provided
        if comment:
            self.add_comment(username, comment)
            
        # Update status
        self.update_status()
        
        return self.is_approved()
        
    def reject(self, username: str, reason: str = None) -> bool:
        """Reject the request.
        
        Args:
            username: Username of the rejector
            reason: Optional rejection reason
            
        Returns:
            bool: True if rejected, False otherwise
            
        Raises:
            ApprovalError: If request is not pending
        """
        # Check if request can be rejected
        if self.status != ApprovalStatus.PENDING:
            raise ApprovalError(f"Cannot reject request with status: {self.status.value}")
            
        if self.is_expired():
            self.status = ApprovalStatus.EXPIRED
            raise ApprovalError("Cannot reject expired request")
            
        # Add rejection
        now = datetime.datetime.utcnow()
        self.rejections[username] = {
            'timestamp': now.isoformat(),
            'reason': reason
        }
        self.updated_at = now
        
        # Add comment if reason provided
        if reason:
            self.add_comment(username, f"Rejection reason: {reason}")
            
        # Update status
        self.update_status()
        
        return True
        
    def cancel(self, username: str, reason: str = None) -> bool:
        """Cancel the request.
        
        Args:
            username: Username of the canceller
            reason: Optional cancellation reason
            
        Returns:
            bool: True if cancelled, False otherwise
            
        Raises:
            ApprovalError: If request is not pending
        """
        # Check if request can be cancelled
        if self.status != ApprovalStatus.PENDING:
            raise ApprovalError(f"Cannot cancel request with status: {self.status.value}")
            
        # Set status to cancelled
        self.status = ApprovalStatus.CANCELLED
        now = datetime.datetime.utcnow()
        self.updated_at = now
        
        # Add comment if reason provided
        if reason:
            self.add_comment(username, f"Cancellation reason: {reason}")
            
        return True
        
    def add_comment(self, username: str, comment: str) -> None:
        """Add a comment to the request.
        
        Args:
            username: Username of the commenter
            comment: Comment text
        """
        self.comments.append({
            'username': username,
            'timestamp': datetime.datetime.utcnow().isoformat(),
            'comment': comment
        })


class ApprovalManager:
    """Manages approval workflows for critical changes."""
    
    def __init__(self, config: Dict = None, storage_path: str = None):
        """Initialize the approval manager.
        
        Args:
            config: Configuration dictionary for approval settings
            storage_path: Path to store approval requests
        """
        self.config = config or {}
        
        # Configure storage
        self.storage_path = storage_path or 'logs/approvals.json'
        
        # Load existing approval requests
        self.requests = {}
        self._load_requests()
        
        # Set up required approvers
        self.default_required_approvers = self.config.get('required_approvers', 1)
        
        # Set up default expiry
        self.default_expiry = self.config.get('approval_timeout', 86400)  # 24 hours
        
        # Set up critical fix types
        self.critical_fix_types = set(self.config.get('critical_fix_types', [
            'database_schema',
            'security',
            'authentication',
            'authorization'
        ]))
        
    def _load_requests(self) -> None:
        """Load approval requests from storage."""
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                
            for request_data in data:
                request = ApprovalRequest.from_dict(request_data)
                self.requests[request.request_id] = request
                
            logger.info(f"Loaded {len(self.requests)} approval requests from {self.storage_path}")
        except FileNotFoundError:
            logger.info(f"No approval requests found at {self.storage_path}")
        except Exception as e:
            logger.error(f"Error loading approval requests: {str(e)}")
            
    def _save_requests(self) -> None:
        """Save approval requests to storage."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            
            with open(self.storage_path, 'w') as f:
                json.dump([r.to_dict() for r in self.requests.values()], f, indent=2)
                
            logger.info(f"Saved {len(self.requests)} approval requests to {self.storage_path}")
        except Exception as e:
            logger.error(f"Error saving approval requests: {str(e)}")
            
    def create_request(self, request_type: Union[ApprovalType, str],
                       requester: str, title: str, description: str,
                       data: Dict, required_approvers: int = None,
                       expiry: int = None) -> ApprovalRequest:
        """Create a new approval request.
        
        Args:
            request_type: Type of approval request
            requester: Username of the requester
            title: Request title
            description: Request description
            data: Request data (varies by type)
            required_approvers: Number of approvers required (defaults to config)
            expiry: Expiry time in seconds (defaults to config)
            
        Returns:
            ApprovalRequest: The created approval request
        """
        # Generate request ID
        request_id = str(uuid.uuid4())
        
        # Use default values if not provided
        if required_approvers is None:
            required_approvers = self.default_required_approvers
            
        if expiry is None:
            expiry = self.default_expiry
            
        # Create approval request
        request = ApprovalRequest(
            request_id=request_id,
            request_type=request_type,
            requester=requester,
            title=title,
            description=description,
            data=data,
            required_approvers=required_approvers,
            expiry=expiry
        )
        
        # Add to requests and save
        self.requests[request_id] = request
        self._save_requests()
        
        # Log creation
        get_audit_logger().log_event(
            event_type='approval_request_created',
            user=requester,
            details={
                'request_id': request_id,
                'request_type': request_type.value if isinstance(request_type, ApprovalType) else request_type,
                'title': title,
                'required_approvers': required_approvers
            }
        )
        
        return request
        
    def get_request(self, request_id: str) -> Optional[ApprovalRequest]:
        """Get an approval request by ID.
        
        Args:
            request_id: Request ID
            
        Returns:
            Optional[ApprovalRequest]: Approval request if found, None otherwise
        """
        return self.requests.get(request_id)
        
    def list_requests(self, status: Optional[ApprovalStatus] = None,
                      request_type: Optional[Union[ApprovalType, str]] = None,
                      requester: Optional[str] = None) -> List[ApprovalRequest]:
        """List approval requests filtered by criteria.
        
        Args:
            status: Filter by status
            request_type: Filter by request type
            requester: Filter by requester
            
        Returns:
            List[ApprovalRequest]: List of matching approval requests
        """
        results = []
        
        for request in self.requests.values():
            # Filter by status if provided
            if status and request.status != status:
                continue
                
            # Filter by request type if provided
            if request_type:
                if isinstance(request_type, ApprovalType) and isinstance(request.request_type, ApprovalType):
                    if request.request_type != request_type:
                        continue
                elif request.request_type != request_type:
                    continue
                    
            # Filter by requester if provided
            if requester and request.requester != requester:
                continue
                
            results.append(request)
            
        return results
        
    def approve_request(self, request_id: str, username: str, comment: str = None) -> bool:
        """Approve an approval request.
        
        Args:
            request_id: Request ID
            username: Username of the approver
            comment: Optional comment
            
        Returns:
            bool: True if approved, False otherwise
            
        Raises:
            ApprovalError: If request not found or cannot be approved
        """
        # Get request
        request = self.get_request(request_id)
        if not request:
            raise ApprovalError(f"Approval request not found: {request_id}")
            
        # Check if user has approval permission
        rbac = get_rbac_manager()
        if not rbac.has_permission({'username': username, 'roles': ['admin', 'operator']}, 'approve_fixes'):
            raise ApprovalError(f"User {username} does not have permission to approve requests")
            
        # Check if user is the requester
        if request.requester == username:
            raise ApprovalError("Cannot approve your own request")
            
        # Approve request
        result = request.approve(username, comment)
        
        # Save changes
        self._save_requests()
        
        # Log approval
        get_audit_logger().log_event(
            event_type='approval_request_approved',
            user=username,
            details={
                'request_id': request_id,
                'request_type': request.request_type.value if isinstance(request.request_type, ApprovalType) else request.request_type,
                'title': request.title,
                'is_fully_approved': result
            }
        )
        
        return result
        
    def reject_request(self, request_id: str, username: str, reason: str = None) -> bool:
        """Reject an approval request.
        
        Args:
            request_id: Request ID
            username: Username of the rejector
            reason: Optional rejection reason
            
        Returns:
            bool: True if rejected, False otherwise
            
        Raises:
            ApprovalError: If request not found or cannot be rejected
        """
        # Get request
        request = self.get_request(request_id)
        if not request:
            raise ApprovalError(f"Approval request not found: {request_id}")
            
        # Check if user has rejection permission
        rbac = get_rbac_manager()
        if not rbac.has_permission({'username': username, 'roles': ['admin', 'operator']}, 'reject_fixes'):
            raise ApprovalError(f"User {username} does not have permission to reject requests")
            
        # Reject request
        result = request.reject(username, reason)
        
        # Save changes
        self._save_requests()
        
        # Log rejection
        get_audit_logger().log_event(
            event_type='approval_request_rejected',
            user=username,
            details={
                'request_id': request_id,
                'request_type': request.request_type.value if isinstance(request.request_type, ApprovalType) else request.request_type,
                'title': request.title,
                'reason': reason
            }
        )
        
        return result
        
    def cancel_request(self, request_id: str, username: str, reason: str = None) -> bool:
        """Cancel an approval request.
        
        Args:
            request_id: Request ID
            username: Username of the canceller
            reason: Optional cancellation reason
            
        Returns:
            bool: True if cancelled, False otherwise
            
        Raises:
            ApprovalError: If request not found or cannot be cancelled
        """
        # Get request
        request = self.get_request(request_id)
        if not request:
            raise ApprovalError(f"Approval request not found: {request_id}")
            
        # Check if user is the requester or has admin permission
        rbac = get_rbac_manager()
        is_admin = rbac.has_permission({'username': username, 'roles': ['admin']}, 'manage_config')
        if request.requester != username and not is_admin:
            raise ApprovalError(f"User {username} cannot cancel this request")
            
        # Cancel request
        result = request.cancel(username, reason)
        
        # Save changes
        self._save_requests()
        
        # Log cancellation
        get_audit_logger().log_event(
            event_type='approval_request_cancelled',
            user=username,
            details={
                'request_id': request_id,
                'request_type': request.request_type.value if isinstance(request.request_type, ApprovalType) else request.request_type,
                'title': request.title,
                'reason': reason
            }
        )
        
        return result
        
    def add_comment(self, request_id: str, username: str, comment: str) -> bool:
        """Add a comment to an approval request.
        
        Args:
            request_id: Request ID
            username: Username of the commenter
            comment: Comment text
            
        Returns:
            bool: True if comment added, False otherwise
            
        Raises:
            ApprovalError: If request not found
        """
        # Get request
        request = self.get_request(request_id)
        if not request:
            raise ApprovalError(f"Approval request not found: {request_id}")
            
        # Add comment
        request.add_comment(username, comment)
        
        # Save changes
        self._save_requests()
        
        return True
        
    def needs_approval(self, fix_type: str) -> bool:
        """Check if a fix type requires approval.
        
        Args:
            fix_type: Type of fix
            
        Returns:
            bool: True if approval is required, False otherwise
        """
        return fix_type in self.critical_fix_types
        
    def cleanup_expired_requests(self) -> int:
        """Update status of expired requests.
        
        Returns:
            int: Number of expired requests
        """
        count = 0
        
        for request in self.requests.values():
            if request.status == ApprovalStatus.PENDING and request.is_expired():
                request.status = ApprovalStatus.EXPIRED
                request.updated_at = datetime.datetime.utcnow()
                count += 1
                
                # Log expiry
                get_audit_logger().log_event(
                    event_type='approval_request_expired',
                    user=request.requester,
                    details={
                        'request_id': request.request_id,
                        'request_type': request.request_type.value if isinstance(request.request_type, ApprovalType) else request.request_type,
                        'title': request.title
                    }
                )
                
        if count > 0:
            self._save_requests()
            
        return count


# Singleton instance for app-wide use
_approval_manager = None


def get_approval_manager(config: Dict = None) -> ApprovalManager:
    """Get or create the singleton ApprovalManager instance.
    
    Args:
        config: Optional configuration to initialize the manager with
        
    Returns:
        ApprovalManager: The approval manager instance
    """
    global _approval_manager
    if _approval_manager is None:
        _approval_manager = ApprovalManager(config)
    return _approval_manager


def create_approval_request(request_type: Union[ApprovalType, str],
                           requester: str, title: str, description: str,
                           data: Dict, required_approvers: int = None,
                           expiry: int = None) -> ApprovalRequest:
    """Create a new approval request.
    
    Args:
        request_type: Type of approval request
        requester: Username of the requester
        title: Request title
        description: Request description
        data: Request data (varies by type)
        required_approvers: Number of approvers required
        expiry: Expiry time in seconds
        
    Returns:
        ApprovalRequest: The created approval request
    """
    return get_approval_manager().create_request(
        request_type=request_type,
        requester=requester,
        title=title,
        description=description,
        data=data,
        required_approvers=required_approvers,
        expiry=expiry
    )


def needs_approval(fix_type: str) -> bool:
    """Check if a fix type requires approval.
    
    Args:
        fix_type: Type of fix
        
    Returns:
        bool: True if approval is required, False otherwise
    """
    return get_approval_manager().needs_approval(fix_type)