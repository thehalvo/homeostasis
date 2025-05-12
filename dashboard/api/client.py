"""
Homeostasis API Client

Provides a client for communicating with the Homeostasis system.
"""

import json
import logging
import os
import requests
import time
from typing import Dict, List, Optional, Any, Tuple, Union

from dashboard.api.errors import APIError, ConnectionError, NotFoundError, AuthenticationError

logger = logging.getLogger(__name__)


class HomeostasisClient:
    """Client for interacting with the Homeostasis system API."""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 8000, 
                timeout: int = 30, use_ssl: bool = False):
        """Initialize the Homeostasis client.
        
        Args:
            host: Homeostasis host address
            port: Homeostasis port
            timeout: Request timeout in seconds
            use_ssl: Whether to use HTTPS
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        self.use_ssl = use_ssl
        self.base_url = f"{'https' if use_ssl else 'http'}://{host}:{port}"
        self.token = None
        
        logger.info(f"Initialized Homeostasis client with base URL: {self.base_url}")
        
    def authenticate(self, username: str, password: str) -> bool:
        """Authenticate with the Homeostasis system.
        
        Args:
            username: Username
            password: Password
            
        Returns:
            bool: True if authentication succeeded
            
        Raises:
            AuthenticationError: If authentication fails
        """
        try:
            response = requests.post(
                f"{self.base_url}/api/auth",
                json={"username": username, "password": password},
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                self.token = data.get("token")
                return True
            elif response.status_code == 401:
                raise AuthenticationError("Invalid username or password")
            else:
                raise APIError(f"Authentication failed: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Connection error: {str(e)}")
            
    def get_errors(self, limit: int = 100, offset: int = 0, 
                  status: Optional[str] = None) -> List[Dict]:
        """Get errors from the Homeostasis system.
        
        Args:
            limit: Maximum number of errors to return
            offset: Offset for pagination
            status: Filter by error status
            
        Returns:
            List[Dict]: List of errors
            
        Raises:
            APIError: If API request fails
        """
        params = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
            
        return self._get_api_resource("/api/errors", params)
        
    def get_fixes(self, limit: int = 100, offset: int = 0,
                 status: Optional[str] = None) -> List[Dict]:
        """Get fixes from the Homeostasis system.
        
        Args:
            limit: Maximum number of fixes to return
            offset: Offset for pagination
            status: Filter by fix status
            
        Returns:
            List[Dict]: List of fixes
            
        Raises:
            APIError: If API request fails
        """
        params = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
            
        return self._get_api_resource("/api/fixes", params)
        
    def get_approvals(self, limit: int = 100, offset: int = 0,
                     status: Optional[str] = None) -> List[Dict]:
        """Get approval requests from the Homeostasis system.
        
        Args:
            limit: Maximum number of approvals to return
            offset: Offset for pagination
            status: Filter by approval status
            
        Returns:
            List[Dict]: List of approval requests
            
        Raises:
            APIError: If API request fails
        """
        params = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
            
        return self._get_api_resource("/api/approvals", params)
        
    def get_metrics(self) -> Dict:
        """Get system metrics from the Homeostasis system.
        
        Returns:
            Dict: System metrics
            
        Raises:
            APIError: If API request fails
        """
        return self._get_api_resource("/api/metrics")
        
    def get_canary(self, service_name: Optional[str] = None,
                  fix_id: Optional[str] = None) -> Dict:
        """Get canary deployment status.
        
        Args:
            service_name: Filter by service name
            fix_id: Filter by fix ID
            
        Returns:
            Dict: Canary deployment status
            
        Raises:
            APIError: If API request fails
        """
        params = {}
        if service_name:
            params["service_name"] = service_name
        if fix_id:
            params["fix_id"] = fix_id
            
        return self._get_api_resource("/api/canary", params)
        
    def approve_fix(self, approval_id: str, comment: Optional[str] = None) -> bool:
        """Approve a fix.
        
        Args:
            approval_id: ID of the approval request
            comment: Optional comment
            
        Returns:
            bool: True if approval succeeded
            
        Raises:
            APIError: If API request fails
        """
        data = {"comment": comment} if comment else {}
        
        try:
            response = self._post_api_resource(f"/api/approvals/{approval_id}/approve", data)
            return response.get("success", False)
        except APIError:
            return False
            
    def reject_fix(self, approval_id: str, reason: Optional[str] = None) -> bool:
        """Reject a fix.
        
        Args:
            approval_id: ID of the approval request
            reason: Optional rejection reason
            
        Returns:
            bool: True if rejection succeeded
            
        Raises:
            APIError: If API request fails
        """
        data = {"reason": reason} if reason else {}
        
        try:
            response = self._post_api_resource(f"/api/approvals/{approval_id}/reject", data)
            return response.get("success", False)
        except APIError:
            return False
            
    def promote_canary(self, service_name: str, fix_id: str) -> bool:
        """Promote a canary deployment.
        
        Args:
            service_name: Service name
            fix_id: Fix ID
            
        Returns:
            bool: True if promotion succeeded
            
        Raises:
            APIError: If API request fails
        """
        data = {"service_name": service_name, "fix_id": fix_id}
        
        try:
            response = self._post_api_resource("/api/canary/promote", data)
            return response.get("success", False)
        except APIError:
            return False
            
    def complete_canary(self, service_name: str, fix_id: str) -> bool:
        """Complete a canary deployment.
        
        Args:
            service_name: Service name
            fix_id: Fix ID
            
        Returns:
            bool: True if completion succeeded
            
        Raises:
            APIError: If API request fails
        """
        data = {"service_name": service_name, "fix_id": fix_id}
        
        try:
            response = self._post_api_resource("/api/canary/complete", data)
            return response.get("success", False)
        except APIError:
            return False
            
    def rollback_canary(self, service_name: str, fix_id: str) -> bool:
        """Roll back a canary deployment.
        
        Args:
            service_name: Service name
            fix_id: Fix ID
            
        Returns:
            bool: True if rollback succeeded
            
        Raises:
            APIError: If API request fails
        """
        data = {"service_name": service_name, "fix_id": fix_id}
        
        try:
            response = self._post_api_resource("/api/canary/rollback", data)
            return response.get("success", False)
        except APIError:
            return False
            
    def _get_api_resource(self, endpoint: str, params: Dict = None) -> Any:
        """Get a resource from the API.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            
        Returns:
            API response data
            
        Raises:
            APIError: If API request fails
        """
        headers = self._get_headers()
        
        try:
            response = requests.get(
                f"{self.base_url}{endpoint}",
                params=params,
                headers=headers,
                timeout=self.timeout
            )
            
            return self._process_response(response)
            
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Connection error: {str(e)}")
            
    def _post_api_resource(self, endpoint: str, data: Dict = None) -> Any:
        """Post to an API endpoint.
        
        Args:
            endpoint: API endpoint
            data: Request data
            
        Returns:
            API response data
            
        Raises:
            APIError: If API request fails
        """
        headers = self._get_headers()
        
        try:
            response = requests.post(
                f"{self.base_url}{endpoint}",
                json=data,
                headers=headers,
                timeout=self.timeout
            )
            
            return self._process_response(response)
            
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Connection error: {str(e)}")
            
    def _process_response(self, response: requests.Response) -> Any:
        """Process an API response.
        
        Args:
            response: API response
            
        Returns:
            API response data
            
        Raises:
            APIError: If API request fails
        """
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            raise NotFoundError("Resource not found")
        elif response.status_code == 401:
            raise AuthenticationError("Authentication required")
        else:
            # Try to extract error message from response
            try:
                error_data = response.json()
                error_message = error_data.get("error", f"API error: {response.status_code}")
            except (ValueError, KeyError):
                error_message = f"API error: {response.status_code}"
                
            raise APIError(error_message)
            
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers.
        
        Returns:
            Dict[str, str]: Request headers
        """
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
            
        return headers