"""
Identity Provider Integration for Homeostasis Enterprise Governance.

This module provides integration with external identity providers for
single sign-on (SSO) and centralized authentication in enterprise environments.

Supported providers:
- OAuth 2.0
- SAML 2.0
- LDAP/Active Directory
- OpenID Connect
"""

import datetime
import json
import logging
import ldap
import jwt
import requests
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
from urllib.parse import urlencode, urlparse
import base64
import hashlib
import secrets

from .auth import get_auth_manager
from .user_management import get_user_management
from .audit import get_audit_logger

logger = logging.getLogger(__name__)


class IdentityProviderType(Enum):
    """Types of identity providers."""
    OAUTH2 = "oauth2"
    SAML2 = "saml2"
    LDAP = "ldap"
    OIDC = "oidc"
    CUSTOM = "custom"


class AuthenticationMethod(Enum):
    """Authentication methods."""
    PASSWORD = "password"
    SSO = "sso"
    MFA = "mfa"
    CERTIFICATE = "certificate"


@dataclass
class IdentityProviderConfig:
    """Configuration for an identity provider."""
    provider_id: str
    name: str
    type: IdentityProviderType
    enabled: bool = True
    priority: int = 1
    config: Dict = field(default_factory=dict)
    attribute_mapping: Dict = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)


@dataclass
class AuthenticationSession:
    """SSO authentication session."""
    session_id: str
    provider_id: str
    state: str
    nonce: Optional[str] = None
    redirect_uri: str = ""
    created_at: datetime.datetime = field(default_factory=datetime.datetime.utcnow)
    expires_at: Optional[datetime.datetime] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class UserIdentity:
    """User identity from external provider."""
    provider_id: str
    external_id: str
    username: str
    email: str
    full_name: str
    groups: List[str] = field(default_factory=list)
    roles: List[str] = field(default_factory=list)
    attributes: Dict = field(default_factory=dict)
    last_authenticated: datetime.datetime = field(default_factory=datetime.datetime.utcnow)


class IdentityProviderIntegration:
    """
    Manages integration with external identity providers for SSO.
    
    Supports OAuth 2.0, SAML 2.0, LDAP, and OpenID Connect.
    """
    
    def __init__(self, config: Dict = None, storage_path: str = None):
        """Initialize the identity provider integration.
        
        Args:
            config: Configuration dictionary
            storage_path: Path to store provider data
        """
        self.config = config or {}
        self.storage_path = Path(storage_path or "data/identity")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Get managers
        self.auth_manager = get_auth_manager(config)
        self.user_management = get_user_management(config)
        self.audit_logger = get_audit_logger(config)
        
        # Initialize stores
        self.providers: Dict[str, IdentityProviderConfig] = {}
        self.sessions: Dict[str, AuthenticationSession] = {}
        self.user_identities: Dict[str, UserIdentity] = {}
        
        # Provider handlers
        self.provider_handlers = {
            IdentityProviderType.OAUTH2: OAuth2Handler(self),
            IdentityProviderType.SAML2: SAML2Handler(self),
            IdentityProviderType.LDAP: LDAPHandler(self),
            IdentityProviderType.OIDC: OIDCHandler(self)
        }
        
        # Load existing data
        self._load_provider_data()
        
        # Initialize default providers if configured
        self._initialize_default_providers()
    
    def configure_provider(self, name: str, type: IdentityProviderType,
                         config: Dict, attribute_mapping: Dict = None) -> str:
        """Configure an identity provider.
        
        Args:
            name: Provider name
            type: Provider type
            config: Provider-specific configuration
            attribute_mapping: Attribute mapping configuration
            
        Returns:
            Provider ID
        """
        provider_id = self._generate_provider_id()
        
        provider = IdentityProviderConfig(
            provider_id=provider_id,
            name=name,
            type=type,
            config=config,
            attribute_mapping=attribute_mapping or self._get_default_attribute_mapping(type)
        )
        
        # Validate configuration
        handler = self.provider_handlers.get(type)
        if handler:
            handler.validate_config(config)
        
        self.providers[provider_id] = provider
        self._save_provider_data()
        
        # Log configuration
        self.audit_logger.log_event(
            event_type='identity_provider_configured',
            user='system',
            details={
                'provider_id': provider_id,
                'name': name,
                'type': type.value
            }
        )
        
        return provider_id
    
    def initiate_authentication(self, provider_id: str, redirect_uri: str,
                              scope: List[str] = None) -> Dict[str, str]:
        """Initiate authentication with a provider.
        
        Args:
            provider_id: Provider ID
            redirect_uri: Redirect URI after authentication
            scope: Requested scopes
            
        Returns:
            Authentication initiation response
        """
        provider = self.providers.get(provider_id)
        if not provider:
            raise ValueError(f"Provider {provider_id} not found")
        
        if not provider.enabled:
            raise ValueError(f"Provider {provider_id} is disabled")
        
        handler = self.provider_handlers.get(provider.type)
        if not handler:
            raise ValueError(f"No handler for provider type {provider.type.value}")
        
        # Create authentication session
        session = AuthenticationSession(
            session_id=self._generate_session_id(),
            provider_id=provider_id,
            state=self._generate_state(),
            redirect_uri=redirect_uri,
            expires_at=datetime.datetime.utcnow() + datetime.timedelta(minutes=15)
        )
        
        self.sessions[session.session_id] = session
        
        # Initiate authentication
        auth_url, additional_params = handler.initiate_auth(provider, session, scope)
        
        return {
            'auth_url': auth_url,
            'session_id': session.session_id,
            'state': session.state,
            **additional_params
        }
    
    def complete_authentication(self, session_id: str, callback_data: Dict) -> Dict[str, Any]:
        """Complete authentication after provider callback.
        
        Args:
            session_id: Authentication session ID
            callback_data: Callback data from provider
            
        Returns:
            Authentication result
        """
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        # Check session expiration
        if session.expires_at and datetime.datetime.utcnow() > session.expires_at:
            del self.sessions[session_id]
            raise ValueError("Authentication session expired")
        
        provider = self.providers.get(session.provider_id)
        if not provider:
            raise ValueError(f"Provider {session.provider_id} not found")
        
        handler = self.provider_handlers.get(provider.type)
        if not handler:
            raise ValueError(f"No handler for provider type {provider.type.value}")
        
        # Complete authentication
        user_identity = handler.complete_auth(provider, session, callback_data)
        
        # Map user identity to local user
        local_user = self._map_user_identity(user_identity, provider)
        
        # Create local session
        auth_result = self.user_management.authenticate_user(
            local_user['username'],
            'sso_authenticated',  # Special marker for SSO auth
            ip_address=callback_data.get('ip_address'),
            user_agent=callback_data.get('user_agent')
        )
        
        # Clean up authentication session
        del self.sessions[session_id]
        
        # Log successful authentication
        self.audit_logger.log_login(
            username=local_user['username'],
            status='success',
            source_ip=callback_data.get('ip_address'),
            details={
                'provider': provider.name,
                'external_id': user_identity.external_id,
                'method': 'sso'
            }
        )
        
        return {
            'user_id': local_user['user_id'],
            'username': local_user['username'],
            'roles': local_user['roles'],
            'provider': provider.name,
            **auth_result
        }
    
    def authenticate_ldap(self, username: str, password: str,
                         provider_id: str = None) -> Optional[Dict[str, Any]]:
        """Authenticate using LDAP.
        
        Args:
            username: Username
            password: Password
            provider_id: Specific LDAP provider ID (optional)
            
        Returns:
            Authentication result if successful
        """
        # Find LDAP provider
        ldap_providers = [p for p in self.providers.values() 
                         if p.type == IdentityProviderType.LDAP and p.enabled]
        
        if provider_id:
            ldap_providers = [p for p in ldap_providers if p.provider_id == provider_id]
        
        if not ldap_providers:
            raise ValueError("No LDAP providers configured")
        
        # Try each LDAP provider
        for provider in sorted(ldap_providers, key=lambda p: p.priority):
            handler = self.provider_handlers[IdentityProviderType.LDAP]
            
            try:
                user_identity = handler.authenticate(provider, username, password)
                if user_identity:
                    # Map user identity to local user
                    local_user = self._map_user_identity(user_identity, provider)
                    
                    # Create local session
                    auth_result = self.user_management.authenticate_user(
                        local_user['username'],
                        'ldap_authenticated'
                    )
                    
                    # Log successful authentication
                    self.audit_logger.log_login(
                        username=local_user['username'],
                        status='success',
                        details={
                            'provider': provider.name,
                            'method': 'ldap'
                        }
                    )
                    
                    return {
                        'user_id': local_user['user_id'],
                        'username': local_user['username'],
                        'roles': local_user['roles'],
                        'provider': provider.name,
                        **auth_result
                    }
            except Exception as e:
                logger.warning(f"LDAP authentication failed with provider {provider.name}: {str(e)}")
                continue
        
        # Log failed authentication
        self.audit_logger.log_login(
            username=username,
            status='failure',
            details={'method': 'ldap'}
        )
        
        return None
    
    def get_user_identities(self, user_id: str) -> List[UserIdentity]:
        """Get all external identities for a user.
        
        Args:
            user_id: Local user ID
            
        Returns:
            List of user identities
        """
        identities = []
        
        for identity in self.user_identities.values():
            # Check if identity maps to this user
            # In a real implementation, this would use a proper mapping table
            if identity.email and identity.email == self.user_management.users.get(user_id, {}).get('email'):
                identities.append(identity)
        
        return identities
    
    def sync_user_groups(self, user_id: str) -> bool:
        """Sync user groups from identity providers.
        
        Args:
            user_id: Local user ID
            
        Returns:
            True if successful
        """
        identities = self.get_user_identities(user_id)
        
        if not identities:
            return False
        
        # Aggregate groups from all identities
        all_groups = set()
        for identity in identities:
            all_groups.update(identity.groups)
        
        # Map external groups to local groups
        local_groups = self._map_external_groups(list(all_groups))
        
        # Update user groups
        user = self.user_management.users.get(user_id)
        if user:
            user['groups'] = local_groups
            self.user_management._save_user_data()
        
        return True
    
    def _map_user_identity(self, identity: UserIdentity, provider: IdentityProviderConfig) -> Dict:
        """Map external identity to local user.
        
        Args:
            identity: External user identity
            provider: Identity provider
            
        Returns:
            Local user info
        """
        # Check if user already exists
        existing_user = None
        for user_id, user in self.user_management.users.items():
            if user['email'] == identity.email:
                existing_user = user
                break
        
        if existing_user:
            # Update existing user
            user_id = existing_user['user_id']
            
            # Update groups if configured
            if provider.config.get('sync_groups', True):
                mapped_groups = self._map_external_groups(identity.groups)
                existing_user['groups'] = mapped_groups
            
            # Update roles if configured
            if provider.config.get('sync_roles', True):
                mapped_roles = self._map_external_roles(identity.roles, provider)
                existing_user['roles'] = mapped_roles
            
            self.user_management._save_user_data()
            
            return existing_user
        else:
            # Create new user
            mapped_roles = self._map_external_roles(identity.roles, provider)
            mapped_groups = self._map_external_groups(identity.groups)
            
            user_id = self.user_management.create_user(
                username=identity.username,
                email=identity.email,
                password=secrets.token_urlsafe(32),  # Random password for SSO users
                full_name=identity.full_name,
                roles=mapped_roles,
                groups=mapped_groups,
                auth_provider=provider.provider_id
            )
            
            return self.user_management.users[user_id]
    
    def _map_external_roles(self, external_roles: List[str], 
                           provider: IdentityProviderConfig) -> List[str]:
        """Map external roles to local roles.
        
        Args:
            external_roles: External role names
            provider: Identity provider
            
        Returns:
            Local role names
        """
        role_mapping = provider.attribute_mapping.get('roles', {})
        
        local_roles = []
        for external_role in external_roles:
            # Check explicit mapping
            if external_role in role_mapping:
                local_roles.append(role_mapping[external_role])
            # Check pattern matching
            elif '*' in role_mapping:
                # Default mapping
                local_roles.append(role_mapping['*'])
        
        # Ensure at least basic user role
        if not local_roles:
            local_roles = ['user']
        
        return local_roles
    
    def _map_external_groups(self, external_groups: List[str]) -> List[str]:
        """Map external groups to local groups.
        
        Args:
            external_groups: External group names
            
        Returns:
            Local group IDs
        """
        local_groups = []
        
        for external_group in external_groups:
            # Find matching local group
            for group_id, group in self.user_management.groups.items():
                if group['name'].lower() == external_group.lower():
                    local_groups.append(group_id)
                    break
        
        return local_groups
    
    def _get_default_attribute_mapping(self, provider_type: IdentityProviderType) -> Dict:
        """Get default attribute mapping for a provider type."""
        if provider_type == IdentityProviderType.OAUTH2:
            return {
                'username': 'preferred_username',
                'email': 'email',
                'full_name': 'name',
                'groups': 'groups',
                'roles': {
                    'admin': 'admin',
                    'developer': 'developer',
                    '*': 'user'
                }
            }
        elif provider_type == IdentityProviderType.SAML2:
            return {
                'username': 'http://schemas.xmlsoap.org/ws/2005/05/identity/claims/name',
                'email': 'http://schemas.xmlsoap.org/ws/2005/05/identity/claims/emailaddress',
                'full_name': 'http://schemas.xmlsoap.org/ws/2005/05/identity/claims/givenname',
                'groups': 'http://schemas.xmlsoap.org/claims/Group',
                'roles': {
                    'Admin': 'admin',
                    'Developer': 'developer',
                    '*': 'user'
                }
            }
        elif provider_type == IdentityProviderType.LDAP:
            return {
                'username': 'sAMAccountName',
                'email': 'mail',
                'full_name': 'displayName',
                'groups': 'memberOf',
                'roles': {
                    'CN=Admins,OU=Groups,DC=example,DC=com': 'admin',
                    'CN=Developers,OU=Groups,DC=example,DC=com': 'developer',
                    '*': 'user'
                }
            }
        else:
            return {}
    
    def _initialize_default_providers(self):
        """Initialize default identity providers from configuration."""
        # OAuth2 providers
        if 'oauth2' in self.config:
            for name, oauth_config in self.config['oauth2'].items():
                self.configure_provider(
                    name=name,
                    type=IdentityProviderType.OAUTH2,
                    config=oauth_config
                )
        
        # SAML2 providers
        if 'saml2' in self.config:
            for name, saml_config in self.config['saml2'].items():
                self.configure_provider(
                    name=name,
                    type=IdentityProviderType.SAML2,
                    config=saml_config
                )
        
        # LDAP providers
        if 'ldap' in self.config:
            for name, ldap_config in self.config['ldap'].items():
                self.configure_provider(
                    name=name,
                    type=IdentityProviderType.LDAP,
                    config=ldap_config
                )
    
    def _generate_provider_id(self) -> str:
        """Generate unique provider ID."""
        return f"idp_{secrets.token_hex(8)}"
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        return f"sso_{secrets.token_hex(16)}"
    
    def _generate_state(self) -> str:
        """Generate state parameter for OAuth/SAML."""
        return secrets.token_urlsafe(32)
    
    def _load_provider_data(self):
        """Load provider data from storage."""
        providers_file = self.storage_path / "providers.json"
        if providers_file.exists():
            with open(providers_file, 'r') as f:
                providers_data = json.load(f)
                for provider_data in providers_data:
                    provider = IdentityProviderConfig(
                        provider_id=provider_data['provider_id'],
                        name=provider_data['name'],
                        type=IdentityProviderType(provider_data['type']),
                        enabled=provider_data.get('enabled', True),
                        priority=provider_data.get('priority', 1),
                        config=provider_data['config'],
                        attribute_mapping=provider_data.get('attribute_mapping', {}),
                        metadata=provider_data.get('metadata', {})
                    )
                    self.providers[provider.provider_id] = provider
    
    def _save_provider_data(self):
        """Save provider data to storage."""
        providers_data = []
        for provider in self.providers.values():
            provider_dict = {
                'provider_id': provider.provider_id,
                'name': provider.name,
                'type': provider.type.value,
                'enabled': provider.enabled,
                'priority': provider.priority,
                'config': provider.config,
                'attribute_mapping': provider.attribute_mapping,
                'metadata': provider.metadata
            }
            providers_data.append(provider_dict)
        
        with open(self.storage_path / "providers.json", 'w') as f:
            json.dump(providers_data, f, indent=2)


class OAuth2Handler:
    """Handler for OAuth 2.0 providers."""
    
    def __init__(self, integration: IdentityProviderIntegration):
        self.integration = integration
    
    def validate_config(self, config: Dict):
        """Validate OAuth2 configuration."""
        required = ['client_id', 'client_secret', 'authorization_url', 'token_url']
        for field in required:
            if field not in config:
                raise ValueError(f"Missing required OAuth2 config field: {field}")
    
    def initiate_auth(self, provider: IdentityProviderConfig, 
                     session: AuthenticationSession, scope: List[str] = None) -> Tuple[str, Dict]:
        """Initiate OAuth2 authentication."""
        config = provider.config
        
        # Build authorization URL
        params = {
            'client_id': config['client_id'],
            'redirect_uri': session.redirect_uri,
            'response_type': 'code',
            'state': session.state,
            'scope': ' '.join(scope or config.get('default_scopes', ['openid', 'profile', 'email']))
        }
        
        auth_url = f"{config['authorization_url']}?{urlencode(params)}"
        
        return auth_url, {}
    
    def complete_auth(self, provider: IdentityProviderConfig,
                     session: AuthenticationSession, callback_data: Dict) -> UserIdentity:
        """Complete OAuth2 authentication."""
        config = provider.config
        
        # Verify state
        if callback_data.get('state') != session.state:
            raise ValueError("Invalid state parameter")
        
        # Exchange code for token
        code = callback_data.get('code')
        if not code:
            raise ValueError("No authorization code received")
        
        token_data = {
            'client_id': config['client_id'],
            'client_secret': config['client_secret'],
            'code': code,
            'redirect_uri': session.redirect_uri,
            'grant_type': 'authorization_code'
        }
        
        response = requests.post(config['token_url'], data=token_data)
        response.raise_for_status()
        
        tokens = response.json()
        access_token = tokens.get('access_token')
        
        # Get user info
        headers = {'Authorization': f'Bearer {access_token}'}
        user_response = requests.get(config['userinfo_url'], headers=headers)
        user_response.raise_for_status()
        
        user_data = user_response.json()
        
        # Map user attributes
        mapping = provider.attribute_mapping
        identity = UserIdentity(
            provider_id=provider.provider_id,
            external_id=user_data.get('sub', user_data.get('id')),
            username=user_data.get(mapping.get('username', 'preferred_username')),
            email=user_data.get(mapping.get('email', 'email')),
            full_name=user_data.get(mapping.get('full_name', 'name')),
            groups=user_data.get(mapping.get('groups', 'groups'), []),
            attributes=user_data
        )
        
        return identity


class SAML2Handler:
    """Handler for SAML 2.0 providers."""
    
    def __init__(self, integration: IdentityProviderIntegration):
        self.integration = integration
    
    def validate_config(self, config: Dict):
        """Validate SAML2 configuration."""
        required = ['entity_id', 'sso_url', 'x509cert']
        for field in required:
            if field not in config:
                raise ValueError(f"Missing required SAML2 config field: {field}")
    
    def initiate_auth(self, provider: IdentityProviderConfig,
                     session: AuthenticationSession, scope: List[str] = None) -> Tuple[str, Dict]:
        """Initiate SAML2 authentication."""
        config = provider.config
        
        # Create SAML AuthnRequest
        request_id = f"id{secrets.token_hex(16)}"
        issue_instant = datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
        
        authn_request = f"""
        <samlp:AuthnRequest xmlns:samlp="urn:oasis:names:tc:SAML:2.0:protocol"
                           xmlns:saml="urn:oasis:names:tc:SAML:2.0:assertion"
                           ID="{request_id}"
                           Version="2.0"
                           IssueInstant="{issue_instant}"
                           Destination="{config['sso_url']}"
                           AssertionConsumerServiceURL="{session.redirect_uri}">
            <saml:Issuer>{config['sp_entity_id']}</saml:Issuer>
        </samlp:AuthnRequest>
        """
        
        # Encode and sign if required
        encoded_request = base64.b64encode(authn_request.encode()).decode()
        
        # Build SSO URL
        params = {
            'SAMLRequest': encoded_request,
            'RelayState': session.state
        }
        
        sso_url = f"{config['sso_url']}?{urlencode(params)}"
        
        return sso_url, {'request_id': request_id}
    
    def complete_auth(self, provider: IdentityProviderConfig,
                     session: AuthenticationSession, callback_data: Dict) -> UserIdentity:
        """Complete SAML2 authentication."""
        config = provider.config
        
        # Get SAML response
        saml_response = callback_data.get('SAMLResponse')
        if not saml_response:
            raise ValueError("No SAML response received")
        
        # Decode response
        decoded_response = base64.b64decode(saml_response).decode()
        
        # Parse XML
        root = ET.fromstring(decoded_response)
        
        # Verify signature if configured
        # (Simplified - real implementation would use xmlsec)
        
        # Extract assertions
        namespaces = {
            'samlp': 'urn:oasis:names:tc:SAML:2.0:protocol',
            'saml': 'urn:oasis:names:tc:SAML:2.0:assertion'
        }
        
        assertion = root.find('.//saml:Assertion', namespaces)
        if not assertion:
            raise ValueError("No SAML assertion found")
        
        # Extract attributes
        attributes = {}
        for attr in assertion.findall('.//saml:Attribute', namespaces):
            name = attr.get('Name')
            values = [v.text for v in attr.findall('.//saml:AttributeValue', namespaces)]
            attributes[name] = values[0] if len(values) == 1 else values
        
        # Map user attributes
        mapping = provider.attribute_mapping
        identity = UserIdentity(
            provider_id=provider.provider_id,
            external_id=attributes.get('NameID', ''),
            username=attributes.get(mapping.get('username', 'name'), ''),
            email=attributes.get(mapping.get('email', 'email'), ''),
            full_name=attributes.get(mapping.get('full_name', 'displayName'), ''),
            groups=attributes.get(mapping.get('groups', 'groups'), []),
            attributes=attributes
        )
        
        return identity


class LDAPHandler:
    """Handler for LDAP providers."""
    
    def __init__(self, integration: IdentityProviderIntegration):
        self.integration = integration
    
    def validate_config(self, config: Dict):
        """Validate LDAP configuration."""
        required = ['server_url', 'bind_dn', 'bind_password', 'user_base_dn']
        for field in required:
            if field not in config:
                raise ValueError(f"Missing required LDAP config field: {field}")
    
    def authenticate(self, provider: IdentityProviderConfig,
                    username: str, password: str) -> Optional[UserIdentity]:
        """Authenticate using LDAP."""
        config = provider.config
        
        try:
            # Connect to LDAP server
            conn = ldap.initialize(config['server_url'])
            conn.protocol_version = ldap.VERSION3
            
            # Bind with service account
            conn.simple_bind_s(config['bind_dn'], config['bind_password'])
            
            # Search for user
            search_filter = f"({config.get('username_attribute', 'sAMAccountName')}={username})"
            result = conn.search_s(
                config['user_base_dn'],
                ldap.SCOPE_SUBTREE,
                search_filter,
                ['*']
            )
            
            if not result:
                return None
            
            user_dn, user_attrs = result[0]
            
            # Try to bind as user
            user_conn = ldap.initialize(config['server_url'])
            user_conn.simple_bind_s(user_dn, password)
            
            # Get user attributes
            mapping = provider.attribute_mapping
            
            # Extract groups
            groups = []
            member_of = user_attrs.get(mapping.get('groups', 'memberOf'), [])
            for group_dn in member_of:
                # Extract CN from DN
                cn_match = re.match(r'CN=([^,]+)', group_dn.decode())
                if cn_match:
                    groups.append(cn_match.group(1))
            
            identity = UserIdentity(
                provider_id=provider.provider_id,
                external_id=user_dn,
                username=user_attrs.get(mapping.get('username', 'sAMAccountName'), [b''])[0].decode(),
                email=user_attrs.get(mapping.get('email', 'mail'), [b''])[0].decode(),
                full_name=user_attrs.get(mapping.get('full_name', 'displayName'), [b''])[0].decode(),
                groups=groups,
                attributes={k: v[0].decode() if v else '' for k, v in user_attrs.items()}
            )
            
            return identity
            
        except ldap.INVALID_CREDENTIALS:
            return None
        except Exception as e:
            logger.error(f"LDAP authentication error: {str(e)}")
            raise
        finally:
            try:
                conn.unbind()
            except:
                pass


class OIDCHandler(OAuth2Handler):
    """Handler for OpenID Connect providers."""
    
    def validate_config(self, config: Dict):
        """Validate OIDC configuration."""
        super().validate_config(config)
        
        # Additional OIDC requirements
        if 'issuer' not in config and 'discovery_url' not in config:
            raise ValueError("OIDC config must include either 'issuer' or 'discovery_url'")
    
    def initiate_auth(self, provider: IdentityProviderConfig,
                     session: AuthenticationSession, scope: List[str] = None) -> Tuple[str, Dict]:
        """Initiate OIDC authentication."""
        config = provider.config
        
        # Discover endpoints if needed
        if 'discovery_url' in config and 'authorization_url' not in config:
            self._discover_endpoints(config)
        
        # Add OIDC-specific parameters
        session.nonce = secrets.token_urlsafe(32)
        
        # Use OAuth2 flow with OIDC additions
        auth_url, params = super().initiate_auth(provider, session, scope)
        
        # Add nonce to URL
        auth_url += f"&nonce={session.nonce}"
        
        return auth_url, params
    
    def complete_auth(self, provider: IdentityProviderConfig,
                     session: AuthenticationSession, callback_data: Dict) -> UserIdentity:
        """Complete OIDC authentication."""
        # Use OAuth2 flow to get tokens
        identity = super().complete_auth(provider, session, callback_data)
        
        # Additionally validate ID token if present
        # (Simplified - real implementation would verify JWT signature)
        
        return identity
    
    def _discover_endpoints(self, config: Dict):
        """Discover OIDC endpoints from discovery URL."""
        discovery_url = config.get('discovery_url') or f"{config['issuer']}/.well-known/openid-configuration"
        
        response = requests.get(discovery_url)
        response.raise_for_status()
        
        discovery = response.json()
        
        # Update config with discovered endpoints
        config['authorization_url'] = discovery['authorization_endpoint']
        config['token_url'] = discovery['token_endpoint']
        config['userinfo_url'] = discovery['userinfo_endpoint']
        config['jwks_uri'] = discovery['jwks_uri']


# Singleton instance
_identity_integration = None

def get_identity_integration(config: Dict = None) -> IdentityProviderIntegration:
    """Get or create the singleton IdentityProviderIntegration instance."""
    global _identity_integration
    if _identity_integration is None:
        _identity_integration = IdentityProviderIntegration(config)
    return _identity_integration