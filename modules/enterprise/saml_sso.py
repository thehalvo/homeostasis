"""
SAML/SSO Authentication Module for Homeostasis Enterprise

Provides SAML 2.0 Single Sign-On (SSO) authentication support for enterprise environments.
Supports multiple identity providers (IdPs) and handles the complete SAML authentication flow.
"""

import base64
import datetime
import logging
import uuid
from typing import Dict, List, Optional, Tuple, Any, Union
from urllib.parse import urlencode
from dataclasses import dataclass, field

import xmlsec
from lxml import etree

from modules.security.auth import AuthenticationManager, AuthenticationError
from modules.security.rbac import get_rbac_manager
from modules.security.user_management import get_user_management
from modules.security.audit import get_audit_logger

logger = logging.getLogger(__name__)


# SAML Namespaces
SAML_NAMESPACE = "urn:oasis:names:tc:SAML:2.0:assertion"
SAMLP_NAMESPACE = "urn:oasis:names:tc:SAML:2.0:protocol"
XMLDSIG_NAMESPACE = "http://www.w3.org/2000/09/xmldsig#"
XSI_NAMESPACE = "http://www.w3.org/2001/XMLSchema-instance"


@dataclass
class SAMLIdentityProvider:
    """Configuration for a SAML Identity Provider"""
    idp_id: str
    name: str
    entity_id: str
    sso_url: str
    slo_url: Optional[str]
    certificate: str  # Base64 encoded X.509 certificate
    metadata_url: Optional[str]
    attribute_mapping: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SAMLServiceProvider:
    """Configuration for SAML Service Provider (our app)"""
    entity_id: str
    acs_url: str  # Assertion Consumer Service URL
    slo_url: Optional[str]  # Single Logout URL
    certificate: str  # Base64 encoded X.509 certificate
    private_key: str  # Base64 encoded private key
    metadata_url: Optional[str]
    name_id_format: str = "urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress"
    authn_requests_signed: bool = True
    want_assertions_signed: bool = True
    want_assertions_encrypted: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SAMLAssertion:
    """Parsed SAML Assertion"""
    name_id: str
    session_index: str
    attributes: Dict[str, Any]
    not_before: datetime.datetime
    not_on_or_after: datetime.datetime
    authn_instant: datetime.datetime
    issuer: str
    audience: str
    raw_assertion: str


@dataclass
class SAMLRequest:
    """SAML Authentication Request"""
    request_id: str
    idp_id: str
    relay_state: Optional[str]
    created_at: datetime.datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class SAMLAuthenticationManager:
    """
    Manages SAML-based authentication for enterprise SSO.
    
    Supports multiple identity providers and handles the complete
    SAML authentication flow including SP-initiated SSO.
    """
    
    def __init__(self, config: Dict[str, Any], auth_manager: AuthenticationManager):
        """Initialize SAML authentication manager.
        
        Args:
            config: SAML configuration
            auth_manager: Base authentication manager
        """
        self.config = config
        self.auth_manager = auth_manager
        
        # Service Provider configuration
        self.sp_config = self._load_sp_config(config.get('service_provider', {}))
        
        # Identity Providers
        self.identity_providers: Dict[str, SAMLIdentityProvider] = {}
        self._load_identity_providers(config.get('identity_providers', []))
        
        # Managers
        self.rbac_manager = get_rbac_manager()
        self.user_management = get_user_management()
        self.audit_logger = get_audit_logger()
        
        # Request tracking
        self.pending_requests: Dict[str, SAMLRequest] = {}
        
        # Session mapping
        self.saml_sessions: Dict[str, Dict[str, Any]] = {}
        
        logger.info("Initialized SAML authentication manager")
    
    def _load_sp_config(self, config: Dict[str, Any]) -> SAMLServiceProvider:
        """Load Service Provider configuration"""
        return SAMLServiceProvider(
            entity_id=config.get('entity_id', 'https://homeostasis.local/saml'),
            acs_url=config.get('acs_url', 'https://homeostasis.local/saml/acs'),
            slo_url=config.get('slo_url'),
            certificate=config.get('certificate', ''),
            private_key=config.get('private_key', ''),
            metadata_url=config.get('metadata_url'),
            name_id_format=config.get('name_id_format', 
                                     'urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress'),
            authn_requests_signed=config.get('authn_requests_signed', True),
            want_assertions_signed=config.get('want_assertions_signed', True),
            want_assertions_encrypted=config.get('want_assertions_encrypted', False),
            metadata=config.get('metadata', {})
        )
    
    def _load_identity_providers(self, idp_configs: List[Dict[str, Any]]):
        """Load Identity Provider configurations"""
        for config in idp_configs:
            idp = SAMLIdentityProvider(
                idp_id=config['idp_id'],
                name=config['name'],
                entity_id=config['entity_id'],
                sso_url=config['sso_url'],
                slo_url=config.get('slo_url'),
                certificate=config['certificate'],
                metadata_url=config.get('metadata_url'),
                attribute_mapping=config.get('attribute_mapping', {}),
                enabled=config.get('enabled', True),
                metadata=config.get('metadata', {})
            )
            self.identity_providers[idp.idp_id] = idp
            logger.info(f"Loaded identity provider: {idp.name}")
    
    def get_identity_providers(self) -> List[Dict[str, Any]]:
        """Get list of configured identity providers"""
        providers = []
        for idp in self.identity_providers.values():
            if idp.enabled:
                providers.append({
                    'idp_id': idp.idp_id,
                    'name': idp.name,
                    'entity_id': idp.entity_id,
                    'metadata_url': idp.metadata_url
                })
        return providers
    
    def create_authn_request(self, idp_id: str, relay_state: Optional[str] = None) -> str:
        """Create SAML Authentication Request.
        
        Args:
            idp_id: Identity provider ID
            relay_state: Optional relay state
            
        Returns:
            URL to redirect user to for authentication
        """
        idp = self.identity_providers.get(idp_id)
        if not idp or not idp.enabled:
            raise AuthenticationError(f"Invalid or disabled identity provider: {idp_id}")
        
        # Generate request ID
        request_id = f"id_{uuid.uuid4().hex}"
        
        # Create AuthnRequest XML
        authn_request = self._build_authn_request(request_id, idp)
        
        # Sign request if required
        if self.sp_config.authn_requests_signed:
            authn_request = self._sign_request(authn_request)
        
        # Encode request
        encoded_request = base64.b64encode(authn_request.encode('utf-8')).decode('utf-8')
        
        # Store request for validation
        saml_request = SAMLRequest(
            request_id=request_id,
            idp_id=idp_id,
            relay_state=relay_state,
            created_at=datetime.datetime.utcnow(),
            metadata={'ip_address': self.config.get('client_ip')}
        )
        self.pending_requests[request_id] = saml_request
        
        # Build redirect URL
        params = {
            'SAMLRequest': encoded_request
        }
        if relay_state:
            params['RelayState'] = relay_state
        
        redirect_url = f"{idp.sso_url}?{urlencode(params)}"
        
        # Log authentication attempt
        self.audit_logger.log_event(
            event_type='saml_auth_initiated',
            user='anonymous',
            details={
                'idp_id': idp_id,
                'request_id': request_id,
                'relay_state': relay_state
            }
        )
        
        return redirect_url
    
    def _build_authn_request(self, request_id: str, idp: SAMLIdentityProvider) -> str:
        """Build SAML AuthnRequest XML"""
        # Create root element
        root = etree.Element(
            "{%s}AuthnRequest" % SAMLP_NAMESPACE,
            nsmap={'samlp': SAMLP_NAMESPACE, 'saml': SAML_NAMESPACE}
        )
        
        # Set attributes
        root.set('ID', request_id)
        root.set('Version', '2.0')
        root.set('IssueInstant', datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'))
        root.set('Destination', idp.sso_url)
        root.set('AssertionConsumerServiceURL', self.sp_config.acs_url)
        root.set('ProtocolBinding', 'urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST')
        
        # Add Issuer
        issuer = etree.SubElement(root, "{%s}Issuer" % SAML_NAMESPACE)
        issuer.text = self.sp_config.entity_id
        
        # Add NameIDPolicy
        name_id_policy = etree.SubElement(root, "{%s}NameIDPolicy" % SAMLP_NAMESPACE)
        name_id_policy.set('Format', self.sp_config.name_id_format)
        name_id_policy.set('AllowCreate', 'true')
        
        # Add RequestedAuthnContext (optional)
        requested_authn_context = etree.SubElement(root, "{%s}RequestedAuthnContext" % SAMLP_NAMESPACE)
        requested_authn_context.set('Comparison', 'exact')
        
        authn_context_class_ref = etree.SubElement(
            requested_authn_context, 
            "{%s}AuthnContextClassRef" % SAML_NAMESPACE
        )
        authn_context_class_ref.text = 'urn:oasis:names:tc:SAML:2.0:ac:classes:PasswordProtectedTransport'
        
        return etree.tostring(root, encoding='unicode', pretty_print=True)
    
    def _sign_request(self, request_xml: str) -> str:
        """Sign SAML request with SP private key"""
        # Load private key
        private_key_pem = base64.b64decode(self.sp_config.private_key)
        
        # Parse XML
        doc = etree.fromstring(request_xml.encode('utf-8'))
        
        # Create signature template
        signature = xmlsec.template.create(
            doc,
            xmlsec.Transform.EXCL_C14N,
            xmlsec.Transform.RSA_SHA256
        )
        
        # Add signature to the document
        doc.insert(0, signature)
        
        # Add reference
        ref = xmlsec.template.add_reference(
            signature,
            xmlsec.Transform.SHA256,
            uri='#' + doc.get('ID')
        )
        
        # Add transforms
        xmlsec.template.add_transform(ref, xmlsec.Transform.ENVELOPED)
        xmlsec.template.add_transform(ref, xmlsec.Transform.EXCL_C14N)
        
        # Add key info
        key_info = xmlsec.template.ensure_key_info(signature)
        xmlsec.template.add_x509_data(key_info)
        
        # Create signature context
        ctx = xmlsec.SignatureContext()
        
        # Load private key
        key = xmlsec.Key.from_memory(private_key_pem, xmlsec.KeyFormat.PEM)
        ctx.key = key
        
        # Sign the document
        ctx.sign(signature)
        
        return etree.tostring(doc, encoding='unicode', pretty_print=True)
    
    def process_saml_response(self, saml_response: str, relay_state: Optional[str] = None) -> Tuple[Dict[str, Any], str]:
        """Process SAML Response from IdP.
        
        Args:
            saml_response: Base64 encoded SAML response
            relay_state: Optional relay state
            
        Returns:
            Tuple of (user_info, access_token)
        """
        try:
            # Decode response
            decoded_response = base64.b64decode(saml_response)
            
            # Parse XML
            doc = etree.fromstring(decoded_response)
            
            # Extract and validate assertion
            assertion = self._extract_and_validate_assertion(doc)
            
            # Map attributes to user info
            user_info = self._map_assertion_to_user(assertion)
            
            # Create or update user
            user = self._create_or_update_user(user_info, assertion)
            
            # Generate tokens
            access_token, refresh_token = self.auth_manager.generate_token({
                'username': user['username'],
                'roles': user['roles']
            })
            
            # Store SAML session
            self.saml_sessions[user['username']] = {
                'session_index': assertion.session_index,
                'name_id': assertion.name_id,
                'idp_id': self._get_idp_by_issuer(assertion.issuer).idp_id,
                'login_time': datetime.datetime.utcnow().isoformat()
            }
            
            # Log successful authentication
            self.audit_logger.log_event(
                event_type='saml_auth_success',
                user=user['username'],
                details={
                    'idp': assertion.issuer,
                    'session_index': assertion.session_index,
                    'attributes': list(assertion.attributes.keys())
                }
            )
            
            return user_info, access_token
            
        except Exception as e:
            logger.error(f"SAML response processing failed: {e}")
            self.audit_logger.log_event(
                event_type='saml_auth_failure',
                user='unknown',
                details={
                    'error': str(e),
                    'relay_state': relay_state
                }
            )
            raise AuthenticationError(f"SAML authentication failed: {str(e)}")
    
    def _extract_and_validate_assertion(self, response_doc: etree.Element) -> SAMLAssertion:
        """Extract and validate SAML assertion from response"""
        # Find assertion
        assertions = response_doc.findall('.//{%s}Assertion' % SAML_NAMESPACE)
        if not assertions:
            raise AuthenticationError("No assertion found in SAML response")
        
        assertion_elem = assertions[0]
        
        # Validate signature if required
        if self.sp_config.want_assertions_signed:
            self._validate_assertion_signature(assertion_elem)
        
        # Extract basic info
        issuer = assertion_elem.find('.//{%s}Issuer' % SAML_NAMESPACE).text
        
        # Validate issuer
        idp = self._get_idp_by_issuer(issuer)
        if not idp:
            raise AuthenticationError(f"Unknown issuer: {issuer}")
        
        # Extract subject
        subject = assertion_elem.find('.//{%s}Subject' % SAML_NAMESPACE)
        name_id_elem = subject.find('.//{%s}NameID' % SAML_NAMESPACE)
        name_id = name_id_elem.text
        
        # Extract session index
        authn_statement = assertion_elem.find('.//{%s}AuthnStatement' % SAML_NAMESPACE)
        session_index = authn_statement.get('SessionIndex', '')
        authn_instant = datetime.datetime.strptime(
            authn_statement.get('AuthnInstant'),
            '%Y-%m-%dT%H:%M:%SZ'
        )
        
        # Extract conditions
        conditions = assertion_elem.find('.//{%s}Conditions' % SAML_NAMESPACE)
        not_before = datetime.datetime.strptime(
            conditions.get('NotBefore'),
            '%Y-%m-%dT%H:%M:%SZ'
        )
        not_on_or_after = datetime.datetime.strptime(
            conditions.get('NotOnOrAfter'),
            '%Y-%m-%dT%H:%M:%SZ'
        )
        
        # Validate time conditions
        now = datetime.datetime.utcnow()
        if now < not_before or now >= not_on_or_after:
            raise AuthenticationError("Assertion is not valid at current time")
        
        # Extract audience
        audience_elem = conditions.find('.//{%s}AudienceRestriction/{%s}Audience' % (SAML_NAMESPACE, SAML_NAMESPACE))
        audience = audience_elem.text if audience_elem is not None else None
        
        # Validate audience
        if audience and audience != self.sp_config.entity_id:
            raise AuthenticationError(f"Invalid audience: {audience}")
        
        # Extract attributes
        attributes = {}
        attribute_statement = assertion_elem.find('.//{%s}AttributeStatement' % SAML_NAMESPACE)
        if attribute_statement is not None:
            for attribute in attribute_statement.findall('.//{%s}Attribute' % SAML_NAMESPACE):
                attr_name = attribute.get('Name')
                attr_values = []
                for value_elem in attribute.findall('.//{%s}AttributeValue' % SAML_NAMESPACE):
                    attr_values.append(value_elem.text)
                
                # Single value or list
                if len(attr_values) == 1:
                    attributes[attr_name] = attr_values[0]
                else:
                    attributes[attr_name] = attr_values
        
        return SAMLAssertion(
            name_id=name_id,
            session_index=session_index,
            attributes=attributes,
            not_before=not_before,
            not_on_or_after=not_on_or_after,
            authn_instant=authn_instant,
            issuer=issuer,
            audience=audience or self.sp_config.entity_id,
            raw_assertion=etree.tostring(assertion_elem, encoding='unicode')
        )
    
    def _validate_assertion_signature(self, assertion_elem: etree.Element):
        """Validate SAML assertion signature"""
        # Find signature
        signature_elem = assertion_elem.find('.//{%s}Signature' % XMLDSIG_NAMESPACE)
        if not signature_elem:
            raise AuthenticationError("Assertion is not signed")
        
        # Get issuer to find the right certificate
        issuer = assertion_elem.find('.//{%s}Issuer' % SAML_NAMESPACE).text
        idp = self._get_idp_by_issuer(issuer)
        
        # Load IdP certificate
        cert_pem = base64.b64decode(idp.certificate)
        
        # Create signature context
        ctx = xmlsec.SignatureContext()
        
        # Load certificate
        key = xmlsec.Key.from_memory(cert_pem, xmlsec.KeyFormat.CERT_PEM)
        ctx.key = key
        
        # Verify signature
        try:
            ctx.verify(signature_elem)
        except Exception as e:
            raise AuthenticationError(f"Invalid assertion signature: {str(e)}")
    
    def _get_idp_by_issuer(self, issuer: str) -> Optional[SAMLIdentityProvider]:
        """Find IdP by issuer/entity ID"""
        for idp in self.identity_providers.values():
            if idp.entity_id == issuer:
                return idp
        return None
    
    def _map_assertion_to_user(self, assertion: SAMLAssertion) -> Dict[str, Any]:
        """Map SAML assertion attributes to user info"""
        idp = self._get_idp_by_issuer(assertion.issuer)
        
        # Default mapping
        user_info = {
            'username': assertion.name_id,
            'email': assertion.name_id,  # Assume email format
            'attributes': assertion.attributes
        }
        
        # Apply attribute mapping
        for local_attr, saml_attr in idp.attribute_mapping.items():
            if saml_attr in assertion.attributes:
                user_info[local_attr] = assertion.attributes[saml_attr]
        
        # Extract common attributes
        common_mappings = {
            'email': ['email', 'mail', 'emailAddress'],
            'first_name': ['firstName', 'givenName', 'given_name'],
            'last_name': ['lastName', 'surname', 'sn', 'family_name'],
            'display_name': ['displayName', 'name'],
            'groups': ['groups', 'memberOf', 'roles']
        }
        
        for local_attr, possible_attrs in common_mappings.items():
            if local_attr not in user_info:
                for saml_attr in possible_attrs:
                    if saml_attr in assertion.attributes:
                        user_info[local_attr] = assertion.attributes[saml_attr]
                        break
        
        return user_info
    
    def _create_or_update_user(self, user_info: Dict[str, Any], assertion: SAMLAssertion) -> Dict[str, Any]:
        """Create or update user from SAML assertion"""
        username = user_info['username']
        
        # Check if user exists
        existing_user = self.user_management.get_user(username)
        
        if existing_user:
            # Update user attributes
            updates = {
                'email': user_info.get('email'),
                'metadata': {
                    'saml_attributes': user_info.get('attributes', {}),
                    'last_login': datetime.datetime.utcnow().isoformat(),
                    'auth_method': 'saml',
                    'idp': assertion.issuer
                }
            }
            
            # Update name if provided
            if 'first_name' in user_info or 'last_name' in user_info:
                updates['metadata']['first_name'] = user_info.get('first_name')
                updates['metadata']['last_name'] = user_info.get('last_name')
            
            self.user_management.update_user(username, **updates)
            
            # Get updated user
            user = self.user_management.get_user(username)
            
        else:
            # Create new user
            email = user_info.get('email', f"{username}@saml.local")
            
            # Determine roles from groups
            roles = self._map_groups_to_roles(user_info.get('groups', []))
            if not roles:
                roles = ['user']  # Default role
            
            self.user_management.create_user(
                username=username,
                email=email,
                roles=roles,
                metadata={
                    'saml_attributes': user_info.get('attributes', {}),
                    'first_name': user_info.get('first_name'),
                    'last_name': user_info.get('last_name'),
                    'created_via': 'saml',
                    'idp': assertion.issuer
                }
            )
            
            user = self.user_management.get_user(username)
        
        return user
    
    def _map_groups_to_roles(self, groups: Union[str, List[str]]) -> List[str]:
        """Map SAML groups to application roles"""
        if isinstance(groups, str):
            groups = [groups]
        
        # Get role mapping from config
        role_mapping = self.config.get('role_mapping', {})
        
        roles = []
        for group in groups:
            # Direct mapping
            if group in role_mapping:
                mapped_role = role_mapping[group]
                if isinstance(mapped_role, list):
                    roles.extend(mapped_role)
                else:
                    roles.append(mapped_role)
            else:
                # Check pattern matching
                for pattern, mapped_role in role_mapping.items():
                    if '*' in pattern:
                        # Simple wildcard matching
                        import fnmatch
                        if fnmatch.fnmatch(group, pattern):
                            if isinstance(mapped_role, list):
                                roles.extend(mapped_role)
                            else:
                                roles.append(mapped_role)
        
        # Remove duplicates
        return list(set(roles))
    
    def create_logout_request(self, username: str) -> Optional[str]:
        """Create SAML Logout Request.
        
        Args:
            username: Username to logout
            
        Returns:
            URL to redirect user to for logout, or None if not applicable
        """
        session = self.saml_sessions.get(username)
        if not session:
            return None
        
        idp = self.identity_providers.get(session['idp_id'])
        if not idp or not idp.slo_url:
            # No SLO support, just clear local session
            del self.saml_sessions[username]
            return None
        
        # Generate logout request ID
        request_id = f"id_{uuid.uuid4().hex}"
        
        # Build LogoutRequest
        logout_request = self._build_logout_request(
            request_id,
            idp,
            session['name_id'],
            session['session_index']
        )
        
        # Sign request if required
        if self.sp_config.authn_requests_signed:
            logout_request = self._sign_request(logout_request)
        
        # Encode request
        encoded_request = base64.b64encode(logout_request.encode('utf-8')).decode('utf-8')
        
        # Build redirect URL
        params = {
            'SAMLRequest': encoded_request
        }
        
        redirect_url = f"{idp.slo_url}?{urlencode(params)}"
        
        # Log logout attempt
        self.audit_logger.log_event(
            event_type='saml_logout_initiated',
            user=username,
            details={
                'idp_id': session['idp_id'],
                'session_index': session['session_index']
            }
        )
        
        return redirect_url
    
    def _build_logout_request(self, request_id: str, idp: SAMLIdentityProvider,
                            name_id: str, session_index: str) -> str:
        """Build SAML LogoutRequest XML"""
        # Create root element
        root = etree.Element(
            "{%s}LogoutRequest" % SAMLP_NAMESPACE,
            nsmap={'samlp': SAMLP_NAMESPACE, 'saml': SAML_NAMESPACE}
        )
        
        # Set attributes
        root.set('ID', request_id)
        root.set('Version', '2.0')
        root.set('IssueInstant', datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'))
        root.set('Destination', idp.slo_url)
        
        # Add Issuer
        issuer = etree.SubElement(root, "{%s}Issuer" % SAML_NAMESPACE)
        issuer.text = self.sp_config.entity_id
        
        # Add NameID
        name_id_elem = etree.SubElement(root, "{%s}NameID" % SAML_NAMESPACE)
        name_id_elem.set('Format', self.sp_config.name_id_format)
        name_id_elem.text = name_id
        
        # Add SessionIndex
        session_index_elem = etree.SubElement(root, "{%s}SessionIndex" % SAMLP_NAMESPACE)
        session_index_elem.text = session_index
        
        return etree.tostring(root, encoding='unicode', pretty_print=True)
    
    def process_logout_response(self, saml_response: str) -> bool:
        """Process SAML Logout Response.
        
        Args:
            saml_response: Base64 encoded SAML logout response
            
        Returns:
            True if logout was successful
        """
        try:
            # Decode response
            decoded_response = base64.b64decode(saml_response)
            
            # Parse XML
            doc = etree.fromstring(decoded_response)
            
            # Check status
            status_code = doc.find('.//{%s}StatusCode' % SAMLP_NAMESPACE)
            if status_code is not None:
                status_value = status_code.get('Value')
                if status_value == 'urn:oasis:names:tc:SAML:2.0:status:Success':
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Logout response processing failed: {e}")
            return False
    
    def generate_metadata(self) -> str:
        """Generate SP metadata XML"""
        # Create root element
        root = etree.Element(
            "{urn:oasis:names:tc:SAML:2.0:metadata}EntityDescriptor",
            nsmap={
                'md': 'urn:oasis:names:tc:SAML:2.0:metadata',
                'ds': XMLDSIG_NAMESPACE
            }
        )
        
        root.set('entityID', self.sp_config.entity_id)
        
        # Add SPSSODescriptor
        sp_sso = etree.SubElement(
            root,
            "{urn:oasis:names:tc:SAML:2.0:metadata}SPSSODescriptor"
        )
        sp_sso.set('AuthnRequestsSigned', str(self.sp_config.authn_requests_signed).lower())
        sp_sso.set('WantAssertionsSigned', str(self.sp_config.want_assertions_signed).lower())
        sp_sso.set('protocolSupportEnumeration', 'urn:oasis:names:tc:SAML:2.0:protocol')
        
        # Add KeyDescriptor for signing
        if self.sp_config.certificate:
            key_descriptor = etree.SubElement(
                sp_sso,
                "{urn:oasis:names:tc:SAML:2.0:metadata}KeyDescriptor"
            )
            key_descriptor.set('use', 'signing')
            
            key_info = etree.SubElement(
                key_descriptor,
                "{%s}KeyInfo" % XMLDSIG_NAMESPACE
            )
            
            x509_data = etree.SubElement(
                key_info,
                "{%s}X509Data" % XMLDSIG_NAMESPACE
            )
            
            x509_cert = etree.SubElement(
                x509_data,
                "{%s}X509Certificate" % XMLDSIG_NAMESPACE
            )
            
            # Clean certificate
            cert = self.sp_config.certificate.replace('-----BEGIN CERTIFICATE-----', '')
            cert = cert.replace('-----END CERTIFICATE-----', '')
            cert = cert.replace('\n', '')
            x509_cert.text = cert
        
        # Add NameIDFormat
        name_id_format = etree.SubElement(
            sp_sso,
            "{urn:oasis:names:tc:SAML:2.0:metadata}NameIDFormat"
        )
        name_id_format.text = self.sp_config.name_id_format
        
        # Add AssertionConsumerService
        acs = etree.SubElement(
            sp_sso,
            "{urn:oasis:names:tc:SAML:2.0:metadata}AssertionConsumerService"
        )
        acs.set('Binding', 'urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST')
        acs.set('Location', self.sp_config.acs_url)
        acs.set('index', '1')
        
        # Add SingleLogoutService if configured
        if self.sp_config.slo_url:
            slo = etree.SubElement(
                sp_sso,
                "{urn:oasis:names:tc:SAML:2.0:metadata}SingleLogoutService"
            )
            slo.set('Binding', 'urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect')
            slo.set('Location', self.sp_config.slo_url)
        
        return etree.tostring(root, encoding='unicode', pretty_print=True)
    
    def validate_metadata(self, metadata_xml: str) -> bool:
        """Validate IdP metadata"""
        try:
            # Parse XML
            doc = etree.fromstring(metadata_xml.encode('utf-8'))
            
            # Check for required elements
            entity_id = doc.get('entityID')
            if not entity_id:
                return False
            
            # Find IDPSSODescriptor
            idp_sso = doc.find('.//{urn:oasis:names:tc:SAML:2.0:metadata}IDPSSODescriptor')
            if idp_sso is None:
                return False
            
            # Check for SSO service
            sso_services = idp_sso.findall(
                './/{urn:oasis:names:tc:SAML:2.0:metadata}SingleSignOnService'
            )
            if not sso_services:
                return False
            
            return True
            
        except Exception:
            return False
    
    def import_idp_metadata(self, metadata_xml: str, idp_id: str, name: str) -> bool:
        """Import IdP configuration from metadata"""
        try:
            # Parse metadata
            doc = etree.fromstring(metadata_xml.encode('utf-8'))
            
            # Extract entity ID
            entity_id = doc.get('entityID')
            
            # Find IDPSSODescriptor
            idp_sso = doc.find('.//{urn:oasis:names:tc:SAML:2.0:metadata}IDPSSODescriptor')
            
            # Extract SSO URL
            sso_service = idp_sso.find(
                './/{urn:oasis:names:tc:SAML:2.0:metadata}SingleSignOnService[@Binding="urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect"]'
            )
            if not sso_service:
                sso_service = idp_sso.find(
                    './/{urn:oasis:names:tc:SAML:2.0:metadata}SingleSignOnService'
                )
            
            sso_url = sso_service.get('Location')
            
            # Extract SLO URL if available
            slo_url = None
            slo_service = idp_sso.find(
                './/{urn:oasis:names:tc:SAML:2.0:metadata}SingleLogoutService'
            )
            if slo_service:
                slo_url = slo_service.get('Location')
            
            # Extract certificate
            x509_cert = idp_sso.find(
                './/{%s}X509Certificate' % XMLDSIG_NAMESPACE
            )
            
            if x509_cert is None:
                raise ValueError("No X509Certificate found in metadata")
            
            certificate = x509_cert.text.strip()
            
            # Create IdP configuration
            idp = SAMLIdentityProvider(
                idp_id=idp_id,
                name=name,
                entity_id=entity_id,
                sso_url=sso_url,
                slo_url=slo_url,
                certificate=certificate,
                metadata_url=None,
                attribute_mapping={},
                enabled=True,
                metadata={'imported_at': datetime.datetime.utcnow().isoformat()}
            )
            
            # Store IdP
            self.identity_providers[idp_id] = idp
            
            logger.info(f"Imported IdP metadata: {name} ({entity_id})")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to import IdP metadata: {e}")
            return False


# Factory function
def create_saml_manager(config: Dict[str, Any], auth_manager: AuthenticationManager) -> SAMLAuthenticationManager:
    """Create SAML authentication manager"""
    return SAMLAuthenticationManager(config, auth_manager)