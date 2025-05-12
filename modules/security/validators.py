"""
Security validation utilities for Homeostasis.

Provides validators for ensuring security standards are met for various inputs.
"""

import ipaddress
import re
from typing import Dict, Optional, Pattern, Tuple, Union

# Regular expressions for validation
EMAIL_REGEX = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
USERNAME_REGEX = re.compile(r'^[a-zA-Z0-9_-]{3,32}$')
PASSWORD_STRENGTH_REGEX = re.compile(
    r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&#])[A-Za-z\d@$!%*?&#]{8,}$'
)
URL_REGEX = re.compile(
    r'^(?:http|https)://(?:[\w-]+\.)+[\w]{2,}(?:/[\w-]+)*/?(?:\?[\w-]+=[\w-]+(?:&[\w-]+=[\w-]+)*)?$'
)
FILENAME_REGEX = re.compile(r'^[a-zA-Z0-9_.-]+$')
PATH_TRAVERSAL_REGEX = re.compile(r'\.\./')
SCRIPT_TAG_REGEX = re.compile(r'<script[^>]*>.*?</script>', re.IGNORECASE | re.DOTALL)
SQL_INJECTION_REGEX = re.compile(
    r'(?i)(?:\'|\%27)(?:\s|\+|\/\*.*\*\/)*(?:(?:select|union|delete|insert|drop|update))',
)


def validate_email(email: str) -> bool:
    """Validate an email address format.
    
    Args:
        email: Email address to validate
        
    Returns:
        bool: True if email is valid, False otherwise
    """
    return bool(EMAIL_REGEX.match(email))


def validate_username(username: str) -> bool:
    """Validate a username.
    
    Args:
        username: Username to validate
        
    Returns:
        bool: True if username is valid, False otherwise
    """
    return bool(USERNAME_REGEX.match(username))


def validate_password_strength(password: str) -> Tuple[bool, Optional[str]]:
    """Validate password strength.
    
    Args:
        password: Password to validate
        
    Returns:
        Tuple[bool, Optional[str]]: (is_valid, reason_if_invalid)
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
        
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter"
        
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter"
        
    if not re.search(r'\d', password):
        return False, "Password must contain at least one digit"
        
    if not re.search(r'[@$!%*?&#]', password):
        return False, "Password must contain at least one special character (@$!%*?&#)"
        
    return True, None


def validate_url(url: str) -> bool:
    """Validate a URL.
    
    Args:
        url: URL to validate
        
    Returns:
        bool: True if URL is valid, False otherwise
    """
    return bool(URL_REGEX.match(url))


def validate_ip_address(ip: str) -> bool:
    """Validate an IP address.
    
    Args:
        ip: IP address to validate
        
    Returns:
        bool: True if IP address is valid, False otherwise
    """
    try:
        ipaddress.ip_address(ip)
        return True
    except ValueError:
        return False


def validate_filename(filename: str) -> bool:
    """Validate a filename.
    
    Args:
        filename: Filename to validate
        
    Returns:
        bool: True if filename is valid, False otherwise
    """
    return bool(FILENAME_REGEX.match(filename))


def check_for_path_traversal(path: str) -> bool:
    """Check if a path contains path traversal attempts.
    
    Args:
        path: Path to check
        
    Returns:
        bool: True if path is safe, False if it contains path traversal
    """
    return not bool(PATH_TRAVERSAL_REGEX.search(path))


def check_for_xss(content: str) -> Tuple[bool, Optional[str]]:
    """Check if content contains potential XSS attacks.
    
    Args:
        content: Content to check
        
    Returns:
        Tuple[bool, Optional[str]]: (is_safe, reason_if_unsafe)
    """
    # Check for script tags
    if SCRIPT_TAG_REGEX.search(content):
        return False, "Content contains script tags"
        
    # Check for javascript: URLs
    if re.search(r'(?i)javascript\s*:', content):
        return False, "Content contains javascript: URLs"
        
    # Check for other common XSS patterns
    xss_patterns = [
        r'(?i)on\w+\s*=',  # onclick, onmouseover, etc.
        r'(?i)<%.*%>',  # JSP/ASP tags
        r'(?i)<\s*img[^>]*\s+src\s*=\s*[\'"]?\s*data:',  # Data URI in img src
        r'(?i)<\s*iframe',  # iframes
        r'(?i)<\s*meta',  # meta refresh/redirect
    ]
    
    for pattern in xss_patterns:
        if re.search(pattern, content):
            return False, f"Content contains potential XSS pattern: {pattern}"
            
    return True, None


def check_for_sql_injection(content: str) -> Tuple[bool, Optional[str]]:
    """Check if content contains potential SQL injection attacks.
    
    Args:
        content: Content to check
        
    Returns:
        Tuple[bool, Optional[str]]: (is_safe, reason_if_unsafe)
    """
    if SQL_INJECTION_REGEX.search(content):
        return False, "Content contains potential SQL injection patterns"
        
    # Check for other common SQL injection patterns
    sql_patterns = [
        r'(?i)--\s*$',  # SQL comment
        r'(?i)\/\*.*\*\/',  # SQL comment block
        r'(?i);.*$',  # Multiple statements
        r'(?i)@@version',  # SQL Server version
        r'(?i)pg_sleep',  # PostgreSQL sleep
        r'(?i)sleep\s*\(',  # MySQL sleep
        r'(?i)waitfor\s+delay',  # SQL Server delay
    ]
    
    for pattern in sql_patterns:
        if re.search(pattern, content):
            return False, f"Content contains potential SQL injection pattern: {pattern}"
            
    return True, None


def sanitize_input(content: str) -> str:
    """Sanitize input to remove potentially malicious content.
    
    Args:
        content: Content to sanitize
        
    Returns:
        str: Sanitized content
    """
    # Replace script tags
    content = SCRIPT_TAG_REGEX.sub('', content)
    
    # Replace javascript: URLs
    content = re.sub(r'(?i)javascript\s*:', 'disabled-javascript:', content)
    
    # Remove other potentially dangerous patterns
    content = re.sub(r'(?i)on\w+\s*=', 'data-removed=', content)
    content = re.sub(r'(?i)<%.*?%>', '', content)
    content = re.sub(r'(?i)<\s*iframe', '<x-iframe', content)
    content = re.sub(r'(?i)<\s*meta', '<x-meta', content)
    
    return content


def sanitize_filename(filename: str) -> str:
    """Sanitize a filename to make it safe.
    
    Args:
        filename: Filename to sanitize
        
    Returns:
        str: Sanitized filename
    """
    # Remove any path traversal
    filename = re.sub(r'\.\.+', '.', filename)
    
    # Remove any directory separators
    filename = re.sub(r'[/\\]', '_', filename)
    
    # Only allow alphanumeric, underscore, dash, and period
    return re.sub(r'[^a-zA-Z0-9_.-]', '_', filename)


def validate_and_sanitize_json(json_data: Dict) -> Tuple[bool, Dict, Optional[str]]:
    """Validate and sanitize JSON data.
    
    Args:
        json_data: JSON data to validate and sanitize
        
    Returns:
        Tuple[bool, Dict, Optional[str]]: (is_valid, sanitized_data, reason_if_invalid)
    """
    sanitized = {}
    
    # Process each field
    for key, value in json_data.items():
        # Sanitize keys
        sanitized_key = re.sub(r'[^\w.-]', '_', key)
        
        # Process values based on type
        if isinstance(value, str):
            # Check for potential attacks
            is_safe, reason = check_for_xss(value)
            if not is_safe:
                return False, {}, f"XSS detected in field {key}: {reason}"
                
            is_safe, reason = check_for_sql_injection(value)
            if not is_safe:
                return False, {}, f"SQL injection detected in field {key}: {reason}"
                
            # Sanitize string values
            sanitized[sanitized_key] = sanitize_input(value)
            
        elif isinstance(value, dict):
            # Recursively process nested dictionaries
            is_valid, nested_sanitized, reason = validate_and_sanitize_json(value)
            if not is_valid:
                return False, {}, reason
                
            sanitized[sanitized_key] = nested_sanitized
            
        elif isinstance(value, list):
            # Process list items
            sanitized_list = []
            for item in value:
                if isinstance(item, str):
                    # Check string items for potential attacks
                    is_safe, reason = check_for_xss(item)
                    if not is_safe:
                        return False, {}, f"XSS detected in list item: {reason}"
                        
                    is_safe, reason = check_for_sql_injection(item)
                    if not is_safe:
                        return False, {}, f"SQL injection detected in list item: {reason}"
                        
                    sanitized_list.append(sanitize_input(item))
                    
                elif isinstance(item, dict):
                    # Recursively process nested dictionaries in lists
                    is_valid, nested_sanitized, reason = validate_and_sanitize_json(item)
                    if not is_valid:
                        return False, {}, reason
                        
                    sanitized_list.append(nested_sanitized)
                    
                else:
                    # Pass through non-string, non-dict items
                    sanitized_list.append(item)
                    
            sanitized[sanitized_key] = sanitized_list
            
        else:
            # Pass through other types
            sanitized[sanitized_key] = value
            
    return True, sanitized, None