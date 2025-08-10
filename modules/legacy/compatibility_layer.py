"""
Compatibility layer for legacy system healing.

This module provides healing capabilities for compatibility issues between
different versions, protocols, and data formats in heterogeneous environments.
"""

import json
import logging
import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import struct
import codecs

logger = logging.getLogger(__name__)


class CompatibilityIssue(Enum):
    """Types of compatibility issues."""
    ENCODING_MISMATCH = "encoding_mismatch"
    DATA_TYPE_MISMATCH = "data_type_mismatch"
    PROTOCOL_VERSION = "protocol_version"
    API_VERSION = "api_version"
    SCHEMA_MISMATCH = "schema_mismatch"
    DATE_FORMAT = "date_format"
    NUMERIC_PRECISION = "numeric_precision"
    CHARACTER_SET = "character_set"
    ENDIANNESS = "endianness"
    FIELD_LENGTH = "field_length"


class DataFormat(Enum):
    """Data format types."""
    JSON = "json"
    XML = "xml"
    CSV = "csv"
    FIXED_WIDTH = "fixed_width"
    BINARY = "binary"
    EBCDIC = "ebcdic"
    ASCII = "ascii"
    COPYBOOK = "copybook"
    EDI = "edi"
    PROPRIETARY = "proprietary"


@dataclass
class CompatibilityError:
    """Represents a compatibility error."""
    issue_type: CompatibilityIssue
    source_format: str
    target_format: str
    field_name: Optional[str]
    source_value: Any
    expected_type: Optional[str]
    message: str
    severity: str  # low, medium, high, critical
    suggested_fix: Optional[str] = None


@dataclass
class ConversionRule:
    """Rule for data conversion between formats."""
    source_type: str
    target_type: str
    converter: callable
    validator: Optional[callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SchemaMapping:
    """Mapping between different schemas."""
    source_field: str
    target_field: str
    transformation: Optional[callable] = None
    default_value: Any = None
    required: bool = False


class CompatibilityLayer:
    """
    Provides compatibility healing between different systems and formats.
    
    Handles encoding conversions, data type mappings, protocol translations,
    and schema reconciliation.
    """
    
    def __init__(self):
        self._encoding_map = self._init_encoding_map()
        self._conversion_rules = self._init_conversion_rules()
        self._date_formats = self._init_date_formats()
        self._schema_mappings: Dict[str, List[SchemaMapping]] = {}
        self._protocol_adapters = self._init_protocol_adapters()
        
    def _init_encoding_map(self) -> Dict[str, Dict[str, Any]]:
        """Initialize encoding conversion mappings."""
        return {
            "ebcdic": {
                "name": "EBCDIC",
                "python_codec": "cp500",  # IBM EBCDIC
                "variants": ["cp037", "cp1140", "cp1047"],
                "description": "Extended Binary Coded Decimal Interchange Code"
            },
            "ascii": {
                "name": "ASCII",
                "python_codec": "ascii",
                "variants": ["us-ascii"],
                "description": "American Standard Code for Information Interchange"
            },
            "utf8": {
                "name": "UTF-8",
                "python_codec": "utf-8",
                "variants": ["utf_8"],
                "description": "8-bit Unicode Transformation Format"
            },
            "utf16": {
                "name": "UTF-16",
                "python_codec": "utf-16",
                "variants": ["utf_16", "utf-16-le", "utf-16-be"],
                "description": "16-bit Unicode Transformation Format"
            },
            "latin1": {
                "name": "Latin-1",
                "python_codec": "latin-1",
                "variants": ["iso-8859-1", "iso8859-1"],
                "description": "ISO 8859-1 Latin alphabet No. 1"
            }
        }
        
    def _init_conversion_rules(self) -> List[ConversionRule]:
        """Initialize data type conversion rules."""
        return [
            # Numeric conversions
            ConversionRule(
                source_type="packed_decimal",
                target_type="decimal",
                converter=self._convert_packed_decimal,
                validator=self._validate_packed_decimal
            ),
            ConversionRule(
                source_type="zoned_decimal",
                target_type="decimal",
                converter=self._convert_zoned_decimal
            ),
            ConversionRule(
                source_type="binary_integer",
                target_type="integer",
                converter=self._convert_binary_integer
            ),
            # String conversions
            ConversionRule(
                source_type="fixed_string",
                target_type="string",
                converter=self._convert_fixed_string
            ),
            ConversionRule(
                source_type="null_terminated",
                target_type="string",
                converter=self._convert_null_terminated
            ),
            # Date conversions
            ConversionRule(
                source_type="julian_date",
                target_type="iso_date",
                converter=self._convert_julian_date
            ),
            ConversionRule(
                source_type="epoch_seconds",
                target_type="iso_datetime",
                converter=self._convert_epoch_time
            ),
            # Boolean conversions
            ConversionRule(
                source_type="flag_byte",
                target_type="boolean",
                converter=self._convert_flag_byte
            )
        ]
        
    def _init_date_formats(self) -> Dict[str, str]:
        """Initialize date format patterns."""
        return {
            "iso": "%Y-%m-%d",
            "iso_datetime": "%Y-%m-%dT%H:%M:%S",
            "us": "%m/%d/%Y",
            "eu": "%d/%m/%Y",
            "julian": "%Y%j",  # Year + day of year
            "mainframe": "%y%m%d",  # YYMMDD
            "sap": "%Y%m%d",  # YYYYMMDD
            "oracle": "%d-%b-%Y",  # DD-MON-YYYY
            "db2": "%Y-%m-%d-%H.%M.%S",
            "epoch": "epoch",  # Special handling
            "excel": "excel"  # Special handling for Excel date numbers
        }
        
    def _init_protocol_adapters(self) -> Dict[str, Any]:
        """Initialize protocol adapters."""
        return {
            "soap_to_rest": self._adapt_soap_to_rest,
            "rest_to_soap": self._adapt_rest_to_soap,
            "mq_to_http": self._adapt_mq_to_http,
            "http_to_mq": self._adapt_http_to_mq,
            "edi_to_json": self._adapt_edi_to_json,
            "json_to_edi": self._adapt_json_to_edi
        }
        
    def detect_compatibility_issues(self, source_data: Any, source_format: DataFormat,
                                  target_format: DataFormat) -> List[CompatibilityError]:
        """Detect compatibility issues between formats."""
        issues = []
        
        # Check encoding compatibility
        if source_format in [DataFormat.EBCDIC, DataFormat.ASCII] and \
           target_format in [DataFormat.JSON, DataFormat.XML]:
            issues.extend(self._check_encoding_issues(source_data, source_format, target_format))
            
        # Check data type compatibility
        if isinstance(source_data, dict):
            issues.extend(self._check_data_type_issues(source_data, source_format, target_format))
            
        # Check schema compatibility
        if source_format == DataFormat.COPYBOOK and target_format == DataFormat.JSON:
            issues.extend(self._check_copybook_compatibility(source_data))
            
        # Check date format compatibility
        issues.extend(self._check_date_format_issues(source_data, source_format, target_format))
        
        # Check numeric precision
        if source_format in [DataFormat.FIXED_WIDTH, DataFormat.COPYBOOK]:
            issues.extend(self._check_numeric_precision(source_data, target_format))
            
        return issues
        
    def _check_encoding_issues(self, data: Any, source: DataFormat, 
                             target: DataFormat) -> List[CompatibilityError]:
        """Check for encoding-related issues."""
        issues = []
        
        if source == DataFormat.EBCDIC:
            # EBCDIC to modern format conversion issues
            issues.append(CompatibilityError(
                issue_type=CompatibilityIssue.ENCODING_MISMATCH,
                source_format=source.value,
                target_format=target.value,
                field_name=None,
                source_value=data,
                expected_type="UTF-8",
                message="EBCDIC to UTF-8 conversion required",
                severity="medium",
                suggested_fix="Use EBCDIC to UTF-8 converter"
            ))
            
        return issues
        
    def _check_data_type_issues(self, data: Dict[str, Any], source: DataFormat,
                              target: DataFormat) -> List[CompatibilityError]:
        """Check for data type compatibility issues."""
        issues = []
        
        for field, value in data.items():
            # Check for packed decimal fields
            if isinstance(value, bytes) and self._is_packed_decimal(value):
                issues.append(CompatibilityError(
                    issue_type=CompatibilityIssue.DATA_TYPE_MISMATCH,
                    source_format="packed_decimal",
                    target_format="decimal",
                    field_name=field,
                    source_value=value,
                    expected_type="decimal",
                    message=f"Field {field} contains packed decimal data",
                    severity="high",
                    suggested_fix="Convert packed decimal to standard decimal"
                ))
                
            # Check for EBCDIC strings
            if isinstance(value, bytes):
                try:
                    value.decode('ascii')
                except UnicodeDecodeError:
                    issues.append(CompatibilityError(
                        issue_type=CompatibilityIssue.CHARACTER_SET,
                        source_format="ebcdic",
                        target_format="utf-8",
                        field_name=field,
                        source_value=value,
                        expected_type="string",
                        message=f"Field {field} appears to contain EBCDIC data",
                        severity="high",
                        suggested_fix="Convert EBCDIC to UTF-8"
                    ))
                    
        return issues
        
    def _check_copybook_compatibility(self, data: Any) -> List[CompatibilityError]:
        """Check COBOL copybook compatibility issues."""
        issues = []
        
        # Check for COMP-3 (packed decimal) fields
        # Check for OCCURS clauses (arrays)
        # Check for REDEFINES (union types)
        # Check for level 88 (conditional values)
        
        issues.append(CompatibilityError(
            issue_type=CompatibilityIssue.SCHEMA_MISMATCH,
            source_format="copybook",
            target_format="json",
            field_name=None,
            source_value=data,
            expected_type="object",
            message="COBOL copybook structure requires transformation",
            severity="high",
            suggested_fix="Apply copybook to JSON transformation"
        ))
        
        return issues
        
    def _check_date_format_issues(self, data: Any, source: DataFormat,
                                target: DataFormat) -> List[CompatibilityError]:
        """Check for date format compatibility issues."""
        issues = []
        
        if isinstance(data, dict):
            for field, value in data.items():
                if self._is_date_field(field, value):
                    # Detect date format
                    detected_format = self._detect_date_format(value)
                    
                    if detected_format and detected_format != "iso":
                        issues.append(CompatibilityError(
                            issue_type=CompatibilityIssue.DATE_FORMAT,
                            source_format=detected_format,
                            target_format="iso",
                            field_name=field,
                            source_value=value,
                            expected_type="date",
                            message=f"Field {field} uses {detected_format} date format",
                            severity="medium",
                            suggested_fix=f"Convert from {detected_format} to ISO format"
                        ))
                        
        return issues
        
    def _check_numeric_precision(self, data: Any, target: DataFormat) -> List[CompatibilityError]:
        """Check for numeric precision issues."""
        issues = []
        
        # Check for potential precision loss when converting
        # from fixed-point to floating-point
        
        return issues
        
    def heal_compatibility_issues(self, data: Any, issues: List[CompatibilityError],
                                source_format: DataFormat, 
                                target_format: DataFormat) -> Tuple[Any, List[str]]:
        """Heal detected compatibility issues."""
        healed_data = data
        actions_taken = []
        
        for issue in issues:
            try:
                if issue.issue_type == CompatibilityIssue.ENCODING_MISMATCH:
                    healed_data, action = self._heal_encoding(healed_data, issue)
                    actions_taken.append(action)
                    
                elif issue.issue_type == CompatibilityIssue.DATA_TYPE_MISMATCH:
                    healed_data, action = self._heal_data_type(healed_data, issue)
                    actions_taken.append(action)
                    
                elif issue.issue_type == CompatibilityIssue.DATE_FORMAT:
                    healed_data, action = self._heal_date_format(healed_data, issue)
                    actions_taken.append(action)
                    
                elif issue.issue_type == CompatibilityIssue.SCHEMA_MISMATCH:
                    healed_data, action = self._heal_schema(healed_data, issue, source_format, target_format)
                    actions_taken.append(action)
                    
                elif issue.issue_type == CompatibilityIssue.CHARACTER_SET:
                    healed_data, action = self._heal_character_set(healed_data, issue)
                    actions_taken.append(action)
                    
            except Exception as e:
                logger.error(f"Failed to heal {issue.issue_type}: {e}")
                actions_taken.append(f"Failed to heal {issue.issue_type}: {str(e)}")
                
        return healed_data, actions_taken
        
    def _heal_encoding(self, data: Any, issue: CompatibilityError) -> Tuple[Any, str]:
        """Heal encoding issues."""
        if issue.source_format == "ebcdic":
            # Convert EBCDIC to UTF-8
            if isinstance(data, bytes):
                converted = self.convert_encoding(data, "ebcdic", "utf8")
                return converted, "Converted EBCDIC to UTF-8"
            elif isinstance(data, str):
                # Already converted
                return data, "Data already in text format"
                
        return data, "No encoding conversion needed"
        
    def _heal_data_type(self, data: Any, issue: CompatibilityError) -> Tuple[Any, str]:
        """Heal data type mismatches."""
        if issue.field_name and isinstance(data, dict):
            field_value = data.get(issue.field_name)
            
            if issue.source_format == "packed_decimal":
                # Convert packed decimal
                converted = self._convert_packed_decimal(field_value)
                data[issue.field_name] = converted
                return data, f"Converted packed decimal in field {issue.field_name}"
                
            elif issue.source_format == "zoned_decimal":
                # Convert zoned decimal
                converted = self._convert_zoned_decimal(field_value)
                data[issue.field_name] = converted
                return data, f"Converted zoned decimal in field {issue.field_name}"
                
        return data, "No data type conversion performed"
        
    def _heal_date_format(self, data: Any, issue: CompatibilityError) -> Tuple[Any, str]:
        """Heal date format issues."""
        if issue.field_name and isinstance(data, dict):
            field_value = data.get(issue.field_name)
            
            if field_value:
                # Convert to ISO format
                converted = self.convert_date_format(
                    field_value,
                    issue.source_format,
                    "iso"
                )
                data[issue.field_name] = converted
                return data, f"Converted date format in field {issue.field_name} from {issue.source_format} to ISO"
                
        return data, "No date conversion performed"
        
    def _heal_schema(self, data: Any, issue: CompatibilityError,
                    source_format: DataFormat, target_format: DataFormat) -> Tuple[Any, str]:
        """Heal schema mismatches."""
        if source_format == DataFormat.COPYBOOK and target_format == DataFormat.JSON:
            # Transform COBOL copybook to JSON
            transformed = self.transform_copybook_to_json(data)
            return transformed, "Transformed COBOL copybook structure to JSON"
            
        elif source_format == DataFormat.XML and target_format == DataFormat.JSON:
            # Transform XML to JSON
            transformed = self.transform_xml_to_json(data)
            return transformed, "Transformed XML to JSON"
            
        return data, "No schema transformation performed"
        
    def _heal_character_set(self, data: Any, issue: CompatibilityError) -> Tuple[Any, str]:
        """Heal character set issues."""
        if issue.field_name and isinstance(data, dict):
            field_value = data.get(issue.field_name)
            
            if isinstance(field_value, bytes):
                try:
                    # Try EBCDIC conversion
                    converted = field_value.decode('cp500')  # IBM EBCDIC
                    data[issue.field_name] = converted
                    return data, f"Converted EBCDIC character set in field {issue.field_name}"
                except:
                    # Try other encodings
                    for encoding in ['cp037', 'cp1140', 'latin-1']:
                        try:
                            converted = field_value.decode(encoding)
                            data[issue.field_name] = converted
                            return data, f"Converted {encoding} character set in field {issue.field_name}"
                        except:
                            continue
                            
        return data, "Character set conversion failed"
        
    def convert_encoding(self, data: bytes, source_encoding: str, 
                        target_encoding: str) -> str:
        """Convert between character encodings."""
        source_codec = self._encoding_map.get(source_encoding, {}).get("python_codec", source_encoding)
        target_codec = self._encoding_map.get(target_encoding, {}).get("python_codec", target_encoding)
        
        try:
            # Decode from source encoding
            text = data.decode(source_codec)
            
            # Encode to target encoding if needed
            if target_encoding != "utf8":
                return text.encode(target_codec)
            else:
                return text
                
        except Exception as e:
            logger.error(f"Encoding conversion failed: {e}")
            raise
            
    def convert_date_format(self, date_value: Any, source_format: str, 
                          target_format: str) -> str:
        """Convert between date formats."""
        if source_format == target_format:
            return date_value
            
        # Parse source date
        if source_format == "epoch":
            # Unix timestamp
            dt = datetime.fromtimestamp(float(date_value))
        elif source_format == "excel":
            # Excel date number (days since 1900-01-01)
            dt = datetime(1900, 1, 1) + timedelta(days=float(date_value) - 2)
        elif source_format == "julian":
            # Julian date YYYYDDD
            year = int(str(date_value)[:4])
            day = int(str(date_value)[4:])
            dt = datetime(year, 1, 1) + timedelta(days=day - 1)
        else:
            # Standard format string
            format_str = self._date_formats.get(source_format, source_format)
            dt = datetime.strptime(str(date_value), format_str)
            
        # Format to target
        if target_format == "epoch":
            return int(dt.timestamp())
        elif target_format == "excel":
            delta = dt - datetime(1900, 1, 1)
            return delta.days + 2
        elif target_format == "julian":
            return dt.strftime("%Y%j")
        else:
            format_str = self._date_formats.get(target_format, target_format)
            return dt.strftime(format_str)
            
    def _convert_packed_decimal(self, packed_bytes: bytes) -> float:
        """Convert packed decimal (COMP-3) to decimal."""
        # Packed decimal stores two digits per byte, with sign in last nibble
        digits = []
        
        for i, byte in enumerate(packed_bytes):
            if i == len(packed_bytes) - 1:
                # Last byte contains one digit and sign
                digits.append(byte >> 4)
                sign = byte & 0x0F
            else:
                # Each byte contains two digits
                digits.append(byte >> 4)
                digits.append(byte & 0x0F)
                
        # Convert to number
        value = 0
        for digit in digits:
            value = value * 10 + digit
            
        # Apply sign (0xC = positive, 0xD = negative)
        if sign == 0x0D:
            value = -value
            
        return value
        
    def _validate_packed_decimal(self, packed_bytes: bytes) -> bool:
        """Validate packed decimal format."""
        if not packed_bytes:
            return False
            
        # Check each byte
        for i, byte in enumerate(packed_bytes[:-1]):
            # Each nibble should be 0-9
            if (byte >> 4) > 9 or (byte & 0x0F) > 9:
                return False
                
        # Check last byte
        last_byte = packed_bytes[-1]
        if (last_byte >> 4) > 9:
            return False
            
        # Sign nibble should be C, D, or F
        sign = last_byte & 0x0F
        if sign not in [0x0C, 0x0D, 0x0F]:
            return False
            
        return True
        
    def _convert_zoned_decimal(self, zoned_bytes: bytes) -> float:
        """Convert zoned decimal to decimal."""
        # Zoned decimal stores one digit per byte with zone in high nibble
        digits = []
        sign = 1
        
        for i, byte in enumerate(zoned_bytes):
            digit = byte & 0x0F
            zone = byte >> 4
            
            digits.append(digit)
            
            # Check for sign in last byte
            if i == len(zoned_bytes) - 1:
                if zone == 0xD:  # Negative
                    sign = -1
                    
        # Convert to number
        value = 0
        for digit in digits:
            value = value * 10 + digit
            
        return value * sign
        
    def _convert_binary_integer(self, binary_bytes: bytes, 
                              big_endian: bool = True) -> int:
        """Convert binary integer."""
        if big_endian:
            return int.from_bytes(binary_bytes, byteorder='big', signed=True)
        else:
            return int.from_bytes(binary_bytes, byteorder='little', signed=True)
            
    def _convert_fixed_string(self, fixed_bytes: bytes, 
                            encoding: str = 'ascii') -> str:
        """Convert fixed-width string."""
        # Remove padding (usually spaces or null bytes)
        text = fixed_bytes.decode(encoding)
        return text.rstrip(' \x00')
        
    def _convert_null_terminated(self, bytes_data: bytes, 
                                encoding: str = 'ascii') -> str:
        """Convert null-terminated string."""
        # Find null terminator
        null_pos = bytes_data.find(b'\x00')
        if null_pos >= 0:
            bytes_data = bytes_data[:null_pos]
            
        return bytes_data.decode(encoding)
        
    def _convert_flag_byte(self, flag_byte: bytes) -> bool:
        """Convert flag byte to boolean."""
        if not flag_byte:
            return False
            
        # Common representations
        byte_val = flag_byte[0] if isinstance(flag_byte, bytes) else flag_byte
        
        # Check common true values
        if byte_val in [0x01, ord('Y'), ord('T'), ord('1')]:
            return True
        # Check common false values
        elif byte_val in [0x00, ord('N'), ord('F'), ord('0'), ord(' ')]:
            return False
        else:
            # Default to True for any non-zero value
            return byte_val != 0
            
    def _convert_julian_date(self, julian_date: str) -> str:
        """Convert Julian date to ISO format."""
        # Julian date format: YYYYDDD or YYDDD
        if len(julian_date) == 7:
            year = int(julian_date[:4])
            day = int(julian_date[4:])
        elif len(julian_date) == 5:
            year = 1900 + int(julian_date[:2])
            if year < 1950:  # Windowing for 2-digit years
                year += 100
            day = int(julian_date[2:])
        else:
            raise ValueError(f"Invalid Julian date: {julian_date}")
            
        # Convert to date
        date = datetime(year, 1, 1) + timedelta(days=day - 1)
        return date.strftime("%Y-%m-%d")
        
    def _convert_epoch_time(self, epoch_seconds: Union[int, float]) -> str:
        """Convert epoch seconds to ISO datetime."""
        dt = datetime.fromtimestamp(float(epoch_seconds))
        return dt.strftime("%Y-%m-%dT%H:%M:%S")
        
    def _is_packed_decimal(self, data: bytes) -> bool:
        """Check if data appears to be packed decimal."""
        if not isinstance(data, bytes) or len(data) == 0:
            return False
            
        # Check last nibble for valid sign
        last_byte = data[-1]
        sign_nibble = last_byte & 0x0F
        
        return sign_nibble in [0x0C, 0x0D, 0x0F]
        
    def _is_date_field(self, field_name: str, value: Any) -> bool:
        """Heuristic to detect date fields."""
        # Check field name
        date_keywords = ['date', 'time', 'timestamp', 'created', 'modified', 
                        'updated', 'expired', 'birth', 'start', 'end']
        
        field_lower = field_name.lower()
        if any(keyword in field_lower for keyword in date_keywords):
            return True
            
        # Check value format
        if isinstance(value, str):
            # Check for common date patterns
            date_patterns = [
                r'^\d{4}-\d{2}-\d{2}',  # ISO date
                r'^\d{2}/\d{2}/\d{4}',  # US date
                r'^\d{2}-\w{3}-\d{4}',  # Oracle date
                r'^\d{8}$',  # YYYYMMDD
                r'^\d{6}$'   # YYMMDD
            ]
            
            for pattern in date_patterns:
                if re.match(pattern, value):
                    return True
                    
        return False
        
    def _detect_date_format(self, value: Any) -> Optional[str]:
        """Detect date format from value."""
        if not isinstance(value, str):
            return None
            
        # Check patterns
        if re.match(r'^\d{4}-\d{2}-\d{2}$', value):
            return "iso"
        elif re.match(r'^\d{2}/\d{2}/\d{4}$', value):
            return "us"
        elif re.match(r'^\d{2}/\d{2}/\d{4}$', value):
            return "eu"
        elif re.match(r'^\d{8}$', value):
            return "sap"
        elif re.match(r'^\d{6}$', value):
            return "mainframe"
        elif re.match(r'^\d{2}-\w{3}-\d{4}$', value):
            return "oracle"
        elif re.match(r'^\d{7}$', value) or re.match(r'^\d{5}$', value):
            return "julian"
            
        return None
        
    def transform_copybook_to_json(self, copybook_data: bytes) -> Dict[str, Any]:
        """Transform COBOL copybook data to JSON."""
        # This is a simplified example - real implementation would need
        # the copybook definition to properly parse the data
        
        result = {}
        
        # Example: Parse fixed-width fields
        # Assuming a simple structure:
        # - Customer ID: 10 bytes
        # - Name: 30 bytes
        # - Balance: 9 bytes packed decimal
        # - Status: 1 byte
        
        offset = 0
        
        # Customer ID
        if len(copybook_data) >= offset + 10:
            result["customer_id"] = self._convert_fixed_string(
                copybook_data[offset:offset+10]
            )
            offset += 10
            
        # Name
        if len(copybook_data) >= offset + 30:
            result["name"] = self._convert_fixed_string(
                copybook_data[offset:offset+30]
            )
            offset += 30
            
        # Balance (packed decimal)
        if len(copybook_data) >= offset + 5:
            packed_balance = copybook_data[offset:offset+5]
            if self._is_packed_decimal(packed_balance):
                result["balance"] = self._convert_packed_decimal(packed_balance) / 100
            offset += 5
            
        # Status
        if len(copybook_data) >= offset + 1:
            result["status"] = chr(copybook_data[offset])
            offset += 1
            
        return result
        
    def transform_xml_to_json(self, xml_data: str) -> Dict[str, Any]:
        """Transform XML to JSON."""
        def element_to_dict(element):
            result = {}
            
            # Add attributes
            if element.attrib:
                result["@attributes"] = element.attrib
                
            # Add text content
            if element.text and element.text.strip():
                if len(element) == 0:  # No children
                    return element.text.strip()
                else:
                    result["#text"] = element.text.strip()
                    
            # Add children
            for child in element:
                child_data = element_to_dict(child)
                
                if child.tag in result:
                    # Multiple children with same tag - convert to list
                    if not isinstance(result[child.tag], list):
                        result[child.tag] = [result[child.tag]]
                    result[child.tag].append(child_data)
                else:
                    result[child.tag] = child_data
                    
            return result
            
        try:
            root = ET.fromstring(xml_data)
            return {root.tag: element_to_dict(root)}
        except Exception as e:
            logger.error(f"XML parsing failed: {e}")
            return {"error": str(e), "raw_data": xml_data}
            
    def add_schema_mapping(self, mapping_name: str, mappings: List[SchemaMapping]):
        """Add schema mapping configuration."""
        self._schema_mappings[mapping_name] = mappings
        
    def apply_schema_mapping(self, data: Dict[str, Any], 
                           mapping_name: str) -> Dict[str, Any]:
        """Apply schema mapping to transform data."""
        if mapping_name not in self._schema_mappings:
            raise ValueError(f"Schema mapping {mapping_name} not found")
            
        mappings = self._schema_mappings[mapping_name]
        result = {}
        
        for mapping in mappings:
            # Get source value
            source_value = data.get(mapping.source_field, mapping.default_value)
            
            # Apply transformation if defined
            if mapping.transformation and source_value is not None:
                try:
                    transformed_value = mapping.transformation(source_value)
                except Exception as e:
                    logger.error(f"Transformation failed for {mapping.source_field}: {e}")
                    transformed_value = mapping.default_value
            else:
                transformed_value = source_value
                
            # Set target field
            result[mapping.target_field] = transformed_value
            
            # Check required fields
            if mapping.required and transformed_value is None:
                raise ValueError(f"Required field {mapping.target_field} is missing")
                
        return result
        
    def _adapt_soap_to_rest(self, soap_request: str) -> Dict[str, Any]:
        """Adapt SOAP request to REST format."""
        # Parse SOAP envelope
        try:
            root = ET.fromstring(soap_request)
            
            # Extract body content
            body = root.find(".//{http://schemas.xmlsoap.org/soap/envelope/}Body")
            if body is None:
                body = root.find(".//Body")  # Try without namespace
                
            if body is not None and len(body) > 0:
                # Get first child of body (the actual request)
                request_elem = body[0]
                
                # Convert to dictionary
                return self._xml_element_to_dict(request_elem)
                
        except Exception as e:
            logger.error(f"SOAP parsing failed: {e}")
            
        return {"error": "Failed to parse SOAP request"}
        
    def _adapt_rest_to_soap(self, rest_data: Dict[str, Any], 
                          operation: str = "Request") -> str:
        """Adapt REST data to SOAP format."""
        soap = f"""<?xml version="1.0" encoding="UTF-8"?>
<soap:Envelope xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">
    <soap:Body>
        <{operation}>"""
        
        for key, value in rest_data.items():
            soap += f"\n            <{key}>{value}</{key}>"
            
        soap += f"""
        </{operation}>
    </soap:Body>
</soap:Envelope>"""
        
        return soap
        
    def _adapt_mq_to_http(self, mq_message: bytes) -> Dict[str, Any]:
        """Adapt MQ message to HTTP format."""
        # Parse MQ message header and body
        # This is simplified - real implementation would handle MQMD and RFH2 headers
        
        return {
            "headers": {
                "X-MQ-MessageId": "generated-id",
                "X-MQ-CorrelationId": "correlation-id"
            },
            "body": mq_message.decode('utf-8', errors='ignore')
        }
        
    def _adapt_http_to_mq(self, http_data: Dict[str, Any]) -> bytes:
        """Adapt HTTP data to MQ message format."""
        # Create MQ message with headers
        # This is simplified - real implementation would create proper MQMD
        
        message = json.dumps(http_data).encode('utf-8')
        return message
        
    def _adapt_edi_to_json(self, edi_data: str) -> Dict[str, Any]:
        """Adapt EDI format to JSON."""
        # Simplified EDI X12 parser
        segments = edi_data.strip().split('~')
        result = {
            "segments": []
        }
        
        for segment in segments:
            if segment:
                elements = segment.split('*')
                if elements:
                    result["segments"].append({
                        "id": elements[0],
                        "elements": elements[1:]
                    })
                    
        return result
        
    def _adapt_json_to_edi(self, json_data: Dict[str, Any]) -> str:
        """Adapt JSON to EDI format."""
        # Simplified EDI X12 generator
        edi_segments = []
        
        for segment in json_data.get("segments", []):
            segment_id = segment.get("id", "")
            elements = segment.get("elements", [])
            
            edi_segment = segment_id + "*" + "*".join(str(e) for e in elements)
            edi_segments.append(edi_segment)
            
        return "~".join(edi_segments) + "~"
        
    def _xml_element_to_dict(self, element) -> Dict[str, Any]:
        """Convert XML element to dictionary."""
        result = {}
        
        # Add attributes with @ prefix
        for key, value in element.attrib.items():
            result[f"@{key}"] = value
            
        # Add text content
        if element.text and element.text.strip():
            result["#text"] = element.text.strip()
            
        # Add child elements
        for child in element:
            child_dict = self._xml_element_to_dict(child)
            
            if child.tag in result:
                # Convert to list if multiple children with same tag
                if not isinstance(result[child.tag], list):
                    result[child.tag] = [result[child.tag]]
                result[child.tag].append(child_dict)
            else:
                result[child.tag] = child_dict
                
        return result if result else element.text