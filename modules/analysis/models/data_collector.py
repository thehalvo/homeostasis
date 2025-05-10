"""
Training data collection system for ML-based error analysis.

This module provides tools for collecting, processing, and managing
training data for the error classification and analysis models.
"""
import json
import os
import re
import uuid
import hashlib
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

# Default paths
DEFAULT_DATA_DIR = Path(__file__).parent / "training_data"
DEFAULT_ERROR_DATA_FILE = DEFAULT_DATA_DIR / "error_data.jsonl"
DEFAULT_LABELS_FILE = DEFAULT_DATA_DIR / "error_labels.jsonl"
DEFAULT_FEEDBACK_FILE = DEFAULT_DATA_DIR / "fix_feedback.jsonl"
DEFAULT_METADATA_FILE = DEFAULT_DATA_DIR / "dataset_metadata.json"


class DataAnonymizer:
    """Anonymizes sensitive data in error logs."""
    
    def __init__(self):
        """Initialize the anonymizer with common patterns to anonymize."""
        self.patterns = {
            'email': (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]'),
            'ip_address': (r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '[IP_ADDRESS]'),
            'password': (r'(password|passwd|pwd)[\s]*[=:]+[\s]*[\'"].*?[\'"]', r'\1=[PASSWORD]'),
            'api_key': (r'(api[_-]?key|token)[\s]*[=:]+[\s]*[\'"].*?[\'"]', r'\1=[API_KEY]'),
            'credit_card': (r'\b(?:\d{4}[-\s]?){3}\d{4}\b', '[CREDIT_CARD]'),
            'phone': (r'\b(?:\+\d{1,2}\s)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b', '[PHONE]'),
            'social_security': (r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]'),
            'address': (r'\b\d+\s+([A-Za-z]+\s+){1,5},\s+[A-Za-z]+,\s+[A-Z]{2}\s+\d{5}\b', '[ADDRESS]'),
            'filepath': (r'[\'"]?(?:/[^/]+)+/[^/\'"]+[\'"]?', '[FILEPATH]'),
            'url': (r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+(?:/[-\w%/.~:?#[\]@!$&\'()*+,;=]*)?', '[URL]'),
        }
    
    def anonymize_text(self, text: str) -> str:
        """
        Anonymize sensitive information in text.
        
        Args:
            text: Text to anonymize
            
        Returns:
            Anonymized text
        """
        if not text:
            return text
            
        result = text
        for pattern_name, (pattern, replacement) in self.patterns.items():
            result = re.sub(pattern, replacement, result)
        
        return result
    
    def anonymize_error_data(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Anonymize sensitive information in error data.
        
        Args:
            error_data: Error data dictionary
            
        Returns:
            Anonymized error data
        """
        # Make a deep copy to avoid modifying the original
        result = {}
        
        for key, value in error_data.items():
            if isinstance(value, str):
                result[key] = self.anonymize_text(value)
            elif isinstance(value, list) and all(isinstance(item, str) for item in value):
                result[key] = [self.anonymize_text(item) for item in value]
            elif isinstance(value, dict):
                result[key] = self.anonymize_error_data(value)
            elif isinstance(value, list) and all(isinstance(item, dict) for item in value):
                result[key] = [self.anonymize_error_data(item) for item in value]
            else:
                result[key] = value
        
        return result


class ErrorDataCollector:
    """Collects and manages training data for error analysis models."""
    
    def __init__(self, 
                 data_dir: Optional[Union[str, Path]] = None,
                 error_file: Optional[Union[str, Path]] = None,
                 labels_file: Optional[Union[str, Path]] = None,
                 feedback_file: Optional[Union[str, Path]] = None,
                 metadata_file: Optional[Union[str, Path]] = None,
                 anonymize: bool = True):
        """
        Initialize the data collector.
        
        Args:
            data_dir: Directory for storing training data
            error_file: File for storing error data
            labels_file: File for storing error labels
            feedback_file: File for storing fix feedback
            metadata_file: File for storing dataset metadata
            anonymize: Whether to anonymize sensitive data
        """
        # Set up data directory and files
        self.data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
        self.error_file = Path(error_file) if error_file else self.data_dir / "error_data.jsonl"
        self.labels_file = Path(labels_file) if labels_file else self.data_dir / "error_labels.jsonl"
        self.feedback_file = Path(feedback_file) if feedback_file else self.data_dir / "fix_feedback.jsonl"
        self.metadata_file = Path(metadata_file) if metadata_file else self.data_dir / "dataset_metadata.json"
        
        # Create directories if they don't exist
        self.data_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize the anonymizer if needed
        self.anonymize = anonymize
        self.anonymizer = DataAnonymizer() if anonymize else None
        
        # Initialize dataset metadata
        self._initialize_metadata()
    
    def _initialize_metadata(self):
        """Initialize dataset metadata if it doesn't exist."""
        if not self.metadata_file.exists():
            metadata = {
                "created_at": datetime.datetime.now().isoformat(),
                "updated_at": datetime.datetime.now().isoformat(),
                "error_count": 0,
                "labeled_count": 0,
                "feedback_count": 0,
                "error_types": {},
                "sources": {},
                "version": "1.0"
            }
            
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
    
    def _update_metadata(self, update_func):
        """
        Update metadata with the given update function.
        
        Args:
            update_func: Function that takes metadata dict and updates it
        """
        # Read current metadata
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {
                "created_at": datetime.datetime.now().isoformat(),
                "error_count": 0,
                "labeled_count": 0,
                "feedback_count": 0,
                "error_types": {},
                "sources": {},
                "version": "1.0"
            }
        
        # Update metadata
        update_func(metadata)
        metadata["updated_at"] = datetime.datetime.now().isoformat()
        
        # Write updated metadata
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _generate_error_id(self, error_data: Dict[str, Any]) -> str:
        """
        Generate a unique ID for an error.
        
        Args:
            error_data: Error data dictionary
            
        Returns:
            Unique error ID
        """
        # Extract key components for creating a stable ID
        components = []
        
        # Add exception type and message
        exception_type = error_data.get('exception_type', '')
        if not exception_type and 'error_details' in error_data:
            exception_type = error_data['error_details'].get('exception_type', '')
        components.append(exception_type)
        
        message = error_data.get('message', '')
        if not message and 'error_details' in error_data:
            message = error_data['error_details'].get('message', '')
        components.append(message)
        
        # Add first frame info if available
        traceback = error_data.get('traceback', [])
        if isinstance(traceback, list) and traceback:
            components.append(traceback[0])
        
        # Add service name if available
        service = error_data.get('service', '')
        components.append(service)
        
        # Create a hash from the components
        hash_input = '|'.join(filter(None, components))
        hash_value = hashlib.md5(hash_input.encode('utf-8')).hexdigest()
        
        # Combine with a timestamp to ensure uniqueness
        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        return f"{hash_value[:16]}_{timestamp}"
    
    def add_error(self, error_data: Dict[str, Any], source: str = "unknown") -> str:
        """
        Add an error to the training dataset.
        
        Args:
            error_data: Error data dictionary
            source: Source of the error data
            
        Returns:
            Unique ID for the error
        """
        # Generate an ID for the error
        error_id = self._generate_error_id(error_data)
        
        # Anonymize if needed
        if self.anonymize and self.anonymizer:
            error_data = self.anonymizer.anonymize_error_data(error_data)
        
        # Add metadata
        error_data['_id'] = error_id
        error_data['_source'] = source
        error_data['_collected_at'] = datetime.datetime.now().isoformat()
        
        # Append to error data file
        with open(self.error_file, 'a') as f:
            f.write(json.dumps(error_data) + '\n')
        
        # Update metadata
        def update_metadata(metadata):
            metadata["error_count"] += 1
            
            # Update source count
            if source not in metadata["sources"]:
                metadata["sources"][source] = 0
            metadata["sources"][source] += 1
            
            # Update error type count if available
            error_type = error_data.get('exception_type', '')
            if not error_type and 'error_details' in error_data:
                error_type = error_data['error_details'].get('exception_type', '')
            
            if error_type:
                if error_type not in metadata["error_types"]:
                    metadata["error_types"][error_type] = 0
                metadata["error_types"][error_type] += 1
        
        self._update_metadata(update_metadata)
        
        return error_id
    
    def add_label(self, error_id: str, label: str, confidence: float = 1.0, 
                 labeler: str = "unknown", notes: Optional[str] = None) -> bool:
        """
        Add a label for an error.
        
        Args:
            error_id: ID of the error
            label: Label for the error
            confidence: Confidence in the label (0.0-1.0)
            labeler: Source of the label
            notes: Additional notes about the label
            
        Returns:
            True if successful, False otherwise
        """
        # Check if the error exists
        error_found = False
        if self.error_file.exists():
            with open(self.error_file, 'r') as f:
                for line in f:
                    try:
                        error_data = json.loads(line)
                        if error_data.get('_id') == error_id:
                            error_found = True
                            break
                    except json.JSONDecodeError:
                        continue
        
        if not error_found:
            return False
        
        # Create label entry
        label_entry = {
            'error_id': error_id,
            'label': label,
            'confidence': confidence,
            'labeler': labeler,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        if notes:
            label_entry['notes'] = notes
        
        # Append to labels file
        with open(self.labels_file, 'a') as f:
            f.write(json.dumps(label_entry) + '\n')
        
        # Update metadata
        def update_metadata(metadata):
            metadata["labeled_count"] += 1
        
        self._update_metadata(update_metadata)
        
        return True
    
    def add_feedback(self, error_id: str, fix_successful: bool, 
                     feedback_source: str = "unknown", 
                     details: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add feedback about a fix attempt.
        
        Args:
            error_id: ID of the error
            fix_successful: Whether the fix was successful
            feedback_source: Source of the feedback
            details: Additional details about the feedback
            
        Returns:
            True if successful, False otherwise
        """
        # Check if the error exists
        error_found = False
        if self.error_file.exists():
            with open(self.error_file, 'r') as f:
                for line in f:
                    try:
                        error_data = json.loads(line)
                        if error_data.get('_id') == error_id:
                            error_found = True
                            break
                    except json.JSONDecodeError:
                        continue
        
        if not error_found:
            return False
        
        # Create feedback entry
        feedback_entry = {
            'error_id': error_id,
            'fix_successful': fix_successful,
            'feedback_source': feedback_source,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        if details:
            feedback_entry['details'] = details
        
        # Append to feedback file
        with open(self.feedback_file, 'a') as f:
            f.write(json.dumps(feedback_entry) + '\n')
        
        # Update metadata
        def update_metadata(metadata):
            metadata["feedback_count"] += 1
        
        self._update_metadata(update_metadata)
        
        return True
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collected dataset.
        
        Returns:
            Dictionary with dataset statistics
        """
        if not self.metadata_file.exists():
            return {"error": "Metadata file not found"}
        
        with open(self.metadata_file, 'r') as f:
            return json.load(f)
    
    def get_training_data(self, limit: Optional[int] = None) -> List[Tuple[Dict[str, Any], str]]:
        """
        Get labeled training data for model training.
        
        Args:
            limit: Maximum number of records to return
            
        Returns:
            List of (error_data, label) tuples
        """
        if not self.error_file.exists() or not self.labels_file.exists():
            return []
        
        # Get all labels
        labels_by_error_id = {}
        with open(self.labels_file, 'r') as f:
            for line in f:
                try:
                    label_entry = json.loads(line)
                    error_id = label_entry.get('error_id')
                    label = label_entry.get('label')
                    
                    if error_id and label:
                        # Use the most recent label for each error
                        if error_id not in labels_by_error_id or label_entry.get('timestamp', '') > labels_by_error_id[error_id][1]:
                            labels_by_error_id[error_id] = (label, label_entry.get('timestamp', ''))
                except json.JSONDecodeError:
                    continue
        
        # Get error data for labeled errors
        training_data = []
        with open(self.error_file, 'r') as f:
            for line in f:
                try:
                    error_data = json.loads(line)
                    error_id = error_data.get('_id')
                    
                    if error_id in labels_by_error_id:
                        label = labels_by_error_id[error_id][0]
                        training_data.append((error_data, label))
                        
                        if limit is not None and len(training_data) >= limit:
                            break
                except json.JSONDecodeError:
                    continue
        
        return training_data
    
    def export_dataset(self, export_dir: Optional[Union[str, Path]] = None,
                      format: str = "jsonl") -> str:
        """
        Export the dataset for sharing or backup.
        
        Args:
            export_dir: Directory to export to
            format: Export format (jsonl or json)
            
        Returns:
            Path to the exported dataset
        """
        export_dir = Path(export_dir) if export_dir else self.data_dir / "exports"
        export_dir.mkdir(exist_ok=True, parents=True)
        
        # Create a timestamp for the export
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        export_path = export_dir / f"error_dataset_{timestamp}"
        
        if format == "jsonl":
            # Just copy the files
            os.makedirs(export_path, exist_ok=True)
            
            # Copy all jsonl files
            for src_file in [self.error_file, self.labels_file, self.feedback_file, self.metadata_file]:
                if src_file.exists():
                    dst_file = export_path / src_file.name
                    with open(src_file, 'r') as src, open(dst_file, 'w') as dst:
                        dst.write(src.read())
            
            return str(export_path)
        
        elif format == "json":
            # Convert to a single JSON file
            dataset = {
                "metadata": self.get_dataset_stats(),
                "errors": [],
                "labels": [],
                "feedback": []
            }
            
            # Read errors
            if self.error_file.exists():
                with open(self.error_file, 'r') as f:
                    for line in f:
                        try:
                            dataset["errors"].append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            
            # Read labels
            if self.labels_file.exists():
                with open(self.labels_file, 'r') as f:
                    for line in f:
                        try:
                            dataset["labels"].append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            
            # Read feedback
            if self.feedback_file.exists():
                with open(self.feedback_file, 'r') as f:
                    for line in f:
                        try:
                            dataset["feedback"].append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            
            # Write the combined JSON file
            export_file = f"{export_path}.json"
            with open(export_file, 'w') as f:
                json.dump(dataset, f, indent=2)
            
            return export_file
        
        else:
            raise ValueError(f"Unsupported export format: {format}")


class SyntheticDataGenerator:
    """Generates synthetic training data for model development."""
    
    def __init__(self, error_patterns_file: Optional[Union[str, Path]] = None):
        """
        Initialize the synthetic data generator.
        
        Args:
            error_patterns_file: File with error patterns for generation
        """
        self.error_patterns = self._load_error_patterns(error_patterns_file)
    
    def _load_error_patterns(self, patterns_file: Optional[Union[str, Path]]) -> Dict[str, Any]:
        """
        Load error patterns from file or use defaults.
        
        Args:
            patterns_file: File with error patterns
            
        Returns:
            Dictionary of error patterns
        """
        if patterns_file and Path(patterns_file).exists():
            with open(patterns_file, 'r') as f:
                return json.load(f)
        
        # Use default patterns if no file provided
        return {
            "KeyError": {
                "messages": [
                    "'{key}'",
                    "'{key}' not found in dictionary"
                ],
                "contexts": [
                    "dictionary access",
                    "config lookup",
                    "parameter extraction",
                    "JSON parsing"
                ],
                "variables": ["user_id", "config", "params", "data", "settings", "options"],
                "function_names": ["get_item", "process_request", "lookup_config", "handle_data"]
            },
            "TypeError": {
                "messages": [
                    "'NoneType' object is not subscriptable",
                    "can't convert {type1} to {type2}",
                    "unsupported operand type(s) for {op}: '{type1}' and '{type2}'"
                ],
                "contexts": [
                    "type conversion",
                    "function call",
                    "arithmetic operation",
                    "object attribute access"
                ],
                "variables": ["result", "data", "value", "obj", "response"],
                "function_names": ["process_data", "convert_types", "calculate_value", "get_result"]
            },
            "IndexError": {
                "messages": [
                    "list index out of range",
                    "tuple index out of range"
                ],
                "contexts": [
                    "list access",
                    "array indexing",
                    "iteration",
                    "sequence processing"
                ],
                "variables": ["items", "values", "results", "elements", "list", "array"],
                "function_names": ["get_item", "process_list", "get_element", "iterate_values"]
            },
            "AttributeError": {
                "messages": [
                    "'{obj}' object has no attribute '{attr}'",
                    "module '{module}' has no attribute '{attr}'"
                ],
                "contexts": [
                    "object attribute access",
                    "module import",
                    "API usage",
                    "class instantiation"
                ],
                "variables": ["user", "obj", "instance", "module", "client"],
                "function_names": ["get_attribute", "process_object", "initialize", "setup"]
            },
            "ImportError": {
                "messages": [
                    "No module named '{module}'",
                    "cannot import name '{name}' from '{module}'"
                ],
                "contexts": [
                    "module import",
                    "package installation",
                    "dependency management",
                    "dynamic loading"
                ],
                "variables": ["module", "package", "dependency", "extension"],
                "function_names": ["import_module", "setup", "initialize_dependencies", "load_extension"]
            }
        }
    
    def generate_error(self, error_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a synthetic error for training.
        
        Args:
            error_type: Type of error to generate, or None for random
            
        Returns:
            Synthetic error data dictionary
        """
        import random
        
        # Select error type if not specified
        if error_type is None or error_type not in self.error_patterns:
            error_type = random.choice(list(self.error_patterns.keys()))
        
        pattern = self.error_patterns[error_type]
        
        # Select message template and fill in variables
        message_template = random.choice(pattern["messages"])
        message = message_template
        
        # Fill in message variables
        if "{key}" in message:
            message = message.replace("{key}", random.choice(pattern["variables"]))
        if "{type1}" in message:
            message = message.replace("{type1}", random.choice(["int", "str", "list", "dict", "NoneType"]))
        if "{type2}" in message:
            message = message.replace("{type2}", random.choice(["int", "str", "list", "dict", "float"]))
        if "{op}" in message:
            message = message.replace("{op}", random.choice(["+", "-", "*", "/"]))
        if "{obj}" in message:
            message = message.replace("{obj}", random.choice(pattern["variables"]))
        if "{attr}" in message:
            message = message.replace("{attr}", random.choice(["name", "id", "value", "data", "config"]))
        if "{module}" in message:
            message = message.replace("{module}", random.choice(["requests", "django", "flask", "numpy", "pandas"]))
        if "{name}" in message:
            message = message.replace("{name}", random.choice(["Function", "Class", "module", "variable"]))
        
        # Generate traceback
        function_name = random.choice(pattern["function_names"])
        context = random.choice(pattern["contexts"])
        file_path = f"/app/services/example_service/{random.choice(['app.py', 'utils.py', 'helpers.py', 'models.py'])}"
        line_number = random.randint(10, 200)
        
        # Create error data
        error_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "service": "example_service",
            "level": "ERROR",
            "message": f"{error_type}: {message}",
            "exception_type": error_type,
            "traceback": [
                "Traceback (most recent call last):",
                f"  File '{file_path}', line {line_number}, in {function_name}",
                f"    # Context: {context}",
                f"{error_type}: {message}"
            ],
            "error_details": {
                "exception_type": error_type,
                "message": message,
                "detailed_frames": [
                    {
                        "file": file_path,
                        "line": line_number,
                        "function": function_name,
                        "locals": self._generate_local_variables(error_type, pattern)
                    }
                ]
            },
            "_synthetic": True
        }
        
        return error_data
    
    def _generate_local_variables(self, error_type: str, pattern: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate synthetic local variables for a frame.
        
        Args:
            error_type: Type of error
            pattern: Error pattern
            
        Returns:
            Dictionary of local variables
        """
        import random
        
        variables = {}
        
        # Add some random variables
        for _ in range(random.randint(1, 5)):
            var_name = random.choice(pattern["variables"])
            
            # Generate different types based on the error
            if error_type == "KeyError":
                variables[var_name] = {"item1": "value1", "item2": "value2"}
            elif error_type == "TypeError":
                if random.random() < 0.3:
                    variables[var_name] = None
                else:
                    variables[var_name] = random.choice([123, "string", [], {}])
            elif error_type == "IndexError":
                variables[var_name] = [1, 2, 3] if random.random() < 0.7 else []
            elif error_type == "AttributeError":
                variables[var_name] = {"name": "value"} if random.random() < 0.7 else {}
            else:
                variables[var_name] = random.choice([123, "string", [1, 2, 3], {"key": "value"}])
        
        return variables
    
    def generate_dataset(self, size: int = 100, 
                         output_file: Optional[Union[str, Path]] = None) -> List[Dict[str, Any]]:
        """
        Generate a synthetic dataset for training.
        
        Args:
            size: Number of samples to generate
            output_file: File to save the dataset to
            
        Returns:
            List of generated errors
        """
        errors = []
        
        # Generate errors
        for _ in range(size):
            error = self.generate_error()
            errors.append(error)
        
        # Save to file if specified
        if output_file:
            with open(output_file, 'w') as f:
                for error in errors:
                    f.write(json.dumps(error) + '\n')
        
        return errors


# Utility functions
def setup_training_environment():
    """
    Set up the environment for training data collection.
    
    Returns:
        Path to the training data directory
    """
    data_dir = DEFAULT_DATA_DIR
    data_dir.mkdir(exist_ok=True, parents=True)
    
    # Create subdirectories
    (data_dir / "exports").mkdir(exist_ok=True)
    (data_dir / "models").mkdir(exist_ok=True)
    
    # Create a README file
    readme_path = data_dir / "README.md"
    if not readme_path.exists():
        with open(readme_path, 'w') as f:
            f.write("# Error Analysis Training Data\n\n")
            f.write("This directory contains training data for error analysis models.\n\n")
            f.write("## Files\n\n")
            f.write("- `error_data.jsonl`: Raw error data\n")
            f.write("- `error_labels.jsonl`: Labels for errors\n")
            f.write("- `fix_feedback.jsonl`: Feedback on fix attempts\n")
            f.write("- `dataset_metadata.json`: Metadata about the dataset\n\n")
            f.write("## Usage\n\n")
            f.write("Use the `data_collector.py` module to interact with this data.\n")
    
    return data_dir


if __name__ == "__main__":
    # Set up the environment
    data_dir = setup_training_environment()
    print(f"Training data directory: {data_dir}")
    
    # Create a data collector
    collector = ErrorDataCollector()
    
    # Generate some synthetic data
    generator = SyntheticDataGenerator()
    print("Generating synthetic training data...")
    
    for _ in range(20):
        error = generator.generate_error()
        error_id = collector.add_error(error, source="synthetic")
        
        # Add a label based on the error type
        error_type = error.get('exception_type', '')
        if error_type == "KeyError":
            label = "missing_dictionary_key"
        elif error_type == "TypeError" and "NoneType" in error.get('message', ''):
            label = "none_value_error"
        elif error_type == "IndexError":
            label = "index_out_of_bounds"
        elif error_type == "AttributeError":
            label = "missing_attribute"
        elif error_type == "ImportError":
            label = "missing_module"
        else:
            label = "unknown_error"
        
        collector.add_label(error_id, label, confidence=0.9, labeler="system")
    
    # Print stats
    stats = collector.get_dataset_stats()
    print("\nDataset statistics:")
    print(f"Total errors: {stats['error_count']}")
    print(f"Labeled errors: {stats['labeled_count']}")
    print(f"Error types: {list(stats['error_types'].keys())}")