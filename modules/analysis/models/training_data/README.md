# Error Analysis Training Data

This directory contains training data for error analysis models. The data is collected from real error logs and synthetic examples to train machine learning models for error classification and analysis.

## Files

- `error_data.jsonl`: Raw error data in JSONL format
- `error_labels.jsonl`: Labels for errors
- `fix_feedback.jsonl`: Feedback on fix attempts
- `dataset_metadata.json`: Metadata about the dataset

## Data Format

### Error Data

Each error entry contains:
- Error message and type
- Stack traces
- Context information (when available)
- Metadata about collection time and source

### Labels

Labels associate errors with their root causes and include:
- Error ID
- Label (root cause category)
- Confidence
- Labeler information

### Feedback

Feedback entries track fix effectiveness:
- Error ID
- Success status
- Source of feedback
- Detailed notes

## Privacy and Anonymization

All data is automatically anonymized to remove:
- Email addresses
- IP addresses
- API keys and passwords
- File paths
- URLs with sensitive information

## Usage

Use the `data_collector.py` module to interact with and manage this data.

Example:
```python
from modules.analysis.models.data_collector import ErrorDataCollector

# Create a collector
collector = ErrorDataCollector()

# Add an error
error_id = collector.add_error(error_data)

# Add a label
collector.add_label(error_id, "missing_dictionary_key")

# Get statistics
stats = collector.get_dataset_stats()
```