# Homeostasis Contribution Workflow

This document provides a comprehensive guide for contributing to the Homeostasis project. It outlines the processes, tools, and best practices for community contributions.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Contribution Paths](#contribution-paths)
3. [Development Environment Setup](#development-environment-setup)
4. [Contribution Workflow](#contribution-workflow)
5. [Code Style and Standards](#code-style-and-standards)
6. [Testing Guidelines](#testing-guidelines)
7. [Documentation Guidelines](#documentation-guidelines)
8. [Review Process](#review-process)
9. [Community Communication](#community-communication)
10. [Contribution Recognition](#contribution-recognition)

## Getting Started

### Understanding the Project

Before contributing, familiarize yourself with the Homeostasis project:

1. Read the project [README.md](/README.md) for an overview
2. Review the [architecture.md](/docs/architecture.md) to understand the system design
3. Explore the [usage.md](/docs/usage.md) to see how Homeostasis works
4. Check out the [examples.md](/docs/examples.md) for practical applications
5. Review the [roadmap.md](/docs/roadmap.md) to understand future plans

### Finding Ways to Contribute

There are many ways to contribute to Homeostasis:

1. **Code Contributions**: Implement new features, fix bugs, or improve performance
2. **Documentation**: Enhance or create documentation, tutorials, or examples
3. **Testing**: Write tests, create test scenarios, or validate fixes
4. **Rule Creation**: Develop error detection rules for different languages and frameworks
5. **Template Creation**: Create patch templates for common errors
6. **Bug Reports**: Identify and report issues in the project
7. **Feature Requests**: Suggest new features or improvements

### Issue Tracker

Browse the [GitHub Issues](https://github.com/thehalvo/homeostasis/issues) to find tasks you can help with:

- Issues labeled `good-first-issue` are ideal for newcomers
- Issues labeled `help-wanted` indicate areas where community contributions are particularly needed
- Issues labeled `bug` represent problems that need fixing
- Issues labeled `enhancement` represent new features or improvements

## Contribution Paths

Homeostasis offers several specialized contribution paths, each with its own guidelines:

### Path 1: Core Framework Development

For those interested in the core functionality of the framework:

- Focus on modules like `orchestrator`, `modules/monitoring`, `modules/analysis`, etc.
- Requires strong Python skills and understanding of system architecture
- Contributions go through rigorous review
- [Core Development Guide](/docs/contributing-core.md)

### Path 2: Rules and Detection

For those interested in improving error detection:

- Focus on creating and enhancing rules in `modules/analysis/rules/`
- Requires understanding of error patterns in different languages
- Easier entry point for new contributors
- [Rules Contribution Guide](/docs/contributing-rules.md)

### Path 3: Patch Templates

For those interested in fixing errors:

- Focus on creating templates in `modules/patch_generation/templates/`
- Requires knowledge of code patterns and fixes
- Good understanding of specific frameworks
- [Template Contribution Guide](/docs/contributing-templates.md)

### Path 4: Language/Framework Support

For those interested in extending support to new languages or frameworks:

- Focus on creating adapters in `modules/analysis/plugins/`
- Requires expertise in specific programming languages
- Involves creating language-specific rules and templates
- [Language Support Contribution Guide](/docs/contributing-language.md)

### Path 5: Documentation and Examples

For those interested in improving documentation:

- Focus on files in the `docs/` directory
- Create examples, tutorials, or improve existing documentation
- Lower technical barrier to entry
- [Documentation Contribution Guide](/docs/contributing-docs.md)

## Development Environment Setup

### Prerequisites

Before setting up your development environment, ensure you have:

- Python 3.8 or higher
- Git
- A GitHub account
- Your favorite code editor (VS Code, PyCharm, etc.)

### Setting Up Your Environment

1. **Fork the repository**

   Visit the [Homeostasis GitHub repository](https://github.com/thehalvo/homeostasis) and click the "Fork" button in the top-right corner.

2. **Clone your fork**

   ```bash
   git clone https://github.com/YOUR-USERNAME/homeostasis.git
   cd homeostasis
   ```

3. **Set up the upstream remote**

   ```bash
   git remote add upstream https://github.com/thehalvo/homeostasis.git
   ```

4. **Create a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

5. **Install dependencies**

   ```bash
   pip install -e .
   pip install -e ".[dev]"
   ```

6. **Verify installation**

   ```bash
   python -m pytest tests/
   ```

### Development Tools

We recommend the following tools for development:

- **Linting**: `flake8` for code quality checks
- **Formatting**: `black` for automatic code formatting
- **Type Checking**: `mypy` for static type checking
- **Testing**: `pytest` for running tests

The project includes configuration files for these tools to ensure consistency.

## Contribution Workflow

### Step 1: Find or Create an Issue

All contributions should be associated with an issue:

1. Check if an issue already exists for the change you want to make
2. If not, create a new issue describing the problem or enhancement
3. Comment on the issue to express your interest in working on it
4. Wait for a maintainer to assign the issue to you

### Step 2: Create a Branch

Once you've been assigned an issue, create a branch for your work:

```bash
# Ensure you're on the main branch and up to date
git checkout main
git pull upstream main

# Create a feature branch
git checkout -b feature/issue-123-short-description
```

Follow our branch naming conventions:
- `feature/issue-NUMBER-short-description` for new features
- `bugfix/issue-NUMBER-short-description` for bug fixes
- `docs/issue-NUMBER-short-description` for documentation changes
- `test/issue-NUMBER-short-description` for test additions

### Step 3: Make Your Changes

1. Write code that follows our [code style guidelines](#code-style-and-standards)
2. Include appropriate tests for your changes
3. Update or add documentation as needed
4. Make atomic commits with clear messages:

```bash
git add path/to/changed/files
git commit -m "feat: Add support for SQLAlchemy 2.0 error patterns"
```

Follow conventional commit message format:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `test:` for test additions or modifications
- `refactor:` for code refactoring
- `chore:` for routine tasks, dependency updates, etc.

### Step 4: Test Your Changes

Ensure all tests pass before submitting:

```bash
# Run all tests
python -m pytest

# Run linting
flake8 modules/ tests/

# Run type checking
mypy modules/

# Run specific tests for your feature
python -m pytest tests/path/to/specific_test.py
```

### Step 5: Push Your Changes

Push your branch to your fork:

```bash
git push origin feature/issue-123-short-description
```

### Step 6: Create a Pull Request

1. Go to the [Homeostasis repository](https://github.com/thehalvo/homeostasis)
2. Click "Pull requests" and then "New pull request"
3. Click "compare across forks"
4. Select your fork and branch
5. Click "Create pull request"
6. Fill out the PR template with details about your changes
7. Link the PR to the issue it addresses using the format `Fixes #123`

### Step 7: Respond to Review Comments

1. Maintainers will review your PR and may request changes
2. Make any requested changes in your branch
3. Push the changes to automatically update the PR
4. Respond to reviewer comments on GitHub

### Step 8: PR Approval and Merge

Once your PR has been approved:

1. A maintainer will merge your PR
2. Your contribution will be part of the next release
3. Your GitHub username will be added to the contributors list

## Code Style and Standards

We follow specific coding standards to ensure consistency across the codebase:

### Python Style Guidelines

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with a few exceptions
- Maximum line length of 100 characters
- Use 4 spaces for indentation (no tabs)
- Use docstrings for all functions, classes, and modules
- Follow [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) for docstrings

### Example Function with Proper Style

```python
def calculate_rule_confidence(rule_match: Dict[str, Any], error_context: Dict[str, Any]) -> float:
    """
    Calculate the confidence score for a rule match.

    This function determines how well a rule matches an error based on various
    factors like pattern match quality and context relevance.

    Args:
        rule_match: Dictionary containing rule match information
        error_context: Dictionary containing error context information

    Returns:
        A float between 0 and 1 representing the confidence score

    Raises:
        ValueError: If rule_match or error_context are invalid
    """
    if not rule_match or not error_context:
        raise ValueError("Both rule_match and error_context must be provided")

    # Base confidence from the rule definition
    base_confidence = rule_match.get("confidence", 0.5)

    # Context relevance factor
    context_relevance = _calculate_context_relevance(rule_match, error_context)

    # Pattern match quality
    pattern_quality = _calculate_pattern_quality(rule_match)

    # Combine factors (simple weighted average)
    final_confidence = (0.6 * base_confidence + 
                        0.3 * context_relevance + 
                        0.1 * pattern_quality)

    return min(1.0, max(0.0, final_confidence))
```

### File Layout

Follow this standard layout for Python files:

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module docstring describing the purpose of the module.
"""

# Standard library imports
import os
import sys
from typing import Dict, List, Optional

# Third-party imports
import numpy as np
import pytest

# Project imports
from modules.analysis.rule_based import RuleEngine
from modules.monitoring.logger import MonitoringLogger

# Constants
MAX_RULES = 100
DEFAULT_THRESHOLD = 0.75

# Classes
class RuleSelector:
    """Class docstring."""
    # Class code...

# Functions
def select_best_rule(rules: List[Dict], threshold: float = DEFAULT_THRESHOLD) -> Optional[Dict]:
    """Function docstring."""
    # Function code...

# Main execution (if applicable)
if __name__ == "__main__":
    # Main execution code...
```

### Variable Naming

- Use descriptive names that reflect the purpose of the variable
- Class names should use `CamelCase`
- Function and variable names should use `snake_case`
- Constants should use `UPPER_CASE_WITH_UNDERSCORES`
- Avoid single-letter variable names except in mathematical formulas or as short-lived loop variables

## Testing Guidelines

Our testing approach ensures high code quality and prevents regressions:

### Test Structure

- Tests are located in the `tests/` directory
- Files are named with the `test_` prefix
- Tests are organized to mirror the structure of the code they test
- Each test file should focus on testing a single module or function

### Writing Tests

Use pytest for all tests:

```python
import pytest
from modules.analysis.rule_based import RuleEngine

def test_rule_matching_simple_pattern():
    """Test that simple patterns match correctly."""
    rule_engine = RuleEngine()
    rule = {
        "name": "test_rule",
        "pattern": ".*Error.*",
        "confidence": 0.8
    }
    
    error_message = "KeyError: 'user_id' not found"
    result = rule_engine.match_rule(rule, error_message)
    
    assert result["matched"] is True
    assert result["rule_name"] == "test_rule"
    assert result["confidence"] == 0.8

def test_rule_matching_no_match():
    """Test that non-matching patterns return correctly."""
    rule_engine = RuleEngine()
    rule = {
        "name": "test_rule",
        "pattern": ".*DatabaseError.*",
        "confidence": 0.8
    }
    
    error_message = "KeyError: 'user_id' not found"
    result = rule_engine.match_rule(rule, error_message)
    
    assert result["matched"] is False

@pytest.mark.parametrize("error_message,expected_match", [
    ("TypeError: cannot convert 'str' to 'int'", True),
    ("ValueError: invalid literal for int()", True),
    ("KeyError: 'id' not found", False),
    ("RuntimeError: unexpected error", False),
])
def test_rule_matching_parametrized(error_message, expected_match):
    """Test rule matching with different error messages."""
    rule_engine = RuleEngine()
    rule = {
        "name": "type_conversion_error",
        "pattern": ".*(TypeError|ValueError).*convert.*",
        "confidence": 0.8
    }
    
    result = rule_engine.match_rule(rule, error_message)
    assert result["matched"] is expected_match
```

### Test Coverage

- Aim for at least 80% code coverage
- All new features must have corresponding tests
- All bug fixes must include a test that verifies the fix
- Run coverage reports with `pytest --cov=modules tests/`

### Integration and System Tests

- Unit tests should test individual components
- Integration tests should test component interactions
- System tests should test end-to-end functionality
- Use fixtures to set up test environments

## Documentation Guidelines

Good documentation is essential for an open-source project:

### Types of Documentation

1. **API Documentation**: Docstrings in the code (for developers)
2. **Usage Guides**: How to use the project (for users)
3. **Tutorials**: Step-by-step examples (for newcomers)
4. **Reference Documentation**: Detailed technical information (for power users)
5. **Architecture Documentation**: System design and components (for contributors)

### Writing Documentation

- Write documentation as Markdown files in the `docs/` directory
- Keep documentation up to date with code changes
- Include examples when explaining concepts
- Use clear, concise language
- Structure documentation with headings and lists for readability
- Include diagrams where appropriate to explain complex concepts

### API Documentation

Document all public APIs with comprehensive docstrings:

```python
def analyze_error(error_data: Dict[str, Any], rules: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze an error using the provided rules.
    
    This function takes error data and a list of rules, then attempts to find
    the best matching rule for the error. It returns analysis results including
    the matched rule, confidence score, and suggested fix template.
    
    Args:
        error_data: A dictionary containing error information
                   Required keys: 'message', 'type'
                   Optional keys: 'stack_trace', 'context'
        rules: A list of rule dictionaries
               Each rule should have 'name', 'pattern', and 'confidence' keys
    
    Returns:
        A dictionary with the analysis results:
        {
            'matched_rule': str,  # Name of the matched rule
            'confidence': float,  # Confidence score (0-1)
            'suggested_template': str,  # Template ID for fixing
            'analysis_time': float,  # Time taken in seconds
            'matches': List[Dict],  # All rules that matched
        }
    
    Raises:
        ValueError: If error_data or rules are invalid
        AnalysisError: If analysis fails
    
    Example:
        >>> error = {'message': "KeyError: 'user_id'", 'type': 'KeyError'}
        >>> rules = [{'name': 'key_error', 'pattern': '.*KeyError.*', 'confidence': 0.8}]
        >>> analyze_error(error, rules)
        {
            'matched_rule': 'key_error', 
            'confidence': 0.8,
            'suggested_template': 'dict_access_fix',
            'analysis_time': 0.005,
            'matches': [{'rule': 'key_error', 'confidence': 0.8}]
        }
    """
    # Implementation...
```

## Review Process

All contributions go through a review process to ensure quality:

### PR Requirements

Before a PR can be merged, it must:

1. Pass all automated checks (tests, linting, type checking)
2. Receive approval from at least one maintainer
3. Address all requested changes
4. Match the project's style and standards
5. Include appropriate tests
6. Include or update documentation as needed

### Review Criteria

Reviewers will evaluate contributions based on:

1. **Correctness**: Does it work as intended?
2. **Code Quality**: Is it well-written and maintainable?
3. **Test Coverage**: Are there sufficient tests?
4. **Documentation**: Is it properly documented?
5. **Performance**: Are there any performance concerns?
6. **Security**: Are there any security issues?
7. **Compatibility**: Does it maintain compatibility with the rest of the codebase?

### Responding to Reviews

- Be open to feedback and suggestions
- Respond to all review comments
- Make requested changes or explain why they shouldn't be made
- Be patient, as reviewers are volunteers
- Thank reviewers for their time and feedback

## Community Communication

Effective communication is vital for a successful open-source project:

### Communication Channels

- **GitHub Issues**: Bug reports, feature requests, and task tracking
- **GitHub Discussions**: General questions and community conversations
- **Slack Workspace**: Real-time communication (join via [invite link](https://join.slack.com/t/homeostasis-community/shared_invite/...))
- **Mailing List**: Announcements and long-form discussions
- **Community Meetings**: Bi-weekly video calls (see [calendar](https://calendar.google.com/...))

### Guidelines for Communication

1. **Be Respectful**: Treat everyone with respect and courtesy
2. **Be Clear**: Communicate clearly and provide context
3. **Be Specific**: When reporting issues, provide specific details
4. **Be Patient**: Remember that contributors have different time zones and commitments
5. **Be Constructive**: Provide constructive feedback focused on improvement

### Getting Help

If you need help with a contribution:

1. Check the documentation first
2. Search existing GitHub issues and discussions
3. Ask in the Slack #contributing channel
4. Post a question in GitHub Discussions
5. Contact a maintainer through GitHub

## Contribution Recognition

We value all contributions and recognize contributors in several ways:

### Contributor Recognition

1. **All Contributors List**: All contributors are listed in the [CONTRIBUTORS.md](/CONTRIBUTORS.md) file
2. **Commit Attribution**: Commits are attributed to the author in Git history
3. **Release Notes**: Significant contributions are highlighted in release notes
4. **Contributor Badges**: Specialized badges for different types of contributions (code, docs, rules, etc.)

### Becoming a Maintainer

Active contributors may be invited to become project maintainers:

1. Make regular, high-quality contributions
2. Show expertise in one or more project areas
3. Help review pull requests
4. Assist other contributors
5. Participate in community discussions

Maintainers have additional responsibilities and privileges, including:
- Approving and merging pull requests
- Triaging issues
- Setting project direction
- Representing the project in the community

### Acknowledgment Levels

We recognize different levels of contribution:

1. **Contributor**: Anyone who has had a PR merged
2. **Regular Contributor**: Contributors with at least 5 merged PRs
3. **Expert Contributor**: Contributors who specialize in certain areas of the project
4. **Maintainer**: Contributors who help maintain the project

## Specialized Contribution Guides

For detailed guidance on specific contribution types, refer to:

- [Rules Contribution Guide](/docs/contributing-rules.md)
- [Template Contribution Guide](/docs/contributing-templates.md)
- [Monitoring Extensions Guide](/docs/contributing-monitoring.md)
- [Analysis Components Guide](/docs/contributing-analysis.md)
- [Language Support Guide](/docs/contributing-language.md)
- [Documentation Guide](/docs/contributing-docs.md)

## Conclusion

Thank you for your interest in contributing to Homeostasis! Your contributions help make the project better for everyone. Remember that every contribution, whether it's code, documentation, or a bug report, is valuable.

If you have any questions about the contribution process, don't hesitate to reach out to the community through one of our communication channels.

Happy contributing!