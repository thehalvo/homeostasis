# Git Workflow Integration

This module provides Git workflow integration for the Homeostasis self-healing framework, including:

- Pre-commit hooks for error prevention
- Pull request analysis and suggestion systems
- Branch-aware healing strategies
- Commit message analysis for context
- Commit signing and verification for healing changes

## Components

### Pre-commit Hooks (`pre_commit_hooks.py`)
Implements git pre-commit hooks that:
- Run static analysis on changed files
- Check for common error patterns before commit
- Suggest fixes for detected issues
- Block commits with critical errors (configurable)

### PR Analysis (`pr_analyzer.py`)
Analyzes pull requests to:
- Identify potential issues in PR changes
- Generate healing suggestions for reviewers
- Track healing patterns across branches
- Provide risk assessment for changes

### Branch Strategy (`branch_strategy.py`)
Implements branch-aware healing that:
- Tracks healing effectiveness per branch
- Applies branch-specific healing rules
- Manages healing scope based on branch type
- Coordinates multi-branch healing scenarios

### Commit Analysis (`commit_analyzer.py`)
Analyzes commit messages and metadata to:
- Extract context from commit messages
- Identify related changes across commits
- Build healing context from git history
- Track healing success patterns

### Commit Security (`commit_security.py`)
Handles commit signing and verification:
- Signs healing-generated commits
- Verifies commit authenticity
- Manages GPG keys for automated commits
- Tracks healing audit trail

## Usage

```python
from modules.git_integration import GitIntegration

# Initialize git integration
git_integration = GitIntegration('/path/to/repo')

# Install pre-commit hooks
git_integration.install_pre_commit_hooks()

# Analyze a pull request
pr_analysis = git_integration.analyze_pull_request(pr_number=123)

# Apply branch-aware healing
git_integration.apply_branch_healing(branch='feature/new-feature')
```

## Configuration

Git integration settings are configured in `config.yaml`:

```yaml
git_integration:
  pre_commit:
    enabled: true
    block_on_critical: true
    analyze_changed_files_only: true
  
  pr_analysis:
    enabled: true
    auto_comment: false
    risk_threshold: 0.7
  
  branch_strategy:
    production_branches: ['main', 'master', 'production']
    feature_branch_prefix: 'feature/'
    hotfix_branch_prefix: 'hotfix/'
  
  commit_signing:
    enabled: false
    gpg_key_id: null
    require_verification: false
```