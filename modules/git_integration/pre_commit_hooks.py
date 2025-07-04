"""
Pre-commit hooks for error prevention in Git workflows.

This module implements pre-commit hooks that analyze code changes before they
are committed, detecting potential errors and suggesting fixes to prevent
issues from entering the repository.
"""

import os
import sys
import json
import subprocess
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import tempfile
import shutil

# Add the homeostasis modules to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from modules.analysis.rule_based import RuleBasedAnalyzer
from modules.analysis.language_adapters import LanguageAdapterManager
from modules.patch_generation.patcher import Patcher
from modules.monitoring.logger import HomeostasisLogger


class PreCommitHooks:
    """Implements Git pre-commit hooks for error prevention."""
    
    def __init__(self, repo_path: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize pre-commit hooks.
        
        Args:
            repo_path: Path to the Git repository
            config: Configuration dictionary for pre-commit hooks
        """
        self.repo_path = Path(repo_path)
        self.config = config or self._load_default_config()
        self.logger = HomeostasisLogger(__name__)
        
        # Initialize analysis components
        self.analyzer = RuleBasedAnalyzer()
        self.language_manager = LanguageAdapterManager()
        self.patcher = Patcher()
        
        # Track hook installation status
        self.hooks_installed = self._check_hooks_installed()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration for pre-commit hooks."""
        return {
            'enabled': True,
            'block_on_critical': True,
            'analyze_changed_files_only': True,
            'auto_fix': False,
            'excluded_files': ['.git/', '__pycache__/', '*.pyc', '*.log'],
            'max_file_size_mb': 10,
            'timeout_seconds': 30,
            'supported_languages': [
                'python', 'javascript', 'typescript', 'java', 'go', 
                'rust', 'php', 'ruby', 'csharp', 'swift', 'kotlin'
            ]
        }
    
    def install_hooks(self) -> bool:
        """
        Install pre-commit hooks in the Git repository.
        
        Returns:
            True if hooks were successfully installed
        """
        try:
            hooks_dir = self.repo_path / '.git' / 'hooks'
            hooks_dir.mkdir(exist_ok=True)
            
            # Create pre-commit hook script
            hook_script = self._generate_hook_script()
            hook_path = hooks_dir / 'pre-commit'
            
            with open(hook_path, 'w') as f:
                f.write(hook_script)
            
            # Make hook executable
            os.chmod(hook_path, 0o755)
            
            # Create pre-commit configuration
            self._create_hook_config()
            
            self.hooks_installed = True
            self.logger.info("Pre-commit hooks installed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to install pre-commit hooks: {e}")
            return False
    
    def uninstall_hooks(self) -> bool:
        """
        Uninstall pre-commit hooks from the Git repository.
        
        Returns:
            True if hooks were successfully uninstalled
        """
        try:
            hook_path = self.repo_path / '.git' / 'hooks' / 'pre-commit'
            if hook_path.exists():
                hook_path.unlink()
            
            config_path = self.repo_path / '.homeostasis-precommit.json'
            if config_path.exists():
                config_path.unlink()
            
            self.hooks_installed = False
            self.logger.info("Pre-commit hooks uninstalled successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to uninstall pre-commit hooks: {e}")
            return False
    
    def _check_hooks_installed(self) -> bool:
        """Check if pre-commit hooks are currently installed."""
        hook_path = self.repo_path / '.git' / 'hooks' / 'pre-commit'
        return hook_path.exists()
    
    def _generate_hook_script(self) -> str:
        """Generate the pre-commit hook shell script."""
        python_path = sys.executable
        script_path = Path(__file__).absolute()
        
        return f'''#!/bin/bash
# Homeostasis Pre-commit Hook
# Auto-generated by Homeostasis Git Integration

# Check if Python is available
if ! command -v {python_path} &> /dev/null; then
    echo "Error: Python not found at {python_path}"
    exit 1
fi

# Run the pre-commit analysis
{python_path} "{script_path}" --hook-mode --repo-path "{self.repo_path}"
exit_code=$?

if [ $exit_code -ne 0 ]; then
    echo ""
    echo "Pre-commit analysis failed. Fix the issues above before committing."
    echo "To skip this check, use: git commit --no-verify"
    echo ""
fi

exit $exit_code
'''
    
    def _create_hook_config(self) -> None:
        """Create configuration file for the pre-commit hook."""
        config_path = self.repo_path / '.homeostasis-precommit.json'
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def run_pre_commit_analysis(self) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Run pre-commit analysis on staged files.
        
        Returns:
            Tuple of (success, issues_found)
        """
        if not self.config.get('enabled', True):
            return True, []
        
        try:
            # Get staged files
            staged_files = self._get_staged_files()
            if not staged_files:
                return True, []
            
            # Filter files based on configuration
            filtered_files = self._filter_files(staged_files)
            
            # Analyze each file
            all_issues = []
            critical_issues = []
            
            for file_path in filtered_files:
                issues = self._analyze_file(file_path)
                all_issues.extend(issues)
                
                # Check for critical issues
                critical_issues.extend([
                    issue for issue in issues 
                    if issue.get('severity') == 'critical'
                ])
            
            # Report results
            self._report_analysis_results(all_issues)
            
            # Determine if commit should be blocked
            block_commit = (
                self.config.get('block_on_critical', True) and 
                len(critical_issues) > 0
            )
            
            return not block_commit, all_issues
            
        except Exception as e:
            self.logger.error(f"Pre-commit analysis failed: {e}")
            return False, []
    
    def _get_staged_files(self) -> List[str]:
        """Get list of files staged for commit."""
        try:
            result = subprocess.run(
                ['git', 'diff', '--cached', '--name-only'],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                files = result.stdout.strip().split('\n')
                return [f for f in files if f]  # Filter empty strings
            else:
                self.logger.warning(f"Failed to get staged files: {result.stderr}")
                return []
                
        except subprocess.TimeoutExpired:
            self.logger.warning("Git command timed out")
            return []
        except Exception as e:
            self.logger.error(f"Error getting staged files: {e}")
            return []
    
    def _filter_files(self, files: List[str]) -> List[str]:
        """Filter files based on configuration."""
        filtered = []
        excluded_patterns = self.config.get('excluded_files', [])
        max_size_mb = self.config.get('max_file_size_mb', 10)
        
        for file_path in files:
            full_path = self.repo_path / file_path
            
            # Skip if file doesn't exist (might be deleted)
            if not full_path.exists():
                continue
            
            # Check exclusion patterns
            if any(self._matches_pattern(file_path, pattern) for pattern in excluded_patterns):
                continue
            
            # Check file size
            file_size_mb = full_path.stat().st_size / (1024 * 1024)
            if file_size_mb > max_size_mb:
                self.logger.warning(f"Skipping large file: {file_path} ({file_size_mb:.1f}MB)")
                continue
            
            # Check if language is supported
            if not self._is_supported_language(file_path):
                continue
            
            filtered.append(file_path)
        
        return filtered
    
    def _matches_pattern(self, file_path: str, pattern: str) -> bool:
        """Check if file path matches exclusion pattern."""
        import fnmatch
        return fnmatch.fnmatch(file_path, pattern) or pattern in file_path
    
    def _is_supported_language(self, file_path: str) -> bool:
        """Check if file language is supported for analysis."""
        try:
            language = self.language_manager.detect_language(file_path)
            return language in self.config.get('supported_languages', [])
        except Exception:
            return False
    
    def _analyze_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Analyze a single file for potential issues.
        
        Args:
            file_path: Relative path to the file
            
        Returns:
            List of issues found in the file
        """
        try:
            full_path = self.repo_path / file_path
            
            # Read file content
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Detect language
            language = self.language_manager.detect_language(file_path)
            
            # Run analysis
            issues = self.analyzer.analyze_code(
                content=content,
                language=language,
                file_path=str(full_path)
            )
            
            # Add file context to issues
            for issue in issues:
                issue['file_path'] = file_path
                issue['language'] = language
            
            return issues
            
        except Exception as e:
            self.logger.error(f"Error analyzing file {file_path}: {e}")
            return []
    
    def _report_analysis_results(self, issues: List[Dict[str, Any]]) -> None:
        """Report analysis results to the user."""
        if not issues:
            print("✓ Pre-commit analysis passed - no issues found")
            return
        
        # Group issues by severity
        critical = [i for i in issues if i.get('severity') == 'critical']
        warning = [i for i in issues if i.get('severity') == 'warning']
        info = [i for i in issues if i.get('severity') == 'info']
        
        print("\n" + "="*60)
        print("Homeostasis Pre-commit Analysis Results")
        print("="*60)
        
        if critical:
            print(f"\n🚨 CRITICAL ISSUES ({len(critical)}):")
            for issue in critical:
                self._print_issue(issue)
        
        if warning:
            print(f"\n⚠️  WARNINGS ({len(warning)}):")
            for issue in warning:
                self._print_issue(issue)
        
        if info:
            print(f"\n💡 INFO ({len(info)}):")
            for issue in info:
                self._print_issue(issue)
        
        print("\n" + "="*60)
        
        # Show suggestions if auto-fix is enabled
        if self.config.get('auto_fix', False):
            self._suggest_auto_fixes(issues)
    
    def _print_issue(self, issue: Dict[str, Any]) -> None:
        """Print a single issue in a formatted way."""
        file_path = issue.get('file_path', 'unknown')
        line_num = issue.get('line_number', '?')
        message = issue.get('message', 'No message')
        rule_id = issue.get('rule_id', 'unknown')
        
        print(f"  {file_path}:{line_num} - {message} [{rule_id}]")
        
        if 'suggestion' in issue:
            print(f"    💡 Suggestion: {issue['suggestion']}")
    
    def _suggest_auto_fixes(self, issues: List[Dict[str, Any]]) -> None:
        """Suggest automatic fixes for detected issues."""
        fixable_issues = [i for i in issues if i.get('fixable', False)]
        
        if fixable_issues:
            print(f"\n🔧 {len(fixable_issues)} issues can be automatically fixed")
            print("Run: homeostasis fix --pre-commit")


def main():
    """Main entry point for pre-commit hook execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Homeostasis Pre-commit Hook')
    parser.add_argument('--hook-mode', action='store_true', 
                       help='Run in hook mode (called by Git)')
    parser.add_argument('--repo-path', required=True,
                       help='Path to the Git repository')
    parser.add_argument('--config-file', 
                       help='Path to custom configuration file')
    
    args = parser.parse_args()
    
    # Load configuration
    config = None
    if args.config_file:
        try:
            with open(args.config_file, 'r') as f:
                config = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load config file: {e}")
    
    # Initialize pre-commit hooks
    hooks = PreCommitHooks(args.repo_path, config)
    
    # Run analysis
    success, issues = hooks.run_pre_commit_analysis()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()