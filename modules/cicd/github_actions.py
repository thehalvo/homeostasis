"""
GitHub Actions Integration for Homeostasis

This module provides integration with GitHub Actions to enable automatic
healing during CI/CD workflows.
"""

import logging
import os
from typing import Dict, List, Optional

import requests
import yaml

logger = logging.getLogger(__name__)


class GitHubActionsIntegration:
    """Integration with GitHub Actions for automated healing"""

    def __init__(self, token: Optional[str] = None, repo: Optional[str] = None):
        """
        Initialize GitHub Actions integration

        Args:
            token: GitHub personal access token
            repo: Repository in format 'owner/repo'
        """
        self.token = token or os.getenv("GITHUB_TOKEN")
        self.repo = repo or os.getenv("GITHUB_REPOSITORY")

        if not self.token:
            raise ValueError("GitHub token required (set GITHUB_TOKEN env var)")
        if not self.repo:
            raise ValueError("Repository required (set GITHUB_REPOSITORY env var)")

        self.headers = {
            "Authorization": f"token {self.token}",
            "Accept": "application/vnd.github.v3+json",
            "Content-Type": "application/json",
        }
        self.base_url = "https://api.github.com"

    def get_workflow_runs(
        self, workflow_id: Optional[str] = None, status: str = "failure"
    ) -> List[Dict]:
        """
        Get workflow runs for the repository

        Args:
            workflow_id: Specific workflow ID to filter
            status: Run status to filter (failure, success, in_progress)

        Returns:
            List of workflow run data
        """
        url = f"{self.base_url}/repos/{self.repo}/actions/runs"
        params = {"status": status, "per_page": 50}

        if workflow_id:
            params["workflow_id"] = workflow_id

        response = requests.get(url, headers=self.headers, params=params, timeout=30)
        response.raise_for_status()

        return response.json().get("workflow_runs", [])

    def get_workflow_run_logs(self, run_id: int) -> str:
        """
        Get logs from a workflow run

        Args:
            run_id: Workflow run ID

        Returns:
            Raw log content
        """
        url = f"{self.base_url}/repos/{self.repo}/actions/runs/{run_id}/logs"

        response = requests.get(url, headers=self.headers, timeout=30)
        response.raise_for_status()

        return response.text

    def get_workflow_run_jobs(self, run_id: int) -> List[Dict]:
        """
        Get jobs from a workflow run

        Args:
            run_id: Workflow run ID

        Returns:
            List of job data
        """
        url = f"{self.base_url}/repos/{self.repo}/actions/runs/{run_id}/jobs"

        response = requests.get(url, headers=self.headers, timeout=30)
        response.raise_for_status()

        return response.json().get("jobs", [])

    def analyze_workflow_failure(self, run_id: int) -> Dict:
        """
        Analyze a failed workflow run to identify healing opportunities

        Args:
            run_id: Failed workflow run ID

        Returns:
            Analysis results with potential fixes
        """
        jobs = self.get_workflow_run_jobs(run_id)
        logs = self.get_workflow_run_logs(run_id)

        analysis = {
            "run_id": run_id,
            "failed_jobs": [],
            "error_patterns": [],
            "suggested_fixes": [],
            "healing_opportunities": [],
        }

        # Analyze failed jobs
        for job in jobs:
            if job["conclusion"] == "failure":
                analysis["failed_jobs"].append(
                    {
                        "job_id": job["id"],
                        "name": job["name"],
                        "failure_reason": self._extract_failure_reason(job, logs),
                    }
                )

        # Extract common error patterns
        analysis["error_patterns"] = self._extract_error_patterns(logs)

        # Generate healing suggestions
        analysis["suggested_fixes"] = self._generate_healing_suggestions(analysis)

        return analysis

    def create_healing_workflow(self, template_type: str = "basic") -> str:
        """
        Create a GitHub Actions workflow file for automated healing

        Args:
            template_type: Type of healing workflow (basic, advanced)

        Returns:
            YAML workflow content
        """
        if template_type == "basic":
            return self._create_basic_healing_workflow()
        elif template_type == "advanced":
            return self._create_advanced_healing_workflow()
        else:
            raise ValueError(f"Unknown template type: {template_type}")

    def _create_basic_healing_workflow(self) -> str:
        """Create basic healing workflow"""
        workflow = {
            "name": "Homeostasis Healing",
            "on": {"workflow_run": {"workflows": ["CI"], "types": ["completed"]}},
            "jobs": {
                "heal-on-failure": {
                    "if": "${{ github.event.workflow_run.conclusion == 'failure' }}",
                    "runs-on": "ubuntu-latest",
                    "steps": [
                        {"name": "Checkout code", "uses": "actions/checkout@v3"},
                        {
                            "name": "Setup Python",
                            "uses": "actions/setup-python@v4",
                            "with": {"python-version": "3.9"},
                        },
                        {
                            "name": "Install Homeostasis",
                            "run": "pip install homeostasis",
                        },
                        {
                            "name": "Analyze failure and apply healing",
                            "run": "homeostasis heal --workflow-run-id ${{ github.event.workflow_run.id }}",
                            "env": {"GITHUB_TOKEN": "${{ secrets.GITHUB_TOKEN }}"},
                        },
                        {
                            "name": "Create PR with fixes",
                            "uses": "peter-evans/create-pull-request@v5",
                            "with": {
                                "token": "${{ secrets.GITHUB_TOKEN }}",
                                "commit-message": "Auto-heal: Fix workflow failures",
                                "title": "Automated healing fixes",
                                "body": "This PR contains automated fixes generated by Homeostasis.",
                                "branch": "homeostasis/auto-heal",
                            },
                        },
                    ],
                }
            },
        }

        return yaml.dump(workflow, default_flow_style=False, sort_keys=False)

    def _create_advanced_healing_workflow(self) -> str:
        """Create advanced healing workflow with multiple strategies"""
        workflow = {
            "name": "Homeostasis Advanced Healing",
            "on": {
                "workflow_run": {
                    "workflows": ["CI", "Build", "Test"],
                    "types": ["completed"],
                },
                "schedule": [{"cron": "0 */6 * * *"}],  # Run every 6 hours
            },
            "jobs": {
                "analyze-failures": {
                    "if": "${{ github.event.workflow_run.conclusion == 'failure' || github.event_name == 'schedule' }}",
                    "runs-on": "ubuntu-latest",
                    "outputs": {
                        "healing-needed": "${{ steps.analysis.outputs.healing-needed }}",
                        "confidence-score": "${{ steps.analysis.outputs.confidence-score }}",
                    },
                    "steps": [
                        {"name": "Checkout code", "uses": "actions/checkout@v3"},
                        {
                            "name": "Setup Python",
                            "uses": "actions/setup-python@v4",
                            "with": {"python-version": "3.9"},
                        },
                        {
                            "name": "Install Homeostasis",
                            "run": "pip install homeostasis[github]",
                        },
                        {
                            "name": "Analyze failures",
                            "id": "analysis",
                            "run": """
                                result=$(homeostasis analyze --platform github --repo ${{ github.repository }})
                                echo "healing-needed=$(echo $result | jq -r '.healing_needed')" >> $GITHUB_OUTPUT
                                echo "confidence-score=$(echo $result | jq -r '.confidence_score')" >> $GITHUB_OUTPUT
                            """,
                            "env": {"GITHUB_TOKEN": "${{ secrets.GITHUB_TOKEN }}"},
                        },
                    ],
                },
                "apply-high-confidence-fixes": {
                    "needs": "analyze-failures",
                    "if": "${{ needs.analyze-failures.outputs.healing-needed == 'true' && needs.analyze-failures.outputs.confidence-score > 0.8 }}",
                    "runs-on": "ubuntu-latest",
                    "steps": [
                        {"name": "Checkout code", "uses": "actions/checkout@v3"},
                        {
                            "name": "Apply automatic fixes",
                            "run": "homeostasis heal --auto-apply --min-confidence 0.8",
                            "env": {"GITHUB_TOKEN": "${{ secrets.GITHUB_TOKEN }}"},
                        },
                        {
                            "name": "Run tests to verify fixes",
                            "run": "npm test || python -m pytest || ./gradlew test",
                        },
                        {
                            "name": "Commit and push fixes",
                            "run": """
                                git config --local user.email "action@github.com"
                                git config --local user.name "Homeostasis Bot"
                                git add .
                                git commit -m "Auto-heal: Apply high-confidence fixes" || exit 0
                                git push
                            """,
                        },
                    ],
                },
                "create-pr-for-manual-review": {
                    "needs": "analyze-failures",
                    "if": "${{ needs.analyze-failures.outputs.healing-needed == 'true' && needs.analyze-failures.outputs.confidence-score <= 0.8 }}",
                    "runs-on": "ubuntu-latest",
                    "steps": [
                        {"name": "Checkout code", "uses": "actions/checkout@v3"},
                        {
                            "name": "Generate healing suggestions",
                            "run": "homeostasis heal --suggest-only --output-format json > healing-suggestions.json",
                        },
                        {
                            "name": "Create PR with suggestions",
                            "uses": "peter-evans/create-pull-request@v5",
                            "with": {
                                "token": "${{ secrets.GITHUB_TOKEN }}",
                                "commit-message": "Homeostasis: Healing suggestions",
                                "title": "Healing suggestions (manual review required)",
                                "body-path": "healing-suggestions.json",
                                "branch": "homeostasis/suggestions",
                            },
                        },
                    ],
                },
            },
        }

        return yaml.dump(workflow, default_flow_style=False, sort_keys=False)

    def _extract_failure_reason(self, job: Dict, logs: str) -> str:
        """Extract failure reason from job and logs"""
        # This would implement log parsing logic
        return f"Job {job['name']} failed during step analysis"

    def _extract_error_patterns(self, logs: str) -> List[str]:
        """Extract common error patterns from logs"""
        patterns = []

        # Common failure patterns
        error_indicators = [
            "npm ERR!",
            "Error:",
            "Exception:",
            "FAIL:",
            "AssertionError",
            "ModuleNotFoundError",
            "ImportError",
            "SyntaxError",
            "TypeError",
        ]

        for line in logs.split("\n"):
            for indicator in error_indicators:
                if indicator in line:
                    patterns.append(line.strip())
                    break

        return list(set(patterns))  # Remove duplicates

    def _generate_healing_suggestions(self, analysis: Dict) -> List[Dict]:
        """Generate healing suggestions based on analysis"""
        suggestions = []

        for pattern in analysis.get("error_patterns", []):
            if "ModuleNotFoundError" in pattern:
                suggestions.append(
                    {
                        "type": "dependency",
                        "description": "Missing dependency detected",
                        "fix": "Add missing dependency to requirements.txt or package.json",
                        "confidence": 0.9,
                    }
                )
            elif "npm ERR!" in pattern:
                suggestions.append(
                    {
                        "type": "npm",
                        "description": "NPM installation error",
                        "fix": "Clear npm cache and retry installation",
                        "confidence": 0.7,
                    }
                )
            elif "TypeError" in pattern:
                suggestions.append(
                    {
                        "type": "code",
                        "description": "Type error in code",
                        "fix": "Review and fix type mismatches",
                        "confidence": 0.6,
                    }
                )

        return suggestions

    def create_action_files(self, output_dir: str = ".github/actions/homeostasis"):
        """
        Create GitHub Action files for Homeostasis integration

        Args:
            output_dir: Directory to create action files
        """
        os.makedirs(output_dir, exist_ok=True)

        # Create action.yml
        action_yml = {
            "name": "Homeostasis Healing Action",
            "description": "Automatically analyze and heal code issues",
            "inputs": {
                "github-token": {
                    "description": "GitHub token for API access",
                    "required": True,
                },
                "confidence-threshold": {
                    "description": "Minimum confidence score for automatic healing",
                    "required": False,
                    "default": "0.8",
                },
                "languages": {
                    "description": "Comma-separated list of languages to analyze",
                    "required": False,
                    "default": "python,javascript,java,go",
                },
            },
            "outputs": {
                "healing-applied": {"description": "Whether healing was applied"},
                "fixes-count": {"description": "Number of fixes applied"},
            },
            "runs": {"using": "docker", "image": "Dockerfile"},
        }

        with open(f"{output_dir}/action.yml", "w") as f:
            yaml.dump(action_yml, f, default_flow_style=False)

        # Create Dockerfile
        dockerfile_content = """FROM python:3.9-slim

RUN pip install homeostasis[github] requests

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
"""

        with open(f"{output_dir}/Dockerfile", "w") as f:
            f.write(dockerfile_content)

        # Create entrypoint script
        entrypoint_content = """#!/bin/bash
set -e

# Parse inputs
GITHUB_TOKEN="${INPUT_GITHUB_TOKEN}"
CONFIDENCE_THRESHOLD="${INPUT_CONFIDENCE_THRESHOLD:-0.8}"
LANGUAGES="${INPUT_LANGUAGES:-python,javascript,java,go}"

export GITHUB_TOKEN

echo "Starting Homeostasis healing process..."
echo "Confidence threshold: $CONFIDENCE_THRESHOLD"
echo "Languages: $LANGUAGES"

# Run healing analysis
result=$(homeostasis heal \\
    --platform github \\
    --confidence-threshold "$CONFIDENCE_THRESHOLD" \\
    --languages "$LANGUAGES" \\
    --output-format json)

# Parse results
healing_applied=$(echo "$result" | jq -r '.healing_applied // false')
fixes_count=$(echo "$result" | jq -r '.fixes_count // 0')

# Set outputs
echo "healing-applied=$healing_applied" >> $GITHUB_OUTPUT
echo "fixes-count=$fixes_count" >> $GITHUB_OUTPUT

echo "Homeostasis healing completed successfully"
echo "Healing applied: $healing_applied"
echo "Fixes count: $fixes_count"
"""

        with open(f"{output_dir}/entrypoint.sh", "w") as f:
            f.write(entrypoint_content)

    def setup_repository_integration(self):
        """
        Set up complete GitHub repository integration
        """
        # Create workflow files
        workflows_dir = ".github/workflows"
        os.makedirs(workflows_dir, exist_ok=True)

        # Basic healing workflow
        with open(f"{workflows_dir}/homeostasis-healing.yml", "w") as f:
            f.write(self.create_healing_workflow("basic"))

        # Advanced healing workflow
        with open(f"{workflows_dir}/homeostasis-advanced.yml", "w") as f:
            f.write(self.create_healing_workflow("advanced"))

        # Create action files
        self.create_action_files()

        logger.info("GitHub Actions integration files created successfully")
        return {
            "workflows_created": 2,
            "action_files_created": 3,
            "integration_complete": True,
        }
