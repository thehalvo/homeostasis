"""
GitLab CI Integration for Homeostasis

This module provides integration with GitLab CI to enable automatic
healing during CI/CD pipelines.
"""

import logging
import os
from typing import Any, Dict, List, Optional, cast

import requests
import yaml

logger = logging.getLogger(__name__)


class GitLabCIIntegration:
    """Integration with GitLab CI for automated healing"""

    def __init__(
        self,
        token: Optional[str] = None,
        project_id: Optional[str] = None,
        gitlab_url: str = "https://gitlab.com",
    ):
        """
        Initialize GitLab CI integration

        Args:
            token: GitLab personal access token
            project_id: GitLab project ID
            gitlab_url: GitLab instance URL
        """
        self.token = token or os.getenv("GITLAB_TOKEN")
        self.project_id = project_id or os.getenv("CI_PROJECT_ID")
        self.gitlab_url = gitlab_url.rstrip("/")

        if not self.token:
            raise ValueError("GitLab token required (set GITLAB_TOKEN env var)")
        if not self.project_id:
            raise ValueError("Project ID required (set CI_PROJECT_ID env var)")

        self.headers = {"PRIVATE-TOKEN": self.token, "Content-Type": "application/json"}
        self.api_url = f"{self.gitlab_url}/api/v4"

    def get_pipelines(self, status: str = "failed", per_page: int = 20) -> List[Dict[str, Any]]:
        """
        Get pipelines for the project

        Args:
            status: Pipeline status to filter (failed, success, running)
            per_page: Number of pipelines per page

        Returns:
            List of pipeline data
        """
        url = f"{self.api_url}/projects/{self.project_id}/pipelines"
        params: Dict[str, Any] = {
            "status": status,
            "per_page": per_page,
            "order_by": "updated_at",
            "sort": "desc",
        }

        response = requests.get(url, headers=self.headers, params=params, timeout=30)
        response.raise_for_status()

        return cast(List[Dict[str, Any]], response.json())

    def get_pipeline_jobs(self, pipeline_id: int) -> List[Dict[str, Any]]:
        """
        Get jobs from a pipeline

        Args:
            pipeline_id: Pipeline ID

        Returns:
            List of job data
        """
        url = f"{self.api_url}/projects/{self.project_id}/pipelines/{pipeline_id}/jobs"

        response = requests.get(url, headers=self.headers, timeout=30)
        response.raise_for_status()

        return cast(List[Dict[str, Any]], response.json())

    def get_job_log(self, job_id: int) -> str:
        """
        Get log from a job

        Args:
            job_id: Job ID

        Returns:
            Raw log content
        """
        url = f"{self.api_url}/projects/{self.project_id}/jobs/{job_id}/trace"

        response = requests.get(url, headers=self.headers, timeout=30)
        response.raise_for_status()

        return response.text

    def analyze_pipeline_failure(self, pipeline_id: int) -> Dict:
        """
        Analyze a failed pipeline to identify healing opportunities

        Args:
            pipeline_id: Failed pipeline ID

        Returns:
            Analysis results with potential fixes
        """
        jobs = self.get_pipeline_jobs(pipeline_id)

        analysis: Dict[str, Any] = {
            "pipeline_id": pipeline_id,
            "failed_jobs": [],
            "error_patterns": [],
            "suggested_fixes": [],
            "healing_opportunities": [],
        }

        # Analyze failed jobs
        for job in jobs:
            if job["status"] == "failed":
                try:
                    log = self.get_job_log(job["id"])
                    analysis["failed_jobs"].append(
                        {
                            "job_id": job["id"],
                            "name": job["name"],
                            "stage": job["stage"],
                            "failure_reason": self._extract_failure_reason(job, log),
                            "log_excerpt": self._extract_error_excerpt(log),
                        }
                    )

                    # Extract error patterns from this job's log
                    patterns = self._extract_error_patterns(log)
                    analysis["error_patterns"].extend(patterns)

                except Exception as e:
                    logger.warning(f"Could not get log for job {job['id']}: {e}")

        # Remove duplicate patterns
        analysis["error_patterns"] = list(set(analysis["error_patterns"]))

        # Generate healing suggestions
        analysis["suggested_fixes"] = self._generate_healing_suggestions(analysis)

        return analysis

    def create_healing_pipeline_config(self, template_type: str = "basic") -> str:
        """
        Create a GitLab CI pipeline configuration for automated healing

        Args:
            template_type: Type of healing pipeline (basic, advanced)

        Returns:
            YAML pipeline configuration
        """
        if template_type == "basic":
            return self._create_basic_healing_config()
        elif template_type == "advanced":
            return self._create_advanced_healing_config()
        else:
            raise ValueError(f"Unknown template type: {template_type}")

    def _create_basic_healing_config(self) -> str:
        """Create basic healing pipeline configuration"""
        config = {
            "stages": ["analyze", "heal", "test"],
            "variables": {"HOMEOSTASIS_CONFIDENCE_THRESHOLD": "0.8"},
            "analyze_failures": {
                "stage": "analyze",
                "image": "python:3.9",
                "script": [
                    "pip install homeostasis[gitlab]",
                    "homeostasis analyze-gitlab-pipeline --pipeline-id $CI_PIPELINE_ID --project-id $CI_PROJECT_ID",
                    "homeostasis analyze-gitlab-pipeline --pipeline-id $CI_PIPELINE_ID --project-id $CI_PROJECT_ID --output analysis.json",
                ],
                "artifacts": {"paths": ["analysis.json"], "expire_in": "1 week"},
                "rules": [
                    {
                        "if": '$CI_PIPELINE_SOURCE == "trigger" && $HEALING_TRIGGER == "true"'
                    }
                ],
            },
            "apply_healing": {
                "stage": "heal",
                "image": "python:3.9",
                "script": [
                    "pip install homeostasis[gitlab]",
                    "homeostasis heal --input analysis.json --min-confidence $HOMEOSTASIS_CONFIDENCE_THRESHOLD",
                ],
                "dependencies": ["analyze_failures"],
                "rules": [
                    {
                        "if": '$CI_PIPELINE_SOURCE == "trigger" && $HEALING_TRIGGER == "true"'
                    }
                ],
            },
            "test_fixes": {
                "stage": "test",
                "image": "python:3.9",
                "script": [
                    'if [ -f "requirements.txt" ]; then pip install -r requirements.txt; fi',
                    'if [ -f "package.json" ]; then npm install && npm test; fi',
                    'if [ -f "requirements.txt" ]; then python -m pytest; fi',
                ],
                "dependencies": ["apply_healing"],
                "allow_failure": True,
                "rules": [
                    {
                        "if": '$CI_PIPELINE_SOURCE == "trigger" && $HEALING_TRIGGER == "true"'
                    }
                ],
            },
        }

        return yaml.dump(config, default_flow_style=False, sort_keys=False)

    def _create_advanced_healing_config(self) -> str:
        """Create advanced healing pipeline configuration"""
        config = {
            "stages": [
                "analyze",
                "heal-high-confidence",
                "heal-manual-review",
                "test",
                "deploy-fix",
            ],
            "variables": {
                "HOMEOSTASIS_HIGH_CONFIDENCE": "0.8",
                "HOMEOSTASIS_MEDIUM_CONFIDENCE": "0.6",
            },
            "analyze_failures": {
                "stage": "analyze",
                "image": "python:3.9-slim",
                "before_script": [
                    "apt-get update && apt-get install -y git",
                    "pip install homeostasis[gitlab] requests",
                ],
                "script": [
                    'echo "Analyzing pipeline failures..."',
                    "homeostasis analyze-gitlab-pipeline --pipeline-id $FAILED_PIPELINE_ID --project-id $CI_PROJECT_ID --output analysis.json",
                    "cat analysis.json",
                    "# Extract analysis results",
                    'HEALING_NEEDED=$(jq -r ".healing_recommended" analysis.json)',
                    'CONFIDENCE_SCORE=$(jq -r ".confidence_score" analysis.json)',
                    'echo "HEALING_NEEDED=$HEALING_NEEDED" >> analyze.env',
                    'echo "CONFIDENCE_SCORE=$CONFIDENCE_SCORE" >> analyze.env',
                ],
                "artifacts": {
                    "paths": ["analysis.json"],
                    "reports": {"dotenv": "analyze.env"},
                    "expire_in": "1 week",
                },
                "rules": [
                    {
                        "if": '$CI_PIPELINE_SOURCE == "trigger" && $HEALING_TRIGGER == "true"'
                    }
                ],
            },
            "heal_high_confidence": {
                "stage": "heal-high-confidence",
                "image": "python:3.9-slim",
                "before_script": [
                    "apt-get update && apt-get install -y git",
                    "pip install homeostasis[gitlab]",
                    'git config --global user.email "homeostasis-bot@gitlab.com"',
                    'git config --global user.name "Homeostasis Bot"',
                ],
                "script": [
                    'echo "Applying high-confidence fixes..."',
                    "homeostasis heal --input analysis.json --min-confidence $HOMEOSTASIS_HIGH_CONFIDENCE --auto-commit",
                    'if git diff --quiet HEAD~1; then echo "No changes applied"; else echo "Changes committed automatically"; fi',
                ],
                "dependencies": ["analyze_failures"],
                "rules": [
                    {
                        "if": '$CI_PIPELINE_SOURCE == "trigger" && $HEALING_TRIGGER == "true" && $HEALING_NEEDED == "true"',
                        "when": "manual",
                        "allow_failure": True,
                    }
                ],
            },
            "create_mr_for_review": {
                "stage": "heal-manual-review",
                "image": "python:3.9-slim",
                "before_script": [
                    "apt-get update && apt-get install -y git curl",
                    "pip install homeostasis[gitlab]",
                ],
                "script": [
                    'echo "Creating MR for manual review fixes..."',
                    "homeostasis heal --input analysis.json --min-confidence $HOMEOSTASIS_MEDIUM_CONFIDENCE --suggest-only",
                    "git checkout -b homeostasis/healing-suggestions-$CI_PIPELINE_ID",
                    "homeostasis heal --input analysis.json --min-confidence $HOMEOSTASIS_MEDIUM_CONFIDENCE",
                    "git add .",
                    'git commit -m "Homeostasis: Healing suggestions for pipeline $FAILED_PIPELINE_ID" || true',
                    "git push -u origin homeostasis/healing-suggestions-$CI_PIPELINE_ID",
                    "# Create MR using GitLab API",
                    'curl -X POST "$CI_API_V4_URL/projects/$CI_PROJECT_ID/merge_requests" \\',
                    '  --header "PRIVATE-TOKEN: $GITLAB_TOKEN" \\',
                    '  --data "source_branch=homeostasis/healing-suggestions-$CI_PIPELINE_ID" \\',
                    '  --data "target_branch=$CI_DEFAULT_BRANCH" \\',
                    '  --data "title=🔧 Homeostasis: Healing suggestions (review required)" \\',
                    '  --data "description=Automated healing suggestions for pipeline failures. Please review before merging."',
                ],
                "dependencies": ["analyze_failures"],
                "rules": [
                    {
                        "if": '$CI_PIPELINE_SOURCE == "trigger" && $HEALING_TRIGGER == "true" && $HEALING_NEEDED == "true"'
                    }
                ],
            },
            "test_healing_fixes": {
                "stage": "test",
                "image": "python:3.9",
                "script": [
                    'echo "Testing applied fixes..."',
                    'if [ -f "package.json" ]; then npm install && npm test; fi',
                    'if [ -f "requirements.txt" ]; then pip install -r requirements.txt && python -m pytest; fi',
                    'if [ -f "pom.xml" ]; then mvn test; fi',
                    'if [ -f "go.mod" ]; then go test ./...; fi',
                ],
                "dependencies": ["heal_high_confidence"],
                "allow_failure": True,
                "rules": [
                    {
                        "if": '$CI_PIPELINE_SOURCE == "trigger" && $HEALING_TRIGGER == "true"'
                    }
                ],
            },
        }

        return yaml.dump(config, default_flow_style=False, sort_keys=False)

    def create_healing_trigger(self, failed_pipeline_id: int) -> Dict[str, Any]:
        """
        Trigger a healing pipeline for a failed pipeline

        Args:
            failed_pipeline_id: ID of the failed pipeline to heal

        Returns:
            Trigger response data
        """
        url = f"{self.api_url}/projects/{self.project_id}/trigger/pipeline"

        data = {
            "token": os.getenv("CI_TRIGGER_TOKEN"),
            "ref": "main",  # or whatever the default branch is
            "variables": {
                "HEALING_TRIGGER": "true",
                "FAILED_PIPELINE_ID": str(failed_pipeline_id),
            },
        }

        response = requests.post(url, headers=self.headers, json=data, timeout=30)
        response.raise_for_status()

        return cast(Dict[str, Any], response.json())

    def _extract_failure_reason(self, job: Dict, log: str) -> str:
        """Extract failure reason from job and logs"""
        # Look for common failure indicators in the log
        error_lines = []
        for line in log.split("\n")[-50:]:  # Look at last 50 lines
            if any(
                indicator in line.lower()
                for indicator in ["error", "failed", "exception"]
            ):
                error_lines.append(line.strip())

        if error_lines:
            return "; ".join(error_lines[:3])  # Return first 3 error lines

        return f"Job {job['name']} failed in stage {job['stage']}"

    def _extract_error_excerpt(self, log: str) -> str:
        """Extract a relevant excerpt from error logs"""
        lines = log.split("\n")
        error_context = []

        for i, line in enumerate(lines):
            if any(
                indicator in line.lower()
                for indicator in ["error", "failed", "exception"]
            ):
                # Get some context around the error
                start = max(0, i - 2)
                end = min(len(lines), i + 3)
                error_context.extend(lines[start:end])
                break

        return "\n".join(error_context[:10])  # Limit to 10 lines

    def _extract_error_patterns(self, log: str) -> List[str]:
        """Extract common error patterns from logs"""
        patterns = []

        # Common failure patterns
        error_indicators = [
            "npm ERR!",
            "Error:",
            "Exception:",
            "FAILED:",
            "AssertionError",
            "ModuleNotFoundError",
            "ImportError",
            "SyntaxError",
            "TypeError",
            "exit code 1",
            "Permission denied",
            "Connection refused",
            "Timeout",
        ]

        for line in log.split("\n"):
            for indicator in error_indicators:
                if indicator in line:
                    patterns.append(line.strip())
                    break

        return list(set(patterns))  # Remove duplicates

    def _generate_healing_suggestions(self, analysis: Dict) -> List[Dict]:
        """Generate healing suggestions based on analysis"""
        suggestions = []

        for pattern in analysis.get("error_patterns", []):
            if "ModuleNotFoundError" in pattern or "ImportError" in pattern:
                suggestions.append(
                    {
                        "type": "dependency",
                        "description": "Missing Python dependency detected",
                        "fix": "Add missing dependency to requirements.txt",
                        "confidence": 0.9,
                        "pattern": pattern,
                    }
                )
            elif "npm ERR!" in pattern:
                suggestions.append(
                    {
                        "type": "npm",
                        "description": "NPM installation or execution error",
                        "fix": "Clear npm cache, update package.json, or fix npm scripts",
                        "confidence": 0.7,
                        "pattern": pattern,
                    }
                )
            elif "Permission denied" in pattern:
                suggestions.append(
                    {
                        "type": "permissions",
                        "description": "File permission error",
                        "fix": "Add chmod commands to fix file permissions",
                        "confidence": 0.8,
                        "pattern": pattern,
                    }
                )
            elif "Timeout" in pattern or "Connection refused" in pattern:
                suggestions.append(
                    {
                        "type": "network",
                        "description": "Network connectivity issue",
                        "fix": "Add retry logic or adjust timeout settings",
                        "confidence": 0.6,
                        "pattern": pattern,
                    }
                )
            elif "exit code 1" in pattern:
                suggestions.append(
                    {
                        "type": "script",
                        "description": "Script execution failure",
                        "fix": "Review script logic and error handling",
                        "confidence": 0.5,
                        "pattern": pattern,
                    }
                )

        return suggestions

    def setup_project_integration(self, create_files: bool = True) -> Dict:
        """
        Set up complete GitLab project integration

        Args:
            create_files: Whether to create integration files

        Returns:
            Setup status information
        """
        result = {
            "healing_config_created": False,
            "advanced_config_created": False,
            "integration_complete": False,
        }

        if create_files:
            # Create basic healing configuration
            with open(".gitlab-ci-healing.yml", "w") as f:
                f.write(self.create_healing_pipeline_config("basic"))
            result["healing_config_created"] = True

            # Create advanced healing configuration
            with open(".gitlab-ci-healing-advanced.yml", "w") as f:
                f.write(self.create_healing_pipeline_config("advanced"))
            result["advanced_config_created"] = True

            logger.info("GitLab CI integration files created successfully")

        result["integration_complete"] = True
        return result
