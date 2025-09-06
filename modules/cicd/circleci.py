"""
CircleCI Integration for Homeostasis

This module provides integration with CircleCI to enable automatic
healing during CI/CD workflows via orbs and API integration.
"""

import logging
import os
from typing import Dict, List, Optional

import requests
import yaml

logger = logging.getLogger(__name__)


class CircleCIIntegration:
    """Integration with CircleCI for automated healing"""

    def __init__(
        self, api_token: Optional[str] = None, project_slug: Optional[str] = None
    ):
        """
        Initialize CircleCI integration

        Args:
            api_token: CircleCI API token
            project_slug: Project slug in format 'vcs-type/org-name/repo-name'
        """
        self.api_token = api_token or os.getenv("CIRCLE_TOKEN")
        self.project_slug = project_slug or self._get_project_slug_from_env()

        if not self.api_token:
            raise ValueError("CircleCI API token required (set CIRCLE_TOKEN env var)")
        if not self.project_slug:
            raise ValueError(
                "Project slug required (format: vcs-type/org-name/repo-name)"
            )

        self.headers = {
            "Circle-Token": self.api_token,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        self.api_url = "https://circleci.com/api/v2"

    def _get_project_slug_from_env(self) -> Optional[str]:
        """Extract project slug from CircleCI environment variables"""
        if os.getenv("CIRCLE_PROJECT_REPONAME") and os.getenv(
            "CIRCLE_PROJECT_USERNAME"
        ):
            vcs_type = "github"  # Default to github, could be 'bitbucket'
            return f"{vcs_type}/{os.getenv('CIRCLE_PROJECT_USERNAME')}/{os.getenv('CIRCLE_PROJECT_REPONAME')}"
        return None

    def get_pipeline_workflows(self, pipeline_id: str) -> List[Dict]:
        """
        Get workflows for a pipeline

        Args:
            pipeline_id: Pipeline ID

        Returns:
            List of workflow data
        """
        url = f"{self.api_url}/pipeline/{pipeline_id}/workflow"

        response = requests.get(url, headers=self.headers, timeout=30)
        response.raise_for_status()

        return response.json().get("items", [])

    def get_workflow_jobs(self, workflow_id: str) -> List[Dict]:
        """
        Get jobs for a workflow

        Args:
            workflow_id: Workflow ID

        Returns:
            List of job data
        """
        url = f"{self.api_url}/workflow/{workflow_id}/job"

        response = requests.get(url, headers=self.headers, timeout=30)
        response.raise_for_status()

        return response.json().get("items", [])

    def get_project_pipelines(self, branch: Optional[str] = None) -> List[Dict]:
        """
        Get recent pipelines for the project

        Args:
            branch: Branch to filter pipelines

        Returns:
            List of pipeline data
        """
        url = f"{self.api_url}/project/{self.project_slug}/pipeline"
        params = {}

        if branch:
            params["branch"] = branch

        response = requests.get(url, headers=self.headers, params=params, timeout=30)
        response.raise_for_status()

        return response.json().get("items", [])

    def analyze_workflow_failure(self, workflow_id: str) -> Dict:
        """
        Analyze a failed workflow to identify healing opportunities

        Args:
            workflow_id: Failed workflow ID

        Returns:
            Analysis results with potential fixes
        """
        jobs = self.get_workflow_jobs(workflow_id)

        analysis = {
            "workflow_id": workflow_id,
            "failed_jobs": [],
            "error_patterns": [],
            "suggested_fixes": [],
            "healing_opportunities": [],
        }

        # Analyze failed jobs
        for job in jobs:
            if job["status"] == "failed":
                analysis["failed_jobs"].append(
                    {
                        "job_number": job["job_number"],
                        "name": job["name"],
                        "failure_reason": self._extract_failure_reason(job),
                        "exit_code": job.get("latest_workflow", {}).get("status"),
                    }
                )

        # For full error analysis, we'd need to fetch job details and logs
        # This is a simplified version
        analysis["suggested_fixes"] = self._generate_healing_suggestions(analysis)

        return analysis

    def create_orb_config(self) -> Dict:
        """
        Create CircleCI orb configuration for Homeostasis

        Returns:
            Orb configuration dictionary
        """
        orb_config = {
            "version": "2.1",
            "description": "Homeostasis healing orb for automated error fixing",
            "display": {
                "home_url": "https://github.com/homeostasis-framework/homeostasis",
                "source_url": "https://github.com/homeostasis-framework/homeostasis",
            },
            "executors": {
                "homeostasis": {
                    "docker": [{"image": "python:3.9-slim"}],
                    "working_directory": "~/project",
                }
            },
            "commands": {
                "install": {
                    "description": "Install Homeostasis healing framework",
                    "parameters": {
                        "version": {
                            "type": "string",
                            "default": "latest",
                            "description": "Version of Homeostasis to install",
                        }
                    },
                    "steps": [
                        "run: apt-get update && apt-get install -y git",
                        {
                            "run": {
                                "name": "Install Homeostasis",
                                "command": """
                                    if [ "<< parameters.version >>" = "latest" ]; then
                                        pip install homeostasis[circleci]
                                    else
                                        pip install homeostasis[circleci]==<< parameters.version >>
                                    fi
                                """,
                            }
                        },
                    ],
                },
                "analyze": {
                    "description": "Analyze CircleCI workflow failures",
                    "parameters": {
                        "workflow-id": {
                            "type": "string",
                            "description": "Workflow ID to analyze",
                        },
                        "output-file": {
                            "type": "string",
                            "default": "homeostasis-analysis.json",
                            "description": "Output file for analysis results",
                        },
                    },
                    "steps": [
                        {
                            "run": {
                                "name": "Analyze workflow failure",
                                "command": """
                                    homeostasis analyze-circleci-workflow \\
                                        --workflow-id "<< parameters.workflow-id >>" \\
                                        --project-slug "$CIRCLE_PROJECT_USERNAME/$CIRCLE_PROJECT_REPONAME" \\
                                        --output "<< parameters.output-file >>"
                                """,
                            }
                        }
                    ],
                },
                "heal": {
                    "description": "Apply Homeostasis healing fixes",
                    "parameters": {
                        "analysis-file": {
                            "type": "string",
                            "default": "homeostasis-analysis.json",
                            "description": "Analysis file to use for healing",
                        },
                        "confidence-threshold": {
                            "type": "string",
                            "default": "0.8",
                            "description": "Minimum confidence for auto-healing",
                        },
                        "auto-apply": {
                            "type": "boolean",
                            "default": True,
                            "description": "Automatically apply high-confidence fixes",
                        },
                    },
                    "steps": [
                        {
                            "run": {
                                "name": "Apply healing fixes",
                                "command": """
                                    APPLY_FLAG=""
                                    if [ "<< parameters.auto-apply >>" = "true" ]; then
                                        APPLY_FLAG="--auto-apply"
                                    fi
                                    
                                    homeostasis heal \\
                                        --input "<< parameters.analysis-file >>" \\
                                        --min-confidence "<< parameters.confidence-threshold >>" \\
                                        $APPLY_FLAG \\
                                        --output healing-results.json
                                """,
                            }
                        }
                    ],
                },
                "test-fixes": {
                    "description": "Test applied healing fixes",
                    "parameters": {
                        "test-command": {
                            "type": "string",
                            "default": "",
                            "description": "Custom test command to run",
                        }
                    },
                    "steps": [
                        {
                            "run": {
                                "name": "Test healing fixes",
                                "command": """
                                    if [ -n "<< parameters.test-command >>" ]; then
                                        << parameters.test-command >>
                                    else
                                        # Auto-detect test framework
                                        if [ -f "package.json" ]; then
                                            npm test
                                        elif [ -f "requirements.txt" ] || [ -f "pyproject.toml" ]; then
                                            python -m pytest
                                        elif [ -f "go.mod" ]; then
                                            go test ./...
                                        else
                                            echo "No test framework detected"
                                        fi
                                    fi
                                """,
                            }
                        }
                    ],
                },
            },
            "jobs": {
                "analyze-and-heal": {
                    "description": "Complete analysis and healing workflow",
                    "executor": "homeostasis",
                    "parameters": {
                        "workflow-id": {
                            "type": "string",
                            "description": "Failed workflow ID to analyze",
                        },
                        "confidence-threshold": {
                            "type": "string",
                            "default": "0.8",
                            "description": "Confidence threshold for healing",
                        },
                        "run-tests": {
                            "type": "boolean",
                            "default": True,
                            "description": "Run tests after healing",
                        },
                    },
                    "steps": [
                        "checkout",
                        "install",
                        {"analyze": {"workflow-id": "<< parameters.workflow-id >>"}},
                        {
                            "heal": {
                                "confidence-threshold": "<< parameters.confidence-threshold >>"
                            }
                        },
                        {
                            "when": {
                                "condition": "<< parameters.run-tests >>",
                                "steps": ["test-fixes"],
                            }
                        },
                        {"store_artifacts": {"path": "homeostasis-analysis.json"}},
                        {"store_artifacts": {"path": "healing-results.json"}},
                    ],
                },
                "healing-pipeline": {
                    "description": "Full healing pipeline with multiple strategies",
                    "executor": "homeostasis",
                    "parameters": {
                        "failed-workflow-id": {
                            "type": "string",
                            "description": "ID of the failed workflow to heal",
                        }
                    },
                    "steps": [
                        "checkout",
                        "install",
                        {
                            "run": {
                                "name": "Setup git configuration",
                                "command": """
                                    git config --global user.email "circleci@homeostasis.bot"
                                    git config --global user.name "CircleCI Homeostasis Bot"
                                """,
                            }
                        },
                        {
                            "analyze": {
                                "workflow-id": "<< parameters.failed-workflow-id >>"
                            }
                        },
                        {
                            "run": {
                                "name": "Determine healing strategy",
                                "command": """
                                    CONFIDENCE=$(jq -r '.confidence_score // 0' homeostasis-analysis.json)
                                    HEALING_NEEDED=$(jq -r '.healing_recommended // false' homeostasis-analysis.json)
                                    
                                    echo "export HEALING_NEEDED=$HEALING_NEEDED" >> $BASH_ENV
                                    echo "export CONFIDENCE_SCORE=$CONFIDENCE" >> $BASH_ENV
                                    
                                    if [ "$HEALING_NEEDED" = "true" ]; then
                                        if (( $(echo "$CONFIDENCE >= 0.8" | bc -l) )); then
                                            echo "export HEALING_STRATEGY=auto" >> $BASH_ENV
                                        elif (( $(echo "$CONFIDENCE >= 0.6" | bc -l) )); then
                                            echo "export HEALING_STRATEGY=manual" >> $BASH_ENV
                                        else
                                            echo "export HEALING_STRATEGY=suggest" >> $BASH_ENV
                                        fi
                                    else
                                        echo "export HEALING_STRATEGY=none" >> $BASH_ENV
                                    fi
                                """,
                            }
                        },
                        {
                            "when": {
                                "condition": {"equal": ["${HEALING_STRATEGY}", "auto"]},
                                "steps": [
                                    {
                                        "heal": {
                                            "confidence-threshold": "0.8",
                                            "auto-apply": True,
                                        }
                                    },
                                    "test-fixes",
                                    {
                                        "run": {
                                            "name": "Commit and push auto-fixes",
                                            "command": """
                                                if ! git diff --quiet; then
                                                    git add .
                                                    git commit -m "🔧 Auto-heal: CircleCI workflow fixes
                                                    
                                                    Fixed workflow: << parameters.failed-workflow-id >>
                                                    Confidence: ${CONFIDENCE_SCORE}
                                                    Applied by Homeostasis CircleCI orb"
                                                    git push origin HEAD
                                                fi
                                            """,
                                        }
                                    },
                                ],
                            }
                        },
                        {
                            "when": {
                                "condition": {
                                    "or": [
                                        {"equal": ["${HEALING_STRATEGY}", "manual"]},
                                        {"equal": ["${HEALING_STRATEGY}", "suggest"]},
                                    ]
                                },
                                "steps": [
                                    {
                                        "heal": {
                                            "confidence-threshold": "0.6",
                                            "auto-apply": False,
                                        }
                                    },
                                    {
                                        "run": {
                                            "name": "Create pull request for manual review",
                                            "command": """
                                                BRANCH_NAME="homeostasis/circleci-healing-${CIRCLE_BUILD_NUM}"
                                                git checkout -b "$BRANCH_NAME"
                                                
                                                # Apply suggestions
                                                homeostasis apply-suggestions --input healing-results.json
                                                
                                                if ! git diff --quiet; then
                                                    git add .
                                                    git commit -m "🔧 Homeostasis: Healing suggestions for CircleCI
                                                    
                                                    Workflow: << parameters.failed-workflow-id >>
                                                    Confidence: ${CONFIDENCE_SCORE}
                                                    Strategy: ${HEALING_STRATEGY}
                                                    
                                                    Please review before merging."
                                                    git push origin "$BRANCH_NAME"
                                                    
                                                    # Create PR using GitHub CLI if available
                                                    if command -v gh >/dev/null 2>&1; then
                                                        gh pr create \\
                                                            --title "🔧 Homeostasis: CircleCI healing suggestions" \\
                                                            --body "Automated healing suggestions for workflow << parameters.failed-workflow-id >>"
                                                    fi
                                                fi
                                            """,
                                        }
                                    },
                                ],
                            }
                        },
                    ],
                },
            },
        }

        return orb_config

    def create_workflow_config(self, template_type: str = "basic") -> Dict:
        """
        Create CircleCI workflow configuration for healing

        Args:
            template_type: Type of workflow (basic, advanced)

        Returns:
            Workflow configuration dictionary
        """
        if template_type == "basic":
            return self._create_basic_workflow()
        elif template_type == "advanced":
            return self._create_advanced_workflow()
        else:
            raise ValueError(f"Unknown template type: {template_type}")

    def _create_basic_workflow(self) -> Dict:
        """Create basic healing workflow configuration"""
        config = {
            "version": "2.1",
            "orbs": {"homeostasis": "homeostasis/healing@1.0.0"},
            "workflows": {
                "healing-on-failure": {
                    "jobs": [
                        {
                            "homeostasis/analyze-and-heal": {
                                "workflow-id": "${FAILED_WORKFLOW_ID}",
                                "confidence-threshold": "0.8",
                                "filters": {
                                    "branches": {"only": ["main", "master", "develop"]}
                                },
                            }
                        }
                    ]
                }
            },
        }

        return config

    def _create_advanced_workflow(self) -> Dict:
        """Create advanced healing workflow configuration"""
        config = {
            "version": "2.1",
            "orbs": {"homeostasis": "homeostasis/healing@1.0.0"},
            "parameters": {
                "failed-workflow-id": {"type": "string", "default": ""},
                "healing-mode": {
                    "type": "enum",
                    "enum": ["auto", "manual", "both"],
                    "default": "both",
                },
            },
            "workflows": {
                "advanced-healing": {
                    "when": {
                        "not": {
                            "equal": [
                                "",
                                "<< pipeline.parameters.failed-workflow-id >>",
                            ]
                        }
                    },
                    "jobs": [
                        {
                            "homeostasis/healing-pipeline": {
                                "failed-workflow-id": "<< pipeline.parameters.failed-workflow-id >>",
                                "context": "homeostasis-healing",
                            }
                        }
                    ],
                }
            },
        }

        return config

    def trigger_healing_pipeline(
        self, failed_workflow_id: str, branch: str = "main"
    ) -> Dict:
        """
        Trigger a healing pipeline for a failed workflow

        Args:
            failed_workflow_id: ID of the failed workflow
            branch: Branch to trigger pipeline on

        Returns:
            Pipeline trigger response
        """
        url = f"{self.api_url}/project/{self.project_slug}/pipeline"

        data = {
            "branch": branch,
            "parameters": {
                "failed-workflow-id": failed_workflow_id,
                "healing-mode": "both",
            },
        }

        response = requests.post(url, headers=self.headers, json=data, timeout=30)
        response.raise_for_status()

        return response.json()

    def _extract_failure_reason(self, job: Dict) -> str:
        """Extract failure reason from job data"""
        # CircleCI job data is limited without fetching detailed logs
        return f"Job {job['name']} failed with status {job['status']}"

    def _generate_healing_suggestions(self, analysis: Dict) -> List[Dict]:
        """Generate healing suggestions based on analysis"""
        suggestions = []

        # Basic suggestions based on job names and common patterns
        for job in analysis.get("failed_jobs", []):
            job_name = job.get("name", "").lower()

            if "test" in job_name:
                suggestions.append(
                    {
                        "type": "test",
                        "description": "Test job failure detected",
                        "fix": "Review test failures and update test cases",
                        "confidence": 0.6,
                        "job": job_name,
                    }
                )
            elif "build" in job_name:
                suggestions.append(
                    {
                        "type": "build",
                        "description": "Build job failure detected",
                        "fix": "Check build configuration and dependencies",
                        "confidence": 0.7,
                        "job": job_name,
                    }
                )
            elif "deploy" in job_name:
                suggestions.append(
                    {
                        "type": "deployment",
                        "description": "Deployment job failure detected",
                        "fix": "Review deployment configuration and credentials",
                        "confidence": 0.5,
                        "job": job_name,
                    }
                )

        return suggestions

    def setup_circleci_integration(self, create_configs: bool = True) -> Dict:
        """
        Set up complete CircleCI integration

        Args:
            create_configs: Whether to create configuration files

        Returns:
            Setup status information
        """
        result = {
            "orb_config_created": False,
            "basic_workflow_created": False,
            "advanced_workflow_created": False,
            "integration_complete": False,
        }

        if create_configs:
            # Create orb configuration
            orb_config = self.create_orb_config()
            with open("orb.yml", "w") as f:
                yaml.dump(orb_config, f, default_flow_style=False, sort_keys=False)
            result["orb_config_created"] = True

            # Create basic workflow
            basic_workflow = self.create_workflow_config("basic")
            with open(".circleci/config-healing-basic.yml", "w") as f:
                yaml.dump(basic_workflow, f, default_flow_style=False, sort_keys=False)
            result["basic_workflow_created"] = True

            # Create advanced workflow
            advanced_workflow = self.create_workflow_config("advanced")
            with open(".circleci/config-healing-advanced.yml", "w") as f:
                yaml.dump(
                    advanced_workflow, f, default_flow_style=False, sort_keys=False
                )
            result["advanced_workflow_created"] = True

            logger.info("CircleCI integration files created successfully")

        result["integration_complete"] = True
        return result
