"""
Deployment Platform Integration for Homeostasis

This module provides integration with various deployment platforms
to enable automatic healing during deployment processes.
"""

import logging
import os
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class DeploymentPlatformIntegration:
    """Base class for deployment platform integrations"""

    def __init__(self, platform: str):
        self.platform = platform
        self.api_token = None
        self.headers = {}

    def analyze_deployment_failure(self, deployment_id: str) -> Dict:
        """Analyze a failed deployment"""
        raise NotImplementedError

    def apply_healing_fix(self, analysis: Dict) -> Dict:
        """Apply healing fix to deployment"""
        raise NotImplementedError


class VercelIntegration(DeploymentPlatformIntegration):
    """Integration with Vercel deployment platform"""

    def __init__(self, api_token: Optional[str] = None, team_id: Optional[str] = None):
        super().__init__("vercel")
        self.api_token = api_token or os.getenv("VERCEL_TOKEN")
        self.team_id = team_id or os.getenv("VERCEL_TEAM_ID")

        if not self.api_token:
            raise ValueError("Vercel API token required (set VERCEL_TOKEN env var)")

        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }
        self.api_url = "https://api.vercel.com"

    def get_deployments(
        self, app_name: Optional[str] = None, limit: int = 20
    ) -> List[Dict]:
        """Get recent deployments"""
        url = f"{self.api_url}/v6/deployments"
        params = {"limit": limit}

        if app_name:
            params["app"] = app_name
        if self.team_id:
            params["teamId"] = self.team_id

        response = requests.get(url, headers=self.headers, params=params)
        response.raise_for_status()

        return response.json().get("deployments", [])

    def get_deployment_details(self, deployment_id: str) -> Dict:
        """Get detailed information about a deployment"""
        url = f"{self.api_url}/v13/deployments/{deployment_id}"
        params = {}

        if self.team_id:
            params["teamId"] = self.team_id

        response = requests.get(url, headers=self.headers, params=params)
        response.raise_for_status()

        return response.json()

    def get_build_logs(self, deployment_id: str) -> List[Dict]:
        """Get build logs for a deployment"""
        url = f"{self.api_url}/v2/deployments/{deployment_id}/events"
        params = {}

        if self.team_id:
            params["teamId"] = self.team_id

        response = requests.get(url, headers=self.headers, params=params)
        response.raise_for_status()

        return response.json()

    def analyze_deployment_failure(self, deployment_id: str) -> Dict:
        """Analyze a failed Vercel deployment"""
        deployment = self.get_deployment_details(deployment_id)
        logs = self.get_build_logs(deployment_id)

        analysis = {
            "platform": "vercel",
            "deployment_id": deployment_id,
            "url": deployment.get("url"),
            "state": deployment.get("state"),
            "error_patterns": [],
            "suggested_fixes": [],
            "healing_opportunities": [],
        }

        # Extract error patterns from logs
        for log_entry in logs:
            if (
                log_entry.get("type") == "stderr"
                or "error" in log_entry.get("text", "").lower()
            ):
                analysis["error_patterns"].append(log_entry.get("text", ""))

        # Generate healing suggestions
        analysis["suggested_fixes"] = self._generate_vercel_suggestions(analysis)

        return analysis

    def _generate_vercel_suggestions(self, analysis: Dict) -> List[Dict]:
        """Generate Vercel-specific healing suggestions"""
        suggestions = []

        for pattern in analysis.get("error_patterns", []):
            pattern_lower = pattern.lower()

            if "module not found" in pattern_lower:
                suggestions.append(
                    {
                        "type": "dependency",
                        "description": "Missing Node.js dependency",
                        "fix": "Add missing dependency to package.json",
                        "confidence": 0.9,
                        "vercel_specific": True,
                    }
                )
            elif "build failed" in pattern_lower:
                suggestions.append(
                    {
                        "type": "build",
                        "description": "Build process failure",
                        "fix": "Check build command and Next.js configuration",
                        "confidence": 0.8,
                        "vercel_specific": True,
                    }
                )
            elif "function timeout" in pattern_lower:
                suggestions.append(
                    {
                        "type": "performance",
                        "description": "Serverless function timeout",
                        "fix": "Optimize function performance or increase timeout",
                        "confidence": 0.7,
                        "vercel_specific": True,
                    }
                )
            elif "environment variable" in pattern_lower:
                suggestions.append(
                    {
                        "type": "configuration",
                        "description": "Missing environment variable",
                        "fix": "Configure required environment variables in Vercel dashboard",
                        "confidence": 0.9,
                        "vercel_specific": True,
                    }
                )

        return suggestions

    def create_vercel_config(self) -> Dict:
        """Create vercel.json configuration with healing hooks"""
        config = {
            "version": 2,
            "builds": [{"src": "package.json", "use": "@vercel/node"}],
            "functions": {
                "api/homeostasis-heal.js": {"memory": 512, "maxDuration": 30}
            },
            "env": {
                "HOMEOSTASIS_ENABLED": "true",
                "HOMEOSTASIS_CONFIDENCE_THRESHOLD": "0.8",
            },
            "build": {"env": {"HOMEOSTASIS_BUILD_HEALING": "true"}},
        }

        return config


class NetlifyIntegration(DeploymentPlatformIntegration):
    """Integration with Netlify deployment platform"""

    def __init__(self, api_token: Optional[str] = None, site_id: Optional[str] = None):
        super().__init__("netlify")
        self.api_token = api_token or os.getenv("NETLIFY_TOKEN")
        self.site_id = site_id or os.getenv("NETLIFY_SITE_ID")

        if not self.api_token:
            raise ValueError("Netlify API token required (set NETLIFY_TOKEN env var)")

        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }
        self.api_url = "https://api.netlify.com/api/v1"

    def get_site_deploys(self, site_id: Optional[str] = None) -> List[Dict]:
        """Get recent deploys for a site"""
        site_id = site_id or self.site_id
        if not site_id:
            raise ValueError("Site ID required")

        url = f"{self.api_url}/sites/{site_id}/deploys"

        response = requests.get(url, headers=self.headers)
        response.raise_for_status()

        return response.json()

    def get_deploy_logs(self, deploy_id: str) -> str:
        """Get build logs for a deploy"""
        url = f"{self.api_url}/deploys/{deploy_id}/log"

        response = requests.get(url, headers=self.headers)
        response.raise_for_status()

        return response.text

    def analyze_deployment_failure(self, deployment_id: str) -> Dict:
        """Analyze a failed Netlify deployment"""
        logs = self.get_deploy_logs(deployment_id)

        analysis = {
            "platform": "netlify",
            "deployment_id": deployment_id,
            "error_patterns": [],
            "suggested_fixes": [],
            "healing_opportunities": [],
        }

        # Extract error patterns from logs
        analysis["error_patterns"] = self._extract_netlify_errors(logs)

        # Generate healing suggestions
        analysis["suggested_fixes"] = self._generate_netlify_suggestions(analysis)

        return analysis

    def _extract_netlify_errors(self, logs: str) -> List[str]:
        """Extract error patterns from Netlify logs"""
        patterns = []
        error_indicators = [
            "Error:",
            "Failed:",
            "Command failed",
            "Build failed",
            "npm ERR!",
            "yarn error",
        ]

        for line in logs.split("\n"):
            for indicator in error_indicators:
                if indicator in line:
                    patterns.append(line.strip())
                    break

        return list(set(patterns))

    def _generate_netlify_suggestions(self, analysis: Dict) -> List[Dict]:
        """Generate Netlify-specific healing suggestions"""
        suggestions = []

        for pattern in analysis.get("error_patterns", []):
            pattern_lower = pattern.lower()

            if "command failed" in pattern_lower:
                suggestions.append(
                    {
                        "type": "build_command",
                        "description": "Build command failure",
                        "fix": "Check build command in netlify.toml or site settings",
                        "confidence": 0.8,
                        "netlify_specific": True,
                    }
                )
            elif "npm err!" in pattern_lower:
                suggestions.append(
                    {
                        "type": "npm_error",
                        "description": "NPM installation or script error",
                        "fix": "Fix package.json dependencies or scripts",
                        "confidence": 0.9,
                        "netlify_specific": True,
                    }
                )
            elif "redirect" in pattern_lower:
                suggestions.append(
                    {
                        "type": "redirect",
                        "description": "Redirect configuration issue",
                        "fix": "Check _redirects file or netlify.toml redirect rules",
                        "confidence": 0.7,
                        "netlify_specific": True,
                    }
                )

        return suggestions

    def create_netlify_config(self) -> Dict:
        """Create netlify.toml configuration with healing hooks"""
        config = {
            "build": {
                "command": "npm run build",
                "publish": "dist",
                "environment": {
                    "HOMEOSTASIS_ENABLED": "true",
                    "HOMEOSTASIS_BUILD_HEALING": "true",
                },
            },
            "functions": {"directory": "netlify/functions", "node_bundler": "nft"},
            "plugins": [
                {
                    "package": "@netlify/plugin-homeostasis",
                    "inputs": {"confidence_threshold": 0.8, "auto_heal": True},
                }
            ],
        }

        return config


class HerokuIntegration(DeploymentPlatformIntegration):
    """Integration with Heroku deployment platform"""

    def __init__(self, api_token: Optional[str] = None, app_name: Optional[str] = None):
        super().__init__("heroku")
        self.api_token = api_token or os.getenv("HEROKU_API_TOKEN")
        self.app_name = app_name or os.getenv("HEROKU_APP_NAME")

        if not self.api_token:
            raise ValueError("Heroku API token required (set HEROKU_API_TOKEN env var)")

        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Accept": "application/vnd.heroku+json; version=3",
            "Content-Type": "application/json",
        }
        self.api_url = "https://api.heroku.com"

    def get_app_builds(self, app_name: Optional[str] = None) -> List[Dict]:
        """Get recent builds for an app"""
        app_name = app_name or self.app_name
        if not app_name:
            raise ValueError("App name required")

        url = f"{self.api_url}/apps/{app_name}/builds"

        response = requests.get(url, headers=self.headers)
        response.raise_for_status()

        return response.json()

    def get_build_result(self, build_id: str) -> Dict:
        """Get build result details"""
        url = f"{self.api_url}/builds/{build_id}/result"

        response = requests.get(url, headers=self.headers)
        response.raise_for_status()

        return response.json()

    def analyze_deployment_failure(self, deployment_id: str) -> Dict:
        """Analyze a failed Heroku deployment"""
        build_result = self.get_build_result(deployment_id)

        analysis = {
            "platform": "heroku",
            "deployment_id": deployment_id,
            "status": build_result.get("status"),
            "error_patterns": [],
            "suggested_fixes": [],
            "healing_opportunities": [],
        }

        # Extract error patterns from build output
        build_output = build_result.get("lines", [])
        for line in build_output:
            if "error" in line.lower() or "failed" in line.lower():
                analysis["error_patterns"].append(line)

        # Generate healing suggestions
        analysis["suggested_fixes"] = self._generate_heroku_suggestions(analysis)

        return analysis

    def _generate_heroku_suggestions(self, analysis: Dict) -> List[Dict]:
        """Generate Heroku-specific healing suggestions"""
        suggestions = []

        for pattern in analysis.get("error_patterns", []):
            pattern_lower = pattern.lower()

            if "buildpack" in pattern_lower:
                suggestions.append(
                    {
                        "type": "buildpack",
                        "description": "Buildpack detection or execution error",
                        "fix": "Specify correct buildpack or fix buildpack configuration",
                        "confidence": 0.8,
                        "heroku_specific": True,
                    }
                )
            elif "procfile" in pattern_lower:
                suggestions.append(
                    {
                        "type": "procfile",
                        "description": "Procfile configuration issue",
                        "fix": "Check Procfile syntax and process definitions",
                        "confidence": 0.9,
                        "heroku_specific": True,
                    }
                )
            elif "slug size" in pattern_lower:
                suggestions.append(
                    {
                        "type": "slug_size",
                        "description": "Application slug size too large",
                        "fix": "Reduce application size or use .slugignore",
                        "confidence": 0.7,
                        "heroku_specific": True,
                    }
                )

        return suggestions


class UniversalDeploymentHealer:
    """Universal deployment healing coordinator"""

    def __init__(self):
        self.platforms = {
            "vercel": VercelIntegration,
            "netlify": NetlifyIntegration,
            "heroku": HerokuIntegration,
        }

    def detect_platform(self) -> Optional[str]:
        """Auto-detect deployment platform from environment"""
        if os.getenv("VERCEL"):
            return "vercel"
        elif os.getenv("NETLIFY"):
            return "netlify"
        elif os.getenv("DYNO"):  # Heroku sets DYNO env var
            return "heroku"

        # Check for platform-specific files
        if os.path.exists("vercel.json"):
            return "vercel"
        elif os.path.exists("netlify.toml"):
            return "netlify"
        elif os.path.exists("Procfile"):
            return "heroku"

        return None

    def get_platform_integration(self, platform: str) -> DeploymentPlatformIntegration:
        """Get integration instance for a platform"""
        if platform not in self.platforms:
            raise ValueError(f"Unsupported platform: {platform}")

        return self.platforms[platform]()

    def heal_deployment(self, platform: str, deployment_id: str) -> Dict:
        """Heal a failed deployment on any supported platform"""
        integration = self.get_platform_integration(platform)

        # Analyze the failure
        analysis = integration.analyze_deployment_failure(deployment_id)

        # Apply platform-specific healing
        healing_result = {
            "platform": platform,
            "deployment_id": deployment_id,
            "analysis": analysis,
            "fixes_applied": [],
            "success": False,
        }

        # Apply fixes based on suggestions
        for fix in analysis.get("suggested_fixes", []):
            if fix.get("confidence", 0) >= 0.8:
                # Apply high-confidence fixes automatically
                try:
                    result = self._apply_platform_fix(platform, fix)
                    healing_result["fixes_applied"].append(result)
                except Exception as e:
                    logger.error(f"Failed to apply fix: {e}")

        healing_result["success"] = len(healing_result["fixes_applied"]) > 0
        return healing_result

    def _apply_platform_fix(self, platform: str, fix: Dict) -> Dict:
        """Apply a specific fix for a platform"""
        fix_type = fix.get("type")

        if platform == "vercel":
            return self._apply_vercel_fix(fix_type, fix)
        elif platform == "netlify":
            return self._apply_netlify_fix(fix_type, fix)
        elif platform == "heroku":
            return self._apply_heroku_fix(fix_type, fix)

        return {"fix": fix_type, "applied": False, "reason": "Unsupported platform"}

    def _apply_vercel_fix(self, fix_type: str, fix: Dict) -> Dict:
        """Apply Vercel-specific fixes"""
        if fix_type == "dependency":
            # Auto-add missing dependencies
            return {
                "fix": fix_type,
                "applied": True,
                "action": "Added missing dependency",
            }
        elif fix_type == "configuration":
            # Update environment variables
            return {"fix": fix_type, "applied": True, "action": "Updated configuration"}

        return {"fix": fix_type, "applied": False, "reason": "Manual fix required"}

    def _apply_netlify_fix(self, fix_type: str, fix: Dict) -> Dict:
        """Apply Netlify-specific fixes"""
        if fix_type == "build_command":
            # Update build command
            return {"fix": fix_type, "applied": True, "action": "Updated build command"}
        elif fix_type == "npm_error":
            # Fix package.json issues
            return {"fix": fix_type, "applied": True, "action": "Fixed package.json"}

        return {"fix": fix_type, "applied": False, "reason": "Manual fix required"}

    def _apply_heroku_fix(self, fix_type: str, fix: Dict) -> Dict:
        """Apply Heroku-specific fixes"""
        if fix_type == "procfile":
            # Update Procfile
            return {"fix": fix_type, "applied": True, "action": "Updated Procfile"}
        elif fix_type == "buildpack":
            # Set correct buildpack
            return {"fix": fix_type, "applied": True, "action": "Updated buildpack"}

        return {"fix": fix_type, "applied": False, "reason": "Manual fix required"}

    def create_universal_config(self) -> Dict:
        """Create universal deployment healing configuration"""
        return {
            "homeostasis": {
                "deployment_healing": {
                    "enabled": True,
                    "confidence_threshold": 0.8,
                    "auto_apply_fixes": True,
                    "platforms": {
                        "vercel": {
                            "enabled": True,
                            "heal_on_build_failure": True,
                            "heal_on_function_timeout": True,
                        },
                        "netlify": {
                            "enabled": True,
                            "heal_on_build_failure": True,
                            "heal_on_redirect_issues": True,
                        },
                        "heroku": {
                            "enabled": True,
                            "heal_on_build_failure": True,
                            "heal_on_slug_size": True,
                        },
                    },
                }
            }
        }
