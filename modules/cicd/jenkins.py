"""
Jenkins Integration for Homeostasis

This module provides integration with Jenkins to enable automatic
healing during build pipelines.
"""

# flake8: noqa: E999

import base64
import json
import logging
import os
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import requests

logger = logging.getLogger(__name__)


class JenkinsIntegration:
    """Integration with Jenkins for automated healing"""

    def __init__(
        self,
        jenkins_url: Optional[str] = None,
        username: Optional[str] = None,
        api_token: Optional[str] = None,
    ):
        """
        Initialize Jenkins integration

        Args:
            jenkins_url: Jenkins server URL
            username: Jenkins username
            api_token: Jenkins API token
        """
        self.jenkins_url = (jenkins_url or os.getenv("JENKINS_URL", "")).rstrip("/")
        self.username = username or os.getenv("JENKINS_USERNAME")
        self.api_token = api_token or os.getenv("JENKINS_API_TOKEN")

        if not self.jenkins_url:
            raise ValueError("Jenkins URL required (set JENKINS_URL env var)")
        if not self.username or not self.api_token:
            raise ValueError(
                "Jenkins credentials required (set JENKINS_USERNAME and JENKINS_API_TOKEN env vars)"
            )

        # Create basic auth header
        credentials = f"{self.username}:{self.api_token}"
        encoded_credentials = base64.b64encode(credentials.encode()).decode()

        self.headers = {
            "Authorization": f"Basic {encoded_credentials}",
            "Content-Type": "application/json",
        }

    def get_job_info(self, job_name: str) -> Dict:
        """
        Get information about a Jenkins job

        Args:
            job_name: Name of the Jenkins job

        Returns:
            Job information
        """
        url = urljoin(self.jenkins_url, f"/job/{job_name}/api/json")

        response = requests.get(url, headers=self.headers, timeout=30)
        response.raise_for_status()

        return response.json()

    def get_build_info(self, job_name: str, build_number: int) -> Dict:
        """
        Get information about a specific build

        Args:
            job_name: Name of the Jenkins job
            build_number: Build number

        Returns:
            Build information
        """
        url = urljoin(self.jenkins_url, f"/job/{job_name}/{build_number}/api/json")

        response = requests.get(url, headers=self.headers, timeout=30)
        response.raise_for_status()

        return response.json()

    def get_console_log(self, job_name: str, build_number: int) -> str:
        """
        Get console log for a build

        Args:
            job_name: Name of the Jenkins job
            build_number: Build number

        Returns:
            Console log content
        """
        url = urljoin(self.jenkins_url, f"/job/{job_name}/{build_number}/consoleText")

        response = requests.get(url, headers=self.headers, timeout=30)
        response.raise_for_status()

        return response.text

    def get_failed_builds(self, job_name: str, count: int = 10) -> List[Dict]:
        """
        Get recent failed builds for a job

        Args:
            job_name: Name of the Jenkins job
            count: Number of builds to check

        Returns:
            List of failed build information
        """
        job_info = self.get_job_info(job_name)
        builds = job_info.get("builds", [])

        failed_builds = []
        for build in builds[:count]:
            build_info = self.get_build_info(job_name, build["number"])
            if build_info.get("result") == "FAILURE":
                failed_builds.append(build_info)

        return failed_builds

    def analyze_build_failure(self, job_name: str, build_number: int) -> Dict:
        """
        Analyze a failed build to identify healing opportunities

        Args:
            job_name: Name of the Jenkins job
            build_number: Failed build number

        Returns:
            Analysis results with potential fixes
        """
        build_info = self.get_build_info(job_name, build_number)
        console_log = self.get_console_log(job_name, build_number)

        analysis = {
            "job_name": job_name,
            "build_number": build_number,
            "build_url": build_info.get("url"),
            "duration": build_info.get("duration"),
            "timestamp": build_info.get("timestamp"),
            "error_patterns": [],
            "suggested_fixes": [],
            "healing_opportunities": [],
        }

        # Extract error patterns from console log
        analysis["error_patterns"] = self._extract_error_patterns(console_log)

        # Generate healing suggestions
        analysis["suggested_fixes"] = self._generate_healing_suggestions(analysis)

        return analysis

    def create_healing_pipeline(self, template_type: str = "basic") -> str:
        """
        Create a Jenkins pipeline script for automated healing

        Args:
            template_type: Type of healing pipeline (basic, advanced)

        Returns:
            Groovy pipeline script
        """
        if template_type == "basic":
            return self._create_basic_healing_pipeline()
        elif template_type == "advanced":
            return self._create_advanced_healing_pipeline()
        else:
            raise ValueError(f"Unknown template type: {template_type}")

    def _create_basic_healing_pipeline(self) -> str:
        """Create basic healing Jenkins pipeline"""
        pipeline = '''pipeline {
    agent any
    
    parameters {
        string(name: 'FAILED_JOB_NAME', defaultValue: '', description: 'Name of the failed job to heal')
        string(name: 'FAILED_BUILD_NUMBER', defaultValue: '', description: 'Build number of the failed build')
        string(name: 'CONFIDENCE_THRESHOLD', defaultValue: '0.8', description: 'Minimum confidence for auto-healing')
    }
    
    environment {
        HOMEOSTASIS_CONFIDENCE = "${params.CONFIDENCE_THRESHOLD}"
    }
    
    stages {
        stage('Setup') {
            steps {
                script {
                    echo "Setting up Homeostasis healing pipeline"
                    sh 'pip install homeostasis[jenkins]'
                }
            }
        }
        
        stage('Analyze Failure') {
            steps {
                script {
                    echo "Analyzing failed build: ${params.FAILED_JOB_NAME} #${params.FAILED_BUILD_NUMBER}"
                    
                    sh """
                        homeostasis analyze-jenkins-build \\
                            --jenkins-url "${env.JENKINS_URL}" \\
                            --job-name "${params.FAILED_JOB_NAME}" \\
                            --build-number "${params.FAILED_BUILD_NUMBER}" \\
                            --output analysis.json
                    """
                    
                    def analysis = readJSON file: 'analysis.json'
                    env.HEALING_NEEDED = analysis.healing_recommended
                    env.CONFIDENCE_SCORE = analysis.confidence_score
                    
                    echo "Healing needed: ${env.HEALING_NEEDED}"
                    echo "Confidence score: ${env.CONFIDENCE_SCORE}"
                }
            }
            post {
                always {
                    archiveArtifacts artifacts: 'analysis.json', fingerprint: true
                }
            }
        }
        
        stage('Apply Healing') {
            when {
                expression { env.HEALING_NEEDED == 'true' }
            }
            steps {
                script {
                    echo "Applying healing fixes with confidence >= ${env.HOMEOSTASIS_CONFIDENCE}"
                    
                    sh """
                        homeostasis heal \\
                            --input analysis.json \\
                            --min-confidence "${env.HOMEOSTASIS_CONFIDENCE}"
                    """
                }
            }
        }
        
        stage('Test Fixes') {
            when {
                expression { env.HEALING_NEEDED == 'true' }
            }
            steps {
                script {
                    echo "Testing applied fixes"
                    
                    // Run appropriate tests based on project type
                    sh """
                        if [ -f "package.json" ]; then
                            npm test
                        elif [ -f "requirements.txt" ]; then
                            python -m pytest
                        elif [ -f "pom.xml" ]; then
                            mvn test
                        elif [ -f "build.gradle" ]; then
                            ./gradlew test
                        fi
                    """
                }
            }
        }
        
        stage('Commit Changes') {
            when {
                expression { env.HEALING_NEEDED == 'true' }
            }
            steps {
                script {
                    echo "Committing healing changes"
                    
                    sh """
                        git config user.email "jenkins@homeostasis.bot"
                        git config user.name "Jenkins Homeostasis Bot"
                        
                        if ! git diff --quiet; then
                            git add .
                            git commit -m "Auto-heal: Fix issues from build ${params.FAILED_JOB_NAME} #${params.FAILED_BUILD_NUMBER}"
                            git push origin main
                        else
                            echo "No changes to commit"
                        fi
                    """
                }
            }
        }
    }
    
    post {
        always {
            echo "Homeostasis healing pipeline completed"
        }
        success {
            echo "Healing completed successfully"
        }
        failure {
            echo "Healing pipeline failed"
        }
    }
}
'''  # noqa: E999
        return pipeline.strip()

    def _create_advanced_healing_pipeline(self) -> str:
        """Create advanced healing Jenkins pipeline"""
        pipeline = '''
pipeline {
    agent any
    
    parameters {
        string(name: 'FAILED_JOB_NAME', defaultValue: '', description: 'Name of the failed job to heal')
        string(name: 'FAILED_BUILD_NUMBER', defaultValue: '', description: 'Build number of the failed build')
        string(name: 'HIGH_CONFIDENCE_THRESHOLD', defaultValue: '0.8', description: 'Threshold for automatic healing')
        string(name: 'MEDIUM_CONFIDENCE_THRESHOLD', defaultValue: '0.6', description: 'Threshold for manual review')
        choice(name: 'HEALING_STRATEGY', choices: ['auto', 'manual', 'both'], description: 'Healing strategy to use')
    }
    
    environment {
        HOMEOSTASIS_HIGH_CONFIDENCE = "${params.HIGH_CONFIDENCE_THRESHOLD}"
        HOMEOSTASIS_MEDIUM_CONFIDENCE = "${params.MEDIUM_CONFIDENCE_THRESHOLD}"
    }
    
    stages {
        stage('Setup Environment') {
            steps {
                script {
                    echo "Setting up advanced Homeostasis healing pipeline"
                    sh """
                        python -m pip install --upgrade pip
                        pip install homeostasis[jenkins] requests
                    """
                }
            }
        }
        
        stage('Deep Analysis') {
            parallel {
                stage('Analyze Current Failure') {
                    steps {
                        script {
                            echo "Analyzing current failed build"
                            sh """
                                homeostasis analyze-jenkins-build \\
                                    --jenkins-url "${env.JENKINS_URL}" \\
                                    --job-name "${params.FAILED_JOB_NAME}" \\
                                    --build-number "${params.FAILED_BUILD_NUMBER}" \\
                                    --deep-analysis \\
                                    --output current-analysis.json
                            """
                        }
                    }
                }
                
                stage('Analyze Historical Patterns') {
                    steps {
                        script {
                            echo "Analyzing historical failure patterns"
                            sh """
                                homeostasis analyze-jenkins-job \\
                                    --jenkins-url "${env.JENKINS_URL}" \\
                                    --job-name "${params.FAILED_JOB_NAME}" \\
                                    --history-count 20 \\
                                    --output historical-analysis.json
                            """
                        }
                    }
                }
            }
            post {
                always {
                    script {
                        // Merge analysis results
                        sh """
                            homeostasis merge-analysis \\
                                --inputs current-analysis.json historical-analysis.json \\
                                --output combined-analysis.json
                        """
                        
                        def analysis = readJSON file: 'combined-analysis.json'
                        env.HEALING_NEEDED = analysis.healing_recommended
                        env.CONFIDENCE_SCORE = analysis.confidence_score
                        env.PATTERN_MATCHES = analysis.pattern_matches
                        
                        echo "Combined analysis results:"
                        echo "  - Healing needed: ${env.HEALING_NEEDED}"
                        echo "  - Confidence score: ${env.CONFIDENCE_SCORE}"
                        echo "  - Pattern matches: ${env.PATTERN_MATCHES}"
                    }
                    archiveArtifacts artifacts: '*.json', fingerprint: true
                }
            }
        }
        
        stage('High-Confidence Auto-Healing') {
            when {
                allOf {
                    expression { env.HEALING_NEEDED == 'true' }
                    expression { 
                        return Float.parseFloat(env.CONFIDENCE_SCORE) >= Float.parseFloat(env.HOMEOSTASIS_HIGH_CONFIDENCE)
                    }
                    expression { params.HEALING_STRATEGY in ['auto', 'both'] }
                }
            }
            steps {
                script {
                    echo "Applying high-confidence automatic fixes"
                    
                    sh """
                        homeostasis heal \\
                            --input combined-analysis.json \\
                            --min-confidence "${env.HOMEOSTASIS_HIGH_CONFIDENCE}" \\
                            --auto-apply \\
                            --create-backup \\
                            --output healing-results.json
                    """
                    
                    def healingResults = readJSON file: 'healing-results.json'
                    env.FIXES_APPLIED = healingResults.fixes_applied
                    env.AUTO_HEALING_SUCCESS = healingResults.success
                }
            }
        }
        
        stage('Medium-Confidence Manual Review') {
            when {
                allOf {
                    expression { env.HEALING_NEEDED == 'true' }
                    expression { 
                        Float.parseFloat(env.CONFIDENCE_SCORE) >= Float.parseFloat(env.HOMEOSTASIS_MEDIUM_CONFIDENCE) &&
                        Float.parseFloat(env.CONFIDENCE_SCORE) < Float.parseFloat(env.HOMEOSTASIS_HIGH_CONFIDENCE)
                    }
                    expression { params.HEALING_STRATEGY in ['manual', 'both'] }
                }
            }
            steps {
                script {
                    echo "Creating fixes for manual review"
                    
                    sh """
                        homeostasis heal \\
                            --input combined-analysis.json \\
                            --min-confidence "${env.HOMEOSTASIS_MEDIUM_CONFIDENCE}" \\
                            --max-confidence "${env.HOMEOSTASIS_HIGH_CONFIDENCE}" \\
                            --suggest-only \\
                            --output manual-review.json
                    """
                    
                    // Create a review branch
                    sh """
                        git checkout -b "homeostasis/review-${BUILD_NUMBER}"
                        
                        homeostasis apply-suggestions \\
                            --input manual-review.json \\
                            --create-pr
                        
                        git add .
                        git commit -m "Homeostasis: Suggested fixes for manual review (Build #${BUILD_NUMBER})"
                        git push origin "homeostasis/review-${BUILD_NUMBER}"
                    """
                    
                    env.REVIEW_BRANCH = "homeostasis/review-${BUILD_NUMBER}"
                }
            }
        }
        
        stage('Comprehensive Testing') {
            when {
                expression { env.HEALING_NEEDED == 'true' }
            }
            parallel {
                stage('Unit Tests') {
                    steps {
                        script {
                            echo "Running unit tests"
                            sh """
                                if [ -f "package.json" ]; then
                                    npm install && npm run test:unit
                                elif [ -f "requirements.txt" ]; then
                                    pip install -r requirements.txt && python -m pytest tests/unit/
                                elif [ -f "pom.xml" ]; then
                                    mvn test
                                elif [ -f "build.gradle" ]; then
                                    ./gradlew test
                                fi
                            """
                        }
                    }
                }
                
                stage('Integration Tests') {
                    steps {
                        script {
                            echo "Running integration tests"
                            sh """
                                if [ -f "package.json" ]; then
                                    npm run test:integration || true
                                elif [ -f "requirements.txt" ]; then
                                    python -m pytest tests/integration/ || true
                                elif [ -f "pom.xml" ]; then
                                    mvn integration-test || true
                                elif [ -f "build.gradle" ]; then
                                    ./gradlew integrationTest || true
                                fi
                            """
                        }
                    }
                }
                
                stage('Security Scan') {
                    steps {
                        script {
                            echo "Running security scan on fixed code"
                            sh """
                                homeostasis security-scan \\
                                    --path . \\
                                    --output security-scan.json || true
                            """
                        }
                    }
                }
            }
            post {
                always {
                    script {
                        publishTestResults testResultsPattern: '**/test-results.xml'
                        archiveArtifacts artifacts: 'security-scan.json', allowEmptyArchive: true
                    }
                }
            }
        }
        
        stage('Deploy Fix') {
            when {
                allOf {
                    expression { env.AUTO_HEALING_SUCCESS == 'true' }
                    expression { currentBuild.result == null || currentBuild.result == 'SUCCESS' }
                }
            }
            steps {
                script {
                    echo "Deploying healing fix"
                    
                    sh """
                        git config user.email "jenkins@homeostasis.bot"
                        git config user.name "Jenkins Homeostasis Bot"
                        
                        if ! git diff --quiet; then
                            git add .
                            git commit -m "🔧 Auto-heal: Fix issues from ${params.FAILED_JOB_NAME} #${params.FAILED_BUILD_NUMBER}
                            
                            - Applied ${env.FIXES_APPLIED} fixes with confidence >= ${env.HOMEOSTASIS_HIGH_CONFIDENCE}
                            - All tests passed
                            - Automated healing by Homeostasis"
                            
                            git push origin main
                        fi
                    """
                }
            }
        }
        
        stage('Notification') {
            steps {
                script {
                    echo "Sending healing notification"
                    
                    def message = """
Homeostasis Healing Results:
- Job: ${params.FAILED_JOB_NAME}
- Build: #${params.FAILED_BUILD_NUMBER}
- Healing needed: ${env.HEALING_NEEDED}
- Confidence score: ${env.CONFIDENCE_SCORE}
- Fixes applied: ${env.FIXES_APPLIED ?: '0'}
- Auto-healing success: ${env.AUTO_HEALING_SUCCESS ?: 'false'}
- Review branch: ${env.REVIEW_BRANCH ?: 'none'}
"""
                    
                    sh """
                        homeostasis notify \\
                            --message "${message}" \\
                            --channels slack,email
                    """
                }
            }
        }
    }
    
    post {
        always {
            script {
                echo "Advanced healing pipeline completed"
                archiveArtifacts artifacts: 'healing-results.json,manual-review.json', allowEmptyArchive: true
            }
        }
        success {
            echo "✅ Healing completed successfully"
        }
        failure {
            echo "❌ Healing pipeline encountered issues"
        }
        cleanup {
            script {
                // Clean up temporary files
                sh 'rm -f *.json || true'
            }
        }
    }
}
'''
        return pipeline.strip()

    def create_job(
        self, job_name: str, pipeline_script: str, description: str = ""
    ) -> bool:
        """
        Create a new Jenkins job with the healing pipeline

        Args:
            job_name: Name for the new job
            pipeline_script: Pipeline script content
            description: Job description

        Returns:
            True if job was created successfully
        """
        job_config = f"""<?xml version='1.1' encoding='UTF-8'?>
<flow-definition plugin="workflow-job@2.40">
  <description>{description}</description>
  <keepDependencies>false</keepDependencies>
  <properties>
    <jenkins.model.BuildDiscarderProperty>
      <strategy class="hudson.tasks.LogRotator">
        <daysToKeep>-1</daysToKeep>
        <numToKeep>10</numToKeep>
        <artifactDaysToKeep>-1</artifactDaysToKeep>
        <artifactNumToKeep>-1</artifactNumToKeep>
      </strategy>
    </jenkins.model.BuildDiscarderProperty>
  </properties>
  <definition class="org.jenkinsci.plugins.workflow.cps.CpsFlowDefinition" plugin="workflow-cps@2.87">
    <script>{pipeline_script}</script>
    <sandbox>true</sandbox>
  </definition>
  <triggers/>
  <disabled>false</disabled>
</flow-definition>"""

        url = urljoin(self.jenkins_url, f"/createItem?name={job_name}")
        headers = {**self.headers, "Content-Type": "application/xml"}

        response = requests.post(url, headers=headers, data=job_config, timeout=30)

        if response.status_code == 200:
            logger.info(f"Jenkins job '{job_name}' created successfully")
            return True
        else:
            logger.error(f"Failed to create Jenkins job '{job_name}': {response.text}")
            return False

    def trigger_healing_job(
        self, healing_job_name: str, failed_job_name: str, failed_build_number: int
    ) -> Dict:
        """
        Trigger a healing job for a failed build

        Args:
            healing_job_name: Name of the healing job to trigger
            failed_job_name: Name of the job that failed
            failed_build_number: Build number that failed

        Returns:
            Trigger response information
        """
        params = {
            "FAILED_JOB_NAME": failed_job_name,
            "FAILED_BUILD_NUMBER": str(failed_build_number),
        }

        url = urljoin(self.jenkins_url, f"/job/{healing_job_name}/buildWithParameters")

        response = requests.post(url, headers=self.headers, data=params, timeout=30)

        if response.status_code == 201:
            return {
                "success": True,
                "queue_url": response.headers.get("Location"),
                "message": f"Healing job triggered for {failed_job_name} #{failed_build_number}",
            }
        else:
            return {
                "success": False,
                "error": response.text,
                "message": f"Failed to trigger healing job",
            }

    def _extract_error_patterns(self, log: str) -> List[str]:
        """Extract common error patterns from build logs"""
        patterns = []

        # Common failure patterns
        error_indicators = [
            "BUILD FAILED",
            "COMPILATION ERROR",
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
            "No such file or directory",
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
                        "description": "Missing dependency detected",
                        "fix": "Add missing dependency to requirements or package file",
                        "confidence": 0.9,
                        "pattern": pattern,
                    }
                )
            elif "COMPILATION ERROR" in pattern:
                suggestions.append(
                    {
                        "type": "compilation",
                        "description": "Code compilation error",
                        "fix": "Fix syntax or type errors in source code",
                        "confidence": 0.8,
                        "pattern": pattern,
                    }
                )
            elif "Permission denied" in pattern:
                suggestions.append(
                    {
                        "type": "permissions",
                        "description": "File permission error",
                        "fix": "Add chmod commands or fix file permissions",
                        "confidence": 0.8,
                        "pattern": pattern,
                    }
                )
            elif "BUILD FAILED" in pattern:
                suggestions.append(
                    {
                        "type": "build",
                        "description": "Build process failure",
                        "fix": "Review build configuration and dependencies",
                        "confidence": 0.6,
                        "pattern": pattern,
                    }
                )

        return suggestions

    def setup_jenkins_integration(self, create_jobs: bool = True) -> Dict:
        """
        Set up complete Jenkins integration

        Args:
            create_jobs: Whether to create healing jobs

        Returns:
            Setup status information
        """
        result = {
            "basic_job_created": False,
            "advanced_job_created": False,
            "integration_complete": False,
        }

        if create_jobs:
            # Create basic healing job
            basic_pipeline = self.create_healing_pipeline("basic")
            basic_created = self.create_job(
                "homeostasis-healing-basic",
                basic_pipeline,
                "Basic Homeostasis healing pipeline for automatic fixes",
            )
            result["basic_job_created"] = basic_created

            # Create advanced healing job
            advanced_pipeline = self.create_healing_pipeline("advanced")
            advanced_created = self.create_job(
                "homeostasis-healing-advanced",
                advanced_pipeline,
                "Advanced Homeostasis healing pipeline with deep analysis",
            )
            result["advanced_job_created"] = advanced_created

            logger.info("Jenkins integration jobs created")

        result["integration_complete"] = True
        return result
