"""
Cross-language end-to-end healing scenario tests.

Tests healing workflows across different programming languages and frameworks,
including polyglot microservices and language-specific error patterns.
"""
import asyncio
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from tests.e2e.healing_scenarios.test_utilities import (
    HealingScenario,
    HealingScenarioRunner,
    TestEnvironment,
    MetricsCollector,
    check_service_healthy,
    check_no_syntax_errors
)


class CrossLanguageTestEnvironment(TestEnvironment):
    """Extended test environment for cross-language scenarios."""
    
    def create_javascript_service(self):
        """Create a Node.js/Express service for testing."""
        self.service_path.mkdir(parents=True, exist_ok=True)
        
        # Create package.json
        package_json = {
            "name": "test-service",
            "version": "1.0.0",
            "main": "app.js",
            "scripts": {
                "start": "node app.js"
            },
            "dependencies": {
                "express": "^4.18.0"
            }
        }
        
        with open(self.service_path / "package.json", "w") as f:
            json.dump(package_json, f, indent=2)
            
        # Create Express app
        js_code = '''
const express = require('express');
const app = express();
const port = 8000;

app.get('/health', (req, res) => {
    res.json({ status: 'healthy' });
});

app.get('/error', (req, res) => {
    // This will be modified to trigger errors
    const data = { key1: 'value1' };
    res.json({ result: data.missingKey.nested }); // TypeError
});

app.listen(port, () => {
    console.log(`Service listening on port ${port}`);
});
'''
        (self.service_path / "app.js").write_text(js_code)
        
        # Update config for Node.js
        self.orchestrator.config["service"]["start_command"] = "cd " + str(self.service_path) + " && npm install && node app.js"
        
    def create_go_service(self):
        """Create a Go service for testing."""
        self.service_path.mkdir(parents=True, exist_ok=True)
        
        # Create go.mod
        go_mod = '''module testservice

go 1.19

require github.com/gin-gonic/gin v1.9.0
'''
        (self.service_path / "go.mod").write_text(go_mod)
        
        # Create Go service
        go_code = '''package main

import (
    "net/http"
    "github.com/gin-gonic/gin"
)

func main() {
    r := gin.Default()
    
    r.GET("/health", func(c *gin.Context) {
        c.JSON(http.StatusOK, gin.H{
            "status": "healthy",
        })
    })
    
    r.GET("/error", func(c *gin.Context) {
        // This will be modified to trigger errors
        var data map[string]interface{}
        value := data["missingKey"] // panic: assignment to entry in nil map
        c.JSON(http.StatusOK, gin.H{
            "result": value,
        })
    })
    
    r.Run(":8000")
}
'''
        (self.service_path / "main.go").write_text(go_code)
        
        # Update config for Go
        self.orchestrator.config["service"]["start_command"] = f"cd {self.service_path} && go mod download && go run main.go"
        
    def create_java_service(self):
        """Create a Java Spring Boot service for testing."""
        self.service_path.mkdir(parents=True, exist_ok=True)
        
        # Create a simple Java service file
        java_code = '''import java.io.*;
import java.net.*;
import java.util.*;

public class SimpleService {
    public static void main(String[] args) throws IOException {
        ServerSocket serverSocket = new ServerSocket(8000);
        System.out.println("Service listening on port 8000");
        
        while (true) {
            Socket clientSocket = serverSocket.accept();
            handleRequest(clientSocket);
        }
    }
    
    private static void handleRequest(Socket socket) throws IOException {
        BufferedReader in = new BufferedReader(new InputStreamReader(socket.getInputStream()));
        PrintWriter out = new PrintWriter(socket.getOutputStream(), true);
        
        String requestLine = in.readLine();
        if (requestLine != null && requestLine.contains("GET /health")) {
            sendResponse(out, 200, "{\\"status\\": \\"healthy\\"}");
        } else if (requestLine != null && requestLine.contains("GET /error")) {
            // This will trigger a NullPointerException
            String data = null;
            int length = data.length(); // NullPointerException
            sendResponse(out, 200, "{\\"length\\": " + length + "}");
        } else {
            sendResponse(out, 404, "{\\"error\\": \\"Not found\\"}");
        }
        
        socket.close();
    }
    
    private static void sendResponse(PrintWriter out, int statusCode, String body) {
        out.println("HTTP/1.1 " + statusCode + " OK");
        out.println("Content-Type: application/json");
        out.println("Content-Length: " + body.length());
        out.println();
        out.println(body);
    }
}
'''
        (self.service_path / "SimpleService.java").write_text(java_code)
        
        # Update config for Java
        self.orchestrator.config["service"]["start_command"] = f"cd {self.service_path} && javac SimpleService.java && java SimpleService"


class TestCrossLanguageHealing:
    """Test cross-language healing scenarios."""
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("language,setup_method,error_pattern", [
        ("javascript", "create_javascript_service", "TypeError: Cannot read property"),
        ("go", "create_go_service", "panic: runtime error"),
        ("java", "create_java_service", "NullPointerException"),
    ])
    async def test_language_specific_healing(
        self, 
        language: str,
        setup_method: str,
        error_pattern: str,
        metrics_collector
    ):
        """Test healing for different programming languages."""
        # Create language-specific environment
        env = CrossLanguageTestEnvironment()
        env.setup()
        
        # Set up language-specific service
        getattr(env, setup_method)()
        
        runner = HealingScenarioRunner(env)
        
        def trigger_language_error():
            env.start_service()
            env.trigger_error()
            
        scenario = HealingScenario(
            name=f"{language.title()} Error Healing",
            description=f"Healing {language} specific error patterns",
            error_type=error_pattern,
            target_service=f"{language}_service",
            error_trigger=trigger_language_error,
            validation_checks=[
                check_service_healthy,
                lambda: check_language_fix_applied(env.service_path, language)
            ],
            expected_fix_type=f"{language}_error_fix"
        )
        
        try:
            result = await runner.run_scenario(scenario)
            
            metrics_collector.record_healing_duration(f"{language}_healing", result.duration)
            metrics_collector.record_success_rate(f"{language}_healing", result.success)
            
            assert result.error_detected, f"{language} error should have been detected"
            assert result.patch_generated or result.patch_applied, f"{language} patch should have been generated/applied"
            
        finally:
            env.cleanup()
            
    @pytest.mark.asyncio
    async def test_polyglot_microservice_healing(self, metrics_collector):
        """Test healing in a polyglot microservice architecture."""
        # Create multiple services in different languages
        services = []
        
        # Python service
        python_env = TestEnvironment()
        python_env.setup()
        services.append(("python", python_env, 8000))
        
        # JavaScript service  
        js_env = CrossLanguageTestEnvironment()
        js_env.setup()
        js_env.create_javascript_service()
        # Modify port
        js_env.orchestrator.config["service"]["health_check_url"] = "http://localhost:8001/health"
        services.append(("javascript", js_env, 8001))
        
        # Create inter-service communication errors
        def trigger_polyglot_error():
            # Modify Python service to call JavaScript service
            python_code = '''
from fastapi import FastAPI, HTTPException
import uvicorn
import requests

app = FastAPI()

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/error")
async def trigger_error():
    # Call JavaScript service with wrong expectations
    try:
        response = requests.get("http://localhost:8001/data")
        data = response.json()
        # Expecting different data structure
        return {"result": data["items"][0]["name"]}  # KeyError - wrong structure
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''
            (services[0][1].service_path / "app.py").write_text(python_code)
            
            # Start both services
            for _, env, _ in services:
                env.start_service()
                
            # Trigger the error
            services[0][1].trigger_error()
            
        scenario = HealingScenario(
            name="Polyglot Microservice Healing",
            description="Cross-service communication error between Python and JavaScript",
            error_type="KeyError",
            target_service="polyglot_system",
            error_trigger=trigger_polyglot_error,
            validation_checks=[
                lambda: all(check_service_healthy_on_port(port) for _, _, port in services),
                lambda: check_interservice_communication(services)
            ],
            expected_fix_type="polyglot_integration_fix"
        )
        
        try:
            runner = HealingScenarioRunner(services[0][1])  # Use Python service for healing
            result = await runner.run_scenario(scenario)
            
            metrics_collector.record_healing_duration("polyglot_healing", result.duration)
            metrics_collector.record_success_rate("polyglot_healing", result.success)
            
            assert result.error_detected, "Polyglot error should have been detected"
            
        finally:
            for _, env, _ in services:
                env.cleanup()
                
    @pytest.mark.asyncio
    async def test_framework_migration_healing(self, metrics_collector):
        """Test healing during framework migration scenarios."""
        env = TestEnvironment()
        env.setup()
        
        # Create a service with mixed framework code (simulating migration)
        migration_code = '''
from fastapi import FastAPI, HTTPException
from flask import Flask, jsonify  # Mixed frameworks!
import uvicorn

# FastAPI app
app = FastAPI()

# Legacy Flask code mixed in
flask_app = Flask(__name__)

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/error")
async def trigger_error():
    # Trying to use Flask patterns in FastAPI
    try:
        # This won't work - mixing Flask response with FastAPI
        return jsonify({"data": "value"})  # Wrong response type
    except Exception as e:
        # Also trying to use Flask error handling
        return flask_app.response_class(
            response=json.dumps({"error": str(e)}),
            status=500,
            mimetype='application/json'
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''
        
        def trigger_migration_error():
            (env.service_path / "app.py").write_text(migration_code)
            env.start_service()
            env.trigger_error()
            
        scenario = HealingScenario(
            name="Framework Migration Healing",
            description="Service with mixed FastAPI/Flask code during migration",
            error_type="TypeError",
            target_service="migration_service",
            error_trigger=trigger_migration_error,
            validation_checks=[
                check_service_healthy,
                lambda: check_no_syntax_errors(env.service_path / "app.py"),
                lambda: check_consistent_framework(env.service_path / "app.py")
            ],
            expected_fix_type="framework_migration_fix"
        )
        
        runner = HealingScenarioRunner(env)
        
        try:
            result = await runner.run_scenario(scenario)
            
            metrics_collector.record_healing_duration("migration_healing", result.duration)
            metrics_collector.record_success_rate("migration_healing", result.success)
            
            assert result.error_detected, "Migration error should have been detected"
            assert result.patch_generated, "Migration fix should have been generated"
            
        finally:
            env.cleanup()


def check_language_fix_applied(service_path: Path, language: str) -> bool:
    """Check if language-specific fix was properly applied."""
    if language == "javascript":
        content = (service_path / "app.js").read_text()
        # Check for null checks or optional chaining
        return "if (" in content or "?." in content
    elif language == "go":
        content = (service_path / "main.go").read_text()
        # Check for nil checks
        return "if data != nil" in content or "make(map" in content
    elif language == "java":
        content = (service_path / "SimpleService.java").read_text()
        # Check for null checks
        return "if (data != null)" in content or "!= null" in content
    return False
    

def check_service_healthy_on_port(port: int) -> bool:
    """Check if a service is healthy on a specific port."""
    import requests
    try:
        response = requests.get(f"http://localhost:{port}/health", timeout=5)
        return response.status_code == 200
    except:
        return False
        

def check_interservice_communication(services: List[tuple]) -> bool:
    """Check if services can communicate properly."""
    import requests
    try:
        # Try to make a call from Python service
        response = requests.get("http://localhost:8000/error", timeout=5)
        # If it returns 200, the error was fixed
        return response.status_code == 200
    except:
        return False
        

def check_consistent_framework(file_path: Path) -> bool:
    """Check if the service uses a consistent framework."""
    content = file_path.read_text()
    
    # Check that it's not mixing FastAPI and Flask
    has_fastapi = "from fastapi import" in content
    has_flask = "from flask import" in content
    
    # Should use only one framework
    return has_fastapi != has_flask  # XOR - only one should be true