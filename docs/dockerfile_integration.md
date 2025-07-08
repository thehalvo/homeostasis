# Dockerfile Integration

The Homeostasis Dockerfile Language Plugin provides error analysis and patch generation for Docker container configurations. It supports Docker build processes, security best practices, and container optimization patterns.

## Overview

The Dockerfile plugin enables Homeostasis to:
- Analyze Dockerfile syntax and instruction errors
- Detect container build and runtime issues
- Handle base image and dependency problems
- Provide intelligent suggestions for container optimization
- Support Docker security and performance best practices

## Supported Container Technologies

### Core Docker
- **Docker Engine** - Standard Docker runtime
- **Docker Buildx** - Extended build capabilities
- **Docker Compose** - Multi-container applications
- **Docker Swarm** - Container orchestration

### Container Runtimes
- **Podman** - Daemonless container engine
- **Buildah** - Container image building
- **Skopeo** - Container image operations
- **containerd** - Industry-standard container runtime

### Orchestration Platforms
- **Kubernetes** - Container orchestration
- **OpenShift** - Enterprise Kubernetes platform
- **Nomad** - Container scheduler

## Key Features

### Error Detection Categories

1. **Syntax Errors**
   - Invalid Dockerfile instructions
   - Missing instruction arguments
   - Malformed instruction syntax
   - Invalid escape sequences

2. **Build Context Errors**
   - Build context preparation failures
   - File not found in build context
   - Path outside build context
   - .dockerignore issues

3. **Base Image Errors**
   - Image pull failures
   - Authentication errors
   - Registry connectivity issues
   - Image tag/manifest problems

4. **Instruction Errors**
   - COPY/ADD failures
   - RUN command failures
   - WORKDIR issues
   - USER permission problems

5. **Network and Storage Errors**
   - Network connectivity failures
   - Disk space issues
   - Permission denied errors
   - Mount point problems

6. **Security and Best Practice Violations**
   - Running as root user
   - Unnecessary privileges
   - Security vulnerabilities
   - Layer optimization issues

## Usage Examples

### Basic Dockerfile Error Analysis

```python
from homeostasis import analyze_error

# Example Dockerfile build error
error_data = {
    "error_type": "DockerError",
    "message": "COPY failed: no such file or directory",
    "build_step": "COPY app.py /app/",
    "command": "docker build .",
    "dockerfile_content": "FROM python:3.9\nCOPY app.py /app/"
}

analysis = analyze_error(error_data, language="dockerfile")
print(analysis["suggested_fix"])
# Output: "Check source file exists in build context"
```

### Base Image Error

```python
# Image pull error
image_error = {
    "error_type": "ImagePullError",
    "message": "pull access denied for myregistry/private-image",
    "build_step": "FROM myregistry/private-image:latest",
    "command": "docker build"
}

analysis = analyze_error(image_error, language="dockerfile")
```

### Build Context Error

```python
# Build context error
context_error = {
    "error_type": "BuildContextError",
    "message": "forbidden path outside the build context: ../config.json",
    "build_step": "COPY ../config.json /app/",
    "dockerfile_content": "COPY ../config.json /app/"
}

analysis = analyze_error(context_error, language="dockerfile")
```

## Configuration

### Plugin Configuration

Configure the Dockerfile plugin in your `homeostasis.yaml`:

```yaml
plugins:
  dockerfile:
    enabled: true
    supported_runtimes: [docker, podman, buildah]
    error_detection:
      syntax_checking: true
      build_validation: true
      security_scanning: true
      best_practices: true
    patch_generation:
      auto_suggest_fixes: true
      security_improvements: true
      optimization_hints: true
```

### Build-Specific Settings

```yaml
plugins:
  dockerfile:
    build:
      cache_optimization: true
      multi_stage_builds: true
      layer_minimization: true
    security:
      scan_base_images: true
      check_privileges: true
      validate_users: true
    performance:
      suggest_optimizations: true
      analyze_layer_size: true
```

## Error Pattern Recognition

### Syntax Error Patterns

```dockerfile
# Unknown instruction
FORM python:3.9  # Error: should be FROM
COPY app.py /app/

# Fix: Correct instruction name
FROM python:3.9
COPY app.py /app/

# Missing instruction argument
WORKDIR  # Error: missing directory argument
COPY app.py /app/

# Fix: Add required argument
WORKDIR /app
COPY app.py /app/

# Invalid instruction format
RUN apt-get update \
    && apt-get install -y python3  # Error: missing continuation
    python3-pip

# Fix: Proper line continuation
RUN apt-get update \
    && apt-get install -y python3 \
    python3-pip
```

### Build Context Errors

```dockerfile
# File not in build context
FROM python:3.9
COPY /usr/local/bin/script.sh /app/  # Error: absolute path outside context

# Fix: Copy from build context
FROM python:3.9
COPY scripts/script.sh /app/

# Parent directory access
FROM python:3.9
COPY ../config.json /app/  # Error: outside build context

# Fix: Include file in build context
FROM python:3.9
COPY config.json /app/
```

### Instruction-Specific Errors

```dockerfile
# COPY/ADD errors
FROM python:3.9
COPY nonexistent.py /app/  # Error: source file doesn't exist

# Fix: Ensure file exists or use correct path
FROM python:3.9
COPY app.py /app/

# RUN command failures
FROM ubuntu:20.04
RUN apt-get update && apt-get install -y nonexistent-package  # Error: package not found

# Fix: Use correct package names
FROM ubuntu:20.04
RUN apt-get update && apt-get install -y python3 python3-pip

# USER instruction issues
FROM ubuntu:20.04
USER nonexistent_user  # Error: user doesn't exist

# Fix: Create user first
FROM ubuntu:20.04
RUN useradd -m -s /bin/bash appuser
USER appuser
```

## Best Practices Integration

### Security Best Practices

```dockerfile
# Poor security practices
FROM ubuntu:latest  # Issue: using latest tag
RUN apt-get update && apt-get install -y python3
# Issue: running as root user

# Improved security
FROM ubuntu:20.04  # Specific version tag
RUN apt-get update && apt-get install -y python3 \
    && rm -rf /var/lib/apt/lists/*  # Clean up cache
RUN useradd -m -s /bin/bash appuser  # Create non-root user
USER appuser  # Switch to non-root user
```

### Multi-Stage Build Optimization

```dockerfile
# Single stage (larger image)
FROM python:3.9
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app.py"]

# Multi-stage optimization
# Build stage
FROM python:3.9 as builder
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Runtime stage
FROM python:3.9-slim
COPY --from=builder /root/.local /root/.local
COPY app.py .
ENV PATH=/root/.local/bin:$PATH
CMD ["python", "app.py"]
```

### Layer Optimization

```dockerfile
# Poor layer optimization (multiple RUN instructions)
FROM ubuntu:20.04
RUN apt-get update
RUN apt-get install -y python3
RUN apt-get install -y python3-pip
RUN apt-get clean

# Optimized layers (combined RUN instructions)
FROM ubuntu:20.04
RUN apt-get update \
    && apt-get install -y python3 python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
```

## Container Runtime Integration

### Docker Build Integration

```python
import subprocess
from homeostasis import analyze_error

def build_docker_image(dockerfile_path=".", tag="myapp:latest"):
    """Build Docker image with error analysis."""
    try:
        result = subprocess.run(
            ["docker", "build", "-t", tag, dockerfile_path],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            # Read Dockerfile for context
            dockerfile_content = ""
            try:
                with open(f"{dockerfile_path}/Dockerfile", "r") as f:
                    dockerfile_content = f.read()
            except FileNotFoundError:
                pass
            
            error_data = {
                "error_type": "DockerBuildError",
                "message": result.stderr,
                "command": f"docker build -t {tag} {dockerfile_path}",
                "dockerfile_content": dockerfile_content,
                "exit_code": result.returncode
            }
            
            analysis = analyze_error(error_data, language="dockerfile")
            
            print(f"Docker build failed: {tag}")
            print(f"Error: {result.stderr}")
            print(f"Suggested fix: {analysis['suggested_fix']}")
            
            # Provide specific guidance based on error type
            if "COPY failed" in result.stderr:
                print("Check that source files exist in the build context")
            elif "pull access denied" in result.stderr:
                print("Check Docker registry authentication")
            elif "no space left on device" in result.stderr:
                print("Free up disk space or use docker system prune")
            
            return False
            
        print(f"Successfully built {tag}")
        return True
        
    except Exception as e:
        print(f"Failed to build Docker image: {e}")
        return False

# Usage
success = build_docker_image("./myapp", "myapp:v1.0.0")
```

### Docker Compose Integration

```yaml
# docker-compose.yml with build error handling
version: '3.8'

services:
  app:
    build:
      context: .
      dockerfile: Dockerfile
    image: myapp:latest
    ports:
      - "8080:8080"
    
  # Health check and restart policies
  web:
    build: ./web
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### CI/CD Pipeline Integration

```yaml
# GitHub Actions workflow
name: Docker Build and Test
on: [push, pull_request]

jobs:
  docker-build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v1
    
    - name: Build Docker image
      run: |
        if ! docker build -t myapp:test .; then
          # Analyze Docker build errors with Homeostasis
          python -c "
          from homeostasis import analyze_error
          import subprocess
          
          # Capture build output
          result = subprocess.run(['docker', 'build', '-t', 'myapp:test', '.'],
                                capture_output=True, text=True)
          
          if result.returncode != 0:
              error_data = {
                  'error_type': 'DockerBuildError',
                  'message': result.stderr,
                  'command': 'docker build -t myapp:test .',
                  'exit_code': result.returncode
              }
              
              analysis = analyze_error(error_data, language='dockerfile')
              print(f'Docker Build Error: {analysis[\"suggested_fix\"]}')
          "
          exit 1
        fi
    
    - name: Test Docker container
      run: |
        docker run --rm myapp:test python -c "import app; print('App imported successfully')"
    
    - name: Security scan
      uses: anchore/scan-action@v3
      with:
        image: "myapp:test"
        fail-build: false
```

## Security Analysis

### Common Security Issues

```dockerfile
# Security issues
FROM ubuntu:latest  # Issue: latest tag is unpredictable
RUN apt-get update && apt-get install -y curl
ADD http://example.com/script.sh /tmp/  # Issue: downloading from internet
RUN chmod +x /tmp/script.sh && /tmp/script.sh
# Issue: no USER instruction - running as root

# Security improvements
FROM ubuntu:20.04  # Specific version
RUN apt-get update && apt-get install -y curl \
    && rm -rf /var/lib/apt/lists/*  # Clean package cache
COPY --chown=1000:1000 script.sh /app/script.sh  # Copy from build context
RUN chmod +x /app/script.sh

# Create and use non-root user
RUN groupadd -r appgroup && useradd -r -g appgroup appuser
USER appuser
WORKDIR /app
CMD ["./script.sh"]
```

### Security Scanning Integration

```python
import subprocess
import json
from homeostasis import analyze_error

def security_scan_image(image_name):
    """Perform security scan with error analysis."""
    try:
        # Run security scan (using grype as example)
        result = subprocess.run(
            ["grype", image_name, "-o", "json"],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            error_data = {
                "error_type": "SecurityScanError",
                "message": result.stderr,
                "command": f"grype {image_name}",
                "image": image_name
            }
            
            analysis = analyze_error(error_data, language="dockerfile")
            print(f"Security scan failed: {analysis['suggested_fix']}")
            return None
        
        # Parse scan results
        scan_results = json.loads(result.stdout)
        vulnerabilities = scan_results.get("matches", [])
        
        print(f"Found {len(vulnerabilities)} vulnerabilities in {image_name}")
        
        # Categorize vulnerabilities
        critical = [v for v in vulnerabilities if v.get("vulnerability", {}).get("severity") == "Critical"]
        high = [v for v in vulnerabilities if v.get("vulnerability", {}).get("severity") == "High"]
        
        if critical:
            print(f"Critical vulnerabilities: {len(critical)}")
        if high:
            print(f"High severity vulnerabilities: {len(high)}")
        
        return scan_results
        
    except Exception as e:
        print(f"Security scan error: {e}")
        return None

# Usage
scan_results = security_scan_image("myapp:latest")
```

## Performance Optimization

### Build Cache Optimization

```dockerfile
# Poor cache utilization
FROM python:3.9
COPY . /app/  # Copies everything, invalidates cache often
RUN pip install -r /app/requirements.txt
WORKDIR /app
CMD ["python", "app.py"]

# Optimized cache utilization
FROM python:3.9
WORKDIR /app

# Copy requirements first (changes less frequently)
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code last (changes more frequently)
COPY . .
CMD ["python", "app.py"]
```

### Build Performance Analysis

```python
import time
import subprocess
from homeostasis import analyze_error

def analyze_build_performance(dockerfile_path=".", tag="perf-test"):
    """Analyze Docker build performance."""
    start_time = time.time()
    
    try:
        # Build with timing
        result = subprocess.run(
            ["docker", "build", "--progress=plain", "-t", tag, dockerfile_path],
            capture_output=True,
            text=True
        )
        
        build_time = time.time() - start_time
        
        if result.returncode != 0:
            error_data = {
                "error_type": "DockerBuildError",
                "message": result.stderr,
                "build_time": build_time,
                "dockerfile_path": dockerfile_path
            }
            
            analysis = analyze_error(error_data, language="dockerfile")
            print(f"Build failed after {build_time:.2f}s: {analysis['suggested_fix']}")
            return None
        
        # Analyze build output for performance insights
        build_output = result.stdout
        
        # Extract layer information
        layers = []
        for line in build_output.split('\n'):
            if 'Step' in line and '/' in line:
                layers.append(line.strip())
        
        print(f"Build completed in {build_time:.2f}s")
        print(f"Total layers: {len(layers)}")
        
        # Check image size
        size_result = subprocess.run(
            ["docker", "images", "--format", "table {{.Size}}", tag],
            capture_output=True,
            text=True
        )
        
        if size_result.returncode == 0:
            size_info = size_result.stdout.strip().split('\n')
            if len(size_info) > 1:
                image_size = size_info[1]
                print(f"Image size: {image_size}")
        
        return {
            "build_time": build_time,
            "layers": layers,
            "success": True
        }
        
    except Exception as e:
        print(f"Performance analysis failed: {e}")
        return None

# Usage
perf_results = analyze_build_performance("./myapp")
```

## Troubleshooting

### Common Issues

1. **Build Context Size**: Large build contexts slow down builds
2. **Layer Caching**: Inefficient layer ordering invalidates cache
3. **Base Image Issues**: Wrong architecture or unavailable images
4. **Network Problems**: Registry connectivity or DNS issues

### Debug Techniques

```bash
# Debug build with detailed output
docker build --progress=plain --no-cache -t debug-image .

# Inspect failed build layers
docker run -it --rm <failed-layer-id> /bin/bash

# Check build context size
docker build --progress=plain . 2>&1 | grep "Sending build context"

# Analyze image layers
docker history myapp:latest

# Inspect image details
docker inspect myapp:latest
```

### .dockerignore Optimization

```
# .dockerignore
# Exclude development files
node_modules/
.git/
.gitignore
README.md
Dockerfile
.dockerignore

# Exclude build artifacts
*.log
*.tmp
dist/
build/

# Exclude OS files
.DS_Store
Thumbs.db

# Exclude IDE files
.vscode/
.idea/
*.swp
*.swo
```

## Contributing

To extend the Dockerfile plugin:

1. Add new error patterns to Dockerfile error detection
2. Implement container runtime-specific handlers
3. Add security and performance analysis rules
4. Update documentation with examples

## Related Documentation

- [Error Schema](error_schema.md) - Standard error format
- [Plugin Architecture](plugin_architecture.md) - Plugin development guide
- [Container Security](container_security.md) - Container security best practices
- [CI/CD Integration](cicd/) - Continuous integration setup
- [Kubernetes Integration](kubernetes_integration.md) - Container orchestration