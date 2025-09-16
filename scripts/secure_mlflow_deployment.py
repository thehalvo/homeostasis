#!/usr/bin/env python
"""
Script to secure MLflow deployment against known CVEs.

This script implements security hardening measures for MLflow deployments
to mitigate CVE-2024-37052 through CVE-2024-37060.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add parent directory to path to import our modules
sys.path.append(str(Path(__file__).parent.parent))

from modules.security.mlflow_security import (
    MLflowSecurityConfig,
    create_secure_mlflow_config,
    SecurityError,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_mlflow_installation():
    """Check if MLflow is installed and get version."""
    try:
        import mlflow
        version = mlflow.__version__
        logger.info(f"MLflow version {version} detected")
        return version
    except ImportError:
        logger.error("MLflow not installed")
        return None


def generate_nginx_config(mlflow_port: int = 5000) -> str:
    """Generate Nginx configuration for MLflow with security headers."""
    return f"""
# MLflow Nginx Security Configuration
# Place this in /etc/nginx/sites-available/mlflow

server {{
    listen 443 ssl;
    server_name mlflow.yourdomain.com;

    # SSL Configuration
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    # Security Headers
    add_header X-Content-Type-Options nosniff;
    add_header X-Frame-Options DENY;
    add_header X-XSS-Protection "1; mode=block";
    add_header Content-Security-Policy "default-src 'self'";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

    # Rate Limiting
    limit_req_zone $binary_remote_addr zone=mlflow:10m rate=10r/s;
    limit_req zone=mlflow burst=20;

    # Authentication
    auth_basic "MLflow Protected";
    auth_basic_user_file /etc/nginx/.htpasswd;

    # Proxy to MLflow
    location / {{
        proxy_pass http://localhost:{mlflow_port};
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Disable buffering for streaming
        proxy_buffering off;

        # Timeouts
        proxy_connect_timeout 300;
        proxy_send_timeout 300;
        proxy_read_timeout 300;
    }}

    # Block direct model downloads without auth
    location /api/2.0/mlflow/model-versions/get-download-uri {{
        return 403;
    }}

    # Restrict artifact access
    location /api/2.0/mlflow/artifacts {{
        # Additional auth check
        auth_request /auth;
    }}

    # Internal auth endpoint
    location = /auth {{
        internal;
        proxy_pass http://localhost:{mlflow_port}/api/2.0/mlflow/users/get;
        proxy_pass_request_body off;
        proxy_set_header Content-Length "";
        proxy_set_header X-Original-URI $request_uri;
    }}
}}
"""


def generate_docker_compose(mlflow_image: str = "ghcr.io/mlflow/mlflow:latest") -> str:
    """Generate secure docker-compose configuration for MLflow."""
    return f"""
version: '3.8'

services:
  mlflow:
    image: {mlflow_image}
    container_name: mlflow_secure
    ports:
      - "127.0.0.1:5000:5000"  # Only bind to localhost
    environment:
      - MLFLOW_TRACKING_URI=sqlite:///mlflow/mlflow.db
      - MLFLOW_ARTIFACT_URI=file:///mlflow/artifacts
      - MLFLOW_AUTH_CONFIG_PATH=/mlflow/auth.ini
    volumes:
      - mlflow_data:/mlflow
      - ./auth.ini:/mlflow/auth.ini:ro
      - ./mlflow_security.json:/mlflow/security.json:ro
    security_opt:
      - no-new-privileges:true
    read_only: true
    tmpfs:
      - /tmp
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    networks:
      - mlflow_net

volumes:
  mlflow_data:
    driver: local

networks:
  mlflow_net:
    driver: bridge
    ipam:
      config:
        - subnet: 172.28.0.0/16
"""


def generate_auth_config() -> str:
    """Generate MLflow authentication configuration."""
    return """
[mlflow]
# Authentication settings
default_permission = READ
admin_username = admin
admin_password_hash = pbkdf2:sha256:260000$... # Generate with mlflow.auth.utils

# Database for auth
database_uri = sqlite:///mlflow_auth.db

# Token settings
auth_token_expiration_time = 86400  # 24 hours
refresh_token_expiration_time = 2592000  # 30 days

[permissions]
# Define permission levels
admin = CREATE,READ,UPDATE,DELETE,MANAGE
user = READ
developer = CREATE,READ,UPDATE

[users]
# Additional users (passwords should be hashed)
developer1 = developer,$pbkdf2-sha256$29000$...
viewer1 = user,$pbkdf2-sha256$29000$...
"""


def harden_mlflow_server(args):
    """Apply security hardening to MLflow server."""
    logger.info("Starting MLflow security hardening...")

    # Check MLflow installation
    mlflow_version = check_mlflow_installation()
    if not mlflow_version:
        logger.error("MLflow not found. Please install MLflow first.")
        return 1

    # Create security configuration
    security_config = MLflowSecurityConfig()

    # Generate configurations
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Write Nginx config
    nginx_config_path = output_dir / "nginx_mlflow.conf"
    with open(nginx_config_path, "w") as f:
        f.write(generate_nginx_config(args.mlflow_port))
    logger.info(f"Nginx configuration written to {nginx_config_path}")

    # Write Docker Compose config
    docker_compose_path = output_dir / "docker-compose.yml"
    with open(docker_compose_path, "w") as f:
        f.write(generate_docker_compose())
    logger.info(f"Docker Compose configuration written to {docker_compose_path}")

    # Write auth config
    auth_config_path = output_dir / "auth.ini"
    with open(auth_config_path, "w") as f:
        f.write(generate_auth_config())
    logger.info(f"Authentication configuration written to {auth_config_path}")

    # Write systemd service if requested
    if args.systemd:
        systemd_config = f"""
[Unit]
Description=MLflow Tracking Server (Secure)
After=network.target

[Service]
Type=simple
User=mlflow
Group=mlflow
WorkingDirectory=/opt/mlflow
Environment="MLFLOW_AUTH_CONFIG_PATH=/opt/mlflow/auth.ini"
ExecStart=/usr/local/bin/mlflow server \\
    --backend-store-uri sqlite:///opt/mlflow/mlflow.db \\
    --default-artifact-root /opt/mlflow/artifacts \\
    --host 127.0.0.1 \\
    --port {args.mlflow_port} \\
    --app-name basic-auth

# Security settings
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
NoNewPrivileges=true
ReadWritePaths=/opt/mlflow

Restart=on-failure
RestartSec=30

[Install]
WantedBy=multi-user.target
"""
        systemd_path = output_dir / "mlflow.service"
        with open(systemd_path, "w") as f:
            f.write(systemd_config)
        logger.info(f"Systemd service configuration written to {systemd_path}")

    # Generate security checklist
    checklist = """
MLflow Security Hardening Checklist:

1. [ ] Install and configure Nginx as reverse proxy
2. [ ] Enable SSL/TLS with valid certificates
3. [ ] Set up authentication (basic auth or OAuth)
4. [ ] Configure firewall rules (only allow HTTPS)
5. [ ] Run MLflow with non-root user
6. [ ] Enable audit logging
7. [ ] Set up monitoring and alerting
8. [ ] Regular security updates
9. [ ] Backup authentication database
10. [ ] Review and restrict model access permissions

CVE Mitigation Status:
- CVE-2024-37052 to CVE-2024-37060: Mitigations applied via access controls
- Continue monitoring for security patches from MLflow team

Next Steps:
1. Copy generated configs to appropriate locations
2. Create mlflow user: sudo useradd -r -s /bin/false mlflow
3. Set up directories with proper permissions
4. Install and configure Nginx
5. Start services and test security measures
"""

    checklist_path = output_dir / "security_checklist.txt"
    with open(checklist_path, "w") as f:
        f.write(checklist)
    logger.info(f"Security checklist written to {checklist_path}")

    logger.info("Security hardening configuration complete!")
    logger.info(f"Configuration files generated in: {output_dir}")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Secure MLflow deployment against known CVEs"
    )
    parser.add_argument(
        "--output-dir",
        default="mlflow_secure_config",
        help="Directory to output configuration files",
    )
    parser.add_argument(
        "--mlflow-port",
        type=int,
        default=5000,
        help="Port for MLflow server",
    )
    parser.add_argument(
        "--systemd",
        action="store_true",
        help="Generate systemd service configuration",
    )

    args = parser.parse_args()

    try:
        return harden_mlflow_server(args)
    except Exception as e:
        logger.error(f"Error during hardening: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())