# Terraform Integration

The Homeostasis Terraform Language Plugin provides error analysis and patch generation for Terraform infrastructure-as-code configurations. It supports multiple cloud providers and provides intelligent error detection for common Terraform issues.

## Overview

The Terraform plugin enables Homeostasis to:
- Analyze Terraform HCL syntax and configuration errors
- Detect provider-specific issues and authentication problems
- Handle resource configuration and dependency errors
- Provide intelligent suggestions for infrastructure optimization
- Support multi-cloud and multi-provider environments

## Supported Providers

### Major Cloud Providers
- **AWS** - Amazon Web Services with resource support
- **Azure (AzureRM)** - Microsoft Azure Resource Manager
- **Google Cloud (GCP)** - Google Cloud Platform services
- **Kubernetes** - Kubernetes resource management
- **Helm** - Kubernetes package management

### Additional Providers
- **Vault** - HashiCorp Vault secrets management
- **Consul** - Service discovery and configuration
- **Nomad** - Container orchestration
- **Random** - Random value generation
- **Local** - Local file and command execution
- **External** - External data sources
- **HTTP** - HTTP data sources
- **TLS** - TLS certificate management
- **Archive** - Archive file creation

## Key Features

### Error Detection Categories

1. **Syntax Errors**
   - Invalid HCL syntax
   - Missing required arguments
   - Unsupported argument errors
   - Invalid function calls

2. **Resource Errors**
   - Resource creation failures
   - Resource not found errors
   - Invalid resource configurations
   - Dependency resolution issues

3. **Provider Errors**
   - Provider authentication failures
   - Provider version constraints
   - Missing provider configurations
   - Invalid provider settings

4. **State Management Errors**
   - State lock conflicts
   - State corruption issues
   - Backend configuration errors
   - State migration problems

5. **Variable and Module Errors**
   - Undefined variables
   - Variable validation failures
   - Module sourcing issues
   - Circular dependencies

### Exit Code Analysis

Terraform exit codes and their meanings:

```
0 - Success
1 - General error
2 - Plan differs (non-zero diff)
3 - Configuration error
4 - Backend error
5 - State lock error
```

## Usage Examples

### Basic Terraform Error Analysis

```python
from homeostasis import analyze_error

# Example Terraform error
error_data = {
    "error_type": "TerraformError",
    "message": "Error creating EC2 instance: InvalidInstanceID.NotFound",
    "command": "terraform apply",
    "provider": "aws",
    "exit_code": 1
}

analysis = analyze_error(error_data, language="terraform")
print(analysis["suggested_fix"])
# Output: "Fix resource configuration or check resource state"
```

### Provider Authentication Error

```python
# AWS authentication error
auth_error = {
    "error_type": "ProviderError",
    "message": "NoCredentialsError: Unable to locate credentials",
    "provider": "aws",
    "command": "terraform plan"
}

analysis = analyze_error(auth_error, language="terraform")
```

### Configuration Syntax Error

```python
# HCL syntax error
syntax_error = {
    "error_type": "ConfigurationError",
    "message": "Argument \"ami\" is required, but no definition was found",
    "line_number": 15,
    "config_file": "main.tf"
}

analysis = analyze_error(syntax_error, language="terraform")
```

## Configuration

### Plugin Configuration

Configure the Terraform plugin in your `homeostasis.yaml`:

```yaml
plugins:
  terraform:
    enabled: true
    supported_providers: [aws, azurerm, google, kubernetes, helm]
    error_detection:
      syntax_checking: true
      resource_validation: true
      provider_checking: true
      state_validation: true
    patch_generation:
      auto_suggest_fixes: true
      provider_specific: true
      best_practices: true
```

### Provider-Specific Settings

```yaml
plugins:
  terraform:
    aws:
      version: "~> 4.0"
      regions: [us-east-1, us-west-2, eu-west-1]
    azurerm:
      version: "~> 3.0"
      features: true
    google:
      version: "~> 4.0"
      project_validation: true
```

## Error Pattern Recognition

### Syntax Error Patterns

```hcl
# Missing required argument
resource "aws_instance" "example" {
  # Error: ami argument is required
  instance_type = "t3.micro"
}

# Fix: Add required ami argument
resource "aws_instance" "example" {
  ami           = "ami-12345678"
  instance_type = "t3.micro"
}

# Invalid function call
locals {
  # Error: invalid function name
  result = invalid_function("test")
}

# Fix: Use valid function
locals {
  result = upper("test")
}
```

### Resource Configuration Errors

```hcl
# Invalid resource reference
resource "aws_security_group_rule" "example" {
  type              = "ingress"
  from_port         = 80
  to_port           = 80
  protocol          = "tcp"
  security_group_id = aws_security_group.nonexistent.id  # Error: resource doesn't exist
}

# Fix: Reference existing resource
resource "aws_security_group" "example" {
  name_prefix = "example-"
}

resource "aws_security_group_rule" "example" {
  type              = "ingress"
  from_port         = 80
  to_port           = 80
  protocol          = "tcp"
  security_group_id = aws_security_group.example.id
}
```

### Variable Definition Errors

```hcl
# Undefined variable usage
resource "aws_instance" "example" {
  ami           = var.ami_id  # Error: variable not defined
  instance_type = "t3.micro"
}

# Fix: Define the variable
variable "ami_id" {
  type        = string
  description = "AMI ID for the instance"
}

# Or provide default value
variable "ami_id" {
  type        = string
  description = "AMI ID for the instance"
  default     = "ami-12345678"
}
```

## Provider-Specific Features

### AWS Provider

- **IAM Permission Analysis**: Detects missing IAM permissions
- **Resource Dependencies**: Validates AWS resource relationships
- **Region Validation**: Checks availability zones and regions
- **Service Limits**: Warns about service limit violations

```hcl
# AWS-specific error handling
resource "aws_instance" "example" {
  ami                    = "ami-12345678"
  instance_type          = "t3.micro"
  vpc_security_group_ids = [aws_security_group.example.id]
  subnet_id              = aws_subnet.example.id
  
  # Common AWS errors:
  # - InvalidAMI.NotFound: AMI doesn't exist in region
  # - InvalidSubnet.NotFound: Subnet doesn't exist
  # - UnauthorizedOperation: Insufficient IAM permissions
}
```

### Azure Provider

- **Resource Group Validation**: Ensures resource groups exist
- **Subscription Checks**: Validates subscription access
- **Feature Validation**: Checks enabled Azure features
- **Location Validation**: Validates Azure regions/locations

```hcl
# Azure-specific configuration
resource "azurerm_virtual_machine" "example" {
  name                = "example-vm"
  location            = azurerm_resource_group.example.location
  resource_group_name = azurerm_resource_group.example.name
  vm_size             = "Standard_DS1_v2"
  
  # Common Azure errors:
  # - ResourceGroupNotFound: Resource group doesn't exist
  # - InvalidSubscriptionId: Invalid subscription
  # - LocationNotAvailableForResourceType: Location not supported
}
```

### Google Cloud Provider

- **Project Validation**: Validates GCP project access
- **API Enablement**: Checks required APIs are enabled
- **Service Account**: Validates service account permissions
- **Resource Quotas**: Monitors resource quota usage

```hcl
# GCP-specific configuration
resource "google_compute_instance" "example" {
  name         = "example-instance"
  machine_type = "e2-micro"
  zone         = "us-central1-a"
  
  boot_disk {
    initialize_params {
      image = "debian-cloud/debian-10"
    }
  }
  
  # Common GCP errors:
  # - googleapi: Error 403: Forbidden
  # - googleapi: Error 404: Resource not found
  # - quotaExceeded: Resource quota exceeded
}
```

## Best Practices

### Code Organization

```hcl
# Good Terraform structure
terraform {
  required_version = ">= 1.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.0"
    }
  }
  
  backend "s3" {
    bucket = "my-terraform-state"
    key    = "infrastructure/terraform.tfstate"
    region = "us-east-1"
  }
}

# Provider configuration
provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Environment = var.environment
      Project     = var.project_name
      ManagedBy   = "terraform"
    }
  }
}

# Data sources
data "aws_ami" "amazon_linux" {
  most_recent = true
  owners      = ["amazon"]
  
  filter {
    name   = "name"
    values = ["amzn2-ami-hvm-*-x86_64-gp2"]
  }
}

# Resources
resource "aws_instance" "web" {
  ami           = data.aws_ami.amazon_linux.id
  instance_type = var.instance_type
  
  vpc_security_group_ids = [aws_security_group.web.id]
  subnet_id              = var.subnet_id
  
  user_data = file("${path.module}/user_data.sh")
  
  tags = {
    Name = "${var.project_name}-web-server"
  }
}
```

### Variable Management

```hcl
# variables.tf
variable "environment" {
  type        = string
  description = "Environment name (dev, staging, prod)"
  
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be dev, staging, or prod."
  }
}

variable "instance_type" {
  type        = string
  description = "EC2 instance type"
  default     = "t3.micro"
  
  validation {
    condition = can(regex("^t3\\.", var.instance_type))
    error_message = "Instance type must be t3 family."
  }
}

# terraform.tfvars
environment   = "dev"
instance_type = "t3.micro"
```

### State Management

```hcl
# Backend configuration
terraform {
  backend "s3" {
    bucket         = "my-terraform-state"
    key            = "infrastructure/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "terraform-state-lock"
  }
}

# Remote state data source
data "terraform_remote_state" "vpc" {
  backend = "s3"
  
  config = {
    bucket = "my-terraform-state"
    key    = "networking/terraform.tfstate"
    region = "us-east-1"
  }
}
```

## Integration Examples

### CI/CD Pipeline Integration

```yaml
# GitHub Actions workflow
name: Terraform
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  terraform:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Setup Terraform
      uses: hashicorp/setup-terraform@v1
      with:
        terraform_version: 1.5.0
    
    - name: Terraform Init
      run: terraform init
      
    - name: Terraform Validate
      run: terraform validate
      
    - name: Terraform Plan
      run: |
        if ! terraform plan -detailed-exitcode; then
          # Analyze Terraform errors with Homeostasis
          python -c "
          from homeostasis import analyze_error
          import subprocess
          
          result = subprocess.run(['terraform', 'plan'], 
                                capture_output=True, text=True)
          
          if result.returncode != 0:
              error_data = {
                  'error_type': 'TerraformError',
                  'message': result.stderr,
                  'command': 'terraform plan',
                  'exit_code': result.returncode
              }
              
              analysis = analyze_error(error_data, language='terraform')
              print(f'Terraform Error Analysis: {analysis[\"suggested_fix\"]}')
          "
          exit 1
        fi
      env:
        AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
        AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
    
    - name: Terraform Apply
      if: github.ref == 'refs/heads/main'
      run: terraform apply -auto-approve
```

### Python Integration

```python
import subprocess
from homeostasis import analyze_error

def run_terraform_command(command, working_dir="."):
    """Run Terraform command with error analysis."""
    full_command = f"terraform {command}"
    
    try:
        result = subprocess.run(
            full_command.split(),
            cwd=working_dir,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            error_data = {
                "error_type": "TerraformError",
                "message": result.stderr,
                "command": full_command,
                "exit_code": result.returncode
            }
            
            analysis = analyze_error(error_data, language="terraform")
            
            print(f"Terraform command failed: {full_command}")
            print(f"Error: {result.stderr}")
            print(f"Suggested fix: {analysis['suggested_fix']}")
            
            # Handle specific error types
            if analysis["category"] == "provider":
                print("Check provider authentication and configuration")
            elif analysis["category"] == "resource":
                print("Review resource configuration and dependencies")
            elif analysis["exit_code"] == 5:
                print("State lock detected - wait or force unlock if safe")
            
            return None
            
        return result.stdout
        
    except Exception as e:
        print(f"Failed to execute Terraform command: {e}")
        return None

# Usage examples
output = run_terraform_command("plan")
if output:
    print("Terraform plan successful")

output = run_terraform_command("apply -auto-approve")
if output:
    print("Terraform apply successful")
```

### Pre-commit Hook

```yaml
# .pre-commit-config.yaml
repos:
- repo: https://github.com/antonbabenko/pre-commit-terraform
  rev: v1.77.0
  hooks:
  - id: terraform_fmt
  - id: terraform_validate
  - id: terraform_tflint

- repo: local
  hooks:
  - id: terraform-error-analysis
    name: Terraform Error Analysis
    entry: python
    language: python
    files: \.tf$
    args:
    - -c
    - |
      import subprocess
      import sys
      from homeostasis import analyze_error
      
      try:
          result = subprocess.run(['terraform', 'validate'], 
                                capture_output=True, text=True)
          if result.returncode != 0:
              error_data = {
                  'error_type': 'TerraformError',
                  'message': result.stderr,
                  'command': 'terraform validate',
                  'exit_code': result.returncode
              }
              analysis = analyze_error(error_data, language='terraform')
              print(f'Terraform validation failed: {analysis["suggested_fix"]}')
              sys.exit(1)
      except Exception as e:
          print(f'Error running terraform validate: {e}')
          sys.exit(1)
```

## Troubleshooting

### Common Issues

1. **Provider Authentication**: Ensure correct credentials are configured
2. **State Locks**: Check for stuck state locks and resolve conflicts
3. **Resource Dependencies**: Verify resource references and dependencies
4. **Version Constraints**: Check provider and Terraform version compatibility

### Debug Mode

Enable Terraform debug logging:

```bash
# Enable debug logging
export TF_LOG=DEBUG
export TF_LOG_PATH=terraform.log

# Run Terraform commands
terraform plan
terraform apply
```

### State Management Issues

```bash
# Check state
terraform state list
terraform state show resource.name

# Force unlock state (use with caution)
terraform force-unlock LOCK_ID

# Import existing resources
terraform import aws_instance.example i-12345678

# Remove resource from state
terraform state rm aws_instance.example
```

### Provider Issues

```bash
# Check provider versions
terraform version

# Initialize with upgrade
terraform init -upgrade

# Validate configuration
terraform validate

# Format configuration
terraform fmt -recursive
```

## Performance Considerations

- **State Size**: Monitor state file size and consider state splitting
- **Provider Caching**: Use provider plugin cache for faster init
- **Parallel Execution**: Leverage Terraform's parallel resource creation
- **Resource Targeting**: Use `-target` for selective operations

## Security Considerations

1. **State File Security**: Encrypt state files and limit access
2. **Secret Management**: Use external secret management systems
3. **Provider Credentials**: Secure provider authentication
4. **Resource Policies**: Implement least-privilege access policies

```hcl
# Secure practices
terraform {
  backend "s3" {
    bucket  = "secure-terraform-state"
    key     = "infrastructure/terraform.tfstate"
    region  = "us-east-1"
    encrypt = true
    
    # State locking
    dynamodb_table = "terraform-state-lock"
    
    # Access control
    acl = "bucket-owner-full-control"
  }
}

# Use data sources for sensitive values
data "aws_secretsmanager_secret_version" "db_password" {
  secret_id = "database-password"
}

resource "aws_db_instance" "example" {
  # ... other configuration ...
  password = data.aws_secretsmanager_secret_version.db_password.secret_string
}
```

## Contributing

To extend the Terraform plugin:

1. Add new provider patterns to provider detection
2. Implement provider-specific error handlers
3. Add test cases for new error types
4. Update documentation with provider examples

## Related Documentation

- [Error Schema](error_schema.md) - Standard error format
- [Plugin Architecture](plugin_architecture.md) - Plugin development guide
- [Cloud Integration](cloud_integration.md) - Multi-cloud support
- [Infrastructure as Code](iac_best_practices.md) - IaC best practices
- [CI/CD Integration](cicd/) - Continuous integration setup