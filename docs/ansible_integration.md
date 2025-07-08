# Ansible Integration

The Homeostasis Ansible Language Plugin provides error analysis and patch generation for Ansible playbooks, roles, and configuration management tasks. It supports the complete Ansible ecosystem with intelligent error detection for common automation issues.

## Overview

The Ansible plugin enables Homeostasis to:
- Analyze Ansible playbook syntax and structure errors
- Detect module configuration and parameter issues
- Handle variable templating and Jinja2 problems
- Provide intelligent suggestions for automation optimization
- Support Ansible best practices and security guidelines

## Supported Ansible Components

### Core Components
- **Ansible Core** - Base automation engine
- **Ansible Collections** - Packaged automation content
- **Ansible Galaxy** - Community automation hub
- **Ansible Vault** - Secrets management
- **Ansible Runner** - Execution interface

### Tools and Extensions
- **Molecule** - Testing framework for Ansible roles
- **Ansible Lint** - Best practices checker
- **AWX/Ansible Tower** - Enterprise automation platform
- **Ansible Navigator** - Content creation tool

## Key Features

### Error Detection Categories

1. **YAML Syntax Errors**
   - Invalid YAML structure
   - Indentation problems
   - Malformed mappings and sequences
   - Character encoding issues

2. **Module Errors**
   - Invalid module parameters
   - Missing required parameters
   - Mutually exclusive parameters
   - Module not found errors

3. **Variable and Templating Errors**
   - Undefined variables
   - Jinja2 template syntax errors
   - Variable scope issues
   - Type mismatch errors

4. **Inventory and Connection Errors**
   - Host pattern matching failures
   - SSH connection problems
   - Privilege escalation issues
   - Inventory parsing errors

5. **Task and Playbook Structure Errors**
   - Invalid task definitions
   - Missing action statements
   - Conflicting task options
   - Handler notification issues

6. **Role and Collection Errors**
   - Role dependency problems
   - Collection not found
   - Version conflicts
   - Import/include failures

## Usage Examples

### Basic Ansible Error Analysis

```python
from homeostasis import analyze_error

# Example Ansible playbook error
error_data = {
    "error_type": "AnsibleError",
    "message": "'mysql_user' is not a legal parameter in the 'user' module",
    "task_name": "Create database user",
    "module_name": "user",
    "playbook_path": "site.yml",
    "line_number": 25
}

analysis = analyze_error(error_data, language="ansible")
print(analysis["suggested_fix"])
# Output: "Remove 'mysql_user' parameter from user module"
```

### Variable Error Detection

```python
# Undefined variable error
variable_error = {
    "error_type": "AnsibleUndefinedVariable",
    "message": "'db_password' is undefined",
    "task_name": "Configure database connection",
    "playbook_path": "database.yml"
}

analysis = analyze_error(variable_error, language="ansible")
```

### SSH Connection Error

```python
# Connection failure
connection_error = {
    "error_type": "ConnectionFailure",
    "message": "Failed to connect to the host via ssh: Permission denied",
    "host": "web01.example.com",
    "command": "ansible-playbook site.yml"
}

analysis = analyze_error(connection_error, language="ansible")
```

## Configuration

### Plugin Configuration

Configure the Ansible plugin in your `homeostasis.yaml`:

```yaml
plugins:
  ansible:
    enabled: true
    supported_versions: ["2.9+", "4.0+", "5.0+"]
    error_detection:
      syntax_checking: true
      module_validation: true
      variable_checking: true
      inventory_validation: true
    patch_generation:
      auto_suggest_fixes: true
      best_practices: true
      security_recommendations: true
```

### Ansible-Specific Settings

```yaml
plugins:
  ansible:
    collections:
      auto_download: true
      validate_versions: true
    vault:
      check_encryption: true
      suggest_vault_usage: true
    testing:
      molecule_integration: true
      lint_integration: true
```

## Error Pattern Recognition

### YAML Syntax Errors

```yaml
# Indentation error (tabs instead of spaces)
---
- hosts: all
  tasks:
	- name: Install package  # Error: tabs not allowed
	  package:
		name: nginx

# Fix: Use spaces for indentation
---
- hosts: all
  tasks:
    - name: Install package
      package:
        name: nginx

# Missing colon in mapping
---
- hosts: all
  tasks
    - name: Install package  # Error: missing colon after 'tasks'
      package:
        name: nginx

# Fix: Add colon
---
- hosts: all
  tasks:
    - name: Install package
      package:
        name: nginx
```

### Module Parameter Errors

```yaml
# Invalid parameter for module
---
- hosts: all
  tasks:
    - name: Create user
      user:
        name: john
        mysql_user: yes  # Error: mysql_user not valid for user module

# Fix: Remove invalid parameter
---
- hosts: all
  tasks:
    - name: Create user
      user:
        name: john
        state: present

# Missing required parameter
---
- hosts: all
  tasks:
    - name: Copy file
      copy:
        dest: /tmp/file.txt  # Error: missing 'src' or 'content'

# Fix: Add required parameter
---
- hosts: all
  tasks:
    - name: Copy file
      copy:
        src: file.txt
        dest: /tmp/file.txt
```

### Variable and Template Errors

```yaml
# Undefined variable
---
- hosts: all
  tasks:
    - name: Install package
      package:
        name: "{{ package_name }}"  # Error: package_name not defined

# Fix: Define variable
---
- hosts: all
  vars:
    package_name: nginx
  tasks:
    - name: Install package
      package:
        name: "{{ package_name }}"

# Jinja2 template syntax error
---
- hosts: all
  tasks:
    - name: Set configuration
      template:
        src: config.j2
        dest: /etc/app/config.conf
      vars:
        config_value: "{{ some_var | default('default') }"  # Error: missing closing }}

# Fix: Close template expression
---
- hosts: all
  tasks:
    - name: Set configuration
      template:
        src: config.j2
        dest: /etc/app/config.conf
      vars:
        config_value: "{{ some_var | default('default') }}"
```

## Module-Specific Error Handling

### Common Module Issues

```yaml
# File module issues
---
- hosts: all
  tasks:
    - name: Create directory
      file:
        path: /opt/app
        state: directory
        owner: root
        group: root
        mode: '0755'
        # Common issues: incorrect mode format, invalid owner/group

# Package module issues
---
- hosts: all
  tasks:
    - name: Install packages
      package:
        name:
          - nginx
          - python3-pip
        state: present
        # Common issues: package not available, wrong package name

# Service module issues
---
- hosts: all
  tasks:
    - name: Start service
      service:
        name: nginx
        state: started
        enabled: yes
        # Common issues: service not found, systemd vs init conflicts

# Template module issues
---
- hosts: all
  tasks:
    - name: Deploy configuration
      template:
        src: nginx.conf.j2
        dest: /etc/nginx/nginx.conf
        backup: yes
        # Common issues: template not found, variable errors
```

### Best Practices Implementation

```yaml
# Good Ansible practices
---
- name: Web server setup
  hosts: webservers
  become: yes
  vars:
    nginx_version: "1.20"
    sites_available: "/etc/nginx/sites-available"
    sites_enabled: "/etc/nginx/sites-enabled"
  
  pre_tasks:
    - name: Update package cache
      package:
        update_cache: yes
      when: ansible_os_family == "Debian"

  tasks:
    - name: Install Nginx
      package:
        name: nginx
        state: present
      notify: restart nginx

    - name: Ensure nginx is running
      service:
        name: nginx
        state: started
        enabled: yes

    - name: Deploy site configuration
      template:
        src: site.conf.j2
        dest: "{{ sites_available }}/{{ site_name }}"
        backup: yes
      notify: reload nginx

    - name: Enable site
      file:
        src: "{{ sites_available }}/{{ site_name }}"
        dest: "{{ sites_enabled }}/{{ site_name }}"
        state: link
      notify: reload nginx

  handlers:
    - name: restart nginx
      service:
        name: nginx
        state: restarted

    - name: reload nginx
      service:
        name: nginx
        state: reloaded
```

## Integration Examples

### CI/CD Pipeline Integration

```yaml
# GitHub Actions workflow for Ansible
name: Ansible Lint and Test
on: [push, pull_request]

jobs:
  ansible-lint:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Setup Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    
    - name: Install Ansible and dependencies
      run: |
        pip install ansible ansible-lint molecule[docker]
        ansible-galaxy install -r requirements.yml
    
    - name: Run Ansible Lint
      run: |
        if ! ansible-lint playbooks/; then
          # Analyze Ansible errors with Homeostasis
          python -c "
          import subprocess
          from homeostasis import analyze_error
          
          result = subprocess.run(['ansible-lint', 'playbooks/'], 
                                capture_output=True, text=True)
          
          if result.returncode != 0:
              for line in result.stdout.split('\n'):
                  if 'ERROR' in line or 'WARNING' in line:
                      error_data = {
                          'error_type': 'AnsibleLintError',
                          'message': line,
                          'command': 'ansible-lint playbooks/'
                      }
                      analysis = analyze_error(error_data, language='ansible')
                      print(f'Lint Error: {analysis[\"suggested_fix\"]}')
          "
          exit 1
        fi
    
    - name: Run Molecule Tests
      run: |
        cd roles/common
        molecule test
```

### Python Integration with Ansible Runner

```python
import ansible_runner
from homeostasis import analyze_error

def run_ansible_playbook(playbook_path, inventory_path, extra_vars=None):
    """Run Ansible playbook with error analysis."""
    try:
        result = ansible_runner.run(
            playbook=playbook_path,
            inventory=inventory_path,
            extravars=extra_vars or {},
            quiet=False,
            verbosity=1
        )
        
        if result.status != "successful":
            # Analyze failures
            for event in result.events:
                if event.get('event') == 'runner_on_failed':
                    task_name = event.get('event_data', {}).get('task', 'Unknown task')
                    error_msg = event.get('event_data', {}).get('res', {}).get('msg', 'Unknown error')
                    module_name = event.get('event_data', {}).get('task_action', 'unknown')
                    
                    error_data = {
                        "error_type": "AnsibleTaskFailure",
                        "message": error_msg,
                        "task_name": task_name,
                        "module_name": module_name,
                        "playbook_path": playbook_path
                    }
                    
                    analysis = analyze_error(error_data, language="ansible")
                    
                    print(f"Task failed: {task_name}")
                    print(f"Error: {error_msg}")
                    print(f"Suggested fix: {analysis['suggested_fix']}")
            
            return False
        
        print(f"Playbook executed successfully: {playbook_path}")
        return True
        
    except Exception as e:
        error_data = {
            "error_type": "AnsibleRunnerError",
            "message": str(e),
            "playbook_path": playbook_path
        }
        
        analysis = analyze_error(error_data, language="ansible")
        print(f"Ansible runner error: {analysis['suggested_fix']}")
        return False

# Usage
success = run_ansible_playbook(
    "site.yml",
    "inventory/production",
    {"nginx_version": "1.20"}
)
```

### Molecule Testing Integration

```python
# molecule/default/molecule.yml
---
dependency:
  name: galaxy
driver:
  name: docker
platforms:
  - name: instance
    image: geerlingguy/docker-ubuntu2004-ansible:latest
    pre_build_image: true
provisioner:
  name: ansible
  config_options:
    defaults:
      callback_whitelist: homeostasis_callback
verifier:
  name: ansible

# Custom callback plugin for error analysis
# plugins/callback/homeostasis_callback.py
from ansible.plugins.callback import CallbackBase
from homeostasis import analyze_error

class CallbackModule(CallbackBase):
    def v2_runner_on_failed(self, result, ignore_errors=False):
        error_data = {
            "error_type": "AnsibleTaskFailure",
            "message": result._result.get('msg', 'Task failed'),
            "task_name": result._task.get_name(),
            "module_name": result._task.action,
            "host": result._host.get_name()
        }
        
        analysis = analyze_error(error_data, language="ansible")
        
        self._display.display(
            f"Homeostasis Analysis: {analysis['suggested_fix']}", 
            color='red'
        )
```

## Ansible Vault Integration

### Vault Error Handling

```python
from ansible_vault import Vault
from homeostasis import analyze_error

def decrypt_vault_file(vault_file, vault_password):
    """Decrypt Ansible Vault file with error analysis."""
    try:
        vault = Vault(vault_password)
        
        with open(vault_file, 'r') as f:
            encrypted_content = f.read()
        
        decrypted_content = vault.load(encrypted_content)
        return decrypted_content
        
    except Exception as e:
        error_data = {
            "error_type": "AnsibleVaultError",
            "message": str(e),
            "vault_file": vault_file
        }
        
        analysis = analyze_error(error_data, language="ansible")
        
        print(f"Vault decryption failed: {vault_file}")
        print(f"Error: {str(e)}")
        print(f"Suggested fix: {analysis['suggested_fix']}")
        
        return None

# Usage
vault_data = decrypt_vault_file("group_vars/production/vault.yml", "vault_password")
```

### Vault Best Practices

```yaml
# Good vault practices
---
# group_vars/production/vars.yml (unencrypted)
database_host: "db.production.example.com"
database_port: 5432
database_name: "myapp_production"
database_user: "myapp_user"

# group_vars/production/vault.yml (encrypted)
$ANSIBLE_VAULT;1.1;AES256
66633...  # Encrypted content containing:
# vault_database_password: "super_secret_password"
# vault_api_key: "secret_api_key"

# Usage in playbook
---
- hosts: production
  vars:
    database_password: "{{ vault_database_password }}"
    api_key: "{{ vault_api_key }}"
  tasks:
    - name: Configure database connection
      template:
        src: database.conf.j2
        dest: /etc/myapp/database.conf
```

## Performance Optimization

### Playbook Optimization

```yaml
# Optimized playbook structure
---
- name: Optimized web server setup
  hosts: webservers
  gather_facts: no  # Skip if facts not needed
  strategy: free    # Parallel execution
  serial: 50%       # Batch processing for large inventories
  
  pre_tasks:
    - name: Gather minimal facts
      setup:
        gather_subset:
          - "!all"
          - "!hardware"
          - "network"
          - "virtual"
      when: ansible_facts is not defined

  tasks:
    - name: Install packages in batch
      package:
        name:
          - nginx
          - python3-pip
          - git
        state: present
      # Batch installation is more efficient

    - name: Configure services
      template:
        src: "{{ item.src }}"
        dest: "{{ item.dest }}"
      loop:
        - { src: nginx.conf.j2, dest: /etc/nginx/nginx.conf }
        - { src: app.conf.j2, dest: /etc/nginx/sites-available/app }
      notify: restart nginx
      # Loop for similar tasks

    - name: Check service status
      service:
        name: nginx
        state: started
      register: nginx_status
      changed_when: false  # Mark as non-changing task

  handlers:
    - name: restart nginx
      service:
        name: nginx
        state: restarted
      listen: restart nginx  # Handler grouping
```

### Inventory Optimization

```ini
# Optimized inventory structure
[webservers]
web01.example.com ansible_host=10.0.1.10
web02.example.com ansible_host=10.0.1.11
web03.example.com ansible_host=10.0.1.12

[databases]
db01.example.com ansible_host=10.0.2.10
db02.example.com ansible_host=10.0.2.11

[production:children]
webservers
databases

[production:vars]
ansible_user=deploy
ansible_ssh_private_key_file=~/.ssh/deploy_key
ansible_ssh_common_args='-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null'
```

## Troubleshooting

### Common Issues

1. **SSH Connection Problems**: Check SSH keys, user permissions, and network connectivity
2. **Variable Scope Issues**: Understand variable precedence and scope rules
3. **Module Dependencies**: Ensure required packages are installed on target hosts
4. **Privilege Escalation**: Configure sudo/become settings correctly

### Debug Techniques

```bash
# Enable verbose output
ansible-playbook -vvv site.yml

# Test connectivity
ansible all -m ping

# Check syntax
ansible-playbook --syntax-check site.yml

# Dry run
ansible-playbook --check site.yml

# Step through playbook
ansible-playbook --step site.yml

# Debug specific tasks
ansible-playbook site.yml --tags debug

# Limit to specific hosts
ansible-playbook site.yml --limit web01.example.com
```

### Error Analysis Script

```python
#!/usr/bin/env python3
import sys
import json
from homeostasis import analyze_error

def analyze_ansible_log(log_file):
    """Analyze Ansible log file for errors."""
    try:
        with open(log_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if any(keyword in line.lower() for keyword in ['error', 'failed', 'fatal']):
                    error_data = {
                        "error_type": "AnsibleLogError",
                        "message": line.strip(),
                        "line_number": line_num,
                        "log_file": log_file
                    }
                    
                    analysis = analyze_error(error_data, language="ansible")
                    
                    print(f"Line {line_num}: {line.strip()}")
                    print(f"Suggested fix: {analysis['suggested_fix']}")
                    print("-" * 50)
    
    except FileNotFoundError:
        print(f"Log file not found: {log_file}")
    except Exception as e:
        print(f"Error analyzing log: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: analyze_ansible_log.py <log_file>")
        sys.exit(1)
    
    analyze_ansible_log(sys.argv[1])
```

## Security Considerations

### Secure Playbook Practices

```yaml
# Security best practices
---
- name: Secure server configuration
  hosts: all
  become: yes
  vars:
    allowed_users: ['admin', 'deploy']
    
  tasks:
    - name: Ensure SSH key authentication
      lineinfile:
        path: /etc/ssh/sshd_config
        line: "PasswordAuthentication no"
        state: present
      notify: restart ssh

    - name: Configure firewall
      ufw:
        rule: allow
        port: "{{ item }}"
        proto: tcp
      loop:
        - '22'   # SSH
        - '80'   # HTTP
        - '443'  # HTTPS

    - name: Create admin users
      user:
        name: "{{ item }}"
        groups: sudo
        shell: /bin/bash
        append: yes
      loop: "{{ allowed_users }}"
      no_log: true  # Don't log sensitive operations

    - name: Set up SSH keys
      authorized_key:
        user: "{{ item }}"
        key: "{{ lookup('file', 'files/ssh_keys/' + item + '.pub') }}"
      loop: "{{ allowed_users }}"
```

## Contributing

To extend the Ansible plugin:

1. Add new error patterns to Ansible error detection
2. Implement module-specific error handlers
3. Add support for new Ansible collections
4. Update documentation with examples

## Related Documentation

- [Error Schema](error_schema.md) - Standard error format
- [Plugin Architecture](plugin_architecture.md) - Plugin development guide
- [YAML/JSON Integration](yaml_json_integration.md) - Configuration file handling
- [CI/CD Integration](cicd/) - Continuous integration setup
- [Infrastructure as Code](iac_best_practices.md) - IaC best practices