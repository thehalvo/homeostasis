"""
Tests for the Ansible language plugin.
"""
import pytest

from modules.analysis.plugins.ansible_plugin import (
    AnsibleLanguagePlugin, 
    AnsibleExceptionHandler, 
    AnsiblePatchGenerator
)


class TestAnsibleExceptionHandler:
    """Test the Ansible exception handler."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.handler = AnsibleExceptionHandler()
    
    def test_analyze_syntax_error(self):
        """Test analysis of syntax errors."""
        error_data = {
            "error_type": "AnsibleError",
            "message": "ERROR! Syntax Error while loading YAML",
            "file_path": "playbook.yml",
            "line_number": 10,
            "column_number": 5
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "ansible"
        assert analysis["subcategory"] == "syntax"
        assert analysis["confidence"] == "high"
        assert "syntax" in analysis["tags"]
    
    def test_analyze_module_error(self):
        """Test analysis of module errors."""
        error_data = {
            "error_type": "AnsibleError",
            "message": "ERROR! couldn't resolve module/action 'mymodule'",
            "file_path": "tasks/main.yml",
            "line_number": 15,
            "column_number": 8
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "ansible"
        assert analysis["subcategory"] == "module"
        assert analysis["confidence"] == "high"
        assert "module" in analysis["tags"]
    
    def test_analyze_variable_error(self):
        """Test analysis of variable errors."""
        error_data = {
            "error_type": "AnsibleError",
            "message": "ERROR! The task includes an option with an undefined variable",
            "file_path": "vars/main.yml",
            "line_number": 20,
            "column_number": 12
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "ansible"
        assert analysis["subcategory"] == "variable"
        assert analysis["confidence"] == "high"
        assert "variable" in analysis["tags"]
    
    def test_analyze_template_error(self):
        """Test analysis of template errors."""
        error_data = {
            "error_type": "AnsibleError",
            "message": "ERROR! template error while templating string",
            "file_path": "templates/config.j2",
            "line_number": 25,
            "column_number": 15
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "ansible"
        assert analysis["subcategory"] == "template"
        assert analysis["confidence"] == "high"
        assert "template" in analysis["tags"]
    
    def test_analyze_role_error(self):
        """Test analysis of role errors."""
        error_data = {
            "error_type": "AnsibleError",
            "message": "ERROR! the role 'myrole' was not found",
            "file_path": "site.yml",
            "line_number": 30,
            "column_number": 10
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "ansible"
        assert analysis["subcategory"] == "role"
        assert analysis["confidence"] == "high"
        assert "role" in analysis["tags"]
    
    def test_analyze_inventory_error(self):
        """Test analysis of inventory errors."""
        error_data = {
            "error_type": "AnsibleError",
            "message": "ERROR! Attempted to read inventory file but it was empty",
            "file_path": "inventory/hosts",
            "line_number": 1,
            "column_number": 1
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "ansible"
        assert analysis["subcategory"] == "inventory"
        assert analysis["confidence"] == "high"
        assert "inventory" in analysis["tags"]
    
    def test_analyze_unknown_error(self):
        """Test analysis of unknown errors."""
        error_data = {
            "error_type": "AnsibleError",
            "message": "Some unknown error message",
            "file_path": "playbook.yml",
            "line_number": 45,
            "column_number": 1
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "ansible"
        assert analysis["subcategory"] == "unknown"
        assert analysis["confidence"] == "low"
        assert "generic" in analysis["tags"]


class TestAnsiblePatchGenerator:
    """Test the Ansible patch generator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = AnsiblePatchGenerator()
    
    def test_generate_syntax_fix(self):
        """Test generation of syntax fixes."""
        error_data = {
            "message": "ERROR! Syntax Error while loading YAML",
            "file_path": "playbook.yml"
        }
        
        analysis = {
            "root_cause": "ansible_syntax_error",
            "subcategory": "syntax",
            "confidence": "high"
        }
        
        syntax_patch = self.generator.generate_patch(error_data, analysis, "")
        
        assert syntax_patch is not None
        assert syntax_patch["type"] == "suggestion"
        assert "yaml" in syntax_patch["description"].lower() or "syntax" in syntax_patch["description"].lower()
    
    def test_generate_module_fix(self):
        """Test generation of module fixes."""
        error_data = {
            "message": "ERROR! couldn't resolve module/action 'mymodule'",
            "file_path": "tasks/main.yml"
        }
        
        analysis = {
            "root_cause": "ansible_module_error",
            "subcategory": "module",
            "confidence": "high"
        }
        
        module_patch = self.generator.generate_patch(error_data, analysis, "")
        
        assert module_patch is not None
        assert module_patch["type"] == "suggestion"
        assert "module" in module_patch["description"].lower()
    
    def test_generate_variable_fix(self):
        """Test generation of variable fixes."""
        error_data = {
            "message": "ERROR! The task includes an option with an undefined variable",
            "file_path": "vars/main.yml"
        }
        
        analysis = {
            "root_cause": "ansible_variable_error",
            "subcategory": "variable",
            "confidence": "high"
        }
        
        variable_patch = self.generator.generate_patch(error_data, analysis, "")
        
        assert variable_patch is not None
        assert variable_patch["type"] == "suggestion"
        assert "variable" in variable_patch["description"].lower()
    
    def test_generate_template_fix(self):
        """Test generation of template fixes."""
        error_data = {
            "message": "ERROR! template error while templating string",
            "file_path": "templates/config.j2"
        }
        
        analysis = {
            "root_cause": "ansible_template_error",
            "subcategory": "template",
            "confidence": "high"
        }
        
        template_patch = self.generator.generate_patch(error_data, analysis, "")
        
        assert template_patch is not None
        assert template_patch["type"] == "suggestion"
        assert "template" in template_patch["description"].lower() or "jinja" in template_patch["description"].lower()
    
    def test_generate_role_fix(self):
        """Test generation of role fixes."""
        error_data = {
            "message": "ERROR! the role 'myrole' was not found",
            "file_path": "site.yml"
        }
        
        analysis = {
            "root_cause": "ansible_role_error",
            "subcategory": "role",
            "confidence": "high"
        }
        
        role_patch = self.generator.generate_patch(error_data, analysis, "")
        
        assert role_patch is not None
        assert role_patch["type"] == "suggestion"
        assert "role" in role_patch["description"].lower()


class TestAnsibleLanguagePlugin:
    """Test the Ansible language plugin."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.plugin = AnsibleLanguagePlugin()
    
    def test_plugin_metadata(self):
        """Test plugin metadata."""
        assert self.plugin.get_language_id() == "ansible"
        assert self.plugin.get_language_name() == "Ansible"
        assert self.plugin.get_language_version() == "2.9+"
        
        frameworks = self.plugin.get_supported_frameworks()
        assert "ansible" in frameworks
        assert "ansible-playbook" in frameworks
    
    def test_normalize_error(self):
        """Test error normalization."""
        ansible_error = {
            "error_type": "AnsibleError",
            "message": "Test error",
            "file": "playbook.yml",
            "line": 10,
            "column": 5,
            "description": "Test error description"
        }
        
        normalized = self.plugin.normalize_error(ansible_error)
        
        assert normalized["language"] == "ansible"
        assert normalized["error_type"] == "AnsibleError"
        assert normalized["message"] == "Test error"
        assert normalized["file_path"] == "playbook.yml"
        assert normalized["line_number"] == 10
        assert normalized["column_number"] == 5
    
    def test_denormalize_error(self):
        """Test error denormalization."""
        standard_error = {
            "language": "ansible",
            "error_type": "AnsibleError",
            "message": "Test error",
            "file_path": "playbook.yml",
            "line_number": 10,
            "column_number": 5,
            "severity": "high"
        }
        
        ansible_error = self.plugin.denormalize_error(standard_error)
        
        assert ansible_error["error_type"] == "AnsibleError"
        assert ansible_error["message"] == "Test error"
        assert ansible_error["file_path"] == "playbook.yml"
        assert ansible_error["line_number"] == 10
        assert ansible_error["column_number"] == 5
        assert ansible_error["file"] == "playbook.yml"  # Alternative format
        assert ansible_error["line"] == 10  # Alternative format
    
    def test_analyze_error(self):
        """Test error analysis."""
        error_data = {
            "error_type": "AnsibleError",
            "message": "ERROR! couldn't resolve module/action 'mymodule'",
            "file_path": "tasks/main.yml",
            "line_number": 15,
            "column_number": 8
        }
        
        analysis = self.plugin.analyze_error(error_data)
        
        assert analysis["plugin"] == "ansible"
        assert analysis["language"] == "ansible"
        assert analysis["plugin_version"] == "1.0.0"
        assert analysis["category"] == "ansible"
        assert analysis["subcategory"] == "module"
    
    def test_generate_fix(self):
        """Test fix generation."""
        analysis = {
            "root_cause": "ansible_module_error",
            "subcategory": "module",
            "confidence": "high",
            "suggested_fix": "Fix module not found"
        }
        
        context = {
            "error_data": {
                "message": "ERROR! couldn't resolve module/action 'mymodule'",
                "file_path": "tasks/main.yml"
            },
            "source_code": "- name: Test task\n  mymodule:\n    param: value"
        }
        
        fix = self.plugin.generate_fix(analysis, context)
        
        assert fix is not None
        assert fix["type"] == "suggestion"
        assert "description" in fix
    
    def test_supported_extensions(self):
        """Test supported file extensions."""
        assert ".yml" in self.plugin.supported_extensions
        assert ".yaml" in self.plugin.supported_extensions
    
    def test_error_analysis_with_invalid_data(self):
        """Test error analysis with invalid data."""
        error_data = {}
        
        analysis = self.plugin.analyze_error(error_data)
        
        assert analysis["plugin"] == "ansible"
        assert analysis["language"] == "ansible"
        # Should handle invalid data gracefully
        assert "category" in analysis
        assert "confidence" in analysis


if __name__ == "__main__":
    pytest.main([__file__])