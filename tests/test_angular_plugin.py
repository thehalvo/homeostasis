"""Tests for Angular plugin functionality."""

import pytest

# Import the Angular plugin
from modules.analysis.plugins.angular_plugin import (
    AngularExceptionHandler,
    AngularLanguagePlugin,
    AngularPatchGenerator,
)


class TestAngularLanguagePlugin:
    """Test cases for Angular language plugin."""

    def setup_method(self):
        """Set up test fixtures."""
        self.plugin = AngularLanguagePlugin()

    def test_plugin_initialization(self):
        """Test plugin initializes correctly."""
        assert self.plugin.language == "angular"
        assert self.plugin.get_language_id() == "angular"
        assert self.plugin.get_language_name() == "Angular"
        assert self.plugin.VERSION == "1.0.0"
        assert ".ts" in self.plugin.supported_extensions
        assert ".html" in self.plugin.supported_extensions

    def test_can_handle_angular_errors(self):
        """Test plugin can identify Angular errors."""
        # Test Angular framework detection
        error_data = {
            "framework": "angular",
            "message": "No provider for UserService",
            "stack_trace": "Error at AppComponent",
        }
        assert self.plugin.can_handle(error_data) is True

        # Test Angular-specific patterns
        angular_error = {
            "message": "Cannot bind to 'ngModel' since it isn't a known property",
            "stack_trace": "at UserComponent.component.ts:25:10",
        }
        assert self.plugin.can_handle(angular_error) is True

        # Test dependency injection patterns
        di_error = {
            "message": "No provider for HttpClient!",
            "stack_trace": "StaticInjectorError",
        }
        assert self.plugin.can_handle(di_error) is True

        # Test NgRx patterns
        ngrx_error = {
            "message": "Store has not been provided",
            "stack_trace": "@ngrx/store",
        }
        assert self.plugin.can_handle(ngrx_error) is True

    def test_cannot_handle_non_angular_errors(self):
        """Test plugin correctly rejects non-Angular errors."""
        # Test non-Angular error
        error_data = {
            "framework": "react",
            "message": "Cannot read property 'map' of undefined",
            "stack_trace": "at Component.render",
        }
        assert self.plugin.can_handle(error_data) is False

        # Test generic JavaScript error
        js_error = {
            "message": "ReferenceError: variable is not defined",
            "stack_trace": "at main.js:10:5",
        }
        assert self.plugin.can_handle(js_error) is False

    def test_analyze_dependency_injection_error(self):
        """Test analysis of dependency injection errors."""
        error_data = {
            "error_type": "Error",
            "message": "No provider for HttpClient!",
            "stack_trace": ["StaticInjectorError at AppModule"],
        }

        analysis = self.plugin.analyze_error(error_data)

        assert analysis["category"] == "angular"
        assert analysis["subcategory"] == "dependency_injection"
        assert analysis["confidence"] == "high"
        assert "provider" in analysis["suggested_fix"].lower()
        assert "angular" in analysis["tags"]
        assert "dependency-injection" in analysis["tags"]

    def test_analyze_ngrx_error(self):
        """Test analysis of NgRx errors."""
        error_data = {
            "error_type": "Error",
            "message": "Store has not been provided",
            "stack_trace": ["@ngrx/store"],
        }

        analysis = self.plugin.analyze_error(error_data)

        assert analysis["category"] == "angular"
        assert analysis["subcategory"] == "ngrx"
        assert analysis["confidence"] == "high"
        assert "storemodule" in analysis["suggested_fix"].lower()
        assert "ngrx" in analysis["tags"]

    def test_analyze_template_binding_error(self):
        """Test analysis of template binding errors."""
        error_data = {
            "error_type": "Error",
            "message": "Can't bind to 'ngModel' since it isn't a known property",
            "stack_trace": ["at UserComponent.template.html:5"],
        }

        analysis = self.plugin.analyze_error(error_data)

        assert analysis["category"] == "angular"
        assert analysis["subcategory"] == "templates"
        assert analysis["confidence"] == "high"
        assert "input" in analysis["suggested_fix"].lower()
        assert "template" in analysis["tags"]

    def test_generate_fix(self):
        """Test fix generation for Angular errors."""
        error_data = {
            "message": "No provider for UserService!",
            "stack_trace": "StaticInjectorError",
        }

        analysis = {
            "root_cause": "angular_no_provider",
            "category": "angular",
            "subcategory": "dependency_injection",
        }

        source_code = """
        @Component({
            selector: 'app-user',
            template: '<div>User Component</div>'
        })
        export class UserComponent {
            constructor(private userService: UserService) {}
        }
        """

        fix = self.plugin.generate_fix(error_data, analysis, source_code)

        assert fix is not None
        assert fix["type"] == "suggestion"
        assert "provider" in fix["description"].lower()
        assert len(fix["fix_commands"]) > 0

    def test_get_language_info(self):
        """Test language info retrieval."""
        info = self.plugin.get_language_info()

        assert info["language"] == "angular"
        assert info["version"] == "1.0.0"
        assert ".ts" in info["supported_extensions"]
        assert "angular" in info["supported_frameworks"]
        assert len(info["features"]) > 0
        assert "dependency injection" in info["features"][0].lower()


class TestAngularExceptionHandler:
    """Test cases for Angular exception handler."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = AngularExceptionHandler()

    def test_handler_initialization(self):
        """Test exception handler initializes correctly."""
        assert hasattr(self.handler, "rule_categories")
        assert "dependency_injection" in self.handler.rule_categories
        assert "ngrx" in self.handler.rule_categories
        assert "templates" in self.handler.rule_categories

    def test_analyze_dependency_injection_error(self):
        """Test dependency injection error analysis."""
        error_data = {
            "message": "No provider for HttpClient!",
            "stack_trace": "StaticInjectorError",
        }

        analysis = self.handler.analyze_dependency_injection_error(error_data)

        assert analysis["category"] == "angular"
        assert analysis["subcategory"] == "dependency_injection"
        assert analysis["confidence"] == "high"
        assert analysis["root_cause"] == "angular_no_provider"
        assert "provider" in analysis["suggested_fix"].lower()

    def test_analyze_ngrx_error(self):
        """Test NgRx error analysis."""
        error_data = {
            "message": "Action must have a type",
            "stack_trace": "@ngrx/store",
        }

        analysis = self.handler.analyze_ngrx_error(error_data)

        assert analysis["category"] == "angular"
        assert analysis["subcategory"] == "ngrx"
        assert analysis["confidence"] == "high"
        assert analysis["root_cause"] == "ngrx_action_no_type"
        assert "createaction" in analysis["suggested_fix"].lower()

    def test_analyze_template_binding_error(self):
        """Test template binding error analysis."""
        error_data = {
            "message": "Cannot read property 'name' of undefined",
            "stack_trace": "at UserComponent.template",
        }

        analysis = self.handler.analyze_template_binding_error(error_data)

        assert analysis["category"] == "angular"
        assert analysis["subcategory"] == "templates"
        assert analysis["confidence"] == "high"
        assert analysis["root_cause"] == "angular_template_property_undefined"
        assert "safe navigation" in analysis["suggested_fix"].lower()

    def test_generic_analysis(self):
        """Test generic error analysis fallback."""
        error_data = {
            "error_type": "UnknownError",
            "message": "Some unknown Angular error",
            "stack_trace": "at SomeComponent",
        }

        analysis = self.handler._generic_analysis(error_data)

        assert analysis["category"] == "angular"
        assert analysis["subcategory"] == "unknown"
        assert analysis["confidence"] == "low"
        assert analysis["rule_id"] == "angular_generic_handler"


class TestAngularPatchGenerator:
    """Test cases for Angular patch generator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.generator = AngularPatchGenerator()

    def test_generator_initialization(self):
        """Test patch generator initializes correctly."""
        assert hasattr(self.generator, "template_dir")
        assert hasattr(self.generator, "angular_template_dir")

    def test_fix_no_provider(self):
        """Test no provider error fix generation."""
        error_data = {"message": "No provider for UserService"}

        analysis = {"root_cause": "angular_no_provider", "category": "angular"}

        source_code = "class UserComponent {}"

        patch = self.generator._fix_no_provider(error_data, analysis, source_code)

        assert patch is not None
        assert patch["type"] == "suggestion"
        assert "userservice" in patch["description"].lower()
        assert len(patch["fix_commands"]) > 0
        assert "provider" in patch["fix_code"].lower()

    def test_fix_ngrx_store_not_provided(self):
        """Test NgRx store not provided fix generation."""
        error_data = {"message": "Store has not been provided"}

        analysis = {"root_cause": "ngrx_store_not_provided", "category": "angular"}

        source_code = "@NgModule({}) export class AppModule {}"

        patch = self.generator._fix_ngrx_store_not_provided(
            error_data, analysis, source_code
        )

        assert patch is not None
        assert patch["type"] == "suggestion"
        assert "storemodule" in patch["description"].lower()
        assert "StoreModule.forRoot" in patch["fix_code"]

    def test_fix_template_property_undefined(self):
        """Test template property undefined fix generation."""
        error_data = {"message": "Cannot read property 'name' of undefined"}

        analysis = {
            "root_cause": "angular_template_property_undefined",
            "category": "angular",
        }

        source_code = "<div>{{ user.name }}</div>"

        patch = self.generator._fix_template_property_undefined(
            error_data, analysis, source_code
        )

        assert patch is not None
        assert patch["type"] == "suggestion"
        assert "safe navigation" in patch["description"].lower()
        assert "?." in patch["fix_code"]

    def test_fix_invalid_property_binding(self):
        """Test invalid property binding fix generation."""
        error_data = {
            "message": "Can't bind to 'customProperty' since it isn't a known property"
        }

        analysis = {
            "root_cause": "angular_invalid_property_binding",
            "category": "angular",
        }

        source_code = "<app-component [customProperty]='value'></app-component>"

        patch = self.generator._fix_invalid_property_binding(
            error_data, analysis, source_code
        )

        assert patch is not None
        assert patch["type"] == "suggestion"
        assert "customproperty" in patch["description"].lower()
        assert "@Input()" in patch["fix_code"]
        assert "customProperty" in patch["fix_code"]


class TestAngularIntegration:
    """Integration tests for Angular plugin."""

    def setup_method(self):
        """Set up test fixtures."""
        self.plugin = AngularLanguagePlugin()

    def test_end_to_end_dependency_injection_workflow(self):
        """Test complete workflow for dependency injection error."""
        # Simulate Angular DI error
        error_data = {
            "error_type": "Error",
            "message": "No provider for HttpClient!",
            "stack_trace": [
                "StaticInjectorError(AppModule)[UserService -> HttpClient]:",
                "StaticInjectorError(Platform: core)[UserService -> HttpClient]:",
                "NullInjectorError: No provider for HttpClient!",
            ],
            "framework": "angular",
            "file_path": "/app/user.service.ts",
            "line_number": 15,
        }

        # Test plugin can handle this error
        assert self.plugin.can_handle(error_data) is True

        # Test error analysis
        analysis = self.plugin.analyze_error(error_data)
        assert analysis["category"] == "angular"
        assert analysis["subcategory"] == "dependency_injection"
        assert analysis["confidence"] == "high"

        # Test fix generation
        source_code = """
        @Injectable()
        export class UserService {
            constructor(private http: HttpClient) {}
        }
        """

        fix = self.plugin.generate_fix(error_data, analysis, source_code)
        assert fix is not None
        assert (
            "HttpClient" in fix["description"] or
            "provider" in fix["description"].lower()
        )

    def test_end_to_end_ngrx_workflow(self):
        """Test complete workflow for NgRx error."""
        # Simulate NgRx error
        error_data = {
            "error_type": "Error",
            "message": "Action must have a type",
            "stack_trace": [
                "at createAction (@ngrx/store:123:45)",
                "at UserActions.loadUsers (user.actions.ts:10:20)",
            ],
            "framework": "angular",
            "file_path": "/app/store/user.actions.ts",
            "line_number": 10,
        }

        # Test plugin can handle this error
        assert self.plugin.can_handle(error_data) is True

        # Test error analysis
        analysis = self.plugin.analyze_error(error_data)
        assert analysis["category"] == "angular"
        assert analysis["subcategory"] == "ngrx"

        # Test fix generation
        source_code = """
        export const loadUsers = {
            payload: { page: 1 }
        };
        """

        fix = self.plugin.generate_fix(error_data, analysis, source_code)
        assert fix is not None
        assert "createAction" in fix["fix_code"] or "type" in fix["description"].lower()

    def test_error_handling_edge_cases(self):
        """Test error handling for edge cases."""
        # Test with malformed error data
        malformed_error = {"message": None, "stack_trace": [], "framework": "angular"}

        analysis = self.plugin.analyze_error(malformed_error)
        assert analysis["category"] == "angular"
        assert "error" in analysis  # Should handle gracefully

        # Test with empty error data
        empty_error = {}

        assert self.plugin.can_handle(empty_error) is False

        # Test fix generation with missing analysis
        incomplete_analysis = {"category": "angular"}
        fix = self.plugin.generate_fix(malformed_error, incomplete_analysis, "")
        # Should not crash, may return None
        assert fix is None or isinstance(fix, dict)


if __name__ == "__main__":
    pytest.main([__file__])
