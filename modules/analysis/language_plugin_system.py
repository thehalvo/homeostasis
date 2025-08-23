"""
Language Plugin System

This module provides a pluggable architecture for extending Homeostasis with support for
different programming languages. It defines interfaces and base classes for language-specific
plugins, as well as a registry for managing them.
"""
import abc
import importlib
import inspect
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Type, Callable, Union, Set, Tuple

logger = logging.getLogger(__name__)

# Directory for language plugins
PLUGINS_DIR = Path(__file__).parent / "plugins"
PLUGINS_DIR.mkdir(exist_ok=True)


class LanguagePluginError(Exception):
    """Exception raised for errors in the language plugin system."""
    pass


class LanguagePlugin(abc.ABC):
    """
    Abstract base class for language plugins.
    
    Each language plugin must implement the required methods to provide language-specific
    functionality for error analysis, code generation, and more.
    """
    
    @abc.abstractmethod
    def get_language_id(self) -> str:
        """
        Get the unique identifier for this language.
        
        Returns:
            Language identifier (lowercase)
        """
        pass
    
    @abc.abstractmethod
    def get_language_name(self) -> str:
        """
        Get the human-readable name of the language.
        
        Returns:
            Language name
        """
        pass
    
    @abc.abstractmethod
    def get_language_version(self) -> str:
        """
        Get the version of the language supported by this plugin.
        
        Returns:
            Language version (e.g., "3.9+" for Python)
        """
        pass
    
    @abc.abstractmethod
    def analyze_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a language-specific error.
        
        Args:
            error_data: Error data in the language-specific format
            
        Returns:
            Analysis results
        """
        pass
    
    @abc.abstractmethod
    def normalize_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize error data to the standard Homeostasis format.
        
        Args:
            error_data: Error data in the language-specific format
            
        Returns:
            Error data in the standard format
        """
        pass
    
    @abc.abstractmethod
    def denormalize_error(self, standard_error: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert standard format error data back to the language-specific format.
        
        Args:
            standard_error: Error data in the standard format
            
        Returns:
            Error data in the language-specific format
        """
        pass
    
    @abc.abstractmethod
    def generate_fix(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a fix for an error based on the analysis.
        
        Args:
            analysis: Error analysis
            context: Additional context for fix generation
            
        Returns:
            Generated fix data
        """
        pass
    
    @abc.abstractmethod
    def get_supported_frameworks(self) -> List[str]:
        """
        Get the list of frameworks supported by this language plugin.
        
        Returns:
            List of supported framework identifiers
        """
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get plugin metadata.
        
        Returns:
            Plugin metadata dictionary
        """
        return {
            "language_id": self.get_language_id(),
            "language_name": self.get_language_name(),
            "language_version": self.get_language_version(),
            "supported_frameworks": self.get_supported_frameworks(),
            "plugin_version": getattr(self, "VERSION", "0.1.0"),
            "plugin_author": getattr(self, "AUTHOR", "Unknown"),
            "plugin_description": getattr(self.__class__, "__doc__", "").strip()
        }
    
    def get_capabilities(self) -> Set[str]:
        """
        Get the capabilities of this plugin.
        
        Returns:
            Set of capability identifiers
        """
        capabilities = set()
        
        # Check for required capabilities
        capabilities.add("analyze_error")
        capabilities.add("normalize_error")
        capabilities.add("denormalize_error")
        capabilities.add("generate_fix")
        
        # Check for optional capabilities
        for capability in ["parse_code", "extract_context", "validate_fix", 
                          "apply_fix", "test_fix", "rollback_fix"]:
            if hasattr(self, capability) and callable(getattr(self, capability)):
                capabilities.add(capability)
        
        return capabilities


class LanguagePluginRegistry:
    """
    Registry for managing language plugins.
    
    This class maintains a registry of available language plugins and provides
    methods for loading, registering, and accessing them.
    """
    
    def __init__(self):
        """Initialize the language plugin registry."""
        self.plugins = {}
        self.plugin_classes = {}
    
    def register_plugin(self, plugin: LanguagePlugin):
        """
        Register a language plugin.
        
        Args:
            plugin: Language plugin instance
            
        Raises:
            LanguagePluginError: If a plugin for the same language is already registered
        """
        language_id = plugin.get_language_id().lower()
        
        if language_id in self.plugins:
            raise LanguagePluginError(f"A plugin for language '{language_id}' is already registered")
        
        self.plugins[language_id] = plugin
        self.plugin_classes[language_id] = plugin.__class__
        
        logger.info(f"Registered plugin for language: {plugin.get_language_name()} ({language_id})")
    
    def register_plugin_class(self, plugin_class: Type[LanguagePlugin]):
        """
        Register a language plugin class.
        
        Args:
            plugin_class: Language plugin class
            
        Raises:
            LanguagePluginError: If the class is not a valid language plugin class
        """
        if not issubclass(plugin_class, LanguagePlugin):
            raise LanguagePluginError(f"Class {plugin_class.__name__} is not a subclass of LanguagePlugin")
        
        # Create an instance
        plugin = plugin_class()
        self.register_plugin(plugin)
    
    def unregister_plugin(self, language_id: str):
        """
        Unregister a language plugin.
        
        Args:
            language_id: Language identifier
            
        Raises:
            LanguagePluginError: If no plugin is registered for the language
        """
        language_id = language_id.lower()
        
        if language_id not in self.plugins:
            raise LanguagePluginError(f"No plugin registered for language '{language_id}'")
        
        del self.plugins[language_id]
        del self.plugin_classes[language_id]
        
        logger.info(f"Unregistered plugin for language: {language_id}")
    
    def get_plugin(self, language_id: str) -> Optional[LanguagePlugin]:
        """
        Get a language plugin by ID.
        
        Args:
            language_id: Language identifier
            
        Returns:
            Language plugin instance or None if not found
        """
        return self.plugins.get(language_id.lower())
    
    def has_plugin(self, language_id: str) -> bool:
        """
        Check if a plugin is registered for a language.
        
        Args:
            language_id: Language identifier
            
        Returns:
            True if a plugin is registered, False otherwise
        """
        return language_id.lower() in self.plugins
    
    def get_registered_languages(self) -> List[str]:
        """
        Get the list of registered languages.
        
        Returns:
            List of language identifiers
        """
        return list(self.plugins.keys())
    
    def get_all_plugins(self) -> Dict[str, LanguagePlugin]:
        """
        Get all registered plugins.
        
        Returns:
            Dictionary mapping language IDs to plugin instances
        """
        return self.plugins.copy()
    
    def get_plugin_metadata(self, language_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a language plugin.
        
        Args:
            language_id: Language identifier
            
        Returns:
            Plugin metadata or None if not found
        """
        plugin = self.get_plugin(language_id)
        
        if plugin:
            return plugin.get_metadata()
        
        return None
    
    def load_plugins_from_directory(self, directory: Optional[Union[str, Path]] = None) -> int:
        """
        Load language plugins from a directory.
        
        Args:
            directory: Directory path (defaults to the plugins directory)
            
        Returns:
            Number of plugins loaded
        """
        if directory is None:
            directory = PLUGINS_DIR
        
        directory = Path(directory)
        
        if not directory.exists():
            logger.warning(f"Plugin directory does not exist: {directory}")
            return 0
        
        # Count loaded plugins
        loaded_count = 0
        
        # Look for Python files in the directory
        for file_path in directory.glob("*.py"):
            # Skip __init__.py
            if file_path.name == "__init__.py":
                continue
            
            try:
                # Import the module
                module_name = file_path.stem
                module_path = str(file_path)
                
                logger.debug(f"Loading plugin module: {module_name} from {module_path}")
                
                # Add the parent directory to sys.path
                sys.path.insert(0, str(directory.parent))
                
                try:
                    # Import module and pass the registry
                    spec = importlib.util.spec_from_file_location(module_name, module_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # Look for LanguagePlugin subclasses in the module
                    for name, obj in inspect.getmembers(module):
                        if (inspect.isclass(obj) and issubclass(obj, LanguagePlugin) and 
                            obj != LanguagePlugin):
                            # Register the plugin class
                            try:
                                self.register_plugin_class(obj)
                                loaded_count += 1
                                logger.info(f"Loaded plugin class: {obj.__name__}")
                            except LanguagePluginError as e:
                                logger.warning(f"Failed to register plugin class {obj.__name__}: {e}")
                finally:
                    # Clean up sys.path
                    if str(directory.parent) in sys.path:
                        sys.path.remove(str(directory.parent))
                        
            except Exception as e:
                logger.error(f"Error loading plugin from {file_path}: {e}")
        
        return loaded_count


# Create a global plugin registry
plugin_registry = LanguagePluginRegistry()

# Alias for backward compatibility
LanguagePluginSystem = LanguagePluginRegistry


def register_plugin(plugin: LanguagePlugin):
    """
    Register a language plugin with the global registry.
    
    Args:
        plugin: Language plugin instance
    """
    plugin_registry.register_plugin(plugin)


def get_plugin(language_id: str) -> Optional[LanguagePlugin]:
    """
    Get a language plugin from the global registry.
    
    Args:
        language_id: Language identifier
        
    Returns:
        Language plugin instance or None if not found
    """
    return plugin_registry.get_plugin(language_id)


def get_all_plugins() -> Dict[str, LanguagePlugin]:
    """
    Get all registered plugins from the global registry.
    
    Returns:
        Dictionary mapping language IDs to plugin instances
    """
    return plugin_registry.get_all_plugins()


def get_supported_languages() -> List[str]:
    """
    Get the list of supported languages from the global registry.
    
    Returns:
        List of language identifiers
    """
    return plugin_registry.get_registered_languages()


def load_all_plugins() -> int:
    """
    Load all language plugins from the plugins directory.
    
    Returns:
        Number of plugins loaded
    """
    return plugin_registry.load_plugins_from_directory()


# Base implementation classes for common languages

class PythonLanguagePlugin(LanguagePlugin):
    """
    Python language plugin implementation.
    
    This plugin provides error analysis and fix generation for Python applications.
    """
    
    VERSION = "0.1.0"
    AUTHOR = "Homeostasis Team"
    
    def __init__(self):
        """Initialize the Python language plugin."""
        # Import required modules lazily
        from .analyzer import Analyzer, AnalysisStrategy
        from .language_adapters import PythonErrorAdapter
        
        self.analyzer = Analyzer(strategy=AnalysisStrategy.HYBRID)
        self.adapter = PythonErrorAdapter()
    
    def get_language_id(self) -> str:
        """Get the language identifier."""
        return "python"
    
    def get_language_name(self) -> str:
        """Get the language name."""
        return "Python"
    
    def get_language_version(self) -> str:
        """Get the language version."""
        return "3.6+"
    
    def analyze_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a Python error."""
        return self.analyzer.analyze_error(error_data)
    
    def normalize_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize a Python error to the standard format."""
        return self.adapter.to_standard_format(error_data)
    
    def denormalize_error(self, standard_error: Dict[str, Any]) -> Dict[str, Any]:
        """Convert standard format error data to Python format."""
        return self.adapter.from_standard_format(standard_error)
    
    def generate_fix(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a fix for a Python error."""
        # Import required modules lazily
        from .patch_generation.patcher import PatchGenerator
        
        patch_generator = PatchGenerator()
        patch = patch_generator.generate_patch_from_analysis(analysis)
        
        return patch
    
    def get_supported_frameworks(self) -> List[str]:
        """Get the list of supported frameworks."""
        return ["django", "flask", "fastapi", "sqlalchemy", "base"]


class JavaScriptLanguagePlugin(LanguagePlugin):
    """
    JavaScript language plugin implementation.
    
    This plugin provides error analysis and fix generation for JavaScript applications.
    """
    
    VERSION = "0.1.0"
    AUTHOR = "Homeostasis Team"
    
    def __init__(self):
        """Initialize the JavaScript language plugin."""
        # Import required modules lazily
        from .javascript_analyzer import JavaScriptAnalyzer
        from .language_adapters import JavaScriptErrorAdapter
        
        self.analyzer = JavaScriptAnalyzer()
        self.adapter = JavaScriptErrorAdapter()
    
    def get_language_id(self) -> str:
        """Get the language identifier."""
        return "javascript"
    
    def get_language_name(self) -> str:
        """Get the language name."""
        return "JavaScript"
    
    def get_language_version(self) -> str:
        """Get the language version."""
        return "ES6+"
    
    def analyze_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a JavaScript error."""
        return self.analyzer.analyze_error(error_data)
    
    def normalize_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize a JavaScript error to the standard format."""
        return self.adapter.to_standard_format(error_data)
    
    def denormalize_error(self, standard_error: Dict[str, Any]) -> Dict[str, Any]:
        """Convert standard format error data to JavaScript format."""
        return self.adapter.from_standard_format(standard_error)
    
    def generate_fix(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a fix for a JavaScript error."""
        # Since we don't have a JavaScript-specific patch generator yet,
        # return a placeholder patch
        return {
            "patch_id": f"js_{analysis.get('rule_id', 'unknown')}",
            "patch_type": "suggestion",
            "suggestion": analysis.get("suggestion", "No suggestion available"),
            "root_cause": analysis.get("root_cause", "unknown"),
            "language": "javascript"
        }
    
    def get_supported_frameworks(self) -> List[str]:
        """Get the list of supported frameworks."""
        return ["nodejs", "express", "reactjs", "base"]


# Initialize plugins directory
def init_plugins_directory():
    """Initialize the plugins directory with __init__.py file."""
    plugins_init = PLUGINS_DIR / "__init__.py"
    
    if not plugins_init.exists():
        with open(plugins_init, "w") as f:
            f.write('''"""
Homeostasis Language Plugins

This package contains language-specific plugins for the Homeostasis framework.
"""

# Import utility functions from the parent module
from ..language_plugin_system import (
    LanguagePlugin,
    register_plugin,
    get_plugin,
    get_all_plugins,
    get_supported_languages
)
''')


# Register built-in plugins
def register_builtin_plugins():
    """Register built-in language plugins."""
    try:
        python_plugin = PythonLanguagePlugin()
        plugin_registry.register_plugin(python_plugin)
        
        # JavaScript plugin is registered in javascript_plugin.py
        # js_plugin = JavaScriptLanguagePlugin()
        # plugin_registry.register_plugin(js_plugin)
        
        logger.info("Registered built-in language plugins")
    except Exception as e:
        logger.error(f"Error registering built-in plugins: {e}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize plugins directory
    init_plugins_directory()
    
    # Register built-in plugins
    register_builtin_plugins()
    
    # Load external plugins
    loaded = load_all_plugins()
    logger.info(f"Loaded {loaded} external plugins")
    
    # List all registered plugins
    languages = get_supported_languages()
    logger.info(f"Supported languages: {', '.join(languages)}")
    
    # Test each plugin
    for lang_id in languages:
        plugin = get_plugin(lang_id)
        metadata = plugin.get_metadata()
        
        logger.info(f"\nPlugin: {metadata['language_name']} ({lang_id})")
        logger.info(f"Version: {metadata['plugin_version']}")
        logger.info(f"Author: {metadata['author'] if 'author' in metadata else metadata.get('plugin_author', 'Unknown')}")
        logger.info(f"Description: {metadata['plugin_description']}")
        logger.info(f"Supported frameworks: {', '.join(metadata['supported_frameworks'])}")
        logger.info(f"Capabilities: {', '.join(plugin.get_capabilities())}")
        
        # Test with a sample error
        if lang_id == "python":
            sample_error = {
                "exception_type": "KeyError",
                "message": "'user_id'",
                "traceback": [
                    "Traceback (most recent call last):",
                    "  File \"app.py\", line 42, in get_user",
                    "    user_id = data['user_id']",
                    "KeyError: 'user_id'"
                ]
            }
            
            # Analyze the error
            analysis = plugin.analyze_error(sample_error)
            logger.info(f"Analysis root cause: {analysis.get('root_cause')}")
            
            # Normalize the error
            normalized = plugin.normalize_error(sample_error)
            logger.info(f"Normalized error type: {normalized.get('error_type')}")
            
            # Generate a fix
            fix = plugin.generate_fix(analysis, {})
            logger.info(f"Generated fix: {fix.get('suggestion') if 'suggestion' in fix else fix.get('patch_type', 'unknown')}")
            
        elif lang_id == "javascript":
            sample_error = {
                "name": "TypeError",
                "message": "Cannot read property 'id' of undefined",
                "stack": "TypeError: Cannot read property 'id' of undefined\n    at getUserId (/app/src/utils.js:45:20)"
            }
            
            # Analyze the error
            analysis = plugin.analyze_error(sample_error)
            logger.info(f"Analysis root cause: {analysis.get('root_cause')}")
            
            # Normalize the error
            normalized = plugin.normalize_error(sample_error)
            logger.info(f"Normalized error type: {normalized.get('error_type')}")
            
            # Generate a fix
            fix = plugin.generate_fix(analysis, {})
            logger.info(f"Generated fix: {fix.get('suggestion') if 'suggestion' in fix else fix.get('patch_type', 'unknown')}")