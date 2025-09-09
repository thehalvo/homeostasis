# Analysis module package
__version__ = "0.3.0"

from .analyzer import AnalysisStrategy, Analyzer, analyze_error_from_log
from .cross_language_orchestrator import (CrossLanguageOrchestrator,
                                          analyze_multi_language_error)
from .fastapi_dependency_analyzer import FastAPIDependencyAnalyzer
from .javascript_analyzer import JavaScriptAnalyzer, analyze_javascript_error
from .language_adapters import (ErrorAdapterFactory,
                                convert_from_standard_format,
                                convert_to_standard_format)
from .language_plugin_system import (LanguagePlugin, get_all_plugins,
                                     get_plugin, get_supported_languages,
                                     init_plugins_directory, load_all_plugins,
                                     plugin_registry, register_builtin_plugins,
                                     register_plugin)
from .ml_analyzer import HybridAnalyzer, MLAnalyzer
from .rule_based import RuleBasedAnalyzer

__all__ = [
    "Analyzer",
    "AnalysisStrategy",
    "analyze_error_from_log",
    "RuleBasedAnalyzer",
    "MLAnalyzer",
    "HybridAnalyzer",
    "FastAPIDependencyAnalyzer",
    "JavaScriptAnalyzer",
    "analyze_javascript_error",
    "ErrorAdapterFactory",
    "convert_to_standard_format",
    "convert_from_standard_format",
    "LanguagePlugin",
    "register_plugin",
    "get_plugin",
    "get_all_plugins",
    "get_supported_languages",
    "load_all_plugins",
    "register_builtin_plugins",
    "init_plugins_directory",
    "plugin_registry",
    "CrossLanguageOrchestrator",
    "analyze_multi_language_error",
]

# Initialize the plugin system
init_plugins_directory()
register_builtin_plugins()
load_all_plugins()
