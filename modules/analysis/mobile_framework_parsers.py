"""
Mobile Framework-Aware Parser Factory

This module provides enhanced parsing capabilities that can detect and use
framework-specific parsers for mobile and cross-platform development.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path

from .comprehensive_error_detector import ErrorContext, LanguageType, ErrorCategory
from .language_parsers import (
    LanguageSpecificParser, create_language_parser,
    DartParser, ReactNativeParser, XamarinParser, UnityParser
)

# Set up logger
logger = logging.getLogger(__name__)

# Import web framework parsers
try:
    from .web_framework_parsers import (
        ReactParser, VueParser, AngularParser, SvelteParser, 
        NextJSParser, EmberParser, WebComponentsParser,
        create_web_framework_parser
    )
    WEB_FRAMEWORKS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Web framework parsers not available: {e}")
    WEB_FRAMEWORKS_AVAILABLE = False

# Import additional mobile framework plugins
try:
    from .plugins.java_android_plugin import AndroidJavaLanguagePlugin
    from .plugins.capacitor_cordova_plugin import CapacitorCordovaLanguagePlugin
    ADDITIONAL_MOBILE_FRAMEWORKS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Additional mobile framework plugins not available: {e}")
    ADDITIONAL_MOBILE_FRAMEWORKS_AVAILABLE = False

# Import styling and build system plugins
try:
    from .plugins.css_plugin import CSSLanguagePlugin
    from .plugins.build_analyzer import BuildFileAnalyzer
    STYLING_BUILD_FRAMEWORKS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Styling and build framework plugins not available: {e}")
    STYLING_BUILD_FRAMEWORKS_AVAILABLE = False

# Import core language plugins
try:
    from .plugins.cpp_plugin import CPPLanguagePlugin
    from .plugins.objc_plugin import ObjCLanguagePlugin
    CORE_LANGUAGE_FRAMEWORKS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Core language framework plugins not available: {e}")
    CORE_LANGUAGE_FRAMEWORKS_AVAILABLE = False


class FrameworkType:
    """Framework detection and identification."""
    
    # Mobile frameworks
    FLUTTER = "flutter"
    REACT_NATIVE = "react_native"
    XAMARIN = "xamarin"
    UNITY = "unity"
    JAVA_ANDROID = "java_android"
    CAPACITOR_CORDOVA = "capacitor_cordova"
    
    # Web frameworks
    REACT = "react"
    VUE = "vue"
    ANGULAR = "angular"
    SVELTE = "svelte"
    NEXTJS = "nextjs"
    EMBER = "ember"
    WEBCOMPONENTS = "webcomponents"
    
    # Styling & Build frameworks
    CSS = "css"
    BUILD_SYSTEMS = "build_systems"
    
    # Core Languages
    CPP = "cpp"
    OBJC = "objc"
    
    # Unknown/Generic
    UNKNOWN = "unknown"


class FrameworkDetector:
    """Detects frameworks from error context and project structure."""
    
    def __init__(self):
        """Initialize framework detector."""
        
        # Framework detection patterns
        self.framework_patterns = {
            # Mobile frameworks
            FrameworkType.FLUTTER: [
                "flutter", "dart", "widget", "renderflex", "fluttererror",
                "pubspec.yaml", "lib/main.dart"
            ],
            FrameworkType.REACT_NATIVE: [
                "react native", "metro bundler", "red box", "yellow box", 
                "bridge module", "rn", "package.json", "metro.config.js"
            ],
            FrameworkType.XAMARIN: [
                "xamarin", "mono-rt", "xa0", "xi0", "foundation.monotouch",
                "xamarin.forms", "xamarin.android", "xamarin.ios"
            ],
            FrameworkType.UNITY: [
                "unity", "unityengine", "gameobject", "transform", "monobehaviour",
                "assets/", "library/", "projectsettings/"
            ],
            FrameworkType.JAVA_ANDROID: [
                "android", "androidx", "android.support", "activity", "fragment",
                "service", "broadcastreceiver", "androidmanifest", "build.gradle"
            ],
            FrameworkType.CAPACITOR_CORDOVA: [
                "capacitor", "cordova", "ionic", "@capacitor", "@ionic", "phonegap",
                "hybrid app", "webview", "deviceready", "plugin not found"
            ],
            
            # Web frameworks
            FrameworkType.REACT: [
                "react", "jsx", "hook", "usestate", "useeffect", "usecallback",
                "usememo", "react-dom", "createelement", "invalid hook call"
            ],
            FrameworkType.VUE: [
                "vue", "[vue warn]", "vuex", "vue-router", "v-model", "v-for",
                "composition api", "reactive", "ref()", "computed"
            ],
            FrameworkType.ANGULAR: [
                "angular", "ng", "@angular", "ngmodule", "component", "directive",
                "injectable", "providers", "dependency injection"
            ],
            FrameworkType.SVELTE: [
                "svelte", "$:", "svelte:component", "beforeupdate", "afterupdate",
                "onmount", "ondestroy", "tick()", "writable", "readable"
            ],
            FrameworkType.NEXTJS: [
                "next.js", "nextjs", "getstaticprops", "getserversideprops",
                "getstaticpaths", "_app.js", "_document.js", "next/router"
            ],
            FrameworkType.EMBER: [
                "ember", "ember.js", "emberobject", "ember-cli", "ember-data",
                "handlebars", "route", "controller", "model"
            ],
            FrameworkType.WEBCOMPONENTS: [
                "custom elements", "shadow dom", "html templates", "customelements",
                "attachshadow", "connectedcallback", "disconnectedcallback"
            ],
            
            # Styling & Build frameworks
            FrameworkType.CSS: [
                "css", "tailwind", "styled-components", "emotion", "css modules",
                "sass", "scss", "less", "stylus", "postcss", "css-in-js",
                "flexbox", "grid", "animation", "transition", "@media"
            ],
            FrameworkType.BUILD_SYSTEMS: [
                "webpack", "rollup", "parcel", "vite", "esbuild", "gulp", "grunt",
                "maven", "gradle", "npm", "yarn", "pnpm", "build error", "compilation failed",
                "module not found", "dependency", "pom.xml", "build.gradle", "package.json"
            ],
            
            # Core Languages
            FrameworkType.CPP: [
                "c++", "cpp", "gcc", "g++", "clang", "clang++", "msvc", "cmake", "makefile",
                "segmentation fault", "undefined reference", "template", "std::", "iostream",
                "compilation terminated", "linker error", "SIGSEGV", "signal 11"
            ],
            FrameworkType.OBJC: [
                "objective-c", "objc", "xcode", "ios", "macos", "uikit", "appkit", "foundation",
                "exc_bad_access", "unrecognized selector", "nib", "storyboard", "arc",
                "cocoapods", "swift interop", "@interface", "@implementation", "nsstring"
            ]
        }
    
    def detect_framework(self, error_context: ErrorContext, project_root: Optional[str] = None) -> str:
        """
        Detect framework from error context and project structure.
        
        Args:
            error_context: Error context to analyze
            project_root: Optional project root directory
            
        Returns:
            Detected framework type
        """
        
        # Check error message for framework-specific patterns
        error_msg = error_context.error_message.lower()
        
        for framework, patterns in self.framework_patterns.items():
            if any(pattern.lower() in error_msg for pattern in patterns):
                logger.debug(f"Detected framework {framework} from error message")
                return framework
        
        # Check file path for framework indicators
        if error_context.file_path:
            file_path = error_context.file_path.lower()
            
            if any(pattern in file_path for pattern in ["assets/", "library/", "projectsettings/"]):
                return FrameworkType.UNITY
            elif "lib/" in file_path and file_path.endswith(".dart"):
                return FrameworkType.FLUTTER
            elif any(pattern in file_path for pattern in ["xamarin", "android", "ios"]):
                return FrameworkType.XAMARIN
        
        # Check project structure if available
        if project_root:
            framework = self._detect_from_project_structure(project_root)
            if framework != FrameworkType.UNKNOWN:
                return framework
        
        # Check service name for hints
        if error_context.service_name:
            service_name = error_context.service_name.lower()
            if "flutter" in service_name or "dart" in service_name:
                return FrameworkType.FLUTTER
            elif "rn" in service_name or "react-native" in service_name:
                return FrameworkType.REACT_NATIVE
            elif "xamarin" in service_name:
                return FrameworkType.XAMARIN
            elif "unity" in service_name:
                return FrameworkType.UNITY
        
        return FrameworkType.UNKNOWN
    
    def _detect_from_project_structure(self, project_root: str) -> str:
        """Detect framework from project file structure."""
        try:
            project_path = Path(project_root)
            
            # Flutter detection
            if (project_path / "pubspec.yaml").exists() or (project_path / "lib" / "main.dart").exists():
                return FrameworkType.FLUTTER
            
            # React Native detection
            if ((project_path / "package.json").exists() and 
                (project_path / "metro.config.js").exists()):
                return FrameworkType.REACT_NATIVE
            
            # Check package.json for React Native and web frameworks
            package_json = project_path / "package.json"
            if package_json.exists():
                try:
                    import json
                    with open(package_json) as f:
                        data = json.load(f)
                        deps = {**data.get("dependencies", {}), **data.get("devDependencies", {})}
                        
                        # Check for specific frameworks in order of priority
                        if "react-native" in deps:
                            return FrameworkType.REACT_NATIVE
                        elif "@capacitor/core" in deps or "@ionic/angular" in deps or "@ionic/react" in deps:
                            return FrameworkType.CAPACITOR_CORDOVA
                        elif "cordova" in deps or "phonegap" in deps:
                            return FrameworkType.CAPACITOR_CORDOVA
                        elif "next" in deps or "next.js" in deps:
                            return FrameworkType.NEXTJS
                        elif "vue" in deps or "@vue/core" in deps:
                            return FrameworkType.VUE
                        elif "@angular/core" in deps:
                            return FrameworkType.ANGULAR
                        elif "svelte" in deps:
                            return FrameworkType.SVELTE
                        elif "ember-source" in deps or "ember-cli" in deps:
                            return FrameworkType.EMBER
                        elif "react" in deps:
                            return FrameworkType.REACT
                        elif any(css_dep in deps for css_dep in ["tailwindcss", "styled-components", "@emotion/react", "sass", "less", "stylus"]):
                            return FrameworkType.CSS
                        elif any(build_dep in deps for build_dep in ["webpack", "rollup", "vite", "parcel", "esbuild", "gulp", "grunt"]):
                            return FrameworkType.BUILD_SYSTEMS
                        
                        # Check scripts for framework indicators
                        scripts = data.get("scripts", {})
                        if any("cap " in script or "capacitor" in script for script in scripts.values()):
                            return FrameworkType.CAPACITOR_CORDOVA
                        elif any("cordova " in script for script in scripts.values()):
                            return FrameworkType.CAPACITOR_CORDOVA
                        elif any("ionic " in script for script in scripts.values()):
                            return FrameworkType.CAPACITOR_CORDOVA
                        elif any("next" in script for script in scripts.values()):
                            return FrameworkType.NEXTJS
                        elif any("vue-cli-service" in script for script in scripts.values()):
                            return FrameworkType.VUE
                        elif any("ng " in script for script in scripts.values()):
                            return FrameworkType.ANGULAR
                        elif any("svelte" in script for script in scripts.values()):
                            return FrameworkType.SVELTE
                        elif any("ember " in script for script in scripts.values()):
                            return FrameworkType.EMBER
                        elif any("tailwind" in script or "postcss" in script or "sass" in script for script in scripts.values()):
                            return FrameworkType.CSS
                        elif any("webpack" in script or "rollup" in script or "vite" in script or "build" in script for script in scripts.values()):
                            return FrameworkType.BUILD_SYSTEMS
                except:
                    pass
            
            # Additional mobile framework detection
            if ((project_path / "AndroidManifest.xml").exists() or 
                (project_path / "app" / "src" / "main" / "AndroidManifest.xml").exists() or
                (project_path / "build.gradle").exists()):
                return FrameworkType.JAVA_ANDROID
            
            # Capacitor/Cordova detection
            if ((project_path / "capacitor.config.ts").exists() or 
                (project_path / "capacitor.config.js").exists() or
                (project_path / "config.xml").exists() or
                (project_path / "ionic.config.json").exists()):
                return FrameworkType.CAPACITOR_CORDOVA
            
            # Web framework-specific file detection
            if (project_path / "next.config.js").exists() or (project_path / "pages").exists():
                return FrameworkType.NEXTJS
            elif (project_path / "vue.config.js").exists() or (project_path / "src" / "main.js").exists():
                return FrameworkType.VUE
            elif (project_path / "angular.json").exists() or (project_path / "src" / "app").exists():
                return FrameworkType.ANGULAR
            elif (project_path / "svelte.config.js").exists() or (project_path / "src" / "App.svelte").exists():
                return FrameworkType.SVELTE
            elif (project_path / ".ember-cli").exists() or (project_path / "ember-cli-build.js").exists():
                return FrameworkType.EMBER
            
            # CSS framework detection
            if ((project_path / "tailwind.config.js").exists() or 
                (project_path / "postcss.config.js").exists() or
                list(project_path.glob("**/*.scss")) or
                list(project_path.glob("**/*.sass")) or
                list(project_path.glob("**/*.less"))):
                return FrameworkType.CSS
            
            # Build system detection
            if ((project_path / "webpack.config.js").exists() or
                (project_path / "rollup.config.js").exists() or
                (project_path / "vite.config.js").exists() or
                (project_path / "vite.config.ts").exists() or
                (project_path / "gulpfile.js").exists() or
                (project_path / "Gruntfile.js").exists()):
                return FrameworkType.BUILD_SYSTEMS
            
            # Unity detection
            if ((project_path / "Assets").exists() and 
                (project_path / "ProjectSettings").exists()):
                return FrameworkType.UNITY
            
            # Xamarin detection
            xamarin_files = [
                "*.sln", "*.csproj", "*.vbproj", "*.fsproj"
            ]
            for pattern in xamarin_files:
                if list(project_path.glob(pattern)):
                    # Check if it's Xamarin specifically
                    for file in project_path.glob("*.csproj"):
                        try:
                            content = file.read_text()
                            if "Xamarin" in content:
                                return FrameworkType.XAMARIN
                        except:
                            pass
            
        except Exception as e:
            logger.debug(f"Error detecting framework from project structure: {e}")
        
        return FrameworkType.UNKNOWN


# Wrapper parsers for plugin-based frameworks
class JavaAndroidParser(LanguageSpecificParser):
    """Java Android parser wrapper for plugin integration."""
    
    def __init__(self):
        super().__init__(LanguageType.JAVA)
        if ADDITIONAL_MOBILE_FRAMEWORKS_AVAILABLE:
            self.plugin = AndroidJavaLanguagePlugin()
        else:
            self.plugin = None
    
    def parse_syntax_error(self, error_message: str, source_code: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if self.plugin and hasattr(self.plugin, 'analyze_error'):
            try:
                error_data = {"message": error_message, "source_code": source_code}
                analysis = self.plugin.analyze_error(error_data)
                return {
                    "error_type": "android_java_syntax",
                    "category": ErrorCategory.SYNTAX,
                    "analysis": analysis
                }
            except Exception as e:
                logger.debug(f"Java Android plugin error: {e}")
        return None
    
    def parse_compilation_error(self, error_message: str, source_code: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if self.plugin and hasattr(self.plugin, 'analyze_error'):
            try:
                error_data = {"message": error_message, "source_code": source_code}
                analysis = self.plugin.analyze_error(error_data)
                return {
                    "error_type": "android_java_compilation",
                    "category": ErrorCategory.COMPILATION,
                    "analysis": analysis
                }
            except Exception as e:
                logger.debug(f"Java Android plugin error: {e}")
        return None
    
    def detect_runtime_issues(self, error_context: ErrorContext) -> List[Dict[str, Any]]:
        if self.plugin and hasattr(self.plugin, 'analyze_error'):
            try:
                error_data = {
                    "message": error_context.error_message,
                    "error_type": error_context.error_type,
                    "stack_trace": error_context.stack_trace_lines,
                    "source_code": error_context.source_code_snippet
                }
                analysis = self.plugin.analyze_error(error_data)
                return [{
                    "error_type": "android_java_runtime",
                    "category": ErrorCategory.RUNTIME,
                    "framework": "java_android",
                    "analysis": analysis
                }]
            except Exception as e:
                logger.debug(f"Java Android plugin error: {e}")
        return []


class CapacitorCordovaParser(LanguageSpecificParser):
    """Capacitor/Cordova parser wrapper for plugin integration."""
    
    def __init__(self):
        super().__init__(LanguageType.JAVASCRIPT)
        if ADDITIONAL_MOBILE_FRAMEWORKS_AVAILABLE:
            self.plugin = CapacitorCordovaLanguagePlugin()
        else:
            self.plugin = None
    
    def parse_syntax_error(self, error_message: str, source_code: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if self.plugin and hasattr(self.plugin, 'analyze_error'):
            try:
                error_data = {"message": error_message, "source_code": source_code}
                analysis = self.plugin.analyze_error(error_data)
                return {
                    "error_type": "capacitor_cordova_syntax",
                    "category": ErrorCategory.SYNTAX,
                    "analysis": analysis
                }
            except Exception as e:
                logger.debug(f"Capacitor/Cordova plugin error: {e}")
        return None
    
    def parse_compilation_error(self, error_message: str, source_code: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if self.plugin and hasattr(self.plugin, 'analyze_error'):
            try:
                error_data = {"message": error_message, "source_code": source_code}
                analysis = self.plugin.analyze_error(error_data)
                return {
                    "error_type": "capacitor_cordova_compilation",
                    "category": ErrorCategory.COMPILATION,
                    "analysis": analysis
                }
            except Exception as e:
                logger.debug(f"Capacitor/Cordova plugin error: {e}")
        return None
    
    def detect_runtime_issues(self, error_context: ErrorContext) -> List[Dict[str, Any]]:
        if self.plugin and hasattr(self.plugin, 'analyze_error'):
            try:
                error_data = {
                    "message": error_context.error_message,
                    "error_type": error_context.error_type,
                    "stack_trace": error_context.stack_trace_lines,
                    "source_code": error_context.source_code_snippet
                }
                analysis = self.plugin.analyze_error(error_data)
                return [{
                    "error_type": "capacitor_cordova_runtime",
                    "category": ErrorCategory.RUNTIME,
                    "framework": "capacitor_cordova",
                    "analysis": analysis
                }]
            except Exception as e:
                logger.debug(f"Capacitor/Cordova plugin error: {e}")
        return []


class CSSParser(LanguageSpecificParser):
    """CSS framework parser wrapper for plugin integration."""
    
    def __init__(self):
        super().__init__(LanguageType.UNKNOWN)  # CSS not defined in LanguageType enum
        if STYLING_BUILD_FRAMEWORKS_AVAILABLE:
            self.plugin = CSSLanguagePlugin()
        else:
            self.plugin = None
    
    def parse_syntax_error(self, error_message: str, source_code: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if self.plugin and hasattr(self.plugin, 'analyze_error'):
            try:
                error_data = {"message": error_message, "source_code": source_code}
                analysis = self.plugin.analyze_error(error_data)
                return {
                    "error_type": "css_syntax",
                    "category": ErrorCategory.SYNTAX,
                    "analysis": analysis
                }
            except Exception as e:
                logger.debug(f"CSS plugin error: {e}")
        return None
    
    def parse_compilation_error(self, error_message: str, source_code: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if self.plugin and hasattr(self.plugin, 'analyze_error'):
            try:
                error_data = {"message": error_message, "source_code": source_code}
                analysis = self.plugin.analyze_error(error_data)
                return {
                    "error_type": "css_compilation",
                    "category": ErrorCategory.COMPILATION,
                    "analysis": analysis
                }
            except Exception as e:
                logger.debug(f"CSS plugin error: {e}")
        return None
    
    def detect_runtime_issues(self, error_context: ErrorContext) -> List[Dict[str, Any]]:
        if self.plugin and hasattr(self.plugin, 'analyze_error'):
            try:
                error_data = {
                    "message": error_context.error_message,
                    "error_type": error_context.error_type,
                    "stack_trace": error_context.stack_trace_lines,
                    "source_code": error_context.source_code_snippet
                }
                analysis = self.plugin.analyze_error(error_data)
                return [{
                    "error_type": "css_runtime",
                    "category": ErrorCategory.RUNTIME,
                    "framework": "css",
                    "analysis": analysis
                }]
            except Exception as e:
                logger.debug(f"CSS plugin error: {e}")
        return []


class CPPParser(LanguageSpecificParser):
    """C/C++ parser wrapper for plugin integration."""
    
    def __init__(self):
        super().__init__(LanguageType.CPP)
        if CORE_LANGUAGE_FRAMEWORKS_AVAILABLE:
            self.plugin = CPPLanguagePlugin()
        else:
            self.plugin = None
    
    def parse_syntax_error(self, error_message: str, source_code: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if self.plugin and hasattr(self.plugin, 'analyze_error'):
            try:
                error_data = {"message": error_message, "source_code": source_code}
                analysis = self.plugin.analyze_error(error_data)
                return {
                    "error_type": "cpp_syntax",
                    "category": ErrorCategory.SYNTAX,
                    "analysis": analysis
                }
            except Exception as e:
                logger.debug(f"C/C++ plugin error: {e}")
        return None
    
    def parse_compilation_error(self, error_message: str, source_code: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if self.plugin and hasattr(self.plugin, 'analyze_error'):
            try:
                error_data = {"message": error_message, "source_code": source_code}
                analysis = self.plugin.analyze_error(error_data)
                return {
                    "error_type": "cpp_compilation",
                    "category": ErrorCategory.COMPILATION,
                    "analysis": analysis
                }
            except Exception as e:
                logger.debug(f"C/C++ plugin error: {e}")
        return None
    
    def detect_runtime_issues(self, error_context: ErrorContext) -> List[Dict[str, Any]]:
        if self.plugin and hasattr(self.plugin, 'analyze_error'):
            try:
                error_data = {
                    "message": error_context.error_message,
                    "error_type": error_context.error_type,
                    "stack_trace": error_context.stack_trace_lines,
                    "source_code": error_context.source_code_snippet
                }
                analysis = self.plugin.analyze_error(error_data)
                return [{
                    "error_type": "cpp_runtime",
                    "category": ErrorCategory.RUNTIME,
                    "framework": "cpp",
                    "analysis": analysis
                }]
            except Exception as e:
                logger.debug(f"C/C++ plugin error: {e}")
        return []


class ObjCParser(LanguageSpecificParser):
    """Objective-C parser wrapper for plugin integration."""
    
    def __init__(self):
        super().__init__(LanguageType.UNKNOWN)  # OBJC not defined in LanguageType enum
        if CORE_LANGUAGE_FRAMEWORKS_AVAILABLE:
            self.plugin = ObjCLanguagePlugin()
        else:
            self.plugin = None
    
    def parse_syntax_error(self, error_message: str, source_code: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if self.plugin and hasattr(self.plugin, 'analyze_error'):
            try:
                error_data = {"message": error_message, "source_code": source_code}
                analysis = self.plugin.analyze_error(error_data)
                return {
                    "error_type": "objc_syntax",
                    "category": ErrorCategory.SYNTAX,
                    "analysis": analysis
                }
            except Exception as e:
                logger.debug(f"Objective-C plugin error: {e}")
        return None
    
    def parse_compilation_error(self, error_message: str, source_code: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if self.plugin and hasattr(self.plugin, 'analyze_error'):
            try:
                error_data = {"message": error_message, "source_code": source_code}
                analysis = self.plugin.analyze_error(error_data)
                return {
                    "error_type": "objc_compilation",
                    "category": ErrorCategory.COMPILATION,
                    "analysis": analysis
                }
            except Exception as e:
                logger.debug(f"Objective-C plugin error: {e}")
        return None
    
    def detect_runtime_issues(self, error_context: ErrorContext) -> List[Dict[str, Any]]:
        if self.plugin and hasattr(self.plugin, 'analyze_error'):
            try:
                error_data = {
                    "message": error_context.error_message,
                    "error_type": error_context.error_type,
                    "stack_trace": error_context.stack_trace_lines,
                    "source_code": error_context.source_code_snippet
                }
                analysis = self.plugin.analyze_error(error_data)
                return [{
                    "error_type": "objc_runtime",
                    "category": ErrorCategory.RUNTIME,
                    "framework": "objc",
                    "analysis": analysis
                }]
            except Exception as e:
                logger.debug(f"Objective-C plugin error: {e}")
        return []


class BuildSystemsParser(LanguageSpecificParser):
    """Build Systems parser wrapper for build analyzer integration."""
    
    def __init__(self):
        super().__init__(LanguageType.JSON)  # JSON for package.json, pom.xml analysis
        if STYLING_BUILD_FRAMEWORKS_AVAILABLE:
            self.analyzer = BuildFileAnalyzer()
        else:
            self.analyzer = None
    
    def parse_syntax_error(self, error_message: str, source_code: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if self.analyzer:
            try:
                # Analyze build configuration issues
                return {
                    "error_type": "build_syntax",
                    "category": ErrorCategory.SYNTAX,
                    "message": error_message,
                    "analyzer": "build_systems"
                }
            except Exception as e:
                logger.debug(f"Build systems analyzer error: {e}")
        return None
    
    def parse_compilation_error(self, error_message: str, source_code: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if self.analyzer:
            try:
                # Check for build tool specific errors
                build_patterns = {
                    "webpack": "webpack compilation error",
                    "maven": "maven build error",
                    "gradle": "gradle build error",
                    "npm": "npm dependency error",
                    "yarn": "yarn dependency error"
                }
                
                for tool, error_type in build_patterns.items():
                    if tool in error_message.lower():
                        return {
                            "error_type": error_type,
                            "category": ErrorCategory.COMPILATION,
                            "build_tool": tool,
                            "message": error_message
                        }
                
                return {
                    "error_type": "build_compilation",
                    "category": ErrorCategory.COMPILATION,
                    "message": error_message,
                    "analyzer": "build_systems"
                }
            except Exception as e:
                logger.debug(f"Build systems analyzer error: {e}")
        return None
    
    def detect_runtime_issues(self, error_context: ErrorContext) -> List[Dict[str, Any]]:
        if self.analyzer:
            try:
                issues = []
                message = error_context.error_message.lower()
                
                # Detect common build issues
                if "module not found" in message or "dependency" in message:
                    issues.append({
                        "error_type": "dependency_missing",
                        "category": ErrorCategory.DEPENDENCY,
                        "framework": "build_systems",
                        "description": "Missing dependency or module"
                    })
                
                if "build failed" in message or "compilation failed" in message:
                    issues.append({
                        "error_type": "build_failure",
                        "category": ErrorCategory.COMPILATION,
                        "framework": "build_systems",
                        "description": "Build or compilation failure"
                    })
                
                if "version conflict" in message or "incompatible" in message:
                    issues.append({
                        "error_type": "version_conflict",
                        "category": ErrorCategory.DEPENDENCY,
                        "framework": "build_systems",
                        "description": "Dependency version conflict"
                    })
                
                return issues
            except Exception as e:
                logger.debug(f"Build systems analyzer error: {e}")
        return []


class UnifiedFrameworkParserFactory:
    """
    Factory for creating framework-aware parsers for mobile, web, and cross-platform development.
    """
    
    def __init__(self):
        """Initialize the mobile framework parser factory."""
        self.framework_detector = FrameworkDetector()
        
        # Framework-specific parser mappings
        self.framework_parsers = {
            # Mobile frameworks
            FrameworkType.FLUTTER: DartParser,
            FrameworkType.REACT_NATIVE: ReactNativeParser,
            FrameworkType.XAMARIN: XamarinParser,
            FrameworkType.UNITY: UnityParser,
        }
        
        # Add additional mobile framework parsers if available
        if ADDITIONAL_MOBILE_FRAMEWORKS_AVAILABLE:
            self.framework_parsers.update({
                FrameworkType.JAVA_ANDROID: JavaAndroidParser,
                FrameworkType.CAPACITOR_CORDOVA: CapacitorCordovaParser,
            })
        
        # Add web framework parsers if available
        if WEB_FRAMEWORKS_AVAILABLE:
            self.framework_parsers.update({
                FrameworkType.REACT: ReactParser,
                FrameworkType.VUE: VueParser,
                FrameworkType.ANGULAR: AngularParser,
                FrameworkType.SVELTE: SvelteParser,
                FrameworkType.NEXTJS: NextJSParser,
                FrameworkType.EMBER: EmberParser,
                FrameworkType.WEBCOMPONENTS: WebComponentsParser,
            })
        
        # Add styling and build system parsers if available
        if STYLING_BUILD_FRAMEWORKS_AVAILABLE:
            self.framework_parsers.update({
                FrameworkType.CSS: CSSParser,
                FrameworkType.BUILD_SYSTEMS: BuildSystemsParser,
            })
        
        # Add core language parsers if available
        if CORE_LANGUAGE_FRAMEWORKS_AVAILABLE:
            self.framework_parsers.update({
                FrameworkType.CPP: CPPParser,
                FrameworkType.OBJC: ObjCParser,
            })
    
    def create_parser(self, error_context: ErrorContext, 
                     project_root: Optional[str] = None) -> Optional[LanguageSpecificParser]:
        """
        Create a framework-aware parser for the given error context.
        
        Args:
            error_context: Error context to analyze
            project_root: Optional project root directory
            
        Returns:
            Appropriate parser instance or None
        """
        
        # First detect the framework
        framework = self.framework_detector.detect_framework(error_context, project_root)
        
        # Try framework-specific parser first
        if framework in self.framework_parsers:
            try:
                parser_class = self.framework_parsers[framework]
                parser = parser_class()
                logger.info(f"Created {framework} framework parser")
                return parser
            except Exception as e:
                logger.warning(f"Error creating {framework} parser: {e}")
        
        # Fallback to language-based parser
        language_parser = create_language_parser(error_context.language)
        if language_parser:
            logger.info(f"Created {error_context.language.value} language parser")
            return language_parser
        
        logger.warning(f"No suitable parser found for {error_context.language.value} / {framework}")
        return None
    
    def analyze_error_with_framework_context(self, 
                                           error_context: ErrorContext,
                                           project_root: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze error with full framework context.
        
        Args:
            error_context: Error context to analyze
            project_root: Optional project root directory
            
        Returns:
            Comprehensive analysis results
        """
        
        # Detect framework
        framework = self.framework_detector.detect_framework(error_context, project_root)
        
        # Create appropriate parser
        parser = self.create_parser(error_context, project_root)
        
        # Initialize results
        analysis_result = {
            "detected_framework": framework,
            "language": error_context.language.value,
            "parser_type": type(parser).__name__ if parser else None,
            "syntax_analysis": None,
            "compilation_analysis": None,
            "runtime_analysis": None,
            "framework_specific_analysis": None
        }
        
        if not parser:
            analysis_result["error"] = "No suitable parser available"
            return analysis_result
        
        try:
            # Perform different types of analysis
            analysis_result["syntax_analysis"] = parser.parse_syntax_error(
                error_context.error_message,
                error_context.source_code_snippet
            )
            
            analysis_result["compilation_analysis"] = parser.parse_compilation_error(
                error_context.error_message,
                error_context.source_code_snippet
            )
            
            analysis_result["runtime_analysis"] = parser.detect_runtime_issues(error_context)
            
            # Add framework-specific analysis if available
            if hasattr(parser, 'get_framework_specific_analysis'):
                analysis_result["framework_specific_analysis"] = parser.get_framework_specific_analysis(error_context)
            
        except Exception as e:
            analysis_result["error"] = str(e)
            logger.error(f"Error during framework analysis: {e}")
        
        return analysis_result
    
    def get_supported_frameworks(self) -> List[str]:
        """Get list of supported frameworks."""
        return list(self.framework_parsers.keys())
    
    def get_framework_recommendations(self, error_context: ErrorContext) -> List[Dict[str, Any]]:
        """
        Get framework-specific recommendations for fixing errors.
        
        Args:
            error_context: Error context to analyze
            
        Returns:
            List of framework-specific recommendations
        """
        
        framework = self.framework_detector.detect_framework(error_context)
        recommendations = []
        
        if framework == FrameworkType.FLUTTER:
            recommendations.extend([
                {
                    "type": "flutter_specific",
                    "title": "Check Flutter SDK Version",
                    "description": "Ensure your Flutter SDK is up to date",
                    "action": "Run 'flutter doctor' to check your installation"
                },
                {
                    "type": "flutter_specific", 
                    "title": "Verify pubspec.yaml Dependencies",
                    "description": "Check for version conflicts in dependencies",
                    "action": "Run 'flutter pub deps' to analyze dependencies"
                }
            ])
        
        elif framework == FrameworkType.REACT_NATIVE:
            recommendations.extend([
                {
                    "type": "react_native_specific",
                    "title": "Clear Metro Bundler Cache",
                    "description": "Metro bundler cache might be corrupted",
                    "action": "Run 'npx react-native start --reset-cache'"
                },
                {
                    "type": "react_native_specific",
                    "title": "Check Native Dependencies",
                    "description": "Verify native module linking",
                    "action": "Run 'npx react-native doctor' to check setup"
                }
            ])
        
        elif framework == FrameworkType.XAMARIN:
            recommendations.extend([
                {
                    "type": "xamarin_specific",
                    "title": "Check Xamarin Versions",
                    "description": "Ensure compatible Xamarin.Forms and platform versions",
                    "action": "Update packages via NuGet Package Manager"
                },
                {
                    "type": "xamarin_specific",
                    "title": "Clean and Rebuild",
                    "description": "Clean solution and rebuild",
                    "action": "Use 'Clean Solution' then 'Rebuild Solution' in Visual Studio"
                }
            ])
        
        elif framework == FrameworkType.UNITY:
            recommendations.extend([
                {
                    "type": "unity_specific",
                    "title": "Reimport Assets",
                    "description": "Reimport project assets to resolve compilation issues",
                    "action": "Go to Assets > Reimport All"
                },
                {
                    "type": "unity_specific",
                    "title": "Check Unity Version Compatibility",
                    "description": "Verify Unity version matches project requirements",
                    "action": "Check Unity Hub for version compatibility"
                }
            ])
        
        # Web framework recommendations
        elif framework == FrameworkType.REACT:
            recommendations.extend([
                {
                    "type": "react_specific",
                    "title": "Check React Hooks Rules",
                    "description": "Follow the Rules of Hooks for proper React development",
                    "action": "Review React Hooks documentation and eslint-plugin-react-hooks"
                },
                {
                    "type": "react_specific",
                    "title": "Add React Developer Tools",
                    "description": "Use React DevTools for debugging components and state",
                    "action": "Install React Developer Tools browser extension"
                }
            ])
        
        elif framework == FrameworkType.VUE:
            recommendations.extend([
                {
                    "type": "vue_specific",
                    "title": "Check Vue.js Version Compatibility",
                    "description": "Ensure Vue version matches your component syntax",
                    "action": "Verify Vue 2 vs Vue 3 compatibility for your code"
                },
                {
                    "type": "vue_specific",
                    "title": "Use Vue DevTools",
                    "description": "Debug Vue components, Vuex state, and reactivity",
                    "action": "Install Vue.js devtools browser extension"
                }
            ])
        
        elif framework == FrameworkType.ANGULAR:
            recommendations.extend([
                {
                    "type": "angular_specific",
                    "title": "Check Angular CLI Version",
                    "description": "Ensure Angular CLI matches your project version",
                    "action": "Run 'ng version' to check versions and 'ng update' to update"
                },
                {
                    "type": "angular_specific",
                    "title": "Use Angular DevTools",
                    "description": "Debug Angular components and dependency injection",
                    "action": "Install Angular DevTools browser extension"
                }
            ])
        
        elif framework == FrameworkType.SVELTE:
            recommendations.extend([
                {
                    "type": "svelte_specific",
                    "title": "Check Svelte Compiler",
                    "description": "Ensure Svelte compiler is properly configured",
                    "action": "Review svelte.config.js and build tool configuration"
                },
                {
                    "type": "svelte_specific",
                    "title": "Use Svelte DevTools",
                    "description": "Debug Svelte components and reactivity",
                    "action": "Install Svelte DevTools browser extension"
                }
            ])
        
        elif framework == FrameworkType.NEXTJS:
            recommendations.extend([
                {
                    "type": "nextjs_specific",
                    "title": "Check Next.js Configuration",
                    "description": "Review next.config.js for proper setup",
                    "action": "Verify Next.js configuration and routing setup"
                },
                {
                    "type": "nextjs_specific",
                    "title": "Clear Next.js Cache",
                    "description": "Clear build cache if experiencing issues",
                    "action": "Delete .next folder and run 'npm run build' again"
                }
            ])
        
        elif framework == FrameworkType.EMBER:
            recommendations.extend([
                {
                    "type": "ember_specific",
                    "title": "Check Ember CLI Version",
                    "description": "Ensure Ember CLI is up to date",
                    "action": "Run 'ember version' and 'ember update' if needed"
                },
                {
                    "type": "ember_specific",
                    "title": "Use Ember Inspector",
                    "description": "Debug Ember routes, components, and data",
                    "action": "Install Ember Inspector browser extension"
                }
            ])
        
        elif framework == FrameworkType.WEBCOMPONENTS:
            recommendations.extend([
                {
                    "type": "webcomponents_specific",
                    "title": "Check Browser Support",
                    "description": "Verify Web Components support in target browsers",
                    "action": "Consider polyfills for older browsers"
                },
                {
                    "type": "webcomponents_specific",
                    "title": "Validate Custom Element Names",
                    "description": "Ensure custom element names follow specifications",
                    "action": "Use kebab-case with at least one hyphen for custom elements"
                }
            ])
        
        elif framework == FrameworkType.JAVA_ANDROID:
            recommendations.extend([
                {
                    "type": "java_android_specific",
                    "title": "Check Android Manifest",
                    "description": "Verify activity declarations and permissions in AndroidManifest.xml",
                    "action": "Add missing activities and permissions to manifest"
                },
                {
                    "type": "java_android_specific",
                    "title": "Optimize Memory Usage",
                    "description": "Implement proper memory management for Android apps",
                    "action": "Use bitmap recycling, LRU cache, and lifecycle awareness"
                },
                {
                    "type": "java_android_specific",
                    "title": "Handle Activity Lifecycle",
                    "description": "Implement proper activity and fragment lifecycle management",
                    "action": "Check component state before UI operations"
                }
            ])
        
        elif framework == FrameworkType.CAPACITOR_CORDOVA:
            recommendations.extend([
                {
                    "type": "capacitor_cordova_specific",
                    "title": "Install Required Plugins",
                    "description": "Ensure all required Capacitor/Cordova plugins are installed",
                    "action": "Run 'npm install @capacitor/plugin-name' and 'npx cap sync'"
                },
                {
                    "type": "capacitor_cordova_specific",
                    "title": "Configure Permissions",
                    "description": "Set up proper mobile permissions and capabilities",
                    "action": "Configure permissions in capacitor.config.ts and native projects"
                },
                {
                    "type": "capacitor_cordova_specific",
                    "title": "Update Content Security Policy",
                    "description": "Configure CSP for hybrid app requirements",
                    "action": "Allow necessary sources in CSP meta tag"
                },
                {
                    "type": "capacitor_cordova_specific",
                    "title": "Platform Build Configuration",
                    "description": "Check platform-specific build settings",
                    "action": "Verify Android SDK, Xcode, and build tool versions"
                }
            ])
        
        elif framework == FrameworkType.CSS:
            recommendations.extend([
                {
                    "type": "css_specific",
                    "title": "Check CSS Syntax and Selectors",
                    "description": "Validate CSS syntax and selector specificity",
                    "action": "Use CSS linters and validate syntax"
                },
                {
                    "type": "css_specific",
                    "title": "Optimize CSS Performance",
                    "description": "Minimize CSS bundle size and improve loading performance",
                    "action": "Use CSS purging, minification, and critical CSS extraction"
                },
                {
                    "type": "css_specific",
                    "title": "Configure CSS Preprocessors",
                    "description": "Setup SASS, LESS, or PostCSS compilation properly",
                    "action": "Check preprocessor configuration and build pipeline"
                },
                {
                    "type": "css_specific",
                    "title": "Fix CSS-in-JS Issues",
                    "description": "Resolve styled-components or emotion configuration problems",
                    "action": "Check CSS-in-JS library setup and theming"
                }
            ])
        
        elif framework == FrameworkType.BUILD_SYSTEMS:
            recommendations.extend([
                {
                    "type": "build_systems_specific",
                    "title": "Check Build Configuration",
                    "description": "Verify webpack, rollup, vite, or other build tool setup",
                    "action": "Review build configuration files and dependencies"
                },
                {
                    "type": "build_systems_specific",
                    "title": "Resolve Dependency Conflicts",
                    "description": "Fix version conflicts and missing dependencies",
                    "action": "Run dependency resolution and update package versions"
                },
                {
                    "type": "build_systems_specific",
                    "title": "Optimize Build Performance",
                    "description": "Improve build speed and bundle optimization",
                    "action": "Enable caching, code splitting, and tree shaking"
                },
                {
                    "type": "build_systems_specific",
                    "title": "Fix Build Tool Integration",
                    "description": "Ensure proper integration between build tools and frameworks",
                    "action": "Check plugin configurations and build pipeline setup"
                }
            ])
        
        elif framework == FrameworkType.CPP:
            recommendations.extend([
                {
                    "type": "cpp_specific",
                    "title": "Use Memory Safety Tools",
                    "description": "Enable AddressSanitizer and other memory debugging tools",
                    "action": "Compile with -fsanitize=address and run with memory analysis tools"
                },
                {
                    "type": "cpp_specific",
                    "title": "Check Compiler Warnings",
                    "description": "Enable all compiler warnings and treat them as errors",
                    "action": "Use -Wall -Wextra -Werror compiler flags"
                },
                {
                    "type": "cpp_specific",
                    "title": "Use Modern C++ Features",
                    "description": "Adopt smart pointers and RAII for safer memory management",
                    "action": "Migrate to std::unique_ptr, std::shared_ptr, and modern C++ patterns"
                },
                {
                    "type": "cpp_specific",
                    "title": "Static Analysis",
                    "description": "Run static analysis tools to catch potential issues",
                    "action": "Use clang-static-analyzer, cppcheck, or PVS-Studio"
                }
            ])
        
        elif framework == FrameworkType.OBJC:
            recommendations.extend([
                {
                    "type": "objc_specific",
                    "title": "Enable ARC",
                    "description": "Use Automatic Reference Counting for memory management",
                    "action": "Enable ARC in project settings and remove manual retain/release"
                },
                {
                    "type": "objc_specific",
                    "title": "Use iOS Instruments",
                    "description": "Profile app with Instruments for memory leaks and performance",
                    "action": "Run Leaks, Allocations, and Time Profiler instruments"
                },
                {
                    "type": "objc_specific",
                    "title": "Check Thread Safety",
                    "description": "Ensure UI updates happen on main thread",
                    "action": "Use dispatch_async for main queue UI updates"
                },
                {
                    "type": "objc_specific",
                    "title": "Validate Interface Builder",
                    "description": "Check all outlets and actions are properly connected",
                    "action": "Review Storyboard connections and remove orphaned outlets"
                }
            ])
        
        return recommendations


# Convenience functions for integration
def create_mobile_framework_parser(error_context: ErrorContext, 
                                  project_root: Optional[str] = None) -> Optional[LanguageSpecificParser]:
    """
    Convenience function to create a mobile framework-aware parser.
    
    Args:
        error_context: Error context to analyze
        project_root: Optional project root directory
        
    Returns:
        Appropriate parser instance or None
    """
    factory = UnifiedFrameworkParserFactory()
    return factory.create_parser(error_context, project_root)


def analyze_mobile_error(error_context: ErrorContext,
                        project_root: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to analyze mobile/cross-platform errors.
    
    Args:
        error_context: Error context to analyze  
        project_root: Optional project root directory
        
    Returns:
        Comprehensive analysis results
    """
    factory = UnifiedFrameworkParserFactory()
    return factory.analyze_error_with_framework_context(error_context, project_root)


# New unified convenience functions
def create_framework_parser(error_context: ErrorContext, 
                           project_root: Optional[str] = None) -> Optional[LanguageSpecificParser]:
    """
    Convenience function to create a framework-aware parser for any supported framework.
    
    Args:
        error_context: Error context to analyze
        project_root: Optional project root directory
        
    Returns:
        Appropriate parser instance or None
    """
    factory = UnifiedFrameworkParserFactory()
    return factory.create_parser(error_context, project_root)


def analyze_framework_error(error_context: ErrorContext,
                           project_root: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to analyze errors with framework-specific context.
    
    Args:
        error_context: Error context to analyze  
        project_root: Optional project root directory
        
    Returns:
        Comprehensive analysis results
    """
    factory = UnifiedFrameworkParserFactory()
    return factory.analyze_error_with_framework_context(error_context, project_root)


# Maintain backward compatibility
MobileFrameworkParserFactory = UnifiedFrameworkParserFactory


if __name__ == "__main__":
    # Test the mobile framework parser factory
    print("Mobile Framework Parser Factory Test")
    print("===================================")
    
    # Test framework detection
    detector = FrameworkDetector()
    
    # Test Flutter error
    flutter_context = ErrorContext(
        error_message="FlutterError: RenderFlex overflowed by 15 pixels on the right",
        language=LanguageType.DART,
        file_path="lib/main.dart"
    )
    
    flutter_framework = detector.detect_framework(flutter_context)
    print(f"Flutter error detected as: {flutter_framework}")
    
    # Test React Native error
    rn_context = ErrorContext(
        error_message="Element type is invalid: expected a string but received undefined",
        language=LanguageType.JAVASCRIPT,
        service_name="rn-app"
    )
    
    rn_framework = detector.detect_framework(rn_context)
    print(f"React Native error detected as: {rn_framework}")
    
    # Test Xamarin error
    xamarin_context = ErrorContext(
        error_message="XA0001: Java.Lang.RuntimeException: Unable to start activity",
        language=LanguageType.CSHARP,
        file_path="MainActivity.cs"
    )
    
    xamarin_framework = detector.detect_framework(xamarin_context)
    print(f"Xamarin error detected as: {xamarin_framework}")
    
    # Test Unity error
    unity_context = ErrorContext(
        error_message="NullReferenceException: Object reference not set to an instance of an object",
        language=LanguageType.CSHARP,
        file_path="Assets/Scripts/PlayerController.cs"
    )
    
    unity_framework = detector.detect_framework(unity_context)
    print(f"Unity error detected as: {unity_framework}")
    
    # Test Java Android error
    android_context = ErrorContext(
        error_message="ActivityNotFoundException: Unable to find explicit activity class MainActivity",
        language=LanguageType.JAVA,
        file_path="app/src/main/java/com/example/MainActivity.java"
    )
    
    android_framework = detector.detect_framework(android_context)
    print(f"Java Android error detected as: {android_framework}")
    
    # Test Capacitor/Cordova error
    capacitor_context = ErrorContext(
        error_message="Plugin Camera not found, or is not installed",
        language=LanguageType.JAVASCRIPT,
        service_name="capacitor-app"
    )
    
    capacitor_framework = detector.detect_framework(capacitor_context)
    print(f"Capacitor/Cordova error detected as: {capacitor_framework}")
    
    # Test CSS error
    css_context = ErrorContext(
        error_message="SassError: Undefined variable: $primary-color",
        language=LanguageType.UNKNOWN,  # CSS not defined in LanguageType
        file_path="src/styles/main.scss"
    )
    
    css_framework = detector.detect_framework(css_context)
    print(f"CSS error detected as: {css_framework}")
    
    # Test Build Systems error
    build_context = ErrorContext(
        error_message="Module not found: Error: Can't resolve './missing-module'",
        language=LanguageType.JAVASCRIPT,
        file_path="webpack.config.js"
    )
    
    build_framework = detector.detect_framework(build_context)
    print(f"Build Systems error detected as: {build_framework}")
    
    # Test C/C++ error
    cpp_context = ErrorContext(
        error_message="Segmentation fault (core dumped): invalid memory access at line 42",
        language=LanguageType.CPP,
        file_path="src/main.cpp"
    )
    
    cpp_framework = detector.detect_framework(cpp_context)
    print(f"C/C++ error detected as: {cpp_framework}")
    
    # Test Objective-C error
    objc_context = ErrorContext(
        error_message="EXC_BAD_ACCESS: unrecognized selector sent to instance",
        language=LanguageType.UNKNOWN,  # OBJC not defined in LanguageType
        file_path="ViewController.m"
    )
    
    objc_framework = detector.detect_framework(objc_context)
    print(f"Objective-C error detected as: {objc_framework}")
    
    # Test framework parser factory
    factory = MobileFrameworkParserFactory()
    
    print(f"\nSupported frameworks: {factory.get_supported_frameworks()}")
    
    # Test comprehensive analysis
    flutter_analysis = factory.analyze_error_with_framework_context(flutter_context)
    print(f"\nFlutter Analysis:")
    print(f"Framework: {flutter_analysis['detected_framework']}")
    print(f"Parser: {flutter_analysis['parser_type']}")
    print(f"Runtime Issues: {len(flutter_analysis.get('runtime_analysis', []))}")
    
    # Test recommendations
    flutter_recommendations = factory.get_framework_recommendations(flutter_context)
    print(f"\nFlutter Recommendations: {len(flutter_recommendations)} items")
    for rec in flutter_recommendations:
        print(f"- {rec['title']}: {rec['description']}")
    
    # Test Java Android analysis
    if ADDITIONAL_MOBILE_FRAMEWORKS_AVAILABLE:
        android_analysis = factory.analyze_error_with_framework_context(android_context)
        print(f"\nJava Android Analysis:")
        print(f"Framework: {android_analysis['detected_framework']}")
        print(f"Parser: {android_analysis['parser_type']}")
        print(f"Runtime Issues: {len(android_analysis.get('runtime_analysis', []))}")
        
        android_recommendations = factory.get_framework_recommendations(android_context)
        print(f"\nJava Android Recommendations: {len(android_recommendations)} items")
        for rec in android_recommendations[:2]:  # Show first 2
            print(f"- {rec['title']}: {rec['description']}")
        
        # Test Capacitor/Cordova analysis
        capacitor_analysis = factory.analyze_error_with_framework_context(capacitor_context)
        print(f"\nCapacitor/Cordova Analysis:")
        print(f"Framework: {capacitor_analysis['detected_framework']}")
        print(f"Parser: {capacitor_analysis['parser_type']}")
        print(f"Runtime Issues: {len(capacitor_analysis.get('runtime_analysis', []))}")
        
        capacitor_recommendations = factory.get_framework_recommendations(capacitor_context)
        print(f"\nCapacitor/Cordova Recommendations: {len(capacitor_recommendations)} items")
        for rec in capacitor_recommendations[:2]:  # Show first 2
            print(f"- {rec['title']}: {rec['description']}")
    else:
        print("\nAdditional mobile framework plugins not available for testing")
    
    # Test CSS and Build Systems analysis
    if STYLING_BUILD_FRAMEWORKS_AVAILABLE:
        css_analysis = factory.analyze_error_with_framework_context(css_context)
        print(f"\nCSS Analysis:")
        print(f"Framework: {css_analysis['detected_framework']}")
        print(f"Parser: {css_analysis['parser_type']}")
        print(f"Runtime Issues: {len(css_analysis.get('runtime_analysis', []))}")
        
        css_recommendations = factory.get_framework_recommendations(css_context)
        print(f"\nCSS Recommendations: {len(css_recommendations)} items")
        for rec in css_recommendations[:2]:  # Show first 2
            print(f"- {rec['title']}: {rec['description']}")
        
        # Test Build Systems analysis
        build_analysis = factory.analyze_error_with_framework_context(build_context)
        print(f"\nBuild Systems Analysis:")
        print(f"Framework: {build_analysis['detected_framework']}")
        print(f"Parser: {build_analysis['parser_type']}")
        print(f"Runtime Issues: {len(build_analysis.get('runtime_analysis', []))}")
        
        build_recommendations = factory.get_framework_recommendations(build_context)
        print(f"\nBuild Systems Recommendations: {len(build_recommendations)} items")
        for rec in build_recommendations[:2]:  # Show first 2
            print(f"- {rec['title']}: {rec['description']}")
    else:
        print("\nStyling and build framework plugins not available for testing")
    
    # Test C/C++ and Objective-C analysis
    if CORE_LANGUAGE_FRAMEWORKS_AVAILABLE:
        cpp_analysis = factory.analyze_error_with_framework_context(cpp_context)
        print(f"\nC/C++ Analysis:")
        print(f"Framework: {cpp_analysis['detected_framework']}")
        print(f"Parser: {cpp_analysis['parser_type']}")
        print(f"Runtime Issues: {len(cpp_analysis.get('runtime_analysis', []))}")
        
        cpp_recommendations = factory.get_framework_recommendations(cpp_context)
        print(f"\nC/C++ Recommendations: {len(cpp_recommendations)} items")
        for rec in cpp_recommendations[:2]:  # Show first 2
            print(f"- {rec['title']}: {rec['description']}")
        
        # Test Objective-C analysis
        objc_analysis = factory.analyze_error_with_framework_context(objc_context)
        print(f"\nObjective-C Analysis:")
        print(f"Framework: {objc_analysis['detected_framework']}")
        print(f"Parser: {objc_analysis['parser_type']}")
        print(f"Runtime Issues: {len(objc_analysis.get('runtime_analysis', []))}")
        
        objc_recommendations = factory.get_framework_recommendations(objc_context)
        print(f"\nObjective-C Recommendations: {len(objc_recommendations)} items")
        for rec in objc_recommendations[:2]:  # Show first 2
            print(f"- {rec['title']}: {rec['description']}")
    else:
        print("\nCore language framework plugins not available for testing")