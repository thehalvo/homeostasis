"""
Cross-Platform Framework Integration

This module ensures that all cross-platform framework plugins are properly loaded
and integrated with the orchestrator system.
"""

import importlib
import logging
import sys
from pathlib import Path

# Import the plugin registration system
from .language_plugin_system import plugin_registry

logger = logging.getLogger(__name__)


def load_cross_platform_plugins():
    """
    Load all cross-platform framework plugins.

    Returns:
        Number of plugins loaded
    """
    plugins_loaded = 0

    # Define cross-platform plugins to load
    cross_platform_plugins = [
        "react_native_plugin",
        "flutter_plugin",
        "xamarin_plugin",
        "unity_plugin",
        "capacitor_cordova_plugin",
    ]

    # Get the plugins directory
    plugins_dir = Path(__file__).parent / "plugins"

    for plugin_name in cross_platform_plugins:
        try:
            # Import the plugin module
            plugin_path = plugins_dir / f"{plugin_name}.py"

            if not plugin_path.exists():
                logger.warning(f"Plugin file not found: {plugin_path}")
                continue

            # Import the module
            spec = importlib.util.spec_from_file_location(plugin_name, plugin_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[plugin_name] = module
                spec.loader.exec_module(module)

                logger.info(f"Successfully loaded cross-platform plugin: {plugin_name}")
                plugins_loaded += 1
            else:
                logger.error(f"Could not create spec for plugin: {plugin_name}")

        except Exception as e:
            logger.error(f"Error loading cross-platform plugin {plugin_name}: {e}")

    return plugins_loaded


def verify_cross_platform_plugins():
    """
    Verify that all cross-platform plugins are properly registered.

    Returns:
        Dictionary with verification results
    """
    verification_results = {
        "total_plugins": 0,
        "cross_platform_plugins": [],
        "missing_plugins": [],
        "plugin_details": {},
    }

    # Expected cross-platform plugins
    expected_plugins = {
        "react_native": "React Native",
        "flutter": "Flutter",
        "xamarin": "Xamarin",
        "unity": "Unity",
        "capacitor_cordova": "Capacitor/Cordova",
    }

    # Check registered plugins
    all_plugins = plugin_registry.get_all_plugins()
    verification_results["total_plugins"] = len(all_plugins)

    for plugin_id, plugin_name in expected_plugins.items():
        if plugin_id in all_plugins:
            plugin = all_plugins[plugin_id]
            verification_results["cross_platform_plugins"].append(plugin_id)

            # Get plugin details
            try:
                details = {
                    "name": plugin.get_language_name(),
                    "version": plugin.get_language_version(),
                    "frameworks": plugin.get_supported_frameworks(),
                    "has_analyzer": hasattr(plugin, "analyze_error"),
                    "has_fix_generator": hasattr(plugin, "generate_fix"),
                    "can_handle_test": True,
                }

                # Test if plugin can handle a basic error
                test_error = {
                    "framework": plugin_id,
                    "message": "Test error",
                    "error_type": "TestError",
                }

                try:
                    can_handle = plugin.can_handle(test_error)
                    details["can_handle_test"] = can_handle
                except Exception as e:
                    details["can_handle_test"] = False
                    details["can_handle_error"] = str(e)

                verification_results["plugin_details"][plugin_id] = details

            except Exception as e:
                verification_results["plugin_details"][plugin_id] = {
                    "error": f"Failed to get details: {e}"
                }
        else:
            verification_results["missing_plugins"].append(plugin_id)

    return verification_results


def test_cross_platform_error_handling():
    """
    Test cross-platform error handling with sample errors.

    Returns:
        Dictionary with test results
    """
    test_results = {"tests_run": 0, "tests_passed": 0, "test_details": {}}

    # Sample errors for each cross-platform framework
    sample_errors = {
        "react_native": {
            "framework": "react-native",
            "message": "Native module RCTCamera cannot be null",
            "error_type": "Error",
            "runtime": "react-native",
        },
        "flutter": {
            "framework": "flutter",
            "message": "RenderFlex overflowed by 123 pixels",
            "error_type": "FlutterError",
            "language": "dart",
        },
        "xamarin": {
            "framework": "xamarin",
            "message": "DependencyService could not resolve IMyService",
            "error_type": "InvalidOperationException",
            "runtime": "xamarin",
        },
        "unity": {
            "framework": "unity",
            "message": "NullReferenceException: Object reference not set to an instance of an object",
            "error_type": "NullReferenceException",
            "runtime": "unity",
        },
        "capacitor_cordova": {
            "framework": "capacitor",
            "message": "Plugin Camera not found",
            "error_type": "PluginNotFoundError",
            "runtime": "capacitor",
        },
    }

    # Test each plugin
    for plugin_id, error_data in sample_errors.items():
        test_results["tests_run"] += 1
        test_detail = {
            "plugin_id": plugin_id,
            "can_handle": False,
            "analysis": None,
            "fix": None,
            "errors": [],
        }

        try:
            # Get the plugin
            plugin = plugin_registry.get_plugin(plugin_id)

            if plugin is None:
                test_detail["errors"].append(f"Plugin {plugin_id} not found")
                test_results["test_details"][plugin_id] = test_detail
                continue

            # Test can_handle
            try:
                can_handle = plugin.can_handle(error_data)
                test_detail["can_handle"] = can_handle

                if not can_handle:
                    test_detail["errors"].append("Plugin cannot handle the test error")

            except Exception as e:
                test_detail["errors"].append(f"Error in can_handle: {e}")

            # Test analyze_error
            if test_detail["can_handle"]:
                try:
                    analysis = plugin.analyze_error(error_data)
                    test_detail["analysis"] = {
                        "category": analysis.get("category"),
                        "subcategory": analysis.get("subcategory"),
                        "confidence": analysis.get("confidence"),
                        "has_suggestion": bool(analysis.get("suggested_fix")),
                    }
                except Exception as e:
                    test_detail["errors"].append(f"Error in analyze_error: {e}")

                # Test generate_fix
                if test_detail["analysis"]:
                    try:
                        fix = plugin.generate_fix(
                            error_data, test_detail["analysis"], ""
                        )
                        test_detail["fix"] = {
                            "generated": fix is not None,
                            "type": fix.get("type") if fix else None,
                            "has_description": (
                                bool(fix.get("description")) if fix else False
                            ),
                        }
                    except Exception as e:
                        test_detail["errors"].append(f"Error in generate_fix: {e}")

            # If no errors, mark as passed
            if not test_detail["errors"]:
                test_results["tests_passed"] += 1

        except Exception as e:
            test_detail["errors"].append(f"General error: {e}")

        test_results["test_details"][plugin_id] = test_detail

    return test_results


def get_cross_platform_integration_status():
    """
    Get comprehensive status of cross-platform integration.

    Returns:
        Dictionary with integration status
    """
    status = {
        "timestamp": str(datetime.now()),
        "plugins_loaded": 0,
        "verification": None,
        "test_results": None,
        "recommendations": [],
    }

    try:
        # Load plugins
        status["plugins_loaded"] = load_cross_platform_plugins()

        # Verify plugins
        status["verification"] = verify_cross_platform_plugins()

        # Test plugins
        status["test_results"] = test_cross_platform_error_handling()

        # Generate recommendations
        if status["verification"]["missing_plugins"]:
            status["recommendations"].append(
                f"Missing plugins: {', '.join(status['verification']['missing_plugins'])}"
            )

        if status["test_results"]["tests_passed"] < status["test_results"]["tests_run"]:
            failed_tests = (
                status["test_results"]["tests_run"]
                - status["test_results"]["tests_passed"]
            )
            status["recommendations"].append(
                f"{failed_tests} plugin tests failed - check test_details for specific issues"
            )

        if status["plugins_loaded"] == 0:
            status["recommendations"].append(
                "No cross-platform plugins were loaded - check plugin files and dependencies"
            )

        # Overall health
        all_plugins_present = len(status["verification"]["missing_plugins"]) == 0
        all_tests_passed = (
            status["test_results"]["tests_passed"]
            == status["test_results"]["tests_run"]
        )

        if all_plugins_present and all_tests_passed and status["plugins_loaded"] > 0:
            status["overall_status"] = "HEALTHY"
        elif (
            status["plugins_loaded"] > 0 and status["test_results"]["tests_passed"] > 0
        ):
            status["overall_status"] = "PARTIAL"
        else:
            status["overall_status"] = "UNHEALTHY"

    except Exception as e:
        status["error"] = str(e)
        status["overall_status"] = "ERROR"
        status["recommendations"].append(f"Integration check failed: {e}")

    return status


def initialize_cross_platform_integration():
    """
    Initialize cross-platform framework integration.

    This function should be called during system startup to ensure
    all cross-platform plugins are loaded and verified.

    Returns:
        True if initialization was successful, False otherwise
    """
    logger.info("Initializing cross-platform framework integration...")

    try:
        # Load plugins
        plugins_loaded = load_cross_platform_plugins()
        logger.info(f"Loaded {plugins_loaded} cross-platform plugins")

        # Verify plugins
        verification = verify_cross_platform_plugins()
        logger.info(
            f"Verified {len(verification['cross_platform_plugins'])} cross-platform plugins"
        )

        if verification["missing_plugins"]:
            logger.warning(
                f"Missing plugins: {', '.join(verification['missing_plugins'])}"
            )

        # Run basic tests
        test_results = test_cross_platform_error_handling()
        logger.info(
            f"Plugin tests: {test_results['tests_passed']}/{test_results['tests_run']} passed"
        )

        # Report any failures
        for plugin_id, test_detail in test_results["test_details"].items():
            if test_detail["errors"]:
                logger.warning(
                    f"Plugin {plugin_id} has issues: {'; '.join(test_detail['errors'])}"
                )

        # Success if we have some working plugins
        success = plugins_loaded > 0 and test_results["tests_passed"] > 0

        if success:
            logger.info("Cross-platform framework integration initialized successfully")
        else:
            logger.error("Cross-platform framework integration initialization failed")

        return success

    except Exception as e:
        logger.error(f"Error during cross-platform integration initialization: {e}")
        return False


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Import datetime for status
    import importlib.util
    from datetime import datetime

    # Initialize and get status
    success = initialize_cross_platform_integration()

    if success:
        # Get detailed status
        status = get_cross_platform_integration_status()

        print("\n" + "=" * 60)
        print("CROSS-PLATFORM INTEGRATION STATUS")
        print("=" * 60)
        print(f"Overall Status: {status['overall_status']}")
        print(f"Plugins Loaded: {status['plugins_loaded']}")
        print(
            f"Cross-Platform Plugins: {len(status['verification']['cross_platform_plugins'])}"
        )
        print(
            f"Plugin Tests: {status['test_results']['tests_passed']}/{status['test_results']['tests_run']}"
        )

        if status["verification"]["cross_platform_plugins"]:
            print(
                f"\nActive Plugins: {', '.join(status['verification']['cross_platform_plugins'])}"
            )

        if status["verification"]["missing_plugins"]:
            print(
                f"Missing Plugins: {', '.join(status['verification']['missing_plugins'])}"
            )

        if status["recommendations"]:
            print("\nRecommendations:")
            for rec in status["recommendations"]:
                print(f"  - {rec}")

        print("\nPlugin Details:")
        for plugin_id, details in status["verification"]["plugin_details"].items():
            if "error" not in details:
                print(f"  {plugin_id}: {details['name']} v{details['version']}")
                print(f"    Frameworks: {', '.join(details['frameworks'])}")
                print(f"    Can Handle Test: {details['can_handle_test']}")
            else:
                print(f"  {plugin_id}: ERROR - {details['error']}")

        print("=" * 60)
    else:
        print("Cross-platform integration initialization failed!")
        sys.exit(1)
