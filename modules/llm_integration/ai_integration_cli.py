#!/usr/bin/env python3
"""
CLI interface for AI integration bridge.

Provides commands for managing and testing AI integration capabilities.
"""

import json
from pathlib import Path
from typing import Optional

import click

from .ai_integration_bridge import AICapability, CodeContext, get_ai_integration_bridge


@click.group(name="ai")
def ai_integration_cli():
    """AI integration and Phase 13 collaboration commands."""
    pass


@ai_integration_cli.command()
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
def status(output_json: bool):
    """Show AI integration status and available capabilities."""
    bridge = get_ai_integration_bridge()
    status_info = bridge.get_integration_status()
    capabilities = bridge.get_available_capabilities()

    if output_json:
        output_data = {
            "status": status_info,
            "available_capabilities": [cap.value for cap in capabilities],
        }
        click.echo(json.dumps(output_data, indent=2))
    else:
        click.echo("🤖 AI Integration Status")
        click.echo("=" * 40)

        # AI Models
        ai_models = status_info.get("ai_models", {})
        click.echo(f"🧠 AI Models: {sum(ai_models.values())} total")
        for capability, count in ai_models.items():
            click.echo(f"   {capability}: {count} model(s)")

        click.echo()

        # Components
        click.echo(f"🎨 Style Preservers: {status_info.get('style_preservers', 0)}")
        click.echo(f"🔍 Code Understanders: {status_info.get('code_understanders', 0)}")
        click.echo(
            f"🔗 Multi-file Coordinators: {status_info.get('multi_file_coordinators', 0)}"
        )
        click.echo(
            f"📚 Continuous Learners: {status_info.get('continuous_learners', 0)}"
        )

        click.echo()

        # Hooks
        hooks = status_info.get("hooks", {})
        click.echo(f"🪝 Integration Hooks: {sum(hooks.values())} total")
        for hook_type, count in hooks.items():
            click.echo(f"   {hook_type}: {count}")

        click.echo()

        # Available Capabilities
        if capabilities:
            click.echo("🚀 Available AI Capabilities:")
            for cap in capabilities:
                click.echo(f"   • {cap.value}")
        else:
            click.echo("⚠️  No AI capabilities available yet")

        click.echo()
        click.echo(
            "💡 Note: Phase 13 components can be registered to extend capabilities"
        )


@ai_integration_cli.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--capabilities", help="Comma-separated list of capabilities to use")
@click.option(
    "--language", help="Programming language (auto-detected if not specified)"
)
@click.option("--framework", help="Framework context")
@click.option("--output", "-o", type=click.Path(), help="Output file for results")
def analyze(
    file_path: str,
    capabilities: Optional[str],
    language: Optional[str],
    framework: Optional[str],
    output: Optional[str],
):
    """Analyze code using available AI capabilities."""
    bridge = get_ai_integration_bridge()
    path = Path(file_path)

    # Read file content
    try:
        with open(path, "r", encoding="utf-8") as f:
            code_content = f.read()
    except Exception as e:
        click.echo(f"❌ Failed to read file: {e}")
        return

    # Auto-detect language if not provided
    if not language:
        # Simple detection based on file extension
        ext_to_lang = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".java": "java",
            ".go": "go",
            ".rs": "rust",
            ".cpp": "cpp",
            ".c": "c",
            ".cs": "csharp",
            ".rb": "ruby",
            ".php": "php",
        }
        language = ext_to_lang.get(path.suffix.lower(), "unknown")

    # Parse capabilities
    if capabilities:
        try:
            capability_list = [
                AICapability(cap.strip()) for cap in capabilities.split(",")
            ]
        except ValueError as e:
            click.echo(f"❌ Invalid capability: {e}")
            return
    else:
        # Use all available capabilities
        capability_list = bridge.get_available_capabilities()
        if not capability_list:
            click.echo("⚠️  No AI capabilities available for analysis")
            return

    # Create code context
    code_context = CodeContext(
        file_path=str(path),
        code_content=code_content,
        language=language,
        framework=framework,
    )

    # Perform analysis
    click.echo(f"🔍 Analyzing {path} with {len(capability_list)} capabilities...")
    try:
        results = bridge.analyze_with_ai(code_context, capability_list)

        if not results:
            click.echo("No analysis results generated.")
            return

        # Output results
        if output:
            output_data = {
                "file_path": str(path),
                "language": language,
                "framework": framework,
                "capabilities_used": [cap.value for cap in capability_list],
                "results": [
                    {
                        "capability": result.capability.value,
                        "confidence": result.confidence,
                        "analysis_data": result.analysis_data,
                        "recommendations": result.recommendations,
                        "processing_time": result.processing_time,
                    }
                    for result in results
                ],
            }

            with open(output, "w") as f:
                json.dump(output_data, f, indent=2)

            click.echo(f"✅ Analysis results saved to {output}")
        else:
            # Display results
            click.echo(f"✅ Analysis completed. Found {len(results)} results:")
            click.echo()

            for result in results:
                confidence_indicator = (
                    "🔴"
                    if result.confidence < 0.3
                    else "🟡" if result.confidence < 0.7 else "🟢"
                )
                click.echo(f"{confidence_indicator} {result.capability.value}")
                click.echo(f"   Confidence: {result.confidence:.2f}")
                click.echo(f"   Processing Time: {result.processing_time:.3f}s")

                if result.recommendations:
                    click.echo(f"   Recommendations: {len(result.recommendations)}")
                    for rec in result.recommendations[:3]:  # Show first 3
                        click.echo(f"     • {rec}")
                    if len(result.recommendations) > 3:
                        click.echo(
                            f"     ... and {len(result.recommendations) - 3} more"
                        )

                click.echo()

    except Exception as e:
        click.echo(f"❌ Analysis failed: {e}")


@ai_integration_cli.command()
@click.argument("original_file", type=click.Path(exists=True))
@click.argument("generated_file", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="Output file for styled code")
@click.option("--language", help="Programming language")
def preserve_style(
    original_file: str,
    generated_file: str,
    output: Optional[str],
    language: Optional[str],
):
    """Apply style preservation to generated code."""
    bridge = get_ai_integration_bridge()

    # Read files
    try:
        with open(original_file, "r") as f:
            original_code = f.read()
        with open(generated_file, "r") as f:
            generated_code = f.read()
    except Exception as e:
        click.echo(f"❌ Failed to read files: {e}")
        return

    # Auto-detect language if not provided
    if not language:
        path = Path(original_file)
        ext_to_lang = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".java": "java",
        }
        language = ext_to_lang.get(path.suffix.lower(), "unknown")

    # Create code context
    code_context = CodeContext(
        file_path=original_file, code_content=original_code, language=language
    )

    # Apply style preservation
    click.echo("🎨 Applying style preservation...")
    try:
        styled_code = bridge.preserve_and_enhance_style(
            original_code, generated_code, code_context
        )

        if output:
            with open(output, "w") as f:
                f.write(styled_code)
            click.echo(f"✅ Styled code saved to {output}")
        else:
            click.echo("✅ Style preservation completed:")
            click.echo("=" * 40)
            click.echo(styled_code)

    except Exception as e:
        click.echo(f"❌ Style preservation failed: {e}")


@ai_integration_cli.command()
@click.argument("config_file", type=click.Path(exists=True))
def test_integration(config_file: str):
    """Test AI integration using a configuration file."""
    bridge = get_ai_integration_bridge()

    # Load test configuration
    try:
        with open(config_file, "r") as f:
            if config_file.endswith(".yaml") or config_file.endswith(".yml"):
                import yaml

                config = yaml.safe_load(f)
            else:
                config = json.load(f)
    except Exception as e:
        click.echo(f"❌ Failed to load configuration: {e}")
        return

    # Extract test cases
    test_cases = config.get("test_cases", [])
    if not test_cases:
        click.echo("❌ No test cases found in configuration")
        return

    click.echo(f"🧪 Running {len(test_cases)} AI integration tests...")
    click.echo()

    passed = 0
    failed = 0

    for i, test_case in enumerate(test_cases, 1):
        click.echo(f"Test {i}: {test_case.get('name', 'Unnamed test')}")

        try:
            # Create code context from test case
            code_context = CodeContext(
                file_path=test_case.get("file_path", "test.py"),
                code_content=test_case.get("code_content", ""),
                language=test_case.get("language", "python"),
                framework=test_case.get("framework"),
                error_info=test_case.get("error_info"),
            )

            # Get capabilities to test
            capabilities = [
                AICapability(cap) for cap in test_case.get("capabilities", [])
            ]

            # Perform analysis
            results = bridge.analyze_with_ai(code_context, capabilities)

            # Check expectations
            expected_results = test_case.get("expected_results", {})

            # Validate results
            validation_passed = True

            if "min_confidence" in expected_results:
                min_confidence = expected_results["min_confidence"]
                actual_confidences = [r.confidence for r in results]
                if not actual_confidences or max(actual_confidences) < min_confidence:
                    validation_passed = False
                    click.echo(
                        f"   ❌ Confidence too low: {max(actual_confidences) if actual_confidences else 0:.2f} < {min_confidence}"
                    )

            if "expected_capabilities" in expected_results:
                expected_caps = set(expected_results["expected_capabilities"])
                actual_caps = set(r.capability.value for r in results)
                if not expected_caps.issubset(actual_caps):
                    validation_passed = False
                    missing = expected_caps - actual_caps
                    click.echo(f"   ❌ Missing capabilities: {missing}")

            if validation_passed:
                click.echo("   ✅ Passed")
                passed += 1
            else:
                click.echo("   ❌ Failed validation")
                failed += 1

        except Exception as e:
            click.echo(f"   ❌ Error: {e}")
            failed += 1

        click.echo()

    # Summary
    total = passed + failed
    click.echo(f"🏁 Test Summary: {passed}/{total} passed ({passed/total*100:.1f}%)")

    if failed > 0:
        exit(1)


@ai_integration_cli.command()
def capabilities():
    """List all available AI capabilities."""
    click.echo("🚀 Available AI Capabilities:")
    click.echo()

    for capability in AICapability:
        click.echo(f"• {capability.value}")

        # Add description for each capability
        descriptions = {
            "deep_learning_classification": "Advanced error classification using deep neural networks",
            "transformer_code_understanding": "Code analysis using transformer-based models",
            "fine_tuned_code_generation": "Code generation using fine-tuned language models",
            "style_preserving_generation": "Generate code while preserving original style patterns",
            "semantic_code_analysis": "Deep semantic understanding of code structure and meaning",
            "hierarchical_error_classification": "Multi-level error classification with context",
            "zero_shot_error_detection": "Detect new error types without specific training",
            "multi_file_coordination": "Coordinate changes across multiple related files",
            "continuous_learning": "Learn and improve from feedback and outcomes",
            "human_in_the_loop": "Integration with human review and annotation systems",
        }

        description = descriptions.get(capability.value, "Advanced AI capability")
        click.echo(f"  {description}")
        click.echo()


@ai_integration_cli.command()
@click.option("--output", "-o", type=click.Path(), help="Output directory for examples")
def examples(output: Optional[str]):
    """Generate example integration test files."""
    from pathlib import Path

    output_dir = Path(output) if output else Path.cwd() / "ai_integration_examples"
    output_dir.mkdir(exist_ok=True)

    # Example test configuration
    test_config = {
        "test_cases": [
            {
                "name": "Python error analysis",
                "file_path": "test.py",
                "language": "python",
                "code_content": "def divide(a, b):\n    return a / b\n\nresult = divide(10, 0)",
                "error_info": {
                    "type": "ZeroDivisionError",
                    "message": "division by zero",
                },
                "capabilities": [
                    "deep_learning_classification",
                    "style_preserving_generation",
                ],
                "expected_results": {
                    "min_confidence": 0.5,
                    "expected_capabilities": ["deep_learning_classification"],
                },
            },
            {
                "name": "JavaScript style preservation",
                "file_path": "test.js",
                "language": "javascript",
                "framework": "react",
                "code_content": "function MyComponent() {\n  const [count, setCount] = useState(0);\n  return <div>{count}</div>;\n}",
                "capabilities": [
                    "style_preserving_generation",
                    "semantic_code_analysis",
                ],
                "expected_results": {"min_confidence": 0.3},
            },
        ]
    }

    # Save test configuration
    config_file = output_dir / "test_config.json"
    with open(config_file, "w") as f:
        json.dump(test_config, f, indent=2)

    # Example Python code file
    python_file = output_dir / "example.py"
    with open(python_file, "w") as f:
        f.write(
            """def calculate_average(numbers):
    total = sum(numbers)
    return total / len(numbers)

# This will cause a division by zero error
result = calculate_average([])
print(result)
"""
        )

    # Example JavaScript file
    js_file = output_dir / "example.js"
    with open(js_file, "w") as f:
        f.write(
            """function fetchUserData(userId) {
    // Missing error handling
    return fetch(`/api/users/${userId}`)
        .then(response => response.json());
}

// Usage without error handling
fetchUserData(123).then(data => {
    console.log(data.name);
});
"""
        )

    click.echo(f"✅ Generated example files in {output_dir}:")
    click.echo(f"  • {config_file} - Test configuration")
    click.echo(f"  • {python_file} - Python code example")
    click.echo(f"  • {js_file} - JavaScript code example")
    click.echo()
    click.echo("To run tests:")
    click.echo(f"  homeostasis ai test-integration {config_file}")
    click.echo()
    click.echo("To analyze individual files:")
    click.echo(f"  homeostasis ai analyze {python_file}")


if __name__ == "__main__":
    ai_integration_cli()
