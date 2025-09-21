#!/usr/bin/env python3
"""
Homeostasis unified CLI entry point.

This script provides a unified command-line interface for all Homeostasis functionality,
including the orchestrator, rule management, and LLM integration.
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from modules.analysis.rule_cli import create_parser as create_rule_parser
from modules.analysis.rule_cli import main as rule_main

# Import CLI modules
from modules.llm_integration.llm_cli import create_llm_cli_parser
from modules.llm_integration.llm_cli import main as llm_main


def create_main_parser() -> argparse.ArgumentParser:
    """
    Create the main CLI parser with subcommands.

    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description="Homeostasis - Self-Healing Systems Framework", prog="homeostasis"
    )

    subparsers = parser.add_subparsers(
        dest="module",
        help="Available modules",
        title="modules",
        description="Choose a module to interact with",
    )

    # LLM integration commands
    llm_parser = subparsers.add_parser(
        "llm", help="LLM integration commands (API keys, provider management)"
    )

    # Add LLM subcommands
    llm_subparsers = llm_parser.add_subparsers(dest="llm_command", help="LLM commands")

    # set-key command
    set_key_parser = llm_subparsers.add_parser(
        "set-key", help="Set API key for an LLM provider"
    )
    set_key_parser.add_argument(
        "provider",
        choices=["openai", "anthropic", "openrouter"],
        help="LLM provider name",
    )
    set_key_parser.add_argument(
        "--key", "-k", help="API key (if not provided, will be prompted securely)"
    )
    set_key_parser.add_argument(
        "--no-validate", action="store_true", help="Skip API key validation"
    )

    # list-keys command
    list_keys_parser = llm_subparsers.add_parser(
        "list-keys", help="List configured API keys"
    )
    list_keys_parser.add_argument(
        "--show-masked", action="store_true", help="Show masked versions of the keys"
    )

    # remove-key command
    remove_key_parser = llm_subparsers.add_parser(
        "remove-key", help="Remove API key for a provider"
    )
    remove_key_parser.add_argument(
        "provider",
        choices=["openai", "anthropic", "openrouter"],
        help="LLM provider name",
    )

    # validate-key command
    validate_key_parser = llm_subparsers.add_parser(
        "validate-key", help="Validate an API key"
    )
    validate_key_parser.add_argument(
        "provider",
        choices=["openai", "anthropic", "openrouter"],
        help="LLM provider name",
    )

    # test-providers command
    test_providers_parser = llm_subparsers.add_parser(
        "test-providers", help="Test all configured providers"
    )
    test_providers_parser.add_argument(
        "--providers",
        "-p",
        nargs="+",
        choices=["openai", "anthropic", "openrouter"],
        help="Specific providers to test (default: all)",
    )

    # Rule management commands
    rule_parser = subparsers.add_parser("rule", help="Rule management commands")

    # Add rule subcommands
    rule_subparsers = rule_parser.add_subparsers(
        dest="rule_command", help="Rule commands"
    )

    # Rule list command
    rule_list_parser = rule_subparsers.add_parser("list", help="List rules")
    rule_list_parser.add_argument("--category", "-c", help="Filter by category")
    rule_list_parser.add_argument("--tag", "-t", help="Filter by tag")
    rule_list_parser.add_argument("--severity", "-s", help="Filter by severity")
    rule_list_parser.add_argument("--confidence", help="Filter by confidence")
    rule_list_parser.add_argument(
        "--format", "-f", choices=["text", "json"], default="text", help="Output format"
    )
    rule_list_parser.add_argument("--output", "-o", help="Output file for JSON format")

    # Rule show command
    rule_show_parser = rule_subparsers.add_parser("show", help="Show rule details")
    rule_show_parser.add_argument("rule_id", help="ID of the rule to show")

    # Rule stats command
    rule_stats_parser = rule_subparsers.add_parser("stats", help="Show rule statistics")
    rule_stats_parser.add_argument("--output", "-o", help="Output file for statistics")

    # Orchestrator commands (simplified)
    orchestrator_parser = subparsers.add_parser(
        "orchestrator", help="Run the main orchestrator"
    )
    orchestrator_parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="config.yaml",
        help="Path to configuration file",
    )
    orchestrator_parser.add_argument(
        "--log-level",
        "-l",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level",
    )
    orchestrator_parser.add_argument(
        "--demo", "-d", action="store_true", help="Run in demonstration mode"
    )

    # Developer tools commands
    dev_parser = subparsers.add_parser(
        "dev", help="Developer tools for testing and integration"
    )

    dev_subparsers = dev_parser.add_subparsers(
        dest="dev_command", help="Developer tool commands"
    )

    # Simulate command
    simulate_parser = dev_subparsers.add_parser(
        "simulate", help="Run healing simulation"
    )
    simulate_parser.add_argument("--name", "-n", required=True, help="Simulation name")
    simulate_parser.add_argument(
        "--language", "-l", required=True, help="Programming language"
    )
    simulate_parser.add_argument(
        "--error-type", "-e", default="generic", help="Error type to simulate"
    )
    simulate_parser.add_argument("--code", "-c", help="Code snippet to test")
    simulate_parser.add_argument(
        "--file", "-f", type=Path, help="File containing code to test"
    )
    simulate_parser.add_argument(
        "--docker", action="store_true", help="Use Docker for isolation"
    )
    simulate_parser.add_argument(
        "--output", "-o", type=Path, help="Output path for results"
    )

    # Calculate effectiveness command
    effectiveness_parser = dev_subparsers.add_parser(
        "effectiveness", help="Calculate healing effectiveness"
    )
    effectiveness_parser.add_argument("--start-date", help="Start date (YYYY-MM-DD)")
    effectiveness_parser.add_argument("--end-date", help="End date (YYYY-MM-DD)")
    effectiveness_parser.add_argument("--language", "-l", help="Filter by language")
    effectiveness_parser.add_argument("--error-type", "-e", help="Filter by error type")
    effectiveness_parser.add_argument(
        "--format",
        "-f",
        choices=["json", "html", "markdown"],
        default="json",
        help="Output format",
    )
    effectiveness_parser.add_argument("--output", "-o", type=Path, help="Output file")

    # Validate templates command
    validate_parser = dev_subparsers.add_parser(
        "validate", help="Validate healing templates"
    )
    validate_parser.add_argument(
        "path", type=Path, help="Path to template file or directory"
    )
    validate_parser.add_argument(
        "-r", "--recursive", action="store_true", help="Recursively validate templates"
    )
    validate_parser.add_argument(
        "-f",
        "--format",
        choices=["text", "json", "markdown"],
        default="text",
        help="Output format",
    )
    validate_parser.add_argument("-o", "--output", type=Path, help="Output file")
    validate_parser.add_argument(
        "--fail-on-warning",
        action="store_true",
        help="Exit with error if warnings found",
    )

    return parser


def handle_llm_commands(args: argparse.Namespace) -> None:
    """Handle LLM module commands."""
    from modules.llm_integration.llm_cli import (
        cmd_list_keys,
        cmd_remove_key,
        cmd_set_key,
        cmd_test_providers,
        cmd_validate_key,
    )

    if args.llm_command == "set-key":
        cmd_set_key(args)
    elif args.llm_command == "list-keys":
        cmd_list_keys(args)
    elif args.llm_command == "remove-key":
        cmd_remove_key(args)
    elif args.llm_command == "validate-key":
        cmd_validate_key(args)
    elif args.llm_command == "test-providers":
        cmd_test_providers(args)
    else:
        print(
            "Invalid LLM command. Use 'homeostasis llm --help' for available commands."
        )
        sys.exit(1)


def handle_rule_commands(args: argparse.Namespace) -> None:
    """Handle rule module commands."""
    from modules.analysis.rule_cli import RuleManager, RuleStats

    if args.rule_command == "list":
        print("Listing rules...")
        # TODO: Implement rule listing with filters
    elif args.rule_command == "show":
        manager = RuleManager()
        rule = manager.show_rule(args.rule_id)
        if rule:
            manager.print_rule_details(rule)
        else:
            print(f"Rule with ID '{args.rule_id}' not found.")
    elif args.rule_command == "stats":
        stats = RuleStats()
        stats.print_summary()
        if args.output:
            stats.export_stats(Path(args.output))
    else:
        print(
            "Invalid rule command. Use 'homeostasis rule --help' for available commands."
        )
        sys.exit(1)


def handle_orchestrator_commands(args: argparse.Namespace) -> None:
    """Handle orchestrator commands."""
    # Import and run the orchestrator
    from orchestrator.orchestrator import main as orchestrator_main

    # Set up sys.argv for the orchestrator
    sys.argv = ["orchestrator.py"]
    if args.config != "config.yaml":
        sys.argv.extend(["--config", args.config])
    if args.log_level != "INFO":
        sys.argv.extend(["--log-level", args.log_level])
    if args.demo:
        sys.argv.append("--demo")

    orchestrator_main()


def handle_dev_commands(args: argparse.Namespace) -> None:
    """Handle developer tool commands."""
    from datetime import datetime

    if args.dev_command == "simulate":
        from modules.developer_tools.sandbox import HealingSimulator, SimulationConfig

        # Prepare simulation config
        config = SimulationConfig(
            name=args.name, language=args.language, error_type=args.error_type
        )

        # Load code from file or argument
        if args.file:
            config.code_snippet = args.file.read_text()
        elif args.code:
            config.code_snippet = args.code
        else:
            print("Error: Either --code or --file must be provided")
            sys.exit(1)

        # Run simulation
        simulator = HealingSimulator()
        with simulator.simulation_context(use_docker=args.docker) as sim:
            result = sim.simulate(config)

            # Output results
            if args.output:
                sim.export_results(args.output, format="json")
                print(f"Results saved to {args.output}")
            else:
                print(f"Simulation ID: {result.simulation_id}")
                print(f"Success: {result.success}")
                print(
                    f"Duration: {(result.end_time - result.start_time).total_seconds():.2f}s"
                )
                if result.logs:
                    print("\nLogs:")
                    for log in result.logs:
                        print(f"  {log}")

    elif args.dev_command == "effectiveness":
        from modules.developer_tools.effectiveness_calculator import (
            EffectivenessCalculator,
        )

        calculator = EffectivenessCalculator()

        # Parse dates if provided
        start_date = (
            datetime.fromisoformat(args.start_date) if args.start_date else None
        )
        end_date = datetime.fromisoformat(args.end_date) if args.end_date else None

        # Calculate effectiveness
        report = calculator.calculate_effectiveness(
            start_date=start_date,
            end_date=end_date,
            language=args.language,
            error_type=args.error_type,
        )

        # Generate report
        output = calculator.export_report(report, format=args.format)

        # Output results
        if args.output:
            args.output.write_text(output)
            print(f"Report saved to {args.output}")
        else:
            print(output)

    elif args.dev_command == "validate":
        from modules.developer_tools.template_validator import TemplateValidator

        validator = TemplateValidator()

        # Validate templates
        if args.path.is_file():
            results = [validator.validate_template(args.path)]
        elif args.path.is_dir():
            results = validator.validate_directory(args.path, recursive=args.recursive)
        else:
            print(f"Error: {args.path} is not a valid file or directory")
            sys.exit(1)

        # Generate report
        report = validator.generate_report(results, format=args.format)

        # Output report
        if args.output:
            args.output.write_text(report)
            print(f"Report saved to {args.output}")
        else:
            print(report)

        # Exit with appropriate code
        invalid_count = sum(1 for r in results if not r.is_valid)
        warning_count = sum(len(r.warnings) for r in results)

        if invalid_count > 0:
            sys.exit(1)
        elif args.fail_on_warning and warning_count > 0:
            sys.exit(1)

    else:
        print(
            "Invalid developer command. Use 'homeostasis dev --help' for available commands."
        )
        sys.exit(1)


def main() -> None:
    """Main entry point for the unified CLI."""
    parser = create_main_parser()

    # If no arguments provided, show help
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    try:
        if args.module == "llm":
            handle_llm_commands(args)
        elif args.module == "rule":
            handle_rule_commands(args)
        elif args.module == "orchestrator":
            handle_orchestrator_commands(args)
        elif args.module == "dev":
            handle_dev_commands(args)
        else:
            parser.print_help()
            sys.exit(1)

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
