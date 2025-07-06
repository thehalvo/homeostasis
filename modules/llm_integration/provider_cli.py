#!/usr/bin/env python3
"""
CLI commands for provider management and extensibility.

Provides commands for discovering, configuring, and managing LLM providers.
"""

import json
import click
from pathlib import Path
from typing import Optional
from .provider_registry import get_provider_registry


@click.group(name="providers")
def provider_cli():
    """Manage LLM providers and plugins."""
    pass


@provider_cli.command()
@click.option("--capability", help="Filter by capability")
@click.option("--feature", help="Filter by feature") 
@click.option("--tag", help="Filter by tag")
@click.option("--tier", help="Filter by pricing tier (free, low, medium, high)")
@click.option("--latency", help="Filter by latency class (low, medium, high)")
@click.option("--min-reliability", type=float, help="Minimum reliability score")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
def list(capability: Optional[str], feature: Optional[str], tag: Optional[str], 
         tier: Optional[str], latency: Optional[str], min_reliability: Optional[float],
         output_json: bool):
    """List available LLM providers."""
    registry = get_provider_registry()
    
    # Apply filters
    providers = registry.list_providers(
        with_capability=capability,
        feature_filter=feature,
        tag_filter=tag
    )
    
    if tier:
        providers = [p for p in providers if p in registry.get_providers_by_tier(tier)]
    
    if latency:
        providers = [p for p in providers if p in registry.get_providers_by_latency(latency)]
    
    if min_reliability is not None:
        reliable_providers = registry.get_most_reliable_providers(min_reliability)
        providers = [p for p in providers if p in reliable_providers]
    
    if output_json:
        provider_info = {}
        for provider_name in providers:
            metadata = registry.get_metadata(provider_name)
            if metadata:
                provider_info[provider_name] = {
                    "display_name": metadata.display_name,
                    "description": metadata.description,
                    "version": metadata.version,
                    "pricing_tier": metadata.pricing_tier,
                    "latency_class": metadata.latency_class,
                    "reliability_score": metadata.reliability_score,
                    "capabilities": [c.name for c in metadata.capabilities],
                    "features": list(metadata.features),
                    "tags": list(metadata.tags)
                }
        click.echo(json.dumps(provider_info, indent=2))
    else:
        if not providers:
            click.echo("No providers found matching criteria.")
            return
        
        click.echo(f"Found {len(providers)} provider(s):")
        click.echo()
        
        for provider_name in providers:
            metadata = registry.get_metadata(provider_name)
            if metadata:
                click.echo(f"📡 {metadata.display_name} ({provider_name})")
                click.echo(f"   Description: {metadata.description}")
                click.echo(f"   Version: {metadata.version}")
                click.echo(f"   Pricing: {metadata.pricing_tier}, Latency: {metadata.latency_class}")
                click.echo(f"   Reliability: {metadata.reliability_score:.2f}")
                click.echo(f"   Features: {', '.join(list(metadata.features)[:5])}")
                if len(metadata.features) > 5:
                    click.echo(f"            (+{len(metadata.features) - 5} more)")
                click.echo()


@provider_cli.command()
@click.argument("provider_name")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
def info(provider_name: str, output_json: bool):
    """Show detailed information about a specific provider."""
    registry = get_provider_registry()
    metadata = registry.get_metadata(provider_name)
    
    if not metadata:
        click.echo(f"❌ Provider '{provider_name}' not found.")
        return
    
    if output_json:
        info_data = {
            "name": metadata.name,
            "display_name": metadata.display_name,
            "description": metadata.description,
            "version": metadata.version,
            "author": metadata.author,
            "homepage": metadata.homepage,
            "documentation": metadata.documentation,
            "pricing_tier": metadata.pricing_tier,
            "latency_class": metadata.latency_class,
            "reliability_score": metadata.reliability_score,
            "default_model": metadata.default_model,
            "authentication_methods": metadata.authentication_methods,
            "capabilities": [
                {
                    "name": c.name,
                    "description": c.description,
                    "required": c.required
                } for c in metadata.capabilities
            ],
            "supported_models": metadata.supported_models,
            "model_families": metadata.model_families,
            "features": list(metadata.features),
            "tags": list(metadata.tags),
            "rate_limits": metadata.rate_limits,
            "context_limits": metadata.context_limits
        }
        click.echo(json.dumps(info_data, indent=2))
    else:
        click.echo(f"📡 {metadata.display_name} ({metadata.name})")
        click.echo(f"{'=' * 50}")
        click.echo(f"Description: {metadata.description}")
        click.echo(f"Version: {metadata.version}")
        click.echo(f"Author: {metadata.author}")
        
        if metadata.homepage:
            click.echo(f"Homepage: {metadata.homepage}")
        if metadata.documentation:
            click.echo(f"Documentation: {metadata.documentation}")
        
        click.echo()
        click.echo("📊 Characteristics:")
        click.echo(f"  Pricing Tier: {metadata.pricing_tier}")
        click.echo(f"  Latency Class: {metadata.latency_class}")
        click.echo(f"  Reliability Score: {metadata.reliability_score:.2f}")
        click.echo(f"  Default Model: {metadata.default_model}")
        
        click.echo()
        click.echo("🔐 Authentication:")
        click.echo(f"  Methods: {', '.join(metadata.authentication_methods)}")
        
        click.echo()
        click.echo("⚡ Capabilities:")
        for capability in metadata.capabilities:
            required_indicator = " (Required)" if capability.required else ""
            click.echo(f"  • {capability.name}: {capability.description}{required_indicator}")
        
        click.echo()
        click.echo("🎯 Features:")
        for feature in sorted(metadata.features):
            click.echo(f"  • {feature}")
        
        click.echo()
        click.echo("🏷️ Tags:")
        click.echo(f"  {', '.join(sorted(metadata.tags))}")
        
        if metadata.supported_models:
            click.echo()
            click.echo("🤖 Supported Models:")
            for family, models in metadata.model_families.items():
                click.echo(f"  {family.upper()}:")
                for model in models[:3]:  # Show first 3 models
                    click.echo(f"    • {model}")
                if len(models) > 3:
                    click.echo(f"    (+{len(models) - 3} more)")
        
        if metadata.rate_limits:
            click.echo()
            click.echo("📈 Rate Limits:")
            for limit_type, value in metadata.rate_limits.items():
                click.echo(f"  {limit_type}: {value:,}")


@provider_cli.command()
def capabilities():
    """List all available provider capabilities."""
    registry = get_provider_registry()
    capabilities = registry.list_capabilities()
    
    if not capabilities:
        click.echo("No capabilities found.")
        return
    
    click.echo(f"Found {len(capabilities)} capability/capabilities:")
    click.echo()
    
    for capability in capabilities:
        providers = registry.find_providers_by_capability(capability)
        click.echo(f"🔧 {capability}")
        click.echo(f"   Providers: {', '.join(providers)}")
        click.echo()


@provider_cli.command()
@click.argument("plugin_path", type=click.Path(exists=True))
def load_plugin(plugin_path: str):
    """Load a provider plugin from a file."""
    registry = get_provider_registry()
    path = Path(plugin_path)
    
    try:
        registry.load_plugin_from_path(path)
        click.echo(f"✅ Successfully loaded plugin from {plugin_path}")
    except Exception as e:
        click.echo(f"❌ Failed to load plugin from {plugin_path}: {e}")


@provider_cli.command()
@click.argument("directory_path", type=click.Path(exists=True))
def load_plugins(directory_path: str):
    """Load all provider plugins from a directory."""
    registry = get_provider_registry()
    directory = Path(directory_path)
    
    try:
        registry.load_plugins_from_directory(directory)
        click.echo(f"✅ Successfully loaded plugins from {directory_path}")
    except Exception as e:
        click.echo(f"❌ Failed to load plugins from {directory_path}: {e}")


@provider_cli.command()
@click.argument("provider_name")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
def get_schema(provider_name: str, output: Optional[str]):
    """Get configuration schema for a provider."""
    registry = get_provider_registry()
    schema = registry.get_provider_schema(provider_name)
    
    if not schema:
        click.echo(f"❌ No schema found for provider '{provider_name}'.")
        return
    
    schema_json = json.dumps(schema, indent=2)
    
    if output:
        with open(output, 'w') as f:
            f.write(schema_json)
        click.echo(f"✅ Schema saved to {output}")
    else:
        click.echo(schema_json)


@provider_cli.command()
@click.argument("provider_name")
@click.argument("config_file", type=click.Path(exists=True))
def validate_config(provider_name: str, config_file: str):
    """Validate a configuration file for a provider."""
    registry = get_provider_registry()
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        is_valid = registry.validate_provider_config(provider_name, config)
        
        if is_valid:
            click.echo(f"✅ Configuration is valid for provider '{provider_name}'.")
        else:
            click.echo(f"❌ Configuration is invalid for provider '{provider_name}'.")
    
    except json.JSONDecodeError as e:
        click.echo(f"❌ Invalid JSON in config file: {e}")
    except Exception as e:
        click.echo(f"❌ Error validating configuration: {e}")


@provider_cli.command()
@click.option("--output", "-o", type=click.Path(), help="Output file path")
def export_registry(output: Optional[str]):
    """Export registry information for debugging."""
    registry = get_provider_registry()
    info = registry.export_registry_info()
    
    info_json = json.dumps(info, indent=2)
    
    if output:
        with open(output, 'w') as f:
            f.write(info_json)
        click.echo(f"✅ Registry info exported to {output}")
    else:
        click.echo(info_json)


@provider_cli.command()
@click.option("--tier", help="Show providers by pricing tier")
@click.option("--latency", help="Show providers by latency class")
@click.option("--min-reliability", type=float, default=0.8, help="Minimum reliability score")
def recommend(tier: Optional[str], latency: Optional[str], min_reliability: float):
    """Get provider recommendations based on criteria."""
    registry = get_provider_registry()
    
    recommendations = []
    
    if tier:
        recommendations.extend(registry.get_providers_by_tier(tier))
    
    if latency:
        latency_providers = registry.get_providers_by_latency(latency)
        if recommendations:
            recommendations = [p for p in recommendations if p in latency_providers]
        else:
            recommendations.extend(latency_providers)
    
    reliable_providers = registry.get_most_reliable_providers(min_reliability)
    if recommendations:
        recommendations = [p for p in recommendations if p in reliable_providers]
    else:
        recommendations.extend(reliable_providers)
    
    if not recommendations:
        click.echo("No providers found matching criteria.")
        return
    
    click.echo(f"Recommended providers (reliability >= {min_reliability}):")
    click.echo()
    
    for provider_name in recommendations:
        metadata = registry.get_metadata(provider_name)
        if metadata:
            click.echo(f"⭐ {metadata.display_name}")
            click.echo(f"   Pricing: {metadata.pricing_tier}, Latency: {metadata.latency_class}")
            click.echo(f"   Reliability: {metadata.reliability_score:.2f}")
            click.echo(f"   Best for: {', '.join(list(metadata.tags)[:3])}")
            click.echo()


if __name__ == "__main__":
    provider_cli()