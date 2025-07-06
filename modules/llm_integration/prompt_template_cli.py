#!/usr/bin/env python3
"""
CLI interface for prompt template management.

Provides commands for creating, editing, and managing user-defined prompt templates.
"""

import json
import yaml
import click
from pathlib import Path
from typing import Optional
from .prompt_template_manager import get_prompt_template_manager, PromptType


@click.group(name="templates")
def template_cli():
    """Manage prompt templates for LLM interactions."""
    pass


@template_cli.command()
@click.option("--type", "prompt_type", type=click.Choice([t.value for t in PromptType]), help="Filter by prompt type")
@click.option("--domain", help="Filter by domain")
@click.option("--language", help="Filter by programming language")
@click.option("--framework", help="Filter by framework")
@click.option("--complexity", help="Filter by complexity level")
@click.option("--tag", help="Filter by tag")
@click.option("--user-only", is_flag=True, help="Show only user-defined templates")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
def list(prompt_type: Optional[str], domain: Optional[str], language: Optional[str], 
         framework: Optional[str], complexity: Optional[str], tag: Optional[str],
         user_only: bool, output_json: bool):
    """List available prompt templates."""
    manager = get_prompt_template_manager()
    
    # Convert string to enum if provided
    prompt_type_enum = PromptType(prompt_type) if prompt_type else None
    
    templates = manager.list_templates(
        prompt_type=prompt_type_enum,
        domain=domain,
        language=language,
        framework=framework,
        complexity_level=complexity,
        tag=tag,
        user_templates_only=user_only
    )
    
    if output_json:
        template_info = {}
        for template_id in templates:
            info = manager.get_template_info(template_id)
            if info:
                template_info[template_id] = info
        click.echo(json.dumps(template_info, indent=2))
    else:
        if not templates:
            click.echo("No templates found matching criteria.")
            return
        
        click.echo(f"Found {len(templates)} template(s):")
        click.echo()
        
        for template_id in templates:
            info = manager.get_template_info(template_id)
            if info:
                user_indicator = "👤 " if info["is_user_template"] else "🏠 "
                click.echo(f"{user_indicator}{info['name']} ({template_id})")
                click.echo(f"   Description: {info['description']}")
                click.echo(f"   Type: {info['prompt_type']}, Domain: {info['domain']}")
                click.echo(f"   Language: {info['language']}, Framework: {info['framework'] or 'Generic'}")
                click.echo(f"   Complexity: {info['complexity_level']}, Author: {info['author']}")
                if info['tags']:
                    click.echo(f"   Tags: {', '.join(list(info['tags']))}")
                click.echo()


@template_cli.command()
@click.argument("template_id")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
def info(template_id: str, output_json: bool):
    """Show detailed information about a template."""
    manager = get_prompt_template_manager()
    info = manager.get_template_info(template_id)
    
    if not info:
        click.echo(f"❌ Template '{template_id}' not found.")
        return
    
    if output_json:
        click.echo(json.dumps(info, indent=2))
    else:
        user_indicator = "👤 User Template" if info["is_user_template"] else "🏠 Built-in Template"
        click.echo(f"{user_indicator}: {info['name']}")
        click.echo(f"{'=' * 50}")
        click.echo(f"Description: {info['description']}")
        click.echo(f"Type: {info['prompt_type']}")
        click.echo(f"Domain: {info['domain']}")
        click.echo(f"Language: {info['language']}")
        click.echo(f"Framework: {info['framework'] or 'Generic'}")
        click.echo(f"Complexity: {info['complexity_level']}")
        click.echo(f"Author: {info['author']}")
        click.echo(f"Version: {info['version']}")
        
        if info['tags']:
            click.echo(f"Tags: {', '.join(list(info['tags']))}")
        
        click.echo()
        click.echo("📝 Required Variables:")
        for var in info['required_variables']:
            click.echo(f"  • {var}")
        
        if info['optional_variables']:
            click.echo()
            click.echo("🔧 Optional Variables:")
            for var in info['optional_variables']:
                click.echo(f"  • {var}")
        
        if info['success_criteria']:
            click.echo()
            click.echo("✅ Success Criteria:")
            for criterion in info['success_criteria']:
                click.echo(f"  • {criterion}")
        
        if info['limitations']:
            click.echo()
            click.echo("⚠️ Limitations:")
            for limitation in info['limitations']:
                click.echo(f"  • {limitation}")
        
        if info['provider_preferences']:
            click.echo()
            click.echo("🔧 Provider Preferences:")
            for provider, score in info['provider_preferences'].items():
                click.echo(f"  • {provider}: {score}")
        
        if info['example_usage']:
            click.echo()
            click.echo("💡 Example Usage:")
            click.echo(f"  {info['example_usage']}")


@template_cli.command()
@click.argument("template_file", type=click.Path(exists=True))
@click.option("--as-builtin", is_flag=True, help="Import as built-in template (requires admin)")
def import_template(template_file: str, as_builtin: bool):
    """Import a template from a file."""
    manager = get_prompt_template_manager()
    path = Path(template_file)
    
    template_id = manager.import_template(path, as_user_template=not as_builtin)
    
    if template_id:
        template_type = "built-in" if as_builtin else "user"
        click.echo(f"✅ Successfully imported {template_type} template: {template_id}")
    else:
        click.echo(f"❌ Failed to import template from {template_file}")


@template_cli.command()
@click.argument("template_id")
@click.argument("output_file", type=click.Path())
@click.option("--format", "output_format", type=click.Choice(["yaml", "json"]), default="yaml", help="Output format")
def export(template_id: str, output_file: str, output_format: str):
    """Export a template to a file."""
    manager = get_prompt_template_manager()
    output_path = Path(output_file)
    
    # Ensure correct extension
    if output_format == "yaml" and not output_path.suffix.lower() in [".yaml", ".yml"]:
        output_path = output_path.with_suffix(".yaml")
    elif output_format == "json" and not output_path.suffix.lower() == ".json":
        output_path = output_path.with_suffix(".json")
    
    success = manager.export_template(template_id, output_path)
    
    if success:
        click.echo(f"✅ Exported template '{template_id}' to {output_path}")
    else:
        click.echo(f"❌ Failed to export template '{template_id}'")


@template_cli.command()
@click.argument("template_id")
@click.confirmation_option(prompt="Are you sure you want to delete this template?")
def delete(template_id: str):
    """Delete a user template."""
    manager = get_prompt_template_manager()
    
    if not template_id.startswith("user:"):
        click.echo("❌ Can only delete user templates (template ID must start with 'user:')")
        return
    
    success = manager.delete_user_template(template_id)
    
    if success:
        click.echo(f"✅ Deleted template: {template_id}")
    else:
        click.echo(f"❌ Failed to delete template: {template_id}")


@template_cli.command()
@click.argument("template_id")
@click.argument("variables_file", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="Output file for rendered prompt")
def test(template_id: str, variables_file: str, output: Optional[str]):
    """Test a template with variables from a file."""
    manager = get_prompt_template_manager()
    template = manager.get_template(template_id)
    
    if not template:
        click.echo(f"❌ Template '{template_id}' not found.")
        return
    
    # Load variables
    try:
        with open(variables_file, 'r') as f:
            if variables_file.endswith('.yaml') or variables_file.endswith('.yml'):
                variables = yaml.safe_load(f)
            else:
                variables = json.load(f)
    except Exception as e:
        click.echo(f"❌ Failed to load variables from {variables_file}: {e}")
        return
    
    # Validate variables
    errors = template.validate_variables(variables)
    if errors:
        click.echo("❌ Variable validation errors:")
        for error in errors:
            click.echo(f"  • {error}")
        return
    
    # Render template
    try:
        rendered = template.render(variables)
        
        if output:
            with open(output, 'w') as f:
                f.write(f"System Prompt:\n{rendered['system_prompt']}\n\n")
                f.write(f"User Prompt:\n{rendered['user_prompt']}\n")
                if 'context' in rendered:
                    f.write(f"\nContext:\n{rendered['context']}\n")
            click.echo(f"✅ Rendered prompt saved to {output}")
        else:
            click.echo("=== System Prompt ===")
            click.echo(rendered['system_prompt'])
            click.echo("\n=== User Prompt ===")
            click.echo(rendered['user_prompt'])
            if 'context' in rendered:
                click.echo("\n=== Context ===")
                click.echo(rendered['context'])
    
    except Exception as e:
        click.echo(f"❌ Failed to render template: {e}")


@template_cli.command()
def create():
    """Create a new user template interactively."""
    manager = get_prompt_template_manager()
    
    click.echo("🎨 Creating a new prompt template")
    click.echo("=" * 40)
    
    # Gather metadata
    name = click.prompt("Template name")
    description = click.prompt("Description")
    
    prompt_type = click.prompt(
        "Prompt type", 
        type=click.Choice([t.value for t in PromptType]),
        default="patch_generation"
    )
    
    domain = click.prompt("Domain (e.g., web_development, data_science)", default="general")
    language = click.prompt("Programming language", default="python")
    framework = click.prompt("Framework (optional)", default="", show_default=False)
    complexity = click.prompt(
        "Complexity level", 
        type=click.Choice(["beginner", "intermediate", "advanced"]),
        default="intermediate"
    )
    author = click.prompt("Author", default="user")
    
    # Gather variables
    click.echo("\n📝 Define required variables (comma-separated):")
    required_vars = click.prompt("Required variables", default="").split(",")
    required_vars = [v.strip() for v in required_vars if v.strip()]
    
    click.echo("\n🔧 Define optional variables (comma-separated):")
    optional_vars = click.prompt("Optional variables", default="").split(",")
    optional_vars = [v.strip() for v in optional_vars if v.strip()]
    
    # Gather prompts
    click.echo("\n📋 Enter system prompt (multi-line, end with empty line):")
    system_prompt_lines = []
    while True:
        line = input()
        if line == "":
            break
        system_prompt_lines.append(line)
    system_prompt = "\n".join(system_prompt_lines)
    
    click.echo("\n📝 Enter user prompt template (multi-line, end with empty line):")
    user_prompt_lines = []
    while True:
        line = input()
        if line == "":
            break
        user_prompt_lines.append(line)
    user_prompt_template = "\n".join(user_prompt_lines)
    
    # Create template data
    template_data = {
        "metadata": {
            "name": name,
            "description": description,
            "prompt_type": prompt_type,
            "domain": domain,
            "language": language,
            "framework": framework if framework else None,
            "complexity_level": complexity,
            "author": author,
            "version": "1.0.0",
            "tags": [],
            "required_variables": required_vars,
            "optional_variables": optional_vars
        },
        "system_prompt": system_prompt,
        "user_prompt_template": user_prompt_template
    }
    
    # Create template
    try:
        template_id = manager.create_user_template(template_data)
        click.echo(f"\n✅ Created template: {template_id}")
        click.echo(f"Template saved to: {manager.user_templates_dir / f'{name}.yaml'}")
    except Exception as e:
        click.echo(f"\n❌ Failed to create template: {e}")


@template_cli.command()
@click.argument("prompt_type", type=click.Choice([t.value for t in PromptType]))
@click.argument("language")
@click.option("--domain", help="Domain context")
@click.option("--framework", help="Framework context")
@click.option("--provider", help="Preferred LLM provider")
def find(prompt_type: str, language: str, domain: Optional[str], 
         framework: Optional[str], provider: Optional[str]):
    """Find the best template for given criteria."""
    manager = get_prompt_template_manager()
    
    prompt_type_enum = PromptType(prompt_type)
    template_id = manager.find_best_template(
        prompt_type=prompt_type_enum,
        language=language,
        domain=domain,
        framework=framework,
        provider=provider
    )
    
    if template_id:
        info = manager.get_template_info(template_id)
        click.echo(f"🎯 Best match: {template_id}")
        if info:
            click.echo(f"   {info['description']}")
            click.echo(f"   Domain: {info['domain']}, Framework: {info['framework'] or 'Generic'}")
    else:
        click.echo("❌ No suitable template found for the given criteria.")


@template_cli.command()
@click.option("--output", "-o", type=click.Path(), help="Output directory for examples")
def examples(output: Optional[str]):
    """Generate example template files."""
    output_dir = Path(output) if output else Path.cwd() / "template_examples"
    output_dir.mkdir(exist_ok=True)
    
    # Example template
    example_template = {
        "metadata": {
            "name": "python_error_fix_example",
            "description": "Example template for fixing Python errors",
            "prompt_type": "patch_generation",
            "domain": "general",
            "language": "python",
            "framework": None,
            "complexity_level": "beginner",
            "author": "example",
            "version": "1.0.0",
            "tags": ["example", "python", "error_fixing"],
            "required_variables": ["error_message", "code_snippet"],
            "optional_variables": ["context", "file_path"]
        },
        "system_prompt": "You are a Python expert helping to fix code errors. Analyze the error and provide a corrected version.",
        "user_prompt_template": """Please fix the following Python error:

Error: {{ error_message }}
{% if file_path %}File: {{ file_path }}{% endif %}
{% if context %}Context: {{ context }}{% endif %}

Code:
```python
{{ code_snippet }}
```

Please provide the corrected code and explain the fix."""
    }
    
    # Save example
    example_file = output_dir / "python_error_fix_example.yaml"
    with open(example_file, 'w') as f:
        yaml.safe_dump(example_template, f, default_flow_style=False)
    
    # Example variables file
    variables_example = {
        "error_message": "NameError: name 'x' is not defined",
        "code_snippet": "print(x)",
        "context": "Simple print statement",
        "file_path": "main.py"
    }
    
    variables_file = output_dir / "example_variables.yaml"
    with open(variables_file, 'w') as f:
        yaml.safe_dump(variables_example, f, default_flow_style=False)
    
    click.echo(f"✅ Generated example files in {output_dir}:")
    click.echo(f"  • {example_file}")
    click.echo(f"  • {variables_file}")
    click.echo()
    click.echo("To test the example:")
    click.echo(f"  homeostasis templates test python_error_fix_example {variables_file}")


if __name__ == "__main__":
    template_cli()