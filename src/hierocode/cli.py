import typer
from hierocode.config import create_default_config, load_config
from hierocode.utils.console import log_info, log_error, log_success, log_warning, console
from hierocode.exceptions import ConfigError
from hierocode.providers import get_provider
from hierocode import __version__
from hierocode.runtime.resources import get_cpu_count, get_total_ram_gb, get_available_ram_gb
from hierocode.runtime.recommendations import suggest_workers

app = typer.Typer(help="Hierocode: Local-first hierarchical coding orchestrator.", no_args_is_help=True)
providers_app = typer.Typer(help="Manage configured providers.", no_args_is_help=True)
models_app = typer.Typer(help="Query available models.", no_args_is_help=True)
workers_app = typer.Typer(help="Manage parallel worker configuration.", no_args_is_help=True)

app.add_typer(providers_app, name="providers")
app.add_typer(models_app, name="models")
app.add_typer(workers_app, name="workers")

@app.command()
def version():
    """Show the application version."""
    log_info(f"Hierocode version {__version__}")

@app.command()
def init(force: bool = typer.Option(False, "--force", "-f", help="Force overwrite existing config")):
    """Initialize a default Hierocode configuration in your home directory."""
    try:
        create_default_config(force=force)
        from hierocode.utils.paths import get_config_path
        log_success(f"Configuration file created at {get_config_path()}")
    except ConfigError as e:
        log_error(str(e))
        raise typer.Exit(code=1)

@app.command()
def resources():
    """Print a summary of local system resources for task sizing."""
    cpus = get_cpu_count()
    total_ram = get_total_ram_gb()
    avail_ram = get_available_ram_gb()
    
    console.print("[bold]System Resources[/bold]")
    console.print(f"CPUs: [magenta]{cpus}[/magenta]")
    console.print(f"Total RAM: [magenta]{total_ram:.1f} GB[/magenta]")
    console.print(f"Available RAM: [magenta]{avail_ram:.1f} GB[/magenta]")


@app.command()
def doctor():
    """Validate config presence and check provider status."""
    try:
        conf = load_config()
        from hierocode.utils.paths import get_config_path
        log_success(f"Configuration valid at {get_config_path()}")
    except ConfigError as e:
        log_error(str(e))
        log_info("Try running 'hierocode init'")
        raise typer.Exit(code=1)

    all_providers_ok = True
    for p_name, p_conf in conf.providers.items():
        prov = get_provider(p_name, p_conf)
        if prov.healthcheck():
            log_success(f"Provider '{p_name}' ({p_conf.type}) is reachable.")
        else:
            log_warning(f"Provider '{p_name}' ({p_conf.type}) is unreachable.")
            all_providers_ok = False
            
    if all_providers_ok:
        log_success("All system checks passed.")
    else:
        log_info("Some providers not reachable. Ensure they are running and config URLs are correct.")

@providers_app.command("list")
def providers_list():
    """List configured providers."""
    try:
        conf = load_config()
        if not conf.providers:
            log_warning("No providers configured.")
            return
            
        for name, p_conf in conf.providers.items():
            console.print(f"- [bold]{name}[/bold] ({p_conf.type}) -> {p_conf.base_url}")
    except ConfigError as e:
        log_error(str(e))
        raise typer.Exit(code=1)

@models_app.command("list")
def models_list():
    """Query configured providers and list discovered models."""
    try:
        conf = load_config()
        for name, p_conf in conf.providers.items():
            prov = get_provider(name, p_conf)
            try:
                models = prov.list_models()
                log_success(f"Provider '{name}' has {len(models)} models:")
                for m in models:
                    console.print(f"  - {m}")
            except Exception as e:
                log_error(f"Could not list models for '{name}': {e}")
    except ConfigError as e:
        log_error(str(e))
        raise typer.Exit(code=1)

@workers_app.command("suggest")
def workers_suggest(provider: str = typer.Option(None, help="Specific provider to evaluate.")):
    """Suggest worker counts based on system resources and provider specs."""
    try:
        conf = load_config()
        targets = [provider] if provider else list(conf.providers.keys())
        
        for name in targets:
            if name not in conf.providers:
                log_error(f"Provider '{name}' not found in configuration.")
                continue
                
            p_conf = conf.providers[name]
            prov = get_provider(name, p_conf)
            loc = "Local" if prov.is_local() else "Remote"
            
            console.print(f"\n[bold]{name}[/bold] ({p_conf.type}, {loc})")
            
            safe = suggest_workers(prov, conf.parallelization, "safe")
            balanced = suggest_workers(prov, conf.parallelization, "balanced")
            aggressive = suggest_workers(prov, conf.parallelization, "aggressive")
            
            console.print(f"  Safe: [green]{safe}[/green]")
            console.print(f"  Balanced: [yellow]{balanced}[/yellow]")
            console.print(f"  Aggressive: [red]{aggressive}[/red]")
            
    except ConfigError as e:
        log_error(str(e))
        raise typer.Exit(code=1)

@app.command()
def plan(
    task: str = typer.Option(..., "--task", "-t", help="The coding task to plan."),
):
    """Generate a plan for a given task using the review (large) model."""
    try:
        conf = load_config()
        from hierocode.broker.router import get_review_model_route
        from hierocode.broker.planner import generate_plan
        
        p_name, model = get_review_model_route(conf)
        prov = get_provider(p_name, conf.providers[p_name])
        
        log_info(f"Planning task using {p_name} ({model})...")
        output = generate_plan(task, prov, model)
        
        console.print("\n[bold]Plan Generated:[/bold]")
        console.print(output)
        
    except Exception as e:
        log_error(str(e))
        raise typer.Exit(code=1)


@app.command()
def draft(
    task: str = typer.Option(..., "--task", "-t", help="The coding task."),
    filepath: str = typer.Option(..., "--file", "-f", help="Target file to modify.")
):
    """Draft a candidate patch for a file using the draft (small) model."""
    try:
        conf = load_config()
        from hierocode.broker.router import get_draft_model_route
        from hierocode.broker.context import build_file_context
        from hierocode.repo.prompts import build_draft_prompt
        from hierocode.repo.diffing import generate_unified_diff
        
        p_name, model = get_draft_model_route(conf)
        prov = get_provider(p_name, conf.providers[p_name])
        
        log_info(f"Drafting patch for {filepath} using {p_name} ({model})...")
        
        content = build_file_context(filepath)
        prompt = build_draft_prompt(task, filepath, content)
        
        # Single worker for simple CLI draft
        drafted_content = prov.generate(prompt=prompt, model=model)
        
        if '<' in drafted_content or '```' in drafted_content:
            # Simple heuristic cleanup if model wrapped it in markdown
            lines = drafted_content.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            drafted_content = "\n".join(lines)
            
        diff = generate_unified_diff(content, drafted_content, filepath)
        
        console.print("\n[bold]Proposed Patch:[/bold]")
        console.print(diff if diff else "No changes proposed.")
        
    except Exception as e:
        log_error(str(e))
        raise typer.Exit(code=1)


@app.command()
def review(
    task: str = typer.Option(..., "--task", "-t", help="The coding task."),
    filepath: str = typer.Option(..., "--file", "-f", help="Target file to review.")
):
    """Review a file related to a task."""
    try:
        conf = load_config()
        from hierocode.broker.router import get_review_model_route
        from hierocode.broker.context import build_file_context
        from hierocode.repo.prompts import build_review_prompt
        
        p_name, model = get_review_model_route(conf)
        prov = get_provider(p_name, conf.providers[p_name])
        
        log_info(f"Reviewing {filepath} using {p_name} ({model})...")
        
        content = build_file_context(filepath)
        prompt = build_review_prompt(task, filepath, content)
        
        output = prov.generate(prompt=prompt, model=model)
        
        console.print("\n[bold]Review Output:[/bold]")
        console.print(output)
        
    except Exception as e:
        log_error(str(e))
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()
