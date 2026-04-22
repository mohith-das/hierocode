import typer
from hierocode.broker.skeleton import build_skeleton
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
cache_app = typer.Typer(help="Manage the planner cache.", no_args_is_help=True)

app.add_typer(providers_app, name="providers")
app.add_typer(models_app, name="models")
app.add_typer(workers_app, name="workers")
app.add_typer(cache_app, name="cache")

@app.command()
def version():
    """Show the application version."""
    log_info(f"Hierocode version {__version__}")

@app.command()
def init(
    force: bool = typer.Option(False, "--force", "-f", help="Force overwrite existing config"),
    wizard: bool = typer.Option(False, "--wizard", "-w",
                                help="Auto-detect environment and write a tailored config."),
):
    """Initialize a Hierocode configuration (default or via auto-detecting wizard)."""
    try:
        from hierocode.utils.paths import get_config_path
        if wizard:
            from hierocode.cli_wizard import run_wizard
            detection, _ = run_wizard(write=True, force=force)
            log_success(f"Config written at {get_config_path()}")
            log_info(f"Recommended planner: [magenta]{detection.recommended_planner_type}[/magenta]")
            log_info(f"Recommended drafter: [magenta]{detection.recommended_drafter_model}[/magenta]")
            if detection.recommended_planner_type == "ollama":
                log_warning("No subscription/API planner detected — falling back to local Ollama. "
                            "Plan quality will be degraded. See plan.md for recommendations.")
        else:
            create_default_config(force=force)
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


@models_app.command("show")
def models_show():
    """Show the current planner / drafter / reviewer role bindings."""
    from hierocode.config_writer import list_roles
    try:
        for role, info in list_roles().items():
            source_color = {"explicit": "green", "legacy": "yellow",
                            "default": "dim"}.get(info["source"], "white")
            console.print(f"[bold]{role:8}[/bold] [cyan]{info['provider']}[/cyan] / "
                          f"{info['model']} [{source_color}]({info['source']})[/{source_color}]")
    except Exception as e:
        log_error(str(e))
        raise typer.Exit(code=1)


@models_app.command("set")
def models_set(
    role: str = typer.Argument(..., help="Role to configure: planner, drafter, or reviewer."),
    model: str = typer.Argument(..., help="Model name (e.g. claude-sonnet-4-6)."),
    provider: str = typer.Option(None, "--provider", "-p",
                                 help="Provider name. Defaults to the role's existing provider."),
):
    """Set the provider + model for a role in ~/.hierocode.yaml."""
    from hierocode.config_writer import ConfigWriteError, set_role_model
    try:
        set_role_model(role, model, provider=provider)
        log_success(f"Updated {role} → {provider or '(inferred)'}:{model}")
    except ConfigWriteError as e:
        log_error(str(e))
        raise typer.Exit(code=1)


@cache_app.command("clear")
def cache_clear():
    """Delete all cached plans under ~/.cache/hierocode/plans/."""
    from hierocode.broker.plan_cache import clear_cache
    count = clear_cache()
    log_success(f"Cleared {count} cached plan{'s' if count != 1 else ''}.")


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

def _resolve_route(conf, role: str, override_model: str | None):
    """Get (provider_name, model) for a role, applying a CLI-level model override if provided."""
    from hierocode.broker.router import get_route
    p_name, model = get_route(conf, role)
    if override_model:
        model = override_model
    return p_name, model


@app.command()
def plan(
    task: str = typer.Option(..., "--task", "-t", help="The coding task to plan."),
    planner_model: str = typer.Option(None, "--planner-model", help="Override planner model."),
    drafter_model: str = typer.Option(None, "--drafter-model", help="Override drafter model."),
    repo_root: str = typer.Option(".", "--repo", help="Repo root to build skeleton from."),
):
    """Generate a structured JSON plan for a task, sized to the drafter's capacity."""
    try:
        conf = load_config()
        from hierocode.broker.planner import generate_plan
        from hierocode.broker.capacity import build_capacity_profile

        planner_p, planner_m = _resolve_route(conf, "planner", planner_model)
        drafter_p, drafter_m = _resolve_route(conf, "drafter", drafter_model)
        planner_prov = get_provider(planner_p, conf.providers[planner_p])
        drafter_prov = get_provider(drafter_p, conf.providers[drafter_p])

        log_info(f"Profiling drafter {drafter_p} ({drafter_m})...")
        profile = build_capacity_profile(drafter_prov, drafter_m)
        log_info(f"Drafter tier: [magenta]{profile.tier}[/magenta] "
                 f"(ctx={profile.context_window}, "
                 f"max_input={profile.max_input_tokens}, "
                 f"max_files={profile.max_files_per_unit})")

        skeleton = build_skeleton(repo_root)

        from hierocode.broker.plan_cache import cache_key, read_cached_plan, write_cached_plan
        key = cache_key(task, skeleton, planner_m, drafter_m)
        planned = read_cached_plan(key)
        if planned is not None:
            log_info(f"Plan cache [green]HIT[/green] (skipping planner call: key={key[:16]}…)")
        else:
            log_info(f"Plan cache MISS — planning with {planner_p} ({planner_m})...")
            with console.status(f"[bold magenta]Planning with {planner_m}[/bold magenta]",
                                spinner="dots"):
                planned = generate_plan(task, skeleton, profile, planner_prov, planner_m)
            write_cached_plan(key, planned)

        console.print(f"\n[bold]Plan ({len(planned.units)} units):[/bold]")
        for u in planned.units:
            console.print(f"  [cyan]{u.id}[/cyan]: {u.goal}")
            if u.target_files:
                console.print(f"    target: {', '.join(u.target_files)}")
            if u.acceptance:
                console.print(f"    [dim]accept: {u.acceptance}[/dim]")

    except Exception as e:
        log_error(str(e))
        raise typer.Exit(code=1)


@app.command()
def run(
    task: str = typer.Option(..., "--task", "-t", help="The coding task."),
    planner_model: str = typer.Option(None, "--planner-model", help="Override planner model."),
    drafter_model: str = typer.Option(None, "--drafter-model", help="Override drafter model."),
    repo_root: str = typer.Option(".", "--repo", help="Repo root."),
    estimate: bool = typer.Option(False, "--estimate", "-e",
                                  help="Predict cost / message quota and exit without running."),
):
    """Plan, draft, and QA an entire task end-to-end."""
    try:
        conf = load_config()
        from hierocode.broker.planner import generate_plan
        from hierocode.broker.capacity import build_capacity_profile
        from hierocode.broker.dispatcher import run_plan

        planner_p, planner_m = _resolve_route(conf, "planner", planner_model)
        drafter_p, drafter_m = _resolve_route(conf, "drafter", drafter_model)
        planner_prov = get_provider(planner_p, conf.providers[planner_p])
        drafter_prov = get_provider(drafter_p, conf.providers[drafter_p])

        log_info(f"Profiling drafter {drafter_p} ({drafter_m})...")
        profile = build_capacity_profile(drafter_prov, drafter_m)
        log_info(f"Drafter tier: [magenta]{profile.tier}[/magenta]")

        skeleton = build_skeleton(repo_root)

        if estimate:
            from hierocode.broker.estimator import estimate_task_cost
            est = estimate_task_cost(
                task=task, skeleton=skeleton, profile=profile,
                planner_provider_config=conf.providers[planner_p], planner_model=planner_m,
                max_revisions_per_unit=conf.policy.max_revisions_per_unit,
            )
            console.print("\n[bold]Cost estimate[/bold]")
            console.print(f"Planner: [magenta]{est.planner_kind}[/magenta] ({planner_m})")
            console.print(f"Expected plan units: {est.expected_plan_units}")
            console.print(f"Expected QA calls: {est.expected_qa_calls}")
            console.print(f"Planner tokens (in/out): {est.planner_input_tokens} / "
                          f"{est.planner_output_tokens}")
            if est.approximate_cost_usd is not None:
                console.print(f"[green]Approx. cost: ${est.approximate_cost_usd:.4f}[/green]")
            if est.approximate_message_count is not None:
                console.print(f"[yellow]Approx. messages against quota: "
                              f"{est.approximate_message_count}[/yellow]")
            for note in est.notes:
                console.print(f"  [dim]• {note}[/dim]")
            return

        from hierocode.broker.plan_cache import cache_key, read_cached_plan, write_cached_plan
        key = cache_key(task, skeleton, planner_m, drafter_m)
        planned = read_cached_plan(key)
        if planned is not None:
            log_info(f"Plan cache [green]HIT[/green] (key={key[:16]}…)")
        else:
            log_info(f"Plan cache MISS — planning with {planner_p} ({planner_m})...")
            with console.status(f"[bold magenta]Planning with {planner_m}[/bold magenta]",
                                spinner="dots"):
                planned = generate_plan(task, skeleton, profile, planner_prov, planner_m)
            write_cached_plan(key, planned)
        log_info(f"Plan has {len(planned.units)} units. Dispatching...")

        with console.status(f"[bold cyan]Dispatching to {drafter_m}[/bold cyan]", spinner="dots"):
            result = run_plan(
                planned, profile, planner_prov, planner_m, drafter_prov, drafter_m, repo_root,
                max_revisions_per_unit=conf.policy.max_revisions_per_unit,
                max_escalations_per_task=conf.policy.max_escalations_per_task,
            )

        console.print(f"\n[bold]Task complete.[/bold] "
                      f"revisions={result.total_revisions} escalations={result.total_escalations}")
        for u in result.units:
            status_color = {"completed": "green", "escalated": "yellow",
                            "failed": "red", "revised": "cyan"}.get(u.status, "white")
            console.print(f"\n[bold][{status_color}]{u.unit_id}[/{status_color}][/bold] "
                          f"({u.status}, revisions={u.revision_count})")
            if u.reason:
                console.print(f"  [dim]{u.reason}[/dim]")
            if u.diff:
                console.print(u.diff if u.diff else "[dim](no changes)[/dim]")

    except Exception as e:
        log_error(str(e))
        raise typer.Exit(code=1)


@app.command()
def draft(
    task: str = typer.Option(..., "--task", "-t", help="The coding task."),
    filepath: str = typer.Option(..., "--file", "-f", help="Target file to modify."),
    drafter_model: str = typer.Option(None, "--drafter-model", help="Override drafter model."),
):
    """Draft a single-file patch using the drafter role. Single-file escape hatch for `run`."""
    try:
        conf = load_config()
        from pathlib import Path

        from hierocode.broker.budget import pack_context
        from hierocode.broker.capacity import build_capacity_profile
        from hierocode.broker.plan_schema import TaskUnit
        from hierocode.broker.prompts import build_drafter_prompt
        from hierocode.repo.diffing import generate_unified_diff
        from hierocode.repo.files import read_file_safe

        drafter_p, drafter_m = _resolve_route(conf, "drafter", drafter_model)
        drafter_prov = get_provider(drafter_p, conf.providers[drafter_p])

        log_info(f"Profiling drafter {drafter_p} ({drafter_m})...")
        profile = build_capacity_profile(drafter_prov, drafter_m)

        unit = TaskUnit(id="draft", goal=task, target_files=[filepath], acceptance="")
        packed = pack_context(unit, profile, repo_root=".")
        prompt = build_drafter_prompt(unit, packed.content)

        log_info(f"Drafting {filepath} with {drafter_p} ({drafter_m})...")
        with console.status(f"[bold cyan]Drafting with {drafter_m}[/bold cyan]", spinner="dots"):
            drafted = drafter_prov.generate(prompt=prompt, model=drafter_m)

        lines = drafted.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines)

        original = read_file_safe(Path(filepath))
        diff = generate_unified_diff(original, cleaned, filepath)

        console.print("\n[bold]Proposed Patch:[/bold]")
        console.print(diff if diff else "No changes proposed.")

    except Exception as e:
        log_error(str(e))
        raise typer.Exit(code=1)


@app.command()
def review(
    task: str = typer.Option(..., "--task", "-t", help="The coding task."),
    filepath: str = typer.Option(..., "--file", "-f", help="Target file to review."),
    reviewer_model: str = typer.Option(None, "--reviewer-model", help="Override reviewer model."),
):
    """Review a file against a task. Runs the reviewer role on a raw file (no diff required)."""
    try:
        conf = load_config()
        from hierocode.repo.files import read_file_safe

        reviewer_p, reviewer_m = _resolve_route(conf, "reviewer", reviewer_model)
        reviewer_prov = get_provider(reviewer_p, conf.providers[reviewer_p])

        content = read_file_safe(filepath)
        prompt = (
            f"You are a strict code reviewer.\n\n"
            f"Task: {task}\n"
            f"File: {filepath}\n\n"
            f"Current content:\n```\n{content}\n```\n\n"
            f"Identify correctness issues relative to the task. Be concise and specific."
        )

        log_info(f"Reviewing {filepath} with {reviewer_p} ({reviewer_m})...")
        with console.status(f"[bold magenta]Reviewing with {reviewer_m}[/bold magenta]",
                            spinner="dots"):
            output = reviewer_prov.generate(prompt=prompt, model=reviewer_m)

        console.print("\n[bold]Review Output:[/bold]")
        console.print(output)

    except Exception as e:
        log_error(str(e))
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
