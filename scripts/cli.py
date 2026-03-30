#!/usr/bin/env python3
"""claude-recall: Local semantic search for Claude Code sessions.

Usage:
    claude-recall index              Index all sessions (incremental)
    claude-recall search "query"     Semantic search (default)
    claude-recall daemon start       Start file watcher daemon
    claude-recall config             Show configuration
    claude-recall stats              Show index statistics
"""

import json
import sys
from dataclasses import asdict
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from config import Config, load_config, save_config, set_config_value

console = Console()


@click.group()
def cli():
    """claude-recall: Search your Claude Code sessions."""
    pass


# --- Index commands ---

@cli.command()
@click.option("--force", is_flag=True, help="Force reindex all files, ignoring cache.")
@click.option("--status", "show_status", is_flag=True, help="Show index status without indexing.")
@click.option("--create-indexes", is_flag=True, help="Create/rebuild search indexes after indexing.")
def index(force, show_status, create_indexes):
    """Index Claude Code sessions for search."""
    from indexer import Indexer

    indexer = Indexer()

    if show_status:
        status = indexer.get_status()
        table = Table(title="Index Status")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Session files", str(status["total_session_files"]))
        table.add_row("Indexed files", str(status["indexed_files"]))
        table.add_row("Total vectors", str(status["total_vectors"]))
        table.add_row("Embedding model", status["embedding_model"])
        table.add_row("Database path", status["db_path"])

        if status.get("indexing_in_progress"):
            table.add_section()
            cur = status["indexing_current"]
            tot = status["indexing_total"]
            pct = status["indexing_percent"]
            eta = status["indexing_eta_seconds"]
            vecs = status["indexing_vectors_so_far"]
            fname = status.get("indexing_current_file", "")

            # Format ETA
            if eta > 3600:
                eta_str = f"{eta/3600:.1f}h"
            elif eta > 60:
                eta_str = f"{eta/60:.0f}m {eta%60:.0f}s"
            else:
                eta_str = f"{eta:.0f}s"

            bar_width = 20
            filled = int(bar_width * pct / 100)
            bar = "#" * filled + "-" * (bar_width - filled)

            table.add_row(
                "[bold yellow]Indexing[/bold yellow]",
                f"[bold yellow][{bar}] {pct:.1f}%[/bold yellow]"
            )
            table.add_row("Progress", f"{cur}/{tot} files")
            table.add_row("Vectors so far", str(vecs))
            table.add_row("ETA", eta_str)
            if fname:
                table.add_row("Current file", fname)

        console.print(table)
        return

    stats = indexer.index_all(force=force)

    if create_indexes or (stats["vectors_added"] > 0):
        console.print("\nBuilding search indexes...")
        indexer.create_search_indexes()


# --- Search commands ---

@cli.command()
@click.argument("query")
@click.option("--exact", "mode", flag_value="exact", help="Exact text match.")
@click.option("--fuzzy", "mode", flag_value="fuzzy", help="Fuzzy text match.")
@click.option("--hybrid", "mode", flag_value="hybrid", help="Combined semantic + keyword.")
@click.option("--semantic", "mode", flag_value="semantic", default=True, help="Semantic similarity (default).")
@click.option("--limit", "-n", default=10, help="Max results.")
@click.option("--project", "-p", default=None, help="Filter by project path (substring match).")
@click.option("--role", "-r", default=None, type=click.Choice(["user", "assistant"]), help="Filter by role.")
@click.option("--after", default=None, help="Only results after this date (ISO format).")
@click.option("--before", default=None, help="Only results before this date (ISO format).")
@click.option("--session", default=None, help="Search within a specific session ID.")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
@click.option("--group-by", "group_by", type=click.Choice(["project", "session"]),
              default=None, help="Group results by project or session.")
def search(query, mode, limit, project, role, after, before, session, as_json, group_by):
    """Search Claude Code sessions.

    Defaults to semantic (embedding-based) search. Use --exact, --fuzzy,
    or --hybrid for other modes.
    """
    from searcher import Searcher, display_results, display_grouped_results

    searcher = Searcher()
    filters = {
        "project": project,
        "role": role,
        "after": after,
        "before": before,
        "session_id": session,
    }
    # Remove None filters
    filters = {k: v for k, v in filters.items() if v is not None}

    if mode == "exact":
        results = searcher.search_exact(query, limit=limit, **filters)
    elif mode == "fuzzy":
        results = searcher.search_fuzzy(query, limit=limit, **filters)
    elif mode == "hybrid":
        results = searcher.search_hybrid(query, limit=limit, **filters)
    else:
        results = searcher.search_semantic(query, limit=limit, **filters)

    if group_by:
        display_grouped_results(results, query, mode or "semantic", group_by=group_by, as_json=as_json)
    else:
        display_results(results, query, mode or "semantic", as_json=as_json)


# --- Daemon commands ---

@cli.group()
def daemon():
    """Manage the background indexing daemon."""
    pass


@daemon.command("start")
@click.option("--foreground", "-f", is_flag=True, help="Run in foreground (don't daemonize).")
def daemon_start(foreground):
    """Start the background file watcher daemon."""
    from daemon import DaemonManager

    mgr = DaemonManager()
    if mgr.is_running():
        console.print(f"[yellow]Daemon already running (PID {mgr.get_pid()}).[/yellow]")
        return

    if foreground:
        console.print("[cyan]Starting daemon in foreground (Ctrl+C to stop)...[/cyan]")
        mgr.run_foreground()
    else:
        mgr.start_background()
        console.print(f"[green]Daemon started (PID {mgr.get_pid()}).[/green]")


@daemon.command("stop")
def daemon_stop():
    """Stop the background daemon."""
    from daemon import DaemonManager

    mgr = DaemonManager()
    if not mgr.is_running():
        console.print("[dim]Daemon is not running.[/dim]")
        return

    pid = mgr.get_pid()
    mgr.stop()
    console.print(f"[green]Daemon stopped (was PID {pid}).[/green]")


@daemon.command("status")
def daemon_status():
    """Show daemon status."""
    from daemon import DaemonManager

    mgr = DaemonManager()
    running = mgr.is_running()
    enabled = mgr.is_enabled()

    table = Table(title="Daemon Status")
    table.add_column("Property", style="cyan")
    table.add_column("Value")

    status_text = "[green]Running[/green]" if running else "[red]Stopped[/red]"
    table.add_row("Process", status_text)
    if running:
        table.add_row("PID", str(mgr.get_pid()))

    enabled_text = "[green]Enabled[/green]" if enabled else "[dim]Disabled[/dim]"
    table.add_row("Auto-start (launchd)", enabled_text)

    config = load_config()
    table.add_row("Watch path", str(config.sessions_path))
    table.add_row("Log file", str(config.log_file))

    console.print(table)


@daemon.command("enable")
def daemon_enable():
    """Install launchd LaunchAgent for auto-start at login."""
    from daemon import DaemonManager

    mgr = DaemonManager()
    mgr.enable()
    console.print("[green]LaunchAgent installed. Daemon will auto-start at login.[/green]")


@daemon.command("disable")
def daemon_disable():
    """Remove launchd LaunchAgent."""
    from daemon import DaemonManager

    mgr = DaemonManager()
    mgr.disable()
    console.print("[green]LaunchAgent removed. Daemon will not auto-start.[/green]")


# Aliases
@daemon.command("install", hidden=True)
def daemon_install():
    """Alias for enable."""
    from daemon import DaemonManager
    DaemonManager().enable()
    console.print("[green]LaunchAgent installed.[/green]")


@daemon.command("uninstall", hidden=True)
def daemon_uninstall():
    """Alias for disable."""
    from daemon import DaemonManager
    DaemonManager().disable()
    console.print("[green]LaunchAgent removed.[/green]")


# --- Config commands ---

@cli.group("config")
def config_cmd():
    """View and modify configuration."""
    pass


@config_cmd.command("show")
def config_show():
    """Show current configuration."""
    config = load_config()
    table = Table(title="Configuration")
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="green")
    for key, value in asdict(config).items():
        table.add_row(key, str(value))
    console.print(table)


@config_cmd.command("set")
@click.argument("key")
@click.argument("value")
def config_set(key, value):
    """Set a configuration value."""
    try:
        config = set_config_value(key, value)
        console.print(f"[green]Set {key} = {value}[/green]")
    except KeyError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)


# --- Stats command ---

@cli.command()
def stats():
    """Show index statistics."""
    from indexer import Indexer

    indexer = Indexer()
    status = indexer.get_status()

    table = Table(title="claude-recall Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Session files", str(status["total_session_files"]))
    table.add_row("Indexed files", str(status["indexed_files"]))
    table.add_row("Total vectors", str(status["total_vectors"]))
    table.add_row("Embedding model", status["embedding_model"])
    table.add_row("Database path", status["db_path"])

    console.print(table)


if __name__ == "__main__":
    cli()
