"""Search logic for claude-recall: exact, fuzzy, semantic, and hybrid search."""

import json
from datetime import datetime
from typing import Optional

import lancedb
import ollama
from rapidfuzz import fuzz
from rich.console import Console
from rich.table import Table
from rich.text import Text

from config import Config, load_config

console = Console()


def arrow_to_dicts(arrow_table) -> list[dict]:
    """Convert a pyarrow table to a list of dicts without pandas."""
    columns = arrow_table.column_names
    rows = []
    for i in range(arrow_table.num_rows):
        row = {}
        for col in columns:
            val = arrow_table.column(col)[i].as_py()
            row[col] = val
        rows.append(row)
    return rows


def embed_query(query: str, config: Config) -> list[float]:
    """Embed a search query using Ollama."""
    response = ollama.embed(
        model=config.embedding_model,
        input=[query],
        truncate=True,
    )
    return response["embeddings"][0]


def format_timestamp(ts: str) -> str:
    """Format a timestamp string for display."""
    try:
        if "T" in ts:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            return dt.strftime("%Y-%m-%d %H:%M")
        # Try unix ms
        dt = datetime.fromtimestamp(int(ts) / 1000)
        return dt.strftime("%Y-%m-%d %H:%M")
    except (ValueError, TypeError, OSError):
        return ts[:16] if ts else "unknown"


def format_project(project_path: str) -> str:
    """Shorten project path for display."""
    home = str(Config().sessions_path.parent.parent)  # ~/.claude -> home
    if project_path.startswith(home):
        return "~" + project_path[len(home):]
    return project_path


def truncate_text(text: str, max_len: int = 200) -> str:
    """Truncate text for display with ellipsis."""
    text = text.replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


class Searcher:
    """Search engine for claude-recall."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or load_config()
        self.db = lancedb.connect(str(self.config.db_path))

    def _get_table(self):
        if "vectors" not in self.db.table_names():
            return None
        return self.db.open_table("vectors")

    def _apply_filters(self, query_builder, project: Optional[str] = None,
                       role: Optional[str] = None, after: Optional[str] = None,
                       before: Optional[str] = None, session_id: Optional[str] = None):
        """Apply common filters to a LanceDB query."""
        filters = []
        if project:
            filters.append(f"project_path LIKE '%{project}%'")
        if role:
            filters.append(f"role = '{role}'")
        if session_id:
            filters.append(f"session_id = '{session_id}'")
        if after:
            filters.append(f"timestamp >= '{after}'")
        if before:
            filters.append(f"timestamp <= '{before}'")

        if filters:
            combined = " AND ".join(filters)
            query_builder = query_builder.where(combined, prefilter=True)

        return query_builder

    def search_semantic(self, query: str, limit: int = 10, **filters) -> list[dict]:
        """Semantic search using vector similarity."""
        table = self._get_table()
        if table is None:
            return []

        vector = embed_query(query, self.config)
        q = table.search(vector).metric("cosine").limit(limit)
        q = self._apply_filters(q, **filters)

        try:
            return arrow_to_dicts(q.to_arrow())
        except Exception as e:
            console.print(f"[red]Semantic search error: {e}[/red]")
            return []

    def search_exact(self, query: str, limit: int = 10, **filters) -> list[dict]:
        """Exact text search using FTS index."""
        table = self._get_table()
        if table is None:
            return []

        try:
            q = table.search(query, query_type="fts").limit(limit)
            q = self._apply_filters(q, **filters)
            return arrow_to_dicts(q.to_arrow())
        except Exception as e:
            # FTS index might not exist, fall back to scan
            console.print(f"[yellow]FTS search failed ({e}), falling back to scan...[/yellow]")
            return self._scan_search(query, limit, **filters)

    def search_fuzzy(self, query: str, limit: int = 10, threshold: int = 60, **filters) -> list[dict]:
        """Fuzzy text search using rapidfuzz on scanned results."""
        table = self._get_table()
        if table is None:
            return []

        # Get a broader set of candidates using FTS first, then fuzzy rank
        try:
            q = table.search(query, query_type="fts").limit(limit * 10)
            q = self._apply_filters(q, **filters)
            candidates = arrow_to_dicts(q.to_arrow())
        except Exception:
            candidates = self._scan_search(query, limit * 10, **filters)

        # If FTS yielded nothing, scan all
        if not candidates:
            candidates = self._scan_all(**filters)

        # Score with rapidfuzz
        scored = []
        for record in candidates:
            score = fuzz.partial_ratio(query.lower(), record.get("text", "").lower())
            if score >= threshold:
                record["_fuzzy_score"] = score
                scored.append(record)

        scored.sort(key=lambda x: x["_fuzzy_score"], reverse=True)
        return scored[:limit]

    def search_hybrid(self, query: str, limit: int = 10, **filters) -> list[dict]:
        """Hybrid search combining semantic and FTS results."""
        table = self._get_table()
        if table is None:
            return []

        vector = embed_query(query, self.config)

        try:
            q = (
                table.search(query, query_type="hybrid", vector_column_name="vector", fts_columns="text")
                .vector(vector)
                .limit(limit)
            )
            q = self._apply_filters(q, **filters)
            return arrow_to_dicts(q.to_arrow())
        except Exception:
            # Fall back to semantic if hybrid not supported
            return self.search_semantic(query, limit, **filters)

    def _scan_search(self, query: str, limit: int = 10, **filters) -> list[dict]:
        """Brute-force text scan (fallback when FTS unavailable)."""
        table = self._get_table()
        if table is None:
            return []

        query_lower = query.lower()
        try:
            q = table.search().select(["id", "text", "session_id", "message_uuid", "role",
                          "timestamp", "project_path", "file_path"]).limit(100000)
            q = self._apply_filters(q, **filters)
            all_records = arrow_to_dicts(q.to_arrow())
            matches = [r for r in all_records if query_lower in r.get("text", "").lower()]
            return matches[:limit]
        except Exception:
            return []

    def _scan_all(self, **filters) -> list[dict]:
        """Get all records (for fuzzy search fallback). Limited to 10000."""
        table = self._get_table()
        if table is None:
            return []
        try:
            q = table.search().select(["id", "text", "session_id", "message_uuid", "role",
                          "timestamp", "project_path", "file_path"]).limit(10000)
            q = self._apply_filters(q, **filters)
            return arrow_to_dicts(q.to_arrow())
        except Exception:
            return []


def display_results(results: list[dict], query: str, mode: str, as_json: bool = False):
    """Display search results in a formatted table or as JSON."""
    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    if as_json:
        # Clean up internal fields and non-serializable types
        clean = []
        for r in results:
            entry = {
                "session_id": r.get("session_id", ""),
                "message_uuid": r.get("message_uuid", ""),
                "role": r.get("role", ""),
                "timestamp": r.get("timestamp", ""),
                "project_path": r.get("project_path", ""),
                "text": r.get("text", ""),
            }
            if "_distance" in r:
                entry["score"] = round(1 - r["_distance"], 4)
            if "_fuzzy_score" in r:
                entry["fuzzy_score"] = r["_fuzzy_score"]
            clean.append(entry)
        console.print_json(json.dumps(clean, indent=2, default=str))
        return

    console.print(f"\n[bold]Search results[/bold] ({mode}, {len(results)} hits)\n")

    for i, r in enumerate(results, 1):
        role_color = "blue" if r.get("role") == "user" else "green"
        role_label = r.get("role", "?")
        ts = format_timestamp(r.get("timestamp", ""))
        project = format_project(r.get("project_path", ""))
        text_preview = truncate_text(r.get("text", ""), 300)
        session = r.get("session_id", "")

        # Score info
        score_str = ""
        if "_distance" in r:
            score_str = f" [dim](similarity: {1 - r['_distance']:.3f})[/dim]"
        if "_fuzzy_score" in r:
            score_str = f" [dim](fuzzy: {r['_fuzzy_score']}%)[/dim]"

        console.print(
            f"[bold]{i}.[/bold] [{role_color}]{role_label}[/{role_color}] "
            f"[dim]{ts}[/dim] [cyan]{project}[/cyan]{score_str}"
        )
        console.print(f"   {text_preview}")
        console.print(f"   [dim]cd {project} && claude -r {session}[/dim]\n")


def display_grouped_results(results: list[dict], query: str, mode: str,
                            group_by: str = "project", as_json: bool = False):
    """Display search results grouped by project or session, ranked by hit count."""
    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    # Group results
    groups: dict[str, list[dict]] = {}
    for r in results:
        if group_by == "project":
            key = r.get("project_path", "unknown")
        else:
            key = r.get("session_id", "unknown")
        groups.setdefault(key, []).append(r)

    # Sort groups by number of hits (descending), then by best similarity
    def group_sort_key(item):
        key, hits = item
        count = len(hits)
        best_score = 0
        for h in hits:
            if "_distance" in h:
                best_score = max(best_score, 1 - h["_distance"])
            elif "_fuzzy_score" in h:
                best_score = max(best_score, h["_fuzzy_score"] / 100)
        return (-count, -best_score)

    sorted_groups = sorted(groups.items(), key=group_sort_key)

    if as_json:
        output = []
        for key, hits in sorted_groups:
            output.append({
                group_by: key,
                "hits": len(hits),
                "top_result": truncate_text(hits[0].get("text", ""), 200),
                "sessions": list(set(h.get("session_id", "")[:8] for h in hits)),
            })
        console.print_json(json.dumps(output, indent=2, default=str))
        return

    label = "Project" if group_by == "project" else "Session"
    console.print(f"\n[bold]Results grouped by {label.lower()}[/bold] ({mode}, {len(results)} hits)\n")

    table = Table()
    table.add_column("#", style="bold", width=3)
    table.add_column(label, style="cyan")
    table.add_column("Hits", style="green", justify="right")
    table.add_column("Sessions", style="dim")
    table.add_column("Top match preview")

    for i, (key, hits) in enumerate(sorted_groups, 1):
        display_key = format_project(key) if group_by == "project" else key
        session_ids = sorted(set(h.get("session_id", "") for h in hits))
        preview = truncate_text(hits[0].get("text", ""), 120)
        table.add_row(str(i), display_key, str(len(hits)),
                      "\n".join(session_ids[:5]) + (f"\n(+{len(session_ids) - 5})" if len(session_ids) > 5 else ""),
                      preview)

    console.print(table)

    # Print resume commands for top sessions with project context
    seen_sessions = []
    for _, hits in sorted_groups:
        for h in hits:
            sid = h.get("session_id", "")
            proj = h.get("project_path", "")
            if sid and sid not in [s for s, _ in seen_sessions]:
                seen_sessions.append((sid, proj))

    if seen_sessions:
        console.print("\n[bold]Resume commands:[/bold]")
        for sid, proj in seen_sessions[:5]:
            proj_short = format_project(proj)
            console.print(f"  [dim]cd {proj_short} && claude -r {sid}[/dim]")
