---
name: claude-recall
description: >
  Search past Claude Code sessions using semantic, exact, or fuzzy search.
  Use when the user wants to find previous conversations, recall how they
  solved a problem, or reference past work across sessions.
argument-hint: "[search query]"
allowed-tools:
  - Bash(uv run --directory ~/.claude/skills/claude-recall/scripts *)
---

# claude-recall

Search your Claude Code session history with semantic, exact, fuzzy, or hybrid search.

## Prerequisites

- **Ollama** must be installed and running: `brew install ollama && ollama serve`
- **Embedding model** must be pulled: `ollama pull nomic-embed-text`

## Quick Start

```bash
# Index all sessions (first run takes a few minutes)
uv run --directory ~/.claude/skills/claude-recall/scripts python cli.py index

# Semantic search (default, best for natural language queries)
uv run --directory ~/.claude/skills/claude-recall/scripts python cli.py search "how did I fix the auth bug"

# Exact text search
uv run --directory ~/.claude/skills/claude-recall/scripts python cli.py search --exact "kubectl exec prod"

# Fuzzy search (tolerates typos)
uv run --directory ~/.claude/skills/claude-recall/scripts python cli.py search --fuzzy "kubernetis deploy"

# Hybrid search (combines semantic + keyword)
uv run --directory ~/.claude/skills/claude-recall/scripts python cli.py search --hybrid "terraform state"
```

## Indexing

```bash
# Incremental index (only changed files)
uv run --directory ~/.claude/skills/claude-recall/scripts python cli.py index

# Force full reindex
uv run --directory ~/.claude/skills/claude-recall/scripts python cli.py index --force

# Check index status
uv run --directory ~/.claude/skills/claude-recall/scripts python cli.py index --status
```

## Search Options

- `--limit N` / `-n N`: Max results (default 10)
- `--project PATH` / `-p PATH`: Filter by project path (substring)
- `--role user|assistant` / `-r`: Filter by message role
- `--after DATE`: Only results after this date
- `--before DATE`: Only results before this date
- `--session ID`: Search within a specific session
- `--group-by project|session`: Group results into a ranked table by project or session (great for finding which session to resume)
- `--json`: Output as JSON

### Grouped Search Examples

```bash
# Find which project has the most relevant sessions for a topic
uv run --directory ~/.claude/skills/claude-recall/scripts python cli.py search "auth bug" --limit 50 --group-by project

# Find which session to resume
uv run --directory ~/.claude/skills/claude-recall/scripts python cli.py search "kubernetes deploy" --limit 30 --group-by session
```

## Daemon (Background Indexing)

```bash
# Start/stop the file watcher
uv run --directory ~/.claude/skills/claude-recall/scripts python cli.py daemon start
uv run --directory ~/.claude/skills/claude-recall/scripts python cli.py daemon stop
uv run --directory ~/.claude/skills/claude-recall/scripts python cli.py daemon status

# Enable/disable auto-start at login
uv run --directory ~/.claude/skills/claude-recall/scripts python cli.py daemon enable
uv run --directory ~/.claude/skills/claude-recall/scripts python cli.py daemon disable
```

## Configuration

```bash
# Show config
uv run --directory ~/.claude/skills/claude-recall/scripts python cli.py config show

# Change embedding model
uv run --directory ~/.claude/skills/claude-recall/scripts python cli.py config set embedding_model nomic-embed-text-v2-moe

# Change batch size
uv run --directory ~/.claude/skills/claude-recall/scripts python cli.py config set batch_size 64
```

## When to Use

- "What did we discuss about X last week?"
- "Find the session where I debugged the auth issue"
- "Search for conversations about terraform migrations"
- "How did I solve [specific problem] before?"

When the user invokes `/claude-recall <query>`, run a semantic search with the provided query using `$ARGUMENTS` as the search term. If `$ARGUMENTS` is empty, show the index status instead.
