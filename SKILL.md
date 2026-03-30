---
name: claude-recall
description: >
  Search past Claude Code sessions using semantic, exact, or fuzzy search.
  Use when the user wants to find previous conversations, recall how they
  solved a problem, or reference past work across sessions.
argument-hint: "[search query]"
allowed-tools:
  - Bash(claude-recall *)
---

# claude-recall

Search your Claude Code session history with semantic, exact, fuzzy, or hybrid search.

## Installation Check

!`which claude-recall > /dev/null 2>&1 && echo "INSTALLED" || echo "NOT_INSTALLED: Run this to install: uv tool install git+https://github.com/JVenberg/claude-recall"`

If the output above says NOT_INSTALLED, tell the user and run the install command before proceeding.

## Prerequisites

- **Ollama** must be installed and running: `brew install ollama && ollama serve`
- **Embedding model** must be pulled: `ollama pull nomic-embed-text`
- **First index**: `claude-recall index` (takes a few minutes on first run)

## Quick Start

```bash
# Semantic search (default)
claude-recall search "how did I fix the auth bug"

# Exact text search
claude-recall search --exact "kubectl exec prod"

# Fuzzy search (tolerates typos)
claude-recall search --fuzzy "kubernetis deploy"

# Hybrid search (semantic + keyword)
claude-recall search --hybrid "terraform state"
```

## Indexing

```bash
claude-recall index              # Incremental (only changed files)
claude-recall index --force      # Force full reindex
claude-recall index --status     # Check status (shows live progress if running)
claude-recall stats              # Show index statistics
```

## Search Options

- `--limit N` / `-n N`: Max results (default 10)
- `--project PATH` / `-p PATH`: Filter by project path (substring)
- `--role user|assistant` / `-r`: Filter by message role
- `--after DATE`: Only results after this date
- `--before DATE`: Only results before this date
- `--session ID`: Search within a specific session
- `--group-by project|session`: Group results into a ranked table (great for finding which session to resume)
- `--json`: Output as JSON

### Grouped Search

```bash
# Find which project has the most relevant sessions
claude-recall search "auth bug" --limit 50 --group-by project

# Find which session to resume
claude-recall search "kubernetes deploy" --limit 30 --group-by session
```

## Daemon (Background Indexing)

```bash
claude-recall daemon start       # Start file watcher
claude-recall daemon stop        # Stop daemon
claude-recall daemon status      # Check status
claude-recall daemon enable      # Auto-start at login (launchd)
claude-recall daemon disable     # Remove auto-start
```

## Configuration

```bash
claude-recall config show
claude-recall config set embedding_model nomic-embed-text-v2-moe
claude-recall config set batch_size 64
```

## When to Use

- "What did we discuss about X last week?"
- "Find the session where I debugged the auth issue"
- "Search for conversations about terraform migrations"
- "How did I solve [specific problem] before?"

When the user invokes `/claude-recall <query>`, run a semantic search with the provided query using `$ARGUMENTS` as the search term. If `$ARGUMENTS` is empty, show the index status instead.
