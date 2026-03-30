# claude-recall

Local semantic search over your Claude Code session history. Find past conversations, recall how you solved a problem, and get copy-pasteable resume commands to pick up where you left off.

Uses [Ollama](https://ollama.com) for local embeddings and [LanceDB](https://lancedb.github.io/lancedb/) for vector storage. Everything runs on your machine, nothing leaves your device.

## Features

- **Semantic search**: natural language queries like "how did I fix the auth bug"
- **Exact search**: full-text keyword matching
- **Fuzzy search**: tolerates typos via rapidfuzz
- **Hybrid search**: combined semantic + keyword with reranking
- **Group by project/session**: ranked table showing which project or session is most relevant
- **Resume commands**: copy-pasteable `cd ~/project && claude -r <session-id>` for every result
- **Background daemon**: auto-reindexes sessions as they change (FSEvents, zero CPU when idle)
- **Incremental indexing**: hash-based change detection skips already-indexed content
- **Claude Code skill**: auto-invoked when you ask about past sessions

## Prerequisites

- macOS with Apple Silicon (M1+)
- [Ollama](https://ollama.com) installed and running
- [uv](https://docs.astral.sh/uv/) for Python dependency management
- Python 3.11+

## Installation

```bash
# 1. Clone into your Claude Code skills directory
git clone https://github.com/JVenberg/claude-recall.git ~/.claude/skills/claude-recall

# 2. Install dependencies
cd ~/.claude/skills/claude-recall/scripts
uv sync

# 3. Pull the embedding model
ollama pull nomic-embed-text

# 4. Index your sessions (first run takes a few minutes)
uv run python cli.py index
```

The Claude Code skill is auto-discovered. You can use `/claude-recall <query>` or just ask Claude about past sessions.

## Usage

### Search

```bash
# Semantic search (default, best for natural language)
uv run --directory ~/.claude/skills/claude-recall/scripts python cli.py search "how did I fix the auth bug"

# Exact text match
uv run --directory ~/.claude/skills/claude-recall/scripts python cli.py search --exact "kubectl exec prod"

# Fuzzy search (tolerates typos)
uv run --directory ~/.claude/skills/claude-recall/scripts python cli.py search --fuzzy "kubernetis deploy"

# Hybrid (semantic + keyword)
uv run --directory ~/.claude/skills/claude-recall/scripts python cli.py search --hybrid "terraform state migration"
```

### Group results to find the right session

```bash
# Which project has the most relevant sessions?
uv run --directory ~/.claude/skills/claude-recall/scripts python cli.py search "splunk CLI" --limit 30 --group-by project

# Which session should I resume?
uv run --directory ~/.claude/skills/claude-recall/scripts python cli.py search "auth bug fix" --limit 20 --group-by session
```

Output includes a ranked table and copy-pasteable resume commands:

```
Resume commands:
  cd ~/Code/my-project && claude -r a1b2c3d4-e5f6-7890-abcd-ef1234567890
  cd ~/Code/other-project && claude -r f9e8d7c6-b5a4-3210-fedc-ba0987654321
```

### Search options

| Option | Description |
|--------|-------------|
| `--limit N` / `-n N` | Max results (default 10) |
| `--project PATH` / `-p` | Filter by project path (substring) |
| `--role user\|assistant` / `-r` | Filter by message role |
| `--after DATE` | Results after this date (ISO format) |
| `--before DATE` | Results before this date |
| `--session ID` | Search within a specific session |
| `--group-by project\|session` | Group results into a ranked table |
| `--json` | Output as JSON |

### Indexing

```bash
# Incremental index (only changed files)
uv run --directory ~/.claude/skills/claude-recall/scripts python cli.py index

# Force full reindex
uv run --directory ~/.claude/skills/claude-recall/scripts python cli.py index --force

# Check index status (shows live progress if indexing is running)
uv run --directory ~/.claude/skills/claude-recall/scripts python cli.py index --status

# View stats
uv run --directory ~/.claude/skills/claude-recall/scripts python cli.py stats
```

### Background daemon

The daemon watches `~/.claude/projects/` for session file changes and auto-reindexes with a 5-second debounce. Uses macOS FSEvents (kernel-level, zero CPU when idle).

```bash
# Start/stop
uv run --directory ~/.claude/skills/claude-recall/scripts python cli.py daemon start
uv run --directory ~/.claude/skills/claude-recall/scripts python cli.py daemon stop
uv run --directory ~/.claude/skills/claude-recall/scripts python cli.py daemon status

# Auto-start at login (installs a launchd LaunchAgent)
uv run --directory ~/.claude/skills/claude-recall/scripts python cli.py daemon enable
uv run --directory ~/.claude/skills/claude-recall/scripts python cli.py daemon disable
```

### Configuration

```bash
# Show config
uv run --directory ~/.claude/skills/claude-recall/scripts python cli.py config show

# Change embedding model
uv run --directory ~/.claude/skills/claude-recall/scripts python cli.py config set embedding_model nomic-embed-text-v2-moe

# Change batch size
uv run --directory ~/.claude/skills/claude-recall/scripts python cli.py config set batch_size 64

# Include agent sub-session files (skipped by default)
uv run --directory ~/.claude/skills/claude-recall/scripts python cli.py config set skip_agent_files false
```

Config is stored in `~/.claude/claude-recall/config.json`.

## How it works

1. **Parses** JSONL session files from `~/.claude/projects/`
2. **Chunks** each message (with overflow splitting for long messages)
3. **Embeds** chunks using Ollama (nomic-embed-text v1.5, 768 dimensions, 8K context)
4. **Stores** vectors + metadata in LanceDB at `~/.claude/claude-recall/db/`
5. **Tracks** file mtimes and content hashes to skip unchanged content on reindex
6. **Searches** via vector similarity (semantic), full-text index (exact), rapidfuzz (fuzzy), or a combination (hybrid)

Agent sub-session files (`agent-*`) are skipped by default since they duplicate content from parent sessions. This reduces the file count by ~85%.

## Architecture

```
~/.claude/skills/claude-recall/
  SKILL.md              # Claude Code skill definition
  scripts/
    cli.py              # Click CLI
    indexer.py           # Parser, chunker, embedder, LanceDB writer
    searcher.py          # Search modes + result display
    daemon.py            # File watcher + launchd management
    config.py            # Configuration
    pyproject.toml       # Dependencies

~/.claude/claude-recall/
    db/                  # LanceDB vector store (created on first index)
    config.json          # User configuration
    daemon.log           # Daemon logs
    daemon.pid           # Daemon PID file
```

## Supported embedding models

Any Ollama embedding model works. Tested with:

| Model | Params | Dimensions | Context | Notes |
|-------|--------|-----------|---------|-------|
| `nomic-embed-text` (default) | 137M | 768 | 8K | Best balance of quality, speed, and context length |
| `nomic-embed-text-v2-moe` | 475M | 768 | 512 | Higher quality, shorter context |
| `mxbai-embed-large` | 335M | 1024 | 512 | Strong retrieval scores |
| `snowflake-arctic-embed` | 22M-335M | varies | varies | Multiple size variants |

## License

MIT
