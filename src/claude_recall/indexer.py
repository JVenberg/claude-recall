"""Session parsing, chunking, embedding, and indexing for claude-recall."""

import asyncio
import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import lancedb
import ollama
import pyarrow as pa
import tiktoken
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn

from claude_recall.config import Config, load_config

console = Console()

# Schema for the main vectors table
VECTORS_SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("vector", pa.list_(pa.float32(), -1)),  # dimension set at runtime
    pa.field("text", pa.string()),
    pa.field("session_id", pa.string()),
    pa.field("message_uuid", pa.string()),
    pa.field("role", pa.string()),
    pa.field("timestamp", pa.string()),
    pa.field("project_path", pa.string()),
    pa.field("file_path", pa.string()),
    pa.field("content_hash", pa.string()),
    pa.field("chunk_idx", pa.int32()),
])

# Schema for file metadata tracking (change detection)
FILE_META_SCHEMA = pa.schema([
    pa.field("file_path", pa.string()),
    pa.field("file_mtime", pa.float64()),
    pa.field("file_size", pa.int64()),
    pa.field("last_indexed_at", pa.float64()),
])


def content_hash(text: str) -> str:
    """SHA-256 hash of text content for deduplication."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def extract_text_content(message: dict) -> str:
    """Extract plain text from a message's content field.

    Handles both string content (user messages) and array-of-blocks
    content (assistant messages with text/tool_use/tool_result blocks).
    """
    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif block.get("type") == "tool_use":
                    # Include tool name and input for searchability
                    tool_name = block.get("name", "unknown_tool")
                    tool_input = block.get("input", {})
                    if isinstance(tool_input, dict):
                        input_str = json.dumps(tool_input, ensure_ascii=False)
                    else:
                        input_str = str(tool_input)
                    # Only include if reasonably sized
                    if len(input_str) < 2000:
                        parts.append(f"[Tool: {tool_name}] {input_str}")
                    else:
                        parts.append(f"[Tool: {tool_name}] (large input)")
                elif block.get("type") == "tool_result":
                    result_content = block.get("content", "")
                    if isinstance(result_content, str) and len(result_content) < 2000:
                        parts.append(f"[Tool result] {result_content}")
                    elif isinstance(result_content, list):
                        for sub in result_content:
                            if isinstance(sub, dict) and sub.get("type") == "text":
                                text = sub.get("text", "")
                                if len(text) < 2000:
                                    parts.append(f"[Tool result] {text}")
        return "\n".join(parts).strip()
    return ""


def chunk_text(text: str, max_tokens: int, overlap_tokens: int, enc) -> list[str]:
    """Split text into chunks respecting token limits with overlap."""
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return [text]

    chunks = []
    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chunk_tokens = tokens[start:end]
        chunks.append(enc.decode(chunk_tokens))
        start = end - overlap_tokens
        if start >= len(tokens):
            break
    return chunks


def parse_session_file(file_path: Path, config: Config) -> list[dict]:
    """Parse a JSONL session file into chunks ready for embedding.

    Returns a list of dicts with keys: id, text, session_id, message_uuid,
    role, timestamp, project_path, file_path, content_hash, chunk_idx.
    """
    enc = tiktoken.get_encoding("cl100k_base")
    chunks = []

    # Derive project path from parent directory name
    parent_dir = file_path.parent.name  # e.g., "-Users-username-Code-myproject"
    project_path = "/" + parent_dir.replace("-", "/")

    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            record_type = record.get("type")
            if record_type not in ("user", "assistant"):
                continue

            message = record.get("message", {})
            text = extract_text_content(message)
            if not text or len(text) < 10:  # Skip trivially short messages
                continue

            session_id = record.get("sessionId", file_path.stem)
            message_uuid = record.get("uuid", "")
            role = record_type
            timestamp = record.get("timestamp", "")
            cwd = record.get("cwd", project_path)

            # Chunk if needed
            text_chunks = chunk_text(
                text, config.max_chunk_tokens, config.chunk_overlap_tokens, enc
            )

            for idx, chunk_text_val in enumerate(text_chunks):
                chunk_hash = content_hash(chunk_text_val)
                chunk_id = f"{session_id}:{message_uuid}:{idx}"
                chunks.append({
                    "id": chunk_id,
                    "text": chunk_text_val,
                    "session_id": session_id,
                    "message_uuid": message_uuid,
                    "role": role,
                    "timestamp": str(timestamp),
                    "project_path": cwd,
                    "file_path": str(file_path),
                    "content_hash": chunk_hash,
                    "chunk_idx": idx,
                })

    return chunks


def discover_session_files(config: Config) -> list[Path]:
    """Find all JSONL session files in the sessions directory.

    Optionally skips agent sub-session files (agent-*) which are subagent
    conversations that duplicate content from parent sessions.
    """
    sessions_dir = config.sessions_path
    if not sessions_dir.exists():
        return []
    all_files = sorted(sessions_dir.rglob("*.jsonl"))
    if config.skip_agent_files:
        all_files = [f for f in all_files if not f.name.startswith("agent-")]
    return all_files


def get_file_meta(file_path: Path) -> dict:
    """Get current file metadata for change detection."""
    stat = file_path.stat()
    return {
        "file_path": str(file_path),
        "file_mtime": stat.st_mtime,
        "file_size": stat.st_size,
    }


async def _async_embed_batch(texts: list[str], config: Config) -> list[list[float]]:
    """Embed a single batch asynchronously."""
    client = ollama.AsyncClient()
    try:
        response = await client.embed(
            model=config.embedding_model,
            input=texts,
            truncate=True,
        )
        return response["embeddings"]
    except Exception:
        # Fall back to one-at-a-time
        results = []
        for text in texts:
            try:
                truncated = text[:24000] if len(text) > 24000 else text
                response = await client.embed(
                    model=config.embedding_model,
                    input=[truncated],
                    truncate=True,
                )
                results.append(response["embeddings"][0])
            except Exception:
                try:
                    response = await client.embed(
                        model=config.embedding_model,
                        input=[text[:4000]],
                        truncate=True,
                    )
                    results.append(response["embeddings"][0])
                except Exception:
                    results.append([0.0] * config.embedding_dimensions)
        return results


async def _async_embed_all(texts: list[str], config: Config, concurrency: int = 4) -> list[list[float]]:
    """Embed all texts with concurrent batch requests."""
    batch_size = config.batch_size
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]

    # Process batches with limited concurrency using a semaphore
    semaphore = asyncio.Semaphore(concurrency)
    results = [None] * len(batches)

    async def process_batch(idx, batch):
        async with semaphore:
            results[idx] = await _async_embed_batch(batch, config)

    await asyncio.gather(*[process_batch(i, b) for i, b in enumerate(batches)])

    # Flatten
    all_vectors = []
    for batch_result in results:
        all_vectors.extend(batch_result)
    return all_vectors


def embed_texts(texts: list[str], config: Config) -> list[list[float]]:
    """Embed texts using Ollama with async concurrency for speed.

    Runs multiple embedding batches in parallel using Ollama's async client.
    Falls back to sync one-at-a-time if a batch fails.
    """
    if not texts:
        return []
    try:
        return asyncio.run(_async_embed_all(texts, config))
    except RuntimeError:
        # Already in an async context; fall back to sync
        try:
            response = ollama.embed(
                model=config.embedding_model,
                input=texts,
                truncate=True,
            )
            return response["embeddings"]
        except Exception:
            # Last resort: one at a time, sync
            results = []
            for text in texts:
                try:
                    truncated = text[:24000] if len(text) > 24000 else text
                    response = ollama.embed(
                        model=config.embedding_model,
                        input=[truncated],
                        truncate=True,
                    )
                    results.append(response["embeddings"][0])
                except Exception:
                    results.append([0.0] * config.embedding_dimensions)
            return results


class Indexer:
    """Manages the LanceDB index for claude-recall."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or load_config()
        self.config.db_path.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(self.config.db_path))

    def _get_or_create_vectors_table(self):
        """Get existing vectors table or create it."""
        if "vectors" in self.db.table_names():
            return self.db.open_table("vectors")
        return None

    def _get_or_create_file_meta_table(self):
        """Get existing file_meta table or create it."""
        if "file_meta" in self.db.table_names():
            return self.db.open_table("file_meta")
        return None

    def _get_indexed_file_meta(self) -> dict[str, dict]:
        """Load file metadata from LanceDB for change detection."""
        table = self._get_or_create_file_meta_table()
        if table is None:
            return {}
        arrow_table = table.to_arrow()
        result = {}
        for i in range(arrow_table.num_rows):
            fp = arrow_table.column("file_path")[i].as_py()
            result[fp] = {
                "file_mtime": arrow_table.column("file_mtime")[i].as_py(),
                "file_size": arrow_table.column("file_size")[i].as_py(),
                "last_indexed_at": arrow_table.column("last_indexed_at")[i].as_py(),
            }
        return result

    def _get_existing_hashes(self, file_path: str) -> set[str]:
        """Get content hashes of existing vectors for a file."""
        table = self._get_or_create_vectors_table()
        if table is None:
            return set()
        try:
            arrow_table = table.search().where(
                f"file_path = '{file_path}'", prefilter=True
            ).select(["content_hash"]).limit(100000).to_arrow()
            return set(arrow_table.column("content_hash").to_pylist())
        except Exception:
            return set()

    def _delete_vectors_for_file(self, file_path: str):
        """Remove all vectors associated with a file."""
        table = self._get_or_create_vectors_table()
        if table is not None:
            try:
                table.delete(f"file_path = '{file_path}'")
            except Exception:
                pass

    def _update_file_meta(self, file_path: str, mtime: float, size: int):
        """Update file metadata after indexing."""
        meta_table = self._get_or_create_file_meta_table()
        record = {
            "file_path": file_path,
            "file_mtime": mtime,
            "file_size": size,
            "last_indexed_at": time.time(),
        }
        if meta_table is not None:
            try:
                meta_table.delete(f"file_path = '{file_path}'")
            except Exception:
                pass
            meta_table.add([record])
        else:
            self.db.create_table("file_meta", [record], schema=FILE_META_SCHEMA)

    def files_needing_index(self, force: bool = False) -> list[Path]:
        """Determine which session files need (re)indexing."""
        all_files = discover_session_files(self.config)
        if force:
            return all_files

        indexed_meta = self._get_indexed_file_meta()
        needs_index = []
        for fpath in all_files:
            fp_str = str(fpath)
            if fp_str not in indexed_meta:
                needs_index.append(fpath)
                continue
            meta = indexed_meta[fp_str]
            stat = fpath.stat()
            if stat.st_mtime != meta["file_mtime"] or stat.st_size != meta["file_size"]:
                needs_index.append(fpath)

        return needs_index

    def index_file(self, file_path: Path) -> int:
        """Index a single session file. Returns number of new vectors added."""
        chunks = parse_session_file(file_path, self.config)
        if not chunks:
            # Update meta even if no chunks (e.g., empty file)
            meta = get_file_meta(file_path)
            self._update_file_meta(str(file_path), meta["file_mtime"], meta["file_size"])
            return 0

        # Check which chunks are already indexed (by content hash)
        existing_hashes = self._get_existing_hashes(str(file_path))
        new_chunks = [c for c in chunks if c["content_hash"] not in existing_hashes]

        # Delete old vectors for this file if there are changes
        if existing_hashes and new_chunks:
            self._delete_vectors_for_file(str(file_path))
            # Re-embed all chunks for this file (simpler than partial update)
            new_chunks = chunks

        if not new_chunks:
            # File unchanged at content level, update mtime
            meta = get_file_meta(file_path)
            self._update_file_meta(str(file_path), meta["file_mtime"], meta["file_size"])
            return 0

        # Embed with async concurrency (batching handled internally)
        texts = [c["text"] for c in new_chunks]
        all_vectors = embed_texts(texts, self.config)

        # Add vectors to chunks
        records = []
        for chunk, vector in zip(new_chunks, all_vectors):
            chunk["vector"] = vector
            records.append(chunk)

        # Write to LanceDB
        vectors_table = self._get_or_create_vectors_table()
        if vectors_table is not None:
            vectors_table.add(records)
        else:
            self.db.create_table("vectors", records)

        # Update file metadata
        meta = get_file_meta(file_path)
        self._update_file_meta(str(file_path), meta["file_mtime"], meta["file_size"])

        return len(records)

    def _write_progress(self, current: int, total: int, vectors: int,
                        current_file: str = "", errors: int = 0,
                        start_time: Optional[float] = None):
        """Write progress state to a file so other processes can read it."""
        progress_file = self.config.config_dir / "index_progress.json"
        elapsed = time.time() - start_time if start_time else 0
        rate = current / elapsed if elapsed > 0 else 0
        eta = (total - current) / rate if rate > 0 else 0

        state = {
            "status": "indexing",
            "current": current,
            "total": total,
            "vectors_added": vectors,
            "errors": errors,
            "current_file": current_file,
            "percent": round(current / total * 100, 1) if total else 0,
            "elapsed_seconds": round(elapsed, 1),
            "eta_seconds": round(eta, 1),
            "started_at": start_time,
        }
        try:
            progress_file.write_text(json.dumps(state))
        except Exception:
            pass

    def _clear_progress(self):
        """Remove the progress file when done."""
        progress_file = self.config.config_dir / "index_progress.json"
        progress_file.unlink(missing_ok=True)

    def get_progress(self) -> Optional[dict]:
        """Read current indexing progress (if running)."""
        progress_file = self.config.config_dir / "index_progress.json"
        if not progress_file.exists():
            return None
        try:
            data = json.loads(progress_file.read_text())
            # Check if stale (no update in 5 minutes)
            if data.get("started_at"):
                elapsed = time.time() - data["started_at"]
                if elapsed > data.get("elapsed_seconds", 0) + 300:
                    self._clear_progress()
                    return None
            return data
        except Exception:
            return None

    def index_all(self, force: bool = False) -> dict:
        """Index all session files. Returns stats dict."""
        files = self.files_needing_index(force=force)
        all_files = discover_session_files(self.config)

        stats = {
            "total_files": len(all_files),
            "files_to_index": len(files),
            "files_skipped": len(all_files) - len(files),
            "vectors_added": 0,
            "errors": [],
        }

        if not files:
            console.print("[dim]All files up to date, nothing to index.[/dim]")
            return stats

        console.print(
            f"Indexing {len(files)} file(s) "
            f"(skipping {stats['files_skipped']} unchanged)..."
        )

        start_time = time.time()
        is_tty = console.is_terminal

        if is_tty:
            progress_ctx = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                console=console,
            )
        else:
            progress_ctx = None

        try:
            if progress_ctx:
                progress_ctx.start()
                task = progress_ctx.add_task("Indexing sessions", total=len(files))

            for i, fpath in enumerate(files):
                if progress_ctx:
                    progress_ctx.update(task, description=f"[cyan]{fpath.name}[/cyan]")

                # Write progress file for external status checks
                self._write_progress(
                    current=i, total=len(files),
                    vectors=stats["vectors_added"],
                    current_file=fpath.name,
                    errors=len(stats["errors"]),
                    start_time=start_time,
                )

                # Plain text progress every 10 files when not on a TTY
                if not is_tty and i % 10 == 0 and i > 0:
                    elapsed = time.time() - start_time
                    rate = i / elapsed
                    eta = (len(files) - i) / rate if rate > 0 else 0
                    print(
                        f"  [{i}/{len(files)}] "
                        f"{i/len(files)*100:.0f}% "
                        f"({stats['vectors_added']} vectors, "
                        f"ETA {eta:.0f}s)"
                    )

                try:
                    n = self.index_file(fpath)
                    stats["vectors_added"] += n
                except Exception as e:
                    stats["errors"].append({"file": str(fpath), "error": str(e)})
                    if is_tty:
                        console.print(f"[red]Error indexing {fpath.name}: {e}[/red]")
                    else:
                        print(f"  ERROR: {fpath.name}: {e}")

                if progress_ctx:
                    progress_ctx.advance(task)
        finally:
            if progress_ctx:
                progress_ctx.stop()
            # Write final progress then clean up
            self._write_progress(
                current=len(files), total=len(files),
                vectors=stats["vectors_added"],
                errors=len(stats["errors"]),
                start_time=start_time,
            )
            self._clear_progress()

        elapsed = time.time() - start_time
        console.print(
            f"[green]Done.[/green] "
            f"Indexed {len(files)} file(s), "
            f"{stats['vectors_added']} vectors added "
            f"in {elapsed:.0f}s."
        )
        if stats["errors"]:
            console.print(f"[yellow]{len(stats['errors'])} error(s) occurred.[/yellow]")

        return stats

    def get_status(self) -> dict:
        """Get index status information, including live progress if running."""
        all_files = discover_session_files(self.config)
        indexed_meta = self._get_indexed_file_meta()

        vectors_table = self._get_or_create_vectors_table()
        vector_count = 0
        if vectors_table is not None:
            vector_count = vectors_table.count_rows()

        status = {
            "total_session_files": len(all_files),
            "indexed_files": len(indexed_meta),
            "total_vectors": vector_count,
            "db_path": str(self.config.db_path),
            "embedding_model": self.config.embedding_model,
        }

        # Check for live indexing progress
        progress = self.get_progress()
        if progress:
            status["indexing_in_progress"] = True
            status["indexing_current"] = progress["current"]
            status["indexing_total"] = progress["total"]
            status["indexing_percent"] = progress["percent"]
            status["indexing_eta_seconds"] = progress["eta_seconds"]
            status["indexing_vectors_so_far"] = progress["vectors_added"]
            status["indexing_current_file"] = progress["current_file"]

        return status

    def delete_file_index(self, file_path: str):
        """Remove all index data for a specific file."""
        self._delete_vectors_for_file(file_path)
        meta_table = self._get_or_create_file_meta_table()
        if meta_table is not None:
            try:
                meta_table.delete(f"file_path = '{file_path}'")
            except Exception:
                pass

    def create_search_indexes(self):
        """Create vector and FTS indexes for search performance."""
        vectors_table = self._get_or_create_vectors_table()
        if vectors_table is None:
            console.print("[yellow]No vectors table found. Run 'index' first.[/yellow]")
            return

        row_count = vectors_table.count_rows()

        with console.status("[bold cyan]Building full-text search index...") as status:
            try:
                vectors_table.create_fts_index("text", replace=True)
                console.print("[green]Full-text search index created.[/green]")
            except Exception as e:
                console.print(f"[yellow]FTS index: {e}[/yellow]")

        # Only create vector index if we have enough data (IVF needs clusters)
        if row_count >= 256:
            with console.status(
                f"[bold cyan]Building vector index ({row_count:,} vectors)... this may take a minute"
            ) as status:
                try:
                    vectors_table.create_index(
                        metric="cosine",
                        num_partitions=min(row_count // 50, 256),
                        num_sub_vectors=min(self.config.embedding_dimensions // 16, 96),
                        replace=True,
                    )
                    console.print("[green]Vector index created.[/green]")
                except Exception as e:
                    console.print(f"[yellow]Vector index: {e}[/yellow]")
        else:
            console.print(
                f"[dim]Skipping vector index ({row_count} rows, need >= 256 for IVF).[/dim]"
            )
