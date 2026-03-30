"""Background file watcher daemon for claude-recall."""

import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from threading import Timer
from typing import Optional

from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent, FileDeletedEvent
from watchdog.observers import Observer

from config import Config, load_config

PLIST_NAME = "com.user.claude-recall"
PLIST_PATH = Path.home() / "Library" / "LaunchAgents" / f"{PLIST_NAME}.plist"
SCRIPTS_DIR = Path(__file__).parent


class DebounceHandler(FileSystemEventHandler):
    """Handles file events with debouncing to avoid redundant reindexing."""

    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self._timers: dict[str, Timer] = {}

    def _debounced_reindex(self, file_path: str):
        """Reindex a file after debounce period."""
        # Cancel any pending timer for this file
        if file_path in self._timers:
            self._timers[file_path].cancel()

        def do_reindex():
            try:
                from indexer import Indexer
                indexer = Indexer(self.config)
                n = indexer.index_file(Path(file_path))
                self.logger.info(f"Reindexed {file_path}: {n} vectors")
            except Exception as e:
                self.logger.error(f"Error reindexing {file_path}: {e}")
            finally:
                self._timers.pop(file_path, None)

        timer = Timer(self.config.debounce_seconds, do_reindex)
        self._timers[file_path] = timer
        timer.start()

    def on_modified(self, event):
        if event.is_directory or not event.src_path.endswith(".jsonl"):
            return
        self.logger.debug(f"Modified: {event.src_path}")
        self._debounced_reindex(event.src_path)

    def on_created(self, event):
        if event.is_directory or not event.src_path.endswith(".jsonl"):
            return
        self.logger.info(f"New session: {event.src_path}")
        self._debounced_reindex(event.src_path)

    def on_deleted(self, event):
        if event.is_directory or not event.src_path.endswith(".jsonl"):
            return
        self.logger.info(f"Deleted: {event.src_path}")
        try:
            from indexer import Indexer
            indexer = Indexer(self.config)
            indexer.delete_file_index(event.src_path)
            self.logger.info(f"Removed index for {event.src_path}")
        except Exception as e:
            self.logger.error(f"Error removing index for {event.src_path}: {e}")

    def cancel_all(self):
        """Cancel all pending timers."""
        for timer in self._timers.values():
            timer.cancel()
        self._timers.clear()


def setup_logger(config: Config) -> logging.Logger:
    """Set up file + console logger."""
    logger = logging.getLogger("claude-recall-daemon")
    logger.setLevel(logging.INFO)

    # File handler
    config.config_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(config.log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)

    return logger


class DaemonManager:
    """Manages the claude-recall daemon lifecycle."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or load_config()

    def get_pid(self) -> Optional[int]:
        """Get the PID of the running daemon, or None."""
        pid_file = self.config.pid_file
        if not pid_file.exists():
            return None
        try:
            pid = int(pid_file.read_text().strip())
            # Check if process is actually running
            os.kill(pid, 0)
            return pid
        except (ValueError, ProcessLookupError, PermissionError):
            pid_file.unlink(missing_ok=True)
            return None

    def is_running(self) -> bool:
        return self.get_pid() is not None

    def is_enabled(self) -> bool:
        """Check if the LaunchAgent is installed."""
        return PLIST_PATH.exists()

    def run_foreground(self):
        """Run the daemon in the foreground (blocking)."""
        logger = setup_logger(self.config)
        logger.info(f"Starting daemon, watching {self.config.sessions_path}")

        handler = DebounceHandler(self.config, logger)
        observer = Observer()
        observer.schedule(handler, str(self.config.sessions_path), recursive=True)

        # Write PID
        self.config.pid_file.parent.mkdir(parents=True, exist_ok=True)
        self.config.pid_file.write_text(str(os.getpid()))

        def shutdown(signum, frame):
            logger.info("Shutting down daemon...")
            handler.cancel_all()
            observer.stop()
            self.config.pid_file.unlink(missing_ok=True)
            sys.exit(0)

        signal.signal(signal.SIGTERM, shutdown)
        signal.signal(signal.SIGINT, shutdown)

        observer.start()
        logger.info("Daemon started, watching for session changes...")

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            shutdown(None, None)

    def start_background(self):
        """Start the daemon as a background process."""
        if self.is_running():
            return

        # Launch as a detached subprocess
        proc = subprocess.Popen(
            [
                sys.executable, "-c",
                "from daemon import DaemonManager; DaemonManager().run_foreground()"
            ],
            cwd=str(SCRIPTS_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        # Give it a moment to start
        time.sleep(0.5)

    def stop(self):
        """Stop the running daemon."""
        pid = self.get_pid()
        if pid is None:
            return
        try:
            os.kill(pid, signal.SIGTERM)
            # Wait for it to exit
            for _ in range(10):
                try:
                    os.kill(pid, 0)
                    time.sleep(0.2)
                except ProcessLookupError:
                    break
        except ProcessLookupError:
            pass
        self.config.pid_file.unlink(missing_ok=True)

    def enable(self):
        """Install the launchd LaunchAgent."""
        # Find uv path
        uv_path = subprocess.run(
            ["which", "uv"], capture_output=True, text=True
        ).stdout.strip() or "/opt/homebrew/bin/uv"

        plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{PLIST_NAME}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{uv_path}</string>
        <string>run</string>
        <string>--directory</string>
        <string>{SCRIPTS_DIR}</string>
        <string>python</string>
        <string>cli.py</string>
        <string>daemon</string>
        <string>start</string>
        <string>--foreground</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>{self.config.log_file}.stdout</string>
    <key>StandardErrorPath</key>
    <string>{self.config.log_file}.stderr</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
    </dict>
</dict>
</plist>
"""
        PLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
        PLIST_PATH.write_text(plist_content)

        # Load the agent
        subprocess.run(["launchctl", "load", str(PLIST_PATH)], check=False)

    def disable(self):
        """Remove the launchd LaunchAgent."""
        if PLIST_PATH.exists():
            subprocess.run(["launchctl", "unload", str(PLIST_PATH)], check=False)
            PLIST_PATH.unlink(missing_ok=True)
