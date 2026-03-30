"""Configuration management for claude-recall."""

import json
from pathlib import Path
from dataclasses import dataclass, field, asdict


DEFAULT_CONFIG_DIR = Path.home() / ".claude" / "claude-recall"
DEFAULT_DB_DIR = DEFAULT_CONFIG_DIR / "db"
DEFAULT_SESSIONS_DIR = Path.home() / ".claude" / "projects"
DEFAULT_CONFIG_FILE = DEFAULT_CONFIG_DIR / "config.json"
DEFAULT_LOG_FILE = DEFAULT_CONFIG_DIR / "daemon.log"
DEFAULT_PID_FILE = DEFAULT_CONFIG_DIR / "daemon.pid"

DEFAULTS = {
    "embedding_model": "nomic-embed-text",
    "embedding_dimensions": 768,
    "sessions_dir": str(DEFAULT_SESSIONS_DIR),
    "db_dir": str(DEFAULT_DB_DIR),
    "batch_size": 128,
    "max_chunk_tokens": 6000,
    "chunk_overlap_tokens": 512,
    "debounce_seconds": 600,
    "skip_agent_files": True,
}


@dataclass
class Config:
    embedding_model: str = DEFAULTS["embedding_model"]
    embedding_dimensions: int = DEFAULTS["embedding_dimensions"]
    sessions_dir: str = str(DEFAULT_SESSIONS_DIR)
    db_dir: str = str(DEFAULT_DB_DIR)
    batch_size: int = DEFAULTS["batch_size"]
    max_chunk_tokens: int = DEFAULTS["max_chunk_tokens"]
    chunk_overlap_tokens: int = DEFAULTS["chunk_overlap_tokens"]
    debounce_seconds: int = DEFAULTS["debounce_seconds"]
    skip_agent_files: bool = DEFAULTS["skip_agent_files"]

    @property
    def sessions_path(self) -> Path:
        return Path(self.sessions_dir).expanduser()

    @property
    def db_path(self) -> Path:
        return Path(self.db_dir).expanduser()

    @property
    def config_dir(self) -> Path:
        return DEFAULT_CONFIG_DIR

    @property
    def log_file(self) -> Path:
        return DEFAULT_LOG_FILE

    @property
    def pid_file(self) -> Path:
        return DEFAULT_PID_FILE


def load_config() -> Config:
    """Load config from disk, falling back to defaults."""
    if DEFAULT_CONFIG_FILE.exists():
        with open(DEFAULT_CONFIG_FILE) as f:
            data = json.load(f)
        # Merge with defaults for any missing keys
        merged = {**DEFAULTS, **data}
        return Config(**{k: v for k, v in merged.items() if k in DEFAULTS})
    return Config()


def save_config(config: Config) -> None:
    """Save config to disk."""
    DEFAULT_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(DEFAULT_CONFIG_FILE, "w") as f:
        json.dump(asdict(config), f, indent=2)


def set_config_value(key: str, value: str) -> Config:
    """Set a single config value, coercing types as needed."""
    config = load_config()
    if key not in DEFAULTS:
        raise KeyError(f"Unknown config key: {key}. Valid keys: {', '.join(DEFAULTS.keys())}")

    # Coerce to the correct type
    expected_type = type(DEFAULTS[key])
    if expected_type == bool:
        value = value.lower() in ("true", "1", "yes")
    elif expected_type == int:
        value = int(value)
    elif expected_type == float:
        value = float(value)

    setattr(config, key, value)
    save_config(config)
    return config
