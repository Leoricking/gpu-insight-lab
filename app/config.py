"""
GPU Insight Lab - Application Configuration
Handles loading/saving config from ~/.gpu_insight_lab/config.json.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from app.branding import APP_SLUG, DATABASE_NAME, NATIVE_EXECUTABLE

logger = logging.getLogger(__name__)

_CONFIG_DIR = Path.home() / ".gpu_insight_lab"
_CONFIG_FILE = _CONFIG_DIR / "config.json"
_DEFAULT_OUTPUT_DIR = Path.home() / "gpu_insight_lab_output"
_DEFAULT_STORAGE_DIR = Path(__file__).parent.parent / "storage"
_DEFAULT_DATABASE_PATH = _DEFAULT_STORAGE_DIR / DATABASE_NAME


def _default_exe_name() -> str:
    """Return platform-correct executable name."""
    if sys.platform == "win32":
        return NATIVE_EXECUTABLE + ".exe"
    return NATIVE_EXECUTABLE


@dataclass
class AppConfig:
    """Application configuration dataclass."""

    output_dir: str = str(_DEFAULT_OUTPUT_DIR)
    database_path: str = str(_DEFAULT_DATABASE_PATH)
    timeout_seconds: int = 120
    default_repeat: int = 10
    native_executable_path: str = ""
    nsys_path: str = ""
    ncu_path: str = ""
    theme: str = "light"
    telemetry_enabled: bool = False
    log_level: str = "INFO"

    def output_path(self) -> Path:
        return Path(self.output_dir)

    def db_path(self) -> Path:
        return Path(self.database_path)

    def native_exe_path(self) -> Path | None:
        """Return resolved native executable path, or None if not configured/found."""
        if self.native_executable_path:
            p = Path(self.native_executable_path)
            if p.exists():
                return p
        # Auto-search bin/
        bin_dir = Path(__file__).parent.parent / "bin"
        exe_name = _default_exe_name()
        candidate = bin_dir / exe_name
        if candidate.exists():
            return candidate
        return None


def load_config() -> AppConfig:
    """Load config from disk; return defaults on any failure."""
    if not _CONFIG_FILE.exists():
        return AppConfig()
    try:
        with _CONFIG_FILE.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        cfg = AppConfig()
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        logger.warning("Failed to load config from %s: %s – using defaults", _CONFIG_FILE, exc)
        return AppConfig()


def save_config(cfg: AppConfig) -> bool:
    """Persist config to disk. Returns True on success."""
    try:
        _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with _CONFIG_FILE.open("w", encoding="utf-8") as fh:
            json.dump(asdict(cfg), fh, indent=2)
        return True
    except OSError as exc:
        logger.error("Failed to save config: %s", exc)
        return False


# Module-level singleton
_cfg: AppConfig | None = None


def get_config() -> AppConfig:
    """Return the global config, loading from disk on first call."""
    global _cfg
    if _cfg is None:
        _cfg = load_config()
    return _cfg


def reset_config() -> None:
    """Reset the in-memory config (used in tests)."""
    global _cfg
    _cfg = None
