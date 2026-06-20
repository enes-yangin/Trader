"""Minimal, dependency-free .env loader.

Reads KEY=VALUE lines from a .env file at the project root into os.environ so
each user can keep their own secrets (HF_TOKEN, API keys, exchange choice)
locally without committing them. A real environment variable always wins over
the .env value (same precedence rule as python-dotenv's default).
"""
import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: str | None = None) -> bool:
    """Populate os.environ from `path` (default: <project root>/.env).

    Returns True if a file was found and read. Existing env vars are NOT
    overridden. Lines that are blank, comments (#...), or lack '=' are skipped;
    surrounding quotes on the value are stripped.
    """
    p = Path(path) if path else _ROOT / ".env"
    if not p.exists():
        return False
    try:
        text = p.read_text(encoding="utf-8")
    except OSError:
        return False
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val
    return True
