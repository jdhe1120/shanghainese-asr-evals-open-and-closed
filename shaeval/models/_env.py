"""Reads the repo's .env (gitignored). Values are never printed."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def env() -> dict[str, str]:
    out = {}
    path = ROOT / ".env"
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
    return out
