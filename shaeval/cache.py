"""One JSON file per (model, clip). Keyed on the clip digest so edited audio or
text is never served a stale hypothesis; paid APIs are never called twice."""
import json
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"


def _path(model: str, clip_id: str, digest: str) -> Path:
    return CACHE_DIR / model / f"{clip_id}.{digest}.json"


def get(model: str, clip_id: str, digest: str) -> dict | None:
    p = _path(model, clip_id, digest)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def put(model: str, clip_id: str, digest: str, entry: dict) -> None:
    p = _path(model, clip_id, digest)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")
