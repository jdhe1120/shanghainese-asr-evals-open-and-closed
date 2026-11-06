"""Qwen3-Omni-30B-A3B-Instruct, stock, with the model card's own Chinese ASR prompt (请将这段中文语音转换为纯文本。),
run remotely on Modal (A100 80 GB). See modal_apps/qwen3_omni.py for versions, loading and cost. The Wu-prompted row
is qwen3_omni_wu.py: the gap between the two says whether naming the language helps.

Batch model: the runner sends every uncached clip in one call (one GPU start, one model load), via transcribe_batch.
Each call starts an ephemeral Modal app, so nothing stays deployed or running afterwards.
"""
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
REVISION = "26291f793822fb6be9555850f06dfe95f2d7e695"
PROMPT = "请将这段中文语音转换为纯文本。"
# Both prompt rows use the one Modal app; the runner starts hosted models in parallel threads, and two app.run() on the
# same app at once tear each other down. So they take turns.
_LOCK = threading.Lock()


def _remote():
    sys.path.insert(0, str(ROOT / "modal_apps"))
    import qwen3_omni as m
    return m


def run_batch(wav_paths: list[str], prompt: str) -> dict[str, str]:
    """wav path -> text for all of them, in one Modal call."""
    m = _remote()
    items = [(Path(p).stem, Path(p).read_bytes()) for p in wav_paths]
    with _LOCK, m.app.run():
        out = m.transcribe.remote(items, prompt)
    return {p: out[Path(p).stem] for p in wav_paths}


def transcribe_batch(wav_paths: list[str]) -> dict[str, str]:
    return run_batch(wav_paths, PROMPT)


def transcribe(wav_path: str) -> str:
    return transcribe_batch([wav_path])[wav_path]
