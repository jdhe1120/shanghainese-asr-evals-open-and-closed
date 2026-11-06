"""Step-Audio2-Wu-ASR (best released Wu model: 12.85% CER on WS-Wu-Bench, 8.68% on the paper's in-house dialogue set):
stepfun-ai/Step-Audio-2-mini + the WenetSpeech-Wu ASR LoRA, run remotely on Modal (A100 40 GB). See
modal_apps/step_audio2_wu.py for versions, prompt and cost.

Batch model: the runner sends every uncached clip in one call (one GPU start, one model load), via transcribe_batch.
Each call starts an ephemeral Modal app, so nothing stays deployed or running afterwards.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def _remote():
    sys.path.insert(0, str(ROOT / "modal_apps"))
    import step_audio2_wu as m
    return m


def transcribe_batch(wav_paths: list[str]) -> dict[str, str]:
    """wav path -> text for all of them, in one Modal call."""
    m = _remote()
    items = [(Path(p).stem, Path(p).read_bytes()) for p in wav_paths]
    with m.app.run():
        out = m.transcribe.remote(items)
    return {p: out[Path(p).stem] for p in wav_paths}


def transcribe(wav_path: str) -> str:
    return transcribe_batch([wav_path])[wav_path]
