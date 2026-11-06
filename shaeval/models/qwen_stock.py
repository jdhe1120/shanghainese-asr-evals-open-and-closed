"""Stock Qwen/Qwen3-ASR-1.7B-hf baseline (~4 GB on first use). Lists Wu among its 22 Chinese dialects; auto language."""
from ._qwen import transcribe as _transcribe

MODEL_ID = "Qwen/Qwen3-ASR-1.7B-hf"
REVISION = "bcd2b5b7f32b480ab5790554cfa8347f246a14f3"  # pinned: a new upload never mixes into old results


def transcribe(wav_path: str) -> str:
    return _transcribe(MODEL_ID, wav_path, REVISION)
