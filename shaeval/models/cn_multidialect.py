"""ASLP-lab/CN-MultiDialect-ASR (Aug 2026): Qwen3-ASR-1.7B after continual pre-training, dialect SFT and on-policy
self-distillation, from the WenetSpeech-Wu authors' lab. Auto language, as on the model card.

The checkpoint is in the original Qwen3-ASR format, which needs the `qwen-asr` package (pins transformers 4.57.6), so
it runs in an isolated worker (_qwen_asr_worker.py). The first start builds that env and downloads ~4.7 GB.
"""
from pathlib import Path

from ._worker import Worker

MODEL_ID = "ASLP-lab/CN-MultiDialect-ASR"
REVISION = "bea6351ed81a5eabf6e1345dfa9f22fa2bf77f76"  # pinned; the worker gets the downloaded snapshot's path
_worker = None


def transcribe(wav_path: str) -> str:
    global _worker
    if _worker is None:
        from huggingface_hub import snapshot_download
        path = snapshot_download(MODEL_ID, revision=REVISION)
        _worker = Worker("cn-multidialect", [
            "uv", "run", "--isolated", "--no-project", "--python", "3.12", "--with", "qwen-asr==0.0.6", "--with", "torch",
            "python", "-I", str(Path(__file__).with_name("_qwen_asr_worker.py")), path])
    return _worker.transcribe(wav_path)
