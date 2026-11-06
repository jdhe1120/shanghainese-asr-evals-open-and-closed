"""Qwen3-Omni-30B-A3B-Instruct, stock weights, prompted for Shanghainese: 请将这段上海话录音逐字转写成上海话口语。
(the Cantonese evals' GLM-ASR Cantonese prompt with the language swapped). Same Modal app and weights as qwen3_omni.py.
"""
from .qwen3_omni import REVISION, run_batch  # noqa: F401  (REVISION: recorded in the run like every HF model)

PROMPT = "请将这段上海话录音逐字转写成上海话口语。"


def transcribe_batch(wav_paths: list[str]) -> dict[str, str]:
    return run_batch(wav_paths, PROMPT)


def transcribe(wav_path: str) -> str:
    return transcribe_batch([wav_path])[wav_path]
