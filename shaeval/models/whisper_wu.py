"""Whisper-medium-Wu from the WenetSpeech-Wu paper (ASLP@NPU, arXiv 2601.11027; 14.33% CER on WS-Wu-Bench): Whisper
medium fine-tuned on WenetSpeech-Wu, WeNet checkpoint. The same model shanghainese-transcription serves.

Weights: `whisper/whisper-medium.pt` of ASLP-lab/WenetSpeech-Wu-Speech-Understanding (sha256 e2755246...), fetched
and fixed by scripts/fetch_models.py into models/whisper-medium-wu/whisper. WU_WHISPER_DIR in .env overrides.
"""
from pathlib import Path

from ._wenet import MODELS, transcriber

transcribe = transcriber("whisper-wu", "WU_WHISPER_DIR", MODELS / "whisper-medium-wu" / "whisper")
