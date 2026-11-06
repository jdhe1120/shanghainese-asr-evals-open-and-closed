"""Conformer-U2pp-Wu from the WenetSpeech-Wu paper (ASLP@NPU; 15.14% CER on WS-Wu-Bench, 15.20% in-house dialogue):
a ~130M-param Conformer (U2++, bidirectional decoder) trained from scratch on WenetSpeech-Wu with WeNet. Default decode.

Weights: `u2++/` of ASLP-lab/WenetSpeech-Wu-Speech-Understanding (u2++.pt sha256 9afd6fd8...), fetched by
scripts/fetch_models.py into models/conformer-u2pp-wu: u2++.pt renamed final.pt, dict/lang_char.txt copied to units.txt, and train.yaml given the
input_dim: 80 / output_dim: 5949 that WeNet's train.py normally writes, with fbank dither set to 0 (training noise; it
made inference random). U2PP_WU_DIR in .env overrides.
"""
from pathlib import Path

from ._wenet import MODELS, transcriber

transcribe = transcriber("u2pp-wu", "U2PP_WU_DIR", MODELS / "conformer-u2pp-wu")
