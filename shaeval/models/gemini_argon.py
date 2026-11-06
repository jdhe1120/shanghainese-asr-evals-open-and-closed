"""Gemini 4 Argon (Google's general multimodal model) prompted to transcribe, over generateContent.

Same setup as the Cantonese evals' gemini_argon.py (2026-11-03): audio inline with a text prompt, default sampling. The
prompt is theirs word for word, with Cantonese -> Shanghainese, its five example words -> everyday Shanghainese ones in
the canonical spelling (rules/wu_canonical.toml), and traditional -> simplified (our references are simplified).
Google names no speech languages for this model, Wu or otherwise.
"""
from . import _gemini as gc

MODEL = "gemini-4-argon"
NAME = "gemini_argon"
PROMPT = ("Transcribe this Shanghainese audio verbatim as spoken Shanghainese (上海话口语, e.g. 阿拉 侬 伊 搿 勿), "
          "not standard written Chinese, in simplified characters. Output only the transcript.")
QuotaExhausted, gap_s = gc.QuotaExhausted, gc.gap_s


def build_request(wav_path: str) -> dict:
    return {"contents": [{"role": "user", "parts": [{"text": PROMPT}, gc.inline_wav(wav_path)]}]}


def parse(text: str) -> str:
    return text.strip()


pace, transcribe = gc.make(__import__(__name__, fromlist=["*"]))
