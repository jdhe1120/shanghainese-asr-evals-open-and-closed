"""Azure Speech standard recognizer, locale wuu-CN (Chinese, Wu, Simplified), via fast transcription.

The only closed API found with a Wu locale (2026-10-13): Gemini 3.5 Transcribe lists only cmn-Hans-CN / yue-Hant-HK,
and MAI-Transcribe-2 names no Wu. Microsoft doesn't say which Wu variety the model was trained on.

Calls are spaced MIN_GAP_S apart (pace() runs outside the timer); a 429 waits Retry-After (default 30 s) up to 3
attempts, and if it is still rate-limited the model is stopped for the rest of the run instead of failing every clip.
"""
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import _azure as az

MIN_GAP_S = 3.0
_last_call = 0.0


class RateLimited(RuntimeError):
    halt_model = True  # the runner stops calling this model for the rest of the run


def gap_s() -> float:
    return MIN_GAP_S


def pace() -> None:
    wait = MIN_GAP_S - (time.time() - _last_call)
    if wait > 0:
        time.sleep(wait)


def transcribe(wav_path: str) -> str:
    global _last_call
    endpoint, key = az.credentials()
    body, ctype = az._multipart([
        ("audio", Path(wav_path).name, Path(wav_path).read_bytes(), "audio/wav"),
        ("definition", "", json.dumps({"locales": ["wuu-CN"]}).encode(), "application/json"),
    ])
    url = f"{endpoint}/speechtotext/transcriptions:transcribe?api-version={az.API_VERSION}"
    for attempt in (1, 2, 3):
        _last_call = time.time()
        req = urllib.request.Request(url, data=body, method="POST",
                                     headers={"Content-Type": ctype, "Ocp-Apim-Subscription-Key": key})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                _last_call = time.time()
                return az._text(json.load(r))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            if e.code != 429:
                raise RuntimeError(f"Azure wuu-CN HTTP {e.code}: {detail}") from None
            if attempt == 3:
                raise RateLimited(f"Azure still rate-limiting after 3 tries: {detail}") from None
            retry = e.headers.get("Retry-After")
            delay = int(retry) if retry and retry.isdigit() else 30
            print(f"[azure-wuu] 429, waiting {delay}s", flush=True)
            time.sleep(min(delay, 120))
    raise RuntimeError("unreachable")
