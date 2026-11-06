"""Shared plumbing for Gemini models called through generateContent (copied from the Cantonese evals' _gemini_gc.py
and gemini.py).

The key is GEMINI_API (or GEMINI_API_KEY) in .env; GEMINI_USE=free uses GEMINI_API_FREE_KEY instead. Calls are spaced GAP_S apart, at most DAILY_CAP (default 100,
override with <NAME>_DAILY_CAP) per Pacific day; a 429 that names a daily limit stops the model for the rest of the run
(finished clips stay cached), any other 429 waits Google's retry delay once. State files: cache/<NAME>_*.
"""
import base64
import json
import os
import re
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from ._env import ROOT, env

API = "https://generativelanguage.googleapis.com"
GAP_S, MAX_RETRY_WAIT_S, WINDOW_S = 5.0, 120.0, 24 * 3600


class QuotaExhausted(RuntimeError):
    halt_model = True  # the runner stops calling this model for the rest of the run


def gap_s() -> float:
    return GAP_S


def inline_wav(wav_path: str) -> dict:
    return {"inline_data": {"mime_type": "audio/wav", "data": base64.b64encode(Path(wav_path).read_bytes()).decode()}}


def response_text(resp: dict, name: str) -> str:
    """Joined non-thought text of the first candidate; raises if there is none."""
    cands = resp.get("candidates") or []
    if not cands:
        raise RuntimeError(f"{name}: no candidates ({json.dumps(resp.get('promptFeedback'))[:200]})")
    return "".join(p.get("text", "") for p in cands[0].get("content", {}).get("parts", []) if not p.get("thought"))


def make(mod):
    """pace() and transcribe() for a module defining MODEL, NAME, build_request(wav_path) and parse(text)."""
    cap = int(os.environ.get(f"{mod.NAME.upper()}_DAILY_CAP", "100"))
    state = lambda n: ROOT / "cache" / f"{mod.NAME}_{n}"

    def read_float(p: Path) -> float:
        try:
            return float(p.read_text())
        except (OSError, ValueError):
            return 0.0

    def recent() -> list[float]:
        try:
            return [t for t in json.loads(state("calls").read_text()) if time.time() - t < WINDOW_S]
        except (OSError, ValueError):
            return []

    def pace() -> None:
        wait = read_float(state("blocked_until")) - time.time()
        if wait > 0:
            raise QuotaExhausted(f"{mod.MODEL} said to retry in {wait / 3600:.1f} more hours.")
        day_start = datetime.now(ZoneInfo("America/Los_Angeles")).replace(hour=0, minute=0, second=0,
                                                                          microsecond=0).timestamp()
        if sum(t >= day_start for t in recent()) >= cap:
            raise QuotaExhausted(f"{mod.MODEL} cap of {cap} requests today (Pacific) reached "
                                 f"({mod.NAME.upper()}_DAILY_CAP overrides).")
        wait = GAP_S - (time.time() - read_float(state("last_call")))
        if wait > 0:
            time.sleep(wait)

    def transcribe(wav_path: str) -> str:
        e = env()
        key = (e.get("GEMINI_API_FREE_KEY") if os.environ.get("GEMINI_USE") == "free"
               else e.get("GEMINI_API") or e.get("GEMINI_API_KEY"))
        if not key:
            raise RuntimeError("no Gemini key in .env (GEMINI_API, or GEMINI_API_FREE_KEY with GEMINI_USE=free)")
        body = json.dumps(mod.build_request(wav_path)).encode()
        state("last_call").parent.mkdir(exist_ok=True)
        for attempt in (1, 2):
            state("last_call").write_text(str(time.time()))
            req = urllib.request.Request(f"{API}/v1beta/models/{mod.MODEL}:generateContent", data=body, method="POST",
                                         headers={"Content-Type": "application/json", "x-goog-api-key": key})
            try:
                with urllib.request.urlopen(req, timeout=180) as r:
                    data = json.load(r)
                state("calls").write_text(json.dumps(recent() + [time.time()]))
                break
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")
                if e.code != 429:
                    raise RuntimeError(f"{mod.MODEL} HTTP {e.code}: {detail[:300]}") from None
                retry_after = e.headers.get("Retry-After")
                if re.search(r"per ?day|daily|PerDay", detail, re.I):
                    if retry_after and retry_after.isdigit():
                        state("blocked_until").write_text(str(time.time() + int(retry_after)))
                    raise QuotaExhausted(f"{mod.MODEL} daily limit reached: {detail[:240]}") from None
                if attempt == 2:
                    raise RuntimeError(f"{mod.MODEL} rate-limited again; giving up. {detail[:200]}") from None
                m = re.search(r"(\d+(?:\.\d+)?)s", detail)
                delay = min(float(m.group(1)) + 2 if m else 60.0, MAX_RETRY_WAIT_S)
                print(f"[{mod.NAME}] 429, waiting {delay:.0f}s", flush=True)
                time.sleep(delay)
        return mod.parse(response_text(data, mod.MODEL))

    return pace, transcribe
