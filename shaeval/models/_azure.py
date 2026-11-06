"""Shared pieces for the Azure Speech fast-transcription REST API.

  POST https://<resource>.cognitiveservices.azure.com/speechtotext/transcriptions:transcribe?api-version=2025-10-15
  header Ocp-Apim-Subscription-Key, multipart form: `audio` (the file) + `definition` (JSON).

Settings come from .env (gitignored), never from the code:
  AZURE_SPEECH_ENDPOINT=https://<resource>.cognitiveservices.azure.com
  AZURE_SPEECH_KEY=<key of that Speech/Foundry resource>
The Cantonese evals' MAI_ENDPOINT / MAI_API_KEY resource (westus2) works too.
"""
import json
import uuid
from ._env import env

API_VERSION = "2025-10-15"


def credentials() -> tuple[str, str]:
    e = env()
    endpoint, key = e.get("AZURE_SPEECH_ENDPOINT", "").rstrip("/"), e.get("AZURE_SPEECH_KEY", "")
    if not endpoint or not key or "<" in endpoint or "<" in key:  # blank, or still the <...> filler
        raise RuntimeError("AZURE_SPEECH_ENDPOINT and AZURE_SPEECH_KEY are missing from .env")
    return endpoint, key


def _multipart(fields: list[tuple[str, str, bytes, str | None]]) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    body = b""
    for name, filename, data, ctype in fields:
        disp = f'form-data; name="{name}"' + (f'; filename="{filename}"' if filename else "")
        body += f"--{boundary}\r\nContent-Disposition: {disp}\r\n".encode()
        if ctype:
            body += f"Content-Type: {ctype}\r\n".encode()
        body += b"\r\n" + data + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


def _text(resp: dict) -> str:
    combined = resp.get("combinedPhrases")
    if combined:
        return "".join(p.get("text", "") for p in combined).strip()
    phrases = resp.get("phrases")
    if phrases:
        return "".join(p.get("text", "") for p in phrases).strip()
    if "combinedPhrases" in resp or "phrases" in resp:
        return ""  # recognised nothing; scored as all deletions
    raise RuntimeError(f"Azure response had neither combinedPhrases nor phrases: {json.dumps(resp)[:300]}")
